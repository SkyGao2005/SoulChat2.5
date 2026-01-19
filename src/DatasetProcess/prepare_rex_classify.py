#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 ESConv_zh.json 数据集转换为 RexUniNLU 模型的 [CLASSIFY] 任务格式

输入格式：[CLASSIFY] + situation + 历史对话 + 当前supporter的话
输出格式：strategy 分类标签
"""

import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter


# 策略英文到中文映射
STRATEGY_MAP = {
    "Question": "提问",
    "Affirmation and Reassurance": "肯定与安慰",
    "Restatement or Paraphrasing": "复述与转述",
    "Self-disclosure": "自我表露",
    "Providing Suggestions": "提供建议",
    "Information": "提供信息",
    "Reflection of feelings": "反映情感",
    "Others": "其他"
}

# 获取所有中文策略标签
STRATEGY_LABELS = list(STRATEGY_MAP.values())


def format_dialog_history(dialog: List[Dict], end_idx: int, max_turns: int = 10) -> str:
    """
    格式化对话历史
    
    Args:
        dialog: 对话列表
        end_idx: 当前轮次索引（不包含当前轮）
        max_turns: 最大保留轮次数
    
    Returns:
        格式化的对话历史字符串
    """
    # 取最近的 max_turns 轮对话
    start_idx = max(0, end_idx - max_turns)
    history_turns = dialog[start_idx:end_idx]
    
    history_parts = []
    for turn in history_turns:
        speaker = turn["speaker"]
        content = turn["content"]
        if speaker == "seeker":
            history_parts.append(f"求助者：{content}")
        else:
            history_parts.append(f"支持者：{content}")
    
    return "；".join(history_parts)


def create_classify_sample(
    dialog_id: int,
    turn_idx: int,
    situation: str,
    dialog_history: str,
    current_utterance: str,
    strategy_cn: str
) -> Dict[str, Any]:
    """
    创建一个分类样本
    
    Args:
        dialog_id: 对话ID
        turn_idx: 轮次索引
        situation: 情境描述
        dialog_history: 对话历史
        current_utterance: 当前支持者发言
        strategy_cn: 策略的中文标签
    
    Returns:
        RexUniNLU 格式的样本
    """
    # 构建输入文本
    text_parts = []
    if situation:
        text_parts.append(f"情境：{situation}")
    if dialog_history:
        text_parts.append(f"对话历史：{dialog_history}")
    text_parts.append(f"当前发言：{current_utterance}")
    
    input_text = "。".join(text_parts)
    full_text = f"[CLASSIFY]{input_text}"
    
    # 构建 info_list
    info_list = [[{
        "type": strategy_cn,
        "span": "[CLASSIFY]",
        "offset": [0, 10]
    }]]
    
    # 构建 schema（所有可能的策略标签）
    schema = {label: None for label in STRATEGY_LABELS}
    
    sample = {
        "id": f"esconv-{dialog_id}-{turn_idx}",
        "text": full_text,
        "info_list": info_list,
        "schema": schema
    }
    
    return sample


def process_dialog(dialog_data: Dict, dialog_id: int, max_history_turns: int = 10) -> List[Dict]:
    """
    处理单个对话，提取所有支持者轮次的分类样本
    
    Args:
        dialog_data: 对话数据
        dialog_id: 对话ID
        max_history_turns: 最大历史轮次数
    
    Returns:
        分类样本列表
    """
    samples = []
    situation = dialog_data.get("situation", "")
    dialog = dialog_data.get("dialog", [])
    
    for turn_idx, turn in enumerate(dialog):
        # 只处理支持者的轮次
        if turn["speaker"] != "supporter":
            continue
        
        # 获取策略标签
        annotation = turn.get("annotation", {})
        strategy_en = annotation.get("strategy")
        
        if not strategy_en:
            continue
        
        # 映射到中文
        strategy_cn = STRATEGY_MAP.get(strategy_en)
        if not strategy_cn:
            print(f"Warning: Unknown strategy '{strategy_en}' in dialog {dialog_id}")
            continue
        
        # 获取对话历史
        dialog_history = format_dialog_history(dialog, turn_idx, max_history_turns)
        
        # 当前发言
        current_utterance = turn["content"]
        
        # 创建样本
        sample = create_classify_sample(
            dialog_id=dialog_id,
            turn_idx=turn_idx,
            situation=situation,
            dialog_history=dialog_history,
            current_utterance=current_utterance,
            strategy_cn=strategy_cn
        )
        samples.append(sample)
    
    return samples


def process_dataset(
    input_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    max_history_turns: int = 10,
    seed: int = 42
) -> None:
    """
    处理整个数据集
    
    Args:
        input_path: 输入文件路径
        output_dir: 输出目录
        train_ratio: 训练集比例
        dev_ratio: 验证集比例
        max_history_turns: 最大历史轮次数
        seed: 随机种子
    """
    random.seed(seed)
    
    # 读取数据
    print(f"Loading data from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} dialogs")
    
    # 处理所有对话
    all_samples = []
    strategy_counter = Counter()
    
    for dialog_data in data:
        dialog_id = dialog_data.get("dialog_id", len(all_samples))
        samples = process_dialog(dialog_data, dialog_id, max_history_turns)
        all_samples.extend(samples)
        
        # 统计策略分布
        for sample in samples:
            strategy = sample["info_list"][0][0]["type"]
            strategy_counter[strategy] += 1
    
    print(f"\nTotal samples: {len(all_samples)}")
    print("\nStrategy distribution:")
    for strategy, count in sorted(strategy_counter.items(), key=lambda x: -x[1]):
        print(f"  {strategy}: {count} ({count/len(all_samples)*100:.1f}%)")
    
    # 按对话ID分组，确保同一对话的样本不会分散到不同集合
    dialog_ids = list(set(s["id"].rsplit("-", 1)[0] for s in all_samples))
    random.shuffle(dialog_ids)
    
    # 计算分割点
    n_dialogs = len(dialog_ids)
    train_end = int(n_dialogs * train_ratio)
    dev_end = int(n_dialogs * (train_ratio + dev_ratio))
    
    train_dialog_ids = set(dialog_ids[:train_end])
    dev_dialog_ids = set(dialog_ids[train_end:dev_end])
    test_dialog_ids = set(dialog_ids[dev_end:])
    
    # 分割样本
    train_samples = [s for s in all_samples if s["id"].rsplit("-", 1)[0] in train_dialog_ids]
    dev_samples = [s for s in all_samples if s["id"].rsplit("-", 1)[0] in dev_dialog_ids]
    test_samples = [s for s in all_samples if s["id"].rsplit("-", 1)[0] in test_dialog_ids]
    
    # 打乱顺序
    random.shuffle(train_samples)
    random.shuffle(dev_samples)
    random.shuffle(test_samples)
    
    print(f"\nSplit: train={len(train_samples)}, dev={len(dev_samples)}, test={len(test_samples)}")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存数据（每行一个JSON对象，符合 RexUniNLU 的格式要求）
    def save_jsonl(samples: List[Dict], filename: str):
        filepath = output_path / filename
        with open(filepath, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"Saved {len(samples)} samples to {filepath}")
    
    save_jsonl(train_samples, "train.json")
    save_jsonl(dev_samples, "dev.json")
    save_jsonl(test_samples, "test.json")
    
    # 保存策略标签映射
    label_map_path = output_path / "label_map.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(STRATEGY_MAP, f, ensure_ascii=False, indent=2)
    print(f"Saved label map to {label_map_path}")
    
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description="Convert ESConv_zh to RexUniNLU CLASSIFY format")
    parser.add_argument(
        "--input",
        type=str,
        default="../../Datasets/ESConv_zh.json",
        help="Input ESConv_zh.json file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../../models/rex/data/esconv_strategy",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train set ratio"
    )
    parser.add_argument(
        "--dev_ratio",
        type=float,
        default=0.1,
        help="Dev set ratio"
    )
    parser.add_argument(
        "--max_history_turns",
        type=int,
        default=10,
        help="Maximum number of history turns to include"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # 处理相对路径
    script_dir = Path(__file__).parent
    input_path = (script_dir / args.input).resolve()
    output_dir = (script_dir / args.output).resolve()
    
    process_dataset(
        input_path=str(input_path),
        output_dir=str(output_dir),
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        max_history_turns=args.max_history_turns,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

