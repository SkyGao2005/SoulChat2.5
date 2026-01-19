"""
对话数据合成脚本

使用 DeepSeek API 基于 situation 和不同人格生成训练对话数据。
支持断点续传、多线程并发、策略验证与重试机制。

使用方法:
    # 测试模式（只生成第一个 situation 的第一个人格）
    python synthesize_dialogs.py --test --api-key YOUR_KEY
    
    # 正式运行
    python synthesize_dialogs.py --api-key YOUR_KEY --concurrency 5
"""

import json
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
import argparse
import re

from tqdm import tqdm
from openai import AsyncOpenAI

# 添加 src 路径
SCRIPT_DIR = Path(__file__).parent.resolve()
SRC_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from DatasetProcess.predict_rex_classify import StrategyPredictor

# ============== 路径配置 ==============
SITUATIONS_PATH = PROJECT_ROOT / "Datasets" / "SynthesizedSituations" / "situations.json"
PERSONAS_DIR = PROJECT_ROOT / "Propmts" / "Patients"
PROMPT_TEMPLATE_PATH = PROJECT_ROOT / "Propmts" / "SynthesizePropmt.txt"
OUTPUT_DIR = PROJECT_ROOT / "Datasets" / "SynthesizedDialogs"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"
MODEL_DIR = PROJECT_ROOT / "models" / "esconv_strategy"

# ============== 策略验证配置 ==============
# 情感支持类：互换可容忍
EMOTIONAL_SUPPORT = {"肯定与安慰", "反映情感", "自我表露"}
# 信息建议类：互换可容忍
INFO_SUGGESTION = {"提供建议", "提供信息"}
# 结构引导类：复述→提问可容忍（单向）
STRUCTURE_GUIDE = {"提问", "复述与转述"}


@dataclass
class ProgressTracker:
    """进度追踪器，支持断点续传"""
    completed: set = field(default_factory=set)  # (situation_idx, persona_idx) 的集合
    failed: list = field(default_factory=list)   # 失败记录
    
    def is_completed(self, situation_idx: int, persona_idx: int) -> bool:
        return (situation_idx, persona_idx) in self.completed
    
    def mark_completed(self, situation_idx: int, persona_idx: int):
        self.completed.add((situation_idx, persona_idx))
    
    def mark_failed(self, situation_idx: int, persona_idx: int, reason: str):
        self.failed.append({
            "situation_idx": situation_idx,
            "persona_idx": persona_idx,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })
    
    def to_dict(self) -> dict:
        return {
            "completed": [list(x) for x in self.completed],
            "failed": self.failed
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ProgressTracker":
        tracker = cls()
        tracker.completed = {tuple(x) for x in data.get("completed", [])}
        tracker.failed = data.get("failed", [])
        return tracker
    
    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "ProgressTracker":
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return cls.from_dict(json.load(f))
        return cls()


def load_situations(path: Path) -> list[str]:
    """加载 situation 列表"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_personas(personas_dir: Path) -> list[tuple[int, str]]:
    """加载所有人格，返回 (persona_idx, persona_content) 列表"""
    personas = []
    for i in range(1, 7):  # 1-6
        persona_file = personas_dir / f"{i}.txt"
        if persona_file.exists():
            with open(persona_file, 'r', encoding='utf-8') as f:
                personas.append((i, f.read().strip()))
    return personas


def load_prompt_template(path: Path) -> str:
    """加载 prompt 模板"""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def build_prompt(template: str, situation: str, persona: str) -> str:
    """构建完整的 prompt"""
    prompt = template.replace("{Situation}", situation)
    prompt = prompt.replace("{PatientPersona}", persona)
    return prompt


def is_strategy_compatible(generated: str, predicted: str) -> bool:
    """
    检查两个策略是否相容
    
    Args:
        generated: 生成的策略标签
        predicted: 分类器预测的策略标签
    
    Returns:
        是否相容
    """
    if generated == predicted:
        return True
    
    # 情感支持类互换可容忍
    if generated in EMOTIONAL_SUPPORT and predicted in EMOTIONAL_SUPPORT:
        return True
    
    # 信息建议类互换可容忍
    if generated in INFO_SUGGESTION and predicted in INFO_SUGGESTION:
        return True
    
    # 结构引导类：提问与复述双向互换可容忍
    if generated in STRUCTURE_GUIDE and predicted in STRUCTURE_GUIDE:
        return True
    
    return False


def is_strategy_match_top_k(generated: str, top_k_strategies: list[dict]) -> bool:
    """
    检查策略是否与 top_k 预测中的任意一个相容
    
    Args:
        generated: 生成的策略标签
        top_k_strategies: top_k 策略列表，每个元素为 {"strategy": str, "confidence": float}
    
    Returns:
        是否匹配（可容忍）
    """
    for pred_item in top_k_strategies:
        predicted = pred_item.get("strategy", "")
        if is_strategy_compatible(generated, predicted):
            return True
    
    return False


def parse_dialog_text(response_text: str) -> tuple[Optional[dict], Optional[str]]:
    """
    解析 API 返回的文本格式对话
    
    格式：
        situation: <situation>
        <role>: <strategy><text>
        
    其中 role 是 seeker 或 supporter，supporter 必须有 [strategy]
    
    Returns:
        (data, error_msg): 解析成功返回 (dict, None)，失败返回 (None, error_msg)
    """
    lines = response_text.strip().split('\n')
    
    if not lines:
        return None, "响应为空"
    
    result = {
        "situation": "",
        "dialog": []
    }
    
    # 解析 situation（可能在第一行或多行）
    situation_lines = []
    dialog_start_idx = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # 检查是否是 situation 行
        if line.lower().startswith('situation:'):
            situation_content = line[len('situation:'):].strip()
            situation_lines.append(situation_content)
            dialog_start_idx = i + 1
            continue
        
        # 检查是否是对话行（seeker: 或 supporter:）
        if line.lower().startswith('seeker:') or line.lower().startswith('supporter:'):
            dialog_start_idx = i
            break
        
        # 如果还没遇到对话行，可能是 situation 的续行
        if not result["dialog"] and situation_lines:
            situation_lines.append(line)
            dialog_start_idx = i + 1
    
    result["situation"] = ' '.join(situation_lines).strip()
    
    if not result["situation"]:
        return None, "未找到 situation 行"
    
    # 解析对话行
    current_turn = None
    
    for i, line in enumerate(lines[dialog_start_idx:], start=dialog_start_idx + 1):
        line = line.strip()
        if not line:
            continue
        
        # 检查是否是新的对话轮
        seeker_match = re.match(r'^seeker:\s*(.*)$', line, re.IGNORECASE)
        supporter_match = re.match(r'^supporter:\s*(.*)$', line, re.IGNORECASE)
        
        if seeker_match:
            # 保存之前的轮次
            if current_turn:
                result["dialog"].append(current_turn)
            
            text = seeker_match.group(1).strip()
            current_turn = {
                "role": "seeker",
                "text": text
            }
        elif supporter_match:
            # 保存之前的轮次
            if current_turn:
                result["dialog"].append(current_turn)
            
            content = supporter_match.group(1).strip()
            
            # 解析 [strategy] 和 text
            strategy_match = re.match(r'^\[([^\]]+)\]\s*(.*)$', content)
            if strategy_match:
                strategy = strategy_match.group(1).strip()
                text = strategy_match.group(2).strip()
            else:
                # 没有找到 strategy，可能格式有误
                strategy = ""
                text = content
            
            current_turn = {
                "role": "supporter",
                "text": text,
                "strategy": strategy
            }
        else:
            # 不是新的对话轮，可能是上一轮的续行
            if current_turn:
                current_turn["text"] += " " + line
    
    # 保存最后一个轮次
    if current_turn:
        result["dialog"].append(current_turn)
    
    # 验证
    if not result["dialog"]:
        return None, f"未找到有效的对话内容\n原文前200字符: {response_text[:200]}..."
    
    # 检查 supporter 是否都有 strategy
    for i, turn in enumerate(result["dialog"]):
        if turn["role"] == "supporter" and not turn.get("strategy"):
            return None, f"第 {i+1} 轮 supporter 缺少 strategy 标签\n内容: {turn['text'][:50]}..."
    
    return result, None


def build_history_text(dialog: list[dict], up_to_turn: int) -> str:
    """构建对话历史文本用于分类器"""
    history_parts = []
    for i, turn in enumerate(dialog[:up_to_turn]):
        role = "求助者" if turn["role"] == "seeker" else "支持者"
        history_parts.append(f"{role}：{turn['text']}")
    return "；".join(history_parts)


class DialogSynthesizer:
    """对话合成器"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        predictor: Optional[StrategyPredictor] = None
    ):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.predictor = predictor
    
    async def generate_dialog(
        self,
        prompt: str,
        situation: str,
        max_retries: int = 2
    ) -> dict:
        """
        生成对话并验证
        
        Args:
            prompt: 完整的生成 prompt
            situation: situation 文本（用于分类器）
            max_retries: 最大重试次数
        
        Returns:
            生成的对话数据
        
        Raises:
            ValueError: 验证失败超过最大重试次数
        """
        messages = [{"role": "user", "content": prompt}]
        
        # 初次生成，如果 JSON 解析失败则重试一次
        dialog_data = None
        last_error = None
        for json_retry in range(2):  # 最多尝试 2 次
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=1.3#,
                #response_format={
                #    'type': 'json_object'
                #}
            )
            
            # 检查 finish_reason，如果是截断则直接抛出错误
            finish_reason = response.choices[0].finish_reason
            if finish_reason == "length":
                raise ValueError(f"响应被截断 (finish_reason=length)，请增加 max_tokens 或简化 prompt")
            
            response_content = response.choices[0].message.content
            
            # 检查响应是否为空
            if not response_content or not response_content.strip():
                last_error = "响应内容为空"
                if json_retry == 0:
                    print(f"    响应内容为空，重试中...")
                continue
            
            response_text = response_content.strip()
            dialog_data, parse_error = parse_dialog_text(response_text)
            
            if dialog_data and dialog_data.get("dialog"):
                break  # 解析成功，跳出循环
            
            if not dialog_data:
                last_error = f"无法解析 JSON 响应:\n{parse_error}\n原文前200字符: {response_text[:200]}..."
            elif not dialog_data.get("dialog"):
                last_error = f"响应中缺少 dialog 字段: {response_text[:200]}..."
            
            if json_retry == 0:
                print(f"    JSON 解析失败: {parse_error}")
        
        # 如果两次都失败，抛出错误
        if not dialog_data or not dialog_data.get("dialog"):
            raise ValueError(last_error)
        
        # 如果没有分类器，直接返回
        if self.predictor is None:
            return dialog_data
        
        # 验证每个 supporter 轮的策略
        dialog = dialog_data["dialog"]
        validated_dialog = []
        
        for i, turn in enumerate(dialog):
            validated_dialog.append(turn)
            
            if turn["role"] != "supporter":
                continue
            
            generated_strategy = turn.get("strategy", "")
            
            # 构建对话历史
            history = build_history_text(validated_dialog, len(validated_dialog) - 1)
            
            # 预测策略（获取 top3）
            pred_result = self.predictor.predict(
                current_utterance=turn["text"],
                situation=situation,
                dialog_history=history,
                top_k=3
            )
            top_k_strategies = pred_result.get("top_k_strategies", [])
            
            # 检查是否与 top3 中任意一个相容
            retry_count = 0
            while not is_strategy_match_top_k(generated_strategy, top_k_strategies):
                if retry_count >= max_retries:
                    top_k_str = ", ".join([f"{s['strategy']}({s['confidence']:.2f})" for s in top_k_strategies])
                    raise ValueError(
                        f"策略验证失败 (超过最大重试次数): "
                        f"生成={generated_strategy}, top3预测=[{top_k_str}], "
                        f"utterance={turn['text'][:50]}..."
                    )
                
                retry_count += 1
                top_k_str = ", ".join([f"{s['strategy']}({s['confidence']:.2f})" for s in top_k_strategies])
                print(f"    策略不匹配 (重试 {retry_count}/{max_retries}): "
                      f"生成={generated_strategy}, top3预测=[{top_k_str}]")
                
                # 移除当前轮及之后的内容，重新生成
                validated_dialog = validated_dialog[:-1]
                
                # 构建续写 prompt
                continuation_prompt = self._build_continuation_prompt(
                    prompt, validated_dialog
                )
                
                # 重新生成
                continuation_response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": continuation_prompt}],
                    temperature=1.3#,
                    #response_format={
                    #    'type': 'json_object'
                    #}
                )
                
                # 检查 finish_reason
                finish_reason = continuation_response.choices[0].finish_reason
                if finish_reason == "length":
                    raise ValueError(f"续写响应被截断 (finish_reason=length)")
                
                continuation_content = continuation_response.choices[0].message.content
                
                # 检查响应是否为空
                if not continuation_content or not continuation_content.strip():
                    raise ValueError("续写响应内容为空")
                
                continuation_text = continuation_content.strip()
                continuation_data, parse_error = parse_dialog_text(continuation_text)
                
                if not continuation_data:
                    raise ValueError(f"续写响应解析失败:\n{parse_error}\n原文前200字符: {continuation_text[:200]}...")
                if not continuation_data.get("dialog"):
                    raise ValueError(f"续写响应缺少 dialog 字段: {continuation_text[:200]}...")
                
                # 合并对话
                new_dialog = continuation_data["dialog"]
                
                # 找到应该继续的位置
                start_idx = len(validated_dialog)
                if start_idx < len(new_dialog):
                    turn = new_dialog[start_idx]
                    validated_dialog.append(turn)
                    
                    if turn["role"] == "supporter":
                        generated_strategy = turn.get("strategy", "")
                        history = build_history_text(validated_dialog, len(validated_dialog) - 1)
                        pred_result = self.predictor.predict(
                            current_utterance=turn["text"],
                            situation=situation,
                            dialog_history=history,
                            top_k=3
                        )
                        top_k_strategies = pred_result.get("top_k_strategies", [])
                    else:
                        # 如果是 seeker，继续处理
                        break
                else:
                    break
        
        dialog_data["dialog"] = validated_dialog
        return dialog_data
    
    def _build_continuation_prompt(self, original_prompt: str, partial_dialog: list[dict]) -> str:
        """构建续写 prompt"""
        dialog_text = json.dumps(partial_dialog, ensure_ascii=False, indent=2)
        
        continuation_prompt = f"""{original_prompt}

【已生成的对话部分】
以下是已经生成并验证通过的对话部分，请从这里继续生成剩余对话：

```json
{dialog_text}
```

请继续完成对话，输出完整的 JSON（包含 situation 和完整的 dialog 数组，将上面已生成的部分也包含在内）。
"""
        return continuation_prompt


async def process_single_task(
    synthesizer: DialogSynthesizer,
    prompt_template: str,
    situation: str,
    situation_idx: int,
    persona: str,
    persona_idx: int,
    progress: ProgressTracker,
    output_dir: Path,
    semaphore: asyncio.Semaphore
) -> bool:
    """处理单个任务（一个 situation + 一个 persona）"""
    async with semaphore:
        # 检查是否已完成
        if progress.is_completed(situation_idx, persona_idx):
            return True
        
        try:
            # 构建 prompt
            prompt = build_prompt(prompt_template, situation, persona)
            
            # 生成对话
            dialog_data = await synthesizer.generate_dialog(prompt, situation)
            
            # 保存结果
            output_file = output_dir / f"dialog_{situation_idx}_{persona_idx}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "situation_idx": situation_idx,
                    "persona_idx": persona_idx,
                    "situation": situation,
                    "data": dialog_data
                }, f, ensure_ascii=False, indent=2)
            
            # 标记完成
            progress.mark_completed(situation_idx, persona_idx)
            return True
            
        except Exception as e:
            progress.mark_failed(situation_idx, persona_idx, str(e))
            print(f"  任务失败 [{situation_idx},{persona_idx}]: {e}")
            raise  # 重新抛出异常以停止程序


async def main():
    parser = argparse.ArgumentParser(description="对话数据合成")
    parser.add_argument("--api-key", type=str, help="DeepSeek API Key")
    parser.add_argument("--base-url", type=str, default="https://api.deepseek.com")
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--concurrency", type=int, default=5, help="并发数")
    parser.add_argument("--test", action="store_true", help="测试模式：只生成第一个 situation 的第一个人格")
    parser.add_argument("--no-validate", action="store_true", help="禁用策略验证")
    parser.add_argument("--device", type=str, default=None, help="分类器设备 (cuda/cpu)")
    
    args = parser.parse_args()
    
    # 获取 API Key
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误: 请提供 DeepSeek API Key (--api-key 或 DEEPSEEK_API_KEY 环境变量)")
        return
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    situations = load_situations(SITUATIONS_PATH)
    personas = load_personas(PERSONAS_DIR)
    prompt_template = load_prompt_template(PROMPT_TEMPLATE_PATH)
    
    print(f"  Situations: {len(situations)}")
    print(f"  Personas: {len(personas)}")
    
    # 测试模式
    if args.test:
        situations = situations[:1]
        personas = personas[:1]
        print("\n[测试模式] 只处理第一个 situation 的第一个人格")
    
    # 加载进度
    progress = ProgressTracker.load(PROGRESS_FILE)
    completed_count = len(progress.completed)
    total_tasks = len(situations) * len(personas)
    print(f"  已完成: {completed_count}/{total_tasks}")
    
    # 初始化分类器（如果需要验证）
    predictor = None
    if not args.no_validate:
        print("\n加载策略分类器...")
        predictor = StrategyPredictor(str(MODEL_DIR), args.device)
    
    # 初始化合成器
    synthesizer = DialogSynthesizer(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        predictor=predictor
    )
    
    # 并发控制
    semaphore = asyncio.Semaphore(args.concurrency)
    
    # 构建任务列表
    pending_tasks = []
    for sit_idx, situation in enumerate(situations):
        for persona_idx, persona in personas:
            if not progress.is_completed(sit_idx, persona_idx):
                pending_tasks.append((sit_idx, situation, persona_idx, persona))
    
    print(f"\n开始处理 {len(pending_tasks)} 个待处理任务...")
    print(f"并发数: {args.concurrency}")
    print("-" * 50)
    
    # 批量并发处理
    batch_size = args.concurrency * 2  # 每批处理的任务数
    pbar = tqdm(total=len(pending_tasks), desc="生成对话")
    
    for batch_start in range(0, len(pending_tasks), batch_size):
        batch = pending_tasks[batch_start:batch_start + batch_size]
        
        # 创建并发任务
        async_tasks = [
            process_single_task(
                synthesizer=synthesizer,
                prompt_template=prompt_template,
                situation=situation,
                situation_idx=sit_idx,
                persona=persona,
                persona_idx=persona_idx,
                progress=progress,
                output_dir=OUTPUT_DIR,
                semaphore=semaphore
            )
            for sit_idx, situation, persona_idx, persona in batch
        ]
        
        try:
            # 并发执行当前批次
            results = await asyncio.gather(*async_tasks, return_exceptions=True)
            
            # 检查是否有异常
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    sit_idx, situation, persona_idx, persona = batch[i]
                    progress.save(PROGRESS_FILE)
                    pbar.close()
                    print(f"\n处理中断 [{sit_idx},{persona_idx}]: {result}")
                    print(f"进度已保存到: {PROGRESS_FILE}")
                    raise result
            
            pbar.update(len(batch))
            
            # 保存进度
            progress.save(PROGRESS_FILE)
            
        except Exception as e:
            pbar.close()
            progress.save(PROGRESS_FILE)
            print(f"\n处理中断: {e}")
            print(f"进度已保存到: {PROGRESS_FILE}")
            raise
    
    pbar.close()
    
    # 保存最终进度
    progress.save(PROGRESS_FILE)
    
    # 汇总所有对话到一个文件
    print("\n合并所有对话...")
    all_dialogs = []
    for dialog_file in sorted(OUTPUT_DIR.glob("dialog_*.json")):
        with open(dialog_file, 'r', encoding='utf-8') as f:
            all_dialogs.append(json.load(f))
    
    merged_path = OUTPUT_DIR / "all_dialogs.json"
    with open(merged_path, 'w', encoding='utf-8') as f:
        json.dump(all_dialogs, f, ensure_ascii=False, indent=2)
    
    print(f"\n完成!")
    print(f"  总对话数: {len(all_dialogs)}")
    print(f"  合并文件: {merged_path}")
    print(f"  失败数: {len(progress.failed)}")


if __name__ == "__main__":
    asyncio.run(main())

