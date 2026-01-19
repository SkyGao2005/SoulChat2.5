#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RexUniNLU 策略分类推理脚本

使用方法:
    python predict_rex_classify.py --utterance "这确实很难，但我觉得你能够成功的。"
    
    python predict_rex_classify.py \
        --situation "我最近工作压力很大" \
        --history "支持者：你好；求助者：最近压力很大" \
        --utterance "听起来你最近确实很辛苦"
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch

# 添加 src 路径到 sys.path（rex 模块在 src/rex）
SCRIPT_DIR = Path(__file__).parent.resolve()
SRC_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from transformers import AutoTokenizer, AutoConfig
from rex.data_utils import data_loader, token_config
from rex.arguments import DataArguments, UIEArguments, ModelArguments
from rex.model.model import RexModel
from rex.Trainer.trainer import RexModelTrainer
from rex.Trainer.utils import compute_metrics


# 策略标签（中文）
STRATEGY_LABELS = [
    "提问", "肯定与安慰", "复述与转述", "自我表露",
    "提供建议", "提供信息", "反映情感", "其他"
]

# 策略中英映射
STRATEGY_MAP_CN_TO_EN = {
    "提问": "Question",
    "肯定与安慰": "Affirmation and Reassurance",
    "复述与转述": "Restatement or Paraphrasing",
    "自我表露": "Self-disclosure",
    "提供建议": "Providing Suggestions",
    "提供信息": "Information",
    "反映情感": "Reflection of feelings",
    "其他": "Others"
}


class StrategyPredictor:
    """策略分类预测器"""
    
    def __init__(self, model_dir: str, device: str = None):
        """
        初始化预测器
        
        Args:
            model_dir: 训练好的模型目录
            device: 设备 ("cuda" 或 "cpu")，默认自动检测
        """
        self.model_dir = model_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载 tokenizer
        print(f"加载 tokenizer: {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": [
                token_config.PREFIX_TOKEN, 
                token_config.TYPE_TOKEN, 
                token_config.CLASSIFY_TOKEN, 
                token_config.MULTI_CLASSIFY_TOKEN
            ]
        })
        
        # 加载配置
        print(f"加载配置: {model_dir}")
        config = AutoConfig.from_pretrained(model_dir)
        
        # 初始化参数
        data_args = DataArguments()
        training_args = UIEArguments(
            output_dir=model_dir,
            bert_model_dir=model_dir,
            no_cuda=(self.device == "cpu")
        )
        model_args = ModelArguments()
        
        # 加载模型
        print("初始化模型...")
        self.model = RexModel(config, training_args, model_args)
        
        # 加载权重（支持 safetensors 和 pytorch_model.bin）
        self._load_weights(model_dir)
        
        self.model.to(self.device)
        self.model.eval()
        
        # 初始化数据加载器和训练器
        self.data_loader = data_loader.UIEDataLoader(
            data_args, self.tokenizer, "", -1, 1, training_args.no_cuda
        )
        
        self.trainer = RexModelTrainer(
            self.model, training_args, 
            self.data_loader.get_collate_fn(),
            processing_class=self.tokenizer,
            compute_metrics=compute_metrics
        )
        self.trainer.rex_dl = self.data_loader
        self.trainer.data_args = data_args
        
        print("模型加载完成！\n")
    
    def _load_weights(self, model_dir: str):
        """加载模型权重"""
        safetensors_path = os.path.join(model_dir, "model.safetensors")
        bin_path = os.path.join(model_dir, "pytorch_model.bin")
        
        if os.path.exists(safetensors_path):
            print(f"加载权重 (safetensors): {safetensors_path}")
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path, device=self.device)
            self.model.load_state_dict(state_dict, strict=False)
        elif os.path.exists(bin_path):
            print(f"加载权重 (pytorch): {bin_path}")
            state_dict = torch.load(bin_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError(f"未找到模型权重文件: {model_dir}")
    
    def predict(
        self, 
        current_utterance: str,
        situation: str = "",
        dialog_history: str = "",
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        预测策略，返回 top_k 个最可能的策略
        
        Args:
            current_utterance: 当前 supporter 发言（必需）
            situation: 情境描述（可选）
            dialog_history: 对话历史，格式如 "支持者：...；求助者：..."（可选）
            top_k: 返回前 k 个最可能的策略（默认 3）
        
        Returns:
            预测结果字典，包含 top_k 策略列表
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
        
        # 构建 schema（所有候选标签）
        schema = {label: None for label in STRATEGY_LABELS}
        
        input_dict = {
            "text": full_text,
            "schema": schema,
            "info_list": []
        }
        
        # 预测
        with torch.no_grad():
            pred_info_list = self.trainer.prediction_step(
                self.model, input_dict, 
                prediction_loss_only=False, 
                do_pred=True,
                top_k=top_k
            )
        
        # 解析 top_k 结果
        top_k_strategies = self._parse_top_k_prediction(pred_info_list)
        
        # 兼容旧接口：第一个为主预测
        predicted_strategy_cn = top_k_strategies[0]["strategy"] if top_k_strategies else "未知"
        predicted_strategy_en = STRATEGY_MAP_CN_TO_EN.get(predicted_strategy_cn, "Unknown")
        
        return {
            "input_text": full_text,
            "predicted_strategy_cn": predicted_strategy_cn,
            "predicted_strategy_en": predicted_strategy_en,
            "top_k_strategies": top_k_strategies,  # 新增：top_k 策略列表
            "raw_output": pred_info_list
        }
    
    def _parse_top_k_prediction(self, pred_info_list) -> List[Dict[str, Any]]:
        """解析 top_k 预测结果"""
        results = []
        if not pred_info_list:
            return [{"strategy": "未知", "confidence": 0.0}]
        
        for items in pred_info_list:
            if isinstance(items, list) and len(items) > 0:
                strategy = items[0].get("type", "未知")
                confidence = items[0].get("avg_confidence", 0.0)
                results.append({
                    "strategy": strategy,
                    "confidence": confidence
                })
            elif isinstance(items, dict):
                strategy = items.get("type", "未知")
                confidence = items.get("avg_confidence", 0.0)
                results.append({
                    "strategy": strategy,
                    "confidence": confidence
                })
        
        return results if results else [{"strategy": "未知", "confidence": 0.0}]
    
    def _parse_prediction(self, pred_info_list) -> str:
        """解析预测结果（兼容旧接口）"""
        if not pred_info_list:
            return "未知"
        
        if isinstance(pred_info_list[0], list) and len(pred_info_list[0]) > 0:
            return pred_info_list[0][0].get("type", "未知")
        elif isinstance(pred_info_list[0], dict):
            return pred_info_list[0].get("type", "未知")
        
        return "未知"
    
    def predict_batch(self, samples: List[Dict]) -> List[Dict]:
        """
        批量预测
        
        Args:
            samples: 样本列表，每个样本包含 utterance, situation(可选), history(可选)
        
        Returns:
            预测结果列表
        """
        results = []
        for sample in samples:
            result = self.predict(
                current_utterance=sample.get("utterance", ""),
                situation=sample.get("situation", ""),
                dialog_history=sample.get("history", "")
            )
            results.append(result)
        return results


def main():
    parser = argparse.ArgumentParser(description="RexUniNLU 策略分类推理")
    parser.add_argument(
        "--model_dir", type=str, 
        default="../../models/esconv_strategy",
        help="训练好的模型目录"
    )
    parser.add_argument(
        "--utterance", type=str, required=True,
        help="当前 supporter 发言（必需）"
    )
    parser.add_argument(
        "--situation", type=str, default="",
        help="情境描述（可选）"
    )
    parser.add_argument(
        "--history", type=str, default="",
        help="对话历史（可选），格式：支持者：...；求助者：..."
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="设备 (cuda/cpu)，默认自动检测"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="交互模式，持续接收输入"
    )
    
    args = parser.parse_args()
    
    # 处理路径
    model_dir = (SCRIPT_DIR / args.model_dir).resolve()
    
    # 初始化预测器
    predictor = StrategyPredictor(str(model_dir), args.device)
    
    if args.interactive:
        # 交互模式
        print("=" * 50)
        print("交互模式 - 输入 'quit' 退出")
        print("=" * 50)
        
        while True:
            print("\n请输入当前发言（或 'quit' 退出）：")
            utterance = input("> ").strip()
            
            if utterance.lower() == 'quit':
                print("再见！")
                break
            
            if not utterance:
                print("发言不能为空！")
                continue
            
            result = predictor.predict(utterance, top_k=3)
            print(f"\nTop3 预测策略:")
            for i, item in enumerate(result.get('top_k_strategies', []), 1):
                strategy_cn = item['strategy']
                strategy_en = STRATEGY_MAP_CN_TO_EN.get(strategy_cn, "Unknown")
                confidence = item.get('confidence', 0.0)
                print(f"  {i}. {strategy_cn} ({strategy_en}) - 置信度: {confidence:.4f}")
    else:
        # 单次预测
        result = predictor.predict(
            current_utterance=args.utterance,
            situation=args.situation,
            dialog_history=args.history,
            top_k=3
        )
        
        print("=" * 50)
        print("预测结果")
        print("=" * 50)
        print(f"输入文本: {result['input_text'][:100]}...")
        print(f"\nTop3 预测策略:")
        for i, item in enumerate(result.get('top_k_strategies', []), 1):
            strategy_cn = item['strategy']
            strategy_en = STRATEGY_MAP_CN_TO_EN.get(strategy_cn, "Unknown")
            confidence = item.get('confidence', 0.0)
            print(f"  {i}. {strategy_cn} ({strategy_en}) - 置信度: {confidence:.4f}")
        print("=" * 50)


if __name__ == "__main__":
    main()

