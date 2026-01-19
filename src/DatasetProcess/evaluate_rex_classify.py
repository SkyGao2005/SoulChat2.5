#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RexUniNLU 策略分类任务评估报告生成器

生成详细的评估报告，包括：
1. 整体指标（Precision, Recall, F1）
2. 每个策略类别的详细指标
3. 混淆矩阵
4. 错误分析
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
import numpy as np

try:
    from sklearn.metrics import classification_report, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# 策略标签列表（与训练数据一致）
STRATEGY_LABELS = [
    "提问", "肯定与安慰", "复述与转述", "自我表露",
    "提供建议", "提供信息", "反映情感", "其他"
]


def load_jsonl(filepath: str) -> List[Any]:
    """加载 JSONL 文件"""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_label(info_list: List) -> str:
    """从 info_list 中提取标签"""
    if not info_list:
        return "未知"
    # info_list 结构: [[{"type": "标签", "span": "...", "offset": [...]}]]
    if isinstance(info_list[0], list) and len(info_list[0]) > 0:
        return info_list[0][0].get("type", "未知")
    elif isinstance(info_list[0], dict):
        return info_list[0].get("type", "未知")
    return "未知"


def compute_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict:
    """计算详细的评估指标"""
    # 整体统计
    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / total if total > 0 else 0
    
    # 每个类别的统计
    class_metrics = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = sum(1 for t in y_true if t == label)
        
        class_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }
    
    # 宏平均
    macro_precision = np.mean([m["precision"] for m in class_metrics.values()])
    macro_recall = np.mean([m["recall"] for m in class_metrics.values()])
    macro_f1 = np.mean([m["f1"] for m in class_metrics.values()])
    
    # 加权平均
    total_support = sum(m["support"] for m in class_metrics.values())
    weighted_precision = sum(m["precision"] * m["support"] for m in class_metrics.values()) / total_support if total_support > 0 else 0
    weighted_recall = sum(m["recall"] * m["support"] for m in class_metrics.values()) / total_support if total_support > 0 else 0
    weighted_f1 = sum(m["f1"] * m["support"] for m in class_metrics.values()) / total_support if total_support > 0 else 0
    
    return {
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "class_metrics": class_metrics
    }


def build_confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str]) -> np.ndarray:
    """构建混淆矩阵"""
    label_to_idx = {label: i for i, label in enumerate(labels)}
    n_labels = len(labels)
    matrix = np.zeros((n_labels, n_labels), dtype=int)
    
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            matrix[label_to_idx[t], label_to_idx[p]] += 1
    
    return matrix


def analyze_errors(
    y_true: List[str], 
    y_pred: List[str], 
    texts: List[str],
    max_errors_per_class: int = 5
) -> Dict[str, List[Dict]]:
    """分析预测错误"""
    errors_by_class = defaultdict(list)
    
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        if t != p:
            text = texts[i] if i < len(texts) else ""
            # 移除 [CLASSIFY] 前缀并截取前200字符
            text_display = text.replace("[CLASSIFY]", "").strip()[:200]
            errors_by_class[t].append({
                "true_label": t,
                "pred_label": p,
                "text": text_display + "..." if len(text.replace("[CLASSIFY]", "").strip()) > 200 else text_display
            })
    
    # 每个类别只保留前 max_errors_per_class 个错误
    for label in errors_by_class:
        errors_by_class[label] = errors_by_class[label][:max_errors_per_class]
    
    return dict(errors_by_class)


def format_confusion_matrix(matrix: np.ndarray, labels: List[str]) -> str:
    """格式化混淆矩阵为字符串"""
    # 计算列宽
    max_label_len = max(len(label) for label in labels)
    col_width = max(max_label_len, 6) + 2
    
    lines = []
    
    # 表头
    header = " " * (col_width + 2) + "预测".center(col_width * len(labels))
    lines.append(header)
    
    header_labels = " " * (col_width + 2) + "".join(label[:6].center(col_width) for label in labels)
    lines.append(header_labels)
    lines.append("-" * (col_width * (len(labels) + 1) + 2))
    
    # 数据行
    for i, label in enumerate(labels):
        row = label[:6].ljust(col_width) + "| "
        for j in range(len(labels)):
            row += str(matrix[i, j]).center(col_width)
        lines.append(row)
    
    return "\n".join(lines)


def generate_report(
    pred_file: str,
    gold_file: str,
    output_file: str,
    test_file: str = None
) -> None:
    """生成评估报告"""
    
    print(f"Loading predictions from {pred_file}...")
    predictions = load_jsonl(pred_file)
    
    print(f"Loading gold labels from {gold_file}...")
    gold_labels = load_jsonl(gold_file)
    
    # 加载测试数据获取原始文本（如果提供）
    texts = []
    if test_file and Path(test_file).exists():
        print(f"Loading test data from {test_file}...")
        test_data = load_jsonl(test_file)
        texts = [item.get("text", "") for item in test_data]
    
    # 提取标签
    y_pred = [extract_label(pred) for pred in predictions]
    y_true = [extract_label(gold) for gold in gold_labels]
    
    # 获取实际出现的标签
    all_labels = list(set(y_true + y_pred))
    # 按预定义顺序排序
    labels = [l for l in STRATEGY_LABELS if l in all_labels]
    # 添加未知标签
    for l in all_labels:
        if l not in labels:
            labels.append(l)
    
    print(f"\nTotal samples: {len(y_true)}")
    print(f"Labels: {labels}")
    
    # 计算指标
    metrics = compute_metrics(y_true, y_pred, labels)
    
    # 构建混淆矩阵
    cm = build_confusion_matrix(y_true, y_pred, labels)
    
    # 错误分析
    errors = analyze_errors(y_true, y_pred, texts) if texts else {}
    
    # 生成报告
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ESConv 策略分类任务评估报告")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 1. 整体指标
    report_lines.append("【1. 整体指标】")
    report_lines.append("-" * 40)
    report_lines.append(f"总样本数: {metrics['total']}")
    report_lines.append(f"正确预测: {metrics['correct']}")
    report_lines.append(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
    report_lines.append("")
    report_lines.append(f"宏平均 Precision: {metrics['macro_precision']:.4f}")
    report_lines.append(f"宏平均 Recall: {metrics['macro_recall']:.4f}")
    report_lines.append(f"宏平均 F1: {metrics['macro_f1']:.4f}")
    report_lines.append("")
    report_lines.append(f"加权平均 Precision: {metrics['weighted_precision']:.4f}")
    report_lines.append(f"加权平均 Recall: {metrics['weighted_recall']:.4f}")
    report_lines.append(f"加权平均 F1: {metrics['weighted_f1']:.4f}")
    report_lines.append("")
    
    # 2. 每个类别的详细指标
    report_lines.append("【2. 各类别详细指标】")
    report_lines.append("-" * 40)
    report_lines.append(f"{'类别':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    report_lines.append("-" * 52)
    
    for label in labels:
        m = metrics["class_metrics"].get(label, {})
        report_lines.append(
            f"{label:<12} {m.get('precision', 0):>10.4f} {m.get('recall', 0):>10.4f} "
            f"{m.get('f1', 0):>10.4f} {m.get('support', 0):>10d}"
        )
    report_lines.append("")
    
    # 3. 混淆矩阵
    report_lines.append("【3. 混淆矩阵】")
    report_lines.append("-" * 40)
    report_lines.append("行: 真实标签, 列: 预测标签")
    report_lines.append("")
    report_lines.append(format_confusion_matrix(cm, labels))
    report_lines.append("")
    
    # 4. 错误分析
    if errors:
        report_lines.append("【4. 错误分析（每类最多5个示例）】")
        report_lines.append("-" * 40)
        for label, error_list in errors.items():
            if error_list:
                report_lines.append(f"\n真实标签: {label}")
                for i, err in enumerate(error_list, 1):
                    report_lines.append(f"  [{i}] 预测为: {err['pred_label']}")
                    report_lines.append(f"      文本: {err['text']}")
        report_lines.append("")
    
    # 5. sklearn 分类报告（如果可用）
    if HAS_SKLEARN:
        report_lines.append("【5. Sklearn 分类报告】")
        report_lines.append("-" * 40)
        sklearn_report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
        report_lines.append(sklearn_report)
    
    report_lines.append("=" * 80)
    report_lines.append("报告生成完成")
    report_lines.append("=" * 80)
    
    # 输出报告
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # 保存报告
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n报告已保存至: {output_file}")
    
    # 保存 JSON 格式的详细结果
    json_output = output_file.replace(".txt", ".json")
    results = {
        "metrics": {
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "weighted_precision": metrics["weighted_precision"],
            "weighted_recall": metrics["weighted_recall"],
            "weighted_f1": metrics["weighted_f1"]
        },
        "class_metrics": metrics["class_metrics"],
        "confusion_matrix": cm.tolist(),
        "labels": labels
    }
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"JSON 结果已保存至: {json_output}")


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation report for RexUniNLU classification")
    parser.add_argument(
        "--pred_file",
        type=str,
        default="../../log/esconv_strategy/test_pred.json",
        help="Prediction file path (eval_pred.json or test_pred.json)"
    )
    parser.add_argument(
        "--gold_file",
        type=str,
        default="../../log/esconv_strategy/test_gold.json",
        help="Gold label file path (eval_gold.json or test_gold.json)"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="../../models/rex/data/esconv_strategy/test.json",
        help="Original test data file (for error analysis)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../../log/esconv_strategy/evaluation_report.txt",
        help="Output report file path"
    )
    
    args = parser.parse_args()
    
    # 处理相对路径
    script_dir = Path(__file__).parent
    pred_file = (script_dir / args.pred_file).resolve()
    gold_file = (script_dir / args.gold_file).resolve()
    test_file = (script_dir / args.test_file).resolve() if args.test_file else None
    output_file = (script_dir / args.output).resolve()
    
    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    generate_report(
        pred_file=str(pred_file),
        gold_file=str(gold_file),
        output_file=str(output_file),
        test_file=str(test_file) if test_file else None
    )


if __name__ == "__main__":
    main()

