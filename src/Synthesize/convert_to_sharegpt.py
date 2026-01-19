"""
将 all_dialogs.json 转换为 ShareGPT 格式

ShareGPT 格式说明:
- 每个对话包含一个 conversations 数组
- 每条消息有 from 和 value 字段
- from: "system" / "human" / "gpt"

使用方法:
    python convert_to_sharegpt.py
    python convert_to_sharegpt.py --input path/to/dialogs.json --output path/to/output.json
"""

import json
import argparse
from pathlib import Path


# ============== 路径配置 ==============
SCRIPT_DIR = Path(__file__).parent.resolve()
SRC_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

DEFAULT_INPUT = PROJECT_ROOT / "Datasets" / "SynthesizedDialogs" / "all_dialogs.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "Datasets" / "SynthesizedDialogs" / "all_dialogs_sharegpt.json"
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "Propmts" / "SystemPropmt.txt"


def load_system_prompt(path: Path) -> str:
    """加载 system prompt"""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def convert_dialog_to_sharegpt(dialog_data: dict, system_prompt: str) -> dict:
    """
    将单个对话转换为 ShareGPT 格式
    
    Args:
        dialog_data: 原始对话数据，包含 situation_idx, persona_idx, situation, data
        system_prompt: system prompt 内容
    
    Returns:
        ShareGPT 格式的对话
    """
    conversations = []
    
    # 转换对话内容
    dialog = dialog_data.get("data", {}).get("dialog", [])
    
    for turn in dialog:
        role = turn.get("role", "")
        text = turn.get("text", "")
        
        # 映射角色: seeker -> human, supporter -> gpt
        if role == "seeker":
            from_role = "human"
        elif role == "supporter":
            from_role = "gpt"
        else:
            # 跳过未知角色
            continue
        
        # 检查是否需要合并连续的同角色消息
        if conversations and conversations[-1]["from"] == from_role:
            # 合并到上一条消息，用换行符连接
            conversations[-1]["value"] += "\n" + text
        else:
            conversations.append({
                "from": from_role,
                "value": text
            })
    
    # 确保最后一条是 gpt，如果是 human 则丢弃
    if conversations and conversations[-1]["from"] == "human":
        conversations.pop()
    
    return {
        "conversations": conversations,
        "system": system_prompt
    }


def convert_all_dialogs(input_path: Path, output_path: Path, system_prompt_path: Path):
    """
    转换所有对话为 ShareGPT 格式
    
    Args:
        input_path: 输入文件路径 (all_dialogs.json)
        output_path: 输出文件路径
        system_prompt_path: system prompt 文件路径
    """
    print(f"加载 system prompt: {system_prompt_path}")
    system_prompt = load_system_prompt(system_prompt_path)
    print(f"System prompt 长度: {len(system_prompt)} 字符")
    
    print(f"\n加载对话数据: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        all_dialogs = json.load(f)
    print(f"总对话数: {len(all_dialogs)}")
    
    print("\n转换为 ShareGPT 格式...")
    sharegpt_dialogs = []
    
    for i, dialog_data in enumerate(all_dialogs):
        converted = convert_dialog_to_sharegpt(dialog_data, system_prompt)
        sharegpt_dialogs.append(converted)
        
        # 进度显示
        if (i + 1) % 1000 == 0:
            print(f"  已处理: {i + 1}/{len(all_dialogs)}")
    
    print(f"\n保存到: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sharegpt_dialogs, f, ensure_ascii=False, indent=2)
    
    # 统计信息
    total_turns = sum(len(d["conversations"]) - 1 for d in sharegpt_dialogs)  # -1 排除 system
    avg_turns = total_turns / len(sharegpt_dialogs) if sharegpt_dialogs else 0
    
    print(f"\n完成!")
    print(f"  转换对话数: {len(sharegpt_dialogs)}")
    print(f"  总对话轮次: {total_turns}")
    print(f"  平均轮次/对话: {avg_turns:.1f}")


def main():
    parser = argparse.ArgumentParser(description="将对话数据转换为 ShareGPT 格式")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"输入文件路径 (默认: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"输出文件路径 (默认: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--system-prompt", "-s",
        type=Path,
        default=SYSTEM_PROMPT_PATH,
        help=f"System prompt 文件路径 (默认: {SYSTEM_PROMPT_PATH})"
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not args.input.exists():
        print(f"错误: 输入文件不存在: {args.input}")
        return 1
    
    if not args.system_prompt.exists():
        print(f"错误: System prompt 文件不存在: {args.system_prompt}")
        return 1
    
    convert_all_dialogs(args.input, args.output, args.system_prompt)
    return 0


if __name__ == "__main__":
    exit(main())

