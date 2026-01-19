import json
from pathlib import Path
from typing import List, Dict, Any


def load_esconv_data(file_path: str) -> List[Dict[str, Any]]:
    """
    加载 ESConv JSON 数据文件
    
    Args:
        file_path: ESConv.json 文件路径
        
    Returns:
        包含所有对话会话的列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def analyze_esconv_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    分析 ESConv 数据并返回统计信息
    
    Args:
        data: 对话会话列表
        
    Returns:
        包含统计信息的字典
    """
    total_conversations = len(data)
    total_dialog_turns = 0
    emotion_types = {}
    problem_types = {}
    experience_types = {}
    
    dialog_lengths = []
    
    for conv in data:
        # 统计对话轮次
        dialog = conv.get('dialog', [])
        num_turns = len(dialog)
        total_dialog_turns += num_turns
        dialog_lengths.append(num_turns)
        
        # 统计情绪类型
        emotion = conv.get('emotion_type', 'unknown')
        emotion_types[emotion] = emotion_types.get(emotion, 0) + 1
        
        # 统计问题类型
        problem = conv.get('problem_type', 'unknown')
        problem_types[problem] = problem_types.get(problem, 0) + 1
        
        # 统计经验类型
        experience = conv.get('experience_type', 'unknown')
        experience_types[experience] = experience_types.get(experience, 0) + 1
    
    stats = {
        'total_conversations': total_conversations,
        'total_dialog_turns': total_dialog_turns,
        'average_turns_per_conversation': total_dialog_turns / total_conversations if total_conversations > 0 else 0,
        'min_turns': min(dialog_lengths) if dialog_lengths else 0,
        'max_turns': max(dialog_lengths) if dialog_lengths else 0,
        'emotion_types': emotion_types,
        'problem_types': problem_types,
        'experience_types': experience_types
    }
    
    return stats


def print_statistics(stats: Dict[str, Any]):
    """
    打印统计信息
    
    Args:
        stats: 统计信息字典
    """
    print("=" * 60)
    print("ESConv 数据集统计信息")
    print("=" * 60)
    print(f"\n对话会话总数: {stats['total_conversations']}")
    print(f"对话轮次总数: {stats['total_dialog_turns']}")
    print(f"平均每个会话的轮次数: {stats['average_turns_per_conversation']:.2f}")
    print(f"最短对话轮次: {stats['min_turns']}")
    print(f"最长对话轮次: {stats['max_turns']}")
    
    print("\n" + "-" * 60)
    print("情绪类型分布:")
    print("-" * 60)
    for emotion, count in sorted(stats['emotion_types'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {emotion}: {count}")
    
    print("\n" + "-" * 60)
    print("问题类型分布:")
    print("-" * 60)
    for problem, count in sorted(stats['problem_types'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {problem}: {count}")
    
    print("\n" + "-" * 60)
    print("经验类型分布:")
    print("-" * 60)
    for experience, count in sorted(stats['experience_types'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {experience}: {count}")
    
    print("\n" + "=" * 60)


def main():
    # 获取数据文件路径
    dataset_path = Path(__file__).parent.parent.parent / "Datasets" / "ESConv.json"
    
    if not dataset_path.exists():
        print(f"错误: 找不到数据文件 {dataset_path}")
        return
    
    print(f"正在加载数据文件: {dataset_path}")
    
    # 加载数据
    data = load_esconv_data(str(dataset_path))
    
    # 分析数据
    stats = analyze_esconv_data(data)
    
    # 打印统计信息
    print_statistics(stats)
    
    # 输出用于测试的对话数量
    print(f"\n用于测试的对话会话数量: {stats['total_conversations']}")
    print(f"用于测试的对话轮次总数: {stats['total_dialog_turns']}")


if __name__ == "__main__":
    main()
