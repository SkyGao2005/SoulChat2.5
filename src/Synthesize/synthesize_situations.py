"""
SoulChatCorpus 数据集统计、抽样与情境合成脚本

功能：
1. 统计数据集内容比例（topic分布、对话轮数分布等）
2. 抽样5000个对话作为话题种子
3. 调用 DeepSeek API 总结来访者遇到的问题作为 situation
"""

import json
import random
import os
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import asyncio
from openai import AsyncOpenAI
import argparse


# 数据集路径
DATASET_PATH = Path(__file__).parent.parent.parent / "Datasets" / "SoulChatCorpus" / "SoulChatCorpus-sft-multi-Turn.json"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "Datasets" / "SynthesizedSituations"


def load_dataset(path: Path) -> list:
    """加载数据集"""
    print(f"正在加载数据集: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"数据集加载完成，共 {len(data)} 条对话")
    return data


def analyze_dataset(data: list) -> dict:
    """分析数据集统计信息"""
    print("\n" + "=" * 60)
    print("数据集统计信息")
    print("=" * 60)
    
    stats = {
        "total_dialogs": len(data),
        "topic_distribution": Counter(),
        "turn_distribution": Counter(),
        "avg_turns": 0,
        "avg_user_msg_length": 0,
        "avg_assistant_msg_length": 0,
    }
    
    total_turns = 0
    total_user_chars = 0
    total_assistant_chars = 0
    total_user_msgs = 0
    total_assistant_msgs = 0
    
    for item in tqdm(data, desc="分析数据"):
        # 统计 topic 分布
        topic = item.get("topic", "未知")
        stats["topic_distribution"][topic] += 1
        
        # 统计对话轮数
        messages = item.get("messages", [])
        num_turns = len(messages) // 2  # 一轮 = 用户消息 + 助手回复
        stats["turn_distribution"][num_turns] += 1
        total_turns += num_turns
        
        # 统计消息长度
        for msg in messages:
            if msg.get("role") == "user":
                total_user_chars += len(msg.get("content", ""))
                total_user_msgs += 1
            elif msg.get("role") == "assistant":
                total_assistant_chars += len(msg.get("content", ""))
                total_assistant_msgs += 1
    
    stats["avg_turns"] = total_turns / len(data) if data else 0
    stats["avg_user_msg_length"] = total_user_chars / total_user_msgs if total_user_msgs else 0
    stats["avg_assistant_msg_length"] = total_assistant_chars / total_assistant_msgs if total_assistant_msgs else 0
    
    # 打印统计信息
    print(f"\n总对话数: {stats['total_dialogs']}")
    print(f"平均轮数: {stats['avg_turns']:.2f}")
    print(f"用户消息平均长度: {stats['avg_user_msg_length']:.2f} 字符")
    print(f"咨询师消息平均长度: {stats['avg_assistant_msg_length']:.2f} 字符")
    
    print(f"\n话题(Topic)分布 (共 {len(stats['topic_distribution'])} 种):")
    print("-" * 40)
    for topic, count in stats["topic_distribution"].most_common(30):
        percentage = count / stats['total_dialogs'] * 100
        print(f"  {topic}: {count} ({percentage:.2f}%)")
    if len(stats['topic_distribution']) > 30:
        print(f"  ... 还有 {len(stats['topic_distribution']) - 30} 种话题")
    
    print(f"\n对话轮数分布:")
    print("-" * 40)
    turn_counts = sorted(stats["turn_distribution"].items())
    for turns, count in turn_counts[:20]:
        percentage = count / stats['total_dialogs'] * 100
        print(f"  {turns} 轮: {count} ({percentage:.2f}%)")
    if len(turn_counts) > 20:
        print(f"  ... 还有 {len(turn_counts) - 20} 种轮数")
    
    return stats


def sample_dialogs(data: list, n: int = 5000, seed: int = 42) -> list:
    """
    从数据集中分层抽样 n 个对话，保证各 topic 比例与原数据集相同
    """
    print(f"\n正在进行分层抽样 {n} 个对话（保持 topic 比例）...")
    random.seed(seed)
    
    if len(data) <= n:
        print(f"数据集大小 ({len(data)}) 小于等于抽样数量 ({n})，返回全部数据")
        return data
    
    # 按 topic 分组
    topic_groups = {}
    for item in data:
        topic = item.get("topic", "未知")
        if topic not in topic_groups:
            topic_groups[topic] = []
        topic_groups[topic].append(item)
    
    print(f"共 {len(topic_groups)} 种话题")
    
    # 计算每个 topic 应抽样的数量（按比例分配）
    total_count = len(data)
    sampled = []
    remaining_quota = n
    
    # 先按比例计算每个 topic 的抽样数（向下取整）
    topic_sample_counts = {}
    for topic, items in topic_groups.items():
        proportion = len(items) / total_count
        sample_count = int(n * proportion)
        topic_sample_counts[topic] = sample_count
        remaining_quota -= sample_count
    
    # 将剩余配额分配给样本数最多的 topic（处理取整误差）
    sorted_topics = sorted(topic_groups.keys(), key=lambda t: len(topic_groups[t]), reverse=True)
    for topic in sorted_topics:
        if remaining_quota <= 0:
            break
        topic_sample_counts[topic] += 1
        remaining_quota -= 1
    
    # 执行分层抽样
    print("\n各话题抽样情况:")
    print("-" * 50)
    for topic, items in sorted(topic_groups.items(), key=lambda x: len(x[1]), reverse=True):
        sample_count = topic_sample_counts[topic]
        # 如果该 topic 的样本数少于需要抽样的数量，则全部选取
        actual_count = min(sample_count, len(items))
        topic_sampled = random.sample(items, actual_count)
        sampled.extend(topic_sampled)
        
        original_pct = len(items) / total_count * 100
        sampled_pct = actual_count / n * 100
        print(f"  {topic}: 原 {len(items)} ({original_pct:.2f}%) -> 抽样 {actual_count} ({sampled_pct:.2f}%)")
    
    # 打乱顺序
    random.shuffle(sampled)
    
    print(f"\n分层抽样完成，共 {len(sampled)} 个对话")
    return sampled


def format_dialog_for_summary(dialog: dict) -> str:
    """格式化对话，只提取来访者（用户）的消息用于总结"""
    messages = dialog.get("messages", [])
    user_messages = []
    
    for msg in messages:
        if msg.get("role") == "user":
            user_messages.append(msg.get("content", ""))
    
    return "\n".join(user_messages)


def create_summary_prompt(user_messages: str) -> str:
    """创建用于总结来访者情境的 prompt"""
    return f"""以下是一位心理咨询来访者在咨询过程中说的话。请总结这位来访者遇到的问题和情况，只描述来访者的状况，不要包含任何关于咨询师的内容。

来访者的话：
{user_messages}

请直接输出总结，不要有任何前缀、后缀或解释性文字。"""


async def summarize_with_deepseek(
    client: AsyncOpenAI,
    dialog: dict,
    semaphore: asyncio.Semaphore,
    model: str = "deepseek-chat"
) -> dict | None:
    """使用 DeepSeek API 总结单个对话的来访者情境，只返回 situation"""
    async with semaphore:
        user_messages = format_dialog_for_summary(dialog)
        prompt = create_summary_prompt(user_messages)
        
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500,
            )
            
            situation = response.choices[0].message.content.strip()
            return situation
        except Exception as e:
            print(f"处理对话 {dialog.get('id')} 时出错: {e}")
            return None


async def batch_summarize(
    dialogs: list,
    api_key: str,
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-chat",
    concurrency: int = 10,
    save_interval: int = 100
) -> list[str]:
    """批量处理对话并生成情境总结，只返回 situation 列表"""
    print(f"\n正在使用 DeepSeek API 生成情境总结...")
    print(f"并发数: {concurrency}, 保存间隔: {save_interval}")
    
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    semaphore = asyncio.Semaphore(concurrency)
    
    situations = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 分批处理并定期保存
    for i in tqdm(range(0, len(dialogs), save_interval), desc="处理批次"):
        batch = dialogs[i:i + save_interval]
        tasks = [
            summarize_with_deepseek(client, dialog, semaphore, model)
            for dialog in batch
        ]
        batch_results = await asyncio.gather(*tasks)
        
        # 只保留成功的 situation
        valid_situations = [s for s in batch_results if s is not None]
        situations.extend(valid_situations)
        
        # 保存中间结果（只保存 situation 列表）
        temp_path = OUTPUT_DIR / "situations_temp.json"
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(situations, f, ensure_ascii=False, indent=2)
        
        success_count = len(valid_situations)
        print(f"  批次 {i // save_interval + 1}: 成功 {success_count}/{len(batch)}")
    
    return situations


def save_situations(situations: list[str], output_path: Path):
    """保存 situation 列表"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(situations, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_path}")


async def main():
    parser = argparse.ArgumentParser(description="SoulChatCorpus 数据集统计、抽样与情境合成")
    parser.add_argument("--api-key", type=str, help="DeepSeek API Key (也可通过 DEEPSEEK_API_KEY 环境变量设置)")
    parser.add_argument("--base-url", type=str, default="https://api.deepseek.com", help="API Base URL")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="模型名称")
    parser.add_argument("--sample-size", type=int, default=5000, help="抽样数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--concurrency", type=int, default=10, help="API 并发数")
    parser.add_argument("--stats-only", action="store_true", help="只进行统计分析，不调用 API")
    
    args = parser.parse_args()
    
    # 获取 API Key
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    
    # 加载数据集
    data = load_dataset(DATASET_PATH)
    
    # 分析数据集（只打印统计信息，不保存）
    analyze_dataset(data)
    
    if args.stats_only:
        print("\n统计分析完成!")
        return
    
    # 检查 API Key
    if not api_key:
        print("\n错误: 未提供 DeepSeek API Key，无法进行情境总结。")
        print("请通过 --api-key 参数或 DEEPSEEK_API_KEY 环境变量提供 API Key。")
        return
    
    # 抽样对话
    sampled = sample_dialogs(data, n=args.sample_size, seed=args.seed)
    
    # 使用 DeepSeek API 生成情境总结
    situations = await batch_summarize(
        sampled,
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        concurrency=args.concurrency
    )
    
    # 保存结果（只保存 situation 列表）
    save_situations(situations, OUTPUT_DIR / "situations.json")
    
    # 打印统计
    print(f"\n处理完成!")
    print(f"成功生成 {len(situations)} 个 situation")


if __name__ == "__main__":
    asyncio.run(main())

