"""
翻译ESConv_zh.json中的元数据字段：experience_type, emotion_type, problem_type, situation
"""

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI


def load_translator_prompt(prompt_path: str) -> str:
    """加载翻译提示词"""
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """加载数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_data(file_path: str, data: List[Dict[str, Any]]):
    """保存数据"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def is_chinese(text: str) -> bool:
    """检查文本是否包含中文字符"""
    if not text:
        return False
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False


def needs_translation(item: Dict[str, Any]) -> bool:
    """检查该条目是否需要翻译（任一字段尚未翻译成中文）"""
    if item is None:
        return False
    
    # 检查situation字段是否已翻译（situation是最能体现是否翻译的字段）
    situation = item.get('situation', '')
    if situation and not is_chinese(situation):
        return True
    
    # 检查其他字段
    experience_type = item.get('experience_type', '')
    emotion_type = item.get('emotion_type', '')
    problem_type = item.get('problem_type', '')
    
    # 如果任一字段是英文，则需要翻译
    if experience_type and not is_chinese(experience_type):
        return True
    if emotion_type and not is_chinese(emotion_type):
        return True
    if problem_type and not is_chinese(problem_type):
        return True
    
    return False


def translate_batch(
    client: OpenAI, 
    prompt: str, 
    items: List[Dict[str, Any]], 
    indices: List[int],
    model: str = "qwen3-max"
) -> Optional[List[Dict[str, Any]]]:
    """
    翻译一批元数据
    
    Args:
        client: OpenAI客户端
        prompt: 翻译提示词
        items: 要翻译的数据列表
        indices: 对应的原始索引
        model: 模型名称
        
    Returns:
        翻译后的元数据列表，如果失败返回None
    """
    # 准备输入数据
    input_data = []
    for idx, item in zip(indices, items):
        input_data.append({
            "index": idx,
            "experience_type": item.get('experience_type', ''),
            "emotion_type": item.get('emotion_type', ''),
            "problem_type": item.get('problem_type', ''),
            "situation": item.get('situation', '')
        })
    
    input_json = json.dumps(input_data, ensure_ascii=False, indent=2)
    
    # 构建消息
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": input_json}
    ]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3  # 翻译任务使用较低温度以保证一致性
        )
        
        # 检查响应是否被截断
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "length":
            print(f"警告: 响应被截断（finish_reason={finish_reason}）")
        
        # 提取返回内容
        content = response.choices[0].message.content.strip()
        
        # 尝试解析JSON（可能包含markdown代码块）
        if content.startswith("```"):
            lines = content.split('\n')
            if lines[0].startswith("```json"):
                lines = lines[1:]
            elif lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].strip() == "```":
                lines = lines[:-1]
            content = '\n'.join(lines)
        
        # 解析JSON
        result = json.loads(content)
        
        if not isinstance(result, list):
            print(f"警告: 返回格式不符合预期，应为列表: {type(result)}")
            return None
        
        if len(result) != len(items):
            print(f"警告: 返回数量 ({len(result)}) 与输入数量 ({len(items)}) 不一致")
            return None
        
        return result
            
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        print(f"返回内容: {content[:500]}...")
        return None
    except Exception as e:
        print(f"翻译错误: {e}")
        return None


def translate_metadata(
    data_path: str,
    prompt_path: str,
    batch_size: int = 10,
    model: str = "deepseek-reasoner",
    max_translations: Optional[int] = None,
    concurrent_batches: int = 5,
    save_interval: int = 50
):
    """
    翻译数据集中的元数据字段
    
    Args:
        data_path: 数据文件路径
        prompt_path: 提示词文件路径
        batch_size: 每批翻译的数量
        model: 使用的模型名称
        max_translations: 最大翻译数量限制（None表示无限制）
        concurrent_batches: 并发请求的批次数量
        save_interval: 每翻译多少条保存一次
    """
    # 初始化OpenAI客户端
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误: 未设置 DASHSCOPE_API_KEY 环境变量")
        return
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )
    
    # 加载提示词
    print("正在加载翻译提示词...")
    prompt = load_translator_prompt(prompt_path)
    
    # 加载数据
    print("正在加载数据...")
    data = load_data(data_path)
    total_items = len(data)
    print(f"数据共有 {total_items} 条记录")
    
    # 找出需要翻译的条目
    items_to_translate = []
    for i, item in enumerate(data):
        if needs_translation(item):
            items_to_translate.append((i, item))
    
    print(f"需要翻译的条目: {len(items_to_translate)} 条")
    
    if not items_to_translate:
        print("所有条目已翻译完成！")
        return
    
    # 应用翻译数量限制
    if max_translations is not None:
        items_to_translate = items_to_translate[:max_translations]
        print(f"翻译数量限制: {max_translations}，本次将翻译 {len(items_to_translate)} 条")
    
    # 开始翻译
    translated_count = 0
    round_num = 0
    
    print(f"并发模式: 每轮同时发送 {concurrent_batches} 个批次请求")
    
    current_pos = 0
    while current_pos < len(items_to_translate):
        round_num += 1
        
        # 准备当前轮次的所有批次
        batches_info = []  # [(batch_indices, batch_items), ...]
        temp_pos = current_pos
        
        for _ in range(concurrent_batches):
            if temp_pos >= len(items_to_translate):
                break
            
            batch_end = min(temp_pos + batch_size, len(items_to_translate))
            batch_data = items_to_translate[temp_pos:batch_end]
            batch_indices = [item[0] for item in batch_data]
            batch_items = [item[1] for item in batch_data]
            batches_info.append((batch_indices, batch_items))
            temp_pos = batch_end
        
        if not batches_info:
            break
        
        print(f"\n[轮次 {round_num}] 并发翻译 {len(batches_info)} 个批次...")
        for batch_indices, _ in batches_info:
            print(f"  - 批次: 索引 {batch_indices[0]}-{batch_indices[-1]}")
        
        # 使用线程池并发发送翻译请求
        batch_results = {}
        
        def translate_single_batch(batch_info):
            batch_indices, batch_items = batch_info
            result = translate_batch(client, prompt, batch_items, batch_indices, model)
            return batch_indices, result
        
        with ThreadPoolExecutor(max_workers=concurrent_batches) as executor:
            futures = {executor.submit(translate_single_batch, info): info for info in batches_info}
            
            for future in as_completed(futures):
                batch_indices, result = future.result()
                batch_results[batch_indices[0]] = (batch_indices, result)
                if result:
                    print(f"  ✓ 批次 {batch_indices[0]}-{batch_indices[-1]} 翻译完成")
                else:
                    print(f"  ✗ 批次 {batch_indices[0]}-{batch_indices[-1]} 翻译失败")
        
        # 按顺序处理结果
        for batch_start in sorted(batch_results.keys()):
            batch_indices, result = batch_results[batch_start]
            
            if result is None:
                print(f"警告: 批次 {batch_start} 翻译失败，跳过")
                continue
            
            # 更新数据
            for translated_item in result:
                idx = translated_item.get('index')
                if idx is None:
                    continue
                
                # 更新对应位置的数据
                if idx < len(data) and data[idx] is not None:
                    data[idx]['experience_type'] = translated_item.get('experience_type', data[idx].get('experience_type', ''))
                    data[idx]['emotion_type'] = translated_item.get('emotion_type', data[idx].get('emotion_type', ''))
                    data[idx]['problem_type'] = translated_item.get('problem_type', data[idx].get('problem_type', ''))
                    data[idx]['situation'] = translated_item.get('situation', data[idx].get('situation', ''))
                    translated_count += 1
        
        # 更新位置
        current_pos = temp_pos
        
        # 定期保存
        if translated_count > 0 and translated_count % save_interval == 0:
            print(f"正在保存进度 ({translated_count} 条已翻译)...")
            save_data(data_path, data)
        
        # 添加延迟，避免API限流
        if current_pos < len(items_to_translate):
            time.sleep(0.5)
    
    # 最终保存
    print(f"\n正在保存最终结果...")
    save_data(data_path, data)
    print(f"翻译完成！共翻译 {translated_count} 条记录")
    print(f"结果已保存到: {data_path}")


def main():
    # 文件路径
    base_path = Path(__file__).parent.parent.parent
    data_path = base_path / "Datasets" / "ESConv_zh.json"
    prompt_path = base_path / "Datasets" / "MetadataTranslatorPrompt.txt"
    
    # 检查文件是否存在
    if not data_path.exists():
        print(f"错误: 找不到数据文件 {data_path}")
        return
    
    if not prompt_path.exists():
        print(f"错误: 找不到提示词文件 {prompt_path}")
        return
    
    # 检查API密钥
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误: 未设置 DASHSCOPE_API_KEY 环境变量")
        return
    
    # 获取翻译数量限制
    max_translations = os.getenv("MAX_TRANSLATIONS")
    if max_translations:
        try:
            max_translations = int(max_translations)
        except ValueError:
            print(f"警告: MAX_TRANSLATIONS 环境变量值无效，使用默认值")
            max_translations = None
    else:
        max_translations = None  # 默认无限制
    
    if max_translations:
        print(f"翻译数量限制: {max_translations} 条")
    else:
        print("无翻译数量限制，将翻译所有需要翻译的条目")
    
    # 开始翻译
    translate_metadata(
        data_path=str(data_path),
        prompt_path=str(prompt_path),
        batch_size=10,  # 元数据较小，可以每批翻译更多
        model="deepseek-reasoner",
        max_translations=max_translations,
        concurrent_batches=5,
        save_interval=100
    )


if __name__ == "__main__":
    main()

