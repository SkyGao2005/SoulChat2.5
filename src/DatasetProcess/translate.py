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


def load_source_data(file_path: str) -> List[Dict[str, Any]]:
    """加载源数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_translated_data(file_path: str) -> List[Dict[str, Any]]:
    """加载已翻译的数据"""
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 将null转换为None，并过滤掉None值，只返回成功翻译的对话数量
    # 但实际上我们需要保持索引对应，所以返回原始数据
    return data


def save_translated_data(file_path: str, data: List[Dict[str, Any]]):
    """保存翻译后的数据"""
    # 保存所有数据，包括None（会保存为null），以保持索引对应
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def remove_survey_score(conversation: Dict[str, Any]) -> Dict[str, Any]:
    """移除survey_score字段"""
    conv_copy = conversation.copy()
    conv_copy.pop('survey_score', None)
    conv_copy.pop('seeker_question1', None)
    conv_copy.pop('seeker_question2', None)
    conv_copy.pop('supporter_question1', None)
    conv_copy.pop('supporter_question2', None)
    return conv_copy


def validate_dialog_structure(original_dialog: List[Dict], translated_dialog: List[Dict]) -> tuple[bool, str]:
    """
    验证翻译后的dialog结构是否与原始dialog对应
    检查speaker和annotation是否匹配
    
    Returns:
        (is_valid, error_message): 验证结果和错误信息（如果验证通过则为空字符串）
    """
    # 检查dialog长度是否一致
    if len(original_dialog) != len(translated_dialog):
        return False, f"dialog长度不一致: 原始={len(original_dialog)}, 翻译后={len(translated_dialog)}"
    
    for i, (orig, trans) in enumerate(zip(original_dialog, translated_dialog)):
        turn_idx = i + 1  # 从1开始的轮次编号
        
        # 检查speaker是否一致
        orig_speaker = orig.get('speaker')
        trans_speaker = trans.get('speaker')
        if orig_speaker != trans_speaker:
            return False, f"第{turn_idx}轮speaker不一致: 原始='{orig_speaker}', 翻译后='{trans_speaker}'"
        
        # 检查annotation是否一致（只检查键，不检查值，因为值可能被翻译）
        orig_annotation = orig.get('annotation', {})
        trans_annotation = trans.get('annotation', {})
        
        # 检查annotation的键是否一致
        orig_keys = set(orig_annotation.keys())
        trans_keys = set(trans_annotation.keys())
        if orig_keys != trans_keys:
            missing = orig_keys - trans_keys
            extra = trans_keys - orig_keys
            error_parts = []
            if missing:
                error_parts.append(f"缺少字段: {missing}")
            if extra:
                error_parts.append(f"多余字段: {extra}")
            return False, f"第{turn_idx}轮annotation键不一致: {', '.join(error_parts)}"
        
        # 如果annotation中有strategy字段，检查是否一致（strategy不应该被翻译）
        if 'strategy' in orig_annotation:
            orig_strategy = orig_annotation['strategy']
            trans_strategy = trans_annotation.get('strategy')
            if orig_strategy != trans_strategy:
                return False, f"第{turn_idx}轮strategy不一致: 原始='{orig_strategy}', 翻译后='{trans_strategy}'"
        
        # 如果annotation中有feedback字段，检查是否一致（feedback不应该被翻译）
        if 'feedback' in orig_annotation:
            orig_feedback = orig_annotation['feedback']
            trans_feedback = trans_annotation.get('feedback')
            if orig_feedback != trans_feedback:
                return False, f"第{turn_idx}轮feedback不一致: 原始='{orig_feedback}', 翻译后='{trans_feedback}'"
    
    return True, ""


def translate_batch(client: OpenAI, prompt: str, conversations: List[Dict[str, Any]], model: str = "qwen3-max") -> Optional[List[Dict[str, Any]]]:
    """
    翻译一批对话（5个）
    
    Args:
        client: OpenAI客户端
        prompt: 翻译提示词
        conversations: 要翻译的对话列表（最多5个）
        model: 模型名称，默认为qwen-plus，但用户要求使用qwen3-max
        
    Returns:
        翻译后的dialog列表，如果失败返回None
    """
    # 准备输入数据（去掉survey_score）
    input_data = [remove_survey_score(conv) for conv in conversations]
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
            temperature=1.3
        )
        
        # 检查响应是否被截断
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "length":
            print(f"警告: 响应被截断（finish_reason={finish_reason}），可能需要减少每批翻译的数量")
        
        # 提取返回内容
        content = response.choices[0].message.content.strip()
        
        # 尝试解析JSON（可能包含markdown代码块）
        if content.startswith("```"):
            # 移除markdown代码块标记
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
        
        # 检查返回格式并提取dialog
        translated_dialogs = []
        
        if isinstance(result, list):
            # 如果返回的是列表，提取每个元素的dialog字段
            for item in result:
                if isinstance(item, dict) and 'dialog' in item:
                    translated_dialogs.append(item['dialog'])
                elif isinstance(item, list):
                    # 如果item本身就是dialog数组
                    translated_dialogs.append(item)
                else:
                    print(f"警告: 列表中的元素格式不符合预期: {type(item)}")
                    return None
        elif isinstance(result, dict):
            if 'dialog' in result:
                # 如果返回的是单个对象，包含dialog字段
                translated_dialogs.append(result['dialog'])
            else:
                print(f"警告: 返回的字典中没有dialog字段")
                return None
        else:
            print(f"警告: 返回格式不符合预期: {type(result)}")
            return None
        
        # 检查返回的dialog数量是否与输入一致
        if len(translated_dialogs) != len(conversations):
            print(f"警告: 返回的dialog数量 ({len(translated_dialogs)}) 与输入数量 ({len(conversations)}) 不一致")
            # 如果数量不匹配，仍然返回，让上层处理
        
        return translated_dialogs
            
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        print(f"返回内容: {content[:500]}...")
        return None
    except Exception as e:
        print(f"翻译错误: {e}")
        return None


def translate_dataset(
    source_path: str,
    target_path: str,
    prompt_path: str,
    batch_size: int = 5,
    model: str = "qwen3-max",
    max_translations: Optional[int] = None,
    concurrent_batches: int = 5
):
    """
    翻译整个数据集
    
    Args:
        source_path: 源数据文件路径
        target_path: 目标数据文件路径
        prompt_path: 提示词文件路径
        batch_size: 每批翻译的对话数量
        model: 使用的模型名称
        max_translations: 最大翻译数量限制（None表示无限制）
        concurrent_batches: 并发请求的批次数量（默认5个batch同时发送）
    """
    # 初始化OpenAI客户端
    # 优先使用环境变量，如果没有则使用代码中的默认值
    api_key = os.getenv("DASHSCOPE_API_KEY")
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",#"https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    # 加载提示词
    print("正在加载翻译提示词...")
    prompt = load_translator_prompt(prompt_path)
    
    # 加载源数据
    print("正在加载源数据...")
    source_data = load_source_data(source_path)
    total_conversations = len(source_data)
    print(f"源数据共有 {total_conversations} 个对话")
    
    # 加载已翻译的数据
    print("正在检查已翻译的数据...")
    translated_data = load_translated_data(target_path)
    # 计算已成功翻译的对话数量（排除None/null）
    translated_count = sum(1 for item in translated_data if item is not None)
    # 从已翻译数据的长度开始（保持索引对应）
    start_index = len(translated_data) if translated_data else 0
    print(f"已翻译 {translated_count} 个对话，从第 {start_index + 1} 个开始翻译")
    
    if start_index >= total_conversations:
        print("所有对话已翻译完成！")
        return
    
    # 检查翻译数量限制
    if max_translations is not None:
        remaining_limit = max_translations - translated_count
        if remaining_limit <= 0:
            print(f"已达到翻译数量限制 ({max_translations})，停止翻译")
            return
        print(f"翻译数量限制: {max_translations}，本次最多翻译 {remaining_limit} 个对话")
    
    # 开始翻译
    current_index = start_index
    round_num = 0
    newly_translated = 0  # 本次新翻译的数量
    
    print(f"并发模式: 每轮同时发送 {concurrent_batches} 个批次请求")
    
    while current_index < total_conversations:
        # 检查是否达到翻译数量限制
        if max_translations is not None:
            current_translated_count = sum(1 for item in translated_data if item is not None)
            if current_translated_count >= max_translations:
                print(f"\n已达到翻译数量限制 ({max_translations})，停止翻译")
                break
        
        round_num += 1
        
        # 准备当前轮次的所有批次
        batches_info = []  # [(batch_start, batch_end, batch_data), ...]
        temp_index = current_index
        
        for _ in range(concurrent_batches):
            if temp_index >= total_conversations:
                break
            # 检查翻译数量限制
            if max_translations is not None:
                remaining = max_translations - (sum(1 for item in translated_data if item is not None) + len(batches_info) * batch_size)
                if remaining <= 0:
                    break
            
            batch_end = min(temp_index + batch_size, total_conversations)
            batch_data = source_data[temp_index:batch_end]
            batches_info.append((temp_index, batch_end, batch_data))
            temp_index = batch_end
        
        if not batches_info:
            break
        
        print(f"\n[轮次 {round_num}] 并发翻译 {len(batches_info)} 个批次...")
        for batch_start, batch_end, _ in batches_info:
            print(f"  - 批次: 第 {batch_start + 1}-{batch_end} 个对话")
        
        # 使用线程池并发发送翻译请求
        batch_results = {}  # {batch_start: (translated_dialogs, batch_data, batch_end)}
        
        def translate_single_batch(batch_info):
            """单个批次的翻译任务"""
            batch_start, batch_end, batch_data = batch_info
            translated_dialogs = translate_batch(client, prompt, batch_data, model)
            return batch_start, batch_end, batch_data, translated_dialogs
        
        with ThreadPoolExecutor(max_workers=concurrent_batches) as executor:
            futures = {executor.submit(translate_single_batch, info): info for info in batches_info}
            
            for future in as_completed(futures):
                batch_start, batch_end, batch_data, translated_dialogs = future.result()
                batch_results[batch_start] = (translated_dialogs, batch_data, batch_end)
                print(f"  ✓ 批次 {batch_start + 1}-{batch_end} 翻译完成")
        
        print(f"\n正在验证 {len(batch_results)} 个批次的翻译结果...")
        
        # 按顺序验证和处理结果
        for batch_start in sorted(batch_results.keys()):
            translated_dialogs, batch_data, batch_end = batch_results[batch_start]
            
            # 检查翻译是否成功
            if translated_dialogs is None:
                # 保存已翻译的内容
                if translated_data:
                    print(f"翻译失败，正在保存已翻译的内容...")
                    save_translated_data(target_path, translated_data)
                    translated_count = sum(1 for item in translated_data if item is not None)
                    print(f"已保存 {translated_count} 个成功翻译的对话")
                raise RuntimeError(f"翻译批次失败（第 {batch_start + 1}-{batch_end} 个对话）")
            
            # 验证返回数量
            if len(translated_dialogs) != len(batch_data):
                # 保存已翻译的内容
                if translated_data:
                    print(f"错误发生，正在保存已翻译的内容...")
                    save_translated_data(target_path, translated_data)
                    translated_count = sum(1 for item in translated_data if item is not None)
                    print(f"已保存 {translated_count} 个成功翻译的对话")
                raise RuntimeError(f"返回的dialog数量 ({len(translated_dialogs)}) 与批次大小 ({len(batch_data)}) 不匹配（第 {batch_start + 1}-{batch_end} 个对话）")
            
            # 验证每个对话的结构
            for i, (orig_conv, trans_dialog) in enumerate(zip(batch_data, translated_dialogs)):
                conv_index = batch_start + i
                
                is_valid, error_msg = validate_dialog_structure(orig_conv['dialog'], trans_dialog)
                if not is_valid:
                    # 保存已翻译的内容
                    if translated_data:
                        print(f"结构验证失败，正在保存已翻译的内容...")
                        save_translated_data(target_path, translated_data)
                        translated_count = sum(1 for item in translated_data if item is not None)
                        print(f"已保存 {translated_count} 个成功翻译的对话")
                    raise RuntimeError(f"第 {conv_index + 1} 个对话的结构验证失败: {error_msg}")
                
                # 构建翻译后的对话对象
                translated_conv = orig_conv.copy()
                translated_conv['dialog'] = trans_dialog
                translated_conv['dialog_id'] = conv_index + 1
                
                # 添加到已翻译数据
                if conv_index >= len(translated_data):
                    while len(translated_data) < conv_index:
                        translated_data.append(None)
                    translated_data.append(translated_conv)
                else:
                    translated_data[conv_index] = translated_conv
                
                newly_translated += 1
            
            print(f"  ✓ 批次 {batch_start + 1}-{batch_end} 验证通过")
        
        # 保存进度
        print(f"正在保存进度...")
        save_translated_data(target_path, translated_data)
        translated_count = sum(1 for item in translated_data if item is not None)
        print(f"已保存 {translated_count} 个成功翻译的对话（共 {len(translated_data)} 个位置）")
        
        # 检查是否达到翻译数量限制
        if max_translations is not None and translated_count >= max_translations:
            print(f"\n已达到翻译数量限制 ({max_translations})，停止翻译")
            break
        
        # 更新索引为最后一个批次的结束位置
        current_index = max(batch_end for _, batch_end, _ in batches_info)
        
        # 添加延迟，避免API限流
        if current_index < total_conversations:
            time.sleep(1)
    
    translated_count = sum(1 for item in translated_data if item is not None)
    print(f"\n翻译完成！共成功翻译 {translated_count} 个对话（共 {len(translated_data)} 个位置）")
    print(f"结果已保存到: {target_path}")


def main():
    # 文件路径
    base_path = Path(__file__).parent.parent.parent
    source_path = base_path / "Datasets" / "ESConv.json"
    target_path = base_path / "Datasets" / "ESConv_zh.json"
    prompt_path = base_path / "Datasets" / "TranslatorPrompt.txt"
    
    # 检查文件是否存在
    if not source_path.exists():
        print(f"错误: 找不到源数据文件 {source_path}")
        return
    
    if not prompt_path.exists():
        print(f"错误: 找不到提示词文件 {prompt_path}")
        return
    
    # 检查API密钥（支持环境变量或代码中的硬编码）
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误: 未设置 API 密钥")
        return
    
    # 获取翻译数量限制（可通过环境变量 MAX_TRANSLATIONS 设置，或直接在代码中修改）
    # 默认值：5（只翻译一批进行实验）
    max_translations = os.getenv("MAX_TRANSLATIONS")
    if max_translations:
        try:
            max_translations = int(max_translations)
        except ValueError:
            print(f"警告: MAX_TRANSLATIONS 环境变量值无效，使用默认值")
            max_translations = 5
    else:
        # 可以在这里直接修改默认值，比如设置为 5 表示只翻译一批
        max_translations = 5  # 设置为 None 表示无限制
    
    if max_translations:
        print(f"翻译数量限制: {max_translations} 个对话")
    else:
        print("无翻译数量限制，将翻译所有对话")
    
    # 开始翻译
    translate_dataset(
        source_path=str(source_path),
        target_path=str(target_path),
        prompt_path=str(prompt_path),
        batch_size=2,
        model="deepseek-reasoner",
        max_translations=max_translations,
        concurrent_batches=5  # 同时发送5个批次请求
    )


if __name__ == "__main__":
    main()

