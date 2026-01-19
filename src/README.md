# SoulChat2.5 源码说明

本目录包含 SoulChat2.5 项目的核心代码，分为三个子模块：

## 目录结构

```
src/
├── DatasetProcess/     # 数据集处理脚本
├── Synthesize/         # 数据合成脚本
└── rex/                # RexUniNLU 策略分类模型
```

---

## 1. DatasetProcess/ - 数据集处理

### translate.py
**功能**：将 ESConv 英文心理对话数据集翻译为中文

**使用方法**：
```bash
# 设置环境变量
export DASHSCOPE_API_KEY="your-api-key"

# 运行翻译（默认翻译 5 条用于测试）
python translate.py

# 设置翻译数量限制
export MAX_TRANSLATIONS=100
python translate.py
```

**配置参数**：
- `batch_size`: 每批翻译的对话数量（默认 2）
- `concurrent_batches`: 并发请求数量（默认 5）
- `model`: 使用的模型（默认 deepseek-reasoner）

**输入/输出**：
- 输入：`Datasets/ESConv.json`
- 输出：`Datasets/ESConv_zh.json`

---

### translate_metadata.py
**功能**：翻译 ESConv_zh.json 中的元数据字段（experience_type, emotion_type, problem_type, situation）

**使用方法**：
```bash
export DASHSCOPE_API_KEY="your-api-key"
python translate_metadata.py
```

**输入/输出**：
- 输入/输出：`Datasets/ESConv_zh.json`（原地更新）

---

### prepare_rex_classify.py
**功能**：将 ESConv_zh.json 转换为 RexUniNLU 模型的 [CLASSIFY] 任务格式

**使用方法**：
```bash
python prepare_rex_classify.py \
    --input ../../Datasets/ESConv_zh.json \
    --output ../../models/rex/data/esconv_strategy \
    --train_ratio 0.8 \
    --dev_ratio 0.1 \
    --max_history_turns 10
```

**参数说明**：
- `--input`: 输入文件路径
- `--output`: 输出目录
- `--train_ratio`: 训练集比例（默认 0.8）
- `--dev_ratio`: 验证集比例（默认 0.1）
- `--max_history_turns`: 最大历史对话轮数（默认 10）
- `--seed`: 随机种子（默认 42）

**输出文件**：
- `train.json`: 训练数据
- `dev.json`: 验证数据
- `test.json`: 测试数据
- `label_map.json`: 策略标签映射

---

### predict_rex_classify.py
**功能**：使用训练好的 RexUniNLU 模型进行策略分类推理

**使用方法**：
```bash
# 单次预测
python predict_rex_classify.py \
    --utterance "这确实很难，但我觉得你能够成功的。" \
    --situation "我最近工作压力很大" \
    --history "支持者：你好；求助者：最近压力很大"

# 交互模式
python predict_rex_classify.py --interactive
```

**参数说明**：
- `--model_dir`: 训练好的模型目录
- `--utterance`: 当前 supporter 发言（必需）
- `--situation`: 情境描述（可选）
- `--history`: 对话历史（可选）
- `--device`: 设备（cuda/cpu）
- `--interactive`: 交互模式

**策略标签**：
| 中文 | 英文 |
|------|------|
| 提问 | Question |
| 肯定与安慰 | Affirmation and Reassurance |
| 复述与转述 | Restatement or Paraphrasing |
| 自我表露 | Self-disclosure |
| 提供建议 | Providing Suggestions |
| 提供信息 | Information |
| 反映情感 | Reflection of feelings |
| 其他 | Others |

---

### evaluate_rex_classify.py
**功能**：生成策略分类任务的详细评估报告

**使用方法**：
```bash
python evaluate_rex_classify.py \
    --pred_file ../../log/esconv_strategy/test_pred.json \
    --gold_file ../../log/esconv_strategy/test_gold.json \
    --output ../../log/esconv_strategy/evaluation_report.txt
```

**输出内容**：
- 整体指标（Precision, Recall, F1, Accuracy）
- 每个策略类别的详细指标
- 混淆矩阵
- 错误分析示例

---

### stat.py
**功能**：统计 ESConv 数据集的基本信息

**使用方法**：
```bash
python stat.py
```

**统计内容**：
- 对话会话总数、对话轮次总数
- 平均每个会话的轮次数
- 情绪类型分布
- 问题类型分布
- 经验类型分布

---

## 2. Synthesize/ - 数据合成

### synthesize_situations.py
**功能**：从 SoulChatCorpus 数据集中抽样并生成情境描述（situation）

**使用方法**：
```bash
# 只统计分析，不调用 API
python synthesize_situations.py --stats-only

# 完整运行
python synthesize_situations.py \
    --api-key YOUR_DEEPSEEK_KEY \
    --sample-size 5000 \
    --concurrency 10
```

**参数说明**：
- `--api-key`: DeepSeek API Key（或设置 DEEPSEEK_API_KEY 环境变量）
- `--base-url`: API Base URL（默认 https://api.deepseek.com）
- `--model`: 模型名称（默认 deepseek-chat）
- `--sample-size`: 抽样数量（默认 5000）
- `--seed`: 随机种子（默认 42）
- `--concurrency`: API 并发数（默认 10）
- `--stats-only`: 只进行统计分析

**输出**：`Datasets/SynthesizedSituations/situations.json`

---

### synthesize_dialogs.py
**功能**：使用 DeepSeek API 基于 situation 和不同人格生成训练对话数据

**使用方法**：
```bash
# 测试模式（只生成第一个 situation 的第一个人格）
python synthesize_dialogs.py --test --api-key YOUR_KEY

# 正式运行（支持断点续传）
python synthesize_dialogs.py --api-key YOUR_KEY --concurrency 5

# 禁用策略验证
python synthesize_dialogs.py --api-key YOUR_KEY --no-validate
```

**参数说明**：
- `--api-key`: DeepSeek API Key
- `--base-url`: API Base URL
- `--model`: 模型名称（默认 deepseek-chat）
- `--concurrency`: 并发数（默认 5）
- `--test`: 测试模式
- `--no-validate`: 禁用策略验证
- `--device`: 分类器设备

**特性**：
- 断点续传：进度保存在 `progress.json`
- 策略验证：使用 RexUniNLU 分类器验证生成的策略标签
- 自动重试：策略不匹配时自动重新生成

**输出**：`Datasets/SynthesizedDialogs/`

---

### convert_to_sharegpt.py
**功能**：将合成的对话数据转换为 ShareGPT 格式，用于 LLM 微调

**使用方法**：
```bash
python convert_to_sharegpt.py \
    --input path/to/all_dialogs.json \
    --output path/to/output.json \
    --system-prompt path/to/SystemPrompt.txt
```

**ShareGPT 格式说明**：
- 每个对话包含 `conversations` 数组和 `system` 字段
- `from` 字段：`"human"` (seeker) 或 `"gpt"` (supporter)
- `value` 字段：对话内容

---

## 3. rex/ - RexUniNLU 策略分类模型

基于 RexUniNLU 架构的策略分类模型，用于识别心理咨询对话中支持者使用的策略类型。

### 目录结构
```
rex/
├── main.py           # 训练/评估入口
├── arguments.py      # 参数定义
├── config.ini        # 默认配置
├── config_esconv.ini # ESConv 策略分类配置
├── data/             # 训练数据
├── data_utils/       # 数据加载工具
├── model/            # 模型定义
├── Trainer/          # 训练器
└── scripts/          # 训练/评估脚本
```

### 训练模型
```bash
cd rex
bash scripts/finetune.sh
```

### 评估模型
```bash
cd rex
bash scripts/eval.sh
```

### 配置文件说明 (config_esconv.ini)
```ini
[default]
mode = finetune       # 运行模式
task_name = esconv_strategy  # 任务名称
bert_model_dir = ./models/esconv_strategy  # 模型目录
data_name = esconv_strategy  # 数据集名称
...
```

---

## 依赖安装

```bash
# 安装基础依赖
pip install openai tqdm torch transformers safetensors

# 如需使用评估报告功能
pip install scikit-learn numpy
```

## 环境变量

```bash
export DASHSCOPE_API_KEY="your-dashscope-api-key"
export DEEPSEEK_API_KEY="your-deepseek-api-key"
```
