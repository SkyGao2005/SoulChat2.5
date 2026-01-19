# SoulChat2.5

SoulChat2.5 是一个中文心理支持对话系统，基于 ESConv 数据集和 LLM 技术构建。项目包含数据处理、对话合成、策略分类等完整的工作流程。

## 项目结构

```
SoulChat2.5/
├── src/                    # 源代码
│   ├── DatasetProcess/     # 数据集处理脚本
│   ├── Synthesize/         # 对话数据合成脚本
│   └── rex/                # RexUniNLU 策略分类模型
├── vllm/                   # vLLM 后端服务
│   ├── start_vllm.py       # 启动脚本
│   ├── qwen3_nonthinking.jinja  # 对话模板
│   └── README.md           # 配置说明
├── Prompts/                # 提示词模板
│   ├── Patients/           # 来访者人格模板 (1-6.txt)
│   ├── SynthesizePropmt.txt   # 对话合成提示词
│   └── SystemPropmt.txt    # 系统提示词
├── LibreChat/              # 聊天界面配置
│   ├── librechat.yaml      # 主配置文件
│   ├── docker-compose.override.yml
│   └── README.md           # 配置说明
├── Datasets/               # 数据集
│   ├── ESConv.json         # 原始英文数据集
│   ├── ESConv_zh.json      # 翻译后的中文数据集
│   ├── SynthesizedSituations/  # 合成的情境数据
│   └── *.txt               # 翻译提示词
└── README.md               # 本文件
```

## 功能特性

### 1. 数据处理
- **ESConv 数据集翻译**：将英文心理对话数据翻译为中文
- **策略分类**：识别支持者使用的 8 种心理咨询策略
- **数据格式转换**：支持转换为 RexUniNLU 和 ShareGPT 格式

### 2. 对话合成
- **情境生成**：从现有对话中提取情境描述
- **对话生成**：基于情境和人格模板生成多样化对话
- **策略验证**：使用分类器验证生成对话的策略标签

### 3. 策略分类模型 (RexUniNLU)
支持识别的 8 种策略：
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

### 4. 对话系统界面
- 基于 LibreChat 构建的 Web 界面
- 支持持久化记忆功能
- 可配置的系统提示词和模型参数

## 快速开始

### 环境准备

```bash
# 安装依赖
pip install openai tqdm torch transformers safetensors scikit-learn numpy

# 设置环境变量
export DEEPSEEK_API_KEY="your-api-key"
export DASHSCOPE_API_KEY="your-api-key"
```

### 数据处理流程

```bash
cd src/DatasetProcess

# 1. 翻译 ESConv 数据集
python translate.py

# 2. 翻译元数据
python translate_metadata.py

# 3. 准备策略分类训练数据
python prepare_rex_classify.py
```

### 对话合成流程

```bash
cd src/Synthesize

# 1. 生成情境描述
python synthesize_situations.py --api-key YOUR_KEY

# 2. 合成对话数据
python synthesize_dialogs.py --api-key YOUR_KEY

# 3. 转换为 ShareGPT 格式
python convert_to_sharegpt.py
```

### 启动后端服务 (vLLM)

vLLM 提供高性能的 LLM 推理服务，是对话系统的后端。

#### 1. 安装 vLLM

```bash
pip install vllm
```

#### 2. 配置模型路径

编辑 `vllm/start_vllm.py`，修改以下配置：

```python
MODEL_PATH = "./models/qwen3-14b-qlora-soulchat"  # 修改为您的模型路径
SERVED_NAME = "qwen3-14b-soulchat"                # 服务名称
HOST = "0.0.0.0"                                   # 监听地址
PORT = "6006"                                      # 监听端口
API_KEY = "sk-local-change-me"                    # API 密钥（建议修改）
```

#### 3. 启动服务

```bash
cd vllm
python start_vllm.py
```

服务启动后，API 端点为：`http://localhost:6006/v1`

#### 4. 验证服务

```bash
curl http://localhost:6006/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-local-change-me" \
  -d '{
    "model": "qwen3-14b-soulchat",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

详细配置说明请参考 [vllm/README.md](vllm/README.md)

---

### 启动前端界面 (LibreChat)

LibreChat 提供 Web 聊天界面，连接 vLLM 后端服务。

```bash
cd LibreChat

# 设置环境变量（与 vLLM 配置对应）
export SOULCHAT_VLLM_API_KEY="sk-local-change-me"
export SOULCHAT_VLLM_BASE_URL="http://localhost:6006/v1"

# 启动服务
docker compose up -d
```

访问 `http://localhost:3080` 开始使用。

## 详细文档

- [src/README.md](src/README.md) - 源码使用说明
- [vllm/README.md](vllm/README.md) - vLLM 后端配置说明
- [LibreChat/README.md](LibreChat/README.md) - 前端界面配置说明

## 数据集说明

### ESConv
ESConv (Emotional Support Conversation) 是一个英文情感支持对话数据集，包含约 1,300 个对话，每个对话都标注了支持者使用的策略。

### SynthesizedSituations
从 SoulChatCorpus 数据集中抽样并生成的情境描述，用于后续对话合成。

## 技术栈

- **推理后端**：vLLM, Python
- **LLM**：Qwen3-14B (LoRA 微调)
- **分类模型**：RexUniNLU, PyTorch, Transformers
- **前端界面**：LibreChat, Docker
- **数据合成**：DeepSeek API

## 参考文献

- ESConv: [Towards Emotional Support Dialog Systems](https://arxiv.org/abs/2106.01144)
- SoulChat: [SoulChat 对话系统](https://github.com/scutcyr/SoulChat)
- RexUniNLU: 统一信息抽取模型

## License

MIT License
