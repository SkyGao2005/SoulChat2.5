# vLLM 后端服务

本目录包含启动 vLLM 推理服务的脚本和配置文件。

## 文件说明

- `start_vllm.py` - vLLM 服务启动脚本
- `qwen3_nonthinking.jinja` - Qwen3 模型的对话模板（禁用思考模式）

## 配置说明

在 `start_vllm.py` 中修改以下配置：

```python
MODEL_PATH = "./models/qwen3-14b-qlora-soulchat"  # 模型路径
SERVED_NAME = "qwen3-14b-soulchat"                # 服务名称
HOST = "0.0.0.0"                                   # 监听地址
PORT = "6006"                                      # 监听端口
API_KEY = "sk-local-change-me"                    # API 密钥
TEMPLATE = "./qwen3_nonthinking.jinja"            # 对话模板路径
```

### 可调参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gpu-memory-utilization` | 0.97 | GPU 显存利用率 |
| `--max-num-seqs` | 1 | 最大并发序列数 |
| `--max-model-len` | 8192 | 最大上下文长度 |
| `--dtype` | auto | 数据类型（auto/float16/bfloat16） |

## 使用方法

### 1. 安装 vLLM

```bash
pip install vllm
```

### 2. 准备模型

确保模型文件已放置在 `MODEL_PATH` 指定的路径下。

### 3. 启动服务

```bash
cd vllm
python start_vllm.py
```

服务启动后，API 端点为：`http://HOST:PORT/v1`

### 4. 测试服务

```bash
curl http://localhost:6006/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-local-change-me" \
  -d '{
    "model": "qwen3-14b-soulchat",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

## 对话模板说明

`qwen3_nonthinking.jinja` 是为 Qwen3 模型定制的对话模板，特点：

- 兼容 OpenAI API 格式
- 支持工具调用（Tool Calling）
- 禁用思考模式（生成时直接输出，不显示 `<think>` 标签）

## 常见问题

### 显存不足

尝试调整以下参数：
```python
"--gpu-memory-utilization", "0.90",  # 降低显存利用率
"--max-model-len", "4096",           # 减少上下文长度
```

或启用 CPU 卸载（取消注释）：
```python
"--cpu-offload-gb", "8",
```

### 多 GPU 部署

添加张量并行参数：
```python
"--tensor-parallel-size", "2",  # 使用 2 张 GPU
```
