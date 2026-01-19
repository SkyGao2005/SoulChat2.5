# LibreChat 配置说明

本目录包含 SoulChat2.5 使用的 LibreChat 配置文件。LibreChat 是一个开源的 AI 聊天界面，本项目使用它作为心理咨询对话系统的前端。

## 文件说明

- `librechat.yaml` - LibreChat 主配置文件
- `docker-compose.override.yml` - Docker Compose 覆盖配置

---

## librechat.yaml 配置详解

### 1. 界面配置 (interface)

```yaml
interface:
  customWelcome: 'Welcome to SoulChat! Enjoy your experience.'  # 欢迎语
  endpointsMenu: true    # 显示端点菜单
  modelSelect: true      # 允许选择模型
  parameters: true       # 显示参数设置
  sidePanel: true        # 显示侧边栏
  presets: true          # 允许预设
  prompts: true          # 显示提示词
  bookmarks: true        # 书签功能
  multiConvo: true       # 多对话支持
  agents: true           # Agent 功能
```

**可修改项**：
- `customWelcome`: 修改为您想显示的欢迎语
- 各功能开关可根据需要设置为 `true` 或 `false`

---

### 2. 允许的域名 (actions.allowedDomains)

```yaml
actions:
  allowedDomains:
    - 'swapi.dev'
    - 'librechat.ai'
    - 'google.com'
    - "localhost"
    - "127.0.0.1"
    - "host.docker.internal"
    - "u508160-r3wv-0866cf91.westd.seetacloud.com"  # vLLM 服务器地址
```

**可修改项**：
- 如果您的 vLLM 服务器部署在不同地址，需要修改最后一行的域名
- 添加其他需要访问的 API 域名

---

### 3. 自定义端点 (endpoints.custom)

```yaml
endpoints:
  custom:
    - name: "SoulChat-vLLM"
      apiKey: "${SOULCHAT_VLLM_API_KEY}"    # 从环境变量读取
      baseURL: "${SOULCHAT_VLLM_BASE_URL}"  # 从环境变量读取
      models:
        default: ["qwen3-14b-soulchat"]     # 默认模型列表
        fetch: false                         # 不从 API 获取模型列表
      customParams:
        defaultParamsEndpoint: "openAI"
        paramDefinitions:
          - key: temperature
            default: 0.7                     # 默认温度
          - key: top_p
            default: 0.8                     # 默认 top_p
      addParams:
        top_k: 20                            # 额外参数 top_k
        min_p: 0                             # 额外参数 min_p
```

**可修改项**：
- `name`: 端点显示名称
- `apiKey`: API 密钥（建议通过环境变量设置）
- `baseURL`: vLLM 服务器地址（建议通过环境变量设置）
- `models.default`: 可用的模型列表
- `paramDefinitions`: 调整 `temperature`、`top_p` 的默认值
- `addParams`: 调整 `top_k`、`min_p` 等参数

---

### 4. 记忆功能 (memory)

```yaml
memory:
  disabled: false              # 启用记忆功能
  personalize: true            # 允许用户个性化设置
  messageWindowSize: 8         # 用于写入记忆的消息窗口大小
  tokenLimit: 1800             # Token 限制
  charLimit: 10000             # 字符限制
  validKeys:                   # 允许存储的记忆类型
    - user_profile             # 用户画像
    - preferences              # 偏好设置
    - goals                    # 目标
    - ongoing_context          # 持续上下文
    - coping_strategies        # 应对策略
    - safety_notes             # 安全备注
```

**可修改项**：
- `disabled`: 设为 `true` 可完全禁用记忆功能
- `messageWindowSize`: 调整参与记忆写入的消息数量
- `tokenLimit`/`charLimit`: 调整记忆存储限制
- `validKeys`: 添加或移除记忆槽位类型

#### Memory Agent 配置

```yaml
  agent:
    provider: "SoulChat-vLLM"
    model: "qwen3-14b-soulchat"
    instructions: >
      你是 SoulChat 的 Memory Agent...
    model_parameters:
      temperature: 0.7
      max_tokens: 1000
```

**可修改项**：
- `provider`/`model`: 切换使用不同的模型
- `instructions`: 修改记忆代理的指令（影响记忆如何被总结和存储）
- `temperature`: 调整生成的随机性

---

### 5. 模型规格 (modelSpecs)

```yaml
modelSpecs:
  enforce: true          # 强制用户只能使用定义的规格
  prioritize: true       # 优先使用定义的规格
  list:
    - name: "soulchat-default"
      label: "SoulChat 咨询师"
      default: true
      preset:
        endpoint: "SoulChat-vLLM"
        model: "qwen3-14b-soulchat"
        greeting: |
          你好，我在这儿陪你聊聊。你想先从哪件事开始？
        promptPrefix: |
          你是一位温暖、平易近人、不过度专业化的中文心理支持型咨询师...
```

**可修改项**：
- `label`: 修改显示给用户的名称
- `greeting`: 修改对话开始时的问候语
- `promptPrefix`: 修改系统提示词
---

## 快速开始

### 1. 设置环境变量

创建 `.env` 文件（在 LibreChat 根目录）：

```bash
# vLLM 服务配置
SOULCHAT_VLLM_API_KEY=your-api-key
SOULCHAT_VLLM_BASE_URL=http://your-server:8000/v1

# MongoDB 配置
MONGO_URI=mongodb://mongodb:27017/librechat

# 其他必要配置...
```

### 2. 启动服务

```bash
# 拉取镜像并启动
docker compose up -d

# 查看日志
docker compose logs -f api
```

### 3. 访问界面

打开浏览器访问 `http://localhost:3080`

---

## 参考资料

- [LibreChat 官方文档](https://www.librechat.ai/docs)
- [LibreChat 配置指南](https://www.librechat.ai/docs/configuration/librechat_yaml)
- [Docker Compose Override](https://www.librechat.ai/docs/configuration/docker_override)
