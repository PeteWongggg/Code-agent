# LLM API 基类库

这是一个标准 OpenAI 格式的 LLM API 基类库，提供统一的接口来访问各种大语言模型服务。

## 文件结构

```
llm_api/
├── base_client.py          # 基类和数据结构定义
├── clients/                # 各厂商客户端实现
│   ├── __init__.py         # 客户端模块导入
│   ├── openai/             # OpenAI 客户端
│   │   ├── __init__.py
│   │   └── openai_client.py
│   ├── anthropic/          # Anthropic Claude 客户端
│   │   ├── __init__.py
│   │   └── anthropic_client.py
│   ├── deepseek/           # DeepSeek 客户端
│   │   ├── __init__.py
│   │   └── deepseek_client.py
│   ├── openrouter/         # OpenRouter 多模型聚合客户端
│   │   ├── __init__.py
│   │   └── openrouter_client.py
│   └── private/            # 私有化部署模型客户端
│       ├── __init__.py
│       └── private_client.py
├── api_tests.py            # API 测试和使用示例
└── README.md              # 说明文档

注：依赖包列表 requirements.txt 已移至仓库根目录统一管理
```

## 特性

- ✅ **标准 OpenAI 格式**: 完全兼容 OpenAI API 规范
- ✅ **多厂商支持**: 支持 OpenAI、Anthropic、DeepSeek、OpenRouter、私有化部署等
- ✅ **统一接口**: 所有客户端使用相同的接口，便于切换
- ✅ **同步/异步**: 同时支持同步和异步操作
- ✅ **流式响应**: 支持实时流式输出
- ✅ **工具调用**: 支持 Function Calling 功能
- ✅ **错误处理**: 内置重试机制和错误处理
- ✅ **类型安全**: 完整的类型注解支持
- ✅ **易于扩展**: 基于抽象基类的设计，便于扩展新的提供商

## 快速开始

### 安装依赖

```bash
# 从仓库根目录安装依赖
pip install -r requirements.txt
```

### 基本使用

```python
from llm_api import OpenAIClient, ChatMessage, MessageRole

# 创建客户端
client = OpenAIClient(api_key="your-api-key")

# 创建消息
messages = [
    ChatMessage(role=MessageRole.SYSTEM, content="你是一个有用的AI助手。"),
    ChatMessage(role=MessageRole.USER, content="你好，世界！")
]

# 发送请求
request = client.create_request(messages=messages, model="gpt-3.5-turbo")
response = client.chat_completions_create(request)

print(response.choices[0].message.content)
```

### 流式响应

```python
# 启用流式响应
request = client.create_request(
    messages=messages, 
    model="gpt-3.5-turbo",
    stream=True
)

for chunk in client.chat_completions_create(request):
    if chunk.choices and chunk.choices[0].get("delta", {}).get("content"):
        content = chunk.choices[0]["delta"]["content"]
        print(content, end="", flush=True)
```

### 异步使用

```python
import asyncio

async def async_example():
    async with OpenAIClient(api_key="your-api-key") as client:
        request = client.create_request(messages=messages, model="gpt-3.5-turbo")
        response = await client.achat_completions_create(request)
        print(response.choices[0].message.content)

asyncio.run(async_example())
```

### 工具调用

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"}
                },
                "required": ["city"]
            }
        }
    }
]

request = client.create_request(
    messages=messages,
    model="gpt-3.5-turbo",
    tools=tools,
    tool_choice="auto"
)

response = client.chat_completions_create(request)
```

## 支持的提供商

### OpenAI 官方 API

```python
from llm_api import OpenAIClient

client = OpenAIClient(
    api_key="your-openai-api-key",
    base_url="https://api.openai.com/v1"  # 默认值
)
```

### Anthropic Claude

```python
from llm_api import AnthropicClient

client = AnthropicClient(
    api_key="your-anthropic-api-key",
    base_url="https://api.anthropic.com"  # 默认值
)
```

### DeepSeek

```python
from llm_api import DeepSeekClient

client = DeepSeekClient(
    api_key="your-deepseek-api-key",
    base_url="https://api.deepseek.com/v1"  # 默认值
)
```

### OpenRouter

```python
from llm_api import OpenRouterClient

client = OpenRouterClient(
    api_key="your-openrouter-api-key",
    base_url="https://openrouter.ai/api/v1",  # 默认值
    app_name="your-app-name",  # 可选
    site_url="https://your-site.com"  # 可选
)

# 使用不同的模型
models = [
    "openai/gpt-4-turbo-preview",
    "anthropic/claude-3-opus", 
    "google/gemini-pro",
    "meta-llama/llama-2-70b-chat"
]
```

### 私有化部署模型

```python
from llm_api import PrivateModelClient

# vLLM 部署
client = PrivateModelClient(
    api_key="EMPTY",  # 某些部署不需要 API key
    base_url="http://localhost:8000/v1",
    deployment_type="vllm"
)

# Ollama 部署
client = PrivateModelClient(
    base_url="http://localhost:11434",
    deployment_type="ollama"
)

# Text Generation Inference (TGI)
client = PrivateModelClient(
    base_url="http://localhost:8080",
    deployment_type="tgi"
)
```

### Azure OpenAI

```python
from llm_api import OpenAIClient

client = OpenAIClient(
    api_key="your-azure-api-key",
    base_url="https://your-resource.openai.azure.com/openai/deployments/your-deployment"
)
```

## API 参考

### 核心类

#### `BaseLLMAPI`
抽象基类，定义了 LLM API 的标准接口。

#### `OpenAIClient`
OpenAI API 的具体实现，支持官方 API 和兼容服务。

#### `ChatMessage`
聊天消息数据类：
- `role`: 消息角色（system/user/assistant/tool）
- `content`: 消息内容
- `tool_calls`: 工具调用（可选）
- `tool_call_id`: 工具调用 ID（可选）

#### `ChatCompletionRequest`
聊天补全请求参数：
- `messages`: 消息列表
- `model`: 模型名称
- `temperature`: 温度参数（0-2）
- `max_tokens`: 最大 token 数
- `stream`: 是否启用流式响应
- `tools`: 可用工具列表
- 等等...

#### `ChatCompletionResponse`
聊天补全响应：
- `choices`: 响应选择列表
- `usage`: Token 使用统计
- `model`: 使用的模型
- 等等...

### 配置选项

```python
client = OpenAIClient(
    api_key="your-api-key",
    base_url="https://api.openai.com/v1",
    timeout=60,           # 请求超时时间（秒）
    max_retries=3,        # 最大重试次数
    retry_delay=1.0,      # 重试延迟时间（秒）
    organization="org-id" # 组织 ID（OpenAI 专用）
)
```

## 错误处理

库内置了完善的错误处理机制：

- **自动重试**: 网络错误和临时故障会自动重试
- **指数退避**: 重试间隔采用指数退避策略
- **详细日志**: 记录请求和错误信息
- **异常传播**: 最终失败时会抛出原始异常

```python
try:
    response = client.chat_completions_create(request)
except Exception as e:
    print(f"请求失败: {e}")
```

## 客户端切换示例

由于所有客户端都实现了相同的接口，可以很容易地在不同提供商之间切换：

```python
from llm_api import OpenAIClient, AnthropicClient, DeepSeekClient, ChatMessage, MessageRole

# 定义通用的聊天函数
def chat_with_llm(client, user_message: str):
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="你是一个有用的AI助手。"),
        ChatMessage(role=MessageRole.USER, content=user_message)
    ]
    
    request = client.create_request(messages=messages, model="gpt-3.5-turbo")
    response = client.chat_completions_create(request)
    
    return response.choices[0].message.content

# 可以轻松切换不同的客户端
clients = {
    "openai": OpenAIClient(api_key="your-openai-key"),
    "anthropic": AnthropicClient(api_key="your-anthropic-key"), 
    "deepseek": DeepSeekClient(api_key="your-deepseek-key")
}

user_input = "解释一下什么是机器学习"

for provider, client in clients.items():
    print(f"\n{provider.upper()} 的回答:")
    try:
        answer = chat_with_llm(client, user_input)
        print(answer)
    except Exception as e:
        print(f"错误: {e}")
```

## 扩展新的提供商

要支持新的 LLM 提供商，只需继承 `BaseLLMAPI` 并实现抽象方法：

```python
from llm_api import BaseLLMAPI, ChatCompletionRequest, ChatCompletionResponse

class CustomLLMClient(BaseLLMAPI):
    def _initialize_client(self):
        # 初始化客户端连接
        pass
    
    def chat_completions_create(self, request: ChatCompletionRequest):
        # 实现同步聊天补全
        pass
    
    async def achat_completions_create(self, request: ChatCompletionRequest):
        # 实现异步聊天补全
        pass
```

### 实际扩展示例

```python
# 添加到 clients/custom/ 目录
class CustomLLMClient(BaseLLMAPI):
    def _initialize_client(self):
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
    
    def chat_completions_create(self, request):
        payload = self._build_custom_payload(request)
        response = self.client.post("/chat", json=payload)
        return self._parse_custom_response(response.json())
    
    def _build_custom_payload(self, request):
        # 转换为自定义 API 格式
        return {
            "model": request.model,
            "messages": self.format_messages_for_api(request.messages),
            # ... 其他参数转换
        }
    
    def _parse_custom_response(self, response_data):
        # 转换为标准 OpenAI 格式
        return ChatCompletionResponse(...)
```

## 示例代码

查看 `api_tests.py` 文件获取更多使用示例：

```bash
# 从项目根目录执行
python3 src/managers/llm_api/api_tests.py
```

### 环境配置

1. 复制环境变量模板：
```bash
cp env.example .env
```

2. 编辑 `.env` 文件，填入你的真实 API 密钥：
```bash
# 编辑 .env 文件
OPENAI_API_KEY=your-real-openai-key
ANTHROPIC_API_KEY=your-real-anthropic-key
DEEPSEEK_API_KEY=your-real-deepseek-key
OPENROUTER_API_KEY=your-real-openrouter-key
```

## 依赖

项目依赖已统一管理在仓库根目录的 `requirements.txt` 文件中：

- `httpx>=0.25.0`: HTTP 客户端
- `pydantic>=2.0.0`: 数据验证（可选）
- `loguru>=0.7.0`: 日志记录（可选）
- `typing-extensions>=4.5.0`: 类型支持

## 许可证

本项目采用 MIT 许可证。
