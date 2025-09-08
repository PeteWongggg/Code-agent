"""
LLM API 管理模块

提供标准 OpenAI 格式的 LLM API 接口，支持：
- 统一的聊天补全接口
- 同步和异步操作
- 流式响应
- 工具调用
- 错误处理和重试机制

支持的提供商：
- OpenAI: 官方 OpenAI API 和兼容服务
- Anthropic: Claude 系列模型
- DeepSeek: DeepSeek 系列模型
- Private: 私有化部署模型（vLLM、TGI、Ollama 等）

使用示例:
    # 方式1: 使用统一管理器（推荐）
    from llm_api import LLMAPIManager
    
    # 创建管理器
    manager = LLMAPIManager(
        client_name="openai",
        model_name="gpt-3.5-turbo",
        stream=False
    )
    
    # 发送消息
    response = manager.chat("你好，世界！")
    print(response)
    
    # 方式2: 直接使用客户端
    from llm_api import OpenAIClient, ChatMessage, MessageRole
    
    # 创建客户端
    client = OpenAIClient(api_key="your-api-key")
    
    # 创建消息
    messages = [
        ChatMessage(role=MessageRole.USER, content="你好，世界！")
    ]
    
    # 发送请求
    request = client.create_request(messages=messages, model="gpt-3.5-turbo")
    response = client.chat_completions_create(request)
    
    print(response.choices[0].message.content)
"""

# 基类和数据类
from src.managers.llm_api.base_client import (
    BaseLLMAPI,
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    MessageRole,
    Choice,
    Usage
)

# 客户端实现
from src.managers.llm_api.clients.openai.openai_client import OpenAIClient
from src.managers.llm_api.clients.anthropic.anthropic_client import AnthropicClient
from src.managers.llm_api.clients.deepseek.deepseek_client import DeepSeekClient
from src.managers.llm_api.clients.openrouter.openrouter_client import OpenRouterClient
from src.managers.llm_api.clients.private.private_client import PrivateModelClient

# API 管理器
from src.managers.llm_api.api_manager import (
    LLMAPIManager,
    create_manager,
    create_common_manager,
    COMMON_CONFIGS
)

__all__ = [
    # 基类和数据类
    "BaseLLMAPI",
    "ChatMessage",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatCompletionChunk",
    "MessageRole",
    "Choice",
    "Usage",
    
    # 客户端实现
    "OpenAIClient",
    "AnthropicClient",
    "DeepSeekClient",
    "OpenRouterClient",
    "PrivateModelClient",
    
    # API 管理器
    "LLMAPIManager",
    "create_manager",
    "create_common_manager",
    "COMMON_CONFIGS",
]

__version__ = "1.0.0"
__author__ = "Tokfinity Team"
__description__ = "标准 OpenAI 格式的 LLM API 基类库"
