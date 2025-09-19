import asyncio
import json
import sys
import os
from typing import List
from dotenv import load_dotenv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

env_path = os.path.join(project_root, '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
    print(f"✅ 已加载环境变量文件: {env_path}")
else:
    print(f"⚠️  未找到 .env 文件: {env_path}")
    print("请复制 env.example 为 .env 并配置你的 API 密钥")

from src.managers.llm_api.clients.openai.openai_client import OpenAIClient
from src.managers.llm_api.clients.anthropic.anthropic_client import AnthropicClient
from src.managers.llm_api.clients.deepseek.deepseek_client import DeepSeekClient
from src.managers.llm_api.clients.openrouter.openrouter_client import OpenRouterClient
from src.managers.llm_api.clients.private.private_client import PrivateModelClient
from src.managers.llm_api.base_client import ChatMessage, MessageRole, ChatCompletionRequest

API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY", "your-openai-api-key-here"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY", "your-anthropic-api-key-here"),
    "deepseek": os.getenv("DEEPSEEK_API_KEY", "your-deepseek-api-key-here"), 
    "openrouter": os.getenv("OPENROUTER_API_KEY", "your-openrouter-api-key-here"),
    "private": os.getenv("PRIVATE_API_KEY", "EMPTY")
}

BASE_URLS = {
    "openai": os.getenv("OPENAI_BASE_URL"), 
    "anthropic": os.getenv("ANTHROPIC_BASE_URL"),
    "deepseek": os.getenv("DEEPSEEK_BASE_URL"),
    "openrouter": os.getenv("OPENROUTER_BASE_URL"),
    "private": os.getenv("PRIVATE_URL", "http://localhost:8000/v1")
}

SPECIAL_CONFIGS = {
    "openrouter": {
        "app_name": os.getenv("OPENROUTER_APP_NAME", "tokfinity-llm-client"),
        "site_url": os.getenv("OPENROUTER_SITE_URL", "https://github.com/your-repo")
    },
    "private": {
        "deployment_type": os.getenv("PRIVATE_DEPLOYMENT_TYPE", "vllm"),
        "model_name": os.getenv("PRIVATE_MODEL_NAME", "your-private-model")
    }
}

AVAILABLE_CLIENTS = {
    "openai": {
        "name": "OpenAI",
        "description": "OpenAI 官方 API，支持 GPT 系列模型",
        "client_class": OpenAIClient,
        "models": [
            "gpt-4-turbo-preview",
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
        ],
        "default_model": "gpt-3.5-turbo"
    },
    "anthropic": {
        "name": "Anthropic Claude",
        "description": "Anthropic 的 Claude 系列模型",
        "client_class": AnthropicClient,
        "models": [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229", 
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0"
        ],
        "default_model": "claude-3-haiku-20240307"
    },
    "deepseek": {
        "name": "DeepSeek",
        "description": "DeepSeek 系列模型，专注于代码和推理",
        "client_class": DeepSeekClient,
        "models": [
            "deepseek-chat",
            "deepseek-coder",
            "deepseek-math"
        ],
        "default_model": "deepseek-chat"
    },
    "openrouter": {
        "name": "OpenRouter",
        "description": "多模型聚合服务，支持多家厂商的模型",
        "client_class": OpenRouterClient,
        "models": [
            "openai/gpt-4-turbo-preview",
            "openai/gpt-3.5-turbo",
            "anthropic/claude-3-opus",
            "anthropic/claude-3-sonnet",
            "anthropic/claude-3-haiku",
            "google/gemini-pro",
            "meta-llama/llama-2-70b-chat",
            "mistralai/mixtral-8x7b-instruct"
        ],
        "default_model": "openai/gpt-3.5-turbo"
    },
    "private": {
        "name": "私有化部署",
        "description": "私有化部署的模型（vLLM、TGI、Ollama等）",
        "client_class": PrivateModelClient,
        "models": [
            os.getenv("PRIVATE_MODEL_NAME", "your-custom-model"),
            "llama-2-7b-chat",
            "qwen-2.5-coder",
            "deepseek-coder",
            "codellama"
        ],
        "default_model": os.getenv("PRIVATE_MODEL_NAME", "qwen-2.5-coder")
    }
}

def create_client(client_type: str, model: str = None, api_key: str = None):
    if client_type not in AVAILABLE_CLIENTS:
        raise ValueError(f"不支持的客户端类型: {client_type}")
    
    config = AVAILABLE_CLIENTS[client_type]
    
    # 使用传入的API密钥或配置中的密钥
    if api_key is None:
        api_key = API_KEYS.get(client_type)
    
    base_url = BASE_URLS.get(client_type)
    
    if not api_key or api_key.startswith('your-'):
        print(f"⚠️  警告: {config['name']} 的 API 密钥未配置，请求可能会失败")
    
    # 创建客户端
    client_kwargs = {
        "api_key": api_key,
        "timeout": 30
    }
    
    if base_url:
        client_kwargs["base_url"] = base_url
    
    # 特殊处理不同客户端的参数
    if client_type == "openrouter":
        openrouter_config = SPECIAL_CONFIGS.get("openrouter", {})
        client_kwargs["app_name"] = openrouter_config.get("app_name", "tokfinity-examples")
        client_kwargs["site_url"] = openrouter_config.get("site_url")
    elif client_type == "private":
        private_config = SPECIAL_CONFIGS.get("private", {})
        client_kwargs["deployment_type"] = private_config.get("deployment_type", "vllm")
        if not base_url:
            client_kwargs["base_url"] = "http://localhost:8000/v1"
    
    return config["client_class"](**client_kwargs)


def test_client(client_type: str, model: str = None, test_message: str = "你好，请简单介绍一下你自己。", api_key: str = None, use_stream=False, tools=None, tool_choice=None):
    config = AVAILABLE_CLIENTS.get(client_type)

    if client_type == "private":
        private_url = os.getenv("PRIVATE_URL", "http://localhost:8000/v1")
        deployment_type = os.getenv("PRIVATE_DEPLOYMENT_TYPE", "vllm")
        print(f"🏠 私有化部署信息:")
        print(f"   服务地址: {private_url}")
        print(f"   部署类型: {deployment_type}")
        print(f"   模型名称: {model}")
    
    client = create_client(client_type, model, api_key)        
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="你是一个有用的AI助手,需要通过搜索工具帮我解决问题。"),
        ChatMessage(role=MessageRole.USER, content=test_message)
    ]
    
    request = client.create_request(
        messages=messages,
        model=model,
        temperature=0.7,
        max_tokens=200,
        stream=use_stream,
        tools=tools,
    )
            
    response = client.chat_completions_create(request)

    print(f"   响应: {response}")
    
    msg = response.choices[0].message
    if getattr(msg, 'tool_calls', None):
        print(f"   检测到工具调用 tool_calls: {msg.tool_calls}")
    print(f"   内容: {msg.content}")
    
    if response.usage:
        print(f"   Token使用: {response.usage.total_tokens} "
                f"(输入: {response.usage.prompt_tokens}, "
                f"输出: {response.usage.completion_tokens})")
    
    return True


if __name__ == "__main__":
    model_map = {
        "openai": ["gpt-4o", "gpt-4o-mini"],
        "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
        "deepseek": ["deepseek-chat", "deepseek-reasoner"],
        "openrouter": ["openai/gpt-4o", "anthropic/claude-3-5-sonnet"],
        "private": [os.getenv("PRIVATE_MODEL_NAME", "qwen-2.5-coder")]
    }

    api_map = {
        "openai": os.getenv("OPENAI_API_KEY", "your-openai-api-key-here"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY", "your-anthropic-api-key-here"),
        "deepseek": os.getenv("DEEPSEEK_API_KEY", "your-deepseek-api-key-here"),
        "openrouter": os.getenv("OPENROUTER_API_KEY", "your-openrouter-api-key-here"),
        "private": os.getenv("PRIVATE_API_KEY", "EMPTY")
    }

    client_2_use = "openrouter"       
    api_key = api_map[client_2_use]   
    model = model_map[client_2_use][0]
    use_stream=False
    
    print(f"\n🔍 环境变量加载状态:")
    for client_id, api_key_value in api_map.items():
        status = "✅ 已配置" if api_key_value and not api_key_value.startswith('your-') else "❌ 需要配置"
        print(f"   {client_id.upper()}_API_KEY: {status}")
    
    
    print(f"\n🎮 当前测试配置:")
    print(f"   客户端: {client_2_use}")
    print(f"   模型: {model}")
    print(f"   可用模型: {model_map[client_2_use]}")
    print(f"   是否启用流式: {use_stream}")
    print(f"   API状态: {'✅ 已配置' if api_key and not api_key.startswith('your-') else '❌ 需要配置'}")
    
    if not api_key or api_key.startswith('your-'):
        print(f"\n⚠️  提示: 请在 .env 文件中配置 {client_2_use.upper()}_API_KEY")
        print(f"   1. 复制 env.example 为 .env")
        print(f"   2. 编辑 .env 文件中的 {client_2_use.upper()}_API_KEY")
    
    # 准备带 tools 的测试
    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "联网搜索最新的事实信息，并返回简要摘要与关键来源链接。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "要搜索的查询语句"},
                        "time_range": {"type": "string", "enum": ["day","week","month","year","all"], "description": "时间范围"}
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    fed_question = "美联储最新的新闻有啥？你需要通过联网搜索后回答我。"

    tools_to_use = tools
    tool_choice_to_use = "auto"

    test_client(
        client_type=client_2_use,
        model=model,
        test_message=fed_question,
        api_key=api_key,
        use_stream=use_stream,
        tools=tools_to_use,
        tool_choice=tool_choice_to_use
    )
