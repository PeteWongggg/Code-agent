"""
LLM API 测试工具

支持测试多种 LLM 客户端：
- OpenAI: GPT 系列模型
- Anthropic: Claude 系列模型
- DeepSeek: 代码和推理模型
- OpenRouter: 多模型聚合服务
- Private: 私有化部署模型

使用方法:
    python3 src/managers/llm_api/api_tests.py
"""

import asyncio
import json
import sys
import os
from typing import List
from dotenv import load_dotenv

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 加载环境变量
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

# =============================================================================
# 配置部分 - 在这里设置你的 API 密钥和模型选择
# =============================================================================

# API 密钥配置 - 从环境变量读取
API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY", "your-openai-api-key-here"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY", "your-anthropic-api-key-here"),
    "deepseek": os.getenv("DEEPSEEK_API_KEY", "your-deepseek-api-key-here"), 
    "openrouter": os.getenv("OPENROUTER_API_KEY", "your-openrouter-api-key-here"),
    "private": os.getenv("PRIVATE_API_KEY", "EMPTY")
}

# 基础 URL 配置 - 从环境变量读取或使用默认值
BASE_URLS = {
    "openai": os.getenv("OPENAI_BASE_URL"),  # 默认 https://api.openai.com/v1
    "anthropic": os.getenv("ANTHROPIC_BASE_URL"),  # 默认 https://api.anthropic.com
    "deepseek": os.getenv("DEEPSEEK_BASE_URL"),  # 默认 https://api.deepseek.com/v1
    "openrouter": os.getenv("OPENROUTER_BASE_URL"),  # 默认 https://openrouter.ai/api/v1
    "private": os.getenv("PRIVATE_URL", "http://localhost:8000/v1")  # 修正为 PRIVATE_URL
}

# 特殊配置参数
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

# 可用的客户端配置
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
    """
    创建指定类型的客户端
    
    Args:
        client_type: 客户端类型 (openai, anthropic, deepseek, openrouter, private)
        model: 模型名称，如果不指定则使用默认模型
        api_key: API密钥，如果不指定则使用配置中的密钥
    
    Returns:
        客户端实例
    """
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
        # 确保私有化部署使用正确的 base_url
        if not base_url:
            client_kwargs["base_url"] = "http://localhost:8000/v1"
    
    return config["client_class"](**client_kwargs)


def test_client(client_type: str, model: str = None, test_message: str = "你好，请简单介绍一下你自己。", api_key: str = None, use_stream=False, tools=None, tool_choice=None):
    """
    测试指定的客户端
    
    Args:
        client_type: 客户端类型
        model: 模型名称
        test_message: 测试消息
        api_key: API密钥
    """
    config = AVAILABLE_CLIENTS.get(client_type)
    if not config:
        print(f"❌ 不支持的客户端类型: {client_type}")
        return
    
    # 使用默认模型或指定模型
    if not model:
        model = config["default_model"]
    elif model not in config["models"]:
        print(f"⚠️  警告: 模型 {model} 不在推荐列表中，但仍会尝试使用")
    
    print(f"\n🧪 测试 {config['name']} - {model}")
    print("-" * 60)
    
    # 私有化部署的特殊处理
    if client_type == "private":
        private_url = os.getenv("PRIVATE_URL", "http://localhost:8000/v1")
        deployment_type = os.getenv("PRIVATE_DEPLOYMENT_TYPE", "vllm")
        print(f"🏠 私有化部署信息:")
        print(f"   服务地址: {private_url}")
        print(f"   部署类型: {deployment_type}")
        print(f"   模型名称: {model}")
    
    try:
        # 创建客户端
        client = create_client(client_type, model, api_key)
        print(f"✅ 客户端创建成功")
        
        # 创建消息
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="你是一个有用的AI助手,需要通过搜索工具帮我解决问题。"),
            ChatMessage(role=MessageRole.USER, content=test_message)
        ]
        
        # 创建请求
        request = client.create_request(
            messages=messages,
            model=model,
            temperature=0.7,
            max_tokens=200,
            stream=use_stream,
            tools=tools,
            #tool_choice=tool_choice
        )
        
        print(f"📤 发送请求...{request}")
        
        # 发送请求
        response = client.chat_completions_create(request)
        
        # 显示结果
        if use_stream:
            print(f"✅ 收到流式响应，开始处理...")
            content_parts = []
            chunk_count = 0
            
            for chunk in response:
                chunk_count += 1
                if chunk and chunk.choices:
                    delta = chunk.choices[0].get('delta', {})
                    if 'content' in delta and delta['content']:
                        content_parts.append(delta['content'])
            
            full_content = ''.join(content_parts)
            print(f"✅ 流式响应完成:")
            print(f"   总块数: {chunk_count}")
            print(f"   完整内容: {full_content}")
        else:
            print(f"✅ 收到响应: ")
            print(f"   模型: {response.model}")
            msg = response.choices[0].message
            # 打印工具调用或文本内容
            if getattr(msg, 'tool_calls', None):
                print(f"   检测到工具调用 tool_calls: {msg.tool_calls}")
            print(f"   内容: {msg.content}")
            
            if response.usage:
                print(f"   Token使用: {response.usage.total_tokens} "
                      f"(输入: {response.usage.prompt_tokens}, "
                      f"输出: {response.usage.completion_tokens})")
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ 测试失败: {error_msg}")
        
        # 特殊处理 502 错误
        if "502" in error_msg or "Bad Gateway" in error_msg:
            print("\n🔍 502 Bad Gateway 错误诊断:")
            print("📋 可能原因:")
            print("   1. vLLM 服务正在启动中 - 模型加载需要时间（通常几分钟）")
            print("   2. GPU 内存不足 - 模型太大无法加载到 GPU")
            print("   3. vLLM 服务崩溃 - 检查服务日志")
            print("   4. 代理服务器问题 - 如果使用了 nginx 等代理")
            print("   5. 端口冲突或网络问题")
            print("\n🛠️  建议解决方案:")
            print("   1. 等待 3-5 分钟后重试（模型加载时间）")
            print("   2. 检查 vLLM 服务日志: docker logs <container_name>")
            print("   3. 检查 GPU 内存使用: nvidia-smi")
            print("   4. 重启 vLLM 服务")
            print("   5. 确认模型路径和配置正确")
            
        return False
    
    finally:
        try:
            client.close()
        except:
            pass


async def test_client_async(client_type: str, model: str = None, test_message: str = "请用一句话解释什么是人工智能。", api_key: str = None):
    """
    异步测试指定的客户端
    
    Args:
        client_type: 客户端类型
        model: 模型名称
        test_message: 测试消息
        api_key: API密钥
    """
    config = AVAILABLE_CLIENTS.get(client_type)
    if not config:
        print(f"❌ 不支持的客户端类型: {client_type}")
        return
    
    if not model:
        model = config["default_model"]
    
    print(f"\n🚀 异步测试 {config['name']} - {model}")
    print("-" * 60)
    
    # 私有化部署的特殊处理
    if client_type == "private":
        private_url = os.getenv("PRIVATE_URL", "http://localhost:8000/v1")
        deployment_type = os.getenv("PRIVATE_DEPLOYMENT_TYPE", "vllm")
        print(f"🏠 私有化部署信息:")
        print(f"   服务地址: {private_url}")
        print(f"   部署类型: {deployment_type}")
        print(f"   模型名称: {model}")
    
    try:
        # 创建客户端
        client = create_client(client_type, model, api_key)
        
        # 创建消息
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="你是一个有用的AI助手。"),
            ChatMessage(role=MessageRole.USER, content=test_message)
        ]
        
        # 创建请求
        request = client.create_request(
            messages=messages,
            model=model,
            temperature=0.7,
            max_tokens=150
        )
        
        print(f"📤 发送异步请求...")
        
        # 发送异步请求
        response = await client.achat_completions_create(request)
        
        # 显示结果
        print(f"✅ 收到异步响应:")
        print(f"   模型: {response.model}")
        print(f"   内容: {response.choices[0].message.content}")
        
        if response.usage:
            print(f"   Token使用: {response.usage.total_tokens}")
        
        return True
        
    except Exception as e:
        print(f"❌ 异步测试失败: {e}")
        return False
    
    finally:
        try:
            await client.aclose()
        except:
            pass


def  main(client=None, model=None, api=None, test_clients=None, test_models=None, test_message=None, run_async=False, use_stream=False):
    """
    主测试函数
    
    Args:
        client: 单个客户端类型，例如 "openai"
        model: 单个模型名称，例如 "gpt-4o"
        api: 单个API密钥，例如 "sk-xxx"
        test_clients: 要测试的客户端列表，例如 ["openai", "deepseek"]
        test_models: 要测试的模型字典，例如 {"openai": "gpt-4", "deepseek": "deepseek-chat"}
        test_message: 自定义测试消息
        run_async: 是否运行异步测试
    
    Examples:
        # 简单模式：测试单个客户端
        main(client="openai", model="gpt-4o", api="sk-xxx")
        
        # 批量模式：测试多个客户端
        main(test_clients=["openai", "deepseek"])
        
        # 测试特定模型
        main(test_models={"openai": "gpt-4", "deepseek": "deepseek-coder"})
        
        # 自定义测试消息
        main(client="openai", test_message="请写一个Python函数计算斐波那契数列")
    """
    print("🚀 LLM API 客户端测试工具")
    print("=" * 80)
    
    # 处理单个客户端测试模式
    if client and model and api:
        print(f"\n🎯 单客户端测试模式")
        print(f"   客户端: {client}")
        print(f"   模型: {model}")
        print(f"   API密钥: {'✅ 已提供' if api and not api.startswith('your-') else '❌ 无效'}")
        
        # 临时更新API密钥
        original_api = API_KEYS.get(client)
        API_KEYS[client] = api
        
        try:
            # 测试单个客户端
            if not test_message:
                test_message = "你好，请简单介绍一下你自己，并说明你的主要功能。"
            
            print(f"📝 测试消息: {test_message}")
            print("\n" + "=" * 80)
            
            if run_async:
                success = asyncio.run(test_client_async(client, model, test_message, api))
            else:
                success = test_client(client, model, test_message, api, use_stream)
            
            print("\n" + "=" * 80)
            if success:
                print("🎉 单客户端测试成功！")
            else:
                print("❌ 单客户端测试失败！")
            
        finally:
            # 恢复原始API密钥
            if original_api:
                API_KEYS[client] = original_api
        
        return
    
    # 确定要测试的客户端（批量模式）
    if test_clients:
        clients_to_test = test_clients
    elif test_models:
        clients_to_test = list(test_models.keys())
    else:
        # 默认测试所有有API密钥的客户端
        clients_to_test = []
        for client_id, api_key in API_KEYS.items():
            if api_key and not api_key.startswith('your-'):
                clients_to_test.append(client_id)
        
        if not clients_to_test:
            print("\n⚠️  没有配置有效的API密钥，将测试所有客户端（可能会失败）")
            clients_to_test = list(AVAILABLE_CLIENTS.keys())
    
    print(f"\n🎯 将测试以下客户端: {', '.join(clients_to_test)}")
    
    # 默认测试消息
    if not test_message:
        test_message = "你好，请简单介绍一下你自己，并说明你的主要功能。"
    
    print(f"📝 测试消息: {test_message}")
    print("\n" + "=" * 80)
    
    # 运行测试
    success_count = 0
    total_count = len(clients_to_test)
    
    for client_type in clients_to_test:
        if client_type not in AVAILABLE_CLIENTS:
            print(f"❌ 跳过未知客户端: {client_type}")
            continue
        
        # 确定要使用的模型
        model = None
        if test_models and client_type in test_models:
            model = test_models[client_type]
        
        # 运行测试
        if run_async:
            success = asyncio.run(test_client_async(client_type, model, test_message))
        else:
            success = test_client(client_type, model, test_message)
        
        if success:
            success_count += 1
    
    # 显示总结
    print("\n" + "=" * 80)
    print(f"🏁 测试完成: {success_count}/{total_count} 个客户端测试成功")
    
    if success_count == 0:
        print("\n💡 提示:")
        print("1. 请确保在文件顶部的 API_KEYS 中配置了有效的API密钥")
        print("2. 检查网络连接是否正常")
        print("3. 确认API密钥有足够的余额和权限")
    elif success_count < total_count:
        print(f"\n⚠️  有 {total_count - success_count} 个客户端测试失败，请检查配置")
    else:
        print("\n🎉 所有测试都成功完成！")


if __name__ == "__main__":
    # =============================================================================
    # 🔧 配置区域 - 从环境变量和预定义列表中获取配置
    # =============================================================================
    
    # 模型映射（最新的模型列表）
    model_map = {
        "openai": ["gpt-4o", "gpt-4o-mini"],
        "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
        "deepseek": ["deepseek-chat", "deepseek-reasoner"],
        "openrouter": ["openai/gpt-4o", "anthropic/claude-3-5-sonnet"],
        "private": [os.getenv("PRIVATE_MODEL_NAME", "qwen-2.5-coder")]
    }

    # API 密钥映射（从环境变量读取）
    api_map = {
        "openai": os.getenv("OPENAI_API_KEY", "your-openai-api-key-here"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY", "your-anthropic-api-key-here"),
        "deepseek": os.getenv("DEEPSEEK_API_KEY", "your-deepseek-api-key-here"),
        "openrouter": os.getenv("OPENROUTER_API_KEY", "your-openrouter-api-key-here"),
        "private": os.getenv("PRIVATE_API_KEY", "EMPTY")
    }

    # 客户端列表
    client_list = ["openai", "anthropic", "deepseek", "openrouter", "private"]
    
    # =============================================================================
    # 🎯 测试配置 - 修改这里来选择要测试的客户端和模型
    # =============================================================================
    
    client_2_use = "private"           # 使用 OpenAI 做函数调用测试
    api_key = api_map[client_2_use]         # 自动从环境变量获取API密钥
    model = model_map[client_2_use][0] # 选择模型
    use_stream=False # 工具调用建议使用非流式，便于观察 tool_calls
    
    # =============================================================================
    # 🎯 主要测试区域 - 修改下面的参数来测试不同的配置
    # =============================================================================
    
    # 显示环境变量加载状态
    print(f"\n🔍 环境变量加载状态:")
    for client_id, api_key_value in api_map.items():
        status = "✅ 已配置" if api_key_value and not api_key_value.startswith('your-') else "❌ 需要配置"
        print(f"   {client_id.upper()}_API_KEY: {status}")
    
    
    # 当前测试配置
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

    # 执行测试（私有 vLLM 可能不支持 OpenAI-style tools，先尝试禁用 tools 以定位 400）
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
    
    
    print("\n" + "=" * 80)
    print("💡 快速使用指南:")
    print("1. 环境配置:")
    print("   - 复制 env.example 为 .env")
    print("   - 在 .env 文件中配置你的API密钥")
    print("2. 客户端选择:")
    print("   - client_2_use = client_list[0-4] 来切换客户端")
    print("   - 0:OpenAI, 1:Anthropic, 2:DeepSeek, 3:OpenRouter, 4:Private")
    print("3. 模型选择:")
    print("   - model = model_map[client_2_use][0-1] 来切换模型")
    print("4. 私有化部署配置:")
    print("   - PRIVATE_URL: 私有化服务地址 (如: https://127.0.0.1:33/v1)")
    print("   - PRIVATE_MODEL_NAME: 模型名称 (如: qwen-2.5-coder)")
    print("   - PRIVATE_DEPLOYMENT_TYPE: 部署类型 (vllm/tgi/ollama)")
    print("   - PRIVATE_API_KEY: API密钥 (通常为 EMPTY)")
    print("5. 高级选项:")
    print("   - 取消注释上面的选项来测试更多功能")
    print("=" * 80)
