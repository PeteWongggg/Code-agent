"""
LLM API 管理器
统一管理和封装不同的 LLM 客户端，提供简化的聊天接口
"""

import os
import time
from typing import Dict, List, Any, Optional, Union, Generator
from dotenv import load_dotenv

from .base_client import (
    BaseLLMAPI,
    ChatMessage,
    MessageRole,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    EmbeddingRequest,
    EmbeddingResponse
)

# 导入所有客户端
from .clients.openai.openai_client import OpenAIClient
from .clients.anthropic.anthropic_client import AnthropicClient
from .clients.deepseek.deepseek_client import DeepSeekClient
from .clients.openrouter.openrouter_client import OpenRouterClient
from .clients.private.private_client import PrivateModelClient


class LLMAPIManager:
    """
    LLM API 统一管理器
    
    提供简化的接口来使用不同的 LLM 客户端，支持：
    - 自动客户端初始化
    - 统一的聊天接口
    - 流式和非流式响应
    - 环境变量自动配置
    """
    
    # 支持的客户端映射
    SUPPORTED_CLIENTS = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient, 
        "deepseek": DeepSeekClient,
        "openrouter": OpenRouterClient,
        "private": PrivateModelClient
    }
    
    # 默认配置
    DEFAULT_CONFIGS = {
        "openai": {
            "base_url": None,  # 使用默认值
            "api_key_env": "OPENAI_API_KEY"
        },
        "anthropic": {
            "base_url": None,
            "api_key_env": "ANTHROPIC_API_KEY"
        },
        "deepseek": {
            "base_url": None,
            "api_key_env": "DEEPSEEK_API_KEY"
        },
        "openrouter": {
            "base_url": None,
            "api_key_env": "OPENROUTER_API_KEY",
            "extra_config": {
                "app_name": "tokfinity-llm-client",
                "site_url": "https://github.com/your-repo"
            }
        },
        "private": {
            "base_url": "http://localhost:8000/v1",
            "api_key_env": "PRIVATE_API_KEY",
            "extra_config": {
                "deployment_type": "vllm"
            }
        }
    }
    
    def __init__(
        self,
        client_name: str,
        stream: bool = False,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        auto_load_env: bool = True,
        logger: Optional[Any] = None,
        **kwargs
    ):
        """
        初始化 LLM API 管理器
        
        Args:
            client_name: 客户端名称 (openai, anthropic, deepseek, openrouter, private)
            stream: 是否启用流式响应
            api_key: API 密钥（可选，会自动从环境变量读取）
            base_url: 基础 URL（可选，使用默认值）
            timeout: 请求超时时间
            max_retries: 最大重试次数
            auto_load_env: 是否自动加载环境变量
            logger: 日志记录器实例（可选）
            **kwargs: 其他配置参数
        """
        self.client_name = client_name.lower()
        self.stream = stream
        self.timeout = timeout
        self.max_retries = max_retries
        self.logger = logger
        
        # 自动加载环境变量
        if auto_load_env:
            self._load_environment()
        
        # 验证客户端名称
        if self.client_name not in self.SUPPORTED_CLIENTS:
            raise ValueError(
                f"不支持的客户端: {client_name}。"
                f"支持的客户端: {list(self.SUPPORTED_CLIENTS.keys())}"
            )
        
        # 初始化客户端
        self.client = self._create_client(api_key, base_url, **kwargs)
        
    def _load_environment(self) -> None:
        """加载环境变量"""
        # 尝试从项目根目录加载 .env 文件
        env_paths = [
            ".env",
            "../.env",
            "../../.env",
            "../../../.env"
        ]
        
        for env_path in env_paths:
            if os.path.exists(env_path):
                load_dotenv(env_path)
                break
    
    def _create_client(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ) -> BaseLLMAPI:
        """
        创建指定类型的客户端
        
        Args:
            api_key: API 密钥
            base_url: 基础 URL
            **kwargs: 其他配置参数
            
        Returns:
            BaseLLMAPI: 客户端实例
        """
        client_class = self.SUPPORTED_CLIENTS[self.client_name]
        config = self.DEFAULT_CONFIGS[self.client_name]
        
        # 获取 API 密钥
        if api_key is None:
            api_key = os.getenv(config["api_key_env"])
            if not api_key:
                if self.client_name == "private":
                    api_key = "EMPTY"  # 私有化部署可能不需要密钥
                else:
                    raise ValueError(
                        f"未找到 API 密钥。请设置环境变量 {config['api_key_env']} "
                        f"或在初始化时传入 api_key 参数"
                    )
        
        # 获取基础 URL
        if base_url is None:
            # 先从环境变量获取
            env_key = f"{self.client_name.upper()}_BASE_URL"
            if self.client_name == "private":
                env_key = "PRIVATE_URL"
            
            base_url = os.getenv(env_key)
            if base_url is None:
                base_url = config.get("base_url")
        
        # 构建客户端参数
        client_kwargs = {
            "api_key": api_key,
            "timeout": self.timeout,
            "max_retries": self.max_retries
        }
        
        if base_url:
            client_kwargs["base_url"] = base_url
        
        # 添加额外配置
        extra_config = config.get("extra_config", {})
        client_kwargs.update(extra_config)
        client_kwargs.update(kwargs)
        
        # 特殊处理
        if self.client_name == "openrouter":
            client_kwargs.setdefault("app_name", "tokfinity-llm-client")
        elif self.client_name == "private":
            client_kwargs.setdefault("deployment_type", "vllm")
        
        return client_class(**client_kwargs)
    
    def chat(
        self,
        model: str,
        message: str,
        system_message: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        retry: Optional[int] = None,
        **kwargs
    ) -> Optional[str]:
        """
        发送聊天消息并获取响应
        
        Args:
            model: 模型名称
            message: 用户消息内容
            system_message: 系统消息（可选）
            conversation_history: 对话历史记录（可选）
                格式: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大 token 数
            timeout: 请求超时时间（秒），如果不指定则使用初始化时的值
            retry: 最大重试次数，如果不指定则使用初始化时的值
            **kwargs: 其他请求参数
            
        Returns:
            Optional[str]: 完整的响应内容，如果所有重试都失败则返回 None
        """
        # 使用传入的参数或默认值
        actual_timeout = timeout if timeout is not None else self.timeout
        actual_retry = retry if retry is not None else self.max_retries
        
        if self.logger:
            self.logger.debug(f"开始聊天请求 - 客户端: {self.client_name}, 模型: {model}, 流式: {self.stream}")
        
        # 构建消息列表
        messages = []
        
        # 添加系统消息
        if system_message:
            messages.append(ChatMessage(
                role=MessageRole.SYSTEM,
                content=system_message
            ))
        
        # 添加对话历史
        if conversation_history:
            for msg in conversation_history:
                role = MessageRole(msg["role"])
                content = msg["content"]
                messages.append(ChatMessage(role=role, content=content))
        
        # 添加当前用户消息
        messages.append(ChatMessage(
            role=MessageRole.USER,
            content=message
        ))
        
        # 重试逻辑
        last_exception = None
        for attempt in range(actual_retry + 1):
            try:
                # 创建请求
                request = self.client.create_request(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=self.stream,
                    **kwargs
                )
                
                # 发送请求
                response = self.client.chat_completions_create(request)
                
                if self.stream:
                    # 流式响应：收集所有内容并拼接后返回
                    content_parts = []
                    for chunk in response:
                        if chunk and chunk.choices:
                            delta = chunk.choices[0].get('delta', {})
                            if 'content' in delta and delta['content']:
                                content_parts.append(delta['content'])
                    result = ''.join(content_parts)
                    if self.logger:
                        self.logger.info(f"流式聊天请求成功 - 模型: {model}, 响应长度: {len(result)} 字符")
                    return result
                else:
                    # 非流式响应：返回完整内容
                    result = response.choices[0].message.content
                    if self.logger:
                        self.logger.info(f"聊天请求成功 - 模型: {model}, 响应长度: {len(result)} 字符")
                    return result
                    
            except Exception as e:
                last_exception = e
                if attempt < actual_retry:
                    # 计算退避延迟时间（指数退避）
                    delay = min(2 ** attempt, 30)  # 最大延迟30秒
                    if self.logger:
                        self.logger.warning(f"第 {attempt + 1} 次尝试失败，{delay}秒后重试: {str(e)}")
                    time.sleep(delay)
                else:
                    if self.logger:
                        self.logger.error(f"所有 {actual_retry + 1} 次尝试都失败了: {str(e)}")
        
        # 所有重试都失败，返回 None
        return None
    
    def create_embeddings(
        self,
        input_text: Union[str, List[str]],
        model: str,
        encoding_format: str = "float",
        dimensions: Optional[int] = None,
        user: Optional[str] = None,
        timeout: Optional[int] = None,
        retry: Optional[int] = None,
        **kwargs
    ) -> Optional[EmbeddingResponse]:
        """
        创建文本嵌入向量（支持批量处理）
        
        Args:
            input_text: 输入文本，支持单个字符串或字符串列表进行批量处理
            model: 嵌入模型名称
            encoding_format: 编码格式，默认为 "float"
            dimensions: 嵌入向量的维度（仅部分模型支持）
            user: 用户标识符（可选）
            timeout: 请求超时时间（秒），如果不指定则使用初始化时的值
            retry: 最大重试次数，如果不指定则使用初始化时的值
            **kwargs: 其他请求参数
            
        Returns:
            Optional[EmbeddingResponse]: 嵌入向量响应，如果所有重试都失败则返回 None
        """
        # 使用传入的参数或默认值
        actual_timeout = timeout if timeout is not None else self.timeout
        actual_retry = retry if retry is not None else self.max_retries
        
        # 确保输入是列表格式，方便批量处理
        if isinstance(input_text, str):
            text_list = [input_text]
            single_input = True
        else:
            text_list = input_text
            single_input = False
            
        if self.logger:
            self.logger.debug(f"开始嵌入请求 - 客户端: {self.client_name}, 模型: {model}, 文本数量: {len(text_list)}")
        
        # 检查客户端是否支持 embedding
        if not hasattr(self.client, 'create_embeddings'):
            error_msg = f"客户端 {self.client_name} 不支持 embedding 功能"
            if self.logger:
                self.logger.error(error_msg)
            return None
        
        # 重试逻辑
        last_exception = None
        for attempt in range(actual_retry + 1):
            try:
                if self.logger:
                    self.logger.debug(f"第 {attempt + 1} 次尝试创建嵌入向量")
                
                # 调用客户端的 embedding 方法
                response = self.client.create_embeddings(
                    input_text=text_list,  # 总是传递列表
                    model=model,
                    encoding_format=encoding_format,
                    dimensions=dimensions,
                    user=user,
                    timeout=actual_timeout,
                    max_retries=1,  # 在 manager 层面控制重试，客户端层面不重试
                    **kwargs
                )
                
                if response:
                    if self.logger:
                        self.logger.info(f"嵌入向量创建成功 - 模型: {model}, 向量数量: {len(response.data)}, Token使用: {response.usage.total_tokens}")
                    return response
                else:
                    raise Exception("客户端返回空响应")
                    
            except Exception as e:
                last_exception = e
                if attempt < actual_retry:
                    # 计算退避延迟时间（指数退避）
                    delay = min(2 ** attempt, 30)  # 最大延迟30秒
                    if self.logger:
                        self.logger.warning(f"第 {attempt + 1} 次嵌入请求失败，{delay}秒后重试: {str(e)}")
                    time.sleep(delay)
                else:
                    if self.logger:
                        self.logger.error(f"所有 {actual_retry + 1} 次嵌入请求尝试都失败了: {str(e)}")
        
        # 所有重试都失败，返回 None
        return None
    
    def get_client_info(self) -> Dict[str, Any]:
        """
        获取客户端信息
        
        Returns:
            Dict[str, Any]: 客户端信息
        """
        return {
            "client_name": self.client_name,
            "stream": self.stream,
            "client_info": self.client.get_model_info() if hasattr(self.client, 'get_model_info') else {}
        }
    
    def close(self) -> None:
        """关闭客户端连接"""
        if hasattr(self.client, 'close'):
            self.client.close()
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.close()
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"LLMAPIManager(client={self.client_name}, stream={self.stream})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (
            f"LLMAPIManager("
            f"client_name='{self.client_name}', "
            f"stream={self.stream})"
        )


# 便捷函数
def create_manager(
    client_name: str,
    stream: bool = False,
    logger: Optional[Any] = None,
    **kwargs
) -> LLMAPIManager:
    """
    创建 LLM API 管理器的便捷函数
    
    Args:
        client_name: 客户端名称
        stream: 是否启用流式响应
        logger: 日志记录器实例（可选）
        **kwargs: 其他参数
        
    Returns:
        LLMAPIManager: 管理器实例
    """
    return LLMAPIManager(
        client_name=client_name,
        stream=stream,
        logger=logger,
        **kwargs
    )


# 预定义的常用配置
COMMON_CONFIGS = {
    "openai_gpt4": {
        "client_name": "openai",
        "model_name": "gpt-4o"
    },
    "openai_gpt35": {
        "client_name": "openai", 
        "model_name": "gpt-3.5-turbo"
    },
    "claude_sonnet": {
        "client_name": "anthropic",
        "model_name": "claude-3-5-sonnet-20241022"
    },
    "claude_haiku": {
        "client_name": "anthropic",
        "model_name": "claude-3-haiku-20240307"
    },
    "deepseek_chat": {
        "client_name": "deepseek",
        "model_name": "deepseek-chat"
    },
    "deepseek_coder": {
        "client_name": "deepseek",
        "model_name": "deepseek-coder"
    }
}


def create_common_manager(config_name: str, stream: bool = False, logger: Optional[Any] = None, **kwargs) -> LLMAPIManager:
    """
    使用预定义配置创建管理器
    
    Args:
        config_name: 配置名称（见 COMMON_CONFIGS）
        stream: 是否启用流式响应
        logger: 日志记录器实例（可选）
        **kwargs: 其他参数
        
    Returns:
        LLMAPIManager: 管理器实例
    """
    if config_name not in COMMON_CONFIGS:
        raise ValueError(f"未知配置: {config_name}。可用配置: {list(COMMON_CONFIGS.keys())}")
    
    config = COMMON_CONFIGS[config_name]
    return LLMAPIManager(
        client_name=config["client_name"],
        stream=stream,
        logger=logger,
        **kwargs
    )
