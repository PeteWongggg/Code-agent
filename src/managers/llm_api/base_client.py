"""
LLM API 基类 - 标准 OpenAI 格式
支持多种大模型提供商的统一接口
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, AsyncGenerator, Generator
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import logging
import requests
import asyncio


class MessageRole(Enum):
    """消息角色枚举"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ChatMessage:
    """聊天消息数据类"""
    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


@dataclass
class ChatCompletionRequest:
    """聊天补全请求参数"""
    messages: List[ChatMessage]
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    user: Optional[str] = None


@dataclass
class Usage:
    """Token 使用统计"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class Choice:
    """响应选择项"""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None


@dataclass
class ChatCompletionResponse:
    """聊天补全响应"""
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None


@dataclass
class ChatCompletionChunk:
    """流式响应数据块"""
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    system_fingerprint: Optional[str] = None


@dataclass
class EmbeddingRequest:
    """文本嵌入请求参数"""
    input: Union[str, List[str]]
    model: str
    encoding_format: str = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None


@dataclass 
class EmbeddingData:
    """单个嵌入向量数据"""
    object: str
    embedding: List[float]
    index: int


@dataclass
class EmbeddingUsage:
    """嵌入请求的Token使用统计"""
    prompt_tokens: int
    total_tokens: int


@dataclass
class EmbeddingResponse:
    """文本嵌入响应"""
    object: str
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage


class BaseLLMAPI(ABC):
    """
    LLM API 基类
    
    提供标准 OpenAI 格式的接口，支持：
    - 同步/异步聊天补全
    - 流式响应
    - 工具调用
    - 错误处理和重试
    - 使用统计
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ):
        """
        初始化 LLM API 客户端
        
        Args:
            api_key: API 密钥
            base_url: API 基础 URL
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            retry_delay: 重试延迟时间（秒）
            **kwargs: 其他配置参数
        """
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.extra_config = kwargs
        
        # 设置日志
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # HTTP 客户端相关配置（使用 requests）
        self.session: Optional[requests.Session] = None
        
        # 初始化客户端
        self._initialize_client()
    
    def _create_http_clients(self, headers: Dict[str, str]) -> None:
        """
        创建通用的 HTTP 会话（使用 requests）
        
        Args:
            headers: HTTP 请求头
        """
        self.session = requests.Session()
        self.session.headers.update(headers)
        self.session.timeout = self.timeout
    
    @abstractmethod
    def _initialize_client(self) -> None:
        """初始化具体的 API 客户端"""
        pass
    
    @abstractmethod
    def _get_chat_endpoint(self) -> str:
        """获取聊天端点路径"""
        pass
    
    @abstractmethod
    def _build_request_payload(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """构建请求载荷"""
        pass
    
    @abstractmethod
    def _parse_response(self, response_data: Dict[str, Any]) -> ChatCompletionResponse:
        """解析 API 响应"""
        pass
    
    @abstractmethod
    def _parse_stream_chunk(self, chunk_data: Dict[str, Any]) -> Optional[ChatCompletionChunk]:
        """解析流式响应数据块"""
        pass
    
    def chat_completions_create(
        self,
        request: ChatCompletionRequest
    ) -> Union[ChatCompletionResponse, Generator[ChatCompletionChunk, None, None]]:
        """
        创建聊天补全（通用实现）
        
        Args:
            request: 聊天补全请求参数
            
        Returns:
            ChatCompletionResponse: 非流式响应
            Generator[ChatCompletionChunk]: 流式响应生成器
        """
        self.validate_request(request)
        
        def _make_request():
            payload = self._build_request_payload(request)
            endpoint = self._get_chat_endpoint()
            if request.stream:
                return self._stream_chat_completion(payload, endpoint)
            else:
                
                # 构建完整的 URL
                full_url = self.base_url.rstrip('/') + endpoint
                
                # 准备请求头
                headers = {
                    "Content-Type": "application/json",
                }
                
                # 添加认证头（如果需要）
                if self.api_key and self.api_key != "EMPTY":
                    headers["Authorization"] = f"Bearer {self.api_key}"
                
                # 使用 requests 发送请求
                response = requests.post(
                    full_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )
                
                response.raise_for_status()
                return self._parse_response(response.json())
        
        return self._retry_with_backoff(_make_request)
    
    async def achat_completions_create(
        self,
        request: ChatCompletionRequest
    ) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionChunk, None]]:
        """
        异步创建聊天补全（使用 requests 实现）
        
        Args:
            request: 聊天补全请求参数
            
        Returns:
            ChatCompletionResponse: 非流式响应
            AsyncGenerator[ChatCompletionChunk]: 异步流式响应生成器
        """
        self.validate_request(request)
        
        def _make_async_request():
            payload = self._build_request_payload(request)
            endpoint = self._get_chat_endpoint()
            
            if request.stream:
                return self._stream_chat_completion(payload, endpoint)  # 使用同步流式方法
            else:
                # 构建完整的 URL
                full_url = self.base_url.rstrip('/') + endpoint
                
                # 准备请求头
                headers = {
                    "Content-Type": "application/json"
                }
                
                # 添加认证头（如果需要）
                if self.api_key and self.api_key != "EMPTY":
                    headers["Authorization"] = f"Bearer {self.api_key}"
                
                # 使用 requests 发送请求
                response = requests.post(
                    full_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return self._parse_response(response.json())
        
        # 在线程池中运行同步代码
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _make_async_request)
    
    def create_message(
        self,
        role: MessageRole,
        content: str,
        **kwargs
    ) -> ChatMessage:
        """
        创建聊天消息
        
        Args:
            role: 消息角色
            content: 消息内容
            **kwargs: 其他消息参数
            
        Returns:
            ChatMessage: 聊天消息对象
        """
        return ChatMessage(role=role, content=content, **kwargs)
    
    def create_request(
        self,
        messages: List[ChatMessage],
        model: str,
        **kwargs
    ) -> ChatCompletionRequest:
        """
        创建聊天补全请求
        
        Args:
            messages: 消息列表
            model: 模型名称
            **kwargs: 其他请求参数
            
        Returns:
            ChatCompletionRequest: 聊天补全请求对象
        """
        return ChatCompletionRequest(messages=messages, model=model, **kwargs)
    
    def _handle_error(self, error: Exception) -> None:
        """
        处理 API 错误
        
        Args:
            error: 异常对象
        """
        self.logger.error(f"API 请求失败: {error}")
        raise error
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """
        带退避的重试机制
        
        Args:
            func: 要重试的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            函数执行结果
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)  # 指数退避
                    self.logger.warning(f"第 {attempt + 1} 次尝试失败，{delay}秒后重试: {e}")
                    time.sleep(delay)
                else:
                    self.logger.error(f"所有重试都失败了: {e}")
        
        raise last_exception
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息字典
        """
        return {
            "provider": self.__class__.__name__,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries
        }
    
    def validate_request(self, request: ChatCompletionRequest) -> bool:
        """
        验证请求参数
        
        Args:
            request: 聊天补全请求
            
        Returns:
            bool: 验证是否通过
        """
        if not request.messages:
            raise ValueError("消息列表不能为空")
        
        if not request.model:
            raise ValueError("模型名称不能为空")
        
        for message in request.messages:
            if not message.content and not message.tool_calls:
                raise ValueError("消息内容或工具调用不能同时为空")
        
        return True
    
    def format_messages_for_api(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """
        将消息格式化为 API 所需格式
        
        Args:
            messages: 消息列表
            
        Returns:
            List[Dict[str, Any]]: 格式化后的消息列表
        """
        formatted_messages = []
        
        for message in messages:
            msg_dict = {
                "role": message.role.value,
                "content": message.content
            }
            
            if message.name:
                msg_dict["name"] = message.name
            
            if message.tool_calls:
                msg_dict["tool_calls"] = message.tool_calls
            
            if message.tool_call_id:
                msg_dict["tool_call_id"] = message.tool_call_id
            
            formatted_messages.append(msg_dict)
        
        return formatted_messages
    
    def _stream_chat_completion(
        self,
        payload: Dict[str, Any],
        endpoint: str
    ) -> Generator[ChatCompletionChunk, None, None]:
        """
        通用流式聊天补全（使用 requests 实现）
        
        Args:
            payload: 请求载荷
            endpoint: 端点路径
            
        Yields:
            ChatCompletionChunk: 流式响应数据块
        """
        # 构建完整的 URL
        full_url = self.base_url.rstrip('/') + endpoint
        # 准备请求头
        headers = {
            "Content-Type": "application/json"
        }
        # 添加认证头（如果需要）
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"
        # 使用 requests 发送流式请求
        response = requests.post(
            full_url,
            json=payload,
            headers=headers,
            timeout=self.timeout,
            stream=True
        )
        response.raise_for_status()
        
        try:
            line_count = 0
            for line in response.iter_lines():
                if line:
                    line_count += 1
                    line_str = line.decode('utf-8') # 只显示前100个字符
                    chunk = self._parse_stream_line(line_str)
                    if chunk:
                        yield chunk
                    else:
                        print(f"跳过无效行")
            print(f"流式响应处理完成，共处理 {line_count} 行")
        finally:
            response.close()
    
    async def _astream_chat_completion(
        self,
        payload: Dict[str, Any],
        endpoint: str
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """
        异步流式聊天补全（使用 requests 在线程池中运行）
        
        Args:
            payload: 请求载荷
            endpoint: 端点路径
            
        Yields:
            ChatCompletionChunk: 流式响应数据块
        """
        import asyncio
        
        def _sync_stream():
            return list(self._stream_chat_completion(payload, endpoint))
        
        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(None, _sync_stream)
        
        for chunk in chunks:
            yield chunk
    
    def _parse_stream_line(self, line: str) -> Optional[ChatCompletionChunk]:
        """
        通用流式响应行解析
        
        Args:
            line: 响应行
            
        Returns:
            Optional[ChatCompletionChunk]: 解析后的数据块
        """
        # 标准 SSE 格式处理
        if line.startswith("data: "):
            data = line[6:]  # 移除 "data: " 前缀
            
            if data.strip() == "[DONE]":
                return None
            
            try:
                chunk_data = json.loads(data)
                return self._parse_stream_chunk(chunk_data)
            except json.JSONDecodeError:
                self.logger.warning(f"无法解析流式数据: {data}")
                return None
        
        return None
    
    def close(self) -> None:
        """关闭客户端连接（使用 requests）"""
        if self.session:
            self.session.close()
    
    async def aclose(self) -> None:
        """异步关闭客户端连接（使用 requests）"""
        if self.session:
            self.session.close()
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.close()
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.aclose()
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.__class__.__name__}(base_url={self.base_url})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (
            f"{self.__class__.__name__}("
            f"base_url={self.base_url}, "
            f"timeout={self.timeout}, "
            f"max_retries={self.max_retries})"
        )
