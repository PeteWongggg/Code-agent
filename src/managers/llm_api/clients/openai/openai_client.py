"""
OpenAI 客户端实现
基于 BaseLLMAPI 的 OpenAI API 具体实现
"""

import time
from typing import Dict, List, Any, Optional, Union

from src.managers.llm_api.base_client import (
    BaseLLMAPI,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    ChatMessage,
    MessageRole,
    Choice,
    Usage,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingData,
    EmbeddingUsage
)


class OpenAIClient(BaseLLMAPI):
    """
    OpenAI API 客户端实现
    
    支持官方 OpenAI API 以及兼容 OpenAI 格式的其他 API 服务
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ):
        """
        初始化 OpenAI 客户端
        
        Args:
            api_key: OpenAI API 密钥
            base_url: API 基础 URL，默认为 OpenAI 官方 API
            organization: 组织 ID（可选）
            timeout: 请求超时时间
            max_retries: 最大重试次数
            retry_delay: 重试延迟时间
            **kwargs: 其他配置参数
        """
        self.organization = organization
        
        # 设置默认 base_url
        if base_url is None:
            base_url = "https://api.openai.com/v1"
        
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            **kwargs
        )
    
    def _initialize_client(self) -> None:
        """初始化 HTTP 客户端"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "tokfinity-llm-client/1.0"
        }
        
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        
        # 使用基类的通用方法创建客户端
        self._create_http_clients(headers)
    
    def _get_chat_endpoint(self) -> str:
        """获取聊天端点路径"""
        return "/chat/completions"
    
    def _build_request_payload(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """
        构建请求载荷
        
        Args:
            request: 聊天补全请求
            
        Returns:
            Dict[str, Any]: 请求载荷
        """
        payload = {
            "model": request.model,
            "messages": self.format_messages_for_api(request.messages),
            "temperature": request.temperature,
            "top_p": request.top_p,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
            "stream": request.stream
        }
        
        # 可选参数
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        
        if request.stop is not None:
            payload["stop"] = request.stop
        
        if request.tools is not None:
            payload["tools"] = request.tools
        
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        
        if request.response_format is not None:
            payload["response_format"] = request.response_format
        
        if request.seed is not None:
            payload["seed"] = request.seed
        
        if request.user is not None:
            payload["user"] = request.user
        
        return payload
    
    def _parse_response(self, response_data: Dict[str, Any]) -> ChatCompletionResponse:
        """
        解析 API 响应
        
        Args:
            response_data: API 响应数据
            
        Returns:
            ChatCompletionResponse: 解析后的响应对象
        """
        choices = []
        for choice_data in response_data.get("choices", []):
            message_data = choice_data.get("message", {})
            
            # 解析工具调用
            tool_calls = message_data.get("tool_calls")
            
            message = ChatMessage(
                role=MessageRole(message_data.get("role", "assistant")),
                content=message_data.get("content", ""),
                tool_calls=tool_calls
            )
            
            choice = Choice(
                index=choice_data.get("index", 0),
                message=message,
                finish_reason=choice_data.get("finish_reason"),
                logprobs=choice_data.get("logprobs")
            )
            choices.append(choice)
        
        # 解析使用统计
        usage_data = response_data.get("usage", {})
        usage = None
        if usage_data:
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0)
            )
        
        return ChatCompletionResponse(
            id=response_data.get("id", ""),
            object=response_data.get("object", "chat.completion"),
            created=response_data.get("created", int(time.time())),
            model=response_data.get("model", ""),
            choices=choices,
            usage=usage,
            system_fingerprint=response_data.get("system_fingerprint")
        )
    
    def _parse_stream_chunk(self, chunk_data: Dict[str, Any]) -> Optional[ChatCompletionChunk]:
        """
        解析流式响应数据块
        
        Args:
            chunk_data: 原始数据块
            
        Returns:
            Optional[ChatCompletionChunk]: 解析后的数据块
        """
        return ChatCompletionChunk(
            id=chunk_data.get("id", ""),
            object=chunk_data.get("object", "chat.completion.chunk"),
            created=chunk_data.get("created", int(time.time())),
            model=chunk_data.get("model", ""),
            choices=chunk_data.get("choices", []),
            system_fingerprint=chunk_data.get("system_fingerprint")
        )
    
    def list_models(self) -> Dict[str, Any]:
        """
        获取可用模型列表
        
        Returns:
            Dict[str, Any]: 模型列表响应
        """
        response = self.client.get("/models")
        response.raise_for_status()
        return response.json()
    
    async def alist_models(self) -> Dict[str, Any]:
        """
        异步获取可用模型列表
        
        Returns:
            Dict[str, Any]: 模型列表响应
        """
        response = await self.async_client.get("/models")
        response.raise_for_status()
        return response.json()
    
    def create_embeddings(
        self,
        input_text: Union[str, List[str]],
        model: str,
        encoding_format: str = "float",
        dimensions: Optional[int] = None,
        user: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None
    ) -> EmbeddingResponse:
        """
        创建文本嵌入向量
        
        Args:
            input_text: 输入文本，可以是单个字符串或字符串列表
            model: 嵌入模型名称 (如 'text-embedding-ada-002', 'text-embedding-3-small')
            encoding_format: 编码格式，'float' 或 'base64'
            dimensions: 嵌入向量的维度（仅部分模型支持）
            user: 用户标识符（可选）
            timeout: 请求超时时间（秒），如果不指定则使用实例默认值
            max_retries: 最大重试次数，如果不指定则使用实例默认值
            retry_delay: 重试延迟时间（秒），如果不指定则使用实例默认值
            
        Returns:
            EmbeddingResponse: 嵌入向量响应
            
        Raises:
            Exception: API 调用失败时抛出异常
        """
        # 使用传入的参数或实例默认值
        actual_timeout = timeout if timeout is not None else self.timeout
        actual_max_retries = max_retries if max_retries is not None else self.max_retries
        actual_retry_delay = retry_delay if retry_delay is not None else self.retry_delay
        
        # 构建请求对象
        request = EmbeddingRequest(
            input=input_text,
            model=model,
            encoding_format=encoding_format,
            dimensions=dimensions,
            user=user
        )
        
        # 构建请求载荷
        payload = self._build_embedding_request_payload(request)
        
        # 执行带重试的请求
        for attempt in range(actual_max_retries + 1):
            try:
                print(f"Debug: 发送嵌入请求 (尝试 {attempt + 1}/{actual_max_retries + 1})")
                print(f"Debug: URL: {self.base_url}/embeddings")
                print(f"Debug: Payload: {payload}")
                
                response = self.session.post(
                    f"{self.base_url}/embeddings",
                    json=payload,
                    timeout=actual_timeout
                )
                
                print(f"Debug: 响应状态码: {response.status_code}")
                
                if response.status_code == 200:
                    response_data = response.json()
                    return self._parse_embedding_response(response_data)
                else:
                    error_msg = f"嵌入请求失败 (尝试 {attempt + 1}): HTTP {response.status_code}"
                    if hasattr(response, 'text'):
                        error_msg += f" - {response.text}"
                    print(f"Debug: {error_msg}")
                    
                    # 如果不是最后一次尝试，等待后重试
                    if attempt < actual_max_retries:
                        print(f"Debug: 等待 {actual_retry_delay} 秒后重试...")
                        time.sleep(actual_retry_delay)
                        continue
                    else:
                        raise Exception(f"所有重试都失败，无法创建嵌入: {error_msg}")
                        
            except Exception as e:
                error_msg = f"嵌入请求异常 (尝试 {attempt + 1}): {str(e)}"
                print(f"Debug: {error_msg}")
                
                # 如果不是最后一次尝试，等待后重试
                if attempt < actual_max_retries:
                    print(f"Debug: 等待 {actual_retry_delay} 秒后重试...")
                    time.sleep(actual_retry_delay)
                    continue
                else:
                    raise Exception(f"所有重试都因异常失败: {str(e)}")
        
        raise Exception("未知错误：无法创建嵌入")
    
    def _build_embedding_request_payload(self, request: EmbeddingRequest) -> Dict[str, Any]:
        """
        构建嵌入请求载荷
        
        Args:
            request: 嵌入请求对象
            
        Returns:
            Dict[str, Any]: 请求载荷
        """
        payload = {
            "input": request.input,
            "model": request.model,
            "encoding_format": request.encoding_format
        }
        
        # 可选参数
        if request.dimensions is not None:
            payload["dimensions"] = request.dimensions
            
        if request.user is not None:
            payload["user"] = request.user
            
        return payload
    
    def _parse_embedding_response(self, response_data: Dict[str, Any]) -> EmbeddingResponse:
        """
        解析嵌入API响应
        
        Args:
            response_data: API响应数据
            
        Returns:
            EmbeddingResponse: 解析后的响应对象
        """
        # 解析嵌入数据
        embedding_data_list = []
        for data_item in response_data.get("data", []):
            embedding_data = EmbeddingData(
                object=data_item.get("object", "embedding"),
                embedding=data_item.get("embedding", []),
                index=data_item.get("index", 0)
            )
            embedding_data_list.append(embedding_data)
        
        # 解析使用统计
        usage_data = response_data.get("usage", {})
        usage = EmbeddingUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0)
        )
        
        return EmbeddingResponse(
            object=response_data.get("object", "list"),
            data=embedding_data_list,
            model=response_data.get("model", ""),
            usage=usage
        )