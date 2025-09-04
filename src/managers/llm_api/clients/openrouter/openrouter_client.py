"""
OpenRouter 客户端实现
基于 BaseLLMAPI 的 OpenRouter API 具体实现

OpenRouter 是一个 LLM API 聚合服务，提供统一接口访问多种大语言模型：
- OpenAI GPT 系列
- Anthropic Claude 系列
- Google PaLM 系列
- Meta LLaMA 系列
- 以及更多开源和闭源模型
"""

import time
from typing import Dict, List, Any, Optional

from src.managers.llm_api.base_client import (
    BaseLLMAPI,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    ChatMessage,
    MessageRole,
    Choice,
    Usage
)


class OpenRouterClient(BaseLLMAPI):
    """
    OpenRouter API 客户端实现
    
    OpenRouter 提供统一的 OpenAI 兼容接口来访问多种 LLM 模型
    支持的模型包括但不限于：
    - openai/gpt-4-turbo-preview
    - anthropic/claude-3-opus
    - google/gemini-pro
    - meta-llama/llama-2-70b-chat
    - mistralai/mixtral-8x7b-instruct
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        app_name: Optional[str] = None,
        site_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ):
        """
        初始化 OpenRouter 客户端
        
        Args:
            api_key: OpenRouter API 密钥
            base_url: API 基础 URL，默认为 OpenRouter 官方 API
            app_name: 应用名称（可选，用于统计）
            site_url: 网站 URL（可选，用于统计）
            timeout: 请求超时时间
            max_retries: 最大重试次数
            retry_delay: 重试延迟时间
            **kwargs: 其他配置参数
        """
        self.app_name = app_name or "tokfinity-llm-client"
        self.site_url = site_url
        
        # 设置默认 base_url
        if base_url is None:
            base_url = "https://openrouter.ai/api/v1"
        
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
            "User-Agent": "tokfinity-llm-client/1.0",
            "X-Title": self.app_name
        }
        
        # 添加可选的网站 URL
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        
        # 使用基类的通用方法创建客户端
        self._create_http_clients(headers)
    
    def _get_chat_endpoint(self) -> str:
        """获取聊天端点路径"""
        return "/chat/completions"
    
    def _build_request_payload(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """
        构建 OpenRouter API 请求载荷
        
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
            "stream": request.stream
        }
        
        # 可选参数
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        
        if request.stop is not None:
            payload["stop"] = request.stop
        
        # OpenRouter 支持 frequency_penalty 和 presence_penalty
        if request.frequency_penalty != 0.0:
            payload["frequency_penalty"] = request.frequency_penalty
        
        if request.presence_penalty != 0.0:
            payload["presence_penalty"] = request.presence_penalty
        
        # 工具调用支持
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
        解析 OpenRouter API 响应
        
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
        获取 OpenRouter 可用模型列表
        
        Returns:
            Dict[str, Any]: 模型列表响应
        """
        response = self.client.get("/models")
        response.raise_for_status()
        return response.json()
    
    async def alist_models(self) -> Dict[str, Any]:
        """
        异步获取 OpenRouter 可用模型列表
        
        Returns:
            Dict[str, Any]: 模型列表响应
        """
        response = await self.async_client.get("/models")
        response.raise_for_status()
        return response.json()
    
    def get_generation_info(self, generation_id: str) -> Dict[str, Any]:
        """
        获取特定生成请求的详细信息（OpenRouter 特有功能）
        
        Args:
            generation_id: 生成请求的 ID
            
        Returns:
            Dict[str, Any]: 生成信息
        """
        response = self.client.get(f"/generation?id={generation_id}")
        response.raise_for_status()
        return response.json()
    
    async def aget_generation_info(self, generation_id: str) -> Dict[str, Any]:
        """
        异步获取特定生成请求的详细信息
        
        Args:
            generation_id: 生成请求的 ID
            
        Returns:
            Dict[str, Any]: 生成信息
        """
        response = await self.async_client.get(f"/generation?id={generation_id}")
        response.raise_for_status()
        return response.json()
    
    def get_account_credits(self) -> Dict[str, Any]:
        """
        获取账户余额信息（OpenRouter 特有功能）
        
        Returns:
            Dict[str, Any]: 账户余额信息
        """
        response = self.client.get("/auth/key")
        response.raise_for_status()
        return response.json()
    
    async def aget_account_credits(self) -> Dict[str, Any]:
        """
        异步获取账户余额信息
        
        Returns:
            Dict[str, Any]: 账户余额信息
        """
        response = await self.async_client.get("/auth/key")
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def get_popular_models() -> List[str]:
        """
        获取 OpenRouter 上热门的模型列表
        
        Returns:
            List[str]: 热门模型名称列表
        """
        return [
            # OpenAI 模型
            "openai/gpt-4-turbo-preview",
            "openai/gpt-4",
            "openai/gpt-3.5-turbo",
            
            # Anthropic 模型
            "anthropic/claude-3-opus",
            "anthropic/claude-3-sonnet",
            "anthropic/claude-3-haiku",
            
            # Google 模型
            "google/gemini-pro",
            "google/gemini-pro-vision",
            
            # Meta 模型
            "meta-llama/llama-2-70b-chat",
            "meta-llama/llama-2-13b-chat",
            
            # Mistral 模型
            "mistralai/mixtral-8x7b-instruct",
            "mistralai/mistral-7b-instruct",
            
            # 开源模型
            "microsoft/wizardlm-2-8x22b",
            "databricks/dbrx-instruct",
            "cohere/command-r-plus",
        ]
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        获取特定模型的详细信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            Dict[str, Any]: 模型信息
        """
        models_response = self.list_models()
        models = models_response.get("data", [])
        
        for model in models:
            if model.get("id") == model_name:
                return model
        
        raise ValueError(f"模型 {model_name} 未找到")
    
    async def aget_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        异步获取特定模型的详细信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            Dict[str, Any]: 模型信息
        """
        models_response = await self.alist_models()
        models = models_response.get("data", [])
        
        for model in models:
            if model.get("id") == model_name:
                return model
        
        raise ValueError(f"模型 {model_name} 未找到")
