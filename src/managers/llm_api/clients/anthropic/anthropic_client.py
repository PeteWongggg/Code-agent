"""
Anthropic Claude 客户端实现
基于 BaseLLMAPI 的 Anthropic API 具体实现
"""

import json
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


class AnthropicClient(BaseLLMAPI):
    """
    Anthropic Claude API 客户端实现
    
    支持 Anthropic Claude 系列模型，包括 Claude-3 等
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        anthropic_version: str = "2023-06-01",
        **kwargs
    ):
        """
        初始化 Anthropic 客户端
        
        Args:
            api_key: Anthropic API 密钥
            base_url: API 基础 URL，默认为 Anthropic 官方 API
            timeout: 请求超时时间
            max_retries: 最大重试次数
            retry_delay: 重试延迟时间
            anthropic_version: API 版本号
            **kwargs: 其他配置参数
        """
        self.anthropic_version = anthropic_version
        
        # 设置默认 base_url
        if base_url is None:
            base_url = "https://api.anthropic.com"
        
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
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": self.anthropic_version,
            "User-Agent": "tokfinity-llm-client/1.0"
        }
        
        # 使用基类的通用方法创建客户端
        self._create_http_clients(headers)
    
    def _get_chat_endpoint(self) -> str:
        """获取聊天端点路径"""
        return "/v1/messages"
    
    def _build_request_payload(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """
        构建 Anthropic API 请求载荷
        
        Args:
            request: 聊天补全请求
            
        Returns:
            Dict[str, Any]: 请求载荷
        """
        # 将消息转换为 Anthropic 格式
        messages, system_prompt = self._convert_messages_to_anthropic_format(request.messages)
        
        payload = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens or 1024,  # Anthropic 需要 max_tokens
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": request.stream
        }
        
        # 如果有系统提示，添加到载荷中
        if system_prompt:
            payload["system"] = system_prompt
        
        # 可选参数
        if request.stop is not None:
            payload["stop_sequences"] = request.stop if isinstance(request.stop, list) else [request.stop]
        
        # Anthropic 不直接支持 frequency_penalty 和 presence_penalty
        # 可以通过其他方式实现或忽略
        
        return payload
    
    def _convert_messages_to_anthropic_format(self, messages: List[ChatMessage]) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """
        将消息转换为 Anthropic 格式
        
        Args:
            messages: 原始消息列表
            
        Returns:
            tuple: (转换后的消息列表, 系统提示)
        """
        anthropic_messages = []
        system_prompt = None
        
        for message in messages:
            if message.role == MessageRole.SYSTEM:
                # Anthropic 将系统消息单独处理
                system_prompt = message.content
            elif message.role in [MessageRole.USER, MessageRole.ASSISTANT]:
                anthropic_messages.append({
                    "role": message.role.value,
                    "content": message.content
                })
            # 暂时忽略 TOOL 角色，可以根据需要扩展
        
        return anthropic_messages, system_prompt
    
    def _parse_response(self, response_data: Dict[str, Any]) -> ChatCompletionResponse:
        """
        解析 Anthropic API 响应为 OpenAI 格式
        
        Args:
            response_data: Anthropic API 响应数据
            
        Returns:
            ChatCompletionResponse: OpenAI 格式的响应对象
        """
        # Anthropic 响应格式转换为 OpenAI 格式
        content = ""
        if response_data.get("content"):
            content = response_data["content"][0].get("text", "") if response_data["content"] else ""
        
        message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content
        )
        
        choice = Choice(
            index=0,
            message=message,
            finish_reason=self._convert_stop_reason(response_data.get("stop_reason"))
        )
        
        # 解析使用统计
        usage_data = response_data.get("usage", {})
        usage = None
        if usage_data:
            usage = Usage(
                prompt_tokens=usage_data.get("input_tokens", 0),
                completion_tokens=usage_data.get("output_tokens", 0),
                total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0)
            )
        
        return ChatCompletionResponse(
            id=response_data.get("id", ""),
            object="chat.completion",
            created=int(time.time()),
            model=response_data.get("model", ""),
            choices=[choice],
            usage=usage
        )
    
    def _convert_stop_reason(self, stop_reason: Optional[str]) -> Optional[str]:
        """
        将 Anthropic 停止原因转换为 OpenAI 格式
        
        Args:
            stop_reason: Anthropic 停止原因
            
        Returns:
            Optional[str]: OpenAI 格式的停止原因
        """
        if stop_reason == "end_turn":
            return "stop"
        elif stop_reason == "max_tokens":
            return "length"
        elif stop_reason == "stop_sequence":
            return "stop"
        else:
            return stop_reason
    
    def _parse_stream_chunk(self, chunk_data: Dict[str, Any]) -> Optional[ChatCompletionChunk]:
        """
        解析 Anthropic 流式响应数据块为 OpenAI 格式
        
        Args:
            chunk_data: Anthropic 原始数据块
            
        Returns:
            Optional[ChatCompletionChunk]: OpenAI 格式的数据块
        """
        event_type = chunk_data.get("type")
        
        if event_type == "content_block_delta":
            # 文本内容增量
            delta = chunk_data.get("delta", {})
            text = delta.get("text", "")
            
            if text:
                choices = [{
                    "index": 0,
                    "delta": {
                        "content": text
                    },
                    "finish_reason": None
                }]
                
                return ChatCompletionChunk(
                    id=chunk_data.get("message", {}).get("id", ""),
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=chunk_data.get("message", {}).get("model", ""),
                    choices=choices
                )
        
        elif event_type == "message_stop":
            # 消息结束
            choices = [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
            
            return ChatCompletionChunk(
                id=chunk_data.get("message", {}).get("id", ""),
                object="chat.completion.chunk",
                created=int(time.time()),
                model=chunk_data.get("message", {}).get("model", ""),
                choices=choices
            )
        
        return None
    
    def _parse_stream_line(self, line: str) -> Optional[ChatCompletionChunk]:
        """
        重写流式响应行解析（Anthropic 特殊处理）
        
        Args:
            line: 响应行
            
        Returns:
            Optional[ChatCompletionChunk]: 解析后的数据块
        """
        # Anthropic 可能使用不同的流格式，这里需要特殊处理
        # 如果是 JSON 行，直接解析
        try:
            chunk_data = json.loads(line)
            return self._parse_stream_chunk(chunk_data)
        except json.JSONDecodeError:
            # 如果不是 JSON，尝试标准 SSE 格式
            return super()._parse_stream_line(line)