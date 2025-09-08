"""
OpenRouter 客户端模块

OpenRouter 是一个 LLM API 聚合服务，提供统一接口访问多种大语言模型
"""

from src.managers.llm_api.clients.openrouter.openrouter_client import OpenRouterClient

__all__ = ["OpenRouterClient"]
