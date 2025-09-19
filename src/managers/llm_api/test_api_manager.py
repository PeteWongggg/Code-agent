#!/usr/bin/env python3
"""
LLM API 管理器测试脚本
测试不同客户端的工具调用功能
"""

import os
import sys
from dotenv import load_dotenv

# 添加项目根目录到 Python 路径（从当前文件上溯四级到仓库根目录）
project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入必要的模块
from src.managers.llm_api.api_manager import LLMAPIManager

def main():
    """主测试函数"""
    # 加载环境变量
    load_dotenv()
    
    # 工具定义
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
    
    # 测试消息
    messages = [
        {"role": "system", "content": "你现在是一个有用的AI助手，需要通过搜索工具帮我解决问题。"},
        {"role": "user", "content": "美联储最新的新闻有啥？你需要通过联网搜索后回答我。"}
    ]
    
    # 仅测试默认从配置读取的逻辑
    try:
        print("\n🔍 使用配置中的默认提供商与模型进行测试")
        manager = LLMAPIManager()
        print("✅ 客户端初始化成功")

        # 非流式请求；不显式传入 model，应走默认模型
        print("📤 发送请求...")
        response = manager.chat(
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.7
        )

        if response:
            print("✅ 请求成功")
            print(f"📝 响应ID: {response.id}")
            print(f"🤖 模型: {response.model}")
            print(f"⏰ 创建时间: {response.created}")

            if response.choices:
                choice = response.choices[0]
                content = choice.message.content if choice.message.content else ""
                print(f"📄 消息内容: {content[:200]}...")
                print(f"🏁 完成原因: {choice.finish_reason}")

                if choice.message.tool_calls:
                    print(f"🔧 工具调用数量: {len(choice.message.tool_calls)}")
                    for i, tool_call in enumerate(choice.message.tool_calls):
                        print(f"   工具 {i+1}: {tool_call.get('function', {}).get('name', 'unknown')}")
                else:
                    print("🔧 无工具调用")

            if response.usage:
                print(f"📊 Token使用: {response.usage.total_tokens} (输入: {response.usage.prompt_tokens}, 输出: {response.usage.completion_tokens})")
            else:
                print("📊 无使用统计信息")
        else:
            print("❌ 请求失败: 返回 None")

    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")

    print("-" * 50)
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)

if __name__ == "__main__":
    main()