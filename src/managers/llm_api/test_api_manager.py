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
    
    # 测试配置
    test_configs = [
        {
            "client_name": "openai",
            "model": "gpt-4o",
            "api_key_env": "OPENAI_API_KEY"
        },
        {
            "client_name": "anthropic", 
            "model": "claude-3-5-sonnet-20241022",
            "api_key_env": "ANTHROPIC_API_KEY"
        },
        {
            "client_name": "deepseek",
            "model": "deepseek-chat",
            "api_key_env": "DEEPSEEK_API_KEY"
        },
        {
            "client_name": "openrouter",
            "model": "openai/gpt-4o",
            "api_key_env": "OPENROUTER_API_KEY"
        }
    ]
    
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
    
    print("=" * 80)
    print("LLM API 客户端工具调用测试")
    print("=" * 80)
    
    for config in test_configs:
        client_name = config["client_name"]
        model = config["model"]
        api_key_env = config["api_key_env"]
        
        print(f"\n🔍 测试客户端: {client_name.upper()}")
        print(f"📋 模型: {model}")
        print("-" * 50)
        
        # 检查API密钥
        api_key = os.getenv(api_key_env)
        if not api_key or api_key.startswith('your-'):
            print(f"⚠️  跳过 {client_name}: 未配置 {api_key_env}")
            continue
        
        try:
            # 创建管理器
            manager = LLMAPIManager(
                client_name=client_name,
                stream=False,
                api_key=api_key,
                timeout=30,
                max_retries=2
            )
            
            print(f"✅ 客户端初始化成功")
            
            # 测试非流式响应
            print("📤 发送请求...")
            response = manager.chat(
                model=model,
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
                
                # 检查选择项
                if response.choices:
                    choice = response.choices[0]
                    content = choice.message.content if choice.message.content else ""
                    print(f"📄 消息内容: {content[:200]}...")
                    print(f"🏁 完成原因: {choice.finish_reason}")
                    
                    # 检查工具调用
                    if choice.message.tool_calls:
                        print(f"🔧 工具调用数量: {len(choice.message.tool_calls)}")
                        for i, tool_call in enumerate(choice.message.tool_calls):
                            print(f"   工具 {i+1}: {tool_call.get('function', {}).get('name', 'unknown')}")
                    else:
                        print("🔧 无工具调用")
                
                # 检查使用统计
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