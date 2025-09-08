import yaml
import os
from src.managers.llm_api.api_manager import LLMAPIManager
from src.managers.log.logger import Logger
from src.managers.data_loader.load_data import SWEBenchLoader

def load_config(config_path: str = "src/config/config.yaml") -> dict:
    """加载 YAML 配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def load_swe_bench_data(config: dict, logger):
    """加载和处理 SWE-bench 数据集"""
    dataset_config = config.get("dataset", {})
    workspace_config = config.get("workspace", {})
    
    dataset_name = dataset_config.get("name", "princeton-nlp/SWE-bench_Lite")
    split_name = dataset_config.get("split", "dev")
    workspace_path = workspace_config.get("path", "workspace")
    
    logger.info(f"开始处理 SWE-bench 数据集: {dataset_name}")
    
    # 创建数据加载器
    swe_loader = SWEBenchLoader(
        dataset_name=dataset_name,
        split_name=split_name,
        workspace_path=workspace_path,
        logger=logger
    )
    
    # 加载并处理数据集（限制处理数量用于测试）
    max_items = int(os.getenv("MAX_ITEMS", "5"))  # 默认处理5个，可通过环境变量调整
    result = swe_loader.load_and_process_all(max_items=max_items)
    
    # 显示统计信息
    stats = swe_loader.get_stats()
    logger.info(f"数据处理统计: {stats}")
    
    print(f"\n📊 SWE-bench 数据处理完成:")
    print(f"   数据集: {dataset_name}")
    print(f"   分割: {split_name}")
    print(f"   工作空间: {workspace_path}")
    print(f"   处理结果: {result}")
    print(f"   统计信息: {stats}")
    
    return swe_loader, result

def test_embedding_functionality(config: dict, logger):
    """测试 embedding 功能"""
    rag_config = config.get("rag", {})
    embedding_config = rag_config.get("embedding", {})
    
    # 检查是否启用 embedding
    if not embedding_config.get("enabled", False):
        logger.info("Embedding 功能未启用，跳过测试")
        print("⏭️  Embedding 功能未启用，跳过测试")
        return
    
    client_name = embedding_config.get("client", "openai")
    model_name = embedding_config.get("model", "text-embedding-3-small")
    
    logger.info(f"开始测试 Embedding 功能 - 客户端: {client_name}, 模型: {model_name}")
    
    print("🧠 RAG Embedding 功能测试")
    print("=" * 60)
    print(f"🔧 配置: 客户端={client_name}, 模型={model_name}")
    print("=" * 60)
    
    # 测试文本
    test_texts = [
        "人工智能是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。",
        "Machine learning algorithms can automatically learn and improve from experience.",
        "深度学习使用多层神经网络来处理和分析复杂的数据模式。",
        "自然语言处理帮助计算机理解和生成人类语言。"
    ]
    
    print(f"📝 测试文本 ({len(test_texts)} 条):")
    for i, text in enumerate(test_texts, 1):
        print(f"   {i}. {text}")
        logger.debug(f"测试文本 {i}: {text}")
    
    try:
        # 使用 LLMAPIManager 创建统一的客户端管理器
        manager = LLMAPIManager(
            client_name=client_name,
            timeout=30,
            max_retries=2,
            logger=logger
        )
        
        logger.info(f"成功创建 {client_name.upper()} 客户端管理器")
        print(f"✅ 成功创建 {client_name.upper()} 客户端管理器")
        
        print(f"\n🚀 开始生成嵌入向量...")
        logger.info(f"开始调用 embedding API - 模型: {model_name}")
        
        # 通过 LLMAPIManager 调用 embedding API（支持批量处理）
        response = manager.create_embeddings(
            input_text=test_texts,  # 直接传递文本列表，支持批量处理
            model=model_name,
            timeout=30,
            retry=2
        )
        
        # 检查响应是否成功
        if response is None:
            error_msg = "Embedding 生成失败: 所有重试都失败"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            return
        
        # 记录成功结果
        logger.info(f"Embedding 生成成功 - 模型: {response.model}, 向量数量: {len(response.data)}, Token使用: {response.usage.total_tokens}")
        
        print("✅ 嵌入向量生成成功!")
        print(f"\n📊 响应统计:")
        print(f"   🤖 使用模型: {response.model}")
        print(f"   📈 嵌入向量数量: {len(response.data)}")
        print(f"   🔢 Token 使用: {response.usage.prompt_tokens} prompt + {response.usage.total_tokens} total")
        
        # 显示每个嵌入向量的详细信息
        print(f"\n🔍 嵌入向量详情:")
        total_dimensions = 0
        for i, embedding_data in enumerate(response.data):
            vector_dim = len(embedding_data.embedding)
            total_dimensions += vector_dim
            first_few = embedding_data.embedding[:3]  # 显示前3个值
            last_few = embedding_data.embedding[-3:]  # 显示后3个值
            
            print(f"   向量 {i+1}: 维度={vector_dim}")
            print(f"           前3个值: {[round(x, 6) for x in first_few]}")
            print(f"           后3个值: {[round(x, 6) for x in last_few]}")
            
            # 记录到日志
            logger.debug(f"向量 {i+1}: 维度={vector_dim}, 索引={embedding_data.index}")
        
        avg_dimension = total_dimensions // len(response.data) if response.data else 0
        logger.info(f"平均向量维度: {avg_dimension}")
        
        print(f"\n🎯 测试总结:")
        print(f"   ✅ 成功生成 {len(response.data)} 个嵌入向量")
        print(f"   📏 平均向量维度: {avg_dimension}")
        print(f"   ⚡ Token 效率: {response.usage.total_tokens / len(test_texts):.1f} tokens/text")
        print(f"   🎉 Embedding 功能测试完成!")
        
        logger.info("Embedding 功能测试成功完成")
        
        # 关闭管理器
        manager.close()
        
    except Exception as e:
        error_msg = f"Embedding 测试失败: {str(e)}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        
        # 提供故障排除建议
        print(f"\n💡 故障排除建议:")
        print(f"   1. 检查 API 密钥是否正确设置")
        print(f"   2. 确认网络连接正常")
        print(f"   3. 验证模型名称是否正确: {model_name}")
        print(f"   4. 检查 API 配额是否充足")

def test_provider_models(config: dict, logger):
    """测试所有配置的提供商和模型"""
    providers = config.get("providers", {})
    
    # 直接在代码中定义测试参数
    test_message = "四大名著有哪些？请简要介绍每一部。"
    system_message = "你是一个有用的AI助手，请用简洁明了的方式回答问题。"
    temperature = 0.1
    max_tokens = 500
    stream = False
    timeout = 30
    
    print("🚀 LLM API 多提供商模型测试")
    print("=" * 80)
    print(f"📝 测试消息: {test_message}")
    print(f"🔧 配置: stream={stream}, temperature={temperature}, max_tokens={max_tokens}")
    print("=" * 80)
    
    total_tests = 0
    successful_tests = 0
    
    # 遍历每个提供商
    for provider_name, models in providers.items():
        logger.info(f"开始测试提供商: {provider_name}")
        print(f"\n🏢 测试提供商: {provider_name.upper()}")
        print("-" * 60)
        
        try:
            # 创建该提供商的管理器
            manager = LLMAPIManager(
                client_name=provider_name,
                stream=stream,
                timeout=timeout,
                logger=logger
            )
            
            logger.info(f"{provider_name} 客户端创建成功")
            print(f"✅ {provider_name} 客户端创建成功")
            
            if provider_name == "private":
                model_name = os.getenv("PRIVATE_MODEL_NAME", "qwen-2.5-coder-3b-instruct")
                total_tests += 1
                response = manager.chat(
                    model=model_name,
                    message=test_message,
                    system_message=system_message,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                if response is not None:
                    logger.info(f"模型 {model_name} 测试成功，响应长度: {len(response)} 字符")
                    print(f"📤 请求成功")
                    print(f"📥 响应内容:")
                    print(f"   {response}")
                    print(f"✅ 模型 {model_name} 测试成功")
                    successful_tests += 1
                else:
                    logger.warning(f"模型 {model_name} 测试失败: 所有重试都失败")
                    print(f"❌ 模型 {model_name} 测试失败: 所有重试都失败，返回 None")
            else:
                # 遍历该提供商的所有模型
                for model_name in models:
                    total_tests += 1
                    print(f"\n🤖 测试模型: {model_name}")
                    print("." * 40)
                    
                    # 调用聊天接口
                    response = manager.chat(
                        model=model_name,
                        message=test_message,
                        system_message=system_message,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    if response is not None:
                        logger.info(f"模型 {model_name} 测试成功，响应长度: {len(response)} 字符")
                        print(f"📤 请求成功")
                        print(f"📥 响应内容:")
                        print(f"   {response}")
                        print(f"✅ 模型 {model_name} 测试成功")
                        successful_tests += 1
                    else:
                        logger.warning(f"模型 {model_name} 测试失败: 所有重试都失败")
                        print(f"❌ 模型 {model_name} 测试失败: 所有重试都失败，返回 None")
            
            # 关闭管理器
            manager.close()
            
        except Exception as e:
            logger.error(f"提供商 {provider_name} 初始化失败: {str(e)}")
            print(f"❌ 提供商 {provider_name} 初始化失败: {str(e)}")
            # 如果提供商初始化失败，跳过该提供商的所有模型
            for _ in models:
                total_tests += 1
    
    # 显示测试总结
    success_rate = (successful_tests/total_tests*100) if total_tests > 0 else 0
    logger.info(f"测试完成 - 总数: {total_tests}, 成功: {successful_tests}, 失败: {total_tests - successful_tests}, 成功率: {success_rate:.1f}%")
    
    print("\n" + "=" * 80)
    print("🏁 测试完成总结")
    print("=" * 80)
    print(f"📊 总测试数: {total_tests}")
    print(f"✅ 成功测试: {successful_tests}")
    print(f"❌ 失败测试: {total_tests - successful_tests}")
    print(f"📈 成功率: {success_rate:.1f}%")
    
    if successful_tests == 0:
        print("\n💡 提示:")
        print("1. 请确保在 .env 文件中配置了相应的 API 密钥")
        print("2. 检查网络连接是否正常")
        print("3. 确认 API 密钥有足够的余额和权限")
        print("4. 检查配置文件中的模型名称是否正确")
    elif successful_tests < total_tests:
        print(f"\n⚠️  有 {total_tests - successful_tests} 个测试失败，请检查相关配置")
    else:
        print("\n🎉 所有测试都成功完成！")

if __name__ == "__main__":
    # 创建日志记录器
    logger = Logger("logs", "swe_bench_processor")
    logger.info("开始 SWE-bench 数据处理和 LLM API 测试程序")
    
    try:
        # 加载配置文件
        config = load_config(os.getenv("CONFIG_PATH", "config/config.yaml"))
        logger.info(f"成功加载配置文件")
        
        # 检查运行模式
        run_mode = os.getenv("RUN_MODE", "embedding").lower()  # 默认运行数据处理模式
        
        if run_mode == "data":
            logger.info("运行模式: SWE-bench 数据处理")
            # 处理 SWE-bench 数据集
            swe_loader, result = load_swe_bench_data(config, logger)
        elif run_mode == "llm":
            logger.info("运行模式: LLM API 测试")
            # 运行 LLM 测试
            test_provider_models(config, logger)
        elif run_mode == "embedding":
            logger.info("运行模式: Embedding 功能测试")
            # 运行 Embedding 测试
            test_embedding_functionality(config, logger)
        elif run_mode == "both":
            logger.info("运行模式: 数据处理 + LLM 测试")
            # 先处理数据
            swe_loader, result = load_swe_bench_data(config, logger)
            # 再运行 LLM 测试
            test_provider_models(config, logger)
        elif run_mode == "all":
            logger.info("运行模式: 全部功能测试")
            # 运行所有测试
            swe_loader, result = load_swe_bench_data(config, logger)
            test_provider_models(config, logger)
            test_embedding_functionality(config, logger)
        else:
            logger.warning(f"未知运行模式: {run_mode}，默认运行数据处理")
            swe_loader, result = load_swe_bench_data(config, logger)
        
    except FileNotFoundError as e:
        logger.error(f"配置文件错误: {e}")
        print(f"❌ 配置文件错误: {e}")
        print("请确保在 src/config/ 目录下有 config.yaml 文件")
    except yaml.YAMLError as e:
        logger.error(f"YAML 解析错误: {e}")
        print(f"❌ YAML 解析错误: {e}")
        print("请检查 config.yaml 文件格式是否正确")
    except Exception as e:
        logger.error(f"程序执行错误: {e}")
        print(f"❌ 程序执行错误: {e}")
    finally:
        logger.info("SWE-bench 处理程序结束")
        logger.close()