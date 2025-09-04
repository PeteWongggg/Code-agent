"""
日志管理模块

提供按时间戳分目录、分级别的日志管理功能

使用示例:
    from src.managers.log import Logger, create_logger
    
    # 基本使用
    logger = Logger("logs", "my_app")
    logger.info("这是一条信息")
    logger.error("这是一条错误")
    logger.close()
    
    # 使用上下文管理器
    with Logger("logs", "my_app") as logger:
        logger.info("自动管理资源")
    
    # 便捷函数
    logger = create_logger("logs", "my_app")
    logger.info("便捷创建")
    logger.close()
"""

from .logger import (
    Logger,
    create_logger,
    init_global_logger,
    get_global_logger,
    set_global_logger
)

__all__ = [
    "Logger",
    "create_logger", 
    "init_global_logger",
    "get_global_logger",
    "set_global_logger"
]

__version__ = "1.0.0"
__author__ = "Tokfinity Team"
__description__ = "时间戳分目录、分级别的日志管理器"
