"""
日志管理器
支持按时间戳创建子文件夹，分级别存储日志文件
"""

import os
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path


"""
模块级自定义 NOTICE 日志级别（介于 INFO 与 WARNING 之间）
"""
NOTICE_LEVEL = 25
if not hasattr(logging, "NOTICE"):
    logging.addLevelName(NOTICE_LEVEL, "NOTICE")

    def notice(self, message, *args, **kwargs):
        if self.isEnabledFor(NOTICE_LEVEL):
            self._log(NOTICE_LEVEL, message, args, **kwargs)

    logging.Logger.notice = notice  # type: ignore[attr-defined]


class Logger:
    """
    日志管理器类
    
    功能：
    - 按时间戳创建子文件夹（年月日时分）
    - 分别创建 debug.log、info.log、notice.log、warning.log、error.log 五个文件
    - 提供标准的日志格式
    - 支持控制台和文件双重输出
    """
    
    def __init__(
        self,
        log_base_path: str,
        logger_name: str = "tokfinity_logger",
        console_output: bool = True,
        log_format: Optional[str] = None,
        instance_id: Optional[str] = None
    ):
        """
        初始化日志管理器
        
        Args:
            log_base_path: 日志基础路径
            logger_name: 日志器名称
            console_output: 是否同时输出到控制台
            log_format: 自定义日志格式
        """
        self.log_base_path = Path(log_base_path)
        self.logger_name = logger_name
        self.console_output = console_output
        self.instance_id = instance_id
        
        # 默认日志格式
        self.log_format = log_format or (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(funcName)s - %(message)s"
        )
        
        # 创建时间戳子文件夹
        self.log_dir = self._create_log_dir()
        
        # 存储文件处理器，便于后续管理
        self.file_handlers = {}
        
        # 初始化日志器
        self.logger = self._setup_logger()
    
    def _create_log_dir(self) -> Path:
        """
        创建日志文件夹
        
        Returns:
            Path: 创建的日志目录路径
        """
        # 如果提供了实例ID，则使用实例ID作为目录名；否则使用时间戳
        if self.instance_id:
            log_dir = self.log_base_path / self.instance_id
        else:
            # 生成时间戳格式：YYYYMMDDHHMM
            timestamp = datetime.now().strftime("%Y%m%d%H%M")
            log_dir = self.log_base_path / timestamp
        
        # 创建目录（如果不存在）
        log_dir.mkdir(parents=True, exist_ok=True)
        
        return log_dir
    
    def _setup_logger(self) -> logging.Logger:
        """
        设置日志器
        
        Returns:
            logging.Logger: 配置好的日志器
        """
        # 创建日志器
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(logging.DEBUG)  # 设置最低级别为DEBUG
        
        # 清除已有的处理器（避免重复）
        logger.handlers.clear()
        
        # 创建格式器
        formatter = logging.Formatter(self.log_format)
        
        # 添加控制台处理器
        if self.console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)  # 控制台显示INFO及以上级别
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # 添加文件处理器
        self._add_file_handlers(logger, formatter)
        
        return logger
    
    def _add_file_handlers(self, logger: logging.Logger, formatter: logging.Formatter):
        """
        添加分级别的文件处理器
        
        Args:
            logger: 日志器对象
            formatter: 日志格式器
        """
        # 定义日志级别和对应的文件名
        log_levels = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'notice': NOTICE_LEVEL,
            'warning': logging.WARNING,
            'error': logging.ERROR
        }
        
        for level_name, level_value in log_levels.items():
            # 创建文件处理器
            log_file = self.log_dir / f"{level_name}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            
            # 为不同级别设置过滤器
            if level_name == 'debug':
                # DEBUG文件记录DEBUG级别
                file_handler.addFilter(lambda record: record.levelno == logging.DEBUG)
            elif level_name == 'info':
                # INFO文件记录INFO级别
                file_handler.addFilter(lambda record: record.levelno == logging.INFO)
            elif level_name == 'notice':
                # NOTICE文件记录NOTICE级别
                file_handler.addFilter(lambda record: record.levelno == NOTICE_LEVEL)
            elif level_name == 'warning':
                # WARNING文件记录WARNING级别
                file_handler.addFilter(lambda record: record.levelno == logging.WARNING)
            elif level_name == 'error':
                # ERROR文件记录ERROR级别
                file_handler.addFilter(lambda record: record.levelno == logging.ERROR)
            
            logger.addHandler(file_handler)
            self.file_handlers[level_name] = file_handler
    
    def debug(self, message: str, *args, **kwargs):
        """记录DEBUG级别日志"""
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """记录INFO级别日志"""
        self.logger.info(message, *args, **kwargs)
    
    def notice(self, message: str, *args, **kwargs):
        """记录NOTICE级别日志"""
        self.logger.log(NOTICE_LEVEL, message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """记录WARNING级别日志"""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """记录ERROR级别日志"""
        self.logger.error(message, *args, **kwargs)
    
    def get_log_dir(self) -> str:
        """
        获取当前日志目录路径
        
        Returns:
            str: 日志目录的绝对路径
        """
        return str(self.log_dir.absolute())
    
    def get_log_files(self) -> dict:
        """
        获取所有日志文件路径
        
        Returns:
            dict: 包含各级别日志文件路径的字典
        """
        return {
            'debug': str(self.log_dir / "debug.log"),
            'info': str(self.log_dir / "info.log"),
            'notice': str(self.log_dir / "notice.log"),
            'warning': str(self.log_dir / "warning.log"),
            'error': str(self.log_dir / "error.log")
        }
    
    def close(self):
        """关闭所有文件处理器"""
        for handler in self.file_handlers.values():
            handler.close()
        
        # 从logger中移除处理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.close()
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"Logger(name={self.logger_name}, dir={self.log_dir})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (
            f"Logger("
            f"name='{self.logger_name}', "
            f"base_path='{self.log_base_path}', "
            f"log_dir='{self.log_dir}', "
            f"console_output={self.console_output})"
        )


# 便捷函数
def create_logger(
    log_base_path: str,
    logger_name: str = "tokfinity_logger",
    console_output: bool = True
) -> Logger:
    """
    创建日志管理器的便捷函数
    
    Args:
        log_base_path: 日志基础路径
        logger_name: 日志器名称
        console_output: 是否输出到控制台
        
    Returns:
        Logger: 日志管理器实例
    """
    return Logger(
        log_base_path=log_base_path,
        logger_name=logger_name,
        console_output=console_output
    )


# 全局日志管理器实例（可选）
_global_logger: Optional[Logger] = None


def get_global_logger() -> Optional[Logger]:
    """获取全局日志管理器"""
    return _global_logger


def set_global_logger(logger: Logger):
    """设置全局日志管理器"""
    global _global_logger
    _global_logger = logger


def init_global_logger(log_base_path: str, logger_name: str = "global_logger") -> Logger:
    """
    初始化全局日志管理器
    
    Args:
        log_base_path: 日志基础路径
        logger_name: 日志器名称
        
    Returns:
        Logger: 全局日志管理器实例
    """
    global _global_logger
    _global_logger = Logger(log_base_path, logger_name)
    return _global_logger
