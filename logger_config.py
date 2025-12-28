import logging
import logging.handlers
from pathlib import Path
from config import Config

def setup_logging():
    """配置结构化日志系统"""
    
    Config.ensure_directories()
    
    # 创建根logger
    root_logger = logging. getLogger()
    root_logger.setLevel(getattr(logging, Config.LOG_LEVEL))
    
    # 格式化器
    formatter = logging.Formatter(Config.LOG_FORMAT)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging. INFO)
    console_handler. setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 文件处理器（日志文件）
    log_file = Config.LOG_DIR / 'kg_builder.log'
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # 错误日志文件
    error_log = Config.LOG_DIR / 'errors.log'
    error_handler = logging.FileHandler(error_log, encoding='utf-8')
    error_handler.setLevel(logging. ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)
    
    root_logger.info("日志系统已初始化")
    
    return root_logger

# 快速获取logger
def get_logger(name: str) -> logging.Logger:
    """获取具名logger"""
    return logging.getLogger(name)