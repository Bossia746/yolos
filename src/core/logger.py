#!/usr/bin/env python3
"""
YOLOS统一日志系统
消除重复日志管理器，提供单一日志入口
"""

import logging
import logging.handlers
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import threading
import json
import traceback
from functools import wraps

class YOLOSLogger:
    """
    YOLOS统一日志管理器
    
    设计原则:
    1. 单例模式 - 每个模块只有一个日志实例
    2. 统一格式 - 所有日志使用相同格式
    3. 分级管理 - 支持不同级别的日志输出
    4. 性能监控 - 内置性能日志功能
    5. 线程安全 - 支持多线程环境
    """
    
    _instances: Dict[str, 'YOLOSLogger'] = {}
    _lock = threading.RLock()
    _initialized = False
    
    # 日志级别映射
    LEVEL_MAP = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    def __new__(cls, name: str = "yolos"):
        """单例模式实现"""
        with cls._lock:
            if name not in cls._instances:
                instance = super().__new__(cls)
                cls._instances[name] = instance
            return cls._instances[name]
    
    def __init__(self, name: str = "yolos"):
        """初始化日志器"""
        if hasattr(self, '_logger_initialized'):
            return
        
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # 创建日志目录
        self.log_dir = Path("logs")
        self.system_dir = self.log_dir / "system"
        self.debug_dir = self.log_dir / "debug"
        self.performance_dir = self.log_dir / "performance"
        
        for dir_path in [self.system_dir, self.debug_dir, self.performance_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 设置日志格式
        self.formatter = logging.Formatter(
            '[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(name)s] [%(funcName)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台格式（简化）
        self.console_formatter = logging.Formatter(
            '[%(levelname)s] [%(name)s] %(message)s'
        )
        
        # 设置处理器
        self._setup_handlers()
        
        # 性能监控
        self.performance_data = {}
        self.start_times = {}
        
        self._logger_initialized = True
    
    def _setup_handlers(self):
        """设置日志处理器"""
        # 清除现有处理器
        self.logger.handlers.clear()
        
        today = datetime.now().strftime("%Y%m%d")
        
        # 1. 系统日志处理器 (INFO及以上)
        system_handler = logging.handlers.RotatingFileHandler(
            self.system_dir / f"yolos_{today}.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        system_handler.setLevel(logging.INFO)
        system_handler.setFormatter(self.formatter)
        self.logger.addHandler(system_handler)
        
        # 2. 错误日志处理器 (ERROR及以上)
        error_handler = logging.handlers.RotatingFileHandler(
            self.system_dir / f"error_{today}.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(self.formatter)
        self.logger.addHandler(error_handler)
        
        # 3. 调试日志处理器 (DEBUG及以上)
        debug_handler = logging.handlers.RotatingFileHandler(
            self.debug_dir / f"{self.name}_debug_{today}.log",
            maxBytes=20*1024*1024,  # 20MB
            backupCount=3,
            encoding='utf-8'
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(self.formatter)
        self.logger.addHandler(debug_handler)
        
        # 4. 控制台处理器 (INFO及以上)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self.console_formatter)
        self.logger.addHandler(console_handler)
    
    def debug(self, message: str, **kwargs):
        """调试日志"""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """信息日志"""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """警告日志"""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """错误日志"""
        if exception:
            message += f" | Exception: {str(exception)}"
            kwargs['traceback'] = traceback.format_exc()
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """严重错误日志"""
        if exception:
            message += f" | Critical Exception: {str(exception)}"
            kwargs['traceback'] = traceback.format_exc()
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    def _log_with_context(self, level: int, message: str, **kwargs):
        """带上下文的日志记录"""
        # 添加调用栈信息
        frame = sys._getframe(2)
        context = {
            'file': frame.f_code.co_filename,
            'function': frame.f_code.co_name,
            'line': frame.f_lineno,
            'thread': threading.current_thread().name,
            'process': os.getpid()
        }
        
        # 合并额外信息
        if kwargs:
            context.update(kwargs)
            # 只在DEBUG级别显示完整上下文
            if level >= logging.INFO and kwargs:
                # 简化上下文信息
                simple_context = {k: v for k, v in kwargs.items() 
                                if k not in ['file', 'function', 'line', 'thread', 'process', 'traceback']}
                if simple_context:
                    message += f" | {json.dumps(simple_context, ensure_ascii=False)}"
            elif level == logging.DEBUG:
                message += f" | Context: {json.dumps(context, ensure_ascii=False)}"
        
        self.logger.log(level, message)
    
    def log_performance(self, operation: str, duration: float, **metrics):
        """记录性能数据"""
        today = datetime.now().strftime("%Y%m%d")
        perf_file = self.performance_dir / f"performance_{today}.log"
        
        perf_data = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'duration_ms': round(duration * 1000, 3),
            'metrics': metrics
        }
        
        with open(perf_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(perf_data, ensure_ascii=False) + '\n')
        
        self.info(f"Performance: {operation} took {duration*1000:.3f}ms", **metrics)
    
    def start_timer(self, operation: str):
        """开始计时"""
        self.start_times[operation] = datetime.now()
        self.debug(f"Timer started: {operation}")
    
    def end_timer(self, operation: str, **metrics) -> float:
        """结束计时并记录性能"""
        if operation in self.start_times:
            duration = (datetime.now() - self.start_times[operation]).total_seconds()
            del self.start_times[operation]
            self.log_performance(operation, duration, **metrics)
            return duration
        else:
            self.warning(f"Timer not found: {operation}")
            return 0.0
    
    def set_level(self, level: str):
        """设置日志级别"""
        if level.upper() in self.LEVEL_MAP:
            self.logger.setLevel(self.LEVEL_MAP[level.upper()])
            self.info(f"日志级别设置为: {level.upper()}")
        else:
            self.warning(f"无效的日志级别: {level}")
    
    def add_file_handler(self, file_path: str, level: str = "INFO"):
        """添加文件处理器"""
        handler = logging.FileHandler(file_path, encoding='utf-8')
        handler.setLevel(self.LEVEL_MAP.get(level.upper(), logging.INFO))
        handler.setFormatter(self.formatter)
        self.logger.addHandler(handler)
        self.info(f"添加文件处理器: {file_path}")
    
    def remove_console_handler(self):
        """移除控制台处理器"""
        for handler in self.logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                self.logger.removeHandler(handler)
                self.info("控制台处理器已移除")
    
    @classmethod
    def get_logger(cls, name: str = "yolos") -> 'YOLOSLogger':
        """获取日志记录器实例"""
        return cls(name)
    
    @classmethod
    def configure_from_config(cls, config):
        """从配置对象配置日志系统"""
        if hasattr(config, 'logging_config'):
            logging_config = config.logging_config
            
            # 设置全局日志级别
            root_logger = cls.get_logger()
            root_logger.set_level(logging_config.level)
            
            # 配置模块日志级别
            for module, level in logging_config.modules.items():
                module_logger = cls.get_logger(module)
                module_logger.set_level(level)
            
            # 控制台输出设置
            if not logging_config.console_output:
                root_logger.remove_console_handler()
    
    def create_debug_snapshot(self, operation: str, data: Dict[str, Any]):
        """创建调试快照"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        snapshot_file = self.debug_dir / f"snapshot_{operation}_{timestamp}.json"
        
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'data': data,
            'stack_trace': traceback.format_stack()
        }
        
        try:
            with open(snapshot_file, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2, default=str)
            self.debug(f"Debug snapshot created: {snapshot_file}")
        except Exception as e:
            self.error(f"Failed to create debug snapshot: {e}")

# 装饰器函数
def log_function_call(logger_name: str = "yolos"):
    """函数调用日志装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = YOLOSLogger.get_logger(logger_name)
            func_name = f"{func.__module__}.{func.__name__}"
            
            # 记录函数调用
            logger.debug(f"Function called: {func_name}",
                        args_count=len(args),
                        kwargs_keys=list(kwargs.keys()))
            
            # 执行函数并计时
            logger.start_timer(func_name)
            try:
                result = func(*args, **kwargs)
                duration = logger.end_timer(func_name, success=True)
                logger.debug(f"Function completed: {func_name}",
                           duration_ms=duration*1000)
                return result
            except Exception as e:
                duration = logger.end_timer(func_name, success=False, error=str(e))
                logger.error(f"Function failed: {func_name}", exception=e,
                           duration_ms=duration*1000)
                raise
        return wrapper
    return decorator

def log_class_methods(logger_name: str = "yolos"):
    """类方法日志装饰器"""
    def decorator(cls):
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if callable(attr) and not attr_name.startswith('_'):
                setattr(cls, attr_name, log_function_call(logger_name)(attr))
        return cls
    return decorator

# 便捷函数
def get_logger(name: str = "yolos") -> YOLOSLogger:
    """获取日志记录器实例"""
    return YOLOSLogger.get_logger(name)

def configure_logging(config=None):
    """配置日志系统"""
    if config:
        YOLOSLogger.configure_from_config(config)

# 创建默认日志记录器
default_logger = get_logger()

# 导出常用函数
debug = default_logger.debug
info = default_logger.info
warning = default_logger.warning
error = default_logger.error
critical = default_logger.critical

if __name__ == "__main__":
    # 测试日志系统
    print("🔍 测试YOLOS日志系统...")
    
    logger = get_logger("test")
    
    logger.info("日志系统测试开始")
    logger.debug("这是调试信息", test_param="test_value")
    logger.warning("这是警告信息")
    
    try:
        raise ValueError("测试异常")
    except Exception as e:
        logger.error("捕获到异常", exception=e)
    
    # 测试性能日志
    logger.start_timer("test_operation")
    import time
    time.sleep(0.1)
    logger.end_timer("test_operation", test_metric=100)
    
    # 测试调试快照
    logger.create_debug_snapshot("test_snapshot", {
        "test_data": "test_value",
        "numbers": [1, 2, 3, 4, 5]
    })
    
    logger.info("日志系统测试完成")
    print("✅ 日志系统测试完成")