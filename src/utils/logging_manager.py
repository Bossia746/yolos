#!/usr/bin/env python3
"""
YOLOS统一日志管理系统
提供可追溯的DEBUG支持和详细的日志记录
"""

import os
import sys
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json
import traceback
import threading
from functools import wraps

class YOLOSLogger:
    """YOLOS统一日志管理器"""
    
    _instances: Dict[str, 'YOLOSLogger'] = {}
    _lock = threading.Lock()
    
    def __new__(cls, name: str = "yolos"):
        """单例模式，每个名称只创建一个实例"""
        with cls._lock:
            if name not in cls._instances:
                instance = super().__new__(cls)
                cls._instances[name] = instance
            return cls._instances[name]
    
    def __init__(self, name: str = "yolos"):
        if hasattr(self, '_initialized'):
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
        
        # 设置处理器
        self._setup_handlers()
        
        # 性能监控
        self.performance_data = {}
        self.start_times = {}
        
        self._initialized = True
    
    def _setup_handlers(self):
        """设置日志处理器"""
        # 清除现有处理器
        self.logger.handlers.clear()
        
        today = datetime.now().strftime("%Y%m%d")
        
        # 1. 系统日志处理器
        system_handler = logging.handlers.RotatingFileHandler(
            self.system_dir / f"yolos_{today}.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        system_handler.setLevel(logging.INFO)
        system_handler.setFormatter(self.formatter)
        self.logger.addHandler(system_handler)
        
        # 2. 错误日志处理器
        error_handler = logging.handlers.RotatingFileHandler(
            self.system_dir / f"error_{today}.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(self.formatter)
        self.logger.addHandler(error_handler)
        
        # 3. 调试日志处理器
        debug_handler = logging.handlers.RotatingFileHandler(
            self.debug_dir / f"{self.name}_debug_{today}.log",
            maxBytes=20*1024*1024,  # 20MB
            backupCount=3,
            encoding='utf-8'
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(self.formatter)
        self.logger.addHandler(debug_handler)
        
        # 4. 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '[%(levelname)s] [%(name)s] %(message)s'
        )
        console_handler.setFormatter(console_formatter)
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
    
    def end_timer(self, operation: str, **metrics):
        """结束计时并记录性能"""
        if operation in self.start_times:
            duration = (datetime.now() - self.start_times[operation]).total_seconds()
            del self.start_times[operation]
            self.log_performance(operation, duration, **metrics)
            return duration
        else:
            self.warning(f"Timer not found: {operation}")
            return 0.0
    
    def log_video_frame(self, frame_id: int, fps: float, objects_detected: int, processing_time: float):
        """记录视频帧处理信息"""
        self.debug(f"Frame {frame_id} processed", 
                  fps=fps, 
                  objects_detected=objects_detected, 
                  processing_time_ms=processing_time*1000)
    
    def log_detection_result(self, detection_type: str, objects: list, confidence_scores: list):
        """记录检测结果"""
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        self.info(f"Detection completed: {detection_type}",
                 objects_count=len(objects),
                 avg_confidence=round(avg_confidence, 3),
                 objects=objects[:5])  # 只记录前5个对象
    
    def log_training_progress(self, epoch: int, loss: float, accuracy: float, lr: float):
        """记录训练进度"""
        self.info(f"Training progress: Epoch {epoch}",
                 loss=round(loss, 6),
                 accuracy=round(accuracy, 4),
                 learning_rate=lr)
    
    def log_system_status(self, cpu_usage: float, memory_usage: float, gpu_usage: Optional[float] = None):
        """记录系统状态"""
        status = {
            'cpu_percent': cpu_usage,
            'memory_percent': memory_usage
        }
        if gpu_usage is not None:
            status['gpu_percent'] = gpu_usage
        
        self.debug("System status", **status)
    
    def log_api_request(self, endpoint: str, method: str, status_code: int, response_time: float):
        """记录API请求"""
        self.info(f"API {method} {endpoint}",
                 status_code=status_code,
                 response_time_ms=round(response_time*1000, 3))
    
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

def get_logger(name: str = "yolos") -> YOLOSLogger:
    """获取日志记录器实例"""
    return YOLOSLogger(name)

def log_function_call(logger_name: str = "yolos"):
    """函数调用日志装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name)
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