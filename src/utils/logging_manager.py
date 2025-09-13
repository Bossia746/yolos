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
from typing import Optional, Dict, Any, Union, List
import json
import traceback
import threading
from functools import wraps
import time
import psutil
import platform
from collections import defaultdict, deque
import uuid
from dataclasses import dataclass, asdict
from enum import Enum

class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class StructuredLogEntry:
    """结构化日志条目"""
    timestamp: str
    level: str
    logger_name: str
    message: str
    module: str
    function: str
    line_number: int
    thread_id: str
    process_id: int
    session_id: str
    context: Dict[str, Any]
    performance_metrics: Optional[Dict[str, Any]] = None
    error_info: Optional[Dict[str, Any]] = None
    
class StructuredFormatter(logging.Formatter):
    """结构化日志格式化器"""
    
    def format(self, record: logging.LogRecord) -> str:
        # 提取结构化数据
        structured_data = getattr(record, 'structured_data', {})
        
        # 创建结构化日志条目
        entry = StructuredLogEntry(
            timestamp=datetime.fromtimestamp(record.created).isoformat(),
            level=record.levelname,
            logger_name=record.name,
            message=record.getMessage(),
            module=record.module,
            function=record.funcName,
            line_number=record.lineno,
            thread_id=str(threading.current_thread().ident),
            process_id=os.getpid(),
            session_id=structured_data.get('session_id', ''),
            context=structured_data.get('context', {}),
            performance_metrics=structured_data.get('performance_metrics'),
            error_info=structured_data.get('error_info')
        )
        
        return json.dumps(asdict(entry), ensure_ascii=False, default=str)

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.active_timers = {}
        self.system_metrics = deque(maxlen=1000)  # 保留最近1000条系统指标
        self._lock = threading.Lock()
        
    def start_timer(self, operation: str) -> str:
        """开始计时，返回计时器ID"""
        timer_id = f"{operation}_{uuid.uuid4().hex[:8]}"
        with self._lock:
            self.active_timers[timer_id] = {
                'operation': operation,
                'start_time': time.perf_counter(),
                'start_timestamp': datetime.now().isoformat()
            }
        return timer_id
    
    def end_timer(self, timer_id: str, **additional_metrics) -> Optional[Dict[str, Any]]:
        """结束计时并返回性能指标"""
        end_time = time.perf_counter()
        
        with self._lock:
            if timer_id not in self.active_timers:
                return None
                
            timer_info = self.active_timers.pop(timer_id)
            duration = end_time - timer_info['start_time']
            
            metrics = {
                'operation': timer_info['operation'],
                'duration_ms': round(duration * 1000, 3),
                'start_timestamp': timer_info['start_timestamp'],
                'end_timestamp': datetime.now().isoformat(),
                **additional_metrics
            }
            
            self.metrics[timer_info['operation']].append(metrics)
            return metrics
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """收集系统性能指标"""
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_mb': round(memory.used / 1024 / 1024, 2),
                'memory_available_mb': round(memory.available / 1024 / 1024, 2),
                'disk_percent': disk.percent,
                'disk_used_gb': round(disk.used / 1024 / 1024 / 1024, 2),
                'disk_free_gb': round(disk.free / 1024 / 1024 / 1024, 2)
            }
            
            # 尝试获取GPU信息
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    metrics.update({
                        'gpu_percent': round(gpu.load * 100, 2),
                        'gpu_memory_percent': round(gpu.memoryUtil * 100, 2),
                        'gpu_memory_used_mb': round(gpu.memoryUsed, 2),
                        'gpu_memory_total_mb': round(gpu.memoryTotal, 2),
                        'gpu_temperature': gpu.temperature
                    })
            except ImportError:
                pass
            
            with self._lock:
                self.system_metrics.append(metrics)
            
            return metrics
        except Exception:
            return {}
    
    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """获取操作统计信息"""
        with self._lock:
            if operation not in self.metrics:
                return {}
                
            durations = [m['duration_ms'] for m in self.metrics[operation]]
            return {
                'operation': operation,
                'count': len(durations),
                'avg_duration_ms': round(sum(durations) / len(durations), 3),
                'min_duration_ms': min(durations),
                'max_duration_ms': max(durations),
                'total_duration_ms': round(sum(durations), 3)
            }

class YOLOSLogger:
    """YOLOS增强统一日志管理器
    
    新增功能:
    - 结构化日志记录
    - 性能监控集成
    - 实时系统指标收集
    - 智能日志分析
    - 多格式输出支持
    """
    
    _instances: Dict[str, 'YOLOSLogger'] = {}
    _lock = threading.Lock()
    _performance_monitor = PerformanceMonitor()
    _session_id = uuid.uuid4().hex[:8]
    
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
        self.structured_dir = self.log_dir / "structured"
        self.metrics_dir = self.log_dir / "metrics"
        
        for dir_path in [self.system_dir, self.debug_dir, self.performance_dir, 
                        self.structured_dir, self.metrics_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 设置日志格式
        self.formatter = logging.Formatter(
            '[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(name)s] [%(funcName)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 结构化日志格式化器
        self.structured_formatter = StructuredFormatter()
        
        # 设置处理器
        self._setup_handlers()
        
        # 性能监控（兼容旧接口）
        self.performance_data = {}
        self.start_times = {}
        
        # 系统指标收集定时器
        self._metrics_collection_enabled = True
        self._start_metrics_collection()
        
        self._initialized = True
    
    def _start_metrics_collection(self):
        """启动系统指标收集"""
        def collect_metrics():
            while self._metrics_collection_enabled:
                try:
                    metrics = self._performance_monitor.collect_system_metrics()
                    if metrics:
                        self._log_system_metrics(metrics)
                    time.sleep(30)  # 每30秒收集一次
                except Exception as e:
                    self.error(f"系统指标收集失败: {e}")
                    time.sleep(60)  # 出错时等待更长时间
        
        metrics_thread = threading.Thread(target=collect_metrics, daemon=True)
        metrics_thread.start()
    
    def _log_system_metrics(self, metrics: Dict[str, Any]):
        """记录系统指标到文件"""
        today = datetime.now().strftime("%Y%m%d")
        metrics_file = self.metrics_dir / f"system_metrics_{today}.jsonl"
        
        try:
            with open(metrics_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + '\n')
        except Exception as e:
            self.error(f"系统指标记录失败: {e}")
    
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
        
        # 4. 结构化日志处理器
        structured_handler = logging.handlers.RotatingFileHandler(
            self.structured_dir / f"structured_{today}.jsonl",
            maxBytes=50*1024*1024,  # 50MB
            backupCount=10,
            encoding='utf-8'
        )
        structured_handler.setLevel(logging.DEBUG)
        structured_handler.setFormatter(self.structured_formatter)
        self.logger.addHandler(structured_handler)
        
        # 5. 性能日志处理器
        performance_handler = logging.handlers.RotatingFileHandler(
            self.performance_dir / f"performance_{today}.jsonl",
            maxBytes=30*1024*1024,  # 30MB
            backupCount=7,
            encoding='utf-8'
        )
        performance_handler.setLevel(logging.INFO)
        performance_handler.setFormatter(self.structured_formatter)
        # 只记录性能相关的日志
        performance_handler.addFilter(lambda record: hasattr(record, 'structured_data') and 
                                    record.structured_data.get('performance_metrics'))
        self.logger.addHandler(performance_handler)
        
        # 6. 控制台处理器
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
        error_info = None
        if exception:
            message += f" | Exception: {str(exception)}"
            error_info = {
                'exception_type': type(exception).__name__,
                'exception_message': str(exception),
                'traceback': traceback.format_exc()
            }
            kwargs['error_info'] = error_info
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """严重错误日志"""
        error_info = None
        if exception:
            message += f" | Critical Exception: {str(exception)}"
            error_info = {
                'exception_type': type(exception).__name__,
                'exception_message': str(exception),
                'traceback': traceback.format_exc(),
                'severity': 'critical'
            }
            kwargs['error_info'] = error_info
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
            'process': os.getpid(),
            'session_id': self._session_id
        }
        
        # 提取特殊字段
        performance_metrics = kwargs.pop('performance_metrics', None)
        error_info = kwargs.pop('error_info', None)
        
        # 合并额外信息到上下文
        if kwargs:
            context.update(kwargs)
        
        # 创建结构化数据
        structured_data = {
            'session_id': self._session_id,
            'context': context,
            'performance_metrics': performance_metrics,
            'error_info': error_info
        }
        
        # 创建日志记录并添加结构化数据
        record = self.logger.makeRecord(
            self.logger.name, level, frame.f_code.co_filename,
            frame.f_lineno, message, (), None, frame.f_code.co_name
        )
        record.structured_data = structured_data
        
        # 为传统格式添加上下文信息
        if kwargs:
            message += f" | Context: {json.dumps(context, ensure_ascii=False, default=str)}"
        
        # 发送日志记录
        record.msg = message
        self.logger.handle(record)
    
    def log_performance(self, operation: str, duration: float, **metrics):
        """记录性能数据（兼容旧接口）"""
        perf_metrics = {
            'operation': operation,
            'duration_ms': round(duration * 1000, 3),
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        # 使用结构化日志记录性能数据
        self.info(f"Performance: {operation} took {duration*1000:.3f}ms", 
                 performance_metrics=perf_metrics, **metrics)
    
    def start_timer(self, operation: str) -> str:
        """开始计时（增强版本）"""
        # 使用新的性能监控器
        timer_id = self._performance_monitor.start_timer(operation)
        
        # 兼容旧接口
        self.start_times[operation] = datetime.now()
        
        self.debug(f"Timer started: {operation}", timer_id=timer_id)
        return timer_id
    
    def end_timer(self, operation: str, timer_id: Optional[str] = None, **metrics) -> float:
        """结束计时并记录性能（增强版本）"""
        duration = 0.0
        
        # 使用新的性能监控器
        if timer_id:
            perf_metrics = self._performance_monitor.end_timer(timer_id, **metrics)
            if perf_metrics:
                duration = perf_metrics['duration_ms'] / 1000
                self.info(f"Performance: {operation} completed", 
                         performance_metrics=perf_metrics)
        
        # 兼容旧接口
        elif operation in self.start_times:
            duration = (datetime.now() - self.start_times[operation]).total_seconds()
            del self.start_times[operation]
            self.log_performance(operation, duration, **metrics)
        else:
            self.warning(f"Timer not found: {operation}")
        
        return duration
    
    def get_performance_stats(self, operation: str) -> Dict[str, Any]:
        """获取操作性能统计"""
        return self._performance_monitor.get_operation_stats(operation)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """获取当前系统指标"""
        return self._performance_monitor.collect_system_metrics()
    
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
            'stack_trace': traceback.format_stack(),
            'session_id': self._session_id,
            'system_info': {
                'platform': platform.system(),
                'python_version': platform.python_version(),
                'process_id': os.getpid(),
                'thread_id': threading.current_thread().ident
            }
        }
        
        try:
            with open(snapshot_file, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2, default=str)
            self.debug(f"Debug snapshot created: {snapshot_file}", 
                      snapshot_file=str(snapshot_file), operation=operation)
        except Exception as e:
            self.error(f"Failed to create debug snapshot: {e}", exception=e)
    
    def log_structured(self, level: Union[int, str], message: str, 
                      event_type: str = "general", **fields):
        """记录结构化日志"""
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        structured_fields = {
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            **fields
        }
        
        self._log_with_context(level, message, **structured_fields)
    
    def log_event(self, event_name: str, event_data: Dict[str, Any], 
                  level: str = "INFO"):
        """记录事件日志"""
        self.log_structured(level, f"Event: {event_name}", 
                           event_type="event", 
                           event_name=event_name, 
                           event_data=event_data)
    
    def log_metric(self, metric_name: str, value: Union[int, float], 
                   unit: str = "", tags: Optional[Dict[str, str]] = None):
        """记录指标日志"""
        metric_data = {
            'metric_name': metric_name,
            'value': value,
            'unit': unit,
            'tags': tags or {}
        }
        
        self.log_structured("INFO", f"Metric: {metric_name}={value}{unit}",
                           event_type="metric", 
                           metric_data=metric_data)
    
    def log_audit(self, action: str, user: str, resource: str, 
                  result: str = "success", **details):
        """记录审计日志"""
        audit_data = {
            'action': action,
            'user': user,
            'resource': resource,
            'result': result,
            'details': details
        }
        
        self.log_structured("INFO", f"Audit: {user} {action} {resource} - {result}",
                           event_type="audit",
                           audit_data=audit_data)
    
    def analyze_performance_trends(self, operation: str, 
                                  time_window_hours: int = 24) -> Dict[str, Any]:
        """分析性能趋势"""
        stats = self.get_performance_stats(operation)
        if not stats:
            return {}
        
        # 获取系统指标趋势
        current_time = datetime.now()
        window_start = current_time - timedelta(hours=time_window_hours)
        
        recent_metrics = [m for m in self._performance_monitor.system_metrics 
                         if datetime.fromisoformat(m['timestamp']) >= window_start]
        
        if recent_metrics:
            avg_cpu = sum(m.get('cpu_percent', 0) for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.get('memory_percent', 0) for m in recent_metrics) / len(recent_metrics)
            
            analysis = {
                'operation': operation,
                'time_window_hours': time_window_hours,
                'performance_stats': stats,
                'system_trends': {
                    'avg_cpu_percent': round(avg_cpu, 2),
                    'avg_memory_percent': round(avg_memory, 2),
                    'samples_count': len(recent_metrics)
                },
                'recommendations': self._generate_performance_recommendations(stats, avg_cpu, avg_memory)
            }
            
            self.log_structured("INFO", f"Performance analysis for {operation}",
                               event_type="analysis",
                               analysis_data=analysis)
            
            return analysis
        
        return {'operation': operation, 'performance_stats': stats}
    
    def _generate_performance_recommendations(self, stats: Dict[str, Any], 
                                            avg_cpu: float, avg_memory: float) -> List[str]:
        """生成性能优化建议"""
        recommendations = []
        
        if stats.get('avg_duration_ms', 0) > 1000:
            recommendations.append("考虑优化算法或使用异步处理")
        
        if avg_cpu > 80:
            recommendations.append("CPU使用率过高，考虑优化计算密集型操作")
        
        if avg_memory > 85:
            recommendations.append("内存使用率过高，检查内存泄漏或优化数据结构")
        
        if stats.get('max_duration_ms', 0) > stats.get('avg_duration_ms', 0) * 3:
            recommendations.append("存在性能异常值，建议检查特定场景下的性能问题")
        
        return recommendations
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """清理旧日志文件"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for log_dir in [self.system_dir, self.debug_dir, self.performance_dir, 
                       self.structured_dir, self.metrics_dir]:
            try:
                for log_file in log_dir.glob("*.log*"):
                    if log_file.stat().st_mtime < cutoff_date.timestamp():
                        log_file.unlink()
                        self.debug(f"Cleaned up old log file: {log_file}")
            except Exception as e:
                self.error(f"Failed to cleanup logs in {log_dir}: {e}", exception=e)
    
    def stop_metrics_collection(self):
        """停止系统指标收集"""
        self._metrics_collection_enabled = False
        self.info("系统指标收集已停止")

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

# 兼容性类 - 为了向后兼容
class LoggingManager:
    """日志管理器 - 兼容性包装"""
    
    def __init__(self, name: str = "yolos"):
        self.logger = get_logger(name)
    
    def get_logger(self):
        return self.logger
    
    def debug(self, message: str, **kwargs):
        return self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        return self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        return self.logger.warning(message, **kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        return self.logger.error(message, exception, **kwargs)

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