#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOS统一性能监控系统

提供全面的性能监控和资源管理功能，包括：
- CPU、内存、GPU使用率监控
- 模型推理时间统计
- 性能预警机制
- 资源优化建议

Author: YOLOS Team
Version: 2.0.0
"""

import time
import psutil
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import statistics
from contextlib import contextmanager

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from .exceptions import (
    ErrorCode, SystemException, create_exception,
    exception_handler
)


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    gpu_percent: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_temperature: Optional[float] = None
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used_mb': self.memory_used_mb,
            'memory_available_mb': self.memory_available_mb,
            'gpu_percent': self.gpu_percent,
            'gpu_memory_percent': self.gpu_memory_percent,
            'gpu_memory_used_mb': self.gpu_memory_used_mb,
            'gpu_temperature': self.gpu_temperature,
            'disk_io_read_mb': self.disk_io_read_mb,
            'disk_io_write_mb': self.disk_io_write_mb,
            'network_sent_mb': self.network_sent_mb,
            'network_recv_mb': self.network_recv_mb
        }


@dataclass
class InferenceMetrics:
    """推理性能指标"""
    model_name: str
    inference_time_ms: float
    preprocessing_time_ms: float
    postprocessing_time_ms: float
    total_time_ms: float
    input_size: tuple
    batch_size: int
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'model_name': self.model_name,
            'inference_time_ms': self.inference_time_ms,
            'preprocessing_time_ms': self.preprocessing_time_ms,
            'postprocessing_time_ms': self.postprocessing_time_ms,
            'total_time_ms': self.total_time_ms,
            'input_size': self.input_size,
            'batch_size': self.batch_size,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class PerformanceAlert:
    """性能预警"""
    alert_type: str
    message: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'alert_type': self.alert_type,
            'message': self.message,
            'severity': self.severity,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'timestamp': self.timestamp.isoformat()
        }


class PerformanceMonitor:
    """统一性能监控器"""
    
    def __init__(
        self,
        monitoring_interval: float = 1.0,
        history_size: int = 1000,
        enable_gpu_monitoring: bool = True,
        logger_name: str = "yolos.performance"
    ):
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.enable_gpu_monitoring = enable_gpu_monitoring and GPU_AVAILABLE
        self.logger = logging.getLogger(logger_name)
        
        # 性能数据存储
        self.metrics_history: deque = deque(maxlen=history_size)
        self.inference_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        
        # 预警配置
        self.alert_thresholds = {
            'cpu_percent': {'high': 80.0, 'critical': 95.0},
            'memory_percent': {'high': 80.0, 'critical': 95.0},
            'gpu_percent': {'high': 85.0, 'critical': 98.0},
            'gpu_memory_percent': {'high': 85.0, 'critical': 95.0},
            'gpu_temperature': {'high': 80.0, 'critical': 90.0}
        }
        
        # 预警回调函数
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        
        # 监控控制
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # 初始化系统信息
        self._init_system_info()
    
    def _init_system_info(self):
        """初始化系统信息"""
        try:
            self.cpu_count = psutil.cpu_count()
            self.memory_total = psutil.virtual_memory().total / (1024**3)  # GB
            
            if self.enable_gpu_monitoring:
                try:
                    gpus = GPUtil.getGPUs()
                    self.gpu_count = len(gpus)
                    if gpus:
                        self.gpu_info = {
                            'name': gpus[0].name,
                            'memory_total': gpus[0].memoryTotal
                        }
                    else:
                        self.gpu_count = 0
                        self.gpu_info = None
                except Exception as e:
                    self.logger.warning(f"GPU信息获取失败: {e}")
                    self.enable_gpu_monitoring = False
                    self.gpu_count = 0
                    self.gpu_info = None
            else:
                self.gpu_count = 0
                self.gpu_info = None
                
        except Exception as e:
            raise create_exception(
                ErrorCode.SYSTEM_ERROR,
                f"系统信息初始化失败: {e}",
                {'component': 'PerformanceMonitor'}
            )
    
    @exception_handler(ErrorCode.SYSTEM_ERROR)
    def start_monitoring(self):
        """开始性能监控"""
        if self._monitoring:
            self.logger.warning("性能监控已在运行")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info("性能监控已启动")
    
    def stop_monitoring(self):
        """停止性能监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        self.logger.info("性能监控已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self._monitoring:
            try:
                metrics = self._collect_metrics()
                
                with self._lock:
                    self.metrics_history.append(metrics)
                
                # 检查预警
                self._check_alerts(metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"性能监控循环错误: {e}")
                time.sleep(self.monitoring_interval)
    
    @exception_handler(ErrorCode.SYSTEM_ERROR)
    def _collect_metrics(self) -> PerformanceMetrics:
        """收集性能指标"""
        # CPU和内存信息
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # 磁盘IO信息
        disk_io = psutil.disk_io_counters()
        disk_read_mb = disk_io.read_bytes / (1024**2) if disk_io else 0.0
        disk_write_mb = disk_io.write_bytes / (1024**2) if disk_io else 0.0
        
        # 网络IO信息
        net_io = psutil.net_io_counters()
        net_sent_mb = net_io.bytes_sent / (1024**2) if net_io else 0.0
        net_recv_mb = net_io.bytes_recv / (1024**2) if net_io else 0.0
        
        # GPU信息
        gpu_percent = None
        gpu_memory_percent = None
        gpu_memory_used_mb = None
        gpu_temperature = None
        
        if self.enable_gpu_monitoring:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # 使用第一个GPU
                    gpu_percent = gpu.load * 100
                    gpu_memory_percent = gpu.memoryUtil * 100
                    gpu_memory_used_mb = gpu.memoryUsed
                    gpu_temperature = gpu.temperature
            except Exception as e:
                self.logger.debug(f"GPU信息收集失败: {e}")
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024**2),
            memory_available_mb=memory.available / (1024**2),
            gpu_percent=gpu_percent,
            gpu_memory_percent=gpu_memory_percent,
            gpu_memory_used_mb=gpu_memory_used_mb,
            gpu_temperature=gpu_temperature,
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            network_sent_mb=net_sent_mb,
            network_recv_mb=net_recv_mb
        )
    
    def _check_alerts(self, metrics: PerformanceMetrics):
        """检查性能预警"""
        alerts = []
        
        # 检查各项指标
        for metric_name, thresholds in self.alert_thresholds.items():
            value = getattr(metrics, metric_name, None)
            if value is None:
                continue
            
            severity = None
            if value >= thresholds.get('critical', float('inf')):
                severity = 'critical'
            elif value >= thresholds.get('high', float('inf')):
                severity = 'high'
            
            if severity:
                alert = PerformanceAlert(
                    alert_type='resource_usage',
                    message=f"{metric_name}使用率过高: {value:.1f}%",
                    severity=severity,
                    metric_name=metric_name,
                    current_value=value,
                    threshold_value=thresholds[severity]
                )
                alerts.append(alert)
        
        # 触发预警回调
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"预警回调执行失败: {e}")
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """添加预警回调函数"""
        self.alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """移除预警回调函数"""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
    
    @contextmanager
    def measure_inference(
        self,
        model_name: str,
        input_size: tuple,
        batch_size: int = 1
    ):
        """推理性能测量上下文管理器"""
        start_time = time.perf_counter()
        preprocessing_time = 0.0
        postprocessing_time = 0.0
        
        class InferenceTimer:
            def __init__(self):
                self.preprocessing_start = None
                self.preprocessing_end = None
                self.postprocessing_start = None
                self.postprocessing_end = None
            
            def start_preprocessing(self):
                self.preprocessing_start = time.perf_counter()
            
            def end_preprocessing(self):
                if self.preprocessing_start:
                    self.preprocessing_end = time.perf_counter()
            
            def start_postprocessing(self):
                self.postprocessing_start = time.perf_counter()
            
            def end_postprocessing(self):
                if self.postprocessing_start:
                    self.postprocessing_end = time.perf_counter()
        
        timer = InferenceTimer()
        
        try:
            yield timer
        finally:
            end_time = time.perf_counter()
            total_time_ms = (end_time - start_time) * 1000
            
            # 计算各阶段时间
            if timer.preprocessing_start and timer.preprocessing_end:
                preprocessing_time = (timer.preprocessing_end - timer.preprocessing_start) * 1000
            
            if timer.postprocessing_start and timer.postprocessing_end:
                postprocessing_time = (timer.postprocessing_end - timer.postprocessing_start) * 1000
            
            inference_time_ms = total_time_ms - preprocessing_time - postprocessing_time
            
            # 记录推理指标
            metrics = InferenceMetrics(
                model_name=model_name,
                inference_time_ms=inference_time_ms,
                preprocessing_time_ms=preprocessing_time,
                postprocessing_time_ms=postprocessing_time,
                total_time_ms=total_time_ms,
                input_size=input_size,
                batch_size=batch_size
            )
            
            with self._lock:
                self.inference_history[model_name].append(metrics)
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """获取当前性能指标"""
        with self._lock:
            return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(
        self,
        duration_minutes: Optional[int] = None
    ) -> List[PerformanceMetrics]:
        """获取性能指标历史"""
        with self._lock:
            if duration_minutes is None:
                return list(self.metrics_history)
            
            cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
            return [
                m for m in self.metrics_history
                if m.timestamp >= cutoff_time
            ]
    
    def get_inference_statistics(
        self,
        model_name: str,
        duration_minutes: Optional[int] = None
    ) -> Dict[str, Any]:
        """获取推理性能统计"""
        with self._lock:
            history = self.inference_history.get(model_name, [])
            
            if duration_minutes is not None:
                cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
                history = [
                    m for m in history
                    if m.timestamp >= cutoff_time
                ]
            
            if not history:
                return {}
            
            # 计算统计信息
            total_times = [m.total_time_ms for m in history]
            inference_times = [m.inference_time_ms for m in history]
            
            return {
                'model_name': model_name,
                'sample_count': len(history),
                'total_time': {
                    'mean': statistics.mean(total_times),
                    'median': statistics.median(total_times),
                    'min': min(total_times),
                    'max': max(total_times),
                    'std': statistics.stdev(total_times) if len(total_times) > 1 else 0.0
                },
                'inference_time': {
                    'mean': statistics.mean(inference_times),
                    'median': statistics.median(inference_times),
                    'min': min(inference_times),
                    'max': max(inference_times),
                    'std': statistics.stdev(inference_times) if len(inference_times) > 1 else 0.0
                },
                'throughput_fps': len(history) / (sum(total_times) / 1000) if total_times else 0.0
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            'cpu_count': self.cpu_count,
            'memory_total_gb': self.memory_total,
            'gpu_count': self.gpu_count,
            'gpu_info': self.gpu_info,
            'gpu_monitoring_enabled': self.enable_gpu_monitoring
        }
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        current_metrics = self.get_current_metrics()
        recent_metrics = self.get_metrics_history(duration_minutes=60)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self.get_system_info(),
            'current_metrics': current_metrics.to_dict() if current_metrics else None,
            'recent_performance': {},
            'inference_statistics': {},
            'recommendations': []
        }
        
        # 计算最近性能统计
        if recent_metrics:
            cpu_values = [m.cpu_percent for m in recent_metrics]
            memory_values = [m.memory_percent for m in recent_metrics]
            
            report['recent_performance'] = {
                'cpu_usage': {
                    'mean': statistics.mean(cpu_values),
                    'max': max(cpu_values),
                    'min': min(cpu_values)
                },
                'memory_usage': {
                    'mean': statistics.mean(memory_values),
                    'max': max(memory_values),
                    'min': min(memory_values)
                }
            }
        
        # 推理统计
        for model_name in self.inference_history:
            stats = self.get_inference_statistics(model_name, duration_minutes=60)
            if stats:
                report['inference_statistics'][model_name] = stats
        
        # 生成优化建议
        report['recommendations'] = self._generate_recommendations(current_metrics, recent_metrics)
        
        return report
    
    def _generate_recommendations(
        self,
        current_metrics: Optional[PerformanceMetrics],
        recent_metrics: List[PerformanceMetrics]
    ) -> List[str]:
        """生成性能优化建议"""
        recommendations = []
        
        if not current_metrics:
            return recommendations
        
        # CPU使用率建议
        if current_metrics.cpu_percent > 80:
            recommendations.append("CPU使用率过高，建议优化算法或增加并行处理")
        
        # 内存使用建议
        if current_metrics.memory_percent > 80:
            recommendations.append("内存使用率过高，建议优化内存管理或增加内存")
        
        # GPU建议
        if current_metrics.gpu_percent and current_metrics.gpu_percent > 85:
            recommendations.append("GPU使用率过高，建议优化模型或使用模型量化")
        
        if current_metrics.gpu_temperature and current_metrics.gpu_temperature > 80:
            recommendations.append("GPU温度过高，建议检查散热或降低负载")
        
        # 推理性能建议
        for model_name, stats in [(name, self.get_inference_statistics(name, 60)) 
                                  for name in self.inference_history]:
            if stats and stats.get('total_time', {}).get('mean', 0) > 100:
                recommendations.append(f"模型{model_name}推理时间过长，建议优化模型或使用更快的硬件")
        
        return recommendations


# 全局性能监控器实例
global_performance_monitor = PerformanceMonitor()
global_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """获取全局性能监控器"""
    return global_performance_monitor


def get_performance_monitor() -> PerformanceMonitor:
    """获取全局性能监控器实例"""
    return global_monitor


# 装饰器：自动性能监控
def monitor_performance(model_name: str, input_size: tuple = None, batch_size: int = 1):
    """性能监控装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 尝试从参数中推断输入尺寸
            actual_input_size = input_size
            if actual_input_size is None and args:
                # 假设第一个参数是输入数据
                try:
                    if hasattr(args[0], 'shape'):
                        actual_input_size = args[0].shape
                    elif isinstance(args[0], (list, tuple)) and len(args[0]) > 0:
                        if hasattr(args[0][0], 'shape'):
                            actual_input_size = args[0][0].shape
                except:
                    actual_input_size = (0, 0)
            
            actual_input_size = actual_input_size or (0, 0)
            
            with global_performance_monitor.measure_inference(
                model_name, actual_input_size, batch_size
            ) as timer:
                timer.start_preprocessing()
                # 这里可以添加预处理逻辑
                timer.end_preprocessing()
                
                result = func(*args, **kwargs)
                
                timer.start_postprocessing()
                # 这里可以添加后处理逻辑
                timer.end_postprocessing()
                
                return result
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # 测试性能监控系统
    monitor = PerformanceMonitor(monitoring_interval=0.5)
    
    # 添加预警回调
    def alert_callback(alert: PerformanceAlert):
        print(f"性能预警: {alert.message} (严重程度: {alert.severity})")
    
    monitor.add_alert_callback(alert_callback)
    
    # 启动监控
    monitor.start_monitoring()
    
    try:
        # 模拟推理测试
        with monitor.measure_inference("test_model", (640, 480), 1) as timer:
            timer.start_preprocessing()
            time.sleep(0.01)  # 模拟预处理
            timer.end_preprocessing()
            
            time.sleep(0.05)  # 模拟推理
            
            timer.start_postprocessing()
            time.sleep(0.01)  # 模拟后处理
            timer.end_postprocessing()
        
        # 等待一段时间收集数据
        time.sleep(3)
        
        # 生成报告
        report = monitor.generate_performance_report()
        print("性能报告:")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        
    finally:
        monitor.stop_monitoring()