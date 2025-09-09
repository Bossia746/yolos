#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOS Performance Monitor
性能监控管理器 - 监控系统性能并提供优化建议
"""

import psutil
import time
import threading
import logging
from typing import Dict, List, Optional, Any, Callable
from collections import deque, defaultdict
from datetime import datetime, timedelta
import json
import yaml
from pathlib import Path
import gc
import tracemalloc
import asyncio
from concurrent.futures import ThreadPoolExecutor


class PerformanceMonitor:
    """性能监控管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # 性能指标存储
        self.metrics = {
            'cpu': deque(maxlen=1000),
            'memory': deque(maxlen=1000),
            'disk': deque(maxlen=1000),
            'network': deque(maxlen=1000),
            'gpu': deque(maxlen=1000),
            'inference_time': deque(maxlen=1000),
            'frame_rate': deque(maxlen=1000)
        }
        
        # 性能阈值
        self.thresholds = self.config.get('thresholds', {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'inference_time': 100.0,  # ms
            'frame_rate_min': 15.0
        })
        
        # 监控状态
        self.monitoring_active = False
        self.monitor_thread = None
        self.alerts = deque(maxlen=100)
        
        # 性能优化器
        self.optimizers = {
            'memory': self._optimize_memory,
            'cpu': self._optimize_cpu,
            'inference': self._optimize_inference
        }
        
        # 启动内存跟踪
        if self.config.get('memory_profiling', {}).get('enabled', False):
            tracemalloc.start()
        
        self.logger.info("Performance Monitor initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载性能监控配置"""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "performance_config.yaml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load performance config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'monitoring': {
                'enabled': True,
                'interval_seconds': 5,
                'detailed_logging': False
            },
            'thresholds': {
                'cpu_usage': 80.0,
                'memory_usage': 85.0,
                'disk_usage': 90.0,
                'inference_time': 100.0,
                'frame_rate_min': 15.0
            },
            'optimization': {
                'auto_optimize': True,
                'memory_cleanup_interval': 300,
                'gc_threshold_adjustment': True
            },
            'memory_profiling': {
                'enabled': False,
                'snapshot_interval': 60
            }
        }
    
    def start_monitoring(self):
        """启动性能监控"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """停止性能监控"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """监控循环"""
        interval = self.config.get('monitoring', {}).get('interval_seconds', 5)
        
        while self.monitoring_active:
            try:
                self._collect_metrics()
                self._check_thresholds()
                
                if self.config.get('optimization', {}).get('auto_optimize', True):
                    self._auto_optimize()
                
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def _collect_metrics(self):
        """收集性能指标"""
        timestamp = datetime.now()
        
        # CPU 使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics['cpu'].append({
            'timestamp': timestamp,
            'value': cpu_percent,
            'per_core': psutil.cpu_percent(percpu=True)
        })
        
        # 内存使用情况
        memory = psutil.virtual_memory()
        self.metrics['memory'].append({
            'timestamp': timestamp,
            'total': memory.total,
            'available': memory.available,
            'percent': memory.percent,
            'used': memory.used
        })
        
        # 磁盘使用情况
        disk = psutil.disk_usage('/')
        self.metrics['disk'].append({
            'timestamp': timestamp,
            'total': disk.total,
            'used': disk.used,
            'free': disk.free,
            'percent': (disk.used / disk.total) * 100
        })
        
        # 网络统计
        network = psutil.net_io_counters()
        self.metrics['network'].append({
            'timestamp': timestamp,
            'bytes_sent': network.bytes_sent,
            'bytes_recv': network.bytes_recv,
            'packets_sent': network.packets_sent,
            'packets_recv': network.packets_recv
        })
        
        # GPU 使用情况（如果可用）
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                self.metrics['gpu'].append({
                    'timestamp': timestamp,
                    'load': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'temperature': gpu.temperature
                })
        except ImportError:
            pass
    
    def _check_thresholds(self):
        """检查性能阈值"""
        if not self.metrics['cpu'] or not self.metrics['memory']:
            return
        
        latest_cpu = self.metrics['cpu'][-1]['value']
        latest_memory = self.metrics['memory'][-1]['percent']
        
        # 检查CPU阈值
        if latest_cpu > self.thresholds['cpu_usage']:
            self._create_alert('cpu_high', {
                'current': latest_cpu,
                'threshold': self.thresholds['cpu_usage']
            })
        
        # 检查内存阈值
        if latest_memory > self.thresholds['memory_usage']:
            self._create_alert('memory_high', {
                'current': latest_memory,
                'threshold': self.thresholds['memory_usage']
            })
    
    def _create_alert(self, alert_type: str, data: Dict[str, Any]):
        """创建性能告警"""
        alert = {
            'timestamp': datetime.now(),
            'type': alert_type,
            'data': data,
            'severity': self._get_alert_severity(alert_type, data)
        }
        
        self.alerts.append(alert)
        self.logger.warning(f"Performance alert: {alert_type} - {data}")
    
    def _get_alert_severity(self, alert_type: str, data: Dict[str, Any]) -> str:
        """获取告警严重程度"""
        if alert_type in ['cpu_high', 'memory_high']:
            current = data.get('current', 0)
            threshold = data.get('threshold', 100)
            
            if current > threshold * 1.2:  # 超过阈值20%
                return 'critical'
            elif current > threshold * 1.1:  # 超过阈值10%
                return 'high'
            else:
                return 'medium'
        
        return 'low'
    
    def _auto_optimize(self):
        """自动优化"""
        # 检查是否需要内存清理
        if self.metrics['memory']:
            latest_memory = self.metrics['memory'][-1]['percent']
            if latest_memory > 75:  # 内存使用超过75%
                self._optimize_memory()
    
    def _optimize_memory(self):
        """内存优化"""
        try:
            # 强制垃圾回收
            collected = gc.collect()
            
            # 调整GC阈值
            if self.config.get('optimization', {}).get('gc_threshold_adjustment', True):
                gc.set_threshold(700, 10, 10)
            
            self.logger.info(f"Memory optimization completed, collected {collected} objects")
        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")
    
    def _optimize_cpu(self):
        """CPU优化"""
        try:
            # 调整进程优先级
            process = psutil.Process()
            if process.nice() > 0:
                process.nice(0)  # 提高优先级
            
            self.logger.info("CPU optimization completed")
        except Exception as e:
            self.logger.error(f"CPU optimization failed: {e}")
    
    def _optimize_inference(self):
        """推理优化"""
        try:
            # 这里可以添加模型推理优化逻辑
            # 例如：批处理优化、模型量化等
            self.logger.info("Inference optimization completed")
        except Exception as e:
            self.logger.error(f"Inference optimization failed: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        summary = {
            'timestamp': datetime.now(),
            'status': 'healthy',
            'metrics': {},
            'alerts': len(self.alerts),
            'recommendations': []
        }
        
        # 计算各项指标的平均值
        for metric_name, metric_data in self.metrics.items():
            if not metric_data:
                continue
            
            if metric_name == 'cpu':
                values = [m['value'] for m in metric_data[-10:]]  # 最近10个值
                summary['metrics']['cpu_avg'] = sum(values) / len(values) if values else 0
            elif metric_name == 'memory':
                values = [m['percent'] for m in metric_data[-10:]]
                summary['metrics']['memory_avg'] = sum(values) / len(values) if values else 0
        
        # 生成建议
        if summary['metrics'].get('cpu_avg', 0) > 70:
            summary['recommendations'].append('Consider reducing CPU-intensive operations')
        
        if summary['metrics'].get('memory_avg', 0) > 80:
            summary['recommendations'].append('Consider memory optimization or increasing available memory')
        
        # 确定整体状态
        if len(self.alerts) > 5:
            summary['status'] = 'warning'
        
        critical_alerts = [a for a in self.alerts if a.get('severity') == 'critical']
        if critical_alerts:
            summary['status'] = 'critical'
        
        return summary
    
    def record_inference_time(self, inference_time_ms: float):
        """记录推理时间"""
        self.metrics['inference_time'].append({
            'timestamp': datetime.now(),
            'value': inference_time_ms
        })
        
        # 检查推理时间阈值
        if inference_time_ms > self.thresholds['inference_time']:
            self._create_alert('inference_slow', {
                'current': inference_time_ms,
                'threshold': self.thresholds['inference_time']
            })
    
    def record_frame_rate(self, fps: float):
        """记录帧率"""
        self.metrics['frame_rate'].append({
            'timestamp': datetime.now(),
            'value': fps
        })
        
        # 检查帧率阈值
        if fps < self.thresholds['frame_rate_min']:
            self._create_alert('frame_rate_low', {
                'current': fps,
                'threshold': self.thresholds['frame_rate_min']
            })
    
    def get_memory_profile(self) -> Optional[Dict[str, Any]]:
        """获取内存分析"""
        if not tracemalloc.is_tracing():
            return None
        
        try:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            profile = {
                'timestamp': datetime.now(),
                'total_memory': sum(stat.size for stat in top_stats),
                'top_allocations': []
            }
            
            for stat in top_stats[:10]:  # 前10个最大内存分配
                profile['top_allocations'].append({
                    'filename': stat.traceback.format()[0],
                    'size_mb': stat.size / 1024 / 1024,
                    'count': stat.count
                })
            
            return profile
        except Exception as e:
            self.logger.error(f"Failed to get memory profile: {e}")
            return None
    
    def export_metrics(self, filepath: str, format: str = 'json'):
        """导出性能指标"""
        try:
            data = {
                'export_time': datetime.now().isoformat(),
                'metrics': {},
                'alerts': list(self.alerts),
                'summary': self.get_performance_summary()
            }
            
            # 转换metrics为可序列化格式
            for metric_name, metric_data in self.metrics.items():
                data['metrics'][metric_name] = [
                    {k: v.isoformat() if isinstance(v, datetime) else v 
                     for k, v in item.items()}
                    for item in metric_data
                ]
            
            with open(filepath, 'w', encoding='utf-8') as f:
                if format.lower() == 'json':
                    json.dump(data, f, indent=2, ensure_ascii=False)
                elif format.lower() == 'yaml':
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            
            self.logger.info(f"Metrics exported to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")


# 全局实例
performance_monitor = PerformanceMonitor()

# 装饰器
def monitor_performance(metric_name: str = 'execution_time'):
    """性能监控装饰器"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = (time.time() - start_time) * 1000  # ms
                if metric_name == 'inference_time':
                    performance_monitor.record_inference_time(execution_time)
                else:
                    performance_monitor.metrics[metric_name].append({
                        'timestamp': datetime.now(),
                        'function': func.__name__,
                        'execution_time': execution_time
                    })
        return wrapper
    return decorator


# 异步版本的性能监控装饰器
def async_monitor_performance(metric_name: str = 'execution_time'):
    """异步性能监控装饰器"""
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                execution_time = (time.time() - start_time) * 1000  # ms
                if metric_name == 'inference_time':
                    performance_monitor.record_inference_time(execution_time)
                else:
                    performance_monitor.metrics[metric_name].append({
                        'timestamp': datetime.now(),
                        'function': func.__name__,
                        'execution_time': execution_time
                    })
        return wrapper
    return decorator