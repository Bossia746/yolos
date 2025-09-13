#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOS性能优化器
提供系统级性能优化和稳定性增强功能
"""

import gc
import os
import sys
import time
import psutil
import threading
import multiprocessing
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import weakref

# 尝试导入GPU相关库
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class OptimizationLevel(Enum):
    """优化级别"""
    CONSERVATIVE = "conservative"  # 保守优化
    BALANCED = "balanced"         # 平衡优化
    AGGRESSIVE = "aggressive"     # 激进优化

class ResourceType(Enum):
    """资源类型"""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"

@dataclass
class SystemMetrics:
    """系统指标"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    memory_available: int = 0
    gpu_usage: float = 0.0
    gpu_memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_io: Dict[str, int] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class OptimizationConfig:
    """优化配置"""
    level: OptimizationLevel = OptimizationLevel.BALANCED
    enable_memory_optimization: bool = True
    enable_cpu_optimization: bool = True
    enable_gpu_optimization: bool = True
    enable_cache_optimization: bool = True
    enable_threading_optimization: bool = True
    max_workers: Optional[int] = None
    memory_threshold: float = 0.8  # 内存使用阈值
    cpu_threshold: float = 0.9     # CPU使用阈值
    gc_frequency: int = 100        # 垃圾回收频率
    cache_size_limit: int = 1000   # 缓存大小限制

class MemoryManager:
    """内存管理器"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.cache = {}
        self.cache_access_count = {}
        self.cache_lock = threading.Lock()
        
    def get_memory_info(self) -> Dict[str, Any]:
        """获取内存信息"""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percentage': memory.percent,
            'free': memory.free
        }
    
    def optimize_memory(self):
        """优化内存使用"""
        memory_info = self.get_memory_info()
        
        if memory_info['percentage'] > self.config.memory_threshold * 100:
            logger.warning(f"内存使用率过高: {memory_info['percentage']:.1f}%")
            
            # 强制垃圾回收
            self.force_garbage_collection()
            
            # 清理缓存
            self.cleanup_cache()
            
            # 如果有PyTorch，清理GPU缓存
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def force_garbage_collection(self):
        """强制垃圾回收"""
        collected = gc.collect()
        logger.debug(f"垃圾回收完成，回收对象数: {collected}")
    
    def cleanup_cache(self, force: bool = False):
        """清理缓存"""
        with self.cache_lock:
            if force or len(self.cache) > self.config.cache_size_limit:
                # 按访问次数排序，删除最少使用的项
                if self.cache:
                    sorted_items = sorted(
                        self.cache_access_count.items(),
                        key=lambda x: x[1]
                    )
                    
                    # 删除一半最少使用的项
                    items_to_remove = len(sorted_items) // 2
                    for key, _ in sorted_items[:items_to_remove]:
                        if key in self.cache:
                            del self.cache[key]
                        if key in self.cache_access_count:
                            del self.cache_access_count[key]
                    
                    logger.debug(f"清理缓存，删除 {items_to_remove} 个项目")
    
    def cache_get(self, key: str) -> Any:
        """从缓存获取数据"""
        with self.cache_lock:
            if key in self.cache:
                self.cache_access_count[key] = self.cache_access_count.get(key, 0) + 1
                return self.cache[key]
            return None
    
    def cache_set(self, key: str, value: Any):
        """设置缓存数据"""
        with self.cache_lock:
            self.cache[key] = value
            self.cache_access_count[key] = 1
            
            # 检查缓存大小
            if len(self.cache) > self.config.cache_size_limit:
                self.cleanup_cache()

class CPUOptimizer:
    """CPU优化器"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.cpu_count = multiprocessing.cpu_count()
        self.optimal_workers = self._calculate_optimal_workers()
        
    def _calculate_optimal_workers(self) -> int:
        """计算最优工作线程数"""
        if self.config.max_workers:
            return min(self.config.max_workers, self.cpu_count)
        
        # 根据优化级别调整
        if self.config.level == OptimizationLevel.CONSERVATIVE:
            return max(1, self.cpu_count // 2)
        elif self.config.level == OptimizationLevel.BALANCED:
            return max(2, self.cpu_count - 1)
        else:  # AGGRESSIVE
            return self.cpu_count
    
    def get_cpu_info(self) -> Dict[str, Any]:
        """获取CPU信息"""
        return {
            'count': self.cpu_count,
            'usage': psutil.cpu_percent(interval=1),
            'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
            'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else []
        }
    
    def optimize_cpu_usage(self):
        """优化CPU使用"""
        cpu_usage = psutil.cpu_percent(interval=1)
        
        if cpu_usage > self.config.cpu_threshold * 100:
            logger.warning(f"CPU使用率过高: {cpu_usage:.1f}%")
            
            # 调整进程优先级
            try:
                current_process = psutil.Process()
                if cpu_usage > 95:
                    # 极高使用率时降低优先级
                    current_process.nice(5)
                elif cpu_usage > 85:
                    # 高使用率时稍微降低优先级
                    current_process.nice(2)
            except Exception as e:
                logger.debug(f"调整进程优先级失败: {e}")

class GPUOptimizer:
    """GPU优化器"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.gpu_available = TORCH_AVAILABLE and torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.gpu_available else 0
        
    def get_gpu_info(self) -> Dict[str, Any]:
        """获取GPU信息"""
        if not self.gpu_available:
            return {'available': False}
        
        info = {
            'available': True,
            'device_count': self.device_count,
            'devices': []
        }
        
        for i in range(self.device_count):
            device_info = {
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'memory_total': torch.cuda.get_device_properties(i).total_memory,
                'memory_allocated': torch.cuda.memory_allocated(i),
                'memory_cached': torch.cuda.memory_reserved(i)
            }
            info['devices'].append(device_info)
        
        return info
    
    def optimize_gpu_memory(self):
        """优化GPU内存"""
        if not self.gpu_available:
            return
        
        for i in range(self.device_count):
            allocated = torch.cuda.memory_allocated(i)
            total = torch.cuda.get_device_properties(i).total_memory
            usage_ratio = allocated / total
            
            if usage_ratio > 0.9:
                logger.warning(f"GPU {i} 内存使用率过高: {usage_ratio:.1%}")
                torch.cuda.empty_cache()
    
    def set_optimal_device(self) -> str:
        """设置最优GPU设备"""
        if not self.gpu_available:
            return 'cpu'
        
        if self.device_count == 1:
            return 'cuda:0'
        
        # 选择内存使用率最低的GPU
        best_device = 0
        min_usage = float('inf')
        
        for i in range(self.device_count):
            allocated = torch.cuda.memory_allocated(i)
            total = torch.cuda.get_device_properties(i).total_memory
            usage = allocated / total
            
            if usage < min_usage:
                min_usage = usage
                best_device = i
        
        return f'cuda:{best_device}'

class ThreadPoolManager:
    """线程池管理器"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.cpu_optimizer = CPUOptimizer(config)
        self.thread_pool = None
        self.process_pool = None
        self._lock = threading.Lock()
        
    def get_thread_pool(self) -> ThreadPoolExecutor:
        """获取线程池"""
        if self.thread_pool is None:
            with self._lock:
                if self.thread_pool is None:
                    max_workers = self.cpu_optimizer.optimal_workers
                    self.thread_pool = ThreadPoolExecutor(
                        max_workers=max_workers,
                        thread_name_prefix="YOLOS-Thread"
                    )
                    logger.debug(f"创建线程池，工作线程数: {max_workers}")
        
        return self.thread_pool
    
    def get_process_pool(self) -> ProcessPoolExecutor:
        """获取进程池"""
        if self.process_pool is None:
            with self._lock:
                if self.process_pool is None:
                    max_workers = min(4, self.cpu_optimizer.optimal_workers)
                    self.process_pool = ProcessPoolExecutor(
                        max_workers=max_workers
                    )
                    logger.debug(f"创建进程池，工作进程数: {max_workers}")
        
        return self.process_pool
    
    def shutdown(self):
        """关闭线程池和进程池"""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
            self.process_pool = None

class PerformanceOptimizer:
    """性能优化器主类"""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.memory_manager = MemoryManager(self.config)
        self.cpu_optimizer = CPUOptimizer(self.config)
        self.gpu_optimizer = GPUOptimizer(self.config)
        self.thread_pool_manager = ThreadPoolManager(self.config)
        
        self.monitoring_active = False
        self.monitoring_thread = None
        self.metrics_history: List[SystemMetrics] = []
        self.optimization_callbacks: List[Callable] = []
        
        # 性能统计
        self.stats = {
            'optimizations_performed': 0,
            'memory_cleanups': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def start_monitoring(self, interval: float = 30.0):
        """开始性能监控"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"性能监控已启动，监控间隔: {interval}秒")
    
    def stop_monitoring(self):
        """停止性能监控"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("性能监控已停止")
    
    def _monitoring_loop(self, interval: float):
        """监控循环"""
        while self.monitoring_active:
            try:
                metrics = self.collect_metrics()
                self.metrics_history.append(metrics)
                
                # 保持历史记录在合理范围内
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-500:]
                
                # 执行优化
                self.auto_optimize(metrics)
                
                # 执行回调
                for callback in self.optimization_callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.error(f"优化回调执行失败: {e}")
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                time.sleep(interval)
    
    def collect_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        metrics = SystemMetrics()
        
        # CPU指标
        metrics.cpu_usage = psutil.cpu_percent(interval=0.1)
        
        # 内存指标
        memory = psutil.virtual_memory()
        metrics.memory_usage = memory.percent
        metrics.memory_available = memory.available
        
        # GPU指标
        if self.gpu_optimizer.gpu_available:
            try:
                device_count = torch.cuda.device_count()
                total_memory = 0
                used_memory = 0
                
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    total_memory += props.total_memory
                    used_memory += torch.cuda.memory_allocated(i)
                
                if total_memory > 0:
                    metrics.gpu_memory_usage = (used_memory / total_memory) * 100
            except Exception as e:
                logger.debug(f"GPU指标收集失败: {e}")
        
        # 磁盘指标
        try:
            disk = psutil.disk_usage('/')
            metrics.disk_usage = disk.percent
        except Exception as e:
            logger.debug(f"磁盘指标收集失败: {e}")
        
        # 网络指标
        try:
            net_io = psutil.net_io_counters()
            metrics.network_io = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
        except Exception as e:
            logger.debug(f"网络指标收集失败: {e}")
        
        return metrics
    
    def auto_optimize(self, metrics: SystemMetrics):
        """自动优化"""
        optimized = False
        
        # 内存优化
        if self.config.enable_memory_optimization:
            if metrics.memory_usage > self.config.memory_threshold * 100:
                self.memory_manager.optimize_memory()
                self.stats['memory_cleanups'] += 1
                optimized = True
        
        # CPU优化
        if self.config.enable_cpu_optimization:
            if metrics.cpu_usage > self.config.cpu_threshold * 100:
                self.cpu_optimizer.optimize_cpu_usage()
                optimized = True
        
        # GPU优化
        if self.config.enable_gpu_optimization:
            if metrics.gpu_memory_usage > 90:
                self.gpu_optimizer.optimize_gpu_memory()
                optimized = True
        
        if optimized:
            self.stats['optimizations_performed'] += 1
    
    def optimize_for_inference(self):
        """为推理优化系统"""
        logger.info("开始推理优化...")
        
        # 设置最优GPU设备
        if self.config.enable_gpu_optimization:
            optimal_device = self.gpu_optimizer.set_optimal_device()
            logger.info(f"设置最优设备: {optimal_device}")
        
        # 预热内存管理
        self.memory_manager.force_garbage_collection()
        
        # 清理缓存
        self.memory_manager.cleanup_cache()
        
        # 设置线程池
        if self.config.enable_threading_optimization:
            thread_pool = self.thread_pool_manager.get_thread_pool()
            logger.info(f"线程池已准备，工作线程数: {thread_pool._max_workers}")
    
    def optimize_for_training(self):
        """为训练优化系统"""
        logger.info("开始训练优化...")
        
        # 更激进的内存管理
        original_threshold = self.config.memory_threshold
        self.config.memory_threshold = 0.7  # 降低内存阈值
        
        # 启用进程池用于数据加载
        if self.config.enable_threading_optimization:
            process_pool = self.thread_pool_manager.get_process_pool()
            logger.info(f"进程池已准备，工作进程数: {process_pool._max_workers}")
        
        # GPU内存优化
        if self.gpu_optimizer.gpu_available:
            self.gpu_optimizer.optimize_gpu_memory()
        
        return original_threshold
    
    def add_optimization_callback(self, callback: Callable[[SystemMetrics], None]):
        """添加优化回调"""
        self.optimization_callbacks.append(callback)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        current_metrics = self.collect_metrics()
        
        report = {
            'current_metrics': {
                'cpu_usage': current_metrics.cpu_usage,
                'memory_usage': current_metrics.memory_usage,
                'memory_available_gb': current_metrics.memory_available / (1024**3),
                'gpu_memory_usage': current_metrics.gpu_memory_usage,
                'disk_usage': current_metrics.disk_usage
            },
            'system_info': {
                'cpu_count': self.cpu_optimizer.cpu_count,
                'optimal_workers': self.cpu_optimizer.optimal_workers,
                'gpu_available': self.gpu_optimizer.gpu_available,
                'gpu_count': self.gpu_optimizer.device_count
            },
            'optimization_stats': self.stats.copy(),
            'config': {
                'level': self.config.level.value,
                'memory_threshold': self.config.memory_threshold,
                'cpu_threshold': self.config.cpu_threshold,
                'cache_size_limit': self.config.cache_size_limit
            }
        }
        
        # 添加历史趋势
        if len(self.metrics_history) > 1:
            recent_metrics = self.metrics_history[-10:]  # 最近10次
            report['trends'] = {
                'avg_cpu_usage': sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
                'avg_memory_usage': sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
                'avg_gpu_memory_usage': sum(m.gpu_memory_usage for m in recent_metrics) / len(recent_metrics)
            }
        
        return report
    
    def cleanup(self):
        """清理资源"""
        self.stop_monitoring()
        self.thread_pool_manager.shutdown()
        self.memory_manager.cleanup_cache(force=True)
        logger.info("性能优化器已清理")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()

# 全局优化器实例
_global_optimizer: Optional[PerformanceOptimizer] = None

def get_global_optimizer(config: OptimizationConfig = None) -> PerformanceOptimizer:
    """获取全局优化器实例"""
    global _global_optimizer
    
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer(config)
    
    return _global_optimizer

def optimize_for_platform(platform: str = "pc") -> OptimizationConfig:
    """为特定平台创建优化配置"""
    if platform.lower() in ["pc", "desktop", "server"]:
        return OptimizationConfig(
            level=OptimizationLevel.AGGRESSIVE,
            max_workers=None,  # 使用所有CPU核心
            memory_threshold=0.8,
            cpu_threshold=0.9,
            cache_size_limit=2000
        )
    elif platform.lower() in ["laptop", "mobile"]:
        return OptimizationConfig(
            level=OptimizationLevel.BALANCED,
            max_workers=4,
            memory_threshold=0.7,
            cpu_threshold=0.8,
            cache_size_limit=1000
        )
    elif platform.lower() in ["raspberry_pi", "jetson", "embedded"]:
        return OptimizationConfig(
            level=OptimizationLevel.CONSERVATIVE,
            max_workers=2,
            memory_threshold=0.6,
            cpu_threshold=0.7,
            cache_size_limit=500
        )
    else:
        return OptimizationConfig()  # 默认配置

# 装饰器：性能优化
def performance_optimized(optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
    """性能优化装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            config = OptimizationConfig(level=optimization_level)
            optimizer = PerformanceOptimizer(config)
            
            try:
                optimizer.optimize_for_inference()
                result = func(*args, **kwargs)
                return result
            finally:
                optimizer.cleanup()
        
        return wrapper
    return decorator

if __name__ == "__main__":
    # 测试性能优化器
    print("🚀 YOLOS性能优化器测试")
    
    config = optimize_for_platform("pc")
    
    with PerformanceOptimizer(config) as optimizer:
        print("性能监控已启动...")
        
        # 模拟一些工作负载
        time.sleep(5)
        
        # 获取性能报告
        report = optimizer.get_performance_report()
        print("\n性能报告:")
        print(f"CPU使用率: {report['current_metrics']['cpu_usage']:.1f}%")
        print(f"内存使用率: {report['current_metrics']['memory_usage']:.1f}%")
        print(f"可用内存: {report['current_metrics']['memory_available_gb']:.1f}GB")
        print(f"优化次数: {report['optimization_stats']['optimizations_performed']}")
        
        print("\n✅ 性能优化器测试完成")