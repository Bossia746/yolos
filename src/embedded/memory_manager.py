#!/usr/bin/env python3
"""
嵌入式设备内存管理器
提供动态内存分配、模型缓存和资源优化功能
"""

import os
import gc
import sys
import time
import threading
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import OrderedDict
import weakref

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

@dataclass
class MemoryStats:
    """内存统计信息"""
    total_mb: float
    used_mb: float
    available_mb: float
    cached_models_mb: float
    peak_usage_mb: float
    fragmentation_ratio: float
    gc_count: int
    last_cleanup_time: float

@dataclass
class ModelCacheEntry:
    """模型缓存条目"""
    model: Any
    size_mb: float
    last_access: float
    access_count: int
    priority: int  # 0=低, 1=中, 2=高
    is_persistent: bool = False

class EmbeddedMemoryManager:
    """嵌入式内存管理器"""
    
    def __init__(self, 
                 memory_limit_mb: int,
                 cache_limit_mb: Optional[int] = None,
                 enable_swap: bool = False,
                 gc_threshold: float = 0.8):
        """
        初始化内存管理器
        
        Args:
            memory_limit_mb: 内存限制(MB)
            cache_limit_mb: 缓存限制(MB)，默认为总限制的50%
            enable_swap: 是否启用交换文件
            gc_threshold: 触发垃圾回收的内存使用阈值
        """
        self.memory_limit_mb = memory_limit_mb
        self.cache_limit_mb = cache_limit_mb or (memory_limit_mb * 0.5)
        self.enable_swap = enable_swap
        self.gc_threshold = gc_threshold
        
        # 模型缓存 (LRU)
        self.model_cache: OrderedDict[str, ModelCacheEntry] = OrderedDict()
        
        # 内存池
        self.memory_pools: Dict[str, List[np.ndarray]] = {}
        
        # 统计信息
        self.stats = MemoryStats(
            total_mb=memory_limit_mb,
            used_mb=0.0,
            available_mb=memory_limit_mb,
            cached_models_mb=0.0,
            peak_usage_mb=0.0,
            fragmentation_ratio=0.0,
            gc_count=0,
            last_cleanup_time=time.time()
        )
        
        # 线程锁
        self.lock = threading.RLock()
        
        # 监控线程
        self.monitoring = False
        self.monitor_thread = None
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        # 初始化
        self._setup_memory_monitoring()
        
    def _setup_memory_monitoring(self):
        """设置内存监控"""
        if PSUTIL_AVAILABLE:
            self.monitoring = True
            self.monitor_thread = threading.Thread(
                target=self._memory_monitor_loop,
                daemon=True
            )
            self.monitor_thread.start()
            
    def _memory_monitor_loop(self):
        """内存监控循环"""
        while self.monitoring:
            try:
                self._update_memory_stats()
                
                # 检查是否需要清理
                if self.stats.used_mb / self.stats.total_mb > self.gc_threshold:
                    self.logger.warning(f"内存使用率过高: {self.stats.used_mb:.1f}MB / {self.stats.total_mb:.1f}MB")
                    self.cleanup_memory(aggressive=True)
                    
                time.sleep(1.0)  # 每秒检查一次
                
            except Exception as e:
                self.logger.error(f"内存监控错误: {e}")
                time.sleep(5.0)
                
    def _update_memory_stats(self):
        """更新内存统计"""
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            self.stats.used_mb = memory_info.rss / 1024 / 1024
            self.stats.available_mb = self.stats.total_mb - self.stats.used_mb
            
            # 更新峰值
            if self.stats.used_mb > self.stats.peak_usage_mb:
                self.stats.peak_usage_mb = self.stats.used_mb
                
            # 计算碎片化率
            if self.stats.total_mb > 0:
                self.stats.fragmentation_ratio = (
                    self.stats.used_mb - self.stats.cached_models_mb
                ) / self.stats.total_mb
                
    def get_memory_usage(self) -> float:
        """获取当前内存使用量(MB)"""
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                return process.memory_info().rss / 1024 / 1024
            except:
                pass
        return 0.0
        
    def get_available_memory(self) -> float:
        """获取可用内存(MB)"""
        used = self.get_memory_usage()
        return max(0, self.memory_limit_mb - used)
        
    def can_allocate(self, size_mb: float, buffer_ratio: float = 0.1) -> bool:
        """检查是否可以分配指定大小的内存"""
        available = self.get_available_memory()
        required = size_mb * (1 + buffer_ratio)  # 添加缓冲
        return available >= required
        
    def cache_model(self, 
                   model_name: str, 
                   model: Any, 
                   size_mb: float,
                   priority: int = 1,
                   is_persistent: bool = False) -> bool:
        """缓存模型"""
        with self.lock:
            # 检查是否有足够空间
            if not self.can_allocate(size_mb):
                # 尝试清理缓存
                self._cleanup_model_cache(size_mb)
                
                if not self.can_allocate(size_mb):
                    self.logger.warning(f"无法缓存模型 {model_name}: 内存不足")
                    return False
                    
            # 如果模型已存在，更新它
            if model_name in self.model_cache:
                old_entry = self.model_cache[model_name]
                self.stats.cached_models_mb -= old_entry.size_mb
                
            # 创建缓存条目
            entry = ModelCacheEntry(
                model=model,
                size_mb=size_mb,
                last_access=time.time(),
                access_count=1,
                priority=priority,
                is_persistent=is_persistent
            )
            
            # 添加到缓存
            self.model_cache[model_name] = entry
            self.stats.cached_models_mb += size_mb
            
            # 移到最后 (LRU)
            self.model_cache.move_to_end(model_name)
            
            self.logger.info(f"模型已缓存: {model_name} ({size_mb:.1f}MB)")
            return True
            
    def get_cached_model(self, model_name: str) -> Optional[Any]:
        """获取缓存的模型"""
        with self.lock:
            if model_name in self.model_cache:
                entry = self.model_cache[model_name]
                entry.last_access = time.time()
                entry.access_count += 1
                
                # 移到最后 (LRU)
                self.model_cache.move_to_end(model_name)
                
                return entry.model
            return None
            
    def remove_cached_model(self, model_name: str) -> bool:
        """移除缓存的模型"""
        with self.lock:
            if model_name in self.model_cache:
                entry = self.model_cache.pop(model_name)
                self.stats.cached_models_mb -= entry.size_mb
                
                # 显式删除模型引用
                del entry.model
                
                self.logger.info(f"模型已移除: {model_name}")
                return True
            return False
            
    def _cleanup_model_cache(self, required_mb: float = 0):
        """清理模型缓存"""
        freed_mb = 0
        models_to_remove = []
        
        # 按优先级和访问时间排序
        sorted_models = sorted(
            self.model_cache.items(),
            key=lambda x: (x[1].is_persistent, x[1].priority, x[1].last_access)
        )
        
        for model_name, entry in sorted_models:
            if entry.is_persistent:
                continue
                
            models_to_remove.append(model_name)
            freed_mb += entry.size_mb
            
            if freed_mb >= required_mb:
                break
                
        # 移除模型
        for model_name in models_to_remove:
            self.remove_cached_model(model_name)
            
        if models_to_remove:
            self.logger.info(f"清理了 {len(models_to_remove)} 个模型，释放 {freed_mb:.1f}MB")
            
    def allocate_buffer(self, 
                       size_mb: float, 
                       pool_name: str = "default",
                       dtype: str = "float32") -> Optional[np.ndarray]:
        """分配内存缓冲区"""
        if not NUMPY_AVAILABLE:
            return None
            
        with self.lock:
            if not self.can_allocate(size_mb):
                self.cleanup_memory()
                if not self.can_allocate(size_mb):
                    return None
                    
            # 计算数组大小
            dtype_sizes = {
                'float32': 4,
                'float16': 2,
                'int8': 1,
                'uint8': 1
            }
            
            element_size = dtype_sizes.get(dtype, 4)
            num_elements = int((size_mb * 1024 * 1024) / element_size)
            
            try:
                buffer = np.zeros(num_elements, dtype=dtype)
                
                # 添加到内存池
                if pool_name not in self.memory_pools:
                    self.memory_pools[pool_name] = []
                self.memory_pools[pool_name].append(buffer)
                
                return buffer
                
            except MemoryError:
                self.logger.error(f"内存分配失败: {size_mb:.1f}MB")
                return None
                
    def free_buffer(self, buffer: np.ndarray, pool_name: str = "default"):
        """释放内存缓冲区"""
        with self.lock:
            if pool_name in self.memory_pools:
                try:
                    self.memory_pools[pool_name].remove(buffer)
                except ValueError:
                    pass
                    
            # 显式删除引用
            del buffer
            
    def cleanup_memory(self, aggressive: bool = False):
        """清理内存"""
        with self.lock:
            self.logger.info("开始内存清理...")
            
            # 清理内存池
            total_freed = 0
            for pool_name, buffers in self.memory_pools.items():
                pool_size = len(buffers)
                if aggressive:
                    buffers.clear()
                else:
                    # 只清理一半
                    buffers[:] = buffers[pool_size//2:]
                total_freed += pool_size
                
            # 清理模型缓存
            if aggressive:
                cache_size = len(self.model_cache)
                self._cleanup_model_cache(self.cache_limit_mb)
                freed_models = cache_size - len(self.model_cache)
            else:
                # 只清理低优先级模型
                freed_models = 0
                for name, entry in list(self.model_cache.items()):
                    if entry.priority == 0 and not entry.is_persistent:
                        self.remove_cached_model(name)
                        freed_models += 1
                        
            # 强制垃圾回收
            collected = gc.collect()
            self.stats.gc_count += 1
            self.stats.last_cleanup_time = time.time()
            
            self.logger.info(
                f"内存清理完成: 释放 {total_freed} 个缓冲区, "
                f"{freed_models} 个模型, 回收 {collected} 个对象"
            )
            
    def optimize_memory_layout(self):
        """优化内存布局"""
        with self.lock:
            # 重新组织模型缓存
            # 按访问频率重新排序
            sorted_cache = OrderedDict(
                sorted(
                    self.model_cache.items(),
                    key=lambda x: x[1].access_count,
                    reverse=True
                )
            )
            self.model_cache = sorted_cache
            
            # 合并内存池
            for pool_name, buffers in self.memory_pools.items():
                if len(buffers) > 10:  # 如果缓冲区太多，合并一些
                    # 这里可以实现更复杂的内存合并逻辑
                    pass
                    
    def get_memory_stats(self) -> MemoryStats:
        """获取内存统计信息"""
        with self.lock:
            self._update_memory_stats()
            return self.stats
            
    def get_cache_info(self) -> Dict:
        """获取缓存信息"""
        with self.lock:
            cache_info = {
                'total_models': len(self.model_cache),
                'total_size_mb': self.stats.cached_models_mb,
                'models': {}
            }
            
            for name, entry in self.model_cache.items():
                cache_info['models'][name] = {
                    'size_mb': entry.size_mb,
                    'last_access': entry.last_access,
                    'access_count': entry.access_count,
                    'priority': entry.priority,
                    'is_persistent': entry.is_persistent
                }
                
            return cache_info
            
    def set_memory_limit(self, new_limit_mb: int):
        """设置新的内存限制"""
        with self.lock:
            old_limit = self.memory_limit_mb
            self.memory_limit_mb = new_limit_mb
            self.stats.total_mb = new_limit_mb
            
            # 如果新限制更小，需要清理
            if new_limit_mb < old_limit:
                current_usage = self.get_memory_usage()
                if current_usage > new_limit_mb * 0.8:
                    self.cleanup_memory(aggressive=True)
                    
            self.logger.info(f"内存限制已更新: {old_limit}MB -> {new_limit_mb}MB")
            
    def enable_memory_profiling(self, enable: bool = True):
        """启用/禁用内存分析"""
        if enable and not self.monitoring:
            self._setup_memory_monitoring()
        elif not enable and self.monitoring:
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=1.0)
                
    def create_memory_snapshot(self) -> Dict:
        """创建内存快照"""
        with self.lock:
            snapshot = {
                'timestamp': time.time(),
                'memory_stats': self.get_memory_stats().__dict__,
                'cache_info': self.get_cache_info(),
                'memory_pools': {
                    pool_name: len(buffers)
                    for pool_name, buffers in self.memory_pools.items()
                },
                'system_info': {}
            }
            
            if PSUTIL_AVAILABLE:
                try:
                    process = psutil.Process()
                    snapshot['system_info'] = {
                        'cpu_percent': process.cpu_percent(),
                        'memory_percent': process.memory_percent(),
                        'num_threads': process.num_threads(),
                        'open_files': len(process.open_files())
                    }
                except:
                    pass
                    
            return snapshot
            
    def cleanup(self):
        """清理所有资源"""
        with self.lock:
            self.monitoring = False
            
            # 清理所有缓存
            self.model_cache.clear()
            self.memory_pools.clear()
            
            # 强制垃圾回收
            gc.collect()
            
            self.logger.info("内存管理器已清理")
            
    def __del__(self):
        """析构函数"""
        self.cleanup()

# 全局内存管理器实例
_global_memory_manager: Optional[EmbeddedMemoryManager] = None

def get_global_memory_manager() -> Optional[EmbeddedMemoryManager]:
    """获取全局内存管理器"""
    return _global_memory_manager
    
def initialize_global_memory_manager(memory_limit_mb: int, **kwargs) -> EmbeddedMemoryManager:
    """初始化全局内存管理器"""
    global _global_memory_manager
    
    if _global_memory_manager is not None:
        _global_memory_manager.cleanup()
        
    _global_memory_manager = EmbeddedMemoryManager(memory_limit_mb, **kwargs)
    return _global_memory_manager
    
def cleanup_global_memory_manager():
    """清理全局内存管理器"""
    global _global_memory_manager
    
    if _global_memory_manager is not None:
        _global_memory_manager.cleanup()
        _global_memory_manager = None

if __name__ == "__main__":
    # 测试代码
    import numpy as np
    
    # 创建内存管理器
    manager = EmbeddedMemoryManager(
        memory_limit_mb=100,
        cache_limit_mb=50,
        enable_swap=False
    )
    
    try:
        # 测试缓冲区分配
        print("测试缓冲区分配...")
        buffer1 = manager.allocate_buffer(10, "test_pool")
        buffer2 = manager.allocate_buffer(20, "test_pool")
        
        print(f"分配了 2 个缓冲区")
        print(f"内存使用: {manager.get_memory_usage():.1f}MB")
        
        # 测试模型缓存
        print("\n测试模型缓存...")
        dummy_model = np.random.rand(1000, 1000)  # 模拟模型
        manager.cache_model("test_model", dummy_model, 8.0, priority=1)
        
        cached = manager.get_cached_model("test_model")
        print(f"模型缓存成功: {cached is not None}")
        
        # 获取统计信息
        stats = manager.get_memory_stats()
        print(f"\n内存统计:")
        print(f"  总内存: {stats.total_mb:.1f}MB")
        print(f"  已使用: {stats.used_mb:.1f}MB")
        print(f"  缓存模型: {stats.cached_models_mb:.1f}MB")
        print(f"  峰值使用: {stats.peak_usage_mb:.1f}MB")
        
        # 测试清理
        print("\n测试内存清理...")
        manager.cleanup_memory()
        
        final_stats = manager.get_memory_stats()
        print(f"清理后内存使用: {final_stats.used_mb:.1f}MB")
        
    finally:
        manager.cleanup()
        print("\n测试完成")