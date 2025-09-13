#!/usr/bin/env python3
"""
嵌入式内存优化器
实现动态内存管理、内存池、垃圾回收等功能
"""

import gc
import os
import sys
import time
import threading
import tracemalloc
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import contextmanager
from enum import Enum
import logging

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

class MemoryPriority(Enum):
    """内存优先级"""
    CRITICAL = 1    # 关键内存，不可释放
    HIGH = 2        # 高优先级
    MEDIUM = 3      # 中等优先级
    LOW = 4         # 低优先级，优先释放
    CACHE = 5       # 缓存数据，可随时释放

@dataclass
class MemoryBlock:
    """内存块信息"""
    id: str
    size_bytes: int
    priority: MemoryPriority
    created_at: float
    last_accessed: float
    access_count: int = 0
    data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_access(self):
        """更新访问信息"""
        self.last_accessed = time.time()
        self.access_count += 1

@dataclass
class MemoryStats:
    """内存统计信息"""
    total_memory_mb: float
    used_memory_mb: float
    available_memory_mb: float
    process_memory_mb: float
    managed_memory_mb: float
    memory_blocks: int
    gc_collections: int
    memory_pressure: float  # 0.0-1.0
    fragmentation_ratio: float

class MemoryPool:
    """内存池"""
    
    def __init__(self, block_size: int, max_blocks: int = 100):
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.available_blocks = deque()
        self.allocated_blocks = set()
        self.lock = threading.RLock()
        
        # 预分配一些块
        self._preallocate_blocks(min(10, max_blocks // 4))
        
    def _preallocate_blocks(self, count: int):
        """预分配内存块"""
        for _ in range(count):
            if NUMPY_AVAILABLE:
                block = np.empty(self.block_size, dtype=np.uint8)
            else:
                block = bytearray(self.block_size)
            self.available_blocks.append(block)
            
    def allocate(self) -> Optional[Any]:
        """分配内存块"""
        with self.lock:
            if self.available_blocks:
                block = self.available_blocks.popleft()
                self.allocated_blocks.add(id(block))
                return block
            elif len(self.allocated_blocks) < self.max_blocks:
                # 动态分配新块
                if NUMPY_AVAILABLE:
                    block = np.empty(self.block_size, dtype=np.uint8)
                else:
                    block = bytearray(self.block_size)
                self.allocated_blocks.add(id(block))
                return block
            else:
                return None
                
    def deallocate(self, block: Any) -> bool:
        """释放内存块"""
        with self.lock:
            block_id = id(block)
            if block_id in self.allocated_blocks:
                self.allocated_blocks.remove(block_id)
                if len(self.available_blocks) < self.max_blocks // 2:
                    # 重置块内容
                    if hasattr(block, 'fill'):
                        block.fill(0)
                    else:
                        for i in range(len(block)):
                            block[i] = 0
                    self.available_blocks.append(block)
                return True
            return False
            
    def get_stats(self) -> Dict[str, int]:
        """获取内存池统计"""
        with self.lock:
            return {
                'block_size': self.block_size,
                'max_blocks': self.max_blocks,
                'available_blocks': len(self.available_blocks),
                'allocated_blocks': len(self.allocated_blocks),
                'total_memory_mb': (len(self.available_blocks) + len(self.allocated_blocks)) * self.block_size / 1024 / 1024
            }

class MemoryOptimizer:
    """内存优化器"""
    
    def __init__(self, max_memory_mb: int = 512, enable_tracing: bool = True):
        self.max_memory_mb = max_memory_mb
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.enable_tracing = enable_tracing
        
        # 内存块管理
        self.memory_blocks: Dict[str, MemoryBlock] = {}
        self.memory_pools: Dict[int, MemoryPool] = {}
        
        # 统计信息
        self.stats = {
            'allocations': 0,
            'deallocations': 0,
            'gc_runs': 0,
            'memory_freed_mb': 0.0,
            'peak_memory_mb': 0.0
        }
        
        # 线程锁
        self.lock = threading.RLock()
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        # 内存监控
        self.monitoring_enabled = False
        self.monitoring_thread = None
        self.monitoring_interval = 5.0  # 秒
        
        # 内存压力阈值
        self.pressure_thresholds = {
            'low': 0.6,      # 60%
            'medium': 0.8,   # 80%
            'high': 0.9,     # 90%
            'critical': 0.95 # 95%
        }
        
        # 垃圾回收策略
        self.gc_strategy = {
            'auto_gc': True,
            'gc_threshold': 0.8,  # 80%内存使用率时触发GC
            'aggressive_gc': False
        }
        
        # 启用内存跟踪
        if self.enable_tracing and not tracemalloc.is_tracing():
            tracemalloc.start()
            
        # 创建常用大小的内存池
        self._initialize_memory_pools()
        
    def _initialize_memory_pools(self):
        """初始化内存池"""
        # 常用的内存块大小 (字节)
        pool_sizes = [1024, 4096, 16384, 65536, 262144]  # 1KB, 4KB, 16KB, 64KB, 256KB
        
        for size in pool_sizes:
            max_blocks = max(10, self.max_memory_mb // (size // 1024 // 1024 + 1))
            self.memory_pools[size] = MemoryPool(size, max_blocks)
            
    def _get_memory_usage(self) -> Tuple[float, float, float]:
        """获取内存使用情况 (总内存MB, 已用内存MB, 进程内存MB)"""
        if PSUTIL_AVAILABLE:
            try:
                # 系统内存
                memory = psutil.virtual_memory()
                total_mb = memory.total / 1024 / 1024
                used_mb = memory.used / 1024 / 1024
                
                # 进程内存
                process = psutil.Process()
                process_mb = process.memory_info().rss / 1024 / 1024
                
                return total_mb, used_mb, process_mb
            except:
                pass
                
        # 回退方案
        if self.enable_tracing and tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            process_mb = current / 1024 / 1024
        else:
            process_mb = 0.0
            
        return 0.0, 0.0, process_mb
        
    def _calculate_memory_pressure(self) -> float:
        """计算内存压力 (0.0-1.0)"""
        total_mb, used_mb, process_mb = self._get_memory_usage()
        
        if total_mb > 0:
            system_pressure = used_mb / total_mb
        else:
            system_pressure = 0.0
            
        if self.max_memory_mb > 0:
            process_pressure = process_mb / self.max_memory_mb
        else:
            process_pressure = 0.0
            
        # 取较高的压力值
        return max(system_pressure, process_pressure)
        
    def _should_trigger_gc(self) -> bool:
        """判断是否应该触发垃圾回收"""
        if not self.gc_strategy['auto_gc']:
            return False
            
        pressure = self._calculate_memory_pressure()
        return pressure >= self.gc_strategy['gc_threshold']
        
    def _run_garbage_collection(self, aggressive: bool = False) -> Dict[str, Any]:
        """运行垃圾回收"""
        start_time = time.time()
        
        # 记录GC前的内存
        _, _, memory_before = self._get_memory_usage()
        
        # 执行垃圾回收
        if aggressive:
            # 激进的垃圾回收
            collected = 0
            for generation in range(3):
                collected += gc.collect(generation)
        else:
            # 标准垃圾回收
            collected = gc.collect()
            
        # 记录GC后的内存
        _, _, memory_after = self._get_memory_usage()
        
        gc_time = time.time() - start_time
        memory_freed = memory_before - memory_after
        
        self.stats['gc_runs'] += 1
        self.stats['memory_freed_mb'] += memory_freed
        
        result = {
            'objects_collected': collected,
            'memory_freed_mb': memory_freed,
            'gc_time_ms': gc_time * 1000,
            'aggressive': aggressive
        }
        
        self.logger.debug(f"垃圾回收完成: {result}")
        
        return result
        
    def allocate_memory(self, size_bytes: int, priority: MemoryPriority = MemoryPriority.MEDIUM,
                       block_id: Optional[str] = None, metadata: Dict[str, Any] = None) -> Optional[str]:
        """分配内存"""
        if block_id is None:
            block_id = f"block_{int(time.time() * 1000000)}_{id(self)}"
            
        # 检查内存压力
        if self._should_trigger_gc():
            self._run_garbage_collection()
            
        # 尝试从内存池分配
        allocated_data = None
        pool_used = False
        
        # 寻找合适的内存池
        for pool_size, pool in self.memory_pools.items():
            if size_bytes <= pool_size:
                allocated_data = pool.allocate()
                if allocated_data is not None:
                    pool_used = True
                    break
                    
        # 如果内存池分配失败，直接分配
        if allocated_data is None:
            try:
                if NUMPY_AVAILABLE and size_bytes > 1024:
                    allocated_data = np.empty(size_bytes, dtype=np.uint8)
                else:
                    allocated_data = bytearray(size_bytes)
            except MemoryError:
                # 内存不足，尝试释放低优先级内存
                freed_mb = self._free_low_priority_memory(size_bytes)
                if freed_mb > 0:
                    try:
                        if NUMPY_AVAILABLE and size_bytes > 1024:
                            allocated_data = np.empty(size_bytes, dtype=np.uint8)
                        else:
                            allocated_data = bytearray(size_bytes)
                    except MemoryError:
                        self.logger.error(f"内存分配失败: {size_bytes} 字节")
                        return None
                else:
                    self.logger.error(f"内存分配失败: {size_bytes} 字节")
                    return None
                    
        # 创建内存块
        now = time.time()
        memory_block = MemoryBlock(
            id=block_id,
            size_bytes=size_bytes,
            priority=priority,
            created_at=now,
            last_accessed=now,
            data=allocated_data,
            metadata=metadata or {}
        )
        
        # 添加到管理列表
        with self.lock:
            self.memory_blocks[block_id] = memory_block
            
        self.stats['allocations'] += 1
        
        # 更新峰值内存
        _, _, current_memory = self._get_memory_usage()
        if current_memory > self.stats['peak_memory_mb']:
            self.stats['peak_memory_mb'] = current_memory
            
        self.logger.debug(f"内存分配成功: {block_id} ({size_bytes} 字节, 优先级: {priority.name})")
        
        return block_id
        
    def deallocate_memory(self, block_id: str) -> bool:
        """释放内存"""
        with self.lock:
            if block_id not in self.memory_blocks:
                return False
                
            memory_block = self.memory_blocks[block_id]
            
            # 尝试返回到内存池
            returned_to_pool = False
            for pool_size, pool in self.memory_pools.items():
                if (hasattr(memory_block.data, '__len__') and 
                    len(memory_block.data) == pool_size):
                    if pool.deallocate(memory_block.data):
                        returned_to_pool = True
                        break
                        
            # 从管理列表中移除
            del self.memory_blocks[block_id]
            
        self.stats['deallocations'] += 1
        
        self.logger.debug(f"内存释放: {block_id} ({'返回内存池' if returned_to_pool else '直接释放'})")
        
        return True
        
    def get_memory_block(self, block_id: str) -> Optional[Any]:
        """获取内存块数据"""
        with self.lock:
            if block_id in self.memory_blocks:
                memory_block = self.memory_blocks[block_id]
                memory_block.update_access()
                return memory_block.data
            return None
            
    def _free_low_priority_memory(self, required_bytes: int) -> float:
        """释放低优先级内存"""
        freed_bytes = 0
        blocks_to_remove = []
        
        # 按优先级和访问时间排序
        with self.lock:
            sorted_blocks = sorted(
                self.memory_blocks.items(),
                key=lambda x: (x[1].priority.value, x[1].last_accessed)
            )
            
        for block_id, memory_block in sorted_blocks:
            if freed_bytes >= required_bytes:
                break
                
            # 只释放低优先级和缓存内存
            if memory_block.priority in [MemoryPriority.LOW, MemoryPriority.CACHE]:
                freed_bytes += memory_block.size_bytes
                blocks_to_remove.append(block_id)
                
        # 释放选中的内存块
        for block_id in blocks_to_remove:
            self.deallocate_memory(block_id)
            
        freed_mb = freed_bytes / 1024 / 1024
        
        if blocks_to_remove:
            self.logger.info(f"释放低优先级内存: {freed_mb:.1f}MB ({len(blocks_to_remove)} 个块)")
            
        return freed_mb
        
    def optimize_memory(self) -> Dict[str, Any]:
        """优化内存使用"""
        self.logger.info("开始内存优化...")
        
        optimization_results = {
            'gc_result': None,
            'freed_blocks': 0,
            'freed_memory_mb': 0.0,
            'defragmented_pools': 0
        }
        
        # 1. 运行垃圾回收
        pressure = self._calculate_memory_pressure()
        aggressive_gc = pressure > self.pressure_thresholds['high']
        
        gc_result = self._run_garbage_collection(aggressive=aggressive_gc)
        optimization_results['gc_result'] = gc_result
        
        # 2. 清理过期的低优先级内存块
        current_time = time.time()
        expired_blocks = []
        
        with self.lock:
            for block_id, memory_block in self.memory_blocks.items():
                # 清理超过5分钟未访问的缓存内存
                if (memory_block.priority == MemoryPriority.CACHE and 
                    current_time - memory_block.last_accessed > 300):
                    expired_blocks.append(block_id)
                # 清理超过10分钟未访问的低优先级内存
                elif (memory_block.priority == MemoryPriority.LOW and 
                      current_time - memory_block.last_accessed > 600):
                    expired_blocks.append(block_id)
                    
        freed_bytes = 0
        for block_id in expired_blocks:
            if block_id in self.memory_blocks:
                freed_bytes += self.memory_blocks[block_id].size_bytes
                self.deallocate_memory(block_id)
                
        optimization_results['freed_blocks'] = len(expired_blocks)
        optimization_results['freed_memory_mb'] = freed_bytes / 1024 / 1024
        
        # 3. 整理内存池
        defragmented_pools = 0
        for pool in self.memory_pools.values():
            # 简单的内存池整理：清理过多的可用块
            with pool.lock:
                if len(pool.available_blocks) > pool.max_blocks // 2:
                    excess_blocks = len(pool.available_blocks) - pool.max_blocks // 4
                    for _ in range(excess_blocks):
                        if pool.available_blocks:
                            pool.available_blocks.popleft()
                    defragmented_pools += 1
                    
        optimization_results['defragmented_pools'] = defragmented_pools
        
        self.logger.info(f"内存优化完成: {optimization_results}")
        
        return optimization_results
        
    def get_memory_stats(self) -> MemoryStats:
        """获取内存统计信息"""
        total_mb, used_mb, process_mb = self._get_memory_usage()
        available_mb = total_mb - used_mb if total_mb > 0 else 0.0
        
        # 计算管理的内存大小
        managed_bytes = sum(block.size_bytes for block in self.memory_blocks.values())
        managed_mb = managed_bytes / 1024 / 1024
        
        # 计算内存压力
        memory_pressure = self._calculate_memory_pressure()
        
        # 计算碎片率 (简化计算)
        total_pool_memory = sum(pool.get_stats()['total_memory_mb'] for pool in self.memory_pools.values())
        fragmentation_ratio = 1.0 - (managed_mb / max(total_pool_memory, 1.0)) if total_pool_memory > 0 else 0.0
        
        return MemoryStats(
            total_memory_mb=total_mb,
            used_memory_mb=used_mb,
            available_memory_mb=available_mb,
            process_memory_mb=process_mb,
            managed_memory_mb=managed_mb,
            memory_blocks=len(self.memory_blocks),
            gc_collections=self.stats['gc_runs'],
            memory_pressure=memory_pressure,
            fragmentation_ratio=fragmentation_ratio
        )
        
    def start_monitoring(self, interval: float = 5.0, 
                        callback: Optional[Callable[[MemoryStats], None]] = None):
        """启动内存监控"""
        if self.monitoring_enabled:
            return
            
        self.monitoring_enabled = True
        self.monitoring_interval = interval
        
        def monitor_loop():
            while self.monitoring_enabled:
                try:
                    stats = self.get_memory_stats()
                    
                    # 检查内存压力
                    if stats.memory_pressure > self.pressure_thresholds['critical']:
                        self.logger.warning(f"内存压力过高: {stats.memory_pressure:.1%}")
                        self.optimize_memory()
                    elif stats.memory_pressure > self.pressure_thresholds['high']:
                        self.logger.info(f"内存压力较高: {stats.memory_pressure:.1%}")
                        
                    # 调用回调函数
                    if callback:
                        callback(stats)
                        
                except Exception as e:
                    self.logger.error(f"内存监控错误: {e}")
                    
                time.sleep(self.monitoring_interval)
                
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info(f"内存监控已启动 (间隔: {interval}秒)")
        
    def stop_monitoring(self):
        """停止内存监控"""
        self.monitoring_enabled = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
            self.monitoring_thread = None
            
        self.logger.info("内存监控已停止")
        
    @contextmanager
    def memory_context(self, max_memory_mb: Optional[int] = None, 
                      auto_cleanup: bool = True):
        """内存上下文管理器"""
        # 记录进入时的内存状态
        initial_blocks = set(self.memory_blocks.keys())
        
        # 临时调整内存限制
        original_max_memory = self.max_memory_mb
        if max_memory_mb is not None:
            self.max_memory_mb = max_memory_mb
            self.max_memory_bytes = max_memory_mb * 1024 * 1024
            
        try:
            yield self
        finally:
            # 恢复原始内存限制
            self.max_memory_mb = original_max_memory
            self.max_memory_bytes = original_max_memory * 1024 * 1024
            
            # 自动清理在上下文中分配的内存
            if auto_cleanup:
                current_blocks = set(self.memory_blocks.keys())
                context_blocks = current_blocks - initial_blocks
                
                for block_id in context_blocks:
                    self.deallocate_memory(block_id)
                    
                if context_blocks:
                    self.logger.debug(f"上下文清理: 释放了 {len(context_blocks)} 个内存块")
                    
    def get_memory_info(self) -> Dict[str, Any]:
        """获取内存信息"""
        stats = self.get_memory_stats()
        
        # 内存池统计
        pool_stats = {}
        for size, pool in self.memory_pools.items():
            pool_stats[f"pool_{size}B"] = pool.get_stats()
            
        return {
            'max_memory_mb': self.max_memory_mb,
            'memory_stats': stats.__dict__,
            'allocation_stats': self.stats.copy(),
            'memory_pools': pool_stats,
            'monitoring_enabled': self.monitoring_enabled,
            'gc_strategy': self.gc_strategy.copy(),
            'pressure_thresholds': self.pressure_thresholds.copy()
        }

# 全局内存优化器实例
_global_memory_optimizer: Optional[MemoryOptimizer] = None

def get_memory_optimizer(max_memory_mb: int = 512, 
                        enable_tracing: bool = True) -> MemoryOptimizer:
    """获取内存优化器"""
    global _global_memory_optimizer
    
    if _global_memory_optimizer is None:
        _global_memory_optimizer = MemoryOptimizer(max_memory_mb, enable_tracing)
        
    return _global_memory_optimizer

def optimize_global_memory() -> Dict[str, Any]:
    """优化全局内存"""
    global _global_memory_optimizer
    
    if _global_memory_optimizer is not None:
        return _global_memory_optimizer.optimize_memory()
    else:
        return {'error': '内存优化器未初始化'}

def clear_global_memory() -> None:
    """清空全局内存"""
    global _global_memory_optimizer
    
    if _global_memory_optimizer is not None:
        # 释放所有内存块
        block_ids = list(_global_memory_optimizer.memory_blocks.keys())
        for block_id in block_ids:
            _global_memory_optimizer.deallocate_memory(block_id)
            
        # 停止监控
        _global_memory_optimizer.stop_monitoring()
        
        _global_memory_optimizer = None

if __name__ == "__main__":
    # 测试代码
    print("内存优化器测试")
    print("=" * 50)
    
    # 创建内存优化器
    optimizer = MemoryOptimizer(max_memory_mb=100, enable_tracing=True)
    
    # 测试内存分配
    print("\n测试内存分配...")
    block_ids = []
    
    for i in range(10):
        size = (i + 1) * 1024  # 1KB, 2KB, ..., 10KB
        priority = MemoryPriority.MEDIUM if i < 5 else MemoryPriority.LOW
        
        block_id = optimizer.allocate_memory(
            size_bytes=size,
            priority=priority,
            metadata={'test_index': i}
        )
        
        if block_id:
            block_ids.append(block_id)
            print(f"  分配内存块 {i+1}: {block_id} ({size} 字节)")
        else:
            print(f"  分配失败: {size} 字节")
            
    # 测试内存访问
    print("\n测试内存访问...")
    for i, block_id in enumerate(block_ids[:3]):
        data = optimizer.get_memory_block(block_id)
        if data is not None:
            print(f"  访问内存块 {i+1}: 成功 (大小: {len(data)} 字节)")
        else:
            print(f"  访问内存块 {i+1}: 失败")
            
    # 获取内存统计
    print("\n内存统计信息:")
    stats = optimizer.get_memory_stats()
    print(f"  进程内存: {stats.process_memory_mb:.1f}MB")
    print(f"  管理内存: {stats.managed_memory_mb:.1f}MB")
    print(f"  内存块数: {stats.memory_blocks}")
    print(f"  内存压力: {stats.memory_pressure:.1%}")
    
    # 测试内存优化
    print("\n执行内存优化...")
    optimization_results = optimizer.optimize_memory()
    for key, value in optimization_results.items():
        print(f"  {key}: {value}")
        
    # 测试内存上下文
    print("\n测试内存上下文...")
    with optimizer.memory_context(max_memory_mb=50, auto_cleanup=True) as ctx:
        # 在上下文中分配内存
        temp_blocks = []
        for i in range(3):
            block_id = ctx.allocate_memory(
                size_bytes=1024,
                priority=MemoryPriority.CACHE
            )
            if block_id:
                temp_blocks.append(block_id)
                
        print(f"  上下文中分配了 {len(temp_blocks)} 个临时内存块")
        
    # 上下文结束后，临时内存块应该被自动清理
    print(f"  上下文结束后，当前内存块数: {len(optimizer.memory_blocks)}")
    
    # 清理测试内存
    print("\n清理测试内存...")
    for block_id in block_ids:
        if optimizer.deallocate_memory(block_id):
            print(f"  释放内存块: {block_id}")
            
    print("\n测试完成")