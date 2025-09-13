#!/usr/bin/env python3
"""
嵌入式存储优化器
实现模型缓存、数据压缩、存储空间管理等功能
"""

import os
import sys
import json
import gzip
import pickle
import hashlib
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import OrderedDict

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    file_path: str
    size_bytes: int
    created_at: datetime
    last_accessed: datetime
    access_count: int
    compressed: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
            
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        return data
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """从字典创建"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        return cls(**data)

@dataclass
class StorageStats:
    """存储统计信息"""
    total_space_gb: float
    used_space_gb: float
    available_space_gb: float
    cache_size_mb: float
    cache_entries: int
    compression_ratio: float
    hit_rate: float
    
class LRUCache:
    """LRU缓存实现"""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        with self.lock:
            if key in self.cache:
                # 移动到末尾 (最近使用)
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None
            
    def put(self, key: str, value: Any) -> None:
        """添加缓存项"""
        with self.lock:
            if key in self.cache:
                # 更新现有项
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # 移除最久未使用的项
                self.cache.popitem(last=False)
                
            self.cache[key] = value
            
    def remove(self, key: str) -> bool:
        """移除缓存项"""
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)
                return True
            return False
            
    def clear(self) -> None:
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            
    def keys(self) -> List[str]:
        """获取所有键"""
        with self.lock:
            return list(self.cache.keys())
            
    def size(self) -> int:
        """获取缓存大小"""
        with self.lock:
            return len(self.cache)

class StorageOptimizer:
    """存储优化器"""
    
    def __init__(self, cache_dir: str = "./cache", max_cache_size_mb: int = 500):
        self.cache_dir = Path(cache_dir)
        self.max_cache_size_mb = max_cache_size_mb
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        
        # 创建缓存目录
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 缓存索引文件
        self.index_file = self.cache_dir / 'cache_index.json'
        
        # 内存缓存 (LRU)
        self.memory_cache = LRUCache(max_size=100)
        
        # 缓存条目索引
        self.cache_entries: Dict[str, CacheEntry] = {}
        
        # 统计信息
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_requests': 0,
            'bytes_saved': 0,
            'compression_saves': 0
        }
        
        # 线程锁
        self.lock = threading.RLock()
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        # 加载缓存索引
        self._load_cache_index()
        
        # 清理过期缓存
        self._cleanup_expired_cache()
        
    def _load_cache_index(self) -> None:
        """加载缓存索引"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for key, entry_data in data.items():
                    try:
                        entry = CacheEntry.from_dict(entry_data)
                        # 检查文件是否存在
                        if Path(entry.file_path).exists():
                            self.cache_entries[key] = entry
                        else:
                            self.logger.warning(f"缓存文件不存在: {entry.file_path}")
                    except Exception as e:
                        self.logger.warning(f"加载缓存条目失败 {key}: {e}")
                        
                self.logger.info(f"加载了 {len(self.cache_entries)} 个缓存条目")
                
            except Exception as e:
                self.logger.error(f"加载缓存索引失败: {e}")
                
    def _save_cache_index(self) -> None:
        """保存缓存索引"""
        try:
            with self.lock:
                data = {key: entry.to_dict() for key, entry in self.cache_entries.items()}
                
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"保存缓存索引失败: {e}")
            
    def _cleanup_expired_cache(self, max_age_days: int = 7) -> None:
        """清理过期缓存"""
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        expired_keys = []
        
        with self.lock:
            for key, entry in self.cache_entries.items():
                if entry.last_accessed < cutoff_time:
                    expired_keys.append(key)
                    
        for key in expired_keys:
            self.remove_from_cache(key)
            
        if expired_keys:
            self.logger.info(f"清理了 {len(expired_keys)} 个过期缓存条目")
            
    def _generate_cache_key(self, data: Any, prefix: str = "") -> str:
        """生成缓存键"""
        if isinstance(data, (str, bytes)):
            content = data if isinstance(data, bytes) else data.encode('utf-8')
        else:
            content = str(data).encode('utf-8')
            
        hash_obj = hashlib.sha256(content)
        hash_hex = hash_obj.hexdigest()[:16]  # 使用前16位
        
        return f"{prefix}_{hash_hex}" if prefix else hash_hex
        
    def _compress_data(self, data: bytes) -> Tuple[bytes, float]:
        """压缩数据"""
        try:
            compressed = gzip.compress(data, compresslevel=6)
            compression_ratio = len(compressed) / len(data) if len(data) > 0 else 1.0
            return compressed, compression_ratio
        except Exception as e:
            self.logger.warning(f"数据压缩失败: {e}")
            return data, 1.0
            
    def _decompress_data(self, compressed_data: bytes) -> bytes:
        """解压数据"""
        try:
            return gzip.decompress(compressed_data)
        except Exception as e:
            self.logger.error(f"数据解压失败: {e}")
            return compressed_data
            
    def _ensure_cache_space(self, required_bytes: int) -> bool:
        """确保缓存空间足够"""
        current_size = self._get_cache_size_bytes()
        
        if current_size + required_bytes <= self.max_cache_size_bytes:
            return True
            
        # 需要清理空间
        target_size = self.max_cache_size_bytes - required_bytes
        freed_bytes = self._free_cache_space(current_size - target_size)
        
        return freed_bytes >= (current_size + required_bytes - self.max_cache_size_bytes)
        
    def _free_cache_space(self, target_bytes: int) -> int:
        """释放缓存空间"""
        # 按最后访问时间排序，优先删除最久未使用的
        sorted_entries = sorted(
            self.cache_entries.items(),
            key=lambda x: (x[1].last_accessed, x[1].access_count)
        )
        
        freed_bytes = 0
        removed_keys = []
        
        for key, entry in sorted_entries:
            if freed_bytes >= target_bytes:
                break
                
            freed_bytes += entry.size_bytes
            removed_keys.append(key)
            
        # 删除选中的缓存条目
        for key in removed_keys:
            self.remove_from_cache(key)
            
        self.logger.info(f"释放缓存空间: {freed_bytes / 1024 / 1024:.1f}MB ({len(removed_keys)} 个条目)")
        
        return freed_bytes
        
    def _get_cache_size_bytes(self) -> int:
        """获取缓存总大小(字节)"""
        with self.lock:
            return sum(entry.size_bytes for entry in self.cache_entries.values())
            
    def cache_model(self, model_data: Any, model_name: str, 
                   compress: bool = True, metadata: Dict[str, Any] = None) -> str:
        """缓存模型数据"""
        cache_key = self._generate_cache_key(model_name, "model")
        
        try:
            # 序列化模型数据
            if hasattr(model_data, 'state_dict'):  # PyTorch模型
                serialized_data = pickle.dumps(model_data.state_dict())
            elif hasattr(model_data, 'save'):  # 其他模型
                # 临时保存到内存
                import io
                buffer = io.BytesIO()
                model_data.save(buffer)
                serialized_data = buffer.getvalue()
            else:
                # 直接序列化
                serialized_data = pickle.dumps(model_data)
                
            # 压缩数据
            if compress:
                compressed_data, compression_ratio = self._compress_data(serialized_data)
                final_data = compressed_data
                self.stats['compression_saves'] += len(serialized_data) - len(compressed_data)
            else:
                final_data = serialized_data
                compression_ratio = 1.0
                
            # 检查缓存空间
            if not self._ensure_cache_space(len(final_data)):
                raise RuntimeError("缓存空间不足")
                
            # 保存到文件
            cache_file = self.cache_dir / f"{cache_key}.cache"
            with open(cache_file, 'wb') as f:
                f.write(final_data)
                
            # 创建缓存条目
            now = datetime.now()
            entry = CacheEntry(
                key=cache_key,
                file_path=str(cache_file),
                size_bytes=len(final_data),
                created_at=now,
                last_accessed=now,
                access_count=0,
                compressed=compress,
                metadata=metadata or {}
            )
            
            # 添加到索引
            with self.lock:
                self.cache_entries[cache_key] = entry
                
            # 保存索引
            self._save_cache_index()
            
            self.logger.info(f"模型缓存成功: {model_name} -> {cache_key} "
                           f"({len(final_data) / 1024 / 1024:.1f}MB, 压缩率: {compression_ratio:.2f})")
            
            return cache_key
            
        except Exception as e:
            self.logger.error(f"模型缓存失败 {model_name}: {e}")
            raise
            
    def load_cached_model(self, cache_key: str) -> Optional[Any]:
        """加载缓存的模型"""
        self.stats['total_requests'] += 1
        
        # 先检查内存缓存
        cached_data = self.memory_cache.get(cache_key)
        if cached_data is not None:
            self.stats['cache_hits'] += 1
            return cached_data
            
        # 检查磁盘缓存
        with self.lock:
            if cache_key not in self.cache_entries:
                self.stats['cache_misses'] += 1
                return None
                
            entry = self.cache_entries[cache_key]
            
        try:
            # 读取缓存文件
            with open(entry.file_path, 'rb') as f:
                file_data = f.read()
                
            # 解压数据
            if entry.compressed:
                serialized_data = self._decompress_data(file_data)
            else:
                serialized_data = file_data
                
            # 反序列化
            model_data = pickle.loads(serialized_data)
            
            # 更新访问信息
            with self.lock:
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                
            # 添加到内存缓存
            self.memory_cache.put(cache_key, model_data)
            
            self.stats['cache_hits'] += 1
            self.logger.debug(f"从缓存加载模型: {cache_key}")
            
            return model_data
            
        except Exception as e:
            self.logger.error(f"加载缓存模型失败 {cache_key}: {e}")
            self.stats['cache_misses'] += 1
            return None
            
    def cache_data(self, data: Any, key: str, compress: bool = True, 
                  ttl_hours: int = 24, metadata: Dict[str, Any] = None) -> bool:
        """缓存通用数据"""
        cache_key = self._generate_cache_key(key, "data")
        
        try:
            # 序列化数据
            serialized_data = pickle.dumps(data)
            
            # 压缩数据
            if compress and len(serialized_data) > 1024:  # 大于1KB才压缩
                compressed_data, compression_ratio = self._compress_data(serialized_data)
                final_data = compressed_data
                self.stats['compression_saves'] += len(serialized_data) - len(compressed_data)
            else:
                final_data = serialized_data
                compression_ratio = 1.0
                compress = False
                
            # 检查缓存空间
            if not self._ensure_cache_space(len(final_data)):
                self.logger.warning(f"缓存空间不足，跳过缓存: {key}")
                return False
                
            # 保存到文件
            cache_file = self.cache_dir / f"{cache_key}.cache"
            with open(cache_file, 'wb') as f:
                f.write(final_data)
                
            # 创建缓存条目
            now = datetime.now()
            entry = CacheEntry(
                key=cache_key,
                file_path=str(cache_file),
                size_bytes=len(final_data),
                created_at=now,
                last_accessed=now,
                access_count=0,
                compressed=compress,
                metadata=metadata or {'ttl_hours': ttl_hours}
            )
            
            # 添加到索引
            with self.lock:
                self.cache_entries[cache_key] = entry
                
            self.logger.debug(f"数据缓存成功: {key} -> {cache_key} "
                            f"({len(final_data) / 1024:.1f}KB)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"数据缓存失败 {key}: {e}")
            return False
            
    def load_cached_data(self, key: str) -> Optional[Any]:
        """加载缓存的数据"""
        cache_key = self._generate_cache_key(key, "data")
        
        self.stats['total_requests'] += 1
        
        # 先检查内存缓存
        cached_data = self.memory_cache.get(cache_key)
        if cached_data is not None:
            self.stats['cache_hits'] += 1
            return cached_data
            
        # 检查磁盘缓存
        with self.lock:
            if cache_key not in self.cache_entries:
                self.stats['cache_misses'] += 1
                return None
                
            entry = self.cache_entries[cache_key]
            
        # 检查TTL
        ttl_hours = entry.metadata.get('ttl_hours', 24)
        if datetime.now() - entry.created_at > timedelta(hours=ttl_hours):
            self.remove_from_cache(cache_key)
            self.stats['cache_misses'] += 1
            return None
            
        try:
            # 读取缓存文件
            with open(entry.file_path, 'rb') as f:
                file_data = f.read()
                
            # 解压数据
            if entry.compressed:
                serialized_data = self._decompress_data(file_data)
            else:
                serialized_data = file_data
                
            # 反序列化
            data = pickle.loads(serialized_data)
            
            # 更新访问信息
            with self.lock:
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                
            # 添加到内存缓存
            self.memory_cache.put(cache_key, data)
            
            self.stats['cache_hits'] += 1
            
            return data
            
        except Exception as e:
            self.logger.error(f"加载缓存数据失败 {cache_key}: {e}")
            self.stats['cache_misses'] += 1
            return None
            
    def remove_from_cache(self, cache_key: str) -> bool:
        """从缓存中移除条目"""
        with self.lock:
            if cache_key not in self.cache_entries:
                return False
                
            entry = self.cache_entries[cache_key]
            
            # 删除文件
            try:
                if os.path.exists(entry.file_path):
                    os.remove(entry.file_path)
            except Exception as e:
                self.logger.warning(f"删除缓存文件失败 {entry.file_path}: {e}")
                
            # 从索引中移除
            del self.cache_entries[cache_key]
            
        # 从内存缓存中移除
        self.memory_cache.remove(cache_key)
        
        self.logger.debug(f"移除缓存条目: {cache_key}")
        
        return True
        
    def clear_cache(self, pattern: Optional[str] = None) -> int:
        """清空缓存"""
        removed_count = 0
        
        with self.lock:
            keys_to_remove = []
            
            for key in self.cache_entries.keys():
                if pattern is None or pattern in key:
                    keys_to_remove.append(key)
                    
        for key in keys_to_remove:
            if self.remove_from_cache(key):
                removed_count += 1
                
        if pattern is None:
            self.memory_cache.clear()
            
        self._save_cache_index()
        
        self.logger.info(f"清空缓存: 移除了 {removed_count} 个条目")
        
        return removed_count
        
    def optimize_storage(self) -> Dict[str, Any]:
        """优化存储"""
        self.logger.info("开始存储优化...")
        
        optimization_results = {
            'cleaned_expired': 0,
            'compressed_files': 0,
            'freed_space_mb': 0,
            'defragmented': False
        }
        
        # 1. 清理过期缓存
        initial_count = len(self.cache_entries)
        self._cleanup_expired_cache(max_age_days=3)  # 3天过期
        optimization_results['cleaned_expired'] = initial_count - len(self.cache_entries)
        
        # 2. 压缩未压缩的大文件
        compressed_count = 0
        freed_bytes = 0
        
        with self.lock:
            entries_to_compress = [
                (key, entry) for key, entry in self.cache_entries.items()
                if not entry.compressed and entry.size_bytes > 10240  # 大于10KB
            ]
            
        for key, entry in entries_to_compress:
            try:
                # 读取原文件
                with open(entry.file_path, 'rb') as f:
                    original_data = f.read()
                    
                # 压缩数据
                compressed_data, compression_ratio = self._compress_data(original_data)
                
                if compression_ratio < 0.9:  # 压缩率超过10%才替换
                    # 写入压缩数据
                    with open(entry.file_path, 'wb') as f:
                        f.write(compressed_data)
                        
                    # 更新条目信息
                    with self.lock:
                        entry.compressed = True
                        entry.size_bytes = len(compressed_data)
                        
                    freed_bytes += len(original_data) - len(compressed_data)
                    compressed_count += 1
                    
            except Exception as e:
                self.logger.warning(f"压缩缓存文件失败 {key}: {e}")
                
        optimization_results['compressed_files'] = compressed_count
        optimization_results['freed_space_mb'] = freed_bytes / 1024 / 1024
        
        # 3. 保存更新的索引
        self._save_cache_index()
        
        self.logger.info(f"存储优化完成: {optimization_results}")
        
        return optimization_results
        
    def get_storage_stats(self) -> StorageStats:
        """获取存储统计信息"""
        # 获取磁盘空间信息
        if PSUTIL_AVAILABLE:
            try:
                disk_usage = psutil.disk_usage(str(self.cache_dir))
                total_space_gb = disk_usage.total / 1024 / 1024 / 1024
                used_space_gb = disk_usage.used / 1024 / 1024 / 1024
                available_space_gb = disk_usage.free / 1024 / 1024 / 1024
            except:
                total_space_gb = used_space_gb = available_space_gb = 0.0
        else:
            total_space_gb = used_space_gb = available_space_gb = 0.0
            
        # 计算缓存统计
        cache_size_bytes = self._get_cache_size_bytes()
        cache_size_mb = cache_size_bytes / 1024 / 1024
        
        # 计算压缩率
        total_compressed = sum(1 for entry in self.cache_entries.values() if entry.compressed)
        compression_ratio = total_compressed / len(self.cache_entries) if self.cache_entries else 0.0
        
        # 计算命中率
        hit_rate = (self.stats['cache_hits'] / self.stats['total_requests'] 
                   if self.stats['total_requests'] > 0 else 0.0)
        
        return StorageStats(
            total_space_gb=total_space_gb,
            used_space_gb=used_space_gb,
            available_space_gb=available_space_gb,
            cache_size_mb=cache_size_mb,
            cache_entries=len(self.cache_entries),
            compression_ratio=compression_ratio,
            hit_rate=hit_rate
        )
        
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        stats = self.get_storage_stats()
        
        return {
            'cache_directory': str(self.cache_dir),
            'max_cache_size_mb': self.max_cache_size_mb,
            'current_cache_size_mb': stats.cache_size_mb,
            'cache_entries': stats.cache_entries,
            'memory_cache_size': self.memory_cache.size(),
            'hit_rate': stats.hit_rate,
            'compression_ratio': stats.compression_ratio,
            'total_requests': self.stats['total_requests'],
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'bytes_saved_by_compression': self.stats['compression_saves'],
            'storage_stats': asdict(stats)
        }

# 全局存储优化器实例
_global_storage_optimizer: Optional[StorageOptimizer] = None

def get_storage_optimizer(cache_dir: str = "./cache", 
                         max_cache_size_mb: int = 500) -> StorageOptimizer:
    """获取存储优化器"""
    global _global_storage_optimizer
    
    if _global_storage_optimizer is None:
        _global_storage_optimizer = StorageOptimizer(cache_dir, max_cache_size_mb)
        
    return _global_storage_optimizer

def clear_global_cache() -> None:
    """清空全局缓存"""
    global _global_storage_optimizer
    
    if _global_storage_optimizer is not None:
        _global_storage_optimizer.clear_cache()
        _global_storage_optimizer = None

if __name__ == "__main__":
    # 测试代码
    import numpy as np
    
    print("存储优化器测试")
    print("=" * 50)
    
    # 创建存储优化器
    optimizer = StorageOptimizer(cache_dir="./test_cache", max_cache_size_mb=100)
    
    # 测试数据缓存
    print("\n测试数据缓存...")
    test_data = {
        'array': np.random.rand(1000, 1000),
        'text': "这是一个测试字符串" * 1000,
        'numbers': list(range(10000))
    }
    
    # 缓存数据
    for key, data in test_data.items():
        success = optimizer.cache_data(data, f"test_{key}", compress=True)
        print(f"  缓存 {key}: {'成功' if success else '失败'}")
        
    # 加载数据
    print("\n测试数据加载...")
    for key in test_data.keys():
        loaded_data = optimizer.load_cached_data(f"test_{key}")
        print(f"  加载 {key}: {'成功' if loaded_data is not None else '失败'}")
        
    # 获取统计信息
    print("\n缓存统计信息:")
    cache_info = optimizer.get_cache_info()
    for key, value in cache_info.items():
        if key != 'storage_stats':
            print(f"  {key}: {value}")
            
    # 存储优化
    print("\n执行存储优化...")
    optimization_results = optimizer.optimize_storage()
    for key, value in optimization_results.items():
        print(f"  {key}: {value}")
        
    # 清理测试缓存
    print("\n清理测试缓存...")
    removed_count = optimizer.clear_cache()
    print(f"  移除了 {removed_count} 个缓存条目")
    
    print("\n测试完成")