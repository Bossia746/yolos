"""数据管理器

提供统一的数据存储和管理功能，包括：
- 模拟数据管理
- 训练数据管理
- 配置数据管理
- 缓存管理
- 数据版本控制
- 数据同步和备份
"""

import os
import json
import pickle
import hashlib
import shutil
import sqlite3
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
import logging

@dataclass
class DataInfo:
    """数据信息"""
    id: str
    name: str
    type: str  # 'mock', 'training', 'config', 'cache'
    path: str
    size: int
    checksum: str
    created_at: datetime
    updated_at: datetime
    version: str = '1.0.0'
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
@dataclass
class StorageConfig:
    """存储配置"""
    base_path: str
    max_size_mb: int = 1024  # 1GB默认
    compression_enabled: bool = True
    backup_enabled: bool = True
    cleanup_enabled: bool = True
    retention_days: int = 30
    cache_size_mb: int = 256  # 256MB缓存
    
class DataStorage:
    """数据存储基类"""
    
    def __init__(self, storage_path: str, config: StorageConfig):
        self.storage_path = Path(storage_path)
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 创建存储目录
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
    def store(self, data: Any, data_info: DataInfo) -> bool:
        """存储数据"""
        raise NotImplementedError
        
    def retrieve(self, data_id: str) -> Tuple[Any, DataInfo]:
        """检索数据"""
        raise NotImplementedError
        
    def delete(self, data_id: str) -> bool:
        """删除数据"""
        raise NotImplementedError
        
    def list_data(self, data_type: Optional[str] = None) -> List[DataInfo]:
        """列出数据"""
        raise NotImplementedError
        
    def get_storage_info(self) -> Dict[str, Any]:
        """获取存储信息"""
        raise NotImplementedError

class FileSystemStorage(DataStorage):
    """文件系统存储"""
    
    def __init__(self, storage_path: str, config: StorageConfig):
        super().__init__(storage_path, config)
        
        # 创建子目录
        self.data_dir = self.storage_path / 'data'
        self.metadata_dir = self.storage_path / 'metadata'
        self.backup_dir = self.storage_path / 'backup'
        
        for dir_path in [self.data_dir, self.metadata_dir, self.backup_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # 初始化数据库
        self.db_path = self.storage_path / 'data_index.db'
        self._init_database()
        
        # 线程锁
        self._lock = threading.RLock()
        
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS data_index (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    path TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    checksum TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    version TEXT NOT NULL,
                    metadata TEXT,
                    tags TEXT
                )
            ''')
            conn.commit()
            
    def _calculate_checksum(self, data: Any) -> str:
        """计算数据校验和"""
        if isinstance(data, (str, bytes)):
            content = data.encode() if isinstance(data, str) else data
        else:
            content = pickle.dumps(data)
        return hashlib.md5(content).hexdigest()
        
    def _serialize_data(self, data: Any, data_info: DataInfo) -> bytes:
        """序列化数据"""
        if data_info.type == 'config' and isinstance(data, dict):
            return json.dumps(data, indent=2).encode()
        elif isinstance(data, (str, bytes)):
            return data.encode() if isinstance(data, str) else data
        else:
            return pickle.dumps(data)
            
    def _deserialize_data(self, content: bytes, data_info: DataInfo) -> Any:
        """反序列化数据"""
        if data_info.type == 'config':
            try:
                return json.loads(content.decode())
            except (json.JSONDecodeError, UnicodeDecodeError):
                return pickle.loads(content)
        else:
            try:
                return pickle.loads(content)
            except pickle.UnpicklingError:
                return content.decode()
                
    def store(self, data: Any, data_info: DataInfo) -> bool:
        """存储数据"""
        with self._lock:
            try:
                # 序列化数据
                serialized_data = self._serialize_data(data, data_info)
                
                # 计算校验和
                data_info.checksum = self._calculate_checksum(serialized_data)
                data_info.size = len(serialized_data)
                data_info.updated_at = datetime.now()
                
                # 确定存储路径
                data_file = self.data_dir / f"{data_info.id}.dat"
                metadata_file = self.metadata_dir / f"{data_info.id}.json"
                
                # 备份现有数据
                if data_file.exists() and self.config.backup_enabled:
                    backup_file = self.backup_dir / f"{data_info.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.dat"
                    shutil.copy2(data_file, backup_file)
                    
                # 写入数据文件
                if self.config.compression_enabled and len(serialized_data) > 1024:
                    import gzip
                    with gzip.open(data_file, 'wb') as f:
                        f.write(serialized_data)
                    data_info.metadata['compressed'] = True
                else:
                    with open(data_file, 'wb') as f:
                        f.write(serialized_data)
                    data_info.metadata['compressed'] = False
                    
                # 写入元数据
                metadata = {
                    'id': data_info.id,
                    'name': data_info.name,
                    'type': data_info.type,
                    'size': data_info.size,
                    'checksum': data_info.checksum,
                    'created_at': data_info.created_at.isoformat(),
                    'updated_at': data_info.updated_at.isoformat(),
                    'version': data_info.version,
                    'metadata': data_info.metadata,
                    'tags': data_info.tags
                }
                
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
                # 更新数据库索引
                data_info.path = str(data_file)
                self._update_index(data_info)
                
                self.logger.info(f"数据已存储: {data_info.id} ({data_info.size} bytes)")
                return True
                
            except Exception as e:
                self.logger.error(f"存储数据失败: {e}")
                return False
                
    def retrieve(self, data_id: str) -> Tuple[Any, DataInfo]:
        """检索数据"""
        with self._lock:
            try:
                # 从数据库获取信息
                data_info = self._get_data_info(data_id)
                if not data_info:
                    raise ValueError(f"数据不存在: {data_id}")
                    
                # 读取数据文件
                data_file = Path(data_info.path)
                if not data_file.exists():
                    raise FileNotFoundError(f"数据文件不存在: {data_file}")
                    
                # 读取数据
                if data_info.metadata.get('compressed', False):
                    import gzip
                    with gzip.open(data_file, 'rb') as f:
                        content = f.read()
                else:
                    with open(data_file, 'rb') as f:
                        content = f.read()
                        
                # 验证校验和
                current_checksum = hashlib.md5(content).hexdigest()
                if current_checksum != data_info.checksum:
                    self.logger.warning(f"数据校验和不匹配: {data_id}")
                    
                # 反序列化数据
                data = self._deserialize_data(content, data_info)
                
                self.logger.debug(f"数据已检索: {data_id}")
                return data, data_info
                
            except Exception as e:
                self.logger.error(f"检索数据失败: {e}")
                raise
                
    def delete(self, data_id: str) -> bool:
        """删除数据"""
        with self._lock:
            try:
                # 获取数据信息
                data_info = self._get_data_info(data_id)
                if not data_info:
                    return False
                    
                # 删除文件
                data_file = Path(data_info.path)
                metadata_file = self.metadata_dir / f"{data_id}.json"
                
                if data_file.exists():
                    data_file.unlink()
                if metadata_file.exists():
                    metadata_file.unlink()
                    
                # 从数据库删除
                with sqlite3.connect(str(self.db_path)) as conn:
                    conn.execute('DELETE FROM data_index WHERE id = ?', (data_id,))
                    conn.commit()
                    
                self.logger.info(f"数据已删除: {data_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"删除数据失败: {e}")
                return False
                
    def list_data(self, data_type: Optional[str] = None) -> List[DataInfo]:
        """列出数据"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                if data_type:
                    cursor = conn.execute(
                        'SELECT * FROM data_index WHERE type = ? ORDER BY updated_at DESC',
                        (data_type,)
                    )
                else:
                    cursor = conn.execute(
                        'SELECT * FROM data_index ORDER BY updated_at DESC'
                    )
                    
                data_list = []
                for row in cursor.fetchall():
                    data_info = DataInfo(
                        id=row[0],
                        name=row[1],
                        type=row[2],
                        path=row[3],
                        size=row[4],
                        checksum=row[5],
                        created_at=datetime.fromisoformat(row[6]),
                        updated_at=datetime.fromisoformat(row[7]),
                        version=row[8],
                        metadata=json.loads(row[9]) if row[9] else {},
                        tags=json.loads(row[10]) if row[10] else []
                    )
                    data_list.append(data_info)
                    
                return data_list
                
        except Exception as e:
            self.logger.error(f"列出数据失败: {e}")
            return []
            
    def _update_index(self, data_info: DataInfo):
        """更新数据库索引"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO data_index 
                (id, name, type, path, size, checksum, created_at, updated_at, version, metadata, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data_info.id,
                data_info.name,
                data_info.type,
                data_info.path,
                data_info.size,
                data_info.checksum,
                data_info.created_at.isoformat(),
                data_info.updated_at.isoformat(),
                data_info.version,
                json.dumps(data_info.metadata),
                json.dumps(data_info.tags)
            ))
            conn.commit()
            
    def _get_data_info(self, data_id: str) -> Optional[DataInfo]:
        """获取数据信息"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                'SELECT * FROM data_index WHERE id = ?',
                (data_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return DataInfo(
                    id=row[0],
                    name=row[1],
                    type=row[2],
                    path=row[3],
                    size=row[4],
                    checksum=row[5],
                    created_at=datetime.fromisoformat(row[6]),
                    updated_at=datetime.fromisoformat(row[7]),
                    version=row[8],
                    metadata=json.loads(row[9]) if row[9] else {},
                    tags=json.loads(row[10]) if row[10] else []
                )
            return None
            
    def get_storage_info(self) -> Dict[str, Any]:
        """获取存储信息"""
        try:
            # 计算存储使用情况
            total_size = 0
            file_count = 0
            
            for file_path in self.data_dir.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
                    
            # 按类型统计
            type_stats = {}
            data_list = self.list_data()
            
            for data_info in data_list:
                if data_info.type not in type_stats:
                    type_stats[data_info.type] = {'count': 0, 'size': 0}
                type_stats[data_info.type]['count'] += 1
                type_stats[data_info.type]['size'] += data_info.size
                
            return {
                'total_size_mb': total_size / (1024 * 1024),
                'file_count': file_count,
                'data_count': len(data_list),
                'type_statistics': type_stats,
                'storage_path': str(self.storage_path),
                'config': {
                    'max_size_mb': self.config.max_size_mb,
                    'compression_enabled': self.config.compression_enabled,
                    'backup_enabled': self.config.backup_enabled
                }
            }
            
        except Exception as e:
            self.logger.error(f"获取存储信息失败: {e}")
            return {}
            
    def cleanup_old_data(self) -> int:
        """清理过期数据"""
        if not self.config.cleanup_enabled:
            return 0
            
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
            deleted_count = 0
            
            data_list = self.list_data()
            for data_info in data_list:
                if data_info.updated_at < cutoff_date:
                    if self.delete(data_info.id):
                        deleted_count += 1
                        
            self.logger.info(f"清理了 {deleted_count} 个过期数据")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"清理数据失败: {e}")
            return 0

class CacheManager:
    """缓存管理器"""
    
    def __init__(self, cache_size_mb: int = 256):
        self.cache_size_bytes = cache_size_mb * 1024 * 1024
        self.cache = {}
        self.access_times = {}
        self.cache_sizes = {}
        self.total_size = 0
        self._lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        with self._lock:
            if key in self.cache:
                self.access_times[key] = datetime.now()
                return self.cache[key]
            return None
            
    def put(self, key: str, value: Any) -> bool:
        """存储缓存数据"""
        with self._lock:
            try:
                # 计算数据大小
                value_size = len(pickle.dumps(value))
                
                # 检查是否需要清理缓存
                while self.total_size + value_size > self.cache_size_bytes and self.cache:
                    self._evict_lru()
                    
                # 存储数据
                if key in self.cache:
                    self.total_size -= self.cache_sizes[key]
                    
                self.cache[key] = value
                self.cache_sizes[key] = value_size
                self.access_times[key] = datetime.now()
                self.total_size += value_size
                
                return True
                
            except Exception as e:
                logging.error(f"缓存存储失败: {e}")
                return False
                
    def _evict_lru(self):
        """清理最近最少使用的缓存"""
        if not self.access_times:
            return
            
        # 找到最久未访问的键
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        # 删除缓存
        self.total_size -= self.cache_sizes[lru_key]
        del self.cache[lru_key]
        del self.cache_sizes[lru_key]
        del self.access_times[lru_key]
        
    def clear(self):
        """清空缓存"""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.cache_sizes.clear()
            self.total_size = 0
            
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self._lock:
            return {
                'total_size_mb': self.total_size / (1024 * 1024),
                'item_count': len(self.cache),
                'cache_size_limit_mb': self.cache_size_bytes / (1024 * 1024),
                'utilization': self.total_size / self.cache_size_bytes if self.cache_size_bytes > 0 else 0
            }

class DataManager:
    """数据管理器"""
    
    def __init__(self, base_path: str, config: Optional[StorageConfig] = None):
        self.base_path = Path(base_path)
        self.config = config or StorageConfig(base_path=base_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 创建存储实例
        self.storage = FileSystemStorage(base_path, self.config)
        
        # 创建缓存管理器
        self.cache = CacheManager(self.config.cache_size_mb)
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        self.logger.info(f"数据管理器已初始化: {base_path}")
        
    def store_mock_data(self, data: Any, name: str, tags: Optional[List[str]] = None) -> str:
        """存储模拟数据"""
        data_id = f"mock_{hashlib.md5(name.encode()).hexdigest()[:8]}"
        data_info = DataInfo(
            id=data_id,
            name=name,
            type='mock',
            path='',  # 将由存储层设置
            size=0,   # 将由存储层计算
            checksum='',  # 将由存储层计算
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=tags or []
        )
        
        if self.storage.store(data, data_info):
            self.logger.info(f"模拟数据已存储: {name} -> {data_id}")
            return data_id
        else:
            raise RuntimeError(f"存储模拟数据失败: {name}")
            
    def store_training_data(self, data: Any, name: str, version: str = '1.0.0', 
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """存储训练数据"""
        data_id = f"train_{hashlib.md5(f'{name}_{version}'.encode()).hexdigest()[:8]}"
        data_info = DataInfo(
            id=data_id,
            name=name,
            type='training',
            path='',
            size=0,
            checksum='',
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=version,
            metadata=metadata or {},
            tags=['training']
        )
        
        if self.storage.store(data, data_info):
            self.logger.info(f"训练数据已存储: {name} v{version} -> {data_id}")
            return data_id
        else:
            raise RuntimeError(f"存储训练数据失败: {name}")
            
    def store_config_data(self, config: Dict[str, Any], name: str) -> str:
        """存储配置数据"""
        data_id = f"config_{hashlib.md5(name.encode()).hexdigest()[:8]}"
        data_info = DataInfo(
            id=data_id,
            name=name,
            type='config',
            path='',
            size=0,
            checksum='',
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=['config']
        )
        
        if self.storage.store(config, data_info):
            self.logger.info(f"配置数据已存储: {name} -> {data_id}")
            return data_id
        else:
            raise RuntimeError(f"存储配置数据失败: {name}")
            
    def retrieve_data(self, data_id: str, use_cache: bool = True) -> Tuple[Any, DataInfo]:
        """检索数据"""
        # 尝试从缓存获取
        if use_cache:
            cached_data = self.cache.get(data_id)
            if cached_data is not None:
                data, data_info = cached_data
                self.logger.debug(f"从缓存检索数据: {data_id}")
                return data, data_info
                
        # 从存储检索
        data, data_info = self.storage.retrieve(data_id)
        
        # 存储到缓存
        if use_cache:
            self.cache.put(data_id, (data, data_info))
            
        return data, data_info
        
    def delete_data(self, data_id: str) -> bool:
        """删除数据"""
        # 从缓存删除
        if data_id in self.cache.cache:
            del self.cache.cache[data_id]
            
        # 从存储删除
        return self.storage.delete(data_id)
        
    def list_data_by_type(self, data_type: str) -> List[DataInfo]:
        """按类型列出数据"""
        return self.storage.list_data(data_type)
        
    def search_data(self, query: str, data_type: Optional[str] = None) -> List[DataInfo]:
        """搜索数据"""
        data_list = self.storage.list_data(data_type)
        
        # 简单的文本搜索
        results = []
        query_lower = query.lower()
        
        for data_info in data_list:
            if (query_lower in data_info.name.lower() or 
                query_lower in ' '.join(data_info.tags).lower() or
                any(query_lower in str(v).lower() for v in data_info.metadata.values())):
                results.append(data_info)
                
        return results
        
    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计"""
        storage_info = self.storage.get_storage_info()
        cache_stats = self.cache.get_stats()
        
        return {
            'storage': storage_info,
            'cache': cache_stats,
            'total_data_count': len(self.storage.list_data())
        }
        
    def cleanup_data(self) -> Dict[str, int]:
        """清理数据"""
        results = {
            'deleted_files': 0,
            'cache_cleared': 0
        }
        
        # 清理过期数据
        if hasattr(self.storage, 'cleanup_old_data'):
            results['deleted_files'] = self.storage.cleanup_old_data()
            
        # 清理缓存
        cache_items = len(self.cache.cache)
        self.cache.clear()
        results['cache_cleared'] = cache_items
        
        return results
        
    def backup_data(self, backup_path: str) -> bool:
        """备份数据"""
        try:
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制整个存储目录
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_target = backup_dir / f"yolos_data_backup_{timestamp}"
            
            shutil.copytree(self.storage.storage_path, backup_target)
            
            self.logger.info(f"数据已备份到: {backup_target}")
            return True
            
        except Exception as e:
            self.logger.error(f"数据备份失败: {e}")
            return False
            
    def restore_data(self, backup_path: str) -> bool:
        """恢复数据"""
        try:
            backup_dir = Path(backup_path)
            if not backup_dir.exists():
                raise FileNotFoundError(f"备份目录不存在: {backup_path}")
                
            # 清空当前存储
            if self.storage.storage_path.exists():
                shutil.rmtree(self.storage.storage_path)
                
            # 恢复备份
            shutil.copytree(backup_dir, self.storage.storage_path)
            
            # 重新初始化存储
            self.storage = FileSystemStorage(str(self.storage.storage_path), self.config)
            
            # 清空缓存
            self.cache.clear()
            
            self.logger.info(f"数据已从备份恢复: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"数据恢复失败: {e}")
            return False
            
    def close(self):
        """关闭数据管理器"""
        self.executor.shutdown(wait=True)
        self.cache.clear()
        self.logger.info("数据管理器已关闭")