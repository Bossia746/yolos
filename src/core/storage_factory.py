"""存储工厂

根据配置和环境创建合适的存储实例，支持：
- 多种存储后端（文件系统、数据库、云存储等）
- 环境自适应配置
- 平台特定优化
- 动态配置加载
"""

import os
import yaml
import platform
from typing import Dict, Any, Optional, Type
from pathlib import Path
import logging
from dataclasses import dataclass

from .data_manager import DataManager, StorageConfig, DataStorage, FileSystemStorage

@dataclass
class StorageEnvironment:
    """存储环境信息"""
    platform: str  # windows, linux, macos, esp32, raspberry_pi
    environment: str  # development, production, testing, embedded, edge, cloud
    available_memory_mb: int
    available_storage_mb: int
    cpu_cores: int
    is_embedded: bool = False
    has_network: bool = True
    
class StorageConfigLoader:
    """存储配置加载器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._config_cache = {}
        
    def _get_default_config_path(self) -> str:
        """获取默认配置文件路径"""
        # 尝试多个可能的配置文件位置
        possible_paths = [
            "./config/storage_config.yaml",
            "../config/storage_config.yaml",
            "./storage_config.yaml",
            os.path.expanduser("~/.yolos/storage_config.yaml"),
            "/etc/yolos/storage_config.yaml"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        # 如果找不到配置文件，返回默认路径
        return "./config/storage_config.yaml"
        
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if self.config_path in self._config_cache:
            return self._config_cache[self.config_path]
            
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                self._config_cache[self.config_path] = config
                self.logger.info(f"配置文件已加载: {self.config_path}")
                return config
            else:
                self.logger.warning(f"配置文件不存在: {self.config_path}，使用默认配置")
                return self._get_default_config()
                
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}，使用默认配置")
            return self._get_default_config()
            
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'default': {
                'base_path': './data',
                'max_size_mb': 1024,
                'compression_enabled': True,
                'backup_enabled': True,
                'cleanup_enabled': True,
                'retention_days': 30,
                'cache_size_mb': 256
            }
        }
        
    def get_config_for_environment(self, env: StorageEnvironment) -> Dict[str, Any]:
        """获取特定环境的配置"""
        config = self.load_config()
        
        # 基础配置
        base_config = config.get('default', {})
        
        # 环境特定配置
        env_config = config.get(env.environment, {})
        
        # 平台特定配置
        platform_config = config.get('platforms', {}).get(env.platform, {})
        
        # 合并配置（优先级：平台 > 环境 > 默认）
        merged_config = {**base_config, **env_config, **platform_config}
        
        # 应用环境变量覆盖
        merged_config = self._apply_environment_variables(merged_config, config)
        
        # 根据硬件限制调整配置
        merged_config = self._adjust_for_hardware(merged_config, env)
        
        return merged_config
        
    def _apply_environment_variables(self, config: Dict[str, Any], 
                                   full_config: Dict[str, Any]) -> Dict[str, Any]:
        """应用环境变量覆盖"""
        env_vars = full_config.get('environment_variables', {})
        
        for env_var, config_key in env_vars.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # 尝试转换类型
                if config_key.endswith('_mb') or config_key.endswith('_days'):
                    try:
                        value = int(value)
                    except ValueError:
                        continue
                elif config_key.endswith('_enabled'):
                    value = value.lower() in ('true', '1', 'yes', 'on')
                    
                config[config_key] = value
                self.logger.info(f"环境变量覆盖: {config_key} = {value}")
                
        return config
        
    def _adjust_for_hardware(self, config: Dict[str, Any], 
                           env: StorageEnvironment) -> Dict[str, Any]:
        """根据硬件限制调整配置"""
        # 调整存储大小限制
        max_storage = min(config.get('max_size_mb', 1024), 
                         env.available_storage_mb // 2)  # 使用一半可用存储
        config['max_size_mb'] = max_storage
        
        # 调整缓存大小
        max_cache = min(config.get('cache_size_mb', 256),
                       env.available_memory_mb // 4)  # 使用1/4可用内存
        config['cache_size_mb'] = max_cache
        
        # 嵌入式设备特殊处理
        if env.is_embedded:
            config['compression_enabled'] = True  # 强制压缩
            config['backup_enabled'] = False     # 禁用备份
            config['cleanup_enabled'] = True     # 强制清理
            config['retention_days'] = min(config.get('retention_days', 30), 7)
            
        self.logger.debug(f"硬件调整后配置: max_size={max_storage}MB, cache={max_cache}MB")
        return config

class EnvironmentDetector:
    """环境检测器"""
    
    @staticmethod
    def detect_environment() -> StorageEnvironment:
        """检测当前运行环境"""
        # 检测平台
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        platform_name = 'unknown'
        is_embedded = False
        
        if system == 'windows':
            platform_name = 'windows'
        elif system == 'linux':
            if 'arm' in machine or 'aarch64' in machine:
                # 检测是否为树莓派
                if os.path.exists('/proc/device-tree/model'):
                    try:
                        with open('/proc/device-tree/model', 'r') as f:
                            model = f.read().lower()
                        if 'raspberry pi' in model:
                            platform_name = 'raspberry_pi'
                            is_embedded = True
                    except:
                        pass
                        
                if platform_name == 'unknown':
                    platform_name = 'linux'
            else:
                platform_name = 'linux'
        elif system == 'darwin':
            platform_name = 'macos'
        elif 'esp32' in machine or 'xtensa' in machine:
            platform_name = 'esp32'
            is_embedded = True
            
        # 检测环境类型
        environment = 'production'  # 默认
        
        if os.environ.get('YOLOS_ENV'):
            environment = os.environ['YOLOS_ENV'].lower()
        elif os.environ.get('NODE_ENV') == 'development':
            environment = 'development'
        elif os.environ.get('PYTEST_CURRENT_TEST'):
            environment = 'testing'
        elif is_embedded:
            environment = 'embedded'
            
        # 检测硬件资源
        available_memory_mb = EnvironmentDetector._get_available_memory_mb()
        available_storage_mb = EnvironmentDetector._get_available_storage_mb()
        cpu_cores = os.cpu_count() or 1
        
        # 检测网络连接
        has_network = EnvironmentDetector._check_network_connectivity()
        
        return StorageEnvironment(
            platform=platform_name,
            environment=environment,
            available_memory_mb=available_memory_mb,
            available_storage_mb=available_storage_mb,
            cpu_cores=cpu_cores,
            is_embedded=is_embedded,
            has_network=has_network
        )
        
    @staticmethod
    def _get_available_memory_mb() -> int:
        """获取可用内存（MB）"""
        try:
            import psutil
            return int(psutil.virtual_memory().available / (1024 * 1024))
        except ImportError:
            # 如果没有psutil，使用系统命令估算
            if platform.system() == 'Linux':
                try:
                    with open('/proc/meminfo', 'r') as f:
                        for line in f:
                            if line.startswith('MemAvailable:'):
                                kb = int(line.split()[1])
                                return kb // 1024
                except:
                    pass
            # 默认假设有512MB可用内存
            return 512
            
    @staticmethod
    def _get_available_storage_mb() -> int:
        """获取可用存储空间（MB）"""
        try:
            import shutil
            total, used, free = shutil.disk_usage('.')
            return int(free / (1024 * 1024))
        except:
            # 默认假设有1GB可用存储
            return 1024
            
    @staticmethod
    def _check_network_connectivity() -> bool:
        """检查网络连接"""
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except:
            return False

class StorageFactory:
    """存储工厂"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_loader = StorageConfigLoader(config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 注册存储后端
        self._storage_backends: Dict[str, Type[DataStorage]] = {
            'filesystem': FileSystemStorage,
            # 可以扩展其他存储后端
            # 'sqlite': SQLiteStorage,
            # 's3': S3Storage,
            # 'redis': RedisStorage,
        }
        
    def create_data_manager(self, 
                          environment: Optional[StorageEnvironment] = None,
                          custom_config: Optional[Dict[str, Any]] = None) -> DataManager:
        """创建数据管理器"""
        # 检测环境
        if environment is None:
            environment = EnvironmentDetector.detect_environment()
            
        self.logger.info(f"检测到环境: {environment.platform}/{environment.environment}")
        
        # 获取配置
        if custom_config:
            config_dict = custom_config
        else:
            config_dict = self.config_loader.get_config_for_environment(environment)
            
        # 创建存储配置
        storage_config = StorageConfig(
            base_path=config_dict.get('base_path', './data'),
            max_size_mb=config_dict.get('max_size_mb', 1024),
            compression_enabled=config_dict.get('compression_enabled', True),
            backup_enabled=config_dict.get('backup_enabled', True),
            cleanup_enabled=config_dict.get('cleanup_enabled', True),
            retention_days=config_dict.get('retention_days', 30),
            cache_size_mb=config_dict.get('cache_size_mb', 256)
        )
        
        # 确保存储路径存在
        storage_path = Path(storage_config.base_path)
        storage_path.mkdir(parents=True, exist_ok=True)
        
        # 创建数据管理器
        data_manager = DataManager(str(storage_path), storage_config)
        
        self.logger.info(f"数据管理器已创建: {storage_path} ")
        self.logger.info(f"配置: 最大{storage_config.max_size_mb}MB, "
                        f"缓存{storage_config.cache_size_mb}MB, "
                        f"压缩{'开启' if storage_config.compression_enabled else '关闭'}")
        
        return data_manager
        
    def create_storage_backend(self, 
                             backend_type: str,
                             storage_path: str,
                             config: StorageConfig) -> DataStorage:
        """创建存储后端"""
        if backend_type not in self._storage_backends:
            raise ValueError(f"不支持的存储后端: {backend_type}")
            
        backend_class = self._storage_backends[backend_type]
        return backend_class(storage_path, config)
        
    def register_storage_backend(self, name: str, backend_class: Type[DataStorage]):
        """注册新的存储后端"""
        self._storage_backends[name] = backend_class
        self.logger.info(f"存储后端已注册: {name}")
        
    def get_available_backends(self) -> List[str]:
        """获取可用的存储后端"""
        return list(self._storage_backends.keys())
        
    def get_environment_info(self) -> StorageEnvironment:
        """获取当前环境信息"""
        return EnvironmentDetector.detect_environment()
        
    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """验证配置有效性"""
        required_fields = ['base_path', 'max_size_mb', 'cache_size_mb']
        
        for field in required_fields:
            if field not in config:
                self.logger.error(f"配置缺少必需字段: {field}")
                return False
                
        # 验证数值范围
        if config.get('max_size_mb', 0) <= 0:
            self.logger.error("max_size_mb 必须大于0")
            return False
            
        if config.get('cache_size_mb', 0) <= 0:
            self.logger.error("cache_size_mb 必须大于0")
            return False
            
        # 验证路径
        base_path = config.get('base_path', '')
        if not base_path:
            self.logger.error("base_path 不能为空")
            return False
            
        try:
            Path(base_path).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"无法创建存储路径 {base_path}: {e}")
            return False
            
        return True
        
    def create_optimized_config(self, 
                              target_platform: str,
                              target_environment: str,
                              constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """创建优化配置"""
        # 创建模拟环境
        env = StorageEnvironment(
            platform=target_platform,
            environment=target_environment,
            available_memory_mb=constraints.get('memory_mb', 1024) if constraints else 1024,
            available_storage_mb=constraints.get('storage_mb', 4096) if constraints else 4096,
            cpu_cores=constraints.get('cpu_cores', 2) if constraints else 2,
            is_embedded=target_platform in ['esp32', 'raspberry_pi'],
            has_network=constraints.get('has_network', True) if constraints else True
        )
        
        # 获取优化配置
        config = self.config_loader.get_config_for_environment(env)
        
        # 应用额外约束
        if constraints:
            if 'max_storage_mb' in constraints:
                config['max_size_mb'] = min(config['max_size_mb'], constraints['max_storage_mb'])
            if 'max_cache_mb' in constraints:
                config['cache_size_mb'] = min(config['cache_size_mb'], constraints['max_cache_mb'])
                
        return config

# 全局工厂实例
_global_factory = None

def get_storage_factory(config_path: Optional[str] = None) -> StorageFactory:
    """获取全局存储工厂实例"""
    global _global_factory
    if _global_factory is None:
        _global_factory = StorageFactory(config_path)
    return _global_factory

def create_data_manager_for_current_environment() -> DataManager:
    """为当前环境创建数据管理器"""
    factory = get_storage_factory()
    return factory.create_data_manager()

def create_data_manager_for_platform(platform: str, 
                                    environment: str = 'production',
                                    constraints: Optional[Dict[str, Any]] = None) -> DataManager:
    """为指定平台创建数据管理器"""
    factory = get_storage_factory()
    
    # 创建优化配置
    config = factory.create_optimized_config(platform, environment, constraints)
    
    # 创建环境对象
    env = StorageEnvironment(
        platform=platform,
        environment=environment,
        available_memory_mb=constraints.get('memory_mb', 1024) if constraints else 1024,
        available_storage_mb=constraints.get('storage_mb', 4096) if constraints else 4096,
        cpu_cores=constraints.get('cpu_cores', 2) if constraints else 2,
        is_embedded=platform in ['esp32', 'raspberry_pi'],
        has_network=constraints.get('has_network', True) if constraints else True
    )
    
    return factory.create_data_manager(env, config)