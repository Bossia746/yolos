"""YOLOS 核心模块"""

from .plugin_manager import (
    PluginManager,
    PluginDependencyError,
    PluginLoadError
)

from .config_manager import (
    ConfigManager,
    ConfigValidationError
)

from .data_manager import (
    DataManager,
    DataStorage,
    FileSystemStorage,
    StorageConfig,
    DataInfo,
    CacheManager
)

from .storage_factory import (
    StorageFactory,
    StorageEnvironment,
    EnvironmentDetector,
    StorageConfigLoader,
    get_storage_factory,
    create_data_manager_for_current_environment,
    create_data_manager_for_platform
)

from .event_bus import EventBus
from .base_plugin import BasePlugin
from .cross_platform_manager import CrossPlatformManager, get_cross_platform_manager

__all__ = [
    # 插件管理
    'PluginManager',
    'PluginDependencyError',
    'PluginLoadError',
    
    # 配置管理
    'ConfigManager',
    'ConfigValidationError',
    
    # 数据存储
    'DataManager',
    'DataStorage',
    'FileSystemStorage',
    'StorageConfig',
    'DataInfo',
    'CacheManager',
    
    # 存储工厂
    'StorageFactory',
    'StorageEnvironment',
    'EnvironmentDetector',
    'StorageConfigLoader',
    'get_storage_factory',
    'create_data_manager_for_current_environment',
    'create_data_manager_for_platform',
    
    # 其他核心模块
    'EventBus',
    'BasePlugin',
    'CrossPlatformManager',
    'get_cross_platform_manager'
]