"""YOLOS 核心模块"""

# 类型定义
from .types import (
    TaskType,
    ObjectType,
    Status,
    Priority,
    BoundingBox,
    DetectionResult,
    ProcessingResult,
    Point2D,
    Keypoint,
    ImageInfo,
    create_detection_result,
    merge_results
)

# 异常处理
from .exceptions import (
    ErrorCode,
    YOLOSException,
    SystemException,
    ModelException,
    DataException,
    ImageException,
    DetectionException,
    HardwareException,
    APIException,
    PlatformException,
    ConfigurationError,
    ExceptionHandler,
    exception_handler
)

# 日志记录
from .logger import (
    get_logger,
    configure_logging,
    YOLOSLogger,
    log_function_call,
    log_class_methods
)

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
    # 类型定义
    'TaskType',
    'ObjectType',
    'Status',
    'Priority',
    'BoundingBox',
    'DetectionResult',
    'ProcessingResult',
    'Point2D',
    'Keypoint',
    'ImageInfo',
    'create_detection_result',
    'merge_results',
    
    # 异常处理
    'ErrorCode',
    'YOLOSException',
    'SystemException',
    'ModelException',
    'DataException',
    'ImageException',
    'DetectionException',
    'HardwareException',
    'APIException',
    'PlatformException',
    'ConfigurationError',
    'ExceptionHandler',
    'exception_handler',
    
    # 日志记录
    'get_logger',
    'configure_logging',
    'YOLOSLogger',
    'log_function_call',
    'log_class_methods',
    
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