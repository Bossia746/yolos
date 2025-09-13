"""插件基类定义"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import logging
from enum import Enum


class PluginStatus(Enum):
    """插件状态枚举"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    RUNNING = "running"
    ERROR = "error"
    DISABLED = "disabled"


class PluginCapability(Enum):
    """插件能力枚举"""
    # 人类识别能力
    FACE_DETECTION = "face_detection"
    FACE_RECOGNITION = "face_recognition"
    GESTURE_RECOGNITION = "gesture_recognition"
    POSE_ESTIMATION = "pose_estimation"
    FALL_DETECTION = "fall_detection"
    AGE_ESTIMATION = "age_estimation"
    EMOTION_RECOGNITION = "emotion_recognition"
    
    # 宠物识别能力
    PET_DETECTION = "pet_detection"
    CAT_RECOGNITION = "cat_recognition"
    DOG_RECOGNITION = "dog_recognition"
    BIRD_RECOGNITION = "bird_recognition"
    
    # 植物识别能力
    PLANT_RECOGNITION = "plant_recognition"
    INDOOR_PLANT_RECOGNITION = "indoor_plant_recognition"
    WILD_PLANT_RECOGNITION = "wild_plant_recognition"
    
    # 物体识别能力
    OBJECT_DETECTION = "object_detection"
    VEHICLE_DETECTION = "vehicle_detection"
    TRAFFIC_SIGN_RECOGNITION = "traffic_sign_recognition"
    
    # 系统能力
    REAL_TIME_PROCESSING = "real_time_processing"
    BATCH_PROCESSING = "batch_processing"
    GPU_ACCELERATION = "gpu_acceleration"
    EDGE_COMPUTING = "edge_computing"


class PluginMetadata:
    """插件元数据"""
    
    def __init__(self, 
                 name: str,
                 version: str,
                 description: str,
                 author: str,
                 capabilities: List[PluginCapability],
                 dependencies: List[str] = None,
                 min_memory_mb: int = 128,
                 min_cpu_cores: int = 1,
                 requires_gpu: bool = False,
                 supported_platforms: List[str] = None):
        self.name = name
        self.version = version
        self.description = description
        self.author = author
        self.capabilities = capabilities or []
        self.dependencies = dependencies or []
        self.min_memory_mb = min_memory_mb
        self.min_cpu_cores = min_cpu_cores
        self.requires_gpu = requires_gpu
        self.supported_platforms = supported_platforms or ["all"]
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "capabilities": [cap.value for cap in self.capabilities],
            "dependencies": self.dependencies,
            "requirements": {
                "min_memory_mb": self.min_memory_mb,
                "min_cpu_cores": self.min_cpu_cores,
                "requires_gpu": self.requires_gpu,
                "supported_platforms": self.supported_platforms
            }
        }


class BasePlugin(ABC):
    """插件基类
    
    所有插件都必须继承此类并实现抽象方法
    提供标准化的插件生命周期管理
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._status = PluginStatus.UNLOADED
        self._metadata = self._create_metadata()
        self._event_handlers = {}
        self._resources = []
        
    @property
    def metadata(self) -> PluginMetadata:
        """获取插件元数据"""
        return self._metadata
    
    @property
    def status(self) -> PluginStatus:
        """获取插件状态"""
        return self._status
    
    def set_status(self, status: PluginStatus):
        """设置插件状态"""
        old_status = self._status
        self._status = status
        self.logger.debug(f"Plugin status changed: {old_status.value} -> {status.value}")
        self._on_status_changed(old_status, status)
    
    @abstractmethod
    def _create_metadata(self) -> PluginMetadata:
        """创建插件元数据
        
        Returns:
            PluginMetadata: 插件元数据
        """
        pass
    
    def validate_requirements(self, system_info: Dict) -> bool:
        """验证系统要求
        
        Args:
            system_info: 系统信息
            
        Returns:
            bool: 是否满足要求
        """
        # 检查内存要求
        if system_info.get('memory_mb', 0) < self.metadata.min_memory_mb:
            self.logger.error(f"Insufficient memory: {system_info.get('memory_mb')} < {self.metadata.min_memory_mb}")
            return False
        
        # 检查CPU要求
        if system_info.get('cpu_cores', 0) < self.metadata.min_cpu_cores:
            self.logger.error(f"Insufficient CPU cores: {system_info.get('cpu_cores')} < {self.metadata.min_cpu_cores}")
            return False
        
        # 检查GPU要求
        if self.metadata.requires_gpu and not system_info.get('has_gpu', False):
            self.logger.error("GPU required but not available")
            return False
        
        # 检查平台支持
        platform = system_info.get('platform', 'unknown')
        if 'all' not in self.metadata.supported_platforms and platform not in self.metadata.supported_platforms:
            self.logger.error(f"Platform {platform} not supported")
            return False
        
        return True
    
    def initialize(self) -> bool:
        """初始化插件
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            self.logger.info(f"Initializing plugin {self.metadata.name}")
            
            # 预初始化钩子
            if not self._pre_initialize():
                return False
            
            # 执行具体初始化
            if not self._do_initialize():
                return False
            
            # 后初始化钩子
            if not self._post_initialize():
                return False
            
            self.logger.info(f"Plugin {self.metadata.name} initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize plugin {self.metadata.name}: {e}")
            return False
    
    def cleanup(self):
        """清理插件资源"""
        try:
            self.logger.info(f"Cleaning up plugin {self.metadata.name}")
            
            # 预清理钩子
            self._pre_cleanup()
            
            # 执行具体清理
            self._do_cleanup()
            
            # 清理资源
            for resource in self._resources:
                try:
                    if hasattr(resource, 'close'):
                        resource.close()
                    elif hasattr(resource, 'cleanup'):
                        resource.cleanup()
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup resource {resource}: {e}")
            
            self._resources.clear()
            
            # 后清理钩子
            self._post_cleanup()
            
            self.logger.info(f"Plugin {self.metadata.name} cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup plugin {self.metadata.name}: {e}")
    
    def register_resource(self, resource: Any):
        """注册需要清理的资源
        
        Args:
            resource: 资源对象
        """
        self._resources.append(resource)
    
    def register_event_handler(self, event_type: str, handler):
        """注册事件处理器
        
        Args:
            event_type: 事件类型
            handler: 处理器函数
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
    
    def handle_event(self, event_type: str, data: Any):
        """处理事件
        
        Args:
            event_type: 事件类型
            data: 事件数据
        """
        if event_type in self._event_handlers:
            for handler in self._event_handlers[event_type]:
                try:
                    handler(data)
                except Exception as e:
                    self.logger.error(f"Error in event handler for {event_type}: {e}")
    
    # 生命周期钩子方法
    def _pre_initialize(self) -> bool:
        """预初始化钩子"""
        return True
    
    @abstractmethod
    def _do_initialize(self) -> bool:
        """执行具体初始化逻辑"""
        pass
    
    def _post_initialize(self) -> bool:
        """后初始化钩子"""
        return True
    
    def _pre_cleanup(self):
        """预清理钩子"""
        pass
    
    def _do_cleanup(self):
        """执行具体清理逻辑"""
        pass
    
    def _post_cleanup(self):
        """后清理钩子"""
        pass
    
    def _on_status_changed(self, old_status: PluginStatus, new_status: PluginStatus):
        """状态变化回调
        
        Args:
            old_status: 旧状态
            new_status: 新状态
        """
        pass
    
    
        
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """插件元数据
        
        Returns:
            PluginMetadata: 插件的元数据信息
        """
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """初始化插件
        
        Returns:
            bool: 初始化是否成功
        """
        pass
    
    @abstractmethod
    def process(self, input_data: Any, **kwargs) -> Any:
        """处理输入数据
        
        Args:
            input_data: 输入数据
            **kwargs: 额外参数
            
        Returns:
            Any: 处理结果
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """清理资源"""
        pass
    
    def validate_requirements(self, system_info: Dict) -> bool:
        """验证系统是否满足插件要求
        
        Args:
            system_info: 系统信息
            
        Returns:
            bool: 是否满足要求
        """
        metadata = self.metadata
        
        # 检查内存要求
        if system_info.get('memory_mb', 0) < metadata.min_memory_mb:
            self.logger.warning(f"Insufficient memory: {system_info.get('memory_mb')}MB < {metadata.min_memory_mb}MB")
            return False
        
        # 检查CPU要求
        if system_info.get('cpu_cores', 0) < metadata.min_cpu_cores:
            self.logger.warning(f"Insufficient CPU cores: {system_info.get('cpu_cores')} < {metadata.min_cpu_cores}")
            return False
        
        # 检查GPU要求
        if metadata.requires_gpu and not system_info.get('has_gpu', False):
            self.logger.warning("GPU required but not available")
            return False
        
        # 检查平台支持
        current_platform = system_info.get('platform', 'unknown')
        if 'all' not in metadata.supported_platforms and current_platform not in metadata.supported_platforms:
            self.logger.warning(f"Platform {current_platform} not supported")
            return False
        
        return True
    
    def get_status(self) -> PluginStatus:
        """获取插件状态"""
        return self.status
    
    def set_status(self, status: PluginStatus) -> None:
        """设置插件状态"""
        old_status = self.status
        self.status = status
        self.logger.debug(f"Plugin status changed: {old_status.value} -> {status.value}")
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self.config.get(key, default)
    
    def update_config(self, config: Dict) -> None:
        """更新配置"""
        self.config.update(config)
        self.logger.info("Plugin configuration updated")
    
    def has_capability(self, capability: PluginCapability) -> bool:
        """检查是否具有指定能力"""
        return capability in self.metadata.capabilities
    
    def get_capabilities(self) -> List[PluginCapability]:
        """获取所有能力"""
        return self.metadata.capabilities.copy()
    
    def __str__(self) -> str:
        return f"{self.metadata.name} v{self.metadata.version} ({self.status.value})"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.metadata.name} v{self.metadata.version}>"


class DomainPlugin(BasePlugin):
    """领域插件基类
    
    用于特定领域的识别任务（如人脸识别、物体检测等）
    """
    
    @abstractmethod
    def detect(self, image: Any, **kwargs) -> List[Dict]:
        """检测功能
        
        Args:
            image: 输入图像
            **kwargs: 额外参数
            
        Returns:
            List[Dict]: 检测结果列表
        """
        pass
    
    def process(self, input_data: Any, **kwargs) -> Any:
        """处理输入数据（默认调用detect方法）"""
        return self.detect(input_data, **kwargs)


class PlatformPlugin(BasePlugin):
    """平台插件基类
    
    用于特定平台的适配（如摄像头、通信、存储等）
    """
    
    @abstractmethod
    def connect(self) -> bool:
        """连接到平台资源
        
        Returns:
            bool: 连接是否成功
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """断开平台资源连接"""
        pass
    
    def process(self, input_data: Any, **kwargs) -> Any:
        """处理输入数据（平台相关处理）"""
        return input_data


class UtilityPlugin(BasePlugin):
    """工具插件基类
    
    用于提供辅助功能（如数据预处理、后处理等）
    """
    
    def process(self, input_data: Any, **kwargs) -> Any:
        """处理输入数据（工具处理）"""
        return self.transform(input_data, **kwargs)
    
    @abstractmethod
    def transform(self, data: Any, **kwargs) -> Any:
        """数据转换
        
        Args:
            data: 输入数据
            **kwargs: 额外参数
            
        Returns:
            Any: 转换后的数据
        """
        pass