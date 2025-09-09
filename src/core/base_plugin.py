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
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.status = PluginStatus.UNLOADED
        self._metadata = None
        self._initialized = False
    
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