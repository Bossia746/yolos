"""插件接口规范

定义标准化的插件接口和约定
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass


class PluginType(Enum):
    """插件类型枚举"""
    DETECTOR = "detector"  # 检测器插件
    RECOGNIZER = "recognizer"  # 识别器插件
    PROCESSOR = "processor"  # 处理器插件
    HARDWARE = "hardware"  # 硬件适配器插件
    PLATFORM = "platform"  # 平台集成插件
    UTILITY = "utility"  # 工具类插件
    FILTER = "filter"  # 过滤器插件
    TRANSFORMER = "transformer"  # 数据转换插件


class PluginPriority(Enum):
    """插件优先级"""
    CRITICAL = 1  # 关键插件
    HIGH = 2  # 高优先级
    NORMAL = 3  # 普通优先级
    LOW = 4  # 低优先级
    OPTIONAL = 5  # 可选插件


@dataclass
class PluginDependency:
    """插件依赖信息"""
    name: str
    version: Optional[str] = None
    optional: bool = False
    reason: Optional[str] = None


@dataclass
class PluginConfig:
    """插件配置信息"""
    enabled: bool = True
    auto_load: bool = True
    priority: PluginPriority = PluginPriority.NORMAL
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}


class IPluginLifecycle(ABC):
    """插件生命周期接口"""
    
    @abstractmethod
    def on_load(self) -> bool:
        """插件加载时调用
        
        Returns:
            bool: 加载是否成功
        """
        pass
    
    @abstractmethod
    def on_unload(self) -> bool:
        """插件卸载时调用
        
        Returns:
            bool: 卸载是否成功
        """
        pass
    
    def on_enable(self) -> bool:
        """插件启用时调用
        
        Returns:
            bool: 启用是否成功
        """
        return True
    
    def on_disable(self) -> bool:
        """插件禁用时调用
        
        Returns:
            bool: 禁用是否成功
        """
        return True
    
    def on_config_changed(self, config: Dict[str, Any]) -> bool:
        """配置变更时调用
        
        Args:
            config: 新配置
            
        Returns:
            bool: 处理是否成功
        """
        return True


class IPluginRegistry(ABC):
    """插件注册接口"""
    
    @abstractmethod
    def register_hook(self, hook_name: str, callback: Callable) -> bool:
        """注册钩子函数
        
        Args:
            hook_name: 钩子名称
            callback: 回调函数
            
        Returns:
            bool: 注册是否成功
        """
        pass
    
    @abstractmethod
    def unregister_hook(self, hook_name: str, callback: Callable) -> bool:
        """取消注册钩子函数
        
        Args:
            hook_name: 钩子名称
            callback: 回调函数
            
        Returns:
            bool: 取消注册是否成功
        """
        pass
    
    @abstractmethod
    def register_service(self, service_name: str, service: Any) -> bool:
        """注册服务
        
        Args:
            service_name: 服务名称
            service: 服务实例
            
        Returns:
            bool: 注册是否成功
        """
        pass
    
    @abstractmethod
    def get_service(self, service_name: str) -> Optional[Any]:
        """获取服务
        
        Args:
            service_name: 服务名称
            
        Returns:
            Optional[Any]: 服务实例
        """
        pass


class IPluginCommunication(ABC):
    """插件通信接口"""
    
    @abstractmethod
    def send_message(self, target_plugin: str, message: Dict[str, Any]) -> bool:
        """发送消息给其他插件
        
        Args:
            target_plugin: 目标插件名称
            message: 消息内容
            
        Returns:
            bool: 发送是否成功
        """
        pass
    
    @abstractmethod
    def broadcast_message(self, message: Dict[str, Any]) -> bool:
        """广播消息
        
        Args:
            message: 消息内容
            
        Returns:
            bool: 广播是否成功
        """
        pass
    
    @abstractmethod
    def subscribe_event(self, event_type: str, callback: Callable) -> bool:
        """订阅事件
        
        Args:
            event_type: 事件类型
            callback: 回调函数
            
        Returns:
            bool: 订阅是否成功
        """
        pass
    
    @abstractmethod
    def publish_event(self, event_type: str, data: Any) -> bool:
        """发布事件
        
        Args:
            event_type: 事件类型
            data: 事件数据
            
        Returns:
            bool: 发布是否成功
        """
        pass


class IPluginValidator(ABC):
    """插件验证接口"""
    
    @abstractmethod
    def validate_plugin(self, plugin_path: str) -> bool:
        """验证插件
        
        Args:
            plugin_path: 插件路径
            
        Returns:
            bool: 验证是否通过
        """
        pass
    
    @abstractmethod
    def validate_dependencies(self, dependencies: List[PluginDependency]) -> bool:
        """验证依赖
        
        Args:
            dependencies: 依赖列表
            
        Returns:
            bool: 验证是否通过
        """
        pass
    
    @abstractmethod
    def validate_compatibility(self, plugin_name: str, system_info: Dict[str, Any]) -> bool:
        """验证兼容性
        
        Args:
            plugin_name: 插件名称
            system_info: 系统信息
            
        Returns:
            bool: 验证是否通过
        """
        pass


class IPluginMonitor(ABC):
    """插件监控接口"""
    
    @abstractmethod
    def get_plugin_metrics(self, plugin_name: str) -> Dict[str, Any]:
        """获取插件指标
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            Dict[str, Any]: 插件指标
        """
        pass
    
    @abstractmethod
    def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标
        
        Returns:
            Dict[str, Any]: 系统指标
        """
        pass
    
    @abstractmethod
    def start_monitoring(self, plugin_name: str) -> bool:
        """开始监控插件
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            bool: 开始监控是否成功
        """
        pass
    
    @abstractmethod
    def stop_monitoring(self, plugin_name: str) -> bool:
        """停止监控插件
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            bool: 停止监控是否成功
        """
        pass


# 插件开发约定
PLUGIN_CONVENTIONS = {
    "naming": {
        "class_suffix": "Plugin",
        "file_suffix": "_plugin.py",
        "config_suffix": "_config.yaml"
    },
    "structure": {
        "required_methods": ["_create_metadata", "_do_initialize", "_do_cleanup"],
        "optional_methods": ["_pre_initialize", "_post_initialize", "_pre_cleanup", "_post_cleanup"],
        "required_attributes": ["metadata", "status", "config"]
    },
    "metadata": {
        "required_fields": ["name", "version", "author", "description"],
        "optional_fields": ["dependencies", "capabilities", "supported_platforms", "min_memory_mb", "min_cpu_cores", "requires_gpu"]
    },
    "versioning": {
        "format": "semantic",  # major.minor.patch
        "compatibility": "backward"  # 向后兼容
    }
}


# 插件事件类型
PLUGIN_EVENTS = {
    "lifecycle": [
        "plugin.loading",
        "plugin.loaded",
        "plugin.unloading",
        "plugin.unloaded",
        "plugin.enabling",
        "plugin.enabled",
        "plugin.disabling",
        "plugin.disabled",
        "plugin.error"
    ],
    "system": [
        "system.startup",
        "system.shutdown",
        "system.config_changed",
        "system.resource_low",
        "system.error"
    ],
    "data": [
        "data.received",
        "data.processed",
        "data.sent",
        "data.error"
    ]
}


# 插件钩子点
PLUGIN_HOOKS = {
    "detection": [
        "before_detection",
        "after_detection",
        "detection_error"
    ],
    "recognition": [
        "before_recognition",
        "after_recognition",
        "recognition_error"
    ],
    "processing": [
        "before_processing",
        "after_processing",
        "processing_error"
    ],
    "system": [
        "system_init",
        "system_cleanup",
        "config_update"
    ]
}