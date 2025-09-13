"""插件管理器"""

import os
import sys
import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Callable, Type
import threading
import logging
from collections import defaultdict, deque
from dataclasses import asdict
import time

from .base_plugin import BasePlugin, PluginStatus, PluginCapability
from .event_bus import EventBus
from .plugin_loader import PluginLoader, PluginLoadError, PluginValidationError
from .plugin_interface import (
    PluginType, PluginPriority, PluginConfig, IPluginRegistry, 
    IPluginCommunication, IPluginValidator, IPluginMonitor,
    PLUGIN_EVENTS, PLUGIN_HOOKS
)


class PluginDependencyError(Exception):
    """插件依赖错误"""
    pass


class PluginLoadError(Exception):
    """插件加载错误"""
    pass


class PluginManager(IPluginRegistry, IPluginCommunication, IPluginMonitor):
    """插件管理器
    
    负责插件的发现、加载、管理和卸载
    实现插件注册、通信和监控接口
    """
    
    def __init__(self, config: Dict = None, event_bus: EventBus = None, validator: IPluginValidator = None):
        self.config = config or {}
        self.event_bus = event_bus or EventBus()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 插件存储
        self._plugins: Dict[str, BasePlugin] = {}
        self._plugin_classes: Dict[str, Type[BasePlugin]] = {}
        self._plugin_paths: Dict[str, str] = {}
        self.plugin_configs: Dict[str, PluginConfig] = {}
        
        # 依赖关系
        self._dependencies: Dict[str, List[str]] = {}
        self._dependents: Dict[str, List[str]] = defaultdict(list)
        
        # 插件加载器
        self.loader = PluginLoader(
            plugin_directories=self.config.get('plugin_directories', [
                'src/plugins',
                'plugins',
                'src/recognition',  # 兼容现有结构
            ]),
            validator=validator
        )
        
        # 服务注册
        self._services: Dict[str, Any] = {}
        self._hooks: Dict[str, List[Callable]] = {}
        
        # 插件监控
        self._plugin_metrics: Dict[str, Dict[str, Any]] = {}
        self._monitoring_enabled: Dict[str, bool] = {}
        
        # 线程安全
        self._lock = threading.RLock()
        
        # 插件目录
        self.plugin_directories = self.config.get('plugin_directories', [
            'src/plugins',
            'plugins',
            'src/recognition',  # 兼容现有结构
        ])
        
        # 系统信息缓存
        self._system_info = None
        
        # 初始化钩子点
        self._initialize_hooks()
    
    def _initialize_hooks(self):
        """初始化钩子点"""
        for category, hooks in PLUGIN_HOOKS.items():
            for hook in hooks:
                self._hooks[hook] = []
    
    def discover_plugins(self) -> List[str]:
        """发现可用插件
        
        Returns:
            List[str]: 发现的插件名称列表
        """
        discovered = []
        
        try:
            # 使用新的插件加载器发现插件
            plugins_info = self.loader.discover_plugins()
            
            for plugin_name, plugin_info in plugins_info.items():
                discovered.append(plugin_name)
                
                # 存储插件类和路径
                self._plugin_classes[plugin_name] = plugin_info['class']
                self._plugin_paths[plugin_name] = plugin_info['path']
                
                # 加载插件配置
                config_path = plugin_info.get('config_path')
                if config_path:
                    try:
                        config_data = self.loader._load_plugin_config(config_path)
                        self.plugin_configs[plugin_name] = PluginConfig(**config_data)
                    except Exception as e:
                        self.logger.warning(f"Failed to load config for {plugin_name}: {e}")
                        self.plugin_configs[plugin_name] = PluginConfig()
                else:
                    self.plugin_configs[plugin_name] = PluginConfig()
                
                self.logger.debug(f"Discovered plugin: {plugin_name}")
            
            self.logger.info(f"Discovered {len(discovered)} plugins")
            
        except Exception as e:
            self.logger.error(f"Error discovering plugins: {e}")
            # 回退到原有方法
            for plugin_dir in self.plugin_directories:
                plugin_path = Path(plugin_dir)
                if not plugin_path.exists():
                    continue
                
                self.logger.info(f"Scanning plugin directory: {plugin_path}")
                
                # 扫描Python文件
                for py_file in plugin_path.rglob('*.py'):
                    if py_file.name.startswith('_'):
                        continue
                    
                    try:
                        plugin_name = self._load_plugin_class(py_file)
                        if plugin_name:
                            discovered.append(plugin_name)
                            self.logger.debug(f"Discovered plugin: {plugin_name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to load plugin from {py_file}: {e}")
        
        return discovered
    
    def _load_plugin_class(self, plugin_file: Path) -> Optional[str]:
        """从文件加载插件类
        
        Args:
            plugin_file: 插件文件路径
            
        Returns:
            Optional[str]: 插件名称，如果加载失败返回None
        """
        try:
            # 构建模块名
            relative_path = plugin_file.relative_to(Path.cwd())
            module_name = str(relative_path.with_suffix('')).replace(os.sep, '.')
            
            # 动态导入模块
            spec = importlib.util.spec_from_file_location(module_name, plugin_file)
            if spec is None or spec.loader is None:
                return None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 查找插件类
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, BasePlugin) and 
                    attr != BasePlugin):
                    
                    plugin_name = attr_name.lower().replace('plugin', '')
                    self._plugin_classes[plugin_name] = attr
                    self._plugin_paths[plugin_name] = str(plugin_file)
                    return plugin_name
        
        except Exception as e:
            self.logger.debug(f"Error loading plugin class from {plugin_file}: {e}")
        
        return None
    
    def load_plugin(self, plugin_name: str, config: Dict = None) -> bool:
        """加载插件
        
        Args:
            plugin_name: 插件名称
            config: 插件配置
            
        Returns:
            bool: 加载是否成功
        """
        with self._lock:
            if plugin_name in self._plugins:
                self.logger.warning(f"Plugin {plugin_name} already loaded")
                return True
            
            if plugin_name not in self._plugin_classes:
                self.logger.error(f"Plugin class {plugin_name} not found")
                return False
            
            try:
                # 检查依赖
                if not self._check_dependencies(plugin_name):
                    return False
                
                # 创建插件实例
                plugin_class = self._plugin_classes[plugin_name]
                plugin_config = config or self.config.get('plugins', {}).get(plugin_name, {})
                plugin = plugin_class(plugin_config)
                
                # 验证系统要求
                if not plugin.validate_requirements(self._get_system_info()):
                    self.logger.error(f"System requirements not met for plugin {plugin_name}")
                    return False
                
                # 设置状态
                plugin.set_status(PluginStatus.LOADING)
                
                # 初始化插件
                if not plugin.initialize():
                    self.logger.error(f"Failed to initialize plugin {plugin_name}")
                    plugin.set_status(PluginStatus.ERROR)
                    return False
                
                # 注册插件
                self._plugins[plugin_name] = plugin
                plugin.set_status(PluginStatus.LOADED)
                
                # 更新依赖关系
                self._update_dependencies(plugin_name, plugin.metadata.dependencies)
                
                # 发送事件
                if self.event_bus:
                    self.event_bus.emit('plugin_loaded', {
                        'plugin_name': plugin_name,
                        'metadata': plugin.metadata.to_dict()
                    })
                
                self.logger.info(f"Plugin {plugin_name} loaded successfully")
                return True
            
            except Exception as e:
                self.logger.error(f"Failed to load plugin {plugin_name}: {e}")
                return False
    
    def unload_plugin(self, plugin_name: str, force: bool = False) -> bool:
        """卸载插件
        
        Args:
            plugin_name: 插件名称
            force: 是否强制卸载（忽略依赖）
            
        Returns:
            bool: 卸载是否成功
        """
        with self._lock:
            if plugin_name not in self._plugins:
                self.logger.warning(f"Plugin {plugin_name} not loaded")
                return True
            
            # 检查是否有其他插件依赖此插件
            if not force and plugin_name in self._dependents:
                dependents = self._dependents[plugin_name]
                loaded_dependents = [dep for dep in dependents if dep in self._plugins]
                if loaded_dependents:
                    self.logger.error(f"Cannot unload plugin {plugin_name}, it has dependents: {loaded_dependents}")
                    return False
            
            try:
                plugin = self._plugins[plugin_name]
                
                # 清理资源
                plugin.cleanup()
                plugin.set_status(PluginStatus.UNLOADED)
                
                # 移除插件
                del self._plugins[plugin_name]
                
                # 更新依赖关系
                self._remove_dependencies(plugin_name)
                
                # 发送事件
                if self.event_bus:
                    self.event_bus.emit('plugin_unloaded', {
                        'plugin_name': plugin_name
                    })
                
                self.logger.info(f"Plugin {plugin_name} unloaded successfully")
                return True
            
            except Exception as e:
                self.logger.error(f"Failed to unload plugin {plugin_name}: {e}")
                return False
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """获取插件实例
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            Optional[BasePlugin]: 插件实例
        """
        return self._plugins.get(plugin_name)
    
    def get_plugins_by_capability(self, capability: PluginCapability) -> List[BasePlugin]:
        """根据能力获取插件列表
        
        Args:
            capability: 插件能力
            
        Returns:
            List[BasePlugin]: 具有指定能力的插件列表
        """
        return [plugin for plugin in self._plugins.values() 
                if plugin.has_capability(capability)]
    
    def list_plugins(self) -> Dict[str, Dict]:
        """列出所有插件信息
        
        Returns:
            Dict[str, Dict]: 插件信息字典
        """
        result = {}
        
        # 已加载的插件
        for name, plugin in self._plugins.items():
            result[name] = {
                'status': plugin.get_status().value,
                'metadata': plugin.metadata.to_dict(),
                'loaded': True
            }
        
        # 未加载的插件
        for name, plugin_class in self._plugin_classes.items():
            if name not in result:
                try:
                    temp_plugin = plugin_class({})
                    result[name] = {
                        'status': 'available',
                        'metadata': temp_plugin.metadata.to_dict(),
                        'loaded': False
                    }
                except Exception as e:
                    result[name] = {
                        'status': 'error',
                        'error': str(e),
                        'loaded': False
                    }
        
        return result
    
    def reload_plugin(self, plugin_name: str) -> bool:
        """重新加载插件
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            bool: 重新加载是否成功
        """
        config = None
        if plugin_name in self._plugins:
            config = self._plugins[plugin_name].config
            if not self.unload_plugin(plugin_name):
                return False
        
        return self.load_plugin(plugin_name, config)
    
    def _check_dependencies(self, plugin_name: str) -> bool:
        """检查插件依赖
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            bool: 依赖是否满足
        """
        if plugin_name not in self._plugin_classes:
            return False
        
        try:
            temp_plugin = self._plugin_classes[plugin_name]({})
            dependencies = temp_plugin.metadata.dependencies
            
            for dep in dependencies:
                if dep not in self._plugins:
                    self.logger.error(f"Plugin {plugin_name} requires {dep} but it's not loaded")
                    return False
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error checking dependencies for {plugin_name}: {e}")
            return False
    
    def _update_dependencies(self, plugin_name: str, dependencies: List[str]) -> None:
        """更新依赖关系
        
        Args:
            plugin_name: 插件名称
            dependencies: 依赖列表
        """
        self._dependencies[plugin_name] = dependencies
        
        for dep in dependencies:
            self._dependents[dep].append(plugin_name)
    
    def _remove_dependencies(self, plugin_name: str) -> None:
        """移除依赖关系
        
        Args:
            plugin_name: 插件名称
        """
        # 移除此插件的依赖记录
        if plugin_name in self._dependencies:
            dependencies = self._dependencies[plugin_name]
            for dep in dependencies:
                if dep in self._dependents:
                    self._dependents[dep] = [p for p in self._dependents[dep] if p != plugin_name]
            del self._dependencies[plugin_name]
        
        # 移除此插件作为其他插件依赖的记录
        if plugin_name in self._dependents:
            del self._dependents[plugin_name]
    
    def _get_system_info(self) -> Dict:
        """获取系统信息
        
        Returns:
            Dict: 系统信息
        """
        if self._system_info is None:
            import psutil
            import platform
            
            self._system_info = {
                'platform': platform.system().lower(),
                'memory_mb': psutil.virtual_memory().total // (1024 * 1024),
                'cpu_cores': psutil.cpu_count(),
                'has_gpu': self._detect_gpu()
            }
        
        return self._system_info
    
    def _detect_gpu(self) -> bool:
        """检测GPU可用性
        
        Returns:
            bool: 是否有可用GPU
        """
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            pass
        
        try:
            import tensorflow as tf
            return len(tf.config.list_physical_devices('GPU')) > 0
        except ImportError:
            pass
        
        return False
    
    def load_enabled_plugins(self) -> None:
        """加载配置中启用的插件"""
        enabled_plugins = self.config.get('plugins', {}).get('enabled', [])
        
        if not enabled_plugins:
            self.logger.info("No plugins enabled in configuration")
            return
        
        # 首先发现所有插件
        self.discover_plugins()
        
        # 按依赖顺序加载插件
        loaded = set()
        remaining = set(enabled_plugins)
        
        while remaining:
            progress = False
            
            for plugin_name in list(remaining):
                if plugin_name not in self._plugin_classes:
                    self.logger.warning(f"Enabled plugin {plugin_name} not found")
                    remaining.remove(plugin_name)
                    continue
                
                # 检查依赖是否已加载
                try:
                    temp_plugin = self._plugin_classes[plugin_name]({})
                    dependencies = temp_plugin.metadata.dependencies
                    
                    if all(dep in loaded for dep in dependencies):
                        if self.load_plugin(plugin_name):
                            loaded.add(plugin_name)
                            remaining.remove(plugin_name)
                            progress = True
                        else:
                            self.logger.error(f"Failed to load enabled plugin {plugin_name}")
                            remaining.remove(plugin_name)
                
                except Exception as e:
                    self.logger.error(f"Error loading plugin {plugin_name}: {e}")
                    remaining.remove(plugin_name)
            
            if not progress and remaining:
                self.logger.error(f"Circular dependency or missing dependencies for plugins: {remaining}")
                break
        
        self.logger.info(f"Loaded {len(loaded)} plugins: {list(loaded)}")
    
    # IPluginRegistry 接口实现
    def register_service(self, name: str, service: Any) -> bool:
        """注册服务"""
        with self._lock:
            if name in self._services:
                self.logger.warning(f"Service {name} already registered")
                return False
            self._services[name] = service
            return True
    
    def unregister_service(self, name: str) -> bool:
        """注销服务"""
        with self._lock:
            if name in self._services:
                del self._services[name]
                return True
            return False
    
    def get_service(self, name: str) -> Optional[Any]:
        """获取服务"""
        return self._services.get(name)
    
    def list_services(self) -> List[str]:
        """列出所有服务"""
        return list(self._services.keys())
    
    # IPluginCommunication 接口实现
    def register_hook(self, hook_name: str, callback: Callable) -> bool:
        """注册钩子"""
        with self._lock:
            if hook_name not in self._hooks:
                self._hooks[hook_name] = []
            self._hooks[hook_name].append(callback)
            return True
    
    def unregister_hook(self, hook_name: str, callback: Callable) -> bool:
        """注销钩子"""
        with self._lock:
            if hook_name in self._hooks:
                try:
                    self._hooks[hook_name].remove(callback)
                    return True
                except ValueError:
                    pass
            return False
    
    def execute_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """执行钩子"""
        results = []
        if hook_name in self._hooks:
            for callback in self._hooks[hook_name]:
                try:
                    result = callback(*args, **kwargs)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error executing hook {hook_name}: {e}")
        return results
    
    def send_message(self, target_plugin: str, message: Dict[str, Any]) -> bool:
        """发送消息给插件"""
        plugin = self.get_plugin(target_plugin)
        if plugin and hasattr(plugin, 'receive_message'):
            try:
                plugin.receive_message(message)
                return True
            except Exception as e:
                self.logger.error(f"Error sending message to {target_plugin}: {e}")
        return False
    
    def broadcast_message(self, message: Dict[str, Any], plugin_filter: Optional[Callable] = None) -> int:
        """广播消息"""
        count = 0
        for plugin_name, plugin in self._plugins.items():
            if plugin_filter and not plugin_filter(plugin):
                continue
            if hasattr(plugin, 'receive_message'):
                try:
                    plugin.receive_message(message)
                    count += 1
                except Exception as e:
                    self.logger.error(f"Error broadcasting message to {plugin_name}: {e}")
        return count
    
    # IPluginMonitor 接口实现
    def start_monitoring(self, plugin_name: str) -> bool:
        """开始监控插件"""
        with self._lock:
            if plugin_name in self._plugins:
                self._monitoring_enabled[plugin_name] = True
                self._plugin_metrics[plugin_name] = {
                    'start_time': time.time(),
                    'call_count': 0,
                    'error_count': 0,
                    'last_activity': time.time()
                }
                return True
            return False
    
    def stop_monitoring(self, plugin_name: str) -> bool:
        """停止监控插件"""
        with self._lock:
            if plugin_name in self._monitoring_enabled:
                self._monitoring_enabled[plugin_name] = False
                return True
            return False
    
    def get_plugin_metrics(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """获取插件指标"""
        return self._plugin_metrics.get(plugin_name)
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """获取所有插件指标"""
        return self._plugin_metrics.copy()
    
    def is_plugin_healthy(self, plugin_name: str) -> bool:
        """检查插件健康状态"""
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            return False
        
        # 检查插件状态
        if plugin.get_status() != PluginStatus.LOADED:
            return False
        
        # 检查最近活动
        metrics = self.get_plugin_metrics(plugin_name)
        if metrics:
            last_activity = metrics.get('last_activity', 0)
            if time.time() - last_activity > 300:  # 5分钟无活动
                return False
        
        return True
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        import psutil
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent if hasattr(psutil.disk_usage('/'), 'percent') else 0,
            'plugin_count': len(self._plugins),
            'active_plugins': len([p for p in self._plugins.values() if p.get_status() == PluginStatus.LOADED])
        }
    
    def publish_event(self, event_type: str, data: Any = None) -> bool:
        """发布事件"""
        try:
            self.event_bus.emit(event_type, data)
            return True
        except Exception as e:
            self.logger.error(f"Failed to publish event {event_type}: {e}")
            return False
    
    def subscribe_event(self, event_type: str, handler: Callable) -> str:
        """订阅事件"""
        try:
            return self.event_bus.on(event_type, handler)
        except Exception as e:
            self.logger.error(f"Failed to subscribe to event {event_type}: {e}")
            return ""
    
    def shutdown(self) -> None:
        """关闭插件管理器"""
        self.logger.info("Shutting down plugin manager")
        
        # 停止所有监控
        for plugin_name in list(self._monitoring_enabled.keys()):
            self.stop_monitoring(plugin_name)
        
        # 卸载所有插件
        plugin_names = list(self._plugins.keys())
        for plugin_name in plugin_names:
            self.unload_plugin(plugin_name, force=True)
        
        # 清理服务和钩子
        self._services.clear()
        self._hooks.clear()
        
        self.logger.info("Plugin manager shutdown complete")