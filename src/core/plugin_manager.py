"""插件管理器"""

import os
import sys
import importlib
import importlib.util
from typing import Dict, List, Optional, Type, Any
import logging
from pathlib import Path
import threading
from collections import defaultdict, deque

from .base_plugin import BasePlugin, PluginStatus, PluginCapability
from .event_bus import EventBus


class PluginDependencyError(Exception):
    """插件依赖错误"""
    pass


class PluginLoadError(Exception):
    """插件加载错误"""
    pass


class PluginManager:
    """插件管理器
    
    负责插件的发现、加载、管理和卸载
    """
    
    def __init__(self, config: Dict = None, event_bus: EventBus = None):
        self.config = config or {}
        self.event_bus = event_bus
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 插件存储
        self._plugins: Dict[str, BasePlugin] = {}
        self._plugin_classes: Dict[str, Type[BasePlugin]] = {}
        self._plugin_paths: Dict[str, str] = {}
        
        # 依赖关系
        self._dependencies: Dict[str, List[str]] = {}
        self._dependents: Dict[str, List[str]] = defaultdict(list)
        
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
    
    def discover_plugins(self) -> List[str]:
        """发现可用插件
        
        Returns:
            List[str]: 发现的插件名称列表
        """
        discovered = []
        
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
        
        self.logger.info(f"Discovered {len(discovered)} plugins")
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
    
    def shutdown(self) -> None:
        """关闭插件管理器"""
        self.logger.info("Shutting down plugin manager")
        
        # 卸载所有插件
        plugin_names = list(self._plugins.keys())
        for plugin_name in plugin_names:
            self.unload_plugin(plugin_name, force=True)
        
        self.logger.info("Plugin manager shutdown complete")