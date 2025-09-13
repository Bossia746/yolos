"""插件加载器

负责插件的发现、加载、验证和管理
"""

import os
import sys
import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Type, Any
import logging
import json
import yaml
from dataclasses import asdict

from .base_plugin import BasePlugin, PluginStatus, PluginMetadata
from .plugin_interface import (
    PluginType, PluginPriority, PluginDependency, PluginConfig,
    IPluginValidator, PLUGIN_CONVENTIONS
)


class PluginLoadError(Exception):
    """插件加载错误"""
    pass


class PluginValidationError(Exception):
    """插件验证错误"""
    pass


class PluginLoader:
    """插件加载器
    
    负责插件的发现、加载、验证和实例化
    """
    
    def __init__(self, plugin_directories: List[str] = None, validator: IPluginValidator = None):
        self.logger = logging.getLogger(__name__)
        self.plugin_directories = plugin_directories or []
        self.validator = validator
        self._loaded_modules = {}
        self._plugin_classes = {}
        self._plugin_configs = {}
        
    def add_plugin_directory(self, directory: str):
        """添加插件目录
        
        Args:
            directory: 插件目录路径
        """
        if os.path.exists(directory) and directory not in self.plugin_directories:
            self.plugin_directories.append(directory)
            self.logger.info(f"Added plugin directory: {directory}")
    
    def discover_plugins(self) -> Dict[str, Dict[str, Any]]:
        """发现所有插件
        
        Returns:
            Dict[str, Dict[str, Any]]: 插件信息字典
        """
        plugins = {}
        
        for directory in self.plugin_directories:
            self.logger.info(f"Discovering plugins in: {directory}")
            plugins.update(self._discover_plugins_in_directory(directory))
        
        self.logger.info(f"Discovered {len(plugins)} plugins")
        return plugins
    
    def _discover_plugins_in_directory(self, directory: str) -> Dict[str, Dict[str, Any]]:
        """在指定目录中发现插件
        
        Args:
            directory: 目录路径
            
        Returns:
            Dict[str, Dict[str, Any]]: 插件信息字典
        """
        plugins = {}
        
        try:
            for root, dirs, files in os.walk(directory):
                # 跳过__pycache__目录
                dirs[:] = [d for d in dirs if d != '__pycache__']
                
                for file in files:
                    if file.endswith('.py') and not file.startswith('__'):
                        file_path = os.path.join(root, file)
                        plugin_info = self._analyze_plugin_file(file_path)
                        if plugin_info:
                            plugins[plugin_info['name']] = plugin_info
                            
        except Exception as e:
            self.logger.error(f"Error discovering plugins in {directory}: {e}")
        
        return plugins
    
    def _analyze_plugin_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """分析插件文件
        
        Args:
            file_path: 插件文件路径
            
        Returns:
            Optional[Dict[str, Any]]: 插件信息
        """
        try:
            # 加载模块
            module_name = Path(file_path).stem
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if not spec or not spec.loader:
                return None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 查找插件类
            plugin_class = self._find_plugin_class(module)
            if not plugin_class:
                return None
            
            # 验证插件类
            if not self._validate_plugin_class(plugin_class):
                return None
            
            # 创建临时实例获取元数据
            try:
                temp_instance = plugin_class()
                metadata = temp_instance._create_metadata()
                
                plugin_info = {
                    'name': metadata.name,
                    'file_path': file_path,
                    'module_name': module_name,
                    'class_name': plugin_class.__name__,
                    'metadata': asdict(metadata),
                    'config_path': self._find_config_file(file_path, metadata.name)
                }
                
                return plugin_info
                
            except Exception as e:
                self.logger.warning(f"Failed to get metadata from {file_path}: {e}")
                return None
                
        except Exception as e:
            self.logger.debug(f"Failed to analyze plugin file {file_path}: {e}")
            return None
    
    def _find_plugin_class(self, module) -> Optional[Type[BasePlugin]]:
        """在模块中查找插件类
        
        Args:
            module: Python模块
            
        Returns:
            Optional[Type[BasePlugin]]: 插件类
        """
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (issubclass(obj, BasePlugin) and 
                obj != BasePlugin and 
                obj.__module__ == module.__name__):
                return obj
        return None
    
    def _validate_plugin_class(self, plugin_class: Type[BasePlugin]) -> bool:
        """验证插件类
        
        Args:
            plugin_class: 插件类
            
        Returns:
            bool: 验证是否通过
        """
        try:
            # 检查必需方法
            required_methods = PLUGIN_CONVENTIONS['structure']['required_methods']
            for method_name in required_methods:
                if not hasattr(plugin_class, method_name):
                    self.logger.error(f"Plugin {plugin_class.__name__} missing required method: {method_name}")
                    return False
            
            # 检查命名约定
            if not plugin_class.__name__.endswith(PLUGIN_CONVENTIONS['naming']['class_suffix']):
                self.logger.warning(f"Plugin {plugin_class.__name__} doesn't follow naming convention")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating plugin class {plugin_class.__name__}: {e}")
            return False
    
    def _find_config_file(self, plugin_file: str, plugin_name: str) -> Optional[str]:
        """查找插件配置文件
        
        Args:
            plugin_file: 插件文件路径
            plugin_name: 插件名称
            
        Returns:
            Optional[str]: 配置文件路径
        """
        plugin_dir = os.path.dirname(plugin_file)
        config_name = f"{plugin_name}_config"
        
        # 查找配置文件
        for ext in ['.yaml', '.yml', '.json']:
            config_path = os.path.join(plugin_dir, f"{config_name}{ext}")
            if os.path.exists(config_path):
                return config_path
        
        return None
    
    def load_plugin(self, plugin_info: Dict[str, Any]) -> Optional[BasePlugin]:
        """加载插件
        
        Args:
            plugin_info: 插件信息
            
        Returns:
            Optional[BasePlugin]: 插件实例
        """
        try:
            plugin_name = plugin_info['name']
            
            # 验证插件
            if self.validator and not self.validator.validate_plugin(plugin_info['file_path']):
                raise PluginValidationError(f"Plugin validation failed: {plugin_name}")
            
            # 加载配置
            config = self._load_plugin_config(plugin_info.get('config_path'))
            
            # 加载模块
            module = self._load_plugin_module(plugin_info)
            
            # 获取插件类
            plugin_class = getattr(module, plugin_info['class_name'])
            
            # 创建插件实例
            plugin_instance = plugin_class(config)
            
            self.logger.info(f"Successfully loaded plugin: {plugin_name}")
            return plugin_instance
            
        except Exception as e:
            self.logger.error(f"Failed to load plugin {plugin_info.get('name', 'unknown')}: {e}")
            raise PluginLoadError(f"Failed to load plugin: {e}")
    
    def _load_plugin_module(self, plugin_info: Dict[str, Any]):
        """加载插件模块
        
        Args:
            plugin_info: 插件信息
            
        Returns:
            module: Python模块
        """
        module_name = plugin_info['module_name']
        file_path = plugin_info['file_path']
        
        # 检查是否已加载
        if module_name in self._loaded_modules:
            return self._loaded_modules[module_name]
        
        # 加载模块
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if not spec or not spec.loader:
            raise PluginLoadError(f"Cannot create module spec for {file_path}")
        
        module = importlib.util.module_from_spec(spec)
        
        # 添加到sys.modules以支持相对导入
        sys.modules[module_name] = module
        
        try:
            spec.loader.exec_module(module)
            self._loaded_modules[module_name] = module
            return module
        except Exception as e:
            # 清理失败的模块
            if module_name in sys.modules:
                del sys.modules[module_name]
            raise PluginLoadError(f"Failed to execute module {module_name}: {e}")
    
    def _load_plugin_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载插件配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            Dict[str, Any]: 配置字典
        """
        if not config_path or not os.path.exists(config_path):
            return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.endswith('.json'):
                    return json.load(f)
                elif config_path.endswith(('.yaml', '.yml')):
                    return yaml.safe_load(f) or {}
                else:
                    self.logger.warning(f"Unsupported config file format: {config_path}")
                    return {}
                    
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
            return {}
    
    def unload_plugin(self, plugin_name: str):
        """卸载插件
        
        Args:
            plugin_name: 插件名称
        """
        # 从已加载模块中移除
        modules_to_remove = []
        for module_name, module in self._loaded_modules.items():
            if hasattr(module, '__file__') and plugin_name in module.__file__:
                modules_to_remove.append(module_name)
        
        for module_name in modules_to_remove:
            del self._loaded_modules[module_name]
            if module_name in sys.modules:
                del sys.modules[module_name]
        
        self.logger.info(f"Unloaded plugin: {plugin_name}")
    
    def reload_plugin(self, plugin_info: Dict[str, Any]) -> Optional[BasePlugin]:
        """重新加载插件
        
        Args:
            plugin_info: 插件信息
            
        Returns:
            Optional[BasePlugin]: 插件实例
        """
        plugin_name = plugin_info['name']
        
        # 先卸载
        self.unload_plugin(plugin_name)
        
        # 重新加载
        return self.load_plugin(plugin_info)
    
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """获取插件信息
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            Optional[Dict[str, Any]]: 插件信息
        """
        plugins = self.discover_plugins()
        return plugins.get(plugin_name)
    
    def list_available_plugins(self) -> List[str]:
        """列出可用插件
        
        Returns:
            List[str]: 插件名称列表
        """
        plugins = self.discover_plugins()
        return list(plugins.keys())