#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOS 模块管理器
统一的模块加载、依赖管理和生命周期控制

作者: YOLOS团队
版本: 1.0.0
创建时间: 2025-09-11
"""

import os
import sys
import json
import importlib
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Type, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .exceptions import YOLOSException, ModuleError
from .performance_monitor import global_monitor


class ModuleStatus(Enum):
    """模块状态枚举"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    UNLOADING = "unloading"


class ModulePriority(Enum):
    """模块优先级"""
    CRITICAL = 0  # 核心模块
    HIGH = 1      # 高优先级
    NORMAL = 2    # 普通优先级
    LOW = 3       # 低优先级


@dataclass
class ModuleInfo:
    """模块信息"""
    name: str
    version: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    optional_dependencies: List[str] = field(default_factory=list)
    priority: ModulePriority = ModulePriority.NORMAL
    auto_load: bool = True
    config_schema: Optional[Dict[str, Any]] = None
    entry_point: Optional[str] = None
    module_path: Optional[str] = None


@dataclass
class ModuleState:
    """模块状态信息"""
    info: ModuleInfo
    status: ModuleStatus = ModuleStatus.UNLOADED
    instance: Optional[Any] = None
    load_time: Optional[float] = None
    error_message: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class BaseModule(ABC):
    """模块基类"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self._initialized = False
        self._lock = threading.RLock()
    
    @abstractmethod
    def initialize(self) -> bool:
        """初始化模块"""
        pass
    
    @abstractmethod
    def cleanup(self) -> bool:
        """清理模块资源"""
        pass
    
    def get_info(self) -> ModuleInfo:
        """获取模块信息"""
        return ModuleInfo(
            name=self.name,
            version="1.0.0",
            description=f"{self.name} module",
            dependencies=[],
            priority=ModulePriority.NORMAL
        )
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """验证配置"""
        return True
    
    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized


class ModuleManager:
    """模块管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.modules: Dict[str, ModuleState] = {}
        self.load_order: List[str] = []
        self._lock = threading.RLock()
        self._hooks: Dict[str, List[Callable]] = {
            'before_load': [],
            'after_load': [],
            'before_unload': [],
            'after_unload': []
        }
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 模块搜索路径
        self.module_paths = self.config.get('module_paths', [
            'src/modules',
            'src/applications',
            'src/optimized'
        ])
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载配置文件"""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                raise ModuleError(f"Failed to load config: {e}")
        
        return {
            'auto_discovery': True,
            'parallel_loading': True,
            'dependency_resolution': True,
            'module_paths': [
                'src/modules',
                'src/applications', 
                'src/optimized'
            ]
        }
    
    def discover_modules(self) -> List[ModuleInfo]:
        """自动发现模块"""
        discovered = []
        
        for module_path in self.module_paths:
            if not os.path.exists(module_path):
                continue
                
            for root, dirs, files in os.walk(module_path):
                for file in files:
                    if file.endswith('.py') and not file.startswith('_'):
                        module_file = os.path.join(root, file)
                        try:
                            info = self._extract_module_info(module_file)
                            if info:
                                discovered.append(info)
                        except Exception as e:
                            print(f"Warning: Failed to analyze {module_file}: {e}")
        
        return discovered
    
    def _extract_module_info(self, module_file: str) -> Optional[ModuleInfo]:
        """从模块文件提取信息"""
        try:
            # 简化的模块信息提取
            module_name = os.path.splitext(os.path.basename(module_file))[0]
            
            # 检查是否有模块清单文件
            manifest_file = os.path.join(os.path.dirname(module_file), f"{module_name}.json")
            if os.path.exists(manifest_file):
                with open(manifest_file, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
                    return ModuleInfo(**manifest)
            
            # 默认模块信息
            return ModuleInfo(
                name=module_name,
                version="1.0.0",
                description=f"Auto-discovered module: {module_name}",
                module_path=module_file
            )
            
        except Exception:
            return None
    
    def register_module(self, info: ModuleInfo) -> bool:
        """注册模块"""
        with self._lock:
            if info.name in self.modules:
                raise ModuleError(f"Module {info.name} already registered")
            
            self.modules[info.name] = ModuleState(info=info)
            return True
    
    def load_module(self, name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """加载模块"""
        with self._lock:
            if name not in self.modules:
                raise ModuleError(f"Module {name} not registered")
            
            state = self.modules[name]
            if state.status == ModuleStatus.READY:
                return True
            
            try:
                # 执行加载前钩子
                self._execute_hooks('before_load', name)
                
                state.status = ModuleStatus.LOADING
                
                # 检查依赖
                if not self._check_dependencies(state.info):
                    raise ModuleError(f"Dependencies not satisfied for {name}")
                
                # 加载模块
                instance = self._load_module_instance(state.info, config)
                
                state.status = ModuleStatus.INITIALIZING
                
                # 初始化模块
                if hasattr(instance, 'initialize'):
                    if not instance.initialize():
                        raise ModuleError(f"Failed to initialize module {name}")
                
                state.instance = instance
                state.status = ModuleStatus.READY
                state.config = config
                
                # 执行加载后钩子
                self._execute_hooks('after_load', name)
                
                return True
                
            except Exception as e:
                state.status = ModuleStatus.ERROR
                state.error_message = str(e)
                raise ModuleError(f"Failed to load module {name}: {e}")
    
    def _load_module_instance(self, info: ModuleInfo, config: Optional[Dict[str, Any]]) -> Any:
        """加载模块实例"""
        if info.module_path:
            # 从文件路径加载
            spec = importlib.util.spec_from_file_location(info.name, info.module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            # 从包路径加载
            module = importlib.import_module(info.name)
        
        # 查找模块类
        if info.entry_point:
            cls = getattr(module, info.entry_point)
        else:
            # 查找继承自BaseModule的类
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, BaseModule) and 
                    attr != BaseModule):
                    cls = attr
                    break
            else:
                raise ModuleError(f"No valid module class found in {info.name}")
        
        return cls(info.name, config)
    
    def _check_dependencies(self, info: ModuleInfo) -> bool:
        """检查模块依赖"""
        for dep in info.dependencies:
            if dep not in self.modules:
                return False
            if self.modules[dep].status != ModuleStatus.READY:
                # 尝试加载依赖
                if not self.load_module(dep):
                    return False
        return True
    
    def unload_module(self, name: str) -> bool:
        """卸载模块"""
        with self._lock:
            if name not in self.modules:
                return True
            
            state = self.modules[name]
            if state.status == ModuleStatus.UNLOADED:
                return True
            
            try:
                # 执行卸载前钩子
                self._execute_hooks('before_unload', name)
                
                state.status = ModuleStatus.UNLOADING
                
                # 清理模块
                if state.instance and hasattr(state.instance, 'cleanup'):
                    state.instance.cleanup()
                
                state.instance = None
                state.status = ModuleStatus.UNLOADED
                
                # 执行卸载后钩子
                self._execute_hooks('after_unload', name)
                
                return True
                
            except Exception as e:
                state.status = ModuleStatus.ERROR
                state.error_message = str(e)
                return False
    
    def get_module(self, name: str) -> Optional[Any]:
        """获取模块实例"""
        with self._lock:
            if name in self.modules:
                state = self.modules[name]
                if state.status == ModuleStatus.READY:
                    return state.instance
        return None
    
    def get_module_status(self, name: str) -> Optional[ModuleStatus]:
        """获取模块状态"""
        with self._lock:
            if name in self.modules:
                return self.modules[name].status
        return None
    
    def list_modules(self) -> Dict[str, ModuleStatus]:
        """列出所有模块及其状态"""
        with self._lock:
            return {name: state.status for name, state in self.modules.items()}
    
    def add_hook(self, event: str, callback: Callable) -> bool:
        """添加钩子函数"""
        if event in self._hooks:
            self._hooks[event].append(callback)
            return True
        return False
    
    def _execute_hooks(self, event: str, module_name: str):
        """执行钩子函数"""
        for callback in self._hooks.get(event, []):
            try:
                callback(module_name)
            except Exception as e:
                print(f"Hook execution failed for {event}: {e}")
    
    def load_all_modules(self) -> Dict[str, bool]:
        """加载所有已注册的模块"""
        results = {}
        
        # 按优先级排序
        sorted_modules = sorted(
            self.modules.items(),
            key=lambda x: x[1].info.priority.value
        )
        
        for name, state in sorted_modules:
            if state.info.auto_load:
                try:
                    results[name] = self.load_module(name)
                except Exception as e:
                    results[name] = False
                    print(f"Failed to load module {name}: {e}")
        
        return results
    
    def shutdown(self):
        """关闭模块管理器"""
        # 按相反顺序卸载模块
        for name in reversed(list(self.modules.keys())):
            try:
                self.unload_module(name)
            except Exception as e:
                print(f"Error unloading module {name}: {e}")


# 全局模块管理器实例
global_module_manager = ModuleManager()


def get_module_manager() -> ModuleManager:
    """获取全局模块管理器"""
    return global_module_manager


def register_module(info: ModuleInfo) -> bool:
    """注册模块的便捷函数"""
    return global_module_manager.register_module(info)


def load_module(name: str, config: Optional[Dict[str, Any]] = None) -> bool:
    """加载模块的便捷函数"""
    return global_module_manager.load_module(name, config)


def get_module(name: str) -> Optional[Any]:
    """获取模块的便捷函数"""
    return global_module_manager.get_module(name)


if __name__ == "__main__":
    # 测试代码
    manager = ModuleManager()
    
    # 自动发现模块
    discovered = manager.discover_modules()
    print(f"Discovered {len(discovered)} modules")
    
    # 注册发现的模块
    for info in discovered:
        try:
            manager.register_module(info)
            print(f"Registered module: {info.name}")
        except Exception as e:
            print(f"Failed to register {info.name}: {e}")
    
    # 列出所有模块
    modules = manager.list_modules()
    print(f"\nRegistered modules: {list(modules.keys())}")