#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOS组件注册系统
借鉴MMDetection的Registry设计，提供灵活的组件管理
"""

import inspect
from typing import Dict, Any, Type, Optional, Callable, Union
from functools import wraps


class Registry:
    """组件注册器"""
    
    def __init__(self, name: str):
        self.name = name
        self._module_dict: Dict[str, Type] = {}
        self._children: Dict[str, 'Registry'] = {}
    
    def register_module(self, 
                       name: Optional[str] = None, 
                       force: bool = False,
                       module: Optional[Type] = None) -> Union[Type, Callable]:
        """注册模块
        
        Args:
            name: 模块名称，如果为None则使用类名
            force: 是否强制覆盖已存在的模块
            module: 要注册的模块类
            
        Returns:
            注册的模块类或装饰器函数
        """
        if not isinstance(force, bool):
            raise TypeError(f'force must be a boolean, but got {type(force)}')
        
        # 作为装饰器使用
        if module is None:
            return self._register_decorator(name, force)
        
        # 直接注册
        self._register_module(module, name, force)
        return module
    
    def _register_decorator(self, name: Optional[str], force: bool) -> Callable:
        """注册装饰器"""
        def decorator(cls: Type) -> Type:
            self._register_module(cls, name, force)
            return cls
        return decorator
    
    def _register_module(self, module: Type, name: Optional[str], force: bool):
        """注册模块实现"""
        if not inspect.isclass(module):
            raise TypeError(f'module must be a class, but got {type(module)}')
        
        if name is None:
            name = module.__name__
        
        if not force and name in self._module_dict:
            raise KeyError(f'{name} is already registered in {self.name}')
        
        self._module_dict[name] = module
    
    def get(self, name: str) -> Type:
        """获取注册的模块"""
        if name not in self._module_dict:
            raise KeyError(f'{name} is not registered in {self.name}')
        return self._module_dict[name]
    
    def build(self, cfg: Dict[str, Any]) -> Any:
        """构建模块实例"""
        if not isinstance(cfg, dict):
            raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
        
        cfg = cfg.copy()
        module_type = cfg.pop('type')
        
        if isinstance(module_type, str):
            module_cls = self.get(module_type)
        elif inspect.isclass(module_type):
            module_cls = module_type
        else:
            raise TypeError(f'type must be a str or class, but got {type(module_type)}')
        
        return module_cls(**cfg)
    
    def list_modules(self) -> list:
        """列出所有注册的模块"""
        return list(self._module_dict.keys())
    
    def __contains__(self, name: str) -> bool:
        """检查模块是否已注册"""
        return name in self._module_dict
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name}, items={list(self._module_dict.keys())})'


# 创建全局注册器
YOLOS_DETECTORS = Registry('detectors')
YOLOS_PROCESSORS = Registry('processors')
YOLOS_ANALYZERS = Registry('analyzers')
YOLOS_HOOKS = Registry('hooks')
YOLOS_TRANSFORMS = Registry('transforms')
YOLOS_EXPORTERS = Registry('exporters')
YOLOS_VISUALIZERS = Registry('visualizers')
YOLOS_MODELS = Registry('models')
YOLOS_TRAINERS = Registry('trainers')
YOLOS_DATASETS = Registry('datasets')
YOLOS_LOSSES = Registry('losses')
YOLOS_OPTIMIZERS = Registry('optimizers')
YOLOS_SCHEDULERS = Registry('schedulers')


def register_detector(name: Optional[str] = None, force: bool = False):
    """注册检测器装饰器"""
    return YOLOS_DETECTORS.register_module(name, force)


def register_processor(name: Optional[str] = None, force: bool = False):
    """注册处理器装饰器"""
    return YOLOS_PROCESSORS.register_module(name, force)


def register_analyzer(name: Optional[str] = None, force: bool = False):
    """注册分析器装饰器"""
    return YOLOS_ANALYZERS.register_module(name, force)


def register_hook(name: Optional[str] = None, force: bool = False):
    """注册Hook装饰器"""
    return YOLOS_HOOKS.register_module(name, force)


def register_transform(name: Optional[str] = None, force: bool = False):
    """注册变换装饰器"""
    return YOLOS_TRANSFORMS.register_module(name, force)


def register_exporter(name: Optional[str] = None, force: bool = False):
    """注册导出器装饰器"""
    return YOLOS_EXPORTERS.register_module(name, force)


def register_visualizer(name: Optional[str] = None, force: bool = False):
    """注册可视化器装饰器"""
    return YOLOS_VISUALIZERS.register_module(name, force)


def register_model(name: Optional[str] = None, force: bool = False):
    """注册模型装饰器"""
    return YOLOS_MODELS.register_module(name, force)


def register_trainer(name: Optional[str] = None, force: bool = False):
    """注册训练器装饰器"""
    return YOLOS_TRAINERS.register_module(name, force)


def register_dataset(name: Optional[str] = None, force: bool = False):
    """注册数据集装饰器"""
    return YOLOS_DATASETS.register_module(name, force)


def register_loss(name: Optional[str] = None, force: bool = False):
    """注册损失函数装饰器"""
    return YOLOS_LOSSES.register_module(name, force)


def register_optimizer(name: Optional[str] = None, force: bool = False):
    """注册优化器装饰器"""
    return YOLOS_OPTIMIZERS.register_module(name, force)


def register_scheduler(name: Optional[str] = None, force: bool = False):
    """注册调度器装饰器"""
    return YOLOS_SCHEDULERS.register_module(name, force)


# 便捷的构建函数
def build_detector(cfg: Dict[str, Any]):
    """构建检测器"""
    return YOLOS_DETECTORS.build(cfg)


def build_processor(cfg: Dict[str, Any]):
    """构建处理器"""
    return YOLOS_PROCESSORS.build(cfg)


def build_analyzer(cfg: Dict[str, Any]):
    """构建分析器"""
    return YOLOS_ANALYZERS.build(cfg)


def build_hook(cfg: Dict[str, Any]):
    """构建Hook"""
    return YOLOS_HOOKS.build(cfg)


def build_transform(cfg: Dict[str, Any]):
    """构建变换"""
    return YOLOS_TRANSFORMS.build(cfg)


def build_exporter(cfg: Dict[str, Any]):
    """构建导出器"""
    return YOLOS_EXPORTERS.build(cfg)


def build_visualizer(cfg: Dict[str, Any]):
    """构建可视化器"""
    return YOLOS_VISUALIZERS.build(cfg)


def build_model(cfg: Dict[str, Any]):
    """构建模型"""
    return YOLOS_MODELS.build(cfg)


def build_trainer(cfg: Dict[str, Any]):
    """构建训练器"""
    return YOLOS_TRAINERS.build(cfg)


def build_dataset(cfg: Dict[str, Any]):
    """构建数据集"""
    return YOLOS_DATASETS.build(cfg)


def build_loss(cfg: Dict[str, Any]):
    """构建损失函数"""
    return YOLOS_LOSSES.build(cfg)


def build_optimizer(cfg: Dict[str, Any]):
    """构建优化器"""
    return YOLOS_OPTIMIZERS.build(cfg)


def build_scheduler(cfg: Dict[str, Any]):
    """构建调度器"""
    return YOLOS_SCHEDULERS.build(cfg)


# 注册管理器
class RegistryManager:
    """注册器管理器"""
    
    _registries = {
        'detectors': YOLOS_DETECTORS,
        'processors': YOLOS_PROCESSORS,
        'analyzers': YOLOS_ANALYZERS,
        'hooks': YOLOS_HOOKS,
        'transforms': YOLOS_TRANSFORMS,
        'exporters': YOLOS_EXPORTERS,
        'visualizers': YOLOS_VISUALIZERS,
        'models': YOLOS_MODELS,
        'trainers': YOLOS_TRAINERS,
        'datasets': YOLOS_DATASETS,
        'losses': YOLOS_LOSSES,
        'optimizers': YOLOS_OPTIMIZERS,
        'schedulers': YOLOS_SCHEDULERS
    }
    
    @classmethod
    def get_registry(cls, name: str) -> Registry:
        """获取注册器"""
        if name not in cls._registries:
            raise KeyError(f'Registry {name} not found')
        return cls._registries[name]
    
    @classmethod
    def list_registries(cls) -> list:
        """列出所有注册器"""
        return list(cls._registries.keys())
    
    @classmethod
    def get_all_modules(cls) -> Dict[str, list]:
        """获取所有注册的模块"""
        result = {}
        for name, registry in cls._registries.items():
            result[name] = registry.list_modules()
        return result
    
    @classmethod
    def build_from_cfg(cls, cfg: Dict[str, Any], registry_name: str):
        """从配置构建模块"""
        registry = cls.get_registry(registry_name)
        return registry.build(cfg)


# 全局注册器管理器实例
registry_manager = RegistryManager()