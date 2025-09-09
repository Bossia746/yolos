#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块化扩展管理器
管理系统的各种功能模块和扩展
"""

import logging
import importlib
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

class ModuleStatus(Enum):
    """模块状态"""
    INACTIVE = "inactive"
    ACTIVE = "active"
    ERROR = "error"
    LOADING = "loading"

@dataclass
class ModuleInfo:
    """模块信息"""
    name: str
    version: str
    description: str
    dependencies: List[str]
    config: Dict[str, Any]
    status: ModuleStatus = ModuleStatus.INACTIVE

class ModularExtensionManager:
    """模块化扩展管理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.modules: Dict[str, ModuleInfo] = {}
        self.loaded_modules: Dict[str, Any] = {}
    
    def register_module(self, module_info: ModuleInfo) -> bool:
        """注册模块"""
        try:
            if module_info.name in self.modules:
                self.logger.warning(f"模块已存在: {module_info.name}")
                return False
            
            self.modules[module_info.name] = module_info
            self.logger.info(f"模块注册成功: {module_info.name}")
            return True
        except Exception as e:
            self.logger.error(f"模块注册失败: {e}")
            return False
    
    def load_module(self, module_name: str) -> bool:
        """加载模块"""
        try:
            if module_name not in self.modules:
                self.logger.error(f"模块不存在: {module_name}")
                return False
            
            module_info = self.modules[module_name]
            module_info.status = ModuleStatus.LOADING
            
            # 检查依赖
            if not self._check_dependencies(module_info.dependencies):
                module_info.status = ModuleStatus.ERROR
                return False
            
            # 动态加载模块
            module = importlib.import_module(f"src.plugins.{module_name}")
            self.loaded_modules[module_name] = module
            
            module_info.status = ModuleStatus.ACTIVE
            self.logger.info(f"模块加载成功: {module_name}")
            return True
        except Exception as e:
            self.logger.error(f"模块加载失败: {e}")
            if module_name in self.modules:
                self.modules[module_name].status = ModuleStatus.ERROR
            return False
    
    def unload_module(self, module_name: str) -> bool:
        """卸载模块"""
        try:
            if module_name not in self.loaded_modules:
                self.logger.warning(f"模块未加载: {module_name}")
                return False
            
            del self.loaded_modules[module_name]
            if module_name in self.modules:
                self.modules[module_name].status = ModuleStatus.INACTIVE
            
            self.logger.info(f"模块卸载成功: {module_name}")
            return True
        except Exception as e:
            self.logger.error(f"模块卸载失败: {e}")
            return False
    
    def get_module_status(self, module_name: str) -> Optional[ModuleStatus]:
        """获取模块状态"""
        if module_name in self.modules:
            return self.modules[module_name].status
        return None
    
    def list_modules(self) -> List[ModuleInfo]:
        """列出所有模块"""
        return list(self.modules.values())
    
    def _check_dependencies(self, dependencies: List[str]) -> bool:
        """检查依赖"""
        for dep in dependencies:
            if dep not in self.loaded_modules:
                self.logger.error(f"缺少依赖: {dep}")
                return False
        return True

if __name__ == "__main__":
    manager = ModularExtensionManager()
    
    # 测试模块注册
    test_module = ModuleInfo(
        name="test_module",
        version="1.0.0",
        description="测试模块",
        dependencies=[],
        config={}
    )
    
    manager.register_module(test_module)
    print(f"模块状态: {manager.get_module_status('test_module')}")