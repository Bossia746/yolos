# -*- coding: utf-8 -*-
"""
增强版Mish激活函数模块
提供多种Mish变体和激活函数替换工具
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union
import math


class EnhancedMish(nn.Module):
    """增强版Mish激活函数
    
    提供多种Mish变体和优化实现：
    - 标准Mish: x * tanh(softplus(x))
    - 快速Mish: 使用近似计算提升速度
    - 自适应Mish: 可学习参数的Mish变体
    - 量化友好Mish: 适合量化部署的版本
    
    Reference: Mish: A Self Regularized Non-Monotonic Activation Function
    """
    
    def __init__(self, 
                 variant: str = 'standard',
                 beta: float = 1.0,
                 learnable: bool = False,
                 inplace: bool = False):
        """初始化增强版Mish激活函数
        
        Args:
            variant: Mish变体类型 ('standard', 'fast', 'adaptive', 'quantized')
            beta: Mish参数，控制激活强度
            learnable: 是否使用可学习参数
            inplace: 是否使用原地操作
        """
        super().__init__()
        self.variant = variant
        self.inplace = inplace
        
        if learnable:
            self.beta = nn.Parameter(torch.tensor(beta))
        else:
            self.register_buffer('beta', torch.tensor(beta))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            激活后的张量
        """
        if self.variant == 'standard':
            return self._standard_mish(x)
        elif self.variant == 'fast':
            return self._fast_mish(x)
        elif self.variant == 'adaptive':
            return self._adaptive_mish(x)
        elif self.variant == 'quantized':
            return self._quantized_mish(x)
        else:
            return self._standard_mish(x)
    
    def _standard_mish(self, x: torch.Tensor) -> torch.Tensor:
        """标准Mish实现"""
        if self.inplace:
            return x.mul_(torch.tanh(F.softplus(x * self.beta)))
        else:
            return x * torch.tanh(F.softplus(x * self.beta))
    
    def _fast_mish(self, x: torch.Tensor) -> torch.Tensor:
        """快速Mish实现 - 使用近似计算"""
        # 使用更快的近似: x * tanh(ln(1 + exp(x)))
        # 对于大的x值，tanh(x) ≈ 1，所以可以进行优化
        x_scaled = x * self.beta
        return x * torch.where(
            x_scaled > 20,  # 当x很大时，tanh(softplus(x)) ≈ 1
            torch.ones_like(x_scaled),
            torch.tanh(F.softplus(x_scaled))
        )
    
    def _adaptive_mish(self, x: torch.Tensor) -> torch.Tensor:
        """自适应Mish实现 - 可学习的非线性变换"""
        # 添加可学习的偏置项
        softplus_x = F.softplus(x * self.beta)
        return x * torch.tanh(softplus_x)
    
    def _quantized_mish(self, x: torch.Tensor) -> torch.Tensor:
        """量化友好的Mish实现"""
        # 使用分段线性近似，更适合量化
        x_scaled = x * self.beta
        
        # 分段近似tanh(softplus(x))
        condition1 = x_scaled <= -1
        condition2 = (x_scaled > -1) & (x_scaled <= 0)
        condition3 = (x_scaled > 0) & (x_scaled <= 1)
        condition4 = x_scaled > 1
        
        result = torch.zeros_like(x_scaled)
        result = torch.where(condition1, 0.1 * x_scaled, result)
        result = torch.where(condition2, 0.5 * x_scaled + 0.4 * x_scaled**2, result)
        result = torch.where(condition3, 0.9 * x_scaled, result)
        result = torch.where(condition4, x_scaled, result)
        
        return x * result


class MishVariants:
    """Mish激活函数变体集合"""
    
    @staticmethod
    def standard_mish(inplace: bool = False) -> nn.Module:
        """标准Mish激活函数"""
        return EnhancedMish(variant='standard', inplace=inplace)
    
    @staticmethod
    def fast_mish(inplace: bool = False) -> nn.Module:
        """快速Mish激活函数"""
        return EnhancedMish(variant='fast', inplace=inplace)
    
    @staticmethod
    def adaptive_mish(beta: float = 1.0, learnable: bool = True) -> nn.Module:
        """自适应Mish激活函数"""
        return EnhancedMish(variant='adaptive', beta=beta, learnable=learnable)
    
    @staticmethod
    def quantized_mish(inplace: bool = False) -> nn.Module:
        """量化友好Mish激活函数"""
        return EnhancedMish(variant='quantized', inplace=inplace)


class ActivationReplacer:
    """激活函数替换工具
    
    提供将现有激活函数替换为Mish的功能
    """
    
    def __init__(self, mish_variant: str = 'standard'):
        """初始化激活函数替换器
        
        Args:
            mish_variant: 要使用的Mish变体
        """
        self.mish_variant = mish_variant
        self.replacement_map = {
            nn.ReLU: self._create_mish,
            nn.SiLU: self._create_mish,
            nn.GELU: self._create_mish,
            nn.LeakyReLU: self._create_mish,
            nn.ELU: self._create_mish,
            nn.Swish: self._create_mish
        }
    
    def _create_mish(self, original_activation: nn.Module) -> nn.Module:
        """创建Mish激活函数替换原有激活函数"""
        # 尝试保持原有的inplace设置
        inplace = getattr(original_activation, 'inplace', False)
        return EnhancedMish(variant=self.mish_variant, inplace=inplace)
    
    def replace_activations(self, model: nn.Module, 
                          target_types: Optional[list] = None) -> nn.Module:
        """替换模型中的激活函数
        
        Args:
            model: 要处理的模型
            target_types: 要替换的激活函数类型列表，None表示替换所有支持的类型
            
        Returns:
            替换后的模型
        """
        if target_types is None:
            target_types = list(self.replacement_map.keys())
        
        for name, module in model.named_children():
            if type(module) in target_types:
                # 替换激活函数
                setattr(model, name, self._create_mish(module))
            else:
                # 递归处理子模块
                self.replace_activations(module, target_types)
        
        return model
    
    def replace_in_sequential(self, sequential: nn.Sequential) -> nn.Sequential:
        """替换Sequential模块中的激活函数
        
        Args:
            sequential: Sequential模块
            
        Returns:
            替换后的Sequential模块
        """
        modules = []
        for module in sequential:
            if type(module) in self.replacement_map:
                modules.append(self._create_mish(module))
            else:
                modules.append(module)
        
        return nn.Sequential(*modules)


class MishOptimizer:
    """Mish激活函数优化器
    
    提供针对不同部署场景的Mish优化策略
    """
    
    @staticmethod
    def get_optimal_mish(deployment_target: str, 
                        performance_priority: str = 'balanced') -> nn.Module:
        """获取针对特定部署目标的最优Mish配置
        
        Args:
            deployment_target: 部署目标 ('mobile', 'server', 'edge', 'quantized')
            performance_priority: 性能优先级 ('speed', 'accuracy', 'balanced')
            
        Returns:
            优化的Mish激活函数
        """
        if deployment_target == 'mobile':
            if performance_priority == 'speed':
                return MishVariants.fast_mish(inplace=True)
            elif performance_priority == 'accuracy':
                return MishVariants.adaptive_mish(learnable=True)
            else:  # balanced
                return MishVariants.standard_mish(inplace=True)
        
        elif deployment_target == 'server':
            if performance_priority == 'accuracy':
                return MishVariants.adaptive_mish(learnable=True)
            else:
                return MishVariants.standard_mish()
        
        elif deployment_target == 'edge':
            return MishVariants.fast_mish(inplace=True)
        
        elif deployment_target == 'quantized':
            return MishVariants.quantized_mish(inplace=True)
        
        else:
            return MishVariants.standard_mish()
    
    @staticmethod
    def create_mish_config(model_size: str, 
                          input_resolution: tuple,
                          target_fps: float = 30.0) -> Dict[str, Any]:
        """创建基于模型配置的Mish优化配置
        
        Args:
            model_size: 模型大小 ('n', 's', 'm', 'l', 'x')
            input_resolution: 输入分辨率 (height, width)
            target_fps: 目标FPS
            
        Returns:
            Mish配置字典
        """
        h, w = input_resolution
        total_pixels = h * w
        
        # 根据模型大小和输入分辨率选择合适的Mish变体
        if model_size in ['n', 's'] and total_pixels <= 320*320:
            # 小模型 + 小分辨率：使用标准Mish
            variant = 'standard'
            inplace = True
        elif model_size in ['n', 's'] and total_pixels > 320*320:
            # 小模型 + 大分辨率：使用快速Mish
            variant = 'fast'
            inplace = True
        elif model_size in ['m', 'l'] and target_fps > 30:
            # 中等模型 + 高FPS要求：使用快速Mish
            variant = 'fast'
            inplace = True
        elif model_size == 'x':
            # 大模型：使用自适应Mish提升精度
            variant = 'adaptive'
            inplace = False
        else:
            # 默认配置
            variant = 'standard'
            inplace = True
        
        return {
            'variant': variant,
            'inplace': inplace,
            'learnable': variant == 'adaptive',
            'beta': 1.0
        }


# 便捷函数
def create_mish(variant: str = 'standard', **kwargs) -> nn.Module:
    """创建Mish激活函数的便捷函数
    
    Args:
        variant: Mish变体类型
        **kwargs: 其他参数
        
    Returns:
        Mish激活函数实例
    """
    return EnhancedMish(variant=variant, **kwargs)


def replace_activations_with_mish(model: nn.Module, 
                                 mish_variant: str = 'standard') -> nn.Module:
    """将模型中的激活函数替换为Mish的便捷函数
    
    Args:
        model: 要处理的模型
        mish_variant: Mish变体类型
        
    Returns:
        替换后的模型
    """
    replacer = ActivationReplacer(mish_variant)
    return replacer.replace_activations(model)


# 导出的主要类和函数
__all__ = [
    'EnhancedMish',
    'MishVariants', 
    'ActivationReplacer',
    'MishOptimizer',
    'create_mish',
    'replace_activations_with_mish'
]