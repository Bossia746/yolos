"""SimAM注意力机制集成模块

基于YOLO-APD架构的SimAM注意力机制实现和集成
支持YOLOv11等主流YOLO模型的无缝集成

Author: YOLOS Team
Date: 2024-01-15
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import yaml
import logging
from pathlib import Path

# 导入项目核心模块
try:
    from ..core.logger import get_logger
    from ..core.config_manager import ConfigManager
except ImportError:
    import logging
    def get_logger(name):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

logger = get_logger(__name__)


class SimAMAttention(nn.Module):
    """SimAM注意力机制实现
    
    SimAM (Simple, Parameter-Free Attention Module) 是一种无参数的注意力机制，
    通过能量函数计算注意力权重，无需额外参数，计算效率高。
    
    Reference: SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks
    """
    
    def __init__(self, lambda_param: float = 1e-4, eps: float = 1e-8):
        """初始化SimAM注意力模块
        
        Args:
            lambda_param: 能量函数的lambda参数，控制注意力强度
            eps: 数值稳定性参数，防止除零错误
        """
        super(SimAMAttention, self).__init__()
        self.lambda_param = lambda_param
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入特征张量，形状为 (B, C, H, W)
            
        Returns:
            torch.Tensor: 应用注意力后的特征张量
        """
        B, C, H, W = x.size()
        
        # 计算空间维度的均值和方差
        # 形状: (B, C, 1, 1)
        mu = x.mean(dim=[2, 3], keepdim=True)
        
        # 计算方差，添加eps保证数值稳定性
        var = ((x - mu) ** 2).mean(dim=[2, 3], keepdim=True)
        
        # 计算SimAM能量函数
        # E = (x - mu)^2 / (4 * (var + lambda)) + 0.5
        energy = (x - mu) ** 2 / (4 * (var + self.lambda_param) + self.eps) + 0.5
        
        # 计算注意力权重 (使用sigmoid激活)
        attention = torch.sigmoid(-energy)
        
        # 应用注意力权重
        return x * attention
    
    def extra_repr(self) -> str:
        """返回模块的额外表示信息"""
        return f'lambda_param={self.lambda_param}, eps={self.eps}'


class SimAMIntegrator:
    """SimAM注意力机制集成器
    
    负责将SimAM注意力机制集成到现有的YOLO模型中
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化集成器
        
        Args:
            config_path: SimAM配置文件路径
        """
        self.config = self._load_config(config_path)
        self.simam_modules = {}
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载SimAM配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            Dict: 配置字典
        """
        if config_path is None:
            # 使用默认配置路径
            config_path = Path(__file__).parent.parent.parent / "config" / "simam_integration_config.yaml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"成功加载SimAM配置: {config_path}")
            return config
        except Exception as e:
            logger.warning(f"加载SimAM配置失败: {e}，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'simam_attention': {
                'enabled': True,
                'lambda_param': 1e-4,
                'eps': 1e-8,
                'integration_points': {
                    'backbone': {
                        'enabled': True,
                        'layers': [3, 6, 9],
                        'position': 'after_conv'
                    },
                    'neck': {
                        'enabled': True,
                        'fpn_levels': ['P3', 'P4', 'P5'],
                        'position': 'before_fusion'
                    }
                }
            }
        }
    
    def create_simam_module(self, name: str) -> SimAMAttention:
        """创建SimAM注意力模块
        
        Args:
            name: 模块名称
            
        Returns:
            SimAMAttention: SimAM注意力模块实例
        """
        simam_config = self.config.get('simam_attention', {})
        
        module = SimAMAttention(
            lambda_param=simam_config.get('lambda_param', 1e-4),
            eps=simam_config.get('eps', 1e-8)
        )
        
        self.simam_modules[name] = module
        logger.info(f"创建SimAM模块: {name}")
        
        return module
    
    def integrate_to_backbone(self, model: nn.Module, target_layers: Optional[List[int]] = None) -> nn.Module:
        """将SimAM集成到backbone中
        
        Args:
            model: 目标模型
            target_layers: 目标层索引列表
            
        Returns:
            nn.Module: 集成SimAM后的模型
        """
        if not self.config['simam_attention']['integration_points']['backbone']['enabled']:
            return model
        
        if target_layers is None:
            target_layers = self.config['simam_attention']['integration_points']['backbone']['layers']
        
        # 遍历模型的子模块，在指定位置插入SimAM
        for name, module in model.named_modules():
            if self._should_add_simam_to_layer(name, target_layers):
                simam_name = f"simam_{name.replace('.', '_')}"
                simam_module = self.create_simam_module(simam_name)
                
                # 创建包装模块
                wrapped_module = SimAMWrapper(module, simam_module)
                
                # 替换原模块
                self._replace_module(model, name, wrapped_module)
                logger.info(f"在 {name} 层集成SimAM注意力机制")
        
        return model
    
    def integrate_to_neck(self, model: nn.Module, fpn_levels: Optional[List[str]] = None) -> nn.Module:
        """将SimAM集成到neck中
        
        Args:
            model: 目标模型
            fpn_levels: FPN层级列表
            
        Returns:
            nn.Module: 集成SimAM后的模型
        """
        if not self.config['simam_attention']['integration_points']['neck']['enabled']:
            return model
        
        if fpn_levels is None:
            fpn_levels = self.config['simam_attention']['integration_points']['neck']['fpn_levels']
        
        # 在FPN的指定层级添加SimAM
        for level in fpn_levels:
            simam_name = f"simam_neck_{level.lower()}"
            simam_module = self.create_simam_module(simam_name)
            
            # 查找对应的FPN层并集成SimAM
            self._integrate_simam_to_fpn_level(model, level, simam_module)
        
        return model
    
    def _should_add_simam_to_layer(self, layer_name: str, target_layers: List[int]) -> bool:
        """判断是否应该在指定层添加SimAM
        
        Args:
            layer_name: 层名称
            target_layers: 目标层索引列表
            
        Returns:
            bool: 是否应该添加SimAM
        """
        # 简化的层匹配逻辑，实际使用时需要根据具体模型结构调整
        for layer_idx in target_layers:
            if f"layer{layer_idx}" in layer_name or f"block{layer_idx}" in layer_name:
                return True
        return False
    
    def _replace_module(self, model: nn.Module, module_name: str, new_module: nn.Module):
        """替换模型中的指定模块
        
        Args:
            model: 目标模型
            module_name: 模块名称
            new_module: 新模块
        """
        module_path = module_name.split('.')
        parent = model
        
        # 导航到父模块
        for part in module_path[:-1]:
            parent = getattr(parent, part)
        
        # 替换目标模块
        setattr(parent, module_path[-1], new_module)
    
    def _integrate_simam_to_fpn_level(self, model: nn.Module, level: str, simam_module: SimAMAttention):
        """将SimAM集成到指定的FPN层级
        
        Args:
            model: 目标模型
            level: FPN层级 (如 'P3', 'P4', 'P5')
            simam_module: SimAM模块
        """
        # 这里需要根据具体的模型结构来实现
        # 以下是示例实现，实际使用时需要调整
        fpn_attr_name = f"fpn_{level.lower()}"
        if hasattr(model, fpn_attr_name):
            original_fpn = getattr(model, fpn_attr_name)
            wrapped_fpn = SimAMWrapper(original_fpn, simam_module)
            setattr(model, fpn_attr_name, wrapped_fpn)
            logger.info(f"在FPN {level}层集成SimAM注意力机制")
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计信息
        
        Returns:
            Dict: 性能统计字典
        """
        stats = {
            'total_simam_modules': len(self.simam_modules),
            'module_names': list(self.simam_modules.keys()),
            'config_summary': {
                'lambda_param': self.config['simam_attention']['lambda_param'],
                'eps': self.config['simam_attention']['eps'],
                'backbone_enabled': self.config['simam_attention']['integration_points']['backbone']['enabled'],
                'neck_enabled': self.config['simam_attention']['integration_points']['neck']['enabled']
            }
        }
        return stats


class SimAMWrapper(nn.Module):
    """SimAM包装器
    
    将SimAM注意力机制包装到现有模块中
    """
    
    def __init__(self, original_module: nn.Module, simam_module: SimAMAttention):
        """初始化包装器
        
        Args:
            original_module: 原始模块
            simam_module: SimAM注意力模块
        """
        super(SimAMWrapper, self).__init__()
        self.original_module = original_module
        self.simam_module = simam_module
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            torch.Tensor: 输出张量
        """
        # 先通过原始模块
        x = self.original_module(x)
        
        # 再通过SimAM注意力机制
        x = self.simam_module(x)
        
        return x


def integrate_simam_to_model(model: nn.Module, config_path: Optional[str] = None) -> nn.Module:
    """便捷函数：将SimAM集成到模型中
    
    Args:
        model: 目标模型
        config_path: 配置文件路径
        
    Returns:
        nn.Module: 集成SimAM后的模型
    """
    integrator = SimAMIntegrator(config_path)
    
    # 集成到backbone
    model = integrator.integrate_to_backbone(model)
    
    # 集成到neck
    model = integrator.integrate_to_neck(model)
    
    logger.info("SimAM注意力机制集成完成")
    logger.info(f"性能统计: {integrator.get_performance_stats()}")
    
    return model


if __name__ == "__main__":
    # 测试代码
    import torch
    
    # 创建测试输入
    x = torch.randn(2, 256, 32, 32)
    
    # 测试SimAM模块
    simam = SimAMAttention()
    output = simam(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"SimAM模块参数数量: {sum(p.numel() for p in simam.parameters())}")
    
    # 测试集成器
    integrator = SimAMIntegrator()
    stats = integrator.get_performance_stats()
    print(f"集成器统计: {stats}")