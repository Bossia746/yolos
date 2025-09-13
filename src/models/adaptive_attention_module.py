#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AF-FPN自适应注意力模块(AAM)实现

本模块实现了基于AF-FPN论文的自适应注意力机制，融合了多种注意力策略：
1. 自适应通道注意力 - 动态调整通道权重
2. 自适应空间注意力 - 智能定位关键区域
3. 多尺度特征融合 - 增强不同尺度特征表达
4. 上下文感知机制 - 利用全局和局部上下文信息

适用场景：
- 医疗健康：小目标病灶检测、医疗器械识别
- AIoT物联网：多设备协同检测、边缘计算优化
- 智能交通：交通标志检测、车辆识别
- 工业安全：缺陷检测、安全监控

Author: YOLOS Team
Date: 2024-12-12
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass

try:
    from .advanced_yolo_optimizations import (
        ChannelAttention, SpatialAttention, ECAAttention, SimAMAttention
    )
except ImportError:
    # 如果导入失败，使用相对导入或创建简化版本
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from advanced_yolo_optimizations import (
        ChannelAttention, SpatialAttention, ECAAttention, SimAMAttention
    )


@dataclass
class AAMConfig:
    """自适应注意力模块配置"""
    # 基础配置
    channels: int = 256
    reduction_ratio: int = 16
    
    # 自适应参数
    adaptive_threshold: float = 0.5
    context_ratio: float = 0.25
    multi_scale_levels: int = 3
    
    # 融合策略
    fusion_method: str = 'weighted_sum'  # 'weighted_sum', 'concat', 'gated'
    attention_types: List[str] = None
    
    # 性能优化
    use_lightweight: bool = False
    enable_gradient_checkpointing: bool = False
    
    def __post_init__(self):
        if self.attention_types is None:
            self.attention_types = ['channel', 'spatial', 'eca']


class AdaptiveChannelAttention(nn.Module):
    """自适应通道注意力机制
    
    相比传统通道注意力，增加了自适应阈值和上下文感知能力
    """
    
    def __init__(self, channels: int, reduction: int = 16, 
                 adaptive_threshold: float = 0.5, context_ratio: float = 0.25):
        super().__init__()
        self.channels = channels
        self.adaptive_threshold = adaptive_threshold
        self.context_ratio = context_ratio
        
        # 全局平均池化和最大池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 上下文感知分支
        context_channels = max(1, int(channels * context_ratio))
        self.context_conv = nn.Sequential(
            nn.Conv2d(channels, context_channels, 1, bias=False),
            nn.BatchNorm2d(context_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 主要特征提取网络
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        # 自适应权重生成
        self.adaptive_fc = nn.Sequential(
            nn.Conv2d(channels + context_channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        
        # 传统通道注意力分支
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        traditional_attention = self.sigmoid(avg_out + max_out)
        
        # 上下文感知分支
        context_features = self.context_conv(x)
        
        # 融合特征
        combined_features = torch.cat([
            self.avg_pool(x), 
            context_features
        ], dim=1)
        
        # 自适应权重生成
        adaptive_weights = self.adaptive_fc(combined_features)
        
        # 自适应融合
        attention_weights = (
            traditional_attention * (1 - adaptive_weights) + 
            adaptive_weights * self.adaptive_threshold
        )
        
        return x * attention_weights


class AdaptiveSpatialAttention(nn.Module):
    """自适应空间注意力机制
    
    增强的空间注意力，支持多尺度特征和自适应感受野
    """
    
    def __init__(self, kernel_size: int = 7, multi_scale_levels: int = 3):
        super().__init__()
        self.multi_scale_levels = multi_scale_levels
        
        # 多尺度卷积分支
        self.multi_scale_convs = nn.ModuleList()
        for i in range(multi_scale_levels):
            scale_kernel = kernel_size + i * 2
            padding = scale_kernel // 2
            self.multi_scale_convs.append(
                nn.Conv2d(2, 1, scale_kernel, padding=padding, bias=False)
            )
        
        # 特征融合网络
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(multi_scale_levels, 1, 1, bias=False),
            nn.BatchNorm2d(1)
        )
        
        # 自适应权重生成
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.adaptive_fc = nn.Sequential(
            nn.Linear(multi_scale_levels, multi_scale_levels),
            nn.ReLU(inplace=True),
            nn.Linear(multi_scale_levels, multi_scale_levels),
            nn.Softmax(dim=1)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算通道统计信息
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        
        # 多尺度特征提取
        multi_scale_features = []
        for conv in self.multi_scale_convs:
            scale_feature = conv(spatial_input)
            multi_scale_features.append(scale_feature)
        
        # 堆叠多尺度特征
        stacked_features = torch.cat(multi_scale_features, dim=1)
        
        # 计算自适应权重
        global_context = self.adaptive_pool(stacked_features).squeeze(-1).squeeze(-1)
        adaptive_weights = self.adaptive_fc(global_context).unsqueeze(-1).unsqueeze(-1)
        
        # 加权融合多尺度特征
        weighted_features = stacked_features * adaptive_weights
        fused_features = self.fusion_conv(weighted_features)
        
        # 生成空间注意力权重
        spatial_attention = self.sigmoid(fused_features)
        
        return x * spatial_attention


class ContextAwareModule(nn.Module):
    """上下文感知模块
    
    利用全局和局部上下文信息增强特征表达
    """
    
    def __init__(self, channels: int, context_ratio: float = 0.25):
        super().__init__()
        self.channels = channels
        context_channels = max(1, int(channels * context_ratio))
        
        # 全局上下文分支
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, context_channels, 1, bias=False),
            nn.BatchNorm2d(context_channels),
            nn.ReLU(inplace=True)
        )
        
        # 局部上下文分支
        self.local_context = nn.Sequential(
            nn.Conv2d(channels, context_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(context_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 上下文融合
        self.context_fusion = nn.Sequential(
            nn.Conv2d(context_channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 提取全局和局部上下文
        global_ctx = self.global_context(x)
        local_ctx = self.local_context(x)
        
        # 上下文融合
        combined_ctx = torch.cat([global_ctx, local_ctx], dim=1)
        context_weights = self.context_fusion(combined_ctx)
        
        # 广播到原始特征图尺寸
        context_weights = F.interpolate(
            context_weights, size=x.shape[2:], 
            mode='bilinear', align_corners=False
        )
        
        return x * context_weights


class AdaptiveAttentionModule(nn.Module):
    """AF-FPN自适应注意力模块
    
    集成多种注意力机制的自适应模块，支持：
    1. 自适应通道注意力
    2. 自适应空间注意力  
    3. 上下文感知机制
    4. 多尺度特征融合
    """
    
    def __init__(self, config: AAMConfig):
        super().__init__()
        self.config = config
        self.channels = config.channels
        
        # 初始化各个注意力组件
        self.components = nn.ModuleDict()
        
        if 'channel' in config.attention_types:
            self.components['adaptive_channel'] = AdaptiveChannelAttention(
                channels=config.channels,
                reduction=config.reduction_ratio,
                adaptive_threshold=config.adaptive_threshold,
                context_ratio=config.context_ratio
            )
            
        if 'spatial' in config.attention_types:
            self.components['adaptive_spatial'] = AdaptiveSpatialAttention(
                multi_scale_levels=config.multi_scale_levels
            )
            
        if 'eca' in config.attention_types:
            self.components['eca'] = ECAAttention(config.channels)
            
        if 'simam' in config.attention_types:
            self.components['simam'] = SimAMAttention()
            
        # 上下文感知模块
        self.context_module = ContextAwareModule(
            channels=config.channels,
            context_ratio=config.context_ratio
        )
        
        # 特征融合策略
        self._init_fusion_strategy()
        
        # 轻量化优化
        if config.use_lightweight:
            self._apply_lightweight_optimization()
            
    def _init_fusion_strategy(self):
        """初始化特征融合策略"""
        num_components = len(self.components)
        
        if self.config.fusion_method == 'weighted_sum':
            # 学习权重参数
            self.fusion_weights = nn.Parameter(
                torch.ones(num_components + 1) / (num_components + 1)  # +1 for context
            )
            
        elif self.config.fusion_method == 'gated':
            # 门控融合机制
            self.gate_conv = nn.Sequential(
                nn.Conv2d(self.channels * (num_components + 1), 
                         self.channels, 1, bias=False),
                nn.BatchNorm2d(self.channels),
                nn.Sigmoid()
            )
            
        elif self.config.fusion_method == 'concat':
            # 拼接后降维
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(self.channels * (num_components + 1), 
                         self.channels, 1, bias=False),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(inplace=True)
            )
            
    def _apply_lightweight_optimization(self):
        """应用轻量化优化"""
        # 减少通道数
        for name, module in self.components.items():
            if hasattr(module, 'reduction'):
                module.reduction = min(module.reduction * 2, self.channels // 2)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入特征图 [B, C, H, W]
            
        Returns:
            torch.Tensor: 增强后的特征图 [B, C, H, W]
        """
        if self.config.enable_gradient_checkpointing and self.training:
            return self._forward_with_checkpointing(x)
        else:
            return self._forward_normal(x)
            
    def _forward_normal(self, x: torch.Tensor) -> torch.Tensor:
        """正常前向传播"""
        # 收集所有注意力增强的特征
        enhanced_features = []
        
        # 应用各个注意力组件
        for name, component in self.components.items():
            enhanced_feature = component(x)
            enhanced_features.append(enhanced_feature)
            
        # 应用上下文感知模块
        context_enhanced = self.context_module(x)
        enhanced_features.append(context_enhanced)
        
        # 特征融合
        return self._fuse_features(enhanced_features, x)
        
    def _forward_with_checkpointing(self, x: torch.Tensor) -> torch.Tensor:
        """使用梯度检查点的前向传播"""
        from torch.utils.checkpoint import checkpoint
        
        def create_forward_fn(component):
            def forward_fn(input_tensor):
                return component(input_tensor)
            return forward_fn
            
        enhanced_features = []
        
        # 使用梯度检查点应用各个组件
        for name, component in self.components.items():
            enhanced_feature = checkpoint(
                create_forward_fn(component), x, use_reentrant=False
            )
            enhanced_features.append(enhanced_feature)
            
        # 上下文模块也使用检查点
        context_enhanced = checkpoint(
            create_forward_fn(self.context_module), x, use_reentrant=False
        )
        enhanced_features.append(context_enhanced)
        
        return self._fuse_features(enhanced_features, x)
        
    def _fuse_features(self, enhanced_features: List[torch.Tensor], 
                      original: torch.Tensor) -> torch.Tensor:
        """融合增强特征"""
        if self.config.fusion_method == 'weighted_sum':
            # 加权求和
            weights = F.softmax(self.fusion_weights, dim=0)
            fused = sum(w * feat for w, feat in zip(weights, enhanced_features))
            
        elif self.config.fusion_method == 'gated':
            # 门控融合
            stacked = torch.cat(enhanced_features, dim=1)
            gate = self.gate_conv(stacked)
            fused = original * gate + stacked.mean(dim=1, keepdim=True) * (1 - gate)
            
        elif self.config.fusion_method == 'concat':
            # 拼接后降维
            stacked = torch.cat(enhanced_features, dim=1)
            fused = self.fusion_conv(stacked)
            
        else:
            # 默认平均融合
            fused = torch.stack(enhanced_features).mean(dim=0)
            
        return fused
        
    def get_attention_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """获取各个注意力组件的注意力图，用于可视化分析"""
        attention_maps = {}
        
        for name, component in self.components.items():
            if hasattr(component, 'get_attention_map'):
                attention_maps[name] = component.get_attention_map(x)
            else:
                # 简单计算注意力权重
                with torch.no_grad():
                    enhanced = component(x)
                    attention_map = torch.abs(enhanced - x).mean(dim=1, keepdim=True)
                    attention_maps[name] = attention_map
                    
        return attention_maps


def create_adaptive_attention_module(
    channels: int,
    attention_types: List[str] = None,
    fusion_method: str = 'weighted_sum',
    use_lightweight: bool = False,
    **kwargs
) -> AdaptiveAttentionModule:
    """创建自适应注意力模块的工厂函数
    
    Args:
        channels: 输入通道数
        attention_types: 注意力类型列表
        fusion_method: 融合方法
        use_lightweight: 是否使用轻量化版本
        **kwargs: 其他配置参数
        
    Returns:
        AdaptiveAttentionModule: 配置好的自适应注意力模块
    """
    if attention_types is None:
        attention_types = ['channel', 'spatial']
        
    config = AAMConfig(
        channels=channels,
        attention_types=attention_types,
        fusion_method=fusion_method,
        use_lightweight=use_lightweight,
        **kwargs
    )
    
    return AdaptiveAttentionModule(config)


# 预定义配置
AAM_CONFIGS = {
    'lightweight': {
        'attention_types': ['channel', 'eca'],
        'fusion_method': 'weighted_sum',
        'use_lightweight': True,
        'reduction_ratio': 32,
        'multi_scale_levels': 2
    },
    'standard': {
        'attention_types': ['channel', 'spatial', 'eca'],
        'fusion_method': 'weighted_sum',
        'use_lightweight': False,
        'reduction_ratio': 16,
        'multi_scale_levels': 3
    },
    'enhanced': {
        'attention_types': ['channel', 'spatial', 'eca', 'simam'],
        'fusion_method': 'gated',
        'use_lightweight': False,
        'reduction_ratio': 8,
        'multi_scale_levels': 4,
        'enable_gradient_checkpointing': True
    }
}


if __name__ == '__main__':
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试不同配置
    test_configs = [
        ('lightweight', 64, (1, 64, 32, 32)),
        ('standard', 128, (1, 128, 64, 64)),
        ('enhanced', 256, (1, 256, 128, 128))
    ]
    
    for config_name, channels, input_shape in test_configs:
        print(f"\n测试配置: {config_name}")
        print(f"输入形状: {input_shape}")
        
        # 创建模块
        config = AAMConfig(
            channels=channels,
            **AAM_CONFIGS[config_name]
        )
        
        aam = AdaptiveAttentionModule(config).to(device)
        
        # 创建测试输入
        x = torch.randn(input_shape).to(device)
        
        # 前向传播
        with torch.no_grad():
            output = aam(x)
            
        print(f"输出形状: {output.shape}")
        print(f"参数数量: {sum(p.numel() for p in aam.parameters()):,}")
        print(f"内存占用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB" if torch.cuda.is_available() else "CPU模式")
        
        # 获取注意力图
        attention_maps = aam.get_attention_maps(x)
        print(f"注意力图数量: {len(attention_maps)}")
        
        del aam, x, output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\n自适应注意力模块测试完成！")