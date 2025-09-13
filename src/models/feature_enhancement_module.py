#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AF-FPN特征增强模块(FEM)实现

本模块实现了基于AF-FPN论文的特征增强机制，专注于多尺度特征融合优化：
1. 多尺度特征提取 - 不同感受野的特征捕获
2. 自适应特征融合 - 智能权重分配和特征组合
3. 特征金字塔增强 - 优化FPN结构的信息流动
4. 小目标检测优化 - 针对医疗和AIoT场景的小目标增强

核心创新：
- 自适应权重学习机制
- 多路径特征融合策略
- 上下文信息保持
- 计算效率优化

适用场景：
- 医疗影像：病灶检测、细胞识别、医疗器械定位
- AIoT设备：多传感器融合、边缘计算优化
- 智能交通：多尺度目标检测、远近目标统一处理
- 工业检测：缺陷检测、质量控制

Author: YOLOS Team
Date: 2024-12-12
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from collections import OrderedDict

try:
    from .adaptive_attention_module import AdaptiveAttentionModule, AAMConfig
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from adaptive_attention_module import AdaptiveAttentionModule, AAMConfig


@dataclass
class FEMConfig:
    """特征增强模块配置"""
    # 基础配置
    in_channels: List[int] = None  # 输入通道数列表
    out_channels: int = 256  # 输出通道数
    num_levels: int = 5  # FPN层级数
    
    # 特征融合配置
    fusion_method: str = 'adaptive_weighted'  # 'sum', 'concat', 'adaptive_weighted', 'gated'
    use_deformable_conv: bool = False  # 是否使用可变形卷积
    use_attention: bool = True  # 是否使用注意力机制
    
    # 多尺度配置
    scale_factors: List[float] = None  # 尺度因子
    kernel_sizes: List[int] = None  # 卷积核尺寸
    
    # 优化配置
    use_separable_conv: bool = False  # 是否使用深度可分离卷积
    use_ghost_conv: bool = False  # 是否使用Ghost卷积
    dropout_rate: float = 0.1  # Dropout率
    
    # 小目标优化
    small_object_enhancement: bool = True  # 小目标增强
    min_object_size: int = 32  # 最小目标尺寸
    
    def __post_init__(self):
        if self.in_channels is None:
            self.in_channels = [256, 512, 1024, 2048]
        if self.scale_factors is None:
            self.scale_factors = [0.5, 1.0, 2.0, 4.0, 8.0]
        if self.kernel_sizes is None:
            self.kernel_sizes = [1, 3, 5, 7]


class SeparableConv2d(nn.Module):
    """深度可分离卷积"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, padding: int = 1, 
                 bias: bool = False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, 
            stride, padding, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class GhostConv(nn.Module):
    """Ghost卷积模块
    
    通过生成更多特征图来减少计算量
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 1, ratio: int = 2, dw_size: int = 3):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)
        
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, 1, 
                     kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )
        
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, 
                     dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]


class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特征提取器
    
    使用不同尺度的卷积核提取多尺度特征
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_sizes: List[int] = None, use_separable: bool = False):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [1, 3, 5, 7]
            
        self.branches = nn.ModuleList()
        
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2
            if use_separable and kernel_size > 1:
                branch = SeparableConv2d(
                    in_channels, out_channels // len(kernel_sizes), 
                    kernel_size, padding=padding
                )
            else:
                branch = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels // len(kernel_sizes), 
                             kernel_size, padding=padding, bias=False),
                    nn.BatchNorm2d(out_channels // len(kernel_sizes)),
                    nn.ReLU(inplace=True)
                )
            self.branches.append(branch)
            
        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))
            
        # 拼接所有分支输出
        concatenated = torch.cat(branch_outputs, dim=1)
        
        # 融合特征
        fused = self.fusion_conv(concatenated)
        
        return fused


class AdaptiveFeatureFusion(nn.Module):
    """自适应特征融合模块
    
    学习不同特征层的重要性权重，实现智能特征融合
    """
    
    def __init__(self, num_features: int, channels: int, 
                 fusion_method: str = 'adaptive_weighted'):
        super().__init__()
        self.num_features = num_features
        self.channels = channels
        self.fusion_method = fusion_method
        
        if fusion_method == 'adaptive_weighted':
            # 学习自适应权重
            self.weight_generator = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels * num_features, channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, num_features, 1),
                nn.Softmax(dim=1)
            )
            
        elif fusion_method == 'gated':
            # 门控融合
            self.gate_conv = nn.Sequential(
                nn.Conv2d(channels * num_features, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.Sigmoid()
            )
            
        elif fusion_method == 'concat':
            # 拼接后降维
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(channels * num_features, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
            
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """融合多个特征图
        
        Args:
            features: 特征图列表，每个特征图形状为 [B, C, H, W]
            
        Returns:
            torch.Tensor: 融合后的特征图 [B, C, H, W]
        """
        if len(features) == 1:
            return features[0]
            
        # 确保所有特征图尺寸一致
        target_size = features[0].shape[2:]
        aligned_features = []
        
        for feat in features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(
                    feat, size=target_size, 
                    mode='bilinear', align_corners=False
                )
            aligned_features.append(feat)
            
        if self.fusion_method == 'sum':
            return torch.stack(aligned_features).sum(dim=0)
            
        elif self.fusion_method == 'adaptive_weighted':
            # 拼接特征用于权重计算
            stacked = torch.cat(aligned_features, dim=1)
            weights = self.weight_generator(stacked)  # [B, num_features, 1, 1]
            
            # 加权求和
            weighted_sum = torch.zeros_like(aligned_features[0])
            for i, feat in enumerate(aligned_features):
                weight = weights[:, i:i+1, :, :]
                weighted_sum += feat * weight
                
            return weighted_sum
            
        elif self.fusion_method == 'gated':
            stacked = torch.cat(aligned_features, dim=1)
            gate = self.gate_conv(stacked)
            
            # 门控融合
            base_feature = aligned_features[0]
            enhanced_feature = torch.stack(aligned_features[1:]).mean(dim=0)
            
            return base_feature * gate + enhanced_feature * (1 - gate)
            
        elif self.fusion_method == 'concat':
            stacked = torch.cat(aligned_features, dim=1)
            return self.fusion_conv(stacked)
            
        else:
            # 默认平均融合
            return torch.stack(aligned_features).mean(dim=0)


class SmallObjectEnhancer(nn.Module):
    """小目标增强模块
    
    专门针对小目标检测进行特征增强
    """
    
    def __init__(self, channels: int, min_size: int = 32, 
                 enhancement_factor: float = 2.0):
        super().__init__()
        self.min_size = min_size
        self.enhancement_factor = enhancement_factor
        
        # 小目标特征增强网络
        self.enhancer = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        
        # 上采样模块
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """增强小目标特征
        
        Args:
            x: 输入特征图 [B, C, H, W]
            
        Returns:
            torch.Tensor: 增强后的特征图
        """
        # 计算当前特征图对应的目标尺寸
        current_size = min(x.shape[2], x.shape[3])
        
        if current_size >= self.min_size:
            # 对于较大的特征图，直接增强
            enhancement_mask = self.enhancer(x)
            enhanced = x * (1 + enhancement_mask * self.enhancement_factor)
            return enhanced
        else:
            # 对于小特征图，先上采样再增强
            upsampled = self.upsample(x)
            enhancement_mask = self.enhancer(upsampled)
            enhanced = upsampled * (1 + enhancement_mask * self.enhancement_factor)
            
            # 下采样回原始尺寸
            enhanced = F.interpolate(
                enhanced, size=x.shape[2:], 
                mode='bilinear', align_corners=False
            )
            
            return enhanced


class FeatureEnhancementModule(nn.Module):
    """AF-FPN特征增强模块
    
    集成多尺度特征提取、自适应融合、小目标增强等功能
    """
    
    def __init__(self, config: FEMConfig):
        super().__init__()
        self.config = config
        self.num_levels = config.num_levels
        
        # 输入特征对齐
        self.input_convs = nn.ModuleList()
        for in_ch in config.in_channels:
            if config.use_ghost_conv:
                conv = GhostConv(in_ch, config.out_channels)
            else:
                conv = nn.Sequential(
                    nn.Conv2d(in_ch, config.out_channels, 1, bias=False),
                    nn.BatchNorm2d(config.out_channels),
                    nn.ReLU(inplace=True)
                )
            self.input_convs.append(conv)
            
        # 多尺度特征提取器
        self.multi_scale_extractors = nn.ModuleList()
        for _ in range(self.num_levels):
            extractor = MultiScaleFeatureExtractor(
                config.out_channels, config.out_channels,
                config.kernel_sizes, config.use_separable_conv
            )
            self.multi_scale_extractors.append(extractor)
            
        # 自适应特征融合
        self.adaptive_fusion = AdaptiveFeatureFusion(
            num_features=len(config.scale_factors),
            channels=config.out_channels,
            fusion_method=config.fusion_method
        )
        
        # 注意力机制
        if config.use_attention:
            self.attention_modules = nn.ModuleList()
            for _ in range(self.num_levels):
                aam_config = AAMConfig(
                    channels=config.out_channels,
                    attention_types=['channel', 'spatial'],
                    use_lightweight=config.use_separable_conv
                )
                attention = AdaptiveAttentionModule(aam_config)
                self.attention_modules.append(attention)
        else:
            self.attention_modules = None
            
        # 小目标增强
        if config.small_object_enhancement:
            self.small_object_enhancers = nn.ModuleList()
            for _ in range(self.num_levels):
                enhancer = SmallObjectEnhancer(
                    config.out_channels, config.min_object_size
                )
                self.small_object_enhancers.append(enhancer)
        else:
            self.small_object_enhancers = None
            
        # 输出卷积
        self.output_convs = nn.ModuleList()
        for _ in range(self.num_levels):
            if config.use_separable_conv:
                conv = SeparableConv2d(config.out_channels, config.out_channels)
            else:
                conv = nn.Sequential(
                    nn.Conv2d(config.out_channels, config.out_channels, 3, 
                             padding=1, bias=False),
                    nn.BatchNorm2d(config.out_channels),
                    nn.ReLU(inplace=True)
                )
            self.output_convs.append(conv)
            
        # Dropout
        if config.dropout_rate > 0:
            self.dropout = nn.Dropout2d(config.dropout_rate)
        else:
            self.dropout = None
            
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """前向传播
        
        Args:
            features: 输入特征列表，来自backbone的不同层级
            
        Returns:
            List[torch.Tensor]: 增强后的特征列表
        """
        # 1. 输入特征对齐
        aligned_features = []
        for i, (feat, conv) in enumerate(zip(features, self.input_convs)):
            aligned_feat = conv(feat)
            aligned_features.append(aligned_feat)
            
        # 2. 多尺度特征提取
        multi_scale_features = []
        for i, (feat, extractor) in enumerate(zip(aligned_features, self.multi_scale_extractors)):
            ms_feat = extractor(feat)
            multi_scale_features.append(ms_feat)
            
        # 3. 特征金字塔融合（自顶向下 + 自底向上）
        enhanced_features = self._pyramid_fusion(multi_scale_features)
        
        # 4. 注意力增强
        if self.attention_modules is not None:
            for i, attention in enumerate(self.attention_modules):
                enhanced_features[i] = attention(enhanced_features[i])
                
        # 5. 小目标增强
        if self.small_object_enhancers is not None:
            for i, enhancer in enumerate(self.small_object_enhancers):
                enhanced_features[i] = enhancer(enhanced_features[i])
                
        # 6. 输出处理
        output_features = []
        for i, (feat, conv) in enumerate(zip(enhanced_features, self.output_convs)):
            output_feat = conv(feat)
            
            # 应用Dropout
            if self.dropout is not None and self.training:
                output_feat = self.dropout(output_feat)
                
            output_features.append(output_feat)
            
        return output_features
        
    def _pyramid_fusion(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """特征金字塔融合
        
        实现自顶向下和自底向上的特征融合
        """
        # 自顶向下路径
        top_down_features = [features[-1]]  # 从最高层开始
        
        for i in range(len(features) - 2, -1, -1):
            # 上采样高层特征
            upsampled = F.interpolate(
                top_down_features[0], 
                size=features[i].shape[2:],
                mode='bilinear', align_corners=False
            )
            
            # 融合当前层特征
            fused = self.adaptive_fusion.forward([features[i], upsampled])
            top_down_features.insert(0, fused)
            
        # 自底向上路径
        bottom_up_features = [top_down_features[0]]  # 从最低层开始
        
        for i in range(1, len(top_down_features)):
            # 下采样低层特征
            downsampled = F.interpolate(
                bottom_up_features[-1],
                size=top_down_features[i].shape[2:],
                mode='bilinear', align_corners=False
            )
            
            # 融合当前层特征
            fused = self.adaptive_fusion.forward([top_down_features[i], downsampled])
            bottom_up_features.append(fused)
            
        return bottom_up_features
        
    def get_feature_statistics(self, features: List[torch.Tensor]) -> Dict[str, float]:
        """获取特征统计信息，用于分析和调试"""
        stats = {}
        
        for i, feat in enumerate(features):
            stats[f'level_{i}_mean'] = feat.mean().item()
            stats[f'level_{i}_std'] = feat.std().item()
            stats[f'level_{i}_max'] = feat.max().item()
            stats[f'level_{i}_min'] = feat.min().item()
            
        return stats


def create_feature_enhancement_module(
    in_channels: List[int],
    out_channels: int = 256,
    fusion_method: str = 'adaptive_weighted',
    use_attention: bool = True,
    small_object_enhancement: bool = True,
    use_lightweight: bool = False,
    **kwargs
) -> FeatureEnhancementModule:
    """创建特征增强模块的工厂函数
    
    Args:
        in_channels: 输入通道数列表
        out_channels: 输出通道数
        fusion_method: 融合方法
        use_attention: 是否使用注意力
        small_object_enhancement: 是否启用小目标增强
        use_lightweight: 是否使用轻量化版本
        **kwargs: 其他配置参数
        
    Returns:
        FeatureEnhancementModule: 配置好的特征增强模块
    """
    config = FEMConfig(
        in_channels=in_channels,
        out_channels=out_channels,
        fusion_method=fusion_method,
        use_attention=use_attention,
        small_object_enhancement=small_object_enhancement,
        use_separable_conv=use_lightweight,
        use_ghost_conv=use_lightweight,
        **kwargs
    )
    
    return FeatureEnhancementModule(config)


# 预定义配置
FEM_CONFIGS = {
    'lightweight': {
        'fusion_method': 'sum',
        'use_attention': False,
        'use_separable_conv': True,
        'use_ghost_conv': True,
        'dropout_rate': 0.0,
        'kernel_sizes': [1, 3]
    },
    'standard': {
        'fusion_method': 'adaptive_weighted',
        'use_attention': True,
        'use_separable_conv': False,
        'use_ghost_conv': False,
        'dropout_rate': 0.1,
        'kernel_sizes': [1, 3, 5]
    },
    'enhanced': {
        'fusion_method': 'gated',
        'use_attention': True,
        'use_separable_conv': False,
        'use_ghost_conv': False,
        'dropout_rate': 0.15,
        'kernel_sizes': [1, 3, 5, 7],
        'small_object_enhancement': True
    }
}


if __name__ == '__main__':
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模拟backbone输出
    test_features = [
        torch.randn(1, 256, 80, 80).to(device),   # P3
        torch.randn(1, 512, 40, 40).to(device),   # P4
        torch.randn(1, 1024, 20, 20).to(device),  # P5
        torch.randn(1, 2048, 10, 10).to(device),  # P6
    ]
    
    # 测试不同配置
    for config_name in ['lightweight', 'standard', 'enhanced']:
        print(f"\n测试配置: {config_name}")
        
        # 创建模块
        config = FEMConfig(
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_levels=4,
            **FEM_CONFIGS[config_name]
        )
        
        fem = FeatureEnhancementModule(config).to(device)
        
        # 前向传播
        with torch.no_grad():
            enhanced_features = fem(test_features)
            
        print(f"输入特征数量: {len(test_features)}")
        print(f"输出特征数量: {len(enhanced_features)}")
        
        for i, feat in enumerate(enhanced_features):
            print(f"  Level {i}: {feat.shape}")
            
        print(f"参数数量: {sum(p.numel() for p in fem.parameters()):,}")
        
        # 获取特征统计
        stats = fem.get_feature_statistics(enhanced_features)
        print(f"特征统计: {len(stats)} 项指标")
        
        del fem, enhanced_features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\n特征增强模块测试完成！")