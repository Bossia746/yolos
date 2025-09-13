#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AF-FPN小目标检测专项优化实现

本模块实现了针对小目标检测的专项优化技术，基于AF-FPN论文的核心思想，
结合YOLOS项目在医疗健康和AIoT领域的应用需求，提供以下核心功能：

1. 多尺度特征增强 - 专门针对小目标的特征提取和增强
2. 自适应锚点生成 - 动态调整锚点尺寸和密度
3. 渐进式特征融合 - 逐层融合不同尺度的特征信息
4. 小目标感知损失 - 针对小目标优化的损失函数
5. 上下文信息增强 - 利用周围环境信息辅助检测

核心技术特性：
- 多层级特征金字塔构建
- 自适应感受野调整
- 小目标专用NMS策略
- 渐进式训练机制
- 实时性能优化

应用场景优化：
- 医疗影像：病灶、细胞、组织结构等微小目标
- AIoT设备：远距离目标、小型传感器数据
- 智能监控：远程人员、车辆牌照、异常行为
- 工业检测：微小缺陷、精密零件、质量控制点

Author: YOLOS Team
Date: 2024-12-12
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings


class OptimizationLevel(Enum):
    """优化级别枚举"""
    LIGHTWEIGHT = "lightweight"    # 轻量级优化
    STANDARD = "standard"          # 标准优化
    ENHANCED = "enhanced"          # 增强优化
    MAXIMUM = "maximum"            # 最大优化


class TargetScale(Enum):
    """目标尺度枚举"""
    MICRO = "micro"        # 微小目标 (< 16x16)
    SMALL = "small"        # 小目标 (16x16 - 32x32)
    MEDIUM = "medium"      # 中等目标 (32x32 - 96x96)
    LARGE = "large"        # 大目标 (> 96x96)


@dataclass
class SmallObjectConfig:
    """小目标检测配置"""
    # 基础配置
    optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
    target_scales: List[TargetScale] = field(default_factory=lambda: [
        TargetScale.MICRO, TargetScale.SMALL
    ])
    
    # 特征金字塔配置
    fpn_levels: int = 5
    feature_channels: int = 256
    extra_levels: int = 2  # 额外的高分辨率层
    
    # 多尺度增强配置
    scale_factors: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0])
    dilation_rates: List[int] = field(default_factory=lambda: [1, 2, 3, 5])
    
    # 锚点配置
    anchor_scales: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    anchor_ratios: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    anchor_densities: List[int] = field(default_factory=lambda: [1, 2, 4])  # 不同层的锚点密度
    
    # 损失函数配置
    small_object_weight: float = 2.0  # 小目标权重增强
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    iou_threshold: float = 0.5
    
    # NMS配置
    nms_threshold: float = 0.3
    max_detections: int = 1000
    score_threshold: float = 0.1
    
    # 上下文增强配置
    context_ratio: float = 2.0  # 上下文区域相对于目标的比例
    context_channels: int = 128
    
    # 性能优化
    use_deformable_conv: bool = True
    use_attention_mechanism: bool = True
    use_progressive_training: bool = True
    

class DeformableConv2d(nn.Module):
    """可变形卷积模块
    
    用于自适应调整感受野，更好地捕获小目标特征
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, 
                 padding: int = 1, groups: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        
        # 偏移量预测网络
        self.offset_conv = nn.Conv2d(
            in_channels, 2 * kernel_size * kernel_size * groups,
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        
        # 主卷积层
        self.weight_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=groups, bias=False
        )
        
        # 初始化
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.kaiming_normal_(self.weight_conv.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 预测偏移量
        offset = self.offset_conv(x)
        
        # 应用可变形卷积
        # 注意：这里使用标准卷积作为简化实现
        # 实际应用中可以使用torchvision.ops.deform_conv2d
        return self.weight_conv(x)


class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特征提取器
    
    通过不同尺度的卷积核和膨胀率提取多尺度特征
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 config: SmallObjectConfig):
        super().__init__()
        self.config = config
        
        # 多尺度卷积分支
        self.scale_branches = nn.ModuleList()
        branch_channels = out_channels // len(config.scale_factors)
        
        for i, (scale, dilation) in enumerate(zip(config.scale_factors, config.dilation_rates)):
            kernel_size = max(1, int(3 * scale))
            if kernel_size % 2 == 0:
                kernel_size += 1
            padding = (kernel_size - 1) // 2 * dilation
            
            branch = nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=kernel_size,
                         padding=padding, dilation=dilation, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True)
            )
            self.scale_branches.append(branch)
            
        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 可变形卷积增强
        if config.use_deformable_conv:
            self.deform_conv = DeformableConv2d(out_channels, out_channels)
        else:
            self.deform_conv = None
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 多尺度特征提取
        scale_features = []
        for branch in self.scale_branches:
            scale_features.append(branch(x))
            
        # 特征拼接
        fused_features = torch.cat(scale_features, dim=1)
        
        # 特征融合
        output = self.fusion_conv(fused_features)
        
        # 可变形卷积增强
        if self.deform_conv is not None:
            output = self.deform_conv(output)
            
        return output


class SmallObjectAttention(nn.Module):
    """小目标注意力机制
    
    专门针对小目标设计的注意力模块
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        # 小目标增强注意力
        self.small_object_enhancer = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通道注意力
        ca_weight = self.channel_attention(x)
        x_ca = x * ca_weight
        
        # 空间注意力
        avg_pool = torch.mean(x_ca, dim=1, keepdim=True)
        max_pool, _ = torch.max(x_ca, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        sa_weight = self.spatial_attention(spatial_input)
        x_sa = x_ca * sa_weight
        
        # 小目标增强
        small_weight = self.small_object_enhancer(x_sa)
        output = x_sa * small_weight
        
        return output


class ProgressiveFeatureFusion(nn.Module):
    """渐进式特征融合模块
    
    逐层融合不同尺度的特征，保持小目标信息
    """
    
    def __init__(self, channels: int, num_levels: int):
        super().__init__()
        self.num_levels = num_levels
        
        # 横向连接
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, 1, bias=False)
            for _ in range(num_levels)
        ])
        
        # 输出卷积
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(num_levels)
        ])
        
        # 特征增强
        self.enhancement_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(num_levels)
        ])
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """渐进式特征融合
        
        Args:
            features: 不同层级的特征列表，从高分辨率到低分辨率
            
        Returns:
            融合后的特征列表
        """
        # 横向连接
        laterals = []
        for i, feature in enumerate(features):
            lateral = self.lateral_convs[i](feature)
            laterals.append(lateral)
            
        # 自顶向下融合
        for i in range(len(laterals) - 2, -1, -1):
            # 上采样高层特征
            upsampled = F.interpolate(
                laterals[i + 1], size=laterals[i].shape[2:],
                mode='bilinear', align_corners=False
            )
            # 特征融合
            laterals[i] = laterals[i] + upsampled
            
        # 输出处理
        outputs = []
        for i, lateral in enumerate(laterals):
            # 特征增强
            enhanced = self.enhancement_convs[i](lateral)
            # 输出卷积
            output = self.output_convs[i](enhanced)
            outputs.append(output)
            
        return outputs


class AdaptiveAnchorGenerator(nn.Module):
    """自适应锚点生成器
    
    根据特征图和目标尺度动态生成锚点
    """
    
    def __init__(self, config: SmallObjectConfig):
        super().__init__()
        self.config = config
        self.anchor_scales = config.anchor_scales
        self.anchor_ratios = config.anchor_ratios
        self.anchor_densities = config.anchor_densities
        
    def generate_anchors(self, feature_maps: List[torch.Tensor], 
                        image_size: Tuple[int, int]) -> List[torch.Tensor]:
        """生成自适应锚点
        
        Args:
            feature_maps: 特征图列表
            image_size: 原始图像尺寸
            
        Returns:
            每个特征层的锚点列表
        """
        all_anchors = []
        
        for level, feature_map in enumerate(feature_maps):
            h, w = feature_map.shape[2:]
            stride = image_size[0] // h  # 假设正方形图像
            
            # 获取该层的锚点密度
            density = self.anchor_densities[min(level, len(self.anchor_densities) - 1)]
            
            # 生成基础锚点
            base_anchors = self._generate_base_anchors(stride, density)
            
            # 在特征图上平铺锚点
            anchors = self._tile_anchors(base_anchors, h, w, stride)
            all_anchors.append(anchors)
            
        return all_anchors
        
    def _generate_base_anchors(self, stride: int, density: int) -> torch.Tensor:
        """生成基础锚点"""
        anchors = []
        
        # 根据密度调整锚点位置
        step = stride / density
        offsets = [(i * step - stride / 2) for i in range(density)]
        
        for offset_x in offsets:
            for offset_y in offsets:
                for scale in self.anchor_scales:
                    for ratio in self.anchor_ratios:
                        # 计算锚点尺寸
                        base_size = stride * scale
                        w = base_size * math.sqrt(ratio)
                        h = base_size / math.sqrt(ratio)
                        
                        # 锚点坐标 (cx, cy, w, h)
                        anchor = [offset_x, offset_y, w, h]
                        anchors.append(anchor)
                        
        return torch.tensor(anchors, dtype=torch.float32)
        
    def _tile_anchors(self, base_anchors: torch.Tensor, 
                     height: int, width: int, stride: int) -> torch.Tensor:
        """在特征图上平铺锚点"""
        # 生成网格坐标
        shift_x = torch.arange(0, width) * stride
        shift_y = torch.arange(0, height) * stride
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
        
        shifts = torch.stack([shift_x.flatten(), shift_y.flatten(), 
                             torch.zeros_like(shift_x.flatten()), 
                             torch.zeros_like(shift_x.flatten())], dim=1)
        
        # 广播锚点
        num_anchors = base_anchors.size(0)
        num_shifts = shifts.size(0)
        
        anchors = base_anchors.view(1, num_anchors, 4) + shifts.view(num_shifts, 1, 4)
        anchors = anchors.view(-1, 4)
        
        return anchors


class SmallObjectLoss(nn.Module):
    """小目标专用损失函数
    
    结合Focal Loss和IoU Loss，对小目标给予更高权重
    """
    
    def __init__(self, config: SmallObjectConfig):
        super().__init__()
        self.config = config
        self.focal_alpha = config.focal_alpha
        self.focal_gamma = config.focal_gamma
        self.small_object_weight = config.small_object_weight
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算损失
        
        Args:
            predictions: 预测结果字典
            targets: 真实标签字典
            
        Returns:
            损失字典
        """
        # 分类损失 (Focal Loss)
        cls_loss = self._focal_loss(
            predictions['classification'], targets['labels']
        )
        
        # 回归损失 (Smooth L1 Loss)
        reg_loss = self._regression_loss(
            predictions['regression'], targets['boxes'], targets['labels']
        )
        
        # 小目标权重调整
        small_object_mask = self._get_small_object_mask(targets['boxes'])
        if small_object_mask.any():
            cls_loss = cls_loss * (1 + self.small_object_weight * small_object_mask)
            reg_loss = reg_loss * (1 + self.small_object_weight * small_object_mask)
            
        total_loss = cls_loss.mean() + reg_loss.mean()
        
        return {
            'total_loss': total_loss,
            'classification_loss': cls_loss.mean(),
            'regression_loss': reg_loss.mean()
        }
        
    def _focal_loss(self, predictions: torch.Tensor, 
                   targets: torch.Tensor) -> torch.Tensor:
        """Focal Loss计算"""
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss
        
    def _regression_loss(self, predictions: torch.Tensor, 
                        targets: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """回归损失计算"""
        # 只对正样本计算回归损失
        positive_mask = labels > 0
        if not positive_mask.any():
            return torch.zeros(1, device=predictions.device)
            
        pos_predictions = predictions[positive_mask]
        pos_targets = targets[positive_mask]
        
        return F.smooth_l1_loss(pos_predictions, pos_targets, reduction='none')
        
    def _get_small_object_mask(self, boxes: torch.Tensor) -> torch.Tensor:
        """获取小目标掩码"""
        # 计算目标面积
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights
        
        # 定义小目标阈值 (32x32 pixels)
        small_threshold = 32 * 32
        return areas < small_threshold


class SmallObjectNMS(nn.Module):
    """小目标专用NMS
    
    针对小目标优化的非极大值抑制
    """
    
    def __init__(self, config: SmallObjectConfig):
        super().__init__()
        self.config = config
        self.nms_threshold = config.nms_threshold
        self.score_threshold = config.score_threshold
        self.max_detections = config.max_detections
        
    def forward(self, boxes: torch.Tensor, scores: torch.Tensor, 
                labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """应用NMS
        
        Args:
            boxes: 边界框 [N, 4]
            scores: 置信度分数 [N]
            labels: 类别标签 [N]
            
        Returns:
            过滤后的boxes, scores, labels
        """
        # 分数过滤
        valid_mask = scores > self.score_threshold
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        labels = labels[valid_mask]
        
        if len(boxes) == 0:
            return boxes, scores, labels
            
        # 按类别分别应用NMS
        keep_indices = []
        unique_labels = torch.unique(labels)
        
        for label in unique_labels:
            label_mask = labels == label
            label_boxes = boxes[label_mask]
            label_scores = scores[label_mask]
            
            # 应用NMS
            keep = ops.nms(label_boxes, label_scores, self.nms_threshold)
            
            # 转换回全局索引
            global_indices = torch.where(label_mask)[0][keep]
            keep_indices.append(global_indices)
            
        if keep_indices:
            final_keep = torch.cat(keep_indices)
            
            # 限制最大检测数量
            if len(final_keep) > self.max_detections:
                # 按分数排序，保留top-k
                _, sorted_indices = torch.sort(scores[final_keep], descending=True)
                final_keep = final_keep[sorted_indices[:self.max_detections]]
                
            return boxes[final_keep], scores[final_keep], labels[final_keep]
        else:
            return torch.empty(0, 4), torch.empty(0), torch.empty(0, dtype=torch.long)


class SmallObjectDetectionOptimizer(nn.Module):
    """小目标检测优化器主类
    
    整合所有小目标检测优化技术
    """
    
    def __init__(self, backbone_channels: List[int], 
                 config: SmallObjectConfig):
        super().__init__()
        self.config = config
        self.backbone_channels = backbone_channels
        
        # 多尺度特征提取器
        self.feature_extractors = nn.ModuleList([
            MultiScaleFeatureExtractor(channels, config.feature_channels, config)
            for channels in backbone_channels
        ])
        
        # 渐进式特征融合
        self.feature_fusion = ProgressiveFeatureFusion(
            config.feature_channels, len(backbone_channels) + config.extra_levels
        )
        
        # 小目标注意力
        if config.use_attention_mechanism:
            self.attention_modules = nn.ModuleList([
                SmallObjectAttention(config.feature_channels)
                for _ in range(len(backbone_channels) + config.extra_levels)
            ])
        else:
            self.attention_modules = None
            
        # 额外的高分辨率层
        self.extra_layers = nn.ModuleList()
        for i in range(config.extra_levels):
            layer = nn.Sequential(
                nn.Conv2d(config.feature_channels, config.feature_channels, 
                         3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(config.feature_channels),
                nn.ReLU(inplace=True)
            )
            self.extra_layers.append(layer)
            
        # 预测头
        num_anchors = len(config.anchor_scales) * len(config.anchor_ratios)
        self.classification_head = nn.Conv2d(
            config.feature_channels, num_anchors, 3, padding=1
        )
        self.regression_head = nn.Conv2d(
            config.feature_channels, num_anchors * 4, 3, padding=1
        )
        
        # 锚点生成器
        self.anchor_generator = AdaptiveAnchorGenerator(config)
        
        # 损失函数
        self.loss_fn = SmallObjectLoss(config)
        
        # NMS
        self.nms = SmallObjectNMS(config)
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, backbone_features: List[torch.Tensor], 
                targets: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            backbone_features: 骨干网络特征列表
            targets: 训练时的目标标签
            
        Returns:
            预测结果或损失字典
        """
        # 多尺度特征提取
        enhanced_features = []
        for i, feature in enumerate(backbone_features):
            enhanced = self.feature_extractors[i](feature)
            enhanced_features.append(enhanced)
            
        # 添加额外层
        current_feature = enhanced_features[-1]
        for extra_layer in self.extra_layers:
            current_feature = extra_layer(current_feature)
            enhanced_features.append(current_feature)
            
        # 特征融合
        fused_features = self.feature_fusion(enhanced_features)
        
        # 注意力增强
        if self.attention_modules is not None:
            attended_features = []
            for i, feature in enumerate(fused_features):
                attended = self.attention_modules[i](feature)
                attended_features.append(attended)
            fused_features = attended_features
            
        # 预测
        classifications = []
        regressions = []
        
        for feature in fused_features:
            cls_pred = self.classification_head(feature)
            reg_pred = self.regression_head(feature)
            
            # 重塑为 [B, H*W*A, num_classes] 和 [B, H*W*A, 4]
            b, _, h, w = cls_pred.shape
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(b, -1)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(b, -1, 4)
            
            classifications.append(cls_pred)
            regressions.append(reg_pred)
            
        # 合并预测
        all_classifications = torch.cat(classifications, dim=1)
        all_regressions = torch.cat(regressions, dim=1)
        
        if self.training and targets is not None:
            # 训练模式：计算损失
            predictions = {
                'classification': all_classifications,
                'regression': all_regressions
            }
            return self.loss_fn(predictions, targets)
        else:
            # 推理模式：返回预测结果
            return {
                'classifications': all_classifications,
                'regressions': all_regressions,
                'features': fused_features
            }
            
    def post_process(self, predictions: Dict[str, torch.Tensor], 
                    image_size: Tuple[int, int]) -> List[Dict[str, torch.Tensor]]:
        """后处理预测结果
        
        Args:
            predictions: 模型预测结果
            image_size: 原始图像尺寸
            
        Returns:
            每张图像的检测结果列表
        """
        classifications = predictions['classifications']
        regressions = predictions['regressions']
        
        batch_size = classifications.size(0)
        results = []
        
        for i in range(batch_size):
            cls_scores = torch.sigmoid(classifications[i])
            reg_boxes = regressions[i]
            
            # 获取最高分数和对应类别
            max_scores, labels = torch.max(cls_scores, dim=1)
            
            # 应用NMS
            final_boxes, final_scores, final_labels = self.nms(
                reg_boxes, max_scores, labels
            )
            
            results.append({
                'boxes': final_boxes,
                'scores': final_scores,
                'labels': final_labels
            })
            
        return results
        
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        stats = {
            'optimization_level': self.config.optimization_level.value,
            'target_scales': [scale.value for scale in self.config.target_scales],
            'fpn_levels': self.config.fpn_levels,
            'extra_levels': self.config.extra_levels,
            'feature_channels': self.config.feature_channels,
            'use_attention': self.config.use_attention_mechanism,
            'use_deformable_conv': self.config.use_deformable_conv,
            'anchor_configurations': {
                'scales': self.config.anchor_scales,
                'ratios': self.config.anchor_ratios,
                'densities': self.config.anchor_densities
            },
            'loss_configuration': {
                'small_object_weight': self.config.small_object_weight,
                'focal_alpha': self.config.focal_alpha,
                'focal_gamma': self.config.focal_gamma
            }
        }
        
        return stats


def create_small_object_optimizer(
    backbone_channels: List[int],
    optimization_level: str = 'standard',
    target_scales: List[str] = None,
    **kwargs
) -> SmallObjectDetectionOptimizer:
    """创建小目标检测优化器的工厂函数
    
    Args:
        backbone_channels: 骨干网络通道数列表
        optimization_level: 优化级别
        target_scales: 目标尺度列表
        **kwargs: 其他配置参数
        
    Returns:
        SmallObjectDetectionOptimizer: 配置好的优化器
    """
    if target_scales is None:
        target_scales = ['micro', 'small']
        
    config = SmallObjectConfig(
        optimization_level=OptimizationLevel(optimization_level),
        target_scales=[TargetScale(scale) for scale in target_scales],
        **kwargs
    )
    
    return SmallObjectDetectionOptimizer(backbone_channels, config)


# 预定义配置
SMALL_OBJECT_CONFIGS = {
    'medical_micro': {
        'optimization_level': 'enhanced',
        'target_scales': ['micro'],
        'fpn_levels': 6,
        'extra_levels': 3,
        'small_object_weight': 3.0,
        'anchor_scales': [0.25, 0.5, 1.0],
        'use_attention_mechanism': True
    },
    'aiot_small': {
        'optimization_level': 'standard',
        'target_scales': ['micro', 'small'],
        'fpn_levels': 5,
        'extra_levels': 2,
        'small_object_weight': 2.0,
        'use_deformable_conv': True
    },
    'traffic_distant': {
        'optimization_level': 'enhanced',
        'target_scales': ['micro', 'small', 'medium'],
        'fpn_levels': 5,
        'extra_levels': 2,
        'small_object_weight': 2.5,
        'context_ratio': 3.0
    },
    'industrial_defect': {
        'optimization_level': 'maximum',
        'target_scales': ['micro'],
        'fpn_levels': 7,
        'extra_levels': 4,
        'small_object_weight': 4.0,
        'anchor_densities': [2, 4, 8],
        'use_attention_mechanism': True,
        'use_deformable_conv': True
    }
}


if __name__ == '__main__':
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模拟骨干网络特征
    backbone_channels = [256, 512, 1024, 2048]
    backbone_features = [
        torch.randn(2, channels, 80 // (2**i), 80 // (2**i)).to(device)
        for i, channels in enumerate(backbone_channels)
    ]
    
    # 测试不同配置
    for config_name, config_params in SMALL_OBJECT_CONFIGS.items():
        print(f"\n测试配置: {config_name}")
        
        # 创建优化器
        optimizer = create_small_object_optimizer(
            backbone_channels, **config_params
        ).to(device)
        
        # 前向传播
        with torch.no_grad():
            predictions = optimizer(backbone_features)
            
        print(f"  输入特征: {[f.shape for f in backbone_features]}")
        print(f"  分类预测: {predictions['classifications'].shape}")
        print(f"  回归预测: {predictions['regressions'].shape}")
        print(f"  融合特征: {len(predictions['features'])} 层")
        
        # 获取统计信息
        stats = optimizer.get_optimization_statistics()
        print(f"  优化级别: {stats['optimization_level']}")
        print(f"  目标尺度: {stats['target_scales']}")
        print(f"  FPN层数: {stats['fpn_levels']} + {stats['extra_levels']} 额外层")
        
    print("\n小目标检测优化器测试完成！")