#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级YOLO优化技术
包括神经架构搜索、Transformer集成、自监督预训练等前沿技术
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import random
import math
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..utils.logging_manager import LoggingManager


@dataclass
class ArchitectureConfig:
    """神经架构配置"""
    backbone_depth: int = 5
    neck_channels: int = 256
    head_layers: int = 3
    activation: str = 'SiLU'
    attention_type: str = 'SE'  # SE, CBAM, ECA
    use_transformer: bool = False
    transformer_layers: int = 6


class AttentionModule(nn.Module):
    """注意力机制模块"""
    
    def __init__(self, channels: int, attention_type: str = 'SE'):
        super().__init__()
        self.attention_type = attention_type
        
        if attention_type == 'SE':
            self.attention = SEAttention(channels)
        elif attention_type == 'CBAM':
            self.attention = CBAMAttention(channels)
        elif attention_type == 'ECA':
            self.attention = ECAAttention(channels)
        else:
            self.attention = nn.Identity()
    
    def forward(self, x):
        return self.attention(x)


class SEAttention(nn.Module):
    """Squeeze-and-Excitation注意力"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CBAMAttention(nn.Module):
    """Convolutional Block Attention Module"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x


class ChannelAttention(nn.Module):
    """通道注意力"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力"""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class ECAAttention(nn.Module):
    """Efficient Channel Attention"""
    
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class TransformerBlock(nn.Module):
    """Transformer块"""
    
    def __init__(self, 
                 d_model: int = 256,
                 nhead: int = 8,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.GELU()
    
    def forward(self, src):
        # Self-attention
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed forward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class YOLOTransformerNeck(nn.Module):
    """YOLO Transformer颈部网络"""
    
    def __init__(self, 
                 in_channels: List[int] = [256, 512, 1024],
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6):
        super().__init__()
        
        self.in_channels = in_channels
        self.d_model = d_model
        
        # 输入投影层
        self.input_projections = nn.ModuleList([
            nn.Conv2d(ch, d_model, 1) for ch in in_channels
        ])
        
        # Transformer编码器
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, nhead) for _ in range(num_layers)
        ])
        
        # 位置编码
        self.pos_encoding = PositionalEncoding2D(d_model)
        
        # 输出投影层
        self.output_projections = nn.ModuleList([
            nn.Conv2d(d_model, ch, 1) for ch in in_channels
        ])
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: 多尺度特征列表 [P3, P4, P5]
        Returns:
            增强后的特征列表
        """
        # 投影到统一维度
        projected_features = []
        for i, feat in enumerate(features):
            proj_feat = self.input_projections[i](feat)
            projected_features.append(proj_feat)
        
        # 特征展平和拼接
        flattened_features = []
        spatial_shapes = []
        
        for feat in projected_features:
            b, c, h, w = feat.shape
            spatial_shapes.append((h, w))
            feat_flat = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
            flattened_features.append(feat_flat)
        
        # 拼接所有尺度的特征
        all_features = torch.cat(flattened_features, dim=1)  # (B, N, C)
        
        # 添加位置编码
        all_features = self.pos_encoding(all_features, spatial_shapes)
        
        # Transformer处理
        enhanced_features = all_features
        for transformer_layer in self.transformer_layers:
            enhanced_features = transformer_layer(enhanced_features)
        
        # 重新分割和重塑特征
        output_features = []
        start_idx = 0
        
        for i, (h, w) in enumerate(spatial_shapes):
            feat_len = h * w
            feat = enhanced_features[:, start_idx:start_idx+feat_len, :]
            feat = feat.transpose(1, 2).reshape(-1, self.d_model, h, w)
            
            # 投影回原始维度
            output_feat = self.output_projections[i](feat)
            output_features.append(output_feat)
            
            start_idx += feat_len
        
        return output_features


class PositionalEncoding2D(nn.Module):
    """2D位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
    
    def forward(self, x: torch.Tensor, spatial_shapes: List[Tuple[int, int]]):
        """
        Args:
            x: (B, N, C) 特征张量
            spatial_shapes: 每个尺度的空间形状列表
        """
        batch_size, seq_len, d_model = x.shape
        
        # 创建位置编码
        pos_encoding = torch.zeros(seq_len, d_model, device=x.device)
        
        position = 0
        for h, w in spatial_shapes:
            # 为当前尺度创建2D位置编码
            y_pos = torch.arange(h, device=x.device).unsqueeze(1).repeat(1, w).flatten()
            x_pos = torch.arange(w, device=x.device).unsqueeze(0).repeat(h, 1).flatten()
            
            # 计算位置编码
            div_term = torch.exp(torch.arange(0, d_model, 2, device=x.device) * 
                               -(math.log(self.max_len) / d_model))
            
            for i in range(h * w):
                pos = position + i
                if pos < seq_len:
                    pos_encoding[pos, 0::4] = torch.sin(y_pos[i] * div_term[::2])
                    pos_encoding[pos, 1::4] = torch.cos(y_pos[i] * div_term[::2])
                    pos_encoding[pos, 2::4] = torch.sin(x_pos[i] * div_term[::2])
                    pos_encoding[pos, 3::4] = torch.cos(x_pos[i] * div_term[::2])
            
            position += h * w
        
        return x + pos_encoding.unsqueeze(0)


class NeuralArchitectureSearch:
    """神经架构搜索"""
    
    def __init__(self, 
                 search_space: Dict[str, List],
                 population_size: int = 20,
                 generations: int = 50,
                 mutation_rate: float = 0.1):
        """
        初始化神经架构搜索
        
        Args:
            search_space: 搜索空间定义
            population_size: 种群大小
            generations: 进化代数
            mutation_rate: 变异率
        """
        self.search_space = search_space
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
        self.logger = LoggingManager().get_logger("NeuralArchitectureSearch")
        
        # 初始化种群
        self.population = self._initialize_population()
        self.fitness_history = []
    
    def _initialize_population(self) -> List[ArchitectureConfig]:
        """初始化种群"""
        population = []
        
        for _ in range(self.population_size):
            config = ArchitectureConfig()
            
            # 随机选择架构参数
            for param, values in self.search_space.items():
                if hasattr(config, param):
                    setattr(config, param, random.choice(values))
            
            population.append(config)
        
        return population
    
    def evaluate_architecture(self, config: ArchitectureConfig) -> float:
        """评估架构性能"""
        # 这里应该实际训练和评估模型
        # 为了演示，使用简化的评估函数
        
        # 基于参数数量和复杂度的简单评估
        complexity_score = (
            config.backbone_depth * 0.1 +
            config.neck_channels / 1000 +
            config.head_layers * 0.05
        )
        
        # 基于架构选择的性能估计
        performance_score = 0.8
        if config.use_transformer:
            performance_score += 0.05
        if config.attention_type in ['CBAM', 'ECA']:
            performance_score += 0.03
        
        # 综合评分（性能 - 复杂度惩罚）
        fitness = performance_score - complexity_score * 0.1
        
        return max(0.0, fitness)
    
    def crossover(self, parent1: ArchitectureConfig, parent2: ArchitectureConfig) -> ArchitectureConfig:
        """交叉操作"""
        child = ArchitectureConfig()
        
        # 随机选择每个参数来自哪个父代
        for param in ['backbone_depth', 'neck_channels', 'head_layers', 
                     'activation', 'attention_type', 'use_transformer', 'transformer_layers']:
            if hasattr(child, param):
                parent = random.choice([parent1, parent2])
                setattr(child, param, getattr(parent, param))
        
        return child
    
    def mutate(self, config: ArchitectureConfig) -> ArchitectureConfig:
        """变异操作"""
        if random.random() < self.mutation_rate:
            # 随机选择一个参数进行变异
            param = random.choice(list(self.search_space.keys()))
            if hasattr(config, param):
                new_value = random.choice(self.search_space[param])
                setattr(config, param, new_value)
        
        return config
    
    def search(self) -> ArchitectureConfig:
        """执行神经架构搜索"""
        self.logger.info(f"开始神经架构搜索，种群大小: {self.population_size}, 进化代数: {self.generations}")
        
        best_config = None
        best_fitness = 0.0
        
        for generation in range(self.generations):
            # 评估当前种群
            fitness_scores = []
            for config in self.population:
                fitness = self.evaluate_architecture(config)
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_config = config
            
            self.fitness_history.append(max(fitness_scores))
            
            # 选择、交叉、变异
            new_population = []
            
            # 保留最优个体（精英策略）
            elite_count = max(1, self.population_size // 10)
            elite_indices = np.argsort(fitness_scores)[-elite_count:]
            for idx in elite_indices:
                new_population.append(self.population[idx])
            
            # 生成新个体
            while len(new_population) < self.population_size:
                # 轮盘赌选择
                parent1 = self._roulette_selection(fitness_scores)
                parent2 = self._roulette_selection(fitness_scores)
                
                # 交叉
                child = self.crossover(parent1, parent2)
                
                # 变异
                child = self.mutate(child)
                
                new_population.append(child)
            
            self.population = new_population
            
            if generation % 10 == 0:
                self.logger.info(f"Generation {generation}, 最佳适应度: {best_fitness:.4f}")
        
        self.logger.info(f"搜索完成，最佳适应度: {best_fitness:.4f}")
        return best_config
    
    def _roulette_selection(self, fitness_scores: List[float]) -> ArchitectureConfig:
        """轮盘赌选择"""
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            return random.choice(self.population)
        
        pick = random.uniform(0, total_fitness)
        current = 0
        
        for i, fitness in enumerate(fitness_scores):
            current += fitness
            if current >= pick:
                return self.population[i]
        
        return self.population[-1]


class SelfSupervisedPretraining:
    """自监督预训练"""
    
    def __init__(self, 
                 model: nn.Module,
                 temperature: float = 0.07,
                 mask_ratio: float = 0.75):
        """
        初始化自监督预训练
        
        Args:
            model: 待预训练的模型
            temperature: 对比学习温度参数
            mask_ratio: 掩码比例
        """
        self.model = model
        self.temperature = temperature
        self.mask_ratio = mask_ratio
        
        self.logger = LoggingManager().get_logger("SelfSupervisedPretraining")
        
        # 对比学习投影头
        self.projection_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # 掩码图像建模解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Sigmoid()
        )
    
    def contrastive_loss(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """计算对比损失"""
        # 归一化特征
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features1, features2.T) / self.temperature
        
        # 创建标签（对角线为正样本）
        batch_size = features1.size(0)
        labels = torch.arange(batch_size, device=features1.device)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
    
    def masked_image_modeling_loss(self, 
                                 original_images: torch.Tensor,
                                 reconstructed_images: torch.Tensor,
                                 mask: torch.Tensor) -> torch.Tensor:
        """计算掩码图像建模损失"""
        # 只在掩码区域计算损失
        loss = F.mse_loss(reconstructed_images * mask, original_images * mask)
        return loss
    
    def create_random_mask(self, batch_size: int, height: int, width: int) -> torch.Tensor:
        """创建随机掩码"""
        mask = torch.rand(batch_size, 1, height, width)
        mask = (mask > self.mask_ratio).float()
        return mask
    
    def pretrain_step(self, images: torch.Tensor) -> Dict[str, float]:
        """单步预训练"""
        batch_size, channels, height, width = images.shape
        
        # 数据增强（创建两个不同的视图）
        augmented_images1 = self._augment_images(images)
        augmented_images2 = self._augment_images(images)
        
        # 提取特征
        features1 = self.model(augmented_images1)
        features2 = self.model(augmented_images2)
        
        # 全局平均池化
        pooled_features1 = F.adaptive_avg_pool2d(features1, 1).flatten(1)
        pooled_features2 = F.adaptive_avg_pool2d(features2, 1).flatten(1)
        
        # 投影到对比学习空间
        projected_features1 = self.projection_head(pooled_features1)
        projected_features2 = self.projection_head(pooled_features2)
        
        # 对比损失
        contrastive_loss = self.contrastive_loss(projected_features1, projected_features2)
        
        # 掩码图像建模
        mask = self.create_random_mask(batch_size, height//8, width//8).to(images.device)
        masked_features = features1 * mask
        reconstructed_images = self.decoder(masked_features)
        
        # 调整重建图像尺寸
        reconstructed_images = F.interpolate(reconstructed_images, size=(height, width))
        image_mask = F.interpolate(mask, size=(height, width))
        
        mim_loss = self.masked_image_modeling_loss(images, reconstructed_images, image_mask)
        
        # 总损失
        total_loss = contrastive_loss + mim_loss
        
        return {
            'total_loss': total_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'mim_loss': mim_loss.item()
        }
    
    def _augment_images(self, images: torch.Tensor) -> torch.Tensor:
        """图像增强"""
        # 简单的增强：随机翻转和颜色抖动
        augmented = images.clone()
        
        # 随机水平翻转
        if random.random() > 0.5:
            augmented = torch.flip(augmented, dims=[3])
        
        # 颜色抖动
        if random.random() > 0.5:
            brightness = 0.8 + random.random() * 0.4
            augmented = augmented * brightness
            augmented = torch.clamp(augmented, 0, 1)
        
        return augmented


class ModelPruning:
    """模型剪枝"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.logger = LoggingManager().get_logger("ModelPruning")
    
    def structured_pruning(self, sparsity: float = 0.3) -> nn.Module:
        """结构化剪枝 - 移除整个通道"""
        self.logger.info(f"开始结构化剪枝，稀疏度: {sparsity}")
        
        # 计算每个卷积层的重要性
        importance_scores = self._calculate_channel_importance()
        
        # 确定要剪枝的通道
        channels_to_prune = self._select_channels_to_prune(importance_scores, sparsity)
        
        # 执行剪枝
        pruned_model = self._prune_channels(channels_to_prune)
        
        self.logger.info("结构化剪枝完成")
        return pruned_model
    
    def unstructured_pruning(self, sparsity: float = 0.3) -> nn.Module:
        """非结构化剪枝 - 移除单个权重"""
        self.logger.info(f"开始非结构化剪枝，稀疏度: {sparsity}")
        
        # 计算权重重要性
        weight_importance = self._calculate_weight_importance()
        
        # 确定剪枝阈值
        threshold = self._calculate_pruning_threshold(weight_importance, sparsity)
        
        # 执行剪枝
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                mask = torch.abs(module.weight.data) > threshold
                module.weight.data *= mask.float()
        
        self.logger.info("非结构化剪枝完成")
        return self.model
    
    def _calculate_channel_importance(self) -> Dict[str, torch.Tensor]:
        """计算通道重要性"""
        importance_scores = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # 使用权重的L1范数作为重要性指标
                weight_norm = torch.norm(module.weight.data, p=1, dim=(1, 2, 3))
                importance_scores[name] = weight_norm
        
        return importance_scores
    
    def _select_channels_to_prune(self, 
                                importance_scores: Dict[str, torch.Tensor], 
                                sparsity: float) -> Dict[str, List[int]]:
        """选择要剪枝的通道"""
        channels_to_prune = {}
        
        for layer_name, scores in importance_scores.items():
            num_channels = len(scores)
            num_to_prune = int(num_channels * sparsity)
            
            # 选择重要性最低的通道
            _, indices = torch.topk(scores, num_to_prune, largest=False)
            channels_to_prune[layer_name] = indices.tolist()
        
        return channels_to_prune
    
    def _prune_channels(self, channels_to_prune: Dict[str, List[int]]) -> nn.Module:
        """执行通道剪枝"""
        # 这里需要重新构建网络结构，移除指定的通道
        # 实际实现会比较复杂，需要处理层间的依赖关系
        
        # 简化实现：将要剪枝的通道权重置零
        for name, module in self.model.named_modules():
            if name in channels_to_prune:
                indices = channels_to_prune[name]
                if isinstance(module, nn.Conv2d):
                    module.weight.data[indices] = 0
                    if module.bias is not None:
                        module.bias.data[indices] = 0
        
        return self.model
    
    def _calculate_weight_importance(self) -> torch.Tensor:
        """计算权重重要性"""
        all_weights = []
        
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                all_weights.append(module.weight.data.abs().flatten())
        
        return torch.cat(all_weights)
    
    def _calculate_pruning_threshold(self, 
                                   weight_importance: torch.Tensor, 
                                   sparsity: float) -> float:
        """计算剪枝阈值"""
        sorted_weights, _ = torch.sort(weight_importance)
        threshold_index = int(len(sorted_weights) * sparsity)
        return sorted_weights[threshold_index].item()


# 使用示例
if __name__ == "__main__":
    # 神经架构搜索示例
    search_space = {
        'backbone_depth': [3, 4, 5, 6],
        'neck_channels': [128, 256, 512],
        'head_layers': [2, 3, 4],
        'activation': ['ReLU', 'SiLU', 'Mish'],
        'attention_type': ['SE', 'CBAM', 'ECA'],
        'use_transformer': [True, False],
        'transformer_layers': [3, 6, 9]
    }
    
    nas = NeuralArchitectureSearch(search_space, population_size=10, generations=20)
    best_architecture = nas.search()
    
    print(f"最佳架构配置:")
    print(f"  骨干深度: {best_architecture.backbone_depth}")
    print(f"  颈部通道: {best_architecture.neck_channels}")
    print(f"  头部层数: {best_architecture.head_layers}")
    print(f"  激活函数: {best_architecture.activation}")
    print(f"  注意力类型: {best_architecture.attention_type}")
    print(f"  使用Transformer: {best_architecture.use_transformer}")