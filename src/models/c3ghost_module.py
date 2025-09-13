"""C3Ghost模块实现

基于YOLO-APD架构的C3Ghost (C3 with Ghost Bottleneck) 轻量化卷积模块
结合Ghost卷积和C3结构，专为嵌入式设备优化

Author: YOLOS Team
Date: 2024-01-15
Version: 1.0.0

Reference:
- GhostNet: More Features from Cheap Operations
- YOLOv5 C3 Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import yaml
import math
from pathlib import Path
from .enhanced_mish_activation import EnhancedMish, MishVariants

try:
    from ..core.logger import get_logger
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


class GhostConv(nn.Module):
    """Ghost卷积模块
    
    通过廉价操作生成更多特征图，减少计算量和参数数量
    """
    
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, g: int = 1, 
                 act: bool = True, ratio: int = 2, dw_size: int = 3):
        """初始化Ghost卷积
        
        Args:
            c1: 输入通道数
            c2: 输出通道数
            k: 卷积核大小
            s: 步长
            g: 分组数
            act: 是否使用激活函数
            ratio: Ghost比例
            dw_size: 深度卷积核大小
        """
        super(GhostConv, self).__init__()
        c_ = c2 // ratio  # 主要特征图通道数
        
        # 主要卷积：生成主要特征图
        self.primary_conv = nn.Sequential(
            nn.Conv2d(c1, c_, k, s, k // 2, groups=g, bias=False),
            nn.BatchNorm2d(c_),
            MishVariants.fast_mish() if act else nn.Identity()
        )
        
        # 廉价操作：生成Ghost特征图
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(c_, c_, dw_size, 1, dw_size // 2, groups=c_, bias=False),
            nn.BatchNorm2d(c_),
            MishVariants.fast_mish() if act else nn.Identity()
        )
        
        # 如果拼接后的通道数与目标不匹配，添加通道调整层
        concat_channels = c_ * 2
        if concat_channels != c2:
            self.channel_adjust = nn.Conv2d(concat_channels, c2, kernel_size=1, bias=False)
        
        self.ratio = ratio
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            torch.Tensor: 输出张量
        """
        # 生成主要特征图
        primary = self.primary_conv(x)
        
        # 生成Ghost特征图
        ghost = self.cheap_operation(primary)
        
        # 拼接主要特征图和Ghost特征图
        out = torch.cat([primary, ghost], dim=1)
        
        # 如果输出通道数不匹配，使用1x1卷积调整
        if hasattr(self, 'channel_adjust'):
            out = self.channel_adjust(out)
            
        return out


class GhostBottleneck(nn.Module):
    """Ghost瓶颈模块
    
    结合Ghost卷积和残差连接的瓶颈结构
    """
    
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, 
                 e: float = 0.5, use_se: bool = False, act: bool = True):
        """初始化Ghost瓶颈
        
        Args:
            c1: 输入通道数
            c2: 输出通道数
            k: 卷积核大小
            s: 步长
            e: 扩展比例
            use_se: 是否使用SE注意力
            act: 是否使用激活函数
        """
        super(GhostBottleneck, self).__init__()
        c_ = int(c2 * e)  # 隐藏层通道数
        
        # 第一个Ghost卷积：扩展
        self.ghost1 = GhostConv(c1, c_, 1, 1, act=act)
        
        # 深度卷积（如果步长>1）
        if s > 1:
            self.dw_conv = nn.Conv2d(c_, c_, k, s, k // 2, groups=c_, bias=False)
            self.bn_dw = nn.BatchNorm2d(c_)
        else:
            self.dw_conv = None
            
        # SE注意力模块
        if use_se:
            self.se = SEModule(c_)
        else:
            self.se = None
            
        # 第二个Ghost卷积：压缩
        self.ghost2 = GhostConv(c_, c2, 1, 1, act=False)
        
        # 残差连接
        self.shortcut = nn.Sequential() if (c1 == c2 and s == 1) else nn.Sequential(
            nn.Conv2d(c1, c2, 1, s, bias=False),
            nn.BatchNorm2d(c2)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        residual = x
        
        # 第一个Ghost卷积
        x = self.ghost1(x)
        
        # 深度卷积
        if self.dw_conv is not None:
            x = self.bn_dw(self.dw_conv(x))
            
        # SE注意力
        if self.se is not None:
            x = self.se(x)
            
        # 第二个Ghost卷积
        x = self.ghost2(x)
        
        # 残差连接
        return x + self.shortcut(residual)


class SEModule(nn.Module):
    """Squeeze-and-Excitation模块"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class C3Ghost(nn.Module):
    """C3Ghost模块
    
    结合C3结构和Ghost卷积的轻量化模块
    """
    
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, 
                 g: int = 1, e: float = 0.5, ghost_ratio: int = 2, 
                 use_se: bool = False, act: str = 'silu'):
        """初始化C3Ghost模块
        
        Args:
            c1: 输入通道数
            c2: 输出通道数
            n: 瓶颈数量
            shortcut: 是否使用残差连接
            g: 分组数
            e: 扩展比例
            ghost_ratio: Ghost比例
            use_se: 是否使用SE注意力
            act: 激活函数类型
        """
        super(C3Ghost, self).__init__()
        c_ = int(c2 * e)  # 隐藏层通道数
        
        # 激活函数选择
        self.act_fn = self._get_activation(act)
        
        # 输入卷积
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c_, 1, 1, bias=False),
            nn.BatchNorm2d(c_),
            self.act_fn
        )
        
        # 分支卷积
        self.cv2 = nn.Sequential(
            nn.Conv2d(c1, c_, 1, 1, bias=False),
            nn.BatchNorm2d(c_),
            self.act_fn
        )
        
        # Ghost瓶颈层
        self.m = nn.Sequential(*[
            GhostBottleneck(c_, c_, e=1.0, use_se=use_se, act=(act != 'identity'))
            for _ in range(n)
        ])
        
        # 输出卷积
        self.cv3 = nn.Sequential(
            nn.Conv2d(2 * c_, c2, 1, 1, bias=False),
            nn.BatchNorm2d(c2),
            self.act_fn
        )
        
        self.shortcut = shortcut and c1 == c2
        
    def _get_activation(self, act: str) -> nn.Module:
        """获取激活函数"""
        activations = {
            'relu': MishVariants.fast_mish(inplace=True),
            'silu': MishVariants.standard_mish(inplace=True),
            'swish': MishVariants.standard_mish(inplace=True),
            'mish': MishVariants.adaptive_mish(learnable=True),
            'enhanced_mish': MishVariants.adaptive_mish(learnable=True),
            'fast_mish': MishVariants.fast_mish(inplace=True),
            'identity': nn.Identity()
        }
        return activations.get(act.lower(), MishVariants.standard_mish(inplace=True))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 主分支
        y1 = self.m(self.cv1(x))
        
        # 辅助分支
        y2 = self.cv2(x)
        
        # 拼接并输出
        out = self.cv3(torch.cat([y1, y2], dim=1))
        
        # 残差连接
        return x + out if self.shortcut else out


class C3GhostEmbedded(nn.Module):
    """嵌入式优化的C3Ghost模块
    
    针对嵌入式设备进行特殊优化的版本
    """
    
    def __init__(self, c1: int, c2: int, platform: str = 'auto', 
                 config_path: Optional[str] = None):
        """初始化嵌入式C3Ghost模块
        
        Args:
            c1: 输入通道数
            c2: 输出通道数
            platform: 目标平台
            config_path: 配置文件路径
        """
        super(C3GhostEmbedded, self).__init__()
        
        # 加载配置
        self.config = self._load_config(config_path, platform)
        self.platform = platform
        
        # 根据平台配置调整参数
        platform_config = self.config.get('platform_configs', {}).get(platform, {})
        c3ghost_config = platform_config.get('c3ghost_config', {})
        
        # 应用通道缩减
        channels_reduction = c3ghost_config.get('channels_reduction', 0.0)
        if channels_reduction > 0:
            c2 = max(1, int(c2 * (1 - channels_reduction)))
            
        # 应用深度和宽度乘数
        depth_multiplier = c3ghost_config.get('depth_multiplier', 1.0)
        width_multiplier = c3ghost_config.get('width_multiplier', 1.0)
        
        n = max(1, int(1 * depth_multiplier))  # 瓶颈数量
        c2 = max(1, int(c2 * width_multiplier))  # 输出通道数
        
        # 其他配置参数
        ghost_ratio = c3ghost_config.get('ghost_ratio', 2)
        use_se = c3ghost_config.get('use_se', False)
        activation = c3ghost_config.get('activation', 'enhanced_mish')
        
        # 创建C3Ghost模块
        self.c3ghost = C3Ghost(
            c1=c1,
            c2=c2,
            n=n,
            ghost_ratio=ghost_ratio,
            use_se=use_se,
            act=activation
        )
        
        # 性能监控
        self.performance_monitor = PerformanceMonitor(platform_config)
        
        logger.info(f"创建嵌入式C3Ghost模块: {platform}, 输入通道: {c1}, 输出通道: {c2}")
        
    def _load_config(self, config_path: Optional[str], platform: str) -> Dict:
        """加载配置文件"""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "c3ghost_embedded_config.yaml"
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"成功加载C3Ghost配置: {config_path}")
            return config
        except Exception as e:
            logger.warning(f"加载C3Ghost配置失败: {e}，使用默认配置")
            return self._get_default_config()
            
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'platform_configs': {
                'auto': {
                    'c3ghost_config': {
                        'ghost_ratio': 2,
                        'channels_reduction': 0.0,
                        'depth_multiplier': 1.0,
                        'width_multiplier': 1.0,
                        'use_se': False,
                        'activation': 'enhanced_mish'
                    }
                }
            }
        }
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 性能监控开始
        self.performance_monitor.start_inference()
        
        # 前向传播
        output = self.c3ghost(x)
        
        # 性能监控结束
        self.performance_monitor.end_inference(x.shape, output.shape)
        
        return output
        
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'platform': self.platform,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024,  # 假设float32
            'performance_stats': self.performance_monitor.get_stats()
        }


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.inference_times = []
        self.memory_usage = []
        self.start_time = None
        
    def start_inference(self):
        """开始推理计时"""
        import time
        self.start_time = time.time()
        
    def end_inference(self, input_shape: torch.Size, output_shape: torch.Size):
        """结束推理计时"""
        if self.start_time is not None:
            import time
            inference_time = (time.time() - self.start_time) * 1000  # 转换为毫秒
            self.inference_times.append(inference_time)
            
            # 估算内存使用
            input_memory = torch.tensor(input_shape).prod().item() * 4 / 1024 / 1024  # MB
            output_memory = torch.tensor(output_shape).prod().item() * 4 / 1024 / 1024  # MB
            total_memory = input_memory + output_memory
            self.memory_usage.append(total_memory)
            
    def get_stats(self) -> Dict:
        """获取统计信息"""
        if not self.inference_times:
            return {}
            
        import statistics
        return {
            'avg_inference_time_ms': statistics.mean(self.inference_times),
            'max_inference_time_ms': max(self.inference_times),
            'min_inference_time_ms': min(self.inference_times),
            'avg_memory_usage_mb': statistics.mean(self.memory_usage),
            'total_inferences': len(self.inference_times)
        }


def create_c3ghost_for_platform(c1: int, c2: int, platform: str = 'auto') -> C3GhostEmbedded:
    """为指定平台创建C3Ghost模块
    
    Args:
        c1: 输入通道数
        c2: 输出通道数
        platform: 目标平台
        
    Returns:
        C3GhostEmbedded: 嵌入式优化的C3Ghost模块
    """
    return C3GhostEmbedded(c1, c2, platform)


if __name__ == "__main__":
    # 测试代码
    import torch
    
    # 测试不同平台的C3Ghost模块
    platforms = ['raspberry_pi_4b', 'jetson_nano', 'intel_nuc']
    
    for platform in platforms:
        print(f"\n测试平台: {platform}")
        
        # 创建模块
        module = create_c3ghost_for_platform(256, 512, platform)
        
        # 测试输入
        x = torch.randn(1, 256, 32, 32)
        
        # 前向传播
        with torch.no_grad():
            output = module(x)
            
        # 输出信息
        model_info = module.get_model_info()
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")
        print(f"模型参数: {model_info['total_parameters']:,}")
        print(f"模型大小: {model_info['model_size_mb']:.2f} MB")
        
        if model_info['performance_stats']:
            stats = model_info['performance_stats']
            print(f"平均推理时间: {stats['avg_inference_time_ms']:.2f} ms")
            print(f"平均内存使用: {stats['avg_memory_usage_mb']:.2f} MB")