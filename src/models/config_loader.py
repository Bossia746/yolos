#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO模型配置加载器
从配置文件加载模型配置并创建ModelConfig实例
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from .unified_model_manager import ModelConfig, ModelType, PlatformType
from ..core.unified_config import get_config


class YOLOConfigLoader:
    """YOLO配置加载器"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def load_from_unified_config(self, model_name: str = 'default') -> ModelConfig:
        """从统一配置系统加载模型配置"""
        try:
            # 获取模型配置
            model_config = get_config('models', {})
            
            # 提取基础配置
            model_type_str = model_config.get('model_type', 'yolov11')
            model_size = model_config.get('model_size', 's')
            model_path = model_config.get('default_model', f'{model_type_str}{model_size}.pt')
            
            # 设备和性能配置
            device = model_config.get('use_gpu', 'auto')
            if device == 'auto':
                device = 'auto'
            elif device is True:
                device = 'cuda'
            else:
                device = 'cpu'
                
            precision = model_config.get('precision', 'fp16')
            half_precision = precision == 'fp16'
            
            # 推理配置
            inference_config = model_config.get('inference', {})
            confidence_threshold = inference_config.get('confidence_threshold', 0.25)
            iou_threshold = inference_config.get('iou_threshold', 0.45)
            tensorrt_optimize = inference_config.get('tensorrt_optimize', True)
            dynamic_batching = inference_config.get('dynamic_batching', True)
            
            # YOLOv11特定配置
            yolov11_config = model_config.get('yolov11', {})
            c2psa_config = yolov11_config.get('c2psa')
            c3k2_config = yolov11_config.get('c3k2')
            
            # 创建ModelConfig实例
            config = ModelConfig(
                model_type=ModelType(model_type_str.lower()),
                model_size=model_size,
                model_path=model_path,
                device=device,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                half_precision=half_precision,
                tensorrt_optimize=tensorrt_optimize,
                platform=PlatformType.PC,  # 默认PC平台
                dynamic_batching=dynamic_batching,
                c2psa_config=c2psa_config,
                c3k2_config=c3k2_config
            )
            
            self.logger.info(f"成功加载模型配置: {model_type_str}{model_size}")
            return config
            
        except Exception as e:
            self.logger.error(f"加载模型配置失败: {e}")
            # 返回默认配置
            return self._get_default_config()
    
    def load_platform_specific_config(self, platform: Optional[str]) -> ModelConfig:
        """加载平台特定配置"""
        base_config = self.load_from_unified_config()
        
        # 平台特定优化
        platform_configs = {
            'esp32': {
                'model_size': 'n',
                'tensorrt_optimize': False,
                'half_precision': False,
                'dynamic_batching': False,
                'c2psa_config': {'enabled': False},
                'c3k2_config': {'enabled': False},
                'attention_type': 'SimAM'  # 轻量级平台默认使用SimAM
            },
            'k210': {
                'model_size': 'n',
                'tensorrt_optimize': False,  # K210不支持TensorRT
                'half_precision': False,  # 极度内存受限
                'dynamic_batching': False,  # 不支持动态批处理
                'c2psa_config': {'enabled': False},  # 关闭所有高级特性
                'c3k2_config': {'enabled': False},
                'input_size': (320, 320),  # 使用更小的输入尺寸
                'max_detections': 100,  # 减少最大检测数量
                'attention_type': 'SimAM'  # 参数无关注意力机制
            },
            'k230': {
                'model_size': 'n',
                'tensorrt_optimize': True,
                'half_precision': False,  # 内存受限平台不启用半精度
                'dynamic_batching': False,  # 内存受限平台不启用动态批处理
                'c2psa_config': {'enabled': True, 'multi_scale': False},
                'c3k2_config': {'enabled': True, 'parallel_conv': False},
                'attention_type': 'SimAM'  # K230平台使用SimAM优化
            },
            'raspberry_pi': {
                'model_size': 'n',
                'tensorrt_optimize': False,
                'half_precision': False,
                'dynamic_batching': False,
                'c2psa_config': {'enabled': True, 'multi_scale': False},
                'c3k2_config': {'enabled': True, 'parallel_conv': False},
                'attention_type': 'SimAM'  # 树莓派使用SimAM提升性能
            },
            'jetson': {
                'model_size': 'm',
                'tensorrt_optimize': True,
                'half_precision': True,
                'dynamic_batching': True,
                'c2psa_config': {'enabled': True, 'multi_scale': True},
                'c3k2_config': {'enabled': True, 'parallel_conv': True}
            },
            'pc': {
                'model_size': 's',
                'tensorrt_optimize': True,
                'half_precision': True,
                'dynamic_batching': True,
                'c2psa_config': {'enabled': True, 'multi_scale': True},
                'c3k2_config': {'enabled': True, 'parallel_conv': True}
            }
        }
        
        if platform and platform.lower() in platform_configs:
            platform_config = platform_configs[platform.lower()]
            
            # 更新配置
            base_config.model_size = platform_config.get('model_size', base_config.model_size)
            # 根据model_size更新model_path
            base_config.model_path = f'yolov11{base_config.model_size}.pt'
            base_config.tensorrt_optimize = platform_config.get('tensorrt_optimize', base_config.tensorrt_optimize)
            base_config.half_precision = platform_config.get('half_precision', base_config.half_precision)
            base_config.dynamic_batching = platform_config.get('dynamic_batching', base_config.dynamic_batching)
            base_config.c2psa_config = platform_config.get('c2psa_config', base_config.c2psa_config)
            base_config.c3k2_config = platform_config.get('c3k2_config', base_config.c3k2_config)
            base_config.attention_type = platform_config.get('attention_type', base_config.attention_type)
            
            # 应用平台特定的输入尺寸和检测数量限制
            if 'input_size' in platform_config:
                base_config.input_size = platform_config['input_size']
            if 'max_detections' in platform_config:
                base_config.max_detections = platform_config['max_detections']
                
            base_config.platform = PlatformType(platform.lower())
            
            self.logger.info(f"应用平台特定配置: {platform}")
        
        return base_config
    
    def _get_default_config(self) -> ModelConfig:
        """获取默认配置"""
        return ModelConfig(
            model_type=ModelType.YOLOV11,
            model_size='s',
            model_path='yolov11s.pt',
            device='auto',
            confidence_threshold=0.25,
            iou_threshold=0.45,
            half_precision=True,
            tensorrt_optimize=True,
            platform=PlatformType.PC,
            dynamic_batching=True,
            c2psa_config={
                'enabled': True,
                'attention_type': 'pyramid_slice',
                'multi_scale': True
            },
            c3k2_config={
                'enabled': True,
                'parallel_conv': True,
                'channel_separation': True,
                'kernel_sizes': [3, 5, 7]
            },
            attention_type='SimAM'  # 默认使用SimAM注意力机制
        )
    
    def create_model_configs_from_environment(self) -> Dict[str, ModelConfig]:
        """根据当前环境创建多个模型配置"""
        configs = {}
        
        # 基础配置
        base_config = self.load_from_unified_config()
        configs['default'] = base_config
        
        # 不同尺寸的模型配置
        for size in ['n', 's', 'm', 'l']:
            config = ModelConfig(
                model_type=base_config.model_type,
                model_size=size,
                model_path=f'yolov11{size}.pt',
                device=base_config.device,
                confidence_threshold=base_config.confidence_threshold,
                iou_threshold=base_config.iou_threshold,
                half_precision=base_config.half_precision,
                tensorrt_optimize=base_config.tensorrt_optimize,
                platform=base_config.platform,
                dynamic_batching=base_config.dynamic_batching,
                c2psa_config=base_config.c2psa_config,
                c3k2_config=base_config.c3k2_config
            )
            configs[f'yolov11{size}'] = config
        
        # 平台特定配置
        for platform in ['esp32', 'k230', 'raspberry_pi', 'jetson']:
            config = self.load_platform_specific_config(platform)
            configs[f'{platform}_optimized'] = config
        
        self.logger.info(f"创建了 {len(configs)} 个模型配置")
        return configs


# 全局配置加载器实例
config_loader = YOLOConfigLoader()


def get_config_loader() -> YOLOConfigLoader:
    """获取配置加载器实例"""
    return config_loader


def load_yolo_config(model_name: str = 'default') -> ModelConfig:
    """便捷函数：加载YOLO配置"""
    return config_loader.load_from_unified_config(model_name)


def load_platform_config(platform: str) -> ModelConfig:
    """便捷函数：加载平台特定配置"""
    return config_loader.load_platform_specific_config(platform)