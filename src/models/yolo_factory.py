"""YOLO模型工厂类 - 重构版本
支持统一模型管理和向后兼容
"""

import torch
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import warnings

from .yolov5_model import YOLOv5Model
from .yolov8_model import YOLOv8Model
from .yolo_world_model import YOLOWorldModel
from .yolov11_detector import YOLOv11Detector
from .unified_model_manager import (
    UnifiedModelManager, ModelConfig, ModelType, PlatformType,
    get_model_manager
)

# 导入YOLOV系列模型（基于论文实现）
try:
    from .yolov_model import YOLOVModel
    YOLOV_AVAILABLE = True
except ImportError:
    YOLOV_AVAILABLE = False
    YOLOVModel = None


class YOLOFactory:
    """YOLO模型工厂类 - 重构版本
    
    提供向后兼容的API，同时集成统一模型管理器
    支持多种YOLO模型的统一创建和管理
    """
    
    # 基础模型注册表
    _base_models = {
        'yolov5': YOLOv5Model,
        'yolov8': YOLOv8Model,
        'yolov11': YOLOv11Detector,
        'yolo-world': YOLOWorldModel,
    }
    
    # 动态添加YOLOV系列模型
    if YOLOV_AVAILABLE:
        _base_models.update({
            'yolov': YOLOVModel,
            'yolov++': YOLOVModel,  # YOLOV++变体
        })
    
    # 自定义模型注册表
    _custom_models = {}
    
    # 统一模型管理器实例
    _manager = None
    
    @classmethod
    def get_all_models(cls) -> Dict[str, Any]:
        """获取所有可用模型"""
        models = cls._base_models.copy()
        models.update(cls._custom_models)
        return models
    
    @classmethod
    def _models(cls):
        """兼容性方法"""
        return cls.get_all_models()
    
    @classmethod
    def get_manager(cls) -> UnifiedModelManager:
        """获取统一模型管理器实例"""
        if cls._manager is None:
            cls._manager = get_model_manager()
        return cls._manager
    
    @classmethod
    def create_model(
        cls,
        model_type: str,
        model_path: Optional[str] = None,
        device: str = 'auto',
        use_unified_manager: bool = True,
        **kwargs
    ) -> Union[Any, 'ModelAdapter']:
        """
        创建YOLO模型实例
        
        Args:
            model_type: 模型类型 ('yolov5', 'yolov8', 'yolov11', 'yolo-world')
            model_path: 模型权重路径
            device: 设备类型 ('cpu', 'cuda', 'auto')
            use_unified_manager: 是否使用统一模型管理器 (推荐)
            **kwargs: 其他参数
            
        Returns:
            YOLO模型实例或ModelAdapter
            
        Raises:
            ValueError: 不支持的模型类型
            RuntimeError: 模型创建失败
        """
        available_models = cls.get_all_models()
        
        if model_type not in available_models:
            available_types = list(available_models.keys())
            raise ValueError(f"不支持的模型类型: {model_type}，可用类型: {available_types}")
        
        try:
            # 使用统一模型管理器 (推荐方式)
            if use_unified_manager:
                return cls._create_with_manager(model_type, model_path, device, **kwargs)
            
            # 传统方式 (向后兼容)
            warnings.warn(
                "传统工厂模式将在未来版本中弃用，请使用 use_unified_manager=True",
                DeprecationWarning,
                stacklevel=2
            )
            
            return cls._create_legacy_model(model_type, model_path, device, **kwargs)
            
        except Exception as e:
            raise RuntimeError(f"创建模型 {model_type} 失败: {str(e)}") from e
    
    @classmethod
    def _create_legacy_model(cls, model_type: str, model_path: Optional[str], device: str, **kwargs):
        """传统方式创建模型"""
        # 自动选择设备
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model_class = cls.get_all_models()[model_type]
        
        # YOLOv11检测器使用不同的初始化参数
        if model_type == 'yolov11':
            model_size = kwargs.get('model_size', 's')
            return model_class(
                model_size=model_size,
                device=device,
                **{k: v for k, v in kwargs.items() if k != 'model_size'}
            )
        else:
            # 为其他模型类型提供默认参数
            init_kwargs = {'device': device}
            if model_path:
                init_kwargs['model_path'] = model_path
            init_kwargs.update(kwargs)
            return model_class(**init_kwargs)
    
    @classmethod
    def _create_with_manager(
        cls,
        model_type: str,
        model_path: Optional[str] = None,
        device: str = 'auto',
        **kwargs
    ):
        """使用统一模型管理器创建模型"""
        # 转换模型类型
        type_mapping = {
            'yolov5': ModelType.YOLOV5,
            'yolov8': ModelType.YOLOV8,
            'yolov11': ModelType.YOLOV11,
            'yolo-world': ModelType.YOLO_WORLD,
            'yolov': ModelType.YOLOV8,  # 使用YOLOv8作为基础
            'yolov++': ModelType.YOLOV8,  # YOLOV++变体
        }
        
        # 创建配置
        config = ModelConfig(
            model_type=type_mapping[model_type],
            model_size=kwargs.get('model_size', 's'),
            model_path=model_path,
            device=device,
            confidence_threshold=kwargs.get('confidence_threshold', 0.25),
            iou_threshold=kwargs.get('iou_threshold', 0.45),
            half_precision=kwargs.get('half_precision', True),
            tensorrt_optimize=kwargs.get('tensorrt_optimize', False),
            platform=PlatformType(kwargs.get('platform', 'pc')),
            max_detections=kwargs.get('max_detections', 1000)
        )
        
        # 注册并返回模型
        manager = cls.get_manager()
        model_name = f"{model_type}_{config.model_size}_{id(config)}"
        
        if manager.register_model(model_name, config):
            manager.switch_model(model_name)
            return manager.models[model_name]
        else:
            raise RuntimeError(f"创建模型失败: {model_type}")
    
    @classmethod
    def register_model(cls, model_type: str, model_class: type, override: bool = False):
        """
        注册自定义模型类型
        
        Args:
            model_type: 模型类型名称
            model_class: 模型类
            override: 是否覆盖已存在的模型类型
            
        Raises:
            ValueError: 模型类型已存在且不允许覆盖
        """
        all_models = cls.get_all_models()
        if model_type in all_models and not override:
            raise ValueError(f"模型类型 {model_type} 已存在，使用 override=True 强制覆盖")
        
        cls._custom_models[model_type] = model_class
    
    @classmethod
    def unregister_model(cls, model_type: str):
        """
        注销自定义模型类型
        
        Args:
            model_type: 模型类型名称
            
        Raises:
            ValueError: 尝试注销基础模型类型
        """
        if model_type in cls._base_models:
            raise ValueError(f"不能注销基础模型类型: {model_type}")
        
        if model_type in cls._custom_models:
            del cls._custom_models[model_type]
    
    @classmethod
    def list_available_models(cls) -> List[str]:
        """列出可用的模型类型"""
        return list(cls.get_all_models().keys())
    
    @classmethod
    def list_base_models(cls) -> List[str]:
        """列出基础模型类型"""
        return list(cls._base_models.keys())
    
    @classmethod
    def list_custom_models(cls) -> List[str]:
        """列出自定义模型类型"""
        return list(cls._custom_models.keys())
    
    @classmethod
    def list_available(cls) -> List[str]:
        """列出可用模型类型（兼容性方法）"""
        return cls.list_available_models()
    
    @classmethod
    def get_available(cls) -> List[str]:
        """获取可用模型类型（兼容性方法）"""
        return cls.list_available_models()
    
    @classmethod
    def list_types(cls) -> List[str]:
        """列出模型类型（兼容性方法）"""
        return cls.list_available_models()
    
    @classmethod
    def get_types(cls) -> List[str]:
        """获取模型类型（兼容性方法）"""
        return cls.list_available_models()
    
    @classmethod
    def create_optimized_model(
        cls,
        model_type: str,
        platform: str = 'pc',
        model_size: str = 's',
        **kwargs
    ):
        """创建平台优化的模型
        
        Args:
            model_type: 模型类型
            platform: 目标平台 ('pc', 'raspberry_pi', 'jetson', 'esp32', 'k230')
            model_size: 模型大小
            **kwargs: 其他参数
        
        Returns:
            优化后的模型适配器
        """
        return cls.create_model(
            model_type=model_type,
            model_size=model_size,
            platform=platform,
            use_unified_manager=True,
            **kwargs
        )
    
    @classmethod
    def get_model_recommendations(cls, platform: str) -> Dict[str, Any]:
        """获取平台推荐配置
        
        Args:
            platform: 目标平台
            
        Returns:
            推荐配置字典
        """
        recommendations = {
            'pc': {
                'model_type': 'yolov' if YOLOV_AVAILABLE else 'yolov11',
                'model_size': 's',
                'tensorrt_optimize': True,
                'half_precision': True,
                'enable_feature_aggregation': True
            },
            'raspberry_pi': {
                'model_type': 'yolov' if YOLOV_AVAILABLE else 'yolov8',
                'model_size': 'n',
                'tensorrt_optimize': False,
                'half_precision': False,
                'enable_feature_aggregation': True,
                'aggregation_buffer_size': 3
            },
            'jetson': {
                'model_type': 'yolov++' if YOLOV_AVAILABLE else 'yolov11',
                'model_size': 's',
                'tensorrt_optimize': True,
                'half_precision': True,
                'enable_feature_aggregation': True
            },
            'esp32': {
                'model_type': 'yolov8',  # ESP32使用轻量级模型
                'model_size': 'n',
                'tensorrt_optimize': False,
                'half_precision': False,
                'input_size': (320, 320),
                'enable_feature_aggregation': True,
                'aggregation_buffer_size': 2
            },
            'k230': {
                'model_type': 'yolov' if YOLOV_AVAILABLE else 'yolov8',
                'model_size': 's',
                'device': 'npu',
                'enable_feature_aggregation': True
            }
        }
        
        return recommendations.get(platform, recommendations['pc'])
    
    @classmethod
    def get_model_info(cls, model_type: str) -> Dict[str, Any]:
        """获取模型信息"""
        if model_type not in cls._models:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        model_class = cls._models[model_type]
        info = {
            'name': model_type,
            'class': model_class.__name__,
            'description': model_class.__doc__ or "无描述",
            'supported_formats': getattr(model_class, 'SUPPORTED_FORMATS', []),
        }
        
        # 添加YOLOV系列特殊信息
        if model_type in ['yolov', 'yolov++']:
            info.update({
                'features': [
                    '特征聚合策略',
                    '视频目标检测优化',
                    '时序信息利用',
                    '多平台适配'
                ],
                'paper_reference': 'YOLOV: Making Still Image Object Detectors Great at Video Object Detection',
                'optimized_for': 'video_detection'
            })
        
        return info
    
    @classmethod
    def create_video_optimized_model(
        cls,
        model_type: str = None,
        platform: str = 'pc',
        enable_tracking: bool = True,
        **kwargs
    ):
        """创建视频检测优化的模型
        
        Args:
            model_type: 模型类型，None时自动选择最佳模型
            platform: 目标平台
            enable_tracking: 是否启用目标跟踪
            **kwargs: 其他参数
        
        Returns:
            优化后的视频检测模型
        """
        # 自动选择最佳模型
        if model_type is None:
            if YOLOV_AVAILABLE:
                model_type = 'yolov++' if platform in ['pc', 'jetson'] else 'yolov'
            else:
                model_type = 'yolov11' if platform in ['pc', 'jetson'] else 'yolov8'
        
        # 获取推荐配置
        config = cls.get_model_recommendations(platform)
        config.update(kwargs)
        
        # 视频检测特定配置
        video_config = {
            'enable_feature_aggregation': True,
            'enable_temporal_aggregation': True,
            'enable_tracking': enable_tracking,
            'confidence_threshold': 0.4,  # 视频检测可以稍微降低阈值
            'iou_threshold': 0.5
        }
        config.update(video_config)
        
        return cls.create_model(
            model_type=model_type,
            use_unified_manager=True,
            **config
        )
    
    @classmethod
    def get_yolov_info(cls) -> Dict[str, Any]:
        """获取YOLOV系列模型信息"""
        return {
            'available': YOLOV_AVAILABLE,
            'models': ['yolov', 'yolov++'] if YOLOV_AVAILABLE else [],
            'features': {
                'feature_aggregation': '特征聚合策略，提升检测稳定性',
                'temporal_consistency': '时序一致性，减少检测抖动',
                'video_optimization': '专为视频检测优化',
                'multi_platform': '支持多平台部署'
            },
            'paper': {
                'title': 'YOLOV: Making Still Image Object Detectors Great at Video Object Detection',
                'authors': 'YuHengsss et al.',
                'year': 2023,
                'arxiv': 'https://arxiv.org/abs/2208.09686'
            }
        }