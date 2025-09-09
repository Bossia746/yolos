"""
YOLO模型工厂类
"""

import torch
from typing import Dict, Any, Optional, List
from pathlib import Path

from .yolov5_model import YOLOv5Model
from .yolov8_model import YOLOv8Model
from .yolo_world_model import YOLOWorldModel
from .yolov11_detector import YOLOv11Detector


class YOLOFactory:
    """YOLO模型工厂类，用于创建不同版本的YOLO模型"""
    
    _models = {
        'yolov5': YOLOv5Model,
        'yolov8': YOLOv8Model,
        'yolov11': YOLOv11Detector,
        'yolo-world': YOLOWorldModel,
    }
    
    @classmethod
    def create_model(
        cls,
        model_type: str,
        model_path: Optional[str] = None,
        device: str = 'auto',
        **kwargs
    ):
        """
        创建YOLO模型实例
        
        Args:
            model_type: 模型类型 ('yolov5', 'yolov8', 'yolov11', 'yolo-world')
            model_path: 模型权重路径
            device: 设备类型 ('cpu', 'cuda', 'auto')
            **kwargs: 其他参数
            
        Returns:
            YOLO模型实例
        """
        if model_type not in cls._models:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 自动选择设备
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model_class = cls._models[model_type]
        
        # YOLOv11检测器使用不同的初始化参数
        if model_type == 'yolov11':
            model_size = kwargs.get('model_size', 's')
            return model_class(
                model_size=model_size,
                device=device,
                **{k: v for k, v in kwargs.items() if k != 'model_size'}
            )
        else:
            return model_class(model_path=model_path, device=device, **kwargs)
    
    @classmethod
    def list_available_models(cls) -> List[str]:
        """获取可用的模型类型列表"""
        return list(cls._models.keys())
    
    @classmethod
    def get_model_info(cls, model_type: str) -> Dict[str, Any]:
        """获取模型信息"""
        if model_type not in cls._models:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        model_class = cls._models[model_type]
        return {
            'name': model_type,
            'class': model_class.__name__,
            'description': model_class.__doc__ or "无描述",
            'supported_formats': getattr(model_class, 'SUPPORTED_FORMATS', []),
        }