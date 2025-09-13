"""YOLO模型实现模块 - 重构版本
支持统一模型管理和多平台优化
"""

from .yolo_factory import YOLOFactory
from .base_model import BaseYOLOModel
from .yolov5_model import YOLOv5Model
from .yolov8_model import YOLOv8Model
from .yolo_world_model import YOLOWorldModel
from .yolov11_detector import YOLOv11Detector
from .model_converter import ModelConverter
from .unified_model_manager import (
    UnifiedModelManager,
    ModelConfig,
    ModelType,
    PlatformType,
    get_model_manager
)

# 向后兼容
BaseModel = BaseYOLOModel

__all__ = [
    'YOLOFactory',
    'BaseYOLOModel',
    'BaseModel',  # 向后兼容
    'YOLOv5Model', 
    'YOLOv8Model',
    'YOLOv11Detector',
    'YOLOWorldModel',
    'ModelConverter',
    'UnifiedModelManager',
    'ModelConfig',
    'ModelType',
    'PlatformType',
    'get_model_manager'
]