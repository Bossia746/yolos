"""
训练模块
"""

from .trainer import YOLOTrainer
from .dataset_manager import DatasetManager
from .augmentation import AugmentationPipeline

__all__ = [
    'YOLOTrainer',
    'DatasetManager',
    'AugmentationPipeline'
]