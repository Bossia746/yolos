"""插件模块"""

from .domain import *
from .platform import *
from .utility import *

__all__ = [
    # 领域插件
    'HumanRecognitionPlugin',
    'PetRecognitionPlugin', 
    'PlantRecognitionPlugin',
    'StaticObjectPlugin',
    'DynamicObjectPlugin',
    
    # 平台插件
    'ESP32Plugin',
    'RaspberryPiPlugin',
    'ROSPlugin',
    
    # 工具插件
    'DataAugmentationPlugin',
    'ModelOptimizationPlugin',
    'VisualizationPlugin'
]