"""领域插件模块"""

from .human_recognition import HumanRecognitionPlugin
from .pet_recognition import PetRecognitionPlugin
from .plant_recognition import PlantRecognitionPlugin
from .static_object import StaticObjectPlugin
from .dynamic_object import DynamicObjectPlugin

__all__ = [
    'HumanRecognitionPlugin',
    'PetRecognitionPlugin',
    'PlantRecognitionPlugin', 
    'StaticObjectPlugin',
    'DynamicObjectPlugin'
]