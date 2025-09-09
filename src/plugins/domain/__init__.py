"""领域插件模块"""

from .human_recognition import HumanRecognitionPlugin
from .pet_recognition import PetRecognitionPlugin
from .plant_recognition import PlantRecognitionPlugin
from .static_object_recognition import StaticObjectRecognitionPlugin
from .dynamic_object_recognition import DynamicObjectRecognitionPlugin

__all__ = [
    'HumanRecognitionPlugin',
    'PetRecognitionPlugin',
    'PlantRecognitionPlugin',
    'StaticObjectRecognitionPlugin',
    'DynamicObjectRecognitionPlugin'
]