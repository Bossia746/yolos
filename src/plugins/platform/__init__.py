"""平台插件模块

提供各种平台的适配插件，包括：
- ESP32平台插件
- 树莓派平台插件
- ROS平台插件
- Windows平台插件
- Linux平台插件
- macOS平台插件
"""

from .esp32_plugin import ESP32Plugin
from .raspberry_pi_plugin import RaspberryPiPlugin
from .ros_integration import ROSHybridRecognitionNode, ROS1HybridRecognitionNode, ROS2HybridRecognitionNode, create_ros_node
from .aiot_boards_adapter import AIoTBoardsAdapter, get_aiot_boards_adapter

__all__ = [
    'ESP32Plugin',
    'RaspberryPiPlugin', 
    'ROSHybridRecognitionNode',
    'ROS1HybridRecognitionNode',
    'ROS2HybridRecognitionNode',
    'create_ros_node',
    'AIoTBoardsAdapter',
    'get_aiot_boards_adapter'
]