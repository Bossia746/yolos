"""平台适配模块
提供不同硬件平台的适配实现，包括ESP32、K230、树莓派等
"""

from .esp32_platform import ESP32Platform, create_esp32_platform
from .k230_platform import K230Platform, create_k230_platform
from .raspberry_pi_platform import RaspberryPiPlatform, create_raspberry_pi_platform

__all__ = [
    'ESP32Platform',
    'create_esp32_platform',
    'K230Platform', 
    'create_k230_platform',
    'RaspberryPiPlatform',
    'create_raspberry_pi_platform'
]