"""硬件抽象层 (HAL) - 统一硬件接口
提供跨平台的硬件抽象，支持ESP32、K230、树莓派等多种平台
"""

import os
import sys
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from pathlib import Path


class PlatformType(Enum):
    """支持的平台类型"""
    PC_WINDOWS = "pc_windows"
    PC_LINUX = "pc_linux"
    PC_MACOS = "pc_macos"
    RASPBERRY_PI = "raspberry_pi"
    ESP32 = "esp32"
    ESP32_S3 = "esp32_s3"
    ESP32_CAM = "esp32_cam"
    K230 = "k230"
    JETSON_NANO = "jetson_nano"
    JETSON_XAVIER = "jetson_xavier"
    JETSON_ORIN = "jetson_orin"
    ARDUINO = "arduino"
    STM32 = "stm32"
    UNKNOWN = "unknown"


class ComputeType(Enum):
    """计算设备类型"""
    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"  # Neural Processing Unit (如K230)
    TPU = "tpu"  # Tensor Processing Unit
    VPU = "vpu"  # Vision Processing Unit
    DSP = "dsp"  # Digital Signal Processor
    FPGA = "fpga"
    ASIC = "asic"


class CameraType(Enum):
    """摄像头类型"""
    USB = "usb"
    CSI = "csi"  # Camera Serial Interface
    MIPI = "mipi"  # Mobile Industry Processor Interface
    IP = "ip"  # IP摄像头
    VIRTUAL = "virtual"  # 虚拟摄像头


class ConnectivityType(Enum):
    """连接类型"""
    WIFI = "wifi"
    ETHERNET = "ethernet"
    BLUETOOTH = "bluetooth"
    ZIGBEE = "zigbee"
    LORA = "lora"
    CELLULAR = "cellular"
    USB = "usb"
    UART = "uart"
    I2C = "i2c"
    SPI = "spi"
    GPIO = "gpio"


@dataclass
class HardwareCapabilities:
    """硬件能力描述"""
    compute_types: List[ComputeType] = field(default_factory=list)
    camera_types: List[CameraType] = field(default_factory=list)
    connectivity_types: List[ConnectivityType] = field(default_factory=list)
    max_memory_mb: int = 0
    max_storage_mb: int = 0
    gpio_pins: int = 0
    max_fps: int = 30
    max_resolution: Tuple[int, int] = (640, 480)
    power_consumption_mw: int = 0  # 毫瓦
    operating_temp_range: Tuple[int, int] = (-40, 85)  # 摄氏度
    supports_real_time: bool = False
    supports_offline: bool = True


@dataclass
class PlatformConstraints:
    """平台约束条件"""
    max_model_size_mb: int = 100
    max_input_resolution: Tuple[int, int] = (640, 640)
    max_batch_size: int = 1
    preferred_precision: str = "fp32"  # fp32, fp16, int8
    max_inference_time_ms: int = 1000
    memory_limit_mb: int = 512
    power_limit_mw: int = 5000
    requires_quantization: bool = False
    requires_pruning: bool = False


class CameraInterface(ABC):
    """摄像头抽象接口"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """初始化摄像头"""
        pass
    
    @abstractmethod
    def capture_frame(self) -> Optional[np.ndarray]:
        """捕获一帧图像"""
        pass
    
    @abstractmethod
    def set_resolution(self, width: int, height: int) -> bool:
        """设置分辨率"""
        pass
    
    @abstractmethod
    def set_fps(self, fps: int) -> bool:
        """设置帧率"""
        pass
    
    @abstractmethod
    def get_supported_resolutions(self) -> List[Tuple[int, int]]:
        """获取支持的分辨率列表"""
        pass
    
    @abstractmethod
    def release(self) -> None:
        """释放摄像头资源"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查摄像头是否可用"""
        pass


class ComputeInterface(ABC):
    """计算设备抽象接口"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """初始化计算设备"""
        pass
    
    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        pass
    
    @abstractmethod
    def load_model(self, model_path: str, **kwargs) -> bool:
        """加载模型"""
        pass
    
    @abstractmethod
    def inference(self, input_data: np.ndarray) -> np.ndarray:
        """执行推理"""
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> Dict[str, int]:
        """获取内存使用情况"""
        pass
    
    @abstractmethod
    def optimize_model(self, model_path: str, optimization_config: Dict[str, Any]) -> str:
        """优化模型"""
        pass
    
    @abstractmethod
    def release(self) -> None:
        """释放计算资源"""
        pass


class ConnectivityInterface(ABC):
    """连接接口"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """初始化连接"""
        pass
    
    @abstractmethod
    def connect(self, **kwargs) -> bool:
        """建立连接"""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """断开连接"""
        pass
    
    @abstractmethod
    def send_data(self, data: bytes) -> bool:
        """发送数据"""
        pass
    
    @abstractmethod
    def receive_data(self, timeout: float = 1.0) -> Optional[bytes]:
        """接收数据"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """检查连接状态"""
        pass


class StorageInterface(ABC):
    """存储接口"""
    
    @abstractmethod
    def save_model(self, model_data: bytes, model_name: str) -> bool:
        """保存模型"""
        pass
    
    @abstractmethod
    def load_model(self, model_name: str) -> Optional[bytes]:
        """加载模型"""
        pass
    
    @abstractmethod
    def save_config(self, config: Dict[str, Any], config_name: str) -> bool:
        """保存配置"""
        pass
    
    @abstractmethod
    def load_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """加载配置"""
        pass
    
    @abstractmethod
    def get_available_space(self) -> int:
        """获取可用存储空间(MB)"""
        pass
    
    @abstractmethod
    def cleanup_old_files(self, max_age_days: int = 7) -> None:
        """清理旧文件"""
        pass


class PowerInterface(ABC):
    """电源管理接口"""
    
    @abstractmethod
    def get_power_status(self) -> Dict[str, Any]:
        """获取电源状态"""
        pass
    
    @abstractmethod
    def set_power_mode(self, mode: str) -> bool:
        """设置电源模式 (performance, balanced, power_save)"""
        pass
    
    @abstractmethod
    def get_battery_level(self) -> Optional[float]:
        """获取电池电量百分比"""
        pass
    
    @abstractmethod
    def enable_sleep_mode(self, duration_seconds: int) -> None:
        """启用睡眠模式"""
        pass


class HardwareAbstractionLayer:
    """硬件抽象层主类"""
    
    def __init__(self, platform_type: PlatformType):
        self.platform_type = platform_type
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 硬件接口实例
        self.camera: Optional[CameraInterface] = None
        self.compute: Optional[ComputeInterface] = None
        self.connectivity: Optional[ConnectivityInterface] = None
        self.storage: Optional[StorageInterface] = None
        self.power: Optional[PowerInterface] = None
        
        # 平台信息
        self.capabilities: Optional[HardwareCapabilities] = None
        self.constraints: Optional[PlatformConstraints] = None
        
    def initialize(self) -> bool:
        """初始化硬件抽象层"""
        try:
            self.logger.info(f"初始化硬件抽象层: {self.platform_type.value}")
            
            # 检测硬件能力
            self.capabilities = self._detect_capabilities()
            self.constraints = self._get_platform_constraints()
            
            # 初始化各个接口
            success = True
            if self.camera:
                success &= self.camera.initialize()
            if self.compute:
                success &= self.compute.initialize()
            if self.connectivity:
                success &= self.connectivity.initialize()
            
            self.logger.info(f"硬件抽象层初始化{'成功' if success else '失败'}")
            return success
            
        except Exception as e:
            self.logger.error(f"硬件抽象层初始化失败: {e}")
            return False
    
    def _detect_capabilities(self) -> HardwareCapabilities:
        """检测硬件能力"""
        # 根据平台类型返回相应的硬件能力
        platform_capabilities = {
            PlatformType.ESP32: HardwareCapabilities(
                compute_types=[ComputeType.CPU],
                camera_types=[CameraType.USB, CameraType.CSI],
                connectivity_types=[ConnectivityType.WIFI, ConnectivityType.BLUETOOTH],
                max_memory_mb=520,  # 512KB SRAM + 8MB PSRAM
                max_storage_mb=16,
                gpio_pins=34,
                max_fps=10,
                max_resolution=(640, 480),
                power_consumption_mw=500,
                supports_real_time=False,
                supports_offline=True
            ),
            PlatformType.K230: HardwareCapabilities(
                compute_types=[ComputeType.CPU, ComputeType.NPU],
                camera_types=[CameraType.MIPI, CameraType.USB],
                connectivity_types=[ConnectivityType.ETHERNET, ConnectivityType.USB],
                max_memory_mb=512,
                max_storage_mb=8192,
                gpio_pins=40,
                max_fps=30,
                max_resolution=(1920, 1080),
                power_consumption_mw=2000,
                supports_real_time=True,
                supports_offline=True
            ),
            PlatformType.RASPBERRY_PI: HardwareCapabilities(
                compute_types=[ComputeType.CPU, ComputeType.GPU],
                camera_types=[CameraType.CSI, CameraType.USB],
                connectivity_types=[ConnectivityType.WIFI, ConnectivityType.ETHERNET, ConnectivityType.BLUETOOTH],
                max_memory_mb=8192,  # 树莓派4B 8GB版本
                max_storage_mb=32768,  # 32GB SD卡
                gpio_pins=40,
                max_fps=30,
                max_resolution=(1920, 1080),
                power_consumption_mw=3000,
                supports_real_time=True,
                supports_offline=True
            )
        }
        
        return platform_capabilities.get(self.platform_type, HardwareCapabilities())
    
    def _get_platform_constraints(self) -> PlatformConstraints:
        """获取平台约束条件"""
        platform_constraints = {
            PlatformType.ESP32: PlatformConstraints(
                max_model_size_mb=4,
                max_input_resolution=(320, 320),
                max_batch_size=1,
                preferred_precision="int8",
                max_inference_time_ms=2000,
                memory_limit_mb=32,
                power_limit_mw=500,
                requires_quantization=True,
                requires_pruning=True
            ),
            PlatformType.K230: PlatformConstraints(
                max_model_size_mb=50,
                max_input_resolution=(640, 640),
                max_batch_size=4,
                preferred_precision="fp16",
                max_inference_time_ms=100,
                memory_limit_mb=256,
                power_limit_mw=2000,
                requires_quantization=False,
                requires_pruning=False
            ),
            PlatformType.RASPBERRY_PI: PlatformConstraints(
                max_model_size_mb=100,
                max_input_resolution=(640, 640),
                max_batch_size=2,
                preferred_precision="fp32",
                max_inference_time_ms=500,
                memory_limit_mb=1024,
                power_limit_mw=3000,
                requires_quantization=False,
                requires_pruning=False
            )
        }
        
        return platform_constraints.get(self.platform_type, PlatformConstraints())
    
    def get_optimal_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """根据平台约束获取最优配置"""
        if not self.constraints:
            return base_config
        
        optimized_config = base_config.copy()
        
        # 调整模型大小
        if 'model_size' in optimized_config:
            if self.platform_type in [PlatformType.ESP32, PlatformType.ESP32_CAM]:
                optimized_config['model_size'] = 'n'  # nano
            elif self.platform_type == PlatformType.K230:
                optimized_config['model_size'] = 's'  # small
        
        # 调整输入分辨率
        if 'input_size' in optimized_config:
            max_res = self.constraints.max_input_resolution
            current_res = optimized_config['input_size']
            if isinstance(current_res, (list, tuple)) and len(current_res) == 2:
                optimized_config['input_size'] = (
                    min(current_res[0], max_res[0]),
                    min(current_res[1], max_res[1])
                )
        
        # 调整精度
        optimized_config['precision'] = self.constraints.preferred_precision
        
        # 调整批处理大小
        optimized_config['batch_size'] = min(
            optimized_config.get('batch_size', 1),
            self.constraints.max_batch_size
        )
        
        return optimized_config
    
    def is_feature_supported(self, feature: str) -> bool:
        """检查是否支持某个特性"""
        if not self.capabilities:
            return False
        
        feature_map = {
            'real_time': self.capabilities.supports_real_time,
            'offline': self.capabilities.supports_offline,
            'gpu': ComputeType.GPU in self.capabilities.compute_types,
            'npu': ComputeType.NPU in self.capabilities.compute_types,
            'camera': len(self.capabilities.camera_types) > 0,
            'wifi': ConnectivityType.WIFI in self.capabilities.connectivity_types,
            'gpio': self.capabilities.gpio_pins > 0
        }
        
        return feature_map.get(feature, False)
    
    def get_platform_info(self) -> Dict[str, Any]:
        """获取平台信息"""
        return {
            'platform_type': self.platform_type.value,
            'capabilities': self.capabilities.__dict__ if self.capabilities else {},
            'constraints': self.constraints.__dict__ if self.constraints else {},
            'interfaces': {
                'camera': self.camera is not None,
                'compute': self.compute is not None,
                'connectivity': self.connectivity is not None,
                'storage': self.storage is not None,
                'power': self.power is not None
            }
        }
    
    def cleanup(self) -> None:
        """清理资源"""
        try:
            if self.camera:
                self.camera.release()
            if self.compute:
                self.compute.release()
            if self.connectivity:
                self.connectivity.disconnect()
            
            self.logger.info("硬件抽象层资源清理完成")
            
        except Exception as e:
            self.logger.error(f"硬件抽象层资源清理失败: {e}")


def create_hardware_abstraction_layer(platform_type: PlatformType) -> HardwareAbstractionLayer:
    """创建硬件抽象层实例"""
    return HardwareAbstractionLayer(platform_type)


def detect_platform() -> PlatformType:
    """自动检测当前平台类型"""
    import platform as plt
    
    system = plt.system().lower()
    machine = plt.machine().lower()
    
    # 检测树莓派
    if os.path.exists('/proc/device-tree/model'):
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().lower()
                if 'raspberry pi' in model:
                    return PlatformType.RASPBERRY_PI
        except:
            pass
    
    # 检测ESP32
    if 'esp32' in machine or os.path.exists('/dev/ttyUSB0'):
        return PlatformType.ESP32
    
    # 检测K230
    if 'k230' in machine or os.path.exists('/proc/k230'):
        return PlatformType.K230
    
    # 检测Jetson
    if os.path.exists('/etc/nv_tegra_release'):
        return PlatformType.JETSON_NANO
    
    # 检测PC平台
    if system == 'windows':
        return PlatformType.PC_WINDOWS
    elif system == 'linux':
        return PlatformType.PC_LINUX
    elif system == 'darwin':
        return PlatformType.PC_MACOS
    
    return PlatformType.UNKNOWN