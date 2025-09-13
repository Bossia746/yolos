"""平台适配器模块
提供不同硬件平台的统一接口和适配功能，集成硬件抽象层
"""

import os
import sys
import platform
import psutil
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
import time
from pathlib import Path

from .event_bus import EventBus

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入硬件抽象层
try:
    from src.core.hardware_abstraction import (
        HardwareAbstractionLayer, ComputeType,
        CameraType, ConnectivityType
    )
except ImportError as e:
    logging.warning(f"硬件抽象层导入失败: {e}")
    HardwareAbstractionLayer = None

# 导入平台实现
try:
    from src.platforms import (
        ESP32Platform, create_esp32_platform,
        K230Platform, create_k230_platform,
        RaspberryPiPlatform, create_raspberry_pi_platform
    )
except ImportError as e:
    logging.warning(f"部分平台实现导入失败: {e}")
    ESP32Platform = None
    K230Platform = None
    RaspberryPiPlatform = None


class PlatformType(Enum):
    """平台类型"""
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    RASPBERRY_PI = "raspberry_pi"
    ESP32 = "esp32"
    JETSON = "jetson"
    UNKNOWN = "unknown"


class DeviceCapability(Enum):
    """设备能力"""
    CAMERA = "camera"
    GPU = "gpu"
    TPU = "tpu"
    NEURAL_ENGINE = "neural_engine"
    WIFI = "wifi"
    BLUETOOTH = "bluetooth"
    GPIO = "gpio"
    I2C = "i2c"
    SPI = "spi"
    UART = "uart"
    DISPLAY = "display"
    AUDIO = "audio"


@dataclass
class HardwareInfo:
    """硬件信息"""
    cpu_count: int
    cpu_freq: float  # MHz
    memory_total: int  # MB
    memory_available: int  # MB
    gpu_info: List[Dict[str, Any]]
    camera_info: List[Dict[str, Any]]
    capabilities: List[DeviceCapability]
    platform_specific: Dict[str, Any]


@dataclass
class ResourceUsage:
    """资源使用情况"""
    cpu_percent: float
    memory_percent: float
    memory_used: int  # MB
    gpu_percent: float
    temperature: Optional[float]  # 摄氏度
    power_consumption: Optional[float]  # 瓦特


class BasePlatformAdapter(ABC):
    """平台适配器基类，集成硬件抽象层"""
    
    def __init__(self, event_bus: EventBus = None, hardware_layer: Optional['HardwareAbstractionLayer'] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.event_bus = event_bus
        self._hardware_info: Optional[HardwareInfo] = None
        self._monitoring_thread: Optional[threading.Thread] = None
        self._monitoring_active = False
        
        # 硬件抽象层
        self.hardware_layer = hardware_layer
        self._initialized = False
        self._capabilities = None
        
    @abstractmethod
    def get_platform_type(self) -> PlatformType:
        """获取平台类型"""
        pass
    
    @abstractmethod
    def detect_hardware(self) -> HardwareInfo:
        """检测硬件信息"""
        pass
    
    @abstractmethod
    def get_resource_usage(self) -> ResourceUsage:
        """获取资源使用情况"""
        pass
    
    @abstractmethod
    def optimize_for_platform(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """为平台优化配置"""
        pass
    
    @abstractmethod
    def get_camera_devices(self) -> List[Dict[str, Any]]:
        """获取摄像头设备列表"""
        pass
    
    @abstractmethod
    def get_compute_devices(self) -> List[Dict[str, Any]]:
        """获取计算设备列表（GPU、TPU等）"""
        pass
    
    def initialize(self) -> bool:
        """初始化平台适配器"""
        try:
            self.logger.info(f"Initializing {self.get_platform_type().value} platform adapter")
            
            # 初始化硬件抽象层
            if self.hardware_layer:
                if not self.hardware_layer.initialize():
                    self.logger.error("硬件抽象层初始化失败")
                    return False
            
            # 检测硬件
            self._hardware_info = self.detect_hardware()
            
            # 启动资源监控
            self.start_monitoring()
            
            self._initialized = True
            self.logger.info("Platform adapter initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize platform adapter: {e}")
            return False
    
    def start_monitoring(self, interval: float = 5.0) -> None:
        """启动资源监控
        
        Args:
            interval: 监控间隔（秒）
        """
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self._monitoring_thread.start()
        self.logger.info(f"Started resource monitoring with {interval}s interval")
    
    def stop_monitoring(self) -> None:
        """停止资源监控"""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=1.0)
        self.logger.info("Stopped resource monitoring")
    
    def _monitoring_loop(self, interval: float) -> None:
        """监控循环
        
        Args:
            interval: 监控间隔
        """
        while self._monitoring_active:
            try:
                usage = self.get_resource_usage()
                
                # 发送资源使用事件
                if self.event_bus:
                    self.event_bus.emit('resource_usage_updated', {
                        'platform': self.get_platform_type().value,
                        'usage': usage
                    })
                
                # 检查资源警告
                self._check_resource_warnings(usage)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
            
            time.sleep(interval)
    
    def _check_resource_warnings(self, usage: ResourceUsage) -> None:
        """检查资源警告
        
        Args:
            usage: 资源使用情况
        """
        # CPU使用率警告
        if usage.cpu_percent > 90:
            self.logger.warning(f"High CPU usage: {usage.cpu_percent:.1f}%")
            if self.event_bus:
                self.event_bus.emit('resource_warning', {
                    'type': 'cpu',
                    'value': usage.cpu_percent,
                    'threshold': 90
                })
        
        # 内存使用率警告
        if usage.memory_percent > 85:
            self.logger.warning(f"High memory usage: {usage.memory_percent:.1f}%")
            if self.event_bus:
                self.event_bus.emit('resource_warning', {
                    'type': 'memory',
                    'value': usage.memory_percent,
                    'threshold': 85
                })
        
        # 温度警告
        if usage.temperature and usage.temperature > 80:
            self.logger.warning(f"High temperature: {usage.temperature:.1f}°C")
            if self.event_bus:
                self.event_bus.emit('resource_warning', {
                    'type': 'temperature',
                    'value': usage.temperature,
                    'threshold': 80
                })
    
    def get_hardware_info(self) -> Optional[HardwareInfo]:
        """获取硬件信息
        
        Returns:
            Optional[HardwareInfo]: 硬件信息
        """
        return self._hardware_info
    
    def is_capability_supported(self, capability: DeviceCapability) -> bool:
        """检查是否支持某种能力
        
        Args:
            capability: 设备能力
            
        Returns:
            bool: 是否支持
        """
        if not self._hardware_info:
            return False
        return capability in self._hardware_info.capabilities
    
    def get_optimal_batch_size(self, model_size: str = "medium") -> int:
        """获取最优批处理大小
        
        Args:
            model_size: 模型大小（small/medium/large）
            
        Returns:
            int: 最优批处理大小
        """
        if not self._hardware_info:
            return 1
        
        # 基于内存大小决定批处理大小
        memory_gb = self._hardware_info.memory_total / 1024
        
        if model_size == "small":
            if memory_gb >= 8:
                return 4
            elif memory_gb >= 4:
                return 2
            else:
                return 1
        elif model_size == "medium":
            if memory_gb >= 16:
                return 4
            elif memory_gb >= 8:
                return 2
            else:
                return 1
        else:  # large
            if memory_gb >= 32:
                return 2
            elif memory_gb >= 16:
                return 1
            else:
                return 1
    
    def get_optimal_thread_count(self) -> int:
        """获取最优线程数
        
        Returns:
            int: 最优线程数
        """
        if not self._hardware_info:
            return 2
        
        # 通常使用CPU核心数的一半到全部
        cpu_count = self._hardware_info.cpu_count
        if cpu_count <= 2:
            return cpu_count
        elif cpu_count <= 4:
            return cpu_count - 1
        else:
            return min(cpu_count // 2 + 2, cpu_count)


class WindowsPlatformAdapter(BasePlatformAdapter):
    """Windows平台适配器"""
    
    def __init__(self, event_bus: EventBus = None, hardware_layer: Optional['HardwareAbstractionLayer'] = None):
        super().__init__(event_bus, hardware_layer)
    
    def get_platform_type(self) -> PlatformType:
        return PlatformType.WINDOWS
    
    def detect_hardware(self) -> HardwareInfo:
        """检测Windows硬件信息"""
        # CPU信息
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else 0
        
        # 内存信息
        memory = psutil.virtual_memory()
        memory_total = memory.total // (1024 * 1024)  # MB
        memory_available = memory.available // (1024 * 1024)  # MB
        
        # GPU信息
        gpu_info = self._detect_windows_gpu()
        
        # 摄像头信息
        camera_info = self._detect_windows_cameras()
        
        # 能力检测
        capabilities = [DeviceCapability.CAMERA, DeviceCapability.DISPLAY, DeviceCapability.AUDIO]
        if gpu_info:
            capabilities.append(DeviceCapability.GPU)
        
        return HardwareInfo(
            cpu_count=cpu_count,
            cpu_freq=cpu_freq,
            memory_total=memory_total,
            memory_available=memory_available,
            gpu_info=gpu_info,
            camera_info=camera_info,
            capabilities=capabilities,
            platform_specific={
                'os_version': platform.platform(),
                'python_version': sys.version
            }
        )
    
    def _detect_windows_gpu(self) -> List[Dict[str, Any]]:
        """检测Windows GPU信息"""
        gpu_info = []
        
        try:
            import wmi
            c = wmi.WMI()
            for gpu in c.Win32_VideoController():
                if gpu.Name:
                    gpu_info.append({
                        'name': gpu.Name,
                        'memory': gpu.AdapterRAM // (1024 * 1024) if gpu.AdapterRAM else 0,
                        'driver_version': gpu.DriverVersion or 'Unknown'
                    })
        except ImportError:
            self.logger.warning("WMI not available, using basic GPU detection")
            # 基础GPU检测
            try:
                import subprocess
                result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')[1:]  # 跳过标题行
                    for line in lines:
                        name = line.strip()
                        if name:
                            gpu_info.append({'name': name, 'memory': 0, 'driver_version': 'Unknown'})
            except Exception as e:
                self.logger.error(f"Failed to detect GPU: {e}")
        
        return gpu_info
    
    def _detect_windows_cameras(self) -> List[Dict[str, Any]]:
        """检测Windows摄像头信息"""
        camera_info = []
        
        try:
            import cv2
            # 尝试检测摄像头
            for i in range(10):  # 检测前10个设备
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    
                    camera_info.append({
                        'index': i,
                        'name': f'Camera {i}',
                        'resolution': [width, height],
                        'fps': fps
                    })
                    cap.release()
                else:
                    break
        except Exception as e:
            self.logger.error(f"Failed to detect cameras: {e}")
        
        return camera_info
    
    def get_resource_usage(self) -> ResourceUsage:
        """获取Windows资源使用情况"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 内存使用情况
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used = memory.used // (1024 * 1024)  # MB
        
        # GPU使用率（简化实现）
        gpu_percent = 0.0
        
        # 温度（Windows较难获取，简化处理）
        temperature = None
        
        return ResourceUsage(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used=memory_used,
            gpu_percent=gpu_percent,
            temperature=temperature,
            power_consumption=None
        )
    
    def optimize_for_platform(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """为Windows平台优化配置"""
        optimized = config.copy()
        
        # Windows特定优化
        if 'performance' not in optimized:
            optimized['performance'] = {}
        
        # 设置合适的线程数
        optimized['performance']['num_threads'] = self.get_optimal_thread_count()
        
        # 设置批处理大小
        optimized['performance']['batch_size'] = self.get_optimal_batch_size()
        
        # Windows特定的模型路径
        if 'models' in optimized:
            for model_config in optimized['models'].values():
                if isinstance(model_config, dict) and 'path' in model_config:
                    # 确保使用Windows路径分隔符
                    model_config['path'] = model_config['path'].replace('/', '\\')
        
        return optimized
    
    def get_camera_devices(self) -> List[Dict[str, Any]]:
        """获取Windows摄像头设备列表"""
        if self._hardware_info:
            return self._hardware_info.camera_info
        return []
    
    def get_compute_devices(self) -> List[Dict[str, Any]]:
        """获取Windows计算设备列表"""
        devices = []
        
        # CPU设备
        devices.append({
            'type': 'cpu',
            'name': f'CPU ({self._hardware_info.cpu_count} cores)' if self._hardware_info else 'CPU',
            'available': True
        })
        
        # GPU设备
        if self._hardware_info and self._hardware_info.gpu_info:
            for gpu in self._hardware_info.gpu_info:
                devices.append({
                    'type': 'gpu',
                    'name': gpu['name'],
                    'memory': gpu['memory'],
                    'available': True
                })
        
        return devices


class LinuxPlatformAdapter(BasePlatformAdapter):
    """Linux平台适配器"""
    
    def __init__(self, event_bus: EventBus = None, hardware_layer: Optional['HardwareAbstractionLayer'] = None):
        super().__init__(event_bus, hardware_layer)
    
    def get_platform_type(self) -> PlatformType:
        # 检测是否为树莓派
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                if 'Raspberry Pi' in cpuinfo or 'BCM' in cpuinfo:
                    return PlatformType.RASPBERRY_PI
        except:
            pass
        
        # 检测是否为Jetson
        if os.path.exists('/etc/nv_tegra_release'):
            return PlatformType.JETSON
        
        return PlatformType.LINUX
    
    def detect_hardware(self) -> HardwareInfo:
        """检测Linux硬件信息"""
        # 基础信息
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else 0
        
        memory = psutil.virtual_memory()
        memory_total = memory.total // (1024 * 1024)
        memory_available = memory.available // (1024 * 1024)
        
        # GPU信息
        gpu_info = self._detect_linux_gpu()
        
        # 摄像头信息
        camera_info = self._detect_linux_cameras()
        
        # 能力检测
        capabilities = [DeviceCapability.CAMERA]
        if gpu_info:
            capabilities.append(DeviceCapability.GPU)
        
        # 检测GPIO等嵌入式能力
        if self.get_platform_type() in [PlatformType.RASPBERRY_PI, PlatformType.JETSON]:
            capabilities.extend([DeviceCapability.GPIO, DeviceCapability.I2C, DeviceCapability.SPI])
        
        return HardwareInfo(
            cpu_count=cpu_count,
            cpu_freq=cpu_freq,
            memory_total=memory_total,
            memory_available=memory_available,
            gpu_info=gpu_info,
            camera_info=camera_info,
            capabilities=capabilities,
            platform_specific={
                'kernel_version': platform.release(),
                'distribution': self._get_linux_distribution()
            }
        )
    
    def _detect_linux_gpu(self) -> List[Dict[str, Any]]:
        """检测Linux GPU信息"""
        gpu_info = []
        
        try:
            # 尝试使用nvidia-smi
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 2:
                            gpu_info.append({
                                'name': parts[0],
                                'memory': int(parts[1]),
                                'type': 'nvidia'
                            })
        except:
            pass
        
        # 检测其他GPU
        try:
            with open('/proc/modules', 'r') as f:
                modules = f.read()
                if 'amdgpu' in modules:
                    gpu_info.append({'name': 'AMD GPU', 'memory': 0, 'type': 'amd'})
                elif 'i915' in modules:
                    gpu_info.append({'name': 'Intel GPU', 'memory': 0, 'type': 'intel'})
        except:
            pass
        
        return gpu_info
    
    def _detect_linux_cameras(self) -> List[Dict[str, Any]]:
        """检测Linux摄像头信息"""
        camera_info = []
        
        # 检测/dev/video*设备
        import glob
        video_devices = glob.glob('/dev/video*')
        
        for device in video_devices:
            try:
                import cv2
                cap = cv2.VideoCapture(device)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    
                    camera_info.append({
                        'device': device,
                        'name': f'Camera {device}',
                        'resolution': [width, height],
                        'fps': fps
                    })
                    cap.release()
            except:
                pass
        
        return camera_info
    
    def _get_linux_distribution(self) -> str:
        """获取Linux发行版信息"""
        try:
            with open('/etc/os-release', 'r') as f:
                for line in f:
                    if line.startswith('PRETTY_NAME='):
                        return line.split('=')[1].strip('"\n')
        except:
            pass
        return 'Unknown Linux'
    
    def get_resource_usage(self) -> ResourceUsage:
        """获取Linux资源使用情况"""
        cpu_percent = psutil.cpu_percent(interval=1)
        
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used = memory.used // (1024 * 1024)
        
        gpu_percent = 0.0
        
        # 尝试获取温度
        temperature = self._get_cpu_temperature()
        
        return ResourceUsage(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used=memory_used,
            gpu_percent=gpu_percent,
            temperature=temperature,
            power_consumption=None
        )
    
    def _get_cpu_temperature(self) -> Optional[float]:
        """获取CPU温度"""
        try:
            # 树莓派温度
            if os.path.exists('/sys/class/thermal/thermal_zone0/temp'):
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp = int(f.read().strip()) / 1000.0
                    return temp
        except:
            pass
        
        try:
            # 其他Linux系统
            sensors = psutil.sensors_temperatures()
            if sensors:
                for name, entries in sensors.items():
                    if entries:
                        return entries[0].current
        except:
            pass
        
        return None
    
    def optimize_for_platform(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """为Linux平台优化配置"""
        optimized = config.copy()
        
        if 'performance' not in optimized:
            optimized['performance'] = {}
        
        # 设置线程数
        optimized['performance']['num_threads'] = self.get_optimal_thread_count()
        
        # 设置批处理大小
        optimized['performance']['batch_size'] = self.get_optimal_batch_size()
        
        # 嵌入式设备特殊优化
        platform_type = self.get_platform_type()
        if platform_type == PlatformType.RASPBERRY_PI:
            # 树莓派优化
            optimized['performance']['memory_optimization'] = True
            optimized['performance']['model_quantization'] = True
            optimized['performance']['batch_size'] = 1
        elif platform_type == PlatformType.JETSON:
            # Jetson优化
            optimized['performance']['use_tensorrt'] = True
            optimized['performance']['fp16_inference'] = True
        
        return optimized
    
    def get_camera_devices(self) -> List[Dict[str, Any]]:
        """获取Linux摄像头设备列表"""
        if self._hardware_info:
            return self._hardware_info.camera_info
        return []
    
    def get_compute_devices(self) -> List[Dict[str, Any]]:
        """获取Linux计算设备列表"""
        devices = []
        
        # CPU设备
        devices.append({
            'type': 'cpu',
            'name': f'CPU ({self._hardware_info.cpu_count} cores)' if self._hardware_info else 'CPU',
            'available': True
        })
        
        # GPU设备
        if self._hardware_info and self._hardware_info.gpu_info:
            for gpu in self._hardware_info.gpu_info:
                devices.append({
                    'type': 'gpu',
                    'name': gpu['name'],
                    'memory': gpu.get('memory', 0),
                    'gpu_type': gpu.get('type', 'unknown'),
                    'available': True
                })
        
        return devices


class MacOSPlatformAdapter(BasePlatformAdapter):
    """macOS平台适配器"""
    
    def __init__(self, event_bus: EventBus = None, hardware_layer: Optional['HardwareAbstractionLayer'] = None):
        super().__init__(event_bus, hardware_layer)
    
    def get_platform_type(self) -> PlatformType:
        return PlatformType.MACOS
    
    def detect_hardware(self) -> HardwareInfo:
        """检测macOS硬件信息"""
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else 0
        
        memory = psutil.virtual_memory()
        memory_total = memory.total // (1024 * 1024)
        memory_available = memory.available // (1024 * 1024)
        
        # macOS GPU检测
        gpu_info = self._detect_macos_gpu()
        
        # 摄像头检测
        camera_info = self._detect_macos_cameras()
        
        capabilities = [DeviceCapability.CAMERA, DeviceCapability.DISPLAY, DeviceCapability.AUDIO]
        if gpu_info:
            capabilities.append(DeviceCapability.GPU)
        
        # 检测Neural Engine（M1/M2芯片）
        if self._has_neural_engine():
            capabilities.append(DeviceCapability.NEURAL_ENGINE)
        
        return HardwareInfo(
            cpu_count=cpu_count,
            cpu_freq=cpu_freq,
            memory_total=memory_total,
            memory_available=memory_available,
            gpu_info=gpu_info,
            camera_info=camera_info,
            capabilities=capabilities,
            platform_specific={
                'macos_version': platform.mac_ver()[0],
                'machine': platform.machine()
            }
        )
    
    def _detect_macos_gpu(self) -> List[Dict[str, Any]]:
        """检测macOS GPU信息"""
        gpu_info = []
        
        try:
            import subprocess
            # 使用system_profiler获取GPU信息
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                # 简化解析
                lines = result.stdout.split('\n')
                current_gpu = {}
                for line in lines:
                    line = line.strip()
                    if 'Chipset Model:' in line:
                        current_gpu['name'] = line.split(':')[1].strip()
                    elif 'VRAM' in line and ':' in line:
                        try:
                            vram_str = line.split(':')[1].strip()
                            # 提取数字
                            import re
                            numbers = re.findall(r'\d+', vram_str)
                            if numbers:
                                current_gpu['memory'] = int(numbers[0])
                        except:
                            pass
                    elif line == '' and current_gpu:
                        gpu_info.append(current_gpu)
                        current_gpu = {}
                
                if current_gpu:
                    gpu_info.append(current_gpu)
        except:
            pass
        
        return gpu_info
    
    def _detect_macos_cameras(self) -> List[Dict[str, Any]]:
        """检测macOS摄像头信息"""
        camera_info = []
        
        try:
            import cv2
            # macOS通常摄像头索引从0开始
            for i in range(5):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    
                    camera_info.append({
                        'index': i,
                        'name': f'Camera {i}',
                        'resolution': [width, height],
                        'fps': fps
                    })
                    cap.release()
                else:
                    break
        except:
            pass
        
        return camera_info
    
    def _has_neural_engine(self) -> bool:
        """检测是否有Neural Engine"""
        machine = platform.machine().lower()
        return 'arm64' in machine or 'm1' in machine or 'm2' in machine
    
    def get_resource_usage(self) -> ResourceUsage:
        """获取macOS资源使用情况"""
        cpu_percent = psutil.cpu_percent(interval=1)
        
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used = memory.used // (1024 * 1024)
        
        gpu_percent = 0.0
        temperature = None
        
        return ResourceUsage(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used=memory_used,
            gpu_percent=gpu_percent,
            temperature=temperature,
            power_consumption=None
        )
    
    def optimize_for_platform(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """为macOS平台优化配置"""
        optimized = config.copy()
        
        if 'performance' not in optimized:
            optimized['performance'] = {}
        
        optimized['performance']['num_threads'] = self.get_optimal_thread_count()
        optimized['performance']['batch_size'] = self.get_optimal_batch_size()
        
        # M1/M2芯片优化
        if self._has_neural_engine():
            optimized['performance']['use_metal'] = True
            optimized['performance']['use_neural_engine'] = True
        
        return optimized
    
    def get_camera_devices(self) -> List[Dict[str, Any]]:
        """获取macOS摄像头设备列表"""
        if self._hardware_info:
            return self._hardware_info.camera_info
        return []
    
    def get_compute_devices(self) -> List[Dict[str, Any]]:
        """获取macOS计算设备列表"""
        devices = []
        
        # CPU设备
        devices.append({
            'type': 'cpu',
            'name': f'CPU ({self._hardware_info.cpu_count} cores)' if self._hardware_info else 'CPU',
            'available': True
        })
        
        # GPU设备
        if self._hardware_info and self._hardware_info.gpu_info:
            for gpu in self._hardware_info.gpu_info:
                devices.append({
                    'type': 'gpu',
                    'name': gpu['name'],
                    'memory': gpu.get('memory', 0),
                    'available': True
                })
        
        # Neural Engine
        if self.is_capability_supported(DeviceCapability.NEURAL_ENGINE):
            devices.append({
                'type': 'neural_engine',
                'name': 'Apple Neural Engine',
                'available': True
            })
        
        return devices


class PlatformAdapterFactory:
    """平台适配器工厂"""
    
    @staticmethod
    def create_adapter(event_bus: EventBus = None, hardware_layer: Optional['HardwareAbstractionLayer'] = None) -> BasePlatformAdapter:
        """创建平台适配器
        
        Args:
            event_bus: 事件总线
            hardware_layer: 硬件抽象层
            
        Returns:
            BasePlatformAdapter: 平台适配器实例
        """
        system = platform.system().lower()
        
        if system == 'windows':
            return WindowsPlatformAdapter(event_bus, hardware_layer)
        elif system == 'linux':
            return LinuxPlatformAdapter(event_bus, hardware_layer)
        elif system == 'darwin':
            return MacOSPlatformAdapter(event_bus, hardware_layer)
        else:
            raise ValueError(f"Unsupported platform: {system}")
    
    @staticmethod
    def get_current_platform() -> PlatformType:
        """获取当前平台类型
        
        Returns:
            PlatformType: 平台类型
        """
        system = platform.system().lower()
        
        if system == 'windows':
            return PlatformType.WINDOWS
        elif system == 'linux':
            # 进一步检测Linux子类型
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                    if 'Raspberry Pi' in cpuinfo or 'BCM' in cpuinfo:
                        return PlatformType.RASPBERRY_PI
            except:
                pass
            
            if os.path.exists('/etc/nv_tegra_release'):
                return PlatformType.JETSON
            
            return PlatformType.LINUX
        elif system == 'darwin':
            return PlatformType.MACOS
        else:
            return PlatformType.UNKNOWN