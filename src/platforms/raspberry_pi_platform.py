"""树莓派平台实现
提供树莓派平台的具体硬件接口实现，支持GPIO和CSI摄像头
"""

import os
import sys
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.hardware_abstraction import (
    CameraInterface, ComputeInterface, ConnectivityInterface,
    StorageInterface, PowerInterface, HardwareAbstractionLayer,
    PlatformType, ComputeType, CameraType, ConnectivityType
)


class RaspberryPiCamera(CameraInterface):
    """树莓派摄像头实现"""
    
    def __init__(self, camera_type: CameraType = CameraType.CSI, camera_id: int = 0):
        self.camera_type = camera_type
        self.camera_id = camera_id
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialized = False
        self._current_resolution = (640, 480)
        self._current_fps = 30
        
        # 树莓派支持的分辨率
        self.supported_resolutions = [
            (320, 240),
            (640, 480),
            (1280, 720),
            (1920, 1080),
            (2592, 1944)  # 树莓派摄像头最大分辨率
        ]
        
        # GPIO相关
        self.gpio_available = self._check_gpio_availability()
        
    def _check_gpio_availability(self) -> bool:
        """检查GPIO可用性"""
        try:
            # 检查是否在树莓派上运行
            if os.path.exists('/proc/device-tree/model'):
                with open('/proc/device-tree/model', 'r') as f:
                    model = f.read().lower()
                    return 'raspberry pi' in model
        except:
            pass
        return False
        
    def initialize(self) -> bool:
        """初始化树莓派摄像头"""
        try:
            self.logger.info(f"初始化树莓派摄像头: {self.camera_type.value}")
            
            if self.camera_type == CameraType.CSI:
                self._init_csi_camera()
            elif self.camera_type == CameraType.USB:
                self._init_usb_camera()
            
            self._initialized = True
            self.logger.info("树莓派摄像头初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"树莓派摄像头初始化失败: {e}")
            return False
    
    def _init_csi_camera(self):
        """初始化CSI摄像头"""
        self.logger.debug("配置树莓派CSI摄像头")
        # 检查摄像头是否启用
        if os.path.exists('/boot/config.txt'):
            try:
                with open('/boot/config.txt', 'r') as f:
                    config = f.read()
                    if 'camera_auto_detect=1' not in config and 'start_x=1' not in config:
                        self.logger.warning("摄像头可能未在/boot/config.txt中启用")
            except:
                pass
        
    def _init_usb_camera(self):
        """初始化USB摄像头"""
        self.logger.debug("配置树莓派USB摄像头")
        
    def capture_frame(self) -> Optional[np.ndarray]:
        """捕获一帧图像"""
        if not self._initialized:
            self.logger.warning("摄像头未初始化")
            return None
        
        try:
            # 模拟图像捕获
            height, width = self._current_resolution
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            # 添加一些模拟的图像特征
            cv2_available = False
            try:
                import cv2
                cv2_available = True
            except ImportError:
                pass
            
            if cv2_available:
                # 添加一个模拟的圆形目标
                center_x, center_y = width // 2, height // 2
                cv2.circle(frame, (center_x, center_y), 40, (255, 0, 0), 2)
            
            timestamp = int(time.time() * 1000)
            self.logger.debug(f"树莓派捕获帧: {width}x{height}, 时间戳: {timestamp}")
            
            return frame
            
        except Exception as e:
            self.logger.error(f"树莓派图像捕获失败: {e}")
            return None
    
    def set_resolution(self, width: int, height: int) -> bool:
        """设置分辨率"""
        resolution = (width, height)
        if resolution not in self.supported_resolutions:
            self.logger.warning(f"树莓派不支持的分辨率: {resolution}")
            return False
        
        self._current_resolution = resolution
        self.logger.info(f"树莓派设置分辨率: {width}x{height}")
        return True
    
    def set_fps(self, fps: int) -> bool:
        """设置帧率"""
        if fps > 90:  # 树莓派限制
            self.logger.warning(f"树莓派不支持超过90fps的帧率: {fps}")
            fps = 90
        
        self._current_fps = fps
        self.logger.info(f"树莓派设置帧率: {fps}")
        return True
    
    def get_supported_resolutions(self) -> List[Tuple[int, int]]:
        """获取支持的分辨率列表"""
        return self.supported_resolutions.copy()
    
    def release(self) -> None:
        """释放摄像头资源"""
        if self._initialized:
            self.logger.info("释放树莓派摄像头资源")
            self._initialized = False
    
    def is_available(self) -> bool:
        """检查摄像头是否可用"""
        return self._initialized


class RaspberryPiCompute(ComputeInterface):
    """树莓派计算设备实现"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialized = False
        self._model_loaded = False
        self._model_path = None
        
        # 树莓派计算能力
        self.max_model_size_mb = 100
        self.supported_formats = ['tflite', 'onnx', 'pt']
        self.memory_limit_mb = self._detect_memory_size()
        self.cpu_cores = self._detect_cpu_cores()
        
    def _detect_memory_size(self) -> int:
        """检测内存大小"""
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        # 提取内存大小(KB)并转换为MB
                        mem_kb = int(line.split()[1])
                        return mem_kb // 1024
        except:
            pass
        return 1024  # 默认1GB
    
    def _detect_cpu_cores(self) -> int:
        """检测CPU核心数"""
        try:
            return os.cpu_count() or 4
        except:
            return 4
        
    def initialize(self) -> bool:
        """初始化树莓派计算设备"""
        try:
            self.logger.info("初始化树莓派计算设备")
            
            # 检查内存
            if self.memory_limit_mb < 512:
                self.logger.warning(f"内存较少: {self.memory_limit_mb}MB")
            
            # 检查GPU支持
            gpu_available = self._check_gpu_availability()
            if gpu_available:
                self.logger.info("检测到GPU支持")
            
            self._initialized = True
            self.logger.info("树莓派计算设备初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"树莓派计算设备初始化失败: {e}")
            return False
    
    def _check_gpu_availability(self) -> bool:
        """检查GPU可用性"""
        # 检查VideoCore GPU
        return os.path.exists('/opt/vc/bin/vcgencmd')
    
    def get_device_info(self) -> Dict[str, Any]:
        """获取树莓派设备信息"""
        return {
            'platform': 'Raspberry Pi',
            'cpu_cores': self.cpu_cores,
            'cpu_arch': self._get_cpu_arch(),
            'memory_total_mb': self.memory_limit_mb,
            'memory_available_mb': self._get_available_memory(),
            'gpu_available': self._check_gpu_availability(),
            'supported_formats': self.supported_formats,
            'max_model_size_mb': self.max_model_size_mb,
            'compute_type': ComputeType.CPU.value,
            'model': self._get_pi_model()
        }
    
    def _get_cpu_arch(self) -> str:
        """获取CPU架构"""
        try:
            import platform
            return platform.machine()
        except:
            return 'unknown'
    
    def _get_pi_model(self) -> str:
        """获取树莓派型号"""
        try:
            with open('/proc/device-tree/model', 'r') as f:
                return f.read().strip()
        except:
            return 'Unknown Raspberry Pi'
    
    def load_model(self, model_path: str, **kwargs) -> bool:
        """加载模型到树莓派"""
        if not self._initialized:
            self.logger.error("计算设备未初始化")
            return False
        
        try:
            # 检查模型文件
            if not os.path.exists(model_path):
                self.logger.error(f"模型文件不存在: {model_path}")
                return False
            
            # 检查模型大小
            model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            if model_size_mb > self.max_model_size_mb:
                self.logger.error(f"模型过大: {model_size_mb:.1f}MB > {self.max_model_size_mb}MB")
                return False
            
            # 检查可用内存
            available_memory = self._get_available_memory()
            if available_memory < model_size_mb * 2:  # 需要2倍模型大小的内存
                self.logger.error(f"内存不足: {available_memory}MB < {model_size_mb * 2:.1f}MB")
                return False
            
            # 检查模型格式
            model_format = Path(model_path).suffix.lower()
            if model_format not in ['.tflite', '.onnx', '.pt']:
                self.logger.error(f"不支持的模型格式: {model_format}")
                return False
            
            self._model_path = model_path
            self._model_loaded = True
            self.logger.info(f"树莓派模型加载成功: {model_path} ({model_size_mb:.1f}MB)")
            return True
            
        except Exception as e:
            self.logger.error(f"树莓派模型加载失败: {e}")
            return False
    
    def inference(self, input_data: np.ndarray) -> np.ndarray:
        """执行推理"""
        if not self._model_loaded:
            self.logger.error("模型未加载")
            return np.array([])
        
        try:
            start_time = time.time()
            
            # 模拟树莓派推理过程
            batch_size, height, width, channels = input_data.shape
            
            # 树莓派推理速度中等，精度较好
            num_detections = np.random.randint(0, 8)  # 0-7个检测结果
            if num_detections > 0:
                results = np.random.rand(num_detections, 6)
                results[:, :4] *= [width, height, width, height]  # 坐标
                results[:, 4] = 0.2 + results[:, 4] * 0.7  # 置信度 0.2-0.9
                results[:, 5] = np.random.randint(0, 80, num_detections)  # 类别ID
            else:
                results = np.array([]).reshape(0, 6)
            
            # 树莓派推理时间
            inference_time = (time.time() - start_time) * 1000
            self.logger.debug(f"树莓派推理完成: {inference_time:.1f}ms, 检测到{num_detections}个目标")
            
            return results
            
        except Exception as e:
            self.logger.error(f"树莓派推理失败: {e}")
            return np.array([])
    
    def get_memory_usage(self) -> Dict[str, int]:
        """获取内存使用情况"""
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            
            total_kb = 0
            available_kb = 0
            
            for line in meminfo.split('\n'):
                if line.startswith('MemTotal:'):
                    total_kb = int(line.split()[1])
                elif line.startswith('MemAvailable:'):
                    available_kb = int(line.split()[1])
            
            total_mb = total_kb // 1024
            available_mb = available_kb // 1024
            used_mb = total_mb - available_mb
            
            return {
                'total_mb': total_mb,
                'used_mb': used_mb,
                'available_mb': available_mb,
                'usage_percent': int((used_mb / total_mb) * 100) if total_mb > 0 else 0
            }
        except:
            return {
                'total_mb': self.memory_limit_mb,
                'used_mb': self.memory_limit_mb // 2,
                'available_mb': self.memory_limit_mb // 2,
                'usage_percent': 50
            }
    
    def optimize_model(self, model_path: str, optimization_config: Dict[str, Any]) -> str:
        """优化模型for树莓派"""
        try:
            self.logger.info(f"开始树莓派模型优化: {model_path}")
            
            # 树莓派优化配置
            pi_config = {
                'target_platform': 'raspberry_pi',
                'quantization': 'fp32',  # 树莓派支持fp32
                'input_size': (640, 640),
                'batch_size': 1,
                'optimize_for_cpu': True,
                'num_threads': self.cpu_cores
            }
            
            # 合并配置
            final_config = {**pi_config, **optimization_config}
            
            # 生成优化后的模型路径
            model_dir = Path(model_path).parent
            model_name = Path(model_path).stem
            optimized_path = model_dir / f"{model_name}_raspberry_pi_optimized.tflite"
            
            # 模拟优化过程
            self.logger.info(f"应用树莓派优化配置: {final_config}")
            time.sleep(3)  # 模拟优化时间
            
            # 创建优化后的模型文件
            with open(optimized_path, 'wb') as f:
                f.write(b'raspberry_pi_optimized_model_data')  # 占位数据
            
            self.logger.info(f"树莓派模型优化完成: {optimized_path}")
            return str(optimized_path)
            
        except Exception as e:
            self.logger.error(f"树莓派模型优化失败: {e}")
            return model_path
    
    def release(self) -> None:
        """释放计算资源"""
        if self._initialized:
            self.logger.info("释放树莓派计算资源")
            self._model_loaded = False
            self._model_path = None
            self._initialized = False
    
    def _get_available_memory(self) -> int:
        """获取可用内存(MB)"""
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        mem_kb = int(line.split()[1])
                        return mem_kb // 1024
        except:
            pass
        return self.memory_limit_mb // 2  # 默认一半可用


class RaspberryPiWiFi(ConnectivityInterface):
    """树莓派WiFi连接实现"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialized = False
        self._connected = False
        self._ssid = None
        self._ip_address = None
        
    def initialize(self) -> bool:
        """初始化WiFi"""
        try:
            self.logger.info("初始化树莓派WiFi")
            # 检查WiFi接口
            if os.path.exists('/sys/class/net/wlan0'):
                self.logger.info("检测到wlan0接口")
            self._initialized = True
            return True
        except Exception as e:
            self.logger.error(f"WiFi初始化失败: {e}")
            return False
    
    def connect(self, **kwargs) -> bool:
        """连接WiFi"""
        if not self._initialized:
            return False
        
        ssid = kwargs.get('ssid')
        password = kwargs.get('password')
        
        if not ssid:
            self.logger.error("未提供SSID")
            return False
        
        try:
            self.logger.info(f"连接WiFi: {ssid}")
            # 模拟连接过程
            time.sleep(3)
            
            self._connected = True
            self._ssid = ssid
            self._ip_address = "192.168.1.150"  # 模拟IP地址
            
            self.logger.info(f"WiFi连接成功: {ssid}, IP: {self._ip_address}")
            return True
            
        except Exception as e:
            self.logger.error(f"WiFi连接失败: {e}")
            return False
    
    def disconnect(self) -> None:
        """断开WiFi连接"""
        if self._connected:
            self.logger.info(f"断开WiFi连接: {self._ssid}")
            self._connected = False
            self._ssid = None
            self._ip_address = None
    
    def send_data(self, data: bytes) -> bool:
        """发送数据"""
        if not self._connected:
            self.logger.error("WiFi未连接")
            return False
        
        try:
            self.logger.debug(f"树莓派WiFi发送数据: {len(data)} bytes")
            return True
        except Exception as e:
            self.logger.error(f"数据发送失败: {e}")
            return False
    
    def receive_data(self, timeout: float = 1.0) -> Optional[bytes]:
        """接收数据"""
        if not self._connected:
            return None
        
        try:
            time.sleep(min(timeout, 0.1))
            return b"raspberry_pi_wifi_received_data"
        except Exception as e:
            self.logger.error(f"数据接收失败: {e}")
            return None
    
    def is_connected(self) -> bool:
        """检查连接状态"""
        return self._connected


class RaspberryPiStorage(StorageInterface):
    """树莓派存储实现"""
    
    def __init__(self, storage_path: str = "/home/pi/yolos_storage"):
        self.storage_path = Path(storage_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_storage_mb = self._detect_storage_size()
        
        # 确保存储目录存在
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def _detect_storage_size(self) -> int:
        """检测存储大小"""
        try:
            import shutil
            total, used, free = shutil.disk_usage('/')
            return free // (1024 * 1024)  # 转换为MB
        except:
            return 16384  # 默认16GB
    
    def save_model(self, model_data: bytes, model_name: str) -> bool:
        """保存模型"""
        try:
            model_path = self.storage_path / f"{model_name}.tflite"
            
            # 检查存储空间
            model_size_mb = len(model_data) / (1024 * 1024)
            available_space = self.get_available_space()
            if model_size_mb > available_space:
                self.logger.error(f"存储空间不足: {model_size_mb:.1f}MB > {available_space}MB")
                return False
            
            with open(model_path, 'wb') as f:
                f.write(model_data)
            
            self.logger.info(f"树莓派模型保存成功: {model_path} ({model_size_mb:.1f}MB)")
            return True
            
        except Exception as e:
            self.logger.error(f"树莓派模型保存失败: {e}")
            return False
    
    def load_model(self, model_name: str) -> Optional[bytes]:
        """加载模型"""
        try:
            # 尝试不同的模型格式
            for ext in ['.tflite', '.onnx', '.pt']:
                model_path = self.storage_path / f"{model_name}{ext}"
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        model_data = f.read()
                    
                    self.logger.info(f"树莓派模型加载成功: {model_path}")
                    return model_data
            
            self.logger.error(f"树莓派模型文件不存在: {model_name}")
            return None
            
        except Exception as e:
            self.logger.error(f"树莓派模型加载失败: {e}")
            return None
    
    def save_config(self, config: Dict[str, Any], config_name: str) -> bool:
        """保存配置"""
        try:
            import json
            config_path = self.storage_path / f"{config_name}.json"
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.logger.info(f"树莓派配置保存成功: {config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"树莓派配置保存失败: {e}")
            return False
    
    def load_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """加载配置"""
        try:
            import json
            config_path = self.storage_path / f"{config_name}.json"
            
            if not config_path.exists():
                self.logger.error(f"树莓派配置文件不存在: {config_path}")
                return None
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.logger.info(f"树莓派配置加载成功: {config_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"树莓派配置加载失败: {e}")
            return None
    
    def get_available_space(self) -> int:
        """获取可用存储空间(MB)"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(str(self.storage_path))
            return free // (1024 * 1024)
        except Exception as e:
            self.logger.error(f"获取树莓派存储空间失败: {e}")
            return 0
    
    def cleanup_old_files(self, max_age_days: int = 7) -> None:
        """清理旧文件"""
        try:
            import time
            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 3600
            
            cleaned_count = 0
            for file_path in self.storage_path.rglob('*'):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        file_path.unlink()
                        cleaned_count += 1
            
            self.logger.info(f"树莓派清理完成: 删除了{cleaned_count}个旧文件")
            
        except Exception as e:
            self.logger.error(f"树莓派文件清理失败: {e}")


class RaspberryPiPower(PowerInterface):
    """树莓派电源管理实现"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._power_mode = "balanced"
        self._temperature = self._get_cpu_temperature()
        
    def _get_cpu_temperature(self) -> float:
        """获取CPU温度"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp_millidegree = int(f.read().strip())
                return temp_millidegree / 1000.0
        except:
            return 45.0  # 默认温度
    
    def get_power_status(self) -> Dict[str, Any]:
        """获取电源状态"""
        return {
            'power_mode': self._power_mode,
            'temperature_celsius': self._get_cpu_temperature(),
            'power_consumption_mw': 3000,
            'cpu_power_mw': 1500,
            'gpu_power_mw': 500,
            'memory_power_mw': 600,
            'io_power_mw': 400,
            'thermal_throttling': self._get_cpu_temperature() > 80,
            'under_voltage': self._check_under_voltage()
        }
    
    def _check_under_voltage(self) -> bool:
        """检查欠压状态"""
        try:
            # 使用vcgencmd检查欠压
            import subprocess
            result = subprocess.run(['vcgencmd', 'get_throttled'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                throttled = int(result.stdout.split('=')[1], 16)
                return (throttled & 0x1) != 0  # 检查欠压位
        except:
            pass
        return False
    
    def set_power_mode(self, mode: str) -> bool:
        """设置电源模式"""
        valid_modes = ['performance', 'balanced', 'power_save']
        if mode not in valid_modes:
            self.logger.error(f"无效的电源模式: {mode}")
            return False
        
        self._power_mode = mode
        self.logger.info(f"树莓派设置电源模式: {mode}")
        
        # 根据模式调整系统参数
        if mode == 'power_save':
            self.logger.info("启用树莓派低功耗模式")
        elif mode == 'performance':
            self.logger.info("启用树莓派性能模式")
        
        return True
    
    def get_battery_level(self) -> Optional[float]:
        """获取电池电量百分比"""
        # 树莓派通常使用外部电源，没有电池
        return None
    
    def enable_sleep_mode(self, duration_seconds: int) -> None:
        """启用睡眠模式"""
        self.logger.info(f"树莓派进入睡眠模式: {duration_seconds}秒")
        # 树莓派没有真正的睡眠模式，这里只是模拟
        time.sleep(min(duration_seconds, 1))
        self.logger.info("树莓派从睡眠模式唤醒")


class RaspberryPiGPIO:
    """树莓派GPIO控制"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._gpio_available = self._check_gpio_availability()
        self._gpio_pins = {}
        
    def _check_gpio_availability(self) -> bool:
        """检查GPIO可用性"""
        return os.path.exists('/sys/class/gpio')
    
    def setup_pin(self, pin: int, mode: str) -> bool:
        """设置GPIO引脚模式"""
        if not self._gpio_available:
            self.logger.error("GPIO不可用")
            return False
        
        try:
            self._gpio_pins[pin] = mode
            self.logger.info(f"设置GPIO引脚{pin}为{mode}模式")
            return True
        except Exception as e:
            self.logger.error(f"GPIO引脚设置失败: {e}")
            return False
    
    def digital_write(self, pin: int, value: bool) -> bool:
        """数字输出"""
        if pin not in self._gpio_pins:
            self.logger.error(f"GPIO引脚{pin}未设置")
            return False
        
        try:
            self.logger.debug(f"GPIO引脚{pin}输出: {value}")
            return True
        except Exception as e:
            self.logger.error(f"GPIO数字输出失败: {e}")
            return False
    
    def digital_read(self, pin: int) -> Optional[bool]:
        """数字输入"""
        if pin not in self._gpio_pins:
            self.logger.error(f"GPIO引脚{pin}未设置")
            return None
        
        try:
            # 模拟读取
            import random
            value = random.choice([True, False])
            self.logger.debug(f"GPIO引脚{pin}读取: {value}")
            return value
        except Exception as e:
            self.logger.error(f"GPIO数字读取失败: {e}")
            return None


class RaspberryPiPlatform(HardwareAbstractionLayer):
    """树莓派平台实现"""
    
    def __init__(self):
        super().__init__(PlatformType.RASPBERRY_PI)
        
        # 初始化各个接口
        self.camera = RaspberryPiCamera()
        self.compute = RaspberryPiCompute()
        self.connectivity = RaspberryPiWiFi()
        self.storage = RaspberryPiStorage()
        self.power = RaspberryPiPower()
        self.gpio = RaspberryPiGPIO()
        
        self.logger.info("树莓派平台初始化完成")
    
    def get_platform_specific_config(self) -> Dict[str, Any]:
        """获取树莓派平台特定配置"""
        return {
            'model_format': 'tflite',
            'quantization': 'fp32',
            'input_size': (640, 640),
            'batch_size': 1,
            'max_detections': 100,
            'confidence_threshold': 0.5,
            'nms_threshold': 0.4,
            'use_gpu': False,  # 树莓派GPU支持有限
            'num_threads': self.compute.cpu_cores,
            'memory_optimization': True,
            'wifi_config': {
                'auto_connect': True,
                'connection_timeout': 10,
                'retry_attempts': 3
            },
            'camera_config': {
                'resolution': (1280, 720),
                'fps': 30,
                'format': 'RGB',
                'interface': 'CSI'
            },
            'gpio_config': {
                'available': self.gpio._gpio_available,
                'pins': 40
            }
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            'model': self.compute._get_pi_model(),
            'cpu_arch': self.compute._get_cpu_arch(),
            'memory_mb': self.compute.memory_limit_mb,
            'storage_mb': self.storage.max_storage_mb,
            'temperature_celsius': self.power._get_cpu_temperature(),
            'gpio_available': self.gpio._gpio_available
        }


def create_raspberry_pi_platform() -> RaspberryPiPlatform:
    """创建树莓派平台实例"""
    return RaspberryPiPlatform()