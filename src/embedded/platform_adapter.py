#!/usr/bin/env python3
"""
嵌入式平台适配器
自动检测硬件平台并提供相应的优化配置
"""

import os
import sys
import platform
import subprocess
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

@dataclass
class HardwareInfo:
    """硬件信息"""
    platform_name: str
    cpu_model: str
    cpu_cores: int
    cpu_freq_mhz: float
    memory_total_mb: int
    memory_available_mb: int
    storage_total_gb: float
    storage_available_gb: float
    has_gpu: bool = False
    gpu_model: str = ""
    gpu_memory_mb: int = 0
    has_npu: bool = False
    npu_model: str = ""
    architecture: str = ""
    os_name: str = ""
    python_version: str = ""
    
@dataclass
class PlatformCapabilities:
    """平台能力"""
    max_model_size_mb: float
    recommended_input_size: Tuple[int, int]
    supported_precisions: List[str]
    supported_formats: List[str]
    max_batch_size: int
    recommended_threads: int
    memory_limit_mb: int
    power_limit_mw: int
    thermal_limit_celsius: int
    can_use_gpu: bool = False
    can_use_npu: bool = False
    optimization_flags: List[str] = None

class PlatformDetector:
    """平台检测器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def detect_hardware(self) -> HardwareInfo:
        """检测硬件信息"""
        info = HardwareInfo(
            platform_name=self._detect_platform_name(),
            cpu_model=self._get_cpu_model(),
            cpu_cores=self._get_cpu_cores(),
            cpu_freq_mhz=self._get_cpu_frequency(),
            memory_total_mb=self._get_total_memory(),
            memory_available_mb=self._get_available_memory(),
            storage_total_gb=self._get_storage_info()[0],
            storage_available_gb=self._get_storage_info()[1],
            architecture=platform.machine(),
            os_name=platform.system(),
            python_version=platform.python_version()
        )
        
        # 检测GPU
        gpu_info = self._detect_gpu()
        if gpu_info:
            info.has_gpu = True
            info.gpu_model = gpu_info[0]
            info.gpu_memory_mb = gpu_info[1]
            
        # 检测NPU
        npu_info = self._detect_npu()
        if npu_info:
            info.has_npu = True
            info.npu_model = npu_info
            
        return info
        
    def _detect_platform_name(self) -> str:
        """检测平台名称"""
        # 检查设备树文件 (Linux)
        device_tree_paths = [
            "/proc/device-tree/model",
            "/proc/device-tree/compatible",
            "/sys/firmware/devicetree/base/model"
        ]
        
        for path in device_tree_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        content = f.read().strip().replace('\x00', '')
                        if content:
                            # 识别常见平台
                            if 'raspberry' in content.lower():
                                if 'pi 4' in content.lower():
                                    return "Raspberry Pi 4B"
                                elif 'pi zero 2' in content.lower():
                                    return "Raspberry Pi Zero 2W"
                                elif 'pi zero' in content.lower():
                                    return "Raspberry Pi Zero"
                                else:
                                    return "Raspberry Pi"
                            elif 'jetson' in content.lower():
                                if 'nano' in content.lower():
                                    return "NVIDIA Jetson Nano"
                                elif 'xavier' in content.lower():
                                    return "NVIDIA Jetson Xavier"
                                else:
                                    return "NVIDIA Jetson"
                            return content
                except:
                    pass
                    
        # 检查CPU信息
        cpu_info = self._get_cpu_info()
        if cpu_info:
            if 'esp32' in cpu_info.lower():
                return "ESP32"
            elif 'cortex-a72' in cpu_info.lower():
                return "Raspberry Pi 4B"
            elif 'cortex-a53' in cpu_info.lower():
                return "Raspberry Pi 3B+"
                
        # 检查环境变量
        if os.environ.get('JETSON_MODEL_NAME'):
            return f"NVIDIA Jetson {os.environ['JETSON_MODEL_NAME']}"
            
        # 默认返回系统信息
        return f"{platform.system()} {platform.machine()}"
        
    def _get_cpu_info(self) -> str:
        """获取CPU信息"""
        try:
            if os.path.exists('/proc/cpuinfo'):
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if line.startswith('model name') or line.startswith('Hardware'):
                            return line.split(':', 1)[1].strip()
        except:
            pass
            
        return platform.processor()
        
    def _get_cpu_model(self) -> str:
        """获取CPU型号"""
        cpu_info = self._get_cpu_info()
        if cpu_info:
            return cpu_info
            
        if PSUTIL_AVAILABLE:
            try:
                return f"{psutil.cpu_count()} cores"
            except:
                pass
                
        return "Unknown"
        
    def _get_cpu_cores(self) -> int:
        """获取CPU核心数"""
        if PSUTIL_AVAILABLE:
            try:
                return psutil.cpu_count(logical=False) or psutil.cpu_count()
            except:
                pass
                
        return os.cpu_count() or 1
        
    def _get_cpu_frequency(self) -> float:
        """获取CPU频率"""
        if PSUTIL_AVAILABLE:
            try:
                freq = psutil.cpu_freq()
                if freq:
                    return freq.max or freq.current
            except:
                pass
                
        # 尝试从/proc/cpuinfo读取
        try:
            if os.path.exists('/proc/cpuinfo'):
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'cpu MHz' in line.lower():
                            return float(line.split(':')[1].strip())
        except:
            pass
            
        return 0.0
        
    def _get_total_memory(self) -> int:
        """获取总内存(MB)"""
        if PSUTIL_AVAILABLE:
            try:
                return int(psutil.virtual_memory().total / 1024 / 1024)
            except:
                pass
                
        # 尝试从/proc/meminfo读取
        try:
            if os.path.exists('/proc/meminfo'):
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            kb = int(line.split()[1])
                            return int(kb / 1024)
        except:
            pass
            
        return 1024  # 默认1GB
        
    def _get_available_memory(self) -> int:
        """获取可用内存(MB)"""
        if PSUTIL_AVAILABLE:
            try:
                return int(psutil.virtual_memory().available / 1024 / 1024)
            except:
                pass
                
        return int(self._get_total_memory() * 0.7)  # 估算70%可用
        
    def _get_storage_info(self) -> Tuple[float, float]:
        """获取存储信息(GB) - (总容量, 可用容量)"""
        if PSUTIL_AVAILABLE:
            try:
                usage = psutil.disk_usage('/')
                total_gb = usage.total / 1024 / 1024 / 1024
                free_gb = usage.free / 1024 / 1024 / 1024
                return total_gb, free_gb
            except:
                pass
                
        return 32.0, 16.0  # 默认32GB总容量，16GB可用
        
    def _detect_gpu(self) -> Optional[Tuple[str, int]]:
        """检测GPU - 返回(型号, 显存MB)"""
        # 检测NVIDIA GPU
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    parts = lines[0].split(', ')
                    if len(parts) >= 2:
                        name = parts[0].strip()
                        memory_mb = int(parts[1].strip())
                        return name, memory_mb
        except:
            pass
            
        # 使用GPUtil检测
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    return gpu.name, int(gpu.memoryTotal)
            except:
                pass
                
        # 检测集成GPU (树莓派等)
        gpu_paths = [
            "/sys/class/drm/card0/device/vendor",
            "/proc/device-tree/soc/gpu/compatible"
        ]
        
        for path in gpu_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        content = f.read().strip()
                        if 'broadcom' in content.lower() or 'videocore' in content.lower():
                            return "VideoCore VI", 76  # 树莓派4B GPU内存
                except:
                    pass
                    
        return None
        
    def _detect_npu(self) -> Optional[str]:
        """检测NPU"""
        # 检测Canaan K230 NPU
        if os.path.exists('/dev/kpu'):
            return "Canaan KPU"
            
        # 检测其他NPU
        npu_indicators = [
            '/sys/class/npu',
            '/dev/npu',
            '/proc/device-tree/npu'
        ]
        
        for indicator in npu_indicators:
            if os.path.exists(indicator):
                return "Generic NPU"
                
        return None

class PlatformAdapter:
    """平台适配器"""
    
    def __init__(self):
        self.detector = PlatformDetector()
        self.logger = logging.getLogger(__name__)
        self.hardware_info = None
        self.capabilities = None
        
    def initialize(self) -> HardwareInfo:
        """初始化适配器"""
        self.hardware_info = self.detector.detect_hardware()
        self.capabilities = self._determine_capabilities(self.hardware_info)
        
        self.logger.info(f"检测到平台: {self.hardware_info.platform_name}")
        self.logger.info(f"CPU: {self.hardware_info.cpu_model} ({self.hardware_info.cpu_cores} cores)")
        self.logger.info(f"内存: {self.hardware_info.memory_total_mb}MB")
        
        if self.hardware_info.has_gpu:
            self.logger.info(f"GPU: {self.hardware_info.gpu_model} ({self.hardware_info.gpu_memory_mb}MB)")
            
        if self.hardware_info.has_npu:
            self.logger.info(f"NPU: {self.hardware_info.npu_model}")
            
        return self.hardware_info
        
    def _determine_capabilities(self, hw_info: HardwareInfo) -> PlatformCapabilities:
        """根据硬件信息确定平台能力"""
        platform_name = hw_info.platform_name.lower()
        
        # ESP32系列
        if 'esp32' in platform_name:
            if 's3' in platform_name:
                return PlatformCapabilities(
                    max_model_size_mb=5.0,
                    recommended_input_size=(160, 160),
                    supported_precisions=['int8'],
                    supported_formats=['tflite'],
                    max_batch_size=1,
                    recommended_threads=1,
                    memory_limit_mb=50,
                    power_limit_mw=600,
                    thermal_limit_celsius=85,
                    optimization_flags=['quantization', 'pruning']
                )
            else:
                return PlatformCapabilities(
                    max_model_size_mb=2.0,
                    recommended_input_size=(96, 96),
                    supported_precisions=['int8'],
                    supported_formats=['tflite_micro'],
                    max_batch_size=1,
                    recommended_threads=1,
                    memory_limit_mb=20,
                    power_limit_mw=500,
                    thermal_limit_celsius=85,
                    optimization_flags=['extreme_quantization', 'model_splitting']
                )
                
        # 树莓派系列
        elif 'raspberry pi' in platform_name:
            if 'zero' in platform_name:
                return PlatformCapabilities(
                    max_model_size_mb=25.0,
                    recommended_input_size=(320, 320),
                    supported_precisions=['fp16', 'int8'],
                    supported_formats=['onnx', 'tflite'],
                    max_batch_size=1,
                    recommended_threads=2,
                    memory_limit_mb=400,
                    power_limit_mw=2000,
                    thermal_limit_celsius=70,
                    optimization_flags=['onnx_optimization', 'dynamic_quantization']
                )
            elif '4' in platform_name:
                return PlatformCapabilities(
                    max_model_size_mb=100.0,
                    recommended_input_size=(416, 416),
                    supported_precisions=['fp32', 'fp16', 'int8'],
                    supported_formats=['onnx', 'tflite', 'pytorch'],
                    max_batch_size=2,
                    recommended_threads=4,
                    memory_limit_mb=3000,
                    power_limit_mw=3000,
                    thermal_limit_celsius=80,
                    can_use_gpu=hw_info.has_gpu,
                    optimization_flags=['gpu_acceleration', 'openvino']
                )
            else:
                return PlatformCapabilities(
                    max_model_size_mb=50.0,
                    recommended_input_size=(320, 320),
                    supported_precisions=['fp16', 'int8'],
                    supported_formats=['onnx', 'tflite'],
                    max_batch_size=1,
                    recommended_threads=4,
                    memory_limit_mb=800,
                    power_limit_mw=2500,
                    thermal_limit_celsius=75,
                    optimization_flags=['cpu_optimization']
                )
                
        # NVIDIA Jetson系列
        elif 'jetson' in platform_name:
            if 'nano' in platform_name:
                return PlatformCapabilities(
                    max_model_size_mb=200.0,
                    recommended_input_size=(640, 640),
                    supported_precisions=['fp32', 'fp16', 'int8'],
                    supported_formats=['onnx', 'tensorrt', 'pytorch'],
                    max_batch_size=4,
                    recommended_threads=4,
                    memory_limit_mb=3500,
                    power_limit_mw=5000,
                    thermal_limit_celsius=90,
                    can_use_gpu=True,
                    optimization_flags=['tensorrt', 'cuda', 'fp16_optimization']
                )
            else:
                return PlatformCapabilities(
                    max_model_size_mb=500.0,
                    recommended_input_size=(640, 640),
                    supported_precisions=['fp32', 'fp16', 'int8'],
                    supported_formats=['onnx', 'tensorrt', 'pytorch'],
                    max_batch_size=8,
                    recommended_threads=6,
                    memory_limit_mb=7000,
                    power_limit_mw=15000,
                    thermal_limit_celsius=95,
                    can_use_gpu=True,
                    optimization_flags=['tensorrt', 'cuda', 'multi_stream']
                )
                
        # Canaan K230
        elif 'k230' in platform_name or hw_info.has_npu:
            return PlatformCapabilities(
                max_model_size_mb=50.0,
                recommended_input_size=(416, 416),
                supported_precisions=['fp16', 'int8'],
                supported_formats=['onnx', 'nncase'],
                max_batch_size=2,
                recommended_threads=2,
                memory_limit_mb=400,
                power_limit_mw=2000,
                thermal_limit_celsius=85,
                can_use_npu=True,
                optimization_flags=['npu_optimization', 'nncase_quantization']
            )
            
        # 通用配置
        else:
            memory_limit = min(hw_info.memory_total_mb * 0.7, 2000)
            return PlatformCapabilities(
                max_model_size_mb=min(memory_limit * 0.3, 100.0),
                recommended_input_size=(416, 416),
                supported_precisions=['fp32', 'fp16'],
                supported_formats=['onnx', 'pytorch'],
                max_batch_size=max(1, hw_info.cpu_cores // 2),
                recommended_threads=hw_info.cpu_cores,
                memory_limit_mb=int(memory_limit),
                power_limit_mw=5000,
                thermal_limit_celsius=80,
                can_use_gpu=hw_info.has_gpu,
                optimization_flags=['cpu_optimization']
            )
            
    def get_optimal_config(self, model_size: str = "n") -> Dict[str, Any]:
        """获取最优配置"""
        if not self.capabilities:
            raise RuntimeError("适配器未初始化")
            
        # 根据模型大小调整配置
        size_multipliers = {
            'n': 1.0,
            's': 2.0,
            'm': 4.0,
            'l': 8.0,
            'x': 16.0
        }
        
        multiplier = size_multipliers.get(model_size, 1.0)
        
        # 调整输入尺寸
        base_w, base_h = self.capabilities.recommended_input_size
        if multiplier > 2.0 and self.capabilities.memory_limit_mb > 1000:
            input_size = (min(640, int(base_w * 1.2)), min(640, int(base_h * 1.2)))
        else:
            input_size = (base_w, base_h)
            
        # 选择精度
        if multiplier > 4.0 and 'fp32' in self.capabilities.supported_precisions:
            precision = 'fp32'
        elif 'fp16' in self.capabilities.supported_precisions:
            precision = 'fp16'
        else:
            precision = 'int8'
            
        # 选择格式
        if 'tensorrt' in self.capabilities.supported_formats and self.capabilities.can_use_gpu:
            model_format = 'tensorrt'
        elif 'onnx' in self.capabilities.supported_formats:
            model_format = 'onnx'
        else:
            model_format = self.capabilities.supported_formats[0]
            
        return {
            'platform_name': self.hardware_info.platform_name,
            'model_format': model_format,
            'precision': precision,
            'input_size': input_size,
            'batch_size': min(self.capabilities.max_batch_size, 2 if multiplier <= 2.0 else 1),
            'num_threads': self.capabilities.recommended_threads,
            'memory_limit_mb': self.capabilities.memory_limit_mb,
            'use_gpu': self.capabilities.can_use_gpu and model_format in ['onnx', 'tensorrt'],
            'use_npu': self.capabilities.can_use_npu,
            'optimization_flags': self.capabilities.optimization_flags,
            'confidence_threshold': 0.5,
            'nms_threshold': 0.4
        }
        
    def get_deployment_recommendations(self) -> List[str]:
        """获取部署建议"""
        if not self.capabilities:
            return []
            
        recommendations = []
        
        # 内存建议
        if self.capabilities.memory_limit_mb < 100:
            recommendations.extend([
                "使用INT8量化减少内存占用",
                "考虑模型分割或边缘-云协同架构",
                "启用动态内存管理"
            ])
        elif self.capabilities.memory_limit_mb < 500:
            recommendations.extend([
                "使用FP16精度平衡性能和内存",
                "启用模型缓存和预加载"
            ])
            
        # GPU建议
        if self.capabilities.can_use_gpu:
            recommendations.extend([
                "启用GPU加速提升推理速度",
                "使用TensorRT或OpenVINO优化"
            ])
            
        # NPU建议
        if self.capabilities.can_use_npu:
            recommendations.extend([
                "使用NPU专用优化获得最佳性能",
                "考虑NPU专用模型格式"
            ])
            
        # 功耗建议
        if self.capabilities.power_limit_mw < 1000:
            recommendations.extend([
                "启用动态频率调节",
                "实现智能休眠机制",
                "监控温度和功耗"
            ])
            
        return recommendations
        
    def monitor_system_health(self) -> Dict[str, Any]:
        """监控系统健康状态"""
        health = {
            'timestamp': time.time(),
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'temperature': 0.0,
            'available_memory_mb': 0,
            'disk_usage': 0.0,
            'status': 'unknown'
        }
        
        if PSUTIL_AVAILABLE:
            try:
                # CPU使用率
                health['cpu_usage'] = psutil.cpu_percent(interval=1)
                
                # 内存使用率
                memory = psutil.virtual_memory()
                health['memory_usage'] = memory.percent
                health['available_memory_mb'] = memory.available / 1024 / 1024
                
                # 磁盘使用率
                disk = psutil.disk_usage('/')
                health['disk_usage'] = (disk.used / disk.total) * 100
                
                # 温度 (如果可用)
                try:
                    temps = psutil.sensors_temperatures()
                    if temps:
                        # 获取CPU温度
                        for name, entries in temps.items():
                            if entries:
                                health['temperature'] = entries[0].current
                                break
                except:
                    pass
                    
            except Exception as e:
                self.logger.warning(f"系统监控错误: {e}")
                
        # 评估状态
        if health['memory_usage'] > 90 or health['cpu_usage'] > 95:
            health['status'] = 'critical'
        elif health['memory_usage'] > 80 or health['cpu_usage'] > 85:
            health['status'] = 'warning'
        else:
            health['status'] = 'healthy'
            
        return health

# 全局适配器实例
_global_adapter: Optional[PlatformAdapter] = None

def get_platform_adapter() -> PlatformAdapter:
    """获取平台适配器"""
    global _global_adapter
    
    if _global_adapter is None:
        _global_adapter = PlatformAdapter()
        _global_adapter.initialize()
        
    return _global_adapter

def detect_platform() -> str:
    """快速检测平台名称"""
    detector = PlatformDetector()
    return detector._detect_platform_name()

if __name__ == "__main__":
    # 测试代码
    import time
    
    print("嵌入式平台适配器测试")
    print("=" * 50)
    
    # 创建适配器
    adapter = PlatformAdapter()
    
    # 初始化并检测硬件
    hw_info = adapter.initialize()
    
    print(f"\n硬件信息:")
    print(f"  平台: {hw_info.platform_name}")
    print(f"  CPU: {hw_info.cpu_model}")
    print(f"  核心数: {hw_info.cpu_cores}")
    print(f"  频率: {hw_info.cpu_freq_mhz:.0f} MHz")
    print(f"  内存: {hw_info.memory_total_mb} MB")
    print(f"  存储: {hw_info.storage_total_gb:.1f} GB")
    print(f"  GPU: {hw_info.gpu_model if hw_info.has_gpu else '无'}")
    print(f"  NPU: {hw_info.npu_model if hw_info.has_npu else '无'}")
    
    # 获取最优配置
    print(f"\n最优配置 (YOLOv11n):")
    config = adapter.get_optimal_config('n')
    for key, value in config.items():
        print(f"  {key}: {value}")
        
    # 部署建议
    print(f"\n部署建议:")
    recommendations = adapter.get_deployment_recommendations()
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
        
    # 系统健康监控
    print(f"\n系统健康状态:")
    health = adapter.monitor_system_health()
    print(f"  CPU使用率: {health['cpu_usage']:.1f}%")
    print(f"  内存使用率: {health['memory_usage']:.1f}%")
    print(f"  可用内存: {health['available_memory_mb']:.0f} MB")
    print(f"  磁盘使用率: {health['disk_usage']:.1f}%")
    print(f"  温度: {health['temperature']:.1f}°C")
    print(f"  状态: {health['status']}")
    
    print("\n测试完成")