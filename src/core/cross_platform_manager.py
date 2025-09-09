#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨平台管理器 - 完整AIoT开发板支持
统一管理不同平台的配置和优化，包括主流AIoT开发板
"""

import os
import sys
import platform
import logging
import psutil
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import subprocess

logger = logging.getLogger(__name__)

class CrossPlatformManager:
    """跨平台管理器"""
    
    def __init__(self):
        self.platform_info = self._detect_platform()
        self.hardware_info = self._detect_hardware()
        self.aiot_board_info = self._detect_aiot_boards()
        self.optimization_config = self._get_optimization_config()
        
        logger.info(f"跨平台管理器初始化完成 - 平台: {self.platform_info['system']}")
        if self.aiot_board_info['detected']:
            logger.info(f"检测到AIoT开发板: {self.aiot_board_info['board_name']}")
    
    def _detect_platform(self) -> Dict[str, Any]:
        """检测平台信息"""
        system = platform.system().lower()
        
        platform_info = {
            'system': system,
            'architecture': platform.machine(),
            'python_version': platform.python_version(),
            'platform_release': platform.release(),
            'is_64bit': sys.maxsize > 2**32
        }
        
        # 特殊平台检测
        if system == 'linux':
            platform_info.update(self._detect_linux_variant())
        elif system == 'darwin':
            platform_info.update(self._detect_macos_variant())
        elif system == 'windows':
            platform_info.update(self._detect_windows_variant())
        
        # Arduino环境检测
        platform_info['arduino_support'] = self._detect_arduino_support()
        
        return platform_info
    
    def _detect_aiot_boards(self) -> Dict[str, Any]:
        """检测AIoT开发板"""
        aiot_info = {
            'detected': False,
            'board_name': 'Unknown',
            'board_type': 'generic',
            'confidence': 0.0,
            'ai_accelerator': None,
            'capabilities': {},
            'supported_frameworks': []
        }
        
        try:
            # 导入AIoT开发板适配器
            from ..plugins.platform.aiot_boards_adapter import get_aiot_boards_adapter
            
            adapter = get_aiot_boards_adapter()
            board_info = adapter.get_board_info()
            
            detected_board = board_info['detected_board']
            
            if detected_board['confidence'] > 0.5:
                aiot_info.update({
                    'detected': True,
                    'board_name': detected_board['name'],
                    'board_type': detected_board.get('id', 'unknown'),
                    'confidence': detected_board['confidence'],
                    'capabilities': detected_board.get('capabilities', {}),
                    'board_config': board_info.get('board_config', {})
                })
                
                # 提取AI加速器信息
                if 'info' in detected_board:
                    aiot_info['ai_accelerator'] = detected_board['info'].get('ai_accelerator')
                
                # 提取支持的框架
                ai_config = aiot_info.get('board_config', {}).get('ai_acceleration', {})
                aiot_info['supported_frameworks'] = ai_config.get('frameworks', [])
        
        except ImportError:
            logger.warning("AIoT开发板适配器未找到")
        except Exception as e:
            logger.error(f"AIoT开发板检测失败: {e}")
        
        return aiot_info
    
    def _detect_arduino_support(self) -> Dict[str, Any]:
        """检测Arduino支持"""
        arduino_info = {
            'arduino_ide_installed': False,
            'arduino_cli_installed': False,
            'serial_ports_available': [],
            'supported_boards': [],
            'pyserial_available': False
        }
        
        try:
            # 检测Arduino IDE
            if self.platform_info['system'] == 'windows':
                arduino_paths = [
                    'C:\\Program Files (x86)\\Arduino\\arduino.exe',
                    'C:\\Program Files\\Arduino\\arduino.exe',
                    os.path.expanduser('~\\AppData\\Local\\Arduino15\\arduino.exe')
                ]
            elif self.platform_info['system'] == 'darwin':
                arduino_paths = [
                    '/Applications/Arduino.app/Contents/MacOS/Arduino'
                ]
            else:  # Linux
                arduino_paths = [
                    '/usr/bin/arduino',
                    '/usr/local/bin/arduino',
                    os.path.expanduser('~/arduino-*/arduino')
                ]
            
            for path in arduino_paths:
                if os.path.exists(path):
                    arduino_info['arduino_ide_installed'] = True
                    arduino_info['arduino_ide_path'] = path
                    break
            
            # 检测Arduino CLI
            try:
                result = subprocess.run(['arduino-cli', 'version'], 
                                     capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    arduino_info['arduino_cli_installed'] = True
                    arduino_info['arduino_cli_version'] = result.stdout.strip()
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            
            # 检测pyserial
            try:
                import serial
                import serial.tools.list_ports
                arduino_info['pyserial_available'] = True
                
                # 获取可用串口
                ports = serial.tools.list_ports.comports()
                arduino_info['serial_ports_available'] = [
                    {
                        'device': port.device,
                        'description': port.description,
                        'hwid': port.hwid
                    }
                    for port in ports
                ]
                
                # 检测Arduino板子
                arduino_info['supported_boards'] = self._detect_arduino_boards(ports)
                
            except ImportError:
                logger.warning("pyserial未安装，Arduino串口功能不可用")
            
        except Exception as e:
            logger.error(f"Arduino支持检测失败: {e}")
        
        return arduino_info
    
    def _detect_arduino_boards(self, ports) -> List[Dict[str, Any]]:
        """检测Arduino板子"""
        arduino_boards = []
        
        # Arduino板子的USB VID/PID
        arduino_identifiers = [
            ('2341', '0043'),  # Arduino Uno
            ('2341', '0001'),  # Arduino Uno (older)
            ('2341', '0010'),  # Arduino Mega 2560
            ('2341', '0042'),  # Arduino Mega 2560 R3
            ('2341', '0243'),  # Arduino Uno R3
            ('1A86', '7523'),  # CH340 (clone boards)
            ('0403', '6001'),  # FTDI (some Arduino boards)
        ]
        
        for port in ports:
            hwid = port.hwid.upper()
            
            for vid, pid in arduino_identifiers:
                if f'VID_{vid}' in hwid and f'PID_{pid}' in hwid:
                    board_info = {
                        'port': port.device,
                        'description': port.description,
                        'vid': vid,
                        'pid': pid,
                        'board_type': self._identify_board_type(vid, pid)
                    }
                    arduino_boards.append(board_info)
                    break
        
        return arduino_boards
    
    def _identify_board_type(self, vid: str, pid: str) -> str:
        """识别Arduino板子类型"""
        board_map = {
            ('2341', '0043'): 'Arduino Uno',
            ('2341', '0001'): 'Arduino Uno (Legacy)',
            ('2341', '0010'): 'Arduino Mega 2560',
            ('2341', '0042'): 'Arduino Mega 2560 R3',
            ('2341', '0243'): 'Arduino Uno R3',
            ('1A86', '7523'): 'Arduino Compatible (CH340)',
            ('0403', '6001'): 'Arduino Compatible (FTDI)',
        }
        
        return board_map.get((vid, pid), 'Unknown Arduino Board')
    
    def _detect_linux_variant(self) -> Dict[str, Any]:
        """检测Linux变体"""
        linux_info = {}
        
        try:
            # 检测是否为树莓派
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                if 'Raspberry Pi' in cpuinfo or 'BCM' in cpuinfo:
                    linux_info['is_raspberry_pi'] = True
                    linux_info['raspberry_pi_model'] = self._get_raspberry_pi_model(cpuinfo)
                else:
                    linux_info['is_raspberry_pi'] = False
        except FileNotFoundError:
            linux_info['is_raspberry_pi'] = False
        
        # 检测发行版
        try:
            with open('/etc/os-release', 'r') as f:
                os_release = f.read()
                for line in os_release.split('\n'):
                    if line.startswith('ID='):
                        linux_info['distribution'] = line.split('=')[1].strip('"')
                    elif line.startswith('VERSION_ID='):
                        linux_info['version'] = line.split('=')[1].strip('"')
        except FileNotFoundError:
            linux_info['distribution'] = 'unknown'
        
        return linux_info
    
    def _get_raspberry_pi_model(self, cpuinfo: str) -> str:
        """获取树莓派型号"""
        if 'Pi 5' in cpuinfo:
            return 'Raspberry Pi 5'
        elif 'Pi 4' in cpuinfo:
            return 'Raspberry Pi 4'
        elif 'Pi 3' in cpuinfo:
            return 'Raspberry Pi 3'
        elif 'Pi 2' in cpuinfo:
            return 'Raspberry Pi 2'
        elif 'Pi Zero' in cpuinfo:
            return 'Raspberry Pi Zero'
        else:
            return 'Raspberry Pi (Unknown Model)'
    
    def _detect_macos_variant(self) -> Dict[str, Any]:
        """检测macOS变体"""
        macos_info = {}
        
        try:
            # 检测Apple Silicon
            result = subprocess.run(['uname', '-m'], capture_output=True, text=True)
            if result.returncode == 0:
                arch = result.stdout.strip()
                macos_info['is_apple_silicon'] = arch == 'arm64'
                macos_info['architecture_detail'] = arch
        except Exception:
            macos_info['is_apple_silicon'] = False
        
        return macos_info
    
    def _detect_windows_variant(self) -> Dict[str, Any]:
        """检测Windows变体"""
        windows_info = {}
        
        try:
            import winreg
            
            # 检测Windows版本
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                               r"SOFTWARE\Microsoft\Windows NT\CurrentVersion")
            
            try:
                windows_info['product_name'] = winreg.QueryValueEx(key, "ProductName")[0]
                windows_info['build_number'] = winreg.QueryValueEx(key, "CurrentBuild")[0]
            except FileNotFoundError:
                pass
            
            winreg.CloseKey(key)
            
        except ImportError:
            # 非Windows系统
            pass
        except Exception as e:
            logger.warning(f"Windows版本检测失败: {e}")
        
        return windows_info
    
    def _detect_hardware(self) -> Dict[str, Any]:
        """检测硬件信息"""
        hardware_info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
            'disk_usage': {}
        }
        
        # 磁盘使用情况
        try:
            disk_usage = psutil.disk_usage('/')
            hardware_info['disk_usage'] = {
                'total_gb': round(disk_usage.total / (1024**3), 2),
                'used_gb': round(disk_usage.used / (1024**3), 2),
                'free_gb': round(disk_usage.free / (1024**3), 2)
            }
        except Exception:
            pass
        
        # GPU检测
        hardware_info['gpu_info'] = self._detect_gpu()
        
        return hardware_info
    
    def _detect_gpu(self) -> Dict[str, Any]:
        """检测GPU信息"""
        gpu_info = {
            'cuda_available': False,
            'mps_available': False,  # Apple Metal Performance Shaders
            'opencl_available': False,
            'gpu_devices': []
        }
        
        try:
            import torch
            
            # CUDA检测
            if torch.cuda.is_available():
                gpu_info['cuda_available'] = True
                gpu_info['cuda_version'] = torch.version.cuda
                gpu_info['gpu_devices'] = [
                    {
                        'name': torch.cuda.get_device_name(i),
                        'memory_gb': round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2)
                    }
                    for i in range(torch.cuda.device_count())
                ]
            
            # MPS检测 (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                gpu_info['mps_available'] = True
                
        except ImportError:
            logger.info("PyTorch未安装，无法检测GPU支持")
        
        return gpu_info
    
    def _get_optimization_config(self) -> Dict[str, Any]:
        """获取平台优化配置"""
        config = {
            'max_workers': min(4, self.hardware_info['cpu_count']),
            'memory_limit_gb': max(1, self.hardware_info['memory_available_gb'] * 0.7),
            'use_gpu': False,
            'batch_size': 1,
            'image_max_size': (640, 480),
            'enable_multiprocessing': True
        }
        
        system = self.platform_info['system']
        
        # AIoT开发板优化（优先级最高）
        if self.aiot_board_info['detected']:
            aiot_config = self.aiot_board_info.get('board_config', {}).get('optimization', {})
            config.update(aiot_config)
            
            # AI加速器配置
            if self.aiot_board_info.get('capabilities', {}).get('edge_tpu', False):
                config['use_ai_accelerator'] = 'edge_tpu'
            elif self.aiot_board_info.get('capabilities', {}).get('npu_acceleration', False):
                config['use_ai_accelerator'] = 'npu'
            elif self.aiot_board_info.get('capabilities', {}).get('hexagon_dsp', False):
                config['use_ai_accelerator'] = 'hexagon_dsp'
        
        # 平台特定优化
        elif system == 'linux' and self.platform_info.get('is_raspberry_pi', False):
            # 树莓派优化
            config.update({
                'max_workers': 2,
                'memory_limit_gb': 0.5,
                'batch_size': 1,
                'image_max_size': (320, 240),
                'enable_multiprocessing': False,
                'use_lightweight_models': True
            })
        
        elif system == 'darwin' and self.platform_info.get('is_apple_silicon', False):
            # Apple Silicon优化
            config.update({
                'use_gpu': self.hardware_info['gpu_info']['mps_available'],
                'max_workers': min(8, self.hardware_info['cpu_count']),
                'batch_size': 4
            })
        
        elif system == 'windows':
            # Windows优化
            config.update({
                'use_gpu': self.hardware_info['gpu_info']['cuda_available'],
                'max_workers': min(6, self.hardware_info['cpu_count']),
                'batch_size': 2 if self.hardware_info['gpu_info']['cuda_available'] else 1
            })
        
        # Arduino环境特殊配置
        if self.platform_info['arduino_support']['pyserial_available']:
            config['arduino_config'] = {
                'enable_arduino_integration': True,
                'serial_timeout': 1.0,
                'max_image_size_arduino': (160, 120),
                'arduino_recognition_modes': ['color_detection', 'motion_detection', 'simple_object']
            }
        
        return config
    
    def get_platform_capabilities(self) -> Dict[str, Any]:
        """获取平台能力"""
        capabilities = {
            'camera_support': True,
            'gpu_acceleration': self.hardware_info['gpu_info']['cuda_available'] or 
                             self.hardware_info['gpu_info']['mps_available'],
            'multiprocessing': self.optimization_config['enable_multiprocessing'],
            'arduino_integration': self.platform_info['arduino_support']['pyserial_available'],
            'ros_support': self._check_ros_support(),
            'container_support': self._check_container_support(),
            'aiot_board_support': self.aiot_board_info['detected']
        }
        
        # AIoT开发板特殊能力
        if self.aiot_board_info['detected']:
            aiot_capabilities = self.aiot_board_info.get('capabilities', {})
            capabilities.update({
                'ai_accelerator': any(aiot_capabilities.get(key, False) 
                                    for key in ['edge_tpu', 'npu_acceleration', 'hexagon_dsp', 'apu_acceleration']),
                'csi_camera': aiot_capabilities.get('camera_csi', False),
                'mipi_camera': aiot_capabilities.get('camera_mipi', False),
                'gpio_control': aiot_capabilities.get('gpio', False),
                'low_power_mode': aiot_capabilities.get('ultra_low_power', False)
            })
        
        return capabilities
    
    def _check_ros_support(self) -> bool:
        """检查ROS支持"""
        try:
            # 检查ROS1
            if 'ROS_DISTRO' in os.environ:
                return True
            
            # 检查ROS2
            result = subprocess.run(['ros2', '--version'], 
                                 capture_output=True, text=True, timeout=5)
            return result.returncode == 0
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _check_container_support(self) -> bool:
        """检查容器支持"""
        try:
            # 检查Docker
            result = subprocess.run(['docker', '--version'], 
                                 capture_output=True, text=True, timeout=5)
            return result.returncode == 0
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def get_recommended_settings(self) -> Dict[str, Any]:
        """获取推荐设置"""
        settings = {
            'recognition_settings': {
                'max_concurrent_streams': self.optimization_config['max_workers'],
                'image_preprocessing': {
                    'max_size': self.optimization_config['image_max_size'],
                    'quality': 'medium' if self.hardware_info['memory_total_gb'] < 4 else 'high'
                },
                'model_settings': {
                    'use_lightweight': self.hardware_info['memory_total_gb'] < 2,
                    'batch_processing': self.optimization_config['batch_size'] > 1,
                    'gpu_acceleration': self.optimization_config['use_gpu']
                }
            },
            'arduino_settings': self.optimization_config.get('arduino_config', {}),
            'performance_settings': {
                'enable_caching': True,
                'cache_size_mb': min(512, int(self.hardware_info['memory_available_gb'] * 100)),
                'parallel_processing': self.optimization_config['enable_multiprocessing']
            }
        }
        
        # AIoT开发板特殊设置
        if self.aiot_board_info['detected']:
            settings['aiot_settings'] = {
                'board_name': self.aiot_board_info['board_name'],
                'ai_accelerator': self.aiot_board_info.get('ai_accelerator'),
                'supported_frameworks': self.aiot_board_info.get('supported_frameworks', []),
                'optimization_level': 'high' if self.aiot_board_info.get('capabilities', {}).get('high_performance', False) else 'medium'
            }
        
        return settings
    
    def install_platform_dependencies(self) -> Dict[str, bool]:
        """安装平台依赖"""
        results = {}
        
        try:
            # 基础依赖
            results['opencv'] = self._install_opencv()
            results['numpy'] = self._install_numpy()
            
            # Arduino依赖
            if self.platform_info['arduino_support']['pyserial_available'] or True:
                results['pyserial'] = self._install_pyserial()
            
            # AIoT开发板依赖
            if self.aiot_board_info['detected']:
                aiot_results = self._install_aiot_dependencies()
                results.update(aiot_results)
            
            # GPU依赖
            if self.hardware_info['gpu_info']['cuda_available']:
                results['torch_cuda'] = self._check_torch_cuda()
            
        except Exception as e:
            logger.error(f"平台依赖安装失败: {e}")
            results['error'] = str(e)
        
        return results
    
    def _install_aiot_dependencies(self) -> Dict[str, bool]:
        """安装AIoT开发板依赖"""
        results = {}
        
        try:
            from ..plugins.platform.aiot_boards_adapter import get_aiot_boards_adapter
            
            adapter = get_aiot_boards_adapter()
            install_results = adapter.install_board_dependencies()
            
            results.update(install_results)
            
        except ImportError:
            logger.warning("AIoT开发板适配器未找到")
            results['aiot_adapter'] = False
        except Exception as e:
            logger.error(f"AIoT依赖安装失败: {e}")
            results['aiot_error'] = str(e)
        
        return results
    
    def _install_opencv(self) -> bool:
        """安装OpenCV"""
        try:
            import cv2
            return True
        except ImportError:
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'opencv-python'], check=True)
                return True
            except subprocess.CalledProcessError:
                return False
    
    def _install_numpy(self) -> bool:
        """安装NumPy"""
        try:
            import numpy
            return True
        except ImportError:
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'numpy'], check=True)
                return True
            except subprocess.CalledProcessError:
                return False
    
    def _install_pyserial(self) -> bool:
        """安装PySerial"""
        try:
            import serial
            return True
        except ImportError:
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyserial'], check=True)
                return True
            except subprocess.CalledProcessError:
                return False
    
    def _check_torch_cuda(self) -> bool:
        """检查PyTorch CUDA支持"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def generate_platform_report(self) -> str:
        """生成平台报告"""
        report = f"""
# YOLOS 跨平台兼容性报告

## 系统信息
- 操作系统: {self.platform_info['system']} {self.platform_info.get('platform_release', '')}
- 架构: {self.platform_info['architecture']}
- Python版本: {self.platform_info['python_version']}

## 硬件信息
- CPU核心数: {self.hardware_info['cpu_count']} (逻辑: {self.hardware_info['cpu_count_logical']})
- 内存: {self.hardware_info['memory_total_gb']} GB (可用: {self.hardware_info['memory_available_gb']} GB)
- GPU支持: {'✓' if self.hardware_info['gpu_info']['cuda_available'] or self.hardware_info['gpu_info']['mps_available'] else '✗'}

## AIoT开发板支持
- 检测到开发板: {'✓' if self.aiot_board_info['detected'] else '✗'}
"""
        
        if self.aiot_board_info['detected']:
            report += f"""- 开发板名称: {self.aiot_board_info['board_name']}
- 检测置信度: {self.aiot_board_info['confidence']:.2f}
- AI加速器: {self.aiot_board_info.get('ai_accelerator', 'None')}
- 支持的框架: {', '.join(self.aiot_board_info.get('supported_frameworks', []))}
"""
        
        report += f"""
## Arduino支持
- Arduino IDE: {'✓' if self.platform_info['arduino_support']['arduino_ide_installed'] else '✗'}
- Arduino CLI: {'✓' if self.platform_info['arduino_support']['arduino_cli_installed'] else '✗'}
- PySerial: {'✓' if self.platform_info['arduino_support']['pyserial_available'] else '✗'}
- 可用串口: {len(self.platform_info['arduino_support']['serial_ports_available'])}
- 检测到的Arduino板: {len(self.platform_info['arduino_support']['supported_boards'])}

## 平台能力
- 摄像头支持: {'✓' if self.get_platform_capabilities()['camera_support'] else '✗'}
- GPU加速: {'✓' if self.get_platform_capabilities()['gpu_acceleration'] else '✗'}
- 多进程处理: {'✓' if self.get_platform_capabilities()['multiprocessing'] else '✗'}
- Arduino集成: {'✓' if self.get_platform_capabilities()['arduino_integration'] else '✗'}
- AIoT开发板: {'✓' if self.get_platform_capabilities()['aiot_board_support'] else '✗'}
- ROS支持: {'✓' if self.get_platform_capabilities()['ros_support'] else '✗'}

## 推荐配置
- 最大并发流: {self.get_recommended_settings()['recognition_settings']['max_concurrent_streams']}
- 图像最大尺寸: {self.get_recommended_settings()['recognition_settings']['image_preprocessing']['max_size']}
- 使用轻量模型: {'是' if self.get_recommended_settings()['recognition_settings']['model_settings']['use_lightweight'] else '否'}
- 批处理: {'是' if self.get_recommended_settings()['recognition_settings']['model_settings']['batch_processing'] else '否'}
"""
        
        return report

# 全局平台管理器实例
_platform_manager = None

def get_cross_platform_manager() -> CrossPlatformManager:
    """获取全局跨平台管理器"""
    global _platform_manager
    if _platform_manager is None:
        _platform_manager = CrossPlatformManager()
    return _platform_manager

if __name__ == "__main__":
    # 测试跨平台管理器
    manager = CrossPlatformManager()
    
    # 生成平台报告
    report = manager.generate_platform_report()
    print(report)
    
    # 检查AIoT开发板支持
    if manager.aiot_board_info['detected']:
        print(f"\n检测到AIoT开发板: {manager.aiot_board_info['board_name']}")
        print(f"置信度: {manager.aiot_board_info['confidence']:.2f}")
        print(f"支持的框架: {manager.aiot_board_info.get('supported_frameworks', [])}")
    
    # 安装依赖
    print(f"\n安装平台依赖...")
    install_results = manager.install_platform_dependencies()
    for dep, success in install_results.items():
        status = '✓' if success else '✗'
        print(f"{dep}: {status}")