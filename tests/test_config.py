"""测试配置管理

为测试提供统一的配置管理，包括：
- 测试环境配置
- 插件测试配置
- 性能测试基准
- 模拟硬件配置
"""

import os
import tempfile
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class TestEnvironment:
    """测试环境配置"""
    temp_dir: str = field(default_factory=lambda: tempfile.mkdtemp(prefix='yolos_test_'))
    log_level: str = 'DEBUG'
    enable_gpu: bool = False
    mock_hardware: bool = True
    test_data_dir: str = field(default_factory=lambda: os.path.join(tempfile.gettempdir(), 'yolos_test_data'))
    
@dataclass
class PerformanceBenchmarks:
    """性能测试基准"""
    max_init_time: float = 5.0  # 最大初始化时间（秒）
    max_frame_processing_time: float = 0.1  # 最大帧处理时间（秒）
    max_memory_usage: float = 500.0  # 最大内存使用（MB）
    min_fps: float = 10.0  # 最小帧率
    max_cpu_usage: float = 80.0  # 最大CPU使用率（%）
    max_startup_time: float = 10.0  # 最大启动时间（秒）
    
@dataclass
class MockHardwareConfig:
    """模拟硬件配置"""
    camera_available: bool = True
    camera_resolution: tuple = (640, 480)
    camera_fps: int = 30
    gpio_pins: int = 40
    i2c_buses: int = 2
    spi_buses: int = 2
    uart_ports: int = 2
    wifi_available: bool = True
    bluetooth_available: bool = True
    
class TestConfig:
    """测试配置管理器"""
    
    def __init__(self):
        """初始化测试配置"""
        self.environment = TestEnvironment()
        self.benchmarks = PerformanceBenchmarks()
        self.mock_hardware = MockHardwareConfig()
        
        # 创建测试数据目录
        os.makedirs(self.environment.test_data_dir, exist_ok=True)
        
    def get_test_config(self) -> Dict[str, Any]:
        """获取基础测试配置"""
        return {
            'system': {
                'log_level': self.environment.log_level,
                'temp_dir': self.environment.temp_dir,
                'test_mode': True,
                'mock_hardware': self.environment.mock_hardware,
                'enable_gpu': self.environment.enable_gpu
            },
            'paths': {
                'data_dir': self.environment.test_data_dir,
                'model_dir': os.path.join(self.environment.test_data_dir, 'models'),
                'config_dir': os.path.join(self.environment.test_data_dir, 'configs'),
                'log_dir': os.path.join(self.environment.test_data_dir, 'logs')
            },
            'performance': {
                'max_init_time': self.benchmarks.max_init_time,
                'max_frame_processing_time': self.benchmarks.max_frame_processing_time,
                'max_memory_usage': self.benchmarks.max_memory_usage,
                'min_fps': self.benchmarks.min_fps,
                'max_cpu_usage': self.benchmarks.max_cpu_usage
            }
        }
        
    def get_plugin_config(self, plugin_type: str = 'generic') -> Dict[str, Any]:
        """获取插件测试配置
        
        Args:
            plugin_type: 插件类型 ('domain', 'platform', 'utility', 'generic')
            
        Returns:
            插件配置字典
        """
        base_config = {
            'enabled': True,
            'test_mode': True,
            'mock_data': True,
            'log_level': 'DEBUG',
            'timeout': 30.0
        }
        
        if plugin_type == 'domain':
            base_config.update(self._get_domain_plugin_config())
        elif plugin_type == 'platform':
            base_config.update(self._get_platform_plugin_config())
        elif plugin_type == 'utility':
            base_config.update(self._get_utility_plugin_config())
            
        return base_config
        
    def _get_domain_plugin_config(self) -> Dict[str, Any]:
        """获取领域插件配置"""
        return {
            'model': {
                'path': os.path.join(self.environment.test_data_dir, 'models', 'test_model.pt'),
                'input_size': (416, 416),
                'confidence_threshold': 0.5,
                'nms_threshold': 0.4,
                'max_objects': 10
            },
            'preprocessing': {
                'normalize': True,
                'resize': True,
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            },
            'postprocessing': {
                'filter_small_objects': True,
                'min_object_size': 20,
                'merge_overlapping': True
            },
            'classes': [
                'person', 'car', 'bicycle', 'dog', 'cat', 'bird',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra'
            ]
        }
        
    def _get_platform_plugin_config(self) -> Dict[str, Any]:
        """获取平台插件配置"""
        return {
            'hardware': {
                'camera': {
                    'device_id': 0,
                    'resolution': self.mock_hardware.camera_resolution,
                    'fps': self.mock_hardware.camera_fps,
                    'format': 'MJPG',
                    'mock': self.environment.mock_hardware
                },
                'gpio': {
                    'pins': self.mock_hardware.gpio_pins,
                    'mock': self.environment.mock_hardware
                },
                'i2c': {
                    'buses': self.mock_hardware.i2c_buses,
                    'mock': self.environment.mock_hardware
                },
                'spi': {
                    'buses': self.mock_hardware.spi_buses,
                    'mock': self.environment.mock_hardware
                }
            },
            'network': {
                'wifi': {
                    'enabled': self.mock_hardware.wifi_available,
                    'mock': self.environment.mock_hardware
                },
                'bluetooth': {
                    'enabled': self.mock_hardware.bluetooth_available,
                    'mock': self.environment.mock_hardware
                }
            }
        }
        
    def _get_utility_plugin_config(self) -> Dict[str, Any]:
        """获取工具插件配置"""
        return {
            'data_augmentation': {
                'enabled': True,
                'rotation_range': 15,
                'zoom_range': 0.1,
                'brightness_range': 0.2,
                'noise_level': 0.05
            },
            'performance_monitor': {
                'enabled': True,
                'sample_interval': 1.0,
                'memory_threshold': self.benchmarks.max_memory_usage,
                'cpu_threshold': self.benchmarks.max_cpu_usage
            },
            'data_storage': {
                'enabled': True,
                'storage_path': os.path.join(self.environment.test_data_dir, 'storage'),
                'max_size': 1024,  # MB
                'compression': True
            }
        }
        
    def get_integration_test_config(self) -> Dict[str, Any]:
        """获取集成测试配置"""
        return {
            'plugins': {
                'load_timeout': 10.0,
                'init_timeout': 15.0,
                'cleanup_timeout': 5.0,
                'max_concurrent': 5
            },
            'communication': {
                'event_timeout': 2.0,
                'message_queue_size': 100,
                'retry_attempts': 3
            },
            'data_flow': {
                'buffer_size': 10,
                'processing_timeout': 5.0,
                'batch_size': 5
            },
            'scenarios': {
                'basic_detection': {
                    'enabled': True,
                    'duration': 30.0,
                    'frame_rate': 10.0
                },
                'multi_plugin': {
                    'enabled': True,
                    'plugin_count': 3,
                    'duration': 60.0
                },
                'stress_test': {
                    'enabled': False,  # 默认禁用压力测试
                    'duration': 300.0,
                    'load_factor': 2.0
                }
            }
        }
        
    def get_performance_test_config(self) -> Dict[str, Any]:
        """获取性能测试配置"""
        return {
            'benchmarks': {
                'frame_processing': {
                    'target_fps': self.benchmarks.min_fps,
                    'max_processing_time': self.benchmarks.max_frame_processing_time,
                    'test_duration': 60.0,
                    'warmup_frames': 10
                },
                'memory_usage': {
                    'max_usage': self.benchmarks.max_memory_usage,
                    'leak_threshold': 50.0,  # MB
                    'test_duration': 120.0
                },
                'cpu_usage': {
                    'max_usage': self.benchmarks.max_cpu_usage,
                    'test_duration': 60.0,
                    'sample_interval': 1.0
                },
                'startup_time': {
                    'max_time': self.benchmarks.max_startup_time,
                    'iterations': 5
                }
            },
            'load_testing': {
                'concurrent_streams': [1, 2, 4, 8],
                'frame_rates': [10, 15, 30],
                'resolutions': [(640, 480), (1280, 720)],
                'test_duration': 30.0
            },
            'stress_testing': {
                'enabled': False,
                'max_load_factor': 5.0,
                'duration': 600.0,
                'ramp_up_time': 60.0
            }
        }
        
    def get_mock_data_config(self) -> Dict[str, Any]:
        """获取模拟数据配置"""
        return {
            'images': {
                'default_size': (640, 480),
                'formats': ['RGB', 'BGR', 'GRAY'],
                'noise_levels': [0.0, 0.1, 0.2],
                'batch_sizes': [1, 5, 10, 20]
            },
            'videos': {
                'default_fps': 30,
                'durations': [1.0, 5.0, 10.0],  # 秒
                'resolutions': [(640, 480), (1280, 720)]
            },
            'detections': {
                'object_counts': [0, 1, 5, 10, 20],
                'confidence_ranges': [(0.5, 1.0), (0.3, 0.8)],
                'class_counts': [5, 10, 20, 80]
            },
            'sensors': {
                'types': ['temperature', 'humidity', 'pressure', 'light'],
                'sample_rates': [1.0, 10.0, 100.0],  # Hz
                'durations': [1.0, 10.0, 60.0]  # 秒
            }
        }
        
    def create_test_model_file(self) -> str:
        """创建测试模型文件"""
        model_dir = os.path.join(self.environment.test_data_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'test_model.pt')
        
        # 创建一个空的模型文件（实际测试中会被mock替换）
        with open(model_path, 'wb') as f:
            f.write(b'mock_model_data')
            
        return model_path
        
    def create_test_config_file(self, config_data: Dict[str, Any], filename: str = 'test_config.yaml') -> str:
        """创建测试配置文件"""
        config_dir = os.path.join(self.environment.test_data_dir, 'configs')
        os.makedirs(config_dir, exist_ok=True)
        
        config_path = os.path.join(config_dir, filename)
        
        # 这里简化处理，实际应该使用yaml库
        import json
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
            
        return config_path
        
    def cleanup_test_data(self) -> None:
        """清理测试数据"""
        import shutil
        
        if os.path.exists(self.environment.test_data_dir):
            shutil.rmtree(self.environment.test_data_dir)
            
        if os.path.exists(self.environment.temp_dir):
            shutil.rmtree(self.environment.temp_dir)
            
    def get_test_dataset_config(self, dataset_type: str = 'detection') -> Dict[str, Any]:
        """获取测试数据集配置
        
        Args:
            dataset_type: 数据集类型 ('detection', 'classification', 'segmentation')
            
        Returns:
            数据集配置字典
        """
        base_config = {
            'name': f'test_{dataset_type}_dataset',
            'version': '1.0.0',
            'description': f'Test dataset for {dataset_type}',
            'path': os.path.join(self.environment.test_data_dir, 'datasets', dataset_type),
            'format': 'YOLO' if dataset_type == 'detection' else 'ImageNet',
            'split': {
                'train': 0.7,
                'val': 0.2,
                'test': 0.1
            }
        }
        
        if dataset_type == 'detection':
            base_config.update({
                'classes': [
                    'person', 'car', 'bicycle', 'dog', 'cat'
                ],
                'annotation_format': 'YOLO',
                'image_count': 100,
                'avg_objects_per_image': 2.5
            })
        elif dataset_type == 'classification':
            base_config.update({
                'classes': [
                    'cat', 'dog', 'bird', 'car', 'airplane'
                ],
                'images_per_class': 20,
                'image_size': (224, 224)
            })
        elif dataset_type == 'segmentation':
            base_config.update({
                'classes': [
                    'background', 'person', 'car', 'road', 'building'
                ],
                'annotation_format': 'mask',
                'image_count': 50,
                'mask_format': 'PNG'
            })
            
        return base_config
        
    def get_ci_config(self) -> Dict[str, Any]:
        """获取CI/CD测试配置"""
        return {
            'timeout': {
                'unit_tests': 300,  # 5分钟
                'integration_tests': 900,  # 15分钟
                'performance_tests': 1800,  # 30分钟
                'full_suite': 3600  # 1小时
            },
            'parallel': {
                'max_workers': 4,
                'unit_tests': True,
                'integration_tests': False,
                'performance_tests': False
            },
            'coverage': {
                'minimum': 80.0,
                'target': 90.0,
                'fail_under': 70.0
            },
            'artifacts': {
                'test_reports': True,
                'coverage_reports': True,
                'performance_reports': True,
                'log_files': True
            },
            'notifications': {
                'on_failure': True,
                'on_success': False,
                'channels': ['email', 'slack']
            }
        }