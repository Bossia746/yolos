"""测试配置管理

为测试提供统一的配置管理，包括：
- 测试环境配置
- 插件测试配置
- 性能测试基准
- 模拟硬件配置
- 集成测试配置
- 基准测试配置
- 结构化日志测试配置
"""

import os
import tempfile
import json
import yaml
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

@dataclass
class YOLOSTestEnvironment:
    """测试环境配置"""
    temp_dir: str = field(default_factory=lambda: tempfile.mkdtemp(prefix='yolos_test_'))
    log_level: str = 'DEBUG'
    enable_gpu: bool = False
    mock_hardware: bool = True
    test_data_dir: str = field(default_factory=lambda: os.path.join(tempfile.gettempdir(), 'yolos_test_data'))
    
@dataclass
class YOLOSPerformanceBenchmarks:
    """性能测试基准"""
    max_init_time: float = 5.0  # 最大初始化时间（秒）
    max_frame_processing_time: float = 0.1  # 最大帧处理时间（秒）
    max_memory_usage: float = 500.0  # 最大内存使用（MB）
    min_fps: float = 10.0  # 最小帧率
    max_cpu_usage: float = 80.0  # 最大CPU使用率（%）
    max_startup_time: float = 10.0  # 最大启动时间（秒）
    
    # 新增基准指标
    max_inference_time: float = 50.0  # 最大推理时间（毫秒）
    min_throughput: float = 20.0  # 最小吞吐量（FPS）
    max_latency: float = 100.0  # 最大延迟（毫秒）
    memory_leak_threshold: float = 10.0  # 内存泄漏阈值（MB）
    cpu_efficiency_threshold: float = 0.8  # CPU效率阈值
    
@dataclass
class YOLOSBenchmarkTestConfig:
    """基准测试配置"""
    name: str
    target_metric: str
    baseline_value: float
    tolerance_percent: float = 10.0
    warmup_iterations: int = 5
    measurement_iterations: int = 20
    timeout_seconds: float = 300.0
    
@dataclass
class YOLOSIntegrationTestConfig:
    """集成测试配置"""
    name: str
    modules: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    timeout: float = 120.0
    retry_on_failure: bool = True
    parallel_execution: bool = False
    
@dataclass
class YOLOSStructuredLogTestConfig:
    """结构化日志测试配置"""
    log_format: str = "json"
    log_level: str = "DEBUG"
    test_events: List[str] = field(default_factory=lambda: [
        "test_start", "test_end", "assertion_pass", "assertion_fail", 
        "performance_metric", "error_occurred"
    ])
    metrics_collection: bool = True
    audit_logging: bool = True
    
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
    
class YOLOSTestConfig:
    """测试配置管理器"""
    
    def __init__(self):
        """初始化测试配置"""
        self.environment = YOLOSTestEnvironment()
        self.benchmarks = YOLOSPerformanceBenchmarks()
        self.mock_hardware = MockHardwareConfig()
        
        # 新增配置
        self.benchmark_tests: Dict[str, YOLOSBenchmarkTestConfig] = {}
        self.integration_tests: Dict[str, YOLOSIntegrationTestConfig] = {}
        self.structured_log_config = YOLOSStructuredLogTestConfig()
        
        # 创建测试数据目录
        os.makedirs(self.environment.test_data_dir, exist_ok=True)
        
        # 初始化默认基准测试配置
        self._setup_default_benchmark_configs()
        self._setup_default_integration_configs()
    
    def _setup_default_benchmark_configs(self):
        """设置默认基准测试配置"""
        self.benchmark_tests = {
            'inference_speed': YOLOSBenchmarkTestConfig(
                name='inference_speed',
                target_metric='inference_time_ms',
                baseline_value=50.0,
                tolerance_percent=15.0,
                warmup_iterations=10,
                measurement_iterations=50
            ),
            'memory_usage': YOLOSBenchmarkTestConfig(
                name='memory_usage',
                target_metric='peak_memory_mb',
                baseline_value=256.0,
                tolerance_percent=20.0,
                measurement_iterations=30
            ),
            'throughput': YOLOSBenchmarkTestConfig(
                name='throughput',
                target_metric='fps',
                baseline_value=30.0,
                tolerance_percent=10.0,
                measurement_iterations=25
            ),
            'startup_time': YOLOSBenchmarkTestConfig(
                name='startup_time',
                target_metric='startup_ms',
                baseline_value=2000.0,
                tolerance_percent=25.0,
                measurement_iterations=10
            )
        }
    
    def _setup_default_integration_configs(self):
        """设置默认集成测试配置"""
        self.integration_tests = {
            'core_modules': YOLOSIntegrationTestConfig(
                name='core_modules',
                modules=['logger', 'config', 'exception_handler', 'performance_monitor'],
                timeout=60.0
            ),
            'logging_system': YOLOSIntegrationTestConfig(
                name='logging_system',
                modules=['structured_logger', 'performance_logger', 'audit_logger'],
                dependencies=['core_modules'],
                timeout=45.0
            ),
            'end_to_end_workflow': YOLOSIntegrationTestConfig(
                name='end_to_end_workflow',
                modules=['all'],
                dependencies=['core_modules', 'logging_system'],
                timeout=180.0,
                retry_on_failure=True
            )
        }
        
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
    
    def get_benchmark_test_config(self, name: str) -> Optional[YOLOSBenchmarkTestConfig]:
        """获取基准测试配置"""
        return self.benchmark_tests.get(name)
    
    def get_integration_test_config(self, name: str) -> Optional[YOLOSIntegrationTestConfig]:
        """获取集成测试配置"""
        return self.integration_tests.get(name)
    
    def get_structured_log_config(self) -> YOLOSStructuredLogTestConfig:
        """获取结构化日志测试配置"""
        return self.structured_log_config
    
    def add_benchmark_test(self, config: YOLOSBenchmarkTestConfig):
        """添加基准测试配置"""
        self.benchmark_tests[config.name] = config
    
    def add_integration_test(self, config: YOLOSIntegrationTestConfig):
        """添加集成测试配置"""
        self.integration_tests[config.name] = config
    
    def get_enhanced_performance_config(self) -> Dict[str, Any]:
        """获取增强性能测试配置"""
        return {
            'benchmarks': {
                name: {
                    'target_metric': config.target_metric,
                    'baseline_value': config.baseline_value,
                    'tolerance_percent': config.tolerance_percent,
                    'warmup_iterations': config.warmup_iterations,
                    'measurement_iterations': config.measurement_iterations,
                    'timeout_seconds': config.timeout_seconds
                }
                for name, config in self.benchmark_tests.items()
            },
            'system_metrics': {
                'cpu_monitoring': True,
                'memory_monitoring': True,
                'disk_io_monitoring': True,
                'network_monitoring': True,
                'gpu_monitoring': self.environment.enable_gpu
            },
            'performance_thresholds': {
                'max_inference_time': self.benchmarks.max_inference_time,
                'min_throughput': self.benchmarks.min_throughput,
                'max_latency': self.benchmarks.max_latency,
                'memory_leak_threshold': self.benchmarks.memory_leak_threshold,
                'cpu_efficiency_threshold': self.benchmarks.cpu_efficiency_threshold
            },
            'test_scenarios': {
                'single_thread': {
                    'enabled': True,
                    'duration': 60.0,
                    'load_factor': 1.0
                },
                'multi_thread': {
                    'enabled': True,
                    'duration': 120.0,
                    'thread_count': 4,
                    'load_factor': 2.0
                },
                'stress_test': {
                    'enabled': False,
                    'duration': 300.0,
                    'load_factor': 5.0,
                    'ramp_up_time': 30.0
                }
            }
        }
    
    def get_enhanced_integration_config(self) -> Dict[str, Any]:
        """获取增强集成测试配置"""
        return {
            'test_suites': {
                name: {
                    'modules': config.modules,
                    'dependencies': config.dependencies,
                    'timeout': config.timeout,
                    'retry_on_failure': config.retry_on_failure,
                    'parallel_execution': config.parallel_execution
                }
                for name, config in self.integration_tests.items()
            },
            'workflow_testing': {
                'enabled': True,
                'max_workflow_duration': 300.0,
                'step_timeout': 30.0,
                'failure_recovery': True
            },
            'cross_platform_testing': {
                'enabled': True,
                'platforms': ['windows', 'linux', 'macos'],
                'compatibility_checks': True
            },
            'error_recovery_testing': {
                'enabled': True,
                'exception_scenarios': [
                    'network_failure',
                    'resource_exhaustion',
                    'invalid_input',
                    'timeout_error'
                ],
                'recovery_timeout': 10.0
            }
        }
    
    def get_structured_logging_test_config(self) -> Dict[str, Any]:
        """获取结构化日志测试配置"""
        return {
            'log_format': self.structured_log_config.log_format,
            'log_level': self.structured_log_config.log_level,
            'test_events': self.structured_log_config.test_events,
            'metrics_collection': self.structured_log_config.metrics_collection,
            'audit_logging': self.structured_log_config.audit_logging,
            'log_validation': {
                'schema_validation': True,
                'field_validation': True,
                'timestamp_validation': True,
                'correlation_id_validation': True
            },
            'performance_logging': {
                'enabled': True,
                'metrics': [
                    'execution_time',
                    'memory_usage',
                    'cpu_usage',
                    'throughput'
                ],
                'sampling_rate': 1.0
            },
            'audit_trail': {
                'enabled': True,
                'events': [
                    'test_execution',
                    'configuration_change',
                    'error_occurrence',
                    'performance_threshold_breach'
                ],
                'retention_days': 30
            }
        }
    
    def export_test_config(self, output_file: str):
        """导出测试配置到文件"""
        config_data = {
            'environment': {
                'temp_dir': self.environment.temp_dir,
                'log_level': self.environment.log_level,
                'enable_gpu': self.environment.enable_gpu,
                'mock_hardware': self.environment.mock_hardware,
                'test_data_dir': self.environment.test_data_dir
            },
            'benchmarks': {
                name: {
                    'target_metric': config.target_metric,
                    'baseline_value': config.baseline_value,
                    'tolerance_percent': config.tolerance_percent,
                    'warmup_iterations': config.warmup_iterations,
                    'measurement_iterations': config.measurement_iterations,
                    'timeout_seconds': config.timeout_seconds
                }
                for name, config in self.benchmark_tests.items()
            },
            'integration_tests': {
                name: {
                    'modules': config.modules,
                    'dependencies': config.dependencies,
                    'timeout': config.timeout,
                    'retry_on_failure': config.retry_on_failure,
                    'parallel_execution': config.parallel_execution
                }
                for name, config in self.integration_tests.items()
            },
            'structured_logging': {
                'log_format': self.structured_log_config.log_format,
                'log_level': self.structured_log_config.log_level,
                'test_events': self.structured_log_config.test_events,
                'metrics_collection': self.structured_log_config.metrics_collection,
                'audit_logging': self.structured_log_config.audit_logging
            }
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                if output_file.endswith('.yaml') or output_file.endswith('.yml'):
                    yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
                else:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"导出配置失败: {e}")
    
    def validate_test_config(self) -> List[str]:
        """验证测试配置"""
        errors = []
        
        # 验证基准测试配置
        for name, config in self.benchmark_tests.items():
            if config.baseline_value <= 0:
                errors.append(f"基准测试 {name} 的基线值必须大于0")
            if config.tolerance_percent < 0:
                errors.append(f"基准测试 {name} 的容差百分比不能为负数")
            if config.measurement_iterations <= 0:
                errors.append(f"基准测试 {name} 的测量迭代次数必须大于0")
        
        # 验证集成测试配置
        for name, config in self.integration_tests.items():
            if config.timeout <= 0:
                errors.append(f"集成测试 {name} 的超时时间必须大于0")
            if not config.modules:
                errors.append(f"集成测试 {name} 必须指定至少一个模块")
        
        # 验证环境配置
        if not os.path.exists(self.environment.test_data_dir):
            errors.append(f"测试数据目录不存在: {self.environment.test_data_dir}")
        
        return errors
    
    def get_all_benchmark_names(self) -> List[str]:
        """获取所有基准测试名称"""
        return list(self.benchmark_tests.keys())
    
    def get_all_integration_test_names(self) -> List[str]:
        """获取所有集成测试名称"""
        return list(self.integration_tests.keys())


# 全局测试配置实例
test_config = YOLOSTestConfig()