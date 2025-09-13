#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试工具模块

提供通用的测试辅助功能：
- 测试数据生成
- 测试环境设置
- 测试结果验证
- 性能测量工具
- 模拟对象创建
"""

import os
import sys
import time
import json
import tempfile
import shutil
import random
import string
import threading
import multiprocessing
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import psutil
import numpy as np
from unittest.mock import Mock, MagicMock, patch

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class YOLOSTestMetrics:
    """测试指标"""
    start_time: float
    end_time: float
    duration: float
    memory_usage: Dict[str, float]
    cpu_usage: float
    disk_io: Dict[str, int]
    network_io: Dict[str, int]
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.process = psutil.Process()
        self.initial_memory = None
        self.initial_cpu_times = None
        self.initial_io_counters = None
        self.custom_metrics = {}
    
    def start(self):
        """开始监控"""
        self.start_time = time.time()
        self.initial_memory = self.process.memory_info()
        self.initial_cpu_times = self.process.cpu_times()
        try:
            self.initial_io_counters = self.process.io_counters()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            self.initial_io_counters = None
    
    def stop(self) -> YOLOSTestMetrics:
        """停止监控并返回指标"""
        self.end_time = time.time()
        
        # 计算持续时间
        duration = self.end_time - self.start_time if self.start_time else 0
        
        # 获取内存使用情况
        current_memory = self.process.memory_info()
        memory_usage = {
            'rss': current_memory.rss,
            'vms': current_memory.vms,
            'rss_delta': current_memory.rss - (self.initial_memory.rss if self.initial_memory else 0),
            'vms_delta': current_memory.vms - (self.initial_memory.vms if self.initial_memory else 0)
        }
        
        # 获取CPU使用情况
        current_cpu_times = self.process.cpu_times()
        cpu_usage = 0
        if self.initial_cpu_times:
            cpu_delta = (
                (current_cpu_times.user - self.initial_cpu_times.user) +
                (current_cpu_times.system - self.initial_cpu_times.system)
            )
            cpu_usage = (cpu_delta / duration) * 100 if duration > 0 else 0
        
        # 获取磁盘I/O情况
        disk_io = {'read_bytes': 0, 'write_bytes': 0, 'read_count': 0, 'write_count': 0}
        try:
            current_io_counters = self.process.io_counters()
            if self.initial_io_counters:
                disk_io = {
                    'read_bytes': current_io_counters.read_bytes - self.initial_io_counters.read_bytes,
                    'write_bytes': current_io_counters.write_bytes - self.initial_io_counters.write_bytes,
                    'read_count': current_io_counters.read_count - self.initial_io_counters.read_count,
                    'write_count': current_io_counters.write_count - self.initial_io_counters.write_count
                }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        
        # 网络I/O（简化版本，实际可能需要更复杂的实现）
        network_io = {'bytes_sent': 0, 'bytes_recv': 0, 'packets_sent': 0, 'packets_recv': 0}
        
        return YOLOSTestMetrics(
            start_time=self.start_time,
            end_time=self.end_time,
            duration=duration,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            disk_io=disk_io,
            network_io=network_io,
            custom_metrics=self.custom_metrics.copy()
        )
    
    def add_custom_metric(self, name: str, value: Any):
        """添加自定义指标"""
        self.custom_metrics[name] = value


@contextmanager
def performance_monitor():
    """性能监控上下文管理器"""
    monitor = PerformanceMonitor()
    monitor.start()
    try:
        yield monitor
    finally:
        metrics = monitor.stop()
        return metrics


class YOLOSTestDataGenerator:
    """测试数据生成器"""
    
    @staticmethod
    def random_string(length: int = 10, charset: str = None) -> str:
        """生成随机字符串"""
        if charset is None:
            charset = string.ascii_letters + string.digits
        return ''.join(random.choice(charset) for _ in range(length))
    
    @staticmethod
    def random_int(min_val: int = 0, max_val: int = 100) -> int:
        """生成随机整数"""
        return random.randint(min_val, max_val)
    
    @staticmethod
    def random_float(min_val: float = 0.0, max_val: float = 1.0) -> float:
        """生成随机浮点数"""
        return random.uniform(min_val, max_val)
    
    @staticmethod
    def random_bool() -> bool:
        """生成随机布尔值"""
        return random.choice([True, False])
    
    @staticmethod
    def random_list(length: int = 10, item_generator: Callable = None) -> List[Any]:
        """生成随机列表"""
        if item_generator is None:
            item_generator = lambda: TestDataGenerator.random_int()
        return [item_generator() for _ in range(length)]
    
    @staticmethod
    def random_dict(size: int = 5, key_generator: Callable = None, value_generator: Callable = None) -> Dict[str, Any]:
        """生成随机字典"""
        if key_generator is None:
            key_generator = lambda: TestDataGenerator.random_string(8)
        if value_generator is None:
            value_generator = lambda: TestDataGenerator.random_int()
        
        return {key_generator(): value_generator() for _ in range(size)}
    
    @staticmethod
    def random_image_data(width: int = 640, height: int = 480, channels: int = 3) -> np.ndarray:
        """生成随机图像数据"""
        return np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
    
    @staticmethod
    def random_detection_data(num_objects: int = 5, image_width: int = 640, image_height: int = 480) -> List[Dict[str, Any]]:
        """生成随机检测数据"""
        detections = []
        for _ in range(num_objects):
            x1 = random.randint(0, image_width - 100)
            y1 = random.randint(0, image_height - 100)
            x2 = random.randint(x1 + 10, min(x1 + 200, image_width))
            y2 = random.randint(y1 + 10, min(y1 + 200, image_height))
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': random.uniform(0.1, 1.0),
                'class_id': random.randint(0, 79),  # COCO classes
                'class_name': f'class_{random.randint(0, 79)}'
            })
        
        return detections


class MockObjectFactory:
    """模拟对象工厂"""
    
    @staticmethod
    def create_mock_model() -> Mock:
        """创建模拟模型"""
        mock_model = Mock()
        mock_model.predict.return_value = TestDataGenerator.random_detection_data()
        mock_model.load_weights.return_value = True
        mock_model.save_weights.return_value = True
        mock_model.evaluate.return_value = {
            'accuracy': TestDataGenerator.random_float(0.7, 0.95),
            'loss': TestDataGenerator.random_float(0.1, 0.5)
        }
        return mock_model
    
    @staticmethod
    def create_mock_dataset() -> Mock:
        """创建模拟数据集"""
        mock_dataset = Mock()
        mock_dataset.__len__.return_value = 1000
        mock_dataset.__getitem__.return_value = (
            TestDataGenerator.random_image_data(),
            TestDataGenerator.random_detection_data()
        )
        mock_dataset.get_annotations.return_value = [
            TestDataGenerator.random_detection_data() for _ in range(100)
        ]
        return mock_dataset
    
    @staticmethod
    def create_mock_detector() -> Mock:
        """创建模拟检测器"""
        mock_detector = Mock()
        mock_detector.detect.return_value = TestDataGenerator.random_detection_data()
        mock_detector.load_model.return_value = True
        mock_detector.is_loaded.return_value = True
        mock_detector.get_model_info.return_value = {
            'name': 'mock_model',
            'version': '1.0.0',
            'input_size': (640, 640),
            'num_classes': 80
        }
        return mock_detector
    
    @staticmethod
    def create_mock_logger() -> Mock:
        """创建模拟日志记录器"""
        mock_logger = Mock()
        mock_logger.info.return_value = None
        mock_logger.warning.return_value = None
        mock_logger.error.return_value = None
        mock_logger.debug.return_value = None
        mock_logger.log_performance.return_value = None
        mock_logger.create_debug_snapshot.return_value = {'snapshot_id': TestDataGenerator.random_string()}
        return mock_logger


class YOLOSTestEnvironmentManager:
    """测试环境管理器"""
    
    def __init__(self):
        self.temp_dirs = []
        self.temp_files = []
        self.original_env = {}
        self.patches = []
    
    def create_temp_dir(self, prefix: str = 'test_') -> Path:
        """创建临时目录"""
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def create_temp_file(self, suffix: str = '.tmp', content: str = None) -> Path:
        """创建临时文件"""
        fd, temp_file_path = tempfile.mkstemp(suffix=suffix)
        temp_file = Path(temp_file_path)
        
        if content:
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(content)
        else:
            os.close(fd)
        
        self.temp_files.append(temp_file)
        return temp_file
    
    def set_env_var(self, key: str, value: str):
        """设置环境变量"""
        if key not in self.original_env:
            self.original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    def patch_object(self, target: str, **kwargs) -> Mock:
        """打补丁"""
        patcher = patch(target, **kwargs)
        mock_obj = patcher.start()
        self.patches.append(patcher)
        return mock_obj
    
    def cleanup(self):
        """清理测试环境"""
        # 停止所有补丁
        for patcher in self.patches:
            try:
                patcher.stop()
            except Exception:
                pass
        self.patches.clear()
        
        # 恢复环境变量
        for key, original_value in self.original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value
        self.original_env.clear()
        
        # 删除临时文件
        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception:
                pass
        self.temp_files.clear()
        
        # 删除临时目录
        for temp_dir in self.temp_dirs:
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
            except Exception:
                pass
        self.temp_dirs.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class YOLOSTestValidator:
    """测试结果验证器"""
    
    @staticmethod
    def validate_detection_result(result: List[Dict[str, Any]], 
                                expected_keys: List[str] = None,
                                min_confidence: float = 0.0,
                                max_detections: int = None) -> bool:
        """验证检测结果"""
        if not isinstance(result, list):
            return False
        
        if max_detections and len(result) > max_detections:
            return False
        
        expected_keys = expected_keys or ['bbox', 'confidence', 'class_id']
        
        for detection in result:
            if not isinstance(detection, dict):
                return False
            
            # 检查必需的键
            for key in expected_keys:
                if key not in detection:
                    return False
            
            # 检查置信度
            if 'confidence' in detection:
                confidence = detection['confidence']
                if not isinstance(confidence, (int, float)) or confidence < min_confidence:
                    return False
            
            # 检查边界框格式
            if 'bbox' in detection:
                bbox = detection['bbox']
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    return False
                if not all(isinstance(coord, (int, float)) for coord in bbox):
                    return False
        
        return True
    
    @staticmethod
    def validate_performance_metrics(metrics: YOLOSTestMetrics,
                                   max_duration: float = None,
                                   max_memory_mb: float = None,
                                   max_cpu_percent: float = None) -> bool:
        """验证性能指标"""
        if max_duration and metrics.duration > max_duration:
            return False
        
        if max_memory_mb:
            memory_mb = metrics.memory_usage.get('rss', 0) / (1024 * 1024)
            if memory_mb > max_memory_mb:
                return False
        
        if max_cpu_percent and metrics.cpu_usage > max_cpu_percent:
            return False
        
        return True
    
    @staticmethod
    def validate_model_output(output: Any, expected_shape: Tuple = None, expected_type: type = None) -> bool:
        """验证模型输出"""
        if expected_type and not isinstance(output, expected_type):
            return False
        
        if expected_shape and hasattr(output, 'shape'):
            if output.shape != expected_shape:
                return False
        
        return True


class ConcurrencyTestHelper:
    """并发测试辅助工具"""
    
    @staticmethod
    def run_concurrent_tasks(tasks: List[Callable], max_workers: int = None) -> List[Any]:
        """并发运行任务"""
        import concurrent.futures
        
        max_workers = max_workers or min(len(tasks), multiprocessing.cpu_count())
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(task) for task in tasks]
            results = []
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(e)
            
            return results
    
    @staticmethod
    def stress_test(target_function: Callable, 
                   duration_seconds: float = 10.0,
                   num_threads: int = 4,
                   args: Tuple = (),
                   kwargs: Dict = None) -> Dict[str, Any]:
        """压力测试"""
        kwargs = kwargs or {}
        results = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'errors': [],
            'start_time': time.time(),
            'end_time': None,
            'duration': 0.0,
            'calls_per_second': 0.0
        }
        
        def worker():
            while time.time() - results['start_time'] < duration_seconds:
                try:
                    target_function(*args, **kwargs)
                    results['successful_calls'] += 1
                except Exception as e:
                    results['failed_calls'] += 1
                    results['errors'].append(str(e))
                finally:
                    results['total_calls'] += 1
        
        # 启动工作线程
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=worker)
            thread.start()
            threads.append(thread)
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        results['end_time'] = time.time()
        results['duration'] = results['end_time'] - results['start_time']
        results['calls_per_second'] = results['total_calls'] / results['duration'] if results['duration'] > 0 else 0
        
        return results


# 便捷函数
def create_test_image(width: int = 640, height: int = 480, channels: int = 3) -> np.ndarray:
    """创建测试图像"""
    return YOLOSTestDataGenerator.random_image_data(width, height, channels)


def create_test_detections(num_objects: int = 5, image_width: int = 640, image_height: int = 480) -> List[Dict[str, Any]]:
    """创建测试检测结果"""
    return YOLOSTestDataGenerator.random_detection_data(num_objects, image_width, image_height)


def measure_execution_time(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """测量函数执行时间"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


def assert_performance(func: Callable, max_duration: float = 1.0, *args, **kwargs):
    """断言性能要求"""
    result, duration = measure_execution_time(func, *args, **kwargs)
    assert duration <= max_duration, f"函数执行时间 {duration:.3f}s 超过了最大允许时间 {max_duration}s"
    return result


def retry_on_failure(func: Callable, max_retries: int = 3, delay: float = 1.0, *args, **kwargs):
    """失败时重试"""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(delay)
    
    return None


@contextmanager
def temporary_directory(prefix: str = 'test_'):
    """临时目录上下文管理器"""
    temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield temp_dir
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


@contextmanager
def temporary_file(suffix: str = '.tmp', content: str = None):
    """临时文件上下文管理器"""
    fd, temp_file_path = tempfile.mkstemp(suffix=suffix)
    temp_file = Path(temp_file_path)
    
    try:
        if content:
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(content)
        else:
            os.close(fd)
        
        yield temp_file
    finally:
        if temp_file.exists():
            temp_file.unlink()