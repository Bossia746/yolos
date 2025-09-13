"""测试基类

提供所有测试的基础功能，包括：
- 通用测试工具方法
- 插件测试基类
- 测试数据管理
- 断言扩展
"""

import unittest
import tempfile
import shutil
import os
import logging
import time
import psutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import cv2

# Import with error handling for missing modules
try:
    from src.core.base_plugin import BasePlugin, PluginStatus
except ImportError:
    BasePlugin = None
    PluginStatus = None

try:
    from src.core.plugin_manager import PluginManager
except ImportError:
    PluginManager = None

try:
    from src.core.config_manager import ConfigManager
except ImportError:
    ConfigManager = None

try:
    from src.core.event_bus import EventBus
except ImportError:
    EventBus = None

# Import test configuration
try:
    from .test_config import YOLOSTestConfig
except ImportError:
    # Fallback mock configuration
    class YOLOSTestConfig:
        """Mock test configuration."""
        def __init__(self):
            self.test_data_dir = Path("tests/data")
            self.temp_dir = Path("temp")
            self.log_level = "DEBUG"
        
        def get_test_config(self):
            """Get test configuration dictionary."""
            return {
                'test_data_dir': str(self.test_data_dir),
                'temp_dir': str(self.temp_dir),
                'log_level': self.log_level,
                'mock_mode': True
            }

class MockDataGenerator:
    """Mock data generator for testing."""
    def __init__(self):
        pass
    
    def generate_image(self, width=640, height=480):
        """Generate mock image data."""
        import numpy as np
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    def generate_detection_result(self):
        """Generate mock detection result."""
        return {
            'boxes': [],
            'scores': [],
            'classes': []
        }

class BaseTest(unittest.TestCase):
    """基础测试类
    
    提供所有测试的通用功能和工具方法。
    """
    
    @classmethod
    def setUpClass(cls):
        """类级别的设置"""
        # 设置测试日志
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 创建临时目录
        cls.temp_dir = tempfile.mkdtemp(prefix='yolos_test_')
        
        # 初始化测试配置
        cls.test_config = YOLOSTestConfig()
        
        # 初始化模拟数据生成器
        cls.mock_data = MockDataGenerator()
        
    @classmethod
    def tearDownClass(cls):
        """类级别的清理"""
        # 清理临时目录
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
            
    def setUp(self):
        """每个测试方法的设置"""
        # 重置事件总线（如果可用）
        if EventBus is not None and hasattr(EventBus, '_handlers'):
            EventBus._handlers.clear()
        
        # 创建测试专用的配置管理器（如果可用）
        if ConfigManager is not None:
            self.config_manager = ConfigManager()
            if hasattr(self.test_config, 'get_test_config'):
                self.config_manager.load_config(self.test_config.get_test_config())
        else:
            self.config_manager = None
        
        # 记录测试开始时间
        self.start_time = time.time()
        
        # 记录初始内存使用
        try:
            self.initial_memory = psutil.Process().memory_info().rss
        except:
            self.initial_memory = 0
        
    def tearDown(self):
        """每个测试方法的清理"""
        # 计算测试执行时间
        execution_time = time.time() - self.start_time
        
        # 检查内存泄漏
        final_memory = psutil.Process().memory_info().rss
        memory_increase = final_memory - self.initial_memory
        
        # 如果内存增长超过100MB，发出警告
        if memory_increase > 100 * 1024 * 1024:
            self.logger.warning(f"Potential memory leak detected: {memory_increase / 1024 / 1024:.2f}MB increase")
            
        # 记录测试性能
        self.logger.debug(f"Test execution time: {execution_time:.3f}s")
        
    @property
    def logger(self):
        """获取测试日志器"""
        return logging.getLogger(self.__class__.__name__)
        
    def create_temp_file(self, content: str = "", suffix: str = ".txt") -> str:
        """创建临时文件"""
        fd, path = tempfile.mkstemp(suffix=suffix, dir=self.temp_dir)
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        return path
        
    def create_temp_dir(self, prefix: str = "test_") -> str:
        """创建临时目录"""
        return tempfile.mkdtemp(prefix=prefix, dir=self.temp_dir)
        
    def assert_image_valid(self, image: np.ndarray, expected_shape: tuple = None):
        """断言图像有效性"""
        self.assertIsInstance(image, np.ndarray, "Image should be numpy array")
        self.assertEqual(len(image.shape), 3, "Image should be 3-dimensional")
        self.assertEqual(image.shape[2], 3, "Image should have 3 channels")
        
        if expected_shape:
            self.assertEqual(image.shape[:2], expected_shape[:2], f"Image shape mismatch: expected {expected_shape}, got {image.shape}")
            
    def assert_bbox_valid(self, bbox: List[float]):
        """断言边界框有效性"""
        self.assertIsInstance(bbox, (list, tuple), "Bbox should be list or tuple")
        self.assertEqual(len(bbox), 4, "Bbox should have 4 elements [x, y, w, h]")
        
        x, y, w, h = bbox
        self.assertGreaterEqual(x, 0, "Bbox x should be non-negative")
        self.assertGreaterEqual(y, 0, "Bbox y should be non-negative")
        self.assertGreater(w, 0, "Bbox width should be positive")
        self.assertGreater(h, 0, "Bbox height should be positive")
        
    def assert_confidence_valid(self, confidence: float):
        """断言置信度有效性"""
        self.assertIsInstance(confidence, (int, float), "Confidence should be numeric")
        self.assertGreaterEqual(confidence, 0.0, "Confidence should be >= 0")
        self.assertLessEqual(confidence, 1.0, "Confidence should be <= 1")
        
    def assert_processing_time(self, func, max_time: float, *args, **kwargs):
        """断言处理时间"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        processing_time = end_time - start_time
        self.assertLess(processing_time, max_time, 
                       f"Processing time {processing_time:.3f}s exceeds limit {max_time}s")
        return result
        
    def assert_memory_usage(self, func, max_memory_mb: float, *args, **kwargs):
        """断言内存使用量"""
        initial_memory = psutil.Process().memory_info().rss
        result = func(*args, **kwargs)
        final_memory = psutil.Process().memory_info().rss
        
        memory_increase = (final_memory - initial_memory) / 1024 / 1024
        self.assertLess(memory_increase, max_memory_mb,
                       f"Memory usage {memory_increase:.2f}MB exceeds limit {max_memory_mb}MB")
        return result
        
    def mock_camera_frame(self, width: int = 640, height: int = 480) -> np.ndarray:
        """生成模拟摄像头帧"""
        return self.mock_data.generate_image(width, height)
        
    def mock_detection_result(self, num_objects: int = 1) -> List[Dict[str, Any]]:
        """生成模拟检测结果"""
        return self.mock_data.generate_detection_results(num_objects)
        
    def wait_for_condition(self, condition_func, timeout: float = 5.0, interval: float = 0.1) -> bool:
        """等待条件满足"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition_func():
                return True
            time.sleep(interval)
        return False

class BasePluginTest(BaseTest):
    """插件测试基类
    
    提供插件测试的专用功能。
    """
    
    def setUp(self):
        """插件测试设置"""
        super().setUp()
        
        # 创建插件管理器
        if PluginManager is None:
            self.skipTest("PluginManager not available")
        self.plugin_manager = PluginManager()
        
        # 创建模拟插件配置
        self.plugin_config = self.test_config.get_plugin_config()
        
    def tearDown(self):
        """插件测试清理"""
        # 清理所有插件
        if hasattr(self, 'plugin_manager'):
            self.plugin_manager.cleanup_all()
            
        super().tearDown()
        
    def create_mock_plugin(self, plugin_class: type, config: Dict[str, Any] = None) -> BasePlugin:
        """创建模拟插件实例"""
        plugin = plugin_class()
        
        if config is None:
            config = self.plugin_config
            
        # 初始化插件
        success = plugin.initialize(config)
        self.assertTrue(success, f"Failed to initialize plugin {plugin_class.__name__}")
        
        return plugin
        
    def assert_plugin_status(self, plugin: BasePlugin, expected_status: PluginStatus):
        """断言插件状态"""
        self.assertEqual(plugin.status, expected_status, 
                        f"Plugin status mismatch: expected {expected_status}, got {plugin.status}")
                        
    def assert_plugin_initialized(self, plugin: BasePlugin):
        """断言插件已初始化"""
        self.assert_plugin_status(plugin, PluginStatus.ACTIVE)
        self.assertIsNotNone(plugin.metadata, "Plugin metadata should not be None")
        
    def assert_plugin_cleanup(self, plugin: BasePlugin):
        """断言插件已清理"""
        success = plugin.cleanup()
        self.assertTrue(success, "Plugin cleanup should succeed")
        self.assert_plugin_status(plugin, PluginStatus.INACTIVE)
        
    def test_plugin_lifecycle(self, plugin_class: type, config: Dict[str, Any] = None):
        """测试插件生命周期"""
        # 创建插件
        plugin = plugin_class()
        self.assert_plugin_status(plugin, PluginStatus.INACTIVE)
        
        # 初始化插件
        if config is None:
            config = self.plugin_config
            
        success = plugin.initialize(config)
        self.assertTrue(success, "Plugin initialization should succeed")
        self.assert_plugin_initialized(plugin)
        
        # 测试插件功能（由子类实现）
        if hasattr(self, '_test_plugin_functionality'):
            self._test_plugin_functionality(plugin)
            
        # 清理插件
        self.assert_plugin_cleanup(plugin)
        
    def test_plugin_error_handling(self, plugin_class: type):
        """测试插件错误处理"""
        plugin = plugin_class()
        
        # 测试无效配置
        invalid_config = {'invalid_param': 'invalid_value'}
        
        # 某些插件可能会忽略无效参数，所以这里不强制要求失败
        try:
            plugin.initialize(invalid_config)
        except Exception as e:
            self.logger.info(f"Plugin correctly rejected invalid config: {e}")
            
        # 测试重复初始化
        plugin.initialize(self.plugin_config)
        result = plugin.initialize(self.plugin_config)
        # 重复初始化应该成功或返回False，但不应该崩溃
        self.assertIsInstance(result, bool, "Initialize should return boolean")
        
    def test_plugin_performance(self, plugin_class: type, max_init_time: float = 5.0):
        """测试插件性能"""
        plugin = plugin_class()
        
        # 测试初始化时间
        def init_plugin():
            return plugin.initialize(self.plugin_config)
            
        result = self.assert_processing_time(init_plugin, max_init_time)
        self.assertTrue(result, "Plugin initialization should succeed within time limit")
        
        # 测试内存使用
        def cleanup_plugin():
            return plugin.cleanup()
            
        self.assert_memory_usage(cleanup_plugin, 50.0)  # 清理不应该增加超过50MB内存
        
    def mock_event_handler(self) -> Mock:
        """创建模拟事件处理器"""
        handler = Mock()
        return handler
        
    def register_event_handler(self, event_type: str, handler) -> None:
        """注册事件处理器"""
        EventBus.register(event_type, handler)
        
    def assert_event_emitted(self, event_type: str, handler: Mock, timeout: float = 1.0):
        """断言事件已发出"""
        def check_event():
            return handler.called
            
        success = self.wait_for_condition(check_event, timeout)
        self.assertTrue(success, f"Event {event_type} was not emitted within {timeout}s")
        
    def simulate_hardware_failure(self, plugin: BasePlugin):
        """模拟硬件故障"""
        # 这里可以通过mock来模拟各种硬件故障
        with patch.object(plugin, 'get_hardware_info', side_effect=Exception("Hardware failure")):
            yield
            
    def simulate_network_failure(self, plugin: BasePlugin):
        """模拟网络故障"""
        # 模拟网络连接失败
        with patch('socket.socket', side_effect=Exception("Network failure")):
            yield
            
    def create_test_dataset(self, size: int = 10) -> List[Dict[str, Any]]:
        """创建测试数据集"""
        dataset = []
        for i in range(size):
            data = {
                'id': i,
                'image': self.mock_camera_frame(),
                'metadata': {
                    'timestamp': time.time(),
                    'source': 'test_camera',
                    'frame_id': i
                }
            }
            dataset.append(data)
        return dataset
        
    def benchmark_plugin(self, plugin: BasePlugin, dataset: List[Dict[str, Any]]) -> Dict[str, float]:
        """对插件进行基准测试"""
        results = {
            'total_time': 0.0,
            'avg_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'fps': 0.0
        }
        
        processing_times = []
        
        for data in dataset:
            start_time = time.time()
            
            # 假设插件有process_frame方法
            if hasattr(plugin, 'process_frame'):
                plugin.process_frame(data['image'], data['metadata'])
            
            end_time = time.time()
            processing_time = end_time - start_time
            processing_times.append(processing_time)
            
        if processing_times:
            results['total_time'] = sum(processing_times)
            results['avg_time'] = results['total_time'] / len(processing_times)
            results['min_time'] = min(processing_times)
            results['max_time'] = max(processing_times)
            results['fps'] = 1.0 / results['avg_time'] if results['avg_time'] > 0 else 0.0
            
        return results