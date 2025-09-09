"""模拟数据生成器

为测试提供各种类型的模拟数据，包括：
- 图像数据
- 检测结果
- 传感器数据
- 配置数据
"""

import random
import time
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import cv2
from dataclasses import dataclass
from enum import Enum

class MockDataType(Enum):
    """模拟数据类型"""
    IMAGE = "image"
    VIDEO = "video"
    DETECTION = "detection"
    SENSOR = "sensor"
    CONFIG = "config"
    NETWORK = "network"

@dataclass
class MockDetection:
    """模拟检测结果"""
    bbox: List[float]  # [x, y, w, h]
    class_id: int
    confidence: float
    label: str
    features: Optional[Dict[str, Any]] = None

@dataclass
class MockSensorData:
    """模拟传感器数据"""
    timestamp: float
    sensor_type: str
    value: float
    unit: str
    metadata: Dict[str, Any]

class MockDataGenerator:
    """模拟数据生成器"""
    
    def __init__(self, seed: int = None):
        """初始化生成器
        
        Args:
            seed: 随机种子，用于可重现的测试
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # 预定义的类别标签
        self.class_labels = [
            'person', 'car', 'bicycle', 'dog', 'cat', 'bird', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball', 'kite',
            'baseball_bat', 'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
            'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted_plant', 'bed', 'dining_table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
        ]
        
        # 传感器类型
        self.sensor_types = [
            'temperature', 'humidity', 'pressure', 'light', 'motion', 'sound',
            'accelerometer', 'gyroscope', 'magnetometer', 'gps', 'proximity'
        ]
        
    def generate_image(self, width: int = 640, height: int = 480, 
                      channels: int = 3, noise_level: float = 0.1) -> np.ndarray:
        """生成模拟图像
        
        Args:
            width: 图像宽度
            height: 图像高度
            channels: 通道数
            noise_level: 噪声水平 (0-1)
            
        Returns:
            生成的图像数组
        """
        # 生成基础图像
        if channels == 3:
            # 彩色图像
            image = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
            
            # 添加一些结构化内容
            self._add_geometric_shapes(image)
            
        else:
            # 灰度图像
            image = np.random.randint(0, 256, (height, width), dtype=np.uint8)
            
        # 添加噪声
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * 255, image.shape)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            
        return image
        
    def _add_geometric_shapes(self, image: np.ndarray) -> None:
        """在图像中添加几何形状"""
        height, width = image.shape[:2]
        
        # 添加随机矩形
        for _ in range(random.randint(1, 5)):
            x1 = random.randint(0, width - 50)
            y1 = random.randint(0, height - 50)
            x2 = x1 + random.randint(20, 100)
            y2 = y1 + random.randint(20, 100)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
            
        # 添加随机圆形
        for _ in range(random.randint(1, 3)):
            center_x = random.randint(50, width - 50)
            center_y = random.randint(50, height - 50)
            radius = random.randint(10, 50)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.circle(image, (center_x, center_y), radius, color, -1)
            
    def generate_video_sequence(self, num_frames: int = 30, width: int = 640, 
                               height: int = 480, fps: float = 30.0) -> List[np.ndarray]:
        """生成模拟视频序列
        
        Args:
            num_frames: 帧数
            width: 视频宽度
            height: 视频高度
            fps: 帧率
            
        Returns:
            视频帧列表
        """
        frames = []
        
        for i in range(num_frames):
            # 生成带有时间变化的图像
            frame = self.generate_image(width, height)
            
            # 添加移动的物体
            self._add_moving_objects(frame, i, num_frames)
            
            frames.append(frame)
            
        return frames
        
    def _add_moving_objects(self, frame: np.ndarray, frame_idx: int, total_frames: int) -> None:
        """在帧中添加移动物体"""
        height, width = frame.shape[:2]
        
        # 添加水平移动的矩形
        progress = frame_idx / total_frames
        x = int(progress * (width - 50))
        y = height // 2
        cv2.rectangle(frame, (x, y), (x + 50, y + 30), (0, 255, 0), -1)
        
        # 添加垂直移动的圆形
        x = width // 2
        y = int(progress * (height - 50))
        cv2.circle(frame, (x, y), 20, (255, 0, 0), -1)
        
    def generate_detection_results(self, num_objects: int = None, 
                                 image_width: int = 640, image_height: int = 480) -> List[Dict[str, Any]]:
        """生成模拟检测结果
        
        Args:
            num_objects: 检测对象数量，None表示随机
            image_width: 图像宽度
            image_height: 图像高度
            
        Returns:
            检测结果列表
        """
        if num_objects is None:
            num_objects = random.randint(0, 10)
            
        results = []
        
        for i in range(num_objects):
            # 生成随机边界框
            x = random.randint(0, image_width - 100)
            y = random.randint(0, image_height - 100)
            w = random.randint(50, min(200, image_width - x))
            h = random.randint(50, min(200, image_height - y))
            
            # 生成随机类别和置信度
            class_id = random.randint(0, len(self.class_labels) - 1)
            confidence = random.uniform(0.5, 1.0)
            
            detection = {
                'bbox': [x, y, w, h],
                'class_id': class_id,
                'confidence': confidence,
                'label': self.class_labels[class_id],
                'features': self._generate_object_features()
            }
            
            results.append(detection)
            
        return results
        
    def _generate_object_features(self) -> Dict[str, Any]:
        """生成对象特征"""
        return {
            'area': random.uniform(100, 10000),
            'aspect_ratio': random.uniform(0.5, 2.0),
            'color_histogram': np.random.rand(256).tolist(),
            'texture_features': np.random.rand(64).tolist(),
            'motion_vector': [random.uniform(-10, 10), random.uniform(-10, 10)]
        }
        
    def generate_sensor_data(self, sensor_type: str = None, 
                           duration: float = 10.0, frequency: float = 1.0) -> List[MockSensorData]:
        """生成模拟传感器数据
        
        Args:
            sensor_type: 传感器类型
            duration: 数据持续时间（秒）
            frequency: 采样频率（Hz）
            
        Returns:
            传感器数据列表
        """
        if sensor_type is None:
            sensor_type = random.choice(self.sensor_types)
            
        data_points = []
        num_samples = int(duration * frequency)
        
        for i in range(num_samples):
            timestamp = time.time() + i / frequency
            value = self._generate_sensor_value(sensor_type, i)
            
            data_point = MockSensorData(
                timestamp=timestamp,
                sensor_type=sensor_type,
                value=value,
                unit=self._get_sensor_unit(sensor_type),
                metadata={
                    'sample_id': i,
                    'quality': random.uniform(0.8, 1.0),
                    'calibrated': True
                }
            )
            
            data_points.append(data_point)
            
        return data_points
        
    def _generate_sensor_value(self, sensor_type: str, sample_idx: int) -> float:
        """生成传感器值"""
        base_values = {
            'temperature': 20.0,
            'humidity': 50.0,
            'pressure': 1013.25,
            'light': 500.0,
            'motion': 0.0,
            'sound': 40.0,
            'accelerometer': 0.0,
            'gyroscope': 0.0,
            'magnetometer': 0.0,
            'gps': 0.0,
            'proximity': 100.0
        }
        
        base_value = base_values.get(sensor_type, 0.0)
        
        # 添加噪声和趋势
        noise = random.gauss(0, base_value * 0.1)
        trend = np.sin(sample_idx * 0.1) * base_value * 0.2
        
        return base_value + noise + trend
        
    def _get_sensor_unit(self, sensor_type: str) -> str:
        """获取传感器单位"""
        units = {
            'temperature': '°C',
            'humidity': '%',
            'pressure': 'hPa',
            'light': 'lux',
            'motion': 'm/s²',
            'sound': 'dB',
            'accelerometer': 'm/s²',
            'gyroscope': '°/s',
            'magnetometer': 'μT',
            'gps': '°',
            'proximity': 'cm'
        }
        
        return units.get(sensor_type, 'unit')
        
    def generate_config_data(self, config_type: str = 'plugin') -> Dict[str, Any]:
        """生成模拟配置数据
        
        Args:
            config_type: 配置类型 ('plugin', 'system', 'hardware')
            
        Returns:
            配置字典
        """
        if config_type == 'plugin':
            return self._generate_plugin_config()
        elif config_type == 'system':
            return self._generate_system_config()
        elif config_type == 'hardware':
            return self._generate_hardware_config()
        else:
            return {}
            
    def _generate_plugin_config(self) -> Dict[str, Any]:
        """生成插件配置"""
        return {
            'enabled': True,
            'confidence_threshold': random.uniform(0.3, 0.8),
            'max_objects': random.randint(5, 20),
            'model_path': f'/models/model_{random.randint(1, 5)}.pt',
            'input_size': random.choice([(416, 416), (608, 608), (832, 832)]),
            'batch_size': random.choice([1, 2, 4, 8]),
            'use_gpu': random.choice([True, False]),
            'preprocessing': {
                'normalize': True,
                'resize': True,
                'augment': random.choice([True, False])
            },
            'postprocessing': {
                'nms_threshold': random.uniform(0.3, 0.7),
                'filter_small_objects': True,
                'min_object_size': random.randint(10, 50)
            }
        }
        
    def _generate_system_config(self) -> Dict[str, Any]:
        """生成系统配置"""
        return {
            'log_level': random.choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
            'max_memory_usage': random.randint(512, 2048),
            'max_cpu_usage': random.randint(50, 90),
            'cache_size': random.randint(100, 1000),
            'worker_threads': random.randint(1, 8),
            'enable_profiling': random.choice([True, False]),
            'auto_cleanup': True,
            'backup_interval': random.randint(300, 3600)
        }
        
    def _generate_hardware_config(self) -> Dict[str, Any]:
        """生成硬件配置"""
        return {
            'camera': {
                'device_id': random.randint(0, 3),
                'resolution': random.choice([(640, 480), (1280, 720), (1920, 1080)]),
                'fps': random.choice([15, 30, 60]),
                'format': random.choice(['MJPG', 'YUYV', 'RGB24'])
            },
            'gpio': {
                'pins': {f'pin_{i}': random.choice(['input', 'output']) for i in range(1, 9)},
                'pull_up': random.choice([True, False]),
                'interrupt_enabled': random.choice([True, False])
            },
            'i2c': {
                'bus_number': random.randint(0, 2),
                'frequency': random.choice([100000, 400000, 1000000]),
                'devices': [random.randint(0x10, 0x77) for _ in range(random.randint(1, 5))]
            },
            'spi': {
                'bus': random.randint(0, 1),
                'device': random.randint(0, 1),
                'speed': random.choice([1000000, 5000000, 10000000]),
                'mode': random.randint(0, 3)
            }
        }
        
    def generate_network_data(self, data_type: str = 'http') -> Dict[str, Any]:
        """生成模拟网络数据
        
        Args:
            data_type: 数据类型 ('http', 'mqtt', 'websocket')
            
        Returns:
            网络数据字典
        """
        if data_type == 'http':
            return self._generate_http_data()
        elif data_type == 'mqtt':
            return self._generate_mqtt_data()
        elif data_type == 'websocket':
            return self._generate_websocket_data()
        else:
            return {}
            
    def _generate_http_data(self) -> Dict[str, Any]:
        """生成HTTP数据"""
        return {
            'method': random.choice(['GET', 'POST', 'PUT', 'DELETE']),
            'url': f'/api/v1/{random.choice(["detection", "config", "status", "data"])}',
            'headers': {
                'Content-Type': 'application/json',
                'User-Agent': 'YOLOS-Client/1.0',
                'Authorization': f'Bearer {self._generate_token()}'
            },
            'body': {
                'timestamp': time.time(),
                'data': self.generate_detection_results(random.randint(1, 5))
            },
            'status_code': random.choice([200, 201, 400, 404, 500]),
            'response_time': random.uniform(0.01, 2.0)
        }
        
    def _generate_mqtt_data(self) -> Dict[str, Any]:
        """生成MQTT数据"""
        return {
            'topic': f'yolos/{random.choice(["detection", "status", "config", "alert"])}',
            'payload': {
                'device_id': f'device_{random.randint(1, 100)}',
                'timestamp': time.time(),
                'data': random.choice([
                    self.generate_detection_results(1),
                    self.generate_sensor_data('temperature', 1.0, 1.0),
                    {'status': 'online', 'uptime': random.randint(0, 86400)}
                ])
            },
            'qos': random.choice([0, 1, 2]),
            'retain': random.choice([True, False])
        }
        
    def _generate_websocket_data(self) -> Dict[str, Any]:
        """生成WebSocket数据"""
        return {
            'event': random.choice(['detection', 'status_update', 'config_change', 'error']),
            'data': {
                'session_id': f'session_{random.randint(1, 1000)}',
                'timestamp': time.time(),
                'payload': random.choice([
                    self.generate_detection_results(random.randint(1, 3)),
                    {'message': 'System status updated'},
                    {'error': 'Connection timeout'}
                ])
            },
            'connection_id': f'conn_{random.randint(1, 100)}'
        }
        
    def _generate_token(self) -> str:
        """生成模拟令牌"""
        import string
        chars = string.ascii_letters + string.digits
        return ''.join(random.choice(chars) for _ in range(32))
        
    def generate_batch_data(self, data_type: MockDataType, batch_size: int = 10) -> List[Any]:
        """批量生成数据
        
        Args:
            data_type: 数据类型
            batch_size: 批次大小
            
        Returns:
            数据列表
        """
        batch = []
        
        for _ in range(batch_size):
            if data_type == MockDataType.IMAGE:
                data = self.generate_image()
            elif data_type == MockDataType.DETECTION:
                data = self.generate_detection_results()
            elif data_type == MockDataType.SENSOR:
                data = self.generate_sensor_data()
            elif data_type == MockDataType.CONFIG:
                data = self.generate_config_data()
            elif data_type == MockDataType.NETWORK:
                data = self.generate_network_data()
            else:
                data = {}
                
            batch.append(data)
            
        return batch