#!/usr/bin/env python3
"""
YOLOS统一配置管理系统
消除重复配置管理器，提供单一配置入口
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict, field
import os
from copy import deepcopy

@dataclass
class CameraConfig:
    """摄像头配置"""
    device_id: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    buffer_size: int = 1
    auto_exposure: bool = True
    brightness: float = 0.5
    contrast: float = 0.5
    saturation: float = 0.5

@dataclass
class DetectionConfig:
    """检测配置"""
    type: str = "yolo"  # yolo, traditional
    model_path: str = "models/yolov5s.pt"
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    max_detections: int = 100
    input_size: int = 640
    device: str = "auto"  # auto, cpu, cuda
    
    # 传统CV检测配置
    color_detection: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'color_ranges': {
            'red': [[0, 50, 50], [10, 255, 255]],
            'green': [[40, 50, 50], [80, 255, 255]],
            'blue': [[100, 50, 50], [130, 255, 255]],
            'yellow': [[20, 50, 50], [30, 255, 255]],
            'white': [[0, 0, 200], [180, 30, 255]],
            'black': [[0, 0, 0], [180, 255, 50]]
        }
    })
    
    motion_detection: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'threshold': 25,
        'min_area': 500,
        'history': 500
    })

@dataclass
class RecognitionConfig:
    """识别配置"""
    types: list = field(default_factory=lambda: ['face', 'gesture', 'object'])
    
    # 人脸识别配置
    face_recognition: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'model_type': 'facenet',
        'confidence_threshold': 0.8,
        'face_database_path': 'data/faces',
        'max_faces': 10
    })
    
    # 手势识别配置
    gesture_recognition: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'model_type': 'mediapipe',
        'confidence_threshold': 0.7,
        'max_hands': 2
    })
    
    # 物体识别配置
    object_recognition: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'model_type': 'yolo',
        'confidence_threshold': 0.6,
        'classes': ['person', 'cat', 'dog', 'bird']
    })

@dataclass
class PreprocessingConfig:
    """预处理配置"""
    enabled: bool = True
    resize_enabled: bool = True
    target_size: tuple = (640, 640)
    normalize_enabled: bool = True
    enhance_enabled: bool = False
    
    # 图像增强配置
    enhancement: Dict[str, Any] = field(default_factory=lambda: {
        'clahe_enabled': False,
        'gamma_correction': 1.0,
        'denoising_enabled': False,
        'sharpening_enabled': False
    })

@dataclass
class PostprocessingConfig:
    """后处理配置"""
    enabled: bool = True
    visualization_enabled: bool = True
    save_results: bool = False
    output_path: str = "results"
    
    # 可视化配置
    visualization: Dict[str, Any] = field(default_factory=lambda: {
        'draw_boxes': True,
        'draw_labels': True,
        'draw_confidence': True,
        'box_color': (0, 255, 0),
        'text_color': (255, 255, 255),
        'font_scale': 0.5,
        'thickness': 2
    })

@dataclass
class TrainingConfig:
    """训练配置"""
    model_architecture: str = "yolov5s"
    input_size: int = 640
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.001
    device: str = "auto"
    
    # 数据配置
    data: Dict[str, str] = field(default_factory=lambda: {
        'train_path': 'data/datasets/train',
        'val_path': 'data/datasets/val',
        'test_path': 'data/datasets/test'
    })
    
    # 优化器配置
    optimizer: Dict[str, Any] = field(default_factory=lambda: {
        'type': 'adam',
        'weight_decay': 0.0005,
        'momentum': 0.937
    })

@dataclass
class APIConfig:
    """API配置"""
    enabled: bool = True
    host: str = "localhost"
    port: int = 8080
    debug: bool = False
    
    # 认证配置
    authentication: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': False,
        'api_key': None,
        'jwt_secret': None
    })
    
    # CORS配置
    cors: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'origins': ["*"],
        'methods': ["GET", "POST"],
        'headers': ["*"]
    })

@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    console_output: bool = True
    file_output: bool = True
    max_file_size: str = "10MB"
    backup_count: int = 5
    
    # 模块日志级别
    modules: Dict[str, str] = field(default_factory=lambda: {
        'engine': 'INFO',
        'detector': 'INFO',
        'recognizer': 'INFO',
        'training': 'INFO',
        'api': 'INFO'
    })

class YOLOSConfig:
    """
    YOLOS统一配置管理器
    
    设计原则:
    1. 单一配置入口 - 所有配置通过此类管理
    2. 类型安全 - 使用dataclass确保类型安全
    3. 默认值 - 提供合理的默认配置
    4. 验证 - 配置加载时进行验证
    5. 热更新 - 支持运行时配置更新
    """
    
    def __init__(self):
        """初始化配置"""
        self.camera_config = CameraConfig()
        self.detection_config = DetectionConfig()
        self.recognition_config = RecognitionConfig()
        self.preprocessing_config = PreprocessingConfig()
        self.postprocessing_config = PostprocessingConfig()
        self.training_config = TrainingConfig()
        self.api_config = APIConfig()
        self.logging_config = LoggingConfig()
        
        # 元数据
        self._config_version = "1.0"
        self._config_source = "default"
        self._last_modified = None
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'YOLOSConfig':
        """
        从配置文件加载配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            YOLOSConfig: 配置实例
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        # 根据文件扩展名选择解析器
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
        
        # 创建配置实例
        config = cls()
        config._load_from_dict(config_data)
        config._config_source = str(config_path)
        config._last_modified = config_path.stat().st_mtime
        
        return config
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'YOLOSConfig':
        """
        从字典加载配置
        
        Args:
            config_dict: 配置字典
            
        Returns:
            YOLOSConfig: 配置实例
        """
        config = cls()
        config._load_from_dict(config_dict)
        config._config_source = "dict"
        
        return config
    
    @classmethod
    def from_env(cls, prefix: str = "YOLOS_") -> 'YOLOSConfig':
        """
        从环境变量加载配置
        
        Args:
            prefix: 环境变量前缀
            
        Returns:
            YOLOSConfig: 配置实例
        """
        config = cls()
        
        # 扫描环境变量
        env_config = {}
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                env_config[config_key] = value
        
        if env_config:
            config._load_from_dict(env_config)
        
        config._config_source = "environment"
        
        return config
    
    def _load_from_dict(self, config_dict: Dict[str, Any]):
        """从字典加载配置"""
        # 加载各个配置模块
        if 'camera' in config_dict:
            self._update_dataclass(self.camera_config, config_dict['camera'])
        
        if 'detection' in config_dict:
            self._update_dataclass(self.detection_config, config_dict['detection'])
        
        if 'recognition' in config_dict:
            self._update_dataclass(self.recognition_config, config_dict['recognition'])
        
        if 'preprocessing' in config_dict:
            self._update_dataclass(self.preprocessing_config, config_dict['preprocessing'])
        
        if 'postprocessing' in config_dict:
            self._update_dataclass(self.postprocessing_config, config_dict['postprocessing'])
        
        if 'training' in config_dict:
            self._update_dataclass(self.training_config, config_dict['training'])
        
        if 'api' in config_dict:
            self._update_dataclass(self.api_config, config_dict['api'])
        
        if 'logging' in config_dict:
            self._update_dataclass(self.logging_config, config_dict['logging'])
    
    def _update_dataclass(self, dataclass_instance, update_dict: Dict[str, Any]):
        """更新dataclass实例"""
        for key, value in update_dict.items():
            if hasattr(dataclass_instance, key):
                current_value = getattr(dataclass_instance, key)
                
                # 如果当前值是字典，进行深度合并
                if isinstance(current_value, dict) and isinstance(value, dict):
                    current_value.update(value)
                else:
                    setattr(dataclass_instance, key, value)
    
    def save(self, config_path: Union[str, Path], format: str = "yaml"):
        """
        保存配置到文件
        
        Args:
            config_path: 配置文件路径
            format: 文件格式 (yaml, json)
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为字典
        config_dict = self.to_dict()
        
        # 保存文件
        if format.lower() == "yaml":
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
        elif format.lower() == "json":
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的格式: {format}")
        
        self._config_source = str(config_path)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'camera': asdict(self.camera_config),
            'detection': asdict(self.detection_config),
            'recognition': asdict(self.recognition_config),
            'preprocessing': asdict(self.preprocessing_config),
            'postprocessing': asdict(self.postprocessing_config),
            'training': asdict(self.training_config),
            'api': asdict(self.api_config),
            'logging': asdict(self.logging_config),
            '_metadata': {
                'version': self._config_version,
                'source': self._config_source,
                'last_modified': self._last_modified
            }
        }
    
    def validate(self) -> bool:
        """
        验证配置有效性
        
        Returns:
            bool: 配置是否有效
        """
        try:
            # 验证摄像头配置
            assert self.camera_config.width > 0, "摄像头宽度必须大于0"
            assert self.camera_config.height > 0, "摄像头高度必须大于0"
            assert self.camera_config.fps > 0, "帧率必须大于0"
            
            # 验证检测配置
            assert 0 < self.detection_config.confidence_threshold <= 1, "置信度阈值必须在(0,1]范围内"
            assert 0 < self.detection_config.nms_threshold <= 1, "NMS阈值必须在(0,1]范围内"
            
            # 验证识别配置
            for recog_type in self.recognition_config.types:
                assert recog_type in ['face', 'gesture', 'object', 'pose'], f"不支持的识别类型: {recog_type}"
            
            # 验证API配置
            assert 1 <= self.api_config.port <= 65535, "端口号必须在1-65535范围内"
            
            return True
            
        except AssertionError as e:
            raise ValueError(f"配置验证失败: {e}")
    
    def update(self, **kwargs):
        """
        更新配置
        
        Args:
            **kwargs: 配置更新参数
        """
        for key, value in kwargs.items():
            if hasattr(self, f"{key}_config"):
                config_obj = getattr(self, f"{key}_config")
                if isinstance(value, dict):
                    self._update_dataclass(config_obj, value)
                else:
                    setattr(self, f"{key}_config", value)
    
    def get_config(self, module: str) -> Any:
        """
        获取指定模块的配置
        
        Args:
            module: 模块名称
            
        Returns:
            配置对象
        """
        config_attr = f"{module}_config"
        if hasattr(self, config_attr):
            return getattr(self, config_attr)
        else:
            raise ValueError(f"未知的配置模块: {module}")
    
    def copy(self) -> 'YOLOSConfig':
        """创建配置副本"""
        new_config = YOLOSConfig()
        new_config._load_from_dict(self.to_dict())
        return new_config
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"YOLOSConfig(source={self._config_source}, version={self._config_version})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return self.__str__()

# 全局配置实例
_global_config: Optional[YOLOSConfig] = None

def get_global_config() -> YOLOSConfig:
    """获取全局配置实例"""
    global _global_config
    if _global_config is None:
        _global_config = YOLOSConfig()
    return _global_config

def set_global_config(config: YOLOSConfig):
    """设置全局配置实例"""
    global _global_config
    _global_config = config

def load_config(config_path: Optional[str] = None) -> YOLOSConfig:
    """
    加载配置
    
    Args:
        config_path: 配置文件路径，如果为None则使用默认配置
        
    Returns:
        YOLOSConfig: 配置实例
    """
    if config_path:
        config = YOLOSConfig.from_file(config_path)
    else:
        # 尝试从标准位置加载配置
        standard_paths = [
            "config/yolos.yaml",
            "config/yolos.yml", 
            "config/yolos.json",
            "yolos.yaml",
            "yolos.yml",
            "yolos.json"
        ]
        
        config = None
        for path in standard_paths:
            if Path(path).exists():
                config = YOLOSConfig.from_file(path)
                break
        
        if config is None:
            config = YOLOSConfig()
    
    # 验证配置
    config.validate()
    
    # 设置为全局配置
    set_global_config(config)
    
    return config

if __name__ == "__main__":
    # 测试配置系统
    print("🔧 测试YOLOS配置系统...")
    
    # 创建默认配置
    config = YOLOSConfig()
    print(f"✅ 默认配置创建成功: {config}")
    
    # 验证配置
    try:
        config.validate()
        print("✅ 配置验证通过")
    except ValueError as e:
        print(f"❌ 配置验证失败: {e}")
    
    # 保存配置
    config.save("test_config.yaml")
    print("✅ 配置保存成功")
    
    # 加载配置
    loaded_config = YOLOSConfig.from_file("test_config.yaml")
    print(f"✅ 配置加载成功: {loaded_config}")
    
    # 清理测试文件
    Path("test_config.yaml").unlink(missing_ok=True)
    print("✅ 测试完成")