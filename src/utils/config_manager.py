"""
配置管理器
"""

import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Union[str, Path]):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self.config = {}
        
        if self.config_path.exists():
            self.load()
        else:
            # 创建默认配置
            self._create_default_config()
    
    def load(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if self.config_path.suffix.lower() == '.json':
                    self.config = json.load(f)
                else:
                    self.config = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            self.config = {}
    
    def save(self):
        """保存配置文件"""
        try:
            # 确保目录存在
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                if self.config_path.suffix.lower() == '.json':
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
                else:
                    yaml.dump(self.config, f, default_flow_style=False, 
                             allow_unicode=True, indent=2)
        except Exception as e:
            print(f"保存配置文件失败: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键，支持点分隔的嵌套键 (如 'model.type')
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        设置配置值
        
        Args:
            key: 配置键，支持点分隔的嵌套键
            value: 配置值
        """
        keys = key.split('.')
        config = self.config
        
        # 创建嵌套字典结构
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]):
        """
        批量更新配置
        
        Args:
            updates: 更新的配置字典
        """
        for key, value in updates.items():
            self.set(key, value)
    
    def delete(self, key: str):
        """
        删除配置项
        
        Args:
            key: 配置键
        """
        keys = key.split('.')
        config = self.config
        
        try:
            for k in keys[:-1]:
                config = config[k]
            del config[keys[-1]]
        except (KeyError, TypeError):
            pass
    
    def has(self, key: str) -> bool:
        """
        检查配置项是否存在
        
        Args:
            key: 配置键
            
        Returns:
            是否存在
        """
        return self.get(key) is not None
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有配置"""
        return self.config.copy()
    
    def reset(self):
        """重置为默认配置"""
        self.config = {}
        self._create_default_config()
    
    def _create_default_config(self):
        """创建默认配置"""
        default_config = {
            'model': {
                'type': 'yolov8',
                'size': 'n',
                'device': 'auto',
                'confidence_threshold': 0.25,
                'iou_threshold': 0.7
            },
            'detection': {
                'input_size': [640, 640],
                'max_detections': 100,
                'save_results': True,
                'draw_labels': True,
                'draw_confidence': True
            },
            'camera': {
                'type': 'usb',
                'device_id': 0,
                'resolution': [640, 480],
                'framerate': 30,
                'detection_interval': 1
            },
            'training': {
                'epochs': 100,
                'batch_size': 16,
                'learning_rate': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3,
                'patience': 50,
                'save_period': 10,
                'workers': 8
            },
            'mqtt': {
                'enabled': False,
                'broker_host': 'localhost',
                'broker_port': 1883,
                'client_id': None,
                'username': None,
                'password': None,
                'topics': {
                    'detection_results': 'yolos/detection/results',
                    'system_status': 'yolos/system/status',
                    'commands': 'yolos/commands'
                }
            },
            'http_server': {
                'enabled': False,
                'host': '0.0.0.0',
                'port': 8000,
                'cors_enabled': True
            },
            'websocket': {
                'enabled': False,
                'host': '0.0.0.0',
                'port': 8001
            },
            'ros': {
                'enabled': False,
                'version': 2,
                'node_name': 'yolos_detector',
                'topics': {
                    'image_input': '/camera/image_raw',
                    'detection_output': '/yolos/detections',
                    'image_output': '/yolos/image_annotated'
                }
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/yolos.log',
                'max_size': '10MB',
                'backup_count': 5
            },
            'performance': {
                'enable_gpu': True,
                'enable_tensorrt': False,
                'enable_openvino': False,
                'batch_processing': False,
                'async_processing': True
            },
            'storage': {
                'save_images': False,
                'save_videos': False,
                'output_dir': 'outputs',
                'image_format': 'jpg',
                'video_format': 'mp4'
            }
        }
        
        self.config = default_config
        self.save()
    
    def merge_config(self, other_config: Union[str, Path, Dict[str, Any]]):
        """
        合并其他配置
        
        Args:
            other_config: 其他配置文件路径或配置字典
        """
        if isinstance(other_config, (str, Path)):
            # 从文件加载
            other_manager = ConfigManager(other_config)
            other_dict = other_manager.get_all()
        else:
            other_dict = other_config
        
        self._deep_merge(self.config, other_dict)
    
    def _deep_merge(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """深度合并字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def validate(self) -> bool:
        """验证配置的有效性"""
        try:
            # 检查必需的配置项
            required_keys = [
                'model.type',
                'detection.input_size',
                'camera.type'
            ]
            
            for key in required_keys:
                if not self.has(key):
                    print(f"缺少必需的配置项: {key}")
                    return False
            
            # 检查配置值的有效性
            model_type = self.get('model.type')
            if model_type not in ['yolov5', 'yolov8', 'yolo-world']:
                print(f"无效的模型类型: {model_type}")
                return False
            
            confidence = self.get('model.confidence_threshold')
            if not (0 <= confidence <= 1):
                print(f"无效的置信度阈值: {confidence}")
                return False
            
            return True
            
        except Exception as e:
            print(f"配置验证失败: {e}")
            return False
    
    def export_template(self, output_path: Union[str, Path]):
        """导出配置模板"""
        template_config = self._create_template_config()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if output_path.suffix.lower() == '.json':
                json.dump(template_config, f, indent=2, ensure_ascii=False)
            else:
                yaml.dump(template_config, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
    
    def _create_template_config(self) -> Dict[str, Any]:
        """创建配置模板"""
        return {
            '# YOLOS配置文件模板': None,
            'model': {
                'type': 'yolov8  # 模型类型: yolov5, yolov8, yolo-world',
                'size': 'n  # 模型大小: n, s, m, l, x',
                'device': 'auto  # 设备: auto, cpu, cuda',
                'confidence_threshold': 0.25,
                'iou_threshold': 0.7
            },
            'detection': {
                'input_size': [640, 640],
                'max_detections': 100,
                'save_results': True
            },
            'camera': {
                'type': 'usb  # 摄像头类型: usb, picamera',
                'device_id': 0,
                'resolution': [640, 480],
                'framerate': 30
            }
        }