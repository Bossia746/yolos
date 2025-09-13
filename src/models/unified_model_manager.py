#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一模型管理器
提供统一的YOLO模型管理接口，支持多版本YOLO模型的加载、切换和优化
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union, Type
from pathlib import Path
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

# 导入各种YOLO模型
from .yolov8_model import YOLOv8Model
from .yolov11_detector import YOLOv11Detector
from .yolo_world_model import YOLOWorldModel
from .yolov5_model import YOLOv5Model


class ModelType(Enum):
    """支持的模型类型"""
    YOLOV5 = "yolov5"
    YOLOV8 = "yolov8"
    YOLOV11 = "yolov11"
    YOLO_WORLD = "yolo_world"


class PlatformType(Enum):
    """支持的平台类型"""
    PC = "pc"
    RASPBERRY_PI = "raspberry_pi"
    JETSON = "jetson"
    ESP32 = "esp32"
    K210 = "k210"
    K230 = "k230"
    MOBILE = "mobile"


@dataclass
class ModelConfig:
    """模型配置"""
    model_type: ModelType
    model_size: str = 's'  # n, s, m, l, x
    model_path: Optional[str] = None
    device: str = 'auto'
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    half_precision: bool = True
    tensorrt_optimize: bool = False
    platform: PlatformType = PlatformType.PC
    max_detections: int = 1000
    input_size: tuple = (640, 640)
    dynamic_batching: bool = True
    
    # YOLOv11特定配置
    c2psa_config: Optional[Dict[str, Any]] = None
    c3k2_config: Optional[Dict[str, Any]] = None
    attention_type: str = 'SE'  # SE, CBAM, ECA, SimAM
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'model_type': self.model_type.value,
            'model_size': self.model_size,
            'model_path': self.model_path,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'half_precision': self.half_precision,
            'tensorrt_optimize': self.tensorrt_optimize,
            'platform': self.platform.value,
            'max_detections': self.max_detections,
            'input_size': self.input_size,
            'dynamic_batching': self.dynamic_batching,
            'c2psa_config': self.c2psa_config,
            'c3k2_config': self.c3k2_config
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """从字典创建配置"""
        return cls(
            model_type=ModelType(data['model_type']),
            model_size=data.get('model_size', 's'),
            model_path=data.get('model_path'),
            device=data.get('device', 'auto'),
            confidence_threshold=data.get('confidence_threshold', 0.25),
            iou_threshold=data.get('iou_threshold', 0.45),
            half_precision=data.get('half_precision', True),
            tensorrt_optimize=data.get('tensorrt_optimize', False),
            platform=PlatformType(data.get('platform', 'pc')),
            max_detections=data.get('max_detections', 1000),
            input_size=tuple(data.get('input_size', (640, 640))),
            dynamic_batching=data.get('dynamic_batching', True),
            c2psa_config=data.get('c2psa_config'),
            c3k2_config=data.get('c3k2_config')
        )


class UnifiedModelInterface(ABC):
    """统一模型接口"""
    
    @abstractmethod
    def detect(self, image: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        """检测接口"""
        pass
    
    @abstractmethod
    def detect_batch(self, images: List[np.ndarray], **kwargs) -> List[List[Dict[str, Any]]]:
        """批量检测接口"""
        pass
    
    @abstractmethod
    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计"""
        pass
    
    @abstractmethod
    def update_config(self, config: ModelConfig):
        """更新配置"""
        pass


class ModelAdapter(UnifiedModelInterface):
    """模型适配器 - 将不同的YOLO模型适配到统一接口"""
    
    def __init__(self, model: Any, config: ModelConfig):
        self.model = model
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def detect(self, image: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        """统一检测接口"""
        try:
            # 根据模型类型调用相应的检测方法
            if hasattr(self.model, 'detect'):
                # YOLOv11Detector 风格
                results = self.model.detect(image, **kwargs)
                return self._normalize_results(results)
            elif hasattr(self.model, 'predict'):
                # YOLOv8Model 风格
                results = self.model.predict(image, **kwargs)
                return self._normalize_results(results)
            else:
                raise NotImplementedError(f"模型 {type(self.model)} 不支持检测方法")
        except Exception as e:
            self.logger.error(f"检测失败: {e}")
            return []
    
    def detect_batch(self, images: List[np.ndarray], **kwargs) -> List[List[Dict[str, Any]]]:
        """批量检测"""
        results = []
        for image in images:
            result = self.detect(image, **kwargs)
            results.append(result)
        return results
    
    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计"""
        if hasattr(self.model, 'get_performance_stats'):
            return self.model.get_performance_stats()
        return {'fps': 0.0, 'avg_inference_time': 0.0}
    
    def update_config(self, config: ModelConfig):
        """更新配置"""
        self.config = config
        # 更新模型参数
        if hasattr(self.model, 'confidence_threshold'):
            self.model.confidence_threshold = config.confidence_threshold
        if hasattr(self.model, 'iou_threshold'):
            self.model.iou_threshold = config.iou_threshold
    
    def _normalize_results(self, results: Any) -> List[Dict[str, Any]]:
        """标准化检测结果格式"""
        if isinstance(results, list) and len(results) > 0:
            # 检查是否已经是标准格式
            if isinstance(results[0], dict) and 'bbox' in results[0]:
                return results
            # 如果是DetectionResult对象，转换为字典
            elif hasattr(results[0], 'bbox'):
                return [{
                    'bbox': [r.bbox.x1, r.bbox.y1, r.bbox.x2, r.bbox.y2],
                    'confidence': r.confidence,
                    'class_id': r.class_id,
                    'class_name': r.class_name
                } for r in results]
        return results if isinstance(results, list) else []


class PlatformOptimizer:
    """平台优化器 - 根据不同平台优化模型配置"""
    
    @staticmethod
    def optimize_for_platform(config: ModelConfig) -> ModelConfig:
        """根据平台优化配置"""
        optimized_config = ModelConfig(**config.__dict__)
        
        if config.platform == PlatformType.ESP32:
            # ESP32优化：使用最小模型，关闭高级功能
            optimized_config.model_size = 'n'
            optimized_config.half_precision = False
            optimized_config.tensorrt_optimize = False
            optimized_config.input_size = (320, 320)
            optimized_config.max_detections = 100
            
        elif config.platform == PlatformType.RASPBERRY_PI:
            # 树莓派优化：平衡性能和精度
            optimized_config.model_size = 's' if config.model_size in ['m', 'l', 'x'] else config.model_size
            optimized_config.half_precision = False  # 树莓派不支持FP16
            optimized_config.tensorrt_optimize = False
            optimized_config.input_size = (416, 416)
            
        elif config.platform == PlatformType.JETSON:
            # Jetson优化：启用GPU加速
            optimized_config.half_precision = True
            optimized_config.tensorrt_optimize = True
            optimized_config.device = 'cuda'
            
        elif config.platform == PlatformType.K230:
            # K230优化：NPU加速
            optimized_config.model_size = 's'
            optimized_config.input_size = (640, 640)
            optimized_config.device = 'npu'  # 特殊设备标识
            
        return optimized_config


class UnifiedModelManager:
    """统一模型管理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.models: Dict[str, ModelAdapter] = {}
        self.current_model: Optional[ModelAdapter] = None
        self.model_configs: Dict[str, ModelConfig] = {}
        
        # 模型类映射
        self.model_classes = {
            ModelType.YOLOV5: YOLOv5Model,
            ModelType.YOLOV8: YOLOv8Model,
            ModelType.YOLOV11: YOLOv11Detector,
            ModelType.YOLO_WORLD: YOLOWorldModel
        }
    
    def register_model(self, name: str, config: ModelConfig) -> bool:
        """注册模型"""
        try:
            # 平台优化
            optimized_config = PlatformOptimizer.optimize_for_platform(config)
            
            # 创建模型实例
            model_class = self.model_classes[config.model_type]
            
            if config.model_type == ModelType.YOLOV11:
                model = model_class(
                    model_size=optimized_config.model_size,
                    device=optimized_config.device,
                    half_precision=optimized_config.half_precision,
                    tensorrt_optimize=optimized_config.tensorrt_optimize,
                    confidence_threshold=optimized_config.confidence_threshold,
                    iou_threshold=optimized_config.iou_threshold,
                    c2psa_config=optimized_config.c2psa_config,
                    c3k2_config=optimized_config.c3k2_config,
                    dynamic_batching=optimized_config.dynamic_batching
                )
            else:
                model = model_class(
                    model_path=optimized_config.model_path,
                    device=optimized_config.device
                )
            
            # 创建适配器
            adapter = ModelAdapter(model, optimized_config)
            
            # 注册模型
            self.models[name] = adapter
            self.model_configs[name] = optimized_config
            
            self.logger.info(f"模型 {name} 注册成功 ({config.model_type.value})")
            return True
            
        except Exception as e:
            self.logger.error(f"注册模型 {name} 失败: {e}")
            return False
    
    def switch_model(self, name: str) -> bool:
        """切换当前模型"""
        if name in self.models:
            self.current_model = self.models[name]
            self.logger.info(f"切换到模型: {name}")
            return True
        else:
            self.logger.error(f"模型 {name} 不存在")
            return False
    
    def detect(self, image: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        """使用当前模型进行检测"""
        if self.current_model is None:
            raise ValueError("没有选择当前模型")
        return self.current_model.detect(image, **kwargs)
    
    def detect_batch(self, images: List[np.ndarray], **kwargs) -> List[List[Dict[str, Any]]]:
        """批量检测"""
        if self.current_model is None:
            raise ValueError("没有选择当前模型")
        return self.current_model.detect_batch(images, **kwargs)
    
    def get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        return list(self.models.keys())
    
    def get_current_model_info(self) -> Optional[Dict[str, Any]]:
        """获取当前模型信息"""
        if self.current_model is None:
            return None
        
        config = self.current_model.config
        stats = self.current_model.get_performance_stats()
        
        return {
            'config': config.to_dict(),
            'performance': stats,
            'model_class': type(self.current_model.model).__name__
        }
    
    def update_model_config(self, name: str, config: ModelConfig) -> bool:
        """更新模型配置"""
        if name in self.models:
            self.models[name].update_config(config)
            self.model_configs[name] = config
            return True
        return False
    
    def save_configs(self, config_path: str):
        """保存配置到文件"""
        configs_dict = {
            name: config.to_dict() 
            for name, config in self.model_configs.items()
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(configs_dict, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"配置已保存到: {config_path}")
    
    def load_configs(self, config_path: str):
        """从文件加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                configs_dict = json.load(f)
            
            for name, config_data in configs_dict.items():
                config = ModelConfig.from_dict(config_data)
                self.register_model(name, config)
            
            self.logger.info(f"配置已从 {config_path} 加载")
            
        except Exception as e:
            self.logger.error(f"加载配置失败: {e}")


# 全局模型管理器实例
model_manager = UnifiedModelManager()


def get_model_manager() -> UnifiedModelManager:
    """获取全局模型管理器实例"""
    return model_manager


if __name__ == "__main__":
    # 示例用法
    manager = UnifiedModelManager()
    
    # 注册YOLOv8模型
    yolov8_config = ModelConfig(
        model_type=ModelType.YOLOV8,
        model_size='s',
        platform=PlatformType.PC
    )
    manager.register_model('yolov8s', yolov8_config)
    
    # 注册YOLOv11模型
    yolov11_config = ModelConfig(
        model_type=ModelType.YOLOV11,
        model_size='s',
        platform=PlatformType.PC,
        tensorrt_optimize=True
    )
    manager.register_model('yolov11s', yolov11_config)
    
    # 切换模型
    manager.switch_model('yolov11s')
    
    # 测试检测
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    results = manager.detect(test_image)
    
    print(f"检测结果: {len(results)} 个目标")
    print(f"当前模型信息: {manager.get_current_model_info()}")