#!/usr/bin/env python3
"""
预处理器工厂类
提供统一的预处理器创建接口
"""

from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np
from .image_preprocessor import ImagePreprocessor

class PreprocessorFactory:
    """预处理器工厂类"""
    
    _preprocessor_registry = {
        'image': ImagePreprocessor,
        'yolo': 'YOLOPreprocessor',
        'default': ImagePreprocessor,
    }
    
    @classmethod
    def create_preprocessor(cls, preprocessor_type: str, config: Dict[str, Any]):
        """
        创建预处理器实例
        
        Args:
            preprocessor_type: 预处理器类型
            config: 配置参数
            
        Returns:
            预处理器实例
        """
        if preprocessor_type not in cls._preprocessor_registry:
            raise ValueError(f"不支持的预处理器类型: {preprocessor_type}")
        
        preprocessor_class = cls._preprocessor_registry[preprocessor_type]
        
        # 处理YOLO预处理器
        if preprocessor_type == 'yolo':
            return cls._create_yolo_preprocessor(config)
        
        try:
            return preprocessor_class(
                target_size=config.get('target_size', (640, 640)),
                normalize=config.get('normalize', True),
                mean=config.get('mean', [0.485, 0.456, 0.406]),
                std=config.get('std', [0.229, 0.224, 0.225])
            )
        except Exception as e:
            raise RuntimeError(f"创建预处理器失败: {e}")
    
    @classmethod
    def _create_yolo_preprocessor(cls, config: Dict[str, Any]):
        """
        创建YOLO预处理器
        
        Args:
            config: 配置参数
            
        Returns:
            YOLO预处理器实例
        """
        class YOLOPreprocessor:
            def __init__(self, target_size=(640, 640), auto_pad=True):
                self.target_size = target_size
                self.auto_pad = auto_pad
            
            def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
                """
                YOLO预处理
                
                Args:
                    image: 输入图像
                    
                Returns:
                    预处理后的图像和元数据
                """
                original_shape = image.shape[:2]
                
                # 调整大小并保持宽高比
                if self.auto_pad:
                    processed_image, ratio, (dw, dh) = self._letterbox(image, self.target_size)
                else:
                    processed_image = cv2.resize(image, self.target_size)
                    ratio = min(self.target_size[0] / original_shape[1], self.target_size[1] / original_shape[0])
                    dw = dh = 0
                
                # 转换为RGB并归一化
                if len(processed_image.shape) == 3:
                    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                
                processed_image = processed_image.astype(np.float32) / 255.0
                
                # 转换为CHW格式
                if len(processed_image.shape) == 3:
                    processed_image = np.transpose(processed_image, (2, 0, 1))
                
                # 添加batch维度
                processed_image = np.expand_dims(processed_image, axis=0)
                
                metadata = {
                    'original_shape': original_shape,
                    'processed_shape': self.target_size,
                    'ratio': ratio,
                    'pad': (dw, dh)
                }
                
                return processed_image, metadata
            
            def _letterbox(self, image, new_shape, color=(114, 114, 114)):
                """
                调整图像大小并添加边框以保持宽高比
                """
                shape = image.shape[:2]  # 当前形状 [height, width]
                if isinstance(new_shape, int):
                    new_shape = (new_shape, new_shape)
                
                # 缩放比例 (new / old)
                r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
                
                # 计算填充
                ratio = r, r  # 宽度, 高度比例
                new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
                dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
                
                dw /= 2  # 分割填充到两边
                dh /= 2
                
                if shape[::-1] != new_unpad:  # 调整大小
                    image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
                
                top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
                left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
                image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
                
                return image, ratio, (dw, dh)
        
        return YOLOPreprocessor(
            target_size=config.get('target_size', (640, 640)),
            auto_pad=config.get('auto_pad', True)
        )
    
    @classmethod
    def register_preprocessor(cls, name: str, preprocessor_class):
        """
        注册新的预处理器类型
        
        Args:
            name: 预处理器名称
            preprocessor_class: 预处理器类
        """
        cls._preprocessor_registry[name] = preprocessor_class
    
    @classmethod
    def get_available_preprocessors(cls):
        """
        获取可用的预处理器类型
        
        Returns:
            可用预处理器类型列表
        """
        return list(cls._preprocessor_registry.keys())