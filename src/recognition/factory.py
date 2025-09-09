#!/usr/bin/env python3
"""
识别器工厂类
提供统一的识别器创建接口
"""

from typing import Dict, Any, Optional
from .face_recognizer import FaceRecognizer
from .pose_recognizer import PoseRecognizer
from .fall_detector import FallDetector

class RecognizerFactory:
    """识别器工厂类"""
    
    _recognizer_registry = {
        'face': FaceRecognizer,
        'pose': PoseRecognizer,
        'fall': FallDetector,
        'multi': 'MultiRecognizer',  # 多模态识别器
    }
    
    @classmethod
    def create_recognizer(cls, recognizer_type: str, config: Dict[str, Any]):
        """
        创建识别器实例
        
        Args:
            recognizer_type: 识别器类型
            config: 配置参数
            
        Returns:
            识别器实例
        """
        if recognizer_type not in cls._recognizer_registry:
            raise ValueError(f"不支持的识别器类型: {recognizer_type}")
        
        recognizer_class = cls._recognizer_registry[recognizer_type]
        
        # 处理多模态识别器
        if recognizer_type == 'multi':
            return cls._create_multi_recognizer(config)
        
        try:
            # 根据不同识别器类型传递不同参数
            if recognizer_type == 'face':
                return recognizer_class(
                    model_path=config.get('model_path'),
                    device=config.get('device', 'auto'),
                    confidence_threshold=config.get('confidence_threshold', 0.5)
                )
            elif recognizer_type == 'pose':
                return recognizer_class(
                    model_path=config.get('model_path'),
                    device=config.get('device', 'auto'),
                    keypoint_threshold=config.get('keypoint_threshold', 0.3)
                )
            elif recognizer_type == 'fall':
                return recognizer_class(
                    model_path=config.get('model_path'),
                    device=config.get('device', 'auto'),
                    sensitivity=config.get('sensitivity', 0.7)
                )
            else:
                return recognizer_class(**config)
                
        except Exception as e:
            raise RuntimeError(f"创建识别器失败: {e}")
    
    @classmethod
    def _create_multi_recognizer(cls, config: Dict[str, Any]):
        """
        创建多模态识别器
        
        Args:
            config: 配置参数
            
        Returns:
            多模态识别器实例
        """
        # 简单的多模态识别器实现
        class MultiRecognizer:
            def __init__(self, recognizers_config):
                self.recognizers = {}
                for name, rec_config in recognizers_config.items():
                    if name in cls._recognizer_registry and name != 'multi':
                        self.recognizers[name] = cls.create_recognizer(name, rec_config)
            
            def recognize(self, image, detections=None):
                results = {}
                for name, recognizer in self.recognizers.items():
                    try:
                        if hasattr(recognizer, 'recognize'):
                            results[name] = recognizer.recognize(image, detections)
                        elif hasattr(recognizer, 'detect'):
                            results[name] = recognizer.detect(image)
                    except Exception as e:
                        results[name] = {'error': str(e)}
                return results
        
        recognizers_config = config.get('recognizers', {})
        return MultiRecognizer(recognizers_config)
    
    @classmethod
    def register_recognizer(cls, name: str, recognizer_class):
        """
        注册新的识别器类型
        
        Args:
            name: 识别器名称
            recognizer_class: 识别器类
        """
        cls._recognizer_registry[name] = recognizer_class
    
    @classmethod
    def get_available_recognizers(cls):
        """
        获取可用的识别器类型
        
        Returns:
            可用识别器类型列表
        """
        return list(cls._recognizer_registry.keys())