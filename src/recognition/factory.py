#!/usr/bin/env python3
"""
识别器工厂类
提供统一的识别器创建接口
"""

from typing import Dict, Any, Optional

# 安全导入识别器类
try:
    from .face_recognizer import FaceRecognizer
except ImportError:
    FaceRecognizer = None

try:
    from .pose_recognizer import PoseRecognizer
except ImportError:
    PoseRecognizer = None

try:
    from .fall_detector import FallDetector
except ImportError:
    FallDetector = None

class RecognizerFactory:
    """识别器工厂类"""
    
    @classmethod
    def _get_recognizer_registry(cls):
        """动态获取识别器注册表"""
        registry = {}
        
        if FaceRecognizer is not None:
            registry['face'] = FaceRecognizer
        
        if PoseRecognizer is not None:
            registry['pose'] = PoseRecognizer
        
        if FallDetector is not None:
            registry['fall'] = FallDetector
        
        registry['multi'] = 'MultiRecognizer'  # 多模态识别器
        
        return registry
    
    @classmethod
    def _get_registry(cls):
        """获取完整的识别器注册表"""
        registry = cls._get_recognizer_registry()
        registry.update(cls._custom_registry)
        return registry
    
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
        registry = cls._get_registry()
        
        if recognizer_type not in registry:
            available_types = list(registry.keys())
            raise ValueError(f"不支持的识别器类型: {recognizer_type}，可用类型: {available_types}")
        
        recognizer_class = registry[recognizer_type]
        
        # 处理多模态识别器
        if recognizer_type == 'multi':
            return cls._create_multi_recognizer(config)
        
        try:
            # 处理多模态识别器
            if recognizer_type == 'multi':
                return cls._create_multi_recognizer(config)
            
            # 检查识别器类是否可用
            if recognizer_class is None:
                raise RuntimeError(f"识别器类 {recognizer_type} 不可用，可能缺少依赖")
            
            # 根据不同识别器类型传递不同参数
            if recognizer_type == 'face' and FaceRecognizer is not None:
                return recognizer_class(
                    model_path=config.get('model_path'),
                    device=config.get('device', 'auto'),
                    confidence_threshold=config.get('confidence_threshold', 0.5)
                )
            elif recognizer_type == 'pose' and PoseRecognizer is not None:
                return recognizer_class(
                    model_path=config.get('model_path'),
                    device=config.get('device', 'auto'),
                    keypoint_threshold=config.get('keypoint_threshold', 0.3)
                )
            elif recognizer_type == 'fall' and FallDetector is not None:
                return recognizer_class(
                    model_path=config.get('model_path'),
                    device=config.get('device', 'auto'),
                    sensitivity=config.get('sensitivity', 0.7)
                )
            else:
                # 对于自定义识别器或其他类型
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
                registry = cls._get_recognizer_registry()
                for name, rec_config in recognizers_config.items():
                    if name in registry and name != 'multi':
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
    
    # 类级别的注册表，用于动态注册
    _custom_registry = {}
    
    @classmethod
    def register_recognizer(cls, name: str, recognizer_class):
        """
        注册新的识别器类型
        
        Args:
            name: 识别器名称
            recognizer_class: 识别器类
        """
        cls._custom_registry[name] = recognizer_class
    
    @classmethod
    def get_available_recognizers(cls):
        """
        获取可用的识别器类型
        
        Returns:
            可用识别器类型列表
        """
        return list(cls._get_registry().keys())
    
    @classmethod
    def list_available(cls):
        """列出可用识别器类型（兼容性方法）"""
        return cls.get_available_recognizers()
    
    @classmethod
    def get_available(cls):
        """获取可用识别器类型（兼容性方法）"""
        return cls.get_available_recognizers()
    
    @classmethod
    def list_types(cls):
        """列出识别器类型（兼容性方法）"""
        return cls.get_available_recognizers()
    
    @classmethod
    def get_types(cls):
        """获取识别器类型（兼容性方法）"""
        return cls.get_available_recognizers()