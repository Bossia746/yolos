#!/usr/bin/env python3
"""
检测器工厂类
提供统一的检测器创建接口
"""

from typing import Dict, Any, Optional
from .realtime_detector import RealtimeDetector
from .image_detector import ImageDetector
from .video_detector import VideoDetector
from .camera_detector import CameraDetector

class DetectorFactory:
    """检测器工厂类"""
    
    _detector_registry = {
        'realtime': RealtimeDetector,
        'image': ImageDetector,
        'video': VideoDetector,
        'camera': CameraDetector,
        'yolo': RealtimeDetector,  # 默认使用实时检测器
    }
    
    @classmethod
    def create_detector(cls, detector_type: str, config: Dict[str, Any]):
        """
        创建检测器实例
        
        Args:
            detector_type: 检测器类型
            config: 配置参数
            
        Returns:
            检测器实例
        """
        if detector_type not in cls._detector_registry:
            raise ValueError(f"不支持的检测器类型: {detector_type}")
        
        detector_class = cls._detector_registry[detector_type]
        
        try:
            # 根据不同检测器类型传递不同参数
            if detector_type in ['realtime', 'yolo']:
                return detector_class(
                    model_type=config.get('model_type', 'yolov8'),
                    model_path=config.get('model_path'),
                    device=config.get('device', 'auto'),
                    confidence_threshold=config.get('confidence_threshold', 0.5),
                    nms_threshold=config.get('nms_threshold', 0.4)
                )
            elif detector_type == 'camera':
                return detector_class(
                    model_type=config.get('model_type', 'yolov8'),
                    model_path=config.get('model_path'),
                    device=config.get('device', 'auto'),
                    camera_type=config.get('camera_type', 'usb')
                )
            elif detector_type in ['image', 'video']:
                return detector_class(
                    model_type=config.get('model_type', 'yolov8'),
                    model_path=config.get('model_path'),
                    device=config.get('device', 'auto')
                )
            else:
                return detector_class(**config)
                
        except Exception as e:
            raise RuntimeError(f"创建检测器失败: {e}")
    
    @classmethod
    def register_detector(cls, name: str, detector_class):
        """
        注册新的检测器类型
        
        Args:
            name: 检测器名称
            detector_class: 检测器类
        """
        cls._detector_registry[name] = detector_class
    
    @classmethod
    def get_available_detectors(cls):
        """
        获取可用的检测器类型
        
        Returns:
            可用检测器类型列表
        """
        return list(cls._detector_registry.keys())