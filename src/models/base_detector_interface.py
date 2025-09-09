#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一检测器接口
为所有YOLO模型提供统一的接口
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import numpy as np
from ..core.types import DetectionResult


class BaseDetectorInterface(ABC):
    """基础检测器接口"""
    
    @abstractmethod
    def detect(self, image: np.ndarray, **kwargs) -> List[DetectionResult]:
        """
        检测单张图像
        
        Args:
            image: 输入图像
            **kwargs: 其他参数
            
        Returns:
            检测结果列表
        """
        pass
    
    def detect_adaptive(self, 
                       image: np.ndarray, 
                       target_fps: float = 30.0,
                       quality_priority: bool = False) -> List[DetectionResult]:
        """
        自适应检测（可选实现）
        
        Args:
            image: 输入图像
            target_fps: 目标FPS
            quality_priority: 是否优先保证质量
            
        Returns:
            检测结果列表
        """
        # 默认实现：调用标准检测
        return self.detect(image)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计（可选实现）
        
        Returns:
            性能统计字典
        """
        return {}
    
    def export_model(self, format: str = 'onnx', output_path: Optional[str] = None) -> str:
        """
        导出模型（可选实现）
        
        Args:
            format: 导出格式
            output_path: 输出路径
            
        Returns:
            导出文件路径
        """
        raise NotImplementedError("此检测器不支持模型导出")


class UnifiedDetectorWrapper(BaseDetectorInterface):
    """统一检测器包装器"""
    
    def __init__(self, detector: Any):
        """
        初始化包装器
        
        Args:
            detector: 原始检测器实例
        """
        self.detector = detector
        self._setup_methods()
    
    def _setup_methods(self):
        """设置方法映射"""
        # 检测方法映射
        if hasattr(self.detector, 'detect'):
            self._detect_method = self.detector.detect
        elif hasattr(self.detector, 'predict'):
            self._detect_method = self._wrap_predict_method
        else:
            raise ValueError("检测器必须有detect或predict方法")
        
        # 自适应检测方法
        if hasattr(self.detector, 'detect_adaptive'):
            self._detect_adaptive_method = self.detector.detect_adaptive
        else:
            self._detect_adaptive_method = None
        
        # 性能统计方法
        if hasattr(self.detector, 'get_performance_stats'):
            self._get_stats_method = self.detector.get_performance_stats
        else:
            self._get_stats_method = None
        
        # 导出方法
        if hasattr(self.detector, 'export_model'):
            self._export_method = self.detector.export_model
        elif hasattr(self.detector, 'export'):
            self._export_method = self.detector.export
        else:
            self._export_method = None
    
    def _wrap_predict_method(self, image: np.ndarray, **kwargs) -> List[DetectionResult]:
        """包装predict方法为detect格式"""
        results = self.detector.predict(image, **kwargs)
        
        # 转换结果格式
        detections = []
        if isinstance(results, list) and len(results) > 0:
            for result in results:
                if isinstance(result, dict):
                    # 字典格式结果
                    detection = self._dict_to_detection_result(result)
                    if detection:
                        detections.append(detection)
        
        return detections
    
    def _dict_to_detection_result(self, result_dict: Dict[str, Any]) -> Optional[DetectionResult]:
        """将字典结果转换为DetectionResult"""
        try:
            from ..core.types import create_detection_result, ObjectType
            
            bbox = result_dict.get('bbox', [0, 0, 0, 0])
            if len(bbox) == 4:
                return create_detection_result(
                    bbox=tuple(bbox),
                    class_id=result_dict.get('class_id', 0),
                    class_name=result_dict.get('class_name', 'unknown'),
                    confidence=result_dict.get('confidence', 0.0),
                    object_type=ObjectType.UNKNOWN
                )
        except Exception:
            pass
        
        return None
    
    def detect(self, image: np.ndarray, **kwargs) -> List[DetectionResult]:
        """检测单张图像"""
        return self._detect_method(image, **kwargs)
    
    def detect_adaptive(self, 
                       image: np.ndarray, 
                       target_fps: float = 30.0,
                       quality_priority: bool = False) -> List[DetectionResult]:
        """自适应检测"""
        if self._detect_adaptive_method:
            return self._detect_adaptive_method(image, target_fps, quality_priority)
        else:
            return self.detect(image)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        if self._get_stats_method:
            return self._get_stats_method()
        else:
            return {}
    
    def export_model(self, format: str = 'onnx', output_path: Optional[str] = None) -> str:
        """导出模型"""
        if self._export_method:
            return self._export_method(format, output_path)
        else:
            raise NotImplementedError("此检测器不支持模型导出")


def create_unified_detector(detector_type: str, **kwargs) -> BaseDetectorInterface:
    """
    创建统一检测器
    
    Args:
        detector_type: 检测器类型
        **kwargs: 初始化参数
        
    Returns:
        统一检测器接口
    """
    from ..models.yolo_factory import YOLOFactory
    
    # 创建原始检测器
    raw_detector = YOLOFactory.create_model(detector_type, **kwargs)
    
    # 如果已经是BaseDetectorInterface的实例，直接返回
    if isinstance(raw_detector, BaseDetectorInterface):
        return raw_detector
    
    # 否则用包装器包装
    return UnifiedDetectorWrapper(raw_detector)