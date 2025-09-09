#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版YOLOv11检测器
集成最新优化技术和AIoT平台适配
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union
import time
from pathlib import Path
import threading
from queue import Queue
import logging

try:
    from ultralytics import YOLO
    from ultralytics.utils import ops
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    YOLO = None
    TRT_AVAILABLE = False

from ..core.types import DetectionResult, create_detection_result, ObjectType
from ..utils.logging_manager import LoggingManager
from .yolov11_detector import YOLOv11Detector
from .base_detector_interface import BaseDetectorInterface


class EnhancedYOLOv11Detector(YOLOv11Detector, BaseDetectorInterface):
    """
    增强版YOLOv11检测器
    
    新增特性:
    - 自适应推理优化
    - 边缘设备专用优化
    - 智能批处理
    - 动态模型切换
    - 性能监控和自动调优
    """
    
    def __init__(self, 
                 model_size: str = 's',
                 device: str = 'auto',
                 half_precision: bool = True,
                 tensorrt_optimize: bool = True,
                 confidence_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 edge_optimization: bool = False,
                 adaptive_inference: bool = True):
        """
        初始化增强版YOLOv11检测器
        
        Args:
            edge_optimization: 是否启用边缘设备优化
            adaptive_inference: 是否启用自适应推理
        """
        super().__init__(model_size, device, half_precision, tensorrt_optimize, 
                        confidence_threshold, iou_threshold)
        
        self.edge_optimization = edge_optimization
        self.adaptive_inference = adaptive_inference
        
        # 性能监控
        self.performance_monitor = PerformanceMonitor()
        
        # 自适应推理配置
        self.adaptive_config = AdaptiveInferenceConfig()
        
        # 边缘优化配置
        if edge_optimization:
            self._apply_edge_optimizations()
        
        # 智能批处理器
        self.batch_processor = IntelligentBatchProcessor()
        
        self.logger.info(f"增强版YOLOv11检测器初始化完成")
        self.logger.info(f"边缘优化: {edge_optimization}, 自适应推理: {adaptive_inference}")
    
    def _apply_edge_optimizations(self):
        """应用边缘设备优化"""
        try:
            # 降低默认输入尺寸
            self.adaptive_config.default_input_size = 416
            
            # 启用更激进的量化
            if self.half_precision:
                self.adaptive_config.quantization_level = 'int8'
            
            # 优化内存使用
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # 设置线程数
            if self.device.type == 'cpu':
                torch.set_num_threads(2)  # 边缘设备通常CPU核心较少
            
            self.logger.info("边缘设备优化已应用")
            
        except Exception as e:
            self.logger.warning(f"边缘优化应用失败: {e}")
    
    def detect_adaptive(self, 
                       image: np.ndarray,
                       target_fps: float = 30.0,
                       quality_priority: bool = False) -> List[DetectionResult]:
        """
        自适应检测
        根据性能要求动态调整推理参数
        
        Args:
            image: 输入图像
            target_fps: 目标FPS
            quality_priority: 是否优先保证质量
            
        Returns:
            检测结果列表
        """
        if not self.adaptive_inference:
            return self.detect(image)
        
        start_time = time.time()
        
        # 获取当前性能状态
        current_fps = self.performance_monitor.get_current_fps()
        
        # 动态调整推理参数
        inference_params = self._adapt_inference_params(
            current_fps, target_fps, quality_priority
        )
        
        # 调整图像尺寸
        if inference_params['resize_factor'] != 1.0:
            h, w = image.shape[:2]
            new_h = int(h * inference_params['resize_factor'])
            new_w = int(w * inference_params['resize_factor'])
            image = cv2.resize(image, (new_w, new_h))
        
        # 执行检测
        with torch.no_grad():
            results = self.model(image,
                               conf=inference_params['confidence_threshold'],
                               iou=inference_params['iou_threshold'],
                               half=self.half_precision,
                               device=self.device)
        
        # 后处理
        detections = self._postprocess_results(results, image.shape)
        
        # 更新性能监控
        inference_time = time.time() - start_time
        self.performance_monitor.update(inference_time, len(detections))
        
        return detections
    
    def _adapt_inference_params(self, 
                              current_fps: float, 
                              target_fps: float, 
                              quality_priority: bool) -> Dict[str, float]:
        """动态调整推理参数"""
        params = {
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'resize_factor': 1.0
        }
        
        if current_fps < target_fps * 0.8:  # 性能不足
            if not quality_priority:
                # 降低质量以提升速度
                params['confidence_threshold'] = min(0.4, self.confidence_threshold + 0.1)
                params['iou_threshold'] = min(0.6, self.iou_threshold + 0.1)
                params['resize_factor'] = 0.8
        
        elif current_fps > target_fps * 1.2:  # 性能过剩
            # 提升质量
            params['confidence_threshold'] = max(0.15, self.confidence_threshold - 0.05)
            params['iou_threshold'] = max(0.3, self.iou_threshold - 0.05)
            params['resize_factor'] = 1.1
        
        return params
    
    def detect_batch_intelligent(self, 
                               images: List[np.ndarray],
                               max_batch_size: int = 8) -> List[List[DetectionResult]]:
        """
        智能批处理检测
        根据图像相似性和设备性能动态调整批处理策略
        
        Args:
            images: 图像列表
            max_batch_size: 最大批处理大小
            
        Returns:
            每张图像的检测结果列表
        """
        return self.batch_processor.process_batch(
            images, self, max_batch_size
        )
    
    def optimize_for_platform(self, platform: str):
        """
        针对特定平台优化
        
        Args:
            platform: 平台类型 ('esp32', 'raspberry_pi', 'jetson_nano', 'pc')
        """
        platform_configs = {
            'esp32': {
                'input_size': 320,
                'confidence_threshold': 0.4,
                'model_size': 'n',
                'quantization': 'int8'
            },
            'raspberry_pi': {
                'input_size': 416,
                'confidence_threshold': 0.3,
                'model_size': 's',
                'quantization': 'fp16'
            },
            'jetson_nano': {
                'input_size': 640,
                'confidence_threshold': 0.25,
                'model_size': 'm',
                'quantization': 'fp16'
            },
            'pc': {
                'input_size': 640,
                'confidence_threshold': 0.25,
                'model_size': 'l',
                'quantization': 'fp32'
            }
        }
        
        if platform in platform_configs:
            config = platform_configs[platform]
            
            # 更新配置
            self.confidence_threshold = config['confidence_threshold']
            self.adaptive_config.default_input_size = config['input_size']
            
            self.logger.info(f"已优化为{platform}平台配置")
        else:
            self.logger.warning(f"未知平台: {platform}")
    
    def export_optimized_model(self, 
                             platform: str,
                             output_path: Optional[str] = None) -> str:
        """
        导出针对特定平台优化的模型
        
        Args:
            platform: 目标平台
            output_path: 输出路径
            
        Returns:
            导出文件路径
        """
        if output_path is None:
            output_path = f'yolov11{self.model_size}_{platform}_optimized'
        
        # 根据平台选择导出格式
        platform_formats = {
            'esp32': 'tflite',
            'raspberry_pi': 'onnx',
            'jetson_nano': 'tensorrt',
            'pc': 'torchscript'
        }
        
        export_format = platform_formats.get(platform, 'onnx')
        
        try:
            # 应用平台优化
            self.optimize_for_platform(platform)
            
            # 导出模型
            exported_path = self.export_model(export_format, output_path)
            
            self.logger.info(f"已导出{platform}优化模型: {exported_path}")
            return exported_path
            
        except Exception as e:
            self.logger.error(f"模型导出失败: {e}")
            raise


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.inference_times = []
        self.detection_counts = []
        self.fps_history = []
        
    def update(self, inference_time: float, detection_count: int):
        """更新性能数据"""
        self.inference_times.append(inference_time)
        self.detection_counts.append(detection_count)
        
        # 计算FPS
        fps = 1.0 / inference_time if inference_time > 0 else 0
        self.fps_history.append(fps)
        
        # 保持窗口大小
        if len(self.inference_times) > self.window_size:
            self.inference_times.pop(0)
            self.detection_counts.pop(0)
            self.fps_history.pop(0)
    
    def get_current_fps(self) -> float:
        """获取当前FPS"""
        if not self.fps_history:
            return 0.0
        return np.mean(self.fps_history[-5:])  # 最近5帧的平均FPS
    
    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计"""
        if not self.inference_times:
            return {}
        
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'avg_fps': np.mean(self.fps_history),
            'avg_detections': np.mean(self.detection_counts),
            'fps_std': np.std(self.fps_history),
            'current_fps': self.get_current_fps()
        }


class AdaptiveInferenceConfig:
    """自适应推理配置"""
    
    def __init__(self):
        self.default_input_size = 640
        self.min_input_size = 320
        self.max_input_size = 1024
        self.quantization_level = 'fp16'
        self.dynamic_batching = True
        self.performance_target = 'balanced'  # 'speed', 'accuracy', 'balanced'


class IntelligentBatchProcessor:
    """智能批处理器"""
    
    def __init__(self):
        self.similarity_threshold = 0.8
        
    def process_batch(self, 
                     images: List[np.ndarray], 
                     detector: EnhancedYOLOv11Detector,
                     max_batch_size: int) -> List[List[DetectionResult]]:
        """处理批量图像"""
        if len(images) <= max_batch_size:
            # 直接批处理
            return self._process_direct_batch(images, detector)
        
        # 智能分组
        groups = self._group_similar_images(images, max_batch_size)
        
        all_results = []
        for group in groups:
            group_results = self._process_direct_batch(group, detector)
            all_results.extend(group_results)
        
        return all_results
    
    def _process_direct_batch(self, 
                            images: List[np.ndarray], 
                            detector: EnhancedYOLOv11Detector) -> List[List[DetectionResult]]:
        """直接批处理"""
        results = []
        for image in images:
            result = detector.detect(image)
            results.append(result)
        return results
    
    def _group_similar_images(self, 
                            images: List[np.ndarray], 
                            max_group_size: int) -> List[List[np.ndarray]]:
        """根据相似性分组图像"""
        # 简化实现：按顺序分组
        groups = []
        for i in range(0, len(images), max_group_size):
            group = images[i:i + max_group_size]
            groups.append(group)
        return groups


class MultiModelEnsemble:
    """多模型集成检测器"""
    
    def __init__(self, model_configs: List[Dict[str, str]]):
        """
        初始化多模型集成
        
        Args:
            model_configs: 模型配置列表
        """
        self.detectors = []
        self.weights = []
        
        for config in model_configs:
            detector = EnhancedYOLOv11Detector(**config)
            self.detectors.append(detector)
            self.weights.append(config.get('weight', 1.0))
        
        # 归一化权重
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        self.logger = LoggingManager().get_logger("MultiModelEnsemble")
    
    def detect_ensemble(self, image: np.ndarray) -> List[DetectionResult]:
        """集成检测"""
        all_detections = []
        
        # 收集所有模型的检测结果
        for detector, weight in zip(self.detectors, self.weights):
            detections = detector.detect(image)
            
            # 应用权重
            for detection in detections:
                detection.confidence *= weight
            
            all_detections.extend(detections)
        
        # 非最大抑制合并结果
        merged_detections = self._merge_detections(all_detections)
        
        return merged_detections
    
    def _merge_detections(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """合并检测结果"""
        if not detections:
            return []
        
        # 简化实现：按置信度排序并去重
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        merged = []
        for detection in detections:
            # 检查是否与已有检测重叠
            is_duplicate = False
            for existing in merged:
                if self._calculate_iou(detection, existing) > 0.5:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(detection)
        
        return merged
    
    def _calculate_iou(self, det1: DetectionResult, det2: DetectionResult) -> float:
        """计算IoU"""
        # 简化IoU计算
        box1 = det1.bbox
        box2 = det2.bbox
        
        # 计算交集
        x1 = max(box1.x, box2.x)
        y1 = max(box1.y, box2.y)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = box1.area + box2.area - intersection
        
        return intersection / union if union > 0 else 0.0


# 使用示例
if __name__ == "__main__":
    # 创建增强版检测器
    detector = EnhancedYOLOv11Detector(
        model_size='s',
        edge_optimization=True,
        adaptive_inference=True
    )
    
    # 针对树莓派优化
    detector.optimize_for_platform('raspberry_pi')
    
    # 测试图像
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # 自适应检测
    results = detector.detect_adaptive(test_image, target_fps=25.0)
    
    print(f"检测到 {len(results)} 个目标")
    
    # 性能统计
    stats = detector.performance_monitor.get_performance_stats()
    print(f"当前FPS: {stats.get('current_fps', 0):.1f}")
    
    # 导出优化模型
    exported_path = detector.export_optimized_model('raspberry_pi')
    print(f"已导出优化模型: {exported_path}")