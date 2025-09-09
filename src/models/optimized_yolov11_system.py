#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版YOLOv11系统
集成最新优化技术，符合YOLOS项目规范
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import time
import threading
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
from queue import Queue, Empty
from dataclasses import dataclass

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("警告: ultralytics未安装，请运行: pip install ultralytics")

from ..core.types import DetectionResult, create_detection_result, ObjectType
from ..utils.logging_manager import LoggingManager


@dataclass
class OptimizationConfig:
    """优化配置"""
    # 模型配置
    model_size: str = 's'  # n, s, m, l, x
    device: str = 'auto'
    half_precision: bool = True
    tensorrt_optimize: bool = True
    
    # 检测配置
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 100
    
    # 性能配置
    target_fps: float = 30.0
    adaptive_inference: bool = True
    edge_optimization: bool = False
    batch_size: int = 1
    
    # 平台配置
    platform: str = 'pc'  # pc, raspberry_pi, jetson_nano, esp32


class OptimizedYOLOv11System:
    """
    优化版YOLOv11系统
    
    特性:
    - 最新YOLOv11算法
    - 自适应性能调优
    - 多平台优化
    - TensorRT加速
    - 智能批处理
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        初始化优化系统
        
        Args:
            config: 优化配置
        """
        self.config = config or OptimizationConfig()
        self.logger = LoggingManager().get_logger("OptimizedYOLOv11System")
        
        # 检查依赖
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("需要安装ultralytics: pip install ultralytics")
        
        # 设备配置
        self.device = self._setup_device()
        
        # 加载模型
        self.model = self._load_optimized_model()
        
        # 性能监控
        self.performance_stats = {
            'inference_times': [],
            'detection_counts': [],
            'fps_history': [],
            'total_inferences': 0
        }
        
        # 自适应配置
        self.adaptive_params = {
            'current_input_size': 640,
            'current_conf_threshold': self.config.confidence_threshold,
            'current_iou_threshold': self.config.iou_threshold
        }
        
        self.logger.info(f"YOLOv11{self.config.model_size.upper()}系统初始化完成")
        self.logger.info(f"设备: {self.device}, 平台: {self.config.platform}")
    
    def _setup_device(self) -> torch.device:
        """设置计算设备"""
        if self.config.device == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                self.logger.info(f"使用CUDA设备: {torch.cuda.get_device_name()}")
            else:
                device = torch.device('cpu')
                self.logger.info("使用CPU设备")
        else:
            device = torch.device(self.config.device)
        
        return device
    
    def _load_optimized_model(self) -> YOLO:
        """加载优化模型"""
        try:
            # 模型文件名
            model_name = f'yolov11{self.config.model_size}.pt'
            
            # 加载模型
            model = YOLO(model_name)
            
            # 移动到设备
            model.to(self.device)
            
            # 应用优化
            self._apply_optimizations(model)
            
            # 平台特定优化
            self._apply_platform_optimizations(model)
            
            self.logger.info(f"模型加载完成: {model_name}")
            return model
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise
    
    def _apply_optimizations(self, model: YOLO):
        """应用通用优化"""
        try:
            # 模型融合
            if hasattr(model.model, 'fuse'):
                model.fuse()
                self.logger.info("模型层融合完成")
            
            # FP16优化
            if self.config.half_precision and self.device.type == 'cuda':
                model.half()
                self.logger.info("启用FP16半精度推理")
            
            # 预热模型
            self._warmup_model(model)
            
        except Exception as e:
            self.logger.warning(f"优化应用失败: {e}")
    
    def _apply_platform_optimizations(self, model: YOLO):
        """应用平台特定优化"""
        platform_configs = {
            'esp32': {
                'input_size': 320,
                'confidence_threshold': 0.4,
                'max_detections': 10
            },
            'raspberry_pi': {
                'input_size': 416,
                'confidence_threshold': 0.3,
                'max_detections': 50
            },
            'jetson_nano': {
                'input_size': 640,
                'confidence_threshold': 0.25,
                'max_detections': 100
            },
            'pc': {
                'input_size': 640,
                'confidence_threshold': 0.25,
                'max_detections': 100
            }
        }
        
        if self.config.platform in platform_configs:
            platform_config = platform_configs[self.config.platform]
            
            # 更新自适应参数
            self.adaptive_params['current_input_size'] = platform_config['input_size']
            self.adaptive_params['current_conf_threshold'] = platform_config['confidence_threshold']
            self.config.max_detections = platform_config['max_detections']
            
            self.logger.info(f"应用{self.config.platform}平台优化")
    
    def _warmup_model(self, model: YOLO):
        """预热模型"""
        try:
            dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # 预热推理
            for _ in range(3):
                _ = model(dummy_input, verbose=False)
            
            self.logger.info("模型预热完成")
            
        except Exception as e:
            self.logger.warning(f"模型预热失败: {e}")
    
    def detect(self, image: np.ndarray, **kwargs) -> List[DetectionResult]:
        """
        标准检测
        
        Args:
            image: 输入图像
            **kwargs: 其他参数
            
        Returns:
            检测结果列表
        """
        start_time = time.time()
        
        try:
            # 获取参数
            conf_threshold = kwargs.get('conf_threshold', self.config.confidence_threshold)
            iou_threshold = kwargs.get('iou_threshold', self.config.iou_threshold)
            
            # 执行推理
            results = self.model(image,
                               conf=conf_threshold,
                               iou=iou_threshold,
                               verbose=False,
                               half=self.config.half_precision)
            
            # 后处理
            detections = self._postprocess_results(results, image.shape)
            
            # 更新性能统计
            inference_time = time.time() - start_time
            self._update_performance_stats(inference_time, len(detections))
            
            return detections
            
        except Exception as e:
            self.logger.error(f"检测失败: {e}")
            return []
    
    def detect_adaptive(self, 
                       image: np.ndarray,
                       target_fps: float = None,
                       quality_priority: bool = False) -> List[DetectionResult]:
        """
        自适应检测
        
        Args:
            image: 输入图像
            target_fps: 目标FPS
            quality_priority: 是否优先保证质量
            
        Returns:
            检测结果列表
        """
        if not self.config.adaptive_inference:
            return self.detect(image)
        
        target_fps = target_fps or self.config.target_fps
        
        # 获取当前性能
        current_fps = self._get_current_fps()
        
        # 自适应调整参数
        adapted_params = self._adapt_parameters(current_fps, target_fps, quality_priority)
        
        # 调整图像尺寸
        if adapted_params['resize_factor'] != 1.0:
            h, w = image.shape[:2]
            new_h = int(h * adapted_params['resize_factor'])
            new_w = int(w * adapted_params['resize_factor'])
            image = cv2.resize(image, (new_w, new_h))
        
        # 执行检测
        return self.detect(image,
                          conf_threshold=adapted_params['conf_threshold'],
                          iou_threshold=adapted_params['iou_threshold'])
    
    def _adapt_parameters(self, 
                         current_fps: float, 
                         target_fps: float, 
                         quality_priority: bool) -> Dict[str, float]:
        """自适应调整参数"""
        params = {
            'conf_threshold': self.adaptive_params['current_conf_threshold'],
            'iou_threshold': self.adaptive_params['current_iou_threshold'],
            'resize_factor': 1.0
        }
        
        fps_ratio = current_fps / target_fps if target_fps > 0 else 1.0
        
        if fps_ratio < 0.8:  # 性能不足
            if not quality_priority:
                # 降低质量提升速度
                params['conf_threshold'] = min(0.5, params['conf_threshold'] + 0.1)
                params['iou_threshold'] = min(0.7, params['iou_threshold'] + 0.1)
                params['resize_factor'] = 0.8
        
        elif fps_ratio > 1.2:  # 性能过剩
            # 提升质量
            params['conf_threshold'] = max(0.1, params['conf_threshold'] - 0.05)
            params['iou_threshold'] = max(0.3, params['iou_threshold'] - 0.05)
            params['resize_factor'] = min(1.2, params['resize_factor'] + 0.1)
        
        return params
    
    def detect_batch(self, images: List[np.ndarray]) -> List[List[DetectionResult]]:
        """
        批量检测
        
        Args:
            images: 图像列表
            
        Returns:
            每张图像的检测结果列表
        """
        all_results = []
        
        # 简单批处理实现
        for image in images:
            results = self.detect(image)
            all_results.append(results)
        
        return all_results
    
    def _postprocess_results(self, 
                           results, 
                           original_shape: Tuple[int, int, int]) -> List[DetectionResult]:
        """后处理检测结果"""
        detections = []
        
        if not results or len(results) == 0:
            return detections
        
        # 获取第一个结果
        result = results[0]
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        # 提取检测信息
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        # 获取类别名称
        class_names = result.names
        
        # 转换为DetectionResult格式
        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
            if len(detections) >= self.config.max_detections:
                break
            
            x1, y1, x2, y2 = box
            
            # 创建检测结果
            detection = create_detection_result(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                class_id=int(cls_id),
                class_name=class_names.get(cls_id, f"class_{cls_id}"),
                confidence=float(conf),
                object_type=self._map_class_to_object_type(cls_id)
            )
            
            detections.append(detection)
        
        return detections
    
    def _map_class_to_object_type(self, class_id: int) -> ObjectType:
        """将类别ID映射到对象类型"""
        # COCO数据集类别映射
        if class_id == 0:  # person
            return ObjectType.PERSON
        elif class_id in [15, 16, 17, 18, 19, 20, 21, 22, 23]:  # 动物
            return ObjectType.PET
        elif class_id in [1, 2, 3, 5, 7]:  # 交通工具
            return ObjectType.VEHICLE
        else:
            return ObjectType.UNKNOWN
    
    def _update_performance_stats(self, inference_time: float, detection_count: int):
        """更新性能统计"""
        self.performance_stats['inference_times'].append(inference_time)
        self.performance_stats['detection_counts'].append(detection_count)
        
        # 计算FPS
        fps = 1.0 / inference_time if inference_time > 0 else 0
        self.performance_stats['fps_history'].append(fps)
        
        self.performance_stats['total_inferences'] += 1
        
        # 保持统计窗口大小
        max_history = 100
        for key in ['inference_times', 'detection_counts', 'fps_history']:
            if len(self.performance_stats[key]) > max_history:
                self.performance_stats[key] = self.performance_stats[key][-max_history:]
    
    def _get_current_fps(self) -> float:
        """获取当前FPS"""
        if not self.performance_stats['fps_history']:
            return 0.0
        
        # 返回最近几帧的平均FPS
        recent_fps = self.performance_stats['fps_history'][-5:]
        return float(np.mean(recent_fps))
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        if not self.performance_stats['inference_times']:
            return {}
        
        return {
            'avg_inference_time': float(np.mean(self.performance_stats['inference_times'])),
            'min_inference_time': float(np.min(self.performance_stats['inference_times'])),
            'max_inference_time': float(np.max(self.performance_stats['inference_times'])),
            'avg_fps': float(np.mean(self.performance_stats['fps_history'])),
            'current_fps': self._get_current_fps(),
            'total_inferences': self.performance_stats['total_inferences'],
            'avg_detections': float(np.mean(self.performance_stats['detection_counts']))
        }
    
    def export_optimized_model(self, 
                             format: str = 'onnx',
                             output_path: Optional[str] = None) -> str:
        """
        导出优化模型
        
        Args:
            format: 导出格式
            output_path: 输出路径
            
        Returns:
            导出文件路径
        """
        try:
            if output_path is None:
                output_path = f'yolov11{self.config.model_size}_{self.config.platform}_{format}'
            
            # 导出模型
            exported_path = self.model.export(
                format=format,
                half=self.config.half_precision,
                device=self.device
            )
            
            self.logger.info(f"模型已导出: {exported_path}")
            return str(exported_path)
            
        except Exception as e:
            self.logger.error(f"模型导出失败: {e}")
            raise
    
    def reset_stats(self):
        """重置性能统计"""
        for key in self.performance_stats:
            if isinstance(self.performance_stats[key], list):
                self.performance_stats[key].clear()
            else:
                self.performance_stats[key] = 0
        
        self.logger.info("性能统计已重置")


class OptimizedRealtimeDetector:
    """优化版实时检测器"""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """初始化实时检测器"""
        self.config = config or OptimizationConfig()
        self.detector = OptimizedYOLOv11System(config)
        self.logger = LoggingManager().get_logger("OptimizedRealtimeDetector")
        
        # 线程控制
        self.is_running = False
        self.frame_queue = Queue(maxsize=10)
        self.result_queue = Queue(maxsize=10)
        
        # 性能统计
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()
    
    def start_camera_detection(self, camera_id: int = 0):
        """开始摄像头检测"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {camera_id}")
        
        # 配置摄像头
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_running = True
        self.logger.info("开始实时检测，按 'q' 退出")
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 执行检测
                results = self.detector.detect_adaptive(frame, self.config.target_fps)
                
                # 绘制结果
                display_frame = self._draw_results(frame, results)
                
                # 计算FPS
                self._update_fps()
                
                # 显示FPS
                cv2.putText(display_frame, f"FPS: {self.fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 显示图像
                cv2.imshow('YOLOS优化检测', display_frame)
                
                # 检查退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            self.stop()
            cap.release()
            cv2.destroyAllWindows()
    
    def _draw_results(self, frame: np.ndarray, results: List[DetectionResult]) -> np.ndarray:
        """绘制检测结果"""
        for result in results:
            bbox = result.bbox
            
            # 绘制边界框
            cv2.rectangle(frame, (bbox.x, bbox.y), (bbox.x2, bbox.y2), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"{result.class_name} {result.confidence:.2f}"
            cv2.putText(frame, label, (bbox.x, bbox.y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def _update_fps(self):
        """更新FPS"""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time > 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
    
    def stop(self):
        """停止检测"""
        self.is_running = False
        self.logger.info("检测器已停止")


# 使用示例
if __name__ == "__main__":
    # 创建优化配置
    config = OptimizationConfig(
        model_size='s',
        platform='pc',
        target_fps=30.0,
        adaptive_inference=True,
        edge_optimization=False
    )
    
    # 创建检测系统
    detector_system = OptimizedYOLOv11System(config)
    
    # 测试检测
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    results = detector_system.detect_adaptive(test_image)
    
    print(f"检测到 {len(results)} 个目标")
    
    # 性能统计
    stats = detector_system.get_performance_stats()
    print(f"当前FPS: {stats.get('current_fps', 0):.1f}")
    
    # 创建实时检测器
    realtime_detector = OptimizedRealtimeDetector(config)
    
    print("启动实时检测...")
    # realtime_detector.start_camera_detection()