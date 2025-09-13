#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOV模型实现
基于论文《YOLOV: Making Still Image Object Detectors Great at Video Object Detection》
实现特征聚合策略，优化视频目标检测性能
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import cv2
from pathlib import Path
import time

from .base_model import BaseYOLOModel
from ..utils.logger import get_logger
from ..detection.feature_aggregation import FeatureAggregator, AggregationConfig, PlatformType
from ..detection.temporal_aggregator import TemporalAggregator, TemporalConfig, AggregationStrategy

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None

class YOLOVModel(BaseYOLOModel):
    """
    YOLOV模型实现
    
    基于YOLO基础架构，集成特征聚合策略和时序信息处理，
    专为视频目标检测场景优化，支持多平台部署。
    
    主要特性：
    - 特征聚合策略：提升检测稳定性
    - 时序信息利用：减少检测抖动
    - 多平台适配：支持PC、树莓派、Jetson等
    - 自适应优化：根据场景动态调整参数
    """
    
    SUPPORTED_FORMATS = ['.pt', '.onnx', '.engine', '.torchscript']
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 model_size: str = 's',
                 device: str = 'auto',
                 platform_type: PlatformType = PlatformType.DESKTOP,
                 enable_feature_aggregation: bool = True,
                 enable_temporal_aggregation: bool = True,
                 variant: str = 'yolov',  # 'yolov' or 'yolov++'
                 **kwargs):
        """
        初始化YOLOV模型
        
        Args:
            model_path: 模型权重路径
            model_size: 模型大小 ('n', 's', 'm', 'l', 'x')
            device: 设备类型
            platform_type: 平台类型
            enable_feature_aggregation: 启用特征聚合
            enable_temporal_aggregation: 启用时序聚合
            variant: 模型变体
            **kwargs: 其他参数
        """
        super().__init__()
        
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("需要安装ultralytics库: pip install ultralytics")
        
        self.logger = get_logger(__name__)
        self.model_size = model_size
        self.platform_type = platform_type
        self.variant = variant
        self.enable_feature_aggregation = enable_feature_aggregation
        self.enable_temporal_aggregation = enable_temporal_aggregation
        
        # 自动选择设备
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = device
        
        # 加载基础模型
        self._load_base_model(model_path)
        
        # 初始化聚合器
        self._init_aggregators(kwargs)
        
        # 性能统计
        self.inference_times = []
        self.frame_count = 0
        
        self.logger.info(f"YOLOV模型初始化完成: {variant}, 大小: {model_size}, 设备: {device}")
    
    def _load_base_model(self, model_path: Optional[str]):
        """加载基础YOLO模型"""
        try:
            if model_path and Path(model_path).exists():
                self.model = YOLO(model_path)
                self.logger.info(f"从路径加载模型: {model_path}")
            else:
                # 使用预训练模型
                model_name = f"yolov8{self.model_size}.pt"
                self.model = YOLO(model_name)
                self.logger.info(f"加载预训练模型: {model_name}")
            
            # 移动到指定设备
            self.model.to(self.device)
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise
    
    def _init_aggregators(self, kwargs: Dict[str, Any]):
        """初始化聚合器"""
        # 特征聚合器
        if self.enable_feature_aggregation:
            agg_config = AggregationConfig(
                platform_type=self.platform_type,
                buffer_size=kwargs.get('aggregation_buffer_size', 5),
                confidence_threshold=kwargs.get('confidence_threshold', 0.25),
                enable_motion_analysis=kwargs.get('enable_motion_analysis', True)
            )
            self.feature_aggregator = FeatureAggregator(agg_config)
        else:
            self.feature_aggregator = None
        
        # 时序聚合器
        if self.enable_temporal_aggregation:
            temporal_config = TemporalConfig(
                buffer_size=kwargs.get('temporal_buffer_size', 5),
                confidence_threshold=kwargs.get('confidence_threshold', 0.25),
                strategy=AggregationStrategy.ADAPTIVE,
                enable_tracking=kwargs.get('enable_tracking', True)
            )
            self.temporal_aggregator = TemporalAggregator(temporal_config)
        else:
            self.temporal_aggregator = None
    
    def predict(self, image: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        """
        执行预测
        
        Args:
            image: 输入图像
            **kwargs: 其他参数
        
        Returns:
            检测结果列表
        """
        start_time = time.time()
        
        try:
            # 基础检测
            results = self.model(image, **kwargs)
            
            # 转换结果格式
            detections = self._parse_results(results)
            
            # 应用YOLOV特征聚合
            if self.enable_feature_aggregation or self.enable_temporal_aggregation:
                detections = self._apply_yolov_aggregation(detections, image)
            
            # 记录性能
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.frame_count += 1
            
            # 保持最近100次的记录
            if len(self.inference_times) > 100:
                self.inference_times = self.inference_times[-100:]
            
            return detections
            
        except Exception as e:
            self.logger.error(f"预测失败: {e}")
            return []
    
    def _parse_results(self, results) -> List[Dict[str, Any]]:
        """解析YOLO结果"""
        detections = []
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                for i in range(len(boxes)):
                    # 提取边界框信息
                    bbox = boxes.xyxy[i].cpu().numpy()
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    class_name = result.names.get(class_id, f"class_{class_id}")
                    
                    detection = {
                        'bbox': tuple(bbox),
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name,
                        'timestamp': time.time(),
                        'frame_id': self.frame_count
                    }
                    
                    detections.append(detection)
        
        return detections
    
    def _apply_yolov_aggregation(self, detections: List[Dict[str, Any]], 
                                image: np.ndarray) -> List[Dict[str, Any]]:
        """应用YOLOV特征聚合策略"""
        try:
            # 转换为DetectionResult格式
            from ..detection.temporal_aggregator import DetectionResult
            
            detection_results = []
            for det in detections:
                detection_results.append(DetectionResult(
                    bbox=det['bbox'],
                    confidence=det['confidence'],
                    class_id=det['class_id'],
                    class_name=det['class_name'],
                    timestamp=det['timestamp'],
                    frame_id=det['frame_id']
                ))
            
            # 应用时序聚合
            if self.temporal_aggregator:
                aggregated_results = self.temporal_aggregator.add_detections(
                    detection_results, image
                )
            else:
                aggregated_results = detection_results
            
            # 转换回字典格式
            final_detections = []
            for result in aggregated_results:
                final_detections.append({
                    'bbox': result.bbox,
                    'confidence': result.confidence,
                    'class_id': result.class_id,
                    'class_name': result.class_name,
                    'timestamp': result.timestamp,
                    'frame_id': result.frame_id,
                    'track_id': getattr(result, 'track_id', None)
                })
            
            return final_detections
            
        except Exception as e:
            self.logger.warning(f"聚合处理失败，使用原始结果: {e}")
            return detections
    
    def predict_batch(self, images: List[np.ndarray], **kwargs) -> List[List[Dict[str, Any]]]:
        """批量预测"""
        results = []
        for image in images:
            result = self.predict(image, **kwargs)
            results.append(result)
        return results
    
    def predict_video(self, video_path: str, output_path: Optional[str] = None, 
                     **kwargs) -> List[List[Dict[str, Any]]]:
        """视频预测 - YOLOV优化版本"""
        cap = cv2.VideoCapture(video_path)
        results = []
        
        # 视频写入器
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 执行检测
            detections = self.predict(frame, **kwargs)
            results.append(detections)
            
            # 绘制结果并保存
            if writer:
                annotated_frame = self.draw_results(frame, detections)
                writer.write(annotated_frame)
            
            frame_idx += 1
            
            # 进度日志
            if frame_idx % 100 == 0:
                self.logger.info(f"已处理 {frame_idx} 帧")
        
        cap.release()
        if writer:
            writer.release()
        
        self.logger.info(f"视频处理完成，共 {frame_idx} 帧")
        return results
    
    def draw_results(self, image: np.ndarray, results: List[Dict[str, Any]]) -> np.ndarray:
        """绘制检测结果"""
        annotated_image = image.copy()
        
        for result in results:
            bbox = result['bbox']
            confidence = result['confidence']
            class_name = result['class_name']
            track_id = result.get('track_id')
            
            # 绘制边界框
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"{class_name}: {confidence:.2f}"
            if track_id is not None:
                label += f" ID:{track_id}"
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated_image
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            'model_type': f'YOLOV-{self.variant}',
            'model_size': self.model_size,
            'device': self.device,
            'platform_type': self.platform_type.value,
            'feature_aggregation_enabled': self.enable_feature_aggregation,
            'temporal_aggregation_enabled': self.enable_temporal_aggregation,
            'frame_count': self.frame_count,
            'supported_formats': self.SUPPORTED_FORMATS
        }
        
        # 性能统计
        if self.inference_times:
            info['performance'] = {
                'avg_inference_time': np.mean(self.inference_times),
                'min_inference_time': np.min(self.inference_times),
                'max_inference_time': np.max(self.inference_times),
                'avg_fps': 1.0 / np.mean(self.inference_times)
            }
        
        # 聚合器统计
        if self.feature_aggregator:
            info['feature_aggregation_stats'] = self.feature_aggregator.get_statistics()
        
        if self.temporal_aggregator:
            info['temporal_aggregation_stats'] = self.temporal_aggregator.get_statistics()
        
        return info
    
    def reset_aggregators(self):
        """重置聚合器状态"""
        if self.feature_aggregator:
            self.feature_aggregator.reset()
        
        if self.temporal_aggregator:
            self.temporal_aggregator.reset()
        
        self.frame_count = 0
        self.inference_times.clear()
        
        self.logger.info("聚合器状态已重置")
    
    def optimize_for_platform(self, platform_type: PlatformType):
        """针对特定平台优化模型"""
        self.platform_type = platform_type
        
        # 根据平台调整配置
        if platform_type in [PlatformType.RASPBERRY_PI, PlatformType.ESP32]:
            # 边缘设备优化
            if self.feature_aggregator:
                self.feature_aggregator.config.buffer_size = 3
                self.feature_aggregator.config.enable_motion_analysis = False
            
            if self.temporal_aggregator:
                self.temporal_aggregator.config.buffer_size = 3
        
        elif platform_type == PlatformType.DESKTOP:
            # 桌面设备优化
            if self.feature_aggregator:
                self.feature_aggregator.config.buffer_size = 5
                self.feature_aggregator.config.enable_motion_analysis = True
            
            if self.temporal_aggregator:
                self.temporal_aggregator.config.buffer_size = 5
        
        self.logger.info(f"已针对平台 {platform_type.value} 优化模型配置")
    
    def __del__(self):
        """析构函数"""
        try:
            if hasattr(self, 'model'):
                del self.model
        except:
            pass

# 使用示例
if __name__ == "__main__":
    # 创建YOLOV模型
    model = YOLOVModel(
        model_size='s',
        device='auto',
        platform_type=PlatformType.DESKTOP,
        enable_feature_aggregation=True,
        enable_temporal_aggregation=True,
        variant='yolov++'
    )
    
    # 模拟图像
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # 执行检测
    results = model.predict(test_image)
    print(f"检测到 {len(results)} 个目标")
    
    # 获取模型信息
    info = model.get_model_info()
    print(f"模型信息: {info}")