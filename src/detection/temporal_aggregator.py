#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多帧时序信息聚合器
基于YOLOV论文思想，实现视频目标检测的时序信息聚合功能
提升检测稳定性和准确性，支持多平台部署
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import cv2
import time
from enum import Enum

class AggregationStrategy(Enum):
    """聚合策略枚举"""
    WEIGHTED_AVERAGE = "weighted_average"  # 加权平均
    CONFIDENCE_BASED = "confidence_based"  # 基于置信度
    MOTION_AWARE = "motion_aware"  # 运动感知
    ADAPTIVE = "adaptive"  # 自适应

@dataclass
class DetectionResult:
    """检测结果数据结构"""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    timestamp: float
    frame_id: int
    
    def __post_init__(self):
        """后处理初始化"""
        if self.timestamp == 0:
            self.timestamp = time.time()

@dataclass
class TemporalConfig:
    """时序聚合配置"""
    buffer_size: int = 5  # 缓冲区大小
    confidence_threshold: float = 0.5  # 置信度阈值
    iou_threshold: float = 0.5  # IoU阈值
    temporal_weight_decay: float = 0.8  # 时序权重衰减
    motion_threshold: float = 0.1  # 运动阈值
    stability_frames: int = 3  # 稳定性所需帧数
    strategy: AggregationStrategy = AggregationStrategy.ADAPTIVE
    enable_tracking: bool = True  # 启用跟踪
    max_missing_frames: int = 5  # 最大丢失帧数

class MotionEstimator:
    """运动估计器"""
    
    def __init__(self):
        self.prev_frame = None
        self.optical_flow = cv2.createOptFlow_DIS(cv2.DISOpticalFlow_PRESET_MEDIUM)
    
    def estimate_motion(self, frame: np.ndarray) -> float:
        """估计帧间运动强度"""
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return 0.0
        
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        try:
            # 计算光流
            flow = cv2.calcOpticalFlowPyrLK(
                self.prev_frame, current_gray, None, None
            )
            
            if flow is not None:
                # 计算运动强度
                motion_magnitude = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
                self.prev_frame = current_gray
                return float(motion_magnitude)
        except Exception:
            pass
        
        self.prev_frame = current_gray
        return 0.0

class ObjectTracker:
    """简单的目标跟踪器"""
    
    def __init__(self, max_missing_frames: int = 5):
        self.tracks = {}  # track_id -> track_info
        self.next_track_id = 0
        self.max_missing_frames = max_missing_frames
    
    def update(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """更新跟踪状态"""
        # 简单的基于IoU的跟踪
        matched_detections = []
        
        for detection in detections:
            best_track_id = None
            best_iou = 0.0
            
            # 寻找最佳匹配的轨迹
            for track_id, track_info in self.tracks.items():
                if track_info['class_id'] == detection.class_id:
                    iou = self._calculate_iou(detection.bbox, track_info['bbox'])
                    if iou > best_iou and iou > 0.3:
                        best_iou = iou
                        best_track_id = track_id
            
            if best_track_id is not None:
                # 更新现有轨迹
                self.tracks[best_track_id].update({
                    'bbox': detection.bbox,
                    'confidence': detection.confidence,
                    'last_seen': detection.frame_id,
                    'missing_frames': 0
                })
                detection.track_id = best_track_id
            else:
                # 创建新轨迹
                track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[track_id] = {
                    'bbox': detection.bbox,
                    'confidence': detection.confidence,
                    'class_id': detection.class_id,
                    'last_seen': detection.frame_id,
                    'missing_frames': 0
                }
                detection.track_id = track_id
            
            matched_detections.append(detection)
        
        # 清理丢失的轨迹
        self._cleanup_lost_tracks()
        
        return matched_detections
    
    def _calculate_iou(self, bbox1: Tuple[float, float, float, float], 
                      bbox2: Tuple[float, float, float, float]) -> float:
        """计算IoU"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 计算交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _cleanup_lost_tracks(self):
        """清理丢失的轨迹"""
        tracks_to_remove = []
        for track_id, track_info in self.tracks.items():
            track_info['missing_frames'] += 1
            if track_info['missing_frames'] > self.max_missing_frames:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]

class TemporalAggregator:
    """时序信息聚合器"""
    
    def __init__(self, config: TemporalConfig = None):
        self.config = config or TemporalConfig()
        self.detection_buffer = deque(maxlen=self.config.buffer_size)
        self.motion_estimator = MotionEstimator()
        self.tracker = ObjectTracker(self.config.max_missing_frames) if self.config.enable_tracking else None
        self.frame_count = 0
        self.last_motion = 0.0
    
    def add_detections(self, detections: List[DetectionResult], frame: np.ndarray = None) -> List[DetectionResult]:
        """添加检测结果并进行时序聚合"""
        self.frame_count += 1
        
        # 估计运动
        motion_strength = 0.0
        if frame is not None:
            motion_strength = self.motion_estimator.estimate_motion(frame)
            self.last_motion = motion_strength
        
        # 更新跟踪
        if self.tracker:
            detections = self.tracker.update(detections)
        
        # 添加到缓冲区
        frame_data = {
            'detections': detections,
            'frame_id': self.frame_count,
            'timestamp': time.time(),
            'motion_strength': motion_strength
        }
        self.detection_buffer.append(frame_data)
        
        # 执行时序聚合
        return self._aggregate_detections()
    
    def _aggregate_detections(self) -> List[DetectionResult]:
        """执行时序聚合"""
        if len(self.detection_buffer) < 2:
            return self.detection_buffer[-1]['detections'] if self.detection_buffer else []
        
        if self.config.strategy == AggregationStrategy.WEIGHTED_AVERAGE:
            return self._weighted_average_aggregation()
        elif self.config.strategy == AggregationStrategy.CONFIDENCE_BASED:
            return self._confidence_based_aggregation()
        elif self.config.strategy == AggregationStrategy.MOTION_AWARE:
            return self._motion_aware_aggregation()
        else:  # ADAPTIVE
            return self._adaptive_aggregation()
    
    def _weighted_average_aggregation(self) -> List[DetectionResult]:
        """加权平均聚合"""
        # 收集所有检测结果
        all_detections = {}
        
        for i, frame_data in enumerate(self.detection_buffer):
            weight = self.config.temporal_weight_decay ** (len(self.detection_buffer) - 1 - i)
            
            for detection in frame_data['detections']:
                key = (detection.class_id, getattr(detection, 'track_id', -1))
                
                if key not in all_detections:
                    all_detections[key] = {
                        'bboxes': [],
                        'confidences': [],
                        'weights': [],
                        'class_name': detection.class_name,
                        'class_id': detection.class_id
                    }
                
                all_detections[key]['bboxes'].append(detection.bbox)
                all_detections[key]['confidences'].append(detection.confidence)
                all_detections[key]['weights'].append(weight)
        
        # 聚合结果
        aggregated_results = []
        for key, data in all_detections.items():
            if len(data['bboxes']) >= self.config.stability_frames:
                # 计算加权平均
                weights = np.array(data['weights'])
                weights = weights / np.sum(weights)
                
                # 边界框加权平均
                avg_bbox = np.average(data['bboxes'], axis=0, weights=weights)
                # 置信度加权平均
                avg_confidence = np.average(data['confidences'], weights=weights)
                
                if avg_confidence >= self.config.confidence_threshold:
                    result = DetectionResult(
                        bbox=tuple(avg_bbox),
                        confidence=float(avg_confidence),
                        class_id=data['class_id'],
                        class_name=data['class_name'],
                        timestamp=time.time(),
                        frame_id=self.frame_count
                    )
                    if hasattr(self, 'track_id'):
                        result.track_id = key[1]
                    aggregated_results.append(result)
        
        return aggregated_results
    
    def _confidence_based_aggregation(self) -> List[DetectionResult]:
        """基于置信度的聚合"""
        # 选择置信度最高的检测结果
        best_detections = {}
        
        for frame_data in self.detection_buffer:
            for detection in frame_data['detections']:
                key = (detection.class_id, getattr(detection, 'track_id', -1))
                
                if key not in best_detections or detection.confidence > best_detections[key].confidence:
                    best_detections[key] = detection
        
        return [det for det in best_detections.values() 
                if det.confidence >= self.config.confidence_threshold]
    
    def _motion_aware_aggregation(self) -> List[DetectionResult]:
        """运动感知聚合"""
        # 根据运动强度调整聚合策略
        if self.last_motion > self.config.motion_threshold:
            # 高运动场景，使用最新检测结果
            return self.detection_buffer[-1]['detections']
        else:
            # 低运动场景，使用加权平均
            return self._weighted_average_aggregation()
    
    def _adaptive_aggregation(self) -> List[DetectionResult]:
        """自适应聚合"""
        # 根据场景特征选择最佳策略
        motion_ratio = self.last_motion / (self.config.motion_threshold + 1e-6)
        
        if motion_ratio > 2.0:
            # 高运动场景
            return self._confidence_based_aggregation()
        elif motion_ratio > 0.5:
            # 中等运动场景
            return self._motion_aware_aggregation()
        else:
            # 低运动场景
            return self._weighted_average_aggregation()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'buffer_size': len(self.detection_buffer),
            'frame_count': self.frame_count,
            'last_motion': self.last_motion,
            'active_tracks': len(self.tracker.tracks) if self.tracker else 0,
            'config': {
                'strategy': self.config.strategy.value,
                'buffer_size': self.config.buffer_size,
                'confidence_threshold': self.config.confidence_threshold
            }
        }
    
    def reset(self):
        """重置聚合器状态"""
        self.detection_buffer.clear()
        self.frame_count = 0
        self.last_motion = 0.0
        if self.tracker:
            self.tracker.tracks.clear()
            self.tracker.next_track_id = 0

# 使用示例
if __name__ == "__main__":
    # 创建配置
    config = TemporalConfig(
        buffer_size=5,
        confidence_threshold=0.5,
        strategy=AggregationStrategy.ADAPTIVE,
        enable_tracking=True
    )
    
    # 创建聚合器
    aggregator = TemporalAggregator(config)
    
    # 模拟检测结果
    detections = [
        DetectionResult(
            bbox=(100, 100, 200, 200),
            confidence=0.8,
            class_id=0,
            class_name="person",
            timestamp=time.time(),
            frame_id=1
        )
    ]
    
    # 添加检测结果
    aggregated = aggregator.add_detections(detections)
    print(f"聚合后检测结果数量: {len(aggregated)}")
    
    # 获取统计信息
    stats = aggregator.get_statistics()
    print(f"统计信息: {stats}")