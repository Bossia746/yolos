#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跟踪系统集成模块
将多目标跟踪功能集成到现有的检测系统中
支持视频检测、实时检测等多种场景
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .multi_object_tracker import MultiObjectTracker, TrackingConfig, Detection, Track, TrackingStrategy
from ..detection.feature_aggregation import FeatureAggregator, AggregationConfig
from ..detection.temporal_aggregator import TemporalAggregator, TemporalConfig

logger = logging.getLogger(__name__)

class TrackingMode(Enum):
    """跟踪模式"""
    DISABLED = "disabled"  # 禁用跟踪
    BASIC = "basic"  # 基础跟踪
    ENHANCED = "enhanced"  # 增强跟踪（含特征聚合）
    TEMPORAL = "temporal"  # 时序跟踪（含时序聚合）
    FULL = "full"  # 完整跟踪（所有功能）

@dataclass
class IntegratedTrackingConfig:
    """集成跟踪配置"""
    mode: TrackingMode = TrackingMode.ENHANCED
    tracking_config: Optional[TrackingConfig] = None
    aggregation_config: Optional[AggregationConfig] = None
    temporal_config: Optional[TemporalConfig] = None
    
    # 性能优化
    enable_feature_extraction: bool = True
    feature_extraction_interval: int = 1  # 特征提取间隔
    tracking_interval: int = 1  # 跟踪更新间隔
    
    # 平台适配
    platform_optimization: bool = True
    max_concurrent_tracks: int = 50
    memory_limit_mb: int = 512

class TrackingIntegration:
    """跟踪系统集成器"""
    
    def __init__(self, config: Optional[IntegratedTrackingConfig] = None):
        self.config = config or IntegratedTrackingConfig()
        
        # 初始化组件
        self.tracker = None
        self.feature_aggregator = None
        self.temporal_aggregator = None
        
        self._initialize_components()
        
        # 状态管理
        self.frame_count = 0
        self.last_tracking_frame = 0
        self.last_feature_frame = 0
        
        # 性能统计
        self.performance_stats = {
            'tracking_time': 0.0,
            'feature_time': 0.0,
            'total_time': 0.0,
            'tracks_per_second': 0.0,
            'memory_usage_mb': 0.0
        }
        
        logger.info(f"跟踪集成系统初始化完成，模式: {self.config.mode.value}")
    
    def _initialize_components(self):
        """初始化组件"""
        if self.config.mode == TrackingMode.DISABLED:
            return
        
        # 初始化跟踪器
        tracking_config = self.config.tracking_config or TrackingConfig()
        if self.config.platform_optimization:
            tracking_config.max_tracks = self.config.max_concurrent_tracks
        
        self.tracker = MultiObjectTracker(tracking_config)
        
        # 初始化特征聚合器
        if self.config.mode in [TrackingMode.ENHANCED, TrackingMode.FULL]:
            aggregation_config = self.config.aggregation_config or AggregationConfig()
            self.feature_aggregator = FeatureAggregator(aggregation_config)
        
        # 初始化时序聚合器
        if self.config.mode in [TrackingMode.TEMPORAL, TrackingMode.FULL]:
            temporal_config = self.config.temporal_config or TemporalConfig()
            self.temporal_aggregator = TemporalAggregator(temporal_config)
    
    def process_detections(self, detections: List[Dict[str, Any]], 
                          frame: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """处理检测结果，添加跟踪信息"""
        if self.config.mode == TrackingMode.DISABLED or not detections:
            return detections
        
        start_time = time.time()
        self.frame_count += 1
        
        try:
            # 转换检测结果格式
            detection_objects = self._convert_to_detection_objects(detections)
            
            # 特征提取和聚合
            if self._should_extract_features():
                detection_objects = self._extract_and_aggregate_features(
                    detection_objects, frame
                )
            
            # 执行跟踪
            if self._should_update_tracking():
                tracked_detections = self.tracker.update(detection_objects)
            else:
                tracked_detections = detection_objects
            
            # 时序聚合
            if self.temporal_aggregator and self.config.mode in [TrackingMode.TEMPORAL, TrackingMode.FULL]:
                tracked_detections = self._apply_temporal_aggregation(tracked_detections)
            
            # 转换回原格式
            result = self._convert_to_dict_format(tracked_detections)
            
            # 更新性能统计
            self._update_performance_stats(time.time() - start_time)
            
            return result
            
        except Exception as e:
            logger.error(f"跟踪处理错误: {e}")
            return detections
    
    def _convert_to_detection_objects(self, detections: List[Dict[str, Any]]) -> List[Detection]:
        """转换检测结果为Detection对象"""
        detection_objects = []
        
        for det in detections:
            # 处理不同的边界框格式
            if 'bbox' in det:
                bbox = det['bbox']
            elif 'box' in det:
                bbox = det['box']
            elif all(k in det for k in ['x1', 'y1', 'x2', 'y2']):
                bbox = [det['x1'], det['y1'], det['x2'], det['y2']]
            elif all(k in det for k in ['x', 'y', 'w', 'h']):
                bbox = [det['x'], det['y'], det['x'] + det['w'], det['y'] + det['h']]
            else:
                logger.warning(f"无法解析边界框格式: {det}")
                continue
            
            # 确保bbox格式正确
            if len(bbox) == 4:
                bbox = tuple(float(x) for x in bbox)
            else:
                continue
            
            detection_obj = Detection(
                bbox=bbox,
                confidence=float(det.get('confidence', det.get('conf', 0.0))),
                class_id=int(det.get('class_id', det.get('class', 0))),
                class_name=str(det.get('class_name', det.get('name', 'unknown'))),
                frame_id=self.frame_count,
                features=det.get('features')
            )
            
            detection_objects.append(detection_obj)
        
        return detection_objects
    
    def _should_extract_features(self) -> bool:
        """判断是否应该提取特征"""
        if not self.config.enable_feature_extraction:
            return False
        
        return (self.frame_count - self.last_feature_frame) >= self.config.feature_extraction_interval
    
    def _should_update_tracking(self) -> bool:
        """判断是否应该更新跟踪"""
        return (self.frame_count - self.last_tracking_frame) >= self.config.tracking_interval
    
    def _extract_and_aggregate_features(self, detections: List[Detection], 
                                       frame: Optional[np.ndarray]) -> List[Detection]:
        """提取和聚合特征"""
        if not self.feature_aggregator or frame is None:
            return detections
        
        feature_start = time.time()
        
        try:
            # 为每个检测结果提取特征
            for detection in detections:
                if detection.features is None:
                    # 从边界框区域提取特征
                    x1, y1, x2, y2 = [int(x) for x in detection.bbox]
                    roi = frame[y1:y2, x1:x2]
                    
                    if roi.size > 0:
                        # 简单的特征提取（可以替换为更复杂的方法）
                        features = self._extract_simple_features(roi)
                        detection.features = features
            
            # 应用特征聚合
            # 这里可以根据需要实现特征聚合逻辑
            
            self.last_feature_frame = self.frame_count
            self.performance_stats['feature_time'] = time.time() - feature_start
            
        except Exception as e:
            logger.error(f"特征提取错误: {e}")
        
        return detections
    
    def _extract_simple_features(self, roi: np.ndarray) -> np.ndarray:
        """提取简单特征"""
        try:
            # 调整ROI大小
            roi_resized = cv2.resize(roi, (64, 64))
            
            # 计算颜色直方图
            hist_b = cv2.calcHist([roi_resized], [0], None, [32], [0, 256])
            hist_g = cv2.calcHist([roi_resized], [1], None, [32], [0, 256])
            hist_r = cv2.calcHist([roi_resized], [2], None, [32], [0, 256])
            
            # 归一化
            hist_b = hist_b.flatten() / (hist_b.sum() + 1e-7)
            hist_g = hist_g.flatten() / (hist_g.sum() + 1e-7)
            hist_r = hist_r.flatten() / (hist_r.sum() + 1e-7)
            
            # 合并特征
            features = np.concatenate([hist_b, hist_g, hist_r])
            
            return features
            
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            return np.zeros(96)  # 返回零向量
    
    def _apply_temporal_aggregation(self, detections: List[Detection]) -> List[Detection]:
        """应用时序聚合"""
        if not self.temporal_aggregator:
            return detections
        
        try:
            # 转换为时序聚合器需要的格式
            temporal_detections = []
            for det in detections:
                temporal_det = {
                    'bbox': det.bbox,
                    'confidence': det.confidence,
                    'class_id': det.class_id,
                    'class_name': det.class_name,
                    'frame_id': det.frame_id,
                    'track_id': det.track_id
                }
                temporal_detections.append(temporal_det)
            
            # 应用时序聚合
            aggregated = self.temporal_aggregator.aggregate_frame(temporal_detections)
            
            # 转换回Detection对象
            result = []
            for det_dict in aggregated:
                det = Detection(
                    bbox=det_dict['bbox'],
                    confidence=det_dict['confidence'],
                    class_id=det_dict['class_id'],
                    class_name=det_dict['class_name'],
                    frame_id=det_dict['frame_id'],
                    track_id=det_dict.get('track_id')
                )
                result.append(det)
            
            return result
            
        except Exception as e:
            logger.error(f"时序聚合错误: {e}")
            return detections
    
    def _convert_to_dict_format(self, detections: List[Detection]) -> List[Dict[str, Any]]:
        """转换为字典格式"""
        result = []
        
        for det in detections:
            det_dict = {
                'bbox': list(det.bbox),
                'confidence': det.confidence,
                'class_id': det.class_id,
                'class_name': det.class_name,
                'frame_id': det.frame_id,
                'timestamp': det.timestamp
            }
            
            if det.track_id is not None:
                det_dict['track_id'] = det.track_id
            
            if det.features is not None:
                det_dict['features'] = det.features
            
            result.append(det_dict)
        
        return result
    
    def _update_performance_stats(self, total_time: float):
        """更新性能统计"""
        self.performance_stats['total_time'] = total_time
        
        if total_time > 0:
            active_tracks = len(self.tracker.get_active_tracks()) if self.tracker else 0
            self.performance_stats['tracks_per_second'] = active_tracks / total_time
        
        # 更新内存使用（简化估算）
        if self.tracker:
            track_count = len(self.tracker.tracks)
            self.performance_stats['memory_usage_mb'] = track_count * 0.1  # 每个轨迹约0.1MB
    
    def get_active_tracks(self) -> List[Dict[str, Any]]:
        """获取活跃轨迹信息"""
        if not self.tracker:
            return []
        
        active_tracks = self.tracker.get_active_tracks()
        result = []
        
        for track in active_tracks:
            track_info = {
                'track_id': track.track_id,
                'class_id': track.class_id,
                'class_name': track.class_name,
                'total_frames': track.total_frames,
                'missing_frames': track.missing_frames,
                'created_time': track.created_time,
                'last_update_time': track.last_update_time,
                'velocity': track.velocity
            }
            
            if track.current_bbox:
                track_info['current_bbox'] = list(track.current_bbox)
            
            if track.current_center:
                track_info['current_center'] = list(track.current_center)
            
            result.append(track_info)
        
        return result
    
    def get_tracking_stats(self) -> Dict[str, Any]:
        """获取跟踪统计信息"""
        stats = {
            'mode': self.config.mode.value,
            'frame_count': self.frame_count,
            'performance': self.performance_stats.copy()
        }
        
        if self.tracker:
            stats['tracker'] = self.tracker.get_stats()
        
        if self.feature_aggregator:
            stats['feature_aggregator'] = self.feature_aggregator.get_stats()
        
        if self.temporal_aggregator:
            stats['temporal_aggregator'] = self.temporal_aggregator.get_stats()
        
        return stats
    
    def reset(self):
        """重置跟踪系统"""
        if self.tracker:
            self.tracker.reset()
        
        if self.feature_aggregator:
            self.feature_aggregator.reset()
        
        if self.temporal_aggregator:
            self.temporal_aggregator.reset()
        
        self.frame_count = 0
        self.last_tracking_frame = 0
        self.last_feature_frame = 0
        
        self.performance_stats = {
            'tracking_time': 0.0,
            'feature_time': 0.0,
            'total_time': 0.0,
            'tracks_per_second': 0.0,
            'memory_usage_mb': 0.0
        }
        
        logger.info("跟踪集成系统已重置")
    
    def draw_tracks(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """在帧上绘制跟踪信息"""
        if not detections:
            return frame
        
        result_frame = frame.copy()
        
        # 定义颜色映射
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]
        
        for det in detections:
            if 'track_id' not in det:
                continue
            
            track_id = det['track_id']
            bbox = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # 选择颜色
            color = colors[track_id % len(colors)]
            
            # 绘制边界框
            x1, y1, x2, y2 = [int(x) for x in bbox]
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"ID:{track_id} {class_name} {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(result_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 绘制统计信息
        if self.tracker:
            stats = self.tracker.get_stats()
            info_text = f"Active Tracks: {stats['active_tracks']}, Total: {stats['total_tracks']}"
            cv2.putText(result_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result_frame

# 使用示例
if __name__ == "__main__":
    # 创建集成配置
    config = IntegratedTrackingConfig(
        mode=TrackingMode.ENHANCED,
        tracking_config=TrackingConfig(
            strategy=TrackingStrategy.HYBRID,
            max_missing_frames=10
        )
    )
    
    # 创建跟踪集成器
    tracking_integration = TrackingIntegration(config)
    
    # 模拟检测结果
    detections = [
        {
            'bbox': [100, 100, 200, 200],
            'confidence': 0.9,
            'class_id': 0,
            'class_name': 'person'
        },
        {
            'bbox': [300, 150, 400, 250],
            'confidence': 0.8,
            'class_id': 2,
            'class_name': 'car'
        }
    ]
    
    # 处理检测结果
    tracked_detections = tracking_integration.process_detections(detections)
    
    # 打印结果
    for det in tracked_detections:
        print(f"Track ID: {det.get('track_id', 'N/A')}, Class: {det['class_name']}, "
              f"Confidence: {det['confidence']:.2f}")
    
    # 获取跟踪统计
    stats = tracking_integration.get_tracking_stats()
    print(f"\n跟踪统计: {stats}")