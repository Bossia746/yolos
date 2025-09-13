#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多目标跟踪系统
基于YOLOV论文思想，集成特征聚合和时序信息的高效多目标跟踪器
支持多平台部署，适配边缘设备性能限制
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

class TrackingStrategy(Enum):
    """跟踪策略"""
    IOU_BASED = "iou_based"  # 基于IoU的跟踪
    CENTROID_BASED = "centroid_based"  # 基于质心的跟踪
    FEATURE_BASED = "feature_based"  # 基于特征的跟踪
    HYBRID = "hybrid"  # 混合策略

class TrackState(Enum):
    """轨迹状态"""
    ACTIVE = "active"  # 活跃
    LOST = "lost"  # 丢失
    TERMINATED = "terminated"  # 终止

@dataclass
class TrackingConfig:
    """跟踪配置"""
    strategy: TrackingStrategy = TrackingStrategy.HYBRID
    max_missing_frames: int = 10  # 最大丢失帧数
    min_track_length: int = 3  # 最小轨迹长度
    iou_threshold: float = 0.3  # IoU阈值
    distance_threshold: float = 100.0  # 距离阈值
    feature_similarity_threshold: float = 0.7  # 特征相似度阈值
    max_tracks: int = 100  # 最大轨迹数
    track_buffer_size: int = 30  # 轨迹缓冲区大小
    enable_prediction: bool = True  # 启用位置预测
    enable_feature_matching: bool = True  # 启用特征匹配
    platform_optimization: bool = True  # 平台优化

@dataclass
class Detection:
    """检测结果"""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    frame_id: int
    timestamp: float = field(default_factory=time.time)
    features: Optional[np.ndarray] = None  # 特征向量
    track_id: Optional[int] = None

    @property
    def center(self) -> Tuple[float, float]:
        """获取中心点"""
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)
    
    @property
    def area(self) -> float:
        """获取面积"""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])

@dataclass
class Track:
    """轨迹信息"""
    track_id: int
    class_id: int
    class_name: str
    state: TrackState = TrackState.ACTIVE
    detections: deque = field(default_factory=lambda: deque(maxlen=30))
    missing_frames: int = 0
    total_frames: int = 0
    created_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    predicted_bbox: Optional[Tuple[float, float, float, float]] = None
    velocity: Tuple[float, float] = (0.0, 0.0)
    
    def add_detection(self, detection: Detection):
        """添加检测结果"""
        self.detections.append(detection)
        self.missing_frames = 0
        self.total_frames += 1
        self.last_update_time = time.time()
        detection.track_id = self.track_id
        
        # 更新速度
        if len(self.detections) >= 2:
            prev_center = self.detections[-2].center
            curr_center = detection.center
            dt = detection.timestamp - self.detections[-2].timestamp
            if dt > 0:
                self.velocity = (
                    (curr_center[0] - prev_center[0]) / dt,
                    (curr_center[1] - prev_center[1]) / dt
                )
    
    def predict_next_position(self, dt: float = 1.0) -> Tuple[float, float, float, float]:
        """预测下一帧位置"""
        if not self.detections:
            return None
        
        last_detection = self.detections[-1]
        center_x, center_y = last_detection.center
        
        # 基于速度预测
        pred_x = center_x + self.velocity[0] * dt
        pred_y = center_y + self.velocity[1] * dt
        
        # 保持边界框大小
        width = last_detection.bbox[2] - last_detection.bbox[0]
        height = last_detection.bbox[3] - last_detection.bbox[1]
        
        return (
            pred_x - width / 2,
            pred_y - height / 2,
            pred_x + width / 2,
            pred_y + height / 2
        )
    
    @property
    def current_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        """当前边界框"""
        return self.detections[-1].bbox if self.detections else None
    
    @property
    def current_center(self) -> Optional[Tuple[float, float]]:
        """当前中心点"""
        return self.detections[-1].center if self.detections else None

class MultiObjectTracker:
    """多目标跟踪器"""
    
    def __init__(self, config: Optional[TrackingConfig] = None):
        self.config = config or TrackingConfig()
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.frame_count = 0
        self.lock = threading.Lock()
        
        # 统计信息
        self.stats = {
            'total_tracks': 0,
            'active_tracks': 0,
            'lost_tracks': 0,
            'terminated_tracks': 0,
            'avg_track_length': 0.0,
            'processing_time': 0.0
        }
        
        logger.info(f"多目标跟踪器初始化完成，策略: {self.config.strategy.value}")
    
    def update(self, detections: List[Detection]) -> List[Detection]:
        """更新跟踪状态"""
        start_time = time.time()
        
        with self.lock:
            self.frame_count += 1
            
            # 为检测结果设置帧ID
            for detection in detections:
                detection.frame_id = self.frame_count
            
            # 执行跟踪匹配
            matched_detections = self._match_detections_to_tracks(detections)
            
            # 更新轨迹状态
            self._update_track_states()
            
            # 清理终止的轨迹
            self._cleanup_tracks()
            
            # 更新统计信息
            self._update_stats()
        
        self.stats['processing_time'] = time.time() - start_time
        return matched_detections
    
    def _match_detections_to_tracks(self, detections: List[Detection]) -> List[Detection]:
        """将检测结果匹配到轨迹"""
        if not detections:
            return []
        
        # 获取活跃轨迹
        active_tracks = {tid: track for tid, track in self.tracks.items() 
                        if track.state == TrackState.ACTIVE}
        
        if not active_tracks:
            # 没有活跃轨迹，创建新轨迹
            return self._create_new_tracks(detections)
        
        # 计算匹配矩阵
        cost_matrix = self._compute_cost_matrix(detections, list(active_tracks.values()))
        
        # 执行匹配
        matches, unmatched_detections, unmatched_tracks = self._solve_assignment(
            cost_matrix, detections, list(active_tracks.values())
        )
        
        matched_detections = []
        
        # 处理匹配的检测结果
        for det_idx, track_idx in matches:
            detection = detections[det_idx]
            track = list(active_tracks.values())[track_idx]
            track.add_detection(detection)
            matched_detections.append(detection)
        
        # 处理未匹配的检测结果（创建新轨迹）
        unmatched_dets = [detections[i] for i in unmatched_detections]
        new_tracks = self._create_new_tracks(unmatched_dets)
        matched_detections.extend(new_tracks)
        
        # 处理未匹配的轨迹（增加丢失帧数）
        for track_idx in unmatched_tracks:
            track = list(active_tracks.values())[track_idx]
            track.missing_frames += 1
        
        return matched_detections
    
    def _compute_cost_matrix(self, detections: List[Detection], tracks: List[Track]) -> np.ndarray:
        """计算代价矩阵"""
        if not detections or not tracks:
            return np.array([])
        
        cost_matrix = np.full((len(detections), len(tracks)), float('inf'))
        
        for i, detection in enumerate(detections):
            for j, track in enumerate(tracks):
                # 只匹配相同类别的目标
                if detection.class_id != track.class_id:
                    continue
                
                cost = self._compute_matching_cost(detection, track)
                if cost < float('inf'):
                    cost_matrix[i, j] = cost
        
        return cost_matrix
    
    def _compute_matching_cost(self, detection: Detection, track: Track) -> float:
        """计算匹配代价"""
        if not track.detections:
            return float('inf')
        
        last_detection = track.detections[-1]
        cost = 0.0
        
        # IoU代价
        if self.config.strategy in [TrackingStrategy.IOU_BASED, TrackingStrategy.HYBRID]:
            iou = self._calculate_iou(detection.bbox, last_detection.bbox)
            if iou < self.config.iou_threshold:
                return float('inf')
            cost += (1.0 - iou) * 0.4
        
        # 距离代价
        if self.config.strategy in [TrackingStrategy.CENTROID_BASED, TrackingStrategy.HYBRID]:
            distance = self._calculate_distance(detection.center, last_detection.center)
            if distance > self.config.distance_threshold:
                return float('inf')
            cost += (distance / self.config.distance_threshold) * 0.3
        
        # 特征代价
        if (self.config.strategy in [TrackingStrategy.FEATURE_BASED, TrackingStrategy.HYBRID] and
            self.config.enable_feature_matching and
            detection.features is not None and last_detection.features is not None):
            similarity = self._calculate_feature_similarity(detection.features, last_detection.features)
            if similarity < self.config.feature_similarity_threshold:
                return float('inf')
            cost += (1.0 - similarity) * 0.3
        
        return cost
    
    def _solve_assignment(self, cost_matrix: np.ndarray, detections: List[Detection], 
                         tracks: List[Track]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """求解分配问题（简化版匈牙利算法）"""
        if cost_matrix.size == 0:
            return [], list(range(len(detections))), list(range(len(tracks)))
        
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(tracks)))
        
        # 贪心匹配
        for _ in range(min(len(detections), len(tracks))):
            min_cost = float('inf')
            best_match = None
            
            for i in unmatched_detections:
                for j in unmatched_tracks:
                    if cost_matrix[i, j] < min_cost:
                        min_cost = cost_matrix[i, j]
                        best_match = (i, j)
            
            if best_match and min_cost < float('inf'):
                matches.append(best_match)
                unmatched_detections.remove(best_match[0])
                unmatched_tracks.remove(best_match[1])
            else:
                break
        
        return matches, unmatched_detections, unmatched_tracks
    
    def _create_new_tracks(self, detections: List[Detection]) -> List[Detection]:
        """创建新轨迹"""
        new_detections = []
        
        for detection in detections:
            if len(self.tracks) >= self.config.max_tracks:
                logger.warning(f"达到最大轨迹数限制: {self.config.max_tracks}")
                break
            
            track = Track(
                track_id=self.next_track_id,
                class_id=detection.class_id,
                class_name=detection.class_name
            )
            track.add_detection(detection)
            
            self.tracks[self.next_track_id] = track
            self.next_track_id += 1
            self.stats['total_tracks'] += 1
            
            new_detections.append(detection)
            logger.debug(f"创建新轨迹 {track.track_id} for {detection.class_name}")
        
        return new_detections
    
    def _update_track_states(self):
        """更新轨迹状态"""
        for track in self.tracks.values():
            if track.state == TrackState.ACTIVE:
                if track.missing_frames > self.config.max_missing_frames:
                    track.state = TrackState.LOST
                    logger.debug(f"轨迹 {track.track_id} 状态变为丢失")
            elif track.state == TrackState.LOST:
                # 丢失轨迹可以考虑终止
                if track.missing_frames > self.config.max_missing_frames * 2:
                    track.state = TrackState.TERMINATED
                    logger.debug(f"轨迹 {track.track_id} 状态变为终止")
    
    def _cleanup_tracks(self):
        """清理终止的轨迹"""
        terminated_tracks = []
        
        for track_id, track in self.tracks.items():
            if (track.state == TrackState.TERMINATED or 
                (track.total_frames < self.config.min_track_length and 
                 track.missing_frames > self.config.max_missing_frames)):
                terminated_tracks.append(track_id)
        
        for track_id in terminated_tracks:
            del self.tracks[track_id]
            logger.debug(f"清理轨迹 {track_id}")
    
    def _calculate_iou(self, bbox1: Tuple[float, float, float, float], 
                      bbox2: Tuple[float, float, float, float]) -> float:
        """计算IoU"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_distance(self, center1: Tuple[float, float], 
                           center2: Tuple[float, float]) -> float:
        """计算欧几里得距离"""
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _calculate_feature_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """计算特征相似度（余弦相似度）"""
        if features1 is None or features2 is None:
            return 0.0
        
        # 归一化
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(features1, features2) / (norm1 * norm2)
    
    def _update_stats(self):
        """更新统计信息"""
        active_tracks = sum(1 for track in self.tracks.values() if track.state == TrackState.ACTIVE)
        lost_tracks = sum(1 for track in self.tracks.values() if track.state == TrackState.LOST)
        
        self.stats.update({
            'active_tracks': active_tracks,
            'lost_tracks': lost_tracks,
            'terminated_tracks': self.stats['total_tracks'] - len(self.tracks)
        })
        
        # 计算平均轨迹长度
        if self.tracks:
            total_length = sum(track.total_frames for track in self.tracks.values())
            self.stats['avg_track_length'] = total_length / len(self.tracks)
    
    def get_active_tracks(self) -> List[Track]:
        """获取活跃轨迹"""
        return [track for track in self.tracks.values() if track.state == TrackState.ACTIVE]
    
    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """根据ID获取轨迹"""
        return self.tracks.get(track_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
    
    def reset(self):
        """重置跟踪器"""
        with self.lock:
            self.tracks.clear()
            self.next_track_id = 1
            self.frame_count = 0
            self.stats = {
                'total_tracks': 0,
                'active_tracks': 0,
                'lost_tracks': 0,
                'terminated_tracks': 0,
                'avg_track_length': 0.0,
                'processing_time': 0.0
            }
        logger.info("多目标跟踪器已重置")

# 使用示例
if __name__ == "__main__":
    # 创建跟踪配置
    config = TrackingConfig(
        strategy=TrackingStrategy.HYBRID,
        max_missing_frames=10,
        iou_threshold=0.3,
        distance_threshold=100.0
    )
    
    # 创建跟踪器
    tracker = MultiObjectTracker(config)
    
    # 模拟检测结果
    detections = [
        Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.9,
            class_id=0,
            class_name="person",
            frame_id=1
        ),
        Detection(
            bbox=(300, 150, 400, 250),
            confidence=0.8,
            class_id=2,
            class_name="car",
            frame_id=1
        )
    ]
    
    # 更新跟踪
    tracked_detections = tracker.update(detections)
    
    # 打印结果
    for detection in tracked_detections:
        print(f"Track ID: {detection.track_id}, Class: {detection.class_name}, "
              f"Confidence: {detection.confidence:.2f}")
    
    # 获取统计信息
    stats = tracker.get_stats()
    print(f"\n跟踪统计: {stats}")