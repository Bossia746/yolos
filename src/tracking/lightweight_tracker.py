#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量化跟踪算法
专门针对K230和ESP32等边缘设备的资源限制进行优化
实现高效的人体跟踪，降低计算复杂度和内存占用
"""

import time
import math
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# 轻量化数据结构
try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)

class TrackingState(Enum):
    """跟踪状态"""
    ACTIVE = "active"  # 活跃跟踪
    TENTATIVE = "tentative"  # 试探性跟踪
    LOST = "lost"  # 丢失
    DELETED = "deleted"  # 已删除

class FeatureType(Enum):
    """特征类型"""
    BBOX = "bbox"  # 边界框
    CENTER = "center"  # 中心点
    HISTOGRAM = "histogram"  # 颜色直方图
    KEYPOINTS = "keypoints"  # 关键点
    OPTICAL_FLOW = "optical_flow"  # 光流

@dataclass
class LightweightDetection:
    """轻量化检测结果"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int = 0  # 类别ID（0=人体）
    center: Tuple[float, float] = field(init=False)
    area: float = field(init=False)
    
    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) / 2, (y1 + y2) / 2)
        self.area = (x2 - x1) * (y2 - y1)

@dataclass
class CompactFeature:
    """紧凑特征表示"""
    feature_type: FeatureType
    data: np.ndarray  # 压缩的特征数据
    timestamp: float
    confidence: float = 1.0
    
    @classmethod
    def from_bbox(cls, bbox: Tuple[int, int, int, int], timestamp: float) -> 'CompactFeature':
        """从边界框创建特征"""
        # 将边界框转换为紧凑表示：[cx, cy, w, h]
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        data = np.array([cx, cy, w, h], dtype=np.float32)
        
        return cls(
            feature_type=FeatureType.BBOX,
            data=data,
            timestamp=timestamp
        )
    
    @classmethod
    def from_histogram(cls, image_patch: np.ndarray, timestamp: float, bins: int = 16) -> 'CompactFeature':
        """从图像块创建颜色直方图特征"""
        if cv2 is None or image_patch is None:
            # 如果没有cv2或图像，返回空特征
            return cls(
                feature_type=FeatureType.HISTOGRAM,
                data=np.zeros(bins * 3, dtype=np.float32),
                timestamp=timestamp,
                confidence=0.0
            )
        
        # 计算RGB直方图
        hist_r = cv2.calcHist([image_patch], [0], None, [bins], [0, 256])
        hist_g = cv2.calcHist([image_patch], [1], None, [bins], [0, 256])
        hist_b = cv2.calcHist([image_patch], [2], None, [bins], [0, 256])
        
        # 归一化并连接
        hist_r = hist_r.flatten() / (hist_r.sum() + 1e-6)
        hist_g = hist_g.flatten() / (hist_g.sum() + 1e-6)
        hist_b = hist_b.flatten() / (hist_b.sum() + 1e-6)
        
        data = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float32)
        
        return cls(
            feature_type=FeatureType.HISTOGRAM,
            data=data,
            timestamp=timestamp
        )
    
    def similarity(self, other: 'CompactFeature') -> float:
        """计算特征相似度
        
        Args:
            other: 另一个特征
            
        Returns:
            float: 相似度 [0, 1]
        """
        if self.feature_type != other.feature_type:
            return 0.0
        
        if self.feature_type == FeatureType.BBOX:
            return self._bbox_similarity(other)
        elif self.feature_type == FeatureType.HISTOGRAM:
            return self._histogram_similarity(other)
        else:
            return self._cosine_similarity(other)
    
    def _bbox_similarity(self, other: 'CompactFeature') -> float:
        """边界框相似度（基于IoU）"""
        # 从紧凑表示恢复边界框
        cx1, cy1, w1, h1 = self.data
        cx2, cy2, w2, h2 = other.data
        
        x1_1, y1_1 = cx1 - w1/2, cy1 - h1/2
        x2_1, y2_1 = cx1 + w1/2, cy1 + h1/2
        x1_2, y1_2 = cx2 - w2/2, cy2 - h2/2
        x2_2, y2_2 = cx2 + w2/2, cy2 + h2/2
        
        # 计算IoU
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        return inter_area / (union_area + 1e-6)
    
    def _histogram_similarity(self, other: 'CompactFeature') -> float:
        """直方图相似度（基于巴氏距离）"""
        # 巴氏系数
        bc = np.sum(np.sqrt(self.data * other.data))
        # 转换为相似度
        return bc
    
    def _cosine_similarity(self, other: 'CompactFeature') -> float:
        """余弦相似度"""
        dot_product = np.dot(self.data, other.data)
        norm1 = np.linalg.norm(self.data)
        norm2 = np.linalg.norm(other.data)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

class SimpleKalmanFilter:
    """简化的卡尔曼滤波器
    
    只跟踪位置和速度，减少计算复杂度
    """
    
    def __init__(self, initial_position: Tuple[float, float]):
        # 状态向量: [x, y, vx, vy]
        self.state = np.array([initial_position[0], initial_position[1], 0.0, 0.0], dtype=np.float32)
        
        # 状态转移矩阵
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # 观测矩阵
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # 协方差矩阵
        self.P = np.eye(4, dtype=np.float32) * 1000
        
        # 过程噪声
        self.Q = np.eye(4, dtype=np.float32) * 0.1
        
        # 观测噪声
        self.R = np.eye(2, dtype=np.float32) * 10
        
        self.last_update_time = time.time()
    
    def predict(self) -> Tuple[float, float]:
        """预测下一个状态
        
        Returns:
            Tuple[float, float]: 预测位置 (x, y)
        """
        current_time = time.time()
        dt = current_time - self.last_update_time
        
        # 更新状态转移矩阵的时间步长
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        
        # 预测
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return (self.state[0], self.state[1])
    
    def update(self, measurement: Tuple[float, float]):
        """更新状态
        
        Args:
            measurement: 观测值 (x, y)
        """
        z = np.array([measurement[0], measurement[1]], dtype=np.float32)
        
        # 计算卡尔曼增益
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 更新状态
        y = z - self.H @ self.state
        self.state = self.state + K @ y
        
        # 更新协方差
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
        
        self.last_update_time = time.time()
    
    def get_velocity(self) -> Tuple[float, float]:
        """获取速度
        
        Returns:
            Tuple[float, float]: 速度 (vx, vy)
        """
        return (self.state[2], self.state[3])

@dataclass
class LightweightTrack:
    """轻量化跟踪目标"""
    track_id: str
    state: TrackingState = TrackingState.TENTATIVE
    
    # 位置信息
    kalman_filter: Optional[SimpleKalmanFilter] = None
    last_detection: Optional[LightweightDetection] = None
    
    # 特征信息
    appearance_features: List[CompactFeature] = field(default_factory=list)
    max_features: int = 3  # 最多保存3个特征
    
    # 时间信息
    creation_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    
    # 跟踪统计
    hit_count: int = 0  # 命中次数
    miss_count: int = 0  # 丢失次数
    confidence: float = 0.0  # 跟踪置信度
    
    # 配置参数
    max_miss_count: int = 5  # 最大丢失次数
    min_hit_count: int = 3  # 最小命中次数（转为活跃状态）
    
    def predict(self) -> Tuple[float, float]:
        """预测下一个位置
        
        Returns:
            Tuple[float, float]: 预测位置
        """
        if self.kalman_filter is None:
            if self.last_detection:
                return self.last_detection.center
            return (0, 0)
        
        return self.kalman_filter.predict()
    
    def update(self, detection: LightweightDetection, image_patch: Optional[np.ndarray] = None):
        """更新跟踪目标
        
        Args:
            detection: 检测结果
            image_patch: 图像块（用于外观特征）
        """
        current_time = time.time()
        
        # 初始化卡尔曼滤波器
        if self.kalman_filter is None:
            self.kalman_filter = SimpleKalmanFilter(detection.center)
        else:
            self.kalman_filter.update(detection.center)
        
        # 更新检测信息
        self.last_detection = detection
        self.last_update_time = current_time
        
        # 更新统计信息
        self.hit_count += 1
        self.miss_count = 0
        self.confidence = min(1.0, self.confidence + 0.1)
        
        # 更新外观特征
        self._update_appearance_features(detection, image_patch, current_time)
        
        # 更新状态
        self._update_state()
    
    def mark_missed(self):
        """标记为丢失"""
        self.miss_count += 1
        self.confidence = max(0.0, self.confidence - 0.2)
        
        # 更新状态
        if self.miss_count >= self.max_miss_count:
            if self.state == TrackingState.TENTATIVE:
                self.state = TrackingState.DELETED
            else:
                self.state = TrackingState.LOST
    
    def _update_appearance_features(self, detection: LightweightDetection, image_patch: Optional[np.ndarray], timestamp: float):
        """更新外观特征
        
        Args:
            detection: 检测结果
            image_patch: 图像块
            timestamp: 时间戳
        """
        # 添加边界框特征
        bbox_feature = CompactFeature.from_bbox(detection.bbox, timestamp)
        self.appearance_features.append(bbox_feature)
        
        # 添加颜色直方图特征（如果有图像）
        if image_patch is not None:
            hist_feature = CompactFeature.from_histogram(image_patch, timestamp)
            self.appearance_features.append(hist_feature)
        
        # 限制特征数量
        if len(self.appearance_features) > self.max_features:
            # 保留最新的特征
            self.appearance_features = self.appearance_features[-self.max_features:]
    
    def _update_state(self):
        """更新跟踪状态"""
        if self.state == TrackingState.TENTATIVE:
            if self.hit_count >= self.min_hit_count:
                self.state = TrackingState.ACTIVE
        elif self.state == TrackingState.LOST:
            if self.miss_count == 0:  # 重新找到
                self.state = TrackingState.ACTIVE
    
    def compute_similarity(self, detection: LightweightDetection, image_patch: Optional[np.ndarray] = None) -> float:
        """计算与检测结果的相似度
        
        Args:
            detection: 检测结果
            image_patch: 图像块
            
        Returns:
            float: 相似度 [0, 1]
        """
        if not self.appearance_features:
            return 0.0
        
        similarities = []
        current_time = time.time()
        
        # 计算边界框相似度
        bbox_feature = CompactFeature.from_bbox(detection.bbox, current_time)
        for feature in self.appearance_features:
            if feature.feature_type == FeatureType.BBOX:
                sim = feature.similarity(bbox_feature)
                similarities.append(sim)
        
        # 计算颜色直方图相似度
        if image_patch is not None:
            hist_feature = CompactFeature.from_histogram(image_patch, current_time)
            for feature in self.appearance_features:
                if feature.feature_type == FeatureType.HISTOGRAM:
                    sim = feature.similarity(hist_feature)
                    similarities.append(sim * 0.7)  # 降低权重
        
        # 计算位置相似度
        if self.kalman_filter is not None:
            predicted_pos = self.predict()
            distance = math.sqrt(
                (predicted_pos[0] - detection.center[0]) ** 2 +
                (predicted_pos[1] - detection.center[1]) ** 2
            )
            # 将距离转换为相似度
            max_distance = 100  # 最大允许距离
            position_sim = max(0, 1 - distance / max_distance)
            similarities.append(position_sim)
        
        # 返回加权平均相似度
        if similarities:
            return np.mean(similarities)
        else:
            return 0.0
    
    def is_active(self) -> bool:
        """是否为活跃状态"""
        return self.state == TrackingState.ACTIVE
    
    def should_delete(self) -> bool:
        """是否应该删除"""
        return self.state == TrackingState.DELETED
    
    def get_age(self) -> float:
        """获取跟踪年龄（秒）"""
        return time.time() - self.creation_time
    
    def get_time_since_update(self) -> float:
        """获取距离上次更新的时间（秒）"""
        return time.time() - self.last_update_time

class HungarianAssignment:
    """简化的匈牙利算法实现
    
    用于解决检测与跟踪的关联问题
    """
    
    @staticmethod
    def solve(cost_matrix: np.ndarray, max_cost: float = 1.0) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """求解分配问题
        
        Args:
            cost_matrix: 代价矩阵 [tracks x detections]
            max_cost: 最大允许代价
            
        Returns:
            Tuple[List[Tuple[int, int]], List[int], List[int]]: (匹配对, 未匹配跟踪, 未匹配检测)
        """
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
        
        # 简化版本：贪心算法
        matches = []
        unmatched_tracks = list(range(cost_matrix.shape[0]))
        unmatched_detections = list(range(cost_matrix.shape[1]))
        
        # 找到所有有效的匹配（代价小于阈值）
        valid_matches = []
        for i in range(cost_matrix.shape[0]):
            for j in range(cost_matrix.shape[1]):
                if cost_matrix[i, j] < max_cost:
                    valid_matches.append((i, j, cost_matrix[i, j]))
        
        # 按代价排序
        valid_matches.sort(key=lambda x: x[2])
        
        # 贪心分配
        used_tracks = set()
        used_detections = set()
        
        for track_idx, det_idx, cost in valid_matches:
            if track_idx not in used_tracks and det_idx not in used_detections:
                matches.append((track_idx, det_idx))
                used_tracks.add(track_idx)
                used_detections.add(det_idx)
        
        # 更新未匹配列表
        unmatched_tracks = [i for i in range(cost_matrix.shape[0]) if i not in used_tracks]
        unmatched_detections = [j for j in range(cost_matrix.shape[1]) if j not in used_detections]
        
        return matches, unmatched_tracks, unmatched_detections

class LightweightTracker:
    """轻量化跟踪器主类
    
    专门为边缘设备优化的高效跟踪算法
    """
    
    def __init__(self, 
                 max_tracks: int = 10,
                 max_age: float = 30.0,
                 similarity_threshold: float = 0.3,
                 confidence_threshold: float = 0.5):
        """
        初始化轻量化跟踪器
        
        Args:
            max_tracks: 最大跟踪目标数
            max_age: 最大跟踪年龄（秒）
            similarity_threshold: 相似度阈值
            confidence_threshold: 置信度阈值
        """
        self.max_tracks = max_tracks
        self.max_age = max_age
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold
        
        # 跟踪目标列表
        self.tracks: List[LightweightTrack] = []
        
        # ID生成器
        self.next_id = 1
        
        # 统计信息
        self.stats = {
            "total_tracks_created": 0,
            "active_tracks": 0,
            "processing_time_ms": 0.0,
            "memory_usage_mb": 0.0
        }
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"轻量化跟踪器初始化完成，最大跟踪数: {max_tracks}")
    
    def update(self, detections: List[LightweightDetection], image: Optional[np.ndarray] = None) -> List[LightweightTrack]:
        """更新跟踪器
        
        Args:
            detections: 检测结果列表
            image: 当前帧图像
            
        Returns:
            List[LightweightTrack]: 活跃的跟踪目标
        """
        start_time = time.time()
        
        try:
            # 1. 预测所有跟踪目标的位置
            self._predict_tracks()
            
            # 2. 过滤高置信度的检测
            valid_detections = [det for det in detections if det.confidence >= self.confidence_threshold]
            
            # 3. 计算相似度矩阵
            similarity_matrix = self._compute_similarity_matrix(valid_detections, image)
            
            # 4. 数据关联
            matches, unmatched_tracks, unmatched_detections = self._associate_detections(
                similarity_matrix, valid_detections
            )
            
            # 5. 更新匹配的跟踪目标
            self._update_matched_tracks(matches, valid_detections, image)
            
            # 6. 处理未匹配的跟踪目标
            self._handle_unmatched_tracks(unmatched_tracks)
            
            # 7. 创建新的跟踪目标
            self._create_new_tracks(unmatched_detections, valid_detections, image)
            
            # 8. 清理过期的跟踪目标
            self._cleanup_tracks()
            
            # 9. 更新统计信息
            processing_time = (time.time() - start_time) * 1000
            self._update_stats(processing_time)
            
            # 返回活跃的跟踪目标
            active_tracks = [track for track in self.tracks if track.is_active()]
            return active_tracks
            
        except Exception as e:
            self.logger.error(f"跟踪更新失败: {e}")
            return []
    
    def _predict_tracks(self):
        """预测所有跟踪目标的位置"""
        for track in self.tracks:
            if not track.should_delete():
                track.predict()
    
    def _compute_similarity_matrix(self, detections: List[LightweightDetection], image: Optional[np.ndarray]) -> np.ndarray:
        """计算相似度矩阵
        
        Args:
            detections: 检测结果
            image: 图像
            
        Returns:
            np.ndarray: 相似度矩阵 [tracks x detections]
        """
        active_tracks = [track for track in self.tracks if not track.should_delete()]
        
        if not active_tracks or not detections:
            return np.array([])
        
        similarity_matrix = np.zeros((len(active_tracks), len(detections)), dtype=np.float32)
        
        for i, track in enumerate(active_tracks):
            for j, detection in enumerate(detections):
                # 提取图像块
                image_patch = None
                if image is not None:
                    image_patch = self._extract_image_patch(image, detection.bbox)
                
                # 计算相似度
                similarity = track.compute_similarity(detection, image_patch)
                similarity_matrix[i, j] = similarity
        
        return similarity_matrix
    
    def _extract_image_patch(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """提取图像块
        
        Args:
            image: 图像
            bbox: 边界框
            
        Returns:
            Optional[np.ndarray]: 图像块
        """
        try:
            x1, y1, x2, y2 = bbox
            h, w = image.shape[:2]
            
            # 边界检查
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            patch = image[y1:y2, x1:x2]
            
            # 调整大小以减少计算量
            if patch.shape[0] > 64 or patch.shape[1] > 64:
                if cv2 is not None:
                    patch = cv2.resize(patch, (64, 64))
            
            return patch
            
        except Exception as e:
            self.logger.warning(f"图像块提取失败: {e}")
            return None
    
    def _associate_detections(self, similarity_matrix: np.ndarray, detections: List[LightweightDetection]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """数据关联
        
        Args:
            similarity_matrix: 相似度矩阵
            detections: 检测结果
            
        Returns:
            Tuple: (匹配对, 未匹配跟踪, 未匹配检测)
        """
        if similarity_matrix.size == 0:
            active_tracks = [track for track in self.tracks if not track.should_delete()]
            return [], list(range(len(active_tracks))), list(range(len(detections)))
        
        # 将相似度转换为代价（1 - 相似度）
        cost_matrix = 1.0 - similarity_matrix
        
        # 使用匈牙利算法求解
        max_cost = 1.0 - self.similarity_threshold
        matches, unmatched_tracks, unmatched_detections = HungarianAssignment.solve(
            cost_matrix, max_cost
        )
        
        return matches, unmatched_tracks, unmatched_detections
    
    def _update_matched_tracks(self, matches: List[Tuple[int, int]], detections: List[LightweightDetection], image: Optional[np.ndarray]):
        """更新匹配的跟踪目标
        
        Args:
            matches: 匹配对
            detections: 检测结果
            image: 图像
        """
        active_tracks = [track for track in self.tracks if not track.should_delete()]
        
        for track_idx, det_idx in matches:
            track = active_tracks[track_idx]
            detection = detections[det_idx]
            
            # 提取图像块
            image_patch = None
            if image is not None:
                image_patch = self._extract_image_patch(image, detection.bbox)
            
            # 更新跟踪目标
            track.update(detection, image_patch)
    
    def _handle_unmatched_tracks(self, unmatched_tracks: List[int]):
        """处理未匹配的跟踪目标
        
        Args:
            unmatched_tracks: 未匹配的跟踪索引
        """
        active_tracks = [track for track in self.tracks if not track.should_delete()]
        
        for track_idx in unmatched_tracks:
            track = active_tracks[track_idx]
            track.mark_missed()
    
    def _create_new_tracks(self, unmatched_detections: List[int], detections: List[LightweightDetection], image: Optional[np.ndarray]):
        """创建新的跟踪目标
        
        Args:
            unmatched_detections: 未匹配的检测索引
            detections: 检测结果
            image: 图像
        """
        for det_idx in unmatched_detections:
            # 检查是否超过最大跟踪数
            if len(self.tracks) >= self.max_tracks:
                break
            
            detection = detections[det_idx]
            
            # 创建新跟踪目标
            track_id = f"track_{self.next_id}"
            self.next_id += 1
            
            new_track = LightweightTrack(track_id=track_id)
            
            # 提取图像块
            image_patch = None
            if image is not None:
                image_patch = self._extract_image_patch(image, detection.bbox)
            
            # 初始化跟踪目标
            new_track.update(detection, image_patch)
            
            self.tracks.append(new_track)
            self.stats["total_tracks_created"] += 1
            
            self.logger.debug(f"创建新跟踪目标: {track_id}")
    
    def _cleanup_tracks(self):
        """清理过期的跟踪目标"""
        before_count = len(self.tracks)
        
        # 移除应该删除的跟踪目标
        self.tracks = [track for track in self.tracks if not track.should_delete()]
        
        # 移除过期的跟踪目标
        current_time = time.time()
        self.tracks = [track for track in self.tracks if track.get_age() <= self.max_age]
        
        after_count = len(self.tracks)
        
        if before_count != after_count:
            self.logger.debug(f"清理跟踪目标: {before_count} -> {after_count}")
    
    def _update_stats(self, processing_time: float):
        """更新统计信息
        
        Args:
            processing_time: 处理时间（毫秒）
        """
        self.stats["processing_time_ms"] = processing_time
        self.stats["active_tracks"] = len([track for track in self.tracks if track.is_active()])
        
        # 简单的内存使用估计
        memory_per_track = 1.0  # KB
        self.stats["memory_usage_mb"] = len(self.tracks) * memory_per_track / 1024
    
    def get_active_tracks(self) -> List[LightweightTrack]:
        """获取活跃的跟踪目标
        
        Returns:
            List[LightweightTrack]: 活跃跟踪目标
        """
        return [track for track in self.tracks if track.is_active()]
    
    def get_primary_target(self) -> Optional[LightweightTrack]:
        """获取主要跟踪目标（置信度最高的）
        
        Returns:
            Optional[LightweightTrack]: 主要目标
        """
        active_tracks = self.get_active_tracks()
        
        if not active_tracks:
            return None
        
        # 按置信度排序
        active_tracks.sort(key=lambda x: x.confidence, reverse=True)
        return active_tracks[0]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return self.stats.copy()
    
    def reset(self):
        """重置跟踪器"""
        self.tracks.clear()
        self.next_id = 1
        self.stats = {
            "total_tracks_created": 0,
            "active_tracks": 0,
            "processing_time_ms": 0.0,
            "memory_usage_mb": 0.0
        }
        self.logger.info("跟踪器已重置")

# 测试代码
if __name__ == "__main__":
    # 创建轻量化跟踪器
    tracker = LightweightTracker(
        max_tracks=5,
        similarity_threshold=0.3,
        confidence_threshold=0.5
    )
    
    # 模拟检测结果
    detections = [
        LightweightDetection(bbox=(100, 100, 200, 300), confidence=0.8),
        LightweightDetection(bbox=(300, 150, 400, 350), confidence=0.7),
        LightweightDetection(bbox=(500, 200, 600, 400), confidence=0.6)
    ]
    
    # 模拟图像
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 更新跟踪器
    active_tracks = tracker.update(detections, image)
    
    print(f"活跃跟踪目标数: {len(active_tracks)}")
    for track in active_tracks:
        print(f"跟踪ID: {track.track_id}, 置信度: {track.confidence:.2f}, 状态: {track.state.value}")
    
    # 获取统计信息
    stats = tracker.get_stats()
    print(f"统计信息: {stats}")
    
    # 获取主要目标
    primary_target = tracker.get_primary_target()
    if primary_target:
        print(f"主要目标: {primary_target.track_id}")