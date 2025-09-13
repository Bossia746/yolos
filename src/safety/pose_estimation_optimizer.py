#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
复杂场景姿态估计优化模块

处理遮挡、多目标、复杂背景等情况下的姿态估计优化
支持多种姿态估计算法的融合和自适应选择

作者: AI Assistant
日期: 2024
"""

import cv2
import numpy as np
import time
import logging
import json
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import math


class PoseModel(Enum):
    """姿态估计模型类型"""
    OPENPOSE = "openpose"
    MEDIAPIPE = "mediapipe"
    ALPHAPOSE = "alphapose"
    HRNET = "hrnet"
    YOLOV8_POSE = "yolov8_pose"
    MOVENET = "movenet"


class OcclusionLevel(Enum):
    """遮挡程度"""
    NONE = "none"          # 无遮挡
    LIGHT = "light"        # 轻微遮挡 (<20%)
    MODERATE = "moderate"  # 中度遮挡 (20-50%)
    HEAVY = "heavy"        # 重度遮挡 (50-80%)
    SEVERE = "severe"      # 严重遮挡 (>80%)


class PoseQuality(Enum):
    """姿态质量等级"""
    EXCELLENT = "excellent"  # 优秀 (>0.9)
    GOOD = "good"           # 良好 (0.7-0.9)
    FAIR = "fair"           # 一般 (0.5-0.7)
    POOR = "poor"           # 较差 (0.3-0.5)
    VERY_POOR = "very_poor" # 很差 (<0.3)


class TrackingState(Enum):
    """跟踪状态"""
    ACTIVE = "active"        # 活跃跟踪
    LOST = "lost"           # 跟踪丢失
    OCCLUDED = "occluded"   # 被遮挡
    RECOVERED = "recovered" # 重新找到
    MERGED = "merged"       # 合并到其他目标


@dataclass
class KeyPoint:
    """关键点信息"""
    x: float
    y: float
    confidence: float
    visibility: float = 1.0  # 可见性 (0-1)
    is_occluded: bool = False
    
    def distance_to(self, other: 'KeyPoint') -> float:
        """计算到另一个关键点的距离"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


@dataclass
class PoseEstimation:
    """姿态估计结果"""
    person_id: int
    keypoints: List[KeyPoint]
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    model_used: PoseModel
    timestamp: float
    occlusion_level: OcclusionLevel = OcclusionLevel.NONE
    pose_quality: PoseQuality = PoseQuality.GOOD
    tracking_state: TrackingState = TrackingState.ACTIVE
    
    def get_center_point(self) -> Tuple[float, float]:
        """获取姿态中心点"""
        valid_points = [kp for kp in self.keypoints if kp.confidence > 0.3]
        if not valid_points:
            # 使用边界框中心
            return ((self.bbox[0] + self.bbox[2]) / 2, 
                   (self.bbox[1] + self.bbox[3]) / 2)
        
        center_x = sum(kp.x for kp in valid_points) / len(valid_points)
        center_y = sum(kp.y for kp in valid_points) / len(valid_points)
        return (center_x, center_y)
    
    def get_pose_area(self) -> float:
        """计算姿态占用面积"""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


@dataclass
class OptimizationConfig:
    """优化配置"""
    # 模型权重
    model_weights: Dict[PoseModel, float] = field(default_factory=lambda: {
        PoseModel.OPENPOSE: 0.3,
        PoseModel.MEDIAPIPE: 0.25,
        PoseModel.YOLOV8_POSE: 0.25,
        PoseModel.HRNET: 0.2
    })
    
    # 质量阈值
    quality_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'min_confidence': 0.3,
        'min_keypoints': 8,
        'max_occlusion': 0.7,
        'temporal_consistency': 0.8
    })
    
    # 融合参数
    fusion_params: Dict[str, Any] = field(default_factory=lambda: {
        'temporal_window': 5,
        'spatial_threshold': 50.0,
        'confidence_weight': 0.4,
        'consistency_weight': 0.3,
        'recency_weight': 0.3
    })
    
    # 跟踪参数
    tracking_params: Dict[str, Any] = field(default_factory=lambda: {
        'max_lost_frames': 10,
        'iou_threshold': 0.3,
        'feature_similarity_threshold': 0.7,
        'motion_prediction_frames': 3
    })


class OcclusionAnalyzer:
    """遮挡分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_occlusion(self, pose: PoseEstimation, 
                         frame: np.ndarray) -> Tuple[OcclusionLevel, Dict[str, Any]]:
        """分析姿态遮挡情况
        
        Args:
            pose: 姿态估计结果
            frame: 输入帧
            
        Returns:
            Tuple[OcclusionLevel, Dict]: 遮挡等级和详细信息
        """
        try:
            # 计算可见关键点比例
            visible_ratio = self._calculate_visible_ratio(pose)
            
            # 分析边界框完整性
            bbox_completeness = self._analyze_bbox_completeness(pose, frame)
            
            # 检测遮挡物
            occlusion_sources = self._detect_occlusion_sources(pose, frame)
            
            # 综合评估遮挡等级
            occlusion_level = self._determine_occlusion_level(
                visible_ratio, bbox_completeness, occlusion_sources
            )
            
            details = {
                'visible_ratio': visible_ratio,
                'bbox_completeness': bbox_completeness,
                'occlusion_sources': occlusion_sources,
                'confidence_drop': self._calculate_confidence_drop(pose)
            }
            
            return occlusion_level, details
            
        except Exception as e:
            self.logger.error(f"遮挡分析失败: {e}")
            return OcclusionLevel.NONE, {}
    
    def _calculate_visible_ratio(self, pose: PoseEstimation) -> float:
        """计算可见关键点比例"""
        if not pose.keypoints:
            return 0.0
        
        visible_count = sum(1 for kp in pose.keypoints 
                          if kp.confidence > 0.3 and kp.visibility > 0.5)
        return visible_count / len(pose.keypoints)
    
    def _analyze_bbox_completeness(self, pose: PoseEstimation, 
                                 frame: np.ndarray) -> float:
        """分析边界框完整性"""
        try:
            x1, y1, x2, y2 = pose.bbox
            h, w = frame.shape[:2]
            
            # 检查边界框是否被图像边界截断
            truncation_score = 1.0
            
            if x1 <= 5:  # 左边界截断
                truncation_score *= 0.8
            if y1 <= 5:  # 上边界截断
                truncation_score *= 0.8
            if x2 >= w - 5:  # 右边界截断
                truncation_score *= 0.8
            if y2 >= h - 5:  # 下边界截断
                truncation_score *= 0.8
            
            return truncation_score
            
        except Exception:
            return 1.0
    
    def _detect_occlusion_sources(self, pose: PoseEstimation, 
                                frame: np.ndarray) -> List[str]:
        """检测遮挡源"""
        sources = []
        
        try:
            x1, y1, x2, y2 = pose.bbox
            roi = frame[y1:y2, x1:x2]
            
            # 简单的遮挡源检测
            # 这里可以集成更复杂的物体检测算法
            
            # 检测边缘密度（可能表示遮挡物边界）
            edges = cv2.Canny(roi, 50, 150)
            edge_density = np.sum(edges > 0) / (roi.shape[0] * roi.shape[1])
            
            if edge_density > 0.15:
                sources.append("复杂背景")
            
            # 检测颜色一致性（遮挡物可能有不同颜色）
            if len(roi.shape) == 3:
                color_std = np.std(roi, axis=(0, 1))
                if np.mean(color_std) > 30:
                    sources.append("颜色变化")
            
            return sources
            
        except Exception:
            return []
    
    def _determine_occlusion_level(self, visible_ratio: float, 
                                 bbox_completeness: float, 
                                 occlusion_sources: List[str]) -> OcclusionLevel:
        """确定遮挡等级"""
        # 综合评分
        score = (visible_ratio * 0.6 + 
                bbox_completeness * 0.3 - 
                len(occlusion_sources) * 0.05)
        
        if score >= 0.8:
            return OcclusionLevel.NONE
        elif score >= 0.6:
            return OcclusionLevel.LIGHT
        elif score >= 0.4:
            return OcclusionLevel.MODERATE
        elif score >= 0.2:
            return OcclusionLevel.HEAVY
        else:
            return OcclusionLevel.SEVERE
    
    def _calculate_confidence_drop(self, pose: PoseEstimation) -> float:
        """计算置信度下降程度"""
        if not pose.keypoints:
            return 1.0
        
        avg_confidence = sum(kp.confidence for kp in pose.keypoints) / len(pose.keypoints)
        expected_confidence = 0.8  # 期望的平均置信度
        
        return max(0.0, (expected_confidence - avg_confidence) / expected_confidence)


class MultiTargetTracker:
    """多目标跟踪器"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.tracks: Dict[int, deque] = defaultdict(lambda: deque(maxlen=10))
        self.lost_tracks: Dict[int, int] = {}  # person_id -> lost_frames
        self.next_id = 1
    
    def update_tracks(self, poses: List[PoseEstimation]) -> List[PoseEstimation]:
        """更新跟踪
        
        Args:
            poses: 当前帧的姿态估计结果
            
        Returns:
            List[PoseEstimation]: 更新后的姿态列表
        """
        try:
            # 匹配现有轨迹
            matched_poses = self._match_poses_to_tracks(poses)
            
            # 处理丢失的轨迹
            self._handle_lost_tracks()
            
            # 创建新轨迹
            self._create_new_tracks(matched_poses)
            
            # 更新跟踪状态
            self._update_tracking_states(matched_poses)
            
            return matched_poses
            
        except Exception as e:
            self.logger.error(f"跟踪更新失败: {e}")
            return poses
    
    def _match_poses_to_tracks(self, poses: List[PoseEstimation]) -> List[PoseEstimation]:
        """将姿态匹配到现有轨迹"""
        matched_poses = []
        used_track_ids = set()
        
        for pose in poses:
            best_match_id = None
            best_score = 0.0
            
            # 寻找最佳匹配轨迹
            for track_id, track_history in self.tracks.items():
                if track_id in used_track_ids or not track_history:
                    continue
                
                last_pose = track_history[-1]
                score = self._calculate_matching_score(pose, last_pose)
                
                if score > best_score and score > self.config.tracking_params['feature_similarity_threshold']:
                    best_score = score
                    best_match_id = track_id
            
            # 分配ID
            if best_match_id is not None:
                pose.person_id = best_match_id
                used_track_ids.add(best_match_id)
                self.tracks[best_match_id].append(pose)
                # 从丢失列表中移除
                if best_match_id in self.lost_tracks:
                    del self.lost_tracks[best_match_id]
            else:
                # 新目标，暂时不分配ID
                pose.person_id = -1
            
            matched_poses.append(pose)
        
        return matched_poses
    
    def _calculate_matching_score(self, current_pose: PoseEstimation, 
                                last_pose: PoseEstimation) -> float:
        """计算姿态匹配分数"""
        try:
            # 位置相似性
            current_center = current_pose.get_center_point()
            last_center = last_pose.get_center_point()
            
            distance = math.sqrt(
                (current_center[0] - last_center[0])**2 + 
                (current_center[1] - last_center[1])**2
            )
            
            # 距离分数 (距离越小分数越高)
            max_distance = self.config.fusion_params['spatial_threshold']
            distance_score = max(0.0, 1.0 - distance / max_distance)
            
            # IoU相似性
            iou_score = self._calculate_bbox_iou(
                current_pose.bbox, last_pose.bbox
            )
            
            # 姿态相似性
            pose_similarity = self._calculate_pose_similarity(
                current_pose.keypoints, last_pose.keypoints
            )
            
            # 综合分数
            total_score = (
                distance_score * 0.4 + 
                iou_score * 0.3 + 
                pose_similarity * 0.3
            )
            
            return total_score
            
        except Exception as e:
            self.logger.debug(f"匹配分数计算失败: {e}")
            return 0.0
    
    def _calculate_bbox_iou(self, bbox1: Tuple[int, int, int, int], 
                          bbox2: Tuple[int, int, int, int]) -> float:
        """计算边界框IoU"""
        try:
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
            
            # 计算并集
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_pose_similarity(self, keypoints1: List[KeyPoint], 
                                 keypoints2: List[KeyPoint]) -> float:
        """计算姿态相似性"""
        try:
            if len(keypoints1) != len(keypoints2):
                return 0.0
            
            similarities = []
            for kp1, kp2 in zip(keypoints1, keypoints2):
                if kp1.confidence > 0.3 and kp2.confidence > 0.3:
                    distance = kp1.distance_to(kp2)
                    # 归一化距离 (假设最大合理距离为100像素)
                    similarity = max(0.0, 1.0 - distance / 100.0)
                    similarities.append(similarity)
            
            return sum(similarities) / len(similarities) if similarities else 0.0
            
        except Exception:
            return 0.0
    
    def _handle_lost_tracks(self) -> None:
        """处理丢失的轨迹"""
        current_track_ids = set(self.tracks.keys())
        
        # 增加丢失帧数
        for track_id in list(self.lost_tracks.keys()):
            self.lost_tracks[track_id] += 1
            
            # 删除长时间丢失的轨迹
            if self.lost_tracks[track_id] > self.config.tracking_params['max_lost_frames']:
                if track_id in self.tracks:
                    del self.tracks[track_id]
                del self.lost_tracks[track_id]
        
        # 标记新丢失的轨迹
        for track_id in current_track_ids:
            if track_id not in self.lost_tracks:
                # 检查轨迹是否在当前帧中更新
                if self.tracks[track_id]:
                    last_update = self.tracks[track_id][-1].timestamp
                    if time.time() - last_update > 1.0:  # 1秒未更新
                        self.lost_tracks[track_id] = 1
    
    def _create_new_tracks(self, poses: List[PoseEstimation]) -> None:
        """为未匹配的姿态创建新轨迹"""
        for pose in poses:
            if pose.person_id == -1:  # 未匹配的姿态
                pose.person_id = self.next_id
                self.tracks[self.next_id].append(pose)
                self.next_id += 1
    
    def _update_tracking_states(self, poses: List[PoseEstimation]) -> None:
        """更新跟踪状态"""
        for pose in poses:
            track_history = self.tracks[pose.person_id]
            
            if len(track_history) == 1:
                pose.tracking_state = TrackingState.ACTIVE
            elif pose.person_id in self.lost_tracks:
                pose.tracking_state = TrackingState.RECOVERED
            elif pose.occlusion_level in [OcclusionLevel.HEAVY, OcclusionLevel.SEVERE]:
                pose.tracking_state = TrackingState.OCCLUDED
            else:
                pose.tracking_state = TrackingState.ACTIVE


class PoseFusionEngine:
    """姿态融合引擎"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.pose_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=config.fusion_params['temporal_window'])
        )
    
    def fuse_multi_model_poses(self, pose_results: Dict[PoseModel, List[PoseEstimation]]) -> List[PoseEstimation]:
        """融合多模型姿态估计结果
        
        Args:
            pose_results: 各模型的姿态估计结果
            
        Returns:
            List[PoseEstimation]: 融合后的姿态列表
        """
        try:
            # 收集所有姿态
            all_poses = []
            for model, poses in pose_results.items():
                for pose in poses:
                    pose.model_used = model
                    all_poses.append(pose)
            
            if not all_poses:
                return []
            
            # 聚类相似姿态
            pose_clusters = self._cluster_similar_poses(all_poses)
            
            # 融合每个聚类
            fused_poses = []
            for cluster in pose_clusters:
                fused_pose = self._fuse_pose_cluster(cluster)
                if fused_pose:
                    fused_poses.append(fused_pose)
            
            # 时序融合
            temporally_fused_poses = self._apply_temporal_fusion(fused_poses)
            
            return temporally_fused_poses
            
        except Exception as e:
            self.logger.error(f"姿态融合失败: {e}")
            return []
    
    def _cluster_similar_poses(self, poses: List[PoseEstimation]) -> List[List[PoseEstimation]]:
        """聚类相似的姿态"""
        clusters = []
        used_indices = set()
        
        for i, pose1 in enumerate(poses):
            if i in used_indices:
                continue
            
            cluster = [pose1]
            used_indices.add(i)
            
            for j, pose2 in enumerate(poses[i+1:], i+1):
                if j in used_indices:
                    continue
                
                # 计算相似性
                similarity = self._calculate_pose_cluster_similarity(pose1, pose2)
                
                if similarity > 0.7:  # 相似性阈值
                    cluster.append(pose2)
                    used_indices.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _calculate_pose_cluster_similarity(self, pose1: PoseEstimation, 
                                         pose2: PoseEstimation) -> float:
        """计算姿态聚类相似性"""
        try:
            # 位置相似性
            center1 = pose1.get_center_point()
            center2 = pose2.get_center_point()
            
            distance = math.sqrt(
                (center1[0] - center2[0])**2 + 
                (center1[1] - center2[1])**2
            )
            
            # 距离阈值
            max_distance = self.config.fusion_params['spatial_threshold']
            if distance > max_distance:
                return 0.0
            
            distance_score = 1.0 - distance / max_distance
            
            # 尺寸相似性
            area1 = pose1.get_pose_area()
            area2 = pose2.get_pose_area()
            
            size_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0.0
            
            return (distance_score * 0.7 + size_ratio * 0.3)
            
        except Exception:
            return 0.0
    
    def _fuse_pose_cluster(self, cluster: List[PoseEstimation]) -> Optional[PoseEstimation]:
        """融合姿态聚类"""
        try:
            if not cluster:
                return None
            
            if len(cluster) == 1:
                return cluster[0]
            
            # 选择最佳姿态作为基础
            base_pose = max(cluster, key=lambda p: p.confidence)
            
            # 加权融合关键点
            fused_keypoints = self._fuse_keypoints(cluster)
            
            # 融合边界框
            fused_bbox = self._fuse_bboxes(cluster)
            
            # 计算融合置信度
            fused_confidence = self._calculate_fused_confidence(cluster)
            
            # 创建融合结果
            fused_pose = PoseEstimation(
                person_id=base_pose.person_id,
                keypoints=fused_keypoints,
                bbox=fused_bbox,
                confidence=fused_confidence,
                model_used=base_pose.model_used,  # 使用最佳模型
                timestamp=time.time()
            )
            
            return fused_pose
            
        except Exception as e:
            self.logger.debug(f"姿态聚类融合失败: {e}")
            return cluster[0] if cluster else None
    
    def _fuse_keypoints(self, cluster: List[PoseEstimation]) -> List[KeyPoint]:
        """融合关键点"""
        if not cluster:
            return []
        
        # 假设所有姿态有相同数量的关键点
        num_keypoints = len(cluster[0].keypoints)
        fused_keypoints = []
        
        for i in range(num_keypoints):
            # 收集第i个关键点
            keypoints_i = []
            weights = []
            
            for pose in cluster:
                if i < len(pose.keypoints):
                    kp = pose.keypoints[i]
                    if kp.confidence > 0.1:  # 最低置信度阈值
                        keypoints_i.append(kp)
                        # 权重基于置信度和模型权重
                        model_weight = self.config.model_weights.get(pose.model_used, 0.25)
                        weight = kp.confidence * model_weight
                        weights.append(weight)
            
            if keypoints_i and weights:
                # 加权平均
                total_weight = sum(weights)
                fused_x = sum(kp.x * w for kp, w in zip(keypoints_i, weights)) / total_weight
                fused_y = sum(kp.y * w for kp, w in zip(keypoints_i, weights)) / total_weight
                fused_conf = sum(kp.confidence * w for kp, w in zip(keypoints_i, weights)) / total_weight
                fused_vis = sum(kp.visibility * w for kp, w in zip(keypoints_i, weights)) / total_weight
                
                fused_kp = KeyPoint(
                    x=fused_x,
                    y=fused_y,
                    confidence=fused_conf,
                    visibility=fused_vis
                )
                fused_keypoints.append(fused_kp)
            else:
                # 使用默认关键点
                fused_keypoints.append(KeyPoint(0, 0, 0.0, 0.0))
        
        return fused_keypoints
    
    def _fuse_bboxes(self, cluster: List[PoseEstimation]) -> Tuple[int, int, int, int]:
        """融合边界框"""
        if not cluster:
            return (0, 0, 0, 0)
        
        # 计算加权平均边界框
        weights = []
        bboxes = []
        
        for pose in cluster:
            model_weight = self.config.model_weights.get(pose.model_used, 0.25)
            weight = pose.confidence * model_weight
            weights.append(weight)
            bboxes.append(pose.bbox)
        
        total_weight = sum(weights)
        
        fused_x1 = sum(bbox[0] * w for bbox, w in zip(bboxes, weights)) / total_weight
        fused_y1 = sum(bbox[1] * w for bbox, w in zip(bboxes, weights)) / total_weight
        fused_x2 = sum(bbox[2] * w for bbox, w in zip(bboxes, weights)) / total_weight
        fused_y2 = sum(bbox[3] * w for bbox, w in zip(bboxes, weights)) / total_weight
        
        return (int(fused_x1), int(fused_y1), int(fused_x2), int(fused_y2))
    
    def _calculate_fused_confidence(self, cluster: List[PoseEstimation]) -> float:
        """计算融合置信度"""
        if not cluster:
            return 0.0
        
        # 加权平均置信度
        weights = []
        confidences = []
        
        for pose in cluster:
            model_weight = self.config.model_weights.get(pose.model_used, 0.25)
            weights.append(model_weight)
            confidences.append(pose.confidence)
        
        total_weight = sum(weights)
        fused_confidence = sum(conf * w for conf, w in zip(confidences, weights)) / total_weight
        
        # 多模型一致性奖励
        consistency_bonus = min(0.1, len(cluster) * 0.02)
        
        return min(1.0, fused_confidence + consistency_bonus)
    
    def _apply_temporal_fusion(self, poses: List[PoseEstimation]) -> List[PoseEstimation]:
        """应用时序融合"""
        fused_poses = []
        
        for pose in poses:
            # 更新历史
            self.pose_history[pose.person_id].append(pose)
            
            # 时序平滑
            smoothed_pose = self._temporal_smoothing(pose)
            fused_poses.append(smoothed_pose)
        
        return fused_poses
    
    def _temporal_smoothing(self, current_pose: PoseEstimation) -> PoseEstimation:
        """时序平滑"""
        history = self.pose_history[current_pose.person_id]
        
        if len(history) < 2:
            return current_pose
        
        # 简单的时序平滑
        # 这里可以实现更复杂的卡尔曼滤波等算法
        
        # 平滑关键点
        smoothed_keypoints = []
        for i, kp in enumerate(current_pose.keypoints):
            if kp.confidence > 0.3:
                # 使用历史数据平滑
                historical_kps = []
                for hist_pose in list(history)[-3:]:  # 最近3帧
                    if i < len(hist_pose.keypoints) and hist_pose.keypoints[i].confidence > 0.3:
                        historical_kps.append(hist_pose.keypoints[i])
                
                if historical_kps:
                    # 加权平均 (当前帧权重更高)
                    weights = [0.5] + [0.5 / len(historical_kps)] * len(historical_kps)
                    all_kps = [kp] + historical_kps
                    
                    smoothed_x = sum(k.x * w for k, w in zip(all_kps, weights))
                    smoothed_y = sum(k.y * w for k, w in zip(all_kps, weights))
                    
                    smoothed_kp = KeyPoint(
                        x=smoothed_x,
                        y=smoothed_y,
                        confidence=kp.confidence,
                        visibility=kp.visibility
                    )
                    smoothed_keypoints.append(smoothed_kp)
                else:
                    smoothed_keypoints.append(kp)
            else:
                smoothed_keypoints.append(kp)
        
        # 创建平滑后的姿态
        smoothed_pose = PoseEstimation(
            person_id=current_pose.person_id,
            keypoints=smoothed_keypoints,
            bbox=current_pose.bbox,
            confidence=current_pose.confidence,
            model_used=current_pose.model_used,
            timestamp=current_pose.timestamp,
            occlusion_level=current_pose.occlusion_level,
            pose_quality=current_pose.pose_quality,
            tracking_state=current_pose.tracking_state
        )
        
        return smoothed_pose


class PoseQualityAssessor:
    """姿态质量评估器"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def assess_pose_quality(self, pose: PoseEstimation) -> Tuple[PoseQuality, Dict[str, float]]:
        """评估姿态质量
        
        Args:
            pose: 姿态估计结果
            
        Returns:
            Tuple[PoseQuality, Dict]: 质量等级和详细评分
        """
        try:
            # 各项质量指标
            confidence_score = self._assess_confidence(pose)
            completeness_score = self._assess_completeness(pose)
            consistency_score = self._assess_consistency(pose)
            stability_score = self._assess_stability(pose)
            
            # 综合评分
            overall_score = (
                confidence_score * 0.3 +
                completeness_score * 0.25 +
                consistency_score * 0.25 +
                stability_score * 0.2
            )
            
            # 确定质量等级
            quality = self._determine_quality_level(overall_score)
            
            details = {
                'confidence_score': confidence_score,
                'completeness_score': completeness_score,
                'consistency_score': consistency_score,
                'stability_score': stability_score,
                'overall_score': overall_score
            }
            
            return quality, details
            
        except Exception as e:
            self.logger.error(f"姿态质量评估失败: {e}")
            return PoseQuality.POOR, {}
    
    def _assess_confidence(self, pose: PoseEstimation) -> float:
        """评估置信度"""
        if not pose.keypoints:
            return 0.0
        
        # 平均关键点置信度
        avg_confidence = sum(kp.confidence for kp in pose.keypoints) / len(pose.keypoints)
        
        # 高置信度关键点比例
        high_conf_ratio = sum(1 for kp in pose.keypoints if kp.confidence > 0.7) / len(pose.keypoints)
        
        # 综合置信度评分
        confidence_score = (avg_confidence * 0.6 + high_conf_ratio * 0.4)
        
        return confidence_score
    
    def _assess_completeness(self, pose: PoseEstimation) -> float:
        """评估完整性"""
        if not pose.keypoints:
            return 0.0
        
        # 可见关键点比例
        visible_ratio = sum(1 for kp in pose.keypoints 
                          if kp.confidence > self.config.quality_thresholds['min_confidence']) / len(pose.keypoints)
        
        # 关键身体部位检测
        critical_parts = [0, 1, 2, 5, 6, 11, 12]  # 头部、肩膀、髋部等关键点
        critical_detected = sum(1 for i in critical_parts 
                              if i < len(pose.keypoints) and pose.keypoints[i].confidence > 0.5)
        critical_ratio = critical_detected / len(critical_parts)
        
        completeness_score = (visible_ratio * 0.4 + critical_ratio * 0.6)
        
        return completeness_score
    
    def _assess_consistency(self, pose: PoseEstimation) -> float:
        """评估一致性"""
        if not pose.keypoints or len(pose.keypoints) < 5:
            return 0.0
        
        try:
            # 身体比例一致性检查
            consistency_score = 1.0
            
            # 检查头肩比例
            if (len(pose.keypoints) > 6 and 
                pose.keypoints[0].confidence > 0.5 and  # 鼻子
                pose.keypoints[5].confidence > 0.5 and  # 左肩
                pose.keypoints[6].confidence > 0.5):    # 右肩
                
                head_pos = pose.keypoints[0]
                left_shoulder = pose.keypoints[5]
                right_shoulder = pose.keypoints[6]
                
                shoulder_width = abs(left_shoulder.x - right_shoulder.x)
                head_shoulder_dist = abs(head_pos.y - (left_shoulder.y + right_shoulder.y) / 2)
                
                # 合理的头肩比例应该在0.3-1.5之间
                if shoulder_width > 0:
                    ratio = head_shoulder_dist / shoulder_width
                    if ratio < 0.3 or ratio > 1.5:
                        consistency_score *= 0.8
            
            # 检查肢体长度一致性
            # 这里可以添加更多的身体比例检查
            
            return consistency_score
            
        except Exception:
            return 0.5
    
    def _assess_stability(self, pose: PoseEstimation) -> float:
        """评估稳定性（需要历史数据）"""
        # 这里需要访问历史姿态数据来评估稳定性
        # 简化实现，返回基于当前姿态的稳定性估计
        
        if not pose.keypoints:
            return 0.0
        
        # 基于关键点分布的稳定性评估
        valid_keypoints = [kp for kp in pose.keypoints if kp.confidence > 0.3]
        
        if len(valid_keypoints) < 3:
            return 0.3
        
        # 计算关键点的分布方差
        x_coords = [kp.x for kp in valid_keypoints]
        y_coords = [kp.y for kp in valid_keypoints]
        
        x_var = np.var(x_coords) if len(x_coords) > 1 else 0
        y_var = np.var(y_coords) if len(y_coords) > 1 else 0
        
        # 合理的方差范围表示稳定的姿态
        # 这里使用简化的评估方法
        stability_score = min(1.0, 1.0 / (1.0 + (x_var + y_var) / 10000.0))
        
        return stability_score
    
    def _determine_quality_level(self, score: float) -> PoseQuality:
        """确定质量等级"""
        if score >= 0.9:
            return PoseQuality.EXCELLENT
        elif score >= 0.7:
            return PoseQuality.GOOD
        elif score >= 0.5:
            return PoseQuality.FAIR
        elif score >= 0.3:
            return PoseQuality.POOR
        else:
            return PoseQuality.VERY_POOR


class PoseEstimationOptimizer:
    """姿态估计优化器主类"""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.occlusion_analyzer = OcclusionAnalyzer()
        self.multi_target_tracker = MultiTargetTracker(self.config)
        self.fusion_engine = PoseFusionEngine(self.config)
        self.quality_assessor = PoseQualityAssessor(self.config)
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 统计信息
        self.stats = {
            'total_frames': 0,
            'total_poses': 0,
            'quality_distribution': defaultdict(int),
            'occlusion_distribution': defaultdict(int),
            'model_usage': defaultdict(int)
        }
    
    def optimize_poses(self, frame: np.ndarray, 
                      pose_results: Dict[PoseModel, List[PoseEstimation]]) -> List[PoseEstimation]:
        """优化姿态估计结果
        
        Args:
            frame: 输入帧
            pose_results: 各模型的姿态估计结果
            
        Returns:
            List[PoseEstimation]: 优化后的姿态列表
        """
        try:
            self.stats['total_frames'] += 1
            
            # 1. 融合多模型结果
            fused_poses = self.fusion_engine.fuse_multi_model_poses(pose_results)
            
            if not fused_poses:
                return []
            
            # 2. 分析遮挡情况
            for pose in fused_poses:
                occlusion_level, occlusion_details = self.occlusion_analyzer.analyze_occlusion(pose, frame)
                pose.occlusion_level = occlusion_level
                self.stats['occlusion_distribution'][occlusion_level.value] += 1
            
            # 3. 评估姿态质量
            for pose in fused_poses:
                quality, quality_details = self.quality_assessor.assess_pose_quality(pose)
                pose.pose_quality = quality
                self.stats['quality_distribution'][quality.value] += 1
            
            # 4. 多目标跟踪
            tracked_poses = self.multi_target_tracker.update_tracks(fused_poses)
            
            # 5. 过滤低质量姿态
            filtered_poses = self._filter_low_quality_poses(tracked_poses)
            
            # 更新统计
            self.stats['total_poses'] += len(filtered_poses)
            for pose in filtered_poses:
                self.stats['model_usage'][pose.model_used.value] += 1
            
            return filtered_poses
            
        except Exception as e:
            self.logger.error(f"姿态优化失败: {e}")
            return []
    
    def _filter_low_quality_poses(self, poses: List[PoseEstimation]) -> List[PoseEstimation]:
        """过滤低质量姿态"""
        filtered_poses = []
        
        for pose in poses:
            # 基本质量检查
            if (pose.confidence >= self.config.quality_thresholds['min_confidence'] and
                pose.pose_quality not in [PoseQuality.VERY_POOR] and
                pose.occlusion_level != OcclusionLevel.SEVERE):
                
                # 关键点数量检查
                valid_keypoints = sum(1 for kp in pose.keypoints 
                                    if kp.confidence > self.config.quality_thresholds['min_confidence'])
                
                if valid_keypoints >= self.config.quality_thresholds['min_keypoints']:
                    filtered_poses.append(pose)
        
        return filtered_poses
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """获取优化统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            'total_frames': self.stats['total_frames'],
            'total_poses': self.stats['total_poses'],
            'average_poses_per_frame': (self.stats['total_poses'] / self.stats['total_frames'] 
                                      if self.stats['total_frames'] > 0 else 0),
            'quality_distribution': dict(self.stats['quality_distribution']),
            'occlusion_distribution': dict(self.stats['occlusion_distribution']),
            'model_usage': dict(self.stats['model_usage']),
            'configuration': {
                'model_weights': self.config.model_weights,
                'quality_thresholds': self.config.quality_thresholds,
                'fusion_params': self.config.fusion_params,
                'tracking_params': self.config.tracking_params
            }
        }
    
    def export_optimization_report(self, output_path: str) -> bool:
        """导出优化报告
        
        Args:
            output_path: 输出路径
            
        Returns:
            bool: 是否成功
        """
        try:
            report = {
                'timestamp': time.time(),
                'statistics': self.get_optimization_statistics(),
                'performance_metrics': {
                    'frames_processed': self.stats['total_frames'],
                    'poses_detected': self.stats['total_poses'],
                    'quality_score': self._calculate_overall_quality_score(),
                    'tracking_efficiency': self._calculate_tracking_efficiency()
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"优化报告已导出到: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"报告导出失败: {e}")
            return False
    
    def _calculate_overall_quality_score(self) -> float:
        """计算整体质量评分"""
        quality_weights = {
            PoseQuality.EXCELLENT.value: 1.0,
            PoseQuality.GOOD.value: 0.8,
            PoseQuality.FAIR.value: 0.6,
            PoseQuality.POOR.value: 0.4,
            PoseQuality.VERY_POOR.value: 0.2
        }
        
        total_score = 0.0
        total_count = 0
        
        for quality, count in self.stats['quality_distribution'].items():
            weight = quality_weights.get(quality, 0.5)
            total_score += weight * count
            total_count += count
        
        return total_score / total_count if total_count > 0 else 0.0
    
    def _calculate_tracking_efficiency(self) -> float:
        """计算跟踪效率"""
        # 简化的跟踪效率计算
        # 实际实现中可以基于跟踪连续性、ID切换次数等指标
        
        if self.stats['total_frames'] == 0:
            return 0.0
        
        # 基于姿态检测率的简单效率计算
        detection_rate = self.stats['total_poses'] / self.stats['total_frames']
        
        # 归一化到0-1范围
        efficiency = min(1.0, detection_rate / 2.0)  # 假设平均每帧2个人
        
        return efficiency
    
    def cleanup(self) -> None:
        """清理资源"""
        try:
            self.executor.shutdown(wait=True)
            self.logger.info("姿态估计优化器资源清理完成")
            
        except Exception as e:
            self.logger.error(f"资源清理失败: {e}")


# 测试代码
if __name__ == "__main__":
    import numpy as np
    
    def test_pose_estimation_optimizer():
        """测试姿态估计优化器"""
        print("=== 复杂场景姿态估计优化器测试 ===")
        
        # 创建优化器
        optimizer = PoseEstimationOptimizer()
        
        # 模拟输入数据
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 模拟多模型姿态估计结果
        pose_results = {
            PoseModel.OPENPOSE: [
                PoseEstimation(
                    person_id=1,
                    keypoints=[
                        KeyPoint(150, 100, 0.9, 1.0),
                        KeyPoint(140, 120, 0.8, 1.0),
                        KeyPoint(160, 120, 0.85, 1.0),
                        KeyPoint(130, 150, 0.7, 0.8),
                        KeyPoint(170, 150, 0.75, 0.9),
                        KeyPoint(120, 200, 0.6, 0.7),
                        KeyPoint(180, 200, 0.65, 0.8)
                    ],
                    bbox=(100, 80, 200, 250),
                    confidence=0.92,
                    model_used=PoseModel.YOLOV8_POSE,
                    timestamp=time.time()
                ),
                # 人员2 - 中等质量
                PoseEstimation(
                    person_id=2,
                    keypoints=[KeyPoint(350+i*8, 120+i*12, 0.7-i*0.03, 0.8) for i in range(7)],
                    bbox=(300, 100, 400, 270),
                    confidence=0.75,
                    model_used=PoseModel.YOLOV8_POSE,
                    timestamp=time.time()
                ),
                # 人员3 - 低质量（部分遮挡）
                PoseEstimation(
                    person_id=3,
                    keypoints=[KeyPoint(550+i*5, 150+i*10, 0.4-i*0.02, 0.5) for i in range(7)],
                    bbox=(500, 130, 600, 300),
                    confidence=0.58,
                    model_used=PoseModel.YOLOV8_POSE,
                    timestamp=time.time()
                )
            ]
        }
        
        multi_poses = optimizer.optimize_poses(frame, multi_person_results)
        print(f"多人场景检测到 {len(multi_poses)} 个姿态")
        for pose in multi_poses:
            print(f"  人员 {pose.person_id}: 质量={pose.pose_quality.value}, 置信度={pose.confidence:.2f}")
        
        # 场景3: 模型融合测试
        print("\n场景3: 多模型融合场景")
        fusion_results = {
            PoseModel.OPENPOSE: [
                PoseEstimation(
                    person_id=1,
                    keypoints=[KeyPoint(150, 100+i*20, 0.85, 1.0) for i in range(7)],
                    bbox=(100, 80, 200, 250),
                    confidence=0.85,
                    model_used=PoseModel.OPENPOSE,
                    timestamp=time.time()
                )
            ],
            PoseModel.MEDIAPIPE: [
                PoseEstimation(
                    person_id=1,
                    keypoints=[KeyPoint(152, 102+i*20, 0.82, 1.0) for i in range(7)],
                    bbox=(98, 78, 202, 252),
                    confidence=0.82,
                    model_used=PoseModel.MEDIAPIPE,
                    timestamp=time.time()
                )
            ],
            PoseModel.HRNET: [
                PoseEstimation(
                    person_id=1,
                    keypoints=[KeyPoint(148, 98+i*20, 0.88, 1.0) for i in range(7)],
                    bbox=(102, 82, 198, 248),
                    confidence=0.88,
                    model_used=PoseModel.HRNET,
                    timestamp=time.time()
                )
            ]
        }
        
        fused_poses = optimizer.optimize_poses(frame, fusion_results)
        print(f"融合场景检测到 {len(fused_poses)} 个姿态")
        for pose in fused_poses:
            print(f"  融合结果: 模型={pose.model_used.value}, 置信度={pose.confidence:.3f}")
        
        # 获取最终统计
        final_stats = optimizer.get_optimization_statistics()
        print(f"\n最终统计:")
        print(f"  总处理帧数: {final_stats['total_frames']}")
        print(f"  总检测姿态数: {final_stats['total_poses']}")
        print(f"  质量分布: {final_stats['quality_distribution']}")
        print(f"  遮挡分布: {final_stats['occlusion_distribution']}")
        
        optimizer.cleanup()
        print("\n复杂场景测试完成")
    
    # 运行测试
    test_pose_estimation_optimizer()
    test_complex_scenarios()
    
    def test_complex_scenarios():
        """测试复杂场景"""
        print("\n=== 复杂场景测试 ===")
        
        optimizer = PoseEstimationOptimizer()
        
        # 场景1: 重度遮挡
        print("\n场景1: 重度遮挡场景")
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        occluded_pose_results = {
            PoseModel.OPENPOSE: [
                PoseEstimation(
                    person_id=1,
                    keypoints=[
                        KeyPoint(150, 100, 0.9, 1.0),
                        KeyPoint(0, 0, 0.1, 0.0),  # 遮挡的关键点
                        KeyPoint(0, 0, 0.1, 0.0),
                        KeyPoint(130, 150, 0.3, 0.2),  # 低置信度
                        KeyPoint(0, 0, 0.1, 0.0),
                        KeyPoint(0, 0, 0.1, 0.0),
                        KeyPoint(180, 200, 0.4, 0.3)
                    ],
                    bbox=(100, 80, 200, 250),
                    confidence=0.45,  # 较低置信度
                    model_used=PoseModel.OPENPOSE,
                    timestamp=time.time()
                )
            ]
        }
        
        occluded_poses = optimizer.optimize_poses(frame, occluded_pose_results)
        print(f"遮挡场景检测到 {len(occluded_poses)} 个姿态")
        for pose in occluded_poses:
            print(f"  遮挡程度: {pose.occlusion_level.value}, 质量: {pose.pose_quality.value}")
        
        # 场景2: 多人场景
        print("\n场景2: 多人混合场景")
        multi_person_results = {
            PoseModel.YOLOV8_POSE: [
                # 人员1 - 高质量
                PoseEstimation(
                    person_id=1,
                    keypoints=[KeyPoint(150+i*10, 100+i*15, 0.9-i*0.05, 1.0) for i in range(7)],
                    bbox=(100, 80, 200, 250),
                    confidence=0.9,
                    model_used=PoseModel.YOLOV8_POSE,
                    timestamp=time.time()
                )
            ]
        }
        
        print("复杂场景测试完成")

if __name__ == "__main__":
    test_pose_estimation_optimizer()
    test_complex_scenarios()