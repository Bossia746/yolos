#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态融合跟踪系统
结合人脸、人体、姿态等多种特征实现鲁棒的人体跟踪
解决单一模态在侧脸、背身、遮挡等场景下的跟踪失效问题
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from collections import deque
import math

# 导入相关模块
try:
    from ..models.human_body_detector import HumanBodyDetection, HumanBodyDetector
    from ..recognition.optimized_face_recognizer import FaceDetectionResult
    from ..plugins.domain.human_recognition import HumanRecognitionPlugin
except ImportError:
    # 兼容性导入
    pass

logger = logging.getLogger(__name__)

class TrackingModality(Enum):
    """跟踪模态类型"""
    FACE = "face"
    BODY = "body"
    POSE = "pose"
    GESTURE = "gesture"
    MOTION = "motion"
    APPEARANCE = "appearance"

class TrackingState(Enum):
    """跟踪状态"""
    ACTIVE = "active"  # 活跃跟踪
    LOST = "lost"  # 暂时丢失
    OCCLUDED = "occluded"  # 被遮挡
    INACTIVE = "inactive"  # 非活跃

class FusionStrategy(Enum):
    """融合策略"""
    WEIGHTED_AVERAGE = "weighted_average"  # 加权平均
    CONFIDENCE_BASED = "confidence_based"  # 基于置信度
    ADAPTIVE = "adaptive"  # 自适应融合
    HIERARCHICAL = "hierarchical"  # 分层融合

@dataclass
class ModalityFeature:
    """模态特征"""
    modality: TrackingModality
    feature_vector: np.ndarray
    confidence: float
    bbox: Tuple[int, int, int, int]
    timestamp: float
    quality_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrackingTarget:
    """跟踪目标"""
    track_id: str
    state: TrackingState
    features: Dict[TrackingModality, ModalityFeature]
    position_history: deque
    velocity: Tuple[float, float]
    acceleration: Tuple[float, float]
    last_update_time: float
    missing_frames: int
    total_frames: int
    confidence_history: deque
    appearance_model: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if not hasattr(self, 'position_history') or self.position_history is None:
            self.position_history = deque(maxlen=30)
        if not hasattr(self, 'confidence_history') or self.confidence_history is None:
            self.confidence_history = deque(maxlen=10)

@dataclass
class FusionConfig:
    """融合配置"""
    # 模态权重
    modality_weights: Dict[TrackingModality, float] = field(default_factory=lambda: {
        TrackingModality.FACE: 0.3,
        TrackingModality.BODY: 0.4,
        TrackingModality.POSE: 0.2,
        TrackingModality.MOTION: 0.1
    })
    
    # 融合策略
    fusion_strategy: FusionStrategy = FusionStrategy.ADAPTIVE
    
    # 跟踪参数
    max_missing_frames: int = 15
    min_confidence_threshold: float = 0.3
    association_threshold: float = 0.7
    
    # 运动模型参数
    motion_noise_std: float = 10.0
    velocity_smoothing: float = 0.7
    
    # 外观模型参数
    appearance_update_rate: float = 0.1
    appearance_similarity_threshold: float = 0.6
    
    # 自适应参数
    adaptive_weight_learning_rate: float = 0.05
    quality_weight_factor: float = 0.3

class MotionPredictor:
    """运动预测器"""
    
    def __init__(self, noise_std: float = 10.0):
        self.noise_std = noise_std
    
    def predict_position(self, target: TrackingTarget, dt: float) -> Tuple[float, float]:
        """预测下一帧位置
        
        Args:
            target: 跟踪目标
            dt: 时间间隔
            
        Returns:
            Tuple[float, float]: 预测位置 (x, y)
        """
        if not target.position_history:
            return (0, 0)
        
        current_pos = target.position_history[-1]
        
        # 使用恒速模型预测
        predicted_x = current_pos[0] + target.velocity[0] * dt
        predicted_y = current_pos[1] + target.velocity[1] * dt
        
        # 添加运动噪声
        noise_x = np.random.normal(0, self.noise_std)
        noise_y = np.random.normal(0, self.noise_std)
        
        return (predicted_x + noise_x, predicted_y + noise_y)
    
    def update_motion_model(self, target: TrackingTarget, new_position: Tuple[float, float]):
        """更新运动模型
        
        Args:
            target: 跟踪目标
            new_position: 新位置
        """
        target.position_history.append(new_position)
        
        # 计算速度
        if len(target.position_history) >= 2:
            prev_pos = target.position_history[-2]
            curr_pos = target.position_history[-1]
            
            dt = target.last_update_time - (target.last_update_time - 1/30.0)  # 假设30fps
            if dt > 0:
                vx = (curr_pos[0] - prev_pos[0]) / dt
                vy = (curr_pos[1] - prev_pos[1]) / dt
                
                # 速度平滑
                smoothing = 0.7
                target.velocity = (
                    target.velocity[0] * smoothing + vx * (1 - smoothing),
                    target.velocity[1] * smoothing + vy * (1 - smoothing)
                )
        
        # 计算加速度
        if len(target.position_history) >= 3:
            # 简化的加速度计算
            target.acceleration = (0, 0)  # 可以实现更复杂的加速度模型

class AppearanceModel:
    """外观模型"""
    
    def __init__(self, feature_dim: int = 512):
        self.feature_dim = feature_dim
    
    def extract_appearance_feature(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """提取外观特征
        
        Args:
            image: 输入图像
            bbox: 边界框
            
        Returns:
            np.ndarray: 外观特征向量
        """
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return np.zeros(self.feature_dim)
        
        # 简化的外观特征提取（实际应用中可使用深度学习特征）
        # 颜色直方图
        roi_resized = cv2.resize(roi, (64, 64))
        hist_b = cv2.calcHist([roi_resized], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([roi_resized], [1], None, [32], [0, 256])
        hist_r = cv2.calcHist([roi_resized], [2], None, [32], [0, 256])
        
        # 纹理特征（LBP简化版）
        gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
        texture_feature = self._extract_texture_feature(gray)
        
        # 合并特征
        color_feature = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])
        appearance_feature = np.concatenate([color_feature, texture_feature])
        
        # 归一化
        if np.linalg.norm(appearance_feature) > 0:
            appearance_feature = appearance_feature / np.linalg.norm(appearance_feature)
        
        # 填充或截断到指定维度
        if len(appearance_feature) > self.feature_dim:
            appearance_feature = appearance_feature[:self.feature_dim]
        elif len(appearance_feature) < self.feature_dim:
            padding = np.zeros(self.feature_dim - len(appearance_feature))
            appearance_feature = np.concatenate([appearance_feature, padding])
        
        return appearance_feature
    
    def _extract_texture_feature(self, gray_image: np.ndarray) -> np.ndarray:
        """提取纹理特征"""
        # 简化的纹理特征（梯度统计）
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # 统计特征
        texture_features = [
            np.mean(grad_mag),
            np.std(grad_mag),
            np.mean(gray_image),
            np.std(gray_image)
        ]
        
        return np.array(texture_features)
    
    def compute_similarity(self, feature1: np.ndarray, feature2: np.ndarray) -> float:
        """计算外观相似度
        
        Args:
            feature1: 特征向量1
            feature2: 特征向量2
            
        Returns:
            float: 相似度 [0, 1]
        """
        if feature1.size == 0 or feature2.size == 0:
            return 0.0
        
        # 余弦相似度
        norm1 = np.linalg.norm(feature1)
        norm2 = np.linalg.norm(feature2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(feature1, feature2) / (norm1 * norm2)
        return max(0.0, similarity)  # 确保非负

class MultimodalFusionTracker:
    """多模态融合跟踪器"""
    
    def __init__(self, config: FusionConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 跟踪目标
        self.targets: Dict[str, TrackingTarget] = {}
        self.next_track_id = 1
        
        # 子模块
        self.motion_predictor = MotionPredictor(config.motion_noise_std)
        self.appearance_model = AppearanceModel()
        
        # 性能统计
        self.stats = {
            'total_tracks': 0,
            'active_tracks': 0,
            'avg_tracking_time': 0.0,
            'fusion_accuracy': 0.0
        }
        
        self.logger.info(f"多模态融合跟踪器初始化完成 - 策略: {config.fusion_strategy.value}")
    
    def update(self, 
               face_detections: List[Any] = None,
               body_detections: List[HumanBodyDetection] = None,
               pose_detections: List[Any] = None,
               image: Optional[np.ndarray] = None) -> List[TrackingTarget]:
        """更新跟踪
        
        Args:
            face_detections: 人脸检测结果
            body_detections: 人体检测结果
            pose_detections: 姿态检测结果
            image: 当前帧图像
            
        Returns:
            List[TrackingTarget]: 更新后的跟踪目标列表
        """
        start_time = time.time()
        
        try:
            # 提取多模态特征
            modality_features = self._extract_multimodal_features(
                face_detections, body_detections, pose_detections, image
            )
            
            # 数据关联
            associations = self._associate_detections(modality_features)
            
            # 更新跟踪目标
            self._update_targets(associations, image)
            
            # 预测丢失目标
            self._predict_lost_targets()
            
            # 清理无效目标
            self._cleanup_targets()
            
            # 更新统计信息
            tracking_time = time.time() - start_time
            self._update_stats(tracking_time)
            
            return list(self.targets.values())
            
        except Exception as e:
            self.logger.error(f"跟踪更新失败: {e}")
            return list(self.targets.values())
    
    def _extract_multimodal_features(self, 
                                   face_detections: List[Any],
                                   body_detections: List[HumanBodyDetection],
                                   pose_detections: List[Any],
                                   image: Optional[np.ndarray]) -> List[Dict[TrackingModality, ModalityFeature]]:
        """提取多模态特征"""
        features_list = []
        current_time = time.time()
        
        # 合并所有检测结果
        all_detections = []
        
        # 处理人脸检测
        if face_detections:
            for face in face_detections:
                if hasattr(face, 'bbox'):
                    bbox = face.bbox
                    confidence = getattr(face, 'confidence', 0.5)
                    
                    feature = ModalityFeature(
                        modality=TrackingModality.FACE,
                        feature_vector=getattr(face, 'embedding', np.array([])),
                        confidence=confidence,
                        bbox=bbox,
                        timestamp=current_time,
                        quality_score=confidence
                    )
                    
                    all_detections.append({TrackingModality.FACE: feature})
        
        # 处理人体检测
        if body_detections:
            for body in body_detections:
                bbox = body.bbox
                confidence = body.confidence
                
                # 人体特征
                body_feature = ModalityFeature(
                    modality=TrackingModality.BODY,
                    feature_vector=np.array([]),  # 可以添加人体特征向量
                    confidence=confidence,
                    bbox=bbox,
                    timestamp=current_time,
                    quality_score=confidence
                )
                
                features = {TrackingModality.BODY: body_feature}
                
                # 姿态特征
                if body.keypoints:
                    pose_vector = self._keypoints_to_vector(body.keypoints)
                    pose_feature = ModalityFeature(
                        modality=TrackingModality.POSE,
                        feature_vector=pose_vector,
                        confidence=confidence * 0.8,  # 姿态置信度稍低
                        bbox=bbox,
                        timestamp=current_time,
                        quality_score=confidence * 0.8
                    )
                    features[TrackingModality.POSE] = pose_feature
                
                # 外观特征
                if image is not None:
                    appearance_vector = self.appearance_model.extract_appearance_feature(image, bbox)
                    appearance_feature = ModalityFeature(
                        modality=TrackingModality.APPEARANCE,
                        feature_vector=appearance_vector,
                        confidence=confidence * 0.6,
                        bbox=bbox,
                        timestamp=current_time,
                        quality_score=confidence * 0.6
                    )
                    features[TrackingModality.APPEARANCE] = appearance_feature
                
                all_detections.append(features)
        
        return all_detections
    
    def _keypoints_to_vector(self, keypoints: Dict) -> np.ndarray:
        """将关键点转换为特征向量"""
        vector = []
        for name in ['nose', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']:
            if name in keypoints:
                kp = keypoints[name]
                vector.extend([kp.x, kp.y, kp.confidence])
            else:
                vector.extend([0, 0, 0])
        return np.array(vector)
    
    def _associate_detections(self, features_list: List[Dict[TrackingModality, ModalityFeature]]) -> Dict[str, Dict[TrackingModality, ModalityFeature]]:
        """数据关联"""
        associations = {}
        
        for features in features_list:
            best_match_id = None
            best_score = 0.0
            
            # 与现有目标匹配
            for track_id, target in self.targets.items():
                if target.state == TrackingState.INACTIVE:
                    continue
                
                score = self._compute_association_score(features, target)
                
                if score > best_score and score > self.config.association_threshold:
                    best_score = score
                    best_match_id = track_id
            
            if best_match_id:
                # 关联到现有目标
                associations[best_match_id] = features
            else:
                # 创建新目标
                new_id = f"track_{self.next_track_id}"
                self.next_track_id += 1
                associations[new_id] = features
        
        return associations
    
    def _compute_association_score(self, features: Dict[TrackingModality, ModalityFeature], target: TrackingTarget) -> float:
        """计算关联分数"""
        total_score = 0.0
        total_weight = 0.0
        
        for modality, feature in features.items():
            if modality in target.features:
                target_feature = target.features[modality]
                
                # 计算相似度
                similarity = self._compute_feature_similarity(feature, target_feature)
                
                # 位置距离
                position_score = self._compute_position_score(feature.bbox, target)
                
                # 综合分数
                modality_score = similarity * 0.7 + position_score * 0.3
                
                # 加权
                weight = self.config.modality_weights.get(modality, 0.1)
                total_score += modality_score * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _compute_feature_similarity(self, feature1: ModalityFeature, feature2: ModalityFeature) -> float:
        """计算特征相似度"""
        if feature1.feature_vector.size == 0 or feature2.feature_vector.size == 0:
            return 0.5  # 默认中等相似度
        
        if feature1.modality == TrackingModality.APPEARANCE:
            return self.appearance_model.compute_similarity(feature1.feature_vector, feature2.feature_vector)
        else:
            # 其他模态使用余弦相似度
            norm1 = np.linalg.norm(feature1.feature_vector)
            norm2 = np.linalg.norm(feature2.feature_vector)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = np.dot(feature1.feature_vector, feature2.feature_vector) / (norm1 * norm2)
            return max(0.0, similarity)
    
    def _compute_position_score(self, bbox: Tuple[int, int, int, int], target: TrackingTarget) -> float:
        """计算位置分数"""
        if not target.position_history:
            return 0.5
        
        # 当前位置
        x1, y1, x2, y2 = bbox
        current_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        # 预测位置
        dt = 1/30.0  # 假设30fps
        predicted_pos = self.motion_predictor.predict_position(target, dt)
        
        # 计算距离
        distance = math.sqrt(
            (current_center[0] - predicted_pos[0]) ** 2 +
            (current_center[1] - predicted_pos[1]) ** 2
        )
        
        # 转换为分数（距离越小分数越高）
        max_distance = 200.0  # 最大允许距离
        score = max(0.0, 1.0 - distance / max_distance)
        
        return score
    
    def _update_targets(self, associations: Dict[str, Dict[TrackingModality, ModalityFeature]], image: Optional[np.ndarray]):
        """更新跟踪目标"""
        current_time = time.time()
        
        for track_id, features in associations.items():
            if track_id in self.targets:
                # 更新现有目标
                target = self.targets[track_id]
                target.features.update(features)
                target.state = TrackingState.ACTIVE
                target.missing_frames = 0
                target.last_update_time = current_time
                target.total_frames += 1
                
                # 更新位置历史
                primary_feature = self._get_primary_feature(features)
                if primary_feature:
                    bbox = primary_feature.bbox
                    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                    self.motion_predictor.update_motion_model(target, center)
                    
                    # 更新置信度历史
                    target.confidence_history.append(primary_feature.confidence)
                
                # 更新外观模型
                if image is not None and TrackingModality.APPEARANCE in features:
                    appearance_feature = features[TrackingModality.APPEARANCE]
                    if target.appearance_model is None:
                        target.appearance_model = appearance_feature.feature_vector.copy()
                    else:
                        # 指数移动平均更新
                        alpha = self.config.appearance_update_rate
                        target.appearance_model = (
                            (1 - alpha) * target.appearance_model +
                            alpha * appearance_feature.feature_vector
                        )
            else:
                # 创建新目标
                primary_feature = self._get_primary_feature(features)
                if primary_feature:
                    bbox = primary_feature.bbox
                    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                    
                    target = TrackingTarget(
                        track_id=track_id,
                        state=TrackingState.ACTIVE,
                        features=features,
                        position_history=deque([center], maxlen=30),
                        velocity=(0.0, 0.0),
                        acceleration=(0.0, 0.0),
                        last_update_time=current_time,
                        missing_frames=0,
                        total_frames=1,
                        confidence_history=deque([primary_feature.confidence], maxlen=10)
                    )
                    
                    # 初始化外观模型
                    if TrackingModality.APPEARANCE in features:
                        target.appearance_model = features[TrackingModality.APPEARANCE].feature_vector.copy()
                    
                    self.targets[track_id] = target
                    self.stats['total_tracks'] += 1
    
    def _get_primary_feature(self, features: Dict[TrackingModality, ModalityFeature]) -> Optional[ModalityFeature]:
        """获取主要特征"""
        # 优先级：人体 > 人脸 > 姿态 > 外观
        priority_order = [TrackingModality.BODY, TrackingModality.FACE, TrackingModality.POSE, TrackingModality.APPEARANCE]
        
        for modality in priority_order:
            if modality in features:
                return features[modality]
        
        # 如果没有优先模态，返回置信度最高的
        if features:
            return max(features.values(), key=lambda f: f.confidence)
        
        return None
    
    def _predict_lost_targets(self):
        """预测丢失目标"""
        current_time = time.time()
        
        for target in self.targets.values():
            if target.state == TrackingState.ACTIVE:
                continue
            
            # 预测位置
            dt = current_time - target.last_update_time
            predicted_pos = self.motion_predictor.predict_position(target, dt)
            
            # 更新预测位置（用于可视化）
            target.position_history.append(predicted_pos)
    
    def _cleanup_targets(self):
        """清理无效目标"""
        to_remove = []
        
        for track_id, target in self.targets.items():
            target.missing_frames += 1
            
            if target.missing_frames > self.config.max_missing_frames:
                to_remove.append(track_id)
            elif target.missing_frames > 5:
                target.state = TrackingState.LOST
        
        for track_id in to_remove:
            del self.targets[track_id]
    
    def get_primary_target(self) -> Optional[TrackingTarget]:
        """获取主要跟踪目标
        
        Returns:
            Optional[TrackingTarget]: 主要跟踪目标
        """
        active_targets = [t for t in self.targets.values() if t.state == TrackingState.ACTIVE]
        
        if not active_targets:
            return None
        
        # 选择置信度最高且跟踪时间最长的目标
        def target_score(target):
            avg_confidence = np.mean(list(target.confidence_history)) if target.confidence_history else 0.0
            duration_score = min(target.total_frames / 100.0, 1.0)  # 归一化持续时间
            return avg_confidence * 0.7 + duration_score * 0.3
        
        return max(active_targets, key=target_score)
    
    def get_target_position(self, track_id: str) -> Optional[Tuple[float, float]]:
        """获取目标位置
        
        Args:
            track_id: 跟踪ID
            
        Returns:
            Optional[Tuple[float, float]]: 目标位置 (x, y)
        """
        if track_id in self.targets and self.targets[track_id].position_history:
            return self.targets[track_id].position_history[-1]
        return None
    
    def visualize_tracking(self, image: np.ndarray) -> np.ndarray:
        """可视化跟踪结果
        
        Args:
            image: 输入图像
            
        Returns:
            np.ndarray: 标注后的图像
        """
        result_image = image.copy()
        
        for target in self.targets.values():
            if target.state == TrackingState.INACTIVE:
                continue
            
            # 获取主要特征的边界框
            primary_feature = self._get_primary_feature(target.features)
            if not primary_feature:
                continue
            
            bbox = primary_feature.bbox
            x1, y1, x2, y2 = bbox
            
            # 根据状态选择颜色
            if target.state == TrackingState.ACTIVE:
                color = (0, 255, 0)  # 绿色
            elif target.state == TrackingState.LOST:
                color = (0, 255, 255)  # 黄色
            else:
                color = (0, 0, 255)  # 红色
            
            # 绘制边界框
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制跟踪ID和信息
            avg_conf = np.mean(list(target.confidence_history)) if target.confidence_history else 0.0
            label = f"ID: {target.track_id} ({avg_conf:.2f})"
            cv2.putText(result_image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 绘制轨迹
            if len(target.position_history) > 1:
                points = [(int(pos[0]), int(pos[1])) for pos in target.position_history]
                for i in range(1, len(points)):
                    cv2.line(result_image, points[i-1], points[i], color, 1)
            
            # 绘制速度向量
            if target.position_history:
                center = target.position_history[-1]
                end_point = (
                    int(center[0] + target.velocity[0] * 0.1),
                    int(center[1] + target.velocity[1] * 0.1)
                )
                cv2.arrowedLine(result_image, 
                              (int(center[0]), int(center[1])), 
                              end_point, color, 2)
        
        return result_image
    
    def _update_stats(self, tracking_time: float):
        """更新统计信息"""
        self.stats['active_tracks'] = len([t for t in self.targets.values() if t.state == TrackingState.ACTIVE])
        
        # 更新平均跟踪时间
        if self.stats['avg_tracking_time'] == 0:
            self.stats['avg_tracking_time'] = tracking_time
        else:
            self.stats['avg_tracking_time'] = (
                self.stats['avg_tracking_time'] * 0.9 + tracking_time * 0.1
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
    
    def reset(self):
        """重置跟踪器"""
        self.targets.clear()
        self.next_track_id = 1
        self.stats['total_tracks'] = 0
        self.logger.info("多模态融合跟踪器已重置")

# 测试代码
if __name__ == "__main__":
    # 创建配置
    config = FusionConfig(
        fusion_strategy=FusionStrategy.ADAPTIVE,
        max_missing_frames=10
    )
    
    # 创建跟踪器
    tracker = MultimodalFusionTracker(config)
    
    # 模拟检测结果
    from ..models.human_body_detector import HumanBodyDetection, HumanBodyKeypoint
    
    mock_body_detection = HumanBodyDetection(
        bbox=(100, 100, 300, 400),
        confidence=0.85,
        keypoints={
            'nose': HumanBodyKeypoint(200, 150, 0.9),
            'left_shoulder': HumanBodyKeypoint(180, 200, 0.8),
            'right_shoulder': HumanBodyKeypoint(220, 200, 0.8)
        },
        body_parts={},
        pose_angle=0.0
    )
    
    # 模拟图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 执行跟踪
    targets = tracker.update(
        face_detections=None,
        body_detections=[mock_body_detection],
        pose_detections=None,
        image=test_image
    )
    
    # 获取主要目标
    primary_target = tracker.get_primary_target()
    
    print(f"跟踪目标数量: {len(targets)}")
    if primary_target:
        print(f"主要目标: {primary_target.track_id}")
    
    # 可视化
    result_image = tracker.visualize_tracking(test_image)
    
    # 统计信息
    stats = tracker.get_stats()
    print(f"跟踪统计: {stats}")