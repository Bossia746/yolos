#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
复杂场景分析模块
专门处理室内复杂环境下的跌倒检测和人体跟踪
包括地面障碍物、家具遮挡、多人混合状态等复杂情况
"""

import time
import logging
import threading
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import json
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)

class PersonState(Enum):
    """人员状态"""
    STANDING = "standing"  # 站立
    SITTING = "sitting"  # 坐着
    LYING = "lying"  # 躺着
    FALLEN = "fallen"  # 跌倒
    CROUCHING = "crouching"  # 蹲着
    BENDING = "bending"  # 弯腰
    UNKNOWN = "unknown"  # 未知

class ObstacleType(Enum):
    """障碍物类型"""
    TOY = "toy"  # 玩具
    FURNITURE = "furniture"  # 家具
    BOOKSHELF = "bookshelf"  # 书柜
    TABLE = "table"  # 桌子
    CHAIR = "chair"  # 椅子
    SOFA = "sofa"  # 沙发
    BED = "bed"  # 床
    CARPET = "carpet"  # 地毯
    STAIRS = "stairs"  # 楼梯
    UNKNOWN_OBJECT = "unknown_object"  # 未知物体

class SceneComplexity(Enum):
    """场景复杂度"""
    SIMPLE = "simple"  # 简单场景
    MODERATE = "moderate"  # 中等复杂
    COMPLEX = "complex"  # 复杂场景
    VERY_COMPLEX = "very_complex"  # 极复杂场景

class AlertLevel(Enum):
    """警报级别"""
    NORMAL = "normal"  # 正常
    CAUTION = "caution"  # 注意
    WARNING = "warning"  # 警告
    CRITICAL = "critical"  # 危急
    EMERGENCY = "emergency"  # 紧急

@dataclass
class PersonDetection:
    """人员检测结果"""
    person_id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    keypoints: Optional[List[List[float]]] = None  # 关键点 [[x, y, conf], ...]
    state: PersonState = PersonState.UNKNOWN
    state_confidence: float = 0.0
    
    # 位置信息
    center_point: Tuple[float, float] = (0.0, 0.0)
    ground_point: Tuple[float, float] = (0.0, 0.0)  # 脚部接地点
    height_ratio: float = 0.0  # 身高比例
    
    # 运动信息
    velocity: Tuple[float, float] = (0.0, 0.0)
    acceleration: Tuple[float, float] = (0.0, 0.0)
    
    # 姿态信息
    body_angle: float = 0.0  # 身体角度
    head_position: Tuple[float, float] = (0.0, 0.0)
    
    # 时间信息
    timestamp: float = 0.0
    duration_in_state: float = 0.0  # 在当前状态持续时间
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()
        
        # 计算中心点
        if len(self.bbox) >= 4:
            self.center_point = (
                (self.bbox[0] + self.bbox[2]) / 2,
                (self.bbox[1] + self.bbox[3]) / 2
            )
            # 估算脚部接地点
            self.ground_point = (
                self.center_point[0],
                self.bbox[3]  # 底部边界
            )
            # 计算身高比例
            self.height_ratio = (self.bbox[3] - self.bbox[1]) / (self.bbox[2] - self.bbox[0])

@dataclass
class ObstacleDetection:
    """障碍物检测结果"""
    obstacle_id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    obstacle_type: ObstacleType
    
    # 位置信息
    center_point: Tuple[float, float] = (0.0, 0.0)
    ground_area: List[Tuple[float, float]] = field(default_factory=list)  # 地面占用区域
    height: float = 0.0  # 高度
    
    # 安全信息
    is_hazardous: bool = False  # 是否危险
    risk_level: float = 0.0  # 风险等级 (0-1)
    
    # 时间信息
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()
        
        # 计算中心点
        if len(self.bbox) >= 4:
            self.center_point = (
                (self.bbox[0] + self.bbox[2]) / 2,
                (self.bbox[1] + self.bbox[3]) / 2
            )
            # 估算高度
            self.height = self.bbox[3] - self.bbox[1]

@dataclass
class SceneAnalysis:
    """场景分析结果"""
    timestamp: float
    complexity: SceneComplexity
    
    # 人员信息
    total_persons: int = 0
    standing_persons: int = 0
    fallen_persons: int = 0
    sitting_persons: int = 0
    
    # 障碍物信息
    total_obstacles: int = 0
    ground_obstacles: int = 0  # 地面障碍物数量
    furniture_count: int = 0
    
    # 风险评估
    overall_risk: float = 0.0  # 整体风险 (0-1)
    fall_risk_areas: List[Tuple[float, float, float, float]] = field(default_factory=list)  # 跌倒风险区域
    
    # 环境因素
    lighting_quality: float = 0.5  # 光照质量
    visibility_score: float = 0.5  # 可见度评分
    occlusion_level: float = 0.0  # 遮挡程度
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()

@dataclass
class FallAlert:
    """跌倒警报"""
    alert_id: str
    person_id: int
    alert_level: AlertLevel
    timestamp: float
    
    # 位置信息
    fall_location: Tuple[float, float]
    affected_area: List[Tuple[float, float]] = field(default_factory=list)
    
    # 跌倒信息
    fall_confidence: float = 0.0
    fall_duration: float = 0.0  # 跌倒持续时间
    impact_severity: float = 0.0  # 冲击严重程度
    
    # 环境因素
    nearby_obstacles: List[int] = field(default_factory=list)  # 附近障碍物ID
    other_persons_nearby: List[int] = field(default_factory=list)  # 附近其他人员ID
    
    # 响应信息
    requires_immediate_attention: bool = False
    suggested_actions: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()

class PoseEstimator:
    """姿态估计器
    
    基于关键点检测分析人体姿态和状态
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 关键点索引（COCO格式）
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # 身体部位组
        self.body_parts = {
            'head': [0, 1, 2, 3, 4],  # 头部
            'torso': [5, 6, 11, 12],  # 躯干
            'arms': [5, 6, 7, 8, 9, 10],  # 手臂
            'legs': [11, 12, 13, 14, 15, 16]  # 腿部
        }
        
        self.logger.info("姿态估计器初始化完成")
    
    def estimate_pose_state(self, keypoints: List[List[float]], bbox: List[float]) -> Tuple[PersonState, float]:
        """估计姿态状态
        
        Args:
            keypoints: 关键点列表 [[x, y, conf], ...]
            bbox: 边界框 [x1, y1, x2, y2]
            
        Returns:
            Tuple[PersonState, float]: (状态, 置信度)
        """
        try:
            if not keypoints or len(keypoints) < 17:
                return PersonState.UNKNOWN, 0.0
            
            # 提取有效关键点
            valid_keypoints = [(kp[0], kp[1]) for kp in keypoints if kp[2] > 0.3]
            
            if len(valid_keypoints) < 5:
                return PersonState.UNKNOWN, 0.2
            
            # 分析身体角度
            body_angle = self._calculate_body_angle(keypoints)
            
            # 分析头部位置
            head_height_ratio = self._calculate_head_height_ratio(keypoints, bbox)
            
            # 分析腿部状态
            leg_state = self._analyze_leg_state(keypoints)
            
            # 分析躯干状态
            torso_state = self._analyze_torso_state(keypoints)
            
            # 综合判断状态
            state, confidence = self._classify_pose_state(
                body_angle, head_height_ratio, leg_state, torso_state
            )
            
            return state, confidence
            
        except Exception as e:
            self.logger.error(f"姿态状态估计失败: {e}")
            return PersonState.UNKNOWN, 0.0
    
    def _calculate_body_angle(self, keypoints: List[List[float]]) -> float:
        """计算身体角度
        
        Args:
            keypoints: 关键点列表
            
        Returns:
            float: 身体角度（度）
        """
        try:
            # 使用肩膀和髋部计算身体主轴角度
            left_shoulder = keypoints[5][:2] if keypoints[5][2] > 0.3 else None
            right_shoulder = keypoints[6][:2] if keypoints[6][2] > 0.3 else None
            left_hip = keypoints[11][:2] if keypoints[11][2] > 0.3 else None
            right_hip = keypoints[12][:2] if keypoints[12][2] > 0.3 else None
            
            # 计算肩膀中点和髋部中点
            if left_shoulder and right_shoulder:
                shoulder_center = (
                    (left_shoulder[0] + right_shoulder[0]) / 2,
                    (left_shoulder[1] + right_shoulder[1]) / 2
                )
            elif left_shoulder:
                shoulder_center = left_shoulder
            elif right_shoulder:
                shoulder_center = right_shoulder
            else:
                return 90.0  # 默认垂直
            
            if left_hip and right_hip:
                hip_center = (
                    (left_hip[0] + right_hip[0]) / 2,
                    (left_hip[1] + right_hip[1]) / 2
                )
            elif left_hip:
                hip_center = left_hip
            elif right_hip:
                hip_center = right_hip
            else:
                return 90.0  # 默认垂直
            
            # 计算角度
            dx = hip_center[0] - shoulder_center[0]
            dy = hip_center[1] - shoulder_center[1]
            
            if abs(dy) < 1e-6:
                return 0.0  # 水平
            
            angle = np.degrees(np.arctan2(abs(dx), abs(dy)))
            return angle
            
        except Exception as e:
            self.logger.debug(f"身体角度计算失败: {e}")
            return 90.0
    
    def _calculate_head_height_ratio(self, keypoints: List[List[float]], bbox: List[float]) -> float:
        """计算头部高度比例
        
        Args:
            keypoints: 关键点列表
            bbox: 边界框
            
        Returns:
            float: 头部高度比例 (0-1)
        """
        try:
            # 获取头部关键点
            head_points = []
            for i in [0, 1, 2, 3, 4]:  # 鼻子、眼睛、耳朵
                if keypoints[i][2] > 0.3:
                    head_points.append(keypoints[i][:2])
            
            if not head_points:
                return 0.5  # 默认中等高度
            
            # 计算头部平均高度
            avg_head_y = np.mean([p[1] for p in head_points])
            
            # 计算相对于边界框的高度比例
            bbox_height = bbox[3] - bbox[1]
            if bbox_height <= 0:
                return 0.5
            
            head_ratio = (avg_head_y - bbox[1]) / bbox_height
            return np.clip(head_ratio, 0.0, 1.0)
            
        except Exception as e:
            self.logger.debug(f"头部高度比例计算失败: {e}")
            return 0.5
    
    def _analyze_leg_state(self, keypoints: List[List[float]]) -> Dict[str, Any]:
        """分析腿部状态
        
        Args:
            keypoints: 关键点列表
            
        Returns:
            Dict[str, Any]: 腿部状态信息
        """
        try:
            leg_info = {
                'knee_angle': 180.0,  # 膝盖角度
                'leg_spread': 0.0,  # 腿部张开程度
                'ankle_height': 0.0,  # 脚踝高度
                'leg_visibility': 0.0  # 腿部可见度
            }
            
            # 获取腿部关键点
            left_hip = keypoints[11][:2] if keypoints[11][2] > 0.3 else None
            right_hip = keypoints[12][:2] if keypoints[12][2] > 0.3 else None
            left_knee = keypoints[13][:2] if keypoints[13][2] > 0.3 else None
            right_knee = keypoints[14][:2] if keypoints[14][2] > 0.3 else None
            left_ankle = keypoints[15][:2] if keypoints[15][2] > 0.3 else None
            right_ankle = keypoints[16][:2] if keypoints[16][2] > 0.3 else None
            
            # 计算腿部可见度
            visible_leg_points = sum([
                1 for p in [left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]
                if p is not None
            ])
            leg_info['leg_visibility'] = visible_leg_points / 6.0
            
            # 计算膝盖角度（如果可见）
            if left_hip and left_knee and left_ankle:
                angle = self._calculate_joint_angle(left_hip, left_knee, left_ankle)
                leg_info['knee_angle'] = min(leg_info['knee_angle'], angle)
            
            if right_hip and right_knee and right_ankle:
                angle = self._calculate_joint_angle(right_hip, right_knee, right_ankle)
                leg_info['knee_angle'] = min(leg_info['knee_angle'], angle)
            
            # 计算腿部张开程度
            if left_ankle and right_ankle:
                leg_info['leg_spread'] = abs(left_ankle[0] - right_ankle[0])
            
            # 计算脚踝平均高度
            ankle_heights = []
            if left_ankle:
                ankle_heights.append(left_ankle[1])
            if right_ankle:
                ankle_heights.append(right_ankle[1])
            
            if ankle_heights:
                leg_info['ankle_height'] = np.mean(ankle_heights)
            
            return leg_info
            
        except Exception as e:
            self.logger.debug(f"腿部状态分析失败: {e}")
            return {'knee_angle': 180.0, 'leg_spread': 0.0, 'ankle_height': 0.0, 'leg_visibility': 0.0}
    
    def _analyze_torso_state(self, keypoints: List[List[float]]) -> Dict[str, Any]:
        """分析躯干状态
        
        Args:
            keypoints: 关键点列表
            
        Returns:
            Dict[str, Any]: 躯干状态信息
        """
        try:
            torso_info = {
                'shoulder_level': True,  # 肩膀是否水平
                'torso_width': 0.0,  # 躯干宽度
                'torso_height': 0.0,  # 躯干高度
                'torso_visibility': 0.0  # 躯干可见度
            }
            
            # 获取躯干关键点
            left_shoulder = keypoints[5][:2] if keypoints[5][2] > 0.3 else None
            right_shoulder = keypoints[6][:2] if keypoints[6][2] > 0.3 else None
            left_hip = keypoints[11][:2] if keypoints[11][2] > 0.3 else None
            right_hip = keypoints[12][:2] if keypoints[12][2] > 0.3 else None
            
            # 计算躯干可见度
            visible_torso_points = sum([
                1 for p in [left_shoulder, right_shoulder, left_hip, right_hip]
                if p is not None
            ])
            torso_info['torso_visibility'] = visible_torso_points / 4.0
            
            # 分析肩膀水平度
            if left_shoulder and right_shoulder:
                shoulder_height_diff = abs(left_shoulder[1] - right_shoulder[1])
                torso_info['shoulder_level'] = shoulder_height_diff < 20  # 像素阈值
                torso_info['torso_width'] = abs(left_shoulder[0] - right_shoulder[0])
            
            # 计算躯干高度
            if left_shoulder and left_hip:
                torso_info['torso_height'] = abs(left_shoulder[1] - left_hip[1])
            elif right_shoulder and right_hip:
                torso_info['torso_height'] = abs(right_shoulder[1] - right_hip[1])
            
            return torso_info
            
        except Exception as e:
            self.logger.debug(f"躯干状态分析失败: {e}")
            return {'shoulder_level': True, 'torso_width': 0.0, 'torso_height': 0.0, 'torso_visibility': 0.0}
    
    def _calculate_joint_angle(self, p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
        """计算关节角度
        
        Args:
            p1: 第一个点
            p2: 关节点
            p3: 第三个点
            
        Returns:
            float: 角度（度）
        """
        try:
            # 计算向量
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            # 计算角度
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            
            return angle
            
        except Exception as e:
            self.logger.debug(f"关节角度计算失败: {e}")
            return 180.0
    
    def _classify_pose_state(self, body_angle: float, head_height_ratio: float, 
                           leg_state: Dict[str, Any], torso_state: Dict[str, Any]) -> Tuple[PersonState, float]:
        """分类姿态状态
        
        Args:
            body_angle: 身体角度
            head_height_ratio: 头部高度比例
            leg_state: 腿部状态
            torso_state: 躯干状态
            
        Returns:
            Tuple[PersonState, float]: (状态, 置信度)
        """
        try:
            # 状态评分
            state_scores = {}
            
            # 跌倒检测
            if body_angle < 30 or head_height_ratio > 0.7:  # 身体接近水平或头部位置过低
                fall_score = 0.8
                if leg_state['leg_visibility'] > 0.5 and leg_state['knee_angle'] > 120:
                    fall_score += 0.1  # 腿部伸直增加跌倒可能性
                if not torso_state['shoulder_level']:
                    fall_score += 0.1  # 肩膀不水平增加跌倒可能性
                state_scores[PersonState.FALLEN] = min(fall_score, 1.0)
            
            # 站立检测
            if body_angle > 70 and head_height_ratio < 0.3:  # 身体接近垂直且头部位置较高
                standing_score = 0.7
                if leg_state['leg_visibility'] > 0.6 and leg_state['knee_angle'] > 160:
                    standing_score += 0.2  # 腿部可见且伸直
                if torso_state['shoulder_level']:
                    standing_score += 0.1  # 肩膀水平
                state_scores[PersonState.STANDING] = min(standing_score, 1.0)
            
            # 坐着检测
            if 40 < body_angle < 80 and 0.3 < head_height_ratio < 0.6:
                sitting_score = 0.6
                if leg_state['knee_angle'] < 120:  # 膝盖弯曲
                    sitting_score += 0.2
                if torso_state['torso_visibility'] > 0.7:
                    sitting_score += 0.1
                state_scores[PersonState.SITTING] = min(sitting_score, 1.0)
            
            # 蹲着检测
            if body_angle > 60 and head_height_ratio > 0.4:
                crouching_score = 0.5
                if leg_state['knee_angle'] < 90:  # 膝盖大幅弯曲
                    crouching_score += 0.3
                if leg_state['leg_spread'] < 50:  # 腿部靠近
                    crouching_score += 0.1
                state_scores[PersonState.CROUCHING] = min(crouching_score, 1.0)
            
            # 弯腰检测
            if 30 < body_angle < 60 and head_height_ratio > 0.5:
                bending_score = 0.6
                if leg_state['knee_angle'] > 140:  # 腿部相对伸直
                    bending_score += 0.2
                state_scores[PersonState.BENDING] = min(bending_score, 1.0)
            
            # 选择最高分的状态
            if state_scores:
                best_state = max(state_scores.items(), key=lambda x: x[1])
                return best_state[0], best_state[1]
            else:
                return PersonState.UNKNOWN, 0.3
                
        except Exception as e:
            self.logger.error(f"姿态状态分类失败: {e}")
            return PersonState.UNKNOWN, 0.0

class ObstacleDetector:
    """障碍物检测器
    
    检测和分类室内环境中的各种障碍物
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 障碍物分类器（这里应该是实际的分类模型）
        self.obstacle_classifier = self._initialize_classifier()
        
        # 障碍物特征
        self.obstacle_features = {
            ObstacleType.TOY: {'min_size': 10, 'max_size': 200, 'typical_height_ratio': 0.5},
            ObstacleType.FURNITURE: {'min_size': 100, 'max_size': 1000, 'typical_height_ratio': 1.5},
            ObstacleType.BOOKSHELF: {'min_size': 200, 'max_size': 800, 'typical_height_ratio': 3.0},
            ObstacleType.TABLE: {'min_size': 150, 'max_size': 600, 'typical_height_ratio': 1.2},
            ObstacleType.CHAIR: {'min_size': 80, 'max_size': 300, 'typical_height_ratio': 1.8},
            ObstacleType.SOFA: {'min_size': 300, 'max_size': 1000, 'typical_height_ratio': 1.0},
        }
        
        self.logger.info("障碍物检测器初始化完成")
    
    def _initialize_classifier(self) -> Dict[str, Any]:
        """初始化障碍物分类器
        
        Returns:
            Dict[str, Any]: 分类器配置
        """
        # 这里应该加载实际的分类模型
        # 为了演示，返回模拟配置
        return {
            'model_type': 'yolo',
            'confidence_threshold': 0.5,
            'nms_threshold': 0.4
        }
    
    def detect_obstacles(self, frame: np.ndarray, existing_detections: List[Dict] = None) -> List[ObstacleDetection]:
        """检测障碍物
        
        Args:
            frame: 输入帧
            existing_detections: 已有检测结果
            
        Returns:
            List[ObstacleDetection]: 障碍物检测结果
        """
        try:
            obstacles = []
            
            # 如果有现有检测结果，过滤出非人员对象
            if existing_detections:
                for i, detection in enumerate(existing_detections):
                    if detection.get('class', '') != 'person':
                        obstacle_type = self._classify_obstacle_type(detection)
                        
                        obstacle = ObstacleDetection(
                            obstacle_id=i,
                            bbox=detection.get('bbox', [0, 0, 100, 100]),
                            confidence=detection.get('confidence', 0.5),
                            obstacle_type=obstacle_type
                        )
                        
                        # 评估风险等级
                        obstacle.risk_level = self._assess_obstacle_risk(obstacle)
                        obstacle.is_hazardous = obstacle.risk_level > 0.6
                        
                        obstacles.append(obstacle)
            
            # 基于图像分析检测额外障碍物
            additional_obstacles = self._detect_image_based_obstacles(frame)
            obstacles.extend(additional_obstacles)
            
            return obstacles
            
        except Exception as e:
            self.logger.error(f"障碍物检测失败: {e}")
            return []
    
    def _classify_obstacle_type(self, detection: Dict[str, Any]) -> ObstacleType:
        """分类障碍物类型
        
        Args:
            detection: 检测结果
            
        Returns:
            ObstacleType: 障碍物类型
        """
        try:
            class_name = detection.get('class', '').lower()
            bbox = detection.get('bbox', [0, 0, 100, 100])
            
            # 计算尺寸特征
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            height_ratio = height / width if width > 0 else 1.0
            
            # 基于类别名称分类
            if 'chair' in class_name:
                return ObstacleType.CHAIR
            elif 'table' in class_name or 'desk' in class_name:
                return ObstacleType.TABLE
            elif 'sofa' in class_name or 'couch' in class_name:
                return ObstacleType.SOFA
            elif 'bed' in class_name:
                return ObstacleType.BED
            elif 'book' in class_name or 'shelf' in class_name:
                return ObstacleType.BOOKSHELF
            elif any(toy in class_name for toy in ['toy', 'ball', 'doll', 'game']):
                return ObstacleType.TOY
            
            # 基于尺寸特征分类
            if area < 5000:  # 小物体
                return ObstacleType.TOY
            elif height_ratio > 2.0:  # 高瘦物体
                return ObstacleType.BOOKSHELF
            elif height_ratio < 0.8:  # 宽扁物体
                return ObstacleType.TABLE
            elif area > 50000:  # 大物体
                return ObstacleType.FURNITURE
            else:
                return ObstacleType.UNKNOWN_OBJECT
                
        except Exception as e:
            self.logger.debug(f"障碍物类型分类失败: {e}")
            return ObstacleType.UNKNOWN_OBJECT
    
    def _assess_obstacle_risk(self, obstacle: ObstacleDetection) -> float:
        """评估障碍物风险等级
        
        Args:
            obstacle: 障碍物检测结果
            
        Returns:
            float: 风险等级 (0-1)
        """
        try:
            risk_score = 0.0
            
            # 基于类型的基础风险
            type_risk = {
                ObstacleType.TOY: 0.8,  # 玩具风险较高（容易绊倒）
                ObstacleType.STAIRS: 0.9,  # 楼梯风险很高
                ObstacleType.CHAIR: 0.4,  # 椅子中等风险
                ObstacleType.TABLE: 0.3,  # 桌子风险较低
                ObstacleType.SOFA: 0.2,  # 沙发风险低
                ObstacleType.BOOKSHELF: 0.1,  # 书柜风险很低（通常靠墙）
                ObstacleType.UNKNOWN_OBJECT: 0.5  # 未知物体中等风险
            }
            
            risk_score = type_risk.get(obstacle.obstacle_type, 0.3)
            
            # 基于位置的风险调整
            # 地面中央的物体风险更高
            center_x = obstacle.center_point[0]
            # 假设图像宽度为640，中央区域风险更高
            if 200 < center_x < 440:  # 中央区域
                risk_score += 0.2
            
            # 基于尺寸的风险调整
            bbox = obstacle.bbox
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            # 小而低的物体（如玩具）风险更高
            if width < 100 and height < 80:
                risk_score += 0.3
            
            # 限制风险分数范围
            return np.clip(risk_score, 0.0, 1.0)
            
        except Exception as e:
            self.logger.debug(f"障碍物风险评估失败: {e}")
            return 0.3
    
    def _detect_image_based_obstacles(self, frame: np.ndarray) -> List[ObstacleDetection]:
        """基于图像分析检测障碍物
        
        Args:
            frame: 输入帧
            
        Returns:
            List[ObstacleDetection]: 检测到的障碍物
        """
        try:
            obstacles = []
            
            # 这里可以实现基于传统计算机视觉的障碍物检测
            # 例如：边缘检测、轮廓分析、颜色分割等
            
            # 为了演示，返回空列表
            # 实际实现中可以添加更多检测逻辑
            
            return obstacles
            
        except Exception as e:
            self.logger.error(f"基于图像的障碍物检测失败: {e}")
            return []

class ComplexSceneAnalyzer:
    """复杂场景分析器
    
    整合人员检测、障碍物检测和姿态估计，提供综合的场景分析
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化组件
        self.pose_estimator = PoseEstimator()
        self.obstacle_detector = ObstacleDetector()
        
        # 历史数据
        self.person_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))
        self.scene_history = deque(maxlen=100)
        self.alert_history = deque(maxlen=50)
        
        # 配置参数
        self.fall_detection_threshold = 0.7  # 跌倒检测阈值
        self.fall_duration_threshold = 2.0  # 跌倒持续时间阈值（秒）
        self.proximity_threshold = 100  # 邻近阈值（像素）
        
        # 运行状态
        self.running = False
        self.analysis_thread = None
        
        self.logger.info("复杂场景分析器初始化完成")
    
    def analyze_scene(self, frame: np.ndarray, detections: List[Dict], 
                     keypoints_data: Optional[List[List[List[float]]]] = None) -> Tuple[SceneAnalysis, List[FallAlert]]:
        """分析复杂场景
        
        Args:
            frame: 输入帧
            detections: 检测结果
            keypoints_data: 关键点数据（可选）
            
        Returns:
            Tuple[SceneAnalysis, List[FallAlert]]: (场景分析结果, 跌倒警报列表)
        """
        try:
            current_time = time.time()
            
            # 分离人员和障碍物检测
            person_detections = [d for d in detections if d.get('class', '') == 'person']
            
            # 检测障碍物
            obstacles = self.obstacle_detector.detect_obstacles(frame, detections)
            
            # 分析人员状态
            persons = self._analyze_persons(person_detections, keypoints_data, current_time)
            
            # 生成场景分析
            scene_analysis = self._generate_scene_analysis(persons, obstacles, frame, current_time)
            
            # 检测跌倒事件
            fall_alerts = self._detect_fall_events(persons, obstacles, current_time)
            
            # 更新历史记录
            self.scene_history.append(scene_analysis)
            self.alert_history.extend(fall_alerts)
            
            return scene_analysis, fall_alerts
            
        except Exception as e:
            self.logger.error(f"场景分析失败: {e}")
            return self._create_default_scene_analysis(), []
    
    def _analyze_persons(self, person_detections: List[Dict], 
                        keypoints_data: Optional[List[List[List[float]]]], 
                        current_time: float) -> List[PersonDetection]:
        """分析人员状态
        
        Args:
            person_detections: 人员检测结果
            keypoints_data: 关键点数据
            current_time: 当前时间
            
        Returns:
            List[PersonDetection]: 人员分析结果
        """
        persons = []
        
        for i, detection in enumerate(person_detections):
            try:
                # 创建人员检测对象
                person = PersonDetection(
                    person_id=detection.get('track_id', i),
                    bbox=detection.get('bbox', [0, 0, 100, 100]),
                    confidence=detection.get('confidence', 0.5),
                    timestamp=current_time
                )
                
                # 添加关键点数据
                if keypoints_data and i < len(keypoints_data):
                    person.keypoints = keypoints_data[i]
                    
                    # 估计姿态状态
                    state, confidence = self.pose_estimator.estimate_pose_state(
                        person.keypoints, person.bbox
                    )
                    person.state = state
                    person.state_confidence = confidence
                    
                    # 计算身体角度
                    person.body_angle = self.pose_estimator._calculate_body_angle(person.keypoints)
                    
                    # 提取头部位置
                    if person.keypoints and len(person.keypoints) > 0:
                        head_points = [kp for kp in person.keypoints[:5] if kp[2] > 0.3]
                        if head_points:
                            person.head_position = (
                                np.mean([p[0] for p in head_points]),
                                np.mean([p[1] for p in head_points])
                            )
                
                # 计算运动信息
                self._calculate_motion_info(person)
                
                # 更新状态持续时间
                self._update_state_duration(person)
                
                persons.append(person)
                
            except Exception as e:
                self.logger.debug(f"人员 {i} 分析失败: {e}")
                continue
        
        return persons
    
    def _calculate_motion_info(self, person: PersonDetection):
        """计算运动信息
        
        Args:
            person: 人员检测对象
        """
        try:
            person_id = person.person_id
            history = self.person_history[person_id]
            
            if len(history) > 0:
                prev_person = history[-1]
                dt = person.timestamp - prev_person.timestamp
                
                if dt > 0:
                    # 计算速度
                    dx = person.center_point[0] - prev_person.center_point[0]
                    dy = person.center_point[1] - prev_person.center_point[1]
                    person.velocity = (dx / dt, dy / dt)
                    
                    # 计算加速度
                    if len(history) > 1:
                        prev_velocity = prev_person.velocity
                        dvx = person.velocity[0] - prev_velocity[0]
                        dvy = person.velocity[1] - prev_velocity[1]
                        person.acceleration = (dvx / dt, dvy / dt)
            
            # 更新历史记录
            history.append(person)
            
        except Exception as e:
            self.logger.debug(f"运动信息计算失败: {e}")
    
    def _update_state_duration(self, person: PersonDetection):
        """更新状态持续时间
        
        Args:
            person: 人员检测对象
        """
        try:
            person_id = person.person_id
            history = self.person_history[person_id]
            
            if len(history) > 0:
                prev_person = history[-1]
                if prev_person.state == person.state:
                    # 状态未改变，累加持续时间
                    person.duration_in_state = prev_person.duration_in_state + (
                        person.timestamp - prev_person.timestamp
                    )
                else:
                    # 状态改变，重置持续时间
                    person.duration_in_state = 0.0
            else:
                person.duration_in_state = 0.0
                
        except Exception as e:
            self.logger.debug(f"状态持续时间更新失败: {e}")
    
    def _generate_scene_analysis(self, persons: List[PersonDetection], 
                               obstacles: List[ObstacleDetection], 
                               frame: np.ndarray, current_time: float) -> SceneAnalysis:
        """生成场景分析
        
        Args:
            persons: 人员列表
            obstacles: 障碍物列表
            frame: 输入帧
            current_time: 当前时间
            
        Returns:
            SceneAnalysis: 场景分析结果
        """
        try:
            # 统计人员状态
            total_persons = len(persons)
            standing_persons = len([p for p in persons if p.state == PersonState.STANDING])
            fallen_persons = len([p for p in persons if p.state == PersonState.FALLEN])
            sitting_persons = len([p for p in persons if p.state == PersonState.SITTING])
            
            # 统计障碍物
            total_obstacles = len(obstacles)
            ground_obstacles = len([o for o in obstacles if o.obstacle_type in [
                ObstacleType.TOY, ObstacleType.CARPET
            ]])
            furniture_count = len([o for o in obstacles if o.obstacle_type in [
                ObstacleType.FURNITURE, ObstacleType.TABLE, ObstacleType.CHAIR, 
                ObstacleType.SOFA, ObstacleType.BOOKSHELF
            ]])
            
            # 评估场景复杂度
            complexity = self._assess_scene_complexity(persons, obstacles)
            
            # 计算整体风险
            overall_risk = self._calculate_overall_risk(persons, obstacles)
            
            # 识别跌倒风险区域
            fall_risk_areas = self._identify_fall_risk_areas(obstacles)
            
            # 分析环境因素
            lighting_quality = self._analyze_lighting_quality(frame)
            visibility_score = self._calculate_visibility_score(frame, persons)
            occlusion_level = self._calculate_occlusion_level(persons, obstacles)
            
            return SceneAnalysis(
                timestamp=current_time,
                complexity=complexity,
                total_persons=total_persons,
                standing_persons=standing_persons,
                fallen_persons=fallen_persons,
                sitting_persons=sitting_persons,
                total_obstacles=total_obstacles,
                ground_obstacles=ground_obstacles,
                furniture_count=furniture_count,
                overall_risk=overall_risk,
                fall_risk_areas=fall_risk_areas,
                lighting_quality=lighting_quality,
                visibility_score=visibility_score,
                occlusion_level=occlusion_level
            )
            
        except Exception as e:
            self.logger.error(f"场景分析生成失败: {e}")
            return self._create_default_scene_analysis()
    
    def _assess_scene_complexity(self, persons: List[PersonDetection], 
                               obstacles: List[ObstacleDetection]) -> SceneComplexity:
        """评估场景复杂度
        
        Args:
            persons: 人员列表
            obstacles: 障碍物列表
            
        Returns:
            SceneComplexity: 场景复杂度
        """
        try:
            complexity_score = 0
            
            # 基于人员数量
            person_count = len(persons)
            if person_count == 0:
                complexity_score += 0
            elif person_count == 1:
                complexity_score += 1
            elif person_count <= 3:
                complexity_score += 2
            else:
                complexity_score += 3
            
            # 基于人员状态多样性
            states = set(p.state for p in persons)
            complexity_score += len(states)
            
            # 基于障碍物数量
            obstacle_count = len(obstacles)
            if obstacle_count <= 2:
                complexity_score += 0
            elif obstacle_count <= 5:
                complexity_score += 1
            elif obstacle_count <= 10:
                complexity_score += 2
            else:
                complexity_score += 3
            
            # 基于跌倒人员
            fallen_count = len([p for p in persons if p.state == PersonState.FALLEN])
            if fallen_count > 0:
                complexity_score += 2
            
            # 基于地面障碍物
            ground_obstacles = len([o for o in obstacles if o.obstacle_type == ObstacleType.TOY])
            complexity_score += min(ground_obstacles, 2)
            
            # 分类复杂度
            if complexity_score <= 2:
                return SceneComplexity.SIMPLE
            elif complexity_score <= 5:
                return SceneComplexity.MODERATE
            elif complexity_score <= 8:
                return SceneComplexity.COMPLEX
            else:
                return SceneComplexity.VERY_COMPLEX
                
        except Exception as e:
            self.logger.debug(f"场景复杂度评估失败: {e}")
            return SceneComplexity.MODERATE
    
    def _calculate_overall_risk(self, persons: List[PersonDetection], 
                              obstacles: List[ObstacleDetection]) -> float:
        """计算整体风险
        
        Args:
            persons: 人员列表
            obstacles: 障碍物列表
            
        Returns:
            float: 整体风险 (0-1)
        """
        try:
            risk_factors = []
            
            # 跌倒人员风险
            fallen_count = len([p for p in persons if p.state == PersonState.FALLEN])
            if fallen_count > 0:
                risk_factors.append(0.9)  # 已有跌倒人员，风险很高
            
            # 障碍物风险
            obstacle_risks = [o.risk_level for o in obstacles]
            if obstacle_risks:
                avg_obstacle_risk = np.mean(obstacle_risks)
                risk_factors.append(avg_obstacle_risk)
            
            # 人员密度风险
            person_count = len(persons)
            if person_count > 3:
                density_risk = min(person_count / 10.0, 0.8)
                risk_factors.append(density_risk)
            
            # 运动风险（快速移动）
            for person in persons:
                speed = np.sqrt(person.velocity[0]**2 + person.velocity[1]**2)
                if speed > 50:  # 像素/秒
                    risk_factors.append(0.6)
            
            # 计算综合风险
            if risk_factors:
                overall_risk = np.mean(risk_factors)
                return np.clip(overall_risk, 0.0, 1.0)
            else:
                return 0.1  # 基础风险
                
        except Exception as e:
            self.logger.debug(f"整体风险计算失败: {e}")
            return 0.3
    
    def _identify_fall_risk_areas(self, obstacles: List[ObstacleDetection]) -> List[Tuple[float, float, float, float]]:
        """识别跌倒风险区域
        
        Args:
            obstacles: 障碍物列表
            
        Returns:
            List[Tuple[float, float, float, float]]: 风险区域列表 [x1, y1, x2, y2]
        """
        try:
            risk_areas = []
            
            # 基于高风险障碍物扩展风险区域
            for obstacle in obstacles:
                if obstacle.is_hazardous:
                    # 扩展障碍物周围区域
                    bbox = obstacle.bbox
                    expansion = 50  # 像素
                    
                    risk_area = (
                        bbox[0] - expansion,
                        bbox[1] - expansion,
                        bbox[2] + expansion,
                        bbox[3] + expansion
                    )
                    risk_areas.append(risk_area)
            
            return risk_areas
            
        except Exception as e:
            self.logger.debug(f"跌倒风险区域识别失败: {e}")
            return []
    
    def _analyze_lighting_quality(self, frame: np.ndarray) -> float:
        """分析光照质量
        
        Args:
            frame: 输入帧
            
        Returns:
            float: 光照质量 (0-1)
        """
        try:
            # 转换为灰度图
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # 计算亮度统计
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)
            
            # 理想亮度范围
            ideal_brightness = 128
            brightness_score = 1.0 - abs(mean_brightness - ideal_brightness) / 128.0
            
            # 对比度评分
            contrast_score = min(std_brightness / 64.0, 1.0)
            
            # 综合光照质量
            lighting_quality = (brightness_score + contrast_score) / 2.0
            return np.clip(lighting_quality, 0.0, 1.0)
            
        except Exception as e:
            self.logger.debug(f"光照质量分析失败: {e}")
            return 0.5
    
    def _calculate_visibility_score(self, frame: np.ndarray, persons: List[PersonDetection]) -> float:
        """计算可见度评分
        
        Args:
            frame: 输入帧
            persons: 人员列表
            
        Returns:
            float: 可见度评分 (0-1)
        """
        try:
            if not persons:
                return 0.5
            
            visibility_scores = []
            
            for person in persons:
                # 基于关键点可见度
                if person.keypoints:
                    visible_keypoints = len([kp for kp in person.keypoints if kp[2] > 0.3])
                    keypoint_visibility = visible_keypoints / len(person.keypoints)
                    visibility_scores.append(keypoint_visibility)
                else:
                    # 基于检测置信度
                    visibility_scores.append(person.confidence)
            
            return np.mean(visibility_scores)
            
        except Exception as e:
            self.logger.debug(f"可见度评分计算失败: {e}")
            return 0.5
    
    def _calculate_occlusion_level(self, persons: List[PersonDetection], 
                                 obstacles: List[ObstacleDetection]) -> float:
        """计算遮挡程度
        
        Args:
            persons: 人员列表
            obstacles: 障碍物列表
            
        Returns:
            float: 遮挡程度 (0-1)
        """
        try:
            if not persons:
                return 0.0
            
            occlusion_scores = []
            
            for person in persons:
                person_bbox = person.bbox
                person_area = (person_bbox[2] - person_bbox[0]) * (person_bbox[3] - person_bbox[1])
                
                occluded_area = 0
                
                # 检查与其他人员的重叠
                for other_person in persons:
                    if other_person.person_id != person.person_id:
                        overlap_area = self._calculate_bbox_overlap(person_bbox, other_person.bbox)
                        occluded_area += overlap_area
                
                # 检查与障碍物的重叠
                for obstacle in obstacles:
                    overlap_area = self._calculate_bbox_overlap(person_bbox, obstacle.bbox)
                    occluded_area += overlap_area
                
                # 计算遮挡比例
                if person_area > 0:
                    occlusion_ratio = min(occluded_area / person_area, 1.0)
                    occlusion_scores.append(occlusion_ratio)
            
            return np.mean(occlusion_scores) if occlusion_scores else 0.0
            
        except Exception as e:
            self.logger.debug(f"遮挡程度计算失败: {e}")
            return 0.0
    
    def _calculate_bbox_overlap(self, bbox1: List[float], bbox2: List[float]) -> float:
        """计算边界框重叠面积
        
        Args:
            bbox1: 边界框1 [x1, y1, x2, y2]
            bbox2: 边界框2 [x1, y1, x2, y2]
            
        Returns:
            float: 重叠面积
        """
        try:
            # 计算交集
            x1 = max(bbox1[0], bbox2[0])
            y1 = max(bbox1[1], bbox2[1])
            x2 = min(bbox1[2], bbox2[2])
            y2 = min(bbox1[3], bbox2[3])
            
            # 检查是否有重叠
            if x1 < x2 and y1 < y2:
                return (x2 - x1) * (y2 - y1)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.debug(f"边界框重叠计算失败: {e}")
            return 0.0
    
    def _detect_fall_events(self, persons: List[PersonDetection], 
                          obstacles: List[ObstacleDetection], 
                          current_time: float) -> List[FallAlert]:
        """检测跌倒事件
        
        Args:
            persons: 人员列表
            obstacles: 障碍物列表
            current_time: 当前时间
            
        Returns:
            List[FallAlert]: 跌倒警报列表
        """
        alerts = []
        
        for person in persons:
            try:
                # 检查是否为跌倒状态
                if person.state == PersonState.FALLEN and person.state_confidence > self.fall_detection_threshold:
                    # 检查跌倒持续时间
                    if person.duration_in_state > self.fall_duration_threshold:
                        alert_level = AlertLevel.CRITICAL
                    elif person.duration_in_state > 1.0:
                        alert_level = AlertLevel.WARNING
                    else:
                        alert_level = AlertLevel.CAUTION
                    
                    # 查找附近的障碍物和人员
                    nearby_obstacles = self._find_nearby_obstacles(person, obstacles)
                    nearby_persons = self._find_nearby_persons(person, persons)
                    
                    # 评估冲击严重程度
                    impact_severity = self._assess_impact_severity(person)
                    
                    # 生成建议行动
                    suggested_actions = self._generate_suggested_actions(person, nearby_obstacles, nearby_persons)
                    
                    # 创建警报
                    alert = FallAlert(
                        alert_id=f"fall_{person.person_id}_{int(current_time)}",
                        person_id=person.person_id,
                        alert_level=alert_level,
                        timestamp=current_time,
                        fall_location=person.center_point,
                        fall_confidence=person.state_confidence,
                        fall_duration=person.duration_in_state,
                        impact_severity=impact_severity,
                        nearby_obstacles=[o.obstacle_id for o in nearby_obstacles],
                        other_persons_nearby=[p.person_id for p in nearby_persons],
                        requires_immediate_attention=alert_level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY],
                        suggested_actions=suggested_actions
                    )
                    
                    alerts.append(alert)
                    
            except Exception as e:
                self.logger.error(f"跌倒事件检测失败 (人员 {person.person_id}): {e}")
                continue
        
        return alerts
    
    def _find_nearby_obstacles(self, person: PersonDetection, 
                             obstacles: List[ObstacleDetection]) -> List[ObstacleDetection]:
        """查找附近的障碍物
        
        Args:
            person: 人员检测对象
            obstacles: 障碍物列表
            
        Returns:
            List[ObstacleDetection]: 附近的障碍物
        """
        nearby = []
        person_center = person.center_point
        
        for obstacle in obstacles:
            obstacle_center = obstacle.center_point
            distance = np.sqrt(
                (person_center[0] - obstacle_center[0])**2 + 
                (person_center[1] - obstacle_center[1])**2
            )
            
            if distance < self.proximity_threshold:
                nearby.append(obstacle)
        
        return nearby
    
    def _find_nearby_persons(self, target_person: PersonDetection, 
                           persons: List[PersonDetection]) -> List[PersonDetection]:
        """查找附近的其他人员
        
        Args:
            target_person: 目标人员
            persons: 人员列表
            
        Returns:
            List[PersonDetection]: 附近的其他人员
        """
        nearby = []
        target_center = target_person.center_point
        
        for person in persons:
            if person.person_id != target_person.person_id:
                person_center = person.center_point
                distance = np.sqrt(
                    (target_center[0] - person_center[0])**2 + 
                    (target_center[1] - person_center[1])**2
                )
                
                if distance < self.proximity_threshold * 2:  # 更大的范围
                    nearby.append(person)
        
        return nearby
    
    def _assess_impact_severity(self, person: PersonDetection) -> float:
        """评估冲击严重程度
        
        Args:
            person: 人员检测对象
            
        Returns:
            float: 冲击严重程度 (0-1)
        """
        try:
            severity = 0.0
            
            # 基于加速度
            acceleration_magnitude = np.sqrt(
                person.acceleration[0]**2 + person.acceleration[1]**2
            )
            if acceleration_magnitude > 100:  # 像素/秒²
                severity += 0.4
            
            # 基于身体角度
            if person.body_angle < 20:  # 非常水平
                severity += 0.3
            
            # 基于状态置信度
            severity += person.state_confidence * 0.3
            
            return np.clip(severity, 0.0, 1.0)
            
        except Exception as e:
            self.logger.debug(f"冲击严重程度评估失败: {e}")
            return 0.5
    
    def _generate_suggested_actions(self, person: PersonDetection, 
                                  nearby_obstacles: List[ObstacleDetection], 
                                  nearby_persons: List[PersonDetection]) -> List[str]:
        """生成建议行动
        
        Args:
            person: 跌倒人员
            nearby_obstacles: 附近障碍物
            nearby_persons: 附近人员
            
        Returns:
            List[str]: 建议行动列表
        """
        actions = []
        
        # 基础建议
        actions.append("立即检查跌倒人员状况")
        
        # 基于持续时间的建议
        if person.duration_in_state > 5.0:
            actions.append("考虑呼叫医疗援助")
        
        # 基于附近障碍物的建议
        if nearby_obstacles:
            hazardous_obstacles = [o for o in nearby_obstacles if o.is_hazardous]
            if hazardous_obstacles:
                actions.append("清理周围危险物品")
        
        # 基于附近人员的建议
        if nearby_persons:
            standing_persons = [p for p in nearby_persons if p.state == PersonState.STANDING]
            if standing_persons:
                actions.append("请求附近人员协助")
        
        # 基于冲击严重程度的建议
        impact_severity = self._assess_impact_severity(person)
        if impact_severity > 0.7:
            actions.append("高冲击跌倒，优先医疗评估")
        
        return actions
    
    def _create_default_scene_analysis(self) -> SceneAnalysis:
        """创建默认场景分析
        
        Returns:
            SceneAnalysis: 默认场景分析
        """
        return SceneAnalysis(
            timestamp=time.time(),
            complexity=SceneComplexity.SIMPLE,
            total_persons=0,
            standing_persons=0,
            fallen_persons=0,
            sitting_persons=0,
            total_obstacles=0,
            ground_obstacles=0,
            furniture_count=0,
            overall_risk=0.1,
            fall_risk_areas=[],
            lighting_quality=0.5,
            visibility_score=0.5,
            occlusion_level=0.0
        )
    
    def get_scene_statistics(self) -> Dict[str, Any]:
        """获取场景统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        try:
            if not self.scene_history:
                return {}
            
            recent_scenes = list(self.scene_history)[-10:]  # 最近10个场景
            
            stats = {
                'total_scenes_analyzed': len(self.scene_history),
                'recent_complexity_distribution': {},
                'average_risk_level': 0.0,
                'total_alerts_generated': len(self.alert_history),
                'recent_alert_levels': {},
                'average_person_count': 0.0,
                'average_obstacle_count': 0.0
            }
            
            # 复杂度分布
            complexity_counts = {}
            for scene in recent_scenes:
                complexity = scene.complexity.value
                complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
            stats['recent_complexity_distribution'] = complexity_counts
            
            # 平均风险等级
            risk_levels = [scene.overall_risk for scene in recent_scenes]
            stats['average_risk_level'] = np.mean(risk_levels) if risk_levels else 0.0
            
            # 警报等级分布
            recent_alerts = list(self.alert_history)[-20:]  # 最近20个警报
            alert_counts = {}
            for alert in recent_alerts:
                level = alert.alert_level.value
                alert_counts[level] = alert_counts.get(level, 0) + 1
            stats['recent_alert_levels'] = alert_counts
            
            # 平均人员和障碍物数量
            person_counts = [scene.total_persons for scene in recent_scenes]
            obstacle_counts = [scene.total_obstacles for scene in recent_scenes]
            stats['average_person_count'] = np.mean(person_counts) if person_counts else 0.0
            stats['average_obstacle_count'] = np.mean(obstacle_counts) if obstacle_counts else 0.0
            
            return stats
            
        except Exception as e:
            self.logger.error(f"场景统计信息获取失败: {e}")
            return {}
    
    def export_analysis_report(self, filepath: str) -> bool:
        """导出分析报告
        
        Args:
            filepath: 报告文件路径
            
        Returns:
            bool: 是否成功导出
        """
        try:
            report = {
                'timestamp': time.time(),
                'analysis_summary': self.get_scene_statistics(),
                'recent_scenes': [{
                    'timestamp': scene.timestamp,
                    'complexity': scene.complexity.value,
                    'total_persons': scene.total_persons,
                    'fallen_persons': scene.fallen_persons,
                    'total_obstacles': scene.total_obstacles,
                    'overall_risk': scene.overall_risk,
                    'lighting_quality': scene.lighting_quality,
                    'visibility_score': scene.visibility_score,
                    'occlusion_level': scene.occlusion_level
                } for scene in list(self.scene_history)[-50:]],
                'recent_alerts': [{
                    'alert_id': alert.alert_id,
                    'person_id': alert.person_id,
                    'alert_level': alert.alert_level.value,
                    'timestamp': alert.timestamp,
                    'fall_confidence': alert.fall_confidence,
                    'fall_duration': alert.fall_duration,
                    'impact_severity': alert.impact_severity,
                    'suggested_actions': alert.suggested_actions
                } for alert in list(self.alert_history)[-30:]]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"分析报告已导出到: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"分析报告导出失败: {e}")
            return False

# 测试代码
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("复杂场景分析器测试")
    print("=" * 50)
    
    # 创建分析器
    analyzer = ComplexSceneAnalyzer()
    
    # 模拟检测数据
    mock_detections = [
        {
            'class': 'person',
            'bbox': [100, 50, 200, 300],
            'confidence': 0.9,
            'track_id': 1
        },
        {
            'class': 'person', 
            'bbox': [300, 200, 400, 450],
            'confidence': 0.8,
            'track_id': 2
        },
        {
            'class': 'chair',
            'bbox': [150, 100, 250, 200],
            'confidence': 0.7
        },
        {
            'class': 'toy',
            'bbox': [80, 280, 120, 320],
            'confidence': 0.6
        }
    ]
    
    # 模拟关键点数据
    mock_keypoints = [
        # 人员1 - 站立状态
        [[150, 60, 0.9], [145, 55, 0.8], [155, 55, 0.8], [140, 58, 0.7], [160, 58, 0.7],
         [130, 100, 0.9], [170, 100, 0.9], [120, 140, 0.8], [180, 140, 0.8], [110, 180, 0.7], [190, 180, 0.7],
         [140, 180, 0.9], [160, 180, 0.9], [135, 220, 0.8], [165, 220, 0.8], [130, 290, 0.9], [170, 290, 0.9]],
        # 人员2 - 跌倒状态
        [[350, 250, 0.8], [345, 245, 0.7], [355, 245, 0.7], [340, 248, 0.6], [360, 248, 0.6],
         [320, 260, 0.8], [380, 260, 0.8], [300, 270, 0.7], [400, 270, 0.7], [280, 280, 0.6], [420, 280, 0.6],
         [330, 280, 0.8], [370, 280, 0.8], [325, 300, 0.7], [375, 300, 0.7], [320, 320, 0.8], [380, 320, 0.8]]
    ]
    
    # 创建模拟帧
    mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("\n1. 执行场景分析...")
    scene_analysis, fall_alerts = analyzer.analyze_scene(mock_frame, mock_detections, mock_keypoints)
    
    print(f"场景复杂度: {scene_analysis.complexity.value}")
    print(f"总人数: {scene_analysis.total_persons}")
    print(f"站立人数: {scene_analysis.standing_persons}")
    print(f"跌倒人数: {scene_analysis.fallen_persons}")
    print(f"总障碍物数: {scene_analysis.total_obstacles}")
    print(f"地面障碍物数: {scene_analysis.ground_obstacles}")
    print(f"整体风险: {scene_analysis.overall_risk:.2f}")
    print(f"光照质量: {scene_analysis.lighting_quality:.2f}")
    print(f"可见度评分: {scene_analysis.visibility_score:.2f}")
    print(f"遮挡程度: {scene_analysis.occlusion_level:.2f}")
    
    print(f"\n2. 跌倒警报数量: {len(fall_alerts)}")
    for alert in fall_alerts:
        print(f"  警报ID: {alert.alert_id}")
        print(f"  人员ID: {alert.person_id}")
        print(f"  警报级别: {alert.alert_level.value}")
        print(f"  跌倒置信度: {alert.fall_confidence:.2f}")
        print(f"  跌倒持续时间: {alert.fall_duration:.1f}秒")
        print(f"  冲击严重程度: {alert.impact_severity:.2f}")
        print(f"  建议行动: {', '.join(alert.suggested_actions)}")
        print()
    
    # 模拟多次分析以测试历史记录
    print("3. 执行多次分析以测试历史记录...")
    for i in range(5):
        # 稍微修改检测数据
        modified_detections = mock_detections.copy()
        if modified_detections:
            modified_detections[0]['bbox'][0] += i * 10  # 移动第一个人员
        
        analyzer.analyze_scene(mock_frame, modified_detections, mock_keypoints)
        time.sleep(0.1)
    
    print("\n4. 获取场景统计信息...")
    stats = analyzer.get_scene_statistics()
    print(f"已分析场景总数: {stats.get('total_scenes_analyzed', 0)}")
    print(f"平均风险等级: {stats.get('average_risk_level', 0):.2f}")
    print(f"生成警报总数: {stats.get('total_alerts_generated', 0)}")
    print(f"平均人员数量: {stats.get('average_person_count', 0):.1f}")
    print(f"平均障碍物数量: {stats.get('average_obstacle_count', 0):.1f}")
    
    print("\n5. 导出分析报告...")
    report_path = "complex_scene_analysis_report.json"
    if analyzer.export_analysis_report(report_path):
        print(f"报告已导出到: {report_path}")
    
    print("\n复杂场景分析器测试完成！")
    print("\n主要功能:")
    print("- 多人混合状态检测（站立、坐着、跌倒等）")
    print("- 复杂障碍物识别（玩具、家具、书柜等）")
    print("- 实时跌倒检测和警报生成")
    print("- 场景复杂度评估")
    print("- 风险区域识别")
    print("- 环境因素分析（光照、可见度、遮挡）")
    print("- 历史数据跟踪和统计分析")
    print("- 智能建议行动生成")