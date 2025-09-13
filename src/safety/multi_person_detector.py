#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多人混合状态检测算法
专门处理站立与跌倒人员同时存在的复杂场景

主要功能:
- 多人同时检测和状态分类
- 混合状态场景分析
- 人员间关系推理
- 群体行为分析
- 优先级排序和警报管理

作者: AI Assistant
日期: 2024
"""

import cv2
import numpy as np
import logging
import time
import json
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonState(Enum):
    """人员状态枚举"""
    STANDING = "standing"
    SITTING = "sitting"
    LYING = "lying"
    FALLEN = "fallen"
    CROUCHING = "crouching"
    WALKING = "walking"
    RUNNING = "running"
    UNKNOWN = "unknown"

class AlertPriority(Enum):
    """警报优先级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class GroupBehavior(Enum):
    """群体行为类型"""
    NORMAL = "normal"
    GATHERING = "gathering"
    DISPERSING = "dispersing"
    HELPING = "helping"
    PANIC = "panic"
    EMERGENCY_RESPONSE = "emergency_response"

class InteractionType(Enum):
    """人员交互类型"""
    NONE = "none"
    APPROACHING = "approaching"
    HELPING = "helping"
    AVOIDING = "avoiding"
    FOLLOWING = "following"
    ASSISTING = "assisting"

@dataclass
class PersonDetection:
    """人员检测结果"""
    person_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center_point: Tuple[float, float]
    keypoints: List[List[float]]  # [[x, y, confidence], ...]
    state: PersonState
    state_confidence: float
    body_angle: float  # 身体角度（度）
    height_ratio: float  # 身高比例
    velocity: Tuple[float, float]  # 速度向量
    acceleration: Tuple[float, float]  # 加速度向量
    duration_in_state: float  # 在当前状态的持续时间
    previous_states: List[PersonState] = field(default_factory=list)
    tracking_confidence: float = 0.0
    is_occluded: bool = False
    occlusion_ratio: float = 0.0

@dataclass
class PersonInteraction:
    """人员交互信息"""
    person1_id: int
    person2_id: int
    interaction_type: InteractionType
    distance: float
    relative_position: Tuple[float, float]
    interaction_confidence: float
    duration: float
    is_helping: bool = False

@dataclass
class GroupAnalysis:
    """群体分析结果"""
    group_id: int
    member_ids: List[int]
    group_center: Tuple[float, float]
    group_behavior: GroupBehavior
    behavior_confidence: float
    group_size: int
    standing_count: int
    fallen_count: int
    helping_count: int
    group_cohesion: float  # 群体凝聚度
    emergency_level: float  # 紧急程度

@dataclass
class MultiPersonAlert:
    """多人场景警报"""
    alert_id: str
    timestamp: float
    alert_priority: AlertPriority
    affected_persons: List[int]
    fallen_persons: List[int]
    helping_persons: List[int]
    bystander_persons: List[int]
    alert_message: str
    suggested_actions: List[str]
    requires_immediate_attention: bool
    estimated_response_time: float
    group_analysis: Optional[GroupAnalysis] = None

class PoseAnalyzer:
    """姿态分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PoseAnalyzer")
        
        # COCO关键点索引
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # 关键点连接关系
        self.skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]
        ]
        
        # 状态判断阈值
        self.angle_thresholds = {
            PersonState.STANDING: (70, 110),  # 身体角度范围
            PersonState.SITTING: (45, 90),
            PersonState.LYING: (0, 30),
            PersonState.FALLEN: (0, 25),
            PersonState.CROUCHING: (30, 60)
        }
        
        self.height_thresholds = {
            PersonState.STANDING: (0.7, 1.0),  # 相对高度比例
            PersonState.SITTING: (0.4, 0.7),
            PersonState.LYING: (0.1, 0.4),
            PersonState.FALLEN: (0.1, 0.3),
            PersonState.CROUCHING: (0.3, 0.6)
        }
    
    def analyze_pose(self, keypoints: List[List[float]], bbox: Tuple[int, int, int, int]) -> Tuple[PersonState, float, float, float]:
        """分析人员姿态
        
        Args:
            keypoints: 关键点列表 [[x, y, confidence], ...]
            bbox: 边界框 (x1, y1, x2, y2)
            
        Returns:
            Tuple[PersonState, float, float, float]: (状态, 置信度, 身体角度, 高度比例)
        """
        try:
            if not keypoints or len(keypoints) < 17:
                return PersonState.UNKNOWN, 0.0, 0.0, 0.0
            
            # 计算身体角度
            body_angle = self._calculate_body_angle(keypoints)
            
            # 计算高度比例
            height_ratio = self._calculate_height_ratio(keypoints, bbox)
            
            # 计算关键点可见性
            visibility_score = self._calculate_visibility_score(keypoints)
            
            # 状态分类
            state, confidence = self._classify_state(body_angle, height_ratio, keypoints, visibility_score)
            
            return state, confidence, body_angle, height_ratio
            
        except Exception as e:
            self.logger.error(f"姿态分析失败: {e}")
            return PersonState.UNKNOWN, 0.0, 0.0, 0.0
    
    def _calculate_body_angle(self, keypoints: List[List[float]]) -> float:
        """计算身体角度
        
        Args:
            keypoints: 关键点列表
            
        Returns:
            float: 身体角度（度）
        """
        try:
            # 使用肩膀和髋部计算身体主轴
            left_shoulder = keypoints[5]  # left_shoulder
            right_shoulder = keypoints[6]  # right_shoulder
            left_hip = keypoints[11]  # left_hip
            right_hip = keypoints[12]  # right_hip
            
            # 检查关键点可见性
            if (left_shoulder[2] < 0.3 or right_shoulder[2] < 0.3 or 
                left_hip[2] < 0.3 or right_hip[2] < 0.3):
                return 90.0  # 默认站立角度
            
            # 计算肩膀和髋部中心点
            shoulder_center = (
                (left_shoulder[0] + right_shoulder[0]) / 2,
                (left_shoulder[1] + right_shoulder[1]) / 2
            )
            hip_center = (
                (left_hip[0] + right_hip[0]) / 2,
                (left_hip[1] + right_hip[1]) / 2
            )
            
            # 计算身体主轴向量
            dx = hip_center[0] - shoulder_center[0]
            dy = hip_center[1] - shoulder_center[1]
            
            # 计算与垂直方向的夹角
            angle = np.degrees(np.arctan2(abs(dx), abs(dy)))
            
            return angle
            
        except Exception as e:
            self.logger.debug(f"身体角度计算失败: {e}")
            return 90.0
    
    def _calculate_height_ratio(self, keypoints: List[List[float]], bbox: Tuple[int, int, int, int]) -> float:
        """计算高度比例
        
        Args:
            keypoints: 关键点列表
            bbox: 边界框
            
        Returns:
            float: 高度比例
        """
        try:
            # 获取头部和脚部关键点
            nose = keypoints[0]  # nose
            left_ankle = keypoints[15]  # left_ankle
            right_ankle = keypoints[16]  # right_ankle
            
            # 计算实际身体高度
            if nose[2] > 0.3 and (left_ankle[2] > 0.3 or right_ankle[2] > 0.3):
                if left_ankle[2] > right_ankle[2]:
                    body_height = abs(nose[1] - left_ankle[1])
                else:
                    body_height = abs(nose[1] - right_ankle[1])
            else:
                # 使用边界框高度作为备选
                body_height = bbox[3] - bbox[1]
            
            # 计算边界框高度
            bbox_height = bbox[3] - bbox[1]
            
            # 计算高度比例
            if bbox_height > 0:
                height_ratio = body_height / bbox_height
            else:
                height_ratio = 0.5
            
            return np.clip(height_ratio, 0.0, 1.0)
            
        except Exception as e:
            self.logger.debug(f"高度比例计算失败: {e}")
            return 0.5
    
    def _calculate_visibility_score(self, keypoints: List[List[float]]) -> float:
        """计算关键点可见性评分
        
        Args:
            keypoints: 关键点列表
            
        Returns:
            float: 可见性评分 (0-1)
        """
        try:
            visible_count = sum(1 for kp in keypoints if kp[2] > 0.3)
            total_count = len(keypoints)
            
            return visible_count / total_count if total_count > 0 else 0.0
            
        except Exception as e:
            self.logger.debug(f"可见性评分计算失败: {e}")
            return 0.0
    
    def _classify_state(self, body_angle: float, height_ratio: float, 
                       keypoints: List[List[float]], visibility_score: float) -> Tuple[PersonState, float]:
        """分类人员状态
        
        Args:
            body_angle: 身体角度
            height_ratio: 高度比例
            keypoints: 关键点列表
            visibility_score: 可见性评分
            
        Returns:
            Tuple[PersonState, float]: (状态, 置信度)
        """
        try:
            state_scores = {}
            
            # 基于角度和高度的评分
            for state, (min_angle, max_angle) in self.angle_thresholds.items():
                angle_score = 0.0
                if min_angle <= body_angle <= max_angle:
                    # 在范围内，计算距离中心的评分
                    center_angle = (min_angle + max_angle) / 2
                    angle_distance = abs(body_angle - center_angle)
                    max_distance = (max_angle - min_angle) / 2
                    angle_score = 1.0 - (angle_distance / max_distance)
                
                # 高度评分
                min_height, max_height = self.height_thresholds[state]
                height_score = 0.0
                if min_height <= height_ratio <= max_height:
                    center_height = (min_height + max_height) / 2
                    height_distance = abs(height_ratio - center_height)
                    max_distance = (max_height - min_height) / 2
                    height_score = 1.0 - (height_distance / max_distance)
                
                # 综合评分
                combined_score = (angle_score * 0.6 + height_score * 0.4) * visibility_score
                state_scores[state] = combined_score
            
            # 特殊情况检测
            # 跌倒检测：低角度 + 低高度 + 快速变化
            if body_angle < 30 and height_ratio < 0.4:
                state_scores[PersonState.FALLEN] += 0.3
            
            # 蹲下检测：中等角度 + 中等高度
            if 30 <= body_angle <= 60 and 0.3 <= height_ratio <= 0.6:
                state_scores[PersonState.CROUCHING] += 0.2
            
            # 找到最高评分的状态
            if state_scores:
                best_state = max(state_scores.items(), key=lambda x: x[1])
                return best_state[0], best_state[1]
            else:
                return PersonState.UNKNOWN, 0.0
                
        except Exception as e:
            self.logger.error(f"状态分类失败: {e}")
            return PersonState.UNKNOWN, 0.0

class InteractionAnalyzer:
    """人员交互分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.InteractionAnalyzer")
        self.interaction_history = defaultdict(list)
        self.proximity_threshold = 150.0  # 像素距离
        self.helping_threshold = 100.0  # 帮助行为距离阈值
    
    def analyze_interactions(self, persons: List[PersonDetection]) -> List[PersonInteraction]:
        """分析人员间交互
        
        Args:
            persons: 人员检测列表
            
        Returns:
            List[PersonInteraction]: 交互列表
        """
        interactions = []
        
        try:
            # 两两分析人员交互
            for i, person1 in enumerate(persons):
                for j, person2 in enumerate(persons[i+1:], i+1):
                    interaction = self._analyze_pair_interaction(person1, person2)
                    if interaction:
                        interactions.append(interaction)
                        
                        # 更新交互历史
                        pair_key = f"{min(person1.person_id, person2.person_id)}_{max(person1.person_id, person2.person_id)}"
                        self.interaction_history[pair_key].append(interaction)
                        
                        # 保持历史记录长度
                        if len(self.interaction_history[pair_key]) > 10:
                            self.interaction_history[pair_key].pop(0)
            
            return interactions
            
        except Exception as e:
            self.logger.error(f"交互分析失败: {e}")
            return []
    
    def _analyze_pair_interaction(self, person1: PersonDetection, person2: PersonDetection) -> Optional[PersonInteraction]:
        """分析两人间的交互
        
        Args:
            person1: 人员1
            person2: 人员2
            
        Returns:
            Optional[PersonInteraction]: 交互信息
        """
        try:
            # 计算距离
            distance = np.sqrt(
                (person1.center_point[0] - person2.center_point[0])**2 + 
                (person1.center_point[1] - person2.center_point[1])**2
            )
            
            # 如果距离太远，不考虑交互
            if distance > self.proximity_threshold:
                return None
            
            # 计算相对位置
            relative_pos = (
                person2.center_point[0] - person1.center_point[0],
                person2.center_point[1] - person1.center_point[1]
            )
            
            # 分析交互类型
            interaction_type, confidence, is_helping = self._classify_interaction(
                person1, person2, distance, relative_pos
            )
            
            # 获取历史持续时间
            pair_key = f"{min(person1.person_id, person2.person_id)}_{max(person1.person_id, person2.person_id)}"
            duration = self._calculate_interaction_duration(pair_key, interaction_type)
            
            return PersonInteraction(
                person1_id=person1.person_id,
                person2_id=person2.person_id,
                interaction_type=interaction_type,
                distance=distance,
                relative_position=relative_pos,
                interaction_confidence=confidence,
                duration=duration,
                is_helping=is_helping
            )
            
        except Exception as e:
            self.logger.debug(f"配对交互分析失败: {e}")
            return None
    
    def _classify_interaction(self, person1: PersonDetection, person2: PersonDetection, 
                            distance: float, relative_pos: Tuple[float, float]) -> Tuple[InteractionType, float, bool]:
        """分类交互类型
        
        Args:
            person1: 人员1
            person2: 人员2
            distance: 距离
            relative_pos: 相对位置
            
        Returns:
            Tuple[InteractionType, float, bool]: (交互类型, 置信度, 是否帮助)
        """
        try:
            # 帮助行为检测
            if (person1.state == PersonState.FALLEN and person2.state == PersonState.STANDING and 
                distance < self.helping_threshold):
                return InteractionType.HELPING, 0.9, True
            
            if (person2.state == PersonState.FALLEN and person1.state == PersonState.STANDING and 
                distance < self.helping_threshold):
                return InteractionType.HELPING, 0.9, True
            
            # 协助行为检测（蹲下帮助）
            if (person1.state == PersonState.FALLEN and person2.state == PersonState.CROUCHING and 
                distance < self.helping_threshold):
                return InteractionType.ASSISTING, 0.8, True
            
            if (person2.state == PersonState.FALLEN and person1.state == PersonState.CROUCHING and 
                distance < self.helping_threshold):
                return InteractionType.ASSISTING, 0.8, True
            
            # 接近行为检测
            if distance < 80:
                # 基于速度判断是否在接近
                person1_speed = np.sqrt(person1.velocity[0]**2 + person1.velocity[1]**2)
                person2_speed = np.sqrt(person2.velocity[0]**2 + person2.velocity[1]**2)
                
                if person1_speed > 5 or person2_speed > 5:  # 有移动
                    return InteractionType.APPROACHING, 0.7, False
                else:
                    return InteractionType.NONE, 0.5, False
            
            # 跟随行为检测
            if self._detect_following_behavior(person1, person2):
                return InteractionType.FOLLOWING, 0.6, False
            
            # 避让行为检测
            if self._detect_avoiding_behavior(person1, person2, relative_pos):
                return InteractionType.AVOIDING, 0.6, False
            
            return InteractionType.NONE, 0.3, False
            
        except Exception as e:
            self.logger.debug(f"交互分类失败: {e}")
            return InteractionType.NONE, 0.0, False
    
    def _detect_following_behavior(self, person1: PersonDetection, person2: PersonDetection) -> bool:
        """检测跟随行为
        
        Args:
            person1: 人员1
            person2: 人员2
            
        Returns:
            bool: 是否存在跟随行为
        """
        try:
            # 简单的跟随检测：速度方向相似且一个在另一个后面
            v1_angle = np.arctan2(person1.velocity[1], person1.velocity[0])
            v2_angle = np.arctan2(person2.velocity[1], person2.velocity[0])
            
            angle_diff = abs(v1_angle - v2_angle)
            if angle_diff > np.pi:
                angle_diff = 2 * np.pi - angle_diff
            
            # 角度相似且有一定速度
            if (angle_diff < np.pi/4 and 
                (np.sqrt(person1.velocity[0]**2 + person1.velocity[1]**2) > 3 or
                 np.sqrt(person2.velocity[0]**2 + person2.velocity[1]**2) > 3)):
                return True
            
            return False
            
        except Exception as e:
            self.logger.debug(f"跟随行为检测失败: {e}")
            return False
    
    def _detect_avoiding_behavior(self, person1: PersonDetection, person2: PersonDetection, 
                                relative_pos: Tuple[float, float]) -> bool:
        """检测避让行为
        
        Args:
            person1: 人员1
            person2: 人员2
            relative_pos: 相对位置
            
        Returns:
            bool: 是否存在避让行为
        """
        try:
            # 检测是否有人在远离另一个人
            # 速度方向与相对位置方向相反
            rel_angle = np.arctan2(relative_pos[1], relative_pos[0])
            
            v1_angle = np.arctan2(person1.velocity[1], person1.velocity[0])
            v2_angle = np.arctan2(person2.velocity[1], person2.velocity[0])
            
            # 检查person1是否在远离person2
            angle_diff1 = abs(v1_angle - rel_angle)
            if angle_diff1 > np.pi:
                angle_diff1 = 2 * np.pi - angle_diff1
            
            # 检查person2是否在远离person1
            angle_diff2 = abs(v2_angle - (rel_angle + np.pi))
            if angle_diff2 > np.pi:
                angle_diff2 = 2 * np.pi - angle_diff2
            
            # 如果有人在远离且有一定速度
            if ((angle_diff1 < np.pi/3 and np.sqrt(person1.velocity[0]**2 + person1.velocity[1]**2) > 3) or
                (angle_diff2 < np.pi/3 and np.sqrt(person2.velocity[0]**2 + person2.velocity[1]**2) > 3)):
                return True
            
            return False
            
        except Exception as e:
            self.logger.debug(f"避让行为检测失败: {e}")
            return False
    
    def _calculate_interaction_duration(self, pair_key: str, interaction_type: InteractionType) -> float:
        """计算交互持续时间
        
        Args:
            pair_key: 人员对键值
            interaction_type: 交互类型
            
        Returns:
            float: 持续时间（秒）
        """
        try:
            if pair_key not in self.interaction_history:
                return 0.0
            
            history = self.interaction_history[pair_key]
            if not history:
                return 0.0
            
            # 计算连续相同交互类型的持续时间
            duration = 0.1  # 基础时间间隔
            for interaction in reversed(history):
                if interaction.interaction_type == interaction_type:
                    duration += 0.1
                else:
                    break
            
            return duration
            
        except Exception as e:
            self.logger.debug(f"交互持续时间计算失败: {e}")
            return 0.0

class GroupAnalyzer:
    """群体分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.GroupAnalyzer")
        self.group_distance_threshold = 200.0  # 群体距离阈值
        self.min_group_size = 2
        self.group_history = {}
    
    def analyze_groups(self, persons: List[PersonDetection], 
                      interactions: List[PersonInteraction]) -> List[GroupAnalysis]:
        """分析群体行为
        
        Args:
            persons: 人员列表
            interactions: 交互列表
            
        Returns:
            List[GroupAnalysis]: 群体分析结果
        """
        try:
            # 形成群体
            groups = self._form_groups(persons, interactions)
            
            # 分析每个群体
            group_analyses = []
            for group_id, group_members in groups.items():
                if len(group_members) >= self.min_group_size:
                    analysis = self._analyze_group_behavior(group_id, group_members, persons, interactions)
                    if analysis:
                        group_analyses.append(analysis)
            
            return group_analyses
            
        except Exception as e:
            self.logger.error(f"群体分析失败: {e}")
            return []
    
    def _form_groups(self, persons: List[PersonDetection], 
                    interactions: List[PersonInteraction]) -> Dict[int, List[int]]:
        """形成群体
        
        Args:
            persons: 人员列表
            interactions: 交互列表
            
        Returns:
            Dict[int, List[int]]: 群体字典 {group_id: [person_ids]}
        """
        try:
            # 使用并查集算法形成群体
            person_ids = [p.person_id for p in persons]
            parent = {pid: pid for pid in person_ids}
            
            def find(x):
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]
            
            def union(x, y):
                px, py = find(x), find(y)
                if px != py:
                    parent[px] = py
            
            # 基于交互连接人员
            for interaction in interactions:
                if (interaction.interaction_type in [InteractionType.HELPING, InteractionType.ASSISTING, 
                                                   InteractionType.APPROACHING] or
                    interaction.distance < self.group_distance_threshold):
                    union(interaction.person1_id, interaction.person2_id)
            
            # 基于距离连接人员
            for i, person1 in enumerate(persons):
                for person2 in persons[i+1:]:
                    distance = np.sqrt(
                        (person1.center_point[0] - person2.center_point[0])**2 + 
                        (person1.center_point[1] - person2.center_point[1])**2
                    )
                    if distance < self.group_distance_threshold:
                        union(person1.person_id, person2.person_id)
            
            # 收集群体
            groups = defaultdict(list)
            for pid in person_ids:
                root = find(pid)
                groups[root].append(pid)
            
            # 重新编号群体
            final_groups = {}
            for i, (root, members) in enumerate(groups.items()):
                final_groups[i] = members
            
            return final_groups
            
        except Exception as e:
            self.logger.error(f"群体形成失败: {e}")
            return {}
    
    def _analyze_group_behavior(self, group_id: int, member_ids: List[int], 
                              persons: List[PersonDetection], 
                              interactions: List[PersonInteraction]) -> Optional[GroupAnalysis]:
        """分析群体行为
        
        Args:
            group_id: 群体ID
            member_ids: 成员ID列表
            persons: 人员列表
            interactions: 交互列表
            
        Returns:
            Optional[GroupAnalysis]: 群体分析结果
        """
        try:
            # 获取群体成员
            group_persons = [p for p in persons if p.person_id in member_ids]
            if not group_persons:
                return None
            
            # 计算群体中心
            center_x = np.mean([p.center_point[0] for p in group_persons])
            center_y = np.mean([p.center_point[1] for p in group_persons])
            group_center = (center_x, center_y)
            
            # 统计状态
            standing_count = sum(1 for p in group_persons if p.state == PersonState.STANDING)
            fallen_count = sum(1 for p in group_persons if p.state == PersonState.FALLEN)
            
            # 统计帮助行为
            helping_count = 0
            for interaction in interactions:
                if (interaction.person1_id in member_ids and interaction.person2_id in member_ids and
                    interaction.is_helping):
                    helping_count += 1
            
            # 计算群体凝聚度
            cohesion = self._calculate_group_cohesion(group_persons)
            
            # 分析群体行为
            behavior, behavior_confidence = self._classify_group_behavior(
                group_persons, interactions, helping_count, fallen_count
            )
            
            # 计算紧急程度
            emergency_level = self._calculate_emergency_level(
                fallen_count, helping_count, len(group_persons), behavior
            )
            
            return GroupAnalysis(
                group_id=group_id,
                member_ids=member_ids,
                group_center=group_center,
                group_behavior=behavior,
                behavior_confidence=behavior_confidence,
                group_size=len(group_persons),
                standing_count=standing_count,
                fallen_count=fallen_count,
                helping_count=helping_count,
                group_cohesion=cohesion,
                emergency_level=emergency_level
            )
            
        except Exception as e:
            self.logger.error(f"群体行为分析失败: {e}")
            return None
    
    def _calculate_group_cohesion(self, group_persons: List[PersonDetection]) -> float:
        """计算群体凝聚度
        
        Args:
            group_persons: 群体成员列表
            
        Returns:
            float: 凝聚度 (0-1)
        """
        try:
            if len(group_persons) < 2:
                return 1.0
            
            # 计算成员间的平均距离
            total_distance = 0.0
            pair_count = 0
            
            for i, person1 in enumerate(group_persons):
                for person2 in group_persons[i+1:]:
                    distance = np.sqrt(
                        (person1.center_point[0] - person2.center_point[0])**2 + 
                        (person1.center_point[1] - person2.center_point[1])**2
                    )
                    total_distance += distance
                    pair_count += 1
            
            if pair_count == 0:
                return 1.0
            
            avg_distance = total_distance / pair_count
            
            # 距离越小，凝聚度越高
            cohesion = max(0.0, 1.0 - (avg_distance / self.group_distance_threshold))
            
            return cohesion
            
        except Exception as e:
            self.logger.debug(f"群体凝聚度计算失败: {e}")
            return 0.5
    
    def _classify_group_behavior(self, group_persons: List[PersonDetection], 
                               interactions: List[PersonInteraction], 
                               helping_count: int, fallen_count: int) -> Tuple[GroupBehavior, float]:
        """分类群体行为
        
        Args:
            group_persons: 群体成员
            interactions: 交互列表
            helping_count: 帮助行为数量
            fallen_count: 跌倒人数
            
        Returns:
            Tuple[GroupBehavior, float]: (群体行为, 置信度)
        """
        try:
            group_size = len(group_persons)
            
            # 紧急响应行为
            if fallen_count > 0 and helping_count > 0:
                confidence = min(1.0, (helping_count / fallen_count) * 0.8 + 0.2)
                return GroupBehavior.EMERGENCY_RESPONSE, confidence
            
            # 帮助行为
            if helping_count > 0:
                return GroupBehavior.HELPING, 0.8
            
            # 恐慌行为（多人快速移动）
            fast_moving_count = sum(1 for p in group_persons 
                                  if np.sqrt(p.velocity[0]**2 + p.velocity[1]**2) > 10)
            if fast_moving_count > group_size * 0.6:
                return GroupBehavior.PANIC, 0.7
            
            # 聚集行为（群体凝聚度高且移动缓慢）
            avg_speed = np.mean([np.sqrt(p.velocity[0]**2 + p.velocity[1]**2) for p in group_persons])
            cohesion = self._calculate_group_cohesion(group_persons)
            
            if cohesion > 0.7 and avg_speed < 3:
                return GroupBehavior.GATHERING, 0.6
            
            # 分散行为（群体凝聚度低且有移动）
            if cohesion < 0.3 and avg_speed > 5:
                return GroupBehavior.DISPERSING, 0.6
            
            # 默认正常行为
            return GroupBehavior.NORMAL, 0.5
            
        except Exception as e:
            self.logger.error(f"群体行为分类失败: {e}")
            return GroupBehavior.NORMAL, 0.0
    
    def _calculate_emergency_level(self, fallen_count: int, helping_count: int, 
                                 group_size: int, behavior: GroupBehavior) -> float:
        """计算紧急程度
        
        Args:
            fallen_count: 跌倒人数
            helping_count: 帮助人数
            group_size: 群体大小
            behavior: 群体行为
            
        Returns:
            float: 紧急程度 (0-1)
        """
        try:
            emergency_level = 0.0
            
            # 基于跌倒人数
            if fallen_count > 0:
                emergency_level += min(1.0, fallen_count / group_size) * 0.6
            
            # 基于帮助情况
            if fallen_count > 0:
                help_ratio = helping_count / fallen_count
                if help_ratio < 0.5:  # 帮助不足
                    emergency_level += 0.3
                elif help_ratio >= 1.0:  # 帮助充足
                    emergency_level -= 0.1
            
            # 基于群体行为
            if behavior == GroupBehavior.PANIC:
                emergency_level += 0.4
            elif behavior == GroupBehavior.EMERGENCY_RESPONSE:
                emergency_level += 0.2  # 有响应但仍是紧急情况
            elif behavior == GroupBehavior.HELPING:
                emergency_level += 0.1
            
            return np.clip(emergency_level, 0.0, 1.0)
            
        except Exception as e:
            self.logger.debug(f"紧急程度计算失败: {e}")
            return 0.5

class MultiPersonDetector:
    """多人混合状态检测器主类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化多人检测器
        
        Args:
            config: 配置参数
        """
        self.logger = logging.getLogger(f"{__name__}.MultiPersonDetector")
        
        # 默认配置
        default_config = {
            'max_persons': 20,
            'tracking_threshold': 0.5,
            'state_change_threshold': 0.7,
            'alert_cooldown': 5.0,
            'history_length': 50
        }
        
        self.config = {**default_config, **(config or {})}
        
        # 初始化组件
        self.pose_analyzer = PoseAnalyzer()
        self.interaction_analyzer = InteractionAnalyzer()
        self.group_analyzer = GroupAnalyzer()
        
        # 状态跟踪
        self.person_history = defaultdict(deque)
        self.alert_history = deque(maxlen=100)
        self.last_alert_time = defaultdict(float)
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        self.logger.info("多人混合状态检测器初始化完成")
    
    def detect_multi_person_scene(self, frame: np.ndarray, detections: List[Dict[str, Any]], 
                                keypoints_list: List[List[List[float]]]) -> Tuple[List[PersonDetection], List[MultiPersonAlert]]:
        """检测多人混合状态场景
        
        Args:
            frame: 输入帧
            detections: 检测结果列表
            keypoints_list: 关键点列表
            
        Returns:
            Tuple[List[PersonDetection], List[MultiPersonAlert]]: (人员检测结果, 警报列表)
        """
        try:
            current_time = time.time()
            
            # 1. 处理人员检测
            persons = self._process_person_detections(detections, keypoints_list, current_time)
            
            # 2. 分析人员交互
            interactions = self.interaction_analyzer.analyze_interactions(persons)
            
            # 3. 分析群体行为
            group_analyses = self.group_analyzer.analyze_groups(persons, interactions)
            
            # 4. 生成警报
            alerts = self._generate_alerts(persons, interactions, group_analyses, current_time)
            
            # 5. 更新历史记录
            self._update_history(persons, alerts)
            
            return persons, alerts
            
        except Exception as e:
            self.logger.error(f"多人场景检测失败: {e}")
            return [], []
    
    def _process_person_detections(self, detections: List[Dict[str, Any]], 
                                 keypoints_list: List[List[List[float]]], 
                                 current_time: float) -> List[PersonDetection]:
        """处理人员检测
        
        Args:
            detections: 检测结果
            keypoints_list: 关键点列表
            current_time: 当前时间
            
        Returns:
            List[PersonDetection]: 人员检测结果
        """
        persons = []
        
        try:
            for i, detection in enumerate(detections):
                if detection.get('class') != 'person':
                    continue
                
                # 获取基本信息
                bbox = detection['bbox']
                person_id = detection.get('track_id', i)
                confidence = detection.get('confidence', 0.0)
                
                # 计算中心点
                center_point = (
                    (bbox[0] + bbox[2]) / 2,
                    (bbox[1] + bbox[3]) / 2
                )
                
                # 获取关键点
                keypoints = keypoints_list[i] if i < len(keypoints_list) else []
                
                # 姿态分析
                state, state_confidence, body_angle, height_ratio = self.pose_analyzer.analyze_pose(keypoints, bbox)
                
                # 计算运动信息
                velocity, acceleration = self._calculate_motion(person_id, center_point, current_time)
                
                # 计算状态持续时间
                duration_in_state = self._calculate_state_duration(person_id, state, current_time)
                
                # 获取历史状态
                previous_states = self._get_previous_states(person_id)
                
                # 检测遮挡
                is_occluded, occlusion_ratio = self._detect_occlusion(keypoints, bbox)
                
                # 创建人员检测对象
                person = PersonDetection(
                    person_id=person_id,
                    bbox=bbox,
                    center_point=center_point,
                    keypoints=keypoints,
                    state=state,
                    state_confidence=state_confidence,
                    body_angle=body_angle,
                    height_ratio=height_ratio,
                    velocity=velocity,
                    acceleration=acceleration,
                    duration_in_state=duration_in_state,
                    previous_states=previous_states,
                    tracking_confidence=confidence,
                    is_occluded=is_occluded,
                    occlusion_ratio=occlusion_ratio
                )
                
                persons.append(person)
            
            return persons
            
        except Exception as e:
            self.logger.error(f"人员检测处理失败: {e}")
            return []
    
    def _calculate_motion(self, person_id: int, center_point: Tuple[float, float], 
                        current_time: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """计算运动信息
        
        Args:
            person_id: 人员ID
            center_point: 中心点
            current_time: 当前时间
            
        Returns:
            Tuple[Tuple[float, float], Tuple[float, float]]: (速度, 加速度)
        """
        try:
            history = self.person_history[person_id]
            
            if len(history) < 2:
                return (0.0, 0.0), (0.0, 0.0)
            
            # 计算速度
            prev_record = history[-1]
            dt = current_time - prev_record['timestamp']
            
            if dt <= 0:
                return (0.0, 0.0), (0.0, 0.0)
            
            velocity = (
                (center_point[0] - prev_record['center_point'][0]) / dt,
                (center_point[1] - prev_record['center_point'][1]) / dt
            )
            
            # 计算加速度
            if len(history) >= 2 and 'velocity' in prev_record:
                prev_velocity = prev_record['velocity']
                acceleration = (
                    (velocity[0] - prev_velocity[0]) / dt,
                    (velocity[1] - prev_velocity[1]) / dt
                )
            else:
                acceleration = (0.0, 0.0)
            
            return velocity, acceleration
            
        except Exception as e:
            self.logger.debug(f"运动信息计算失败: {e}")
            return (0.0, 0.0), (0.0, 0.0)
    
    def _calculate_state_duration(self, person_id: int, current_state: PersonState, 
                                current_time: float) -> float:
        """计算状态持续时间
        
        Args:
            person_id: 人员ID
            current_state: 当前状态
            current_time: 当前时间
            
        Returns:
            float: 持续时间（秒）
        """
        try:
            history = self.person_history[person_id]
            
            if not history:
                return 0.0
            
            # 从最近的记录开始向前查找状态变化
            duration = 0.0
            for record in reversed(history):
                if record.get('state') == current_state:
                    duration = current_time - record['timestamp']
                else:
                    break
            
            return duration
            
        except Exception as e:
            self.logger.debug(f"状态持续时间计算失败: {e}")
            return 0.0
    
    def _get_previous_states(self, person_id: int) -> List[PersonState]:
        """获取历史状态
        
        Args:
            person_id: 人员ID
            
        Returns:
            List[PersonState]: 历史状态列表
        """
        try:
            history = self.person_history[person_id]
            states = [record.get('state', PersonState.UNKNOWN) for record in history]
            return states[-5:]  # 返回最近5个状态
            
        except Exception as e:
            self.logger.debug(f"历史状态获取失败: {e}")
            return []
    
    def _detect_occlusion(self, keypoints: List[List[float]], 
                         bbox: Tuple[int, int, int, int]) -> Tuple[bool, float]:
        """检测遮挡
        
        Args:
            keypoints: 关键点列表
            bbox: 边界框
            
        Returns:
            Tuple[bool, float]: (是否遮挡, 遮挡比例)
        """
        try:
            if not keypoints or len(keypoints) < 17:
                return True, 1.0
            
            # 计算可见关键点比例
            visible_count = sum(1 for kp in keypoints if kp[2] > 0.3)
            total_count = len(keypoints)
            
            visible_ratio = visible_count / total_count
            occlusion_ratio = 1.0 - visible_ratio
            
            is_occluded = occlusion_ratio > 0.3
            
            return is_occluded, occlusion_ratio
            
        except Exception as e:
            self.logger.debug(f"遮挡检测失败: {e}")
            return False, 0.0
    
    def _generate_alerts(self, persons: List[PersonDetection], 
                        interactions: List[PersonInteraction], 
                        group_analyses: List[GroupAnalysis], 
                        current_time: float) -> List[MultiPersonAlert]:
        """生成警报
        
        Args:
            persons: 人员列表
            interactions: 交互列表
            group_analyses: 群体分析列表
            current_time: 当前时间
            
        Returns:
            List[MultiPersonAlert]: 警报列表
        """
        alerts = []
        
        try:
            # 1. 跌倒警报
            fallen_persons = [p for p in persons if p.state == PersonState.FALLEN]
            
            for fallen_person in fallen_persons:
                # 检查冷却时间
                last_alert = self.last_alert_time.get(f"fall_{fallen_person.person_id}", 0)
                if current_time - last_alert < self.config['alert_cooldown']:
                    continue
                
                # 查找帮助者和旁观者
                helpers = []
                bystanders = []
                
                for interaction in interactions:
                    if (interaction.person1_id == fallen_person.person_id or 
                        interaction.person2_id == fallen_person.person_id):
                        other_id = (interaction.person2_id if interaction.person1_id == fallen_person.person_id 
                                  else interaction.person1_id)
                        
                        if interaction.is_helping:
                            helpers.append(other_id)
                        elif interaction.distance < 150:  # 附近的人
                            bystanders.append(other_id)
                
                # 确定警报优先级
                if fallen_person.duration_in_state > 10.0:  # 长时间跌倒
                    priority = AlertPriority.EMERGENCY
                elif fallen_person.duration_in_state > 5.0:
                    priority = AlertPriority.CRITICAL
                elif len(helpers) == 0 and len(bystanders) > 0:  # 有人在附近但没人帮助
                    priority = AlertPriority.HIGH
                else:
                    priority = AlertPriority.MEDIUM
                
                # 生成建议行动
                suggested_actions = self._generate_fall_suggestions(
                    fallen_person, helpers, bystanders, persons
                )
                
                # 估算响应时间
                response_time = self._estimate_response_time(priority, len(helpers), len(bystanders))
                
                alert = MultiPersonAlert(
                    alert_id=f"fall_{fallen_person.person_id}_{int(current_time)}",
                    timestamp=current_time,
                    alert_priority=priority,
                    affected_persons=[fallen_person.person_id],
                    fallen_persons=[fallen_person.person_id],
                    helping_persons=helpers,
                    bystander_persons=bystanders,
                    alert_message=f"检测到人员 {fallen_person.person_id} 跌倒，持续时间 {fallen_person.duration_in_state:.1f} 秒",
                    suggested_actions=suggested_actions,
                    requires_immediate_attention=priority in [AlertPriority.CRITICAL, AlertPriority.EMERGENCY],
                    estimated_response_time=response_time
                )
                
                alerts.append(alert)
                self.last_alert_time[f"fall_{fallen_person.person_id}"] = current_time
            
            # 2. 群体紧急情况警报
            for group in group_analyses:
                if group.emergency_level > 0.7:
                    alert_id = f"group_{group.group_id}_{int(current_time)}"
                    
                    # 检查冷却时间
                    last_alert = self.last_alert_time.get(f"group_{group.group_id}", 0)
                    if current_time - last_alert < self.config['alert_cooldown']:
                        continue
                    
                    # 确定优先级
                    if group.emergency_level > 0.9:
                        priority = AlertPriority.EMERGENCY
                    elif group.emergency_level > 0.8:
                        priority = AlertPriority.CRITICAL
                    else:
                        priority = AlertPriority.HIGH
                    
                    # 分类人员
                    fallen_ids = [p.person_id for p in persons 
                                 if p.person_id in group.member_ids and p.state == PersonState.FALLEN]
                    helping_ids = []
                    for interaction in interactions:
                        if (interaction.person1_id in group.member_ids and 
                            interaction.person2_id in group.member_ids and 
                            interaction.is_helping):
                            if interaction.person1_id not in fallen_ids:
                                helping_ids.append(interaction.person1_id)
                            if interaction.person2_id not in fallen_ids:
                                helping_ids.append(interaction.person2_id)
                    
                    helping_ids = list(set(helping_ids))  # 去重
                    bystander_ids = [pid for pid in group.member_ids 
                                   if pid not in fallen_ids and pid not in helping_ids]
                    
                    # 生成警报消息
                    message = f"群体紧急情况：{group.fallen_count} 人跌倒，{group.helping_count} 人提供帮助，群体行为：{group.group_behavior.value}"
                    
                    # 生成建议行动
                    suggested_actions = self._generate_group_suggestions(group, fallen_ids, helping_ids)
                    
                    alert = MultiPersonAlert(
                        alert_id=alert_id,
                        timestamp=current_time,
                        alert_priority=priority,
                        affected_persons=group.member_ids,
                        fallen_persons=fallen_ids,
                        helping_persons=helping_ids,
                        bystander_persons=bystander_ids,
                        alert_message=message,
                        suggested_actions=suggested_actions,
                        requires_immediate_attention=priority in [AlertPriority.CRITICAL, AlertPriority.EMERGENCY],
                        estimated_response_time=self._estimate_response_time(priority, len(helping_ids), len(bystander_ids)),
                        group_analysis=group
                    )
                    
                    alerts.append(alert)
                    self.last_alert_time[f"group_{group.group_id}"] = current_time
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"警报生成失败: {e}")
            return []
    
    def _generate_fall_suggestions(self, fallen_person: PersonDetection, 
                                 helpers: List[int], bystanders: List[int], 
                                 all_persons: List[PersonDetection]) -> List[str]:
        """生成跌倒建议行动
        
        Args:
            fallen_person: 跌倒人员
            helpers: 帮助者ID列表
            bystanders: 旁观者ID列表
            all_persons: 所有人员列表
            
        Returns:
            List[str]: 建议行动列表
        """
        suggestions = []
        
        try:
            # 基础建议
            suggestions.append("立即检查跌倒人员状况")
            
            # 基于持续时间的建议
            if fallen_person.duration_in_state > 10.0:
                suggestions.append("紧急呼叫医疗援助")
            elif fallen_person.duration_in_state > 5.0:
                suggestions.append("考虑呼叫医疗援助")
            
            # 基于帮助情况的建议
            if not helpers and bystanders:
                suggestions.append("引导旁观者提供帮助")
            elif not helpers and not bystanders:
                suggestions.append("派遣工作人员前往现场")
            elif helpers:
                suggestions.append("监控帮助过程")
            
            # 基于遮挡情况的建议
            if fallen_person.is_occluded:
                suggestions.append("调整摄像头角度以获得更好视野")
            
            return suggestions
            
        except Exception as e:
            self.logger.debug(f"跌倒建议生成失败: {e}")
            return ["检查跌倒人员状况"]
    
    def _generate_group_suggestions(self, group: GroupAnalysis, 
                                 fallen_ids: List[int], helping_ids: List[int]) -> List[str]:
        """生成群体建议行动
        
        Args:
            group: 群体分析结果
            fallen_ids: 跌倒人员ID列表
            helping_ids: 帮助人员ID列表
            
        Returns:
            List[str]: 建议行动列表
        """
        suggestions = []
        
        try:
            # 基于群体行为的建议
            if group.group_behavior == GroupBehavior.PANIC:
                suggestions.extend([
                    "立即疏散人群",
                    "维持现场秩序",
                    "呼叫安保人员"
                ])
            elif group.group_behavior == GroupBehavior.EMERGENCY_RESPONSE:
                suggestions.extend([
                    "支持现场救援行动",
                    "确保救援通道畅通",
                    "准备医疗设备"
                ])
            elif group.group_behavior == GroupBehavior.HELPING:
                suggestions.append("监控帮助过程")
            
            # 基于跌倒人数的建议
            if len(fallen_ids) > 1:
                suggestions.extend([
                    "启动多人事故应急预案",
                    "呼叫多个医疗单位"
                ])
            
            # 基于帮助比例的建议
            if len(fallen_ids) > 0:
                help_ratio = len(helping_ids) / len(fallen_ids)
                if help_ratio < 0.5:
                    suggestions.append("增派救援人员")
            
            return suggestions
            
        except Exception as e:
            self.logger.debug(f"群体建议生成失败: {e}")
            return ["监控群体状况"]
    
    def _estimate_response_time(self, priority: AlertPriority, 
                              helper_count: int, bystander_count: int) -> float:
        """估算响应时间
        
        Args:
            priority: 警报优先级
            helper_count: 帮助者数量
            bystander_count: 旁观者数量
            
        Returns:
            float: 估算响应时间（秒）
        """
        try:
            # 基础响应时间
            base_times = {
                AlertPriority.EMERGENCY: 30.0,
                AlertPriority.CRITICAL: 60.0,
                AlertPriority.HIGH: 120.0,
                AlertPriority.MEDIUM: 300.0,
                AlertPriority.LOW: 600.0
            }
            
            base_time = base_times.get(priority, 300.0)
            
            # 基于现场帮助情况调整
            if helper_count > 0:
                base_time *= 0.7  # 有人帮助，响应时间可以稍长
            elif bystander_count > 0:
                base_time *= 0.8  # 有旁观者，可能提供帮助
            else:
                base_time *= 1.2  # 无人在场，需要更快响应
            
            return base_time
            
        except Exception as e:
            self.logger.debug(f"响应时间估算失败: {e}")
            return 300.0
    
    def _update_history(self, persons: List[PersonDetection], 
                       alerts: List[MultiPersonAlert]) -> None:
        """更新历史记录
        
        Args:
            persons: 人员列表
            alerts: 警报列表
        """
        try:
            current_time = time.time()
            
            # 更新人员历史
            for person in persons:
                history = self.person_history[person.person_id]
                
                record = {
                    'timestamp': current_time,
                    'center_point': person.center_point,
                    'state': person.state,
                    'velocity': person.velocity,
                    'bbox': person.bbox
                }
                
                history.append(record)
                
                # 保持历史长度
                while len(history) > self.config['history_length']:
                    history.popleft()
            
            # 更新警报历史
            for alert in alerts:
                self.alert_history.append(alert)
            
        except Exception as e:
            self.logger.error(f"历史记录更新失败: {e}")
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """获取检测统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        try:
            stats = {
                'tracked_persons': len(self.person_history),
                'total_alerts': len(self.alert_history),
                'alert_types': {},
                'average_tracking_duration': 0.0,
                'most_common_states': {},
                'interaction_statistics': {}
            }
            
            # 统计警报类型
            for alert in self.alert_history:
                priority = alert.alert_priority.value
                stats['alert_types'][priority] = stats['alert_types'].get(priority, 0) + 1
            
            # 统计跟踪持续时间
            if self.person_history:
                total_duration = 0.0
                for person_id, history in self.person_history.items():
                    if len(history) >= 2:
                        duration = history[-1]['timestamp'] - history[0]['timestamp']
                        total_duration += duration
                
                stats['average_tracking_duration'] = total_duration / len(self.person_history)
            
            # 统计常见状态
            state_counts = defaultdict(int)
            for history in self.person_history.values():
                for record in history:
                    state = record.get('state', PersonState.UNKNOWN)
                    state_counts[state.value] += 1
            
            stats['most_common_states'] = dict(state_counts)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"统计信息获取失败: {e}")
            return {}
    
    def export_detection_report(self, output_path: str) -> bool:
        """导出检测报告
        
        Args:
            output_path: 输出路径
            
        Returns:
            bool: 是否成功
        """
        try:
            report = {
                'timestamp': time.time(),
                'statistics': self.get_detection_statistics(),
                'recent_alerts': [{
                    'alert_id': alert.alert_id,
                    'timestamp': alert.timestamp,
                    'priority': alert.alert_priority.value,
                    'message': alert.alert_message,
                    'affected_persons': alert.affected_persons,
                    'fallen_persons': alert.fallen_persons,
                    'helping_persons': alert.helping_persons
                } for alert in list(self.alert_history)[-20:]],  # 最近20个警报
                'configuration': self.config
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"检测报告已导出到: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"报告导出失败: {e}")
            return False
    
    def cleanup(self) -> None:
        """清理资源"""
        try:
            self.executor.shutdown(wait=True)
            self.logger.info("多人检测器资源清理完成")
            
        except Exception as e:
            self.logger.error(f"资源清理失败: {e}")


# 测试代码
if __name__ == "__main__":
    import numpy as np
    
    def test_multi_person_detector():
        """测试多人检测器"""
        print("=== 多人混合状态检测器测试 ===")
        
        # 创建检测器
        detector = MultiPersonDetector()
        
        # 模拟复杂场景数据
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 模拟多人检测结果
        persons = [
            PersonDetection(
                person_id=1,
                bbox=(100, 100, 200, 300),
                center_point=(150, 200),
                confidence=0.95,
                state=PersonState.FALLEN,
                velocity=(0.1, 0.2),
                duration_in_state=8.0,
                is_occluded=False
            ),
            PersonDetection(
                person_id=2,
                bbox=(300, 80, 400, 280),
                center_point=(350, 180),
                confidence=0.88,
                state=PersonState.STANDING,
                velocity=(0.5, 0.3),
                duration_in_state=2.0,
                is_occluded=False
            ),
            PersonDetection(
                person_id=3,
                bbox=(500, 120, 580, 320),
                center_point=(540, 220),
                confidence=0.92,
                state=PersonState.HELPING,
                velocity=(0.2, 0.1),
                duration_in_state=5.0,
                is_occluded=True
            )
        ]
        
        print(f"检测到 {len(persons)} 个人员")
        for person in persons:
            print(f"  人员 {person.person_id}: {person.state.value}, 置信度: {person.confidence:.2f}")
        
        # 执行检测
        alerts = detector.detect_multi_person_scene(frame, persons)
        
        print(f"\n生成 {len(alerts)} 个警报:")
        for alert in alerts:
            print(f"  警报 {alert.alert_id}:")
            print(f"    优先级: {alert.alert_priority.value}")
            print(f"    消息: {alert.alert_message}")
            print(f"    涉及人员: {alert.affected_persons}")
            print(f"    跌倒人员: {alert.fallen_persons}")
            print(f"    帮助人员: {alert.helping_persons}")
            print(f"    建议行动: {', '.join(alert.suggested_actions)}")
            print(f"    估算响应时间: {alert.estimated_response_time:.1f}秒")
        
        # 获取统计信息
        stats = detector.get_detection_statistics()
        print(f"\n检测统计:")
        print(f"  跟踪人员数: {stats.get('tracked_persons', 0)}")
        print(f"  总警报数: {stats.get('total_alerts', 0)}")
        print(f"  警报类型分布: {stats.get('alert_types', {})}")
        
        # 导出报告
        report_path = "multi_person_detection_report.json"
        if detector.export_detection_report(report_path):
            print(f"\n检测报告已导出到: {report_path}")
        
        # 清理资源
        detector.cleanup()
        print("\n测试完成")
    
    def test_complex_scenarios():
        """测试复杂场景"""
        print("\n=== 复杂场景测试 ===")
        
        detector = MultiPersonDetector()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 场景1: 多人跌倒
        print("\n场景1: 多人跌倒事故")
        persons_scenario1 = [
            PersonDetection(1, (100, 100, 200, 300), (150, 200), 0.95, 
                          PersonState.FALLEN, (0.0, 0.0), 12.0, False),
            PersonDetection(2, (250, 120, 350, 320), (300, 220), 0.90, 
                          PersonState.FALLEN, (0.0, 0.0), 8.0, False),
            PersonDetection(3, (400, 80, 500, 280), (450, 180), 0.88, 
                          PersonState.STANDING, (0.3, 0.2), 3.0, False)
        ]
        
        alerts1 = detector.detect_multi_person_scene(frame, persons_scenario1)
        print(f"生成 {len(alerts1)} 个警报")
        for alert in alerts1:
            print(f"  {alert.alert_priority.value}: {alert.alert_message}")
        
        # 场景2: 帮助场景
        print("\n场景2: 救援帮助场景")
        persons_scenario2 = [
            PersonDetection(1, (100, 100, 200, 300), (150, 200), 0.95, 
                          PersonState.FALLEN, (0.0, 0.0), 5.0, False),
            PersonDetection(2, (180, 80, 280, 280), (230, 180), 0.92, 
                          PersonState.HELPING, (0.1, 0.1), 8.0, False),
            PersonDetection(3, (300, 120, 400, 320), (350, 220), 0.88, 
                          PersonState.HELPING, (0.2, 0.1), 6.0, False)
        ]
        
        alerts2 = detector.detect_multi_person_scene(frame, persons_scenario2)
        print(f"生成 {len(alerts2)} 个警报")
        for alert in alerts2:
            print(f"  {alert.alert_priority.value}: {alert.alert_message}")
        
        # 场景3: 遮挡场景
        print("\n场景3: 视野遮挡场景")
        persons_scenario3 = [
            PersonDetection(1, (100, 100, 200, 300), (150, 200), 0.75, 
                          PersonState.FALLEN, (0.0, 0.0), 15.0, True),  # 被遮挡
            PersonDetection(2, (300, 80, 400, 280), (350, 180), 0.88, 
                          PersonState.STANDING, (0.4, 0.3), 2.0, False)
        ]
        
        alerts3 = detector.detect_multi_person_scene(frame, persons_scenario3)
        print(f"生成 {len(alerts3)} 个警报")
        for alert in alerts3:
            print(f"  {alert.alert_priority.value}: {alert.alert_message}")
            print(f"  建议: {', '.join(alert.suggested_actions)}")
        
        detector.cleanup()
        print("\n复杂场景测试完成")
    
    # 运行测试
    test_multi_person_detector()
    test_complex_scenarios()