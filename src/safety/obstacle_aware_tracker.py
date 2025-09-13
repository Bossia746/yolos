#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
障碍物感知跟踪模块

处理室内环境中的各种障碍物对人体跟踪的影响
包括玩具、家具、装饰品等物体的检测和跟踪优化

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


class ObstacleType(Enum):
    """障碍物类型"""
    FURNITURE = "furniture"        # 家具 (桌子、椅子、沙发等)
    TOY = "toy"                   # 玩具
    DECORATION = "decoration"     # 装饰品
    APPLIANCE = "appliance"       # 家电
    PLANT = "plant"               # 植物
    BOOK_SHELF = "book_shelf"     # 书柜
    CABINET = "cabinet"           # 柜子
    UNKNOWN = "unknown"           # 未知物体


class ObstacleImpact(Enum):
    """障碍物影响程度"""
    NONE = "none"           # 无影响
    MINIMAL = "minimal"     # 最小影响
    MODERATE = "moderate"   # 中等影响
    SIGNIFICANT = "significant"  # 显著影响
    SEVERE = "severe"       # 严重影响


class TrackingDifficulty(Enum):
    """跟踪难度等级"""
    EASY = "easy"           # 简单
    NORMAL = "normal"       # 正常
    HARD = "hard"           # 困难
    VERY_HARD = "very_hard" # 非常困难
    IMPOSSIBLE = "impossible"  # 不可能


class AdaptationStrategy(Enum):
    """适应策略"""
    STANDARD = "standard"           # 标准跟踪
    ENHANCED_DETECTION = "enhanced_detection"  # 增强检测
    MULTI_ANGLE = "multi_angle"     # 多角度跟踪
    PREDICTIVE = "predictive"       # 预测跟踪
    COLLABORATIVE = "collaborative" # 协作跟踪


@dataclass
class ObstacleInfo:
    """障碍物信息"""
    obstacle_id: int
    obstacle_type: ObstacleType
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center_point: Tuple[float, float]
    confidence: float
    height_estimate: float  # 估计高度 (像素)
    area: float
    is_static: bool = True  # 是否静态
    last_seen: float = field(default_factory=time.time)
    
    def get_area(self) -> float:
        """获取障碍物面积"""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
    
    def distance_to_point(self, point: Tuple[float, float]) -> float:
        """计算到指定点的距离"""
        return math.sqrt(
            (self.center_point[0] - point[0])**2 + 
            (self.center_point[1] - point[1])**2
        )


@dataclass
class TrackingContext:
    """跟踪上下文"""
    person_id: int
    current_position: Tuple[float, float]
    predicted_position: Tuple[float, float]
    velocity: Tuple[float, float]
    nearby_obstacles: List[ObstacleInfo]
    tracking_difficulty: TrackingDifficulty
    recommended_strategy: AdaptationStrategy
    confidence_penalty: float = 0.0
    
    def get_obstacle_density(self) -> float:
        """获取周围障碍物密度"""
        if not self.nearby_obstacles:
            return 0.0
        
        total_area = sum(obs.get_area() for obs in self.nearby_obstacles)
        # 假设检测区域为200x200像素
        detection_area = 200 * 200
        return min(1.0, total_area / detection_area)


@dataclass
class AdaptationConfig:
    """适应配置"""
    # 障碍物检测参数
    detection_params: Dict[str, Any] = field(default_factory=lambda: {
        'min_obstacle_area': 500,
        'max_obstacle_area': 50000,
        'confidence_threshold': 0.3,
        'nms_threshold': 0.5
    })
    
    # 影响评估参数
    impact_params: Dict[str, Any] = field(default_factory=lambda: {
        'proximity_threshold': 100.0,  # 像素
        'occlusion_threshold': 0.3,
        'height_factor': 0.5,
        'area_factor': 0.3
    })
    
    # 跟踪适应参数
    adaptation_params: Dict[str, Any] = field(default_factory=lambda: {
        'difficulty_thresholds': {
            'easy': 0.2,
            'normal': 0.4,
            'hard': 0.6,
            'very_hard': 0.8
        },
        'strategy_weights': {
            AdaptationStrategy.STANDARD: 1.0,
            AdaptationStrategy.ENHANCED_DETECTION: 1.2,
            AdaptationStrategy.MULTI_ANGLE: 1.5,
            AdaptationStrategy.PREDICTIVE: 1.3,
            AdaptationStrategy.COLLABORATIVE: 1.8
        }
    })


class ObstacleDetector:
    """障碍物检测器"""
    
    def __init__(self, config: AdaptationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.obstacle_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=10))
        self.next_obstacle_id = 1
        
        # 预定义的障碍物特征
        self.obstacle_features = {
            ObstacleType.FURNITURE: {
                'typical_height_range': (50, 200),
                'typical_area_range': (1000, 20000),
                'color_patterns': ['brown', 'black', 'white']
            },
            ObstacleType.TOY: {
                'typical_height_range': (10, 80),
                'typical_area_range': (100, 2000),
                'color_patterns': ['bright', 'colorful']
            },
            ObstacleType.BOOK_SHELF: {
                'typical_height_range': (100, 300),
                'typical_area_range': (5000, 30000),
                'color_patterns': ['brown', 'white', 'black']
            }
        }
    
    def detect_obstacles(self, frame: np.ndarray, 
                        exclude_regions: List[Tuple[int, int, int, int]] = None) -> List[ObstacleInfo]:
        """检测帧中的障碍物
        
        Args:
            frame: 输入帧
            exclude_regions: 排除区域 (人体检测区域)
            
        Returns:
            List[ObstacleInfo]: 检测到的障碍物列表
        """
        try:
            # 1. 预处理
            processed_frame = self._preprocess_frame(frame)
            
            # 2. 边缘检测
            edges = self._detect_edges(processed_frame)
            
            # 3. 轮廓检测
            contours = self._find_contours(edges)
            
            # 4. 过滤和分类轮廓
            obstacle_candidates = self._filter_contours(contours, exclude_regions)
            
            # 5. 障碍物分类
            obstacles = self._classify_obstacles(obstacle_candidates, frame)
            
            # 6. 跟踪匹配
            tracked_obstacles = self._match_obstacles_to_tracks(obstacles)
            
            return tracked_obstacles
            
        except Exception as e:
            self.logger.error(f"障碍物检测失败: {e}")
            return []
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """预处理帧"""
        # 转换为灰度图
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 直方图均衡化
        equalized = cv2.equalizeHist(blurred)
        
        return equalized
    
    def _detect_edges(self, frame: np.ndarray) -> np.ndarray:
        """检测边缘"""
        # 使用Canny边缘检测
        edges = cv2.Canny(frame, 50, 150)
        
        # 形态学操作，连接断开的边缘
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return edges
    
    def _find_contours(self, edges: np.ndarray) -> List[np.ndarray]:
        """查找轮廓"""
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def _filter_contours(self, contours: List[np.ndarray], 
                        exclude_regions: List[Tuple[int, int, int, int]] = None) -> List[Dict[str, Any]]:
        """过滤轮廓"""
        candidates = []
        exclude_regions = exclude_regions or []
        
        for contour in contours:
            # 计算轮廓属性
            area = cv2.contourArea(contour)
            
            # 面积过滤
            if (area < self.config.detection_params['min_obstacle_area'] or 
                area > self.config.detection_params['max_obstacle_area']):
                continue
            
            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)
            bbox = (x, y, x + w, y + h)
            
            # 检查是否与排除区域重叠
            if self._overlaps_with_excluded_regions(bbox, exclude_regions):
                continue
            
            # 计算其他属性
            center_x = x + w / 2
            center_y = y + h / 2
            aspect_ratio = w / h if h > 0 else 0
            
            candidate = {
                'contour': contour,
                'bbox': bbox,
                'center': (center_x, center_y),
                'area': area,
                'width': w,
                'height': h,
                'aspect_ratio': aspect_ratio
            }
            
            candidates.append(candidate)
        
        return candidates
    
    def _overlaps_with_excluded_regions(self, bbox: Tuple[int, int, int, int], 
                                      exclude_regions: List[Tuple[int, int, int, int]]) -> bool:
        """检查是否与排除区域重叠"""
        x1, y1, x2, y2 = bbox
        
        for ex_x1, ex_y1, ex_x2, ex_y2 in exclude_regions:
            # 计算重叠面积
            overlap_x1 = max(x1, ex_x1)
            overlap_y1 = max(y1, ex_y1)
            overlap_x2 = min(x2, ex_x2)
            overlap_y2 = min(y2, ex_y2)
            
            if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                bbox_area = (x2 - x1) * (y2 - y1)
                
                # 如果重叠面积超过30%，认为重叠
                if overlap_area / bbox_area > 0.3:
                    return True
        
        return False
    
    def _classify_obstacles(self, candidates: List[Dict[str, Any]], 
                          frame: np.ndarray) -> List[ObstacleInfo]:
        """分类障碍物"""
        obstacles = []
        
        for candidate in candidates:
            # 基于几何特征分类
            obstacle_type = self._classify_by_geometry(candidate)
            
            # 基于颜色特征进一步分类
            obstacle_type = self._refine_classification_by_color(
                obstacle_type, candidate, frame
            )
            
            # 创建障碍物信息
            obstacle = ObstacleInfo(
                obstacle_id=self.next_obstacle_id,
                obstacle_type=obstacle_type,
                bbox=candidate['bbox'],
                center_point=candidate['center'],
                confidence=self._calculate_classification_confidence(candidate, obstacle_type),
                height_estimate=candidate['height'],
                area=candidate['area']
            )
            
            obstacles.append(obstacle)
            self.next_obstacle_id += 1
        
        return obstacles
    
    def _classify_by_geometry(self, candidate: Dict[str, Any]) -> ObstacleType:
        """基于几何特征分类"""
        width = candidate['width']
        height = candidate['height']
        area = candidate['area']
        aspect_ratio = candidate['aspect_ratio']
        
        # 基于尺寸和比例的简单分类
        if area < 1000:
            return ObstacleType.TOY
        elif height > 150 and aspect_ratio < 0.8:  # 高且窄
            if area > 10000:
                return ObstacleType.BOOK_SHELF
            else:
                return ObstacleType.DECORATION
        elif aspect_ratio > 1.5:  # 宽且矮
            return ObstacleType.FURNITURE
        elif area > 5000:
            return ObstacleType.FURNITURE
        else:
            return ObstacleType.UNKNOWN
    
    def _refine_classification_by_color(self, initial_type: ObstacleType, 
                                      candidate: Dict[str, Any], 
                                      frame: np.ndarray) -> ObstacleType:
        """基于颜色特征细化分类"""
        try:
            # 提取ROI
            x1, y1, x2, y2 = candidate['bbox']
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                return initial_type
            
            # 计算颜色特征
            if len(roi.shape) == 3:
                # 计算平均颜色
                mean_color = np.mean(roi, axis=(0, 1))
                
                # 计算颜色方差（彩色程度）
                color_variance = np.var(roi, axis=(0, 1))
                colorfulness = np.mean(color_variance)
                
                # 基于颜色特征调整分类
                if colorfulness > 800:  # 高彩色度
                    if initial_type == ObstacleType.UNKNOWN and candidate['area'] < 2000:
                        return ObstacleType.TOY
                elif colorfulness < 200:  # 低彩色度（单色）
                    if initial_type == ObstacleType.UNKNOWN:
                        return ObstacleType.FURNITURE
            
            return initial_type
            
        except Exception:
            return initial_type
    
    def _calculate_classification_confidence(self, candidate: Dict[str, Any], 
                                           obstacle_type: ObstacleType) -> float:
        """计算分类置信度"""
        base_confidence = 0.7
        
        # 基于面积的置信度调整
        area = candidate['area']
        if obstacle_type in self.obstacle_features:
            expected_range = self.obstacle_features[obstacle_type]['typical_area_range']
            if expected_range[0] <= area <= expected_range[1]:
                base_confidence += 0.2
            else:
                base_confidence -= 0.1
        
        # 基于高度的置信度调整
        height = candidate['height']
        if obstacle_type in self.obstacle_features:
            expected_range = self.obstacle_features[obstacle_type]['typical_height_range']
            if expected_range[0] <= height <= expected_range[1]:
                base_confidence += 0.1
            else:
                base_confidence -= 0.05
        
        return max(0.1, min(1.0, base_confidence))
    
    def _match_obstacles_to_tracks(self, obstacles: List[ObstacleInfo]) -> List[ObstacleInfo]:
        """将障碍物匹配到跟踪轨迹"""
        matched_obstacles = []
        
        for obstacle in obstacles:
            best_match_id = None
            best_distance = float('inf')
            
            # 寻找最近的历史障碍物
            for track_id, history in self.obstacle_history.items():
                if not history:
                    continue
                
                last_obstacle = history[-1]
                distance = obstacle.distance_to_point(last_obstacle.center_point)
                
                # 距离阈值
                if distance < 50.0 and distance < best_distance:
                    best_distance = distance
                    best_match_id = track_id
            
            # 分配ID
            if best_match_id is not None:
                obstacle.obstacle_id = best_match_id
            
            # 更新历史
            self.obstacle_history[obstacle.obstacle_id].append(obstacle)
            matched_obstacles.append(obstacle)
        
        return matched_obstacles


class ImpactAnalyzer:
    """影响分析器"""
    
    def __init__(self, config: AdaptationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze_obstacle_impact(self, person_position: Tuple[float, float], 
                              person_bbox: Tuple[int, int, int, int],
                              obstacles: List[ObstacleInfo]) -> Tuple[ObstacleImpact, Dict[str, Any]]:
        """分析障碍物对人体跟踪的影响
        
        Args:
            person_position: 人体位置
            person_bbox: 人体边界框
            obstacles: 障碍物列表
            
        Returns:
            Tuple[ObstacleImpact, Dict]: 影响程度和详细信息
        """
        try:
            # 筛选附近的障碍物
            nearby_obstacles = self._find_nearby_obstacles(
                person_position, obstacles
            )
            
            if not nearby_obstacles:
                return ObstacleImpact.NONE, {'nearby_obstacles': []}
            
            # 计算各种影响因子
            proximity_impact = self._calculate_proximity_impact(
                person_position, nearby_obstacles
            )
            
            occlusion_impact = self._calculate_occlusion_impact(
                person_bbox, nearby_obstacles
            )
            
            movement_impact = self._calculate_movement_impact(
                person_position, nearby_obstacles
            )
            
            # 综合影响评估
            overall_impact = self._calculate_overall_impact(
                proximity_impact, occlusion_impact, movement_impact
            )
            
            impact_level = self._determine_impact_level(overall_impact)
            
            details = {
                'nearby_obstacles': [obs.obstacle_id for obs in nearby_obstacles],
                'proximity_impact': proximity_impact,
                'occlusion_impact': occlusion_impact,
                'movement_impact': movement_impact,
                'overall_impact': overall_impact,
                'obstacle_count': len(nearby_obstacles),
                'dominant_obstacle_type': self._get_dominant_obstacle_type(nearby_obstacles)
            }
            
            return impact_level, details
            
        except Exception as e:
            self.logger.error(f"影响分析失败: {e}")
            return ObstacleImpact.NONE, {}
    
    def _find_nearby_obstacles(self, person_position: Tuple[float, float], 
                             obstacles: List[ObstacleInfo]) -> List[ObstacleInfo]:
        """查找附近的障碍物"""
        nearby = []
        threshold = self.config.impact_params['proximity_threshold']
        
        for obstacle in obstacles:
            distance = obstacle.distance_to_point(person_position)
            if distance <= threshold:
                nearby.append(obstacle)
        
        return nearby
    
    def _calculate_proximity_impact(self, person_position: Tuple[float, float], 
                                  obstacles: List[ObstacleInfo]) -> float:
        """计算接近度影响"""
        if not obstacles:
            return 0.0
        
        total_impact = 0.0
        threshold = self.config.impact_params['proximity_threshold']
        
        for obstacle in obstacles:
            distance = obstacle.distance_to_point(person_position)
            # 距离越近影响越大
            proximity_factor = max(0.0, 1.0 - distance / threshold)
            
            # 障碍物大小影响
            size_factor = min(1.0, obstacle.area / 5000.0)
            
            impact = proximity_factor * size_factor
            total_impact += impact
        
        return min(1.0, total_impact)
    
    def _calculate_occlusion_impact(self, person_bbox: Tuple[int, int, int, int], 
                                  obstacles: List[ObstacleInfo]) -> float:
        """计算遮挡影响"""
        if not obstacles:
            return 0.0
        
        person_area = (person_bbox[2] - person_bbox[0]) * (person_bbox[3] - person_bbox[1])
        total_occlusion = 0.0
        
        for obstacle in obstacles:
            # 计算重叠面积
            overlap_area = self._calculate_overlap_area(person_bbox, obstacle.bbox)
            
            if overlap_area > 0:
                occlusion_ratio = overlap_area / person_area
                total_occlusion += occlusion_ratio
        
        return min(1.0, total_occlusion)
    
    def _calculate_overlap_area(self, bbox1: Tuple[int, int, int, int], 
                              bbox2: Tuple[int, int, int, int]) -> float:
        """计算两个边界框的重叠面积"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 计算交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        return (x2_i - x1_i) * (y2_i - y1_i)
    
    def _calculate_movement_impact(self, person_position: Tuple[float, float], 
                                 obstacles: List[ObstacleInfo]) -> float:
        """计算运动影响"""
        if not obstacles:
            return 0.0
        
        # 简化的运动影响计算
        # 实际实现中可以考虑人体运动方向和障碍物位置的关系
        
        movement_impact = 0.0
        
        for obstacle in obstacles:
            # 基于障碍物类型的运动影响
            type_impact = {
                ObstacleType.TOY: 0.3,        # 玩具可能移动
                ObstacleType.FURNITURE: 0.1,  # 家具通常静止
                ObstacleType.PLANT: 0.05,     # 植物基本静止
                ObstacleType.APPLIANCE: 0.15, # 家电偶尔移动
                ObstacleType.DECORATION: 0.05 # 装饰品基本静止
            }.get(obstacle.obstacle_type, 0.2)
            
            # 基于距离的影响衰减
            distance = obstacle.distance_to_point(person_position)
            distance_factor = max(0.0, 1.0 - distance / 100.0)
            
            movement_impact += type_impact * distance_factor
        
        return min(1.0, movement_impact)
    
    def _calculate_overall_impact(self, proximity: float, occlusion: float, 
                                movement: float) -> float:
        """计算综合影响"""
        # 加权综合
        weights = {
            'proximity': 0.4,
            'occlusion': 0.4,
            'movement': 0.2
        }
        
        overall = (proximity * weights['proximity'] + 
                  occlusion * weights['occlusion'] + 
                  movement * weights['movement'])
        
        return overall
    
    def _determine_impact_level(self, overall_impact: float) -> ObstacleImpact:
        """确定影响等级"""
        if overall_impact < 0.1:
            return ObstacleImpact.NONE
        elif overall_impact < 0.3:
            return ObstacleImpact.MINIMAL
        elif overall_impact < 0.5:
            return ObstacleImpact.MODERATE
        elif overall_impact < 0.7:
            return ObstacleImpact.SIGNIFICANT
        else:
            return ObstacleImpact.SEVERE
    
    def _get_dominant_obstacle_type(self, obstacles: List[ObstacleInfo]) -> str:
        """获取主要障碍物类型"""
        if not obstacles:
            return "none"
        
        type_counts = defaultdict(int)
        for obstacle in obstacles:
            type_counts[obstacle.obstacle_type.value] += 1
        
        return max(type_counts.items(), key=lambda x: x[1])[0]


class AdaptiveTracker:
    """自适应跟踪器"""
    
    def __init__(self, config: AdaptationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.tracking_contexts: Dict[int, TrackingContext] = {}
    
    def adapt_tracking_strategy(self, person_id: int, 
                              person_position: Tuple[float, float],
                              person_bbox: Tuple[int, int, int, int],
                              velocity: Tuple[float, float],
                              obstacles: List[ObstacleInfo],
                              impact_info: Dict[str, Any]) -> TrackingContext:
        """适应跟踪策略
        
        Args:
            person_id: 人员ID
            person_position: 当前位置
            person_bbox: 边界框
            velocity: 速度
            obstacles: 障碍物列表
            impact_info: 影响信息
            
        Returns:
            TrackingContext: 跟踪上下文
        """
        try:
            # 预测下一位置
            predicted_position = (
                person_position[0] + velocity[0] * 3,  # 3帧预测
                person_position[1] + velocity[1] * 3
            )
            
            # 查找附近障碍物
            nearby_obstacles = self._find_obstacles_in_trajectory(
                person_position, predicted_position, obstacles
            )
            
            # 评估跟踪难度
            difficulty = self._assess_tracking_difficulty(
                impact_info, nearby_obstacles
            )
            
            # 选择适应策略
            strategy = self._select_adaptation_strategy(
                difficulty, nearby_obstacles, impact_info
            )
            
            # 计算置信度惩罚
            confidence_penalty = self._calculate_confidence_penalty(
                difficulty, impact_info
            )
            
            # 创建跟踪上下文
            context = TrackingContext(
                person_id=person_id,
                current_position=person_position,
                predicted_position=predicted_position,
                velocity=velocity,
                nearby_obstacles=nearby_obstacles,
                tracking_difficulty=difficulty,
                recommended_strategy=strategy,
                confidence_penalty=confidence_penalty
            )
            
            # 更新跟踪上下文
            self.tracking_contexts[person_id] = context
            
            return context
            
        except Exception as e:
            self.logger.error(f"跟踪策略适应失败: {e}")
            # 返回默认上下文
            return TrackingContext(
                person_id=person_id,
                current_position=person_position,
                predicted_position=person_position,
                velocity=velocity,
                nearby_obstacles=[],
                tracking_difficulty=TrackingDifficulty.NORMAL,
                recommended_strategy=AdaptationStrategy.STANDARD
            )
    
    def _find_obstacles_in_trajectory(self, current_pos: Tuple[float, float], 
                                    predicted_pos: Tuple[float, float],
                                    obstacles: List[ObstacleInfo]) -> List[ObstacleInfo]:
        """查找轨迹上的障碍物"""
        trajectory_obstacles = []
        
        # 定义轨迹缓冲区
        buffer_radius = 50.0
        
        for obstacle in obstacles:
            # 计算点到线段的距离
            distance = self._point_to_line_distance(
                obstacle.center_point, current_pos, predicted_pos
            )
            
            if distance <= buffer_radius:
                trajectory_obstacles.append(obstacle)
        
        return trajectory_obstacles
    
    def _point_to_line_distance(self, point: Tuple[float, float], 
                              line_start: Tuple[float, float], 
                              line_end: Tuple[float, float]) -> float:
        """计算点到线段的距离"""
        try:
            px, py = point
            x1, y1 = line_start
            x2, y2 = line_end
            
            # 线段长度
            line_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            if line_length == 0:
                return math.sqrt((px - x1)**2 + (py - y1)**2)
            
            # 计算投影参数
            t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_length**2))
            
            # 投影点
            proj_x = x1 + t * (x2 - x1)
            proj_y = y1 + t * (y2 - y1)
            
            # 距离
            return math.sqrt((px - proj_x)**2 + (py - proj_y)**2)
            
        except Exception:
            return float('inf')
    
    def _assess_tracking_difficulty(self, impact_info: Dict[str, Any], 
                                  nearby_obstacles: List[ObstacleInfo]) -> TrackingDifficulty:
        """评估跟踪难度"""
        # 基于影响信息的难度评估
        overall_impact = impact_info.get('overall_impact', 0.0)
        obstacle_count = len(nearby_obstacles)
        
        # 计算难度分数
        difficulty_score = overall_impact
        
        # 障碍物数量影响
        if obstacle_count > 3:
            difficulty_score += 0.2
        elif obstacle_count > 1:
            difficulty_score += 0.1
        
        # 遮挡影响
        occlusion_impact = impact_info.get('occlusion_impact', 0.0)
        if occlusion_impact > 0.5:
            difficulty_score += 0.3
        
        # 确定难度等级
        thresholds = self.config.adaptation_params['difficulty_thresholds']
        
        if difficulty_score < thresholds['easy']:
            return TrackingDifficulty.EASY
        elif difficulty_score < thresholds['normal']:
            return TrackingDifficulty.NORMAL
        elif difficulty_score < thresholds['hard']:
            return TrackingDifficulty.HARD
        elif difficulty_score < thresholds['very_hard']:
            return TrackingDifficulty.VERY_HARD
        else:
            return TrackingDifficulty.IMPOSSIBLE
    
    def _select_adaptation_strategy(self, difficulty: TrackingDifficulty, 
                                  nearby_obstacles: List[ObstacleInfo],
                                  impact_info: Dict[str, Any]) -> AdaptationStrategy:
        """选择适应策略"""
        # 基于难度的策略选择
        if difficulty == TrackingDifficulty.EASY:
            return AdaptationStrategy.STANDARD
        elif difficulty == TrackingDifficulty.NORMAL:
            # 基于障碍物类型选择
            if any(obs.obstacle_type == ObstacleType.TOY for obs in nearby_obstacles):
                return AdaptationStrategy.ENHANCED_DETECTION
            else:
                return AdaptationStrategy.STANDARD
        elif difficulty == TrackingDifficulty.HARD:
            # 基于遮挡情况选择
            occlusion_impact = impact_info.get('occlusion_impact', 0.0)
            if occlusion_impact > 0.4:
                return AdaptationStrategy.MULTI_ANGLE
            else:
                return AdaptationStrategy.ENHANCED_DETECTION
        elif difficulty == TrackingDifficulty.VERY_HARD:
            # 多策略组合
            if len(nearby_obstacles) > 2:
                return AdaptationStrategy.COLLABORATIVE
            else:
                return AdaptationStrategy.PREDICTIVE
        else:  # IMPOSSIBLE
            return AdaptationStrategy.COLLABORATIVE
    
    def _calculate_confidence_penalty(self, difficulty: TrackingDifficulty, 
                                    impact_info: Dict[str, Any]) -> float:
        """计算置信度惩罚"""
        base_penalty = {
            TrackingDifficulty.EASY: 0.0,
            TrackingDifficulty.NORMAL: 0.05,
            TrackingDifficulty.HARD: 0.15,
            TrackingDifficulty.VERY_HARD: 0.25,
            TrackingDifficulty.IMPOSSIBLE: 0.4
        }.get(difficulty, 0.1)
        
        # 基于具体影响调整
        overall_impact = impact_info.get('overall_impact', 0.0)
        impact_penalty = overall_impact * 0.2
        
        return min(0.5, base_penalty + impact_penalty)
    
    def get_tracking_recommendations(self, person_id: int) -> Dict[str, Any]:
        """获取跟踪建议
        
        Args:
            person_id: 人员ID
            
        Returns:
            Dict[str, Any]: 跟踪建议
        """
        if person_id not in self.tracking_contexts:
            return {}
        
        context = self.tracking_contexts[person_id]
        
        recommendations = {
            'strategy': context.recommended_strategy.value,
            'difficulty': context.tracking_difficulty.value,
            'confidence_penalty': context.confidence_penalty,
            'obstacle_count': len(context.nearby_obstacles),
            'obstacle_density': context.get_obstacle_density(),
            'suggested_actions': self._generate_action_suggestions(context)
        }
        
        return recommendations
    
    def _generate_action_suggestions(self, context: TrackingContext) -> List[str]:
        """生成行动建议"""
        suggestions = []
        
        # 基于策略的建议
        if context.recommended_strategy == AdaptationStrategy.ENHANCED_DETECTION:
            suggestions.append("增强检测灵敏度")
            suggestions.append("使用多尺度检测")
        elif context.recommended_strategy == AdaptationStrategy.MULTI_ANGLE:
            suggestions.append("启用多角度跟踪")
            suggestions.append("增加摄像头覆盖")
        elif context.recommended_strategy == AdaptationStrategy.PREDICTIVE:
            suggestions.append("启用运动预测")
            suggestions.append("增加预测窗口")
        elif context.recommended_strategy == AdaptationStrategy.COLLABORATIVE:
            suggestions.append("启用协作跟踪")
            suggestions.append("融合多传感器数据")
        
        # 基于障碍物的建议
        if len(context.nearby_obstacles) > 2:
            suggestions.append("清理跟踪区域障碍物")
        
        toy_obstacles = [obs for obs in context.nearby_obstacles 
                        if obs.obstacle_type == ObstacleType.TOY]
        if toy_obstacles:
            suggestions.append("注意地面玩具移动")
        
        return suggestions


class ObstacleAwareTracker:
    """障碍物感知跟踪器主类"""
    
    def __init__(self, config: Optional[AdaptationConfig] = None):
        self.config = config or AdaptationConfig()
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.obstacle_detector = ObstacleDetector(self.config)
        self.impact_analyzer = ImpactAnalyzer(self.config)
        self.adaptive_tracker = AdaptiveTracker(self.config)
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # 统计信息
        self.stats = {
            'total_frames': 0,
            'obstacles_detected': 0,
            'tracking_adaptations': 0,
            'obstacle_types': defaultdict(int),
            'impact_levels': defaultdict(int),
            'strategies_used': defaultdict(int)
        }
    
    def process_frame(self, frame: np.ndarray, 
                     person_detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """处理帧并进行障碍物感知跟踪
        
        Args:
            frame: 输入帧
            person_detections: 人体检测结果列表
                每个检测包含: {'person_id', 'bbox', 'position', 'velocity', 'confidence'}
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        try:
            self.stats['total_frames'] += 1
            
            # 1. 检测障碍物
            exclude_regions = [det['bbox'] for det in person_detections]
            obstacles = self.obstacle_detector.detect_obstacles(frame, exclude_regions)
            
            self.stats['obstacles_detected'] += len(obstacles)
            for obstacle in obstacles:
                self.stats['obstacle_types'][obstacle.obstacle_type.value] += 1
            
            # 2. 分析影响并适应跟踪策略
            tracking_results = []
            
            for detection in person_detections:
                person_id = detection['person_id']
                position = detection['position']
                bbox = detection['bbox']
                velocity = detection.get('velocity', (0.0, 0.0))
                
                # 分析障碍物影响
                impact_level, impact_info = self.impact_analyzer.analyze_obstacle_impact(
                    position, bbox, obstacles
                )
                
                self.stats['impact_levels'][impact_level.value] += 1
                
                # 适应跟踪策略
                tracking_context = self.adaptive_tracker.adapt_tracking_strategy(
                    person_id, position, bbox, velocity, obstacles, impact_info
                )
                
                self.stats['strategies_used'][tracking_context.recommended_strategy.value] += 1
                self.stats['tracking_adaptations'] += 1
                
                # 获取跟踪建议
                recommendations = self.adaptive_tracker.get_tracking_recommendations(person_id)
                
                tracking_result = {
                    'person_id': person_id,
                    'original_detection': detection,
                    'impact_level': impact_level.value,
                    'impact_info': impact_info,
                    'tracking_context': {
                        'difficulty': tracking_context.tracking_difficulty.value,
                        'strategy': tracking_context.recommended_strategy.value,
                        'confidence_penalty': tracking_context.confidence_penalty,
                        'predicted_position': tracking_context.predicted_position
                    },
                    'recommendations': recommendations,
                    'nearby_obstacles': [
                        {
                            'id': obs.obstacle_id,
                            'type': obs.obstacle_type.value,
                            'bbox': obs.bbox,
                            'confidence': obs.confidence
                        } for obs in tracking_context.nearby_obstacles
                    ]
                }
                
                tracking_results.append(tracking_result)
            
            # 3. 生成处理结果
            result = {
                'frame_info': {
                    'timestamp': time.time(),
                    'frame_size': frame.shape[:2],
                    'obstacles_detected': len(obstacles),
                    'persons_tracked': len(person_detections)
                },
                'obstacles': [
                    {
                        'id': obs.obstacle_id,
                        'type': obs.obstacle_type.value,
                        'bbox': obs.bbox,
                        'center': obs.center_point,
                        'confidence': obs.confidence,
                        'area': obs.area
                    } for obs in obstacles
                ],
                'tracking_results': tracking_results,
                'scene_analysis': {
                    'obstacle_density': len(obstacles) / (frame.shape[0] * frame.shape[1] / 10000),
                    'dominant_obstacle_type': self._get_dominant_obstacle_type(obstacles),
                    'average_impact_level': self._calculate_average_impact_level(tracking_results),
                    'tracking_complexity': self._assess_scene_complexity(obstacles, person_detections)
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"帧处理失败: {e}")
            return {'error': str(e)}
    
    def _get_dominant_obstacle_type(self, obstacles: List[ObstacleInfo]) -> str:
        """获取主要障碍物类型"""
        if not obstacles:
            return "none"
        
        type_counts = defaultdict(int)
        for obstacle in obstacles:
            type_counts[obstacle.obstacle_type.value] += 1
        
        return max(type_counts.items(), key=lambda x: x[1])[0]
    
    def _calculate_average_impact_level(self, tracking_results: List[Dict[str, Any]]) -> str:
        """计算平均影响等级"""
        if not tracking_results:
            return "none"
        
        impact_values = {
            'none': 0, 'minimal': 1, 'moderate': 2, 'significant': 3, 'severe': 4
        }
        
        total_impact = sum(impact_values.get(result['impact_level'], 0) 
                          for result in tracking_results)
        avg_impact = total_impact / len(tracking_results)
        
        # 转换回等级
        for level, value in impact_values.items():
            if avg_impact <= value + 0.5:
                return level
        
        return "severe"
    
    def _assess_scene_complexity(self, obstacles: List[ObstacleInfo], 
                               person_detections: List[Dict[str, Any]]) -> str:
        """评估场景复杂度"""
        obstacle_count = len(obstacles)
        person_count = len(person_detections)
        
        # 简单的复杂度评估
        complexity_score = obstacle_count * 0.3 + person_count * 0.7
        
        if complexity_score < 2:
            return "simple"
        elif complexity_score < 4:
            return "moderate"
        elif complexity_score < 6:
            return "complex"
        else:
            return "very_complex"
    
    def get_tracking_statistics(self) -> Dict[str, Any]:
        """获取跟踪统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            'total_frames': self.stats['total_frames'],
            'obstacles_detected': self.stats['obstacles_detected'],
            'tracking_adaptations': self.stats['tracking_adaptations'],
            'average_obstacles_per_frame': (self.stats['obstacles_detected'] / 
                                          self.stats['total_frames'] 
                                          if self.stats['total_frames'] > 0 else 0),
            'obstacle_type_distribution': dict(self.stats['obstacle_types']),
            'impact_level_distribution': dict(self.stats['impact_levels']),
            'strategy_usage': dict(self.stats['strategies_used']),
            'configuration': {
                'detection_params': self.config.detection_params,
                'impact_params': self.config.impact_params,
                'adaptation_params': self.config.adaptation_params
            }
        }
    
    def export_tracking_report(self, output_path: str) -> bool:
        """导出跟踪报告
        
        Args:
            output_path: 输出路径
            
        Returns:
            bool: 是否成功
        """
        try:
            report = {
                'timestamp': time.time(),
                'statistics': self.get_tracking_statistics(),
                'performance_metrics': {
                    'frames_processed': self.stats['total_frames'],
                    'obstacles_detected': self.stats['obstacles_detected'],
                    'adaptations_made': self.stats['tracking_adaptations'],
                    'detection_rate': (self.stats['obstacles_detected'] / 
                                     self.stats['total_frames'] 
                                     if self.stats['total_frames'] > 0 else 0),
                    'adaptation_efficiency': self._calculate_adaptation_efficiency()
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"跟踪报告已导出到: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"报告导出失败: {e}")
            return False
    
    def _calculate_adaptation_efficiency(self) -> float:
        """计算适应效率"""
        if self.stats['total_frames'] == 0:
            return 0.0
        
        # 简化的效率计算
        adaptation_rate = self.stats['tracking_adaptations'] / self.stats['total_frames']
        
        # 归一化到0-1范围
        efficiency = min(1.0, adaptation_rate)
        
        return efficiency
    
    def cleanup(self) -> None:
        """清理资源"""
        try:
            self.executor.shutdown(wait=True)
            self.logger.info("障碍物感知跟踪器资源清理完成")
            
        except Exception as e:
            self.logger.error(f"资源清理失败: {e}")


# 测试代码
if __name__ == "__main__":
    import numpy as np
    
    def test_obstacle_aware_tracker():
        """测试障碍物感知跟踪器"""
        print("=== 障碍物感知跟踪器测试 ===")
        
        # 创建跟踪器
        tracker = ObstacleAwareTracker()
        
        # 模拟输入数据
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 模拟人体检测结果
        person_detections = [
            {
                'person_id': 1,
                'bbox': (150, 100, 250, 300),
                'position': (200, 200),
                'velocity': (2.0, 1.0),
                'confidence': 0.9
            },
            {
                'person_id': 2,
                'bbox': (350, 120, 450, 320),
                'position': (400, 220),
                'velocity': (-1.0, 0.5),
                'confidence': 0.85
            }
        ]
        
        print(f"输入: {len(person_detections)} 个人体检测结果")
        for detection in person_detections:
            print(f"  人员 {detection['person_id']}: 位置 {detection['position']}, 置信度 {detection['confidence']}")
        
        # 处理帧
        result = tracker.process_frame(frame, person_detections)
        
        if 'error' in result:
            print(f"处理失败: {result['error']}")
            return
        
        # 显示结果
        print(f"\n检测到 {result['frame_info']['obstacles_detected']} 个障碍物")
        
        for obstacle in result['obstacles']:
            print(f"  障碍物 {obstacle['id']}: {obstacle['type']}, 置信度 {obstacle['confidence']:.2f}")
        
        print(f"\n跟踪结果:")
        for tracking_result in result['tracking_results']:
            person_id = tracking_result['person_id']
            impact = tracking_result['impact_level']
            strategy = tracking_result['tracking_context']['strategy']
            difficulty = tracking_result['tracking_context']['difficulty']
            
            print(f"  人员 {person_id}:")
            print(f"    影响等级: {impact}")
            print(f"    跟踪难度: {difficulty}")
            print(f"    推荐策略: {strategy}")
            print(f"    附近障碍物: {len(tracking_result['nearby_obstacles'])} 个")
            
            recommendations = tracking_result['recommendations']
            if recommendations.get('suggested_actions'):
                print(f"    建议行动: {', '.join(recommendations['suggested_actions'])}")
        
        # 场景分析
        scene = result['scene_analysis']
        print(f"\n场景分析:")
        print(f"  障碍物密度: {scene['obstacle_density']:.3f}")
        print(f"  主要障碍物类型: {scene['dominant_obstacle_type']}")
        print(f"  平均影响等级: {scene['average_impact_level']}")
        print(f"  跟踪复杂度: {scene['tracking_complexity']}")
        
        # 获取统计信息
        stats = tracker.get_tracking_statistics()
        print(f"\n跟踪统计:")
        print(f"  处理帧数: {stats['total_frames']}")
        print(f"  检测障碍物数: {stats['obstacles_detected']}")
        print(f"  跟踪适应次数: {stats['tracking_adaptations']}")
        print(f"  平均每帧障碍物数: {stats['average_obstacles_per_frame']:.2f}")
        
        # 导出报告
        report_path = "obstacle_tracking_report.json"
        if tracker.export_tracking_report(report_path):
            print(f"\n跟踪报告已导出到: {report_path}")
        
        # 清理资源
        tracker.cleanup()
        print("\n测试完成")
    
    def test_complex_scenarios():
        """测试复杂场景"""
        print("\n=== 复杂场景测试 ===")
        
        tracker = ObstacleAwareTracker()
        
        # 场景1: 高障碍物密度
        print("\n场景1: 高障碍物密度场景")
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 添加一些模拟障碍物区域（通过修改像素值）
        # 模拟玩具
        frame[100:150, 50:100] = [255, 0, 0]  # 红色玩具
        frame[200:230, 150:200] = [0, 255, 0]  # 绿色玩具
        
        # 模拟家具
        frame[300:400, 400:500] = [139, 69, 19]  # 棕色家具
        
        person_detections = [
            {
                'person_id': 1,
                'bbox': (120, 80, 180, 250),
                'position': (150, 165),
                'velocity': (1.5, 0.8),
                'confidence': 0.88
            }
        ]
        
        result1 = tracker.process_frame(frame, person_detections)
        print(f"高密度场景: 检测到 {result1['frame_info']['obstacles_detected']} 个障碍物")
        print(f"场景复杂度: {result1['scene_analysis']['tracking_complexity']}")
        
        # 场景2: 多人多障碍物
        print("\n场景2: 多人多障碍物场景")
        multi_person_detections = [
            {
                'person_id': 1,
                'bbox': (100, 80, 200, 280),
                'position': (150, 180),
                'velocity': (2.0, 1.0),
                'confidence': 0.92
            },
            {
                'person_id': 2,
                'bbox': (300, 100, 400, 300),
                'position': (350, 200),
                'velocity': (-1.5, 0.5),
                'confidence': 0.87
            },
            {
                'person_id': 3,
                'bbox': (500, 120, 580, 320),
                'position': (540, 220),
                'velocity': (0.5, -1.0),
                'confidence': 0.79
            }
        ]
        
        result2 = tracker.process_frame(frame, multi_person_detections)
        print(f"多人场景: 跟踪 {len(result2['tracking_results'])} 个人员")
        
        for i, tracking_result in enumerate(result2['tracking_results']):
            person_id = tracking_result['person_id']
            impact = tracking_result['impact_level']
            strategy = tracking_result['tracking_context']['strategy']
            print(f"  人员 {person_id}: 影响等级 {impact}, 策略 {strategy}")
        
        # 场景3: 严重遮挡场景
        print("\n场景3: 严重遮挡场景")
        # 模拟大型障碍物（书柜）
        frame[50:350, 200:280] = [101, 67, 33]  # 深棕色书柜
        
        occluded_person_detections = [
            {
                'person_id': 1,
                'bbox': (180, 100, 300, 300),  # 与书柜重叠
                'position': (240, 200),
                'velocity': (0.5, 0.2),
                'confidence': 0.65  # 较低置信度
            }
        ]
        
        result3 = tracker.process_frame(frame, occluded_person_detections)
        if result3['tracking_results']:
            tracking_result = result3['tracking_results'][0]
            print(f"遮挡场景: 影响等级 {tracking_result['impact_level']}")
            print(f"跟踪难度: {tracking_result['tracking_context']['difficulty']}")
            print(f"置信度惩罚: {tracking_result['tracking_context']['confidence_penalty']:.2f}")
        
        # 获取最终统计
        final_stats = tracker.get_tracking_statistics()
        print(f"\n最终统计:")
        print(f"  总处理帧数: {final_stats['total_frames']}")
        print(f"  总检测障碍物数: {final_stats['obstacles_detected']}")
        print(f"  总适应次数: {final_stats['tracking_adaptations']}")
        
        if final_stats['obstacle_type_distribution']:
            print(f"  障碍物类型分布: {final_stats['obstacle_type_distribution']}")
        
        if final_stats['strategy_usage']:
            print(f"  策略使用情况: {final_stats['strategy_usage']}")
        
        # 清理资源
        tracker.cleanup()
        print("\n复杂场景测试完成")
    
    # 运行测试
    test_obstacle_aware_tracker()
    test_complex_scenarios()