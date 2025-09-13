#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境上下文分析模块

集成室内布局分析和安全区域识别功能
为复杂场景跌倒检测提供环境上下文信息

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


class RoomType(Enum):
    """房间类型"""
    LIVING_ROOM = "living_room"     # 客厅
    BEDROOM = "bedroom"             # 卧室
    KITCHEN = "kitchen"             # 厨房
    BATHROOM = "bathroom"           # 浴室
    STUDY = "study"                 # 书房
    HALLWAY = "hallway"             # 走廊
    UNKNOWN = "unknown"             # 未知


class SafetyLevel(Enum):
    """安全等级"""
    VERY_SAFE = "very_safe"         # 非常安全
    SAFE = "safe"                   # 安全
    MODERATE = "moderate"           # 中等
    RISKY = "risky"                 # 有风险
    DANGEROUS = "dangerous"         # 危险


class AreaType(Enum):
    """区域类型"""
    SAFE_ZONE = "safe_zone"         # 安全区域
    CAUTION_ZONE = "caution_zone"   # 注意区域
    DANGER_ZONE = "danger_zone"     # 危险区域
    OBSTACLE_ZONE = "obstacle_zone" # 障碍物区域
    FURNITURE_ZONE = "furniture_zone" # 家具区域
    WALKWAY = "walkway"             # 通道
    RESTRICTED = "restricted"       # 限制区域


class LayoutFeature(Enum):
    """布局特征"""
    OPEN_SPACE = "open_space"       # 开放空间
    NARROW_PASSAGE = "narrow_passage" # 狭窄通道
    CORNER_AREA = "corner_area"     # 角落区域
    CENTRAL_AREA = "central_area"   # 中央区域
    WALL_ADJACENT = "wall_adjacent" # 靠墙区域
    FURNITURE_CLUSTER = "furniture_cluster" # 家具聚集区
    CLEAR_PATH = "clear_path"       # 清晰路径


@dataclass
class SafetyZone:
    """安全区域信息"""
    zone_id: int
    area_type: AreaType
    safety_level: SafetyLevel
    polygon: List[Tuple[int, int]]  # 区域多边形顶点
    center_point: Tuple[float, float]
    area_size: float
    risk_factors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 0.8
    
    def contains_point(self, point: Tuple[float, float]) -> bool:
        """检查点是否在区域内"""
        try:
            # 使用射线法判断点是否在多边形内
            x, y = point
            n = len(self.polygon)
            inside = False
            
            p1x, p1y = self.polygon[0]
            for i in range(1, n + 1):
                p2x, p2y = self.polygon[i % n]
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xinters:
                                inside = not inside
                p1x, p1y = p2x, p2y
            
            return inside
        except Exception:
            return False
    
    def distance_to_point(self, point: Tuple[float, float]) -> float:
        """计算点到区域中心的距离"""
        return math.sqrt(
            (self.center_point[0] - point[0])**2 + 
            (self.center_point[1] - point[1])**2
        )


@dataclass
class LayoutAnalysis:
    """布局分析结果"""
    room_type: RoomType
    layout_features: List[LayoutFeature]
    furniture_layout: Dict[str, List[Tuple[int, int, int, int]]]  # 家具类型 -> 边界框列表
    walkable_areas: List[List[Tuple[int, int]]]  # 可行走区域多边形列表
    obstacle_density: float
    space_utilization: float
    accessibility_score: float
    fall_risk_areas: List[Tuple[int, int, int, int]]  # 跌倒风险区域
    
    def get_layout_complexity(self) -> str:
        """获取布局复杂度"""
        complexity_score = len(self.layout_features) * 0.3 + self.obstacle_density * 0.7
        
        if complexity_score < 0.3:
            return "simple"
        elif complexity_score < 0.6:
            return "moderate"
        elif complexity_score < 0.8:
            return "complex"
        else:
            return "very_complex"


@dataclass
class EnvironmentContext:
    """环境上下文"""
    timestamp: float
    frame_size: Tuple[int, int]
    room_analysis: LayoutAnalysis
    safety_zones: List[SafetyZone]
    environmental_factors: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    monitoring_recommendations: List[str]
    
    def get_zone_for_point(self, point: Tuple[float, float]) -> Optional[SafetyZone]:
        """获取点所在的安全区域"""
        for zone in self.safety_zones:
            if zone.contains_point(point):
                return zone
        return None
    
    def get_overall_safety_level(self) -> SafetyLevel:
        """获取整体安全等级"""
        if not self.safety_zones:
            return SafetyLevel.MODERATE
        
        # 基于最危险区域确定整体安全等级
        min_safety = min(zone.safety_level for zone in self.safety_zones)
        return min_safety


class RoomClassifier:
    """房间分类器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 房间特征模板
        self.room_features = {
            RoomType.LIVING_ROOM: {
                'typical_furniture': ['sofa', 'tv', 'coffee_table', 'bookshelf'],
                'space_characteristics': ['open_space', 'central_area'],
                'size_range': (20, 100),  # 相对面积
                'furniture_density': (0.2, 0.6)
            },
            RoomType.BEDROOM: {
                'typical_furniture': ['bed', 'wardrobe', 'nightstand'],
                'space_characteristics': ['wall_adjacent', 'corner_area'],
                'size_range': (15, 50),
                'furniture_density': (0.3, 0.7)
            },
            RoomType.KITCHEN: {
                'typical_furniture': ['cabinet', 'counter', 'appliance'],
                'space_characteristics': ['narrow_passage', 'furniture_cluster'],
                'size_range': (10, 40),
                'furniture_density': (0.5, 0.9)
            },
            RoomType.STUDY: {
                'typical_furniture': ['desk', 'bookshelf', 'chair'],
                'space_characteristics': ['corner_area', 'furniture_cluster'],
                'size_range': (8, 30),
                'furniture_density': (0.4, 0.8)
            }
        }
    
    def classify_room(self, furniture_layout: Dict[str, List[Tuple[int, int, int, int]]], 
                     frame_size: Tuple[int, int]) -> RoomType:
        """分类房间类型
        
        Args:
            furniture_layout: 家具布局
            frame_size: 帧尺寸
            
        Returns:
            RoomType: 房间类型
        """
        try:
            # 计算房间特征
            total_area = frame_size[0] * frame_size[1]
            furniture_types = set(furniture_layout.keys())
            
            # 计算家具密度
            furniture_area = 0
            for furniture_list in furniture_layout.values():
                for bbox in furniture_list:
                    furniture_area += (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            furniture_density = furniture_area / total_area if total_area > 0 else 0
            
            # 匹配房间类型
            best_match = RoomType.UNKNOWN
            best_score = 0.0
            
            for room_type, features in self.room_features.items():
                score = self._calculate_room_match_score(
                    furniture_types, furniture_density, features
                )
                
                if score > best_score:
                    best_score = score
                    best_match = room_type
            
            # 如果匹配分数太低，返回未知
            if best_score < 0.3:
                return RoomType.UNKNOWN
            
            return best_match
            
        except Exception as e:
            self.logger.error(f"房间分类失败: {e}")
            return RoomType.UNKNOWN
    
    def _calculate_room_match_score(self, furniture_types: set, 
                                  furniture_density: float, 
                                  room_features: Dict[str, Any]) -> float:
        """计算房间匹配分数"""
        score = 0.0
        
        # 家具类型匹配
        typical_furniture = set(room_features['typical_furniture'])
        furniture_match = len(furniture_types & typical_furniture) / len(typical_furniture)
        score += furniture_match * 0.6
        
        # 家具密度匹配
        density_range = room_features['furniture_density']
        if density_range[0] <= furniture_density <= density_range[1]:
            density_score = 1.0
        else:
            # 计算偏离程度
            if furniture_density < density_range[0]:
                density_score = furniture_density / density_range[0]
            else:
                density_score = density_range[1] / furniture_density
        
        score += density_score * 0.4
        
        return min(1.0, score)


class LayoutAnalyzer:
    """布局分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.room_classifier = RoomClassifier()
    
    def analyze_layout(self, frame: np.ndarray, 
                      furniture_detections: List[Dict[str, Any]],
                      obstacle_detections: List[Dict[str, Any]]) -> LayoutAnalysis:
        """分析室内布局
        
        Args:
            frame: 输入帧
            furniture_detections: 家具检测结果
            obstacle_detections: 障碍物检测结果
            
        Returns:
            LayoutAnalysis: 布局分析结果
        """
        try:
            frame_height, frame_width = frame.shape[:2]
            
            # 1. 整理家具布局
            furniture_layout = self._organize_furniture_layout(furniture_detections)
            
            # 2. 分类房间类型
            room_type = self.room_classifier.classify_room(
                furniture_layout, (frame_width, frame_height)
            )
            
            # 3. 分析布局特征
            layout_features = self._analyze_layout_features(
                furniture_layout, (frame_width, frame_height)
            )
            
            # 4. 计算可行走区域
            walkable_areas = self._calculate_walkable_areas(
                frame, furniture_detections, obstacle_detections
            )
            
            # 5. 计算各种指标
            obstacle_density = self._calculate_obstacle_density(
                furniture_detections + obstacle_detections, (frame_width, frame_height)
            )
            
            space_utilization = self._calculate_space_utilization(
                furniture_layout, (frame_width, frame_height)
            )
            
            accessibility_score = self._calculate_accessibility_score(
                walkable_areas, (frame_width, frame_height)
            )
            
            # 6. 识别跌倒风险区域
            fall_risk_areas = self._identify_fall_risk_areas(
                furniture_detections, obstacle_detections, walkable_areas
            )
            
            return LayoutAnalysis(
                room_type=room_type,
                layout_features=layout_features,
                furniture_layout=furniture_layout,
                walkable_areas=walkable_areas,
                obstacle_density=obstacle_density,
                space_utilization=space_utilization,
                accessibility_score=accessibility_score,
                fall_risk_areas=fall_risk_areas
            )
            
        except Exception as e:
            self.logger.error(f"布局分析失败: {e}")
            # 返回默认分析结果
            return LayoutAnalysis(
                room_type=RoomType.UNKNOWN,
                layout_features=[],
                furniture_layout={},
                walkable_areas=[],
                obstacle_density=0.0,
                space_utilization=0.0,
                accessibility_score=0.5,
                fall_risk_areas=[]
            )
    
    def _organize_furniture_layout(self, furniture_detections: List[Dict[str, Any]]) -> Dict[str, List[Tuple[int, int, int, int]]]:
        """整理家具布局"""
        layout = defaultdict(list)
        
        for detection in furniture_detections:
            furniture_type = detection.get('type', 'unknown')
            bbox = detection.get('bbox', (0, 0, 0, 0))
            layout[furniture_type].append(bbox)
        
        return dict(layout)
    
    def _analyze_layout_features(self, furniture_layout: Dict[str, List[Tuple[int, int, int, int]]], 
                               frame_size: Tuple[int, int]) -> List[LayoutFeature]:
        """分析布局特征"""
        features = []
        frame_width, frame_height = frame_size
        
        # 计算家具分布
        all_furniture = []
        for furniture_list in furniture_layout.values():
            all_furniture.extend(furniture_list)
        
        if not all_furniture:
            features.append(LayoutFeature.OPEN_SPACE)
            return features
        
        # 分析空间特征
        furniture_centers = []
        for bbox in all_furniture:
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            furniture_centers.append((center_x, center_y))
        
        # 检查是否有开放空间
        if len(all_furniture) < 3:
            features.append(LayoutFeature.OPEN_SPACE)
        
        # 检查家具聚集
        if self._has_furniture_cluster(furniture_centers, frame_size):
            features.append(LayoutFeature.FURNITURE_CLUSTER)
        
        # 检查中央区域
        center_x, center_y = frame_width / 2, frame_height / 2
        central_furniture = sum(1 for x, y in furniture_centers 
                              if abs(x - center_x) < frame_width * 0.2 and 
                                 abs(y - center_y) < frame_height * 0.2)
        
        if central_furniture > 0:
            features.append(LayoutFeature.CENTRAL_AREA)
        
        # 检查靠墙区域
        wall_furniture = sum(1 for bbox in all_furniture 
                           if (bbox[0] < frame_width * 0.1 or bbox[2] > frame_width * 0.9 or
                               bbox[1] < frame_height * 0.1 or bbox[3] > frame_height * 0.9))
        
        if wall_furniture > len(all_furniture) * 0.5:
            features.append(LayoutFeature.WALL_ADJACENT)
        
        # 检查角落区域
        corner_furniture = sum(1 for bbox in all_furniture 
                             if ((bbox[0] < frame_width * 0.2 and bbox[1] < frame_height * 0.2) or
                                 (bbox[2] > frame_width * 0.8 and bbox[1] < frame_height * 0.2) or
                                 (bbox[0] < frame_width * 0.2 and bbox[3] > frame_height * 0.8) or
                                 (bbox[2] > frame_width * 0.8 and bbox[3] > frame_height * 0.8)))
        
        if corner_furniture > 0:
            features.append(LayoutFeature.CORNER_AREA)
        
        return features
    
    def _has_furniture_cluster(self, furniture_centers: List[Tuple[float, float]], 
                             frame_size: Tuple[int, int]) -> bool:
        """检查是否有家具聚集"""
        if len(furniture_centers) < 2:
            return False
        
        frame_width, frame_height = frame_size
        cluster_threshold = min(frame_width, frame_height) * 0.3
        
        for i, (x1, y1) in enumerate(furniture_centers):
            nearby_count = 0
            for j, (x2, y2) in enumerate(furniture_centers):
                if i != j:
                    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if distance < cluster_threshold:
                        nearby_count += 1
            
            if nearby_count >= 2:  # 至少3个家具聚集
                return True
        
        return False
    
    def _calculate_walkable_areas(self, frame: np.ndarray, 
                                furniture_detections: List[Dict[str, Any]],
                                obstacle_detections: List[Dict[str, Any]]) -> List[List[Tuple[int, int]]]:
        """计算可行走区域"""
        try:
            frame_height, frame_width = frame.shape[:2]
            
            # 创建占用地图
            occupancy_map = np.zeros((frame_height, frame_width), dtype=np.uint8)
            
            # 标记家具和障碍物区域
            all_detections = furniture_detections + obstacle_detections
            for detection in all_detections:
                bbox = detection.get('bbox', (0, 0, 0, 0))
                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame_width, x2), min(frame_height, y2)
                occupancy_map[y1:y2, x1:x2] = 255
            
            # 形态学操作，扩展障碍物区域（安全边距）
            kernel = np.ones((10, 10), np.uint8)
            occupancy_map = cv2.dilate(occupancy_map, kernel, iterations=1)
            
            # 查找可行走区域轮廓
            free_space = 255 - occupancy_map
            contours, _ = cv2.findContours(free_space, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 转换轮廓为多边形
            walkable_areas = []
            for contour in contours:
                # 过滤太小的区域
                area = cv2.contourArea(contour)
                if area > 1000:  # 最小可行走区域
                    # 简化轮廓
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # 转换为点列表
                    polygon = [(int(point[0][0]), int(point[0][1])) for point in approx]
                    walkable_areas.append(polygon)
            
            return walkable_areas
            
        except Exception as e:
            self.logger.error(f"可行走区域计算失败: {e}")
            return []
    
    def _calculate_obstacle_density(self, all_detections: List[Dict[str, Any]], 
                                  frame_size: Tuple[int, int]) -> float:
        """计算障碍物密度"""
        if not all_detections:
            return 0.0
        
        frame_width, frame_height = frame_size
        total_area = frame_width * frame_height
        
        occupied_area = 0
        for detection in all_detections:
            bbox = detection.get('bbox', (0, 0, 0, 0))
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            occupied_area += area
        
        return min(1.0, occupied_area / total_area)
    
    def _calculate_space_utilization(self, furniture_layout: Dict[str, List[Tuple[int, int, int, int]]], 
                                   frame_size: Tuple[int, int]) -> float:
        """计算空间利用率"""
        frame_width, frame_height = frame_size
        total_area = frame_width * frame_height
        
        furniture_area = 0
        for furniture_list in furniture_layout.values():
            for bbox in furniture_list:
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                furniture_area += area
        
        return min(1.0, furniture_area / total_area)
    
    def _calculate_accessibility_score(self, walkable_areas: List[List[Tuple[int, int]]], 
                                     frame_size: Tuple[int, int]) -> float:
        """计算可达性分数"""
        if not walkable_areas:
            return 0.0
        
        frame_width, frame_height = frame_size
        total_area = frame_width * frame_height
        
        # 计算可行走区域总面积
        walkable_area = 0
        for polygon in walkable_areas:
            if len(polygon) >= 3:
                # 使用鞋带公式计算多边形面积
                area = 0
                n = len(polygon)
                for i in range(n):
                    j = (i + 1) % n
                    area += polygon[i][0] * polygon[j][1]
                    area -= polygon[j][0] * polygon[i][1]
                walkable_area += abs(area) / 2
        
        accessibility = walkable_area / total_area
        return min(1.0, accessibility)
    
    def _identify_fall_risk_areas(self, furniture_detections: List[Dict[str, Any]],
                                obstacle_detections: List[Dict[str, Any]],
                                walkable_areas: List[List[Tuple[int, int]]]) -> List[Tuple[int, int, int, int]]:
        """识别跌倒风险区域"""
        risk_areas = []
        
        try:
            # 1. 家具边缘区域
            for detection in furniture_detections:
                bbox = detection.get('bbox', (0, 0, 0, 0))
                furniture_type = detection.get('type', 'unknown')
                
                # 某些家具类型风险更高
                high_risk_furniture = ['table', 'chair', 'bed', 'sofa']
                if furniture_type in high_risk_furniture:
                    # 扩展边界框作为风险区域
                    margin = 20
                    risk_bbox = (
                        max(0, bbox[0] - margin),
                        max(0, bbox[1] - margin),
                        bbox[2] + margin,
                        bbox[3] + margin
                    )
                    risk_areas.append(risk_bbox)
            
            # 2. 狭窄通道
            for polygon in walkable_areas:
                if len(polygon) >= 4:
                    # 计算多边形的最小边界矩形
                    xs = [p[0] for p in polygon]
                    ys = [p[1] for p in polygon]
                    
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    
                    width = max_x - min_x
                    height = max_y - min_y
                    
                    # 如果通道很窄，标记为风险区域
                    if width < 80 or height < 80:  # 像素阈值
                        risk_areas.append((min_x, min_y, max_x, max_y))
            
            # 3. 障碍物密集区域
            if len(obstacle_detections) > 2:
                # 计算障碍物聚集区域
                obstacle_centers = []
                for detection in obstacle_detections:
                    bbox = detection.get('bbox', (0, 0, 0, 0))
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    obstacle_centers.append((center_x, center_y))
                
                # 查找聚集区域
                cluster_areas = self._find_obstacle_clusters(obstacle_centers)
                risk_areas.extend(cluster_areas)
            
            return risk_areas
            
        except Exception as e:
            self.logger.error(f"跌倒风险区域识别失败: {e}")
            return []
    
    def _find_obstacle_clusters(self, obstacle_centers: List[Tuple[float, float]]) -> List[Tuple[int, int, int, int]]:
        """查找障碍物聚集区域"""
        clusters = []
        cluster_threshold = 100.0  # 聚集距离阈值
        
        visited = [False] * len(obstacle_centers)
        
        for i, center in enumerate(obstacle_centers):
            if visited[i]:
                continue
            
            # 开始新聚集
            cluster_points = [center]
            visited[i] = True
            
            # 查找附近的障碍物
            for j, other_center in enumerate(obstacle_centers):
                if not visited[j]:
                    distance = math.sqrt(
                        (center[0] - other_center[0])**2 + 
                        (center[1] - other_center[1])**2
                    )
                    
                    if distance < cluster_threshold:
                        cluster_points.append(other_center)
                        visited[j] = True
            
            # 如果聚集包含多个障碍物，创建风险区域
            if len(cluster_points) >= 2:
                xs = [p[0] for p in cluster_points]
                ys = [p[1] for p in cluster_points]
                
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                
                # 扩展边界
                margin = 30
                cluster_bbox = (
                    int(max(0, min_x - margin)),
                    int(max(0, min_y - margin)),
                    int(max_x + margin),
                    int(max_y + margin)
                )
                
                clusters.append(cluster_bbox)
        
        return clusters


class SafetyZoneGenerator:
    """安全区域生成器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.zone_id_counter = 1
    
    def generate_safety_zones(self, layout_analysis: LayoutAnalysis, 
                            frame_size: Tuple[int, int]) -> List[SafetyZone]:
        """生成安全区域
        
        Args:
            layout_analysis: 布局分析结果
            frame_size: 帧尺寸
            
        Returns:
            List[SafetyZone]: 安全区域列表
        """
        try:
            zones = []
            frame_width, frame_height = frame_size
            
            # 1. 基于可行走区域生成安全区域
            for walkable_area in layout_analysis.walkable_areas:
                if len(walkable_area) >= 3:
                    zone = self._create_zone_from_walkable_area(
                        walkable_area, layout_analysis
                    )
                    if zone:
                        zones.append(zone)
            
            # 2. 基于跌倒风险区域生成危险区域
            for risk_area in layout_analysis.fall_risk_areas:
                zone = self._create_risk_zone(risk_area)
                if zone:
                    zones.append(zone)
            
            # 3. 基于家具布局生成家具区域
            for furniture_type, furniture_list in layout_analysis.furniture_layout.items():
                for furniture_bbox in furniture_list:
                    zone = self._create_furniture_zone(furniture_bbox, furniture_type)
                    if zone:
                        zones.append(zone)
            
            # 4. 如果没有生成任何区域，创建默认区域
            if not zones:
                default_zone = self._create_default_zone(frame_size)
                zones.append(default_zone)
            
            return zones
            
        except Exception as e:
            self.logger.error(f"安全区域生成失败: {e}")
            return []
    
    def _create_zone_from_walkable_area(self, walkable_area: List[Tuple[int, int]], 
                                      layout_analysis: LayoutAnalysis) -> Optional[SafetyZone]:
        """从可行走区域创建安全区域"""
        try:
            # 计算区域中心
            xs = [p[0] for p in walkable_area]
            ys = [p[1] for p in walkable_area]
            center_x = sum(xs) / len(xs)
            center_y = sum(ys) / len(ys)
            
            # 计算区域面积
            area = self._calculate_polygon_area(walkable_area)
            
            # 评估安全等级
            safety_level = self._assess_walkable_area_safety(
                walkable_area, area, layout_analysis
            )
            
            # 生成建议
            recommendations = self._generate_walkable_area_recommendations(
                safety_level, area
            )
            
            zone = SafetyZone(
                zone_id=self.zone_id_counter,
                area_type=AreaType.SAFE_ZONE if safety_level in [SafetyLevel.SAFE, SafetyLevel.VERY_SAFE] else AreaType.CAUTION_ZONE,
                safety_level=safety_level,
                polygon=walkable_area,
                center_point=(center_x, center_y),
                area_size=area,
                risk_factors=[],
                recommendations=recommendations,
                confidence=0.8
            )
            
            self.zone_id_counter += 1
            return zone
            
        except Exception as e:
            self.logger.error(f"可行走区域安全区域创建失败: {e}")
            return None
    
    def _create_risk_zone(self, risk_area: Tuple[int, int, int, int]) -> Optional[SafetyZone]:
        """创建风险区域"""
        try:
            x1, y1, x2, y2 = risk_area
            
            # 转换为多边形
            polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            
            # 计算中心和面积
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            area = (x2 - x1) * (y2 - y1)
            
            # 风险因子
            risk_factors = ["跌倒风险区域", "需要特别注意"]
            
            # 建议
            recommendations = [
                "增加监控密度",
                "设置警告标识",
                "考虑重新布置家具"
            ]
            
            zone = SafetyZone(
                zone_id=self.zone_id_counter,
                area_type=AreaType.DANGER_ZONE,
                safety_level=SafetyLevel.RISKY,
                polygon=polygon,
                center_point=(center_x, center_y),
                area_size=area,
                risk_factors=risk_factors,
                recommendations=recommendations,
                confidence=0.7
            )
            
            self.zone_id_counter += 1
            return zone
            
        except Exception as e:
            self.logger.error(f"风险区域创建失败: {e}")
            return None
    
    def _create_furniture_zone(self, furniture_bbox: Tuple[int, int, int, int], 
                             furniture_type: str) -> Optional[SafetyZone]:
        """创建家具区域"""
        try:
            x1, y1, x2, y2 = furniture_bbox
            
            # 转换为多边形
            polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            
            # 计算中心和面积
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            area = (x2 - x1) * (y2 - y1)
            
            # 基于家具类型评估安全等级
            safety_level = self._assess_furniture_safety(furniture_type)
            
            # 风险因子和建议
            risk_factors, recommendations = self._get_furniture_risk_info(furniture_type)
            
            zone = SafetyZone(
                zone_id=self.zone_id_counter,
                area_type=AreaType.FURNITURE_ZONE,
                safety_level=safety_level,
                polygon=polygon,
                center_point=(center_x, center_y),
                area_size=area,
                risk_factors=risk_factors,
                recommendations=recommendations,
                confidence=0.6
            )
            
            self.zone_id_counter += 1
            return zone
            
        except Exception as e:
            self.logger.error(f"家具区域创建失败: {e}")
            return None
    
    def _create_default_zone(self, frame_size: Tuple[int, int]) -> SafetyZone:
        """创建默认区域"""
        frame_width, frame_height = frame_size
        
        # 整个帧作为默认区域
        polygon = [(0, 0), (frame_width, 0), (frame_width, frame_height), (0, frame_height)]
        
        zone = SafetyZone(
            zone_id=self.zone_id_counter,
            area_type=AreaType.SAFE_ZONE,
            safety_level=SafetyLevel.MODERATE,
            polygon=polygon,
            center_point=(frame_width / 2, frame_height / 2),
            area_size=frame_width * frame_height,
            risk_factors=["未知环境"],
            recommendations=["需要进一步分析环境"],
            confidence=0.5
        )
        
        self.zone_id_counter += 1
        return zone
    
    def _calculate_polygon_area(self, polygon: List[Tuple[int, int]]) -> float:
        """计算多边形面积"""
        if len(polygon) < 3:
            return 0.0
        
        area = 0
        n = len(polygon)
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]
        
        return abs(area) / 2
    
    def _assess_walkable_area_safety(self, walkable_area: List[Tuple[int, int]], 
                                   area: float, 
                                   layout_analysis: LayoutAnalysis) -> SafetyLevel:
        """评估可行走区域安全等级"""
        # 基于面积大小
        if area > 10000:  # 大面积
            base_safety = SafetyLevel.SAFE
        elif area > 5000:  # 中等面积
            base_safety = SafetyLevel.MODERATE
        else:  # 小面积
            base_safety = SafetyLevel.RISKY
        
        # 基于障碍物密度调整
        if layout_analysis.obstacle_density > 0.7:
            # 降低安全等级
            if base_safety == SafetyLevel.SAFE:
                base_safety = SafetyLevel.MODERATE
            elif base_safety == SafetyLevel.MODERATE:
                base_safety = SafetyLevel.RISKY
        elif layout_analysis.obstacle_density < 0.3:
            # 提高安全等级
            if base_safety == SafetyLevel.RISKY:
                base_safety = SafetyLevel.MODERATE
            elif base_safety == SafetyLevel.MODERATE:
                base_safety = SafetyLevel.SAFE
        
        return base_safety
    
    def _generate_walkable_area_recommendations(self, safety_level: SafetyLevel, 
                                              area: float) -> List[str]:
        """生成可行走区域建议"""
        recommendations = []
        
        if safety_level == SafetyLevel.VERY_SAFE:
            recommendations.append("理想的监控区域")
        elif safety_level == SafetyLevel.SAFE:
            recommendations.append("适合正常监控")
        elif safety_level == SafetyLevel.MODERATE:
            recommendations.extend(["需要适度关注", "考虑增加监控点"])
        elif safety_level == SafetyLevel.RISKY:
            recommendations.extend(["需要密切监控", "考虑清理障碍物"])
        else:  # DANGEROUS
            recommendations.extend(["高风险区域", "建议重新布置", "增加安全措施"])
        
        if area < 2000:
            recommendations.append("空间较小，需要特别注意")
        
        return recommendations
    
    def _assess_furniture_safety(self, furniture_type: str) -> SafetyLevel:
        """评估家具安全等级"""
        # 基于家具类型的风险评估
        high_risk_furniture = ['table', 'chair', 'stool']
        medium_risk_furniture = ['sofa', 'bed', 'cabinet']
        low_risk_furniture = ['bookshelf', 'wardrobe', 'tv_stand']
        
        if furniture_type in high_risk_furniture:
            return SafetyLevel.RISKY
        elif furniture_type in medium_risk_furniture:
            return SafetyLevel.MODERATE
        elif furniture_type in low_risk_furniture:
            return SafetyLevel.SAFE
        else:
            return SafetyLevel.MODERATE
    
    def _get_furniture_risk_info(self, furniture_type: str) -> Tuple[List[str], List[str]]:
        """获取家具风险信息"""
        furniture_info = {
            'table': {
                'risks': ['桌角碰撞风险', '绊倒风险'],
                'recommendations': ['安装防撞角', '保持桌面整洁']
            },
            'chair': {
                'risks': ['移动障碍', '绊倒风险'],
                'recommendations': ['固定位置', '及时归位']
            },
            'sofa': {
                'risks': ['视线遮挡'],
                'recommendations': ['调整监控角度']
            },
            'bed': {
                'risks': ['床边跌倒风险'],
                'recommendations': ['床边防护', '夜间照明']
            }
        }
        
        info = furniture_info.get(furniture_type, {
            'risks': ['一般家具风险'],
            'recommendations': ['保持整洁']
        })
        
        return info['risks'], info['recommendations']


class EnvironmentContextAnalyzer:
    """环境上下文分析器主类"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.layout_analyzer = LayoutAnalyzer()
        self.safety_zone_generator = SafetyZoneGenerator()
        
        # 统计信息
        self.stats = {
            'total_analyses': 0,
            'room_types': defaultdict(int),
            'safety_levels': defaultdict(int),
            'zone_types': defaultdict(int),
            'layout_features': defaultdict(int)
        }
    
    def analyze_environment(self, frame: np.ndarray, 
                          furniture_detections: List[Dict[str, Any]] = None,
                          obstacle_detections: List[Dict[str, Any]] = None) -> EnvironmentContext:
        """分析环境上下文
        
        Args:
            frame: 输入帧
            furniture_detections: 家具检测结果
            obstacle_detections: 障碍物检测结果
            
        Returns:
            EnvironmentContext: 环境上下文
        """
        try:
            self.stats['total_analyses'] += 1
            
            furniture_detections = furniture_detections or []
            obstacle_detections = obstacle_detections or []
            
            frame_height, frame_width = frame.shape[:2]
            frame_size = (frame_width, frame_height)
            
            # 1. 布局分析
            layout_analysis = self.layout_analyzer.analyze_layout(
                frame, furniture_detections, obstacle_detections
            )
            
            # 更新统计
            self.stats['room_types'][layout_analysis.room_type.value] += 1
            for feature in layout_analysis.layout_features:
                self.stats['layout_features'][feature.value] += 1
            
            # 2. 生成安全区域
            safety_zones = self.safety_zone_generator.generate_safety_zones(
                layout_analysis, frame_size
            )
            
            # 更新统计
            for zone in safety_zones:
                self.stats['safety_levels'][zone.safety_level.value] += 1
                self.stats['zone_types'][zone.area_type.value] += 1
            
            # 3. 环境因素分析
            environmental_factors = self._analyze_environmental_factors(
                layout_analysis, safety_zones, frame_size
            )
            
            # 4. 风险评估
            risk_assessment = self._perform_risk_assessment(
                layout_analysis, safety_zones
            )
            
            # 5. 监控建议
            monitoring_recommendations = self._generate_monitoring_recommendations(
                layout_analysis, safety_zones, risk_assessment
            )
            
            # 6. 创建环境上下文
            context = EnvironmentContext(
                timestamp=time.time(),
                frame_size=frame_size,
                room_analysis=layout_analysis,
                safety_zones=safety_zones,
                environmental_factors=environmental_factors,
                risk_assessment=risk_assessment,
                monitoring_recommendations=monitoring_recommendations
            )
            
            return context
            
        except Exception as e:
            self.logger.error(f"环境上下文分析失败: {e}")
            # 返回默认上下文
            return self._create_default_context(frame.shape[:2])
    
    def _analyze_environmental_factors(self, layout_analysis: LayoutAnalysis, 
                                     safety_zones: List[SafetyZone],
                                     frame_size: Tuple[int, int]) -> Dict[str, Any]:
        """分析环境因素"""
        factors = {
            'lighting_conditions': 'normal',  # 简化处理
            'space_complexity': layout_analysis.get_layout_complexity(),
            'furniture_density': layout_analysis.space_utilization,
            'obstacle_density': layout_analysis.obstacle_density,
            'accessibility_score': layout_analysis.accessibility_score,
            'total_zones': len(safety_zones),
            'safe_zone_ratio': self._calculate_safe_zone_ratio(safety_zones, frame_size),
            'risk_zone_count': len([z for z in safety_zones 
                                  if z.area_type in [AreaType.DANGER_ZONE, AreaType.CAUTION_ZONE]]),
            'walkable_area_ratio': self._calculate_walkable_area_ratio(
                layout_analysis.walkable_areas, frame_size
            )
        }
        
        return factors
    
    def _perform_risk_assessment(self, layout_analysis: LayoutAnalysis, 
                               safety_zones: List[SafetyZone]) -> Dict[str, Any]:
        """执行风险评估"""
        # 计算整体风险分数
        risk_score = 0.0
        
        # 基于障碍物密度
        risk_score += layout_analysis.obstacle_density * 0.3
        
        # 基于可达性
        risk_score += (1.0 - layout_analysis.accessibility_score) * 0.2
        
        # 基于跌倒风险区域
        risk_area_count = len(layout_analysis.fall_risk_areas)
        risk_score += min(1.0, risk_area_count * 0.1) * 0.3
        
        # 基于安全区域分布
        danger_zones = [z for z in safety_zones if z.safety_level == SafetyLevel.DANGEROUS]
        risk_score += len(danger_zones) * 0.05
        
        # 确定风险等级
        if risk_score < 0.2:
            risk_level = "low"
        elif risk_score < 0.4:
            risk_level = "moderate"
        elif risk_score < 0.6:
            risk_level = "high"
        else:
            risk_level = "very_high"
        
        assessment = {
            'overall_risk_score': min(1.0, risk_score),
            'risk_level': risk_level,
            'primary_risk_factors': self._identify_primary_risk_factors(
                layout_analysis, safety_zones
            ),
            'fall_risk_areas_count': len(layout_analysis.fall_risk_areas),
            'high_risk_zones': [z.zone_id for z in safety_zones 
                              if z.safety_level in [SafetyLevel.RISKY, SafetyLevel.DANGEROUS]],
            'monitoring_priority': self._determine_monitoring_priority(risk_level)
        }
        
        return assessment
    
    def _generate_monitoring_recommendations(self, layout_analysis: LayoutAnalysis, 
                                          safety_zones: List[SafetyZone],
                                          risk_assessment: Dict[str, Any]) -> List[str]:
        """生成监控建议"""
        recommendations = []
        
        # 基于风险等级的建议
        risk_level = risk_assessment['risk_level']
        
        if risk_level == "very_high":
            recommendations.extend([
                "建议增加多个监控点",
                "启用实时警报系统",
                "考虑重新布置环境"
            ])
        elif risk_level == "high":
            recommendations.extend([
                "增加监控密度",
                "设置重点监控区域"
            ])
        elif risk_level == "moderate":
            recommendations.append("保持正常监控频率")
        else:
            recommendations.append("标准监控配置即可")
        
        # 基于房间类型的建议
        room_type = layout_analysis.room_type
        
        if room_type == RoomType.BATHROOM:
            recommendations.extend(["重点监控湿滑区域", "确保充足照明"])
        elif room_type == RoomType.KITCHEN:
            recommendations.extend(["注意地面障碍物", "监控操作区域"])
        elif room_type == RoomType.BEDROOM:
            recommendations.extend(["床边重点监控", "夜间模式监控"])
        
        # 基于布局特征的建议
        if LayoutFeature.NARROW_PASSAGE in layout_analysis.layout_features:
            recommendations.append("狭窄通道需要特别关注")
        
        if LayoutFeature.FURNITURE_CLUSTER in layout_analysis.layout_features:
            recommendations.append("家具聚集区域增加监控")
        
        # 基于障碍物密度的建议
        if layout_analysis.obstacle_density > 0.6:
            recommendations.extend(["考虑清理部分障碍物", "优化空间布局"])
        
        return list(set(recommendations))  # 去重
    
    def _calculate_safe_zone_ratio(self, safety_zones: List[SafetyZone], 
                                 frame_size: Tuple[int, int]) -> float:
        """计算安全区域比例"""
        if not safety_zones:
            return 0.0
        
        frame_width, frame_height = frame_size
        total_area = frame_width * frame_height
        
        safe_area = sum(zone.area_size for zone in safety_zones 
                       if zone.safety_level in [SafetyLevel.SAFE, SafetyLevel.VERY_SAFE])
        
        return min(1.0, safe_area / total_area)
    
    def _calculate_walkable_area_ratio(self, walkable_areas: List[List[Tuple[int, int]]], 
                                     frame_size: Tuple[int, int]) -> float:
        """计算可行走区域比例"""
        if not walkable_areas:
            return 0.0
        
        frame_width, frame_height = frame_size
        total_area = frame_width * frame_height
        
        walkable_area = 0
        for polygon in walkable_areas:
            if len(polygon) >= 3:
                area = self.safety_zone_generator._calculate_polygon_area(polygon)
                walkable_area += area
        
        return min(1.0, walkable_area / total_area)
    
    def _identify_primary_risk_factors(self, layout_analysis: LayoutAnalysis, 
                                     safety_zones: List[SafetyZone]) -> List[str]:
        """识别主要风险因素"""
        risk_factors = []
        
        # 高障碍物密度
        if layout_analysis.obstacle_density > 0.6:
            risk_factors.append("高障碍物密度")
        
        # 低可达性
        if layout_analysis.accessibility_score < 0.4:
            risk_factors.append("可达性差")
        
        # 跌倒风险区域
        if len(layout_analysis.fall_risk_areas) > 2:
            risk_factors.append("多个跌倒风险区域")
        
        # 狭窄通道
        if LayoutFeature.NARROW_PASSAGE in layout_analysis.layout_features:
            risk_factors.append("狭窄通道")
        
        # 家具聚集
        if LayoutFeature.FURNITURE_CLUSTER in layout_analysis.layout_features:
            risk_factors.append("家具聚集")
        
        # 危险区域
        danger_zones = [z for z in safety_zones if z.safety_level == SafetyLevel.DANGEROUS]
        if danger_zones:
            risk_factors.append("存在危险区域")
        
        return risk_factors
    
    def _determine_monitoring_priority(self, risk_level: str) -> str:
        """确定监控优先级"""
        priority_map = {
            "low": "normal",
            "moderate": "elevated",
            "high": "high",
            "very_high": "critical"
        }
        
        return priority_map.get(risk_level, "normal")
    
    def _create_default_context(self, frame_shape: Tuple[int, int]) -> EnvironmentContext:
        """创建默认环境上下文"""
        frame_height, frame_width = frame_shape
        frame_size = (frame_width, frame_height)
        
        # 默认布局分析
        default_layout = LayoutAnalysis(
            room_type=RoomType.UNKNOWN,
            layout_features=[],
            furniture_layout={},
            walkable_areas=[],
            obstacle_density=0.0,
            space_utilization=0.0,
            accessibility_score=0.5,
            fall_risk_areas=[]
        )
        
        # 默认安全区域
        default_zone = SafetyZone(
            zone_id=1,
            area_type=AreaType.SAFE_ZONE,
            safety_level=SafetyLevel.MODERATE,
            polygon=[(0, 0), (frame_width, 0), (frame_width, frame_height), (0, frame_height)],
            center_point=(frame_width / 2, frame_height / 2),
            area_size=frame_width * frame_height,
            risk_factors=["未知环境"],
            recommendations=["需要进一步分析"],
            confidence=0.5
        )
        
        return EnvironmentContext(
            timestamp=time.time(),
            frame_size=frame_size,
            room_analysis=default_layout,
            safety_zones=[default_zone],
            environmental_factors={'space_complexity': 'unknown'},
            risk_assessment={'risk_level': 'moderate', 'overall_risk_score': 0.5},
            monitoring_recommendations=["需要完整的环境分析"]
        )
    
    def get_zone_recommendations_for_person(self, person_position: Tuple[float, float], 
                                          context: EnvironmentContext) -> Dict[str, Any]:
        """获取人员位置的区域建议
        
        Args:
            person_position: 人员位置
            context: 环境上下文
            
        Returns:
            Dict[str, Any]: 区域建议
        """
        try:
            # 查找人员所在区域
            current_zone = context.get_zone_for_point(person_position)
            
            if not current_zone:
                return {
                    'current_zone': None,
                    'safety_level': 'unknown',
                    'recommendations': ['无法确定当前区域'],
                    'risk_factors': ['位置未知']
                }
            
            # 分析当前区域
            recommendations = list(current_zone.recommendations)
            risk_factors = list(current_zone.risk_factors)
            
            # 查找附近的危险区域
            nearby_danger_zones = []
            for zone in context.safety_zones:
                if (zone.area_type == AreaType.DANGER_ZONE and 
                    zone.distance_to_point(person_position) < 100):  # 100像素范围内
                    nearby_danger_zones.append(zone)
            
            if nearby_danger_zones:
                recommendations.append("附近存在危险区域，请保持警惕")
                risk_factors.append("临近危险区域")
            
            # 基于安全等级的建议
            if current_zone.safety_level == SafetyLevel.DANGEROUS:
                recommendations.extend([
                    "立即离开当前区域",
                    "寻求帮助"
                ])
            elif current_zone.safety_level == SafetyLevel.RISKY:
                recommendations.extend([
                    "小心行走",
                    "注意周围环境"
                ])
            
            return {
                'current_zone': {
                    'zone_id': current_zone.zone_id,
                    'area_type': current_zone.area_type.value,
                    'safety_level': current_zone.safety_level.value
                },
                'safety_level': current_zone.safety_level.value,
                'recommendations': recommendations,
                'risk_factors': risk_factors,
                'nearby_danger_zones': len(nearby_danger_zones)
            }
            
        except Exception as e:
            self.logger.error(f"获取区域建议失败: {e}")
            return {
                'current_zone': None,
                'safety_level': 'unknown',
                'recommendations': ['分析失败'],
                'risk_factors': ['系统错误']
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_analyses': self.stats['total_analyses'],
            'room_type_distribution': dict(self.stats['room_types']),
            'safety_level_distribution': dict(self.stats['safety_levels']),
            'zone_type_distribution': dict(self.stats['zone_types']),
            'layout_feature_frequency': dict(self.stats['layout_features'])
        }
    
    def export_context_report(self, context: EnvironmentContext, 
                            output_path: str = None) -> Dict[str, Any]:
        """导出环境上下文报告
        
        Args:
            context: 环境上下文
            output_path: 输出路径（可选）
            
        Returns:
            Dict[str, Any]: 报告数据
        """
        try:
            report = {
                'timestamp': context.timestamp,
                'frame_size': context.frame_size,
                'room_analysis': {
                    'room_type': context.room_analysis.room_type.value,
                    'layout_complexity': context.room_analysis.get_layout_complexity(),
                    'layout_features': [f.value for f in context.room_analysis.layout_features],
                    'furniture_count': sum(len(furniture_list) 
                                         for furniture_list in context.room_analysis.furniture_layout.values()),
                    'obstacle_density': context.room_analysis.obstacle_density,
                    'space_utilization': context.room_analysis.space_utilization,
                    'accessibility_score': context.room_analysis.accessibility_score,
                    'fall_risk_areas_count': len(context.room_analysis.fall_risk_areas)
                },
                'safety_zones': [
                    {
                        'zone_id': zone.zone_id,
                        'area_type': zone.area_type.value,
                        'safety_level': zone.safety_level.value,
                        'area_size': zone.area_size,
                        'center_point': zone.center_point,
                        'risk_factors': zone.risk_factors,
                        'recommendations': zone.recommendations,
                        'confidence': zone.confidence
                    }
                    for zone in context.safety_zones
                ],
                'environmental_factors': context.environmental_factors,
                'risk_assessment': context.risk_assessment,
                'monitoring_recommendations': context.monitoring_recommendations,
                'overall_safety_level': context.get_overall_safety_level().value
            }
            
            # 如果指定了输出路径，保存到文件
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                self.logger.info(f"环境上下文报告已保存到: {output_path}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"导出环境上下文报告失败: {e}")
            return {}
    
    def cleanup_resources(self):
        """清理资源"""
        try:
            # 重置统计信息
            self.stats = {
                'total_analyses': 0,
                'room_types': defaultdict(int),
                'safety_levels': defaultdict(int),
                'zone_types': defaultdict(int),
                'layout_features': defaultdict(int)
            }
            
            # 重置区域ID计数器
            self.safety_zone_generator.zone_id_counter = 1
            
            self.logger.info("环境上下文分析器资源清理完成")
            
        except Exception as e:
            self.logger.error(f"资源清理失败: {e}")


# 测试代码
def test_environment_context_analyzer():
    """测试环境上下文分析器"""
    print("=== 环境上下文分析器测试 ===")
    
    # 创建分析器
    analyzer = EnvironmentContextAnalyzer()
    
    # 模拟输入数据
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 模拟家具检测结果
    furniture_detections = [
        {'type': 'sofa', 'bbox': (50, 100, 200, 180), 'confidence': 0.9},
        {'type': 'table', 'bbox': (250, 150, 350, 200), 'confidence': 0.8},
        {'type': 'chair', 'bbox': (300, 120, 340, 160), 'confidence': 0.7},
        {'type': 'bookshelf', 'bbox': (500, 50, 580, 250), 'confidence': 0.85}
    ]
    
    # 模拟障碍物检测结果
    obstacle_detections = [
        {'type': 'toy', 'bbox': (150, 200, 180, 220), 'confidence': 0.6},
        {'type': 'box', 'bbox': (400, 180, 450, 220), 'confidence': 0.7}
    ]
    
    # 执行环境分析
    print("\n1. 执行环境上下文分析...")
    context = analyzer.analyze_environment(
        frame, furniture_detections, obstacle_detections
    )
    
    print(f"房间类型: {context.room_analysis.room_type.value}")
    print(f"布局复杂度: {context.room_analysis.get_layout_complexity()}")
    print(f"障碍物密度: {context.room_analysis.obstacle_density:.3f}")
    print(f"空间利用率: {context.room_analysis.space_utilization:.3f}")
    print(f"可达性分数: {context.room_analysis.accessibility_score:.3f}")
    print(f"安全区域数量: {len(context.safety_zones)}")
    print(f"整体安全等级: {context.get_overall_safety_level().value}")
    
    # 测试人员位置建议
    print("\n2. 测试人员位置建议...")
    person_positions = [(100, 150), (300, 180), (450, 200)]
    
    for i, pos in enumerate(person_positions, 1):
        recommendations = analyzer.get_zone_recommendations_for_person(pos, context)
        print(f"\n人员{i} 位置 {pos}:")
        print(f"  当前区域: {recommendations.get('current_zone', 'None')}")
        print(f"  安全等级: {recommendations['safety_level']}")
        print(f"  建议: {recommendations['recommendations'][:2]}")
        print(f"  风险因素: {recommendations['risk_factors'][:2]}")
    
    # 获取统计信息
    print("\n3. 统计信息:")
    stats = analyzer.get_statistics()
    print(f"总分析次数: {stats['total_analyses']}")
    print(f"房间类型分布: {stats['room_type_distribution']}")
    print(f"安全等级分布: {stats['safety_level_distribution']}")
    
    # 导出报告
    print("\n4. 导出环境上下文报告...")
    report = analyzer.export_context_report(context)
    print(f"报告包含 {len(report)} 个主要部分")
    print(f"监控建议数量: {len(report.get('monitoring_recommendations', []))}")
    
    # 清理资源
    analyzer.cleanup_resources()
    print("\n资源清理完成")


def test_complex_environment_scenarios():
    """测试复杂环境场景"""
    print("\n=== 复杂环境场景测试 ===")
    
    analyzer = EnvironmentContextAnalyzer()
    
    # 场景1: 高密度障碍物环境
    print("\n场景1: 高密度障碍物环境")
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    
    furniture_detections1 = [
        {'type': 'sofa', 'bbox': (50, 100, 200, 180)},
        {'type': 'table', 'bbox': (220, 120, 320, 170)},
        {'type': 'chair', 'bbox': (340, 100, 380, 140)},
        {'type': 'chair', 'bbox': (340, 150, 380, 190)},
        {'type': 'bookshelf', 'bbox': (400, 50, 480, 250)},
        {'type': 'cabinet', 'bbox': (500, 200, 600, 300)}
    ]
    
    obstacle_detections1 = [
        {'type': 'toy', 'bbox': (150, 200, 180, 220)},
        {'type': 'box', 'bbox': (250, 200, 290, 240)},
        {'type': 'bag', 'bbox': (350, 200, 380, 230)},
        {'type': 'toy', 'bbox': (450, 180, 480, 200)}
    ]
    
    context1 = analyzer.analyze_environment(
        frame1, furniture_detections1, obstacle_detections1
    )
    
    print(f"障碍物密度: {context1.room_analysis.obstacle_density:.3f}")
    print(f"风险等级: {context1.risk_assessment['risk_level']}")
    print(f"主要风险因素: {context1.risk_assessment['primary_risk_factors'][:3]}")
    
    # 场景2: 开放空间环境
    print("\n场景2: 开放空间环境")
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
    
    furniture_detections2 = [
        {'type': 'sofa', 'bbox': (50, 350, 200, 430)},
        {'type': 'tv_stand', 'bbox': (500, 200, 600, 250)}
    ]
    
    obstacle_detections2 = []
    
    context2 = analyzer.analyze_environment(
        frame2, furniture_detections2, obstacle_detections2
    )
    
    print(f"障碍物密度: {context2.room_analysis.obstacle_density:.3f}")
    print(f"可达性分数: {context2.room_analysis.accessibility_score:.3f}")
    print(f"风险等级: {context2.risk_assessment['risk_level']}")
    print(f"布局特征: {[f.value for f in context2.room_analysis.layout_features]}")
    
    # 场景3: 狭窄空间环境
    print("\n场景3: 狭窄空间环境")
    frame3 = np.zeros((480, 640, 3), dtype=np.uint8)
    
    furniture_detections3 = [
        {'type': 'bed', 'bbox': (50, 100, 300, 250)},
        {'type': 'wardrobe', 'bbox': (350, 50, 450, 300)},
        {'type': 'desk', 'bbox': (480, 100, 600, 180)},
        {'type': 'chair', 'bbox': (500, 200, 540, 240)}
    ]
    
    obstacle_detections3 = [
        {'type': 'clothes', 'bbox': (320, 260, 360, 290)},
        {'type': 'shoes', 'bbox': (460, 250, 490, 270)}
    ]
    
    context3 = analyzer.analyze_environment(
        frame3, furniture_detections3, obstacle_detections3
    )
    
    print(f"房间类型: {context3.room_analysis.room_type.value}")
    print(f"空间利用率: {context3.room_analysis.space_utilization:.3f}")
    print(f"监控建议: {context3.monitoring_recommendations[:3]}")
    
    analyzer.cleanup_resources()


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行测试
    test_environment_context_analyzer()
    test_complex_environment_scenarios()