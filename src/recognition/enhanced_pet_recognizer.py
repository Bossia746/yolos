#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强宠物识别器 - 基于业界最佳实践
集成物种识别、品种分类、姿态估计、行为分析、健康监测
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import time
import logging
from pathlib import Path
import json

# MediaPipe for pose estimation
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available, pose estimation disabled")

# YOLO for detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: Ultralytics YOLO not available, using OpenCV DNN")

logger = logging.getLogger(__name__)


class PetSpecies(Enum):
    """宠物物种枚举"""
    DOG = "dog"
    CAT = "cat"
    BIRD = "bird"
    RABBIT = "rabbit"
    HAMSTER = "hamster"
    GUINEA_PIG = "guinea_pig"
    FERRET = "ferret"
    FISH = "fish"
    REPTILE = "reptile"
    UNKNOWN = "unknown"


class PetBehavior(Enum):
    """宠物行为枚举"""
    SLEEPING = "sleeping"
    EATING = "eating"
    DRINKING = "drinking"
    PLAYING = "playing"
    WALKING = "walking"
    RUNNING = "running"
    SITTING = "sitting"
    LYING = "lying"
    STANDING = "standing"
    GROOMING = "grooming"
    ALERT = "alert"
    AGGRESSIVE = "aggressive"
    HIDING = "hiding"
    EXPLORING = "exploring"
    UNKNOWN = "unknown"


class PetHealthStatus(Enum):
    """宠物健康状态枚举"""
    HEALTHY = "healthy"
    SICK = "sick"
    INJURED = "injured"
    STRESSED = "stressed"
    LETHARGIC = "lethargic"
    HYPERACTIVE = "hyperactive"
    UNKNOWN = "unknown"


@dataclass
class PetDetection:
    """宠物检测结果"""
    species: PetSpecies
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    breed: Optional[str] = None
    age_estimate: Optional[str] = None  # puppy/kitten, adult, senior
    size_category: Optional[str] = None  # small, medium, large
    color_pattern: Optional[str] = None
    

@dataclass
class PetPose:
    """宠物姿态结果"""
    keypoints: List[Tuple[float, float, float]]  # x, y, confidence
    pose_confidence: float
    pose_type: str  # standing, sitting, lying, etc.
    body_orientation: float  # 身体朝向角度


@dataclass
class PetBehaviorResult:
    """宠物行为分析结果"""
    behavior: PetBehavior
    confidence: float
    duration: float  # 行为持续时间（秒）
    intensity: float  # 行为强度 0-1
    context: Dict[str, Any]  # 行为上下文信息


@dataclass
class PetHealthResult:
    """宠物健康评估结果"""
    status: PetHealthStatus
    confidence: float
    indicators: Dict[str, float]  # 健康指标
    recommendations: List[str]  # 建议
    risk_factors: List[str]  # 风险因素


@dataclass
class EnhancedPetResult:
    """增强宠物识别综合结果"""
    detection: PetDetection
    pose: Optional[PetPose] = None
    behavior: Optional[PetBehaviorResult] = None
    health: Optional[PetHealthResult] = None
    tracking_id: Optional[int] = None
    timestamp: float = 0.0


class PetSpeciesClassifier:
    """宠物物种分类器"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        self.species_map = {
            0: PetSpecies.DOG,
            1: PetSpecies.CAT,
            2: PetSpecies.BIRD,
            3: PetSpecies.RABBIT,
            4: PetSpecies.HAMSTER,
            5: PetSpecies.GUINEA_PIG,
            6: PetSpecies.FERRET,
            7: PetSpecies.FISH,
            8: PetSpecies.REPTILE
        }
        self._load_model()
    
    def _load_model(self):
        """加载物种分类模型"""
        try:
            if self.model_path and Path(self.model_path).exists():
                self.model = torch.load(self.model_path, map_location='cpu')
                self.model.eval()
                logger.info(f"Pet species classifier loaded: {self.model_path}")
            else:
                # 使用预训练模型作为基础
                from torchvision.models import resnet50
                self.model = resnet50(pretrained=True)
                self.model.fc = nn.Linear(self.model.fc.in_features, len(self.species_map))
                self.model.eval()
                logger.warning("Using pretrained ResNet50 for pet species classification")
        except Exception as e:
            logger.error(f"Failed to load pet species classifier: {e}")
            self.model = None
    
    def classify(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[PetSpecies, float]:
        """分类宠物物种"""
        if self.model is None:
            return PetSpecies.UNKNOWN, 0.0
        
        try:
            # 裁剪宠物区域
            x, y, w, h = bbox
            pet_crop = image[y:y+h, x:x+w]
            
            if pet_crop.size == 0:
                return PetSpecies.UNKNOWN, 0.0
            
            # 预处理
            input_tensor = self.transform(pet_crop).unsqueeze(0)
            
            # 推理
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                species = self.species_map.get(predicted.item(), PetSpecies.UNKNOWN)
                return species, confidence.item()
                
        except Exception as e:
            logger.debug(f"Pet species classification error: {e}")
            return PetSpecies.UNKNOWN, 0.0


class PetPoseEstimator:
    """宠物姿态估计器"""
    
    def __init__(self):
        self.mp_pose = None
        self.pose_detector = None
        
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("Pet pose estimator initialized with MediaPipe")
        else:
            logger.warning("MediaPipe not available, pose estimation disabled")
    
    def estimate_pose(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[PetPose]:
        """估计宠物姿态"""
        if not self.pose_detector:
            return None
        
        try:
            # 裁剪宠物区域
            x, y, w, h = bbox
            pet_crop = image[y:y+h, x:x+w]
            
            if pet_crop.size == 0:
                return None
            
            # 转换为RGB
            rgb_crop = cv2.cvtColor(pet_crop, cv2.COLOR_BGR2RGB)
            
            # 姿态检测
            results = self.pose_detector.process(rgb_crop)
            
            if results.pose_landmarks:
                # 提取关键点
                keypoints = []
                for landmark in results.pose_landmarks.landmark:
                    # 转换回原图坐标
                    abs_x = x + landmark.x * w
                    abs_y = y + landmark.y * h
                    keypoints.append((abs_x, abs_y, landmark.visibility))
                
                # 计算姿态置信度
                pose_confidence = np.mean([kp[2] for kp in keypoints])
                
                # 分析姿态类型
                pose_type = self._analyze_pose_type(keypoints)
                
                # 计算身体朝向
                body_orientation = self._calculate_body_orientation(keypoints)
                
                return PetPose(
                    keypoints=keypoints,
                    pose_confidence=pose_confidence,
                    pose_type=pose_type,
                    body_orientation=body_orientation
                )
            
        except Exception as e:
            logger.debug(f"Pet pose estimation error: {e}")
        
        return None
    
    def _analyze_pose_type(self, keypoints: List[Tuple[float, float, float]]) -> str:
        """分析姿态类型"""
        if len(keypoints) < 33:  # MediaPipe pose has 33 landmarks
            return "unknown"
        
        try:
            # 简化的姿态分析
            # 基于关键点的相对位置判断姿态
            
            # 获取关键点 (MediaPipe pose landmarks)
            nose = keypoints[0]
            left_shoulder = keypoints[11]
            right_shoulder = keypoints[12]
            left_hip = keypoints[23]
            right_hip = keypoints[24]
            left_knee = keypoints[25]
            right_knee = keypoints[26]
            
            # 计算身体中心
            body_center_y = (left_shoulder[1] + right_shoulder[1] + left_hip[1] + right_hip[1]) / 4
            
            # 判断姿态
            if nose[1] > body_center_y + 50:  # 头部低于身体中心
                return "lying"
            elif left_knee[1] < left_hip[1] - 20 and right_knee[1] < right_hip[1] - 20:  # 膝盖高于臀部
                return "sitting"
            else:
                return "standing"
                
        except Exception:
            return "unknown"
    
    def _calculate_body_orientation(self, keypoints: List[Tuple[float, float, float]]) -> float:
        """计算身体朝向角度"""
        try:
            if len(keypoints) < 12:
                return 0.0
            
            # 使用肩膀连线计算朝向
            left_shoulder = keypoints[11]
            right_shoulder = keypoints[12]
            
            dx = right_shoulder[0] - left_shoulder[0]
            dy = right_shoulder[1] - left_shoulder[1]
            
            angle = np.arctan2(dy, dx) * 180 / np.pi
            return angle
            
        except Exception:
            return 0.0


class PetBehaviorAnalyzer:
    """宠物行为分析器"""
    
    def __init__(self, history_length: int = 30):
        self.history_length = history_length
        self.detection_history = {}
        self.behavior_history = {}
        self.behavior_start_time = {}
    
    def analyze_behavior(self, detection: PetDetection, pose: Optional[PetPose], 
                        tracking_id: int, timestamp: float) -> Optional[PetBehaviorResult]:
        """分析宠物行为"""
        try:
            # 更新检测历史
            if tracking_id not in self.detection_history:
                self.detection_history[tracking_id] = []
            
            self.detection_history[tracking_id].append({
                'detection': detection,
                'pose': pose,
                'timestamp': timestamp
            })
            
            # 限制历史长度
            if len(self.detection_history[tracking_id]) > self.history_length:
                self.detection_history[tracking_id] = self.detection_history[tracking_id][-self.history_length:]
            
            history = self.detection_history[tracking_id]
            
            if len(history) < 3:  # 需要足够的历史数据
                return None
            
            # 分析运动模式
            movement_analysis = self._analyze_movement(history)
            
            # 分析姿态模式
            pose_analysis = self._analyze_pose_pattern(history)
            
            # 综合判断行为
            behavior, confidence = self._classify_behavior(movement_analysis, pose_analysis)
            
            # 计算行为持续时间
            duration = self._calculate_behavior_duration(tracking_id, behavior, timestamp)
            
            # 计算行为强度
            intensity = self._calculate_behavior_intensity(movement_analysis, pose_analysis)
            
            # 构建上下文信息
            context = {
                'movement_speed': movement_analysis.get('avg_speed', 0),
                'movement_direction': movement_analysis.get('direction_stability', 0),
                'pose_stability': pose_analysis.get('stability', 0),
                'location_preference': movement_analysis.get('location_preference', 'center')
            }
            
            return PetBehaviorResult(
                behavior=behavior,
                confidence=confidence,
                duration=duration,
                intensity=intensity,
                context=context
            )
            
        except Exception as e:
            logger.debug(f"Pet behavior analysis error: {e}")
            return None
    
    def _analyze_movement(self, history: List[Dict]) -> Dict[str, Any]:
        """分析运动模式"""
        if len(history) < 2:
            return {}
        
        movements = []
        speeds = []
        
        for i in range(1, len(history)):
            prev_center = self._get_bbox_center(history[i-1]['detection'].bbox)
            curr_center = self._get_bbox_center(history[i]['detection'].bbox)
            
            # 计算移动距离
            distance = np.linalg.norm(np.array(curr_center) - np.array(prev_center))
            movements.append(distance)
            
            # 计算速度 (像素/秒)
            time_diff = history[i]['timestamp'] - history[i-1]['timestamp']
            if time_diff > 0:
                speed = distance / time_diff
                speeds.append(speed)
        
        return {
            'avg_movement': np.mean(movements) if movements else 0,
            'max_movement': np.max(movements) if movements else 0,
            'movement_variance': np.var(movements) if movements else 0,
            'avg_speed': np.mean(speeds) if speeds else 0,
            'direction_stability': self._calculate_direction_stability(history),
            'location_preference': self._analyze_location_preference(history)
        }
    
    def _analyze_pose_pattern(self, history: List[Dict]) -> Dict[str, Any]:
        """分析姿态模式"""
        poses = [h['pose'] for h in history if h['pose'] is not None]
        
        if not poses:
            return {}
        
        pose_types = [pose.pose_type for pose in poses]
        pose_confidences = [pose.pose_confidence for pose in poses]
        
        return {
            'dominant_pose': max(set(pose_types), key=pose_types.count) if pose_types else 'unknown',
            'pose_changes': len(set(pose_types)),
            'avg_pose_confidence': np.mean(pose_confidences) if pose_confidences else 0,
            'stability': 1.0 - (len(set(pose_types)) / len(pose_types)) if pose_types else 0
        }
    
    def _classify_behavior(self, movement_analysis: Dict, pose_analysis: Dict) -> Tuple[PetBehavior, float]:
        """分类行为"""
        avg_speed = movement_analysis.get('avg_speed', 0)
        movement_variance = movement_analysis.get('movement_variance', 0)
        dominant_pose = pose_analysis.get('dominant_pose', 'unknown')
        pose_stability = pose_analysis.get('stability', 0)
        
        # 基于规则的行为分类
        if avg_speed < 5 and pose_stability > 0.8:
            if dominant_pose == 'lying':
                return PetBehavior.SLEEPING, 0.9
            elif dominant_pose == 'sitting':
                return PetBehavior.SITTING, 0.8
            else:
                return PetBehavior.RESTING, 0.7
        
        elif avg_speed < 20 and movement_variance < 100:
            if dominant_pose == 'standing':
                return PetBehavior.STANDING, 0.8
            else:
                return PetBehavior.WALKING, 0.7
        
        elif avg_speed > 50:
            return PetBehavior.RUNNING, 0.8
        
        elif movement_variance > 200:
            return PetBehavior.PLAYING, 0.7
        
        else:
            return PetBehavior.UNKNOWN, 0.5
    
    def _calculate_behavior_duration(self, tracking_id: int, behavior: PetBehavior, timestamp: float) -> float:
        """计算行为持续时间"""
        if tracking_id not in self.behavior_history:
            self.behavior_history[tracking_id] = []
            self.behavior_start_time[tracking_id] = {}
        
        # 检查是否是新行为
        if (not self.behavior_history[tracking_id] or 
            self.behavior_history[tracking_id][-1] != behavior):
            self.behavior_start_time[tracking_id][behavior] = timestamp
            self.behavior_history[tracking_id].append(behavior)
            return 0.0
        else:
            start_time = self.behavior_start_time[tracking_id].get(behavior, timestamp)
            return timestamp - start_time
    
    def _calculate_behavior_intensity(self, movement_analysis: Dict, pose_analysis: Dict) -> float:
        """计算行为强度"""
        speed_factor = min(movement_analysis.get('avg_speed', 0) / 100, 1.0)
        variance_factor = min(movement_analysis.get('movement_variance', 0) / 500, 1.0)
        confidence_factor = pose_analysis.get('avg_pose_confidence', 0.5)
        
        intensity = (speed_factor + variance_factor + confidence_factor) / 3
        return np.clip(intensity, 0.0, 1.0)
    
    def _get_bbox_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """获取边界框中心"""
        x, y, w, h = bbox
        return (x + w / 2, y + h / 2)
    
    def _calculate_direction_stability(self, history: List[Dict]) -> float:
        """计算方向稳定性"""
        if len(history) < 3:
            return 0.5
        
        directions = []
        for i in range(2, len(history)):
            prev_center = self._get_bbox_center(history[i-2]['detection'].bbox)
            curr_center = self._get_bbox_center(history[i]['detection'].bbox)
            
            dx = curr_center[0] - prev_center[0]
            dy = curr_center[1] - prev_center[1]
            
            if dx != 0 or dy != 0:
                direction = np.arctan2(dy, dx)
                directions.append(direction)
        
        if not directions:
            return 0.5
        
        # 计算方向变化的标准差
        direction_std = np.std(directions)
        stability = 1.0 / (1.0 + direction_std)
        return stability
    
    def _analyze_location_preference(self, history: List[Dict]) -> str:
        """分析位置偏好"""
        centers = [self._get_bbox_center(h['detection'].bbox) for h in history]
        
        if not centers:
            return 'center'
        
        avg_x = np.mean([c[0] for c in centers])
        avg_y = np.mean([c[1] for c in centers])
        
        # 假设图像尺寸为640x480 (可以从实际图像获取)
        img_width, img_height = 640, 480
        
        if avg_x < img_width * 0.3:
            return 'left'
        elif avg_x > img_width * 0.7:
            return 'right'
        elif avg_y < img_height * 0.3:
            return 'top'
        elif avg_y > img_height * 0.7:
            return 'bottom'
        else:
            return 'center'


class PetHealthMonitor:
    """宠物健康监测器"""
    
    def __init__(self):
        self.health_history = {}
        self.alert_thresholds = {
            'low_activity': 0.2,
            'high_activity': 0.9,
            'behavior_monotony': 0.3,
            'pose_instability': 0.3
        }
    
    def assess_health(self, detection: PetDetection, behavior_result: Optional[PetBehaviorResult],
                     tracking_id: int, timestamp: float) -> Optional[PetHealthResult]:
        """评估宠物健康状态"""
        try:
            # 更新健康历史
            if tracking_id not in self.health_history:
                self.health_history[tracking_id] = []
            
            self.health_history[tracking_id].append({
                'detection': detection,
                'behavior': behavior_result,
                'timestamp': timestamp
            })
            
            # 限制历史长度 (保留最近1小时的数据，假设30fps)
            max_history = 30 * 60 * 60  # 1小时
            if len(self.health_history[tracking_id]) > max_history:
                self.health_history[tracking_id] = self.health_history[tracking_id][-max_history:]
            
            history = self.health_history[tracking_id]
            
            if len(history) < 10:  # 需要足够的历史数据
                return None
            
            # 分析健康指标
            indicators = self._analyze_health_indicators(history)
            
            # 评估健康状态
            status, confidence = self._assess_health_status(indicators)
            
            # 生成建议和风险因素
            recommendations = self._generate_recommendations(status, indicators)
            risk_factors = self._identify_risk_factors(indicators)
            
            return PetHealthResult(
                status=status,
                confidence=confidence,
                indicators=indicators,
                recommendations=recommendations,
                risk_factors=risk_factors
            )
            
        except Exception as e:
            logger.debug(f"Pet health assessment error: {e}")
            return None
    
    def _analyze_health_indicators(self, history: List[Dict]) -> Dict[str, float]:
        """分析健康指标"""
        indicators = {}
        
        # 活动水平分析
        behaviors = [h['behavior'] for h in history if h['behavior'] is not None]
        if behaviors:
            activity_levels = [b.intensity for b in behaviors]
            indicators['activity_level'] = np.mean(activity_levels)
            indicators['activity_variance'] = np.var(activity_levels)
        else:
            indicators['activity_level'] = 0.5
            indicators['activity_variance'] = 0.0
        
        # 行为多样性分析
        behavior_types = [b.behavior.value for b in behaviors if b is not None]
        unique_behaviors = len(set(behavior_types))
        indicators['behavior_diversity'] = min(unique_behaviors / 5.0, 1.0)  # 归一化到0-1
        
        # 休息质量分析
        sleep_behaviors = [b for b in behaviors if b.behavior == PetBehavior.SLEEPING]
        if sleep_behaviors:
            sleep_durations = [b.duration for b in sleep_behaviors]
            indicators['sleep_quality'] = min(np.mean(sleep_durations) / 3600, 1.0)  # 归一化到小时
        else:
            indicators['sleep_quality'] = 0.0
        
        # 食欲分析 (基于进食行为)
        eating_behaviors = [b for b in behaviors if b.behavior == PetBehavior.EATING]
        indicators['appetite'] = min(len(eating_behaviors) / 10.0, 1.0)  # 归一化
        
        # 社交性分析 (基于警觉和探索行为)
        social_behaviors = [b for b in behaviors if b.behavior in [PetBehavior.ALERT, PetBehavior.EXPLORING]]
        indicators['social_engagement'] = min(len(social_behaviors) / 20.0, 1.0)
        
        # 压力指标 (基于躲藏和攻击行为)
        stress_behaviors = [b for b in behaviors if b.behavior in [PetBehavior.HIDING, PetBehavior.AGGRESSIVE]]
        indicators['stress_level'] = min(len(stress_behaviors) / 5.0, 1.0)
        
        return indicators
    
    def _assess_health_status(self, indicators: Dict[str, float]) -> Tuple[PetHealthStatus, float]:
        """评估健康状态"""
        activity_level = indicators.get('activity_level', 0.5)
        behavior_diversity = indicators.get('behavior_diversity', 0.5)
        stress_level = indicators.get('stress_level', 0.0)
        sleep_quality = indicators.get('sleep_quality', 0.5)
        
        # 健康评分计算
        health_score = (
            activity_level * 0.3 +
            behavior_diversity * 0.2 +
            (1 - stress_level) * 0.2 +
            sleep_quality * 0.2 +
            indicators.get('appetite', 0.5) * 0.1
        )
        
        # 状态判断
        if health_score > 0.8:
            return PetHealthStatus.HEALTHY, 0.9
        elif health_score > 0.6:
            return PetHealthStatus.HEALTHY, 0.7
        elif stress_level > 0.6:
            return PetHealthStatus.STRESSED, 0.8
        elif activity_level < 0.2:
            return PetHealthStatus.LETHARGIC, 0.7
        elif activity_level > 0.9 and behavior_diversity < 0.3:
            return PetHealthStatus.HYPERACTIVE, 0.7
        elif health_score < 0.4:
            return PetHealthStatus.SICK, 0.6
        else:
            return PetHealthStatus.UNKNOWN, 0.5
    
    def _generate_recommendations(self, status: PetHealthStatus, indicators: Dict[str, float]) -> List[str]:
        """生成健康建议"""
        recommendations = []
        
        if status == PetHealthStatus.HEALTHY:
            recommendations.append("宠物状态良好，继续保持当前的护理方式")
        
        elif status == PetHealthStatus.LETHARGIC:
            recommendations.append("宠物活动水平较低，建议增加互动和运动")
            recommendations.append("观察是否有其他疾病症状，必要时咨询兽医")
        
        elif status == PetHealthStatus.STRESSED:
            recommendations.append("宠物可能处于压力状态，提供安静舒适的环境")
            recommendations.append("减少环境变化，保持日常作息规律")
        
        elif status == PetHealthStatus.HYPERACTIVE:
            recommendations.append("宠物过度活跃，可能需要更多的运动和刺激")
            recommendations.append("检查是否有焦虑或其他行为问题")
        
        elif status == PetHealthStatus.SICK:
            recommendations.append("宠物健康指标异常，建议尽快咨询兽医")
            recommendations.append("密切观察食欲、精神状态和排泄情况")
        
        # 基于具体指标的建议
        if indicators.get('activity_level', 0.5) < 0.3:
            recommendations.append("增加日常运动和玩耍时间")
        
        if indicators.get('behavior_diversity', 0.5) < 0.3:
            recommendations.append("提供更多样化的活动和玩具")
        
        if indicators.get('sleep_quality', 0.5) < 0.3:
            recommendations.append("改善睡眠环境，确保充足的休息时间")
        
        return recommendations
    
    def _identify_risk_factors(self, indicators: Dict[str, float]) -> List[str]:
        """识别风险因素"""
        risk_factors = []
        
        if indicators.get('activity_level', 0.5) < self.alert_thresholds['low_activity']:
            risk_factors.append("活动水平过低")
        
        if indicators.get('activity_level', 0.5) > self.alert_thresholds['high_activity']:
            risk_factors.append("活动水平过高")
        
        if indicators.get('behavior_diversity', 0.5) < self.alert_thresholds['behavior_monotony']:
            risk_factors.append("行为模式单一")
        
        if indicators.get('stress_level', 0.0) > 0.5:
            risk_factors.append("压力水平较高")
        
        if indicators.get('sleep_quality', 0.5) < 0.3:
            risk_factors.append("睡眠质量不佳")
        
        if indicators.get('appetite', 0.5) < 0.3:
            risk_factors.append("食欲不振")
        
        return risk_factors


class EnhancedPetRecognizer:
    """增强宠物识别器 - 主类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 初始化各个组件
        self.detector = None
        self.species_classifier = None
        self.pose_estimator = None
        self.behavior_analyzer = None
        self.health_monitor = None
        
        # 跟踪相关
        self.tracking_history = {}
        self.next_tracking_id = 1
        
        # 统计信息
        self.stats = {
            'total_detections': 0,
            'species_counts': {species.value: 0 for species in PetSpecies},
            'behavior_counts': {behavior.value: 0 for behavior in PetBehavior},
            'health_alerts': 0,
            'processing_time': []
        }
        
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化各个组件"""
        try:
            # 初始化检测器
            if YOLO_AVAILABLE:
                model_path = self.config.get('yolo_model_path', 'yolov8n.pt')
                self.detector = YOLO(model_path)
                logger.info("YOLO detector initialized")
            else:
                logger.warning("YOLO not available, using basic detection")
            
            # 初始化物种分类器
            species_model_path = self.config.get('species_model_path')
            self.species_classifier = PetSpeciesClassifier(species_model_path)
            
            # 初始化姿态估计器
            self.pose_estimator = PetPoseEstimator()
            
            # 初始化行为分析器
            history_length = self.config.get('behavior_history_length', 30)
            self.behavior_analyzer = PetBehaviorAnalyzer(history_length)
            
            # 初始化健康监测器
            self.health_monitor = PetHealthMonitor()
            
            logger.info("Enhanced pet recognizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced pet recognizer: {e}")
    
    def detect_and_analyze(self, frame: np.ndarray, timestamp: Optional[float] = None) -> List[EnhancedPetResult]:
        """检测和分析宠物"""
        if timestamp is None:
            timestamp = time.time()
        
        start_time = time.time()
        results = []
        
        try:
            # 1. 宠物检测
            detections = self._detect_pets(frame)
            
            for detection in detections:
                self.stats['total_detections'] += 1
                self.stats['species_counts'][detection.species.value] += 1
                
                # 2. 分配跟踪ID
                tracking_id = self._assign_tracking_id(detection, timestamp)
                
                # 3. 姿态估计
                pose = None
                if self.pose_estimator:
                    pose = self.pose_estimator.estimate_pose(frame, detection.bbox)
                
                # 4. 行为分析
                behavior_result = None
                if self.behavior_analyzer:
                    behavior_result = self.behavior_analyzer.analyze_behavior(
                        detection, pose, tracking_id, timestamp
                    )
                    if behavior_result:
                        self.stats['behavior_counts'][behavior_result.behavior.value] += 1
                
                # 5. 健康监测
                health_result = None
                if self.health_monitor:
                    health_result = self.health_monitor.assess_health(
                        detection, behavior_result, tracking_id, timestamp
                    )
                    if health_result and health_result.status in [PetHealthStatus.SICK, PetHealthStatus.INJURED]:
                        self.stats['health_alerts'] += 1
                
                # 6. 创建综合结果
                result = EnhancedPetResult(
                    detection=detection,
                    pose=pose,
                    behavior=behavior_result,
                    health=health_result,
                    tracking_id=tracking_id,
                    timestamp=timestamp
                )
                results.append(result)
        
        except Exception as e:
            logger.error(f"Error in pet detection and analysis: {e}")
        
        # 记录处理时间
        processing_time = time.time() - start_time
        self.stats['processing_time'].append(processing_time)
        
        return results
    
    def _detect_pets(self, frame: np.ndarray) -> List[PetDetection]:
        """检测宠物"""
        detections = []
        
        try:
            if self.detector and YOLO_AVAILABLE:
                # 使用YOLO检测
                results = self.detector(frame, verbose=False)
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # 获取边界框
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            bbox = (int(x1), int(y1), int(x2-x1), int(y2-y1))
                            
                            # 获取置信度
                            confidence = float(box.conf[0])
                            
                            # 获取类别 (假设已经训练了宠物检测模型)
                            class_id = int(box.cls[0])
                            
                            # 物种分类
                            species, species_conf = self.species_classifier.classify(frame, bbox)
                            
                            detection = PetDetection(
                                species=species,
                                confidence=confidence * species_conf,
                                bbox=bbox,
                                breed=None,  # 可以添加品种识别
                                age_estimate=None,  # 可以添加年龄估计
                                size_category=self._estimate_size_category(bbox),
                                color_pattern=None  # 可以添加颜色模式识别
                            )
                            detections.append(detection)
            
            else:
                # 使用简单的颜色检测作为备选
                detections = self._simple_pet_detection(frame)
        
        except Exception as e:
            logger.debug(f"Pet detection error: {e}")
        
        return detections
    
    def _simple_pet_detection(self, frame: np.ndarray) -> List[PetDetection]:
        """简单的宠物检测 (备选方案)"""
        detections = []
        
        try:
            # 转换到HSV色彩空间
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 定义宠物常见颜色范围
            color_ranges = [
                # 棕色 (狗)
                ([10, 50, 50], [20, 255, 255], PetSpecies.DOG),
                # 灰色 (猫)
                ([0, 0, 50], [180, 50, 200], PetSpecies.CAT),
                # 白色 (各种宠物)
                ([0, 0, 200], [180, 30, 255], PetSpecies.UNKNOWN)
            ]
            
            for lower, upper, species in color_ranges:
                lower = np.array(lower)
                upper = np.array(upper)
                
                # 创建掩码
                mask = cv2.inRange(hsv, lower, upper)
                
                # 形态学操作
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
                # 查找轮廓
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 2000:  # 最小面积阈值
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # 计算置信度
                        confidence = min(0.7, area / 10000)
                        
                        detection = PetDetection(
                            species=species,
                            confidence=confidence,
                            bbox=(x, y, w, h),
                            size_category=self._estimate_size_category((x, y, w, h))
                        )
                        detections.append(detection)
        
        except Exception as e:
            logger.debug(f"Simple pet detection error: {e}")
        
        return detections
    
    def _estimate_size_category(self, bbox: Tuple[int, int, int, int]) -> str:
        """估计宠物大小类别"""
        x, y, w, h = bbox
        area = w * h
        
        if area < 5000:
            return "small"
        elif area < 20000:
            return "medium"
        else:
            return "large"
    
    def _assign_tracking_id(self, detection: PetDetection, timestamp: float) -> int:
        """分配跟踪ID"""
        min_distance = float('inf')
        best_id = None
        
        current_center = self._get_bbox_center(detection.bbox)
        
        # 查找最近的历史轨迹
        for tracking_id, history in self.tracking_history.items():
            if history and history[-1]['detection'].species == detection.species:
                last_center = self._get_bbox_center(history[-1]['detection'].bbox)
                distance = np.linalg.norm(np.array(current_center) - np.array(last_center))
                
                # 检查时间间隔
                time_diff = timestamp - history[-1]['timestamp']
                if distance < min_distance and distance < 100 and time_diff < 5.0:  # 5秒内
                    min_distance = distance
                    best_id = tracking_id
        
        # 如果没有找到匹配的轨迹，创建新的
        if best_id is None:
            best_id = self.next_tracking_id
            self.next_tracking_id += 1
            self.tracking_history[best_id] = []
        
        # 更新轨迹历史
        self.tracking_history[best_id].append({
            'detection': detection,
            'timestamp': timestamp
        })
        
        # 限制历史长度
        if len(self.tracking_history[best_id]) > 100:
            self.tracking_history[best_id] = self.tracking_history[best_id][-100:]
        
        return best_id
    
    def _get_bbox_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """获取边界框中心"""
        x, y, w, h = bbox
        return (x + w / 2, y + h / 2)
    
    def draw_results(self, frame: np.ndarray, results: List[EnhancedPetResult]) -> np.ndarray:
        """绘制识别结果"""
        annotated_frame = frame.copy()
        
        for result in results:
            detection = result.detection
            x, y, w, h = detection.bbox
            
            # 绘制边界框
            color = self._get_species_color(detection.species)
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
            
            # 绘制标签
            label_parts = [
                f"{detection.species.value}",
                f"{detection.confidence:.2f}"
            ]
            
            if detection.breed:
                label_parts.append(detection.breed)
            
            if result.tracking_id:
                label_parts.append(f"ID:{result.tracking_id}")
            
            label = " | ".join(label_parts)
            
            # 绘制标签背景
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (x, y - label_h - 10), (x + label_w, y), color, -1)
            cv2.putText(annotated_frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 绘制姿态关键点
            if result.pose:
                self._draw_pose_keypoints(annotated_frame, result.pose)
            
            # 绘制行为信息
            if result.behavior:
                behavior_text = f"{result.behavior.behavior.value} ({result.behavior.confidence:.2f})"
                cv2.putText(annotated_frame, behavior_text, (x, y + h + 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # 绘制健康状态
            if result.health:
                health_color = self._get_health_color(result.health.status)
                health_text = f"Health: {result.health.status.value}"
                cv2.putText(annotated_frame, health_text, (x, y + h + 40), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, health_color, 1)
        
        return annotated_frame
    
    def _get_species_color(self, species: PetSpecies) -> Tuple[int, int, int]:
        """获取物种对应的颜色"""
        color_map = {
            PetSpecies.DOG: (0, 255, 0),      # 绿色
            PetSpecies.CAT: (255, 0, 0),      # 蓝色
            PetSpecies.BIRD: (0, 255, 255),   # 黄色
            PetSpecies.RABBIT: (255, 0, 255), # 洋红
            PetSpecies.HAMSTER: (0, 165, 255), # 橙色
            PetSpecies.GUINEA_PIG: (128, 0, 128), # 紫色
            PetSpecies.FERRET: (0, 128, 255),  # 橙红色
            PetSpecies.FISH: (255, 255, 0),    # 青色
            PetSpecies.REPTILE: (128, 128, 0), # 橄榄色
            PetSpecies.UNKNOWN: (128, 128, 128) # 灰色
        }
        return color_map.get(species, (128, 128, 128))
    
    def _get_health_color(self, status: PetHealthStatus) -> Tuple[int, int, int]:
        """获取健康状态对应的颜色"""
        color_map = {
            PetHealthStatus.HEALTHY: (0, 255, 0),      # 绿色
            PetHealthStatus.SICK: (0, 0, 255),         # 红色
            PetHealthStatus.INJURED: (0, 0, 255),      # 红色
            PetHealthStatus.STRESSED: (0, 165, 255),   # 橙色
            PetHealthStatus.LETHARGIC: (0, 255, 255),  # 黄色
            PetHealthStatus.HYPERACTIVE: (255, 0, 255), # 洋红
            PetHealthStatus.UNKNOWN: (128, 128, 128)   # 灰色
        }
        return color_map.get(status, (128, 128, 128))
    
    def _draw_pose_keypoints(self, frame: np.ndarray, pose: PetPose):
        """绘制姿态关键点"""
        for i, (x, y, confidence) in enumerate(pose.keypoints):
            if confidence > 0.5:  # 只绘制置信度高的关键点
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        
        if self.stats['processing_time']:
            stats['avg_processing_time'] = np.mean(self.stats['processing_time'])
            stats['max_processing_time'] = np.max(self.stats['processing_time'])
            stats['min_processing_time'] = np.min(self.stats['processing_time'])
        
        return stats
    
    def reset_statistics(self):
        """重置统计信息"""
        self.stats = {
            'total_detections': 0,
            'species_counts': {species.value: 0 for species in PetSpecies},
            'behavior_counts': {behavior.value: 0 for behavior in PetBehavior},
            'health_alerts': 0,
            'processing_time': []
        }
    
    def cleanup(self):
        """清理资源"""
        self.tracking_history.clear()
        self.reset_statistics()
        
        if self.pose_estimator and self.pose_estimator.pose_detector:
            self.pose_estimator.pose_detector.close()
        
        logger.info("Enhanced pet recognizer cleaned up")


# 导出主要类
__all__ = [
    'EnhancedPetRecognizer',
    'PetSpecies',
    'PetBehavior',
    'PetHealthStatus',
    'PetDetection',
    'PetPose',
    'PetBehaviorResult',
    'PetHealthResult',
    'EnhancedPetResult'
]