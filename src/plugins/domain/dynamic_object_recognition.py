"""动态物体识别插件

支持多种动态物体的检测、跟踪和行为分析，包括移动的车辆、飞行物、运动设备等。
提供运动检测、轨迹跟踪、速度估算、行为预测等功能。
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Deque
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from collections import deque
import math

from ...core.base_plugin import DomainPlugin, PluginMetadata, PluginCapability
from ...core.event_bus import EventBus


class DynamicObjectType(Enum):
    """动态物体类型枚举"""
    VEHICLE = "vehicle"
    AIRCRAFT = "aircraft"
    WATERCRAFT = "watercraft"
    SPORTS_EQUIPMENT = "sports_equipment"
    PROJECTILE = "projectile"
    DEBRIS = "debris"
    UNKNOWN_MOVING = "unknown_moving"


class VehicleType(Enum):
    """车辆类型枚举"""
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"
    SCOOTER = "scooter"
    TRAIN = "train"
    UNKNOWN = "unknown"


class MovementPattern(Enum):
    """运动模式枚举"""
    LINEAR = "linear"
    CIRCULAR = "circular"
    ZIGZAG = "zigzag"
    RANDOM = "random"
    STATIONARY = "stationary"
    ACCELERATING = "accelerating"
    DECELERATING = "decelerating"
    UNKNOWN = "unknown"


class MotionState(Enum):
    """运动状态枚举"""
    MOVING = "moving"
    STOPPED = "stopped"
    TURNING = "turning"
    REVERSING = "reversing"
    PARKING = "parking"
    DEPARTING = "departing"
    UNKNOWN = "unknown"


@dataclass
class MotionVector:
    """运动向量"""
    velocity: Tuple[float, float]  # x, y方向速度 (像素/帧)
    speed: float  # 速度大小
    direction: float  # 方向角度（弧度）
    acceleration: Tuple[float, float]  # 加速度


@dataclass
class TrajectoryPoint:
    """轨迹点"""
    position: Tuple[int, int]
    timestamp: float
    velocity: Optional[MotionVector] = None


@dataclass
class DynamicObjectDetection:
    """动态物体检测结果"""
    object_type: DynamicObjectType
    vehicle_type: Optional[VehicleType]
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    motion_vector: Optional[MotionVector] = None
    size_estimate: Optional[Tuple[float, float]] = None  # 长宽估计
    

@dataclass
class MotionAnalysisResult:
    """运动分析结果"""
    pattern: MovementPattern
    state: MotionState
    confidence: float
    trajectory: List[TrajectoryPoint]
    predicted_path: List[Tuple[int, int]]  # 预测路径
    collision_risk: float  # 碰撞风险评估
    

@dataclass
class DynamicObjectResult:
    """动态物体识别综合结果"""
    detection: DynamicObjectDetection
    motion_analysis: Optional[MotionAnalysisResult] = None
    tracking_id: Optional[int] = None
    timestamp: float = 0.0


class BaseDynamicObjectDetector(ABC):
    """动态物体检测器基类"""
    
    @abstractmethod
    def detect(self, frame: np.ndarray, motion_mask: np.ndarray) -> List[DynamicObjectDetection]:
        """检测动态物体"""
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[DynamicObjectType]:
        """获取支持的物体类型"""
        pass


class BaseMotionAnalyzer(ABC):
    """运动分析器基类"""
    
    @abstractmethod
    def analyze_motion(self, detection: DynamicObjectDetection,
                      trajectory: List[TrajectoryPoint]) -> MotionAnalysisResult:
        """分析物体运动"""
        pass


class MotionBasedDetector(BaseDynamicObjectDetector):
    """基于运动的物体检测器"""
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50
        )
        self.min_contour_area = 500
    
    def detect(self, frame: np.ndarray, motion_mask: Optional[np.ndarray] = None) -> List[DynamicObjectDetection]:
        """检测动态物体"""
        detections = []
        
        # 如果没有提供运动掩码，使用背景减除生成
        if motion_mask is None:
            motion_mask = self.background_subtractor.apply(frame)
        
        # 形态学操作去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue
            
            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 分析物体特征
            object_type, vehicle_type = self._classify_object(frame, x, y, w, h, contour)
            
            # 计算置信度
            confidence = self._calculate_confidence(area, w, h, contour)
            
            if confidence >= self.confidence_threshold:
                detection = DynamicObjectDetection(
                    object_type=object_type,
                    vehicle_type=vehicle_type,
                    confidence=confidence,
                    bbox=(x, y, w, h),
                    size_estimate=(w * 0.1, h * 0.1)  # 简单的尺寸估计
                )
                detections.append(detection)
        
        return detections
    
    def _classify_object(self, frame: np.ndarray, x: int, y: int, w: int, h: int,
                        contour: np.ndarray) -> Tuple[DynamicObjectType, Optional[VehicleType]]:
        """分类物体类型"""
        aspect_ratio = w / h if h > 0 else 0
        area = cv2.contourArea(contour)
        
        # 基于尺寸和形状特征分类
        if 1.5 <= aspect_ratio <= 4.0 and area > 2000:  # 车辆特征
            if area > 10000:
                return DynamicObjectType.VEHICLE, VehicleType.TRUCK
            elif area > 5000:
                return DynamicObjectType.VEHICLE, VehicleType.CAR
            else:
                return DynamicObjectType.VEHICLE, VehicleType.MOTORCYCLE
        elif aspect_ratio > 4.0:  # 长条形物体
            return DynamicObjectType.VEHICLE, VehicleType.TRAIN
        elif 0.8 <= aspect_ratio <= 1.2:  # 接近正方形
            if area < 1000:
                return DynamicObjectType.PROJECTILE, None
            else:
                return DynamicObjectType.SPORTS_EQUIPMENT, None
        elif aspect_ratio < 0.5:  # 高瘦物体
            return DynamicObjectType.UNKNOWN_MOVING, None
        else:
            return DynamicObjectType.UNKNOWN_MOVING, None
    
    def _calculate_confidence(self, area: float, width: int, height: int,
                            contour: np.ndarray) -> float:
        """计算检测置信度"""
        # 基于面积、形状规整度等因素
        aspect_ratio = width / height if height > 0 else 0
        
        # 面积因子
        area_factor = min(1.0, area / 5000)
        
        # 形状因子（接近矩形的形状得分更高）
        rect_area = width * height
        shape_factor = area / rect_area if rect_area > 0 else 0
        
        # 长宽比因子（合理的长宽比得分更高）
        ratio_factor = 1.0 - abs(aspect_ratio - 2.0) / 3.0
        ratio_factor = max(0.0, ratio_factor)
        
        confidence = (area_factor * 0.4 + shape_factor * 0.4 + ratio_factor * 0.2)
        return min(1.0, confidence)
    
    def get_supported_types(self) -> List[DynamicObjectType]:
        """获取支持的物体类型"""
        return [DynamicObjectType.VEHICLE, DynamicObjectType.SPORTS_EQUIPMENT,
                DynamicObjectType.PROJECTILE, DynamicObjectType.UNKNOWN_MOVING]


class MotionAnalyzer(BaseMotionAnalyzer):
    """运动分析器"""
    
    def __init__(self, max_trajectory_length: int = 30):
        self.max_trajectory_length = max_trajectory_length
        self.prediction_steps = 10  # 预测未来10帧
    
    def analyze_motion(self, detection: DynamicObjectDetection,
                      trajectory: List[TrajectoryPoint]) -> MotionAnalysisResult:
        """分析物体运动"""
        if len(trajectory) < 2:
            return MotionAnalysisResult(
                pattern=MovementPattern.UNKNOWN,
                state=MotionState.UNKNOWN,
                confidence=0.0,
                trajectory=trajectory,
                predicted_path=[],
                collision_risk=0.0
            )
        
        # 分析运动模式
        pattern = self._analyze_movement_pattern(trajectory)
        
        # 分析运动状态
        state = self._analyze_motion_state(trajectory)
        
        # 预测路径
        predicted_path = self._predict_path(trajectory)
        
        # 评估碰撞风险
        collision_risk = self._assess_collision_risk(trajectory, predicted_path)
        
        return MotionAnalysisResult(
            pattern=pattern,
            state=state,
            confidence=0.8,
            trajectory=trajectory,
            predicted_path=predicted_path,
            collision_risk=collision_risk
        )
    
    def _analyze_movement_pattern(self, trajectory: List[TrajectoryPoint]) -> MovementPattern:
        """分析运动模式"""
        if len(trajectory) < 3:
            return MovementPattern.UNKNOWN
        
        # 计算方向变化
        direction_changes = []
        speeds = []
        
        for i in range(1, len(trajectory)):
            p1 = trajectory[i-1].position
            p2 = trajectory[i].position
            
            # 计算速度
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            speed = math.sqrt(dx*dx + dy*dy)
            speeds.append(speed)
            
            # 计算方向
            if speed > 0:
                direction = math.atan2(dy, dx)
                if len(direction_changes) > 0:
                    angle_diff = abs(direction - direction_changes[-1])
                    # 处理角度跳跃
                    if angle_diff > math.pi:
                        angle_diff = 2 * math.pi - angle_diff
                    direction_changes.append(angle_diff)
                else:
                    direction_changes.append(0)
        
        if not direction_changes or not speeds:
            return MovementPattern.UNKNOWN
        
        avg_speed = np.mean(speeds)
        avg_direction_change = np.mean(direction_changes)
        speed_variance = np.var(speeds)
        
        # 判断运动模式
        if avg_speed < 1.0:
            return MovementPattern.STATIONARY
        elif avg_direction_change < 0.1:  # 方向变化很小
            return MovementPattern.LINEAR
        elif avg_direction_change > 1.0:  # 方向变化很大
            return MovementPattern.ZIGZAG
        elif speed_variance > avg_speed * 0.5:  # 速度变化大
            if np.mean(speeds[-3:]) > np.mean(speeds[:3]):
                return MovementPattern.ACCELERATING
            else:
                return MovementPattern.DECELERATING
        else:
            # 检查是否为圆周运动
            if self._is_circular_motion(trajectory):
                return MovementPattern.CIRCULAR
            else:
                return MovementPattern.RANDOM
    
    def _is_circular_motion(self, trajectory: List[TrajectoryPoint]) -> bool:
        """检测是否为圆周运动"""
        if len(trajectory) < 5:
            return False
        
        positions = [p.position for p in trajectory]
        
        # 计算轨迹的曲率
        curvatures = []
        for i in range(1, len(positions) - 1):
            p1, p2, p3 = positions[i-1], positions[i], positions[i+1]
            
            # 计算三点的曲率
            curvature = self._calculate_curvature(p1, p2, p3)
            if curvature is not None:
                curvatures.append(curvature)
        
        if not curvatures:
            return False
        
        # 如果曲率相对稳定且不为零，可能是圆周运动
        avg_curvature = np.mean(curvatures)
        curvature_std = np.std(curvatures)
        
        return avg_curvature > 0.01 and curvature_std < avg_curvature * 0.5
    
    def _calculate_curvature(self, p1: Tuple[int, int], p2: Tuple[int, int], 
                           p3: Tuple[int, int]) -> Optional[float]:
        """计算三点的曲率"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        
        # 计算向量
        v1 = (x2 - x1, y2 - y1)
        v2 = (x3 - x2, y3 - y2)
        
        # 计算叉积和模长
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]
        v1_mag = math.sqrt(v1[0]**2 + v1[1]**2)
        v2_mag = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if v1_mag == 0 or v2_mag == 0:
            return None
        
        # 曲率 = |叉积| / (|v1| * |v2|)
        curvature = abs(cross_product) / (v1_mag * v2_mag)
        return curvature
    
    def _analyze_motion_state(self, trajectory: List[TrajectoryPoint]) -> MotionState:
        """分析运动状态"""
        if len(trajectory) < 2:
            return MotionState.UNKNOWN
        
        # 计算最近几帧的速度
        recent_speeds = []
        for i in range(max(1, len(trajectory) - 5), len(trajectory)):
            p1 = trajectory[i-1].position
            p2 = trajectory[i].position
            speed = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            recent_speeds.append(speed)
        
        avg_speed = np.mean(recent_speeds)
        
        if avg_speed < 1.0:
            return MotionState.STOPPED
        elif avg_speed < 3.0:
            return MotionState.PARKING
        else:
            # 检查方向变化
            if len(trajectory) >= 3:
                last_positions = [p.position for p in trajectory[-3:]]
                direction_changes = []
                
                for i in range(1, len(last_positions)):
                    p1, p2 = last_positions[i-1], last_positions[i]
                    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                    if dx != 0 or dy != 0:
                        direction = math.atan2(dy, dx)
                        direction_changes.append(direction)
                
                if len(direction_changes) >= 2:
                    angle_diff = abs(direction_changes[-1] - direction_changes[-2])
                    if angle_diff > math.pi:
                        angle_diff = 2 * math.pi - angle_diff
                    
                    if angle_diff > math.pi / 4:  # 45度以上的方向变化
                        return MotionState.TURNING
            
            return MotionState.MOVING
    
    def _predict_path(self, trajectory: List[TrajectoryPoint]) -> List[Tuple[int, int]]:
        """预测未来路径"""
        if len(trajectory) < 2:
            return []
        
        # 使用最近的运动向量进行线性预测
        last_pos = trajectory[-1].position
        
        if len(trajectory) >= 2:
            prev_pos = trajectory[-2].position
            velocity = (last_pos[0] - prev_pos[0], last_pos[1] - prev_pos[1])
        else:
            velocity = (0, 0)
        
        # 预测未来位置
        predicted_path = []
        current_pos = last_pos
        
        for step in range(1, self.prediction_steps + 1):
            next_x = current_pos[0] + velocity[0] * step
            next_y = current_pos[1] + velocity[1] * step
            predicted_path.append((int(next_x), int(next_y)))
        
        return predicted_path
    
    def _assess_collision_risk(self, trajectory: List[TrajectoryPoint],
                             predicted_path: List[Tuple[int, int]]) -> float:
        """评估碰撞风险"""
        if not predicted_path or len(trajectory) < 2:
            return 0.0
        
        # 简单的碰撞风险评估
        # 基于速度和方向变化
        recent_speeds = []
        for i in range(max(1, len(trajectory) - 3), len(trajectory)):
            p1 = trajectory[i-1].position
            p2 = trajectory[i].position
            speed = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            recent_speeds.append(speed)
        
        avg_speed = np.mean(recent_speeds) if recent_speeds else 0
        
        # 速度越高，风险越大
        speed_risk = min(1.0, avg_speed / 20.0)
        
        # 如果预测路径超出边界，风险增加
        boundary_risk = 0.0
        for pos in predicted_path:
            if pos[0] < 0 or pos[1] < 0:  # 简化的边界检查
                boundary_risk = 0.3
                break
        
        return min(1.0, speed_risk + boundary_risk)


class DynamicObjectRecognitionPlugin(DomainPlugin):
    """动态物体识别插件"""
    
    def __init__(self):
        metadata = PluginMetadata(
            name="dynamic_object_recognition",
            version="1.0.0",
            description="动态物体识别和运动分析插件",
            author="YOLOS Team",
            capabilities=[
                PluginCapability.DETECTION,
                PluginCapability.TRACKING,
                PluginCapability.ANALYSIS
            ],
            dependencies=["opencv-python", "numpy"]
        )
        super().__init__(metadata)
        
        self.detector = None
        self.motion_analyzer = None
        self.tracking_history = {}
        self.trajectory_history = {}
        self.next_tracking_id = 1
        
        # 统计信息
        self.stats = {
            'total_detections': 0,
            'object_type_counts': {obj_type.value: 0 for obj_type in DynamicObjectType},
            'vehicle_type_counts': {v_type.value: 0 for v_type in VehicleType},
            'motion_pattern_counts': {pattern.value: 0 for pattern in MovementPattern},
            'collision_alerts': 0
        }
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化插件"""
        try:
            # 初始化检测器
            confidence_threshold = config.get('dynamic_confidence_threshold', 0.5)
            self.detector = MotionBasedDetector(confidence_threshold)
            
            # 初始化运动分析器
            max_trajectory_length = config.get('max_trajectory_length', 30)
            self.motion_analyzer = MotionAnalyzer(max_trajectory_length)
            
            self.logger.info("Dynamic object recognition plugin initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize dynamic object recognition plugin: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray, **kwargs) -> List[DynamicObjectResult]:
        """处理视频帧"""
        if not self.detector:
            return []
        
        results = []
        
        try:
            # 检测动态物体
            motion_mask = kwargs.get('motion_mask')
            detections = self.detector.detect(frame, motion_mask)
            
            current_time = kwargs.get('timestamp', 0.0)
            
            for detection in detections:
                # 更新统计
                self.stats['total_detections'] += 1
                self.stats['object_type_counts'][detection.object_type.value] += 1
                if detection.vehicle_type:
                    self.stats['vehicle_type_counts'][detection.vehicle_type.value] += 1
                
                # 跟踪处理
                tracking_id = self._assign_tracking_id(detection)
                
                # 更新轨迹
                center = self._get_center(detection.bbox)
                trajectory_point = TrajectoryPoint(
                    position=center,
                    timestamp=current_time
                )
                
                if tracking_id not in self.trajectory_history:
                    self.trajectory_history[tracking_id] = deque(maxlen=30)
                self.trajectory_history[tracking_id].append(trajectory_point)
                
                # 运动分析
                motion_analysis = None
                if self.motion_analyzer and len(self.trajectory_history[tracking_id]) >= 2:
                    trajectory = list(self.trajectory_history[tracking_id])
                    motion_analysis = self.motion_analyzer.analyze_motion(detection, trajectory)
                    
                    self.stats['motion_pattern_counts'][motion_analysis.pattern.value] += 1
                    
                    # 碰撞风险警报
                    if motion_analysis.collision_risk > 0.7:
                        self.stats['collision_alerts'] += 1
                        EventBus.emit('collision_risk_alert', {
                            'tracking_id': tracking_id,
                            'object_type': detection.object_type.value,
                            'risk_level': motion_analysis.collision_risk,
                            'predicted_path': motion_analysis.predicted_path
                        })
                
                # 更新跟踪历史
                if tracking_id not in self.tracking_history:
                    self.tracking_history[tracking_id] = []
                self.tracking_history[tracking_id].append(detection)
                
                # 限制历史长度
                if len(self.tracking_history[tracking_id]) > 20:
                    self.tracking_history[tracking_id] = self.tracking_history[tracking_id][-20:]
                
                # 创建结果
                result = DynamicObjectResult(
                    detection=detection,
                    motion_analysis=motion_analysis,
                    tracking_id=tracking_id,
                    timestamp=current_time
                )
                results.append(result)
        
        except Exception as e:
            self.logger.error(f"Error processing frame in dynamic object recognition: {e}")
        
        return results
    
    def _assign_tracking_id(self, detection: DynamicObjectDetection) -> int:
        """分配跟踪ID"""
        # 基于位置距离和物体类型的跟踪
        min_distance = float('inf')
        best_id = None
        
        current_center = self._get_center(detection.bbox)
        
        for tracking_id, history in self.tracking_history.items():
            if history and history[-1].object_type == detection.object_type:
                last_center = self._get_center(history[-1].bbox)
                distance = np.linalg.norm(np.array(current_center) - np.array(last_center))
                
                # 动态物体可能移动较快，增大距离阈值
                if distance < min_distance and distance < 100:
                    min_distance = distance
                    best_id = tracking_id
        
        if best_id is None:
            best_id = self.next_tracking_id
            self.next_tracking_id += 1
        
        return best_id
    
    def _get_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """获取边界框中心点"""
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)
    
    def get_supported_domains(self) -> List[str]:
        """获取支持的识别领域"""
        return ['dynamic_objects', 'vehicles', 'motion_tracking', 'traffic_monitoring']
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
    
    def cleanup(self):
        """清理资源"""
        self.tracking_history.clear()
        self.trajectory_history.clear()
        self.stats = {
            'total_detections': 0,
            'object_type_counts': {obj_type.value: 0 for obj_type in DynamicObjectType},
            'vehicle_type_counts': {v_type.value: 0 for v_type in VehicleType},
            'motion_pattern_counts': {pattern.value: 0 for pattern in MovementPattern},
            'collision_alerts': 0
        }
        self.logger.info("Dynamic object recognition plugin cleaned up")


# 导出插件类
__all__ = [
    'DynamicObjectRecognitionPlugin',
    'DynamicObjectType',
    'VehicleType',
    'MovementPattern',
    'MotionState',
    'MotionVector',
    'TrajectoryPoint',
    'DynamicObjectDetection',
    'MotionAnalysisResult',
    'DynamicObjectResult'
]