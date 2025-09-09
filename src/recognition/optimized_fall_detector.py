"""优化版摔倒检测模块 - 高精度摔倒检测，减少误报"""

import cv2
import numpy as np
import math
import time
import os
from typing import List, Dict, Tuple, Optional, Any
from collections import deque
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 尝试导入相关库
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not available")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("YOLO not available")

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


@dataclass
class FallDetectionResult:
    """摔倒检测结果数据类"""
    person_id: int
    is_falling: bool
    fall_confidence: float
    fall_stage: str  # 'pre_fall', 'falling', 'fallen', 'recovery', 'normal'
    fall_direction: str  # 'forward', 'backward', 'left', 'right', 'unknown'
    fall_severity: str  # 'mild', 'moderate', 'severe'
    bbox: Tuple[int, int, int, int]
    keypoints: List[Tuple[float, float, float]]
    body_orientation: float  # 身体倾斜角度
    velocity_vector: Tuple[float, float]  # 运动速度向量
    acceleration: float  # 加速度
    stability_score: float  # 稳定性评分
    ground_contact_ratio: float  # 地面接触比例
    alert_level: str  # 'none', 'warning', 'critical', 'emergency'
    timestamp: float
    duration: float  # 摔倒持续时间


@dataclass
class MultiFallDetectionResult:
    """多人摔倒检测结果数据类"""
    total_persons: int
    fall_detections: List[FallDetectionResult]
    active_falls: int
    critical_alerts: int
    scene_risk_level: str  # 'low', 'medium', 'high', 'critical'
    environmental_factors: Dict[str, Any]
    performance_metrics: Dict[str, float]
    timestamp: float


class OptimizedFallDetector:
    """优化版摔倒检测器 - 高精度、低误报、多阶段检测"""
    
    def __init__(self,
                 sensitivity: float = 0.7,
                 min_fall_duration: float = 0.5,
                 max_false_positive_rate: float = 0.05,
                 enable_multi_stage: bool = True,
                 enable_velocity_analysis: bool = True,
                 enable_pose_analysis: bool = True,
                 enable_environmental_analysis: bool = True,
                 sequence_length: int = 30,
                 alert_delay: float = 1.0):
        """
        初始化优化版摔倒检测器
        
        Args:
            sensitivity: 检测敏感度 (0.0-1.0)
            min_fall_duration: 最小摔倒持续时间（秒）
            max_false_positive_rate: 最大误报率
            enable_multi_stage: 是否启用多阶段检测
            enable_velocity_analysis: 是否启用速度分析
            enable_pose_analysis: 是否启用姿势分析
            enable_environmental_analysis: 是否启用环境分析
            sequence_length: 历史序列长度
            alert_delay: 警报延迟时间（秒）
        """
        self.sensitivity = sensitivity
        self.min_fall_duration = min_fall_duration
        self.max_false_positive_rate = max_false_positive_rate
        self.enable_multi_stage = enable_multi_stage
        self.enable_velocity_analysis = enable_velocity_analysis
        self.enable_pose_analysis = enable_pose_analysis
        self.enable_environmental_analysis = enable_environmental_analysis
        self.sequence_length = sequence_length
        self.alert_delay = alert_delay
        
        # 检测阈值配置
        self.thresholds = {
            'body_angle_threshold': 45,  # 身体倾斜角度阈值（度）
            'velocity_threshold': 50,    # 速度阈值（像素/帧）
            'acceleration_threshold': 30, # 加速度阈值
            'ground_contact_threshold': 0.3, # 地面接触阈值
            'stability_threshold': 0.4,  # 稳定性阈值
            'aspect_ratio_threshold': 1.5, # 长宽比阈值
            'height_change_threshold': 0.3, # 高度变化阈值
            'duration_threshold': 0.5    # 持续时间阈值
        }
        
        # 历史数据存储
        self.person_histories = {}  # 每个人的历史数据
        self.fall_events = {}       # 活跃的摔倒事件
        self.alert_timers = {}      # 警报计时器
        
        # 性能统计
        self.performance_stats = {
            'total_detections': 0,
            'fall_detections': 0,
            'false_positives': 0,
            'true_positives': 0,
            'avg_processing_time': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
        
        # 线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        
        # 初始化检测模型
        self._init_detection_models()
        
        # 摔倒阶段定义
        self.fall_stages = {
            'normal': '正常状态',
            'pre_fall': '摔倒前兆',
            'falling': '正在摔倒',
            'fallen': '已摔倒',
            'recovery': '恢复中'
        }
        
        # 警报级别定义
        self.alert_levels = {
            'none': '无警报',
            'warning': '警告',
            'critical': '严重',
            'emergency': '紧急'
        }
        
        logger.info("优化版摔倒检测器初始化完成")
    
    def _init_detection_models(self):
        """初始化检测模型"""
        try:
            # 初始化姿势检测器（用于获取关键点）
            if MEDIAPIPE_AVAILABLE:
                self.mp_pose = mp.solutions.pose
                self.pose_detector = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5
                )
                logger.info("MediaPipe姿势检测器初始化成功")
            
            # 初始化YOLO模型（用于人体检测）
            if YOLO_AVAILABLE:
                self.yolo_model = YOLO(os.path.join(os.getcwd(), 'module', 'yolov8n-pose.pt'))
                logger.info("YOLO模型初始化成功")
            
            # 初始化深度学习摔倒检测模型（如果可用）
            if PYTORCH_AVAILABLE:
                self._init_deep_learning_model()
            
        except Exception as e:
            logger.error(f"检测模型初始化失败: {e}")
    
    def _init_deep_learning_model(self):
        """初始化深度学习摔倒检测模型"""
        try:
            # 简单的LSTM模型用于序列分析
            class FallDetectionLSTM(nn.Module):
                def __init__(self, input_size=34, hidden_size=64, num_layers=2, num_classes=2):
                    super(FallDetectionLSTM, self).__init__()
                    self.hidden_size = hidden_size
                    self.num_layers = num_layers
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                    self.fc = nn.Linear(hidden_size, num_classes)
                    self.dropout = nn.Dropout(0.2)
                
                def forward(self, x):
                    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                    out, _ = self.lstm(x, (h0, c0))
                    out = self.dropout(out[:, -1, :])
                    out = self.fc(out)
                    return out
            
            self.lstm_model = FallDetectionLSTM()
            self.lstm_model.eval()
            logger.info("深度学习摔倒检测模型初始化成功")
            
        except Exception as e:
            logger.error(f"深度学习模型初始化失败: {e}")
            self.lstm_model = None
    
    def detect_falls(self, persons_data: List[Any]) -> MultiFallDetectionResult:
        """
        检测多人摔倒情况
        
        Args:
            persons_data: 人员姿势数据列表
            
        Returns:
            MultiFallDetectionResult: 多人摔倒检测结果
        """
        start_time = time.time()
        
        try:
            fall_detections = []
            active_falls = 0
            critical_alerts = 0
            
            # 为每个人员进行摔倒检测
            for person_data in persons_data:
                fall_result = self._detect_person_fall(person_data)
                fall_detections.append(fall_result)
                
                if fall_result.is_falling:
                    active_falls += 1
                
                if fall_result.alert_level in ['critical', 'emergency']:
                    critical_alerts += 1
            
            # 场景风险评估
            scene_risk_level = self._assess_scene_risk(fall_detections)
            
            # 环境因素分析
            environmental_factors = self._analyze_environmental_factors(persons_data)
            
            # 更新性能统计
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time, len(persons_data), active_falls)
            
            return MultiFallDetectionResult(
                total_persons=len(persons_data),
                fall_detections=fall_detections,
                active_falls=active_falls,
                critical_alerts=critical_alerts,
                scene_risk_level=scene_risk_level,
                environmental_factors=environmental_factors,
                performance_metrics=self._get_performance_metrics(),
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"摔倒检测错误: {e}")
            return self._get_default_result()
    
    def _detect_person_fall(self, person_data: Any) -> FallDetectionResult:
        """检测单个人员的摔倒情况"""
        try:
            person_id = getattr(person_data, 'person_id', 0)
            bbox = getattr(person_data, 'bbox', (0, 0, 0, 0))
            keypoints = getattr(person_data, 'keypoints', [])
            
            # 更新人员历史数据
            self._update_person_history(person_id, person_data)
            
            # 多阶段摔倒检测
            fall_indicators = self._analyze_fall_indicators(person_id, person_data)
            
            # 综合判断是否摔倒
            is_falling, fall_confidence = self._determine_fall_status(fall_indicators)
            
            # 确定摔倒阶段
            fall_stage = self._determine_fall_stage(person_id, fall_indicators)
            
            # 分析摔倒方向
            fall_direction = self._analyze_fall_direction(person_id, keypoints)
            
            # 评估摔倒严重程度
            fall_severity = self._assess_fall_severity(fall_indicators)
            
            # 计算身体方向
            body_orientation = self._calculate_body_orientation(keypoints)
            
            # 计算运动向量
            velocity_vector = self._calculate_velocity_vector(person_id)
            
            # 计算加速度
            acceleration = self._calculate_acceleration(person_id)
            
            # 计算稳定性评分
            stability_score = self._calculate_stability_score(keypoints)
            
            # 计算地面接触比例
            ground_contact_ratio = self._calculate_ground_contact_ratio(keypoints, bbox)
            
            # 确定警报级别
            alert_level = self._determine_alert_level(fall_indicators, is_falling)
            
            # 计算摔倒持续时间
            duration = self._calculate_fall_duration(person_id, is_falling)
            
            return FallDetectionResult(
                person_id=person_id,
                is_falling=is_falling,
                fall_confidence=fall_confidence,
                fall_stage=fall_stage,
                fall_direction=fall_direction,
                fall_severity=fall_severity,
                bbox=bbox,
                keypoints=keypoints,
                body_orientation=body_orientation,
                velocity_vector=velocity_vector,
                acceleration=acceleration,
                stability_score=stability_score,
                ground_contact_ratio=ground_contact_ratio,
                alert_level=alert_level,
                timestamp=time.time(),
                duration=duration
            )
            
        except Exception as e:
            logger.error(f"单人摔倒检测错误: {e}")
            return self._get_default_person_result(getattr(person_data, 'person_id', 0))
    
    def _update_person_history(self, person_id: int, person_data: Any):
        """更新人员历史数据"""
        try:
            if person_id not in self.person_histories:
                self.person_histories[person_id] = deque(maxlen=self.sequence_length)
            
            # 提取关键特征
            features = {
                'timestamp': time.time(),
                'bbox': getattr(person_data, 'bbox', (0, 0, 0, 0)),
                'keypoints': getattr(person_data, 'keypoints', []),
                'pose_type': getattr(person_data, 'pose_type', 'unknown'),
                'stability_score': getattr(person_data, 'stability_score', 0.5),
                'confidence': getattr(person_data, 'confidence', 0.5)
            }
            
            self.person_histories[person_id].append(features)
            
        except Exception as e:
            logger.error(f"人员历史数据更新错误: {e}")
    
    def _analyze_fall_indicators(self, person_id: int, person_data: Any) -> Dict[str, float]:
        """分析摔倒指标"""
        indicators = {
            'body_angle_score': 0.0,
            'velocity_score': 0.0,
            'acceleration_score': 0.0,
            'pose_change_score': 0.0,
            'stability_score': 0.0,
            'aspect_ratio_score': 0.0,
            'height_change_score': 0.0,
            'ground_contact_score': 0.0,
            'temporal_consistency_score': 0.0
        }
        
        try:
            if person_id not in self.person_histories or len(self.person_histories[person_id]) < 2:
                return indicators
            
            history = list(self.person_histories[person_id])
            current_data = history[-1]
            previous_data = history[-2]
            
            # 身体角度分析
            indicators['body_angle_score'] = self._analyze_body_angle(current_data['keypoints'])
            
            # 速度分析
            if self.enable_velocity_analysis:
                indicators['velocity_score'] = self._analyze_velocity(current_data, previous_data)
            
            # 加速度分析
            if len(history) >= 3:
                indicators['acceleration_score'] = self._analyze_acceleration(history[-3:])
            
            # 姿势变化分析
            if self.enable_pose_analysis:
                indicators['pose_change_score'] = self._analyze_pose_change(current_data, previous_data)
            
            # 稳定性分析
            indicators['stability_score'] = 1.0 - current_data['stability_score']
            
            # 长宽比分析
            indicators['aspect_ratio_score'] = self._analyze_aspect_ratio(current_data['bbox'])
            
            # 高度变化分析
            indicators['height_change_score'] = self._analyze_height_change(current_data, previous_data)
            
            # 地面接触分析
            indicators['ground_contact_score'] = self._analyze_ground_contact(current_data['keypoints'], current_data['bbox'])
            
            # 时间一致性分析
            if len(history) >= 5:
                indicators['temporal_consistency_score'] = self._analyze_temporal_consistency(history[-5:])
            
        except Exception as e:
            logger.error(f"摔倒指标分析错误: {e}")
        
        return indicators
    
    def _analyze_body_angle(self, keypoints: List[Tuple[float, float, float]]) -> float:
        """分析身体倾斜角度"""
        try:
            if len(keypoints) < 17:  # YOLO格式检查
                return 0.0
            
            # 获取关键点（假设YOLO格式）
            left_shoulder = keypoints[5] if keypoints[5][2] > 0.3 else None
            right_shoulder = keypoints[6] if keypoints[6][2] > 0.3 else None
            left_hip = keypoints[11] if keypoints[11][2] > 0.3 else None
            right_hip = keypoints[12] if keypoints[12][2] > 0.3 else None
            
            if not (left_shoulder is not None and right_shoulder is not None and left_hip is not None and right_hip is not None):
                return 0.0
            
            # 计算身体中轴线
            shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2,
                             (left_shoulder[1] + right_shoulder[1]) / 2)
            hip_center = ((left_hip[0] + right_hip[0]) / 2,
                         (left_hip[1] + right_hip[1]) / 2)
            
            # 计算与垂直线的夹角
            dx = shoulder_center[0] - hip_center[0]
            dy = shoulder_center[1] - hip_center[1]
            
            if dy == 0:
                angle = 90
            else:
                angle = abs(math.degrees(math.atan(dx / dy)))
            
            # 转换为评分（角度越大，评分越高）
            score = min(1.0, angle / self.thresholds['body_angle_threshold'])
            return score
            
        except Exception as e:
            logger.error(f"身体角度分析错误: {e}")
            return 0.0
    
    def _analyze_velocity(self, current_data: Dict, previous_data: Dict) -> float:
        """分析运动速度"""
        try:
            # 计算边界框中心点移动
            curr_bbox = current_data['bbox']
            prev_bbox = previous_data['bbox']
            
            curr_center = ((curr_bbox[0] + curr_bbox[2]) / 2, (curr_bbox[1] + curr_bbox[3]) / 2)
            prev_center = ((prev_bbox[0] + prev_bbox[2]) / 2, (prev_bbox[1] + prev_bbox[3]) / 2)
            
            # 计算速度
            dx = curr_center[0] - prev_center[0]
            dy = curr_center[1] - prev_center[1]
            velocity = math.sqrt(dx**2 + dy**2)
            
            # 时间差
            dt = current_data['timestamp'] - previous_data['timestamp']
            if dt > 0:
                velocity = velocity / dt
            
            # 转换为评分
            score = min(1.0, velocity / self.thresholds['velocity_threshold'])
            return score
            
        except Exception as e:
            logger.error(f"速度分析错误: {e}")
            return 0.0
    
    def _analyze_acceleration(self, history_data: List[Dict]) -> float:
        """分析加速度"""
        try:
            if len(history_data) < 3:
                return 0.0
            
            # 计算连续两个时间段的速度
            velocities = []
            for i in range(1, len(history_data)):
                curr_bbox = history_data[i]['bbox']
                prev_bbox = history_data[i-1]['bbox']
                
                curr_center = ((curr_bbox[0] + curr_bbox[2]) / 2, (curr_bbox[1] + curr_bbox[3]) / 2)
                prev_center = ((prev_bbox[0] + prev_bbox[2]) / 2, (prev_bbox[1] + prev_bbox[3]) / 2)
                
                dx = curr_center[0] - prev_center[0]
                dy = curr_center[1] - prev_center[1]
                velocity = math.sqrt(dx**2 + dy**2)
                
                dt = history_data[i]['timestamp'] - history_data[i-1]['timestamp']
                if dt > 0:
                    velocity = velocity / dt
                
                velocities.append(velocity)
            
            # 计算加速度
            if len(velocities) >= 2:
                acceleration = abs(velocities[-1] - velocities[-2])
                score = min(1.0, acceleration / self.thresholds['acceleration_threshold'])
                return score
            
            return 0.0
            
        except Exception as e:
            logger.error(f"加速度分析错误: {e}")
            return 0.0
    
    def _analyze_pose_change(self, current_data: Dict, previous_data: Dict) -> float:
        """分析姿势变化"""
        try:
            curr_pose = current_data['pose_type']
            prev_pose = previous_data['pose_type']
            
            # 定义摔倒相关的姿势变化
            fall_transitions = {
                ('standing', 'lying'): 1.0,
                ('sitting', 'lying'): 0.8,
                ('standing', 'leaning'): 0.6,
                ('walking', 'lying'): 1.0,
                ('running', 'lying'): 1.0
            }
            
            transition = (prev_pose, curr_pose)
            score = fall_transitions.get(transition, 0.0)
            
            return score
            
        except Exception as e:
            logger.error(f"姿势变化分析错误: {e}")
            return 0.0
    
    def _analyze_aspect_ratio(self, bbox: Tuple[int, int, int, int]) -> float:
        """分析边界框长宽比"""
        try:
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            if height == 0:
                return 0.0
            
            aspect_ratio = width / height
            
            # 摔倒时长宽比通常会增大
            if aspect_ratio > self.thresholds['aspect_ratio_threshold']:
                score = min(1.0, (aspect_ratio - 1.0) / self.thresholds['aspect_ratio_threshold'])
            else:
                score = 0.0
            
            return score
            
        except Exception as e:
            logger.error(f"长宽比分析错误: {e}")
            return 0.0
    
    def _analyze_height_change(self, current_data: Dict, previous_data: Dict) -> float:
        """分析高度变化"""
        try:
            curr_bbox = current_data['bbox']
            prev_bbox = previous_data['bbox']
            
            curr_height = curr_bbox[3] - curr_bbox[1]
            prev_height = prev_bbox[3] - prev_bbox[1]
            
            if prev_height == 0:
                return 0.0
            
            height_change_ratio = abs(curr_height - prev_height) / prev_height
            
            # 摔倒时高度通常会显著变化
            score = min(1.0, height_change_ratio / self.thresholds['height_change_threshold'])
            return score
            
        except Exception as e:
            logger.error(f"高度变化分析错误: {e}")
            return 0.0
    
    def _analyze_ground_contact(self, keypoints: List[Tuple[float, float, float]], bbox: Tuple[int, int, int, int]) -> float:
        """分析地面接触情况"""
        try:
            if len(keypoints) < 17:  # YOLO格式检查
                return 0.0
            
            # 获取下半身关键点
            lower_body_points = []
            for i in [11, 12, 13, 14, 15, 16]:  # 髋部、膝盖、脚踝
                if i < len(keypoints) and keypoints[i][2] > 0.3:
                    lower_body_points.append(keypoints[i])
            
            if not lower_body_points:
                return 0.0
            
            # 计算下半身点的平均高度
            avg_lower_y = sum(point[1] for point in lower_body_points) / len(lower_body_points)
            
            # 边界框底部
            bbox_bottom = bbox[3]
            
            # 计算地面接触比例
            if bbox_bottom > 0:
                contact_ratio = avg_lower_y / bbox_bottom
                score = min(1.0, contact_ratio / self.thresholds['ground_contact_threshold'])
            else:
                score = 0.0
            
            return score
            
        except Exception as e:
            logger.error(f"地面接触分析错误: {e}")
            return 0.0
    
    def _analyze_temporal_consistency(self, history_data: List[Dict]) -> float:
        """分析时间一致性"""
        try:
            if len(history_data) < 3:
                return 0.0
            
            # 检查连续帧中的姿势一致性
            pose_changes = 0
            for i in range(1, len(history_data)):
                if history_data[i]['pose_type'] != history_data[i-1]['pose_type']:
                    pose_changes += 1
            
            # 计算一致性评分（变化越多，一致性越低，摔倒可能性越高）
            consistency_ratio = pose_changes / (len(history_data) - 1)
            score = min(1.0, consistency_ratio * 2)  # 放大系数
            
            return score
            
        except Exception as e:
            logger.error(f"时间一致性分析错误: {e}")
            return 0.0
    
    def _determine_fall_status(self, indicators: Dict[str, float]) -> Tuple[bool, float]:
        """确定摔倒状态"""
        try:
            # 加权计算综合评分
            weights = {
                'body_angle_score': 0.25,
                'velocity_score': 0.15,
                'acceleration_score': 0.15,
                'pose_change_score': 0.20,
                'stability_score': 0.10,
                'aspect_ratio_score': 0.05,
                'height_change_score': 0.05,
                'ground_contact_score': 0.03,
                'temporal_consistency_score': 0.02
            }
            
            weighted_score = sum(indicators[key] * weights[key] for key in weights if key in indicators)
            
            # 应用敏感度调整
            adjusted_score = weighted_score * self.sensitivity
            
            # 确定是否摔倒
            is_falling = adjusted_score > 0.5
            
            return is_falling, min(1.0, adjusted_score)
            
        except Exception as e:
            logger.error(f"摔倒状态确定错误: {e}")
            return False, 0.0
    
    def _determine_fall_stage(self, person_id: int, indicators: Dict[str, float]) -> str:
        """确定摔倒阶段"""
        try:
            if person_id not in self.person_histories:
                return 'normal'
            
            history = list(self.person_histories[person_id])
            if len(history) < 3:
                return 'normal'
            
            # 基于指标和历史数据判断阶段
            current_pose = history[-1]['pose_type']
            
            # 综合评分
            total_score = sum(indicators.values()) / len(indicators)
            
            if current_pose == 'lying' and total_score > 0.7:
                return 'fallen'
            elif total_score > 0.6:
                return 'falling'
            elif total_score > 0.3:
                return 'pre_fall'
            elif current_pose == 'lying' and total_score < 0.3:
                return 'recovery'
            else:
                return 'normal'
                
        except Exception as e:
            logger.error(f"摔倒阶段确定错误: {e}")
            return 'normal'
    
    def _analyze_fall_direction(self, person_id: int, keypoints: List[Tuple[float, float, float]]) -> str:
        """分析摔倒方向"""
        try:
            if person_id not in self.person_histories or len(self.person_histories[person_id]) < 2:
                return 'unknown'
            
            history = list(self.person_histories[person_id])
            current_bbox = history[-1]['bbox']
            previous_bbox = history[-2]['bbox']
            
            # 计算中心点移动
            curr_center = ((current_bbox[0] + current_bbox[2]) / 2, (current_bbox[1] + current_bbox[3]) / 2)
            prev_center = ((previous_bbox[0] + previous_bbox[2]) / 2, (previous_bbox[1] + previous_bbox[3]) / 2)
            
            dx = curr_center[0] - prev_center[0]
            dy = curr_center[1] - prev_center[1]
            
            # 基于移动方向判断摔倒方向
            if abs(dx) > abs(dy):
                return 'right' if dx > 0 else 'left'
            else:
                return 'forward' if dy > 0 else 'backward'
                
        except Exception as e:
            logger.error(f"摔倒方向分析错误: {e}")
            return 'unknown'
    
    def _assess_fall_severity(self, indicators: Dict[str, float]) -> str:
        """评估摔倒严重程度"""
        try:
            # 基于关键指标评估严重程度
            severity_score = (
                indicators.get('velocity_score', 0) * 0.3 +
                indicators.get('acceleration_score', 0) * 0.3 +
                indicators.get('body_angle_score', 0) * 0.2 +
                indicators.get('pose_change_score', 0) * 0.2
            )
            
            if severity_score > 0.8:
                return 'severe'
            elif severity_score > 0.5:
                return 'moderate'
            else:
                return 'mild'
                
        except Exception as e:
            logger.error(f"摔倒严重程度评估错误: {e}")
            return 'mild'
    
    def _calculate_body_orientation(self, keypoints: List[Tuple[float, float, float]]) -> float:
        """计算身体方向角度"""
        try:
            if len(keypoints) < 17:
                return 0.0
            
            # 使用肩膀和髋部计算身体轴线
            left_shoulder = keypoints[5] if keypoints[5][2] > 0.3 else None
            right_shoulder = keypoints[6] if keypoints[6][2] > 0.3 else None
            left_hip = keypoints[11] if keypoints[11][2] > 0.3 else None
            right_hip = keypoints[12] if keypoints[12][2] > 0.3 else None
            
            if not (left_shoulder is not None and right_shoulder is not None and left_hip is not None and right_hip is not None):
                return 0.0
            
            shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2,
                             (left_shoulder[1] + right_shoulder[1]) / 2)
            hip_center = ((left_hip[0] + right_hip[0]) / 2,
                         (left_hip[1] + right_hip[1]) / 2)
            
            dx = shoulder_center[0] - hip_center[0]
            dy = shoulder_center[1] - hip_center[1]
            
            angle = math.degrees(math.atan2(dy, dx))
            return angle
            
        except Exception as e:
            logger.error(f"身体方向计算错误: {e}")
            return 0.0
    
    def _calculate_velocity_vector(self, person_id: int) -> Tuple[float, float]:
        """计算速度向量"""
        try:
            if person_id not in self.person_histories or len(self.person_histories[person_id]) < 2:
                return (0.0, 0.0)
            
            history = list(self.person_histories[person_id])
            current_bbox = history[-1]['bbox']
            previous_bbox = history[-2]['bbox']
            
            curr_center = ((current_bbox[0] + current_bbox[2]) / 2, (current_bbox[1] + current_bbox[3]) / 2)
            prev_center = ((previous_bbox[0] + previous_bbox[2]) / 2, (previous_bbox[1] + previous_bbox[3]) / 2)
            
            dt = history[-1]['timestamp'] - history[-2]['timestamp']
            if dt > 0:
                vx = (curr_center[0] - prev_center[0]) / dt
                vy = (curr_center[1] - prev_center[1]) / dt
            else:
                vx, vy = 0.0, 0.0
            
            return (float(vx), float(vy))
            
        except Exception as e:
            logger.error(f"速度向量计算错误: {e}")
            return (0.0, 0.0)
    
    def _calculate_acceleration(self, person_id: int) -> float:
        """计算加速度"""
        try:
            if person_id not in self.person_histories or len(self.person_histories[person_id]) < 3:
                return 0.0
            
            history = list(self.person_histories[person_id])
            
            # 计算最近两个速度向量
            v1 = self._calculate_velocity_between_frames(history[-3], history[-2])
            v2 = self._calculate_velocity_between_frames(history[-2], history[-1])
            
            # 计算加速度
            dt = history[-1]['timestamp'] - history[-2]['timestamp']
            if dt > 0:
                ax = (v2[0] - v1[0]) / dt
                ay = (v2[1] - v1[1]) / dt
                acceleration = math.sqrt(ax**2 + ay**2)
            else:
                acceleration = 0.0
            
            return float(acceleration)
            
        except Exception as e:
            logger.error(f"加速度计算错误: {e}")
            return 0.0
    
    def _calculate_velocity_between_frames(self, frame1: Dict, frame2: Dict) -> Tuple[float, float]:
        """计算两帧之间的速度"""
        try:
            bbox1 = frame1['bbox']
            bbox2 = frame2['bbox']
            
            center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
            center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
            
            dt = frame2['timestamp'] - frame1['timestamp']
            if dt > 0:
                vx = (center2[0] - center1[0]) / dt
                vy = (center2[1] - center1[1]) / dt
            else:
                vx, vy = 0.0, 0.0
            
            return (vx, vy)
            
        except Exception:
            return (0.0, 0.0)
    
    def _calculate_stability_score(self, keypoints: List[Tuple[float, float, float]]) -> float:
        """计算稳定性评分"""
        try:
            if len(keypoints) < 17:
                return 0.5
            
            # 检查关键点的对称性
            symmetry_score = 0.0
            symmetry_pairs = [(5, 6), (11, 12), (13, 14), (15, 16)]  # 肩膀、髋部、膝盖、脚踝
            
            valid_pairs = 0
            for left_idx, right_idx in symmetry_pairs:
                if (left_idx < len(keypoints) and right_idx < len(keypoints) and
                    keypoints[left_idx][2] > 0.3 and keypoints[right_idx][2] > 0.3):
                    
                    left_point = keypoints[left_idx]
                    right_point = keypoints[right_idx]
                    
                    # 计算高度差异
                    height_diff = abs(left_point[1] - right_point[1])
                    symmetry = max(0, 1.0 - height_diff / 100.0)  # 标准化
                    symmetry_score += symmetry
                    valid_pairs += 1
            
            if valid_pairs > 0:
                return symmetry_score / valid_pairs
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"稳定性评分计算错误: {e}")
            return 0.5
    
    def _calculate_ground_contact_ratio(self, keypoints: List[Tuple[float, float, float]], bbox: Tuple[int, int, int, int]) -> float:
        """计算地面接触比例"""
        try:
            if len(keypoints) < 17:
                return 0.0
            
            # 获取脚部关键点
            foot_points = []
            for i in [15, 16]:  # 左右脚踝
                if i < len(keypoints) and keypoints[i][2] > 0.3:
                    foot_points.append(keypoints[i])
            
            if not foot_points:
                return 0.0
            
            # 计算脚部平均高度
            avg_foot_y = sum(point[1] for point in foot_points) / len(foot_points)
            
            # 边界框高度
            bbox_height = bbox[3] - bbox[1]
            bbox_bottom = bbox[3]
            
            if bbox_height > 0:
                # 计算脚部相对于边界框底部的位置
                relative_position = (bbox_bottom - avg_foot_y) / bbox_height
                contact_ratio = max(0, min(1, 1 - relative_position))
            else:
                contact_ratio = 0.0
            
            return contact_ratio
            
        except Exception as e:
            logger.error(f"地面接触比例计算错误: {e}")
            return 0.0
    
    def _determine_alert_level(self, indicators: Dict[str, float], is_falling: bool) -> str:
        """确定警报级别"""
        try:
            if not is_falling:
                return 'none'
            
            # 计算综合风险评分
            risk_score = sum(indicators.values()) / len(indicators)
            
            if risk_score > 0.8:
                return 'emergency'
            elif risk_score > 0.6:
                return 'critical'
            elif risk_score > 0.4:
                return 'warning'
            else:
                return 'none'
                
        except Exception as e:
            logger.error(f"警报级别确定错误: {e}")
            return 'none'
    
    def _calculate_fall_duration(self, person_id: int, is_falling: bool) -> float:
        """计算摔倒持续时间"""
        try:
            if not is_falling:
                # 清除摔倒事件记录
                if person_id in self.fall_events:
                    del self.fall_events[person_id]
                return 0.0
            
            current_time = time.time()
            
            if person_id not in self.fall_events:
                # 新的摔倒事件
                self.fall_events[person_id] = current_time
                return 0.0
            else:
                # 计算持续时间
                duration = current_time - self.fall_events[person_id]
                return duration
                
        except Exception as e:
            logger.error(f"摔倒持续时间计算错误: {e}")
            return 0.0
    
    def _assess_scene_risk(self, fall_detections: List[FallDetectionResult]) -> str:
        """评估场景风险级别"""
        try:
            if not fall_detections:
                return 'low'
            
            # 统计不同级别的警报
            alert_counts = {'emergency': 0, 'critical': 0, 'warning': 0, 'none': 0}
            
            for detection in fall_detections:
                alert_counts[detection.alert_level] += 1
            
            # 基于警报分布确定场景风险
            if alert_counts['emergency'] > 0:
                return 'critical'
            elif alert_counts['critical'] > 0:
                return 'high'
            elif alert_counts['warning'] > 0:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            logger.error(f"场景风险评估错误: {e}")
            return 'low'
    
    def _analyze_environmental_factors(self, persons_data: List[Any]) -> Dict[str, Any]:
        """分析环境因素"""
        factors = {
            'person_density': 0.0,
            'movement_activity': 'low',
            'space_utilization': 'normal',
            'interaction_level': 'minimal'
        }
        
        try:
            if len(persons_data) == 0:
                return factors
            
            # 人员密度
            factors['person_density'] = len(persons_data)
            
            # 运动活动水平
            active_persons = sum(1 for person in persons_data 
                               if hasattr(person, 'activity') and person.activity not in ['static', 'unknown'])
            
            if len(persons_data) > 0:
                activity_ratio = active_persons / len(persons_data)
                if activity_ratio > 0.7:
                    factors['movement_activity'] = 'high'
                elif activity_ratio > 0.3:
                    factors['movement_activity'] = 'medium'
                else:
                    factors['movement_activity'] = 'low'
            
            # 空间利用率（基于边界框分布）
            if len(persons_data) > 3:
                factors['space_utilization'] = 'crowded'
            elif len(persons_data) > 1:
                factors['space_utilization'] = 'normal'
            else:
                factors['space_utilization'] = 'sparse'
            
            # 交互水平
            if len(persons_data) > 5:
                factors['interaction_level'] = 'high'
            elif len(persons_data) > 2:
                factors['interaction_level'] = 'moderate'
            else:
                factors['interaction_level'] = 'minimal'
            
        except Exception as e:
            logger.error(f"环境因素分析错误: {e}")
        
        return factors
    
    def _update_performance_stats(self, processing_time: float, total_persons: int, active_falls: int):
        """更新性能统计"""
        try:
            self.performance_stats['total_detections'] += 1
            
            # 更新平均处理时间
            total_time = (self.performance_stats['avg_processing_time'] * 
                         (self.performance_stats['total_detections'] - 1) + processing_time)
            self.performance_stats['avg_processing_time'] = total_time / self.performance_stats['total_detections']
            
            # 更新摔倒检测数
            self.performance_stats['fall_detections'] += active_falls
            
            # 计算准确率（需要真实标签数据）
            # 这里简化处理，实际应用中需要标注数据
            if self.performance_stats['total_detections'] > 0:
                self.performance_stats['accuracy'] = (
                    (self.performance_stats['true_positives'] + 
                     (self.performance_stats['total_detections'] - self.performance_stats['fall_detections'])) /
                    self.performance_stats['total_detections']
                )
            
        except Exception as e:
            logger.error(f"性能统计更新错误: {e}")
    
    def _get_performance_metrics(self) -> Dict[str, float]:
        """获取性能指标"""
        return {
            'avg_processing_time': self.performance_stats['avg_processing_time'],
            'total_detections': self.performance_stats['total_detections'],
            'fall_detection_rate': (self.performance_stats['fall_detections'] / 
                                   max(1, self.performance_stats['total_detections'])),
            'accuracy': self.performance_stats['accuracy'],
            'false_positive_rate': (self.performance_stats['false_positives'] / 
                                   max(1, self.performance_stats['total_detections']))
        }
    
    def _get_default_result(self) -> MultiFallDetectionResult:
        """获取默认结果"""
        return MultiFallDetectionResult(
            total_persons=0,
            fall_detections=[],
            active_falls=0,
            critical_alerts=0,
            scene_risk_level='low',
            environmental_factors={'person_density': 0.0, 'movement_activity': 'low', 
                                 'space_utilization': 'normal', 'interaction_level': 'minimal'},
            performance_metrics=self._get_performance_metrics(),
            timestamp=time.time()
        )
    
    def _get_default_person_result(self, person_id: int) -> FallDetectionResult:
        """获取默认人员结果"""
        return FallDetectionResult(
            person_id=person_id,
            is_falling=False,
            fall_confidence=0.0,
            fall_stage='normal',
            fall_direction='unknown',
            fall_severity='mild',
            bbox=(0, 0, 0, 0),
            keypoints=[],
            body_orientation=0.0,
            velocity_vector=(0.0, 0.0),
            acceleration=0.0,
            stability_score=0.5,
            ground_contact_ratio=0.0,
            alert_level='none',
            timestamp=time.time(),
            duration=0.0
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取检测统计信息"""
        return {
            'performance_stats': self.performance_stats.copy(),
            'active_persons': len(self.person_histories),
            'active_fall_events': len(self.fall_events),
            'alert_timers': len(self.alert_timers),
            'thresholds': self.thresholds.copy(),
            'configuration': {
                'sensitivity': self.sensitivity,
                'min_fall_duration': self.min_fall_duration,
                'max_false_positive_rate': self.max_false_positive_rate,
                'sequence_length': self.sequence_length,
                'alert_delay': self.alert_delay
            }
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.performance_stats = {
            'total_detections': 0,
            'fall_detections': 0,
            'false_positives': 0,
            'true_positives': 0,
            'avg_processing_time': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
        self.person_histories.clear()
        self.fall_events.clear()
        self.alert_timers.clear()
        logger.info("摔倒检测统计信息已重置")
    
    def cleanup(self):
        """清理资源"""
        try:
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=True)
            
            if hasattr(self, 'pose_detector'):
                self.pose_detector.close()
            
            logger.info("优化版摔倒检测器资源清理完成")
        except Exception as e:
            logger.error(f"资源清理错误: {e}")


def create_optimized_fall_detector(**kwargs) -> OptimizedFallDetector:
    """创建优化版摔倒检测器的工厂函数"""
    return OptimizedFallDetector(**kwargs)


# 预设配置
FALL_DETECTOR_CONFIGS = {
    'high_sensitivity': {
        'sensitivity': 0.9,
        'min_fall_duration': 0.3,
        'max_false_positive_rate': 0.1,
        'enable_multi_stage': True,
        'enable_velocity_analysis': True,
        'enable_pose_analysis': True,
        'alert_delay': 0.5
    },
    'balanced': {
        'sensitivity': 0.7,
        'min_fall_duration': 0.5,
        'max_false_positive_rate': 0.05,
        'enable_multi_stage': True,
        'enable_velocity_analysis': True,
        'enable_pose_analysis': True,
        'alert_delay': 1.0
    },
    'low_false_positive': {
        'sensitivity': 0.5,
        'min_fall_duration': 1.0,
        'max_false_positive_rate': 0.01,
        'enable_multi_stage': True,
        'enable_velocity_analysis': True,
        'enable_pose_analysis': True,
        'alert_delay': 2.0
    }
}


def create_fall_detector_from_config(config_name: str = 'balanced') -> OptimizedFallDetector:
    """根据预设配置创建摔倒检测器"""
    if config_name not in FALL_DETECTOR_CONFIGS:
        logger.warning(f"未知配置: {config_name}，使用默认配置")
        config_name = 'balanced'
    
    config = FALL_DETECTOR_CONFIGS[config_name]
    return OptimizedFallDetector(**config)