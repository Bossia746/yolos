"""优化版身体姿势识别模块 - 集成多种姿势估计算法，支持多人检测"""

import os
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Tuple, Optional, Any, Union
import math
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import logging
from dataclasses import dataclass
import json

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
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


@dataclass
class PersonPoseResult:
    """单人姿势检测结果数据类"""
    person_id: int
    bbox: Tuple[int, int, int, int]
    keypoints: List[Tuple[float, float, float]]  # (x, y, confidence)
    pose_type: str
    pose_name: str
    activity: str
    angles: Dict[str, float]
    body_ratios: Dict[str, float]
    stability_score: float
    movement_vector: Tuple[float, float]
    confidence: float
    timestamp: float


@dataclass
class MultiPersonPoseResult:
    """多人姿势识别结果数据类"""
    persons_detected: int
    persons_data: List[PersonPoseResult]
    group_interactions: Dict[str, Any]
    scene_analysis: Dict[str, Any]
    performance_metrics: Dict[str, float]
    timestamp: float


class OptimizedPoseRecognizer:
    """优化版身体姿势识别器 - 高性能、多人检测、多算法集成"""
    
    def __init__(self,
                 max_num_persons: int = 10,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5,
                 model_complexity: int = 1,
                 enable_yolo: bool = True,
                 enable_mediapipe: bool = True,
                 enable_threading: bool = True,
                 enable_tracking: bool = True,
                 enable_activity_recognition: bool = True,
                 sequence_length: int = 30):
        """
        初始化优化版身体姿势识别器
        
        Args:
            max_num_persons: 最大检测人数
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
            model_complexity: MediaPipe模型复杂度
            enable_yolo: 是否启用YOLO检测
            enable_mediapipe: 是否启用MediaPipe检测
            enable_threading: 是否启用多线程
            enable_tracking: 是否启用人员跟踪
            enable_activity_recognition: 是否启用活动识别
            sequence_length: 序列长度（用于活动识别）
        """
        self.max_num_persons = max_num_persons
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        self.enable_yolo = enable_yolo and YOLO_AVAILABLE
        self.enable_mediapipe = enable_mediapipe and MEDIAPIPE_AVAILABLE
        self.enable_threading = enable_threading
        self.enable_tracking = enable_tracking
        self.enable_activity_recognition = enable_activity_recognition
        self.sequence_length = sequence_length
        
        # 性能统计
        self.performance_stats = {
            'total_detections': 0,
            'avg_processing_time': 0.0,
            'successful_poses': 0,
            'failed_detections': 0,
            'fps': 0.0,
            'persons_per_frame': 0.0
        }
        
        # 线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=3) if enable_threading else None
        
        # 初始化检测器
        self._init_detectors()
        
        # 姿势类别定义
        self._init_pose_categories()
        
        # 历史数据存储
        self.pose_history = deque(maxlen=sequence_length)
        self.activity_history = deque(maxlen=10)
        self.fps_history = deque(maxlen=30)
        
        # 人员跟踪
        self.person_trackers = {}
        self.next_person_id = 1
        self.tracking_threshold = 100  # 像素距离阈值
        
        # 质量评估阈值
        self.quality_thresholds = {
            'min_keypoints_visible': 8,
            'min_confidence': 0.3,
            'min_person_size': 50
        }
        
        logger.info(f"优化版姿势识别器初始化完成 - YOLO: {self.enable_yolo}, MediaPipe: {self.enable_mediapipe}")
    
    def _init_detectors(self):
        """初始化检测器"""
        # MediaPipe初始化
        if self.enable_mediapipe:
            try:
                self.mp_pose = mp.solutions.pose
                self.pose_detector = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=self.model_complexity,
                    enable_segmentation=False,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence
                )
                self.mp_drawing = mp.solutions.drawing_utils
                self.mp_drawing_styles = mp.solutions.drawing_styles
                logger.info("MediaPipe姿势检测初始化成功")
            except Exception as e:
                logger.error(f"MediaPipe初始化失败: {e}")
                self.enable_mediapipe = False
                self.pose_detector = None
        
        # YOLO初始化
        if self.enable_yolo:
            try:
                self.yolo_model = YOLO(os.path.join(os.getcwd(), 'module', 'yolov8n-pose.pt'))
                logger.info("YOLO姿势检测初始化成功")
            except Exception as e:
                logger.error(f"YOLO初始化失败: {e}")
                self.enable_yolo = False
                self.yolo_model = None
    
    def _init_pose_categories(self):
        """初始化姿势类别"""
        # 基础姿势
        self.basic_poses = {
            'standing': '站立',
            'sitting': '坐着',
            'lying': '躺着',
            'squatting': '蹲着',
            'kneeling': '跪着',
            'leaning': '倾斜',
            'bending': '弯腰',
            'unknown': '未知姿势'
        }
        
        # 动作活动
        self.activities = {
            'walking': '行走',
            'running': '跑步',
            'jumping': '跳跃',
            'dancing': '跳舞',
            'exercising': '运动',
            'waving': '挥手',
            'clapping': '鼓掌',
            'pointing': '指向',
            'stretching': '伸展',
            'falling': '摔倒',
            'climbing': '攀爬',
            'lifting': '举重',
            'pushing': '推',
            'pulling': '拉',
            'throwing': '投掷',
            'catching': '接球',
            'kicking': '踢',
            'punching': '出拳',
            'static': '静止',
            'unknown': '未知活动'
        }
        
        # 关键点索引（MediaPipe 33点模型）
        self.mp_keypoint_indices = {
            'nose': 0, 'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
            'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
            'left_ear': 7, 'right_ear': 8, 'mouth_left': 9, 'mouth_right': 10,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_pinky': 17, 'right_pinky': 18,
            'left_index': 19, 'right_index': 20,
            'left_thumb': 21, 'right_thumb': 22,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
            'left_heel': 29, 'right_heel': 30,
            'left_foot_index': 31, 'right_foot_index': 32
        }
        
        # YOLO关键点索引（COCO 17点模型）
        self.yolo_keypoint_indices = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
        }
    
    def detect_poses(self, image: np.ndarray) -> MultiPersonPoseResult:
        """
        检测多人姿势并进行分析
        
        Args:
            image: 输入图像
            
        Returns:
            MultiPersonPoseResult: 多人姿势识别结果
        """
        start_time = time.time()
        
        # 输入验证
        if image is None or image.size == 0:
            logger.warning("输入图像为空")
            return self._get_default_result()
        
        # 确保图像编码正确
        image = self._ensure_proper_encoding(image)
        
        try:
            # 多算法检测
            persons_data = []
            
            # 使用YOLO检测（优先，支持多人）
            if self.enable_yolo:
                yolo_results = self._detect_with_yolo(image)
                persons_data.extend(yolo_results)
            
            # 使用MediaPipe检测（补充单人检测）
            if self.enable_mediapipe and len(persons_data) == 0:
                mp_results = self._detect_with_mediapipe(image)
                persons_data.extend(mp_results)
            
            # 人员跟踪
            if self.enable_tracking:
                persons_data = self._track_persons(persons_data)
            
            # 活动识别
            if self.enable_activity_recognition:
                persons_data = self._enhance_with_activity_recognition(persons_data)
            
            # 群体交互分析
            group_interactions = self._analyze_group_interactions(persons_data)
            
            # 场景分析
            scene_analysis = self._analyze_scene(persons_data, image.shape)
            
            # 更新性能统计
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time, len(persons_data))
            
            return MultiPersonPoseResult(
                persons_detected=len(persons_data),
                persons_data=persons_data,
                group_interactions=group_interactions,
                scene_analysis=scene_analysis,
                performance_metrics=self._get_performance_metrics(),
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"姿势识别错误: {e}")
            return self._get_default_result()
    
    def _ensure_proper_encoding(self, image: np.ndarray) -> np.ndarray:
        """确保图像编码正确"""
        try:
            # 检查图像数据类型
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            
            # 检查图像范围
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            
            # 确保图像连续性
            if not image.flags['C_CONTIGUOUS']:
                image = np.ascontiguousarray(image)
            
            return image
        except Exception as e:
            logger.error(f"图像编码处理错误: {e}")
            return image
    
    def _detect_with_yolo(self, image: np.ndarray) -> List[PersonPoseResult]:
        """使用YOLO进行多人姿势检测"""
        persons_data = []
        
        if not self.yolo_model:
            return persons_data
        
        try:
            results = self.yolo_model(image, 
                                    conf=self.min_detection_confidence,
                                    iou=0.45,
                                    verbose=False)
            
            for result in results:
                if result.keypoints is not None:
                    keypoints_data = result.keypoints.data.cpu().numpy()
                    boxes_data = result.boxes.data.cpu().numpy() if result.boxes is not None else None
                    
                    for i, kpts in enumerate(keypoints_data):
                        # 提取关键点
                        keypoints = self._extract_yolo_keypoints(kpts)
                        
                        if len(keypoints) > 0:
                            # 质量评估
                            quality_score = self._assess_pose_quality(keypoints)
                            if quality_score < 0.3:
                                continue
                            
                            # 计算边界框
                            if boxes_data is not None and i < len(boxes_data):
                                # 确保边界框数据为标量
                                box_coords = boxes_data[i][:4]
                                bbox = tuple(int(coord.item()) if hasattr(coord, 'item') else int(coord) for coord in box_coords)
                            else:
                                bbox = self._calculate_pose_bbox(keypoints)
                            
                            # 姿势分类
                            pose_type = self._classify_pose_yolo(keypoints)
                            
                            # 计算关节角度
                            angles = self._calculate_joint_angles_yolo(keypoints)
                            
                            # 计算身体比例
                            body_ratios = self._calculate_body_ratios_yolo(keypoints)
                            
                            # 稳定性评分
                            stability_score = self._calculate_stability_score(keypoints)
                            
                            # 创建人员姿势结果
                            person_data = PersonPoseResult(
                                person_id=i,  # 临时ID，后续跟踪时会更新
                                bbox=bbox,
                                keypoints=keypoints,
                                pose_type=pose_type,
                                pose_name=self.basic_poses.get(pose_type, '未知'),
                                activity='static',  # 初始活动状态
                                angles=angles,
                                body_ratios=body_ratios,
                                stability_score=stability_score,
                                movement_vector=(0.0, 0.0),  # 初始运动向量
                                confidence=quality_score,
                                timestamp=time.time()
                            )
                            
                            persons_data.append(person_data)
                            
        except Exception as e:
            logger.error(f"YOLO姿势检测错误: {e}")
        
        return persons_data
    
    def _detect_with_mediapipe(self, image: np.ndarray) -> List[PersonPoseResult]:
        """使用MediaPipe进行单人姿势检测"""
        persons_data = []
        
        if not self.pose_detector:
            return persons_data
        
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose_detector.process(rgb_image)
            
            if results.pose_landmarks:
                # 提取关键点
                keypoints = self._extract_mediapipe_keypoints(results.pose_landmarks, image.shape)
                
                if len(keypoints) > 0:
                    # 质量评估
                    quality_score = self._assess_pose_quality(keypoints)
                    if quality_score >= 0.3:
                        # 计算边界框
                        bbox = self._calculate_pose_bbox(keypoints)
                        
                        # 姿势分类
                        pose_type = self._classify_pose_mediapipe(keypoints)
                        
                        # 计算关节角度
                        angles = self._calculate_joint_angles_mediapipe(keypoints)
                        
                        # 计算身体比例
                        body_ratios = self._calculate_body_ratios_mediapipe(keypoints)
                        
                        # 稳定性评分
                        stability_score = self._calculate_stability_score(keypoints)
                        
                        # 创建人员姿势结果
                        person_data = PersonPoseResult(
                            person_id=0,  # MediaPipe单人检测
                            bbox=bbox,
                            keypoints=keypoints,
                            pose_type=pose_type,
                            pose_name=self.basic_poses.get(pose_type, '未知'),
                            activity='static',
                            angles=angles,
                            body_ratios=body_ratios,
                            stability_score=stability_score,
                            movement_vector=(0.0, 0.0),
                            confidence=quality_score,
                            timestamp=time.time()
                        )
                        
                        persons_data.append(person_data)
                        
        except Exception as e:
            logger.error(f"MediaPipe姿势检测错误: {e}")
        
        return persons_data
    
    def _extract_yolo_keypoints(self, keypoints_data: np.ndarray) -> List[Tuple[float, float, float]]:
        """提取YOLO关键点数据"""
        keypoints = []
        
        try:
            # 检查输入数据
            if keypoints_data is None or keypoints_data.size == 0:
                return keypoints
            
            # 确保keypoints_data是二维数组 (num_keypoints, 3)
            if keypoints_data.ndim == 1:
                # 如果是一维数组，重塑为 (num_keypoints, 3)
                if len(keypoints_data) % 3 == 0:
                    keypoints_data = keypoints_data.reshape(-1, 3)
                else:
                    logger.warning(f"YOLO关键点数据长度不是3的倍数: {len(keypoints_data)}")
                    return keypoints
            
            # 处理每个关键点 [x, y, confidence]
            for kpt in keypoints_data:
                if len(kpt) >= 3:
                    x, y, conf = kpt[0], kpt[1], kpt[2]
                    # 安全转换为标量
                    try:
                        x_val = float(x.item()) if hasattr(x, 'item') else float(x)
                        y_val = float(y.item()) if hasattr(y, 'item') else float(y)
                        conf_val = float(conf.item()) if hasattr(conf, 'item') else float(conf)
                        
                        if conf_val > 0.1:  # 只保留置信度较高的关键点
                            keypoints.append((x_val, y_val, conf_val))
                        else:
                            keypoints.append((0.0, 0.0, 0.0))  # 不可见关键点
                    except (ValueError, TypeError) as ve:
                        logger.warning(f"关键点数值转换错误: {ve}, 使用默认值")
                        keypoints.append((0.0, 0.0, 0.0))
                        
        except Exception as e:
            logger.error(f"YOLO关键点提取错误: {e}")
            logger.error(f"关键点数据形状: {keypoints_data.shape if hasattr(keypoints_data, 'shape') else 'unknown'}")
        
        return keypoints
    
    def _extract_mediapipe_keypoints(self, landmarks, image_shape: Tuple[int, int, int]) -> List[Tuple[float, float, float]]:
        """提取MediaPipe关键点数据"""
        keypoints = []
        h, w, _ = image_shape
        
        try:
            for landmark in landmarks.landmark:
                x = landmark.x * w
                y = landmark.y * h
                conf = landmark.visibility
                keypoints.append((float(x.item()) if hasattr(x, 'item') else float(x), 
                                float(y.item()) if hasattr(y, 'item') else float(y), 
                                float(conf.item()) if hasattr(conf, 'item') else float(conf)))
        except Exception as e:
            logger.error(f"MediaPipe关键点提取错误: {e}")
        
        return keypoints
    
    def _assess_pose_quality(self, keypoints: List[Tuple[float, float, float]]) -> float:
        """评估姿势检测质量"""
        try:
            if len(keypoints) == 0:
                return 0.0
            
            # 计算可见关键点数量
            visible_keypoints = sum(1 for kp in keypoints if kp[2] > 0.3)
            
            # 计算平均置信度
            valid_confidences = [kp[2] for kp in keypoints if kp[2] > 0.1]
            avg_confidence = sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0.0
            
            # 计算姿势完整性
            completeness = visible_keypoints / len(keypoints)
            
            # 综合质量分数
            quality_score = (completeness * 0.6 + avg_confidence * 0.4)
            return min(1.0, quality_score)
            
        except Exception as e:
            logger.error(f"姿势质量评估错误: {e}")
            return 0.5
    
    def _calculate_pose_bbox(self, keypoints: List[Tuple[float, float, float]]) -> Tuple[int, int, int, int]:
        """计算姿势边界框"""
        if len(keypoints) == 0:
            return (0, 0, 0, 0)
        
        try:
            valid_points = [(kp[0], kp[1]) for kp in keypoints if kp[2] > 0.1]
            if not valid_points:
                return (0, 0, 0, 0)
            
            xs = [p[0] for p in valid_points]
            ys = [p[1] for p in valid_points]
            
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
            
            # 添加边距
            margin = 20
            # 确保标量转换，避免数组转换错误
            x1 = max(0, int(x1.item() if hasattr(x1, 'item') else x1) - margin)
            y1 = max(0, int(y1.item() if hasattr(y1, 'item') else y1) - margin)
            x2 = int((x2.item() if hasattr(x2, 'item') else x2) + margin)
            y2 = int((y2.item() if hasattr(y2, 'item') else y2) + margin)
            
            return (x1, y1, x2, y2)
            
        except Exception as e:
            logger.error(f"边界框计算错误: {e}")
            return (0, 0, 0, 0)
    
    def _classify_pose_yolo(self, keypoints: List[Tuple[float, float, float]]) -> str:
        """基于YOLO关键点分类姿势"""
        if len(keypoints) < 17:
            return 'unknown'
        
        try:
            # 获取关键关节点（COCO 17点格式）
            nose = keypoints[0] if keypoints[0][2] > 0.3 else None
            left_shoulder = keypoints[5] if keypoints[5][2] > 0.3 else None
            right_shoulder = keypoints[6] if keypoints[6][2] > 0.3 else None
            left_hip = keypoints[11] if keypoints[11][2] > 0.3 else None
            right_hip = keypoints[12] if keypoints[12][2] > 0.3 else None
            left_knee = keypoints[13] if keypoints[13][2] > 0.3 else None
            right_knee = keypoints[14] if keypoints[14][2] > 0.3 else None
            left_ankle = keypoints[15] if keypoints[15][2] > 0.3 else None
            right_ankle = keypoints[16] if keypoints[16][2] > 0.3 else None
            
            # 基础姿势判断
            if self._is_lying_yolo(keypoints):
                return 'lying'
            elif self._is_sitting_yolo(keypoints):
                return 'sitting'
            elif self._is_squatting_yolo(keypoints):
                return 'squatting'
            elif self._is_standing_yolo(keypoints):
                return 'standing'
            elif self._is_leaning_yolo(keypoints):
                return 'leaning'
            else:
                return 'unknown'
                
        except Exception as e:
            logger.error(f"YOLO姿势分类错误: {e}")
            return 'unknown'
    
    def _classify_pose_mediapipe(self, keypoints: List[Tuple[float, float, float]]) -> str:
        """基于MediaPipe关键点分类姿势"""
        if len(keypoints) < 33:
            return 'unknown'
        
        try:
            # 基础姿势判断
            if self._is_lying_mediapipe(keypoints):
                return 'lying'
            elif self._is_sitting_mediapipe(keypoints):
                return 'sitting'
            elif self._is_squatting_mediapipe(keypoints):
                return 'squatting'
            elif self._is_standing_mediapipe(keypoints):
                return 'standing'
            elif self._is_leaning_mediapipe(keypoints):
                return 'leaning'
            else:
                return 'unknown'
                
        except Exception as e:
            logger.error(f"MediaPipe姿势分类错误: {e}")
            return 'unknown'
    
    def _is_standing_yolo(self, keypoints: List[Tuple[float, float, float]]) -> bool:
        """判断是否为站立姿势（YOLO）"""
        try:
            # 检查数组长度
            if len(keypoints) < 17:
                return False
                
            # 检查关键点可见性
            left_hip = keypoints[11] if len(keypoints) > 11 and keypoints[11][2] > 0.3 else None
            right_hip = keypoints[12] if len(keypoints) > 12 and keypoints[12][2] > 0.3 else None
            left_knee = keypoints[13] if len(keypoints) > 13 and keypoints[13][2] > 0.3 else None
            right_knee = keypoints[14] if len(keypoints) > 14 and keypoints[14][2] > 0.3 else None
            left_ankle = keypoints[15] if len(keypoints) > 15 and keypoints[15][2] > 0.3 else None
            right_ankle = keypoints[16] if len(keypoints) > 16 and keypoints[16][2] > 0.3 else None
            
            if not (left_hip is not None and right_hip is not None and left_knee is not None and right_knee is not None):
                return False
            
            # 计算腿部角度
            left_leg_straight = self._is_leg_straight(left_hip, left_knee, left_ankle) if left_ankle else False
            right_leg_straight = self._is_leg_straight(right_hip, right_knee, right_ankle) if right_ankle else False
            
            # 检查身体垂直度
            hip_center_y = (left_hip[1] + right_hip[1]) / 2
            knee_center_y = (left_knee[1] + right_knee[1]) / 2
            
            # 站立：膝盖在臀部下方，腿部相对伸直
            return (knee_center_y > hip_center_y and 
                   (left_leg_straight or right_leg_straight))
                   
        except Exception:
            return False
    
    def _is_sitting_yolo(self, keypoints: List[Tuple[float, float, float]]) -> bool:
        """判断是否为坐着姿势（YOLO）"""
        try:
            # 检查数组长度
            if len(keypoints) < 15:
                return False
                
            left_hip = keypoints[11] if len(keypoints) > 11 and keypoints[11][2] > 0.3 else None
            right_hip = keypoints[12] if len(keypoints) > 12 and keypoints[12][2] > 0.3 else None
            left_knee = keypoints[13] if len(keypoints) > 13 and keypoints[13][2] > 0.3 else None
            right_knee = keypoints[14] if len(keypoints) > 14 and keypoints[14][2] > 0.3 else None
            
            if not (left_hip is not None and right_hip is not None and left_knee is not None and right_knee is not None):
                return False
            
            # 坐着：膝盖与臀部高度相近或更高
            hip_center_y = (left_hip[1] + right_hip[1]) / 2
            knee_center_y = (left_knee[1] + right_knee[1]) / 2
            
            return abs(knee_center_y - hip_center_y) < 50 or knee_center_y < hip_center_y
            
        except Exception:
            return False
    
    def _is_lying_yolo(self, keypoints: List[Tuple[float, float, float]]) -> bool:
        """判断是否为躺着姿势（YOLO）"""
        try:
            # 检查数组长度
            if len(keypoints) < 13:
                return False
                
            nose = keypoints[0] if len(keypoints) > 0 and keypoints[0][2] > 0.3 else None
            left_shoulder = keypoints[5] if len(keypoints) > 5 and keypoints[5][2] > 0.3 else None
            right_shoulder = keypoints[6] if len(keypoints) > 6 and keypoints[6][2] > 0.3 else None
            left_hip = keypoints[11] if len(keypoints) > 11 and keypoints[11][2] > 0.3 else None
            right_hip = keypoints[12] if len(keypoints) > 12 and keypoints[12][2] > 0.3 else None
            
            if not (nose is not None and left_shoulder is not None and right_shoulder is not None and left_hip is not None and right_hip is not None):
                return False
            
            # 躺着：头部、肩膀、臀部在相似的水平线上
            shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
            hip_center_y = (left_hip[1] + right_hip[1]) / 2
            
            # 身体水平度检查
            body_height_diff = abs(shoulder_center_y - hip_center_y)
            body_width = abs(left_shoulder[0] - right_shoulder[0])
            
            return body_height_diff < body_width * 0.3
            
        except Exception:
            return False
    
    def _is_squatting_yolo(self, keypoints: List[Tuple[float, float, float]]) -> bool:
        """判断是否为蹲着姿势（YOLO）"""
        try:
            # 检查数组长度
            if len(keypoints) < 15:
                return False
                
            left_hip = keypoints[11] if len(keypoints) > 11 and keypoints[11][2] > 0.3 else None
            right_hip = keypoints[12] if len(keypoints) > 12 and keypoints[12][2] > 0.3 else None
            left_knee = keypoints[13] if len(keypoints) > 13 and keypoints[13][2] > 0.3 else None
            right_knee = keypoints[14] if len(keypoints) > 14 and keypoints[14][2] > 0.3 else None
            left_ankle = keypoints[15] if len(keypoints) > 15 and keypoints[15][2] > 0.3 else None
            right_ankle = keypoints[16] if len(keypoints) > 16 and keypoints[16][2] > 0.3 else None
            
            if not (left_hip is not None and right_hip is not None and left_knee is not None and right_knee is not None):
                return False
            
            # 蹲着：膝盖明显高于臀部，腿部弯曲
            hip_center_y = (left_hip[1] + right_hip[1]) / 2
            knee_center_y = (left_knee[1] + right_knee[1]) / 2
            
            # 检查腿部弯曲角度
            left_leg_bent = not self._is_leg_straight(left_hip, left_knee, left_ankle) if left_ankle else True
            right_leg_bent = not self._is_leg_straight(right_hip, right_knee, right_ankle) if right_ankle else True
            
            return (knee_center_y < hip_center_y - 30 and 
                   (left_leg_bent or right_leg_bent))
                   
        except Exception:
            return False
    
    def _is_leaning_yolo(self, keypoints: List[Tuple[float, float, float]]) -> bool:
        """判断是否为倾斜姿势（YOLO）"""
        try:
            # 检查数组长度
            if len(keypoints) < 13:
                return False
                
            left_shoulder = keypoints[5] if len(keypoints) > 5 and keypoints[5][2] > 0.3 else None
            right_shoulder = keypoints[6] if len(keypoints) > 6 and keypoints[6][2] > 0.3 else None
            left_hip = keypoints[11] if len(keypoints) > 11 and keypoints[11][2] > 0.3 else None
            right_hip = keypoints[12] if len(keypoints) > 12 and keypoints[12][2] > 0.3 else None
            
            if not (left_shoulder is not None and right_shoulder is not None and left_hip is not None and right_hip is not None):
                return False
            
            # 计算身体倾斜角度
            shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2, 
                             (left_shoulder[1] + right_shoulder[1]) / 2)
            hip_center = ((left_hip[0] + right_hip[0]) / 2, 
                         (left_hip[1] + right_hip[1]) / 2)
            
            # 身体轴线偏离垂直的程度
            body_angle = math.atan2(abs(shoulder_center[0] - hip_center[0]), 
                                  abs(shoulder_center[1] - hip_center[1]))
            
            return body_angle > math.pi / 6  # 超过30度倾斜
            
        except Exception:
            return False
    
    def _is_leg_straight(self, hip: Tuple[float, float, float], 
                        knee: Tuple[float, float, float], 
                        ankle: Optional[Tuple[float, float, float]]) -> bool:
        """判断腿部是否伸直"""
        if not ankle:
            return False
        
        try:
            # 计算髋-膝-踝角度
            angle = self._calculate_angle(hip[:2], knee[:2], ankle[:2])
            return angle > 160  # 角度大于160度认为是伸直
        except Exception:
            return False
    
    def _calculate_angle(self, point1: Tuple[float, float], 
                        point2: Tuple[float, float], 
                        point3: Tuple[float, float]) -> float:
        """计算三点之间的角度"""
        try:
            # 向量计算
            v1 = (point1[0] - point2[0], point1[1] - point2[1])
            v2 = (point3[0] - point2[0], point3[1] - point2[1])
            
            # 点积和模长
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            magnitude1 = math.sqrt(v1[0]**2 + v1[1]**2)
            magnitude2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0
            
            # 计算角度
            cos_angle = dot_product / (magnitude1 * magnitude2)
            cos_angle = max(-1, min(1, cos_angle))  # 限制范围
            angle = math.acos(cos_angle)
            
            return math.degrees(angle)
        except Exception:
            return 0
    
    def _is_standing_mediapipe(self, keypoints: List[Tuple[float, float, float]]) -> bool:
        """判断是否为站立姿势（MediaPipe）"""
        # 类似YOLO的实现，但使用MediaPipe的33点索引
        try:
            # 检查数组长度
            if len(keypoints) < 29:
                return False
                
            left_hip = keypoints[23] if len(keypoints) > 23 and keypoints[23][2] > 0.3 else None
            right_hip = keypoints[24] if len(keypoints) > 24 and keypoints[24][2] > 0.3 else None
            left_knee = keypoints[25] if len(keypoints) > 25 and keypoints[25][2] > 0.3 else None
            right_knee = keypoints[26] if len(keypoints) > 26 and keypoints[26][2] > 0.3 else None
            left_ankle = keypoints[27] if len(keypoints) > 27 and keypoints[27][2] > 0.3 else None
            right_ankle = keypoints[28] if len(keypoints) > 28 and keypoints[28][2] > 0.3 else None
            
            if not (left_hip is not None and right_hip is not None and left_knee is not None and right_knee is not None):
                return False
            
            # 站立判断逻辑
            hip_center_y = (left_hip[1] + right_hip[1]) / 2
            knee_center_y = (left_knee[1] + right_knee[1]) / 2
            
            left_leg_straight = self._is_leg_straight(left_hip, left_knee, left_ankle) if left_ankle else False
            right_leg_straight = self._is_leg_straight(right_hip, right_knee, right_ankle) if right_ankle else False
            
            return (knee_center_y > hip_center_y and 
                   (left_leg_straight or right_leg_straight))
                   
        except Exception:
            return False
    
    def _is_sitting_mediapipe(self, keypoints: List[Tuple[float, float, float]]) -> bool:
        """判断是否为坐着姿势（MediaPipe）"""
        try:
            # 检查数组长度
            if len(keypoints) < 27:
                return False
                
            left_hip = keypoints[23] if len(keypoints) > 23 and keypoints[23][2] > 0.3 else None
            right_hip = keypoints[24] if len(keypoints) > 24 and keypoints[24][2] > 0.3 else None
            left_knee = keypoints[25] if len(keypoints) > 25 and keypoints[25][2] > 0.3 else None
            right_knee = keypoints[26] if len(keypoints) > 26 and keypoints[26][2] > 0.3 else None
            
            if not (left_hip is not None and right_hip is not None and left_knee is not None and right_knee is not None):
                return False
            
            hip_center_y = (left_hip[1] + right_hip[1]) / 2
            knee_center_y = (left_knee[1] + right_knee[1]) / 2
            
            return abs(knee_center_y - hip_center_y) < 50 or knee_center_y < hip_center_y
            
        except Exception:
            return False
    
    def _is_lying_mediapipe(self, keypoints: List[Tuple[float, float, float]]) -> bool:
        """判断是否为躺着姿势（MediaPipe）"""
        try:
            nose = keypoints[0] if keypoints[0][2] > 0.3 else None
            left_shoulder = keypoints[11] if keypoints[11][2] > 0.3 else None
            right_shoulder = keypoints[12] if keypoints[12][2] > 0.3 else None
            left_hip = keypoints[23] if keypoints[23][2] > 0.3 else None
            right_hip = keypoints[24] if keypoints[24][2] > 0.3 else None
            
            if not (nose is not None and left_shoulder is not None and right_shoulder is not None and left_hip is not None and right_hip is not None):
                return False
            
            shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
            hip_center_y = (left_hip[1] + right_hip[1]) / 2
            
            body_height_diff = abs(shoulder_center_y - hip_center_y)
            body_width = abs(left_shoulder[0] - right_shoulder[0])
            
            return body_height_diff < body_width * 0.3
            
        except Exception:
            return False
    
    def _is_squatting_mediapipe(self, keypoints: List[Tuple[float, float, float]]) -> bool:
        """判断是否为蹲着姿势（MediaPipe）"""
        try:
            left_hip = keypoints[23] if keypoints[23][2] > 0.3 else None
            right_hip = keypoints[24] if keypoints[24][2] > 0.3 else None
            left_knee = keypoints[25] if keypoints[25][2] > 0.3 else None
            right_knee = keypoints[26] if keypoints[26][2] > 0.3 else None
            left_ankle = keypoints[27] if keypoints[27][2] > 0.3 else None
            right_ankle = keypoints[28] if keypoints[28][2] > 0.3 else None
            
            if not (left_hip is not None and right_hip is not None and left_knee is not None and right_knee is not None):
                return False
            
            hip_center_y = (left_hip[1] + right_hip[1]) / 2
            knee_center_y = (left_knee[1] + right_knee[1]) / 2
            
            left_leg_bent = not self._is_leg_straight(left_hip, left_knee, left_ankle) if left_ankle else True
            right_leg_bent = not self._is_leg_straight(right_hip, right_knee, right_ankle) if right_ankle else True
            
            return (knee_center_y < hip_center_y - 30 and 
                   (left_leg_bent or right_leg_bent))
                   
        except Exception:
            return False
    
    def _is_leaning_mediapipe(self, keypoints: List[Tuple[float, float, float]]) -> bool:
        """判断是否为倾斜姿势（MediaPipe）"""
        try:
            left_shoulder = keypoints[11] if keypoints[11][2] > 0.3 else None
            right_shoulder = keypoints[12] if keypoints[12][2] > 0.3 else None
            left_hip = keypoints[23] if keypoints[23][2] > 0.3 else None
            right_hip = keypoints[24] if keypoints[24][2] > 0.3 else None
            
            if not (left_shoulder is not None and right_shoulder is not None and left_hip is not None and right_hip is not None):
                return False
            
            shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2, 
                             (left_shoulder[1] + right_shoulder[1]) / 2)
            hip_center = ((left_hip[0] + right_hip[0]) / 2, 
                         (left_hip[1] + right_hip[1]) / 2)
            
            body_angle = math.atan2(abs(shoulder_center[0] - hip_center[0]), 
                                  abs(shoulder_center[1] - hip_center[1]))
            
            return body_angle > math.pi / 6
            
        except Exception:
            return False
    
    def _calculate_joint_angles_yolo(self, keypoints: List[Tuple[float, float, float]]) -> Dict[str, float]:
        """计算关节角度（YOLO）"""
        angles = {}
        
        try:
            # 左臂角度（肩-肘-腕）
            if all(keypoints[i][2] > 0.3 for i in [5, 7, 9]):  # left_shoulder, left_elbow, left_wrist
                angles['left_arm'] = self._calculate_angle(
                    keypoints[5][:2], keypoints[7][:2], keypoints[9][:2]
                )
            
            # 右臂角度
            if all(keypoints[i][2] > 0.3 for i in [6, 8, 10]):  # right_shoulder, right_elbow, right_wrist
                angles['right_arm'] = self._calculate_angle(
                    keypoints[6][:2], keypoints[8][:2], keypoints[10][:2]
                )
            
            # 左腿角度（髋-膝-踝）
            if all(keypoints[i][2] > 0.3 for i in [11, 13, 15]):  # left_hip, left_knee, left_ankle
                angles['left_leg'] = self._calculate_angle(
                    keypoints[11][:2], keypoints[13][:2], keypoints[15][:2]
                )
            
            # 右腿角度
            if all(keypoints[i][2] > 0.3 for i in [12, 14, 16]):  # right_hip, right_knee, right_ankle
                angles['right_leg'] = self._calculate_angle(
                    keypoints[12][:2], keypoints[14][:2], keypoints[16][:2]
                )
            
        except Exception as e:
            logger.error(f"关节角度计算错误: {e}")
        
        return angles
    
    def _calculate_joint_angles_mediapipe(self, keypoints: List[Tuple[float, float, float]]) -> Dict[str, float]:
        """计算关节角度（MediaPipe）"""
        angles = {}
        
        try:
            # 左臂角度（肩-肘-腕）
            if all(keypoints[i][2] > 0.3 for i in [11, 13, 15]):  # left_shoulder, left_elbow, left_wrist
                angles['left_arm'] = self._calculate_angle(
                    keypoints[11][:2], keypoints[13][:2], keypoints[15][:2]
                )
            
            # 右臂角度
            if all(keypoints[i][2] > 0.3 for i in [12, 14, 16]):  # right_shoulder, right_elbow, right_wrist
                angles['right_arm'] = self._calculate_angle(
                    keypoints[12][:2], keypoints[14][:2], keypoints[16][:2]
                )
            
            # 左腿角度（髋-膝-踝）
            if all(keypoints[i][2] > 0.3 for i in [23, 25, 27]):  # left_hip, left_knee, left_ankle
                angles['left_leg'] = self._calculate_angle(
                    keypoints[23][:2], keypoints[25][:2], keypoints[27][:2]
                )
            
            # 右腿角度
            if all(keypoints[i][2] > 0.3 for i in [24, 26, 28]):  # right_hip, right_knee, right_ankle
                angles['right_leg'] = self._calculate_angle(
                    keypoints[24][:2], keypoints[26][:2], keypoints[28][:2]
                )
            
        except Exception as e:
            logger.error(f"关节角度计算错误: {e}")
        
        return angles
    
    def _calculate_body_ratios_yolo(self, keypoints: List[Tuple[float, float, float]]) -> Dict[str, float]:
        """计算身体比例（YOLO）"""
        ratios = {}
        
        try:
            # 检查keypoints数组长度
            if len(keypoints) < 17:
                return ratios
                
            # 肩宽
            if len(keypoints) > 6 and keypoints[5][2] > 0.3 and keypoints[6][2] > 0.3:  # left_shoulder, right_shoulder
                shoulder_width = abs(keypoints[5][0] - keypoints[6][0])
                ratios['shoulder_width'] = shoulder_width
            
            # 身高（头到脚）
            if len(keypoints) > 0 and keypoints[0][2] > 0.3:  # nose
                ankle_y = 0
                if len(keypoints) > 15 and keypoints[15][2] > 0.3:  # left_ankle
                    ankle_y = max(ankle_y, keypoints[15][1])
                if len(keypoints) > 16 and keypoints[16][2] > 0.3:  # right_ankle
                    ankle_y = max(ankle_y, keypoints[16][1])
                
                if ankle_y > 0:
                    body_height = ankle_y - keypoints[0][1]
                    ratios['body_height'] = body_height
            
            # 躯干长度（肩到髋）
            if (keypoints[5][2] > 0.3 and keypoints[6][2] > 0.3 and 
                keypoints[11][2] > 0.3 and keypoints[12][2] > 0.3):
                shoulder_center_y = (keypoints[5][1] + keypoints[6][1]) / 2
                hip_center_y = (keypoints[11][1] + keypoints[12][1]) / 2
                torso_length = abs(hip_center_y - shoulder_center_y)
                ratios['torso_length'] = torso_length
            
        except Exception as e:
            logger.error(f"身体比例计算错误: {e}")
        
        return ratios
    
    def _calculate_body_ratios_mediapipe(self, keypoints: List[Tuple[float, float, float]]) -> Dict[str, float]:
        """计算身体比例（MediaPipe）"""
        ratios = {}
        
        try:
            # 检查keypoints数组长度
            if len(keypoints) < 29:
                return ratios
                
            # 肩宽
            if len(keypoints) > 12 and keypoints[11][2] > 0.3 and keypoints[12][2] > 0.3:  # left_shoulder, right_shoulder
                shoulder_width = abs(keypoints[11][0] - keypoints[12][0])
                ratios['shoulder_width'] = shoulder_width
            
            # 身高（头到脚）
            if len(keypoints) > 0 and keypoints[0][2] > 0.3:  # nose
                ankle_y = 0
                if len(keypoints) > 27 and keypoints[27][2] > 0.3:  # left_ankle
                    ankle_y = max(ankle_y, keypoints[27][1])
                if len(keypoints) > 28 and keypoints[28][2] > 0.3:  # right_ankle
                    ankle_y = max(ankle_y, keypoints[28][1])
                
                if ankle_y > 0:
                    body_height = ankle_y - keypoints[0][1]
                    ratios['body_height'] = body_height
            
            # 躯干长度（肩到髋）
            if (len(keypoints) > 24 and keypoints[11][2] > 0.3 and keypoints[12][2] > 0.3 and 
                keypoints[23][2] > 0.3 and keypoints[24][2] > 0.3):
                shoulder_center_y = (keypoints[11][1] + keypoints[12][1]) / 2
                hip_center_y = (keypoints[23][1] + keypoints[24][1]) / 2
                torso_length = abs(hip_center_y - shoulder_center_y)
                ratios['torso_length'] = torso_length
            
        except Exception as e:
            logger.error(f"身体比例计算错误: {e}")
        
        return ratios
    
    def _calculate_stability_score(self, keypoints: List[Tuple[float, float, float]]) -> float:
        """计算姿势稳定性评分"""
        try:
            # 基于关键点的对称性和平衡性
            stability_factors = []
            
            # 检查左右对称性
            if len(keypoints) >= 17:  # YOLO格式
                # 肩膀对称性
                if keypoints[5][2] > 0.3 and keypoints[6][2] > 0.3:
                    shoulder_symmetry = 1.0 - abs(keypoints[5][1] - keypoints[6][1]) / 100.0
                    stability_factors.append(max(0, shoulder_symmetry))
                
                # 髋部对称性
                if keypoints[11][2] > 0.3 and keypoints[12][2] > 0.3:
                    hip_symmetry = 1.0 - abs(keypoints[11][1] - keypoints[12][1]) / 100.0
                    stability_factors.append(max(0, hip_symmetry))
            
            elif len(keypoints) >= 33:  # MediaPipe格式
                # 肩膀对称性
                if keypoints[11][2] > 0.3 and keypoints[12][2] > 0.3:
                    shoulder_symmetry = 1.0 - abs(keypoints[11][1] - keypoints[12][1]) / 100.0
                    stability_factors.append(max(0, shoulder_symmetry))
                
                # 髋部对称性
                if keypoints[23][2] > 0.3 and keypoints[24][2] > 0.3:
                    hip_symmetry = 1.0 - abs(keypoints[23][1] - keypoints[24][1]) / 100.0
                    stability_factors.append(max(0, hip_symmetry))
            
            # 计算平均稳定性
            if len(stability_factors) > 0:
                return sum(stability_factors) / len(stability_factors)
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"稳定性评分计算错误: {e}")
            return 0.5
    
    def _track_persons(self, persons_data: List[PersonPoseResult]) -> List[PersonPoseResult]:
        """人员跟踪"""
        if not self.enable_tracking:
            return persons_data
        
        try:
            # 为每个检测到的人员分配或更新ID
            for person in persons_data:
                best_match_id = None
                min_distance = float('inf')
                
                # 计算与现有跟踪器的距离
                person_center = ((person.bbox[0] + person.bbox[2]) / 2, 
                               (person.bbox[1] + person.bbox[3]) / 2)
                
                for tracker_id, tracker_data in self.person_trackers.items():
                    tracker_center = tracker_data['center']
                    distance = math.sqrt(
                        (person_center[0] - tracker_center[0])**2 + 
                        (person_center[1] - tracker_center[1])**2
                    )
                    
                    if distance < min_distance and distance < self.tracking_threshold:
                        min_distance = distance
                        best_match_id = tracker_id
                
                # 分配ID
                if best_match_id is not None:
                    person.person_id = best_match_id
                    # 更新跟踪器
                    self.person_trackers[best_match_id] = {
                        'center': person_center,
                        'last_seen': time.time(),
                        'pose_history': self.person_trackers[best_match_id].get('pose_history', [])
                    }
                else:
                    # 创建新跟踪器
                    person.person_id = self.next_person_id
                    self.person_trackers[self.next_person_id] = {
                        'center': person_center,
                        'last_seen': time.time(),
                        'pose_history': []
                    }
                    self.next_person_id += 1
            
            # 清理过期的跟踪器
            current_time = time.time()
            expired_trackers = []
            for tracker_id, tracker_data in self.person_trackers.items():
                if current_time - tracker_data['last_seen'] > 5.0:  # 5秒超时
                    expired_trackers.append(tracker_id)
            
            for tracker_id in expired_trackers:
                del self.person_trackers[tracker_id]
            
        except Exception as e:
            logger.error(f"人员跟踪错误: {e}")
        
        return persons_data
    
    def _enhance_with_activity_recognition(self, persons_data: List[PersonPoseResult]) -> List[PersonPoseResult]:
        """增强活动识别"""
        if not self.enable_activity_recognition:
            return persons_data
        
        try:
            # 更新历史数据
            current_poses = {person.person_id: person for person in persons_data}
            self.pose_history.append(current_poses)
            
            # 为每个人员分析活动
            for person in persons_data:
                activity = self._recognize_activity(person.person_id)
                person.activity = activity
                
                # 计算运动向量
                movement_vector = self._calculate_movement_vector(person.person_id)
                person.movement_vector = movement_vector
            
        except Exception as e:
            logger.error(f"活动识别错误: {e}")
        
        return persons_data
    
    def _recognize_activity(self, person_id: int) -> str:
        """识别人员活动"""
        try:
            if len(self.pose_history) < 3:
                return 'static'
            
            # 获取历史姿势数据
            recent_poses = []
            for frame_data in list(self.pose_history)[-5:]:
                if person_id in frame_data:
                    recent_poses.append(frame_data[person_id])
            
            if len(recent_poses) < 2:
                return 'static'
            
            # 分析运动模式
            movement_patterns = self._analyze_movement_patterns(recent_poses)
            
            # 基于运动模式识别活动
            if movement_patterns['is_falling']:
                return 'falling'
            elif movement_patterns['is_running']:
                return 'running'
            elif movement_patterns['is_walking']:
                return 'walking'
            elif movement_patterns['is_jumping']:
                return 'jumping'
            elif movement_patterns['is_waving']:
                return 'waving'
            elif movement_patterns['has_significant_movement']:
                return 'exercising'
            else:
                return 'static'
                
        except Exception as e:
            logger.error(f"活动识别错误: {e}")
            return 'unknown'
    
    def _analyze_movement_patterns(self, poses: List[PersonPoseResult]) -> Dict[str, bool]:
        """分析运动模式"""
        patterns = {
            'is_falling': False,
            'is_running': False,
            'is_walking': False,
            'is_jumping': False,
            'is_waving': False,
            'has_significant_movement': False
        }
        
        try:
            if len(poses) < 2:
                return patterns
            
            # 计算位置变化
            position_changes = []
            for i in range(1, len(poses)):
                prev_center = ((poses[i-1].bbox[0] + poses[i-1].bbox[2]) / 2,
                             (poses[i-1].bbox[1] + poses[i-1].bbox[3]) / 2)
                curr_center = ((poses[i].bbox[0] + poses[i].bbox[2]) / 2,
                             (poses[i].bbox[1] + poses[i].bbox[3]) / 2)
                
                dx = curr_center[0] - prev_center[0]
                dy = curr_center[1] - prev_center[1]
                distance = math.sqrt(dx**2 + dy**2)
                
                position_changes.append({
                    'dx': dx, 'dy': dy, 'distance': distance,
                    'prev_pose': poses[i-1].pose_type,
                    'curr_pose': poses[i].pose_type
                })
            
            # 分析模式
            avg_movement = sum(change['distance'] for change in position_changes) / len(position_changes)
            
            # 摔倒检测：快速向下运动 + 姿势变化为躺着
            for change in position_changes:
                if (change['dy'] > 30 and change['distance'] > 50 and 
                    change['curr_pose'] == 'lying'):
                    patterns['is_falling'] = True
                    break
            
            # 跑步检测：快速水平移动
            if avg_movement > 20:
                horizontal_movement = sum(abs(change['dx']) for change in position_changes) / len(position_changes)
                if horizontal_movement > 15:
                    patterns['is_running'] = True
            
            # 行走检测：中等水平移动
            elif avg_movement > 8:
                horizontal_movement = sum(abs(change['dx']) for change in position_changes) / len(position_changes)
                if horizontal_movement > 5:
                    patterns['is_walking'] = True
            
            # 跳跃检测：垂直运动模式
            vertical_changes = [change['dy'] for change in position_changes]
            if len(vertical_changes) >= 3:
                # 检查上下运动模式
                up_down_pattern = any(
                    vertical_changes[i] < -10 and vertical_changes[i+1] > 10
                    for i in range(len(vertical_changes)-1)
                )
                if up_down_pattern:
                    patterns['is_jumping'] = True
            
            # 挥手检测：基于手部关键点运动（需要更详细的关键点分析）
            # 这里简化为检测上半身的快速运动
            if avg_movement > 5 and not patterns['is_walking'] and not patterns['is_running']:
                patterns['is_waving'] = True
            
            # 显著运动检测
            if avg_movement > 3:
                patterns['has_significant_movement'] = True
            
        except Exception as e:
            logger.error(f"运动模式分析错误: {e}")
        
        return patterns
    
    def _calculate_movement_vector(self, person_id: int) -> Tuple[float, float]:
        """计算运动向量"""
        try:
            if len(self.pose_history) < 2:
                return (0.0, 0.0)
            
            # 获取最近两帧的位置
            recent_frames = list(self.pose_history)[-2:]
            
            if person_id not in recent_frames[0] or person_id not in recent_frames[1]:
                return (0.0, 0.0)
            
            prev_person = recent_frames[0][person_id]
            curr_person = recent_frames[1][person_id]
            
            # 计算中心点移动
            prev_center = ((prev_person.bbox[0] + prev_person.bbox[2]) / 2,
                         (prev_person.bbox[1] + prev_person.bbox[3]) / 2)
            curr_center = ((curr_person.bbox[0] + curr_person.bbox[2]) / 2,
                         (curr_person.bbox[1] + curr_person.bbox[3]) / 2)
            
            dx = curr_center[0] - prev_center[0]
            dy = curr_center[1] - prev_center[1]
            
            # 确保标量转换，避免数组转换错误
            return (float(dx.item() if hasattr(dx, 'item') else dx), 
                   float(dy.item() if hasattr(dy, 'item') else dy))
            
        except Exception as e:
            logger.error(f"运动向量计算错误: {e}")
            return (0.0, 0.0)
    
    def _analyze_group_interactions(self, persons_data: List[PersonPoseResult]) -> Dict[str, Any]:
        """分析群体交互"""
        interactions = {
            'total_persons': len(persons_data),
            'interactions_detected': [],
            'crowd_density': 0.0,
            'group_activities': []
        }
        
        try:
            if len(persons_data) < 2:
                return interactions
            
            # 计算人员间距离
            for i, person1 in enumerate(persons_data):
                for j, person2 in enumerate(persons_data[i+1:], i+1):
                    center1 = ((person1.bbox[0] + person1.bbox[2]) / 2,
                             (person1.bbox[1] + person1.bbox[3]) / 2)
                    center2 = ((person2.bbox[0] + person2.bbox[2]) / 2,
                             (person2.bbox[1] + person2.bbox[3]) / 2)
                    
                    distance = math.sqrt(
                        (center1[0] - center2[0])**2 + 
                        (center1[1] - center2[1])**2
                    )
                    
                    # 检测交互
                    if distance < 150:  # 近距离交互
                        interaction = {
                            'person1_id': person1.person_id,
                            'person2_id': person2.person_id,
                            'distance': distance,
                            'interaction_type': 'close_proximity'
                        }
                        
                        # 分析交互类型
                        if person1.activity == person2.activity and person1.activity != 'static':
                            interaction['interaction_type'] = 'synchronized_activity'
                        elif distance < 80:
                            interaction['interaction_type'] = 'intimate_interaction'
                        
                        interactions['interactions_detected'].append(interaction)
            
            # 计算人群密度
            if len(persons_data) > 1:
                total_area = 0
                for person in persons_data:
                    bbox_area = (person.bbox[2] - person.bbox[0]) * (person.bbox[3] - person.bbox[1])
                    total_area += bbox_area
                
                interactions['crowd_density'] = len(persons_data) / (total_area / 10000)  # 标准化
            
            # 检测群体活动
            activities = [person.activity for person in persons_data]
            activity_counts = {}
            for activity in activities:
                activity_counts[activity] = activity_counts.get(activity, 0) + 1
            
            for activity, count in activity_counts.items():
                if count >= 2 and activity != 'static':
                    interactions['group_activities'].append({
                        'activity': activity,
                        'participant_count': count
                    })
            
        except Exception as e:
            logger.error(f"群体交互分析错误: {e}")
        
        return interactions
    
    def _analyze_scene(self, persons_data: List[PersonPoseResult], image_shape: Tuple[int, int, int]) -> Dict[str, Any]:
        """场景分析"""
        scene_analysis = {
            'scene_type': 'unknown',
            'activity_level': 'low',
            'safety_alerts': [],
            'spatial_distribution': {},
            'dominant_poses': {},
            'temporal_patterns': {}
        }
        
        try:
            h, w, _ = image_shape
            
            # 分析空间分布
            if len(persons_data) > 0:
                # 计算人员分布
                positions = []
                for person in persons_data:
                    center_x = (person.bbox[0] + person.bbox[2]) / 2 / w
                    center_y = (person.bbox[1] + person.bbox[3]) / 2 / h
                    positions.append((center_x, center_y))
                
                # 空间分布统计
                left_count = sum(1 for pos in positions if pos[0] < 0.33)
                center_count = sum(1 for pos in positions if 0.33 <= pos[0] <= 0.67)
                right_count = sum(1 for pos in positions if pos[0] > 0.67)
                
                scene_analysis['spatial_distribution'] = {
                    'left_region': left_count,
                    'center_region': center_count,
                    'right_region': right_count
                }
            
            # 分析主要姿势
            pose_counts = {}
            activity_counts = {}
            
            for person in persons_data:
                pose_counts[person.pose_type] = pose_counts.get(person.pose_type, 0) + 1
                activity_counts[person.activity] = activity_counts.get(person.activity, 0) + 1
            
            scene_analysis['dominant_poses'] = pose_counts
            
            # 活动水平评估
            active_persons = sum(1 for person in persons_data 
                               if person.activity not in ['static', 'unknown'])
            
            if len(persons_data) > 0:
                activity_ratio = active_persons / len(persons_data)
                if activity_ratio > 0.7:
                    scene_analysis['activity_level'] = 'high'
                elif activity_ratio > 0.3:
                    scene_analysis['activity_level'] = 'medium'
                else:
                    scene_analysis['activity_level'] = 'low'
            
            # 安全警报
            for person in persons_data:
                if person.activity == 'falling':
                    scene_analysis['safety_alerts'].append({
                        'type': 'fall_detected',
                        'person_id': person.person_id,
                        'confidence': person.confidence,
                        'location': person.bbox
                    })
                
                if person.stability_score < 0.3:
                    scene_analysis['safety_alerts'].append({
                        'type': 'unstable_pose',
                        'person_id': person.person_id,
                        'stability_score': person.stability_score,
                        'location': person.bbox
                    })
            
            # 场景类型推断
            if len(persons_data) > 5:
                scene_analysis['scene_type'] = 'crowded'
            elif activity_counts.get('exercising', 0) > 0:
                scene_analysis['scene_type'] = 'fitness'
            elif activity_counts.get('walking', 0) > 0 or activity_counts.get('running', 0) > 0:
                scene_analysis['scene_type'] = 'transit'
            elif pose_counts.get('sitting', 0) > pose_counts.get('standing', 0):
                scene_analysis['scene_type'] = 'meeting'
            else:
                scene_analysis['scene_type'] = 'general'
            
        except Exception as e:
            logger.error(f"场景分析错误: {e}")
        
        return scene_analysis
    
    def _update_performance_stats(self, processing_time: float, persons_detected: int):
        """更新性能统计"""
        try:
            self.performance_stats['total_detections'] += 1
            
            # 更新平均处理时间
            total_time = (self.performance_stats['avg_processing_time'] * 
                         (self.performance_stats['total_detections'] - 1) + processing_time)
            self.performance_stats['avg_processing_time'] = total_time / self.performance_stats['total_detections']
            
            # 更新成功检测数
            if persons_detected > 0:
                self.performance_stats['successful_poses'] += 1
            else:
                self.performance_stats['failed_detections'] += 1
            
            # 计算FPS
            if processing_time > 0:
                current_fps = 1.0 / processing_time
                self.fps_history.append(current_fps)
                self.performance_stats['fps'] = sum(self.fps_history) / len(self.fps_history)
            
            # 更新平均检测人数
            total_persons = (self.performance_stats['persons_per_frame'] * 
                           (self.performance_stats['total_detections'] - 1) + persons_detected)
            self.performance_stats['persons_per_frame'] = total_persons / self.performance_stats['total_detections']
            
        except Exception as e:
            logger.error(f"性能统计更新错误: {e}")
    
    def _get_performance_metrics(self) -> Dict[str, float]:
        """获取性能指标"""
        return {
            'avg_processing_time': self.performance_stats['avg_processing_time'],
            'fps': self.performance_stats['fps'],
            'success_rate': (self.performance_stats['successful_poses'] / 
                           max(1, self.performance_stats['total_detections'])),
            'avg_persons_per_frame': self.performance_stats['persons_per_frame']
        }
    
    def _get_default_result(self) -> MultiPersonPoseResult:
        """获取默认结果"""
        return MultiPersonPoseResult(
            persons_detected=0,
            persons_data=[],
            group_interactions={'total_persons': 0, 'interactions_detected': [], 
                              'crowd_density': 0.0, 'group_activities': []},
            scene_analysis={'scene_type': 'unknown', 'activity_level': 'low', 
                          'safety_alerts': [], 'spatial_distribution': {}, 
                          'dominant_poses': {}, 'temporal_patterns': {}},
            performance_metrics=self._get_performance_metrics(),
            timestamp=time.time()
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取识别统计信息"""
        return {
            'performance_stats': self.performance_stats.copy(),
            'active_trackers': len(self.person_trackers),
            'pose_history_length': len(self.pose_history),
            'activity_history_length': len(self.activity_history),
            'enabled_features': {
                'yolo': self.enable_yolo,
                'mediapipe': self.enable_mediapipe,
                'threading': self.enable_threading,
                'tracking': self.enable_tracking,
                'activity_recognition': self.enable_activity_recognition
            }
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.performance_stats = {
            'total_detections': 0,
            'avg_processing_time': 0.0,
            'successful_poses': 0,
            'failed_detections': 0,
            'fps': 0.0,
            'persons_per_frame': 0.0
        }
        self.pose_history.clear()
        self.activity_history.clear()
        self.fps_history.clear()
        self.person_trackers.clear()
        self.next_person_id = 1
        logger.info("姿势识别统计信息已重置")
    
    def cleanup(self):
        """清理资源"""
        try:
            if self.thread_pool:
                self.thread_pool.shutdown(wait=True)
            
            if self.pose_detector:
                self.pose_detector.close()
            
            logger.info("优化版姿势识别器资源清理完成")
        except Exception as e:
            logger.error(f"资源清理错误: {e}")


def create_optimized_pose_recognizer(**kwargs) -> OptimizedPoseRecognizer:
    """创建优化版姿势识别器的工厂函数"""
    return OptimizedPoseRecognizer(**kwargs)


# 预设配置
POSE_RECOGNIZER_CONFIGS = {
    'high_performance': {
        'max_num_persons': 5,
        'min_detection_confidence': 0.8,
        'model_complexity': 2,
        'enable_threading': True,
        'enable_tracking': True,
        'enable_activity_recognition': True
    },
    'balanced': {
        'max_num_persons': 8,
        'min_detection_confidence': 0.7,
        'model_complexity': 1,
        'enable_threading': True,
        'enable_tracking': True,
        'enable_activity_recognition': True
    },
    'lightweight': {
        'max_num_persons': 3,
        'min_detection_confidence': 0.6,
        'model_complexity': 0,
        'enable_threading': False,
        'enable_tracking': False,
        'enable_activity_recognition': False
    }
}


def create_pose_recognizer_from_config(config_name: str = 'balanced') -> OptimizedPoseRecognizer:
    """根据预设配置创建姿势识别器"""
    if config_name not in POSE_RECOGNIZER_CONFIGS:
        logger.warning(f"未知配置: {config_name}，使用默认配置")
        config_name = 'balanced'
    
    config = POSE_RECOGNIZER_CONFIGS[config_name]
    return OptimizedPoseRecognizer(**config)