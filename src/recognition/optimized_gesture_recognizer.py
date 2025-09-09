"""优化版手势识别模块 - 集成多种手势识别模型，提升实时性和准确率"""

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
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available")

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


@dataclass
class HandDetectionResult:
    """手部检测结果数据类"""
    hand_id: int
    handedness: str  # 'Left' or 'Right'
    handedness_confidence: float
    landmarks: List[Tuple[int, int]]
    static_gesture: Dict[str, Any]
    dynamic_gesture: Dict[str, Any]
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    orientation: float
    quality_score: float
    timestamp: float


@dataclass
class GestureRecognitionResult:
    """手势识别结果数据类"""
    hands_detected: int
    hands_data: List[HandDetectionResult]
    interaction: Dict[str, Any]
    performance_metrics: Dict[str, float]
    timestamp: float


class OptimizedGestureRecognizer:
    """优化版手势识别器 - 高性能、多模型集成"""
    
    def __init__(self, 
                 max_num_hands: int = 4,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5,
                 sequence_length: int = 30,
                 enable_threading: bool = True,
                 enable_dynamic_gestures: bool = True,
                 enable_interaction_detection: bool = True):
        """
        初始化优化版手势识别器
        
        Args:
            max_num_hands: 最大检测手数
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
            sequence_length: 序列长度（用于动态手势）
            enable_threading: 是否启用多线程
            enable_dynamic_gestures: 是否启用动态手势识别
            enable_interaction_detection: 是否启用交互检测
        """
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.sequence_length = sequence_length
        self.enable_threading = enable_threading
        self.enable_dynamic_gestures = enable_dynamic_gestures
        self.enable_interaction_detection = enable_interaction_detection
        
        # 性能统计
        self.performance_stats = {
            'total_detections': 0,
            'avg_processing_time': 0.0,
            'successful_gestures': 0,
            'failed_detections': 0,
            'fps': 0.0
        }
        
        # 线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=2) if enable_threading else None
        
        # MediaPipe初始化
        self._init_mediapipe()
        
        # 手势类别定义
        self._init_gesture_categories()
        
        # 历史数据存储
        self.landmark_history = deque(maxlen=sequence_length)
        self.gesture_history = deque(maxlen=10)
        self.fps_history = deque(maxlen=30)
        
        # 手势状态跟踪
        self.hand_trackers = {}
        self.gesture_states = {}
        
        # 质量评估阈值
        self.quality_thresholds = {
            'min_hand_size': 30,
            'max_blur_variance': 50,
            'min_visibility': 0.5
        }
        
        logger.info(f"优化版手势识别器初始化完成 - MediaPipe: {MEDIAPIPE_AVAILABLE}")
    
    def _init_mediapipe(self):
        """初始化MediaPipe组件"""
        if not MEDIAPIPE_AVAILABLE:
            logger.error("MediaPipe不可用")
            self.hands = None
            return
        
        try:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            logger.info("MediaPipe手势检测初始化成功")
        except Exception as e:
            logger.error(f"MediaPipe初始化失败: {e}")
            self.hands = None
    
    def _init_gesture_categories(self):
        """初始化手势类别"""
        # 静态手势
        self.static_gestures = {
            'fist': '拳头',
            'open_palm': '张开手掌',
            'thumbs_up': '点赞',
            'thumbs_down': '点踩',
            'peace': '胜利手势',
            'ok': 'OK手势',
            'pointing': '指向',
            'rock': '摇滚手势',
            'call_me': '打电话手势',
            'stop': '停止手势',
            'gun': '手枪手势',
            'heart': '爱心手势',
            'spider': '蜘蛛手势'
        }
        
        # 动态手势
        self.dynamic_gestures = {
            'wave': '挥手',
            'swipe_left': '向左滑动',
            'swipe_right': '向右滑动',
            'swipe_up': '向上滑动',
            'swipe_down': '向下滑动',
            'circle_clockwise': '顺时针画圆',
            'circle_counterclockwise': '逆时针画圆',
            'zoom_in': '放大手势',
            'zoom_out': '缩小手势',
            'grab': '抓取手势',
            'pinch': '捏合手势',
            'tap': '点击手势'
        }
        
        # 交互手势
        self.interaction_gestures = {
            'clap': '拍手',
            'high_five': '击掌',
            'handshake': '握手',
            'prayer': '祈祷',
            'applause': '鼓掌'
        }
    
    def detect_gestures(self, image: np.ndarray) -> GestureRecognitionResult:
        """
        检测手势并进行识别
        
        Args:
            image: 输入图像
            
        Returns:
            GestureRecognitionResult: 手势识别结果
        """
        start_time = time.time()
        
        # 输入验证
        if image is None or image.size == 0:
            logger.warning("输入图像为空")
            return self._get_default_result()
        
        # 确保图像编码正确
        image = self._ensure_proper_encoding(image)
        
        try:
            # 转换颜色空间
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 手部检测
            hands_data = self._detect_hands(rgb_image, image)
            
            # 动态手势识别
            if self.enable_dynamic_gestures:
                hands_data = self._enhance_with_dynamic_gestures(hands_data)
            
            # 交互检测
            interaction = {'type': 'none', 'confidence': 0.0}
            if self.enable_interaction_detection and len(hands_data) >= 2:
                interaction = self._detect_hand_interaction(hands_data)
            
            # 更新性能统计
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time, len(hands_data))
            
            return GestureRecognitionResult(
                hands_detected=len(hands_data),
                hands_data=hands_data,
                interaction=interaction,
                performance_metrics=self._get_performance_metrics(),
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"手势识别错误: {e}")
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
    
    def _detect_hands(self, rgb_image: np.ndarray, original_image: np.ndarray) -> List[HandDetectionResult]:
        """检测手部"""
        hands_data = []
        
        if not self.hands:
            return hands_data
        
        try:
            results = self.hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                for hand_idx, (hand_landmarks, handedness) in enumerate(
                    zip(results.multi_hand_landmarks, results.multi_handedness)
                ):
                    # 提取关键点
                    landmarks = self._extract_landmarks(hand_landmarks, rgb_image.shape)
                    
                    # 质量评估
                    quality_score = self._assess_hand_quality(landmarks, rgb_image)
                    if quality_score < 0.3:
                        continue
                    
                    # 计算边界框
                    bbox = self._calculate_hand_bbox(landmarks)
                    
                    # 计算手部中心和方向
                    center, orientation = self._calculate_hand_center_orientation(landmarks)
                    
                    # 静态手势识别
                    static_gesture = self._recognize_static_gesture(landmarks)
                    
                    # 创建手部检测结果
                    hand_data = HandDetectionResult(
                        hand_id=hand_idx,
                        handedness=handedness.classification[0].label,
                        handedness_confidence=handedness.classification[0].score,
                        landmarks=landmarks,
                        static_gesture=static_gesture,
                        dynamic_gesture={'gesture': 'none', 'confidence': 0.0},
                        bbox=bbox,
                        center=center,
                        orientation=orientation,
                        quality_score=quality_score,
                        timestamp=time.time()
                    )
                    
                    hands_data.append(hand_data)
                    
        except Exception as e:
            logger.error(f"手部检测错误: {e}")
        
        return hands_data
    
    def _extract_landmarks(self, hand_landmarks, image_shape: Tuple[int, int, int]) -> List[Tuple[int, int]]:
        """提取手部关键点坐标"""
        h, w, _ = image_shape
        landmarks = []
        
        for landmark in hand_landmarks.landmark:
            # 确保标量转换，避免数组转换错误
            x_val = landmark.x * w
            y_val = landmark.y * h
            x = int(x_val.item() if hasattr(x_val, 'item') else x_val)
            y = int(y_val.item() if hasattr(y_val, 'item') else y_val)
            landmarks.append((x, y))
        
        return landmarks
    
    def _assess_hand_quality(self, landmarks: List[Tuple[int, int]], image: np.ndarray) -> float:
        """评估手部检测质量"""
        try:
            if len(landmarks) == 0:
                return 0.0
            
            # 计算手部尺寸
            xs = [lm[0] for lm in landmarks]
            ys = [lm[1] for lm in landmarks]
            hand_width = max(xs) - min(xs)
            hand_height = max(ys) - min(ys)
            hand_size = max(hand_width, hand_height)
            
            # 尺寸评分
            size_score = min(1.0, hand_size / self.quality_thresholds['min_hand_size'])
            
            # 可见性评分（基于关键点分布）
            visibility_score = len(landmarks) / 21.0  # MediaPipe有21个关键点
            
            # 综合质量分数
            quality_score = (size_score * 0.6 + visibility_score * 0.4)
            return min(1.0, quality_score)
            
        except Exception as e:
            logger.error(f"质量评估错误: {e}")
            return 0.5
    
    def _calculate_hand_bbox(self, landmarks: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        """计算手部边界框"""
        if len(landmarks) == 0:
            return (0, 0, 0, 0)
        
        xs = [lm[0] for lm in landmarks]
        ys = [lm[1] for lm in landmarks]
        
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        
        # 添加边距
        margin = 10
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = x2 + margin
        y2 = y2 + margin
        
        return (x1, y1, x2, y2)
    
    def _calculate_hand_center_orientation(self, landmarks: List[Tuple[int, int]]) -> Tuple[Tuple[float, float], float]:
        """计算手部中心和方向"""
        if len(landmarks) < 21:
            return ((0, 0), 0)
        
        # 手腕到中指MCP的向量作为主方向
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        
        center_x = (wrist[0] + middle_mcp[0]) / 2
        center_y = (wrist[1] + middle_mcp[1]) / 2
        
        # 计算方向角度
        dx = middle_mcp[0] - wrist[0]
        dy = middle_mcp[1] - wrist[1]
        orientation = math.atan2(dy, dx)
        
        return ((center_x, center_y), orientation)
    
    def _recognize_static_gesture(self, landmarks: List[Tuple[int, int]]) -> Dict[str, Any]:
        """识别静态手势"""
        if len(landmarks) < 21:
            return {'gesture': 'unknown', 'confidence': 0.0}
        
        try:
            # 检测各种静态手势
            gesture_scores = {
                'fist': self._detect_fist(landmarks),
                'open_palm': self._detect_open_palm(landmarks),
                'thumbs_up': self._detect_thumbs_up(landmarks),
                'thumbs_down': self._detect_thumbs_down(landmarks),
                'peace': self._detect_peace_sign(landmarks),
                'ok': self._detect_ok_sign(landmarks),
                'pointing': self._detect_pointing(landmarks),
                'rock': self._detect_rock_sign(landmarks),
                'call_me': self._detect_call_me(landmarks),
                'stop': self._detect_stop_sign(landmarks)
            }
            
            # 找到最高分数的手势
            best_gesture = max(gesture_scores, key=gesture_scores.get)
            best_score = gesture_scores[best_gesture]
            
            if best_score > 0.6:
                return {
                    'gesture': best_gesture,
                    'gesture_name': self.static_gestures.get(best_gesture, '未知'),
                    'confidence': best_score,
                    'all_scores': gesture_scores
                }
            else:
                return {'gesture': 'unknown', 'confidence': 0.0}
                
        except Exception as e:
            logger.error(f"静态手势识别错误: {e}")
            return {'gesture': 'unknown', 'confidence': 0.0}
    
    def _detect_fist(self, landmarks: List[Tuple[int, int]]) -> float:
        """检测拳头手势"""
        try:
            # 检查所有手指是否弯曲
            finger_tips = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]  # 食指、中指、无名指、小指尖
            finger_mcps = [landmarks[6], landmarks[10], landmarks[14], landmarks[18]]  # 对应MCP关节
            
            bent_fingers = 0
            for tip, mcp in zip(finger_tips, finger_mcps):
                if tip[1] > mcp[1]:  # 手指尖在MCP关节下方（弯曲）
                    bent_fingers += 1
            
            # 检查拇指是否弯曲
            thumb_tip = landmarks[4]
            thumb_mcp = landmarks[2]
            thumb_bent = abs(thumb_tip[0] - thumb_mcp[0]) < 30  # 拇指收缩
            
            if bent_fingers >= 3 and thumb_bent:
                return 0.9
            elif bent_fingers >= 3:
                return 0.7
            else:
                return 0.0
                
        except (IndexError, TypeError):
            return 0.0
    
    def _detect_open_palm(self, landmarks: List[Tuple[int, int]]) -> float:
        """检测张开手掌"""
        try:
            # 检查所有手指是否伸直
            finger_tips = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
            finger_mcps = [landmarks[6], landmarks[10], landmarks[14], landmarks[18]]
            
            extended_fingers = 0
            for tip, mcp in zip(finger_tips, finger_mcps):
                if tip[1] < mcp[1]:  # 手指尖在MCP关节上方（伸直）
                    extended_fingers += 1
            
            # 检查拇指是否伸直
            thumb_tip = landmarks[4]
            thumb_mcp = landmarks[2]
            thumb_extended = abs(thumb_tip[0] - thumb_mcp[0]) > 40
            
            if extended_fingers >= 4 and thumb_extended:
                return 0.9
            elif extended_fingers >= 3:
                return 0.6
            else:
                return 0.0
                
        except (IndexError, TypeError):
            return 0.0
    
    def _detect_thumbs_up(self, landmarks: List[Tuple[int, int]]) -> float:
        """检测点赞手势"""
        try:
            # 拇指伸直向上
            thumb_tip = landmarks[4]
            thumb_mcp = landmarks[2]
            wrist = landmarks[0]
            
            # 拇指向上
            thumb_up = thumb_tip[1] < thumb_mcp[1] and thumb_tip[1] < wrist[1]
            
            # 其他手指弯曲
            finger_tips = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
            finger_mcps = [landmarks[6], landmarks[10], landmarks[14], landmarks[18]]
            
            bent_fingers = sum(1 for tip, mcp in zip(finger_tips, finger_mcps) if tip[1] > mcp[1])
            
            if thumb_up and bent_fingers >= 3:
                return 0.9
            elif thumb_up and bent_fingers >= 2:
                return 0.6
            else:
                return 0.0
                
        except (IndexError, TypeError):
            return 0.0
    
    def _detect_thumbs_down(self, landmarks: List[Tuple[int, int]]) -> float:
        """检测点踩手势"""
        try:
            # 拇指伸直向下
            thumb_tip = landmarks[4]
            thumb_mcp = landmarks[2]
            wrist = landmarks[0]
            
            # 拇指向下
            thumb_down = thumb_tip[1] > thumb_mcp[1] and thumb_tip[1] > wrist[1]
            
            # 其他手指弯曲
            finger_tips = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
            finger_mcps = [landmarks[6], landmarks[10], landmarks[14], landmarks[18]]
            
            bent_fingers = sum(1 for tip, mcp in zip(finger_tips, finger_mcps) if tip[1] > mcp[1])
            
            if thumb_down and bent_fingers >= 3:
                return 0.9
            else:
                return 0.0
                
        except (IndexError, TypeError):
            return 0.0
    
    def _detect_peace_sign(self, landmarks: List[Tuple[int, int]]) -> float:
        """检测胜利手势（V字）"""
        try:
            # 食指和中指伸直
            index_tip = landmarks[8]
            index_mcp = landmarks[6]
            middle_tip = landmarks[12]
            middle_mcp = landmarks[10]
            
            index_extended = index_tip[1] < index_mcp[1]
            middle_extended = middle_tip[1] < middle_mcp[1]
            
            # 无名指和小指弯曲
            ring_tip = landmarks[16]
            ring_mcp = landmarks[14]
            pinky_tip = landmarks[20]
            pinky_mcp = landmarks[18]
            
            ring_bent = ring_tip[1] > ring_mcp[1]
            pinky_bent = pinky_tip[1] > pinky_mcp[1]
            
            # 食指和中指分开
            fingers_apart = abs(index_tip[0] - middle_tip[0]) > 20
            
            if index_extended and middle_extended and ring_bent and pinky_bent and fingers_apart:
                return 0.9
            elif index_extended and middle_extended:
                return 0.6
            else:
                return 0.0
                
        except (IndexError, TypeError):
            return 0.0
    
    def _detect_ok_sign(self, landmarks: List[Tuple[int, int]]) -> float:
        """检测OK手势"""
        try:
            # 拇指和食指形成圆圈
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            
            # 计算拇指和食指尖的距离
            distance = math.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)
            
            # 其他手指伸直
            middle_tip = landmarks[12]
            middle_mcp = landmarks[10]
            ring_tip = landmarks[16]
            ring_mcp = landmarks[14]
            pinky_tip = landmarks[20]
            pinky_mcp = landmarks[18]
            
            other_fingers_extended = (
                middle_tip[1] < middle_mcp[1] and
                ring_tip[1] < ring_mcp[1] and
                pinky_tip[1] < pinky_mcp[1]
            )
            
            if distance < 30 and other_fingers_extended:
                return 0.9
            elif distance < 40:
                return 0.6
            else:
                return 0.0
                
        except (IndexError, TypeError):
            return 0.0
    
    def _detect_pointing(self, landmarks: List[Tuple[int, int]]) -> float:
        """检测指向手势"""
        try:
            # 检查数组长度
            if len(landmarks) < 21:
                return 0.0
                
            # 只有食指伸直
            index_tip = landmarks[8]
            index_mcp = landmarks[6]
            index_extended = index_tip[1] < index_mcp[1]
            
            # 其他手指弯曲
            other_tips = [landmarks[12], landmarks[16], landmarks[20]]
            other_mcps = [landmarks[10], landmarks[14], landmarks[18]]
            
            other_fingers_bent = sum(1 for tip, mcp in zip(other_tips, other_mcps) if tip[1] > mcp[1])
            
            if index_extended and other_fingers_bent >= 2:
                return 0.9
            elif index_extended:
                return 0.6
            else:
                return 0.0
                
        except (IndexError, TypeError):
            return 0.0
    
    def _detect_rock_sign(self, landmarks: List[Tuple[int, int]]) -> float:
        """检测摇滚手势"""
        try:
            # 食指和小指伸直
            index_tip = landmarks[8]
            index_mcp = landmarks[6]
            pinky_tip = landmarks[20]
            pinky_mcp = landmarks[18]
            
            index_extended = index_tip[1] < index_mcp[1]
            pinky_extended = pinky_tip[1] < pinky_mcp[1]
            
            # 中指和无名指弯曲
            middle_tip = landmarks[12]
            middle_mcp = landmarks[10]
            ring_tip = landmarks[16]
            ring_mcp = landmarks[14]
            
            middle_bent = middle_tip[1] > middle_mcp[1]
            ring_bent = ring_tip[1] > ring_mcp[1]
            
            if index_extended and pinky_extended and middle_bent and ring_bent:
                return 0.9
            else:
                return 0.0
                
        except (IndexError, TypeError):
            return 0.0
    
    def _detect_call_me(self, landmarks: List[Tuple[int, int]]) -> float:
        """检测打电话手势"""
        try:
            # 拇指和小指伸直
            thumb_tip = landmarks[4]
            thumb_mcp = landmarks[2]
            pinky_tip = landmarks[20]
            pinky_mcp = landmarks[18]
            
            thumb_extended = abs(thumb_tip[0] - thumb_mcp[0]) > 30
            pinky_extended = pinky_tip[1] < pinky_mcp[1]
            
            # 其他手指弯曲
            other_tips = [landmarks[8], landmarks[12], landmarks[16]]
            other_mcps = [landmarks[6], landmarks[10], landmarks[14]]
            
            other_fingers_bent = sum(1 for tip, mcp in zip(other_tips, other_mcps) if tip[1] > mcp[1])
            
            if thumb_extended and pinky_extended and other_fingers_bent >= 2:
                return 0.9
            else:
                return 0.0
                
        except (IndexError, TypeError):
            return 0.0
    
    def _detect_stop_sign(self, landmarks: List[Tuple[int, int]]) -> float:
        """检测停止手势"""
        try:
            # 所有手指伸直，手掌朝前
            finger_tips = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
            finger_mcps = [landmarks[6], landmarks[10], landmarks[14], landmarks[18]]
            
            extended_fingers = sum(1 for tip, mcp in zip(finger_tips, finger_mcps) if tip[1] < mcp[1])
            
            # 拇指也要伸直
            thumb_tip = landmarks[4]
            thumb_mcp = landmarks[2]
            thumb_extended = abs(thumb_tip[0] - thumb_mcp[0]) > 30
            
            if extended_fingers >= 4 and thumb_extended:
                return 0.8  # 与open_palm区分
            else:
                return 0.0
                
        except (IndexError, TypeError):
            return 0.0
    
    def _enhance_with_dynamic_gestures(self, hands_data: List[HandDetectionResult]) -> List[HandDetectionResult]:
        """增强动态手势识别"""
        if not self.enable_dynamic_gestures:
            return hands_data
        
        try:
            # 更新历史数据
            current_landmarks = [hand.landmarks for hand in hands_data]
            self.landmark_history.append(current_landmarks)
            
            # 为每个手部添加动态手势识别结果
            for i, hand_data in enumerate(hands_data):
                if i < len(current_landmarks):
                    dynamic_gesture = self._recognize_dynamic_gesture(i)
                    hand_data.dynamic_gesture = dynamic_gesture
            
            return hands_data
            
        except Exception as e:
            logger.error(f"动态手势识别错误: {e}")
            return hands_data
    
    def _recognize_dynamic_gesture(self, hand_index: int) -> Dict[str, Any]:
        """识别动态手势"""
        if len(self.landmark_history) < 10:
            return {'gesture': 'none', 'confidence': 0.0}
        
        try:
            # 获取该手部的轨迹
            trajectory = []
            for frame_landmarks in list(self.landmark_history)[-10:]:
                if hand_index < len(frame_landmarks) and frame_landmarks[hand_index]:
                    # 使用手腕作为轨迹点
                    wrist = frame_landmarks[hand_index][0]
                    trajectory.append(wrist)
            
            if len(trajectory) < 5:
                return {'gesture': 'none', 'confidence': 0.0}
            
            # 分析轨迹模式
            gesture, confidence = self._analyze_trajectory(trajectory)
            
            return {
                'gesture': gesture,
                'gesture_name': self.dynamic_gestures.get(gesture, '未知'),
                'confidence': confidence,
                'trajectory_length': len(trajectory)
            }
            
        except Exception as e:
            logger.error(f"动态手势识别错误: {e}")
            return {'gesture': 'none', 'confidence': 0.0}
    
    def _analyze_trajectory(self, trajectory: List[Tuple[int, int]]) -> Tuple[str, float]:
        """分析轨迹模式"""
        if len(trajectory) < 3:
            return 'none', 0.0
        
        try:
            # 计算轨迹特征
            total_distance = 0
            direction_changes = 0
            dx_total = 0
            dy_total = 0
            
            for i in range(1, len(trajectory)):
                dx = trajectory[i][0] - trajectory[i-1][0]
                dy = trajectory[i][1] - trajectory[i-1][1]
                
                distance = math.sqrt(dx*dx + dy*dy)
                total_distance += distance
                
                dx_total += dx
                dy_total += dy
                
                # 检测方向变化
                if i > 1:
                    prev_dx = trajectory[i-1][0] - trajectory[i-2][0]
                    prev_dy = trajectory[i-1][1] - trajectory[i-2][1]
                    
                    if (dx * prev_dx + dy * prev_dy) < 0:  # 方向相反
                        direction_changes += 1
            
            # 基于特征识别手势
            if total_distance < 20:
                return 'none', 0.0
            
            # 挥手：多次方向变化
            if direction_changes >= 3 and total_distance > 100:
                return 'wave', 0.8
            
            # 滑动手势：主要在一个方向
            if abs(dx_total) > abs(dy_total) * 2:
                if dx_total > 50:
                    return 'swipe_right', 0.7
                elif dx_total < -50:
                    return 'swipe_left', 0.7
            elif abs(dy_total) > abs(dx_total) * 2:
                if dy_total > 50:
                    return 'swipe_down', 0.7
                elif dy_total < -50:
                    return 'swipe_up', 0.7
            
            # 圆形轨迹
            if self._is_circular_trajectory(trajectory):
                return 'circle_clockwise', 0.6
            
            return 'none', 0.0
            
        except Exception as e:
            logger.error(f"轨迹分析错误: {e}")
            return 'none', 0.0
    
    def _is_circular_trajectory(self, trajectory: List[Tuple[int, int]]) -> bool:
        """检测是否为圆形轨迹"""
        if len(trajectory) < 8:
            return False
        
        try:
            # 计算轨迹的中心点
            center_x = sum(p[0] for p in trajectory) / len(trajectory)
            center_y = sum(p[1] for p in trajectory) / len(trajectory)
            
            # 计算到中心的距离变化
            distances = []
            for point in trajectory:
                dist = math.sqrt((point[0] - center_x)**2 + (point[1] - center_y)**2)
                distances.append(dist)
            
            # 检查距离是否相对稳定（圆形特征）
            avg_distance = sum(distances) / len(distances)
            variance = sum((d - avg_distance)**2 for d in distances) / len(distances)
            
            return variance < (avg_distance * 0.3)**2
            
        except Exception:
            return False
    
    def _detect_hand_interaction(self, hands_data: List[HandDetectionResult]) -> Dict[str, Any]:
        """检测双手交互"""
        if len(hands_data) < 2:
            return {'type': 'none', 'confidence': 0.0}
        
        try:
            hand1, hand2 = hands_data[0], hands_data[1]
            
            # 计算双手距离
            distance = math.sqrt(
                (hand1.center[0] - hand2.center[0])**2 + 
                (hand1.center[1] - hand2.center[1])**2
            )
            
            # 拍手检测
            if distance < 100:
                # 检查是否都是张开的手掌
                hand1_open = hand1.static_gesture.get('gesture') == 'open_palm'
                hand2_open = hand2.static_gesture.get('gesture') == 'open_palm'
                
                if hand1_open and hand2_open:
                    return {'type': 'clap', 'confidence': 0.8, 'distance': distance}
            
            # 祈祷手势检测
            if distance < 50:
                # 检查手掌是否相对
                orientation_diff = abs(hand1.orientation - hand2.orientation)
                if orientation_diff > math.pi * 0.8:  # 接近相对
                    return {'type': 'prayer', 'confidence': 0.7, 'distance': distance}
            
            return {'type': 'none', 'confidence': 0.0}
            
        except Exception as e:
            logger.error(f"交互检测错误: {e}")
            return {'type': 'none', 'confidence': 0.0}
    
    def _update_performance_stats(self, processing_time: float, num_hands: int):
        """更新性能统计"""
        self.performance_stats['total_detections'] += 1
        
        # 计算平均处理时间
        total_time = (self.performance_stats['avg_processing_time'] * 
                     (self.performance_stats['total_detections'] - 1) + processing_time)
        self.performance_stats['avg_processing_time'] = total_time / self.performance_stats['total_detections']
        
        # 更新FPS
        if processing_time > 0:
            current_fps = 1.0 / processing_time
            self.fps_history.append(current_fps)
            self.performance_stats['fps'] = sum(self.fps_history) / len(self.fps_history)
        
        if num_hands > 0:
            self.performance_stats['successful_gestures'] += 1
        else:
            self.performance_stats['failed_detections'] += 1
    
    def _get_performance_metrics(self) -> Dict[str, float]:
        """获取性能指标"""
        stats = self.performance_stats.copy()
        if stats['total_detections'] > 0:
            stats['success_rate'] = stats['successful_gestures'] / stats['total_detections']
        else:
            stats['success_rate'] = 0.0
        return stats
    
    def _get_default_result(self) -> GestureRecognitionResult:
        """获取默认结果"""
        return GestureRecognitionResult(
            hands_detected=0,
            hands_data=[],
            interaction={'type': 'none', 'confidence': 0.0},
            performance_metrics=self._get_performance_metrics(),
            timestamp=time.time()
        )
    
    def draw_annotations(self, image: np.ndarray, result: GestureRecognitionResult) -> np.ndarray:
        """在图像上绘制手势识别结果"""
        annotated_image = image.copy()
        
        try:
            for hand_data in result.hands_data:
                # 绘制手部关键点
                self._draw_hand_landmarks(annotated_image, hand_data.landmarks)
                
                # 绘制边界框
                self._draw_hand_bbox(annotated_image, hand_data.bbox, hand_data.handedness)
                
                # 绘制手势标签
                self._draw_gesture_labels(annotated_image, hand_data)
            
            # 绘制交互信息
            if result.interaction['type'] != 'none':
                self._draw_interaction_info(annotated_image, result.interaction)
            
            # 绘制性能信息
            self._draw_performance_info(annotated_image, result.performance_metrics)
            
        except Exception as e:
            logger.error(f"绘制标注错误: {e}")
        
        return annotated_image
    
    def _draw_hand_landmarks(self, image: np.ndarray, landmarks: List[Tuple[int, int]]):
        """绘制手部关键点"""
        if len(landmarks) == 0 or not MEDIAPIPE_AVAILABLE:
            return
        
        # 绘制关键点
        for point in landmarks:
            cv2.circle(image, point, 3, (0, 255, 0), -1)
        
        # 绘制连接线（简化版）
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
            (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
            (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
            (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
            (0, 17), (17, 18), (18, 19), (19, 20)  # 小指
        ]
        
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                cv2.line(image, landmarks[start_idx], landmarks[end_idx], (255, 255, 255), 2)
    
    def _draw_hand_bbox(self, image: np.ndarray, bbox: Tuple[int, int, int, int], handedness: str):
        """绘制手部边界框"""
        x1, y1, x2, y2 = bbox
        # 修复镜像显示问题：在镜像视频中左右手标签需要反转
        display_handedness = 'Left' if handedness == 'Right' else 'Right'
        color = (0, 255, 0) if display_handedness == 'Right' else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签
        label = f"{display_handedness} Hand"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def _draw_gesture_labels(self, image: np.ndarray, hand_data: HandDetectionResult):
        """绘制手势标签"""
        # 确保标量转换，避免数组转换错误
        center_x = hand_data.center[0]
        center_y = hand_data.center[1]
        x = int(center_x.item() if hasattr(center_x, 'item') else center_x)
        y = int(center_y.item() if hasattr(center_y, 'item') else center_y)
        
        # 静态手势 - 使用英文避免乱码
        static_gesture = hand_data.static_gesture
        if static_gesture.get('gesture') != 'unknown':
            gesture_name = static_gesture.get('gesture', 'unknown')
            static_label = f"Static: {gesture_name}"
            cv2.putText(image, static_label, (x - 50, y - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # 动态手势 - 使用英文避免乱码
        dynamic_gesture = hand_data.dynamic_gesture
        if dynamic_gesture.get('gesture') != 'none':
            gesture_name = dynamic_gesture.get('gesture', 'none')
            dynamic_label = f"Dynamic: {gesture_name}"
            cv2.putText(image, dynamic_label, (x - 50, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    def _draw_interaction_info(self, image: np.ndarray, interaction: Dict[str, Any]):
        """绘制交互信息"""
        if interaction['type'] != 'none':
            label = f"Interaction: {interaction['type']} ({interaction['confidence']:.2f})"
            cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    def _draw_performance_info(self, image: np.ndarray, metrics: Dict[str, float]):
        """绘制性能信息"""
        fps_text = f"FPS: {metrics.get('fps', 0):.1f}"
        cv2.putText(image, fps_text, (10, image.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        return self.performance_stats.copy()
    
    def get_supported_gestures(self) -> Dict[str, Dict[str, str]]:
        """获取支持的手势列表"""
        return {
            'static_gestures': self.static_gestures,
            'dynamic_gestures': self.dynamic_gestures,
            'interaction_gestures': self.interaction_gestures
        }
    
    def optimize_for_realtime(self):
        """优化实时性能"""
        try:
            # 降低检测置信度以提高速度
            if self.hands:
                self.hands.close()
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=min(2, self.max_num_hands),  # 减少最大检测手数
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.3
                )
            
            # 减少历史数据长度
            self.sequence_length = min(15, self.sequence_length)
            self.landmark_history = deque(maxlen=self.sequence_length)
            
            logger.info("手势识别实时性能优化完成")
        except Exception as e:
            logger.error(f"性能优化失败: {e}")
    
    def close(self):
        """清理资源"""
        try:
            if self.thread_pool:
                self.thread_pool.shutdown(wait=True)
            
            if hasattr(self, 'hands') and self.hands:
                self.hands.close()
            
            logger.info("优化版手势识别器资源清理完成")
        except Exception as e:
            logger.error(f"资源清理错误: {e}")


# 工厂函数
def create_optimized_gesture_recognizer(**kwargs) -> OptimizedGestureRecognizer:
    """创建优化版手势识别器"""
    return OptimizedGestureRecognizer(**kwargs)