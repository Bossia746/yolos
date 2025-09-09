"""增强手势识别模块 - 基于MediaPipe和深度学习的高精度手势识别"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import math
from collections import deque
import time

try:
    import mediapipe as mp
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe or TensorFlow not available for enhanced gesture recognition")


class EnhancedGestureRecognizer:
    """增强手势识别器 - 基于MediaPipe Hands和深度学习的手势识别"""
    
    def __init__(self, 
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5,
                 sequence_length: int = 30):
        """
        初始化增强手势识别器
        
        Args:
            max_num_hands: 最大检测手数
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
            sequence_length: 序列长度（用于动态手势）
        """
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.sequence_length = sequence_length
        
        # MediaPipe初始化
        if MEDIAPIPE_AVAILABLE:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
        else:
            self.hands = None
        
        # 手势类别定义
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
            'stop': '停止手势'
        }
        
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
            'grab': '抓取手势'
        }
        
        # 历史数据存储
        self.landmark_history = deque(maxlen=sequence_length)
        self.gesture_history = deque(maxlen=10)
        
        # 深度学习模型
        self.dynamic_model = None
        self.model_trained = False
        
        # 手势状态
        self.current_gesture = None
        self.gesture_confidence = 0.0
        self.gesture_start_time = None
        
        print("增强手势识别器初始化完成")
    
    def recognize_gestures(self, image: np.ndarray) -> Dict[str, Any]:
        """
        识别手势
        
        Args:
            image: 输入图像
            
        Returns:
            Dict: 识别结果
        """
        if not MEDIAPIPE_AVAILABLE or self.hands is None:
            return self._get_default_result()
        
        try:
            # 转换颜色空间
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb_image.flags.writeable = False
            
            # 检测手部
            results = self.hands.process(rgb_image)
            
            # 恢复图像可写性
            rgb_image.flags.writeable = True
            
            hands_data = []
            
            if results.multi_hand_landmarks:
                for hand_idx, (hand_landmarks, handedness) in enumerate(
                    zip(results.multi_hand_landmarks, results.multi_handedness)
                ):
                    # 提取关键点
                    landmarks = self._extract_landmarks(hand_landmarks)
                    
                    # 识别静态手势
                    static_gesture = self._recognize_static_gesture(landmarks)
                    
                    # 更新历史数据
                    self._update_history(landmarks)
                    
                    # 识别动态手势
                    dynamic_gesture = self._recognize_dynamic_gesture()
                    
                    # 计算手部边界框
                    bbox = self._calculate_hand_bbox(landmarks, image.shape)
                    
                    # 计算手部中心和方向
                    center, orientation = self._calculate_hand_center_orientation(landmarks)
                    
                    hand_data = {
                        'hand_id': hand_idx,
                        'handedness': handedness.classification[0].label,
                        'handedness_confidence': handedness.classification[0].score,
                        'landmarks': landmarks,
                        'static_gesture': static_gesture,
                        'dynamic_gesture': dynamic_gesture,
                        'bbox': bbox,
                        'center': center,
                        'orientation': orientation,
                        'timestamp': time.time()
                    }
                    
                    hands_data.append(hand_data)
            
            # 多手交互检测
            interaction = self._detect_hand_interaction(hands_data)
            
            return {
                'hands_detected': len(hands_data),
                'hands_data': hands_data,
                'interaction': interaction,
                'timestamp': time.time()
            }
            
        except Exception as e:
            print(f"手势识别错误: {e}")
            return self._get_default_result()
    
    def _extract_landmarks(self, hand_landmarks) -> List[Tuple[float, float, float]]:
        """提取手部关键点"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y, landmark.z))
        return landmarks
    
    def _recognize_static_gesture(self, landmarks: List[Tuple[float, float, float]]) -> Dict[str, Any]:
        """识别静态手势"""
        if len(landmarks) != 21:  # MediaPipe手部有21个关键点
            return {'gesture': 'unknown', 'confidence': 0.0}
        
        try:
            # 计算手指状态
            finger_states = self._get_finger_states(landmarks)
            
            # 基于手指状态识别手势
            gesture, confidence = self._classify_static_gesture(finger_states, landmarks)
            
            return {
                'gesture': gesture,
                'confidence': confidence,
                'finger_states': finger_states
            }
            
        except Exception as e:
            print(f"静态手势识别错误: {e}")
            return {'gesture': 'unknown', 'confidence': 0.0}
    
    def _get_finger_states(self, landmarks: List[Tuple[float, float, float]]) -> Dict[str, bool]:
        """获取手指状态（伸展/弯曲）"""
        # MediaPipe手部关键点索引
        THUMB_TIP = 4
        THUMB_IP = 3
        THUMB_MCP = 2
        
        INDEX_TIP = 8
        INDEX_PIP = 6
        INDEX_MCP = 5
        
        MIDDLE_TIP = 12
        MIDDLE_PIP = 10
        MIDDLE_MCP = 9
        
        RING_TIP = 16
        RING_PIP = 14
        RING_MCP = 13
        
        PINKY_TIP = 20
        PINKY_PIP = 18
        PINKY_MCP = 17
        
        # 判断手指是否伸展
        def is_finger_extended(tip_idx, pip_idx, mcp_idx):
            tip_y = landmarks[tip_idx][1]
            pip_y = landmarks[pip_idx][1]
            mcp_y = landmarks[mcp_idx][1]
            
            # 指尖应该比关节更靠近手腕（y值更小）
            return tip_y < pip_y and pip_y < mcp_y
        
        # 拇指特殊处理（横向判断）
        def is_thumb_extended():
            thumb_tip_x = landmarks[THUMB_TIP][0]
            thumb_ip_x = landmarks[THUMB_IP][0]
            thumb_mcp_x = landmarks[THUMB_MCP][0]
            
            # 拇指伸展时，指尖应该远离手掌中心
            wrist_x = landmarks[0][0]
            return abs(thumb_tip_x - wrist_x) > abs(thumb_ip_x - wrist_x)
        
        return {
            'thumb': is_thumb_extended(),
            'index': is_finger_extended(INDEX_TIP, INDEX_PIP, INDEX_MCP),
            'middle': is_finger_extended(MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
            'ring': is_finger_extended(RING_TIP, RING_PIP, RING_MCP),
            'pinky': is_finger_extended(PINKY_TIP, PINKY_PIP, PINKY_MCP)
        }
    
    def _classify_static_gesture(self, finger_states: Dict[str, bool], 
                               landmarks: List[Tuple[float, float, float]]) -> Tuple[str, float]:
        """基于手指状态分类静态手势"""
        extended_fingers = [name for name, extended in finger_states.items() if extended]
        extended_count = len(extended_fingers)
        
        # 手势识别规则
        if extended_count == 0:
            return 'fist', 0.9
        
        elif extended_count == 5:
            return 'open_palm', 0.9
        
        elif extended_count == 1:
            if finger_states['thumb']:
                # 判断拇指方向
                thumb_tip = landmarks[4]
                thumb_mcp = landmarks[2]
                if thumb_tip[1] < thumb_mcp[1]:  # 拇指向上
                    return 'thumbs_up', 0.85
                else:
                    return 'thumbs_down', 0.85
            elif finger_states['index']:
                return 'pointing', 0.85
        
        elif extended_count == 2:
            if finger_states['index'] and finger_states['middle']:
                # 检查是否是胜利手势
                index_tip = landmarks[8]
                middle_tip = landmarks[12]
                distance = math.sqrt((index_tip[0] - middle_tip[0])**2 + (index_tip[1] - middle_tip[1])**2)
                if distance > 0.05:  # 手指分开
                    return 'peace', 0.85
            elif finger_states['thumb'] and finger_states['index']:
                # 检查是否是OK手势
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                distance = math.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)
                if distance < 0.03:  # 拇指和食指接触
                    return 'ok', 0.85
        
        elif extended_count == 3:
            if finger_states['index'] and finger_states['middle'] and finger_states['ring']:
                return 'rock', 0.8
            elif finger_states['thumb'] and finger_states['index'] and finger_states['pinky']:
                return 'call_me', 0.8
        
        elif extended_count == 4:
            if not finger_states['thumb']:
                return 'stop', 0.8
        
        return 'unknown', 0.0
    
    def _recognize_dynamic_gesture(self) -> Dict[str, Any]:
        """识别动态手势"""
        if len(self.landmark_history) < 10:
            return {'gesture': 'none', 'confidence': 0.0}
        
        try:
            # 获取最近的轨迹
            recent_landmarks = list(self.landmark_history)[-10:]
            
            # 计算手部中心轨迹
            trajectory = []
            for landmarks in recent_landmarks:
                if len(landmarks) > 9:
                    # 计算手掌中心（手腕到中指MCP的中点）
                    wrist = landmarks[0]
                    middle_mcp = landmarks[9]
                    center = ((wrist[0] + middle_mcp[0]) / 2, (wrist[1] + middle_mcp[1]) / 2)
                    trajectory.append(center)
            
            if len(trajectory) < 5:
                return {'gesture': 'none', 'confidence': 0.0}
            
            # 分析轨迹模式
            gesture, confidence = self._analyze_trajectory(trajectory)
            
            return {
                'gesture': gesture,
                'confidence': confidence,
                'trajectory': trajectory
            }
            
        except Exception as e:
            print(f"动态手势识别错误: {e}")
            return {'gesture': 'none', 'confidence': 0.0}
    
    def _analyze_trajectory(self, trajectory: List[Tuple[float, float]]) -> Tuple[str, float]:
        """分析轨迹模式"""
        if len(trajectory) < 5:
            return 'none', 0.0
        
        # 计算总体运动方向和距离
        start_point = trajectory[0]
        end_point = trajectory[-1]
        
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        total_distance = math.sqrt(dx**2 + dy**2)
        
        # 如果运动距离太小，认为是静止
        if total_distance < 0.05:
            return 'none', 0.0
        
        # 计算运动角度
        angle = math.degrees(math.atan2(dy, dx))
        
        # 分析轨迹特征
        # 1. 直线运动检测
        if self._is_linear_motion(trajectory):
            if abs(dx) > abs(dy):  # 水平运动
                if dx > 0:
                    return 'swipe_right', 0.8
                else:
                    return 'swipe_left', 0.8
            else:  # 垂直运动
                if dy > 0:
                    return 'swipe_down', 0.8
                else:
                    return 'swipe_up', 0.8
        
        # 2. 圆形运动检测
        if self._is_circular_motion(trajectory):
            if self._is_clockwise(trajectory):
                return 'circle_clockwise', 0.75
            else:
                return 'circle_counterclockwise', 0.75
        
        # 3. 挥手检测
        if self._is_waving_motion(trajectory):
            return 'wave', 0.7
        
        return 'unknown', 0.0
    
    def _is_linear_motion(self, trajectory: List[Tuple[float, float]]) -> bool:
        """检测是否为直线运动"""
        if len(trajectory) < 3:
            return False
        
        # 计算轨迹点到直线的平均距离
        start = trajectory[0]
        end = trajectory[-1]
        
        total_deviation = 0
        for point in trajectory[1:-1]:
            # 计算点到直线的距离
            deviation = self._point_to_line_distance(point, start, end)
            total_deviation += deviation
        
        avg_deviation = total_deviation / max(1, len(trajectory) - 2)
        return avg_deviation < 0.02  # 阈值可调整
    
    def _is_circular_motion(self, trajectory: List[Tuple[float, float]]) -> bool:
        """检测是否为圆形运动"""
        if len(trajectory) < 8:
            return False
        
        # 计算轨迹的重心
        center_x = sum(p[0] for p in trajectory) / len(trajectory)
        center_y = sum(p[1] for p in trajectory) / len(trajectory)
        center = (center_x, center_y)
        
        # 计算每个点到重心的距离
        distances = [math.sqrt((p[0] - center[0])**2 + (p[1] - center[1])**2) for p in trajectory]
        
        # 检查距离的方差（圆形运动距离应该相对稳定）
        avg_distance = sum(distances) / len(distances)
        variance = sum((d - avg_distance)**2 for d in distances) / len(distances)
        
        return variance < 0.001 and avg_distance > 0.03  # 阈值可调整
    
    def _is_clockwise(self, trajectory: List[Tuple[float, float]]) -> bool:
        """判断圆形运动是否为顺时针"""
        if len(trajectory) < 3:
            return True
        
        # 计算角度变化的总和
        angle_sum = 0
        for i in range(len(trajectory) - 2):
            p1, p2, p3 = trajectory[i], trajectory[i+1], trajectory[i+2]
            
            # 计算向量角度
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # 计算叉积（判断转向）
            cross_product = v1[0] * v2[1] - v1[1] * v2[0]
            angle_sum += cross_product
        
        return angle_sum < 0  # 负值表示顺时针
    
    def _is_waving_motion(self, trajectory: List[Tuple[float, float]]) -> bool:
        """检测是否为挥手运动"""
        if len(trajectory) < 6:
            return False
        
        # 检测左右摆动模式
        x_coords = [p[0] for p in trajectory]
        
        # 寻找局部极值点
        peaks = []
        for i in range(1, len(x_coords) - 1):
            if (x_coords[i] > x_coords[i-1] and x_coords[i] > x_coords[i+1]) or \
               (x_coords[i] < x_coords[i-1] and x_coords[i] < x_coords[i+1]):
                peaks.append(i)
        
        # 挥手应该有多个峰值
        return len(peaks) >= 2
    
    def _point_to_line_distance(self, point: Tuple[float, float], 
                               line_start: Tuple[float, float], 
                               line_end: Tuple[float, float]) -> float:
        """计算点到直线的距离"""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # 直线方程: (y2-y1)x - (x2-x1)y + x2*y1 - y2*x1 = 0
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        
        return numerator / max(denominator, 1e-6)
    
    def _update_history(self, landmarks: List[Tuple[float, float, float]]):
        """更新历史数据"""
        self.landmark_history.append(landmarks)
    
    def _calculate_hand_bbox(self, landmarks: List[Tuple[float, float, float]], 
                           image_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        """计算手部边界框"""
        h, w = image_shape[:2]
        
        x_coords = [lm[0] * w for lm in landmarks]
        y_coords = [lm[1] * h for lm in landmarks]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # 添加边距
        margin = 20
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(w, x_max + margin)
        y_max = min(h, y_max + margin)
        
        return (x_min, y_min, x_max, y_max)
    
    def _calculate_hand_center_orientation(self, landmarks: List[Tuple[float, float, float]]) -> Tuple[Tuple[float, float], float]:
        """计算手部中心和方向"""
        # 手掌中心（手腕到中指MCP的中点）
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        center = ((wrist[0] + middle_mcp[0]) / 2, (wrist[1] + middle_mcp[1]) / 2)
        
        # 手部方向（从手腕到中指方向）
        dx = middle_mcp[0] - wrist[0]
        dy = middle_mcp[1] - wrist[1]
        orientation = math.degrees(math.atan2(dy, dx))
        
        return center, orientation
    
    def _detect_hand_interaction(self, hands_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """检测双手交互"""
        if len(hands_data) < 2:
            return {'type': 'none', 'confidence': 0.0}
        
        try:
            hand1, hand2 = hands_data[0], hands_data[1]
            
            # 计算双手距离
            center1 = hand1['center']
            center2 = hand2['center']
            distance = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
            
            # 检测特定的双手手势
            gesture1 = hand1['static_gesture']['gesture']
            gesture2 = hand2['static_gesture']['gesture']
            
            # 拍手检测
            if distance < 0.1 and gesture1 == 'open_palm' and gesture2 == 'open_palm':
                return {'type': 'clap', 'confidence': 0.8, 'distance': distance}
            
            # 缩放手势检测
            if gesture1 == 'fist' and gesture2 == 'fist':
                if distance > 0.2:
                    return {'type': 'zoom_out', 'confidence': 0.7, 'distance': distance}
                elif distance < 0.1:
                    return {'type': 'zoom_in', 'confidence': 0.7, 'distance': distance}
            
            return {'type': 'proximity', 'confidence': 0.5, 'distance': distance}
            
        except Exception as e:
            print(f"双手交互检测错误: {e}")
            return {'type': 'none', 'confidence': 0.0}
    
    def draw_annotations(self, image: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """绘制手势识别结果"""
        annotated_image = image.copy()
        
        try:
            for hand_data in results['hands_data']:
                # 绘制手部关键点
                landmarks = hand_data['landmarks']
                if len(landmarks) > 0 and MEDIAPIPE_AVAILABLE:
                    # 转换为MediaPipe格式
                    mp_landmarks = self.mp_hands.HandLandmarks()
                    for i, (x, y, z) in enumerate(landmarks):
                        mp_landmarks.landmark.add(x=x, y=y, z=z)
                    
                    # 绘制关键点和连接线
                    self.mp_drawing.draw_landmarks(
                        annotated_image,
                        mp_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                # 绘制边界框
                bbox = hand_data['bbox']
                cv2.rectangle(annotated_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                # 绘制手势标签
                static_gesture = hand_data['static_gesture']['gesture']
                confidence = hand_data['static_gesture']['confidence']
                handedness = hand_data['handedness']
                
                # 修复镜像显示问题：在镜像视频中左右手标签需要反转
                display_handedness = 'Left' if handedness == 'Right' else 'Right'
                
                label = f"{display_handedness}: {static_gesture} ({confidence:.2f})"
                cv2.putText(annotated_image, label, (bbox[0], bbox[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 绘制动态手势
                dynamic_gesture = hand_data['dynamic_gesture']['gesture']
                if dynamic_gesture != 'none':
                    dynamic_confidence = hand_data['dynamic_gesture']['confidence']
                    dynamic_label = f"Dynamic: {dynamic_gesture} ({dynamic_confidence:.2f})"
                    cv2.putText(annotated_image, dynamic_label, (bbox[0], bbox[3] + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # 绘制交互信息
            interaction = results['interaction']
            if interaction['type'] != 'none':
                interaction_label = f"Interaction: {interaction['type']} ({interaction['confidence']:.2f})"
                cv2.putText(annotated_image, interaction_label, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        except Exception as e:
            print(f"绘制标注错误: {e}")
        
        return annotated_image
    
    def _get_default_result(self) -> Dict[str, Any]:
        """获取默认识别结果"""
        return {
            'hands_detected': 0,
            'hands_data': [],
            'interaction': {'type': 'none', 'confidence': 0.0},
            'timestamp': time.time()
        }
    
    def get_recognizer_info(self) -> Dict[str, Any]:
        """获取识别器信息"""
        return {
            'name': 'Enhanced Gesture Recognizer',
            'version': '1.0.0',
            'description': '基于MediaPipe和深度学习的高精度手势识别',
            'static_gestures': list(self.static_gestures.keys()),
            'dynamic_gestures': list(self.dynamic_gestures.keys()),
            'max_hands': self.max_num_hands,
            'sequence_length': self.sequence_length,
            'mediapipe_enabled': MEDIAPIPE_AVAILABLE
        }
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'hands') and self.hands:
            self.hands.close()