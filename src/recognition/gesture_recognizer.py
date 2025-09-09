"""手势识别模块 - 基于MediaPipe的手部关键点检测和手势分类"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Tuple, Optional, Any
import math


class GestureRecognizer:
    """手势识别器"""
    
    def __init__(self, 
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5,
                 max_num_hands: int = 2):
        """
        初始化手势识别器
        
        Args:
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
            max_num_hands: 最大手部数量
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # 手势类型定义
        self.gesture_names = {
            'fist': '拳头',
            'open_palm': '张开手掌',
            'thumbs_up': '点赞',
            'thumbs_down': '点踩',
            'peace': '胜利手势',
            'ok': 'OK手势',
            'pointing': '指向',
            'rock': '摇滚手势',
            'call_me': '打电话手势',
            'unknown': '未知手势'
        }
    
    def detect_hands(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        检测手部并识别手势
        
        Args:
            image: 输入图像
            
        Returns:
            annotated_image: 标注后的图像
            results: 检测结果列表
        """
        # 转换颜色空间
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        annotated_image = image.copy()
        detection_results = []
        
        if results.multi_hand_landmarks:
            for idx, (hand_landmarks, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                # 获取手部信息
                hand_label = handedness.classification[0].label  # 'Left' or 'Right'
                hand_score = handedness.classification[0].score
                
                # 提取关键点坐标
                landmarks = self._extract_landmarks(hand_landmarks, image.shape)
                
                # 识别手势
                gesture = self._classify_gesture(landmarks)
                
                # 绘制手部关键点和连接线
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # 添加手势标签
                self._draw_gesture_label(annotated_image, landmarks, gesture, hand_label)
                
                # 保存检测结果
                detection_results.append({
                    'hand_id': idx,
                    'hand_label': hand_label,
                    'hand_score': hand_score,
                    'gesture': gesture,
                    'gesture_name': self.gesture_names.get(gesture, '未知'),
                    'landmarks': landmarks,
                    'bbox': self._get_hand_bbox(landmarks)
                })
        
        return annotated_image, detection_results
    
    def _extract_landmarks(self, hand_landmarks, image_shape: Tuple[int, int, int]) -> List[Tuple[int, int]]:
        """提取手部关键点坐标"""
        h, w = image_shape[:2]
        landmarks = []
        
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append((x, y))
        
        return landmarks
    
    def _classify_gesture(self, landmarks: List[Tuple[int, int]]) -> str:
        """基于关键点坐标分类手势"""
        if len(landmarks) != 21:
            return 'unknown'
        
        # 计算手指状态（伸直或弯曲）
        finger_states = self._get_finger_states(landmarks)
        
        # 根据手指状态判断手势
        if all(not state for state in finger_states):  # 所有手指都弯曲
            return 'fist'
        elif all(finger_states):  # 所有手指都伸直
            return 'open_palm'
        elif finger_states[0] and not any(finger_states[1:]):  # 只有拇指伸直
            # 判断拇指方向
            if self._is_thumb_up(landmarks):
                return 'thumbs_up'
            else:
                return 'thumbs_down'
        elif finger_states[1] and finger_states[2] and not finger_states[3] and not finger_states[4]:  # 食指和中指伸直
            return 'peace'
        elif finger_states[1] and not finger_states[2] and not finger_states[3] and not finger_states[4]:  # 只有食指伸直
            return 'pointing'
        elif self._is_ok_gesture(landmarks, finger_states):  # OK手势
            return 'ok'
        elif finger_states[1] and finger_states[4] and not finger_states[2] and not finger_states[3]:  # 食指和小指伸直
            return 'rock'
        elif finger_states[0] and finger_states[4] and not finger_states[1] and not finger_states[2] and not finger_states[3]:  # 拇指和小指伸直
            return 'call_me'
        else:
            return 'unknown'
    
    def _get_finger_states(self, landmarks: List[Tuple[int, int]]) -> List[bool]:
        """获取手指状态（True表示伸直，False表示弯曲）"""
        finger_tips = [4, 8, 12, 16, 20]  # 拇指、食指、中指、无名指、小指的指尖
        finger_pips = [3, 6, 10, 14, 18]  # 对应的PIP关节
        
        states = []
        
        for i, (tip, pip) in enumerate(zip(finger_tips, finger_pips)):
            if i == 0:  # 拇指特殊处理
                # 拇指的判断基于x坐标
                states.append(abs(landmarks[tip][0] - landmarks[pip][0]) > abs(landmarks[tip][1] - landmarks[pip][1]))
            else:
                # 其他手指基于y坐标
                states.append(landmarks[tip][1] < landmarks[pip][1])
        
        return states
    
    def _is_thumb_up(self, landmarks: List[Tuple[int, int]]) -> bool:
        """判断是否为点赞手势"""
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        return thumb_tip[1] < thumb_mcp[1]  # 拇指指尖在MCP关节上方
    
    def _is_ok_gesture(self, landmarks: List[Tuple[int, int]], finger_states: List[bool]) -> bool:
        """判断是否为OK手势"""
        # OK手势：拇指和食指形成圆圈，其他手指伸直
        if not (finger_states[2] and finger_states[3] and finger_states[4]):
            return False
        
        # 计算拇指指尖和食指指尖的距离
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        distance = math.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)
        
        # 如果距离小于阈值，认为是OK手势
        return distance < 40
    
    def _get_hand_bbox(self, landmarks: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        """获取手部边界框"""
        x_coords = [point[0] for point in landmarks]
        y_coords = [point[1] for point in landmarks]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # 添加一些边距
        margin = 20
        return (x_min - margin, y_min - margin, x_max + margin, y_max + margin)
    
    def _draw_gesture_label(self, image: np.ndarray, landmarks: List[Tuple[int, int]], 
                           gesture: str, hand_label: str):
        """在图像上绘制手势标签"""
        # 获取手部中心点
        center_x = sum(point[0] for point in landmarks) // len(landmarks)
        center_y = sum(point[1] for point in landmarks) // len(landmarks)
        
        # 准备标签文本 - 使用英文避免乱码
        gesture_name = self.gesture_names.get(gesture, 'unknown')
        label = f"{hand_label}: {gesture_name}"
        
        # 绘制背景矩形
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(image, 
                     (center_x - text_width//2 - 5, center_y - text_height - 10),
                     (center_x + text_width//2 + 5, center_y + 5),
                     (0, 0, 0), -1)
        
        # 绘制文本
        cv2.putText(image, label, 
                   (center_x - text_width//2, center_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def get_gesture_info(self) -> Dict[str, str]:
        """获取支持的手势信息"""
        return self.gesture_names.copy()
    
    def close(self):
        """释放资源"""
        if hasattr(self, 'hands'):
            self.hands.close()