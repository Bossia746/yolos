"""身体姿势识别模块 - 基于MediaPipe的人体姿态估计和动作分析"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Tuple, Optional, Any
import math


class PoseRecognizer:
    """身体姿势识别器"""
    
    def __init__(self, 
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5,
                 model_complexity: int = 1):
        """
        初始化身体姿势识别器
        
        Args:
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
            model_complexity: 模型复杂度 (0, 1, 2)
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # 姿势类型定义
        self.pose_names = {
            'standing': '站立',
            'sitting': '坐着',
            'lying': '躺着',
            'walking': '行走',
            'running': '跑步',
            'jumping': '跳跃',
            'waving': '挥手',
            'clapping': '鼓掌',
            'arms_up': '举手',
            'arms_crossed': '抱臂',
            'leaning_left': '左倾',
            'leaning_right': '右倾',
            'unknown': '未知姿势'
        }
        
        # 关键点索引定义
        self.pose_landmarks = {
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
    
    def detect_pose(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        检测人体姿态并分析姿势
        
        Args:
            image: 输入图像
            
        Returns:
            annotated_image: 标注后的图像
            result: 检测结果
        """
        # 转换颜色空间
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        
        annotated_image = image.copy()
        detection_result = {
            'pose_detected': False,
            'pose_type': 'unknown',
            'pose_name': '未知姿势',
            'landmarks': None,
            'angles': {},
            'body_ratios': {},
            'confidence': 0.0
        }
        
        if results.pose_landmarks:
            # 提取关键点坐标
            landmarks = self._extract_landmarks(results.pose_landmarks, image.shape)
            
            # 分析姿势
            pose_type = self._classify_pose(landmarks)
            
            # 计算关节角度
            angles = self._calculate_joint_angles(landmarks)
            
            # 计算身体比例
            body_ratios = self._calculate_body_ratios(landmarks)
            
            # 绘制姿态关键点
            self.mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # 添加姿势标签
            self._draw_pose_label(annotated_image, landmarks, pose_type)
            
            # 更新检测结果
            detection_result.update({
                'pose_detected': True,
                'pose_type': pose_type,
                'pose_name': self.pose_names.get(pose_type, '未知'),
                'landmarks': landmarks,
                'angles': angles,
                'body_ratios': body_ratios,
                'confidence': self._calculate_pose_confidence(results.pose_landmarks)
            })
        
        return annotated_image, detection_result
    
    def _extract_landmarks(self, pose_landmarks, image_shape: Tuple[int, int, int]) -> List[Tuple[float, float, float]]:
        """提取姿态关键点坐标"""
        h, w = image_shape[:2]
        landmarks = []
        
        for landmark in pose_landmarks.landmark:
            x = landmark.x * w
            y = landmark.y * h
            z = landmark.z  # 深度信息
            visibility = landmark.visibility  # 可见性
            landmarks.append((x, y, z, visibility))
        
        return landmarks
    
    def _classify_pose(self, landmarks: List[Tuple[float, float, float, float]]) -> str:
        """基于关键点坐标分类姿势"""
        if len(landmarks) != 33:
            return 'unknown'
        
        try:
            # 获取关键关节点
            left_shoulder = landmarks[self.pose_landmarks['left_shoulder']]
            right_shoulder = landmarks[self.pose_landmarks['right_shoulder']]
            left_hip = landmarks[self.pose_landmarks['left_hip']]
            right_hip = landmarks[self.pose_landmarks['right_hip']]
            left_knee = landmarks[self.pose_landmarks['left_knee']]
            right_knee = landmarks[self.pose_landmarks['right_knee']]
            left_ankle = landmarks[self.pose_landmarks['left_ankle']]
            right_ankle = landmarks[self.pose_landmarks['right_ankle']]
            left_wrist = landmarks[self.pose_landmarks['left_wrist']]
            right_wrist = landmarks[self.pose_landmarks['right_wrist']]
            nose = landmarks[self.pose_landmarks['nose']]
            
            # 计算身体中心线
            shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
            hip_center_y = (left_hip[1] + right_hip[1]) / 2
            knee_center_y = (left_knee[1] + right_knee[1]) / 2
            ankle_center_y = (left_ankle[1] + right_ankle[1]) / 2
            
            # 判断站立姿势
            if self._is_standing(landmarks):
                # 进一步判断具体动作
                if self._is_arms_up(landmarks):
                    return 'arms_up'
                elif self._is_waving(landmarks):
                    return 'waving'
                elif self._is_clapping(landmarks):
                    return 'clapping'
                elif self._is_arms_crossed(landmarks):
                    return 'arms_crossed'
                else:
                    return 'standing'
            
            # 判断坐着姿势
            elif self._is_sitting(landmarks):
                return 'sitting'
            
            # 判断躺着姿势
            elif self._is_lying(landmarks):
                return 'lying'
            
            # 判断倾斜姿势
            elif self._is_leaning(landmarks):
                if self._is_leaning_left(landmarks):
                    return 'leaning_left'
                else:
                    return 'leaning_right'
            
            else:
                return 'unknown'
        
        except Exception as e:
            print(f"姿势分类错误: {e}")
            return 'unknown'
    
    def _is_standing(self, landmarks: List[Tuple[float, float, float, float]]) -> bool:
        """判断是否为站立姿势"""
        try:
            left_hip = landmarks[self.pose_landmarks['left_hip']]
            right_hip = landmarks[self.pose_landmarks['right_hip']]
            left_knee = landmarks[self.pose_landmarks['left_knee']]
            right_knee = landmarks[self.pose_landmarks['right_knee']]
            left_ankle = landmarks[self.pose_landmarks['left_ankle']]
            right_ankle = landmarks[self.pose_landmarks['right_ankle']]
            
            # 检查腿部是否基本垂直
            hip_knee_angle_left = self._calculate_angle(
                (left_hip[0], left_hip[1] - 50),  # 假设的上方点
                left_hip[:2],
                left_knee[:2]
            )
            
            knee_ankle_angle_left = self._calculate_angle(
                left_hip[:2],
                left_knee[:2],
                left_ankle[:2]
            )
            
            # 站立时腿部角度应该接近180度
            return (160 < hip_knee_angle_left < 200) and (160 < knee_ankle_angle_left < 200)
        
        except:
            return False
    
    def _is_sitting(self, landmarks: List[Tuple[float, float, float, float]]) -> bool:
        """判断是否为坐着姿势"""
        try:
            left_hip = landmarks[self.pose_landmarks['left_hip']]
            right_hip = landmarks[self.pose_landmarks['right_hip']]
            left_knee = landmarks[self.pose_landmarks['left_knee']]
            right_knee = landmarks[self.pose_landmarks['right_knee']]
            
            # 坐着时膝盖通常高于或接近臀部
            hip_y = (left_hip[1] + right_hip[1]) / 2
            knee_y = (left_knee[1] + right_knee[1]) / 2
            
            return knee_y <= hip_y + 50  # 允许一定误差
        
        except:
            return False
    
    def _is_lying(self, landmarks: List[Tuple[float, float, float, float]]) -> bool:
        """判断是否为躺着姿势"""
        try:
            nose = landmarks[self.pose_landmarks['nose']]
            left_shoulder = landmarks[self.pose_landmarks['left_shoulder']]
            right_shoulder = landmarks[self.pose_landmarks['right_shoulder']]
            left_hip = landmarks[self.pose_landmarks['left_hip']]
            right_hip = landmarks[self.pose_landmarks['right_hip']]
            
            # 躺着时身体各部位的y坐标差异较小
            y_coords = [nose[1], left_shoulder[1], right_shoulder[1], left_hip[1], right_hip[1]]
            y_range = max(y_coords) - min(y_coords)
            
            return y_range < 100  # 身体基本水平
        
        except:
            return False
    
    def _is_arms_up(self, landmarks: List[Tuple[float, float, float, float]]) -> bool:
        """判断是否为举手姿势"""
        try:
            left_wrist = landmarks[self.pose_landmarks['left_wrist']]
            right_wrist = landmarks[self.pose_landmarks['right_wrist']]
            left_shoulder = landmarks[self.pose_landmarks['left_shoulder']]
            right_shoulder = landmarks[self.pose_landmarks['right_shoulder']]
            
            # 手腕高于肩膀
            return (left_wrist[1] < left_shoulder[1] - 50) or (right_wrist[1] < right_shoulder[1] - 50)
        
        except:
            return False
    
    def _is_waving(self, landmarks: List[Tuple[float, float, float, float]]) -> bool:
        """判断是否为挥手姿势"""
        try:
            left_wrist = landmarks[self.pose_landmarks['left_wrist']]
            right_wrist = landmarks[self.pose_landmarks['right_wrist']]
            left_elbow = landmarks[self.pose_landmarks['left_elbow']]
            right_elbow = landmarks[self.pose_landmarks['right_elbow']]
            left_shoulder = landmarks[self.pose_landmarks['left_shoulder']]
            right_shoulder = landmarks[self.pose_landmarks['right_shoulder']]
            
            # 一只手举起且肘部弯曲
            left_arm_up = left_wrist[1] < left_shoulder[1] and left_wrist[1] < left_elbow[1]
            right_arm_up = right_wrist[1] < right_shoulder[1] and right_wrist[1] < right_elbow[1]
            
            return left_arm_up or right_arm_up
        
        except:
            return False
    
    def _is_clapping(self, landmarks: List[Tuple[float, float, float, float]]) -> bool:
        """判断是否为鼓掌姿势"""
        try:
            left_wrist = landmarks[self.pose_landmarks['left_wrist']]
            right_wrist = landmarks[self.pose_landmarks['right_wrist']]
            
            # 双手距离很近
            distance = math.sqrt((left_wrist[0] - right_wrist[0])**2 + (left_wrist[1] - right_wrist[1])**2)
            return distance < 100
        
        except:
            return False
    
    def _is_arms_crossed(self, landmarks: List[Tuple[float, float, float, float]]) -> bool:
        """判断是否为抱臂姿势"""
        try:
            left_wrist = landmarks[self.pose_landmarks['left_wrist']]
            right_wrist = landmarks[self.pose_landmarks['right_wrist']]
            left_shoulder = landmarks[self.pose_landmarks['left_shoulder']]
            right_shoulder = landmarks[self.pose_landmarks['right_shoulder']]
            
            # 左手在右侧，右手在左侧
            left_hand_right = left_wrist[0] > (left_shoulder[0] + right_shoulder[0]) / 2
            right_hand_left = right_wrist[0] < (left_shoulder[0] + right_shoulder[0]) / 2
            
            return left_hand_right and right_hand_left
        
        except:
            return False
    
    def _is_leaning(self, landmarks: List[Tuple[float, float, float, float]]) -> bool:
        """判断是否为倾斜姿势"""
        try:
            left_shoulder = landmarks[self.pose_landmarks['left_shoulder']]
            right_shoulder = landmarks[self.pose_landmarks['right_shoulder']]
            
            # 计算肩膀倾斜角度
            shoulder_angle = math.atan2(
                right_shoulder[1] - left_shoulder[1],
                right_shoulder[0] - left_shoulder[0]
            ) * 180 / math.pi
            
            return abs(shoulder_angle) > 15  # 倾斜超过15度
        
        except:
            return False
    
    def _is_leaning_left(self, landmarks: List[Tuple[float, float, float, float]]) -> bool:
        """判断是否为左倾"""
        try:
            left_shoulder = landmarks[self.pose_landmarks['left_shoulder']]
            right_shoulder = landmarks[self.pose_landmarks['right_shoulder']]
            
            return left_shoulder[1] > right_shoulder[1]  # 左肩低于右肩
        
        except:
            return False
    
    def _calculate_angle(self, point1: Tuple[float, float], 
                        point2: Tuple[float, float], 
                        point3: Tuple[float, float]) -> float:
        """计算三点之间的角度"""
        try:
            # 计算向量
            vector1 = (point1[0] - point2[0], point1[1] - point2[1])
            vector2 = (point3[0] - point2[0], point3[1] - point2[1])
            
            # 计算角度
            dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
            magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
            magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0
            
            cos_angle = dot_product / (magnitude1 * magnitude2)
            cos_angle = max(-1, min(1, cos_angle))  # 限制在[-1, 1]范围内
            
            angle = math.acos(cos_angle) * 180 / math.pi
            return angle
        
        except:
            return 0
    
    def _calculate_joint_angles(self, landmarks: List[Tuple[float, float, float, float]]) -> Dict[str, float]:
        """计算关节角度"""
        angles = {}
        
        try:
            # 左肘角度
            angles['left_elbow'] = self._calculate_angle(
                landmarks[self.pose_landmarks['left_shoulder']][:2],
                landmarks[self.pose_landmarks['left_elbow']][:2],
                landmarks[self.pose_landmarks['left_wrist']][:2]
            )
            
            # 右肘角度
            angles['right_elbow'] = self._calculate_angle(
                landmarks[self.pose_landmarks['right_shoulder']][:2],
                landmarks[self.pose_landmarks['right_elbow']][:2],
                landmarks[self.pose_landmarks['right_wrist']][:2]
            )
            
            # 左膝角度
            angles['left_knee'] = self._calculate_angle(
                landmarks[self.pose_landmarks['left_hip']][:2],
                landmarks[self.pose_landmarks['left_knee']][:2],
                landmarks[self.pose_landmarks['left_ankle']][:2]
            )
            
            # 右膝角度
            angles['right_knee'] = self._calculate_angle(
                landmarks[self.pose_landmarks['right_hip']][:2],
                landmarks[self.pose_landmarks['right_knee']][:2],
                landmarks[self.pose_landmarks['right_ankle']][:2]
            )
        
        except Exception as e:
            print(f"计算关节角度错误: {e}")
        
        return angles
    
    def _calculate_body_ratios(self, landmarks: List[Tuple[float, float, float, float]]) -> Dict[str, float]:
        """计算身体比例"""
        ratios = {}
        
        try:
            # 肩宽
            left_shoulder = landmarks[self.pose_landmarks['left_shoulder']]
            right_shoulder = landmarks[self.pose_landmarks['right_shoulder']]
            shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
            
            # 身高（鼻子到脚踝的距离）
            nose = landmarks[self.pose_landmarks['nose']]
            left_ankle = landmarks[self.pose_landmarks['left_ankle']]
            right_ankle = landmarks[self.pose_landmarks['right_ankle']]
            ankle_center_y = (left_ankle[1] + right_ankle[1]) / 2
            body_height = abs(ankle_center_y - nose[1])
            
            if body_height > 0:
                ratios['shoulder_to_height'] = shoulder_width / body_height
            
            # 臂展比例
            left_wrist = landmarks[self.pose_landmarks['left_wrist']]
            right_wrist = landmarks[self.pose_landmarks['right_wrist']]
            arm_span = abs(right_wrist[0] - left_wrist[0])
            
            if body_height > 0:
                ratios['arm_span_to_height'] = arm_span / body_height
        
        except Exception as e:
            print(f"计算身体比例错误: {e}")
        
        return ratios
    
    def _calculate_pose_confidence(self, pose_landmarks) -> float:
        """计算姿态检测置信度"""
        try:
            visibilities = [landmark.visibility for landmark in pose_landmarks.landmark]
            return sum(visibilities) / len(visibilities)
        except:
            return 0.0
    
    def _draw_pose_label(self, image: np.ndarray, landmarks: List[Tuple[float, float, float, float]], 
                        pose_type: str):
        """在图像上绘制姿势标签"""
        try:
            # 获取头部位置
            nose = landmarks[self.pose_landmarks['nose']]
            
            # 准备标签文本
            pose_name = self.pose_names.get(pose_type, '未知')
            label = f"姿势: {pose_name}"
            
            # 绘制标签背景
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(image, 
                         (int(nose[0]) - text_width//2 - 5, int(nose[1]) - text_height - 30),
                         (int(nose[0]) + text_width//2 + 5, int(nose[1]) - 10),
                         (0, 0, 0), -1)
            
            # 绘制标签文本
            cv2.putText(image, label, 
                       (int(nose[0]) - text_width//2, int(nose[1]) - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        except Exception as e:
            print(f"绘制姿势标签错误: {e}")
    
    def get_pose_info(self) -> Dict[str, str]:
        """获取支持的姿势信息"""
        return self.pose_names.copy()
    
    def close(self):
        """释放资源"""
        if hasattr(self, 'pose'):
            self.pose.close()