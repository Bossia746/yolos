"""YOLOv7 Pose身体姿势识别模块 - 基于YOLOv7的多人实时姿态估计"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Any
import math
import os
from ultralytics import YOLO


class YOLOv7PoseRecognizer:
    """YOLOv7身体姿势识别器 - 支持多人实时检测"""
    
    def __init__(self, 
                 model_path: str = None,
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.45,
                 device: str = 'auto'):
        """
        初始化YOLOv7姿势识别器
        
        Args:
            model_path: 模型路径
            confidence_threshold: 置信度阈值
            iou_threshold: IoU阈值
            device: 设备类型 ('auto', 'cpu', 'cuda')
        """
        try:
            if model_path is None:
                model_path = os.path.join(os.getcwd(), 'module', 'yolov8n-pose.pt')
            self.model = YOLO(model_path)
            self.confidence_threshold = confidence_threshold
            self.iou_threshold = iou_threshold
            self.device = device
            
            # COCO 17关键点定义
            self.keypoint_names = [
                'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
            ]
            
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
                'falling': '摔倒',
                'unknown': '未知'
            }
            
            # 骨架连接定义
            self.skeleton = [
                [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
                [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
                [2, 4], [3, 5], [4, 6], [5, 7]
            ]
            
            print(f"YOLOv7 Pose模型加载成功: {model_path}")
            
        except Exception as e:
            print(f"YOLOv7 Pose模型加载失败: {e}")
            self.model = None
    
    def detect_poses(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        检测图像中的多个人体姿势
        
        Args:
            image: 输入图像
            
        Returns:
            Tuple[np.ndarray, List[Dict]]: (标注图像, 检测结果列表)
        """
        if self.model is None:
            return image, []
        
        try:
            # 使用YOLOv7进行推理
            results = self.model(image, 
                               conf=self.confidence_threshold,
                               iou=self.iou_threshold,
                               verbose=False)
            
            annotated_image = image.copy()
            detections = []
            
            for result in results:
                if result.keypoints is not None:
                    keypoints = result.keypoints.data.cpu().numpy()
                    boxes = result.boxes.data.cpu().numpy() if result.boxes is not None else None
                    
                    for i, kpts in enumerate(keypoints):
                        # 提取关键点坐标和置信度
                        landmarks = self._extract_keypoints(kpts)
                        
                        if len(landmarks) > 0:
                            # 分类姿势
                            pose_type = self._classify_pose(landmarks)
                            
                            # 计算边界框
                            bbox = self._calculate_bbox_from_keypoints(landmarks) if boxes is None else boxes[i][:4]
                            
                            # 绘制姿势
                            annotated_image = self._draw_pose(annotated_image, landmarks, pose_type)
                            
                            # 添加检测结果
                            detection = {
                                'person_id': i,
                                'pose_type': pose_type,
                                'pose_name': self.pose_names.get(pose_type, '未知'),
                                'landmarks': landmarks,
                                'bbox': bbox.tolist() if isinstance(bbox, np.ndarray) else bbox,
                                'confidence': np.mean([lm[2] for lm in landmarks if lm[2] > 0])
                            }
                            detections.append(detection)
            
            return annotated_image, detections
            
        except Exception as e:
            print(f"姿势检测错误: {e}")
            return image, []
    
    def _extract_keypoints(self, keypoints: np.ndarray) -> List[Tuple[float, float, float]]:
        """
        提取关键点坐标和置信度
        
        Args:
            keypoints: YOLOv7输出的关键点数据
            
        Returns:
            List[Tuple[float, float, float]]: [(x, y, confidence), ...]
        """
        landmarks = []
        
        for i in range(len(keypoints)):
            if len(keypoints[i]) >= 3:
                x, y, conf = keypoints[i][:3]
                # 确保所有值都是标量
                x_val = float(x.item()) if hasattr(x, 'item') else float(x)
                y_val = float(y.item()) if hasattr(y, 'item') else float(y)
                conf_val = float(conf.item()) if hasattr(conf, 'item') else float(conf)
                landmarks.append((x_val, y_val, conf_val))
            else:
                landmarks.append((0.0, 0.0, 0.0))
        
        return landmarks
    
    def _classify_pose(self, landmarks: List[Tuple[float, float, float]]) -> str:
        """
        基于关键点分类姿势
        
        Args:
            landmarks: 关键点列表
            
        Returns:
            str: 姿势类型
        """
        if len(landmarks) < 17:
            return 'unknown'
        
        try:
            # 检查摔倒
            if self._is_falling(landmarks):
                return 'falling'
            
            # 检查躺着
            if self._is_lying(landmarks):
                return 'lying'
            
            # 检查坐着
            if self._is_sitting(landmarks):
                return 'sitting'
            
            # 检查举手
            if self._is_arms_up(landmarks):
                return 'arms_up'
            
            # 检查挥手
            if self._is_waving(landmarks):
                return 'waving'
            
            # 检查鼓掌
            if self._is_clapping(landmarks):
                return 'clapping'
            
            # 检查倾斜
            if self._is_leaning_left(landmarks):
                return 'leaning_left'
            elif self._is_leaning_right(landmarks):
                return 'leaning_right'
            
            # 默认站立
            return 'standing'
            
        except Exception as e:
            print(f"姿势分类错误: {e}")
            return 'unknown'
    
    def _is_falling(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        """检测摔倒姿势"""
        try:
            # 获取关键点
            nose = landmarks[0]
            left_shoulder = landmarks[5]
            right_shoulder = landmarks[6]
            left_hip = landmarks[11]
            right_hip = landmarks[12]
            left_ankle = landmarks[15]
            right_ankle = landmarks[16]
            
            # 检查关键点是否可见
            visible_points = [p for p in [nose, left_shoulder, right_shoulder, left_hip, right_hip] if p[2] > 0.3]
            if len(visible_points) < 3:
                return False
            
            # 计算身体中心点
            center_y = np.mean([p[1] for p in visible_points])
            
            # 计算头部和臀部的相对位置
            if nose[2] > 0.3 and left_hip[2] > 0.3 and right_hip[2] > 0.3:
                hip_center_y = (left_hip[1] + right_hip[1]) / 2
                
                # 如果头部低于臀部，可能是摔倒
                if nose[1] > hip_center_y:
                    return True
            
            # 检查身体水平程度
            if left_shoulder[2] > 0.3 and right_shoulder[2] > 0.3:
                shoulder_angle = abs(left_shoulder[1] - right_shoulder[1]) / max(abs(left_shoulder[0] - right_shoulder[0]), 1)
                if shoulder_angle < 0.3:  # 肩膀接近水平
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _is_lying(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        """检测躺着姿势"""
        try:
            left_shoulder = landmarks[5]
            right_shoulder = landmarks[6]
            left_hip = landmarks[11]
            right_hip = landmarks[12]
            
            if all(p[2] > 0.3 for p in [left_shoulder, right_shoulder, left_hip, right_hip]):
                # 计算躯干角度
                shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
                hip_center = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
                
                # 如果躯干接近水平，认为是躺着
                torso_angle = math.degrees(math.atan2(abs(shoulder_center[1] - hip_center[1]), 
                                                    abs(shoulder_center[0] - hip_center[0])))
                return torso_angle < 30
            
            return False
        except Exception:
            return False
    
    def _is_sitting(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        """检测坐着姿势"""
        try:
            left_hip = landmarks[11]
            right_hip = landmarks[12]
            left_knee = landmarks[13]
            right_knee = landmarks[14]
            
            if all(p[2] > 0.3 for p in [left_hip, right_hip, left_knee, right_knee]):
                # 检查膝盖是否弯曲且在臀部上方
                hip_y = (left_hip[1] + right_hip[1]) / 2
                knee_y = (left_knee[1] + right_knee[1]) / 2
                
                return knee_y > hip_y and (knee_y - hip_y) > 50
            
            return False
        except Exception:
            return False
    
    def _is_arms_up(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        """检测举手姿势"""
        try:
            left_shoulder = landmarks[5]
            right_shoulder = landmarks[6]
            left_wrist = landmarks[9]
            right_wrist = landmarks[10]
            
            if all(p[2] > 0.3 for p in [left_shoulder, right_shoulder, left_wrist, right_wrist]):
                # 检查手腕是否在肩膀上方
                left_up = left_wrist[1] < left_shoulder[1] - 30
                right_up = right_wrist[1] < right_shoulder[1] - 30
                
                return left_up or right_up
            
            return False
        except Exception:
            return False
    
    def _is_waving(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        """检测挥手姿势"""
        try:
            left_shoulder = landmarks[5]
            right_shoulder = landmarks[6]
            left_elbow = landmarks[7]
            right_elbow = landmarks[8]
            left_wrist = landmarks[9]
            right_wrist = landmarks[10]
            
            # 检查右手挥手
            if all(p[2] > 0.3 for p in [right_shoulder, right_elbow, right_wrist]):
                if (right_wrist[1] < right_shoulder[1] and 
                    right_elbow[1] < right_shoulder[1] and
                    abs(right_wrist[0] - right_shoulder[0]) > 50):
                    return True
            
            # 检查左手挥手
            if all(p[2] > 0.3 for p in [left_shoulder, left_elbow, left_wrist]):
                if (left_wrist[1] < left_shoulder[1] and 
                    left_elbow[1] < left_shoulder[1] and
                    abs(left_wrist[0] - left_shoulder[0]) > 50):
                    return True
            
            return False
        except Exception:
            return False
    
    def _is_clapping(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        """检测鼓掌姿势"""
        try:
            left_wrist = landmarks[9]
            right_wrist = landmarks[10]
            
            if left_wrist[2] > 0.3 and right_wrist[2] > 0.3:
                # 检查双手是否靠近
                distance = math.sqrt((left_wrist[0] - right_wrist[0])**2 + 
                                   (left_wrist[1] - right_wrist[1])**2)
                return distance < 100
            
            return False
        except Exception:
            return False
    
    def _is_leaning_left(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        """检测左倾姿势"""
        try:
            left_shoulder = landmarks[5]
            right_shoulder = landmarks[6]
            
            if left_shoulder[2] > 0.3 and right_shoulder[2] > 0.3:
                # 计算肩膀倾斜角度
                angle = math.degrees(math.atan2(right_shoulder[1] - left_shoulder[1], 
                                              right_shoulder[0] - left_shoulder[0]))
                return angle > 15
            
            return False
        except Exception:
            return False
    
    def _is_leaning_right(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        """检测右倾姿势"""
        try:
            left_shoulder = landmarks[5]
            right_shoulder = landmarks[6]
            
            if left_shoulder[2] > 0.3 and right_shoulder[2] > 0.3:
                # 计算肩膀倾斜角度
                angle = math.degrees(math.atan2(right_shoulder[1] - left_shoulder[1], 
                                              right_shoulder[0] - left_shoulder[0]))
                return angle < -15
            
            return False
        except Exception:
            return False
    
    def _calculate_bbox_from_keypoints(self, landmarks: List[Tuple[float, float, float]]) -> List[float]:
        """从关键点计算边界框"""
        visible_points = [(x, y) for x, y, conf in landmarks if conf > 0.3]
        
        if len(visible_points) == 0:
            return [0, 0, 100, 100]
        
        xs, ys = zip(*visible_points)
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # 添加边距
        margin = 20
        return [x_min - margin, y_min - margin, x_max + margin, y_max + margin]
    
    def _draw_pose(self, image: np.ndarray, landmarks: List[Tuple[float, float, float]], pose_type: str) -> np.ndarray:
        """绘制姿势关键点和骨架"""
        try:
            # 绘制骨架连接
            for connection in self.skeleton:
                start_idx, end_idx = connection[0] - 1, connection[1] - 1
                if (0 <= start_idx < len(landmarks) and 0 <= end_idx < len(landmarks) and
                    landmarks[start_idx][2] > 0.3 and landmarks[end_idx][2] > 0.3):
                    
                    start_point = (int(landmarks[start_idx][0]), int(landmarks[start_idx][1]))
                    end_point = (int(landmarks[end_idx][0]), int(landmarks[end_idx][1]))
                    
                    cv2.line(image, start_point, end_point, (0, 255, 0), 2)
            
            # 绘制关键点
            for i, (x, y, conf) in enumerate(landmarks):
                if conf > 0.3:
                    cv2.circle(image, (int(x), int(y)), 4, (0, 0, 255), -1)
            
            # 绘制姿势标签
            if len(landmarks) > 0 and landmarks[0][2] > 0.3:  # 使用鼻子位置
                label_pos = (int(landmarks[0][0]), int(landmarks[0][1]) - 20)
                pose_name = self.pose_names.get(pose_type, '未知')
                cv2.putText(image, pose_name, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2)
            
            return image
            
        except Exception as e:
            print(f"绘制姿势错误: {e}")
            return image
    
    def get_pose_info(self) -> Dict[str, str]:
        """获取姿势识别器信息"""
        return {
            'name': 'YOLOv7 Pose Recognizer',
            'version': '1.0.0',
            'description': '基于YOLOv7的多人实时姿态估计',
            'keypoints': '17个COCO关键点',
            'features': '多人检测, 实时处理, 摔倒检测'
        }
    
    def close(self):
        """关闭识别器"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None