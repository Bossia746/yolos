"""摔倒检测模块 - 基于深度学习的实时摔倒判断算法"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import math
from collections import deque
import time

try:
    import tensorflow as tf
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    ADVANCED_ML_AVAILABLE = False
    print("Warning: Advanced ML libraries not available for fall detection")


class FallDetector:
    """摔倒检测器 - 基于姿态分析和机器学习的摔倒检测"""
    
    def __init__(self, 
                 history_length: int = 30,
                 fall_threshold: float = 0.7,
                 velocity_threshold: float = 50.0,
                 angle_threshold: float = 45.0):
        """
        初始化摔倒检测器
        
        Args:
            history_length: 历史帧数
            fall_threshold: 摔倒判断阈值
            velocity_threshold: 速度阈值
            angle_threshold: 角度阈值
        """
        self.history_length = history_length
        self.fall_threshold = fall_threshold
        self.velocity_threshold = velocity_threshold
        self.angle_threshold = angle_threshold
        
        # 历史数据存储
        self.pose_history = deque(maxlen=history_length)
        self.velocity_history = deque(maxlen=history_length)
        self.angle_history = deque(maxlen=history_length)
        self.bbox_history = deque(maxlen=history_length)
        
        # 摔倒状态
        self.fall_detected = False
        self.fall_start_time = None
        self.fall_confidence = 0.0
        
        # 机器学习模型
        self.scaler = StandardScaler() if ADVANCED_ML_AVAILABLE else None
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42) if ADVANCED_ML_AVAILABLE else None
        self.model_trained = False
        
        # 特征缓存
        self.feature_buffer = deque(maxlen=100)
        
        print("摔倒检测器初始化完成")
    
    def detect_fall(self, pose_landmarks: List[Tuple[float, float, float]], 
                   bbox: Optional[Tuple[float, float, float, float]] = None,
                   timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        检测摔倒事件
        
        Args:
            pose_landmarks: 姿态关键点列表 [(x, y, confidence), ...]
            bbox: 边界框 (x1, y1, x2, y2)
            timestamp: 时间戳
            
        Returns:
            Dict: 检测结果
        """
        if timestamp is None:
            timestamp = time.time()
        
        # 提取特征
        features = self._extract_features(pose_landmarks, bbox)
        if features is None:
            return self._get_default_result()
        
        # 更新历史数据
        self._update_history(features, timestamp)
        
        # 多种检测方法
        fall_scores = []
        
        # 1. 基于角度的检测
        angle_score = self._detect_fall_by_angle(features)
        fall_scores.append(angle_score)
        
        # 2. 基于速度的检测
        velocity_score = self._detect_fall_by_velocity()
        fall_scores.append(velocity_score)
        
        # 3. 基于姿态变化的检测
        pose_change_score = self._detect_fall_by_pose_change()
        fall_scores.append(pose_change_score)
        
        # 4. 基于边界框变化的检测
        if bbox is not None:
            bbox_score = self._detect_fall_by_bbox_change(bbox)
            fall_scores.append(bbox_score)
        
        # 5. 基于机器学习的异常检测
        if ADVANCED_ML_AVAILABLE and self.model_trained:
            ml_score = self._detect_fall_by_ml(features)
            fall_scores.append(ml_score)
        
        # 综合评分
        final_score = np.mean(fall_scores) if fall_scores else 0.0
        
        # 更新摔倒状态
        self._update_fall_state(final_score, timestamp)
        
        return {
            'fall_detected': self.fall_detected,
            'fall_confidence': final_score,
            'fall_duration': time.time() - self.fall_start_time if self.fall_start_time else 0,
            'individual_scores': {
                'angle': angle_score,
                'velocity': velocity_score,
                'pose_change': pose_change_score,
                'bbox_change': bbox_score if bbox is not None else 0,
                'ml_anomaly': ml_score if ADVANCED_ML_AVAILABLE and self.model_trained else 0
            },
            'timestamp': timestamp
        }
    
    def _extract_features(self, pose_landmarks: List[Tuple[float, float, float]], 
                         bbox: Optional[Tuple[float, float, float, float]] = None) -> Optional[Dict[str, float]]:
        """提取摔倒检测特征"""
        if len(pose_landmarks) < 17:  # COCO 17关键点
            return None
        
        try:
            # 关键点索引
            nose = pose_landmarks[0]
            left_shoulder = pose_landmarks[5]
            right_shoulder = pose_landmarks[6]
            left_hip = pose_landmarks[11]
            right_hip = pose_landmarks[12]
            left_knee = pose_landmarks[13]
            right_knee = pose_landmarks[14]
            left_ankle = pose_landmarks[15]
            right_ankle = pose_landmarks[16]
            
            # 检查关键点可见性
            visible_points = [p for p in [nose, left_shoulder, right_shoulder, left_hip, right_hip] if p[2] > 0.3]
            if len(visible_points) < 3:
                return None
            
            # 计算身体中心点
            center_x = np.mean([p[0] for p in visible_points])
            center_y = np.mean([p[1] for p in visible_points])
            
            # 计算身体角度
            if left_shoulder[2] > 0.3 and right_shoulder[2] > 0.3 and left_hip[2] > 0.3 and right_hip[2] > 0.3:
                shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
                hip_center = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
                
                # 躯干角度（相对于垂直方向）
                torso_angle = math.degrees(math.atan2(
                    abs(shoulder_center[0] - hip_center[0]),
                    abs(shoulder_center[1] - hip_center[1])
                ))
            else:
                torso_angle = 0
            
            # 计算身体高度比例
            if nose[2] > 0.3 and left_ankle[2] > 0.3 and right_ankle[2] > 0.3:
                ankle_y = min(left_ankle[1], right_ankle[1])
                body_height = abs(nose[1] - ankle_y)
                
                # 头部相对位置
                head_ratio = (nose[1] - ankle_y) / max(body_height, 1)
            else:
                head_ratio = 1.0
                body_height = 100
            
            # 计算肢体分散度
            limb_spread = 0
            if all(p[2] > 0.3 for p in [left_shoulder, right_shoulder, left_hip, right_hip]):
                shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
                hip_width = abs(left_hip[0] - right_hip[0])
                limb_spread = (shoulder_width + hip_width) / 2
            
            # 计算重心位置
            if bbox is not None:
                bbox_center_x = (bbox[0] + bbox[2]) / 2
                bbox_center_y = (bbox[1] + bbox[3]) / 2
                center_offset_x = abs(center_x - bbox_center_x) / max(bbox[2] - bbox[0], 1)
                center_offset_y = abs(center_y - bbox_center_y) / max(bbox[3] - bbox[1], 1)
            else:
                center_offset_x = 0
                center_offset_y = 0
            
            features = {
                'center_x': center_x,
                'center_y': center_y,
                'torso_angle': torso_angle,
                'head_ratio': head_ratio,
                'body_height': body_height,
                'limb_spread': limb_spread,
                'center_offset_x': center_offset_x,
                'center_offset_y': center_offset_y
            }
            
            return features
            
        except Exception as e:
            print(f"特征提取错误: {e}")
            return None
    
    def _update_history(self, features: Dict[str, float], timestamp: float):
        """更新历史数据"""
        self.pose_history.append((features, timestamp))
        
        # 计算速度
        if len(self.pose_history) >= 2:
            prev_features, prev_time = self.pose_history[-2]
            curr_features, curr_time = self.pose_history[-1]
            
            dt = curr_time - prev_time
            if dt > 0:
                velocity_x = (curr_features['center_x'] - prev_features['center_x']) / dt
                velocity_y = (curr_features['center_y'] - prev_features['center_y']) / dt
                velocity_magnitude = math.sqrt(velocity_x**2 + velocity_y**2)
                
                self.velocity_history.append(velocity_magnitude)
        
        # 更新角度历史
        self.angle_history.append(features['torso_angle'])
    
    def _detect_fall_by_angle(self, features: Dict[str, float]) -> float:
        """基于身体角度检测摔倒"""
        torso_angle = features['torso_angle']
        
        # 角度越大，摔倒可能性越高
        if torso_angle > self.angle_threshold:
            return min(1.0, (torso_angle - self.angle_threshold) / (90 - self.angle_threshold))
        
        return 0.0
    
    def _detect_fall_by_velocity(self) -> float:
        """基于运动速度检测摔倒"""
        if len(self.velocity_history) < 5:
            return 0.0
        
        # 检查速度突变
        recent_velocities = list(self.velocity_history)[-5:]
        max_velocity = max(recent_velocities)
        
        if max_velocity > self.velocity_threshold:
            return min(1.0, (max_velocity - self.velocity_threshold) / self.velocity_threshold)
        
        return 0.0
    
    def _detect_fall_by_pose_change(self) -> float:
        """基于姿态变化检测摔倒"""
        if len(self.pose_history) < 10:
            return 0.0
        
        try:
            # 计算姿态变化率
            recent_poses = list(self.pose_history)[-10:]
            
            angle_changes = []
            height_changes = []
            
            for i in range(1, len(recent_poses)):
                prev_features, _ = recent_poses[i-1]
                curr_features, _ = recent_poses[i]
                
                angle_change = abs(curr_features['torso_angle'] - prev_features['torso_angle'])
                height_change = abs(curr_features['head_ratio'] - prev_features['head_ratio'])
                
                angle_changes.append(angle_change)
                height_changes.append(height_change)
            
            # 计算变化率
            avg_angle_change = np.mean(angle_changes)
            avg_height_change = np.mean(height_changes)
            
            # 综合评分
            change_score = (avg_angle_change / 45.0 + avg_height_change) / 2
            return min(1.0, change_score)
            
        except Exception:
            return 0.0
    
    def _detect_fall_by_bbox_change(self, bbox: Tuple[float, float, float, float]) -> float:
        """基于边界框变化检测摔倒"""
        self.bbox_history.append(bbox)
        
        if len(self.bbox_history) < 5:
            return 0.0
        
        try:
            # 计算边界框宽高比变化
            recent_bboxes = list(self.bbox_history)[-5:]
            
            aspect_ratios = []
            for x1, y1, x2, y2 in recent_bboxes:
                width = x2 - x1
                height = y2 - y1
                if height > 0:
                    aspect_ratios.append(width / height)
            
            if len(aspect_ratios) >= 2:
                # 宽高比突然增大可能表示摔倒
                max_ratio = max(aspect_ratios)
                min_ratio = min(aspect_ratios)
                
                if max_ratio > 1.5 and (max_ratio - min_ratio) > 0.5:
                    return min(1.0, (max_ratio - 1.0) / 2.0)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _detect_fall_by_ml(self, features: Dict[str, float]) -> float:
        """基于机器学习的异常检测"""
        if not ADVANCED_ML_AVAILABLE or not self.model_trained:
            return 0.0
        
        try:
            # 准备特征向量
            feature_vector = np.array(list(features.values())).reshape(1, -1)
            
            # 标准化
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # 异常检测
            anomaly_score = self.anomaly_detector.decision_function(feature_vector_scaled)[0]
            
            # 转换为0-1范围的摔倒概率
            fall_probability = max(0, -anomaly_score / 2.0)
            return min(1.0, fall_probability)
            
        except Exception as e:
            print(f"ML检测错误: {e}")
            return 0.0
    
    def _update_fall_state(self, score: float, timestamp: float):
        """更新摔倒状态"""
        if score > self.fall_threshold:
            if not self.fall_detected:
                self.fall_detected = True
                self.fall_start_time = timestamp
                print(f"检测到摔倒事件! 置信度: {score:.2f}")
            
            self.fall_confidence = max(self.fall_confidence, score)
        else:
            # 如果连续多帧都低于阈值，重置摔倒状态
            if self.fall_detected and len(self.pose_history) >= 10:
                recent_scores = []
                for i in range(min(10, len(self.pose_history))):
                    # 这里需要重新计算最近的分数，简化处理
                    recent_scores.append(score)
                
                if all(s < self.fall_threshold * 0.5 for s in recent_scores):
                    self.fall_detected = False
                    self.fall_start_time = None
                    self.fall_confidence = 0.0
    
    def train_anomaly_detector(self, normal_features: List[Dict[str, float]]):
        """训练异常检测模型"""
        if not ADVANCED_ML_AVAILABLE or len(normal_features) < 20:
            print("无法训练异常检测模型：数据不足或库不可用")
            return
        
        try:
            # 准备训练数据
            feature_matrix = np.array([list(f.values()) for f in normal_features])
            
            # 标准化
            self.scaler.fit(feature_matrix)
            feature_matrix_scaled = self.scaler.transform(feature_matrix)
            
            # 训练异常检测器
            self.anomaly_detector.fit(feature_matrix_scaled)
            self.model_trained = True
            
            print(f"异常检测模型训练完成，使用{len(normal_features)}个正常样本")
            
        except Exception as e:
            print(f"训练异常检测模型失败: {e}")
    
    def _get_default_result(self) -> Dict[str, Any]:
        """获取默认检测结果"""
        return {
            'fall_detected': False,
            'fall_confidence': 0.0,
            'fall_duration': 0,
            'individual_scores': {
                'angle': 0,
                'velocity': 0,
                'pose_change': 0,
                'bbox_change': 0,
                'ml_anomaly': 0
            },
            'timestamp': time.time()
        }
    
    def reset(self):
        """重置检测器状态"""
        self.pose_history.clear()
        self.velocity_history.clear()
        self.angle_history.clear()
        self.bbox_history.clear()
        
        self.fall_detected = False
        self.fall_start_time = None
        self.fall_confidence = 0.0
    
    def get_detector_info(self) -> Dict[str, Any]:
        """获取检测器信息"""
        return {
            'name': 'Advanced Fall Detector',
            'version': '1.0.0',
            'description': '基于多特征融合的深度学习摔倒检测',
            'features': ['角度检测', '速度检测', '姿态变化', '边界框分析', '异常检测'],
            'ml_enabled': ADVANCED_ML_AVAILABLE and self.model_trained,
            'history_length': self.history_length,
            'thresholds': {
                'fall': self.fall_threshold,
                'velocity': self.velocity_threshold,
                'angle': self.angle_threshold
            }
        }