#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的多模态检测器
集成预训练模型和深度学习特征，提升识别准确性
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json
import time
from dataclasses import dataclass

# 导入现有模块
from .enhanced_face_recognizer import EnhancedFaceRecognizer
from .enhanced_gesture_recognizer import EnhancedGestureRecognizer
from .enhanced_pose_recognizer import EnhancedPoseRecognizer
from .enhanced_fall_detector import EnhancedFallDetector
from ..models.pretrained_model_loader import PretrainedModelLoader, EnhancedFeatureExtractor

logger = logging.getLogger(__name__)

@dataclass
class RecognitionResult:
    """识别结果"""
    detection_type: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    features: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class MultimodalResult:
    """多模态识别结果"""
    faces: List[RecognitionResult]
    gestures: List[RecognitionResult]
    poses: List[RecognitionResult]
    actions: List[RecognitionResult]
    falls: List[RecognitionResult]
    timestamp: float
    frame_id: int
    confidence_scores: Dict[str, float]

class ActionRecognitionModule:
    """动作识别模块"""
    
    def __init__(self, model_loader: PretrainedModelLoader):
        self.model_loader = model_loader
        self.action_classifier = None
        self.feature_extractor = EnhancedFeatureExtractor(model_loader)
        
        # 动作类别定义
        self.action_classes = [
            'standing', 'walking', 'running', 'sitting', 'jumping',
            'waving', 'pointing', 'clapping', 'stretching', 'bending'
        ]
        
        # 加载预训练的动作分类器
        self.load_action_classifier()
        
        # 动作历史缓存
        self.action_history = []
        self.history_size = 10
    
    def load_action_classifier(self):
        """加载动作分类器"""
        try:
            # 尝试加载已训练的模型
            model_path = Path("models/human_recognition/best_model.pth")
            
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location='cpu')
                
                # 重建模型架构
                from ..training.enhanced_human_trainer import MultiModalHumanNet
                self.action_classifier = MultiModalHumanNet(
                    num_action_classes=len(self.action_classes),
                    num_gesture_classes=8
                )
                
                self.action_classifier.load_state_dict(checkpoint['model_state_dict'])
                self.action_classifier.eval()
                
                logger.info("加载训练好的动作分类器")
            else:
                # 使用预训练backbone创建分类器
                self.action_classifier = self.model_loader.create_action_classifier(
                    'resnet50_action',
                    num_action_classes=len(self.action_classes),
                    freeze_backbone=True
                )
                
                logger.info("使用预训练backbone创建动作分类器")
                
        except Exception as e:
            logger.warning(f"加载动作分类器失败: {e}")
            self.action_classifier = None
    
    def recognize_action(self, 
                        image: np.ndarray, 
                        pose_keypoints: Optional[np.ndarray] = None) -> List[RecognitionResult]:
        """识别动作"""
        results = []
        
        if self.action_classifier is None:
            return results
        
        try:
            # 提取多尺度特征
            multi_features = self.feature_extractor.extract_multi_scale_features(image)
            
            if not multi_features:
                return results
            
            # 融合特征
            fused_features = self.feature_extractor.fuse_features(multi_features)
            
            # 预处理图像用于模型推理
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0)
            
            # 处理姿势关键点
            if pose_keypoints is not None and len(pose_keypoints) > 0:
                pose_tensor = torch.from_numpy(pose_keypoints).float().unsqueeze(0)
            else:
                # 创建零填充的姿势数据
                pose_tensor = torch.zeros(1, 34)  # 17个关键点 * 2坐标
            
            # 模型推理
            with torch.no_grad():
                if hasattr(self.action_classifier, 'forward'):
                    # 自定义多模态模型
                    action_pred, gesture_pred = self.action_classifier(image_tensor, pose_tensor)
                    action_probs = torch.softmax(action_pred, dim=1)
                else:
                    # 预训练分类器
                    action_pred = self.action_classifier(image_tensor)
                    action_probs = torch.softmax(action_pred, dim=1)
            
            # 获取top-k预测
            top_k = min(3, len(self.action_classes))
            top_probs, top_indices = torch.topk(action_probs, top_k)
            
            for i in range(top_k):
                confidence = top_probs[0][i].item()
                action_idx = top_indices[0][i].item()
                
                if confidence > 0.3:  # 置信度阈值
                    action_name = self.action_classes[action_idx]
                    
                    result = RecognitionResult(
                        detection_type='action',
                        confidence=confidence,
                        bbox=(0, 0, image.shape[1], image.shape[0]),  # 全图
                        features=fused_features,
                        metadata={
                            'action_name': action_name,
                            'action_id': action_idx,
                            'feature_dim': len(fused_features),
                            'model_type': 'deep_learning'
                        }
                    )
                    results.append(result)
            
            # 更新动作历史
            if results:
                self.action_history.append(results[0].metadata['action_name'])
                if len(self.action_history) > self.history_size:
                    self.action_history.pop(0)
            
        except Exception as e:
            logger.error(f"动作识别失败: {e}")
        
        return results
    
    def get_action_trend(self) -> Dict[str, Any]:
        """获取动作趋势分析"""
        if not self.action_history:
            return {}
        
        # 统计动作频率
        action_counts = {}
        for action in self.action_history:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # 计算趋势
        total_actions = len(self.action_history)
        action_percentages = {
            action: count / total_actions 
            for action, count in action_counts.items()
        }
        
        # 主要动作
        dominant_action = max(action_counts.items(), key=lambda x: x[1])
        
        return {
            'history_length': total_actions,
            'action_counts': action_counts,
            'action_percentages': action_percentages,
            'dominant_action': dominant_action[0],
            'dominant_percentage': dominant_action[1] / total_actions,
            'recent_actions': self.action_history[-5:]  # 最近5个动作
        }

class ImprovedMultimodalDetector:
    """改进的多模态检测器"""
    
    def __init__(self,
                 face_database_path: Optional[str] = None,
                 use_pretrained_models: bool = True,
                 detection_interval: int = 1):
        """
        初始化改进的多模态检测器
        
        Args:
            face_database_path: 人脸数据库路径
            use_pretrained_models: 是否使用预训练模型
            detection_interval: 检测间隔（帧数）
        """
        self.face_database_path = face_database_path
        self.use_pretrained_models = use_pretrained_models
        self.detection_interval = detection_interval
        
        # 初始化各个识别器
        self.face_recognizer = None
        self.gesture_recognizer = None
        self.pose_recognizer = None
        self.fall_detector = None
        self.action_recognizer = None
        
        # 预训练模型加载器
        self.model_loader = None
        if use_pretrained_models:
            self.model_loader = PretrainedModelLoader()
        
        # 初始化所有识别器
        self._init_recognizers()
        
        # 检测历史和统计
        self.detection_history = []
        self.frame_count = 0
        self.performance_stats = {
            'total_detections': 0,
            'detection_types': {},
            'average_confidence': 0.0,
            'processing_times': []
        }
        
        logger.info("改进的多模态检测器初始化完成")
    
    def _init_recognizers(self):
        """初始化各个识别器"""
        try:
            # 面部识别器
            self.face_recognizer = EnhancedFaceRecognizer(
                face_database_path=self.face_database_path,
                min_detection_confidence=0.7,
                use_insightface=True
            )
            logger.info("面部识别器初始化完成")
            
            # 手势识别器
            self.gesture_recognizer = EnhancedGestureRecognizer(
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            logger.info("手势识别器初始化完成")
            
            # 姿势识别器
            self.pose_recognizer = EnhancedPoseRecognizer(
                use_yolo=True,
                use_mediapipe=True,
                min_detection_confidence=0.7
            )
            logger.info("姿势识别器初始化完成")
            
            # 摔倒检测器
            self.fall_detector = EnhancedFallDetector(
                use_pose_analysis=True,
                use_motion_analysis=True,
                use_ml_model=True
            )
            logger.info("摔倒检测器初始化完成")
            
            # 动作识别器（使用预训练模型）
            if self.model_loader:
                self.action_recognizer = ActionRecognitionModule(self.model_loader)
                logger.info("动作识别器初始化完成")
            
        except Exception as e:
            logger.error(f"识别器初始化失败: {e}")
    
    def detect_multimodal(self, frame: np.ndarray) -> MultimodalResult:
        """多模态检测"""
        start_time = time.time()
        self.frame_count += 1
        
        # 初始化结果
        result = MultimodalResult(
            faces=[],
            gestures=[],
            poses=[],
            actions=[],
            falls=[],
            timestamp=start_time,
            frame_id=self.frame_count,
            confidence_scores={}
        )
        
        # 检测控制
        should_detect = (self.frame_count % self.detection_interval == 0)
        if not should_detect:
            return result
        
        try:
            # 1. 面部检测和识别
            if self.face_recognizer:
                annotated_frame, face_results = self.face_recognizer.detect_faces(frame)
                for face_result in face_results:
                    face_recognition = RecognitionResult(
                        detection_type='face',
                        confidence=face_result.get('confidence', 0.0),
                        bbox=face_result.get('bbox', (0, 0, 0, 0)),
                        metadata=face_result
                    )
                    result.faces.append(face_recognition)
                
                if result.faces:
                    avg_face_conf = np.mean([f.confidence for f in result.faces])
                    result.confidence_scores['face'] = avg_face_conf
            
            # 2. 手势检测和识别
            if self.gesture_recognizer:
                annotated_frame, gesture_results = self.gesture_recognizer.detect_gestures(frame)
                for gesture_result in gesture_results:
                    gesture_recognition = RecognitionResult(
                        detection_type='gesture',
                        confidence=gesture_result.get('confidence', 0.0),
                        bbox=gesture_result.get('bbox', (0, 0, 0, 0)),
                        metadata=gesture_result
                    )
                    result.gestures.append(gesture_recognition)
                
                if result.gestures:
                    avg_gesture_conf = np.mean([g.confidence for g in result.gestures])
                    result.confidence_scores['gesture'] = avg_gesture_conf
            
            # 3. 姿势检测和识别
            pose_keypoints = None
            if self.pose_recognizer:
                annotated_frame, pose_results = self.pose_recognizer.detect_poses(frame)
                for pose_result in pose_results:
                    pose_recognition = RecognitionResult(
                        detection_type='pose',
                        confidence=pose_result.get('confidence', 0.0),
                        bbox=pose_result.get('bbox', (0, 0, 0, 0)),
                        metadata=pose_result
                    )
                    result.poses.append(pose_recognition)
                    
                    # 提取关键点用于动作识别
                    if 'keypoints' in pose_result:
                        pose_keypoints = np.array(pose_result['keypoints'])
                
                if result.poses:
                    avg_pose_conf = np.mean([p.confidence for p in result.poses])
                    result.confidence_scores['pose'] = avg_pose_conf
            
            # 4. 动作识别（使用深度学习模型）
            if self.action_recognizer:
                action_results = self.action_recognizer.recognize_action(frame, pose_keypoints)
                result.actions.extend(action_results)
                
                if result.actions:
                    avg_action_conf = np.mean([a.confidence for a in result.actions])
                    result.confidence_scores['action'] = avg_action_conf
            
            # 5. 摔倒检测
            if self.fall_detector:
                fall_results = self.fall_detector.detect_fall(frame, pose_keypoints)
                if fall_results:
                    for fall_result in fall_results:
                        fall_recognition = RecognitionResult(
                            detection_type='fall',
                            confidence=fall_result.get('confidence', 0.0),
                            bbox=fall_result.get('bbox', (0, 0, frame.shape[1], frame.shape[0])),
                            metadata=fall_result
                        )
                        result.falls.append(fall_recognition)
                    
                    avg_fall_conf = np.mean([f.confidence for f in result.falls])
                    result.confidence_scores['fall'] = avg_fall_conf
            
            # 更新统计信息
            self._update_statistics(result, time.time() - start_time)
            
        except Exception as e:
            logger.error(f"多模态检测失败: {e}")
        
        return result
    
    def _update_statistics(self, result: MultimodalResult, processing_time: float):
        """更新统计信息"""
        # 统计检测数量
        total_detections = (len(result.faces) + len(result.gestures) + 
                          len(result.poses) + len(result.actions) + len(result.falls))
        
        self.performance_stats['total_detections'] += total_detections
        self.performance_stats['processing_times'].append(processing_time)
        
        # 保持处理时间历史在合理范围内
        if len(self.performance_stats['processing_times']) > 100:
            self.performance_stats['processing_times'] = self.performance_stats['processing_times'][-100:]
        
        # 统计各类型检测数量
        for detection_type in ['face', 'gesture', 'pose', 'action', 'fall']:
            count = len(getattr(result, f"{detection_type}s", []))
            if detection_type not in self.performance_stats['detection_types']:
                self.performance_stats['detection_types'][detection_type] = 0
            self.performance_stats['detection_types'][detection_type] += count
        
        # 计算平均置信度
        all_confidences = []
        for detection_list in [result.faces, result.gestures, result.poses, result.actions, result.falls]:
            all_confidences.extend([d.confidence for d in detection_list])
        
        if all_confidences:
            self.performance_stats['average_confidence'] = np.mean(all_confidences)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        processing_times = self.performance_stats['processing_times']
        
        report = {
            'total_frames': self.frame_count,
            'total_detections': self.performance_stats['total_detections'],
            'detection_types': self.performance_stats['detection_types'].copy(),
            'average_confidence': self.performance_stats['average_confidence'],
            'processing_performance': {
                'average_time': np.mean(processing_times) if processing_times else 0,
                'min_time': np.min(processing_times) if processing_times else 0,
                'max_time': np.max(processing_times) if processing_times else 0,
                'fps': 1.0 / np.mean(processing_times) if processing_times else 0
            }
        }
        
        # 添加动作趋势分析
        if self.action_recognizer:
            report['action_trends'] = self.action_recognizer.get_action_trend()
        
        return report
    
    def save_detection_log(self, filepath: str):
        """保存检测日志"""
        log_data = {
            'performance_report': self.get_performance_report(),
            'detection_history': self.detection_history[-100:],  # 最近100条记录
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        logger.info(f"检测日志已保存: {filepath}")
    
    def load_face_database(self, filepath: str) -> bool:
        """加载人脸数据库"""
        if self.face_recognizer:
            return self.face_recognizer.load_face_database(filepath)
        return False
    
    def save_face_database(self, filepath: str):
        """保存人脸数据库"""
        if self.face_recognizer:
            self.face_recognizer.save_face_database(filepath)

def create_improved_multimodal_system(config: Optional[Dict[str, Any]] = None) -> ImprovedMultimodalDetector:
    """创建改进的多模态识别系统"""
    if config is None:
        config = {
            'face_database_path': './data/face_database.pkl',
            'use_pretrained_models': True,
            'detection_interval': 1
        }
    
    detector = ImprovedMultimodalDetector(**config)
    
    # 下载预训练模型（如果需要）
    if detector.model_loader:
        detector.model_loader.download_all_models()
    
    return detector

if __name__ == "__main__":
    # 示例使用
    detector = create_improved_multimodal_system()
    
    # 测试检测
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = detector.detect_multimodal(test_frame)
    
    print(f"检测结果:")
    print(f"  面部: {len(result.faces)}")
    print(f"  手势: {len(result.gestures)}")
    print(f"  姿势: {len(result.poses)}")
    print(f"  动作: {len(result.actions)}")
    print(f"  摔倒: {len(result.falls)}")
    
    # 性能报告
    report = detector.get_performance_report()
    print(f"\n性能报告: {report}")