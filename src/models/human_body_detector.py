#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AF-FPN人体检测模块
基于自适应特征金字塔网络实现全身人体检测和跟踪
解决人脸跟踪在侧脸、背身等场景下的局限性
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

class HumanBodyPart(Enum):
    """人体部位枚举"""
    HEAD = "head"
    TORSO = "torso"
    LEFT_ARM = "left_arm"
    RIGHT_ARM = "right_arm"
    LEFT_LEG = "left_leg"
    RIGHT_LEG = "right_leg"
    FULL_BODY = "full_body"

class DetectionMode(Enum):
    """检测模式"""
    FAST = "fast"  # 快速模式，适用于实时跟踪
    ACCURATE = "accurate"  # 精确模式，适用于高精度检测
    BALANCED = "balanced"  # 平衡模式，速度与精度兼顾

@dataclass
class HumanBodyKeypoint:
    """人体关键点"""
    x: float
    y: float
    confidence: float
    visibility: bool = True

@dataclass
class HumanBodyDetection:
    """人体检测结果"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    keypoints: Dict[str, HumanBodyKeypoint]
    body_parts: Dict[HumanBodyPart, Tuple[int, int, int, int]]
    pose_angle: float  # 身体朝向角度
    movement_vector: Optional[Tuple[float, float]] = None
    tracking_id: Optional[str] = None
    detection_time: float = 0.0

@dataclass
class HumanDetectorConfig:
    """人体检测器配置"""
    # 模型配置
    model_size: str = "s"  # n, s, m, l, x
    input_size: Tuple[int, int] = (640, 640)
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    
    # 检测配置
    detection_mode: DetectionMode = DetectionMode.BALANCED
    enable_keypoints: bool = True
    enable_pose_estimation: bool = True
    enable_body_parts: bool = True
    
    # 跟踪配置
    max_tracking_distance: float = 100.0
    tracking_smoothing: float = 0.7
    max_missing_frames: int = 10
    
    # 性能配置
    use_gpu: bool = True
    batch_size: int = 1
    num_threads: int = 4
    
    # 平台适配
    platform: str = "k230"  # k230, esp32, pc
    memory_limit_mb: int = 256
    enable_quantization: bool = True

class AdaptiveFeaturePyramid(nn.Module):
    """自适应特征金字塔网络"""
    
    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 横向连接
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels
        ])
        
        # 输出卷积
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels
        ])
        
        # 自适应注意力模块
        self.attention_modules = nn.ModuleList([
            self._build_attention_module(out_channels) for _ in in_channels
        ])
        
        # 特征融合权重
        self.fusion_weights = nn.Parameter(torch.ones(len(in_channels)))
        
    def _build_attention_module(self, channels: int) -> nn.Module:
        """构建注意力模块"""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """前向传播"""
        # 横向连接
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        
        # 自顶向下路径
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] += F.interpolate(
                laterals[i + 1], size=laterals[i].shape[-2:], mode='nearest'
            )
        
        # 应用注意力和输出卷积
        outputs = []
        for lateral, fpn_conv, attention in zip(laterals, self.fpn_convs, self.attention_modules):
            # 注意力增强
            att_weight = attention(lateral)
            enhanced_feat = lateral * att_weight
            
            # 输出卷积
            output = fpn_conv(enhanced_feat)
            outputs.append(output)
        
        return outputs

class HumanKeypointDetector(nn.Module):
    """人体关键点检测器"""
    
    def __init__(self, in_channels: int, num_keypoints: int = 17):
        super().__init__()
        self.num_keypoints = num_keypoints
        
        # 关键点检测头
        self.keypoint_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_keypoints, 1)
        )
        
        # 可见性预测头
        self.visibility_head = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_keypoints, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        keypoints = self.keypoint_head(x)
        visibility = self.visibility_head(x)
        return keypoints, visibility

class HumanBodyDetector:
    """AF-FPN人体检测器"""
    
    def __init__(self, config: HumanDetectorConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化模型
        self._init_models()
        
        # 跟踪器
        self.trackers = {}
        self.next_track_id = 1
        
        # 性能统计
        self.stats = {
            'total_detections': 0,
            'avg_detection_time': 0.0,
            'active_tracks': 0,
            'fps': 0.0
        }
        
        # 关键点定义（COCO格式）
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        self.logger.info(f"人体检测器初始化完成 - 模式: {config.detection_mode.value}")
    
    def _init_models(self):
        """初始化模型"""
        try:
            # 根据平台选择模型
            if self.config.platform == "k230":
                self._init_k230_model()
            elif self.config.platform == "esp32":
                self._init_esp32_model()
            else:
                self._init_pc_model()
                
        except Exception as e:
            self.logger.error(f"模型初始化失败: {e}")
            raise
    
    def _init_k230_model(self):
        """初始化K230模型"""
        self.logger.info("初始化K230优化模型")
        # K230特定的模型加载逻辑
        # 使用量化模型以适应内存限制
        pass
    
    def _init_esp32_model(self):
        """初始化ESP32模型"""
        self.logger.info("初始化ESP32轻量级模型")
        # ESP32特定的超轻量级模型
        pass
    
    def _init_pc_model(self):
        """初始化PC模型"""
        self.logger.info("初始化PC高精度模型")
        # PC平台的完整模型
        pass
    
    def detect_humans(self, image: np.ndarray) -> List[HumanBodyDetection]:
        """检测图像中的人体
        
        Args:
            image: 输入图像
            
        Returns:
            List[HumanBodyDetection]: 检测结果列表
        """
        start_time = time.time()
        
        try:
            # 预处理
            processed_image = self._preprocess_image(image)
            
            # 人体检测
            detections = self._detect_bodies(processed_image)
            
            # 关键点检测
            if self.config.enable_keypoints:
                detections = self._detect_keypoints(image, detections)
            
            # 姿态估计
            if self.config.enable_pose_estimation:
                detections = self._estimate_poses(detections)
            
            # 身体部位分割
            if self.config.enable_body_parts:
                detections = self._segment_body_parts(detections)
            
            # 更新统计信息
            detection_time = time.time() - start_time
            self._update_stats(len(detections), detection_time)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"人体检测失败: {e}")
            return []
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """图像预处理"""
        # 调整尺寸
        target_size = self.config.input_size
        resized = cv2.resize(image, target_size)
        
        # 归一化
        normalized = resized.astype(np.float32) / 255.0
        
        # 通道转换
        if len(normalized.shape) == 3:
            normalized = normalized.transpose(2, 0, 1)
        
        return normalized
    
    def _detect_bodies(self, image: np.ndarray) -> List[HumanBodyDetection]:
        """检测人体边界框"""
        # 模拟检测结果（实际应用中需要加载真实模型）
        detections = []
        
        # 这里应该是实际的模型推理代码
        # 为了演示，创建一个模拟检测结果
        mock_detection = HumanBodyDetection(
            bbox=(100, 100, 300, 500),
            confidence=0.85,
            keypoints={},
            body_parts={},
            pose_angle=0.0,
            detection_time=time.time()
        )
        detections.append(mock_detection)
        
        return detections
    
    def _detect_keypoints(self, image: np.ndarray, detections: List[HumanBodyDetection]) -> List[HumanBodyDetection]:
        """检测人体关键点"""
        for detection in detections:
            # 提取人体区域
            x1, y1, x2, y2 = detection.bbox
            body_region = image[y1:y2, x1:x2]
            
            # 关键点检测（模拟）
            keypoints = {}
            for i, name in enumerate(self.keypoint_names):
                # 模拟关键点坐标
                x = x1 + (x2 - x1) * np.random.random()
                y = y1 + (y2 - y1) * np.random.random()
                confidence = 0.7 + 0.3 * np.random.random()
                
                keypoints[name] = HumanBodyKeypoint(
                    x=float(x), y=float(y), confidence=float(confidence)
                )
            
            detection.keypoints = keypoints
        
        return detections
    
    def _estimate_poses(self, detections: List[HumanBodyDetection]) -> List[HumanBodyDetection]:
        """估计身体姿态"""
        for detection in detections:
            if 'left_shoulder' in detection.keypoints and 'right_shoulder' in detection.keypoints:
                left_shoulder = detection.keypoints['left_shoulder']
                right_shoulder = detection.keypoints['right_shoulder']
                
                # 计算肩膀连线角度
                dx = right_shoulder.x - left_shoulder.x
                dy = right_shoulder.y - left_shoulder.y
                angle = np.arctan2(dy, dx) * 180 / np.pi
                
                detection.pose_angle = float(angle)
        
        return detections
    
    def _segment_body_parts(self, detections: List[HumanBodyDetection]) -> List[HumanBodyDetection]:
        """分割身体部位"""
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            w, h = x2 - x1, y2 - y1
            
            # 基于比例估计身体部位
            body_parts = {
                HumanBodyPart.HEAD: (x1, y1, x2, y1 + int(h * 0.2)),
                HumanBodyPart.TORSO: (x1, y1 + int(h * 0.2), x2, y1 + int(h * 0.7)),
                HumanBodyPart.LEFT_LEG: (x1, y1 + int(h * 0.7), x1 + int(w * 0.5), y2),
                HumanBodyPart.RIGHT_LEG: (x1 + int(w * 0.5), y1 + int(h * 0.7), x2, y2),
            }
            
            detection.body_parts = body_parts
        
        return detections
    
    def track_humans(self, detections: List[HumanBodyDetection]) -> List[HumanBodyDetection]:
        """跟踪人体目标
        
        Args:
            detections: 当前帧检测结果
            
        Returns:
            List[HumanBodyDetection]: 带跟踪ID的检测结果
        """
        # 简化的跟踪算法
        for detection in detections:
            # 查找最近的跟踪目标
            best_match_id = None
            min_distance = float('inf')
            
            detection_center = self._get_detection_center(detection)
            
            for track_id, tracker_info in self.trackers.items():
                if tracker_info['missing_frames'] > self.config.max_missing_frames:
                    continue
                
                track_center = tracker_info['last_center']
                distance = np.sqrt(
                    (detection_center[0] - track_center[0]) ** 2 +
                    (detection_center[1] - track_center[1]) ** 2
                )
                
                if distance < min_distance and distance < self.config.max_tracking_distance:
                    min_distance = distance
                    best_match_id = track_id
            
            if best_match_id:
                # 更新现有跟踪
                detection.tracking_id = best_match_id
                self.trackers[best_match_id].update({
                    'last_center': detection_center,
                    'missing_frames': 0,
                    'last_detection': detection
                })
            else:
                # 创建新跟踪
                new_id = f"human_{self.next_track_id}"
                self.next_track_id += 1
                detection.tracking_id = new_id
                
                self.trackers[new_id] = {
                    'last_center': detection_center,
                    'missing_frames': 0,
                    'last_detection': detection,
                    'created_time': time.time()
                }
        
        # 更新丢失帧数
        for track_id in list(self.trackers.keys()):
            if not any(d.tracking_id == track_id for d in detections):
                self.trackers[track_id]['missing_frames'] += 1
                
                # 删除长时间丢失的跟踪
                if self.trackers[track_id]['missing_frames'] > self.config.max_missing_frames:
                    del self.trackers[track_id]
        
        return detections
    
    def _get_detection_center(self, detection: HumanBodyDetection) -> Tuple[float, float]:
        """获取检测框中心点"""
        x1, y1, x2, y2 = detection.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def get_tracking_target(self, detections: List[HumanBodyDetection]) -> Optional[HumanBodyDetection]:
        """获取主要跟踪目标
        
        Args:
            detections: 检测结果列表
            
        Returns:
            Optional[HumanBodyDetection]: 主要跟踪目标
        """
        if not detections:
            return None
        
        # 选择置信度最高的目标作为主要跟踪目标
        primary_target = max(detections, key=lambda d: d.confidence)
        
        # 或者选择最大的目标
        # primary_target = max(detections, key=lambda d: self._get_bbox_area(d.bbox))
        
        return primary_target
    
    def _get_bbox_area(self, bbox: Tuple[int, int, int, int]) -> int:
        """计算边界框面积"""
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)
    
    def visualize_detections(self, image: np.ndarray, detections: List[HumanBodyDetection]) -> np.ndarray:
        """可视化检测结果
        
        Args:
            image: 原始图像
            detections: 检测结果
            
        Returns:
            np.ndarray: 标注后的图像
        """
        result_image = image.copy()
        
        for detection in detections:
            # 绘制边界框
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制跟踪ID和置信度
            label = f"ID: {detection.tracking_id or 'N/A'} ({detection.confidence:.2f})"
            cv2.putText(result_image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 绘制关键点
            if detection.keypoints:
                for name, keypoint in detection.keypoints.items():
                    if keypoint.confidence > 0.5:
                        cv2.circle(result_image, 
                                 (int(keypoint.x), int(keypoint.y)), 
                                 3, (255, 0, 0), -1)
            
            # 绘制身体部位
            if detection.body_parts:
                colors = {
                    HumanBodyPart.HEAD: (255, 255, 0),
                    HumanBodyPart.TORSO: (0, 255, 255),
                    HumanBodyPart.LEFT_LEG: (255, 0, 255),
                    HumanBodyPart.RIGHT_LEG: (255, 0, 255)
                }
                
                for part, bbox in detection.body_parts.items():
                    if part in colors:
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(result_image, (x1, y1), (x2, y2), 
                                    colors[part], 1)
        
        return result_image
    
    def _update_stats(self, num_detections: int, detection_time: float):
        """更新性能统计"""
        self.stats['total_detections'] += num_detections
        
        # 更新平均检测时间
        if self.stats['avg_detection_time'] == 0:
            self.stats['avg_detection_time'] = detection_time
        else:
            self.stats['avg_detection_time'] = (
                self.stats['avg_detection_time'] * 0.9 + detection_time * 0.1
            )
        
        # 更新FPS
        if detection_time > 0:
            self.stats['fps'] = 1.0 / detection_time
        
        # 更新活跃跟踪数
        self.stats['active_tracks'] = len(self.trackers)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return self.stats.copy()
    
    def reset_tracking(self):
        """重置跟踪状态"""
        self.trackers.clear()
        self.next_track_id = 1
        self.logger.info("跟踪状态已重置")

# 测试代码
if __name__ == "__main__":
    # 创建配置
    config = HumanDetectorConfig(
        detection_mode=DetectionMode.BALANCED,
        platform="k230",
        confidence_threshold=0.6
    )
    
    # 创建检测器
    detector = HumanBodyDetector(config)
    
    # 模拟图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 执行检测
    detections = detector.detect_humans(test_image)
    
    # 执行跟踪
    tracked_detections = detector.track_humans(detections)
    
    # 可视化结果
    result_image = detector.visualize_detections(test_image, tracked_detections)
    
    # 获取主要跟踪目标
    primary_target = detector.get_tracking_target(tracked_detections)
    
    print(f"检测到 {len(tracked_detections)} 个人体目标")
    if primary_target:
        print(f"主要跟踪目标: ID={primary_target.tracking_id}, 置信度={primary_target.confidence:.2f}")
    
    # 打印统计信息
    stats = detector.get_stats()
    print(f"性能统计: {stats}")