#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实场景适应模块
实现光照变化、遮挡处理、多人场景等复杂环境下的跟踪优化
提升系统在真实场景中的鲁棒性和适应性
"""

import time
import logging
import threading
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque, defaultdict
import math

# 可选依赖
try:
    import cv2
except ImportError:
    cv2 = None

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    DBSCAN = None

logger = logging.getLogger(__name__)

class SceneCondition(Enum):
    """场景条件"""
    NORMAL = "normal"  # 正常条件
    LOW_LIGHT = "low_light"  # 低光照
    HIGH_LIGHT = "high_light"  # 强光照
    BACKLIGHT = "backlight"  # 逆光
    SHADOW = "shadow"  # 阴影
    OCCLUSION = "occlusion"  # 遮挡
    CROWDED = "crowded"  # 拥挤场景
    MOTION_BLUR = "motion_blur"  # 运动模糊
    WEATHER = "weather"  # 恶劣天气

class OcclusionType(Enum):
    """遮挡类型"""
    NONE = "none"  # 无遮挡
    PARTIAL = "partial"  # 部分遮挡
    SEVERE = "severe"  # 严重遮挡
    COMPLETE = "complete"  # 完全遮挡
    INTER_OBJECT = "inter_object"  # 物体间遮挡
    SELF_OCCLUSION = "self_occlusion"  # 自遮挡

class AdaptationStrategy(Enum):
    """适应策略"""
    CONSERVATIVE = "conservative"  # 保守策略
    AGGRESSIVE = "aggressive"  # 激进策略
    BALANCED = "balanced"  # 平衡策略
    ADAPTIVE = "adaptive"  # 自适应策略
    SCENE_SPECIFIC = "scene_specific"  # 场景特定策略

@dataclass
class SceneAnalysis:
    """场景分析结果"""
    timestamp: float
    conditions: List[SceneCondition]
    lighting_score: float  # 光照评分 (0-1)
    contrast_score: float  # 对比度评分 (0-1)
    noise_level: float  # 噪声水平 (0-1)
    motion_level: float  # 运动水平 (0-1)
    crowd_density: float  # 人群密度 (0-1)
    occlusion_level: float  # 遮挡水平 (0-1)
    complexity_score: float  # 场景复杂度 (0-1)
    confidence: float = 0.8  # 分析置信度
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()

@dataclass
class OcclusionInfo:
    """遮挡信息"""
    occlusion_type: OcclusionType
    occlusion_ratio: float  # 遮挡比例 (0-1)
    occluded_parts: List[str]  # 被遮挡的部位
    occluder_info: Optional[Dict[str, Any]] = None  # 遮挡物信息
    recovery_probability: float = 0.5  # 恢复概率
    duration: float = 0.0  # 遮挡持续时间
    
@dataclass
class AdaptationConfig:
    """适应配置"""
    # 光照适应
    enable_lighting_adaptation: bool = True
    auto_exposure_adjustment: bool = True
    histogram_equalization: bool = True
    gamma_correction: bool = True
    
    # 遮挡处理
    enable_occlusion_handling: bool = True
    occlusion_threshold: float = 0.3
    recovery_timeout: float = 5.0
    prediction_during_occlusion: bool = True
    
    # 多人场景
    enable_multi_person_optimization: bool = True
    max_persons: int = 10
    person_association_threshold: float = 0.7
    crowd_detection_threshold: float = 0.6
    
    # 运动适应
    enable_motion_adaptation: bool = True
    motion_blur_detection: bool = True
    adaptive_tracking_window: bool = True
    
    # 场景分析
    analysis_interval: float = 1.0  # 分析间隔（秒）
    scene_memory_size: int = 30  # 场景记忆大小
    adaptation_sensitivity: float = 0.5  # 适应敏感度

class LightingAdapter:
    """光照适应器
    
    处理各种光照条件下的图像增强和适应
    """
    
    def __init__(self, config: AdaptationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 光照历史
        self.lighting_history = deque(maxlen=30)
        self.exposure_history = deque(maxlen=10)
        
        # 自适应参数
        self.current_gamma = 1.0
        self.current_exposure = 0.0
        self.adaptive_clahe = None
        
        if cv2 is not None:
            self.adaptive_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def analyze_lighting(self, image: np.ndarray) -> Dict[str, float]:
        """分析图像光照条件
        
        Args:
            image: 输入图像
            
        Returns:
            Dict[str, float]: 光照分析结果
        """
        if cv2 is None:
            return {"brightness": 0.5, "contrast": 0.5, "uniformity": 0.5}
        
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 计算亮度统计
            mean_brightness = np.mean(gray) / 255.0
            std_brightness = np.std(gray) / 255.0
            
            # 计算对比度
            contrast = std_brightness
            
            # 计算直方图
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_norm = hist.flatten() / hist.sum()
            
            # 计算光照均匀性
            # 将图像分成网格，计算各区域亮度差异
            h, w = gray.shape
            grid_size = 4
            grid_h, grid_w = h // grid_size, w // grid_size
            
            grid_means = []
            for i in range(grid_size):
                for j in range(grid_size):
                    y1, y2 = i * grid_h, (i + 1) * grid_h
                    x1, x2 = j * grid_w, (j + 1) * grid_w
                    grid_mean = np.mean(gray[y1:y2, x1:x2])
                    grid_means.append(grid_mean)
            
            uniformity = 1.0 - (np.std(grid_means) / 255.0)
            
            # 检测过曝和欠曝
            overexposed = np.sum(gray > 240) / gray.size
            underexposed = np.sum(gray < 15) / gray.size
            
            analysis = {
                "brightness": mean_brightness,
                "contrast": contrast,
                "uniformity": uniformity,
                "overexposed_ratio": overexposed,
                "underexposed_ratio": underexposed,
                "histogram_entropy": -np.sum(hist_norm * np.log(hist_norm + 1e-10))
            }
            
            self.lighting_history.append(analysis)
            return analysis
            
        except Exception as e:
            self.logger.error(f"光照分析失败: {e}")
            return {"brightness": 0.5, "contrast": 0.5, "uniformity": 0.5}
    
    def adapt_lighting(self, image: np.ndarray, lighting_analysis: Dict[str, float]) -> np.ndarray:
        """根据光照条件适应图像
        
        Args:
            image: 输入图像
            lighting_analysis: 光照分析结果
            
        Returns:
            np.ndarray: 适应后的图像
        """
        if cv2 is None:
            return image
        
        try:
            adapted_image = image.copy()
            
            # 获取光照参数
            brightness = lighting_analysis.get("brightness", 0.5)
            contrast = lighting_analysis.get("contrast", 0.5)
            uniformity = lighting_analysis.get("uniformity", 0.5)
            overexposed = lighting_analysis.get("overexposed_ratio", 0.0)
            underexposed = lighting_analysis.get("underexposed_ratio", 0.0)
            
            # 自动曝光调整
            if self.config.auto_exposure_adjustment:
                if brightness < 0.3:  # 图像过暗
                    exposure_adjustment = (0.5 - brightness) * 0.5
                    adapted_image = cv2.convertScaleAbs(adapted_image, alpha=1.0, beta=exposure_adjustment * 255)
                elif brightness > 0.7:  # 图像过亮
                    exposure_adjustment = (brightness - 0.5) * 0.3
                    adapted_image = cv2.convertScaleAbs(adapted_image, alpha=1.0, beta=-exposure_adjustment * 255)
            
            # Gamma校正
            if self.config.gamma_correction:
                if brightness < 0.4:
                    gamma = 0.7  # 提亮暗部
                elif brightness > 0.6:
                    gamma = 1.3  # 压暗亮部
                else:
                    gamma = 1.0
                
                if gamma != 1.0:
                    gamma_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
                    adapted_image = cv2.LUT(adapted_image, gamma_table)
            
            # 直方图均衡化
            if self.config.histogram_equalization and contrast < 0.3:
                if len(adapted_image.shape) == 3:
                    # 彩色图像：在LAB空间进行CLAHE
                    lab = cv2.cvtColor(adapted_image, cv2.COLOR_BGR2LAB)
                    lab[:, :, 0] = self.adaptive_clahe.apply(lab[:, :, 0])
                    adapted_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                else:
                    # 灰度图像：直接CLAHE
                    adapted_image = self.adaptive_clahe.apply(adapted_image)
            
            # 对比度增强
            if contrast < 0.2:
                alpha = 1.5  # 对比度增强因子
                beta = 0     # 亮度调整
                adapted_image = cv2.convertScaleAbs(adapted_image, alpha=alpha, beta=beta)
            
            return adapted_image
            
        except Exception as e:
            self.logger.error(f"光照适应失败: {e}")
            return image
    
    def detect_lighting_conditions(self, lighting_analysis: Dict[str, float]) -> List[SceneCondition]:
        """检测光照条件
        
        Args:
            lighting_analysis: 光照分析结果
            
        Returns:
            List[SceneCondition]: 检测到的光照条件
        """
        conditions = []
        
        brightness = lighting_analysis.get("brightness", 0.5)
        contrast = lighting_analysis.get("contrast", 0.5)
        uniformity = lighting_analysis.get("uniformity", 0.5)
        overexposed = lighting_analysis.get("overexposed_ratio", 0.0)
        underexposed = lighting_analysis.get("underexposed_ratio", 0.0)
        
        # 低光照
        if brightness < 0.3 or underexposed > 0.2:
            conditions.append(SceneCondition.LOW_LIGHT)
        
        # 强光照
        if brightness > 0.7 or overexposed > 0.1:
            conditions.append(SceneCondition.HIGH_LIGHT)
        
        # 逆光（高对比度 + 不均匀光照）
        if contrast > 0.6 and uniformity < 0.4:
            conditions.append(SceneCondition.BACKLIGHT)
        
        # 阴影（不均匀光照）
        if uniformity < 0.5 and 0.3 < brightness < 0.7:
            conditions.append(SceneCondition.SHADOW)
        
        # 正常条件
        if not conditions and 0.4 <= brightness <= 0.6 and contrast > 0.2 and uniformity > 0.6:
            conditions.append(SceneCondition.NORMAL)
        
        return conditions

class OcclusionHandler:
    """遮挡处理器
    
    检测和处理目标遮挡情况
    """
    
    def __init__(self, config: AdaptationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 遮挡历史
        self.occlusion_history: Dict[int, List[OcclusionInfo]] = defaultdict(list)
        self.recovery_timers: Dict[int, float] = {}
        
        # 预测模型（简单的线性预测）
        self.position_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=10))
        self.velocity_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=5))
    
    def detect_occlusion(self, target_id: int, detection_box: Optional[Tuple[int, int, int, int]], 
                        confidence: float, image: np.ndarray) -> OcclusionInfo:
        """检测目标遮挡情况
        
        Args:
            target_id: 目标ID
            detection_box: 检测框 (x, y, w, h)
            confidence: 检测置信度
            image: 当前图像
            
        Returns:
            OcclusionInfo: 遮挡信息
        """
        current_time = time.time()
        
        # 如果没有检测框，认为是完全遮挡
        if detection_box is None or confidence < 0.3:
            occlusion_info = OcclusionInfo(
                occlusion_type=OcclusionType.COMPLETE,
                occlusion_ratio=1.0,
                occluded_parts=["全身"],
                recovery_probability=0.2
            )
        else:
            # 分析检测框质量
            x, y, w, h = detection_box
            
            # 计算遮挡比例（基于置信度和框大小）
            expected_size = self._estimate_expected_size(target_id)
            current_size = w * h
            
            size_ratio = current_size / max(expected_size, 1) if expected_size > 0 else 1.0
            confidence_factor = confidence
            
            # 综合评估遮挡程度
            occlusion_ratio = 1.0 - (size_ratio * confidence_factor)
            occlusion_ratio = np.clip(occlusion_ratio, 0.0, 1.0)
            
            # 确定遮挡类型
            if occlusion_ratio < 0.2:
                occlusion_type = OcclusionType.NONE
            elif occlusion_ratio < 0.5:
                occlusion_type = OcclusionType.PARTIAL
            elif occlusion_ratio < 0.8:
                occlusion_type = OcclusionType.SEVERE
            else:
                occlusion_type = OcclusionType.COMPLETE
            
            # 分析被遮挡的部位
            occluded_parts = self._analyze_occluded_parts(detection_box, image)
            
            # 计算恢复概率
            recovery_prob = self._calculate_recovery_probability(target_id, occlusion_ratio)
            
            occlusion_info = OcclusionInfo(
                occlusion_type=occlusion_type,
                occlusion_ratio=occlusion_ratio,
                occluded_parts=occluded_parts,
                recovery_probability=recovery_prob
            )
        
        # 更新遮挡历史
        self.occlusion_history[target_id].append(occlusion_info)
        if len(self.occlusion_history[target_id]) > 20:
            self.occlusion_history[target_id].pop(0)
        
        # 更新恢复计时器
        if occlusion_info.occlusion_type in [OcclusionType.SEVERE, OcclusionType.COMPLETE]:
            if target_id not in self.recovery_timers:
                self.recovery_timers[target_id] = current_time
        else:
            self.recovery_timers.pop(target_id, None)
        
        return occlusion_info
    
    def predict_position_during_occlusion(self, target_id: int) -> Optional[Tuple[int, int, int, int]]:
        """在遮挡期间预测目标位置
        
        Args:
            target_id: 目标ID
            
        Returns:
            Optional[Tuple[int, int, int, int]]: 预测的位置框
        """
        if not self.config.prediction_during_occlusion:
            return None
        
        positions = self.position_history.get(target_id)
        if not positions or len(positions) < 3:
            return None
        
        try:
            # 简单的线性预测
            recent_positions = list(positions)[-3:]
            
            # 计算平均速度
            velocities = []
            for i in range(1, len(recent_positions)):
                prev_pos = recent_positions[i-1]
                curr_pos = recent_positions[i]
                
                vx = curr_pos[0] - prev_pos[0]
                vy = curr_pos[1] - prev_pos[1]
                velocities.append((vx, vy))
            
            if not velocities:
                return None
            
            # 平均速度
            avg_vx = np.mean([v[0] for v in velocities])
            avg_vy = np.mean([v[1] for v in velocities])
            
            # 预测下一个位置
            last_pos = recent_positions[-1]
            predicted_x = int(last_pos[0] + avg_vx)
            predicted_y = int(last_pos[1] + avg_vy)
            
            # 保持原有的宽高
            predicted_w = last_pos[2]
            predicted_h = last_pos[3]
            
            return (predicted_x, predicted_y, predicted_w, predicted_h)
            
        except Exception as e:
            self.logger.error(f"位置预测失败: {e}")
            return None
    
    def update_position_history(self, target_id: int, position: Tuple[int, int, int, int]):
        """更新位置历史
        
        Args:
            target_id: 目标ID
            position: 位置 (x, y, w, h)
        """
        self.position_history[target_id].append(position)
    
    def _estimate_expected_size(self, target_id: int) -> float:
        """估计目标的期望大小
        
        Args:
            target_id: 目标ID
            
        Returns:
            float: 期望大小
        """
        positions = self.position_history.get(target_id)
        if not positions:
            return 0.0
        
        sizes = [pos[2] * pos[3] for pos in positions]
        return np.median(sizes) if sizes else 0.0
    
    def _analyze_occluded_parts(self, detection_box: Tuple[int, int, int, int], image: np.ndarray) -> List[str]:
        """分析被遮挡的身体部位
        
        Args:
            detection_box: 检测框
            image: 图像
            
        Returns:
            List[str]: 被遮挡的部位
        """
        # 简化实现：基于检测框的位置和大小推断
        x, y, w, h = detection_box
        
        occluded_parts = []
        
        # 基于框的高度比例判断可能遮挡的部位
        if h < w * 1.5:  # 正常人体比例约为1.7-2.0
            occluded_parts.append("下半身")
        
        if w < h * 0.3:  # 宽度过窄
            occluded_parts.append("侧面")
        
        # 如果没有明显遮挡，返回空列表
        return occluded_parts if occluded_parts else ["未知部位"]
    
    def _calculate_recovery_probability(self, target_id: int, occlusion_ratio: float) -> float:
        """计算恢复概率
        
        Args:
            target_id: 目标ID
            occlusion_ratio: 遮挡比例
            
        Returns:
            float: 恢复概率
        """
        # 基于历史遮挡模式计算恢复概率
        history = self.occlusion_history.get(target_id, [])
        
        if not history:
            return 0.5  # 默认概率
        
        # 计算历史恢复率
        recovery_count = 0
        total_occlusions = 0
        
        for i in range(len(history) - 1):
            if history[i].occlusion_ratio > 0.5:  # 严重遮挡
                total_occlusions += 1
                if history[i + 1].occlusion_ratio < 0.3:  # 恢复
                    recovery_count += 1
        
        if total_occlusions == 0:
            base_probability = 0.7
        else:
            base_probability = recovery_count / total_occlusions
        
        # 根据当前遮挡程度调整
        occlusion_factor = 1.0 - occlusion_ratio
        
        # 根据遮挡持续时间调整
        current_time = time.time()
        occlusion_start = self.recovery_timers.get(target_id, current_time)
        occlusion_duration = current_time - occlusion_start
        
        time_factor = max(0.1, 1.0 - occlusion_duration / self.config.recovery_timeout)
        
        recovery_probability = base_probability * occlusion_factor * time_factor
        return np.clip(recovery_probability, 0.1, 0.9)
    
    def should_drop_target(self, target_id: int) -> bool:
        """判断是否应该丢弃目标
        
        Args:
            target_id: 目标ID
            
        Returns:
            bool: 是否丢弃
        """
        current_time = time.time()
        occlusion_start = self.recovery_timers.get(target_id)
        
        if occlusion_start is None:
            return False
        
        occlusion_duration = current_time - occlusion_start
        return occlusion_duration > self.config.recovery_timeout
    
    def get_occlusion_stats(self, target_id: int) -> Dict[str, Any]:
        """获取遮挡统计信息
        
        Args:
            target_id: 目标ID
            
        Returns:
            Dict[str, Any]: 遮挡统计
        """
        history = self.occlusion_history.get(target_id, [])
        
        if not history:
            return {"total_occlusions": 0, "avg_occlusion_ratio": 0.0, "recovery_rate": 0.0}
        
        total_occlusions = len([h for h in history if h.occlusion_ratio > 0.3])
        avg_occlusion_ratio = np.mean([h.occlusion_ratio for h in history])
        
        # 计算恢复率
        recoveries = 0
        severe_occlusions = 0
        
        for i in range(len(history) - 1):
            if history[i].occlusion_ratio > 0.5:
                severe_occlusions += 1
                if history[i + 1].occlusion_ratio < 0.3:
                    recoveries += 1
        
        recovery_rate = recoveries / max(severe_occlusions, 1)
        
        return {
            "total_occlusions": total_occlusions,
            "avg_occlusion_ratio": avg_occlusion_ratio,
            "recovery_rate": recovery_rate,
            "current_occlusion_duration": time.time() - self.recovery_timers.get(target_id, time.time())
        }

class MultiPersonOptimizer:
    """多人场景优化器
    
    处理拥挤场景中的多目标跟踪优化
    """
    
    def __init__(self, config: AdaptationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 人群分析
        self.crowd_history = deque(maxlen=30)
        self.person_associations: Dict[int, List[int]] = {}  # 人员关联
        
        # 聚类器（用于人群分析）
        self.clusterer = None
        if DBSCAN is not None:
            self.clusterer = DBSCAN(eps=50, min_samples=2)
    
    def analyze_crowd_density(self, detections: List[Tuple[int, int, int, int, float]]) -> float:
        """分析人群密度
        
        Args:
            detections: 检测结果列表 [(x, y, w, h, confidence), ...]
            
        Returns:
            float: 人群密度 (0-1)
        """
        if not detections:
            return 0.0
        
        # 计算检测框的中心点
        centers = []
        for x, y, w, h, conf in detections:
            if conf > 0.5:  # 只考虑高置信度检测
                center_x = x + w // 2
                center_y = y + h // 2
                centers.append([center_x, center_y])
        
        if len(centers) < 2:
            density = len(centers) / max(self.config.max_persons, 1)
        else:
            # 使用聚类分析人群分布
            if self.clusterer is not None:
                try:
                    clusters = self.clusterer.fit_predict(centers)
                    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
                    
                    # 计算平均簇内距离
                    avg_distances = []
                    for cluster_id in set(clusters):
                        if cluster_id == -1:  # 噪声点
                            continue
                        
                        cluster_points = [centers[i] for i, c in enumerate(clusters) if c == cluster_id]
                        if len(cluster_points) > 1:
                            distances = []
                            for i in range(len(cluster_points)):
                                for j in range(i + 1, len(cluster_points)):
                                    dist = np.linalg.norm(np.array(cluster_points[i]) - np.array(cluster_points[j]))
                                    distances.append(dist)
                            if distances:
                                avg_distances.append(np.mean(distances))
                    
                    # 密度 = 人数 / 平均距离
                    if avg_distances:
                        avg_distance = np.mean(avg_distances)
                        density = len(centers) / max(avg_distance / 100.0, 1.0)  # 归一化
                    else:
                        density = len(centers) / max(self.config.max_persons, 1)
                        
                except Exception as e:
                    self.logger.error(f"聚类分析失败: {e}")
                    density = len(centers) / max(self.config.max_persons, 1)
            else:
                # 简单的密度计算
                density = len(centers) / max(self.config.max_persons, 1)
        
        density = np.clip(density, 0.0, 1.0)
        self.crowd_history.append(density)
        
        return density
    
    def optimize_for_crowd(self, detections: List[Tuple[int, int, int, int, float]], 
                          crowd_density: float) -> Dict[str, Any]:
        """针对人群场景进行优化
        
        Args:
            detections: 检测结果
            crowd_density: 人群密度
            
        Returns:
            Dict[str, Any]: 优化建议
        """
        optimization = {
            "tracking_strategy": "normal",
            "association_threshold": self.config.person_association_threshold,
            "max_tracking_targets": self.config.max_persons,
            "enable_prediction": True,
            "enable_re_identification": False
        }
        
        if crowd_density > 0.7:  # 高密度人群
            optimization.update({
                "tracking_strategy": "conservative",
                "association_threshold": 0.8,  # 提高关联阈值
                "max_tracking_targets": min(self.config.max_persons, 6),  # 限制跟踪目标数
                "enable_prediction": True,
                "enable_re_identification": True,  # 启用重识别
                "temporal_consistency_weight": 0.7,  # 增加时序一致性权重
                "spatial_consistency_weight": 0.8   # 增加空间一致性权重
            })
            
        elif crowd_density > 0.4:  # 中等密度
            optimization.update({
                "tracking_strategy": "balanced",
                "association_threshold": 0.75,
                "enable_re_identification": True,
                "temporal_consistency_weight": 0.5,
                "spatial_consistency_weight": 0.6
            })
        
        return optimization
    
    def detect_person_interactions(self, detections: List[Tuple[int, int, int, int, float]]) -> List[Dict[str, Any]]:
        """检测人员交互
        
        Args:
            detections: 检测结果
            
        Returns:
            List[Dict[str, Any]]: 交互信息
        """
        interactions = []
        
        if len(detections) < 2:
            return interactions
        
        # 计算所有检测框之间的距离和重叠
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                x1, y1, w1, h1, conf1 = detections[i]
                x2, y2, w2, h2, conf2 = detections[j]
                
                # 计算中心点距离
                center1 = (x1 + w1 // 2, y1 + h1 // 2)
                center2 = (x2 + w2 // 2, y2 + h2 // 2)
                distance = np.linalg.norm(np.array(center1) - np.array(center2))
                
                # 计算重叠面积
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                
                # 计算IoU
                area1 = w1 * h1
                area2 = w2 * h2
                union_area = area1 + area2 - overlap_area
                iou = overlap_area / max(union_area, 1)
                
                # 判断交互类型
                interaction_type = "none"
                if iou > 0.3:
                    interaction_type = "occlusion"
                elif distance < max(w1, w2, h1, h2) * 0.5:
                    interaction_type = "proximity"
                elif distance < max(w1, w2, h1, h2) * 1.0:
                    interaction_type = "nearby"
                
                if interaction_type != "none":
                    interactions.append({
                        "person1_index": i,
                        "person2_index": j,
                        "interaction_type": interaction_type,
                        "distance": distance,
                        "iou": iou,
                        "confidence": min(conf1, conf2)
                    })
        
        return interactions
    
    def get_crowd_analysis(self) -> Dict[str, Any]:
        """获取人群分析结果
        
        Returns:
            Dict[str, Any]: 人群分析
        """
        if not self.crowd_history:
            return {"avg_density": 0.0, "max_density": 0.0, "density_trend": "stable"}
        
        densities = list(self.crowd_history)
        avg_density = np.mean(densities)
        max_density = np.max(densities)
        
        # 分析密度趋势
        if len(densities) >= 5:
            recent_avg = np.mean(densities[-5:])
            earlier_avg = np.mean(densities[-10:-5]) if len(densities) >= 10 else avg_density
            
            if recent_avg > earlier_avg * 1.2:
                trend = "increasing"
            elif recent_avg < earlier_avg * 0.8:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "unknown"
        
        return {
            "avg_density": avg_density,
            "max_density": max_density,
            "current_density": densities[-1],
            "density_trend": trend,
            "crowd_level": "high" if avg_density > 0.7 else ("medium" if avg_density > 0.4 else "low")
        }

class SceneAdapter:
    """场景适应器主类
    
    整合各种场景适应技术
    """
    
    def __init__(self, config: AdaptationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化组件
        self.lighting_adapter = LightingAdapter(config)
        self.occlusion_handler = OcclusionHandler(config)
        self.multi_person_optimizer = MultiPersonOptimizer(config)
        
        # 场景分析历史
        self.scene_history = deque(maxlen=config.scene_memory_size)
        self.last_analysis_time = 0.0
        
        # 自适应参数
        self.current_strategy = AdaptationStrategy.BALANCED
        self.adaptation_weights = {
            "lighting": 0.3,
            "occlusion": 0.4,
            "crowd": 0.3
        }
        
        self.logger.info("场景适应器初始化完成")
    
    def analyze_scene(self, image: np.ndarray, detections: List[Tuple[int, int, int, int, float]]) -> SceneAnalysis:
        """分析当前场景
        
        Args:
            image: 输入图像
            detections: 检测结果
            
        Returns:
            SceneAnalysis: 场景分析结果
        """
        current_time = time.time()
        
        # 检查分析间隔
        if current_time - self.last_analysis_time < self.config.analysis_interval:
            # 返回最近的分析结果
            if self.scene_history:
                return self.scene_history[-1]
        
        self.last_analysis_time = current_time
        
        # 光照分析
        lighting_analysis = self.lighting_adapter.analyze_lighting(image)
        lighting_conditions = self.lighting_adapter.detect_lighting_conditions(lighting_analysis)
        
        # 人群分析
        crowd_density = self.multi_person_optimizer.analyze_crowd_density(detections)
        
        # 运动分析（简化实现）
        motion_level = self._analyze_motion_level(image)
        
        # 噪声分析
        noise_level = self._analyze_noise_level(image)
        
        # 综合场景条件
        scene_conditions = lighting_conditions.copy()
        
        if crowd_density > 0.6:
            scene_conditions.append(SceneCondition.CROWDED)
        
        if motion_level > 0.7:
            scene_conditions.append(SceneCondition.MOTION_BLUR)
        
        # 计算复杂度评分
        complexity_score = self._calculate_complexity_score(
            lighting_analysis, crowd_density, motion_level, noise_level
        )
        
        # 创建场景分析结果
        analysis = SceneAnalysis(
            timestamp=current_time,
            conditions=scene_conditions,
            lighting_score=lighting_analysis.get("brightness", 0.5),
            contrast_score=lighting_analysis.get("contrast", 0.5),
            noise_level=noise_level,
            motion_level=motion_level,
            crowd_density=crowd_density,
            occlusion_level=self._estimate_overall_occlusion_level(),
            complexity_score=complexity_score
        )
        
        # 更新历史
        self.scene_history.append(analysis)
        
        return analysis
    
    def adapt_to_scene(self, image: np.ndarray, scene_analysis: SceneAnalysis) -> Tuple[np.ndarray, Dict[str, Any]]:
        """根据场景分析适应图像和跟踪策略
        
        Args:
            image: 输入图像
            scene_analysis: 场景分析结果
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: (适应后图像, 跟踪策略)
        """
        adapted_image = image.copy()
        tracking_strategy = {}
        
        # 光照适应
        if self.config.enable_lighting_adaptation:
            lighting_analysis = {
                "brightness": scene_analysis.lighting_score,
                "contrast": scene_analysis.contrast_score,
                "uniformity": 1.0 - scene_analysis.complexity_score * 0.5
            }
            adapted_image = self.lighting_adapter.adapt_lighting(adapted_image, lighting_analysis)
        
        # 多人场景优化
        if self.config.enable_multi_person_optimization and scene_analysis.crowd_density > 0.3:
            crowd_optimization = self.multi_person_optimizer.optimize_for_crowd([], scene_analysis.crowd_density)
            tracking_strategy.update(crowd_optimization)
        
        # 根据场景条件调整策略
        if SceneCondition.LOW_LIGHT in scene_analysis.conditions:
            tracking_strategy.update({
                "detection_threshold": 0.3,  # 降低检测阈值
                "temporal_smoothing": 0.8,   # 增加时序平滑
                "enable_enhancement": True
            })
        
        if SceneCondition.CROWDED in scene_analysis.conditions:
            tracking_strategy.update({
                "association_method": "hungarian",
                "max_disappeared_frames": 5,
                "enable_reid": True
            })
        
        if SceneCondition.MOTION_BLUR in scene_analysis.conditions:
            tracking_strategy.update({
                "prediction_weight": 0.7,
                "motion_model": "constant_velocity",
                "kalman_process_noise": 0.1
            })
        
        # 自适应策略选择
        if self.config.adaptation_sensitivity > 0.5:
            self.current_strategy = self._select_adaptation_strategy(scene_analysis)
            tracking_strategy["adaptation_strategy"] = self.current_strategy.value
        
        return adapted_image, tracking_strategy
    
    def handle_target_occlusion(self, target_id: int, detection_box: Optional[Tuple[int, int, int, int]], 
                               confidence: float, image: np.ndarray) -> Dict[str, Any]:
        """处理目标遮挡
        
        Args:
            target_id: 目标ID
            detection_box: 检测框
            confidence: 置信度
            image: 图像
            
        Returns:
            Dict[str, Any]: 遮挡处理结果
        """
        if not self.config.enable_occlusion_handling:
            return {"action": "none"}
        
        # 检测遮挡
        occlusion_info = self.occlusion_handler.detect_occlusion(target_id, detection_box, confidence, image)
        
        # 更新位置历史
        if detection_box is not None:
            self.occlusion_handler.update_position_history(target_id, detection_box)
        
        # 决定处理动作
        action_info = {"action": "continue", "occlusion_info": occlusion_info}
        
        if occlusion_info.occlusion_type == OcclusionType.COMPLETE:
            # 完全遮挡：使用预测位置
            predicted_pos = self.occlusion_handler.predict_position_during_occlusion(target_id)
            if predicted_pos:
                action_info.update({
                    "action": "predict",
                    "predicted_position": predicted_pos
                })
            else:
                action_info["action"] = "wait"
        
        elif occlusion_info.occlusion_type == OcclusionType.SEVERE:
            # 严重遮挡：降低置信度要求，增加预测权重
            action_info.update({
                "action": "adapt",
                "detection_threshold": 0.2,
                "prediction_weight": 0.8
            })
        
        # 检查是否应该丢弃目标
        if self.occlusion_handler.should_drop_target(target_id):
            action_info["action"] = "drop"
        
        return action_info
    
    def _analyze_motion_level(self, image: np.ndarray) -> float:
        """分析运动水平
        
        Args:
            image: 输入图像
            
        Returns:
            float: 运动水平 (0-1)
        """
        if cv2 is None or not hasattr(self, '_previous_frame'):
            self._previous_frame = image
            return 0.0
        
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                current_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                current_gray = image
            
            if len(self._previous_frame.shape) == 3:
                prev_gray = cv2.cvtColor(self._previous_frame, cv2.COLOR_BGR2GRAY)
            else:
                prev_gray = self._previous_frame
            
            # 调整尺寸
            if current_gray.shape != prev_gray.shape:
                prev_gray = cv2.resize(prev_gray, (current_gray.shape[1], current_gray.shape[0]))
            
            # 计算光流
            flow = cv2.calcOpticalFlowPyrLK(
                prev_gray, current_gray, 
                np.array([[x, y] for x in range(0, current_gray.shape[1], 20) 
                         for y in range(0, current_gray.shape[0], 20)], dtype=np.float32).reshape(-1, 1, 2),
                None
            )[0]
            
            if flow is not None and len(flow) > 0:
                # 计算运动幅度
                motion_magnitudes = np.linalg.norm(flow.reshape(-1, 2), axis=1)
                motion_level = np.mean(motion_magnitudes) / 50.0  # 归一化
            else:
                motion_level = 0.0
            
            self._previous_frame = image
            return np.clip(motion_level, 0.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"运动分析失败: {e}")
            return 0.0
    
    def _analyze_noise_level(self, image: np.ndarray) -> float:
        """分析噪声水平
        
        Args:
            image: 输入图像
            
        Returns:
            float: 噪声水平 (0-1)
        """
        if cv2 is None:
            return 0.0
        
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # 使用拉普拉斯算子检测噪声
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            noise_variance = laplacian.var()
            
            # 归一化噪声水平
            noise_level = min(noise_variance / 1000.0, 1.0)
            
            return noise_level
            
        except Exception as e:
            self.logger.error(f"噪声分析失败: {e}")
            return 0.0
    
    def _calculate_complexity_score(self, lighting_analysis: Dict[str, float], 
                                  crowd_density: float, motion_level: float, noise_level: float) -> float:
        """计算场景复杂度评分
        
        Args:
            lighting_analysis: 光照分析
            crowd_density: 人群密度
            motion_level: 运动水平
            noise_level: 噪声水平
            
        Returns:
            float: 复杂度评分 (0-1)
        """
        # 光照复杂度
        lighting_complexity = 1.0 - lighting_analysis.get("uniformity", 0.5)
        
        # 综合复杂度
        complexity = (
            lighting_complexity * 0.25 +
            crowd_density * 0.35 +
            motion_level * 0.25 +
            noise_level * 0.15
        )
        
        return np.clip(complexity, 0.0, 1.0)
    
    def _estimate_overall_occlusion_level(self) -> float:
        """估计整体遮挡水平
        
        Returns:
            float: 遮挡水平 (0-1)
        """
        # 基于所有目标的遮挡历史估计
        all_occlusions = []
        
        for target_id, history in self.occlusion_handler.occlusion_history.items():
            if history:
                recent_occlusions = [h.occlusion_ratio for h in history[-5:]]  # 最近5次
                all_occlusions.extend(recent_occlusions)
        
        if not all_occlusions:
            return 0.0
        
        return np.mean(all_occlusions)
    
    def _select_adaptation_strategy(self, scene_analysis: SceneAnalysis) -> AdaptationStrategy:
        """选择适应策略
        
        Args:
            scene_analysis: 场景分析
            
        Returns:
            AdaptationStrategy: 适应策略
        """
        # 根据场景复杂度选择策略
        if scene_analysis.complexity_score > 0.8:
            return AdaptationStrategy.CONSERVATIVE
        elif scene_analysis.complexity_score > 0.6:
            return AdaptationStrategy.BALANCED
        elif scene_analysis.complexity_score > 0.3:
            return AdaptationStrategy.AGGRESSIVE
        else:
            return AdaptationStrategy.ADAPTIVE
    
    def get_adaptation_report(self) -> Dict[str, Any]:
        """获取适应报告
        
        Returns:
            Dict[str, Any]: 适应报告
        """
        if not self.scene_history:
            return {"status": "no_data"}
        
        recent_analysis = self.scene_history[-1]
        
        # 统计场景条件
        condition_counts = defaultdict(int)
        for analysis in self.scene_history:
            for condition in analysis.conditions:
                condition_counts[condition.value] += 1
        
        # 计算平均指标
        avg_complexity = np.mean([a.complexity_score for a in self.scene_history])
        avg_crowd_density = np.mean([a.crowd_density for a in self.scene_history])
        avg_lighting = np.mean([a.lighting_score for a in self.scene_history])
        
        return {
            "current_scene": {
                "conditions": [c.value for c in recent_analysis.conditions],
                "complexity_score": recent_analysis.complexity_score,
                "crowd_density": recent_analysis.crowd_density,
                "lighting_score": recent_analysis.lighting_score
            },
            "historical_averages": {
                "complexity_score": avg_complexity,
                "crowd_density": avg_crowd_density,
                "lighting_score": avg_lighting
            },
            "condition_frequency": dict(condition_counts),
            "adaptation_strategy": self.current_strategy.value,
            "lighting_adapter_stats": len(self.lighting_adapter.lighting_history),
            "crowd_analysis": self.multi_person_optimizer.get_crowd_analysis(),
            "recommendations": self._generate_adaptation_recommendations(recent_analysis)
        }
    
    def _generate_adaptation_recommendations(self, scene_analysis: SceneAnalysis) -> List[str]:
        """生成适应建议
        
        Args:
            scene_analysis: 场景分析
            
        Returns:
            List[str]: 建议列表
        """
        recommendations = []
        
        # 光照建议
        if scene_analysis.lighting_score < 0.3:
            recommendations.append("光照不足，建议启用低光增强")
        elif scene_analysis.lighting_score > 0.8:
            recommendations.append("光照过强，建议启用曝光控制")
        
        # 人群建议
        if scene_analysis.crowd_density > 0.7:
            recommendations.append("人群密度高，建议启用保守跟踪策略")
        
        # 复杂度建议
        if scene_analysis.complexity_score > 0.8:
            recommendations.append("场景复杂度高，建议降低跟踪目标数量")
        
        # 遮挡建议
        if scene_analysis.occlusion_level > 0.6:
            recommendations.append("遮挡严重，建议启用预测跟踪")
        
        return recommendations if recommendations else ["当前场景适应良好"]
    
    def cleanup(self):
        """清理资源"""
        try:
            # 清理历史数据
            self.scene_history.clear()
            self.lighting_adapter.lighting_history.clear()
            self.multi_person_optimizer.crowd_history.clear()
            
            # 清理遮挡处理器
            self.occlusion_handler.occlusion_history.clear()
            self.occlusion_handler.recovery_timers.clear()
            self.occlusion_handler.position_history.clear()
            
            self.logger.info("场景适应器资源清理完成")
            
        except Exception as e:
            self.logger.error(f"资源清理失败: {e}")

# 测试代码
if __name__ == "__main__":
    # 创建配置
    config = AdaptationConfig(
        enable_lighting_adaptation=True,
        enable_occlusion_handling=True,
        enable_multi_person_optimization=True,
        analysis_interval=0.5
    )
    
    # 创建适应器
    adapter = SceneAdapter(config)
    
    print("开始场景适应测试...")
    
    # 模拟测试数据
    for i in range(10):
        # 模拟图像
        if i < 3:
            # 正常光照
            image = np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8)
        elif i < 6:
            # 低光照
            image = np.random.randint(20, 80, (480, 640, 3), dtype=np.uint8)
        else:
            # 强光照
            image = np.random.randint(180, 255, (480, 640, 3), dtype=np.uint8)
        
        # 模拟检测结果
        num_persons = np.random.randint(1, 6)
        detections = []
        for j in range(num_persons):
            x = np.random.randint(0, 500)
            y = np.random.randint(0, 350)
            w = np.random.randint(50, 150)
            h = np.random.randint(100, 200)
            conf = np.random.uniform(0.5, 0.95)
            detections.append((x, y, w, h, conf))
        
        # 场景分析
        scene_analysis = adapter.analyze_scene(image, detections)
        
        print(f"\n帧 {i}:")
        print(f"  场景条件: {[c.value for c in scene_analysis.conditions]}")
        print(f"  复杂度: {scene_analysis.complexity_score:.2f}")
        print(f"  人群密度: {scene_analysis.crowd_density:.2f}")
        print(f"  光照评分: {scene_analysis.lighting_score:.2f}")
        
        # 场景适应
        adapted_image, tracking_strategy = adapter.adapt_to_scene(image, scene_analysis)
        print(f"  跟踪策略: {tracking_strategy}")
        
        # 测试遮挡处理
        if detections:
            target_id = 1
            detection_box = detections[0][:4]  # 第一个检测框
            confidence = detections[0][4]
            
            occlusion_result = adapter.handle_target_occlusion(target_id, detection_box, confidence, image)
            print(f"  遮挡处理: {occlusion_result['action']}")
        
        time.sleep(0.1)  # 模拟处理间隔
    
    # 获取适应报告
    report = adapter.get_adaptation_report()
    print("\n=== 适应报告 ===")
    print(f"当前场景条件: {report['current_scene']['conditions']}")
    print(f"平均复杂度: {report['historical_averages']['complexity_score']:.2f}")
    print(f"适应策略: {report['adaptation_strategy']}")
    print(f"建议: {report['recommendations']}")
    
    # 清理资源
    adapter.cleanup()
    print("\n场景适应测试完成！")