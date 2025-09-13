#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自适应算法模块
根据场景动态选择最优跟踪策略
整合多种跟踪算法，实现智能策略切换
"""

import time
import logging
import threading
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import json
import cv2

logger = logging.getLogger(__name__)

class TrackingAlgorithm(Enum):
    """跟踪算法类型"""
    KALMAN_FILTER = "kalman_filter"  # 卡尔曼滤波
    PARTICLE_FILTER = "particle_filter"  # 粒子滤波
    OPTICAL_FLOW = "optical_flow"  # 光流法
    CORRELATION_FILTER = "correlation_filter"  # 相关滤波
    DEEP_SORT = "deep_sort"  # DeepSORT
    BYTE_TRACK = "byte_track"  # ByteTrack
    FAIR_MOT = "fair_mot"  # FairMOT
    CENTERNET = "centernet"  # CenterNet
    YOLO_TRACK = "yolo_track"  # YOLO跟踪
    MULTI_MODAL = "multi_modal"  # 多模态融合

class SceneType(Enum):
    """场景类型"""
    INDOOR = "indoor"  # 室内场景
    OUTDOOR = "outdoor"  # 室外场景
    CROWDED = "crowded"  # 拥挤场景
    SPARSE = "sparse"  # 稀疏场景
    LOW_LIGHT = "low_light"  # 低光照
    HIGH_LIGHT = "high_light"  # 强光照
    DYNAMIC = "dynamic"  # 动态场景
    STATIC = "static"  # 静态场景
    OCCLUDED = "occluded"  # 遮挡场景
    CLEAR = "clear"  # 清晰场景

class PerformanceMetric(Enum):
    """性能指标"""
    ACCURACY = "accuracy"  # 准确率
    PRECISION = "precision"  # 精确率
    RECALL = "recall"  # 召回率
    F1_SCORE = "f1_score"  # F1分数
    SPEED = "speed"  # 速度 (FPS)
    LATENCY = "latency"  # 延迟 (ms)
    MEMORY_USAGE = "memory_usage"  # 内存使用
    CPU_USAGE = "cpu_usage"  # CPU使用率
    GPU_USAGE = "gpu_usage"  # GPU使用率
    POWER_CONSUMPTION = "power_consumption"  # 功耗
    STABILITY = "stability"  # 稳定性
    ROBUSTNESS = "robustness"  # 鲁棒性

@dataclass
class SceneAnalysis:
    """场景分析结果"""
    timestamp: float
    scene_type: SceneType
    confidence: float  # 场景类型置信度 (0-1)
    
    # 场景特征
    lighting_condition: float  # 光照条件 (0-1)
    motion_intensity: float  # 运动强度 (0-1)
    crowd_density: float  # 人群密度 (0-1)
    occlusion_level: float  # 遮挡程度 (0-1)
    background_complexity: float  # 背景复杂度 (0-1)
    noise_level: float  # 噪声水平 (0-1)
    
    # 环境因素
    indoor_probability: float = 0.5  # 室内概率
    weather_condition: str = "unknown"  # 天气条件
    time_of_day: str = "unknown"  # 时间段
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()

@dataclass
class AlgorithmPerformance:
    """算法性能评估"""
    algorithm: TrackingAlgorithm
    scene_type: SceneType
    timestamp: float
    
    # 性能指标
    accuracy: float = 0.0  # 准确率 (0-1)
    precision: float = 0.0  # 精确率 (0-1)
    recall: float = 0.0  # 召回率 (0-1)
    f1_score: float = 0.0  # F1分数 (0-1)
    
    # 效率指标
    fps: float = 0.0  # 帧率
    latency: float = 0.0  # 延迟 (ms)
    memory_usage: float = 0.0  # 内存使用 (MB)
    cpu_usage: float = 0.0  # CPU使用率 (0-100)
    gpu_usage: float = 0.0  # GPU使用率 (0-100)
    power_consumption: float = 0.0  # 功耗 (W)
    
    # 稳定性指标
    stability_score: float = 0.0  # 稳定性分数 (0-1)
    robustness_score: float = 0.0  # 鲁棒性分数 (0-1)
    
    # 综合评分
    overall_score: float = 0.0  # 综合评分 (0-1)
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()
        
        # 计算F1分数
        if self.precision > 0 and self.recall > 0:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        
        # 计算综合评分
        self._calculate_overall_score()
    
    def _calculate_overall_score(self):
        """计算综合评分"""
        # 权重配置
        weights = {
            'accuracy': 0.25,
            'f1_score': 0.20,
            'fps': 0.15,
            'stability': 0.15,
            'robustness': 0.10,
            'efficiency': 0.15  # CPU/GPU/内存效率
        }
        
        # 效率分数 (越低越好，需要归一化)
        efficiency_score = 0.0
        if self.cpu_usage > 0 or self.memory_usage > 0:
            cpu_eff = max(0, 1.0 - self.cpu_usage / 100.0)
            mem_eff = max(0, 1.0 - min(self.memory_usage / 1000.0, 1.0))  # 假设1GB为满分
            efficiency_score = (cpu_eff + mem_eff) / 2
        
        # FPS分数 (归一化到0-1)
        fps_score = min(self.fps / 30.0, 1.0)  # 30FPS为满分
        
        # 综合评分
        self.overall_score = (
            weights['accuracy'] * self.accuracy +
            weights['f1_score'] * self.f1_score +
            weights['fps'] * fps_score +
            weights['stability'] * self.stability_score +
            weights['robustness'] * self.robustness_score +
            weights['efficiency'] * efficiency_score
        )

@dataclass
class AdaptiveConfig:
    """自适应算法配置"""
    # 场景分析配置
    scene_analysis_interval: float = 1.0  # 场景分析间隔 (秒)
    scene_history_size: int = 50  # 场景历史大小
    
    # 算法切换配置
    algorithm_switch_threshold: float = 0.1  # 算法切换阈值
    switch_cooldown: float = 5.0  # 切换冷却时间 (秒)
    min_performance_samples: int = 10  # 最小性能样本数
    
    # 性能评估配置
    performance_window: float = 30.0  # 性能评估窗口 (秒)
    performance_update_interval: float = 2.0  # 性能更新间隔 (秒)
    
    # 学习配置
    enable_online_learning: bool = True  # 启用在线学习
    learning_rate: float = 0.01  # 学习率
    adaptation_sensitivity: float = 0.5  # 适应敏感度
    
    # 算法权重
    algorithm_weights: Dict[TrackingAlgorithm, float] = field(default_factory=lambda: {
        TrackingAlgorithm.KALMAN_FILTER: 1.0,
        TrackingAlgorithm.PARTICLE_FILTER: 1.0,
        TrackingAlgorithm.OPTICAL_FLOW: 1.0,
        TrackingAlgorithm.CORRELATION_FILTER: 1.0,
        TrackingAlgorithm.DEEP_SORT: 1.2,
        TrackingAlgorithm.BYTE_TRACK: 1.1,
        TrackingAlgorithm.MULTI_MODAL: 1.3
    })
    
    # 场景偏好
    scene_preferences: Dict[SceneType, List[TrackingAlgorithm]] = field(default_factory=lambda: {
        SceneType.INDOOR: [TrackingAlgorithm.KALMAN_FILTER, TrackingAlgorithm.OPTICAL_FLOW],
        SceneType.OUTDOOR: [TrackingAlgorithm.DEEP_SORT, TrackingAlgorithm.BYTE_TRACK],
        SceneType.CROWDED: [TrackingAlgorithm.BYTE_TRACK, TrackingAlgorithm.FAIR_MOT],
        SceneType.LOW_LIGHT: [TrackingAlgorithm.PARTICLE_FILTER, TrackingAlgorithm.MULTI_MODAL],
        SceneType.OCCLUDED: [TrackingAlgorithm.PARTICLE_FILTER, TrackingAlgorithm.MULTI_MODAL],
        SceneType.DYNAMIC: [TrackingAlgorithm.OPTICAL_FLOW, TrackingAlgorithm.CORRELATION_FILTER]
    })

class SceneAnalyzer:
    """场景分析器
    
    分析当前场景特征，识别场景类型
    """
    
    def __init__(self, config: AdaptiveConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 场景历史
        self.scene_history = deque(maxlen=config.scene_history_size)
        
        # 分析状态
        self.analyzing = False
        self.analysis_thread = None
        
        # 场景特征提取器
        self.feature_extractors = self._initialize_feature_extractors()
        
        self.logger.info("场景分析器初始化完成")
    
    def _initialize_feature_extractors(self) -> Dict[str, Callable]:
        """初始化特征提取器
        
        Returns:
            Dict[str, Callable]: 特征提取器字典
        """
        return {
            'lighting': self._analyze_lighting,
            'motion': self._analyze_motion,
            'crowd': self._analyze_crowd_density,
            'occlusion': self._analyze_occlusion,
            'background': self._analyze_background_complexity,
            'noise': self._analyze_noise_level
        }
    
    def start_analysis(self):
        """开始场景分析"""
        if self.analyzing:
            return
        
        self.analyzing = True
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.analysis_thread.start()
        self.logger.info("场景分析已启动")
    
    def stop_analysis(self):
        """停止场景分析"""
        self.analyzing = False
        if self.analysis_thread:
            self.analysis_thread.join(timeout=2.0)
        self.logger.info("场景分析已停止")
    
    def _analysis_loop(self):
        """分析循环"""
        while self.analyzing:
            try:
                # 这里应该从视频流获取当前帧
                # 为了演示，我们创建一个模拟的分析结果
                analysis = self._simulate_scene_analysis()
                self.scene_history.append(analysis)
                
                time.sleep(self.config.scene_analysis_interval)
            except Exception as e:
                self.logger.error(f"场景分析异常: {e}")
                time.sleep(self.config.scene_analysis_interval)
    
    def analyze_frame(self, frame: np.ndarray, detections: List[Dict] = None) -> SceneAnalysis:
        """分析单帧
        
        Args:
            frame: 输入帧
            detections: 检测结果
            
        Returns:
            SceneAnalysis: 场景分析结果
        """
        try:
            # 提取场景特征
            features = {}
            for name, extractor in self.feature_extractors.items():
                features[name] = extractor(frame, detections)
            
            # 分析场景类型
            scene_type, confidence = self._classify_scene(features)
            
            # 创建分析结果
            analysis = SceneAnalysis(
                timestamp=time.time(),
                scene_type=scene_type,
                confidence=confidence,
                lighting_condition=features['lighting'],
                motion_intensity=features['motion'],
                crowd_density=features['crowd'],
                occlusion_level=features['occlusion'],
                background_complexity=features['background'],
                noise_level=features['noise']
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"帧分析失败: {e}")
            return self._create_default_analysis()
    
    def _analyze_lighting(self, frame: np.ndarray, detections: List[Dict] = None) -> float:
        """分析光照条件
        
        Args:
            frame: 输入帧
            detections: 检测结果
            
        Returns:
            float: 光照条件 (0-1)
        """
        try:
            # 转换为灰度图
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # 计算平均亮度
            mean_brightness = np.mean(gray) / 255.0
            
            # 计算亮度分布
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_norm = hist / (gray.shape[0] * gray.shape[1])
            
            # 分析亮度分布特征
            low_light_ratio = np.sum(hist_norm[:64])  # 低亮度像素比例
            high_light_ratio = np.sum(hist_norm[192:])  # 高亮度像素比例
            
            # 综合评估光照条件
            if low_light_ratio > 0.6:
                lighting_score = 0.2  # 低光照
            elif high_light_ratio > 0.3:
                lighting_score = 0.9  # 强光照
            else:
                lighting_score = mean_brightness  # 正常光照
            
            return np.clip(lighting_score, 0.0, 1.0)
            
        except Exception as e:
            self.logger.debug(f"光照分析失败: {e}")
            return 0.5
    
    def _analyze_motion(self, frame: np.ndarray, detections: List[Dict] = None) -> float:
        """分析运动强度
        
        Args:
            frame: 输入帧
            detections: 检测结果
            
        Returns:
            float: 运动强度 (0-1)
        """
        try:
            # 如果有检测结果，基于检测框分析运动
            if detections and len(detections) > 0:
                motion_scores = []
                for det in detections:
                    # 假设检测结果包含速度信息
                    velocity = det.get('velocity', 0.0)
                    motion_scores.append(min(velocity / 10.0, 1.0))  # 归一化
                
                return np.mean(motion_scores) if motion_scores else 0.0
            
            # 基于帧差分析运动
            # 这里需要保存前一帧进行比较
            # 为了简化，返回模拟值
            return np.random.uniform(0.2, 0.8)
            
        except Exception as e:
            self.logger.debug(f"运动分析失败: {e}")
            return 0.3
    
    def _analyze_crowd_density(self, frame: np.ndarray, detections: List[Dict] = None) -> float:
        """分析人群密度
        
        Args:
            frame: 输入帧
            detections: 检测结果
            
        Returns:
            float: 人群密度 (0-1)
        """
        try:
            if detections:
                # 基于检测数量计算密度
                person_count = len([d for d in detections if d.get('class', '') == 'person'])
                
                # 计算检测框覆盖面积
                frame_area = frame.shape[0] * frame.shape[1]
                total_box_area = 0
                
                for det in detections:
                    if det.get('class', '') == 'person':
                        bbox = det.get('bbox', [0, 0, 0, 0])
                        box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        total_box_area += box_area
                
                coverage_ratio = total_box_area / frame_area
                
                # 综合人数和覆盖率
                density_score = min((person_count / 10.0 + coverage_ratio) / 2.0, 1.0)
                return density_score
            
            return 0.1  # 默认低密度
            
        except Exception as e:
            self.logger.debug(f"人群密度分析失败: {e}")
            return 0.1
    
    def _analyze_occlusion(self, frame: np.ndarray, detections: List[Dict] = None) -> float:
        """分析遮挡程度
        
        Args:
            frame: 输入帧
            detections: 检测结果
            
        Returns:
            float: 遮挡程度 (0-1)
        """
        try:
            if detections and len(detections) > 1:
                # 计算检测框重叠
                overlap_count = 0
                total_pairs = 0
                
                for i in range(len(detections)):
                    for j in range(i + 1, len(detections)):
                        bbox1 = detections[i].get('bbox', [0, 0, 0, 0])
                        bbox2 = detections[j].get('bbox', [0, 0, 0, 0])
                        
                        # 计算IoU
                        iou = self._calculate_iou(bbox1, bbox2)
                        if iou > 0.1:  # 有重叠
                            overlap_count += 1
                        total_pairs += 1
                
                if total_pairs > 0:
                    occlusion_ratio = overlap_count / total_pairs
                    return min(occlusion_ratio * 2.0, 1.0)  # 放大效果
            
            return 0.1  # 默认低遮挡
            
        except Exception as e:
            self.logger.debug(f"遮挡分析失败: {e}")
            return 0.1
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """计算IoU
        
        Args:
            bbox1: 边界框1 [x1, y1, x2, y2]
            bbox2: 边界框2 [x1, y1, x2, y2]
            
        Returns:
            float: IoU值
        """
        # 计算交集
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # 计算并集
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _analyze_background_complexity(self, frame: np.ndarray, detections: List[Dict] = None) -> float:
        """分析背景复杂度
        
        Args:
            frame: 输入帧
            detections: 检测结果
            
        Returns:
            float: 背景复杂度 (0-1)
        """
        try:
            # 转换为灰度图
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # 计算梯度
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # 计算纹理复杂度
            complexity_score = np.mean(gradient_magnitude) / 255.0
            
            return np.clip(complexity_score, 0.0, 1.0)
            
        except Exception as e:
            self.logger.debug(f"背景复杂度分析失败: {e}")
            return 0.5
    
    def _analyze_noise_level(self, frame: np.ndarray, detections: List[Dict] = None) -> float:
        """分析噪声水平
        
        Args:
            frame: 输入帧
            detections: 检测结果
            
        Returns:
            float: 噪声水平 (0-1)
        """
        try:
            # 转换为灰度图
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # 使用拉普拉斯算子检测噪声
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            noise_variance = laplacian.var()
            
            # 归一化噪声水平
            noise_score = min(noise_variance / 1000.0, 1.0)
            
            return noise_score
            
        except Exception as e:
            self.logger.debug(f"噪声分析失败: {e}")
            return 0.2
    
    def _classify_scene(self, features: Dict[str, float]) -> Tuple[SceneType, float]:
        """分类场景类型
        
        Args:
            features: 场景特征
            
        Returns:
            Tuple[SceneType, float]: (场景类型, 置信度)
        """
        try:
            # 基于规则的场景分类
            lighting = features['lighting']
            motion = features['motion']
            crowd = features['crowd']
            occlusion = features['occlusion']
            background = features['background']
            noise = features['noise']
            
            # 场景类型评分
            scene_scores = {}
            
            # 低光照场景
            if lighting < 0.3:
                scene_scores[SceneType.LOW_LIGHT] = 0.8 + (0.3 - lighting)
            
            # 强光照场景
            if lighting > 0.8:
                scene_scores[SceneType.HIGH_LIGHT] = 0.7 + (lighting - 0.8)
            
            # 拥挤场景
            if crowd > 0.6:
                scene_scores[SceneType.CROWDED] = 0.7 + (crowd - 0.6) * 0.5
            else:
                scene_scores[SceneType.SPARSE] = 0.6 + (0.6 - crowd) * 0.3
            
            # 遮挡场景
            if occlusion > 0.5:
                scene_scores[SceneType.OCCLUDED] = 0.7 + (occlusion - 0.5)
            else:
                scene_scores[SceneType.CLEAR] = 0.6 + (0.5 - occlusion) * 0.4
            
            # 动态场景
            if motion > 0.6:
                scene_scores[SceneType.DYNAMIC] = 0.7 + (motion - 0.6) * 0.5
            else:
                scene_scores[SceneType.STATIC] = 0.6 + (0.6 - motion) * 0.3
            
            # 室内/室外判断（基于光照和背景复杂度）
            if lighting > 0.4 and background > 0.6:
                scene_scores[SceneType.OUTDOOR] = 0.6 + (background - 0.6) * 0.5
            else:
                scene_scores[SceneType.INDOOR] = 0.6 + max(0, 0.6 - background) * 0.3
            
            # 选择最高分的场景类型
            if scene_scores:
                best_scene = max(scene_scores.items(), key=lambda x: x[1])
                return best_scene[0], min(best_scene[1], 1.0)
            else:
                return SceneType.INDOOR, 0.5
            
        except Exception as e:
            self.logger.error(f"场景分类失败: {e}")
            return SceneType.INDOOR, 0.3
    
    def _simulate_scene_analysis(self) -> SceneAnalysis:
        """模拟场景分析（用于测试）
        
        Returns:
            SceneAnalysis: 模拟的场景分析结果
        """
        # 随机生成场景特征
        features = {
            'lighting': np.random.uniform(0.2, 0.9),
            'motion': np.random.uniform(0.1, 0.8),
            'crowd': np.random.uniform(0.0, 0.7),
            'occlusion': np.random.uniform(0.0, 0.6),
            'background': np.random.uniform(0.3, 0.9),
            'noise': np.random.uniform(0.1, 0.4)
        }
        
        scene_type, confidence = self._classify_scene(features)
        
        return SceneAnalysis(
            timestamp=time.time(),
            scene_type=scene_type,
            confidence=confidence,
            lighting_condition=features['lighting'],
            motion_intensity=features['motion'],
            crowd_density=features['crowd'],
            occlusion_level=features['occlusion'],
            background_complexity=features['background'],
            noise_level=features['noise']
        )
    
    def _create_default_analysis(self) -> SceneAnalysis:
        """创建默认场景分析
        
        Returns:
            SceneAnalysis: 默认场景分析
        """
        return SceneAnalysis(
            timestamp=time.time(),
            scene_type=SceneType.INDOOR,
            confidence=0.5,
            lighting_condition=0.5,
            motion_intensity=0.3,
            crowd_density=0.2,
            occlusion_level=0.1,
            background_complexity=0.5,
            noise_level=0.2
        )
    
    def get_current_scene(self) -> Optional[SceneAnalysis]:
        """获取当前场景分析
        
        Returns:
            Optional[SceneAnalysis]: 当前场景分析
        """
        if self.scene_history:
            return self.scene_history[-1]
        return None
    
    def get_scene_trend(self, duration: float = 30.0) -> Dict[SceneType, float]:
        """获取场景趋势
        
        Args:
            duration: 时间窗口 (秒)
            
        Returns:
            Dict[SceneType, float]: 场景类型分布
        """
        if not self.scene_history:
            return {}
        
        current_time = time.time()
        recent_scenes = [
            s for s in self.scene_history 
            if current_time - s.timestamp <= duration
        ]
        
        if not recent_scenes:
            return {}
        
        # 统计场景类型分布
        scene_counts = defaultdict(int)
        for scene in recent_scenes:
            scene_counts[scene.scene_type] += 1
        
        total_count = len(recent_scenes)
        scene_distribution = {
            scene_type: count / total_count 
            for scene_type, count in scene_counts.items()
        }
        
        return scene_distribution

class PerformanceEvaluator:
    """性能评估器
    
    评估不同算法在不同场景下的性能
    """
    
    def __init__(self, config: AdaptiveConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 性能历史
        self.performance_history: Dict[TrackingAlgorithm, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        
        # 性能统计
        self.performance_stats: Dict[Tuple[TrackingAlgorithm, SceneType], List[AlgorithmPerformance]] = defaultdict(list)
        
        # 评估状态
        self.evaluating = False
        self.evaluation_thread = None
        
        self.logger.info("性能评估器初始化完成")
    
    def start_evaluation(self):
        """开始性能评估"""
        if self.evaluating:
            return
        
        self.evaluating = True
        self.evaluation_thread = threading.Thread(target=self._evaluation_loop, daemon=True)
        self.evaluation_thread.start()
        self.logger.info("性能评估已启动")
    
    def stop_evaluation(self):
        """停止性能评估"""
        self.evaluating = False
        if self.evaluation_thread:
            self.evaluation_thread.join(timeout=2.0)
        self.logger.info("性能评估已停止")
    
    def _evaluation_loop(self):
        """评估循环"""
        while self.evaluating:
            try:
                # 更新性能统计
                self._update_performance_stats()
                time.sleep(self.config.performance_update_interval)
            except Exception as e:
                self.logger.error(f"性能评估异常: {e}")
                time.sleep(self.config.performance_update_interval)
    
    def record_performance(self, algorithm: TrackingAlgorithm, scene_type: SceneType, 
                         performance_data: Dict[str, float]):
        """记录算法性能
        
        Args:
            algorithm: 跟踪算法
            scene_type: 场景类型
            performance_data: 性能数据
        """
        try:
            # 创建性能记录
            performance = AlgorithmPerformance(
                algorithm=algorithm,
                scene_type=scene_type,
                timestamp=time.time(),
                **performance_data
            )
            
            # 添加到历史记录
            self.performance_history[algorithm].append(performance)
            
            # 添加到统计数据
            key = (algorithm, scene_type)
            self.performance_stats[key].append(performance)
            
            # 限制统计数据大小
            if len(self.performance_stats[key]) > 100:
                self.performance_stats[key] = self.performance_stats[key][-100:]
            
            self.logger.debug(f"记录性能: {algorithm.value} in {scene_type.value}, 综合评分: {performance.overall_score:.3f}")
            
        except Exception as e:
            self.logger.error(f"性能记录失败: {e}")
    
    def _update_performance_stats(self):
        """更新性能统计"""
        try:
            # 清理过期数据
            current_time = time.time()
            window = self.config.performance_window
            
            for algorithm in self.performance_history:
                # 移除过期记录
                while (self.performance_history[algorithm] and 
                       current_time - self.performance_history[algorithm][0].timestamp > window):
                    self.performance_history[algorithm].popleft()
            
        except Exception as e:
            self.logger.error(f"性能统计更新失败: {e}")
    
    def get_algorithm_performance(self, algorithm: TrackingAlgorithm, 
                                scene_type: Optional[SceneType] = None) -> Optional[AlgorithmPerformance]:
        """获取算法性能
        
        Args:
            algorithm: 跟踪算法
            scene_type: 场景类型（可选）
            
        Returns:
            Optional[AlgorithmPerformance]: 平均性能
        """
        try:
            if scene_type:
                # 获取特定场景下的性能
                key = (algorithm, scene_type)
                performances = self.performance_stats.get(key, [])
            else:
                # 获取所有场景下的性能
                performances = list(self.performance_history[algorithm])
            
            if not performances:
                return None
            
            # 计算平均性能
            avg_performance = self._calculate_average_performance(performances)
            return avg_performance
            
        except Exception as e:
            self.logger.error(f"获取算法性能失败: {e}")
            return None
    
    def _calculate_average_performance(self, performances: List[AlgorithmPerformance]) -> AlgorithmPerformance:
        """计算平均性能
        
        Args:
            performances: 性能记录列表
            
        Returns:
            AlgorithmPerformance: 平均性能
        """
        if not performances:
            raise ValueError("性能记录列表为空")
        
        # 计算各项指标的平均值
        avg_data = {
            'accuracy': np.mean([p.accuracy for p in performances]),
            'precision': np.mean([p.precision for p in performances]),
            'recall': np.mean([p.recall for p in performances]),
            'fps': np.mean([p.fps for p in performances]),
            'latency': np.mean([p.latency for p in performances]),
            'memory_usage': np.mean([p.memory_usage for p in performances]),
            'cpu_usage': np.mean([p.cpu_usage for p in performances]),
            'gpu_usage': np.mean([p.gpu_usage for p in performances]),
            'power_consumption': np.mean([p.power_consumption for p in performances]),
            'stability_score': np.mean([p.stability_score for p in performances]),
            'robustness_score': np.mean([p.robustness_score for p in performances])
        }
        
        return AlgorithmPerformance(
            algorithm=performances[0].algorithm,
            scene_type=performances[0].scene_type,
            timestamp=time.time(),
            **avg_data
        )
    
    def get_best_algorithm(self, scene_type: SceneType, 
                          metric: PerformanceMetric = PerformanceMetric.F1_SCORE) -> Optional[TrackingAlgorithm]:
        """获取最佳算法
        
        Args:
            scene_type: 场景类型
            metric: 评估指标
            
        Returns:
            Optional[TrackingAlgorithm]: 最佳算法
        """
        try:
            algorithm_scores = {}
            
            for algorithm in TrackingAlgorithm:
                performance = self.get_algorithm_performance(algorithm, scene_type)
                if performance:
                    # 根据指标获取分数
                    score = self._get_metric_score(performance, metric)
                    
                    # 应用算法权重
                    weight = self.config.algorithm_weights.get(algorithm, 1.0)
                    algorithm_scores[algorithm] = score * weight
            
            if algorithm_scores:
                best_algorithm = max(algorithm_scores.items(), key=lambda x: x[1])
                return best_algorithm[0]
            
            return None
            
        except Exception as e:
            self.logger.error(f"获取最佳算法失败: {e}")
            return None
    
    def _get_metric_score(self, performance: AlgorithmPerformance, metric: PerformanceMetric) -> float:
        """获取指标分数
        
        Args:
            performance: 算法性能
            metric: 性能指标
            
        Returns:
            float: 指标分数
        """
        metric_map = {
            PerformanceMetric.ACCURACY: performance.accuracy,
            PerformanceMetric.PRECISION: performance.precision,
            PerformanceMetric.RECALL: performance.recall,
            PerformanceMetric.F1_SCORE: performance.f1_score,
            PerformanceMetric.SPEED: min(performance.fps / 30.0, 1.0),  # 归一化到30FPS
            PerformanceMetric.LATENCY: max(0, 1.0 - performance.latency / 100.0),  # 延迟越低越好
            PerformanceMetric.MEMORY_USAGE: max(0, 1.0 - performance.memory_usage / 1000.0),  # 内存越低越好
            PerformanceMetric.CPU_USAGE: max(0, 1.0 - performance.cpu_usage / 100.0),  # CPU越低越好
            PerformanceMetric.STABILITY: performance.stability_score,
            PerformanceMetric.ROBUSTNESS: performance.robustness_score
        }
        
        return metric_map.get(metric, performance.overall_score)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要
        
        Returns:
            Dict[str, Any]: 性能摘要
        """
        summary = {
            'total_algorithms': len(TrackingAlgorithm),
            'evaluated_algorithms': len(self.performance_history),
            'total_records': sum(len(records) for records in self.performance_history.values()),
            'algorithm_performance': {},
            'scene_performance': {}
        }
        
        # 算法性能摘要
        for algorithm in TrackingAlgorithm:
            performance = self.get_algorithm_performance(algorithm)
            if performance:
                summary['algorithm_performance'][algorithm.value] = {
                    'overall_score': performance.overall_score,
                    'accuracy': performance.accuracy,
                    'fps': performance.fps,
                    'stability': performance.stability_score
                }
        
        # 场景性能摘要
        for scene_type in SceneType:
            best_algorithm = self.get_best_algorithm(scene_type)
            if best_algorithm:
                summary['scene_performance'][scene_type.value] = {
                    'best_algorithm': best_algorithm.value,
                    'performance': self.get_algorithm_performance(best_algorithm, scene_type).overall_score
                }
        
        return summary

class AdaptiveAlgorithm:
    """自适应算法主类
    
    整合场景分析和性能评估，实现智能算法切换
    """
    
    def __init__(self, config: AdaptiveConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化组件
        self.scene_analyzer = SceneAnalyzer(config)
        self.performance_evaluator = PerformanceEvaluator(config)
        
        # 当前状态
        self.current_algorithm = TrackingAlgorithm.KALMAN_FILTER
        self.last_switch_time = 0.0
        
        # 算法实例（这里应该是实际的跟踪算法实例）
        self.algorithm_instances = self._initialize_algorithms()
        
        # 自适应状态
        self.adaptive_enabled = True
        self.learning_enabled = config.enable_online_learning
        
        # 运行状态
        self.running = False
        self.adaptation_thread = None
        
        self.logger.info(f"自适应算法初始化完成，当前算法: {self.current_algorithm.value}")
    
    def _initialize_algorithms(self) -> Dict[TrackingAlgorithm, Any]:
        """初始化算法实例
        
        Returns:
            Dict[TrackingAlgorithm, Any]: 算法实例字典
        """
        # 这里应该初始化实际的跟踪算法实例
        # 为了演示，我们创建模拟的算法实例
        algorithms = {}
        
        for algorithm in TrackingAlgorithm:
            # 创建模拟的算法实例
            algorithms[algorithm] = self._create_mock_algorithm(algorithm)
        
        return algorithms
    
    def _create_mock_algorithm(self, algorithm: TrackingAlgorithm) -> Dict[str, Any]:
        """创建模拟算法实例
        
        Args:
            algorithm: 算法类型
            
        Returns:
            Dict[str, Any]: 模拟算法实例
        """
        return {
            'name': algorithm.value,
            'initialized': True,
            'config': {},
            'performance_baseline': {
                'accuracy': np.random.uniform(0.7, 0.95),
                'fps': np.random.uniform(15, 60),
                'memory_usage': np.random.uniform(100, 500)
            }
        }
    
    def start(self):
        """启动自适应算法"""
        if self.running:
            return
        
        self.running = True
        
        # 启动组件
        self.scene_analyzer.start_analysis()
        self.performance_evaluator.start_evaluation()
        
        # 启动自适应线程
        self.adaptation_thread = threading.Thread(target=self._adaptation_loop, daemon=True)
        self.adaptation_thread.start()
        
        self.logger.info("自适应算法已启动")
    
    def stop(self):
        """停止自适应算法"""
        if not self.running:
            return
        
        self.running = False
        
        # 停止组件
        self.scene_analyzer.stop_analysis()
        self.performance_evaluator.stop_evaluation()
        
        # 等待自适应线程结束
        if self.adaptation_thread:
            self.adaptation_thread.join(timeout=3.0)
        
        self.logger.info("自适应算法已停止")
    
    def _adaptation_loop(self):
        """自适应循环"""
        while self.running:
            try:
                if self.adaptive_enabled:
                    self._perform_adaptation()
                
                time.sleep(self.config.scene_analysis_interval)
                
            except Exception as e:
                self.logger.error(f"自适应循环异常: {e}")
                time.sleep(self.config.scene_analysis_interval)
    
    def _perform_adaptation(self):
        """执行自适应"""
        try:
            # 获取当前场景
            current_scene = self.scene_analyzer.get_current_scene()
            if not current_scene:
                return
            
            # 检查是否需要切换算法
            optimal_algorithm = self._determine_optimal_algorithm(current_scene)
            
            if optimal_algorithm and optimal_algorithm != self.current_algorithm:
                # 检查切换条件
                if self._should_switch_algorithm(optimal_algorithm, current_scene):
                    self._switch_algorithm(optimal_algorithm, current_scene)
            
            # 在线学习
            if self.learning_enabled:
                self._perform_online_learning(current_scene)
                
        except Exception as e:
            self.logger.error(f"自适应执行失败: {e}")
    
    def _determine_optimal_algorithm(self, scene: SceneAnalysis) -> Optional[TrackingAlgorithm]:
        """确定最优算法
        
        Args:
            scene: 场景分析结果
            
        Returns:
            Optional[TrackingAlgorithm]: 最优算法
        """
        try:
            # 基于性能评估选择最佳算法
            best_algorithm = self.performance_evaluator.get_best_algorithm(scene.scene_type)
            
            if best_algorithm:
                return best_algorithm
            
            # 如果没有性能数据，使用场景偏好
            preferred_algorithms = self.config.scene_preferences.get(scene.scene_type, [])
            if preferred_algorithms:
                return preferred_algorithms[0]
            
            # 默认算法选择逻辑
            return self._get_default_algorithm_for_scene(scene)
            
        except Exception as e:
            self.logger.error(f"最优算法确定失败: {e}")
            return None
    
    def _get_default_algorithm_for_scene(self, scene: SceneAnalysis) -> TrackingAlgorithm:
        """获取场景的默认算法
        
        Args:
            scene: 场景分析结果
            
        Returns:
            TrackingAlgorithm: 默认算法
        """
        # 基于场景特征选择算法
        if scene.lighting_condition < 0.3:  # 低光照
            return TrackingAlgorithm.PARTICLE_FILTER
        elif scene.crowd_density > 0.6:  # 拥挤场景
            return TrackingAlgorithm.BYTE_TRACK
        elif scene.motion_intensity > 0.7:  # 高运动
            return TrackingAlgorithm.OPTICAL_FLOW
        elif scene.occlusion_level > 0.5:  # 高遮挡
            return TrackingAlgorithm.MULTI_MODAL
        else:  # 一般场景
            return TrackingAlgorithm.DEEP_SORT
    
    def _should_switch_algorithm(self, target_algorithm: TrackingAlgorithm, scene: SceneAnalysis) -> bool:
        """判断是否应该切换算法
        
        Args:
            target_algorithm: 目标算法
            scene: 场景分析
            
        Returns:
            bool: 是否应该切换
        """
        current_time = time.time()
        
        # 检查冷却时间
        if current_time - self.last_switch_time < self.config.switch_cooldown:
            return False
        
        # 检查性能差异
        current_performance = self.performance_evaluator.get_algorithm_performance(
            self.current_algorithm, scene.scene_type
        )
        target_performance = self.performance_evaluator.get_algorithm_performance(
            target_algorithm, scene.scene_type
        )
        
        if current_performance and target_performance:
            performance_diff = target_performance.overall_score - current_performance.overall_score
            
            # 只有在性能提升超过阈值时才切换
            if performance_diff > self.config.algorithm_switch_threshold:
                return True
        
        # 如果没有足够的性能数据，基于场景置信度决定
        return scene.confidence > 0.8
    
    def _switch_algorithm(self, target_algorithm: TrackingAlgorithm, scene: SceneAnalysis):
        """切换算法
        
        Args:
            target_algorithm: 目标算法
            scene: 场景分析
        """
        try:
            previous_algorithm = self.current_algorithm
            
            # 执行算法切换
            self.current_algorithm = target_algorithm
            self.last_switch_time = time.time()
            
            self.logger.info(
                f"算法切换: {previous_algorithm.value} -> {target_algorithm.value}, "
                f"场景: {scene.scene_type.value}, 置信度: {scene.confidence:.3f}"
            )
            
            # 通知算法切换事件
            self._on_algorithm_switched(previous_algorithm, target_algorithm, scene)
            
        except Exception as e:
            self.logger.error(f"算法切换失败: {e}")
    
    def _on_algorithm_switched(self, previous: TrackingAlgorithm, current: TrackingAlgorithm, scene: SceneAnalysis):
        """算法切换事件处理
        
        Args:
            previous: 之前的算法
            current: 当前算法
            scene: 场景分析
        """
        # 这里可以添加算法切换后的处理逻辑
        # 例如：重新初始化跟踪器、调整参数等
        pass
    
    def _perform_online_learning(self, scene: SceneAnalysis):
        """执行在线学习
        
        Args:
            scene: 场景分析
        """
        try:
            # 在线学习逻辑
            # 这里可以实现算法参数的自适应调整
            # 基于当前性能反馈调整算法权重等
            
            # 示例：调整算法权重
            current_performance = self.performance_evaluator.get_algorithm_performance(
                self.current_algorithm, scene.scene_type
            )
            
            if current_performance:
                # 基于性能调整权重
                current_weight = self.config.algorithm_weights.get(self.current_algorithm, 1.0)
                
                if current_performance.overall_score > 0.8:
                    # 性能好，增加权重
                    new_weight = current_weight * (1 + self.config.learning_rate)
                elif current_performance.overall_score < 0.5:
                    # 性能差，减少权重
                    new_weight = current_weight * (1 - self.config.learning_rate)
                else:
                    new_weight = current_weight
                
                # 限制权重范围
                new_weight = np.clip(new_weight, 0.1, 2.0)
                self.config.algorithm_weights[self.current_algorithm] = new_weight
                
                self.logger.debug(
                    f"在线学习更新权重: {self.current_algorithm.value} "
                    f"{current_weight:.3f} -> {new_weight:.3f}"
                )
            
        except Exception as e:
            self.logger.error(f"在线学习失败: {e}")
    
    def process_frame(self, frame: np.ndarray, detections: List[Dict] = None) -> Tuple[List[Dict], Dict[str, Any]]:
        """处理帧
        
        Args:
            frame: 输入帧
            detections: 检测结果
            
        Returns:
            Tuple[List[Dict], Dict[str, Any]]: (跟踪结果, 元信息)
        """
        try:
            start_time = time.time()
            
            # 场景分析
            scene_analysis = self.scene_analyzer.analyze_frame(frame, detections)
            
            # 使用当前算法处理帧
            tracking_results = self._process_with_current_algorithm(frame, detections)
            
            # 计算处理时间
            processing_time = (time.time() - start_time) * 1000  # ms
            
            # 记录性能
            self._record_frame_performance(scene_analysis, processing_time, tracking_results)
            
            # 构建元信息
            meta_info = {
                'algorithm': self.current_algorithm.value,
                'scene_type': scene_analysis.scene_type.value,
                'scene_confidence': scene_analysis.confidence,
                'processing_time': processing_time,
                'frame_timestamp': time.time()
            }
            
            return tracking_results, meta_info
            
        except Exception as e:
            self.logger.error(f"帧处理失败: {e}")
            return [], {'error': str(e)}
    
    def _process_with_current_algorithm(self, frame: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """使用当前算法处理帧
        
        Args:
            frame: 输入帧
            detections: 检测结果
            
        Returns:
            List[Dict]: 跟踪结果
        """
        # 这里应该调用实际的跟踪算法
        # 为了演示，我们返回模拟的跟踪结果
        
        if not detections:
            return []
        
        tracking_results = []
        for i, detection in enumerate(detections):
            track_result = {
                'track_id': i + 1,
                'bbox': detection.get('bbox', [0, 0, 100, 100]),
                'confidence': detection.get('confidence', 0.8),
                'class': detection.get('class', 'person'),
                'algorithm': self.current_algorithm.value,
                'timestamp': time.time()
            }
            tracking_results.append(track_result)
        
        return tracking_results
    
    def _record_frame_performance(self, scene: SceneAnalysis, processing_time: float, results: List[Dict]):
        """记录帧处理性能
        
        Args:
            scene: 场景分析
            processing_time: 处理时间 (ms)
            results: 跟踪结果
        """
        try:
            # 计算性能指标
            fps = 1000.0 / processing_time if processing_time > 0 else 0
            
            # 模拟其他性能指标
            performance_data = {
                'accuracy': np.random.uniform(0.7, 0.95),
                'precision': np.random.uniform(0.7, 0.9),
                'recall': np.random.uniform(0.6, 0.9),
                'fps': fps,
                'latency': processing_time,
                'memory_usage': np.random.uniform(100, 300),
                'cpu_usage': np.random.uniform(20, 80),
                'stability_score': np.random.uniform(0.7, 0.95),
                'robustness_score': np.random.uniform(0.6, 0.9)
            }
            
            # 记录性能
            self.performance_evaluator.record_performance(
                self.current_algorithm, scene.scene_type, performance_data
            )
            
        except Exception as e:
            self.logger.debug(f"性能记录失败: {e}")
    
    def get_current_algorithm(self) -> TrackingAlgorithm:
        """获取当前算法
        
        Returns:
            TrackingAlgorithm: 当前算法
        """
        return self.current_algorithm
    
    def set_algorithm(self, algorithm: TrackingAlgorithm, force: bool = False):
        """设置算法
        
        Args:
            algorithm: 目标算法
            force: 是否强制切换
        """
        if force or algorithm != self.current_algorithm:
            previous = self.current_algorithm
            self.current_algorithm = algorithm
            self.last_switch_time = time.time()
            
            self.logger.info(f"手动切换算法: {previous.value} -> {algorithm.value}")
    
    def enable_adaptation(self, enabled: bool = True):
        """启用/禁用自适应
        
        Args:
            enabled: 是否启用
        """
        self.adaptive_enabled = enabled
        self.logger.info(f"自适应{'启用' if enabled else '禁用'}")
    
    def enable_learning(self, enabled: bool = True):
        """启用/禁用在线学习
        
        Args:
            enabled: 是否启用
        """
        self.learning_enabled = enabled
        self.logger.info(f"在线学习{'启用' if enabled else '禁用'}")
    
    def get_adaptation_status(self) -> Dict[str, Any]:
        """获取自适应状态
        
        Returns:
            Dict[str, Any]: 自适应状态
        """
        current_scene = self.scene_analyzer.get_current_scene()
        
        status = {
            'running': self.running,
            'adaptive_enabled': self.adaptive_enabled,
            'learning_enabled': self.learning_enabled,
            'current_algorithm': self.current_algorithm.value,
            'last_switch_time': self.last_switch_time,
            'current_scene': {
                'type': current_scene.scene_type.value if current_scene else 'unknown',
                'confidence': current_scene.confidence if current_scene else 0.0
            } if current_scene else None,
            'performance_summary': self.performance_evaluator.get_performance_summary()
        }
        
        return status
    
    def save_config(self, filepath: str):
        """保存配置
        
        Args:
            filepath: 配置文件路径
        """
        try:
            config_data = {
                'scene_analysis_interval': self.config.scene_analysis_interval,
                'algorithm_switch_threshold': self.config.algorithm_switch_threshold,
                'switch_cooldown': self.config.switch_cooldown,
                'enable_online_learning': self.config.enable_online_learning,
                'learning_rate': self.config.learning_rate,
                'algorithm_weights': {
                    alg.value: weight for alg, weight in self.config.algorithm_weights.items()
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"配置已保存到: {filepath}")
            
        except Exception as e:
            self.logger.error(f"配置保存失败: {e}")
    
    def load_config(self, filepath: str):
        """加载配置
        
        Args:
            filepath: 配置文件路径
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 更新配置
            self.config.scene_analysis_interval = config_data.get(
                'scene_analysis_interval', self.config.scene_analysis_interval
            )
            self.config.algorithm_switch_threshold = config_data.get(
                'algorithm_switch_threshold', self.config.algorithm_switch_threshold
            )
            self.config.switch_cooldown = config_data.get(
                'switch_cooldown', self.config.switch_cooldown
            )
            self.config.enable_online_learning = config_data.get(
                'enable_online_learning', self.config.enable_online_learning
            )
            self.config.learning_rate = config_data.get(
                'learning_rate', self.config.learning_rate
            )
            
            # 更新算法权重
            if 'algorithm_weights' in config_data:
                for alg_name, weight in config_data['algorithm_weights'].items():
                    try:
                        algorithm = TrackingAlgorithm(alg_name)
                        self.config.algorithm_weights[algorithm] = weight
                    except ValueError:
                        self.logger.warning(f"未知算法类型: {alg_name}")
            
            self.logger.info(f"配置已从 {filepath} 加载")
            
        except Exception as e:
            self.logger.error(f"配置加载失败: {e}")

# 测试代码
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=== 自适应算法测试 ===")
    
    try:
        # 创建配置
        config = AdaptiveConfig(
            scene_analysis_interval=2.0,
            algorithm_switch_threshold=0.1,
            switch_cooldown=3.0,
            enable_online_learning=True,
            learning_rate=0.02
        )
        
        print(f"配置创建完成: 分析间隔={config.scene_analysis_interval}s")
        
        # 创建自适应算法
        adaptive_algo = AdaptiveAlgorithm(config)
        print(f"自适应算法初始化完成，当前算法: {adaptive_algo.get_current_algorithm().value}")
        
        # 启动自适应算法
        adaptive_algo.start()
        print("自适应算法已启动")
        
        # 模拟处理帧
        print("\n开始模拟帧处理...")
        for i in range(10):
            # 创建模拟帧和检测结果
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            detections = [
                {
                    'bbox': [100 + i*10, 100, 200 + i*10, 300],
                    'confidence': 0.8 + np.random.uniform(-0.1, 0.1),
                    'class': 'person'
                }
            ]
            
            # 处理帧
            results, meta = adaptive_algo.process_frame(frame, detections)
            
            print(f"帧 {i+1}: 算法={meta['algorithm']}, "
                  f"场景={meta['scene_type']}, "
                  f"处理时间={meta['processing_time']:.1f}ms, "
                  f"跟踪数={len(results)}")
            
            time.sleep(0.5)
        
        # 获取自适应状态
        print("\n=== 自适应状态 ===")
        status = adaptive_algo.get_adaptation_status()
        print(f"运行状态: {status['running']}")
        print(f"自适应启用: {status['adaptive_enabled']}")
        print(f"在线学习启用: {status['learning_enabled']}")
        print(f"当前算法: {status['current_algorithm']}")
        
        if status['current_scene']:
            print(f"当前场景: {status['current_scene']['type']} "
                  f"(置信度: {status['current_scene']['confidence']:.3f})")
        
        # 测试手动切换算法
        print("\n=== 测试手动切换算法 ===")
        adaptive_algo.set_algorithm(TrackingAlgorithm.PARTICLE_FILTER, force=True)
        print(f"切换后算法: {adaptive_algo.get_current_algorithm().value}")
        
        # 测试配置保存和加载
        print("\n=== 测试配置管理 ===")
        config_file = "adaptive_config_test.json"
        adaptive_algo.save_config(config_file)
        print(f"配置已保存到: {config_file}")
        
        # 修改配置并重新加载
        adaptive_algo.config.learning_rate = 0.05
        adaptive_algo.load_config(config_file)
        print(f"配置重新加载，学习率: {adaptive_algo.config.learning_rate}")
        
        # 等待一段时间观察自适应行为
        print("\n观察自适应行为 10 秒...")
        time.sleep(10)
        
        # 获取性能摘要
        print("\n=== 性能摘要 ===")
        summary = adaptive_algo.performance_evaluator.get_performance_summary()
        print(f"总算法数: {summary['total_algorithms']}")
        print(f"已评估算法数: {summary['evaluated_algorithms']}")
        print(f"总记录数: {summary['total_records']}")
        
        if summary['algorithm_performance']:
            print("\n算法性能:")
            for alg, perf in summary['algorithm_performance'].items():
                print(f"  {alg}: 综合评分={perf['overall_score']:.3f}, "
                      f"准确率={perf['accuracy']:.3f}, FPS={perf['fps']:.1f}")
        
        # 停止自适应算法
        adaptive_algo.stop()
        print("\n自适应算法已停止")
        
        # 清理测试文件
        import os
        if os.path.exists(config_file):
            os.remove(config_file)
            print(f"测试配置文件已删除: {config_file}")
        
        print("\n=== 测试完成 ===")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()