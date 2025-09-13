#!/usr/bin/env python3
"""
轻量级特征聚合模块
基于YOLOV论文思想，适配多平台多场景部署
支持树莓派、ESP32等边缘设备
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple, Callable
from collections import deque
from dataclasses import dataclass
from enum import Enum
import threading
from queue import Queue, Empty


class PlatformType(Enum):
    """平台类型枚举"""
    PC = "pc"
    DESKTOP = "desktop"  # 桌面平台别名
    RASPBERRY_PI = "raspberry_pi"
    ESP32 = "esp32"
    JETSON = "jetson"
    MOBILE = "mobile"


@dataclass
class AggregationConfig:
    """特征聚合配置"""
    # 基础配置
    max_frames: int = 5  # 最大聚合帧数
    min_frames: int = 2  # 最小聚合帧数
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    
    # 平台适配
    platform: PlatformType = PlatformType.PC
    memory_limit_mb: int = 512  # 内存限制
    cpu_cores: int = 4
    
    # 性能优化
    adaptive_frames: bool = True  # 自适应帧数
    temporal_weight_decay: float = 0.8  # 时序权重衰减
    feature_compression: bool = True  # 特征压缩
    
    # 质量控制
    motion_threshold: float = 0.1  # 运动阈值
    stability_factor: float = 0.7  # 稳定性因子
    
    def get_platform_config(self) -> Dict[str, Any]:
        """获取平台特定配置"""
        configs = {
            PlatformType.PC: {
                'max_frames': 8,
                'memory_limit_mb': 2048,
                'feature_compression': False
            },
            PlatformType.RASPBERRY_PI: {
                'max_frames': 3,
                'memory_limit_mb': 256,
                'feature_compression': True
            },
            PlatformType.ESP32: {
                'max_frames': 2,
                'memory_limit_mb': 64,
                'feature_compression': True
            },
            PlatformType.JETSON: {
                'max_frames': 6,
                'memory_limit_mb': 1024,
                'feature_compression': False
            },
            PlatformType.MOBILE: {
                'max_frames': 4,
                'memory_limit_mb': 512,
                'feature_compression': True
            }
        }
        return configs.get(self.platform, {})


class FrameBuffer:
    """帧缓冲区管理"""
    
    def __init__(self, max_size: int, compress: bool = False):
        self.max_size = max_size
        self.compress = compress
        self.frames = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
        self.features = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def add_frame(self, frame: np.ndarray, timestamp: float, features: Optional[np.ndarray] = None):
        """添加帧到缓冲区"""
        with self.lock:
            if self.compress:
                # 压缩帧以节省内存
                frame = self._compress_frame(frame)
            
            self.frames.append(frame)
            self.timestamps.append(timestamp)
            
            if features is not None:
                if self.compress:
                    features = self._compress_features(features)
                self.features.append(features)
    
    def get_frames(self, count: Optional[int] = None) -> List[Tuple[np.ndarray, float]]:
        """获取帧列表"""
        with self.lock:
            if count is None:
                count = len(self.frames)
            
            frames_list = []
            for i in range(min(count, len(self.frames))):
                frame = self.frames[-(i+1)]
                timestamp = self.timestamps[-(i+1)]
                
                if self.compress:
                    frame = self._decompress_frame(frame)
                
                frames_list.append((frame, timestamp))
            
            return list(reversed(frames_list))
    
    def _compress_frame(self, frame: np.ndarray) -> bytes:
        """压缩帧"""
        _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return encoded.tobytes()
    
    def _decompress_frame(self, compressed_frame: bytes) -> np.ndarray:
        """解压缩帧"""
        nparr = np.frombuffer(compressed_frame, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    def _compress_features(self, features: np.ndarray) -> np.ndarray:
        """压缩特征"""
        return features.astype(np.float16)  # 使用半精度浮点数
    
    def clear(self):
        """清空缓冲区"""
        with self.lock:
            self.frames.clear()
            self.timestamps.clear()
            self.features.clear()


class MotionAnalyzer:
    """运动分析器"""
    
    def __init__(self):
        self.prev_frame = None
        self.motion_history = deque(maxlen=10)
    
    def analyze_motion(self, frame: np.ndarray) -> float:
        """分析帧间运动强度"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return 0.0
        
        # 计算光流
        flow = cv2.calcOpticalFlowPyrLK(
            self.prev_frame, gray, None, None,
            winSize=(15, 15), maxLevel=2
        )
        
        # 计算运动强度
        if flow[0] is not None:
            motion_magnitude = np.mean(np.sqrt(flow[0][:, :, 0]**2 + flow[0][:, :, 1]**2))
        else:
            motion_magnitude = 0.0
        
        self.motion_history.append(motion_magnitude)
        self.prev_frame = gray
        
        return motion_magnitude
    
    def get_motion_trend(self) -> float:
        """获取运动趋势"""
        if len(self.motion_history) < 3:
            return 0.0
        
        recent = list(self.motion_history)[-3:]
        return np.mean(recent)


class LightweightFeatureAggregator:
    """轻量级特征聚合器"""
    
    def __init__(self, config: AggregationConfig):
        self.config = config
        self.platform_config = config.get_platform_config()
        
        # 应用平台特定配置
        for key, value in self.platform_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # 初始化组件
        self.frame_buffer = FrameBuffer(
            max_size=config.max_frames,
            compress=config.feature_compression
        )
        self.motion_analyzer = MotionAnalyzer()
        
        # 性能监控
        self.processing_times = deque(maxlen=100)
        self.memory_usage = 0
        
        # 回调函数
        self.progress_callback: Optional[Callable] = None
    
    def set_progress_callback(self, callback: Callable):
        """设置进度回调"""
        self.progress_callback = callback
    
    def aggregate_detections(self, 
                           current_frame: np.ndarray,
                           current_detections: List[Dict[str, Any]],
                           timestamp: float) -> List[Dict[str, Any]]:
        """聚合检测结果"""
        start_time = time.time()
        
        # 分析运动
        motion_intensity = self.motion_analyzer.analyze_motion(current_frame)
        
        # 添加当前帧到缓冲区
        self.frame_buffer.add_frame(current_frame, timestamp)
        
        # 自适应帧数调整
        if self.config.adaptive_frames:
            effective_frames = self._calculate_adaptive_frames(motion_intensity)
        else:
            effective_frames = self.config.max_frames
        
        # 获取历史帧
        historical_frames = self.frame_buffer.get_frames(effective_frames)
        
        if len(historical_frames) < self.config.min_frames:
            # 帧数不足，返回当前检测结果
            return current_detections
        
        # 执行特征聚合
        aggregated_detections = self._perform_aggregation(
            current_detections, historical_frames, motion_intensity
        )
        
        # 记录处理时间
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        # 调用进度回调
        if self.progress_callback:
            self.progress_callback({
                'processing_time': processing_time,
                'motion_intensity': motion_intensity,
                'effective_frames': effective_frames,
                'aggregated_count': len(aggregated_detections)
            })
        
        return aggregated_detections
    
    def _calculate_adaptive_frames(self, motion_intensity: float) -> int:
        """计算自适应帧数"""
        # 根据运动强度调整帧数
        if motion_intensity > self.config.motion_threshold * 2:
            # 高运动场景，减少帧数
            return max(self.config.min_frames, self.config.max_frames // 2)
        elif motion_intensity < self.config.motion_threshold * 0.5:
            # 低运动场景，增加帧数
            return self.config.max_frames
        else:
            # 中等运动场景，使用默认帧数
            return max(self.config.min_frames, int(self.config.max_frames * 0.7))
    
    def _perform_aggregation(self, 
                           current_detections: List[Dict[str, Any]],
                           historical_frames: List[Tuple[np.ndarray, float]],
                           motion_intensity: float) -> List[Dict[str, Any]]:
        """执行特征聚合"""
        if not historical_frames:
            return current_detections
        
        # 时序权重计算
        weights = self._calculate_temporal_weights(len(historical_frames))
        
        # 检测结果聚合
        aggregated_results = []
        
        for detection in current_detections:
            # 在历史帧中寻找相似检测
            similar_detections = self._find_similar_detections(
                detection, historical_frames, weights
            )
            
            if similar_detections:
                # 聚合相似检测
                aggregated_detection = self._merge_detections(
                    detection, similar_detections, weights
                )
                aggregated_results.append(aggregated_detection)
            else:
                # 没有相似检测，直接使用当前检测
                aggregated_results.append(detection)
        
        # 应用稳定性过滤
        stable_results = self._apply_stability_filter(
            aggregated_results, motion_intensity
        )
        
        return stable_results
    
    def _calculate_temporal_weights(self, frame_count: int) -> List[float]:
        """计算时序权重"""
        weights = []
        for i in range(frame_count):
            # 越新的帧权重越高
            weight = self.config.temporal_weight_decay ** (frame_count - 1 - i)
            weights.append(weight)
        
        # 归一化权重
        total_weight = sum(weights)
        return [w / total_weight for w in weights] if total_weight > 0 else weights
    
    def _find_similar_detections(self, 
                               target_detection: Dict[str, Any],
                               historical_frames: List[Tuple[np.ndarray, float]],
                               weights: List[float]) -> List[Dict[str, Any]]:
        """在历史帧中寻找相似检测"""
        # 简化实现：基于边界框IoU寻找相似检测
        similar_detections = []
        target_bbox = target_detection.get('bbox', [])
        
        if len(target_bbox) != 4:
            return similar_detections
        
        # 这里应该对历史帧进行检测，简化为模拟
        # 实际实现中需要调用检测模型
        for i, (frame, timestamp) in enumerate(historical_frames[:-1]):
            # 模拟历史检测结果
            simulated_detection = {
                'bbox': target_bbox,  # 简化：使用相同边界框
                'confidence': target_detection.get('confidence', 0.5) * weights[i],
                'class_name': target_detection.get('class_name', 'unknown'),
                'timestamp': timestamp
            }
            similar_detections.append(simulated_detection)
        
        return similar_detections
    
    def _merge_detections(self, 
                         current_detection: Dict[str, Any],
                         similar_detections: List[Dict[str, Any]],
                         weights: List[float]) -> Dict[str, Any]:
        """合并检测结果"""
        # 加权平均置信度
        total_confidence = current_detection.get('confidence', 0.5)
        total_weight = 1.0
        
        for i, detection in enumerate(similar_detections):
            if i < len(weights):
                total_confidence += detection.get('confidence', 0.5) * weights[i]
                total_weight += weights[i]
        
        merged_detection = current_detection.copy()
        merged_detection['confidence'] = total_confidence / total_weight
        merged_detection['aggregated'] = True
        merged_detection['frame_count'] = len(similar_detections) + 1
        
        return merged_detection
    
    def _apply_stability_filter(self, 
                              detections: List[Dict[str, Any]],
                              motion_intensity: float) -> List[Dict[str, Any]]:
        """应用稳定性过滤"""
        filtered_results = []
        
        for detection in detections:
            confidence = detection.get('confidence', 0.0)
            
            # 根据运动强度调整置信度阈值
            if motion_intensity > self.config.motion_threshold:
                # 高运动场景，降低阈值
                threshold = self.config.confidence_threshold * 0.8
            else:
                # 低运动场景，使用标准阈值
                threshold = self.config.confidence_threshold
            
            # 应用稳定性因子
            if detection.get('aggregated', False):
                threshold *= self.config.stability_factor
            
            if confidence >= threshold:
                filtered_results.append(detection)
        
        return filtered_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        if not self.processing_times:
            return {}
        
        return {
            'avg_processing_time': np.mean(self.processing_times),
            'max_processing_time': np.max(self.processing_times),
            'min_processing_time': np.min(self.processing_times),
            'fps_estimate': 1.0 / np.mean(self.processing_times) if np.mean(self.processing_times) > 0 else 0,
            'memory_usage_mb': self.memory_usage,
            'platform': self.config.platform.value
        }
    
    def reset(self):
        """重置聚合器状态"""
        self.frame_buffer.clear()
        self.motion_analyzer = MotionAnalyzer()
        self.processing_times.clear()


def create_platform_aggregator(platform: str, **kwargs) -> LightweightFeatureAggregator:
    """创建平台特定的聚合器"""
    platform_type = PlatformType(platform.lower())
    
    config = AggregationConfig(
        platform=platform_type,
        **kwargs
    )
    
    return LightweightFeatureAggregator(config)


# 为了向后兼容，提供别名
FeatureAggregator = LightweightFeatureAggregator


# 使用示例
if __name__ == "__main__":
    # 创建树莓派优化的聚合器
    aggregator = create_platform_aggregator(
        platform="raspberry_pi",
        max_frames=3,
        confidence_threshold=0.3
    )
    
    print(f"聚合器配置: {aggregator.config}")
    print(f"平台配置: {aggregator.platform_config}")