# -*- coding: utf-8 -*-
"""
自适应ROI机制应用模块
基于FastTracker论文的自适应ROI优化实现
支持多种应用场景的智能ROI预测和调整
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import yaml
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import os
from .enhanced_mish_activation import EnhancedMish, MishVariants

# 配置日志
def get_logger(name: str) -> logging.Logger:
    """获取日志记录器"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

logger = get_logger(__name__)

class ROIStrategy(Enum):
    """ROI策略枚举"""
    PATIENT_FOCUSED = "patient_focused"  # 医疗监控
    MOVEMENT_ADAPTIVE = "movement_adaptive"  # 宠物监控
    INTRUSION_DETECTION = "intrusion_detection"  # 安全监控
    HAND_TRACKING = "hand_tracking"  # 手势识别
    CENTER_CROP = "center_crop"  # 中心裁剪（备用策略）

@dataclass
class ROIParameters:
    """ROI参数配置"""
    base_roi_size: Tuple[int, int]
    min_roi_size: Tuple[int, int]
    max_roi_size: Tuple[int, int]
    expansion_factor: float
    tracking_sensitivity: float = 0.8
    update_frequency: int = 3
    # 可选参数，用于不同场景的特殊配置
    fall_detection_roi: bool = False
    vital_signs_roi: bool = False
    activity_threshold: float = 0.6
    behavior_analysis_roi: bool = False
    multi_pet_support: bool = False
    motion_sensitivity: float = 0.7
    perimeter_roi: bool = False
    danger_zone_roi: bool = False
    hand_detection_threshold: float = 0.5
    gesture_roi_padding: int = 20
    multi_hand_support: bool = False
    
class SceneAnalyzer(nn.Module):
    """场景分析器
    
    分析当前场景特征，为ROI预测提供上下文信息
    """
    
    def __init__(self, input_channels: int = 3, feature_dim: int = 64):
        super().__init__()
        
        # 场景特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            MishVariants.fast_mish(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            MishVariants.fast_mish(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, feature_dim)
        )
        
        # 运动分析
        self.motion_analyzer = nn.Sequential(
            nn.Linear(feature_dim * 2, 32),  # 当前帧 + 前一帧
            MishVariants.standard_mish(),
            nn.Linear(32, 16),
            MishVariants.standard_mish(),
            nn.Linear(16, 4)  # 运动向量 (dx, dy, speed, direction)
        )
        
    def forward(self, current_frame: torch.Tensor, 
                prev_frame: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """分析场景特征
        
        Args:
            current_frame: 当前帧 [B, C, H, W]
            prev_frame: 前一帧 [B, C, H, W]
            
        Returns:
            场景分析结果
        """
        # 提取当前帧特征
        current_features = self.feature_extractor(current_frame)
        
        results = {
            'scene_features': current_features,
            'motion_info': torch.zeros(current_frame.size(0), 4).to(current_frame.device)
        }
        
        # 如果有前一帧，分析运动信息
        if prev_frame is not None:
            prev_features = self.feature_extractor(prev_frame)
            combined_features = torch.cat([current_features, prev_features], dim=1)
            motion_info = self.motion_analyzer(combined_features)
            results['motion_info'] = motion_info
            
        return results

class AdaptiveROIPredictor(nn.Module):
    """自适应ROI预测器
    
    基于场景分析和历史信息预测最优ROI区域
    """
    
    def __init__(self, scene_feature_dim: int = 64, history_length: int = 5):
        super().__init__()
        
        self.history_length = history_length
        
        # ROI预测网络
        input_dim = scene_feature_dim + 4 + 4 * history_length  # 场景特征 + 运动信息 + 历史ROI
        
        self.roi_predictor = nn.Sequential(
            nn.Linear(input_dim, 128),
            MishVariants.adaptive_mish(learnable=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            MishVariants.adaptive_mish(learnable=True),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            MishVariants.standard_mish(),
            nn.Linear(32, 4),  # ROI边界框 (x1, y1, x2, y2)
            nn.Sigmoid()
        )
        
        # 置信度预测
        self.confidence_predictor = nn.Sequential(
            nn.Linear(input_dim, 32),
            MishVariants.standard_mish(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, scene_features: torch.Tensor, 
                motion_info: torch.Tensor,
                roi_history: torch.Tensor) -> Dict[str, torch.Tensor]:
        """预测ROI区域
        
        Args:
            scene_features: 场景特征 [B, feature_dim]
            motion_info: 运动信息 [B, 4]
            roi_history: ROI历史 [B, history_length, 4]
            
        Returns:
            ROI预测结果
        """
        with torch.no_grad():  # 确保预测一致性，避免梯度问题
            batch_size = scene_features.size(0)
            
            # 展平ROI历史
            roi_history_flat = roi_history.view(batch_size, -1)
            
            # 拼接所有特征
            combined_features = torch.cat([
                scene_features, motion_info, roi_history_flat
            ], dim=1)
            
            # 预测ROI和置信度
            predicted_roi = self.roi_predictor(combined_features)
            confidence = self.confidence_predictor(combined_features)
            
            return {
                'roi': predicted_roi,
                'confidence': confidence
            }

class AdaptiveROIApplication:
    """自适应ROI应用系统
    
    整合场景分析、ROI预测和策略应用的完整系统
    """
    
    def __init__(self, config_path: str = None):
        """初始化自适应ROI应用系统
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化组件
        self.scene_analyzer = SceneAnalyzer()
        self.roi_predictor = AdaptiveROIPredictor()
        
        # 设置为评估模式以确保预测一致性
        self.scene_analyzer.eval()
        self.roi_predictor.eval()
        
        # 历史信息缓存
        self.roi_history = []
        self.frame_history = []
        self.performance_stats = {
            'prediction_times': [],
            'roi_accuracies': [],
            'memory_usage': []
        }
        
        # 当前策略
        self.current_strategy = ROIStrategy.CENTER_CROP
        self.strategy_parameters = None
        
        logger.info(f"自适应ROI应用系统初始化完成")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        if config_path is None:
            config_path = "config/adaptive_roi_application_config.yaml"
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"成功加载配置文件: {config_path}")
            return config
        except Exception as e:
            logger.warning(f"配置文件加载失败: {e}，使用默认配置")
            return self._get_default_config()
            
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'global': {'enabled': True, 'debug_mode': False},
            'application_scenarios': {
                'medical_monitoring': {
                    'enabled': True,
                    'parameters': {
                        'base_roi_size': [416, 416],
                        'min_roi_size': [224, 224],
                        'max_roi_size': [640, 640],
                        'expansion_factor': 1.3
                    }
                }
            }
        }
        
    def set_scenario(self, scenario: str) -> bool:
        """设置应用场景
        
        Args:
            scenario: 场景名称
            
        Returns:
            设置是否成功
        """
        scenarios = self.config.get('application_scenarios', {})
        
        if scenario not in scenarios:
            logger.error(f"未知场景: {scenario}")
            return False
            
        scenario_config = scenarios[scenario]
        if not scenario_config.get('enabled', False):
            logger.error(f"场景未启用: {scenario}")
            return False
            
        # 设置策略
        strategy_map = {
            'medical_monitoring': ROIStrategy.PATIENT_FOCUSED,
            'pet_monitoring': ROIStrategy.MOVEMENT_ADAPTIVE,
            'security_monitoring': ROIStrategy.INTRUSION_DETECTION,
            'gesture_recognition': ROIStrategy.HAND_TRACKING
        }
        
        self.current_strategy = strategy_map.get(scenario, ROIStrategy.CENTER_CROP)
        self.strategy_parameters = ROIParameters(**scenario_config['parameters'])
        
        logger.info(f"切换到场景: {scenario}, 策略: {self.current_strategy.value}")
        return True
        
    def predict_roi(self, image: np.ndarray, 
                   detection_results: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """预测ROI区域
        
        Args:
            image: 输入图像
            detection_results: 检测结果（可选）
            
        Returns:
            ROI预测结果
        """
        start_time = time.time()
        
        try:
            # 预处理图像
            image_tensor = self._preprocess_image(image)
            
            # 场景分析
            prev_frame = self.frame_history[-1] if self.frame_history else None
            scene_analysis = self.scene_analyzer(image_tensor, prev_frame)
            
            # 准备ROI历史
            roi_history_tensor = self._prepare_roi_history()
            
            # ROI预测
            with torch.no_grad():
                prediction_results = self.roi_predictor(
                    scene_analysis['scene_features'],
                    scene_analysis['motion_info'],
                    roi_history_tensor
                )
            
            # 应用策略调整
            adjusted_roi = self._apply_strategy_adjustment(
                prediction_results['roi'],
                prediction_results['confidence'],
                detection_results
            )
            
            # 更新历史
            self._update_history(image_tensor, adjusted_roi)
            
            # 计算性能统计
            prediction_time = (time.time() - start_time) * 1000
            self.performance_stats['prediction_times'].append(prediction_time)
            
            results = {
                'roi': adjusted_roi.cpu().numpy(),
                'confidence': prediction_results['confidence'].cpu().numpy(),
                'strategy': self.current_strategy.value,
                'prediction_time_ms': prediction_time,
                'scene_features': scene_analysis['scene_features'].cpu().numpy()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"ROI预测失败: {e}")
            return self._get_fallback_roi(image.shape[:2])
            
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """预处理图像"""
        # 调整大小
        image_resized = cv2.resize(image, (224, 224))
        
        # 归一化
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # 转换为张量
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor
        
    def _prepare_roi_history(self) -> torch.Tensor:
        """准备ROI历史数据"""
        history_length = 5
        
        if len(self.roi_history) < history_length:
            # 用默认ROI填充
            default_roi = torch.tensor([0.25, 0.25, 0.75, 0.75])
            padding_length = history_length - len(self.roi_history)
            padded_history = [default_roi] * padding_length + self.roi_history
        else:
            padded_history = self.roi_history[-history_length:]
            
        return torch.stack(padded_history).unsqueeze(0)
        
    def _apply_strategy_adjustment(self, 
                                 predicted_roi: torch.Tensor,
                                 confidence: torch.Tensor,
                                 detection_results: Optional[List[Dict]]) -> torch.Tensor:
        """应用策略调整"""
        if self.strategy_parameters is None:
            return predicted_roi
            
        # 基于置信度调整
        conf_value = confidence.item()
        if conf_value < 0.7:
            # 低置信度时扩大ROI
            expansion = self.strategy_parameters.expansion_factor
            center_x = (predicted_roi[0, 0] + predicted_roi[0, 2]) / 2
            center_y = (predicted_roi[0, 1] + predicted_roi[0, 3]) / 2
            width = (predicted_roi[0, 2] - predicted_roi[0, 0]) * expansion
            height = (predicted_roi[0, 3] - predicted_roi[0, 1]) * expansion
            
            adjusted_roi = torch.tensor([[
                max(0, center_x - width/2),
                max(0, center_y - height/2),
                min(1, center_x + width/2),
                min(1, center_y + height/2)
            ]])
        else:
            adjusted_roi = predicted_roi
            
        # 基于检测结果调整
        if detection_results and self.current_strategy == ROIStrategy.PATIENT_FOCUSED:
            # 医疗监控场景：优先关注人体检测结果
            for detection in detection_results:
                if detection.get('class') == 'person' and detection.get('confidence', 0) > 0.8:
                    # 调整ROI以包含检测到的人体
                    bbox = detection['bbox']  # [x1, y1, x2, y2]
                    # 这里可以添加更复杂的ROI调整逻辑
                    pass
                    
        return adjusted_roi
        
    def _update_history(self, image_tensor: torch.Tensor, roi: torch.Tensor):
        """更新历史信息"""
        max_history = 10
        
        # 更新ROI历史
        self.roi_history.append(roi.squeeze(0))
        if len(self.roi_history) > max_history:
            self.roi_history.pop(0)
            
        # 更新帧历史
        self.frame_history.append(image_tensor)
        if len(self.frame_history) > max_history:
            self.frame_history.pop(0)
            
    def _get_fallback_roi(self, image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """获取备用ROI"""
        h, w = image_shape
        center_roi = np.array([[0.25, 0.25, 0.75, 0.75]])
        
        return {
            'roi': center_roi,
            'confidence': np.array([[0.5]]),
            'strategy': 'fallback',
            'prediction_time_ms': 0.0,
            'scene_features': np.zeros((1, 64))
        }
        
    def apply_roi_to_image(self, image: np.ndarray, roi: np.ndarray) -> np.ndarray:
        """将ROI应用到图像
        
        Args:
            image: 输入图像
            roi: ROI边界框 [x1, y1, x2, y2] (归一化坐标)
            
        Returns:
            ROI裁剪后的图像
        """
        h, w = image.shape[:2]
        
        # 转换为像素坐标
        x1 = int(roi[0] * w)
        y1 = int(roi[1] * h)
        x2 = int(roi[2] * w)
        y2 = int(roi[3] * h)
        
        # 确保坐标在有效范围内
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        return image[y1:y2, x1:x2]
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        stats = {}
        
        if self.performance_stats['prediction_times']:
            times = self.performance_stats['prediction_times']
            stats['avg_prediction_time_ms'] = np.mean(times)
            stats['max_prediction_time_ms'] = np.max(times)
            stats['min_prediction_time_ms'] = np.min(times)
            
        stats['total_predictions'] = len(self.performance_stats['prediction_times'])
        stats['current_strategy'] = self.current_strategy.value
        
        return stats
        
    def reset_history(self):
        """重置历史信息"""
        self.roi_history.clear()
        self.frame_history.clear()
        self.performance_stats = {
            'prediction_times': [],
            'roi_accuracies': [],
            'memory_usage': []
        }
        logger.info("历史信息已重置")

# 便捷函数
def create_adaptive_roi_system(scenario: str = 'medical_monitoring', 
                              config_path: str = None) -> AdaptiveROIApplication:
    """创建自适应ROI系统
    
    Args:
        scenario: 应用场景
        config_path: 配置文件路径
        
    Returns:
        自适应ROI应用系统实例
    """
    system = AdaptiveROIApplication(config_path)
    system.set_scenario(scenario)
    return system

if __name__ == "__main__":
    # 测试代码
    print("=== 自适应ROI应用系统测试 ===")
    
    # 创建系统
    roi_system = create_adaptive_roi_system('medical_monitoring')
    
    # 模拟图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 预测ROI
    results = roi_system.predict_roi(test_image)
    
    print(f"ROI预测结果: {results['roi']}")
    print(f"置信度: {results['confidence']}")
    print(f"预测时间: {results['prediction_time_ms']:.2f} ms")
    print(f"当前策略: {results['strategy']}")
    
    # 应用ROI
    roi_image = roi_system.apply_roi_to_image(test_image, results['roi'][0])
    print(f"ROI图像尺寸: {roi_image.shape}")
    
    # 性能统计
    stats = roi_system.get_performance_stats()
    print(f"性能统计: {stats}")
    
    print("✅ 自适应ROI应用系统测试完成")