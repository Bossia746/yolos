#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人体姿态识别和运动计数模块

基于YOLO的人体关键点检测，实现多种运动的自动识别和计数功能。
支持俯卧撑、深蹲、压腿、高抬腿等多种运动类型。

主要功能:
- 人体关键点检测和角度计算
- 多种运动类型的自动识别
- 实时视频处理和计数
- 可视化和统计分析
- 灵活的配置和扩展接口
"""

# 导入核心类和枚举
try:
    from .pose_recognition import (
        PoseRecognizer,
        PoseRecognizerConfig,
        ExerciseType,
        PoseState,
        KeypointConfig,
        ExerciseStats,
        PoseAnalysisResult
    )
except ImportError as e:
    # 提供默认实现
    from enum import Enum
    from dataclasses import dataclass
    from typing import Dict, Any, List
    
    class ExerciseType(Enum):
        PUSHUP = "pushup"
        SQUAT = "squat"
        LEG_PRESS = "leg_press"
        HIGH_KNEE = "high_knee"
        CUSTOM = "custom"
    
    class PoseState(Enum):
        UP = "up"
        DOWN = "down"
        TRANSITION = "transition"
        UNKNOWN = "unknown"
    
    @dataclass
    class KeypointConfig:
        keypoints: List[int]
        up_angle: float = 145.0
        down_angle: float = 100.0
        angle_tolerance: float = 10.0
    
    @dataclass
    class ExerciseStats:
        count: int = 0
        total_time: float = 0.0
        avg_time_per_rep: float = 0.0
    
    @dataclass
    class PoseAnalysisResult:
        state: PoseState
        angle: float
        confidence: float
        stats: ExerciseStats
    
    @dataclass
    class PoseRecognizerConfig:
        model_path: str
        exercise_type: ExerciseType = ExerciseType.PUSHUP
        confidence_threshold: float = 0.5
        device: str = "auto"
    
    class PoseRecognizer:
        def __init__(self, config: PoseRecognizerConfig):
            self.config = config
        
        def recognize(self, image, **kwargs):
            return PoseAnalysisResult(
                state=PoseState.UNKNOWN,
                angle=0.0,
                confidence=0.0,
                stats=ExerciseStats()
            )

from .base_recognizer import (
    BaseRecognizer,
    RecognizerType,
    RecognizerConfig,
    BatchRecognizer
)

from .factory import RecognizerFactory

try:
    from .exercise_factory import (
        ExerciseFactory,
        create_pushup_counter,
        create_squat_counter,
        create_exercise_counter
    )
except ImportError:
    # 如果exercise_factory不存在，提供默认实现
    class ExerciseFactory:
        @staticmethod
        def get_available_presets():
            return {"pushup": "俯卧撑", "squat": "深蹲"}
        
        @staticmethod
        def get_model_info():
            return {"fast": "快速模式", "balanced": "平衡模式", "high_accuracy": "高精度模式"}
        
        @staticmethod
        def recommend_config(device_type):
            return {"model_quality": "balanced"}
        
        @staticmethod
        def create_recognizer(preset_name, model_quality):
            return PoseRecognizer(PoseRecognizerConfig(
                model_path="",
                exercise_type=ExerciseType.PUSHUP
            ))
    
    def create_pushup_counter(model_quality='balanced'):
        return PoseRecognizer(PoseRecognizerConfig(
            model_path="",
            exercise_type=ExerciseType.PUSHUP
        ))
    
    def create_squat_counter(model_quality='balanced'):
        return PoseRecognizer(PoseRecognizerConfig(
            model_path="",
            exercise_type=ExerciseType.SQUAT
        ))
    
    def create_exercise_counter(exercise_type, model_quality='balanced'):
        return PoseRecognizer(PoseRecognizerConfig(
            model_path="",
            exercise_type=exercise_type
        ))

# 定义公开接口
__all__ = [
    # 核心类
    'PoseRecognizer',
    'BaseRecognizer',
    'BatchRecognizer',
    'RecognizerFactory',
    
    # 配置类
    'PoseRecognizerConfig',
    'RecognizerConfig',
    'KeypointConfig',
    
    # 枚举类型
    'ExerciseType',
    'PoseState',
    'RecognizerType',
    
    # 结果和统计类
    'ExerciseStats',
    'PoseAnalysisResult',
    
    # 工厂类
    'ExerciseFactory',
    
    # 便捷函数
    'create_pushup_counter',
    'create_squat_counter',
    'create_exercise_counter',
    
    # 模块级便捷函数
    'get_supported_exercises',
    'get_model_recommendations',
    'get_keypoint_names',
    'quick_start_pushup',
    'quick_start_squat',
    'create_recognizer_for_device',
]

# 模块信息
__version__ = '1.0.0'
__author__ = 'YOLOS Team'
__description__ = 'Human pose recognition and exercise counting system based on YOLO'

# 模块级别的便捷函数
def get_supported_exercises():
    """
    获取支持的运动类型
    
    Returns:
        Dict[str, str]: 运动名称和描述的字典
    """
    try:
        return ExerciseFactory.get_available_presets()
    except:
        return {"pushup": "俯卧撑", "squat": "深蹲", "leg_press": "压腿", "high_knee": "高抬腿"}

def get_model_recommendations():
    """
    获取模型推荐信息
    
    Returns:
        Dict[str, str]: 模型质量和描述的字典
    """
    try:
        return ExerciseFactory.get_model_info()
    except:
        return {"fast": "快速模式", "balanced": "平衡模式", "high_accuracy": "高精度模式"}

def get_keypoint_names():
    """
    获取关键点名称映射
    
    Returns:
        Dict[int, str]: 关键点索引和名称的字典
    """
    try:
        return KeypointConfig.KEYPOINT_NAMES
    except:
        return {
            0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
            5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
            9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
            13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"
        }

def quick_start_pushup(model_quality='balanced'):
    """
    快速开始俯卧撑计数
    
    Args:
        model_quality (str): 模型质量 ('fast', 'balanced', 'high_accuracy')
    
    Returns:
        PoseRecognizer: 配置好的俯卧撑识别器
    """
    return create_pushup_counter(model_quality)

def quick_start_squat(model_quality='balanced'):
    """
    快速开始深蹲计数
    
    Args:
        model_quality (str): 模型质量 ('fast', 'balanced', 'high_accuracy')
    
    Returns:
        PoseRecognizer: 配置好的深蹲识别器
    """
    return create_squat_counter(model_quality)

def create_recognizer_for_device(device_type='desktop', exercise='pushup'):
    """
    根据设备类型创建优化的识别器
    
    Args:
        device_type (str): 设备类型 ('desktop', 'laptop', 'mobile', 'embedded')
        exercise (str): 运动类型 ('pushup', 'squat', 'leg_press', 'high_knee')
    
    Returns:
        PoseRecognizer: 优化配置的识别器
    """
    # 获取设备推荐配置
    config = ExerciseFactory.recommend_config(device_type)
    model_quality = config['model_quality']
    
    # 根据运动类型创建识别器
    if exercise == 'pushup':
        return create_pushup_counter(model_quality)
    elif exercise == 'squat':
        return create_squat_counter(model_quality)
    else:
        # 使用预设名称
        preset_name = f"{exercise}_standard"
        return ExerciseFactory.create_recognizer(preset_name, model_quality)

# 模块初始化时的信息输出
import logging
logger = logging.getLogger(__name__)

try:
    logger.info(f"YOLOS Recognition Module v{__version__} loaded")
    logger.info(f"Supported exercises: {len(get_supported_exercises())}")
    logger.info(f"Available models: {list(get_model_recommendations().keys())}")
except Exception as e:
    logger.warning(f"Recognition module initialization warning: {e}")