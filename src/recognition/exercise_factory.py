#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运动识别工厂类
提供便捷的接口创建各种运动类型的姿态识别器
"""

import logging
from typing import Dict, Optional, Any
from pathlib import Path

from .pose_recognition import (
    PoseRecognizer, ExerciseType, KeypointConfig, PoseState
)

logger = logging.getLogger(__name__)

class ExerciseFactory:
    """运动识别工厂类"""
    
    # 推荐的模型配置
    MODEL_RECOMMENDATIONS = {
        'high_accuracy': 'yolo11x-pose.pt',  # 高精度，适合离线分析
        'balanced': 'yolo11m-pose.pt',       # 平衡性能，适合一般使用
        'fast': 'yolo11n-pose.pt',           # 快速，适合实时应用
        'ultra_fast': 'yolo11s-pose.pt'      # 超快速，适合低配置设备
    }
    
    # 预定义的运动配置
    EXERCISE_PRESETS = {
        'pushup_standard': {
            'exercise_type': ExerciseType.PUSHUP,
            'config': KeypointConfig(
                keypoints=[5, 7, 9],  # 左肩、左肘、左腕
                up_angle=145.0,
                down_angle=100.0,
                angle_tolerance=15.0
            ),
            'description': '标准俯卧撑，监测左臂弯曲'
        },
        'pushup_wide': {
            'exercise_type': ExerciseType.PUSHUP,
            'config': KeypointConfig(
                keypoints=[6, 8, 10],  # 右肩、右肘、右腕
                up_angle=150.0,
                down_angle=95.0,
                angle_tolerance=12.0
            ),
            'description': '宽距俯卧撑，监测右臂弯曲'
        },
        'squat_standard': {
            'exercise_type': ExerciseType.SQUAT,
            'config': KeypointConfig(
                keypoints=[6, 12, 14],  # 右肩、右髋、右膝
                up_angle=160.0,
                down_angle=90.0,
                angle_tolerance=20.0
            ),
            'description': '标准深蹲，监测右腿弯曲'
        },
        'squat_deep': {
            'exercise_type': ExerciseType.SQUAT,
            'config': KeypointConfig(
                keypoints=[5, 11, 13],  # 左肩、左髋、左膝
                up_angle=165.0,
                down_angle=70.0,
                angle_tolerance=25.0
            ),
            'description': '深蹲，更大的角度范围'
        },
        'leg_press_left': {
            'exercise_type': ExerciseType.LEG_PRESS,
            'config': KeypointConfig(
                keypoints=[11, 13, 15],  # 左髋、左膝、左踝
                up_angle=150.0,
                down_angle=100.0,
                angle_tolerance=15.0
            ),
            'description': '左腿压腿练习'
        },
        'leg_press_right': {
            'exercise_type': ExerciseType.LEG_PRESS,
            'config': KeypointConfig(
                keypoints=[12, 14, 16],  # 右髋、右膝、右踝
                up_angle=150.0,
                down_angle=100.0,
                angle_tolerance=15.0
            ),
            'description': '右腿压腿练习'
        },
        'high_knee_left': {
            'exercise_type': ExerciseType.HIGH_KNEE,
            'config': KeypointConfig(
                keypoints=[11, 13, 15],  # 左髋、左膝、左踝
                up_angle=90.0,   # 高抬腿时角度小
                down_angle=160.0, # 放下时角度大
                angle_tolerance=20.0
            ),
            'description': '左腿高抬腿练习'
        },
        'high_knee_right': {
            'exercise_type': ExerciseType.HIGH_KNEE,
            'config': KeypointConfig(
                keypoints=[12, 14, 16],  # 右髋、右膝、右踝
                up_angle=90.0,
                down_angle=160.0,
                angle_tolerance=20.0
            ),
            'description': '右腿高抬腿练习'
        },
        'plank_hold': {
            'exercise_type': ExerciseType.PLANK,
            'config': KeypointConfig(
                keypoints=[5, 11, 15],  # 左肩、左髋、左踝
                up_angle=175.0,  # 平板支撑保持直线
                down_angle=160.0,
                angle_tolerance=10.0
            ),
            'description': '平板支撑姿态保持'
        }
    }
    
    @classmethod
    def create_recognizer(cls, preset_name: str = 'pushup_standard',
                         model_quality: str = 'balanced',
                         enable_visualization: bool = True,
                         custom_model_path: Optional[str] = None) -> PoseRecognizer:
        """
        创建姿态识别器
        
        Args:
            preset_name: 预设配置名称
            model_quality: 模型质量 ('high_accuracy', 'balanced', 'fast', 'ultra_fast')
            enable_visualization: 是否启用可视化
            custom_model_path: 自定义模型路径
        
        Returns:
            PoseRecognizer: 姿态识别器实例
        """
        # 获取预设配置
        if preset_name not in cls.EXERCISE_PRESETS:
            logger.warning(f"未知预设: {preset_name}，使用默认配置")
            preset_name = 'pushup_standard'
        
        preset = cls.EXERCISE_PRESETS[preset_name]
        
        # 选择模型
        if custom_model_path:
            model_path = custom_model_path
        else:
            model_path = cls.MODEL_RECOMMENDATIONS.get(model_quality, 'yolo11n-pose.pt')
        
        # 创建识别器
        recognizer = PoseRecognizer(
            model_path=model_path,
            exercise_type=preset['exercise_type'],
            custom_config=preset['config'],
            enable_visualization=enable_visualization
        )
        
        logger.info(f"创建识别器: {preset['description']}，模型: {model_path}")
        return recognizer
    
    @classmethod
    def create_pushup_recognizer(cls, model_quality: str = 'balanced',
                                side: str = 'left') -> PoseRecognizer:
        """创建俯卧撑识别器"""
        preset_name = 'pushup_standard' if side == 'left' else 'pushup_wide'
        return cls.create_recognizer(preset_name, model_quality)
    
    @classmethod
    def create_squat_recognizer(cls, model_quality: str = 'balanced',
                               deep_squat: bool = False) -> PoseRecognizer:
        """创建深蹲识别器"""
        preset_name = 'squat_deep' if deep_squat else 'squat_standard'
        return cls.create_recognizer(preset_name, model_quality)
    
    @classmethod
    def create_leg_press_recognizer(cls, model_quality: str = 'balanced',
                                   side: str = 'left') -> PoseRecognizer:
        """创建压腿识别器"""
        preset_name = f'leg_press_{side}'
        return cls.create_recognizer(preset_name, model_quality)
    
    @classmethod
    def create_high_knee_recognizer(cls, model_quality: str = 'balanced',
                                   side: str = 'left') -> PoseRecognizer:
        """创建高抬腿识别器"""
        preset_name = f'high_knee_{side}'
        return cls.create_recognizer(preset_name, model_quality)
    
    @classmethod
    def create_plank_recognizer(cls, model_quality: str = 'balanced') -> PoseRecognizer:
        """创建平板支撑识别器"""
        return cls.create_recognizer('plank_hold', model_quality)
    
    @classmethod
    def create_custom_recognizer(cls, keypoints: list, up_angle: float, down_angle: float,
                                exercise_type: ExerciseType = ExerciseType.CUSTOM,
                                model_quality: str = 'balanced',
                                angle_tolerance: float = 15.0,
                                min_confidence: float = 0.5) -> PoseRecognizer:
        """
        创建自定义识别器
        
        Args:
            keypoints: 关键点索引列表 [p1, p2, p3]
            up_angle: 伸展角度阈值
            down_angle: 收缩角度阈值
            exercise_type: 运动类型
            model_quality: 模型质量
            angle_tolerance: 角度容差
            min_confidence: 最小置信度
        
        Returns:
            PoseRecognizer: 自定义姿态识别器
        """
        custom_config = KeypointConfig(
            keypoints=keypoints,
            up_angle=up_angle,
            down_angle=down_angle,
            angle_tolerance=angle_tolerance,
            min_confidence=min_confidence
        )
        
        model_path = cls.MODEL_RECOMMENDATIONS.get(model_quality, 'yolo11n-pose.pt')
        
        recognizer = PoseRecognizer(
            model_path=model_path,
            exercise_type=exercise_type,
            custom_config=custom_config,
            enable_visualization=True
        )
        
        logger.info(f"创建自定义识别器: 关键点{keypoints}, 角度范围[{down_angle}, {up_angle}]")
        return recognizer
    
    @classmethod
    def get_available_presets(cls) -> Dict[str, str]:
        """获取可用的预设配置"""
        return {name: config['description'] for name, config in cls.EXERCISE_PRESETS.items()}
    
    @classmethod
    def get_model_info(cls) -> Dict[str, str]:
        """获取模型信息"""
        return {
            'high_accuracy': 'YOLOv11x-pose - 最高精度，适合离线分析，需要较强GPU',
            'balanced': 'YOLOv11m-pose - 平衡性能，适合一般使用场景',
            'fast': 'YOLOv11n-pose - 快速推理，适合实时应用',
            'ultra_fast': 'YOLOv11s-pose - 超快速，适合低配置设备'
        }
    
    @classmethod
    def get_keypoint_info(cls) -> Dict[int, str]:
        """获取关键点信息"""
        return KeypointConfig.KEYPOINT_NAMES
    
    @classmethod
    def recommend_config(cls, device_type: str = 'desktop',
                        accuracy_priority: bool = False) -> Dict[str, Any]:
        """
        根据设备类型推荐配置
        
        Args:
            device_type: 设备类型 ('desktop', 'laptop', 'mobile', 'embedded')
            accuracy_priority: 是否优先考虑精度
        
        Returns:
            推荐配置字典
        """
        recommendations = {
            'desktop': {
                'model_quality': 'high_accuracy' if accuracy_priority else 'balanced',
                'enable_visualization': True,
                'suggested_fps': 30,
                'description': '桌面设备，性能充足'
            },
            'laptop': {
                'model_quality': 'balanced' if accuracy_priority else 'fast',
                'enable_visualization': True,
                'suggested_fps': 20,
                'description': '笔记本设备，中等性能'
            },
            'mobile': {
                'model_quality': 'fast',
                'enable_visualization': False,
                'suggested_fps': 15,
                'description': '移动设备，性能受限'
            },
            'embedded': {
                'model_quality': 'ultra_fast',
                'enable_visualization': False,
                'suggested_fps': 10,
                'description': '嵌入式设备，极限性能'
            }
        }
        
        return recommendations.get(device_type, recommendations['desktop'])
    
    @classmethod
    def create_multi_exercise_session(cls, exercise_list: list,
                                     model_quality: str = 'balanced') -> Dict[str, PoseRecognizer]:
        """
        创建多运动会话
        
        Args:
            exercise_list: 运动列表，如 ['pushup_standard', 'squat_standard']
            model_quality: 模型质量
        
        Returns:
            运动识别器字典
        """
        recognizers = {}
        
        for exercise_name in exercise_list:
            try:
                recognizer = cls.create_recognizer(
                    preset_name=exercise_name,
                    model_quality=model_quality,
                    enable_visualization=True
                )
                recognizers[exercise_name] = recognizer
                logger.info(f"添加运动: {exercise_name}")
            except Exception as e:
                logger.error(f"创建运动识别器失败 {exercise_name}: {e}")
        
        logger.info(f"多运动会话创建完成，包含 {len(recognizers)} 种运动")
        return recognizers

# 便捷函数
def create_pushup_counter(model_quality: str = 'balanced') -> PoseRecognizer:
    """快速创建俯卧撑计数器"""
    return ExerciseFactory.create_pushup_recognizer(model_quality)

def create_squat_counter(model_quality: str = 'balanced') -> PoseRecognizer:
    """快速创建深蹲计数器"""
    return ExerciseFactory.create_squat_recognizer(model_quality)

def create_exercise_counter(exercise_name: str, model_quality: str = 'balanced') -> PoseRecognizer:
    """快速创建运动计数器"""
    return ExerciseFactory.create_recognizer(exercise_name, model_quality)

# 使用示例
if __name__ == "__main__":
    # 显示可用配置
    print("可用的运动预设:")
    for name, desc in ExerciseFactory.get_available_presets().items():
        print(f"  {name}: {desc}")
    
    print("\n可用的模型:")
    for quality, desc in ExerciseFactory.get_model_info().items():
        print(f"  {quality}: {desc}")
    
    # 创建俯卧撑识别器
    pushup_recognizer = create_pushup_counter('fast')
    print(f"\n创建俯卧撑识别器: {pushup_recognizer.exercise_type.value}")
    
    # 创建多运动会话
    exercises = ['pushup_standard', 'squat_standard', 'high_knee_left']
    session = ExerciseFactory.create_multi_exercise_session(exercises, 'balanced')
    print(f"\n多运动会话包含: {list(session.keys())}")
    
    # 设备推荐
    desktop_config = ExerciseFactory.recommend_config('desktop', accuracy_priority=True)
    print(f"\n桌面设备推荐配置: {desktop_config}")