#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人体姿态识别系统测试脚本
验证基本功能是否正常工作
"""

import sys
import os
import time
from pathlib import Path
import logging

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """
    测试模块导入
    """
    print("=== 测试模块导入 ===")
    
    try:
        from src.recognition import (
            PoseRecognizer, ExerciseFactory, ExerciseType,
            get_supported_exercises, get_model_recommendations
        )
        print("✓ 核心模块导入成功")
        return True
    except ImportError as e:
        print(f"✗ 模块导入失败: {e}")
        return False

def test_factory_creation():
    """
    测试工厂类创建
    """
    print("\n=== 测试工厂类创建 ===")
    
    try:
        from src.recognition import ExerciseFactory, get_supported_exercises
        
        # 测试获取支持的运动
        exercises = get_supported_exercises()
        print(f"✓ 支持的运动类型: {len(exercises)} 种")
        for name, desc in list(exercises.items())[:3]:  # 显示前3个
            print(f"  - {name}: {desc}")
        
        # 测试创建识别器
        recognizer = ExerciseFactory.create_pushup_recognizer('fast')
        print(f"✓ 俯卧撑识别器创建成功: {recognizer.exercise_type.value}")
        
        return True
    except Exception as e:
        print(f"✗ 工厂类测试失败: {e}")
        return False

def test_keypoint_config():
    """
    测试关键点配置
    """
    print("\n=== 测试关键点配置 ===")
    
    try:
        from src.recognition import KeypointConfig, get_keypoint_names
        
        # 测试关键点名称
        keypoints = get_keypoint_names()
        print(f"✓ 关键点数量: {len(keypoints)}")
        print(f"  示例关键点: {dict(list(keypoints.items())[:5])}")
        
        # 测试配置创建
        config = KeypointConfig(
            keypoints=[5, 7, 9],
            up_angle=145.0,
            down_angle=100.0
        )
        print(f"✓ 关键点配置创建成功: {config.keypoints}")
        
        return True
    except Exception as e:
        print(f"✗ 关键点配置测试失败: {e}")
        return False

def test_angle_calculation():
    """
    测试角度计算功能
    """
    print("\n=== 测试角度计算 ===")
    
    try:
        from src.recognition.pose_recognition import PoseRecognizer
        import numpy as np
        
        # 创建测试点
        p1 = np.array([0, 0])    # 肩膀
        p2 = np.array([1, 0])    # 肘部
        p3 = np.array([1, 1])    # 手腕
        
        # 计算角度
        angle = PoseRecognizer._calculate_angle(p1, p2, p3)
        print(f"✓ 角度计算成功: {angle:.1f}°")
        
        # 测试不同角度
        test_cases = [
            ([0, 0], [1, 0], [2, 0]),    # 180度
            ([0, 0], [1, 0], [1, 1]),    # 90度
            ([0, 0], [1, 0], [0, 1]),    # 45度
        ]
        
        for i, (pt1, pt2, pt3) in enumerate(test_cases):
            angle = PoseRecognizer._calculate_angle(
                np.array(pt1), np.array(pt2), np.array(pt3)
            )
            print(f"  测试案例 {i+1}: {angle:.1f}°")
        
        return True
    except Exception as e:
        print(f"✗ 角度计算测试失败: {e}")
        return False

def test_model_recommendations():
    """
    测试模型推荐
    """
    print("\n=== 测试模型推荐 ===")
    
    try:
        from src.recognition import get_model_recommendations, ExerciseFactory
        
        # 获取模型信息
        models = get_model_recommendations()
        print(f"✓ 可用模型: {len(models)} 个")
        for quality, desc in models.items():
            print(f"  - {quality}: {desc}")
        
        # 测试设备推荐
        devices = ['desktop', 'laptop', 'mobile']
        for device in devices:
            config = ExerciseFactory.recommend_config(device)
            print(f"  {device}: {config['model_quality']} - {config['description']}")
        
        return True
    except Exception as e:
        print(f"✗ 模型推荐测试失败: {e}")
        return False

def test_exercise_types():
    """
    测试运动类型枚举
    """
    print("\n=== 测试运动类型 ===")
    
    try:
        from src.recognition import ExerciseType
        
        # 测试所有运动类型
        exercise_types = list(ExerciseType)
        print(f"✓ 运动类型数量: {len(exercise_types)}")
        for exercise in exercise_types:
            print(f"  - {exercise.name}: {exercise.value}")
        
        return True
    except Exception as e:
        print(f"✗ 运动类型测试失败: {e}")
        return False

def test_statistics():
    """
    测试统计功能
    """
    print("\n=== 测试统计功能 ===")
    
    try:
        from src.recognition.pose_recognition import ExerciseStats, PoseState, ExerciseType
        
        # 创建统计对象
        stats = ExerciseStats(exercise_type=ExerciseType.PUSHUP)
        print(f"✓ 统计对象创建成功: 初始计数 {stats.count}")
        
        # 模拟更新统计
        stats.update_stats(120.0, 0.8, time.time())
        stats.current_state = PoseState.UP
        print(f"  更新后角度历史长度: {len(stats.angles_history)}")
        print(f"  当前状态: {stats.current_state.value}")
        
        # 测试计数增加
        stats.count += 1
        print(f"  计数增加后: {stats.count}")
        
        return True
    except Exception as e:
        print(f"✗ 统计功能测试失败: {e}")
        return False

def test_custom_recognizer():
    """
    测试自定义识别器创建
    """
    print("\n=== 测试自定义识别器 ===")
    
    try:
        from src.recognition import ExerciseFactory, ExerciseType
        
        # 创建自定义识别器
        recognizer = ExerciseFactory.create_custom_recognizer(
            keypoints=[5, 7, 9],
            up_angle=150.0,
            down_angle=90.0,
            exercise_type=ExerciseType.CUSTOM
        )
        
        print(f"✓ 自定义识别器创建成功")
        print(f"  运动类型: {recognizer.exercise_type.value}")
        print(f"  关键点: {recognizer.config.keypoints}")
        print(f"  角度范围: {recognizer.config.down_angle}° - {recognizer.config.up_angle}°")
        
        return True
    except Exception as e:
        print(f"✗ 自定义识别器测试失败: {e}")
        return False

def run_all_tests():
    """
    运行所有测试
    """
    print("人体姿态识别系统测试")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_factory_creation,
        test_keypoint_config,
        test_angle_calculation,
        test_model_recommendations,
        test_exercise_types,
        test_statistics,
        test_custom_recognizer,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ 测试 {test_func.__name__} 异常: {e}")
    
    print(f"\n=== 测试结果 ===")
    print(f"通过: {passed}/{total}")
    print(f"成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 所有测试通过！人体姿态识别系统基本功能正常")
        return True
    else:
        print("⚠️  部分测试失败，请检查相关模块")
        return False

def main():
    """
    主函数
    """
    try:
        # 检查依赖
        import numpy
        print(f"NumPy 版本: {numpy.__version__}")
        
        try:
            import cv2
            print(f"OpenCV 版本: {cv2.__version__}")
        except ImportError:
            print("警告: OpenCV 未安装，部分功能可能不可用")
        
        # 运行测试
        success = run_all_tests()
        
        if success:
            print("\n✅ 系统测试完成，可以开始使用人体姿态识别功能")
            print("\n快速开始:")
            print("  from src.recognition import quick_start_pushup")
            print("  recognizer = quick_start_pushup('fast')")
            print("  # 然后使用 recognizer.process_video() 或 process_frame()")
        else:
            print("\n❌ 系统测试未完全通过，请检查错误信息")
            
    except ImportError as e:
        print(f"依赖检查失败: {e}")
        print("请确保安装了必要的依赖包")
    except Exception as e:
        print(f"测试过程中出现异常: {e}")

if __name__ == "__main__":
    main()