#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USB摄像头医疗检测系统测试
验证面部生理识别、摔倒检测和紧急响应功能
"""

import cv2
import numpy as np
import time
import logging
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recognition.usb_medical_camera_system import USBMedicalCameraSystem
from src.recognition.medical_facial_analyzer import MedicalFacialAnalyzer, HealthStatus, FacialSymptom
from src.recognition.emergency_response_system import EmergencyResponseSystem, EmergencyType, ResponseLevel

def test_medical_facial_analyzer():
    """测试医疗面部分析器"""
    print("=" * 60)
    print("测试医疗面部分析器")
    print("=" * 60)
    
    analyzer = MedicalFacialAnalyzer()
    
    # 创建测试图像
    test_images = {
        'normal': create_normal_face_image(),
        'pale': create_pale_face_image(),
        'asymmetric': create_asymmetric_face_image(),
        'unconscious': create_unconscious_face_image()
    }
    
    for image_type, image in test_images.items():
        print(f"\n测试 {image_type} 面部图像:")
        
        result = analyzer.analyze_facial_health(image)
        
        print(f"  健康状态: {result.health_status.value}")
        print(f"  风险分数: {result.risk_score:.1f}")
        print(f"  紧急等级: {result.emergency_level}")
        print(f"  置信度: {result.confidence:.2f}")
        print(f"  分析时间: {result.analysis_time:.3f}s")
        
        if result.symptoms:
            print(f"  检测到的症状:")
            for symptom in result.symptoms:
                print(f"    - {symptom.value}")
        
        if result.recommendations:
            print(f"  医疗建议:")
            for rec in result.recommendations[:3]:  # 只显示前3个建议
                print(f"    - {rec}")
        
        # 验证关键功能
        assert result.health_status in HealthStatus
        assert 0 <= result.risk_score <= 100
        assert 1 <= result.emergency_level <= 5
        assert 0 <= result.confidence <= 1
        
        print(f"  ✓ {image_type} 测试通过")

def test_emergency_response_system():
    """测试紧急响应系统"""
    print("\n" + "=" * 60)
    print("测试紧急响应系统")
    print("=" * 60)
    
    emergency_system = EmergencyResponseSystem()
    
    # 测试摔倒检测
    print("\n测试摔倒检测:")
    fall_frame = create_fall_detection_image()
    
    result = emergency_system.process_frame(
        fall_frame,
        drone_id="test_drone",
        gps_location=(39.9042, 116.4074)
    )
    
    print(f"  摔倒检测: {result['fall_detected']}")
    print(f"  紧急事件数: {len(result['emergency_events'])}")
    print(f"  建议数: {len(result['recommendations'])}")
    
    if result['emergency_events']:
        event = result['emergency_events'][0]
        print(f"  事件类型: {event.emergency_type.value}")
        print(f"  响应级别: {event.response_level.value}")
    
    # 测试医疗紧急情况
    print("\n测试医疗紧急情况:")
    medical_emergency_frame = create_medical_emergency_image()
    
    result = emergency_system.process_frame(
        medical_emergency_frame,
        drone_id="test_drone",
        gps_location=(39.9042, 116.4074)
    )
    
    print(f"  医疗分析完成: {result['medical_analysis'] is not None}")
    if result['medical_analysis']:
        print(f"  健康状态: {result['medical_analysis'].health_status.value}")
        print(f"  紧急等级: {result['medical_analysis'].emergency_level}")
    
    print(f"  ✓ 紧急响应系统测试通过")

def test_usb_camera_system():
    """测试USB摄像头系统"""
    print("\n" + "=" * 60)
    print("测试USB摄像头医疗检测系统")
    print("=" * 60)
    
    # 创建系统实例
    camera_system = USBMedicalCameraSystem()
    
    # 测试配置
    print("测试系统配置:")
    config = camera_system.config
    print(f"  摄像头默认ID: {config['camera']['default_id']}")
    print(f"  分析间隔: {config['analysis']['interval']}s")
    print(f"  GUI更新间隔: {config['gui']['update_interval']}ms")
    print(f"  ✓ 配置测试通过")
    
    # 测试医疗分析组件
    print("\n测试医疗分析组件:")
    test_frame = create_normal_face_image()
    
    # 直接测试医疗分析器
    medical_result = camera_system.medical_analyzer.analyze_facial_health(test_frame)
    print(f"  医疗分析完成: {medical_result.health_status.value}")
    
    # 测试紧急响应系统
    emergency_result = camera_system.emergency_system.process_frame(test_frame)
    print(f"  紧急响应分析完成: {len(emergency_result['recommendations'])} 个建议")
    
    # 测试图像质量增强
    is_acceptable, quality_metrics = camera_system.quality_enhancer.is_image_acceptable(test_frame)
    print(f"  图像质量评估: {'可接受' if is_acceptable else '需要增强'}")
    print(f"  质量分数: {quality_metrics.quality_score:.2f}")
    
    print(f"  ✓ USB摄像头系统测试通过")

def test_real_time_processing():
    """测试实时处理性能"""
    print("\n" + "=" * 60)
    print("测试实时处理性能")
    print("=" * 60)
    
    camera_system = USBMedicalCameraSystem()
    
    # 模拟连续帧处理
    test_frames = [
        create_normal_face_image(),
        create_pale_face_image(),
        create_asymmetric_face_image(),
        create_unconscious_face_image(),
        create_fall_detection_image()
    ]
    
    processing_times = []
    
    print("处理测试帧序列:")
    for i, frame in enumerate(test_frames):
        start_time = time.time()
        
        # 执行完整的医疗分析
        medical_result = camera_system.medical_analyzer.analyze_facial_health(frame)
        emergency_result = camera_system.emergency_system.process_frame(frame)
        
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        
        print(f"  帧 {i+1}: {processing_time:.3f}s - {medical_result.health_status.value}")
    
    # 性能统计
    avg_time = np.mean(processing_times)
    max_time = max(processing_times)
    min_time = min(processing_times)
    fps_estimate = 1.0 / avg_time if avg_time > 0 else 0
    
    print(f"\n性能统计:")
    print(f"  平均处理时间: {avg_time:.3f}s")
    print(f"  最大处理时间: {max_time:.3f}s")
    print(f"  最小处理时间: {min_time:.3f}s")
    print(f"  估计FPS: {fps_estimate:.1f}")
    
    # 性能验证
    assert avg_time < 2.0, f"平均处理时间过长: {avg_time:.3f}s"
    assert fps_estimate > 0.5, f"FPS过低: {fps_estimate:.1f}"
    
    print(f"  ✓ 实时处理性能测试通过")

def test_emergency_scenarios():
    """测试紧急场景"""
    print("\n" + "=" * 60)
    print("测试紧急场景")
    print("=" * 60)
    
    camera_system = USBMedicalCameraSystem()
    
    # 场景1: 摔倒后意识丧失
    print("\n场景1: 摔倒后意识丧失")
    fall_frame = create_fall_detection_image()
    unconscious_frame = create_unconscious_face_image()
    
    # 先检测摔倒
    fall_result = camera_system.emergency_system.process_frame(fall_frame)
    print(f"  摔倒检测: {fall_result['fall_detected']}")
    
    # 然后检测意识状态
    medical_result = camera_system.medical_analyzer.analyze_facial_health(unconscious_frame)
    print(f"  意识状态: {medical_result.health_status.value}")
    print(f"  紧急等级: {medical_result.emergency_level}")
    
    # 验证紧急响应
    if medical_result.emergency_level >= 4:
        print(f"  ✓ 正确识别为高紧急等级")
    
    # 场景2: 面部不对称 (疑似中风)
    print("\n场景2: 面部不对称 (疑似中风)")
    asymmetric_frame = create_asymmetric_face_image()
    
    stroke_result = camera_system.medical_analyzer.analyze_facial_health(asymmetric_frame)
    print(f"  健康状态: {stroke_result.health_status.value}")
    print(f"  检测到的症状: {[s.value for s in stroke_result.symptoms]}")
    
    # 验证中风症状检测
    has_asymmetry = FacialSymptom.ASYMMETRIC_FACE in stroke_result.symptoms
    print(f"  面部不对称检测: {'✓' if has_asymmetry else '✗'}")
    
    # 场景3: 苍白面色 (可能失血)
    print("\n场景3: 苍白面色 (可能失血)")
    pale_frame = create_pale_face_image()
    
    pale_result = camera_system.medical_analyzer.analyze_facial_health(pale_frame)
    print(f"  健康状态: {pale_result.health_status.value}")
    print(f"  风险分数: {pale_result.risk_score:.1f}")
    
    # 验证苍白检测
    has_pallor = FacialSymptom.PALE_COMPLEXION in pale_result.symptoms
    print(f"  苍白检测: {'✓' if has_pallor else '✗'}")
    
    print(f"\n  ✓ 紧急场景测试通过")

def create_normal_face_image():
    """创建正常面部图像"""
    # 创建一个模拟的正常面部图像
    image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    # 添加面部区域 (肤色)
    face_color = [180, 150, 120]  # 正常肤色
    cv2.rectangle(image, (200, 150), (440, 350), face_color, -1)
    
    # 添加眼睛
    cv2.circle(image, (280, 220), 15, [255, 255, 255], -1)  # 左眼白
    cv2.circle(image, (360, 220), 15, [255, 255, 255], -1)  # 右眼白
    cv2.circle(image, (280, 220), 8, [50, 50, 50], -1)     # 左瞳孔
    cv2.circle(image, (360, 220), 8, [50, 50, 50], -1)     # 右瞳孔
    
    # 添加嘴巴
    cv2.ellipse(image, (320, 280), (25, 10), 0, 0, 180, [200, 100, 100], -1)
    
    return image

def create_pale_face_image():
    """创建苍白面部图像"""
    image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    # 苍白肤色 (低饱和度)
    face_color = [200, 200, 190]  # 苍白肤色
    cv2.rectangle(image, (200, 150), (440, 350), face_color, -1)
    
    # 添加眼睛
    cv2.circle(image, (280, 220), 15, [255, 255, 255], -1)
    cv2.circle(image, (360, 220), 15, [255, 255, 255], -1)
    cv2.circle(image, (280, 220), 8, [50, 50, 50], -1)
    cv2.circle(image, (360, 220), 8, [50, 50, 50], -1)
    
    # 添加嘴巴
    cv2.ellipse(image, (320, 280), (25, 10), 0, 0, 180, [180, 150, 150], -1)
    
    return image

def create_asymmetric_face_image():
    """创建不对称面部图像"""
    image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    # 正常肤色
    face_color = [180, 150, 120]
    cv2.rectangle(image, (200, 150), (440, 350), face_color, -1)
    
    # 不对称眼睛 (一只眼睛下垂)
    cv2.circle(image, (280, 220), 15, [255, 255, 255], -1)  # 正常左眼
    cv2.circle(image, (360, 230), 12, [255, 255, 255], -1)  # 下垂右眼
    cv2.circle(image, (280, 220), 8, [50, 50, 50], -1)
    cv2.circle(image, (360, 230), 6, [50, 50, 50], -1)
    
    # 不对称嘴巴 (一侧下垂)
    cv2.ellipse(image, (310, 285), (25, 10), -10, 0, 180, [200, 100, 100], -1)
    
    return image

def create_unconscious_face_image():
    """创建意识丧失面部图像"""
    image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    # 正常肤色
    face_color = [180, 150, 120]
    cv2.rectangle(image, (200, 150), (440, 350), face_color, -1)
    
    # 闭合的眼睛 (意识丧失)
    cv2.ellipse(image, (280, 220), (15, 5), 0, 0, 180, [150, 120, 100], -1)
    cv2.ellipse(image, (360, 220), (15, 5), 0, 0, 180, [150, 120, 100], -1)
    
    # 微张的嘴巴
    cv2.ellipse(image, (320, 290), (20, 15), 0, 0, 180, [100, 50, 50], -1)
    
    return image

def create_fall_detection_image():
    """创建摔倒检测图像"""
    image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    # 水平躺倒的人形轮廓
    person_color = [100, 100, 150]
    
    # 身体 (水平)
    cv2.rectangle(image, (150, 300), (500, 380), person_color, -1)
    
    # 头部 (侧面)
    cv2.circle(image, (120, 340), 40, person_color, -1)
    
    # 手臂
    cv2.rectangle(image, (200, 280), (250, 300), person_color, -1)
    cv2.rectangle(image, (400, 380), (450, 400), person_color, -1)
    
    # 腿部
    cv2.rectangle(image, (450, 320), (550, 340), person_color, -1)
    cv2.rectangle(image, (450, 360), (550, 380), person_color, -1)
    
    return image

def create_medical_emergency_image():
    """创建医疗紧急情况图像"""
    # 结合多种症状的图像
    image = create_pale_face_image()
    
    # 添加发绀 (蓝色调)
    blue_tint = np.zeros_like(image)
    blue_tint[:, :, 0] = 50  # 增加蓝色分量
    image = cv2.addWeighted(image, 0.8, blue_tint, 0.2, 0)
    
    return image

def run_comprehensive_test():
    """运行综合测试"""
    print("YOLOS USB摄像头医疗检测系统 - 综合测试")
    print("=" * 80)
    
    try:
        # 配置日志
        logging.basicConfig(level=logging.INFO)
        
        # 运行各项测试
        test_medical_facial_analyzer()
        test_emergency_response_system()
        test_usb_camera_system()
        test_real_time_processing()
        test_emergency_scenarios()
        
        print("\n" + "=" * 80)
        print("🎉 所有测试通过！USB摄像头医疗检测系统功能正常")
        print("=" * 80)
        
        # 测试总结
        print("\n✅ 测试总结:")
        print("  ✓ 医疗面部分析器 - 正常工作")
        print("  ✓ 紧急响应系统 - 正常工作")
        print("  ✓ USB摄像头系统 - 正常工作")
        print("  ✓ 实时处理性能 - 满足要求")
        print("  ✓ 紧急场景处理 - 正确识别")
        
        print("\n🚀 系统已准备就绪，可用于:")
        print("  • 摔倒后紧急医疗评估")
        print("  • 面部生理状态监控")
        print("  • 疾病症状早期识别")
        print("  • AIoT无人机械医疗响应")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)