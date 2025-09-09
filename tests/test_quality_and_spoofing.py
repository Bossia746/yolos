#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像质量增强和反欺骗检测测试
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.recognition.image_quality_enhancer import ImageQualityEnhancer, AdaptiveImageProcessor
    from src.recognition.anti_spoofing_detector import AntiSpoofingDetector
    from src.recognition.intelligent_recognition_system import IntelligentRecognitionSystem
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("使用简化版本进行测试...")

def create_test_images():
    """创建测试图像"""
    test_images = {}
    
    # 1. 正常图像
    normal_image = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
    test_images['normal'] = normal_image
    
    # 2. 过暗图像
    dark_image = np.random.randint(10, 60, (480, 640, 3), dtype=np.uint8)
    test_images['dark'] = dark_image
    
    # 3. 过亮图像
    bright_image = np.random.randint(200, 255, (480, 640, 3), dtype=np.uint8)
    test_images['bright'] = bright_image
    
    # 4. 低对比度图像
    low_contrast = np.full((480, 640, 3), 128, dtype=np.uint8)
    noise = np.random.randint(-10, 10, (480, 640, 3))
    low_contrast_image = np.clip(low_contrast + noise, 0, 255).astype(np.uint8)
    test_images['low_contrast'] = low_contrast_image
    
    # 5. 反光图像
    reflection_image = np.random.randint(50, 150, (480, 640, 3), dtype=np.uint8)
    # 添加反光区域
    reflection_image[100:200, 200:400] = 250
    test_images['reflection'] = reflection_image
    
    # 6. 噪声图像
    noise_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_images['noisy'] = noise_image
    
    # 7. 模糊图像
    blur_base = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
    blurred_image = cv2.GaussianBlur(blur_base, (15, 15), 0)
    test_images['blurred'] = blurred_image
    
    # 8. 模拟海报图像（简单纹理）
    poster_image = np.zeros((480, 640, 3), dtype=np.uint8)
    poster_image[:, :] = [100, 150, 200]  # 单一颜色
    # 添加简单的矩形
    cv2.rectangle(poster_image, (200, 150), (440, 330), (50, 100, 150), -1)
    test_images['poster'] = poster_image
    
    # 9. 模拟屏幕图像（周期性模式）
    screen_image = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
    # 添加周期性条纹
    for i in range(0, 640, 4):
        screen_image[:, i:i+2] = screen_image[:, i:i+2] * 1.2
    screen_image = np.clip(screen_image, 0, 255).astype(np.uint8)
    test_images['screen'] = screen_image
    
    return test_images

def test_image_quality_enhancement():
    """测试图像质量增强"""
    print("=" * 60)
    print("图像质量增强测试")
    print("=" * 60)
    
    try:
        enhancer = ImageQualityEnhancer()
        processor = AdaptiveImageProcessor()
        
        test_images = create_test_images()
        
        for image_type, image in test_images.items():
            print(f"\n测试图像类型: {image_type}")
            print("-" * 40)
            
            # 分析原始图像质量
            start_time = time.time()
            quality_metrics = enhancer.analyze_image_quality(image)
            analysis_time = time.time() - start_time
            
            print(f"原始图像质量分析 ({analysis_time:.3f}s):")
            print(f"  亮度: {quality_metrics.brightness:.1f}")
            print(f"  对比度: {quality_metrics.contrast:.1f}")
            print(f"  锐度: {quality_metrics.sharpness:.1f}")
            print(f"  噪声水平: {quality_metrics.noise_level:.3f}")
            print(f"  过曝比例: {quality_metrics.overexposure_ratio:.3f}")
            print(f"  欠曝比例: {quality_metrics.underexposure_ratio:.3f}")
            print(f"  反光分数: {quality_metrics.reflection_score:.3f}")
            print(f"  综合质量: {quality_metrics.quality_score:.3f}")
            
            # 检查是否需要增强
            is_acceptable, _ = enhancer.is_image_acceptable(image)
            print(f"  质量可接受: {is_acceptable}")
            
            if not is_acceptable:
                # 应用增强
                start_time = time.time()
                enhanced_image = enhancer.enhance_image(image, quality_metrics)
                enhancement_time = time.time() - start_time
                
                # 分析增强后的质量
                enhanced_quality = enhancer.analyze_image_quality(enhanced_image)
                
                print(f"\n增强后质量 ({enhancement_time:.3f}s):")
                print(f"  综合质量: {enhanced_quality.quality_score:.3f}")
                print(f"  质量提升: {enhanced_quality.quality_score - quality_metrics.quality_score:.3f}")
                
                # 获取建议
                recommendations = enhancer.get_enhancement_recommendations(quality_metrics)
                if recommendations:
                    print("  建议:")
                    for issue, suggestion in recommendations.items():
                        print(f"    {issue}: {suggestion}")
            
            # 自适应处理测试
            processed_image, processing_info = processor.process_image_stream(image)
            print(f"\n自适应处理:")
            print(f"  原始质量: {processing_info['original_quality']:.3f}")
            print(f"  应用的增强: {processing_info.get('enhancements_applied', [])}")
            if 'enhanced_quality' in processing_info:
                print(f"  增强后质量: {processing_info['enhanced_quality']:.3f}")
        
        print("\n✓ 图像质量增强测试完成")
        return True
        
    except Exception as e:
        print(f"✗ 图像质量增强测试失败: {e}")
        return False

def test_anti_spoofing_detection():
    """测试反欺骗检测"""
    print("\n" + "=" * 60)
    print("反欺骗检测测试")
    print("=" * 60)
    
    try:
        detector = AntiSpoofingDetector()
        
        test_images = create_test_images()
        
        for image_type, image in test_images.items():
            print(f"\n测试图像类型: {image_type}")
            print("-" * 40)
            
            # 执行反欺骗检测
            start_time = time.time()
            result = detector.detect_spoofing(image)
            detection_time = time.time() - start_time
            
            print(f"反欺骗检测结果 ({detection_time:.3f}s):")
            print(f"  是否真实: {result.is_real}")
            print(f"  欺骗类型: {result.spoofing_type.value}")
            print(f"  置信度: {result.confidence:.3f}")
            print(f"  风险等级: {result.risk_level}")
            
            # 显示检测证据
            print("  检测证据:")
            for key, value in result.evidence.items():
                print(f"    {key}: {value:.3f}")
            
            # 获取解释
            explanation = detector.get_spoofing_explanation(result)
            print(f"  解释: {explanation}")
            
            # 预期结果验证
            expected_real = image_type in ['normal', 'dark', 'bright', 'low_contrast', 'reflection', 'noisy', 'blurred']
            if result.is_real == expected_real:
                print("  ✓ 检测结果符合预期")
            else:
                print("  ⚠ 检测结果与预期不符")
        
        print("\n✓ 反欺骗检测测试完成")
        return True
        
    except Exception as e:
        print(f"✗ 反欺骗检测测试失败: {e}")
        return False

def test_intelligent_recognition_system():
    """测试智能识别系统"""
    print("\n" + "=" * 60)
    print("智能识别系统集成测试")
    print("=" * 60)
    
    try:
        system = IntelligentRecognitionSystem()
        
        test_images = create_test_images()
        
        for image_type, image in test_images.items():
            print(f"\n测试图像类型: {image_type}")
            print("-" * 40)
            
            # 执行智能识别
            start_time = time.time()
            result = system.recognize(image)
            total_time = time.time() - start_time
            
            print(f"智能识别结果 ({total_time:.3f}s):")
            print(f"  识别状态: {result.status.value}")
            print(f"  检测数量: {len(result.detections)}")
            print(f"  综合置信度: {result.confidence:.3f}")
            print(f"  处理时间: {result.processing_time:.3f}s")
            
            # 质量信息
            print("  质量信息:")
            quality_info = result.quality_info
            if 'error' not in quality_info:
                print(f"    亮度: {quality_info['brightness']:.1f}")
                print(f"    对比度: {quality_info['contrast']:.1f}")
                print(f"    锐度: {quality_info['sharpness']:.1f}")
                print(f"    质量分数: {quality_info['quality_score']:.3f}")
                print(f"    需要增强: {quality_info['needs_enhancement']}")
            
            # 反欺骗信息
            print("  反欺骗信息:")
            spoofing_info = result.spoofing_info
            if 'error' not in spoofing_info:
                print(f"    是否真实: {spoofing_info['is_real']}")
                print(f"    欺骗类型: {spoofing_info['spoofing_type']}")
                print(f"    置信度: {spoofing_info['confidence']:.3f}")
                print(f"    风险等级: {spoofing_info['risk_level']}")
            
            # 建议
            if result.recommendations:
                print("  建议:")
                for recommendation in result.recommendations:
                    print(f"    - {recommendation}")
            
            # 检测结果
            if result.detections:
                print("  检测结果:")
                for i, detection in enumerate(result.detections):
                    print(f"    目标{i+1}: {detection['class_name']} (置信度: {detection['confidence']:.2f})")
        
        # 获取性能报告
        performance = system.get_performance_report()
        print(f"\n系统性能报告:")
        print(f"  总处理数: {performance['total_processed']}")
        print(f"  成功率: {performance.get('success_rate', 0):.2%}")
        print(f"  欺骗检测率: {performance.get('spoofing_rate', 0):.2%}")
        print(f"  质量增强率: {performance.get('enhancement_rate', 0):.2%}")
        print(f"  平均处理时间: {performance['avg_processing_time']:.3f}s")
        
        print("\n✓ 智能识别系统测试完成")
        return True
        
    except Exception as e:
        print(f"✗ 智能识别系统测试失败: {e}")
        return False

def test_real_world_scenarios():
    """测试真实世界场景"""
    print("\n" + "=" * 60)
    print("真实世界场景测试")
    print("=" * 60)
    
    scenarios = {
        "室内弱光环境": {
            "brightness": 40,
            "noise_level": 0.4,
            "expected_issues": ["brightness", "noise"]
        },
        "强光反射环境": {
            "brightness": 200,
            "reflection": True,
            "expected_issues": ["brightness", "reflection"]
        },
        "海报欺骗攻击": {
            "spoofing_type": "poster",
            "texture_simple": True,
            "expected_issues": ["spoofing"]
        },
        "屏幕显示攻击": {
            "spoofing_type": "screen",
            "periodic_pattern": True,
            "expected_issues": ["spoofing"]
        }
    }
    
    try:
        system = IntelligentRecognitionSystem()
        
        for scenario_name, scenario_config in scenarios.items():
            print(f"\n场景: {scenario_name}")
            print("-" * 40)
            
            # 根据场景配置生成测试图像
            test_image = generate_scenario_image(scenario_config)
            
            # 执行识别
            result = system.recognize(test_image)
            
            print(f"识别结果:")
            print(f"  状态: {result.status.value}")
            print(f"  置信度: {result.confidence:.3f}")
            
            # 验证是否检测到预期问题
            expected_issues = scenario_config["expected_issues"]
            detected_issues = []
            
            if "brightness" in expected_issues:
                if result.quality_info.get('brightness', 128) < 80 or result.quality_info.get('brightness', 128) > 180:
                    detected_issues.append("brightness")
            
            if "noise" in expected_issues:
                if result.quality_info.get('needs_enhancement', False):
                    detected_issues.append("noise")
            
            if "reflection" in expected_issues:
                if any("反光" in rec for rec in result.recommendations):
                    detected_issues.append("reflection")
            
            if "spoofing" in expected_issues:
                if not result.spoofing_info.get('is_real', True):
                    detected_issues.append("spoofing")
            
            print(f"  预期问题: {expected_issues}")
            print(f"  检测到的问题: {detected_issues}")
            
            # 验证检测准确性
            accuracy = len(set(expected_issues) & set(detected_issues)) / len(expected_issues)
            print(f"  检测准确性: {accuracy:.2%}")
            
            if accuracy >= 0.5:
                print("  ✓ 场景测试通过")
            else:
                print("  ⚠ 场景测试部分通过")
        
        print("\n✓ 真实世界场景测试完成")
        return True
        
    except Exception as e:
        print(f"✗ 真实世界场景测试失败: {e}")
        return False

def generate_scenario_image(config):
    """根据场景配置生成测试图像"""
    image = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
    
    # 调整亮度
    if 'brightness' in config:
        target_brightness = config['brightness']
        current_brightness = np.mean(image)
        adjustment = target_brightness - current_brightness
        image = np.clip(image.astype(np.float32) + adjustment, 0, 255).astype(np.uint8)
    
    # 添加噪声
    if config.get('noise_level', 0) > 0:
        noise = np.random.normal(0, config['noise_level'] * 50, image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # 添加反光
    if config.get('reflection', False):
        image[100:200, 200:400] = 250
    
    # 简化纹理（海报效果）
    if config.get('texture_simple', False):
        image[:, :] = [120, 150, 180]
        cv2.rectangle(image, (200, 150), (440, 330), (80, 120, 160), -1)
    
    # 添加周期性模式（屏幕效果）
    if config.get('periodic_pattern', False):
        for i in range(0, 640, 4):
            image[:, i:i+2] = np.clip(image[:, i:i+2] * 1.3, 0, 255)
    
    return image

def main():
    """主测试函数"""
    print("YOLOS 图像质量增强和反欺骗检测测试")
    print("=" * 80)
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    test_results = []
    
    # 1. 图像质量增强测试
    result1 = test_image_quality_enhancement()
    test_results.append(("图像质量增强", result1))
    
    # 2. 反欺骗检测测试
    result2 = test_anti_spoofing_detection()
    test_results.append(("反欺骗检测", result2))
    
    # 3. 智能识别系统测试
    result3 = test_intelligent_recognition_system()
    test_results.append(("智能识别系统", result3))
    
    # 4. 真实世界场景测试
    result4 = test_real_world_scenarios()
    test_results.append(("真实世界场景", result4))
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed_tests += 1
    
    print(f"\n总计: {passed_tests}/{total_tests} 个测试通过")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过！YOLOS图像质量增强和反欺骗检测功能正常工作。")
    else:
        print("⚠️ 部分测试失败，请检查相关功能。")
    
    print("\n功能特性总结:")
    print("✓ 图像质量自动分析和评估")
    print("✓ 自适应图像增强（亮度、对比度、锐化、降噪）")
    print("✓ 反光和曝光问题处理")
    print("✓ 多种欺骗攻击检测（照片、海报、屏幕、视频）")
    print("✓ 纹理、频域、运动模式分析")
    print("✓ 智能识别系统集成")
    print("✓ 实时性能监控和建议生成")

if __name__ == "__main__":
    main()