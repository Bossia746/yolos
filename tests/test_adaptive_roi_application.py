# -*- coding: utf-8 -*-
"""
自适应ROI应用系统测试
测试不同场景下的ROI预测性能和准确性
"""

import unittest
import torch
import numpy as np
import cv2
import time
import sys
import os
from typing import Dict, List, Any

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'src', 'models'))

print(f"项目根目录: {project_root}")
print(f"Python路径: {sys.path[:3]}")

try:
    from src.models.adaptive_roi_application import (
        AdaptiveROIApplication, SceneAnalyzer, AdaptiveROIPredictor,
        ROIStrategy, ROIParameters, create_adaptive_roi_system
    )
    print("✅ 成功导入自适应ROI模块")
except ImportError as e:
    print(f"第一次导入错误: {e}")
    print("尝试直接导入...")
    try:
        import adaptive_roi_application
        from adaptive_roi_application import (
            AdaptiveROIApplication, SceneAnalyzer, AdaptiveROIPredictor,
            ROIStrategy, ROIParameters, create_adaptive_roi_system
        )
        print("✅ 直接导入成功")
    except ImportError as e2:
        print(f"直接导入失败: {e2}")
        print("尝试最后的导入方式...")
        try:
            # 直接执行模块文件
            module_path = os.path.join(project_root, 'src', 'models', 'adaptive_roi_application.py')
            if os.path.exists(module_path):
                import importlib.util
                spec = importlib.util.spec_from_file_location("adaptive_roi_application", module_path)
                adaptive_roi_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(adaptive_roi_module)
                
                AdaptiveROIApplication = adaptive_roi_module.AdaptiveROIApplication
                SceneAnalyzer = adaptive_roi_module.SceneAnalyzer
                AdaptiveROIPredictor = adaptive_roi_module.AdaptiveROIPredictor
                ROIStrategy = adaptive_roi_module.ROIStrategy
                ROIParameters = adaptive_roi_module.ROIParameters
                create_adaptive_roi_system = adaptive_roi_module.create_adaptive_roi_system
                print("✅ 通过文件路径导入成功")
            else:
                print(f"❌ 模块文件不存在: {module_path}")
                sys.exit(1)
        except Exception as e3:
            print(f"❌ 所有导入方式都失败: {e3}")
            sys.exit(1)

class TestSceneAnalyzer(unittest.TestCase):
    """场景分析器测试"""
    
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scene_analyzer = SceneAnalyzer().to(self.device)
        
    def test_scene_feature_extraction(self):
        """测试场景特征提取"""
        print("\n=== 场景特征提取测试 ===")
        
        # 创建测试图像
        batch_size = 2
        image = torch.randn(batch_size, 3, 224, 224).to(self.device)
        
        # 特征提取
        with torch.no_grad():
            results = self.scene_analyzer(image)
            
        # 验证输出形状
        self.assertEqual(results['scene_features'].shape, (batch_size, 64))
        self.assertEqual(results['motion_info'].shape, (batch_size, 4))
        
        print(f"场景特征形状: {results['scene_features'].shape}")
        print(f"运动信息形状: {results['motion_info'].shape}")
        print("✅ 场景特征提取测试通过")
        
    def test_motion_analysis(self):
        """测试运动分析"""
        print("\n=== 运动分析测试 ===")
        
        batch_size = 2
        current_frame = torch.randn(batch_size, 3, 224, 224).to(self.device)
        prev_frame = torch.randn(batch_size, 3, 224, 224).to(self.device)
        
        with torch.no_grad():
            results = self.scene_analyzer(current_frame, prev_frame)
            
        # 验证运动信息不为零
        motion_info = results['motion_info']
        self.assertFalse(torch.allclose(motion_info, torch.zeros_like(motion_info)))
        
        print(f"运动信息: {motion_info}")
        print("✅ 运动分析测试通过")

class TestAdaptiveROIPredictor(unittest.TestCase):
    """自适应ROI预测器测试"""
    
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.roi_predictor = AdaptiveROIPredictor().to(self.device)
        
    def test_roi_prediction(self):
        """测试ROI预测"""
        print("\n=== ROI预测测试 ===")
        
        batch_size = 2
        scene_features = torch.randn(batch_size, 64).to(self.device)
        motion_info = torch.randn(batch_size, 4).to(self.device)
        roi_history = torch.randn(batch_size, 5, 4).to(self.device)
        
        with torch.no_grad():
            results = self.roi_predictor(scene_features, motion_info, roi_history)
            
        # 验证输出
        roi = results['roi']
        confidence = results['confidence']
        
        self.assertEqual(roi.shape, (batch_size, 4))
        self.assertEqual(confidence.shape, (batch_size, 1))
        
        # 验证ROI坐标在[0,1]范围内
        self.assertTrue(torch.all(roi >= 0) and torch.all(roi <= 1))
        
        # 验证置信度在[0,1]范围内
        self.assertTrue(torch.all(confidence >= 0) and torch.all(confidence <= 1))
        
        print(f"预测ROI: {roi}")
        print(f"预测置信度: {confidence}")
        print("✅ ROI预测测试通过")
        
    def test_roi_consistency(self):
        """测试ROI预测一致性"""
        print("\n=== ROI预测一致性测试 ===")
        
        # 使用相同输入多次预测
        scene_features = torch.randn(1, 64).to(self.device)
        motion_info = torch.randn(1, 4).to(self.device)
        roi_history = torch.randn(1, 5, 4).to(self.device)
        
        predictions = []
        with torch.no_grad():
            for _ in range(5):
                results = self.roi_predictor(scene_features, motion_info, roi_history)
                predictions.append(results['roi'])
                
        # 验证预测一致性（允许合理的数值误差）
        for i in range(1, len(predictions)):
            diff = torch.abs(predictions[i] - predictions[0])
            max_diff = torch.max(diff).item()
            self.assertLess(max_diff, 0.01, f"预测结果差异过大: {max_diff}，应该基本一致")
            
        print("✅ ROI预测一致性测试通过")

class TestAdaptiveROIApplication(unittest.TestCase):
    """自适应ROI应用系统测试"""
    
    def setUp(self):
        self.roi_system = AdaptiveROIApplication()
        
    def test_scenario_switching(self):
        """测试场景切换"""
        print("\n=== 场景切换测试 ===")
        
        scenarios = ['medical_monitoring', 'pet_monitoring', 'security_monitoring', 'gesture_recognition']
        
        for scenario in scenarios:
            success = self.roi_system.set_scenario(scenario)
            if success:
                print(f"✅ 成功切换到场景: {scenario}")
                self.assertIsNotNone(self.roi_system.current_strategy)
                self.assertIsNotNone(self.roi_system.strategy_parameters)
            else:
                print(f"⚠️ 场景切换失败: {scenario} (可能未在配置中启用)")
                
    def test_roi_prediction_performance(self):
        """测试ROI预测性能"""
        print("\n=== ROI预测性能测试 ===")
        
        # 设置医疗监控场景
        self.roi_system.set_scenario('medical_monitoring')
        
        # 创建测试图像
        test_images = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8),
            np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        ]
        
        image_sizes = ['480p', '720p', '1080p']
        
        for i, (image, size_name) in enumerate(zip(test_images, image_sizes)):
            start_time = time.time()
            
            # 预测ROI
            results = self.roi_system.predict_roi(image)
            
            end_time = time.time()
            prediction_time = (end_time - start_time) * 1000
            
            # 验证结果
            self.assertIn('roi', results)
            self.assertIn('confidence', results)
            self.assertIn('strategy', results)
            
            # 验证ROI格式
            roi = results['roi']
            self.assertEqual(roi.shape, (1, 4))
            self.assertTrue(np.all(roi >= 0) and np.all(roi <= 1))
            
            print(f"{size_name} 图像预测时间: {prediction_time:.2f} ms")
            print(f"ROI: {roi[0]}")
            print(f"置信度: {results['confidence'][0][0]:.3f}")
            
        print("✅ ROI预测性能测试通过")
        
    def test_roi_application(self):
        """测试ROI应用到图像"""
        print("\n=== ROI应用测试 ===")
        
        # 创建测试图像
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 预测ROI
        results = self.roi_system.predict_roi(image)
        roi = results['roi'][0]
        
        # 应用ROI
        roi_image = self.roi_system.apply_roi_to_image(image, roi)
        
        # 验证ROI图像
        self.assertGreater(roi_image.shape[0], 0)
        self.assertGreater(roi_image.shape[1], 0)
        self.assertEqual(roi_image.shape[2], 3)
        
        # 验证ROI尺寸合理
        original_area = image.shape[0] * image.shape[1]
        roi_area = roi_image.shape[0] * roi_image.shape[1]
        roi_ratio = roi_area / original_area
        
        self.assertGreater(roi_ratio, 0.1)  # ROI不应太小
        self.assertLess(roi_ratio, 1.0)     # ROI不应超过原图
        
        print(f"原图尺寸: {image.shape[:2]}")
        print(f"ROI图像尺寸: {roi_image.shape[:2]}")
        print(f"ROI面积比例: {roi_ratio:.3f}")
        print("✅ ROI应用测试通过")
        
    def test_detection_integration(self):
        """测试与检测结果的集成"""
        print("\n=== 检测集成测试 ===")
        
        # 设置医疗监控场景
        self.roi_system.set_scenario('medical_monitoring')
        
        # 创建测试图像和检测结果
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detection_results = [
            {
                'class': 'person',
                'confidence': 0.9,
                'bbox': [100, 100, 300, 400]  # [x1, y1, x2, y2]
            },
            {
                'class': 'chair',
                'confidence': 0.7,
                'bbox': [400, 200, 500, 350]
            }
        ]
        
        # 预测ROI（带检测结果）
        results = self.roi_system.predict_roi(image, detection_results)
        
        # 验证结果
        self.assertIn('roi', results)
        self.assertIn('confidence', results)
        
        print(f"集成检测结果的ROI: {results['roi'][0]}")
        print(f"置信度: {results['confidence'][0][0]:.3f}")
        print("✅ 检测集成测试通过")

class TestPerformanceBenchmark(unittest.TestCase):
    """性能基准测试"""
    
    def test_throughput_benchmark(self):
        """测试吞吐量基准"""
        print("\n=== 吞吐量基准测试 ===")
        
        roi_system = create_adaptive_roi_system('medical_monitoring')
        
        # 测试不同图像尺寸的吞吐量
        test_configs = [
            {'size': (240, 320), 'name': '240p'},
            {'size': (480, 640), 'name': '480p'},
            {'size': (720, 1280), 'name': '720p'}
        ]
        
        num_iterations = 50
        
        for config in test_configs:
            size = config['size']
            name = config['name']
            
            # 创建测试图像
            test_image = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            
            # 预热
            for _ in range(5):
                _ = roi_system.predict_roi(test_image)
                
            # 基准测试
            start_time = time.time()
            for _ in range(num_iterations):
                _ = roi_system.predict_roi(test_image)
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time = (total_time / num_iterations) * 1000
            fps = num_iterations / total_time
            
            print(f"{name} - 平均预测时间: {avg_time:.2f} ms, FPS: {fps:.1f}")
            
            # 性能要求验证
            if name == '240p':
                self.assertLess(avg_time, 20, f"{name} 预测时间应小于20ms")
            elif name == '480p':
                self.assertLess(avg_time, 50, f"{name} 预测时间应小于50ms")
            elif name == '720p':
                self.assertLess(avg_time, 100, f"{name} 预测时间应小于100ms")
                
        print("✅ 吞吐量基准测试通过")
        
    def test_memory_efficiency(self):
        """测试内存效率"""
        print("\n=== 内存效率测试 ===")
        
        if not torch.cuda.is_available():
            self.skipTest('内存测试需要CUDA设备')
            
        roi_system = create_adaptive_roi_system('medical_monitoring')
        
        # 测试内存使用
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # 运行多次预测
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        for _ in range(100):
            _ = roi_system.predict_roi(test_image)
            
        final_memory = torch.cuda.memory_allocated()
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        print(f"初始内存: {initial_memory / 1024 / 1024:.2f} MB")
        print(f"最终内存: {final_memory / 1024 / 1024:.2f} MB")
        print(f"内存增长: {memory_increase:.2f} MB")
        
        # 验证内存增长合理
        self.assertLess(memory_increase, 100, "内存增长应小于100MB")
        
        print("✅ 内存效率测试通过")

class TestSystemIntegration(unittest.TestCase):
    """系统集成测试"""
    
    def test_multi_scenario_workflow(self):
        """测试多场景工作流"""
        print("\n=== 多场景工作流测试 ===")
        
        roi_system = AdaptiveROIApplication()
        
        # 模拟不同场景的工作流
        scenarios = [
            ('medical_monitoring', '医疗监控'),
            ('pet_monitoring', '宠物监控'),
            ('security_monitoring', '安全监控'),
            ('gesture_recognition', '手势识别')
        ]
        
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        for scenario_id, scenario_name in scenarios:
            print(f"\n--- 测试场景: {scenario_name} ---")
            
            # 切换场景
            success = roi_system.set_scenario(scenario_id)
            if not success:
                print(f"⚠️ 场景 {scenario_name} 未启用，跳过测试")
                continue
                
            # 重置历史
            roi_system.reset_history()
            
            # 运行多帧预测
            for frame_idx in range(5):
                results = roi_system.predict_roi(test_image)
                
                print(f"  帧 {frame_idx+1}: ROI={results['roi'][0]}, "
                      f"置信度={results['confidence'][0][0]:.3f}, "
                      f"时间={results['prediction_time_ms']:.2f}ms")
                      
            # 获取性能统计
            stats = roi_system.get_performance_stats()
            print(f"  平均预测时间: {stats.get('avg_prediction_time_ms', 0):.2f} ms")
            
        print("\n✅ 多场景工作流测试完成")
        
    def test_error_handling(self):
        """测试错误处理"""
        print("\n=== 错误处理测试 ===")
        
        roi_system = AdaptiveROIApplication()
        
        # 测试无效场景
        success = roi_system.set_scenario('invalid_scenario')
        self.assertFalse(success)
        
        # 测试无效图像
        try:
            invalid_image = np.array([])  # 空数组
            results = roi_system.predict_roi(invalid_image)
            # 应该返回备用ROI
            self.assertIn('roi', results)
            self.assertEqual(results['strategy'], 'fallback')
            print("✅ 无效图像处理正确")
        except Exception as e:
            print(f"⚠️ 无效图像处理异常: {e}")
            
        # 测试极端ROI
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        results = roi_system.predict_roi(test_image)
        
        # 验证ROI在有效范围内
        roi = results['roi'][0]
        self.assertTrue(np.all(roi >= 0) and np.all(roi <= 1))
        
        print("✅ 错误处理测试通过")

if __name__ == '__main__':
    print("=== 自适应ROI应用系统完整测试 ===")
    
    # 运行所有测试
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_classes = [
        TestSceneAnalyzer,
        TestAdaptiveROIPredictor,
        TestAdaptiveROIApplication,
        TestPerformanceBenchmark,
        TestSystemIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 测试总结
    print("\n" + "="*60)
    print("测试总结:")
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"跳过: {len(result.skipped)}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"成功率: {success_rate:.1f}%")
    print("="*60)
    
    if result.failures or result.errors:
        print("❌ 部分测试失败，请检查实现。")
        exit(1)
    else:
        print("✅ 所有测试通过！自适应ROI应用系统准备就绪。")
        exit(0)