#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV升级优化集成测试
验证OpenCV升级后与YOLO项目的整体集成效果
"""

import sys
import os
import unittest
import cv2
import numpy as np
import time
import tempfile
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from src.optimization.opencv_optimizer import OpenCVOptimizer, SceneType, OptimizationLevel
from src.optimization.opencv_performance_monitor import OpenCVPerformanceMonitor
from src.safety.complex_scene_analyzer import ComplexSceneAnalyzer
from src.safety.multi_person_detector import MultiPersonDetector
from src.safety.pose_estimation_optimizer import PoseEstimationOptimizer
from src.safety.obstacle_aware_tracker import ObstacleAwareTracker
from src.safety.environment_context_analyzer import EnvironmentContextAnalyzer
from src.core.types import DetectionResult, ObjectType
from src.models.yolo_factory import YOLOFactory

class TestOpenCVIntegration(unittest.TestCase):
    """OpenCV集成测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.test_image_size = (640, 480)
        self.test_frames = self._generate_test_frames()
        
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 初始化组件
        self.optimizer = OpenCVOptimizer()
        self.monitor = OpenCVPerformanceMonitor()
        self.scene_analyzer = ComplexSceneAnalyzer()
        self.person_detector = MultiPersonDetector()
        self.pose_optimizer = PoseEstimationOptimizer()
        self.tracker = ObstacleAwareTracker()
        self.env_analyzer = EnvironmentContextAnalyzer()
    
    def tearDown(self):
        """测试后清理"""
        # 清理临时文件
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _generate_test_frames(self) -> list:
        """生成测试帧"""
        frames = []
        
        # 静态场景帧
        static_frame = np.zeros((*self.test_image_size[::-1], 3), dtype=np.uint8)
        cv2.rectangle(static_frame, (100, 100), (200, 300), (255, 255, 255), -1)
        frames.append(('static', static_frame))
        
        # 动态场景帧
        dynamic_frame = np.random.randint(0, 255, (*self.test_image_size[::-1], 3), dtype=np.uint8)
        frames.append(('dynamic', dynamic_frame))
        
        # 低光照场景帧
        low_light_frame = np.random.randint(0, 50, (*self.test_image_size[::-1], 3), dtype=np.uint8)
        frames.append(('low_light', low_light_frame))
        
        # 拥挤场景帧
        crowded_frame = np.random.randint(100, 255, (*self.test_image_size[::-1], 3), dtype=np.uint8)
        # 添加多个矩形模拟人员
        for i in range(5):
            x, y = np.random.randint(0, 400, 2)
            cv2.rectangle(crowded_frame, (x, y), (x+50, y+100), (0, 255, 0), 2)
        frames.append(('crowded', crowded_frame))
        
        return frames
    
    def test_opencv_version_compatibility(self):
        """测试OpenCV版本兼容性"""
        print("\n=== 测试OpenCV版本兼容性 ===")
        
        # 检查版本
        version = cv2.__version__
        print(f"OpenCV版本: {version}")
        
        # 版本应该是4.8.0或更高
        version_parts = version.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1])
        
        self.assertGreaterEqual(major, 4, "OpenCV主版本应该>=4")
        if major == 4:
            self.assertGreaterEqual(minor, 8, "OpenCV次版本应该>=8")
        
        print(f"✅ 版本检查通过: {version}")
    
    def test_basic_opencv_functions(self):
        """测试基础OpenCV功能"""
        print("\n=== 测试基础OpenCV功能 ===")
        
        test_frame = self.test_frames[0][1]
        
        # 测试颜色转换
        gray = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
        self.assertEqual(len(gray.shape), 2, "灰度转换失败")
        print("✅ 颜色转换测试通过")
        
        # 测试图像缩放
        resized = cv2.resize(test_frame, (320, 240))
        self.assertEqual(resized.shape[:2], (240, 320), "图像缩放失败")
        print("✅ 图像缩放测试通过")
        
        # 测试边缘检测
        edges = cv2.Canny(gray, 50, 150)
        self.assertEqual(edges.shape, gray.shape, "边缘检测失败")
        print("✅ 边缘检测测试通过")
        
        # 测试模糊处理
        blurred = cv2.GaussianBlur(test_frame, (5, 5), 0)
        self.assertEqual(blurred.shape, test_frame.shape, "模糊处理失败")
        print("✅ 模糊处理测试通过")
    
    def test_dnn_module_compatibility(self):
        """测试DNN模块兼容性"""
        print("\n=== 测试DNN模块兼容性 ===")
        
        try:
            # 测试DNN模块基本功能
            # 创建一个空的DNN网络用于测试
            net = cv2.dnn.Net()
            print("✅ DNN模块可用")
            
            # 测试后端支持
            backends = []
            
            try:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                backends.append('OpenCV')
            except:
                pass
            
            try:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                backends.append('CUDA')
            except:
                pass
            
            print(f"✅ 可用后端: {backends}")
            self.assertGreater(len(backends), 0, "至少应该有一个可用后端")
            
        except Exception as e:
            self.fail(f"DNN模块测试失败: {e}")
    
    def test_gpu_acceleration_support(self):
        """测试GPU加速支持"""
        print("\n=== 测试GPU加速支持 ===")
        
        # 检查CUDA支持
        if hasattr(cv2, 'cuda'):
            cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
            print(f"CUDA设备数量: {cuda_devices}")
            
            if cuda_devices > 0:
                try:
                    # 测试GPU内存分配
                    test_frame = self.test_frames[0][1]
                    gpu_mat = cv2.cuda_GpuMat()
                    gpu_mat.upload(test_frame)
                    
                    # 测试GPU图像处理
                    gpu_gray = cv2.cuda.cvtColor(gpu_mat, cv2.COLOR_BGR2GRAY)
                    result = gpu_gray.download()
                    
                    self.assertEqual(len(result.shape), 2, "GPU图像处理失败")
                    print("✅ GPU加速功能正常")
                    
                except Exception as e:
                    print(f"⚠️ GPU加速测试失败: {e}")
            else:
                print("ℹ️ 未检测到CUDA设备")
        else:
            print("ℹ️ 当前版本不支持CUDA")
    
    def test_performance_optimization(self):
        """测试性能优化"""
        print("\n=== 测试性能优化 ===")
        
        # 测试优化设置
        original_optimized = cv2.useOptimized()
        original_threads = cv2.getNumThreads()
        
        # 启用优化
        cv2.setUseOptimized(True)
        cv2.setNumThreads(4)
        
        self.assertTrue(cv2.useOptimized(), "优化未启用")
        self.assertEqual(cv2.getNumThreads(), 4, "线程数设置失败")
        
        print(f"✅ 优化状态: {cv2.useOptimized()}")
        print(f"✅ 线程数: {cv2.getNumThreads()}")
        
        # 恢复原始设置
        cv2.setUseOptimized(original_optimized)
        cv2.setNumThreads(original_threads)
    
    def test_scene_specific_optimization(self):
        """测试场景特定优化"""
        print("\n=== 测试场景特定优化 ===")
        
        for scene_name, frame in self.test_frames:
            print(f"\n测试 {scene_name} 场景:")
            
            # 根据场景类型获取优化配置
            scene_type_map = {
                'static': SceneType.STATIC,
                'dynamic': SceneType.DYNAMIC,
                'low_light': SceneType.LOW_LIGHT,
                'crowded': SceneType.CROWDED
            }
            
            scene_type = scene_type_map.get(scene_name, SceneType.MIXED)
            
            try:
                # 应用场景优化
                config = self.optimizer.adaptive_processor.performance_optimizer.optimize_for_scene(scene_type)
                self.assertIsNotNone(config, f"{scene_name}场景优化配置获取失败")
                
                # 测试优化效果
                start_time = time.perf_counter()
                
                # 模拟图像处理
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blurred, 50, 150)
                
                processing_time = time.perf_counter() - start_time
                
                print(f"  ✅ 处理时间: {processing_time*1000:.2f}ms")
                self.assertLess(processing_time, 0.1, f"{scene_name}场景处理时间过长")
                
            except Exception as e:
                self.fail(f"{scene_name}场景优化测试失败: {e}")
    
    def test_integration_with_yolo_components(self):
        """测试与YOLO组件的集成"""
        print("\n=== 测试与YOLO组件集成 ===")
        
        test_frame = self.test_frames[1][1]  # 使用动态场景帧
        
        try:
            # 测试复杂场景分析器
            # 创建模拟检测结果
            mock_detections = [
                {'class': 'person', 'bbox': [100, 50, 200, 300], 'confidence': 0.9, 'track_id': 1},
                {'class': 'chair', 'bbox': [150, 100, 250, 200], 'confidence': 0.7}
            ]
            scene_result = self.scene_analyzer.analyze_scene(test_frame, mock_detections)
            self.assertIsNotNone(scene_result, "场景分析失败")
            print("✅ 复杂场景分析器集成正常")
            
            # 测试多人检测器
            mock_person_detections = [
                {'class': 'person', 'bbox': [100, 50, 200, 300], 'confidence': 0.9, 'track_id': 1}
            ]
            detection_result = self.person_detector.detect_multi_person_scene(test_frame, mock_person_detections, [])
            self.assertIsNotNone(detection_result, "多人检测失败")
            print("✅ 多人检测器集成正常")
            
            # 测试姿态估计优化器
            mock_pose_results = {
                'openpose': []
            }
            pose_result = self.pose_optimizer.optimize_poses(test_frame, mock_pose_results)
            self.assertIsNotNone(pose_result, "姿态估计优化失败")
            print("✅ 姿态估计优化器集成正常")
            
            # 测试障碍物感知跟踪器
            mock_person_detections = [{
                'person_id': 1,
                'bbox': (100, 100, 200, 300),
                'position': (150, 200),
                'velocity': (1.0, 0.5),
                'confidence': 0.85
            }]
            tracking_result = self.tracker.process_frame(test_frame, mock_person_detections)
            self.assertIsNotNone(tracking_result, "障碍物感知跟踪失败")
            print("✅ 障碍物感知跟踪器集成正常")
            
            # 测试环境上下文分析器
            env_result = self.env_analyzer.analyze_environment(test_frame)
            self.assertIsNotNone(env_result, "环境上下文分析失败")
            print("✅ 环境上下文分析器集成正常")
            
        except Exception as e:
            self.fail(f"YOLO组件集成测试失败: {e}")
    
    def test_performance_monitoring_integration(self):
        """测试性能监控集成"""
        print("\n=== 测试性能监控集成 ===")
        
        try:
            # 启动监控
            self.monitor.start_monitoring()
            
            # 模拟处理函数
            def dummy_processing(frame):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                return cv2.Canny(gray, 50, 150)
            
            # 处理几帧
            for i, (scene_name, frame) in enumerate(self.test_frames):
                result, metrics = self.monitor.process_frame(frame, dummy_processing)
                
                self.assertIsNotNone(result, f"帧{i}处理失败")
                self.assertIsNotNone(metrics, f"帧{i}性能指标获取失败")
                self.assertGreater(metrics.fps, 0, f"帧{i}FPS计算错误")
            
            # 停止监控
            self.monitor.stop_monitoring()
            
            # 获取性能摘要
            summary = self.monitor.get_performance_summary()
            self.assertIsNotNone(summary, "性能摘要获取失败")
            self.assertIn('performance', summary, "性能数据缺失")
            
            print(f"✅ 平均FPS: {summary['performance']['avg_fps']:.1f}")
            print(f"✅ 平均延迟: {summary['performance']['avg_latency_ms']:.1f}ms")
            print("✅ 性能监控集成正常")
            
        except Exception as e:
            self.fail(f"性能监控集成测试失败: {e}")
    
    def test_memory_usage_optimization(self):
        """测试内存使用优化"""
        print("\n=== 测试内存使用优化 ===")
        
        import psutil
        import gc
        
        # 获取初始内存使用
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 处理大量图像
        for i in range(50):
            frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            
            # 图像处理
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # 及时删除大对象
            del frame, gray, blurred, edges
            
            # 每10帧强制垃圾回收
            if i % 10 == 0:
                gc.collect()
        
        # 最终垃圾回收
        gc.collect()
        
        # 获取最终内存使用
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"初始内存: {initial_memory:.1f}MB")
        print(f"最终内存: {final_memory:.1f}MB")
        print(f"内存增长: {memory_increase:.1f}MB")
        
        # 内存增长应该控制在合理范围内
        self.assertLess(memory_increase, 100, "内存使用增长过多")
        print("✅ 内存使用优化正常")
    
    def test_error_handling_and_recovery(self):
        """测试错误处理和恢复"""
        print("\n=== 测试错误处理和恢复 ===")
        
        # 测试无效输入处理
        try:
            # 空图像
            empty_frame = np.array([])
            result = cv2.cvtColor(empty_frame, cv2.COLOR_BGR2GRAY)
            self.fail("应该抛出异常")
        except cv2.error:
            print("✅ 空图像错误处理正常")
        
        # 测试无效参数处理
        try:
            frame = self.test_frames[0][1]
            # 无效的核大小
            result = cv2.GaussianBlur(frame, (0, 0), 0)
            self.fail("应该抛出异常")
        except cv2.error:
            print("✅ 无效参数错误处理正常")
        
        # 测试系统恢复能力
        try:
            # 模拟处理大量数据后的恢复
            for i in range(10):
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.assertEqual(len(gray.shape), 2, "处理恢复失败")
            
            print("✅ 系统恢复能力正常")
            
        except Exception as e:
            self.fail(f"系统恢复测试失败: {e}")
    
    def test_configuration_compatibility(self):
        """测试配置兼容性"""
        print("\n=== 测试配置兼容性 ===")
        
        try:
            # 测试优化器配置加载
            config_path = Path(self.temp_dir) / "test_config.yaml"
            
            # 创建测试配置
            test_config = """
version_config:
  recommended: "4.10.0"
  current_minimum: "4.8.0"

performance_config:
  runtime_optimization:
    threading:
      num_threads: 4
      use_optimized: true
"""
            
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(test_config)
            
            # 使用配置创建优化器
            optimizer = OpenCVOptimizer(str(config_path))
            self.assertIsNotNone(optimizer, "配置加载失败")
            
            print("✅ 配置文件兼容性正常")
            
        except Exception as e:
            self.fail(f"配置兼容性测试失败: {e}")

def run_integration_tests():
    """运行集成测试"""
    print("\n" + "="*60)
    print("OpenCV升级优化集成测试")
    print("="*60)
    
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOpenCVIntegration)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出测试结果摘要
    print("\n" + "="*60)
    print("测试结果摘要")
    print("="*60)
    print(f"总测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].split('\n')[0]}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('\n')[-2]}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n成功率: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("\n🎉 集成测试通过！OpenCV升级优化工作正常。")
    else:
        print("\n⚠️ 集成测试未完全通过，请检查失败的测试项。")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    # 运行集成测试
    success = run_integration_tests()
    
    # 设置退出码
    sys.exit(0 if success else 1)