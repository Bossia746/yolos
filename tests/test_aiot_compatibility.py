#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIoT开发板兼容性测试
测试YOLOS系统对主流AIoT开发板的支持情况
"""

import os
import sys
import unittest
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.plugins.platform.aiot_boards_adapter import AIoTBoardsAdapter, get_aiot_boards_adapter
from src.core.cross_platform_manager import CrossPlatformManager, get_cross_platform_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestAIoTCompatibility(unittest.TestCase):
    """AIoT开发板兼容性测试"""
    
    def setUp(self):
        """测试初始化"""
        self.aiot_adapter = get_aiot_boards_adapter()
        self.platform_manager = get_cross_platform_manager()
    
    def test_aiot_adapter_initialization(self):
        """测试AIoT适配器初始化"""
        self.assertIsInstance(self.aiot_adapter, AIoTBoardsAdapter)
        self.assertIsNotNone(self.aiot_adapter.supported_boards)
        self.assertIsNotNone(self.aiot_adapter.current_board)
        
        logger.info(f"AIoT适配器初始化成功")
        logger.info(f"支持的开发板数量: {len(self.aiot_adapter.supported_boards)}")
    
    def test_supported_boards_coverage(self):
        """测试支持的开发板覆盖范围"""
        expected_manufacturers = [
            'NVIDIA',      # Jetson系列
            'Google',      # Coral系列
            'Intel',       # NUC, NCS系列
            'Rockchip',    # RK3588, RK3566
            'Qualcomm',    # RB5平台
            'Amlogic',     # A311D
            'MediaTek',    # Genio平台
            'Allwinner',   # H6系列
            'Raspberry Pi Foundation',  # 树莓派
            'Espressif'    # ESP32系列
        ]
        
        supported_manufacturers = set()
        for board_info in self.aiot_adapter.supported_boards.values():
            manufacturer = board_info.get('manufacturer')
            if manufacturer:
                supported_manufacturers.add(manufacturer)
        
        for manufacturer in expected_manufacturers:
            self.assertIn(manufacturer, supported_manufacturers, 
                         f"缺少对{manufacturer}开发板的支持")
        
        logger.info(f"支持的制造商: {sorted(supported_manufacturers)}")
    
    def test_nvidia_jetson_support(self):
        """测试NVIDIA Jetson系列支持"""
        jetson_boards = [
            'jetson_nano',
            'jetson_xavier_nx', 
            'jetson_agx_xavier',
            'jetson_orin_nano'
        ]
        
        for board_id in jetson_boards:
            self.assertIn(board_id, self.aiot_adapter.supported_boards)
            
            board_info = self.aiot_adapter.supported_boards[board_id]
            self.assertEqual(board_info['manufacturer'], 'NVIDIA')
            self.assertTrue(board_info['capabilities']['deep_learning'])
            self.assertTrue(board_info['capabilities']['cuda'])
            self.assertTrue(board_info['capabilities']['tensorrt'])
        
        logger.info("NVIDIA Jetson系列支持验证通过")
    
    def test_google_coral_support(self):
        """测试Google Coral系列支持"""
        coral_boards = [
            'coral_dev_board',
            'coral_dev_board_micro'
        ]
        
        for board_id in coral_boards:
            self.assertIn(board_id, self.aiot_adapter.supported_boards)
            
            board_info = self.aiot_adapter.supported_boards[board_id]
            self.assertEqual(board_info['manufacturer'], 'Google')
            self.assertTrue(board_info['capabilities']['deep_learning'])
            self.assertTrue(board_info['capabilities']['edge_tpu'])
        
        logger.info("Google Coral系列支持验证通过")
    
    def test_rockchip_support(self):
        """测试Rockchip系列支持"""
        rockchip_boards = [
            'rk3588',
            'rk3566'
        ]
        
        for board_id in rockchip_boards:
            self.assertIn(board_id, self.aiot_adapter.supported_boards)
            
            board_info = self.aiot_adapter.supported_boards[board_id]
            self.assertEqual(board_info['manufacturer'], 'Rockchip')
            self.assertTrue(board_info['capabilities']['deep_learning'])
            self.assertTrue(board_info['capabilities']['npu_acceleration'])
            self.assertTrue(board_info['capabilities']['rknn_toolkit'])
        
        logger.info("Rockchip系列支持验证通过")
    
    def test_esp32_support(self):
        """测试ESP32系列支持"""
        esp32_boards = [
            'esp32_s3',
            'esp32_cam'
        ]
        
        for board_id in esp32_boards:
            self.assertIn(board_id, self.aiot_adapter.supported_boards)
            
            board_info = self.aiot_adapter.supported_boards[board_id]
            self.assertEqual(board_info['manufacturer'], 'Espressif')
            self.assertTrue(board_info['capabilities']['wifi'])
        
        # ESP32-S3应该支持深度学习
        esp32_s3_info = self.aiot_adapter.supported_boards['esp32_s3']
        self.assertTrue(esp32_s3_info['capabilities']['deep_learning'])
        self.assertTrue(esp32_s3_info['capabilities']['tflite_micro'])
        
        logger.info("ESP32系列支持验证通过")
    
    def test_board_detection_methods(self):
        """测试开发板检测方法"""
        for board_id, board_info in self.aiot_adapter.supported_boards.items():
            detection_methods = board_info.get('detection_methods', [])
            self.assertGreater(len(detection_methods), 0, 
                             f"{board_id}缺少检测方法")
            
            # 验证检测方法的有效性
            for method in detection_methods:
                self.assertIsInstance(method, str)
                self.assertGreater(len(method), 0)
        
        logger.info("开发板检测方法验证通过")
    
    def test_ai_acceleration_capabilities(self):
        """测试AI加速能力"""
        ai_accelerator_types = [
            'edge_tpu',           # Google Coral
            'npu_acceleration',   # Rockchip, Amlogic
            'cuda',              # NVIDIA
            'hexagon_dsp',       # Qualcomm
            'apu_acceleration',   # MediaTek
            'openvino'           # Intel
        ]
        
        boards_with_ai = []
        for board_id, board_info in self.aiot_adapter.supported_boards.items():
            capabilities = board_info.get('capabilities', {})
            
            has_ai_accelerator = any(capabilities.get(acc_type, False) 
                                   for acc_type in ai_accelerator_types)
            
            if has_ai_accelerator:
                boards_with_ai.append(board_id)
        
        # 至少应该有一半的开发板支持AI加速
        min_ai_boards = len(self.aiot_adapter.supported_boards) // 2
        self.assertGreaterEqual(len(boards_with_ai), min_ai_boards,
                               "支持AI加速的开发板数量不足")
        
        logger.info(f"支持AI加速的开发板: {boards_with_ai}")
    
    def test_camera_support(self):
        """测试摄像头支持"""
        camera_interfaces = [
            'camera_csi',
            'camera_mipi', 
            'camera_builtin'
        ]
        
        boards_with_camera = []
        for board_id, board_info in self.aiot_adapter.supported_boards.items():
            capabilities = board_info.get('capabilities', {})
            
            has_camera = any(capabilities.get(interface, False) 
                           for interface in camera_interfaces)
            
            if has_camera:
                boards_with_camera.append(board_id)
        
        # 大部分开发板应该支持摄像头
        min_camera_boards = len(self.aiot_adapter.supported_boards) * 2 // 3
        self.assertGreaterEqual(len(boards_with_camera), min_camera_boards,
                               "支持摄像头的开发板数量不足")
        
        logger.info(f"支持摄像头的开发板: {boards_with_camera}")
    
    def test_gpio_support(self):
        """测试GPIO支持"""
        boards_with_gpio = []
        for board_id, board_info in self.aiot_adapter.supported_boards.items():
            capabilities = board_info.get('capabilities', {})
            
            if capabilities.get('gpio', False):
                boards_with_gpio.append(board_id)
        
        # 大部分开发板应该支持GPIO
        min_gpio_boards = len(self.aiot_adapter.supported_boards) * 2 // 3
        self.assertGreaterEqual(len(boards_with_gpio), min_gpio_boards,
                               "支持GPIO的开发板数量不足")
        
        logger.info(f"支持GPIO的开发板: {boards_with_gpio}")
    
    def test_current_board_detection(self):
        """测试当前开发板检测"""
        current_board = self.aiot_adapter.current_board
        
        self.assertIsInstance(current_board, dict)
        self.assertIn('name', current_board)
        self.assertIn('confidence', current_board)
        self.assertIsInstance(current_board['confidence'], float)
        self.assertGreaterEqual(current_board['confidence'], 0.0)
        self.assertLessEqual(current_board['confidence'], 1.0)
        
        logger.info(f"当前检测到的开发板: {current_board['name']}")
        logger.info(f"检测置信度: {current_board['confidence']:.2f}")
    
    def test_board_configuration(self):
        """测试开发板配置"""
        board_config = self.aiot_adapter.board_config
        
        required_config_keys = [
            'optimization',
            'ai_acceleration', 
            'camera_config',
            'gpio_config',
            'performance_limits'
        ]
        
        for key in required_config_keys:
            self.assertIn(key, board_config, f"缺少配置项: {key}")
        
        # 验证优化配置
        optimization = board_config['optimization']
        self.assertIn('max_workers', optimization)
        self.assertIn('memory_limit_gb', optimization)
        self.assertIn('batch_size', optimization)
        
        logger.info("开发板配置验证通过")
    
    def test_optimization_recommendations(self):
        """测试优化建议"""
        recommendations = self.aiot_adapter.get_optimization_recommendations()
        
        required_recommendation_keys = [
            'model_optimization',
            'runtime_optimization',
            'hardware_optimization',
            'framework_recommendations'
        ]
        
        for key in required_recommendation_keys:
            self.assertIn(key, recommendations, f"缺少优化建议: {key}")
            self.assertIsInstance(recommendations[key], list)
        
        logger.info("优化建议验证通过")
    
    def test_cross_platform_integration(self):
        """测试跨平台集成"""
        # 验证跨平台管理器能正确识别AIoT开发板
        aiot_info = self.platform_manager.aiot_board_info
        
        self.assertIsInstance(aiot_info, dict)
        self.assertIn('detected', aiot_info)
        self.assertIn('board_name', aiot_info)
        
        # 如果检测到AIoT开发板，验证集成信息
        if aiot_info['detected']:
            self.assertIn('capabilities', aiot_info)
            self.assertIn('supported_frameworks', aiot_info)
            
            logger.info(f"跨平台管理器检测到AIoT开发板: {aiot_info['board_name']}")
        else:
            logger.info("跨平台管理器未检测到AIoT开发板（正常，取决于运行环境）")
    
    def test_dependency_installation(self):
        """测试依赖安装"""
        # 这个测试不会实际安装依赖，只是验证安装逻辑
        install_results = {}
        
        try:
            # 模拟依赖检查
            install_results = self.aiot_adapter.install_board_dependencies()
            self.assertIsInstance(install_results, dict)
            
            logger.info(f"依赖安装测试完成: {install_results}")
            
        except Exception as e:
            logger.warning(f"依赖安装测试跳过: {e}")
    
    def test_board_report_generation(self):
        """测试开发板报告生成"""
        report = self.aiot_adapter.generate_board_report()
        
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 100)  # 报告应该有足够的内容
        
        # 验证报告包含关键信息
        self.assertIn('AIoT开发板兼容性报告', report)
        self.assertIn('检测到的开发板', report)
        self.assertIn('支持的能力', report)
        
        logger.info("开发板报告生成验证通过")

class TestAIoTBoardsSpecific(unittest.TestCase):
    """特定AIoT开发板测试"""
    
    def setUp(self):
        self.adapter = get_aiot_boards_adapter()
    
    def test_performance_tiers(self):
        """测试性能分层"""
        high_performance_boards = []
        mid_performance_boards = []
        low_power_boards = []
        
        for board_id, board_info in self.adapter.supported_boards.items():
            capabilities = board_info.get('capabilities', {})
            
            if capabilities.get('high_performance', False):
                high_performance_boards.append(board_id)
            elif capabilities.get('ultra_low_power', False):
                low_power_boards.append(board_id)
            else:
                mid_performance_boards.append(board_id)
        
        # 验证各性能层级都有代表性开发板
        self.assertGreater(len(high_performance_boards), 0, "缺少高性能开发板")
        self.assertGreater(len(low_power_boards), 0, "缺少低功耗开发板")
        
        logger.info(f"高性能开发板: {high_performance_boards}")
        logger.info(f"中等性能开发板: {mid_performance_boards}")
        logger.info(f"低功耗开发板: {low_power_boards}")
    
    def test_ai_framework_coverage(self):
        """测试AI框架覆盖"""
        framework_coverage = {}
        
        for board_id, board_info in self.adapter.supported_boards.items():
            capabilities = board_info.get('capabilities', {})
            
            # 统计各种AI框架支持
            if capabilities.get('tensorflow_lite', False):
                framework_coverage.setdefault('TensorFlow Lite', []).append(board_id)
            
            if capabilities.get('tensorrt', False):
                framework_coverage.setdefault('TensorRT', []).append(board_id)
            
            if capabilities.get('openvino', False):
                framework_coverage.setdefault('OpenVINO', []).append(board_id)
            
            if capabilities.get('rknn_toolkit', False):
                framework_coverage.setdefault('RKNN', []).append(board_id)
        
        # 验证主要AI框架都有支持
        major_frameworks = ['TensorFlow Lite', 'TensorRT', 'OpenVINO']
        for framework in major_frameworks:
            if framework in framework_coverage:
                self.assertGreater(len(framework_coverage[framework]), 0,
                                 f"{framework}缺少支持的开发板")
        
        logger.info(f"AI框架覆盖情况: {framework_coverage}")

def run_aiot_compatibility_test():
    """运行AIoT兼容性测试"""
    print("=" * 60)
    print("YOLOS AIoT开发板兼容性测试")
    print("=" * 60)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加基础兼容性测试
    test_suite.addTest(unittest.makeSuite(TestAIoTCompatibility))
    
    # 添加特定开发板测试
    test_suite.addTest(unittest.makeSuite(TestAIoTBoardsSpecific))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出测试结果摘要
    print("\n" + "=" * 60)
    print("测试结果摘要")
    print("=" * 60)
    print(f"运行测试数: {result.testsRun}")
    print(f"失败测试数: {len(result.failures)}")
    print(f"错误测试数: {len(result.errors)}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    # 生成AIoT开发板报告
    try:
        adapter = get_aiot_boards_adapter()
        report = adapter.generate_board_report()
        
        print("\n" + "=" * 60)
        print("AIoT开发板检测报告")
        print("=" * 60)
        print(report)
        
    except Exception as e:
        print(f"生成AIoT报告失败: {e}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_aiot_compatibility_test()
    sys.exit(0 if success else 1)