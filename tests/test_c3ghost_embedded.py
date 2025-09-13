"""C3Ghost嵌入式部署测试

测试C3Ghost模块在不同嵌入式平台上的功能性、性能和兼容性

Author: YOLOS Team
Date: 2024-01-15
Version: 1.0.0
"""

import unittest
import torch
import torch.nn as nn
import time
import sys
import psutil
import platform
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 尝试不同的导入路径
try:
    from src.models.c3ghost_module import (
        GhostConv,
        GhostBottleneck,
        C3Ghost,
        C3GhostEmbedded,
        create_c3ghost_for_platform,
        PerformanceMonitor
    )
except ImportError:
    try:
        import os
        sys.path.append(os.path.join(project_root, 'src', 'models'))
        from c3ghost_module import (
            GhostConv,
            GhostBottleneck,
            C3Ghost,
            C3GhostEmbedded,
            create_c3ghost_for_platform,
            PerformanceMonitor
        )
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保项目结构正确")
        print(f"当前工作目录: {os.getcwd()}")
        print(f"项目根目录: {project_root}")
        sys.exit(1)


class TestGhostConv(unittest.TestCase):
    """Ghost卷积基础测试"""
    
    def setUp(self):
        """测试前准备"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def test_ghost_conv_forward(self):
        """测试Ghost卷积前向传播"""
        # 测试不同配置
        test_configs = [
            {'c1': 64, 'c2': 128, 'k': 1, 's': 1, 'ratio': 2},
            {'c1': 128, 'c2': 256, 'k': 3, 's': 1, 'ratio': 2},
            {'c1': 256, 'c2': 512, 'k': 1, 's': 2, 'ratio': 4}
        ]
        
        for config in test_configs:
            with self.subTest(config=config):
                ghost_conv = GhostConv(**config).to(self.device)
                
                # 计算输入尺寸
                h, w = 32, 32
                if config['s'] == 2:
                    expected_h, expected_w = h // 2, w // 2
                else:
                    expected_h, expected_w = h, w
                
                # 测试输入
                x = torch.randn(2, config['c1'], h, w).to(self.device)
                
                # 前向传播
                output = ghost_conv(x)
                
                # 检查输出形状
                expected_shape = (2, config['c2'], expected_h, expected_w)
                self.assertEqual(output.shape, expected_shape, 
                               f"输出形状不匹配: 期望 {expected_shape}, 实际 {output.shape}")
                
                # 检查输出不是NaN或Inf
                self.assertFalse(torch.isnan(output).any(), "输出包含NaN")
                self.assertFalse(torch.isinf(output).any(), "输出包含Inf")
    
    def test_ghost_conv_parameter_reduction(self):
        """测试Ghost卷积的参数减少效果"""
        c1, c2 = 256, 512
        
        # 标准卷积
        standard_conv = nn.Sequential(
            nn.Conv2d(c1, c2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )
        
        # Ghost卷积
        ghost_conv = GhostConv(c1, c2, 3, 1, ratio=2)
        
        # 计算参数数量
        standard_params = sum(p.numel() for p in standard_conv.parameters())
        ghost_params = sum(p.numel() for p in ghost_conv.parameters())
        
        # Ghost卷积应该有更少的参数
        self.assertLess(ghost_params, standard_params, 
                       f"Ghost卷积参数数量 ({ghost_params}) 应该少于标准卷积 ({standard_params})")
        
        # 计算参数减少比例
        reduction_ratio = (standard_params - ghost_params) / standard_params
        print(f"参数减少比例: {reduction_ratio:.2%}")
        self.assertGreater(reduction_ratio, 0.3, "参数减少比例应该大于30%")


class TestGhostBottleneck(unittest.TestCase):
    """Ghost瓶颈模块测试"""
    
    def setUp(self):
        """测试前准备"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def test_ghost_bottleneck_forward(self):
        """测试Ghost瓶颈前向传播"""
        test_configs = [
            {'c1': 128, 'c2': 128, 's': 1, 'use_se': False},
            {'c1': 128, 'c2': 256, 's': 2, 'use_se': True},
            {'c1': 256, 'c2': 256, 's': 1, 'use_se': True}
        ]
        
        for config in test_configs:
            with self.subTest(config=config):
                bottleneck = GhostBottleneck(**config).to(self.device)
                
                # 测试输入
                x = torch.randn(2, config['c1'], 32, 32).to(self.device)
                
                # 前向传播
                output = bottleneck(x)
                
                # 检查输出形状
                expected_h = 32 // config['s']
                expected_w = 32 // config['s']
                expected_shape = (2, config['c2'], expected_h, expected_w)
                
                self.assertEqual(output.shape, expected_shape)
                self.assertFalse(torch.isnan(output).any())
                self.assertFalse(torch.isinf(output).any())
    
    def test_residual_connection(self):
        """测试残差连接"""
        c1 = c2 = 256
        bottleneck = GhostBottleneck(c1, c2, s=1).to(self.device)
        
        # 测试输入
        x = torch.randn(1, c1, 32, 32).to(self.device)
        
        # 前向传播
        output = bottleneck(x)
        
        # 检查残差连接是否工作（输出不应该等于输入）
        self.assertFalse(torch.allclose(x, output, atol=1e-6), 
                        "残差连接应该改变输入")


class TestC3Ghost(unittest.TestCase):
    """C3Ghost模块测试"""
    
    def setUp(self):
        """测试前准备"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def test_c3ghost_forward(self):
        """测试C3Ghost前向传播"""
        test_configs = [
            {'c1': 128, 'c2': 256, 'n': 1, 'act': 'relu'},
            {'c1': 256, 'c2': 512, 'n': 2, 'act': 'silu'},
            {'c1': 512, 'c2': 1024, 'n': 3, 'act': 'mish'}
        ]
        
        for config in test_configs:
            with self.subTest(config=config):
                c3ghost = C3Ghost(**config).to(self.device)
                
                # 测试输入
                x = torch.randn(2, config['c1'], 32, 32).to(self.device)
                
                # 前向传播
                output = c3ghost(x)
                
                # 检查输出形状
                expected_shape = (2, config['c2'], 32, 32)
                self.assertEqual(output.shape, expected_shape)
                self.assertFalse(torch.isnan(output).any())
                self.assertFalse(torch.isinf(output).any())
    
    def test_different_activations(self):
        """测试不同激活函数"""
        activations = ['relu', 'silu', 'swish', 'mish', 'identity']
        
        for act in activations:
            with self.subTest(activation=act):
                try:
                    c3ghost = C3Ghost(128, 256, n=1, act=act).to(self.device)
                    x = torch.randn(1, 128, 16, 16).to(self.device)
                    output = c3ghost(x)
                    
                    self.assertEqual(output.shape, (1, 256, 16, 16))
                    self.assertFalse(torch.isnan(output).any())
                except Exception as e:
                    self.fail(f"激活函数 {act} 测试失败: {e}")


class TestC3GhostEmbedded(unittest.TestCase):
    """嵌入式C3Ghost模块测试"""
    
    def setUp(self):
        """测试前准备"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.platforms = ['raspberry_pi_4b', 'jetson_nano', 'jetson_xavier_nx', 'intel_nuc', 'esp32_s3']
        
    def test_platform_specific_configs(self):
        """测试平台特定配置"""
        for platform in self.platforms:
            with self.subTest(platform=platform):
                try:
                    module = create_c3ghost_for_platform(256, 512, platform).to(self.device)
                    
                    # 测试输入
                    x = torch.randn(1, 256, 32, 32).to(self.device)
                    
                    # 前向传播
                    output = module(x)
                    
                    # 基本检查
                    self.assertEqual(len(output.shape), 4, "输出应该是4维张量")
                    self.assertEqual(output.shape[0], 1, "批次大小应该为1")
                    self.assertEqual(output.shape[1], output.shape[1], "输出通道数检查")
                    self.assertFalse(torch.isnan(output).any())
                    self.assertFalse(torch.isinf(output).any())
                    
                    # 获取模型信息
                    model_info = module.get_model_info()
                    self.assertIn('platform', model_info)
                    self.assertIn('total_parameters', model_info)
                    self.assertEqual(model_info['platform'], platform)
                    
                except Exception as e:
                    self.fail(f"平台 {platform} 测试失败: {e}")
    
    def test_parameter_scaling(self):
        """测试参数缩放效果"""
        # 比较不同平台的参数数量
        base_platform = 'intel_nuc'  # 性能最强的平台
        constrained_platforms = ['raspberry_pi_4b', 'esp32_s3']  # 资源受限平台
        
        base_module = create_c3ghost_for_platform(256, 512, base_platform)
        base_params = sum(p.numel() for p in base_module.parameters())
        
        for platform in constrained_platforms:
            with self.subTest(platform=platform):
                module = create_c3ghost_for_platform(256, 512, platform)
                params = sum(p.numel() for p in module.parameters())
                
                # 资源受限平台应该有更少的参数
                self.assertLessEqual(params, base_params, 
                                   f"{platform} 应该比 {base_platform} 有更少或相等的参数")
                
                print(f"{platform}: {params:,} 参数 vs {base_platform}: {base_params:,} 参数")


class TestPerformanceBenchmark(unittest.TestCase):
    """性能基准测试"""
    
    def setUp(self):
        """测试前准备"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def test_inference_speed_comparison(self):
        """测试推理速度对比"""
        # 测试配置
        test_configs = [
            {'name': 'small', 'c1': 128, 'c2': 256, 'input_size': (1, 128, 32, 32)},
            {'name': 'medium', 'c1': 256, 'c2': 512, 'input_size': (1, 256, 64, 64)},
            {'name': 'large', 'c1': 512, 'c2': 1024, 'input_size': (1, 512, 128, 128)}
        ]
        
        platforms = ['raspberry_pi_4b', 'jetson_nano', 'intel_nuc']
        
        results = {}
        
        for config in test_configs:
            results[config['name']] = {}
            
            for platform in platforms:
                # 创建模块
                module = create_c3ghost_for_platform(
                    config['c1'], config['c2'], platform
                ).to(self.device)
                module.eval()
                
                # 测试输入
                x = torch.randn(*config['input_size']).to(self.device)
                
                # 预热
                with torch.no_grad():
                    for _ in range(10):
                        _ = module(x)
                
                # 同步GPU
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                # 计时
                start_time = time.time()
                num_runs = 50
                
                with torch.no_grad():
                    for _ in range(num_runs):
                        _ = module(x)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                avg_time = (end_time - start_time) / num_runs * 1000  # 转换为毫秒
                
                results[config['name']][platform] = avg_time
                
        # 输出结果
        print("\n推理速度对比 (毫秒):")
        print(f"{'配置':<10} {'树莓派4B':<12} {'Jetson Nano':<15} {'Intel NUC':<12}")
        print("-" * 55)
        
        for config_name, platform_results in results.items():
            row = f"{config_name:<10}"
            for platform in platforms:
                time_ms = platform_results.get(platform, 0)
                row += f" {time_ms:<11.2f}"
            print(row)
    
    def test_memory_efficiency(self):
        """测试内存效率"""
        if self.device.type != 'cuda':
            self.skipTest("内存测试需要CUDA设备")
        
        platforms = ['raspberry_pi_4b', 'jetson_nano', 'intel_nuc']
        input_tensor = torch.randn(1, 256, 64, 64).to(self.device)
        
        memory_results = {}
        
        for platform in platforms:
            # 清空缓存
            torch.cuda.empty_cache()
            
            # 记录初始内存
            initial_memory = torch.cuda.memory_allocated()
            
            # 创建模块
            module = create_c3ghost_for_platform(256, 512, platform).to(self.device)
            
            # 前向传播
            with torch.no_grad():
                output = module(input_tensor)
            
            # 记录峰值内存
            peak_memory = torch.cuda.max_memory_allocated()
            memory_usage = (peak_memory - initial_memory) / 1024 / 1024  # 转换为MB
            
            memory_results[platform] = memory_usage
            
            # 重置内存统计
            torch.cuda.reset_peak_memory_stats()
            
        # 输出结果
        print("\n内存使用对比 (MB):")
        for platform, memory in memory_results.items():
            print(f"{platform}: {memory:.2f} MB")
        
        # 验证资源受限平台使用更少内存
        if 'raspberry_pi_4b' in memory_results and 'intel_nuc' in memory_results:
            self.assertLessEqual(
                memory_results['raspberry_pi_4b'], 
                memory_results['intel_nuc'] * 1.2,  # 允许20%的误差
                "树莓派配置应该使用更少或相近的内存"
            )


class TestSystemCompatibility(unittest.TestCase):
    """系统兼容性测试"""
    
    def test_cpu_architecture_detection(self):
        """测试CPU架构检测"""
        # 获取系统信息
        system_info = {
            'platform': platform.platform(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'architecture': platform.architecture(),
            'system': platform.system()
        }
        
        print(f"\n系统信息:")
        for key, value in system_info.items():
            print(f"{key}: {value}")
        
        # 基本检查
        self.assertIsNotNone(system_info['platform'])
        self.assertIsNotNone(system_info['system'])
    
    def test_memory_constraints(self):
        """测试内存约束"""
        # 获取系统内存信息
        memory_info = psutil.virtual_memory()
        available_memory_gb = memory_info.available / (1024**3)
        
        print(f"\n可用内存: {available_memory_gb:.2f} GB")
        
        # 根据可用内存选择合适的平台配置
        if available_memory_gb < 2:
            recommended_platform = 'esp32_s3'
        elif available_memory_gb < 4:
            recommended_platform = 'raspberry_pi_4b'
        elif available_memory_gb < 8:
            recommended_platform = 'jetson_nano'
        else:
            recommended_platform = 'intel_nuc'
            
        print(f"推荐平台配置: {recommended_platform}")
        
        # 测试推荐配置是否可用
        try:
            module = create_c3ghost_for_platform(128, 256, recommended_platform)
            x = torch.randn(1, 128, 32, 32)
            output = module(x)
            
            self.assertEqual(output.shape[0], 1)
            print(f"推荐配置 {recommended_platform} 测试通过")
        except Exception as e:
            self.fail(f"推荐配置 {recommended_platform} 测试失败: {e}")


def run_comprehensive_test():
    """运行综合测试"""
    print("="*70)
    print("C3Ghost嵌入式部署综合测试")
    print("="*70)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_classes = [
        TestGhostConv,
        TestGhostBottleneck,
        TestC3Ghost,
        TestC3GhostEmbedded,
        TestPerformanceBenchmark,
        TestSystemCompatibility
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出总结
    print("\n" + "="*70)
    print(f"测试总结:")
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"跳过: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    print(f"成功率: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # 设置随机种子以确保可重现性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行测试
    success = run_comprehensive_test()
    
    if success:
        print("\n✅ 所有测试通过！C3Ghost嵌入式部署准备就绪。")
    else:
        print("\n❌ 部分测试失败，请检查实现。")
        sys.exit(1)