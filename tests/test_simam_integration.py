"""SimAM注意力机制集成测试

测试SimAM注意力机制的功能性、性能和集成效果

Author: YOLOS Team
Date: 2024-01-15
Version: 1.0.0
"""

import unittest
import torch
import torch.nn as nn
import time
import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 尝试不同的导入路径
try:
    from src.models.simam_integration import (
        SimAMAttention, 
        SimAMIntegrator, 
        SimAMWrapper,
        integrate_simam_to_model
    )
except ImportError:
    try:
        # 如果上面失败，尝试直接导入
        import os
        sys.path.append(os.path.join(project_root, 'src', 'models'))
        from simam_integration import (
            SimAMAttention, 
            SimAMIntegrator, 
            SimAMWrapper,
            integrate_simam_to_model
        )
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保项目结构正确")
        print(f"当前工作目录: {os.getcwd()}")
        print(f"项目根目录: {project_root}")
        sys.exit(1)


class TestSimAMAttention(unittest.TestCase):
    """SimAM注意力机制基础测试"""
    
    def setUp(self):
        """测试前准备"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.simam = SimAMAttention().to(self.device)
        
        # 测试用的输入张量
        self.test_inputs = {
            'small': torch.randn(1, 64, 16, 16).to(self.device),
            'medium': torch.randn(2, 256, 32, 32).to(self.device),
            'large': torch.randn(4, 512, 64, 64).to(self.device)
        }
    
    def test_forward_pass(self):
        """测试前向传播"""
        for size_name, input_tensor in self.test_inputs.items():
            with self.subTest(size=size_name):
                output = self.simam(input_tensor)
                
                # 检查输出形状
                self.assertEqual(output.shape, input_tensor.shape, 
                               f"输出形状不匹配: {size_name}")
                
                # 检查输出不是NaN或Inf
                self.assertFalse(torch.isnan(output).any(), 
                               f"输出包含NaN: {size_name}")
                self.assertFalse(torch.isinf(output).any(), 
                               f"输出包含Inf: {size_name}")
    
    def test_parameter_count(self):
        """测试参数数量（SimAM应该是无参数的）"""
        param_count = sum(p.numel() for p in self.simam.parameters())
        self.assertEqual(param_count, 0, "SimAM应该是无参数的注意力机制")
    
    def test_gradient_flow(self):
        """测试梯度流"""
        input_tensor = self.test_inputs['medium'].requires_grad_(True)
        output = self.simam(input_tensor)
        
        # 计算损失并反向传播
        loss = output.sum()
        loss.backward()
        
        # 检查输入张量是否有梯度
        self.assertIsNotNone(input_tensor.grad, "输入张量应该有梯度")
        self.assertFalse(torch.isnan(input_tensor.grad).any(), "梯度不应该包含NaN")
    
    def test_different_lambda_params(self):
        """测试不同的lambda参数"""
        lambda_values = [1e-6, 1e-4, 1e-2, 1e-1]
        input_tensor = self.test_inputs['medium']
        
        outputs = []
        for lambda_val in lambda_values:
            simam = SimAMAttention(lambda_param=lambda_val).to(self.device)
            output = simam(input_tensor)
            outputs.append(output)
        
        # 检查不同lambda值产生不同的输出
        for i in range(len(outputs) - 1):
            self.assertFalse(torch.allclose(outputs[i], outputs[i+1], atol=1e-6),
                           f"lambda={lambda_values[i]}和lambda={lambda_values[i+1]}产生了相同的输出")


class TestSimAMIntegrator(unittest.TestCase):
    """SimAM集成器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.integrator = SimAMIntegrator()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_config_loading(self):
        """测试配置加载"""
        # 测试默认配置
        self.assertIsInstance(self.integrator.config, dict)
        self.assertIn('simam_attention', self.integrator.config)
        
        # 检查关键配置项
        simam_config = self.integrator.config['simam_attention']
        self.assertIn('enabled', simam_config)
        self.assertIn('lambda_param', simam_config)
        self.assertIn('eps', simam_config)
        self.assertIn('integration_points', simam_config)
    
    def test_create_simam_module(self):
        """测试创建SimAM模块"""
        module_name = "test_simam"
        simam_module = self.integrator.create_simam_module(module_name)
        
        # 检查模块类型
        self.assertIsInstance(simam_module, SimAMAttention)
        
        # 检查模块是否被记录
        self.assertIn(module_name, self.integrator.simam_modules)
        self.assertEqual(self.integrator.simam_modules[module_name], simam_module)
    
    def test_performance_stats(self):
        """测试性能统计"""
        # 创建几个模块
        self.integrator.create_simam_module("test1")
        self.integrator.create_simam_module("test2")
        
        stats = self.integrator.get_performance_stats()
        
        # 检查统计信息
        self.assertEqual(stats['total_simam_modules'], 2)
        self.assertEqual(len(stats['module_names']), 2)
        self.assertIn('test1', stats['module_names'])
        self.assertIn('test2', stats['module_names'])
        self.assertIn('config_summary', stats)


class TestSimAMWrapper(unittest.TestCase):
    """SimAM包装器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建原始模块（简单的卷积层）
        self.original_module = nn.Conv2d(256, 256, 3, padding=1).to(self.device)
        
        # 创建SimAM模块
        self.simam_module = SimAMAttention().to(self.device)
        
        # 创建包装器
        self.wrapper = SimAMWrapper(self.original_module, self.simam_module).to(self.device)
        
        # 测试输入
        self.test_input = torch.randn(2, 256, 32, 32).to(self.device)
    
    def test_wrapper_forward(self):
        """测试包装器前向传播"""
        output = self.wrapper(self.test_input)
        
        # 检查输出形状
        self.assertEqual(output.shape, self.test_input.shape)
        
        # 检查输出不是NaN或Inf
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_wrapper_vs_separate(self):
        """测试包装器输出与分别调用的一致性"""
        # 包装器输出
        wrapper_output = self.wrapper(self.test_input)
        
        # 分别调用
        conv_output = self.original_module(self.test_input)
        separate_output = self.simam_module(conv_output)
        
        # 检查一致性
        self.assertTrue(torch.allclose(wrapper_output, separate_output, atol=1e-6),
                       "包装器输出与分别调用的结果不一致")


class TestPerformanceBenchmark(unittest.TestCase):
    """性能基准测试"""
    
    def setUp(self):
        """测试前准备"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.simam = SimAMAttention().to(self.device)
        
        # 不同尺寸的测试输入
        self.benchmark_inputs = {
            'small': torch.randn(1, 256, 32, 32).to(self.device),
            'medium': torch.randn(4, 512, 64, 64).to(self.device),
            'large': torch.randn(8, 1024, 128, 128).to(self.device)
        }
    
    def test_inference_speed(self):
        """测试推理速度"""
        results = {}
        
        for size_name, input_tensor in self.benchmark_inputs.items():
            # 预热
            for _ in range(10):
                _ = self.simam(input_tensor)
            
            # 同步GPU（如果使用）
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            # 计时
            start_time = time.time()
            num_runs = 100
            
            for _ in range(num_runs):
                _ = self.simam(input_tensor)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            avg_time = (end_time - start_time) / num_runs * 1000  # 转换为毫秒
            
            results[size_name] = avg_time
            print(f"SimAM推理时间 ({size_name}): {avg_time:.3f} ms")
        
        # 检查推理时间是否合理（根据输入尺寸调整阈值）
        time_thresholds = {
            'small': 50,   # 小尺寸: 50ms
            'medium': 200, # 中等尺寸: 200ms
            'large': 800   # 大尺寸: 800ms
        }
        
        for size_name, avg_time in results.items():
            threshold = time_thresholds.get(size_name, 100)
            self.assertLess(avg_time, threshold, f"{size_name}尺寸的推理时间过长: {avg_time:.3f} ms (阈值: {threshold} ms)")
    
    def test_memory_usage(self):
        """测试内存使用"""
        if self.device.type != 'cuda':
            self.skipTest("内存测试需要CUDA设备")
        
        for size_name, input_tensor in self.benchmark_inputs.items():
            # 清空缓存
            torch.cuda.empty_cache()
            
            # 记录初始内存
            initial_memory = torch.cuda.memory_allocated()
            
            # 前向传播
            output = self.simam(input_tensor)
            
            # 记录峰值内存
            peak_memory = torch.cuda.max_memory_allocated()
            memory_usage = (peak_memory - initial_memory) / 1024 / 1024  # 转换为MB
            
            print(f"SimAM内存使用 ({size_name}): {memory_usage:.2f} MB")
            
            # 重置内存统计
            torch.cuda.reset_peak_memory_stats()
            
            # 检查内存使用是否合理
            input_size_mb = input_tensor.numel() * 4 / 1024 / 1024  # float32 = 4 bytes
            self.assertLess(memory_usage, input_size_mb * 3, 
                          f"{size_name}尺寸的内存使用过多: {memory_usage:.2f} MB")


class TestIntegrationScenarios(unittest.TestCase):
    """集成场景测试"""
    
    def setUp(self):
        """测试前准备"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_simple_model_integration(self):
        """测试简单模型集成"""
        # 创建简单的测试模型
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
                self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
                self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
                self.relu = nn.ReLU(inplace=True)
            
            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = self.relu(self.conv3(x))
                return x
        
        # 创建模型
        model = SimpleModel().to(self.device)
        original_param_count = sum(p.numel() for p in model.parameters())
        
        # 集成SimAM
        integrated_model = integrate_simam_to_model(model)
        integrated_param_count = sum(p.numel() for p in integrated_model.parameters())
        
        # 检查参数数量（SimAM不应该增加参数）
        self.assertEqual(original_param_count, integrated_param_count,
                        "SimAM集成不应该增加模型参数")
        
        # 测试前向传播
        test_input = torch.randn(2, 3, 224, 224).to(self.device)
        output = integrated_model(test_input)
        
        # 检查输出
        self.assertEqual(output.shape, (2, 256, 224, 224))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_attention_effectiveness(self):
        """测试注意力机制的有效性"""
        # 创建测试输入：一个有明显特征的图像
        batch_size, channels, height, width = 1, 256, 32, 32
        
        # 创建有噪声的输入
        noise_input = torch.randn(batch_size, channels, height, width).to(self.device)
        
        # 在中心区域添加强信号
        center_h, center_w = height // 2, width // 2
        noise_input[:, :, center_h-4:center_h+4, center_w-4:center_w+4] += 5.0
        
        # 应用SimAM
        simam = SimAMAttention().to(self.device)
        attended_output = simam(noise_input)
        
        # 计算中心区域和边缘区域的平均激活
        center_activation = attended_output[:, :, center_h-4:center_h+4, center_w-4:center_w+4].mean()
        edge_activation = attended_output[:, :, :8, :8].mean()  # 左上角区域
        
        # 检查注意力是否突出了中心区域
        self.assertGreater(center_activation.item(), edge_activation.item(),
                          "SimAM应该突出显著特征区域")


def run_comprehensive_test():
    """运行综合测试"""
    print("="*60)
    print("SimAM注意力机制集成测试")
    print("="*60)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_classes = [
        TestSimAMAttention,
        TestSimAMIntegrator,
        TestSimAMWrapper,
        TestPerformanceBenchmark,
        TestIntegrationScenarios
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出总结
    print("\n" + "="*60)
    print(f"测试总结:")
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"成功率: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print("="*60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # 设置随机种子以确保可重现性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行测试
    success = run_comprehensive_test()
    
    if success:
        print("\n✅ 所有测试通过！SimAM集成准备就绪。")
    else:
        print("\n❌ 部分测试失败，请检查实现。")
        sys.exit(1)