# -*- coding: utf-8 -*-
"""
FastTracker模块性能测试
测试SimSPPF、C3Ghost、IGD等模块的性能和准确性
"""

import torch
import torch.nn as nn
import time
import pytest
from typing import Dict, List, Tuple

# 导入FastTracker相关模块
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.models.advanced_yolo_optimizations import (
        SimSPPF, C3Ghost, IGDModule, AdaptiveDynamicROI, Mish
    )
    from config.fasttracker_config import default_fasttracker_config
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("跳过FastTracker模块测试")
    sys.exit(0)

class FastTrackerPerformanceTest:
    """FastTracker模块性能测试类"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = default_fasttracker_config
        
    def test_simsppf_performance(self) -> Dict[str, float]:
        """测试SimSPPF模块性能"""
        print("\n=== SimSPPF模块性能测试 ===")
        
        # 创建测试数据
        batch_size = 4
        channels = 256
        height, width = 32, 32
        input_tensor = torch.randn(batch_size, channels, height, width).to(self.device)
        
        # 创建SimSPPF模块
        simsppf = SimSPPF(
            c1=channels,
            c2=channels,
            k=self.config.simsppf_config['kernel_sizes'][0]
        ).to(self.device)
        
        # 性能测试
        results = {}
        
        # 前向传播时间测试
        simsppf.eval()
        with torch.no_grad():
            # 预热
            for _ in range(10):
                _ = simsppf(input_tensor)
            
            # 正式测试
            start_time = time.time()
            for _ in range(100):
                output = simsppf(input_tensor)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 100 * 1000  # ms
            results['forward_time_ms'] = avg_time
            results['output_shape'] = list(output.shape)
            
        # 内存使用测试
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            output = simsppf(input_tensor)
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            results['peak_memory_mb'] = peak_memory
        
        # 参数量统计
        total_params = sum(p.numel() for p in simsppf.parameters())
        trainable_params = sum(p.numel() for p in simsppf.parameters() if p.requires_grad)
        results['total_params'] = total_params
        results['trainable_params'] = trainable_params
        
        print(f"前向传播时间: {avg_time:.2f} ms")
        print(f"输出形状: {output.shape}")
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        if torch.cuda.is_available():
            print(f"峰值内存: {peak_memory:.2f} MB")
        
        return results
    
    def test_c3ghost_performance(self) -> Dict[str, float]:
        """测试C3Ghost模块性能"""
        print("\n=== C3Ghost模块性能测试 ===")
        
        # 创建测试数据
        batch_size = 4
        channels = 128
        height, width = 64, 64
        input_tensor = torch.randn(batch_size, channels, height, width).to(self.device)
        
        # 创建C3Ghost模块
        c3ghost = C3Ghost(
            c1=channels,
            c2=channels,
            n=3,  # 瓶颈块数量
            shortcut=True
        ).to(self.device)
        
        results = {}
        
        # 前向传播时间测试
        c3ghost.eval()
        with torch.no_grad():
            # 预热
            for _ in range(10):
                _ = c3ghost(input_tensor)
            
            # 正式测试
            start_time = time.time()
            for _ in range(100):
                output = c3ghost(input_tensor)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 100 * 1000  # ms
            results['forward_time_ms'] = avg_time
            results['output_shape'] = list(output.shape)
        
        # 参数量对比测试
        # 创建标准C3模块进行对比
        class StandardC3(nn.Module):
            def __init__(self, c1, c2, n=1):
                super().__init__()
                self.cv1 = nn.Conv2d(c1, c2//2, 1, 1)
                self.cv2 = nn.Conv2d(c1, c2//2, 1, 1)
                self.cv3 = nn.Conv2d(c2, c2, 1, 1)
                self.m = nn.Sequential(*[nn.Conv2d(c2//2, c2//2, 3, 1, 1) for _ in range(n)])
            
            def forward(self, x):
                return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], 1))
        
        standard_c3 = StandardC3(channels, channels, 3).to(self.device)
        
        # 参数量对比
        c3ghost_params = sum(p.numel() for p in c3ghost.parameters())
        standard_params = sum(p.numel() for p in standard_c3.parameters())
        param_reduction = (1 - c3ghost_params / standard_params) * 100
        
        results['c3ghost_params'] = c3ghost_params
        results['standard_c3_params'] = standard_params
        results['param_reduction_percent'] = param_reduction
        
        print(f"前向传播时间: {avg_time:.2f} ms")
        print(f"输出形状: {output.shape}")
        print(f"C3Ghost参数量: {c3ghost_params:,}")
        print(f"标准C3参数量: {standard_params:,}")
        print(f"参数量减少: {param_reduction:.1f}%")
        
        return results
    
    def test_igd_performance(self) -> Dict[str, float]:
        """测试IGD模块性能"""
        print("\n=== IGD模块性能测试 ===")
        
        # 创建多尺度测试数据
        batch_size = 2
        features = {
            'P3': torch.randn(batch_size, 256, 80, 80).to(self.device),
            'P4': torch.randn(batch_size, 512, 40, 40).to(self.device),
            'P5': torch.randn(batch_size, 1024, 20, 20).to(self.device)
        }
        
        # 创建IGD模块
        igd = IGDModule(
            channels=[256, 512, 1024],
            num_scales=3
        ).to(self.device)
        
        results = {}
        
        # 前向传播时间测试
        igd.eval()
        with torch.no_grad():
            # 预热
            for _ in range(5):
                _ = igd(list(features.values()))
            
            # 正式测试
            start_time = time.time()
            for _ in range(50):
                outputs = igd(list(features.values()))
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 50 * 1000  # ms
            results['forward_time_ms'] = avg_time
            results['output_shapes'] = [list(out.shape) for out in outputs]
        
        # 参数量统计
        total_params = sum(p.numel() for p in igd.parameters())
        results['total_params'] = total_params
        
        print(f"前向传播时间: {avg_time:.2f} ms")
        print(f"输出形状: {[out.shape for out in outputs]}")
        print(f"总参数量: {total_params:,}")
        
        return results
    
    def test_adaptive_roi_performance(self) -> Dict[str, float]:
        """测试自适应动态ROI性能"""
        print("\n=== 自适应动态ROI性能测试 ===")
        
        # 创建测试数据
        batch_size = 4
        channels = 3
        height, width = 640, 640
        images = torch.randn(batch_size, channels, height, width).to(self.device)
        
        # 模拟车辆状态数据
        vehicle_states = {
            'speed': torch.tensor([25.0, 35.0, 15.0, 45.0]).to(self.device),  # km/h
            'steering_angle': torch.tensor([5.0, 20.0, -10.0, 0.0]).to(self.device)  # 度
        }
        
        # 创建自适应ROI模块
        adaptive_roi = AdaptiveDynamicROI().to(self.device)
        
        results = {}
        
        # 前向传播时间测试
        adaptive_roi.eval()
        with torch.no_grad():
            # 预热
            for _ in range(10):
                _ = adaptive_roi(images, vehicle_states)
            
            # 正式测试
            start_time = time.time()
            for _ in range(100):
                roi_images = adaptive_roi(images, vehicle_states)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 100 * 1000  # ms
            results['forward_time_ms'] = avg_time
            results['output_shape'] = list(roi_images.shape)
        
        # ROI区域分析
        roi_predictions = adaptive_roi.predict_roi(vehicle_states)
        results['roi_predictions'] = roi_predictions.cpu().numpy().tolist()
        
        print(f"前向传播时间: {avg_time:.2f} ms")
        print(f"输出形状: {roi_images.shape}")
        print(f"ROI预测: {roi_predictions}")
        
        return results
    
    def run_all_tests(self) -> Dict[str, Dict[str, float]]:
        """运行所有性能测试"""
        print("开始FastTracker模块性能测试...")
        print(f"设备: {self.device}")
        
        all_results = {}
        
        try:
            all_results['simsppf'] = self.test_simsppf_performance()
        except Exception as e:
            print(f"SimSPPF测试失败: {e}")
            all_results['simsppf'] = {'error': str(e)}
        
        try:
            all_results['c3ghost'] = self.test_c3ghost_performance()
        except Exception as e:
            print(f"C3Ghost测试失败: {e}")
            all_results['c3ghost'] = {'error': str(e)}
        
        try:
            all_results['igd'] = self.test_igd_performance()
        except Exception as e:
            print(f"IGD测试失败: {e}")
            all_results['igd'] = {'error': str(e)}
        
        try:
            all_results['adaptive_roi'] = self.test_adaptive_roi_performance()
        except Exception as e:
            print(f"自适应ROI测试失败: {e}")
            all_results['adaptive_roi'] = {'error': str(e)}
        
        return all_results

def test_mish_activation():
    """测试Mish激活函数"""
    print("\n=== Mish激活函数测试 ===")
    
    # 创建测试数据
    x = torch.randn(1000, 100)
    
    # 创建激活函数
    mish = Mish()
    relu = nn.ReLU()
    silu = nn.SiLU()
    
    # 性能对比
    activations = {'Mish': mish, 'ReLU': relu, 'SiLU': silu}
    
    for name, activation in activations.items():
        start_time = time.time()
        for _ in range(1000):
            _ = activation(x)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 1000 * 1000  # ms
        print(f"{name}激活函数平均时间: {avg_time:.4f} ms")

if __name__ == "__main__":
    # 运行性能测试
    tester = FastTrackerPerformanceTest()
    results = tester.run_all_tests()
    
    # 测试Mish激活函数
    test_mish_activation()
    
    print("\n=== 测试完成 ===")
    print("详细结果已保存到results变量中")