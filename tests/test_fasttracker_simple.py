# -*- coding: utf-8 -*-
"""
FastTracker模块简化测试
直接测试核心模块功能
"""

import torch
import torch.nn as nn
import time
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_mish_activation():
    """测试Mish激活函数"""
    print("=== Mish激活函数测试 ===")
    
    class Mish(nn.Module):
        """Mish激活函数: Mish(x) = x * tanh(softplus(x))"""
        def __init__(self, beta=1.0):
            super().__init__()
            self.beta = beta
        
        def forward(self, x):
            return x * torch.tanh(F.softplus(x * self.beta))
    
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
    
    # 功能测试
    test_input = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    mish_output = mish(test_input)
    print(f"\n输入: {test_input}")
    print(f"Mish输出: {mish_output}")
    
    return True

def test_simsppf_concept():
    """测试SimSPPF概念实现"""
    print("\n=== SimSPPF概念测试 ===")
    
    class SimpleSPPF(nn.Module):
        """简化的SPPF实现"""
        def __init__(self, c1, c2, k=5):
            super().__init__()
            c_ = c1 // 2
            self.cv1 = nn.Conv2d(c1, c_, 1, 1)
            self.cv2 = nn.Conv2d(c_ * 4, c2, 1, 1)
            self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        
        def forward(self, x):
            x = self.cv1(x)
            y1 = self.m(x)
            y2 = self.m(y1)
            y3 = self.m(y2)
            return self.cv2(torch.cat([x, y1, y2, y3], 1))
    
    # 创建测试数据
    batch_size = 2
    channels = 256
    height, width = 32, 32
    input_tensor = torch.randn(batch_size, channels, height, width)
    
    # 创建模块
    sppf = SimpleSPPF(c1=channels, c2=channels, k=5)
    
    # 前向传播测试
    with torch.no_grad():
        output = sppf(input_tensor)
        print(f"输入形状: {input_tensor.shape}")
        print(f"输出形状: {output.shape}")
        
        # 性能测试
        start_time = time.time()
        for _ in range(100):
            _ = sppf(input_tensor)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100 * 1000
        print(f"平均前向传播时间: {avg_time:.2f} ms")
    
    # 参数统计
    total_params = sum(p.numel() for p in sppf.parameters())
    print(f"总参数量: {total_params:,}")
    
    return True

def test_ghost_conv_concept():
    """测试Ghost卷积概念"""
    print("\n=== Ghost卷积概念测试 ===")
    
    class SimpleGhostConv(nn.Module):
        """简化的Ghost卷积实现"""
        def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
            super().__init__()
            c_ = c2 // 2
            self.primary_conv = nn.Conv2d(c1, c_, k, s, k//2, groups=g, bias=False)
            self.cheap_operation = nn.Conv2d(c_, c_, 5, 1, 2, groups=c_, bias=False)
            self.bn1 = nn.BatchNorm2d(c_)
            self.bn2 = nn.BatchNorm2d(c_)
            self.act = nn.SiLU() if act else nn.Identity()
        
        def forward(self, x):
            x1 = self.act(self.bn1(self.primary_conv(x)))
            x2 = self.act(self.bn2(self.cheap_operation(x1)))
            return torch.cat([x1, x2], 1)
    
    # 创建测试数据
    input_tensor = torch.randn(2, 128, 64, 64)
    
    # 对比标准卷积和Ghost卷积
    standard_conv = nn.Sequential(
        nn.Conv2d(128, 128, 3, 1, 1, bias=False),
        nn.BatchNorm2d(128),
        nn.SiLU()
    )
    
    ghost_conv = SimpleGhostConv(128, 128, 3, 1)
    
    # 参数量对比
    standard_params = sum(p.numel() for p in standard_conv.parameters())
    ghost_params = sum(p.numel() for p in ghost_conv.parameters())
    
    print(f"标准卷积参数量: {standard_params:,}")
    print(f"Ghost卷积参数量: {ghost_params:,}")
    print(f"参数减少: {(1 - ghost_params/standard_params)*100:.1f}%")
    
    # 性能测试
    with torch.no_grad():
        # 标准卷积
        start_time = time.time()
        for _ in range(100):
            _ = standard_conv(input_tensor)
        standard_time = (time.time() - start_time) / 100 * 1000
        
        # Ghost卷积
        start_time = time.time()
        for _ in range(100):
            _ = ghost_conv(input_tensor)
        ghost_time = (time.time() - start_time) / 100 * 1000
        
        print(f"标准卷积时间: {standard_time:.2f} ms")
        print(f"Ghost卷积时间: {ghost_time:.2f} ms")
        print(f"速度提升: {(standard_time/ghost_time-1)*100:.1f}%")
    
    return True

def test_adaptive_roi_concept():
    """测试自适应ROI概念"""
    print("\n=== 自适应ROI概念测试 ===")
    
    def predict_roi(vehicle_states, base_size=(416, 416)):
        """基于车辆状态预测ROI"""
        speed = vehicle_states['speed']
        angle = vehicle_states['steering_angle']
        
        # 基于速度调整ROI大小
        speed_factor = torch.clamp(speed / 50.0, 0.8, 1.5)  # 速度因子
        
        # 基于转向角调整ROI位置
        angle_factor = torch.clamp(torch.abs(angle) / 30.0, 0.0, 1.0)  # 角度因子
        
        # 计算ROI参数
        roi_width = int(base_size[0] * speed_factor.mean())
        roi_height = int(base_size[1] * speed_factor.mean())
        
        # 限制ROI大小
        roi_width = max(224, min(640, roi_width))
        roi_height = max(224, min(640, roi_height))
        
        return (roi_width, roi_height), speed_factor, angle_factor
    
    # 测试不同车辆状态
    test_cases = [
        {'speed': torch.tensor([25.0]), 'steering_angle': torch.tensor([5.0])},   # 低速直行
        {'speed': torch.tensor([45.0]), 'steering_angle': torch.tensor([20.0])},  # 高速转弯
        {'speed': torch.tensor([15.0]), 'steering_angle': torch.tensor([-10.0])}, # 低速转弯
        {'speed': torch.tensor([60.0]), 'steering_angle': torch.tensor([0.0])},   # 高速直行
    ]
    
    for i, vehicle_state in enumerate(test_cases):
        roi_size, speed_factor, angle_factor = predict_roi(vehicle_state)
        print(f"\n测试案例 {i+1}:")
        print(f"  速度: {vehicle_state['speed'].item():.1f} km/h")
        print(f"  转向角: {vehicle_state['steering_angle'].item():.1f}°")
        print(f"  预测ROI大小: {roi_size}")
        print(f"  速度因子: {speed_factor.item():.2f}")
        print(f"  角度因子: {angle_factor.item():.2f}")
    
    return True

def main():
    """主测试函数"""
    print("FastTracker模块概念验证测试")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 导入torch.nn.functional
    import torch.nn.functional as F
    globals()['F'] = F
    
    try:
        # 运行各项测试
        tests = [
            ("Mish激活函数", test_mish_activation),
            ("SimSPPF概念", test_simsppf_concept),
            ("Ghost卷积概念", test_ghost_conv_concept),
            ("自适应ROI概念", test_adaptive_roi_concept)
        ]
        
        results = {}
        for test_name, test_func in tests:
            try:
                print(f"\n开始测试: {test_name}")
                result = test_func()
                results[test_name] = "通过" if result else "失败"
                print(f"{test_name} 测试完成")
            except Exception as e:
                print(f"{test_name} 测试失败: {e}")
                results[test_name] = f"错误: {e}"
        
        # 输出测试结果
        print("\n" + "=" * 50)
        print("测试结果汇总:")
        for test_name, result in results.items():
            print(f"  {test_name}: {result}")
        
        # 统计
        passed = sum(1 for r in results.values() if r == "通过")
        total = len(results)
        print(f"\n通过率: {passed}/{total} ({passed/total*100:.1f}%)")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)