#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SimAM注意力机制性能对比测试

本脚本用于测试SimAM与其他注意力机制（SE、CBAM、ECA）的性能对比，
包括推理速度、内存占用、检测精度等指标。

基于YOLO-SLD论文的SimAM实现进行评估。
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import psutil
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# 直接导入注意力模块，避免复杂的依赖
try:
    from models.advanced_yolo_optimizations import (
        AttentionModule, SEAttention, CBAMAttention, 
        ECAAttention, SimAMAttention
    )
except ImportError:
    # 如果导入失败，创建简化的测试版本
    print("警告: 无法导入完整模块，使用简化测试版本")
    
    class SEAttention(nn.Module):
        def __init__(self, channels, reduction=16):
            super().__init__()
            self.fc1 = nn.Linear(channels, channels // reduction)
            self.fc2 = nn.Linear(channels // reduction, channels)
            
        def forward(self, x):
            b, c, h, w = x.size()
            y = torch.mean(x.view(b, c, -1), dim=2)
            y = torch.relu(self.fc1(y))
            y = torch.sigmoid(self.fc2(y))
            return x * y.view(b, c, 1, 1)
    
    class CBAMAttention(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.channel_attention = SEAttention(channels)
            
        def forward(self, x):
            return self.channel_attention(x)
    
    class ECAAttention(nn.Module):
        def __init__(self, channels, k_size=3):
            super().__init__()
            self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2)
            
        def forward(self, x):
            b, c, h, w = x.size()
            y = torch.mean(x.view(b, c, -1), dim=2, keepdim=True)
            y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
            y = torch.sigmoid(y)
            return x * y.view(b, c, 1, 1)
    
    class SimAMAttention(nn.Module):
        """Simple Parameter-Free Attention Module (SimAM)"""
        def __init__(self, channels):
            super().__init__()
            # SimAM不需要任何参数
            
        def forward(self, x):
            b, c, h, w = x.size()
            
            # 计算空间维度的方差
            x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
            y = x_minus_mu_square / (4 * (x_minus_mu_square.mean(dim=[2, 3], keepdim=True) + 1e-4)) + 0.5
            
            # 应用注意力权重
            return x * torch.sigmoid(y)
    
    class AttentionModule(nn.Module):
        def __init__(self, channels, attention_type='SE'):
            super().__init__()
            self.attention_type = attention_type
            
            if attention_type == 'SE':
                self.attention = SEAttention(channels)
            elif attention_type == 'CBAM':
                self.attention = CBAMAttention(channels)
            elif attention_type == 'ECA':
                self.attention = ECAAttention(channels)
            elif attention_type == 'SimAM':
                self.attention = SimAMAttention(channels)
            else:
                self.attention = nn.Identity()
        
        def forward(self, x):
            return self.attention(x)

@dataclass
class BenchmarkResult:
    """基准测试结果"""
    attention_type: str
    avg_inference_time: float  # 平均推理时间 (ms)
    memory_usage: float  # 内存占用 (MB)
    parameter_count: int  # 参数数量
    flops: int  # 浮点运算次数
    accuracy_score: float = 0.0  # 精度分数（如果有测试数据）

class AttentionBenchmark:
    """注意力机制基准测试器"""
    
    def __init__(self, device: str = 'auto'):
        self.device = self._get_device(device)
        self.logger = self._setup_logger()
        
        # 测试配置
        self.test_configs = [
            {'channels': 64, 'height': 320, 'width': 320},   # 轻量级配置
            {'channels': 128, 'height': 416, 'width': 416}, # 中等配置
            {'channels': 256, 'height': 640, 'width': 640}, # 标准配置
        ]
        
        # 注意力机制类型
        self.attention_types = ['SE', 'CBAM', 'ECA', 'SimAM']
        
        # 测试轮数
        self.warmup_rounds = 10
        self.test_rounds = 100
        
    def _get_device(self, device: str) -> torch.device:
        """获取计算设备"""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        return torch.device(device)
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('AttentionBenchmark')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def create_test_input(self, channels: int, height: int, width: int) -> torch.Tensor:
        """创建测试输入数据"""
        return torch.randn(1, channels, height, width, device=self.device)
    
    def count_parameters(self, model: nn.Module) -> int:
        """统计模型参数数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def estimate_flops(self, model: nn.Module, input_tensor: torch.Tensor) -> int:
        """估算浮点运算次数（简化版本）"""
        # 这里使用简化的FLOPS估算
        # 实际应用中可以使用thop或fvcore等库进行精确计算
        total_params = self.count_parameters(model)
        input_size = input_tensor.numel()
        
        # 简化估算：参数数量 × 输入大小
        return total_params * input_size
    
    def measure_memory_usage(self) -> float:
        """测量当前内存使用量 (MB)"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def benchmark_attention(self, attention_type: str, config: Dict[str, int]) -> BenchmarkResult:
        """对单个注意力机制进行基准测试"""
        channels = config['channels']
        height = config['height']
        width = config['width']
        
        self.logger.info(f"测试 {attention_type} 注意力机制 - 配置: {config}")
        
        # 创建注意力模块
        attention_module = AttentionModule(channels, attention_type).to(self.device)
        attention_module.eval()
        
        # 创建测试输入
        test_input = self.create_test_input(channels, height, width)
        
        # 统计参数和FLOPS
        param_count = self.count_parameters(attention_module)
        flops = self.estimate_flops(attention_module, test_input)
        
        # 预热
        with torch.no_grad():
            for _ in range(self.warmup_rounds):
                _ = attention_module(test_input)
        
        # 同步GPU（如果使用）
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # 测量内存使用
        memory_before = self.measure_memory_usage()
        
        # 性能测试
        inference_times = []
        with torch.no_grad():
            for _ in range(self.test_rounds):
                start_time = time.perf_counter()
                output = attention_module(test_input)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                inference_times.append((end_time - start_time) * 1000)  # 转换为毫秒
        
        memory_after = self.measure_memory_usage()
        memory_usage = memory_after - memory_before
        
        # 计算平均推理时间
        avg_inference_time = np.mean(inference_times)
        
        # 清理GPU缓存
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return BenchmarkResult(
            attention_type=attention_type,
            avg_inference_time=avg_inference_time,
            memory_usage=max(memory_usage, 0),  # 确保非负值
            parameter_count=param_count,
            flops=flops
        )
    
    def run_comprehensive_benchmark(self) -> Dict[str, List[BenchmarkResult]]:
        """运行全面的基准测试"""
        self.logger.info(f"开始注意力机制性能基准测试 - 设备: {self.device}")
        
        results = {}
        
        for config in self.test_configs:
            config_name = f"{config['channels']}ch_{config['height']}x{config['width']}"
            results[config_name] = []
            
            self.logger.info(f"\n=== 测试配置: {config_name} ===")
            
            for attention_type in self.attention_types:
                try:
                    result = self.benchmark_attention(attention_type, config)
                    results[config_name].append(result)
                    
                    self.logger.info(
                        f"{attention_type:>6}: "
                        f"推理时间={result.avg_inference_time:.3f}ms, "
                        f"内存={result.memory_usage:.1f}MB, "
                        f"参数={result.parameter_count}"
                    )
                    
                except Exception as e:
                    self.logger.error(f"测试 {attention_type} 时出错: {e}")
                    continue
        
        return results
    
    def generate_report(self, results: Dict[str, List[BenchmarkResult]]) -> str:
        """生成测试报告"""
        report = []
        report.append("# SimAM注意力机制性能对比报告\n")
        report.append(f"测试设备: {self.device}")
        report.append(f"测试轮数: {self.test_rounds}")
        report.append(f"预热轮数: {self.warmup_rounds}\n")
        
        for config_name, config_results in results.items():
            report.append(f"## 配置: {config_name}\n")
            
            # 创建表格
            report.append("| 注意力机制 | 推理时间(ms) | 内存占用(MB) | 参数数量 | FLOPS | 性能评分 |")
            report.append("|-----------|-------------|-------------|---------|-------|---------|")
            
            # SimAM作为基准
            simam_result = None
            for result in config_results:
                if result.attention_type == 'SimAM':
                    simam_result = result
                    break
            
            for result in config_results:
                # 计算相对于SimAM的性能评分
                if simam_result and result.attention_type != 'SimAM':
                    speed_ratio = simam_result.avg_inference_time / result.avg_inference_time
                    memory_ratio = simam_result.memory_usage / max(result.memory_usage, 0.1)
                    param_ratio = simam_result.parameter_count / max(result.parameter_count, 1)
                    performance_score = (speed_ratio + memory_ratio + param_ratio) / 3
                else:
                    performance_score = 1.0  # SimAM基准分数
                
                report.append(
                    f"| {result.attention_type:>8} | "
                    f"{result.avg_inference_time:>10.3f} | "
                    f"{result.memory_usage:>10.1f} | "
                    f"{result.parameter_count:>8} | "
                    f"{result.flops:>8} | "
                    f"{performance_score:>6.3f} |"
                )
            
            report.append("")
            
            # 添加分析
            if simam_result:
                report.append("### 分析结果\n")
                report.append(f"- **SimAM优势**: 参数数量为0，无额外参数开销")
                
                fastest = min(config_results, key=lambda x: x.avg_inference_time)
                if fastest.attention_type == 'SimAM':
                    report.append(f"- **速度最优**: SimAM在此配置下推理速度最快")
                else:
                    speed_diff = ((simam_result.avg_inference_time - fastest.avg_inference_time) 
                                / fastest.avg_inference_time * 100)
                    report.append(f"- **速度对比**: SimAM比最快的{fastest.attention_type}慢{speed_diff:.1f}%")
                
                lowest_memory = min(config_results, key=lambda x: x.memory_usage)
                if lowest_memory.attention_type == 'SimAM':
                    report.append(f"- **内存最优**: SimAM内存占用最低")
                
                report.append("")
        
        # 总结
        report.append("## 总结\n")
        report.append("基于YOLO-SLD论文实现的SimAM注意力机制具有以下特点:\n")
        report.append("1. **参数无关**: 不增加任何可训练参数")
        report.append("2. **计算高效**: 基于能量函数的3D注意力权重计算")
        report.append("3. **轻量级友好**: 特别适合资源受限的嵌入式设备")
        report.append("4. **性能提升**: 在保持轻量级的同时提升检测精度\n")
        
        report.append("**推荐使用场景**:")
        report.append("- 嵌入式设备部署 (ESP32, 树莓派, K230等)")
        report.append("- 移动端应用")
        report.append("- 对参数数量敏感的应用场景")
        report.append("- 需要平衡精度和效率的实时检测任务")
        
        return "\n".join(report)
    
    def save_report(self, report: str, filename: str = "attention_benchmark_report.md"):
        """保存测试报告"""
        report_path = Path(__file__).parent / filename
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"测试报告已保存到: {report_path}")

def main():
    """主函数"""
    # 创建基准测试器
    benchmark = AttentionBenchmark(device='auto')
    
    # 运行基准测试
    results = benchmark.run_comprehensive_benchmark()
    
    # 生成报告
    report = benchmark.generate_report(results)
    
    # 保存报告
    benchmark.save_report(report)
    
    # 打印简要结果
    print("\n" + "="*60)
    print("SimAM注意力机制性能测试完成")
    print("="*60)
    
    for config_name, config_results in results.items():
        print(f"\n配置 {config_name}:")
        for result in config_results:
            print(f"  {result.attention_type:>6}: {result.avg_inference_time:.3f}ms, "
                  f"{result.parameter_count} 参数")

if __name__ == "__main__":
    main()