#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOS 基准测试示例
演示如何使用性能基准测试系统
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.benchmark import (
    BenchmarkRunner,
    PerformanceBenchmark,
    BenchmarkConfig,
    BenchmarkType,
    TestScenario
)
from src.detection.feature_aggregation import PlatformType

def run_basic_benchmark():
    """运行基础基准测试"""
    print("🚀 开始基础性能基准测试")
    print("=" * 50)
    
    # 使用运行器进行快速测试
    runner = BenchmarkRunner()
    runner.run_quick_test('desktop')
    
    print("\n✅ 基础测试完成")

def run_custom_benchmark():
    """运行自定义基准测试"""
    print("\n🔧 开始自定义基准测试")
    print("=" * 50)
    
    # 创建自定义配置
    config = BenchmarkConfig(
        test_types=[
            BenchmarkType.DETECTION_SPEED,
            BenchmarkType.MEMORY_USAGE,
            BenchmarkType.TRACKING_PERFORMANCE
        ],
        test_scenarios=[
            TestScenario.SINGLE_OBJECT,
            TestScenario.MULTI_OBJECT,
            TestScenario.FAST_MOTION
        ],
        target_platforms=[
            PlatformType.DESKTOP
        ],
        test_duration=20,  # 20秒测试
        warmup_duration=3,  # 3秒预热
        output_dir="custom_benchmark_results"
    )
    
    # 创建基准测试器
    benchmark = PerformanceBenchmark(config)
    
    # 运行测试
    results = benchmark.run_benchmark()
    
    # 打印结果
    print(f"\n📊 测试完成，共 {len(results)} 项测试")
    for result in results:
        print(f"  - {result.test_type.value} ({result.scenario.value}): "
              f"{result.avg_fps:.1f} FPS, {result.avg_memory_mb:.1f} MB")
    
    print("\n✅ 自定义测试完成")

def run_tracking_benchmark():
    """运行跟踪性能测试"""
    print("\n🎯 开始跟踪性能基准测试")
    print("=" * 50)
    
    runner = BenchmarkRunner()
    runner.run_tracking_test('desktop')
    
    print("\n✅ 跟踪测试完成")

def run_aggregation_benchmark():
    """运行聚合效果测试"""
    print("\n🔄 开始聚合效果基准测试")
    print("=" * 50)
    
    runner = BenchmarkRunner()
    runner.run_aggregation_test('desktop')
    
    print("\n✅ 聚合测试完成")

def demonstrate_benchmark_features():
    """演示基准测试功能"""
    print("\n🎪 基准测试功能演示")
    print("=" * 50)
    
    print("\n📋 可用的测试类型:")
    for test_type in BenchmarkType:
        print(f"  - {test_type.value}")
    
    print("\n🎬 可用的测试场景:")
    for scenario in TestScenario:
        print(f"  - {scenario.value}")
    
    print("\n💻 支持的平台:")
    for platform in PlatformType:
        print(f"  - {platform.value}")
    
    print("\n🛠️ 基准测试配置选项:")
    print("  - test_duration: 测试持续时间")
    print("  - warmup_duration: 预热时间")
    print("  - sample_interval: 采样间隔")
    print("  - output_dir: 输出目录")
    print("  - save_plots: 保存图表")
    print("  - generate_report: 生成报告")
    
    print("\n📈 输出文件:")
    print("  - detailed_results.json: 详细测试结果")
    print("  - benchmark_report.md: 测试报告")
    print("  - system_info.json: 系统信息")
    print("  - *.png: 性能图表")

def main():
    """主函数"""
    print("🎯 YOLOS 基准测试系统示例")
    print("=" * 60)
    
    try:
        # 演示功能
        demonstrate_benchmark_features()
        
        # 运行基础测试
        run_basic_benchmark()
        
        # 运行自定义测试
        run_custom_benchmark()
        
        # 运行跟踪测试
        run_tracking_benchmark()
        
        # 运行聚合测试
        run_aggregation_benchmark()
        
        print("\n🎉 所有示例测试完成!")
        print("\n💡 提示:")
        print("  - 查看 benchmark_results/ 目录获取详细结果")
        print("  - 使用 benchmark_runner.py 进行命令行测试")
        print("  - 根据需要调整测试配置参数")
        
    except KeyboardInterrupt:
        print("\n⏹️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()