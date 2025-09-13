#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基准测试运行器
提供简化的接口来运行各种性能基准测试
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .performance_benchmark import (
    PerformanceBenchmark, BenchmarkConfig, BenchmarkType, 
    TestScenario, PlatformType
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('benchmark.log')
    ]
)

logger = logging.getLogger(__name__)

class BenchmarkRunner:
    """基准测试运行器"""
    
    def __init__(self):
        self.benchmark = None
    
    def run_quick_test(self, platform: str = 'desktop') -> None:
        """运行快速测试"""
        logger.info("开始快速性能测试")
        
        platform_type = self._parse_platform(platform)
        
        config = BenchmarkConfig(
            test_types=[
                BenchmarkType.DETECTION_SPEED,
                BenchmarkType.MEMORY_USAGE
            ],
            test_scenarios=[
                TestScenario.SINGLE_OBJECT,
                TestScenario.MULTI_OBJECT
            ],
            target_platforms=[platform_type],
            test_duration=15,
            warmup_duration=3
        )
        
        self.benchmark = PerformanceBenchmark(config)
        results = self.benchmark.run_benchmark()
        
        self._print_summary(results)
    
    def run_comprehensive_test(self, platforms: List[str] = None) -> None:
        """运行全面测试"""
        logger.info("开始全面性能测试")
        
        if platforms is None:
            platforms = ['desktop']
        
        platform_types = [self._parse_platform(p) for p in platforms]
        
        config = BenchmarkConfig(
            test_types=list(BenchmarkType),
            test_scenarios=list(TestScenario),
            target_platforms=platform_types,
            test_duration=60,
            warmup_duration=10
        )
        
        self.benchmark = PerformanceBenchmark(config)
        results = self.benchmark.run_benchmark()
        
        self._print_summary(results)
    
    def run_tracking_test(self, platform: str = 'desktop') -> None:
        """运行跟踪性能测试"""
        logger.info("开始跟踪性能测试")
        
        platform_type = self._parse_platform(platform)
        
        config = BenchmarkConfig(
            test_types=[
                BenchmarkType.TRACKING_PERFORMANCE,
                BenchmarkType.DETECTION_SPEED,
                BenchmarkType.MEMORY_USAGE
            ],
            test_scenarios=[
                TestScenario.MULTI_OBJECT,
                TestScenario.FAST_MOTION,
                TestScenario.OCCLUSION
            ],
            target_platforms=[platform_type],
            test_duration=45,
            warmup_duration=5
        )
        
        self.benchmark = PerformanceBenchmark(config)
        results = self.benchmark.run_benchmark()
        
        self._print_summary(results)
    
    def run_aggregation_test(self, platform: str = 'desktop') -> None:
        """运行聚合效果测试"""
        logger.info("开始聚合效果测试")
        
        platform_type = self._parse_platform(platform)
        
        config = BenchmarkConfig(
            test_types=[
                BenchmarkType.AGGREGATION_EFFECTIVENESS,
                BenchmarkType.DETECTION_ACCURACY,
                BenchmarkType.DETECTION_SPEED
            ],
            test_scenarios=[
                TestScenario.CROWDED_SCENE,
                TestScenario.LIGHTING_CHANGE,
                TestScenario.SCALE_VARIATION
            ],
            target_platforms=[platform_type],
            test_duration=30,
            warmup_duration=5
        )
        
        self.benchmark = PerformanceBenchmark(config)
        results = self.benchmark.run_benchmark()
        
        self._print_summary(results)
    
    def run_platform_comparison(self) -> None:
        """运行平台对比测试"""
        logger.info("开始平台对比测试")
        
        config = BenchmarkConfig(
            test_types=[
                BenchmarkType.DETECTION_SPEED,
                BenchmarkType.MEMORY_USAGE,
                BenchmarkType.CPU_USAGE
            ],
            test_scenarios=[
                TestScenario.SINGLE_OBJECT,
                TestScenario.MULTI_OBJECT
            ],
            target_platforms=[
                PlatformType.DESKTOP,
                PlatformType.RASPBERRY_PI,
                PlatformType.ESP32
            ],
            test_duration=30,
            warmup_duration=5
        )
        
        self.benchmark = PerformanceBenchmark(config)
        results = self.benchmark.run_benchmark()
        
        self._print_summary(results)
    
    def run_custom_test(self, config_file: str) -> None:
        """运行自定义测试"""
        logger.info(f"从配置文件运行测试: {config_file}")
        
        # 这里可以实现从JSON/YAML文件加载配置的逻辑
        # 暂时使用默认配置
        config = BenchmarkConfig()
        
        self.benchmark = PerformanceBenchmark(config)
        results = self.benchmark.run_benchmark()
        
        self._print_summary(results)
    
    def _parse_platform(self, platform: str) -> PlatformType:
        """解析平台类型"""
        platform_map = {
            'desktop': PlatformType.DESKTOP,
            'pc': PlatformType.DESKTOP,
            'raspberry_pi': PlatformType.RASPBERRY_PI,
            'rpi': PlatformType.RASPBERRY_PI,
            'jetson': PlatformType.JETSON,
            'esp32': PlatformType.ESP32,
            'k230': PlatformType.K230
        }
        
        platform_lower = platform.lower()
        if platform_lower not in platform_map:
            logger.warning(f"未知平台类型: {platform}，使用默认平台 desktop")
            return PlatformType.DESKTOP
        
        return platform_map[platform_lower]
    
    def _print_summary(self, results) -> None:
        """打印测试结果摘要"""
        if not results:
            logger.warning("没有测试结果")
            return
        
        print("\n" + "="*60)
        print("基准测试结果摘要")
        print("="*60)
        
        # 按平台分组显示
        platform_results = {}
        for result in results:
            platform = result.platform.value
            if platform not in platform_results:
                platform_results[platform] = []
            platform_results[platform].append(result)
        
        for platform, platform_results_list in platform_results.items():
            print(f"\n📱 平台: {platform.upper()}")
            print("-" * 40)
            
            for result in platform_results_list:
                status_icon = "✅" if result.error_count == 0 else "⚠️"
                print(f"{status_icon} {result.test_type.value} - {result.scenario.value}")
                
                if result.avg_fps > 0:
                    print(f"   📊 FPS: {result.avg_fps:.1f} (min: {result.min_fps:.1f}, max: {result.max_fps:.1f})")
                
                if result.avg_latency > 0:
                    print(f"   ⏱️  延迟: {result.avg_latency:.1f}ms")
                
                if result.avg_memory_mb > 0:
                    print(f"   💾 内存: {result.avg_memory_mb:.1f}MB (峰值: {result.peak_memory_mb:.1f}MB)")
                
                if result.avg_cpu_percent > 0:
                    print(f"   🔥 CPU: {result.avg_cpu_percent:.1f}% (峰值: {result.peak_cpu_percent:.1f}%)")
                
                if result.detection_count > 0:
                    print(f"   🎯 检测数: {result.detection_count}")
                
                if result.error_count > 0:
                    print(f"   ❌ 错误数: {result.error_count}")
                
                print()
        
        # 整体统计
        total_tests = len(results)
        successful_tests = len([r for r in results if r.error_count == 0])
        avg_fps_all = sum(r.avg_fps for r in results if r.avg_fps > 0) / max(1, len([r for r in results if r.avg_fps > 0]))
        
        print("\n📈 整体统计")
        print("-" * 40)
        print(f"总测试数: {total_tests}")
        print(f"成功测试: {successful_tests} ({successful_tests/total_tests*100:.1f}%)")
        print(f"平均FPS: {avg_fps_all:.1f}")
        
        if self.benchmark:
            output_dir = self.benchmark.output_dir
            print(f"\n📁 详细报告保存在: {output_dir}")
            print(f"   - 详细结果: {output_dir}/detailed_results.json")
            print(f"   - 测试报告: {output_dir}/benchmark_report.md")
            print(f"   - 系统信息: {output_dir}/system_info.json")
        
        print("\n" + "="*60)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOS 性能基准测试运行器')
    parser.add_argument('test_type', choices=[
        'quick', 'comprehensive', 'tracking', 'aggregation', 'platform', 'custom'
    ], help='测试类型')
    parser.add_argument('--platform', '-p', default='desktop', 
                       help='目标平台 (desktop, raspberry_pi, jetson, esp32, k230)')
    parser.add_argument('--platforms', nargs='+', 
                       help='多个平台 (用于comprehensive和platform测试)')
    parser.add_argument('--config', '-c', help='自定义配置文件路径')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    runner = BenchmarkRunner()
    
    try:
        if args.test_type == 'quick':
            runner.run_quick_test(args.platform)
        elif args.test_type == 'comprehensive':
            platforms = args.platforms or [args.platform]
            runner.run_comprehensive_test(platforms)
        elif args.test_type == 'tracking':
            runner.run_tracking_test(args.platform)
        elif args.test_type == 'aggregation':
            runner.run_aggregation_test(args.platform)
        elif args.test_type == 'platform':
            runner.run_platform_comparison()
        elif args.test_type == 'custom':
            if not args.config:
                print("错误: 自定义测试需要指定配置文件 (--config)")
                sys.exit(1)
            runner.run_custom_test(args.config)
        
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
    except Exception as e:
        logger.error(f"测试执行失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()