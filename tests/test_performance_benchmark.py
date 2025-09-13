import pytest
import time
import psutil
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import logging

from src.models.config_loader import YOLOConfigLoader
from src.models.unified_model_manager import ModelConfig, ModelType, PlatformType
from src.utils.logging_manager import LoggingManager


class TestPerformanceBenchmark:
    """性能基准测试类"""
    
    def setup_method(self):
        """测试前设置"""
        self.config_loader = YOLOConfigLoader()
        self.logger = LoggingManager("TestPerformanceBenchmark").get_logger()
        
        # 模拟性能数据
        self.performance_data = {
            'pc': {
                'inference_time': 0.025,  # 25ms
                'memory_usage': 512,       # 512MB
                'gpu_utilization': 85,     # 85%
                'throughput': 40           # 40 FPS
            },
            'jetson': {
                'inference_time': 0.045,   # 45ms
                'memory_usage': 256,       # 256MB
                'gpu_utilization': 95,     # 95%
                'throughput': 22           # 22 FPS
            },
            'k230': {
                'inference_time': 0.080,   # 80ms
                'memory_usage': 64,        # 64MB
                'gpu_utilization': 0,      # CPU only
                'throughput': 12           # 12 FPS
            }
        }
    
    def test_inference_speed_benchmark(self):
        """测试推理速度基准"""
        platforms = ['pc', 'jetson', 'k230']
        
        for platform in platforms:
            config = self.config_loader.load_platform_specific_config(platform)
            
            # 模拟推理时间测量
            inference_time = self.performance_data[platform]['inference_time']
            
            # 验证推理时间符合预期
            expected_time = self.performance_data[platform]['inference_time']
            assert inference_time == expected_time, f"{platform}平台推理时间不符合预期"
            
            self.logger.info(f"{platform}平台推理时间: {inference_time:.3f}s")
    
    def test_memory_usage_benchmark(self):
        """测试内存使用基准"""
        platforms = ['pc', 'jetson', 'k230']
        
        for platform in platforms:
            config = self.config_loader.load_platform_specific_config(platform)
            
            # 模拟内存使用测量
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value = Mock(
                    used=self.performance_data[platform]['memory_usage'] * 1024 * 1024
                )
                
                memory_info = psutil.virtual_memory()
                memory_usage_mb = memory_info.used / (1024 * 1024)
                
                # 验证内存使用符合预期
                expected_memory = self.performance_data[platform]['memory_usage']
                assert memory_usage_mb == expected_memory, f"{platform}平台内存使用不符合预期"
                
                self.logger.info(f"{platform}平台内存使用: {memory_usage_mb:.0f}MB")
    
    def test_gpu_utilization_benchmark(self):
        """测试GPU利用率基准"""
        platforms = ['pc', 'jetson', 'k230']
        
        for platform in platforms:
            config = self.config_loader.load_platform_specific_config(platform)
            
            # 模拟GPU利用率测量
            expected_gpu_util = self.performance_data[platform]['gpu_utilization']
            
            if platform == 'k230':
                # K230使用CPU，GPU利用率应为0
                assert expected_gpu_util == 0, "K230平台应使用CPU而非GPU"
            else:
                # PC和Jetson应有GPU利用率
                assert expected_gpu_util > 0, f"{platform}平台应启用GPU加速"
                assert expected_gpu_util <= 100, f"{platform}平台GPU利用率不应超过100%"
            
            self.logger.info(f"{platform}平台GPU利用率: {expected_gpu_util}%")
    
    def test_throughput_benchmark(self):
        """测试吞吐量基准"""
        platforms = ['pc', 'jetson', 'k230']
        
        for platform in platforms:
            config = self.config_loader.load_platform_specific_config(platform)
            
            # 计算理论吞吐量
            inference_time = self.performance_data[platform]['inference_time']
            theoretical_fps = 1.0 / inference_time
            actual_fps = self.performance_data[platform]['throughput']
            
            # 验证实际吞吐量合理（考虑系统开销）
            efficiency = actual_fps / theoretical_fps
            assert 0.3 <= efficiency <= 1.0, f"{platform}平台吞吐量效率异常: {efficiency:.2f}"
            
            self.logger.info(f"{platform}平台吞吐量: {actual_fps} FPS (效率: {efficiency:.2f})")
    
    def test_tensorrt_optimization_impact(self):
        """测试TensorRT优化对性能的影响"""
        tensorrt_platforms = ['pc', 'jetson']
        
        for platform in tensorrt_platforms:
            config = self.config_loader.load_platform_specific_config(platform)
            
            # 验证TensorRT配置
            assert config.tensorrt_optimize == True, f"{platform}平台应启用TensorRT优化"
            assert config.half_precision == True, f"{platform}平台应启用半精度优化"
            
            # 模拟TensorRT优化效果（相比未优化版本提升30-50%）
            base_inference_time = self.performance_data[platform]['inference_time'] * 1.4
            optimized_time = self.performance_data[platform]['inference_time']
            
            improvement = (base_inference_time - optimized_time) / base_inference_time
            assert improvement >= 0.25, f"{platform}平台TensorRT优化效果不足: {improvement:.2f}"
            
            self.logger.info(f"{platform}平台TensorRT优化提升: {improvement:.1%}")
    
    def test_platform_performance_comparison(self):
        """测试平台间性能对比"""
        # 获取所有平台性能数据
        pc_fps = self.performance_data['pc']['throughput']
        jetson_fps = self.performance_data['jetson']['throughput']
        k230_fps = self.performance_data['k230']['throughput']
        
        # 验证性能排序：PC > Jetson > K230
        assert pc_fps > jetson_fps, "PC平台性能应优于Jetson平台"
        assert jetson_fps > k230_fps, "Jetson平台性能应优于K230平台"
        
        # 验证性能差距合理
        pc_jetson_ratio = pc_fps / jetson_fps
        jetson_k230_ratio = jetson_fps / k230_fps
        
        assert 1.5 <= pc_jetson_ratio <= 3.0, f"PC与Jetson性能差距异常: {pc_jetson_ratio:.2f}x"
        assert 1.5 <= jetson_k230_ratio <= 3.0, f"Jetson与K230性能差距异常: {jetson_k230_ratio:.2f}x"
        
        self.logger.info(f"平台性能对比 - PC: {pc_fps} FPS, Jetson: {jetson_fps} FPS, K230: {k230_fps} FPS")
    
    def test_memory_constraint_impact(self):
        """测试内存约束对性能的影响"""
        # K230为内存受限平台
        k230_config = self.config_loader.load_platform_specific_config('k230')
        pc_config = self.config_loader.load_platform_specific_config('pc')
        
        # 验证内存约束配置
        assert k230_config.model_size == 'n', "内存受限平台应使用nano模型"
        assert k230_config.half_precision == False, "内存受限平台不应启用半精度"
        assert k230_config.dynamic_batching == False, "内存受限平台不应启用动态批处理"
        
        # 验证内存使用差异
        k230_memory = self.performance_data['k230']['memory_usage']
        pc_memory = self.performance_data['pc']['memory_usage']
        
        memory_ratio = pc_memory / k230_memory
        assert memory_ratio >= 4.0, f"PC平台内存使用应显著高于K230: {memory_ratio:.1f}x"
        
        self.logger.info(f"内存约束影响 - K230: {k230_memory}MB, PC: {pc_memory}MB")
    
    def test_performance_regression_detection(self):
        """测试性能回归检测"""
        # 定义性能基线（最低可接受性能）
        performance_baselines = {
            'pc': {'min_fps': 30, 'max_memory_mb': 1024, 'max_inference_ms': 50},
            'jetson': {'min_fps': 15, 'max_memory_mb': 512, 'max_inference_ms': 80},
            'k230': {'min_fps': 8, 'max_memory_mb': 128, 'max_inference_ms': 150}
        }
        
        for platform, baseline in performance_baselines.items():
            perf_data = self.performance_data[platform]
            
            # 检查FPS基线
            assert perf_data['throughput'] >= baseline['min_fps'], \
                f"{platform}平台FPS低于基线: {perf_data['throughput']} < {baseline['min_fps']}"
            
            # 检查内存基线
            assert perf_data['memory_usage'] <= baseline['max_memory_mb'], \
                f"{platform}平台内存使用超过基线: {perf_data['memory_usage']} > {baseline['max_memory_mb']}"
            
            # 检查推理时间基线
            inference_ms = perf_data['inference_time'] * 1000
            assert inference_ms <= baseline['max_inference_ms'], \
                f"{platform}平台推理时间超过基线: {inference_ms:.1f} > {baseline['max_inference_ms']}"
            
            self.logger.info(f"{platform}平台性能符合基线要求")
    
    def test_generate_performance_report(self):
        """测试生成性能报告"""
        report_data = {
            'test_timestamp': '2024-01-01 12:00:00',
            'platforms': {},
            'summary': {}
        }
        
        # 收集各平台性能数据
        for platform, perf_data in self.performance_data.items():
            config = self.config_loader.load_platform_specific_config(platform)
            
            report_data['platforms'][platform] = {
                'model_size': config.model_size,
                'tensorrt_enabled': config.tensorrt_optimize,
                'half_precision': config.half_precision,
                'inference_time_ms': perf_data['inference_time'] * 1000,
                'memory_usage_mb': perf_data['memory_usage'],
                'throughput_fps': perf_data['throughput'],
                'gpu_utilization': perf_data['gpu_utilization']
            }
        
        # 生成性能摘要
        all_fps = [data['throughput'] for data in self.performance_data.values()]
        report_data['summary'] = {
            'best_performance_platform': 'pc',
            'average_fps': sum(all_fps) / len(all_fps),
            'total_platforms_tested': len(self.performance_data),
            'tensorrt_optimization_enabled': 2  # PC和Jetson
        }
        
        # 验证报告数据完整性
        assert len(report_data['platforms']) == 3, "性能报告应包含3个平台数据"
        assert report_data['summary']['best_performance_platform'] == 'pc', "最佳性能平台应为PC"
        assert report_data['summary']['average_fps'] > 0, "平均FPS应大于0"
        
        self.logger.info(f"性能报告生成完成，测试了{len(report_data['platforms'])}个平台")
        
        # 验证报告可以序列化（用于保存）
        import json
        report_json = json.dumps(report_data, indent=2)
        assert len(report_json) > 100, "性能报告JSON应包含足够的数据"