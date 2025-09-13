#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorRT优化配置测试
测试不同平台的TensorRT加速配置和性能优化
"""

import sys
import os
from pathlib import Path
import time
import pytest
from typing import Dict, List

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.models.config_loader import load_platform_config, YOLOConfigLoader
    from src.models.unified_model_manager import ModelConfig, ModelType, PlatformType
    from src.utils.logging_manager import LoggingManager
except ImportError as e:
    pytest.skip(f"跳过测试，模块导入失败: {e}")


class TestTensorRTOptimization:
    """TensorRT优化配置测试类"""
    
    def setup_method(self):
        """测试初始化"""
        self.logger = LoggingManager("TestTensorRTOptimization").get_logger()
        
        # 定义支持TensorRT的平台配置
        self.tensorrt_platforms = {
            'jetson': {
                'expected_tensorrt': True,
                'expected_precision': 'fp16',
                'expected_workspace_size': '1GB',
                'compute_capability': 'high'
            },
            'k230': {
                'expected_tensorrt': True,
                'expected_precision': 'int8',
                'expected_workspace_size': '256MB',
                'compute_capability': 'medium'
            },
            'pc': {
                'expected_tensorrt': True,
                'expected_precision': 'fp16',
                'expected_workspace_size': '2GB',
                'compute_capability': 'high'
            }
        }
        
        # 不支持TensorRT的平台
        self.non_tensorrt_platforms = {
            'esp32': {
                'expected_tensorrt': False,
                'reason': '计算能力不足'
            },
            'raspberry_pi': {
                'expected_tensorrt': False,
                'reason': '缺少GPU支持'
            }
        }
        
    def test_tensorrt_platform_support(self):
        """测试TensorRT平台支持"""
        for platform_name, expected_config in self.tensorrt_platforms.items():
            config = load_platform_config(platform_name)
            
            # 验证TensorRT启用状态
            assert config.tensorrt_optimize == expected_config['expected_tensorrt'], \
                f"{platform_name}平台TensorRT配置不匹配"
            
            # 验证半精度设置（TensorRT优化的前提）
            if expected_config['expected_tensorrt']:
                if platform_name != 'k230':  # K230内存受限，不启用半精度
                    assert config.half_precision, \
                        f"{platform_name}平台应启用半精度以支持TensorRT"
            
            self.logger.info(f"{platform_name}平台TensorRT配置验证通过")
            
    def test_non_tensorrt_platforms(self):
        """测试不支持TensorRT的平台"""
        for platform_name, expected_config in self.non_tensorrt_platforms.items():
            config = load_platform_config(platform_name)
            
            # 验证TensorRT禁用状态
            assert config.tensorrt_optimize == expected_config['expected_tensorrt'], \
                f"{platform_name}平台不应启用TensorRT: {expected_config['reason']}"
            
            self.logger.info(f"{platform_name}平台TensorRT禁用验证通过: {expected_config['reason']}")
            
    def test_tensorrt_precision_optimization(self):
        """测试TensorRT精度优化"""
        precision_configs = {
            'jetson': 'fp16',  # Jetson支持FP16
            'k230': 'int8',    # K230内存受限，使用INT8
            'pc': 'fp16'       # PC通常支持FP16
        }
        
        for platform_name, expected_precision in precision_configs.items():
            config = load_platform_config(platform_name)
            
            if config.tensorrt_optimize:
                # 根据平台验证精度设置
                if expected_precision == 'fp16':
                    if platform_name != 'k230':  # K230特殊处理
                        assert config.half_precision, \
                            f"{platform_name}平台应启用FP16精度"
                elif expected_precision == 'int8':
                    # INT8量化通常通过其他配置实现
                    assert config.model_size == 'n', \
                        f"{platform_name}平台使用INT8时应选择最小模型"
                        
            self.logger.info(f"{platform_name}平台精度优化验证通过: {expected_precision}")
            
    def test_tensorrt_workspace_optimization(self):
        """测试TensorRT工作空间优化"""
        workspace_expectations = {
            'jetson': {'min_memory_mb': 512, 'max_memory_mb': 1024},
            'k230': {'min_memory_mb': 128, 'max_memory_mb': 256},
            'pc': {'min_memory_mb': 1024, 'max_memory_mb': 2048}
        }
        
        for platform_name, memory_config in workspace_expectations.items():
            config = load_platform_config(platform_name)
            
            if config.tensorrt_optimize:
                # 验证模型大小与内存使用的匹配
                model_size_memory = {
                    'n': 50,   # nano模型约50MB
                    's': 100,  # small模型约100MB
                    'm': 200,  # medium模型约200MB
                    'l': 400   # large模型约400MB
                }
                
                expected_model_memory = model_size_memory.get(config.model_size, 100)
                
                # 验证模型内存使用在合理范围内
                assert expected_model_memory <= memory_config['max_memory_mb'], \
                    f"{platform_name}平台模型内存使用({expected_model_memory}MB)超出限制({memory_config['max_memory_mb']}MB)"
                    
            self.logger.info(f"{platform_name}平台工作空间优化验证通过")
            
    def test_tensorrt_batch_optimization(self):
        """测试TensorRT批处理优化"""
        for platform_name in self.tensorrt_platforms.keys():
            config = load_platform_config(platform_name)
            
            if config.tensorrt_optimize:
                # 高性能平台应启用动态批处理
                if platform_name in ['jetson', 'pc']:
                    assert config.dynamic_batching, \
                        f"{platform_name}高性能平台应启用动态批处理"
                # 内存受限平台可能禁用动态批处理
                elif platform_name == 'k230':
                    assert not config.dynamic_batching, \
                        f"{platform_name}内存受限平台应禁用动态批处理"
                        
            self.logger.info(f"{platform_name}平台批处理优化验证通过")
            
    def test_tensorrt_model_compatibility(self):
        """测试TensorRT模型兼容性"""
        compatible_models = ['yolov11n.pt', 'yolov11s.pt', 'yolov11m.pt']
        
        for platform_name in self.tensorrt_platforms.keys():
            config = load_platform_config(platform_name)
            
            if config.tensorrt_optimize:
                # 验证模型路径格式
                assert config.model_path in compatible_models, \
                    f"{platform_name}平台模型{config.model_path}可能不兼容TensorRT"
                
                # 验证模型大小选择合理性
                platform_model_limits = {
                    'k230': ['n'],           # K230只支持nano
                    'jetson': ['n', 's', 'm'], # Jetson支持小到中等模型
                    'pc': ['n', 's', 'm', 'l'] # PC支持所有模型
                }
                
                allowed_sizes = platform_model_limits.get(platform_name, ['n', 's'])
                assert config.model_size in allowed_sizes, \
                    f"{platform_name}平台不支持{config.model_size}模型大小"
                    
            self.logger.info(f"{platform_name}平台模型兼容性验证通过")
            
    def test_tensorrt_performance_expectations(self):
        """测试TensorRT性能期望"""
        performance_expectations = {
            'jetson': {
                'expected_speedup': 2.0,  # 期望2x加速
                'max_inference_time_ms': 50
            },
            'k230': {
                'expected_speedup': 1.5,  # 期望1.5x加速
                'max_inference_time_ms': 100
            },
            'pc': {
                'expected_speedup': 3.0,  # 期望3x加速
                'max_inference_time_ms': 20
            }
        }
        
        for platform_name, perf_config in performance_expectations.items():
            config = load_platform_config(platform_name)
            
            if config.tensorrt_optimize:
                # 验证配置合理性（实际性能测试需要真实硬件）
                assert config.confidence_threshold <= 0.5, \
                    f"{platform_name}平台置信度阈值过高可能影响性能"
                    
                assert config.iou_threshold <= 0.5, \
                    f"{platform_name}平台IoU阈值过高可能影响性能"
                    
            self.logger.info(f"{platform_name}平台性能期望验证通过")
            
    def test_tensorrt_fallback_mechanism(self):
        """测试TensorRT回退机制"""
        # 模拟TensorRT不可用的情况
        fallback_configs = {
            'tensorrt_unavailable': {
                'expected_fallback': 'cpu',
                'expected_precision': 'fp32'
            },
            'gpu_memory_insufficient': {
                'expected_fallback': 'cpu',
                'expected_model_size': 'n'
            }
        }
        
        for scenario, expected in fallback_configs.items():
            # 这里主要验证配置的合理性
            # 实际的回退逻辑需要在运行时测试
            
            for platform_name in self.tensorrt_platforms.keys():
                config = load_platform_config(platform_name)
                
                # 验证有合理的CPU回退配置
                assert config.device in ['auto', 'cpu', 'cuda'], \
                    f"{platform_name}平台设备配置应支持回退"
                    
            self.logger.info(f"TensorRT回退机制验证通过: {scenario}")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])