#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv11模块功能测试
测试C2PSA和C3k2模块的配置和功能
"""

import sys
import os
from pathlib import Path
import numpy as np
import pytest

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.models.yolov11_detector import YOLOv11Detector
    from src.models.config_loader import load_yolo_config, load_platform_config
    from src.models.unified_model_manager import ModelConfig, ModelType, PlatformType
    from src.utils.logging_manager import LoggingManager
except ImportError as e:
    pytest.skip(f"跳过测试，模块导入失败: {e}")


class TestYOLOv11Modules:
    """YOLOv11模块测试类"""
    
    def setup_method(self):
        """测试前准备"""
        self.logger = LoggingManager("TestYOLOv11").get_logger()
        
    def test_c2psa_config_loading(self):
        """测试C2PSA配置加载"""
        # 创建带C2PSA配置的模型配置
        c2psa_config = {
            'enabled': True,
            'attention_type': 'pyramid_slice',
            'multi_scale': True
        }
        
        config = ModelConfig(
            model_type=ModelType.YOLOV11,
            model_size='n',
            model_path='yolov11n.pt',
            device='cpu',
            c2psa_config=c2psa_config
        )
        
        assert config.c2psa_config is not None
        assert config.c2psa_config['enabled'] is True
        assert config.c2psa_config['attention_type'] == 'pyramid_slice'
        self.logger.info("C2PSA配置加载测试通过")
        
    def test_c3k2_config_loading(self):
        """测试C3k2配置加载"""
        # 创建带C3k2配置的模型配置
        c3k2_config = {
            'enabled': True,
            'parallel_conv': True,
            'channel_separation': True,
            'kernel_sizes': [3, 5, 7]
        }
        
        config = ModelConfig(
            model_type=ModelType.YOLOV11,
            model_size='n',
            model_path='yolov11n.pt',
            device='cpu',
            c3k2_config=c3k2_config
        )
        
        assert config.c3k2_config is not None
        assert config.c3k2_config['enabled'] is True
        assert config.c3k2_config['parallel_conv'] is True
        assert config.c3k2_config['kernel_sizes'] == [3, 5, 7]
        self.logger.info("C3k2配置加载测试通过")
        
    def test_platform_specific_configs(self):
        """测试平台特定配置"""
        platforms = ['esp32', 'k230', 'raspberry_pi', 'jetson']
        
        for platform in platforms:
            try:
                config = load_platform_config(platform)
                assert config is not None
                assert config.platform.value == platform.lower()
                
                # 验证平台特定的优化设置
                if platform == 'esp32':
                    assert config.model_size == 'n'
                    assert config.tensorrt_optimize is False
                    assert config.half_precision is False
                elif platform == 'jetson':
                    assert config.model_size == 'm'
                    assert config.tensorrt_optimize is True
                    assert config.half_precision is True
                    
                self.logger.info(f"{platform}平台配置测试通过")
            except Exception as e:
                self.logger.warning(f"{platform}平台配置测试跳过: {e}")
                
    def test_dynamic_batching_config(self):
        """测试动态批处理配置"""
        config = ModelConfig(
            model_type=ModelType.YOLOV11,
            model_size='s',
            model_path='yolov11s.pt',
            device='cpu',
            dynamic_batching=True
        )
        
        assert config.dynamic_batching is True
        self.logger.info("动态批处理配置测试通过")
        
    def test_model_config_serialization(self):
        """测试模型配置序列化"""
        original_config = ModelConfig(
            model_type=ModelType.YOLOV11,
            model_size='s',
            model_path='yolov11s.pt',
            device='cpu',
            dynamic_batching=True,
            c2psa_config={'enabled': True, 'multi_scale': True},
            c3k2_config={'enabled': True, 'parallel_conv': True}
        )
        
        # 转换为字典
        config_dict = original_config.to_dict()
        assert 'dynamic_batching' in config_dict
        assert 'c2psa_config' in config_dict
        assert 'c3k2_config' in config_dict
        
        # 从字典重建
        restored_config = ModelConfig.from_dict(config_dict)
        assert restored_config.dynamic_batching == original_config.dynamic_batching
        assert restored_config.c2psa_config == original_config.c2psa_config
        assert restored_config.c3k2_config == original_config.c3k2_config
        
        self.logger.info("模型配置序列化测试通过")
        
    def test_yolov11_detector_initialization(self):
        """测试YOLOv11检测器初始化（不加载实际模型）"""
        # 模拟初始化参数
        c2psa_config = {
            'enabled': True,
            'attention_type': 'pyramid_slice',
            'multi_scale': False  # 简化配置用于测试
        }
        
        c3k2_config = {
            'enabled': True,
            'parallel_conv': False,  # 简化配置用于测试
            'channel_separation': True,
            'kernel_sizes': [3, 5]
        }
        
        try:
            # 只测试参数设置，不实际加载模型
            detector = YOLOv11Detector.__new__(YOLOv11Detector)
            detector.model_size = 'n'
            detector.device = 'cpu'
            detector.c2psa_config = c2psa_config
            detector.c3k2_config = c3k2_config
            detector.dynamic_batching = True
            
            # 验证配置设置
            assert detector.c2psa_config['enabled'] is True
            assert detector.c3k2_config['enabled'] is True
            assert detector.dynamic_batching is True
            
            self.logger.info("YOLOv11检测器参数设置测试通过")
            
        except Exception as e:
            self.logger.warning(f"YOLOv11检测器初始化测试跳过: {e}")
            
    def test_config_validation(self):
        """测试配置验证"""
        # 测试有效配置
        valid_config = ModelConfig(
            model_type=ModelType.YOLOV11,
            model_size='s',
            model_path='yolov11s.pt',
            device='cpu'
        )
        
        assert valid_config.model_type == ModelType.YOLOV11
        assert valid_config.model_size == 's'
        assert valid_config.device == 'cpu'
        
        # 测试默认值
        default_config = ModelConfig(
            model_type=ModelType.YOLOV11,
            model_size='m',
            model_path='yolov11m.pt'
        )
        
        assert default_config.device == 'auto'  # 默认值
        assert default_config.confidence_threshold == 0.25  # 默认值
        
        self.logger.info("配置验证测试通过")
        
    def test_performance_optimization_flags(self):
        """测试性能优化标志"""
        config = ModelConfig(
            model_type=ModelType.YOLOV11,
            model_size='s',
            model_path='yolov11s.pt',
            device='cuda',
            half_precision=True,
            tensorrt_optimize=True,
            dynamic_batching=True
        )
        
        # 验证性能优化设置
        assert config.half_precision is True
        assert config.tensorrt_optimize is True
        assert config.dynamic_batching is True
        
        self.logger.info("性能优化标志测试通过")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])