# -*- coding: utf-8 -*-
"""
FastTracker模块配置文件
基于FastTracker论文的优化模块配置
"""

from typing import Dict, Any

class FastTrackerConfig:
    """FastTracker相关模块的配置类"""
    
    def __init__(self):
        # SimSPPF模块配置
        self.simsppf_config = {
            'kernel_sizes': [5, 9, 13],  # 多尺度池化核大小
            'use_mish': True,  # 是否使用Mish激活函数
            'use_simam': True,  # 是否使用SimAM注意力
            'dropout_rate': 0.1  # Dropout比例
        }
        
        # C3Ghost模块配置
        self.c3ghost_config = {
            'ratio': 2,  # Ghost卷积比例
            'dw_size': 3,  # 深度卷积核大小
            'use_se': False,  # 是否使用SE注意力
            'act': 'relu'  # 激活函数类型
        }
        
        # IGD模块配置
        self.igd_config = {
            'num_scales': 3,  # 多尺度数量
            'fusion_method': 'adaptive',  # 融合方法: 'adaptive', 'concat', 'add'
            'use_simam': True,  # 是否使用SimAM注意力
            'channel_reduction': 4  # 通道降维比例
        }
        
        # 自适应动态ROI配置
        self.adaptive_roi_config = {
            'base_roi_size': (416, 416),  # 基础ROI大小
            'speed_threshold': 30.0,  # 速度阈值(km/h)
            'angle_threshold': 15.0,  # 转向角阈值(度)
            'expansion_factor': 1.2,  # ROI扩展因子
            'min_roi_size': (224, 224),  # 最小ROI大小
            'max_roi_size': (640, 640)  # 最大ROI大小
        }
        
        # Mish激活函数配置
        self.mish_config = {
            'beta': 1.0,  # Mish参数
            'replace_relu': True,  # 是否替换ReLU
            'replace_silu': False  # 是否替换SiLU
        }
        
        # 模型集成配置
        self.integration_config = {
            'backbone_modifications': {
                'use_c3ghost': True,  # 在backbone中使用C3Ghost
                'ghost_layers': [3, 6, 9]  # 使用Ghost的层索引
            },
            'neck_modifications': {
                'use_igd': True,  # 在neck中使用IGD
                'use_simsppf': True,  # 在neck中使用SimSPPF
                'igd_positions': ['P3', 'P4', 'P5']  # IGD模块位置
            },
            'head_modifications': {
                'use_adaptive_roi': True,  # 使用自适应ROI
                'roi_stages': ['detection', 'tracking']  # ROI应用阶段
            }
        }
    
    def get_config(self, module_name: str) -> Dict[str, Any]:
        """获取指定模块的配置"""
        config_map = {
            'simsppf': self.simsppf_config,
            'c3ghost': self.c3ghost_config,
            'igd': self.igd_config,
            'adaptive_roi': self.adaptive_roi_config,
            'mish': self.mish_config,
            'integration': self.integration_config
        }
        return config_map.get(module_name, {})
    
    def update_config(self, module_name: str, config: Dict[str, Any]):
        """更新指定模块的配置"""
        if module_name == 'simsppf':
            self.simsppf_config.update(config)
        elif module_name == 'c3ghost':
            self.c3ghost_config.update(config)
        elif module_name == 'igd':
            self.igd_config.update(config)
        elif module_name == 'adaptive_roi':
            self.adaptive_roi_config.update(config)
        elif module_name == 'mish':
            self.mish_config.update(config)
        elif module_name == 'integration':
            self.integration_config.update(config)
    
    def validate_config(self) -> bool:
        """验证配置的有效性"""
        try:
            # 验证SimSPPF配置
            assert len(self.simsppf_config['kernel_sizes']) > 0
            assert all(k > 0 for k in self.simsppf_config['kernel_sizes'])
            assert 0 <= self.simsppf_config['dropout_rate'] <= 1
            
            # 验证C3Ghost配置
            assert self.c3ghost_config['ratio'] >= 1
            assert self.c3ghost_config['dw_size'] > 0
            
            # 验证IGD配置
            assert self.igd_config['num_scales'] > 0
            assert self.igd_config['fusion_method'] in ['adaptive', 'concat', 'add']
            assert self.igd_config['channel_reduction'] > 0
            
            # 验证自适应ROI配置
            assert self.adaptive_roi_config['speed_threshold'] > 0
            assert self.adaptive_roi_config['angle_threshold'] > 0
            assert self.adaptive_roi_config['expansion_factor'] > 1
            
            return True
        except AssertionError:
            return False

# 默认配置实例
default_fasttracker_config = FastTrackerConfig()