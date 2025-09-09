#!/usr/bin/env python3
"""
后处理器工厂
用于创建和管理不同类型的后处理器
"""

from typing import Dict, Any, Optional
from .base_processor import BaseProcessor
from .nms_processor import NMSProcessor

class ProcessorFactory:
    """后处理器工厂类"""
    
    # 注册的处理器类型
    _processors = {
        'nms': NMSProcessor,
    }
    
    @classmethod
    def create_processor(cls, 
                        processor_type: str, 
                        config: Optional[Dict[str, Any]] = None) -> BaseProcessor:
        """
        创建后处理器实例
        
        Args:
            processor_type: 处理器类型 ('nms', 'yolo', 'ssd'等)
            config: 处理器配置参数
            
        Returns:
            处理器实例
            
        Raises:
            ValueError: 不支持的处理器类型
        """
        if processor_type not in cls._processors:
            available_types = list(cls._processors.keys())
            raise ValueError(f"不支持的处理器类型: {processor_type}. 可用类型: {available_types}")
        
        processor_class = cls._processors[processor_type]
        
        if config is None:
            config = {}
        
        try:
            return processor_class(**config)
        except TypeError as e:
            raise ValueError(f"创建{processor_type}处理器失败: {e}")
    
    @classmethod
    def register_processor(cls, name: str, processor_class: type):
        """
        注册新的处理器类型
        
        Args:
            name: 处理器名称
            processor_class: 处理器类
        """
        if not issubclass(processor_class, BaseProcessor):
            raise ValueError(f"处理器类必须继承自BaseProcessor")
        
        cls._processors[name] = processor_class
    
    @classmethod
    def get_available_processors(cls) -> list:
        """
        获取所有可用的处理器类型
        
        Returns:
            可用处理器类型列表
        """
        return list(cls._processors.keys())
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> BaseProcessor:
        """
        从配置字典创建处理器
        
        Args:
            config: 包含type和其他参数的配置字典
            
        Returns:
            处理器实例
            
        Example:
            config = {
                'type': 'nms',
                'confidence_threshold': 0.5,
                'nms_threshold': 0.4
            }
        """
        if 'type' not in config:
            raise ValueError("配置中必须包含'type'字段")
        
        processor_type = config.pop('type')
        return cls.create_processor(processor_type, config)

# 便捷函数
def create_nms_processor(confidence_threshold: float = 0.5,
                        nms_threshold: float = 0.4,
                        max_detections: int = 100) -> NMSProcessor:
    """
    创建NMS处理器的便捷函数
    
    Args:
        confidence_threshold: 置信度阈值
        nms_threshold: NMS阈值
        max_detections: 最大检测数量
        
    Returns:
        NMS处理器实例
    """
    return ProcessorFactory.create_processor('nms', {
        'confidence_threshold': confidence_threshold,
        'nms_threshold': nms_threshold,
        'max_detections': max_detections
    })