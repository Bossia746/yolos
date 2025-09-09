#!/usr/bin/env python3
"""
后处理器工厂
用于创建和管理不同类型的后处理器
"""

from typing import Dict, Any, Optional
from .base_processor import BaseProcessor
from .nms_processor import NMSProcessor

class PostprocessorFactory:
    """后处理器工厂类"""
    
    # 注册的后处理器类型
    _postprocessors = {
        'nms': NMSProcessor,
        'yolo': NMSProcessor,  # YOLO使用NMS后处理
    }
    
    @classmethod
    def create_postprocessor(cls, 
                           postprocessor_type: str, 
                           config: Optional[Dict[str, Any]] = None) -> BaseProcessor:
        """
        创建后处理器实例
        
        Args:
            postprocessor_type: 后处理器类型
            config: 后处理器配置参数
            
        Returns:
            后处理器实例
            
        Raises:
            ValueError: 不支持的后处理器类型
        """
        if postprocessor_type not in cls._postprocessors:
            available_types = list(cls._postprocessors.keys())
            raise ValueError(f"不支持的后处理器类型: {postprocessor_type}. 可用类型: {available_types}")
        
        postprocessor_class = cls._postprocessors[postprocessor_type]
        
        if config is None:
            config = {}
        
        try:
            return postprocessor_class(**config)
        except TypeError as e:
            raise ValueError(f"创建{postprocessor_type}后处理器失败: {e}")
    

    
    @classmethod
    def register_postprocessor(cls, name: str, postprocessor_class: type):
        """
        注册新的后处理器类型
        
        Args:
            name: 后处理器名称
            postprocessor_class: 后处理器类
        """
        if not issubclass(postprocessor_class, BaseProcessor):
            raise ValueError(f"后处理器类必须继承自BaseProcessor")
        
        cls._postprocessors[name] = postprocessor_class
    
    @classmethod
    def get_available_postprocessors(cls) -> list:
        """
        获取所有可用的后处理器类型
        
        Returns:
            可用后处理器类型列表
        """
        return list(cls._postprocessors.keys())