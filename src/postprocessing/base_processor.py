#!/usr/bin/env python3
"""
基础后处理器
定义后处理器的抽象接口
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

class BaseProcessor(ABC):
    """后处理器基类"""
    
    @abstractmethod
    def postprocess(self, predictions: np.ndarray, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        对模型预测结果进行后处理
        
        Args:
            predictions: 模型预测结果
            metadata: 预处理阶段的元数据
            
        Returns:
            处理后的结果列表，每个元素包含检测信息
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}()"
    
    def __repr__(self) -> str:
        return self.__str__()