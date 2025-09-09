#!/usr/bin/env python3
"""
图像预处理器
提供基础的图像预处理功能
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional

class ImagePreprocessor:
    """图像预处理器"""
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (640, 640),
                 normalize: bool = True,
                 mean: list = None,
                 std: list = None):
        """
        初始化图像预处理器
        
        Args:
            target_size: 目标图像尺寸 (width, height)
            normalize: 是否归一化
            mean: 归一化均值
            std: 归一化标准差
        """
        self.target_size = target_size
        self.normalize = normalize
        self.mean = np.array(mean or [0.485, 0.456, 0.406])
        self.std = np.array(std or [0.229, 0.224, 0.225])
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        预处理图像
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            预处理后的图像和元数据
        """
        if image is None:
            raise ValueError("输入图像不能为空")
        
        original_shape = image.shape[:2]  # (height, width)
        
        # 调整图像大小
        processed_image = cv2.resize(image, self.target_size)
        
        # 转换颜色空间 BGR -> RGB
        if len(processed_image.shape) == 3 and processed_image.shape[2] == 3:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        
        # 归一化
        if self.normalize:
            processed_image = processed_image.astype(np.float32) / 255.0
            
            # 应用均值和标准差
            if len(processed_image.shape) == 3:
                processed_image = (processed_image - self.mean) / self.std
        
        # 转换为CHW格式 (Channels, Height, Width)
        if len(processed_image.shape) == 3:
            processed_image = np.transpose(processed_image, (2, 0, 1))
        
        # 添加batch维度
        processed_image = np.expand_dims(processed_image, axis=0)
        
        # 计算缩放比例
        scale_x = self.target_size[0] / original_shape[1]
        scale_y = self.target_size[1] / original_shape[0]
        
        metadata = {
            'original_shape': original_shape,
            'processed_shape': self.target_size,
            'scale_x': scale_x,
            'scale_y': scale_y,
            'normalized': self.normalize
        }
        
        return processed_image, metadata
    
    def postprocess_coordinates(self, coordinates: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """
        将坐标从处理后的图像空间转换回原始图像空间
        
        Args:
            coordinates: 处理后图像空间的坐标 [x1, y1, x2, y2, ...]
            metadata: 预处理元数据
            
        Returns:
            原始图像空间的坐标
        """
        if len(coordinates) == 0:
            return coordinates
        
        coords = coordinates.copy()
        scale_x = metadata.get('scale_x', 1.0)
        scale_y = metadata.get('scale_y', 1.0)
        original_shape = metadata.get('original_shape', (640, 640))
        
        # 转换坐标
        coords[0::2] /= scale_x  # x坐标
        coords[1::2] /= scale_y  # y坐标
        
        # 裁剪到原始图像边界
        coords[0::2] = np.clip(coords[0::2], 0, original_shape[1])  # x坐标
        coords[1::2] = np.clip(coords[1::2], 0, original_shape[0])  # y坐标
        
        return coords
    
    def denormalize(self, image: np.ndarray) -> np.ndarray:
        """
        反归一化图像
        
        Args:
            image: 归一化的图像
            
        Returns:
            反归一化的图像
        """
        if not self.normalize:
            return image
        
        # 移除batch维度
        if len(image.shape) == 4:
            image = image[0]
        
        # 转换为HWC格式
        if len(image.shape) == 3 and image.shape[0] <= 4:
            image = np.transpose(image, (1, 2, 0))
        
        # 反归一化
        image = image * self.std + self.mean
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        
        # 转换回BGR
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image
    
    def __str__(self) -> str:
        return f"ImagePreprocessor(target_size={self.target_size}, normalize={self.normalize})"
    
    def __repr__(self) -> str:
        return self.__str__()