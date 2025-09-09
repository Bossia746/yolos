#!/usr/bin/env python3
"""
YOLO目标检测器
主要的检测器类，整合预处理、推理和后处理
"""

import os
import logging
from typing import List, Dict, Any, Union, Optional
import numpy as np
from PIL import Image

from .models.yolo_factory import YOLOFactory
from .preprocessing.image_preprocessor import ImagePreprocessor
from .postprocessing.factory import PostprocessorFactory
from .utils.logger import setup_logger
from .core.config import YOLOSConfig

class YOLODetector:
    """
    YOLO目标检测器
    
    整合了预处理、推理和后处理的完整检测流程
    """
    
    def __init__(self, 
                 model_path: str,
                 config: Optional[YOLOSConfig] = None,
                 device: str = 'cpu',
                 logger: Optional[logging.Logger] = None):
        """
        初始化YOLO检测器
        
        Args:
            model_path: ONNX模型文件路径
            config: YOLOS配置对象
            device: 推理设备 ('cpu' 或 'cuda')
            logger: 日志记录器
        """
        self.model_path = model_path
        self.device = device
        self.logger = logger or setup_logger('YOLODetector')
        
        # 验证模型文件
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 使用配置或默认配置
        self.config = config or YOLOSConfig()
        
        # 初始化组件
        self._init_components()
        
        self.logger.info(f"YOLO检测器初始化完成，模型: {model_path}")
    
    def _init_components(self):
        """
        初始化检测器组件
        """
        try:
            # 初始化YOLO模型
            self.model = YOLOFactory.create_model(
                model_type='yolov8',  # 默认使用YOLOv8
                model_path=self.model_path,
                device=self.device
            )
            
            # 初始化预处理器
            preprocessing_config = self.config.preprocessing
            self.preprocessor = ImagePreprocessor(
                target_size=preprocessing_config.target_size,
                normalize=preprocessing_config.normalize,
                mean=preprocessing_config.mean,
                std=preprocessing_config.std
            )
            
            # 初始化后处理器
            postprocessing_config = self.config.postprocessing
            self.postprocessor = PostprocessorFactory.create_postprocessor(
                'nms',  # 使用NMS后处理器
                {
                    'confidence_threshold': postprocessing_config.confidence_threshold,
                    'nms_threshold': postprocessing_config.nms_threshold,
                    'max_detections': postprocessing_config.max_detections
                }
            )
            
        except Exception as e:
            self.logger.error(f"初始化组件失败: {e}")
            raise
    
    def detect(self, 
               image: Union[str, np.ndarray, Image.Image],
               return_image: bool = False) -> Dict[str, Any]:
        """
        执行目标检测
        
        Args:
            image: 输入图像（文件路径、numpy数组或PIL图像）
            return_image: 是否返回处理后的图像
            
        Returns:
            检测结果字典，包含检测框、置信度等信息
        """
        try:
            # 预处理
            processed_image, metadata = self.preprocessor.preprocess(image)
            
            # 推理
            predictions = self.model.predict(processed_image)
            
            # 后处理
            detections = self.postprocessor.postprocess(predictions, metadata)
            
            # 构建结果
            result = {
                'detections': detections,
                'num_detections': len(detections),
                'metadata': metadata
            }
            
            if return_image:
                result['processed_image'] = processed_image
            
            self.logger.info(f"检测完成，发现 {len(detections)} 个目标")
            return result
            
        except Exception as e:
            self.logger.error(f"检测失败: {e}")
            raise
    
    def detect_batch(self, 
                     images: List[Union[str, np.ndarray, Image.Image]],
                     return_images: bool = False) -> List[Dict[str, Any]]:
        """
        批量检测
        
        Args:
            images: 输入图像列表
            return_images: 是否返回处理后的图像
            
        Returns:
            检测结果列表
        """
        results = []
        
        for i, image in enumerate(images):
            try:
                result = self.detect(image, return_images)
                results.append(result)
            except Exception as e:
                self.logger.error(f"批量检测第{i+1}张图像失败: {e}")
                results.append({
                    'detections': [],
                    'num_detections': 0,
                    'error': str(e)
                })
        
        return results
    
    def update_config(self, new_config: YOLOSConfig):
        """
        更新配置
        
        Args:
            new_config: 新的配置对象
        """
        self.config = new_config
        self._init_components()
        self.logger.info("配置已更新，组件已重新初始化")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            'model_path': self.model_path,
            'device': self.device,
            'config': self.config.to_dict()
        }
    
    def __del__(self):
        """
        析构函数，清理资源
        """
        if hasattr(self, 'model'):
            del self.model
        
        if hasattr(self, 'logger'):
            self.logger.info("YOLO检测器已清理")