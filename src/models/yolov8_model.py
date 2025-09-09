"""
YOLOv8模型实现
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
import cv2
from pathlib import Path

from .base_model import BaseYOLOModel

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class YOLOv8Model(BaseYOLOModel):
    """YOLOv8模型实现"""
    
    SUPPORTED_FORMATS = ['pt', 'onnx', 'torchscript', 'tensorrt']
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        super().__init__(model_path, device)
        self.conf_threshold = 0.25
        self.iou_threshold = 0.7
        
        if YOLO is None:
            raise ImportError("请安装ultralytics: pip install ultralytics")
    
    def load_model(self, model_path: str):
        """加载YOLOv8模型"""
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            
            # 获取类别名称
            if hasattr(self.model.model, 'names'):
                self.class_names = list(self.model.model.names.values())
            else:
                self.class_names = [f"class_{i}" for i in range(80)]  # COCO默认80类
            
            self.model_path = model_path
            
        except Exception as e:
            raise RuntimeError(f"加载YOLOv8模型失败: {e}")
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """预处理图像 - YOLOv8内部处理"""
        return image  # YOLOv8内部会处理预处理
    
    def predict(self, image: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        """预测单张图像"""
        if self.model is None:
            raise ValueError("模型未加载")
        
        # 设置参数
        conf_threshold = kwargs.get('conf_threshold', self.conf_threshold)
        iou_threshold = kwargs.get('iou_threshold', self.iou_threshold)
        
        # 预测
        results = self.model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)
        
        # 解析结果
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # 获取置信度和类别
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name
                    })
        
        return detections
    
    def postprocess(self, outputs: torch.Tensor, image_shape: tuple) -> List[Dict[str, Any]]:
        """后处理预测结果"""
        # YOLOv8的后处理已在predict中完成
        return []
    
    def train(self, data_config: str, epochs: int = 100, **kwargs):
        """训练模型"""
        if self.model is None:
            raise ValueError("模型未加载")
        
        # 训练参数
        train_args = {
            'data': data_config,
            'epochs': epochs,
            'device': self.device,
            **kwargs
        }
        
        # 开始训练
        results = self.model.train(**train_args)
        return results
    
    def validate(self, data_config: str, **kwargs):
        """验证模型"""
        if self.model is None:
            raise ValueError("模型未加载")
        
        val_args = {
            'data': data_config,
            'device': self.device,
            **kwargs
        }
        
        results = self.model.val(**val_args)
        return results
    
    def export(self, format: str = 'onnx', **kwargs):
        """导出模型"""
        if self.model is None:
            raise ValueError("模型未加载")
        
        export_args = {
            'format': format,
            **kwargs
        }
        
        path = self.model.export(**export_args)
        print(f"模型已导出到: {path}")
        return path
    
    def track(self, source, **kwargs):
        """目标跟踪"""
        if self.model is None:
            raise ValueError("模型未加载")
        
        track_args = {
            'source': source,
            'device': self.device,
            **kwargs
        }
        
        results = self.model.track(**track_args)
        return results