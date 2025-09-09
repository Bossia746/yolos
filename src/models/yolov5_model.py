"""
YOLOv5模型实现
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
import cv2
from pathlib import Path

from .base_model import BaseYOLOModel


class YOLOv5Model(BaseYOLOModel):
    """YOLOv5模型实现"""
    
    SUPPORTED_FORMATS = ['pt', 'onnx', 'torchscript']
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        super().__init__(model_path, device)
        self.stride = 32
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
    
    def load_model(self, model_path: str):
        """加载YOLOv5模型"""
        try:
            # 尝试加载本地模型
            if Path(model_path).exists():
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                          path=model_path, device=self.device)
            else:
                # 加载预训练模型
                self.model = torch.hub.load('ultralytics/yolov5', model_path, 
                                          device=self.device)
            
            self.class_names = self.model.names
            self.model_path = model_path
            
        except Exception as e:
            raise RuntimeError(f"加载YOLOv5模型失败: {e}")
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """预处理图像"""
        # 调整图像大小
        img = cv2.resize(image, self.input_size)
        
        # BGR转RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 归一化
        img = img.astype(np.float32) / 255.0
        
        # HWC转CHW
        img = np.transpose(img, (2, 0, 1))
        
        # 添加batch维度
        img = np.expand_dims(img, axis=0)
        
        return torch.from_numpy(img).to(self.device)
    
    def predict(self, image: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        """预测单张图像"""
        if self.model is None:
            raise ValueError("模型未加载")
        
        # 设置参数
        conf_threshold = kwargs.get('conf_threshold', self.conf_threshold)
        iou_threshold = kwargs.get('iou_threshold', self.iou_threshold)
        
        # 预测
        results = self.model(image)
        
        # 解析结果
        detections = []
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            if conf >= conf_threshold:
                x1, y1, x2, y2 = box
                class_id = int(cls)
                class_name = self.class_names[class_id]
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'class_id': class_id,
                    'class_name': class_name
                })
        
        return detections
    
    def postprocess(self, outputs: torch.Tensor, image_shape: tuple) -> List[Dict[str, Any]]:
        """后处理预测结果"""
        # YOLOv5的后处理已在predict中完成
        return []
    
    def export_onnx(self, export_path: str, input_size: tuple = (640, 640)):
        """导出ONNX模型"""
        if self.model is None:
            raise ValueError("模型未加载")
        
        dummy_input = torch.randn(1, 3, *input_size).to(self.device)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            export_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"模型已导出到: {export_path}")
    
    def set_confidence_threshold(self, threshold: float):
        """设置置信度阈值"""
        self.conf_threshold = threshold
        if self.model:
            self.model.conf = threshold
    
    def set_iou_threshold(self, threshold: float):
        """设置IoU阈值"""
        self.iou_threshold = threshold
        if self.model:
            self.model.iou = threshold