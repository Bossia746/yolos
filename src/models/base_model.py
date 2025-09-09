"""
YOLO模型基类
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import numpy as np
import cv2
import torch
from pathlib import Path


class BaseYOLOModel(ABC):
    """YOLO模型基类"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.class_names = []
        self.input_size = (640, 640)
        
        if model_path:
            self.load_model(model_path)
    
    @abstractmethod
    def load_model(self, model_path: str):
        """加载模型"""
        pass
    
    @abstractmethod
    def predict(self, image: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        """预测"""
        pass
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """预处理"""
        pass
    
    @abstractmethod
    def postprocess(self, outputs: torch.Tensor, image_shape: tuple) -> List[Dict[str, Any]]:
        """后处理"""
        pass
    
    def predict_batch(self, images: List[np.ndarray], **kwargs) -> List[List[Dict[str, Any]]]:
        """批量预测"""
        results = []
        for image in images:
            result = self.predict(image, **kwargs)
            results.append(result)
        return results
    
    def predict_video(self, video_path: str, output_path: Optional[str] = None, **kwargs):
        """视频预测"""
        cap = cv2.VideoCapture(video_path)
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 预测
            results = self.predict(frame, **kwargs)
            
            # 绘制结果
            annotated_frame = self.draw_results(frame, results)
            
            if output_path:
                out.write(annotated_frame)
            else:
                cv2.imshow('YOLO Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
    
    def draw_results(self, image: np.ndarray, results: List[Dict[str, Any]]) -> np.ndarray:
        """绘制检测结果"""
        annotated_image = image.copy()
        
        for result in results:
            bbox = result['bbox']
            confidence = result['confidence']
            class_name = result['class_name']
            
            # 绘制边界框
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated_image
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_path': self.model_path,
            'device': self.device,
            'input_size': self.input_size,
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
        }
    
    def save_model(self, save_path: str):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型未加载")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'input_size': self.input_size,
        }, save_path)