#!/usr/bin/env python3
"""
NMS后处理器
提供非极大值抑制功能
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from .base_processor import BaseProcessor

class NMSProcessor(BaseProcessor):
    """非极大值抑制后处理器"""
    
    def __init__(self, 
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4,
                 max_detections: int = 100):
        """
        初始化NMS处理器
        
        Args:
            confidence_threshold: 置信度阈值
            nms_threshold: NMS阈值
            max_detections: 最大检测数量
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
    
    def postprocess(self, predictions: np.ndarray, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        对检测结果进行后处理
        
        Args:
            predictions: 模型预测结果
            metadata: 预处理元数据
            
        Returns:
            处理后的检测结果列表
        """
        if predictions is None or len(predictions) == 0:
            return []
        
        # 确保predictions是2D数组
        if len(predictions.shape) == 3:
            predictions = predictions[0]  # 移除batch维度
        
        # 过滤低置信度检测
        if predictions.shape[1] >= 5:  # 至少包含 [x, y, w, h, confidence]
            valid_mask = predictions[:, 4] >= self.confidence_threshold
            valid_detections = predictions[valid_mask]
        else:
            valid_detections = predictions
        
        if len(valid_detections) == 0:
            return []
        
        # 提取边界框和置信度
        boxes = valid_detections[:, :4]
        confidences = valid_detections[:, 4] if valid_detections.shape[1] > 4 else np.ones(len(boxes))
        
        # 提取类别信息（如果存在）
        if valid_detections.shape[1] > 5:
            class_scores = valid_detections[:, 5:]
            class_ids = np.argmax(class_scores, axis=1)
            class_confidences = np.max(class_scores, axis=1)
        else:
            class_ids = np.zeros(len(boxes), dtype=int)
            class_confidences = confidences
        
        # 转换边界框格式（如果需要）
        if self._is_xywh_format(boxes):
            boxes = self._xywh_to_xyxy(boxes)
        
        # 应用NMS
        keep_indices = self._apply_nms(boxes, confidences)
        
        # 构建结果
        results = []
        for idx in keep_indices[:self.max_detections]:
            # 转换坐标回原始图像（如果有元数据）
            box = boxes[idx]
            if metadata:
                box = self._scale_coordinates(box, metadata)
            
            result = {
                'bbox': box.tolist(),
                'confidence': float(confidences[idx]),
                'class_id': int(class_ids[idx]),
                'class_score': float(class_confidences[idx])
            }
            results.append(result)
        
        return results
    
    def _is_xywh_format(self, boxes: np.ndarray) -> bool:
        """
        判断边界框是否为xywh格式
        简单启发式：如果所有w,h都是正数且x,y看起来像中心点，则认为是xywh
        """
        if len(boxes) == 0:
            return False
        
        # 检查宽高是否都为正数
        w_h_positive = np.all(boxes[:, 2:4] > 0)
        
        # 检查是否可能是中心点格式（这是一个简单的启发式）
        # 如果x+w/2和y+h/2都在合理范围内，可能是中心点格式
        max_coord = np.max(boxes[:, :2] + boxes[:, 2:4] / 2)
        
        return w_h_positive and max_coord < 2.0  # 假设归一化坐标
    
    def _xywh_to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """
        将边界框从(center_x, center_y, width, height)转换为(x1, y1, x2, y2)
        """
        xyxy = np.copy(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        return xyxy
    
    def _apply_nms(self, boxes: np.ndarray, scores: np.ndarray) -> List[int]:
        """
        应用非极大值抑制
        
        Args:
            boxes: 边界框数组 (N, 4) - (x1, y1, x2, y2)
            scores: 置信度数组 (N,)
            
        Returns:
            保留的索引列表
        """
        if len(boxes) == 0:
            return []
        
        # 计算面积
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # 按置信度排序
        order = scores.argsort()[::-1]
        
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
            
            # 计算IoU
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h
            
            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / (union + 1e-6)  # 避免除零
            
            # 保留IoU小于阈值的检测
            indices = np.where(iou <= self.nms_threshold)[0]
            order = order[indices + 1]
        
        return keep
    
    def _scale_coordinates(self, box: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """
        将坐标从处理后的图像缩放回原始图像
        
        Args:
            box: 边界框 [x1, y1, x2, y2]
            metadata: 预处理元数据
            
        Returns:
            缩放后的边界框
        """
        original_shape = metadata.get('original_shape', (640, 640))
        processed_shape = metadata.get('processed_shape', (640, 640))
        
        # 计算缩放比例
        scale_x = original_shape[1] / processed_shape[0]  # width scale
        scale_y = original_shape[0] / processed_shape[1]  # height scale
        
        # 缩放坐标
        scaled_box = box.copy()
        scaled_box[0] *= scale_x  # x1
        scaled_box[1] *= scale_y  # y1
        scaled_box[2] *= scale_x  # x2
        scaled_box[3] *= scale_y  # y2
        
        # 裁剪到原始图像边界
        scaled_box[0] = max(0, min(scaled_box[0], original_shape[1]))
        scaled_box[1] = max(0, min(scaled_box[1], original_shape[0]))
        scaled_box[2] = max(0, min(scaled_box[2], original_shape[1]))
        scaled_box[3] = max(0, min(scaled_box[3], original_shape[0]))
        
        return scaled_box
    
    def __str__(self) -> str:
        return f"NMSProcessor(conf={self.confidence_threshold}, nms={self.nms_threshold}, max={self.max_detections})"
    
    def __repr__(self) -> str:
        return self.__str__()