#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOS原生检测器模块
提供YOLO原生检测功能
"""

import os
import cv2
import time
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent / "src"))

# 尝试导入YOLOS模块
try:
    from models.yolo_factory import YOLOFactory
    from detection.image_detector import ImageDetector
    YOLOS_AVAILABLE = True
except ImportError as e:
    YOLOS_AVAILABLE = False
    print(f"⚠️ YOLOS模块导入失败: {e}")


class YOLOSNativeDetector:
    """YOLOS原生检测器"""
    
    def __init__(self):
        """初始化原生检测器"""
        self.available = YOLOS_AVAILABLE
        self.detector = None
        
        if self.available:
            try:
                # 尝试创建YOLOv8检测器
                self.detector = ImageDetector(model_type='yolov8', device='cpu')
                print("✓ YOLOS原生检测器初始化成功")
            except Exception as e:
                print(f"⚠️ YOLOS检测器初始化失败: {e}")
                self.available = False
        
        if not self.available:
            print("⚠️ 使用模拟YOLO检测结果")
    
    def detect_objects(self, image_path: str) -> Dict[str, Any]:
        """检测图像中的物体"""
        try:
            start_time = time.time()
            
            if self.available and self.detector:
                # 使用真实的YOLO检测
                results = self.detector.detect_image(image_path, save_results=False)
                processing_time = time.time() - start_time
                
                return {
                    "success": True,
                    "method": "YOLOS Native YOLO",
                    "detections": results,
                    "processing_time": round(processing_time, 3),
                    "detection_count": len(results) if results else 0
                }
            else:
                # 模拟YOLO检测结果
                processing_time = time.time() - start_time
                mock_detections = self._generate_mock_yolo_results(image_path)
                
                return {
                    "success": True,
                    "method": "Mock YOLO Detection",
                    "detections": mock_detections,
                    "processing_time": round(processing_time, 3),
                    "detection_count": len(mock_detections),
                    "note": "模拟结果 - 实际部署时将使用真实YOLO检测"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "method": "YOLOS Native",
                "processing_time": time.time() - start_time
            }
    
    def _generate_mock_yolo_results(self, image_path: str) -> List[Dict[str, Any]]:
        """生成模拟的YOLO检测结果"""
        try:
            # 读取图像获取基本信息
            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                if img is not None:
                    height, width = img.shape[:2]
                else:
                    width, height = 640, 480
            else:
                width, height = 640, 480
            
            # 生成模拟检测结果
            mock_objects = [
                {
                    "class": "person",
                    "confidence": 0.85,
                    "bbox": [width*0.2, height*0.1, width*0.4, height*0.8],
                    "class_id": 0
                },
                {
                    "class": "chair", 
                    "confidence": 0.72,
                    "bbox": [width*0.6, height*0.5, width*0.3, height*0.4],
                    "class_id": 56
                },
                {
                    "class": "bottle",
                    "confidence": 0.68,
                    "bbox": [width*0.1, height*0.3, width*0.1, height*0.2],
                    "class_id": 39
                },
                {
                    "class": "book",
                    "confidence": 0.61,
                    "bbox": [width*0.7, height*0.2, width*0.15, height*0.1],
                    "class_id": 73
                },
                {
                    "class": "cup",
                    "confidence": 0.59,
                    "bbox": [width*0.8, height*0.4, width*0.08, height*0.12],
                    "class_id": 41
                }
            ]
            
            return mock_objects
            
        except Exception as e:
            print(f"生成模拟结果时出错: {e}")
            return []
    
    def is_available(self) -> bool:
        """检查检测器是否可用"""
        return self.available
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if self.available and self.detector:
            return {
                "model_type": "YOLOv8",
                "device": "cpu",
                "available": True
            }
        else:
            return {
                "model_type": "Mock",
                "device": "cpu", 
                "available": False,
                "note": "使用模拟检测结果"
            }