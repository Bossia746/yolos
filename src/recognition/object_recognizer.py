#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
静物识别模块 - 基于OpenCV和深度学习
识别日常物品的颜色、外观、形状、字帖等
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Any
import colorsys
import math

logger = logging.getLogger(__name__)

class ObjectRecognizer:
    """静物识别器"""
    
    def __init__(self):
        """初始化静物识别器"""
        self.color_ranges = self._init_color_ranges()
        self.shape_detector = ShapeDetector()
        self.text_detector = TextDetector()
        self.contour_min_area = 500  # 最小轮廓面积
        
    def _init_color_ranges(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """初始化颜色范围"""
        return {
            'red': (np.array([0, 50, 50]), np.array([10, 255, 255])),
            'red2': (np.array([170, 50, 50]), np.array([180, 255, 255])),
            'orange': (np.array([11, 50, 50]), np.array([25, 255, 255])),
            'yellow': (np.array([26, 50, 50]), np.array([35, 255, 255])),
            'green': (np.array([36, 50, 50]), np.array([85, 255, 255])),
            'blue': (np.array([86, 50, 50]), np.array([125, 255, 255])),
            'purple': (np.array([126, 50, 50]), np.array([145, 255, 255])),
            'pink': (np.array([146, 50, 50]), np.array([169, 255, 255])),
            'white': (np.array([0, 0, 200]), np.array([180, 30, 255])),
            'black': (np.array([0, 0, 0]), np.array([180, 255, 50])),
            'gray': (np.array([0, 0, 51]), np.array([180, 30, 199]))
        }
    
    def detect_objects(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """检测静物对象"""
        try:
            results = []
            annotated_frame = frame.copy()
            
            # 预处理
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            
            # 边缘检测用于形状识别
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # 查找轮廓
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area < self.contour_min_area:
                    continue
                
                # 获取边界框
                x, y, w, h = cv2.boundingRect(contour)
                roi = frame[y:y+h, x:x+w]
                roi_hsv = hsv[y:y+h, x:x+w]
                
                # 检测颜色
                dominant_color = self._detect_dominant_color(roi_hsv)
                color_name = self._get_color_name(dominant_color)
                
                # 检测形状
                shape_info = self.shape_detector.detect_shape(contour)
                
                # 检测文字（如果是矩形区域且面积足够大）
                text_info = None
                if shape_info['name'] in ['rectangle', 'square'] and area > 2000:
                    text_info = self.text_detector.detect_text(roi)
                
                # 计算物体特征
                object_info = {
                    'id': i,
                    'bbox': (x, y, w, h),
                    'area': area,
                    'color': {
                        'name': color_name,
                        'hsv': dominant_color,
                        'rgb': self._hsv_to_rgb(dominant_color)
                    },
                    'shape': shape_info,
                    'text': text_info,
                    'appearance': self._analyze_appearance(roi),
                    'center': (x + w//2, y + h//2)
                }
                
                results.append(object_info)
                
                # 绘制检测结果
                self._draw_object_info(annotated_frame, object_info)
            
            return annotated_frame, results
            
        except Exception as e:
            logger.error(f"Object detection error: {e}")
            return frame, []
    
    def _detect_dominant_color(self, hsv_roi: np.ndarray) -> Tuple[int, int, int]:
        """检测主导颜色"""
        try:
            # 计算HSV直方图
            hist_h = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
            hist_s = cv2.calcHist([hsv_roi], [1], None, [256], [0, 256])
            hist_v = cv2.calcHist([hsv_roi], [2], None, [256], [0, 256])
            
            # 找到主导值
            dominant_h = np.argmax(hist_h)
            dominant_s = np.argmax(hist_s)
            dominant_v = np.argmax(hist_v)
            
            return (int(dominant_h), int(dominant_s), int(dominant_v))
            
        except Exception as e:
            logger.debug(f"Color detection error: {e}")
            return (0, 0, 0)
    
    def _get_color_name(self, hsv_color: Tuple[int, int, int]) -> str:
        """根据HSV值获取颜色名称"""
        h, s, v = hsv_color
        
        # 低饱和度判断为灰度
        if s < 30:
            if v < 50:
                return 'black'
            elif v > 200:
                return 'white'
            else:
                return 'gray'
        
        # 根据色相判断颜色
        for color_name, (lower, upper) in self.color_ranges.items():
            if color_name in ['red2', 'white', 'black', 'gray']:
                continue
            
            if (lower[0] <= h <= upper[0] and 
                lower[1] <= s <= upper[1] and 
                lower[2] <= v <= upper[2]):
                return color_name
        
        # 特殊处理红色（跨越0度）
        if h <= 10 or h >= 170:
            return 'red'
        
        return 'unknown'
    
    def _hsv_to_rgb(self, hsv_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """HSV转RGB"""
        h, s, v = hsv_color
        h = h / 180.0
        s = s / 255.0
        v = v / 255.0
        
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (int(r * 255), int(g * 255), int(b * 255))
    
    def _analyze_appearance(self, roi: np.ndarray) -> Dict[str, Any]:
        """分析物体外观特征"""
        try:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # 纹理分析
            texture_score = np.std(gray_roi)
            
            # 边缘密度
            edges = cv2.Canny(gray_roi, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # 亮度分析
            brightness = np.mean(gray_roi)
            
            # 对比度分析
            contrast = np.std(gray_roi)
            
            return {
                'texture_score': float(texture_score),
                'edge_density': float(edge_density),
                'brightness': float(brightness),
                'contrast': float(contrast),
                'surface_type': self._classify_surface(texture_score, edge_density)
            }
            
        except Exception as e:
            logger.debug(f"Appearance analysis error: {e}")
            return {}
    
    def _classify_surface(self, texture_score: float, edge_density: float) -> str:
        """分类表面类型"""
        if texture_score < 20 and edge_density < 0.1:
            return 'smooth'
        elif texture_score > 50 and edge_density > 0.3:
            return 'rough'
        elif edge_density > 0.2:
            return 'textured'
        else:
            return 'regular'
    
    def _draw_object_info(self, frame: np.ndarray, obj_info: Dict[str, Any]):
        """绘制物体信息"""
        try:
            x, y, w, h = obj_info['bbox']
            color_name = obj_info['color']['name']
            shape_name = obj_info['shape']['name']
            
            # 绘制边界框
            color_bgr = obj_info['color']['rgb'][::-1]  # RGB转BGR
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)
            
            # 绘制标签
            label_lines = [
                f"OBJ_{obj_info['id']+1}",
                f"Color: {color_name}",
                f"Shape: {shape_name}",
                f"Area: {int(obj_info['area'])}"
            ]
            
            # 添加文字信息
            if obj_info.get('text') and obj_info['text'].get('text'):
                label_lines.append(f"Text: {obj_info['text']['text'][:10]}")
            
            # 添加表面类型
            if obj_info.get('appearance') and obj_info['appearance'].get('surface_type'):
                label_lines.append(f"Surface: {obj_info['appearance']['surface_type']}")
            
            # 绘制多行标签
            for i, line in enumerate(label_lines):
                label_y = y - 10 - (len(label_lines) - 1 - i) * 15
                if label_y < 15:
                    label_y = y + h + 15 + i * 15
                
                cv2.putText(frame, line, (x, label_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 1)
            
            # 绘制中心点
            center_x, center_y = obj_info['center']
            cv2.circle(frame, (center_x, center_y), 3, color_bgr, -1)
            
        except Exception as e:
            logger.debug(f"Draw object info error: {e}")


class ShapeDetector:
    """形状检测器"""
    
    def detect_shape(self, contour: np.ndarray) -> Dict[str, Any]:
        """检测形状"""
        try:
            # 计算轮廓近似
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            vertices = len(approx)
            
            # 计算面积和周长
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # 计算圆形度
            if perimeter > 0:
                circularity = 4 * math.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
            
            # 计算长宽比
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # 形状分类
            shape_name = self._classify_shape(vertices, circularity, aspect_ratio, area)
            
            return {
                'name': shape_name,
                'vertices': vertices,
                'area': area,
                'perimeter': perimeter,
                'circularity': circularity,
                'aspect_ratio': aspect_ratio,
                'bounding_rect': (x, y, w, h)
            }
            
        except Exception as e:
            logger.debug(f"Shape detection error: {e}")
            return {'name': 'unknown', 'vertices': 0}
    
    def _classify_shape(self, vertices: int, circularity: float, aspect_ratio: float, area: float) -> str:
        """分类形状"""
        # 圆形
        if circularity > 0.7:
            return 'circle'
        
        # 根据顶点数分类
        if vertices == 3:
            return 'triangle'
        elif vertices == 4:
            if 0.9 <= aspect_ratio <= 1.1:
                return 'square'
            else:
                return 'rectangle'
        elif vertices == 5:
            return 'pentagon'
        elif vertices == 6:
            return 'hexagon'
        elif vertices > 6:
            if circularity > 0.5:
                return 'circle'
            else:
                return 'polygon'
        else:
            return 'irregular'


class TextDetector:
    """文字检测器"""
    
    def __init__(self):
        """初始化文字检测器"""
        try:
            # 尝试导入OCR库
            import easyocr
            self.reader = easyocr.Reader(['en', 'ch_sim'])
            self.ocr_available = True
        except ImportError:
            logger.warning("EasyOCR not available, using basic text detection")
            self.ocr_available = False
    
    def detect_text(self, roi: np.ndarray) -> Dict[str, Any]:
        """检测文字"""
        try:
            if self.ocr_available:
                return self._detect_text_ocr(roi)
            else:
                return self._detect_text_basic(roi)
                
        except Exception as e:
            logger.debug(f"Text detection error: {e}")
            return {}
    
    def _detect_text_ocr(self, roi: np.ndarray) -> Dict[str, Any]:
        """使用OCR检测文字"""
        try:
            results = self.reader.readtext(roi)
            
            if results:
                # 取置信度最高的结果
                best_result = max(results, key=lambda x: x[2])
                bbox, text, confidence = best_result
                
                return {
                    'text': text,
                    'confidence': confidence,
                    'bbox': bbox,
                    'method': 'ocr'
                }
            
            return {}
            
        except Exception as e:
            logger.debug(f"OCR text detection error: {e}")
            return {}
    
    def _detect_text_basic(self, roi: np.ndarray) -> Dict[str, Any]:
        """基础文字检测"""
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # 检测文字区域特征
            # 计算水平和垂直投影
            h_projection = np.sum(gray < 128, axis=0)
            v_projection = np.sum(gray < 128, axis=1)
            
            # 检测是否有文字特征
            h_variance = np.var(h_projection)
            v_variance = np.var(v_projection)
            
            # 简单的文字区域判断
            if h_variance > 100 and v_variance > 100:
                return {
                    'text': 'TEXT_DETECTED',
                    'confidence': 0.5,
                    'method': 'basic',
                    'h_variance': h_variance,
                    'v_variance': v_variance
                }
            
            return {}
            
        except Exception as e:
            logger.debug(f"Basic text detection error: {e}")
            return {}