#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强静物识别模块 - 支持二维码、条形码、车牌、交通符号等
基于OpenCV和深度学习的多功能识别系统
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Any, Optional
import colorsys
import math
import re

logger = logging.getLogger(__name__)

class EnhancedObjectRecognizer:
    """增强静物识别器 - 支持多种特殊对象识别"""
    
    def __init__(self):
        """初始化增强静物识别器"""
        # 基础识别功能
        self.color_ranges = self._init_color_ranges()
        self.shape_detector = ShapeDetector()
        self.text_detector = TextDetector()
        
        # 新增识别功能
        self.qr_detector = QRCodeDetector()
        self.barcode_detector = BarcodeDetector()
        self.license_plate_detector = LicensePlateDetector()
        self.traffic_sign_detector = TrafficSignDetector()
        self.traffic_light_detector = TrafficLightDetector()
        self.facility_sign_detector = FacilitySignDetector()
        
        self.contour_min_area = 500
        
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
        """检测所有类型的对象"""
        try:
            results = []
            annotated_frame = frame.copy()
            
            # 1. 二维码检测
            qr_results = self.qr_detector.detect_qr_codes(frame)
            for qr in qr_results:
                qr['type'] = 'qr_code'
                results.append(qr)
                self._draw_qr_code(annotated_frame, qr)
            
            # 2. 条形码检测
            barcode_results = self.barcode_detector.detect_barcodes(frame)
            for barcode in barcode_results:
                barcode['type'] = 'barcode'
                results.append(barcode)
                self._draw_barcode(annotated_frame, barcode)
            
            # 3. 车牌检测
            license_results = self.license_plate_detector.detect_license_plates(frame)
            for license in license_results:
                license['type'] = 'license_plate'
                results.append(license)
                self._draw_license_plate(annotated_frame, license)
            
            # 4. 交通标志检测
            traffic_sign_results = self.traffic_sign_detector.detect_traffic_signs(frame)
            for sign in traffic_sign_results:
                sign['type'] = 'traffic_sign'
                results.append(sign)
                self._draw_traffic_sign(annotated_frame, sign)
            
            # 5. 交通灯检测
            traffic_light_results = self.traffic_light_detector.detect_traffic_lights(frame)
            for light in traffic_light_results:
                light['type'] = 'traffic_light'
                results.append(light)
                self._draw_traffic_light(annotated_frame, light)
            
            # 6. 公共设施标志检测
            facility_results = self.facility_sign_detector.detect_facility_signs(frame)
            for facility in facility_results:
                facility['type'] = 'facility_sign'
                results.append(facility)
                self._draw_facility_sign(annotated_frame, facility)
            
            # 7. 基础静物检测（原有功能）
            basic_results = self._detect_basic_objects(frame)
            for obj in basic_results:
                obj['type'] = 'basic_object'
                results.append(obj)
                self._draw_basic_object(annotated_frame, obj)
            
            return annotated_frame, results
            
        except Exception as e:
            logger.error(f"Enhanced object detection error: {e}")
            return frame, []
    
    def _detect_basic_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """基础静物检测（原有功能）"""
        try:
            results = []
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area < self.contour_min_area:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                roi = frame[y:y+h, x:x+w]
                roi_hsv = hsv[y:y+h, x:x+w]
                
                # 检测颜色
                dominant_color = self._detect_dominant_color(roi_hsv)
                color_name = self._get_color_name(dominant_color)
                
                # 检测形状
                shape_info = self.shape_detector.detect_shape(contour)
                
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
                    'center': (x + w//2, y + h//2)
                }
                
                results.append(object_info)
            
            return results
            
        except Exception as e:
            logger.debug(f"Basic object detection error: {e}")
            return []
    
    def _detect_dominant_color(self, hsv_roi: np.ndarray) -> Tuple[int, int, int]:
        """检测主导颜色"""
        try:
            hist_h = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
            hist_s = cv2.calcHist([hsv_roi], [1], None, [256], [0, 256])
            hist_v = cv2.calcHist([hsv_roi], [2], None, [256], [0, 256])
            
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
        
        if s < 30:
            if v < 50:
                return 'black'
            elif v > 200:
                return 'white'
            else:
                return 'gray'
        
        for color_name, (lower, upper) in self.color_ranges.items():
            if color_name in ['red2', 'white', 'black', 'gray']:
                continue
            
            if (lower[0] <= h <= upper[0] and 
                lower[1] <= s <= upper[1] and 
                lower[2] <= v <= upper[2]):
                return color_name
        
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
    
    # 绘制函数
    def _draw_qr_code(self, frame: np.ndarray, qr_info: Dict[str, Any]):
        """绘制二维码信息"""
        try:
            bbox = qr_info.get('bbox')
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(frame, f"QR: {qr_info.get('data', 'Unknown')[:20]}", 
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        except Exception as e:
            logger.debug(f"Draw QR code error: {e}")
    
    def _draw_barcode(self, frame: np.ndarray, barcode_info: Dict[str, Any]):
        """绘制条形码信息"""
        try:
            bbox = barcode_info.get('bbox')
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                cv2.putText(frame, f"Barcode: {barcode_info.get('data', 'Unknown')}", 
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        except Exception as e:
            logger.debug(f"Draw barcode error: {e}")
    
    def _draw_license_plate(self, frame: np.ndarray, license_info: Dict[str, Any]):
        """绘制车牌信息"""
        try:
            bbox = license_info.get('bbox')
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)
                cv2.putText(frame, f"License: {license_info.get('plate_number', 'Unknown')}", 
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        except Exception as e:
            logger.debug(f"Draw license plate error: {e}")
    
    def _draw_traffic_sign(self, frame: np.ndarray, sign_info: Dict[str, Any]):
        """绘制交通标志信息"""
        try:
            bbox = sign_info.get('bbox')
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 3)
                cv2.putText(frame, f"Sign: {sign_info.get('sign_type', 'Unknown')}", 
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        except Exception as e:
            logger.debug(f"Draw traffic sign error: {e}")
    
    def _draw_traffic_light(self, frame: np.ndarray, light_info: Dict[str, Any]):
        """绘制交通灯信息"""
        try:
            bbox = light_info.get('bbox')
            if bbox:
                x, y, w, h = bbox
                color = light_info.get('light_color', 'unknown')
                draw_color = (0, 0, 255) if color == 'red' else (0, 255, 255) if color == 'yellow' else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), draw_color, 3)
                cv2.putText(frame, f"Light: {color.upper()}", 
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)
        except Exception as e:
            logger.debug(f"Draw traffic light error: {e}")
    
    def _draw_facility_sign(self, frame: np.ndarray, facility_info: Dict[str, Any]):
        """绘制公共设施标志信息"""
        try:
            bbox = facility_info.get('bbox')
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3)
                cv2.putText(frame, f"Facility: {facility_info.get('facility_type', 'Unknown')}", 
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        except Exception as e:
            logger.debug(f"Draw facility sign error: {e}")
    
    def _draw_basic_object(self, frame: np.ndarray, obj_info: Dict[str, Any]):
        """绘制基础物体信息"""
        try:
            x, y, w, h = obj_info['bbox']
            color_name = obj_info['color']['name']
            shape_name = obj_info['shape']['name']
            
            color_bgr = obj_info['color']['rgb'][::-1]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)
            
            label = f"Obj: {color_name} {shape_name}"
            cv2.putText(frame, label, (x, y-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 1)
        except Exception as e:
            logger.debug(f"Draw basic object error: {e}")


class QRCodeDetector:
    """二维码检测器"""
    
    def __init__(self):
        """初始化二维码检测器"""
        self.qr_detector = cv2.QRCodeDetector()
    
    def detect_qr_codes(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """检测二维码"""
        try:
            results = []
            data, bbox, _ = self.qr_detector.detectAndDecode(frame)
            
            if data and bbox is not None:
                # 计算边界框
                bbox = bbox.astype(int)
                x_min, y_min = np.min(bbox[0], axis=0)
                x_max, y_max = np.max(bbox[0], axis=0)
                
                qr_info = {
                    'data': data,
                    'bbox': (x_min, y_min, x_max - x_min, y_max - y_min),
                    'confidence': 1.0,
                    'corners': bbox[0].tolist()
                }
                results.append(qr_info)
            
            return results
            
        except Exception as e:
            logger.debug(f"QR code detection error: {e}")
            return []


class BarcodeDetector:
    """条形码检测器"""
    
    def __init__(self):
        """初始化条形码检测器"""
        try:
            import pyzbar.pyzbar as pyzbar
            self.pyzbar = pyzbar
            self.available = True
        except ImportError:
            logger.warning("pyzbar not available, using basic barcode detection")
            self.available = False
    
    def detect_barcodes(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """检测条形码"""
        try:
            results = []
            
            if self.available:
                # 使用pyzbar检测
                barcodes = self.pyzbar.decode(frame)
                for barcode in barcodes:
                    x, y, w, h = barcode.rect
                    data = barcode.data.decode('utf-8')
                    
                    barcode_info = {
                        'data': data,
                        'type': barcode.type,
                        'bbox': (x, y, w, h),
                        'confidence': 1.0
                    }
                    results.append(barcode_info)
            else:
                # 基础条形码检测
                results = self._detect_barcodes_basic(frame)
            
            return results
            
        except Exception as e:
            logger.debug(f"Barcode detection error: {e}")
            return []
    
    def _detect_barcodes_basic(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """基础条形码检测"""
        try:
            results = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 检测条形码特征（垂直线条）
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
            closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # 最小面积阈值
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / float(h)
                    
                    # 条形码通常是宽长形
                    if aspect_ratio > 2.0:
                        barcode_info = {
                            'data': 'BARCODE_DETECTED',
                            'type': 'unknown',
                            'bbox': (x, y, w, h),
                            'confidence': 0.7
                        }
                        results.append(barcode_info)
            
            return results
            
        except Exception as e:
            logger.debug(f"Basic barcode detection error: {e}")
            return []


class LicensePlateDetector:
    """车牌检测器"""
    
    def __init__(self):
        """初始化车牌检测器"""
        self.plate_cascade = self._load_cascade()
    
    def _load_cascade(self):
        """加载车牌检测级联分类器"""
        try:
            # 尝试加载预训练的车牌检测模型
            cascade_path = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
            return cv2.CascadeClassifier(cascade_path)
        except:
            logger.warning("License plate cascade not available, using basic detection")
            return None
    
    def detect_license_plates(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """检测车牌"""
        try:
            results = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if self.plate_cascade and not self.plate_cascade.empty():
                # 使用级联分类器检测
                plates = self.plate_cascade.detectMultiScale(gray, 1.1, 4)
                for (x, y, w, h) in plates:
                    plate_roi = frame[y:y+h, x:x+w]
                    plate_number = self._extract_plate_number(plate_roi)
                    
                    plate_info = {
                        'plate_number': plate_number,
                        'bbox': (x, y, w, h),
                        'confidence': 0.8
                    }
                    results.append(plate_info)
            else:
                # 基础车牌检测
                results = self._detect_plates_basic(frame)
            
            return results
            
        except Exception as e:
            logger.debug(f"License plate detection error: {e}")
            return []
    
    def _detect_plates_basic(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """基础车牌检测"""
        try:
            results = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 车牌特征检测
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 1000 < area < 10000:  # 车牌大小范围
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / float(h)
                    
                    # 车牌长宽比通常在2-5之间
                    if 2.0 < aspect_ratio < 5.0:
                        plate_info = {
                            'plate_number': 'PLATE_DETECTED',
                            'bbox': (x, y, w, h),
                            'confidence': 0.6
                        }
                        results.append(plate_info)
            
            return results
            
        except Exception as e:
            logger.debug(f"Basic plate detection error: {e}")
            return []
    
    def _extract_plate_number(self, plate_roi: np.ndarray) -> str:
        """提取车牌号码"""
        try:
            # 简单的车牌号码提取（实际应用中需要OCR）
            gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
            # 这里可以集成OCR来识别具体的车牌号码
            return "PLATE_NUMBER"
        except:
            return "UNKNOWN"


class TrafficSignDetector:
    """交通标志检测器"""
    
    def __init__(self):
        """初始化交通标志检测器"""
        self.sign_templates = self._load_sign_templates()
    
    def _load_sign_templates(self) -> Dict[str, Any]:
        """加载交通标志模板"""
        return {
            'stop': {'color': 'red', 'shape': 'octagon'},
            'yield': {'color': 'red', 'shape': 'triangle'},
            'speed_limit': {'color': 'white', 'shape': 'circle'},
            'warning': {'color': 'yellow', 'shape': 'triangle'},
            'no_entry': {'color': 'red', 'shape': 'circle'}
        }
    
    def detect_traffic_signs(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """检测交通标志"""
        try:
            results = []
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 检测红色标志（停车标志、禁止标志等）
            red_signs = self._detect_red_signs(frame, hsv)
            results.extend(red_signs)
            
            # 检测黄色标志（警告标志）
            yellow_signs = self._detect_yellow_signs(frame, hsv)
            results.extend(yellow_signs)
            
            # 检测蓝色标志（指示标志）
            blue_signs = self._detect_blue_signs(frame, hsv)
            results.extend(blue_signs)
            
            return results
            
        except Exception as e:
            logger.debug(f"Traffic sign detection error: {e}")
            return []
    
    def _detect_red_signs(self, frame: np.ndarray, hsv: np.ndarray) -> List[Dict[str, Any]]:
        """检测红色交通标志"""
        try:
            results = []
            
            # 红色范围
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = mask1 + mask2
            
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 500 < area < 5000:  # 标志大小范围
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # 分析形状确定标志类型
                    sign_type = self._classify_red_sign(contour)
                    
                    sign_info = {
                        'sign_type': sign_type,
                        'color': 'red',
                        'bbox': (x, y, w, h),
                        'confidence': 0.7
                    }
                    results.append(sign_info)
            
            return results
            
        except Exception as e:
            logger.debug(f"Red sign detection error: {e}")
            return []
    
    def _detect_yellow_signs(self, frame: np.ndarray, hsv: np.ndarray) -> List[Dict[str, Any]]:
        """检测黄色交通标志"""
        try:
            results = []
            
            # 黄色范围
            lower_yellow = np.array([20, 50, 50])
            upper_yellow = np.array([30, 255, 255])
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 500 < area < 5000:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    sign_info = {
                        'sign_type': 'warning',
                        'color': 'yellow',
                        'bbox': (x, y, w, h),
                        'confidence': 0.7
                    }
                    results.append(sign_info)
            
            return results
            
        except Exception as e:
            logger.debug(f"Yellow sign detection error: {e}")
            return []
    
    def _detect_blue_signs(self, frame: np.ndarray, hsv: np.ndarray) -> List[Dict[str, Any]]:
        """检测蓝色交通标志"""
        try:
            results = []
            
            # 蓝色范围
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([130, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 500 < area < 5000:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    sign_info = {
                        'sign_type': 'information',
                        'color': 'blue',
                        'bbox': (x, y, w, h),
                        'confidence': 0.7
                    }
                    results.append(sign_info)
            
            return results
            
        except Exception as e:
            logger.debug(f"Blue sign detection error: {e}")
            return []
    
    def _classify_red_sign(self, contour: np.ndarray) -> str:
        """分类红色标志类型"""
        try:
            # 计算轮廓近似
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            vertices = len(approx)
            
            if vertices == 8:  # 八边形
                return 'stop'
            elif vertices == 3:  # 三角形
                return 'yield'
            else:  # 圆形或其他
                return 'no_entry'
                
        except Exception as e:
            logger.debug(f"Red sign classification error: {e}")
            return 'unknown'


class TrafficLightDetector:
    """交通灯检测器"""
    
    def __init__(self):
        """初始化交通灯检测器"""
        pass
    
    def detect_traffic_lights(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """检测交通灯"""
        try:
            results = []
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 检测红灯
            red_lights = self._detect_red_lights(frame, hsv)
            results.extend(red_lights)
            
            # 检测黄灯
            yellow_lights = self._detect_yellow_lights(frame, hsv)
            results.extend(yellow_lights)
            
            # 检测绿灯
            green_lights = self._detect_green_lights(frame, hsv)
            results.extend(green_lights)
            
            return results
            
        except Exception as e:
            logger.debug(f"Traffic light detection error: {e}")
            return []
    
    def _detect_red_lights(self, frame: np.ndarray, hsv: np.ndarray) -> List[Dict[str, Any]]:
        """检测红灯"""
        try:
            results = []
            
            # 红色范围（更严格的范围用于交通灯）
            lower_red = np.array([0, 100, 100])
            upper_red = np.array([10, 255, 255])
            red_mask = cv2.inRange(hsv, lower_red, upper_red)
            
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < 2000:  # 交通灯大小范围
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # 检查圆形度
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * math.pi * area / (perimeter * perimeter)
                        if circularity > 0.6:  # 相对圆形
                            light_info = {
                                'light_color': 'red',
                                'bbox': (x, y, w, h),
                                'confidence': 0.8
                            }
                            results.append(light_info)
            
            return results
            
        except Exception as e:
            logger.debug(f"Red light detection error: {e}")
            return []
    
    def _detect_yellow_lights(self, frame: np.ndarray, hsv: np.ndarray) -> List[Dict[str, Any]]:
        """检测黄灯"""
        try:
            results = []
            
            lower_yellow = np.array([15, 100, 100])
            upper_yellow = np.array([35, 255, 255])
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < 2000:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * math.pi * area / (perimeter * perimeter)
                        if circularity > 0.6:
                            light_info = {
                                'light_color': 'yellow',
                                'bbox': (x, y, w, h),
                                'confidence': 0.8
                            }
                            results.append(light_info)
            
            return results
            
        except Exception as e:
            logger.debug(f"Yellow light detection error: {e}")
            return []
    
    def _detect_green_lights(self, frame: np.ndarray, hsv: np.ndarray) -> List[Dict[str, Any]]:
        """检测绿灯"""
        try:
            results = []
            
            lower_green = np.array([40, 100, 100])
            upper_green = np.array([80, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < 2000:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * math.pi * area / (perimeter * perimeter)
                        if circularity > 0.6:
                            light_info = {
                                'light_color': 'green',
                                'bbox': (x, y, w, h),
                                'confidence': 0.8
                            }
                            results.append(light_info)
            
            return results
            
        except Exception as e:
            logger.debug(f"Green light detection error: {e}")
            return []


class FacilitySignDetector:
    """公共设施标志检测器"""
    
    def __init__(self):
        """初始化公共设施标志检测器"""
        self.facility_types = {
            'restroom': {'keywords': ['WC', 'TOILET', 'RESTROOM'], 'color': 'blue'},
            'parking': {'keywords': ['P', 'PARKING'], 'color': 'blue'},
            'hospital': {'keywords': ['H', 'HOSPITAL', '+'], 'color': 'red'},
            'exit': {'keywords': ['EXIT', 'EMERGENCY'], 'color': 'green'},
            'elevator': {'keywords': ['ELEVATOR', 'LIFT'], 'color': 'blue'},
            'stairs': {'keywords': ['STAIRS'], 'color': 'blue'}
        }
    
    def detect_facility_signs(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """检测公共设施标志"""
        try:
            results = []
            
            # 检测蓝色设施标志
            blue_facilities = self._detect_blue_facilities(frame)
            results.extend(blue_facilities)
            
            # 检测绿色设施标志
            green_facilities = self._detect_green_facilities(frame)
            results.extend(green_facilities)
            
            # 检测红色设施标志
            red_facilities = self._detect_red_facilities(frame)
            results.extend(red_facilities)
            
            return results
            
        except Exception as e:
            logger.debug(f"Facility sign detection error: {e}")
            return []
    
    def _detect_blue_facilities(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """检测蓝色设施标志"""
        try:
            results = []
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 蓝色范围
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([130, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 200 < area < 3000:  # 设施标志大小范围
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # 检查长宽比（设施标志通常是正方形或矩形）
                    aspect_ratio = w / float(h)
                    if 0.5 < aspect_ratio < 2.0:
                        facility_info = {
                            'facility_type': 'information',
                            'color': 'blue',
                            'bbox': (x, y, w, h),
                            'confidence': 0.6
                        }
                        results.append(facility_info)
            
            return results
            
        except Exception as e:
            logger.debug(f"Blue facility detection error: {e}")
            return []
    
    def _detect_green_facilities(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """检测绿色设施标志（如安全出口）"""
        try:
            results = []
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 绿色范围
            lower_green = np.array([40, 50, 50])
            upper_green = np.array([80, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 200 < area < 3000:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    aspect_ratio = w / float(h)
                    if 0.5 < aspect_ratio < 3.0:  # 出口标志可能是长方形
                        facility_info = {
                            'facility_type': 'exit',
                            'color': 'green',
                            'bbox': (x, y, w, h),
                            'confidence': 0.7
                        }
                        results.append(facility_info)
            
            return results
            
        except Exception as e:
            logger.debug(f"Green facility detection error: {e}")
            return []
    
    def _detect_red_facilities(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """检测红色设施标志（如医院标志）"""
        try:
            results = []
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 红色范围
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = mask1 + mask2
            
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 200 < area < 3000:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    aspect_ratio = w / float(h)
                    if 0.7 < aspect_ratio < 1.3:  # 医院标志通常是正方形
                        facility_info = {
                            'facility_type': 'medical',
                            'color': 'red',
                            'bbox': (x, y, w, h),
                            'confidence': 0.7
                        }
                        results.append(facility_info)
            
            return results
            
        except Exception as e:
            logger.debug(f"Red facility detection error: {e}")
            return []


# 保留原有的形状检测器和文字检测器
class ShapeDetector:
    """形状检测器"""
    
    def detect_shape(self, contour: np.ndarray) -> Dict[str, Any]:
        """检测形状"""
        try:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            vertices = len(approx)
            
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter > 0:
                circularity = 4 * math.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
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
        if circularity > 0.7:
            return 'circle'
        
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
        elif vertices == 8:
            return 'octagon'
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
            
            h_projection = np.sum(gray < 128, axis=0)
            v_projection = np.sum(gray < 128, axis=1)
            
            h_variance = np.var(h_projection)
            v_variance = np.var(v_projection)
            
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