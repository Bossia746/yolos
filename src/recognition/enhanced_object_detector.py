"""增强的目标检测器 - 专为距离测量优化的物体检测功能

提供多种目标检测方法，包括颜色检测、形状检测、边缘检测等，
专门优化用于距离测量场景中的目标物体识别。
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import math


class EnhancedObjectDetector:
    """增强的目标检测器"""
    
    def __init__(self):
        """
        初始化增强目标检测器
        """
        # 颜色检测参数
        self.color_ranges = {
            'red': [(0, 50, 50), (10, 255, 255), (170, 50, 50), (180, 255, 255)],
            'green': [(40, 50, 50), (80, 255, 255)],
            'blue': [(100, 50, 50), (130, 255, 255)],
            'yellow': [(20, 50, 50), (30, 255, 255)],
            'white': [(0, 0, 200), (180, 30, 255)],
            'black': [(0, 0, 0), (180, 255, 50)]
        }
        
        # 检测参数
        self.min_contour_area = 1000
        self.max_contour_area = 50000
        self.blur_kernel_size = 5
        self.canny_low = 50
        self.canny_high = 150
        self.morphology_kernel_size = 5
        
        # 形状检测参数
        self.shape_epsilon_factor = 0.02
        self.circularity_threshold = 0.7
        self.rectangularity_threshold = 0.8
    
    def detect_by_color(self, image: np.ndarray, target_color: str) -> List[Dict[str, Any]]:
        """
        基于颜色检测目标物体
        
        Args:
            image: 输入图像
            target_color: 目标颜色名称
            
        Returns:
            检测到的目标物体列表
        """
        if target_color not in self.color_ranges:
            raise ValueError(f"不支持的颜色: {target_color}")
        
        # 转换到HSV色彩空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 创建颜色掩码
        color_range = self.color_ranges[target_color]
        if len(color_range) == 4:  # 红色有两个范围
            mask1 = cv2.inRange(hsv, color_range[0], color_range[1])
            mask2 = cv2.inRange(hsv, color_range[2], color_range[3])
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, color_range[0], color_range[1])
        
        # 形态学操作去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (self.morphology_kernel_size, self.morphology_kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area <= area <= self.max_contour_area:
                detection = self._analyze_contour(contour, image)
                detection['detection_method'] = 'color'
                detection['target_color'] = target_color
                detection['area'] = area
                detections.append(detection)
        
        return sorted(detections, key=lambda x: x['area'], reverse=True)
    
    def detect_by_edge(self, image: np.ndarray, target_shape: str = 'any') -> List[Dict[str, Any]]:
        """
        基于边缘检测目标物体
        
        Args:
            image: 输入图像
            target_shape: 目标形状 ('any', 'rectangle', 'circle', 'triangle')
            
        Returns:
            检测到的目标物体列表
        """
        # 预处理
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)
        
        # 边缘检测
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # 形态学操作连接断开的边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area <= area <= self.max_contour_area:
                detection = self._analyze_contour(contour, image)
                
                # 形状过滤
                if target_shape != 'any':
                    shape_type = detection['shape_info']['type']
                    if shape_type != target_shape:
                        continue
                
                detection['detection_method'] = 'edge'
                detection['target_shape'] = target_shape
                detection['area'] = area
                detections.append(detection)
        
        return sorted(detections, key=lambda x: x['area'], reverse=True)
    
    def detect_by_template(self, image: np.ndarray, template: np.ndarray, 
                          threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        基于模板匹配检测目标物体
        
        Args:
            image: 输入图像
            template: 模板图像
            threshold: 匹配阈值
            
        Returns:
            检测到的目标物体列表
        """
        # 转换为灰度图
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        # 模板匹配
        result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)
        
        # 查找匹配位置
        locations = np.where(result >= threshold)
        
        detections = []
        h, w = gray_template.shape
        
        for pt in zip(*locations[::-1]):
            # 创建边界框
            x, y = pt
            bbox = (x, y, w, h)
            
            # 计算中心点
            center = (x + w // 2, y + h // 2)
            
            detection = {
                'bbox': bbox,
                'center': center,
                'width': w,
                'height': h,
                'area': w * h,
                'confidence': float(result[y, x]),
                'detection_method': 'template',
                'shape_info': {
                    'type': 'template_match',
                    'confidence': float(result[y, x])
                }
            }
            detections.append(detection)
        
        return sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    def detect_largest_object(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        检测图像中最大的物体
        
        Args:
            image: 输入图像
            
        Returns:
            最大物体的检测结果
        """
        edge_detections = self.detect_by_edge(image)
        if edge_detections:
            return edge_detections[0]  # 已按面积排序
        return None
    
    def detect_paper_like_object(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        检测类似纸张的矩形物体（专为距离测量优化）
        
        Args:
            image: 输入图像
            
        Returns:
            纸张类物体的检测结果
        """
        # 使用边缘检测寻找矩形物体
        detections = self.detect_by_edge(image, 'rectangle')
        
        # 进一步筛选：寻找长宽比接近A4纸的物体
        paper_detections = []
        for detection in detections:
            width = detection['width']
            height = detection['height']
            aspect_ratio = max(width, height) / min(width, height)
            
            # A4纸的长宽比约为1.414 (√2)
            if 1.2 <= aspect_ratio <= 1.8:  # 允许一定误差
                detection['paper_likelihood'] = 1.0 / abs(aspect_ratio - 1.414)
                paper_detections.append(detection)
        
        if paper_detections:
            # 按纸张相似度排序
            paper_detections.sort(key=lambda x: x['paper_likelihood'], reverse=True)
            return paper_detections[0]
        
        return None
    
    def _analyze_contour(self, contour: np.ndarray, image: np.ndarray) -> Dict[str, Any]:
        """
        分析轮廓信息
        
        Args:
            contour: 轮廓
            image: 原始图像
            
        Returns:
            轮廓分析结果
        """
        # 基本几何信息
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # 边界框
        x, y, w, h = cv2.boundingRect(contour)
        bbox = (x, y, w, h)
        center = (x + w // 2, y + h // 2)
        
        # 最小外接矩形
        min_rect = cv2.minAreaRect(contour)
        min_rect_area = min_rect[1][0] * min_rect[1][1]
        
        # 最小外接圆
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        circle_area = np.pi * radius * radius
        
        # 形状分析
        shape_info = self._analyze_shape(contour, area, perimeter)
        
        # 颜色分析（在边界框区域）
        roi = image[y:y+h, x:x+w]
        dominant_color = self._get_dominant_color(roi)
        
        return {
            'contour': contour,
            'bbox': bbox,
            'center': center,
            'width': w,
            'height': h,
            'area': area,
            'perimeter': perimeter,
            'min_rect': min_rect,
            'min_rect_area': min_rect_area,
            'circle_center': (cx, cy),
            'circle_radius': radius,
            'circle_area': circle_area,
            'shape_info': shape_info,
            'dominant_color': dominant_color,
            'aspect_ratio': max(w, h) / min(w, h) if min(w, h) > 0 else 0
        }
    
    def _analyze_shape(self, contour: np.ndarray, area: float, perimeter: float) -> Dict[str, Any]:
        """
        分析轮廓形状
        
        Args:
            contour: 轮廓
            area: 面积
            perimeter: 周长
            
        Returns:
            形状分析结果
        """
        # 轮廓近似
        epsilon = self.shape_epsilon_factor * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)
        
        # 圆形度 (4π*面积/周长²)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # 矩形度 (轮廓面积/最小外接矩形面积)
        min_rect = cv2.minAreaRect(contour)
        rect_area = min_rect[1][0] * min_rect[1][1]
        rectangularity = area / rect_area if rect_area > 0 else 0
        
        # 凸性
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # 形状分类
        shape_type = 'unknown'
        confidence = 0.0
        
        if circularity > self.circularity_threshold:
            shape_type = 'circle'
            confidence = circularity
        elif rectangularity > self.rectangularity_threshold and vertices <= 6:
            shape_type = 'rectangle'
            confidence = rectangularity
        elif vertices == 3:
            shape_type = 'triangle'
            confidence = solidity
        elif vertices >= 5 and circularity > 0.5:
            shape_type = 'polygon'
            confidence = circularity
        
        return {
            'type': shape_type,
            'confidence': confidence,
            'vertices': vertices,
            'circularity': circularity,
            'rectangularity': rectangularity,
            'solidity': solidity,
            'approx_contour': approx
        }
    
    def _get_dominant_color(self, roi: np.ndarray) -> Tuple[int, int, int]:
        """
        获取区域的主导颜色
        
        Args:
            roi: 感兴趣区域
            
        Returns:
            主导颜色的BGR值
        """
        if roi.size == 0:
            return (0, 0, 0)
        
        # 重塑为像素列表
        pixels = roi.reshape(-1, 3)
        
        # 使用K-means聚类找到主导颜色
        from sklearn.cluster import KMeans
        
        try:
            kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)
            kmeans.fit(pixels)
            dominant_color = kmeans.cluster_centers_[0]
            return tuple(map(int, dominant_color))
        except:
            # 如果K-means失败，使用平均值
            return tuple(map(int, np.mean(pixels, axis=0)))
    
    def set_detection_params(self, min_area: int = None, max_area: int = None,
                           blur_kernel: int = None, canny_low: int = None,
                           canny_high: int = None):
        """
        设置检测参数
        
        Args:
            min_area: 最小轮廓面积
            max_area: 最大轮廓面积
            blur_kernel: 模糊核大小
            canny_low: Canny低阈值
            canny_high: Canny高阈值
        """
        if min_area is not None:
            self.min_contour_area = min_area
        if max_area is not None:
            self.max_contour_area = max_area
        if blur_kernel is not None:
            self.blur_kernel_size = blur_kernel
        if canny_low is not None:
            self.canny_low = canny_low
        if canny_high is not None:
            self.canny_high = canny_high
    
    def visualize_detection(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image: 原始图像
            detections: 检测结果列表
            
        Returns:
            标注后的图像
        """
        result_image = image.copy()
        
        for i, detection in enumerate(detections):
            # 绘制边界框
            bbox = detection['bbox']
            x, y, w, h = bbox
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 绘制中心点
            center = detection['center']
            cv2.circle(result_image, center, 5, (255, 0, 0), -1)
            
            # 显示信息
            info_text = f"{i+1}: {detection.get('shape_info', {}).get('type', 'unknown')}"
            cv2.putText(result_image, info_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 显示面积
            area_text = f"Area: {detection['area']:.0f}"
            cv2.putText(result_image, area_text, (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_image


if __name__ == "__main__":
    # 使用示例
    detector = EnhancedObjectDetector()
    
    # 从摄像头检测
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测最大物体
        detection = detector.detect_largest_object(frame)
        
        if detection:
            # 可视化结果
            result_frame = detector.visualize_detection(frame, [detection])
        else:
            result_frame = frame
        
        cv2.imshow('Enhanced Object Detection', result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()