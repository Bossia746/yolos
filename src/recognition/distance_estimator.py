"""距离估算器 - 基于相似三角形原理的摄像头测距功能

使用相似三角形方法计算照片中目标物体到相机的距离。
支持已知物体尺寸的距离测量和相机焦距标定。
"""

import cv2
import numpy as np
import imutils
from typing import Tuple, Optional, Dict, Any, List
import json
import os
from pathlib import Path


class DistanceEstimator:
    """距离估算器类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化距离估算器
        
        Args:
            config_path: 配置文件路径，包含已标定的焦距信息
        """
        self.focal_length = None
        self.config_path = config_path or "distance_config.json"
        self.calibration_data = {}
        
        # 默认检测参数
        self.blur_kernel = (5, 5)
        self.canny_low = 35
        self.canny_high = 125
        self.min_contour_area = 1000
        
        # 加载配置
        self._load_config()
    
    def _load_config(self):
        """加载配置文件"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.focal_length = config.get('focal_length')
                    self.calibration_data = config.get('calibration_data', {})
                    print(f"已加载配置: 焦距={self.focal_length}")
            except Exception as e:
                print(f"加载配置文件失败: {e}")
    
    def _save_config(self):
        """保存配置文件"""
        config = {
            'focal_length': self.focal_length,
            'calibration_data': self.calibration_data
        }
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"配置已保存到: {self.config_path}")
        except Exception as e:
            print(f"保存配置文件失败: {e}")
    
    def find_marker(self, image: np.ndarray, target_type: str = 'largest') -> Optional[Tuple]:
        """
        在图像中找到目标物体的轮廓
        
        Args:
            image: 输入图像
            target_type: 目标类型 ('largest', 'rectangular', 'circular')
            
        Returns:
            目标物体的最小外接矩形信息 (center, (width, height), angle)
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊去噪
        gray = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        
        # 边缘检测
        edged = cv2.Canny(gray, self.canny_low, self.canny_high)
        
        # 查找轮廓
        contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        
        if not contours:
            return None
        
        # 根据目标类型选择轮廓
        if target_type == 'largest':
            # 选择面积最大的轮廓
            target_contour = max(contours, key=cv2.contourArea)
        elif target_type == 'rectangular':
            # 选择最接近矩形的轮廓
            target_contour = self._find_rectangular_contour(contours)
        elif target_type == 'circular':
            # 选择最接近圆形的轮廓
            target_contour = self._find_circular_contour(contours)
        else:
            target_contour = max(contours, key=cv2.contourArea)
        
        if target_contour is None or cv2.contourArea(target_contour) < self.min_contour_area:
            return None
        
        # 计算最小外接矩形
        return cv2.minAreaRect(target_contour)
    
    def _find_rectangular_contour(self, contours: List) -> Optional[np.ndarray]:
        """找到最接近矩形的轮廓"""
        best_contour = None
        best_score = float('inf')
        
        for contour in contours:
            if cv2.contourArea(contour) < self.min_contour_area:
                continue
            
            # 轮廓近似
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 计算矩形度（4个顶点的轮廓更接近矩形）
            if len(approx) >= 4:
                # 计算轮廓面积与外接矩形面积的比值
                rect_area = cv2.contourArea(cv2.boxPoints(cv2.minAreaRect(contour)))
                contour_area = cv2.contourArea(contour)
                rectangularity = abs(1 - contour_area / rect_area)
                
                if rectangularity < best_score:
                    best_score = rectangularity
                    best_contour = contour
        
        return best_contour
    
    def _find_circular_contour(self, contours: List) -> Optional[np.ndarray]:
        """找到最接近圆形的轮廓"""
        best_contour = None
        best_score = float('inf')
        
        for contour in contours:
            if cv2.contourArea(contour) < self.min_contour_area:
                continue
            
            # 计算圆形度
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                circularity_score = abs(1 - circularity)
                
                if circularity_score < best_score:
                    best_score = circularity_score
                    best_contour = contour
        
        return best_contour
    
    def calibrate_camera(self, image: np.ndarray, known_width: float, known_distance: float, 
                        target_type: str = 'largest') -> bool:
        """
        标定相机焦距
        
        Args:
            image: 标定图像
            known_width: 已知物体的实际宽度（单位：厘米或英寸）
            known_distance: 已知物体到相机的距离（单位：厘米或英寸）
            target_type: 目标类型
            
        Returns:
            标定是否成功
        """
        marker = self.find_marker(image, target_type)
        if marker is None:
            print("标定失败：未找到目标物体")
            return False
        
        # 获取物体在图像中的像素宽度
        pixel_width = marker[1][0]  # 最小外接矩形的宽度
        
        # 计算焦距：F = (P × D) / W
        self.focal_length = (pixel_width * known_distance) / known_width
        
        # 保存标定数据
        self.calibration_data = {
            'known_width': known_width,
            'known_distance': known_distance,
            'pixel_width': float(pixel_width),
            'target_type': target_type
        }
        
        self._save_config()
        
        print(f"相机标定完成:")
        print(f"  已知宽度: {known_width}")
        print(f"  已知距离: {known_distance}")
        print(f"  像素宽度: {pixel_width:.2f}")
        print(f"  计算焦距: {self.focal_length:.2f}")
        
        return True
    
    def estimate_distance(self, image: np.ndarray, known_width: float, 
                         target_type: str = 'largest') -> Optional[Dict[str, Any]]:
        """
        估算目标物体到相机的距离
        
        Args:
            image: 输入图像
            known_width: 已知物体的实际宽度
            target_type: 目标类型
            
        Returns:
            包含距离信息的字典，如果失败则返回None
        """
        if self.focal_length is None:
            print("错误：相机未标定，请先调用calibrate_camera()")
            return None
        
        marker = self.find_marker(image, target_type)
        if marker is None:
            print("未找到目标物体")
            return None
        
        # 获取物体在图像中的像素宽度
        pixel_width = marker[1][0]
        
        # 计算距离：D = (W × F) / P
        distance = (known_width * self.focal_length) / pixel_width
        
        # 获取边界框信息
        box = cv2.boxPoints(marker)
        box = np.int0(box)
        
        result = {
            'distance': float(distance),
            'pixel_width': float(pixel_width),
            'known_width': known_width,
            'focal_length': self.focal_length,
            'bounding_box': box.tolist(),
            'marker_info': {
                'center': marker[0],
                'size': marker[1],
                'angle': marker[2]
            }
        }
        
        return result
    
    def distance_to_camera(self, known_width: float, focal_length: float, pixel_width: float) -> float:
        """
        使用相似三角形公式计算距离
        
        Args:
            known_width: 已知物体实际宽度
            focal_length: 相机焦距
            pixel_width: 物体在图像中的像素宽度
            
        Returns:
            计算得到的距离
        """
        return (known_width * focal_length) / pixel_width
    
    def draw_results(self, image: np.ndarray, result: Dict[str, Any], 
                    unit: str = "cm") -> np.ndarray:
        """
        在图像上绘制距离测量结果
        
        Args:
            image: 输入图像
            result: 距离测量结果
            unit: 距离单位
            
        Returns:
            标注后的图像
        """
        annotated_image = image.copy()
        
        # 绘制边界框
        box = np.array(result['bounding_box'], dtype=np.int32)
        cv2.drawContours(annotated_image, [box], -1, (0, 255, 0), 2)
        
        # 显示距离信息
        distance = result['distance']
        distance_text = f"{distance:.1f}{unit}"
        
        # 在图像上显示距离
        text_position = (annotated_image.shape[1] - 200, annotated_image.shape[0] - 20)
        cv2.putText(annotated_image, distance_text, text_position, 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
        
        # 显示物体中心点
        center = tuple(map(int, result['marker_info']['center']))
        cv2.circle(annotated_image, center, 5, (255, 0, 0), -1)
        
        # 显示详细信息
        info_text = f"Width: {result['pixel_width']:.1f}px"
        cv2.putText(annotated_image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_image
    
    def set_detection_params(self, blur_kernel: Tuple[int, int] = (5, 5),
                           canny_low: int = 35, canny_high: int = 125,
                           min_contour_area: int = 1000):
        """
        设置检测参数
        
        Args:
            blur_kernel: 高斯模糊核大小
            canny_low: Canny边缘检测低阈值
            canny_high: Canny边缘检测高阈值
            min_contour_area: 最小轮廓面积
        """
        self.blur_kernel = blur_kernel
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_contour_area = min_contour_area
    
    def get_calibration_info(self) -> Dict[str, Any]:
        """获取标定信息"""
        return {
            'focal_length': self.focal_length,
            'calibration_data': self.calibration_data,
            'is_calibrated': self.focal_length is not None
        }
    
    def reset_calibration(self):
        """重置标定信息"""
        self.focal_length = None
        self.calibration_data = {}
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        print("标定信息已重置")


class RealTimeDistanceEstimator:
    """实时距离估算器"""
    
    def __init__(self, distance_estimator: DistanceEstimator):
        """
        初始化实时距离估算器
        
        Args:
            distance_estimator: 距离估算器实例
        """
        self.estimator = distance_estimator
        self.is_running = False
    
    def start_realtime_estimation(self, camera_id: int = 0, known_width: float = 11.0,
                                 target_type: str = 'largest', unit: str = "cm"):
        """
        开始实时距离估算
        
        Args:
            camera_id: 摄像头ID
            known_width: 已知物体宽度
            target_type: 目标类型
            unit: 距离单位
        """
        if self.estimator.focal_length is None:
            print("错误：相机未标定，请先进行标定")
            return
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"无法打开摄像头 {camera_id}")
            return
        
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_running = True
        print(f"开始实时距离估算，按 'q' 退出，按 'c' 重新标定")
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 估算距离
                result = self.estimator.estimate_distance(frame, known_width, target_type)
                
                if result:
                    # 绘制结果
                    display_frame = self.estimator.draw_results(frame, result, unit)
                else:
                    display_frame = frame.copy()
                    cv2.putText(display_frame, "No target found", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow('Distance Estimation', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    print("请将已知物体放在指定距离处，然后按任意键进行标定...")
                    cv2.waitKey(0)
                    # 这里可以添加标定逻辑
                    
        finally:
            self.stop()
            cap.release()
            cv2.destroyAllWindows()
    
    def stop(self):
        """停止实时估算"""
        self.is_running = False


if __name__ == "__main__":
    # 使用示例
    estimator = DistanceEstimator()
    
    # 如果有标定图像，可以进行标定
    # calibration_image = cv2.imread("calibration.jpg")
    # estimator.calibrate_camera(calibration_image, known_width=11.0, known_distance=24.0)
    
    # 实时距离估算
    realtime_estimator = RealTimeDistanceEstimator(estimator)
    # realtime_estimator.start_realtime_estimation(known_width=11.0)