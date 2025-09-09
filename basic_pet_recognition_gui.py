#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础宠物识别GUI - 使用OpenCV级联分类器和颜色检测
不依赖YOLO，适用于网络受限环境
"""

import cv2
import numpy as np
import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import sys
import os

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pet_recognition.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BasicPetRecognizer:
    """基础宠物识别器 - 使用传统计算机视觉方法"""
    
    def __init__(self):
        # 尝试加载Haar级联分类器
        self.face_cascade = None
        self.body_cascade = None
        
        # 加载级联分类器
        self.load_cascades()
        
        # 颜色范围定义 (HSV)
        self.color_ranges = {
            'brown': [(10, 50, 20), (20, 255, 200)],    # 棕色 (狗、猫常见)
            'black': [(0, 0, 0), (180, 255, 30)],       # 黑色
            'white': [(0, 0, 200), (180, 30, 255)],     # 白色
            'gray': [(0, 0, 50), (180, 30, 200)],       # 灰色
            'orange': [(5, 50, 50), (15, 255, 255)],    # 橙色 (橘猫)
            'yellow': [(20, 50, 50), (30, 255, 255)]    # 黄色 (金毛等)
        }
        
        # 运动检测
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.motion_threshold = 500  # 运动区域最小面积
        
        # 检测历史
        self.detection_history = []
        self.max_history = 10
    
    def load_cascades(self):
        """加载Haar级联分类器"""
        try:
            # 尝试加载人脸检测器 (可能检测到宠物脸部)
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(face_cascade_path):
                self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
                logger.info("Face cascade loaded successfully")
            
            # 尝试加载全身检测器
            body_cascade_path = cv2.data.haarcascades + 'haarcascade_fullbody.xml'
            if os.path.exists(body_cascade_path):
                self.body_cascade = cv2.CascadeClassifier(body_cascade_path)
                logger.info("Body cascade loaded successfully")
                
        except Exception as e:
            logger.warning(f"Failed to load cascades: {e}")
    
    def detect_by_color(self, frame: np.ndarray) -> List[Dict]:
        """基于颜色的检测"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detections = []
        
        for color_name, (lower, upper) in self.color_ranges.items():
            # 创建颜色掩码
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            
            # 形态学操作清理掩码
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # 最小面积阈值
                    # 获取边界框
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # 计算形状特征
                    aspect_ratio = w / h
                    extent = area / (w * h)
                    
                    # 基于形状特征推断可能的宠物类型
                    pet_type = self.classify_by_shape(aspect_ratio, extent, area)
                    
                    detection = {
                        'type': 'color_based',
                        'species': pet_type,
                        'color': color_name,
                        'confidence': min(0.8, extent + 0.2),  # 基于形状匹配度的置信度
                        'bbox': (x, y, w, h),
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'extent': extent
                    }
                    
                    detections.append(detection)
        
        return detections
    
    def detect_by_motion(self, frame: np.ndarray) -> List[Dict]:
        """基于运动的检测"""
        # 背景减除
        fg_mask = self.bg_subtractor.apply(frame)
        
        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # 查找运动区域
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.motion_threshold:
                x, y, w, h = cv2.boundingRect(contour)
                
                # 分析运动模式
                motion_type = self.analyze_motion_pattern(x, y, w, h, area)
                
                detection = {
                    'type': 'motion_based',
                    'species': 'moving_object',
                    'motion_pattern': motion_type,
                    'confidence': min(0.7, area / 5000),  # 基于运动区域大小
                    'bbox': (x, y, w, h),
                    'area': area
                }
                
                detections.append(detection)
        
        return detections
    
    def detect_by_cascade(self, frame: np.ndarray) -> List[Dict]:
        """使用级联分类器检测"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = []
        
        # 人脸检测 (可能检测到宠物脸部)
        if self.face_cascade is not None:
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                detection = {
                    'type': 'cascade_face',
                    'species': 'pet_face',
                    'confidence': 0.6,
                    'bbox': (x, y, w, h),
                    'area': w * h
                }
                detections.append(detection)
        
        # 全身检测
        if self.body_cascade is not None:
            bodies = self.body_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in bodies:
                detection = {
                    'type': 'cascade_body',
                    'species': 'pet_body',
                    'confidence': 0.7,
                    'bbox': (x, y, w, h),
                    'area': w * h
                }
                detections.append(detection)
        
        return detections
    
    def classify_by_shape(self, aspect_ratio: float, extent: float, area: int) -> str:
        """基于形状特征分类宠物类型"""
        # 简单的形状分类规则
        if aspect_ratio > 1.5:  # 长条形
            if area > 5000:
                return "dog_lying"
            else:
                return "cat_lying"
        elif aspect_ratio < 0.7:  # 高瘦形
            if area > 3000:
                return "dog_sitting"
            else:
                return "cat_sitting"
        else:  # 接近正方形
            if extent > 0.7:  # 填充度高
                if area > 4000:
                    return "dog"
                else:
                    return "cat"
            else:
                return "bird_or_small_pet"
    
    def analyze_motion_pattern(self, x: int, y: int, w: int, h: int, area: int) -> str:
        """分析运动模式"""
        aspect_ratio = w / h
        
        if aspect_ratio > 2.0:
            return "horizontal_movement"  # 水平移动 (可能是跑动)
        elif aspect_ratio < 0.5:
            return "vertical_movement"    # 垂直移动 (可能是跳跃)
        elif area > 3000:
            return "large_movement"       # 大范围移动
        else:
            return "small_movement"       # 小范围移动
    
    def detect_pets(self, frame: np.ndarray) -> List[Dict]:
        """综合检测方法"""
        all_detections = []
        
        # 颜色检测
        color_detections = self.detect_by_color(frame)
        all_detections.extend(color_detections)
        
        # 运动检测
        motion_detections = self.detect_by_motion(frame)
        all_detections.extend(motion_detections)
        
        # 级联分类器检测
        cascade_detections = self.detect_by_cascade(frame)
        all_detections.extend(cascade_detections)
        
        # 去重和融合
        filtered_detections = self.filter_and_merge_detections(all_detections)
        
        # 更新检测历史
        self.update_detection_history(filtered_detections)
        
        return filtered_detections
    
    def filter_and_merge_detections(self, detections: List[Dict]) -> List[Dict]:
        """过滤和合并重叠的检测"""
        if not detections:
            return []
        
        # 按置信度排序
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        filtered = []
        for detection in detections:
            # 检查是否与已有检测重叠
            is_duplicate = False
            for existing in filtered:
                if self.calculate_iou(detection['bbox'], existing['bbox']) > 0.3:
                    is_duplicate = True
                    # 如果新检测置信度更高，替换
                    if detection['confidence'] > existing['confidence']:
                        filtered.remove(existing)
                        filtered.append(detection)
                    break
            
            if not is_duplicate:
                filtered.append(detection)
        
        return filtered
    
    def calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """计算两个边界框的IoU"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # 计算交集
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def update_detection_history(self, detections: List[Dict]):
        """更新检测历史"""
        self.detection_history.append(len(detections))
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)


class BasicPetRecognitionGUI:
    """基础宠物识别GUI"""
    
    def __init__(self):
        # 摄像头相关
        self.cap = None
        self.frame_width = 640
        self.frame_height = 480
        
        # 识别器
        self.pet_recognizer = BasicPetRecognizer()
        
        # 性能统计
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        
        # 检测控制
        self.detection_interval = 5  # 每5帧检测一次
        self.display_interval = 2    # 每2帧更新显示
        self.result_hold_frames = 20 # 结果保持20帧
        
        # 结果缓存
        self.cached_results = []
        self.result_cache_time = 0
        
        # 统计信息
        self.stats = {
            'total_detections': 0,
            'detection_types': {},
            'session_start': time.time()
        }
        
        self.initialize_camera()
    
    def initialize_camera(self) -> bool:
        """初始化摄像头 - 优先使用内置摄像头"""
        # 首先尝试内置摄像头 (index 0)
        logger.info("Trying builtin camera (index 0)")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                self.cap = cap
                logger.info(f"Builtin camera OK: {frame.shape}")
            else:
                cap.release()
        
        # 如果内置摄像头失败，尝试外部摄像头
        if self.cap is None:
            for index in [1, 2, 3, 4]:
                logger.info(f"Trying camera index {index}")
                cap = cv2.VideoCapture(index)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.cap = cap
                        logger.info(f"Camera {index} OK: {frame.shape}")
                        break
                    else:
                        cap.release()
                else:
                    cap.release()
        
        if self.cap is None:
            logger.error("No camera available")
            return False
        
        # 设置摄像头参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 获取实际尺寸
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Camera setup complete: {self.frame_width}x{self.frame_height}")
        return True
    
    def draw_info_panel(self, frame: np.ndarray, results: List[Dict]) -> np.ndarray:
        """绘制信息面板"""
        # 左上角信息面板
        panel_width = 200
        panel_height = 80
        
        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 统计信息
        detection_types = set(r['type'] for r in results)
        avg_detections = np.mean(self.pet_recognizer.detection_history) if self.pet_recognizer.detection_history else 0
        
        # 显示信息
        font_scale = 0.4
        thickness = 1
        y_offset = 18
        
        cv2.putText(frame, f"Detections: {len(results)}", 
                   (8, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        cv2.putText(frame, f"Methods: {len(detection_types)}", 
                   (8, y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        cv2.putText(frame, f"Avg: {avg_detections:.1f}", 
                   (8, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        cv2.putText(frame, f"FPS: {self.fps:.1f} | Frame: {self.frame_count}", 
                   (8, y_offset + 45), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        return frame
    
    def draw_legend(self, frame: np.ndarray) -> np.ndarray:
        """绘制图例"""
        # 底部图例
        legend_height = 30
        legend_y = self.frame_height - legend_height
        
        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, legend_y), (self.frame_width, self.frame_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # 图例文本
        legend_text = "Detection: Color🎨 Motion🏃 Shape📐 | Colors: Brown🤎 Black⚫ White⚪ Gray🔘"
        font_scale = 0.35
        thickness = 1
        
        cv2.putText(frame, legend_text, (5, legend_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        return frame
    
    def draw_detections(self, frame: np.ndarray, results: List[Dict]) -> np.ndarray:
        """绘制检测结果"""
        for detection in results:
            detection_type = detection['type']
            species = detection['species']
            confidence = detection['confidence']
            x, y, w, h = detection['bbox']
            
            # 根据检测类型选择颜色
            if detection_type == 'color_based':
                color = (0, 255, 255)  # 黄色
            elif detection_type == 'motion_based':
                color = (255, 0, 255)  # 洋红
            elif detection_type.startswith('cascade'):
                color = (0, 255, 0)    # 绿色
            else:
                color = (128, 128, 128)  # 灰色
            
            # 绘制边界框
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # 绘制标签
            label_parts = [species, f"{confidence:.2f}"]
            
            if 'color' in detection:
                label_parts.append(detection['color'])
            
            if 'motion_pattern' in detection:
                label_parts.append(detection['motion_pattern'])
            
            label = " ".join(label_parts)
            
            # 标签背景
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(frame, (x, y - label_h - 8), (x + label_w + 4, y), color, -1)
            cv2.putText(frame, label, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 绘制检测类型标识
            type_text = detection_type.replace('_', ' ').title()
            cv2.putText(frame, type_text, (x, y + h + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # 绘制中心点
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(frame, (center_x, center_y), 3, color, -1)
        
        return frame
    
    def log_detections(self, results: List[Dict]):
        """记录检测结果"""
        for detection in results:
            detection_type = detection['type']
            species = detection['species']
            confidence = detection['confidence']
            
            log_msg = f"[PET DETECTED] Type: {detection_type}, Species: {species}, Confidence: {confidence:.3f}"
            
            if 'color' in detection:
                log_msg += f", Color: {detection['color']}"
            
            if 'motion_pattern' in detection:
                log_msg += f", Motion: {detection['motion_pattern']}"
            
            logger.info(log_msg)
            print(f"🐾 {log_msg}")
            
            # 更新统计
            if detection_type not in self.stats['detection_types']:
                self.stats['detection_types'][detection_type] = 0
            self.stats['detection_types'][detection_type] += 1
            self.stats['total_detections'] += 1
    
    def run(self):
        """运行主循环"""
        if not self.cap:
            logger.error("Camera not initialized")
            return
        
        logger.info("Starting basic pet recognition GUI...")
        logger.info("Detection methods: Color analysis, Motion detection, Shape recognition")
        logger.info("Controls: 'q' or ESC to quit, 's' to save screenshot, 'r' to reset stats")
        
        # 稳定化等待
        logger.info("Camera stabilizing... (30 frames)")
        for i in range(30):
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to read frame during stabilization")
                return
            cv2.imshow('Pet Recognition - Stabilizing...', frame)
            cv2.waitKey(1)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break
                
                self.frame_count += 1
                current_time = time.time()
                
                # 计算FPS
                if self.frame_count % 30 == 0:
                    elapsed = current_time - self.start_time
                    self.fps = self.frame_count / elapsed if elapsed > 0 else 0
                
                # 检测控制
                should_detect = (self.frame_count % self.detection_interval == 0)
                should_update_display = (self.frame_count % self.display_interval == 0)
                
                # 执行检测
                if should_detect:
                    results = self.pet_recognizer.detect_pets(frame)
                    
                    if results:
                        self.cached_results = results
                        self.result_cache_time = self.frame_count
                        self.log_detections(results)
                
                # 使用缓存结果
                display_results = []
                if self.cached_results and (self.frame_count - self.result_cache_time) < self.result_hold_frames:
                    display_results = self.cached_results
                
                # 绘制结果
                if display_results and should_update_display:
                    frame = self.draw_detections(frame, display_results)
                
                # 绘制界面元素
                frame = self.draw_info_panel(frame, display_results)
                frame = self.draw_legend(frame)
                
                # 显示画面
                cv2.imshow('Basic Pet Recognition - Traditional CV Methods', frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('s'):  # 保存截图
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"basic_pet_screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    logger.info(f"Screenshot saved: {filename}")
                    print(f"📸 Screenshot saved: {filename}")
                elif key == ord('r'):  # 重置统计
                    self.stats = {
                        'total_detections': 0,
                        'detection_types': {},
                        'session_start': time.time()
                    }
                    self.pet_recognizer.detection_history = []
                    logger.info("Statistics reset")
                    print("📊 Statistics reset")
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        logger.info("Cleaning up resources...")
        
        # 显示最终统计
        session_duration = time.time() - self.stats['session_start']
        
        stats_text = f"""
🐾 Basic Pet Recognition Session Summary
========================================
Duration: {session_duration:.1f} seconds
Total Frames: {self.frame_count}
Average FPS: {self.fps:.1f}
Total Detections: {self.stats['total_detections']}

Detection Methods:
"""
        
        for method, count in self.stats['detection_types'].items():
            stats_text += f"  {method.replace('_', ' ').title()}: {count} detections\n"
        
        print(stats_text)
        logger.info("Session completed")
        
        # 释放摄像头
        if self.cap:
            self.cap.release()
        
        # 关闭窗口
        cv2.destroyAllWindows()


def main():
    """主函数"""
    print("🐾 Basic Pet Recognition System")
    print("Using Traditional Computer Vision Methods")
    print("=" * 45)
    print("Features:")
    print("  🎨 Color-based detection")
    print("  🏃 Motion analysis")
    print("  📐 Shape classification")
    print("  🔍 Cascade classifiers")
    print()
    
    try:
        gui = BasicPetRecognitionGUI()
        gui.run()
        
    except Exception as e:
        logger.error(f"Failed to start pet recognition system: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()