#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化宠物识别GUI - 基于YOLO的宠物检测系统
支持多种宠物识别和基础行为分析
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

# 尝试导入YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Warning: ultralytics not available, using basic detection")
    YOLO_AVAILABLE = False


class SimplePetRecognizer:
    """简化的宠物识别器"""
    
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = None
        self.model_path = model_path
        
        # 宠物类别映射 (COCO数据集中的动物类别)
        self.pet_classes = {
            15: 'bird',      # 鸟
            16: 'cat',       # 猫
            17: 'dog',       # 狗
            18: 'horse',     # 马
            19: 'sheep',     # 羊
            20: 'cow',       # 牛
            21: 'elephant',  # 大象
            22: 'bear',      # 熊
            23: 'zebra',     # 斑马
            24: 'giraffe'    # 长颈鹿
        }
        
        # 宠物颜色映射
        self.pet_colors = {
            'bird': (0, 255, 255),    # 黄色
            'cat': (255, 0, 0),       # 蓝色
            'dog': (0, 255, 0),       # 绿色
            'horse': (255, 255, 0),   # 青色
            'sheep': (255, 255, 255), # 白色
            'cow': (0, 0, 0),         # 黑色
            'elephant': (128, 128, 128), # 灰色
            'bear': (139, 69, 19),    # 棕色
            'zebra': (255, 0, 255),   # 洋红
            'giraffe': (255, 165, 0)  # 橙色
        }
        
        self.initialize_model()
    
    def initialize_model(self):
        """初始化YOLO模型"""
        if not YOLO_AVAILABLE:
            logger.warning("YOLO not available, using dummy detection")
            return
        
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"YOLO model loaded: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None
    
    def detect_pets(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
        """检测宠物"""
        if not self.model:
            return []
        
        try:
            # YOLO检测
            results = self.model(frame, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 获取检测信息
                        cls_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # 只处理宠物类别且置信度足够高的检测
                        if cls_id in self.pet_classes and confidence >= confidence_threshold:
                            # 获取边界框
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                            
                            # 创建检测结果
                            detection = {
                                'species': self.pet_classes[cls_id],
                                'confidence': confidence,
                                'bbox': (x, y, w, h),
                                'class_id': cls_id,
                                'center': (x + w//2, y + h//2),
                                'area': w * h
                            }
                            
                            # 添加尺寸分类
                            detection['size'] = self.classify_size(w * h, frame.shape)
                            
                            # 添加基础行为分析
                            detection['behavior'] = self.analyze_basic_behavior(detection, frame.shape)
                            
                            detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.debug(f"Detection error: {e}")
            return []
    
    def classify_size(self, area: int, frame_shape: Tuple[int, int]) -> str:
        """分类宠物尺寸"""
        frame_area = frame_shape[0] * frame_shape[1]
        ratio = area / frame_area
        
        if ratio > 0.3:
            return "Large"
        elif ratio > 0.1:
            return "Medium"
        else:
            return "Small"
    
    def analyze_basic_behavior(self, detection: Dict, frame_shape: Tuple[int, int]) -> str:
        """基础行为分析"""
        x, y, w, h = detection['bbox']
        center_x, center_y = detection['center']
        
        # 基于位置的简单行为推断
        frame_height = frame_shape[0]
        
        # 如果在画面下方，可能在地面活动
        if center_y > frame_height * 0.7:
            return "Ground Activity"
        # 如果在画面中上方，可能在休息或观察
        elif center_y < frame_height * 0.4:
            return "Resting/Observing"
        else:
            return "Active"


class SimplePetRecognitionGUI:
    """简化宠物识别GUI"""
    
    def __init__(self, config_path: str = "camera_config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        
        # 摄像头相关
        self.cap = None
        self.frame_width = 640
        self.frame_height = 480
        
        # 识别器
        self.pet_recognizer = SimplePetRecognizer()
        
        # 性能统计
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        
        # 检测控制
        self.detection_interval = 8  # 每8帧检测一次
        self.display_interval = 3    # 每3帧更新显示
        self.result_hold_frames = 25 # 结果保持25帧
        
        # 结果缓存
        self.cached_results = []
        self.result_cache_time = 0
        
        # 统计信息
        self.stats = {
            'total_detections': 0,
            'species_count': {},
            'session_start': time.time()
        }
        
        self.initialize_camera()
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = {
            "camera": {
                "preferred_index": 1,
                "fallback_indices": [0, 2, 3, 4],
                "use_builtin": False,
                "width": 640,
                "height": 480
            },
            "detection": {
                "confidence_threshold": 0.5,
                "interval": 8
            }
        }
        
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # 合并默认配置
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            else:
                return default_config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return default_config
    
    def initialize_camera(self) -> bool:
        """初始化摄像头"""
        camera_config = self.config.get('camera', {})
        
        # 尝试外部摄像头
        if not camera_config.get('use_builtin', False):
            preferred_index = camera_config.get('preferred_index', 1)
            fallback_indices = camera_config.get('fallback_indices', [0, 2, 3, 4])
            
            indices_to_try = [preferred_index] + [i for i in fallback_indices if i != preferred_index]
            
            for index in indices_to_try:
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
        width = camera_config.get('width', 640)
        height = camera_config.get('height', 480)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # 获取实际尺寸
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Camera setup complete: {self.frame_width}x{self.frame_height}")
        return True
    
    def draw_info_panel(self, frame: np.ndarray, results: List[Dict]) -> np.ndarray:
        """绘制信息面板"""
        # 左上角信息面板
        panel_width = 180
        panel_height = 60
        
        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 统计信息
        species_in_frame = set(r['species'] for r in results)
        
        # 显示信息
        font_scale = 0.4
        thickness = 1
        y_offset = 18
        
        cv2.putText(frame, f"Pets Detected: {len(results)}", 
                   (8, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        cv2.putText(frame, f"Species: {len(species_in_frame)}", 
                   (8, y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        cv2.putText(frame, f"FPS: {self.fps:.1f} | Frame: {self.frame_count}", 
                   (8, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        return frame
    
    def draw_legend(self, frame: np.ndarray) -> np.ndarray:
        """绘制图例"""
        # 底部图例
        legend_height = 25
        legend_y = self.frame_height - legend_height
        
        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, legend_y), (self.frame_width, self.frame_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # 图例文本
        legend_text = "Pets: Dog🐕 Cat🐱 Bird🐦 Horse🐴 | Behaviors: Ground Activity, Resting, Active"
        font_scale = 0.35
        thickness = 1
        
        cv2.putText(frame, legend_text, (5, legend_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        return frame
    
    def draw_detections(self, frame: np.ndarray, results: List[Dict]) -> np.ndarray:
        """绘制检测结果"""
        for detection in results:
            species = detection['species']
            confidence = detection['confidence']
            x, y, w, h = detection['bbox']
            
            # 获取颜色
            color = self.pet_recognizer.pet_colors.get(species, (128, 128, 128))
            
            # 绘制边界框
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # 绘制标签
            label = f"{species.title()} {confidence:.2f}"
            if 'size' in detection:
                label += f" ({detection['size']})"
            
            # 标签背景
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x, y - label_h - 8), (x + label_w + 4, y), color, -1)
            cv2.putText(frame, label, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 绘制行为信息
            if 'behavior' in detection:
                behavior_text = f"Behavior: {detection['behavior']}"
                cv2.putText(frame, behavior_text, (x, y + h + 15), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # 绘制中心点
            center_x, center_y = detection['center']
            cv2.circle(frame, (center_x, center_y), 3, color, -1)
        
        return frame
    
    def log_detections(self, results: List[Dict]):
        """记录检测结果"""
        for detection in results:
            species = detection['species']
            confidence = detection['confidence']
            size = detection.get('size', 'Unknown')
            behavior = detection.get('behavior', 'Unknown')
            
            log_msg = f"[PET DETECTED] Species: {species}, Confidence: {confidence:.3f}, Size: {size}, Behavior: {behavior}"
            logger.info(log_msg)
            print(f"🐾 {log_msg}")
            
            # 更新统计
            if species not in self.stats['species_count']:
                self.stats['species_count'][species] = 0
            self.stats['species_count'][species] += 1
            self.stats['total_detections'] += 1
    
    def run(self):
        """运行主循环"""
        if not self.cap:
            logger.error("Camera not initialized")
            return
        
        logger.info("Starting simple pet recognition GUI...")
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
                    confidence_threshold = self.config.get('detection', {}).get('confidence_threshold', 0.5)
                    results = self.pet_recognizer.detect_pets(frame, confidence_threshold)
                    
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
                cv2.imshow('Pet Recognition - Simple Detection System', frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('s'):  # 保存截图
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"pet_screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    logger.info(f"Screenshot saved: {filename}")
                    print(f"📸 Screenshot saved: {filename}")
                elif key == ord('r'):  # 重置统计
                    self.stats = {
                        'total_detections': 0,
                        'species_count': {},
                        'session_start': time.time()
                    }
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
🐾 Pet Recognition Session Summary
==================================
Duration: {session_duration:.1f} seconds
Total Frames: {self.frame_count}
Average FPS: {self.fps:.1f}
Total Detections: {self.stats['total_detections']}

Species Detected:
"""
        
        for species, count in self.stats['species_count'].items():
            stats_text += f"  {species.title()}: {count} detections\n"
        
        print(stats_text)
        logger.info("Session completed")
        
        # 释放摄像头
        if self.cap:
            self.cap.release()
        
        # 关闭窗口
        cv2.destroyAllWindows()


def main():
    """主函数"""
    print("🐾 Simple Pet Recognition System")
    print("Based on YOLO Object Detection")
    print("=" * 40)
    
    if not YOLO_AVAILABLE:
        print("⚠️ YOLO not available - install ultralytics: pip install ultralytics")
        print("Running in basic mode...")
    
    try:
        gui = SimplePetRecognitionGUI()
        gui.run()
        
    except Exception as e:
        logger.error(f"Failed to start pet recognition system: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()