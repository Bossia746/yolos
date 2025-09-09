#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
宠物识别GUI - 基于业界最佳实践的增强宠物视觉捕捉系统
支持物种识别、品种分类、姿态估计、行为分析、健康监测
"""

import cv2
import numpy as np
import time
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.recognition.enhanced_pet_recognizer import (
        EnhancedPetRecognizer, PetSpecies, PetBehavior, PetHealthStatus,
        EnhancedPetResult
    )
    ENHANCED_PET_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Enhanced pet recognizer not available: {e}")
    ENHANCED_PET_AVAILABLE = False

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


class PetRecognitionGUI:
    """宠物识别GUI主类"""
    
    def __init__(self, config_path: str = "camera_config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        
        # 摄像头相关
        self.cap = None
        self.frame_width = 640
        self.frame_height = 480
        
        # 识别器
        self.pet_recognizer = None
        
        # 性能统计
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        
        # 检测控制
        self.detection_interval = self.config.get('detection', {}).get('interval', 10)  # 每10帧检测一次
        self.display_interval = self.config.get('display', {}).get('interval', 5)      # 每5帧更新显示
        self.result_hold_frames = self.config.get('display', {}).get('hold_frames', 30) # 结果保持30帧
        
        # 结果缓存
        self.cached_results = []
        self.result_cache_time = 0
        
        # 统计信息
        self.stats = {
            'total_pets': 0,
            'species_detected': set(),
            'behaviors_observed': set(),
            'health_alerts': 0,
            'session_start': time.time()
        }
        
        # 日志输出
        self.enable_logging = self.config.get('logging', {}).get('enabled', True)
        self.log_detections = self.config.get('logging', {}).get('detections', True)
        self.log_behaviors = self.config.get('logging', {}).get('behaviors', True)
        self.log_health = self.config.get('logging', {}).get('health', True)
        
        self.initialize_system()
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = {
            "camera": {
                "preferred_index": 1,
                "fallback_indices": [0, 2, 3, 4],
                "use_builtin": False,
                "width": 640,
                "height": 480,
                "fps": 30
            },
            "detection": {
                "interval": 10,
                "confidence_threshold": 0.5,
                "enable_pose": True,
                "enable_behavior": True,
                "enable_health": True
            },
            "display": {
                "interval": 5,
                "hold_frames": 30,
                "show_keypoints": True,
                "show_behavior": True,
                "show_health": True,
                "compact_layout": True
            },
            "logging": {
                "enabled": True,
                "detections": True,
                "behaviors": True,
                "health": True,
                "file_path": "pet_recognition.log"
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
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if subkey not in config[key]:
                                config[key][subkey] = subvalue
                return config
            else:
                logger.info(f"Config file not found, using defaults: {self.config_path}")
                return default_config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return default_config
    
    def initialize_system(self):
        """初始化系统"""
        logger.info("Initializing Pet Recognition System...")
        
        # 初始化摄像头
        if not self.initialize_camera():
            logger.error("Failed to initialize camera")
            return False
        
        # 初始化宠物识别器
        if not self.initialize_pet_recognizer():
            logger.error("Failed to initialize pet recognizer")
            return False
        
        logger.info("Pet Recognition System initialized successfully")
        return True
    
    def initialize_camera(self) -> bool:
        """初始化摄像头"""
        camera_config = self.config.get('camera', {})
        
        # 尝试外部摄像头
        if not camera_config.get('use_builtin', False):
            preferred_index = camera_config.get('preferred_index', 1)
            fallback_indices = camera_config.get('fallback_indices', [0, 2, 3, 4])
            
            # 首先尝试首选索引
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
        
        # 如果外部摄像头失败，尝试内置摄像头
        if self.cap is None and camera_config.get('use_builtin', True):
            logger.info("Trying builtin camera (index 0)")
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.cap = cap
                    logger.info(f"Builtin camera OK: {frame.shape}")
                else:
                    cap.release()
        
        if self.cap is None:
            logger.error("No camera available")
            return False
        
        # 设置摄像头参数
        width = camera_config.get('width', 640)
        height = camera_config.get('height', 480)
        fps = camera_config.get('fps', 30)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        # 获取实际尺寸
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Camera setup complete: {self.frame_width}x{self.frame_height}")
        return True
    
    def initialize_pet_recognizer(self) -> bool:
        """初始化宠物识别器"""
        if not ENHANCED_PET_AVAILABLE:
            logger.error("Enhanced pet recognizer not available")
            return False
        
        try:
            # 配置识别器
            recognizer_config = {
                'yolo_model_path': self.config.get('models', {}).get('yolo_path', 'yolov8n.pt'),
                'species_model_path': self.config.get('models', {}).get('species_path'),
                'behavior_history_length': self.config.get('detection', {}).get('behavior_history', 30)
            }
            
            self.pet_recognizer = EnhancedPetRecognizer(recognizer_config)
            logger.info("Enhanced pet recognizer initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize pet recognizer: {e}")
            return False
    
    def detect_pets(self, frame: np.ndarray, timestamp: float) -> List[EnhancedPetResult]:
        """检测宠物"""
        if not self.pet_recognizer:
            return []
        
        try:
            results = self.pet_recognizer.detect_and_analyze(frame, timestamp)
            
            # 记录检测结果到日志
            if self.enable_logging and results:
                self.log_detection_results(results, timestamp)
            
            return results
            
        except Exception as e:
            logger.debug(f"Pet detection error: {e}")
            return []
    
    def log_detection_results(self, results: List[EnhancedPetResult], timestamp: float):
        """记录检测结果到日志和控制台"""
        for result in results:
            detection = result.detection
            
            # 基本检测信息
            if self.log_detections:
                detection_info = (
                    f"[PET DETECTED] Species: {detection.species.value}, "
                    f"Confidence: {detection.confidence:.3f}, "
                    f"Size: {detection.size_category}, "
                    f"ID: {result.tracking_id}"
                )
                logger.info(detection_info)
                print(f"🐾 {detection_info}")
            
            # 行为信息
            if self.log_behaviors and result.behavior:
                behavior_info = (
                    f"[BEHAVIOR] ID: {result.tracking_id}, "
                    f"Behavior: {result.behavior.behavior.value}, "
                    f"Confidence: {result.behavior.confidence:.3f}, "
                    f"Duration: {result.behavior.duration:.1f}s, "
                    f"Intensity: {result.behavior.intensity:.3f}"
                )
                logger.info(behavior_info)
                print(f"🎭 {behavior_info}")
            
            # 健康信息
            if self.log_health and result.health:
                health_info = (
                    f"[HEALTH] ID: {result.tracking_id}, "
                    f"Status: {result.health.status.value}, "
                    f"Confidence: {result.health.confidence:.3f}"
                )
                
                if result.health.risk_factors:
                    health_info += f", Risk Factors: {', '.join(result.health.risk_factors)}"
                
                if result.health.status in [PetHealthStatus.SICK, PetHealthStatus.INJURED, PetHealthStatus.STRESSED]:
                    logger.warning(health_info)
                    print(f"⚠️ {health_info}")
                    self.stats['health_alerts'] += 1
                else:
                    logger.info(health_info)
                    print(f"💚 {health_info}")
    
    def draw_compact_info_panel(self, frame: np.ndarray, results: List[EnhancedPetResult]) -> np.ndarray:
        """绘制紧凑的信息面板"""
        if not results:
            return frame
        
        # 紧凑的左上角信息面板 (更小尺寸)
        panel_width = 160
        panel_height = 45
        
        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 统计信息
        species_count = len(set(r.detection.species for r in results))
        behavior_count = len(set(r.behavior.behavior for r in results if r.behavior))
        health_alerts = sum(1 for r in results if r.health and 
                          r.health.status in [PetHealthStatus.SICK, PetHealthStatus.INJURED, PetHealthStatus.STRESSED])
        
        # 显示信息 (更小字体)
        font_scale = 0.35
        thickness = 1
        y_offset = 15
        
        cv2.putText(frame, f"Pets: {len(results)} | Species: {species_count}", 
                   (8, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        cv2.putText(frame, f"Behaviors: {behavior_count} | Alerts: {health_alerts}", 
                   (8, y_offset + 12), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        cv2.putText(frame, f"FPS: {self.fps:.1f} | Frame: {self.frame_count}", 
                   (8, y_offset + 24), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        return frame
    
    def draw_compact_legend(self, frame: np.ndarray) -> np.ndarray:
        """绘制紧凑的图例"""
        # 底部紧凑图例
        legend_height = 25
        legend_y = self.frame_height - legend_height
        
        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, legend_y), (self.frame_width, self.frame_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # 图例文本 (单行显示)
        legend_text = "Species: Dog🐕 Cat🐱 Bird🐦 | Behavior: Sleep😴 Play🎾 Eat🍽️ | Health: ✅Healthy ⚠️Alert"
        font_scale = 0.3
        thickness = 1
        
        cv2.putText(frame, legend_text, (5, legend_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        return frame
    
    def draw_pet_annotations(self, frame: np.ndarray, results: List[EnhancedPetResult]) -> np.ndarray:
        """绘制宠物标注"""
        for result in results:
            detection = result.detection
            x, y, w, h = detection.bbox
            
            # 获取物种颜色
            color = self.get_species_color(detection.species)
            
            # 绘制边界框
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # 绘制标签
            label_parts = [
                f"{detection.species.value}",
                f"{detection.confidence:.2f}"
            ]
            
            if result.tracking_id:
                label_parts.append(f"ID{result.tracking_id}")
            
            if detection.size_category:
                label_parts.append(detection.size_category)
            
            label = " ".join(label_parts)
            
            # 标签背景
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(frame, (x, y - label_h - 8), (x + label_w + 4, y), color, -1)
            cv2.putText(frame, label, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 绘制姿态关键点
            if self.config.get('display', {}).get('show_keypoints', True) and result.pose:
                self.draw_pose_keypoints(frame, result.pose, color)
            
            # 绘制行为信息
            if self.config.get('display', {}).get('show_behavior', True) and result.behavior:
                behavior_text = f"{result.behavior.behavior.value} {result.behavior.intensity:.1f}"
                cv2.putText(frame, behavior_text, (x, y + h + 15), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            
            # 绘制健康状态
            if self.config.get('display', {}).get('show_health', True) and result.health:
                health_color = self.get_health_color(result.health.status)
                health_text = f"Health: {result.health.status.value}"
                cv2.putText(frame, health_text, (x, y + h + 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.35, health_color, 1)
        
        return frame
    
    def draw_pose_keypoints(self, frame: np.ndarray, pose, color: tuple):
        """绘制姿态关键点"""
        if not pose or not pose.keypoints:
            return
        
        # 绘制关键点
        for i, (x, y, confidence) in enumerate(pose.keypoints):
            if confidence > 0.5:  # 只绘制置信度高的关键点
                cv2.circle(frame, (int(x), int(y)), 2, color, -1)
        
        # 绘制姿态类型
        if pose.pose_type != 'unknown':
            bbox_center = (int(pose.keypoints[0][0]), int(pose.keypoints[0][1]) - 20)
            cv2.putText(frame, pose.pose_type, bbox_center, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    def get_species_color(self, species: PetSpecies) -> tuple:
        """获取物种对应的颜色"""
        color_map = {
            PetSpecies.DOG: (0, 255, 0),      # 绿色
            PetSpecies.CAT: (255, 0, 0),      # 蓝色
            PetSpecies.BIRD: (0, 255, 255),   # 黄色
            PetSpecies.RABBIT: (255, 0, 255), # 洋红
            PetSpecies.HAMSTER: (0, 165, 255), # 橙色
            PetSpecies.GUINEA_PIG: (128, 0, 128), # 紫色
            PetSpecies.FERRET: (0, 128, 255),  # 橙红色
            PetSpecies.FISH: (255, 255, 0),    # 青色
            PetSpecies.REPTILE: (128, 128, 0), # 橄榄色
            PetSpecies.UNKNOWN: (128, 128, 128) # 灰色
        }
        return color_map.get(species, (128, 128, 128))
    
    def get_health_color(self, status: PetHealthStatus) -> tuple:
        """获取健康状态对应的颜色"""
        color_map = {
            PetHealthStatus.HEALTHY: (0, 255, 0),      # 绿色
            PetHealthStatus.SICK: (0, 0, 255),         # 红色
            PetHealthStatus.INJURED: (0, 0, 255),      # 红色
            PetHealthStatus.STRESSED: (0, 165, 255),   # 橙色
            PetHealthStatus.LETHARGIC: (0, 255, 255),  # 黄色
            PetHealthStatus.HYPERACTIVE: (255, 0, 255), # 洋红
            PetHealthStatus.UNKNOWN: (128, 128, 128)   # 灰色
        }
        return color_map.get(status, (128, 128, 128))
    
    def update_statistics(self, results: List[EnhancedPetResult]):
        """更新统计信息"""
        for result in results:
            self.stats['total_pets'] += 1
            self.stats['species_detected'].add(result.detection.species.value)
            
            if result.behavior:
                self.stats['behaviors_observed'].add(result.behavior.behavior.value)
            
            if (result.health and 
                result.health.status in [PetHealthStatus.SICK, PetHealthStatus.INJURED, PetHealthStatus.STRESSED]):
                self.stats['health_alerts'] += 1
    
    def run(self):
        """运行主循环"""
        if not self.cap or not self.pet_recognizer:
            logger.error("System not properly initialized")
            return
        
        logger.info("Starting pet recognition GUI...")
        logger.info("Controls: 'q' or ESC to quit, 's' to save screenshot, 'r' to reset stats, 'h' for help")
        
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
                
                # 检测控制 - 降低检测频率
                should_detect = (self.frame_count % self.detection_interval == 0)
                should_update_display = (self.frame_count % self.display_interval == 0)
                
                # 执行检测
                if should_detect:
                    results = self.detect_pets(frame, current_time)
                    if results:
                        self.cached_results = results
                        self.result_cache_time = self.frame_count
                        self.update_statistics(results)
                
                # 使用缓存结果
                display_results = []
                if self.cached_results and (self.frame_count - self.result_cache_time) < self.result_hold_frames:
                    display_results = self.cached_results
                
                # 绘制结果
                if display_results and should_update_display:
                    frame = self.pet_recognizer.draw_results(frame, display_results)
                    frame = self.draw_pet_annotations(frame, display_results)
                
                # 绘制界面元素
                if self.config.get('display', {}).get('compact_layout', True):
                    frame = self.draw_compact_info_panel(frame, display_results)
                    frame = self.draw_compact_legend(frame)
                
                # 显示画面
                cv2.imshow('Pet Recognition - Enhanced Vision System', frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('s'):  # 保存截图
                    self.save_screenshot(frame)
                elif key == ord('r'):  # 重置统计
                    self.reset_statistics()
                elif key == ord('h'):  # 帮助
                    self.show_help()
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.cleanup()
    
    def save_screenshot(self, frame: np.ndarray):
        """保存截图"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"pet_recognition_screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        logger.info(f"Screenshot saved: {filename}")
        print(f"📸 Screenshot saved: {filename}")
    
    def reset_statistics(self):
        """重置统计信息"""
        self.stats = {
            'total_pets': 0,
            'species_detected': set(),
            'behaviors_observed': set(),
            'health_alerts': 0,
            'session_start': time.time()
        }
        
        if self.pet_recognizer:
            self.pet_recognizer.reset_statistics()
        
        logger.info("Statistics reset")
        print("📊 Statistics reset")
    
    def show_help(self):
        """显示帮助信息"""
        help_text = """
🐾 Pet Recognition System - Help
================================
Controls:
  'q' or ESC  - Quit application
  's'         - Save screenshot
  'r'         - Reset statistics
  'h'         - Show this help

Features:
  🔍 Species Recognition - Dog, Cat, Bird, Rabbit, etc.
  🎭 Behavior Analysis - Sleep, Play, Eat, Walk, etc.
  💚 Health Monitoring - Activity, Stress, Wellness
  📊 Real-time Statistics
  📝 Detailed Logging

System Status:
  Camera: Active
  Detection: Every {self.detection_interval} frames
  Display: Every {self.display_interval} frames
  Logging: {'Enabled' if self.enable_logging else 'Disabled'}
        """.format(self=self)
        
        print(help_text)
        logger.info("Help information displayed")
    
    def cleanup(self):
        """清理资源"""
        logger.info("Cleaning up resources...")
        
        # 显示最终统计
        self.show_final_statistics()
        
        # 释放摄像头
        if self.cap:
            self.cap.release()
        
        # 清理识别器
        if self.pet_recognizer:
            self.pet_recognizer.cleanup()
        
        # 关闭窗口
        cv2.destroyAllWindows()
        
        logger.info("Pet recognition system shutdown complete")
    
    def show_final_statistics(self):
        """显示最终统计信息"""
        session_duration = time.time() - self.stats['session_start']
        
        stats_text = f"""
🐾 Pet Recognition Session Summary
==================================
Duration: {session_duration:.1f} seconds
Total Frames: {self.frame_count}
Average FPS: {self.fps:.1f}

Detection Results:
  Total Pets Detected: {self.stats['total_pets']}
  Species Identified: {len(self.stats['species_detected'])} types
  Species List: {', '.join(self.stats['species_detected']) if self.stats['species_detected'] else 'None'}
  
Behavior Analysis:
  Behaviors Observed: {len(self.stats['behaviors_observed'])} types
  Behavior List: {', '.join(self.stats['behaviors_observed']) if self.stats['behaviors_observed'] else 'None'}
  
Health Monitoring:
  Health Alerts: {self.stats['health_alerts']}
  
System Performance:
  Detection Interval: Every {self.detection_interval} frames
  Display Interval: Every {self.display_interval} frames
  Logging: {'Enabled' if self.enable_logging else 'Disabled'}
        """
        
        print(stats_text)
        logger.info("Session statistics displayed")
        
        # 保存统计到文件
        stats_file = f"pet_recognition_stats_{time.strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                stats_data = {
                    'session_duration': session_duration,
                    'total_frames': self.frame_count,
                    'average_fps': self.fps,
                    'total_pets': self.stats['total_pets'],
                    'species_detected': list(self.stats['species_detected']),
                    'behaviors_observed': list(self.stats['behaviors_observed']),
                    'health_alerts': self.stats['health_alerts'],
                    'detection_interval': self.detection_interval,
                    'display_interval': self.display_interval,
                    'logging_enabled': self.enable_logging
                }
                json.dump(stats_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Statistics saved to: {stats_file}")
        except Exception as e:
            logger.error(f"Failed to save statistics: {e}")


def main():
    """主函数"""
    print("🐾 Pet Recognition System - Enhanced Vision Capture")
    print("Based on Industry Best Practices")
    print("=" * 50)
    
    if not ENHANCED_PET_AVAILABLE:
        print("❌ Enhanced pet recognizer not available")
        print("Please check the installation of required dependencies")
        return
    
    try:
        # 创建并运行GUI
        gui = PetRecognitionGUI()
        gui.run()
        
    except Exception as e:
        logger.error(f"Failed to start pet recognition system: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()