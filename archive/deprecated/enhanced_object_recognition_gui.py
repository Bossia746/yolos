#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强静物识别GUI - 支持二维码、条形码、车牌、交通符号等
基于OpenCV和深度学习的多功能识别系统
"""

import cv2
import numpy as np
import os
import sys
import logging
import time
import json
from datetime import datetime
import locale

# 设置编码和locale
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['LC_ALL'] = 'C'
locale.setlocale(locale.LC_ALL, 'C')

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_object_recognition.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class EnhancedObjectRecognitionGUI:
    """增强静物识别GUI类 - 支持多种特殊对象识别"""
    
    def __init__(self):
        self.enhanced_recognizer = None
        self.cap = None
        self.config = self.load_config()
        self.detection_stats = {
            'qr_codes': 0,
            'barcodes': 0,
            'license_plates': 0,
            'traffic_signs': 0,
            'traffic_lights': 0,
            'facility_signs': 0,
            'basic_objects': 0,
            'total_objects': 0,
            'frames': 0
        }
        self.start_time = time.time()
        self.detection_interval = self.config['detection']['detection_interval']  # 15帧检测一次，大幅降低刷新
        self.display_interval = self.config['detection']['display_interval']     # 8帧更新显示一次
        self.result_hold_frames = self.config['detection']['result_hold_frames'] # 45帧保持结果显示
        self.frame_count = 0
        self.last_results = None    # 缓存上次检测结果
        self.last_detection_frame = 0  # 上次检测的帧数
        self.focus_stabilize_frames = self.config['detection']['focus_stabilize_frames']
        
    def load_config(self):
        """加载配置文件"""
        try:
            with open('camera_config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info("Configuration loaded successfully")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config, using defaults: {e}")
            return {
                "camera": {
                    "preferred_index": 1,
                    "fallback_indices": [2, 3],
                    "use_builtin": False,
                    "width": 640,
                    "height": 480,
                    "fps": 30
                },
                "detection": {
                    "detection_interval": 15,
                    "display_interval": 8,
                    "result_hold_frames": 45,
                    "focus_stabilize_frames": 30,
                    "confidence_threshold": 0.5
                },
                "display": {
                    "show_fps": True,
                    "show_categories": True,
                    "show_summary": True,
                    "panel_transparency": 0.8
                }
            }
        
    def initialize_recognizer(self):
        """初始化增强静物识别器"""
        logger.info("Initializing enhanced object recognizer...")
        
        try:
            from recognition.enhanced_object_recognizer import EnhancedObjectRecognizer
            self.enhanced_recognizer = EnhancedObjectRecognizer()
            logger.info("Enhanced object recognizer OK")
        except Exception as e:
            logger.error(f"Enhanced object recognizer failed: {e}")
            raise
    
    def initialize_camera(self):
        """初始化摄像头 - 强制使用外部USB摄像头"""
        logger.info("Initializing camera (USB external only)...")
        
        # 根据配置强制使用外部摄像头
        camera_config = self.config['camera']
        
        if camera_config['use_builtin']:
            camera_indices = [0, int(camera_config['preferred_index'])] + list(camera_config['fallback_indices'])
        else:
            # 强制不使用内置摄像头(索引0)
            camera_indices = [int(camera_config['preferred_index'])] + list(camera_config['fallback_indices'])
            logger.info("Built-in camera disabled, using external USB camera only")
        
        camera_found = False
        for camera_index in camera_indices:
            logger.info(f"Trying camera index {camera_index}...")
            self.cap = cv2.VideoCapture(int(camera_index))
            
            if self.cap.isOpened():
                # 测试是否能读取帧
                ret, frame = self.cap.read()
                if ret:
                    logger.info(f"Camera {camera_index} OK: {frame.shape}")
                    camera_found = True
                    break
                else:
                    self.cap.release()
            else:
                if self.cap:
                    self.cap.release()
        
        if not camera_found:
            if not camera_config['use_builtin']:
                logger.error("No external USB camera found! Please:")
                logger.error("1. Connect USB camera")
                logger.error("2. Or set 'use_builtin': true in camera_config.json")
                raise RuntimeError("No external USB camera available")
            else:
                raise RuntimeError("Cannot open any camera")
        
        # 设置摄像头参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera_config['width']))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera_config['height']))
        self.cap.set(cv2.CAP_PROP_FPS, int(camera_config['fps']))
        
        # 验证最终设置
        ret, frame = self.cap.read()
        if ret:
            logger.info(f"Final camera setup OK: {frame.shape}")
        else:
            raise RuntimeError("Cannot read from selected camera")
    
    def detect_enhanced_objects(self, frame):
        """检测增强静物对象 - 优化对焦和刷新频率"""
        results = {
            'objects': [],
            'summary': {
                'qr_codes': 0,
                'barcodes': 0,
                'license_plates': 0,
                'traffic_signs': 0,
                'traffic_lights': 0,
                'facility_signs': 0,
                'basic_objects': 0,
                'total_objects': 0
            }
        }
        
        try:
            # 前30帧用于对焦稳定，不进行检测
            if self.frame_count < int(self.focus_stabilize_frames):
                # 显示对焦提示
                focus_frame = frame.copy()
                remaining_frames = int(self.focus_stabilize_frames) - self.frame_count
                focus_text = f"Camera focusing... {remaining_frames} frames remaining"
                cv2.putText(focus_frame, focus_text, (50, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(focus_frame, "Please hold objects steady for better detection", (50, 80), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                return focus_frame, results
            
            # 每隔15帧进行一次检测，大幅降低刷新频率
            if self.frame_count % self.detection_interval == 0:
                annotated_frame, object_results = self.enhanced_recognizer.detect_objects(frame)
                
                # 统计各类型对象并输出识别内容到日志和控制台
                for obj in object_results:
                    obj_type = obj.get('type', 'unknown')
                    results['summary']['total_objects'] += 1
                    
                    # 输出识别内容到日志和控制台
                    self._log_detection_content(obj)
                    
                    if obj_type == 'qr_code':
                        results['summary']['qr_codes'] += 1
                        self.detection_stats['qr_codes'] += 1
                    elif obj_type == 'barcode':
                        results['summary']['barcodes'] += 1
                        self.detection_stats['barcodes'] += 1
                    elif obj_type == 'license_plate':
                        results['summary']['license_plates'] += 1
                        self.detection_stats['license_plates'] += 1
                    elif obj_type == 'traffic_sign':
                        results['summary']['traffic_signs'] += 1
                        self.detection_stats['traffic_signs'] += 1
                    elif obj_type == 'traffic_light':
                        results['summary']['traffic_lights'] += 1
                        self.detection_stats['traffic_lights'] += 1
                    elif obj_type == 'facility_sign':
                        results['summary']['facility_signs'] += 1
                        self.detection_stats['facility_signs'] += 1
                    elif obj_type == 'basic_object':
                        results['summary']['basic_objects'] += 1
                        self.detection_stats['basic_objects'] += 1
                    
                    self.detection_stats['total_objects'] += 1
                
                results['objects'] = object_results
                self.last_results = (annotated_frame, results)  # 缓存结果
                self.last_detection_frame = self.frame_count
                return annotated_frame, results
            
            # 在结果保持期内，继续显示上次检测结果，避免闪烁
            elif (self.last_results is not None and 
                  self.frame_count - self.last_detection_frame < int(self.result_hold_frames)):
                # 使用缓存的检测结果，但在原始帧上重新绘制，保持稳定显示
                cached_frame, cached_results = self.last_results
                return cached_frame, cached_results
            else:
                # 超出保持期，显示原始帧
                return frame, results
                
        except Exception as e:
            logger.debug(f"Enhanced object detection error: {e}")
            return frame, results
    
    def _log_detection_content(self, obj):
        """输出识别内容到日志和控制台"""
        try:
            obj_type = obj.get('type', 'unknown')
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            if obj_type == 'qr_code':
                content = obj.get('content', 'No content')
                message = f"[{timestamp}] QR Code detected: {content}"
                logger.info(message)
                print(f"🔍 {message}")
                
            elif obj_type == 'barcode':
                content = obj.get('content', 'No content')
                format_type = obj.get('format', 'Unknown')
                message = f"[{timestamp}] Barcode detected ({format_type}): {content}"
                logger.info(message)
                print(f"📊 {message}")
                
            elif obj_type == 'basic_object' and 'text' in obj:
                text_content = obj.get('text', '')
                if text_content and text_content.strip():
                    message = f"[{timestamp}] Text detected: {text_content.strip()}"
                    logger.info(message)
                    print(f"📝 {message}")
                    
            elif obj_type == 'license_plate':
                plate_number = obj.get('content', 'Unknown')
                message = f"[{timestamp}] License plate detected: {plate_number}"
                logger.info(message)
                print(f"🚗 {message}")
                
            elif obj_type == 'traffic_sign':
                sign_type = obj.get('content', 'Unknown sign')
                message = f"[{timestamp}] Traffic sign detected: {sign_type}"
                logger.info(message)
                print(f"🚦 {message}")
                
            elif obj_type == 'traffic_light':
                light_state = obj.get('content', 'Unknown state')
                message = f"[{timestamp}] Traffic light detected: {light_state}"
                logger.info(message)
                print(f"🚥 {message}")
                
            elif obj_type == 'facility_sign':
                facility_type = obj.get('content', 'Unknown facility')
                message = f"[{timestamp}] Facility sign detected: {facility_type}"
                logger.info(message)
                print(f"🏢 {message}")
                
        except Exception as e:
            logger.debug(f"Log detection content error: {e}")
    
    def draw_enhanced_summary_info(self, frame, results):
        """绘制增强汇总信息 - 紧凑版本"""
        try:
            summary = results['summary']
            
            # 准备紧凑信息文本
            info_lines = [
                f"Enhanced Detection",
                f"QR:{summary['qr_codes']} Bar:{summary['barcodes']} Plate:{summary['license_plates']}",
                f"Sign:{summary['traffic_signs']} Light:{summary['traffic_lights']} Total:{summary['total_objects']}"
            ]
            
            # 绘制紧凑信息面板背景
            panel_height = len(info_lines) * 16 + 10
            panel_width = 180
            cv2.rectangle(frame, (5, 5), (panel_width, panel_height), (0, 0, 0), -1)
            cv2.rectangle(frame, (5, 5), (panel_width, panel_height), (255, 255, 255), 1)
            
            # 绘制信息文本
            for i, line in enumerate(info_lines):
                y_pos = 18 + i * 16
                color = (0, 255, 255) if i == 0 else (255, 255, 255)
                font_scale = 0.4 if i == 0 else 0.35
                cv2.putText(frame, line, (8, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
            
        except Exception as e:
            logger.debug(f"Draw enhanced summary info error: {e}")
    
    def draw_detection_categories(self, frame):
        """绘制检测类别说明 - 紧凑版本"""
        try:
            categories = [
                "Categories: QR(Green) Bar(Blue) Plate(Cyan) Sign(Magenta) Light(Color) Facility(Yellow)"
            ]
            
            # 绘制紧凑类别面板
            start_y = frame.shape[0] - 25
            panel_width = min(frame.shape[1] - 10, 600)
            
            cv2.rectangle(frame, (5, start_y), (panel_width, frame.shape[0] - 5), (50, 50, 50), -1)
            cv2.rectangle(frame, (5, start_y), (panel_width, frame.shape[0] - 5), (200, 200, 200), 1)
            
            cv2.putText(frame, categories[0], (8, start_y + 15), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            
        except Exception as e:
            logger.debug(f"Draw detection categories error: {e}")
    
    def run(self):
        """运行增强GUI"""
        try:
            self.initialize_recognizer()
            self.initialize_camera()
            
            logger.info("Starting enhanced object detection...")
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.detection_stats['frames'] += 1
                self.frame_count += 1
                
                # 增强静物检测
                annotated_frame, results = self.detect_enhanced_objects(frame)
                
                # 绘制增强汇总信息
                self.draw_enhanced_summary_info(annotated_frame, results)
                
                # 绘制检测类别说明
                self.draw_detection_categories(annotated_frame)
                
                # 显示统计信息
                elapsed_time = time.time() - self.start_time
                fps = self.detection_stats['frames'] / elapsed_time if elapsed_time > 0 else 0
                
                stats_text = f"FPS: {fps:.1f} | QR: {self.detection_stats['qr_codes']} | Barcode: {self.detection_stats['barcodes']} | License: {self.detection_stats['license_plates']} | Traffic: {self.detection_stats['traffic_signs']} | Lights: {self.detection_stats['traffic_lights']} | Facility: {self.detection_stats['facility_signs']} | Objects: {self.detection_stats['basic_objects']}"
                
                # 绘制状态栏
                cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 25), (0, 0, 0), -1)
                cv2.putText(annotated_frame, stats_text, (5, 18), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                
                # 显示帧
                cv2.imshow('Enhanced Object Recognition GUI', annotated_frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # ESC
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"enhanced_detection_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    logger.info(f"Screenshot saved: {filename}")
                elif key == ord('r'):
                    self.detection_stats = {
                        'qr_codes': 0, 'barcodes': 0, 'license_plates': 0,
                        'traffic_signs': 0, 'traffic_lights': 0, 'facility_signs': 0,
                        'basic_objects': 0, 'total_objects': 0, 'frames': 0
                    }
                    self.start_time = time.time()
                    self.frame_count = 0
                    logger.info("Statistics reset")
                elif key == ord('h'):
                    self._show_enhanced_help()
        
        except Exception as e:
            logger.error(f"Error in enhanced main loop: {e}")
        
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            logger.info("Enhanced object recognition GUI closed")
            
            # 生成最终报告
            self._generate_enhanced_final_report()
    
    def _show_enhanced_help(self):
        """显示增强帮助信息"""
        help_text = """
Enhanced Object Recognition GUI Help
===================================

Controls:
• 'q' or ESC - Exit program
• 's' - Save screenshot
• 'r' - Reset statistics
• 'h' - Show this help

Enhanced Detection Features:
• QR Code Recognition - Decodes QR code content
• Barcode Detection - Identifies various barcode formats
• License Plate Detection - Recognizes vehicle plates
• Traffic Sign Recognition - Identifies road signs
• Traffic Light Detection - Detects red/yellow/green lights
• Facility Sign Recognition - Public facility symbols
• Basic Object Analysis - Color, shape, text detection

Detection Categories:
• QR Codes - Green boxes with decoded content
• Barcodes - Blue boxes with barcode data
• License Plates - Cyan boxes with plate numbers
• Traffic Signs - Magenta boxes with sign types
• Traffic Lights - Color-coded boxes (red/yellow/green)
• Facility Signs - Yellow boxes with facility types
• Basic Objects - Various colors based on object properties

Tips for Better Detection:
• Use good lighting conditions
• Hold objects steady for text/code detection
• Ensure clear view of signs and symbols
• USB camera recommended for better quality
        """
        print(help_text)
    
    def _generate_enhanced_final_report(self):
        """生成增强最终报告"""
        try:
            elapsed_time = time.time() - self.start_time
            avg_fps = self.detection_stats['frames'] / elapsed_time if elapsed_time > 0 else 0
            
            report = f"""
Enhanced Object Recognition Session Report
=========================================
Duration: {elapsed_time:.1f} seconds
Average FPS: {avg_fps:.1f}
Total Frames: {self.detection_stats['frames']}

Enhanced Detection Results:
• QR Codes Detected: {self.detection_stats['qr_codes']}
• Barcodes Detected: {self.detection_stats['barcodes']}
• License Plates Detected: {self.detection_stats['license_plates']}
• Traffic Signs Detected: {self.detection_stats['traffic_signs']}
• Traffic Lights Detected: {self.detection_stats['traffic_lights']}
• Facility Signs Detected: {self.detection_stats['facility_signs']}
• Basic Objects Detected: {self.detection_stats['basic_objects']}
• Total Objects Detected: {self.detection_stats['total_objects']}

Performance Metrics:
• Detection Rate: {self.detection_stats['total_objects']/max(1, self.detection_stats['frames']*1.0):.3f} objects/frame
• QR Code Detection Rate: {self.detection_stats['qr_codes']/max(1, self.detection_stats['total_objects']):.2%}
• Barcode Detection Rate: {self.detection_stats['barcodes']/max(1, self.detection_stats['total_objects']):.2%}
• License Plate Detection Rate: {self.detection_stats['license_plates']/max(1, self.detection_stats['total_objects']):.2%}
• Traffic Sign Detection Rate: {self.detection_stats['traffic_signs']/max(1, self.detection_stats['total_objects']):.2%}
• Traffic Light Detection Rate: {self.detection_stats['traffic_lights']/max(1, self.detection_stats['total_objects']):.2%}
• Facility Sign Detection Rate: {self.detection_stats['facility_signs']/max(1, self.detection_stats['total_objects']):.2%}
• Basic Object Detection Rate: {self.detection_stats['basic_objects']/max(1, self.detection_stats['total_objects']):.2%}

System Capabilities:
• Multi-format code recognition (QR, barcodes)
• Vehicle identification (license plates)
• Traffic infrastructure detection (signs, lights)
• Public facility recognition (restrooms, exits, etc.)
• Basic object analysis (color, shape, text)
• Real-time processing with optimized performance
            """
            
            print(report)
            
            # 保存报告到文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"enhanced_recognition_report_{timestamp}.txt"
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"Enhanced final report saved: {report_filename}")
            
        except Exception as e:
            logger.error(f"Generate enhanced final report error: {e}")


def main():
    print("Enhanced Object Recognition GUI")
    print("=" * 50)
    print("Advanced Features:")
    print("  • QR Code Recognition - Decodes QR code content")
    print("  • Barcode Detection - Identifies various barcode formats")
    print("  • License Plate Detection - Recognizes vehicle plates")
    print("  • Traffic Sign Recognition - Identifies road signs")
    print("  • Traffic Light Detection - Detects red/yellow/green lights")
    print("  • Facility Sign Recognition - Public facility symbols")
    print("  • Basic Object Analysis - Color, shape, text detection")
    print("  • Real-time Processing - Optimized USB camera support")
    print()
    print("Controls:")
    print("  • Press 'q' or ESC to exit")
    print("  • Press 's' to save screenshot")
    print("  • Press 'r' to reset statistics")
    print("  • Press 'h' for detailed help")
    print()
    print("Detection Capabilities:")
    print("  • Codes: QR codes, various barcode formats")
    print("  • Vehicles: License plate recognition")
    print("  • Traffic: Road signs, traffic lights")
    print("  • Facilities: Restrooms, exits, parking, hospitals")
    print("  • Objects: Colors, shapes, text, surface analysis")
    print("=" * 50)
    
    gui = EnhancedObjectRecognitionGUI()
    gui.run()

if __name__ == "__main__":
    main()