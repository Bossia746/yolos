#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
静物识别GUI - 基于OpenCV和深度学习
识别日常物品的颜色、外观、形状、字帖等
"""

import cv2
import numpy as np
import os
import sys
import logging
import time
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
        logging.FileHandler('object_recognition.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ObjectRecognitionGUI:
    """静物识别GUI类"""
    
    def __init__(self):
        self.object_recognizer = None
        self.cap = None
        self.detection_stats = {
            'objects': 0,
            'colors': 0,
            'shapes': 0,
            'texts': 0,
            'frames': 0
        }
        self.start_time = time.time()
        self.detection_interval = 8  # 每8帧检测一次，降低刷新频率
        self.display_interval = 2   # 每2帧更新显示一次
        self.frame_count = 0
        self.last_results = None    # 缓存上次检测结果
        self.focus_stabilize_frames = 30  # 对焦稳定帧数
        
    def initialize_recognizer(self):
        """初始化静物识别器"""
        logger.info("Initializing object recognizer...")
        
        try:
            from recognition.object_recognizer import ObjectRecognizer
            self.object_recognizer = ObjectRecognizer()
            logger.info("Object recognizer OK")
        except Exception as e:
            logger.error(f"Object recognizer failed: {e}")
            raise
    
    def initialize_camera(self):
        """初始化摄像头 - 支持USB外接摄像头"""
        logger.info("Initializing camera...")
        
        # 尝试多个摄像头索引，优先使用USB外接摄像头
        camera_indices = [1, 2, 0, 3]  # 1,2通常是USB摄像头，0是内置摄像头
        
        for camera_index in camera_indices:
            logger.info(f"Trying camera index {camera_index}...")
            self.cap = cv2.VideoCapture(camera_index)
            
            if self.cap.isOpened():
                # 测试是否能读取帧
                ret, frame = self.cap.read()
                if ret:
                    logger.info(f"Camera {camera_index} OK: {frame.shape}")
                    break
                else:
                    self.cap.release()
            else:
                if self.cap:
                    self.cap.release()
        
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Cannot open any camera (tried indices: 0,1,2,3)")
        
        # 设置摄像头参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # 验证最终设置
        ret, frame = self.cap.read()
        if ret:
            logger.info(f"Final camera setup OK: {frame.shape}")
        else:
            raise RuntimeError("Cannot read from selected camera")
    
    def detect_objects(self, frame):
        """检测静物对象 - 优化对焦和刷新频率"""
        results = {
            'objects': [],
            'summary': {
                'total_objects': 0,
                'colors_detected': set(),
                'shapes_detected': set(),
                'texts_detected': []
            }
        }
        
        try:
            # 前30帧用于对焦稳定，不进行检测
            if self.frame_count < self.focus_stabilize_frames:
                # 显示对焦提示
                focus_frame = frame.copy()
                remaining_frames = self.focus_stabilize_frames - self.frame_count
                focus_text = f"Camera focusing... {remaining_frames} frames remaining"
                cv2.putText(focus_frame, focus_text, (50, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(focus_frame, "Please hold objects steady", (50, 80), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                return focus_frame, results
            
            # 每隔8帧进行一次检测，大幅降低刷新频率
            if self.frame_count % self.detection_interval == 0:
                annotated_frame, object_results = self.object_recognizer.detect_objects(frame)
                
                for obj in object_results:
                    # 统计信息
                    results['summary']['total_objects'] += 1
                    results['summary']['colors_detected'].add(obj['color']['name'])
                    results['summary']['shapes_detected'].add(obj['shape']['name'])
                    
                    if obj.get('text') and obj['text'].get('text'):
                        results['summary']['texts_detected'].append(obj['text']['text'])
                    
                    # 更新统计
                    self.detection_stats['objects'] += 1
                    if obj['color']['name'] != 'unknown':
                        self.detection_stats['colors'] += 1
                    if obj['shape']['name'] != 'unknown':
                        self.detection_stats['shapes'] += 1
                    if obj.get('text') and obj['text'].get('text'):
                        self.detection_stats['texts'] += 1
                
                results['objects'] = object_results
                self.last_results = (annotated_frame, results)  # 缓存结果
                return annotated_frame, results
            
            # 非检测帧，使用缓存结果或原始帧
            elif self.last_results is not None and self.frame_count % self.display_interval == 0:
                # 每2帧更新一次显示，使用缓存的检测结果
                return self.last_results[0], self.last_results[1]
            else:
                # 其他帧直接返回原始帧，减少处理
                return frame, results
                
        except Exception as e:
            logger.debug(f"Object detection error: {e}")
            return frame, results
    
    def draw_summary_info(self, frame, results):
        """绘制汇总信息"""
        try:
            summary = results['summary']
            
            # 准备信息文本
            info_lines = [
                f"Objects: {summary['total_objects']}",
                f"Colors: {len(summary['colors_detected'])}",
                f"Shapes: {len(summary['shapes_detected'])}",
                f"Texts: {len(summary['texts_detected'])}"
            ]
            
            # 显示检测到的颜色
            if summary['colors_detected']:
                colors_str = ', '.join(list(summary['colors_detected'])[:3])
                if len(summary['colors_detected']) > 3:
                    colors_str += '...'
                info_lines.append(f"Color Types: {colors_str}")
            
            # 显示检测到的形状
            if summary['shapes_detected']:
                shapes_str = ', '.join(list(summary['shapes_detected'])[:3])
                if len(summary['shapes_detected']) > 3:
                    shapes_str += '...'
                info_lines.append(f"Shape Types: {shapes_str}")
            
            # 显示检测到的文字
            if summary['texts_detected']:
                text_str = summary['texts_detected'][0][:15]
                if len(text_str) < len(summary['texts_detected'][0]):
                    text_str += '...'
                info_lines.append(f"Text Found: {text_str}")
            
            # 绘制信息面板背景
            panel_height = len(info_lines) * 20 + 20
            cv2.rectangle(frame, (10, 10), (300, panel_height), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (300, panel_height), (255, 255, 255), 2)
            
            # 绘制信息文本
            for i, line in enumerate(info_lines):
                y_pos = 30 + i * 20
                cv2.putText(frame, line, (15, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        except Exception as e:
            logger.debug(f"Draw summary info error: {e}")
    
    def draw_detection_guide(self, frame):
        """绘制检测指南"""
        try:
            guide_lines = [
                "Object Detection Guide:",
                "• Place objects clearly in view",
                "• Ensure good lighting",
                "• Avoid cluttered background",
                "• Hold objects steady for text detection"
            ]
            
            # 绘制指南面板
            start_y = frame.shape[0] - len(guide_lines) * 20 - 20
            panel_width = 350
            
            cv2.rectangle(frame, (10, start_y), (panel_width, frame.shape[0] - 10), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, start_y), (panel_width, frame.shape[0] - 10), (200, 200, 200), 1)
            
            for i, line in enumerate(guide_lines):
                y_pos = start_y + 15 + i * 20
                color = (0, 255, 255) if i == 0 else (200, 200, 200)
                font_scale = 0.5 if i == 0 else 0.4
                cv2.putText(frame, line, (15, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
            
        except Exception as e:
            logger.debug(f"Draw detection guide error: {e}")
    
    def run(self):
        """运行GUI"""
        try:
            self.initialize_recognizer()
            self.initialize_camera()
            
            logger.info("Starting object detection...")
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.detection_stats['frames'] += 1
                self.frame_count += 1
                
                # 静物检测
                annotated_frame, results = self.detect_objects(frame)
                
                # 绘制汇总信息
                self.draw_summary_info(annotated_frame, results)
                
                # 绘制检测指南
                self.draw_detection_guide(annotated_frame)
                
                # 显示统计信息
                elapsed_time = time.time() - self.start_time
                fps = self.detection_stats['frames'] / elapsed_time if elapsed_time > 0 else 0
                
                stats_text = f"FPS: {fps:.1f} | Objects: {self.detection_stats['objects']} | Colors: {self.detection_stats['colors']} | Shapes: {self.detection_stats['shapes']} | Texts: {self.detection_stats['texts']}"
                
                # 绘制状态栏
                cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 25), (0, 0, 0), -1)
                cv2.putText(annotated_frame, stats_text, (5, 18), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # 显示帧
                cv2.imshow('Object Recognition GUI', annotated_frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # ESC
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"object_detection_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    logger.info(f"Screenshot saved: {filename}")
                elif key == ord('r'):
                    self.detection_stats = {'objects': 0, 'colors': 0, 'shapes': 0, 'texts': 0, 'frames': 0}
                    self.start_time = time.time()
                    self.frame_count = 0
                    logger.info("Statistics reset")
                elif key == ord('h'):
                    self._show_help()
        
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            logger.info("Object recognition GUI closed")
            
            # 生成最终报告
            self._generate_final_report()
    
    def _show_help(self):
        """显示帮助信息"""
        help_text = """
Object Recognition GUI Help
==========================

Controls:
• 'q' or ESC - Exit program
• 's' - Save screenshot
• 'r' - Reset statistics
• 'h' - Show this help

Detection Features:
• Color Recognition - Identifies dominant colors
• Shape Detection - Recognizes geometric shapes
• Text Detection - Finds text in objects
• Appearance Analysis - Analyzes surface properties

Tips for Better Detection:
• Use good lighting
• Place objects against plain background
• Keep objects steady for text detection
• Ensure objects are clearly visible
        """
        print(help_text)
    
    def _generate_final_report(self):
        """生成最终报告"""
        try:
            elapsed_time = time.time() - self.start_time
            avg_fps = self.detection_stats['frames'] / elapsed_time if elapsed_time > 0 else 0
            
            report = f"""
Object Recognition Session Report
================================
Duration: {elapsed_time:.1f} seconds
Average FPS: {avg_fps:.1f}
Total Frames: {self.detection_stats['frames']}

Detection Results:
• Objects Detected: {self.detection_stats['objects']}
• Colors Identified: {self.detection_stats['colors']}
• Shapes Recognized: {self.detection_stats['shapes']}
• Texts Found: {self.detection_stats['texts']}

Performance:
• Detection Rate: {self.detection_stats['objects']/max(1, self.detection_stats['frames']*1.0):.3f} objects/frame
• Color Recognition Rate: {self.detection_stats['colors']/max(1, self.detection_stats['objects']):.2%}
• Shape Recognition Rate: {self.detection_stats['shapes']/max(1, self.detection_stats['objects']):.2%}
• Text Detection Rate: {self.detection_stats['texts']/max(1, self.detection_stats['objects']):.2%}
            """
            
            print(report)
            
            # 保存报告到文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"object_recognition_report_{timestamp}.txt"
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"Final report saved: {report_filename}")
            
        except Exception as e:
            logger.error(f"Generate final report error: {e}")


def main():
    print("Object Recognition GUI")
    print("=" * 40)
    print("Features:")
    print("  • Color Recognition - Identifies dominant colors")
    print("  • Shape Detection - Recognizes geometric shapes")
    print("  • Text Detection - Finds text in objects")
    print("  • Appearance Analysis - Analyzes surface properties")
    print("  • Real-time Processing - Live camera feed")
    print()
    print("Controls:")
    print("  • Press 'q' or ESC to exit")
    print("  • Press 's' to save screenshot")
    print("  • Press 'r' to reset statistics")
    print("  • Press 'h' for help")
    print()
    print("Detection Categories:")
    print("  • Colors: red, orange, yellow, green, blue, purple, pink, white, black, gray")
    print("  • Shapes: circle, triangle, square, rectangle, pentagon, hexagon, polygon")
    print("  • Surfaces: smooth, rough, textured, regular")
    print("  • Text: English and Chinese character recognition")
    print("=" * 40)
    
    gui = ObjectRecognitionGUI()
    gui.run()

if __name__ == "__main__":
    main()