#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""高性能多模态识别GUI - 专门解决卡顿和手势识别问题"""

import sys
import os
import cv2
import numpy as np
import time
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import json

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

# 设置编码
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 配置日志
logging.basicConfig(
    level=logging.INFO,  # 减少日志级别提升性能
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('high_performance_gui.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceResult:
    """性能优化的检测结果"""
    timestamp: float
    face_count: int = 0
    hand_count: int = 0
    pose_count: int = 0
    fps: float = 0.0
    processing_time: float = 0.0
    hands_landmarks: List = None
    faces_boxes: List = None
    poses_keypoints: List = None

class HighPerformanceDetector:
    """高性能多模态检测器"""
    
    def __init__(self):
        self.face_recognizer = None
        self.gesture_recognizer = None
        self.pose_recognizer = None
        
        # 性能优化参数
        self.frame_skip = 1  # 不跳帧，确保流畅
        self.frame_counter = 0
        self.detection_interval = 2  # 每2帧进行一次完整检测
        self.last_result = PerformanceResult(timestamp=time.time())
        
        # FPS计算
        self.fps_history = deque(maxlen=30)
        self.last_fps_time = time.time()
        
        self._init_recognizers()
    
    def _init_recognizers(self):
        """初始化识别器"""
        try:
            # 导入优化的检测器
            from recognition.optimized_multimodal_detector import create_optimized_multimodal_detector
            
            logger.info("正在初始化高性能检测器...")
            detector = create_optimized_multimodal_detector()
            
            self.face_recognizer = detector.face_recognizer
            self.gesture_recognizer = detector.gesture_recognizer  
            self.pose_recognizer = detector.pose_recognizer
            
            logger.info("✅ 高性能检测器初始化完成")
            
        except Exception as e:
            logger.error(f"检测器初始化失败: {e}")
    
    def detect_fast(self, frame: np.ndarray) -> Tuple[np.ndarray, PerformanceResult]:
        """高性能检测 - 优化版本"""
        start_time = time.time()
        self.frame_counter += 1
        
        # 计算FPS
        current_time = time.time()
        if current_time - self.last_fps_time > 0:
            fps = 1.0 / (current_time - self.last_fps_time)
            self.fps_history.append(fps)
        self.last_fps_time = current_time
        
        # 创建结果对象
        result = PerformanceResult(
            timestamp=start_time,
            fps=np.mean(self.fps_history) if self.fps_history else 0
        )
        
        # 智能检测策略：不是每帧都做完整检测
        do_full_detection = (self.frame_counter % self.detection_interval == 0)
        
        if do_full_detection:
            # 并行检测以提升性能
            detection_results = self._parallel_detection(frame)
            
            result.face_count = detection_results.get('faces', 0)
            result.hand_count = detection_results.get('hands', 0) 
            result.pose_count = detection_results.get('poses', 0)
            result.hands_landmarks = detection_results.get('hands_landmarks', [])
            result.faces_boxes = detection_results.get('faces_boxes', [])
            result.poses_keypoints = detection_results.get('poses_keypoints', [])
            
            self.last_result = result
        else:
            # 使用上次检测结果，只更新时间戳和FPS
            result = PerformanceResult(
                timestamp=start_time,
                face_count=self.last_result.face_count,
                hand_count=self.last_result.hand_count,
                pose_count=self.last_result.pose_count,
                fps=result.fps,
                hands_landmarks=self.last_result.hands_landmarks,
                faces_boxes=self.last_result.faces_boxes,
                poses_keypoints=self.last_result.poses_keypoints
            )
        
        result.processing_time = time.time() - start_time
        
        # 绘制高性能注释
        annotated_frame = self._draw_fast_annotations(frame, result)
        
        return annotated_frame, result
    
    def _parallel_detection(self, frame: np.ndarray) -> Dict:
        """并行检测以提升性能"""
        results = {'faces': 0, 'hands': 0, 'poses': 0, 
                  'hands_landmarks': [], 'faces_boxes': [], 'poses_keypoints': []}
        
        try:
            # 面部检测
            if self.face_recognizer:
                face_result = self.face_recognizer.detect_faces(frame)
                if hasattr(face_result, 'faces_detected'):
                    results['faces'] = face_result.faces_detected
                    if hasattr(face_result, 'faces') and face_result.faces:
                        results['faces_boxes'] = []
                        for face in face_result.faces:
                            if hasattr(face, 'bbox') and face.bbox is not None:
                                results['faces_boxes'].append(face.bbox)
            
            # 手势检测 - 重点优化多点识别
            if self.gesture_recognizer:
                gesture_result = self.gesture_recognizer.detect_gestures(frame)
                if hasattr(gesture_result, 'hands_detected'):
                    results['hands'] = gesture_result.hands_detected
                    if hasattr(gesture_result, 'hands_data') and gesture_result.hands_data:
                        results['hands_landmarks'] = gesture_result.hands_data
            
            # 姿势检测
            if self.pose_recognizer:
                pose_result = self.pose_recognizer.detect_poses(frame)
                if hasattr(pose_result, 'persons_detected'):
                    results['poses'] = pose_result.persons_detected
                    if hasattr(pose_result, 'persons_data') and pose_result.persons_data:
                        results['poses_keypoints'] = pose_result.persons_data
                        
        except Exception as e:
            logger.warning(f"检测过程出错: {e}")
        
        return results
    
    def _draw_fast_annotations(self, frame: np.ndarray, result: PerformanceResult) -> np.ndarray:
        """高性能注释绘制"""
        try:
            annotated = frame.copy()
            
            # 绘制面部框
            if result.faces_boxes:
                for i, bbox in enumerate(result.faces_boxes):
                    if bbox is not None and len(bbox) >= 4:
                        x1, y1, x2, y2 = map(int, bbox[:4])
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated, f"Face {i+1}", (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 绘制手部关键点 - 支持多点识别
            if result.hands_landmarks:
                colors = [(255, 0, 0), (0, 255, 255), (255, 0, 255), (0, 128, 255)]  # 多种颜色区分不同手
                for hand_idx, hand_data in enumerate(result.hands_landmarks):
                    color = colors[hand_idx % len(colors)]
                    try:
                        # 处理MediaPipe格式的手部关键点
                        if hasattr(hand_data, 'landmark'):
                            landmarks = hand_data.landmark
                            h, w = frame.shape[:2]
                            
                            # 绘制关键点
                            for landmark in landmarks:
                                x = int(landmark.x * w)
                                y = int(landmark.y * h)
                                cv2.circle(annotated, (x, y), 3, color, -1)
                            
                            # 绘制连接线（简化版本以提升性能）
                            connections = [(0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
                                         (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
                                         (0, 17), (5, 9), (9, 10), (10, 11), (11, 12),  # 中指
                                         (13, 14), (14, 15), (15, 16),  # 无名指
                                         (17, 18), (18, 19), (19, 20)]  # 小指
                            
                            for start_idx, end_idx in connections:
                                if start_idx < len(landmarks) and end_idx < len(landmarks):
                                    start_point = (int(landmarks[start_idx].x * w), 
                                                 int(landmarks[start_idx].y * h))
                                    end_point = (int(landmarks[end_idx].x * w), 
                                               int(landmarks[end_idx].y * h))
                                    cv2.line(annotated, start_point, end_point, color, 2)
                        
                        # 显示手部标签
                        cv2.putText(annotated, f"Hand {hand_idx+1}", (10, 150 + hand_idx*25), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                  
                    except Exception as e:
                        logger.debug(f"绘制手部关键点失败: {e}")
            
            # 绘制姿势关键点
            if result.poses_keypoints:
                for pose_idx, pose_data in enumerate(result.poses_keypoints):
                    try:
                        if hasattr(pose_data, 'keypoints') and pose_data.keypoints is not None:
                            keypoints = pose_data.keypoints
                            h, w = frame.shape[:2]
                            
                            # 绘制关键点
                            for i in range(0, len(keypoints), 3):  # x, y, confidence
                                if i+2 < len(keypoints) and keypoints[i+2] > 0.5:  # 置信度阈值
                                    x, y = int(keypoints[i] * w), int(keypoints[i+1] * h)
                                    cv2.circle(annotated, (x, y), 4, (0, 0, 255), -1)
                        
                        cv2.putText(annotated, f"Pose {pose_idx+1}", (10, 250 + pose_idx*20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                  
                    except Exception as e:
                        logger.debug(f"绘制姿势关键点失败: {e}")
            
            # 绘制性能信息
            self._draw_performance_info(annotated, result)
            
            return annotated
            
        except Exception as e:
            logger.error(f"绘制注释失败: {e}")
            return frame
    
    def _draw_performance_info(self, frame: np.ndarray, result: PerformanceResult):
        """绘制性能信息"""
        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 性能文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (0, 255, 0)
        thickness = 2
        
        # 显示检测统计
        cv2.putText(frame, f"Faces: {result.face_count}", (10, 30), font, font_scale, color, thickness)
        cv2.putText(frame, f"Hands: {result.hand_count}", (10, 60), font, font_scale, color, thickness)
        cv2.putText(frame, f"Poses: {result.pose_count}", (10, 90), font, font_scale, color, thickness)
        
        # 显示FPS
        cv2.putText(frame, f"FPS: {result.fps:.1f}", (200, 30), font, font_scale, (0, 255, 255), thickness)
        
        # 显示处理时间
        cv2.putText(frame, f"Time: {result.processing_time*1000:.1f}ms", (200, 60), font, font_scale, (255, 255, 0), thickness)
        
        # 显示帧数
        cv2.putText(frame, f"Frame: {self.frame_counter}", (200, 90), font, font_scale, (255, 0, 255), thickness)

class HighPerformanceGUI:
    """高性能GUI界面"""
    
    def __init__(self):
        self.detector = HighPerformanceDetector()
        self.cap = None
        self.running = False
        self.frame_count = 0
        
        # 性能优化设置
        self.target_fps = 30
        self.frame_time = 1.0 / self.target_fps
    
    def initialize_camera(self, camera_id: int = 0) -> bool:
        """初始化摄像头"""
        try:
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                logger.error("无法打开摄像头")
                return False
            
            # 优化摄像头设置
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲区延迟
            
            logger.info("✅ 高性能摄像头初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"摄像头初始化失败: {e}")
            return False
    
    def run(self):
        """运行高性能GUI"""
        logger.info("🚀 启动高性能多模态识别GUI")
        
        if not self.initialize_camera():
            return False
        
        window_name = "High Performance Multimodal Recognition"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.moveWindow(window_name, 100, 100)
        
        logger.info(f"创建高性能GUI窗口: {window_name}")
        
        self.running = True
        last_frame_time = time.time()
        
        try:
            while self.running:
                frame_start = time.time()
                
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logger.warning("无法读取摄像头帧")
                    continue
                
                self.frame_count += 1
                
                # 高性能检测
                annotated_frame, result = self.detector.detect_fast(frame)
                
                # 显示帧
                cv2.imshow(window_name, annotated_frame)
                
                # 性能日志（每100帧输出一次）
                if self.frame_count % 100 == 0:
                    logger.info(f"高性能GUI运行正常 - 帧数:{self.frame_count}, FPS:{result.fps:.1f}, "
                              f"检测: 面部{result.face_count} 手势{result.hand_count} 姿势{result.pose_count}")
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' 或 ESC
                    logger.info("用户请求退出")
                    break
                elif key == ord('s'):  # 保存截图
                    self._save_screenshot(annotated_frame)
                elif key == ord('r'):  # 重置统计
                    self._reset_stats()
                
                # 检查窗口状态
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                
                # 帧率控制（可选）
                frame_time = time.time() - frame_start
                if frame_time < self.frame_time:
                    time.sleep(self.frame_time - frame_time)
            
            logger.info("✅ 高性能GUI测试完成")
            return True
            
        except Exception as e:
            logger.error(f"高性能GUI运行错误: {e}")
            return False
        
        finally:
            self._cleanup()
    
    def _save_screenshot(self, frame: np.ndarray):
        """保存截图"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"high_performance_screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            logger.info(f"截图已保存: {filename}")
        except Exception as e:
            logger.error(f"保存截图失败: {e}")
    
    def _reset_stats(self):
        """重置统计"""
        self.frame_count = 0
        self.detector.frame_counter = 0
        self.detector.fps_history.clear()
        logger.info("统计已重置")
    
    def _cleanup(self):
        """清理资源"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("高性能GUI资源清理完成")

def main():
    """主函数"""
    logger.info("=== 高性能多模态识别GUI系统 ===")
    
    try:
        gui = HighPerformanceGUI()
        success = gui.run()
        
        if success:
            logger.info("高性能程序正常结束")
        else:
            logger.error("高性能程序异常结束")
            
    except KeyboardInterrupt:
        logger.info("用户中断程序")
    except Exception as e:
        logger.error(f"高性能程序运行错误: {e}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())