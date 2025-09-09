#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""稳定版多模态识别GUI - 解决躯干显示、刷新频率、乱码问题"""

import sys
import os
import cv2
import numpy as np
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

# 设置编码避免乱码
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('stable_multimodal_test.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class StableMultimodalTester:
    """稳定版多模态识别测试器"""
    
    def __init__(self):
        self.face_recognizer = None
        self.gesture_recognizer = None
        self.pose_recognizer = None
        self.cap = None
        self.running = False
        self.frame_count = 0
        
        # 简化统计
        self.stats = {
            'faces': 0,
            'hands': 0,
            'poses': 0
        }
        
        # 控制刷新频率
        self.detection_interval = 3  # 每3帧检测一次，降低刷新频率
        self.display_interval = 2    # 每2帧更新一次显示
        self.frame_skip_counter = 0
        self.display_skip_counter = 0
        
        # 缓存检测结果
        self.cached_detections = {
            'faces': [],
            'hands': [],
            'pose_keypoints': []
        }
        
    def initialize_recognizers(self):
        """初始化识别器"""
        logger.info("Initializing recognizers...")
        
        # 初始化面部识别器
        try:
            from recognition.face_recognizer import FaceRecognizer
            self.face_recognizer = FaceRecognizer(min_detection_confidence=0.6)
            logger.info("Face recognizer OK")
        except Exception as e:
            logger.warning(f"Face recognizer failed: {e}")
        
        # 初始化手势识别器
        try:
            from recognition.gesture_recognizer import GestureRecognizer
            self.gesture_recognizer = GestureRecognizer(
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5,
                max_num_hands=2
            )
            logger.info("Gesture recognizer OK")
        except Exception as e:
            logger.warning(f"Gesture recognizer failed: {e}")
        
        # 初始化姿势识别器
        try:
            from recognition.pose_recognizer import PoseRecognizer
            self.pose_recognizer = PoseRecognizer(
                min_detection_confidence=0.4,  # 降低阈值确保检测到更多点
                min_tracking_confidence=0.4,
                model_complexity=1
            )
            logger.info("Pose recognizer OK")
        except Exception as e:
            logger.warning(f"Pose recognizer failed: {e}")
        
        return True
    
    def initialize_camera(self):
        """初始化摄像头"""
        logger.info("Initializing camera...")
        
        try:
            self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                logger.error("Cannot open camera")
                return False
            
            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # 测试读取
            ret, frame = self.cap.read()
            if ret and frame is not None:
                logger.info(f"Camera OK: {frame.shape}")
                return True
            else:
                logger.error("Camera test failed")
                return False
                
        except Exception as e:
            logger.error(f"Camera init failed: {e}")
            return False
    
    def detect_faces_stable(self, frame):
        """稳定版面部检测"""
        faces_info = []
        
        if self.face_recognizer:
            try:
                annotated_frame, face_results = self.face_recognizer.detect_faces(frame)
                
                for i, face in enumerate(face_results):
                    face_info = {
                        'id': i,
                        'bbox': face.get('bbox', None),
                        'identity': 'Person' if face.get('identity', 'Unknown') != 'Unknown' else 'Unknown',
                        'confidence': face.get('confidence', 0.0)
                    }
                    faces_info.append(face_info)
                    self.stats['faces'] += 1
                
                return annotated_frame, faces_info
                
            except Exception as e:
                logger.debug(f"Face detection error: {e}")
        
        return frame, faces_info
    
    def detect_gestures_stable(self, frame):
        """稳定版手势检测"""
        hands_info = []
        
        if self.gesture_recognizer:
            try:
                annotated_frame, gesture_results = self.gesture_recognizer.detect_hands(frame)
                
                for i, hand in enumerate(gesture_results):
                    hand_info = {
                        'id': i,
                        'hand_label': hand.get('hand_label', 'Unknown'),
                        'gesture': hand.get('gesture', 'none'),
                        'landmarks': hand.get('landmarks', [])
                    }
                    hands_info.append(hand_info)
                    self.stats['hands'] += 1
                
                return annotated_frame, hands_info
                
            except Exception as e:
                logger.debug(f"Gesture detection error: {e}")
        
        return frame, hands_info
    
    def detect_pose_stable(self, frame):
        """稳定版姿势检测 - 确保显示躯干关键点"""
        pose_keypoints = []
        
        if self.pose_recognizer:
            try:
                annotated_frame, pose_result = self.pose_recognizer.detect_pose(frame)
                
                if pose_result and pose_result.get('pose_detected', False):
                    landmarks = pose_result.get('landmarks', [])
                    
                    if landmarks and len(landmarks) >= 33:
                        # 确保显示所有33个关键点，特别是躯干部分
                        keypoint_names = [
                            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
                            'right_eye_inner', 'right_eye', 'right_eye_outer',
                            'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
                            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                            'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
                            'left_index', 'right_index', 'left_thumb', 'right_thumb',
                            'left_hip', 'right_hip', 'left_knee', 'right_knee',
                            'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
                            'left_foot_index', 'right_foot_index'
                        ]
                        
                        # 重新绘制关键点，确保躯干可见
                        for i, landmark in enumerate(landmarks):
                            if i < len(keypoint_names) and len(landmark) >= 4:
                                x, y, z, visibility = landmark
                                
                                # 降低可见性阈值，确保更多点显示
                                if visibility > 0.2:
                                    keypoint_info = {
                                        'name': keypoint_names[i],
                                        'position': (int(x), int(y)),
                                        'visibility': visibility
                                    }
                                    pose_keypoints.append(keypoint_info)
                                    
                                    # 躯干关键点特殊处理
                                    if keypoint_names[i] in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']:
                                        # 躯干用蓝色大圆点
                                        cv2.circle(annotated_frame, (int(x), int(y)), 6, (255, 0, 0), -1)
                                        cv2.putText(annotated_frame, keypoint_names[i][:4], 
                                                  (int(x)+8, int(y)-8), cv2.FONT_HERSHEY_SIMPLEX, 
                                                  0.4, (255, 0, 0), 1)
                                    elif keypoint_names[i] in ['left_elbow', 'right_elbow', 'left_knee', 'right_knee']:
                                        # 关节用绿色中圆点
                                        cv2.circle(annotated_frame, (int(x), int(y)), 4, (0, 255, 0), -1)
                                    elif keypoint_names[i] in ['left_wrist', 'right_wrist', 'left_ankle', 'right_ankle']:
                                        # 末端用红色小圆点
                                        cv2.circle(annotated_frame, (int(x), int(y)), 3, (0, 0, 255), -1)
                                    elif 'eye' in keypoint_names[i] or 'ear' in keypoint_names[i] or keypoint_names[i] == 'nose':
                                        # 头部用黄色小圆点
                                        cv2.circle(annotated_frame, (int(x), int(y)), 2, (0, 255, 255), -1)
                                    else:
                                        # 其他用紫色小圆点
                                        cv2.circle(annotated_frame, (int(x), int(y)), 2, (255, 0, 255), -1)
                        
                        # 绘制躯干连接线
                        self.draw_body_connections(annotated_frame, landmarks)
                    
                    self.stats['poses'] += 1
                
                return annotated_frame, pose_keypoints
                
            except Exception as e:
                logger.debug(f"Pose detection error: {e}")
        
        return frame, pose_keypoints
    
    def draw_body_connections(self, frame, landmarks):
        """绘制身体连接线，突出躯干结构"""
        try:
            if len(landmarks) < 33:
                return
            
            # 躯干连接线（肩膀-臀部）
            connections = [
                (11, 12),  # 左肩-右肩
                (11, 23),  # 左肩-左臀
                (12, 24),  # 右肩-右臀
                (23, 24),  # 左臀-右臀
                (11, 13),  # 左肩-左肘
                (13, 15),  # 左肘-左腕
                (12, 14),  # 右肩-右肘
                (14, 16),  # 右肘-右腕
                (23, 25),  # 左臀-左膝
                (25, 27),  # 左膝-左踝
                (24, 26),  # 右臀-右膝
                (26, 28),  # 右膝-右踝
            ]
            
            for start_idx, end_idx in connections:
                if (start_idx < len(landmarks) and end_idx < len(landmarks) and
                    len(landmarks[start_idx]) >= 4 and len(landmarks[end_idx]) >= 4):
                    
                    start_point = landmarks[start_idx]
                    end_point = landmarks[end_idx]
                    
                    if start_point[3] > 0.3 and end_point[3] > 0.3:  # 可见性检查
                        start_pos = (int(start_point[0]), int(start_point[1]))
                        end_pos = (int(end_point[0]), int(end_point[1]))
                        
                        # 躯干主线用粗蓝线
                        if (start_idx, end_idx) in [(11, 12), (11, 23), (12, 24), (23, 24)]:
                            cv2.line(frame, start_pos, end_pos, (255, 0, 0), 3)
                        else:
                            cv2.line(frame, start_pos, end_pos, (0, 255, 0), 2)
        
        except Exception as e:
            logger.debug(f"Draw connections error: {e}")
    
    def detect_multimodal_stable(self, frame):
        """稳定版多模态检测"""
        # 控制检测频率
        self.frame_skip_counter += 1
        should_detect = (self.frame_skip_counter % self.detection_interval == 0)
        
        if should_detect:
            # 执行检测
            annotated_frame = frame.copy()
            
            # 面部检测
            face_frame, faces = self.detect_faces_stable(annotated_frame)
            annotated_frame = face_frame
            
            # 手势检测
            gesture_frame, hands = self.detect_gestures_stable(annotated_frame)
            annotated_frame = gesture_frame
            
            # 姿势检测
            pose_frame, pose_keypoints = self.detect_pose_stable(annotated_frame)
            annotated_frame = pose_frame
            
            # 更新缓存
            self.cached_detections = {
                'faces': faces,
                'hands': hands,
                'pose_keypoints': pose_keypoints
            }
            
            return annotated_frame, self.cached_detections
        else:
            # 使用缓存结果，减少计算
            return self.draw_cached_results(frame), self.cached_detections
    
    def draw_cached_results(self, frame):
        """绘制缓存的检测结果"""
        annotated_frame = frame.copy()
        
        # 绘制缓存的面部框
        for face in self.cached_detections.get('faces', []):
            if face.get('bbox'):
                x1, y1, x2, y2 = face['bbox']
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Face {face['id']}: {face['identity']}"
                cv2.putText(annotated_frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 绘制缓存的手部关键点
        for hand in self.cached_detections.get('hands', []):
            landmarks = hand.get('landmarks', [])
            for point in landmarks:
                if len(point) >= 2:
                    cv2.circle(annotated_frame, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)
        
        # 绘制缓存的姿势关键点
        for keypoint in self.cached_detections.get('pose_keypoints', []):
            pos = keypoint.get('position')
            name = keypoint.get('name', '')
            if pos:
                if name in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']:
                    cv2.circle(annotated_frame, pos, 6, (255, 0, 0), -1)
                elif name in ['left_elbow', 'right_elbow', 'left_knee', 'right_knee']:
                    cv2.circle(annotated_frame, pos, 4, (0, 255, 0), -1)
                else:
                    cv2.circle(annotated_frame, pos, 2, (0, 0, 255), -1)
        
        return annotated_frame
    
    def draw_stable_overlay(self, frame, fps):
        """绘制稳定信息覆盖层"""
        # 控制显示更新频率
        self.display_skip_counter += 1
        if self.display_skip_counter % self.display_interval != 0:
            return  # 跳过显示更新
        
        # 顶部状态栏
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        
        # 基本信息
        current_time = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, f"Time: {current_time}", (10, 20), font, 0.5, color, 1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 40), font, 0.5, color, 1)
        
        # 检测统计
        stats_text = f"Faces: {len(self.cached_detections.get('faces', []))} | Hands: {len(self.cached_detections.get('hands', []))} | Pose Points: {len(self.cached_detections.get('pose_keypoints', []))}"
        cv2.putText(frame, stats_text, (150, 20), font, 0.4, color, 1)
        
        # 颜色说明
        legend_text = "Blue: Torso | Green: Joints | Red: Ends | Yellow: Head"
        cv2.putText(frame, legend_text, (150, 40), font, 0.4, (200, 200, 200), 1)
        
        # 控制说明
        control_text = "Controls: [Q]uit | [S]ave | [R]eset"
        cv2.putText(frame, control_text, (10, frame.shape[0] - 10), font, 0.4, (0, 255, 0), 1)
    
    def save_screenshot(self, frame):
        """保存截图"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stable_multimodal_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            logger.info(f"Screenshot saved: {filename}")
        except Exception as e:
            logger.error(f"Save failed: {e}")
    
    def reset_stats(self):
        """重置统计"""
        self.stats = {'faces': 0, 'hands': 0, 'poses': 0}
        self.frame_count = 0
        self.cached_detections = {'faces': [], 'hands': [], 'pose_keypoints': []}
        logger.info("Stats reset")
    
    def run_test(self):
        """运行稳定版测试"""
        logger.info("Starting stable multimodal GUI test")
        
        # 初始化识别器
        if not self.initialize_recognizers():
            logger.error("Recognizer init failed")
            return False
        
        # 初始化摄像头
        if not self.initialize_camera():
            logger.error("Camera init failed")
            return False
        
        # 创建窗口
        window_name = "Stable Multimodal Recognition - Body Torso Visible"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        logger.info("Starting stable detection...")
        
        self.running = True
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        try:
            while self.running:
                # 读取摄像头帧
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    logger.warning("Cannot read frame")
                    continue
                
                self.frame_count += 1
                fps_counter += 1
                
                # 执行稳定版多模态检测
                try:
                    annotated_frame, detections = self.detect_multimodal_stable(frame)
                except Exception as e:
                    logger.warning(f"Detection failed: {e}")
                    annotated_frame = frame
                
                # 计算FPS
                current_time = time.time()
                if current_time - fps_start_time >= 1.0:
                    current_fps = fps_counter / (current_time - fps_start_time)
                    fps_counter = 0
                    fps_start_time = current_time
                
                # 绘制稳定信息覆盖层
                self.draw_stable_overlay(annotated_frame, current_fps)
                
                # 显示帧
                cv2.imshow(window_name, annotated_frame)
                
                # 处理键盘输入
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' 或 ESC 退出
                    logger.info("User exit")
                    break
                elif key == ord('s'):  # 's' 保存截图
                    self.save_screenshot(annotated_frame)
                elif key == ord('r'):  # 'r' 重置统计
                    self.reset_stats()
                
                # 检查窗口是否被关闭
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("Window closed")
                    break
            
            logger.info("Test completed")
            return True
            
        except Exception as e:
            logger.error(f"Test error: {e}")
            return False
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        logger.info("Cleaning up...")
        
        self.running = False
        
        # 释放摄像头
        if self.cap:
            self.cap.release()
        
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()
        
        logger.info("Cleanup done")


def main():
    """主函数"""
    print("Stable Multimodal Recognition GUI Test")
    print("=" * 45)
    print("Features:")
    print("  • Multi-face detection")
    print("  • Dual-hand tracking")
    print("  • Complete 33 pose keypoints")
    print("  • Body torso clearly visible")
    print("  • Stable refresh rate")
    print("  • No text encoding issues")
    print()
    print("Controls:")
    print("  • Press 'q' or ESC to exit")
    print("  • Press 's' to save screenshot")
    print("  • Press 'r' to reset stats")
    print()
    print("Keypoint Colors:")
    print("  • Blue: Body torso (shoulders, hips)")
    print("  • Green: Joints (elbows, knees)")
    print("  • Red: Ends (wrists, ankles)")
    print("  • Yellow: Head features")
    print("=" * 45)
    
    try:
        tester = StableMultimodalTester()
        success = tester.run_test()
        
        if success:
            print("\nTest completed successfully!")
            sys.exit(0)
        else:
            print("\nTest failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nTest interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\nTest exception: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()