#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
保存的人体姿势识别GUI代码
包含面部、手势、身体姿势多点动态捕捉功能
已解决乱码、镜像、躯干关键点显示等问题
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
        logging.FileHandler('pose_recognition.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SavedPoseRecognitionGUI:
    """保存的人体姿势识别GUI类"""
    
    def __init__(self):
        self.face_recognizer = None
        self.gesture_recognizer = None
        self.pose_recognizer = None
        self.cap = None
        self.detection_stats = {
            'faces': 0,
            'gestures': 0,
            'poses': 0,
            'frames': 0
        }
        self.start_time = time.time()
        
    def initialize_recognizers(self):
        """初始化所有识别器"""
        logger.info("Initializing recognizers...")
        
        # 初始化面部识别器
        try:
            from recognition.face_recognizer import FaceRecognizer
            self.face_recognizer = FaceRecognizer()
            logger.info("Face recognizer OK")
        except Exception as e:
            logger.warning(f"Face recognizer failed: {e}")
            
        # 初始化手势识别器
        try:
            from recognition.gesture_recognizer import GestureRecognizer
            self.gesture_recognizer = GestureRecognizer()
            logger.info("Gesture recognizer OK")
        except Exception as e:
            logger.warning(f"Gesture recognizer failed: {e}")
            
        # 初始化姿势识别器
        try:
            from recognition.pose_recognizer import PoseRecognizer
            self.pose_recognizer = PoseRecognizer()
            logger.info("Pose recognizer OK")
        except Exception as e:
            logger.warning(f"Pose recognizer failed: {e}")
    
    def initialize_camera(self):
        """初始化摄像头"""
        logger.info("Initializing camera...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        ret, frame = self.cap.read()
        if ret:
            logger.info(f"Camera OK: {frame.shape}")
        else:
            raise RuntimeError("Cannot read from camera")
    
    def detect_multimodal(self, frame):
        """多模态检测"""
        results = {
            'faces': [],
            'gestures': [],
            'poses': []
        }
        
        annotated_frame = frame.copy()
        frame_width = frame.shape[1]
        
        # 面部检测 - 避免乱码
        if self.face_recognizer:
            try:
                flipped_frame = cv2.flip(annotated_frame, 1)
                face_frame, face_results = self.face_recognizer.detect_faces(flipped_frame)
                
                for i, face in enumerate(face_results):
                    bbox = face.get('bbox', None)
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        x1_fixed = frame_width - x2
                        x2_fixed = frame_width - x1
                        
                        cv2.rectangle(annotated_frame, (x1_fixed, y1), (x2_fixed, y2), (0, 255, 0), 2)
                        label = f"FACE_{i+1}"
                        cv2.putText(annotated_frame, label, (x1_fixed, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    face_info = {
                        'bbox': bbox,
                        'identity': f'PERSON_{i+1}',
                        'confidence': face.get('confidence', 0.0)
                    }
                    results['faces'].append(face_info)
                    self.detection_stats['faces'] += 1
                    
            except Exception as e:
                logger.debug(f"Face detection error: {e}")
        
        # 手势检测 - 避免乱码
        if self.gesture_recognizer:
            try:
                flipped_frame = cv2.flip(annotated_frame, 1)
                gesture_frame, gesture_results = self.gesture_recognizer.detect_hands(flipped_frame)
                
                for i, gesture in enumerate(gesture_results):
                    landmarks = gesture.get('landmarks', [])
                    hand_label = gesture.get('hand_label', 'Unknown')
                    
                    for point in landmarks:
                        if len(point) >= 2:
                            x_fixed = frame_width - point[0]
                            y_fixed = point[1]
                            cv2.circle(annotated_frame, (int(x_fixed), int(y_fixed)), 3, (255, 0, 0), -1)
                    
                    if landmarks:
                        wrist_x = frame_width - landmarks[0][0]
                        wrist_y = landmarks[0][1]
                        
                        if hand_label == 'Left':
                            label = "L_HAND"
                        elif hand_label == 'Right':
                            label = "R_HAND"
                        else:
                            label = f"HAND_{i+1}"
                        
                        cv2.putText(annotated_frame, label, 
                                  (int(wrist_x)+10, int(wrist_y)-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    gesture_info = {
                        'hand_label': f'HAND_{i+1}',
                        'gesture': gesture.get('gesture', 'none'),
                        'landmarks': landmarks
                    }
                    results['gestures'].append(gesture_info)
                    self.detection_stats['gestures'] += 1
                    
            except Exception as e:
                logger.debug(f"Gesture detection error: {e}")
        
        # 姿势检测 - 确保显示33个关键点
        if self.pose_recognizer:
            try:
                flipped_frame = cv2.flip(annotated_frame, 1)
                pose_frame, pose_results = self.pose_recognizer.detect_pose(flipped_frame)
                
                for pose in pose_results:
                    landmarks = pose.get('landmarks', [])
                    if len(landmarks) >= 33:
                        # 绘制关键点
                        for i, point in enumerate(landmarks):
                            if len(point) >= 4 and point[3] > 0.1:  # 降低可见性阈值
                                x_fixed = frame_width - point[0]
                                y_fixed = point[1]
                                
                                # 躯干关键点用蓝色大圆点
                                if i in [11, 12, 23, 24]:  # 肩膀和臀部
                                    cv2.circle(annotated_frame, (int(x_fixed), int(y_fixed)), 10, (255, 0, 0), -1)
                                    cv2.putText(annotated_frame, f"T{i}", (int(x_fixed)+12, int(y_fixed)), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                                # 关节点用绿色中圆点
                                elif i in [13, 14, 25, 26]:  # 肘部和膝盖
                                    cv2.circle(annotated_frame, (int(x_fixed), int(y_fixed)), 6, (0, 255, 0), -1)
                                    cv2.putText(annotated_frame, f"J{i}", (int(x_fixed)+8, int(y_fixed)), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                                # 末端点用红色小圆点
                                elif i in [15, 16, 27, 28]:  # 手腕和脚踝
                                    cv2.circle(annotated_frame, (int(x_fixed), int(y_fixed)), 4, (0, 0, 255), -1)
                                # 头部特征用黄色小圆点
                                elif i <= 10:
                                    cv2.circle(annotated_frame, (int(x_fixed), int(y_fixed)), 3, (0, 255, 255), -1)
                                    if i == 0:  # 鼻子
                                        cv2.putText(annotated_frame, "NOSE", (int(x_fixed)+5, int(y_fixed)), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                                else:
                                    cv2.circle(annotated_frame, (int(x_fixed), int(y_fixed)), 2, (128, 128, 128), -1)
                        
                        # 绘制身体结构连接线
                        self.draw_body_structure_fixed(annotated_frame, landmarks)
                        
                        pose_info = {
                            'landmarks': landmarks,
                            'pose_class': pose.get('pose_class', 'unknown')
                        }
                        results['poses'].append(pose_info)
                        self.detection_stats['poses'] += 1
                        
            except Exception as e:
                logger.debug(f"Pose detection error: {e}")
        
        return annotated_frame, results
    
    def draw_body_structure_fixed(self, frame, landmarks):
        """绘制修复版身体结构连接线"""
        try:
            if len(landmarks) < 33:
                return
            
            frame_width = frame.shape[1]
            
            # 躯干主要连接线
            torso_connections = [
                (11, 12),  # 左肩-右肩
                (11, 23),  # 左肩-左臀
                (12, 24),  # 右肩-右臀
                (23, 24),  # 左臀-右臀
            ]
            
            # 四肢连接线
            limb_connections = [
                (11, 13), (13, 15),  # 左臂
                (12, 14), (14, 16),  # 右臂
                (23, 25), (25, 27),  # 左腿
                (24, 26), (26, 28),  # 右腿
            ]
            
            # 绘制躯干主线 - 粗蓝线
            for start_idx, end_idx in torso_connections:
                if (start_idx < len(landmarks) and end_idx < len(landmarks) and
                    len(landmarks[start_idx]) >= 4 and len(landmarks[end_idx]) >= 4):
                    
                    start_point = landmarks[start_idx]
                    end_point = landmarks[end_idx]
                    
                    if start_point[3] > 0.1 and end_point[3] > 0.1:
                        start_x = frame_width - start_point[0]
                        start_y = start_point[1]
                        end_x = frame_width - end_point[0]
                        end_y = end_point[1]
                        
                        start_pos = (int(start_x), int(start_y))
                        end_pos = (int(end_x), int(end_y))
                        cv2.line(frame, start_pos, end_pos, (255, 0, 0), 6)
            
            # 绘制四肢连接线 - 绿线
            for start_idx, end_idx in limb_connections:
                if (start_idx < len(landmarks) and end_idx < len(landmarks) and
                    len(landmarks[start_idx]) >= 4 and len(landmarks[end_idx]) >= 4):
                    
                    start_point = landmarks[start_idx]
                    end_point = landmarks[end_idx]
                    
                    if start_point[3] > 0.1 and end_point[3] > 0.1:
                        start_x = frame_width - start_point[0]
                        start_y = start_point[1]
                        end_x = frame_width - end_point[0]
                        end_y = end_point[1]
                        
                        start_pos = (int(start_x), int(start_y))
                        end_pos = (int(end_x), int(end_y))
                        cv2.line(frame, start_pos, end_pos, (0, 255, 0), 3)
        
        except Exception as e:
            logger.debug(f"Draw body structure error: {e}")
    
    def run(self):
        """运行GUI"""
        try:
            self.initialize_recognizers()
            self.initialize_camera()
            
            logger.info("Starting detection...")
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.detection_stats['frames'] += 1
                
                # 多模态检测
                annotated_frame, results = self.detect_multimodal(frame)
                
                # 显示统计信息
                elapsed_time = time.time() - self.start_time
                fps = self.detection_stats['frames'] / elapsed_time if elapsed_time > 0 else 0
                
                info_text = f"FPS: {fps:.1f} | Faces: {self.detection_stats['faces']} | Hands: {self.detection_stats['gestures']} | Poses: {self.detection_stats['poses']}"
                cv2.putText(annotated_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 显示帧
                cv2.imshow('Saved Pose Recognition GUI', annotated_frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # ESC
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"pose_recognition_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    logger.info(f"Screenshot saved: {filename}")
                elif key == ord('r'):
                    self.detection_stats = {'faces': 0, 'gestures': 0, 'poses': 0, 'frames': 0}
                    self.start_time = time.time()
                    logger.info("Statistics reset")
        
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            logger.info("GUI closed")

def main():
    print("Saved Pose Recognition GUI Test")
    print("=" * 40)
    print("Features:")
    print("  • Multi-face detection and recognition")
    print("  • Dual-hand gesture recognition")
    print("  • Complete 33 pose keypoints display")
    print("  • Body torso clearly visible")
    print("  • No text encoding issues")
    print()
    print("Controls:")
    print("  • Press 'q' or ESC to exit")
    print("  • Press 's' to save screenshot")
    print("  • Press 'r' to reset statistics")
    print()
    print("Keypoint Colors:")
    print("  • Blue: Body torso (shoulders hips)")
    print("  • Green: Joints (elbows knees)")
    print("  • Red: Ends (wrists ankles)")
    print("  • Yellow: Head features")
    print("=" * 40)
    
    gui = SavedPoseRecognitionGUI()
    gui.run()

if __name__ == "__main__":
    main()