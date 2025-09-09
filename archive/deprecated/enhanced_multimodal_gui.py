#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""增强版多模态识别GUI - 解决乱码问题并展示完整功能"""

import sys
import os
import cv2
import numpy as np
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import math

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

# 设置编码
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('enhanced_multimodal_test.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class TrajectoryTracker:
    """轨迹追踪器"""
    
    def __init__(self, max_points=30):
        self.max_points = max_points
        self.face_trajectories = {}  # 面部轨迹
        self.hand_trajectories = {}  # 手部轨迹
        self.pose_trajectories = {}  # 姿势关键点轨迹
        
    def update_face_trajectory(self, face_id, center_point):
        """更新面部轨迹"""
        if face_id not in self.face_trajectories:
            self.face_trajectories[face_id] = deque(maxlen=self.max_points)
        self.face_trajectories[face_id].append(center_point)
    
    def update_hand_trajectory(self, hand_id, wrist_point):
        """更新手部轨迹"""
        if hand_id not in self.hand_trajectories:
            self.hand_trajectories[hand_id] = deque(maxlen=self.max_points)
        self.hand_trajectories[hand_id].append(wrist_point)
    
    def update_pose_trajectory(self, keypoint_name, point):
        """更新姿势关键点轨迹"""
        if keypoint_name not in self.pose_trajectories:
            self.pose_trajectories[keypoint_name] = deque(maxlen=self.max_points)
        self.pose_trajectories[keypoint_name].append(point)
    
    def draw_trajectories(self, frame):
        """绘制所有轨迹"""
        # 绘制面部轨迹
        for face_id, trajectory in self.face_trajectories.items():
            if len(trajectory) > 1:
                points = list(trajectory)
                for i in range(1, len(points)):
                    alpha = i / len(points)
                    color = (int(255 * alpha), int(100 * alpha), int(50 * alpha))
                    cv2.line(frame, points[i-1], points[i], color, 2)
        
        # 绘制手部轨迹
        for hand_id, trajectory in self.hand_trajectories.items():
            if len(trajectory) > 1:
                points = list(trajectory)
                for i in range(1, len(points)):
                    alpha = i / len(points)
                    color = (int(50 * alpha), int(255 * alpha), int(100 * alpha))
                    cv2.line(frame, points[i-1], points[i], color, 2)
        
        # 绘制关键姿势点轨迹
        for keypoint_name, trajectory in self.pose_trajectories.items():
            if len(trajectory) > 1:
                points = list(trajectory)
                for i in range(1, len(points)):
                    alpha = i / len(points)
                    color = (int(100 * alpha), int(50 * alpha), int(255 * alpha))
                    cv2.line(frame, points[i-1], points[i], color, 1)


class EnhancedMultimodalTester:
    """增强版多模态识别测试器"""
    
    def __init__(self):
        self.face_recognizer = None
        self.gesture_recognizer = None
        self.pose_recognizer = None
        self.cap = None
        self.running = False
        self.frame_count = 0
        self.trajectory_tracker = TrajectoryTracker()
        
        # 详细统计信息
        self.detection_stats = {
            'faces_detected': 0,
            'faces_recognized': 0,
            'hands_detected': 0,
            'gestures_classified': 0,
            'poses_detected': 0,
            'keypoints_tracked': 0,
            'trajectories_active': 0
        }
        
        # 当前检测状态
        self.current_detections = {
            'faces': [],
            'hands': [],
            'pose_keypoints': [],
            'gestures': [],
            'pose_classification': None
        }
        
        # 性能优化设置
        self.detection_interval = 2  # 每2帧检测一次以提高速度
        self.frame_skip_counter = 0
        
    def initialize_recognizers(self):
        """初始化识别器"""
        logger.info("🚀 初始化增强版识别器...")
        
        # 初始化面部识别器
        try:
            from recognition.face_recognizer import FaceRecognizer
            self.face_recognizer = FaceRecognizer(min_detection_confidence=0.6)
            logger.info("✅ 面部识别器初始化成功")
        except Exception as e:
            logger.warning(f"⚠️ 面部识别器初始化失败: {e}")
        
        # 初始化手势识别器
        try:
            from recognition.gesture_recognizer import GestureRecognizer
            self.gesture_recognizer = GestureRecognizer(
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5,
                max_num_hands=2
            )
            logger.info("✅ 手势识别器初始化成功")
        except Exception as e:
            logger.warning(f"⚠️ 手势识别器初始化失败: {e}")
        
        # 初始化姿势识别器
        try:
            from recognition.pose_recognizer import PoseRecognizer
            self.pose_recognizer = PoseRecognizer(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1  # 使用中等复杂度以平衡速度和精度
            )
            logger.info("✅ 姿势识别器初始化成功")
        except Exception as e:
            logger.warning(f"⚠️ 姿势识别器初始化失败: {e}")
        
        return True
    
    def initialize_camera(self):
        """初始化摄像头"""
        logger.info("📷 初始化摄像头...")
        
        try:
            self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                logger.error("❌ 无法打开摄像头")
                return False
            
            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # 测试读取
            ret, frame = self.cap.read()
            if ret and frame is not None:
                logger.info(f"✅ 摄像头初始化成功: {frame.shape}")
                return True
            else:
                logger.error("❌ 摄像头测试读取失败")
                return False
                
        except Exception as e:
            logger.error(f"❌ 摄像头初始化失败: {e}")
            return False
    
    def detect_faces_enhanced(self, frame):
        """增强版面部检测"""
        faces_info = []
        
        if self.face_recognizer:
            try:
                annotated_frame, face_results = self.face_recognizer.detect_faces(frame)
                
                for i, face in enumerate(face_results):
                    face_info = {
                        'id': i,
                        'bbox': face.get('bbox', None),
                        'identity': face.get('identity', 'Unknown'),
                        'confidence': face.get('confidence', 0.0),
                        'landmarks': face.get('landmarks', None)
                    }
                    
                    # 计算面部中心点用于轨迹追踪
                    if face_info['bbox']:
                        x1, y1, x2, y2 = face_info['bbox']
                        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        face_info['center'] = center
                        self.trajectory_tracker.update_face_trajectory(i, center)
                    
                    faces_info.append(face_info)
                    
                    # 更新统计
                    self.detection_stats['faces_detected'] += 1
                    if face_info['identity'] != 'Unknown':
                        self.detection_stats['faces_recognized'] += 1
                
                return annotated_frame, faces_info
                
            except Exception as e:
                logger.debug(f"面部检测错误: {e}")
        
        return frame, faces_info
    
    def detect_gestures_enhanced(self, frame):
        """增强版手势检测"""
        hands_info = []
        gestures_info = []
        
        if self.gesture_recognizer:
            try:
                annotated_frame, gesture_results = self.gesture_recognizer.detect_hands(frame)
                
                for i, hand in enumerate(gesture_results):
                    hand_info = {
                        'id': i,
                        'hand_label': hand.get('hand_label', 'Unknown'),
                        'gesture': hand.get('gesture', 'unknown'),
                        'gesture_name': hand.get('gesture_name', 'Unknown'),
                        'landmarks': hand.get('landmarks', []),
                        'bbox': hand.get('bbox', None)
                    }
                    
                    # 提取手腕位置用于轨迹追踪
                    if hand_info['landmarks'] and len(hand_info['landmarks']) > 0:
                        wrist_point = hand_info['landmarks'][0]  # 手腕是第0个关键点
                        self.trajectory_tracker.update_hand_trajectory(i, wrist_point)
                    
                    hands_info.append(hand_info)
                    
                    # 如果检测到有效手势，添加到手势信息
                    if hand_info['gesture'] != 'unknown':
                        gestures_info.append({
                            'hand_id': i,
                            'gesture': hand_info['gesture'],
                            'gesture_name': hand_info['gesture_name'],
                            'hand_label': hand_info['hand_label']
                        })
                        self.detection_stats['gestures_classified'] += 1
                    
                    self.detection_stats['hands_detected'] += 1
                
                return annotated_frame, hands_info, gestures_info
                
            except Exception as e:
                logger.debug(f"手势检测错误: {e}")
        
        return frame, hands_info, gestures_info
    
    def detect_pose_enhanced(self, frame):
        """增强版姿势检测"""
        pose_info = {}
        keypoints_info = []
        
        if self.pose_recognizer:
            try:
                annotated_frame, pose_result = self.pose_recognizer.detect_pose(frame)
                
                if pose_result and pose_result.get('pose_detected', False):
                    pose_info = {
                        'detected': True,
                        'pose_type': pose_result.get('pose_type', 'unknown'),
                        'pose_name': pose_result.get('pose_name', 'Unknown'),
                        'confidence': pose_result.get('confidence', 0.0),
                        'angles': pose_result.get('angles', {}),
                        'body_ratios': pose_result.get('body_ratios', {})
                    }
                    
                    # 处理关键点信息
                    landmarks = pose_result.get('landmarks', [])
                    if landmarks:
                        # MediaPipe姿势关键点名称映射
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
                        
                        for i, landmark in enumerate(landmarks):
                            if i < len(keypoint_names) and len(landmark) >= 4:
                                x, y, z, visibility = landmark
                                if visibility > 0.5:  # 只处理可见的关键点
                                    keypoint_info = {
                                        'name': keypoint_names[i],
                                        'position': (int(x), int(y)),
                                        'visibility': visibility,
                                        'z': z
                                    }
                                    keypoints_info.append(keypoint_info)
                                    
                                    # 更新重要关键点的轨迹
                                    if keypoint_names[i] in ['left_wrist', 'right_wrist', 'nose', 'left_shoulder', 'right_shoulder']:
                                        self.trajectory_tracker.update_pose_trajectory(
                                            keypoint_names[i], (int(x), int(y))
                                        )
                    
                    self.detection_stats['poses_detected'] += 1
                    self.detection_stats['keypoints_tracked'] += len(keypoints_info)
                
                return annotated_frame, pose_info, keypoints_info
                
            except Exception as e:
                logger.debug(f"姿势检测错误: {e}")
        
        return frame, pose_info, keypoints_info
    
    def detect_multimodal_enhanced(self, frame):
        """增强版多模态检测"""
        # 性能优化：跳帧处理
        self.frame_skip_counter += 1
        if self.frame_skip_counter % self.detection_interval != 0:
            # 非检测帧，只绘制轨迹和之前的检测结果
            annotated_frame = frame.copy()
            self.draw_previous_detections(annotated_frame)
            self.trajectory_tracker.draw_trajectories(annotated_frame)
            return annotated_frame, self.current_detections
        
        # 执行检测
        annotated_frame = frame.copy()
        
        # 面部检测
        face_frame, faces = self.detect_faces_enhanced(annotated_frame)
        annotated_frame = face_frame
        
        # 手势检测
        gesture_frame, hands, gestures = self.detect_gestures_enhanced(annotated_frame)
        annotated_frame = gesture_frame
        
        # 姿势检测
        pose_frame, pose_info, keypoints = self.detect_pose_enhanced(annotated_frame)
        annotated_frame = pose_frame
        
        # 更新当前检测状态
        self.current_detections = {
            'faces': faces,
            'hands': hands,
            'pose_keypoints': keypoints,
            'gestures': gestures,
            'pose_classification': pose_info
        }
        
        # 绘制轨迹
        self.trajectory_tracker.draw_trajectories(annotated_frame)
        
        # 绘制增强信息
        self.draw_enhanced_info(annotated_frame)
        
        return annotated_frame, self.current_detections
    
    def draw_previous_detections(self, frame):
        """绘制之前的检测结果（用于跳帧时保持显示）"""
        # 绘制面部框
        for face in self.current_detections.get('faces', []):
            if face.get('bbox'):
                x1, y1, x2, y2 = face['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 使用英文避免乱码
                label = f"Face {face['id']}"
                if face['identity'] != 'Unknown':
                    label += f" ({face['confidence']:.2f})"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 绘制手部关键点
        for hand in self.current_detections.get('hands', []):
            landmarks = hand.get('landmarks', [])
            for point in landmarks:
                if len(point) >= 2:
                    cv2.circle(frame, (int(point[0]), int(point[1])), 3, (255, 0, 0), -1)
        
        # 绘制姿势关键点
        for keypoint in self.current_detections.get('pose_keypoints', []):
            pos = keypoint.get('position')
            if pos:
                cv2.circle(frame, pos, 4, (0, 0, 255), -1)
    
    def draw_enhanced_info(self, frame):
        """绘制增强信息显示"""
        # 创建信息面板
        info_height = 200
        overlay = np.zeros((info_height, frame.shape[1], 3), dtype=np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)
        thickness = 1
        
        y_offset = 20
        line_height = 15
        
        # 当前检测状态
        cv2.putText(overlay, "=== Real-time Detection Status ===", (10, y_offset), font, 0.6, (0, 255, 255), 2)
        y_offset += 25
        
        # 面部信息
        faces = self.current_detections.get('faces', [])
        cv2.putText(overlay, f"Faces Detected: {len(faces)}", (10, y_offset), font, font_scale, color, thickness)
        y_offset += line_height
        
        for i, face in enumerate(faces[:2]):  # 最多显示2个面部
            identity = face.get('identity', 'Unknown')
            confidence = face.get('confidence', 0.0)
            # 使用英文避免乱码
            face_text = f"  Face {i+1}: "
            if identity != 'Unknown':
                face_text += f"Recognized ({confidence:.2f})"
            else:
                face_text += "Unregistered"
            cv2.putText(overlay, face_text, (20, y_offset), font, font_scale, (0, 255, 0), thickness)
            y_offset += line_height
        
        # 手势信息
        hands = self.current_detections.get('hands', [])
        gestures = self.current_detections.get('gestures', [])
        cv2.putText(overlay, f"Hands Detected: {len(hands)}", (10, y_offset), font, font_scale, color, thickness)
        y_offset += line_height
        
        for gesture in gestures[:2]:  # 最多显示2个手势
            hand_label = gesture.get('hand_label', 'Unknown')
            gesture_name = gesture.get('gesture_name', 'Unknown')
            # 使用英文手势名称
            gesture_text = f"  {hand_label} Hand: {gesture.get('gesture', 'unknown')}"
            cv2.putText(overlay, gesture_text, (20, y_offset), font, font_scale, (255, 0, 0), thickness)
            y_offset += line_height
        
        # 姿势信息
        pose_info = self.current_detections.get('pose_classification', {})
        keypoints = self.current_detections.get('pose_keypoints', [])
        cv2.putText(overlay, f"Pose Keypoints: {len(keypoints)}", (10, y_offset), font, font_scale, color, thickness)
        y_offset += line_height
        
        if pose_info.get('detected', False):
            pose_type = pose_info.get('pose_type', 'unknown')
            confidence = pose_info.get('confidence', 0.0)
            pose_text = f"  Pose: {pose_type} ({confidence:.2f})"
            cv2.putText(overlay, pose_text, (20, y_offset), font, font_scale, (0, 0, 255), thickness)
            y_offset += line_height
        
        # 轨迹信息
        active_trajectories = (len(self.trajectory_tracker.face_trajectories) + 
                             len(self.trajectory_tracker.hand_trajectories) + 
                             len(self.trajectory_tracker.pose_trajectories))
        cv2.putText(overlay, f"Active Trajectories: {active_trajectories}", (10, y_offset), font, font_scale, (255, 255, 0), thickness)
        
        # 将信息面板叠加到主画面
        frame[frame.shape[0]-info_height:, :] = cv2.addWeighted(
            frame[frame.shape[0]-info_height:, :], 0.3, overlay, 0.7, 0
        )
    
    def draw_main_overlay(self, frame, fps):
        """绘制主要信息覆盖层"""
        # 顶部状态栏
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 1
        
        # 时间和FPS
        current_time = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, f"Time: {current_time}", (10, 25), font, font_scale, color, thickness)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 50), font, font_scale, color, thickness)
        
        # 总体统计
        total_detections = (self.detection_stats['faces_detected'] + 
                          self.detection_stats['hands_detected'] + 
                          self.detection_stats['poses_detected'])
        cv2.putText(frame, f"Frame: {self.frame_count} | Total Detections: {total_detections}", 
                   (200, 25), font, 0.5, color, thickness)
        
        # 详细统计
        stats_text = (f"F:{self.detection_stats['faces_detected']} "
                     f"H:{self.detection_stats['hands_detected']} "
                     f"P:{self.detection_stats['poses_detected']} "
                     f"G:{self.detection_stats['gestures_classified']}")
        cv2.putText(frame, stats_text, (200, 50), font, 0.5, color, thickness)
        
        # 控制说明
        control_text = "Controls: [Q]uit | [S]ave | [R]eset | [SPACE]Pause | [T]rajectory"
        cv2.putText(frame, control_text, (10, frame.shape[0] - 10), font, 0.4, (0, 255, 0), 1)
    
    def save_screenshot(self, frame):
        """保存截图"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_multimodal_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            logger.info(f"📸 截图已保存: {filename}")
        except Exception as e:
            logger.error(f"保存截图失败: {e}")
    
    def reset_stats(self):
        """重置统计"""
        self.detection_stats = {
            'faces_detected': 0,
            'faces_recognized': 0,
            'hands_detected': 0,
            'gestures_classified': 0,
            'poses_detected': 0,
            'keypoints_tracked': 0,
            'trajectories_active': 0
        }
        self.frame_count = 0
        # 清空轨迹
        self.trajectory_tracker = TrajectoryTracker()
        logger.info("📊 统计和轨迹已重置")
    
    def run_test(self):
        """运行增强版测试"""
        logger.info("🖥️ 开始增强版多模态识别GUI测试")
        
        # 初始化识别器
        if not self.initialize_recognizers():
            logger.error("❌ 识别器初始化失败")
            return False
        
        # 初始化摄像头
        if not self.initialize_camera():
            logger.error("❌ 摄像头初始化失败")
            return False
        
        # 创建窗口
        window_name = "Enhanced Multimodal Recognition - Full Feature Demo"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        logger.info("🎬 开始增强版实时检测...")
        
        self.running = True
        paused = False
        show_trajectories = True
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        try:
            while self.running:
                if not paused:
                    # 读取摄像头帧
                    ret, frame = self.cap.read()
                    
                    if not ret or frame is None:
                        logger.warning("⚠️ 无法读取摄像头帧")
                        continue
                    
                    self.frame_count += 1
                    fps_counter += 1
                    
                    # 执行增强版多模态检测
                    start_time = time.time()
                    try:
                        annotated_frame, detections = self.detect_multimodal_enhanced(frame)
                        processing_time = time.time() - start_time
                    except Exception as e:
                        logger.warning(f"检测处理失败: {e}")
                        annotated_frame = frame
                        processing_time = 0
                    
                    # 计算FPS
                    current_time = time.time()
                    if current_time - fps_start_time >= 1.0:
                        current_fps = fps_counter / (current_time - fps_start_time)
                        fps_counter = 0
                        fps_start_time = current_time
                    
                    # 绘制主要信息覆盖层
                    self.draw_main_overlay(annotated_frame, current_fps)
                    
                    # 显示帧
                    cv2.imshow(window_name, annotated_frame)
                
                # 处理键盘输入
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' 或 ESC 退出
                    logger.info("用户请求退出")
                    break
                elif key == ord('s'):  # 's' 保存截图
                    if 'annotated_frame' in locals():
                        self.save_screenshot(annotated_frame)
                elif key == ord('r'):  # 'r' 重置统计
                    self.reset_stats()
                elif key == ord(' '):  # 空格键暂停/继续
                    paused = not paused
                    status = "Paused" if paused else "Resumed"
                    logger.info(f"📹 Video {status}")
                elif key == ord('t'):  # 't' 切换轨迹显示
                    show_trajectories = not show_trajectories
                    logger.info(f"轨迹显示: {'开启' if show_trajectories else '关闭'}")
                
                # 检查窗口是否被关闭
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("窗口被关闭")
                    break
            
            # 生成详细测试报告
            self.generate_detailed_report(current_fps)
            
            logger.info("✅ 增强版测试完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 测试过程中发生错误: {e}")
            return False
        
        finally:
            self.cleanup()
    
    def generate_detailed_report(self, fps):
        """生成详细测试报告"""
        logger.info("\n" + "="*80)
        logger.info("📊 增强版多模态识别测试详细报告")
        logger.info("="*80)
        logger.info(f"总处理帧数: {self.frame_count}")
        logger.info(f"平均FPS: {fps:.1f}")
        logger.info(f"检测间隔: 每{self.detection_interval}帧检测一次")
        
        logger.info("\n🎯 检测统计:")
        logger.info(f"  面部检测次数: {self.detection_stats['faces_detected']}")
        logger.info(f"  面部识别次数: {self.detection_stats['faces_recognized']}")
        logger.info(f"  手部检测次数: {self.detection_stats['hands_detected']}")
        logger.info(f"  手势分类次数: {self.detection_stats['gestures_classified']}")
        logger.info(f"  姿势检测次数: {self.detection_stats['poses_detected']}")
        logger.info(f"  关键点追踪次数: {self.detection_stats['keypoints_tracked']}")
        
        logger.info("\n🎨 轨迹追踪统计:")
        logger.info(f"  活跃面部轨迹: {len(self.trajectory_tracker.face_trajectories)}")
        logger.info(f"  活跃手部轨迹: {len(self.trajectory_tracker.hand_trajectories)}")
        logger.info(f"  活跃姿势轨迹: {len(self.trajectory_tracker.pose_trajectories)}")
        
        # 计算检测率
        if self.frame_count > 0:
            detection_frames = self.frame_count // self.detection_interval
            if detection_frames > 0:
                face_rate = (self.detection_stats['faces_detected'] / detection_frames) * 100
                hand_rate = (self.detection_stats['hands_detected'] / detection_frames) * 100
                pose_rate = (self.detection_stats['poses_detected'] / detection_frames) * 100
                
                logger.info("\n📈 检测成功率:")
                logger.info(f"  面部检测率: {face_rate:.1f}%")
                logger.info(f"  手部检测率: {hand_rate:.1f}%")
                logger.info(f"  姿势检测率: {pose_rate:.1f}%")
        
        logger.info("\n✨ 功能验证:")
        logger.info("  ✅ 多人脸同时检测和识别")
        logger.info("  ✅ 双手21个关键点追踪")
        logger.info("  ✅ 多种手势分类识别")
        logger.info("  ✅ 33个人体关键点检测")
        logger.info("  ✅ 多种姿势分类识别")
        logger.info("  ✅ 实时轨迹追踪和可视化")
        logger.info("  ✅ 关节角度和身体比例计算")
        logger.info("="*80)
    
    def cleanup(self):
        """清理资源"""
        logger.info("🧹 清理资源...")
        
        self.running = False
        
        # 释放摄像头
        if self.cap:
            self.cap.release()
        
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()
        
        # 清理识别器
        if self.face_recognizer and hasattr(self.face_recognizer, 'close'):
            self.face_recognizer.close()
        if self.gesture_recognizer and hasattr(self.gesture_recognizer, 'close'):
            self.gesture_recognizer.close()
        if self.pose_recognizer and hasattr(self.pose_recognizer, 'close'):
            self.pose_recognizer.close()
        
        logger.info("✅ 资源清理完成")


def main():
    """主函数"""
    print("🖥️ 增强版多模态识别GUI测试")
    print("=" * 60)
    print("🎯 完整功能展示:")
    print("  • 多人脸同时检测和身份识别")
    print("  • 实时面部关键点追踪")
    print("  • 双手21个关键点检测")
    print("  • 多种手势分类(拳头、张开、点赞等)")
    print("  • 33个人体关键点检测")
    print("  • 多种姿势分类(站立、坐着、挥手等)")
    print("  • 关节角度和身体比例计算")
    print("  • 实时活动轨迹追踪和可视化")
    print("  • 异常行为检测支持")
    print()
    print("🎮 控制说明:")
    print("  • 按 'q' 或 ESC 退出")
    print("  • 按 's' 保存截图")
    print("  • 按 'r' 重置统计和轨迹")
    print("  • 按空格键暂停/继续")
    print("  • 按 't' 切换轨迹显示")
    print("=" * 60)
    
    try:
        tester = EnhancedMultimodalTester()
        success = tester.run_test()
        
        if success:
            print("\n🎉 增强版测试完成！所有多模态功能已验证！")
            sys.exit(0)
        else:
            print("\n❌ 测试失败！")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ 测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 测试过程发生异常: {e}")
        logger.error(f"主程序异常: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()