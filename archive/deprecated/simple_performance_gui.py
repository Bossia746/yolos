#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化高性能多模态识别GUI系统
基于最佳实践，解决卡顿和多手检测问题
避免复杂依赖导入问题

参考最佳实践:
- OpenCV多线程优化
- MediaPipe性能优化
- GUI性能优化
"""

import cv2
import numpy as np
import threading
import queue
import time
import logging
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from collections import deque
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simple_performance_gui.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class FrameData:
    """帧数据结构"""
    frame: np.ndarray
    timestamp: float
    frame_id: int

@dataclass
class DetectionResult:
    """检测结果结构"""
    frame: np.ndarray
    hands: List[Dict[str, Any]]
    faces: List[Dict[str, Any]]
    poses: List[Dict[str, Any]]
    fps: float
    processing_time: float
    frame_id: int

class ThreadSafeCamera:
    """线程安全的摄像头类"""
    
    def __init__(self, camera_id: int = 0, buffer_size: int = 2):
        self.camera_id = camera_id
        self.buffer_size = buffer_size
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.running = False
        self.capture_thread = None
        self.frame_count = 0
        self.fps_counter = deque(maxlen=30)
        
    def start(self) -> bool:
        """启动摄像头捕获"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                logger.error(f"无法打开摄像头 {self.camera_id}")
                return False
                
            # 优化摄像头设置
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲区延迟
            
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.capture_thread.start()
            
            logger.info(f"📹 摄像头 {self.camera_id} 启动成功")
            return True
            
        except Exception as e:
            logger.error(f"摄像头启动失败: {e}")
            return False
    
    def _capture_frames(self):
        """帧捕获线程"""
        last_time = time.time()
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("无法读取摄像头帧")
                    continue
                
                current_time = time.time()
                self.fps_counter.append(current_time - last_time)
                last_time = current_time
                
                frame_data = FrameData(
                    frame=frame,
                    timestamp=current_time,
                    frame_id=self.frame_count
                )
                
                # 非阻塞放入队列
                try:
                    self.frame_queue.put_nowait(frame_data)
                    self.frame_count += 1
                except queue.Full:
                    # 队列满时丢弃最旧的帧
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame_data)
                    except queue.Empty:
                        pass
                        
            except Exception as e:
                logger.error(f"帧捕获错误: {e}")
                time.sleep(0.01)
    
    def get_frame(self) -> Optional[FrameData]:
        """获取最新帧"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_fps(self) -> float:
        """获取实际FPS"""
        if len(self.fps_counter) < 2:
            return 0.0
        avg_interval = sum(self.fps_counter) / len(self.fps_counter)
        return 1.0 / avg_interval if avg_interval > 0 else 0.0
    
    def stop(self):
        """停止摄像头"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        logger.info("摄像头已停止")

class SimpleMultiModalDetector:
    """简化多模态检测器 - 直接使用MediaPipe"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_detection
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = None
        self.face_detection = None
        self.pose = None
        
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.detection_stats = {
            'total_detections': 0,
            'avg_processing_time': 0.0,
            'last_fps': 0.0
        }
        
    def initialize(self) -> bool:
        """初始化所有检测器"""
        try:
            logger.info("🔧 初始化简化多模态检测器...")
            
            # 初始化手势检测器 - 优化配置
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=4,  # 支持多手检测
                min_detection_confidence=0.6,  # 降低阈值提高检测率
                min_tracking_confidence=0.4,   # 降低跟踪阈值
                model_complexity=0  # 使用轻量级模型
            )
            
            # 初始化人脸检测器
            self.face_detection = self.mp_face.FaceDetection(
                model_selection=0,  # 短距离模型
                min_detection_confidence=0.6
            )
            
            # 初始化姿势检测器
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,  # 轻量级模型
                enable_segmentation=False,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.4
            )
            
            logger.info("✅ 简化多模态检测器初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"检测器初始化失败: {e}")
            return False
    
    def detect_parallel(self, frame: np.ndarray) -> DetectionResult:
        """并行检测"""
        start_time = time.time()
        
        # 转换颜色空间
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 提交并行任务
        hand_future = self.executor.submit(self._detect_hands, rgb_frame)
        face_future = self.executor.submit(self._detect_faces, rgb_frame)
        pose_future = self.executor.submit(self._detect_poses, rgb_frame)
        
        # 收集结果
        try:
            hands = hand_future.result(timeout=0.1)  # 100ms超时
        except:
            hands = []
            
        try:
            faces = face_future.result(timeout=0.1)
        except:
            faces = []
            
        try:
            poses = pose_future.result(timeout=0.1)
        except:
            poses = []
        
        processing_time = time.time() - start_time
        fps = 1.0 / processing_time if processing_time > 0 else 0.0
        
        # 更新统计
        self._update_stats(processing_time, fps)
        
        return DetectionResult(
            frame=frame,
            hands=hands,
            faces=faces,
            poses=poses,
            fps=fps,
            processing_time=processing_time,
            frame_id=0
        )
    
    def _detect_hands(self, rgb_frame: np.ndarray) -> List[Dict[str, Any]]:
        """手势检测"""
        try:
            if self.hands:
                results = self.hands.process(rgb_frame)
                hands_data = []
                
                if results.multi_hand_landmarks and results.multi_handedness:
                    for i, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                        # 提取关键点
                        landmarks = []
                        h, w = rgb_frame.shape[:2]
                        for lm in hand_landmarks.landmark:
                            landmarks.append((int(lm.x * w), int(lm.y * h)))
                        
                        # 计算边界框
                        x_coords = [lm[0] for lm in landmarks]
                        y_coords = [lm[1] for lm in landmarks]
                        bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                        
                        # 简单手势识别
                        gesture = self._simple_gesture_recognition(landmarks)
                        
                        hands_data.append({
                            'type': 'hand',
                            'landmarks': landmarks,
                            'bbox': bbox,
                            'handedness': handedness.classification[0].label,
                            'handedness_confidence': handedness.classification[0].score,
                            'gesture': gesture,
                            'confidence': handedness.classification[0].score
                        })
                
                return hands_data
        except Exception as e:
            logger.debug(f"手势检测错误: {e}")
        return []
    
    def _simple_gesture_recognition(self, landmarks: List[Tuple[int, int]]) -> Dict[str, Any]:
        """简单手势识别"""
        try:
            # 基于关键点位置的简单手势识别
            # 拇指尖、食指尖、中指尖、无名指尖、小指尖
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]
            pinky_tip = landmarks[20]
            
            # 手腕
            wrist = landmarks[0]
            
            # 计算手指是否伸展
            fingers_up = []
            
            # 拇指 (比较x坐标)
            if thumb_tip[0] > landmarks[3][0]:  # 拇指尖 > 拇指关节
                fingers_up.append(1)
            else:
                fingers_up.append(0)
            
            # 其他四指 (比较y坐标)
            for tip_id in [8, 12, 16, 20]:
                if landmarks[tip_id][1] < landmarks[tip_id - 2][1]:  # 指尖 < 关节
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)
            
            total_fingers = sum(fingers_up)
            
            # 简单手势分类
            if total_fingers == 0:
                gesture = "Fist"
                confidence = 0.8
            elif total_fingers == 5:
                gesture = "Open Palm"
                confidence = 0.8
            elif total_fingers == 1 and fingers_up[1] == 1:  # 只有食指
                gesture = "Pointing"
                confidence = 0.7
            elif total_fingers == 2 and fingers_up[1] == 1 and fingers_up[2] == 1:  # 食指和中指
                gesture = "Peace Sign"
                confidence = 0.7
            elif total_fingers == 1 and fingers_up[0] == 1:  # 只有拇指
                gesture = "Thumbs Up"
                confidence = 0.7
            else:
                gesture = f"{total_fingers} Fingers"
                confidence = 0.5
            
            return {
                'gesture': gesture,
                'confidence': confidence,
                'fingers_up': fingers_up,
                'total_fingers': total_fingers
            }
            
        except Exception as e:
            logger.debug(f"手势识别错误: {e}")
            return {'gesture': 'Unknown', 'confidence': 0.0}
    
    def _detect_faces(self, rgb_frame: np.ndarray) -> List[Dict[str, Any]]:
        """人脸检测"""
        try:
            if self.face_detection:
                results = self.face_detection.process(rgb_frame)
                faces_data = []
                
                if results.detections:
                    h, w = rgb_frame.shape[:2]
                    for detection in results.detections:
                        bbox_rel = detection.location_data.relative_bounding_box
                        bbox = (
                            int(bbox_rel.xmin * w),
                            int(bbox_rel.ymin * h),
                            int((bbox_rel.xmin + bbox_rel.width) * w),
                            int((bbox_rel.ymin + bbox_rel.height) * h)
                        )
                        
                        faces_data.append({
                            'type': 'face',
                            'bbox': bbox,
                            'confidence': detection.score[0]
                        })
                
                return faces_data
        except Exception as e:
            logger.debug(f"人脸检测错误: {e}")
        return []
    
    def _detect_poses(self, rgb_frame: np.ndarray) -> List[Dict[str, Any]]:
        """姿势检测"""
        try:
            if self.pose:
                results = self.pose.process(rgb_frame)
                poses_data = []
                
                if results.pose_landmarks:
                    h, w = rgb_frame.shape[:2]
                    keypoints = []
                    for lm in results.pose_landmarks.landmark:
                        keypoints.append((int(lm.x * w), int(lm.y * h)))
                    
                    poses_data.append({
                        'type': 'pose',
                        'keypoints': keypoints,
                        'confidence': 0.8  # MediaPipe不直接提供置信度
                    })
                
                return poses_data
        except Exception as e:
            logger.debug(f"姿势检测错误: {e}")
        return []
    
    def _update_stats(self, processing_time: float, fps: float):
        """更新统计信息"""
        self.detection_stats['total_detections'] += 1
        self.detection_stats['avg_processing_time'] = (
            (self.detection_stats['avg_processing_time'] * (self.detection_stats['total_detections'] - 1) + processing_time) /
            self.detection_stats['total_detections']
        )
        self.detection_stats['last_fps'] = fps
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.detection_stats.copy()
    
    def close(self):
        """关闭检测器"""
        self.executor.shutdown(wait=True)
        if self.hands:
            self.hands.close()
        if self.face_detection:
            self.face_detection.close()
        if self.pose:
            self.pose.close()

class SimplePerformanceGUI:
    """简化高性能GUI主类"""
    
    def __init__(self):
        self.camera = None
        self.detector = None
        self.running = False
        self.display_thread = None
        self.result_queue = queue.Queue(maxsize=5)
        
        # 性能统计
        self.frame_times = deque(maxlen=30)
        self.total_frames = 0
        self.start_time = time.time()
        
        # GUI设置
        self.window_name = "简化高性能多模态识别系统"
        self.window_size = (1280, 720)
        
    def initialize(self) -> bool:
        """初始化系统"""
        logger.info("🚀 启动简化高性能多模态识别GUI系统")
        
        # 初始化摄像头
        self.camera = ThreadSafeCamera(camera_id=0, buffer_size=2)
        if not self.camera.start():
            logger.error("摄像头初始化失败")
            return False
        
        # 初始化检测器
        self.detector = SimpleMultiModalDetector()
        if not self.detector.initialize():
            logger.error("检测器初始化失败")
            return False
        
        # 创建GUI窗口
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, *self.window_size)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
        
        logger.info("✅ 系统初始化完成")
        return True
    
    def run(self):
        """运行主循环"""
        if not self.initialize():
            logger.error("系统初始化失败")
            return
        
        self.running = True
        
        # 启动显示线程
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()
        
        logger.info("🎯 开始主处理循环")
        
        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("用户中断程序")
        except Exception as e:
            logger.error(f"主循环错误: {e}")
        finally:
            self._cleanup()
    
    def _main_loop(self):
        """主处理循环"""
        frame_skip = 0  # 帧跳跃计数
        
        while self.running:
            frame_start = time.time()
            
            # 获取最新帧
            frame_data = self.camera.get_frame()
            if frame_data is None:
                time.sleep(0.001)  # 1ms等待
                continue
            
            # 帧跳跃优化 - 每2帧处理一次检测
            if frame_skip % 2 == 0:
                # 执行检测
                detection_result = self.detector.detect_parallel(frame_data.frame)
                detection_result.frame_id = frame_data.frame_id
                
                # 非阻塞放入显示队列
                try:
                    self.result_queue.put_nowait(detection_result)
                except queue.Full:
                    # 队列满时丢弃最旧结果
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait(detection_result)
                    except queue.Empty:
                        pass
            else:
                # 跳过检测，直接显示原始帧
                simple_result = DetectionResult(
                    frame=frame_data.frame,
                    hands=[], faces=[], poses=[],
                    fps=0.0, processing_time=0.0,
                    frame_id=frame_data.frame_id
                )
                try:
                    self.result_queue.put_nowait(simple_result)
                except queue.Full:
                    pass
            
            frame_skip += 1
            self.total_frames += 1
            
            # 记录帧时间
            frame_time = time.time() - frame_start
            self.frame_times.append(frame_time)
            
            # 每50帧输出一次统计
            if self.total_frames % 50 == 0:
                self._log_performance_stats()
            
            # 控制帧率 - 目标30FPS
            target_frame_time = 1.0 / 30.0
            if frame_time < target_frame_time:
                time.sleep(target_frame_time - frame_time)
    
    def _display_loop(self):
        """显示循环线程"""
        logger.info("🖥️ 显示线程启动")
        
        while self.running:
            try:
                # 获取检测结果
                result = self.result_queue.get(timeout=0.1)
                
                # 绘制注释
                annotated_frame = self._draw_annotations(result)
                
                # 显示帧
                cv2.imshow(self.window_name, annotated_frame)
                
                # 检查按键
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC键
                    self.running = False
                    break
                elif key == ord('q'):
                    self.running = False
                    break
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"显示循环错误: {e}")
                time.sleep(0.01)
    
    def _draw_annotations(self, result: DetectionResult) -> np.ndarray:
        """绘制注释信息"""
        frame = result.frame.copy()
        
        # 绘制手势
        for hand in result.hands:
            if hand['type'] == 'hand':
                self._draw_hand_info(frame, hand)
        
        # 绘制人脸
        for face in result.faces:
            if face['type'] == 'face':
                self._draw_face_info(frame, face)
        
        # 绘制姿势
        for pose in result.poses:
            if pose['type'] == 'pose':
                self._draw_pose_info(frame, pose)
        
        # 绘制性能信息
        self._draw_performance_info(frame, result)
        
        return frame
    
    def _draw_hand_info(self, frame: np.ndarray, hand_data: Dict[str, Any]):
        """绘制手部信息"""
        try:
            # 绘制关键点
            if 'landmarks' in hand_data:
                for i, (x, y) in enumerate(hand_data['landmarks']):
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                    if i in [4, 8, 12, 16, 20]:  # 指尖
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), 2)
            
            # 绘制边界框
            if 'bbox' in hand_data:
                x1, y1, x2, y2 = hand_data['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 绘制手势标签
                gesture_info = hand_data.get('gesture', {})
                gesture_name = gesture_info.get('gesture', 'Unknown')
                confidence = gesture_info.get('confidence', 0.0)
                handedness = hand_data.get('handedness', 'Unknown')
                
                label = f"{handedness} {gesture_name} ({confidence:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
        except Exception as e:
            logger.debug(f"绘制手部信息错误: {e}")
    
    def _draw_face_info(self, frame: np.ndarray, face_data: Dict[str, Any]):
        """绘制人脸信息"""
        try:
            if 'bbox' in face_data:
                x1, y1, x2, y2 = face_data['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"Face ({face_data.get('confidence', 0.0):.2f})",
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        except Exception as e:
            logger.debug(f"绘制人脸信息错误: {e}")
    
    def _draw_pose_info(self, frame: np.ndarray, pose_data: Dict[str, Any]):
        """绘制姿势信息"""
        try:
            if 'keypoints' in pose_data:
                # 只绘制主要关键点
                important_points = [0, 11, 12, 13, 14, 15, 16]  # 鼻子、肩膀、手肘、手腕
                for i, (x, y) in enumerate(pose_data['keypoints']):
                    if i in important_points:
                        cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)
        except Exception as e:
            logger.debug(f"绘制姿势信息错误: {e}")
    
    def _draw_performance_info(self, frame: np.ndarray, result: DetectionResult):
        """绘制性能信息"""
        try:
            # 计算FPS
            camera_fps = self.camera.get_fps()
            avg_frame_time = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0
            display_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            
            # 性能文本
            perf_text = [
                f"Camera FPS: {camera_fps:.1f}",
                f"Display FPS: {display_fps:.1f}",
                f"Processing: {result.processing_time*1000:.1f}ms",
                f"Hands: {len(result.hands)}",
                f"Faces: {len(result.faces)}",
                f"Poses: {len(result.poses)}",
                f"Total Frames: {self.total_frames}"
            ]
            
            # 绘制性能信息背景
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, 5), (350, 200), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # 绘制性能信息
            y_offset = 30
            for i, text in enumerate(perf_text):
                cv2.putText(frame, text, (10, y_offset + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
        except Exception as e:
            logger.debug(f"绘制性能信息错误: {e}")
    
    def _log_performance_stats(self):
        """记录性能统计"""
        try:
            runtime = time.time() - self.start_time
            avg_fps = self.total_frames / runtime if runtime > 0 else 0
            camera_fps = self.camera.get_fps()
            detector_stats = self.detector.get_stats()
            
            logger.info(f"📊 性能统计 - 总帧数: {self.total_frames}, "
                       f"平均FPS: {avg_fps:.1f}, 摄像头FPS: {camera_fps:.1f}, "
                       f"检测FPS: {detector_stats['last_fps']:.1f}")
        except Exception as e:
            logger.debug(f"性能统计错误: {e}")
    
    def _cleanup(self):
        """清理资源"""
        logger.info("🧹 清理系统资源...")
        
        self.running = False
        
        if self.display_thread:
            self.display_thread.join(timeout=1.0)
        
        if self.camera:
            self.camera.stop()
        
        if self.detector:
            self.detector.close()
        
        cv2.destroyAllWindows()
        
        # 最终统计
        runtime = time.time() - self.start_time
        avg_fps = self.total_frames / runtime if runtime > 0 else 0
        logger.info(f"🏁 系统运行完成 - 总运行时间: {runtime:.1f}s, "
                   f"总帧数: {self.total_frames}, 平均FPS: {avg_fps:.1f}")

def main():
    """主函数"""
    try:
        gui = SimplePerformanceGUI()
        gui.run()
    except Exception as e:
        logger.error(f"程序运行错误: {e}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())