#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超高性能多模态识别GUI系统
基于最佳实践重构，解决卡顿和多手检测问题

参考最佳实践:
- OpenCV多线程优化: https://nrsyed.com/2018/07/05/multithreading-with-opencv-python-to-improve-video-processing-performance/
- MediaPipe性能优化: https://mediapipe.readthedocs.io/en/latest/solutions/hands.html
- GUI性能优化: https://pysource.com/2024/10/15/increase-opencv-speed-by-2x-with-python-and-multithreading-tutorial/
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

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.recognition.optimized_gesture_recognizer import OptimizedGestureRecognizer
from src.recognition.optimized_face_recognizer import OptimizedFaceRecognizer
from src.recognition.optimized_pose_recognizer import OptimizedPoseRecognizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultra_performance_gui.log'),
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
            
            logger.info(f"摄像头 {self.camera_id} 启动成功")
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

class MultiModalDetector:
    """多模态检测器 - 并行处理"""
    
    def __init__(self):
        self.gesture_recognizer = None
        self.face_recognizer = None
        self.pose_recognizer = None
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.detection_stats = {
            'total_detections': 0,
            'avg_processing_time': 0.0,
            'last_fps': 0.0
        }
        
    def initialize(self) -> bool:
        """初始化所有检测器"""
        try:
            logger.info("初始化多模态检测器...")
            
            # 初始化手势识别器 - 优化配置
            self.gesture_recognizer = OptimizedGestureRecognizer(
                max_num_hands=4,  # 支持多手检测
                min_detection_confidence=0.6,  # 降低阈值提高检测率
                min_tracking_confidence=0.4,   # 降低跟踪阈值
                enable_threading=True,
                enable_dynamic_gestures=True,
                enable_interaction_detection=True
            )
            
            # 初始化人脸识别器
            self.face_recognizer = OptimizedFaceRecognizer(
                min_detection_confidence=0.6,
                enable_threading=True
            )
            
            # 初始化姿势识别器
            self.pose_recognizer = OptimizedPoseRecognizer(
                min_detection_confidence=0.6,
                enable_threading=True
            )
            
            logger.info("多模态检测器初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"检测器初始化失败: {e}")
            return False
    
    def detect_parallel(self, frame: np.ndarray) -> DetectionResult:
        """并行检测"""
        start_time = time.time()
        
        # 提交并行任务
        hand_future = self.executor.submit(self._detect_hands, frame)
        face_future = self.executor.submit(self._detect_faces, frame)
        pose_future = self.executor.submit(self._detect_poses, frame)
        
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
    
    def _detect_hands(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """手势检测"""
        try:
            if self.gesture_recognizer:
                result = self.gesture_recognizer.detect_gestures(frame)
                return [{
                    'type': 'hand',
                    'data': hand,
                    'confidence': hand.quality_score
                } for hand in result.hands_data]
        except Exception as e:
            logger.debug(f"手势检测错误: {e}")
        return []
    
    def _detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """人脸检测"""
        try:
            if self.face_recognizer:
                result = self.face_recognizer.detect_faces(frame)
                return [{
                    'type': 'face',
                    'data': face,
                    'confidence': face.get('confidence', 0.0)
                } for face in result.get('faces', [])]
        except Exception as e:
            logger.debug(f"人脸检测错误: {e}")
        return []
    
    def _detect_poses(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """姿势检测"""
        try:
            if self.pose_recognizer:
                result = self.pose_recognizer.detect_poses(frame)
                return [{
                    'type': 'pose',
                    'data': pose,
                    'confidence': pose.get('confidence', 0.0)
                } for pose in result.get('poses', [])]
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
        if self.gesture_recognizer:
            self.gesture_recognizer.close()
        if self.face_recognizer:
            self.face_recognizer.close()
        if self.pose_recognizer:
            self.pose_recognizer.close()

class UltraPerformanceGUI:
    """超高性能GUI主类"""
    
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
        self.window_name = "超高性能多模态识别系统"
        self.window_size = (1280, 720)
        
    def initialize(self) -> bool:
        """初始化系统"""
        logger.info("🚀 启动超高性能多模态识别GUI系统")
        
        # 初始化摄像头
        self.camera = ThreadSafeCamera(camera_id=0, buffer_size=2)
        if not self.camera.start():
            logger.error("摄像头初始化失败")
            return False
        
        # 初始化检测器
        self.detector = MultiModalDetector()
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
            
            # 帧跳跃优化 - 每3帧处理一次检测
            if frame_skip % 3 == 0:
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
            
            # 每100帧输出一次统计
            if self.total_frames % 100 == 0:
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
                self._draw_hand_info(frame, hand['data'])
        
        # 绘制人脸
        for face in result.faces:
            if face['type'] == 'face':
                self._draw_face_info(frame, face['data'])
        
        # 绘制姿势
        for pose in result.poses:
            if pose['type'] == 'pose':
                self._draw_pose_info(frame, pose['data'])
        
        # 绘制性能信息
        self._draw_performance_info(frame, result)
        
        return frame
    
    def _draw_hand_info(self, frame: np.ndarray, hand_data: Any):
        """绘制手部信息"""
        try:
            if hasattr(hand_data, 'landmarks') and hand_data.landmarks:
                # 绘制关键点
                for i, (x, y) in enumerate(hand_data.landmarks):
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                
                # 绘制边界框
                if hasattr(hand_data, 'bbox'):
                    x1, y1, x2, y2 = hand_data.bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 绘制手势标签
                if hasattr(hand_data, 'static_gesture'):
                    gesture = hand_data.static_gesture.get('gesture', 'Unknown')
                    confidence = hand_data.static_gesture.get('confidence', 0.0)
                    label = f"{gesture} ({confidence:.2f})"
                    cv2.putText(frame, label, (int(hand_data.landmarks[0][0]), int(hand_data.landmarks[0][1]) - 10),
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
                for x, y in pose_data['keypoints']:
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)
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
            
            # 绘制性能信息
            y_offset = 30
            for i, text in enumerate(perf_text):
                cv2.putText(frame, text, (10, y_offset + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, text, (10, y_offset + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
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
        gui = UltraPerformanceGUI()
        gui.run()
    except Exception as e:
        logger.error(f"程序运行错误: {e}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())