#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线高性能多模态识别GUI系统
基于最佳实践，解决网络依赖和卡顿问题
使用OpenCV内置功能，避免MediaPipe网络下载

参考最佳实践:
- OpenCV多线程优化
- 离线模型使用
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
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('offline_performance_gui.log'),
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

class OfflineMultiModalDetector:
    """离线多模态检测器 - 使用OpenCV内置功能"""
    
    def __init__(self):
        self.face_cascade = None
        self.hand_cascade = None
        self.body_cascade = None
        
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.detection_stats = {
            'total_detections': 0,
            'avg_processing_time': 0.0,
            'last_fps': 0.0
        }
        
        # 手势跟踪状态
        self.hand_tracker = HandTracker()
        
    def initialize(self) -> bool:
        """初始化所有检测器"""
        try:
            logger.info("🔧 初始化离线多模态检测器...")
            
            # 加载OpenCV内置的Haar级联分类器
            cascade_path = cv2.data.haarcascades
            
            # 人脸检测器
            face_cascade_file = os.path.join(cascade_path, 'haarcascade_frontalface_default.xml')
            if os.path.exists(face_cascade_file):
                self.face_cascade = cv2.CascadeClassifier(face_cascade_file)
                logger.info("✅ 人脸检测器加载成功")
            else:
                logger.warning("⚠️ 人脸检测器文件未找到")
            
            # 尝试加载其他级联分类器
            try:
                # 全身检测器
                body_cascade_file = os.path.join(cascade_path, 'haarcascade_fullbody.xml')
                if os.path.exists(body_cascade_file):
                    self.body_cascade = cv2.CascadeClassifier(body_cascade_file)
                    logger.info("✅ 全身检测器加载成功")
            except:
                logger.info("全身检测器不可用，将使用替代方法")
            
            logger.info("✅ 离线多模态检测器初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"检测器初始化失败: {e}")
            return False
    
    def detect_parallel(self, frame: np.ndarray) -> DetectionResult:
        """并行检测"""
        start_time = time.time()
        
        # 转换为灰度图像以提高性能
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 提交并行任务
        hand_future = self.executor.submit(self._detect_hands, frame, gray_frame)
        face_future = self.executor.submit(self._detect_faces, frame, gray_frame)
        pose_future = self.executor.submit(self._detect_poses, frame, gray_frame)
        
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
    
    def _detect_hands(self, frame: np.ndarray, gray_frame: np.ndarray) -> List[Dict[str, Any]]:
        """手势检测 - 使用轮廓和颜色检测"""
        try:
            hands_data = []
            
            # 使用肤色检测
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 定义肤色范围 (HSV)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            # 创建肤色掩码
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # 形态学操作去噪
            kernel = np.ones((3, 3), np.uint8)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 筛选手部轮廓
            for contour in contours:
                area = cv2.contourArea(contour)
                if 1000 < area < 50000:  # 手部面积范围
                    # 计算边界框
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # 计算轮廓中心
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = x + w//2, y + h//2
                    
                    # 计算凸包
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    
                    # 计算凸性缺陷
                    defects = []
                    if len(contour) > 10:
                        hull_indices = cv2.convexHull(contour, returnPoints=False)
                        if len(hull_indices) > 3:
                            defects_raw = cv2.convexityDefects(contour, hull_indices)
                            if defects_raw is not None:
                                for i in range(defects_raw.shape[0]):
                                    s, e, f, d = defects_raw[i, 0]
                                    if d > 8000:  # 深度阈值
                                        defects.append((s, e, f, d))
                    
                    # 简单手势识别
                    gesture = self._analyze_hand_gesture(contour, hull, defects, area, hull_area)
                    
                    # 更新手部跟踪
                    hand_id = self.hand_tracker.update_hand(cx, cy, area)
                    
                    hands_data.append({
                        'type': 'hand',
                        'bbox': (x, y, x + w, y + h),
                        'center': (cx, cy),
                        'area': area,
                        'contour': contour,
                        'hull': hull,
                        'defects': defects,
                        'gesture': gesture,
                        'hand_id': hand_id,
                        'confidence': min(0.9, area / 10000)  # 基于面积的置信度
                    })
            
            return hands_data[:4]  # 最多返回4只手
            
        except Exception as e:
            logger.debug(f"手势检测错误: {e}")
            return []
    
    def _analyze_hand_gesture(self, contour, hull, defects, area, hull_area) -> Dict[str, Any]:
        """分析手势"""
        try:
            # 计算凸性比率
            solidity = area / hull_area if hull_area > 0 else 0
            
            # 基于凸性缺陷数量判断手势
            defect_count = len(defects)
            
            if solidity > 0.95:  # 很凸，可能是拳头
                gesture = "Fist"
                confidence = 0.8
            elif defect_count == 0:
                gesture = "Closed Hand"
                confidence = 0.7
            elif defect_count == 1:
                gesture = "Pointing"
                confidence = 0.6
            elif defect_count == 2:
                gesture = "Peace Sign"
                confidence = 0.6
            elif defect_count >= 3:
                gesture = "Open Hand"
                confidence = 0.7
            else:
                gesture = f"{defect_count} Fingers"
                confidence = 0.5
            
            return {
                'gesture': gesture,
                'confidence': confidence,
                'defect_count': defect_count,
                'solidity': solidity
            }
            
        except Exception as e:
            logger.debug(f"手势分析错误: {e}")
            return {'gesture': 'Unknown', 'confidence': 0.0}
    
    def _detect_faces(self, frame: np.ndarray, gray_frame: np.ndarray) -> List[Dict[str, Any]]:
        """人脸检测"""
        try:
            faces_data = []
            
            if self.face_cascade is not None:
                # 使用Haar级联检测人脸
                faces = self.face_cascade.detectMultiScale(
                    gray_frame,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                for (x, y, w, h) in faces:
                    faces_data.append({
                        'type': 'face',
                        'bbox': (x, y, x + w, y + h),
                        'confidence': 0.8  # Haar级联不提供置信度
                    })
            
            return faces_data
            
        except Exception as e:
            logger.debug(f"人脸检测错误: {e}")
            return []
    
    def _detect_poses(self, frame: np.ndarray, gray_frame: np.ndarray) -> List[Dict[str, Any]]:
        """姿势检测 - 简化版本"""
        try:
            poses_data = []
            
            if self.body_cascade is not None:
                # 使用全身检测器
                bodies = self.body_cascade.detectMultiScale(
                    gray_frame,
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(50, 100)
                )
                
                for (x, y, w, h) in bodies:
                    # 估算关键点位置
                    keypoints = self._estimate_keypoints(x, y, w, h)
                    
                    poses_data.append({
                        'type': 'pose',
                        'bbox': (x, y, x + w, y + h),
                        'keypoints': keypoints,
                        'confidence': 0.6
                    })
            
            return poses_data
            
        except Exception as e:
            logger.debug(f"姿势检测错误: {e}")
            return []
    
    def _estimate_keypoints(self, x: int, y: int, w: int, h: int) -> List[Tuple[int, int]]:
        """估算关键点位置"""
        # 基于边界框估算主要关键点
        keypoints = [
            (x + w//2, y + h//8),      # 头部
            (x + w//4, y + h//3),      # 左肩
            (x + 3*w//4, y + h//3),    # 右肩
            (x + w//4, y + 2*h//3),    # 左肘
            (x + 3*w//4, y + 2*h//3),  # 右肘
            (x + w//4, y + h),         # 左手
            (x + 3*w//4, y + h),       # 右手
        ]
        return keypoints
    
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

class HandTracker:
    """手部跟踪器"""
    
    def __init__(self, max_distance: float = 100.0):
        self.hands = {}  # hand_id -> (x, y, last_seen)
        self.next_id = 0
        self.max_distance = max_distance
        self.max_age = 30  # 最大帧数
    
    def update_hand(self, x: int, y: int, area: float) -> int:
        """更新手部位置，返回手部ID"""
        current_time = time.time()
        
        # 查找最近的已知手部
        best_id = None
        best_distance = float('inf')
        
        for hand_id, (hx, hy, last_seen, _) in self.hands.items():
            distance = np.sqrt((x - hx)**2 + (y - hy)**2)
            if distance < self.max_distance and distance < best_distance:
                best_distance = distance
                best_id = hand_id
        
        if best_id is not None:
            # 更新现有手部
            self.hands[best_id] = (x, y, current_time, area)
            return best_id
        else:
            # 创建新手部
            new_id = self.next_id
            self.next_id += 1
            self.hands[new_id] = (x, y, current_time, area)
            return new_id
    
    def cleanup_old_hands(self):
        """清理旧的手部跟踪"""
        current_time = time.time()
        to_remove = []
        
        for hand_id, (_, _, last_seen, _) in self.hands.items():
            if current_time - last_seen > 1.0:  # 1秒未见
                to_remove.append(hand_id)
        
        for hand_id in to_remove:
            del self.hands[hand_id]

class OfflinePerformanceGUI:
    """离线高性能GUI主类"""
    
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
        self.window_name = "离线高性能多模态识别系统"
        self.window_size = (1280, 720)
        
    def initialize(self) -> bool:
        """初始化系统"""
        logger.info("🚀 启动离线高性能多模态识别GUI系统")
        
        # 初始化摄像头
        self.camera = ThreadSafeCamera(camera_id=0, buffer_size=2)
        if not self.camera.start():
            logger.error("摄像头初始化失败")
            return False
        
        # 初始化检测器
        self.detector = OfflineMultiModalDetector()
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
                # 清理旧的手部跟踪
                self.detector.hand_tracker.cleanup_old_hands()
            
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
            # 绘制轮廓
            if 'contour' in hand_data:
                cv2.drawContours(frame, [hand_data['contour']], -1, (0, 255, 0), 2)
            
            # 绘制凸包
            if 'hull' in hand_data:
                cv2.drawContours(frame, [hand_data['hull']], -1, (0, 255, 255), 2)
            
            # 绘制边界框
            if 'bbox' in hand_data:
                x1, y1, x2, y2 = hand_data['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 绘制手势标签
                gesture_info = hand_data.get('gesture', {})
                gesture_name = gesture_info.get('gesture', 'Unknown')
                confidence = gesture_info.get('confidence', 0.0)
                hand_id = hand_data.get('hand_id', 0)
                
                label = f"Hand{hand_id}: {gesture_name} ({confidence:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 绘制中心点
            if 'center' in hand_data:
                cx, cy = hand_data['center']
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                
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
                # 绘制关键点
                for i, (x, y) in enumerate(pose_data['keypoints']):
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(i), (int(x)+5, int(y)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
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
                f"Total Frames: {self.total_frames}",
                f"Mode: Offline OpenCV"
            ]
            
            # 绘制性能信息背景
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, 5), (350, 220), (0, 0, 0), -1)
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
        gui = OfflinePerformanceGUI()
        gui.run()
    except Exception as e:
        logger.error(f"程序运行错误: {e}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())