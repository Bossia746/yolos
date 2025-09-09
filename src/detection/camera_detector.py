"""
摄像头检测器 - 专门用于树莓派等嵌入式设备
"""

import cv2
import numpy as np
import time
import threading
from queue import Queue
from typing import Optional, Callable, Dict, Any
import json
from pathlib import Path

from ..models.yolo_factory import YOLOFactory

# 树莓派摄像头支持
try:
    from picamera import PiCamera
    from picamera.array import PiRGBArray
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False


class CameraDetector:
    """摄像头检测器"""
    
    def __init__(self, 
                 model_type: str = 'yolov8',
                 model_path: Optional[str] = None,
                 device: str = 'auto',
                 camera_type: str = 'usb'):
        """
        初始化摄像头检测器
        
        Args:
            model_type: 模型类型
            model_path: 模型路径
            device: 设备类型
            camera_type: 摄像头类型 ('usb', 'picamera')
        """
        self.model = YOLOFactory.create_model(model_type, model_path, device)
        self.camera_type = camera_type
        self.is_running = False
        
        # 摄像头参数
        self.resolution = (640, 480)
        self.framerate = 30
        self.camera_id = 0
        
        # 检测参数
        self.detection_interval = 1  # 每N帧检测一次
        self.frame_count = 0
        
        # 回调函数
        self.detection_callback: Optional[Callable] = None
        self.frame_callback: Optional[Callable] = None
        
        # 性能统计
        self.fps = 0
        self.detection_count = 0
        self.start_time = time.time()
    
    def set_camera_params(self, resolution: tuple = (640, 480), framerate: int = 30, camera_id: int = 0):
        """设置摄像头参数"""
        self.resolution = resolution
        self.framerate = framerate
        self.camera_id = camera_id
    
    def set_detection_params(self, interval: int = 1):
        """设置检测参数"""
        self.detection_interval = interval
    
    def set_callbacks(self, detection_callback: Optional[Callable] = None, 
                     frame_callback: Optional[Callable] = None):
        """设置回调函数"""
        self.detection_callback = detection_callback
        self.frame_callback = frame_callback
    
    def start_usb_camera(self, display: bool = True, save_video: Optional[str] = None):
        """启动USB摄像头检测"""
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"无法打开USB摄像头 {self.camera_id}")
        
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        cap.set(cv2.CAP_PROP_FPS, self.framerate)
        
        # 设置视频录制
        out = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_video, fourcc, self.framerate, self.resolution)
        
        self.is_running = True
        self.start_time = time.time()
        
        print("USB摄像头检测已启动，按 'q' 退出")
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                display_frame = frame.copy()
                
                # 按间隔执行检测
                if self.frame_count % self.detection_interval == 0:
                    results = self.model.predict(frame)
                    self.detection_count += len(results)
                    
                    # 绘制检测结果
                    display_frame = self.model.draw_results(frame, results)
                    
                    # 调用检测回调
                    if self.detection_callback:
                        self.detection_callback(frame, results)
                
                # 计算FPS
                elapsed = time.time() - self.start_time
                if elapsed > 0:
                    self.fps = self.frame_count / elapsed
                
                # 显示信息
                self._draw_info(display_frame)
                
                # 调用帧回调
                if self.frame_callback:
                    display_frame = self.frame_callback(display_frame)
                
                # 保存视频
                if out:
                    out.write(display_frame)
                
                # 显示图像
                if display:
                    cv2.imshow('USB摄像头检测', display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                        
        finally:
            cap.release()
            if out:
                out.release()
            if display:
                cv2.destroyAllWindows()
            self.is_running = False
    
    def start_pi_camera(self, display: bool = True, save_video: Optional[str] = None):
        """启动树莓派摄像头检测"""
        if not PICAMERA_AVAILABLE:
            raise ImportError("树莓派摄像头库未安装，请运行: pip install picamera")
        
        camera = PiCamera()
        camera.resolution = self.resolution
        camera.framerate = self.framerate
        
        # 预热摄像头
        time.sleep(2)
        
        # 设置视频录制
        out = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_video, fourcc, self.framerate, self.resolution)
        
        self.is_running = True
        self.start_time = time.time()
        
        print("树莓派摄像头检测已启动，按 'q' 退出")
        
        try:
            rawCapture = PiRGBArray(camera, size=self.resolution)
            
            for frame_data in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
                if not self.is_running:
                    break
                
                frame = frame_data.array
                self.frame_count += 1
                display_frame = frame.copy()
                
                # 按间隔执行检测
                if self.frame_count % self.detection_interval == 0:
                    results = self.model.predict(frame)
                    self.detection_count += len(results)
                    
                    # 绘制检测结果
                    display_frame = self.model.draw_results(frame, results)
                    
                    # 调用检测回调
                    if self.detection_callback:
                        self.detection_callback(frame, results)
                
                # 计算FPS
                elapsed = time.time() - self.start_time
                if elapsed > 0:
                    self.fps = self.frame_count / elapsed
                
                # 显示信息
                self._draw_info(display_frame)
                
                # 调用帧回调
                if self.frame_callback:
                    display_frame = self.frame_callback(display_frame)
                
                # 保存视频
                if out:
                    out.write(display_frame)
                
                # 显示图像
                if display:
                    cv2.imshow('树莓派摄像头检测', display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                
                # 清空缓冲区
                rawCapture.truncate(0)
                
        finally:
            camera.close()
            if out:
                out.release()
            if display:
                cv2.destroyAllWindows()
            self.is_running = False
    
    def start_detection(self, display: bool = True, save_video: Optional[str] = None):
        """启动检测（自动选择摄像头类型）"""
        if self.camera_type == 'picamera':
            self.start_pi_camera(display, save_video)
        else:
            self.start_usb_camera(display, save_video)
    
    def stop_detection(self):
        """停止检测"""
        self.is_running = False
    
    def _draw_info(self, frame: np.ndarray):
        """绘制信息到帧上"""
        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 帧数
        cv2.putText(frame, f"Frame: {self.frame_count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 检测数量
        cv2.putText(frame, f"Detections: {self.detection_count}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 摄像头类型
        cv2.putText(frame, f"Camera: {self.camera_type}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取检测统计信息"""
        elapsed = time.time() - self.start_time
        return {
            'camera_type': self.camera_type,
            'resolution': self.resolution,
            'framerate': self.framerate,
            'frame_count': self.frame_count,
            'detection_count': self.detection_count,
            'fps': self.fps,
            'elapsed_time': elapsed,
            'model_info': self.model.get_model_info()
        }
    
    def save_stats(self, filepath: str):
        """保存统计信息到文件"""
        stats = self.get_stats()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"统计信息已保存到: {filepath}")