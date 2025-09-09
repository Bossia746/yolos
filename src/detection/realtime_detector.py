"""
实时检测器
"""

import cv2
import time
import threading
from queue import Queue
from typing import Optional, Callable, Dict, Any
import numpy as np

from ..models.yolo_factory import YOLOFactory


class RealtimeDetector:
    """实时检测器"""
    
    def __init__(self, 
                 model_type: str = 'yolov8',
                 model_path: Optional[str] = None,
                 device: str = 'auto'):
        """
        初始化实时检测器
        
        Args:
            model_type: 模型类型
            model_path: 模型路径
            device: 设备类型
        """
        self.model = YOLOFactory.create_model(model_type, model_path, device)
        self.is_running = False
        self.frame_queue = Queue(maxsize=10)
        self.result_queue = Queue(maxsize=10)
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # 回调函数
        self.detection_callback: Optional[Callable] = None
        self.frame_callback: Optional[Callable] = None
    
    def set_detection_callback(self, callback: Callable):
        """设置检测结果回调函数"""
        self.detection_callback = callback
    
    def set_frame_callback(self, callback: Callable):
        """设置帧处理回调函数"""
        self.frame_callback = callback
    
    def _detection_worker(self):
        """检测工作线程"""
        while self.is_running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get(timeout=0.1)
                    
                    # 执行检测
                    results = self.model.predict(frame)
                    
                    # 放入结果队列
                    if not self.result_queue.full():
                        self.result_queue.put((frame, results))
                    
                    # 调用检测回调
                    if self.detection_callback:
                        self.detection_callback(frame, results)
                        
            except Exception as e:
                print(f"检测线程错误: {e}")
                time.sleep(0.01)
    
    def start_camera_detection(self, camera_id: int = 0, window_name: str = "实时检测"):
        """开始摄像头检测"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {camera_id}")
        
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_running = True
        
        # 启动检测线程
        detection_thread = threading.Thread(target=self._detection_worker)
        detection_thread.daemon = True
        detection_thread.start()
        
        print("开始实时检测，按 'q' 退出")
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 添加帧到队列
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())
                
                # 获取检测结果并显示
                display_frame = frame.copy()
                if not self.result_queue.empty():
                    result_frame, results = self.result_queue.get()
                    display_frame = self.model.draw_results(result_frame, results)
                
                # 计算FPS
                self.frame_count += 1
                elapsed_time = time.time() - self.start_time
                if elapsed_time > 1.0:
                    self.fps = self.frame_count / elapsed_time
                    self.frame_count = 0
                    self.start_time = time.time()
                
                # 显示FPS
                cv2.putText(display_frame, f"FPS: {self.fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 调用帧回调
                if self.frame_callback:
                    display_frame = self.frame_callback(display_frame)
                
                # 显示图像
                cv2.imshow(window_name, display_frame)
                
                # 检查退出条件
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
        finally:
            self.stop()
            cap.release()
            cv2.destroyAllWindows()
    
    def start_video_detection(self, video_path: str, output_path: Optional[str] = None):
        """开始视频检测"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {video_path}")
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 设置输出视频
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        self.is_running = True
        frame_idx = 0
        
        print(f"开始处理视频: {video_path}")
        print(f"总帧数: {total_frames}, FPS: {fps}")
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 执行检测
                results = self.model.predict(frame)
                
                # 绘制结果
                annotated_frame = self.model.draw_results(frame, results)
                
                # 保存或显示
                if out:
                    out.write(annotated_frame)
                else:
                    cv2.imshow('视频检测', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # 调用回调
                if self.detection_callback:
                    self.detection_callback(frame, results)
                
                # 显示进度
                frame_idx += 1
                if frame_idx % 30 == 0:
                    progress = (frame_idx / total_frames) * 100
                    print(f"处理进度: {progress:.1f}% ({frame_idx}/{total_frames})")
                    
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            print("视频处理完成")
    
    def stop(self):
        """停止检测"""
        self.is_running = False
    
    def get_fps(self) -> float:
        """获取当前FPS"""
        return self.fps
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return self.model.get_model_info()