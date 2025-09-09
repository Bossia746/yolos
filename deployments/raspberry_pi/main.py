#!/usr/bin/env python3
"""
YOLOS 树莓派版本主文件
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False

class RaspberryPiYOLOS:
    """树莓派版本YOLOS"""
    
    def __init__(self):
        self.camera = None
        self.model_path = "yolov8n.tflite"
        self.input_size = (640, 640)
        
    def init_camera(self):
        """初始化摄像头"""
        if PICAMERA_AVAILABLE:
            try:
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                    main={"size": self.input_size}
                )
                self.camera.configure(config)
                self.camera.start()
                print("树莓派摄像头初始化成功")
                return True
            except Exception as e:
                print(f"PiCamera初始化失败: {e}")
                
        if OPENCV_AVAILABLE:
            try:
                self.camera = cv2.VideoCapture(0)
                if self.camera.isOpened():
                    print("USB摄像头初始化成功")
                    return True
            except Exception as e:
                print(f"USB摄像头初始化失败: {e}")
                
        print("⚠️ 无可用摄像头，使用模拟模式")
        return False
        
    def capture_frame(self):
        """捕获帧"""
        if PICAMERA_AVAILABLE and hasattr(self, 'camera') and self.camera:
            try:
                frame = self.camera.capture_array()
                return frame
            except Exception as e:
                print(f"PiCamera捕获失败: {e}")
                
        if OPENCV_AVAILABLE and hasattr(self, 'camera') and self.camera:
            try:
                ret, frame = self.camera.read()
                if ret:
                    return frame
            except Exception as e:
                print(f"USB摄像头捕获失败: {e}")
                
        # 返回模拟帧
        return np.random.randint(0, 255, (*self.input_size, 3), dtype=np.uint8)
        
    def detect_frame(self, frame):
        """检测帧"""
        # 模拟检测结果
        detections = [
            {"class": "person", "confidence": 0.85, "bbox": [100, 100, 200, 300]}
        ]
        return detections
        
    def run(self):
        """运行检测"""
        print("🚀 启动YOLOS 树莓派版本...")
        
        self.init_camera()
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                frame = self.capture_frame()
                detections = self.detect_frame(frame)
                
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                print(f"帧 {frame_count}: 检测到 {len(detections)} 个目标, FPS: {fps:.1f}")
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n检测被用户中断")
        finally:
            if hasattr(self, 'camera') and self.camera:
                if PICAMERA_AVAILABLE:
                    self.camera.stop()
                elif OPENCV_AVAILABLE:
                    self.camera.release()
            print("树莓派版本退出")

def main():
    """主函数"""
    yolos = RaspberryPiYOLOS()
    yolos.run()

if __name__ == "__main__":
    main()
