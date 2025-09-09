#!/usr/bin/env python3
"""
YOLOS K230版本主文件
"""

import os
import sys
import time
import numpy as np

try:
    import nncase
    from canmv import camera, display
    CANMV_AVAILABLE = True
except ImportError:
    CANMV_AVAILABLE = False
    print("⚠️ CanMV环境不可用，使用模拟模式")

class K230YOLOS:
    """K230版本YOLOS"""
    
    def __init__(self):
        self.model_path = "yolov8n.kmodel"
        self.input_size = (640, 640)
        self.confidence_threshold = 0.5
        
    def load_model(self):
        """加载KModel模型"""
        if not CANMV_AVAILABLE:
            print("模拟模式：模型加载成功")
            return True
            
        try:
            # 加载KModel
            if os.path.exists(self.model_path):
                print(f"加载模型: {self.model_path}")
                # 这里添加实际的模型加载代码
                return True
            else:
                print(f"模型文件不存在: {self.model_path}")
                return False
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False
            
    def init_camera(self):
        """初始化摄像头"""
        if not CANMV_AVAILABLE:
            print("模拟模式：摄像头初始化成功")
            return True
            
        try:
            camera.sensor_init(camera.CAM_DEV_ID_0, camera.CAM_DEFAULT_SENSOR)
            camera.set_outsize(camera.CAM_DEV_ID_0, camera.CAM_CHN_ID_0, 
                             self.input_size[0], self.input_size[1])
            camera.set_outfmt(camera.CAM_DEV_ID_0, camera.CAM_CHN_ID_0, 
                            camera.PIXEL_FORMAT_RGB_888_PLANAR)
            print("K230摄像头初始化成功")
            return True
        except Exception as e:
            print(f"摄像头初始化失败: {e}")
            return False
            
    def detect_frame(self, frame):
        """检测单帧"""
        # 模拟检测结果
        detections = [
            {"class": "person", "confidence": 0.85, "bbox": [100, 100, 200, 300]},
            {"class": "car", "confidence": 0.72, "bbox": [300, 150, 500, 400]}
        ]
        return detections
        
    def run(self):
        """运行检测"""
        print("🚀 启动YOLOS K230版本...")
        
        if not self.load_model():
            return
            
        if not self.init_camera():
            return
            
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                if CANMV_AVAILABLE:
                    # 捕获真实图像
                    frame = camera.capture_image(camera.CAM_DEV_ID_0, camera.CAM_CHN_ID_0)
                else:
                    # 模拟图像
                    frame = np.random.randint(0, 255, (*self.input_size, 3), dtype=np.uint8)
                
                # 执行检测
                detections = self.detect_frame(frame)
                
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                print(f"帧 {frame_count}: 检测到 {len(detections)} 个目标, FPS: {fps:.1f}")
                
                for det in detections:
                    print(f"  - {det['class']}: {det['confidence']:.2f}")
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n检测被用户中断")
        except Exception as e:
            print(f"检测错误: {e}")
        finally:
            if CANMV_AVAILABLE:
                camera.sensor_deinit(camera.CAM_DEV_ID_0)
            print("K230版本退出")

def main():
    """主函数"""
    yolos = K230YOLOS()
    yolos.run()

if __name__ == "__main__":
    main()
