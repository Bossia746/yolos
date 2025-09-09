#!/usr/bin/env python3
"""
YOLOS ESP32版本主文件
"""

import machine
import camera
import time
from src.core.minimal_yolos import MinimalYOLOS

def init_camera():
    """初始化摄像头"""
    try:
        camera.init(0, format=camera.JPEG, framesize=camera.FRAME_QVGA)
        print("摄像头初始化成功")
        return True
    except Exception as e:
        print(f"摄像头初始化失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 启动YOLOS ESP32版本...")
    
    # 初始化硬件
    if not init_camera():
        return
    
    # 启动检测循环
    yolos = MinimalYOLOS()
    
    while True:
        try:
            # 捕获图像
            buf = camera.capture()
            if buf:
                print(f"捕获图像，大小: {len(buf)} bytes")
                # 这里添加检测逻辑
                
            time.sleep(0.1)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"检测错误: {e}")
            time.sleep(1)
    
    camera.deinit()
    print("ESP32版本退出")

if __name__ == "__main__":
    main()
