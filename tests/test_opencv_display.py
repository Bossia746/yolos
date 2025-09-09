#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV窗口显示测试程序
用于诊断GUI显示问题的根本原因
"""

import cv2
import numpy as np
import time
import sys

def test_basic_window():
    """测试基础窗口显示"""
    print("=== OpenCV窗口显示测试 ===")
    print(f"OpenCV版本: {cv2.__version__}")
    print(f"Python版本: {sys.version}")
    
    # 创建测试图像
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (0, 100, 0)  # 绿色背景
    
    # 添加文字
    cv2.putText(img, 'OpenCV Display Test', (50, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    # 创建窗口
    window_name = "OpenCV Display Test"
    print(f"创建窗口: {window_name}")
    
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        print("窗口创建成功")
        
        # 显示图像
        cv2.imshow(window_name, img)
        print("图像显示命令已执行")
        
        # 检查窗口属性
        try:
            visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
            print(f"窗口可见性: {visible}")
        except Exception as e:
            print(f"无法获取窗口属性: {e}")
        
        print("等待5秒钟...")
        print("如果您能看到绿色窗口，说明OpenCV显示正常")
        print("按任意键或等待5秒后程序将退出")
        
        # 等待按键或超时
        key = cv2.waitKey(5000)
        if key != -1:
            print(f"检测到按键: {key}")
        else:
            print("超时，未检测到按键")
            
    except Exception as e:
        print(f"窗口显示错误: {e}")
        return False
    
    finally:
        cv2.destroyAllWindows()
        print("窗口已关闭")
    
    return True

def test_camera_display():
    """测试摄像头显示"""
    print("\n=== 摄像头显示测试 ===")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return False
    
    print("摄像头打开成功")
    window_name = "Camera Test"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while frame_count < 50:  # 只测试50帧
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头帧")
                break
            
            frame_count += 1
            
            # 添加帧计数
            cv2.putText(frame, f'Frame: {frame_count}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow(window_name, frame)
            
            if frame_count % 10 == 0:
                print(f"已处理 {frame_count} 帧")
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("用户请求退出")
                break
        
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"测试完成: {frame_count} 帧, 耗时 {elapsed:.2f}秒, FPS: {fps:.1f}")
        
    except Exception as e:
        print(f"摄像头显示错误: {e}")
        return False
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("摄像头已释放")
    
    return True

def main():
    """主函数"""
    print("开始OpenCV显示诊断...")
    
    # 测试基础窗口显示
    if not test_basic_window():
        print("❌ 基础窗口显示测试失败")
        return False
    
    print("✅ 基础窗口显示测试通过")
    
    # 测试摄像头显示
    if not test_camera_display():
        print("❌ 摄像头显示测试失败")
        return False
    
    print("✅ 摄像头显示测试通过")
    print("\n=== 诊断结果 ===")
    print("OpenCV显示功能正常，问题可能在于:")
    print("1. 多模态检测程序的窗口管理逻辑")
    print("2. 程序运行环境或线程问题")
    print("3. 窗口焦点或前台显示问题")
    
    return True

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n用户中断程序")
        cv2.destroyAllWindows()
        sys.exit(0)
    except Exception as e:
        print(f"程序异常: {e}")
        cv2.destroyAllWindows()
        sys.exit(1)