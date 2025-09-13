#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运动识别示例脚本
展示如何使用姿态识别系统进行各种运动计数
"""

import cv2
import sys
import os
from pathlib import Path
import logging

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.recognition.exercise_factory import ExerciseFactory, create_pushup_counter
from src.recognition.pose_recognition import ExerciseType

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demo_pushup_recognition():
    """
    俯卧撑识别演示
    """
    print("=== 俯卧撑识别演示 ===")
    
    # 创建俯卧撑识别器
    recognizer = create_pushup_counter('balanced')
    
    # 模拟视频文件路径（用户需要提供实际视频）
    video_path = "pushups_demo.mp4"  # 替换为实际视频路径
    
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        print("请提供俯卧撑演示视频文件")
        return
    
    # 处理视频
    try:
        results = recognizer.process_video(video_path, output_path="pushup_result.mp4")
        print(f"俯卧撑计数结果: {results.total_count}")
        print(f"平均角度: {results.average_angle:.1f}°")
        print(f"处理帧数: {results.total_frames}")
    except Exception as e:
        logger.error(f"处理视频失败: {e}")

def demo_squat_recognition():
    """
    深蹲识别演示
    """
    print("\n=== 深蹲识别演示 ===")
    
    # 创建深蹲识别器
    recognizer = ExerciseFactory.create_squat_recognizer('balanced', deep_squat=True)
    
    video_path = "squats_demo.mp4"  # 替换为实际视频路径
    
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        print("请提供深蹲演示视频文件")
        return
    
    try:
        results = recognizer.process_video(video_path, output_path="squat_result.mp4")
        print(f"深蹲计数结果: {results.total_count}")
        print(f"平均角度: {results.average_angle:.1f}°")
    except Exception as e:
        logger.error(f"处理视频失败: {e}")

def demo_realtime_recognition():
    """
    实时识别演示（使用摄像头）
    """
    print("\n=== 实时识别演示 ===")
    
    # 创建识别器
    recognizer = ExerciseFactory.create_pushup_recognizer('fast')  # 使用快速模型
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("按 'q' 退出实时识别")
    print("按 'r' 重置计数")
    print("按 's' 切换到深蹲模式")
    
    current_mode = 'pushup'
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 处理帧
            result_frame, stats = recognizer.process_frame(frame)
            
            # 显示信息
            cv2.putText(result_frame, f"Mode: {current_mode.upper()}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result_frame, f"Count: {stats.total_count}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result_frame, f"Angle: {stats.current_angle:.1f}", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result_frame, f"State: {stats.current_state.value}", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Exercise Recognition', result_frame)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                recognizer.reset_stats()
                print("计数已重置")
            elif key == ord('s'):
                # 切换模式
                if current_mode == 'pushup':
                    recognizer = ExerciseFactory.create_squat_recognizer('fast')
                    current_mode = 'squat'
                else:
                    recognizer = ExerciseFactory.create_pushup_recognizer('fast')
                    current_mode = 'pushup'
                print(f"切换到 {current_mode} 模式")
    
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # 显示最终统计
        final_stats = recognizer.get_statistics()
        print(f"\n最终统计:")
        print(f"总计数: {final_stats.total_count}")
        print(f"平均角度: {final_stats.average_angle:.1f}°")
        print(f"处理帧数: {final_stats.total_frames}")

def demo_multi_exercise_session():
    """
    多运动会话演示
    """
    print("\n=== 多运动会话演示 ===")
    
    # 创建多运动会话
    exercises = ['pushup_standard', 'squat_standard', 'high_knee_left']
    session = ExerciseFactory.create_multi_exercise_session(exercises, 'balanced')
    
    print(f"创建了 {len(session)} 种运动识别器:")
    for name, recognizer in session.items():
        print(f"  - {name}: {recognizer.exercise_type.value}")
    
    # 模拟处理不同运动的视频
    video_files = {
        'pushup_standard': 'pushups.mp4',
        'squat_standard': 'squats.mp4',
        'high_knee_left': 'high_knees.mp4'
    }
    
    results = {}
    for exercise_name, recognizer in session.items():
        video_path = video_files.get(exercise_name)
        if video_path and os.path.exists(video_path):
            try:
                result = recognizer.process_video(video_path)
                results[exercise_name] = result
                print(f"{exercise_name}: {result.total_count} 次")
            except Exception as e:
                print(f"{exercise_name} 处理失败: {e}")
        else:
            print(f"{exercise_name}: 视频文件不存在 ({video_path})")
    
    return results

def demo_custom_exercise():
    """
    自定义运动演示
    """
    print("\n=== 自定义运动演示 ===")
    
    # 创建自定义运动识别器（例如：仰卧起坐）
    # 使用肩膀、髋关节、膝盖作为关键点
    recognizer = ExerciseFactory.create_custom_recognizer(
        keypoints=[5, 11, 13],  # 左肩、左髋、左膝
        up_angle=120.0,         # 起身角度
        down_angle=60.0,        # 躺下角度
        exercise_type=ExerciseType.CUSTOM,
        model_quality='balanced',
        angle_tolerance=20.0
    )
    
    print("创建了自定义运动识别器（仰卧起坐）")
    print(f"关键点: [5, 11, 13] (左肩、左髋、左膝)")
    print(f"角度范围: 60° - 120°")
    
    # 如果有视频文件，可以处理
    video_path = "situps.mp4"
    if os.path.exists(video_path):
        try:
            results = recognizer.process_video(video_path, output_path="situp_result.mp4")
            print(f"仰卧起坐计数: {results.total_count}")
        except Exception as e:
            print(f"处理失败: {e}")
    else:
        print(f"视频文件不存在: {video_path}")

def show_available_options():
    """
    显示可用选项
    """
    print("=== 可用配置信息 ===")
    
    print("\n运动预设:")
    presets = ExerciseFactory.get_available_presets()
    for name, desc in presets.items():
        print(f"  {name}: {desc}")
    
    print("\n模型选项:")
    models = ExerciseFactory.get_model_info()
    for quality, desc in models.items():
        print(f"  {quality}: {desc}")
    
    print("\n关键点信息:")
    keypoints = ExerciseFactory.get_keypoint_info()
    for idx, name in keypoints.items():
        print(f"  {idx}: {name}")
    
    print("\n设备推荐:")
    devices = ['desktop', 'laptop', 'mobile', 'embedded']
    for device in devices:
        config = ExerciseFactory.recommend_config(device)
        print(f"  {device}: {config['model_quality']} - {config['description']}")

def interactive_demo():
    """
    交互式演示
    """
    print("\n=== 交互式演示 ===")
    print("选择演示模式:")
    print("1. 俯卧撑识别")
    print("2. 深蹲识别")
    print("3. 实时识别（摄像头）")
    print("4. 多运动会话")
    print("5. 自定义运动")
    print("6. 显示配置信息")
    print("0. 退出")
    
    while True:
        try:
            choice = input("\n请选择 (0-6): ").strip()
            
            if choice == '0':
                print("退出演示")
                break
            elif choice == '1':
                demo_pushup_recognition()
            elif choice == '2':
                demo_squat_recognition()
            elif choice == '3':
                demo_realtime_recognition()
            elif choice == '4':
                demo_multi_exercise_session()
            elif choice == '5':
                demo_custom_exercise()
            elif choice == '6':
                show_available_options()
            else:
                print("无效选择，请重试")
        
        except KeyboardInterrupt:
            print("\n用户中断")
            break
        except Exception as e:
            print(f"演示出错: {e}")

def main():
    """
    主函数
    """
    print("运动识别系统演示")
    print("=" * 50)
    
    # 检查依赖
    try:
        import ultralytics
        print(f"Ultralytics 版本: {ultralytics.__version__}")
    except ImportError:
        print("警告: 未安装 ultralytics，请运行: pip install ultralytics")
        return
    
    # 显示系统信息
    print(f"OpenCV 版本: {cv2.__version__}")
    print(f"项目根目录: {project_root}")
    
    # 运行演示
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == 'pushup':
            demo_pushup_recognition()
        elif mode == 'squat':
            demo_squat_recognition()
        elif mode == 'realtime':
            demo_realtime_recognition()
        elif mode == 'multi':
            demo_multi_exercise_session()
        elif mode == 'custom':
            demo_custom_exercise()
        elif mode == 'info':
            show_available_options()
        else:
            print(f"未知模式: {mode}")
            interactive_demo()
    else:
        interactive_demo()

if __name__ == "__main__":
    main()