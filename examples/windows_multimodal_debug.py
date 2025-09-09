#!/usr/bin/env python3
"""Windows摄像头多模态识别调试示例

专门用于Windows平台的摄像头测试和多模态识别功能调试
支持手势识别、面部识别、身体姿势识别
"""

import sys
import os
import argparse
import cv2
import time
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent / "src"))

from recognition.multimodal_detector import MultimodalDetector


def detection_callback(frame, results):
    """检测结果回调函数"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # 打印手势识别结果
    gesture_results = results.get('gesture_results', {})
    if isinstance(gesture_results, dict) and gesture_results.get('hands_detected', 0) > 0:
        # 增强手势识别结果
        for hand_data in gesture_results.get('hands_data', []):
            static_gesture = hand_data.get('static_gesture', {})
            dynamic_gesture = hand_data.get('dynamic_gesture', {})
            handedness = hand_data.get('handedness', 'Unknown')
            print(f"[{timestamp}] 手势检测: {handedness} - 静态:{static_gesture.get('gesture', 'unknown')} 动态:{dynamic_gesture.get('gesture', 'none')}")
    elif isinstance(gesture_results, list) and gesture_results:
        # 基础手势识别结果
        for gesture in gesture_results:
            print(f"[{timestamp}] 手势检测: {gesture.get('hand_label', 'Unknown')} - {gesture.get('gesture_name', 'unknown')}")
    
    # 打印面部识别结果
    face_results = results.get('face_results', {})
    if isinstance(face_results, dict) and face_results.get('faces_detected', 0) > 0:
        # 增强面部识别结果
        for face_data in face_results.get('faces_data', []):
            identity = face_data.get('identity', 'Unknown')
            confidence = face_data.get('confidence', 0.0)
            print(f"[{timestamp}] 人脸检测: {identity} (置信度: {confidence:.2f})")
    elif isinstance(face_results, list) and face_results:
        # 基础面部识别结果
        for face in face_results:
            print(f"[{timestamp}] 人脸检测: {face.get('identity', 'Unknown')} (置信度: {face.get('confidence', 0.0):.2f})")
    
    # 打印身体姿势识别结果
    pose_results = results.get('pose_results', {})
    if pose_results.get('persons_detected', 0) > 0:
        # YOLOv7姿态识别结果
        for person_data in pose_results.get('persons', []):
            pose_type = person_data.get('pose_type', 'unknown')
            confidence = person_data.get('confidence', 0.0)
            print(f"[{timestamp}] 姿势检测: {pose_type} (置信度: {confidence:.2f})")
    elif pose_results.get('pose_detected', False):
        # 基础姿态识别结果
        pose_name = pose_results.get('pose_name', 'unknown')
        confidence = pose_results.get('confidence', 0.0)
        print(f"[{timestamp}] 姿势检测: {pose_name} (置信度: {confidence:.2f})")
    
    # 打印摔倒检测结果
    fall_results = results.get('fall_results', {})
    if fall_results.get('fall_detected', False):
        confidence = fall_results.get('fall_confidence', 0.0)
        duration = fall_results.get('fall_duration', 0.0)
        print(f"[{timestamp}] ⚠️ 摔倒检测: 检测到摔倒! (置信度: {confidence:.2f}, 持续时间: {duration:.1f}s)")


def test_camera_availability():
    """测试摄像头可用性"""
    print("正在测试摄像头可用性...")
    
    available_cameras = []
    for i in range(5):  # 测试前5个摄像头ID
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                height, width = frame.shape[:2]
                available_cameras.append({
                    'id': i,
                    'resolution': f"{width}x{height}",
                    'fps': cap.get(cv2.CAP_PROP_FPS)
                })
                print(f"摄像头 {i}: 可用 - 分辨率 {width}x{height}, FPS {cap.get(cv2.CAP_PROP_FPS)}")
            cap.release()
        else:
            print(f"摄像头 {i}: 不可用")
    
    if not available_cameras:
        print("警告: 未找到可用的摄像头!")
        return None
    
    return available_cameras


def create_face_database_interactive(detector):
    """交互式创建人脸数据库"""
    print("\n=== 人脸数据库管理 ===")
    print("1. 添加新人脸")
    print("2. 查看数据库信息")
    print("3. 保存数据库")
    print("4. 加载数据库")
    print("5. 跳过")
    
    choice = input("请选择操作 (1-5): ").strip()
    
    if choice == '1':
        name = input("请输入人脸标识名称: ").strip()
        if name:
            print(f"请面向摄像头，将为 '{name}' 采集人脸数据...")
            print("按空格键采集人脸，按ESC键取消")
            
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                while True:
                    ret, frame = cap.read()
                    if ret:
                        cv2.imshow('人脸采集 - 按空格键采集', frame)
                        key = cv2.waitKey(1) & 0xFF
                        
                        if key == ord(' '):  # 空格键采集
                            success = detector.add_face_to_database(frame, name)
                            if success:
                                print(f"成功添加人脸: {name}")
                            else:
                                print(f"添加人脸失败: {name}")
                            break
                        elif key == 27:  # ESC键取消
                            print("取消人脸采集")
                            break
                
                cap.release()
                cv2.destroyAllWindows()
    
    elif choice == '2':
        info = detector.get_face_database_info()
        print(f"人脸数据库信息: {info}")
    
    elif choice == '3':
        filepath = input("请输入保存路径 (默认: face_database.pkl): ").strip()
        if not filepath:
            filepath = "face_database.pkl"
        detector.save_face_database(filepath)
    
    elif choice == '4':
        filepath = input("请输入数据库文件路径: ").strip()
        if filepath and os.path.exists(filepath):
            detector.load_face_database(filepath)
        else:
            print("文件不存在")


def main():
    parser = argparse.ArgumentParser(description='Windows摄像头多模态识别调试')
    parser.add_argument('--camera', type=int, default=0, help='摄像头ID (默认: 0)')
    parser.add_argument('--no-gesture', action='store_true', help='禁用手势识别')
    parser.add_argument('--no-face', action='store_true', help='禁用面部识别')
    parser.add_argument('--no-pose', action='store_true', help='禁用身体姿势识别')
    parser.add_argument('--face-db', type=str, help='人脸数据库文件路径')
    parser.add_argument('--interval', type=int, default=1, help='检测间隔 (帧数, 默认: 1)')
    parser.add_argument('--save-video', type=str, help='保存视频文件路径')
    parser.add_argument('--test-camera', action='store_true', help='仅测试摄像头可用性')
    
    args = parser.parse_args()
    
    print("Windows摄像头多模态识别调试工具")
    print("=" * 50)
    
    # 测试摄像头
    available_cameras = test_camera_availability()
    if args.test_camera:
        return
    
    if not available_cameras:
        print("错误: 没有可用的摄像头")
        return
    
    # 检查指定的摄像头是否可用
    camera_ids = [cam['id'] for cam in available_cameras]
    if args.camera not in camera_ids:
        print(f"警告: 摄像头 {args.camera} 不可用，使用摄像头 {camera_ids[0]}")
        args.camera = camera_ids[0]
    
    print(f"\n使用摄像头: {args.camera}")
    print(f"启用功能: 手势识别={not args.no_gesture}, 面部识别={not args.no_face}, 身体姿势识别={not args.no_pose}")
    
    try:
        # 创建多模态检测器 - 启用增强算法
        detector = MultimodalDetector(
            enable_gesture=not args.no_gesture,
            enable_face=not args.no_face,
            enable_pose=not args.no_pose,
            face_database_path=args.face_db,
            detection_interval=args.interval,
            use_enhanced_algorithms=True,  # 启用增强算法
            enable_fall_detection=True     # 启用摔倒检测
        )
        
        # 设置回调函数
        detector.set_callbacks(detection_callback=detection_callback)
        
        # 交互式人脸数据库管理
        if not args.no_face:
            create_face_database_interactive(detector)
        
        # 显示支持的手势和姿势
        if not args.no_gesture:
            gestures = detector.get_supported_gestures()
            print(f"\n支持的手势: {list(gestures.values())}")
        
        if not args.no_pose:
            poses = detector.get_supported_poses()
            print(f"支持的姿势: {list(poses.values())}")
        
        print("\n=== 控制说明 ===")
        print("q: 退出程序")
        print("s: 保存当前帧")
        print("r: 重置统计信息")
        print("f: 添加当前帧中的人脸到数据库")
        print("d: 显示详细检测信息")
        print("h: 显示/隐藏帮助信息")
        
        # 开始检测
        print("\n开始多模态识别检测...")
        
        # 使用自定义循环以支持更多交互功能
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {args.camera}")
        
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # 视频写入器
        video_writer = None
        if args.save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(args.save_video, fourcc, 20.0, (640, 480))
        
        show_help = True
        detailed_info = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头帧")
                break
            
            # 执行检测
            annotated_frame, results = detector.detect(frame)
            
            # 添加帮助信息
            if show_help:
                help_text = [
                    "按键: q=退出 s=保存 r=重置 f=添加人脸 d=详细信息 h=隐藏帮助"
                ]
                for i, text in enumerate(help_text):
                    cv2.putText(annotated_frame, text, (10, annotated_frame.shape[0] - 30 + i * 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # 显示详细信息
            if detailed_info:
                info_text = [
                    f"处理时间: {results['processing_time']:.3f}s",
                    f"手势数: {len(results.get('gesture_results', []))}",
                    f"人脸数: {len(results.get('face_results', []))}",
                    f"姿势检测: {'是' if results.get('pose_results', {}).get('pose_detected') else '否'}"
                ]
                for i, text in enumerate(info_text):
                    cv2.putText(annotated_frame, text, (annotated_frame.shape[1] - 200, 30 + i * 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # 保存视频
            if video_writer:
                video_writer.write(annotated_frame)
            
            # 显示结果
            cv2.imshow('Windows多模态识别调试', annotated_frame)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存当前帧
                timestamp = int(time.time())
                filename = f"debug_frame_{timestamp}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"已保存帧: {filename}")
            elif key == ord('r'):
                # 重置统计信息
                detector.reset_stats()
                print("统计信息已重置")
            elif key == ord('f'):
                # 添加人脸到数据库
                name = input("\n请输入人脸标识名称: ").strip()
                if name:
                    success = detector.add_face_to_database(frame, name)
                    if success:
                        print(f"成功添加人脸: {name}")
                    else:
                        print(f"添加人脸失败: {name}")
            elif key == ord('d'):
                # 切换详细信息显示
                detailed_info = not detailed_info
                print(f"详细信息显示: {'开启' if detailed_info else '关闭'}")
            elif key == ord('h'):
                # 切换帮助信息显示
                show_help = not show_help
                print(f"帮助信息显示: {'开启' if show_help else '关闭'}")
        
        # 显示最终统计信息
        final_stats = detector.get_stats()
        print("\n=== 最终统计信息 ===")
        for key, value in final_stats.items():
            print(f"{key}: {value}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        if 'cap' in locals():
            cap.release()
        if 'video_writer' in locals() and video_writer:
            video_writer.release()
        if 'detector' in locals():
            detector.close()
        cv2.destroyAllWindows()
        print("程序已退出")


if __name__ == "__main__":
    main()