#!/usr/bin/env python3
"""
YOLOS 实时检测示例
演示如何使用YOLOS进行实时摄像头检测
"""

import sys
import argparse
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent / "src"))

from detection.realtime_detector import RealtimeDetector


def detection_callback(frame, results):
    """检测结果回调函数"""
    if results:
        print(f"检测到 {len(results)} 个目标: {[r['class_name'] for r in results]}")


def main():
    parser = argparse.ArgumentParser(description='YOLOS实时检测示例')
    parser.add_argument('--model', default='yolov8', choices=['yolov5', 'yolov8', 'yolo-world'],
                       help='模型类型')
    parser.add_argument('--model-path', default=None, help='模型路径')
    parser.add_argument('--device', default='auto', help='设备类型')
    parser.add_argument('--camera', type=int, default=0, help='摄像头ID')
    parser.add_argument('--source', default=None, help='视频文件路径')
    
    args = parser.parse_args()
    
    print("YOLOS 实时检测示例")
    print("==================")
    print(f"模型: {args.model}")
    print(f"设备: {args.device}")
    
    # 创建实时检测器
    detector = RealtimeDetector(
        model_type=args.model,
        model_path=args.model_path,
        device=args.device
    )
    
    # 设置回调函数
    detector.set_detection_callback(detection_callback)
    
    try:
        if args.source:
            # 视频文件检测
            print(f"开始检测视频: {args.source}")
            detector.start_video_detection(args.source)
        else:
            # 摄像头检测
            print(f"开始摄像头检测 (ID: {args.camera})")
            print("按 'q' 键退出")
            detector.start_camera_detection(args.camera)
            
    except KeyboardInterrupt:
        print("\n检测被用户中断")
    except Exception as e:
        print(f"检测过程中出错: {e}")
    finally:
        detector.stop()
        print("检测结束")


if __name__ == "__main__":
    main()