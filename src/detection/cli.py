#!/usr/bin/env python3
"""
YOLOS 命令行检测工具
"""

import argparse
import sys
from pathlib import Path

from .image_detector import ImageDetector
from .video_detector import VideoDetector
from .realtime_detector import RealtimeDetector
from .camera_detector import CameraDetector


def detect_image_command(args):
    """图像检测命令"""
    detector = ImageDetector(
        model_type=args.model,
        model_path=args.weights,
        device=args.device
    )
    
    if args.batch:
        # 批量检测
        results = detector.detect_batch(
            args.source,
            output_dir=args.output,
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )
        print(f"批量检测完成，处理了 {len(results)} 个文件")
    else:
        # 单张图像检测
        results = detector.detect_image(
            args.source[0],
            output_path=args.output,
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )
        print(f"检测完成，发现 {len(results)} 个目标")


def detect_video_command(args):
    """视频检测命令"""
    detector = VideoDetector(
        model_type=args.model,
        model_path=args.weights,
        device=args.device
    )
    
    detector.detect_video(
        args.source,
        output_path=args.output,
        save_frames=args.save_frames,
        frame_interval=args.interval
    )


def detect_realtime_command(args):
    """实时检测命令"""
    if args.camera:
        # 摄像头检测
        detector = CameraDetector(
            model_type=args.model,
            model_path=args.weights,
            device=args.device,
            camera_type=args.camera_type
        )
        
        detector.set_camera_params(
            resolution=tuple(args.resolution),
            framerate=args.fps
        )
        
        detector.start_detection(
            display=not args.no_display,
            save_video=args.output
        )
    else:
        # 实时检测器
        detector = RealtimeDetector(
            model_type=args.model,
            model_path=args.weights,
            device=args.device
        )
        
        if args.source.isdigit():
            # 摄像头ID
            detector.start_camera_detection(int(args.source))
        else:
            # 视频文件
            detector.start_video_detection(args.source, args.output)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='YOLOS - 多平台AIoT视觉大模型检测工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检测单张图像
  yolos-detect image photo.jpg --output result.jpg
  
  # 批量检测图像
  yolos-detect image *.jpg --batch --output results/
  
  # 检测视频
  yolos-detect video input.mp4 --output output.mp4
  
  # 实时摄像头检测
  yolos-detect realtime --camera --camera-type usb
  
  # 树莓派摄像头检测
  yolos-detect realtime --camera --camera-type picamera
        """
    )
    
    # 全局参数
    parser.add_argument('--model', default='yolov8', 
                       choices=['yolov5', 'yolov8', 'yolo-world'],
                       help='模型类型')
    parser.add_argument('--weights', help='模型权重路径')
    parser.add_argument('--device', default='auto', help='设备类型 (auto/cpu/cuda)')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.7, help='IoU阈值')
    parser.add_argument('--output', help='输出路径')
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='检测模式')
    
    # 图像检测
    image_parser = subparsers.add_parser('image', help='图像检测')
    image_parser.add_argument('source', nargs='+', help='输入图像路径')
    image_parser.add_argument('--batch', action='store_true', help='批量处理')
    
    # 视频检测
    video_parser = subparsers.add_parser('video', help='视频检测')
    video_parser.add_argument('source', help='输入视频路径')
    video_parser.add_argument('--save-frames', action='store_true', help='保存检测帧')
    video_parser.add_argument('--interval', type=int, default=1, help='检测间隔')
    
    # 实时检测
    realtime_parser = subparsers.add_parser('realtime', help='实时检测')
    realtime_parser.add_argument('--source', default='0', help='输入源 (摄像头ID或视频文件)')
    realtime_parser.add_argument('--camera', action='store_true', help='使用摄像头检测器')
    realtime_parser.add_argument('--camera-type', default='usb', 
                                choices=['usb', 'picamera'], help='摄像头类型')
    realtime_parser.add_argument('--resolution', nargs=2, type=int, default=[640, 480],
                                help='分辨率 (宽 高)')
    realtime_parser.add_argument('--fps', type=int, default=30, help='帧率')
    realtime_parser.add_argument('--no-display', action='store_true', help='不显示窗口')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'image':
            detect_image_command(args)
        elif args.command == 'video':
            detect_video_command(args)
        elif args.command == 'realtime':
            detect_realtime_command(args)
    except KeyboardInterrupt:
        print("\n检测被用户中断")
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()