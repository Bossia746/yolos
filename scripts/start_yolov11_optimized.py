#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOS YOLOv11优化系统启动脚本
快速启动优化后的检测系统
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import load_config
from src.detection.factory import DetectorFactory
from src.models.optimized_yolov11_system import OptimizedRealtimeDetector, OptimizationConfig
from src.utils.logging_manager import LoggingManager


def create_optimized_config(args):
    """创建优化配置"""
    config = OptimizationConfig(
        model_size=args.model_size,
        device=args.device,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        target_fps=args.fps,
        platform=args.platform,
        adaptive_inference=args.adaptive,
        edge_optimization=args.edge_opt,
        half_precision=args.half_precision,
        tensorrt_optimize=args.tensorrt
    )
    return config


def start_camera_detection(args):
    """启动摄像头检测"""
    print("🚀 启动YOLOv11优化摄像头检测...")
    
    # 创建配置
    config = create_optimized_config(args)
    
    # 创建检测器
    detector = OptimizedRealtimeDetector(config)
    
    print(f"📹 摄像头ID: {args.camera_id}")
    print(f"🎯 模型: YOLOv11{args.model_size.upper()}")
    print(f"💻 设备: {args.device}")
    print(f"🎮 平台: {args.platform}")
    print(f"⚡ 目标FPS: {args.fps}")
    print(f"🧠 自适应推理: {'启用' if args.adaptive else '禁用'}")
    print(f"🔧 边缘优化: {'启用' if args.edge_opt else '禁用'}")
    print("\n按 'q' 退出检测")
    
    try:
        detector.start_camera_detection(args.camera_id)
    except KeyboardInterrupt:
        print("\n🛑 用户中断检测")
    except Exception as e:
        print(f"❌ 检测失败: {e}")
    finally:
        detector.stop()
        print("✅ 检测器已停止")


def start_video_detection(args):
    """启动视频检测"""
    print(f"🎬 启动YOLOv11优化视频检测: {args.video_path}")
    
    # 创建配置
    config = create_optimized_config(args)
    
    # 创建检测系统
    from src.models.optimized_yolov11_system import OptimizedYOLOv11System
    detector_system = OptimizedYOLOv11System(config)
    
    # 处理视频
    import cv2
    import numpy as np
    
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {args.video_path}")
        return
    
    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 视频信息: {width}x{height}, {fps}FPS, {total_frames}帧")
    
    # 设置输出视频
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"💾 输出视频: {args.output}")
    
    frame_idx = 0
    
    try:
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 执行检测
            results = detector_system.detect_adaptive(frame)
            
            # 绘制结果
            for result in results:
                bbox = result.bbox
                cv2.rectangle(frame, (bbox.x, bbox.y), (bbox.x2, bbox.y2), (0, 255, 0), 2)
                label = f"{result.class_name} {result.confidence:.2f}"
                cv2.putText(frame, label, (bbox.x, bbox.y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 保存或显示
            if out:
                out.write(frame)
            else:
                cv2.imshow('YOLOv11视频检测', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # 显示进度
            frame_idx += 1
            if frame_idx % 30 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"📈 处理进度: {progress:.1f}% ({frame_idx}/{total_frames})")
                
    except KeyboardInterrupt:
        print("\n🛑 用户中断处理")
    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # 显示性能统计
        stats = detector_system.get_performance_stats()
        print(f"\n📊 性能统计:")
        print(f"   平均FPS: {stats.get('avg_fps', 0):.1f}")
        print(f"   平均检测时间: {stats.get('avg_inference_time', 0)*1000:.1f}ms")
        print(f"   总检测次数: {stats.get('total_inferences', 0)}")
        print("✅ 视频处理完成")


def benchmark_performance(args):
    """性能基准测试"""
    print("🏃 启动YOLOv11性能基准测试...")
    
    # 创建配置
    config = create_optimized_config(args)
    
    # 创建检测系统
    from src.models.optimized_yolov11_system import OptimizedYOLOv11System
    detector_system = OptimizedYOLOv11System(config)
    
    # 生成测试图像
    import numpy as np
    test_images = []
    for i in range(args.test_frames):
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        test_images.append(image)
    
    print(f"🖼️  测试图像数量: {args.test_frames}")
    print(f"🎯 模型: YOLOv11{args.model_size.upper()}")
    
    # 执行基准测试
    import time
    
    print("⏱️  开始基准测试...")
    start_time = time.time()
    
    total_detections = 0
    for i, image in enumerate(test_images):
        results = detector_system.detect_adaptive(image)
        total_detections += len(results)
        
        if (i + 1) % 10 == 0:
            print(f"   已处理: {i + 1}/{args.test_frames}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 显示结果
    print(f"\n📊 基准测试结果:")
    print(f"   总处理时间: {total_time:.2f}秒")
    print(f"   平均FPS: {args.test_frames / total_time:.1f}")
    print(f"   总检测数量: {total_detections}")
    print(f"   平均每帧检测: {total_detections / args.test_frames:.1f}")
    
    # 获取详细统计
    stats = detector_system.get_performance_stats()
    print(f"   平均推理时间: {stats.get('avg_inference_time', 0)*1000:.1f}ms")
    print(f"   最小推理时间: {stats.get('min_inference_time', 0)*1000:.1f}ms")
    print(f"   最大推理时间: {stats.get('max_inference_time', 0)*1000:.1f}ms")
    
    print("✅ 基准测试完成")


def export_optimized_model(args):
    """导出优化模型"""
    print(f"📦 导出YOLOv11优化模型...")
    
    # 创建配置
    config = create_optimized_config(args)
    
    # 创建检测系统
    from src.models.optimized_yolov11_system import OptimizedYOLOv11System
    detector_system = OptimizedYOLOv11System(config)
    
    print(f"🎯 模型: YOLOv11{args.model_size.upper()}")
    print(f"📱 平台: {args.platform}")
    print(f"📄 格式: {args.export_format}")
    
    try:
        exported_path = detector_system.export_optimized_model(
            format=args.export_format,
            output_path=args.output
        )
        print(f"✅ 模型导出成功: {exported_path}")
    except Exception as e:
        print(f"❌ 模型导出失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="YOLOS YOLOv11优化系统")
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 摄像头检测命令
    camera_parser = subparsers.add_parser('camera', help='摄像头实时检测')
    camera_parser.add_argument('--camera-id', type=int, default=0, help='摄像头ID')
    
    # 视频检测命令
    video_parser = subparsers.add_parser('video', help='视频文件检测')
    video_parser.add_argument('video_path', help='视频文件路径')
    video_parser.add_argument('--output', '-o', help='输出视频路径')
    
    # 基准测试命令
    benchmark_parser = subparsers.add_parser('benchmark', help='性能基准测试')
    benchmark_parser.add_argument('--test-frames', type=int, default=100, help='测试帧数')
    
    # 模型导出命令
    export_parser = subparsers.add_parser('export', help='导出优化模型')
    export_parser.add_argument('--format', dest='export_format', default='onnx', 
                              choices=['onnx', 'tensorrt', 'tflite', 'coreml'],
                              help='导出格式')
    export_parser.add_argument('--output', '-o', help='输出文件路径')
    
    # 通用参数
    for subparser in [camera_parser, video_parser, benchmark_parser, export_parser]:
        subparser.add_argument('--model-size', default='s', choices=['n', 's', 'm', 'l', 'x'],
                              help='模型大小')
        subparser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                              help='计算设备')
        subparser.add_argument('--confidence', type=float, default=0.25,
                              help='置信度阈值')
        subparser.add_argument('--iou', type=float, default=0.45,
                              help='IoU阈值')
        subparser.add_argument('--fps', type=float, default=30.0,
                              help='目标FPS')
        subparser.add_argument('--platform', default='pc',
                              choices=['pc', 'raspberry_pi', 'jetson_nano', 'esp32'],
                              help='目标平台')
        subparser.add_argument('--adaptive', action='store_true',
                              help='启用自适应推理')
        subparser.add_argument('--edge-opt', action='store_true',
                              help='启用边缘优化')
        subparser.add_argument('--half-precision', action='store_true', default=True,
                              help='启用FP16半精度')
        subparser.add_argument('--tensorrt', action='store_true', default=True,
                              help='启用TensorRT优化')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 设置日志
    logger = LoggingManager().get_logger("YOLOv11Launcher")
    logger.info(f"启动YOLOv11优化系统: {args.command}")
    
    # 执行命令
    try:
        if args.command == 'camera':
            start_camera_detection(args)
        elif args.command == 'video':
            start_video_detection(args)
        elif args.command == 'benchmark':
            benchmark_performance(args)
        elif args.command == 'export':
            export_optimized_model(args)
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        logger.error(f"命令执行失败: {e}")


if __name__ == "__main__":
    main()