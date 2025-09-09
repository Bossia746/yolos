#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOS统一命令行接口
借鉴Ultralytics的CLI设计，提供简洁统一的命令行体验
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import load_config
from src.detection.factory import DetectorFactory
from src.models.optimized_yolov11_system import OptimizedYOLOv11System, OptimizationConfig
from src.utils.logging_manager import LoggingManager


class YOLOSCLIError(Exception):
    """YOLOS CLI异常"""
    pass


class YOLOSCommand:
    """YOLOS命令基类"""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.logger = LoggingManager().get_logger(self.__class__.__name__)
    
    def execute(self):
        """执行命令"""
        raise NotImplementedError
    
    def _create_detector_config(self) -> Dict[str, Any]:
        """创建检测器配置"""
        return {
            'model_size': getattr(self.args, 'model_size', 's'),
            'device': getattr(self.args, 'device', 'auto'),
            'confidence_threshold': getattr(self.args, 'confidence', 0.25),
            'iou_threshold': getattr(self.args, 'iou', 0.45),
            'platform': getattr(self.args, 'platform', 'pc'),
            'adaptive_inference': getattr(self.args, 'adaptive', False),
            'edge_optimization': getattr(self.args, 'edge_opt', False),
            'target_fps': getattr(self.args, 'fps', 30.0)
        }


class DetectCommand(YOLOSCommand):
    """检测命令"""
    
    def execute(self):
        """执行检测"""
        source = self.args.source
        
        if source == 'camera' or source.isdigit():
            self._detect_camera()
        elif source.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            self._detect_video()
        elif source.startswith(('http://', 'https://', 'rtsp://')):
            self._detect_stream()
        else:
            self._detect_image()
    
    def _detect_camera(self):
        """摄像头检测"""
        camera_id = int(self.args.source) if self.args.source.isdigit() else 0
        
        print(f"🎥 启动摄像头检测 (ID: {camera_id})")
        
        # 创建检测器
        config = self._create_detector_config()
        detector = DetectorFactory.create_detector('yolov11', config)
        
        # 医疗模式配置
        if getattr(self.args, 'medical_mode', False):
            self._setup_medical_mode(detector)
        
        try:
            detector.start_camera_detection(camera_id)
        except KeyboardInterrupt:
            print("\n🛑 检测已停止")
        finally:
            detector.stop()
    
    def _detect_video(self):
        """视频文件检测"""
        input_path = self.args.source
        output_path = getattr(self.args, 'output', None)
        
        print(f"🎬 处理视频文件: {input_path}")
        
        # 创建检测器
        config = self._create_detector_config()
        detector = DetectorFactory.create_detector('yolov11', config)
        
        # 处理视频
        detector.process_video(input_path, output_path)
        
        print(f"✅ 视频处理完成")
        if output_path:
            print(f"📁 输出文件: {output_path}")
    
    def _detect_stream(self):
        """网络流检测"""
        stream_url = self.args.source
        
        print(f"📡 连接网络流: {stream_url}")
        
        # 创建检测器
        config = self._create_detector_config()
        detector = DetectorFactory.create_detector('yolov11', config)
        
        try:
            detector.start_stream_detection(stream_url)
        except KeyboardInterrupt:
            print("\n🛑 流检测已停止")
        finally:
            detector.stop()
    
    def _detect_image(self):
        """图像检测"""
        image_path = self.args.source
        
        print(f"🖼️ 处理图像: {image_path}")
        
        # 创建检测器
        config = self._create_detector_config()
        detector = DetectorFactory.create_detector('yolov11', config)
        
        # 处理图像
        results = detector.detect_image(image_path)
        
        # 显示结果
        print(f"✅ 检测完成，发现 {len(results)} 个目标")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result.class_name}: {result.confidence:.3f}")
    
    def _setup_medical_mode(self, detector):
        """设置医疗模式"""
        print("🏥 启用医疗模式")
        
        # 添加医疗监控Hook
        from src.core.hooks import MedicalMonitoringHook
        
        alert_config = {
            'fall_detection': True,
            'medication_tracking': True,
            'vital_monitoring': True,
            'emergency_contact': getattr(self.args, 'emergency_contact', None)
        }
        
        medical_hook = MedicalMonitoringHook(alert_config)
        detector.add_hook(medical_hook)


class TrainCommand(YOLOSCommand):
    """训练命令"""
    
    def execute(self):
        """执行训练"""
        data_config = self.args.data
        
        print(f"🎯 开始训练模型")
        print(f"📊 数据配置: {data_config}")
        
        # 创建训练配置
        train_config = self._create_train_config()
        
        # 创建训练器
        from src.training.trainer import YOLOSTrainer
        trainer = YOLOSTrainer(train_config)
        
        # 自学习模式
        if getattr(self.args, 'self_learning', False):
            self._setup_self_learning(trainer)
        
        # 开始训练
        trainer.train()
        
        print("✅ 训练完成")
    
    def _create_train_config(self) -> Dict[str, Any]:
        """创建训练配置"""
        return {
            'data': self.args.data,
            'model_size': getattr(self.args, 'model_size', 's'),
            'epochs': getattr(self.args, 'epochs', 100),
            'batch_size': getattr(self.args, 'batch_size', 16),
            'learning_rate': getattr(self.args, 'lr', 0.001),
            'device': getattr(self.args, 'device', 'auto'),
            'resume': getattr(self.args, 'resume', None),
            'pretrained': getattr(self.args, 'pretrained', True)
        }
    
    def _setup_self_learning(self, trainer):
        """设置自学习模式"""
        print("🧠 启用自学习模式")
        
        # 配置大模型API
        llm_config = {
            'gpt4v': {
                'enabled': True,
                'api_key': self.args.gpt4v_key if hasattr(self.args, 'gpt4v_key') else None
            },
            'claude3': {
                'enabled': True,
                'api_key': self.args.claude3_key if hasattr(self.args, 'claude3_key') else None
            }
        }
        
        trainer.enable_self_learning(llm_config)


class ExportCommand(YOLOSCommand):
    """导出命令"""
    
    def execute(self):
        """执行导出"""
        model_path = getattr(self.args, 'model', 'yolov11s.pt')
        export_format = getattr(self.args, 'format', 'onnx')
        
        print(f"📦 导出模型: {model_path}")
        print(f"📄 导出格式: {export_format}")
        
        # 创建导出器
        from src.deployment.exporter import ModelExporter
        exporter = ModelExporter()
        
        # 平台优化
        platform = getattr(self.args, 'platform', 'pc')
        if platform != 'pc':
            print(f"🎯 目标平台: {platform}")
        
        # 执行导出
        output_path = exporter.export(
            model_path=model_path,
            format=export_format,
            platform=platform,
            quantize=getattr(self.args, 'quantize', None),
            optimize=getattr(self.args, 'optimize', True)
        )
        
        print(f"✅ 导出完成: {output_path}")


class ServeCommand(YOLOSCommand):
    """服务命令"""
    
    def execute(self):
        """启动服务"""
        host = getattr(self.args, 'host', 'localhost')
        port = getattr(self.args, 'port', 8080)
        
        print(f"🚀 启动YOLOS服务")
        print(f"🌐 地址: http://{host}:{port}")
        
        # 创建服务配置
        service_config = {
            'host': host,
            'port': port,
            'model_config': self._create_detector_config(),
            'cors_enabled': getattr(self.args, 'cors', True),
            'auth_enabled': getattr(self.args, 'auth', False)
        }
        
        # 启动服务
        from src.api.server import YOLOSServer
        server = YOLOSServer(service_config)
        server.start()


class MedicalCommand(YOLOSCommand):
    """医疗专用命令"""
    
    def execute(self):
        """执行医疗功能"""
        medical_function = self.args.function
        
        if medical_function == 'fall-monitor':
            self._fall_monitor()
        elif medical_function == 'medication-check':
            self._medication_check()
        elif medical_function == 'vital-analysis':
            self._vital_analysis()
        elif medical_function == 'emergency-system':
            self._emergency_system()
        else:
            raise YOLOSCLIError(f"未知的医疗功能: {medical_function}")
    
    def _fall_monitor(self):
        """跌倒监控"""
        print("🚨 启动跌倒监控系统")
        
        # 创建专用跌倒检测器
        from src.medical.fall_detector import FallMonitoringSystem
        
        config = {
            'camera_id': getattr(self.args, 'camera', 0),
            'sensitivity': getattr(self.args, 'sensitivity', 0.8),
            'alert_phone': getattr(self.args, 'alert_phone', None),
            'alert_email': getattr(self.args, 'alert_email', None)
        }
        
        monitor = FallMonitoringSystem(config)
        
        try:
            monitor.start_monitoring()
        except KeyboardInterrupt:
            print("\n🛑 监控已停止")
        finally:
            monitor.stop()
    
    def _medication_check(self):
        """药物检查"""
        print("💊 启动药物识别系统")
        
        # 实现药物识别功能
        pass
    
    def _vital_analysis(self):
        """生命体征分析"""
        print("❤️ 启动生命体征分析")
        
        # 实现生命体征分析功能
        pass
    
    def _emergency_system(self):
        """紧急响应系统"""
        print("🚑 启动紧急响应系统")
        
        # 实现紧急响应功能
        pass


class YOLOSCLI:
    """YOLOS统一命令行接口"""
    
    def __init__(self):
        self.parser = self._create_parser()
        self.commands = {
            'detect': DetectCommand,
            'train': TrainCommand,
            'export': ExportCommand,
            'serve': ServeCommand,
            'medical': MedicalCommand
        }
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """创建命令行解析器"""
        parser = argparse.ArgumentParser(
            prog='yolos',
            description='YOLOS - 多模态AI识别系统统一命令行工具',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
示例用法:
  yolos detect camera --model-size s --adaptive --medical-mode
  yolos detect video input.mp4 --output output.mp4 --fall-detection
  yolos train --data medical_dataset.yaml --epochs 100 --self-learning
  yolos export --model yolov11s.pt --format onnx --platform raspberry_pi
  yolos serve --port 8080 --cors --gpu-acceleration
  yolos medical fall-monitor --camera 0 --alert-phone +1234567890
            """
        )
        
        # 全局参数
        parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
        parser.add_argument('--config', '-c', help='配置文件路径')
        
        # 子命令
        subparsers = parser.add_subparsers(dest='command', help='可用命令')
        
        # detect命令
        detect_parser = subparsers.add_parser('detect', help='执行检测任务')
        detect_parser.add_argument('source', help='输入源 (camera/0, video.mp4, image.jpg, rtsp://...)')
        detect_parser.add_argument('--output', '-o', help='输出路径')
        detect_parser.add_argument('--medical-mode', action='store_true', help='启用医疗模式')
        detect_parser.add_argument('--fall-detection', action='store_true', help='启用跌倒检测')
        detect_parser.add_argument('--medication-check', action='store_true', help='启用药物检查')
        detect_parser.add_argument('--emergency-contact', help='紧急联系方式')
        self._add_common_args(detect_parser)
        
        # train命令
        train_parser = subparsers.add_parser('train', help='训练模型')
        train_parser.add_argument('--data', required=True, help='数据配置文件')
        train_parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
        train_parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
        train_parser.add_argument('--lr', type=float, default=0.001, help='学习率')
        train_parser.add_argument('--resume', help='恢复训练检查点')
        train_parser.add_argument('--self-learning', action='store_true', help='启用自学习')
        train_parser.add_argument('--gpt4v-key', help='GPT-4V API密钥')
        train_parser.add_argument('--claude3-key', help='Claude-3 API密钥')
        self._add_common_args(train_parser)
        
        # export命令
        export_parser = subparsers.add_parser('export', help='导出模型')
        export_parser.add_argument('--model', default='yolov11s.pt', help='模型路径')
        export_parser.add_argument('--format', default='onnx', 
                                  choices=['onnx', 'tensorrt', 'tflite', 'coreml', 'openvino'],
                                  help='导出格式')
        export_parser.add_argument('--quantize', choices=['int8', 'fp16'], help='量化类型')
        export_parser.add_argument('--optimize', action='store_true', default=True, help='启用优化')
        self._add_common_args(export_parser)
        
        # serve命令
        serve_parser = subparsers.add_parser('serve', help='启动API服务')
        serve_parser.add_argument('--host', default='localhost', help='服务主机')
        serve_parser.add_argument('--port', type=int, default=8080, help='服务端口')
        serve_parser.add_argument('--cors', action='store_true', help='启用CORS')
        serve_parser.add_argument('--auth', action='store_true', help='启用认证')
        serve_parser.add_argument('--gpu-acceleration', action='store_true', help='GPU加速')
        self._add_common_args(serve_parser)
        
        # medical命令
        medical_parser = subparsers.add_parser('medical', help='医疗专用功能')
        medical_parser.add_argument('function', 
                                   choices=['fall-monitor', 'medication-check', 'vital-analysis', 'emergency-system'],
                                   help='医疗功能')
        medical_parser.add_argument('--camera', type=int, default=0, help='摄像头ID')
        medical_parser.add_argument('--sensitivity', type=float, default=0.8, help='检测灵敏度')
        medical_parser.add_argument('--alert-phone', help='报警电话')
        medical_parser.add_argument('--alert-email', help='报警邮箱')
        self._add_common_args(medical_parser)
        
        return parser
    
    def _add_common_args(self, parser):
        """添加通用参数"""
        parser.add_argument('--model-size', default='s', choices=['n', 's', 'm', 'l', 'x'],
                           help='模型大小')
        parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'],
                           help='计算设备')
        parser.add_argument('--confidence', type=float, default=0.25, help='置信度阈值')
        parser.add_argument('--iou', type=float, default=0.45, help='IoU阈值')
        parser.add_argument('--fps', type=float, default=30.0, help='目标FPS')
        parser.add_argument('--platform', default='pc',
                           choices=['pc', 'raspberry_pi', 'jetson_nano', 'esp32'],
                           help='目标平台')
        parser.add_argument('--adaptive', action='store_true', help='启用自适应推理')
        parser.add_argument('--edge-opt', action='store_true', help='启用边缘优化')
    
    def run(self, args: Optional[List[str]] = None):
        """运行CLI"""
        parsed_args = self.parser.parse_args(args)
        
        if not parsed_args.command:
            self.parser.print_help()
            return
        
        # 设置日志级别
        if parsed_args.verbose:
            import logging
            logging.getLogger().setLevel(logging.DEBUG)
        
        # 加载配置文件
        if parsed_args.config:
            config = load_config(parsed_args.config)
            # 将配置文件参数合并到命令行参数
            for key, value in config.items():
                if not hasattr(parsed_args, key) or getattr(parsed_args, key) is None:
                    setattr(parsed_args, key, value)
        
        # 执行命令
        try:
            command_class = self.commands[parsed_args.command]
            command = command_class(parsed_args)
            command.execute()
        except KeyboardInterrupt:
            print("\n🛑 操作已取消")
        except Exception as e:
            print(f"❌ 执行失败: {e}")
            if parsed_args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)


def main():
    """主入口函数"""
    cli = YOLOSCLI()
    cli.run()


if __name__ == '__main__':
    main()