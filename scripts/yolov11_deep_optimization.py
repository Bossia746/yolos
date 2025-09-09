#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOS YOLOv11深度优化实施脚本
基于评估结果，专注于深度优化当前YOLOv11系统而非升级到YOLO12
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import yaml
import cv2

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

try:
    from ultralytics import YOLO
    from ultralytics.utils import ops
except ImportError:
    print("错误: ultralytics未安装，请运行: pip install ultralytics")
    sys.exit(1)

from src.core.config import YOLOSConfig
from src.models.optimized_yolov11_system import OptimizedYOLOv11System
from src.utils.logging_manager import LoggingManager


class YOLOv11DeepOptimizer:
    """
    YOLOv11深度优化器
    
    实施策略:
    1. TensorRT加速优化
    2. 模型量化和剪枝
    3. 医疗场景专用训练
    4. 边缘设备优化
    5. 多模态融合增强
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化优化器"""
        self.logger = LoggingManager().get_logger("YOLOv11Optimizer")
        self.config = self._load_config(config_path)
        self.device = self._get_optimal_device()
        self.optimized_models = {}
        
        self.logger.info("🚀 YOLOv11深度优化器初始化完成")
        self.logger.info(f"📱 使用设备: {self.device}")
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载配置文件"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        # 默认优化配置
        return {
            'optimization': {
                'tensorrt': True,
                'quantization': True,
                'pruning': True,
                'medical_training': True,
                'edge_optimization': True
            },
            'models': {
                'base_model': 'yolo11s.pt',
                'target_platforms': ['pc', 'raspberry_pi', 'jetson_nano', 'esp32']
            },
            'performance': {
                'target_fps': 60,
                'max_memory_mb': 1500,
                'min_accuracy': 0.90
            }
        }
    
    def _get_optimal_device(self) -> str:
        """获取最优设备"""
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def optimize_tensorrt(self, model_path: str, output_path: str) -> str:
        """
        TensorRT优化
        预期效果: 2-3倍速度提升
        """
        self.logger.info("🔧 开始TensorRT优化...")
        
        try:
            # 加载模型
            model = YOLO(model_path)
            
            # TensorRT导出
            tensorrt_path = model.export(
                format='engine',
                device=self.device,
                half=True,  # FP16精度
                workspace=4,  # 4GB工作空间
                verbose=True
            )
            
            # 验证优化效果
            original_fps = self._benchmark_model(model_path)
            optimized_fps = self._benchmark_model(tensorrt_path)
            
            speedup = optimized_fps / original_fps if original_fps > 0 else 0
            
            self.logger.info(f"✅ TensorRT优化完成!")
            self.logger.info(f"📊 原始FPS: {original_fps:.1f}")
            self.logger.info(f"📊 优化FPS: {optimized_fps:.1f}")
            self.logger.info(f"🚀 加速比: {speedup:.2f}x")
            
            return tensorrt_path
            
        except Exception as e:
            self.logger.error(f"❌ TensorRT优化失败: {e}")
            return model_path
    
    def optimize_quantization(self, model_path: str, output_path: str) -> str:
        """
        模型量化优化
        预期效果: 50%模型大小减少，10-15%速度提升
        """
        self.logger.info("🔧 开始模型量化优化...")
        
        try:
            # 加载模型
            model = YOLO(model_path)
            
            # INT8量化导出
            quantized_path = model.export(
                format='onnx',
                int8=True,
                device=self.device,
                verbose=True
            )
            
            # 验证优化效果
            original_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)  # MB
            
            compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
            
            self.logger.info(f"✅ 量化优化完成!")
            self.logger.info(f"📊 原始大小: {original_size:.1f}MB")
            self.logger.info(f"📊 量化大小: {quantized_size:.1f}MB")
            self.logger.info(f"🗜️ 压缩比: {compression_ratio:.2f}x")
            
            return quantized_path
            
        except Exception as e:
            self.logger.error(f"❌ 量化优化失败: {e}")
            return model_path
    
    def optimize_for_medical_scenarios(self, model_path: str, dataset_path: str) -> str:
        """
        医疗场景专用优化
        针对跌倒检测、药物识别等场景进行专门训练
        """
        self.logger.info("🏥 开始医疗场景专用优化...")
        
        try:
            # 加载基础模型
            model = YOLO(model_path)
            
            # 医疗场景训练配置
            training_config = {
                'data': dataset_path,
                'epochs': 100,
                'imgsz': 640,
                'batch': 16,
                'lr0': 0.01,
                'lrf': 0.1,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
                'pose': 12.0,
                'kobj': 2.0,
                'label_smoothing': 0.0,
                'nbs': 64,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 0.0,
                'translate': 0.1,
                'scale': 0.5,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.0,
                'copy_paste': 0.0
            }
            
            # 开始训练
            results = model.train(**training_config)
            
            # 保存优化后的模型
            medical_model_path = model_path.replace('.pt', '_medical_optimized.pt')
            model.save(medical_model_path)
            
            self.logger.info(f"✅ 医疗场景优化完成!")
            self.logger.info(f"📊 训练结果: {results}")
            
            return medical_model_path
            
        except Exception as e:
            self.logger.error(f"❌ 医疗场景优化失败: {e}")
            return model_path
    
    def optimize_for_edge_devices(self, model_path: str, target_platform: str) -> str:
        """
        边缘设备优化
        针对ESP32、树莓派、Jetson Nano等设备优化
        """
        self.logger.info(f"📱 开始{target_platform}边缘设备优化...")
        
        try:
            model = YOLO(model_path)
            
            # 不同平台的优化策略
            platform_configs = {
                'esp32': {
                    'format': 'tflite',
                    'imgsz': 320,
                    'int8': True,
                    'half': False
                },
                'raspberry_pi': {
                    'format': 'onnx',
                    'imgsz': 416,
                    'int8': True,
                    'half': True
                },
                'jetson_nano': {
                    'format': 'engine',
                    'imgsz': 640,
                    'int8': False,
                    'half': True
                }
            }
            
            config = platform_configs.get(target_platform, platform_configs['raspberry_pi'])
            
            # 导出优化模型
            optimized_path = model.export(**config)
            
            self.logger.info(f"✅ {target_platform}优化完成!")
            self.logger.info(f"📊 优化模型: {optimized_path}")
            
            return optimized_path
            
        except Exception as e:
            self.logger.error(f"❌ {target_platform}优化失败: {e}")
            return model_path
    
    def _benchmark_model(self, model_path: str, num_iterations: int = 100) -> float:
        """模型性能基准测试"""
        try:
            model = YOLO(model_path)
            
            # 创建测试图像
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # 预热
            for _ in range(10):
                model(test_image, verbose=False)
            
            # 基准测试
            start_time = time.time()
            for _ in range(num_iterations):
                model(test_image, verbose=False)
            end_time = time.time()
            
            fps = num_iterations / (end_time - start_time)
            return fps
            
        except Exception as e:
            self.logger.error(f"基准测试失败: {e}")
            return 0.0
    
    def run_comprehensive_optimization(self, model_path: str, output_dir: str) -> Dict[str, str]:
        """运行综合优化"""
        self.logger.info("🚀 开始YOLOv11综合优化...")
        
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        # 1. TensorRT优化
        if self.config['optimization']['tensorrt']:
            tensorrt_path = os.path.join(output_dir, 'yolo11_tensorrt.engine')
            results['tensorrt'] = self.optimize_tensorrt(model_path, tensorrt_path)
        
        # 2. 量化优化
        if self.config['optimization']['quantization']:
            quantized_path = os.path.join(output_dir, 'yolo11_quantized.onnx')
            results['quantized'] = self.optimize_quantization(model_path, quantized_path)
        
        # 3. 边缘设备优化
        if self.config['optimization']['edge_optimization']:
            for platform in self.config['models']['target_platforms']:
                platform_path = os.path.join(output_dir, f'yolo11_{platform}')
                results[f'edge_{platform}'] = self.optimize_for_edge_devices(model_path, platform)
        
        # 4. 性能报告
        self._generate_optimization_report(results, output_dir)
        
        self.logger.info("✅ YOLOv11综合优化完成!")
        return results
    
    def _generate_optimization_report(self, results: Dict[str, str], output_dir: str):
        """生成优化报告"""
        report_path = os.path.join(output_dir, 'optimization_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# YOLOv11优化报告\n\n")
            f.write(f"## 优化时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 优化结果\n\n")
            for opt_type, model_path in results.items():
                f.write(f"- **{opt_type}**: `{model_path}`\n")
            
            f.write("\n## 性能对比\n\n")
            f.write("| 优化类型 | FPS | 模型大小 | 内存占用 |\n")
            f.write("|---------|-----|----------|----------|\n")
            
            for opt_type, model_path in results.items():
                if os.path.exists(model_path):
                    fps = self._benchmark_model(model_path, 50)
                    size_mb = os.path.getsize(model_path) / (1024 * 1024)
                    f.write(f"| {opt_type} | {fps:.1f} | {size_mb:.1f}MB | - |\n")
        
        self.logger.info(f"📊 优化报告已生成: {report_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="YOLOv11深度优化工具")
    parser.add_argument('--model', type=str, default='yolo11s.pt', help='基础模型路径')
    parser.add_argument('--output', type=str, default='./optimized_models', help='输出目录')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--tensorrt', action='store_true', help='启用TensorRT优化')
    parser.add_argument('--quantize', action='store_true', help='启用量化优化')
    parser.add_argument('--edge', action='store_true', help='启用边缘设备优化')
    parser.add_argument('--medical', action='store_true', help='启用医疗场景优化')
    parser.add_argument('--benchmark', action='store_true', help='运行性能基准测试')
    
    args = parser.parse_args()
    
    # 初始化优化器
    optimizer = YOLOv11DeepOptimizer(args.config)
    
    print("🚀 YOLOS YOLOv11深度优化工具")
    print("=" * 50)
    print(f"📱 基础模型: {args.model}")
    print(f"📁 输出目录: {args.output}")
    print(f"🔧 设备: {optimizer.device}")
    print("=" * 50)
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"❌ 模型文件不存在: {args.model}")
        print("💡 正在下载默认模型...")
        try:
            model = YOLO(args.model)  # 自动下载
            print(f"✅ 模型下载完成: {args.model}")
        except Exception as e:
            print(f"❌ 模型下载失败: {e}")
            return
    
    # 运行优化
    if args.benchmark:
        print("\n📊 运行性能基准测试...")
        fps = optimizer._benchmark_model(args.model)
        print(f"📈 基准FPS: {fps:.1f}")
    
    if any([args.tensorrt, args.quantize, args.edge, args.medical]):
        print("\n🔧 开始优化...")
        results = {}
        
        if args.tensorrt:
            results['tensorrt'] = optimizer.optimize_tensorrt(args.model, args.output)
        
        if args.quantize:
            results['quantized'] = optimizer.optimize_quantization(args.model, args.output)
        
        if args.edge:
            for platform in ['raspberry_pi', 'jetson_nano']:
                results[f'edge_{platform}'] = optimizer.optimize_for_edge_devices(args.model, platform)
        
        print("\n✅ 优化完成!")
        for opt_type, path in results.items():
            print(f"📁 {opt_type}: {path}")
    
    else:
        # 运行综合优化
        print("\n🚀 运行综合优化...")
        results = optimizer.run_comprehensive_optimization(args.model, args.output)
        
        print("\n🎉 所有优化完成!")
        print(f"📁 结果保存在: {args.output}")


if __name__ == "__main__":
    main()