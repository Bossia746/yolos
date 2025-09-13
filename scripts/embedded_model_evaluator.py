#!/usr/bin/env python3
"""
嵌入式设备模型性能评估器
评估YOLO模型在资源受限设备上的性能表现
"""

import os
import sys
import time
import psutil
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

try:
    import torch
    import torchvision
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

@dataclass
class PlatformSpec:
    """平台规格"""
    name: str
    memory_mb: int
    storage_mb: int
    cpu_cores: int
    cpu_freq_mhz: int
    has_gpu: bool = False
    has_npu: bool = False
    power_limit_mw: int = 5000

@dataclass
class ModelMetrics:
    """模型性能指标"""
    model_size_mb: float
    memory_usage_mb: float
    inference_time_ms: float
    fps: float
    accuracy: float = 0.0
    power_consumption_mw: float = 0.0
    is_feasible: bool = True
    bottlenecks: List[str] = None

class EmbeddedModelEvaluator:
    """嵌入式模型评估器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # 定义目标平台
        self.platforms = {
            'esp32': PlatformSpec(
                name='ESP32',
                memory_mb=32,  # 512KB SRAM + 8MB PSRAM
                storage_mb=16,
                cpu_cores=2,
                cpu_freq_mhz=240,
                power_limit_mw=500
            ),
            'esp32_s3': PlatformSpec(
                name='ESP32-S3',
                memory_mb=64,  # 512KB SRAM + 32MB PSRAM
                storage_mb=32,
                cpu_cores=2,
                cpu_freq_mhz=240,
                power_limit_mw=600
            ),
            'raspberry_pi_zero': PlatformSpec(
                name='Raspberry Pi Zero 2W',
                memory_mb=512,
                storage_mb=8192,  # 8GB SD卡
                cpu_cores=4,
                cpu_freq_mhz=1000,
                power_limit_mw=2000
            ),
            'raspberry_pi_4': PlatformSpec(
                name='Raspberry Pi 4B',
                memory_mb=4096,
                storage_mb=32768,  # 32GB SD卡
                cpu_cores=4,
                cpu_freq_mhz=1500,
                has_gpu=True,
                power_limit_mw=3000
            ),
            'jetson_nano': PlatformSpec(
                name='NVIDIA Jetson Nano',
                memory_mb=4096,
                storage_mb=16384,
                cpu_cores=4,
                cpu_freq_mhz=1430,
                has_gpu=True,
                power_limit_mw=5000
            ),
            'k230': PlatformSpec(
                name='Canaan K230',
                memory_mb=512,
                storage_mb=8192,
                cpu_cores=2,
                cpu_freq_mhz=1600,
                has_npu=True,
                power_limit_mw=2000
            )
        }
        
        # 模型配置
        self.model_configs = {
            'yolov11n': {'size': 'n', 'expected_size_mb': 5.9},
            'yolov11s': {'size': 's', 'expected_size_mb': 21.5},
            'yolov11m': {'size': 'm', 'expected_size_mb': 49.7},
            'yolov11l': {'size': 'l', 'expected_size_mb': 86.9},
            'yolov11x': {'size': 'x', 'expected_size_mb': 137.3}
        }
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def evaluate_all_combinations(self) -> Dict:
        """评估所有平台和模型组合"""
        results = {}
        
        for platform_name, platform_spec in self.platforms.items():
            results[platform_name] = {}
            
            for model_name, model_config in self.model_configs.items():
                self.logger.info(f"评估 {model_name} 在 {platform_name} 上的性能")
                
                metrics = self.evaluate_model_on_platform(
                    model_name, model_config, platform_spec
                )
                
                results[platform_name][model_name] = metrics
                
        return results
        
    def evaluate_model_on_platform(self, model_name: str, 
                                  model_config: Dict, 
                                  platform: PlatformSpec) -> ModelMetrics:
        """评估特定模型在特定平台上的性能"""
        
        # 估算模型大小
        model_size_mb = model_config['expected_size_mb']
        
        # 估算内存使用（模型大小 + 推理缓冲区）
        inference_buffer_mb = self._estimate_inference_buffer(model_config['size'])
        total_memory_mb = model_size_mb + inference_buffer_mb
        
        # 估算推理时间
        inference_time_ms = self._estimate_inference_time(
            model_config['size'], platform
        )
        
        # 计算FPS
        fps = 1000.0 / inference_time_ms if inference_time_ms > 0 else 0
        
        # 可行性分析
        bottlenecks = []
        is_feasible = True
        
        # 内存检查
        if total_memory_mb > platform.memory_mb:
            bottlenecks.append(f"内存不足: 需要{total_memory_mb}MB, 可用{platform.memory_mb}MB")
            is_feasible = False
            
        # 存储检查
        if model_size_mb > platform.storage_mb * 0.5:  # 模型不应占用超过50%存储
            bottlenecks.append(f"存储不足: 模型{model_size_mb}MB, 可用{platform.storage_mb}MB")
            is_feasible = False
            
        # 性能检查
        if fps < 1.0:  # 至少1FPS
            bottlenecks.append(f"性能不足: FPS={fps:.2f}")
            is_feasible = False
            
        # 功耗检查
        power_consumption = self._estimate_power_consumption(
            model_config['size'], platform
        )
        if power_consumption > platform.power_limit_mw:
            bottlenecks.append(f"功耗过高: {power_consumption}mW > {platform.power_limit_mw}mW")
            is_feasible = False
            
        return ModelMetrics(
            model_size_mb=model_size_mb,
            memory_usage_mb=total_memory_mb,
            inference_time_ms=inference_time_ms,
            fps=fps,
            power_consumption_mw=power_consumption,
            is_feasible=is_feasible,
            bottlenecks=bottlenecks or []
        )
        
    def _estimate_inference_buffer(self, model_size: str) -> float:
        """估算推理缓冲区大小"""
        buffer_sizes = {
            'n': 20,   # 20MB
            's': 40,   # 40MB
            'm': 80,   # 80MB
            'l': 120,  # 120MB
            'x': 200   # 200MB
        }
        return buffer_sizes.get(model_size, 50)
        
    def _estimate_inference_time(self, model_size: str, platform: PlatformSpec) -> float:
        """估算推理时间"""
        # 基础推理时间（毫秒）
        base_times = {
            'n': 50,
            's': 100,
            'm': 200,
            'l': 400,
            'x': 800
        }
        
        base_time = base_times.get(model_size, 200)
        
        # 根据平台调整
        if platform.name == 'ESP32':
            # ESP32性能很低，但通常不会运行完整YOLO
            return base_time * 50  # 假设性能差50倍
        elif 'Raspberry Pi Zero' in platform.name:
            return base_time * 10  # 性能差10倍
        elif 'Raspberry Pi 4' in platform.name:
            if platform.has_gpu:
                return base_time * 2  # GPU加速
            else:
                return base_time * 5  # CPU推理
        elif 'Jetson' in platform.name:
            return base_time * 0.5  # GPU加速，性能好
        elif 'K230' in platform.name:
            if platform.has_npu:
                return base_time * 0.3  # NPU加速
            else:
                return base_time * 3
        else:
            return base_time
            
    def _estimate_power_consumption(self, model_size: str, platform: PlatformSpec) -> float:
        """估算功耗"""
        # 基础功耗（毫瓦）
        base_power = {
            'n': 200,
            's': 400,
            'm': 800,
            'l': 1200,
            'x': 2000
        }
        
        power = base_power.get(model_size, 500)
        
        # 平台调整
        if platform.name == 'ESP32':
            return min(power * 0.1, platform.power_limit_mw)  # 低功耗
        elif 'Raspberry Pi' in platform.name:
            return power * 0.8
        elif 'Jetson' in platform.name:
            return power * 1.5  # GPU功耗高
        else:
            return power
            
    def generate_optimization_recommendations(self, results: Dict) -> Dict:
        """生成优化建议"""
        recommendations = {}
        
        for platform_name, platform_results in results.items():
            platform_spec = self.platforms[platform_name]
            recommendations[platform_name] = {
                'feasible_models': [],
                'optimization_strategies': [],
                'alternative_approaches': []
            }
            
            # 找出可行的模型
            for model_name, metrics in platform_results.items():
                if metrics.is_feasible:
                    recommendations[platform_name]['feasible_models'].append({
                        'model': model_name,
                        'fps': metrics.fps,
                        'memory_mb': metrics.memory_usage_mb
                    })
                    
            # 生成优化策略
            if platform_spec.memory_mb < 100:  # 极低内存设备
                recommendations[platform_name]['optimization_strategies'].extend([
                    "使用INT8量化减少内存占用",
                    "实现模型分割，分阶段推理",
                    "使用专用的轻量级检测算法",
                    "考虑边缘-云协同架构"
                ])
                recommendations[platform_name]['alternative_approaches'].extend([
                    "使用TensorFlow Lite Micro",
                    "使用OpenMV或类似的计算机视觉库",
                    "实现简单的特征检测算法"
                ])
            elif platform_spec.memory_mb < 1000:  # 中等内存设备
                recommendations[platform_name]['optimization_strategies'].extend([
                    "使用ONNX Runtime优化推理",
                    "启用FP16精度",
                    "使用模型剪枝技术",
                    "优化输入分辨率"
                ])
            else:  # 高内存设备
                recommendations[platform_name]['optimization_strategies'].extend([
                    "使用TensorRT加速（如果有GPU）",
                    "启用批处理推理",
                    "使用更大的模型获得更好精度"
                ])
                
        return recommendations
        
    def save_results(self, results: Dict, recommendations: Dict, output_file: str):
        """保存评估结果"""
        # 转换为可序列化的格式
        serializable_results = {}
        for platform, models in results.items():
            serializable_results[platform] = {}
            for model, metrics in models.items():
                serializable_results[platform][model] = {
                    'model_size_mb': metrics.model_size_mb,
                    'memory_usage_mb': metrics.memory_usage_mb,
                    'inference_time_ms': metrics.inference_time_ms,
                    'fps': metrics.fps,
                    'power_consumption_mw': metrics.power_consumption_mw,
                    'is_feasible': metrics.is_feasible,
                    'bottlenecks': metrics.bottlenecks
                }
                
        output_data = {
            'evaluation_results': serializable_results,
            'optimization_recommendations': recommendations,
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'platform_specifications': {
                name: {
                    'memory_mb': spec.memory_mb,
                    'storage_mb': spec.storage_mb,
                    'cpu_cores': spec.cpu_cores,
                    'cpu_freq_mhz': spec.cpu_freq_mhz,
                    'has_gpu': spec.has_gpu,
                    'has_npu': spec.has_npu,
                    'power_limit_mw': spec.power_limit_mw
                } for name, spec in self.platforms.items()
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"评估结果已保存到: {output_file}")
        
    def print_summary(self, results: Dict, recommendations: Dict):
        """打印评估摘要"""
        print("\n" + "="*80)
        print("嵌入式设备模型性能评估摘要")
        print("="*80)
        
        for platform_name, platform_results in results.items():
            platform_spec = self.platforms[platform_name]
            print(f"\n📱 {platform_spec.name}")
            print(f"   内存: {platform_spec.memory_mb}MB | 存储: {platform_spec.storage_mb}MB")
            print(f"   CPU: {platform_spec.cpu_cores}核@{platform_spec.cpu_freq_mhz}MHz")
            
            feasible_count = sum(1 for metrics in platform_results.values() if metrics.is_feasible)
            total_count = len(platform_results)
            
            print(f"   ✅ 可行模型: {feasible_count}/{total_count}")
            
            # 显示最佳模型
            best_model = None
            best_fps = 0
            for model_name, metrics in platform_results.items():
                if metrics.is_feasible and metrics.fps > best_fps:
                    best_model = model_name
                    best_fps = metrics.fps
                    
            if best_model:
                metrics = platform_results[best_model]
                print(f"   🏆 推荐模型: {best_model}")
                print(f"      FPS: {metrics.fps:.1f} | 内存: {metrics.memory_usage_mb:.1f}MB")
            else:
                print(f"   ❌ 无可行模型")
                
        print("\n" + "="*80)
        
if __name__ == "__main__":
    evaluator = EmbeddedModelEvaluator()
    
    print("开始嵌入式设备模型性能评估...")
    
    # 执行评估
    results = evaluator.evaluate_all_combinations()
    
    # 生成建议
    recommendations = evaluator.generate_optimization_recommendations(results)
    
    # 保存结果
    output_file = "embedded_model_evaluation_results.json"
    evaluator.save_results(results, recommendations, output_file)
    
    # 打印摘要
    evaluator.print_summary(results, recommendations)
    
    print(f"\n详细结果已保存到: {output_file}")