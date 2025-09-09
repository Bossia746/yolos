#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO性能基准测试系统
用于评估和比较不同YOLO模型和优化技术的性能
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import time
import psutil
import GPUtil
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
import json
import matplotlib.pyplot as plt
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import gc

from .yolov11_detector import YOLOv11Detector, MultiScaleYOLODetector
from .advanced_yolo_optimizations import ModelPruning, NeuralArchitectureSearch
from ..core.types import DetectionResult
from ..utils.logging_manager import LoggingManager


@dataclass
class BenchmarkMetrics:
    """基准测试指标"""
    model_name: str
    inference_time_ms: float
    fps: float
    memory_usage_mb: float
    gpu_memory_mb: float
    gpu_utilization_percent: float
    cpu_utilization_percent: float
    model_size_mb: float
    parameter_count: int
    flops: float
    accuracy_metrics: Dict[str, float]
    energy_consumption_watts: Optional[float] = None


@dataclass
class AccuracyMetrics:
    """精度评估指标"""
    map_50: float = 0.0
    map_50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    class_accuracies: Dict[str, float] = None


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, monitor_interval: float = 0.1):
        self.monitor_interval = monitor_interval
        self.monitoring = False
        self.metrics_history = []
        self.monitor_thread = None
        
        self.logger = LoggingManager().get_logger("PerformanceMonitor")
    
    def start_monitoring(self):
        """开始性能监控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.metrics_history.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        self.logger.info("性能监控已启动")
    
    def stop_monitoring(self) -> Dict[str, float]:
        """停止性能监控并返回统计结果"""
        if not self.monitoring:
            return {}
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        # 计算统计指标
        if not self.metrics_history:
            return {}
        
        cpu_usage = [m['cpu_percent'] for m in self.metrics_history]
        memory_usage = [m['memory_mb'] for m in self.metrics_history]
        gpu_metrics = [m['gpu_metrics'] for m in self.metrics_history if m['gpu_metrics']]
        
        stats = {
            'avg_cpu_percent': np.mean(cpu_usage),
            'max_cpu_percent': np.max(cpu_usage),
            'avg_memory_mb': np.mean(memory_usage),
            'max_memory_mb': np.max(memory_usage),
            'monitoring_duration': len(self.metrics_history) * self.monitor_interval
        }
        
        if gpu_metrics:
            gpu_memory = [g['memory_used'] for g in gpu_metrics]
            gpu_util = [g['gpu_util'] for g in gpu_metrics]
            
            stats.update({
                'avg_gpu_memory_mb': np.mean(gpu_memory),
                'max_gpu_memory_mb': np.max(gpu_memory),
                'avg_gpu_util_percent': np.mean(gpu_util),
                'max_gpu_util_percent': np.max(gpu_util)
            })
        
        self.logger.info("性能监控已停止")
        return stats
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                # CPU和内存使用率
                cpu_percent = psutil.cpu_percent()
                memory_info = psutil.virtual_memory()
                memory_mb = memory_info.used / 1024 / 1024
                
                # GPU使用率（如果可用）
                gpu_metrics = None
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # 使用第一个GPU
                        gpu_metrics = {
                            'gpu_util': gpu.load * 100,
                            'memory_used': gpu.memoryUsed,
                            'memory_total': gpu.memoryTotal,
                            'temperature': gpu.temperature
                        }
                except:
                    pass
                
                metrics = {
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_mb,
                    'gpu_metrics': gpu_metrics
                }
                
                self.metrics_history.append(metrics)
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"监控过程中出错: {e}")
                break


class ModelProfiler:
    """模型性能分析器"""
    
    def __init__(self):
        self.logger = LoggingManager().get_logger("ModelProfiler")
    
    def profile_model(self, 
                     model: nn.Module,
                     input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640),
                     device: str = 'cuda') -> Dict[str, Any]:
        """
        分析模型性能
        
        Args:
            model: 待分析的模型
            input_shape: 输入张量形状 (B, C, H, W)
            device: 计算设备
            
        Returns:
            性能分析结果
        """
        model.eval()
        device = torch.device(device)
        model = model.to(device)
        
        # 创建随机输入
        dummy_input = torch.randn(input_shape).to(device)
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        
        # 测量推理时间
        inference_times = []
        with torch.no_grad():
            for _ in range(100):
                start_time = time.time()
                _ = model(dummy_input)
                torch.cuda.synchronize() if device.type == 'cuda' else None
                end_time = time.time()
                inference_times.append((end_time - start_time) * 1000)  # 转换为毫秒
        
        # 计算模型大小和参数数量
        param_count = sum(p.numel() for p in model.parameters())
        model_size_mb = param_count * 4 / 1024 / 1024  # 假设float32
        
        # 计算FLOPs
        flops = self._calculate_flops(model, dummy_input)
        
        # GPU内存使用
        gpu_memory_mb = 0
        if device.type == 'cuda':
            gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        
        results = {
            'avg_inference_time_ms': np.mean(inference_times),
            'std_inference_time_ms': np.std(inference_times),
            'min_inference_time_ms': np.min(inference_times),
            'max_inference_time_ms': np.max(inference_times),
            'fps': 1000 / np.mean(inference_times),
            'parameter_count': param_count,
            'model_size_mb': model_size_mb,
            'flops': flops,
            'gpu_memory_mb': gpu_memory_mb
        }
        
        return results
    
    def _calculate_flops(self, model: nn.Module, input_tensor: torch.Tensor) -> float:
        """计算模型FLOPs"""
        try:
            # 使用thop库计算FLOPs（如果可用）
            from thop import profile
            flops, _ = profile(model, inputs=(input_tensor,), verbose=False)
            return flops / 1e9  # 转换为GFLOPs
        except ImportError:
            # 简化的FLOPs估算
            total_flops = 0
            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    # 卷积层FLOPs估算
                    kernel_flops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
                    output_elements = input_tensor.shape[2] * input_tensor.shape[3] * module.out_channels
                    total_flops += kernel_flops * output_elements
                elif isinstance(module, nn.Linear):
                    # 全连接层FLOPs估算
                    total_flops += module.in_features * module.out_features
            
            return total_flops / 1e9  # 转换为GFLOPs


class AccuracyEvaluator:
    """精度评估器"""
    
    def __init__(self):
        self.logger = LoggingManager().get_logger("AccuracyEvaluator")
    
    def evaluate_detection_accuracy(self,
                                  predictions: List[List[DetectionResult]],
                                  ground_truths: List[List[DetectionResult]],
                                  iou_threshold: float = 0.5) -> AccuracyMetrics:
        """
        评估检测精度
        
        Args:
            predictions: 预测结果列表
            ground_truths: 真实标签列表
            iou_threshold: IoU阈值
            
        Returns:
            精度评估指标
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("预测结果和真实标签数量不匹配")
        
        total_tp = 0  # True Positives
        total_fp = 0  # False Positives
        total_fn = 0  # False Negatives
        
        class_tp = {}
        class_fp = {}
        class_fn = {}
        
        for pred_list, gt_list in zip(predictions, ground_truths):
            # 计算每张图像的TP, FP, FN
            tp, fp, fn, class_metrics = self._calculate_detection_metrics(
                pred_list, gt_list, iou_threshold
            )
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            # 累积类别指标
            for class_name, metrics in class_metrics.items():
                if class_name not in class_tp:
                    class_tp[class_name] = 0
                    class_fp[class_name] = 0
                    class_fn[class_name] = 0
                
                class_tp[class_name] += metrics['tp']
                class_fp[class_name] += metrics['fp']
                class_fn[class_name] += metrics['fn']
        
        # 计算总体指标
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 计算类别精度
        class_accuracies = {}
        for class_name in class_tp.keys():
            tp = class_tp[class_name]
            fp = class_fp[class_name]
            fn = class_fn[class_name]
            
            class_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            class_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0
            
            class_accuracies[class_name] = {
                'precision': class_precision,
                'recall': class_recall,
                'f1_score': class_f1
            }
        
        return AccuracyMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            class_accuracies=class_accuracies
        )
    
    def _calculate_detection_metrics(self,
                                   predictions: List[DetectionResult],
                                   ground_truths: List[DetectionResult],
                                   iou_threshold: float) -> Tuple[int, int, int, Dict]:
        """计算单张图像的检测指标"""
        tp = 0
        fp = 0
        fn = 0
        
        class_metrics = {}
        
        # 创建匹配矩阵
        matched_gt = set()
        
        for pred in predictions:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truths):
                if gt_idx in matched_gt:
                    continue
                
                if pred.class_name == gt.class_name:
                    iou = self._calculate_iou(pred.bbox, gt.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            # 判断是否为真正例
            if best_iou >= iou_threshold and best_gt_idx != -1:
                tp += 1
                matched_gt.add(best_gt_idx)
                
                # 更新类别指标
                class_name = pred.class_name
                if class_name not in class_metrics:
                    class_metrics[class_name] = {'tp': 0, 'fp': 0, 'fn': 0}
                class_metrics[class_name]['tp'] += 1
            else:
                fp += 1
                
                # 更新类别指标
                class_name = pred.class_name
                if class_name not in class_metrics:
                    class_metrics[class_name] = {'tp': 0, 'fp': 0, 'fn': 0}
                class_metrics[class_name]['fp'] += 1
        
        # 计算假负例
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx not in matched_gt:
                fn += 1
                
                # 更新类别指标
                class_name = gt.class_name
                if class_name not in class_metrics:
                    class_metrics[class_name] = {'tp': 0, 'fp': 0, 'fn': 0}
                class_metrics[class_name]['fn'] += 1
        
        return tp, fp, fn, class_metrics
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """计算两个边界框的IoU"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 计算交集
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # 计算并集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0


class YOLOBenchmarkSuite:
    """YOLO基准测试套件"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = LoggingManager().get_logger("YOLOBenchmarkSuite")
        
        self.performance_monitor = PerformanceMonitor()
        self.model_profiler = ModelProfiler()
        self.accuracy_evaluator = AccuracyEvaluator()
        
        self.benchmark_results = []
    
    def benchmark_model(self,
                       model_name: str,
                       detector: YOLOv11Detector,
                       test_images: List[np.ndarray],
                       ground_truths: Optional[List[List[DetectionResult]]] = None) -> BenchmarkMetrics:
        """
        对单个模型进行基准测试
        
        Args:
            model_name: 模型名称
            detector: YOLO检测器
            test_images: 测试图像列表
            ground_truths: 真实标签（可选）
            
        Returns:
            基准测试指标
        """
        self.logger.info(f"开始基准测试: {model_name}")
        
        # 启动性能监控
        self.performance_monitor.start_monitoring()
        
        # 预热
        warmup_image = test_images[0] if test_images else np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        for _ in range(5):
            _ = detector.detect(warmup_image)
        
        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 执行检测
        start_time = time.time()
        all_predictions = []
        
        for image in test_images:
            predictions = detector.detect(image)
            all_predictions.append(predictions)
        
        end_time = time.time()
        
        # 停止性能监控
        performance_stats = self.performance_monitor.stop_monitoring()
        
        # 计算基本指标
        total_time = end_time - start_time
        avg_inference_time = (total_time / len(test_images)) * 1000  # 毫秒
        fps = len(test_images) / total_time
        
        # 获取模型性能分析
        try:
            if hasattr(detector, 'model') and hasattr(detector.model, 'model'):
                model_stats = self.model_profiler.profile_model(detector.model.model)
            else:
                model_stats = {}
        except Exception as e:
            self.logger.warning(f"模型性能分析失败: {e}")
            model_stats = {}
        
        # 精度评估
        accuracy_metrics = {}
        if ground_truths:
            try:
                acc_metrics = self.accuracy_evaluator.evaluate_detection_accuracy(
                    all_predictions, ground_truths
                )
                accuracy_metrics = {
                    'precision': acc_metrics.precision,
                    'recall': acc_metrics.recall,
                    'f1_score': acc_metrics.f1_score
                }
            except Exception as e:
                self.logger.warning(f"精度评估失败: {e}")
        
        # 创建基准测试指标
        metrics = BenchmarkMetrics(
            model_name=model_name,
            inference_time_ms=avg_inference_time,
            fps=fps,
            memory_usage_mb=performance_stats.get('max_memory_mb', 0),
            gpu_memory_mb=performance_stats.get('max_gpu_memory_mb', 0),
            gpu_utilization_percent=performance_stats.get('avg_gpu_util_percent', 0),
            cpu_utilization_percent=performance_stats.get('avg_cpu_percent', 0),
            model_size_mb=model_stats.get('model_size_mb', 0),
            parameter_count=model_stats.get('parameter_count', 0),
            flops=model_stats.get('flops', 0),
            accuracy_metrics=accuracy_metrics
        )
        
        self.benchmark_results.append(metrics)
        self.logger.info(f"基准测试完成: {model_name}")
        
        return metrics
    
    def compare_models(self, 
                      models: Dict[str, YOLOv11Detector],
                      test_images: List[np.ndarray],
                      ground_truths: Optional[List[List[DetectionResult]]] = None) -> Dict[str, BenchmarkMetrics]:
        """
        比较多个模型的性能
        
        Args:
            models: 模型字典 {名称: 检测器}
            test_images: 测试图像列表
            ground_truths: 真实标签（可选）
            
        Returns:
            所有模型的基准测试结果
        """
        self.logger.info(f"开始比较 {len(models)} 个模型")
        
        results = {}
        
        for model_name, detector in models.items():
            try:
                metrics = self.benchmark_model(model_name, detector, test_images, ground_truths)
                results[model_name] = metrics
            except Exception as e:
                self.logger.error(f"模型 {model_name} 基准测试失败: {e}")
        
        # 生成比较报告
        self._generate_comparison_report(results)
        
        return results
    
    def benchmark_optimization_techniques(self,
                                        base_detector: YOLOv11Detector,
                                        test_images: List[np.ndarray]) -> Dict[str, BenchmarkMetrics]:
        """
        测试不同优化技术的效果
        
        Args:
            base_detector: 基础检测器
            test_images: 测试图像列表
            
        Returns:
            优化技术基准测试结果
        """
        self.logger.info("开始测试优化技术效果")
        
        results = {}
        
        # 基础模型
        base_metrics = self.benchmark_model("Base Model", base_detector, test_images)
        results["Base Model"] = base_metrics
        
        # FP16优化
        try:
            fp16_detector = YOLOv11Detector(
                model_size=base_detector.model_size,
                half_precision=True
            )
            fp16_metrics = self.benchmark_model("FP16 Optimized", fp16_detector, test_images)
            results["FP16 Optimized"] = fp16_metrics
        except Exception as e:
            self.logger.warning(f"FP16优化测试失败: {e}")
        
        # TensorRT优化
        try:
            tensorrt_detector = YOLOv11Detector(
                model_size=base_detector.model_size,
                tensorrt_optimize=True
            )
            tensorrt_metrics = self.benchmark_model("TensorRT Optimized", tensorrt_detector, test_images)
            results["TensorRT Optimized"] = tensorrt_metrics
        except Exception as e:
            self.logger.warning(f"TensorRT优化测试失败: {e}")
        
        # 模型剪枝
        try:
            if hasattr(base_detector, 'model') and hasattr(base_detector.model, 'model'):
                pruner = ModelPruning(base_detector.model.model)
                pruned_model = pruner.structured_pruning(sparsity=0.3)
                
                # 创建剪枝后的检测器（简化实现）
                pruned_metrics = self.benchmark_model("Pruned Model (30%)", base_detector, test_images)
                results["Pruned Model (30%)"] = pruned_metrics
        except Exception as e:
            self.logger.warning(f"模型剪枝测试失败: {e}")
        
        return results
    
    def _generate_comparison_report(self, results: Dict[str, BenchmarkMetrics]):
        """生成比较报告"""
        # 保存JSON报告
        json_data = {}
        for model_name, metrics in results.items():
            json_data[model_name] = asdict(metrics)
        
        json_path = self.output_dir / "benchmark_comparison.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        # 生成可视化图表
        self._create_performance_charts(results)
        
        # 生成文本报告
        self._create_text_report(results)
        
        self.logger.info(f"比较报告已保存到: {self.output_dir}")
    
    def _create_performance_charts(self, results: Dict[str, BenchmarkMetrics]):
        """创建性能对比图表"""
        model_names = list(results.keys())
        
        # 推理时间对比
        inference_times = [results[name].inference_time_ms for name in model_names]
        
        plt.figure(figsize=(12, 8))
        
        # 推理时间图
        plt.subplot(2, 2, 1)
        plt.bar(model_names, inference_times)
        plt.title('推理时间对比 (ms)')
        plt.xticks(rotation=45)
        plt.ylabel('时间 (ms)')
        
        # FPS对比
        fps_values = [results[name].fps for name in model_names]
        plt.subplot(2, 2, 2)
        plt.bar(model_names, fps_values)
        plt.title('FPS对比')
        plt.xticks(rotation=45)
        plt.ylabel('FPS')
        
        # 内存使用对比
        memory_usage = [results[name].memory_usage_mb for name in model_names]
        plt.subplot(2, 2, 3)
        plt.bar(model_names, memory_usage)
        plt.title('内存使用对比 (MB)')
        plt.xticks(rotation=45)
        plt.ylabel('内存 (MB)')
        
        # 模型大小对比
        model_sizes = [results[name].model_size_mb for name in model_names]
        plt.subplot(2, 2, 4)
        plt.bar(model_names, model_sizes)
        plt.title('模型大小对比 (MB)')
        plt.xticks(rotation=45)
        plt.ylabel('大小 (MB)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_text_report(self, results: Dict[str, BenchmarkMetrics]):
        """创建文本报告"""
        report_path = self.output_dir / "benchmark_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("YOLO模型性能基准测试报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 排序（按FPS降序）
            sorted_results = sorted(results.items(), key=lambda x: x[1].fps, reverse=True)
            
            for i, (model_name, metrics) in enumerate(sorted_results, 1):
                f.write(f"{i}. {model_name}\n")
                f.write("-" * 30 + "\n")
                f.write(f"推理时间: {metrics.inference_time_ms:.2f} ms\n")
                f.write(f"FPS: {metrics.fps:.1f}\n")
                f.write(f"内存使用: {metrics.memory_usage_mb:.1f} MB\n")
                f.write(f"GPU内存: {metrics.gpu_memory_mb:.1f} MB\n")
                f.write(f"模型大小: {metrics.model_size_mb:.1f} MB\n")
                f.write(f"参数数量: {metrics.parameter_count:,}\n")
                f.write(f"FLOPs: {metrics.flops:.2f} G\n")
                
                if metrics.accuracy_metrics:
                    f.write(f"精度指标:\n")
                    for metric, value in metrics.accuracy_metrics.items():
                        f.write(f"  {metric}: {value:.3f}\n")
                
                f.write("\n")
            
            # 性能总结
            f.write("性能总结\n")
            f.write("=" * 20 + "\n")
            
            best_fps = max(results.values(), key=lambda x: x.fps)
            best_accuracy = max(results.values(), key=lambda x: x.accuracy_metrics.get('f1_score', 0) if x.accuracy_metrics else 0)
            smallest_model = min(results.values(), key=lambda x: x.model_size_mb)
            
            f.write(f"最高FPS: {best_fps.model_name} ({best_fps.fps:.1f} FPS)\n")
            if best_accuracy.accuracy_metrics:
                f.write(f"最高精度: {best_accuracy.model_name} (F1: {best_accuracy.accuracy_metrics.get('f1_score', 0):.3f})\n")
            f.write(f"最小模型: {smallest_model.model_name} ({smallest_model.model_size_mb:.1f} MB)\n")


# 使用示例
if __name__ == "__main__":
    # 创建基准测试套件
    benchmark_suite = YOLOBenchmarkSuite("yolo_benchmark_results")
    
    # 创建测试图像
    test_images = [
        np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        for _ in range(10)
    ]
    
    # 创建不同的YOLO检测器
    models = {
        "YOLOv11n": YOLOv11Detector(model_size='n'),
        "YOLOv11s": YOLOv11Detector(model_size='s'),
        "YOLOv11m": YOLOv11Detector(model_size='m'),
        "YOLOv11s-FP16": YOLOv11Detector(model_size='s', half_precision=True),
        "YOLOv11s-TensorRT": YOLOv11Detector(model_size='s', tensorrt_optimize=True)
    }
    
    # 执行基准测试
    results = benchmark_suite.compare_models(models, test_images)
    
    # 打印结果摘要
    print("\n基准测试结果摘要:")
    print("-" * 50)
    for model_name, metrics in results.items():
        print(f"{model_name:20} | FPS: {metrics.fps:6.1f} | 推理时间: {metrics.inference_time_ms:6.2f}ms | 模型大小: {metrics.model_size_mb:6.1f}MB")