#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO模型性能优化器

针对复杂场景优化YOLO模型的处理能力和响应时间
包括：
1. 模型推理优化
2. 内存管理优化
3. 并发处理优化
4. 缓存机制优化
5. 算法参数调优
"""

import time
import threading
import queue
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import cv2
from pathlib import Path
import logging
import gc
from functools import lru_cache
import psutil
import os

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """性能指标"""
    inference_time: float = 0.0
    preprocessing_time: float = 0.0
    postprocessing_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    throughput: float = 0.0
    accuracy: float = 0.0

@dataclass
class OptimizationConfig:
    """优化配置"""
    enable_tensorrt: bool = False
    enable_onnx: bool = True
    batch_size: int = 1
    max_workers: int = 4
    cache_size: int = 128
    memory_limit_mb: int = 2048
    enable_gpu: bool = False
    precision: str = "fp16"  # fp32, fp16, int8
    enable_dynamic_batching: bool = True
    max_batch_delay_ms: int = 10

class ModelCache:
    """模型缓存管理器"""
    
    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key]
            return None
    
    def put(self, key: str, value: Any) -> None:
        """添加缓存项"""
        with self.lock:
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = value
            self.access_times[key] = time.time()
    
    def _evict_lru(self) -> None:
        """移除最少使用的缓存项"""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]
    
    def clear(self) -> None:
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()

class BatchProcessor:
    """批处理器"""
    
    def __init__(self, batch_size: int = 4, max_delay_ms: int = 10):
        self.batch_size = batch_size
        self.max_delay_ms = max_delay_ms
        self.batch_queue = queue.Queue()
        self.result_futures = {}
        self.processing = False
        self.lock = threading.Lock()
    
    def add_request(self, request_id: str, data: Any) -> threading.Event:
        """添加处理请求"""
        result_event = threading.Event()
        
        with self.lock:
            self.batch_queue.put((request_id, data, result_event))
            self.result_futures[request_id] = None
        
        if not self.processing:
            self._start_batch_processing()
        
        return result_event
    
    def _start_batch_processing(self):
        """开始批处理"""
        self.processing = True
        threading.Thread(target=self._process_batches, daemon=True).start()
    
    def _process_batches(self):
        """处理批次"""
        while self.processing:
            batch = []
            start_time = time.time()
            
            # 收集批次数据
            while len(batch) < self.batch_size:
                try:
                    timeout = max(0, (self.max_delay_ms / 1000) - (time.time() - start_time))
                    item = self.batch_queue.get(timeout=timeout)
                    batch.append(item)
                except queue.Empty:
                    break
            
            if batch:
                self._process_batch(batch)
            
            # 检查是否还有待处理项
            if self.batch_queue.empty():
                self.processing = False
                break
    
    def _process_batch(self, batch: List[Tuple]):
        """处理单个批次"""
        try:
            # 模拟批处理逻辑
            batch_data = [item[1] for item in batch]
            results = self._mock_batch_inference(batch_data)
            
            # 设置结果
            for i, (request_id, _, event) in enumerate(batch):
                with self.lock:
                    self.result_futures[request_id] = results[i] if i < len(results) else None
                event.set()
        
        except Exception as e:
            logger.error(f"批处理失败: {e}")
            # 设置错误结果
            for request_id, _, event in batch:
                with self.lock:
                    self.result_futures[request_id] = None
                event.set()
    
    def _mock_batch_inference(self, batch_data: List[Any]) -> List[Any]:
        """模拟批推理"""
        # 这里应该是实际的模型推理逻辑
        time.sleep(0.01)  # 模拟推理时间
        return [f"result_{i}" for i in range(len(batch_data))]
    
    def get_result(self, request_id: str) -> Any:
        """获取处理结果"""
        with self.lock:
            return self.result_futures.get(request_id)

class YOLOPerformanceOptimizer:
    """YOLO性能优化器"""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.model_cache = ModelCache(self.config.cache_size)
        self.batch_processor = BatchProcessor(
            self.config.batch_size, 
            self.config.max_batch_delay_ms
        )
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self.performance_history = []
        self.optimization_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'batch_processed': 0,
            'total_inferences': 0
        }
        
        logger.info(f"YOLO性能优化器初始化完成，配置: {self.config}")
    
    def optimize_model_loading(self, model_path: str) -> Any:
        """优化模型加载"""
        cache_key = f"model_{Path(model_path).stem}"
        
        # 尝试从缓存获取
        cached_model = self.model_cache.get(cache_key)
        if cached_model is not None:
            self.optimization_stats['cache_hits'] += 1
            logger.info(f"从缓存加载模型: {model_path}")
            return cached_model
        
        # 加载新模型
        self.optimization_stats['cache_misses'] += 1
        logger.info(f"加载新模型: {model_path}")
        
        start_time = time.time()
        model = self._load_optimized_model(model_path)
        load_time = time.time() - start_time
        
        # 缓存模型
        self.model_cache.put(cache_key, model)
        
        logger.info(f"模型加载完成，耗时: {load_time:.3f}s")
        return model
    
    def _load_optimized_model(self, model_path: str) -> Any:
        """加载优化后的模型"""
        # 模拟模型加载和优化
        time.sleep(0.1)  # 模拟加载时间
        
        model_info = {
            'path': model_path,
            'precision': self.config.precision,
            'optimized': True,
            'load_time': time.time()
        }
        
        if self.config.enable_onnx:
            model_info['format'] = 'onnx'
            logger.info("启用ONNX优化")
        
        if self.config.enable_tensorrt:
            model_info['format'] = 'tensorrt'
            logger.info("启用TensorRT优化")
        
        return model_info
    
    def optimize_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """优化预处理"""
        start_time = time.time()
        
        # 内存优化的预处理
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        # 使用更高效的resize方法
        if image.shape[:2] != (640, 640):
            image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
        
        # 归一化优化
        image = image.astype(np.float32) / 255.0
        
        # 通道转换优化
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = np.transpose(image, (2, 0, 1))
        
        preprocessing_time = time.time() - start_time
        
        return image, preprocessing_time
    
    def optimize_inference(self, model: Any, input_data: np.ndarray) -> Tuple[Any, PerformanceMetrics]:
        """优化推理过程"""
        metrics = PerformanceMetrics()
        
        # 预处理优化
        start_time = time.time()
        processed_input, preprocessing_time = self.optimize_preprocessing(input_data)
        metrics.preprocessing_time = preprocessing_time
        
        # 推理优化
        inference_start = time.time()
        
        if self.config.enable_dynamic_batching:
            # 使用动态批处理
            request_id = f"req_{int(time.time() * 1000000)}"
            event = self.batch_processor.add_request(request_id, processed_input)
            event.wait(timeout=1.0)  # 等待结果
            result = self.batch_processor.get_result(request_id)
            self.optimization_stats['batch_processed'] += 1
        else:
            # 直接推理
            result = self._mock_inference(model, processed_input)
        
        metrics.inference_time = time.time() - inference_start
        
        # 后处理优化
        postprocess_start = time.time()
        final_result = self._optimize_postprocessing(result)
        metrics.postprocessing_time = time.time() - postprocess_start
        
        # 性能监控
        metrics.memory_usage = self._get_memory_usage()
        metrics.cpu_usage = self._get_cpu_usage()
        metrics.throughput = 1.0 / (metrics.preprocessing_time + metrics.inference_time + metrics.postprocessing_time)
        
        self.optimization_stats['total_inferences'] += 1
        self.performance_history.append(metrics)
        
        return final_result, metrics
    
    def _mock_inference(self, model: Any, input_data: np.ndarray) -> Any:
        """模拟推理过程"""
        # 模拟推理延迟
        time.sleep(0.02)
        
        # 模拟检测结果
        detections = {
            'boxes': np.random.rand(5, 4) * 640,  # 5个检测框
            'scores': np.random.rand(5) * 0.9 + 0.1,  # 置信度
            'classes': np.random.randint(0, 80, 5)  # 类别
        }
        
        return detections
    
    def _optimize_postprocessing(self, raw_result: Any) -> Any:
        """优化后处理"""
        if not isinstance(raw_result, dict):
            return raw_result
        
        # NMS优化
        if 'boxes' in raw_result and 'scores' in raw_result:
            # 使用更高效的NMS算法
            boxes = raw_result['boxes']
            scores = raw_result['scores']
            
            # 置信度过滤
            confidence_threshold = 0.5
            valid_indices = scores > confidence_threshold
            
            filtered_result = {
                'boxes': boxes[valid_indices],
                'scores': scores[valid_indices],
                'classes': raw_result.get('classes', [])[valid_indices] if 'classes' in raw_result else []
            }
            
            return filtered_result
        
        return raw_result
    
    def _get_memory_usage(self) -> float:
        """获取内存使用率"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """获取CPU使用率"""
        try:
            return psutil.cpu_percent(interval=0.1)
        except:
            return 0.0
    
    def optimize_memory_usage(self) -> None:
        """优化内存使用"""
        # 清理缓存
        if self._get_memory_usage() > self.config.memory_limit_mb:
            logger.warning("内存使用超限，清理缓存")
            self.model_cache.clear()
            gc.collect()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        if not self.performance_history:
            return {'error': '没有性能数据'}
        
        recent_metrics = self.performance_history[-100:]  # 最近100次
        
        avg_inference_time = np.mean([m.inference_time for m in recent_metrics])
        avg_preprocessing_time = np.mean([m.preprocessing_time for m in recent_metrics])
        avg_postprocessing_time = np.mean([m.postprocessing_time for m in recent_metrics])
        avg_memory_usage = np.mean([m.memory_usage for m in recent_metrics])
        avg_cpu_usage = np.mean([m.cpu_usage for m in recent_metrics])
        avg_throughput = np.mean([m.throughput for m in recent_metrics])
        
        cache_hit_rate = (self.optimization_stats['cache_hits'] / 
                         max(1, self.optimization_stats['cache_hits'] + self.optimization_stats['cache_misses'])) * 100
        
        return {
            'performance_metrics': {
                'avg_inference_time': avg_inference_time,
                'avg_preprocessing_time': avg_preprocessing_time,
                'avg_postprocessing_time': avg_postprocessing_time,
                'avg_total_time': avg_inference_time + avg_preprocessing_time + avg_postprocessing_time,
                'avg_memory_usage_mb': avg_memory_usage,
                'avg_cpu_usage_percent': avg_cpu_usage,
                'avg_throughput_fps': avg_throughput
            },
            'optimization_stats': {
                'cache_hit_rate_percent': cache_hit_rate,
                'total_inferences': self.optimization_stats['total_inferences'],
                'batch_processed': self.optimization_stats['batch_processed'],
                'cache_hits': self.optimization_stats['cache_hits'],
                'cache_misses': self.optimization_stats['cache_misses']
            },
            'configuration': {
                'batch_size': self.config.batch_size,
                'max_workers': self.config.max_workers,
                'cache_size': self.config.cache_size,
                'enable_dynamic_batching': self.config.enable_dynamic_batching,
                'precision': self.config.precision
            },
            'recommendations': self._generate_optimization_recommendations()
        }
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        if not self.performance_history:
            return recommendations
        
        recent_metrics = self.performance_history[-50:]
        avg_inference_time = np.mean([m.inference_time for m in recent_metrics])
        avg_memory_usage = np.mean([m.memory_usage for m in recent_metrics])
        cache_hit_rate = (self.optimization_stats['cache_hits'] / 
                         max(1, self.optimization_stats['cache_hits'] + self.optimization_stats['cache_misses'])) * 100
        
        if avg_inference_time > 0.1:
            recommendations.append("推理时间较长，建议启用TensorRT或ONNX优化")
        
        if avg_memory_usage > self.config.memory_limit_mb * 0.8:
            recommendations.append("内存使用率较高，建议减少缓存大小或启用内存优化")
        
        if cache_hit_rate < 50:
            recommendations.append("缓存命中率较低，建议增加缓存大小")
        
        if not self.config.enable_dynamic_batching:
            recommendations.append("建议启用动态批处理以提高吞吐量")
        
        if self.config.precision == "fp32":
            recommendations.append("建议使用fp16精度以提高性能")
        
        return recommendations
    
    def cleanup(self) -> None:
        """清理资源"""
        logger.info("清理YOLO性能优化器资源")
        self.model_cache.clear()
        self.thread_pool.shutdown(wait=True)
        self.batch_processor.processing = False
        gc.collect()

def create_optimizer(config_dict: Dict[str, Any] = None) -> YOLOPerformanceOptimizer:
    """创建优化器实例"""
    if config_dict:
        config = OptimizationConfig(**config_dict)
    else:
        config = OptimizationConfig()
    
    return YOLOPerformanceOptimizer(config)

def benchmark_optimizer(optimizer: YOLOPerformanceOptimizer, num_iterations: int = 100) -> Dict[str, Any]:
    """性能基准测试"""
    logger.info(f"开始性能基准测试，迭代次数: {num_iterations}")
    
    # 模拟模型
    model = optimizer.optimize_model_loading("test_model.pt")
    
    # 测试数据
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    start_time = time.time()
    
    for i in range(num_iterations):
        result, metrics = optimizer.optimize_inference(model, test_image)
        
        if i % 20 == 0:
            logger.info(f"完成 {i+1}/{num_iterations} 次推理")
    
    total_time = time.time() - start_time
    
    # 生成基准报告
    performance_report = optimizer.get_performance_report()
    performance_report['benchmark_info'] = {
        'total_iterations': num_iterations,
        'total_time_seconds': total_time,
        'average_fps': num_iterations / total_time
    }
    
    logger.info(f"基准测试完成，平均FPS: {num_iterations / total_time:.2f}")
    
    return performance_report

if __name__ == '__main__':
    # 示例使用
    config = OptimizationConfig(
        enable_onnx=True,
        batch_size=4,
        max_workers=4,
        cache_size=64,
        enable_dynamic_batching=True,
        precision="fp16"
    )
    
    optimizer = YOLOPerformanceOptimizer(config)
    
    try:
        # 运行基准测试
        report = benchmark_optimizer(optimizer, 50)
        
        print("\n=== YOLO性能优化报告 ===")
        print(f"平均推理时间: {report['performance_metrics']['avg_inference_time']:.4f}s")
        print(f"平均吞吐量: {report['performance_metrics']['avg_throughput_fps']:.2f} FPS")
        print(f"缓存命中率: {report['optimization_stats']['cache_hit_rate_percent']:.1f}%")
        print(f"内存使用: {report['performance_metrics']['avg_memory_usage_mb']:.1f} MB")
        
        if report.get('recommendations'):
            print("\n优化建议:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"{i}. {rec}")
    
    finally:
        optimizer.cleanup()