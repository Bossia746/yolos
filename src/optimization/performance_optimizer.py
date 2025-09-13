#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能优化模块
提供模型量化、推理加速、GPU/NPU并行计算等性能优化功能
针对K230和ESP32等边缘设备进行深度优化
"""

import time
import threading
import multiprocessing
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import gc

# 可选依赖
try:
    import cv2
except ImportError:
    cv2 = None

try:
    import torch
    import torch.nn as nn
    import torch.quantization as quant
except ImportError:
    torch = None
    nn = None
    quant = None

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    onnx = None
    ort = None

try:
    import tensorrt as trt
except ImportError:
    trt = None

# 日志记录器
logger = logging.getLogger(__name__)

class ComplexPathProcessor:
    """复杂路径处理器 - 专门处理复杂场景下的路径优化"""
    
    def __init__(self, max_path_length: int = 1000, cache_size: int = 500):
        self.max_path_length = max_path_length
        self.cache_size = cache_size
        self.path_cache = {}
        self.complexity_threshold = 0.7
        self.processing_stats = {
            'total_processed': 0,
            'complex_paths': 0,
            'cache_hits': 0,
            'processing_time': 0.0
        }
        
    def process_complex_paths(self, paths: List[str], parallel: bool = True) -> List[Dict[str, Any]]:
        """处理复杂路径列表"""
        start_time = time.time()
        results = []
        
        if parallel and len(paths) > 10:
            # 并行处理大量路径
            with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), 8)) as executor:
                futures = [executor.submit(self._process_single_path, path) for path in paths]
                
                for future in futures:
                    try:
                        result = future.result(timeout=30)
                        results.append(result)
                    except Exception as e:
                        logger.warning(f"路径处理失败: {e}")
                        results.append({'error': str(e), 'processed': False})
        else:
            # 串行处理
            for path in paths:
                result = self._process_single_path(path)
                results.append(result)
        
        # 更新统计信息
        processing_time = time.time() - start_time
        self.processing_stats['total_processed'] += len(paths)
        self.processing_stats['processing_time'] += processing_time
        
        success_count = sum(1 for r in results if r.get('processed', False))
        logger.info(f"路径处理完成: {success_count}/{len(paths)} 成功, 耗时: {processing_time:.2f}秒")
        
        return results
    
    def _process_single_path(self, path: str) -> Dict[str, Any]:
        """处理单个路径"""
        # 检查缓存
        path_hash = hash(path)
        if path_hash in self.path_cache:
            self.processing_stats['cache_hits'] += 1
            return self.path_cache[path_hash]
        
        try:
            # 计算路径复杂度
            complexity = self._calculate_path_complexity(path)
            
            # 根据复杂度选择处理策略
            if complexity > self.complexity_threshold:
                processed_path = self._handle_complex_path(path)
                self.processing_stats['complex_paths'] += 1
            else:
                processed_path = self._handle_simple_path(path)
            
            result = {
                'original_path': path,
                'processed_path': processed_path,
                'complexity_score': complexity,
                'processed': True,
                'processing_method': 'complex' if complexity > self.complexity_threshold else 'simple'
            }
            
            # 缓存结果
            if len(self.path_cache) < self.cache_size:
                self.path_cache[path_hash] = result
            
            return result
            
        except Exception as e:
            logger.error(f"路径处理异常: {path} - {e}")
            return {
                'original_path': path,
                'processed_path': path,
                'complexity_score': 0.0,
                'processed': False,
                'error': str(e)
            }
    
    def _calculate_path_complexity(self, path: str) -> float:
        """计算路径复杂度"""
        if not path:
            return 0.0
        
        factors = {
            'length': min(len(path) / self.max_path_length, 1.0),
            'depth': min(path.count('/') + path.count('\\'), 10) / 10,
            'special_chars': sum(1 for c in path if not c.isalnum() and c not in '/\\.-_') / len(path),
            'unicode_chars': sum(1 for c in path if ord(c) > 127) / len(path),
            'repeated_patterns': self._detect_repeated_patterns(path)
        }
        
        # 加权计算复杂度
        weights = {
            'length': 0.25,
            'depth': 0.20,
            'special_chars': 0.20,
            'unicode_chars': 0.15,
            'repeated_patterns': 0.20
        }
        
        complexity = sum(factors[k] * weights[k] for k in weights)
        return min(complexity, 1.0)
    
    def _detect_repeated_patterns(self, path: str) -> float:
        """检测重复模式"""
        if len(path) < 4:
            return 0.0
        
        pattern_count = 0
        for i in range(len(path) - 3):
            pattern = path[i:i+4]
            if path.count(pattern) > 1:
                pattern_count += 1
        
        return min(pattern_count / len(path), 1.0)
    
    def _handle_simple_path(self, path: str) -> str:
        """处理简单路径"""
        # 标准化路径分隔符
        normalized = path.replace('\\', '/')
        
        # 移除重复分隔符
        while '//' in normalized:
            normalized = normalized.replace('//', '/')
        
        return normalized.strip('/')
    
    def _handle_complex_path(self, path: str) -> str:
        """处理复杂路径"""
        # 分段处理复杂路径
        segments = path.replace('\\', '/').split('/')
        processed_segments = []
        
        for segment in segments:
            if not segment:
                continue
                
            # 处理长段落
            if len(segment) > 100:
                # 截断过长的段落
                processed_segment = segment[:97] + '...'
            else:
                processed_segment = segment
            
            # 处理特殊字符
            processed_segment = self._sanitize_segment(processed_segment)
            processed_segments.append(processed_segment)
        
        return '/'.join(processed_segments)
    
    def _sanitize_segment(self, segment: str) -> str:
        """清理路径段"""
        # 处理Unicode字符
        try:
            # 尝试编码/解码以处理特殊字符
            sanitized = segment.encode('utf-8', errors='replace').decode('utf-8')
        except Exception:
            # 如果失败，移除非ASCII字符
            sanitized = ''.join(c if ord(c) < 128 else '_' for c in segment)
        
        return sanitized
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        stats = self.processing_stats.copy()
        
        if stats['total_processed'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_processed']
            stats['complex_path_rate'] = stats['complex_paths'] / stats['total_processed']
            stats['avg_processing_time'] = stats['processing_time'] / stats['total_processed']
        else:
            stats['cache_hit_rate'] = 0.0
            stats['complex_path_rate'] = 0.0
            stats['avg_processing_time'] = 0.0
        
        return stats
    
    def clear_cache(self):
        """清空缓存"""
        self.path_cache.clear()
        logger.info("路径处理缓存已清空")

class OptimizationLevel(Enum):
    """优化级别"""
    NONE = "none"  # 无优化
    BASIC = "basic"  # 基础优化
    AGGRESSIVE = "aggressive"  # 激进优化
    ULTRA = "ultra"  # 极限优化

class QuantizationType(Enum):
    """量化类型"""
    DYNAMIC = "dynamic"  # 动态量化
    STATIC = "static"  # 静态量化
    QAT = "qat"  # 量化感知训练
    INT8 = "int8"  # INT8量化
    FP16 = "fp16"  # FP16量化

class AccelerationType(Enum):
    """加速类型"""
    CPU = "cpu"  # CPU加速
    GPU = "gpu"  # GPU加速
    NPU = "npu"  # NPU加速
    TENSORRT = "tensorrt"  # TensorRT加速
    OPENVINO = "openvino"  # OpenVINO加速
    ONNX = "onnx"  # ONNX Runtime加速

@dataclass
class PerformanceMetrics:
    """性能指标"""
    inference_time: float  # 推理时间（毫秒）
    fps: float  # 帧率
    memory_usage: float  # 内存使用（MB）
    cpu_usage: float  # CPU使用率（%）
    gpu_usage: float = 0.0  # GPU使用率（%）
    power_consumption: float = 0.0  # 功耗（瓦特）
    model_size: float = 0.0  # 模型大小（MB）
    accuracy: float = 0.0  # 准确率
    latency: float = 0.0  # 延迟（毫秒）
    throughput: float = 0.0  # 吞吐量（samples/s）
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            "inference_time": self.inference_time,
            "fps": self.fps,
            "memory_usage": self.memory_usage,
            "cpu_usage": self.cpu_usage,
            "gpu_usage": self.gpu_usage,
            "power_consumption": self.power_consumption,
            "model_size": self.model_size,
            "accuracy": self.accuracy,
            "latency": self.latency,
            "throughput": self.throughput
        }

@dataclass
class OptimizationConfig:
    """优化配置"""
    optimization_level: OptimizationLevel = OptimizationLevel.BASIC
    quantization_type: QuantizationType = QuantizationType.DYNAMIC
    acceleration_type: AccelerationType = AccelerationType.CPU
    target_fps: float = 30.0
    max_memory_mb: float = 512.0
    enable_parallel: bool = True
    num_threads: int = 4
    batch_size: int = 1
    enable_caching: bool = True
    cache_size: int = 100
    enable_profiling: bool = False
    
class ModelQuantizer:
    """模型量化器
    
    支持多种量化方法和框架
    """
    
    def __init__(self, quantization_type: QuantizationType = QuantizationType.DYNAMIC):
        self.quantization_type = quantization_type
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def quantize_pytorch_model(self, model, calibration_data: Optional[List] = None) -> Any:
        """量化PyTorch模型
        
        Args:
            model: PyTorch模型
            calibration_data: 校准数据
            
        Returns:
            Any: 量化后的模型
        """
        if torch is None:
            raise ImportError("PyTorch未安装")
        
        try:
            if self.quantization_type == QuantizationType.DYNAMIC:
                # 动态量化
                quantized_model = torch.quantization.quantize_dynamic(
                    model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
                )
                
            elif self.quantization_type == QuantizationType.STATIC:
                # 静态量化
                model.eval()
                model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                torch.quantization.prepare(model, inplace=True)
                
                # 校准
                if calibration_data:
                    with torch.no_grad():
                        for data in calibration_data[:100]:  # 限制校准数据量
                            model(data)
                
                quantized_model = torch.quantization.convert(model, inplace=False)
                
            elif self.quantization_type == QuantizationType.FP16:
                # FP16量化
                quantized_model = model.half()
                
            else:
                self.logger.warning(f"不支持的量化类型: {self.quantization_type}")
                return model
            
            self.logger.info(f"PyTorch模型量化完成: {self.quantization_type.value}")
            return quantized_model
            
        except Exception as e:
            self.logger.error(f"PyTorch模型量化失败: {e}")
            return model
    
    def quantize_onnx_model(self, model_path: str, output_path: str, calibration_data: Optional[List] = None) -> str:
        """量化ONNX模型
        
        Args:
            model_path: 输入模型路径
            output_path: 输出模型路径
            calibration_data: 校准数据
            
        Returns:
            str: 量化后模型路径
        """
        if onnx is None or ort is None:
            raise ImportError("ONNX或ONNXRuntime未安装")
        
        try:
            from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType
            
            if self.quantization_type == QuantizationType.DYNAMIC:
                # 动态量化
                quantize_dynamic(
                    model_path,
                    output_path,
                    weight_type=QuantType.QUInt8
                )
                
            elif self.quantization_type == QuantizationType.STATIC:
                # 静态量化（需要校准数据）
                if calibration_data:
                    # 创建校准数据集
                    def calibration_data_reader():
                        for data in calibration_data:
                            yield {"input": data}
                    
                    quantize_static(
                        model_path,
                        output_path,
                        calibration_data_reader
                    )
                else:
                    # 回退到动态量化
                    quantize_dynamic(
                        model_path,
                        output_path,
                        weight_type=QuantType.QUInt8
                    )
            
            self.logger.info(f"ONNX模型量化完成: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"ONNX模型量化失败: {e}")
            return model_path
    
    def estimate_quantization_speedup(self, original_size: float, quantized_size: float) -> Dict[str, float]:
        """估算量化加速效果
        
        Args:
            original_size: 原始模型大小（MB）
            quantized_size: 量化后模型大小（MB）
            
        Returns:
            Dict[str, float]: 加速效果估算
        """
        size_reduction = (original_size - quantized_size) / original_size
        
        # 经验公式估算
        if self.quantization_type == QuantizationType.INT8:
            speed_improvement = 1.5 + size_reduction * 0.5
            memory_reduction = size_reduction * 0.8
        elif self.quantization_type == QuantizationType.FP16:
            speed_improvement = 1.2 + size_reduction * 0.3
            memory_reduction = size_reduction * 0.6
        else:
            speed_improvement = 1.1 + size_reduction * 0.2
            memory_reduction = size_reduction * 0.4
        
        return {
            "size_reduction": size_reduction,
            "speed_improvement": speed_improvement,
            "memory_reduction": memory_reduction,
            "estimated_fps_gain": speed_improvement - 1.0
        }

class InferenceAccelerator:
    """推理加速器
    
    支持多种推理引擎和硬件加速
    """
    
    def __init__(self, acceleration_type: AccelerationType = AccelerationType.CPU):
        self.acceleration_type = acceleration_type
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session = None
        self.providers = []
        
        # 初始化推理引擎
        self._initialize_engine()
    
    def _initialize_engine(self):
        """初始化推理引擎"""
        try:
            if self.acceleration_type == AccelerationType.ONNX:
                if ort is None:
                    raise ImportError("ONNXRuntime未安装")
                
                # 配置执行提供者
                if self.acceleration_type == AccelerationType.GPU:
                    self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                else:
                    self.providers = ['CPUExecutionProvider']
                
            elif self.acceleration_type == AccelerationType.TENSORRT:
                if trt is None:
                    raise ImportError("TensorRT未安装")
                
                self.providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
            
            self.logger.info(f"推理引擎初始化完成: {self.acceleration_type.value}")
            
        except Exception as e:
            self.logger.error(f"推理引擎初始化失败: {e}")
            # 回退到CPU
            self.acceleration_type = AccelerationType.CPU
            self.providers = ['CPUExecutionProvider']
    
    def load_model(self, model_path: str) -> bool:
        """加载模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            bool: 是否成功
        """
        try:
            if self.acceleration_type in [AccelerationType.ONNX, AccelerationType.TENSORRT]:
                # ONNX Runtime
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                # 设置线程数
                session_options.intra_op_num_threads = multiprocessing.cpu_count()
                session_options.inter_op_num_threads = multiprocessing.cpu_count()
                
                self.session = ort.InferenceSession(
                    model_path,
                    sess_options=session_options,
                    providers=self.providers
                )
                
                self.logger.info(f"模型加载成功: {model_path}")
                self.logger.info(f"使用提供者: {self.session.get_providers()}")
                return True
                
            else:
                self.logger.warning(f"不支持的加速类型: {self.acceleration_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            return False
    
    def inference(self, input_data: np.ndarray) -> Optional[np.ndarray]:
        """执行推理
        
        Args:
            input_data: 输入数据
            
        Returns:
            Optional[np.ndarray]: 推理结果
        """
        if self.session is None:
            return None
        
        try:
            # 获取输入名称
            input_name = self.session.get_inputs()[0].name
            
            # 执行推理
            start_time = time.time()
            outputs = self.session.run(None, {input_name: input_data})
            inference_time = (time.time() - start_time) * 1000
            
            self.logger.debug(f"推理完成，耗时: {inference_time:.2f}ms")
            
            return outputs[0] if outputs else None
            
        except Exception as e:
            self.logger.error(f"推理失败: {e}")
            return None
    
    def batch_inference(self, input_batch: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """批量推理
        
        Args:
            input_batch: 输入批次
            
        Returns:
            List[Optional[np.ndarray]]: 推理结果列表
        """
        if not input_batch:
            return []
        
        try:
            # 堆叠为批次
            batch_data = np.stack(input_batch, axis=0)
            
            # 执行批量推理
            batch_output = self.inference(batch_data)
            
            if batch_output is not None:
                # 分解批次结果
                return [batch_output[i] for i in range(len(input_batch))]
            else:
                return [None] * len(input_batch)
                
        except Exception as e:
            self.logger.error(f"批量推理失败: {e}")
            return [None] * len(input_batch)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        if self.session is None:
            return {}
        
        try:
            inputs = self.session.get_inputs()
            outputs = self.session.get_outputs()
            
            return {
                "input_shapes": [inp.shape for inp in inputs],
                "input_types": [inp.type for inp in inputs],
                "output_shapes": [out.shape for out in outputs],
                "output_types": [out.type for out in outputs],
                "providers": self.session.get_providers()
            }
            
        except Exception as e:
            self.logger.error(f"获取模型信息失败: {e}")
            return {}

class ParallelProcessor:
    """并行处理器
    
    支持多线程和多进程并行计算
    """
    
    def __init__(self, num_workers: int = 4, use_processes: bool = False):
        self.num_workers = num_workers
        self.use_processes = use_processes
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 创建执行器
        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=num_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
        # 任务队列
        self.task_queue = Queue()
        self.result_queue = Queue()
        
        # 工作线程
        self.workers = []
        self.running = False
    
    def start(self):
        """启动并行处理"""
        if self.running:
            return
        
        self.running = True
        
        # 启动工作线程
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
        
        self.logger.info(f"并行处理器启动，工作线程数: {self.num_workers}")
    
    def stop(self):
        """停止并行处理"""
        if not self.running:
            return
        
        self.running = False
        
        # 等待工作线程结束
        for worker in self.workers:
            worker.join(timeout=1.0)
        
        self.workers.clear()
        self.executor.shutdown(wait=True)
        
        self.logger.info("并行处理器已停止")
    
    def _worker_loop(self, worker_id: int):
        """工作线程循环
        
        Args:
            worker_id: 工作线程ID
        """
        while self.running:
            try:
                # 获取任务
                task = self.task_queue.get(timeout=0.1)
                if task is None:
                    break
                
                task_id, func, args, kwargs = task
                
                # 执行任务
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = str(e)
                
                execution_time = time.time() - start_time
                
                # 返回结果
                self.result_queue.put({
                    "task_id": task_id,
                    "worker_id": worker_id,
                    "result": result,
                    "success": success,
                    "error": error,
                    "execution_time": execution_time
                })
                
                self.task_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"工作线程 {worker_id} 错误: {e}")
    
    def submit_task(self, task_id: str, func: Callable, *args, **kwargs):
        """提交任务
        
        Args:
            task_id: 任务ID
            func: 执行函数
            *args: 位置参数
            **kwargs: 关键字参数
        """
        if not self.running:
            self.start()
        
        self.task_queue.put((task_id, func, args, kwargs))
    
    def get_result(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """获取结果
        
        Args:
            timeout: 超时时间
            
        Returns:
            Optional[Dict[str, Any]]: 结果
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def parallel_map(self, func: Callable, data_list: List[Any], chunk_size: int = 1) -> List[Any]:
        """并行映射
        
        Args:
            func: 处理函数
            data_list: 数据列表
            chunk_size: 块大小
            
        Returns:
            List[Any]: 结果列表
        """
        if not data_list:
            return []
        
        # 分块处理
        chunks = [data_list[i:i + chunk_size] for i in range(0, len(data_list), chunk_size)]
        
        # 提交任务
        futures = []
        for i, chunk in enumerate(chunks):
            if chunk_size == 1:
                future = self.executor.submit(func, chunk[0])
            else:
                future = self.executor.submit(lambda c: [func(item) for item in c], chunk)
            futures.append(future)
        
        # 收集结果
        results = []
        for future in futures:
            try:
                result = future.result(timeout=10.0)
                if chunk_size == 1:
                    results.append(result)
                else:
                    results.extend(result)
            except Exception as e:
                self.logger.error(f"并行任务失败: {e}")
                if chunk_size == 1:
                    results.append(None)
                else:
                    results.extend([None] * chunk_size)
        
        return results

class MemoryManager:
    """内存管理器
    
    优化内存使用和垃圾回收
    """
    
    def __init__(self, max_memory_mb: float = 512.0):
        self.max_memory_mb = max_memory_mb
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 内存监控
        self.memory_usage_history = []
        self.gc_count = 0
        
        # 缓存管理
        self.cache = {}
        self.cache_access_count = {}
        self.max_cache_size = 100
    
    def get_memory_usage(self) -> float:
        """获取当前内存使用量（MB）
        
        Returns:
            float: 内存使用量
        """
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb
        except ImportError:
            # 回退方法
            import sys
            return sys.getsizeof(gc.get_objects()) / 1024 / 1024
    
    def check_memory_pressure(self) -> bool:
        """检查内存压力
        
        Returns:
            bool: 是否存在内存压力
        """
        current_usage = self.get_memory_usage()
        self.memory_usage_history.append(current_usage)
        
        # 保持历史记录长度
        if len(self.memory_usage_history) > 100:
            self.memory_usage_history.pop(0)
        
        return current_usage > self.max_memory_mb * 0.8
    
    def force_garbage_collection(self):
        """强制垃圾回收"""
        before_usage = self.get_memory_usage()
        
        # 执行垃圾回收
        collected = gc.collect()
        
        after_usage = self.get_memory_usage()
        freed_mb = before_usage - after_usage
        
        self.gc_count += 1
        
        self.logger.info(f"垃圾回收完成，释放内存: {freed_mb:.2f}MB，回收对象: {collected}")
    
    def manage_cache(self, key: str, value: Any = None) -> Any:
        """管理缓存
        
        Args:
            key: 缓存键
            value: 缓存值（如果为None则为获取操作）
            
        Returns:
            Any: 缓存值
        """
        if value is not None:
            # 存储操作
            if len(self.cache) >= self.max_cache_size:
                # 清理最少使用的缓存
                lru_key = min(self.cache_access_count.keys(), 
                             key=lambda k: self.cache_access_count[k])
                del self.cache[lru_key]
                del self.cache_access_count[lru_key]
            
            self.cache[key] = value
            self.cache_access_count[key] = 1
            return value
        else:
            # 获取操作
            if key in self.cache:
                self.cache_access_count[key] += 1
                return self.cache[key]
            return None
    
    def clear_cache(self):
        """清空缓存"""
        cleared_count = len(self.cache)
        self.cache.clear()
        self.cache_access_count.clear()
        
        self.logger.info(f"缓存已清空，清理项目数: {cleared_count}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计信息
        
        Returns:
            Dict[str, Any]: 内存统计
        """
        current_usage = self.get_memory_usage()
        
        return {
            "current_usage_mb": current_usage,
            "max_memory_mb": self.max_memory_mb,
            "usage_percentage": (current_usage / self.max_memory_mb) * 100,
            "gc_count": self.gc_count,
            "cache_size": len(self.cache),
            "memory_pressure": self.check_memory_pressure(),
            "avg_usage_mb": np.mean(self.memory_usage_history) if self.memory_usage_history else 0
        }

class PerformanceOptimizer:
    """性能优化器主类
    
    整合各种优化技术
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化组件
        self.quantizer = ModelQuantizer(config.quantization_type)
        self.accelerator = InferenceAccelerator(config.acceleration_type)
        self.parallel_processor = ParallelProcessor(
            num_workers=config.num_threads,
            use_processes=False
        )
        self.memory_manager = MemoryManager(config.max_memory_mb)
        
        # 性能监控
        self.performance_history = []
        self.optimization_applied = []
        
        # 自适应优化
        self.adaptive_enabled = True
        self.performance_threshold = 0.8  # 性能阈值
        
        self.logger.info(f"性能优化器初始化完成，优化级别: {config.optimization_level.value}")
    
    def optimize_model(self, model_path: str, output_path: str, calibration_data: Optional[List] = None) -> str:
        """优化模型
        
        Args:
            model_path: 输入模型路径
            output_path: 输出模型路径
            calibration_data: 校准数据
            
        Returns:
            str: 优化后模型路径
        """
        try:
            optimized_path = model_path
            
            # 应用量化
            if self.config.optimization_level != OptimizationLevel.NONE:
                self.logger.info("开始模型量化...")
                optimized_path = self.quantizer.quantize_onnx_model(
                    optimized_path, output_path, calibration_data
                )
                self.optimization_applied.append("quantization")
            
            # 加载优化后的模型
            if self.accelerator.load_model(optimized_path):
                self.optimization_applied.append("acceleration")
            
            # 启动并行处理
            if self.config.enable_parallel:
                self.parallel_processor.start()
                self.optimization_applied.append("parallel")
            
            self.logger.info(f"模型优化完成: {optimized_path}")
            self.logger.info(f"应用的优化: {self.optimization_applied}")
            
            return optimized_path
            
        except Exception as e:
            self.logger.error(f"模型优化失败: {e}")
            return model_path
    
    def optimized_inference(self, input_data: np.ndarray) -> Tuple[Optional[np.ndarray], PerformanceMetrics]:
        """优化推理
        
        Args:
            input_data: 输入数据
            
        Returns:
            Tuple[Optional[np.ndarray], PerformanceMetrics]: 推理结果和性能指标
        """
        start_time = time.time()
        
        # 检查内存压力
        if self.memory_manager.check_memory_pressure():
            self.memory_manager.force_garbage_collection()
        
        # 执行推理
        result = self.accelerator.inference(input_data)
        
        # 计算性能指标
        inference_time = (time.time() - start_time) * 1000
        fps = 1000.0 / inference_time if inference_time > 0 else 0
        memory_usage = self.memory_manager.get_memory_usage()
        
        metrics = PerformanceMetrics(
            inference_time=inference_time,
            fps=fps,
            memory_usage=memory_usage,
            cpu_usage=0.0,  # 需要额外监控
            latency=inference_time
        )
        
        # 记录性能历史
        self.performance_history.append(metrics)
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
        
        # 自适应优化
        if self.adaptive_enabled:
            self._adaptive_optimization(metrics)
        
        return result, metrics
    
    def batch_optimized_inference(self, input_batch: List[np.ndarray]) -> Tuple[List[Optional[np.ndarray]], PerformanceMetrics]:
        """批量优化推理
        
        Args:
            input_batch: 输入批次
            
        Returns:
            Tuple[List[Optional[np.ndarray]], PerformanceMetrics]: 推理结果和性能指标
        """
        start_time = time.time()
        
        # 并行处理
        if self.config.enable_parallel and len(input_batch) > 1:
            results = self.parallel_processor.parallel_map(
                self.accelerator.inference, input_batch
            )
        else:
            results = [self.accelerator.inference(data) for data in input_batch]
        
        # 计算性能指标
        total_time = (time.time() - start_time) * 1000
        avg_inference_time = total_time / len(input_batch) if input_batch else 0
        fps = 1000.0 / avg_inference_time if avg_inference_time > 0 else 0
        throughput = len(input_batch) / (total_time / 1000) if total_time > 0 else 0
        
        metrics = PerformanceMetrics(
            inference_time=avg_inference_time,
            fps=fps,
            memory_usage=self.memory_manager.get_memory_usage(),
            cpu_usage=0.0,
            throughput=throughput
        )
        
        return results, metrics
    
    def _adaptive_optimization(self, metrics: PerformanceMetrics):
        """自适应优化
        
        Args:
            metrics: 性能指标
        """
        # 计算性能得分
        fps_score = min(metrics.fps / self.config.target_fps, 1.0)
        memory_score = max(1.0 - metrics.memory_usage / self.config.max_memory_mb, 0.0)
        overall_score = (fps_score + memory_score) / 2
        
        # 如果性能低于阈值，应用更激进的优化
        if overall_score < self.performance_threshold:
            if self.config.optimization_level == OptimizationLevel.BASIC:
                self.config.optimization_level = OptimizationLevel.AGGRESSIVE
                self.logger.info("切换到激进优化模式")
            
            # 清理缓存
            if metrics.memory_usage > self.config.max_memory_mb * 0.9:
                self.memory_manager.clear_cache()
                self.memory_manager.force_garbage_collection()
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """获取优化报告
        
        Returns:
            Dict[str, Any]: 优化报告
        """
        if not self.performance_history:
            return {"error": "无性能数据"}
        
        # 计算统计信息
        recent_metrics = self.performance_history[-10:]  # 最近10次
        avg_fps = np.mean([m.fps for m in recent_metrics])
        avg_inference_time = np.mean([m.inference_time for m in recent_metrics])
        avg_memory = np.mean([m.memory_usage for m in recent_metrics])
        
        # 性能改进估算
        if len(self.performance_history) >= 20:
            early_metrics = self.performance_history[:10]
            early_fps = np.mean([m.fps for m in early_metrics])
            fps_improvement = (avg_fps - early_fps) / early_fps if early_fps > 0 else 0
        else:
            fps_improvement = 0
        
        return {
            "optimization_config": {
                "level": self.config.optimization_level.value,
                "quantization": self.config.quantization_type.value,
                "acceleration": self.config.acceleration_type.value,
                "parallel_enabled": self.config.enable_parallel
            },
            "applied_optimizations": self.optimization_applied,
            "performance_metrics": {
                "avg_fps": avg_fps,
                "avg_inference_time_ms": avg_inference_time,
                "avg_memory_usage_mb": avg_memory,
                "fps_improvement_percentage": fps_improvement * 100
            },
            "memory_stats": self.memory_manager.get_memory_stats(),
            "model_info": self.accelerator.get_model_info(),
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """生成优化建议
        
        Returns:
            List[str]: 优化建议
        """
        recommendations = []
        
        if not self.performance_history:
            return ["需要更多性能数据来生成建议"]
        
        recent_metrics = self.performance_history[-5:]
        avg_fps = np.mean([m.fps for m in recent_metrics])
        avg_memory = np.mean([m.memory_usage for m in recent_metrics])
        
        # FPS建议
        if avg_fps < self.config.target_fps * 0.8:
            recommendations.append("FPS低于目标，建议启用更激进的量化或减少模型复杂度")
        
        # 内存建议
        if avg_memory > self.config.max_memory_mb * 0.8:
            recommendations.append("内存使用率高，建议启用更多缓存清理或减少批处理大小")
        
        # 并行建议
        if not self.config.enable_parallel and self.config.num_threads > 1:
            recommendations.append("建议启用并行处理以提升性能")
        
        # 量化建议
        if self.config.quantization_type == QuantizationType.DYNAMIC:
            recommendations.append("考虑使用静态量化或INT8量化以获得更好性能")
        
        return recommendations if recommendations else ["当前配置已经很好优化"]
    
    def cleanup(self):
        """清理资源"""
        try:
            self.parallel_processor.stop()
            self.memory_manager.clear_cache()
            self.memory_manager.force_garbage_collection()
            
            self.logger.info("性能优化器资源清理完成")
            
        except Exception as e:
            self.logger.error(f"资源清理失败: {e}")

# 测试代码
if __name__ == "__main__":
    # 创建优化配置
    config = OptimizationConfig(
        optimization_level=OptimizationLevel.AGGRESSIVE,
        quantization_type=QuantizationType.DYNAMIC,
        acceleration_type=AccelerationType.ONNX,
        target_fps=30.0,
        max_memory_mb=512.0,
        enable_parallel=True,
        num_threads=4
    )
    
    # 创建性能优化器
    optimizer = PerformanceOptimizer(config)
    
    # 模拟推理测试
    test_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    print("开始性能测试...")
    
    # 单次推理测试
    for i in range(10):
        result, metrics = optimizer.optimized_inference(test_data)
        print(f"推理 {i+1}: FPS={metrics.fps:.2f}, 内存={metrics.memory_usage:.2f}MB")
    
    # 批量推理测试
    batch_data = [test_data for _ in range(5)]
    batch_results, batch_metrics = optimizer.batch_optimized_inference(batch_data)
    print(f"批量推理: 吞吐量={batch_metrics.throughput:.2f} samples/s")
    
    # 获取优化报告
    report = optimizer.get_optimization_report()
    print("\n优化报告:")
    for key, value in report.items():
        print(f"  {key}: {value}")
    
    # 清理资源
    optimizer.cleanup()