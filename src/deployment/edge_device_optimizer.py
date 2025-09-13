#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
边缘设备部署优化器
针对K230和ESP32等边缘设备进行模型压缩、量化和性能优化
实现人体跟踪功能在资源受限环境下的高效部署
"""

import os
import json
import time
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import cv2

# 导入相关模块
try:
    from ..models.human_body_detector import HumanBodyDetector, HumanBodyDetection
    from ..tracking.multimodal_fusion_tracker import MultimodalFusionTracker, FusionConfig
    from ..platforms.k230_platform import K230Platform
    from ..platforms.esp32_platform import ESP32Platform
except ImportError:
    # 兼容性导入
    pass

logger = logging.getLogger(__name__)

class EdgeDevice(Enum):
    """边缘设备类型"""
    K230 = "k230"
    ESP32 = "esp32"
    ESP32_S3 = "esp32_s3"
    RASPBERRY_PI = "raspberry_pi"
    JETSON_NANO = "jetson_nano"

class OptimizationLevel(Enum):
    """优化级别"""
    SPEED = "speed"  # 速度优先
    ACCURACY = "accuracy"  # 精度优先
    BALANCED = "balanced"  # 平衡模式
    ULTRA_LIGHT = "ultra_light"  # 超轻量

class ModelFormat(Enum):
    """模型格式"""
    ONNX = "onnx"
    TFLITE = "tflite"
    NCNN = "ncnn"
    OPENVINO = "openvino"
    TENSORRT = "tensorrt"
    KMODEL = "kmodel"  # K230专用

@dataclass
class DeviceSpec:
    """设备规格"""
    device_type: EdgeDevice
    cpu_cores: int
    cpu_freq_mhz: int
    ram_mb: int
    flash_mb: int
    npu_tops: float = 0.0  # NPU算力
    gpu_gflops: float = 0.0  # GPU算力
    camera_resolution: Tuple[int, int] = (640, 480)
    max_fps: int = 30
    power_budget_mw: int = 1000
    supported_formats: List[ModelFormat] = field(default_factory=list)
    
@dataclass
class OptimizationConfig:
    """优化配置"""
    device_spec: DeviceSpec
    optimization_level: OptimizationLevel
    target_fps: int = 15
    max_latency_ms: int = 100
    accuracy_threshold: float = 0.8
    
    # 模型压缩参数
    quantization_bits: int = 8
    pruning_ratio: float = 0.3
    knowledge_distillation: bool = True
    
    # 推理优化
    batch_size: int = 1
    input_resolution: Tuple[int, int] = (320, 240)
    enable_tensorrt: bool = False
    enable_npu: bool = True
    
    # 跟踪优化
    max_targets: int = 3
    tracking_interval: int = 2  # 每N帧执行一次跟踪
    feature_cache_size: int = 10

class ModelCompressor:
    """模型压缩器"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def quantize_model(self, model_path: str, output_path: str) -> bool:
        """模型量化
        
        Args:
            model_path: 原始模型路径
            output_path: 量化后模型路径
            
        Returns:
            bool: 是否成功
        """
        try:
            self.logger.info(f"开始量化模型: {model_path}")
            
            # 根据设备类型选择量化方法
            if self.config.device_spec.device_type == EdgeDevice.K230:
                return self._quantize_for_k230(model_path, output_path)
            elif self.config.device_spec.device_type in [EdgeDevice.ESP32, EdgeDevice.ESP32_S3]:
                return self._quantize_for_esp32(model_path, output_path)
            else:
                return self._quantize_generic(model_path, output_path)
                
        except Exception as e:
            self.logger.error(f"模型量化失败: {e}")
            return False
    
    def _quantize_for_k230(self, model_path: str, output_path: str) -> bool:
        """K230专用量化"""
        # K230使用kmodel格式，支持INT8量化
        self.logger.info("执行K230 INT8量化")
        
        # 模拟量化过程（实际需要使用K230工具链）
        quantization_config = {
            "input_model": model_path,
            "output_model": output_path,
            "quantization_type": "int8",
            "calibration_dataset": "auto",
            "optimization_level": self.config.optimization_level.value
        }
        
        # 这里应该调用K230的量化工具
        # 例如: nncase或其他K230专用工具
        
        self.logger.info(f"K230量化配置: {quantization_config}")
        return True
    
    def _quantize_for_esp32(self, model_path: str, output_path: str) -> bool:
        """ESP32专用量化"""
        # ESP32使用TensorFlow Lite量化
        self.logger.info("执行ESP32 TFLite量化")
        
        try:
            # 模拟TFLite量化
            quantization_config = {
                "input_model": model_path,
                "output_model": output_path,
                "quantization_type": f"int{self.config.quantization_bits}",
                "optimization": "OPTIMIZE_FOR_SIZE"
            }
            
            self.logger.info(f"ESP32量化配置: {quantization_config}")
            return True
            
        except Exception as e:
            self.logger.error(f"ESP32量化失败: {e}")
            return False
    
    def _quantize_generic(self, model_path: str, output_path: str) -> bool:
        """通用量化"""
        self.logger.info("执行通用模型量化")
        
        # 通用量化逻辑
        quantization_config = {
            "input_model": model_path,
            "output_model": output_path,
            "quantization_bits": self.config.quantization_bits,
            "calibration_method": "entropy"
        }
        
        self.logger.info(f"通用量化配置: {quantization_config}")
        return True
    
    def prune_model(self, model_path: str, output_path: str) -> bool:
        """模型剪枝
        
        Args:
            model_path: 原始模型路径
            output_path: 剪枝后模型路径
            
        Returns:
            bool: 是否成功
        """
        try:
            self.logger.info(f"开始剪枝模型: {model_path}")
            
            pruning_config = {
                "input_model": model_path,
                "output_model": output_path,
                "pruning_ratio": self.config.pruning_ratio,
                "pruning_method": "magnitude",
                "fine_tune_epochs": 10
            }
            
            # 这里应该实现具体的剪枝算法
            self.logger.info(f"剪枝配置: {pruning_config}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"模型剪枝失败: {e}")
            return False
    
    def knowledge_distillation(self, teacher_model: str, student_model: str, output_path: str) -> bool:
        """知识蒸馏
        
        Args:
            teacher_model: 教师模型路径
            student_model: 学生模型路径
            output_path: 蒸馏后模型路径
            
        Returns:
            bool: 是否成功
        """
        try:
            self.logger.info(f"开始知识蒸馏: {teacher_model} -> {student_model}")
            
            distillation_config = {
                "teacher_model": teacher_model,
                "student_model": student_model,
                "output_model": output_path,
                "temperature": 4.0,
                "alpha": 0.7,
                "training_epochs": 50
            }
            
            # 这里应该实现知识蒸馏训练
            self.logger.info(f"蒸馏配置: {distillation_config}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"知识蒸馏失败: {e}")
            return False

class InferenceOptimizer:
    """推理优化器"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 性能统计
        self.inference_times = []
        self.memory_usage = []
        
    def optimize_inference_engine(self, model_path: str) -> Dict[str, Any]:
        """优化推理引擎
        
        Args:
            model_path: 模型路径
            
        Returns:
            Dict[str, Any]: 优化配置
        """
        device_type = self.config.device_spec.device_type
        
        if device_type == EdgeDevice.K230:
            return self._optimize_for_k230(model_path)
        elif device_type in [EdgeDevice.ESP32, EdgeDevice.ESP32_S3]:
            return self._optimize_for_esp32(model_path)
        else:
            return self._optimize_generic(model_path)
    
    def _optimize_for_k230(self, model_path: str) -> Dict[str, Any]:
        """K230推理优化"""
        config = {
            "runtime": "nncase",
            "device": "kpu",  # K230的NPU
            "model_format": "kmodel",
            "input_layout": "NCHW",
            "output_layout": "NCHW",
            "precision": "int8",
            "batch_size": 1,
            "num_threads": self.config.device_spec.cpu_cores,
            "enable_npu": True,
            "npu_frequency": "high",
            "memory_pool_size": min(self.config.device_spec.ram_mb // 4, 64) * 1024 * 1024
        }
        
        self.logger.info(f"K230推理配置: {config}")
        return config
    
    def _optimize_for_esp32(self, model_path: str) -> Dict[str, Any]:
        """ESP32推理优化"""
        config = {
            "runtime": "tflite_micro",
            "device": "cpu",
            "model_format": "tflite",
            "precision": "int8",
            "batch_size": 1,
            "num_threads": 1,  # ESP32通常单核
            "enable_xnnpack": False,  # ESP32不支持
            "memory_arena_size": min(self.config.device_spec.ram_mb // 8, 32) * 1024,
            "enable_gpu": False
        }
        
        self.logger.info(f"ESP32推理配置: {config}")
        return config
    
    def _optimize_generic(self, model_path: str) -> Dict[str, Any]:
        """通用推理优化"""
        config = {
            "runtime": "onnxruntime",
            "device": "cpu",
            "model_format": "onnx",
            "precision": "fp16",
            "batch_size": self.config.batch_size,
            "num_threads": self.config.device_spec.cpu_cores,
            "enable_tensorrt": self.config.enable_tensorrt,
            "memory_limit_mb": self.config.device_spec.ram_mb // 2
        }
        
        self.logger.info(f"通用推理配置: {config}")
        return config
    
    def benchmark_inference(self, model_path: str, test_data: np.ndarray, iterations: int = 100) -> Dict[str, float]:
        """推理性能基准测试
        
        Args:
            model_path: 模型路径
            test_data: 测试数据
            iterations: 测试迭代次数
            
        Returns:
            Dict[str, float]: 性能指标
        """
        self.logger.info(f"开始推理性能测试: {iterations}次迭代")
        
        inference_times = []
        memory_usage = []
        
        for i in range(iterations):
            start_time = time.time()
            
            # 模拟推理过程
            # 实际应用中这里应该调用真实的推理引擎
            time.sleep(0.01)  # 模拟推理延迟
            
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000  # 转换为毫秒
            inference_times.append(inference_time)
            
            # 模拟内存使用
            memory_usage.append(np.random.uniform(10, 50))  # MB
        
        # 计算统计指标
        avg_inference_time = np.mean(inference_times)
        min_inference_time = np.min(inference_times)
        max_inference_time = np.max(inference_times)
        std_inference_time = np.std(inference_times)
        
        avg_memory = np.mean(memory_usage)
        max_memory = np.max(memory_usage)
        
        fps = 1000.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        results = {
            "avg_inference_time_ms": avg_inference_time,
            "min_inference_time_ms": min_inference_time,
            "max_inference_time_ms": max_inference_time,
            "std_inference_time_ms": std_inference_time,
            "avg_memory_mb": avg_memory,
            "max_memory_mb": max_memory,
            "fps": fps,
            "meets_latency_requirement": avg_inference_time <= self.config.max_latency_ms,
            "meets_fps_requirement": fps >= self.config.target_fps
        }
        
        self.logger.info(f"推理性能测试结果: {results}")
        return results

class LightweightTracker:
    """轻量化跟踪器"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 轻量化跟踪参数
        self.max_targets = config.max_targets
        self.tracking_interval = config.tracking_interval
        self.feature_cache_size = config.feature_cache_size
        
        # 跟踪状态
        self.frame_count = 0
        self.active_tracks = {}
        self.feature_cache = {}
        
    def update(self, detections: List[Any], image: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """轻量化跟踪更新
        
        Args:
            detections: 检测结果
            image: 当前帧图像
            
        Returns:
            List[Dict[str, Any]]: 跟踪结果
        """
        self.frame_count += 1
        
        # 根据跟踪间隔决定是否执行完整跟踪
        if self.frame_count % self.tracking_interval == 0:
            return self._full_tracking_update(detections, image)
        else:
            return self._lightweight_update(detections)
    
    def _full_tracking_update(self, detections: List[Any], image: Optional[np.ndarray]) -> List[Dict[str, Any]]:
        """完整跟踪更新"""
        self.logger.debug(f"执行完整跟踪更新 - 帧{self.frame_count}")
        
        # 限制检测数量
        if len(detections) > self.max_targets:
            # 按置信度排序，保留前N个
            detections = sorted(detections, key=lambda x: getattr(x, 'confidence', 0), reverse=True)[:self.max_targets]
        
        tracking_results = []
        
        for i, detection in enumerate(detections):
            track_info = {
                "track_id": f"track_{i}",
                "bbox": getattr(detection, 'bbox', (0, 0, 100, 100)),
                "confidence": getattr(detection, 'confidence', 0.5),
                "timestamp": time.time(),
                "frame_id": self.frame_count
            }
            
            # 提取轻量化特征
            if image is not None:
                feature = self._extract_lightweight_feature(image, track_info["bbox"])
                track_info["feature"] = feature
                
                # 缓存特征
                self._cache_feature(track_info["track_id"], feature)
            
            tracking_results.append(track_info)
        
        return tracking_results
    
    def _lightweight_update(self, detections: List[Any]) -> List[Dict[str, Any]]:
        """轻量化更新（跳帧）"""
        self.logger.debug(f"执行轻量化更新 - 帧{self.frame_count}")
        
        # 简化的跟踪逻辑，主要基于位置预测
        tracking_results = []
        
        for i, detection in enumerate(detections[:self.max_targets]):
            track_info = {
                "track_id": f"track_{i}",
                "bbox": getattr(detection, 'bbox', (0, 0, 100, 100)),
                "confidence": getattr(detection, 'confidence', 0.5),
                "timestamp": time.time(),
                "frame_id": self.frame_count,
                "is_predicted": True  # 标记为预测结果
            }
            
            tracking_results.append(track_info)
        
        return tracking_results
    
    def _extract_lightweight_feature(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """提取轻量化特征"""
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return np.zeros(16)  # 返回固定长度的零向量
        
        # 超轻量化特征：颜色直方图 + 简单统计
        roi_small = cv2.resize(roi, (32, 32))
        
        # 颜色特征（每个通道4个bin）
        hist_b = cv2.calcHist([roi_small], [0], None, [4], [0, 256])
        hist_g = cv2.calcHist([roi_small], [1], None, [4], [0, 256])
        hist_r = cv2.calcHist([roi_small], [2], None, [4], [0, 256])
        
        # 统计特征
        mean_color = np.mean(roi_small, axis=(0, 1))
        
        # 合并特征
        feature = np.concatenate([
            hist_b.flatten(),
            hist_g.flatten(), 
            hist_r.flatten(),
            mean_color
        ])
        
        # 归一化
        if np.linalg.norm(feature) > 0:
            feature = feature / np.linalg.norm(feature)
        
        return feature[:16]  # 限制特征维度
    
    def _cache_feature(self, track_id: str, feature: np.ndarray):
        """缓存特征"""
        if track_id not in self.feature_cache:
            self.feature_cache[track_id] = []
        
        self.feature_cache[track_id].append(feature)
        
        # 限制缓存大小
        if len(self.feature_cache[track_id]) > self.feature_cache_size:
            self.feature_cache[track_id].pop(0)
    
    def get_cached_feature(self, track_id: str) -> Optional[np.ndarray]:
        """获取缓存的特征"""
        if track_id in self.feature_cache and self.feature_cache[track_id]:
            # 返回最新的特征
            return self.feature_cache[track_id][-1]
        return None

class EdgeDeviceOptimizer:
    """边缘设备优化器主类"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 子模块
        self.model_compressor = ModelCompressor(config)
        self.inference_optimizer = InferenceOptimizer(config)
        self.lightweight_tracker = LightweightTracker(config)
        
        # 性能监控
        self.performance_stats = {
            "total_frames": 0,
            "avg_fps": 0.0,
            "avg_latency_ms": 0.0,
            "memory_usage_mb": 0.0,
            "power_consumption_mw": 0.0
        }
        
        self.logger.info(f"边缘设备优化器初始化完成 - 设备: {config.device_spec.device_type.value}")
    
    def optimize_model_pipeline(self, model_path: str, output_dir: str) -> Dict[str, str]:
        """优化模型管道
        
        Args:
            model_path: 原始模型路径
            output_dir: 输出目录
            
        Returns:
            Dict[str, str]: 优化后的模型路径
        """
        self.logger.info(f"开始优化模型管道: {model_path}")
        
        output_paths = {}
        
        try:
            # 1. 模型剪枝
            if self.config.pruning_ratio > 0:
                pruned_path = os.path.join(output_dir, "pruned_model.onnx")
                if self.model_compressor.prune_model(model_path, pruned_path):
                    output_paths["pruned"] = pruned_path
                    model_path = pruned_path  # 使用剪枝后的模型继续优化
            
            # 2. 模型量化
            quantized_path = os.path.join(output_dir, f"quantized_model_{self.config.device_spec.device_type.value}")
            if self.config.device_spec.device_type == EdgeDevice.K230:
                quantized_path += ".kmodel"
            elif self.config.device_spec.device_type in [EdgeDevice.ESP32, EdgeDevice.ESP32_S3]:
                quantized_path += ".tflite"
            else:
                quantized_path += ".onnx"
            
            if self.model_compressor.quantize_model(model_path, quantized_path):
                output_paths["quantized"] = quantized_path
            
            # 3. 知识蒸馏（如果启用）
            if self.config.knowledge_distillation:
                distilled_path = os.path.join(output_dir, "distilled_model.onnx")
                # 这里需要一个预训练的轻量化学生模型
                student_model_path = os.path.join(output_dir, "student_model.onnx")
                if os.path.exists(student_model_path):
                    if self.model_compressor.knowledge_distillation(model_path, student_model_path, distilled_path):
                        output_paths["distilled"] = distilled_path
            
            self.logger.info(f"模型优化完成: {output_paths}")
            return output_paths
            
        except Exception as e:
            self.logger.error(f"模型优化失败: {e}")
            return {}
    
    def deploy_to_device(self, model_path: str, device_config: Dict[str, Any]) -> bool:
        """部署到设备
        
        Args:
            model_path: 优化后的模型路径
            device_config: 设备配置
            
        Returns:
            bool: 是否部署成功
        """
        try:
            self.logger.info(f"开始部署到设备: {self.config.device_spec.device_type.value}")
            
            # 优化推理引擎
            inference_config = self.inference_optimizer.optimize_inference_engine(model_path)
            
            # 根据设备类型执行部署
            if self.config.device_spec.device_type == EdgeDevice.K230:
                return self._deploy_to_k230(model_path, inference_config, device_config)
            elif self.config.device_spec.device_type in [EdgeDevice.ESP32, EdgeDevice.ESP32_S3]:
                return self._deploy_to_esp32(model_path, inference_config, device_config)
            else:
                return self._deploy_generic(model_path, inference_config, device_config)
                
        except Exception as e:
            self.logger.error(f"设备部署失败: {e}")
            return False
    
    def _deploy_to_k230(self, model_path: str, inference_config: Dict[str, Any], device_config: Dict[str, Any]) -> bool:
        """部署到K230"""
        self.logger.info("执行K230部署")
        
        deployment_config = {
            "model_path": model_path,
            "inference_config": inference_config,
            "device_config": device_config,
            "camera_config": {
                "resolution": self.config.device_spec.camera_resolution,
                "fps": min(self.config.target_fps, self.config.device_spec.max_fps),
                "format": "RGB888"
            },
            "tracking_config": {
                "max_targets": self.config.max_targets,
                "tracking_interval": self.config.tracking_interval
            }
        }
        
        # 这里应该调用K230的部署API
        self.logger.info(f"K230部署配置: {deployment_config}")
        return True
    
    def _deploy_to_esp32(self, model_path: str, inference_config: Dict[str, Any], device_config: Dict[str, Any]) -> bool:
        """部署到ESP32"""
        self.logger.info("执行ESP32部署")
        
        deployment_config = {
            "model_path": model_path,
            "inference_config": inference_config,
            "device_config": device_config,
            "camera_config": {
                "resolution": self.config.device_spec.camera_resolution,
                "fps": min(self.config.target_fps, self.config.device_spec.max_fps),
                "format": "RGB565"
            },
            "memory_config": {
                "model_arena_size": inference_config.get("memory_arena_size", 32 * 1024),
                "input_buffer_size": np.prod(self.config.input_resolution) * 3,
                "output_buffer_size": 1024
            }
        }
        
        # 这里应该调用ESP32的部署API
        self.logger.info(f"ESP32部署配置: {deployment_config}")
        return True
    
    def _deploy_generic(self, model_path: str, inference_config: Dict[str, Any], device_config: Dict[str, Any]) -> bool:
        """通用部署"""
        self.logger.info("执行通用部署")
        
        deployment_config = {
            "model_path": model_path,
            "inference_config": inference_config,
            "device_config": device_config
        }
        
        self.logger.info(f"通用部署配置: {deployment_config}")
        return True
    
    def run_performance_test(self, model_path: str, test_images: List[np.ndarray]) -> Dict[str, Any]:
        """运行性能测试
        
        Args:
            model_path: 模型路径
            test_images: 测试图像列表
            
        Returns:
            Dict[str, Any]: 性能测试结果
        """
        self.logger.info("开始性能测试")
        
        results = {
            "device_info": {
                "device_type": self.config.device_spec.device_type.value,
                "cpu_cores": self.config.device_spec.cpu_cores,
                "ram_mb": self.config.device_spec.ram_mb,
                "npu_tops": self.config.device_spec.npu_tops
            },
            "model_info": {
                "model_path": model_path,
                "input_resolution": self.config.input_resolution,
                "optimization_level": self.config.optimization_level.value
            }
        }
        
        # 推理性能测试
        if test_images:
            test_data = test_images[0]  # 使用第一张图像作为测试数据
            inference_results = self.inference_optimizer.benchmark_inference(model_path, test_data)
            results["inference_performance"] = inference_results
        
        # 跟踪性能测试
        tracking_results = self._test_tracking_performance(test_images)
        results["tracking_performance"] = tracking_results
        
        # 整体性能评估
        overall_score = self._calculate_overall_score(results)
        results["overall_score"] = overall_score
        
        self.logger.info(f"性能测试完成: 总分 {overall_score:.2f}")
        return results
    
    def _test_tracking_performance(self, test_images: List[np.ndarray]) -> Dict[str, float]:
        """测试跟踪性能"""
        if not test_images:
            return {}
        
        tracking_times = []
        
        for i, image in enumerate(test_images[:10]):  # 测试前10帧
            start_time = time.time()
            
            # 模拟检测结果
            mock_detections = [{
                "bbox": (100, 100, 200, 300),
                "confidence": 0.8
            }]
            
            # 执行跟踪
            tracking_results = self.lightweight_tracker.update(mock_detections, image)
            
            end_time = time.time()
            tracking_time = (end_time - start_time) * 1000
            tracking_times.append(tracking_time)
        
        return {
            "avg_tracking_time_ms": np.mean(tracking_times) if tracking_times else 0,
            "max_tracking_time_ms": np.max(tracking_times) if tracking_times else 0,
            "tracking_fps": 1000.0 / np.mean(tracking_times) if tracking_times and np.mean(tracking_times) > 0 else 0
        }
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """计算总体性能分数"""
        score = 0.0
        
        # 推理性能分数 (40%)
        if "inference_performance" in results:
            inference_perf = results["inference_performance"]
            fps_score = min(inference_perf.get("fps", 0) / self.config.target_fps, 1.0) * 40
            latency_score = max(0, 1.0 - inference_perf.get("avg_inference_time_ms", 100) / self.config.max_latency_ms) * 40
            score += (fps_score + latency_score) / 2
        
        # 跟踪性能分数 (30%)
        if "tracking_performance" in results:
            tracking_perf = results["tracking_performance"]
            tracking_fps = tracking_perf.get("tracking_fps", 0)
            tracking_score = min(tracking_fps / self.config.target_fps, 1.0) * 30
            score += tracking_score
        
        # 资源使用分数 (30%)
        if "inference_performance" in results:
            memory_usage = results["inference_performance"].get("avg_memory_mb", 0)
            memory_limit = self.config.device_spec.ram_mb * 0.5  # 允许使用50%内存
            memory_score = max(0, 1.0 - memory_usage / memory_limit) * 30
            score += memory_score
        
        return min(score, 100.0)
    
    def generate_deployment_report(self, results: Dict[str, Any], output_path: str):
        """生成部署报告
        
        Args:
            results: 性能测试结果
            output_path: 报告输出路径
        """
        report = {
            "deployment_summary": {
                "device_type": self.config.device_spec.device_type.value,
                "optimization_level": self.config.optimization_level.value,
                "target_fps": self.config.target_fps,
                "max_latency_ms": self.config.max_latency_ms,
                "overall_score": results.get("overall_score", 0)
            },
            "performance_results": results,
            "recommendations": self._generate_recommendations(results),
            "deployment_config": {
                "model_compression": {
                    "quantization_bits": self.config.quantization_bits,
                    "pruning_ratio": self.config.pruning_ratio,
                    "knowledge_distillation": self.config.knowledge_distillation
                },
                "inference_optimization": {
                    "input_resolution": self.config.input_resolution,
                    "batch_size": self.config.batch_size,
                    "enable_npu": self.config.enable_npu
                },
                "tracking_optimization": {
                    "max_targets": self.config.max_targets,
                    "tracking_interval": self.config.tracking_interval,
                    "feature_cache_size": self.config.feature_cache_size
                }
            }
        }
        
        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"部署报告已保存: {output_path}")
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 基于性能结果生成建议
        if "inference_performance" in results:
            inference_perf = results["inference_performance"]
            
            if not inference_perf.get("meets_fps_requirement", False):
                recommendations.append("建议降低输入分辨率或增加量化程度以提升FPS")
            
            if not inference_perf.get("meets_latency_requirement", False):
                recommendations.append("建议启用硬件加速或优化模型结构以降低延迟")
            
            if inference_perf.get("max_memory_mb", 0) > self.config.device_spec.ram_mb * 0.7:
                recommendations.append("内存使用过高，建议进一步压缩模型或减少批处理大小")
        
        # 设备特定建议
        if self.config.device_spec.device_type == EdgeDevice.K230:
            recommendations.append("建议充分利用K230的NPU加速能力")
        elif self.config.device_spec.device_type in [EdgeDevice.ESP32, EdgeDevice.ESP32_S3]:
            recommendations.append("ESP32资源有限，建议使用超轻量化配置")
        
        return recommendations

# 预定义设备规格
DEVICE_SPECS = {
    EdgeDevice.K230: DeviceSpec(
        device_type=EdgeDevice.K230,
        cpu_cores=2,
        cpu_freq_mhz=1600,
        ram_mb=512,
        flash_mb=16,
        npu_tops=1.0,
        camera_resolution=(1920, 1080),
        max_fps=30,
        power_budget_mw=2000,
        supported_formats=[ModelFormat.KMODEL, ModelFormat.ONNX]
    ),
    EdgeDevice.ESP32_S3: DeviceSpec(
        device_type=EdgeDevice.ESP32_S3,
        cpu_cores=2,
        cpu_freq_mhz=240,
        ram_mb=8,
        flash_mb=16,
        npu_tops=0.0,
        camera_resolution=(640, 480),
        max_fps=15,
        power_budget_mw=500,
        supported_formats=[ModelFormat.TFLITE]
    ),
    EdgeDevice.ESP32: DeviceSpec(
        device_type=EdgeDevice.ESP32,
        cpu_cores=2,
        cpu_freq_mhz=240,
        ram_mb=4,
        flash_mb=4,
        npu_tops=0.0,
        camera_resolution=(320, 240),
        max_fps=10,
        power_budget_mw=300,
        supported_formats=[ModelFormat.TFLITE]
    )
}

# 测试代码
if __name__ == "__main__":
    # 创建K230优化配置
    k230_config = OptimizationConfig(
        device_spec=DEVICE_SPECS[EdgeDevice.K230],
        optimization_level=OptimizationLevel.BALANCED,
        target_fps=20,
        max_latency_ms=50,
        input_resolution=(416, 416)
    )
    
    # 创建优化器
    optimizer = EdgeDeviceOptimizer(k230_config)
    
    # 模拟模型优化
    model_path = "./models/human_detector.onnx"
    output_dir = "./optimized_models"
    
    print(f"开始优化模型: {model_path}")
    optimized_models = optimizer.optimize_model_pipeline(model_path, output_dir)
    print(f"优化完成: {optimized_models}")
    
    # 模拟部署
    if optimized_models:
        quantized_model = optimized_models.get("quantized", model_path)
        device_config = {"ip": "192.168.1.100", "port": 8080}
        
        success = optimizer.deploy_to_device(quantized_model, device_config)
        print(f"部署结果: {'成功' if success else '失败'}")
    
    # 性能测试
    test_images = [np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8) for _ in range(5)]
    results = optimizer.run_performance_test(model_path, test_images)
    
    print(f"性能测试结果:")
    print(f"  总体分数: {results.get('overall_score', 0):.2f}")
    if "inference_performance" in results:
        print(f"  推理FPS: {results['inference_performance'].get('fps', 0):.2f}")
        print(f"  平均延迟: {results['inference_performance'].get('avg_inference_time_ms', 0):.2f}ms")
    
    # 生成报告
    report_path = "./deployment_report.json"
    optimizer.generate_deployment_report(results, report_path)
    print(f"部署报告已生成: {report_path}")