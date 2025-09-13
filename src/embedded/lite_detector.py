#!/usr/bin/env python3
"""
轻量级YOLO检测器 - 专为嵌入式设备优化
支持多种推理引擎和动态资源管理
"""

import os
import sys
import time
import logging
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

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

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

@dataclass
class DetectionResult:
    """检测结果"""
    boxes: np.ndarray  # [N, 4] (x1, y1, x2, y2)
    scores: np.ndarray  # [N]
    class_ids: np.ndarray  # [N]
    class_names: List[str]  # [N]
    inference_time: float  # 推理时间(ms)
    preprocessing_time: float  # 预处理时间(ms)
    postprocessing_time: float  # 后处理时间(ms)

@dataclass
class PlatformConfig:
    """平台配置"""
    name: str
    memory_limit_mb: int
    cpu_threads: int
    use_gpu: bool = False
    use_npu: bool = False
    inference_engine: str = "onnx"  # onnx, tflite, tensorrt
    model_format: str = "onnx"
    precision: str = "fp32"  # fp32, fp16, int8
    input_size: Tuple[int, int] = (640, 640)
    batch_size: int = 1
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4

class EmbeddedMemoryManager:
    """嵌入式内存管理器"""
    
    def __init__(self, memory_limit_mb: int):
        self.memory_limit_mb = memory_limit_mb
        self.model_cache = {}
        self.last_access = {}
        self.lock = threading.Lock()
        
    def get_memory_usage(self) -> float:
        """获取当前内存使用量(MB)"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
            
    def can_load_model(self, model_size_mb: float) -> bool:
        """检查是否可以加载模型"""
        current_usage = self.get_memory_usage()
        available = self.memory_limit_mb - current_usage
        return available >= model_size_mb * 1.2  # 20%缓冲
        
    def cleanup_unused_models(self, keep_recent: int = 1):
        """清理未使用的模型"""
        with self.lock:
            if len(self.model_cache) <= keep_recent:
                return
                
            # 按最后访问时间排序
            sorted_models = sorted(
                self.last_access.items(),
                key=lambda x: x[1]
            )
            
            # 删除最旧的模型
            models_to_remove = sorted_models[:-keep_recent]
            for model_name, _ in models_to_remove:
                if model_name in self.model_cache:
                    del self.model_cache[model_name]
                    del self.last_access[model_name]
                    
    def cache_model(self, model_name: str, model_session):
        """缓存模型"""
        with self.lock:
            self.model_cache[model_name] = model_session
            self.last_access[model_name] = time.time()
            
    def get_cached_model(self, model_name: str):
        """获取缓存的模型"""
        with self.lock:
            if model_name in self.model_cache:
                self.last_access[model_name] = time.time()
                return self.model_cache[model_name]
            return None

class LiteYOLODetector:
    """轻量级YOLO检测器"""
    
    def __init__(self, config: PlatformConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.memory_manager = EmbeddedMemoryManager(config.memory_limit_mb)
        self.session = None
        self.input_name = None
        self.output_names = None
        self.class_names = []
        self.is_initialized = False
        
        # 性能统计
        self.stats = {
            'total_inferences': 0,
            'total_time': 0.0,
            'avg_fps': 0.0,
            'memory_peaks': []
        }
        
    def initialize(self, model_path: str, class_names: List[str]):
        """初始化检测器"""
        try:
            self.class_names = class_names
            
            # 检查模型文件
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
                
            # 检查内存
            model_size_mb = os.path.getsize(model_path) / 1024 / 1024
            if not self.memory_manager.can_load_model(model_size_mb):
                self.memory_manager.cleanup_unused_models()
                if not self.memory_manager.can_load_model(model_size_mb):
                    raise RuntimeError(f"内存不足，无法加载模型 ({model_size_mb:.1f}MB)")
                    
            # 加载模型
            self._load_model(model_path)
            self.is_initialized = True
            
            self.logger.info(f"检测器初始化成功: {model_path}")
            self.logger.info(f"推理引擎: {self.config.inference_engine}")
            self.logger.info(f"输入尺寸: {self.config.input_size}")
            
        except Exception as e:
            self.logger.error(f"检测器初始化失败: {e}")
            raise
            
    def _load_model(self, model_path: str):
        """加载模型"""
        model_name = os.path.basename(model_path)
        
        # 检查缓存
        cached_session = self.memory_manager.get_cached_model(model_name)
        if cached_session is not None:
            self.session = cached_session
            self._setup_io_info()
            return
            
        # 根据推理引擎加载模型
        if self.config.inference_engine == "onnx" and ONNX_AVAILABLE:
            self._load_onnx_model(model_path)
        elif self.config.inference_engine == "tflite" and TF_AVAILABLE:
            self._load_tflite_model(model_path)
        elif self.config.inference_engine == "tensorrt" and TENSORRT_AVAILABLE:
            self._load_tensorrt_model(model_path)
        else:
            raise RuntimeError(f"不支持的推理引擎: {self.config.inference_engine}")
            
        # 缓存模型
        self.memory_manager.cache_model(model_name, self.session)
        
    def _load_onnx_model(self, model_path: str):
        """加载ONNX模型"""
        providers = []
        
        # 配置执行提供者
        if self.config.use_gpu:
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.append('CUDAExecutionProvider')
            elif 'OpenVINOExecutionProvider' in ort.get_available_providers():
                providers.append('OpenVINOExecutionProvider')
                
        providers.append('CPUExecutionProvider')
        
        # 会话选项
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self.config.cpu_threads
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        self._setup_io_info()
        
    def _load_tflite_model(self, model_path: str):
        """加载TensorFlow Lite模型"""
        self.session = tf.lite.Interpreter(
            model_path=model_path,
            num_threads=self.config.cpu_threads
        )
        self.session.allocate_tensors()
        
        # 获取输入输出信息
        input_details = self.session.get_input_details()
        output_details = self.session.get_output_details()
        
        self.input_name = input_details[0]['index']
        self.output_names = [detail['index'] for detail in output_details]
        
    def _load_tensorrt_model(self, model_path: str):
        """加载TensorRT模型"""
        # TensorRT实现需要更复杂的设置
        raise NotImplementedError("TensorRT支持正在开发中")
        
    def _setup_io_info(self):
        """设置输入输出信息(ONNX)"""
        if hasattr(self.session, 'get_inputs'):
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """预处理图像"""
        start_time = time.time()
        
        # 调整大小
        input_h, input_w = self.config.input_size
        resized = cv2.resize(image, (input_w, input_h))
        
        # 归一化
        normalized = resized.astype(np.float32) / 255.0
        
        # 转换为NCHW格式
        if self.config.inference_engine == "onnx":
            input_tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...]
        else:  # TensorFlow Lite使用NHWC
            input_tensor = normalized[np.newaxis, ...]
            
        preprocessing_time = (time.time() - start_time) * 1000
        return input_tensor, preprocessing_time
        
    def inference(self, input_tensor: np.ndarray) -> Tuple[np.ndarray, float]:
        """执行推理"""
        start_time = time.time()
        
        if self.config.inference_engine == "onnx":
            outputs = self.session.run(
                self.output_names,
                {self.input_name: input_tensor}
            )
        elif self.config.inference_engine == "tflite":
            self.session.set_tensor(self.input_name, input_tensor)
            self.session.invoke()
            outputs = [self.session.get_tensor(idx) for idx in self.output_names]
        else:
            raise RuntimeError(f"不支持的推理引擎: {self.config.inference_engine}")
            
        inference_time = (time.time() - start_time) * 1000
        return outputs, inference_time
        
    def postprocess(self, outputs: List[np.ndarray], 
                   original_shape: Tuple[int, int]) -> Tuple[DetectionResult, float]:
        """后处理输出"""
        start_time = time.time()
        
        # 解析YOLO输出
        if len(outputs) == 1:
            # YOLOv8/v11格式: [batch, 84, 8400]
            output = outputs[0][0]  # 移除batch维度
            
            # 转置为 [8400, 84]
            if output.shape[0] < output.shape[1]:
                output = output.T
                
            boxes = output[:, :4]  # x_center, y_center, width, height
            scores = output[:, 4:].max(axis=1)
            class_ids = output[:, 4:].argmax(axis=1)
            
        else:
            # 其他格式处理
            raise NotImplementedError("暂不支持此输出格式")
            
        # 过滤低置信度检测
        valid_indices = scores > self.config.confidence_threshold
        boxes = boxes[valid_indices]
        scores = scores[valid_indices]
        class_ids = class_ids[valid_indices]
        
        if len(boxes) > 0:
            # 转换坐标格式 (center_x, center_y, w, h) -> (x1, y1, x2, y2)
            boxes = self._convert_boxes(boxes, original_shape)
            
            # NMS
            keep_indices = self._nms(boxes, scores, self.config.nms_threshold)
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            class_ids = class_ids[keep_indices]
            
        # 获取类别名称
        class_names = [self.class_names[int(id)] if int(id) < len(self.class_names) 
                      else f"class_{int(id)}" for id in class_ids]
        
        postprocessing_time = (time.time() - start_time) * 1000
        
        return DetectionResult(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            class_names=class_names,
            inference_time=0,  # 将在detect方法中设置
            preprocessing_time=0,
            postprocessing_time=postprocessing_time
        ), postprocessing_time
        
    def _convert_boxes(self, boxes: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        """转换边界框坐标"""
        orig_h, orig_w = original_shape
        input_h, input_w = self.config.input_size
        
        # 缩放因子
        scale_x = orig_w / input_w
        scale_y = orig_h / input_h
        
        # 转换格式并缩放
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        
        x1 = (x_center - width / 2) * scale_x
        y1 = (y_center - height / 2) * scale_y
        x2 = (x_center + width / 2) * scale_x
        y2 = (y_center + height / 2) * scale_y
        
        return np.column_stack([x1, y1, x2, y2])
        
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
        """非极大值抑制"""
        if CV2_AVAILABLE:
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(),
                scores.tolist(),
                self.config.confidence_threshold,
                threshold
            )
            return indices.flatten() if len(indices) > 0 else np.array([])
        else:
            # 简单的NMS实现
            return self._simple_nms(boxes, scores, threshold)
            
    def _simple_nms(self, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
        """简单NMS实现"""
        indices = np.argsort(scores)[::-1]
        keep = []
        
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
                
            # 计算IoU
            current_box = boxes[current]
            other_boxes = boxes[indices[1:]]
            
            ious = self._calculate_iou(current_box, other_boxes)
            
            # 保留IoU小于阈值的框
            indices = indices[1:][ious < threshold]
            
        return np.array(keep)
        
    def _calculate_iou(self, box1: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """计算IoU"""
        x1_max = np.maximum(box1[0], boxes[:, 0])
        y1_max = np.maximum(box1[1], boxes[:, 1])
        x2_min = np.minimum(box1[2], boxes[:, 2])
        y2_min = np.minimum(box1[3], boxes[:, 3])
        
        intersection = np.maximum(0, x2_min - x1_max) * np.maximum(0, y2_min - y1_max)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
        
    def detect(self, image: np.ndarray) -> DetectionResult:
        """检测物体"""
        if not self.is_initialized:
            raise RuntimeError("检测器未初始化")
            
        original_shape = image.shape[:2]
        
        # 预处理
        input_tensor, preprocessing_time = self.preprocess(image)
        
        # 推理
        outputs, inference_time = self.inference(input_tensor)
        
        # 后处理
        result, postprocessing_time = self.postprocess(outputs, original_shape)
        
        # 设置时间信息
        result.inference_time = inference_time
        result.preprocessing_time = preprocessing_time
        result.postprocessing_time = postprocessing_time
        
        # 更新统计信息
        self._update_stats(inference_time + preprocessing_time + postprocessing_time)
        
        return result
        
    def _update_stats(self, total_time: float):
        """更新性能统计"""
        self.stats['total_inferences'] += 1
        self.stats['total_time'] += total_time
        
        if self.stats['total_inferences'] > 0:
            avg_time = self.stats['total_time'] / self.stats['total_inferences']
            self.stats['avg_fps'] = 1000.0 / avg_time if avg_time > 0 else 0
            
        # 记录内存峰值
        current_memory = self.memory_manager.get_memory_usage()
        self.stats['memory_peaks'].append(current_memory)
        
        # 只保留最近100次的内存记录
        if len(self.stats['memory_peaks']) > 100:
            self.stats['memory_peaks'] = self.stats['memory_peaks'][-100:]
            
    def get_performance_stats(self) -> Dict:
        """获取性能统计信息"""
        stats = self.stats.copy()
        if self.stats['memory_peaks']:
            stats['avg_memory_mb'] = np.mean(self.stats['memory_peaks'])
            stats['max_memory_mb'] = np.max(self.stats['memory_peaks'])
        else:
            stats['avg_memory_mb'] = 0
            stats['max_memory_mb'] = 0
            
        return stats
        
    def cleanup(self):
        """清理资源"""
        if self.session is not None:
            if hasattr(self.session, 'close'):
                self.session.close()
            self.session = None
            
        self.memory_manager.cleanup_unused_models(keep_recent=0)
        self.is_initialized = False
        
    def __del__(self):
        """析构函数"""
        self.cleanup()

# 工厂函数
def create_lite_detector(platform: str, model_size: str = "n") -> Tuple[LiteYOLODetector, PlatformConfig]:
    """创建适合特定平台的轻量级检测器"""
    
    configs = {
        "esp32_s3": PlatformConfig(
            name="ESP32-S3",
            memory_limit_mb=50,
            cpu_threads=1,
            inference_engine="tflite",
            model_format="tflite",
            precision="int8",
            input_size=(160, 160),
            confidence_threshold=0.6
        ),
        "raspberry_pi_zero": PlatformConfig(
            name="Raspberry Pi Zero 2W",
            memory_limit_mb=400,
            cpu_threads=2,
            inference_engine="onnx",
            model_format="onnx",
            precision="fp16",
            input_size=(320, 320),
            confidence_threshold=0.5
        ),
        "raspberry_pi_4": PlatformConfig(
            name="Raspberry Pi 4B",
            memory_limit_mb=3000,
            cpu_threads=4,
            use_gpu=True,
            inference_engine="onnx",
            model_format="onnx",
            precision="fp16",
            input_size=(416, 416) if model_size in ["n", "s"] else (640, 640),
            confidence_threshold=0.5
        ),
        "jetson_nano": PlatformConfig(
            name="NVIDIA Jetson Nano",
            memory_limit_mb=3500,
            cpu_threads=4,
            use_gpu=True,
            inference_engine="onnx",  # 或 tensorrt
            model_format="onnx",
            precision="fp16",
            input_size=(640, 640),
            confidence_threshold=0.5
        )
    }
    
    if platform not in configs:
        raise ValueError(f"不支持的平台: {platform}")
        
    config = configs[platform]
    detector = LiteYOLODetector(config)
    
    return detector, config

if __name__ == "__main__":
    # 示例用法
    import argparse
    
    parser = argparse.ArgumentParser(description="轻量级YOLO检测器测试")
    parser.add_argument("--platform", required=True, choices=["esp32_s3", "raspberry_pi_zero", "raspberry_pi_4", "jetson_nano"])
    parser.add_argument("--model", required=True, help="模型文件路径")
    parser.add_argument("--image", required=True, help="测试图像路径")
    parser.add_argument("--classes", help="类别文件路径")
    
    args = parser.parse_args()
    
    # 加载类别名称
    if args.classes and os.path.exists(args.classes):
        with open(args.classes, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    else:
        class_names = [f"class_{i}" for i in range(80)]  # COCO默认
        
    # 创建检测器
    detector, config = create_lite_detector(args.platform)
    
    try:
        # 初始化
        detector.initialize(args.model, class_names)
        
        # 加载图像
        if CV2_AVAILABLE:
            image = cv2.imread(args.image)
            if image is None:
                raise ValueError(f"无法加载图像: {args.image}")
        else:
            raise RuntimeError("需要安装OpenCV")
            
        # 检测
        result = detector.detect(image)
        
        # 输出结果
        print(f"检测到 {len(result.boxes)} 个物体")
        print(f"推理时间: {result.inference_time:.2f}ms")
        print(f"总时间: {result.inference_time + result.preprocessing_time + result.postprocessing_time:.2f}ms")
        
        # 性能统计
        stats = detector.get_performance_stats()
        print(f"平均FPS: {stats['avg_fps']:.2f}")
        print(f"内存使用: {stats['avg_memory_mb']:.1f}MB")
        
    finally:
        detector.cleanup()