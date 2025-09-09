#!/usr/bin/env python3
"""
YOLO优化核心检测器
支持多平台高效运行的统一YOLO检测接口
"""

import os
import sys
import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import logging

# 平台检测
PLATFORM = sys.platform
IS_WINDOWS = PLATFORM == 'win32'
IS_LINUX = PLATFORM.startswith('linux')
IS_MACOS = PLATFORM == 'darwin'

# 尝试导入可选依赖
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    import torch
    import torchvision
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow.lite as tflite
        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False

class DetectionResult:
    """检测结果类"""
    
    def __init__(self, class_id: int, class_name: str, confidence: float, 
                 bbox: Tuple[int, int, int, int]):
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2)
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox': self.bbox
        }
        
    def __str__(self) -> str:
        return f"{self.class_name}({self.confidence:.2f}): {self.bbox}"

class YOLOOptimizedCore:
    """YOLO优化核心检测器"""
    
    def __init__(self, model_path: Optional[str] = None, 
                 device: str = 'auto',
                 input_size: Tuple[int, int] = (640, 640),
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4):
        """
        初始化YOLO检测器
        
        Args:
            model_path: 模型文件路径
            device: 运行设备 ('cpu', 'cuda', 'auto')
            input_size: 输入尺寸 (width, height)
            confidence_threshold: 置信度阈值
            nms_threshold: NMS阈值
        """
        self.model_path = model_path
        self.device = self._determine_device(device)
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # 模型相关
        self.model = None
        self.model_type = None
        self.session = None
        self.interpreter = None
        
        # 类别名称
        self.class_names = self._get_default_class_names()
        
        # 性能统计
        self.inference_times = []
        self.preprocess_times = []
        self.postprocess_times = []
        
        # 日志
        self.logger = self._setup_logger()
        
        # 自动加载模型
        if model_path:
            self.load_model(model_path)
            
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('YOLOOptimizedCore')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def _determine_device(self, device: str) -> str:
        """确定运行设备"""
        if device == 'auto':
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        return device
        
    def _get_default_class_names(self) -> List[str]:
        """获取默认类别名称(COCO数据集)"""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
    def load_model(self, model_path: str) -> bool:
        """
        加载模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            bool: 加载是否成功
        """
        if not os.path.exists(model_path):
            self.logger.error(f"模型文件不存在: {model_path}")
            return False
            
        self.model_path = model_path
        file_ext = Path(model_path).suffix.lower()
        
        try:
            if file_ext == '.pt' and TORCH_AVAILABLE:
                return self._load_pytorch_model(model_path)
            elif file_ext == '.onnx' and ONNX_AVAILABLE:
                return self._load_onnx_model(model_path)
            elif file_ext == '.tflite' and TFLITE_AVAILABLE:
                return self._load_tflite_model(model_path)
            else:
                self.logger.error(f"不支持的模型格式: {file_ext}")
                return False
                
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            return False
            
    def _load_pytorch_model(self, model_path: str) -> bool:
        """加载PyTorch模型"""
        try:
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            self.model_type = 'pytorch'
            self.logger.info(f"PyTorch模型加载成功: {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"PyTorch模型加载失败: {e}")
            return False
            
    def _load_onnx_model(self, model_path: str) -> bool:
        """加载ONNX模型"""
        try:
            providers = ['CPUExecutionProvider']
            if self.device == 'cuda':
                providers.insert(0, 'CUDAExecutionProvider')
                
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.model_type = 'onnx'
            self.logger.info(f"ONNX模型加载成功: {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"ONNX模型加载失败: {e}")
            return False
            
    def _load_tflite_model(self, model_path: str) -> bool:
        """加载TFLite模型"""
        try:
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.model_type = 'tflite'
            self.logger.info(f"TFLite模型加载成功: {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"TFLite模型加载失败: {e}")
            return False
            
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理
        
        Args:
            image: 输入图像 (H, W, C)
            
        Returns:
            np.ndarray: 预处理后的图像
        """
        start_time = time.time()
        
        # 调整尺寸
        if image.shape[:2] != self.input_size[::-1]:  # (H, W) vs (W, H)
            if OPENCV_AVAILABLE:
                image = cv2.resize(image, self.input_size)
            else:
                # 简单的最近邻插值
                image = self._simple_resize(image, self.input_size)
        
        # 归一化
        image = image.astype(np.float32) / 255.0
        
        # 转换为模型输入格式
        if self.model_type in ['pytorch', 'onnx']:
            # (H, W, C) -> (1, C, H, W)
            image = np.transpose(image, (2, 0, 1))
            image = np.expand_dims(image, axis=0)
        elif self.model_type == 'tflite':
            # (H, W, C) -> (1, H, W, C)
            image = np.expand_dims(image, axis=0)
            
        self.preprocess_times.append(time.time() - start_time)
        return image
        
    def _simple_resize(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """简单的图像缩放(最近邻插值)"""
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # 计算缩放比例
        scale_x = w / target_w
        scale_y = h / target_h
        
        # 创建输出图像
        if len(image.shape) == 3:
            resized = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
        else:
            resized = np.zeros((target_h, target_w), dtype=image.dtype)
            
        # 最近邻插值
        for y in range(target_h):
            for x in range(target_w):
                src_x = int(x * scale_x)
                src_y = int(y * scale_y)
                src_x = min(src_x, w - 1)
                src_y = min(src_y, h - 1)
                resized[y, x] = image[src_y, src_x]
                
        return resized
        
    def inference(self, preprocessed_image: np.ndarray) -> np.ndarray:
        """
        模型推理
        
        Args:
            preprocessed_image: 预处理后的图像
            
        Returns:
            np.ndarray: 推理结果
        """
        start_time = time.time()
        
        try:
            if self.model_type == 'pytorch':
                with torch.no_grad():
                    input_tensor = torch.from_numpy(preprocessed_image).to(self.device)
                    output = self.model(input_tensor)
                    if isinstance(output, (list, tuple)):
                        output = output[0]
                    result = output.cpu().numpy()
                    
            elif self.model_type == 'onnx':
                input_name = self.session.get_inputs()[0].name
                result = self.session.run(None, {input_name: preprocessed_image})[0]
                
            elif self.model_type == 'tflite':
                input_details = self.interpreter.get_input_details()
                output_details = self.interpreter.get_output_details()
                
                self.interpreter.set_tensor(input_details[0]['index'], preprocessed_image)
                self.interpreter.invoke()
                result = self.interpreter.get_tensor(output_details[0]['index'])
                
            else:
                # 模拟推理结果
                batch_size = preprocessed_image.shape[0]
                result = np.random.random((batch_size, 25200, 85)).astype(np.float32)
                
        except Exception as e:
            self.logger.error(f"推理失败: {e}")
            # 返回空结果
            batch_size = preprocessed_image.shape[0]
            result = np.zeros((batch_size, 25200, 85), dtype=np.float32)
            
        self.inference_times.append(time.time() - start_time)
        return result
        
    def postprocess_detections(self, predictions: np.ndarray, 
                             original_shape: Tuple[int, int]) -> List[DetectionResult]:
        """
        后处理检测结果
        
        Args:
            predictions: 模型预测结果
            original_shape: 原始图像尺寸 (H, W)
            
        Returns:
            List[DetectionResult]: 检测结果列表
        """
        start_time = time.time()
        
        detections = []
        
        try:
            # 假设输出格式为 (batch, num_boxes, 85) 其中85 = 4(bbox) + 1(conf) + 80(classes)
            if len(predictions.shape) == 3:
                predictions = predictions[0]  # 取第一个batch
                
            # 过滤低置信度检测
            confidences = predictions[:, 4]
            valid_indices = confidences > self.confidence_threshold
            valid_predictions = predictions[valid_indices]
            
            if len(valid_predictions) == 0:
                self.postprocess_times.append(time.time() - start_time)
                return detections
                
            # 解析检测结果
            boxes = valid_predictions[:, :4]  # x_center, y_center, width, height
            confidences = valid_predictions[:, 4]
            class_scores = valid_predictions[:, 5:]
            
            # 获取最高分类别
            class_ids = np.argmax(class_scores, axis=1)
            class_confidences = np.max(class_scores, axis=1)
            
            # 计算最终置信度
            final_confidences = confidences * class_confidences
            
            # 转换边界框格式 (center_x, center_y, w, h) -> (x1, y1, x2, y2)
            x_centers, y_centers, widths, heights = boxes.T
            x1 = x_centers - widths / 2
            y1 = y_centers - heights / 2
            x2 = x_centers + widths / 2
            y2 = y_centers + heights / 2
            
            # 缩放到原始图像尺寸
            orig_h, orig_w = original_shape
            scale_x = orig_w / self.input_size[0]
            scale_y = orig_h / self.input_size[1]
            
            x1 = (x1 * scale_x).astype(int)
            y1 = (y1 * scale_y).astype(int)
            x2 = (x2 * scale_x).astype(int)
            y2 = (y2 * scale_y).astype(int)
            
            # 应用NMS
            if OPENCV_AVAILABLE:
                indices = cv2.dnn.NMSBoxes(
                    [(int(x1[i]), int(y1[i]), int(x2[i]-x1[i]), int(y2[i]-y1[i])) 
                     for i in range(len(x1))],
                    final_confidences.tolist(),
                    self.confidence_threshold,
                    self.nms_threshold
                )
                
                if len(indices) > 0:
                    indices = indices.flatten()
                else:
                    indices = []
            else:
                # 简单的NMS实现
                indices = self._simple_nms(
                    np.column_stack([x1, y1, x2, y2]),
                    final_confidences,
                    self.nms_threshold
                )
            
            # 创建检测结果
            for i in indices:
                class_id = class_ids[i]
                class_name = (self.class_names[class_id] 
                            if class_id < len(self.class_names) 
                            else f'class_{class_id}')
                confidence = final_confidences[i]
                bbox = (int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i]))
                
                detections.append(DetectionResult(
                    class_id=int(class_id),
                    class_name=class_name,
                    confidence=float(confidence),
                    bbox=bbox
                ))
                
        except Exception as e:
            self.logger.error(f"后处理失败: {e}")
            
        self.postprocess_times.append(time.time() - start_time)
        return detections
        
    def _simple_nms(self, boxes: np.ndarray, scores: np.ndarray, 
                   threshold: float) -> List[int]:
        """简单的NMS实现"""
        if len(boxes) == 0:
            return []
            
        # 计算面积
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # 按分数排序
        order = scores.argsort()[::-1]
        
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
                
            # 计算IoU
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h
            
            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / (union + 1e-6)
            
            # 保留IoU小于阈值的框
            indices = np.where(iou <= threshold)[0]
            order = order[indices + 1]
            
        return keep
        
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """
        检测图像中的目标
        
        Args:
            image: 输入图像 (H, W, C)
            
        Returns:
            List[DetectionResult]: 检测结果列表
        """
        if self.model is None and self.session is None and self.interpreter is None:
            self.logger.warning("模型未加载，返回模拟结果")
            return self._generate_mock_detections(image.shape[:2])
            
        # 预处理
        preprocessed = self.preprocess_image(image)
        
        # 推理
        predictions = self.inference(preprocessed)
        
        # 后处理
        detections = self.postprocess_detections(predictions, image.shape[:2])
        
        return detections
        
    def _generate_mock_detections(self, image_shape: Tuple[int, int]) -> List[DetectionResult]:
        """生成模拟检测结果"""
        h, w = image_shape
        
        # 生成1-3个随机检测结果
        num_detections = np.random.randint(1, 4)
        detections = []
        
        for i in range(num_detections):
            # 随机类别
            class_id = np.random.randint(0, len(self.class_names))
            class_name = self.class_names[class_id]
            
            # 随机置信度
            confidence = np.random.uniform(0.5, 0.95)
            
            # 随机边界框
            x1 = np.random.randint(0, w // 2)
            y1 = np.random.randint(0, h // 2)
            x2 = np.random.randint(x1 + 50, min(w, x1 + 200))
            y2 = np.random.randint(y1 + 50, min(h, y1 + 200))
            
            detections.append(DetectionResult(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                bbox=(x1, y1, x2, y2)
            ))
            
        return detections
        
    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计"""
        stats = {}
        
        if self.preprocess_times:
            stats['avg_preprocess_time'] = np.mean(self.preprocess_times)
            stats['max_preprocess_time'] = np.max(self.preprocess_times)
            
        if self.inference_times:
            stats['avg_inference_time'] = np.mean(self.inference_times)
            stats['max_inference_time'] = np.max(self.inference_times)
            
        if self.postprocess_times:
            stats['avg_postprocess_time'] = np.mean(self.postprocess_times)
            stats['max_postprocess_time'] = np.max(self.postprocess_times)
            
        total_times = []
        for i in range(min(len(self.preprocess_times), 
                          len(self.inference_times), 
                          len(self.postprocess_times))):
            total_time = (self.preprocess_times[i] + 
                         self.inference_times[i] + 
                         self.postprocess_times[i])
            total_times.append(total_time)
            
        if total_times:
            stats['avg_total_time'] = np.mean(total_times)
            stats['avg_fps'] = 1.0 / np.mean(total_times)
            
        return stats
        
    def reset_performance_stats(self):
        """重置性能统计"""
        self.preprocess_times.clear()
        self.inference_times.clear()
        self.postprocess_times.clear()
        
    def set_confidence_threshold(self, threshold: float):
        """设置置信度阈值"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        
    def set_nms_threshold(self, threshold: float):
        """设置NMS阈值"""
        self.nms_threshold = max(0.0, min(1.0, threshold))
        
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_path': self.model_path,
            'model_type': self.model_type,
            'device': self.device,
            'input_size': self.input_size,
            'confidence_threshold': self.confidence_threshold,
            'nms_threshold': self.nms_threshold,
            'num_classes': len(self.class_names),
            'opencv_available': OPENCV_AVAILABLE,
            'torch_available': TORCH_AVAILABLE,
            'onnx_available': ONNX_AVAILABLE,
            'tflite_available': TFLITE_AVAILABLE
        }

def main():
    """测试函数"""
    print("🧪 测试YOLO优化核心检测器...")
    
    # 创建检测器
    detector = YOLOOptimizedCore()
    
    # 显示模型信息
    info = detector.get_model_info()
    print("\n📊 模型信息:")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print(f"\n🖼️ 测试图像尺寸: {test_image.shape}")
    
    # 执行检测
    print("\n🎯 执行检测...")
    detections = detector.detect(test_image)
    
    print(f"检测到 {len(detections)} 个目标:")
    for det in detections:
        print(f"   - {det}")
    
    # 显示性能统计
    stats = detector.get_performance_stats()
    print(f"\n⚡ 性能统计:")
    for key, value in stats.items():
        if 'time' in key:
            print(f"   {key}: {value*1000:.2f}ms")
        elif 'fps' in key:
            print(f"   {key}: {value:.1f}")
    
    print("\n✅ 测试完成")

if __name__ == "__main__":
    main()