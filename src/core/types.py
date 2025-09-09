#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOS核心类型定义
统一的数据类型和结果类型
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import numpy as np

# ============================================================================
# 基础枚举类型
# ============================================================================

class TaskType(Enum):
    """任务类型"""
    DETECTION = "detection"
    RECOGNITION = "recognition"
    TRACKING = "tracking"
    CLASSIFICATION = "classification"

class ObjectType(Enum):
    """目标类型"""
    PERSON = "person"
    PET = "pet"
    VEHICLE = "vehicle"
    STATIC_OBJECT = "static_object"
    DYNAMIC_OBJECT = "dynamic_object"
    UNKNOWN = "unknown"

class Priority(Enum):
    """优先级"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class Status(Enum):
    """状态"""
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"
    PROCESSING = "processing"

# ============================================================================
# 基础数据类型
# ============================================================================

@dataclass
class BoundingBox:
    """边界框"""
    x: int
    y: int
    width: int
    height: int
    confidence: float = 0.0
    
    @property
    def x2(self) -> int:
        return self.x + self.width
    
    @property
    def y2(self) -> int:
        return self.y + self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def area(self) -> int:
        return self.width * self.height

@dataclass
class Point2D:
    """2D点"""
    x: float
    y: float
    confidence: float = 1.0

@dataclass
class Keypoint:
    """关键点"""
    point: Point2D
    label: str
    visible: bool = True

@dataclass
class ImageInfo:
    """图像信息"""
    width: int
    height: int
    channels: int
    format: str = "BGR"
    source: str = "unknown"
    timestamp: float = 0.0

# ============================================================================
# 结果类型
# ============================================================================

@dataclass
class DetectionResult:
    """检测结果"""
    bbox: BoundingBox
    class_id: int
    class_name: str
    confidence: float
    object_type: ObjectType = ObjectType.UNKNOWN
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'bbox': {
                'x': self.bbox.x,
                'y': self.bbox.y,
                'width': self.bbox.width,
                'height': self.bbox.height
            },
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'object_type': self.object_type.value
        }

@dataclass
class RecognitionResult:
    """识别结果"""
    detection: DetectionResult
    features: Dict[str, Any]
    attributes: Dict[str, Any]
    keypoints: List[Keypoint] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'detection': self.detection.to_dict(),
            'features': self.features,
            'attributes': self.attributes
        }
        
        if self.keypoints:
            result['keypoints'] = [
                {
                    'point': {'x': kp.point.x, 'y': kp.point.y},
                    'label': kp.label,
                    'confidence': kp.point.confidence,
                    'visible': kp.visible
                }
                for kp in self.keypoints
            ]
        
        return result

@dataclass
class TrackingResult:
    """跟踪结果"""
    track_id: int
    recognition: RecognitionResult
    trajectory: List[Point2D]
    velocity: Tuple[float, float] = (0.0, 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'track_id': self.track_id,
            'recognition': self.recognition.to_dict(),
            'trajectory': [
                {'x': p.x, 'y': p.y, 'confidence': p.confidence}
                for p in self.trajectory
            ],
            'velocity': {'vx': self.velocity[0], 'vy': self.velocity[1]}
        }

@dataclass
class ProcessingResult:
    """处理结果"""
    task_type: TaskType
    status: Status
    detections: List[DetectionResult] = None
    recognitions: List[RecognitionResult] = None
    trackings: List[TrackingResult] = None
    image_info: ImageInfo = None
    processing_time: float = 0.0
    error_message: str = ""
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'task_type': self.task_type.value,
            'status': self.status.value,
            'processing_time': self.processing_time,
            'error_message': self.error_message
        }
        
        if self.detections:
            result['detections'] = [d.to_dict() for d in self.detections]
        
        if self.recognitions:
            result['recognitions'] = [r.to_dict() for r in self.recognitions]
        
        if self.trackings:
            result['trackings'] = [t.to_dict() for t in self.trackings]
        
        if self.image_info:
            result['image_info'] = {
                'width': self.image_info.width,
                'height': self.image_info.height,
                'channels': self.image_info.channels,
                'format': self.image_info.format,
                'source': self.image_info.source,
                'timestamp': self.image_info.timestamp
            }
        
        if self.metadata:
            result['metadata'] = self.metadata
        
        return result

# ============================================================================
# 配置类型
# ============================================================================

@dataclass
class CameraConfig:
    """相机配置"""
    device_id: Union[int, str] = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    format: str = "BGR"
    auto_exposure: bool = True
    exposure: int = -1
    gain: int = -1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'device_id': self.device_id,
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'format': self.format,
            'auto_exposure': self.auto_exposure,
            'exposure': self.exposure,
            'gain': self.gain
        }

@dataclass
class ModelConfig:
    """模型配置"""
    model_path: str
    model_type: str
    device: str = "auto"
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    input_size: Tuple[int, int] = (640, 640)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'model_path': self.model_path,
            'model_type': self.model_type,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'nms_threshold': self.nms_threshold,
            'input_size': self.input_size
        }

@dataclass
class TrainingConfig:
    """训练配置"""
    dataset_path: str
    output_path: str
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.001
    device: str = "auto"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'dataset_path': self.dataset_path,
            'output_path': self.output_path,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'device': self.device
        }

# ============================================================================
# 工具函数
# ============================================================================

def create_detection_result(
    bbox: Tuple[int, int, int, int],
    class_id: int,
    class_name: str,
    confidence: float,
    object_type: ObjectType = ObjectType.UNKNOWN
) -> DetectionResult:
    """创建检测结果"""
    return DetectionResult(
        bbox=BoundingBox(bbox[0], bbox[1], bbox[2], bbox[3], confidence),
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        object_type=object_type
    )

def create_image_info(
    width: int,
    height: int,
    channels: int = 3,
    format: str = "BGR",
    source: str = "camera",
    timestamp: float = 0.0
) -> ImageInfo:
    """创建图像信息"""
    return ImageInfo(
        width=width,
        height=height,
        channels=channels,
        format=format,
        source=source,
        timestamp=timestamp
    )

def merge_results(results: List[ProcessingResult]) -> ProcessingResult:
    """合并多个处理结果"""
    if not results:
        return ProcessingResult(
            task_type=TaskType.DETECTION,
            status=Status.FAILED,
            error_message="No results to merge"
        )
    
    merged = ProcessingResult(
        task_type=results[0].task_type,
        status=Status.SUCCESS,
        detections=[],
        recognitions=[],
        trackings=[],
        processing_time=sum(r.processing_time for r in results),
        metadata={}
    )
    
    for result in results:
        if result.detections:
            merged.detections.extend(result.detections)
        if result.recognitions:
            merged.recognitions.extend(result.recognitions)
        if result.trackings:
            merged.trackings.extend(result.trackings)
        if result.metadata:
            merged.metadata.update(result.metadata)
    
    return merged

# ============================================================================
# 类型别名
# ============================================================================

# 兼容性别名
Detection = DetectionResult
Recognition = RecognitionResult
Tracking = TrackingResult
Result = ProcessingResult

# 数组类型
DetectionArray = List[DetectionResult]
RecognitionArray = List[RecognitionResult]
TrackingArray = List[TrackingResult]

if __name__ == "__main__":
    # 测试代码
    print("🧪 测试YOLOS核心类型...")
    
    # 创建检测结果
    detection = create_detection_result(
        bbox=(100, 100, 200, 150),
        class_id=0,
        class_name="person",
        confidence=0.95,
        object_type=ObjectType.PERSON
    )
    
    print(f"检测结果: {detection.class_name}, 置信度: {detection.confidence}")
    print(f"边界框中心: {detection.bbox.center}")
    print(f"字典格式: {detection.to_dict()}")
    
    # 创建图像信息
    img_info = create_image_info(640, 480, source="test_camera")
    print(f"图像信息: {img_info.width}x{img_info.height}")
    
    print("✅ 类型系统测试完成")