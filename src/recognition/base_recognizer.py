#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础识别器接口
定义所有识别器必须实现的标准接口
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np
from dataclasses import dataclass
from enum import Enum

# 安全导入核心模块
try:
    from ..core.types import ProcessingResult, TaskType, Status
except ImportError:
    # 提供默认实现
    from enum import Enum
    from dataclasses import dataclass
    from typing import Any, Dict, List, Optional
    
    class TaskType(Enum):
        DETECTION = "detection"
        RECOGNITION = "recognition"
        TRACKING = "tracking"
        CLASSIFICATION = "classification"
    
    class Status(Enum):
        SUCCESS = "success"
        FAILED = "failed"
        PENDING = "pending"
        PROCESSING = "processing"
    
    @dataclass
    class ProcessingResult:
        task_type: TaskType
        status: Status
        processing_time: float = 0.0
        error_message: str = ""
        metadata: Dict[str, Any] = None

try:
    from ..core.exceptions import YOLOSException, ErrorCode, exception_handler
except ImportError:
    # 提供默认异常实现
    from enum import Enum
    
    class ErrorCode(Enum):
        CONFIGURATION_ERROR = (1002, "配置错误")
        DATA_VALIDATION_ERROR = (3000, "数据验证失败")
        DATA_PROCESSING_ERROR = (3005, "数据处理失败")
        RECOGNITION_ERROR = (5001, "识别失败")
        
        def __init__(self, code: int, message: str):
            self.code = code
            self.message = message
    
    class YOLOSException(Exception):
        def __init__(self, error_code: ErrorCode, detail: str = None, context: Dict[str, Any] = None, cause: Exception = None):
            self.error_code = error_code
            self.detail = detail or error_code.message
            self.context = context or {}
            self.cause = cause
            super().__init__(self.detail)
    
    def exception_handler(func):
        """简单的异常处理装饰器"""
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                raise YOLOSException(ErrorCode.RECOGNITION_ERROR, str(e), cause=e)
        return wrapper


class RecognizerType(Enum):
    """识别器类型枚举"""
    POSE = "pose"
    GESTURE = "gesture"
    FACE = "face"
    OBJECT = "object"
    CUSTOM = "custom"


@dataclass
class RecognizerConfig:
    """识别器基础配置"""
    model_path: str
    confidence_threshold: float = 0.5
    device: str = "auto"
    enable_visualization: bool = True
    batch_size: int = 1
    
    def validate(self) -> bool:
        """验证配置有效性"""
        if not (0.0 <= self.confidence_threshold <= 1.0):
            return False
        if self.batch_size <= 0:
            return False
        return True


class BaseRecognizer(ABC):
    """基础识别器抽象类
    
    所有识别器都必须继承此类并实现抽象方法。
    提供统一的接口标准和基础功能。
    
    Attributes:
        config: 识别器配置
        recognizer_type: 识别器类型
        model: 加载的模型对象
        initialized: 是否已初始化
    """
    
    def __init__(self, config: RecognizerConfig, recognizer_type: RecognizerType):
        """初始化基础识别器
        
        Args:
            config: 识别器配置
            recognizer_type: 识别器类型
        """
        if not config.validate():
            raise YOLOSException(
                ErrorCode.CONFIGURATION_ERROR,
                "Invalid recognizer configuration"
            )
        
        self.config = config
        self.recognizer_type = recognizer_type
        self.model = None
        self.initialized = False
        self._processing_stats = {
            'total_frames': 0,
            'successful_frames': 0,
            'failed_frames': 0,
            'total_processing_time': 0.0
        }
    
    @abstractmethod
    def initialize(self) -> bool:
        """初始化识别器
        
        加载模型、设置参数等初始化操作。
        
        Returns:
            bool: 初始化是否成功
            
        Raises:
            YOLOSException: 初始化失败时
        """
        pass
    
    @abstractmethod
    def recognize(self, image: np.ndarray, **kwargs) -> ProcessingResult:
        """执行识别
        
        Args:
            image: 输入图像，形状为 (H, W, C)，BGR格式
            **kwargs: 额外参数
            
        Returns:
            ProcessingResult: 标准化识别结果
            
        Raises:
            YOLOSException: 识别失败时
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """清理资源
        
        释放模型、清理缓存等清理操作。
        """
        pass
    
    def validate_input(self, image: np.ndarray) -> bool:
        """验证输入图像
        
        Args:
            image: 输入图像
            
        Returns:
            bool: 输入是否有效
            
        Raises:
            YOLOSException: 输入无效时
        """
        if image is None:
            raise YOLOSException(
                ErrorCode.DATA_VALIDATION_ERROR,
                "Input image is None"
            )
        
        if not isinstance(image, np.ndarray):
            raise YOLOSException(
                ErrorCode.DATA_VALIDATION_ERROR,
                f"Input must be numpy array, got {type(image)}"
            )
        
        if len(image.shape) != 3:
            raise YOLOSException(
                ErrorCode.DATA_VALIDATION_ERROR,
                f"Input image must be 3D (H, W, C), got shape {image.shape}"
            )
        
        if image.shape[2] != 3:
            raise YOLOSException(
                ErrorCode.DATA_VALIDATION_ERROR,
                f"Input image must have 3 channels, got {image.shape[2]}"
            )
        
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """获取识别器信息
        
        Returns:
            Dict[str, Any]: 识别器信息
        """
        return {
            'type': self.recognizer_type.value,
            'model_path': self.config.model_path,
            'confidence_threshold': self.config.confidence_threshold,
            'device': self.config.device,
            'initialized': self.initialized,
            'processing_stats': self._processing_stats.copy()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取处理统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        stats = self._processing_stats.copy()
        if stats['total_frames'] > 0:
            stats['success_rate'] = stats['successful_frames'] / stats['total_frames']
            stats['avg_processing_time'] = stats['total_processing_time'] / stats['total_frames']
        else:
            stats['success_rate'] = 0.0
            stats['avg_processing_time'] = 0.0
        
        return stats
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self._processing_stats = {
            'total_frames': 0,
            'successful_frames': 0,
            'failed_frames': 0,
            'total_processing_time': 0.0
        }
    
    def _update_stats(self, success: bool, processing_time: float) -> None:
        """更新统计信息
        
        Args:
            success: 处理是否成功
            processing_time: 处理时间
        """
        self._processing_stats['total_frames'] += 1
        self._processing_stats['total_processing_time'] += processing_time
        
        if success:
            self._processing_stats['successful_frames'] += 1
        else:
            self._processing_stats['failed_frames'] += 1
    
    def __enter__(self):
        """上下文管理器入口"""
        if not self.initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()
    
    def __del__(self):
        """析构函数"""
        try:
            self.cleanup()
        except:
            pass  # 忽略析构时的异常


class BatchRecognizer(BaseRecognizer):
    """批处理识别器基类
    
    支持批量处理多张图像的识别器。
    """
    
    @abstractmethod
    def recognize_batch(self, images: List[np.ndarray], **kwargs) -> List[ProcessingResult]:
        """批量识别
        
        Args:
            images: 输入图像列表
            **kwargs: 额外参数
            
        Returns:
            List[ProcessingResult]: 识别结果列表
            
        Raises:
            YOLOSException: 批量识别失败时
        """
        pass
    
    def validate_batch_input(self, images: List[np.ndarray]) -> bool:
        """验证批量输入
        
        Args:
            images: 输入图像列表
            
        Returns:
            bool: 输入是否有效
            
        Raises:
            YOLOSException: 输入无效时
        """
        if not images:
            raise YOLOSException(
                ErrorCode.DATA_VALIDATION_ERROR,
                "Input image list is empty"
            )
        
        if len(images) > self.config.batch_size:
            raise YOLOSException(
                ErrorCode.DATA_VALIDATION_ERROR,
                f"Batch size {len(images)} exceeds maximum {self.config.batch_size}"
            )
        
        for i, image in enumerate(images):
            try:
                self.validate_input(image)
            except YOLOSException as e:
                raise YOLOSException(
                    ErrorCode.DATA_VALIDATION_ERROR,
                    f"Invalid image at index {i}: {e.message}",
                    cause=e
                )
        
        return True


# 便捷函数
def create_recognizer_config(
    model_path: str,
    confidence_threshold: float = 0.5,
    device: str = "auto",
    enable_visualization: bool = True,
    batch_size: int = 1
) -> RecognizerConfig:
    """创建识别器配置
    
    Args:
        model_path: 模型文件路径
        confidence_threshold: 置信度阈值
        device: 设备类型
        enable_visualization: 是否启用可视化
        batch_size: 批处理大小
        
    Returns:
        RecognizerConfig: 识别器配置对象
    """
    return RecognizerConfig(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        device=device,
        enable_visualization=enable_visualization,
        batch_size=batch_size
    )