# YOLOS API接口设计规范

## 概述

本文档定义了YOLOS项目的API接口设计标准，旨在确保整个系统的接口一致性、可维护性和可扩展性。

## 1. 基本原则

### 1.1 一致性原则
- 所有模块必须遵循统一的命名约定
- 参数传递方式保持一致
- 返回值格式标准化
- 异常处理机制统一

### 1.2 简洁性原则
- 接口设计简洁明了
- 避免不必要的复杂性
- 参数数量合理控制
- 功能职责单一

### 1.3 可扩展性原则
- 预留扩展接口
- 支持向后兼容
- 模块化设计
- 插件化架构

## 2. 命名约定

### 2.1 类命名
```python
# 使用PascalCase（帕斯卡命名法）
class PoseRecognizer:     # ✅ 正确
class pose_recognizer:    # ❌ 错误
class poseRecognizer:     # ❌ 错误
```

### 2.2 方法命名
```python
# 使用snake_case（蛇形命名法）
def analyze_pose(self):        # ✅ 正确
def analyzePose(self):         # ❌ 错误
def AnalyzePose(self):         # ❌ 错误

# 私有方法使用单下划线前缀
def _calculate_angle(self):    # ✅ 正确
def __calculate_angle(self):   # ❌ 避免双下划线
```

### 2.3 变量命名
```python
# 使用snake_case
exercise_type = ExerciseType.PUSHUP    # ✅ 正确
exerciseType = ExerciseType.PUSHUP     # ❌ 错误

# 常量使用UPPER_CASE
DEFAULT_CONFIDENCE_THRESHOLD = 0.5     # ✅ 正确
default_confidence_threshold = 0.5     # ❌ 错误
```

### 2.4 枚举命名
```python
# 枚举类使用PascalCase，枚举值使用UPPER_CASE
class ExerciseType(Enum):
    PUSHUP = "pushup"          # ✅ 正确
    SQUAT = "squat"            # ✅ 正确
    pushup = "pushup"          # ❌ 错误
```

## 3. 接口设计标准

### 3.1 统一的基础接口

所有检测器和识别器必须继承自基础接口：

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
from ..core.types import ProcessingResult

class BaseDetector(ABC):
    """基础检测器接口"""
    
    @abstractmethod
    def detect(self, image: np.ndarray, **kwargs) -> ProcessingResult:
        """检测方法
        
        Args:
            image: 输入图像 (H, W, C)
            **kwargs: 额外参数
            
        Returns:
            ProcessingResult: 标准化处理结果
        """
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """初始化检测器
        
        Returns:
            bool: 初始化是否成功
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """清理资源"""
        pass
```

### 3.2 标准化参数传递

#### 3.2.1 必需参数
- 使用位置参数传递核心数据（如图像、模型路径）
- 必需参数不应有默认值

#### 3.2.2 可选参数
- 使用关键字参数传递配置选项
- 提供合理的默认值
- 使用类型注解

```python
def analyze_pose(
    self,
    image: np.ndarray,                    # 必需参数
    confidence_threshold: float = 0.5,    # 可选参数，有默认值
    enable_visualization: bool = False,   # 可选参数，有默认值
    **kwargs                              # 扩展参数
) -> ProcessingResult:
    """姿态分析方法"""
    pass
```

### 3.3 标准化返回值

#### 3.3.1 使用统一的结果类型
所有处理方法必须返回 `ProcessingResult` 或其子类：

```python
from ..core.types import ProcessingResult, TaskType, Status

def detect(self, image: np.ndarray) -> ProcessingResult:
    try:
        # 执行检测逻辑
        detections = self._perform_detection(image)
        
        return ProcessingResult(
            task_type=TaskType.DETECTION,
            status=Status.SUCCESS,
            detections=detections,
            processing_time=processing_time
        )
    except Exception as e:
        return ProcessingResult(
            task_type=TaskType.DETECTION,
            status=Status.FAILED,
            error_message=str(e)
        )
```

#### 3.3.2 结果字典格式
当需要返回字典时，使用标准化格式：

```python
{
    "success": bool,              # 操作是否成功
    "data": Any,                 # 主要数据
    "metadata": Dict[str, Any],  # 元数据
    "timestamp": float,          # 时间戳
    "processing_time": float,    # 处理时间
    "error": Optional[str]       # 错误信息
}
```

## 4. 异常处理标准

### 4.1 使用统一异常体系
```python
from ..core.exceptions import (
    YOLOSException, ErrorCode, 
    ModelLoadError, DataValidationError
)

def load_model(self, model_path: str):
    try:
        # 模型加载逻辑
        pass
    except FileNotFoundError:
        raise ModelLoadError(
            f"Model file not found: {model_path}",
            details={'model_path': model_path}
        )
    except Exception as e:
        raise YOLOSException(
            ErrorCode.MODEL_LOAD_ERROR,
            f"Failed to load model: {str(e)}",
            cause=e
        )
```

### 4.2 异常处理装饰器
```python
from ..core.exceptions import exception_handler, ErrorCode

@exception_handler(ErrorCode.DETECTION_ERROR)
def detect(self, image: np.ndarray):
    # 检测逻辑
    pass
```

## 5. 配置管理标准

### 5.1 配置类设计
```python
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class DetectorConfig:
    """检测器配置"""
    model_path: str
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    input_size: Tuple[int, int] = (640, 640)
    device: str = "auto"
    
    def validate(self) -> bool:
        """验证配置有效性"""
        if not (0.0 <= self.confidence_threshold <= 1.0):
            return False
        if not (0.0 <= self.nms_threshold <= 1.0):
            return False
        return True
```

### 5.2 配置加载
```python
def load_config(config_path: str) -> DetectorConfig:
    """加载配置文件"""
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        config = DetectorConfig(**config_data)
        if not config.validate():
            raise ConfigValidationError("Invalid configuration")
        
        return config
    except Exception as e:
        raise ConfigurationError(f"Failed to load config: {e}")
```

## 6. 日志记录标准

### 6.1 日志级别使用
```python
import logging

logger = logging.getLogger(__name__)

# DEBUG: 详细的调试信息
logger.debug(f"Processing image with shape: {image.shape}")

# INFO: 一般信息
logger.info("Model loaded successfully")

# WARNING: 警告信息
logger.warning("Low confidence detection: {confidence}")

# ERROR: 错误信息
logger.error(f"Failed to process image: {error}")

# CRITICAL: 严重错误
logger.critical("System initialization failed")
```

### 6.2 结构化日志
```python
logger.info(
    "Detection completed",
    extra={
        'detection_count': len(detections),
        'processing_time': processing_time,
        'confidence_avg': avg_confidence
    }
)
```

## 7. 文档标准

### 7.1 类文档
```python
class PoseRecognizer:
    """人体姿态识别器
    
    基于YOLO模型的人体关键点检测和姿态分析，
    支持多种运动类型的识别和计数。
    
    Attributes:
        model_path: 模型文件路径
        exercise_type: 运动类型
        config: 配置对象
        
    Example:
        >>> recognizer = PoseRecognizer("yolo11n-pose.pt")
        >>> result = recognizer.analyze_pose(image)
        >>> print(f"Count: {result.count}")
    """
```

### 7.2 方法文档
```python
def analyze_pose(
    self, 
    image: np.ndarray,
    confidence_threshold: float = 0.5
) -> ProcessingResult:
    """分析图像中的人体姿态
    
    Args:
        image: 输入图像，形状为 (H, W, C)，BGR格式
        confidence_threshold: 置信度阈值，范围 [0.0, 1.0]
        
    Returns:
        ProcessingResult: 包含检测结果的标准化对象
        
    Raises:
        DataValidationError: 输入图像格式无效
        ModelInferenceError: 模型推理失败
        
    Example:
        >>> image = cv2.imread("test.jpg")
        >>> result = recognizer.analyze_pose(image, 0.7)
        >>> if result.status == Status.SUCCESS:
        ...     print(f"Detected {len(result.detections)} poses")
    """
```

## 8. 测试标准

### 8.1 单元测试结构
```python
import unittest
from unittest.mock import Mock, patch
import numpy as np

class TestPoseRecognizer(unittest.TestCase):
    """姿态识别器测试类"""
    
    def setUp(self):
        """测试前置设置"""
        self.recognizer = PoseRecognizer("test_model.pt")
        self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    def tearDown(self):
        """测试后置清理"""
        self.recognizer.cleanup()
    
    def test_analyze_pose_success(self):
        """测试姿态分析成功情况"""
        result = self.recognizer.analyze_pose(self.test_image)
        self.assertEqual(result.status, Status.SUCCESS)
        self.assertIsNotNone(result.detections)
    
    def test_analyze_pose_invalid_input(self):
        """测试无效输入处理"""
        with self.assertRaises(DataValidationError):
            self.recognizer.analyze_pose(None)
```

### 8.2 集成测试
```python
class TestSystemIntegration(unittest.TestCase):
    """系统集成测试"""
    
    def test_end_to_end_workflow(self):
        """测试端到端工作流程"""
        # 初始化系统
        system = YOLOSSystem()
        
        # 加载模型
        system.load_model("yolo11n-pose.pt")
        
        # 处理图像
        result = system.process_image("test_image.jpg")
        
        # 验证结果
        self.assertEqual(result.status, Status.SUCCESS)
```

## 9. 性能标准

### 9.1 响应时间要求
- 图像检测: < 100ms (640x640)
- 视频处理: > 30 FPS (实时)
- 模型加载: < 5s

### 9.2 内存使用
- 单个模型: < 500MB
- 系统总内存: < 2GB
- 内存泄漏: 0 tolerance

### 9.3 性能监控
```python
import time
from functools import wraps

def performance_monitor(func):
    """性能监控装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.info(
            f"Performance: {func.__name__}",
            extra={
                'execution_time': end_time - start_time,
                'function': func.__name__
            }
        )
        return result
    return wrapper
```

## 10. 版本兼容性

### 10.1 语义化版本控制
- 主版本号: 不兼容的API修改
- 次版本号: 向后兼容的功能性新增
- 修订号: 向后兼容的问题修正

### 10.2 废弃警告
```python
import warnings

def deprecated_method(self):
    """已废弃的方法"""
    warnings.warn(
        "This method is deprecated and will be removed in v2.0. "
        "Use new_method() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return self.new_method()
```

## 11. 代码审查清单

### 11.1 接口设计检查
- [ ] 命名约定符合规范
- [ ] 参数类型注解完整
- [ ] 返回值类型明确
- [ ] 异常处理完善
- [ ] 文档字符串完整

### 11.2 实现质量检查
- [ ] 单元测试覆盖率 > 80%
- [ ] 性能要求满足
- [ ] 内存使用合理
- [ ] 日志记录适当
- [ ] 错误处理健壮

## 12. 示例实现

### 12.1 标准检测器实现
```python
from typing import Optional
import numpy as np
from ..core.types import ProcessingResult, TaskType, Status
from ..core.exceptions import exception_handler, ErrorCode
from .base_detector import BaseDetector

class StandardDetector(BaseDetector):
    """标准检测器实现示例"""
    
    def __init__(self, model_path: str, config: Optional[DetectorConfig] = None):
        self.model_path = model_path
        self.config = config or DetectorConfig(model_path)
        self.model = None
        self._initialized = False
    
    @exception_handler(ErrorCode.INITIALIZATION_ERROR)
    def initialize(self) -> bool:
        """初始化检测器"""
        if self._initialized:
            return True
            
        # 加载模型逻辑
        self.model = self._load_model()
        self._initialized = True
        return True
    
    @exception_handler(ErrorCode.DETECTION_ERROR)
    def detect(self, image: np.ndarray, **kwargs) -> ProcessingResult:
        """执行检测"""
        if not self._initialized:
            self.initialize()
        
        # 验证输入
        self._validate_input(image)
        
        # 执行检测
        start_time = time.time()
        detections = self._perform_detection(image, **kwargs)
        processing_time = time.time() - start_time
        
        return ProcessingResult(
            task_type=TaskType.DETECTION,
            status=Status.SUCCESS,
            detections=detections,
            processing_time=processing_time
        )
    
    def cleanup(self) -> None:
        """清理资源"""
        if self.model:
            del self.model
            self.model = None
        self._initialized = False
```

## 总结

本规范定义了YOLOS项目的API接口设计标准，涵盖了命名约定、接口设计、异常处理、配置管理、日志记录、文档编写、测试标准等各个方面。

遵循这些标准将确保：
- 代码的一致性和可维护性
- 模块间的良好协作
- 系统的稳定性和可扩展性
- 开发效率的提升

所有开发者在编写代码时都应严格遵循本规范，并在代码审查时进行检查。