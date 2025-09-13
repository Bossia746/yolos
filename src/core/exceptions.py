#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOS统一异常处理框架

提供标准化的异常处理机制，包括：
- 统一异常基类
- 标准错误码定义
- 异常日志记录
- 错误信息格式化

Author: YOLOS Team
Version: 1.0.0
"""

import logging
import traceback
import threading
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Union, Callable, List
import json
import sys
import os
from functools import wraps
from contextlib import contextmanager


class ErrorCode(Enum):
    """标准错误码定义"""
    
    # 系统级错误 (1000-1999)
    SYSTEM_ERROR = (1000, "系统内部错误")
    INITIALIZATION_ERROR = (1001, "系统初始化失败")
    CONFIGURATION_ERROR = (1002, "配置错误")
    DEPENDENCY_ERROR = (1003, "依赖库错误")
    RESOURCE_ERROR = (1004, "资源不足")
    PERMISSION_ERROR = (1005, "权限不足")
    
    # 模型相关错误 (2000-2999)
    MODEL_LOAD_ERROR = (2000, "模型加载失败")
    MODEL_INFERENCE_ERROR = (2001, "模型推理失败")
    MODEL_VERSION_ERROR = (2002, "模型版本不兼容")
    MODEL_FORMAT_ERROR = (2003, "模型格式错误")
    MODEL_MEMORY_ERROR = (2004, "模型内存不足")
    
    # 数据处理错误 (3000-3999)
    DATA_VALIDATION_ERROR = (3000, "数据验证失败")
    DATA_FORMAT_ERROR = (3001, "数据格式错误")
    DATA_SIZE_ERROR = (3002, "数据大小超限")
    DATA_ENCODING_ERROR = (3003, "数据编码错误")
    DATA_CORRUPTION_ERROR = (3004, "数据损坏")
    DATA_PROCESSING_ERROR = (3005, "数据处理失败")
    
    # 图像处理错误 (4000-4999)
    IMAGE_LOAD_ERROR = (4000, "图像加载失败")
    IMAGE_FORMAT_ERROR = (4001, "图像格式不支持")
    IMAGE_SIZE_ERROR = (4002, "图像尺寸错误")
    IMAGE_QUALITY_ERROR = (4003, "图像质量不足")
    IMAGE_PROCESSING_ERROR = (4004, "图像处理失败")
    
    # 检测识别错误 (5000-5999)
    DETECTION_ERROR = (5000, "目标检测失败")
    RECOGNITION_ERROR = (5001, "识别失败")
    TRACKING_ERROR = (5002, "目标跟踪失败")
    CLASSIFICATION_ERROR = (5003, "分类失败")
    POSE_ESTIMATION_ERROR = (5004, "姿态估计失败")
    GESTURE_RECOGNITION_ERROR = (5005, "手势识别失败")
    FACE_RECOGNITION_ERROR = (5006, "面部识别失败")
    FALL_DETECTION_ERROR = (5007, "摔倒检测失败")
    
    # 硬件相关错误 (6000-6999)
    CAMERA_ERROR = (6000, "摄像头错误")
    GPU_ERROR = (6001, "GPU错误")
    MEMORY_ERROR = (6002, "内存错误")
    STORAGE_ERROR = (6003, "存储错误")
    NETWORK_ERROR = (6004, "网络错误")
    
    # API相关错误 (7000-7999)
    API_ERROR = (7000, "API调用失败")
    PARAMETER_ERROR = (7001, "参数错误")
    AUTHENTICATION_ERROR = (7002, "认证失败")
    AUTHORIZATION_ERROR = (7003, "授权失败")
    RATE_LIMIT_ERROR = (7004, "请求频率超限")
    
    # 平台相关错误 (8000-8999)
    PLATFORM_ERROR = (8000, "平台不支持")
    ESP32_ERROR = (8001, "ESP32平台错误")
    K230_ERROR = (8002, "K230平台错误")
    RASPBERRY_PI_ERROR = (8003, "树莓派平台错误")
    WINDOWS_ERROR = (8004, "Windows平台错误")
    LINUX_ERROR = (8005, "Linux平台错误")
    MACOS_ERROR = (8006, "macOS平台错误")
    
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message


class YOLOSException(Exception):
    """YOLOS统一异常基类"""
    
    def __init__(
        self,
        error_code: ErrorCode,
        detail: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        recovery_suggestions: Optional[List[str]] = None,
        severity: str = "error"
    ):
        self.error_code = error_code
        self.detail = detail or error_code.message
        self.context = context or {}
        self.cause = cause
        self.recovery_suggestions = recovery_suggestions or []
        self.severity = severity  # "critical", "error", "warning", "info"
        self.timestamp = datetime.now().isoformat()
        self.thread_id = threading.get_ident()
        self.process_id = os.getpid()
        
        # 收集调用栈信息
        self.stack_trace = traceback.format_stack()[:-1]  # 排除当前帧
        
        # 收集系统信息
        self.system_info = {
            "platform": sys.platform,
            "python_version": sys.version,
            "thread_id": self.thread_id,
            "process_id": self.process_id
        }
        
        # 构建完整错误信息
        message = f"[{error_code.code}] {self.detail}"
        if context:
            message += f" | Context: {json.dumps(context, ensure_ascii=False)}"
        
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "error_code": self.error_code.code,
            "error_name": self.error_code.name,
            "message": self.error_code.message,
            "detail": self.detail,
            "context": self.context,
            "timestamp": self.timestamp,
            "severity": self.severity,
            "recovery_suggestions": self.recovery_suggestions,
            "system_info": self.system_info,
            "cause": str(self.cause) if self.cause else None,
            "cause_type": type(self.cause).__name__ if self.cause else None
        }
    
    def get_detailed_info(self) -> Dict[str, Any]:
        """获取详细异常信息，包括调用栈"""
        info = self.to_dict()
        info.update({
            "stack_trace": self.stack_trace,
            "full_traceback": traceback.format_exception(
                type(self.cause), self.cause, self.cause.__traceback__
            ) if self.cause else None
        })
        return info
    
    def get_recovery_info(self) -> Dict[str, Any]:
        """获取恢复建议信息"""
        return {
            "error_code": self.error_code.code,
            "severity": self.severity,
            "recovery_suggestions": self.recovery_suggestions,
            "context": self.context,
            "is_recoverable": len(self.recovery_suggestions) > 0
        }
    
    def add_recovery_suggestion(self, suggestion: str):
        """添加恢复建议"""
        if suggestion not in self.recovery_suggestions:
            self.recovery_suggestions.append(suggestion)
    
    def is_critical(self) -> bool:
        """判断是否为严重错误"""
        return self.severity == "critical"
    
    def is_recoverable(self) -> bool:
        """判断是否可恢复"""
        return len(self.recovery_suggestions) > 0
    
    def to_json(self) -> str:
        """转换为JSON格式"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class SystemException(YOLOSException):
    """系统级异常"""
    pass


class ModelException(YOLOSException):
    """模型相关异常"""
    pass


class ModuleError(YOLOSException):
    """模块相关异常"""
    pass


class DataException(YOLOSException):
    """数据处理异常"""
    pass


class ImageException(YOLOSException):
    """图像处理异常"""
    pass


class DetectionException(YOLOSException):
    """检测识别异常"""
    pass


class HardwareException(YOLOSException):
    """硬件相关异常"""
    pass


class APIException(YOLOSException):
    """API相关异常"""
    pass


class PlatformException(YOLOSException):
    """平台相关异常"""
    pass


class ConfigurationError(SystemException):
    """配置错误异常"""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        if config_key:
            context['config_key'] = config_key
        
        super().__init__(
            error_code=ErrorCode.CONFIGURATION_ERROR,
            detail=message,
            context=context,
            cause=kwargs.get('cause')
        )


class ConfigValidationError(ConfigurationError):
    """配置验证错误异常"""
    
    def __init__(self, message: str, config_path: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        if config_path:
            context['config_path'] = config_path
        
        super().__init__(
            message=message,
            config_key=config_path,
            **kwargs
        )


class RecoveryStrategy:
    """异常恢复策略"""
    
    def __init__(self, name: str, handler: Callable, max_attempts: int = 3):
        self.name = name
        self.handler = handler
        self.max_attempts = max_attempts
    
    def attempt_recovery(self, exception: YOLOSException, attempt: int) -> bool:
        """尝试恢复
        
        Args:
            exception: 异常对象
            attempt: 当前尝试次数
            
        Returns:
            bool: 是否恢复成功
        """
        try:
            return self.handler(exception, attempt)
        except Exception:
            return False


class ExceptionHandler:
    """增强的统一异常处理器"""
    
    def __init__(self, logger_name: str = "yolos.exceptions"):
        self.logger = logging.getLogger(logger_name)
        self._setup_logger()
        self._recovery_strategies: Dict[int, List[RecoveryStrategy]] = {}
        self._exception_stats = {
            "total_count": 0,
            "by_error_code": {},
            "by_severity": {},
            "recovery_attempts": 0,
            "recovery_successes": 0
        }
        self._stats_lock = threading.Lock()
    
    def _setup_logger(self):
        """设置日志记录器"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.ERROR)
    
    def register_recovery_strategy(
        self,
        error_code: ErrorCode,
        strategy: RecoveryStrategy
    ):
        """注册恢复策略"""
        if error_code.code not in self._recovery_strategies:
            self._recovery_strategies[error_code.code] = []
        self._recovery_strategies[error_code.code].append(strategy)
    
    def handle_exception(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
        log_level: int = logging.ERROR,
        attempt_recovery: bool = True
    ) -> YOLOSException:
        """处理异常
        
        Args:
            exception: 原始异常
            context: 异常上下文信息
            log_level: 日志级别
            attempt_recovery: 是否尝试恢复
            
        Returns:
            YOLOSException: 标准化异常对象
        """
        # 如果已经是YOLOS异常，直接处理
        if isinstance(exception, YOLOSException):
            yolos_exception = exception
        else:
            # 根据异常类型映射到标准错误码
            error_code = self._map_exception_to_error_code(exception)
            
            # 添加恢复建议
            recovery_suggestions = self._generate_recovery_suggestions(exception, error_code)
            
            # 创建标准异常
            yolos_exception = YOLOSException(
                error_code=error_code,
                detail=str(exception),
                context=context,
                cause=exception,
                recovery_suggestions=recovery_suggestions,
                severity=self._determine_severity(error_code, exception)
            )
        
        # 更新统计信息
        self._update_stats(yolos_exception)
        
        # 尝试恢复
        if attempt_recovery and yolos_exception.is_recoverable():
            recovery_success = self._attempt_recovery(yolos_exception)
            if recovery_success:
                self.logger.info(f"Successfully recovered from exception: {yolos_exception.error_code.code}")
                return yolos_exception
        
        # 记录异常日志
        self._log_exception(yolos_exception, log_level)
        
        return yolos_exception
    
    def _generate_recovery_suggestions(
        self,
        exception: Exception,
        error_code: ErrorCode
    ) -> List[str]:
        """生成恢复建议"""
        suggestions = []
        
        # 基于错误码的通用建议
        if error_code == ErrorCode.CONFIGURATION_ERROR:
            suggestions.extend([
                "检查配置文件格式是否正确",
                "验证配置文件路径是否存在",
                "确认配置项的值类型是否匹配"
            ])
        elif error_code == ErrorCode.MODEL_LOAD_ERROR:
            suggestions.extend([
                "检查模型文件是否存在",
                "验证模型文件格式是否正确",
                "确认有足够的内存加载模型"
            ])
        elif error_code == ErrorCode.MEMORY_ERROR:
            suggestions.extend([
                "释放不必要的内存占用",
                "减少批处理大小",
                "考虑使用内存映射文件"
            ])
        elif error_code == ErrorCode.NETWORK_ERROR:
            suggestions.extend([
                "检查网络连接状态",
                "验证服务器地址是否正确",
                "尝试重新连接"
            ])
        
        # 基于异常类型的建议
        if isinstance(exception, FileNotFoundError):
            suggestions.append("确认文件路径是否正确")
        elif isinstance(exception, PermissionError):
            suggestions.append("检查文件或目录的访问权限")
        elif isinstance(exception, ValueError):
            suggestions.append("验证输入参数的值是否在有效范围内")
        
        return suggestions
    
    def _determine_severity(self, error_code: ErrorCode, exception: Exception) -> str:
        """确定异常严重程度"""
        # 严重错误
        critical_codes = [
            ErrorCode.SYSTEM_ERROR,
            ErrorCode.MEMORY_ERROR,
            ErrorCode.INITIALIZATION_ERROR
        ]
        
        if error_code in critical_codes:
            return "critical"
        
        # 基于异常类型判断
        if isinstance(exception, (MemoryError, SystemError)):
            return "critical"
        elif isinstance(exception, (ValueError, TypeError, KeyError)):
            return "warning"
        
        return "error"
    
    def _attempt_recovery(self, exception: YOLOSException) -> bool:
        """尝试异常恢复"""
        strategies = self._recovery_strategies.get(exception.error_code.code, [])
        
        with self._stats_lock:
            self._exception_stats["recovery_attempts"] += 1
        
        for strategy in strategies:
            for attempt in range(strategy.max_attempts):
                try:
                    if strategy.attempt_recovery(exception, attempt + 1):
                        with self._stats_lock:
                            self._exception_stats["recovery_successes"] += 1
                        return True
                except Exception as recovery_error:
                    self.logger.warning(
                        f"Recovery strategy '{strategy.name}' failed on attempt {attempt + 1}: {recovery_error}"
                    )
        
        return False
    
    def _update_stats(self, exception: YOLOSException):
        """更新异常统计信息"""
        with self._stats_lock:
            self._exception_stats["total_count"] += 1
            
            # 按错误码统计
            code = exception.error_code.code
            if code not in self._exception_stats["by_error_code"]:
                self._exception_stats["by_error_code"][code] = 0
            self._exception_stats["by_error_code"][code] += 1
            
            # 按严重程度统计
            severity = exception.severity
            if severity not in self._exception_stats["by_severity"]:
                self._exception_stats["by_severity"][severity] = 0
            self._exception_stats["by_severity"][severity] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取异常统计信息"""
        with self._stats_lock:
            return self._exception_stats.copy()
    
    def _map_exception_to_error_code(self, exception: Exception) -> ErrorCode:
        """将异常映射到标准错误码"""
        exception_type = type(exception).__name__
        
        # 常见异常类型映射
        mapping = {
            'FileNotFoundError': ErrorCode.DATA_VALIDATION_ERROR,
            'PermissionError': ErrorCode.PERMISSION_ERROR,
            'MemoryError': ErrorCode.MEMORY_ERROR,
            'ValueError': ErrorCode.DATA_VALIDATION_ERROR,
            'TypeError': ErrorCode.DATA_FORMAT_ERROR,
            'ImportError': ErrorCode.DEPENDENCY_ERROR,
            'ModuleNotFoundError': ErrorCode.DEPENDENCY_ERROR,
            'ConnectionError': ErrorCode.NETWORK_ERROR,
            'TimeoutError': ErrorCode.NETWORK_ERROR,
            'KeyError': ErrorCode.PARAMETER_ERROR,
            'AttributeError': ErrorCode.CONFIGURATION_ERROR,
        }
        
        return mapping.get(exception_type, ErrorCode.SYSTEM_ERROR)
    
    def _log_exception(
        self,
        exception: YOLOSException,
        log_level: int = logging.ERROR
    ):
        """记录异常日志"""
        log_message = f"Exception occurred: {exception.to_json()}"
        
        if exception.cause:
            log_message += f"\nOriginal traceback:\n{traceback.format_exc()}"
        
        self.logger.log(log_level, log_message)
    
    def create_exception(
        self,
        error_code: ErrorCode,
        detail: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        exception_class: type = YOLOSException
    ) -> YOLOSException:
        """创建标准异常
        
        Args:
            error_code: 错误码
            detail: 详细信息
            context: 上下文信息
            exception_class: 异常类型
            
        Returns:
            YOLOSException: 标准异常对象
        """
        return exception_class(
            error_code=error_code,
            detail=detail,
            context=context
        )


# 全局异常处理器实例
global_exception_handler = ExceptionHandler()


def handle_exception(
    exception: Exception,
    context: Optional[Dict[str, Any]] = None,
    log_level: int = logging.ERROR,
    attempt_recovery: bool = True
) -> YOLOSException:
    """全局异常处理函数"""
    return global_exception_handler.handle_exception(
        exception, context, log_level, attempt_recovery
    )


def create_exception(
    error_code: ErrorCode,
    detail: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    exception_class: type = YOLOSException,
    recovery_suggestions: Optional[List[str]] = None,
    severity: str = "error"
) -> YOLOSException:
    """全局异常创建函数"""
    return exception_class(
        error_code=error_code,
        detail=detail,
        context=context,
        recovery_suggestions=recovery_suggestions,
        severity=severity
    )


def register_recovery_strategy(
    error_code: ErrorCode,
    strategy: RecoveryStrategy
):
    """注册全局恢复策略"""
    global_exception_handler.register_recovery_strategy(error_code, strategy)


def get_exception_stats() -> Dict[str, Any]:
    """获取全局异常统计信息"""
    return global_exception_handler.get_stats()


def get_exception_handler() -> ExceptionHandler:
    """获取全局异常处理器实例"""
    return global_exception_handler


@contextmanager
def exception_context(
    context: Optional[Dict[str, Any]] = None,
    suppress: bool = False,
    attempt_recovery: bool = True
):
    """异常处理上下文管理器
    
    Args:
        context: 异常上下文信息
        suppress: 是否抑制异常
        attempt_recovery: 是否尝试恢复
    """
    try:
        yield
    except Exception as e:
        yolos_exception = handle_exception(
            e, context, attempt_recovery=attempt_recovery
        )
        
        if not suppress:
            raise yolos_exception


# 重试装饰器
def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable] = None
):
    """重试装饰器
    
    Args:
        max_attempts: 最大重试次数
        delay: 初始延迟时间（秒）
        backoff: 退避倍数
        exceptions: 需要重试的异常类型
        on_retry: 重试时的回调函数
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:  # 不是最后一次尝试
                        if on_retry:
                            on_retry(e, attempt + 1, max_attempts)
                        
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        # 最后一次尝试失败，抛出异常
                        yolos_exception = handle_exception(
                            e,
                            {
                                'function': func.__name__,
                                'module': func.__module__,
                                'retry_attempts': max_attempts,
                                'final_attempt': True
                            }
                        )
                        raise yolos_exception
            
            # 理论上不会到达这里
            raise last_exception
        
        return wrapper
    return decorator


# 断路器模式
class CircuitBreaker:
    """断路器模式实现"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self._lock:
                if self.state == "OPEN":
                    if time.time() - self.last_failure_time < self.recovery_timeout:
                        raise YOLOSException(
                            ErrorCode.SYSTEM_ERROR,
                            "Circuit breaker is OPEN",
                            {
                                "function": func.__name__,
                                "failure_count": self.failure_count,
                                "state": self.state
                            }
                        )
                    else:
                        self.state = "HALF_OPEN"
                
                try:
                    result = func(*args, **kwargs)
                    
                    # 成功调用，重置计数器
                    if self.state == "HALF_OPEN":
                        self.state = "CLOSED"
                    self.failure_count = 0
                    
                    return result
                    
                except self.expected_exception as e:
                    self.failure_count += 1
                    self.last_failure_time = time.time()
                    
                    if self.failure_count >= self.failure_threshold:
                        self.state = "OPEN"
                    
                    raise e
        
        return wrapper


# 装饰器：自动异常处理
def exception_handler(
    error_code: Optional[ErrorCode] = None,
    context: Optional[Dict[str, Any]] = None,
    reraise: bool = True,
    attempt_recovery: bool = True
):
    """异常处理装饰器
    
    Args:
        error_code: 默认错误码
        context: 默认上下文
        reraise: 是否重新抛出异常
        attempt_recovery: 是否尝试恢复
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 构建上下文信息
                func_context = {
                    'function': func.__name__,
                    'module': func.__module__,
                    'args': str(args)[:200],  # 限制长度
                    'kwargs': str(kwargs)[:200]
                }
                if context:
                    func_context.update(context)
                
                # 处理异常
                yolos_exception = handle_exception(
                    e, func_context, attempt_recovery=attempt_recovery
                )
                
                if reraise:
                    raise yolos_exception
                else:
                    return None
        
        return wrapper
    return decorator


# 上下文管理器：异常处理
class ExceptionContext:
    """异常处理上下文管理器"""
    
    def __init__(
        self,
        context: Optional[Dict[str, Any]] = None,
        suppress: bool = False
    ):
        self.context = context or {}
        self.suppress = suppress
        self.exception = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            self.exception = handle_exception(exc_val, self.context)
            return self.suppress
        return False


if __name__ == "__main__":
    # 测试异常处理框架
    try:
        # 测试基本异常创建
        exc = create_exception(
            ErrorCode.MODEL_LOAD_ERROR,
            "测试模型加载失败",
            {"model_path": "/path/to/model", "model_type": "YOLO"}
        )
        print("创建的异常:")
        print(exc.to_json())
        
        # 测试异常处理
        try:
            raise ValueError("测试值错误")
        except Exception as e:
            handled_exc = handle_exception(e, {"test_context": "异常处理测试"})
            print("\n处理后的异常:")
            print(handled_exc.to_json())
        
        # 测试装饰器
        @exception_handler(ErrorCode.SYSTEM_ERROR)
        def test_function():
            raise RuntimeError("测试运行时错误")
        
        test_function()
        
    except YOLOSException as e:
        print("\n捕获的YOLOS异常:")
        print(e.to_json())