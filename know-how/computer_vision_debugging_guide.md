# 计算机视觉常见错误调试指南

## 概述
本文档整理了计算机视觉项目开发中的常见错误、调试技巧和解决方案，涵盖模型加载、数据处理、性能优化、部署等各个环节的问题排查方法。

## 模型加载与初始化错误

### 1. 模型文件路径错误

#### 常见错误信息
```
FileNotFoundError: [Errno 2] No such file or directory: 'models/buffalo_l.onnx'
OSError: Unable to load model from path
PermissionError: [Errno 13] Permission denied
```

#### 问题原因
- 模型文件路径不正确
- 文件不存在或下载不完整
- 权限不足
- 相对路径与绝对路径混用

#### 解决方案
```python
import os
from pathlib import Path

class ModelPathValidator:
    @staticmethod
    def validate_model_path(model_path):
        """验证模型路径"""
        # 转换为绝对路径
        abs_path = Path(model_path).resolve()
        
        # 检查文件是否存在
        if not abs_path.exists():
            raise FileNotFoundError(f"Model file not found: {abs_path}")
        
        # 检查文件大小（避免下载不完整）
        file_size = abs_path.stat().st_size
        if file_size < 1024:  # 小于1KB可能是损坏文件
            raise ValueError(f"Model file too small ({file_size} bytes), possibly corrupted")
        
        # 检查文件权限
        if not os.access(abs_path, os.R_OK):
            raise PermissionError(f"No read permission for model file: {abs_path}")
        
        return str(abs_path)
    
    @staticmethod
    def create_model_directory(model_dir):
        """创建模型目录"""
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        print(f"Model directory created: {model_dir}")

# 使用示例
try:
    model_path = ModelPathValidator.validate_model_path("models/buffalo_l.onnx")
    print(f"Valid model path: {model_path}")
except (FileNotFoundError, ValueError, PermissionError) as e:
    print(f"Model path error: {e}")
    # 自动下载或提示用户
```

### 2. 模型版本兼容性问题

#### 常见错误信息
```
RuntimeError: ONNX Runtime only supports opset version 9 or higher
ValueError: Unsupported model format
AttributeError: 'NoneType' object has no attribute 'get'
```

#### 解决方案
```python
import onnx
import onnxruntime as ort

class ModelCompatibilityChecker:
    @staticmethod
    def check_onnx_model(model_path):
        """检查ONNX模型兼容性"""
        try:
            # 加载ONNX模型
            model = onnx.load(model_path)
            
            # 检查opset版本
            opset_version = model.opset_import[0].version
            print(f"Model opset version: {opset_version}")
            
            if opset_version < 9:
                raise ValueError(f"Unsupported opset version: {opset_version}")
            
            # 验证模型
            onnx.checker.check_model(model)
            print("Model validation passed")
            
            # 检查输入输出形状
            input_info = [(inp.name, [dim.dim_value for dim in inp.type.tensor_type.shape.dim]) 
                         for inp in model.graph.input]
            output_info = [(out.name, [dim.dim_value for dim in out.type.tensor_type.shape.dim]) 
                          for out in model.graph.output]
            
            print(f"Input shapes: {input_info}")
            print(f"Output shapes: {output_info}")
            
            return True
            
        except Exception as e:
            print(f"Model compatibility check failed: {e}")
            return False
    
    @staticmethod
    def check_runtime_providers():
        """检查可用的运行时提供者"""
        available_providers = ort.get_available_providers()
        print(f"Available providers: {available_providers}")
        
        # 推荐的提供者优先级
        preferred_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        for provider in preferred_providers:
            if provider in available_providers:
                print(f"Recommended provider: {provider}")
                return provider
        
        return 'CPUExecutionProvider'
```

## 数据类型与维度错误

### 1. 数组维度不匹配

#### 常见错误信息
```
ValueError: cannot reshape array of size 150528 into shape (224,224,3)
IndexError: too many indices for array: array is 2-d, but 3 were given
TypeError: 'NoneType' object is not subscriptable
```

#### 解决方案
```python
import numpy as np
import cv2

class DataShapeValidator:
    @staticmethod
    def validate_image_shape(image, expected_shape=None):
        """验证图像形状"""
        if image is None:
            raise ValueError("Image is None")
        
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(image)}")
        
        print(f"Image shape: {image.shape}")
        print(f"Image dtype: {image.dtype}")
        
        # 检查维度
        if len(image.shape) < 2:
            raise ValueError(f"Invalid image dimensions: {image.shape}")
        
        # 检查通道数
        if len(image.shape) == 3:
            channels = image.shape[2]
            if channels not in [1, 3, 4]:
                raise ValueError(f"Unsupported channel count: {channels}")
        
        # 检查数据类型
        if image.dtype not in [np.uint8, np.float32, np.float64]:
            print(f"Warning: Unusual image dtype: {image.dtype}")
        
        # 检查数值范围
        if image.dtype == np.uint8:
            if image.min() < 0 or image.max() > 255:
                raise ValueError(f"Invalid uint8 range: [{image.min()}, {image.max()}]")
        elif image.dtype in [np.float32, np.float64]:
            if image.min() < 0 or image.max() > 1:
                print(f"Warning: Float image range [{image.min():.3f}, {image.max():.3f}] not in [0,1]")
        
        return True
    
    @staticmethod
    def safe_reshape(array, target_shape):
        """安全的数组重塑"""
        if array is None:
            raise ValueError("Array is None")
        
        original_size = array.size
        target_size = np.prod(target_shape)
        
        if original_size != target_size:
            raise ValueError(
                f"Cannot reshape array of size {original_size} into shape {target_shape} "
                f"(target size: {target_size})"
            )
        
        return array.reshape(target_shape)
    
    @staticmethod
    def normalize_image(image):
        """标准化图像数据"""
        if image.dtype == np.uint8:
            # 转换为float32并归一化到[0,1]
            normalized = image.astype(np.float32) / 255.0
        elif image.dtype in [np.float32, np.float64]:
            # 确保在[0,1]范围内
            normalized = np.clip(image, 0, 1).astype(np.float32)
        else:
            raise ValueError(f"Unsupported image dtype: {image.dtype}")
        
        return normalized

# 使用示例
try:
    image = cv2.imread("test.jpg")
    DataShapeValidator.validate_image_shape(image)
    normalized_image = DataShapeValidator.normalize_image(image)
except (ValueError, TypeError) as e:
    print(f"Data validation error: {e}")
```

### 2. 数据类型转换错误

#### 常见错误信息
```
TypeError: 'NoneType' object cannot be compared to 'float'
ValueError: cannot convert float NaN to integer
OverflowError: cannot convert float infinity to integer
```

#### 解决方案
```python
class SafeDataConverter:
    @staticmethod
    def safe_float_to_int(value, default=0):
        """安全的浮点数转整数"""
        if value is None:
            return default
        
        if np.isnan(value) or np.isinf(value):
            print(f"Warning: Invalid float value {value}, using default {default}")
            return default
        
        try:
            return int(value)
        except (ValueError, OverflowError) as e:
            print(f"Conversion error: {e}, using default {default}")
            return default
    
    @staticmethod
    def safe_compare(value1, value2, default_result=False):
        """安全的数值比较"""
        if value1 is None or value2 is None:
            return default_result
        
        if np.isnan(value1) or np.isnan(value2):
            return default_result
        
        try:
            return float(value1) < float(value2)
        except (ValueError, TypeError):
            return default_result
    
    @staticmethod
    def clean_landmarks(landmarks):
        """清理关键点数据"""
        cleaned = []
        for lm in landmarks:
            if lm is None:
                continue
            
            # 确保是3元素列表/元组 [x, y, confidence]
            if len(lm) < 3:
                continue
            
            x = SafeDataConverter.safe_float_to_int(lm[0])
            y = SafeDataConverter.safe_float_to_int(lm[1])
            conf = max(0.0, min(1.0, float(lm[2]) if lm[2] is not None else 0.0))
            
            cleaned.append([x, y, conf])
        
        return cleaned
```

## 性能与内存问题

### 1. 内存泄漏

#### 问题症状
- 程序运行时间越长，内存占用越高
- 最终导致系统卡顿或程序崩溃
- GPU内存不足错误

#### 解决方案
```python
import gc
import psutil
import threading
from contextlib import contextmanager

class MemoryMonitor:
    def __init__(self, threshold_mb=1000):
        self.threshold_mb = threshold_mb
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """开始内存监控"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止内存监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            
            if memory_mb > self.threshold_mb:
                print(f"Warning: High memory usage: {memory_mb:.1f} MB")
                # 强制垃圾回收
                gc.collect()
            
            time.sleep(5)  # 每5秒检查一次
    
    @staticmethod
    def get_memory_usage():
        """获取当前内存使用情况"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # 物理内存
            'vms_mb': memory_info.vms / 1024 / 1024,  # 虚拟内存
            'percent': process.memory_percent()        # 内存占用百分比
        }

@contextmanager
def memory_cleanup():
    """内存清理上下文管理器"""
    try:
        yield
    finally:
        # 强制垃圾回收
        gc.collect()
        
        # 清理OpenCV缓存
        try:
            cv2.destroyAllWindows()
        except:
            pass

class ResourceManager:
    def __init__(self):
        self.resources = []
    
    def add_resource(self, resource):
        """添加需要管理的资源"""
        self.resources.append(resource)
    
    def cleanup(self):
        """清理所有资源"""
        for resource in self.resources:
            try:
                if hasattr(resource, 'release'):
                    resource.release()
                elif hasattr(resource, 'close'):
                    resource.close()
                elif hasattr(resource, '__del__'):
                    del resource
            except Exception as e:
                print(f"Resource cleanup error: {e}")
        
        self.resources.clear()
        gc.collect()

# 使用示例
with memory_cleanup():
    # 处理大量图像
    for image_path in image_list:
        image = cv2.imread(image_path)
        # 处理图像...
        del image  # 显式删除
```

### 2. 性能瓶颈识别

#### 解决方案
```python
import time
import cProfile
import pstats
from functools import wraps

class PerformanceProfiler:
    def __init__(self):
        self.timing_data = {}
    
    def time_function(self, func_name=None):
        """函数执行时间装饰器"""
        def decorator(func):
            name = func_name or func.__name__
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.time()
                    execution_time = end_time - start_time
                    
                    if name not in self.timing_data:
                        self.timing_data[name] = []
                    self.timing_data[name].append(execution_time)
                    
                    print(f"{name}: {execution_time:.3f}s")
            
            return wrapper
        return decorator
    
    def get_performance_report(self):
        """获取性能报告"""
        report = {}
        for func_name, times in self.timing_data.items():
            report[func_name] = {
                'count': len(times),
                'total_time': sum(times),
                'avg_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times)
            }
        return report
    
    @staticmethod
    def profile_code(func):
        """代码性能分析"""
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = func()
        finally:
            profiler.disable()
            
            # 生成报告
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.print_stats(20)  # 显示前20个最耗时的函数
        
        return result

# 使用示例
profiler = PerformanceProfiler()

@profiler.time_function("face_detection")
def detect_faces(image):
    # 人脸检测代码
    pass

@profiler.time_function("pose_estimation")
def estimate_pose(image):
    # 姿态估计代码
    pass
```

## 模型推理错误

### 1. 输入预处理错误

#### 常见错误信息
```
RuntimeError: Input tensor shape mismatch
ValueError: Input image must be 3-channel RGB
TypeError: Input must be numpy array or tensor
```

#### 解决方案
```python
class InputPreprocessor:
    def __init__(self, target_size=(224, 224), normalize=True):
        self.target_size = target_size
        self.normalize = normalize
    
    def preprocess_image(self, image):
        """标准化图像预处理"""
        try:
            # 1. 验证输入
            if image is None:
                raise ValueError("Input image is None")
            
            if not isinstance(image, np.ndarray):
                raise TypeError(f"Expected numpy array, got {type(image)}")
            
            # 2. 处理灰度图像
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                # RGBA转RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                # BGR转RGB (OpenCV默认是BGR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 3. 调整大小
            if image.shape[:2] != self.target_size:
                image = cv2.resize(image, self.target_size)
            
            # 4. 数据类型转换
            if image.dtype != np.float32:
                image = image.astype(np.float32)
            
            # 5. 归一化
            if self.normalize:
                if image.max() > 1.0:
                    image = image / 255.0
            
            # 6. 添加batch维度
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            # 返回默认图像
            default_image = np.zeros((1, *self.target_size, 3), dtype=np.float32)
            return default_image
    
    def validate_model_input(self, input_tensor, expected_shape):
        """验证模型输入"""
        if input_tensor.shape != expected_shape:
            raise ValueError(
                f"Input shape mismatch: got {input_tensor.shape}, "
                f"expected {expected_shape}"
            )
        
        if input_tensor.dtype != np.float32:
            print(f"Warning: Input dtype is {input_tensor.dtype}, expected float32")
        
        # 检查数值范围
        if input_tensor.min() < 0 or input_tensor.max() > 1:
            print(f"Warning: Input range [{input_tensor.min():.3f}, {input_tensor.max():.3f}] not in [0,1]")
        
        return True
```

### 2. 后处理错误

#### 解决方案
```python
class OutputPostprocessor:
    @staticmethod
    def safe_extract_keypoints(model_output, confidence_threshold=0.5):
        """安全提取关键点"""
        try:
            if model_output is None:
                return []
            
            # 处理不同的输出格式
            if isinstance(model_output, (list, tuple)):
                # 多输出模型
                keypoints_output = model_output[0] if len(model_output) > 0 else None
            else:
                keypoints_output = model_output
            
            if keypoints_output is None:
                return []
            
            # 确保是numpy数组
            if not isinstance(keypoints_output, np.ndarray):
                keypoints_output = np.array(keypoints_output)
            
            # 处理batch维度
            if len(keypoints_output.shape) == 3:
                keypoints_output = keypoints_output[0]  # 取第一个batch
            
            keypoints = []
            for i in range(0, len(keypoints_output), 3):  # x, y, confidence
                if i + 2 < len(keypoints_output):
                    x, y, conf = keypoints_output[i:i+3]
                    
                    # 验证置信度
                    if conf >= confidence_threshold:
                        keypoints.append([float(x), float(y), float(conf)])
                    else:
                        keypoints.append([0.0, 0.0, 0.0])  # 低置信度点设为0
            
            return keypoints
            
        except Exception as e:
            print(f"Keypoint extraction error: {e}")
            return []
    
    @staticmethod
    def safe_extract_bboxes(model_output, confidence_threshold=0.5):
        """安全提取边界框"""
        try:
            if model_output is None:
                return []
            
            bboxes = []
            
            # 处理YOLO格式输出 [x1, y1, x2, y2, conf, class]
            for detection in model_output:
                if len(detection) >= 6:
                    x1, y1, x2, y2, conf, cls = detection[:6]
                    
                    if conf >= confidence_threshold:
                        # 确保坐标有效
                        x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])
                        
                        # 修正坐标顺序
                        if x1 > x2:
                            x1, x2 = x2, x1
                        if y1 > y2:
                            y1, y2 = y2, y1
                        
                        bboxes.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(conf),
                            'class': int(cls)
                        })
            
            return bboxes
            
        except Exception as e:
            print(f"Bbox extraction error: {e}")
            return []
```

## 多线程与并发问题

### 1. 线程安全问题

#### 常见错误
- 模型在多线程中共享导致的竞态条件
- OpenCV在多线程中的不稳定行为
- 内存访问冲突

#### 解决方案
```python
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

class ThreadSafeModelWrapper:
    def __init__(self, model_class, model_args):
        self.model_class = model_class
        self.model_args = model_args
        self.local = threading.local()
    
    def get_model(self):
        """获取线程本地模型实例"""
        if not hasattr(self.local, 'model'):
            self.local.model = self.model_class(**self.model_args)
        return self.local.model
    
    def predict(self, *args, **kwargs):
        """线程安全的预测"""
        model = self.get_model()
        return model.predict(*args, **kwargs)

class SafeVideoProcessor:
    def __init__(self, max_workers=4):
        self.max_workers = max_workers
        self.frame_queue = Queue(maxsize=100)
        self.result_queue = Queue(maxsize=100)
        self.processing = False
        self.executor = None
    
    def start_processing(self, processor_func):
        """开始多线程处理"""
        self.processing = True
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # 启动工作线程
        for _ in range(self.max_workers):
            self.executor.submit(self._worker, processor_func)
    
    def _worker(self, processor_func):
        """工作线程"""
        while self.processing:
            try:
                frame_data = self.frame_queue.get(timeout=1.0)
                if frame_data is None:  # 停止信号
                    break
                
                # 处理帧
                result = processor_func(frame_data)
                
                # 将结果放入结果队列
                try:
                    self.result_queue.put(result, timeout=1.0)
                except:
                    pass  # 结果队列满时丢弃
                
            except Empty:
                continue
            except Exception as e:
                print(f"Worker error: {e}")
    
    def add_frame(self, frame):
        """添加待处理帧"""
        try:
            self.frame_queue.put(frame, block=False)
        except:
            # 队列满时丢弃最旧的帧
            try:
                self.frame_queue.get(block=False)
                self.frame_queue.put(frame, block=False)
            except:
                pass
    
    def get_result(self):
        """获取处理结果"""
        try:
            return self.result_queue.get(block=False)
        except Empty:
            return None
    
    def stop_processing(self):
        """停止处理"""
        self.processing = False
        
        # 发送停止信号
        for _ in range(self.max_workers):
            try:
                self.frame_queue.put(None, timeout=1.0)
            except:
                pass
        
        if self.executor:
            self.executor.shutdown(wait=True)
```

## 调试工具与技巧

### 1. 可视化调试

```python
class VisualDebugger:
    def __init__(self, save_debug_images=True, debug_dir="debug_output"):
        self.save_debug_images = save_debug_images
        self.debug_dir = debug_dir
        
        if save_debug_images:
            Path(debug_dir).mkdir(exist_ok=True)
    
    def visualize_keypoints(self, image, keypoints, title="Keypoints"):
        """可视化关键点"""
        debug_image = image.copy()
        
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.5:
                color = (0, 255, 0)  # 绿色：高置信度
            elif conf > 0.3:
                color = (0, 255, 255)  # 黄色：中等置信度
            else:
                color = (0, 0, 255)  # 红色：低置信度
            
            cv2.circle(debug_image, (int(x), int(y)), 5, color, -1)
            cv2.putText(debug_image, f"{i}:{conf:.2f}", (int(x), int(y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        if self.save_debug_images:
            timestamp = int(time.time() * 1000)
            filename = f"{self.debug_dir}/{title}_{timestamp}.jpg"
            cv2.imwrite(filename, debug_image)
        
        return debug_image
    
    def visualize_detection_pipeline(self, original_image, processed_image, 
                                   detections, stage_name):
        """可视化检测流水线"""
        # 创建组合图像
        h1, w1 = original_image.shape[:2]
        h2, w2 = processed_image.shape[:2]
        
        # 调整图像大小使其高度一致
        target_height = max(h1, h2)
        
        if h1 != target_height:
            scale = target_height / h1
            original_resized = cv2.resize(original_image, (int(w1 * scale), target_height))
        else:
            original_resized = original_image
        
        if h2 != target_height:
            scale = target_height / h2
            processed_resized = cv2.resize(processed_image, (int(w2 * scale), target_height))
        else:
            processed_resized = processed_image
        
        # 水平拼接
        combined = np.hstack([original_resized, processed_resized])
        
        # 添加检测信息
        info_text = f"Stage: {stage_name}, Detections: {len(detections)}"
        cv2.putText(combined, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if self.save_debug_images:
            timestamp = int(time.time() * 1000)
            filename = f"{self.debug_dir}/pipeline_{stage_name}_{timestamp}.jpg"
            cv2.imwrite(filename, combined)
        
        return combined
    
    def log_tensor_info(self, tensor, name):
        """记录张量信息"""
        if tensor is None:
            print(f"{name}: None")
            return
        
        if isinstance(tensor, np.ndarray):
            print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, "
                  f"range=[{tensor.min():.3f}, {tensor.max():.3f}], "
                  f"mean={tensor.mean():.3f}, std={tensor.std():.3f}")
        else:
            print(f"{name}: type={type(tensor)}, value={tensor}")
```

### 2. 错误日志系统

```python
import logging
from datetime import datetime

class CVLogger:
    def __init__(self, log_file="cv_debug.log", level=logging.DEBUG):
        self.logger = logging.getLogger("CVDebugger")
        self.logger.setLevel(level)
        
        # 文件处理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_error(self, error, context=None):
        """记录错误信息"""
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat(),
            'context': context or {}
        }
        
        self.logger.error(f"Error occurred: {error_info}")
        
        # 记录堆栈跟踪
        import traceback
        self.logger.debug(f"Stack trace: {traceback.format_exc()}")
    
    def log_performance(self, function_name, execution_time, input_size=None):
        """记录性能信息"""
        perf_info = {
            'function': function_name,
            'execution_time': execution_time,
            'input_size': input_size,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Performance: {perf_info}")
    
    def log_model_info(self, model_name, model_path, input_shape, output_shape):
        """记录模型信息"""
        model_info = {
            'model_name': model_name,
            'model_path': model_path,
            'input_shape': input_shape,
            'output_shape': output_shape,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Model loaded: {model_info}")
```

## 部署与生产环境问题

### 1. 环境依赖问题

#### 解决方案
```python
import sys
import importlib

class DependencyChecker:
    def __init__(self):
        self.required_packages = {
            'cv2': 'opencv-python',
            'numpy': 'numpy',
            'onnxruntime': 'onnxruntime',
            'mediapipe': 'mediapipe',
            'ultralytics': 'ultralytics'
        }
    
    def check_dependencies(self):
        """检查依赖包"""
        missing_packages = []
        version_info = {}
        
        for package, pip_name in self.required_packages.items():
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                version_info[package] = version
                print(f"✓ {package}: {version}")
            except ImportError:
                missing_packages.append(pip_name)
                print(f"✗ {package}: not found")
        
        if missing_packages:
            print(f"\nMissing packages: {missing_packages}")
            print(f"Install with: pip install {' '.join(missing_packages)}")
            return False
        
        return True
    
    def check_system_info(self):
        """检查系统信息"""
        print(f"Python version: {sys.version}")
        print(f"Platform: {sys.platform}")
        
        # 检查GPU支持
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            print(f"ONNX Runtime providers: {providers}")
        except:
            print("ONNX Runtime not available")
        
        # 检查OpenCV构建信息
        try:
            import cv2
            print(f"OpenCV version: {cv2.__version__}")
            print(f"OpenCV build info: {cv2.getBuildInformation()}")
        except:
            print("OpenCV not available")
```

### 2. 配置管理

```python
import json
import yaml
from pathlib import Path

class ConfigManager:
    def __init__(self, config_path="config.yaml"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
    
    def load_config(self):
        """加载配置文件"""
        if not self.config_path.exists():
            # 创建默认配置
            default_config = self.get_default_config()
            self.save_config(default_config)
            return default_config
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if self.config_path.suffix.lower() == '.json':
                    return json.load(f)
                else:
                    return yaml.safe_load(f)
        except Exception as e:
            print(f"Config loading error: {e}")
            return self.get_default_config()
    
    def get_default_config(self):
        """获取默认配置"""
        return {
            'models': {
                'face_recognition': {
                    'model_path': 'models/buffalo_l.onnx',
                    'confidence_threshold': 0.5
                },
                'pose_estimation': {
                    'model_path': 'models/yolo11n-pose.pt',
                    'confidence_threshold': 0.5
                }
            },
            'processing': {
                'max_workers': 4,
                'frame_skip': 1,
                'target_fps': 30
            },
            'debug': {
                'save_debug_images': False,
                'log_level': 'INFO',
                'performance_monitoring': True
            }
        }
    
    def save_config(self, config=None):
        """保存配置文件"""
        config = config or self.config
        
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                if self.config_path.suffix.lower() == '.json':
                    json.dump(config, f, indent=2, ensure_ascii=False)
                else:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            print(f"Config saving error: {e}")
    
    def get(self, key_path, default=None):
        """获取配置值（支持嵌套键）"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path, value):
        """设置配置值（支持嵌套键）"""
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
        self.save_config()
```

## 最佳实践总结

### 1. 错误处理策略
- **防御性编程**: 始终检查输入有效性
- **优雅降级**: 遇到错误时提供备选方案
- **详细日志**: 记录足够的上下文信息
- **快速失败**: 尽早发现和报告错误

### 2. 性能优化原则
- **预处理优化**: 减少不必要的图像转换
- **内存管理**: 及时释放资源，避免内存泄漏
- **并行处理**: 合理使用多线程/多进程
- **缓存策略**: 缓存重复计算的结果

### 3. 调试技巧
- **分步验证**: 逐步验证每个处理环节
- **可视化输出**: 将中间结果可视化
- **单元测试**: 为关键函数编写测试
- **性能监控**: 持续监控系统性能

### 4. 生产部署建议
- **环境隔离**: 使用Docker或虚拟环境
- **配置外部化**: 将配置与代码分离
- **监控告警**: 建立完善的监控体系
- **版本管理**: 记录模型和代码版本

---

**更新日期**: 2024-01-XX  
**维护者**: YOLOS项目团队  
**版本**: 1.0