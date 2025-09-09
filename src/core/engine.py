#!/usr/bin/env python3
"""
YOLOS核心引擎
统一的系统入口点，实现高内聚低耦合的架构设计
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import threading
import queue
import time
from dataclasses import dataclass
from enum import Enum

from .config import YOLOSConfig
from .logger import YOLOSLogger

class ProcessingMode(Enum):
    """处理模式"""
    REALTIME = "realtime"
    BATCH = "batch"
    SINGLE = "single"

@dataclass
class DetectionResult:
    """检测结果数据结构"""
    bbox: List[int]  # [x, y, w, h]
    confidence: float
    class_id: int
    class_name: str
    timestamp: float

@dataclass
class RecognitionResult:
    """识别结果数据结构"""
    detection: DetectionResult
    features: Dict[str, Any]
    metadata: Dict[str, Any]
    processing_time: float

@dataclass
class FrameResult:
    """帧处理结果"""
    frame_id: int
    timestamp: float
    detections: List[DetectionResult]
    recognitions: List[RecognitionResult]
    processing_time: float
    frame_shape: tuple

class YOLOSEngine:
    """
    YOLOS核心引擎
    
    设计原则:
    1. 单一入口点 - 所有功能通过引擎访问
    2. 高内聚 - 相关功能集中管理
    3. 低耦合 - 模块间通过接口通信
    4. 可扩展 - 支持插件式扩展
    """
    
    def __init__(self, config: Optional[YOLOSConfig] = None):
        """初始化引擎"""
        self.config = config or YOLOSConfig()
        self.logger = YOLOSLogger.get_logger("engine")
        
        # 核心组件
        self._detector = None
        self._recognizer = None
        self._preprocessor = None
        self._postprocessor = None
        
        # 状态管理
        self._initialized = False
        self._running = False
        self._frame_counter = 0
        
        # 性能监控
        self._performance_stats = {
            'total_frames': 0,
            'total_processing_time': 0.0,
            'average_fps': 0.0,
            'last_fps': 0.0
        }
        
        # 线程安全
        self._lock = threading.RLock()
        self._result_queue = queue.Queue(maxsize=100)
        
        self.logger.info("YOLOS引擎创建成功")
    
    def initialize(self) -> bool:
        """
        初始化引擎
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            with self._lock:
                if self._initialized:
                    self.logger.warning("引擎已经初始化")
                    return True
                
                self.logger.info("开始初始化YOLOS引擎...")
                
                # 1. 初始化检测器
                self._initialize_detector()
                
                # 2. 初始化识别器
                self._initialize_recognizer()
                
                # 3. 初始化预处理器
                self._initialize_preprocessor()
                
                # 4. 初始化后处理器
                self._initialize_postprocessor()
                
                self._initialized = True
                self.logger.info("YOLOS引擎初始化完成")
                return True
                
        except Exception as e:
            self.logger.error(f"引擎初始化失败: {e}")
            return False
    
    def _initialize_detector(self):
        """初始化检测器"""
        from ..detection.factory import DetectorFactory
        
        detector_config = self.config.detection_config
        detector_type = detector_config.get('type', 'yolo')
        
        self._detector = DetectorFactory.create_detector(detector_type, detector_config)
        self.logger.info(f"检测器初始化完成: {detector_type}")
    
    def _initialize_recognizer(self):
        """初始化识别器"""
        from ..recognition.factory import RecognizerFactory
        
        recognizer_config = self.config.recognition_config
        recognizer_types = recognizer_config.get('types', ['face', 'gesture', 'object'])
        
        self._recognizer = RecognizerFactory.create_multi_recognizer(
            recognizer_types, recognizer_config
        )
        self.logger.info(f"识别器初始化完成: {recognizer_types}")
    
    def _initialize_preprocessor(self):
        """初始化预处理器"""
        from ..preprocessing.factory import PreprocessorFactory
        
        preprocess_config = self.config.preprocessing_config
        self._preprocessor = PreprocessorFactory.create_preprocessor(preprocess_config)
        self.logger.info("预处理器初始化完成")
    
    def _initialize_postprocessor(self):
        """初始化后处理器"""
        from ..postprocessing.factory import PostprocessorFactory
        
        postprocess_config = self.config.postprocessing_config
        self._postprocessor = PostprocessorFactory.create_postprocessor(postprocess_config)
        self.logger.info("后处理器初始化完成")
    
    def process_frame(self, frame: np.ndarray, frame_id: Optional[int] = None) -> FrameResult:
        """
        处理单帧图像
        
        Args:
            frame: 输入图像帧
            frame_id: 帧ID，如果为None则自动生成
            
        Returns:
            FrameResult: 处理结果
        """
        if not self._initialized:
            raise RuntimeError("引擎未初始化，请先调用initialize()")
        
        start_time = time.time()
        
        if frame_id is None:
            frame_id = self._frame_counter
            self._frame_counter += 1
        
        try:
            # 1. 预处理
            processed_frame = self._preprocessor.process(frame)
            
            # 2. 检测
            detections = self._detector.detect(processed_frame)
            
            # 3. 识别
            recognitions = []
            for detection in detections:
                recognition = self._recognizer.recognize(processed_frame, detection)
                if recognition:
                    recognitions.append(recognition)
            
            # 4. 后处理
            final_result = self._postprocessor.process(
                frame, detections, recognitions
            )
            
            processing_time = time.time() - start_time
            
            # 5. 创建结果
            result = FrameResult(
                frame_id=frame_id,
                timestamp=start_time,
                detections=detections,
                recognitions=recognitions,
                processing_time=processing_time,
                frame_shape=frame.shape
            )
            
            # 6. 更新性能统计
            self._update_performance_stats(processing_time)
            
            # 7. 记录日志
            self.logger.debug(
                f"帧 {frame_id} 处理完成",
                processing_time=processing_time,
                detections_count=len(detections),
                recognitions_count=len(recognitions)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"帧 {frame_id} 处理失败: {e}")
            raise
    
    def process_video(self, video_source: Union[str, int], 
                     mode: ProcessingMode = ProcessingMode.REALTIME) -> None:
        """
        处理视频流
        
        Args:
            video_source: 视频源 (文件路径或摄像头ID)
            mode: 处理模式
        """
        if not self._initialized:
            raise RuntimeError("引擎未初始化，请先调用initialize()")
        
        self.logger.info(f"开始处理视频: {video_source}, 模式: {mode.value}")
        
        # 打开视频源
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {video_source}")
        
        try:
            self._running = True
            frame_id = 0
            
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 处理帧
                result = self.process_frame(frame, frame_id)
                
                # 将结果放入队列
                try:
                    self._result_queue.put_nowait(result)
                except queue.Full:
                    # 队列满时丢弃最旧的结果
                    try:
                        self._result_queue.get_nowait()
                        self._result_queue.put_nowait(result)
                    except queue.Empty:
                        pass
                
                frame_id += 1
                
                # 实时模式下的帧率控制
                if mode == ProcessingMode.REALTIME:
                    target_fps = self.config.camera_config.get('fps', 30)
                    time.sleep(max(0, 1.0/target_fps - result.processing_time))
                
        finally:
            cap.release()
            self._running = False
            self.logger.info("视频处理完成")
    
    def process_image(self, image_path: str) -> FrameResult:
        """
        处理单张图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            FrameResult: 处理结果
        """
        if not self._initialized:
            raise RuntimeError("引擎未初始化，请先调用initialize()")
        
        # 加载图像
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        self.logger.info(f"处理图像: {image_path}")
        return self.process_frame(frame)
    
    def get_result(self, timeout: Optional[float] = None) -> Optional[FrameResult]:
        """
        获取处理结果
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            FrameResult: 处理结果，如果超时返回None
        """
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop(self):
        """停止引擎"""
        self._running = False
        self.logger.info("引擎已停止")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        with self._lock:
            return self._performance_stats.copy()
    
    def _update_performance_stats(self, processing_time: float):
        """更新性能统计"""
        with self._lock:
            self._performance_stats['total_frames'] += 1
            self._performance_stats['total_processing_time'] += processing_time
            
            if self._performance_stats['total_frames'] > 0:
                avg_time = (self._performance_stats['total_processing_time'] / 
                           self._performance_stats['total_frames'])
                self._performance_stats['average_fps'] = 1.0 / avg_time if avg_time > 0 else 0
            
            self._performance_stats['last_fps'] = 1.0 / processing_time if processing_time > 0 else 0
    
    def reset_stats(self):
        """重置性能统计"""
        with self._lock:
            self._performance_stats = {
                'total_frames': 0,
                'total_processing_time': 0.0,
                'average_fps': 0.0,
                'last_fps': 0.0
            }
            self.logger.info("性能统计已重置")
    
    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized
    
    def is_running(self) -> bool:
        """检查是否正在运行"""
        return self._running
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """获取支持的格式"""
        return {
            'image_formats': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'],
            'video_formats': ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'],
            'detection_types': self._detector.get_supported_classes() if self._detector else [],
            'recognition_types': self._recognizer.get_supported_types() if self._recognizer else []
        }
    
    def __enter__(self):
        """上下文管理器入口"""
        if not self.initialize():
            raise RuntimeError("引擎初始化失败")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()

# 便捷函数
def create_engine(config_path: Optional[str] = None) -> YOLOSEngine:
    """
    创建YOLOS引擎实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        YOLOSEngine: 引擎实例
    """
    if config_path:
        config = YOLOSConfig.from_file(config_path)
    else:
        config = YOLOSConfig()
    
    return YOLOSEngine(config)

def quick_detect(image_path: str, config_path: Optional[str] = None) -> FrameResult:
    """
    快速检测单张图像
    
    Args:
        image_path: 图像路径
        config_path: 配置文件路径
        
    Returns:
        FrameResult: 检测结果
    """
    with create_engine(config_path) as engine:
        return engine.process_image(image_path)

if __name__ == "__main__":
    # 测试引擎
    engine = create_engine()
    
    if engine.initialize():
        print("✅ 引擎初始化成功")
        print(f"📊 支持的格式: {engine.get_supported_formats()}")
        
        # 测试性能统计
        stats = engine.get_performance_stats()
        print(f"📈 性能统计: {stats}")
    else:
        print("❌ 引擎初始化失败")