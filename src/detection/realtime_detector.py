"""
实时检测器
"""

import cv2
import time
import threading
from queue import Queue
from typing import Optional, Callable, Dict, Any, List
import numpy as np

from ..models.yolo_factory import YOLOFactory
from .feature_aggregation import FeatureAggregator, AggregationConfig, PlatformType
from .temporal_aggregator import TemporalAggregator, TemporalConfig, DetectionResult, AggregationStrategy
from ..tracking.tracking_integration import TrackingIntegration, IntegratedTrackingConfig, TrackingMode
from ..tracking.multi_object_tracker import TrackingConfig, TrackingStrategy


class RealtimeDetector:
    """实时检测器"""
    
    def __init__(self, 
                 model_type: str = 'yolov8',
                 model_path: Optional[str] = None,
                 device: str = 'auto',
                 enable_aggregation: bool = True,
                 platform_type: PlatformType = PlatformType.DESKTOP,
                 enable_tracking: bool = True,
                 tracking_config: Optional[IntegratedTrackingConfig] = None):
        """
        初始化实时检测器
        
        Args:
            model_type: 模型类型
            model_path: 模型路径
            device: 设备类型
            enable_aggregation: 是否启用特征聚合
            platform_type: 平台类型
        """
        self.model = YOLOFactory.create_model(model_type, model_path, device)
        self.is_running = False
        self.frame_queue = Queue(maxsize=10)
        self.result_queue = Queue(maxsize=10)
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self._adaptive_fps = 30.0  # 自适应帧率
        self._target_fps = 30.0
        
        # 特征聚合器
        self.enable_aggregation = enable_aggregation
        if enable_aggregation:
            agg_config = AggregationConfig(platform_type=platform_type)
            self.feature_aggregator = FeatureAggregator(agg_config)
            
            # 时序聚合器
            temporal_config = TemporalConfig(
                buffer_size=3 if platform_type in [PlatformType.RASPBERRY_PI, PlatformType.ESP32] else 5,
                strategy=AggregationStrategy.ADAPTIVE,
                enable_tracking=True
            )
            self.temporal_aggregator = TemporalAggregator(temporal_config)
        else:
            self.feature_aggregator = None
            self.temporal_aggregator = None
        
        # 跟踪功能
        self.enable_tracking = enable_tracking
        self.tracking_integration = None
        if enable_tracking:
            config = tracking_config or IntegratedTrackingConfig(
                mode=TrackingMode.ENHANCED,
                tracking_config=TrackingConfig(
                    strategy=TrackingStrategy.HYBRID,
                    max_missing_frames=10
                ),
                aggregation_config=agg_config if enable_aggregation else None,
                temporal_config=temporal_config if enable_aggregation else None
            )
            self.tracking_integration = TrackingIntegration(config)
        
        # 性能监控
        self._performance_history = []
        self._last_performance_check = time.time()
        
        # 回调函数
        self.detection_callback: Optional[Callable] = None
        self.frame_callback: Optional[Callable] = None
    
    def set_detection_callback(self, callback: Callable):
        """设置检测结果回调函数"""
        self.detection_callback = callback
    
    def set_frame_callback(self, callback: Callable):
        """设置帧处理回调函数"""
        self.frame_callback = callback
    
    def _detection_worker(self):
        """检测工作线程"""
        while self.is_running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get(timeout=0.1)
                    
                    # 执行检测
                    start_time = time.time()
                    results = self.model.predict(frame)
                    detection_time = time.time() - start_time
                    
                    # 应用聚合和跟踪策略
                    if self.tracking_integration:
                        # 转换检测结果格式
                        detection_results = self._convert_to_detection_results(results)
                        
                        # 跟踪集成器会自动处理特征聚合和时序聚合
                        tracked_results = self.tracking_integration.process_detections(detection_results, frame)
                        
                        # 转换回原格式
                        results = self._convert_from_detection_results(tracked_results)
                    elif self.enable_aggregation and self.temporal_aggregator:
                        # 传统聚合方式（向后兼容）
                        # 转换检测结果格式
                        detection_results = self._convert_to_detection_results(results)
                        
                        # 应用时序聚合
                        aggregated_results = self.temporal_aggregator.add_detections(detection_results, frame)
                        
                        # 转换回原格式
                        results = self._convert_from_detection_results(aggregated_results)
                    
                    # 自适应帧率调整
                    self._adjust_adaptive_fps(detection_time)
                    
                    # 放入结果队列
                    if not self.result_queue.full():
                        self.result_queue.put((frame, results))
                    
                    # 调用检测回调
                    if self.detection_callback:
                        self.detection_callback(frame, results)
                        
            except Exception as e:
                print(f"检测线程错误: {e}")
                time.sleep(0.01)
    
    def start_camera_detection(self, camera_id: int = 0, window_name: str = "实时检测"):
        """开始摄像头检测"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {camera_id}")
        
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_running = True
        
        # 启动检测线程
        detection_thread = threading.Thread(target=self._detection_worker)
        detection_thread.daemon = True
        detection_thread.start()
        
        print("开始实时检测，按 'q' 退出")
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 添加帧到队列
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())
                
                # 获取检测结果并显示
                display_frame = frame.copy()
                if not self.result_queue.empty():
                    result_frame, results = self.result_queue.get()
                    display_frame = self.model.draw_results(result_frame, results)
                
                # 计算FPS
                self.frame_count += 1
                elapsed_time = time.time() - self.start_time
                if elapsed_time > 1.0:
                    self.fps = self.frame_count / elapsed_time
                    self.frame_count = 0
                    self.start_time = time.time()
                
                # 显示FPS
                cv2.putText(display_frame, f"FPS: {self.fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 调用帧回调
                if self.frame_callback:
                    display_frame = self.frame_callback(display_frame)
                
                # 显示图像
                cv2.imshow(window_name, display_frame)
                
                # 检查退出条件
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
        finally:
            self.stop()
            cap.release()
            cv2.destroyAllWindows()
    
    def start_video_detection(self, video_path: str, output_path: Optional[str] = None):
        """开始视频检测"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {video_path}")
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 设置输出视频
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        self.is_running = True
        frame_idx = 0
        
        print(f"开始处理视频: {video_path}")
        print(f"总帧数: {total_frames}, FPS: {fps}")
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 执行检测
                results = self.model.predict(frame)
                
                # 绘制结果
                annotated_frame = self.model.draw_results(frame, results)
                
                # 保存或显示
                if out:
                    out.write(annotated_frame)
                else:
                    cv2.imshow('视频检测', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # 调用回调
                if self.detection_callback:
                    self.detection_callback(frame, results)
                
                # 显示进度
                frame_idx += 1
                if frame_idx % 30 == 0:
                    progress = (frame_idx / total_frames) * 100
                    print(f"处理进度: {progress:.1f}% ({frame_idx}/{total_frames})")
                    
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            print("视频处理完成")
    
    def stop(self):
        """停止检测"""
        self.is_running = False
    
    def get_fps(self) -> float:
        """获取当前FPS"""
        return self.fps
    
    def _convert_to_detection_results(self, results) -> List[DetectionResult]:
        """转换检测结果为DetectionResult格式"""
        detection_results = []
        
        if hasattr(results, 'boxes') and results.boxes is not None:
            boxes = results.boxes
            for i in range(len(boxes)):
                if hasattr(boxes, 'xyxy') and hasattr(boxes, 'conf') and hasattr(boxes, 'cls'):
                    bbox = boxes.xyxy[i].cpu().numpy()
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    class_name = results.names.get(class_id, f"class_{class_id}") if hasattr(results, 'names') else f"class_{class_id}"
                    
                    detection_results.append(DetectionResult(
                        bbox=tuple(bbox),
                        confidence=confidence,
                        class_id=class_id,
                        class_name=class_name,
                        timestamp=time.time(),
                        frame_id=self.frame_count
                    ))
        
        return detection_results
    
    def _convert_from_detection_results(self, detection_results: List[DetectionResult]):
        """从DetectionResult格式转换回原格式"""
        # 这里需要根据具体的结果格式进行转换
        # 简化实现，返回原始格式的模拟
        return detection_results
    
    def _adjust_adaptive_fps(self, detection_time: float):
        """自适应帧率调整"""
        # 记录性能数据
        current_time = time.time()
        self._performance_history.append({
            'timestamp': current_time,
            'detection_time': detection_time
        })
        
        # 保持最近10秒的数据
        cutoff_time = current_time - 10.0
        self._performance_history = [p for p in self._performance_history if p['timestamp'] > cutoff_time]
        
        # 每秒调整一次帧率
        if current_time - self._last_performance_check > 1.0:
            if len(self._performance_history) > 5:
                avg_detection_time = np.mean([p['detection_time'] for p in self._performance_history])
                
                # 计算理论最大帧率
                max_fps = 1.0 / (avg_detection_time + 0.01)  # 加上一些缓冲
                
                # 自适应调整
                if max_fps < self._target_fps * 0.8:
                    self._adaptive_fps = max(max_fps * 0.9, 5.0)  # 最低5fps
                elif max_fps > self._target_fps * 1.2:
                    self._adaptive_fps = min(self._target_fps, max_fps * 0.95)
                
            self._last_performance_check = current_time
    
    def set_target_fps(self, fps: float):
        """设置目标帧率"""
        self._target_fps = max(1.0, min(fps, 60.0))
        self._adaptive_fps = self._target_fps
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """获取聚合统计信息"""
        stats = {}
        if self.feature_aggregator:
            stats['feature_aggregation'] = self.feature_aggregator.get_statistics()
        if self.temporal_aggregator:
            stats['temporal_aggregation'] = self.temporal_aggregator.get_statistics()
        if self.tracking_integration:
            stats['tracking_stats'] = self.tracking_integration.get_tracking_stats()
        return stats
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            'model_type': type(self.model).__name__,
            'model_info': self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {},
            'current_fps': self.fps,
            'adaptive_fps': self._adaptive_fps,
            'target_fps': self._target_fps,
            'aggregation_enabled': self.enable_aggregation,
            'tracking_enabled': self.enable_tracking,
            'queue_sizes': {
                'frame_queue': self.frame_queue.qsize(),
                'result_queue': self.result_queue.qsize()
            }
        }
        
        if self.tracking_integration:
            info['tracking_stats'] = self.tracking_integration.get_tracking_stats()
            info['active_tracks'] = len(self.tracking_integration.get_active_tracks())
        
        if self.enable_aggregation:
            info['aggregation_stats'] = self.get_aggregation_stats()
        
        return info