#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版实时检测器
集成YOLOv11优化和AIoT平台适配
"""

import cv2
import time
import threading
import numpy as np
from queue import Queue, Empty
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path

from models.enhanced_yolov11_detector import EnhancedYOLOv11Detector
from models.yolo_factory import YOLOFactory
from core.types import DetectionResult, ProcessingResult, TaskType, Status
from utils.logger import get_logger


@dataclass
class RealtimeConfig:
    """实时检测配置"""
    model_type: str = 'yolov11'
    model_size: str = 's'
    device: str = 'auto'
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    
    # 性能配置
    target_fps: float = 30.0
    max_queue_size: int = 10
    detection_interval: int = 1  # 每N帧检测一次
    
    # 优化配置
    edge_optimization: bool = False
    adaptive_inference: bool = True
    tensorrt_optimize: bool = True
    half_precision: bool = True
    
    # 显示配置
    show_fps: bool = True
    show_confidence: bool = True
    draw_boxes: bool = True
    box_thickness: int = 2


class EnhancedRealtimeDetector:
    """
    增强版实时检测器
    
    特性:
    - YOLOv11集成
    - 自适应性能调优
    - 多线程优化
    - 边缘设备适配
    - 智能帧跳过
    - 性能监控
    """
    
    def __init__(self, config: Optional[RealtimeConfig] = None):
        """
        初始化增强版实时检测器
        
        Args:
            config: 检测配置
        """
        self.config = config or RealtimeConfig()
        self.logger = get_logger("EnhancedRealtimeDetector")
        
        # 初始化检测器
        self._init_detector()
        
        # 线程控制
        self.is_running = False
        self.detection_thread = None
        self.display_thread = None
        
        # 队列管理
        self.frame_queue = Queue(maxsize=self.config.max_queue_size)
        self.result_queue = Queue(maxsize=self.config.max_queue_size)
        
        # 性能统计
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        self.avg_detection_time = 0.0
        
        # 回调函数
        self.detection_callback: Optional[Callable] = None
        self.frame_callback: Optional[Callable] = None
        
        # 帧跳过控制
        self.frame_skip_counter = 0
        
        self.logger.info(f"增强版实时检测器初始化完成")
        self.logger.info(f"模型: {self.config.model_type}{self.config.model_size}")
        self.logger.info(f"目标FPS: {self.config.target_fps}")
    
    def _init_detector(self):
        """初始化检测器"""
        try:
            if self.config.model_type == 'yolov11':
                self.detector = EnhancedYOLOv11Detector(
                    model_size=self.config.model_size,
                    device=self.config.device,
                    half_precision=self.config.half_precision,
                    tensorrt_optimize=self.config.tensorrt_optimize,
                    confidence_threshold=self.config.confidence_threshold,
                    iou_threshold=self.config.iou_threshold,
                    edge_optimization=self.config.edge_optimization,
                    adaptive_inference=self.config.adaptive_inference
                )
            else:
                # 使用工厂创建其他类型的检测器
                self.detector = YOLOFactory.create_model(
                    model_type=self.config.model_type,
                    device=self.config.device,
                    model_size=self.config.model_size
                )
            
            self.logger.info("检测器初始化成功")
            
        except Exception as e:
            self.logger.error(f"检测器初始化失败: {e}")
            raise
    
    def set_detection_callback(self, callback: Callable[[np.ndarray, List[DetectionResult]], None]):
        """设置检测结果回调函数"""
        self.detection_callback = callback
    
    def set_frame_callback(self, callback: Callable[[np.ndarray], np.ndarray]):
        """设置帧处理回调函数"""
        self.frame_callback = callback
    
    def _detection_worker(self):
        """检测工作线程"""
        self.logger.info("检测线程启动")
        
        while self.is_running:
            try:
                # 获取帧
                frame = self.frame_queue.get(timeout=0.1)
                
                # 执行检测
                start_time = time.time()
                
                if hasattr(self.detector, 'detect_adaptive'):
                    # 使用自适应检测
                    results = self.detector.detect_adaptive(
                        frame, 
                        target_fps=self.config.target_fps
                    )
                else:
                    # 使用标准检测
                    results = self.detector.detect(frame)
                
                detection_time = time.time() - start_time
                
                # 更新统计
                self.detection_count += 1
                self.avg_detection_time = (
                    (self.avg_detection_time * (self.detection_count - 1) + detection_time) 
                    / self.detection_count
                )
                
                # 放入结果队列
                if not self.result_queue.full():
                    self.result_queue.put((frame, results, detection_time))
                
                # 调用检测回调
                if self.detection_callback:
                    self.detection_callback(frame, results)
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"检测线程错误: {e}")
                time.sleep(0.01)
        
        self.logger.info("检测线程结束")
    
    def _should_skip_frame(self) -> bool:
        """判断是否应该跳过当前帧"""
        self.frame_skip_counter += 1
        
        if self.frame_skip_counter >= self.config.detection_interval:
            self.frame_skip_counter = 0
            return False
        
        return True
    
    def start_camera_detection(self, 
                             camera_id: int = 0, 
                             window_name: str = "YOLOS实时检测",
                             camera_config: Optional[Dict[str, Any]] = None):
        """
        开始摄像头检测
        
        Args:
            camera_id: 摄像头ID
            window_name: 显示窗口名称
            camera_config: 摄像头配置
        """
        # 初始化摄像头
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {camera_id}")
        
        # 配置摄像头
        default_config = {
            'width': 640,
            'height': 480,
            'fps': 30,
            'buffer_size': 1
        }
        
        if camera_config:
            default_config.update(camera_config)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, default_config['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, default_config['height'])
        cap.set(cv2.CAP_PROP_FPS, default_config['fps'])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, default_config['buffer_size'])
        
        self.logger.info(f"摄像头配置: {default_config}")
        
        # 启动检测
        self._start_detection_threads()
        
        self.logger.info("开始实时检测，按 'q' 退出，按 's' 保存截图")
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("无法读取摄像头帧")
                    break
                
                # 帧跳过逻辑
                if not self._should_skip_frame():
                    # 添加帧到检测队列
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame.copy())
                
                # 显示处理
                display_frame = self._process_display_frame(frame)
                
                # 应用帧回调
                if self.frame_callback:
                    display_frame = self.frame_callback(display_frame)
                
                # 显示图像
                cv2.imshow(window_name, display_frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_screenshot(display_frame)
                elif key == ord('p'):
                    self._print_performance_stats()
                
        except KeyboardInterrupt:
            self.logger.info("用户中断检测")
        except Exception as e:
            self.logger.error(f"检测过程错误: {e}")
        finally:
            self.stop()
            cap.release()
            cv2.destroyAllWindows()
    
    def _start_detection_threads(self):
        """启动检测线程"""
        self.is_running = True
        self.start_time = time.time()
        
        # 启动检测线程
        self.detection_thread = threading.Thread(target=self._detection_worker)
        self.detection_thread.daemon = True
        self.detection_thread.start()
    
    def _process_display_frame(self, frame: np.ndarray) -> np.ndarray:
        """处理显示帧"""
        display_frame = frame.copy()
        
        # 获取最新检测结果
        latest_result = None
        try:
            while not self.result_queue.empty():
                latest_result = self.result_queue.get_nowait()
        except Empty:
            pass
        
        # 绘制检测结果
        if latest_result and self.config.draw_boxes:
            result_frame, detections, detection_time = latest_result
            display_frame = self._draw_detections(display_frame, detections)
        
        # 绘制性能信息
        if self.config.show_fps:
            display_frame = self._draw_performance_info(display_frame)
        
        return display_frame
    
    def _draw_detections(self, frame: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        """绘制检测结果"""
        for detection in detections:
            bbox = detection.bbox
            
            # 绘制边界框
            color = self._get_class_color(detection.class_id)
            cv2.rectangle(frame, 
                         (bbox.x, bbox.y), 
                         (bbox.x2, bbox.y2), 
                         color, 
                         self.config.box_thickness)
            
            # 绘制标签
            label = f"{detection.class_name}"
            if self.config.show_confidence:
                label += f" {detection.confidence:.2f}"
            
            # 计算文本位置
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # 绘制文本背景
            cv2.rectangle(frame,
                         (bbox.x, bbox.y - text_height - 5),
                         (bbox.x + text_width, bbox.y),
                         color, -1)
            
            # 绘制文本
            cv2.putText(frame, label,
                       (bbox.x, bbox.y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 255), 1)
        
        return frame
    
    def _draw_performance_info(self, frame: np.ndarray) -> np.ndarray:
        """绘制性能信息"""
        # 计算FPS
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time > 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
        
        # 绘制FPS
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 绘制检测时间
        if self.avg_detection_time > 0:
            det_time_text = f"Det: {self.avg_detection_time*1000:.1f}ms"
            cv2.putText(frame, det_time_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # 绘制队列状态
        queue_text = f"Q: {self.frame_queue.qsize()}/{self.config.max_queue_size}"
        cv2.putText(frame, queue_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        return frame
    
    def _get_class_color(self, class_id: int) -> tuple:
        """获取类别颜色"""
        colors = [
            (255, 0, 0),    # 红色
            (0, 255, 0),    # 绿色
            (0, 0, 255),    # 蓝色
            (255, 255, 0),  # 黄色
            (255, 0, 255),  # 紫色
            (0, 255, 255),  # 青色
            (128, 0, 128),  # 紫色
            (255, 165, 0),  # 橙色
        ]
        return colors[class_id % len(colors)]
    
    def _save_screenshot(self, frame: np.ndarray):
        """保存截图"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"yolos_screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        self.logger.info(f"截图已保存: {filename}")
    
    def _print_performance_stats(self):
        """打印性能统计"""
        stats = {
            'FPS': self.fps,
            '平均检测时间': f"{self.avg_detection_time*1000:.1f}ms",
            '总检测次数': self.detection_count,
            '帧队列大小': self.frame_queue.qsize(),
            '结果队列大小': self.result_queue.qsize()
        }
        
        print("\n=== 性能统计 ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
        print("================\n")
    
    def start_video_detection(self, 
                            video_path: str, 
                            output_path: Optional[str] = None,
                            show_progress: bool = True):
        """
        开始视频检测
        
        Args:
            video_path: 视频文件路径
            output_path: 输出视频路径
            show_progress: 是否显示进度
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {video_path}")
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.logger.info(f"视频信息: {width}x{height}, {fps}FPS, {total_frames}帧")
        
        # 设置输出视频
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        self.is_running = True
        frame_idx = 0
        
        try:
            while self.is_running and frame_idx < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 执行检测
                if hasattr(self.detector, 'detect_adaptive'):
                    results = self.detector.detect_adaptive(frame)
                else:
                    results = self.detector.detect(frame)
                
                # 绘制结果
                annotated_frame = self._draw_detections(frame, results)
                
                # 保存或显示
                if out:
                    out.write(annotated_frame)
                
                if not output_path:  # 实时显示
                    cv2.imshow('视频检测', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # 调用回调
                if self.detection_callback:
                    self.detection_callback(frame, results)
                
                # 显示进度
                frame_idx += 1
                if show_progress and frame_idx % 30 == 0:
                    progress = (frame_idx / total_frames) * 100
                    self.logger.info(f"处理进度: {progress:.1f}% ({frame_idx}/{total_frames})")
                    
        except KeyboardInterrupt:
            self.logger.info("用户中断视频处理")
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            self.logger.info("视频处理完成")
    
    def stop(self):
        """停止检测"""
        self.is_running = False
        
        # 等待线程结束
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2.0)
        
        self.logger.info("检测器已停止")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        stats = {
            'fps': self.fps,
            'avg_detection_time': self.avg_detection_time,
            'total_detections': self.detection_count,
            'frame_queue_size': self.frame_queue.qsize(),
            'result_queue_size': self.result_queue.qsize(),
            'is_running': self.is_running
        }
        
        # 如果使用增强检测器，获取其性能统计
        if hasattr(self.detector, 'get_performance_stats'):
            detector_stats = self.detector.get_performance_stats()
            stats.update({'detector_' + k: v for k, v in detector_stats.items()})
        
        return stats
    
    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"配置更新: {key} = {value}")
        
        # 如果更新了检测相关配置，重新初始化检测器
        detection_params = ['model_type', 'model_size', 'device', 'confidence_threshold', 'iou_threshold']
        if any(key in detection_params for key in kwargs.keys()):
            self.logger.info("检测配置已更新，重新初始化检测器")
            self._init_detector()


# 使用示例
if __name__ == "__main__":
    # 创建配置
    config = RealtimeConfig(
        model_type='yolov11',
        model_size='s',
        target_fps=25.0,
        edge_optimization=True,
        adaptive_inference=True
    )
    
    # 创建检测器
    detector = EnhancedRealtimeDetector(config)
    
    # 设置回调函数
    def detection_callback(frame, results):
        print(f"检测到 {len(results)} 个目标")
    
    detector.set_detection_callback(detection_callback)
    
    # 开始检测
    try:
        detector.start_camera_detection(camera_id=0)
    except KeyboardInterrupt:
        print("检测已停止")
    finally:
        detector.stop()