#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合稳定版YOLOS GUI - 结合稳定摄像头处理和Tkinter显示
解决OpenCV GUI在Windows上的兼容性问题
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import json
import logging
import random
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import sys
import os
from PIL import Image, ImageTk

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hybrid_stable_yolos.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class StableYOLOSDetector:
    """稳定版YOLOS检测器 - 模拟YOLO检测"""
    
    def __init__(self):
        # 检测参数
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        
        # 模拟检测历史
        self.detection_history = []
        self.max_history = 10
        
        # 目标类别分类 - 按运动特性分组
        self.moving_objects = ['person', 'car', 'dog', 'cat', 'bicycle']  # 动态目标
        self.static_objects = ['bottle', 'chair', 'book', 'cup', 'laptop']  # 静态目标
        self.all_classes = self.moving_objects + self.static_objects
        
        # 检测频率控制
        self.moving_detection_interval = 2   # 动态目标每2帧检测一次
        self.static_detection_interval = 10  # 静态目标每10帧检测一次
        self.last_moving_detection = 0
        self.last_static_detection = 0
        
        # 目标稳定性跟踪
        self.object_positions = {}  # 跟踪目标位置变化
        self.position_threshold = 30  # 位置变化阈值
        
    def update_parameters(self, confidence: float, nms: float):
        """更新检测参数"""
        self.confidence_threshold = confidence
        self.nms_threshold = nms
        
    def should_detect_moving_objects(self, frame_count: int) -> bool:
        """判断是否应该检测动态目标"""
        return (frame_count - self.last_moving_detection) >= self.moving_detection_interval
    
    def should_detect_static_objects(self, frame_count: int) -> bool:
        """判断是否应该检测静态目标"""
        return (frame_count - self.last_static_detection) >= self.static_detection_interval
    
    def detect_objects(self, frame: np.ndarray, frame_count: int) -> List[Dict]:
        """智能目标检测 - 根据目标类型调整检测频率"""
        detections = []
        h, w = frame.shape[:2]
        
        # 检查是否需要检测动态目标
        detect_moving = self.should_detect_moving_objects(frame_count)
        detect_static = self.should_detect_static_objects(frame_count)
        
        if not detect_moving and not detect_static:
            return detections
        
        # 动态目标检测
        if detect_moving:
            self.last_moving_detection = frame_count
            moving_detections = self._detect_moving_objects(frame, h, w)
            detections.extend(moving_detections)
        
        # 静态目标检测
        if detect_static:
            self.last_static_detection = frame_count
            static_detections = self._detect_static_objects(frame, h, w)
            detections.extend(static_detections)
        
        # 更新目标位置跟踪
        self._update_object_tracking(detections)
        
        # 记录检测历史
        self.detection_history.append(len(detections))
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)
            
        return detections
    
    def _detect_moving_objects(self, frame: np.ndarray, h: int, w: int) -> List[Dict]:
        """检测动态目标"""
        detections = []
        num_moving = random.randint(0, 2)  # 动态目标通常较少
        
        for i in range(num_moving):
            class_name = random.choice(self.moving_objects)
            
            # 动态目标位置变化较大
            if class_name in self.object_positions:
                # 基于上次位置生成新位置（模拟运动）
                last_x, last_y = self.object_positions[class_name]
                x = max(50, min(w-200, last_x + random.randint(-50, 50)))
                y = max(50, min(h-150, last_y + random.randint(-30, 30)))
            else:
                x = random.randint(50, max(51, w - 200))
                y = random.randint(50, max(51, h - 150))
            
            width = random.randint(80, min(200, w - x))
            height = random.randint(60, min(150, h - y))
            
            # 动态目标置信度通常较高
            confidence = random.uniform(max(0.6, self.confidence_threshold), 1.0)
            
            detection = {
                'class': class_name,
                'confidence': confidence,
                'bbox': (x, y, width, height),
                'center': (x + width//2, y + height//2),
                'type': 'moving',
                'detection_time': time.time()
            }
            
            detections.append(detection)
        
        return detections
    
    def _detect_static_objects(self, frame: np.ndarray, h: int, w: int) -> List[Dict]:
        """检测静态目标"""
        detections = []
        num_static = random.randint(1, 3)  # 静态目标通常较多
        
        for i in range(num_static):
            class_name = random.choice(self.static_objects)
            
            # 静态目标位置变化很小
            if class_name in self.object_positions:
                # 基于上次位置，位置变化很小
                last_x, last_y = self.object_positions[class_name]
                x = max(50, min(w-200, last_x + random.randint(-10, 10)))
                y = max(50, min(h-150, last_y + random.randint(-5, 5)))
            else:
                x = random.randint(50, max(51, w - 200))
                y = random.randint(50, max(51, h - 150))
            
            width = random.randint(60, min(150, w - x))
            height = random.randint(40, min(120, h - y))
            
            # 静态目标置信度相对稳定
            confidence = random.uniform(self.confidence_threshold, 0.9)
            
            detection = {
                'class': class_name,
                'confidence': confidence,
                'bbox': (x, y, width, height),
                'center': (x + width//2, y + height//2),
                'type': 'static',
                'detection_time': time.time()
            }
            
            detections.append(detection)
        
        return detections
    
    def _update_object_tracking(self, detections: List[Dict]):
        """更新目标位置跟踪"""
        for detection in detections:
            class_name = detection['class']
            center = detection['center']
            self.object_positions[class_name] = center


class HybridStableYOLOSGUI:
    """混合稳定版YOLOS GUI - Tkinter界面 + 稳定摄像头处理"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("YOLOS - 混合稳定版目标检测系统")
        self.root.geometry("1200x800")
        
        # 摄像头相关
        self.cap = None
        self.is_camera_running = False
        self.camera_thread = None
        self.current_frame = None
        
        # 检测器
        self.detector = StableYOLOSDetector()
        
        # 检测控制
        self.is_detecting = False
        self.result_hold_frames = 30 # 结果保持30帧（静态目标保持更久）
        
        # 性能统计
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        
        # 结果缓存
        self.cached_results = []
        self.result_cache_time = 0
        
        # 统计信息
        self.stats = {
            'total_detections': 0,
            'session_start': time.time()
        }
        
        # 创建界面
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)
        
        # 右侧显示区域
        display_frame = ttk.LabelFrame(main_frame, text="视频显示")
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_control_panel(control_frame)
        self.setup_display_area(display_frame)
        
    def setup_control_panel(self, parent):
        """设置控制面板"""
        # 摄像头控制
        camera_frame = ttk.LabelFrame(parent, text="摄像头控制")
        camera_frame.pack(fill=tk.X, pady=5)
        
        button_frame = ttk.Frame(camera_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        self.start_btn = ttk.Button(button_frame, text="启动摄像头", 
                                   command=self.start_camera)
        self.start_btn.pack(side=tk.LEFT, padx=2)
        
        self.stop_btn = ttk.Button(button_frame, text="停止摄像头", 
                                  command=self.stop_camera, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2)
        
        # 检测控制
        detection_frame = ttk.LabelFrame(parent, text="检测控制")
        detection_frame.pack(fill=tk.X, pady=5)
        
        self.detect_btn = ttk.Button(detection_frame, text="开始检测", 
                                    command=self.toggle_detection, state=tk.DISABLED)
        self.detect_btn.pack(pady=5)
        
        # 参数设置
        params_frame = ttk.LabelFrame(parent, text="检测参数")
        params_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(params_frame, text="置信度阈值:").pack(anchor=tk.W)
        self.conf_var = tk.DoubleVar(value=0.5)
        conf_scale = ttk.Scale(params_frame, from_=0.1, to=1.0, 
                              variable=self.conf_var, orient=tk.HORIZONTAL,
                              command=self.update_parameters)
        conf_scale.pack(fill=tk.X, pady=2)
        
        self.conf_label = ttk.Label(params_frame, text="0.50")
        self.conf_label.pack(anchor=tk.W)
        
        ttk.Label(params_frame, text="NMS阈值:").pack(anchor=tk.W, pady=(10,0))
        self.nms_var = tk.DoubleVar(value=0.4)
        nms_scale = ttk.Scale(params_frame, from_=0.1, to=1.0, 
                             variable=self.nms_var, orient=tk.HORIZONTAL,
                             command=self.update_parameters)
        nms_scale.pack(fill=tk.X, pady=2)
        
        self.nms_label = ttk.Label(params_frame, text="0.40")
        self.nms_label.pack(anchor=tk.W)
        
        # 检测频率控制
        frequency_frame = ttk.LabelFrame(parent, text="智能检测频率")
        frequency_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(frequency_frame, text="动态目标间隔:").pack(anchor=tk.W)
        self.moving_interval_var = tk.IntVar(value=2)
        moving_scale = ttk.Scale(frequency_frame, from_=1, to=10, 
                                variable=self.moving_interval_var, orient=tk.HORIZONTAL,
                                command=self.update_detection_intervals)
        moving_scale.pack(fill=tk.X, pady=2)
        
        self.moving_label = ttk.Label(frequency_frame, text="2帧")
        self.moving_label.pack(anchor=tk.W)
        
        ttk.Label(frequency_frame, text="静态目标间隔:").pack(anchor=tk.W, pady=(5,0))
        self.static_interval_var = tk.IntVar(value=10)
        static_scale = ttk.Scale(frequency_frame, from_=5, to=30, 
                                variable=self.static_interval_var, orient=tk.HORIZONTAL,
                                command=self.update_detection_intervals)
        static_scale.pack(fill=tk.X, pady=2)
        
        self.static_label = ttk.Label(frequency_frame, text="10帧")
        self.static_label.pack(anchor=tk.W)
        
        # 状态信息
        status_frame = ttk.LabelFrame(parent, text="状态信息")
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_text = tk.Text(status_frame, height=10, width=30)
        scrollbar = ttk.Scrollbar(status_frame, orient=tk.VERTICAL, 
                                 command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=scrollbar.set)
        
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_message("YOLOS混合稳定版系统已启动")
        
    def setup_display_area(self, parent):
        """设置显示区域"""
        # 创建Canvas用于显示视频
        self.canvas = tk.Canvas(parent, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 显示提示信息
        self.canvas.create_text(400, 300, text="请启动摄像头开始检测", 
                               fill='white', font=('Arial', 16))
        
    def log_message(self, message):
        """记录日志信息"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.status_text.insert(tk.END, log_entry)
        self.status_text.see(tk.END)
        
        # 限制日志长度
        lines = self.status_text.get("1.0", tk.END).split('\n')
        if len(lines) > 100:
            self.status_text.delete("1.0", "10.0")
            
    def update_parameters(self, value=None):
        """更新检测参数"""
        conf = self.conf_var.get()
        nms = self.nms_var.get()
        
        self.conf_label.config(text=f"{conf:.2f}")
        self.nms_label.config(text=f"{nms:.2f}")
        
        # 更新检测器参数
        self.detector.update_parameters(conf, nms)
        
    def update_detection_intervals(self, value=None):
        """更新检测间隔"""
        moving_interval = int(self.moving_interval_var.get())
        static_interval = int(self.static_interval_var.get())
        
        self.moving_label.config(text=f"{moving_interval}帧")
        self.static_label.config(text=f"{static_interval}帧")
        
        # 更新检测器间隔
        self.detector.moving_detection_interval = moving_interval
        self.detector.static_detection_interval = static_interval
        
        self.log_message(f"检测间隔已更新: 动态{moving_interval}帧, 静态{static_interval}帧")
        
    def initialize_camera(self) -> bool:
        """初始化摄像头 - 使用稳定的初始化逻辑"""
        logger.info("正在初始化摄像头...")
        
        # 首先尝试内置摄像头 (index 0)
        logger.info("尝试内置摄像头 (索引 0)")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                self.cap = cap
                logger.info(f"内置摄像头启动成功: {frame.shape}")
                self.log_message("内置摄像头启动成功")
                return True
            else:
                cap.release()
        
        # 如果内置摄像头失败，尝试外部摄像头
        for index in [1, 2, 3, 4]:
            logger.info(f"尝试摄像头索引 {index}")
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.cap = cap
                    logger.info(f"摄像头 {index} 启动成功: {frame.shape}")
                    self.log_message(f"摄像头 {index} 启动成功")
                    return True
                else:
                    cap.release()
            else:
                cap.release()
        
        logger.error("没有可用的摄像头")
        self.log_message("错误: 没有可用的摄像头")
        return False
    
    def start_camera(self):
        """启动摄像头"""
        if not self.initialize_camera():
            messagebox.showerror("错误", "无法启动摄像头")
            return
            
        # 设置摄像头参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.is_camera_running = True
        self.start_time = time.time()
        self.frame_count = 0
        
        # 更新按钮状态
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.detect_btn.config(state=tk.NORMAL)
        
        # 启动摄像头线程
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
        
        self.log_message("摄像头已启动")
        
    def stop_camera(self):
        """停止摄像头"""
        self.is_camera_running = False
        self.is_detecting = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
            
        # 等待线程结束
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2.0)
        
        # 更新按钮状态
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.detect_btn.config(state=tk.DISABLED)
        
        # 清空显示
        self.canvas.delete("all")
        self.canvas.create_text(400, 300, text="摄像头已停止", 
                               fill='white', font=('Arial', 16))
        
        self.log_message("摄像头已停止")
        
    def toggle_detection(self):
        """切换检测状态"""
        self.is_detecting = not self.is_detecting
        
        if self.is_detecting:
            self.detect_btn.config(text="停止检测")
            self.log_message("开始目标检测")
        else:
            self.detect_btn.config(text="开始检测")
            self.log_message("停止目标检测")
            
    def camera_loop(self):
        """摄像头循环线程 - 使用稳定的处理逻辑"""
        while self.is_camera_running:
            try:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        self.frame_count += 1
                        self.current_frame = frame.copy()
                        
                        # 计算FPS
                        if self.frame_count % 30 == 0:
                            elapsed = time.time() - self.start_time
                            self.fps = self.frame_count / elapsed if elapsed > 0 else 0
                        
                        # 智能检测控制
                        if self.is_detecting:
                            results = self.detector.detect_objects(frame, self.frame_count)
                            if results:
                                self.cached_results = results
                                self.result_cache_time = self.frame_count
                                self.root.after(0, lambda: self.log_detections(results))
                        
                        # 使用缓存结果
                        display_results = []
                        if self.cached_results and (self.frame_count - self.result_cache_time) < self.result_hold_frames:
                            display_results = self.cached_results
                        
                        # 绘制检测结果
                        if display_results:
                            frame = self.draw_detections(frame, display_results)
                        
                        # 绘制信息
                        frame = self.draw_info_overlay(frame, display_results)
                        
                        # 在主线程中显示帧
                        self.root.after(0, lambda f=frame: self.display_frame(f))
                        
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"摄像头循环错误: {e}")
                self.root.after(0, lambda: self.log_message(f"摄像头错误: {e}"))
                break
                
    def draw_detections(self, frame: np.ndarray, results: List[Dict]) -> np.ndarray:
        """绘制检测结果 - 区分动态和静态目标"""
        for detection in results:
            class_name = detection['class']
            confidence = detection['confidence']
            x, y, w, h = detection['bbox']
            obj_type = detection.get('type', 'unknown')
            
            # 根据目标类型和置信度选择颜色
            if obj_type == 'moving':
                # 动态目标使用暖色调
                if confidence > 0.8:
                    color = (0, 255, 0)    # 绿色 - 高置信度动态目标
                elif confidence > 0.6:
                    color = (0, 255, 255)  # 黄色 - 中等置信度动态目标
                else:
                    color = (0, 165, 255)  # 橙色 - 低置信度动态目标
                thickness = 3  # 动态目标边框更粗
            else:
                # 静态目标使用冷色调
                if confidence > 0.8:
                    color = (255, 0, 0)    # 蓝色 - 高置信度静态目标
                elif confidence > 0.6:
                    color = (255, 255, 0)  # 青色 - 中等置信度静态目标
                else:
                    color = (255, 0, 255)  # 紫色 - 低置信度静态目标
                thickness = 2  # 静态目标边框较细
            
            # 绘制边界框
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # 绘制标签
            type_indicator = "🏃" if obj_type == 'moving' else "📦"
            label = f"{type_indicator}{class_name}: {confidence:.2f}"
            
            # 标签背景
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x, y - label_h - 10), (x + label_w + 4, y), color, -1)
            cv2.putText(frame, label, (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 绘制中心点 - 动态目标用圆形，静态目标用方形
            center_x, center_y = detection['center']
            if obj_type == 'moving':
                cv2.circle(frame, (center_x, center_y), 4, color, -1)
            else:
                cv2.rectangle(frame, (center_x-3, center_y-3), (center_x+3, center_y+3), color, -1)
        
        return frame
    
    def draw_info_overlay(self, frame: np.ndarray, results: List[Dict]) -> np.ndarray:
        """绘制信息覆盖层"""
        # 左上角信息面板 - 扩大面板
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (350, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 统计不同类型的目标
        moving_count = len([r for r in results if r.get('type') == 'moving'])
        static_count = len([r for r in results if r.get('type') == 'static'])
        
        # 显示信息
        font_scale = 0.4
        thickness = 1
        y_offset = 20
        
        cv2.putText(frame, f"总检测: {len(results)} (动态:{moving_count} 静态:{static_count})", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        cv2.putText(frame, f"检测状态: {'智能检测开启' if self.is_detecting else '检测关闭'}", 
                   (10, y_offset + 18), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0) if self.is_detecting else (0, 0, 255), thickness)
        
        cv2.putText(frame, f"FPS: {self.fps:.1f} | 帧数: {self.frame_count}", 
                   (10, y_offset + 36), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        cv2.putText(frame, f"置信度: {self.conf_var.get():.2f} | NMS: {self.nms_var.get():.2f}", 
                   (10, y_offset + 54), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), thickness)
        
        # 显示检测间隔策略
        moving_interval = getattr(self, 'moving_interval_var', None)
        static_interval = getattr(self, 'static_interval_var', None)
        if moving_interval and static_interval:
            cv2.putText(frame, f"检测间隔 - 动态:{moving_interval.get()}帧 静态:{static_interval.get()}帧", 
                       (10, y_offset + 72), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
        
        # 显示当前检测的目标类型
        if results:
            detected_classes = list(set([r['class'] for r in results]))
            class_text = "当前目标: " + ", ".join(detected_classes[:3])  # 最多显示3个
            if len(detected_classes) > 3:
                class_text += f" +{len(detected_classes)-3}个"
            cv2.putText(frame, class_text, 
                       (10, y_offset + 90), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 165, 0), thickness)
        
        return frame
        
    def display_frame(self, frame):
        """显示帧到Canvas"""
        try:
            # 调整帧大小以适应Canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                # 计算缩放比例
                h, w = frame.shape[:2]
                scale = min(canvas_width/w, canvas_height/h)
                
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                # 调整大小
                frame_resized = cv2.resize(frame, (new_w, new_h))
                
                # 转换颜色格式
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                
                # 转换为PIL图像
                pil_image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(pil_image)
                
                # 清空Canvas并显示新图像
                self.canvas.delete("all")
                
                # 居中显示
                x = (canvas_width - new_w) // 2
                y = (canvas_height - new_h) // 2
                
                self.canvas.create_image(x, y, anchor=tk.NW, image=photo)
                self.canvas.image = photo  # 保持引用
                
        except Exception as e:
            logger.error(f"显示帧错误: {e}")
            
    def log_detections(self, results: List[Dict]):
        """记录检测结果"""
        if results:
            for detection in results:
                class_name = detection['class']
                confidence = detection['confidence']
                
                log_msg = f"检测到 {class_name} (置信度: {confidence:.3f})"
                self.log_message(log_msg)
            
            # 更新统计
            self.stats['total_detections'] += len(results)
    
    def on_closing(self):
        """关闭程序时的清理工作"""
        self.is_camera_running = False
        self.is_detecting = False
        
        if self.cap:
            self.cap.release()
            
        if hasattr(self, 'camera_thread') and self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2.0)
            
        self.root.destroy()
        
    def run(self):
        """运行GUI"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.on_closing()


def main():
    """主函数"""
    print("🎯 YOLOS 混合稳定版目标检测系统")
    print("结合稳定摄像头处理和Tkinter界面")
    print("=" * 45)
    print("功能特性:")
    print("  🎥 稳定的摄像头处理")
    print("  🖥️ Tkinter图形界面")
    print("  🎯 实时目标检测模拟")
    print("  ⚙️ 动态参数调整")
    print("  📊 性能监控")
    print()
    
    try:
        app = HybridStableYOLOSGUI()
        app.run()
        
    except Exception as e:
        logger.error(f"启动系统失败: {e}")
        print(f"❌ 错误: {e}")


if __name__ == "__main__":
    main()