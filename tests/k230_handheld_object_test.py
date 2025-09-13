#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOS K230手持物体识别GUI测试
专门针对YAHBOOM K230设备的实时物体识别测试
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
import time
import threading
from datetime import datetime
from pathlib import Path
import json
from PIL import Image, ImageTk
import sys
import os

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ Ultralytics YOLO不可用，将使用OpenCV检测")

class K230HandheldObjectGUI:
    """K230手持物体识别GUI"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("YOLOS K230 手持物体识别测试")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # 检测状态
        self.is_detecting = False
        self.camera = None
        self.detection_thread = None
        
        # 模型相关
        self.models = {}
        self.current_model = None
        self.model_names = []
        
        # 统计数据
        self.detection_stats = {
            'total_detections': 0,
            'session_start': datetime.now(),
            'fps': 0,
            'frame_count': 0,
            'last_fps_time': time.time()
        }
        
        # 检测结果
        self.current_detections = []
        self.detection_history = []
        
        # 初始化界面
        self.setup_gui()
        self.load_available_models()
        self.check_camera_devices()
        
        print("🎯 YOLOS K230手持物体识别GUI已启动")
    
    def setup_gui(self):
        """设置GUI界面"""
        # 主标题
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill=tk.X, padx=10, pady=5)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="🎯 YOLOS K230 手持物体识别测试", 
                              font=('Arial', 18, 'bold'), fg='#ecf0f1', bg='#2c3e50')
        title_label.pack(expand=True)
        
        # 主容器
        main_container = tk.Frame(self.root, bg='#2c3e50')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 左侧控制面板
        self.setup_control_panel(main_container)
        
        # 右侧显示区域
        self.setup_display_area(main_container)
        
        # 底部状态栏
        self.setup_status_bar()
    
    def setup_control_panel(self, parent):
        """设置控制面板"""
        control_frame = tk.Frame(parent, bg='#34495e', width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)
        
        # 设备状态
        device_frame = tk.LabelFrame(control_frame, text="📱 设备状态", 
                                   font=('Arial', 12, 'bold'), fg='#ecf0f1', bg='#34495e')
        device_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.device_status_label = tk.Label(device_frame, text="检查中...", 
                                          font=('Arial', 10), fg='#f39c12', bg='#34495e')
        self.device_status_label.pack(pady=5)
        
        # 摄像头选择
        camera_frame = tk.LabelFrame(control_frame, text="📷 摄像头设置", 
                                   font=('Arial', 12, 'bold'), fg='#ecf0f1', bg='#34495e')
        camera_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(camera_frame, text="摄像头ID:", font=('Arial', 10), 
                fg='#ecf0f1', bg='#34495e').pack(anchor=tk.W, padx=5)
        
        self.camera_var = tk.StringVar(value="0")
        camera_combo = ttk.Combobox(camera_frame, textvariable=self.camera_var, 
                                  values=["0", "1", "2", "USB", "K230"], width=15)
        camera_combo.pack(padx=5, pady=5)
        
        # 模型选择
        model_frame = tk.LabelFrame(control_frame, text="🤖 模型设置", 
                                  font=('Arial', 12, 'bold'), fg='#ecf0f1', bg='#34495e')
        model_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(model_frame, text="检测模型:", font=('Arial', 10), 
                fg='#ecf0f1', bg='#34495e').pack(anchor=tk.W, padx=5)
        
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, 
                                       width=20, state="readonly")
        self.model_combo.pack(padx=5, pady=5)
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_changed)
        
        # 检测参数
        param_frame = tk.LabelFrame(control_frame, text="⚙️ 检测参数", 
                                  font=('Arial', 12, 'bold'), fg='#ecf0f1', bg='#34495e')
        param_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 置信度阈值
        tk.Label(param_frame, text="置信度阈值:", font=('Arial', 10), 
                fg='#ecf0f1', bg='#34495e').pack(anchor=tk.W, padx=5)
        
        self.confidence_var = tk.DoubleVar(value=0.5)
        confidence_scale = tk.Scale(param_frame, from_=0.1, to=0.9, resolution=0.1,
                                  orient=tk.HORIZONTAL, variable=self.confidence_var,
                                  bg='#34495e', fg='#ecf0f1', highlightbackground='#34495e')
        confidence_scale.pack(fill=tk.X, padx=5, pady=2)
        
        # 控制按钮
        button_frame = tk.Frame(control_frame, bg='#34495e')
        button_frame.pack(fill=tk.X, padx=10, pady=20)
        
        self.start_button = tk.Button(button_frame, text="🚀 开始检测", 
                                    font=('Arial', 12, 'bold'), bg='#27ae60', fg='white',
                                    command=self.start_detection, relief=tk.FLAT, pady=8)
        self.start_button.pack(fill=tk.X, pady=2)
        
        self.stop_button = tk.Button(button_frame, text="⏹️ 停止检测", 
                                   font=('Arial', 12, 'bold'), bg='#e74c3c', fg='white',
                                   command=self.stop_detection, relief=tk.FLAT, pady=8,
                                   state=tk.DISABLED)
        self.stop_button.pack(fill=tk.X, pady=2)
        
        self.test_button = tk.Button(button_frame, text="📸 测试图像", 
                                   font=('Arial', 12, 'bold'), bg='#3498db', fg='white',
                                   command=self.test_with_image, relief=tk.FLAT, pady=8)
        self.test_button.pack(fill=tk.X, pady=2)
        
        # 统计信息
        stats_frame = tk.LabelFrame(control_frame, text="📊 检测统计", 
                                  font=('Arial', 12, 'bold'), fg='#ecf0f1', bg='#34495e')
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.stats_text = tk.Text(stats_frame, height=8, width=30, font=('Consolas', 9),
                                bg='#2c3e50', fg='#ecf0f1', insertbackground='#ecf0f1')
        self.stats_text.pack(padx=5, pady=5)
        
        self.update_stats_display()
    
    def setup_display_area(self, parent):
        """设置显示区域"""
        display_frame = tk.Frame(parent, bg='#2c3e50')
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 视频显示
        video_frame = tk.LabelFrame(display_frame, text="📺 实时检测画面", 
                                  font=('Arial', 12, 'bold'), fg='#ecf0f1', bg='#2c3e50')
        video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.video_label = tk.Label(video_frame, bg='#34495e', 
                                  text="等待开始检测...\n\n🎯 YOLOS K230\n手持物体识别", 
                                  font=('Arial', 16), fg='#95a5a6')
        self.video_label.pack(expand=True, padx=10, pady=10)
        
        # 检测结果显示
        result_frame = tk.LabelFrame(display_frame, text="🎯 检测结果", 
                                   font=('Arial', 12, 'bold'), fg='#ecf0f1', bg='#2c3e50')
        result_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.result_text = tk.Text(result_frame, height=6, font=('Consolas', 10),
                                 bg='#34495e', fg='#ecf0f1', insertbackground='#ecf0f1')
        result_scrollbar = tk.Scrollbar(result_frame, command=self.result_text.yview)
        self.result_text.config(yscrollcommand=result_scrollbar.set)
        
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        result_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
    
    def setup_status_bar(self):
        """设置状态栏"""
        status_frame = tk.Frame(self.root, bg='#34495e', height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(status_frame, text="就绪", font=('Arial', 10), 
                                   fg='#ecf0f1', bg='#34495e')
        self.status_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.fps_label = tk.Label(status_frame, text="FPS: 0", font=('Arial', 10), 
                                fg='#ecf0f1', bg='#34495e')
        self.fps_label.pack(side=tk.RIGHT, padx=10, pady=5)
    
    def load_available_models(self):
        """加载可用模型"""
        models_dir = Path("models")
        self.model_names = []
        
        if models_dir.exists():
            # 检查YOLO模型文件
            for model_file in models_dir.glob("yolo*.pt"):
                model_name = model_file.stem
                self.model_names.append(model_name)
                print(f"✅ 发现模型: {model_name}")
        
        # 添加默认模型选项
        if YOLO_AVAILABLE:
            default_models = ['yolov8n', 'yolov8s', 'yolo11n']
            for model in default_models:
                if model not in self.model_names:
                    self.model_names.append(model)
        
        if not self.model_names:
            self.model_names = ['OpenCV检测']
        
        # 更新下拉框
        self.model_combo['values'] = self.model_names
        if self.model_names:
            self.model_combo.set(self.model_names[0])
        
        print(f"📋 可用模型: {', '.join(self.model_names)}")
    
    def check_camera_devices(self):
        """检查摄像头设备"""
        available_cameras = []
        
        # 检查USB摄像头
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(str(i))
                cap.release()
        
        # 更新设备状态
        if available_cameras:
            status_text = f"✅ 发现摄像头: {', '.join(available_cameras)}"
            self.device_status_label.config(text=status_text, fg='#27ae60')
        else:
            status_text = "⚠️ 未检测到USB摄像头\n将使用测试图像模式"
            self.device_status_label.config(text=status_text, fg='#f39c12')
        
        # 添加K230选项
        available_cameras.extend(['USB', 'K230'])
        
        # 更新摄像头选择
        current_cameras = list(self.camera_var.get() if hasattr(self, 'camera_var') else ['0'])
        current_cameras.extend(available_cameras)
        
        print(f"📷 可用摄像头: {available_cameras}")
    
    def on_model_changed(self, event=None):
        """模型选择改变"""
        model_name = self.model_var.get()
        self.load_model(model_name)
    
    def load_model(self, model_name):
        """加载指定模型"""
        try:
            if model_name == 'OpenCV检测':
                self.current_model = None
                self.update_status(f"使用OpenCV检测方法")
                return True
            
            if not YOLO_AVAILABLE:
                self.update_status("YOLO不可用，使用OpenCV检测")
                return False
            
            # 检查本地模型文件
            model_path = Path(f"models/{model_name}.pt")
            if model_path.exists():
                self.current_model = YOLO(str(model_path))
                self.update_status(f"✅ 已加载本地模型: {model_name}")
            else:
                # 尝试下载预训练模型
                self.update_status(f"📥 加载模型: {model_name}")
                self.current_model = YOLO(f"{model_name}.pt")
                self.update_status(f"✅ 已加载模型: {model_name}")
            
            return True
            
        except Exception as e:
            self.update_status(f"❌ 模型加载失败: {e}")
            self.current_model = None
            return False
    
    def start_detection(self):
        """开始检测"""
        if self.is_detecting:
            return
        
        # 加载模型
        model_name = self.model_var.get()
        if not self.load_model(model_name):
            messagebox.showerror("错误", "模型加载失败")
            return
        
        # 初始化摄像头
        camera_id = self.camera_var.get()
        if not self.init_camera(camera_id):
            messagebox.showerror("错误", "摄像头初始化失败")
            return
        
        # 启动检测线程
        self.is_detecting = True
        self.detection_stats['session_start'] = datetime.now()
        self.detection_stats['frame_count'] = 0
        self.detection_stats['total_detections'] = 0
        
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()
        
        # 更新按钮状态
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        self.update_status("🚀 检测已启动")
    
    def stop_detection(self):
        """停止检测"""
        self.is_detecting = False
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        # 更新按钮状态
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
        self.update_status("⏹️ 检测已停止")
        
        # 显示停止画面
        self.video_label.config(image='', text="检测已停止\n\n🎯 YOLOS K230\n手持物体识别", 
                              font=('Arial', 16), fg='#95a5a6')
    
    def init_camera(self, camera_id):
        """初始化摄像头"""
        try:
            if camera_id in ['USB', 'K230']:
                # 尝试不同的摄像头ID
                for i in range(5):
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        self.camera = cap
                        break
                    cap.release()
            else:
                self.camera = cv2.VideoCapture(int(camera_id))
            
            if not self.camera or not self.camera.isOpened():
                return False
            
            # 设置摄像头参数
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            return True
            
        except Exception as e:
            print(f"摄像头初始化错误: {e}")
            return False
    
    def detection_loop(self):
        """检测循环"""
        while self.is_detecting:
            try:
                if self.camera:
                    ret, frame = self.camera.read()
                    if not ret:
                        continue
                else:
                    # 使用测试图像
                    frame = self.create_test_frame()
                
                # 执行检测
                detections = self.detect_objects(frame)
                
                # 绘制结果
                display_frame = self.draw_detections(frame, detections)
                
                # 更新显示
                self.update_video_display(display_frame)
                self.update_detection_results(detections)
                
                # 更新统计
                self.update_detection_stats(detections)
                
                time.sleep(0.03)  # 约30 FPS
                
            except Exception as e:
                print(f"检测循环错误: {e}")
                break
    
    def detect_objects(self, frame):
        """检测物体"""
        detections = []
        
        try:
            if self.current_model:
                # 使用YOLO模型检测
                results = self.current_model(frame, conf=self.confidence_var.get())
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            cls = int(box.cls[0].cpu().numpy())
                            
                            # 获取类别名称
                            class_name = self.current_model.names[cls] if hasattr(self.current_model, 'names') else f"class_{cls}"
                            
                            detections.append({
                                'class': class_name,
                                'confidence': float(conf),
                                'bbox': [int(x1), int(y1), int(x2), int(y2)]
                            })
            else:
                # 使用OpenCV检测方法
                detections = self.opencv_detection(frame)
                
        except Exception as e:
            print(f"检测错误: {e}")
        
        return detections
    
    def opencv_detection(self, frame):
        """OpenCV检测方法"""
        detections = []
        
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 轮廓检测
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # 过滤小轮廓
                x, y, w, h = cv2.boundingRect(contour)
                
                # 计算置信度（基于面积）
                confidence = min(area / 10000, 1.0)
                
                if confidence > self.confidence_var.get():
                    detections.append({
                        'class': 'object',
                        'confidence': confidence,
                        'bbox': [x, y, x+w, y+h]
                    })
        
        return detections
    
    def draw_detections(self, frame, detections):
        """绘制检测结果"""
        display_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            confidence = det['confidence']
            
            # 绘制边界框
            color = (0, 255, 0)  # 绿色
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # 标签背景
            cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # 标签文字
            cv2.putText(display_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 添加信息文字
        info_text = f"检测到 {len(detections)} 个物体"
        cv2.putText(display_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return display_frame
    
    def create_test_frame(self):
        """创建测试帧"""
        # 创建一个带有几何图形的测试图像
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 添加一些移动的几何图形
        t = time.time()
        
        # 移动的圆形
        center_x = int(320 + 100 * np.sin(t))
        center_y = int(240 + 50 * np.cos(t))
        cv2.circle(frame, (center_x, center_y), 50, (0, 255, 0), -1)
        
        # 移动的矩形
        rect_x = int(100 + 50 * np.sin(t * 0.5))
        rect_y = int(100 + 30 * np.cos(t * 0.5))
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + 100, rect_y + 80), (255, 0, 0), -1)
        
        # 添加文字
        cv2.putText(frame, "K230 Test Mode", (200, 400), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame
    
    def update_video_display(self, frame):
        """更新视频显示"""
        try:
            # 调整图像大小
            height, width = frame.shape[:2]
            max_width, max_height = 600, 400
            
            if width > max_width or height > max_height:
                scale = min(max_width/width, max_height/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # 转换为PIL图像
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(pil_image)
            
            # 更新显示
            self.video_label.config(image=photo, text="")
            self.video_label.image = photo  # 保持引用
            
        except Exception as e:
            print(f"视频显示更新错误: {e}")
    
    def update_detection_results(self, detections):
        """更新检测结果显示"""
        try:
            # 清空之前的结果
            self.result_text.delete(1.0, tk.END)
            
            if detections:
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.result_text.insert(tk.END, f"[{timestamp}] 检测结果:\n")
                
                for i, det in enumerate(detections, 1):
                    result_line = f"{i}. {det['class']}: {det['confidence']:.2f}\n"
                    self.result_text.insert(tk.END, result_line)
                
                self.result_text.insert(tk.END, f"\n总计: {len(detections)} 个物体\n")
            else:
                self.result_text.insert(tk.END, "未检测到物体\n")
            
            # 滚动到底部
            self.result_text.see(tk.END)
            
        except Exception as e:
            print(f"结果显示更新错误: {e}")
    
    def update_detection_stats(self, detections):
        """更新检测统计"""
        self.detection_stats['frame_count'] += 1
        self.detection_stats['total_detections'] += len(detections)
        
        # 计算FPS
        current_time = time.time()
        if current_time - self.detection_stats['last_fps_time'] >= 1.0:
            self.detection_stats['fps'] = self.detection_stats['frame_count'] / (current_time - self.detection_stats['last_fps_time'])
            self.detection_stats['frame_count'] = 0
            self.detection_stats['last_fps_time'] = current_time
            
            # 更新FPS显示
            self.fps_label.config(text=f"FPS: {self.detection_stats['fps']:.1f}")
        
        # 更新统计显示
        self.update_stats_display()
    
    def update_stats_display(self):
        """更新统计显示"""
        try:
            self.stats_text.delete(1.0, tk.END)
            
            session_time = datetime.now() - self.detection_stats['session_start']
            
            stats_info = f"""会话时间: {str(session_time).split('.')[0]}
总检测数: {self.detection_stats['total_detections']}
当前FPS: {self.detection_stats['fps']:.1f}
模型: {self.model_var.get()}
摄像头: {self.camera_var.get()}
置信度: {self.confidence_var.get():.1f}

状态: {'🟢 运行中' if self.is_detecting else '🔴 已停止'}"""
            
            self.stats_text.insert(tk.END, stats_info)
            
        except Exception as e:
            print(f"统计显示更新错误: {e}")
    
    def test_with_image(self):
        """使用图像测试"""
        file_path = filedialog.askopenfilename(
            title="选择测试图像",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            try:
                # 读取图像
                frame = cv2.imread(file_path)
                if frame is None:
                    messagebox.showerror("错误", "无法读取图像文件")
                    return
                
                # 加载模型
                model_name = self.model_var.get()
                if not self.load_model(model_name):
                    messagebox.showerror("错误", "模型加载失败")
                    return
                
                # 执行检测
                detections = self.detect_objects(frame)
                
                # 绘制结果
                display_frame = self.draw_detections(frame, detections)
                
                # 更新显示
                self.update_video_display(display_frame)
                self.update_detection_results(detections)
                
                self.update_status(f"✅ 图像测试完成，检测到 {len(detections)} 个物体")
                
            except Exception as e:
                messagebox.showerror("错误", f"图像测试失败: {e}")
    
    def update_status(self, message):
        """更新状态栏"""
        self.status_label.config(text=message)
        print(message)
    
    def run(self):
        """运行GUI"""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()
        except KeyboardInterrupt:
            self.on_closing()
    
    def on_closing(self):
        """关闭程序"""
        if self.is_detecting:
            self.stop_detection()
        
        self.root.quit()
        self.root.destroy()

def main():
    """主函数"""
    print("🚀 启动YOLOS K230手持物体识别GUI测试")
    
    try:
        app = K230HandheldObjectGUI()
        app.run()
    except Exception as e:
        print(f"程序运行错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()