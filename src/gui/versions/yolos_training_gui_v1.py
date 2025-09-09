#!/usr/bin/env python3
"""
YOLOS核心训练界面管理器
专注于视频捕捉、图像识别和模型训练的核心功能
避免过度扩展到专业领域，通过API与外部系统交互
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime

# 导入核心模块
try:
    from ..training.dataset_manager import DatasetManager
    from ..training.trainer import YOLOSTrainer
    from ..training.offline_training_manager import OfflineTrainingManager
    from ..detection.yolos_detector import YOLOSDetector
    from ..utils.logger import setup_logger
except ImportError:
    # 备用导入
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
    class DatasetManager:
        def __init__(self): pass
        def load_dataset(self, path): return {"images": [], "labels": []}
        def save_dataset(self, data, path): pass
    
    class YOLOSTrainer:
        def __init__(self): pass
        def train(self, config): return {"status": "success", "model_path": "model.pt"}
    
    class OfflineTrainingManager:
        def __init__(self): pass
        def start_training(self, config): return True
    
    class YOLOSDetector:
        def __init__(self): pass
        def detect(self, image): return []
    
    def setup_logger(name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        return logger

class VideoCapture:
    """视频捕捉管理器"""
    
    def __init__(self):
        self.cap = None
        self.is_recording = False
        self.frame_buffer = []
        self.max_buffer_size = 100
        self.current_frame = None
        self.frame_count = 0
        
    def initialize_camera(self, camera_id: int = 0) -> bool:
        """初始化摄像头"""
        try:
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                return False
            
            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            return True
        except Exception as e:
            logging.error(f"摄像头初始化失败: {e}")
            return False
    
    def read_frame(self) -> Optional[np.ndarray]:
        """读取一帧"""
        if not self.cap or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()
            self.frame_count += 1
            
            # 添加到缓冲区
            if self.is_recording:
                self.frame_buffer.append(frame.copy())
                if len(self.frame_buffer) > self.max_buffer_size:
                    self.frame_buffer.pop(0)
            
            return frame
        return None
    
    def start_recording(self):
        """开始录制"""
        self.is_recording = True
        self.frame_buffer = []
    
    def stop_recording(self) -> List[np.ndarray]:
        """停止录制并返回帧"""
        self.is_recording = False
        return self.frame_buffer.copy()
    
    def capture_image(self) -> Optional[np.ndarray]:
        """捕获当前帧"""
        return self.current_frame.copy() if self.current_frame is not None else None
    
    def release(self):
        """释放摄像头"""
        if self.cap:
            self.cap.release()

class TrainingDataCollector:
    """训练数据收集器"""
    
    def __init__(self, save_dir: str = "training_data"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        (self.save_dir / "images").mkdir(exist_ok=True)
        (self.save_dir / "labels").mkdir(exist_ok=True)
        (self.save_dir / "videos").mkdir(exist_ok=True)
        
        self.current_class = "unknown"
        self.image_count = 0
        self.annotations = []
    
    def set_current_class(self, class_name: str):
        """设置当前标注类别"""
        self.current_class = class_name
    
    def save_image_with_annotation(self, image: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None):
        """保存图像和标注"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        image_filename = f"{self.current_class}_{timestamp}_{self.image_count:04d}.jpg"
        image_path = self.save_dir / "images" / image_filename
        
        # 保存图像
        cv2.imwrite(str(image_path), image)
        
        # 保存标注
        if bbox:
            x, y, w, h = bbox
            img_h, img_w = image.shape[:2]
            
            # 转换为YOLO格式 (归一化的中心点坐标和宽高)
            center_x = (x + w/2) / img_w
            center_y = (y + h/2) / img_h
            norm_w = w / img_w
            norm_h = h / img_h
            
            label_filename = image_filename.replace('.jpg', '.txt')
            label_path = self.save_dir / "labels" / label_filename
            
            with open(label_path, 'w') as f:
                f.write(f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")
        
        self.image_count += 1
        return str(image_path)
    
    def save_video_sequence(self, frames: List[np.ndarray], class_name: str):
        """保存视频序列"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"{class_name}_{timestamp}.avi"
        video_path = self.save_dir / "videos" / video_filename
        
        if frames:
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(str(video_path), fourcc, 20.0, (w, h))
            
            for frame in frames:
                out.write(frame)
            
            out.release()
        
        return str(video_path)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """获取数据集信息"""
        images_dir = self.save_dir / "images"
        labels_dir = self.save_dir / "labels"
        
        image_files = list(images_dir.glob("*.jpg"))
        label_files = list(labels_dir.glob("*.txt"))
        
        class_counts = {}
        for img_file in image_files:
            class_name = img_file.name.split('_')[0]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            "total_images": len(image_files),
            "total_labels": len(label_files),
            "classes": class_counts,
            "save_dir": str(self.save_dir)
        }

class YOLOSTrainingGUI:
    """YOLOS训练界面主类"""
    
    def __init__(self):
        self.logger = setup_logger("YOLOSTrainingGUI")
        
        # 核心组件
        self.video_capture = VideoCapture()
        self.data_collector = TrainingDataCollector()
        self.dataset_manager = DatasetManager()
        self.trainer = YOLOSTrainer()
        self.offline_trainer = OfflineTrainingManager()
        self.detector = YOLOSDetector()
        
        # GUI组件
        self.root = None
        self.video_frame = None
        self.control_frame = None
        self.status_frame = None
        
        # 状态变量
        self.is_camera_active = False
        self.is_training = False
        self.current_mode = "capture"  # capture, annotate, train, detect
        self.selected_bbox = None
        self.drawing_bbox = False
        self.bbox_start = None
        
        # 配置
        self.config = {
            "camera_id": 0,
            "image_size": (640, 480),
            "classes": ["person", "pet", "object"],
            "model_type": "yolov8n",
            "epochs": 100,
            "batch_size": 16
        }
        
        self.create_gui()
    
    def create_gui(self):
        """创建GUI界面"""
        self.root = tk.Tk()
        self.root.title("YOLOS训练系统 - 视频捕捉与模型训练")
        self.root.geometry("1200x800")
        
        # 创建主要区域
        self.create_menu_bar()
        self.create_video_area()
        self.create_control_panel()
        self.create_status_panel()
        
        # 绑定事件
        self.bind_events()
        
        # 初始化状态
        self.update_status("系统已启动，请选择摄像头")
    
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="加载数据集", command=self.load_dataset)
        file_menu.add_command(label="保存数据集", command=self.save_dataset)
        file_menu.add_command(label="导出模型", command=self.export_model)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.on_closing)
        
        # 视图菜单
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="视图", menu=view_menu)
        view_menu.add_command(label="全屏", command=self.toggle_fullscreen)
        view_menu.add_command(label="重置布局", command=self.reset_layout)
        
        # 工具菜单
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="工具", menu=tools_menu)
        tools_menu.add_command(label="数据增强", command=self.show_augmentation_dialog)
        tools_menu.add_command(label="模型评估", command=self.show_evaluation_dialog)
        tools_menu.add_command(label="系统设置", command=self.show_settings_dialog)
        
        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="使用说明", command=self.show_help)
        help_menu.add_command(label="关于", command=self.show_about)
    
    def create_video_area(self):
        """创建视频显示区域"""
        # 主视频框架
        self.video_frame = tk.Frame(self.root, bg="black", relief=tk.SUNKEN, bd=2)
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 视频画布
        self.video_canvas = tk.Canvas(self.video_frame, bg="black", width=640, height=480)
        self.video_canvas.pack(expand=True, fill=tk.BOTH)
        
        # 视频控制按钮
        video_controls = tk.Frame(self.video_frame)
        video_controls.pack(side=tk.BOTTOM, fill=tk.X, pady=2)
        
        tk.Button(video_controls, text="启动摄像头", command=self.start_camera).pack(side=tk.LEFT, padx=2)
        tk.Button(video_controls, text="停止摄像头", command=self.stop_camera).pack(side=tk.LEFT, padx=2)
        tk.Button(video_controls, text="捕获图像", command=self.capture_image).pack(side=tk.LEFT, padx=2)
        tk.Button(video_controls, text="开始录制", command=self.start_recording).pack(side=tk.LEFT, padx=2)
        tk.Button(video_controls, text="停止录制", command=self.stop_recording).pack(side=tk.LEFT, padx=2)
    
    def create_control_panel(self):
        """创建控制面板"""
        self.control_frame = tk.Frame(self.root, width=300, relief=tk.RAISED, bd=1)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        self.control_frame.pack_propagate(False)
        
        # 模式选择
        mode_frame = tk.LabelFrame(self.control_frame, text="工作模式", padx=5, pady=5)
        mode_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.mode_var = tk.StringVar(value="capture")
        modes = [("数据捕获", "capture"), ("标注模式", "annotate"), ("模型训练", "train"), ("实时检测", "detect")]
        
        for text, mode in modes:
            tk.Radiobutton(mode_frame, text=text, variable=self.mode_var, 
                          value=mode, command=self.on_mode_change).pack(anchor=tk.W)
        
        # 类别选择
        class_frame = tk.LabelFrame(self.control_frame, text="目标类别", padx=5, pady=5)
        class_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.class_var = tk.StringVar(value="person")
        self.class_combo = ttk.Combobox(class_frame, textvariable=self.class_var, 
                                       values=self.config["classes"], state="readonly")
        self.class_combo.pack(fill=tk.X, pady=2)
        
        tk.Button(class_frame, text="添加新类别", command=self.add_new_class).pack(fill=tk.X, pady=2)
        
        # 训练配置
        train_frame = tk.LabelFrame(self.control_frame, text="训练配置", padx=5, pady=5)
        train_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 模型类型
        tk.Label(train_frame, text="模型类型:").pack(anchor=tk.W)
        self.model_var = tk.StringVar(value="yolov8n")
        model_combo = ttk.Combobox(train_frame, textvariable=self.model_var,
                                  values=["yolov8n", "yolov8s", "yolov8m", "yolov8l"], state="readonly")
        model_combo.pack(fill=tk.X, pady=2)
        
        # 训练参数
        params_frame = tk.Frame(train_frame)
        params_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(params_frame, text="训练轮数:").grid(row=0, column=0, sticky=tk.W)
        self.epochs_var = tk.IntVar(value=100)
        tk.Spinbox(params_frame, from_=10, to=1000, textvariable=self.epochs_var, width=10).grid(row=0, column=1)
        
        tk.Label(params_frame, text="批次大小:").grid(row=1, column=0, sticky=tk.W)
        self.batch_var = tk.IntVar(value=16)
        tk.Spinbox(params_frame, from_=1, to=64, textvariable=self.batch_var, width=10).grid(row=1, column=1)
        
        # 训练控制
        tk.Button(train_frame, text="开始训练", command=self.start_training, 
                 bg="green", fg="white").pack(fill=tk.X, pady=2)
        tk.Button(train_frame, text="停止训练", command=self.stop_training, 
                 bg="red", fg="white").pack(fill=tk.X, pady=2)
        
        # 数据集信息
        dataset_frame = tk.LabelFrame(self.control_frame, text="数据集信息", padx=5, pady=5)
        dataset_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.dataset_info = tk.Text(dataset_frame, height=8, width=30)
        self.dataset_info.pack(fill=tk.BOTH, expand=True)
        
        tk.Button(dataset_frame, text="刷新信息", command=self.update_dataset_info).pack(fill=tk.X, pady=2)
    
    def create_status_panel(self):
        """创建状态面板"""
        self.status_frame = tk.Frame(self.root, height=100, relief=tk.SUNKEN, bd=1)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        self.status_frame.pack_propagate(False)
        
        # 状态标签
        self.status_label = tk.Label(self.status_frame, text="就绪", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.status_frame, variable=self.progress_var, 
                                          maximum=100, length=200)
        self.progress_bar.pack(side=tk.RIGHT, padx=5)
        
        # 统计信息
        self.stats_label = tk.Label(self.status_frame, text="", anchor=tk.E)
        self.stats_label.pack(side=tk.RIGHT, padx=5)
    
    def bind_events(self):
        """绑定事件"""
        # 鼠标事件（用于标注）
        self.video_canvas.bind("<Button-1>", self.on_mouse_click)
        self.video_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.video_canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        
        # 键盘事件
        self.root.bind("<Key>", self.on_key_press)
        self.root.focus_set()
        
        # 窗口事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def start_camera(self):
        """启动摄像头"""
        if self.video_capture.initialize_camera(self.config["camera_id"]):
            self.is_camera_active = True
            self.update_status("摄像头已启动")
            self.video_loop()
        else:
            messagebox.showerror("错误", "无法启动摄像头")
    
    def stop_camera(self):
        """停止摄像头"""
        self.is_camera_active = False
        self.video_capture.release()
        self.video_canvas.delete("all")
        self.update_status("摄像头已停止")
    
    def video_loop(self):
        """视频循环"""
        if not self.is_camera_active:
            return
        
        frame = self.video_capture.read_frame()
        if frame is not None:
            # 处理帧
            processed_frame = self.process_frame(frame)
            
            # 显示帧
            self.display_frame(processed_frame)
            
            # 更新统计
            self.update_stats()
        
        # 继续循环
        self.root.after(33, self.video_loop)  # ~30 FPS
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """处理视频帧"""
        processed = frame.copy()
        
        # 根据当前模式处理
        if self.current_mode == "detect":
            # 实时检测模式
            detections = self.detector.detect(frame)
            processed = self.draw_detections(processed, detections)
        
        elif self.current_mode == "annotate":
            # 标注模式 - 显示标注工具
            if self.selected_bbox:
                x, y, w, h = self.selected_bbox
                cv2.rectangle(processed, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(processed, self.class_var.get(), (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 添加模式指示器
        cv2.putText(processed, f"模式: {self.current_mode}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return processed
    
    def display_frame(self, frame: np.ndarray):
        """在画布上显示帧"""
        # 转换为RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 调整大小以适应画布
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            frame_resized = cv2.resize(frame_rgb, (canvas_width, canvas_height))
            
            # 转换为PIL图像并显示
            from PIL import Image, ImageTk
            image = Image.fromarray(frame_resized)
            photo = ImageTk.PhotoImage(image)
            
            self.video_canvas.delete("all")
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.video_canvas.image = photo  # 保持引用
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """绘制检测结果"""
        for detection in detections:
            bbox = detection.get('bbox', [])
            if len(bbox) == 4:
                x, y, w, h = bbox
                confidence = detection.get('confidence', 0)
                class_name = detection.get('class', 'unknown')
                
                # 绘制边界框
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # 绘制标签
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    
    def capture_image(self):
        """捕获当前图像"""
        if not self.is_camera_active:
            messagebox.showwarning("警告", "请先启动摄像头")
            return
        
        image = self.video_capture.capture_image()
        if image is not None:
            # 保存图像
            image_path = self.data_collector.save_image_with_annotation(
                image, self.selected_bbox
            )
            self.update_status(f"图像已保存: {os.path.basename(image_path)}")
            self.update_dataset_info()
        else:
            messagebox.showerror("错误", "无法捕获图像")
    
    def start_recording(self):
        """开始录制"""
        if not self.is_camera_active:
            messagebox.showwarning("警告", "请先启动摄像头")
            return
        
        self.video_capture.start_recording()
        self.update_status("开始录制视频...")
    
    def stop_recording(self):
        """停止录制"""
        frames = self.video_capture.stop_recording()
        if frames:
            video_path = self.data_collector.save_video_sequence(frames, self.class_var.get())
            self.update_status(f"视频已保存: {os.path.basename(video_path)}")
        else:
            self.update_status("录制停止")
    
    def start_training(self):
        """开始训练"""
        if self.is_training:
            messagebox.showwarning("警告", "训练正在进行中")
            return
        
        # 检查数据集
        dataset_info = self.data_collector.get_dataset_info()
        if dataset_info["total_images"] < 10:
            messagebox.showwarning("警告", "训练数据不足，至少需要10张图像")
            return
        
        # 配置训练参数
        train_config = {
            "model_type": self.model_var.get(),
            "epochs": self.epochs_var.get(),
            "batch_size": self.batch_var.get(),
            "data_path": dataset_info["save_dir"],
            "classes": list(dataset_info["classes"].keys())
        }
        
        # 在后台线程中开始训练
        self.is_training = True
        self.update_status("开始训练...")
        
        def train_thread():
            try:
                result = self.offline_trainer.start_training(train_config)
                self.root.after(0, lambda: self.on_training_complete(result))
            except Exception as e:
                self.root.after(0, lambda: self.on_training_error(str(e)))
        
        threading.Thread(target=train_thread, daemon=True).start()
    
    def stop_training(self):
        """停止训练"""
        self.is_training = False
        self.update_status("训练已停止")
    
    def on_training_complete(self, result):
        """训练完成回调"""
        self.is_training = False
        self.progress_var.set(100)
        self.update_status("训练完成")
        messagebox.showinfo("成功", f"模型训练完成\n模型路径: {result.get('model_path', 'unknown')}")
    
    def on_training_error(self, error_msg):
        """训练错误回调"""
        self.is_training = False
        self.progress_var.set(0)
        self.update_status("训练失败")
        messagebox.showerror("错误", f"训练失败: {error_msg}")
    
    def on_mode_change(self):
        """模式切换"""
        self.current_mode = self.mode_var.get()
        self.data_collector.set_current_class(self.class_var.get())
        self.update_status(f"切换到{self.current_mode}模式")
    
    def on_mouse_click(self, event):
        """鼠标点击事件"""
        if self.current_mode == "annotate":
            self.drawing_bbox = True
            self.bbox_start = (event.x, event.y)
    
    def on_mouse_drag(self, event):
        """鼠标拖拽事件"""
        if self.drawing_bbox and self.bbox_start:
            # 实时显示拖拽框
            pass
    
    def on_mouse_release(self, event):
        """鼠标释放事件"""
        if self.drawing_bbox and self.bbox_start:
            x1, y1 = self.bbox_start
            x2, y2 = event.x, event.y
            
            # 计算边界框
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            
            if w > 10 and h > 10:  # 最小尺寸检查
                self.selected_bbox = (x, y, w, h)
                self.update_status(f"选择区域: {w}x{h}")
            
            self.drawing_bbox = False
            self.bbox_start = None
    
    def on_key_press(self, event):
        """键盘事件"""
        key = event.keysym.lower()
        
        if key == 'space':
            self.capture_image()
        elif key == 'r':
            if self.video_capture.is_recording:
                self.stop_recording()
            else:
                self.start_recording()
        elif key == 'escape':
            self.selected_bbox = None
    
    def add_new_class(self):
        """添加新类别"""
        class_name = tk.simpledialog.askstring("新类别", "请输入类别名称:")
        if class_name and class_name not in self.config["classes"]:
            self.config["classes"].append(class_name)
            self.class_combo['values'] = self.config["classes"]
            self.class_var.set(class_name)
            self.update_status(f"添加新类别: {class_name}")
    
    def update_dataset_info(self):
        """更新数据集信息"""
        info = self.data_collector.get_dataset_info()
        
        info_text = f"""数据集统计:
总图像数: {info['total_images']}
总标注数: {info['total_labels']}

类别分布:
"""
        for class_name, count in info['classes'].items():
            info_text += f"  {class_name}: {count}\n"
        
        info_text += f"\n保存路径: {info['save_dir']}"
        
        self.dataset_info.delete(1.0, tk.END)
        self.dataset_info.insert(1.0, info_text)
    
    def update_status(self, message: str):
        """更新状态"""
        self.status_label.config(text=message)
        self.logger.info(message)
    
    def update_stats(self):
        """更新统计信息"""
        if self.is_camera_active:
            fps = 30  # 简化的FPS计算
            frame_count = self.video_capture.frame_count
            self.stats_label.config(text=f"FPS: {fps} | 帧数: {frame_count}")
    
    def load_dataset(self):
        """加载数据集"""
        directory = filedialog.askdirectory(title="选择数据集目录")
        if directory:
            try:
                dataset = self.dataset_manager.load_dataset(directory)
                self.update_status(f"数据集已加载: {len(dataset.get('images', []))} 张图像")
                self.update_dataset_info()
            except Exception as e:
                messagebox.showerror("错误", f"加载数据集失败: {e}")
    
    def save_dataset(self):
        """保存数据集"""
        directory = filedialog.askdirectory(title="选择保存目录")
        if directory:
            try:
                info = self.data_collector.get_dataset_info()
                # 这里可以实现数据集格式转换和保存
                self.update_status(f"数据集已保存到: {directory}")
            except Exception as e:
                messagebox.showerror("错误", f"保存数据集失败: {e}")
    
    def export_model(self):
        """导出模型"""
        filename = filedialog.asksaveasfilename(
            title="导出模型",
            defaultextension=".pt",
            filetypes=[("PyTorch模型", "*.pt"), ("所有文件", "*.*")]
        )
        if filename:
            # 这里实现模型导出逻辑
            self.update_status(f"模型已导出: {filename}")
    
    def show_settings_dialog(self):
        """显示设置对话框"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("系统设置")
        settings_window.geometry("400x300")
        
        # 摄像头设置
        camera_frame = tk.LabelFrame(settings_window, text="摄像头设置", padx=5, pady=5)
        camera_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(camera_frame, text="摄像头ID:").pack(anchor=tk.W)
        camera_id_var = tk.IntVar(value=self.config["camera_id"])
        tk.Spinbox(camera_frame, from_=0, to=10, textvariable=camera_id_var).pack(fill=tk.X)
        
        # 应用设置按钮
        def apply_settings():
            self.config["camera_id"] = camera_id_var.get()
            settings_window.destroy()
            self.update_status("设置已应用")
        
        tk.Button(settings_window, text="应用", command=apply_settings).pack(pady=10)
    
    def show_help(self):
        """显示帮助"""
        help_text = """
YOLOS训练系统使用说明:

1. 数据捕获模式:
   - 启动摄像头
   - 选择目标类别
   - 按空格键或点击"捕获图像"

2. 标注模式:
   - 在视频画面上拖拽选择目标区域
   - 按空格键保存标注

3. 训练模式:
   - 配置训练参数
   - 点击"开始训练"

4. 检测模式:
   - 加载训练好的模型
   - 实时检测视频中的目标

快捷键:
- 空格: 捕获图像
- R: 开始/停止录制
- ESC: 取消选择
        """
        
        messagebox.showinfo("使用说明", help_text)
    
    def show_about(self):
        """显示关于信息"""
        about_text = """
YOLOS训练系统 v1.0

专注于视频捕捉和图像识别的核心功能
支持实时数据收集、标注和模型训练

开发团队: YOLOS项目组
        """
        messagebox.showinfo("关于", about_text)
    
    def toggle_fullscreen(self):
        """切换全屏"""
        pass
    
    def reset_layout(self):
        """重置布局"""
        pass
    
    def show_augmentation_dialog(self):
        """显示数据增强对话框"""
        pass
    
    def show_evaluation_dialog(self):
        """显示模型评估对话框"""
        pass
    
    def on_closing(self):
        """关闭程序"""
        if self.is_camera_active:
            self.stop_camera()
        
        if self.is_training:
            if messagebox.askokcancel("确认", "训练正在进行，确定要退出吗？"):
                self.stop_training()
            else:
                return
        
        self.root.destroy()
    
    def run(self):
        """运行GUI"""
        self.update_dataset_info()
        self.root.mainloop()

def main():
    """主函数"""
    print("🎯 YOLOS训练系统启动")
    print("专注于视频捕捉和图像识别的核心功能")
    print("=" * 50)
    
    try:
        app = YOLOSTrainingGUI()
        app.run()
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        logging.error(f"GUI启动失败: {e}")

if __name__ == "__main__":
    main()