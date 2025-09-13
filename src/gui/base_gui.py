#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOS GUI基类
提供统一的GUI框架和通用功能
支持多平台部署和扩展
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
import threading
import time
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Callable
from pathlib import Path
import sys
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaseGUIConfig:
    """GUI配置管理类"""
    
    def __init__(self, config_file: str = "gui_config.json"):
        self.config_file = Path(config_file)
        self.default_config = {
            'window_width': 1200,
            'window_height': 800,
            'camera_index': 0,
            'confidence_threshold': 0.5,
            'nms_threshold': 0.4,
            'model_path': 'models/yolov8n.pt',
            'theme': 'clam',
            'auto_save': True,
            'log_level': 'INFO'
        }
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # 合并默认配置
                merged_config = self.default_config.copy()
                merged_config.update(config)
                return merged_config
        except Exception as e:
            logger.warning(f"加载配置失败: {e}，使用默认配置")
        return self.default_config.copy()
    
    def save_config(self) -> bool:
        """保存配置文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            return False
    
    def get(self, key: str, default=None):
        """获取配置值"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """设置配置值"""
        self.config[key] = value
        if self.config.get('auto_save', True):
            self.save_config()


class CameraManager:
    """摄像头管理类"""
    
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.is_running = False
        self.frame_width = 640
        self.frame_height = 480
        self.fps = 30
    
    def initialize(self) -> bool:
        """初始化摄像头"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                logger.error(f"无法打开摄像头 {self.camera_index}")
                return False
            
            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            logger.info(f"摄像头 {self.camera_index} 初始化成功")
            return True
        except Exception as e:
            logger.error(f"摄像头初始化失败: {e}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """读取帧"""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        
        ret, frame = self.cap.read()
        if ret and frame is not None:
            return True, frame
        return False, None
    
    def release(self):
        """释放摄像头资源"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_running = False
        logger.info("摄像头资源已释放")


class BaseYOLOSGUI(ABC):
    """YOLOS GUI基类
    
    提供统一的GUI框架，子类需要实现具体的检测逻辑
    """
    
    def __init__(self, title: str = "YOLOS GUI", config_file: str = "gui_config.json"):
        # 配置管理
        self.config_manager = BaseGUIConfig(config_file)
        
        # 主窗口
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(f"{self.config_manager.get('window_width')}x{self.config_manager.get('window_height')}")
        
        # 摄像头管理
        self.camera_manager = CameraManager(self.config_manager.get('camera_index'))
        
        # 状态变量
        self.is_detecting = False
        self.current_frame = None
        self.detection_results = []
        self.camera_thread = None
        self.detection_thread = None
        
        # UI组件
        self.video_label = None
        self.status_var = tk.StringVar(value="就绪")
        self.log_text = None
        
        # 初始化界面
        self.setup_style()
        self.setup_ui()
        self.setup_menu()
        
        # 绑定关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_style(self):
        """设置界面样式"""
        style = ttk.Style()
        theme = self.config_manager.get('theme', 'clam')
        try:
            style.theme_use(theme)
        except tk.TclError:
            style.theme_use('clam')
        
        # 自定义样式
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Subtitle.TLabel', font=('Arial', 10))
        style.configure('Action.TButton', font=('Arial', 12), padding=10)
    
    @abstractmethod
    def setup_ui(self):
        """设置用户界面 - 子类必须实现"""
        pass
    
    def setup_menu(self):
        """设置菜单栏"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="打开图片", command=self.load_image)
        file_menu.add_command(label="打开视频", command=self.load_video)
        file_menu.add_separator()
        file_menu.add_command(label="保存结果", command=self.save_results)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.on_closing)
        
        # 设置菜单
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="设置", menu=settings_menu)
        settings_menu.add_command(label="摄像头设置", command=self.open_camera_settings)
        settings_menu.add_command(label="检测参数", command=self.open_detection_settings)
        settings_menu.add_command(label="保存配置", command=self.save_config)
        
        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="使用说明", command=self.show_help)
        help_menu.add_command(label="关于", command=self.show_about)
    
    def create_control_panel(self, parent) -> ttk.Frame:
        """创建通用控制面板"""
        control_frame = ttk.LabelFrame(parent, text="控制面板", width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)
        
        # 摄像头控制
        camera_frame = ttk.LabelFrame(control_frame, text="摄像头控制")
        camera_frame.pack(fill=tk.X, pady=5)
        
        # 摄像头索引设置
        ttk.Label(camera_frame, text="摄像头索引:").pack(anchor=tk.W)
        self.camera_var = tk.StringVar(value=str(self.config_manager.get('camera_index')))
        camera_entry = ttk.Entry(camera_frame, textvariable=self.camera_var, width=10)
        camera_entry.pack(anchor=tk.W, pady=2)
        
        # 摄像头控制按钮
        button_frame = ttk.Frame(camera_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        self.start_btn = ttk.Button(button_frame, text="启动摄像头", command=self.start_camera)
        self.start_btn.pack(side=tk.LEFT, padx=2)
        
        self.stop_btn = ttk.Button(button_frame, text="停止摄像头", command=self.stop_camera, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2)
        
        # 检测控制
        detection_frame = ttk.LabelFrame(control_frame, text="检测控制")
        detection_frame.pack(fill=tk.X, pady=5)
        
        self.detect_btn = ttk.Button(detection_frame, text="开始检测", 
                                   command=self.toggle_detection, state=tk.DISABLED)
        self.detect_btn.pack(pady=5)
        
        return control_frame
    
    def create_display_area(self, parent) -> ttk.Frame:
        """创建显示区域"""
        display_frame = ttk.LabelFrame(parent, text="视频显示")
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 视频显示标签
        self.video_label = ttk.Label(display_frame, text="请启动摄像头")
        self.video_label.pack(expand=True)
        
        return display_frame
    
    def create_status_bar(self, parent):
        """创建状态栏"""
        status_frame = ttk.Frame(parent)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        ttk.Label(status_frame, text="状态:").pack(side=tk.LEFT)
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=5)
    
    # 摄像头控制方法
    def start_camera(self):
        """启动摄像头"""
        try:
            camera_index = int(self.camera_var.get())
            self.camera_manager.camera_index = camera_index
            
            if self.camera_manager.initialize():
                self.camera_manager.is_running = True
                self.start_btn.config(state=tk.DISABLED)
                self.stop_btn.config(state=tk.NORMAL)
                self.detect_btn.config(state=tk.NORMAL)
                
                # 启动摄像头线程
                self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
                self.camera_thread.start()
                
                self.update_status("摄像头已启动")
                self.log_message(f"摄像头 {camera_index} 启动成功")
            else:
                messagebox.showerror("错误", "摄像头启动失败")
        except ValueError:
            messagebox.showerror("错误", "请输入有效的摄像头索引")
        except Exception as e:
            messagebox.showerror("错误", f"启动摄像头时发生错误: {e}")
    
    def stop_camera(self):
        """停止摄像头"""
        self.camera_manager.is_running = False
        self.is_detecting = False
        
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1.0)
        
        self.camera_manager.release()
        
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.detect_btn.config(state=tk.DISABLED)
        
        if self.video_label:
            self.video_label.config(image='', text="请启动摄像头")
        
        self.update_status("摄像头已停止")
        self.log_message("摄像头已停止")
    
    def toggle_detection(self):
        """切换检测状态"""
        if self.is_detecting:
            self.is_detecting = False
            self.detect_btn.config(text="开始检测")
            self.update_status("检测已停止")
        else:
            self.is_detecting = True
            self.detect_btn.config(text="停止检测")
            self.update_status("检测中...")
    
    def camera_loop(self):
        """摄像头循环"""
        while self.camera_manager.is_running:
            ret, frame = self.camera_manager.read_frame()
            if ret and frame is not None:
                self.current_frame = frame.copy()
                
                # 如果正在检测，执行检测
                if self.is_detecting:
                    self.detection_results = self.perform_detection(frame)
                    frame = self.draw_detections(frame, self.detection_results)
                
                # 更新显示
                self.root.after(0, self.update_video_display, frame)
            
            time.sleep(1/30)  # 30 FPS
    
    @abstractmethod
    def perform_detection(self, frame: np.ndarray) -> List[Dict]:
        """执行检测 - 子类必须实现"""
        pass
    
    @abstractmethod
    def draw_detections(self, frame: np.ndarray, results: List[Dict]) -> np.ndarray:
        """绘制检测结果 - 子类必须实现"""
        pass
    
    def update_video_display(self, frame: np.ndarray):
        """更新视频显示"""
        if self.video_label is None:
            return
        
        try:
            # 转换为RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 调整大小以适应显示区域
            height, width = frame_rgb.shape[:2]
            max_width, max_height = 800, 600
            
            if width > max_width or height > max_height:
                scale = min(max_width/width, max_height/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
            
            # 转换为PhotoImage
            from PIL import Image, ImageTk
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image)
            
            self.video_label.config(image=photo, text="")
            self.video_label.image = photo  # 保持引用
        except Exception as e:
            logger.error(f"更新视频显示失败: {e}")
    
    # 工具方法
    def update_status(self, message: str):
        """更新状态栏"""
        self.status_var.set(message)
        logger.info(message)
    
    def log_message(self, message: str):
        """记录日志消息"""
        if self.log_text:
            self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
            self.log_text.see(tk.END)
        logger.info(message)
    
    # 文件操作
    def load_image(self):
        """加载图片"""
        file_path = filedialog.askopenfilename(
            title="选择图片文件",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            try:
                frame = cv2.imread(file_path)
                if frame is not None:
                    self.current_frame = frame
                    if self.is_detecting:
                        self.detection_results = self.perform_detection(frame)
                        frame = self.draw_detections(frame, self.detection_results)
                    self.update_video_display(frame)
                    self.log_message(f"已加载图片: {Path(file_path).name}")
                else:
                    messagebox.showerror("错误", "无法加载图片文件")
            except Exception as e:
                messagebox.showerror("错误", f"加载图片时发生错误: {e}")
    
    def load_video(self):
        """加载视频"""
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv")]
        )
        if file_path:
            # 这里可以实现视频播放逻辑
            self.log_message(f"已选择视频: {Path(file_path).name}")
    
    def save_results(self):
        """保存检测结果"""
        if not self.detection_results:
            messagebox.showwarning("警告", "没有检测结果可保存")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="保存检测结果",
            defaultextension=".json",
            filetypes=[("JSON文件", "*.json")]
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.detection_results, f, indent=2, ensure_ascii=False)
                self.log_message(f"检测结果已保存: {Path(file_path).name}")
            except Exception as e:
                messagebox.showerror("错误", f"保存结果时发生错误: {e}")
    
    def save_config(self):
        """保存配置"""
        if self.config_manager.save_config():
            messagebox.showinfo("信息", "配置已保存")
        else:
            messagebox.showerror("错误", "保存配置失败")
    
    # 设置对话框
    def open_camera_settings(self):
        """打开摄像头设置"""
        # 这里可以实现摄像头设置对话框
        messagebox.showinfo("信息", "摄像头设置功能待实现")
    
    def open_detection_settings(self):
        """打开检测参数设置"""
        # 这里可以实现检测参数设置对话框
        messagebox.showinfo("信息", "检测参数设置功能待实现")
    
    # 帮助和关于
    def show_help(self):
        """显示帮助信息"""
        help_text = """
YOLOS GUI 使用说明:

1. 摄像头控制:
   - 设置摄像头索引（通常为0）
   - 点击"启动摄像头"开始视频流
   - 点击"停止摄像头"结束视频流

2. 目标检测:
   - 启动摄像头后，点击"开始检测"
   - 检测结果将实时显示在视频上
   - 点击"停止检测"暂停检测

3. 文件操作:
   - 可以加载图片或视频进行检测
   - 检测结果可以保存为JSON格式

4. 设置:
   - 可以调整摄像头和检测参数
   - 配置会自动保存
"""
        messagebox.showinfo("使用说明", help_text)
    
    def show_about(self):
        """显示关于信息"""
        about_text = """
YOLOS - 智能目标检测系统

版本: 2.0.0
作者: YOLOS Team

基于深度学习的实时目标检测系统，
支持多平台部署和扩展。
"""
        messagebox.showinfo("关于", about_text)
    
    def on_closing(self):
        """关闭程序"""
        if self.camera_manager.is_running:
            self.stop_camera()
        
        # 保存配置
        self.config_manager.save_config()
        
        self.root.destroy()
    
    def run(self):
        """运行GUI"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.on_closing()
        except Exception as e:
            logger.error(f"GUI运行时发生错误: {e}")
            self.on_closing()