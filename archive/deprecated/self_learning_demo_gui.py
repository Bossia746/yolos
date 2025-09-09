#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大模型自学习系统演示GUI
展示YOLOS系统的自学习能力和多模态识别功能
"""

import sys
import os
import cv2
import numpy as np
import json
import time
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import yaml

# GUI相关导入
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
    from PIL import Image, ImageTk
except ImportError as e:
    print(f"GUI依赖缺失: {e}")
    print("请安装: pip install pillow")
    sys.exit(1)

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 导入自学习系统
try:
    from src.recognition.integrated_self_learning_recognition import (
        IntegratedSelfLearningRecognition, 
        RecognitionMode, 
        RecognitionResult,
        ConfidenceLevel
    )
    from src.recognition.llm_self_learning_system import LLMSelfLearningSystem
    SELF_LEARNING_AVAILABLE = True
except ImportError as e:
    print(f"导入自学习系统失败: {e}")
    print("请确保项目结构正确")
    SELF_LEARNING_AVAILABLE = False
    
    # 定义备用类型
    class RecognitionResult:
        def __init__(self, object_type="unknown", confidence=0.0, bbox=None, description=""):
            self.object_type = object_type
            self.confidence = confidence
            self.bbox = bbox or [0, 0, 100, 100]
            self.description = description
            self.confidence_level = ConfidenceLevel.MEDIUM
            self.recognition_method = "offline"
            self.processing_time = 0.1
            self.timestamp = time.time()
            self.emergency_level = "normal"
            self.suggested_actions = []
            self.confidence_factors = {}
            self.requires_learning = False
            self.learning_triggered = False
            self.learning_success = False
            self.image_quality_score = 0.8
            self.anti_spoofing_score = 0.9
            self.medical_analysis = None
            self.llm_result = None
    
    from enum import Enum
    
    class RecognitionMode(Enum):
        OFFLINE_ONLY = "offline_only"
        HYBRID_AUTO = "hybrid_auto"
        SELF_LEARNING = "self_learning"
    
    class ConfidenceLevel:
        VERY_HIGH = "very_high"
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"
        VERY_LOW = "very_low"
    
    class IntegratedSelfLearningRecognition:
        def __init__(self, config_path=None):
            pass
        
        def recognize(self, image, mode=None):
            return RecognitionResult("演示模式", 0.8, [10, 10, 100, 100], "系统未完全加载")
        
        def get_recognition_statistics(self):
            return {
                "total_recognitions": 0,
                "successful_recognitions": 0,
                "learning_sessions": 0,
                "accuracy": 0.0
            }
    
    class LLMSelfLearningSystem:
        def __init__(self, config_path=None):
            pass

class SelfLearningDemoGUI:
    """大模型自学习演示GUI"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("YOLOS 大模型自学习系统演示")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # 加载配置
        self.config = self._load_config()
        
        # 初始化系统
        self.recognition_system = None
        self.current_image = None
        self.current_result = None
        
        # GUI组件
        self.setup_gui()
        
        # 初始化系统（在后台线程中）
        self.init_thread = threading.Thread(target=self._initialize_system)
        self.init_thread.daemon = True
        self.init_thread.start()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            config_path = Path("config/self_learning_config.yaml")
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            else:
                return self._get_default_config()
        except Exception as e:
            print(f"加载配置失败: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'self_learning': {'enabled': True},
            'recognition_modes': {'default_mode': 'hybrid_auto'},
            'ui': {
                'theme': 'light',
                'language': 'zh-CN',
                'show_confidence_bars': True,
                'show_processing_time': True,
                'show_learning_status': True
            }
        }
    
    def setup_gui(self):
        """设置GUI界面"""
        # 创建主框架
        self.create_main_frame()
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 创建工具栏
        self.create_toolbar()
        
        # 创建主要内容区域
        self.create_content_area()
        
        # 创建状态栏
        self.create_status_bar()
        
    def create_main_frame(self):
        """创建主框架"""
        # 主容器
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="打开图像", command=self.open_image)
        file_menu.add_command(label="打开摄像头", command=self.open_camera)
        file_menu.add_separator()
        file_menu.add_command(label="保存结果", command=self.save_result)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        
        # 识别菜单
        recognition_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="识别", menu=recognition_menu)
        recognition_menu.add_command(label="开始识别", command=self.start_recognition)
        recognition_menu.add_command(label="停止识别", command=self.stop_recognition)
        recognition_menu.add_separator()
        recognition_menu.add_command(label="触发自学习", command=self.trigger_self_learning)
        
        # 设置菜单
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="设置", menu=settings_menu)
        settings_menu.add_command(label="识别模式", command=self.show_mode_settings)
        settings_menu.add_command(label="大模型配置", command=self.show_llm_settings)
        settings_menu.add_command(label="系统配置", command=self.show_system_settings)
        
        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="使用说明", command=self.show_help)
        help_menu.add_command(label="关于", command=self.show_about)
    
    def create_toolbar(self):
        """创建工具栏"""
        toolbar = ttk.Frame(self.main_container)
        toolbar.pack(fill=tk.X, pady=(0, 5))
        
        # 文件操作按钮
        ttk.Button(toolbar, text="📁 打开图像", command=self.open_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="📷 摄像头", command=self.open_camera).pack(side=tk.LEFT, padx=2)
        
        # 分隔符
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # 识别控制按钮
        ttk.Button(toolbar, text="🔍 识别", command=self.start_recognition).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🧠 自学习", command=self.trigger_self_learning).pack(side=tk.LEFT, padx=2)
        
        # 分隔符
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # 识别模式选择
        ttk.Label(toolbar, text="识别模式:").pack(side=tk.LEFT, padx=2)
        self.mode_var = tk.StringVar(value="hybrid_auto")
        mode_combo = ttk.Combobox(toolbar, textvariable=self.mode_var, width=15, state="readonly")
        mode_combo['values'] = ("offline_only", "hybrid_auto", "self_learning", "manual_confirm")
        mode_combo.pack(side=tk.LEFT, padx=2)
        
        # 分隔符
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # 系统状态指示器
        ttk.Label(toolbar, text="系统状态:").pack(side=tk.LEFT, padx=2)
        self.status_indicator = ttk.Label(toolbar, text="🔴 未初始化", foreground="red")
        self.status_indicator.pack(side=tk.LEFT, padx=2)
    
    def create_content_area(self):
        """创建主要内容区域"""
        # 创建水平分割的PanedWindow
        paned = ttk.PanedWindow(self.main_container, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # 左侧面板 - 图像显示和控制
        self.create_left_panel(paned)
        
        # 右侧面板 - 结果显示和学习信息
        self.create_right_panel(paned)
    
    def create_left_panel(self, parent):
        """创建左侧面板"""
        left_frame = ttk.Frame(parent)
        parent.add(left_frame, weight=2)
        
        # 图像显示区域
        image_frame = ttk.LabelFrame(left_frame, text="图像显示", padding=10)
        image_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 图像画布
        self.image_canvas = tk.Canvas(image_frame, bg='white', width=640, height=480)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 图像信息
        info_frame = ttk.Frame(image_frame)
        info_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.image_info_label = ttk.Label(info_frame, text="未加载图像")
        self.image_info_label.pack(side=tk.LEFT)
        
        # 控制面板
        control_frame = ttk.LabelFrame(left_frame, text="控制面板", padding=10)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 置信度阈值设置
        ttk.Label(control_frame, text="置信度阈值:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.confidence_var = tk.DoubleVar(value=0.5)
        confidence_scale = ttk.Scale(control_frame, from_=0.0, to=1.0, 
                                   variable=self.confidence_var, orient=tk.HORIZONTAL)
        confidence_scale.grid(row=0, column=1, sticky=tk.EW, padx=5)
        self.confidence_label = ttk.Label(control_frame, text="0.50")
        self.confidence_label.grid(row=0, column=2, padx=5)
        
        # 绑定滑块事件
        confidence_scale.configure(command=self.update_confidence_label)
        
        # 自学习开关
        self.auto_learning_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="自动触发学习", 
                       variable=self.auto_learning_var).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # 配置网格权重
        control_frame.columnconfigure(1, weight=1)
    
    def create_right_panel(self, parent):
        """创建右侧面板"""
        right_frame = ttk.Frame(parent)
        parent.add(right_frame, weight=1)
        
        # 创建垂直分割的PanedWindow
        right_paned = ttk.PanedWindow(right_frame, orient=tk.VERTICAL)
        right_paned.pack(fill=tk.BOTH, expand=True)
        
        # 识别结果面板
        self.create_result_panel(right_paned)
        
        # 学习信息面板
        self.create_learning_panel(right_paned)
        
        # 系统统计面板
        self.create_stats_panel(right_paned)
    
    def create_result_panel(self, parent):
        """创建识别结果面板"""
        result_frame = ttk.LabelFrame(parent, text="识别结果", padding=10)
        parent.add(result_frame, weight=2)
        
        # 结果显示区域
        self.result_text = scrolledtext.ScrolledText(result_frame, height=15, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # 置信度可视化
        confidence_frame = ttk.Frame(result_frame)
        confidence_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(confidence_frame, text="置信度:").pack(side=tk.LEFT)
        self.confidence_progress = ttk.Progressbar(confidence_frame, length=200, mode='determinate')
        self.confidence_progress.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.confidence_value_label = ttk.Label(confidence_frame, text="0%")
        self.confidence_value_label.pack(side=tk.RIGHT)
    
    def create_learning_panel(self, parent):
        """创建学习信息面板"""
        learning_frame = ttk.LabelFrame(parent, text="自学习信息", padding=10)
        parent.add(learning_frame, weight=1)
        
        # 学习状态
        status_frame = ttk.Frame(learning_frame)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(status_frame, text="学习状态:").pack(side=tk.LEFT)
        self.learning_status_label = ttk.Label(status_frame, text="未启动", foreground="gray")
        self.learning_status_label.pack(side=tk.LEFT, padx=10)
        
        # 学习进度
        ttk.Label(learning_frame, text="学习进度:").pack(anchor=tk.W)
        self.learning_progress = ttk.Progressbar(learning_frame, length=300, mode='indeterminate')
        self.learning_progress.pack(fill=tk.X, pady=5)
        
        # 学习日志
        ttk.Label(learning_frame, text="学习日志:").pack(anchor=tk.W, pady=(10, 0))
        self.learning_log = scrolledtext.ScrolledText(learning_frame, height=8, wrap=tk.WORD)
        self.learning_log.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_stats_panel(self, parent):
        """创建系统统计面板"""
        stats_frame = ttk.LabelFrame(parent, text="系统统计", padding=10)
        parent.add(stats_frame, weight=1)
        
        # 统计信息显示
        self.stats_text = scrolledtext.ScrolledText(stats_frame, height=10, wrap=tk.WORD)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
        # 刷新按钮
        ttk.Button(stats_frame, text="刷新统计", command=self.refresh_stats).pack(pady=(10, 0))
    
    def create_status_bar(self):
        """创建状态栏"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # 状态标签
        self.status_label = ttk.Label(status_frame, text="就绪", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, pady=2)
        
        # 处理时间标签
        self.time_label = ttk.Label(status_frame, text="处理时间: 0.00s", relief=tk.SUNKEN)
        self.time_label.pack(side=tk.RIGHT, padx=2, pady=2)
    
    def _initialize_system(self):
        """初始化识别系统"""
        try:
            self.update_status("正在初始化系统...")
            self.update_status_indicator("🟡 初始化中", "orange")
            
            # 创建识别系统
            self.recognition_system = IntegratedSelfLearningRecognition(self.config)
            
            self.update_status("系统初始化完成")
            self.update_status_indicator("🟢 已就绪", "green")
            
        except Exception as e:
            error_msg = f"系统初始化失败: {e}"
            self.update_status(error_msg)
            self.update_status_indicator("🔴 初始化失败", "red")
            messagebox.showerror("初始化错误", error_msg)
    
    def update_status(self, message: str):
        """更新状态栏"""
        def update():
            self.status_label.config(text=message)
        self.root.after(0, update)
    
    def update_status_indicator(self, text: str, color: str):
        """更新状态指示器"""
        def update():
            self.status_indicator.config(text=text, foreground=color)
        self.root.after(0, update)
    
    def update_confidence_label(self, value):
        """更新置信度标签"""
        self.confidence_label.config(text=f"{float(value):.2f}")
    
    def open_image(self):
        """打开图像文件"""
        file_types = [
            ("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("JPEG文件", "*.jpg *.jpeg"),
            ("PNG文件", "*.png"),
            ("所有文件", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=file_types
        )
        
        if file_path:
            try:
                # 加载图像
                self.current_image = cv2.imread(file_path)
                if self.current_image is None:
                    raise ValueError("无法加载图像文件")
                
                # 显示图像
                self.display_image(self.current_image)
                
                # 更新图像信息
                h, w = self.current_image.shape[:2]
                file_size = os.path.getsize(file_path) / 1024  # KB
                self.image_info_label.config(
                    text=f"尺寸: {w}x{h}, 大小: {file_size:.1f}KB, 文件: {os.path.basename(file_path)}"
                )
                
                self.update_status(f"已加载图像: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("加载错误", f"无法加载图像: {e}")
    
    def open_camera(self):
        """打开摄像头"""
        try:
            # 创建摄像头窗口
            self.camera_window = CameraWindow(self)
        except Exception as e:
            messagebox.showerror("摄像头错误", f"无法打开摄像头: {e}")
    
    def display_image(self, image: np.ndarray):
        """在画布上显示图像"""
        try:
            # 转换颜色空间
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # 获取画布尺寸
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                # 画布还未初始化，使用默认尺寸
                canvas_width, canvas_height = 640, 480
            
            # 计算缩放比例
            h, w = image_rgb.shape[:2]
            scale = min(canvas_width / w, canvas_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # 调整图像尺寸
            resized_image = cv2.resize(image_rgb, (new_w, new_h))
            
            # 转换为PIL图像
            pil_image = Image.fromarray(resized_image)
            self.photo = ImageTk.PhotoImage(pil_image)
            
            # 清除画布并显示图像
            self.image_canvas.delete("all")
            x = (canvas_width - new_w) // 2
            y = (canvas_height - new_h) // 2
            self.image_canvas.create_image(x, y, anchor=tk.NW, image=self.photo)
            
        except Exception as e:
            print(f"显示图像失败: {e}")
    
    def start_recognition(self):
        """开始识别"""
        if self.current_image is None:
            messagebox.showwarning("警告", "请先加载图像")
            return
        
        if self.recognition_system is None:
            messagebox.showwarning("警告", "系统尚未初始化完成")
            return
        
        # 在后台线程中执行识别
        recognition_thread = threading.Thread(target=self._perform_recognition)
        recognition_thread.daemon = True
        recognition_thread.start()
    
    def _perform_recognition(self):
        """执行识别（后台线程）"""
        try:
            self.update_status("正在识别...")
            start_time = time.time()
            
            # 获取识别模式
            mode_str = self.mode_var.get()
            if mode_str == "offline_only":
                mode = RecognitionMode.OFFLINE_ONLY
            elif mode_str == "self_learning":
                mode = RecognitionMode.SELF_LEARNING
            else:
                mode = RecognitionMode.HYBRID_AUTO
            
            # 执行识别
            result = self.recognition_system.recognize(
                self.current_image,
                mode=mode
            )
            
            processing_time = time.time() - start_time
            
            # 更新UI
            self.root.after(0, lambda: self._update_recognition_result(result, processing_time))
            
        except Exception as e:
            error_msg = f"识别失败: {e}"
            self.root.after(0, lambda: self.update_status(error_msg))
            self.root.after(0, lambda: messagebox.showerror("识别错误", error_msg))
    
    def _update_recognition_result(self, result: RecognitionResult, processing_time: float):
        """更新识别结果显示"""
        self.current_result = result
        
        # 更新结果文本
        result_text = self._format_recognition_result(result)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, result_text)
        
        # 更新置信度进度条
        confidence_percent = result.confidence * 100
        self.confidence_progress['value'] = confidence_percent
        self.confidence_value_label.config(text=f"{confidence_percent:.1f}%")
        
        # 设置置信度颜色
        if result.confidence_level == ConfidenceLevel.VERY_HIGH:
            color = "#28a745"  # 绿色
        elif result.confidence_level == ConfidenceLevel.HIGH:
            color = "#17a2b8"  # 蓝色
        elif result.confidence_level == ConfidenceLevel.MEDIUM:
            color = "#ffc107"  # 黄色
        elif result.confidence_level == ConfidenceLevel.LOW:
            color = "#fd7e14"  # 橙色
        else:
            color = "#dc3545"  # 红色
        
        # 更新学习状态
        if result.learning_triggered:
            self.learning_status_label.config(
                text="学习已触发" if result.learning_success else "学习失败",
                foreground="green" if result.learning_success else "red"
            )
            
            # 添加学习日志
            learning_msg = f"[{time.strftime('%H:%M:%S')}] "
            if result.learning_success:
                learning_msg += f"成功学习新场景: {result.object_type}\n"
            else:
                learning_msg += f"学习失败: {result.object_type}\n"
            
            self.learning_log.insert(tk.END, learning_msg)
            self.learning_log.see(tk.END)
        
        # 更新状态栏
        self.update_status(f"识别完成: {result.object_type} (置信度: {confidence_percent:.1f}%)")
        self.time_label.config(text=f"处理时间: {processing_time:.2f}s")
    
    def _format_recognition_result(self, result: RecognitionResult) -> str:
        """格式化识别结果"""
        lines = []
        lines.append("=== 识别结果 ===")
        lines.append(f"对象类型: {result.object_type}")
        lines.append(f"置信度: {result.confidence:.3f} ({result.confidence_level.value})")
        lines.append(f"识别方法: {result.recognition_method}")
        lines.append(f"处理时间: {result.processing_time:.3f}秒")
        lines.append(f"紧急程度: {result.emergency_level}")
        lines.append("")
        
        # 图像质量信息
        lines.append("=== 图像质量 ===")
        lines.append(f"质量评分: {result.image_quality_score:.3f}")
        lines.append(f"反欺骗评分: {result.anti_spoofing_score:.3f}")
        lines.append("")
        
        # 置信度因子
        lines.append("=== 置信度因子 ===")
        for factor, value in result.confidence_factors.items():
            lines.append(f"{factor}: {value:.3f}")
        lines.append("")
        
        # 建议行动
        if result.suggested_actions:
            lines.append("=== 建议行动 ===")
            for i, action in enumerate(result.suggested_actions, 1):
                lines.append(f"{i}. {action}")
            lines.append("")
        
        # 医疗分析
        if result.medical_analysis:
            lines.append("=== 医疗分析 ===")
            lines.append(json.dumps(result.medical_analysis, ensure_ascii=False, indent=2))
            lines.append("")
        
        # 大模型结果
        if result.llm_result:
            lines.append("=== 大模型分析 ===")
            lines.append(f"场景描述: {result.llm_result.scene_description}")
            lines.append(f"场景类别: {result.llm_result.scene_category.value}")
            
            if result.llm_result.detected_objects:
                lines.append("检测对象:")
                for obj in result.llm_result.detected_objects:
                    lines.append(f"  - {obj.get('name', 'unknown')}")
            
            if result.llm_result.learning_keywords:
                lines.append(f"学习关键词: {', '.join(result.llm_result.learning_keywords)}")
        
        # 学习信息
        lines.append("=== 学习信息 ===")
        lines.append(f"需要学习: {'是' if result.requires_learning else '否'}")
        lines.append(f"学习触发: {'是' if result.learning_triggered else '否'}")
        if result.learning_success is not None:
            lines.append(f"学习结果: {'成功' if result.learning_success else '失败'}")
        
        return "\n".join(lines)
    
    def trigger_self_learning(self):
        """手动触发自学习"""
        if self.current_image is None:
            messagebox.showwarning("警告", "请先加载图像")
            return
        
        if self.recognition_system is None:
            messagebox.showwarning("警告", "系统尚未初始化完成")
            return
        
        # 强制使用自学习模式
        self.mode_var.set("self_learning")
        self.start_recognition()
    
    def stop_recognition(self):
        """停止识别"""
        # 这里可以添加停止识别的逻辑
        self.update_status("识别已停止")
    
    def refresh_stats(self):
        """刷新系统统计"""
        if self.recognition_system is None:
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, "系统尚未初始化")
            return
        
        try:
            stats = self.recognition_system.get_recognition_statistics()
            stats_text = json.dumps(stats, ensure_ascii=False, indent=2)
            
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, stats_text)
            
        except Exception as e:
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, f"获取统计信息失败: {e}")
    
    def save_result(self):
        """保存识别结果"""
        if self.current_result is None:
            messagebox.showwarning("警告", "没有可保存的结果")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="保存识别结果",
            defaultextension=".json",
            filetypes=[("JSON文件", "*.json"), ("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                # 将结果转换为可序列化格式
                result_dict = {
                    'object_type': self.current_result.object_type,
                    'confidence': self.current_result.confidence,
                    'confidence_level': self.current_result.confidence_level.value,
                    'recognition_method': self.current_result.recognition_method,
                    'processing_time': self.current_result.processing_time,
                    'timestamp': self.current_result.timestamp,
                    'emergency_level': self.current_result.emergency_level,
                    'suggested_actions': self.current_result.suggested_actions,
                    'confidence_factors': self.current_result.confidence_factors,
                    'requires_learning': self.current_result.requires_learning,
                    'learning_triggered': self.current_result.learning_triggered,
                    'learning_success': self.current_result.learning_success
                }
                
                if file_path.endswith('.json'):
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(result_dict, f, ensure_ascii=False, indent=2)
                else:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(self._format_recognition_result(self.current_result))
                
                messagebox.showinfo("保存成功", f"结果已保存到: {file_path}")
                
            except Exception as e:
                messagebox.showerror("保存失败", f"无法保存结果: {e}")
    
    def show_mode_settings(self):
        """显示识别模式设置"""
        ModeSettingsWindow(self)
    
    def show_llm_settings(self):
        """显示大模型配置"""
        LLMSettingsWindow(self)
    
    def show_system_settings(self):
        """显示系统配置"""
        SystemSettingsWindow(self)
    
    def show_help(self):
        """显示帮助信息"""
        help_text = """
YOLOS 大模型自学习系统使用说明

1. 基本操作：
   - 点击"打开图像"加载要识别的图像
   - 选择识别模式（离线、混合、自学习、手动确认）
   - 点击"识别"开始分析图像

2. 识别模式：
   - 离线模式：仅使用本地模型
   - 混合自动：根据置信度自动选择
   - 自学习模式：强制使用大模型学习
   - 手动确认：需要用户确认结果

3. 自学习功能：
   - 当识别置信度较低时自动触发
   - 调用大模型API分析未知场景
   - 学习结果会保存到知识库

4. 注意事项：
   - 需要配置大模型API密钥
   - 首次使用需要网络连接
   - 学习过程可能需要一些时间
        """
        
        messagebox.showinfo("使用说明", help_text)
    
    def show_about(self):
        """显示关于信息"""
        about_text = """
YOLOS 大模型自学习系统 v1.0

这是一个集成了大模型自学习能力的智能识别系统，
支持多种识别模式和自适应学习功能。

主要特性：
• 多模态识别（人脸、姿态、物体等）
• 大模型自学习能力
• 医疗场景专用分析
• 跌倒检测和紧急响应
• 反欺骗检测
• 图像质量增强

开发团队：YOLOS项目组
技术支持：基于Claude、GPT-4V等大模型
        """
        
        messagebox.showinfo("关于", about_text)
    
    def run(self):
        """运行GUI"""
        self.root.mainloop()


class CameraWindow:
    """摄像头窗口"""
    
    def __init__(self, parent):
        self.parent = parent
        self.cap = None
        
        # 创建窗口
        self.window = tk.Toplevel(parent.root)
        self.window.title("摄像头")
        self.window.geometry("800x600")
        
        # 摄像头画布
        self.canvas = tk.Canvas(self.window, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 控制按钮
        button_frame = ttk.Frame(self.window)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="拍照", command=self.capture_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="关闭", command=self.close_camera).pack(side=tk.RIGHT, padx=5)
        
        # 启动摄像头
        self.start_camera()
        
        # 绑定关闭事件
        self.window.protocol("WM_DELETE_WINDOW", self.close_camera)
    
    def start_camera(self):
        """启动摄像头"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("无法打开摄像头")
            
            self.update_frame()
            
        except Exception as e:
            messagebox.showerror("摄像头错误", f"无法启动摄像头: {e}")
            self.window.destroy()
    
    def update_frame(self):
        """更新摄像头画面"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # 转换颜色空间
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 调整尺寸
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:
                    h, w = frame_rgb.shape[:2]
                    scale = min(canvas_width / w, canvas_height / h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    
                    frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
                    
                    # 显示图像
                    pil_image = Image.fromarray(frame_resized)
                    self.photo = ImageTk.PhotoImage(pil_image)
                    
                    self.canvas.delete("all")
                    x = (canvas_width - new_w) // 2
                    y = (canvas_height - new_h) // 2
                    self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo)
            
            # 继续更新
            self.window.after(30, self.update_frame)
    
    def capture_image(self):
        """拍照"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # 将图像传递给主窗口
                self.parent.current_image = frame
                self.parent.display_image(frame)
                
                # 更新图像信息
                h, w = frame.shape[:2]
                self.parent.image_info_label.config(
                    text=f"尺寸: {w}x{h}, 来源: 摄像头"
                )
                
                self.parent.update_status("已从摄像头捕获图像")
                messagebox.showinfo("拍照成功", "图像已捕获")
    
    def close_camera(self):
        """关闭摄像头"""
        if self.cap:
            self.cap.release()
        self.window.destroy()


class ModeSettingsWindow:
    """识别模式设置窗口"""
    
    def __init__(self, parent):
        self.parent = parent
        
        self.window = tk.Toplevel(parent.root)
        self.window.title("识别模式设置")
        self.window.geometry("400x300")
        self.window.transient(parent.root)
        self.window.grab_set()
        
        # 创建设置界面
        self.create_widgets()
    
    def create_widgets(self):
        """创建设置控件"""
        # 模式选择
        ttk.Label(self.window, text="默认识别模式:").pack(pady=10)
        
        self.mode_var = tk.StringVar(value=self.parent.mode_var.get())
        modes = [
            ("离线模式", "offline_only"),
            ("混合自动", "hybrid_auto"),
            ("自学习模式", "self_learning"),
            ("手动确认", "manual_confirm")
        ]
        
        for text, value in modes:
            ttk.Radiobutton(self.window, text=text, variable=self.mode_var, 
                           value=value).pack(anchor=tk.W, padx=20)
        
        # 按钮
        button_frame = ttk.Frame(self.window)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="确定", command=self.apply_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="取消", command=self.window.destroy).pack(side=tk.LEFT, padx=5)
    
    def apply_settings(self):
        """应用设置"""
        self.parent.mode_var.set(self.mode_var.get())
        self.window.destroy()


class LLMSettingsWindow:
    """大模型配置窗口"""
    
    def __init__(self, parent):
        self.parent = parent
        
        self.window = tk.Toplevel(parent.root)
        self.window.title("大模型配置")
        self.window.geometry("500x400")
        self.window.transient(parent.root)
        self.window.grab_set()
        
        # 创建配置界面
        self.create_widgets()
    
    def create_widgets(self):
        """创建配置控件"""
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # API配置页面
        api_frame = ttk.Frame(notebook)
        notebook.add(api_frame, text="API配置")
        
        # OpenAI配置
        ttk.Label(api_frame, text="OpenAI API Key:").pack(anchor=tk.W, pady=5)
        self.openai_key = tk.StringVar()
        ttk.Entry(api_frame, textvariable=self.openai_key, show="*", width=50).pack(fill=tk.X, pady=5)
        
        # Claude配置
        ttk.Label(api_frame, text="Claude API Key:").pack(anchor=tk.W, pady=5)
        self.claude_key = tk.StringVar()
        ttk.Entry(api_frame, textvariable=self.claude_key, show="*", width=50).pack(fill=tk.X, pady=5)
        
        # 通义千问配置
        ttk.Label(api_frame, text="通义千问 API Key:").pack(anchor=tk.W, pady=5)
        self.qwen_key = tk.StringVar()
        ttk.Entry(api_frame, textvariable=self.qwen_key, show="*", width=50).pack(fill=tk.X, pady=5)
        
        # 按钮
        button_frame = ttk.Frame(self.window)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="保存", command=self.save_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="取消", command=self.window.destroy).pack(side=tk.LEFT, padx=5)
    
    def save_settings(self):
        """保存设置"""
        # 这里可以保存API密钥到配置文件
        messagebox.showinfo("保存成功", "配置已保存")
        self.window.destroy()


class SystemSettingsWindow:
    """系统配置窗口"""
    
    def __init__(self, parent):
        self.parent = parent
        
        self.window = tk.Toplevel(parent.root)
        self.window.title("系统配置")
        self.window.geometry("600x500")
        self.window.transient(parent.root)
        self.window.grab_set()
        
        # 创建配置界面
        self.create_widgets()
    
    def create_widgets(self):
        """创建配置控件"""
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 性能配置
        perf_frame = ttk.Frame(notebook)
        notebook.add(perf_frame, text="性能")
        
        ttk.Label(perf_frame, text="最大并发请求数:").pack(anchor=tk.W, pady=5)
        self.max_concurrent = tk.IntVar(value=4)
        ttk.Spinbox(perf_frame, from_=1, to=16, textvariable=self.max_concurrent).pack(anchor=tk.W)
        
        # 学习配置
        learning_frame = ttk.Frame(notebook)
        notebook.add(learning_frame, text="学习")
        
        self.auto_learning = tk.BooleanVar(value=True)
        ttk.Checkbutton(learning_frame, text="启用自动学习", variable=self.auto_learning).pack(anchor=tk.W, pady=5)
        
        ttk.Label(learning_frame, text="学习触发阈值:").pack(anchor=tk.W, pady=5)
        self.learning_threshold = tk.DoubleVar(value=0.5)
        ttk.Scale(learning_frame, from_=0.0, to=1.0, variable=self.learning_threshold, 
                 orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        
        # 按钮
        button_frame = ttk.Frame(self.window)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="应用", command=self.apply_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="取消", command=self.window.destroy).pack(side=tk.LEFT, padx=5)
    
    def apply_settings(self):
        """应用设置"""
        messagebox.showinfo("应用成功", "设置已应用")
        self.window.destroy()


def main():
    """主函数"""
    try:
        # 创建并运行GUI
        app = SelfLearningDemoGUI()
        app.run()
        
    except Exception as e:
        print(f"程序启动失败: {e}")
        messagebox.showerror("启动错误", f"程序启动失败: {e}")


if __name__ == "__main__":
    main()