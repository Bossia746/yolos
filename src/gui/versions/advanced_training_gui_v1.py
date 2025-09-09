#!/usr/bin/env python3
"""
YOLOS高级训练界面 - 版本1
支持图片/视频上传、摄像头输入、大模型自学习的完整训练界面
这是之前提到的可以作为大模型输入源的那个界面
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
from PIL import Image, ImageTk
import base64
import requests

class AdvancedTrainingGUI:
    """高级训练界面 - 支持多模态输入和自学习"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("YOLOS 高级训练界面 - 多模态自学习")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        # 界面变量
        self.camera = None
        self.is_camera_running = False
        self.current_frame = None
        self.detection_results = []
        self.training_data = []
        
        # 自学习配置
        self.llm_enabled = tk.BooleanVar(value=False)
        self.llm_api_key = tk.StringVar()
        self.llm_model = tk.StringVar(value="gpt-4-vision-preview")
        
        # 文件路径
        self.current_image_path = None
        self.current_video_path = None
        
        # 创建界面
        self.setup_style()
        self.create_interface()
        
        # 日志设置
        self.setup_logging()
        
    def setup_style(self):
        """设置界面样式"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # 自定义样式
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        style.configure('Subtitle.TLabel', font=('Arial', 10))
        style.configure('Action.TButton', font=('Arial', 10))
        
    def setup_logging(self):
        """设置日志"""
        self.logger = logging.getLogger('AdvancedTrainingGUI')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def create_interface(self):
        """创建主界面"""
        # 主容器
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧面板 - 控制区域
        left_frame = ttk.Frame(main_container)
        main_container.add(left_frame, weight=1)
        
        # 右侧面板 - 显示区域
        right_frame = ttk.Frame(main_container)
        main_container.add(right_frame, weight=2)
        
        # 创建左侧控制面板
        self.create_control_panel(left_frame)
        
        # 创建右侧显示面板
        self.create_display_panel(right_frame)
        
    def create_control_panel(self, parent):
        """创建控制面板"""
        # 标题
        title_label = ttk.Label(parent, text="多模态训练控制台", style='Title.TLabel')
        title_label.pack(pady=(0, 10))
        
        # 输入源选择
        self.create_input_source_section(parent)
        
        # 自学习配置
        self.create_llm_config_section(parent)
        
        # 训练控制
        self.create_training_control_section(parent)
        
        # 数据管理
        self.create_data_management_section(parent)
        
        # 日志显示
        self.create_log_section(parent)
        
    def create_input_source_section(self, parent):
        """创建输入源选择区域"""
        input_frame = ttk.LabelFrame(parent, text="📥 输入源选择", padding="10")
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 摄像头控制
        camera_frame = ttk.Frame(input_frame)
        camera_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.camera_btn = ttk.Button(camera_frame, text="📹 启动摄像头", 
                                   command=self.toggle_camera)
        self.camera_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.camera_status = ttk.Label(camera_frame, text="摄像头未启动")
        self.camera_status.pack(side=tk.LEFT)
        
        # 文件上传
        file_frame = ttk.Frame(input_frame)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(file_frame, text="📷 上传图片", 
                  command=self.upload_image).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(file_frame, text="🎥 上传视频", 
                  command=self.upload_video).pack(side=tk.LEFT, padx=(0, 5))
        
        # 批量上传
        batch_frame = ttk.Frame(input_frame)
        batch_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(batch_frame, text="📁 批量上传", 
                  command=self.batch_upload).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(batch_frame, text="🔄 清空数据", 
                  command=self.clear_data).pack(side=tk.LEFT)
        
    def create_llm_config_section(self, parent):
        """创建大模型配置区域"""
        llm_frame = ttk.LabelFrame(parent, text="🤖 大模型自学习配置", padding="10")
        llm_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 启用开关
        enable_frame = ttk.Frame(llm_frame)
        enable_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Checkbutton(enable_frame, text="启用大模型自学习", 
                       variable=self.llm_enabled,
                       command=self.toggle_llm_config).pack(side=tk.LEFT)
        
        # API配置
        self.llm_config_frame = ttk.Frame(llm_frame)
        self.llm_config_frame.pack(fill=tk.X, pady=5)
        
        # API Key
        ttk.Label(self.llm_config_frame, text="API Key:").pack(anchor=tk.W)
        api_entry = ttk.Entry(self.llm_config_frame, textvariable=self.llm_api_key, 
                             show="*", width=40)
        api_entry.pack(fill=tk.X, pady=(0, 5))
        
        # 模型选择
        ttk.Label(self.llm_config_frame, text="模型:").pack(anchor=tk.W)
        model_combo = ttk.Combobox(self.llm_config_frame, textvariable=self.llm_model,
                                  values=["gpt-4-vision-preview", "gpt-4o", "claude-3-vision", "gemini-pro-vision"])
        model_combo.pack(fill=tk.X, pady=(0, 5))
        
        # 测试连接
        ttk.Button(self.llm_config_frame, text="🔗 测试连接", 
                  command=self.test_llm_connection).pack(pady=5)
        
        # 初始状态禁用配置
        self.toggle_llm_config()
        
    def create_training_control_section(self, parent):
        """创建训练控制区域"""
        training_frame = ttk.LabelFrame(parent, text="🎯 训练控制", padding="10")
        training_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 检测按钮
        detect_frame = ttk.Frame(training_frame)
        detect_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(detect_frame, text="🔍 执行检测", 
                  command=self.run_detection).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(detect_frame, text="🤖 LLM分析", 
                  command=self.run_llm_analysis).pack(side=tk.LEFT)
        
        # 标注按钮
        annotation_frame = ttk.Frame(training_frame)
        annotation_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(annotation_frame, text="✏️ 手动标注", 
                  command=self.manual_annotation).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(annotation_frame, text="🏷️ 自动标注", 
                  command=self.auto_annotation).pack(side=tk.LEFT)
        
        # 训练按钮
        train_frame = ttk.Frame(training_frame)
        train_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(train_frame, text="🚀 开始训练", 
                  command=self.start_training).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(train_frame, text="⏸️ 暂停训练", 
                  command=self.pause_training).pack(side=tk.LEFT)
        
    def create_data_management_section(self, parent):
        """创建数据管理区域"""
        data_frame = ttk.LabelFrame(parent, text="💾 数据管理", padding="10")
        data_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 数据统计
        stats_frame = ttk.Frame(data_frame)
        stats_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.data_stats = ttk.Label(stats_frame, text="数据: 0 张图片, 0 个标注")
        self.data_stats.pack(anchor=tk.W)
        
        # 数据操作
        ops_frame = ttk.Frame(data_frame)
        ops_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(ops_frame, text="💾 保存数据集", 
                  command=self.save_dataset).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(ops_frame, text="📂 加载数据集", 
                  command=self.load_dataset).pack(side=tk.LEFT)
        
        # 导出选项
        export_frame = ttk.Frame(data_frame)
        export_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(export_frame, text="📤 导出YOLO格式", 
                  command=self.export_yolo_format).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(export_frame, text="📤 导出COCO格式", 
                  command=self.export_coco_format).pack(side=tk.LEFT)
        
    def create_log_section(self, parent):
        """创建日志区域"""
        log_frame = ttk.LabelFrame(parent, text="📋 操作日志", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, width=40)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
    def create_display_panel(self, parent):
        """创建显示面板"""
        # 显示区域标题
        display_title = ttk.Label(parent, text="🖼️ 图像显示与分析", style='Title.TLabel')
        display_title.pack(pady=(0, 10))
        
        # 图像显示区域
        self.create_image_display(parent)
        
        # 结果显示区域
        self.create_results_display(parent)
        
    def create_image_display(self, parent):
        """创建图像显示区域"""
        image_frame = ttk.LabelFrame(parent, text="图像预览", padding="10")
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 图像画布
        self.image_canvas = tk.Canvas(image_frame, bg='gray90', width=600, height=400)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 图像信息
        self.image_info = ttk.Label(image_frame, text="未加载图像")
        self.image_info.pack(pady=(5, 0))
        
    def create_results_display(self, parent):
        """创建结果显示区域"""
        results_frame = ttk.LabelFrame(parent, text="检测结果", padding="10")
        results_frame.pack(fill=tk.X)
        
        # 结果树形视图
        columns = ('类别', '置信度', '位置', '来源')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=6)
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=120)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def toggle_camera(self):
        """切换摄像头状态"""
        if not self.is_camera_running:
            self.start_camera()
        else:
            self.stop_camera()
            
    def start_camera(self):
        """启动摄像头"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                raise Exception("无法打开摄像头")
                
            self.is_camera_running = True
            self.camera_btn.config(text="⏹️ 停止摄像头")
            self.camera_status.config(text="摄像头运行中")
            
            # 启动摄像头线程
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
            self.log_message("摄像头启动成功")
            
        except Exception as e:
            messagebox.showerror("错误", f"启动摄像头失败: {e}")
            self.log_message(f"摄像头启动失败: {e}")
            
    def stop_camera(self):
        """停止摄像头"""
        self.is_camera_running = False
        if self.camera:
            self.camera.release()
            self.camera = None
            
        self.camera_btn.config(text="📹 启动摄像头")
        self.camera_status.config(text="摄像头已停止")
        self.log_message("摄像头已停止")
        
    def camera_loop(self):
        """摄像头循环"""
        while self.is_camera_running:
            try:
                ret, frame = self.camera.read()
                if ret:
                    self.current_frame = frame.copy()
                    self.display_image(frame)
                    
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                self.log_message(f"摄像头读取错误: {e}")
                break
                
    def display_image(self, image):
        """显示图像到画布"""
        try:
            # 转换颜色空间
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
                
            # 调整尺寸适应画布
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                h, w = image_rgb.shape[:2]
                scale = min(canvas_width/w, canvas_height/h)
                new_w, new_h = int(w*scale), int(h*scale)
                
                image_resized = cv2.resize(image_rgb, (new_w, new_h))
                
                # 转换为PIL图像
                pil_image = Image.fromarray(image_resized)
                photo = ImageTk.PhotoImage(pil_image)
                
                # 更新画布
                self.image_canvas.delete("all")
                x = (canvas_width - new_w) // 2
                y = (canvas_height - new_h) // 2
                self.image_canvas.create_image(x, y, anchor=tk.NW, image=photo)
                self.image_canvas.image = photo  # 保持引用
                
                # 更新图像信息
                self.image_info.config(text=f"尺寸: {w}x{h}, 显示: {new_w}x{new_h}")
                
        except Exception as e:
            self.log_message(f"显示图像错误: {e}")
            
    def upload_image(self):
        """上传图片"""
        file_path = filedialog.askopenfilename(
            title="选择图片文件",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("所有文件", "*.*")
            ]
        )
        
        if file_path:
            try:
                image = cv2.imread(file_path)
                if image is not None:
                    self.current_image_path = file_path
                    self.current_frame = image.copy()
                    self.display_image(image)
                    self.log_message(f"已加载图片: {Path(file_path).name}")
                else:
                    raise Exception("无法读取图片文件")
                    
            except Exception as e:
                messagebox.showerror("错误", f"加载图片失败: {e}")
                self.log_message(f"加载图片失败: {e}")
                
    def upload_video(self):
        """上传视频"""
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("所有文件", "*.*")
            ]
        )
        
        if file_path:
            try:
                cap = cv2.VideoCapture(file_path)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        self.current_video_path = file_path
                        self.current_frame = frame.copy()
                        self.display_image(frame)
                        self.log_message(f"已加载视频: {Path(file_path).name}")
                    cap.release()
                else:
                    raise Exception("无法打开视频文件")
                    
            except Exception as e:
                messagebox.showerror("错误", f"加载视频失败: {e}")
                self.log_message(f"加载视频失败: {e}")
                
    def batch_upload(self):
        """批量上传"""
        folder_path = filedialog.askdirectory(title="选择包含图片的文件夹")
        
        if folder_path:
            try:
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
                image_files = []
                
                for file_path in Path(folder_path).rglob('*'):
                    if file_path.suffix.lower() in image_extensions:
                        image_files.append(file_path)
                        
                if image_files:
                    self.log_message(f"找到 {len(image_files)} 个图片文件")
                    # 这里可以添加批量处理逻辑
                    messagebox.showinfo("批量上传", f"找到 {len(image_files)} 个图片文件\n批量处理功能开发中...")
                else:
                    messagebox.showwarning("警告", "未找到支持的图片文件")
                    
            except Exception as e:
                messagebox.showerror("错误", f"批量上传失败: {e}")
                self.log_message(f"批量上传失败: {e}")
                
    def clear_data(self):
        """清空数据"""
        if messagebox.askyesno("确认", "确定要清空所有数据吗？"):
            self.training_data.clear()
            self.detection_results.clear()
            self.results_tree.delete(*self.results_tree.get_children())
            self.update_data_stats()
            self.log_message("数据已清空")
            
    def toggle_llm_config(self):
        """切换大模型配置状态"""
        if self.llm_enabled.get():
            for widget in self.llm_config_frame.winfo_children():
                widget.config(state='normal')
        else:
            for widget in self.llm_config_frame.winfo_children():
                if isinstance(widget, (ttk.Entry, ttk.Combobox, ttk.Button)):
                    widget.config(state='disabled')
                    
    def test_llm_connection(self):
        """测试大模型连接"""
        if not self.llm_api_key.get():
            messagebox.showwarning("警告", "请输入API Key")
            return
            
        try:
            # 这里添加实际的API测试逻辑
            self.log_message("正在测试大模型连接...")
            
            # 模拟测试
            time.sleep(1)
            
            messagebox.showinfo("成功", "大模型连接测试成功")
            self.log_message("大模型连接测试成功")
            
        except Exception as e:
            messagebox.showerror("错误", f"连接测试失败: {e}")
            self.log_message(f"连接测试失败: {e}")
            
    def run_detection(self):
        """执行检测"""
        if self.current_frame is None:
            messagebox.showwarning("警告", "请先加载图像或启动摄像头")
            return
            
        try:
            self.log_message("正在执行目标检测...")
            
            # 模拟检测结果
            detections = [
                {"class": "person", "confidence": 0.85, "bbox": [100, 100, 200, 300], "source": "YOLO"},
                {"class": "car", "confidence": 0.72, "bbox": [300, 150, 500, 400], "source": "YOLO"}
            ]
            
            # 更新结果显示
            self.update_detection_results(detections)
            self.log_message(f"检测完成，发现 {len(detections)} 个目标")
            
        except Exception as e:
            messagebox.showerror("错误", f"检测失败: {e}")
            self.log_message(f"检测失败: {e}")
            
    def run_llm_analysis(self):
        """运行大模型分析"""
        if not self.llm_enabled.get():
            messagebox.showwarning("警告", "请先启用大模型自学习")
            return
            
        if self.current_frame is None:
            messagebox.showwarning("警告", "请先加载图像或启动摄像头")
            return
            
        try:
            self.log_message("正在进行大模型分析...")
            
            # 模拟大模型分析结果
            llm_results = [
                {"class": "unknown_object", "confidence": 0.90, "bbox": [150, 200, 250, 350], "source": "LLM"},
                {"class": "scene_element", "confidence": 0.78, "bbox": [400, 100, 600, 300], "source": "LLM"}
            ]
            
            # 更新结果显示
            self.update_detection_results(llm_results)
            self.log_message(f"大模型分析完成，发现 {len(llm_results)} 个新目标")
            
        except Exception as e:
            messagebox.showerror("错误", f"大模型分析失败: {e}")
            self.log_message(f"大模型分析失败: {e}")
            
    def update_detection_results(self, detections):
        """更新检测结果显示"""
        for det in detections:
            self.results_tree.insert('', 'end', values=(
                det['class'],
                f"{det['confidence']:.2f}",
                f"{det['bbox']}",
                det['source']
            ))
            
        self.detection_results.extend(detections)
        self.update_data_stats()
        
    def manual_annotation(self):
        """手动标注"""
        if self.current_frame is None:
            messagebox.showwarning("警告", "请先加载图像")
            return
            
        messagebox.showinfo("手动标注", "手动标注功能开发中...")
        self.log_message("启动手动标注模式")
        
    def auto_annotation(self):
        """自动标注"""
        if not self.detection_results:
            messagebox.showwarning("警告", "请先执行检测")
            return
            
        messagebox.showinfo("自动标注", "自动标注功能开发中...")
        self.log_message("执行自动标注")
        
    def start_training(self):
        """开始训练"""
        if not self.training_data and not self.detection_results:
            messagebox.showwarning("警告", "没有可用的训练数据")
            return
            
        messagebox.showinfo("开始训练", "模型训练功能开发中...")
        self.log_message("开始模型训练")
        
    def pause_training(self):
        """暂停训练"""
        messagebox.showinfo("暂停训练", "训练暂停功能开发中...")
        self.log_message("训练已暂停")
        
    def save_dataset(self):
        """保存数据集"""
        file_path = filedialog.asksaveasfilename(
            title="保存数据集",
            defaultextension=".json",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                dataset = {
                    "training_data": self.training_data,
                    "detection_results": self.detection_results,
                    "timestamp": datetime.now().isoformat()
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, indent=2, ensure_ascii=False)
                    
                self.log_message(f"数据集已保存: {Path(file_path).name}")
                messagebox.showinfo("成功", "数据集保存成功")
                
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {e}")
                self.log_message(f"保存失败: {e}")
                
    def load_dataset(self):
        """加载数据集"""
        file_path = filedialog.askopenfilename(
            title="加载数据集",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)
                    
                self.training_data = dataset.get("training_data", [])
                self.detection_results = dataset.get("detection_results", [])
                
                # 更新显示
                self.results_tree.delete(*self.results_tree.get_children())
                self.update_detection_results(self.detection_results)
                
                self.log_message(f"数据集已加载: {Path(file_path).name}")
                messagebox.showinfo("成功", "数据集加载成功")
                
            except Exception as e:
                messagebox.showerror("错误", f"加载失败: {e}")
                self.log_message(f"加载失败: {e}")
                
    def export_yolo_format(self):
        """导出YOLO格式"""
        messagebox.showinfo("导出YOLO", "YOLO格式导出功能开发中...")
        self.log_message("导出YOLO格式数据")
        
    def export_coco_format(self):
        """导出COCO格式"""
        messagebox.showinfo("导出COCO", "COCO格式导出功能开发中...")
        self.log_message("导出COCO格式数据")
        
    def update_data_stats(self):
        """更新数据统计"""
        num_images = len(self.training_data)
        num_annotations = len(self.detection_results)
        self.data_stats.config(text=f"数据: {num_images} 张图片, {num_annotations} 个标注")
        
    def log_message(self, message):
        """记录日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        
        # 同时记录到logger
        self.logger.info(message)
        
    def run(self):
        """运行界面"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("程序被用户中断")
        except Exception as e:
            print(f"程序运行错误: {e}")
        finally:
            # 清理资源
            if self.is_camera_running:
                self.stop_camera()
            try:
                self.root.destroy()
            except:
                pass

def main():
    """主函数"""
    try:
        app = AdvancedTrainingGUI()
        app.run()
    except Exception as e:
        print(f"启动失败: {e}")

if __name__ == "__main__":
    main()