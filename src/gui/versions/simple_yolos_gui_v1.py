#!/usr/bin/env python3
"""
简化的YOLOS GUI界面
确保PC版本能够正常启动和运行
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

class SimpleYOLOSGUI:
    """简化的YOLOS图形界面"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("YOLOS - 目标检测系统")
        self.root.geometry("1200x800")
        
        # 状态变量
        self.camera = None
        self.is_detecting = False
        self.current_frame = None
        self.detection_results = []
        
        # 配置
        self.config = {
            'confidence_threshold': 0.5,
            'nms_threshold': 0.4,
            'model_path': 'models/yolov8n.pt',
            'camera_index': 0
        }
        
        # 初始化界面
        self.setup_ui()
        self.setup_menu()
        
        # 加载配置
        self.load_config()
        
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
        
        ttk.Label(camera_frame, text="摄像头索引:").pack(anchor=tk.W)
        self.camera_var = tk.StringVar(value=str(self.config['camera_index']))
        camera_entry = ttk.Entry(camera_frame, textvariable=self.camera_var, width=10)
        camera_entry.pack(anchor=tk.W, pady=2)
        
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
        self.conf_var = tk.DoubleVar(value=self.config['confidence_threshold'])
        conf_scale = ttk.Scale(params_frame, from_=0.1, to=1.0, 
                              variable=self.conf_var, orient=tk.HORIZONTAL)
        conf_scale.pack(fill=tk.X, pady=2)
        
        self.conf_label = ttk.Label(params_frame, text=f"{self.conf_var.get():.2f}")
        self.conf_label.pack(anchor=tk.W)
        conf_scale.configure(command=self.update_conf_label)
        
        ttk.Label(params_frame, text="NMS阈值:").pack(anchor=tk.W, pady=(10,0))
        self.nms_var = tk.DoubleVar(value=self.config['nms_threshold'])
        nms_scale = ttk.Scale(params_frame, from_=0.1, to=1.0, 
                             variable=self.nms_var, orient=tk.HORIZONTAL)
        nms_scale.pack(fill=tk.X, pady=2)
        
        self.nms_label = ttk.Label(params_frame, text=f"{self.nms_var.get():.2f}")
        self.nms_label.pack(anchor=tk.W)
        nms_scale.configure(command=self.update_nms_label)
        
        # 文件操作
        file_frame = ttk.LabelFrame(parent, text="文件操作")
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(file_frame, text="加载图片", 
                  command=self.load_image).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="加载视频", 
                  command=self.load_video).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="保存结果", 
                  command=self.save_results).pack(fill=tk.X, pady=2)
        
        # 状态信息
        status_frame = ttk.LabelFrame(parent, text="状态信息")
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_text = tk.Text(status_frame, height=8, width=30)
        scrollbar = ttk.Scrollbar(status_frame, orient=tk.VERTICAL, 
                                 command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=scrollbar.set)
        
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_message("YOLOS系统已启动")
        
    def setup_display_area(self, parent):
        """设置显示区域"""
        # 创建Canvas用于显示视频
        self.canvas = tk.Canvas(parent, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 显示提示信息
        self.canvas.create_text(400, 300, text="请启动摄像头或加载图片/视频", 
                               fill='white', font=('Arial', 16))
        
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
        file_menu.add_command(label="保存配置", command=self.save_config)
        file_menu.add_command(label="加载配置", command=self.load_config)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        
        # 工具菜单
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="工具", menu=tools_menu)
        tools_menu.add_command(label="模型管理", command=self.open_model_manager)
        tools_menu.add_command(label="数据集管理", command=self.open_dataset_manager)
        tools_menu.add_command(label="训练模型", command=self.open_training_dialog)
        
        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="使用说明", command=self.show_help)
        help_menu.add_command(label="关于", command=self.show_about)
        
    def update_conf_label(self, value):
        """更新置信度标签"""
        self.conf_label.config(text=f"{float(value):.2f}")
        
    def update_nms_label(self, value):
        """更新NMS标签"""
        self.nms_label.config(text=f"{float(value):.2f}")
        
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
            
    def start_camera(self):
        """启动摄像头"""
        try:
            camera_index = int(self.camera_var.get())
            self.camera = cv2.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                raise Exception(f"无法打开摄像头 {camera_index}")
                
            self.log_message(f"摄像头 {camera_index} 启动成功")
            
            # 更新按钮状态
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.detect_btn.config(state=tk.NORMAL)
            
            # 开始视频流
            self.update_video_stream()
            
        except Exception as e:
            self.log_message(f"启动摄像头失败: {e}")
            messagebox.showerror("错误", f"启动摄像头失败: {e}")
            
    def stop_camera(self):
        """停止摄像头"""
        if self.camera:
            self.camera.release()
            self.camera = None
            
        self.is_detecting = False
        
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
            
    def update_video_stream(self):
        """更新视频流"""
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            
            if ret:
                self.current_frame = frame.copy()
                
                # 如果启用检测，进行目标检测
                if self.is_detecting:
                    frame = self.perform_detection(frame)
                
                # 显示帧
                self.display_frame(frame)
                
            # 继续更新
            self.root.after(30, self.update_video_stream)
        
    def perform_detection(self, frame):
        """执行目标检测"""
        try:
            # 这里使用简化的检测逻辑
            # 在实际应用中，这里会调用YOLO模型
            
            # 模拟检测结果
            h, w = frame.shape[:2]
            
            # 绘制一个示例检测框
            cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 2)
            cv2.putText(frame, "Object: 0.85", (w//4, h//4-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示参数信息
            info_text = f"Conf: {self.conf_var.get():.2f} | NMS: {self.nms_var.get():.2f}"
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        except Exception as e:
            self.log_message(f"检测失败: {e}")
            
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
                from PIL import Image, ImageTk
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
            # 如果PIL不可用，显示错误信息
            self.canvas.delete("all")
            self.canvas.create_text(canvas_width//2, canvas_height//2, 
                                   text=f"显示错误: {e}", 
                                   fill='red', font=('Arial', 12))
            
    def load_image(self):
        """加载图片"""
        file_path = filedialog.askopenfilename(
            title="选择图片文件",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                image = cv2.imread(file_path)
                if image is not None:
                    self.current_frame = image
                    
                    # 如果启用检测，进行检测
                    if self.is_detecting:
                        image = self.perform_detection(image)
                    
                    self.display_frame(image)
                    self.log_message(f"已加载图片: {os.path.basename(file_path)}")
                else:
                    raise Exception("无法读取图片文件")
                    
            except Exception as e:
                self.log_message(f"加载图片失败: {e}")
                messagebox.showerror("错误", f"加载图片失败: {e}")
                
    def load_video(self):
        """加载视频"""
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv")]
        )
        
        if file_path:
            try:
                # 停止当前摄像头
                if self.camera:
                    self.stop_camera()
                
                # 打开视频文件
                self.camera = cv2.VideoCapture(file_path)
                
                if not self.camera.isOpened():
                    raise Exception("无法打开视频文件")
                
                self.log_message(f"已加载视频: {os.path.basename(file_path)}")
                
                # 更新按钮状态
                self.start_btn.config(state=tk.DISABLED)
                self.stop_btn.config(state=tk.NORMAL)
                self.detect_btn.config(state=tk.NORMAL)
                
                # 开始播放视频
                self.update_video_stream()
                
            except Exception as e:
                self.log_message(f"加载视频失败: {e}")
                messagebox.showerror("错误", f"加载视频失败: {e}")
                
    def save_results(self):
        """保存检测结果"""
        if self.current_frame is not None:
            file_path = filedialog.asksaveasfilename(
                title="保存检测结果",
                defaultextension=".jpg",
                filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
            )
            
            if file_path:
                try:
                    # 如果启用检测，保存带检测框的图像
                    if self.is_detecting:
                        result_frame = self.perform_detection(self.current_frame.copy())
                    else:
                        result_frame = self.current_frame
                    
                    cv2.imwrite(file_path, result_frame)
                    self.log_message(f"结果已保存: {os.path.basename(file_path)}")
                    
                except Exception as e:
                    self.log_message(f"保存失败: {e}")
                    messagebox.showerror("错误", f"保存失败: {e}")
        else:
            messagebox.showwarning("警告", "没有可保存的图像")
            
    def save_config(self):
        """保存配置"""
        config = {
            'confidence_threshold': self.conf_var.get(),
            'nms_threshold': self.nms_var.get(),
            'camera_index': int(self.camera_var.get()),
            'model_path': self.config.get('model_path', 'models/yolov8n.pt')
        }
        
        try:
            config_dir = Path('config')
            config_dir.mkdir(exist_ok=True)
            
            with open(config_dir / 'gui_config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                
            self.log_message("配置已保存")
            
        except Exception as e:
            self.log_message(f"保存配置失败: {e}")
            
    def load_config(self):
        """加载配置"""
        try:
            config_path = Path('config/gui_config.json')
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                self.config.update(config)
                
                # 更新界面
                self.conf_var.set(config.get('confidence_threshold', 0.5))
                self.nms_var.set(config.get('nms_threshold', 0.4))
                self.camera_var.set(str(config.get('camera_index', 0)))
                
                self.log_message("配置已加载")
                
        except Exception as e:
            self.log_message(f"加载配置失败: {e}")
            
    def open_model_manager(self):
        """打开模型管理器"""
        messagebox.showinfo("提示", "模型管理功能开发中...")
        
    def open_dataset_manager(self):
        """打开数据集管理器"""
        messagebox.showinfo("提示", "数据集管理功能开发中...")
        
    def open_training_dialog(self):
        """打开训练对话框"""
        messagebox.showinfo("提示", "模型训练功能开发中...")
        
    def show_help(self):
        """显示帮助信息"""
        help_text = """
YOLOS使用说明:

1. 摄像头检测:
   - 设置摄像头索引(通常为0)
   - 点击"启动摄像头"
   - 点击"开始检测"进行实时检测

2. 图片检测:
   - 点击"加载图片"选择图片文件
   - 启用检测后会自动显示检测结果

3. 视频检测:
   - 点击"加载视频"选择视频文件
   - 启用检测后会逐帧进行检测

4. 参数调整:
   - 置信度阈值: 控制检测的敏感度
   - NMS阈值: 控制重叠框的过滤

5. 结果保存:
   - 点击"保存结果"保存当前检测结果
        """
        
        messagebox.showinfo("使用说明", help_text)
        
    def show_about(self):
        """显示关于信息"""
        about_text = """
YOLOS目标检测系统 v1.0

基于YOLO算法的实时目标检测系统
支持摄像头、图片、视频检测

开发团队: YOLOS Team
版本: 1.0.0
        """
        
        messagebox.showinfo("关于", about_text)
        
    def on_closing(self):
        """关闭程序时的清理工作"""
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
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
    app = SimpleYOLOSGUI()
    app.run()

if __name__ == "__main__":
    main()