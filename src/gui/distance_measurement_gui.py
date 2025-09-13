"""距离测量GUI界面 - 集成摄像头测距功能

基于hybrid_stable_gui架构，提供完整的距离测量功能界面。
包括相机标定、实时测距、结果显示等功能。
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any

# 导入距离测量相关模块
try:
    from ..recognition.distance_estimator import DistanceEstimator, RealTimeDistanceEstimator
    from ..recognition.enhanced_object_detector import EnhancedObjectDetector
    from ..recognition.camera_calibration_tool import CameraCalibrationTool
    from .base_gui import BaseGUI
except ImportError:
    # 开发环境下的导入
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from recognition.distance_estimator import DistanceEstimator, RealTimeDistanceEstimator
    from recognition.enhanced_object_detector import EnhancedObjectDetector
    from recognition.camera_calibration_tool import CameraCalibrationTool
    from gui.base_gui import BaseYOLOSGUI


class DistanceMeasurementGUI(BaseYOLOSGUI):
    """距离测量GUI主类"""
    
    def __init__(self):
        super().__init__()
        self.root.title("YOLOS - 摄像头距离测量系统")
        self.root.geometry("1400x900")
        
        # 距离测量组件
        self.distance_estimator = DistanceEstimator()
        self.realtime_estimator = RealTimeDistanceEstimator()
        self.object_detector = EnhancedObjectDetector()
        self.calibration_tool = CameraCalibrationTool()
        
        # GUI状态变量
        self.measurement_mode = tk.StringVar(value="single")
        self.object_type = tk.StringVar(value="A4_paper")
        self.known_width = tk.DoubleVar(value=21.0)
        self.distance_unit = tk.StringVar(value="cm")
        self.show_detection_info = tk.BooleanVar(value=True)
        self.auto_measure = tk.BooleanVar(value=False)
        
        # 测量结果
        self.current_distance = None
        self.measurement_history = []
        self.calibration_status = "未标定"
        
        # 界面组件
        self.distance_label = None
        self.history_tree = None
        self.calibration_status_label = None
        
        # 初始化界面
        self.setup_ui()
        self.load_calibration_status()
    
    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建左侧控制面板
        control_frame = self.create_control_panel(main_frame)
        
        # 创建右侧显示区域
        display_frame = self.create_display_area(main_frame)
        
        # 创建底部状态栏
        self.create_status_bar(self.root)
        
        # 设置样式
        self.setup_styles()
    
    def create_control_panel(self, parent) -> ttk.Frame:
        """创建控制面板"""
        control_frame = ttk.LabelFrame(parent, text="距离测量控制", width=400)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)
        
        # 摄像头控制区域
        self.create_camera_controls(control_frame)
        
        # 标定控制区域
        self.create_calibration_controls(control_frame)
        
        # 测量参数区域
        self.create_measurement_controls(control_frame)
        
        # 测量结果区域
        self.create_results_display(control_frame)
        
        # 历史记录区域
        self.create_history_display(control_frame)
        
        return control_frame
    
    def create_camera_controls(self, parent):
        """创建摄像头控制区域"""
        camera_frame = ttk.LabelFrame(parent, text="摄像头控制")
        camera_frame.pack(fill=tk.X, pady=5)
        
        # 摄像头索引
        index_frame = ttk.Frame(camera_frame)
        index_frame.pack(fill=tk.X, pady=2)
        ttk.Label(index_frame, text="摄像头索引:").pack(side=tk.LEFT)
        self.camera_var = tk.StringVar(value="0")
        ttk.Entry(index_frame, textvariable=self.camera_var, width=5).pack(side=tk.RIGHT)
        
        # 控制按钮
        button_frame = ttk.Frame(camera_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        self.start_btn = ttk.Button(button_frame, text="启动摄像头", 
                                   command=self.start_camera)
        self.start_btn.pack(side=tk.LEFT, padx=2)
        
        self.stop_btn = ttk.Button(button_frame, text="停止摄像头", 
                                  command=self.stop_camera, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2)
    
    def create_calibration_controls(self, parent):
        """创建标定控制区域"""
        calib_frame = ttk.LabelFrame(parent, text="相机标定")
        calib_frame.pack(fill=tk.X, pady=5)
        
        # 标定状态显示
        status_frame = ttk.Frame(calib_frame)
        status_frame.pack(fill=tk.X, pady=2)
        ttk.Label(status_frame, text="标定状态:").pack(side=tk.LEFT)
        self.calibration_status_label = ttk.Label(status_frame, text=self.calibration_status, 
                                                 foreground="red")
        self.calibration_status_label.pack(side=tk.RIGHT)
        
        # 标定按钮
        calib_btn_frame = ttk.Frame(calib_frame)
        calib_btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(calib_btn_frame, text="交互式标定", 
                  command=self.start_interactive_calibration).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(calib_btn_frame, text="从图片标定", 
                  command=self.calibrate_from_images).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(calib_btn_frame, text="标定历史", 
                  command=self.show_calibration_history).pack(side=tk.LEFT, padx=2)
    
    def create_measurement_controls(self, parent):
        """创建测量参数控制区域"""
        measure_frame = ttk.LabelFrame(parent, text="测量参数")
        measure_frame.pack(fill=tk.X, pady=5)
        
        # 测量模式
        mode_frame = ttk.Frame(measure_frame)
        mode_frame.pack(fill=tk.X, pady=2)
        ttk.Label(mode_frame, text="测量模式:").pack(side=tk.LEFT)
        mode_combo = ttk.Combobox(mode_frame, textvariable=self.measurement_mode, 
                                 values=["single", "continuous"], state="readonly", width=10)
        mode_combo.pack(side=tk.RIGHT)
        mode_combo.bind('<<ComboboxSelected>>', self.on_mode_change)
        
        # 物体类型
        object_frame = ttk.Frame(measure_frame)
        object_frame.pack(fill=tk.X, pady=2)
        ttk.Label(object_frame, text="物体类型:").pack(side=tk.LEFT)
        object_combo = ttk.Combobox(object_frame, textvariable=self.object_type,
                                   values=list(self.calibration_tool.known_objects.keys()),
                                   state="readonly", width=12)
        object_combo.pack(side=tk.RIGHT)
        object_combo.bind('<<ComboboxSelected>>', self.on_object_change)
        
        # 已知宽度
        width_frame = ttk.Frame(measure_frame)
        width_frame.pack(fill=tk.X, pady=2)
        ttk.Label(width_frame, text="已知宽度:").pack(side=tk.LEFT)
        width_entry = ttk.Entry(width_frame, textvariable=self.known_width, width=8)
        width_entry.pack(side=tk.RIGHT, padx=(0, 2))
        ttk.Label(width_frame, textvariable=self.distance_unit).pack(side=tk.RIGHT)
        
        # 显示选项
        option_frame = ttk.Frame(measure_frame)
        option_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(option_frame, text="显示检测信息", 
                       variable=self.show_detection_info).pack(anchor=tk.W)
        ttk.Checkbutton(option_frame, text="自动测量", 
                       variable=self.auto_measure).pack(anchor=tk.W)
        
        # 测量按钮
        measure_btn = ttk.Button(measure_frame, text="单次测量", 
                               command=self.single_measurement, style="Accent.TButton")
        measure_btn.pack(pady=5)
    
    def create_results_display(self, parent):
        """创建结果显示区域"""
        result_frame = ttk.LabelFrame(parent, text="测量结果")
        result_frame.pack(fill=tk.X, pady=5)
        
        # 当前距离显示
        self.distance_label = ttk.Label(result_frame, text="距离: -- cm", 
                                       font=("Arial", 14, "bold"), foreground="blue")
        self.distance_label.pack(pady=10)
        
        # 详细信息
        self.info_text = tk.Text(result_frame, height=4, width=40, font=("Consolas", 9))
        self.info_text.pack(pady=5)
        
        # 保存结果按钮
        ttk.Button(result_frame, text="保存当前结果", 
                  command=self.save_current_result).pack(pady=2)
    
    def create_history_display(self, parent):
        """创建历史记录显示区域"""
        history_frame = ttk.LabelFrame(parent, text="测量历史")
        history_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 创建树形视图
        columns = ("时间", "物体", "距离", "精度")
        self.history_tree = ttk.Treeview(history_frame, columns=columns, show="headings", height=6)
        
        # 设置列标题
        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=80)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        
        # 布局
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 清空历史按钮
        ttk.Button(history_frame, text="清空历史", 
                  command=self.clear_history).pack(pady=2)
    
    def create_display_area(self, parent) -> ttk.Frame:
        """创建显示区域"""
        display_frame = ttk.LabelFrame(parent, text="视频显示")
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 视频显示画布
        self.video_canvas = tk.Canvas(display_frame, bg="black", width=800, height=600)
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 在画布上显示提示文本
        self.video_canvas.create_text(400, 300, text="请启动摄像头", 
                                     fill="white", font=("Arial", 16))
        
        return display_frame
    
    def setup_styles(self):
        """设置界面样式"""
        style = ttk.Style()
        style.configure("Accent.TButton", foreground="white", background="#0078d4")
    
    def load_calibration_status(self):
        """加载标定状态"""
        try:
            summary = self.calibration_tool.get_calibration_summary()
            if summary.get('is_calibrated', False):
                focal_length = summary['latest_calibration']['focal_length']
                self.calibration_status = f"已标定 (f={focal_length:.1f})"
                self.calibration_status_label.config(text=self.calibration_status, foreground="green")
            else:
                self.calibration_status = "未标定"
                self.calibration_status_label.config(text=self.calibration_status, foreground="red")
        except Exception as e:
            self.logger.error(f"加载标定状态失败: {e}")
    
    def start_interactive_calibration(self):
        """启动交互式标定"""
        def calibration_thread():
            try:
                # 停止当前摄像头
                if self.is_camera_running:
                    self.stop_camera()
                    time.sleep(1)
                
                # 启动标定
                camera_id = int(self.camera_var.get())
                object_type = self.object_type.get()
                
                self.calibration_tool.interactive_calibration(camera_id, object_type)
                
                # 更新标定状态
                self.root.after(0, self.load_calibration_status)
                
            except Exception as e:
                self.logger.error(f"交互式标定失败: {e}")
                messagebox.showerror("标定失败", f"标定过程中出现错误: {e}")
        
        # 在新线程中运行标定
        threading.Thread(target=calibration_thread, daemon=True).start()
    
    def calibrate_from_images(self):
        """从图片进行标定"""
        # 选择图片目录
        image_dir = filedialog.askdirectory(title="选择包含标定图片的目录")
        if not image_dir:
            return
        
        # 获取距离信息
        dialog = DistanceInputDialog(self.root, image_dir)
        if dialog.result:
            distances = dialog.result
            object_type = self.object_type.get()
            
            def calibration_thread():
                try:
                    success = self.calibration_tool.batch_calibration_from_images(
                        image_dir, object_type, distances)
                    
                    if success:
                        self.root.after(0, self.load_calibration_status)
                        messagebox.showinfo("标定成功", "批量标定完成！")
                    else:
                        messagebox.showerror("标定失败", "批量标定失败，请检查图片和参数")
                        
                except Exception as e:
                    self.logger.error(f"批量标定失败: {e}")
                    messagebox.showerror("标定失败", f"标定过程中出现错误: {e}")
            
            threading.Thread(target=calibration_thread, daemon=True).start()
    
    def show_calibration_history(self):
        """显示标定历史"""
        history_window = CalibrationHistoryWindow(self.root, self.calibration_tool)
    
    def on_mode_change(self, event=None):
        """测量模式改变事件"""
        mode = self.measurement_mode.get()
        if mode == "continuous":
            self.auto_measure.set(True)
        else:
            self.auto_measure.set(False)
    
    def on_object_change(self, event=None):
        """物体类型改变事件"""
        object_type = self.object_type.get()
        if object_type in self.calibration_tool.known_objects:
            obj_info = self.calibration_tool.known_objects[object_type]
            self.known_width.set(obj_info['width'])
            self.distance_unit.set(obj_info['unit'])
    
    def single_measurement(self):
        """执行单次测量"""
        if not self.is_camera_running:
            messagebox.showwarning("警告", "请先启动摄像头")
            return
        
        if self.distance_estimator.focal_length is None:
            messagebox.showwarning("警告", "请先进行相机标定")
            return
        
        try:
            # 获取当前帧
            if hasattr(self, 'current_frame') and self.current_frame is not None:
                known_width = self.known_width.get()
                result = self.distance_estimator.estimate_distance(self.current_frame, known_width)
                
                if result:
                    self.display_measurement_result(result)
                    self.add_to_history(result)
                else:
                    messagebox.showinfo("测量失败", "未检测到目标物体")
            else:
                messagebox.showwarning("警告", "无法获取摄像头图像")
                
        except Exception as e:
            self.logger.error(f"测量失败: {e}")
            messagebox.showerror("测量失败", f"测量过程中出现错误: {e}")
    
    def display_measurement_result(self, result: Dict[str, Any]):
        """显示测量结果"""
        distance = result['distance']
        unit = self.distance_unit.get()
        
        # 更新距离显示
        self.distance_label.config(text=f"距离: {distance:.1f} {unit}")
        
        # 更新详细信息
        info_text = f"像素宽度: {result['pixel_width']:.1f}\n"
        info_text += f"已知宽度: {self.known_width.get():.1f} {unit}\n"
        info_text += f"焦距: {self.distance_estimator.focal_length:.1f}\n"
        info_text += f"检测置信度: {result.get('confidence', 'N/A')}"
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_text)
        
        self.current_distance = result
    
    def add_to_history(self, result: Dict[str, Any]):
        """添加到历史记录"""
        timestamp = time.strftime("%H:%M:%S")
        object_type = self.object_type.get()
        distance = f"{result['distance']:.1f} {self.distance_unit.get()}"
        confidence = f"{result.get('confidence', 0):.2f}" if 'confidence' in result else "N/A"
        
        # 添加到树形视图
        self.history_tree.insert("", 0, values=(timestamp, object_type, distance, confidence))
        
        # 保存到历史列表
        history_item = {
            'timestamp': timestamp,
            'object_type': object_type,
            'result': result,
            'known_width': self.known_width.get(),
            'unit': self.distance_unit.get()
        }
        self.measurement_history.append(history_item)
        
        # 限制历史记录数量
        if len(self.measurement_history) > 100:
            self.measurement_history.pop(0)
            # 删除树形视图中最后一项
            items = self.history_tree.get_children()
            if items:
                self.history_tree.delete(items[-1])
    
    def save_current_result(self):
        """保存当前结果"""
        if self.current_distance is None:
            messagebox.showwarning("警告", "没有可保存的测量结果")
            return
        
        filename = filedialog.asksaveasfilename(
            title="保存测量结果",
            defaultextension=".json",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        
        if filename:
            try:
                save_data = {
                    'measurement_result': self.current_distance,
                    'object_type': self.object_type.get(),
                    'known_width': self.known_width.get(),
                    'unit': self.distance_unit.get(),
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'calibration_info': {
                        'focal_length': self.distance_estimator.focal_length,
                        'calibration_data': getattr(self.distance_estimator, 'calibration_data', None)
                    }
                }
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, indent=2, ensure_ascii=False)
                
                messagebox.showinfo("保存成功", f"测量结果已保存到: {filename}")
                
            except Exception as e:
                self.logger.error(f"保存结果失败: {e}")
                messagebox.showerror("保存失败", f"保存过程中出现错误: {e}")
    
    def clear_history(self):
        """清空历史记录"""
        if messagebox.askyesno("确认", "确定要清空所有历史记录吗？"):
            self.measurement_history.clear()
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)
    
    def update_video_stream(self):
        """更新视频流显示"""
        if self.is_camera_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                
                # 如果启用自动测量
                if self.auto_measure.get() and self.distance_estimator.focal_length is not None:
                    try:
                        known_width = self.known_width.get()
                        result = self.distance_estimator.estimate_distance(frame, known_width)
                        if result:
                            self.display_measurement_result(result)
                            
                            # 在图像上绘制测量结果
                            if self.show_detection_info.get():
                                frame = self.draw_measurement_overlay(frame, result)
                    except Exception as e:
                        self.logger.error(f"自动测量失败: {e}")
                
                # 显示图像
                self.display_frame(frame)
        
        # 继续更新
        if self.is_camera_running:
            self.root.after(30, self.update_video_stream)
    
    def draw_measurement_overlay(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """在图像上绘制测量信息覆盖层"""
        overlay_frame = frame.copy()
        
        # 绘制检测框
        if 'bbox' in result:
            x, y, w, h = result['bbox']
            cv2.rectangle(overlay_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 绘制中心点
            center_x, center_y = x + w // 2, y + h // 2
            cv2.circle(overlay_frame, (center_x, center_y), 5, (255, 0, 0), -1)
            
            # 显示距离信息
            distance_text = f"Distance: {result['distance']:.1f} {self.distance_unit.get()}"
            cv2.putText(overlay_frame, distance_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示像素宽度
            pixel_text = f"Pixel Width: {result['pixel_width']:.1f}"
            cv2.putText(overlay_frame, pixel_text, (x, y + h + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return overlay_frame
    
    def display_frame(self, frame: np.ndarray):
        """显示视频帧"""
        try:
            # 调整图像大小以适应画布
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                # 保持宽高比缩放
                h, w = frame.shape[:2]
                scale = min(canvas_width / w, canvas_height / h)
                new_w, new_h = int(w * scale), int(h * scale)
                
                frame_resized = cv2.resize(frame, (new_w, new_h))
                
                # 转换为RGB并创建PhotoImage
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image)
                
                # 清空画布并显示图像
                self.video_canvas.delete("all")
                x = (canvas_width - new_w) // 2
                y = (canvas_height - new_h) // 2
                self.video_canvas.create_image(x, y, anchor=tk.NW, image=photo)
                
                # 保持引用
                self.video_canvas.image = photo
                
        except Exception as e:
            self.logger.error(f"显示帧失败: {e}")


class DistanceInputDialog:
    """距离输入对话框"""
    
    def __init__(self, parent, image_dir):
        self.result = None
        
        # 获取图片文件
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        image_files.sort()
        
        if not image_files:
            messagebox.showerror("错误", "目录中没有找到图片文件")
            return
        
        # 创建对话框
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("输入图片对应的距离")
        self.dialog.geometry("400x300")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # 说明文本
        ttk.Label(self.dialog, text=f"请为以下 {len(image_files)} 张图片输入对应的距离 (cm):").pack(pady=10)
        
        # 创建输入框
        self.entries = []
        frame = ttk.Frame(self.dialog)
        frame.pack(fill=tk.BOTH, expand=True, padx=20)
        
        # 滚动框
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        for i, img_file in enumerate(image_files):
            row_frame = ttk.Frame(scrollable_frame)
            row_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(row_frame, text=f"{img_file.name}:").pack(side=tk.LEFT)
            entry = ttk.Entry(row_frame, width=10)
            entry.pack(side=tk.RIGHT)
            self.entries.append(entry)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 按钮
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="确定", command=self.ok_clicked).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="取消", command=self.cancel_clicked).pack(side=tk.LEFT, padx=5)
        
        # 居中显示
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (self.dialog.winfo_width() // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (self.dialog.winfo_height() // 2)
        self.dialog.geometry(f"+{x}+{y}")
    
    def ok_clicked(self):
        try:
            distances = []
            for entry in self.entries:
                value = entry.get().strip()
                if not value:
                    raise ValueError("所有距离值都必须填写")
                distances.append(float(value))
            
            self.result = distances
            self.dialog.destroy()
            
        except ValueError as e:
            messagebox.showerror("输入错误", f"请输入有效的数字: {e}")
    
    def cancel_clicked(self):
        self.dialog.destroy()


class CalibrationHistoryWindow:
    """标定历史窗口"""
    
    def __init__(self, parent, calibration_tool):
        self.calibration_tool = calibration_tool
        
        # 创建窗口
        self.window = tk.Toplevel(parent)
        self.window.title("相机标定历史")
        self.window.geometry("800x600")
        self.window.transient(parent)
        
        # 创建界面
        self.create_interface()
        self.load_history()
    
    def create_interface(self):
        # 主框架
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        ttk.Label(main_frame, text="相机标定历史记录", font=("Arial", 14, "bold")).pack(pady=(0, 10))
        
        # 历史记录树形视图
        columns = ("时间", "物体类型", "焦距", "样本数", "标准差")
        self.history_tree = ttk.Treeview(main_frame, columns=columns, show="headings")
        
        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=120)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        
        # 布局
        tree_frame = ttk.Frame(main_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 详细信息区域
        detail_frame = ttk.LabelFrame(main_frame, text="详细信息")
        detail_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.detail_text = tk.Text(detail_frame, height=8, font=("Consolas", 9))
        self.detail_text.pack(fill=tk.X, padx=5, pady=5)
        
        # 绑定选择事件
        self.history_tree.bind('<<TreeviewSelect>>', self.on_select)
        
        # 按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(10, 0))
        
        ttk.Button(button_frame, text="应用选中标定", command=self.apply_calibration).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="删除选中记录", command=self.delete_calibration).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="关闭", command=self.window.destroy).pack(side=tk.LEFT, padx=5)
    
    def load_history(self):
        # 清空现有项目
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        
        # 加载历史记录
        for i, record in enumerate(self.calibration_tool.calibration_history):
            timestamp = record.get('timestamp', 'N/A')[:19]  # 只显示日期时间部分
            object_type = record.get('object_type', 'N/A')
            focal_length = f"{record.get('focal_length', 0):.2f}"
            sample_count = str(record.get('sample_count', 0))
            std_dev = f"{record.get('std_deviation', 0):.2f}"
            
            self.history_tree.insert("", tk.END, iid=i, 
                                    values=(timestamp, object_type, focal_length, sample_count, std_dev))
    
    def on_select(self, event):
        selection = self.history_tree.selection()
        if selection:
            item_id = int(selection[0])
            record = self.calibration_tool.calibration_history[item_id]
            
            # 显示详细信息
            detail_info = f"标定时间: {record.get('timestamp', 'N/A')}\n"
            detail_info += f"物体类型: {record.get('object_type', 'N/A')}\n"
            detail_info += f"已知宽度: {record.get('known_width', 'N/A')} {record.get('unit', '')}\n"
            detail_info += f"焦距: {record.get('focal_length', 0):.4f}\n"
            detail_info += f"标准差: {record.get('std_deviation', 0):.4f}\n"
            detail_info += f"样本数量: {record.get('sample_count', 0)}\n"
            detail_info += f"标定方法: {record.get('calibration_method', 'interactive')}\n"
            
            if 'samples' in record:
                detail_info += f"\n样本详情:\n"
                for j, sample in enumerate(record['samples'][:5]):  # 只显示前5个样本
                    detail_info += f"  样本{j+1}: 距离={sample.get('actual_distance', 'N/A')}cm, "
                    detail_info += f"像素宽度={sample.get('pixel_width', 'N/A'):.1f}\n"
                if len(record['samples']) > 5:
                    detail_info += f"  ... 还有 {len(record['samples']) - 5} 个样本\n"
            
            self.detail_text.delete(1.0, tk.END)
            self.detail_text.insert(1.0, detail_info)
    
    def apply_calibration(self):
        selection = self.history_tree.selection()
        if not selection:
            messagebox.showwarning("警告", "请先选择一个标定记录")
            return
        
        item_id = int(selection[0])
        record = self.calibration_tool.calibration_history[item_id]
        
        # 应用标定
        focal_length = record.get('focal_length')
        if focal_length:
            self.calibration_tool.distance_estimator.focal_length = focal_length
            self.calibration_tool.distance_estimator.calibration_data = record
            self.calibration_tool.distance_estimator._save_config()
            
            messagebox.showinfo("成功", f"已应用标定记录，焦距: {focal_length:.2f}")
            self.window.destroy()
        else:
            messagebox.showerror("错误", "标定记录数据不完整")
    
    def delete_calibration(self):
        selection = self.history_tree.selection()
        if not selection:
            messagebox.showwarning("警告", "请先选择一个标定记录")
            return
        
        if messagebox.askyesno("确认", "确定要删除选中的标定记录吗？"):
            item_id = int(selection[0])
            del self.calibration_tool.calibration_history[item_id]
            self.calibration_tool.save_calibration_history()
            self.load_history()
            self.detail_text.delete(1.0, tk.END)


if __name__ == "__main__":
    # 创建并运行GUI
    app = DistanceMeasurementGUI()
    app.run()