#!/usr/bin/env python3
"""
简化的YOLOS GUI界面
基于BaseYOLOSGUI的简化实现
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import random
from typing import List, Dict
from pathlib import Path
import sys
import os

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.gui.base_gui import BaseYOLOSGUI

class SimpleYOLOSGUI(BaseYOLOSGUI):
    """简化的YOLOS图形界面"""
    
    def __init__(self):
        # 模拟检测类别
        self.classes = ['person', 'car', 'dog', 'cat', 'bicycle', 'bottle', 'chair', 'book']
        super().__init__(title="YOLOS - 简化目标检测系统", config_file="simple_gui_config.json")
        
    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建控制面板和显示区域
        control_frame = self.create_control_panel(main_frame)
        display_frame = self.create_display_area(main_frame)
        
        # 添加参数控制
        self.setup_parameter_controls(control_frame)
        
        # 添加日志区域
        self.setup_log_area(control_frame)
        
        # 创建状态栏
        self.create_status_bar(self.root)
        
    def setup_parameter_controls(self, parent):
        """设置参数控制"""
        # 参数设置
        params_frame = ttk.LabelFrame(parent, text="检测参数")
        params_frame.pack(fill=tk.X, pady=5)
        
        # 置信度阈值
        ttk.Label(params_frame, text="置信度阈值:").pack(anchor=tk.W)
        self.conf_var = tk.DoubleVar(value=self.config_manager.get('confidence_threshold'))
        conf_scale = ttk.Scale(params_frame, from_=0.1, to=1.0, 
                              variable=self.conf_var, orient=tk.HORIZONTAL,
                              command=self.update_conf_label)
        conf_scale.pack(fill=tk.X, pady=2)
        
        self.conf_label = ttk.Label(params_frame, text=f"当前值: {self.config_manager.get('confidence_threshold'):.2f}")
        self.conf_label.pack(anchor=tk.W)
        
        # NMS阈值
        ttk.Label(params_frame, text="NMS阈值:").pack(anchor=tk.W, pady=(10, 0))
        self.nms_var = tk.DoubleVar(value=self.config_manager.get('nms_threshold'))
        nms_scale = ttk.Scale(params_frame, from_=0.1, to=1.0, 
                             variable=self.nms_var, orient=tk.HORIZONTAL,
                             command=self.update_nms_label)
        nms_scale.pack(fill=tk.X, pady=2)
        
        self.nms_label = ttk.Label(params_frame, text=f"当前值: {self.config_manager.get('nms_threshold'):.2f}")
        self.nms_label.pack(anchor=tk.W)
    
    def setup_log_area(self, parent):
        """设置日志区域"""
        # 日志区域
        log_frame = ttk.LabelFrame(parent, text="运行日志")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 创建滚动文本框
        log_scroll_frame = ttk.Frame(log_frame)
        log_scroll_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text = tk.Text(log_scroll_frame, height=8, wrap=tk.WORD)
        log_scrollbar = ttk.Scrollbar(log_scroll_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    def update_conf_label(self, value):
        """更新置信度标签"""
        self.conf_label.config(text=f"当前值: {float(value):.2f}")
        
    def update_nms_label(self, value):
        """更新NMS标签"""
        self.nms_label.config(text=f"当前值: {float(value):.2f}")
        
    def log_message(self, message):
        """记录日志信息"""
        import time
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        if hasattr(self, 'log_text'):
            self.log_text.insert(tk.END, log_entry)
            self.log_text.see(tk.END)
            
            # 限制日志长度
            lines = self.log_text.get("1.0", tk.END).split('\n')
            if len(lines) > 100:
                self.log_text.delete("1.0", "10.0")
        
        # 调用父类方法
        super().log_message(message)
            

            

        

            

                

        
    def perform_detection(self, frame):
        """执行目标检测"""
        try:
            # 模拟检测结果
            h, w = frame.shape[:2]
            
            # 获取当前参数值
            conf_threshold = self.conf_var.get()
            nms_threshold = self.nms_var.get()
            
            # 模拟动态检测结果
            confidence = random.uniform(conf_threshold, 1.0)
            
            # 模拟检测框位置的微小变化
            offset_x = random.randint(-20, 20)
            offset_y = random.randint(-20, 20)
            
            x1 = max(0, w//4 + offset_x)
            y1 = max(0, h//4 + offset_y)
            x2 = min(w, 3*w//4 + offset_x)
            y2 = min(h, 3*h//4 + offset_y)
            
            # 随机选择类别
            class_name = random.choice(self.classes)
            
            # 绘制检测框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name}: {confidence:.2f}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 更新检测结果
            self.detection_results = [{
                'class': class_name,
                'confidence': confidence,
                'bbox': [x1, y1, x2-x1, y2-y1]
            }]
            
            # 显示实时参数信息
            info_text = f"Conf: {conf_threshold:.2f} | NMS: {nms_threshold:.2f}"
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 显示帧率信息
            fps_text = f"FPS: ~30"
            cv2.putText(frame, fps_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 更新状态
            if hasattr(self, 'status_var'):
                self.status_var.set(f"检测到 1 个目标")
            
        except Exception as e:
            self.log_message(f"检测失败: {e}")
            
        return frame
    
    def load_model(self, model_path: str) -> bool:
        """加载模型（简化版本，仅模拟）"""
        try:
            self.log_message(f"模拟加载模型: {model_path}")
            return True
        except Exception as e:
            self.log_message(f"加载模型失败: {e}")
            return False
    
    def process_frame(self, frame):
        """处理单帧图像"""
        if self.is_detecting:
            return self.perform_detection(frame)
        return frame
    
    def get_detection_results(self) -> List[Dict]:
        """获取检测结果"""
        return getattr(self, 'detection_results', [])
    
    def on_model_changed(self, model_path: str):
        """模型变更回调"""
        self.log_message(f"切换到模型: {model_path}")
        self.load_model(model_path)
        

            

                

                

            

            

            

        


def main():
    """主函数"""
    app = SimpleYOLOSGUI()
    app.run()

if __name__ == "__main__":
    main()