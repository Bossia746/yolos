#!/usr/bin/env python3
"""
YOLOS主界面入口 - 版本1
整合视频捕捉、图像识别和训练功能
专注于核心功能，避免过度扩展
"""

import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.gui.yolos_training_gui import YOLOSTrainingGUI
    from basic_pet_recognition_gui import BasicPetRecognitionGUI
except ImportError as e:
    print(f"导入模块失败: {e}")
    # 创建备用类
    class YOLOSTrainingGUI:
        def run(self): print("训练界面暂不可用")
    
    class BasicPetRecognitionGUI:
        def run(self): print("基础识别界面暂不可用")

class YOLOSMainGUI:
    """YOLOS主界面"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("YOLOS - 智能视频识别系统")
        self.root.geometry("600x400")
        self.root.resizable(True, True)
        
        # 设置图标和样式
        self.setup_style()
        self.create_interface()
        
        # 居中显示
        self.center_window()
    
    def setup_style(self):
        """设置界面样式"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # 自定义样式
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Subtitle.TLabel', font=('Arial', 10))
        style.configure('Action.TButton', font=('Arial', 11))
        
    def center_window(self):
        """窗口居中显示"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def create_interface(self):
        """创建界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="YOLOS 智能视频识别系统", 
                               style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        subtitle_label = ttk.Label(main_frame, 
                                  text="专注视频捕捉和图像识别的核心功能", 
                                  style='Subtitle.TLabel')
        subtitle_label.grid(row=1, column=0, columnspan=2, pady=(0, 20))
        
        # 功能按钮区域
        self.create_function_buttons(main_frame)
        
        # 状态栏
        self.create_status_bar(main_frame)
        
    def create_function_buttons(self, parent):
        """创建功能按钮"""
        # 按钮框架
        button_frame = ttk.LabelFrame(parent, text="核心功能", padding="15")
        button_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        
        # 训练界面按钮
        training_btn = ttk.Button(button_frame, text="🎯 训练界面", 
                                 command=self.open_training_gui,
                                 style='Action.TButton')
        training_btn.grid(row=0, column=0, padx=(0, 10), pady=5, sticky=(tk.W, tk.E))
        
        # 基础识别按钮
        recognition_btn = ttk.Button(button_frame, text="📷 基础识别", 
                                   command=self.open_recognition_gui,
                                   style='Action.TButton')
        recognition_btn.grid(row=0, column=1, padx=(10, 0), pady=5, sticky=(tk.W, tk.E))
        
        # 实时检测按钮
        realtime_btn = ttk.Button(button_frame, text="🎥 实时检测", 
                                command=self.start_realtime_detection,
                                style='Action.TButton')
        realtime_btn.grid(row=1, column=0, padx=(0, 10), pady=5, sticky=(tk.W, tk.E))
        
        # 模型管理按钮
        model_btn = ttk.Button(button_frame, text="🔧 模型管理", 
                             command=self.open_model_manager,
                             style='Action.TButton')
        model_btn.grid(row=1, column=1, padx=(10, 0), pady=5, sticky=(tk.W, tk.E))
        
        # 系统设置按钮
        settings_btn = ttk.Button(button_frame, text="⚙️ 系统设置", 
                                command=self.open_settings,
                                style='Action.TButton')
        settings_btn.grid(row=2, column=0, padx=(0, 10), pady=5, sticky=(tk.W, tk.E))
        
        # 帮助文档按钮
        help_btn = ttk.Button(button_frame, text="📖 帮助文档", 
                            command=self.show_help,
                            style='Action.TButton')
        help_btn.grid(row=2, column=1, padx=(10, 0), pady=5, sticky=(tk.W, tk.E))
        
    def create_status_bar(self, parent):
        """创建状态栏"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E))
        status_frame.columnconfigure(0, weight=1)
        
        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.grid(row=0, column=0, sticky=tk.W)
        
        # 版本信息
        version_label = ttk.Label(status_frame, text="v1.0.0")
        version_label.grid(row=0, column=1, sticky=tk.E)
        
    def open_training_gui(self):
        """打开训练界面"""
        try:
            self.status_var.set("启动训练界面...")
            self.root.update()
            
            training_gui = YOLOSTrainingGUI()
            training_gui.run()
            
            self.status_var.set("训练界面已启动")
        except Exception as e:
            messagebox.showerror("错误", f"启动训练界面失败: {e}")
            self.status_var.set("就绪")
    
    def open_recognition_gui(self):
        """打开基础识别界面"""
        try:
            self.status_var.set("启动识别界面...")
            self.root.update()
            
            recognition_gui = BasicPetRecognitionGUI()
            recognition_gui.run()
            
            self.status_var.set("识别界面已启动")
        except Exception as e:
            messagebox.showerror("错误", f"启动识别界面失败: {e}")
            self.status_var.set("就绪")
    
    def start_realtime_detection(self):
        """启动实时检测"""
        try:
            self.status_var.set("启动实时检测...")
            messagebox.showinfo("实时检测", "实时检测功能正在开发中...")
            self.status_var.set("就绪")
        except Exception as e:
            messagebox.showerror("错误", f"启动实时检测失败: {e}")
            self.status_var.set("就绪")
    
    def open_model_manager(self):
        """打开模型管理"""
        try:
            self.status_var.set("打开模型管理...")
            messagebox.showinfo("模型管理", "模型管理功能正在开发中...")
            self.status_var.set("就绪")
        except Exception as e:
            messagebox.showerror("错误", f"打开模型管理失败: {e}")
            self.status_var.set("就绪")
    
    def open_settings(self):
        """打开系统设置"""
        try:
            self.status_var.set("打开系统设置...")
            messagebox.showinfo("系统设置", "系统设置功能正在开发中...")
            self.status_var.set("就绪")
        except Exception as e:
            messagebox.showerror("错误", f"打开系统设置失败: {e}")
            self.status_var.set("就绪")
    
    def show_help(self):
        """显示帮助文档"""
        help_text = """
YOLOS 智能视频识别系统

核心功能:
• 训练界面: 数据收集、标注和模型训练
• 基础识别: 图像和视频的目标检测
• 实时检测: 摄像头实时目标检测
• 模型管理: 模型加载、切换和优化
• 系统设置: 参数配置和系统管理

使用说明:
1. 点击对应功能按钮启动相应模块
2. 遵循界面提示进行操作
3. 查看状态栏了解当前系统状态

技术支持:
如有问题请查看项目文档或联系开发团队
        """
        
        messagebox.showinfo("帮助文档", help_text)
    
    def run(self):
        """运行主界面"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("程序被用户中断")
        except Exception as e:
            print(f"程序运行错误: {e}")
        finally:
            try:
                self.root.destroy()
            except:
                pass

def main():
    """主函数"""
    try:
        app = YOLOSMainGUI()
        app.run()
    except Exception as e:
        print(f"启动失败: {e}")

if __name__ == "__main__":
    main()