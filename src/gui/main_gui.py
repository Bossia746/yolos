#!/usr/bin/env python3
"""
YOLOS主界面入口
基于BaseYOLOSGUI的功能选择界面
"""

import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os
from pathlib import Path
from typing import List, Dict

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.gui.base_gui import BaseYOLOSGUI

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

class YOLOSMainGUI(BaseYOLOSGUI):
    """YOLOS主界面"""
    
    def __init__(self):
        super().__init__(title="YOLOS - 智能视频识别系统", 
                        config_file="main_gui_config.json",
                        geometry="600x400")
        
        # 设置为功能选择模式
        self.is_function_selector = True
        
        # 重新创建界面
        self.create_main_interface()
    
    def setup_ui(self):
        """重写基类的UI设置方法"""
        # 功能选择模式不需要标准的摄像头界面
        pass
    
    def create_main_interface(self):
        """创建主功能选择界面"""
        # 清空现有内容
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # 设置样式
        self.setup_style()
        
        # 主框架
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题区域
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = ttk.Label(title_frame, text="YOLOS", style='Title.TLabel')
        title_label.pack()
        
        subtitle_label = ttk.Label(title_frame, text="智能视频识别与训练系统", style='Subtitle.TLabel')
        subtitle_label.pack()
        
        # 功能选择区域
        function_frame = ttk.LabelFrame(main_frame, text="选择功能模块", padding="15")
        function_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # 基础识别模块
        basic_frame = ttk.Frame(function_frame)
        basic_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(basic_frame, text="🎯 基础识别模式", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        ttk.Label(basic_frame, text="使用传统计算机视觉方法进行实时识别", 
                 foreground='gray').pack(anchor=tk.W, pady=(2, 5))
        
        basic_features = "• 颜色检测  • 运动分析  • 形状识别  • 实时处理"
        ttk.Label(basic_frame, text=basic_features, font=('Arial', 9)).pack(anchor=tk.W)
        
        ttk.Button(basic_frame, text="启动基础识别", style='Action.TButton',
                  command=self.start_basic_recognition).pack(anchor=tk.W, pady=(5, 0))
        
        # 分隔线
        ttk.Separator(function_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # 训练模块
        training_frame = ttk.Frame(function_frame)
        training_frame.pack(fill=tk.X, pady=(15, 0))
        
        ttk.Label(training_frame, text="🚀 训练模式", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        ttk.Label(training_frame, text="数据收集、标注和模型训练的完整工作流", 
                 foreground='gray').pack(anchor=tk.W, pady=(2, 5))
        
        training_features = "• 视频捕捉  • 数据标注  • 模型训练  • 实时检测"
        ttk.Label(training_frame, text=training_features, font=('Arial', 9)).pack(anchor=tk.W)
        
        ttk.Button(training_frame, text="启动训练界面", style='Action.TButton',
                  command=self.start_training_interface).pack(anchor=tk.W, pady=(5, 0))
        
        # 创建状态栏
        self.create_status_bar(self.root)
        
        # 菜单栏
        self.create_menu()
        
        # 居中显示
        self.center_window()
    
    def setup_style(self):
        """设置界面样式"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # 配置颜色
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Subtitle.TLabel', font=('Arial', 10))
        style.configure('Action.TButton', font=('Arial', 12), padding=10)
    
    def create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="打开项目", command=self.open_project)
        file_menu.add_command(label="最近项目", command=self.show_recent_projects)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.on_closing)
        
        # 工具菜单
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="工具", menu=tools_menu)
        tools_menu.add_command(label="系统检查", command=self.system_check)
        tools_menu.add_command(label="摄像头测试", command=self.camera_test)
        tools_menu.add_command(label="性能监控", command=self.performance_monitor)
        
        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="使用指南", command=self.show_user_guide)
        help_menu.add_command(label="API文档", command=self.show_api_docs)
        help_menu.add_command(label="关于YOLOS", command=self.show_about)
    
    def center_window(self):
        """窗口居中显示"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
    
    # 实现基类的抽象方法
    def load_model(self, model_path: str) -> bool:
        """加载模型（功能选择界面不需要）"""
        return True
    
    def perform_detection(self, frame):
        """执行检测（功能选择界面不需要）"""
        return frame
    
    def process_frame(self, frame):
        """处理帧（功能选择界面不需要）"""
        return frame
    
    def get_detection_results(self) -> List[Dict]:
        """获取检测结果（功能选择界面不需要）"""
        return []
    
    def on_model_changed(self, model_path: str):
        """模型变更回调（功能选择界面不需要）"""
        pass
    
    def update_status(self, message: str):
        """更新状态栏"""
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def start_basic_recognition(self):
        """启动基础识别模式"""
        self.update_status("启动基础识别模式...")
        
        try:
            # 隐藏主窗口
            self.root.withdraw()
            
            # 启动基础识别GUI
            basic_gui = BasicPetRecognitionGUI()
            basic_gui.run()
            
            # 恢复主窗口
            self.root.deiconify()
            self.update_status("基础识别模式已关闭")
            
        except Exception as e:
            self.root.deiconify()
            messagebox.showerror("错误", f"启动基础识别模式失败:\n{str(e)}")
            self.update_status("启动失败")
    
    def start_training_interface(self):
        """启动训练界面"""
        self.update_status("启动训练界面...")
        
        try:
            # 隐藏主窗口
            self.root.withdraw()
            
            # 启动训练GUI
            training_gui = YOLOSTrainingGUI()
            training_gui.run()
            
            # 恢复主窗口
            self.root.deiconify()
            self.update_status("训练界面已关闭")
            
        except Exception as e:
            self.root.deiconify()
            messagebox.showerror("错误", f"启动训练界面失败:\n{str(e)}")
            self.update_status("启动失败")
    
    def open_project(self):
        """打开项目"""
        from tkinter import filedialog
        
        project_dir = filedialog.askdirectory(title="选择YOLOS项目目录")
        if project_dir:
            self.update_status(f"项目已打开: {os.path.basename(project_dir)}")
            # 这里可以加载项目配置
    
    def show_recent_projects(self):
        """显示最近项目"""
        recent_window = tk.Toplevel(self.root)
        recent_window.title("最近项目")
        recent_window.geometry("400x300")
        recent_window.transient(self.root)
        recent_window.grab_set()
        
        ttk.Label(recent_window, text="最近打开的项目:", font=('Arial', 12, 'bold')).pack(pady=10)
        
        # 项目列表（示例）
        projects = [
            "宠物识别项目_20250909",
            "交通标志检测_20250908", 
            "人脸识别训练_20250907"
        ]
        
        listbox = tk.Listbox(recent_window, height=10)
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        for project in projects:
            listbox.insert(tk.END, project)
        
        button_frame = ttk.Frame(recent_window)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(button_frame, text="打开", command=recent_window.destroy).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="取消", command=recent_window.destroy).pack(side=tk.RIGHT)
    
    def system_check(self):
        """系统检查"""
        self.update_status("正在进行系统检查...")
        
        check_window = tk.Toplevel(self.root)
        check_window.title("系统检查")
        check_window.geometry("500x400")
        check_window.transient(self.root)
        
        text_widget = tk.Text(check_window, wrap=tk.WORD, padx=10, pady=10)
        scrollbar = ttk.Scrollbar(check_window, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 执行系统检查
        check_results = self.perform_system_check()
        text_widget.insert(tk.END, check_results)
        text_widget.config(state=tk.DISABLED)
        
        self.update_status("系统检查完成")
    
    def perform_system_check(self) -> str:
        """执行系统检查"""
        results = []
        results.append("YOLOS系统检查报告")
        results.append("=" * 30)
        results.append("")
        
        # 检查Python版本
        import sys
        results.append(f"✅ Python版本: {sys.version}")
        results.append("")
        
        # 检查关键依赖
        dependencies = [
            ("OpenCV", "cv2"),
            ("NumPy", "numpy"),
            ("Tkinter", "tkinter"),
            ("PIL", "PIL")
        ]
        
        results.append("依赖检查:")
        for name, module in dependencies:
            try:
                __import__(module)
                results.append(f"✅ {name}: 已安装")
            except ImportError:
                results.append(f"❌ {name}: 未安装")
        
        results.append("")
        
        # 检查摄像头
        results.append("摄像头检查:")
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                results.append("✅ 摄像头: 可用")
                cap.release()
            else:
                results.append("❌ 摄像头: 不可用")
        except:
            results.append("❌ 摄像头: 检查失败")
        
        results.append("")
        
        # 检查项目结构
        results.append("项目结构检查:")
        required_dirs = ["src", "config", "logs", "models"]
        for dir_name in required_dirs:
            if os.path.exists(dir_name):
                results.append(f"✅ {dir_name}/: 存在")
            else:
                results.append(f"⚠️ {dir_name}/: 不存在")
        
        return "\n".join(results)
    
    def camera_test(self):
        """摄像头测试"""
        self.update_status("正在测试摄像头...")
        
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                messagebox.showerror("错误", "无法打开摄像头")
                return
            
            messagebox.showinfo("摄像头测试", 
                              "摄像头测试窗口已打开\n按任意键关闭测试窗口")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                cv2.putText(frame, "Camera Test - Press any key to exit", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('YOLOS Camera Test', frame)
                
                if cv2.waitKey(1) & 0xFF != 255:  # 任意键退出
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            self.update_status("摄像头测试完成")
            
        except Exception as e:
            messagebox.showerror("错误", f"摄像头测试失败:\n{str(e)}")
            self.update_status("摄像头测试失败")
    
    def performance_monitor(self):
        """性能监控"""
        monitor_window = tk.Toplevel(self.root)
        monitor_window.title("性能监控")
        monitor_window.geometry("600x400")
        monitor_window.transient(self.root)
        
        # 创建监控界面
        notebook = ttk.Notebook(monitor_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # CPU监控
        cpu_frame = ttk.Frame(notebook)
        notebook.add(cpu_frame, text="CPU")
        
        # 内存监控
        memory_frame = ttk.Frame(notebook)
        notebook.add(memory_frame, text="内存")
        
        # GPU监控
        gpu_frame = ttk.Frame(notebook)
        notebook.add(gpu_frame, text="GPU")
        
        # 简单的性能信息显示
        try:
            import psutil
            
            # CPU信息
            cpu_info = f"CPU使用率: {psutil.cpu_percent()}%\n"
            cpu_info += f"CPU核心数: {psutil.cpu_count()}\n"
            ttk.Label(cpu_frame, text=cpu_info, justify=tk.LEFT).pack(pady=20)
            
            # 内存信息
            memory = psutil.virtual_memory()
            memory_info = f"内存使用率: {memory.percent}%\n"
            memory_info += f"总内存: {memory.total // (1024**3)} GB\n"
            memory_info += f"可用内存: {memory.available // (1024**3)} GB\n"
            ttk.Label(memory_frame, text=memory_info, justify=tk.LEFT).pack(pady=20)
            
        except ImportError:
            ttk.Label(cpu_frame, text="需要安装psutil库来显示性能信息").pack(pady=20)
        
        ttk.Label(gpu_frame, text="GPU监控功能开发中...").pack(pady=20)
    
    def show_user_guide(self):
        """显示用户指南"""
        guide_window = tk.Toplevel(self.root)
        guide_window.title("YOLOS用户指南")
        guide_window.geometry("700x500")
        guide_window.transient(self.root)
        
        text_widget = tk.Text(guide_window, wrap=tk.WORD, padx=15, pady=15)
        scrollbar = ttk.Scrollbar(guide_window, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        guide_content = """
YOLOS用户指南

1. 系统概述
YOLOS是一个专注于视频捕捉和图像识别的智能系统，提供两种主要工作模式：

2. 基础识别模式
• 使用传统计算机视觉方法
• 实时处理视频流
• 支持颜色检测、运动分析、形状识别
• 适合快速原型和测试

使用步骤：
1) 点击"启动基础识别"
2) 系统会自动检测并启动摄像头
3) 实时显示识别结果
4) 按'q'或ESC键退出

3. 训练模式
• 完整的机器学习工作流
• 支持数据收集、标注和模型训练
• 基于YOLO架构的深度学习

使用步骤：
1) 点击"启动训练界面"
2) 选择工作模式（捕获/标注/训练/检测）
3) 收集训练数据
4) 标注目标对象
5) 配置训练参数
6) 开始模型训练

4. 系统要求
• Python 3.8+
• OpenCV 4.0+
• 摄像头设备
• 足够的存储空间（用于训练数据）

5. 常见问题
Q: 摄像头无法启动？
A: 检查摄像头连接，确保没有其他程序占用

Q: 训练速度慢？
A: 考虑使用GPU加速，减少训练数据量或降低模型复杂度

Q: 识别准确率低？
A: 增加训练数据，改善数据质量，调整模型参数

6. 技术支持
如需技术支持，请查看API文档或联系开发团队。
        """
        
        text_widget.insert(tk.END, guide_content)
        text_widget.config(state=tk.DISABLED)
    
    def show_api_docs(self):
        """显示API文档"""
        messagebox.showinfo("API文档", 
                           "API文档功能开发中...\n"
                           "请查看项目目录下的docs/文件夹获取详细文档")
    
    def show_about(self):
        """显示关于信息"""
        about_window = tk.Toplevel(self.root)
        about_window.title("关于YOLOS")
        about_window.geometry("400x300")
        about_window.transient(self.root)
        about_window.grab_set()
        
        # 居中显示
        about_window.update_idletasks()
        x = (about_window.winfo_screenwidth() // 2) - (about_window.winfo_width() // 2)
        y = (about_window.winfo_screenheight() // 2) - (about_window.winfo_height() // 2)
        about_window.geometry(f"+{x}+{y}")
        
        main_frame = ttk.Frame(about_window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Logo区域
        ttk.Label(main_frame, text="🎯", font=('Arial', 48)).pack(pady=(0, 10))
        
        # 标题
        ttk.Label(main_frame, text="YOLOS", font=('Arial', 20, 'bold')).pack()
        ttk.Label(main_frame, text="智能视频识别系统", font=('Arial', 12)).pack(pady=(0, 20))
        
        # 版本信息
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=10)
        
        info_text = """版本: 1.0.0
开发团队: YOLOS项目组
专注于视频捕捉和图像识别的核心功能

核心特性:
• 实时视频处理
• 智能目标识别  
• 机器学习训练
• 模块化架构"""
        
        ttk.Label(info_frame, text=info_text, justify=tk.LEFT).pack()
        
        # 关闭按钮
        ttk.Button(main_frame, text="确定", command=about_window.destroy).pack(pady=(20, 0))
    
    def on_closing(self):
        """关闭程序"""
        if messagebox.askokcancel("退出", "确定要退出YOLOS系统吗？"):
            self.root.destroy()
    
    def run(self):
        """运行主界面"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

def main():
    """主函数"""
    print("🎯 YOLOS智能视频识别系统")
    print("专注于视频捕捉和图像识别的核心功能")
    print("=" * 50)
    
    try:
        app = YOLOSMainGUI()
        app.run()
    except Exception as e:
        print(f"❌ 系统启动失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()