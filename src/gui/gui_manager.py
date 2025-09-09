#!/usr/bin/env python3
"""
GUI版本管理器
管理所有GUI界面版本，提供统一的启动入口
"""

import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os
from pathlib import Path
import importlib.util
from typing import Dict, List, Optional, Any

class GUIManager:
    """GUI版本管理器"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("YOLOS GUI管理器")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # GUI版本配置
        self.gui_versions = {
            "main_gui_v1": {
                "name": "主界面 v1",
                "description": "整合视频捕捉、图像识别和训练功能的主界面",
                "file": "versions/main_gui_v1.py",
                "class": "YOLOSMainGUI",
                "features": ["视频捕捉", "基础识别", "训练控制"],
                "status": "可用"
            },
            "advanced_training_gui_v1": {
                "name": "高级训练界面 v1",
                "description": "支持图片/视频上传、摄像头输入、大模型自学习的完整训练界面",
                "file": "versions/advanced_training_gui_v1.py", 
                "class": "AdvancedTrainingGUI",
                "features": ["多模态输入", "大模型自学习", "批量处理", "数据管理"],
                "status": "可用"
            },
            "yolos_training_gui_v1": {
                "name": "核心训练界面 v1",
                "description": "专注于视频捕捉、图像识别和模型训练的核心功能",
                "file": "versions/yolos_training_gui_v1.py",
                "class": "YOLOSTrainingGUI", 
                "features": ["数据收集", "模型训练", "实时检测"],
                "status": "可用"
            },
            "simple_yolos_gui_v1": {
                "name": "简化界面 v1",
                "description": "轻量级图形界面，确保PC版本能够正常启动",
                "file": "versions/simple_yolos_gui_v1.py",
                "class": "SimpleYOLOSGUI",
                "features": ["实时预览", "基础检测", "参数调节"],
                "status": "可用"
            }
        }
        
        # 创建界面
        self.setup_style()
        self.create_interface()
        
    def setup_style(self):
        """设置界面样式"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # 自定义样式
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Subtitle.TLabel', font=('Arial', 10))
        style.configure('Feature.TLabel', font=('Arial', 9), foreground='blue')
        
    def create_interface(self):
        """创建主界面"""
        # 主容器
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title_label = ttk.Label(main_frame, text="YOLOS GUI版本管理器", style='Title.TLabel')
        title_label.pack(pady=(0, 10))
        
        subtitle_label = ttk.Label(main_frame, 
                                  text="选择要启动的GUI界面版本", 
                                  style='Subtitle.TLabel')
        subtitle_label.pack(pady=(0, 20))
        
        # GUI版本列表
        self.create_gui_list(main_frame)
        
        # 控制按钮
        self.create_control_buttons(main_frame)
        
        # 状态栏
        self.create_status_bar(main_frame)
        
    def create_gui_list(self, parent):
        """创建GUI版本列表"""
        list_frame = ttk.LabelFrame(parent, text="可用的GUI版本", padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # 创建Treeview
        columns = ('名称', '描述', '功能', '状态')
        self.gui_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=10)
        
        # 设置列标题和宽度
        self.gui_tree.heading('名称', text='名称')
        self.gui_tree.heading('描述', text='描述')
        self.gui_tree.heading('功能', text='主要功能')
        self.gui_tree.heading('状态', text='状态')
        
        self.gui_tree.column('名称', width=150)
        self.gui_tree.column('描述', width=300)
        self.gui_tree.column('功能', width=200)
        self.gui_tree.column('状态', width=80)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.gui_tree.yview)
        self.gui_tree.configure(yscrollcommand=scrollbar.set)
        
        # 布局
        self.gui_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 填充数据
        self.populate_gui_list()
        
        # 绑定双击事件
        self.gui_tree.bind('<Double-1>', self.on_gui_double_click)
        
    def populate_gui_list(self):
        """填充GUI版本列表"""
        for gui_id, gui_info in self.gui_versions.items():
            features_str = ", ".join(gui_info['features'][:3])  # 显示前3个功能
            if len(gui_info['features']) > 3:
                features_str += "..."
                
            self.gui_tree.insert('', 'end', iid=gui_id, values=(
                gui_info['name'],
                gui_info['description'][:50] + "..." if len(gui_info['description']) > 50 else gui_info['description'],
                features_str,
                gui_info['status']
            ))
            
    def create_control_buttons(self, parent):
        """创建控制按钮"""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 启动按钮
        ttk.Button(button_frame, text="🚀 启动选中的GUI", 
                  command=self.launch_selected_gui).pack(side=tk.LEFT, padx=(0, 10))
        
        # 查看详情按钮
        ttk.Button(button_frame, text="📋 查看详情", 
                  command=self.show_gui_details).pack(side=tk.LEFT, padx=(0, 10))
        
        # 刷新按钮
        ttk.Button(button_frame, text="🔄 刷新列表", 
                  command=self.refresh_gui_list).pack(side=tk.LEFT, padx=(0, 10))
        
        # 退出按钮
        ttk.Button(button_frame, text="❌ 退出", 
                  command=self.root.quit).pack(side=tk.RIGHT)
        
    def create_status_bar(self, parent):
        """创建状态栏"""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X)
        
        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT)
        
        # 版本信息
        version_label = ttk.Label(status_frame, text="GUI Manager v1.0.0")
        version_label.pack(side=tk.RIGHT)
        
    def on_gui_double_click(self, event):
        """双击启动GUI"""
        self.launch_selected_gui()
        
    def get_selected_gui(self) -> Optional[str]:
        """获取选中的GUI"""
        selection = self.gui_tree.selection()
        if selection:
            return selection[0]
        return None
        
    def launch_selected_gui(self):
        """启动选中的GUI"""
        gui_id = self.get_selected_gui()
        if not gui_id:
            messagebox.showwarning("警告", "请先选择要启动的GUI版本")
            return
            
        gui_info = self.gui_versions[gui_id]
        
        try:
            self.status_var.set(f"正在启动 {gui_info['name']}...")
            self.root.update()
            
            # 动态导入GUI模块
            gui_module = self.load_gui_module(gui_info['file'])
            if gui_module:
                # 获取GUI类
                gui_class = getattr(gui_module, gui_info['class'])
                
                # 创建并运行GUI实例
                gui_instance = gui_class()
                
                # 隐藏管理器窗口
                self.root.withdraw()
                
                # 运行GUI
                gui_instance.run()
                
                # GUI关闭后显示管理器窗口
                self.root.deiconify()
                
            self.status_var.set("就绪")
            
        except Exception as e:
            messagebox.showerror("错误", f"启动GUI失败: {e}")
            self.status_var.set("就绪")
            self.root.deiconify()  # 确保窗口可见
            
    def load_gui_module(self, file_path: str):
        """动态加载GUI模块"""
        try:
            # 构建完整路径
            gui_dir = Path(__file__).parent
            full_path = gui_dir / file_path
            
            if not full_path.exists():
                raise FileNotFoundError(f"GUI文件不存在: {full_path}")
                
            # 动态导入模块
            spec = importlib.util.spec_from_file_location("gui_module", full_path)
            module = importlib.util.module_from_spec(spec)
            
            # 添加路径到sys.path
            module_dir = str(full_path.parent)
            if module_dir not in sys.path:
                sys.path.insert(0, module_dir)
                
            spec.loader.exec_module(module)
            return module
            
        except Exception as e:
            print(f"加载GUI模块失败: {e}")
            return None
            
    def show_gui_details(self):
        """显示GUI详情"""
        gui_id = self.get_selected_gui()
        if not gui_id:
            messagebox.showwarning("警告", "请先选择一个GUI版本")
            return
            
        gui_info = self.gui_versions[gui_id]
        
        # 创建详情窗口
        detail_window = tk.Toplevel(self.root)
        detail_window.title(f"GUI详情 - {gui_info['name']}")
        detail_window.geometry("500x400")
        detail_window.resizable(False, False)
        
        # 详情内容
        detail_frame = ttk.Frame(detail_window, padding="20")
        detail_frame.pack(fill=tk.BOTH, expand=True)
        
        # 名称
        ttk.Label(detail_frame, text="名称:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        ttk.Label(detail_frame, text=gui_info['name']).pack(anchor=tk.W, pady=(0, 10))
        
        # 描述
        ttk.Label(detail_frame, text="描述:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        desc_text = tk.Text(detail_frame, height=4, wrap=tk.WORD)
        desc_text.insert('1.0', gui_info['description'])
        desc_text.config(state='disabled')
        desc_text.pack(fill=tk.X, pady=(0, 10))
        
        # 功能列表
        ttk.Label(detail_frame, text="主要功能:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        features_frame = ttk.Frame(detail_frame)
        features_frame.pack(fill=tk.X, pady=(0, 10))
        
        for i, feature in enumerate(gui_info['features']):
            ttk.Label(features_frame, text=f"• {feature}").pack(anchor=tk.W)
            
        # 文件信息
        ttk.Label(detail_frame, text="文件路径:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        ttk.Label(detail_frame, text=gui_info['file']).pack(anchor=tk.W, pady=(0, 10))
        
        # 类名
        ttk.Label(detail_frame, text="主类名:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        ttk.Label(detail_frame, text=gui_info['class']).pack(anchor=tk.W, pady=(0, 10))
        
        # 状态
        ttk.Label(detail_frame, text="状态:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        status_color = 'green' if gui_info['status'] == '可用' else 'red'
        status_label = ttk.Label(detail_frame, text=gui_info['status'])
        status_label.pack(anchor=tk.W, pady=(0, 20))
        
        # 按钮
        button_frame = ttk.Frame(detail_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="启动此GUI", 
                  command=lambda: [detail_window.destroy(), self.launch_gui_by_id(gui_id)]).pack(side=tk.LEFT)
        
        ttk.Button(button_frame, text="关闭", 
                  command=detail_window.destroy).pack(side=tk.RIGHT)
        
    def launch_gui_by_id(self, gui_id: str):
        """根据ID启动GUI"""
        # 选中指定的GUI
        self.gui_tree.selection_set(gui_id)
        self.launch_selected_gui()
        
    def refresh_gui_list(self):
        """刷新GUI列表"""
        # 清空现有项目
        for item in self.gui_tree.get_children():
            self.gui_tree.delete(item)
            
        # 重新扫描GUI文件
        self.scan_gui_versions()
        
        # 重新填充列表
        self.populate_gui_list()
        
        self.status_var.set("列表已刷新")
        
    def scan_gui_versions(self):
        """扫描GUI版本文件"""
        versions_dir = Path(__file__).parent / "versions"
        if not versions_dir.exists():
            return
            
        # 扫描Python文件
        for py_file in versions_dir.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
                
            gui_id = py_file.stem
            if gui_id not in self.gui_versions:
                # 添加新发现的GUI文件
                self.gui_versions[gui_id] = {
                    "name": gui_id.replace("_", " ").title(),
                    "description": f"自动发现的GUI文件: {py_file.name}",
                    "file": f"versions/{py_file.name}",
                    "class": "MainGUI",  # 默认类名
                    "features": ["未知功能"],
                    "status": "未测试"
                }
                
    def run(self):
        """运行GUI管理器"""
        try:
            # 居中显示窗口
            self.center_window()
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
                
    def center_window(self):
        """窗口居中显示"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

def main():
    """主函数"""
    try:
        manager = GUIManager()
        manager.run()
    except Exception as e:
        print(f"启动GUI管理器失败: {e}")

if __name__ == "__main__":
    main()