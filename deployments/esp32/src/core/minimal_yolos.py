#!/usr/bin/env python3
"""
YOLOS最小化核心模块
确保在任何环境下都能基本运行
"""

import sys
import os
import time
from pathlib import Path

class MinimalYOLOS:
    """最小化YOLOS系统"""
    
    def __init__(self):
        self.name = "YOLOS Minimal"
        self.version = "1.0.0"
        self.status = "ready"
        
    def check_environment(self):
        """检查运行环境"""
        print("🔍 检查运行环境...")
        
        env_info = {
            'python_version': sys.version,
            'platform': sys.platform,
            'working_directory': os.getcwd(),
            'project_root': str(Path(__file__).parent.parent.parent)
        }
        
        print(f"   Python版本: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        print(f"   操作系统: {sys.platform}")
        print(f"   工作目录: {os.getcwd()}")
        
        # 检查关键目录
        project_root = Path(__file__).parent.parent.parent
        required_dirs = ['src', 'config', 'models']
        
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            if dir_path.exists():
                print(f"   ✅ {dir_name}/ 目录存在")
            else:
                print(f"   ⚠️ {dir_name}/ 目录不存在，将创建")
                dir_path.mkdir(exist_ok=True)
        
        return env_info
        
    def check_dependencies(self):
        """检查依赖包"""
        print("\n📦 检查依赖包...")
        
        dependencies = {
            'opencv-python': 'cv2',
            'numpy': 'numpy', 
            'pillow': 'PIL',
            'tkinter': 'tkinter'
        }
        
        available = {}
        missing = []
        
        for package, import_name in dependencies.items():
            try:
                if import_name == 'cv2':
                    import cv2
                    available[package] = cv2.__version__
                elif import_name == 'numpy':
                    import numpy as np
                    available[package] = np.__version__
                elif import_name == 'PIL':
                    from PIL import Image
                    available[package] = Image.__version__
                elif import_name == 'tkinter':
                    import tkinter as tk
                    available[package] = tk.TkVersion
                    
                print(f"   ✅ {package}: {available[package]}")
                
            except ImportError:
                missing.append(package)
                print(f"   ❌ {package}: 未安装")
        
        if missing:
            print(f"\n⚠️ 缺少依赖包: {', '.join(missing)}")
            print("请运行以下命令安装:")
            print(f"pip install {' '.join(missing)}")
            
        return available, missing
        
    def create_basic_structure(self):
        """创建基础项目结构"""
        print("\n🏗️ 创建基础项目结构...")
        
        project_root = Path(__file__).parent.parent.parent
        
        # 基础目录结构
        directories = [
            'config',
            'models', 
            'data/images',
            'data/videos',
            'results/images',
            'results/videos',
            'logs',
            'deployments/pc',
            'deployments/esp32',
            'deployments/k230',
            'deployments/raspberry_pi'
        ]
        
        for dir_path in directories:
            full_path = project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"   📁 {dir_path}/")
            
        # 创建基础配置文件
        self.create_basic_config(project_root)
        
    def create_basic_config(self, project_root):
        """创建基础配置文件"""
        config_dir = project_root / 'config'
        
        # 基础配置
        basic_config = {
            "system": {
                "name": "YOLOS",
                "version": "1.0.0",
                "debug": True
            },
            "camera": {
                "default_index": 0,
                "width": 640,
                "height": 480,
                "fps": 30
            },
            "detection": {
                "confidence_threshold": 0.5,
                "nms_threshold": 0.4,
                "max_detections": 100
            },
            "model": {
                "default_model": "yolov8n.pt",
                "model_dir": "models/",
                "device": "auto"
            }
        }
        
        import json
        config_file = config_dir / 'basic_config.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(basic_config, f, indent=2, ensure_ascii=False)
            
        print(f"   📄 配置文件: {config_file}")
        
    def run_basic_demo(self):
        """运行基础演示"""
        print("\n🎯 运行基础演示...")
        
        try:
            # 尝试导入OpenCV
            import cv2
            print("   📹 OpenCV可用，启动摄像头演示...")
            self.camera_demo()
            
        except ImportError:
            print("   ⚠️ OpenCV不可用，运行文本演示...")
            self.text_demo()
            
    def camera_demo(self):
        """摄像头演示"""
        import cv2
        
        print("   正在启动摄像头...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("   ❌ 无法打开摄像头")
            return
            
        print("   ✅ 摄像头启动成功")
        print("   按 'q' 键退出演示")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                
                # 添加信息文本
                cv2.putText(frame, f"YOLOS Minimal Demo", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 计算FPS
                elapsed = time.time() - start_time
                if elapsed > 0:
                    fps = frame_count / elapsed
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 绘制示例检测框
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 2)
                cv2.putText(frame, "Demo Object", (w//4, h//4-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('YOLOS Minimal Demo', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n   演示被用户中断")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("   摄像头演示结束")
            
    def text_demo(self):
        """文本演示"""
        print("   🖥️ YOLOS文本模式演示")
        print("   " + "="*40)
        
        # 模拟检测过程
        for i in range(5):
            print(f"   正在处理帧 {i+1}/5...")
            time.sleep(0.5)
            
            # 模拟检测结果
            detections = [
                {"class": "person", "confidence": 0.85, "bbox": [100, 100, 200, 300]},
                {"class": "car", "confidence": 0.72, "bbox": [300, 150, 500, 400]}
            ]
            
            print(f"   检测到 {len(detections)} 个目标:")
            for det in detections:
                print(f"     - {det['class']}: {det['confidence']:.2f}")
                
        print("   文本演示完成")
        
    def show_system_info(self):
        """显示系统信息"""
        print(f"\n📊 {self.name} 系统信息")
        print("="*50)
        print(f"版本: {self.version}")
        print(f"状态: {self.status}")
        print(f"Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        print(f"平台: {sys.platform}")
        print(f"工作目录: {os.getcwd()}")
        
    def run_interactive_mode(self):
        """运行交互模式"""
        print(f"\n🎮 {self.name} 交互模式")
        print("="*50)
        
        while True:
            print("\n可用命令:")
            print("  1. 检查环境 (check)")
            print("  2. 运行演示 (demo)")
            print("  3. 系统信息 (info)")
            print("  4. 创建结构 (setup)")
            print("  5. 退出 (quit)")
            
            try:
                choice = input("\n请选择操作 (1-5): ").strip().lower()
                
                if choice in ['1', 'check']:
                    self.check_environment()
                    self.check_dependencies()
                elif choice in ['2', 'demo']:
                    self.run_basic_demo()
                elif choice in ['3', 'info']:
                    self.show_system_info()
                elif choice in ['4', 'setup']:
                    self.create_basic_structure()
                elif choice in ['5', 'quit', 'exit']:
                    print("👋 再见！")
                    break
                else:
                    print("❌ 无效选择，请重试")
                    
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 操作失败: {e}")
                
    def run(self):
        """运行最小化系统"""
        print(f"🚀 启动 {self.name} v{self.version}")
        print("="*50)
        
        # 检查环境
        env_info = self.check_environment()
        available, missing = self.check_dependencies()
        
        # 根据依赖情况选择运行模式
        if not missing:
            print("\n✅ 所有依赖都可用，启动完整模式")
            try:
                # 尝试启动GUI
                from gui.simple_yolos_gui import SimpleYOLOSGUI
                print("🖼️ 启动图形界面...")
                app = SimpleYOLOSGUI()
                app.run()
                return
            except Exception as e:
                print(f"⚠️ GUI启动失败: {e}")
                print("🔄 切换到交互模式...")
        else:
            print(f"\n⚠️ 缺少依赖: {', '.join(missing)}")
            print("🔄 启动最小化模式...")
            
        # 运行交互模式
        self.run_interactive_mode()

def main():
    """主函数"""
    minimal_yolos = MinimalYOLOS()
    minimal_yolos.run()

if __name__ == "__main__":
    main()