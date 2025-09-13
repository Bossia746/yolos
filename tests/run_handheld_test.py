#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOS手持物体识别测试启动器
发挥项目完整能力的专业测试
"""

import cv2
import numpy as np
import time
import json
import os
import sys
from datetime import datetime
from pathlib import Path

def check_dependencies():
    """检查必要的依赖"""
    print("🔍 检查系统依赖...")
    
    required_packages = {
        'cv2': 'OpenCV',
        'numpy': 'NumPy', 
        'torch': 'PyTorch',
        'ultralytics': 'Ultralytics YOLO',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn'
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {name}: 已安装")
        except ImportError:
            print(f"❌ {name}: 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ 缺少依赖包: {', '.join(missing_packages)}")
        print("请运行以下命令安装:")
        print("pip install torch torchvision ultralytics matplotlib seaborn")
        return False
    
    return True

def check_camera():
    """检查摄像头"""
    print("\n📷 检查摄像头...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return False
    
    ret, frame = cap.read()
    if not ret:
        print("❌ 无法读取摄像头画面")
        cap.release()
        return False
    
    height, width = frame.shape[:2]
    print(f"✅ 摄像头正常: {width}x{height}")
    cap.release()
    return True

def check_project_structure():
    """检查项目结构"""
    print("\n📁 检查项目结构...")
    
    required_dirs = [
        'src/models',
        'src/detection', 
        'src/core',
        'src/utils',
        'models'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
            print(f"❌ 缺少目录: {dir_path}")
        else:
            print(f"✅ 目录存在: {dir_path}")
    
    if missing_dirs:
        print(f"\n⚠️ 项目结构不完整，缺少: {', '.join(missing_dirs)}")
        return False
    
    return True

def setup_python_path():
    """设置Python路径"""
    current_dir = Path.cwd()
    src_dir = current_dir / 'src'
    
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
        print(f"✅ 已添加到Python路径: {src_dir}")

def main():
    """主函数"""
    print("🎯 YOLOS手持静态物体识别专业测试启动器")
    print("="*60)
    
    # 系统检查
    if not check_dependencies():
        print("\n❌ 依赖检查失败，请安装必要的包后重试")
        return
    
    if not check_camera():
        print("\n❌ 摄像头检查失败，请检查摄像头连接")
        return
    
    if not check_project_structure():
        print("\n❌ 项目结构检查失败，请确保在正确的项目目录中运行")
        return
    
    # 设置环境
    setup_python_path()
    
    print("\n✅ 所有检查通过，启动完整测试...")
    print("="*60)
    
    try:
        # 导入并运行完整测试
        from handheld_object_recognition_test import main as run_test
        run_test()
        
    except ImportError as e:
        print(f"❌ 导入测试模块失败: {e}")
        print("请确保handheld_object_recognition_test.py文件存在")
    except Exception as e:
        print(f"❌ 测试运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()