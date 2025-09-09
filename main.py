#!/usr/bin/env python3
"""
YOLOS主启动器
支持PC版本直接启动，以及生成不同平台的部署版本
"""

import sys
import os
import argparse
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

def check_dependencies():
    """检查必要依赖"""
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'torch': 'torch',
        'tkinter': 'tkinter (系统自带)',
    }
    
    missing_packages = []
    
    for package, install_name in required_packages.items():
        try:
            if package == 'cv2':
                import cv2
            elif package == 'torch':
                import torch
            elif package == 'numpy':
                import numpy
            elif package == 'tkinter':
                import tkinter
        except ImportError:
            missing_packages.append(install_name)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n请运行以下命令安装:")
        print("pip install opencv-python numpy torch torchvision")
        return False
    
    print("✅ 所有依赖检查通过")
    return True

def launch_pc_version():
    """启动PC版本"""
    print("🚀 启动YOLOS PC版本...")
    
    try:
        from gui.simple_yolos_gui import SimpleYOLOSGUI
        
        # 创建并启动GUI
        app = SimpleYOLOSGUI()
        app.run()
        
    except ImportError as e:
        print(f"❌ 导入GUI模块失败: {e}")
        print("🔧 启动简化版本...")
        launch_minimal_version()
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return False
    
    return True

def launch_minimal_version():
    """启动最小化版本"""
    print("🔧 启动YOLOS最小化版本...")
    
    try:
        from core.minimal_yolos import MinimalYOLOS
        
        app = MinimalYOLOS()
        app.run()
        
    except Exception as e:
        print(f"❌ 最小化版本启动失败: {e}")
        print("📝 请检查项目完整性")

def generate_deployment():
    """生成部署版本"""
    print("📦 生成部署版本...")
    
    try:
        from deployment.deployment_generator import DeploymentGenerator
        
        generator = DeploymentGenerator()
        
        # 生成不同平台版本
        platforms = ['pc', 'esp32', 'k230', 'raspberry_pi']
        
        for platform in platforms:
            print(f"   生成{platform}版本...")
            success = generator.generate_platform_version(platform)
            if success:
                print(f"   ✅ {platform}版本生成成功")
            else:
                print(f"   ❌ {platform}版本生成失败")
                
    except ImportError:
        print("❌ 部署生成器未找到，创建基础版本...")
        create_basic_deployments()
    except Exception as e:
        print(f"❌ 部署生成失败: {e}")

def create_basic_deployments():
    """创建基础部署版本"""
    deployments_dir = project_root / 'deployments'
    deployments_dir.mkdir(exist_ok=True)
    
    # PC版本部署脚本
    pc_script = deployments_dir / 'deploy_pc.py'
    pc_script.write_text('''#!/usr/bin/env python3
"""PC版本部署脚本"""
import subprocess
import sys

def install_dependencies():
    """安装依赖"""
    packages = ['opencv-python', 'numpy', 'torch', 'torchvision']
    for pkg in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])

def main():
    print("🚀 部署YOLOS PC版本...")
    install_dependencies()
    print("✅ PC版本部署完成")

if __name__ == "__main__":
    main()
''')
    
    # ESP32版本部署脚本
    esp32_script = deployments_dir / 'deploy_esp32.py'
    esp32_script.write_text('''#!/usr/bin/env python3
"""ESP32版本部署脚本"""
import shutil
from pathlib import Path

def main():
    print("🔧 生成ESP32版本...")
    
    # 创建ESP32项目结构
    esp32_dir = Path('esp32_yolos')
    esp32_dir.mkdir(exist_ok=True)
    
    # 复制核心文件
    core_files = ['src/core/minimal_yolos.py', 'src/models/yolo_lite.py']
    
    for file_path in core_files:
        if Path(file_path).exists():
            shutil.copy2(file_path, esp32_dir)
    
    print("✅ ESP32版本生成完成")

if __name__ == "__main__":
    main()
''')
    
    print("✅ 基础部署脚本已创建")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOS启动器')
    parser.add_argument('--mode', choices=['pc', 'deploy', 'minimal'], 
                       default='pc', help='启动模式')
    parser.add_argument('--platform', choices=['pc', 'esp32', 'k230', 'raspberry_pi'],
                       help='目标平台')
    parser.add_argument('--check', action='store_true', help='检查依赖')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("🎯 YOLOS多平台目标检测系统")
    print("=" * 50)
    
    # 检查依赖
    if args.check or args.mode == 'pc':
        if not check_dependencies():
            if args.mode == 'pc':
                print("⚠️ 依赖检查失败，尝试启动最小化版本...")
                args.mode = 'minimal'
    
    # 根据模式启动
    if args.mode == 'pc':
        success = launch_pc_version()
        if not success:
            print("🔄 PC版本启动失败，尝试最小化版本...")
            launch_minimal_version()
            
    elif args.mode == 'minimal':
        launch_minimal_version()
        
    elif args.mode == 'deploy':
        generate_deployment()
    
    print("\n👋 感谢使用YOLOS系统！")

if __name__ == "__main__":
    main()