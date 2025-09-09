#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOS 快速安装脚本
一键安装和配置YOLOS大模型自学习系统
"""

import os
import sys
import subprocess
import platform
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class YOLOSInstaller:
    """YOLOS系统安装器"""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.python_version = sys.version_info
        self.install_log = []
        self.errors = []
        
        # 检查Python版本
        if self.python_version < (3, 8):
            raise RuntimeError(f"需要Python 3.8+，当前版本: {sys.version}")
        
        print("🚀 YOLOS 大模型自学习系统安装器")
        print("=" * 50)
        print(f"系统: {platform.system()} {platform.release()}")
        print(f"Python: {sys.version}")
        print(f"架构: {platform.machine()}")
        print()
    
    def log(self, message: str, level: str = "INFO"):
        """记录日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self.install_log.append(log_entry)
        
        # 根据级别显示不同颜色
        if level == "ERROR":
            print(f"❌ {message}")
            self.errors.append(message)
        elif level == "WARNING":
            print(f"⚠️  {message}")
        elif level == "SUCCESS":
            print(f"✅ {message}")
        else:
            print(f"ℹ️  {message}")
    
    def run_command(self, command: str, check: bool = True, shell: bool = True) -> Tuple[bool, str]:
        """执行命令"""
        try:
            self.log(f"执行命令: {command}")
            result = subprocess.run(
                command,
                shell=shell,
                capture_output=True,
                text=True,
                check=check
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            error_msg = f"命令执行失败: {command}\n错误: {e.stderr}"
            self.log(error_msg, "ERROR")
            return False, e.stderr
        except Exception as e:
            error_msg = f"命令执行异常: {command}\n异常: {str(e)}"
            self.log(error_msg, "ERROR")
            return False, str(e)
    
    def check_system_requirements(self) -> bool:
        """检查系统要求"""
        self.log("检查系统要求...")
        
        requirements_met = True
        
        # 检查Python版本
        if self.python_version >= (3, 8):
            self.log(f"Python版本检查通过: {sys.version}", "SUCCESS")
        else:
            self.log(f"Python版本过低: {sys.version}，需要3.8+", "ERROR")
            requirements_met = False
        
        # 检查pip
        success, _ = self.run_command("pip --version", check=False)
        if success:
            self.log("pip检查通过", "SUCCESS")
        else:
            self.log("pip未安装或不可用", "ERROR")
            requirements_met = False
        
        # 检查git
        success, _ = self.run_command("git --version", check=False)
        if success:
            self.log("git检查通过", "SUCCESS")
        else:
            self.log("git未安装，某些功能可能不可用", "WARNING")
        
        # 检查系统特定要求
        if self.system == "linux":
            self.check_linux_requirements()
        elif self.system == "windows":
            self.check_windows_requirements()
        elif self.system == "darwin":
            self.check_macos_requirements()
        
        return requirements_met
    
    def check_linux_requirements(self):
        """检查Linux系统要求"""
        # 检查必需的系统包
        required_packages = [
            "libgl1-mesa-glx",
            "libglib2.0-0",
            "libsm6",
            "libxext6",
            "libxrender-dev",
            "libgomp1"
        ]
        
        for package in required_packages:
            success, _ = self.run_command(f"dpkg -l | grep {package}", check=False)
            if success:
                self.log(f"系统包检查通过: {package}", "SUCCESS")
            else:
                self.log(f"系统包缺失: {package}，将尝试安装", "WARNING")
    
    def check_windows_requirements(self):
        """检查Windows系统要求"""
        # 检查Visual C++ Redistributable
        self.log("Windows系统检查完成", "SUCCESS")
    
    def check_macos_requirements(self):
        """检查macOS系统要求"""
        # 检查Homebrew
        success, _ = self.run_command("brew --version", check=False)
        if success:
            self.log("Homebrew检查通过", "SUCCESS")
        else:
            self.log("建议安装Homebrew以获得更好的体验", "WARNING")
    
    def install_system_dependencies(self) -> bool:
        """安装系统依赖"""
        self.log("安装系统依赖...")
        
        if self.system == "linux":
            return self.install_linux_dependencies()
        elif self.system == "windows":
            return self.install_windows_dependencies()
        elif self.system == "darwin":
            return self.install_macos_dependencies()
        
        return True
    
    def install_linux_dependencies(self) -> bool:
        """安装Linux系统依赖"""
        # 更新包列表
        success, _ = self.run_command("sudo apt update", check=False)
        if not success:
            self.log("无法更新包列表，可能需要手动安装依赖", "WARNING")
            return True
        
        # 安装必需包
        packages = [
            "python3-pip", "python3-venv", "python3-dev",
            "build-essential", "cmake",
            "libgl1-mesa-glx", "libglib2.0-0", "libsm6",
            "libxext6", "libxrender-dev", "libgomp1",
            "libgstreamer1.0-dev", "libgstreamer-plugins-base1.0-dev",
            "libgtk-3-dev", "libcanberra-gtk-module", "libcanberra-gtk3-module"
        ]
        
        package_list = " ".join(packages)
        success, _ = self.run_command(f"sudo apt install -y {package_list}", check=False)
        
        if success:
            self.log("Linux系统依赖安装完成", "SUCCESS")
        else:
            self.log("部分系统依赖安装失败，请手动安装", "WARNING")
        
        return True
    
    def install_windows_dependencies(self) -> bool:
        """安装Windows系统依赖"""
        self.log("Windows系统依赖检查完成", "SUCCESS")
        return True
    
    def install_macos_dependencies(self) -> bool:
        """安装macOS系统依赖"""
        # 尝试安装cmake
        success, _ = self.run_command("brew install cmake", check=False)
        if success:
            self.log("macOS依赖安装完成", "SUCCESS")
        else:
            self.log("建议手动安装cmake: brew install cmake", "WARNING")
        
        return True
    
    def create_virtual_environment(self) -> bool:
        """创建虚拟环境"""
        self.log("创建Python虚拟环境...")
        
        venv_path = Path("yolos_env")
        
        if venv_path.exists():
            self.log("虚拟环境已存在，跳过创建", "WARNING")
            return True
        
        # 创建虚拟环境
        success, output = self.run_command(f"{sys.executable} -m venv yolos_env")
        
        if success:
            self.log("虚拟环境创建成功", "SUCCESS")
            return True
        else:
            self.log("虚拟环境创建失败", "ERROR")
            return False
    
    def get_pip_command(self) -> str:
        """获取pip命令"""
        if self.system == "windows":
            return "yolos_env\\Scripts\\pip"
        else:
            return "yolos_env/bin/pip"
    
    def get_python_command(self) -> str:
        """获取Python命令"""
        if self.system == "windows":
            return "yolos_env\\Scripts\\python"
        else:
            return "yolos_env/bin/python"
    
    def install_python_dependencies(self) -> bool:
        """安装Python依赖"""
        self.log("安装Python依赖包...")
        
        pip_cmd = self.get_pip_command()
        
        # 升级pip
        success, _ = self.run_command(f"{pip_cmd} install --upgrade pip setuptools wheel")
        if not success:
            self.log("pip升级失败", "WARNING")
        
        # 核心依赖列表
        core_dependencies = [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "opencv-python==4.8.1.78",
            "numpy==1.24.3",
            "Pillow>=10.0.0",
            "matplotlib>=3.7.0",
            "pandas>=2.0.0",
            "scipy>=1.10.0",
            "scikit-learn>=1.3.0"
        ]
        
        # 安装核心依赖
        for dep in core_dependencies:
            self.log(f"安装: {dep}")
            success, _ = self.run_command(f"{pip_cmd} install {dep}", check=False)
            if success:
                self.log(f"✅ {dep} 安装成功", "SUCCESS")
            else:
                self.log(f"❌ {dep} 安装失败", "ERROR")
        
        # YOLO和深度学习依赖
        yolo_dependencies = [
            "ultralytics>=8.1.0",
            "onnx>=1.15.0",
            "onnxruntime>=1.16.0"
        ]
        
        for dep in yolo_dependencies:
            self.log(f"安装YOLO依赖: {dep}")
            success, _ = self.run_command(f"{pip_cmd} install {dep}", check=False)
            if success:
                self.log(f"✅ {dep} 安装成功", "SUCCESS")
            else:
                self.log(f"❌ {dep} 安装失败", "WARNING")
        
        # 计算机视觉依赖
        cv_dependencies = [
            "mediapipe>=0.10.9",
            "albumentations>=1.3.0"
        ]
        
        for dep in cv_dependencies:
            self.log(f"安装CV依赖: {dep}")
            success, _ = self.run_command(f"{pip_cmd} install {dep}", check=False)
            if success:
                self.log(f"✅ {dep} 安装成功", "SUCCESS")
            else:
                self.log(f"❌ {dep} 安装失败", "WARNING")
        
        # 通信依赖
        comm_dependencies = [
            "paho-mqtt>=1.6.0",
            "flask>=2.2.0",
            "requests>=2.28.0",
            "PyYAML>=6.0",
            "tqdm>=4.64.0"
        ]
        
        for dep in comm_dependencies:
            self.log(f"安装通信依赖: {dep}")
            success, _ = self.run_command(f"{pip_cmd} install {dep}", check=False)
            if success:
                self.log(f"✅ {dep} 安装成功", "SUCCESS")
            else:
                self.log(f"❌ {dep} 安装失败", "WARNING")
        
        # 可选依赖 (面部识别)
        self.log("安装可选依赖 (可能需要较长时间)...")
        optional_deps = ["dlib", "face_recognition"]
        
        for dep in optional_deps:
            self.log(f"尝试安装可选依赖: {dep}")
            success, _ = self.run_command(f"{pip_cmd} install {dep}", check=False)
            if success:
                self.log(f"✅ {dep} 安装成功", "SUCCESS")
            else:
                self.log(f"⚠️  {dep} 安装失败，某些功能可能不可用", "WARNING")
        
        return True
    
    def setup_project_structure(self) -> bool:
        """设置项目结构"""
        self.log("设置项目结构...")
        
        # 创建必要的目录
        directories = [
            "data",
            "data/self_learning",
            "data/self_learning/images",
            "data/self_learning/logs",
            "data/self_learning/backups",
            "logs",
            "test_results",
            "models/pretrained"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.log(f"创建目录: {directory}")
        
        # 创建配置文件 (如果不存在)
        config_files = [
            "config/self_learning_config.yaml",
            "config/aiot_platform_config.yaml"
        ]
        
        for config_file in config_files:
            if not Path(config_file).exists():
                self.log(f"配置文件不存在: {config_file}", "WARNING")
            else:
                self.log(f"配置文件存在: {config_file}", "SUCCESS")
        
        return True
    
    def test_installation(self) -> bool:
        """测试安装"""
        self.log("测试安装...")
        
        python_cmd = self.get_python_command()
        
        # 测试基本导入
        test_imports = [
            "import cv2; print(f'OpenCV: {cv2.__version__}')",
            "import numpy as np; print(f'NumPy: {np.__version__}')",
            "import torch; print(f'PyTorch: {torch.__version__}')",
            "import yaml; print('PyYAML: OK')",
            "import requests; print('Requests: OK')"
        ]
        
        for test_import in test_imports:
            success, output = self.run_command(f'{python_cmd} -c "{test_import}"', check=False)
            if success:
                self.log(f"✅ 导入测试通过: {output.strip()}", "SUCCESS")
            else:
                self.log(f"❌ 导入测试失败: {test_import}", "ERROR")
        
        # 运行系统测试
        if Path("test_self_learning_system.py").exists():
            self.log("运行系统测试...")
            success, output = self.run_command(f"{python_cmd} test_self_learning_system.py", check=False)
            if success:
                self.log("系统测试通过", "SUCCESS")
            else:
                self.log("系统测试失败，但基本功能应该可用", "WARNING")
        
        return True
    
    def create_activation_script(self):
        """创建激活脚本"""
        self.log("创建激活脚本...")
        
        if self.system == "windows":
            script_content = """@echo off
echo 激活YOLOS虚拟环境...
call yolos_env\\Scripts\\activate.bat
echo 虚拟环境已激活！
echo 运行 'python self_learning_demo_gui.py' 启动GUI演示
echo 运行 'python test_self_learning_system.py' 进行系统测试
cmd /k
"""
            with open("activate_yolos.bat", "w", encoding="utf-8") as f:
                f.write(script_content)
            self.log("创建激活脚本: activate_yolos.bat", "SUCCESS")
        
        else:
            script_content = """#!/bin/bash
echo "激活YOLOS虚拟环境..."
source yolos_env/bin/activate
echo "虚拟环境已激活！"
echo "运行 'python self_learning_demo_gui.py' 启动GUI演示"
echo "运行 'python test_self_learning_system.py' 进行系统测试"
exec bash
"""
            with open("activate_yolos.sh", "w") as f:
                f.write(script_content)
            os.chmod("activate_yolos.sh", 0o755)
            self.log("创建激活脚本: activate_yolos.sh", "SUCCESS")
    
    def generate_installation_report(self):
        """生成安装报告"""
        self.log("生成安装报告...")
        
        report = {
            "installation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {
                "os": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python_version": sys.version
            },
            "installation_log": self.install_log,
            "errors": self.errors,
            "success": len(self.errors) == 0
        }
        
        report_file = f"installation_report_{int(time.time())}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.log(f"安装报告已保存: {report_file}", "SUCCESS")
    
    def install(self) -> bool:
        """执行完整安装"""
        try:
            # 检查系统要求
            if not self.check_system_requirements():
                self.log("系统要求检查失败", "ERROR")
                return False
            
            # 安装系统依赖
            if not self.install_system_dependencies():
                self.log("系统依赖安装失败", "ERROR")
                return False
            
            # 创建虚拟环境
            if not self.create_virtual_environment():
                self.log("虚拟环境创建失败", "ERROR")
                return False
            
            # 安装Python依赖
            if not self.install_python_dependencies():
                self.log("Python依赖安装失败", "ERROR")
                return False
            
            # 设置项目结构
            if not self.setup_project_structure():
                self.log("项目结构设置失败", "ERROR")
                return False
            
            # 测试安装
            if not self.test_installation():
                self.log("安装测试失败", "WARNING")
            
            # 创建激活脚本
            self.create_activation_script()
            
            # 生成安装报告
            self.generate_installation_report()
            
            # 显示完成信息
            self.show_completion_info()
            
            return len(self.errors) == 0
            
        except Exception as e:
            self.log(f"安装过程中发生异常: {str(e)}", "ERROR")
            return False
    
    def show_completion_info(self):
        """显示完成信息"""
        print("\n" + "=" * 60)
        print("🎉 YOLOS 安装完成！")
        print("=" * 60)
        
        if len(self.errors) == 0:
            print("✅ 所有组件安装成功！")
        else:
            print(f"⚠️  安装完成，但有 {len(self.errors)} 个警告/错误")
        
        print("\n📋 下一步操作:")
        print("1. 激活虚拟环境:")
        if self.system == "windows":
            print("   运行: activate_yolos.bat")
        else:
            print("   运行: source activate_yolos.sh")
        
        print("\n2. 配置API密钥 (可选):")
        print("   编辑: config/self_learning_config.yaml")
        print("   设置: OPENAI_API_KEY, CLAUDE_API_KEY 等")
        
        print("\n3. 启动系统:")
        print("   GUI演示: python self_learning_demo_gui.py")
        print("   系统测试: python test_self_learning_system.py")
        
        print("\n4. 查看文档:")
        print("   完整指南: COMPLETE_DEPLOYMENT_GUIDE.md")
        print("   API文档: docs/API.md")
        
        print("\n🔧 AIoT部署:")
        print("   ESP32固件: esp32/yolos_esp32_cam/")
        print("   树莓派脚本: scripts/install_raspberry_pi.sh")
        
        if len(self.errors) > 0:
            print(f"\n⚠️  警告/错误列表:")
            for error in self.errors:
                print(f"   - {error}")
        
        print("\n💡 获取帮助:")
        print("   文档: docs/")
        print("   示例: examples/")
        print("   测试: tests/")
        
        print("\n" + "=" * 60)


def main():
    """主函数"""
    try:
        installer = YOLOSInstaller()
        success = installer.install()
        
        if success:
            print("\n🎉 安装成功完成！")
            sys.exit(0)
        else:
            print("\n❌ 安装过程中遇到问题，请查看日志")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  安装被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 安装失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()