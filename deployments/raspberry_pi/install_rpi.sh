#!/bin/bash
# YOLOS 树莓派版本安装脚本

echo "🚀 YOLOS 树莓派版本安装程序"
echo "=================================="

# 更新系统
echo "📦 更新系统包..."
sudo apt update

# 安装系统依赖
echo "📦 安装系统依赖..."
sudo apt install -y python3-pip python3-opencv python3-numpy

# 安装Python依赖
echo "📦 安装Python依赖..."
pip3 install -r requirements.txt

# 启用摄像头
echo "📷 配置摄像头..."
sudo raspi-config nonint do_camera 0

echo "✅ 安装完成！"
echo "运行 python3 main.py 启动程序"
