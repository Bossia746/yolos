#!/bin/bash

# YOLOS 树莓派安装脚本

echo "开始安装YOLOS到树莓派..."

# 检查系统
if ! grep -q "Raspberry Pi" /proc/cpuinfo; then
    echo "警告: 这不是树莓派系统"
fi

# 更新系统
echo "更新系统包..."
sudo apt update
sudo apt upgrade -y

# 安装系统依赖
echo "安装系统依赖..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    cmake \
    pkg-config \
    libjpeg-dev \
    libtiff5-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libfontconfig1-dev \
    libcairo2-dev \
    libgdk-pixbuf2.0-dev \
    libpango1.0-dev \
    libgtk2.0-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-103 \
    libqt5gui5 \
    libqt5webkit5 \
    libqt5test5 \
    python3-pyqt5 \
    git \
    wget \
    curl

# 安装树莓派摄像头支持
echo "安装树莓派摄像头支持..."
sudo apt install -y \
    python3-picamera \
    python3-picamera2

# 启用摄像头
echo "启用摄像头接口..."
sudo raspi-config nonint do_camera 0

# 启用I2C和SPI
echo "启用I2C和SPI..."
sudo raspi-config nonint do_i2c 0
sudo raspi-config nonint do_spi 0

# 创建虚拟环境
echo "创建Python虚拟环境..."
python3 -m venv venv
source venv/bin/activate

# 升级pip
pip install --upgrade pip

# 安装PyTorch (树莓派版本)
echo "安装PyTorch..."
if [[ $(uname -m) == "aarch64" ]]; then
    # 64位系统
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    # 32位系统
    pip install https://download.pytorch.org/whl/cpu/torch-1.13.1%2Bcpu-cp39-cp39-linux_armv7l.whl
    pip install https://download.pytorch.org/whl/cpu/torchvision-0.14.1%2Bcpu-cp39-cp39-linux_armv7l.whl
fi

# 安装OpenCV
echo "安装OpenCV..."
pip install opencv-python-headless

# 安装其他依赖
echo "安装项目依赖..."
pip install -r requirements.txt

# 安装树莓派特定依赖
echo "安装树莓派特定依赖..."
pip install RPi.GPIO gpiozero picamera

# 创建系统服务
echo "创建系统服务..."
sudo tee /etc/systemd/system/yolos.service > /dev/null <<EOF
[Unit]
Description=YOLOS Detection Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/yolos
Environment=PATH=/home/pi/yolos/venv/bin
ExecStart=/home/pi/yolos/venv/bin/python src/detection/realtime_detector.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 重新加载systemd
sudo systemctl daemon-reload

# 创建配置文件
echo "创建树莓派配置文件..."
mkdir -p configs
cp configs/default_config.yaml configs/raspberry_pi_config.yaml

# 修改树莓派特定配置
cat >> configs/raspberry_pi_config.yaml << EOF

# 树莓派特定配置
camera:
  type: "picamera"
  resolution: [640, 480]
  framerate: 15

performance:
  enable_gpu: false
  batch_processing: false
  async_processing: false

raspberry_pi:
  gpio_pins:
    led_status: 18
    button_trigger: 24
  camera_module: "v2"
  enable_hardware_pwm: true
EOF

# 设置权限
echo "设置权限..."
sudo usermod -a -G video pi
sudo usermod -a -G gpio pi

# 创建启动脚本
echo "创建启动脚本..."
cat > start_yolos.sh << 'EOF'
#!/bin/bash
cd /home/pi/yolos
source venv/bin/activate
python src/detection/camera_detector.py --config configs/raspberry_pi_config.yaml
EOF

chmod +x start_yolos.sh

# 创建测试脚本
echo "创建测试脚本..."
cat > test_camera.py << 'EOF'
#!/usr/bin/env python3
import cv2
import sys
import os
sys.path.append('src')

from detection.camera_detector import CameraDetector

def test_camera():
    try:
        print("测试摄像头...")
        detector = CameraDetector(camera_type='picamera')
        detector.set_camera_params(resolution=(320, 240), framerate=15)
        
        print("开始检测，按Ctrl+C退出")
        detector.start_detection(display=True)
        
    except KeyboardInterrupt:
        print("测试结束")
    except Exception as e:
        print(f"测试失败: {e}")

if __name__ == "__main__":
    test_camera()
EOF

chmod +x test_camera.py

echo "安装完成!"
echo ""
echo "使用说明:"
echo "1. 重启系统: sudo reboot"
echo "2. 测试摄像头: python test_camera.py"
echo "3. 启动服务: sudo systemctl start yolos"
echo "4. 开机自启: sudo systemctl enable yolos"
echo "5. 查看日志: sudo journalctl -u yolos -f"
echo ""
echo "配置文件: configs/raspberry_pi_config.yaml"
echo "启动脚本: ./start_yolos.sh"