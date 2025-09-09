# YOLOS 部署指南

## 概述

YOLOS支持多种平台的部署，包括PC、服务器、树莓派、Jetson设备和ESP32等嵌入式设备。

## 平台支持

### 1. PC/服务器部署

#### 系统要求
- Python 3.8+
- CUDA 11.0+ (GPU加速)
- 8GB+ RAM
- 10GB+ 存储空间

#### 安装步骤

```bash
# 克隆项目
git clone https://github.com/your-username/yolos.git
cd yolos

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 测试安装
python examples/basic_detection.py
```

#### GPU支持

```bash
# 安装CUDA版本的PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 验证GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. 树莓派部署

#### 支持的型号
- Raspberry Pi 4B (推荐)
- Raspberry Pi 5
- Raspberry Pi 3B+

#### 自动安装

```bash
# 运行安装脚本
chmod +x scripts/install_raspberry_pi.sh
./scripts/install_raspberry_pi.sh
```

#### 手动安装

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装依赖
sudo apt install -y python3-pip python3-venv libopencv-dev

# 安装树莓派摄像头支持
sudo apt install -y python3-picamera python3-picamera2

# 启用摄像头
sudo raspi-config nonint do_camera 0

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装轻量级依赖
pip install -r requirements_raspberry_pi.txt
```

#### 性能优化

```bash
# 增加GPU内存分配
sudo raspi-config
# Advanced Options -> Memory Split -> 128

# 启用硬件加速
echo 'gpu_mem=128' | sudo tee -a /boot/config.txt

# 重启
sudo reboot
```

### 3. NVIDIA Jetson部署

#### 支持的设备
- Jetson Nano
- Jetson Xavier NX
- Jetson AGX Xavier
- Jetson Orin

#### 安装JetPack

```bash
# 下载并安装JetPack SDK
# https://developer.nvidia.com/jetpack

# 安装PyTorch (Jetson版本)
wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.12.0-cp38-cp38-linux_aarch64.whl
pip install torch-1.12.0-cp38-cp38-linux_aarch64.whl

# 安装torchvision
sudo apt install -y libjpeg-dev zlib1g-dev
pip install torchvision
```

#### TensorRT优化

```bash
# 导出TensorRT模型
python -c "
from models.yolo_factory import YOLOFactory
model = YOLOFactory.create_model('yolov8', 'yolov8n.pt')
model.export('yolov8n.engine', format='tensorrt')
"
```

### 4. ESP32部署

#### 硬件要求
- ESP32-CAM 或 ESP32-S3
- 4MB+ Flash
- 外部天线 (推荐)

#### 开发环境

```bash
# 安装Arduino IDE
# 添加ESP32开发板支持
# 安装必要库: ArduinoJson, PubSubClient

# 或使用PlatformIO
pip install platformio
pio init --board esp32cam
```

#### 配置和烧录

```cpp
// 修改WiFi配置
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// 修改MQTT服务器
const char* mqtt_server = "192.168.1.100";
```

```bash
# 编译和上传
arduino-cli compile --fqbn esp32:esp32:esp32cam esp32/yolos_esp32_cam/
arduino-cli upload -p /dev/ttyUSB0 --fqbn esp32:esp32:esp32cam esp32/yolos_esp32_cam/
```

## ROS集成部署

### ROS1 (Noetic)

```bash
# 安装ROS1
./scripts/install_ros.sh

# 构建工作空间
cd ~/catkin_ws
catkin_make

# 启动检测节点
roslaunch yolos_ros detection.launch
```

### ROS2 (Humble)

```bash
# 安装ROS2
./scripts/install_ros.sh

# 构建工作空间
cd ~/ros2_ws
colcon build

# 启动检测节点
ros2 launch yolos_ros detection.launch.py
```

## Docker部署

### 创建Docker镜像

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . .

# 安装Python依赖
RUN pip install -r requirements.txt

# 暴露端口
EXPOSE 5000 8000

# 启动命令
CMD ["python", "web/app.py"]
```

### 构建和运行

```bash
# 构建镜像
docker build -t yolos:latest .

# 运行容器
docker run -d \
  --name yolos \
  -p 5000:5000 \
  -p 8000:8000 \
  --device /dev/video0:/dev/video0 \
  yolos:latest

# GPU支持
docker run --gpus all -d \
  --name yolos-gpu \
  -p 5000:5000 \
  yolos:latest
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  yolos:
    build: .
    ports:
      - "5000:5000"
      - "8000:8000"
    devices:
      - "/dev/video0:/dev/video0"
    volumes:
      - "./configs:/app/configs"
      - "./outputs:/app/outputs"
    environment:
      - CUDA_VISIBLE_DEVICES=0
    restart: unless-stopped

  mqtt:
    image: eclipse-mosquitto:latest
    ports:
      - "1883:1883"
      - "9001:9001"
    volumes:
      - "./mqtt.conf:/mosquitto/config/mosquitto.conf"
    restart: unless-stopped
```

```bash
# 启动服务
docker-compose up -d
```

## 云端部署

### AWS部署

```bash
# 使用EC2实例
# 选择Deep Learning AMI
# 配置安全组开放端口

# 安装YOLOS
git clone https://github.com/your-username/yolos.git
cd yolos
pip install -r requirements.txt

# 启动服务
python web/app.py
```

### Google Cloud部署

```bash
# 使用AI Platform
gcloud ai-platform models create yolos_model

# 部署模型
gcloud ai-platform versions create v1 \
  --model yolos_model \
  --origin gs://your-bucket/model/ \
  --runtime-version 2.8 \
  --framework tensorflow \
  --python-version 3.8
```

### Azure部署

```bash
# 使用Azure Container Instances
az container create \
  --resource-group myResourceGroup \
  --name yolos-container \
  --image yolos:latest \
  --ports 5000 8000 \
  --cpu 2 \
  --memory 4
```

## 性能调优

### 模型优化

```python
# 量化模型
from models.model_converter import ModelConverter

converter = ModelConverter()
converter.quantize_model('yolov8n.pt', 'yolov8n_int8.pt')

# 剪枝模型
converter.prune_model('yolov8n.pt', 'yolov8n_pruned.pt', sparsity=0.3)
```

### 推理优化

```python
# 批量推理
detector.predict_batch(images, batch_size=8)

# 异步推理
import asyncio

async def async_detect(image):
    return await detector.predict_async(image)

# 多线程推理
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(detector.predict, img) for img in images]
    results = [f.result() for f in futures]
```

### 系统优化

```bash
# 增加交换空间 (树莓派)
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=100/CONF_SWAPSIZE=1024/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# CPU频率调节
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# GPU频率调节 (Jetson)
sudo jetson_clocks
```

## 监控和维护

### 系统监控

```python
# 性能监控
from utils.metrics import MetricsCalculator

metrics = MetricsCalculator()
stats = metrics.get_system_stats()
print(f"CPU使用率: {stats['cpu_percent']}%")
print(f"内存使用率: {stats['memory_percent']}%")
print(f"GPU使用率: {stats['gpu_percent']}%")
```

### 日志管理

```python
# 配置日志
from utils.logger import setup_logger

logger = setup_logger('yolos', 'logs/yolos.log')
logger.info("系统启动")
```

### 自动更新

```bash
# 创建更新脚本
cat > update_yolos.sh << 'EOF'
#!/bin/bash
cd /path/to/yolos
git pull origin main
pip install -r requirements.txt
sudo systemctl restart yolos
EOF

# 设置定时任务
crontab -e
# 添加: 0 2 * * 0 /path/to/update_yolos.sh
```

## 故障排除

### 常见问题

1. **摄像头无法打开**
   ```bash
   # 检查摄像头权限
   ls -l /dev/video*
   sudo usermod -a -G video $USER
   ```

2. **CUDA内存不足**
   ```python
   # 减少批量大小
   detector.set_batch_size(4)
   
   # 清理GPU缓存
   torch.cuda.empty_cache()
   ```

3. **模型加载失败**
   ```python
   # 检查模型路径
   import os
   print(os.path.exists('model.pt'))
   
   # 重新下载模型
   model = YOLOFactory.create_model('yolov8', 'yolov8n.pt')
   ```

### 性能问题

1. **检测速度慢**
   - 降低输入分辨率
   - 使用更小的模型 (yolov8n)
   - 启用GPU加速
   - 减少检测频率

2. **内存占用高**
   - 减少批量大小
   - 使用模型量化
   - 定期清理缓存

3. **网络连接问题**
   - 检查防火墙设置
   - 验证MQTT服务器状态
   - 使用本地网络测试