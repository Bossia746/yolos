# YOLOS用户指南

## 📖 目录

1. [快速开始](#快速开始)
2. [系统要求](#系统要求)
3. [安装部署](#安装部署)
4. [基础使用](#基础使用)
5. [训练模式](#训练模式)
6. [高级功能](#高级功能)
7. [配置说明](#配置说明)
8. [故障排除](#故障排除)
9. [常见问题](#常见问题)

## 🚀 快速开始

### 什么是YOLOS？

YOLOS是一个专注于**视频捕捉和图像识别**的智能系统，提供：

- 🎥 **实时视频处理**: 支持多种摄像头设备
- 🔍 **智能目标检测**: 颜色、运动、形状识别
- 🎯 **训练工具**: 完整的数据标注和模型训练流程
- 🔌 **API接口**: 标准化的外部集成接口

### 核心特性

- ✅ **专精定位**: 专注视频捕捉和图像识别核心功能
- ✅ **实时性能**: 30FPS稳定处理，低延迟响应
- ✅ **易于使用**: 图形界面，无需编程经验
- ✅ **可扩展**: 模块化设计，支持功能扩展

## 💻 系统要求

### 最低要求
- **操作系统**: Windows 10/11, macOS 10.14+, Ubuntu 18.04+
- **Python**: 3.8或更高版本
- **内存**: 4GB RAM
- **存储**: 2GB可用空间
- **摄像头**: USB摄像头或内置摄像头

### 推荐配置
- **操作系统**: Windows 11, macOS 12+, Ubuntu 20.04+
- **Python**: 3.9或3.10
- **内存**: 8GB RAM或更多
- **存储**: 10GB可用空间
- **GPU**: 支持CUDA的NVIDIA显卡 (可选)
- **摄像头**: 1080p USB摄像头

### 支持的硬件平台
- **桌面系统**: Windows, macOS, Linux
- **嵌入式**: Raspberry Pi 4, Jetson Nano
- **AIoT开发板**: ESP32, Arduino, STM32系列

## 📦 安装部署

### 方法1: 快速安装 (推荐)

```bash
# 1. 克隆项目
git clone https://github.com/your-repo/yolos.git
cd yolos

# 2. 运行安装脚本
python scripts/install.py

# 3. 启动系统
python basic_pet_recognition_gui.py
```

### 方法2: 手动安装

```bash
# 1. 创建虚拟环境
python -m venv yolos_env
source yolos_env/bin/activate  # Linux/Mac
# 或
yolos_env\Scripts\activate     # Windows

# 2. 安装依赖
pip install -r requirements.txt

# 3. 安装OpenCV
pip install opencv-python opencv-contrib-python

# 4. 验证安装
python -c "import cv2; print('OpenCV版本:', cv2.__version__)"
```

### 方法3: Docker部署

```bash
# 构建镜像
docker build -t yolos:latest .

# 运行容器
docker run -it --device=/dev/video0 -p 8080:8080 yolos:latest
```

## 🎮 基础使用

### 启动主界面

```bash
# 启动主GUI
python src/gui/main_gui.py
```

主界面提供三种模式：
1. **基础识别模式**: 实时视频识别
2. **训练模式**: 数据收集和模型训练
3. **API模式**: 外部接口服务

### 基础识别模式

#### 1. 启动摄像头
- 点击"启动摄像头"按钮
- 系统自动检测可用摄像头
- 选择合适的摄像头设备

#### 2. 选择检测方法
- **颜色检测**: 基于HSV颜色空间
- **运动检测**: 背景减除算法
- **形状检测**: 几何特征分析
- **级联检测**: Haar特征检测

#### 3. 调整参数
```
置信度阈值: 0.1 - 0.9 (默认0.5)
检测区域: 全屏 / 自定义区域
帧率设置: 15-30 FPS
分辨率: 640x480 / 1280x720 / 1920x1080
```

#### 4. 查看结果
- 实时视频显示检测结果
- 检测信息面板显示详细数据
- 日志窗口显示处理过程

### 快速识别示例

```python
# 使用Python API进行快速识别
from src.core.detector import YOLOSDetector
import cv2

# 初始化检测器
detector = YOLOSDetector()

# 读取图像
image = cv2.imread("test_image.jpg")

# 执行检测
results = detector.detect(image)

# 显示结果
for result in results:
    print(f"检测到: {result['type']} - 置信度: {result['confidence']:.3f}")
```

## 🎯 训练模式

### 数据收集

#### 1. 启动训练界面
```bash
python src/gui/yolos_training_gui.py
```

#### 2. 数据捕获
- 点击"开始捕获"收集训练图像
- 支持视频录制和图像序列
- 自动保存到`data/datasets/`目录

#### 3. 数据标注
- 使用鼠标拖拽标注目标区域
- 支持多类别标注
- 自动生成YOLO格式标注文件

### 模型训练

#### 1. 训练配置
```yaml
# config/training.yaml
model:
  architecture: "yolov5s"
  input_size: 640
  classes: ["cat", "dog", "bird", "person"]

training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.001
  device: "auto"  # auto, cpu, cuda

data:
  train_path: "data/datasets/train"
  val_path: "data/datasets/val"
  test_path: "data/datasets/test"
```

#### 2. 开始训练
- 配置训练参数
- 点击"开始训练"
- 监控训练进度和损失曲线

#### 3. 模型评估
- 自动生成评估报告
- 查看精度、召回率等指标
- 导出训练好的模型

### 训练最佳实践

1. **数据质量**:
   - 每类至少100张标注图像
   - 包含不同光照、角度、背景
   - 标注框准确包围目标

2. **数据增强**:
   - 自动应用旋转、缩放、翻转
   - 调整亮度、对比度、饱和度
   - 添加噪声和模糊效果

3. **训练策略**:
   - 使用预训练模型加速收敛
   - 逐步降低学习率
   - 早停机制防止过拟合

## ⚙️ 高级功能

### API接口使用

#### 1. 启动API服务
```bash
python src/api/external_api_system.py
```

#### 2. 基础API调用
```python
import requests

# 图像检测
response = requests.post("http://localhost:8080/api/detect", 
                        files={"image": open("test.jpg", "rb")})
result = response.json()
print(result)
```

#### 3. 实时检测
```python
import websocket
import json

def on_message(ws, message):
    result = json.loads(message)
    print(f"检测结果: {result}")

ws = websocket.WebSocketApp("ws://localhost:8080/ws/realtime")
ws.on_message = on_message
ws.run_forever()
```

### 性能优化

#### 1. GPU加速
```bash
# 安装CUDA支持
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 验证GPU可用性
python -c "import torch; print('CUDA可用:', torch.cuda.is_available())"
```

#### 2. 多线程处理
```python
# 配置多线程
config = {
    "video_threads": 2,      # 视频处理线程数
    "detection_threads": 4,  # 检测线程数
    "max_queue_size": 10     # 最大队列大小
}
```

#### 3. 内存优化
```python
# 内存管理配置
memory_config = {
    "max_cache_size": 1000,     # 最大缓存帧数
    "garbage_collect_interval": 100,  # GC间隔
    "low_memory_mode": False    # 低内存模式
}
```

## 🔧 配置说明

### 摄像头配置

```yaml
# config/camera.yaml
camera:
  default_id: 0
  resolution:
    width: 640
    height: 480
  fps: 30
  buffer_size: 1
  
  # 高级设置
  auto_exposure: true
  brightness: 0.5
  contrast: 0.5
  saturation: 0.5
```

### 检测配置

```yaml
# config/detection.yaml
detection:
  confidence_threshold: 0.5
  nms_threshold: 0.4
  max_detections: 100
  
  color_detection:
    enabled: true
    color_ranges:
      red: [[0, 50, 50], [10, 255, 255]]
      green: [[40, 50, 50], [80, 255, 255]]
      blue: [[100, 50, 50], [130, 255, 255]]
  
  motion_detection:
    enabled: true
    threshold: 25
    min_area: 500
```

### 日志配置

```yaml
# config/logging.yaml
logging:
  level: INFO
  console_output: true
  file_output: true
  max_file_size: 10MB
  backup_count: 5
  
  modules:
    video_capture: DEBUG
    detector: INFO
    training: INFO
    gui: WARNING
```

## 🔍 故障排除

### 常见问题及解决方案

#### 1. 摄像头无法启动

**问题**: 提示"无法打开摄像头"

**解决方案**:
```bash
# 检查摄像头设备
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"

# 检查权限 (Linux)
sudo usermod -a -G video $USER

# 检查驱动 (Windows)
# 设备管理器 -> 图像设备 -> 更新驱动程序
```

#### 2. 检测精度低

**问题**: 检测结果不准确或漏检

**解决方案**:
1. **调整置信度阈值**: 降低到0.3-0.4
2. **改善光照条件**: 确保充足均匀光照
3. **清洁摄像头镜头**: 去除灰尘和污渍
4. **重新训练模型**: 使用更多高质量标注数据

#### 3. 性能问题

**问题**: 帧率低或处理延迟高

**解决方案**:
```python
# 降低分辨率
config["camera"]["resolution"] = {"width": 320, "height": 240}

# 减少检测频率
config["detection"]["skip_frames"] = 2  # 每2帧检测一次

# 启用GPU加速
config["device"] = "cuda"
```

#### 4. 内存泄漏

**问题**: 长时间运行后内存占用过高

**解决方案**:
```python
# 启用内存监控
import psutil
import gc

def monitor_memory():
    memory_percent = psutil.virtual_memory().percent
    if memory_percent > 80:
        gc.collect()  # 强制垃圾回收
```

### 日志分析

#### 1. 查看系统日志
```bash
# 查看最新系统日志
tail -f logs/system/yolos_20250909.log

# 查看错误日志
grep "ERROR" logs/system/error_20250909.log
```

#### 2. 性能分析
```bash
# 分析性能日志
python scripts/analyze_performance.py logs/performance/
```

#### 3. 调试模式
```bash
# 启用详细调试
export YOLOS_DEBUG=1
python basic_pet_recognition_gui.py
```

## ❓ 常见问题

### Q1: 支持哪些图像格式？
**A**: 支持常见格式：JPG, PNG, BMP, TIFF, WebP

### Q2: 可以同时使用多个摄像头吗？
**A**: 是的，支持多摄像头同时处理，在配置文件中设置多个camera_id

### Q3: 训练需要多长时间？
**A**: 取决于数据量和硬件配置：
- CPU: 1000张图像约需2-4小时
- GPU: 1000张图像约需30-60分钟

### Q4: 如何提高检测速度？
**A**: 
1. 使用GPU加速
2. 降低输入分辨率
3. 减少检测类别数量
4. 使用轻量级模型

### Q5: 支持视频文件检测吗？
**A**: 是的，支持MP4, AVI, MOV等格式的视频文件检测

### Q6: 如何导出检测结果？
**A**: 
- JSON格式: 包含坐标、置信度等详细信息
- CSV格式: 适合数据分析
- 图像格式: 带标注的可视化结果

### Q7: 可以在没有显示器的服务器上运行吗？
**A**: 是的，支持无头模式运行：
```bash
python src/core/headless_detector.py --input video.mp4 --output results.json
```

### Q8: 如何集成到现有系统？
**A**: 
1. 使用REST API接口
2. 导入Python模块
3. 使用命令行工具
4. WebSocket实时通信

## 📞 技术支持

### 获取帮助

1. **文档**: 查看`docs/`目录下的详细文档
2. **示例**: 参考`examples/`目录下的代码示例
3. **日志**: 检查`logs/`目录下的日志文件
4. **社区**: 访问项目GitHub页面提交Issue

### 报告问题

提交Issue时请包含：
1. 系统信息 (操作系统、Python版本)
2. 错误信息和日志
3. 复现步骤
4. 预期行为和实际行为

### 贡献代码

欢迎贡献代码！请参考：
1. `docs/development_guide.md` - 开发指南
2. `docs/api_reference.md` - API参考
3. GitHub Pull Request流程

---

*用户指南版本: v1.0*  
*最后更新: 2025-09-09*  
*维护团队: YOLOS开发组*