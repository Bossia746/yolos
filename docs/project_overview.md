# YOLOS 项目总览

## 🎯 项目简介

YOLOS是一个功能齐全的多平台AIoT视觉大模型项目，基于Python开发，支持YOLOv5、YOLOv8、YOLO-World等主流YOLO模型。项目专为AIoT场景设计，支持树莓派、ROS1/2、ESP32等多种开发板和平台。

## 🏗️ 项目架构

```
yolos/
├── 📁 src/                          # 核心源代码
│   ├── 📁 models/                   # YOLO模型实现
│   │   ├── base_model.py           # 模型基类
│   │   ├── yolo_factory.py         # 模型工厂
│   │   ├── yolov5_model.py         # YOLOv5实现
│   │   ├── yolov8_model.py         # YOLOv8实现
│   │   ├── yolo_world_model.py     # YOLO-World实现
│   │   └── model_converter.py      # 模型转换器
│   ├── 📁 detection/               # 检测模块
│   │   ├── image_detector.py       # 图像检测
│   │   ├── video_detector.py       # 视频检测
│   │   ├── realtime_detector.py    # 实时检测
│   │   ├── camera_detector.py      # 摄像头检测
│   │   └── cli.py                  # 命令行工具
│   ├── 📁 training/                # 训练模块
│   │   └── trainer.py              # YOLO训练器
│   ├── 📁 communication/           # 通信模块
│   │   └── mqtt_client.py          # MQTT客户端
│   └── 📁 utils/                   # 工具模块
│       └── config_manager.py       # 配置管理
├── 📁 ros_workspace/               # ROS工作空间
│   └── 📁 src/yolos_ros/          # ROS包
│       ├── package.xml             # ROS包配置
│       ├── CMakeLists.txt          # CMake配置
│       ├── 📁 msg/                 # ROS消息定义
│       └── 📁 scripts/             # ROS节点脚本
├── 📁 esp32/                       # ESP32代码
│   └── 📁 yolos_esp32_cam/        # ESP32-CAM项目
├── 📁 examples/                    # 示例代码
│   ├── basic_detection.py          # 基础检测示例
│   ├── realtime_detection.py       # 实时检测示例
│   └── mqtt_detection.py           # MQTT通信示例
├── 📁 web/                         # Web界面
│   └── app.py                      # Flask Web应用
├── 📁 scripts/                     # 安装脚本
│   ├── install_raspberry_pi.sh     # 树莓派安装
│   └── install_ros.sh              # ROS安装
├── 📁 configs/                     # 配置文件
│   └── default_config.yaml         # 默认配置
├── 📁 docs/                        # 文档
│   ├── API.md                      # API文档
│   └── DEPLOYMENT.md               # 部署指南
├── README.md                       # 项目说明
├── requirements.txt                # Python依赖
├── setup.py                        # 安装配置
├── LICENSE                         # 开源协议
└── .gitignore                      # Git忽略文件
```

## 🚀 核心功能

### 1. 多模型支持
- **YOLOv5**: 经典高效的目标检测模型
- **YOLOv8**: 最新的YOLO架构，性能更优
- **YOLO-World**: 开放词汇目标检测模型
- **模型转换**: 支持ONNX、TensorRT、OpenVINO等格式

### 2. 多平台检测
- **图像检测**: 单张图像和批量图像处理
- **视频检测**: 视频文件处理和结果保存
- **实时检测**: 摄像头实时检测和显示
- **摄像头检测**: 专为嵌入式设备优化

### 3. AIoT通信
- **MQTT**: 物联网消息传输协议支持
- **HTTP API**: RESTful API接口
- **WebSocket**: 实时双向通信
- **ROS集成**: 机器人操作系统支持

### 4. 多设备支持
- **PC/服务器**: Windows、Linux、macOS
- **树莓派**: ARM架构优化
- **Jetson**: NVIDIA边缘计算平台
- **ESP32**: 微控制器摄像头模块

### 5. 训练和优化
- **模型训练**: 自定义数据集训练
- **模型量化**: INT8量化加速推理
- **模型剪枝**: 减少模型参数
- **性能优化**: 多种加速方案

## 🛠️ 技术栈

### 核心技术
- **深度学习**: PyTorch, ONNX Runtime
- **计算机视觉**: OpenCV, PIL
- **Web框架**: Flask, WebSocket
- **通信协议**: MQTT, HTTP, ROS
- **配置管理**: YAML, JSON

### 硬件加速
- **GPU加速**: CUDA, cuDNN
- **边缘推理**: TensorRT, OpenVINO
- **移动端**: ONNX Mobile, TensorFlow Lite

### 开发工具
- **版本控制**: Git
- **包管理**: pip, conda
- **容器化**: Docker (可选)
- **CI/CD**: GitHub Actions (可选)

## 📦 安装和使用

### 快速开始

1. **克隆项目**
```bash
git clone https://github.com/your-repo/yolos.git
cd yolos
```

2. **安装依赖**
```bash
pip install -r requirements.txt
pip install -e .
```

3. **基础检测**
```bash
# 图像检测
python examples/basic_detection.py

# 实时检测
python examples/realtime_detection.py

# MQTT通信
python examples/mqtt_detection.py
```

### 命令行工具

```bash
# 图像检测
yolos-detect image photo.jpg --output result.jpg

# 视频检测
yolos-detect video input.mp4 --output output.mp4

# 实时摄像头检测
yolos-detect realtime --camera --camera-type usb

# 树莓派摄像头检测
yolos-detect realtime --camera --camera-type picamera
```

### Web界面

```bash
# 启动Web服务
python web/app.py

# 访问 http://localhost:5000
```

## 🔧 平台特定安装

### 树莓派安装
```bash
chmod +x scripts/install_raspberry_pi.sh
./scripts/install_raspberry_pi.sh
```

### ROS安装
```bash
chmod +x scripts/install_ros.sh
./scripts/install_ros.sh
```

### ESP32开发
1. 使用Arduino IDE打开 `esp32/yolos_esp32_cam/yolos_esp32_cam.ino`
2. 配置WiFi和MQTT参数
3. 上传到ESP32-CAM开发板

## 🎯 应用场景

### 智能监控
- 安防监控系统
- 人员车辆检测
- 异常行为识别
- 入侵检测报警

### 工业检测
- 产品质量检测
- 缺陷识别分析
- 自动化生产线
- 设备状态监控

### 机器人视觉
- 自主导航避障
- 物体识别抓取
- 环境感知理解
- 人机交互界面

### 智慧农业
- 作物生长监测
- 病虫害识别
- 自动化收获
- 环境参数监控

### 智能交通
- 车辆检测统计
- 交通流量分析
- 违章行为识别
- 智能信号控制

## 🔮 未来规划

### 短期目标 (1-3个月)
- [ ] 完善单元测试覆盖
- [ ] 添加更多YOLO模型支持
- [ ] 优化移动端性能
- [ ] 增加更多示例项目

### 中期目标 (3-6个月)
- [ ] 支持3D目标检测
- [ ] 集成语义分割功能
- [ ] 添加目标跟踪算法
- [ ] 开发图形化配置工具

### 长期目标 (6-12个月)
- [ ] 支持联邦学习训练
- [ ] 集成大语言模型
- [ ] 开发云端部署方案
- [ ] 构建生态系统平台

## 🤝 贡献指南

我们欢迎社区贡献！请查看以下方式参与项目：

1. **报告问题**: 在GitHub Issues中报告bug
2. **功能建议**: 提出新功能需求
3. **代码贡献**: 提交Pull Request
4. **文档改进**: 完善项目文档
5. **测试用例**: 添加测试覆盖

### 开发环境设置
```bash
# 克隆开发分支
git clone -b develop https://github.com/your-repo/yolos.git

# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
python -m pytest tests/

# 代码格式检查
flake8 src/
black src/
```

## 📄 开源协议

本项目采用MIT开源协议，详见 [LICENSE](LICENSE) 文件。

## 📞 联系我们

- **项目主页**: https://github.com/your-repo/yolos
- **文档网站**: https://yolos.readthedocs.io
- **问题反馈**: https://github.com/your-repo/yolos/issues
- **讨论社区**: https://github.com/your-repo/yolos/discussions

## 🙏 致谢

感谢以下开源项目和社区的支持：

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [YOLOv5](https://github.com/ultralytics/yolov5)
- [YOLO-World](https://github.com/AILab-CVC/YOLO-World)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)
- [ROS](https://www.ros.org/)

---

**YOLOS - 让AI视觉触手可及！** 🚀✨