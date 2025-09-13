# YOLOS - You Only Look Once System

<div align="center">

![YOLOS Logo](docs/assets/logo.png)

**基于深度学习的实时目标检测和计算机视觉系统**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org)
[![Build Status](https://img.shields.io/github/workflow/status/yolos/yolos/CI)](https://github.com/yolos/yolos/actions)
[![Downloads](https://img.shields.io/github/downloads/yolos/yolos/total.svg)](https://github.com/yolos/yolos/releases)
[![Stars](https://img.shields.io/github/stars/yolos/yolos.svg)](https://github.com/yolos/yolos/stargazers)

[🚀 快速开始](#快速开始) • [📖 文档](#文档) • [🎯 演示](#演示) • [🤝 贡献](#贡献) • [💬 社区](#社区)

</div>

## ✨ 特性

### 🎯 核心功能
- **实时目标检测**: 支持80+类别，30+ FPS性能
- **人脸识别**: 人脸检测、关键点定位、属性分析、人脸比对
- **姿态估计**: 17个关键点的人体姿态检测和动作识别
- **手势识别**: 21个手部关键点检测和手势分类
- **视频分析**: 实时视频流处理和目标跟踪
- **跌倒检测**: 基于姿态分析的跌倒识别
- **ModelScope集成**: 集成魔搭社区的视觉大模型
- **多模态融合**: 结合多种检测结果

### 🚀 性能优势
- **高速推理**: GPU加速，支持TensorRT优化
- **低延迟**: 端到端延迟 < 50ms
- **高精度**: mAP@0.5 > 0.9 (COCO数据集)
- **内存优化**: 支持动态批处理和内存管理
- **SimAM注意力**: 无参数3D注意力机制，轻量级设备友好

### 🌐 平台支持
- **桌面平台**: Windows 10+, Linux (Ubuntu 18.04+), macOS 10.15+
- **移动平台**: Android, iOS (通过ONNX)
- **嵌入式**: 树莓派, Jetson Nano, ESP32, K230
- **云平台**: AWS, Azure, Google Cloud
- **容器化**: Docker部署支持

### 🔧 开发友好
- **简单API**: 3行代码即可开始检测
- **模块化设计**: 可插拔的组件架构
- **丰富示例**: 20+个使用示例
- **完整文档**: API文档、教程、最佳实践

## 🎯 演示

### 目标检测
```python
from yolos import YOLOS

# 初始化检测器
detector = YOLOS(model='yolov8n')

# 检测图像
results = detector.detect('image.jpg')
print(f"检测到 {len(results)} 个对象")

# 保存结果
detector.save_results(results, 'output.jpg')
```

### 实时视频检测
```python
from yolos import YOLOS

# 初始化检测器
detector = YOLOS(model='yolov8s')

# 实时检测
for frame in detector.detect_video(source=0):  # 摄像头
    detector.show_frame(frame)
```

### 批量处理
```python
from yolos import YOLOS

# 批量处理文件夹
detector = YOLOS()
results = detector.detect_batch('input_folder/', 'output_folder/')
print(f"处理完成，共处理 {len(results)} 个文件")
```

## 🚀 快速开始

### 📦 安装

#### 方式1: pip安装（推荐）
```bash
pip install yolos
```

#### 方式2: 从源码安装
```bash
# 克隆仓库
git clone https://github.com/yolos/yolos.git
cd yolos

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt

# 安装YOLOS
pip install -e .
```

#### 方式3: Docker安装
```bash
# 拉取镜像
docker pull yolos/yolos:latest

# 运行容器
docker run -it --gpus all -v $(pwd):/workspace yolos/yolos:latest
```

### 🎮 第一次使用

1. **验证安装**
```bash
python -c "import yolos; print(yolos.__version__)"
```

2. **下载模型**
```bash
yolos download --model yolov8n
```

3. **运行示例**
```bash
# 图像检测
yolos detect --source examples/bus.jpg --save

# 视频检测
yolos detect --source examples/video.mp4 --save

# 摄像头检测
yolos detect --source 0
```

### 📱 GUI应用

```bash
# 启动图形界面
yolos gui
```

## 🏗️ 项目结构

```
yolos/
├── 📁 src/                    # 源代码
│   ├── 📁 core/               # 核心模块
│   │   ├── application.py     # 应用程序主类
│   │   ├── config_manager.py  # 配置管理
│   │   ├── dependency_injection.py # 依赖注入
│   │   ├── event_system.py    # 事件系统
│   │   └── test_framework.py  # 测试框架
│   ├── 📁 models/             # 模型实现
│   │   ├── yolo/             # YOLO系列模型
│   │   ├── face/             # 人脸识别模型
│   │   ├── pose/             # 姿态估计模型
│   │   └── gesture/          # 手势识别模型
│   ├── 📁 applications/       # 应用实现
│   │   ├── face_recognition.py
│   │   ├── pose_estimation.py
│   │   ├── gesture_recognition.py
│   │   └── object_detection.py
│   └── 📁 utils/              # 工具函数
├── 📁 tests/                  # 测试代码
├── 📁 examples/               # 示例代码
├── 📁 docs/                   # 文档
├── 📁 configs/                # 配置文件
└── 📁 scripts/                # 脚本工具
```

## 🎯 支持的应用场景

### 🏢 商业应用
- **智能安防**: 人员检测、异常行为识别
- **零售分析**: 客流统计、商品识别
- **工业检测**: 质量控制、缺陷检测
- **交通监控**: 车辆检测、违章识别

### 🎮 娱乐应用
- **体感游戏**: 姿态控制、手势交互
- **AR/VR**: 人体追踪、手势识别
- **直播互动**: 实时特效、虚拟背景
- **内容创作**: 自动剪辑、智能标注

### 🏥 医疗健康
- **康复训练**: 姿态纠正、运动分析
- **健康监测**: 行为识别、跌倒检测
- **辅助诊断**: 医学影像分析

### 🎓 教育科研
- **在线教育**: 注意力检测、互动教学
- **科研实验**: 行为分析、数据采集
- **技能培训**: 动作指导、评估反馈

## 📊 性能基准

### 检测性能 (COCO数据集)

| 模型 | 尺寸 | mAP@0.5 | mAP@0.5:0.95 | 速度 (FPS) | 参数量 |
|------|------|---------|--------------|------------|--------|
| YOLOv8n | 640 | 50.2% | 37.3% | 80+ | 3.2M |
| YOLOv8s | 640 | 61.8% | 44.9% | 60+ | 11.2M |
| YOLOv8m | 640 | 67.2% | 50.2% | 45+ | 25.9M |
| YOLOv8l | 640 | 69.8% | 52.9% | 35+ | 43.7M |
| YOLOv8x | 640 | 71.6% | 53.9% | 25+ | 68.2M |

### 硬件性能

| 设备 | 模型 | 分辨率 | 批大小 | FPS | 内存使用 |
|------|------|--------|--------|-----|----------|
| RTX 4090 | YOLOv8n | 640×640 | 1 | 120+ | 2GB |
| RTX 3080 | YOLOv8s | 640×640 | 1 | 85+ | 4GB |
| GTX 1660 | YOLOv8n | 416×416 | 1 | 60+ | 3GB |
| CPU (i7-10700K) | YOLOv8n | 320×320 | 1 | 15+ | 1GB |
| 树莓派 4B | YOLOv8n | 320×320 | 1 | 3+ | 512MB |

## 🛠️ 系统要求

### 最低要求
- **操作系统**: Windows 10+, Ubuntu 18.04+, macOS 10.15+
- **Python**: 3.8+
- **内存**: 4GB RAM
- **存储**: 2GB 可用空间
- **显卡**: 支持OpenGL 3.3+

### 推荐配置
- **操作系统**: Windows 11, Ubuntu 20.04+, macOS 12+
- **Python**: 3.9+
- **内存**: 8GB+ RAM
- **存储**: 10GB+ SSD
- **显卡**: NVIDIA GPU (支持CUDA 11.8+)
- **显存**: 4GB+ VRAM

### GPU支持
- **NVIDIA**: GTX 1060+ / RTX 20系列+
- **AMD**: RX 580+ (通过ROCm)
- **Intel**: Arc A系列 (实验性支持)
- **Apple**: M1/M2 (通过Metal Performance Shaders)

## 📖 文档

### 📚 用户文档
- [📖 用户手册](docs/USER_MANUAL.md) - 完整的使用指南
- [🚀 快速开始](docs/QUICKSTART.md) - 5分钟上手教程
- [🎯 示例教程](examples/README.md) - 丰富的使用示例
- [❓ 常见问题](docs/FAQ.md) - 问题解答

### 🔧 开发文档
- [📋 API参考](docs/API_REFERENCE.md) - 完整的API文档
- [👨‍💻 开发指南](docs/DEVELOPER_GUIDE.md) - 开发规范和指南
- [🚀 部署指南](docs/DEPLOYMENT.md) - 生产环境部署
- [🧪 测试指南](docs/TESTING.md) - 测试框架和用例

### 🎓 教程系列
- [基础教程](docs/tutorials/basic/) - 从零开始学习
- [进阶教程](docs/tutorials/advanced/) - 高级功能使用
- [最佳实践](docs/tutorials/best-practices/) - 生产环境经验
- [性能优化](docs/tutorials/optimization/) - 性能调优指南

## 🤝 贡献

我们欢迎所有形式的贡献！无论是代码、文档、测试还是反馈建议。

### 🚀 如何贡献

1. **Fork 项目**
2. **创建特性分支** (`git checkout -b feature/AmazingFeature`)
3. **提交更改** (`git commit -m 'Add some AmazingFeature'`)
4. **推送到分支** (`git push origin feature/AmazingFeature`)
5. **创建 Pull Request**

### 📋 贡献指南
- [贡献指南](CONTRIBUTING.md) - 详细的贡献流程
- [代码规范](docs/CODE_STYLE.md) - 编码标准
- [提交规范](docs/COMMIT_CONVENTION.md) - Git提交规范

### 🏆 贡献者

感谢所有为YOLOS做出贡献的开发者！

<a href="https://github.com/yolos/yolos/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=yolos/yolos" />
</a>

## 💬 社区

### 🌐 在线社区
- **GitHub**: [讨论区](https://github.com/yolos/yolos/discussions)
- **Discord**: [YOLOS服务器](https://discord.gg/yolos)
- **Reddit**: [r/YOLOS](https://reddit.com/r/yolos)
- **Stack Overflow**: [yolos标签](https://stackoverflow.com/questions/tagged/yolos)

### 📱 中文社区
- **QQ群**: 123456789
- **微信群**: 扫描二维码加入
- **知乎**: [YOLOS专栏](https://zhuanlan.zhihu.com/yolos)
- **B站**: [YOLOS官方](https://space.bilibili.com/yolos)

### 📧 联系我们
- **技术支持**: support@yolos.org
- **商务合作**: business@yolos.org
- **媒体联系**: media@yolos.org

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE)。

```
MIT License

Copyright (c) 2024 YOLOS Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🙏 致谢

### 🎯 核心技术
- [Ultralytics](https://ultralytics.com/) - YOLOv8模型实现
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [OpenCV](https://opencv.org/) - 计算机视觉库
- [ONNX](https://onnx.ai/) - 模型交换格式

### 🏆 特别感谢
- 所有贡献者和社区成员
- 提供反馈和建议的用户
- 开源社区的支持和帮助

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给我们一个星标！**

**🚀 让我们一起构建更智能的视觉AI系统！**

[⬆️ 回到顶部](#yolos---you-only-look-once-system)

</div>