# YOLOS 大模型自学习系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🌟 项目概述

YOLOS是一个革命性的多模态AI识别系统，结合了传统计算机视觉技术和大模型自学习能力。系统特别针对AIoT应用场景优化，支持老人防摔监控、医疗健康检测、智能安防等多种应用。

### 🎯 核心特性

- **🧠 大模型自学习**: 集成GPT-4V、Claude-3等多种大模型API，实现未知场景的自动学习
- **🔄 离线+在线混合**: 弱网环境下优先使用离线模型，网络良好时启用在线学习
- **🏥 医疗级检测**: 专业的面部生理分析、跌倒检测、药物识别功能
- **🤖 AIoT全平台**: 支持ESP32、树莓派、STM32等主流AIoT开发板
- **📱 跨平台部署**: Windows、Linux、macOS全平台支持
- **🔧 模块化架构**: 可插拔的功能模块，按需启用

### 🏆 主要功能

#### 1. 多模态识别
- **人体检测**: 人脸、手势、姿态、表情识别
- **物体识别**: 动态/静态物体、交通标志、二维码识别
- **生物识别**: 宠物、植物（叶子、花朵、果实）识别
- **医疗识别**: 药物外观、医疗器械识别

#### 2. 智能监控
- **跌倒检测**: 实时监控老人跌倒，自动报警
- **健康分析**: 面部生理状态检测，疾病预警
- **行为分析**: 异常行为识别，安全监控
- **环境监测**: 危险物品检测，安全评估

#### 3. AIoT集成
- **边缘计算**: 本地AI推理，降低延迟
- **云端协同**: 复杂场景云端分析
- **设备联动**: 多设备协同工作
- **远程监控**: 手机APP远程查看

## 🚀 快速开始

### 一键安装

```bash
# 克隆项目
git clone <repository_url>
cd yolos

# 运行快速安装脚本
python quick_install.py
```

### 手动安装

```bash
# 1. 创建虚拟环境
python -m venv yolos_env

# 2. 激活虚拟环境
# Windows:
yolos_env\Scripts\activate
# Linux/macOS:
source yolos_env/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 运行测试
python test_self_learning_system.py
```

### 启动GUI演示

```bash
# 启动自学习演示界面
python self_learning_demo_gui.py

# 启动多模态识别界面
python fixed_multimodal_gui.py
```

## 📋 系统要求

### 最低配置
- **CPU**: Intel i5 或 AMD Ryzen 5
- **内存**: 8GB RAM
- **存储**: 20GB 可用空间
- **Python**: 3.8+

### 推荐配置
- **CPU**: Intel i7 或 AMD Ryzen 7
- **内存**: 16GB RAM
- **GPU**: NVIDIA GTX 1060+ (可选)
- **存储**: 50GB SSD

### 支持平台
- **桌面**: Windows 10+, Ubuntu 18.04+, macOS 10.15+
- **AIoT**: ESP32, 树莓派, STM32, Jetson Nano
- **云端**: Docker, Kubernetes

## 🏥 老人防摔应用示例

### 硬件配置
```
ESP32-CAM (摄像头节点)
    ↓ WiFi/MQTT
树莓派4B (分析服务器)
    ↓ 4G/WiFi
云端服务 (紧急响应)
```

### 部署步骤

1. **ESP32-CAM配置**
```cpp
// 编辑 esp32/yolos_esp32_cam/yolos_esp32_cam.ino
const char* ssid = "YOUR_WIFI";
const char* password = "YOUR_PASSWORD";
const char* mqtt_server = "192.168.1.100";
```

2. **树莓派服务器**
```bash
# 安装系统
python -m src.recognition.enhanced_fall_detection_system

# 启用开机自启
sudo systemctl enable yolos-fall-detection.service
```

3. **配置紧急联系**
```yaml
# config/aiot_platform_config.yaml
emergency_response:
  enabled: true
  contacts:
    - phone: "+86138xxxxxxxx"
      name: "家属1"
    - phone: "+86139xxxxxxxx"
      name: "家属2"
```

## 🔧 配置说明

### 大模型API配置

编辑 `config/self_learning_config.yaml`:

```yaml
llm_providers:
  openai:
    api_key: "your_openai_api_key"
    model: "gpt-4-vision-preview"
    enabled: true
  
  claude:
    api_key: "your_claude_api_key"
    model: "claude-3-sonnet-20240229"
    enabled: true
  
  local_llm:
    model_path: "models/local_llm"
    enabled: true
```

### 硬件平台配置

```yaml
platform_config:
  esp32:
    enabled: true
    mqtt_broker: "192.168.1.100"
    camera_resolution: "SVGA"
  
  raspberry_pi:
    enabled: true
    gpio_pins:
      led_status: 18
      button_reset: 24
  
  stm32:
    enabled: true
    uart_port: "/dev/ttyUSB0"
    baud_rate: 115200
```

## 📚 文档目录

- **[完整部署指南](COMPLETE_DEPLOYMENT_GUIDE.md)** - 详细的安装和配置指南
- **[自学习系统指南](docs/self_learning_system_guide.md)** - 大模型自学习功能说明
- **[AIoT平台指南](docs/modular_aiot_platform_complete_guide.md)** - AIoT平台使用指南
- **[API文档](docs/API.md)** - 开发者API参考
- **[架构文档](docs/ARCHITECTURE.md)** - 系统架构说明

## 🧪 测试和验证

### 运行测试套件
```bash
# 完整系统测试
python test_self_learning_system.py

# AIoT兼容性测试
python -m tests.test_aiot_compatibility

# 性能基准测试
python -m tests.performance_test
```

### GUI功能测试
```bash
# 启动自学习演示
python self_learning_demo_gui.py

# 测试摄像头识别
python basic_pet_recognition_gui.py

# 多模态识别测试
python fixed_multimodal_gui.py
```

## 🔌 扩展开发

### 添加新的识别模块

```python
# src/plugins/domain/custom_recognition.py
from ..base_plugin import BasePlugin

class CustomRecognitionPlugin(BasePlugin):
    def __init__(self):
        super().__init__("custom_recognition", "1.0.0")
    
    def process(self, image, context):
        # 实现自定义识别逻辑
        return recognition_result
```

### 集成新的硬件平台

```python
# src/plugins/platform/custom_platform_adapter.py
from .base_adapter import BasePlatformAdapter

class CustomPlatformAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__("custom_platform")
    
    def initialize(self):
        # 初始化硬件
        pass
    
    def capture_image(self):
        # 获取图像
        return image
```

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🆘 获取帮助

### 常见问题
- **安装问题**: 查看 [COMPLETE_DEPLOYMENT_GUIDE.md](COMPLETE_DEPLOYMENT_GUIDE.md)
- **配置问题**: 查看 [docs/](docs/) 目录下的相关文档
- **API使用**: 查看 [docs/API.md](docs/API.md)

### 技术支持
- **GitHub Issues**: 报告问题和功能请求
- **文档**: 查看 `docs/` 目录
- **示例代码**: 查看 `examples/` 目录

### 联系方式
- **项目主页**: [GitHub Repository]
- **技术文档**: [Documentation Site]
- **社区论坛**: [Community Forum]

## 🎯 路线图

### v1.0 (当前版本)
- ✅ 基础多模态识别
- ✅ 大模型自学习
- ✅ AIoT平台支持
- ✅ 老人防摔应用

### v1.1 (计划中)
- 🔄 更多大模型支持
- 🔄 增强的边缘计算
- 🔄 移动端APP
- 🔄 云端管理平台

### v2.0 (未来版本)
- 📋 3D场景理解
- 📋 多模态对话
- 📋 自动化部署
- 📋 企业级管理

## 🌟 致谢

感谢以下开源项目的支持：
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [OpenCV](https://opencv.org/) - 计算机视觉库
- [Ultralytics](https://ultralytics.com/) - YOLO实现
- [MediaPipe](https://mediapipe.dev/) - 多模态检测
- [ESP32](https://www.espressif.com/) - AIoT硬件平台

---

<div align="center">

**🚀 让AI触手可及，让智能无处不在 🚀**

[⭐ 给项目点个星](https://github.com/your-repo/yolos) | [📖 查看文档](docs/) | [🐛 报告问题](https://github.com/your-repo/yolos/issues)

</div>