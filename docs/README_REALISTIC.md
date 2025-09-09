# YOLOS - 多平台目标检测系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)

## 📋 项目概述

YOLOS是一个基于YOLO的多平台目标检测系统，支持多种计算机视觉任务。项目采用模块化架构，可部署在不同硬件平台上。

### ✅ 已实现功能

#### 1. 核心检测能力
- **目标检测**: 基于YOLOv5/v8的物体识别
- **人脸检测**: 使用MediaPipe和InsightFace
- **姿态检测**: 人体关键点检测和姿态分析
- **手势识别**: 基本手势识别功能
- **跌倒检测**: 基于姿态分析的跌倒识别

#### 2. 图像处理
- **图像质量增强**: 去噪、锐化、对比度调整
- **反欺骗检测**: 基础的活体检测
- **多模态融合**: 结合多种检测结果

#### 3. 平台支持
- **桌面平台**: Windows、Linux、macOS
- **嵌入式**: ESP32、树莓派支持（部分功能）
- **容器化**: Docker部署支持

#### 4. 第三方集成
- **ModelScope API**: 集成魔搭社区的视觉大模型
- **本地推理**: 支持离线模式运行

## 🚀 快速开始

### 环境要求
- Python 3.8+
- 8GB+ RAM
- 支持的操作系统: Windows 10+, Ubuntu 18.04+, macOS 10.15+

### 安装步骤

```bash
# 1. 克隆项目
git clone <repository_url>
cd yolos

# 2. 创建虚拟环境
python -m venv yolos_env

# 3. 激活虚拟环境
# Windows:
yolos_env\Scripts\activate
# Linux/macOS:
source yolos_env/bin/activate

# 4. 安装依赖
pip install -r requirements.txt

# 5. 运行基础测试
python real_yolo_test.py
```

### 基础使用示例

```python
from ultralytics import YOLO

# 加载YOLO模型
model = YOLO('yolov8n.pt')

# 进行检测
results = model('path/to/image.jpg')

# 显示结果
results[0].show()
```

## 📁 项目结构

```
yolos/
├── src/                    # 源代码
│   ├── recognition/        # 识别模块
│   ├── detection/          # 检测算法
│   ├── gui/               # 图形界面
│   ├── utils/             # 工具函数
│   └── plugins/           # 插件系统
├── config/                # 配置文件
├── docs/                  # 文档
├── tests/                 # 测试代码
└── requirements.txt       # 依赖列表
```

## 🔧 配置说明

### ModelScope API配置

编辑 `config/modelscope_llm_config.yaml`:

```yaml
api:
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  api_key: "your_api_key_here"
  model: "qwen-vl-max"

settings:
  max_tokens: 1000
  temperature: 0.1
  timeout: 30
```

### 硬件平台配置

```yaml
# config/platform_config.yaml
platforms:
  esp32:
    enabled: true
    camera_resolution: "SVGA"
  
  raspberry_pi:
    enabled: true
    gpio_pins:
      status_led: 18
```

## 🧪 测试功能

### 运行检测测试
```bash
# YOLO检测测试
python real_yolo_test.py

# 生成可视化报告
python enhanced_visual_yolo_test.py
```

### 功能测试
```bash
# 多模态检测测试
python -m src.recognition.multimodal_detector

# 图像质量测试
python -m src.recognition.image_quality_enhancer
```

## 📊 性能指标

基于实际测试结果：

| 功能 | 处理速度 | 准确率 | 支持平台 |
|------|----------|--------|----------|
| YOLO检测 | 50-100ms/图 | 中等 | 全平台 |
| 人脸检测 | 30-80ms/图 | 良好 | 全平台 |
| 姿态检测 | 100-200ms/图 | 中等 | 桌面平台 |
| 手势识别 | 80-150ms/图 | 基础 | 桌面平台 |

## ⚠️ 当前限制

### 功能限制
- **大模型集成**: 仅支持ModelScope API，其他API需要额外配置
- **实时性能**: 在低端硬件上性能有限
- **检测精度**: 部分场景下置信度偏低
- **平台兼容**: 嵌入式平台功能受限

### 已知问题
- ESP32部署需要额外优化
- 某些GUI组件可能不稳定
- 文档与代码存在不一致

## 🔄 开发状态

### ✅ 已完成
- [x] 基础YOLO检测
- [x] 多模态识别框架
- [x] ModelScope API集成
- [x] 可视化报告生成
- [x] 基础平台适配

### 🚧 开发中
- [ ] 性能优化
- [ ] 更多大模型支持
- [ ] 移动端适配
- [ ] 完整的ESP32支持

### 📋 计划中
- [ ] YOLOv11集成
- [ ] 云端部署方案
- [ ] 企业级管理界面
- [ ] 完整的文档更新

## 📚 文档

- [部署指南](docs/deployment_guide.md)
- [API文档](docs/API.md)
- [架构说明](docs/ARCHITECTURE.md)
- [ModelScope集成](docs/modelscope_integration_guide.md)

## 🤝 贡献

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 发起 Pull Request

## 📄 许可证

MIT License - 查看 [LICENSE](LICENSE) 文件

## 🆘 获取帮助

- **问题报告**: GitHub Issues
- **功能请求**: GitHub Discussions
- **文档**: `docs/` 目录

## 🎯 实际应用场景

### 适用场景
- 基础物体检测和识别
- 教育和研究项目
- 原型开发和概念验证
- 小规模部署应用

### 不适用场景
- 高精度商业应用（需要进一步优化）
- 大规模生产环境（需要性能调优）
- 实时关键任务（需要专业优化）

---

**注意**: 本项目处于持续开发中，功能和性能会不断改进。使用前请仔细评估是否满足您的具体需求。