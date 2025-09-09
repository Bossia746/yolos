# YOLOS - 多平台目标检测系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)

## 📋 项目概述

YOLOS是一个基于YOLO的多平台目标检测系统，专注于计算机视觉任务的实际应用。项目采用模块化架构，支持多种检测功能和硬件平台部署。

### ✅ 核心功能

#### 1. 目标检测
- **YOLO检测**: 支持YOLOv5/YOLOv8模型，可检测多种日常物体
- **实时处理**: 单张图像处理时间50-100ms
- **多格式支持**: JPG、PNG、BMP等常见图像格式

#### 2. 多模态识别
- **人脸检测**: 基于MediaPipe和InsightFace
- **姿态分析**: 人体关键点检测
- **手势识别**: 基础手势识别功能
- **跌倒检测**: 基于姿态分析的安全监控

#### 3. 图像处理
- **质量增强**: 图像去噪、锐化、对比度调整
- **可视化**: 检测结果标注和报告生成
- **批量处理**: 支持多图像批量检测

#### 4. 第三方集成
- **ModelScope API**: 集成阿里云魔搭社区视觉大模型
- **离线模式**: 支持无网络环境下的本地推理

## 🚀 快速开始

### 系统要求
- **操作系统**: Windows 10+, Ubuntu 18.04+, macOS 10.15+
- **Python**: 3.8或更高版本
- **内存**: 8GB RAM (推荐16GB)
- **存储**: 10GB可用空间

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

# 4. 安装核心依赖
pip install ultralytics opencv-python numpy pyyaml psutil

# 5. 运行基础测试
python real_yolo_test.py
```

### 快速测试

```bash
# 生成YOLO检测报告
python enhanced_visual_yolo_test.py

# 查看检测结果
# 报告将在浏览器中自动打开
```

## 📁 项目结构

```
yolos/
├── src/                    # 源代码目录
│   ├── recognition/        # 识别算法模块
│   ├── detection/          # 检测功能
│   ├── gui/               # 图形界面
│   ├── utils/             # 工具函数
│   ├── plugins/           # 插件系统
│   └── core/              # 核心功能
├── config/                # 配置文件
├── docs/                  # 项目文档
├── resource/              # 资源文件
│   └── training image/    # 测试图像
├── tests/                 # 测试代码
├── scripts/               # 脚本工具
└── deployments/           # 部署配置
    ├── esp32/             # ESP32平台
    ├── raspberry_pi/      # 树莓派平台
    └── pc/                # PC平台
```

## 🔧 配置说明

### ModelScope API配置

如需使用大模型增强功能，编辑 `config/modelscope_llm_config.yaml`:

```yaml
api:
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  api_key: "your_api_key_here"  # 需要申请API密钥
  model: "qwen-vl-max"

settings:
  max_tokens: 1000
  temperature: 0.1
  timeout: 30
```

### 基础使用示例

```python
from ultralytics import YOLO
import cv2

# 加载YOLO模型
model = YOLO('yolov8n.pt')  # 首次运行会自动下载

# 读取图像
image = cv2.imread('path/to/your/image.jpg')

# 进行检测
results = model(image)

# 获取检测结果
for result in results:
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            # 获取类别和置信度
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]
            print(f"检测到: {class_name}, 置信度: {confidence:.2f}")
```

## 📊 实际性能表现

基于真实测试数据（2025年9月测试）:

| 功能模块 | 处理速度 | 检测精度 | 资源占用 |
|----------|----------|----------|----------|
| YOLOv8n检测 | 50-100ms | 中等 | 低 |
| 人脸检测 | 30-80ms | 良好 | 中等 |
| 姿态检测 | 100-200ms | 中等 | 高 |
| 图像增强 | 20-50ms | - | 低 |

### 测试结果示例
- **处理图像**: 8张测试图像
- **检测物体**: 10个物体（人、瓶子、书籍、交通标志等）
- **识别类别**: 5种不同类别
- **平均置信度**: 31.9%-50.1%

## 🧪 功能测试

### 可用的测试脚本

```bash
# YOLO检测测试（生成可视化报告）
python real_yolo_test.py

# 增强版可视化测试（原图vs检测结果对比）
python enhanced_visual_yolo_test.py

# 可视化报告生成
python visual_yolo_report.py
```

### 测试图像
项目包含测试图像位于 `resource/training image/` 目录，包括：
- 人物图像
- 日常物品
- 交通场景
- 医疗相关图像

## ⚠️ 当前限制和已知问题

### 功能限制
1. **检测精度**: 在复杂场景下置信度可能偏低
2. **实时性能**: 在低端硬件上处理速度有限
3. **模型支持**: 主要支持YOLOv8，其他版本需要额外配置
4. **平台兼容**: 嵌入式平台功能受硬件限制

### 已知问题
- 某些GUI组件可能不稳定
- ESP32部署需要进一步优化
- 部分文档可能与代码不完全同步

### 改进建议
- 建议在配置较好的设备上运行以获得最佳性能
- 复杂应用场景建议进行针对性优化
- 生产环境使用前请充分测试

## 🔄 开发路线图

### ✅ 已完成 (v1.0)
- [x] 基础YOLO检测功能
- [x] 多模态识别框架
- [x] 可视化报告生成
- [x] ModelScope API集成
- [x] 基础平台适配

### 🚧 开发中 (v1.1)
- [ ] 检测精度优化
- [ ] 性能提升
- [ ] 更完善的文档
- [ ] 更多测试用例

### 📋 计划中 (v2.0)
- [ ] YOLOv11支持
- [ ] 移动端适配
- [ ] 云端部署方案
- [ ] 企业级功能

## 📚 文档资源

### 核心文档
- [YOLO最新研究报告](docs/YOLO_Latest_Research_and_Optimization_Report.md)
- [ModelScope集成指南](docs/modelscope_integration_guide.md)
- [部署指南](docs/deployment_guide.md)
- [架构文档](docs/ARCHITECTURE.md)

### 技术文档
- [API文档](docs/API.md)
- [开发标准](docs/DEVELOPMENT_STANDARDS.md)
- [调试指南](docs/debug_guide.md)

## 🤝 贡献指南

1. Fork 本项目
2. 创建功能分支 (`git checkout -b feature/NewFeature`)
3. 提交更改 (`git commit -m 'Add NewFeature'`)
4. 推送到分支 (`git push origin feature/NewFeature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🆘 获取帮助

### 问题反馈
- **Bug报告**: 通过 GitHub Issues 提交
- **功能请求**: 通过 GitHub Discussions 讨论
- **使用问题**: 查看 `docs/` 目录下的相关文档

### 社区支持
- **项目文档**: 查看 `docs/` 目录
- **示例代码**: 查看项目根目录的测试脚本
- **技术交流**: 欢迎提交 Issue 或 PR

## 🎯 适用场景

### ✅ 推荐使用场景
- 计算机视觉学习和研究
- 原型开发和概念验证
- 小规模检测应用
- 教育培训项目

### ⚠️ 需要评估的场景
- 高精度商业应用（需要针对性优化）
- 大规模生产环境（需要性能调优）
- 实时关键任务（需要专业优化）

---

<div align="center">

**🔍 专注实用，持续改进 🔍**

[📖 查看文档](docs/) | [🐛 报告问题](../../issues) | [💡 功能建议](../../discussions)

</div>

---

> **免责声明**: 本项目仅供学习和研究使用。在生产环境中使用前，请根据具体需求进行充分的测试和优化。项目功能和性能会持续改进，使用时请以实际测试结果为准。