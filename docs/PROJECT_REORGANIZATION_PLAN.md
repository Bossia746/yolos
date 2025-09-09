# YOLOS项目重组计划

## 🎯 重组目标

1. **清理根目录**: 移除非核心文件到合适目录
2. **合并同类文档**: 整合重复和相似文档
3. **建立标准结构**: 符合业界标准的项目布局
4. **完善日志系统**: 可追溯的DEBUG支持
5. **维护核心边界**: 专注视频捕捉功能

## 📁 目标项目结构

```
yolos/
├── README.md                    # 项目主文档
├── LICENSE                      # 许可证
├── requirements.txt             # 依赖列表
├── setup.py                     # 安装脚本
├── .gitignore                   # Git忽略文件
├── 
├── src/                         # 核心源代码
│   ├── __init__.py
│   ├── core/                    # 核心功能
│   │   ├── video_capture.py     # 视频捕捉
│   │   ├── image_processor.py   # 图像处理
│   │   └── detector.py          # 检测器
│   ├── gui/                     # 图形界面
│   │   ├── main_gui.py          # 主界面
│   │   └── training_gui.py      # 训练界面
│   ├── training/                # 训练模块
│   ├── utils/                   # 工具模块
│   └── api/                     # API接口
│
├── config/                      # 配置文件
│   ├── default.yaml             # 默认配置
│   ├── camera.yaml              # 摄像头配置
│   └── logging.yaml             # 日志配置
│
├── docs/                        # 文档目录
│   ├── README.md                # 文档索引
│   ├── user_guide.md            # 用户指南
│   ├── api_reference.md         # API参考
│   ├── development_guide.md     # 开发指南
│   └── deployment_guide.md      # 部署指南
│
├── tests/                       # 测试代码
│   ├── unit/                    # 单元测试
│   ├── integration/             # 集成测试
│   └── performance/             # 性能测试
│
├── examples/                    # 示例代码
│   ├── basic_usage.py           # 基础使用
│   ├── training_example.py      # 训练示例
│   └── api_example.py           # API示例
│
├── scripts/                     # 工具脚本
│   ├── install.py               # 安装脚本
│   ├── setup_env.py             # 环境设置
│   └── cleanup.py               # 清理脚本
│
├── logs/                        # 日志目录
│   ├── system/                  # 系统日志
│   ├── debug/                   # 调试日志
│   └── performance/             # 性能日志
│
├── data/                        # 数据目录
│   ├── models/                  # 模型文件
│   ├── datasets/                # 数据集
│   └── cache/                   # 缓存文件
│
└── archive/                     # 归档目录
    ├── old_versions/            # 旧版本文件
    ├── deprecated/              # 废弃文件
    └── backup/                  # 备份文件
```

## 🗂️ 文件重组计划

### 需要移动的文件

#### 1. 版本号文件 → archive/old_versions/
```
0.10.9, 0.15.0, 1.3.0, 1.6.0, 1.10.0, 1.15.0, 1.16.0, 
2.0.0, 2.2.0, 2.28.0, 3.7.0, 4.64.0, 6.0, 8.1.0, 10.0.0
```

#### 2. 旧GUI文件 → archive/deprecated/
```
clean_multimodal_gui.py
enhanced_multimodal_gui.py
enhanced_object_recognition_gui.py
fixed_multimodal_gui.py
high_performance_gui.py
object_recognition_gui.py
offline_performance_gui.py
pet_recognition_gui.py
saved_pose_recognition_gui.py
self_learning_demo_gui.py
simple_multimodal_gui_test.py
simple_performance_gui.py
simple_pet_recognition_gui.py
stable_multimodal_gui.py
ultra_performance_gui.py
```

#### 3. 测试文件 → tests/
```
test_gui_multimodal.py
test_headless_multimodal.py
test_multi_target_priority_system.py
test_opencv_display.py
test_optimized_multimodal.py
test_self_learning_system.py
```

#### 4. 日志文件 → logs/system/
```
activity_analysis.log
clean_multimodal_test.log
enhanced_multimodal_test.log
enhanced_object_recognition.log
fixed_gui.log
gui_test.log
headless_test.log
high_performance_gui.log
multimodal_test.log
object_recognition.log
offline_performance_gui.log
pet_recognition.log
simple_multimodal_test.log
simple_performance_gui.log
stable_gui.log
stable_multimodal_test.log
```

#### 5. 报告文件 → docs/reports/
```
enhanced_recognition_report_20250908_175434.txt
enhanced_recognition_report_20250908_180348.txt
object_recognition_report_20250908_173810.txt
installation_report_1757355866.json
```

#### 6. 文档整合 → docs/
```
COMPLETE_DEPLOYMENT_GUIDE.md → docs/deployment_guide.md
DEBUG_GUIDE.md → docs/debug_guide.md
FINAL_MULTIMODAL_SOLUTION.md → docs/archive/
MULTIMODAL_GUI_IMPROVEMENTS.md → docs/archive/
OBJECT_RECOGNITION_SUMMARY.md → docs/archive/
PROBLEM_RESOLUTION_FINAL.md → docs/archive/
PROJECT_OVERVIEW.md → docs/project_overview.md
PROJECT_STATUS_REPORT.md → docs/project_status.md
```

#### 7. 脚本文件 → scripts/
```
activate_yolos.bat → scripts/
quick_install.py → scripts/install.py
setup_mirrors.py → scripts/
```

#### 8. 配置文件 → config/
```
camera_config.json → config/camera.yaml
```

### 保留在根目录的文件
```
README.md                    # 项目主文档
LICENSE                      # 许可证
requirements.txt             # 依赖列表
setup.py                     # 安装脚本
.gitignore                   # Git忽略文件
basic_pet_recognition_gui.py # 核心GUI (临时保留)
```

## 📋 文档合并计划

### 1. 用户文档整合
- **目标文件**: `docs/user_guide.md`
- **合并内容**:
  - 基础使用说明
  - 界面操作指南
  - 常见问题解答
  - 故障排除

### 2. 开发文档整合
- **目标文件**: `docs/development_guide.md`
- **合并内容**:
  - 代码结构说明
  - 开发环境设置
  - 编码规范
  - 贡献指南

### 3. 部署文档整合
- **目标文件**: `docs/deployment_guide.md`
- **合并内容**:
  - 系统要求
  - 安装步骤
  - 配置说明
  - 性能优化

### 4. API文档整合
- **目标文件**: `docs/api_reference.md`
- **合并内容**:
  - 核心API接口
  - 参数说明
  - 返回值格式
  - 使用示例

## 🔍 日志系统设计

### 日志分类
```
logs/
├── system/                  # 系统运行日志
│   ├── yolos_YYYYMMDD.log   # 主系统日志
│   ├── error_YYYYMMDD.log   # 错误日志
│   └── access_YYYYMMDD.log  # 访问日志
├── debug/                   # 调试日志
│   ├── video_YYYYMMDD.log   # 视频处理调试
│   ├── detect_YYYYMMDD.log  # 检测算法调试
│   └── gui_YYYYMMDD.log     # 界面调试
└── performance/             # 性能日志
    ├── fps_YYYYMMDD.log     # 帧率统计
    ├── memory_YYYYMMDD.log  # 内存使用
    └── cpu_YYYYMMDD.log     # CPU使用
```

### 日志格式标准
```
[YYYY-MM-DD HH:MM:SS.mmm] [LEVEL] [MODULE] [FUNCTION] MESSAGE
[2025-09-09 09:15:30.123] [INFO] [VideoCapture] [initialize_camera] Camera initialized successfully: 640x480
[2025-09-09 09:15:30.124] [DEBUG] [Detector] [detect_objects] Processing frame 1234, found 3 objects
[2025-09-09 09:15:30.125] [ERROR] [Training] [load_model] Failed to load model: file not found
```

### 日志级别定义
- **CRITICAL**: 系统崩溃级错误
- **ERROR**: 功能错误，但系统可继续运行
- **WARNING**: 警告信息，可能影响性能
- **INFO**: 一般信息，记录重要操作
- **DEBUG**: 详细调试信息，开发时使用

## 🚀 执行步骤

### 阶段1: 目录结构创建
1. 创建标准目录结构
2. 设置目录权限和属性
3. 创建必要的__init__.py文件

### 阶段2: 文件移动和重命名
1. 移动版本号文件到archive
2. 移动旧GUI文件到deprecated
3. 移动测试文件到tests目录
4. 移动日志文件到logs目录

### 阶段3: 文档整合
1. 合并用户文档
2. 合并开发文档
3. 合并部署文档
4. 创建API参考文档

### 阶段4: 日志系统实施
1. 创建日志配置文件
2. 实现统一日志管理器
3. 更新所有模块的日志调用
4. 测试日志系统功能

### 阶段5: 清理和验证
1. 清理根目录
2. 验证项目结构
3. 测试核心功能
4. 更新README文档

## ✅ 验收标准

### 项目结构
- [ ] 根目录只包含核心文件
- [ ] 所有文件都在合适的目录中
- [ ] 目录结构符合业界标准
- [ ] 文档组织清晰合理

### 日志系统
- [ ] 所有操作都有日志记录
- [ ] 日志格式统一标准
- [ ] 日志分类清晰
- [ ] 支持DEBUG追溯

### 功能完整性
- [ ] 核心功能正常工作
- [ ] 界面操作流畅
- [ ] API接口稳定
- [ ] 性能指标达标

### 文档质量
- [ ] 用户文档完整易懂
- [ ] 开发文档详细准确
- [ ] API文档规范完整
- [ ] 部署文档可操作

---

*重组计划制定时间: 2025-09-09 09:20*  
*预计完成时间: 2025-09-09 10:00*  
*负责团队: CodeBuddy项目组*