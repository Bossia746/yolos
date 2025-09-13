# YOLOS 系统架构文档

## 概述

YOLOS (You Only Look Once System) 是一个基于 YOLO 算法的智能目标检测系统，支持多平台部署和实时检测功能。

## 系统架构

### 核心模块结构

```
src/
├── core/                    # 核心功能模块
│   ├── __init__.py         # 模块导出接口
│   ├── config_manager.py   # 配置管理
│   ├── data_manager.py     # 数据管理
│   ├── event_bus.py        # 事件总线
│   ├── exceptions.py       # 异常处理
│   ├── logger.py           # 日志系统
│   ├── plugin_manager.py   # 插件管理
│   └── types.py            # 类型定义
├── models/                  # 模型管理模块
├── detection/              # 检测功能模块
├── recognition/            # 识别功能模块
├── utils/                  # 工具模块
└── gui/                    # 图形界面模块
```

### 模块依赖关系

1. **核心模块 (core)** - 系统基础设施
   - 配置管理 (ConfigManager)
   - 数据管理 (DataManager)
   - 事件系统 (EventBus)
   - 异常处理 (Exception Classes)
   - 日志系统 (YOLOSLogger)

2. **工具模块 (utils)** - 通用工具
   - 文件操作 (FileUtils)
   - 可视化 (Visualizer)
   - 性能指标 (MetricsCalculator)

3. **模型模块 (models)** - AI模型管理
   - YOLO工厂 (YOLOFactory)
   - 基础模型 (BaseModel)

4. **检测模块 (detection)** - 目标检测
   - 检测器工厂 (DetectorFactory)

5. **识别模块 (recognition)** - 目标识别
   - 识别工厂 (RecognitionFactory)

## 核心组件详解

### 1. 配置管理系统

**类**: `ConfigManager`
**位置**: `src/core/config_manager.py`

**主要方法**:
- `get_config(key)` - 获取配置项
- `set_config(key, value)` - 设置配置项
- `reload_config()` - 重新加载配置
- `watch_config()` - 监控配置变化

### 2. 数据管理系统

**类**: `DataManager`
**位置**: `src/core/data_manager.py`

**主要方法**:
- `store_training_data(data)` - 存储训练数据
- `retrieve_data(query)` - 检索数据
- `delete_data(id)` - 删除数据
- `backup_data()` - 备份数据

### 3. 异常处理系统

**基类**: `YOLOSException`
**位置**: `src/core/exceptions.py`

**异常层次结构**:
```
YOLOSException
├── SystemException      # 系统级异常
├── ModelException       # 模型相关异常
├── DataException        # 数据处理异常
├── ImageException       # 图像处理异常
├── DetectionException   # 检测过程异常
├── HardwareException    # 硬件相关异常
├── APIException         # API调用异常
├── PlatformException    # 平台相关异常
└── ConfigurationError   # 配置错误
```

### 4. 日志系统

**类**: `YOLOSLogger`
**位置**: `src/core/logger.py`

**功能**:
- 统一日志格式
- 多级别日志记录
- 性能监控
- 函数调用跟踪

## 开发规范

### 1. 模块导入规范

- 所有模块必须在 `__init__.py` 中明确导出
- 使用绝对导入，避免相对导入
- 导出列表必须包含在 `__all__` 中

**示例**:
```python
# 正确的导入方式
from core import ConfigManager, DataManager
from utils import FileUtils, Visualizer

# 避免的导入方式
from ..core import ConfigManager  # 相对导入
```

### 2. 异常处理规范

- 使用系统定义的异常类
- 提供详细的错误信息
- 记录异常上下文

**示例**:
```python
try:
    result = process_data(data)
except Exception as e:
    raise DataException(f"数据处理失败: {str(e)}") from e
```

### 3. 日志记录规范

- 使用统一的日志接口
- 记录关键操作和错误
- 包含必要的上下文信息

**示例**:
```python
from core import get_logger

logger = get_logger(__name__)
logger.info("开始处理数据", extra={"data_size": len(data)})
```

### 4. 配置管理规范

- 所有配置通过 ConfigManager 管理
- 配置文件使用 YAML 格式
- 支持环境变量覆盖

### 5. 测试规范

- 每个模块必须有对应的测试
- 使用自动化测试框架验证
- 测试覆盖率要求 > 80%

## 部署架构

### 支持的平台

1. **PC平台** - Windows/Linux/macOS
2. **嵌入式平台** - Raspberry Pi
3. **微控制器** - ESP32
4. **专用硬件** - K230

### 部署方式

1. **本地部署** - 直接运行
2. **容器部署** - Docker
3. **云端部署** - 支持各大云平台

## 性能指标

### 当前测试结果

- **模块导入成功率**: 40% (2/5 模块)
- **核心功能完整性**: 100%
- **工具模块完整性**: 100%
- **待修复模块**: models, detection, recognition

### 性能目标

- **检测延迟**: < 100ms
- **准确率**: > 90%
- **系统稳定性**: 99.9%

## 开发路线图

### 短期目标 (1-2周)

- [x] 修复核心模块导入问题
- [x] 建立自动化测试框架
- [ ] 修复相对导入问题
- [ ] 完善接口文档

### 中期目标 (1-2月)

- [ ] 优化检测性能
- [ ] 增强多平台支持
- [ ] 完善插件系统
- [ ] 添加更多模型支持

### 长期目标 (3-6月)

- [ ] 实现分布式部署
- [ ] 添加模型训练功能
- [ ] 支持更多硬件平台
- [ ] 建立完整的生态系统

## 故障排除

### 常见问题

1. **相对导入错误**
   - 原因: 模块使用了超出顶级包的相对导入
   - 解决: 修改为绝对导入或调整包结构

2. **模块导入失败**
   - 原因: __init__.py 中导出的项目不存在
   - 解决: 检查实际文件内容，更新导出列表

3. **配置加载失败**
   - 原因: 配置文件格式错误或路径不正确
   - 解决: 验证配置文件格式，检查文件路径

## 贡献指南

1. 遵循代码规范
2. 编写单元测试
3. 更新文档
4. 提交前运行测试框架

---

**文档版本**: 1.0  
**最后更新**: 2024年12月  
**维护者**: YOLOS开发团队