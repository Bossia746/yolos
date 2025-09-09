# YOLOS 开发规范

本文档定义了YOLOS项目的开发标准，确保代码质量、一致性和可维护性。

## 1. 插件开发标准

### 1.1 插件结构规范

所有插件必须遵循以下结构：

```python
class YourPlugin(BasePlugin):
    def __init__(self):
        metadata = PluginMetadata(
            name="YourPlugin",
            version="1.0.0",
            description="插件描述",
            author="作者名称",
            dependencies=["依赖列表"]
        )
        super().__init__(metadata)
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化插件"""
        pass
    
    def cleanup(self) -> bool:
        """清理插件资源"""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """获取插件状态"""
        pass
```

### 1.2 插件类型规范

#### 领域插件 (DomainPlugin)
- 继承自 `DomainPlugin`
- 实现特定领域的识别功能
- 必须实现 `process_frame()` 方法
- 支持多模态输入处理

#### 平台插件 (PlatformPlugin)
- 继承自 `PlatformPlugin`
- 提供平台特定的硬件抽象
- 实现 `get_hardware_info()` 方法
- 支持跨平台兼容性

#### 工具插件 (UtilityPlugin)
- 继承自 `UtilityPlugin`
- 提供通用工具功能
- 无平台依赖
- 可被其他插件调用

### 1.3 插件生命周期

1. **注册阶段**: 插件被发现并注册到插件管理器
2. **初始化阶段**: 调用 `initialize()` 方法
3. **运行阶段**: 插件处于活跃状态，响应事件
4. **清理阶段**: 调用 `cleanup()` 方法释放资源

### 1.4 错误处理规范

```python
try:
    # 插件逻辑
    result = self.process_data(data)
    return result
except Exception as e:
    logger.error(f"Plugin {self.metadata.name} error: {e}")
    self.status = PluginStatus.ERROR
    EventBus.emit('plugin_error', {
        'plugin': self.metadata.name,
        'error': str(e)
    })
    return None
```

## 2. 代码规范

### 2.1 Python代码风格

遵循 PEP 8 标准，具体要求：

- 使用4个空格缩进
- 行长度不超过88字符
- 类名使用 PascalCase
- 函数和变量名使用 snake_case
- 常量使用 UPPER_CASE

### 2.2 导入规范

```python
# 标准库导入
import os
import sys
from typing import Dict, List, Optional

# 第三方库导入
import numpy as np
import cv2

# 本地导入
from ..core.base_plugin import BasePlugin
from ..core.event_bus import EventBus
```

### 2.3 类型注解

所有公共方法必须包含类型注解：

```python
def process_frame(self, frame: np.ndarray, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """处理视频帧
    
    Args:
        frame: 输入图像帧
        metadata: 帧元数据
        
    Returns:
        处理结果字典，失败时返回None
    """
    pass
```

### 2.4 文档字符串

使用Google风格的文档字符串：

```python
def detect_objects(self, image: np.ndarray, confidence: float = 0.5) -> List[Dict[str, Any]]:
    """检测图像中的物体
    
    Args:
        image: 输入图像，格式为BGR
        confidence: 置信度阈值，范围[0, 1]
        
    Returns:
        检测结果列表，每个元素包含:
            - bbox: 边界框坐标 [x, y, w, h]
            - class_id: 类别ID
            - confidence: 置信度
            - label: 类别标签
            
    Raises:
        ValueError: 当confidence不在有效范围时
        RuntimeError: 当模型未初始化时
        
    Example:
        >>> detector = ObjectDetector()
        >>> results = detector.detect_objects(image, 0.7)
        >>> print(f"Found {len(results)} objects")
    """
    pass
```

### 2.5 日志规范

使用标准logging模块：

```python
import logging

logger = logging.getLogger(__name__)

# 日志级别使用
logger.debug("调试信息")
logger.info("一般信息")
logger.warning("警告信息")
logger.error("错误信息")
logger.critical("严重错误")
```

### 2.6 配置管理

使用配置管理器统一管理配置：

```python
from ..core.config_manager import ConfigManager

config = ConfigManager.get_instance()
model_path = config.get('models.yolo.path', 'default_path')
confidence = config.get('detection.confidence', 0.5)
```

## 3. 测试规范

### 3.1 单元测试

每个插件必须包含单元测试：

```python
import unittest
from unittest.mock import Mock, patch

class TestYourPlugin(unittest.TestCase):
    def setUp(self):
        self.plugin = YourPlugin()
        
    def test_initialize(self):
        config = {'param1': 'value1'}
        result = self.plugin.initialize(config)
        self.assertTrue(result)
        
    def test_process_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.plugin.process_frame(frame, {})
        self.assertIsNotNone(result)
        
    def tearDown(self):
        self.plugin.cleanup()
```

### 3.2 集成测试

测试插件间的交互：

```python
class TestPluginIntegration(unittest.TestCase):
    def test_plugin_communication(self):
        # 测试插件间事件通信
        pass
        
    def test_plugin_dependencies(self):
        # 测试插件依赖关系
        pass
```

### 3.3 性能测试

```python
import time
import psutil

class TestPluginPerformance(unittest.TestCase):
    def test_processing_speed(self):
        plugin = YourPlugin()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        start_time = time.time()
        result = plugin.process_frame(frame, {})
        end_time = time.time()
        
        processing_time = end_time - start_time
        self.assertLess(processing_time, 0.1)  # 要求处理时间小于100ms
        
    def test_memory_usage(self):
        # 测试内存使用情况
        pass
```

## 4. 文档规范

### 4.1 README文档

每个插件目录必须包含README.md：

```markdown
# 插件名称

## 概述
简要描述插件功能

## 安装
依赖安装说明

## 配置
配置参数说明

## 使用示例
代码示例

## API文档
接口说明

## 性能指标
性能基准测试结果
```

### 4.2 API文档

使用Sphinx生成API文档：

```python
class YourPlugin(BasePlugin):
    """插件类描述
    
    这是一个示例插件，展示了如何实现基本功能。
    
    Attributes:
        model: 机器学习模型实例
        config: 插件配置字典
        
    Example:
        >>> plugin = YourPlugin()
        >>> plugin.initialize({'param': 'value'})
        >>> result = plugin.process_frame(frame, {})
    """
    pass
```

### 4.3 变更日志

维护CHANGELOG.md记录版本变更：

```markdown
# 变更日志

## [1.1.0] - 2024-01-15
### 新增
- 添加新的检测算法
- 支持GPU加速

### 修改
- 优化内存使用
- 提升处理速度

### 修复
- 修复内存泄漏问题
- 修复配置加载错误
```

## 5. 版本管理

### 5.1 语义化版本

使用语义化版本号 (MAJOR.MINOR.PATCH)：

- MAJOR: 不兼容的API修改
- MINOR: 向后兼容的功能性新增
- PATCH: 向后兼容的问题修正

### 5.2 分支策略

- `main`: 主分支，稳定版本
- `develop`: 开发分支
- `feature/*`: 功能分支
- `hotfix/*`: 热修复分支

## 6. 代码审查

### 6.1 审查清单

- [ ] 代码符合PEP 8规范
- [ ] 包含完整的类型注解
- [ ] 包含详细的文档字符串
- [ ] 包含单元测试
- [ ] 错误处理完善
- [ ] 性能符合要求
- [ ] 安全性检查通过

### 6.2 性能要求

- 帧处理时间 < 100ms (标准硬件)
- 内存使用 < 500MB (单个插件)
- CPU使用率 < 80% (峰值)
- 启动时间 < 5s

## 7. 安全规范

### 7.1 输入验证

```python
def process_input(self, data: Any) -> bool:
    """验证输入数据"""
    if not isinstance(data, expected_type):
        raise ValueError(f"Expected {expected_type}, got {type(data)}")
        
    if not self._validate_data(data):
        raise ValueError("Invalid data format")
        
    return True
```

### 7.2 敏感信息处理

- 不在日志中记录敏感信息
- 使用环境变量存储密钥
- 加密存储用户数据

### 7.3 权限控制

- 最小权限原则
- 文件访问权限检查
- 网络访问限制

## 8. 部署规范

### 8.1 容器化

提供Dockerfile：

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```

### 8.2 配置管理

- 使用环境变量覆盖配置
- 提供默认配置文件
- 支持配置热重载

### 8.3 监控和日志

- 结构化日志输出
- 健康检查端点
- 性能指标收集

## 9. 贡献指南

### 9.1 提交规范

使用约定式提交格式：

```
type(scope): description

[optional body]

[optional footer]
```

类型：
- `feat`: 新功能
- `fix`: 修复
- `docs`: 文档
- `style`: 格式
- `refactor`: 重构
- `test`: 测试
- `chore`: 构建过程或辅助工具的变动

### 9.2 Pull Request

- 描述清晰的变更内容
- 包含相关的测试
- 通过所有CI检查
- 获得代码审查批准

## 10. 工具配置

### 10.1 代码格式化

使用black进行代码格式化：

```bash
black --line-length 88 src/
```

### 10.2 代码检查

使用flake8进行代码检查：

```bash
flake8 src/ --max-line-length=88
```

### 10.3 类型检查

使用mypy进行类型检查：

```bash
mypy src/ --strict
```

### 10.4 测试覆盖率

使用pytest-cov检查测试覆盖率：

```bash
pytest --cov=src/ --cov-report=html
```

要求测试覆盖率 > 80%

---

遵循这些开发规范将确保YOLOS项目的代码质量、可维护性和团队协作效率。所有贡献者都应该熟悉并遵循这些标准。