# YOLOS 开发者指南

## 概述

欢迎参与YOLOS项目的开发！本指南将帮助您了解项目结构、开发流程、编码规范和贡献方式。

## 项目架构

### 目录结构

```
yolos/
├── src/                    # 源代码
│   ├── core/              # 核心模块
│   │   ├── application.py # 应用程序主类
│   │   ├── config_manager.py # 配置管理
│   │   ├── event_system.py # 事件系统
│   │   ├── module_manager.py # 模块管理
│   │   └── test_framework.py # 测试框架
│   ├── applications/      # 应用模块
│   │   ├── face_recognition.py
│   │   ├── pose_estimation.py
│   │   └── object_detection.py
│   ├── models/           # 模型相关
│   ├── utils/            # 工具函数
│   └── gui/              # 图形界面
├── tests/                # 测试代码
├── docs/                 # 文档
├── config/               # 配置文件
├── deployments/          # 部署相关
└── examples/             # 示例代码
```

### 核心架构设计

#### 1. 模块化架构

```python
# 模块接口定义
class IModule:
    def initialize(self) -> bool:
        """模块初始化"""
        pass
    
    def cleanup(self) -> None:
        """模块清理"""
        pass
    
    def get_info(self) -> dict:
        """获取模块信息"""
        pass
```

#### 2. 事件驱动系统

```python
# 事件发布/订阅模式
from core.event_system import EventBus

event_bus = EventBus()

# 发布事件
event_bus.publish('model_loaded', {'model_name': 'yolov8n'})

# 订阅事件
@event_bus.subscribe('model_loaded')
def on_model_loaded(event_data):
    print(f"模型已加载: {event_data['model_name']}")
```

#### 3. 依赖注入

```python
# 服务注册和解析
from core.dependency_injection import DIContainer

container = DIContainer()

# 注册服务
container.register('logger', LoggingService)
container.register('config', ConfigService)

# 解析服务
logger = container.resolve('logger')
```

## 开发环境设置

### 1. 环境要求

- Python 3.8+
- Git
- 推荐使用虚拟环境

### 2. 克隆项目

```bash
git clone https://github.com/yolos/yolos.git
cd yolos
```

### 3. 安装依赖

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# 安装开发依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 4. 开发工具配置

#### Pre-commit钩子

```bash
# 安装pre-commit
pip install pre-commit

# 安装钩子
pre-commit install
```

#### IDE配置

**VS Code推荐设置** (`.vscode/settings.json`):

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

## 编码规范

### 1. Python代码风格

我们遵循PEP 8标准，并使用以下工具：

- **Black**: 代码格式化
- **isort**: 导入排序
- **pylint**: 代码检查
- **mypy**: 类型检查

#### 代码格式化

```bash
# 格式化代码
black src/ tests/

# 排序导入
isort src/ tests/

# 代码检查
pylint src/

# 类型检查
mypy src/
```

### 2. 命名规范

```python
# 类名：PascalCase
class ModelManager:
    pass

# 函数和变量：snake_case
def load_model(model_path: str) -> bool:
    model_config = {}
    return True

# 常量：UPPER_SNAKE_CASE
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
MAX_DETECTIONS = 1000

# 私有成员：前缀下划线
class MyClass:
    def __init__(self):
        self._private_var = None
        self.__very_private = None
```

### 3. 文档字符串

使用Google风格的文档字符串：

```python
def detect_objects(image: np.ndarray, 
                  confidence_threshold: float = 0.5) -> List[Detection]:
    """检测图像中的对象。
    
    Args:
        image: 输入图像，numpy数组格式
        confidence_threshold: 置信度阈值，默认0.5
    
    Returns:
        检测结果列表，每个元素包含边界框和类别信息
    
    Raises:
        ValueError: 当输入图像格式不正确时
        ModelError: 当模型推理失败时
    
    Example:
        >>> import numpy as np
        >>> image = np.random.rand(640, 640, 3)
        >>> detections = detect_objects(image)
        >>> print(len(detections))
        5
    """
    pass
```

### 4. 类型注解

```python
from typing import List, Dict, Optional, Union, Tuple
import numpy as np

def process_batch(images: List[np.ndarray], 
                 batch_size: int = 4) -> Dict[str, Union[int, float]]:
    """批量处理图像。"""
    pass

class Detection:
    """检测结果类。"""
    
    def __init__(self, 
                 bbox: Tuple[float, float, float, float],
                 confidence: float,
                 class_id: int,
                 class_name: Optional[str] = None):
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
```

## 测试指南

### 1. 测试结构

```
tests/
├── unit/                  # 单元测试
│   ├── test_core/
│   ├── test_models/
│   └── test_utils/
├── integration/           # 集成测试
├── performance/           # 性能测试
└── fixtures/             # 测试数据
```

### 2. 编写测试

#### 单元测试示例

```python
import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.models.yolo_model import YOLOModel
from src.core.exceptions import ModelLoadError

class TestYOLOModel:
    """YOLO模型测试类。"""
    
    def setup_method(self):
        """测试前置设置。"""
        self.model = YOLOModel()
        self.test_image = np.random.rand(640, 640, 3).astype(np.uint8)
    
    def test_model_initialization(self):
        """测试模型初始化。"""
        assert self.model is not None
        assert hasattr(self.model, 'predict')
    
    def test_predict_with_valid_image(self):
        """测试有效图像预测。"""
        with patch.object(self.model, '_load_model'):
            result = self.model.predict(self.test_image)
            assert isinstance(result, list)
    
    def test_predict_with_invalid_image(self):
        """测试无效图像预测。"""
        with pytest.raises(ValueError):
            self.model.predict(None)
    
    @pytest.mark.parametrize("confidence", [0.1, 0.5, 0.9])
    def test_different_confidence_thresholds(self, confidence):
        """测试不同置信度阈值。"""
        with patch.object(self.model, '_load_model'):
            result = self.model.predict(self.test_image, confidence=confidence)
            # 验证结果...
```

#### 集成测试示例

```python
import pytest
from src.core.application import Application
from src.applications.object_detection import ObjectDetectionApp

class TestApplicationIntegration:
    """应用集成测试。"""
    
    def setup_method(self):
        """测试前置设置。"""
        self.app = Application()
        self.app.initialize()
    
    def teardown_method(self):
        """测试后清理。"""
        self.app.cleanup()
    
    def test_object_detection_workflow(self):
        """测试目标检测完整流程。"""
        # 注册应用
        obj_det = ObjectDetectionApp()
        self.app.register_application('object_detection', obj_det)
        
        # 执行检测
        result = self.app.process_image('test_images/sample.jpg')
        
        # 验证结果
        assert 'detections' in result
        assert isinstance(result['detections'], list)
```

### 3. 性能测试

```python
import time
import pytest
from src.models.yolo_model import YOLOModel

class TestPerformance:
    """性能测试类。"""
    
    def test_inference_speed(self):
        """测试推理速度。"""
        model = YOLOModel()
        image = np.random.rand(640, 640, 3).astype(np.uint8)
        
        # 预热
        for _ in range(5):
            model.predict(image)
        
        # 性能测试
        start_time = time.time()
        for _ in range(100):
            model.predict(image)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        assert avg_time < 0.1  # 平均推理时间应小于100ms
    
    @pytest.mark.benchmark
    def test_batch_processing_benchmark(self, benchmark):
        """批处理性能基准测试。"""
        model = YOLOModel()
        images = [np.random.rand(640, 640, 3).astype(np.uint8) for _ in range(10)]
        
        result = benchmark(model.predict_batch, images)
        assert len(result) == 10
```

### 4. 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/unit/test_models.py

# 运行特定测试类
pytest tests/unit/test_models.py::TestYOLOModel

# 运行特定测试方法
pytest tests/unit/test_models.py::TestYOLOModel::test_model_initialization

# 生成覆盖率报告
pytest --cov=src --cov-report=html

# 运行性能测试
pytest -m benchmark
```

## 贡献流程

### 1. 分支策略

我们使用Git Flow分支模型：

- `main`: 主分支，稳定版本
- `develop`: 开发分支，最新功能
- `feature/*`: 功能分支
- `hotfix/*`: 热修复分支
- `release/*`: 发布分支

### 2. 功能开发流程

```bash
# 1. 从develop创建功能分支
git checkout develop
git pull origin develop
git checkout -b feature/new-detection-algorithm

# 2. 开发功能
# ... 编写代码 ...

# 3. 提交代码
git add .
git commit -m "feat: add new detection algorithm"

# 4. 推送分支
git push origin feature/new-detection-algorithm

# 5. 创建Pull Request
# 在GitHub上创建PR，目标分支为develop
```

### 3. 提交信息规范

使用Conventional Commits规范：

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**类型说明：**
- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式化
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动

**示例：**
```
feat(detection): add real-time object tracking

Implement ByteTrack algorithm for multi-object tracking
with improved performance on edge devices.

Closes #123
```

### 4. Pull Request规范

#### PR模板

```markdown
## 描述
简要描述此PR的目的和内容。

## 变更类型
- [ ] Bug修复
- [ ] 新功能
- [ ] 重构
- [ ] 文档更新
- [ ] 性能优化

## 测试
- [ ] 单元测试通过
- [ ] 集成测试通过
- [ ] 手动测试完成

## 检查清单
- [ ] 代码遵循项目规范
- [ ] 添加了必要的测试
- [ ] 更新了相关文档
- [ ] 通过了所有CI检查

## 相关Issue
Closes #issue_number
```

### 5. 代码审查

#### 审查要点

1. **功能正确性**
   - 代码是否实现了预期功能
   - 边界条件处理是否正确
   - 错误处理是否完善

2. **代码质量**
   - 代码可读性和可维护性
   - 是否遵循项目规范
   - 性能考虑

3. **测试覆盖**
   - 是否有足够的测试
   - 测试是否覆盖主要场景
   - 测试是否可靠

4. **文档完整性**
   - API文档是否更新
   - 代码注释是否清晰
   - 使用示例是否正确

#### 审查流程

```bash
# 1. 检出PR分支
git fetch origin
git checkout feature/new-feature

# 2. 运行测试
pytest

# 3. 代码检查
black --check src/
pylint src/
mypy src/

# 4. 手动测试
python examples/test_new_feature.py
```

## 发布流程

### 1. 版本号规范

使用语义化版本控制 (SemVer)：

- `MAJOR.MINOR.PATCH`
- `MAJOR`: 不兼容的API变更
- `MINOR`: 向后兼容的功能性新增
- `PATCH`: 向后兼容的问题修正

### 2. 发布步骤

```bash
# 1. 创建发布分支
git checkout develop
git pull origin develop
git checkout -b release/v2.1.0

# 2. 更新版本号
# 更新 setup.py, __init__.py 等文件中的版本号

# 3. 更新CHANGELOG
# 记录此版本的主要变更

# 4. 提交变更
git add .
git commit -m "chore: bump version to 2.1.0"

# 5. 合并到main
git checkout main
git merge release/v2.1.0
git tag v2.1.0
git push origin main --tags

# 6. 合并回develop
git checkout develop
git merge release/v2.1.0
git push origin develop

# 7. 删除发布分支
git branch -d release/v2.1.0
```

### 3. 自动化发布

使用GitHub Actions自动化发布流程：

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      
      - name: Install dependencies
        run: |
          pip install build twine
      
      - name: Build package
        run: |
          python -m build
      
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: |
          twine upload dist/*
```

## 性能优化指南

### 1. 代码优化

#### 避免常见性能陷阱

```python
# 不好的做法
def process_images_slow(images):
    results = []
    for image in images:
        # 每次都重新加载模型
        model = load_model()
        result = model.predict(image)
        results.append(result)
    return results

# 好的做法
def process_images_fast(images):
    # 只加载一次模型
    model = load_model()
    results = []
    for image in images:
        result = model.predict(image)
        results.append(result)
    return results

# 更好的做法：使用批处理
def process_images_batch(images, batch_size=4):
    model = load_model()
    results = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch_results = model.predict_batch(batch)
        results.extend(batch_results)
    return results
```

#### 内存优化

```python
import gc
from contextlib import contextmanager

@contextmanager
def memory_efficient_processing():
    """内存高效处理上下文管理器。"""
    try:
        yield
    finally:
        gc.collect()  # 强制垃圾回收

def process_large_dataset(dataset):
    """处理大型数据集。"""
    for batch in dataset.iter_batches(batch_size=32):
        with memory_efficient_processing():
            # 处理批次
            results = process_batch(batch)
            # 立即保存结果，释放内存
            save_results(results)
```

### 2. 性能分析

#### 使用profiler

```python
import cProfile
import pstats
from functools import wraps

def profile_function(func):
    """函数性能分析装饰器。"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(10)  # 显示前10个最耗时的函数
        
        return result
    return wrapper

@profile_function
def expensive_function():
    # 需要分析的函数
    pass
```

#### 内存分析

```python
from memory_profiler import profile

@profile
def memory_intensive_function():
    # 需要分析内存使用的函数
    large_list = [i for i in range(1000000)]
    return sum(large_list)
```

## 调试指南

### 1. 日志记录

```python
import logging
from src.utils.logging_manager import LoggingManager

# 获取logger
logger = LoggingManager("MyModule").get_logger()

def complex_function(data):
    """复杂函数示例。"""
    logger.info(f"开始处理数据，大小: {len(data)}")
    
    try:
        # 处理逻辑
        result = process_data(data)
        logger.debug(f"处理结果: {result}")
        return result
    
    except Exception as e:
        logger.error(f"处理失败: {e}", exc_info=True)
        raise
    
    finally:
        logger.info("函数执行完成")
```

### 2. 断点调试

```python
import pdb

def debug_function(data):
    """需要调试的函数。"""
    # 设置断点
    pdb.set_trace()
    
    # 处理逻辑
    processed = preprocess(data)
    result = model.predict(processed)
    
    return result
```

### 3. 单元测试调试

```python
import pytest

def test_with_debugging():
    """带调试的测试。"""
    data = create_test_data()
    
    # 在测试中使用断点
    import pdb; pdb.set_trace()
    
    result = function_under_test(data)
    assert result is not None
```

## 文档编写

### 1. API文档

使用Sphinx生成API文档：

```bash
# 安装Sphinx
pip install sphinx sphinx-rtd-theme

# 初始化文档
sphinx-quickstart docs

# 生成API文档
sphinx-apidoc -o docs/source src/

# 构建文档
cd docs
make html
```

### 2. 文档结构

```
docs/
├── source/
│   ├── conf.py           # Sphinx配置
│   ├── index.rst         # 主页
│   ├── api/              # API文档
│   ├── tutorials/        # 教程
│   └── examples/         # 示例
├── build/                # 构建输出
└── Makefile             # 构建脚本
```

### 3. 文档规范

#### README文件

```markdown
# 项目名称

简短的项目描述。

## 特性

- 特性1
- 特性2
- 特性3

## 快速开始

### 安装

```bash
pip install project-name
```

### 基本使用

```python
from project import MainClass

# 示例代码
instance = MainClass()
result = instance.method()
```

## 文档

- [API参考](docs/API_REFERENCE.md)
- [开发者指南](docs/DEVELOPER_GUIDE.md)
- [部署指南](docs/DEPLOYMENT.md)

## 贡献

欢迎贡献！请阅读[贡献指南](CONTRIBUTING.md)。

## 许可证

[MIT License](LICENSE)
```

## 常见问题

### Q: 如何添加新的检测模型？

A: 1. 在`src/models/`目录下创建新的模型类
   2. 继承`BaseModel`类
   3. 实现必要的方法
   4. 在模型工厂中注册
   5. 添加相应的测试

### Q: 如何优化推理速度？

A: 1. 使用较小的模型（如yolov8n）
   2. 启用TensorRT优化
   3. 使用批处理
   4. 考虑模型量化

### Q: 如何处理内存不足问题？

A: 1. 减少批处理大小
   2. 使用较小的输入分辨率
   3. 定期进行垃圾回收
   4. 使用内存映射文件

### Q: 如何添加新的应用场景？

A: 1. 在`src/applications/`目录下创建新的应用类
   2. 继承`BaseApplication`类
   3. 实现特定的处理逻辑
   4. 注册到应用管理器
   5. 添加相应的测试和文档

## 联系方式

- **项目主页**: https://github.com/yolos/yolos
- **问题报告**: https://github.com/yolos/yolos/issues
- **讨论区**: https://github.com/yolos/yolos/discussions
- **邮箱**: dev@yolos.org

---

感谢您对YOLOS项目的贡献！