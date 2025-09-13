# YOLOS API 参考文档

## 概述

YOLOS (You Only Look Once System) 提供了一套完整的计算机视觉API，支持多种检测和识别任务。本文档详细描述了所有可用的API接口、参数和返回值。

## 版本信息

- **API版本**: v2.0
- **文档版本**: 2024.1
- **兼容性**: Python 3.8+
- **支持平台**: Windows, Linux, macOS, AIoT设备

## 快速开始

### 安装

```bash
pip install yolos
```

### 基本使用

```python
from yolos import YOLOS

# 初始化YOLOS实例
yolos = YOLOS()

# 加载图像并进行检测
result = yolos.detect('path/to/image.jpg')
print(result)

# 批量检测示例
from detection.batch_detector import BatchDetector

detector = BatchDetector(model_type='yolov8n')
results = detector.detect_batch(['img1.jpg', 'img2.jpg'])

# 实时检测示例
from detection.realtime_detector import RealtimeDetector

detector = RealtimeDetector()
detector.set_detection_callback(lambda frame, results: print(f"检测到 {len(results)} 个目标"))
detector.start_camera_detection(camera_id=0)
```

## 核心API

### YOLOS 主类

#### `class YOLOS`

主要的YOLOS接口类，提供统一的检测和识别功能。

##### 构造函数

```python
YOLOS(model_type='yolov8n', device='auto', config=None)
```

**参数:**
- `model_type` (str, 可选): 模型类型，默认为'yolov8n'
  - 可选值: 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'
- `device` (str, 可选): 运行设备，默认为'auto'
  - 可选值: 'auto', 'cpu', 'cuda', 'mps'
- `config` (dict, 可选): 自定义配置字典

**返回:**
- `YOLOS`: YOLOS实例

**异常:**
- `ModelLoadError`: 模型加载失败
- `DeviceError`: 设备不可用

##### 方法

#### `detect(source, **kwargs)`

执行目标检测。

```python
result = yolos.detect(
    source='image.jpg',
    conf_threshold=0.5,
    iou_threshold=0.45,
    max_detections=1000,
    classes=None,
    save_results=False
)
```

**参数:**
- `source` (str|np.ndarray|PIL.Image): 输入源
  - 支持: 图像路径、numpy数组、PIL图像、视频路径、摄像头索引
- `conf_threshold` (float, 可选): 置信度阈值，默认0.5
- `iou_threshold` (float, 可选): IoU阈值，默认0.45
- `max_detections` (int, 可选): 最大检测数量，默认1000
- `classes` (list, 可选): 指定检测类别，默认None（所有类别）
- `save_results` (bool, 可选): 是否保存结果，默认False

**返回:**
```python
{
    'detections': [
        {
            'class_id': int,
            'class_name': str,
            'confidence': float,
            'bbox': {
                'x1': float, 'y1': float,
                'x2': float, 'y2': float,
                'width': float, 'height': float
            }
        }
    ],
    'image_info': {
        'width': int,
        'height': int,
        'channels': int
    },
    'processing_time': float,
    'model_info': {
        'name': str,
        'version': str
    }
}
```

**异常:**
- `InputError`: 输入格式错误
- `ProcessingError`: 处理过程出错

#### `detect_batch(sources, **kwargs)`

批量检测多个输入。

```python
results = yolos.detect_batch(
    sources=['img1.jpg', 'img2.jpg'],
    batch_size=4,
    **kwargs
)
```

**参数:**
- `sources` (list): 输入源列表
- `batch_size` (int, 可选): 批处理大小，默认4
- `**kwargs`: 其他参数同`detect()`方法

**返回:**
- `list`: 检测结果列表，每个元素格式同`detect()`返回值

#### `track(source, **kwargs)`

目标跟踪（适用于视频输入）。

```python
for frame_result in yolos.track('video.mp4'):
    print(frame_result)
```

**参数:**
- `source` (str): 视频路径或摄像头索引
- `tracker_type` (str, 可选): 跟踪器类型，默认'bytetrack'
- `**kwargs`: 其他参数同`detect()`方法

**返回:**
- `generator`: 生成器，每次迭代返回一帧的跟踪结果

**帧结果格式:**
```python
{
    'frame_id': int,
    'timestamp': float,
    'tracks': [
        {
            'track_id': int,
            'class_id': int,
            'class_name': str,
            'confidence': float,
            'bbox': {...},  # 同detect()格式
            'velocity': {'vx': float, 'vy': float}
        }
    ]
}
```

## 应用专用API

### 人脸识别

#### `class FaceRecognition`

```python
from yolos.applications import FaceRecognition

face_rec = FaceRecognition()
result = face_rec.detect_faces('image.jpg')
```

#### `detect_faces(image, **kwargs)`

**返回格式:**
```python
{
    'faces': [
        {
            'bbox': {...},
            'confidence': float,
            'landmarks': {
                'left_eye': [x, y],
                'right_eye': [x, y],
                'nose': [x, y],
                'left_mouth': [x, y],
                'right_mouth': [x, y]
            },
            'attributes': {
                'age': int,
                'gender': str,
                'emotion': str
            }
        }
    ]
}
```

### 姿态估计

#### `class PoseEstimation`

```python
from yolos.applications import PoseEstimation

pose_est = PoseEstimation()
result = pose_est.estimate_pose('image.jpg')
```

#### `estimate_pose(image, **kwargs)`

**返回格式:**
```python
{
    'poses': [
        {
            'bbox': {...},
            'confidence': float,
            'keypoints': [
                {
                    'name': str,
                    'x': float,
                    'y': float,
                    'confidence': float,
                    'visible': bool
                }
            ],
            'skeleton': [
                {'from': int, 'to': int}  # 关键点连接
            ]
        }
    ]
}
```

### 物体检测

#### `class ObjectDetection`

通用物体检测，支持COCO数据集的80个类别。

```python
from yolos.applications import ObjectDetection

obj_det = ObjectDetection()
result = obj_det.detect_objects('image.jpg')
```

### 宠物检测

#### `class PetDetection`

专门针对宠物优化的检测模型。

```python
from yolos.applications import PetDetection

pet_det = PetDetection()
result = pet_det.detect_pets('image.jpg')
```

**支持的宠物类别:**
- 狗 (dog)
- 猫 (cat)
- 鸟 (bird)
- 鱼 (fish)
- 兔子 (rabbit)
- 仓鼠 (hamster)

### 植物识别

#### `class PlantRecognition`

```python
from yolos.applications import PlantRecognition

plant_rec = PlantRecognition()
result = plant_rec.recognize_plants('image.jpg')
```

**返回格式:**
```python
{
    'plants': [
        {
            'bbox': {...},
            'species': str,
            'common_name': str,
            'confidence': float,
            'health_status': str,
            'care_tips': [str]
        }
    ]
}
```

## 配置API

### 配置管理

#### `class ConfigManager`

```python
from yolos.core import ConfigManager

config = ConfigManager()
config.load_config('config.yaml')
```

#### 配置文件格式

```yaml
# config.yaml
model:
  type: "yolov8n"
  device: "auto"
  precision: "fp16"

detection:
  conf_threshold: 0.5
  iou_threshold: 0.45
  max_detections: 1000

performance:
  batch_size: 4
  num_workers: 4
  use_tensorrt: false

logging:
  level: "INFO"
  file: "yolos.log"
```

## 工具API

### 图像处理

#### `yolos.utils.image`

```python
from yolos.utils.image import (
    load_image, save_image, resize_image,
    draw_detections, create_mosaic
)

# 加载图像
image = load_image('path/to/image.jpg')

# 调整大小
resized = resize_image(image, (640, 640))

# 绘制检测结果
drawn = draw_detections(image, detections)

# 保存图像
save_image(drawn, 'output.jpg')
```

### 视频处理

#### `yolos.utils.video`

```python
from yolos.utils.video import (
    VideoProcessor, extract_frames, create_video
)

# 视频处理器
processor = VideoProcessor('input.mp4')
for frame in processor:
    # 处理每一帧
    result = yolos.detect(frame)
    processed_frame = draw_detections(frame, result['detections'])
    processor.write_frame(processed_frame)
```

### 数据转换

#### `yolos.utils.convert`

```python
from yolos.utils.convert import (
    coco_to_yolo, yolo_to_coco,
    export_to_json, export_to_xml
)

# 格式转换
yolo_format = coco_to_yolo(coco_annotations)
coco_format = yolo_to_coco(yolo_annotations)

# 导出结果
export_to_json(results, 'output.json')
export_to_xml(results, 'output.xml')
```

## 性能优化API

### TensorRT优化

```python
from yolos.optimization import TensorRTOptimizer

# 创建TensorRT优化器
optimizer = TensorRTOptimizer()

# 优化模型
optimized_model = optimizer.optimize(
    model_path='yolov8n.pt',
    precision='fp16',
    workspace_size=1024  # MB
)

# 使用优化后的模型
yolos = YOLOS(model=optimized_model)
```

### 量化

```python
from yolos.optimization import Quantizer

quantizer = Quantizer()
quantized_model = quantizer.quantize(
    model_path='yolov8n.pt',
    calibration_data='calibration_images/',
    method='int8'
)
```

## 错误处理

### 异常类型

```python
from yolos.exceptions import (
    YOLOSError,           # 基础异常
    ModelLoadError,       # 模型加载错误
    InputError,           # 输入错误
    ProcessingError,      # 处理错误
    DeviceError,          # 设备错误
    ConfigurationError    # 配置错误
)

try:
    yolos = YOLOS(model_type='invalid_model')
except ModelLoadError as e:
    print(f"模型加载失败: {e}")
except DeviceError as e:
    print(f"设备错误: {e}")
```

### 错误码

| 错误码 | 描述 | 解决方案 |
|--------|------|----------|
| E001 | 模型文件不存在 | 检查模型路径 |
| E002 | 设备不可用 | 检查CUDA/设备状态 |
| E003 | 输入格式错误 | 检查输入数据格式 |
| E004 | 内存不足 | 减少批处理大小 |
| E005 | 配置文件错误 | 检查配置文件格式 |

## 回调和事件

### 事件系统

```python
from yolos.events import EventBus

# 创建事件总线
event_bus = EventBus()

# 注册事件处理器
@event_bus.on('detection_complete')
def on_detection_complete(event):
    print(f"检测完成: {event.data}")

# 触发事件
event_bus.emit('detection_complete', {
    'detections': results,
    'processing_time': 0.1
})
```

### 进度回调

```python
def progress_callback(current, total, message):
    print(f"进度: {current}/{total} - {message}")

# 批量处理时使用回调
results = yolos.detect_batch(
    sources=image_list,
    progress_callback=progress_callback
)
```

## 部署API

### Web服务

```python
from yolos.deployment import WebService

# 创建Web服务
service = WebService(yolos_instance=yolos)

# 启动服务
service.run(host='0.0.0.0', port=8080)
```

**REST API端点:**

- `POST /detect` - 单张图像检测
- `POST /detect/batch` - 批量检测
- `POST /track` - 视频跟踪
- `GET /models` - 获取可用模型列表
- `GET /health` - 健康检查

### 边缘设备部署

```python
from yolos.deployment import EdgeDeployment

# 边缘设备部署
edge = EdgeDeployment(
    platform='raspberry_pi',  # 或 'jetson', 'esp32'
    optimization_level='high'
)

optimized_yolos = edge.optimize(yolos)
```

## 最佳实践

### 性能优化建议

1. **选择合适的模型大小**
   ```python
   # 高精度场景
   yolos = YOLOS(model_type='yolov8x')
   
   # 实时场景
   yolos = YOLOS(model_type='yolov8n')
   ```

2. **批处理优化**
   ```python
   # 推荐的批处理大小
   batch_sizes = {
       'cpu': 1,
       'gpu_4gb': 4,
       'gpu_8gb': 8,
       'gpu_16gb': 16
   }
   ```

3. **内存管理**
   ```python
   # 处理大量图像时定期清理
   import gc
   
   for i, image in enumerate(images):
       result = yolos.detect(image)
       # 处理结果...
       
       if i % 100 == 0:
           gc.collect()  # 垃圾回收
   ```

### 错误处理模式

```python
from yolos.exceptions import YOLOSError
import logging

logger = logging.getLogger(__name__)

def safe_detect(yolos, image_path):
    try:
        return yolos.detect(image_path)
    except YOLOSError as e:
        logger.error(f"检测失败: {e}")
        return None
    except Exception as e:
        logger.exception(f"未知错误: {e}")
        return None
```

## 版本兼容性

### API版本历史

- **v2.0** (当前)
  - 新增批处理API
  - 改进错误处理
  - 添加事件系统

- **v1.5**
  - 添加跟踪功能
  - 性能优化

- **v1.0**
  - 基础检测功能
  - 核心API

### 迁移指南

从v1.x迁移到v2.0:

```python
# v1.x
result = yolos.predict(image)

# v2.0
result = yolos.detect(image)
```

## 许可证和支持

- **许可证**: MIT License
- **文档**: https://yolos.readthedocs.io
- **GitHub**: https://github.com/yolos/yolos
- **问题报告**: https://github.com/yolos/yolos/issues
- **社区**: https://discord.gg/yolos

---

*最后更新: 2024年1月*