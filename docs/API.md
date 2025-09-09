# YOLOS API 文档

## 概述

YOLOS提供了完整的Python API，支持多种YOLO模型的加载、训练、检测和部署。

## 核心模块

### 1. 模型工厂 (YOLOFactory)

```python
from models.yolo_factory import YOLOFactory

# 创建模型
model = YOLOFactory.create_model(
    model_type='yolov8',
    model_path='yolov8n.pt',
    device='auto'
)

# 获取可用模型
models = YOLOFactory.list_available_models()
```

### 2. 检测器

#### 图像检测器

```python
from detection.image_detector import ImageDetector

detector = ImageDetector(model_type='yolov8')

# 检测单张图像
results = detector.detect_image('image.jpg')

# 批量检测
results = detector.detect_batch(['img1.jpg', 'img2.jpg'])

# 检测目录
results = detector.detect_directory('images/')
```

#### 实时检测器

```python
from detection.realtime_detector import RealtimeDetector

detector = RealtimeDetector(model_type='yolov8')

# 设置回调函数
def on_detection(frame, results):
    print(f"检测到 {len(results)} 个目标")

detector.set_detection_callback(on_detection)

# 启动摄像头检测
detector.start_camera_detection(camera_id=0)

# 启动视频检测
detector.start_video_detection('video.mp4', 'output.mp4')
```

#### 摄像头检测器

```python
from detection.camera_detector import CameraDetector

# USB摄像头
detector = CameraDetector(camera_type='usb')
detector.start_detection()

# 树莓派摄像头
detector = CameraDetector(camera_type='picamera')
detector.start_detection()
```

### 3. 训练器

```python
from training.trainer import YOLOTrainer

trainer = YOLOTrainer(model_type='yolov8', model_size='n')

# 准备数据集配置
dataset_config = {
    'path': './datasets/coco',
    'train': 'train/images',
    'val': 'val/images',
    'nc': 80,
    'names': ['person', 'bicycle', ...]
}

# 开始训练
results = trainer.train(
    dataset_config=dataset_config,
    epochs=100,
    batch_size=16
)

# 验证模型
val_results = trainer.validate(dataset_config)

# 导出模型
trainer.export_model('model.pt', 'onnx')
```

### 4. 通信模块

#### MQTT客户端

```python
from communication.mqtt_client import MQTTClient

# 创建MQTT客户端
mqtt = MQTTClient(
    broker_host='localhost',
    broker_port=1883
)

# 连接
mqtt.connect()

# 发布检测结果
mqtt.publish_detection_result(results, image_info)

# 订阅命令
def handle_command(topic, message):
    print(f"收到命令: {message}")

mqtt.subscribe('yolos/commands', handle_command)
```

## 配置系统

### 默认配置

```python
from utils.config_manager import ConfigManager

config = ConfigManager('configs/default_config.yaml')

# 获取配置
model_type = config.get('model.type', 'yolov8')
confidence = config.get('detection.confidence_threshold', 0.25)

# 设置配置
config.set('model.device', 'cuda')
config.save()
```

### 配置文件结构

```yaml
# 模型配置
model:
  type: "yolov8"
  size: "n"
  device: "auto"
  confidence_threshold: 0.25
  iou_threshold: 0.7

# 检测配置
detection:
  input_size: [640, 640]
  max_detections: 100
  save_results: true

# 摄像头配置
camera:
  type: "usb"
  device_id: 0
  resolution: [640, 480]
  framerate: 30

# MQTT配置
mqtt:
  enabled: false
  broker_host: "localhost"
  broker_port: 1883
```

## 数据格式

### 检测结果格式

```python
{
    'bbox': [x1, y1, x2, y2],      # 边界框坐标
    'confidence': 0.85,             # 置信度
    'class_id': 0,                  # 类别ID
    'class_name': 'person'          # 类别名称
}
```

### 图像信息格式

```python
{
    'width': 640,
    'height': 480,
    'channels': 3,
    'source': 'camera',
    'timestamp': '2024-01-01T12:00:00'
}
```

## 错误处理

```python
try:
    results = detector.detect_image('image.jpg')
except FileNotFoundError:
    print("图像文件不存在")
except RuntimeError as e:
    print(f"检测失败: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```

## 性能优化

### GPU加速

```python
# 使用GPU
detector = ImageDetector(device='cuda')

# 批量处理
results = detector.detect_batch(image_paths, batch_size=8)
```

### 模型优化

```python
# 导出ONNX模型
model.export('model.onnx', format='onnx')

# 使用TensorRT
model.export('model.engine', format='tensorrt')
```

## 扩展开发

### 自定义模型

```python
from models.base_model import BaseYOLOModel

class CustomYOLOModel(BaseYOLOModel):
    def load_model(self, model_path):
        # 实现模型加载逻辑
        pass
    
    def predict(self, image, **kwargs):
        # 实现预测逻辑
        pass

# 注册到工厂
YOLOFactory._models['custom'] = CustomYOLOModel
```

### 自定义检测器

```python
from detection.realtime_detector import RealtimeDetector

class CustomDetector(RealtimeDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 自定义初始化
    
    def process_results(self, results):
        # 自定义结果处理
        return results
```

## 示例代码

完整的示例代码请参考 `examples/` 目录：

- `basic_detection.py` - 基础检测示例
- `realtime_detection.py` - 实时检测示例
- `mqtt_detection.py` - MQTT通信示例
- `training_example.py` - 训练示例
- `ros_example.py` - ROS集成示例