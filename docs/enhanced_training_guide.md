# 增强人体识别训练指南

## 概述

本指南介绍如何使用预训练模型和深度学习技术来提升多模态人体识别系统的准确性。相比传统的规则和阈值判断，新系统通过以下方式实现显著改进：

1. **预训练模型集成** - 使用ResNet50、MobileNetV3、EfficientNet等预训练backbone
2. **多尺度特征提取** - 从不同尺度提取图像特征并融合
3. **深度学习分类器** - 基于大规模数据集训练的动作识别模型
4. **多模态特征融合** - 结合图像特征和姿势关键点信息

## 系统架构

```
输入图像 → 预训练特征提取器 → 多模态融合 → 动作分类器 → 识别结果
    ↓              ↓                ↓           ↓
姿势关键点 → 关键点处理器 ────────┘     置信度评估
```

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install torch torchvision opencv-python numpy scikit-learn
pip install albumentations tqdm requests
pip install mediapipe insightface

# 创建必要目录
mkdir -p datasets models/pretrained models/human_recognition
```

### 2. 下载训练数据集

```bash
# 下载公开数据集
python scripts/download_training_datasets.py --all

# 或者创建合成数据集用于快速测试
python scripts/download_training_datasets.py --synthetic
```

### 3. 下载预训练模型

```bash
# 下载所有预训练模型
python -c "
from src.models.pretrained_model_loader import PretrainedModelLoader
loader = PretrainedModelLoader()
loader.download_all_models()
"
```

### 4. 训练自定义模型

```python
from src.training.enhanced_human_trainer import EnhancedHumanTrainer, TrainingConfig

# 配置训练参数
config = TrainingConfig(
    batch_size=16,
    learning_rate=0.001,
    epochs=50,
    validation_split=0.2
)

# 创建训练器
trainer = EnhancedHumanTrainer(config)

# 数据集配置
dataset_configs = [
    {
        'name': 'synthetic_human_actions',
        'path': './datasets/synthetic_human_actions',
        'format': 'custom',
        'description': '合成人体动作数据集'
    }
]

# 开始训练
model_path = trainer.train(dataset_configs)
print(f"训练完成，模型保存在: {model_path}")
```

### 5. 使用改进的识别系统

```python
from src.recognition.improved_multimodal_detector import create_improved_multimodal_system
import cv2

# 创建改进的识别系统
detector = create_improved_multimodal_system({
    'face_database_path': './data/face_database.pkl',
    'use_pretrained_models': True,
    'detection_interval': 1
})

# 使用摄像头进行实时识别
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 多模态检测
    result = detector.detect_multimodal(frame)
    
    # 显示结果
    print(f"检测到: 面部{len(result.faces)}, 手势{len(result.gestures)}, "
          f"姿势{len(result.poses)}, 动作{len(result.actions)}")
    
    # 显示置信度
    for action in result.actions:
        print(f"动作: {action.metadata['action_name']}, "
              f"置信度: {action.confidence:.3f}")
    
    cv2.imshow('Enhanced Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 数据集准备

### 支持的数据集格式

1. **COCO格式**
```json
{
  "images": [{"id": 1, "file_name": "image1.jpg"}],
  "annotations": [
    {
      "image_id": 1,
      "category_id": 1,
      "keypoints": [x1, y1, v1, x2, y2, v2, ...],
      "bbox": [x, y, width, height]
    }
  ]
}
```

2. **自定义格式**
```json
[
  {
    "image_path": "images/action1.jpg",
    "action_label": 0,
    "action_name": "walking",
    "pose_keypoints": [x1, y1, v1, ...],
    "gesture_label": 2
  }
]
```

### 创建自定义数据集

```python
import json
import cv2
import numpy as np
from pathlib import Path

def create_custom_dataset(images_dir, output_dir):
    """创建自定义数据集标注"""
    annotations = []
    
    for img_path in Path(images_dir).glob("*.jpg"):
        # 从文件名或文件夹推断动作类别
        action_name = img_path.parent.name
        action_label = get_action_id(action_name)
        
        annotation = {
            'image_path': str(img_path.relative_to(output_dir)),
            'action_label': action_label,
            'action_name': action_name,
            'pose_keypoints': [],  # 可以使用MediaPipe提取
            'gesture_label': -1
        }
        annotations.append(annotation)
    
    # 保存标注文件
    with open(Path(output_dir) / 'annotations.json', 'w') as f:
        json.dump(annotations, f, indent=2)
```

## 模型训练详解

### 1. 数据增强策略

系统使用Albumentations库进行数据增强：

```python
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),           # 水平翻转
    A.RandomBrightnessContrast(p=0.3), # 亮度对比度调整
    A.GaussNoise(p=0.2),               # 高斯噪声
    A.Blur(blur_limit=3, p=0.2),       # 模糊
    A.RandomRotate90(p=0.2),           # 随机旋转
    A.ShiftScaleRotate(                # 平移缩放旋转
        shift_limit=0.1,
        scale_limit=0.1,
        rotate_limit=15,
        p=0.3
    )
])
```

### 2. 多模态网络架构

```python
class MultiModalHumanNet(nn.Module):
    def __init__(self, num_action_classes=10, num_gesture_classes=8):
        super().__init__()
        
        # 图像特征提取器 (ResNet-like)
        self.image_backbone = self._build_backbone()
        
        # 姿势关键点处理器
        self.pose_processor = nn.Sequential(
            nn.Linear(34, 128),  # 17个关键点 * 2坐标
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        
        # 多模态融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(512 + 256, 512),  # 图像 + 姿势特征
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256)
        )
        
        # 分类头
        self.action_classifier = nn.Linear(256, num_action_classes)
        self.gesture_classifier = nn.Linear(256, num_gesture_classes)
```

### 3. 训练策略

- **学习率调度**: ReduceLROnPlateau，验证损失不下降时减少学习率
- **早停机制**: 验证损失连续15个epoch不改善时停止训练
- **模型保存**: 每10个epoch保存检查点，保存最佳验证性能模型
- **损失函数**: 动作分类损失 + 0.5 * 手势分类损失

## 性能优化

### 1. 特征提取优化

```python
# 多尺度特征提取
scales = [(224, 224), (256, 256), (192, 192)]
features = {}

for scale in scales:
    resized_image = cv2.resize(image, scale)
    feature = extract_features(model, resized_image)
    features[f"scale_{scale[0]}"] = feature

# 特征融合
fused_feature = np.concatenate([
    normalize_feature(f) for f in features.values()
])
```

### 2. 推理加速

```python
# 模型量化
model_int8 = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# 批处理推理
batch_size = 4
images_batch = torch.stack([preprocess(img) for img in images])
with torch.no_grad():
    predictions = model(images_batch)
```

### 3. 内存优化

```python
# 梯度累积
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 评估指标

### 1. 分类准确率
```python
def calculate_accuracy(predictions, labels):
    correct = (predictions.argmax(dim=1) == labels).float()
    return correct.mean().item()
```

### 2. Top-K准确率
```python
def top_k_accuracy(predictions, labels, k=3):
    _, top_k_pred = predictions.topk(k, dim=1)
    correct = top_k_pred.eq(labels.view(-1, 1).expand_as(top_k_pred))
    return correct.any(dim=1).float().mean().item()
```

### 3. 混淆矩阵
```python
from sklearn.metrics import confusion_matrix, classification_report

y_true = labels.cpu().numpy()
y_pred = predictions.argmax(dim=1).cpu().numpy()

cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=class_names)
```

## 部署建议

### 1. 模型导出
```python
# 导出ONNX格式
torch.onnx.export(
    model,
    dummy_input,
    "human_recognition_model.onnx",
    export_params=True,
    opset_version=11,
    input_names=['image', 'pose_keypoints'],
    output_names=['action_pred', 'gesture_pred']
)
```

### 2. 推理优化
```python
# 使用TensorRT加速（NVIDIA GPU）
import tensorrt as trt

# 或使用OpenVINO（Intel CPU）
from openvino.inference_engine import IECore
```

### 3. 边缘设备部署
```python
# 使用量化模型
model_quantized = torch.quantization.quantize_dynamic(
    model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
)

# 或使用TensorFlow Lite
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少batch_size
   - 使用梯度累积
   - 启用混合精度训练

2. **训练收敛慢**
   - 调整学习率
   - 使用预训练权重
   - 检查数据质量

3. **过拟合**
   - 增加数据增强
   - 使用更多Dropout
   - 减少模型复杂度

4. **推理速度慢**
   - 使用模型量化
   - 批处理推理
   - 使用专用推理引擎

### 调试技巧

```python
# 可视化特征
import matplotlib.pyplot as plt

def visualize_features(features, title="Features"):
    plt.figure(figsize=(12, 8))
    plt.imshow(features.reshape(-1, 1), aspect='auto', cmap='viridis')
    plt.title(title)
    plt.colorbar()
    plt.show()

# 监控训练过程
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('Loss')
    ax1.legend()
    
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_title('Accuracy')
    ax2.legend()
    
    plt.show()
```

## 总结

通过集成预训练模型和深度学习技术，新的多模态人体识别系统相比传统方法具有以下优势：

1. **更高准确率** - 基于大规模数据集训练的特征提取器
2. **更强泛化能力** - 多尺度特征融合和数据增强
3. **更智能的决策** - 深度学习模型替代规则判断
4. **持续学习能力** - 支持增量训练和模型更新

建议在实际部署前，使用您的特定数据集进行微调训练，以获得最佳性能。