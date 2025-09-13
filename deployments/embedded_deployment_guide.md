# 嵌入式设备轻量级部署指南

## 概述

本指南基于模型性能评估结果，为不同嵌入式平台提供优化的YOLO部署方案。

## 平台部署策略

### 1. ESP32系列

#### ESP32 (基础版)
- **状态**: ❌ 不推荐直接部署YOLO
- **内存**: 32MB | **存储**: 16MB
- **瓶颈**: 所有YOLO模型都超出硬件限制

**替代方案**:
```yaml
# ESP32轻量级检测配置
detection_method: "feature_based"
libraries:
  - "OpenMV"
  - "TensorFlow Lite Micro"
  - "Edge Impulse"

recommended_approach:
  - 使用简单的边缘检测
  - 实现基于颜色/形状的物体识别
  - 边缘-云协同架构
```

#### ESP32-S3
- **状态**: ⚠️ 仅支持超轻量级模型
- **推荐模型**: YOLOv11n (量化后)
- **预期性能**: 20 FPS

**部署配置**:
```yaml
# ESP32-S3部署配置
model:
  name: "yolov11n"
  format: "tflite"
  quantization: "int8"
  input_size: [160, 160]  # 降低输入分辨率

optimizations:
  - 模型剪枝 (50%权重)
  - INT8量化
  - 分阶段推理
  - 动态内存管理

deployment:
  framework: "TensorFlow Lite Micro"
  memory_pool: "20MB"
  inference_threads: 1
```

### 2. 树莓派系列

#### Raspberry Pi Zero 2W
- **状态**: ✅ 支持轻量级模型
- **推荐模型**: YOLOv11n, YOLOv11s
- **最佳性能**: YOLOv11n @ 2 FPS

**部署配置**:
```yaml
# 树莓派Zero部署配置
model:
  primary: "yolov11n"
  backup: "yolov11s"  # 高精度需求时
  format: "onnx"
  quantization: "fp16"

optimizations:
  - ONNX Runtime优化
  - 输入分辨率自适应 (320x320 -> 416x416)
  - 批处理推理 (batch_size=1)
  - CPU多线程优化

deployment:
  runtime: "onnxruntime"
  providers: ["CPUExecutionProvider"]
  memory_limit: "400MB"
  swap_usage: "enabled"
```

#### Raspberry Pi 4B
- **状态**: ✅ 支持多种模型
- **推荐模型**: YOLOv11n (实时), YOLOv11s (平衡), YOLOv11m (高精度)
- **最佳性能**: YOLOv11n @ 10 FPS

**部署配置**:
```yaml
# 树莓派4B部署配置
model:
  real_time: "yolov11n"     # 10 FPS
  balanced: "yolov11s"      # 5 FPS
  high_accuracy: "yolov11m" # 2.5 FPS
  format: "onnx"

optimizations:
  - GPU加速 (VideoCore VI)
  - ONNX Runtime + OpenVINO
  - 动态模型切换
  - 智能帧跳过

deployment:
  runtime: "onnxruntime"
  providers: ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
  gpu_memory: "76MB"  # GPU内存分配
  cpu_threads: 4
```

### 3. 专用AI芯片

#### NVIDIA Jetson Nano
- **状态**: ✅ 全模型支持
- **推荐模型**: YOLOv11s (平衡), YOLOv11m (高性能)
- **最佳性能**: YOLOv11s @ 25 FPS

**部署配置**:
```yaml
# Jetson Nano部署配置
model:
  performance: "yolov11m"   # 8.3 FPS, 高精度
  balanced: "yolov11s"      # 25 FPS, 平衡
  real_time: "yolov11n"     # 50 FPS, 实时
  format: "tensorrt"

optimizations:
  - TensorRT优化
  - FP16精度
  - 动态batch size
  - CUDA加速

deployment:
  runtime: "tensorrt"
  precision: "fp16"
  max_batch_size: 4
  workspace_size: "1GB"
```

#### Canaan K230
- **状态**: ✅ NPU加速支持
- **推荐模型**: YOLOv11n, YOLOv11s
- **最佳性能**: YOLOv11n @ 16.7 FPS (NPU)

**部署配置**:
```yaml
# K230部署配置
model:
  npu_optimized: "yolov11n"
  cpu_fallback: "yolov11n"
  format: "nncase"

optimizations:
  - NPU专用优化
  - INT8量化
  - 算子融合
  - 内存池管理

deployment:
  runtime: "nncase"
  device: "kpu"  # Kendryte Processing Unit
  memory_pool: "400MB"
```

## 轻量级部署包结构

```
embedded_yolo_lite/
├── models/
│   ├── yolov11n_int8.tflite      # ESP32-S3
│   ├── yolov11n_fp16.onnx        # 树莓派
│   ├── yolov11s_fp16.onnx        # 树莓派4B
│   └── yolov11n_tensorrt.engine  # Jetson
├── configs/
│   ├── esp32_s3.yaml
│   ├── raspberry_pi.yaml
│   ├── jetson_nano.yaml
│   └── k230.yaml
├── src/
│   ├── lite_detector.py          # 轻量级检测器
│   ├── memory_manager.py         # 内存管理
│   ├── platform_adapter.py       # 平台适配
│   └── optimization_utils.py     # 优化工具
├── scripts/
│   ├── deploy.sh                 # 部署脚本
│   ├── optimize_model.py         # 模型优化
│   └── benchmark.py              # 性能测试
└── requirements_embedded.txt     # 最小依赖
```

## 内存优化策略

### 1. 动态内存管理
```python
class EmbeddedMemoryManager:
    def __init__(self, platform_spec):
        self.max_memory = platform_spec.memory_mb * 0.8  # 80%可用
        self.model_cache = {}
        
    def load_model_on_demand(self, model_name):
        """按需加载模型，自动释放未使用模型"""
        if self.get_memory_usage() > self.max_memory:
            self.cleanup_unused_models()
            
    def adaptive_batch_size(self, available_memory):
        """根据可用内存动态调整批处理大小"""
        if available_memory < 100:  # MB
            return 1
        elif available_memory < 500:
            return 2
        else:
            return 4
```

### 2. 模型量化配置
```yaml
# 量化策略
quantization:
  esp32_s3:
    method: "int8"
    calibration_dataset: "coco_mini_100"
    accuracy_threshold: 0.85
    
  raspberry_pi:
    method: "fp16"
    dynamic_quantization: true
    
  jetson_nano:
    method: "fp16"
    tensorrt_optimization: true
    calibration_cache: "enabled"
```

## 部署脚本

### 自动部署脚本
```bash
#!/bin/bash
# deploy.sh - 自动化部署脚本

PLATFORM=$1
MODEL_SIZE=$2

case $PLATFORM in
    "esp32_s3")
        echo "部署到ESP32-S3..."
        python optimize_model.py --platform esp32_s3 --model yolov11n --format tflite
        ;;
    "raspberry_pi")
        echo "部署到树莓派..."
        python optimize_model.py --platform raspberry_pi --model $MODEL_SIZE --format onnx
        sudo systemctl enable yolo_detector.service
        ;;
    "jetson_nano")
        echo "部署到Jetson Nano..."
        python optimize_model.py --platform jetson_nano --model $MODEL_SIZE --format tensorrt
        ;;
esac

echo "部署完成！"
```

## 性能监控

### 实时监控指标
```python
class EmbeddedMonitor:
    def __init__(self):
        self.metrics = {
            'fps': 0,
            'memory_usage': 0,
            'cpu_usage': 0,
            'temperature': 0,
            'power_consumption': 0
        }
        
    def adaptive_performance_control(self):
        """根据系统状态自动调整性能"""
        if self.metrics['temperature'] > 70:  # 过热保护
            self.reduce_inference_frequency()
        if self.metrics['memory_usage'] > 0.9:  # 内存不足
            self.trigger_garbage_collection()
```

## 故障排除

### 常见问题解决

1. **内存不足**
   - 减少输入分辨率
   - 启用模型量化
   - 使用更小的模型

2. **推理速度慢**
   - 检查CPU/GPU利用率
   - 优化预处理流程
   - 启用硬件加速

3. **精度下降**
   - 调整量化参数
   - 增加校准数据集
   - 使用混合精度

## 最佳实践

1. **模型选择原则**
   - ESP32系列: 避免使用YOLO，选择轻量级替代方案
   - 树莓派Zero: YOLOv11n，优化输入分辨率
   - 树莓派4B: YOLOv11s，平衡性能和精度
   - Jetson系列: YOLOv11m，充分利用GPU加速

2. **部署优化**
   - 始终进行模型量化
   - 实现动态内存管理
   - 监控系统资源使用
   - 准备降级策略

3. **维护建议**
   - 定期更新模型
   - 监控性能指标
   - 备份配置文件
   - 测试故障恢复

## 结论

通过合理的模型选择、优化策略和部署配置，YOLO可以在大多数嵌入式设备上实现可接受的性能。关键是根据硬件限制选择合适的模型大小和优化方法。