# YOLOS AIoT开发板部署指南

## 概述

YOLOS系统现已全面支持主流AIoT开发板，提供从高性能到超低功耗的完整解决方案。本指南将帮助您在各种AIoT开发板上成功部署YOLOS识别系统。

## 支持的AIoT开发板

### 🚀 高性能AI开发板

#### NVIDIA Jetson系列
- **Jetson AGX Xavier** - 8核ARM + 512 CUDA核心 + DLA
- **Jetson Xavier NX** - 6核ARM + 384 CUDA核心 + DLA  
- **Jetson Orin Nano** - 6核ARM + 1024 CUDA核心 + DLA + PVA
- **Jetson Nano** - 4核ARM + 128 CUDA核心

**特点**: CUDA加速、TensorRT优化、高性能计算
**适用场景**: 实时视频分析、复杂AI模型推理

#### Qualcomm RB5平台
- **CPU**: Snapdragon 865 (Kryo 585)
- **AI加速器**: Hexagon 698 DSP (15 TOPS)
- **特点**: 5G支持、移动端优化、SNPE运行时

### 🎯 专用AI加速开发板

#### Google Coral系列
- **Coral Dev Board** - Edge TPU专用AI加速
- **Coral Dev Board Micro** - 超低功耗Edge TPU

**特点**: Edge TPU加速、TensorFlow Lite优化、低延迟推理
**适用场景**: 边缘AI推理、实时目标检测

#### Intel系列
- **Intel NUC** - x86架构 + Intel GPU/Movidius
- **Neural Compute Stick 2** - USB AI加速棒

**特点**: OpenVINO优化、x86兼容性、便携式AI加速

### 🔧 通用高性能开发板

#### Rockchip系列
- **RK3588** - 8核ARM + Mali GPU + 6 TOPS NPU
- **RK3566** - 4核ARM + Mali GPU + 0.8 TOPS NPU

**特点**: NPU加速、RKNN工具链、成本效益高

#### MediaTek Genio平台
- **CPU**: ARM Cortex-A78 + A55
- **AI加速器**: APU 3.0 (4 TOPS)
- **特点**: WiFi6支持、APU加速

#### Amlogic A311D
- **CPU**: ARM Cortex-A73 + A53
- **AI加速器**: NPU 5.0 TOPS
- **特点**: 成本效益、Android支持

### 🌱 入门级开发板

#### 树莓派系列
- **Raspberry Pi 5** - 4核ARM Cortex-A76
- **Raspberry Pi 4** - 4核ARM Cortex-A72

**特点**: 社区支持丰富、教育友好、GPIO丰富

#### ESP32系列
- **ESP32-S3** - 支持TensorFlow Lite Micro
- **ESP32-CAM** - 内置摄像头、超低成本

**特点**: 超低功耗、WiFi/蓝牙、IoT专用

#### STM32系列
- **STM32H7** - ARM Cortex-M7 (550MHz) + CMSIS-NN
- **STM32F7** - ARM Cortex-M7 (216MHz) + Chrom-ART
- **STM32F4** - ARM Cortex-M4 (180MHz) + DSP
- **STM32L4** - 超低功耗ARM Cortex-M4
- **STM32MP1** - ARM Cortex-A7 + M4双核 + Linux支持

**特点**: 实时性能、工业级可靠性、STM32Cube.AI生态系统
**适用场景**: 工业IoT、实时AI推理、边缘计算、传感器融合

## 快速部署

### 1. 自动检测和配置

```bash
# 克隆项目
git clone https://github.com/your-repo/yolos.git
cd yolos

# 运行AIoT兼容性检测
python tests/test_aiot_compatibility.py

# 查看检测报告
python -c "
from src.plugins.platform.aiot_boards_adapter import get_aiot_boards_adapter
adapter = get_aiot_boards_adapter()
print(adapter.generate_board_report())
"
```

### 2. 安装依赖

```bash
# 基础依赖
pip install -r requirements.txt

# 自动安装平台特定依赖
python -c "
from src.core.cross_platform_manager import get_cross_platform_manager
manager = get_cross_platform_manager()
results = manager.install_platform_dependencies()
print('依赖安装结果:', results)
"
```

### 3. 启动识别系统

```python
from src.recognition.hybrid_recognition_system import HybridRecognitionSystem
from src.core.cross_platform_manager import get_cross_platform_manager

# 获取平台优化配置
manager = get_cross_platform_manager()
config = manager.get_recommended_settings()

# 初始化识别系统
recognition_system = HybridRecognitionSystem(config)

# 开始识别
recognition_system.start_recognition()
```

## 平台特定优化

### NVIDIA Jetson优化

```python
# Jetson专用配置
jetson_config = {
    'use_tensorrt': True,
    'enable_dla': True,  # 深度学习加速器
    'cuda_optimization': True,
    'memory_fraction': 0.8,
    'max_batch_size': 4
}

# 启用TensorRT优化
recognition_system.enable_tensorrt_optimization(jetson_config)
```

### Google Coral优化

```python
# Coral Edge TPU配置
coral_config = {
    'use_edge_tpu': True,
    'tpu_model_path': 'models/coral_optimized_model.tflite',
    'enable_pycoral': True,
    'inference_threads': 1
}

# 启用Edge TPU加速
recognition_system.enable_edge_tpu_acceleration(coral_config)
```

### Rockchip NPU优化

```python
# Rockchip NPU配置
rockchip_config = {
    'use_rknn': True,
    'npu_model_path': 'models/rknn_optimized_model.rknn',
    'npu_core_mask': 0x7,  # 使用所有NPU核心
    'enable_zero_copy': True
}

# 启用NPU加速
recognition_system.enable_npu_acceleration(rockchip_config)
```

### ESP32优化

```python
# ESP32超低功耗配置
esp32_config = {
    'image_size': (160, 120),
    'recognition_interval': 5.0,  # 5秒间隔
    'enable_deep_sleep': True,
    'wifi_power_save': True,
    'simple_models_only': True
}

# 启用低功耗模式
recognition_system.enable_low_power_mode(esp32_config)
```

### STM32优化

```python
# STM32实时AI配置
stm32_config = {
    'use_cmsis_nn': True,
    'enable_dsp_acceleration': True,
    'model_path': 'models/stm32_optimized_model.tflite',
    'quantization': 'int8',
    'memory_optimization': True,
    'real_time_priority': True
}

# 启用STM32 Cube.AI优化
recognition_system.enable_stm32_optimization(stm32_config)

# STM32MP1 Linux配置
stm32mp1_config = {
    'use_gpu_acceleration': True,
    'enable_heterogeneous_computing': True,  # A7+M4双核
    'cortex_a7_tasks': ['preprocessing', 'postprocessing'],
    'cortex_m4_tasks': ['real_time_inference', 'sensor_fusion'],
    'shared_memory_size': '64MB'
}

# 启用异构计算
recognition_system.enable_heterogeneous_computing(stm32mp1_config)
```

## 性能基准测试

### 推理性能对比

| 开发板 | AI加速器 | FPS (YOLO) | 延迟 (ms) | 功耗 (W) |
|--------|----------|------------|-----------|----------|
| Jetson AGX Xavier | GPU+DLA | 60+ | <20 | 15-30 |
| Jetson Orin Nano | GPU+DLA+PVA | 45+ | <25 | 7-15 |
| Coral Dev Board | Edge TPU | 30+ | <35 | 2-4 |
| RK3588 | NPU | 25+ | <40 | 5-10 |
| Raspberry Pi 5 | CPU | 10+ | <100 | 3-5 |
| ESP32-S3 | CPU | 1-2 | <500 | <1 |
| STM32H7 | CMSIS-NN | 5-10 | <200 | <2 |
| STM32MP1 | GPU+CMSIS | 15-20 | <100 | 3-5 |

### 内存使用对比

| 开发板 | 系统内存 | YOLOS占用 | 可用内存 |
|--------|----------|-----------|----------|
| Jetson AGX Xavier | 32GB | 2-4GB | 充足 |
| Jetson Orin Nano | 8GB | 1-2GB | 充足 |
| Coral Dev Board | 1GB | 200-400MB | 紧张 |
| RK3588 | 8GB | 500MB-1GB | 充足 |
| Raspberry Pi 5 | 8GB | 300-600MB | 充足 |
| ESP32-S3 | 8MB | 2-4MB | 极紧张 |
| STM32H7 | 1MB | 100-300KB | 紧张 |
| STM32MP1 | 512MB | 50-100MB | 充足 |

## 部署最佳实践

### 1. 模型选择策略

```python
def select_optimal_model(board_info):
    """根据开发板选择最优模型"""
    
    if board_info['ai_accelerator']:
        if 'edge_tpu' in board_info['capabilities']:
            return 'yolov5s_edgetpu.tflite'
        elif 'cuda' in board_info['capabilities']:
            return 'yolov5s_tensorrt.engine'
        elif 'npu_acceleration' in board_info['capabilities']:
            return 'yolov5s_rknn.rknn'
    
    # 根据内存选择模型大小
    memory_gb = board_info['memory_gb']
    if memory_gb >= 4:
        return 'yolov5s.onnx'
    elif memory_gb >= 1:
        return 'yolov5n.onnx'
    else:
        return 'yolov5n_quantized.tflite'
```

### 2. 动态资源管理

```python
class AIoTResourceManager:
    """AIoT开发板资源管理器"""
    
    def __init__(self, board_config):
        self.board_config = board_config
        self.thermal_monitor = ThermalMonitor()
        self.power_monitor = PowerMonitor()
    
    def adaptive_performance_scaling(self):
        """自适应性能调节"""
        
        # 温度管理
        if self.thermal_monitor.get_temperature() > 70:
            self.reduce_inference_frequency()
            self.lower_cpu_frequency()
        
        # 功耗管理
        if self.power_monitor.get_power() > self.board_config['max_power']:
            self.enable_power_save_mode()
        
        # 内存管理
        if self.get_memory_usage() > 0.8:
            self.clear_model_cache()
            self.reduce_batch_size()
```

### 3. 网络连接管理

```python
class NetworkManager:
    """网络连接管理"""
    
    def __init__(self):
        self.offline_mode = True
        self.sync_queue = []
    
    def handle_weak_network(self):
        """处理弱网络环境"""
        
        if not self.is_network_stable():
            # 切换到离线模式
            self.offline_mode = True
            self.use_cached_models()
        else:
            # 同步离线数据
            self.sync_offline_data()
            self.update_models_if_needed()
```

## 故障排除

### 常见问题

#### 1. 开发板检测失败
```bash
# 检查系统信息
cat /proc/cpuinfo
cat /proc/device-tree/model

# 手动指定开发板类型
export YOLOS_BOARD_TYPE="jetson_nano"
```

#### 2. AI加速器不可用
```bash
# 检查CUDA支持
nvidia-smi

# 检查Edge TPU
lsusb | grep -i coral

# 检查NPU支持
dmesg | grep -i npu
```

#### 3. 内存不足
```python
# 启用内存优化模式
config = {
    'enable_memory_optimization': True,
    'use_lightweight_models': True,
    'reduce_image_size': True,
    'enable_model_quantization': True
}
```

#### 4. 性能不达预期
```python
# 性能调优
config = {
    'enable_gpu_acceleration': True,
    'use_tensorrt_optimization': True,
    'increase_batch_size': True,
    'enable_mixed_precision': True
}
```

## 开发和调试

### 1. 开发环境设置

```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 启用调试模式
export YOLOS_DEBUG=1
export YOLOS_LOG_LEVEL=DEBUG
```

### 2. 性能分析

```python
from src.utils.profiler import AIoTProfiler

# 创建性能分析器
profiler = AIoTProfiler()

# 分析推理性能
with profiler.profile('inference'):
    result = recognition_system.recognize(image)

# 生成性能报告
profiler.generate_report()
```

### 3. 远程调试

```python
# 启用远程调试服务器
from src.utils.remote_debug import RemoteDebugServer

debug_server = RemoteDebugServer(port=8888)
debug_server.start()

# 通过Web界面访问: http://board_ip:8888
```

## 生产部署

### 1. 容器化部署

```dockerfile
# Dockerfile for AIoT boards
FROM nvcr.io/nvidia/l4t-base:r32.6.1  # For Jetson

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt
RUN python setup.py install

CMD ["python", "src/main.py"]
```

### 2. 系统服务

```bash
# 创建systemd服务
sudo cp scripts/yolos-aiot.service /etc/systemd/system/
sudo systemctl enable yolos-aiot
sudo systemctl start yolos-aiot
```

### 3. 监控和维护

```python
# 健康检查
from src.monitoring.health_checker import AIoTHealthChecker

health_checker = AIoTHealthChecker()
status = health_checker.check_system_health()

# 自动更新
from src.utils.auto_updater import AIoTAutoUpdater

updater = AIoTAutoUpdater()
updater.check_and_update()
```

## 社区和支持

- **GitHub Issues**: 报告问题和功能请求
- **讨论区**: 技术交流和经验分享
- **Wiki**: 详细文档和教程
- **示例项目**: 各种AIoT开发板的完整示例

## 贡献指南

欢迎为YOLOS AIoT支持贡献代码：

1. Fork项目仓库
2. 创建功能分支
3. 添加新的AIoT开发板支持
4. 编写测试用例
5. 提交Pull Request

---

**注意**: 本指南会随着新的AIoT开发板支持而持续更新。建议定期查看最新版本。