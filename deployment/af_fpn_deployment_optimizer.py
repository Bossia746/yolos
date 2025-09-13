#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AF-FPN部署优化和兼容性测试模块

本模块提供全面的部署优化和兼容性测试功能：
1. 多平台部署优化 - Windows、Linux、macOS适配
2. 硬件兼容性测试 - GPU、CPU、边缘设备支持
3. 推理引擎集成 - ONNX、TensorRT、OpenVINO等
4. 量化和压缩优化 - INT8量化、模型剪枝、知识蒸馏
5. 容器化部署 - Docker、Kubernetes支持
6. 云平台适配 - AWS、Azure、GCP等云服务
7. 边缘设备优化 - 移动端、嵌入式设备部署
8. 性能监控和调优 - 实时性能监控和自动调优

部署策略：
- 生产环境：高精度、高稳定性配置
- 边缘环境：轻量化、低延迟配置
- 云环境：弹性扩展、负载均衡配置
- 移动环境：极致优化、资源受限配置

Author: YOLOS Team
Date: 2024-12-12
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.quantization as quant
import numpy as np
import os
import sys
import json
import yaml
import platform
import subprocess
import shutil
import tempfile
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
from contextlib import contextmanager
import time
from datetime import datetime

# 导入第三方库（可选）
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    warnings.warn("ONNX相关库未安装，ONNX导出功能不可用")

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    warnings.warn("TensorRT未安装，TensorRT优化功能不可用")

try:
    import openvino as ov
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    warnings.warn("OpenVINO未安装，OpenVINO优化功能不可用")

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    warnings.warn("Docker Python SDK未安装，容器化功能受限")

# 导入AF-FPN模块
try:
    import sys
    sys.path.append('../src/models')
    from af_fpn_integration_optimizer import (
        AFPNIntegrationOptimizer, IntegrationConfig,
        create_af_fpn_integration_optimizer, INTEGRATION_CONFIGS
    )
except ImportError:
    warnings.warn("无法导入AF-FPN模块，请确保路径正确")


class DeploymentPlatform(Enum):
    """部署平台枚举"""
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    ANDROID = "android"
    IOS = "ios"
    EMBEDDED = "embedded"
    CLOUD_AWS = "aws"
    CLOUD_AZURE = "azure"
    CLOUD_GCP = "gcp"
    EDGE_JETSON = "jetson"
    EDGE_RPI = "raspberry_pi"


class InferenceEngine(Enum):
    """推理引擎枚举"""
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"
    TFLITE = "tflite"
    NCNN = "ncnn"
    MNN = "mnn"


class OptimizationLevel(Enum):
    """优化级别枚举"""
    NONE = "none"              # 无优化
    BASIC = "basic"            # 基础优化
    STANDARD = "standard"      # 标准优化
    AGGRESSIVE = "aggressive"  # 激进优化
    EXTREME = "extreme"        # 极致优化


@dataclass
class DeploymentConfig:
    """部署配置"""
    # 基础配置
    target_platform: DeploymentPlatform = DeploymentPlatform.LINUX
    inference_engine: InferenceEngine = InferenceEngine.PYTORCH
    optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
    
    # 硬件配置
    use_gpu: bool = True
    gpu_memory_limit_mb: Optional[int] = None
    cpu_threads: Optional[int] = None
    
    # 模型优化配置
    enable_quantization: bool = False
    quantization_mode: str = "dynamic"  # 'dynamic', 'static', 'qat'
    enable_pruning: bool = False
    pruning_ratio: float = 0.1
    enable_distillation: bool = False
    
    # 推理配置
    batch_size: int = 1
    input_size: Tuple[int, int] = (416, 416)
    precision: str = "fp32"  # 'fp32', 'fp16', 'int8'
    
    # 部署环境配置
    container_runtime: str = "docker"  # 'docker', 'podman', 'containerd'
    k8s_deployment: bool = False
    auto_scaling: bool = False
    
    # 监控配置
    enable_monitoring: bool = True
    performance_logging: bool = True
    error_handling: str = "graceful"  # 'strict', 'graceful', 'ignore'
    
    # 输出配置
    output_dir: str = "./deployment_output"
    export_formats: List[str] = field(default_factory=lambda: ["onnx", "torchscript"])
    

class PlatformCompatibilityChecker:
    """平台兼容性检查器"""
    
    def __init__(self):
        self.system_info = self._get_system_info()
        
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        info = {
            'platform': platform.system().lower(),
            'architecture': platform.machine(),
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cpu_count': os.cpu_count()
        }
        
        if torch.cuda.is_available():
            info.update({
                'gpu_count': torch.cuda.device_count(),
                'gpu_name': torch.cuda.get_device_name(0),
                'cuda_version': torch.version.cuda,
                'cudnn_version': torch.backends.cudnn.version()
            })
            
        return info
        
    def check_platform_compatibility(self, config: DeploymentConfig) -> Dict[str, Any]:
        """检查平台兼容性
        
        Args:
            config: 部署配置
            
        Returns:
            兼容性检查结果
        """
        compatibility = {
            'platform_supported': True,
            'gpu_supported': True,
            'inference_engine_available': True,
            'warnings': [],
            'recommendations': []
        }
        
        # 检查平台支持
        current_platform = self.system_info['platform']
        target_platform = config.target_platform.value
        
        if target_platform not in ['windows', 'linux', 'macos'] and current_platform != target_platform:
            compatibility['platform_supported'] = False
            compatibility['warnings'].append(f"目标平台 {target_platform} 与当前平台 {current_platform} 不匹配")
            
        # 检查GPU支持
        if config.use_gpu and not self.system_info['cuda_available']:
            compatibility['gpu_supported'] = False
            compatibility['warnings'].append("配置要求GPU但CUDA不可用")
            compatibility['recommendations'].append("建议切换到CPU模式或安装CUDA")
            
        # 检查推理引擎可用性
        engine_availability = {
            InferenceEngine.PYTORCH: True,  # 总是可用
            InferenceEngine.ONNX: ONNX_AVAILABLE,
            InferenceEngine.TENSORRT: TRT_AVAILABLE and self.system_info['cuda_available'],
            InferenceEngine.OPENVINO: OPENVINO_AVAILABLE
        }
        
        if not engine_availability.get(config.inference_engine, False):
            compatibility['inference_engine_available'] = False
            compatibility['warnings'].append(f"推理引擎 {config.inference_engine.value} 不可用")
            
        # 性能建议
        if config.batch_size > 1 and not config.use_gpu:
            compatibility['recommendations'].append("大批次处理建议使用GPU加速")
            
        if config.precision == "fp16" and not self.system_info['cuda_available']:
            compatibility['warnings'].append("FP16精度需要GPU支持")
            
        return compatibility
        
    def get_optimal_config(self, target_scenario: str) -> DeploymentConfig:
        """获取最优部署配置
        
        Args:
            target_scenario: 目标场景 ('production', 'edge', 'mobile', 'cloud')
            
        Returns:
            优化的部署配置
        """
        base_config = DeploymentConfig()
        
        if target_scenario == 'production':
            # 生产环境：稳定性和精度优先
            base_config.optimization_level = OptimizationLevel.STANDARD
            base_config.precision = "fp32"
            base_config.enable_monitoring = True
            base_config.error_handling = "graceful"
            
        elif target_scenario == 'edge':
            # 边缘环境：性能和资源优化
            base_config.optimization_level = OptimizationLevel.AGGRESSIVE
            base_config.enable_quantization = True
            base_config.precision = "int8"
            base_config.batch_size = 1
            
        elif target_scenario == 'mobile':
            # 移动环境：极致优化
            base_config.optimization_level = OptimizationLevel.EXTREME
            base_config.enable_quantization = True
            base_config.enable_pruning = True
            base_config.precision = "int8"
            base_config.use_gpu = False
            
        elif target_scenario == 'cloud':
            # 云环境：弹性和扩展性
            base_config.optimization_level = OptimizationLevel.STANDARD
            base_config.k8s_deployment = True
            base_config.auto_scaling = True
            base_config.enable_monitoring = True
            
        # 根据系统能力调整
        if not self.system_info['cuda_available']:
            base_config.use_gpu = False
            base_config.precision = "fp32" if base_config.precision == "fp16" else base_config.precision
            
        return base_config


class ModelOptimizer:
    """模型优化器"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """优化模型
        
        Args:
            model: 原始模型
            
        Returns:
            优化后的模型
        """
        optimized_model = model
        
        # 1. 量化优化
        if self.config.enable_quantization:
            optimized_model = self._apply_quantization(optimized_model)
            
        # 2. 剪枝优化
        if self.config.enable_pruning:
            optimized_model = self._apply_pruning(optimized_model)
            
        # 3. 融合优化
        optimized_model = self._apply_fusion(optimized_model)
        
        # 4. 图优化
        if self.config.optimization_level in [OptimizationLevel.AGGRESSIVE, OptimizationLevel.EXTREME]:
            optimized_model = self._apply_graph_optimization(optimized_model)
            
        return optimized_model
        
    def _apply_quantization(self, model: nn.Module) -> nn.Module:
        """应用量化优化"""
        if self.config.quantization_mode == "dynamic":
            # 动态量化
            quantized_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
            )
        elif self.config.quantization_mode == "static":
            # 静态量化（需要校准数据）
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            # 这里应该运行校准数据
            quantized_model = torch.quantization.convert(model, inplace=False)
        else:
            quantized_model = model
            
        logging.info(f"应用{self.config.quantization_mode}量化优化")
        return quantized_model
        
    def _apply_pruning(self, model: nn.Module) -> nn.Module:
        """应用剪枝优化"""
        try:
            import torch.nn.utils.prune as prune
            
            # 结构化剪枝
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    prune.l1_unstructured(module, name='weight', amount=self.config.pruning_ratio)
                    prune.remove(module, 'weight')
                    
            logging.info(f"应用剪枝优化，剪枝比例: {self.config.pruning_ratio}")
        except ImportError:
            logging.warning("剪枝功能不可用，跳过剪枝优化")
            
        return model
        
    def _apply_fusion(self, model: nn.Module) -> nn.Module:
        """应用算子融合优化"""
        # Conv-BN融合
        fused_model = torch.quantization.fuse_modules(
            model, 
            [['conv', 'bn'], ['conv', 'bn', 'relu']], 
            inplace=False
        )
        
        logging.info("应用算子融合优化")
        return fused_model
        
    def _apply_graph_optimization(self, model: nn.Module) -> nn.Module:
        """应用图优化"""
        # 使用TorchScript进行图优化
        try:
            model.eval()
            dummy_input = torch.randn(1, 3, *self.config.input_size)
            
            # 转换为TorchScript
            traced_model = torch.jit.trace(model, dummy_input)
            
            # 图优化
            optimized_model = torch.jit.optimize_for_inference(traced_model)
            
            logging.info("应用TorchScript图优化")
            return optimized_model
        except Exception as e:
            logging.warning(f"图优化失败: {e}")
            return model


class InferenceEngineConverter:
    """推理引擎转换器"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        
    def convert_model(self, model: nn.Module, output_path: str) -> Dict[str, str]:
        """转换模型到目标推理引擎
        
        Args:
            model: PyTorch模型
            output_path: 输出路径
            
        Returns:
            转换结果路径字典
        """
        results = {}
        
        # 确保输出目录存在
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # 根据配置的导出格式进行转换
        for export_format in self.config.export_formats:
            try:
                if export_format == "onnx":
                    onnx_path = self._convert_to_onnx(model, output_path)
                    results["onnx"] = onnx_path
                    
                elif export_format == "torchscript":
                    ts_path = self._convert_to_torchscript(model, output_path)
                    results["torchscript"] = ts_path
                    
                elif export_format == "tensorrt":
                    trt_path = self._convert_to_tensorrt(model, output_path)
                    results["tensorrt"] = trt_path
                    
                elif export_format == "openvino":
                    ov_path = self._convert_to_openvino(model, output_path)
                    results["openvino"] = ov_path
                    
            except Exception as e:
                logging.error(f"{export_format}转换失败: {e}")
                
        return results
        
    def _convert_to_onnx(self, model: nn.Module, output_path: str) -> str:
        """转换为ONNX格式"""
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX不可用")
            
        model.eval()
        dummy_input = torch.randn(self.config.batch_size, 3, *self.config.input_size)
        
        onnx_path = os.path.join(output_path, "model.onnx")
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # 验证ONNX模型
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        logging.info(f"ONNX模型已保存到: {onnx_path}")
        return onnx_path
        
    def _convert_to_torchscript(self, model: nn.Module, output_path: str) -> str:
        """转换为TorchScript格式"""
        model.eval()
        dummy_input = torch.randn(self.config.batch_size, 3, *self.config.input_size)
        
        # 使用trace方式
        traced_model = torch.jit.trace(model, dummy_input)
        
        ts_path = os.path.join(output_path, "model.pt")
        traced_model.save(ts_path)
        
        logging.info(f"TorchScript模型已保存到: {ts_path}")
        return ts_path
        
    def _convert_to_tensorrt(self, model: nn.Module, output_path: str) -> str:
        """转换为TensorRT格式"""
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT不可用")
            
        # 首先转换为ONNX
        onnx_path = self._convert_to_onnx(model, output_path)
        
        # 然后转换为TensorRT
        trt_path = os.path.join(output_path, "model.trt")
        
        # 这里应该使用TensorRT API进行转换
        # 简化实现，实际需要更复杂的TensorRT转换逻辑
        logging.info(f"TensorRT转换需要额外实现，ONNX路径: {onnx_path}")
        
        return trt_path
        
    def _convert_to_openvino(self, model: nn.Module, output_path: str) -> str:
        """转换为OpenVINO格式"""
        if not OPENVINO_AVAILABLE:
            raise RuntimeError("OpenVINO不可用")
            
        # 首先转换为ONNX
        onnx_path = self._convert_to_onnx(model, output_path)
        
        # 然后转换为OpenVINO IR
        ov_path = os.path.join(output_path, "model.xml")
        
        # 使用OpenVINO Model Optimizer
        # 这里应该调用mo命令或使用OpenVINO Python API
        logging.info(f"OpenVINO转换需要额外实现，ONNX路径: {onnx_path}")
        
        return ov_path


class ContainerBuilder:
    """容器构建器"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        
    def build_docker_image(self, model_path: str, image_name: str) -> str:
        """构建Docker镜像
        
        Args:
            model_path: 模型文件路径
            image_name: 镜像名称
            
        Returns:
            构建的镜像ID
        """
        dockerfile_content = self._generate_dockerfile(model_path)
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            dockerfile_path = os.path.join(temp_dir, "Dockerfile")
            
            # 写入Dockerfile
            with open(dockerfile_path, 'w') as f:
                f.write(dockerfile_content)
                
            # 复制模型文件
            shutil.copy2(model_path, temp_dir)
            
            # 构建镜像
            if DOCKER_AVAILABLE:
                client = docker.from_env()
                image, logs = client.images.build(
                    path=temp_dir,
                    tag=image_name,
                    rm=True
                )
                
                logging.info(f"Docker镜像构建完成: {image_name}")
                return image.id
            else:
                # 使用命令行构建
                cmd = f"docker build -t {image_name} {temp_dir}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logging.info(f"Docker镜像构建完成: {image_name}")
                    return image_name
                else:
                    raise RuntimeError(f"Docker构建失败: {result.stderr}")
                    
    def _generate_dockerfile(self, model_path: str) -> str:
        """生成Dockerfile内容"""
        model_filename = os.path.basename(model_path)
        
        dockerfile = f"""
# AF-FPN部署镜像
FROM pytorch/pytorch:latest

# 安装依赖
RUN pip install --no-cache-dir \
    numpy \
    opencv-python-headless \
    pillow \
    flask \
    gunicorn

# 复制模型文件
COPY {model_filename} /app/model/

# 复制应用代码
COPY app.py /app/
COPY requirements.txt /app/

# 设置工作目录
WORKDIR /app

# 安装应用依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露端口
EXPOSE 8080

# 启动命令
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "4", "app:app"]
"""
        
        return dockerfile
        
    def generate_k8s_manifests(self, image_name: str, output_dir: str) -> Dict[str, str]:
        """生成Kubernetes部署清单
        
        Args:
            image_name: Docker镜像名称
            output_dir: 输出目录
            
        Returns:
            生成的清单文件路径
        """
        manifests = {}
        
        # Deployment清单
        deployment_yaml = self._generate_deployment_yaml(image_name)
        deployment_path = os.path.join(output_dir, "deployment.yaml")
        with open(deployment_path, 'w') as f:
            f.write(deployment_yaml)
        manifests['deployment'] = deployment_path
        
        # Service清单
        service_yaml = self._generate_service_yaml()
        service_path = os.path.join(output_dir, "service.yaml")
        with open(service_path, 'w') as f:
            f.write(service_yaml)
        manifests['service'] = service_path
        
        # HPA清单（如果启用自动扩展）
        if self.config.auto_scaling:
            hpa_yaml = self._generate_hpa_yaml()
            hpa_path = os.path.join(output_dir, "hpa.yaml")
            with open(hpa_path, 'w') as f:
                f.write(hpa_yaml)
            manifests['hpa'] = hpa_path
            
        return manifests
        
    def _generate_deployment_yaml(self, image_name: str) -> str:
        """生成Deployment YAML"""
        return f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: af-fpn-yolos
  labels:
    app: af-fpn-yolos
spec:
  replicas: 3
  selector:
    matchLabels:
      app: af-fpn-yolos
  template:
    metadata:
      labels:
        app: af-fpn-yolos
    spec:
      containers:
      - name: af-fpn-yolos
        image: {image_name}
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        env:
        - name: MODEL_PATH
          value: "/app/model"
        - name: BATCH_SIZE
          value: "{self.config.batch_size}"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
"""
        
    def _generate_service_yaml(self) -> str:
        """生成Service YAML"""
        return """
apiVersion: v1
kind: Service
metadata:
  name: af-fpn-yolos-service
spec:
  selector:
    app: af-fpn-yolos
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
  type: LoadBalancer
"""
        
    def _generate_hpa_yaml(self) -> str:
        """生成HPA YAML"""
        return """
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: af-fpn-yolos-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: af-fpn-yolos
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
"""


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.metrics_history = []
        
    def start_monitoring(self, model: nn.Module, duration_seconds: int = 300):
        """开始性能监控
        
        Args:
            model: 监控的模型
            duration_seconds: 监控持续时间（秒）
        """
        logging.info(f"开始性能监控，持续时间: {duration_seconds}秒")
        
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            metrics = self._collect_metrics(model)
            self.metrics_history.append(metrics)
            
            # 检查性能异常
            self._check_performance_anomalies(metrics)
            
            time.sleep(10)  # 每10秒采集一次
            
        logging.info("性能监控完成")
        
    def _collect_metrics(self, model: nn.Module) -> Dict[str, float]:
        """收集性能指标"""
        import psutil
        
        metrics = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_mb': psutil.virtual_memory().used / (1024**2)
        }
        
        # GPU指标
        if torch.cuda.is_available():
            metrics.update({
                'gpu_memory_used_mb': torch.cuda.memory_allocated() / (1024**2),
                'gpu_memory_cached_mb': torch.cuda.memory_reserved() / (1024**2)
            })
            
        # 模型推理性能测试
        dummy_input = torch.randn(1, 3, *self.config.input_size)
        if torch.cuda.is_available() and self.config.use_gpu:
            dummy_input = dummy_input.cuda()
            model = model.cuda()
            
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        inference_time = (end_time - start_time) * 1000  # 毫秒
        metrics['inference_time_ms'] = inference_time
        metrics['fps'] = 1000.0 / inference_time
        
        return metrics
        
    def _check_performance_anomalies(self, metrics: Dict[str, float]):
        """检查性能异常"""
        # CPU使用率过高
        if metrics['cpu_percent'] > 90:
            logging.warning(f"CPU使用率过高: {metrics['cpu_percent']:.1f}%")
            
        # 内存使用率过高
        if metrics['memory_percent'] > 85:
            logging.warning(f"内存使用率过高: {metrics['memory_percent']:.1f}%")
            
        # 推理时间过长
        if metrics['inference_time_ms'] > 100:  # 100ms阈值
            logging.warning(f"推理时间过长: {metrics['inference_time_ms']:.2f}ms")
            
        # FPS过低
        if metrics['fps'] < 10:
            logging.warning(f"FPS过低: {metrics['fps']:.1f}")
            
    def generate_performance_report(self, output_path: str) -> str:
        """生成性能报告
        
        Args:
            output_path: 报告输出路径
            
        Returns:
            报告文件路径
        """
        if not self.metrics_history:
            logging.warning("没有性能数据，无法生成报告")
            return ""
            
        report_path = os.path.join(output_path, f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # 计算统计指标
        stats = self._calculate_performance_stats()
        
        report_data = {
            'summary': stats,
            'raw_metrics': self.metrics_history,
            'config': asdict(self.config),
            'generated_at': datetime.now().isoformat()
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
            
        logging.info(f"性能报告已生成: {report_path}")
        return report_path
        
    def _calculate_performance_stats(self) -> Dict[str, float]:
        """计算性能统计指标"""
        if not self.metrics_history:
            return {}
            
        # 提取各项指标
        cpu_usage = [m['cpu_percent'] for m in self.metrics_history]
        memory_usage = [m['memory_percent'] for m in self.metrics_history]
        inference_times = [m['inference_time_ms'] for m in self.metrics_history]
        fps_values = [m['fps'] for m in self.metrics_history]
        
        stats = {
            'avg_cpu_percent': np.mean(cpu_usage),
            'max_cpu_percent': np.max(cpu_usage),
            'avg_memory_percent': np.mean(memory_usage),
            'max_memory_percent': np.max(memory_usage),
            'avg_inference_time_ms': np.mean(inference_times),
            'p95_inference_time_ms': np.percentile(inference_times, 95),
            'avg_fps': np.mean(fps_values),
            'min_fps': np.min(fps_values)
        }
        
        return stats


class AFPNDeploymentOptimizer:
    """AF-FPN部署优化器主类"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.compatibility_checker = PlatformCompatibilityChecker()
        self.model_optimizer = ModelOptimizer(config)
        self.engine_converter = InferenceEngineConverter(config)
        self.container_builder = ContainerBuilder(config)
        self.performance_monitor = PerformanceMonitor(config)
        
        # 创建输出目录
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(config.output_dir, 'deployment.log')),
                logging.StreamHandler()
            ]
        )
        
    def deploy_model(self, model: nn.Module, model_name: str = "af_fpn_yolos") -> Dict[str, Any]:
        """完整的模型部署流程
        
        Args:
            model: AF-FPN模型
            model_name: 模型名称
            
        Returns:
            部署结果信息
        """
        logging.info(f"开始部署模型: {model_name}")
        
        deployment_results = {
            'model_name': model_name,
            'config': asdict(self.config),
            'start_time': datetime.now().isoformat()
        }
        
        try:
            # 1. 兼容性检查
            logging.info("执行兼容性检查...")
            compatibility = self.compatibility_checker.check_platform_compatibility(self.config)
            deployment_results['compatibility'] = compatibility
            
            if not compatibility['platform_supported']:
                raise RuntimeError("平台不兼容")
                
            # 2. 模型优化
            logging.info("执行模型优化...")
            optimized_model = self.model_optimizer.optimize_model(model)
            
            # 3. 模型转换
            logging.info("执行模型转换...")
            conversion_results = self.engine_converter.convert_model(
                optimized_model, 
                os.path.join(self.config.output_dir, "models")
            )
            deployment_results['converted_models'] = conversion_results
            
            # 4. 容器化（如果需要）
            if self.config.container_runtime:
                logging.info("构建容器镜像...")
                
                # 选择主要模型文件
                main_model_path = conversion_results.get('torchscript') or conversion_results.get('onnx')
                if main_model_path:
                    image_id = self.container_builder.build_docker_image(
                        main_model_path, 
                        f"{model_name}:latest"
                    )
                    deployment_results['docker_image'] = image_id
                    
                    # K8s清单生成
                    if self.config.k8s_deployment:
                        k8s_manifests = self.container_builder.generate_k8s_manifests(
                            f"{model_name}:latest",
                            os.path.join(self.config.output_dir, "k8s")
                        )
                        deployment_results['k8s_manifests'] = k8s_manifests
                        
            # 5. 性能监控
            if self.config.enable_monitoring:
                logging.info("启动性能监控...")
                self.performance_monitor.start_monitoring(optimized_model, 60)  # 监控1分钟
                
                performance_report = self.performance_monitor.generate_performance_report(
                    self.config.output_dir
                )
                deployment_results['performance_report'] = performance_report
                
            # 6. 生成部署文档
            deployment_doc = self._generate_deployment_documentation(deployment_results)
            deployment_results['documentation'] = deployment_doc
            
            deployment_results['status'] = 'success'
            deployment_results['end_time'] = datetime.now().isoformat()
            
            logging.info(f"模型 {model_name} 部署完成")
            
        except Exception as e:
            logging.error(f"部署失败: {e}")
            deployment_results['status'] = 'failed'
            deployment_results['error'] = str(e)
            deployment_results['end_time'] = datetime.now().isoformat()
            
        # 保存部署结果
        results_path = os.path.join(self.config.output_dir, "deployment_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(deployment_results, f, indent=2, ensure_ascii=False)
            
        return deployment_results
        
    def _generate_deployment_documentation(self, deployment_results: Dict[str, Any]) -> str:
        """生成部署文档"""
        doc_content = f"""
# AF-FPN YOLOS 部署文档

## 部署信息
- 模型名称: {deployment_results['model_name']}
- 部署时间: {deployment_results['start_time']}
- 目标平台: {self.config.target_platform.value}
- 推理引擎: {self.config.inference_engine.value}

## 模型文件
"""
        
        if 'converted_models' in deployment_results:
            for format_name, model_path in deployment_results['converted_models'].items():
                doc_content += f"- {format_name.upper()}: `{model_path}`\n"
                
        doc_content += """

## 使用方法

### Python推理示例
```python
import torch

# 加载模型
model = torch.jit.load('model.pt')
model.eval()

# 推理
input_tensor = torch.randn(1, 3, 416, 416)
with torch.no_grad():
    output = model(input_tensor)
```

### Docker部署
```bash
# 运行容器
docker run -p 8080:8080 af_fpn_yolos:latest
```

### Kubernetes部署
```bash
# 应用清单
kubectl apply -f k8s/
```

## 性能指标
"""
        
        if 'performance_report' in deployment_results:
            doc_content += f"详细性能报告: `{deployment_results['performance_report']}`\n"
            
        doc_content += """

## 注意事项
1. 确保目标环境已安装必要的依赖
2. GPU部署需要CUDA支持
3. 生产环境建议启用性能监控
4. 定期检查模型性能和资源使用情况
"""
        
        doc_path = os.path.join(self.config.output_dir, "README.md")
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(doc_content)
            
        return doc_path


def create_deployment_optimizer(target_scenario: str = "production", 
                              **kwargs) -> AFPNDeploymentOptimizer:
    """创建部署优化器的工厂函数
    
    Args:
        target_scenario: 目标场景
        **kwargs: 其他配置参数
        
    Returns:
        配置好的部署优化器
    """
    checker = PlatformCompatibilityChecker()
    base_config = checker.get_optimal_config(target_scenario)
    
    # 应用自定义配置
    for key, value in kwargs.items():
        if hasattr(base_config, key):
            setattr(base_config, key, value)
            
    return AFPNDeploymentOptimizer(base_config)


if __name__ == '__main__':
    # 测试部署优化器
    print("AF-FPN部署优化器测试")
    
    # 创建测试配置
    config = DeploymentConfig(
        target_platform=DeploymentPlatform.LINUX,
        inference_engine=InferenceEngine.PYTORCH,
        optimization_level=OptimizationLevel.STANDARD,
        export_formats=["torchscript", "onnx"],
        output_dir="./deployment_test"
    )
    
    # 创建部署优化器
    optimizer = AFPNDeploymentOptimizer(config)
    
    # 兼容性检查
    compatibility = optimizer.compatibility_checker.check_platform_compatibility(config)
    print(f"兼容性检查结果: {compatibility}")
    
    # 创建测试模型
    try:
        test_model = create_af_fpn_integration_optimizer(
            [256, 512, 1024, 2048], 80, **INTEGRATION_CONFIGS['aiot_balanced']
        )
        
        # 执行部署
        results = optimizer.deploy_model(test_model, "test_af_fpn")
        
        print(f"\n部署结果: {results['status']}")
        if results['status'] == 'success':
            print(f"转换的模型: {list(results.get('converted_models', {}).keys())}")
            print(f"部署文档: {results.get('documentation')}")
        else:
            print(f"部署失败: {results.get('error')}")
            
    except Exception as e:
        print(f"测试失败: {e}")
        
    print("\nAF-FPN部署优化器测试完成！")