#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
硬件平台兼容性测试器

执行YOLOS系统在不同硬件平台上的兼容性测试，包括：
1. K230芯片平台兼容性测试
2. ESP32系列微控制器测试
3. 树莓派系列单板计算机测试
4. NVIDIA Jetson系列边缘计算平台测试
5. 通用x86/ARM架构测试
6. 硬件资源限制测试
7. 跨平台性能对比测试
"""

import platform
import psutil
import time
import subprocess
import json
import numpy as np
import cv2
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging
import sys
import os
from pathlib import Path
import tempfile
import shutil
import threading
import multiprocessing
import socket
import struct

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HardwarePlatform(Enum):
    """硬件平台类型"""
    K230 = "k230"                    # K230芯片
    ESP32 = "esp32"                  # ESP32系列
    RASPBERRY_PI = "raspberry_pi"    # 树莓派
    JETSON_NANO = "jetson_nano"      # Jetson Nano
    JETSON_XAVIER = "jetson_xavier"  # Jetson Xavier
    X86_64 = "x86_64"               # x86_64架构
    ARM64 = "arm64"                 # ARM64架构
    GENERIC_ARM = "generic_arm"      # 通用ARM
    UNKNOWN = "unknown"             # 未知平台

class TestCategory(Enum):
    """测试类别"""
    HARDWARE_DETECTION = "hardware_detection"      # 硬件检测
    PERFORMANCE_BASELINE = "performance_baseline"  # 性能基准
    MEMORY_CONSTRAINTS = "memory_constraints"      # 内存限制
    COMPUTE_CAPABILITY = "compute_capability"      # 计算能力
    IO_PERFORMANCE = "io_performance"              # IO性能
    POWER_CONSUMPTION = "power_consumption"        # 功耗测试
    THERMAL_BEHAVIOR = "thermal_behavior"          # 热行为
    CROSS_PLATFORM = "cross_platform"             # 跨平台

class TestSeverity(Enum):
    """测试严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class HardwareInfo:
    """硬件信息"""
    platform: HardwarePlatform
    architecture: str
    cpu_count: int
    cpu_freq: float
    total_memory: float  # GB
    available_memory: float  # GB
    gpu_available: bool
    gpu_memory: Optional[float] = None  # GB
    storage_type: str = "unknown"
    network_interfaces: List[str] = field(default_factory=list)
    additional_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HardwareTestCase:
    """硬件测试用例"""
    name: str
    category: TestCategory
    severity: TestSeverity
    target_platforms: List[HardwarePlatform]
    description: str
    test_function: callable
    min_memory_gb: float = 0.5
    min_cpu_cores: int = 1
    requires_gpu: bool = False
    timeout: float = 60.0
    expected_performance: Dict[str, float] = field(default_factory=dict)

@dataclass
class HardwareTestResult:
    """硬件测试结果"""
    test_name: str
    platform: HardwarePlatform
    success: bool
    execution_time: float
    performance_metrics: Dict[str, Any]
    resource_usage: Dict[str, float]
    compatibility_score: float  # 0-100
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

class HardwareDetector:
    """硬件检测器"""
    
    @staticmethod
    def detect_platform() -> HardwarePlatform:
        """检测当前硬件平台"""
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        # 检查特定平台标识
        try:
            # 检查是否为树莓派
            if os.path.exists('/proc/device-tree/model'):
                with open('/proc/device-tree/model', 'r') as f:
                    model = f.read().lower()
                    if 'raspberry pi' in model:
                        return HardwarePlatform.RASPBERRY_PI
            
            # 检查是否为Jetson
            if os.path.exists('/proc/device-tree/model'):
                with open('/proc/device-tree/model', 'r') as f:
                    model = f.read().lower()
                    if 'jetson' in model:
                        if 'nano' in model:
                            return HardwarePlatform.JETSON_NANO
                        elif 'xavier' in model:
                            return HardwarePlatform.JETSON_XAVIER
            
            # 检查CPU信息
            if os.path.exists('/proc/cpuinfo'):
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read().lower()
                    if 'k230' in cpuinfo:
                        return HardwarePlatform.K230
                    elif 'esp32' in cpuinfo:
                        return HardwarePlatform.ESP32
        
        except Exception:
            pass
        
        # 基于架构判断
        if machine in ['x86_64', 'amd64']:
            return HardwarePlatform.X86_64
        elif machine in ['aarch64', 'arm64']:
            return HardwarePlatform.ARM64
        elif machine.startswith('arm'):
            return HardwarePlatform.GENERIC_ARM
        
        return HardwarePlatform.UNKNOWN
    
    @staticmethod
    def get_hardware_info() -> HardwareInfo:
        """获取硬件信息"""
        platform_type = HardwareDetector.detect_platform()
        
        # 基本信息
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else 0.0
        memory = psutil.virtual_memory()
        total_memory = memory.total / (1024**3)  # GB
        available_memory = memory.available / (1024**3)  # GB
        
        # GPU检测
        gpu_available = False
        gpu_memory = None
        try:
            # 尝试检测NVIDIA GPU
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gpu_available = True
                gpu_memory = float(result.stdout.strip()) / 1024  # GB
        except Exception:
            pass
        
        # 网络接口
        network_interfaces = list(psutil.net_if_addrs().keys())
        
        return HardwareInfo(
            platform=platform_type,
            architecture=platform.machine(),
            cpu_count=cpu_count,
            cpu_freq=cpu_freq,
            total_memory=total_memory,
            available_memory=available_memory,
            gpu_available=gpu_available,
            gpu_memory=gpu_memory,
            network_interfaces=network_interfaces,
            additional_info={
                'system': platform.system(),
                'release': platform.release(),
                'python_version': platform.python_version()
            }
        )

class MockDataGenerator:
    """模拟数据生成器"""
    
    @staticmethod
    def generate_test_workload(complexity: str = "medium") -> Dict[str, Any]:
        """生成测试工作负载"""
        if complexity == "light":
            return {
                'image_size': (320, 240),
                'batch_size': 1,
                'iterations': 10,
                'compute_ops': 100
            }
        elif complexity == "medium":
            return {
                'image_size': (640, 480),
                'batch_size': 4,
                'iterations': 50,
                'compute_ops': 1000
            }
        else:  # heavy
            return {
                'image_size': (1280, 720),
                'batch_size': 8,
                'iterations': 100,
                'compute_ops': 10000
            }
    
    @staticmethod
    def generate_test_image(width: int, height: int) -> np.ndarray:
        """生成测试图像"""
        return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

class HardwareCompatibilityTester:
    """硬件兼容性测试器"""
    
    def __init__(self):
        self.hardware_info = HardwareDetector.get_hardware_info()
        self.test_cases: List[HardwareTestCase] = []
        self.results: List[HardwareTestResult] = []
        self.logger = logging.getLogger(__name__)
        self._initialize_test_cases()
    
    def _initialize_test_cases(self):
        """初始化测试用例"""
        # 硬件检测测试
        self.test_cases.append(HardwareTestCase(
            name="hardware_detection_test",
            category=TestCategory.HARDWARE_DETECTION,
            severity=TestSeverity.CRITICAL,
            target_platforms=list(HardwarePlatform),
            description="硬件平台检测和信息获取测试",
            test_function=self._test_hardware_detection
        ))
        
        # 性能基准测试
        self.test_cases.append(HardwareTestCase(
            name="performance_baseline_test",
            category=TestCategory.PERFORMANCE_BASELINE,
            severity=TestSeverity.HIGH,
            target_platforms=list(HardwarePlatform),
            description="硬件平台性能基准测试",
            test_function=self._test_performance_baseline,
            timeout=120.0
        ))
        
        # 内存限制测试
        self.test_cases.append(HardwareTestCase(
            name="memory_constraints_test",
            category=TestCategory.MEMORY_CONSTRAINTS,
            severity=TestSeverity.HIGH,
            target_platforms=list(HardwarePlatform),
            description="内存限制和管理测试",
            test_function=self._test_memory_constraints,
            min_memory_gb=0.1
        ))
        
        # 计算能力测试
        self.test_cases.append(HardwareTestCase(
            name="compute_capability_test",
            category=TestCategory.COMPUTE_CAPABILITY,
            severity=TestSeverity.MEDIUM,
            target_platforms=list(HardwarePlatform),
            description="计算能力和并行处理测试",
            test_function=self._test_compute_capability
        ))
        
        # IO性能测试
        self.test_cases.append(HardwareTestCase(
            name="io_performance_test",
            category=TestCategory.IO_PERFORMANCE,
            severity=TestSeverity.MEDIUM,
            target_platforms=list(HardwarePlatform),
            description="IO性能和存储访问测试",
            test_function=self._test_io_performance
        ))
    
    def _test_hardware_detection(self) -> Dict[str, Any]:
        """硬件检测测试"""
        results = {
            'platform_detected': self.hardware_info.platform.value,
            'architecture': self.hardware_info.architecture,
            'cpu_count': self.hardware_info.cpu_count,
            'total_memory_gb': self.hardware_info.total_memory,
            'gpu_available': self.hardware_info.gpu_available,
            'detection_accuracy': 100.0
        }
        
        try:
            # 验证检测结果的合理性
            if self.hardware_info.cpu_count <= 0:
                results['detection_accuracy'] -= 20
                results['warnings'] = ['CPU核心数检测异常']
            
            if self.hardware_info.total_memory <= 0:
                results['detection_accuracy'] -= 20
                results['warnings'] = results.get('warnings', []) + ['内存大小检测异常']
            
            # 平台特定验证
            if self.hardware_info.platform == HardwarePlatform.UNKNOWN:
                results['detection_accuracy'] -= 30
                results['warnings'] = results.get('warnings', []) + ['平台类型未能识别']
        
        except Exception as e:
            results['error'] = str(e)
            results['detection_accuracy'] = 0
            raise
        
        return results
    
    def _test_performance_baseline(self) -> Dict[str, Any]:
        """性能基准测试"""
        results = {
            'cpu_benchmark_score': 0,
            'memory_bandwidth_mbps': 0,
            'image_processing_fps': 0,
            'overall_performance_score': 0
        }
        
        try:
            # CPU基准测试
            start_time = time.time()
            
            # 矩阵运算测试
            matrix_size = min(500, int(100 * np.sqrt(self.hardware_info.cpu_count)))
            a = np.random.random((matrix_size, matrix_size))
            b = np.random.random((matrix_size, matrix_size))
            c = np.dot(a, b)
            
            cpu_time = time.time() - start_time
            results['cpu_benchmark_score'] = matrix_size * matrix_size / cpu_time
            
            # 内存带宽测试
            start_time = time.time()
            memory_size = min(100, int(self.hardware_info.available_memory * 0.1)) * 1024 * 1024  # bytes
            data = np.random.bytes(memory_size)
            copied_data = data[:]
            memory_time = time.time() - start_time
            results['memory_bandwidth_mbps'] = (memory_size * 2) / (memory_time * 1024 * 1024)
            
            # 图像处理测试
            start_time = time.time()
            frame_count = 20
            for i in range(frame_count):
                image = MockDataGenerator.generate_test_image(320, 240)
                resized = cv2.resize(image, (640, 480))
                blurred = cv2.GaussianBlur(resized, (5, 5), 0)
            
            image_time = time.time() - start_time
            results['image_processing_fps'] = frame_count / image_time
            
            # 综合性能评分
            cpu_score = min(100, results['cpu_benchmark_score'] / 10000)
            memory_score = min(100, results['memory_bandwidth_mbps'] / 1000)
            image_score = min(100, results['image_processing_fps'] / 10)
            results['overall_performance_score'] = (cpu_score + memory_score + image_score) / 3
        
        except Exception as e:
            results['error'] = str(e)
            raise
        
        return results
    
    def _test_memory_constraints(self) -> Dict[str, Any]:
        """内存限制测试"""
        results = {
            'max_allocation_mb': 0,
            'allocation_success_rate': 0,
            'memory_efficiency': 0,
            'gc_performance': 0
        }
        
        try:
            import gc
            
            # 逐步分配内存测试
            allocated_blocks = []
            block_size_mb = 10
            max_attempts = min(50, int(self.hardware_info.available_memory * 100))  # 限制尝试次数
            successful_allocations = 0
            
            for i in range(max_attempts):
                try:
                    # 分配内存块
                    block = np.random.random((block_size_mb * 1024 * 1024 // 8,)).astype(np.float64)
                    allocated_blocks.append(block)
                    successful_allocations += 1
                    results['max_allocation_mb'] = (i + 1) * block_size_mb
                    
                    # 检查可用内存
                    available = psutil.virtual_memory().available / (1024**3)
                    if available < 0.1:  # 少于100MB时停止
                        break
                        
                except MemoryError:
                    break
                except Exception:
                    break
            
            results['allocation_success_rate'] = successful_allocations / max_attempts
            
            # 内存效率测试
            if allocated_blocks:
                start_time = time.time()
                # 访问分配的内存
                for block in allocated_blocks[:5]:  # 只访问前5个块
                    _ = np.sum(block[:1000])  # 部分访问
                access_time = time.time() - start_time
                results['memory_efficiency'] = len(allocated_blocks) / max(access_time, 0.001)
            
            # 垃圾回收性能测试
            start_time = time.time()
            del allocated_blocks
            gc.collect()
            gc_time = time.time() - start_time
            results['gc_performance'] = successful_allocations / max(gc_time, 0.001)
        
        except Exception as e:
            results['error'] = str(e)
            raise
        
        return results
    
    def _test_compute_capability(self) -> Dict[str, Any]:
        """计算能力测试"""
        results = {
            'single_thread_score': 0,
            'multi_thread_score': 0,
            'parallel_efficiency': 0,
            'compute_intensity_score': 0
        }
        
        try:
            # 单线程计算测试
            start_time = time.time()
            
            # 复杂数学运算
            for i in range(1000):
                x = np.random.random(100)
                y = np.fft.fft(x)
                z = np.real(np.fft.ifft(y))
            
            single_time = time.time() - start_time
            results['single_thread_score'] = 1000 / single_time
            
            # 多线程计算测试
            def compute_worker():
                for i in range(200):
                    x = np.random.random(100)
                    y = np.fft.fft(x)
                    z = np.real(np.fft.ifft(y))
            
            start_time = time.time()
            threads = []
            thread_count = min(self.hardware_info.cpu_count, 4)
            
            for i in range(thread_count):
                thread = threading.Thread(target=compute_worker)
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            multi_time = time.time() - start_time
            results['multi_thread_score'] = (thread_count * 200) / multi_time
            
            # 并行效率
            theoretical_speedup = thread_count
            actual_speedup = results['multi_thread_score'] / results['single_thread_score'] * (1000/200)
            results['parallel_efficiency'] = min(100, (actual_speedup / theoretical_speedup) * 100)
            
            # 计算强度评分
            results['compute_intensity_score'] = (results['single_thread_score'] + results['multi_thread_score']) / 2
        
        except Exception as e:
            results['error'] = str(e)
            raise
        
        return results
    
    def _test_io_performance(self) -> Dict[str, Any]:
        """IO性能测试"""
        results = {
            'sequential_read_mbps': 0,
            'sequential_write_mbps': 0,
            'random_read_iops': 0,
            'random_write_iops': 0,
            'io_efficiency_score': 0
        }
        
        try:
            # 创建临时测试文件
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # 顺序写入测试
                test_data = np.random.bytes(10 * 1024 * 1024)  # 10MB
                
                start_time = time.time()
                with open(temp_path, 'wb') as f:
                    f.write(test_data)
                    f.flush()
                    os.fsync(f.fileno())
                write_time = time.time() - start_time
                results['sequential_write_mbps'] = len(test_data) / (write_time * 1024 * 1024)
                
                # 顺序读取测试
                start_time = time.time()
                with open(temp_path, 'rb') as f:
                    read_data = f.read()
                read_time = time.time() - start_time
                results['sequential_read_mbps'] = len(read_data) / (read_time * 1024 * 1024)
                
                # 随机IO测试（简化版）
                block_size = 4096  # 4KB blocks
                block_count = 100
                
                # 随机写入
                start_time = time.time()
                with open(temp_path, 'r+b') as f:
                    for i in range(block_count):
                        offset = (i * 12345) % (len(test_data) - block_size)  # 伪随机偏移
                        f.seek(offset)
                        f.write(os.urandom(block_size))
                random_write_time = time.time() - start_time
                results['random_write_iops'] = block_count / random_write_time
                
                # 随机读取
                start_time = time.time()
                with open(temp_path, 'rb') as f:
                    for i in range(block_count):
                        offset = (i * 54321) % (len(test_data) - block_size)  # 伪随机偏移
                        f.seek(offset)
                        _ = f.read(block_size)
                random_read_time = time.time() - start_time
                results['random_read_iops'] = block_count / random_read_time
                
                # IO效率评分
                seq_score = (results['sequential_read_mbps'] + results['sequential_write_mbps']) / 2
                random_score = (results['random_read_iops'] + results['random_write_iops']) / 2
                results['io_efficiency_score'] = (seq_score + random_score / 100) / 2  # 归一化
            
            finally:
                # 清理临时文件
                try:
                    os.unlink(temp_path)
                except:
                    pass
        
        except Exception as e:
            results['error'] = str(e)
            raise
        
        return results
    
    def execute_test_case(self, test_case: HardwareTestCase) -> HardwareTestResult:
        """执行单个测试用例"""
        self.logger.info(f"开始执行硬件测试: {test_case.name}")
        
        # 检查平台兼容性
        if (test_case.target_platforms and 
            self.hardware_info.platform not in test_case.target_platforms):
            return HardwareTestResult(
                test_name=test_case.name,
                platform=self.hardware_info.platform,
                success=False,
                execution_time=0.0,
                performance_metrics={},
                resource_usage={},
                compatibility_score=0.0,
                error_message=f"平台不兼容: {self.hardware_info.platform.value}"
            )
        
        # 检查资源要求
        if self.hardware_info.total_memory < test_case.min_memory_gb:
            return HardwareTestResult(
                test_name=test_case.name,
                platform=self.hardware_info.platform,
                success=False,
                execution_time=0.0,
                performance_metrics={},
                resource_usage={},
                compatibility_score=0.0,
                error_message=f"内存不足: 需要{test_case.min_memory_gb}GB，可用{self.hardware_info.total_memory:.2f}GB"
            )
        
        if self.hardware_info.cpu_count < test_case.min_cpu_cores:
            return HardwareTestResult(
                test_name=test_case.name,
                platform=self.hardware_info.platform,
                success=False,
                execution_time=0.0,
                performance_metrics={},
                resource_usage={},
                compatibility_score=0.0,
                error_message=f"CPU核心不足: 需要{test_case.min_cpu_cores}核，可用{self.hardware_info.cpu_count}核"
            )
        
        start_time = time.time()
        initial_memory = psutil.virtual_memory().used / (1024**3)
        
        try:
            # 执行测试
            performance_metrics = test_case.test_function()
            
            execution_time = time.time() - start_time
            final_memory = psutil.virtual_memory().used / (1024**3)
            memory_usage = final_memory - initial_memory
            
            # 计算兼容性评分
            compatibility_score = self._calculate_compatibility_score(
                test_case, performance_metrics, execution_time
            )
            
            result = HardwareTestResult(
                test_name=test_case.name,
                platform=self.hardware_info.platform,
                success=True,
                execution_time=execution_time,
                performance_metrics=performance_metrics,
                resource_usage={
                    'memory_delta_gb': memory_usage,
                    'cpu_time': execution_time
                },
                compatibility_score=compatibility_score,
                warnings=performance_metrics.get('warnings', [])
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            result = HardwareTestResult(
                test_name=test_case.name,
                platform=self.hardware_info.platform,
                success=False,
                execution_time=execution_time,
                performance_metrics={},
                resource_usage={},
                compatibility_score=0.0,
                error_message=str(e)
            )
        
        self.logger.info(f"硬件测试完成: {test_case.name}, 成功: {result.success}, 兼容性: {result.compatibility_score:.1f}%")
        return result
    
    def _calculate_compatibility_score(self, test_case: HardwareTestCase, 
                                     metrics: Dict[str, Any], execution_time: float) -> float:
        """计算兼容性评分"""
        score = 100.0
        
        # 基于执行时间的评分
        if execution_time > test_case.timeout * 0.8:
            score -= 20
        elif execution_time > test_case.timeout * 0.5:
            score -= 10
        
        # 基于性能指标的评分
        if 'error' in metrics:
            score = 0
        elif 'detection_accuracy' in metrics:
            score = metrics['detection_accuracy']
        elif 'overall_performance_score' in metrics:
            score = min(100, metrics['overall_performance_score'])
        
        # 平台特定调整
        platform_adjustments = {
            HardwarePlatform.K230: -10,      # 嵌入式平台性能较低
            HardwarePlatform.ESP32: -20,     # 微控制器性能最低
            HardwarePlatform.RASPBERRY_PI: -5,  # 单板机性能中等
            HardwarePlatform.X86_64: 0,      # 标准平台
            HardwarePlatform.ARM64: -5       # ARM平台略低
        }
        
        adjustment = platform_adjustments.get(self.hardware_info.platform, -15)
        score += adjustment
        
        return max(0, min(100, score))
    
    def run_all_tests(self) -> List[HardwareTestResult]:
        """运行所有硬件测试"""
        self.logger.info(f"开始运行 {len(self.test_cases)} 个硬件兼容性测试")
        self.logger.info(f"当前平台: {self.hardware_info.platform.value}")
        
        results = []
        
        for test_case in self.test_cases:
            try:
                result = self.execute_test_case(test_case)
                results.append(result)
                self.results.append(result)
            except Exception as e:
                self.logger.error(f"测试执行失败: {test_case.name}, 错误: {e}")
                failed_result = HardwareTestResult(
                    test_name=test_case.name,
                    platform=self.hardware_info.platform,
                    success=False,
                    execution_time=0.0,
                    performance_metrics={},
                    resource_usage={},
                    compatibility_score=0.0,
                    error_message=f"测试执行失败: {str(e)}"
                )
                results.append(failed_result)
                self.results.append(failed_result)
        
        return results
    
    def generate_compatibility_report(self) -> Dict[str, Any]:
        """生成兼容性报告"""
        if not self.results:
            return {'error': '没有测试结果'}
        
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - successful_tests
        
        # 计算平均兼容性评分
        compatibility_scores = [r.compatibility_score for r in self.results if r.success]
        avg_compatibility = np.mean(compatibility_scores) if compatibility_scores else 0
        
        # 按测试类别统计
        category_stats = {}
        for test_case in self.test_cases:
            category = test_case.category.value
            if category not in category_stats:
                category_stats[category] = {'total': 0, 'passed': 0, 'avg_score': 0}
            category_stats[category]['total'] += 1
        
        for result in self.results:
            test_case = next((tc for tc in self.test_cases if tc.name == result.test_name), None)
            if test_case:
                category = test_case.category.value
                if result.success:
                    category_stats[category]['passed'] += 1
                    if category_stats[category]['avg_score'] == 0:
                        category_stats[category]['avg_score'] = result.compatibility_score
                    else:
                        category_stats[category]['avg_score'] = (
                            category_stats[category]['avg_score'] + result.compatibility_score
                        ) / 2
        
        # 性能指标汇总
        performance_summary = {}
        for result in self.results:
            if result.success and result.performance_metrics:
                for key, value in result.performance_metrics.items():
                    if isinstance(value, (int, float)) and key != 'error':
                        if key not in performance_summary:
                            performance_summary[key] = []
                        performance_summary[key].append(value)
        
        # 计算性能指标统计
        perf_stats = {}
        for key, values in performance_summary.items():
            perf_stats[key] = {
                'avg': np.mean(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        report = {
            'hardware_info': {
                'platform': self.hardware_info.platform.value,
                'architecture': self.hardware_info.architecture,
                'cpu_count': self.hardware_info.cpu_count,
                'total_memory_gb': self.hardware_info.total_memory,
                'gpu_available': self.hardware_info.gpu_available
            },
            'test_summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'failed_tests': failed_tests,
                'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
                'avg_compatibility_score': avg_compatibility
            },
            'category_analysis': category_stats,
            'performance_summary': perf_stats,
            'compatibility_rating': self._get_compatibility_rating(avg_compatibility),
            'recommendations': self._generate_platform_recommendations()
        }
        
        return report
    
    def _get_compatibility_rating(self, score: float) -> str:
        """获取兼容性等级"""
        if score >= 90:
            return "优秀 (Excellent)"
        elif score >= 75:
            return "良好 (Good)"
        elif score >= 60:
            return "一般 (Fair)"
        elif score >= 40:
            return "较差 (Poor)"
        else:
            return "不兼容 (Incompatible)"
    
    def _generate_platform_recommendations(self) -> List[str]:
        """生成平台建议"""
        recommendations = []
        
        # 基于平台类型的建议
        platform_advice = {
            HardwarePlatform.K230: [
                "K230平台适合轻量级AI推理任务",
                "建议使用量化模型以提高性能",
                "注意内存使用优化"
            ],
            HardwarePlatform.ESP32: [
                "ESP32适合简单的边缘计算任务",
                "建议使用TinyML模型",
                "考虑使用外部存储扩展"
            ],
            HardwarePlatform.RASPBERRY_PI: [
                "树莓派适合原型开发和教育用途",
                "建议使用GPU加速（如果可用）",
                "考虑散热解决方案"
            ],
            HardwarePlatform.X86_64: [
                "x86_64平台性能优秀，适合复杂AI任务",
                "可以运行完整的YOLO模型",
                "建议使用GPU加速"
            ]
        }
        
        platform_recs = platform_advice.get(self.hardware_info.platform, 
                                           ["通用平台，建议根据具体性能调整配置"])
        recommendations.extend(platform_recs)
        
        # 基于测试结果的建议
        failed_tests = [r for r in self.results if not r.success]
        if failed_tests:
            recommendations.append(f"有{len(failed_tests)}个测试失败，需要检查兼容性问题")
        
        # 基于性能的建议
        avg_score = np.mean([r.compatibility_score for r in self.results if r.success])
        if avg_score < 60:
            recommendations.append("整体兼容性较低，建议升级硬件或优化软件配置")
        
        return recommendations
    
    def save_report(self, filename: str = None) -> str:
        """保存兼容性报告"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            platform_name = self.hardware_info.platform.value
            filename = f'hardware_compatibility_report_{platform_name}_{timestamp}.json'
        
        report = self.generate_compatibility_report()
        
        # 添加详细结果
        detailed_results = []
        for result in self.results:
            detailed_results.append({
                'test_name': result.test_name,
                'platform': result.platform.value,
                'success': result.success,
                'execution_time': result.execution_time,
                'compatibility_score': result.compatibility_score,
                'performance_metrics': result.performance_metrics,
                'resource_usage': result.resource_usage,
                'error_message': result.error_message,
                'warnings': result.warnings,
                'timestamp': result.timestamp.isoformat()
            })
        
        report['detailed_results'] = detailed_results
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        return filename
    
    def print_summary(self):
        """打印测试摘要"""
        if not self.results:
            print("没有测试结果")
            return
        
        report = self.generate_compatibility_report()
        
        print("\n" + "="*80)
        print("YOLOS硬件平台兼容性测试报告")
        print("="*80)
        
        # 硬件信息
        hw_info = report['hardware_info']
        print(f"\n🖥️  硬件平台信息:")
        print(f"   平台类型: {hw_info['platform']}")
        print(f"   架构: {hw_info['architecture']}")
        print(f"   CPU核心数: {hw_info['cpu_count']}")
        print(f"   总内存: {hw_info['total_memory_gb']:.2f}GB")
        print(f"   GPU可用: {'是' if hw_info['gpu_available'] else '否'}")
        
        # 测试摘要
        summary = report['test_summary']
        print(f"\n📊 测试摘要:")
        print(f"   总测试数: {summary['total_tests']}")
        print(f"   成功测试: {summary['successful_tests']}")
        print(f"   失败测试: {summary['failed_tests']}")
        print(f"   成功率: {summary['success_rate']:.1%}")
        print(f"   平均兼容性评分: {summary['avg_compatibility_score']:.1f}/100")
        print(f"   兼容性等级: {report['compatibility_rating']}")
        
        # 类别分析
        print(f"\n🔍 测试类别分析:")
        for category, stats in report['category_analysis'].items():
            success_rate = stats['passed'] / stats['total'] if stats['total'] > 0 else 0
            print(f"   {category}: {stats['passed']}/{stats['total']} ({success_rate:.1%}) - 平均评分: {stats['avg_score']:.1f}")
        
        # 建议
        print(f"\n💡 平台建议:")
        for rec in report['recommendations']:
            print(f"   - {rec}")
        
        print("\n" + "="*80)

def main():
    """主函数"""
    print("YOLOS硬件平台兼容性测试器")
    print("作为资深AIoT测试专家，执行全面的硬件兼容性测试")
    
    tester = HardwareCompatibilityTester()
    
    try:
        # 显示当前硬件信息
        hw_info = tester.hardware_info
        print(f"\n检测到硬件平台: {hw_info.platform.value}")
        print(f"架构: {hw_info.architecture}, CPU: {hw_info.cpu_count}核, 内存: {hw_info.total_memory:.2f}GB")
        
        # 运行所有测试
        print(f"\n开始执行 {len(tester.test_cases)} 个硬件兼容性测试...")
        results = tester.run_all_tests()
        
        # 打印摘要
        tester.print_summary()
        
        # 保存报告
        report_file = tester.save_report()
        print(f"\n详细报告已保存到: {report_file}")
        
        # 返回兼容性评分作为退出码
        avg_score = np.mean([r.compatibility_score for r in results if r.success])
        return 0 if avg_score >= 60 else 1  # 60分以上认为兼容
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        return 1
    except Exception as e:
        print(f"\n测试执行失败: {e}")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)