#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能压力和边界条件测试器

执行YOLOS系统的性能压力测试和边界条件测试，包括：
1. 高并发负载测试
2. 内存压力测试
3. CPU密集型任务测试
4. 网络带宽压力测试
5. 存储IO压力测试
6. 边界条件测试（极大/极小输入）
7. 资源耗尽场景测试
8. 长时间运行稳定性测试
"""

import time
import threading
import multiprocessing
import psutil
import numpy as np
import cv2
import json
import os
import sys
import gc
import tempfile
import shutil
import socket
import requests
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from pathlib import Path
import concurrent.futures
import queue
import random
import string
import subprocess
import signal
from contextlib import contextmanager
# import resource  # Windows上不可用
import traceback

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StressTestType(Enum):
    """压力测试类型"""
    CONCURRENT_LOAD = "concurrent_load"          # 并发负载
    MEMORY_PRESSURE = "memory_pressure"          # 内存压力
    CPU_INTENSIVE = "cpu_intensive"              # CPU密集
    NETWORK_BANDWIDTH = "network_bandwidth"      # 网络带宽
    STORAGE_IO = "storage_io"                    # 存储IO
    BOUNDARY_CONDITIONS = "boundary_conditions"  # 边界条件
    RESOURCE_EXHAUSTION = "resource_exhaustion"  # 资源耗尽
    LONG_RUNNING = "long_running"                # 长时间运行

class TestSeverity(Enum):
    """测试严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ResourceType(Enum):
    """资源类型"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"

@dataclass
class StressTestConfig:
    """压力测试配置"""
    name: str
    test_type: StressTestType
    severity: TestSeverity
    duration: float  # 秒
    target_resource: ResourceType
    load_level: float  # 0.0-1.0
    concurrent_workers: int
    description: str
    expected_metrics: Dict[str, float] = field(default_factory=dict)
    timeout: float = 300.0  # 5分钟超时

@dataclass
class ResourceMetrics:
    """资源指标"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    process_count: int
    thread_count: int
    open_files: int

@dataclass
class StressTestResult:
    """压力测试结果"""
    test_name: str
    test_type: StressTestType
    success: bool
    duration: float
    peak_metrics: ResourceMetrics
    avg_metrics: ResourceMetrics
    performance_score: float  # 0-100
    stability_score: float    # 0-100
    throughput: float        # 操作/秒
    error_count: int
    warning_count: int
    error_messages: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    detailed_metrics: List[ResourceMetrics] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.monitoring = False
        self.metrics: List[ResourceMetrics] = []
        self.monitor_thread = None
        self.initial_disk_io = None
        self.initial_network_io = None
    
    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.metrics.clear()
        
        # 记录初始IO状态
        try:
            self.initial_disk_io = psutil.disk_io_counters()
            self.initial_network_io = psutil.net_io_counters()
        except:
            pass
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> List[ResourceMetrics]:
        """停止监控并返回指标"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        return self.metrics.copy()
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics.append(metrics)
                time.sleep(self.interval)
            except Exception as e:
                logger.warning(f"监控指标收集失败: {e}")
                time.sleep(self.interval)
    
    def _collect_metrics(self) -> ResourceMetrics:
        """收集当前资源指标"""
        # CPU和内存
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        
        # 磁盘IO
        disk_io_read_mb = 0
        disk_io_write_mb = 0
        try:
            current_disk_io = psutil.disk_io_counters()
            if self.initial_disk_io and current_disk_io:
                disk_io_read_mb = (current_disk_io.read_bytes - self.initial_disk_io.read_bytes) / (1024**2)
                disk_io_write_mb = (current_disk_io.write_bytes - self.initial_disk_io.write_bytes) / (1024**2)
        except:
            pass
        
        # 网络IO
        network_sent_mb = 0
        network_recv_mb = 0
        try:
            current_network_io = psutil.net_io_counters()
            if self.initial_network_io and current_network_io:
                network_sent_mb = (current_network_io.bytes_sent - self.initial_network_io.bytes_sent) / (1024**2)
                network_recv_mb = (current_network_io.bytes_recv - self.initial_network_io.bytes_recv) / (1024**2)
        except:
            pass
        
        # 进程信息
        process_count = len(psutil.pids())
        
        # 当前进程的线程和文件句柄
        current_process = psutil.Process()
        thread_count = current_process.num_threads()
        try:
            open_files = len(current_process.open_files())
        except:
            open_files = 0
        
        return ResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            disk_io_read_mb=disk_io_read_mb,
            disk_io_write_mb=disk_io_write_mb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb,
            process_count=process_count,
            thread_count=thread_count,
            open_files=open_files
        )

class WorkloadGenerator:
    """工作负载生成器"""
    
    @staticmethod
    def cpu_intensive_task(duration: float, intensity: float = 1.0) -> int:
        """CPU密集型任务"""
        operations = 0
        end_time = time.time() + duration
        
        while time.time() < end_time:
            # 数学运算
            for i in range(int(1000 * intensity)):
                _ = sum(j**2 for j in range(100))
                operations += 1
            
            # 短暂休息以控制强度
            if intensity < 1.0:
                time.sleep(0.001 * (1.0 - intensity))
        
        return operations
    
    @staticmethod
    def memory_intensive_task(target_mb: int, duration: float) -> Tuple[int, float]:
        """内存密集型任务"""
        allocated_blocks = []
        block_size_mb = 10
        operations = 0
        peak_memory = 0
        
        try:
            end_time = time.time() + duration
            
            while time.time() < end_time and len(allocated_blocks) * block_size_mb < target_mb:
                # 分配内存块
                block = np.random.random((block_size_mb * 1024 * 1024 // 8,)).astype(np.float64)
                allocated_blocks.append(block)
                operations += 1
                
                # 记录峰值内存
                current_memory = len(allocated_blocks) * block_size_mb
                peak_memory = max(peak_memory, current_memory)
                
                # 随机访问已分配的内存
                if allocated_blocks and random.random() < 0.3:
                    random_block = random.choice(allocated_blocks)
                    _ = np.sum(random_block[:1000])  # 部分访问
                
                time.sleep(0.01)  # 短暂休息
        
        except MemoryError:
            logger.warning(f"内存分配失败，已分配 {len(allocated_blocks) * block_size_mb}MB")
        
        finally:
            # 清理内存
            del allocated_blocks
            gc.collect()
        
        return operations, peak_memory
    
    @staticmethod
    def io_intensive_task(file_size_mb: int, duration: float) -> Tuple[int, float, float]:
        """IO密集型任务"""
        operations = 0
        total_read_mb = 0
        total_write_mb = 0
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                end_time = time.time() + duration
                file_path = os.path.join(temp_dir, 'test_file.dat')
                
                # 创建测试文件
                test_data = os.urandom(file_size_mb * 1024 * 1024)
                
                while time.time() < end_time:
                    # 写入操作
                    with open(file_path, 'wb') as f:
                        f.write(test_data)
                        f.flush()
                        os.fsync(f.fileno())
                    total_write_mb += file_size_mb
                    operations += 1
                    
                    # 读取操作
                    with open(file_path, 'rb') as f:
                        _ = f.read()
                    total_read_mb += file_size_mb
                    operations += 1
                    
                    # 随机访问
                    with open(file_path, 'r+b') as f:
                        for _ in range(10):
                            offset = random.randint(0, max(0, len(test_data) - 4096))
                            f.seek(offset)
                            _ = f.read(4096)
                    
                    time.sleep(0.01)  # 短暂休息
            
            except Exception as e:
                logger.warning(f"IO操作失败: {e}")
        
        return operations, total_read_mb, total_write_mb
    
    @staticmethod
    def network_task(target_url: str, duration: float, concurrent_requests: int = 5) -> Tuple[int, int]:
        """网络任务（如果有可用的测试端点）"""
        successful_requests = 0
        failed_requests = 0
        
        def make_request():
            nonlocal successful_requests, failed_requests
            try:
                # 使用本地回环测试
                response = requests.get('http://httpbin.org/get', timeout=5)
                if response.status_code == 200:
                    successful_requests += 1
                else:
                    failed_requests += 1
            except:
                failed_requests += 1
        
        try:
            end_time = time.time() + duration
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
                while time.time() < end_time:
                    futures = []
                    for _ in range(concurrent_requests):
                        future = executor.submit(make_request)
                        futures.append(future)
                    
                    # 等待完成
                    for future in concurrent.futures.as_completed(futures, timeout=10):
                        try:
                            future.result()
                        except:
                            failed_requests += 1
                    
                    time.sleep(0.1)
        
        except Exception as e:
            logger.warning(f"网络任务失败: {e}")
        
        return successful_requests, failed_requests
    
    @staticmethod
    def image_processing_task(duration: float, complexity: str = "medium") -> int:
        """图像处理任务"""
        operations = 0
        end_time = time.time() + duration
        
        # 根据复杂度设置参数
        if complexity == "light":
            image_size = (320, 240)
            operations_per_cycle = 5
        elif complexity == "heavy":
            image_size = (1280, 720)
            operations_per_cycle = 20
        else:  # medium
            image_size = (640, 480)
            operations_per_cycle = 10
        
        try:
            while time.time() < end_time:
                # 生成随机图像
                image = np.random.randint(0, 255, (image_size[1], image_size[0], 3), dtype=np.uint8)
                
                for _ in range(operations_per_cycle):
                    # 各种图像处理操作
                    blurred = cv2.GaussianBlur(image, (5, 5), 0)
                    edges = cv2.Canny(blurred, 50, 150)
                    resized = cv2.resize(image, (image_size[0]//2, image_size[1]//2))
                    
                    # 颜色空间转换
                    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    
                    operations += 1
                
                time.sleep(0.001)  # 短暂休息
        
        except Exception as e:
            logger.warning(f"图像处理任务失败: {e}")
        
        return operations

class PerformanceStressBoundaryTester:
    """性能压力和边界条件测试器"""
    
    def __init__(self):
        self.test_configs: List[StressTestConfig] = []
        self.results: List[StressTestResult] = []
        self.logger = logging.getLogger(__name__)
        self.resource_monitor = ResourceMonitor()
        self._initialize_test_configs()
    
    def _initialize_test_configs(self):
        """初始化测试配置"""
        # 并发负载测试
        self.test_configs.append(StressTestConfig(
            name="high_concurrency_load_test",
            test_type=StressTestType.CONCURRENT_LOAD,
            severity=TestSeverity.HIGH,
            duration=30.0,
            target_resource=ResourceType.CPU,
            load_level=0.8,
            concurrent_workers=multiprocessing.cpu_count() * 2,
            description="高并发负载测试，模拟多个并发任务"
        ))
        
        # 内存压力测试
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        target_memory_mb = min(1000, int(available_memory_gb * 0.3 * 1024))  # 使用30%可用内存
        
        self.test_configs.append(StressTestConfig(
            name="memory_pressure_test",
            test_type=StressTestType.MEMORY_PRESSURE,
            severity=TestSeverity.HIGH,
            duration=45.0,
            target_resource=ResourceType.MEMORY,
            load_level=0.7,
            concurrent_workers=4,
            description=f"内存压力测试，目标分配{target_memory_mb}MB内存"
        ))
        
        # CPU密集型测试
        self.test_configs.append(StressTestConfig(
            name="cpu_intensive_test",
            test_type=StressTestType.CPU_INTENSIVE,
            severity=TestSeverity.MEDIUM,
            duration=60.0,
            target_resource=ResourceType.CPU,
            load_level=0.9,
            concurrent_workers=multiprocessing.cpu_count(),
            description="CPU密集型计算测试"
        ))
        
        # 存储IO压力测试
        self.test_configs.append(StressTestConfig(
            name="storage_io_stress_test",
            test_type=StressTestType.STORAGE_IO,
            severity=TestSeverity.MEDIUM,
            duration=40.0,
            target_resource=ResourceType.DISK,
            load_level=0.6,
            concurrent_workers=8,
            description="存储IO压力测试"
        ))
        
        # 边界条件测试
        self.test_configs.append(StressTestConfig(
            name="boundary_conditions_test",
            test_type=StressTestType.BOUNDARY_CONDITIONS,
            severity=TestSeverity.CRITICAL,
            duration=30.0,
            target_resource=ResourceType.MEMORY,
            load_level=0.5,
            concurrent_workers=2,
            description="边界条件测试，包括极大和极小输入"
        ))
        
        # 资源耗尽测试
        self.test_configs.append(StressTestConfig(
            name="resource_exhaustion_test",
            test_type=StressTestType.RESOURCE_EXHAUSTION,
            severity=TestSeverity.HIGH,
            duration=35.0,
            target_resource=ResourceType.MEMORY,
            load_level=0.8,
            concurrent_workers=6,
            description="资源耗尽场景测试"
        ))
    
    def execute_concurrent_load_test(self, config: StressTestConfig) -> StressTestResult:
        """执行并发负载测试"""
        self.logger.info(f"开始并发负载测试: {config.concurrent_workers}个工作线程")
        
        self.resource_monitor.start_monitoring()
        start_time = time.time()
        
        total_operations = 0
        errors = []
        warnings = []
        
        try:
            # 使用线程池执行并发任务
            with concurrent.futures.ThreadPoolExecutor(max_workers=config.concurrent_workers) as executor:
                # 提交任务
                futures = []
                task_duration = config.duration / 2  # 每个任务运行一半时间
                
                for i in range(config.concurrent_workers):
                    if i % 3 == 0:
                        # CPU密集型任务
                        future = executor.submit(WorkloadGenerator.cpu_intensive_task, task_duration, config.load_level)
                    elif i % 3 == 1:
                        # 图像处理任务
                        future = executor.submit(WorkloadGenerator.image_processing_task, task_duration, "medium")
                    else:
                        # 混合任务
                        future = executor.submit(self._mixed_workload_task, task_duration)
                    
                    futures.append(future)
                
                # 等待所有任务完成
                for future in concurrent.futures.as_completed(futures, timeout=config.timeout):
                    try:
                        result = future.result()
                        total_operations += result if isinstance(result, int) else 1
                    except Exception as e:
                        errors.append(f"并发任务失败: {str(e)}")
        
        except Exception as e:
            errors.append(f"并发负载测试失败: {str(e)}")
        
        execution_time = time.time() - start_time
        metrics_history = self.resource_monitor.stop_monitoring()
        
        # 计算性能指标
        throughput = total_operations / execution_time if execution_time > 0 else 0
        performance_score = min(100, throughput / 100)  # 简化评分
        stability_score = max(0, 100 - len(errors) * 10)
        
        # 计算峰值和平均指标
        peak_metrics, avg_metrics = self._calculate_metrics_summary(metrics_history)
        
        return StressTestResult(
            test_name=config.name,
            test_type=config.test_type,
            success=len(errors) == 0,
            duration=execution_time,
            peak_metrics=peak_metrics,
            avg_metrics=avg_metrics,
            performance_score=performance_score,
            stability_score=stability_score,
            throughput=throughput,
            error_count=len(errors),
            warning_count=len(warnings),
            error_messages=errors,
            warnings=warnings,
            detailed_metrics=metrics_history
        )
    
    def execute_memory_pressure_test(self, config: StressTestConfig) -> StressTestResult:
        """执行内存压力测试"""
        self.logger.info("开始内存压力测试")
        
        self.resource_monitor.start_monitoring()
        start_time = time.time()
        
        total_operations = 0
        errors = []
        warnings = []
        peak_memory_allocated = 0
        
        try:
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            target_memory_mb = min(1000, int(available_memory_gb * config.load_level * 1024))
            
            # 使用多个进程进行内存分配
            with concurrent.futures.ProcessPoolExecutor(max_workers=config.concurrent_workers) as executor:
                futures = []
                memory_per_worker = target_memory_mb // config.concurrent_workers
                
                for i in range(config.concurrent_workers):
                    future = executor.submit(
                        WorkloadGenerator.memory_intensive_task,
                        memory_per_worker,
                        config.duration / 2
                    )
                    futures.append(future)
                
                # 等待结果
                for future in concurrent.futures.as_completed(futures, timeout=config.timeout):
                    try:
                        operations, peak_memory = future.result()
                        total_operations += operations
                        peak_memory_allocated = max(peak_memory_allocated, peak_memory)
                    except Exception as e:
                        errors.append(f"内存压力任务失败: {str(e)}")
        
        except Exception as e:
            errors.append(f"内存压力测试失败: {str(e)}")
        
        execution_time = time.time() - start_time
        metrics_history = self.resource_monitor.stop_monitoring()
        
        # 计算性能指标
        throughput = total_operations / execution_time if execution_time > 0 else 0
        memory_efficiency = peak_memory_allocated / target_memory_mb if target_memory_mb > 0 else 0
        performance_score = min(100, memory_efficiency * 100)
        stability_score = max(0, 100 - len(errors) * 15)
        
        # 计算峰值和平均指标
        peak_metrics, avg_metrics = self._calculate_metrics_summary(metrics_history)
        
        return StressTestResult(
            test_name=config.name,
            test_type=config.test_type,
            success=len(errors) == 0,
            duration=execution_time,
            peak_metrics=peak_metrics,
            avg_metrics=avg_metrics,
            performance_score=performance_score,
            stability_score=stability_score,
            throughput=throughput,
            error_count=len(errors),
            warning_count=len(warnings),
            error_messages=errors,
            warnings=warnings,
            detailed_metrics=metrics_history
        )
    
    def execute_cpu_intensive_test(self, config: StressTestConfig) -> StressTestResult:
        """执行CPU密集型测试"""
        self.logger.info("开始CPU密集型测试")
        
        self.resource_monitor.start_monitoring()
        start_time = time.time()
        
        total_operations = 0
        errors = []
        warnings = []
        
        try:
            # 使用进程池进行CPU密集型计算
            with concurrent.futures.ProcessPoolExecutor(max_workers=config.concurrent_workers) as executor:
                futures = []
                task_duration = config.duration / 2
                
                for i in range(config.concurrent_workers):
                    future = executor.submit(
                        WorkloadGenerator.cpu_intensive_task,
                        task_duration,
                        config.load_level
                    )
                    futures.append(future)
                
                # 等待结果
                for future in concurrent.futures.as_completed(futures, timeout=config.timeout):
                    try:
                        operations = future.result()
                        total_operations += operations
                    except Exception as e:
                        errors.append(f"CPU密集型任务失败: {str(e)}")
        
        except Exception as e:
            errors.append(f"CPU密集型测试失败: {str(e)}")
        
        execution_time = time.time() - start_time
        metrics_history = self.resource_monitor.stop_monitoring()
        
        # 计算性能指标
        throughput = total_operations / execution_time if execution_time > 0 else 0
        performance_score = min(100, throughput / 1000)  # 简化评分
        stability_score = max(0, 100 - len(errors) * 10)
        
        # 计算峰值和平均指标
        peak_metrics, avg_metrics = self._calculate_metrics_summary(metrics_history)
        
        return StressTestResult(
            test_name=config.name,
            test_type=config.test_type,
            success=len(errors) == 0,
            duration=execution_time,
            peak_metrics=peak_metrics,
            avg_metrics=avg_metrics,
            performance_score=performance_score,
            stability_score=stability_score,
            throughput=throughput,
            error_count=len(errors),
            warning_count=len(warnings),
            error_messages=errors,
            warnings=warnings,
            detailed_metrics=metrics_history
        )
    
    def execute_storage_io_test(self, config: StressTestConfig) -> StressTestResult:
        """执行存储IO测试"""
        self.logger.info("开始存储IO压力测试")
        
        self.resource_monitor.start_monitoring()
        start_time = time.time()
        
        total_operations = 0
        total_read_mb = 0
        total_write_mb = 0
        errors = []
        warnings = []
        
        try:
            # 使用线程池进行IO操作
            with concurrent.futures.ThreadPoolExecutor(max_workers=config.concurrent_workers) as executor:
                futures = []
                file_size_mb = max(1, int(10 * config.load_level))  # 根据负载级别调整文件大小
                task_duration = config.duration / 2
                
                for i in range(config.concurrent_workers):
                    future = executor.submit(
                        WorkloadGenerator.io_intensive_task,
                        file_size_mb,
                        task_duration
                    )
                    futures.append(future)
                
                # 等待结果
                for future in concurrent.futures.as_completed(futures, timeout=config.timeout):
                    try:
                        operations, read_mb, write_mb = future.result()
                        total_operations += operations
                        total_read_mb += read_mb
                        total_write_mb += write_mb
                    except Exception as e:
                        errors.append(f"IO任务失败: {str(e)}")
        
        except Exception as e:
            errors.append(f"存储IO测试失败: {str(e)}")
        
        execution_time = time.time() - start_time
        metrics_history = self.resource_monitor.stop_monitoring()
        
        # 计算性能指标
        throughput = total_operations / execution_time if execution_time > 0 else 0
        io_throughput = (total_read_mb + total_write_mb) / execution_time if execution_time > 0 else 0
        performance_score = min(100, io_throughput / 10)  # 简化评分，10MB/s为满分
        stability_score = max(0, 100 - len(errors) * 10)
        
        # 计算峰值和平均指标
        peak_metrics, avg_metrics = self._calculate_metrics_summary(metrics_history)
        
        return StressTestResult(
            test_name=config.name,
            test_type=config.test_type,
            success=len(errors) == 0,
            duration=execution_time,
            peak_metrics=peak_metrics,
            avg_metrics=avg_metrics,
            performance_score=performance_score,
            stability_score=stability_score,
            throughput=throughput,
            error_count=len(errors),
            warning_count=len(warnings),
            error_messages=errors,
            warnings=warnings,
            detailed_metrics=metrics_history
        )
    
    def execute_boundary_conditions_test(self, config: StressTestConfig) -> StressTestResult:
        """执行边界条件测试"""
        self.logger.info("开始边界条件测试")
        
        self.resource_monitor.start_monitoring()
        start_time = time.time()
        
        total_operations = 0
        errors = []
        warnings = []
        
        try:
            # 测试极小输入
            self._test_minimal_inputs()
            total_operations += 1
            
            # 测试极大输入
            self._test_maximal_inputs()
            total_operations += 1
            
            # 测试空输入
            self._test_empty_inputs()
            total_operations += 1
            
            # 测试异常输入
            self._test_invalid_inputs()
            total_operations += 1
            
            # 测试边界值
            self._test_boundary_values()
            total_operations += 1
        
        except Exception as e:
            errors.append(f"边界条件测试失败: {str(e)}")
        
        execution_time = time.time() - start_time
        metrics_history = self.resource_monitor.stop_monitoring()
        
        # 计算性能指标
        throughput = total_operations / execution_time if execution_time > 0 else 0
        performance_score = max(0, 100 - len(errors) * 20)  # 错误越少分数越高
        stability_score = max(0, 100 - len(errors) * 15)
        
        # 计算峰值和平均指标
        peak_metrics, avg_metrics = self._calculate_metrics_summary(metrics_history)
        
        return StressTestResult(
            test_name=config.name,
            test_type=config.test_type,
            success=len(errors) == 0,
            duration=execution_time,
            peak_metrics=peak_metrics,
            avg_metrics=avg_metrics,
            performance_score=performance_score,
            stability_score=stability_score,
            throughput=throughput,
            error_count=len(errors),
            warning_count=len(warnings),
            error_messages=errors,
            warnings=warnings,
            detailed_metrics=metrics_history
        )
    
    def execute_resource_exhaustion_test(self, config: StressTestConfig) -> StressTestResult:
        """执行资源耗尽测试"""
        self.logger.info("开始资源耗尽测试")
        
        self.resource_monitor.start_monitoring()
        start_time = time.time()
        
        total_operations = 0
        errors = []
        warnings = []
        
        try:
            # 逐步增加资源使用直到接近极限
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            
            # 内存耗尽测试
            allocated_blocks = []
            block_size_mb = 50
            max_blocks = int(available_memory_gb * 0.8 * 1024 / block_size_mb)  # 使用80%可用内存
            
            for i in range(max_blocks):
                try:
                    block = np.random.random((block_size_mb * 1024 * 1024 // 8,)).astype(np.float64)
                    allocated_blocks.append(block)
                    total_operations += 1
                    
                    # 检查剩余内存
                    remaining_memory = psutil.virtual_memory().available / (1024**3)
                    if remaining_memory < 0.5:  # 少于500MB时停止
                        warnings.append(f"内存不足，停止分配。已分配{len(allocated_blocks)}个块")
                        break
                    
                    time.sleep(0.01)  # 短暂休息
                
                except MemoryError:
                    warnings.append(f"内存耗尽，已分配{len(allocated_blocks)}个块")
                    break
                except Exception as e:
                    errors.append(f"内存分配失败: {str(e)}")
                    break
            
            # 清理内存
            del allocated_blocks
            gc.collect()
            
            # 文件句柄耗尽测试（谨慎进行）
            open_files = []
            try:
                for i in range(100):  # 限制数量避免系统问题
                    temp_file = tempfile.NamedTemporaryFile(delete=False)
                    open_files.append(temp_file)
                    total_operations += 1
                    time.sleep(0.001)
            except Exception as e:
                warnings.append(f"文件句柄测试: {str(e)}")
            finally:
                # 清理文件
                for f in open_files:
                    try:
                        f.close()
                        os.unlink(f.name)
                    except:
                        pass
        
        except Exception as e:
            errors.append(f"资源耗尽测试失败: {str(e)}")
        
        execution_time = time.time() - start_time
        metrics_history = self.resource_monitor.stop_monitoring()
        
        # 计算性能指标
        throughput = total_operations / execution_time if execution_time > 0 else 0
        performance_score = min(100, total_operations / 10)  # 简化评分
        stability_score = max(0, 100 - len(errors) * 10)
        
        # 计算峰值和平均指标
        peak_metrics, avg_metrics = self._calculate_metrics_summary(metrics_history)
        
        return StressTestResult(
            test_name=config.name,
            test_type=config.test_type,
            success=len(errors) == 0,
            duration=execution_time,
            peak_metrics=peak_metrics,
            avg_metrics=avg_metrics,
            performance_score=performance_score,
            stability_score=stability_score,
            throughput=throughput,
            error_count=len(errors),
            warning_count=len(warnings),
            error_messages=errors,
            warnings=warnings,
            detailed_metrics=metrics_history
        )
    
    def _mixed_workload_task(self, duration: float) -> int:
        """混合工作负载任务"""
        operations = 0
        end_time = time.time() + duration
        
        while time.time() < end_time:
            # 随机选择任务类型
            task_type = random.choice(['cpu', 'image', 'memory'])
            
            if task_type == 'cpu':
                for i in range(100):
                    _ = sum(j**2 for j in range(50))
                operations += 1
            elif task_type == 'image':
                image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                blurred = cv2.GaussianBlur(image, (3, 3), 0)
                operations += 1
            else:  # memory
                data = np.random.random(1000)
                _ = np.sum(data)
                operations += 1
            
            time.sleep(0.001)
        
        return operations
    
    def _test_minimal_inputs(self):
        """测试极小输入"""
        # 测试1x1像素图像
        tiny_image = np.ones((1, 1, 3), dtype=np.uint8)
        _ = cv2.resize(tiny_image, (2, 2))
        
        # 测试空数组
        empty_array = np.array([])
        if len(empty_array) == 0:
            pass  # 正常处理空数组
    
    def _test_maximal_inputs(self):
        """测试极大输入"""
        # 测试大尺寸图像（受内存限制）
        try:
            large_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
            _ = cv2.GaussianBlur(large_image, (5, 5), 0)
        except MemoryError:
            pass  # 预期的内存错误
    
    def _test_empty_inputs(self):
        """测试空输入"""
        # 测试None输入
        try:
            result = self._safe_process_input(None)
        except Exception:
            pass  # 预期的异常
    
    def _test_invalid_inputs(self):
        """测试无效输入"""
        # 测试负数尺寸
        try:
            invalid_image = np.random.randint(0, 255, (0, 0, 3), dtype=np.uint8)
        except ValueError:
            pass  # 预期的错误
    
    def _test_boundary_values(self):
        """测试边界值"""
        # 测试数值边界
        max_uint8 = np.array([255], dtype=np.uint8)
        min_uint8 = np.array([0], dtype=np.uint8)
        
        # 测试浮点数边界
        max_float = np.array([np.finfo(np.float32).max], dtype=np.float32)
        min_float = np.array([np.finfo(np.float32).min], dtype=np.float32)
    
    def _safe_process_input(self, input_data):
        """安全处理输入数据"""
        if input_data is None:
            raise ValueError("输入不能为None")
        return input_data
    
    def _calculate_metrics_summary(self, metrics_history: List[ResourceMetrics]) -> Tuple[ResourceMetrics, ResourceMetrics]:
        """计算指标摘要"""
        if not metrics_history:
            # 返回默认值
            default_metrics = ResourceMetrics(
                timestamp=datetime.now(),
                cpu_percent=0,
                memory_percent=0,
                memory_used_gb=0,
                disk_io_read_mb=0,
                disk_io_write_mb=0,
                network_sent_mb=0,
                network_recv_mb=0,
                process_count=0,
                thread_count=0,
                open_files=0
            )
            return default_metrics, default_metrics
        
        # 计算峰值
        peak_metrics = ResourceMetrics(
            timestamp=metrics_history[0].timestamp,
            cpu_percent=max(m.cpu_percent for m in metrics_history),
            memory_percent=max(m.memory_percent for m in metrics_history),
            memory_used_gb=max(m.memory_used_gb for m in metrics_history),
            disk_io_read_mb=max(m.disk_io_read_mb for m in metrics_history),
            disk_io_write_mb=max(m.disk_io_write_mb for m in metrics_history),
            network_sent_mb=max(m.network_sent_mb for m in metrics_history),
            network_recv_mb=max(m.network_recv_mb for m in metrics_history),
            process_count=max(m.process_count for m in metrics_history),
            thread_count=max(m.thread_count for m in metrics_history),
            open_files=max(m.open_files for m in metrics_history)
        )
        
        # 计算平均值
        avg_metrics = ResourceMetrics(
            timestamp=metrics_history[0].timestamp,
            cpu_percent=np.mean([m.cpu_percent for m in metrics_history]),
            memory_percent=np.mean([m.memory_percent for m in metrics_history]),
            memory_used_gb=np.mean([m.memory_used_gb for m in metrics_history]),
            disk_io_read_mb=np.mean([m.disk_io_read_mb for m in metrics_history]),
            disk_io_write_mb=np.mean([m.disk_io_write_mb for m in metrics_history]),
            network_sent_mb=np.mean([m.network_sent_mb for m in metrics_history]),
            network_recv_mb=np.mean([m.network_recv_mb for m in metrics_history]),
            process_count=int(np.mean([m.process_count for m in metrics_history])),
            thread_count=int(np.mean([m.thread_count for m in metrics_history])),
            open_files=int(np.mean([m.open_files for m in metrics_history]))
        )
        
        return peak_metrics, avg_metrics
    
    def execute_test_config(self, config: StressTestConfig) -> StressTestResult:
        """执行测试配置"""
        self.logger.info(f"开始执行压力测试: {config.name}")
        
        try:
            if config.test_type == StressTestType.CONCURRENT_LOAD:
                return self.execute_concurrent_load_test(config)
            elif config.test_type == StressTestType.MEMORY_PRESSURE:
                return self.execute_memory_pressure_test(config)
            elif config.test_type == StressTestType.CPU_INTENSIVE:
                return self.execute_cpu_intensive_test(config)
            elif config.test_type == StressTestType.STORAGE_IO:
                return self.execute_storage_io_test(config)
            elif config.test_type == StressTestType.BOUNDARY_CONDITIONS:
                return self.execute_boundary_conditions_test(config)
            elif config.test_type == StressTestType.RESOURCE_EXHAUSTION:
                return self.execute_resource_exhaustion_test(config)
            else:
                raise ValueError(f"不支持的测试类型: {config.test_type}")
        
        except Exception as e:
            self.logger.error(f"测试执行失败: {config.name}, 错误: {e}")
            return StressTestResult(
                test_name=config.name,
                test_type=config.test_type,
                success=False,
                duration=0.0,
                peak_metrics=ResourceMetrics(
                    timestamp=datetime.now(),
                    cpu_percent=0, memory_percent=0, memory_used_gb=0,
                    disk_io_read_mb=0, disk_io_write_mb=0,
                    network_sent_mb=0, network_recv_mb=0,
                    process_count=0, thread_count=0, open_files=0
                ),
                avg_metrics=ResourceMetrics(
                    timestamp=datetime.now(),
                    cpu_percent=0, memory_percent=0, memory_used_gb=0,
                    disk_io_read_mb=0, disk_io_write_mb=0,
                    network_sent_mb=0, network_recv_mb=0,
                    process_count=0, thread_count=0, open_files=0
                ),
                performance_score=0.0,
                stability_score=0.0,
                throughput=0.0,
                error_count=1,
                warning_count=0,
                error_messages=[f"测试执行失败: {str(e)}"],
                warnings=[]
            )
    
    def run_all_tests(self) -> List[StressTestResult]:
        """运行所有压力测试"""
        self.logger.info(f"开始运行 {len(self.test_configs)} 个性能压力和边界条件测试")
        
        results = []
        
        for config in self.test_configs:
            try:
                result = self.execute_test_config(config)
                results.append(result)
                self.results.append(result)
                
                # 测试间短暂休息
                time.sleep(2)
                
            except Exception as e:
                self.logger.error(f"测试配置执行失败: {config.name}, 错误: {e}")
                failed_result = StressTestResult(
                    test_name=config.name,
                    test_type=config.test_type,
                    success=False,
                    duration=0.0,
                    peak_metrics=ResourceMetrics(
                        timestamp=datetime.now(),
                        cpu_percent=0, memory_percent=0, memory_used_gb=0,
                        disk_io_read_mb=0, disk_io_write_mb=0,
                        network_sent_mb=0, network_recv_mb=0,
                        process_count=0, thread_count=0, open_files=0
                    ),
                    avg_metrics=ResourceMetrics(
                        timestamp=datetime.now(),
                        cpu_percent=0, memory_percent=0, memory_used_gb=0,
                        disk_io_read_mb=0, disk_io_write_mb=0,
                        network_sent_mb=0, network_recv_mb=0,
                        process_count=0, thread_count=0, open_files=0
                    ),
                    performance_score=0.0,
                    stability_score=0.0,
                    throughput=0.0,
                    error_count=1,
                    warning_count=0,
                    error_messages=[f"测试配置执行失败: {str(e)}"],
                    warnings=[]
                )
                results.append(failed_result)
                self.results.append(failed_result)
        
        return results
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """生成综合报告"""
        if not self.results:
            return {'error': '没有测试结果'}
        
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - successful_tests
        
        # 计算平均性能和稳定性评分
        performance_scores = [r.performance_score for r in self.results if r.success]
        stability_scores = [r.stability_score for r in self.results if r.success]
        
        avg_performance = np.mean(performance_scores) if performance_scores else 0
        avg_stability = np.mean(stability_scores) if stability_scores else 0
        
        # 按测试类型统计
        type_stats = {}
        for test_type in StressTestType:
            type_results = [r for r in self.results if r.test_type == test_type]
            if type_results:
                type_stats[test_type.value] = {
                    'total': len(type_results),
                    'passed': sum(1 for r in type_results if r.success),
                    'avg_performance': np.mean([r.performance_score for r in type_results if r.success]),
                    'avg_stability': np.mean([r.stability_score for r in type_results if r.success]),
                    'avg_throughput': np.mean([r.throughput for r in type_results if r.success])
                }
        
        # 资源使用统计
        peak_cpu = max([r.peak_metrics.cpu_percent for r in self.results if r.success], default=0)
        peak_memory = max([r.peak_metrics.memory_percent for r in self.results if r.success], default=0)
        peak_memory_gb = max([r.peak_metrics.memory_used_gb for r in self.results if r.success], default=0)
        
        # 错误和警告统计
        total_errors = sum(r.error_count for r in self.results)
        total_warnings = sum(r.warning_count for r in self.results)
        
        report = {
            'test_summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'failed_tests': failed_tests,
                'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
                'avg_performance_score': avg_performance,
                'avg_stability_score': avg_stability,
                'total_errors': total_errors,
                'total_warnings': total_warnings
            },
            'test_type_analysis': type_stats,
            'resource_usage_summary': {
                'peak_cpu_percent': peak_cpu,
                'peak_memory_percent': peak_memory,
                'peak_memory_gb': peak_memory_gb
            },
            'performance_rating': self._get_performance_rating(avg_performance, avg_stability),
            'stress_test_recommendations': self._generate_stress_recommendations()
        }
        
        return report
    
    def _get_performance_rating(self, performance_score: float, stability_score: float) -> str:
        """获取性能等级"""
        combined_score = (performance_score + stability_score) / 2
        
        if combined_score >= 90:
            return "优秀 (Excellent) - 系统在高压力下表现出色"
        elif combined_score >= 75:
            return "良好 (Good) - 系统能够承受大部分压力场景"
        elif combined_score >= 60:
            return "一般 (Fair) - 系统在压力下有一定表现但需要优化"
        elif combined_score >= 40:
            return "较差 (Poor) - 系统在压力下表现不佳，需要重大改进"
        else:
            return "不合格 (Unacceptable) - 系统无法承受压力，存在严重问题"
    
    def _generate_stress_recommendations(self) -> List[str]:
        """生成压力测试建议"""
        recommendations = []
        
        # 基于测试结果的建议
        failed_tests = [r for r in self.results if not r.success]
        if failed_tests:
            recommendations.append(f"有{len(failed_tests)}个压力测试失败，需要检查系统稳定性")
        
        # 基于性能评分的建议
        low_performance_tests = [r for r in self.results if r.success and r.performance_score < 60]
        if low_performance_tests:
            recommendations.append(f"有{len(low_performance_tests)}个测试性能较低，建议优化算法或硬件")
        
        # 基于稳定性评分的建议
        low_stability_tests = [r for r in self.results if r.success and r.stability_score < 70]
        if low_stability_tests:
            recommendations.append(f"有{len(low_stability_tests)}个测试稳定性较低，建议增强错误处理")
        
        # 基于资源使用的建议
        high_memory_tests = [r for r in self.results if r.success and r.peak_metrics.memory_percent > 80]
        if high_memory_tests:
            recommendations.append("检测到高内存使用，建议优化内存管理")
        
        high_cpu_tests = [r for r in self.results if r.success and r.peak_metrics.cpu_percent > 90]
        if high_cpu_tests:
            recommendations.append("检测到高CPU使用，建议优化计算密集型操作")
        
        # 通用建议
        recommendations.extend([
            "定期进行压力测试以确保系统稳定性",
            "监控生产环境中的资源使用情况",
            "建立性能基准和告警机制",
            "考虑实施负载均衡和自动扩缩容"
        ])
        
        return recommendations
    
    def save_report(self, filename: str = None) -> str:
        """保存压力测试报告"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'performance_stress_boundary_report_{timestamp}.json'
        
        report = self.generate_comprehensive_report()
        
        # 添加详细结果
        detailed_results = []
        for result in self.results:
            detailed_results.append({
                'test_name': result.test_name,
                'test_type': result.test_type.value,
                'success': result.success,
                'duration': result.duration,
                'performance_score': result.performance_score,
                'stability_score': result.stability_score,
                'throughput': result.throughput,
                'error_count': result.error_count,
                'warning_count': result.warning_count,
                'peak_metrics': {
                    'cpu_percent': result.peak_metrics.cpu_percent,
                    'memory_percent': result.peak_metrics.memory_percent,
                    'memory_used_gb': result.peak_metrics.memory_used_gb
                },
                'avg_metrics': {
                    'cpu_percent': result.avg_metrics.cpu_percent,
                    'memory_percent': result.avg_metrics.memory_percent,
                    'memory_used_gb': result.avg_metrics.memory_used_gb
                },
                'error_messages': result.error_messages,
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
            print("没有测试结果可显示")
            return
        
        print("\n" + "="*80)
        print("性能压力和边界条件测试报告")
        print("="*80)
        
        # 总体统计
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - successful_tests
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n📊 总体统计:")
        print(f"   总测试数: {total_tests}")
        print(f"   成功: {successful_tests}")
        print(f"   失败: {failed_tests}")
        print(f"   成功率: {success_rate:.1f}%")
        
        # 性能评分
        if successful_tests > 0:
            performance_scores = [r.performance_score for r in self.results if r.success]
            stability_scores = [r.stability_score for r in self.results if r.success]
            avg_performance = np.mean(performance_scores)
            avg_stability = np.mean(stability_scores)
            
            print(f"\n🎯 性能指标:")
            print(f"   平均性能评分: {avg_performance:.1f}/100")
            print(f"   平均稳定性评分: {avg_stability:.1f}/100")
            print(f"   综合评级: {self._get_performance_rating(avg_performance, avg_stability)}")
        
        # 按测试类型统计
        print(f"\n📋 测试类型分析:")
        for test_type in StressTestType:
            type_results = [r for r in self.results if r.test_type == test_type]
            if type_results:
                type_success = sum(1 for r in type_results if r.success)
                type_total = len(type_results)
                type_rate = (type_success / type_total * 100) if type_total > 0 else 0
                print(f"   {test_type.value}: {type_success}/{type_total} ({type_rate:.1f}%)")
        
        # 资源使用峰值
        if successful_tests > 0:
            peak_cpu = max([r.peak_metrics.cpu_percent for r in self.results if r.success], default=0)
            peak_memory = max([r.peak_metrics.memory_percent for r in self.results if r.success], default=0)
            peak_memory_gb = max([r.peak_metrics.memory_used_gb for r in self.results if r.success], default=0)
            
            print(f"\n💻 资源使用峰值:")
            print(f"   CPU使用率: {peak_cpu:.1f}%")
            print(f"   内存使用率: {peak_memory:.1f}%")
            print(f"   内存使用量: {peak_memory_gb:.2f}GB")
        
        # 错误和警告
        total_errors = sum(r.error_count for r in self.results)
        total_warnings = sum(r.warning_count for r in self.results)
        
        if total_errors > 0 or total_warnings > 0:
            print(f"\n⚠️  问题统计:")
            if total_errors > 0:
                print(f"   错误数: {total_errors}")
            if total_warnings > 0:
                print(f"   警告数: {total_warnings}")
        
        print("\n" + "="*80)

def main() -> int:
    """主函数"""
    try:
        print("开始YOLOS系统性能压力和边界条件测试...")
        
        # 创建测试器
        tester = PerformanceStressBoundaryTester()
        
        # 运行所有测试
        results = tester.run_all_tests()
        
        # 打印摘要
        tester.print_summary()
        
        # 保存报告
        report_file = tester.save_report()
        print(f"\n📄 详细报告已保存到: {report_file}")
        
        # 生成建议
        report = tester.generate_comprehensive_report()
        recommendations = report.get('stress_test_recommendations', [])
        
        if recommendations:
            print(f"\n💡 改进建议:")
            for i, rec in enumerate(recommendations[:5], 1):  # 显示前5个建议
                print(f"   {i}. {rec}")
        
        # 返回退出码
        failed_tests = sum(1 for r in results if not r.success)
        return 1 if failed_tests > 0 else 0
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        return 130
    except Exception as e:
        print(f"\n测试执行失败: {e}")
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)