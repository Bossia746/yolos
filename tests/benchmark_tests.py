#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能基准测试套件

提供全面的性能基准测试，包括：
- 模型推理性能基准
- 内存使用基准
- 并发处理基准
- 系统资源使用基准
- 端到端性能基准
"""

import time
import threading
import multiprocessing
import psutil
import gc
import statistics
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
import json
from datetime import datetime

from .base_test import BaseTest
from .test_config import YOLOSTestConfig
from .mock_data import MockDataGenerator


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    test_name: str
    metric_name: str
    value: float
    unit: str
    timestamp: str
    system_info: Dict[str, Any]
    baseline_value: Optional[float] = None
    performance_ratio: Optional[float] = None
    
    def __post_init__(self):
        if self.baseline_value and self.baseline_value > 0:
            self.performance_ratio = self.value / self.baseline_value


@dataclass
class SystemMetrics:
    """系统指标"""
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_io_read: float
    disk_io_write: float
    network_sent: float
    network_recv: float
    gpu_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None


class BenchmarkTestFramework(BaseTest):
    """基准测试框架"""
    
    def __init__(self):
        super().__init__()
        self.test_config = YOLOSTestConfig()
        self.mock_data = MockDataGenerator()
        self.results: List[BenchmarkResult] = []
        self.baselines = self._load_baselines()
        
    def _load_baselines(self) -> Dict[str, float]:
        """加载基准值"""
        baseline_file = Path(__file__).parent / "baselines.json"
        if baseline_file.exists():
            try:
                with open(baseline_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            'inference_fps': 30.0,
            'memory_usage_mb': 512.0,
            'startup_time_ms': 2000.0,
            'detection_accuracy': 0.85,
            'cpu_usage_percent': 50.0
        }
    
    def _save_baselines(self):
        """保存基准值"""
        baseline_file = Path(__file__).parent / "baselines.json"
        try:
            with open(baseline_file, 'w', encoding='utf-8') as f:
                json.dump(self.baselines, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save baselines: {e}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            'platform': psutil.WINDOWS if psutil.WINDOWS else 'linux',
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': f"{psutil.version_info.major}.{psutil.version_info.minor}"
        }
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_mb=memory.used / (1024**2),
            disk_io_read=disk_io.read_bytes / (1024**2) if disk_io else 0,
            disk_io_write=disk_io.write_bytes / (1024**2) if disk_io else 0,
            network_sent=network_io.bytes_sent / (1024**2) if network_io else 0,
            network_recv=network_io.bytes_recv / (1024**2) if network_io else 0
        )
    
    def benchmark_inference_performance(self, duration_seconds: float = 30.0) -> BenchmarkResult:
        """基准测试推理性能"""
        self.logger.info(f"开始推理性能基准测试 (持续 {duration_seconds} 秒)")
        
        # 模拟推理函数
        def mock_inference():
            # 模拟推理计算
            data = np.random.rand(640, 480, 3).astype(np.float32)
            result = np.mean(data) * np.std(data)
            time.sleep(0.01)  # 模拟推理时间
            return result
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration_seconds:
            mock_inference()
            frame_count += 1
        
        actual_duration = time.time() - start_time
        fps = frame_count / actual_duration
        
        result = BenchmarkResult(
            test_name="inference_performance",
            metric_name="fps",
            value=fps,
            unit="frames/second",
            timestamp=datetime.now().isoformat(),
            system_info=self._get_system_info(),
            baseline_value=self.baselines.get('inference_fps')
        )
        
        self.results.append(result)
        self.logger.info(f"推理性能: {fps:.2f} FPS")
        return result
    
    def benchmark_memory_usage(self, duration_seconds: float = 60.0) -> BenchmarkResult:
        """基准测试内存使用"""
        self.logger.info(f"开始内存使用基准测试 (持续 {duration_seconds} 秒)")
        
        memory_samples = []
        start_time = time.time()
        
        # 模拟内存密集型操作
        data_arrays = []
        
        def memory_intensive_task():
            # 创建和释放大量数据
            for _ in range(10):
                arr = np.random.rand(1000, 1000).astype(np.float32)
                data_arrays.append(arr)
                if len(data_arrays) > 50:
                    data_arrays.pop(0)
                time.sleep(0.1)
        
        # 在后台运行内存密集型任务
        thread = threading.Thread(target=memory_intensive_task)
        thread.start()
        
        # 监控内存使用
        while time.time() - start_time < duration_seconds:
            memory_info = psutil.virtual_memory()
            memory_samples.append(memory_info.used / (1024**2))  # MB
            time.sleep(1.0)
        
        thread.join()
        
        # 计算内存统计
        avg_memory = statistics.mean(memory_samples)
        max_memory = max(memory_samples)
        
        result = BenchmarkResult(
            test_name="memory_usage",
            metric_name="average_memory_mb",
            value=avg_memory,
            unit="MB",
            timestamp=datetime.now().isoformat(),
            system_info=self._get_system_info(),
            baseline_value=self.baselines.get('memory_usage_mb')
        )
        
        self.results.append(result)
        self.logger.info(f"平均内存使用: {avg_memory:.2f} MB, 峰值: {max_memory:.2f} MB")
        return result
    
    def benchmark_concurrent_processing(self, num_threads: int = 4, 
                                      duration_seconds: float = 30.0) -> BenchmarkResult:
        """基准测试并发处理性能"""
        self.logger.info(f"开始并发处理基准测试 ({num_threads} 线程, 持续 {duration_seconds} 秒)")
        
        def worker_task(worker_id: int) -> int:
            """工作线程任务"""
            task_count = 0
            start_time = time.time()
            
            while time.time() - start_time < duration_seconds:
                # 模拟处理任务
                data = np.random.rand(100, 100)
                result = np.sum(data * data)
                task_count += 1
                time.sleep(0.001)  # 模拟处理时间
            
            return task_count
        
        # 使用线程池执行并发任务
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_task, i) for i in range(num_threads)]
            task_counts = [future.result() for future in futures]
        
        total_tasks = sum(task_counts)
        throughput = total_tasks / duration_seconds
        
        result = BenchmarkResult(
            test_name="concurrent_processing",
            metric_name="throughput",
            value=throughput,
            unit="tasks/second",
            timestamp=datetime.now().isoformat(),
            system_info=self._get_system_info()
        )
        
        self.results.append(result)
        self.logger.info(f"并发处理吞吐量: {throughput:.2f} tasks/second")
        return result
    
    def benchmark_startup_time(self, iterations: int = 5) -> BenchmarkResult:
        """基准测试启动时间"""
        self.logger.info(f"开始启动时间基准测试 ({iterations} 次迭代)")
        
        startup_times = []
        
        for i in range(iterations):
            start_time = time.time()
            
            # 模拟应用启动过程
            self._simulate_startup()
            
            startup_time = (time.time() - start_time) * 1000  # 转换为毫秒
            startup_times.append(startup_time)
            
            self.logger.debug(f"第 {i+1} 次启动时间: {startup_time:.2f} ms")
        
        avg_startup_time = statistics.mean(startup_times)
        
        result = BenchmarkResult(
            test_name="startup_time",
            metric_name="average_startup_ms",
            value=avg_startup_time,
            unit="milliseconds",
            timestamp=datetime.now().isoformat(),
            system_info=self._get_system_info(),
            baseline_value=self.baselines.get('startup_time_ms')
        )
        
        self.results.append(result)
        self.logger.info(f"平均启动时间: {avg_startup_time:.2f} ms")
        return result
    
    def _simulate_startup(self):
        """模拟应用启动过程"""
        # 模拟配置加载
        time.sleep(0.1)
        
        # 模拟模型加载
        time.sleep(0.5)
        
        # 模拟初始化
        time.sleep(0.2)
        
        # 模拟资源分配
        temp_data = [np.random.rand(100, 100) for _ in range(10)]
        del temp_data
        gc.collect()
    
    def benchmark_end_to_end_latency(self, num_samples: int = 100) -> BenchmarkResult:
        """基准测试端到端延迟"""
        self.logger.info(f"开始端到端延迟基准测试 ({num_samples} 个样本)")
        
        latencies = []
        
        for i in range(num_samples):
            start_time = time.perf_counter()
            
            # 模拟端到端处理流程
            # 1. 数据预处理
            input_data = np.random.rand(640, 480, 3).astype(np.float32)
            processed_data = input_data / 255.0
            
            # 2. 推理
            time.sleep(0.01)  # 模拟推理时间
            
            # 3. 后处理
            result = np.mean(processed_data)
            
            end_time = time.perf_counter()
            latency = (end_time - start_time) * 1000  # 转换为毫秒
            latencies.append(latency)
        
        avg_latency = statistics.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        result = BenchmarkResult(
            test_name="end_to_end_latency",
            metric_name="average_latency_ms",
            value=avg_latency,
            unit="milliseconds",
            timestamp=datetime.now().isoformat(),
            system_info=self._get_system_info()
        )
        
        self.results.append(result)
        self.logger.info(f"端到端延迟 - 平均: {avg_latency:.2f} ms, P95: {p95_latency:.2f} ms, P99: {p99_latency:.2f} ms")
        return result
    
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """运行所有基准测试"""
        self.logger.info("开始运行所有基准测试")
        
        benchmarks = [
            lambda: self.benchmark_inference_performance(30.0),
            lambda: self.benchmark_memory_usage(60.0),
            lambda: self.benchmark_concurrent_processing(4, 30.0),
            lambda: self.benchmark_startup_time(5),
            lambda: self.benchmark_end_to_end_latency(100)
        ]
        
        for benchmark in benchmarks:
            try:
                benchmark()
            except Exception as e:
                self.logger.error(f"基准测试失败: {e}")
        
        return self.results
    
    def generate_benchmark_report(self) -> Dict[str, Any]:
        """生成基准测试报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'results': [],
            'summary': {
                'total_tests': len(self.results),
                'performance_improvements': 0,
                'performance_regressions': 0
            }
        }
        
        for result in self.results:
            result_dict = {
                'test_name': result.test_name,
                'metric_name': result.metric_name,
                'value': result.value,
                'unit': result.unit,
                'baseline_value': result.baseline_value,
                'performance_ratio': result.performance_ratio
            }
            
            if result.performance_ratio:
                if result.performance_ratio > 1.05:  # 5% 改进
                    report['summary']['performance_improvements'] += 1
                    result_dict['status'] = 'improved'
                elif result.performance_ratio < 0.95:  # 5% 退化
                    report['summary']['performance_regressions'] += 1
                    result_dict['status'] = 'regressed'
                else:
                    result_dict['status'] = 'stable'
            else:
                result_dict['status'] = 'baseline'
            
            report['results'].append(result_dict)
        
        return report
    
    def save_benchmark_report(self, filename: Optional[str] = None):
        """保存基准测试报告"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"benchmark_report_{timestamp}.json"
        
        report = self.generate_benchmark_report()
        
        try:
            report_path = Path(__file__).parent / "reports" / filename
            report_path.parent.mkdir(exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"基准测试报告已保存: {report_path}")
        except Exception as e:
            self.logger.error(f"保存基准测试报告失败: {e}")


class BenchmarkTestSuite:
    """基准测试套件"""
    
    def __init__(self):
        self.framework = BenchmarkTestFramework()
    
    def run_quick_benchmarks(self) -> List[BenchmarkResult]:
        """运行快速基准测试"""
        results = []
        
        # 快速推理测试
        results.append(self.framework.benchmark_inference_performance(10.0))
        
        # 快速内存测试
        results.append(self.framework.benchmark_memory_usage(20.0))
        
        # 启动时间测试
        results.append(self.framework.benchmark_startup_time(3))
        
        return results
    
    def run_comprehensive_benchmarks(self) -> List[BenchmarkResult]:
        """运行全面基准测试"""
        return self.framework.run_all_benchmarks()


if __name__ == "__main__":
    # 运行基准测试示例
    suite = BenchmarkTestSuite()
    
    print("运行快速基准测试...")
    results = suite.run_quick_benchmarks()
    
    print("\n基准测试结果:")
    for result in results:
        print(f"{result.test_name}: {result.value:.2f} {result.unit}")
        if result.performance_ratio:
            print(f"  相对基准: {result.performance_ratio:.2f}x")
    
    # 保存报告
    suite.framework.save_benchmark_report()