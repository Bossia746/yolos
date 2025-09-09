"""性能测试工具

提供全面的性能测试功能，包括：
- 帧率测试
- 内存使用测试
- CPU使用测试
- 延迟测试
- 负载测试
- 基准测试
"""

import time
import threading
import multiprocessing
import psutil
import gc
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from unittest.mock import Mock
import statistics

from .base_test import BaseTest
from .test_config import TestConfig
from .mock_data import MockDataGenerator

@dataclass
class PerformanceMetrics:
    """性能指标"""
    fps: float = 0.0
    avg_processing_time: float = 0.0
    min_processing_time: float = float('inf')
    max_processing_time: float = 0.0
    memory_usage_mb: float = 0.0
    peak_memory_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    peak_cpu_percent: float = 0.0
    latency_ms: float = 0.0
    throughput_ops_per_sec: float = 0.0
    error_rate: float = 0.0
    
@dataclass
class PerformanceTestResult:
    """性能测试结果"""
    test_name: str
    success: bool
    duration: float
    metrics: PerformanceMetrics
    benchmark_comparison: Dict[str, bool] = field(default_factory=dict)
    error_message: Optional[str] = None
    detailed_stats: Dict[str, Any] = field(default_factory=dict)

class FrameRateTest(BaseTest):
    """帧率测试"""
    
    def __init__(self):
        super().__init__()
        self.test_config = TestConfig()
        self.mock_data = MockDataGenerator()
        
    def test_single_thread_fps(self, duration: float = 30.0) -> PerformanceTestResult:
        """测试单线程帧率
        
        Args:
            duration: 测试持续时间（秒）
            
        Returns:
            性能测试结果
        """
        start_time = time.time()
        test_name = "single_thread_fps"
        
        try:
            # system = YOLOSSystem()  # 实际系统类
            system = Mock()  # 临时使用Mock
            system.initialize()
            system.start()
            
            frame_count = 0
            processing_times = []
            error_count = 0
            
            while time.time() - start_time < duration:
                try:
                    # 生成测试帧
                    frame = self.mock_data.generate_image_data()
                    
                    # 处理帧并记录时间
                    frame_start = time.time()
                    result = system.process_frame(frame)
                    frame_end = time.time()
                    
                    processing_times.append(frame_end - frame_start)
                    frame_count += 1
                    
                except Exception as e:
                    error_count += 1
                    
            system.stop()
            system.cleanup()
            
            # 计算性能指标
            total_duration = time.time() - start_time
            fps = frame_count / total_duration
            avg_processing_time = statistics.mean(processing_times) if processing_times else 0
            min_processing_time = min(processing_times) if processing_times else 0
            max_processing_time = max(processing_times) if processing_times else 0
            error_rate = error_count / (frame_count + error_count) if (frame_count + error_count) > 0 else 0
            
            metrics = PerformanceMetrics(
                fps=fps,
                avg_processing_time=avg_processing_time,
                min_processing_time=min_processing_time,
                max_processing_time=max_processing_time,
                error_rate=error_rate
            )
            
            # 基准比较
            benchmarks = self.test_config.benchmarks
            benchmark_comparison = {
                'fps_meets_target': fps >= benchmarks.min_fps,
                'processing_time_acceptable': avg_processing_time <= benchmarks.max_frame_processing_time,
                'error_rate_acceptable': error_rate <= 0.05  # 5%错误率阈值
            }
            
            success = all(benchmark_comparison.values())
            
            return PerformanceTestResult(
                test_name=test_name,
                success=success,
                duration=total_duration,
                metrics=metrics,
                benchmark_comparison=benchmark_comparison,
                detailed_stats={
                    'frames_processed': frame_count,
                    'errors': error_count,
                    'processing_times_std': statistics.stdev(processing_times) if len(processing_times) > 1 else 0,
                    'processing_times_median': statistics.median(processing_times) if processing_times else 0
                }
            )
            
        except Exception as e:
            return PerformanceTestResult(
                test_name=test_name,
                success=False,
                duration=time.time() - start_time,
                metrics=PerformanceMetrics(),
                error_message=str(e)
            )
            
    def test_multi_thread_fps(self, thread_count: int = 4, duration: float = 30.0) -> PerformanceTestResult:
        """测试多线程帧率
        
        Args:
            thread_count: 线程数量
            duration: 测试持续时间（秒）
            
        Returns:
            性能测试结果
        """
        start_time = time.time()
        test_name = f"multi_thread_fps_{thread_count}_threads"
        
        try:
            # 共享统计数据
            frame_counts = [0] * thread_count
            processing_times = [[] for _ in range(thread_count)]
            error_counts = [0] * thread_count
            
            def worker_thread(thread_id: int):
                """工作线程函数"""
                # system = YOLOSSystem()  # 每个线程独立的系统实例
                system = Mock()  # 临时使用Mock
                system.initialize()
                system.start()
                
                thread_start = time.time()
                
                while time.time() - thread_start < duration:
                    try:
                        frame = self.mock_data.generate_image_data()
                        
                        frame_start = time.time()
                        result = system.process_frame(frame)
                        frame_end = time.time()
                        
                        processing_times[thread_id].append(frame_end - frame_start)
                        frame_counts[thread_id] += 1
                        
                    except Exception:
                        error_counts[thread_id] += 1
                        
                system.stop()
                system.cleanup()
                
            # 启动多线程
            threads = []
            for i in range(thread_count):
                thread = threading.Thread(target=worker_thread, args=(i,))
                threads.append(thread)
                thread.start()
                
            # 等待所有线程完成
            for thread in threads:
                thread.join()
                
            # 汇总统计
            total_frames = sum(frame_counts)
            total_errors = sum(error_counts)
            all_processing_times = [t for times in processing_times for t in times]
            
            total_duration = time.time() - start_time
            fps = total_frames / total_duration
            avg_processing_time = statistics.mean(all_processing_times) if all_processing_times else 0
            min_processing_time = min(all_processing_times) if all_processing_times else 0
            max_processing_time = max(all_processing_times) if all_processing_times else 0
            error_rate = total_errors / (total_frames + total_errors) if (total_frames + total_errors) > 0 else 0
            
            metrics = PerformanceMetrics(
                fps=fps,
                avg_processing_time=avg_processing_time,
                min_processing_time=min_processing_time,
                max_processing_time=max_processing_time,
                error_rate=error_rate
            )
            
            # 基准比较
            benchmarks = self.test_config.benchmarks
            expected_fps = benchmarks.min_fps * thread_count * 0.8  # 80%效率预期
            benchmark_comparison = {
                'fps_meets_scaled_target': fps >= expected_fps,
                'processing_time_acceptable': avg_processing_time <= benchmarks.max_frame_processing_time,
                'error_rate_acceptable': error_rate <= 0.05,
                'thread_efficiency': fps / (benchmarks.min_fps * thread_count) if benchmarks.min_fps > 0 else 0
            }
            
            success = benchmark_comparison['fps_meets_scaled_target'] and \
                     benchmark_comparison['processing_time_acceptable'] and \
                     benchmark_comparison['error_rate_acceptable']
            
            return PerformanceTestResult(
                test_name=test_name,
                success=success,
                duration=total_duration,
                metrics=metrics,
                benchmark_comparison=benchmark_comparison,
                detailed_stats={
                    'thread_count': thread_count,
                    'total_frames': total_frames,
                    'total_errors': total_errors,
                    'frames_per_thread': frame_counts,
                    'errors_per_thread': error_counts,
                    'expected_fps': expected_fps
                }
            )
            
        except Exception as e:
            return PerformanceTestResult(
                test_name=test_name,
                success=False,
                duration=time.time() - start_time,
                metrics=PerformanceMetrics(),
                error_message=str(e)
            )

class MemoryUsageTest(BaseTest):
    """内存使用测试"""
    
    def __init__(self):
        super().__init__()
        self.test_config = TestConfig()
        self.mock_data = MockDataGenerator()
        
    def test_memory_usage_pattern(self, duration: float = 120.0) -> PerformanceTestResult:
        """测试内存使用模式
        
        Args:
            duration: 测试持续时间（秒）
            
        Returns:
            性能测试结果
        """
        start_time = time.time()
        test_name = "memory_usage_pattern"
        
        try:
            process = psutil.Process()
            
            # 记录初始内存
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # system = YOLOSSystem()  # 实际系统类
            system = Mock()  # 临时使用Mock
            system.initialize()
            system.start()
            
            memory_samples = []
            frame_count = 0
            peak_memory = initial_memory
            
            while time.time() - start_time < duration:
                # 处理帧
                frame = self.mock_data.generate_image_data()
                system.process_frame(frame)
                frame_count += 1
                
                # 每5帧采样一次内存
                if frame_count % 5 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_samples.append(current_memory)
                    peak_memory = max(peak_memory, current_memory)
                    
                # 每100帧强制垃圾回收
                if frame_count % 100 == 0:
                    gc.collect()
                    
            system.stop()
            system.cleanup()
            
            # 最终内存检查
            gc.collect()
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # 计算内存统计
            avg_memory = statistics.mean(memory_samples) if memory_samples else final_memory
            memory_growth = final_memory - initial_memory
            memory_variance = statistics.variance(memory_samples) if len(memory_samples) > 1 else 0
            
            metrics = PerformanceMetrics(
                memory_usage_mb=avg_memory,
                peak_memory_mb=peak_memory
            )
            
            # 基准比较
            benchmarks = self.test_config.benchmarks
            benchmark_comparison = {
                'peak_memory_acceptable': peak_memory <= benchmarks.max_memory_usage,
                'avg_memory_acceptable': avg_memory <= benchmarks.max_memory_usage * 0.8,
                'memory_growth_acceptable': memory_growth <= 100.0,  # 100MB增长阈值
                'memory_stable': memory_variance <= 1000.0  # 内存方差阈值
            }
            
            success = all(benchmark_comparison.values())
            
            total_duration = time.time() - start_time
            return PerformanceTestResult(
                test_name=test_name,
                success=success,
                duration=total_duration,
                metrics=metrics,
                benchmark_comparison=benchmark_comparison,
                detailed_stats={
                    'frames_processed': frame_count,
                    'initial_memory_mb': initial_memory,
                    'final_memory_mb': final_memory,
                    'memory_growth_mb': memory_growth,
                    'memory_samples_count': len(memory_samples),
                    'memory_variance': memory_variance,
                    'memory_samples': memory_samples[-10:]  # 最后10个样本
                }
            )
            
        except Exception as e:
            return PerformanceTestResult(
                test_name=test_name,
                success=False,
                duration=time.time() - start_time,
                metrics=PerformanceMetrics(),
                error_message=str(e)
            )
            
    def test_memory_leak_detection(self, iterations: int = 100) -> PerformanceTestResult:
        """测试内存泄漏检测
        
        Args:
            iterations: 迭代次数
            
        Returns:
            性能测试结果
        """
        start_time = time.time()
        test_name = "memory_leak_detection"
        
        try:
            process = psutil.Process()
            memory_snapshots = []
            
            for i in range(iterations):
                # 记录迭代开始内存
                iter_start_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                # 创建和销毁系统实例
                # system = YOLOSSystem()  # 实际系统类
                system = Mock()  # 临时使用Mock
                system.initialize()
                system.start()
                
                # 处理一些帧
                for _ in range(10):
                    frame = self.mock_data.generate_image_data()
                    system.process_frame(frame)
                    
                system.stop()
                system.cleanup()
                del system
                
                # 强制垃圾回收
                gc.collect()
                
                # 记录迭代结束内存
                iter_end_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_snapshots.append({
                    'iteration': i,
                    'start_memory': iter_start_memory,
                    'end_memory': iter_end_memory,
                    'growth': iter_end_memory - iter_start_memory
                })
                
            # 分析内存泄漏
            total_growth = memory_snapshots[-1]['end_memory'] - memory_snapshots[0]['start_memory']
            avg_growth_per_iteration = total_growth / iterations
            
            # 计算趋势
            growth_values = [s['growth'] for s in memory_snapshots]
            positive_growth_count = sum(1 for g in growth_values if g > 1.0)  # 1MB阈值
            
            metrics = PerformanceMetrics(
                memory_usage_mb=memory_snapshots[-1]['end_memory'],
                peak_memory_mb=max(s['end_memory'] for s in memory_snapshots)
            )
            
            # 泄漏检测基准
            benchmark_comparison = {
                'total_growth_acceptable': total_growth <= 50.0,  # 50MB总增长阈值
                'avg_growth_acceptable': avg_growth_per_iteration <= 0.5,  # 0.5MB平均增长阈值
                'leak_frequency_acceptable': positive_growth_count / iterations <= 0.3  # 30%泄漏频率阈值
            }
            
            success = all(benchmark_comparison.values())
            
            total_duration = time.time() - start_time
            return PerformanceTestResult(
                test_name=test_name,
                success=success,
                duration=total_duration,
                metrics=metrics,
                benchmark_comparison=benchmark_comparison,
                detailed_stats={
                    'iterations': iterations,
                    'total_growth_mb': total_growth,
                    'avg_growth_per_iteration_mb': avg_growth_per_iteration,
                    'positive_growth_count': positive_growth_count,
                    'memory_snapshots': memory_snapshots[-5:]  # 最后5个快照
                }
            )
            
        except Exception as e:
            return PerformanceTestResult(
                test_name=test_name,
                success=False,
                duration=time.time() - start_time,
                metrics=PerformanceMetrics(),
                error_message=str(e)
            )

class CPUUsageTest(BaseTest):
    """CPU使用测试"""
    
    def __init__(self):
        super().__init__()
        self.test_config = TestConfig()
        self.mock_data = MockDataGenerator()
        
    def test_cpu_usage_pattern(self, duration: float = 60.0) -> PerformanceTestResult:
        """测试CPU使用模式
        
        Args:
            duration: 测试持续时间（秒）
            
        Returns:
            性能测试结果
        """
        start_time = time.time()
        test_name = "cpu_usage_pattern"
        
        try:
            process = psutil.Process()
            cpu_samples = []
            frame_count = 0
            peak_cpu = 0.0
            
            # system = YOLOSSystem()  # 实际系统类
            system = Mock()  # 临时使用Mock
            system.initialize()
            system.start()
            
            # CPU监控线程
            def monitor_cpu():
                while time.time() - start_time < duration:
                    cpu_percent = process.cpu_percent(interval=0.1)
                    cpu_samples.append(cpu_percent)
                    nonlocal peak_cpu
                    peak_cpu = max(peak_cpu, cpu_percent)
                    
            monitor_thread = threading.Thread(target=monitor_cpu)
            monitor_thread.start()
            
            # 主处理循环
            while time.time() - start_time < duration:
                frame = self.mock_data.generate_image_data()
                system.process_frame(frame)
                frame_count += 1
                
            monitor_thread.join()
            system.stop()
            system.cleanup()
            
            # 计算CPU统计
            avg_cpu = statistics.mean(cpu_samples) if cpu_samples else 0
            cpu_variance = statistics.variance(cpu_samples) if len(cpu_samples) > 1 else 0
            
            metrics = PerformanceMetrics(
                cpu_usage_percent=avg_cpu,
                peak_cpu_percent=peak_cpu,
                fps=frame_count / duration
            )
            
            # 基准比较
            benchmarks = self.test_config.benchmarks
            benchmark_comparison = {
                'avg_cpu_acceptable': avg_cpu <= benchmarks.max_cpu_usage,
                'peak_cpu_acceptable': peak_cpu <= benchmarks.max_cpu_usage * 1.2,  # 允许20%峰值超出
                'cpu_efficiency': frame_count / (avg_cpu * duration) if avg_cpu > 0 else 0,
                'cpu_stable': cpu_variance <= 100.0  # CPU方差阈值
            }
            
            success = benchmark_comparison['avg_cpu_acceptable'] and \
                     benchmark_comparison['peak_cpu_acceptable']
            
            total_duration = time.time() - start_time
            return PerformanceTestResult(
                test_name=test_name,
                success=success,
                duration=total_duration,
                metrics=metrics,
                benchmark_comparison=benchmark_comparison,
                detailed_stats={
                    'frames_processed': frame_count,
                    'cpu_samples_count': len(cpu_samples),
                    'cpu_variance': cpu_variance,
                    'cpu_samples': cpu_samples[-10:]  # 最后10个样本
                }
            )
            
        except Exception as e:
            return PerformanceTestResult(
                test_name=test_name,
                success=False,
                duration=time.time() - start_time,
                metrics=PerformanceMetrics(),
                error_message=str(e)
            )

class LatencyTest(BaseTest):
    """延迟测试"""
    
    def __init__(self):
        super().__init__()
        self.test_config = TestConfig()
        self.mock_data = MockDataGenerator()
        
    def test_end_to_end_latency(self, sample_count: int = 100) -> PerformanceTestResult:
        """测试端到端延迟
        
        Args:
            sample_count: 样本数量
            
        Returns:
            性能测试结果
        """
        start_time = time.time()
        test_name = "end_to_end_latency"
        
        try:
            # system = YOLOSSystem()  # 实际系统类
            system = Mock()  # 临时使用Mock
            system.initialize()
            system.start()
            
            latencies = []
            
            for i in range(sample_count):
                # 生成测试数据
                frame = self.mock_data.generate_image_data()
                
                # 测量端到端延迟
                request_start = time.time()
                result = system.process_frame(frame)
                request_end = time.time()
                
                latency_ms = (request_end - request_start) * 1000
                latencies.append(latency_ms)
                
                # 避免过快请求
                time.sleep(0.01)
                
            system.stop()
            system.cleanup()
            
            # 计算延迟统计
            avg_latency = statistics.mean(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
            p99_latency = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
            
            metrics = PerformanceMetrics(
                latency_ms=avg_latency,
                avg_processing_time=avg_latency / 1000  # 转换为秒
            )
            
            # 基准比较
            benchmarks = self.test_config.benchmarks
            max_latency_ms = benchmarks.max_frame_processing_time * 1000
            benchmark_comparison = {
                'avg_latency_acceptable': avg_latency <= max_latency_ms,
                'p95_latency_acceptable': p95_latency <= max_latency_ms * 1.5,
                'p99_latency_acceptable': p99_latency <= max_latency_ms * 2.0,
                'max_latency_acceptable': max_latency <= max_latency_ms * 3.0
            }
            
            success = all(benchmark_comparison.values())
            
            total_duration = time.time() - start_time
            return PerformanceTestResult(
                test_name=test_name,
                success=success,
                duration=total_duration,
                metrics=metrics,
                benchmark_comparison=benchmark_comparison,
                detailed_stats={
                    'sample_count': sample_count,
                    'avg_latency_ms': avg_latency,
                    'min_latency_ms': min_latency,
                    'max_latency_ms': max_latency,
                    'p95_latency_ms': p95_latency,
                    'p99_latency_ms': p99_latency,
                    'latency_std_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0
                }
            )
            
        except Exception as e:
            return PerformanceTestResult(
                test_name=test_name,
                success=False,
                duration=time.time() - start_time,
                metrics=PerformanceMetrics(),
                error_message=str(e)
            )

class LoadTest(BaseTest):
    """负载测试"""
    
    def __init__(self):
        super().__init__()
        self.test_config = TestConfig()
        self.mock_data = MockDataGenerator()
        
    def test_concurrent_load(self, concurrent_users: int = 10, duration: float = 60.0) -> PerformanceTestResult:
        """测试并发负载
        
        Args:
            concurrent_users: 并发用户数
            duration: 测试持续时间（秒）
            
        Returns:
            性能测试结果
        """
        start_time = time.time()
        test_name = f"concurrent_load_{concurrent_users}_users"
        
        try:
            # 共享统计数据
            total_requests = multiprocessing.Value('i', 0)
            total_errors = multiprocessing.Value('i', 0)
            response_times = multiprocessing.Manager().list()
            
            def user_simulation(user_id: int):
                """模拟用户行为"""
                # system = YOLOSSystem()  # 每个用户独立的系统实例
                system = Mock()  # 临时使用Mock
                system.initialize()
                system.start()
                
                user_start = time.time()
                
                while time.time() - user_start < duration:
                    try:
                        frame = self.mock_data.generate_image_data()
                        
                        request_start = time.time()
                        result = system.process_frame(frame)
                        request_end = time.time()
                        
                        with total_requests.get_lock():
                            total_requests.value += 1
                            
                        response_times.append(request_end - request_start)
                        
                        # 模拟用户思考时间
                        time.sleep(0.1)
                        
                    except Exception:
                        with total_errors.get_lock():
                            total_errors.value += 1
                            
                system.stop()
                system.cleanup()
                
            # 启动并发进程
            processes = []
            for i in range(concurrent_users):
                process = multiprocessing.Process(target=user_simulation, args=(i,))
                processes.append(process)
                process.start()
                
            # 等待所有进程完成
            for process in processes:
                process.join()
                
            # 计算负载统计
            total_duration = time.time() - start_time
            requests_count = total_requests.value
            errors_count = total_errors.value
            
            if requests_count > 0:
                throughput = requests_count / total_duration
                error_rate = errors_count / (requests_count + errors_count)
                avg_response_time = statistics.mean(response_times) if response_times else 0
            else:
                throughput = 0
                error_rate = 1.0
                avg_response_time = 0
                
            metrics = PerformanceMetrics(
                throughput_ops_per_sec=throughput,
                error_rate=error_rate,
                avg_processing_time=avg_response_time
            )
            
            # 基准比较
            expected_throughput = concurrent_users * 5  # 每用户每秒5个请求的预期
            benchmark_comparison = {
                'throughput_acceptable': throughput >= expected_throughput * 0.7,  # 70%效率
                'error_rate_acceptable': error_rate <= 0.05,  # 5%错误率
                'response_time_acceptable': avg_response_time <= 0.5,  # 500ms响应时间
                'system_stability': errors_count < requests_count * 0.1  # 10%错误阈值
            }
            
            success = all(benchmark_comparison.values())
            
            return PerformanceTestResult(
                test_name=test_name,
                success=success,
                duration=total_duration,
                metrics=metrics,
                benchmark_comparison=benchmark_comparison,
                detailed_stats={
                    'concurrent_users': concurrent_users,
                    'total_requests': requests_count,
                    'total_errors': errors_count,
                    'expected_throughput': expected_throughput,
                    'response_times_count': len(response_times)
                }
            )
            
        except Exception as e:
            return PerformanceTestResult(
                test_name=test_name,
                success=False,
                duration=time.time() - start_time,
                metrics=PerformanceMetrics(),
                error_message=str(e)
            )

class PerformanceTestSuite:
    """性能测试套件"""
    
    def __init__(self):
        self.frame_rate_test = FrameRateTest()
        self.memory_test = MemoryUsageTest()
        self.cpu_test = CPUUsageTest()
        self.latency_test = LatencyTest()
        self.load_test = LoadTest()
        
    def run_basic_performance_tests(self) -> List[PerformanceTestResult]:
        """运行基础性能测试
        
        Returns:
            性能测试结果列表
        """
        results = []
        
        # 帧率测试
        results.append(self.frame_rate_test.test_single_thread_fps(30.0))
        
        # 内存测试
        results.append(self.memory_test.test_memory_usage_pattern(60.0))
        
        # CPU测试
        results.append(self.cpu_test.test_cpu_usage_pattern(30.0))
        
        # 延迟测试
        results.append(self.latency_test.test_end_to_end_latency(50))
        
        return results
        
    def run_comprehensive_performance_tests(self) -> List[PerformanceTestResult]:
        """运行全面性能测试
        
        Returns:
            性能测试结果列表
        """
        results = []
        
        # 帧率测试
        results.append(self.frame_rate_test.test_single_thread_fps(60.0))
        results.append(self.frame_rate_test.test_multi_thread_fps(4, 60.0))
        
        # 内存测试
        results.append(self.memory_test.test_memory_usage_pattern(120.0))
        results.append(self.memory_test.test_memory_leak_detection(50))
        
        # CPU测试
        results.append(self.cpu_test.test_cpu_usage_pattern(60.0))
        
        # 延迟测试
        results.append(self.latency_test.test_end_to_end_latency(100))
        
        # 负载测试
        results.append(self.load_test.test_concurrent_load(5, 60.0))
        
        return results
        
    def generate_performance_report(self, results: List[PerformanceTestResult]) -> Dict[str, Any]:
        """生成性能测试报告
        
        Args:
            results: 性能测试结果列表
            
        Returns:
            性能测试报告
        """
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.success)
        failed_tests = total_tests - passed_tests
        
        # 汇总性能指标
        all_fps = [r.metrics.fps for r in results if r.metrics.fps > 0]
        all_memory = [r.metrics.memory_usage_mb for r in results if r.metrics.memory_usage_mb > 0]
        all_cpu = [r.metrics.cpu_usage_percent for r in results if r.metrics.cpu_usage_percent > 0]
        all_latency = [r.metrics.latency_ms for r in results if r.metrics.latency_ms > 0]
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'total_duration': sum(r.duration for r in results)
            },
            'performance_summary': {
                'avg_fps': statistics.mean(all_fps) if all_fps else 0,
                'max_fps': max(all_fps) if all_fps else 0,
                'avg_memory_mb': statistics.mean(all_memory) if all_memory else 0,
                'peak_memory_mb': max(all_memory) if all_memory else 0,
                'avg_cpu_percent': statistics.mean(all_cpu) if all_cpu else 0,
                'peak_cpu_percent': max(all_cpu) if all_cpu else 0,
                'avg_latency_ms': statistics.mean(all_latency) if all_latency else 0,
                'max_latency_ms': max(all_latency) if all_latency else 0
            },
            'test_results': [
                {
                    'name': r.test_name,
                    'success': r.success,
                    'duration': r.duration,
                    'metrics': {
                        'fps': r.metrics.fps,
                        'memory_mb': r.metrics.memory_usage_mb,
                        'cpu_percent': r.metrics.cpu_usage_percent,
                        'latency_ms': r.metrics.latency_ms,
                        'error_rate': r.metrics.error_rate
                    },
                    'benchmarks_met': r.benchmark_comparison,
                    'error': r.error_message
                }
                for r in results
            ],
            'failed_tests': [
                {
                    'name': r.test_name,
                    'error': r.error_message,
                    'duration': r.duration,
                    'benchmarks_failed': [k for k, v in r.benchmark_comparison.items() if not v]
                }
                for r in results if not r.success
            ]
        }
        
        return report