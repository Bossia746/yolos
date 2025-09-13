#!/usr/bin/env python3
"""
嵌入式设备测试框架
针对资源受限环境的专用测试系统
"""

import os
import sys
import time
import json
import psutil
import threading
import traceback
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
from contextlib import contextmanager

class TestType(Enum):
    """测试类型"""
    UNIT = "unit"                    # 单元测试
    INTEGRATION = "integration"      # 集成测试
    PERFORMANCE = "performance"      # 性能测试
    MEMORY = "memory"                # 内存测试
    STRESS = "stress"                # 压力测试
    COMPATIBILITY = "compatibility"  # 兼容性测试
    RESOURCE = "resource"            # 资源测试

class TestStatus(Enum):
    """测试状态"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

class TestPriority(Enum):
    """测试优先级"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ResourceConstraints:
    """资源约束"""
    max_memory_mb: int = 100
    max_cpu_percent: float = 80.0
    max_execution_time_s: float = 30.0
    max_storage_mb: int = 50
    max_temperature_c: float = 80.0

@dataclass
class TestMetrics:
    """测试指标"""
    execution_time_s: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    storage_usage_mb: float = 0.0
    temperature_c: float = 0.0
    
    # 性能指标
    inference_time_ms: float = 0.0
    fps: float = 0.0
    accuracy: float = 0.0
    
    # 资源效率
    memory_efficiency: float = 0.0
    cpu_efficiency: float = 0.0
    power_efficiency: float = 0.0

@dataclass
class TestResult:
    """测试结果"""
    test_name: str
    test_type: TestType
    status: TestStatus
    start_time: float
    end_time: float
    metrics: TestMetrics
    constraints: ResourceConstraints
    
    # 结果详情
    message: str = ""
    error_details: str = ""
    logs: List[str] = field(default_factory=list)
    
    # 约束检查
    constraint_violations: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
        
    @property
    def passed(self) -> bool:
        return self.status == TestStatus.PASSED
        
    @property
    def failed(self) -> bool:
        return self.status in [TestStatus.FAILED, TestStatus.ERROR]

class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.monitoring = False
        self.metrics_history: List[Dict[str, float]] = []
        self._monitor_thread: Optional[threading.Thread] = None
        
    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.metrics_history.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        
    def stop_monitoring(self) -> TestMetrics:
        """停止监控并返回指标"""
        self.monitoring = False
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
            
        return self._calculate_metrics()
        
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                metrics = self._collect_current_metrics()
                self.metrics_history.append(metrics)
                time.sleep(self.interval)
            except Exception as e:
                logging.warning(f"资源监控错误: {e}")
                
    def _collect_current_metrics(self) -> Dict[str, float]:
        """收集当前指标"""
        metrics = {
            'timestamp': time.time(),
            'memory_usage_mb': 0.0,
            'cpu_usage_percent': 0.0,
            'temperature_c': 0.0
        }
        
        try:
            # 内存使用
            process = psutil.Process()
            memory_info = process.memory_info()
            metrics['memory_usage_mb'] = memory_info.rss / 1024 / 1024
            
            # CPU使用
            metrics['cpu_usage_percent'] = process.cpu_percent()
            
            # 系统温度（如果可用）
            try:
                if hasattr(psutil, 'sensors_temperatures'):
                    temps = psutil.sensors_temperatures()
                    if temps:
                        for name, entries in temps.items():
                            if entries:
                                metrics['temperature_c'] = entries[0].current
                                break
            except:
                pass
                
        except Exception as e:
            logging.warning(f"指标收集错误: {e}")
            
        return metrics
        
    def _calculate_metrics(self) -> TestMetrics:
        """计算测试指标"""
        if not self.metrics_history:
            return TestMetrics()
            
        # 计算平均值和峰值
        memory_values = [m['memory_usage_mb'] for m in self.metrics_history]
        cpu_values = [m['cpu_usage_percent'] for m in self.metrics_history]
        temp_values = [m['temperature_c'] for m in self.metrics_history if m['temperature_c'] > 0]
        
        metrics = TestMetrics(
            memory_usage_mb=max(memory_values) if memory_values else 0.0,
            cpu_usage_percent=sum(cpu_values) / len(cpu_values) if cpu_values else 0.0,
            temperature_c=max(temp_values) if temp_values else 0.0
        )
        
        return metrics

class TestCase:
    """测试用例基类"""
    
    def __init__(self, name: str, test_type: TestType, priority: TestPriority = TestPriority.MEDIUM):
        self.name = name
        self.test_type = test_type
        self.priority = priority
        self.constraints = ResourceConstraints()
        self.setup_done = False
        
    def setup(self) -> bool:
        """测试设置"""
        self.setup_done = True
        return True
        
    def teardown(self):
        """测试清理"""
        pass
        
    def run_test(self) -> Tuple[bool, str]:
        """运行测试，返回(成功, 消息)"""
        raise NotImplementedError("子类必须实现run_test方法")
        
    def validate_constraints(self, metrics: TestMetrics) -> List[str]:
        """验证资源约束"""
        violations = []
        
        if metrics.memory_usage_mb > self.constraints.max_memory_mb:
            violations.append(f"内存使用超限: {metrics.memory_usage_mb:.1f}MB > {self.constraints.max_memory_mb}MB")
            
        if metrics.cpu_usage_percent > self.constraints.max_cpu_percent:
            violations.append(f"CPU使用超限: {metrics.cpu_usage_percent:.1f}% > {self.constraints.max_cpu_percent}%")
            
        if metrics.temperature_c > self.constraints.max_temperature_c:
            violations.append(f"温度超限: {metrics.temperature_c:.1f}°C > {self.constraints.max_temperature_c}°C")
            
        return violations

class ModelLoadTest(TestCase):
    """模型加载测试"""
    
    def __init__(self, model_path: str, max_load_time_s: float = 10.0):
        super().__init__("模型加载测试", TestType.UNIT, TestPriority.HIGH)
        self.model_path = model_path
        self.max_load_time_s = max_load_time_s
        self.constraints.max_memory_mb = 200
        self.constraints.max_execution_time_s = max_load_time_s
        
    def run_test(self) -> Tuple[bool, str]:
        """运行模型加载测试"""
        try:
            start_time = time.time()
            
            # 模拟模型加载
            if not os.path.exists(self.model_path):
                return False, f"模型文件不存在: {self.model_path}"
                
            # 检查文件大小
            file_size_mb = os.path.getsize(self.model_path) / 1024 / 1024
            if file_size_mb > self.constraints.max_memory_mb:
                return False, f"模型文件过大: {file_size_mb:.1f}MB"
                
            load_time = time.time() - start_time
            
            if load_time > self.max_load_time_s:
                return False, f"加载时间过长: {load_time:.2f}s > {self.max_load_time_s}s"
                
            return True, f"模型加载成功，耗时: {load_time:.2f}s，大小: {file_size_mb:.1f}MB"
            
        except Exception as e:
            return False, f"模型加载异常: {str(e)}"

class InferencePerformanceTest(TestCase):
    """推理性能测试"""
    
    def __init__(self, target_fps: float = 1.0, test_iterations: int = 10):
        super().__init__("推理性能测试", TestType.PERFORMANCE, TestPriority.HIGH)
        self.target_fps = target_fps
        self.test_iterations = test_iterations
        self.constraints.max_memory_mb = 100
        self.constraints.max_execution_time_s = 60.0
        
    def run_test(self) -> Tuple[bool, str]:
        """运行推理性能测试"""
        try:
            inference_times = []
            
            for i in range(self.test_iterations):
                start_time = time.time()
                
                # 模拟推理过程
                time.sleep(0.1)  # 模拟推理延迟
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
            # 计算性能指标
            avg_inference_time = sum(inference_times) / len(inference_times)
            actual_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
            
            success = actual_fps >= self.target_fps
            message = f"平均推理时间: {avg_inference_time*1000:.1f}ms, FPS: {actual_fps:.2f}"
            
            if not success:
                message += f" (目标FPS: {self.target_fps:.2f})"
                
            return success, message
            
        except Exception as e:
            return False, f"性能测试异常: {str(e)}"

class MemoryStressTest(TestCase):
    """内存压力测试"""
    
    def __init__(self, max_memory_mb: int = 50, duration_s: float = 30.0):
        super().__init__("内存压力测试", TestType.STRESS, TestPriority.MEDIUM)
        self.max_memory_mb = max_memory_mb
        self.duration_s = duration_s
        self.constraints.max_memory_mb = max_memory_mb
        self.constraints.max_execution_time_s = duration_s + 10.0
        
    def run_test(self) -> Tuple[bool, str]:
        """运行内存压力测试"""
        try:
            start_time = time.time()
            allocated_blocks = []
            
            # 逐步分配内存
            while time.time() - start_time < self.duration_s:
                try:
                    # 分配1MB内存块
                    block = bytearray(1024 * 1024)
                    allocated_blocks.append(block)
                    
                    # 检查内存使用
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    
                    if memory_mb > self.max_memory_mb:
                        return False, f"内存使用超限: {memory_mb:.1f}MB > {self.max_memory_mb}MB"
                        
                    time.sleep(0.1)
                    
                except MemoryError:
                    return False, "内存分配失败"
                    
            # 清理内存
            allocated_blocks.clear()
            
            return True, f"内存压力测试通过，持续时间: {self.duration_s}s"
            
        except Exception as e:
            return False, f"内存压力测试异常: {str(e)}"

class CompatibilityTest(TestCase):
    """兼容性测试"""
    
    def __init__(self, platform_requirements: Dict[str, Any]):
        super().__init__("平台兼容性测试", TestType.COMPATIBILITY, TestPriority.HIGH)
        self.platform_requirements = platform_requirements
        
    def run_test(self) -> Tuple[bool, str]:
        """运行兼容性测试"""
        try:
            issues = []
            
            # 检查Python版本
            if 'python_version' in self.platform_requirements:
                required_version = self.platform_requirements['python_version']
                current_version = sys.version_info[:2]
                if current_version < tuple(map(int, required_version.split('.'))):
                    issues.append(f"Python版本不兼容: {current_version} < {required_version}")
                    
            # 检查内存
            if 'min_memory_mb' in self.platform_requirements:
                required_memory = self.platform_requirements['min_memory_mb']
                try:
                    available_memory = psutil.virtual_memory().available / 1024 / 1024
                    if available_memory < required_memory:
                        issues.append(f"可用内存不足: {available_memory:.0f}MB < {required_memory}MB")
                except:
                    issues.append("无法检测内存信息")
                    
            # 检查存储空间
            if 'min_storage_mb' in self.platform_requirements:
                required_storage = self.platform_requirements['min_storage_mb']
                try:
                    disk_usage = psutil.disk_usage('.')
                    available_storage = disk_usage.free / 1024 / 1024
                    if available_storage < required_storage:
                        issues.append(f"可用存储不足: {available_storage:.0f}MB < {required_storage}MB")
                except:
                    issues.append("无法检测存储信息")
                    
            if issues:
                return False, "; ".join(issues)
            else:
                return True, "平台兼容性检查通过"
                
        except Exception as e:
            return False, f"兼容性测试异常: {str(e)}"

class EmbeddedTestFramework:
    """嵌入式测试框架"""
    
    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.test_cases: List[TestCase] = []
        self.test_results: List[TestResult] = []
        self.logger = logging.getLogger(__name__)
        
        # 配置日志
        self._setup_logging()
        
    def _setup_logging(self):
        """设置日志"""
        log_file = self.output_dir / "test.log"
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 文件处理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.DEBUG)
        
    def add_test_case(self, test_case: TestCase):
        """添加测试用例"""
        self.test_cases.append(test_case)
        self.logger.info(f"添加测试用例: {test_case.name}")
        
    def run_all_tests(self, stop_on_failure: bool = False) -> Dict[str, Any]:
        """运行所有测试"""
        self.logger.info(f"开始运行 {len(self.test_cases)} 个测试用例")
        
        start_time = time.time()
        passed_count = 0
        failed_count = 0
        skipped_count = 0
        
        # 按优先级排序
        sorted_tests = sorted(self.test_cases, key=lambda t: t.priority.value, reverse=True)
        
        for test_case in sorted_tests:
            try:
                result = self._run_single_test(test_case)
                self.test_results.append(result)
                
                if result.passed:
                    passed_count += 1
                    self.logger.info(f"✓ {test_case.name}: PASSED")
                elif result.status == TestStatus.SKIPPED:
                    skipped_count += 1
                    self.logger.info(f"- {test_case.name}: SKIPPED")
                else:
                    failed_count += 1
                    self.logger.error(f"✗ {test_case.name}: FAILED - {result.message}")
                    
                    if stop_on_failure:
                        self.logger.info("遇到失败，停止测试")
                        break
                        
            except Exception as e:
                failed_count += 1
                self.logger.error(f"✗ {test_case.name}: ERROR - {str(e)}")
                
                if stop_on_failure:
                    break
                    
        end_time = time.time()
        
        # 生成测试报告
        summary = {
            'total_tests': len(self.test_cases),
            'passed': passed_count,
            'failed': failed_count,
            'skipped': skipped_count,
            'success_rate': passed_count / len(self.test_cases) * 100 if self.test_cases else 0,
            'total_time': end_time - start_time,
            'timestamp': time.time()
        }
        
        self.logger.info(f"测试完成: {passed_count}通过, {failed_count}失败, {skipped_count}跳过")
        self.logger.info(f"成功率: {summary['success_rate']:.1f}%, 总耗时: {summary['total_time']:.2f}s")
        
        # 保存测试报告
        self._save_test_report(summary)
        
        return summary
        
    def _run_single_test(self, test_case: TestCase) -> TestResult:
        """运行单个测试"""
        start_time = time.time()
        
        # 创建资源监控器
        monitor = ResourceMonitor()
        
        # 初始化结果
        result = TestResult(
            test_name=test_case.name,
            test_type=test_case.test_type,
            status=TestStatus.PENDING,
            start_time=start_time,
            end_time=start_time,
            metrics=TestMetrics(),
            constraints=test_case.constraints
        )
        
        try:
            # 设置测试
            if not test_case.setup():
                result.status = TestStatus.SKIPPED
                result.message = "测试设置失败"
                return result
                
            result.status = TestStatus.RUNNING
            
            # 开始监控
            monitor.start_monitoring()
            
            # 运行测试
            success, message = test_case.run_test()
            
            # 停止监控
            metrics = monitor.stop_monitoring()
            result.metrics = metrics
            
            # 检查执行时间约束
            execution_time = time.time() - start_time
            result.metrics.execution_time_s = execution_time
            
            if execution_time > test_case.constraints.max_execution_time_s:
                success = False
                message += f" (执行超时: {execution_time:.2f}s > {test_case.constraints.max_execution_time_s}s)"
                
            # 验证资源约束
            violations = test_case.validate_constraints(metrics)
            result.constraint_violations = violations
            
            if violations:
                success = False
                message += f" (资源约束违规: {'; '.join(violations)})"
                
            # 设置最终状态
            result.status = TestStatus.PASSED if success else TestStatus.FAILED
            result.message = message
            
        except Exception as e:
            result.status = TestStatus.ERROR
            result.message = f"测试执行异常: {str(e)}"
            result.error_details = traceback.format_exc()
            
        finally:
            result.end_time = time.time()
            
            # 清理测试
            try:
                test_case.teardown()
            except Exception as e:
                self.logger.warning(f"测试清理失败: {e}")
                
        return result
        
    def _save_test_report(self, summary: Dict[str, Any]):
        """保存测试报告"""
        try:
            # 详细报告
            report = {
                'summary': summary,
                'test_results': [
                    {
                        'name': r.test_name,
                        'type': r.test_type.value,
                        'status': r.status.value,
                        'duration': r.duration,
                        'message': r.message,
                        'metrics': {
                            'execution_time_s': r.metrics.execution_time_s,
                            'memory_usage_mb': r.metrics.memory_usage_mb,
                            'cpu_usage_percent': r.metrics.cpu_usage_percent,
                            'temperature_c': r.metrics.temperature_c
                        },
                        'constraint_violations': r.constraint_violations
                    }
                    for r in self.test_results
                ]
            }
            
            # 保存JSON报告
            report_file = self.output_dir / f"test_report_{int(time.time())}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"测试报告已保存: {report_file}")
            
        except Exception as e:
            self.logger.error(f"保存测试报告失败: {e}")
            
    def get_test_summary(self) -> Dict[str, Any]:
        """获取测试摘要"""
        if not self.test_results:
            return {'message': '没有测试结果'}
            
        passed = sum(1 for r in self.test_results if r.passed)
        failed = sum(1 for r in self.test_results if r.failed)
        total = len(self.test_results)
        
        avg_memory = sum(r.metrics.memory_usage_mb for r in self.test_results) / total
        avg_cpu = sum(r.metrics.cpu_usage_percent for r in self.test_results) / total
        total_time = sum(r.duration for r in self.test_results)
        
        return {
            'total_tests': total,
            'passed': passed,
            'failed': failed,
            'success_rate': passed / total * 100,
            'avg_memory_mb': avg_memory,
            'avg_cpu_percent': avg_cpu,
            'total_time_s': total_time,
            'failed_tests': [r.test_name for r in self.test_results if r.failed]
        }

def create_embedded_test_suite() -> EmbeddedTestFramework:
    """创建嵌入式测试套件"""
    framework = EmbeddedTestFramework()
    
    # 添加基础测试用例
    framework.add_test_case(ModelLoadTest("models/yolov11n.pt", max_load_time_s=5.0))
    framework.add_test_case(InferencePerformanceTest(target_fps=2.0, test_iterations=5))
    framework.add_test_case(MemoryStressTest(max_memory_mb=100, duration_s=10.0))
    
    # 兼容性测试
    platform_requirements = {
        'python_version': '3.7',
        'min_memory_mb': 50,
        'min_storage_mb': 100
    }
    framework.add_test_case(CompatibilityTest(platform_requirements))
    
    return framework

if __name__ == "__main__":
    # 测试代码
    print("嵌入式测试框架演示")
    print("=" * 50)
    
    # 创建测试框架
    framework = create_embedded_test_suite()
    
    # 运行测试
    print("\n开始运行测试...")
    summary = framework.run_all_tests()
    
    # 显示结果
    print(f"\n测试摘要:")
    print(f"  总测试数: {summary['total_tests']}")
    print(f"  通过: {summary['passed']}")
    print(f"  失败: {summary['failed']}")
    print(f"  跳过: {summary['skipped']}")
    print(f"  成功率: {summary['success_rate']:.1f}%")
    print(f"  总耗时: {summary['total_time']:.2f}s")
    
    # 详细摘要
    detailed_summary = framework.get_test_summary()
    if 'failed_tests' in detailed_summary and detailed_summary['failed_tests']:
        print(f"\n失败的测试: {', '.join(detailed_summary['failed_tests'])}")
        
    print(f"\n平均资源使用:")
    print(f"  内存: {detailed_summary.get('avg_memory_mb', 0):.1f}MB")
    print(f"  CPU: {detailed_summary.get('avg_cpu_percent', 0):.1f}%")
    
    print("\n测试完成")