#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试框架模块

提供统一的测试基础设施，支持单元测试、集成测试、性能测试等。
包含测试运行器、测试报告、模拟对象、测试数据管理等功能。

作者: YOLOS团队
日期: 2024
"""

import asyncio
import inspect
import json
import time
import traceback
import unittest
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union
import logging
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# 导入核心模块
from .application import Application, ApplicationConfig
from .event_system import Event, EventType, publish_event
from .performance_monitor import PerformanceMetrics


class TestType(Enum):
    """测试类型"""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SYSTEM = "system"
    ACCEPTANCE = "acceptance"


class TestStatus(Enum):
    """测试状态"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestResult:
    """测试结果"""
    name: str
    status: TestStatus
    duration: float = 0.0
    message: Optional[str] = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TestSuite:
    """测试套件"""
    name: str
    test_type: TestType
    tests: List['BaseTest'] = field(default_factory=list)
    setup_func: Optional[Callable] = None
    teardown_func: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestReport:
    """测试报告"""
    suite_name: str
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration: float = 0.0
    results: List[TestResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_tests == 0:
            return 0.0
        return self.passed / self.total_tests
    
    @property
    def is_successful(self) -> bool:
        """是否成功"""
        return self.failed == 0 and self.errors == 0


class BaseTest(ABC):
    """测试基类"""
    
    def __init__(self, name: str, test_type: TestType = TestType.UNIT):
        self.name = name
        self.test_type = test_type
        self.logger = logging.getLogger(f"Test.{name}")
        self.metadata: Dict[str, Any] = {}
        self._setup_done = False
        self._teardown_done = False
    
    def setup(self):
        """测试前置设置"""
        pass
    
    def teardown(self):
        """测试后置清理"""
        pass
    
    @abstractmethod
    def run_test(self) -> bool:
        """执行测试
        
        Returns:
            bool: 测试是否通过
        """
        pass
    
    def skip(self, reason: str = ""):
        """跳过测试"""
        raise TestSkippedException(reason)
    
    def assert_true(self, condition: bool, message: str = ""):
        """断言为真"""
        if not condition:
            raise AssertionError(message or "Assertion failed: expected True")
    
    def assert_false(self, condition: bool, message: str = ""):
        """断言为假"""
        if condition:
            raise AssertionError(message or "Assertion failed: expected False")
    
    def assert_equal(self, actual: Any, expected: Any, message: str = ""):
        """断言相等"""
        if actual != expected:
            raise AssertionError(
                message or f"Assertion failed: {actual} != {expected}"
            )
    
    def assert_not_equal(self, actual: Any, expected: Any, message: str = ""):
        """断言不相等"""
        if actual == expected:
            raise AssertionError(
                message or f"Assertion failed: {actual} == {expected}"
            )
    
    def assert_raises(self, exception_type: Type[Exception], func: Callable, *args, **kwargs):
        """断言抛出异常"""
        try:
            func(*args, **kwargs)
            raise AssertionError(f"Expected {exception_type.__name__} but no exception was raised")
        except exception_type:
            pass  # 期望的异常
        except Exception as e:
            raise AssertionError(f"Expected {exception_type.__name__} but got {type(e).__name__}: {e}")


class AsyncTest(BaseTest):
    """异步测试基类"""
    
    @abstractmethod
    async def run_async_test(self) -> bool:
        """异步执行测试"""
        pass
    
    def run_test(self) -> bool:
        """同步包装器"""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.run_async_test())
        except RuntimeError:
            return asyncio.run(self.run_async_test())


class PerformanceTest(BaseTest):
    """性能测试基类"""
    
    def __init__(self, name: str, max_duration: float = 1.0, min_throughput: float = 0.0):
        super().__init__(name, TestType.PERFORMANCE)
        self.max_duration = max_duration
        self.min_throughput = min_throughput
        self.performance_data: Dict[str, Any] = {}
    
    @abstractmethod
    def run_performance_test(self) -> Dict[str, Any]:
        """执行性能测试
        
        Returns:
            Dict[str, Any]: 性能数据
        """
        pass
    
    def run_test(self) -> bool:
        """执行性能测试并验证结果"""
        start_time = time.time()
        
        try:
            self.performance_data = self.run_performance_test()
            duration = time.time() - start_time
            
            # 验证性能指标
            if duration > self.max_duration:
                raise AssertionError(f"Test duration {duration:.3f}s exceeds maximum {self.max_duration}s")
            
            if self.min_throughput > 0:
                throughput = self.performance_data.get('throughput', 0)
                if throughput < self.min_throughput:
                    raise AssertionError(f"Throughput {throughput} below minimum {self.min_throughput}")
            
            self.performance_data['duration'] = duration
            return True
        
        except Exception as e:
            self.performance_data['error'] = str(e)
            raise


class TestSkippedException(Exception):
    """测试跳过异常"""
    pass


class MockObject:
    """模拟对象"""
    
    def __init__(self, **kwargs):
        self._mock_data = kwargs
        self._call_history: List[Dict[str, Any]] = []
    
    def __getattr__(self, name: str) -> Any:
        if name in self._mock_data:
            return self._mock_data[name]
        
        # 返回一个可调用的模拟方法
        def mock_method(*args, **kwargs):
            self._call_history.append({
                'method': name,
                'args': args,
                'kwargs': kwargs,
                'timestamp': datetime.now()
            })
            return self._mock_data.get(f"{name}_return")
        
        return mock_method
    
    def set_return_value(self, method_name: str, value: Any):
        """设置方法返回值"""
        self._mock_data[f"{method_name}_return"] = value
    
    def get_call_history(self, method_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取调用历史"""
        if method_name:
            return [call for call in self._call_history if call['method'] == method_name]
        return self._call_history
    
    def was_called(self, method_name: str) -> bool:
        """检查方法是否被调用"""
        return any(call['method'] == method_name for call in self._call_history)
    
    def call_count(self, method_name: str) -> int:
        """获取方法调用次数"""
        return len([call for call in self._call_history if call['method'] == method_name])


class TestDataManager:
    """测试数据管理器"""
    
    def __init__(self, data_dir: str = "test_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Any] = {}
    
    def load_test_data(self, filename: str) -> Any:
        """加载测试数据"""
        if filename in self._cache:
            return self._cache[filename]
        
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Test data file not found: {file_path}")
        
        try:
            if filename.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = f.read()
            
            self._cache[filename] = data
            return data
        
        except Exception as e:
            raise RuntimeError(f"Failed to load test data from {filename}: {e}")
    
    def save_test_data(self, filename: str, data: Any):
        """保存测试数据"""
        file_path = self.data_dir / filename
        
        try:
            if filename.endswith('.json'):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(str(data))
            
            self._cache[filename] = data
        
        except Exception as e:
            raise RuntimeError(f"Failed to save test data to {filename}: {e}")
    
    def create_mock_image(self, width: int = 640, height: int = 480, channels: int = 3) -> 'np.ndarray':
        """创建模拟图像数据"""
        try:
            import numpy as np
            return np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
        except ImportError:
            raise RuntimeError("NumPy is required for creating mock images")
    
    def create_mock_detection_result(self, num_objects: int = 3) -> Dict[str, Any]:
        """创建模拟检测结果"""
        import random
        
        objects = []
        for i in range(num_objects):
            objects.append({
                'class_id': random.randint(0, 10),
                'class_name': f'object_{i}',
                'confidence': random.uniform(0.5, 1.0),
                'bbox': {
                    'x': random.randint(0, 500),
                    'y': random.randint(0, 400),
                    'width': random.randint(50, 150),
                    'height': random.randint(50, 150)
                }
            })
        
        return {
            'objects': objects,
            'inference_time': random.uniform(0.01, 0.1),
            'image_size': {'width': 640, 'height': 480}
        }


class TestRunner:
    """测试运行器"""
    
    def __init__(self, max_workers: int = 4):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_workers = max_workers
        self.data_manager = TestDataManager()
        self._executor: Optional[ThreadPoolExecutor] = None
    
    def run_suite(self, suite: TestSuite) -> TestReport:
        """运行测试套件"""
        self.logger.info(f"Running test suite: {suite.name}")
        
        report = TestReport(
            suite_name=suite.name,
            total_tests=len(suite.tests)
        )
        
        start_time = time.time()
        
        try:
            # 执行套件设置
            if suite.setup_func:
                suite.setup_func()
            
            # 运行测试
            for test in suite.tests:
                result = self._run_single_test(test)
                report.results.append(result)
                
                # 更新统计
                if result.status == TestStatus.PASSED:
                    report.passed += 1
                elif result.status == TestStatus.FAILED:
                    report.failed += 1
                elif result.status == TestStatus.SKIPPED:
                    report.skipped += 1
                elif result.status == TestStatus.ERROR:
                    report.errors += 1
        
        except Exception as e:
            self.logger.error(f"Suite setup failed: {e}")
            report.errors += 1
        
        finally:
            # 执行套件清理
            if suite.teardown_func:
                try:
                    suite.teardown_func()
                except Exception as e:
                    self.logger.error(f"Suite teardown failed: {e}")
        
        report.duration = time.time() - start_time
        
        # 发布测试完成事件
        publish_event(EventType.CUSTOM, {
            'event_type': 'test_suite_completed',
            'suite_name': suite.name,
            'report': report.__dict__
        })
        
        self.logger.info(f"Test suite completed: {report.passed}/{report.total_tests} passed")
        return report
    
    def _run_single_test(self, test: BaseTest) -> TestResult:
        """运行单个测试"""
        result = TestResult(name=test.name, status=TestStatus.RUNNING)
        start_time = time.time()
        
        try:
            # 执行测试设置
            if not test._setup_done:
                test.setup()
                test._setup_done = True
            
            # 执行测试
            success = test.run_test()
            
            result.status = TestStatus.PASSED if success else TestStatus.FAILED
            
            # 添加性能数据
            if isinstance(test, PerformanceTest):
                result.metadata.update(test.performance_data)
        
        except TestSkippedException as e:
            result.status = TestStatus.SKIPPED
            result.message = str(e)
        
        except AssertionError as e:
            result.status = TestStatus.FAILED
            result.message = str(e)
        
        except Exception as e:
            result.status = TestStatus.ERROR
            result.error = str(e)
            result.traceback = traceback.format_exc()
        
        finally:
            # 执行测试清理
            if not test._teardown_done:
                try:
                    test.teardown()
                    test._teardown_done = True
                except Exception as e:
                    self.logger.error(f"Test teardown failed: {e}")
        
        result.duration = time.time() - start_time
        
        self.logger.debug(f"Test {test.name}: {result.status.value} ({result.duration:.3f}s)")
        return result
    
    def run_parallel(self, suites: List[TestSuite]) -> List[TestReport]:
        """并行运行多个测试套件"""
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        try:
            futures = []
            for suite in suites:
                future = self._executor.submit(self.run_suite, suite)
                futures.append(future)
            
            reports = []
            for future in futures:
                try:
                    report = future.result(timeout=300)  # 5分钟超时
                    reports.append(report)
                except Exception as e:
                    self.logger.error(f"Test suite execution failed: {e}")
            
            return reports
        
        finally:
            if self._executor:
                self._executor.shutdown(wait=True)
    
    def generate_html_report(self, reports: List[TestReport], output_file: str = "test_report.html"):
        """生成HTML测试报告"""
        html_content = self._create_html_report(reports)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report generated: {output_file}")
    
    def _create_html_report(self, reports: List[TestReport]) -> str:
        """创建HTML报告内容"""
        total_tests = sum(r.total_tests for r in reports)
        total_passed = sum(r.passed for r in reports)
        total_failed = sum(r.failed for r in reports)
        total_errors = sum(r.errors for r in reports)
        total_skipped = sum(r.skipped for r in reports)
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>YOLOS Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .passed {{ color: green; }}
        .failed {{ color: red; }}
        .error {{ color: orange; }}
        .skipped {{ color: blue; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>YOLOS Test Report</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p>Total Tests: {total_tests}</p>
        <p class="passed">Passed: {total_passed}</p>
        <p class="failed">Failed: {total_failed}</p>
        <p class="error">Errors: {total_errors}</p>
        <p class="skipped">Skipped: {total_skipped}</p>
        <p>Success Rate: {(total_passed/total_tests*100 if total_tests > 0 else 0):.1f}%</p>
    </div>
    
    <h2>Test Suites</h2>
    <table>
        <tr>
            <th>Suite Name</th>
            <th>Total</th>
            <th>Passed</th>
            <th>Failed</th>
            <th>Errors</th>
            <th>Skipped</th>
            <th>Duration</th>
            <th>Success Rate</th>
        </tr>
"""
        
        for report in reports:
            html += f"""
        <tr>
            <td>{report.suite_name}</td>
            <td>{report.total_tests}</td>
            <td class="passed">{report.passed}</td>
            <td class="failed">{report.failed}</td>
            <td class="error">{report.errors}</td>
            <td class="skipped">{report.skipped}</td>
            <td>{report.duration:.2f}s</td>
            <td>{report.success_rate*100:.1f}%</td>
        </tr>
"""
        
        html += """
    </table>
    
    <h2>Detailed Results</h2>
"""
        
        for report in reports:
            html += f"""
    <h3>{report.suite_name}</h3>
    <table>
        <tr>
            <th>Test Name</th>
            <th>Status</th>
            <th>Duration</th>
            <th>Message</th>
        </tr>
"""
            
            for result in report.results:
                status_class = result.status.value
                message = result.message or result.error or ""
                html += f"""
        <tr>
            <td>{result.name}</td>
            <td class="{status_class}">{result.status.value.upper()}</td>
            <td>{result.duration:.3f}s</td>
            <td>{message}</td>
        </tr>
"""
            
            html += "</table>"
        
        html += """
</body>
</html>
"""
        
        return html


@contextmanager
def test_application(config: Optional[ApplicationConfig] = None):
    """测试应用程序上下文管理器"""
    from .application import Application
    
    test_config = config or ApplicationConfig(
        name="YOLOS_Test",
        debug=True,
        log_level="DEBUG",
        auto_load_modules=False
    )
    
    app = Application(test_config)
    
    try:
        if not app.initialize():
            raise RuntimeError("Failed to initialize test application")
        yield app
    finally:
        app.shutdown()


# 便捷函数
def create_test_suite(name: str, test_type: TestType = TestType.UNIT) -> TestSuite:
    """创建测试套件"""
    return TestSuite(name=name, test_type=test_type)


def run_tests(suites: List[TestSuite], parallel: bool = False) -> List[TestReport]:
    """运行测试的便捷函数"""
    runner = TestRunner()
    
    if parallel:
        return runner.run_parallel(suites)
    else:
        return [runner.run_suite(suite) for suite in suites]