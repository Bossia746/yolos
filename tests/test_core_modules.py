#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心模块测试

测试YOLOS核心模块的功能，包括配置管理、事件系统、依赖注入等。

作者: YOLOS团队
日期: 2024
"""

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

# 导入测试框架
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from core.test_framework import (
    BaseTest, AsyncTest, PerformanceTest, TestType, TestSuite,
    MockObject, TestDataManager, test_application
)

# 导入被测试的模块
from core.config_manager import ConfigManager, ConfigSource, ConfigSourceType, ConfigFormat
from core.event_system import EventBus, Event, EventType, EventPriority, IEventHandler
from core.dependency_injection import ServiceContainer, ServiceLifetime
from core.exceptions import ExceptionHandler
from core.performance_monitor import PerformanceMonitor
from core.application import Application, ApplicationConfig


class TestConfigManager(BaseTest):
    """配置管理器测试"""
    
    def __init__(self):
        super().__init__("ConfigManager", TestType.UNIT)
        self.temp_dir = None
        self.config_manager = None
    
    def setup(self):
        """测试前置设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager()
        
        # 创建测试配置文件
        config_data = {
            'application': {
                'name': 'test_app',
                'version': '1.0.0'
            },
            'database': {
                'host': 'localhost',
                'port': 5432
            }
        }
        
        config_file = Path(self.temp_dir) / 'test_config.yaml'
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # 添加测试配置源
        source = ConfigSource(
            name="test_config",
            source_type=ConfigSourceType.FILE,
            location=str(config_file),
            format=ConfigFormat.YAML,
            priority=100
        )
        self.config_manager.add_source(source)
    
    def teardown(self):
        """测试后置清理"""
        import shutil
        if self.temp_dir:
            shutil.rmtree(self.temp_dir)
    
    def run_test(self) -> bool:
        """执行配置管理器测试"""
        # 测试配置加载
        config = self.config_manager.load_all_configs()
        self.assert_true('application' in config, "应用配置应该存在")
        self.assert_equal(config['application']['name'], 'test_app', "应用名称应该正确")
        
        # 测试配置获取
        app_name = self.config_manager.get_config('application.name')
        self.assert_equal(app_name, 'test_app', "通过路径获取配置应该正确")
        
        # 测试配置设置
        self.config_manager.set_config('application.debug', True)
        debug_value = self.config_manager.get_config('application.debug')
        self.assert_true(debug_value, "设置的配置值应该正确")
        
        # 测试默认值
        default_value = self.config_manager.get_config('nonexistent.key', 'default')
        self.assert_equal(default_value, 'default', "不存在的键应该返回默认值")
        
        return True


class TestEventSystem(BaseTest):
    """事件系统测试"""
    
    def __init__(self):
        super().__init__("EventSystem", TestType.UNIT)
        self.event_bus = None
        self.received_events = []
    
    def setup(self):
        """测试前置设置"""
        self.event_bus = EventBus()
        self.received_events = []
    
    def run_test(self) -> bool:
        """执行事件系统测试"""
        # 创建测试事件处理器
        class TestHandler(IEventHandler):
            def __init__(self, test_instance):
                self.test_instance = test_instance
            
            def handle(self, event: Event) -> bool:
                self.test_instance.received_events.append(event)
                return True
        
        handler = TestHandler(self)
        
        # 测试事件订阅
        subscription_id = self.event_bus.subscribe(EventType.SYSTEM_STARTUP, handler)
        self.assert_true(subscription_id is not None, "订阅应该返回ID")
        
        # 测试事件发布
        test_event = Event(
            type=EventType.SYSTEM_STARTUP,
            data={'test': 'data'},
            source='test'
        )
        
        success = self.event_bus.publish(test_event)
        self.assert_true(success, "事件发布应该成功")
        
        # 验证事件接收
        self.assert_equal(len(self.received_events), 1, "应该接收到一个事件")
        received_event = self.received_events[0]
        self.assert_equal(received_event.type, EventType.SYSTEM_STARTUP.value, "事件类型应该正确")
        self.assert_equal(received_event.data['test'], 'data', "事件数据应该正确")
        
        # 测试事件取消订阅
        self.event_bus.unsubscribe(EventType.SYSTEM_STARTUP, handler)
        
        # 发布另一个事件，应该不会被接收
        self.event_bus.publish(test_event)
        self.assert_equal(len(self.received_events), 1, "取消订阅后不应该接收新事件")
        
        return True


class TestDependencyInjection(BaseTest):
    """依赖注入测试"""
    
    def __init__(self):
        super().__init__("DependencyInjection", TestType.UNIT)
        self.container = None
    
    def setup(self):
        """测试前置设置"""
        self.container = ServiceContainer()
    
    def run_test(self) -> bool:
        """执行依赖注入测试"""
        # 定义测试服务
        class TestService:
            def __init__(self, name: str = "test"):
                self.name = name
            
            def get_name(self) -> str:
                return self.name
        
        class DependentService:
            def __init__(self, test_service: TestService):
                self.test_service = test_service
            
            def get_service_name(self) -> str:
                return self.test_service.get_name()
        
        # 测试单例注册
        self.container.register_singleton(TestService, lambda: TestService("singleton"))
        
        # 测试服务解析
        service1 = self.container.resolve(TestService)
        service2 = self.container.resolve(TestService)
        
        self.assert_true(service1 is service2, "单例服务应该返回同一个实例")
        self.assert_equal(service1.get_name(), "singleton", "服务名称应该正确")
        
        # 测试瞬态注册
        self.container.register_transient(DependentService, DependentService)
        
        dependent1 = self.container.resolve(DependentService)
        dependent2 = self.container.resolve(DependentService)
        
        self.assert_true(dependent1 is not dependent2, "瞬态服务应该返回不同实例")
        self.assert_equal(dependent1.get_service_name(), "singleton", "依赖注入应该正确")
        
        return True


class TestExceptionHandler(BaseTest):
    """异常处理器测试"""
    
    def __init__(self):
        super().__init__("ExceptionHandler", TestType.UNIT)
        self.exception_handler = None
        self.handled_exceptions = []
    
    def setup(self):
        """测试前置设置"""
        self.exception_handler = ExceptionHandler()
        self.handled_exceptions = []
        
        # 添加测试处理器
        def test_handler(exc_info, context):
            self.handled_exceptions.append((exc_info, context))
            return True
        
        self.exception_handler.add_handler(Exception, test_handler)
    
    def run_test(self) -> bool:
        """执行异常处理器测试"""
        # 测试异常处理
        test_exception = ValueError("Test exception")
        context = {'source': 'test'}
        
        handled = self.exception_handler.handle_exception(
            test_exception, context, ExceptionSeverity.WARNING
        )
        
        self.assert_true(handled, "异常应该被处理")
        self.assert_equal(len(self.handled_exceptions), 1, "应该记录一个异常")
        
        exc_info, handled_context = self.handled_exceptions[0]
        self.assert_true(isinstance(exc_info[1], ValueError), "异常类型应该正确")
        self.assert_equal(handled_context['source'], 'test', "上下文应该正确")
        
        return True


class TestPerformanceMonitor(PerformanceTest):
    """性能监控器测试"""
    
    def __init__(self):
        super().__init__("PerformanceMonitor", max_duration=2.0)
        self.monitor = None
    
    def setup(self):
        """测试前置设置"""
        self.monitor = PerformanceMonitor()
    
    def run_performance_test(self) -> dict:
        """执行性能监控测试"""
        # 启动监控
        self.monitor.start_monitoring()
        
        # 模拟一些工作负载
        time.sleep(0.1)
        
        # 获取性能指标
        metrics = self.monitor.get_current_metrics()
        
        # 停止监控
        self.monitor.stop_monitoring()
        
        # 验证指标
        self.assert_true(metrics.cpu_usage >= 0, "CPU使用率应该非负")
        self.assert_true(metrics.memory_usage >= 0, "内存使用率应该非负")
        
        return {
            'cpu_usage': metrics.cpu_usage,
            'memory_usage': metrics.memory_usage,
            'throughput': 100.0  # 模拟吞吐量
        }


class TestApplicationIntegration(BaseTest):
    """应用程序集成测试"""
    
    def __init__(self):
        super().__init__("ApplicationIntegration", TestType.INTEGRATION)
    
    def run_test(self) -> bool:
        """执行应用程序集成测试"""
        # 使用测试应用程序上下文
        with test_application() as app:
            # 验证应用程序初始化
            self.assert_true(app.config_manager is not None, "配置管理器应该初始化")
            self.assert_true(app.event_bus is not None, "事件总线应该初始化")
            self.assert_true(app.container is not None, "依赖容器应该初始化")
            
            # 测试服务获取
            config_manager = app.get_service(ConfigManager)
            self.assert_true(config_manager is not None, "应该能获取配置管理器服务")
            
            # 测试应用程序状态
            status = app.get_status()
            self.assert_equal(status['name'], 'YOLOS_Test', "应用名称应该正确")
            
        return True


class TestAsyncEventHandling(AsyncTest):
    """异步事件处理测试"""
    
    def __init__(self):
        super().__init__("AsyncEventHandling", TestType.UNIT)
        self.event_bus = None
        self.async_results = []
    
    def setup(self):
        """测试前置设置"""
        self.event_bus = EventBus()
        self.async_results = []
    
    async def run_async_test(self) -> bool:
        """执行异步事件处理测试"""
        from core.event_system import AsyncEventHandler
        
        class AsyncTestHandler(AsyncEventHandler):
            def __init__(self, test_instance):
                self.test_instance = test_instance
            
            async def handle_async(self, event: Event) -> bool:
                # 模拟异步处理
                await asyncio.sleep(0.1)
                self.test_instance.async_results.append(event.data)
                return True
        
        handler = AsyncTestHandler(self)
        
        # 订阅异步事件
        self.event_bus.subscribe(EventType.CUSTOM, handler)
        
        # 发布测试事件
        test_event = Event(
            type=EventType.CUSTOM,
            data={'async_test': True}
        )
        
        success = self.event_bus.publish(test_event, async_mode=True)
        
        # 等待异步处理完成
        await asyncio.sleep(0.2)
        
        self.assert_true(success, "异步事件发布应该成功")
        self.assert_equal(len(self.async_results), 1, "应该接收到异步事件")
        self.assert_true(self.async_results[0]['async_test'], "异步事件数据应该正确")
        
        return True


def create_core_test_suites() -> list:
    """创建核心模块测试套件"""
    # 单元测试套件
    unit_suite = TestSuite("CoreModulesUnit", TestType.UNIT)
    unit_suite.tests = [
        TestConfigManager(),
        TestEventSystem(),
        TestDependencyInjection(),
        TestExceptionHandler(),
        TestAsyncEventHandling()
    ]
    
    # 性能测试套件
    performance_suite = TestSuite("CoreModulesPerformance", TestType.PERFORMANCE)
    performance_suite.tests = [
        TestPerformanceMonitor()
    ]
    
    # 集成测试套件
    integration_suite = TestSuite("CoreModulesIntegration", TestType.INTEGRATION)
    integration_suite.tests = [
        TestApplicationIntegration()
    ]
    
    return [unit_suite, performance_suite, integration_suite]


if __name__ == "__main__":
    import sys
    from core.test_framework import TestRunner
    
    # 运行核心模块测试
    suites = create_core_test_suites()
    runner = TestRunner()
    
    reports = []
    for suite in suites:
        report = runner.run_suite(suite)
        reports.append(report)
        
        print(f"\n=== {suite.name} ===")
        print(f"Total: {report.total_tests}, Passed: {report.passed}, Failed: {report.failed}")
        print(f"Success Rate: {report.success_rate*100:.1f}%")
        
        if report.failed > 0 or report.errors > 0:
            print("\nFailed Tests:")
            for result in report.results:
                if result.status.value in ['failed', 'error']:
                    print(f"  - {result.name}: {result.message or result.error}")
    
    # 生成HTML报告
    runner.generate_html_report(reports, "core_modules_test_report.html")
    
    # 返回退出码
    total_failed = sum(r.failed + r.errors for r in reports)
    sys.exit(0 if total_failed == 0 else 1)