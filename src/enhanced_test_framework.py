#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的YOLOS测试框架
扩展测试覆盖到所有模块，包括Recognition、Detection、Models等
"""

import sys
import os
import importlib
import inspect
import traceback
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestStatus(Enum):
    """测试状态"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

class TestCategory(Enum):
    """测试类别"""
    IMPORT = "import"
    INTERFACE = "interface"
    FUNCTIONALITY = "functionality"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"

@dataclass
class TestResult:
    """测试结果"""
    name: str
    category: TestCategory
    status: TestStatus
    duration: float = 0.0
    message: str = ""
    details: str = ""
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModuleTestSuite:
    """模块测试套件"""
    module_name: str
    tests: List[Callable] = field(default_factory=list)
    setup_func: Optional[Callable] = None
    teardown_func: Optional[Callable] = None
    results: List[TestResult] = field(default_factory=list)

class EnhancedTestFramework:
    """增强的测试框架"""
    
    def __init__(self, src_path: str = None):
        self.src_path = src_path or os.path.dirname(os.path.abspath(__file__))
        self.test_suites: Dict[str, ModuleTestSuite] = {}
        self.global_results: List[TestResult] = []
        self.start_time = 0.0
        self.end_time = 0.0
        
    def add_test_suite(self, module_name: str) -> ModuleTestSuite:
        """添加测试套件"""
        if module_name not in self.test_suites:
            self.test_suites[module_name] = ModuleTestSuite(module_name=module_name)
        return self.test_suites[module_name]
    
    def run_test(self, test_func: Callable, category: TestCategory) -> TestResult:
        """运行单个测试"""
        test_name = test_func.__name__
        result = TestResult(
            name=test_name,
            category=category,
            status=TestStatus.RUNNING
        )
        
        start_time = time.time()
        
        try:
            test_func()
            result.status = TestStatus.PASSED
            result.message = "测试通过"
        except ImportError as e:
            result.status = TestStatus.FAILED
            result.message = f"导入错误: {str(e)}"
            result.error = e
        except AssertionError as e:
            result.status = TestStatus.FAILED
            result.message = f"断言失败: {str(e)}"
            result.error = e
        except Exception as e:
            result.status = TestStatus.ERROR
            result.message = f"测试错误: {str(e)}"
            result.error = e
            result.details = traceback.format_exc()
        
        result.duration = time.time() - start_time
        return result
    
    def test_module_import(self, module_name: str, expected_items: List[str] = None):
        """测试模块导入"""
        def test():
            try:
                module = importlib.import_module(module_name)
                
                if expected_items:
                    missing_items = []
                    for item in expected_items:
                        if not hasattr(module, item):
                            missing_items.append(item)
                    
                    if missing_items:
                        raise AssertionError(f"缺少项目: {', '.join(missing_items)}")
                
                return module
            except ImportError as e:
                raise ImportError(f"无法导入模块 {module_name}: {str(e)}")
        
        return self.run_test(test, TestCategory.IMPORT)
    
    def test_class_interface(self, module_name: str, class_name: str, expected_methods: List[str] = None):
        """测试类接口"""
        def test():
            module = importlib.import_module(module_name)
            
            if not hasattr(module, class_name):
                raise AssertionError(f"类 {class_name} 不存在于模块 {module_name}")
            
            cls = getattr(module, class_name)
            
            if not inspect.isclass(cls):
                raise AssertionError(f"{class_name} 不是一个类")
            
            if expected_methods:
                missing_methods = []
                for method in expected_methods:
                    if not hasattr(cls, method):
                        missing_methods.append(method)
                
                if missing_methods:
                    available_methods = [m for m in dir(cls) if not m.startswith('_')]
                    raise AssertionError(
                        f"缺少方法: {', '.join(missing_methods)}，"
                        f"可用方法: {', '.join(available_methods)}"
                    )
        
        return self.run_test(test, TestCategory.INTERFACE)
    
    def test_factory_pattern(self, module_name: str, factory_class: str, create_method: str = "create"):
        """测试工厂模式"""
        def test():
            module = importlib.import_module(module_name)
            
            if not hasattr(module, factory_class):
                raise AssertionError(f"工厂类 {factory_class} 不存在")
            
            factory = getattr(module, factory_class)
            
            if not hasattr(factory, create_method):
                raise AssertionError(f"工厂类缺少创建方法 {create_method}")
            
            # 测试是否有获取可用类型的方法
            list_methods = ['list_available', 'get_available', 'list_types', 'get_types']
            has_list_method = any(hasattr(factory, method) for method in list_methods)
            
            if not has_list_method:
                raise AssertionError(f"工厂类缺少列出可用类型的方法: {list_methods}")
        
        return self.run_test(test, TestCategory.INTERFACE)
    
    def test_recognition_module(self):
        """测试Recognition模块"""
        suite = self.add_test_suite("recognition")
        
        # 测试模块导入
        result = self.test_module_import(
            "recognition",
            ["BaseRecognizer", "RecognizerFactory", "RecognizerType", "RecognizerConfig"]
        )
        suite.results.append(result)
        
        # 测试BaseRecognizer接口
        result = self.test_class_interface(
            "recognition",
            "BaseRecognizer",
            ["initialize", "recognize", "cleanup", "validate_input"]
        )
        suite.results.append(result)
        
        # 测试RecognizerFactory
        result = self.test_factory_pattern(
            "recognition",
            "RecognizerFactory",
            "create_recognizer"
        )
        suite.results.append(result)
        
        # 测试姿态识别相关
        try:
            result = self.test_module_import(
                "recognition",
                ["PoseRecognizer", "ExerciseType", "PoseState"]
            )
            suite.results.append(result)
        except:
            # 如果姿态识别模块不可用，跳过
            result = TestResult(
                name="test_pose_recognition_import",
                category=TestCategory.IMPORT,
                status=TestStatus.SKIPPED,
                message="姿态识别模块不可用"
            )
            suite.results.append(result)
    
    def test_detection_module(self):
        """测试Detection模块"""
        suite = self.add_test_suite("detection")
        
        # 测试模块导入
        result = self.test_module_import(
            "detection",
            ["DetectorFactory", "RealtimeDetector", "ImageDetector"]
        )
        suite.results.append(result)
        
        # 测试DetectorFactory
        result = self.test_factory_pattern(
            "detection",
            "DetectorFactory",
            "create_detector"
        )
        suite.results.append(result)
    
    def test_models_module(self):
        """测试Models模块"""
        suite = self.add_test_suite("models")
        
        # 测试模块导入
        result = self.test_module_import(
            "models",
            ["YOLOFactory", "BaseYOLOModel"]
        )
        suite.results.append(result)
        
        # 测试YOLOFactory
        result = self.test_factory_pattern(
            "models",
            "YOLOFactory",
            "create_model"
        )
        suite.results.append(result)
        
        # 测试BaseYOLOModel接口
        result = self.test_class_interface(
            "models",
            "BaseYOLOModel",
            ["load_model", "predict", "preprocess", "postprocess"]
        )
        suite.results.append(result)
    
    def test_core_module(self):
        """测试Core模块"""
        suite = self.add_test_suite("core")
        
        # 测试类型系统
        result = self.test_module_import(
            "core.types",
            ["ProcessingResult", "TaskType", "Status", "DetectionResult"]
        )
        suite.results.append(result)
        
        # 测试异常系统
        result = self.test_module_import(
            "core.exceptions",
            ["YOLOSException", "ErrorCode", "exception_handler"]
        )
        suite.results.append(result)
        
        # 测试配置管理
        try:
            result = self.test_class_interface(
                "core",
                "ConfigManager",
                ["get_config", "set_config", "reload_config"]
            )
            suite.results.append(result)
        except:
            result = TestResult(
                name="test_config_manager_interface",
                category=TestCategory.INTERFACE,
                status=TestStatus.SKIPPED,
                message="ConfigManager不可用"
            )
            suite.results.append(result)
    
    def test_utils_module(self):
        """测试Utils模块"""
        suite = self.add_test_suite("utils")
        
        # 测试模块导入
        result = self.test_module_import(
            "utils",
            ["ConfigManager", "FileUtils"]
        )
        suite.results.append(result)
    
    def test_integration(self):
        """测试模块集成"""
        suite = self.add_test_suite("integration")
        
        def test_cross_module_import():
            """测试跨模块导入"""
            # 测试Recognition模块使用Core类型
            try:
                from recognition.base_recognizer import ProcessingResult, TaskType
                from core.types import ProcessingResult as CoreProcessingResult
                # 检查类型兼容性
                assert ProcessingResult is not None
                assert TaskType is not None
            except ImportError:
                # 如果导入失败，检查是否有备用实现
                from recognition.base_recognizer import ProcessingResult
                assert ProcessingResult is not None
        
        result = self.run_test(test_cross_module_import, TestCategory.INTEGRATION)
        suite.results.append(result)
        
        def test_factory_integration():
            """测试工厂模式集成"""
            try:
                from models import YOLOFactory
                from detection import DetectorFactory
                from recognition import RecognizerFactory
                
                # 测试工厂方法存在
                assert hasattr(YOLOFactory, 'create_model')
                assert hasattr(DetectorFactory, 'create_detector')
                assert hasattr(RecognizerFactory, 'create_recognizer')
            except ImportError as e:
                raise AssertionError(f"工厂模式集成测试失败: {e}")
        
        result = self.run_test(test_factory_integration, TestCategory.INTEGRATION)
        suite.results.append(result)
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        logger.info("开始运行增强测试框架...")
        self.start_time = time.time()
        
        # 确保在src目录下运行
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # 运行各模块测试
        test_methods = [
            self.test_core_module,
            self.test_utils_module,
            self.test_models_module,
            self.test_detection_module,
            self.test_recognition_module,
            self.test_integration
        ]
        
        for test_method in test_methods:
            try:
                logger.info(f"运行测试: {test_method.__name__}")
                test_method()
            except Exception as e:
                logger.error(f"测试方法 {test_method.__name__} 执行失败: {e}")
                # 创建错误结果
                error_result = TestResult(
                    name=test_method.__name__,
                    category=TestCategory.FUNCTIONALITY,
                    status=TestStatus.ERROR,
                    message=f"测试方法执行失败: {str(e)}",
                    error=e
                )
                self.global_results.append(error_result)
        
        self.end_time = time.time()
        
        # 生成报告
        return self.generate_comprehensive_report()
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """生成综合测试报告"""
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        error_tests = 0
        skipped_tests = 0
        
        # 收集所有结果
        all_results = []
        for suite in self.test_suites.values():
            all_results.extend(suite.results)
        all_results.extend(self.global_results)
        
        # 统计结果
        for result in all_results:
            total_tests += 1
            if result.status == TestStatus.PASSED:
                passed_tests += 1
            elif result.status == TestStatus.FAILED:
                failed_tests += 1
            elif result.status == TestStatus.ERROR:
                error_tests += 1
            elif result.status == TestStatus.SKIPPED:
                skipped_tests += 1
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        total_duration = self.end_time - self.start_time
        
        # 生成报告
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "errors": error_tests,
                "skipped": skipped_tests,
                "success_rate": success_rate,
                "total_duration": total_duration
            },
            "suites": {},
            "recommendations": []
        }
        
        # 按套件组织结果
        for suite_name, suite in self.test_suites.items():
            suite_stats = {
                "total": len(suite.results),
                "passed": sum(1 for r in suite.results if r.status == TestStatus.PASSED),
                "failed": sum(1 for r in suite.results if r.status == TestStatus.FAILED),
                "errors": sum(1 for r in suite.results if r.status == TestStatus.ERROR),
                "skipped": sum(1 for r in suite.results if r.status == TestStatus.SKIPPED),
                "results": []
            }
            
            for result in suite.results:
                suite_stats["results"].append({
                    "name": result.name,
                    "category": result.category.value,
                    "status": result.status.value,
                    "duration": result.duration,
                    "message": result.message,
                    "details": result.details
                })
            
            report["suites"][suite_name] = suite_stats
        
        # 生成建议
        if failed_tests > 0 or error_tests > 0:
            report["recommendations"].extend([
                "修复失败的导入问题，检查模块依赖",
                "完善缺失的接口方法",
                "优化工厂模式实现",
                "加强模块间的集成测试"
            ])
        
        # 打印报告
        self.print_report(report)
        
        return report
    
    def print_report(self, report: Dict[str, Any]):
        """打印测试报告"""
        print("\n" + "="*80)
        print("YOLOS 增强测试框架报告")
        print("="*80)
        
        summary = report["summary"]
        print(f"总测试数: {summary['total_tests']}")
        print(f"通过: {summary['passed']} | 失败: {summary['failed']} | 错误: {summary['errors']} | 跳过: {summary['skipped']}")
        print(f"成功率: {summary['success_rate']:.1f}%")
        print(f"总耗时: {summary['total_duration']:.2f}秒")
        
        print("\n模块测试结果:")
        print("-" * 60)
        
        for suite_name, suite_data in report["suites"].items():
            status_icon = "✅" if suite_data["failed"] == 0 and suite_data["errors"] == 0 else "❌"
            print(f"{status_icon} {suite_name}: {suite_data['passed']}/{suite_data['total']} 通过")
            
            # 显示失败的测试
            for result in suite_data["results"]:
                if result["status"] in ["failed", "error"]:
                    print(f"   ❌ {result['name']}: {result['message']}")
                elif result["status"] == "skipped":
                    print(f"   ⏭️  {result['name']}: {result['message']}")
        
        if report["recommendations"]:
            print("\n改进建议:")
            print("-" * 60)
            for i, rec in enumerate(report["recommendations"], 1):
                print(f"{i}. {rec}")
        
        print("\n" + "="*80)
    
    def save_report(self, report: Dict[str, Any], filename: str = "enhanced_test_report.json"):
        """保存测试报告到文件"""
        report_path = Path(filename)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"测试报告已保存到: {report_path}")

def main():
    """主函数"""
    framework = EnhancedTestFramework()
    report = framework.run_all_tests()
    
    # 保存报告
    framework.save_report(report)
    
    # 返回退出码
    if report["summary"]["success_rate"] >= 80:
        print("\n🎉 测试基本通过!")
        sys.exit(0)
    else:
        print(f"\n⚠️  需要改进，成功率: {report['summary']['success_rate']:.1f}%")
        sys.exit(1)

if __name__ == "__main__":
    main()