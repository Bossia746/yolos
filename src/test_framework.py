#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化测试框架
包含依赖检查、接口验证和模块完整性测试
"""

import sys
import os
import importlib
import inspect
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import traceback

class TestResult:
    """测试结果类"""
    def __init__(self, name: str, success: bool, message: str = "", details: str = ""):
        self.name = name
        self.success = success
        self.message = message
        self.details = details

class ModuleTestFramework:
    """模块测试框架"""
    
    def __init__(self, src_path: str = None):
        self.src_path = src_path or os.path.dirname(os.path.abspath(__file__))
        self.results: List[TestResult] = []
        self.modules_tested = 0
        self.modules_passed = 0
        
    def add_result(self, result: TestResult):
        """添加测试结果"""
        self.results.append(result)
        self.modules_tested += 1
        if result.success:
            self.modules_passed += 1
            
    def test_module_import(self, module_name: str, expected_items: List[str] = None) -> TestResult:
        """测试模块导入"""
        try:
            # 尝试导入模块
            module = importlib.import_module(module_name)
            
            # 检查预期的项目
            missing_items = []
            if expected_items:
                for item in expected_items:
                    if not hasattr(module, item):
                        missing_items.append(item)
            
            if missing_items:
                return TestResult(
                    name=f"Import {module_name}",
                    success=False,
                    message=f"Missing items: {', '.join(missing_items)}",
                    details=f"Available items: {', '.join(dir(module))}"
                )
            
            return TestResult(
                name=f"Import {module_name}",
                success=True,
                message=f"Successfully imported with {len(dir(module))} items"
            )
            
        except Exception as e:
            return TestResult(
                name=f"Import {module_name}",
                success=False,
                message=f"Import failed: {str(e)}",
                details=traceback.format_exc()
            )
    
    def test_class_interface(self, module_name: str, class_name: str, expected_methods: List[str] = None) -> TestResult:
        """测试类接口"""
        try:
            module = importlib.import_module(module_name)
            if not hasattr(module, class_name):
                return TestResult(
                    name=f"Class {class_name} in {module_name}",
                    success=False,
                    message=f"Class {class_name} not found in module"
                )
            
            cls = getattr(module, class_name)
            
            # 检查是否是类
            if not inspect.isclass(cls):
                return TestResult(
                    name=f"Class {class_name} in {module_name}",
                    success=False,
                    message=f"{class_name} is not a class"
                )
            
            # 检查预期方法
            missing_methods = []
            if expected_methods:
                for method in expected_methods:
                    if not hasattr(cls, method):
                        missing_methods.append(method)
            
            if missing_methods:
                return TestResult(
                    name=f"Class {class_name} in {module_name}",
                    success=False,
                    message=f"Missing methods: {', '.join(missing_methods)}",
                    details=f"Available methods: {', '.join([m for m in dir(cls) if not m.startswith('_')])}"
                )
            
            return TestResult(
                name=f"Class {class_name} in {module_name}",
                success=True,
                message=f"Class interface validated successfully"
            )
            
        except Exception as e:
            return TestResult(
                name=f"Class {class_name} in {module_name}",
                success=False,
                message=f"Interface test failed: {str(e)}",
                details=traceback.format_exc()
            )
    
    def test_dependency_chain(self, modules: List[str]) -> TestResult:
        """测试模块依赖链"""
        try:
            imported_modules = {}
            for module_name in modules:
                try:
                    module = importlib.import_module(module_name)
                    imported_modules[module_name] = module
                except Exception as e:
                    return TestResult(
                        name="Dependency Chain",
                        success=False,
                        message=f"Failed to import {module_name}: {str(e)}",
                        details=f"Dependency chain broken at {module_name}"
                    )
            
            return TestResult(
                name="Dependency Chain",
                success=True,
                message=f"All {len(modules)} modules in dependency chain imported successfully"
            )
            
        except Exception as e:
            return TestResult(
                name="Dependency Chain",
                success=False,
                message=f"Dependency chain test failed: {str(e)}",
                details=traceback.format_exc()
            )
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """运行综合测试"""
        print("\n=== YOLOS 自动化测试框架 ===")
        print(f"测试路径: {self.src_path}")
        print("\n开始模块测试...\n")
        
        # 核心模块测试
        core_items = [
            "ConfigManager", "DataManager", "EventBus", "PluginManager",
            "SystemException", "ModelException", "DataException", "ConfigurationError",
            "configure_logging", "YOLOSLogger", "CrossPlatformManager"
        ]
        result = self.test_module_import("core", core_items)
        self.add_result(result)
        print(f"✓ 核心模块: {result.message}")
        
        # 工具模块测试
        utils_items = ["ConfigManager", "FileUtils", "Visualizer", "MetricsCalculator"]
        result = self.test_module_import("utils", utils_items)
        self.add_result(result)
        print(f"✓ 工具模块: {result.message}")
        
        # 模型模块测试
        models_items = ["YOLOFactory", "BaseModel"]
        result = self.test_module_import("models", models_items)
        self.add_result(result)
        if result.success:
            print(f"✓ 模型模块: {result.message}")
        else:
            print(f"✗ 模型模块: {result.message}")
        
        # 检测模块测试
        detection_items = ["DetectorFactory"]
        result = self.test_module_import("detection", detection_items)
        self.add_result(result)
        if result.success:
            print(f"✓ 检测模块: {result.message}")
        else:
            print(f"✗ 检测模块: {result.message}")
        
        # 识别模块测试
        recognition_items = ["RecognitionFactory"]
        result = self.test_module_import("recognition", recognition_items)
        self.add_result(result)
        if result.success:
            print(f"✓ 识别模块: {result.message}")
        else:
            print(f"✗ 识别模块: {result.message}")
        
        # 类接口测试
        print("\n开始接口测试...\n")
        
        # 测试ConfigManager接口
        result = self.test_class_interface("core", "ConfigManager", ["get_config", "set_config", "reload_config"])
        self.add_result(result)
        if result.success:
            print(f"✓ ConfigManager接口: {result.message}")
        else:
            print(f"✗ ConfigManager接口: {result.message}")
        
        # 测试DataManager接口
        result = self.test_class_interface("core", "DataManager", ["store_training_data", "retrieve_data", "delete_data"])
        self.add_result(result)
        if result.success:
            print(f"✓ DataManager接口: {result.message}")
        else:
            print(f"✗ DataManager接口: {result.message}")
        
        # 依赖链测试
        print("\n开始依赖链测试...\n")
        
        dependency_modules = ["core", "utils"]
        result = self.test_dependency_chain(dependency_modules)
        self.add_result(result)
        print(f"✓ 依赖链测试: {result.message}")
        
        # 生成测试报告
        return self.generate_report()
    
    def generate_report(self) -> Dict[str, Any]:
        """生成测试报告"""
        print("\n" + "="*50)
        print("测试报告")
        print("="*50)
        
        success_rate = (self.modules_passed / self.modules_tested * 100) if self.modules_tested > 0 else 0
        
        print(f"总测试数: {self.modules_tested}")
        print(f"通过测试: {self.modules_passed}")
        print(f"失败测试: {self.modules_tested - self.modules_passed}")
        print(f"成功率: {success_rate:.1f}%")
        
        print("\n详细结果:")
        for result in self.results:
            status = "✓" if result.success else "✗"
            print(f"{status} {result.name}: {result.message}")
            if not result.success and result.details:
                print(f"   详情: {result.details[:200]}...")
        
        # 建议
        failed_tests = [r for r in self.results if not r.success]
        if failed_tests:
            print("\n修复建议:")
            for result in failed_tests:
                if "Import" in result.name and "relative import" in result.message:
                    print(f"- {result.name}: 修复相对导入问题，检查__init__.py文件")
                elif "Missing" in result.message:
                    print(f"- {result.name}: 添加缺失的项目或更新导入列表")
                else:
                    print(f"- {result.name}: {result.message}")
        
        return {
            "total_tests": self.modules_tested,
            "passed_tests": self.modules_passed,
            "success_rate": success_rate,
            "results": [(r.name, r.success, r.message) for r in self.results]
        }

def main():
    """主函数"""
    # 确保在src目录下运行
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not current_dir.endswith('src'):
        print("警告: 请在src目录下运行此脚本")
        return
    
    # 添加src目录到Python路径
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # 运行测试框架
    framework = ModuleTestFramework(current_dir)
    report = framework.run_comprehensive_tests()
    
    # 返回退出码
    if report["success_rate"] == 100:
        print("\n🎉 所有测试通过!")
        sys.exit(0)
    else:
        print(f"\n⚠️  测试完成，成功率: {report['success_rate']:.1f}%")
        sys.exit(1)

if __name__ == "__main__":
    main()