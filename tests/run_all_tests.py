#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整测试套件运行器

整合所有测试模块，提供统一的测试入口和报告生成。

作者: YOLOS团队
日期: 2024
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

# 导入测试框架
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from core.test_framework import (
    TestRunner, TestSuite, TestType, TestResult, TestReport
)

# 导入所有测试模块
try:
    from test_core_modules import create_core_test_suites
    from test_applications import create_application_test_suites
except ImportError as e:
    print(f"警告: 无法导入测试模块: {e}")
    
    def create_core_test_suites():
        return []
    
    def create_application_test_suites():
        return []

# 尝试导入平台兼容性测试
try:
    from test_platform_compatibility import create_platform_compatibility_test_suites
except ImportError:
    def create_platform_compatibility_test_suites():
        return []


class ComprehensiveTestRunner:
    """综合测试运行器"""
    
    def __init__(self, output_dir: str = "test_reports"):
        self.runner = TestRunner()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.all_suites = {
            'core': create_core_test_suites,
            'applications': create_application_test_suites,
            'platform': create_platform_compatibility_test_suites
        }
    
    def run_test_category(self, category: str, test_types: List[str] = None) -> List[TestReport]:
        """运行指定类别的测试"""
        if category not in self.all_suites:
            print(f"错误: 未知的测试类别 '{category}'")
            return []
        
        print(f"\n=== 运行 {category.upper()} 测试 ===")
        
        try:
            suites = self.all_suites[category]()
        except Exception as e:
            print(f"错误: 无法创建{category}测试套件: {e}")
            return []
        
        if not suites:
            print(f"警告: {category}测试套件为空")
            return []
        
        # 过滤测试类型
        if test_types:
            filtered_suites = []
            for suite in suites:
                if any(tt.lower() in suite.name.lower() for tt in test_types):
                    filtered_suites.append(suite)
            suites = filtered_suites
        
        reports = []
        for suite in suites:
            print(f"\n--- 运行套件: {suite.name} ---")
            try:
                report = self.runner.run_suite(suite)
                reports.append(report)
                
                # 显示结果摘要
                print(f"总计: {report.total_tests}, 通过: {report.passed}, 失败: {report.failed}")
                print(f"成功率: {report.success_rate*100:.1f}%")
                
                if report.failed > 0 or report.errors > 0:
                    print("失败的测试:")
                    for result in report.results:
                        if result.status.value in ['failed', 'error']:
                            print(f"  - {result.name}: {result.message or result.error}")
            
            except Exception as e:
                print(f"错误: 运行套件{suite.name}失败: {e}")
                # 创建错误报告
                error_report = TestReport(
                    suite_name=suite.name,
                    test_type=suite.test_type,
                    start_time=time.time(),
                    end_time=time.time(),
                    results=[],
                    summary={"error": str(e)}
                )
                reports.append(error_report)
        
        return reports
    
    def run_all_tests(self, test_types: List[str] = None, categories: List[str] = None) -> Dict[str, List[TestReport]]:
        """运行所有测试"""
        print("\n" + "="*60)
        print("           YOLOS 完整测试套件")
        print("="*60)
        
        # 确定要运行的类别
        if categories:
            categories_to_run = [cat for cat in categories if cat in self.all_suites]
        else:
            categories_to_run = list(self.all_suites.keys())
        
        all_reports = {}
        total_start_time = time.time()
        
        for category in categories_to_run:
            reports = self.run_test_category(category, test_types)
            all_reports[category] = reports
        
        total_duration = time.time() - total_start_time
        
        # 生成总体统计
        self._print_overall_summary(all_reports, total_duration)
        
        return all_reports
    
    def _print_overall_summary(self, all_reports: Dict[str, List[TestReport]], duration: float):
        """打印总体测试摘要"""
        print("\n" + "="*60)
        print("                测试总结")
        print("="*60)
        
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_errors = 0
        
        for category, reports in all_reports.items():
            category_tests = sum(r.total_tests for r in reports)
            category_passed = sum(r.passed for r in reports)
            category_failed = sum(r.failed for r in reports)
            category_errors = sum(r.errors for r in reports)
            
            total_tests += category_tests
            total_passed += category_passed
            total_failed += category_failed
            total_errors += category_errors
            
            if category_tests > 0:
                success_rate = category_passed / category_tests * 100
                print(f"{category.upper():12} | 总计: {category_tests:3d} | 通过: {category_passed:3d} | 失败: {category_failed:3d} | 错误: {category_errors:3d} | 成功率: {success_rate:5.1f}%")
        
        print("-" * 60)
        overall_success_rate = total_passed / total_tests * 100 if total_tests > 0 else 0
        print(f"{'总计':12} | 总计: {total_tests:3d} | 通过: {total_passed:3d} | 失败: {total_failed:3d} | 错误: {total_errors:3d} | 成功率: {overall_success_rate:5.1f}%")
        print(f"\n总耗时: {duration:.2f}秒")
        
        # 测试质量评估
        if overall_success_rate >= 95:
            quality = "优秀 ✅"
        elif overall_success_rate >= 85:
            quality = "良好 ✅"
        elif overall_success_rate >= 70:
            quality = "一般 ⚠️"
        else:
            quality = "需要改进 ❌"
        
        print(f"测试质量: {quality}")
    
    def generate_reports(self, all_reports: Dict[str, List[TestReport]], format_type: str = "html"):
        """生成测试报告"""
        print(f"\n生成{format_type.upper()}测试报告...")
        
        # 合并所有报告
        all_report_list = []
        for reports in all_reports.values():
            all_report_list.extend(reports)
        
        if format_type.lower() == "html":
            report_file = self.output_dir / "comprehensive_test_report.html"
            self.runner.generate_html_report(all_report_list, str(report_file))
            print(f"HTML报告已生成: {report_file}")
        
        # 生成分类报告
        for category, reports in all_reports.items():
            if reports:
                category_file = self.output_dir / f"{category}_test_report.html"
                self.runner.generate_html_report(reports, str(category_file))
                print(f"{category.upper()}报告已生成: {category_file}")
    
    def run_quick_test(self) -> bool:
        """运行快速测试（仅单元测试）"""
        print("\n运行快速测试（仅单元测试）...")
        
        reports = self.run_test_category('core', ['unit'])
        
        if not reports:
            print("警告: 没有可运行的快速测试")
            return True
        
        # 检查是否有失败
        total_failed = sum(r.failed + r.errors for r in reports)
        return total_failed == 0
    
    def run_performance_test(self) -> Dict[str, Any]:
        """运行性能测试"""
        print("\n运行性能测试...")
        
        performance_results = {}
        
        for category in ['core', 'applications', 'platform']:
            reports = self.run_test_category(category, ['performance'])
            
            category_results = []
            for report in reports:
                for result in report.results:
                    if hasattr(result, 'performance_data') and result.performance_data:
                        category_results.append({
                            'test_name': result.name,
                            'duration': result.duration,
                            'performance_data': result.performance_data
                        })
            
            if category_results:
                performance_results[category] = category_results
        
        return performance_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="YOLOS 完整测试套件")
    parser.add_argument('--categories', '-c', nargs='+', 
                       choices=['core', 'applications', 'platform', 'all'],
                       default=['all'], help="要运行的测试类别")
    parser.add_argument('--types', '-t', nargs='+',
                       choices=['unit', 'integration', 'performance', 'all'],
                       default=['all'], help="要运行的测试类型")
    parser.add_argument('--quick', '-q', action='store_true',
                       help="运行快速测试（仅核心单元测试）")
    parser.add_argument('--performance', '-p', action='store_true',
                       help="仅运行性能测试")
    parser.add_argument('--output', '-o', default="test_reports",
                       help="测试报告输出目录")
    parser.add_argument('--no-report', action='store_true',
                       help="不生成HTML报告")
    
    args = parser.parse_args()
    
    # 创建测试运行器
    test_runner = ComprehensiveTestRunner(args.output)
    
    try:
        if args.quick:
            # 快速测试
            success = test_runner.run_quick_test()
            return 0 if success else 1
        
        elif args.performance:
            # 性能测试
            results = test_runner.run_performance_test()
            
            print("\n=== 性能测试结果 ===")
            for category, tests in results.items():
                print(f"\n{category.upper()}:")
                for test in tests:
                    print(f"  {test['test_name']}: {test['duration']:.3f}s")
                    if 'throughput' in test['performance_data']:
                        print(f"    吞吐量: {test['performance_data']['throughput']:.2f}")
            
            return 0
        
        else:
            # 完整测试
            categories = None if 'all' in args.categories else args.categories
            test_types = None if 'all' in args.types else args.types
            
            all_reports = test_runner.run_all_tests(test_types, categories)
            
            # 生成报告
            if not args.no_report:
                test_runner.generate_reports(all_reports)
            
            # 计算退出码
            total_failed = 0
            for reports in all_reports.values():
                total_failed += sum(r.failed + r.errors for r in reports)
            
            return 0 if total_failed == 0 else 1
    
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        return 130
    
    except Exception as e:
        print(f"\n测试运行出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())