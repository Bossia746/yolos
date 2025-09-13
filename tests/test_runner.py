#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试运行器

统一的测试执行和管理工具，支持：
- 单元测试执行
- 集成测试执行
- 性能基准测试执行
- 测试报告生成
- 测试结果分析
"""

import os
import sys
import time
import json
import argparse
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import logging

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from .test_config import YOLOSTestConfig
from .benchmark_tests import BenchmarkTestSuite
from .enhanced_integration_tests import EnhancedIntegrationTestSuite
from .base_test import BaseTest


@dataclass
class YOLOSTestResult:
    """测试结果"""
    test_name: str
    test_type: str  # 'unit', 'integration', 'benchmark', 'performance'
    success: bool
    duration: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error_info: Optional[Dict[str, Any]] = None


@dataclass
class YOLOSTestSuiteResult:
    """测试套件结果"""
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    total_duration: float
    success_rate: float
    test_results: List[YOLOSTestResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class YOLOSTestRunner:
    """测试运行器"""
    
    def __init__(self, config: Optional[YOLOSTestConfig] = None):
        self.config = config or YOLOSTestConfig()
        self.logger = self._setup_logger()
        self.results: List[YOLOSTestSuiteResult] = []
        self.start_time = None
        self.end_time = None
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('TestRunner')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def run_unit_tests(self, test_pattern: str = "test_*.py", parallel: bool = True) -> YOLOSTestSuiteResult:
        """运行单元测试"""
        self.logger.info("开始运行单元测试")
        start_time = time.time()
        
        try:
            # 查找单元测试文件
            test_dir = Path(__file__).parent
            test_files = list(test_dir.glob(test_pattern))
            
            # 过滤掉非单元测试文件
            unit_test_files = [
                f for f in test_files 
                if not any(exclude in f.name for exclude in [
                    'integration', 'benchmark', 'performance', 'test_runner', 
                    'test_config', 'base_test', 'mock_data'
                ])
            ]
            
            test_results = []
            
            if parallel and len(unit_test_files) > 1:
                # 并行执行
                with ThreadPoolExecutor(max_workers=self.config.environment.parallel_workers or 4) as executor:
                    futures = {
                        executor.submit(self._run_single_test_file, test_file): test_file
                        for test_file in unit_test_files
                    }
                    
                    for future in as_completed(futures):
                        test_file = futures[future]
                        try:
                            result = future.result()
                            test_results.extend(result)
                        except Exception as e:
                            self.logger.error(f"运行测试文件 {test_file} 失败: {e}")
                            test_results.append(YOLOSTestResult(
                                test_name=test_file.name,
                                test_type='unit',
                                success=False,
                                duration=0.0,
                                message=f"测试执行异常: {e}",
                                error_info={'exception': str(e)}
                            ))
            else:
                # 串行执行
                for test_file in unit_test_files:
                    try:
                        result = self._run_single_test_file(test_file)
                        test_results.extend(result)
                    except Exception as e:
                        self.logger.error(f"运行测试文件 {test_file} 失败: {e}")
                        test_results.append(YOLOSTestResult(
                            test_name=test_file.name,
                            test_type='unit',
                            success=False,
                            duration=0.0,
                            message=f"测试执行异常: {e}",
                            error_info={'exception': str(e)}
                        ))
            
            # 统计结果
            total_tests = len(test_results)
            passed_tests = sum(1 for r in test_results if r.success)
            failed_tests = total_tests - passed_tests
            total_duration = time.time() - start_time
            success_rate = passed_tests / total_tests if total_tests > 0 else 0
            
            suite_result = YOLOSTestSuiteResult(
                suite_name='unit_tests',
                total_tests=total_tests,
                passed_tests=passed_tests,
                failed_tests=failed_tests,
                skipped_tests=0,
                total_duration=total_duration,
                success_rate=success_rate,
                test_results=test_results
            )
            
            self.results.append(suite_result)
            self.logger.info(f"单元测试完成: {passed_tests}/{total_tests} 通过")
            return suite_result
            
        except Exception as e:
            self.logger.error(f"单元测试执行异常: {e}")
            suite_result = YOLOSTestSuiteResult(
                suite_name='unit_tests',
                total_tests=0,
                passed_tests=0,
                failed_tests=1,
                skipped_tests=0,
                total_duration=time.time() - start_time,
                success_rate=0.0,
                test_results=[YOLOSTestResult(
                    test_name='unit_test_suite',
                    test_type='unit',
                    success=False,
                    duration=0.0,
                    message=f"单元测试套件执行异常: {e}",
                    error_info={'exception': str(e)}
                )]
            )
            self.results.append(suite_result)
            return suite_result
    
    def _run_single_test_file(self, test_file: Path) -> List[YOLOSTestResult]:
        """运行单个测试文件"""
        results = []
        
        try:
            # 使用pytest运行测试文件
            cmd = [sys.executable, '-m', 'pytest', str(test_file), '-v', '--tb=short']
            
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.environment.timeout or 60.0
            )
            duration = time.time() - start_time
            
            # 解析pytest输出
            success = result.returncode == 0
            
            results.append(YOLOSTestResult(
                test_name=test_file.name,
                test_type='unit',
                success=success,
                duration=duration,
                message='测试通过' if success else '测试失败',
                details={
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'return_code': result.returncode
                }
            ))
            
        except subprocess.TimeoutExpired:
            results.append(YOLOSTestResult(
                test_name=test_file.name,
                test_type='unit',
                success=False,
                duration=self.config.environment.timeout or 60.0,
                message='测试超时',
                error_info={'error_type': 'timeout'}
            ))
        except Exception as e:
            results.append(YOLOSTestResult(
                test_name=test_file.name,
                test_type='unit',
                success=False,
                duration=0.0,
                message=f'测试执行异常: {e}',
                error_info={'exception': str(e)}
            ))
        
        return results
    
    def run_integration_tests(self) -> YOLOSTestSuiteResult:
        """运行集成测试"""
        self.logger.info("开始运行集成测试")
        start_time = time.time()
        
        try:
            # 创建集成测试套件
            integration_suite = EnhancedIntegrationTestSuite()
            
            # 运行所有集成测试
            integration_results = integration_suite.run_all_tests()
            
            # 转换结果格式
            test_results = []
            for result in integration_results:
                test_results.append(YOLOSTestResult(
                    test_name=result.test_name,
                    test_type='integration',
                    success=result.success,
                    duration=result.duration,
                    message=result.message,
                    details=result.details
                ))
            
            # 清理资源
            integration_suite.cleanup()
            
            # 统计结果
            total_tests = len(test_results)
            passed_tests = sum(1 for r in test_results if r.success)
            failed_tests = total_tests - passed_tests
            total_duration = time.time() - start_time
            success_rate = passed_tests / total_tests if total_tests > 0 else 0
            
            suite_result = YOLOSTestSuiteResult(
                suite_name='integration_tests',
                total_tests=total_tests,
                passed_tests=passed_tests,
                failed_tests=failed_tests,
                skipped_tests=0,
                total_duration=total_duration,
                success_rate=success_rate,
                test_results=test_results
            )
            
            self.results.append(suite_result)
            self.logger.info(f"集成测试完成: {passed_tests}/{total_tests} 通过")
            return suite_result
            
        except Exception as e:
            self.logger.error(f"集成测试执行异常: {e}")
            suite_result = YOLOSTestSuiteResult(
                suite_name='integration_tests',
                total_tests=0,
                passed_tests=0,
                failed_tests=1,
                skipped_tests=0,
                total_duration=time.time() - start_time,
                success_rate=0.0,
                test_results=[YOLOSTestResult(
                    test_name='integration_test_suite',
                    test_type='integration',
                    success=False,
                    duration=0.0,
                    message=f"集成测试套件执行异常: {e}",
                    error_info={'exception': str(e)}
                )]
            )
            self.results.append(suite_result)
            return suite_result
    
    def run_benchmark_tests(self) -> YOLOSTestSuiteResult:
        """运行基准测试"""
        self.logger.info("开始运行基准测试")
        start_time = time.time()
        
        try:
            # 创建基准测试套件
            benchmark_suite = BenchmarkTestSuite()
            
            # 运行所有基准测试
            benchmark_results = benchmark_suite.run_all_benchmarks()
            
            # 转换结果格式
            test_results = []
            for result in benchmark_results:
                test_results.append(YOLOSTestResult(
                    test_name=result.test_name,
                    test_type='benchmark',
                    success=result.success,
                    duration=result.duration,
                    message=result.message,
                    details=result.details
                ))
            
            # 清理资源
            benchmark_suite.cleanup()
            
            # 统计结果
            total_tests = len(test_results)
            passed_tests = sum(1 for r in test_results if r.success)
            failed_tests = total_tests - passed_tests
            total_duration = time.time() - start_time
            success_rate = passed_tests / total_tests if total_tests > 0 else 0
            
            suite_result = YOLOSTestSuiteResult(
                suite_name='benchmark_tests',
                total_tests=total_tests,
                passed_tests=passed_tests,
                failed_tests=failed_tests,
                skipped_tests=0,
                total_duration=total_duration,
                success_rate=success_rate,
                test_results=test_results
            )
            
            self.results.append(suite_result)
            self.logger.info(f"基准测试完成: {passed_tests}/{total_tests} 通过")
            return suite_result
            
        except Exception as e:
            self.logger.error(f"基准测试执行异常: {e}")
            suite_result = YOLOSTestSuiteResult(
                suite_name='benchmark_tests',
                total_tests=0,
                passed_tests=0,
                failed_tests=1,
                skipped_tests=0,
                total_duration=time.time() - start_time,
                success_rate=0.0,
                test_results=[YOLOSTestResult(
                    test_name='benchmark_test_suite',
                    test_type='benchmark',
                    success=False,
                    duration=0.0,
                    message=f"基准测试套件执行异常: {e}",
                    error_info={'exception': str(e)}
                )]
            )
            self.results.append(suite_result)
            return suite_result
    
    def run_all_tests(self, include_benchmarks: bool = True) -> List[YOLOSTestSuiteResult]:
        """运行所有测试"""
        self.logger.info("开始运行完整测试套件")
        self.start_time = time.time()
        
        try:
            # 运行单元测试
            self.run_unit_tests()
            
            # 运行集成测试
            self.run_integration_tests()
            
            # 运行基准测试（可选）
            if include_benchmarks:
                self.run_benchmark_tests()
            
            self.end_time = time.time()
            self.logger.info(f"所有测试完成，总耗时: {self.end_time - self.start_time:.2f}秒")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"测试套件执行异常: {e}")
            self.end_time = time.time()
            return self.results
    
    def generate_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """生成测试报告"""
        if not self.results:
            self.logger.warning("没有测试结果可生成报告")
            return {}
        
        # 计算总体统计
        total_tests = sum(suite.total_tests for suite in self.results)
        total_passed = sum(suite.passed_tests for suite in self.results)
        total_failed = sum(suite.failed_tests for suite in self.results)
        total_skipped = sum(suite.skipped_tests for suite in self.results)
        total_duration = sum(suite.total_duration for suite in self.results)
        overall_success_rate = total_passed / total_tests if total_tests > 0 else 0
        
        # 生成报告数据
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': total_tests,
                'passed_tests': total_passed,
                'failed_tests': total_failed,
                'skipped_tests': total_skipped,
                'success_rate': overall_success_rate,
                'total_duration': total_duration,
                'start_time': self.start_time,
                'end_time': self.end_time
            },
            'test_suites': [
                {
                    'suite_name': suite.suite_name,
                    'total_tests': suite.total_tests,
                    'passed_tests': suite.passed_tests,
                    'failed_tests': suite.failed_tests,
                    'skipped_tests': suite.skipped_tests,
                    'success_rate': suite.success_rate,
                    'duration': suite.total_duration,
                    'timestamp': suite.timestamp
                }
                for suite in self.results
            ],
            'detailed_results': [
                {
                    'suite_name': suite.suite_name,
                    'tests': [
                        {
                            'test_name': test.test_name,
                            'test_type': test.test_type,
                            'success': test.success,
                            'duration': test.duration,
                            'message': test.message,
                            'timestamp': test.timestamp,
                            'error_info': test.error_info
                        }
                        for test in suite.test_results
                    ]
                }
                for suite in self.results
            ]
        }
        
        # 保存报告到文件
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                self.logger.info(f"测试报告已保存到: {output_file}")
            except Exception as e:
                self.logger.error(f"保存测试报告失败: {e}")
        
        return report
    
    def print_summary(self):
        """打印测试摘要"""
        if not self.results:
            print("没有测试结果")
            return
        
        print("\n" + "="*60)
        print("测试结果摘要")
        print("="*60)
        
        for suite in self.results:
            status = "✓" if suite.success_rate == 1.0 else "✗" if suite.success_rate == 0.0 else "⚠"
            print(f"{status} {suite.suite_name}:")
            print(f"  通过: {suite.passed_tests}/{suite.total_tests} ({suite.success_rate:.1%})")
            print(f"  耗时: {suite.total_duration:.2f}秒")
            
            # 显示失败的测试
            failed_tests = [test for test in suite.test_results if not test.success]
            if failed_tests:
                print(f"  失败测试:")
                for test in failed_tests[:3]:  # 只显示前3个失败测试
                    print(f"    - {test.test_name}: {test.message}")
                if len(failed_tests) > 3:
                    print(f"    ... 还有 {len(failed_tests) - 3} 个失败测试")
            print()
        
        # 总体统计
        total_tests = sum(suite.total_tests for suite in self.results)
        total_passed = sum(suite.passed_tests for suite in self.results)
        total_duration = sum(suite.total_duration for suite in self.results)
        overall_success_rate = total_passed / total_tests if total_tests > 0 else 0
        
        print(f"总体结果: {total_passed}/{total_tests} 通过 ({overall_success_rate:.1%})")
        print(f"总耗时: {total_duration:.2f}秒")
        print("="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOS 测试运行器')
    parser.add_argument('--type', choices=['unit', 'integration', 'benchmark', 'all'], 
                       default='all', help='测试类型')
    parser.add_argument('--pattern', default='test_*.py', help='单元测试文件模式')
    parser.add_argument('--parallel', action='store_true', help='并行执行测试')
    parser.add_argument('--no-benchmarks', action='store_true', help='跳过基准测试')
    parser.add_argument('--output', help='测试报告输出文件')
    parser.add_argument('--config', help='测试配置文件')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建测试配置
    config = YOLOSTestConfig()
    if args.config:
        # 这里可以加载自定义配置文件
        pass
    
    # 创建测试运行器
    runner = YOLOSTestRunner(config)
    
    try:
        # 根据参数运行相应的测试
        if args.type == 'unit':
            runner.run_unit_tests(args.pattern, args.parallel)
        elif args.type == 'integration':
            runner.run_integration_tests()
        elif args.type == 'benchmark':
            runner.run_benchmark_tests()
        elif args.type == 'all':
            runner.run_all_tests(include_benchmarks=not args.no_benchmarks)
        
        # 打印摘要
        runner.print_summary()
        
        # 生成报告
        output_file = args.output or f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        runner.generate_report(output_file)
        
        # 返回适当的退出码
        total_failed = sum(suite.failed_tests for suite in runner.results)
        sys.exit(0 if total_failed == 0 else 1)
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"测试运行器异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()