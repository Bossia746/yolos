#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合测试报告生成器

汇总所有测试结果，生成综合测试报告和改进建议
包括：
1. 基础功能测试结果
2. 核心功能复杂路径测试结果
3. 硬件平台兼容性测试结果
4. 性能压力和边界条件测试结果
5. 部署和集成场景测试结果
6. 综合分析和改进建议
"""

import json
import os
import sys
import glob
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestSummary:
    """测试摘要"""
    test_type: str
    total_tests: int
    successful_tests: int
    failed_tests: int
    success_rate: float
    avg_performance_score: float
    key_metrics: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class ComprehensiveTestReportGenerator:
    """综合测试报告生成器"""
    
    def __init__(self, test_results_dir: str = "."):
        self.test_results_dir = test_results_dir
        self.test_summaries: List[TestSummary] = []
        self.logger = logging.getLogger(__name__)
    
    def find_test_reports(self) -> Dict[str, str]:
        """查找测试报告文件"""
        report_files = {}
        
        # 查找各类测试报告
        patterns = {
            'basic_function': '*basic_function_test_report*.json',
            'core_complex': '*core_function_complex_test_report*.json',
            'hardware_compatibility': '*hardware_compatibility_report*.json',
            'performance_stress': '*performance_stress_boundary_report*.json',
            'deployment_integration': '*deployment_integration_report*.json'
        }
        
        for test_type, pattern in patterns.items():
            files = glob.glob(os.path.join(self.test_results_dir, pattern))
            if files:
                # 选择最新的报告文件
                latest_file = max(files, key=os.path.getctime)
                report_files[test_type] = latest_file
                self.logger.info(f"找到{test_type}测试报告: {latest_file}")
            else:
                self.logger.warning(f"未找到{test_type}测试报告")
        
        return report_files
    
    def load_test_report(self, file_path: str) -> Optional[Dict[str, Any]]:
        """加载测试报告"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"加载测试报告失败 {file_path}: {e}")
            return None
    
    def analyze_basic_function_report(self, report: Dict[str, Any]) -> TestSummary:
        """分析基础功能测试报告"""
        test_summary = report.get('test_summary', {})
        
        total_tests = test_summary.get('total_tests', 0)
        successful_tests = test_summary.get('successful_tests', 0)
        failed_tests = test_summary.get('failed_tests', 0)
        success_rate = test_summary.get('success_rate', 0)
        
        # 提取关键指标
        key_metrics = {
            'avg_execution_time': test_summary.get('avg_execution_time', 0),
            'total_execution_time': test_summary.get('total_execution_time', 0)
        }
        
        # 提取问题和建议
        issues = []
        recommendations = []
        
        if failed_tests > 0:
            issues.append(f"有{failed_tests}个基础功能测试失败")
            recommendations.append("修复失败的基础功能测试")
        
        if success_rate < 90:
            issues.append(f"基础功能测试成功率较低: {success_rate:.1f}%")
            recommendations.append("提高基础功能的稳定性和可靠性")
        
        return TestSummary(
            test_type="基础功能测试",
            total_tests=total_tests,
            successful_tests=successful_tests,
            failed_tests=failed_tests,
            success_rate=success_rate,
            avg_performance_score=success_rate,
            key_metrics=key_metrics,
            issues=issues,
            recommendations=recommendations
        )
    
    def analyze_core_complex_report(self, report: Dict[str, Any]) -> TestSummary:
        """分析核心功能复杂路径测试报告"""
        test_summary = report.get('test_summary', {})
        
        total_tests = test_summary.get('total_tests', 0)
        successful_tests = test_summary.get('successful_tests', 0)
        failed_tests = test_summary.get('failed_tests', 0)
        success_rate = test_summary.get('success_rate', 0)
        avg_performance_score = test_summary.get('avg_performance_score', 0)
        
        # 提取关键指标
        key_metrics = {
            'avg_execution_time': test_summary.get('avg_execution_time', 0),
            'avg_memory_usage': test_summary.get('avg_memory_usage', 0),
            'avg_cpu_usage': test_summary.get('avg_cpu_usage', 0)
        }
        
        # 提取问题和建议
        issues = []
        recommendations = []
        
        if failed_tests > 0:
            issues.append(f"有{failed_tests}个复杂路径测试失败")
            recommendations.append("优化复杂场景下的算法处理能力")
        
        if avg_performance_score < 80:
            issues.append(f"复杂路径测试性能评分较低: {avg_performance_score:.1f}")
            recommendations.append("优化复杂场景的性能表现")
        
        return TestSummary(
            test_type="核心功能复杂路径测试",
            total_tests=total_tests,
            successful_tests=successful_tests,
            failed_tests=failed_tests,
            success_rate=success_rate,
            avg_performance_score=avg_performance_score,
            key_metrics=key_metrics,
            issues=issues,
            recommendations=recommendations
        )
    
    def analyze_hardware_compatibility_report(self, report: Dict[str, Any]) -> TestSummary:
        """分析硬件兼容性测试报告"""
        test_summary = report.get('test_summary', {})
        
        total_tests = test_summary.get('total_tests', 0)
        successful_tests = test_summary.get('successful_tests', 0)
        failed_tests = test_summary.get('failed_tests', 0)
        success_rate = test_summary.get('success_rate', 0)
        avg_compatibility_score = test_summary.get('avg_compatibility_score', 0)
        
        # 提取关键指标
        key_metrics = {
            'compatibility_rating': report.get('compatibility_rating', ''),
            'avg_compatibility_score': avg_compatibility_score,
            'hardware_platform': report.get('hardware_info', {}).get('platform', '')
        }
        
        # 提取问题和建议
        issues = []
        recommendations = []
        
        if failed_tests > 0:
            issues.append(f"有{failed_tests}个硬件兼容性测试失败")
            recommendations.append("改进硬件平台适配性")
        
        if avg_compatibility_score < 85:
            issues.append(f"硬件兼容性评分较低: {avg_compatibility_score:.1f}")
            recommendations.append("优化硬件资源利用效率")
        
        return TestSummary(
            test_type="硬件平台兼容性测试",
            total_tests=total_tests,
            successful_tests=successful_tests,
            failed_tests=failed_tests,
            success_rate=success_rate,
            avg_performance_score=avg_compatibility_score,
            key_metrics=key_metrics,
            issues=issues,
            recommendations=recommendations
        )
    
    def analyze_performance_stress_report(self, report: Dict[str, Any]) -> TestSummary:
        """分析性能压力测试报告"""
        test_summary = report.get('test_summary', {})
        
        total_tests = test_summary.get('total_tests', 0)
        successful_tests = test_summary.get('successful_tests', 0)
        failed_tests = test_summary.get('failed_tests', 0)
        success_rate = test_summary.get('success_rate', 0)
        avg_performance_score = test_summary.get('avg_performance_score', 0)
        
        # 提取关键指标
        key_metrics = {
            'performance_rating': report.get('performance_rating', ''),
            'avg_performance_score': avg_performance_score,
            'peak_memory_usage': test_summary.get('peak_memory_usage', 0),
            'peak_cpu_usage': test_summary.get('peak_cpu_usage', 0)
        }
        
        # 提取问题和建议
        issues = []
        recommendations = []
        
        if failed_tests > 0:
            issues.append(f"有{failed_tests}个性能压力测试失败")
            recommendations.append("提高系统在高负载下的稳定性")
        
        if avg_performance_score < 80:
            issues.append(f"性能压力测试评分较低: {avg_performance_score:.1f}")
            recommendations.append("优化系统性能和资源管理")
        
        return TestSummary(
            test_type="性能压力和边界条件测试",
            total_tests=total_tests,
            successful_tests=successful_tests,
            failed_tests=failed_tests,
            success_rate=success_rate,
            avg_performance_score=avg_performance_score,
            key_metrics=key_metrics,
            issues=issues,
            recommendations=recommendations
        )
    
    def analyze_deployment_integration_report(self, report: Dict[str, Any]) -> TestSummary:
        """分析部署集成测试报告"""
        test_summary = report.get('test_summary', {})
        
        total_tests = test_summary.get('total_tests', 0)
        successful_tests = test_summary.get('successful_tests', 0)
        failed_tests = test_summary.get('failed_tests', 0)
        success_rate = test_summary.get('success_rate', 0)
        avg_availability = test_summary.get('avg_availability', 0)
        
        # 提取关键指标
        key_metrics = {
            'integration_rating': report.get('integration_rating', ''),
            'avg_response_time': test_summary.get('avg_response_time', 0),
            'avg_throughput': test_summary.get('avg_throughput', 0),
            'avg_availability': avg_availability
        }
        
        # 提取问题和建议
        issues = []
        recommendations = []
        
        if failed_tests > 0:
            issues.append(f"有{failed_tests}个部署集成测试失败")
            recommendations.append("修复部署和集成问题")
        
        if success_rate < 80:
            issues.append(f"部署集成测试成功率较低: {success_rate:.1f}%")
            recommendations.append("改进系统集成和部署流程")
        
        return TestSummary(
            test_type="部署和集成场景测试",
            total_tests=total_tests,
            successful_tests=successful_tests,
            failed_tests=failed_tests,
            success_rate=success_rate,
            avg_performance_score=avg_availability,
            key_metrics=key_metrics,
            issues=issues,
            recommendations=recommendations
        )
    
    def generate_comprehensive_analysis(self) -> Dict[str, Any]:
        """生成综合分析"""
        if not self.test_summaries:
            return {'error': '没有测试摘要数据'}
        
        # 总体统计
        total_tests = sum(s.total_tests for s in self.test_summaries)
        total_successful = sum(s.successful_tests for s in self.test_summaries)
        total_failed = sum(s.failed_tests for s in self.test_summaries)
        overall_success_rate = (total_successful / total_tests * 100) if total_tests > 0 else 0
        
        # 平均性能评分
        avg_performance_score = sum(s.avg_performance_score for s in self.test_summaries) / len(self.test_summaries)
        
        # 按测试类型统计
        test_type_stats = {}
        for summary in self.test_summaries:
            test_type_stats[summary.test_type] = {
                'total_tests': summary.total_tests,
                'successful_tests': summary.successful_tests,
                'failed_tests': summary.failed_tests,
                'success_rate': summary.success_rate,
                'performance_score': summary.avg_performance_score
            }
        
        # 汇总所有问题和建议
        all_issues = []
        all_recommendations = []
        
        for summary in self.test_summaries:
            all_issues.extend(summary.issues)
            all_recommendations.extend(summary.recommendations)
        
        # 去重建议
        unique_recommendations = list(set(all_recommendations))
        
        # 生成系统质量评级
        quality_rating = self._get_system_quality_rating(overall_success_rate, avg_performance_score)
        
        # 生成改进优先级
        improvement_priorities = self._generate_improvement_priorities()
        
        return {
            'overall_statistics': {
                'total_tests': total_tests,
                'successful_tests': total_successful,
                'failed_tests': total_failed,
                'overall_success_rate': overall_success_rate,
                'avg_performance_score': avg_performance_score,
                'system_quality_rating': quality_rating
            },
            'test_type_breakdown': test_type_stats,
            'identified_issues': all_issues,
            'improvement_recommendations': unique_recommendations,
            'improvement_priorities': improvement_priorities,
            'deployment_readiness': self._assess_deployment_readiness(overall_success_rate, avg_performance_score)
        }
    
    def _get_system_quality_rating(self, success_rate: float, performance_score: float) -> str:
        """获取系统质量评级"""
        combined_score = (success_rate + performance_score) / 2
        
        if combined_score >= 95:
            return "优秀 (Excellent) - 系统质量卓越，可以投入生产"
        elif combined_score >= 85:
            return "良好 (Good) - 系统质量良好，可以部署使用"
        elif combined_score >= 70:
            return "一般 (Fair) - 系统质量一般，需要优化改进"
        elif combined_score >= 50:
            return "较差 (Poor) - 系统质量较差，需要重大改进"
        else:
            return "不合格 (Unacceptable) - 系统质量不合格，不可投入使用"
    
    def _generate_improvement_priorities(self) -> List[Dict[str, Any]]:
        """生成改进优先级"""
        priorities = []
        
        # 基于测试结果确定优先级
        for summary in self.test_summaries:
            if summary.failed_tests > 0:
                priority_level = "高" if summary.success_rate < 70 else "中"
                priorities.append({
                    'area': summary.test_type,
                    'priority': priority_level,
                    'reason': f"成功率{summary.success_rate:.1f}%，有{summary.failed_tests}个失败测试",
                    'actions': summary.recommendations[:3]  # 取前3个建议
                })
        
        # 按优先级排序
        priority_order = {'高': 3, '中': 2, '低': 1}
        priorities.sort(key=lambda x: priority_order.get(x['priority'], 0), reverse=True)
        
        return priorities
    
    def _assess_deployment_readiness(self, success_rate: float, performance_score: float) -> Dict[str, Any]:
        """评估部署就绪性"""
        readiness_score = (success_rate + performance_score) / 2
        
        if readiness_score >= 90:
            status = "就绪 (Ready)"
            recommendation = "系统已准备好部署到生产环境"
            risk_level = "低"
        elif readiness_score >= 75:
            status = "基本就绪 (Mostly Ready)"
            recommendation = "系统基本就绪，建议修复关键问题后部署"
            risk_level = "中"
        elif readiness_score >= 60:
            status = "需要改进 (Needs Improvement)"
            recommendation = "系统需要重要改进才能部署"
            risk_level = "高"
        else:
            status = "未就绪 (Not Ready)"
            recommendation = "系统未准备好部署，需要重大改进"
            risk_level = "很高"
        
        return {
            'status': status,
            'readiness_score': readiness_score,
            'recommendation': recommendation,
            'risk_level': risk_level,
            'required_actions': self._get_deployment_actions(readiness_score)
        }
    
    def _get_deployment_actions(self, readiness_score: float) -> List[str]:
        """获取部署行动建议"""
        actions = []
        
        if readiness_score < 90:
            actions.append("修复所有失败的测试用例")
        
        if readiness_score < 80:
            actions.append("优化系统性能和稳定性")
            actions.append("加强错误处理和异常恢复")
        
        if readiness_score < 70:
            actions.append("重新设计关键组件")
            actions.append("增加更多测试覆盖")
        
        if readiness_score < 60:
            actions.append("进行架构重构")
            actions.append("重新评估技术选型")
        
        # 通用建议
        actions.extend([
            "建立持续集成/持续部署流水线",
            "实施监控和告警系统",
            "准备回滚计划",
            "进行生产环境预演"
        ])
        
        return actions[:8]  # 限制建议数量
    
    def generate_report(self) -> Dict[str, Any]:
        """生成综合测试报告"""
        self.logger.info("开始生成综合测试报告")
        
        # 查找测试报告文件
        report_files = self.find_test_reports()
        
        if not report_files:
            return {'error': '未找到任何测试报告文件'}
        
        # 分析各类测试报告
        for test_type, file_path in report_files.items():
            report_data = self.load_test_report(file_path)
            if report_data:
                try:
                    if test_type == 'basic_function':
                        summary = self.analyze_basic_function_report(report_data)
                    elif test_type == 'core_complex':
                        summary = self.analyze_core_complex_report(report_data)
                    elif test_type == 'hardware_compatibility':
                        summary = self.analyze_hardware_compatibility_report(report_data)
                    elif test_type == 'performance_stress':
                        summary = self.analyze_performance_stress_report(report_data)
                    elif test_type == 'deployment_integration':
                        summary = self.analyze_deployment_integration_report(report_data)
                    else:
                        continue
                    
                    self.test_summaries.append(summary)
                    self.logger.info(f"已分析{test_type}测试报告")
                
                except Exception as e:
                    self.logger.error(f"分析{test_type}测试报告失败: {e}")
        
        if not self.test_summaries:
            return {'error': '没有成功分析的测试报告'}
        
        # 生成综合分析
        comprehensive_analysis = self.generate_comprehensive_analysis()
        
        # 构建最终报告
        final_report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'generator_version': '1.0.0',
                'analyzed_reports': len(self.test_summaries),
                'report_files': report_files
            },
            'test_summaries': [
                {
                    'test_type': s.test_type,
                    'total_tests': s.total_tests,
                    'successful_tests': s.successful_tests,
                    'failed_tests': s.failed_tests,
                    'success_rate': s.success_rate,
                    'avg_performance_score': s.avg_performance_score,
                    'key_metrics': s.key_metrics,
                    'issues': s.issues,
                    'recommendations': s.recommendations
                } for s in self.test_summaries
            ],
            'comprehensive_analysis': comprehensive_analysis
        }
        
        return final_report
    
    def save_report(self, filename: str = None) -> str:
        """保存综合测试报告"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'comprehensive_test_report_{timestamp}.json'
        
        report = self.generate_report()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        return filename
    
    def print_summary(self):
        """打印综合测试摘要"""
        report = self.generate_report()
        
        if 'error' in report:
            print(f"错误: {report['error']}")
            return
        
        print("\n" + "="*100)
        print("YOLOS系统综合测试报告")
        print("="*100)
        
        # 总体统计
        overall_stats = report['comprehensive_analysis']['overall_statistics']
        print(f"\n🎯 总体测试统计:")
        print(f"   总测试数: {overall_stats['total_tests']}")
        print(f"   成功测试: {overall_stats['successful_tests']}")
        print(f"   失败测试: {overall_stats['failed_tests']}")
        print(f"   总体成功率: {overall_stats['overall_success_rate']:.1f}%")
        print(f"   平均性能评分: {overall_stats['avg_performance_score']:.1f}/100")
        print(f"   系统质量评级: {overall_stats['system_quality_rating']}")
        
        # 各测试类型详情
        print(f"\n📊 各测试类型详情:")
        for summary in report['test_summaries']:
            print(f"   {summary['test_type']}:")
            print(f"     - 测试数: {summary['total_tests']} (成功: {summary['successful_tests']}, 失败: {summary['failed_tests']})")
            print(f"     - 成功率: {summary['success_rate']:.1f}%")
            print(f"     - 性能评分: {summary['avg_performance_score']:.1f}/100")
        
        # 部署就绪性评估
        deployment = report['comprehensive_analysis']['deployment_readiness']
        print(f"\n🚀 部署就绪性评估:")
        print(f"   状态: {deployment['status']}")
        print(f"   就绪评分: {deployment['readiness_score']:.1f}/100")
        print(f"   风险等级: {deployment['risk_level']}")
        print(f"   建议: {deployment['recommendation']}")
        
        # 改进优先级
        priorities = report['comprehensive_analysis']['improvement_priorities']
        if priorities:
            print(f"\n⚡ 改进优先级 (前5项):")
            for i, priority in enumerate(priorities[:5], 1):
                print(f"   {i}. [{priority['priority']}] {priority['area']}")
                print(f"      原因: {priority['reason']}")
        
        # 关键问题
        issues = report['comprehensive_analysis']['identified_issues']
        if issues:
            print(f"\n⚠️  识别的关键问题 (前5项):")
            for i, issue in enumerate(issues[:5], 1):
                print(f"   {i}. {issue}")
        
        # 改进建议
        recommendations = report['comprehensive_analysis']['improvement_recommendations']
        if recommendations:
            print(f"\n💡 改进建议 (前8项):")
            for i, rec in enumerate(recommendations[:8], 1):
                print(f"   {i}. {rec}")
        
        print("\n" + "="*100)

def main() -> int:
    """主函数"""
    try:
        print("开始生成YOLOS系统综合测试报告...")
        
        # 创建报告生成器
        generator = ComprehensiveTestReportGenerator()
        
        # 生成并打印摘要
        generator.print_summary()
        
        # 保存详细报告
        report_file = generator.save_report()
        print(f"\n📄 综合测试报告已保存到: {report_file}")
        
        print("\n✅ 综合测试报告生成完成")
        return 0
    
    except KeyboardInterrupt:
        print("\n报告生成被用户中断")
        return 1
    except Exception as e:
        print(f"\n报告生成失败: {e}")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)