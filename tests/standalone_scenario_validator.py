#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立场景验证器 - 重构版本

使用拆分后的验证器模块来验证各种应用场景的合理性和可行性
"""

import sys
import os
import logging
from typing import Dict, List, Any

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入拆分后的验证器模块
from tests.validators.medical_validator import MedicalScenarioValidator
from tests.validators.safety_validator import SafetyScenarioValidator
from tests.validators.lifestyle_validator import LifestyleScenarioValidator
from tests.validators.technical_validator import TechnicalScenarioValidator


class StandaloneScenarioValidator:
    """独立场景验证器 - 主控制器"""
    
    def __init__(self):
        """初始化验证器"""
        self.setup_logging()
        
        # 初始化各个验证器
        self.medical_validator = MedicalScenarioValidator(self.logger)
        self.safety_validator = SafetyScenarioValidator(self.logger)
        self.lifestyle_validator = LifestyleScenarioValidator(self.logger)
        self.technical_validator = TechnicalScenarioValidator(self.logger)
        
        self.all_results = []
    
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('scenario_validation.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def validate_all_scenarios(self) -> Dict:
        """验证所有场景"""
        self.logger.info("开始验证所有场景")
        
        # 医疗健康场景验证
        self.logger.info("=== 医疗健康场景验证 ===")
        medical_results = []
        medical_results.extend(self.medical_validator.validate_facial_symptom_scenarios())
        medical_results.extend(self.medical_validator.validate_vital_sign_scenarios())
        medical_results.extend(self.medical_validator.validate_medication_scenarios())
        medical_results.extend(self.medical_validator.validate_rehabilitation_scenarios())
        self.all_results.extend(medical_results)
        
        # 安全防护场景验证
        self.logger.info("=== 安全防护场景验证 ===")
        safety_results = []
        safety_results.extend(self.safety_validator.validate_fire_detection_scenarios())
        safety_results.extend(self.safety_validator.validate_intrusion_detection_scenarios())
        safety_results.extend(self.safety_validator.validate_fall_detection_scenarios())
        safety_results.extend(self.safety_validator.validate_emergency_response_scenarios())
        self.all_results.extend(safety_results)
        
        # 生活场景验证
        self.logger.info("=== 生活场景验证 ===")
        lifestyle_results = []
        lifestyle_results.extend(self.lifestyle_validator.validate_smart_home_scenarios())
        lifestyle_results.extend(self.lifestyle_validator.validate_elderly_care_scenarios())
        lifestyle_results.extend(self.lifestyle_validator.validate_child_safety_scenarios())
        lifestyle_results.extend(self.lifestyle_validator.validate_pet_care_scenarios())
        self.all_results.extend(lifestyle_results)
        
        # 技术场景验证
        self.logger.info("=== 技术场景验证 ===")
        technical_results = []
        technical_results.extend(self.technical_validator.validate_performance_scenarios())
        technical_results.extend(self.technical_validator.validate_scalability_scenarios())
        technical_results.extend(self.technical_validator.validate_compatibility_scenarios())
        technical_results.extend(self.technical_validator.validate_deployment_scenarios())
        self.all_results.extend(technical_results)
        
        # 生成综合报告
        return self.generate_comprehensive_report()
    
    def generate_comprehensive_report(self) -> Dict:
        """生成综合验证报告"""
        self.logger.info("生成综合验证报告")
        
        # 统计各类验证器的结果
        medical_summary = self.medical_validator.get_validation_summary()
        safety_summary = self.safety_validator.get_validation_summary()
        lifestyle_summary = self.lifestyle_validator.get_validation_summary()
        technical_summary = self.technical_validator.get_validation_summary()
        
        # 计算总体统计
        total_scenarios = len(self.all_results)
        critical_failures = sum(r["critical_failures"] for r in self.all_results)
        warnings = sum(r["warnings"] for r in self.all_results)
        passed = sum(1 for r in self.all_results if r["overall_status"] == "PASS")
        
        # 按类别统计
        category_stats = {}
        for result in self.all_results:
            category = result["category"]
            if category not in category_stats:
                category_stats[category] = {"total": 0, "passed": 0, "failed": 0, "warnings": 0}
            
            category_stats[category]["total"] += 1
            if result["overall_status"] == "PASS":
                category_stats[category]["passed"] += 1
            elif result["overall_status"] == "FAIL":
                category_stats[category]["failed"] += 1
            else:
                category_stats[category]["warnings"] += 1
        
        # 识别关键问题
        critical_issues = []
        for result in self.all_results:
            if result["critical_failures"] > 0:
                critical_issues.append({
                    "category": result["category"],
                    "scenario": result["name"],
                    "failures": result["critical_failures"]
                })
        
        # 生成建议
        recommendations = self._generate_overall_recommendations()
        
        report = {
            "validation_timestamp": self._get_timestamp(),
            "overall_summary": {
                "total_scenarios": total_scenarios,
                "passed": passed,
                "critical_failures": critical_failures,
                "warnings": warnings,
                "pass_rate": passed / total_scenarios if total_scenarios > 0 else 0,
                "overall_status": "PASS" if critical_failures == 0 else "FAIL"
            },
            "category_summaries": {
                "medical": medical_summary,
                "safety": safety_summary,
                "lifestyle": lifestyle_summary,
                "technical": technical_summary
            },
            "category_statistics": category_stats,
            "critical_issues": critical_issues,
            "detailed_results": self.all_results,
            "recommendations": recommendations
        }
        
        return report
    
    def _generate_overall_recommendations(self) -> List[str]:
        """生成总体建议"""
        recommendations = [
            "建议进行实际场景测试验证理论分析结果",
            "建立持续的性能监控和质量保证机制",
            "定期更新验证标准以适应技术发展",
            "建立用户反馈收集和处理机制",
            "考虑建立A/B测试框架验证改进效果"
        ]
        
        # 根据验证结果添加特定建议
        critical_failures = sum(r["critical_failures"] for r in self.all_results)
        if critical_failures > 0:
            recommendations.insert(0, "优先解决关键失败项，这些问题可能影响系统核心功能")
        
        warnings = sum(r["warnings"] for r in self.all_results)
        if warnings > 10:
            recommendations.append("关注警告项较多的情况，建议制定改进计划")
        
        return recommendations
    
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def save_report(self, report: Dict, filename: str = "scenario_validation_report.json"):
        """保存验证报告"""
        import json
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            self.logger.info(f"验证报告已保存到: {filename}")
        except Exception as e:
            self.logger.error(f"保存报告失败: {e}")
    
    def print_summary(self, report: Dict):
        """打印验证摘要"""
        print("\n" + "="*80)
        print("场景验证报告摘要")
        print("="*80)
        
        summary = report["overall_summary"]
        print(f"验证时间: {report['validation_timestamp']}")
        print(f"总场景数: {summary['total_scenarios']}")
        print(f"通过数: {summary['passed']}")
        print(f"关键失败: {summary['critical_failures']}")
        print(f"警告数: {summary['warnings']}")
        print(f"通过率: {summary['pass_rate']:.1%}")
        print(f"总体状态: {summary['overall_status']}")
        
        print("\n各类别统计:")
        for category, stats in report["category_statistics"].items():
            print(f"  {category}: {stats['passed']}/{stats['total']} 通过")
        
        if report["critical_issues"]:
            print("\n关键问题:")
            for issue in report["critical_issues"]:
                print(f"  - {issue['category']}/{issue['scenario']}: {issue['failures']}个关键失败")
        
        print("\n主要建议:")
        for i, rec in enumerate(report["recommendations"][:5], 1):
            print(f"  {i}. {rec}")
        
        print("="*80)


def main():
    """主函数"""
    print("YOLOS 场景验证器")
    print("正在验证各种应用场景的合理性和可行性...")
    
    validator = StandaloneScenarioValidator()
    
    try:
        # 执行验证
        report = validator.validate_all_scenarios()
        
        # 保存报告
        validator.save_report(report)
        
        # 打印摘要
        validator.print_summary(report)
        
        print("\n验证完成！详细报告已保存到 scenario_validation_report.json")
        
    except Exception as e:
        print(f"验证过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()