#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医疗场景验证器

专门用于验证医疗健康相关场景的合理性和可行性
"""

import logging
from typing import Dict, List, Any


class MedicalScenarioValidator:
    """医疗场景验证器"""
    
    def __init__(self, logger: logging.Logger = None):
        """初始化医疗场景验证器"""
        self.logger = logger or logging.getLogger(__name__)
        self.validation_results = []
        
        # 医疗标准知识库
        self.medical_standards = {
            "accuracy_requirements": {
                "medical": 0.99,  # 99%
                "safety_critical": 0.98,  # 98%
                "general": 0.90  # 90%
            },
            "response_time_limits": {
                "emergency": 30,  # seconds
                "routine": 300,  # seconds
                "monitoring": 60  # seconds
            },
            "regulatory_requirements": {
                "medical_device_class": "Class I",
                "fda_approval": "Not required for wellness devices",
                "disclaimer_required": True,
                "data_privacy": "HIPAA compliance recommended"
            }
        }
    
    def validate_medical_scenarios(self) -> List[Dict]:
        """验证医疗健康场景"""
        self.logger.info("验证医疗健康场景")
        
        scenarios = [
            {
                "name": "面部症状检测",
                "description": "通过面部表情检测健康状态",
                "validations": [
                    {
                        "aspect": "检测准确性",
                        "requirement": "医疗辅助检测准确率应≥99%",
                        "current_claim": "95%准确率",
                        "assessment": self._assess_medical_accuracy,
                        "critical": True
                    },
                    {
                        "aspect": "响应时间",
                        "requirement": "紧急症状检测应在30秒内响应",
                        "current_claim": "实时检测",
                        "assessment": self._assess_medical_response_time,
                        "critical": True
                    },
                    {
                        "aspect": "医疗免责",
                        "requirement": "必须明确标注为辅助工具，非诊断设备",
                        "current_claim": "辅助检测系统",
                        "assessment": self._assess_medical_disclaimer,
                        "critical": True
                    }
                ]
            },
            {
                "name": "生命体征监测",
                "description": "通过摄像头监测基本生命体征",
                "validations": [
                    {
                        "aspect": "技术可行性",
                        "requirement": "摄像头监测心率准确率应≥90%",
                        "current_claim": "支持心率监测",
                        "assessment": self._assess_vital_signs_feasibility,
                        "critical": True
                    },
                    {
                        "aspect": "环境限制",
                        "requirement": "需要良好光照和稳定环境",
                        "current_claim": "适应各种环境",
                        "assessment": self._assess_environmental_requirements,
                        "critical": False
                    }
                ]
            },
            {
                "name": "药物识别",
                "description": "识别药物信息和安全检查",
                "validations": [
                    {
                        "aspect": "识别准确性",
                        "requirement": "药物识别准确率必须≥99.5%",
                        "current_claim": "高精度OCR识别",
                        "assessment": self._assess_drug_recognition_accuracy,
                        "critical": True
                    },
                    {
                        "aspect": "安全检查",
                        "requirement": "必须包含相互作用和禁忌检查",
                        "current_claim": "完整安全检查",
                        "assessment": self._assess_drug_safety_checks,
                        "critical": True
                    }
                ]
            }
        ]
        
        return self._validate_scenario_group("医疗健康", scenarios)
    
    def _validate_scenario_group(self, category: str, scenarios: List[Dict]) -> List[Dict]:
        """验证场景组"""
        results = []
        
        for scenario in scenarios:
            result = self._validate_single_scenario(category, scenario)
            results.append(result)
            self.validation_results.append(result)
        
        return results
    
    def _validate_single_scenario(self, category: str, scenario: Dict) -> Dict:
        """验证单个场景"""
        scenario_result = {
            "category": category,
            "name": scenario["name"],
            "description": scenario["description"],
            "validations": [],
            "overall_status": "PASS",
            "critical_failures": 0,
            "warnings": 0
        }
        
        for validation in scenario["validations"]:
            assessment_result = validation["assessment"](validation)
            
            validation_result = {
                "aspect": validation["aspect"],
                "requirement": validation["requirement"],
                "current_claim": validation["current_claim"],
                "critical": validation["critical"],
                "status": assessment_result["status"],
                "score": assessment_result["score"],
                "issues": assessment_result["issues"],
                "recommendations": assessment_result["recommendations"]
            }
            
            scenario_result["validations"].append(validation_result)
            
            if assessment_result["status"] == "FAIL" and validation["critical"]:
                scenario_result["critical_failures"] += 1
                scenario_result["overall_status"] = "FAIL"
            elif assessment_result["status"] == "WARNING":
                scenario_result["warnings"] += 1
                if scenario_result["overall_status"] != "FAIL":
                    scenario_result["overall_status"] = "WARNING"
        
        return scenario_result
    
    def _assess_medical_accuracy(self, validation: Dict) -> Dict:
        """评估医疗检测准确性"""
        # 解析当前声称的准确率
        claim = validation["current_claim"]
        
        if "95%" in claim:
            current_accuracy = 0.95
        else:
            current_accuracy = 0.90  # 保守估计
        
        required_accuracy = self.medical_standards["accuracy_requirements"]["medical"]
        
        if current_accuracy >= required_accuracy:
            return {
                "status": "PASS",
                "score": 100,
                "issues": [],
                "recommendations": ["继续保持高准确率标准"]
            }
        else:
            return {
                "status": "FAIL",
                "score": int(current_accuracy / required_accuracy * 100),
                "issues": [
                    f"当前准确率{current_accuracy*100}%低于医疗标准要求{required_accuracy*100}%",
                    "医疗辅助设备需要极高的准确率以避免误诊风险"
                ],
                "recommendations": [
                    "提升算法准确率至99%以上",
                    "增加更多医疗数据集进行训练",
                    "考虑多模型集成提升准确率"
                ]
            }
    
    def _assess_medical_response_time(self, validation: Dict) -> Dict:
        """评估医疗响应时间"""
        claim = validation["current_claim"]
        required_time = self.medical_standards["response_time_limits"]["emergency"]
        
        if "实时" in claim:
            # 实时检测通常意味着<1秒响应
            return {
                "status": "PASS",
                "score": 100,
                "issues": [],
                "recommendations": ["确保在各种硬件条件下都能保持实时响应"]
            }
        else:
            return {
                "status": "WARNING",
                "score": 70,
                "issues": ["响应时间描述不够具体"],
                "recommendations": [
                    f"明确响应时间指标，确保紧急情况下{required_time}秒内响应",
                    "建立响应时间监控机制"
                ]
            }
    
    def _assess_medical_disclaimer(self, validation: Dict) -> Dict:
        """评估医疗免责声明"""
        claim = validation["current_claim"]
        
        if "辅助" in claim and "检测" in claim:
            return {
                "status": "PASS",
                "score": 90,
                "issues": [],
                "recommendations": [
                    "确保在所有用户界面都显示医疗免责声明",
                    "明确说明这不是医疗诊断设备",
                    "建议用户咨询专业医生"
                ]
            }
        else:
            return {
                "status": "FAIL",
                "score": 30,
                "issues": [
                    "缺乏明确的医疗免责声明",
                    "可能存在法律风险"
                ],
                "recommendations": [
                    "添加明确的医疗免责声明",
                    "咨询法律专家确保合规性",
                    "在产品文档中明确设备定位"
                ]
            }
    
    def _assess_vital_signs_feasibility(self, validation: Dict) -> Dict:
        """评估生命体征监测可行性"""
        return {
            "status": "WARNING",
            "score": 60,
            "issues": [
                "摄像头监测心率技术仍在发展中",
                "准确率受环境因素影响较大",
                "需要特定的硬件和算法支持"
            ],
            "recommendations": [
                "进行充分的技术验证和临床测试",
                "明确技术限制和使用条件",
                "考虑与专业医疗设备厂商合作",
                "建立数据验证机制"
            ]
        }
    
    def _assess_environmental_requirements(self, validation: Dict) -> Dict:
        """评估环境要求"""
        claim = validation["current_claim"]
        
        if "适应各种环境" in claim:
            return {
                "status": "WARNING",
                "score": 50,
                "issues": [
                    "声称过于绝对，实际上摄像头监测对环境有严格要求",
                    "光照条件、距离、角度都会影响准确性"
                ],
                "recommendations": [
                    "明确具体的环境要求（光照、距离、角度等）",
                    "提供环境优化建议",
                    "建立环境质量检测机制"
                ]
            }
        else:
            return {
                "status": "PASS",
                "score": 85,
                "issues": [],
                "recommendations": ["继续完善环境适应性"]
            }
    
    def _assess_drug_recognition_accuracy(self, validation: Dict) -> Dict:
        """评估药物识别准确性"""
        claim = validation["current_claim"]
        required_accuracy = 0.995  # 99.5%
        
        if "高精度" in claim:
            # 需要具体数据验证
            return {
                "status": "WARNING",
                "score": 70,
                "issues": [
                    "缺乏具体的准确率数据",
                    "药物识别错误可能导致严重后果"
                ],
                "recommendations": [
                    f"提供具体的准确率数据，确保≥{required_accuracy*100}%",
                    "建立药物数据库和验证机制",
                    "增加人工确认环节",
                    "建立错误报告和改进机制"
                ]
            }
        else:
            return {
                "status": "FAIL",
                "score": 40,
                "issues": ["药物识别准确性要求极高，当前描述不足"],
                "recommendations": [
                    "进行大规模药物识别测试",
                    "建立完整的药物数据库",
                    "实施多重验证机制"
                ]
            }
    
    def _assess_drug_safety_checks(self, validation: Dict) -> Dict:
        """评估药物安全检查"""
        claim = validation["current_claim"]
        
        if "完整安全检查" in claim:
            return {
                "status": "WARNING",
                "score": 75,
                "issues": ["需要明确安全检查的具体内容和范围"],
                "recommendations": [
                    "详细说明安全检查包含的项目",
                    "建立药物相互作用数据库",
                    "包含过敏史和禁忌症检查",
                    "提供专业医生咨询建议",
                    "建立紧急情况处理流程"
                ]
            }
        else:
            return {
                "status": "FAIL",
                "score": 30,
                "issues": ["缺乏药物安全检查机制"],
                "recommendations": [
                    "建立完整的药物安全检查系统",
                    "集成药物相互作用数据库",
                    "添加用户健康档案管理"
                ]
            }
    
    def get_validation_summary(self) -> Dict:
        """获取验证摘要"""
        if not self.validation_results:
            return {"status": "No validations performed"}
        
        total_scenarios = len(self.validation_results)
        critical_failures = sum(r["critical_failures"] for r in self.validation_results)
        warnings = sum(r["warnings"] for r in self.validation_results)
        passed = sum(1 for r in self.validation_results if r["overall_status"] == "PASS")
        
        return {
            "total_scenarios": total_scenarios,
            "passed": passed,
            "critical_failures": critical_failures,
            "warnings": warnings,
            "pass_rate": passed / total_scenarios if total_scenarios > 0 else 0,
            "overall_status": "PASS" if critical_failures == 0 else "FAIL"
        }