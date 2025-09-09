#!/usr/bin/env python3
"""
医疗安全管理器
处理所有医疗相关功能的安全性和合规性
"""

import logging
import warnings
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class MedicalRiskLevel(Enum):
    """医疗风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MedicalConfidenceLevel(Enum):
    """医疗置信度等级"""
    VERY_LOW = 0.0
    LOW = 0.7
    MEDIUM = 0.85
    HIGH = 0.95
    VERY_HIGH = 0.99

@dataclass
class MedicalResult:
    """医疗检测结果"""
    result: str
    confidence: float
    risk_level: MedicalRiskLevel
    disclaimer: str
    recommendations: List[str]
    timestamp: datetime
    requires_professional_consultation: bool

class MedicalSafetyManager:
    """医疗安全管理器"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.min_confidence_threshold = MedicalConfidenceLevel.HIGH.value  # 95%
        self.critical_confidence_threshold = MedicalConfidenceLevel.VERY_HIGH.value  # 99%
        
        # 医疗免责声明
        self.medical_disclaimer = self._get_medical_disclaimer()
        
        # 初始化安全检查
        self._initialize_safety_checks()
    
    def _setup_logger(self):
        """设置日志"""
        logger = logging.getLogger('MedicalSafetyManager')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler('logs/medical_safety.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _get_medical_disclaimer(self) -> str:
        """获取医疗免责声明"""
        return """
⚠️ 重要医疗免责声明 ⚠️

本系统仅为辅助参考工具，不能替代专业医疗诊断。

• 本系统不是医疗设备，未获得医疗器械认证
• 所有检测结果仅供参考，不构成医疗建议
• 任何健康问题请咨询专业医生或医疗机构
• 紧急情况请立即拨打急救电话
• 用户使用本系统的风险由用户自行承担

如有任何健康疑虑，请寻求专业医疗帮助。
        """.strip()
    
    def _initialize_safety_checks(self):
        """初始化安全检查"""
        self.safety_checks = {
            "confidence_check": self._check_confidence_level,
            "risk_assessment": self._assess_medical_risk,
            "disclaimer_required": self._check_disclaimer_requirement,
            "professional_consultation": self._check_professional_consultation_need
        }
    
    def validate_medical_detection(self, 
                                 detection_type: str,
                                 raw_result: Dict,
                                 confidence: float) -> MedicalResult:
        """验证医疗检测结果"""
        
        self.logger.info(f"验证医疗检测: {detection_type}, 置信度: {confidence:.3f}")
        
        # 1. 置信度检查
        if not self._check_confidence_level(confidence):
            return self._create_low_confidence_result(detection_type, confidence)
        
        # 2. 风险评估
        risk_level = self._assess_medical_risk(detection_type, raw_result)
        
        # 3. 生成安全的医疗结果
        safe_result = self._create_safe_medical_result(
            detection_type, raw_result, confidence, risk_level
        )
        
        # 4. 记录医疗检测
        self._log_medical_detection(safe_result)
        
        return safe_result
    
    def _check_confidence_level(self, confidence: float) -> bool:
        """检查置信度是否达到安全标准"""
        return confidence >= self.min_confidence_threshold
    
    def _assess_medical_risk(self, detection_type: str, raw_result: Dict) -> MedicalRiskLevel:
        """评估医疗风险等级"""
        
        # 根据检测类型评估风险
        high_risk_types = [
            "cardiac_symptoms", "stroke_symptoms", "breathing_difficulty",
            "severe_pain", "consciousness_loss", "severe_bleeding"
        ]
        
        medium_risk_types = [
            "fever", "fatigue", "mild_pain", "skin_changes",
            "sleep_disturbance", "appetite_changes"
        ]
        
        if detection_type in high_risk_types:
            return MedicalRiskLevel.HIGH
        elif detection_type in medium_risk_types:
            return MedicalRiskLevel.MEDIUM
        else:
            return MedicalRiskLevel.LOW
    
    def _check_disclaimer_requirement(self, risk_level: MedicalRiskLevel) -> bool:
        """检查是否需要免责声明"""
        # 所有医疗检测都需要免责声明
        return True
    
    def _check_professional_consultation_need(self, 
                                            risk_level: MedicalRiskLevel,
                                            confidence: float) -> bool:
        """检查是否需要专业咨询"""
        
        # 高风险或低置信度都需要专业咨询
        return (risk_level in [MedicalRiskLevel.HIGH, MedicalRiskLevel.CRITICAL] or 
                confidence < self.critical_confidence_threshold)
    
    def _create_low_confidence_result(self, 
                                    detection_type: str, 
                                    confidence: float) -> MedicalResult:
        """创建低置信度结果"""
        
        return MedicalResult(
            result=f"检测置信度过低({confidence:.1%})，无法提供可靠结果",
            confidence=confidence,
            risk_level=MedicalRiskLevel.MEDIUM,
            disclaimer=self.medical_disclaimer,
            recommendations=[
                "建议在更好的光照条件下重新检测",
                "如有健康疑虑，请咨询专业医生",
                "不要仅依赖此检测结果做出医疗决定"
            ],
            timestamp=datetime.now(),
            requires_professional_consultation=True
        )
    
    def _create_safe_medical_result(self,
                                  detection_type: str,
                                  raw_result: Dict,
                                  confidence: float,
                                  risk_level: MedicalRiskLevel) -> MedicalResult:
        """创建安全的医疗结果"""
        
        # 生成安全的结果描述
        safe_result = self._generate_safe_result_description(detection_type, raw_result, risk_level)
        
        # 生成建议
        recommendations = self._generate_medical_recommendations(detection_type, risk_level)
        
        # 检查是否需要专业咨询
        needs_consultation = self._check_professional_consultation_need(risk_level, confidence)
        
        return MedicalResult(
            result=safe_result,
            confidence=confidence,
            risk_level=risk_level,
            disclaimer=self.medical_disclaimer,
            recommendations=recommendations,
            timestamp=datetime.now(),
            requires_professional_consultation=needs_consultation
        )
    
    def _generate_safe_result_description(self,
                                        detection_type: str,
                                        raw_result: Dict,
                                        risk_level: MedicalRiskLevel) -> str:
        """生成安全的结果描述"""
        
        # 基础安全前缀
        safety_prefix = "⚠️ 辅助检测结果（仅供参考）: "
        
        # 根据风险等级调整描述
        if risk_level == MedicalRiskLevel.HIGH:
            return f"{safety_prefix}检测到可能的异常症状，强烈建议立即咨询医生"
        elif risk_level == MedicalRiskLevel.MEDIUM:
            return f"{safety_prefix}检测到一些症状表现，建议关注并考虑医疗咨询"
        else:
            return f"{safety_prefix}未检测到明显异常，但请注意这不能替代专业医疗检查"
    
    def _generate_medical_recommendations(self,
                                        detection_type: str,
                                        risk_level: MedicalRiskLevel) -> List[str]:
        """生成医疗建议"""
        
        base_recommendations = [
            "此结果仅供参考，不能替代专业医疗诊断",
            "如有任何健康疑虑，请咨询专业医生",
            "定期进行专业健康检查"
        ]
        
        if risk_level == MedicalRiskLevel.HIGH:
            base_recommendations.extend([
                "🚨 建议立即寻求专业医疗帮助",
                "如有紧急情况，请拨打急救电话",
                "不要延误医疗治疗"
            ])
        elif risk_level == MedicalRiskLevel.MEDIUM:
            base_recommendations.extend([
                "建议在适当时候咨询医生",
                "注意观察症状变化",
                "保持健康的生活方式"
            ])
        
        return base_recommendations
    
    def _log_medical_detection(self, result: MedicalResult):
        """记录医疗检测"""
        
        self.logger.info(f"医疗检测完成 - 置信度: {result.confidence:.3f}, "
                        f"风险等级: {result.risk_level.value}, "
                        f"需要专业咨询: {result.requires_professional_consultation}")
        
        # 高风险情况特别记录
        if result.risk_level in [MedicalRiskLevel.HIGH, MedicalRiskLevel.CRITICAL]:
            self.logger.warning(f"高风险医疗检测: {result.result}")
    
    def get_medical_disclaimer(self) -> str:
        """获取医疗免责声明"""
        return self.medical_disclaimer
    
    def is_medical_feature_safe(self, feature_name: str) -> Tuple[bool, str]:
        """检查医疗功能是否安全可用"""
        
        # 当前不安全的功能列表
        unsafe_features = [
            "drug_identification",  # 药物识别准确率不足
            "vital_signs_monitoring",  # 生命体征监测技术限制
            "disease_diagnosis"  # 疾病诊断功能
        ]
        
        if feature_name in unsafe_features:
            return False, f"功能 {feature_name} 当前不安全，已暂时禁用"
        
        return True, "功能可用，但请注意医疗免责声明"
    
    def create_safety_warning(self, message: str, risk_level: MedicalRiskLevel) -> str:
        """创建安全警告"""
        
        warning_icons = {
            MedicalRiskLevel.LOW: "ℹ️",
            MedicalRiskLevel.MEDIUM: "⚠️",
            MedicalRiskLevel.HIGH: "🚨",
            MedicalRiskLevel.CRITICAL: "🆘"
        }
        
        icon = warning_icons.get(risk_level, "⚠️")
        
        return f"{icon} 医疗安全提醒: {message}\n\n{self.medical_disclaimer}"

class DrugSafetyManager:
    """药物安全管理器"""
    
    def __init__(self):
        self.logger = logging.getLogger('DrugSafetyManager')
        self.min_accuracy_threshold = 0.995  # 99.5%
        
        # 药物安全检查项目
        self.safety_checks = [
            "drug_interaction_check",
            "allergy_check", 
            "dosage_safety_check",
            "age_appropriateness_check",
            "pregnancy_safety_check"
        ]
    
    def validate_drug_identification(self, 
                                   drug_name: str,
                                   confidence: float,
                                   user_profile: Dict) -> Dict:
        """验证药物识别结果"""
        
        # 1. 置信度检查
        if confidence < self.min_accuracy_threshold:
            return {
                "status": "unsafe",
                "message": f"药物识别置信度 {confidence:.1%} 低于安全标准 {self.min_accuracy_threshold:.1%}",
                "recommendation": "请人工确认药物信息，不要仅依赖自动识别结果",
                "allow_usage": False
            }
        
        # 2. 安全检查
        safety_results = self._perform_safety_checks(drug_name, user_profile)
        
        # 3. 生成安全报告
        return self._generate_safety_report(drug_name, confidence, safety_results)
    
    def _perform_safety_checks(self, drug_name: str, user_profile: Dict) -> Dict:
        """执行安全检查"""
        
        results = {}
        
        # 药物相互作用检查
        results["interaction_check"] = self._check_drug_interactions(drug_name, user_profile)
        
        # 过敏史检查
        results["allergy_check"] = self._check_allergies(drug_name, user_profile)
        
        # 剂量安全检查
        results["dosage_check"] = self._check_dosage_safety(drug_name, user_profile)
        
        # 年龄适宜性检查
        results["age_check"] = self._check_age_appropriateness(drug_name, user_profile)
        
        # 妊娠期安全检查
        results["pregnancy_check"] = self._check_pregnancy_safety(drug_name, user_profile)
        
        return results
    
    def _check_drug_interactions(self, drug_name: str, user_profile: Dict) -> Dict:
        """检查药物相互作用"""
        # 模拟检查逻辑
        current_medications = user_profile.get("current_medications", [])
        
        # 这里应该查询药物相互作用数据库
        interactions = []  # 模拟无相互作用
        
        return {
            "status": "safe" if not interactions else "warning",
            "interactions": interactions,
            "message": "未发现药物相互作用" if not interactions else f"发现 {len(interactions)} 个潜在相互作用"
        }
    
    def _check_allergies(self, drug_name: str, user_profile: Dict) -> Dict:
        """检查过敏史"""
        allergies = user_profile.get("allergies", [])
        
        # 检查是否对该药物过敏
        is_allergic = drug_name.lower() in [allergy.lower() for allergy in allergies]
        
        return {
            "status": "danger" if is_allergic else "safe",
            "message": f"用户对 {drug_name} 过敏" if is_allergic else "未发现过敏风险"
        }
    
    def _check_dosage_safety(self, drug_name: str, user_profile: Dict) -> Dict:
        """检查剂量安全"""
        # 模拟剂量检查
        return {
            "status": "safe",
            "message": "剂量在安全范围内"
        }
    
    def _check_age_appropriateness(self, drug_name: str, user_profile: Dict) -> Dict:
        """检查年龄适宜性"""
        age = user_profile.get("age", 0)
        
        # 检查儿童禁用药物
        pediatric_contraindicated = ["aspirin"]  # 示例
        
        if age < 16 and drug_name.lower() in pediatric_contraindicated:
            return {
                "status": "danger",
                "message": f"{drug_name} 不适合 {age} 岁儿童使用"
            }
        
        return {
            "status": "safe",
            "message": "年龄适宜性检查通过"
        }
    
    def _check_pregnancy_safety(self, drug_name: str, user_profile: Dict) -> Dict:
        """检查妊娠期安全"""
        is_pregnant = user_profile.get("is_pregnant", False)
        
        if not is_pregnant:
            return {
                "status": "safe",
                "message": "非妊娠期用药"
            }
        
        # 检查妊娠期安全等级
        pregnancy_categories = {
            "A": "safe",
            "B": "safe", 
            "C": "warning",
            "D": "danger",
            "X": "danger"
        }
        
        # 模拟查询药物妊娠期分类
        category = "B"  # 示例
        status = pregnancy_categories.get(category, "warning")
        
        return {
            "status": status,
            "category": category,
            "message": f"妊娠期安全等级: {category}"
        }
    
    def _generate_safety_report(self, 
                              drug_name: str,
                              confidence: float,
                              safety_results: Dict) -> Dict:
        """生成安全报告"""
        
        # 检查是否有危险项目
        has_danger = any(result.get("status") == "danger" for result in safety_results.values())
        has_warning = any(result.get("status") == "warning" for result in safety_results.values())
        
        if has_danger:
            overall_status = "danger"
            allow_usage = False
            message = f"⚠️ 危险: {drug_name} 存在安全风险，不建议使用"
        elif has_warning:
            overall_status = "warning"
            allow_usage = True
            message = f"⚠️ 警告: {drug_name} 存在潜在风险，请谨慎使用"
        else:
            overall_status = "safe"
            allow_usage = True
            message = f"✅ {drug_name} 安全检查通过"
        
        return {
            "status": overall_status,
            "message": message,
            "confidence": confidence,
            "allow_usage": allow_usage,
            "safety_checks": safety_results,
            "recommendations": self._generate_drug_recommendations(overall_status, safety_results)
        }
    
    def _generate_drug_recommendations(self, status: str, safety_results: Dict) -> List[str]:
        """生成药物使用建议"""
        
        recommendations = [
            "请仔细阅读药物说明书",
            "按照医生处方或说明书用药",
            "如有不适请立即停药并咨询医生"
        ]
        
        if status == "danger":
            recommendations.extend([
                "🚨 不要使用此药物",
                "立即咨询医生或药师",
                "寻找替代药物"
            ])
        elif status == "warning":
            recommendations.extend([
                "⚠️ 谨慎使用，密切观察反应",
                "建议咨询医生或药师",
                "注意药物相互作用"
            ])
        
        return recommendations

# 使用示例和测试
def test_medical_safety():
    """测试医疗安全管理器"""
    
    print("🧪 测试医疗安全管理器...")
    
    # 创建管理器
    medical_manager = MedicalSafetyManager()
    drug_manager = DrugSafetyManager()
    
    # 测试医疗检测验证
    print("\n1. 测试医疗检测验证:")
    
    # 低置信度测试
    low_confidence_result = medical_manager.validate_medical_detection(
        "fever_detection", {"temperature": "elevated"}, 0.85
    )
    print(f"低置信度结果: {low_confidence_result.result}")
    
    # 高置信度测试
    high_confidence_result = medical_manager.validate_medical_detection(
        "fever_detection", {"temperature": "elevated"}, 0.96
    )
    print(f"高置信度结果: {high_confidence_result.result}")
    
    # 测试药物安全检查
    print("\n2. 测试药物安全检查:")
    
    user_profile = {
        "age": 25,
        "allergies": ["penicillin"],
        "current_medications": ["ibuprofen"],
        "is_pregnant": False
    }
    
    # 安全药物测试
    safe_drug_result = drug_manager.validate_drug_identification(
        "paracetamol", 0.996, user_profile
    )
    print(f"安全药物结果: {safe_drug_result['message']}")
    
    # 过敏药物测试
    allergic_drug_result = drug_manager.validate_drug_identification(
        "penicillin", 0.998, user_profile
    )
    print(f"过敏药物结果: {allergic_drug_result['message']}")
    
    # 低置信度药物测试
    low_confidence_drug_result = drug_manager.validate_drug_identification(
        "unknown_drug", 0.92, user_profile
    )
    print(f"低置信度药物结果: {low_confidence_drug_result['message']}")
    
    print("\n✅ 医疗安全管理器测试完成")

if __name__ == "__main__":
    test_medical_safety()