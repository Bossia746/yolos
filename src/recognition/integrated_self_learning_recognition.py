#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成自学习识别系统
结合传统识别方法和大模型自学习能力的统一识别系统
"""

import cv2
import numpy as np
import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入现有模块
from .llm_self_learning_system import LLMSelfLearningSystem, LLMAnalysisResult, SceneCategory
from .hybrid_recognition_system import HybridRecognitionSystem
from .medical_facial_analyzer import MedicalFacialAnalyzer
from .enhanced_fall_detection_system import EnhancedFallDetectionSystem
from .medication_recognition_system import MedicationRecognitionSystem
from .anti_spoofing_detector import AntiSpoofingDetector
from .image_quality_enhancer import ImageQualityEnhancer

class RecognitionMode(Enum):
    """识别模式"""
    OFFLINE_ONLY = "offline_only"           # 仅离线识别
    HYBRID_AUTO = "hybrid_auto"             # 混合自动模式
    SELF_LEARNING = "self_learning"         # 自学习模式
    MANUAL_CONFIRM = "manual_confirm"       # 手动确认模式

class ConfidenceLevel(Enum):
    """置信度等级"""
    VERY_HIGH = "very_high"    # 0.9+
    HIGH = "high"              # 0.7-0.9
    MEDIUM = "medium"          # 0.5-0.7
    LOW = "low"                # 0.3-0.5
    VERY_LOW = "very_low"      # <0.3

@dataclass
class RecognitionResult:
    """识别结果"""
    # 基础信息
    object_type: str
    confidence: float
    confidence_level: ConfidenceLevel
    bounding_box: Optional[Tuple[int, int, int, int]]
    
    # 识别详情
    recognition_method: str
    processing_time: float
    timestamp: float
    
    # 多模态结果
    traditional_result: Optional[Dict[str, Any]]
    llm_result: Optional[LLMAnalysisResult]
    
    # 质量评估
    image_quality_score: float
    anti_spoofing_score: float
    
    # 医疗相关
    medical_analysis: Optional[Dict[str, Any]]
    emergency_level: str
    
    # 自学习相关
    requires_learning: bool
    learning_triggered: bool
    learning_success: Optional[bool]
    
    # 建议行动
    suggested_actions: List[str]
    confidence_factors: Dict[str, float]

class IntegratedSelfLearningRecognition:
    """集成自学习识别系统"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # 初始化子系统
        self._initialize_subsystems()
        
        # 识别统计
        self.stats = {
            'total_recognitions': 0,
            'offline_recognitions': 0,
            'hybrid_recognitions': 0,
            'llm_recognitions': 0,
            'learning_triggered': 0,
            'learning_successful': 0,
            'average_confidence': 0.0,
            'average_processing_time': 0.0
        }
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 缓存
        self.recognition_cache = {}
        self.cache_lock = threading.Lock()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            # 识别模式配置
            'recognition': {
                'default_mode': RecognitionMode.HYBRID_AUTO,
                'confidence_thresholds': {
                    'very_high': 0.9,
                    'high': 0.7,
                    'medium': 0.5,
                    'low': 0.3
                },
                'self_learning_threshold': 0.5,
                'enable_caching': True,
                'cache_ttl': 300,  # 缓存生存时间(秒)
                'parallel_processing': True
            },
            
            # 质量控制配置
            'quality_control': {
                'min_image_quality': 0.3,
                'min_anti_spoofing_score': 0.7,
                'enable_quality_enhancement': True,
                'enable_anti_spoofing': True
            },
            
            # 医疗识别配置
            'medical': {
                'enable_medical_analysis': True,
                'enable_fall_detection': True,
                'enable_medication_recognition': True,
                'emergency_confidence_threshold': 0.8
            },
            
            # 自学习配置
            'self_learning': {
                'enabled': True,
                'auto_trigger': True,
                'require_confirmation': False,
                'learning_batch_size': 10,
                'learning_interval': 3600  # 批量学习间隔(秒)
            },
            
            # 性能配置
            'performance': {
                'max_processing_time': 30.0,  # 最大处理时间(秒)
                'enable_gpu_acceleration': True,
                'memory_limit_mb': 2048,
                'concurrent_requests': 4
            }
        }
    
    def _initialize_subsystems(self):
        """初始化子系统"""
        try:
            # 图像质量增强器
            self.quality_enhancer = ImageQualityEnhancer(
                self.config.get('quality_enhancement', {})
            )
            
            # 反欺骗检测器
            self.anti_spoofing = AntiSpoofingDetector(
                self.config.get('anti_spoofing', {})
            )
            
            # 混合识别系统
            self.hybrid_recognition = HybridRecognitionSystem(
                self.config.get('hybrid_recognition', {})
            )
            
            # 医疗面部分析器
            self.medical_analyzer = MedicalFacialAnalyzer(
                self.config.get('medical_analysis', {})
            )
            
            # 跌倒检测系统
            self.fall_detector = EnhancedFallDetectionSystem(
                self.config.get('fall_detection', {})
            )
            
            # 药物识别系统
            self.medication_recognizer = MedicationRecognitionSystem(
                self.config.get('medication_recognition', {})
            )
            
            # 大模型自学习系统
            self.llm_learning = LLMSelfLearningSystem(
                self.config.get('llm_learning', {})
            )
            
            self.logger.info("所有子系统初始化完成")
            
        except Exception as e:
            self.logger.error(f"子系统初始化失败: {e}")
            raise
    
    def recognize(self, image: np.ndarray, 
                 context: Optional[Dict[str, Any]] = None,
                 mode: Optional[RecognitionMode] = None) -> RecognitionResult:
        """统一识别接口"""
        start_time = time.time()
        
        try:
            # 确定识别模式
            recognition_mode = mode or self.config['recognition']['default_mode']
            
            # 检查缓存
            if self.config['recognition']['enable_caching']:
                cached_result = self._check_cache(image)
                if cached_result:
                    return cached_result
            
            # 图像预处理和质量检查
            processed_image, quality_info = self._preprocess_and_check_quality(image)
            
            # 执行识别
            if recognition_mode == RecognitionMode.OFFLINE_ONLY:
                result = self._offline_recognition(processed_image, context, quality_info)
            elif recognition_mode == RecognitionMode.HYBRID_AUTO:
                result = self._hybrid_recognition(processed_image, context, quality_info)
            elif recognition_mode == RecognitionMode.SELF_LEARNING:
                result = self._self_learning_recognition(processed_image, context, quality_info)
            else:  # MANUAL_CONFIRM
                result = self._manual_confirm_recognition(processed_image, context, quality_info)
            
            # 后处理
            result.processing_time = time.time() - start_time
            result.timestamp = time.time()
            
            # 更新统计
            self._update_statistics(result)
            
            # 缓存结果
            if self.config['recognition']['enable_caching']:
                self._cache_result(image, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"识别过程失败: {e}")
            return self._create_error_result(str(e), time.time() - start_time)
    
    def _preprocess_and_check_quality(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """预处理和质量检查"""
        quality_info = {}
        
        # 图像质量评估
        if self.config['quality_control']['enable_quality_enhancement']:
            enhanced_image, quality_score = self.quality_enhancer.enhance_image(image)
            quality_info['quality_score'] = quality_score
            quality_info['enhanced'] = True
        else:
            enhanced_image = image
            quality_info['quality_score'] = 0.8  # 默认质量分数
            quality_info['enhanced'] = False
        
        # 反欺骗检测
        if self.config['quality_control']['enable_anti_spoofing']:
            spoofing_result = self.anti_spoofing.detect_spoofing(enhanced_image)
            quality_info['anti_spoofing_score'] = spoofing_result['confidence']
            quality_info['is_real'] = spoofing_result['is_real']
        else:
            quality_info['anti_spoofing_score'] = 1.0
            quality_info['is_real'] = True
        
        return enhanced_image, quality_info
    
    def _offline_recognition(self, image: np.ndarray, 
                           context: Optional[Dict[str, Any]],
                           quality_info: Dict[str, Any]) -> RecognitionResult:
        """离线识别"""
        # 使用混合识别系统的离线部分
        traditional_result = self.hybrid_recognition.recognize_offline(image, context)
        
        # 医疗分析
        medical_result = None
        if self.config['medical']['enable_medical_analysis']:
            medical_result = self.medical_analyzer.analyze_face(image)
        
        # 跌倒检测
        fall_result = None
        if self.config['medical']['enable_fall_detection']:
            fall_result = self.fall_detector.detect_fall(image)
        
        # 药物识别
        medication_result = None
        if self.config['medical']['enable_medication_recognition']:
            medication_result = self.medication_recognizer.recognize_medication(image)
        
        # 构建结果
        confidence = traditional_result.get('confidence', 0.0)
        
        return RecognitionResult(
            object_type=traditional_result.get('object_type', 'unknown'),
            confidence=confidence,
            confidence_level=self._get_confidence_level(confidence),
            bounding_box=traditional_result.get('bounding_box'),
            recognition_method='offline',
            processing_time=0.0,  # 将在外部设置
            timestamp=0.0,        # 将在外部设置
            traditional_result=traditional_result,
            llm_result=None,
            image_quality_score=quality_info['quality_score'],
            anti_spoofing_score=quality_info['anti_spoofing_score'],
            medical_analysis=medical_result,
            emergency_level=self._assess_emergency_level(medical_result, fall_result),
            requires_learning=confidence < self.config['recognition']['self_learning_threshold'],
            learning_triggered=False,
            learning_success=None,
            suggested_actions=self._generate_suggested_actions(traditional_result, medical_result, fall_result),
            confidence_factors=self._calculate_confidence_factors(traditional_result, quality_info)
        )
    
    def _hybrid_recognition(self, image: np.ndarray, 
                          context: Optional[Dict[str, Any]],
                          quality_info: Dict[str, Any]) -> RecognitionResult:
        """混合识别"""
        # 首先尝试离线识别
        offline_result = self._offline_recognition(image, context, quality_info)
        
        # 判断是否需要大模型辅助
        needs_llm = (
            offline_result.confidence < self.config['recognition']['self_learning_threshold'] or
            offline_result.object_type == 'unknown' or
            not quality_info['is_real']
        )
        
        if needs_llm and self.config['self_learning']['enabled']:
            # 调用大模型分析
            try:
                llm_result = self.llm_learning.analyze_unknown_scene(
                    image, context, offline_result.object_type
                )
                
                # 融合结果
                fused_result = self._fuse_recognition_results(offline_result, llm_result)
                fused_result.recognition_method = 'hybrid'
                fused_result.llm_result = llm_result
                
                # 触发学习
                if self.config['self_learning']['auto_trigger']:
                    learning_success = self.llm_learning.learn_from_analysis(
                        image, llm_result, offline_result.object_type
                    )
                    fused_result.learning_triggered = True
                    fused_result.learning_success = learning_success
                
                return fused_result
                
            except Exception as e:
                self.logger.warning(f"大模型分析失败，使用离线结果: {e}")
                offline_result.recognition_method = 'hybrid_fallback'
                return offline_result
        
        offline_result.recognition_method = 'hybrid_offline'
        return offline_result
    
    def _self_learning_recognition(self, image: np.ndarray, 
                                 context: Optional[Dict[str, Any]],
                                 quality_info: Dict[str, Any]) -> RecognitionResult:
        """自学习识别"""
        # 强制使用大模型分析
        try:
            llm_result = self.llm_learning.analyze_unknown_scene(image, context)
            
            # 也获取传统识别结果作为参考
            traditional_result = self.hybrid_recognition.recognize_offline(image, context)
            
            # 构建结果（以大模型结果为主）
            result = RecognitionResult(
                object_type=llm_result.detected_objects[0]['name'] if llm_result.detected_objects else 'unknown',
                confidence=llm_result.confidence,
                confidence_level=self._get_confidence_level(llm_result.confidence),
                bounding_box=None,  # 大模型通常不提供精确边界框
                recognition_method='self_learning',
                processing_time=0.0,
                timestamp=0.0,
                traditional_result=traditional_result,
                llm_result=llm_result,
                image_quality_score=quality_info['quality_score'],
                anti_spoofing_score=quality_info['anti_spoofing_score'],
                medical_analysis=llm_result.medical_relevance,
                emergency_level=self._assess_emergency_level_from_llm(llm_result),
                requires_learning=True,
                learning_triggered=True,
                learning_success=None,
                suggested_actions=llm_result.suggested_actions,
                confidence_factors={'llm_confidence': llm_result.confidence}
            )
            
            # 执行学习
            learning_success = self.llm_learning.learn_from_analysis(
                image, llm_result, traditional_result.get('object_type')
            )
            result.learning_success = learning_success
            
            return result
            
        except Exception as e:
            self.logger.error(f"自学习识别失败: {e}")
            # 回退到混合识别
            return self._hybrid_recognition(image, context, quality_info)
    
    def _manual_confirm_recognition(self, image: np.ndarray, 
                                  context: Optional[Dict[str, Any]],
                                  quality_info: Dict[str, Any]) -> RecognitionResult:
        """手动确认识别"""
        # 获取所有可能的识别结果
        offline_result = self._offline_recognition(image, context, quality_info)
        
        # 如果置信度足够高，直接返回
        if offline_result.confidence >= self.config['recognition']['confidence_thresholds']['high']:
            offline_result.recognition_method = 'manual_auto_approved'
            return offline_result
        
        # 否则标记为需要手动确认
        offline_result.recognition_method = 'manual_confirm_required'
        offline_result.suggested_actions.append("需要人工确认识别结果")
        
        return offline_result
    
    def _fuse_recognition_results(self, offline_result: RecognitionResult, 
                                llm_result: LLMAnalysisResult) -> RecognitionResult:
        """融合识别结果"""
        # 权重配置
        offline_weight = 0.4
        llm_weight = 0.6
        
        # 融合置信度
        fused_confidence = (
            offline_result.confidence * offline_weight +
            llm_result.confidence * llm_weight
        )
        
        # 选择更可信的对象类型
        if llm_result.confidence > offline_result.confidence:
            object_type = llm_result.detected_objects[0]['name'] if llm_result.detected_objects else offline_result.object_type
        else:
            object_type = offline_result.object_type
        
        # 融合建议行动
        fused_actions = list(set(offline_result.suggested_actions + llm_result.suggested_actions))
        
        # 创建融合结果
        fused_result = RecognitionResult(
            object_type=object_type,
            confidence=fused_confidence,
            confidence_level=self._get_confidence_level(fused_confidence),
            bounding_box=offline_result.bounding_box,
            recognition_method='fused',
            processing_time=offline_result.processing_time,
            timestamp=offline_result.timestamp,
            traditional_result=offline_result.traditional_result,
            llm_result=llm_result,
            image_quality_score=offline_result.image_quality_score,
            anti_spoofing_score=offline_result.anti_spoofing_score,
            medical_analysis=offline_result.medical_analysis,
            emergency_level=offline_result.emergency_level,
            requires_learning=offline_result.requires_learning,
            learning_triggered=offline_result.learning_triggered,
            learning_success=offline_result.learning_success,
            suggested_actions=fused_actions,
            confidence_factors={
                'offline_confidence': offline_result.confidence,
                'llm_confidence': llm_result.confidence,
                'fused_confidence': fused_confidence
            }
        )
        
        return fused_result
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """获取置信度等级"""
        thresholds = self.config['recognition']['confidence_thresholds']
        
        if confidence >= thresholds['very_high']:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= thresholds['high']:
            return ConfidenceLevel.HIGH
        elif confidence >= thresholds['medium']:
            return ConfidenceLevel.MEDIUM
        elif confidence >= thresholds['low']:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _assess_emergency_level(self, medical_result: Optional[Dict[str, Any]], 
                              fall_result: Optional[Dict[str, Any]]) -> str:
        """评估紧急程度"""
        if fall_result and fall_result.get('fall_detected', False):
            return 'critical'
        
        if medical_result:
            emergency_indicators = medical_result.get('emergency_indicators', [])
            if emergency_indicators:
                return 'high'
            
            health_score = medical_result.get('health_score', 1.0)
            if health_score < 0.3:
                return 'medium'
        
        return 'low'
    
    def _assess_emergency_level_from_llm(self, llm_result: LLMAnalysisResult) -> str:
        """从大模型结果评估紧急程度"""
        safety_assessment = llm_result.safety_assessment
        medical_relevance = llm_result.medical_relevance
        
        # 检查安全评估
        if safety_assessment.get('danger_level') == 'high':
            return 'critical'
        
        # 检查医疗相关性
        if medical_relevance.get('emergency_level') == 'critical':
            return 'critical'
        elif medical_relevance.get('emergency_level') == 'high':
            return 'high'
        
        # 检查建议行动中的紧急关键词
        emergency_keywords = ['紧急', '急救', '立即', '危险', 'emergency', 'urgent', 'critical']
        for action in llm_result.suggested_actions:
            if any(keyword in action.lower() for keyword in emergency_keywords):
                return 'high'
        
        return 'low'
    
    def _generate_suggested_actions(self, traditional_result: Dict[str, Any],
                                  medical_result: Optional[Dict[str, Any]],
                                  fall_result: Optional[Dict[str, Any]]) -> List[str]:
        """生成建议行动"""
        actions = []
        
        # 基于识别结果的行动
        confidence = traditional_result.get('confidence', 0.0)
        if confidence < 0.5:
            actions.append("建议人工确认识别结果")
        
        # 基于医疗分析的行动
        if medical_result:
            if medical_result.get('health_score', 1.0) < 0.5:
                actions.append("建议医疗检查")
            
            symptoms = medical_result.get('detected_symptoms', [])
            if symptoms:
                actions.append(f"检测到症状: {', '.join(symptoms)}")
        
        # 基于跌倒检测的行动
        if fall_result and fall_result.get('fall_detected', False):
            actions.append("检测到跌倒，立即响应")
            actions.append("联系紧急联系人")
        
        return actions
    
    def _calculate_confidence_factors(self, traditional_result: Dict[str, Any],
                                    quality_info: Dict[str, Any]) -> Dict[str, float]:
        """计算置信度因子"""
        factors = {
            'recognition_confidence': traditional_result.get('confidence', 0.0),
            'image_quality': quality_info['quality_score'],
            'anti_spoofing': quality_info['anti_spoofing_score']
        }
        
        # 计算综合置信度因子
        factors['overall'] = (
            factors['recognition_confidence'] * 0.5 +
            factors['image_quality'] * 0.3 +
            factors['anti_spoofing'] * 0.2
        )
        
        return factors
    
    def _check_cache(self, image: np.ndarray) -> Optional[RecognitionResult]:
        """检查缓存"""
        try:
            # 生成图像哈希作为缓存键
            import hashlib
            image_hash = hashlib.md5(image.tobytes()).hexdigest()
            
            with self.cache_lock:
                if image_hash in self.recognition_cache:
                    cached_data = self.recognition_cache[image_hash]
                    
                    # 检查缓存是否过期
                    if time.time() - cached_data['timestamp'] < self.config['recognition']['cache_ttl']:
                        return cached_data['result']
                    else:
                        # 删除过期缓存
                        del self.recognition_cache[image_hash]
            
        except Exception as e:
            self.logger.warning(f"缓存检查失败: {e}")
        
        return None
    
    def _cache_result(self, image: np.ndarray, result: RecognitionResult):
        """缓存结果"""
        try:
            import hashlib
            image_hash = hashlib.md5(image.tobytes()).hexdigest()
            
            with self.cache_lock:
                self.recognition_cache[image_hash] = {
                    'result': result,
                    'timestamp': time.time()
                }
                
                # 限制缓存大小
                if len(self.recognition_cache) > 1000:
                    # 删除最旧的缓存项
                    oldest_key = min(self.recognition_cache.keys(), 
                                   key=lambda k: self.recognition_cache[k]['timestamp'])
                    del self.recognition_cache[oldest_key]
                    
        except Exception as e:
            self.logger.warning(f"缓存保存失败: {e}")
    
    def _create_error_result(self, error_message: str, processing_time: float) -> RecognitionResult:
        """创建错误结果"""
        return RecognitionResult(
            object_type='error',
            confidence=0.0,
            confidence_level=ConfidenceLevel.VERY_LOW,
            bounding_box=None,
            recognition_method='error',
            processing_time=processing_time,
            timestamp=time.time(),
            traditional_result={'error': error_message},
            llm_result=None,
            image_quality_score=0.0,
            anti_spoofing_score=0.0,
            medical_analysis=None,
            emergency_level='unknown',
            requires_learning=False,
            learning_triggered=False,
            learning_success=None,
            suggested_actions=[f"处理错误: {error_message}"],
            confidence_factors={'error': 1.0}
        )
    
    def _update_statistics(self, result: RecognitionResult):
        """更新统计信息"""
        self.stats['total_recognitions'] += 1
        
        # 按识别方法分类统计
        if 'offline' in result.recognition_method:
            self.stats['offline_recognitions'] += 1
        elif 'hybrid' in result.recognition_method:
            self.stats['hybrid_recognitions'] += 1
        elif 'self_learning' in result.recognition_method:
            self.stats['llm_recognitions'] += 1
        
        # 学习统计
        if result.learning_triggered:
            self.stats['learning_triggered'] += 1
            if result.learning_success:
                self.stats['learning_successful'] += 1
        
        # 平均置信度
        total = self.stats['total_recognitions']
        self.stats['average_confidence'] = (
            (self.stats['average_confidence'] * (total - 1) + result.confidence) / total
        )
        
        # 平均处理时间
        self.stats['average_processing_time'] = (
            (self.stats['average_processing_time'] * (total - 1) + result.processing_time) / total
        )
    
    def batch_recognize(self, images: List[np.ndarray], 
                       contexts: Optional[List[Dict[str, Any]]] = None,
                       mode: Optional[RecognitionMode] = None) -> List[RecognitionResult]:
        """批量识别"""
        if not self.config['recognition']['parallel_processing']:
            # 串行处理
            results = []
            for i, image in enumerate(images):
                context = contexts[i] if contexts else None
                result = self.recognize(image, context, mode)
                results.append(result)
            return results
        
        # 并行处理
        futures = []
        for i, image in enumerate(images):
            context = contexts[i] if contexts else None
            future = self.executor.submit(self.recognize, image, context, mode)
            futures.append(future)
        
        results = []
        for future in as_completed(futures):
            try:
                result = future.result(timeout=self.config['performance']['max_processing_time'])
                results.append(result)
            except Exception as e:
                self.logger.error(f"批量识别中的任务失败: {e}")
                results.append(self._create_error_result(str(e), 0.0))
        
        return results
    
    def get_recognition_statistics(self) -> Dict[str, Any]:
        """获取识别统计信息"""
        llm_stats = self.llm_learning.get_learning_statistics()
        
        return {
            'recognition_stats': self.stats,
            'learning_stats': llm_stats,
            'cache_size': len(self.recognition_cache),
            'subsystem_status': {
                'quality_enhancer': 'active',
                'anti_spoofing': 'active',
                'hybrid_recognition': 'active',
                'medical_analyzer': 'active',
                'fall_detector': 'active',
                'medication_recognizer': 'active',
                'llm_learning': 'active' if self.config['self_learning']['enabled'] else 'disabled'
            }
        }
    
    def update_configuration(self, new_config: Dict[str, Any]):
        """更新配置"""
        # 深度合并配置
        def deep_merge(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_merge(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_merge(self.config, new_config)
        
        # 更新子系统配置
        if hasattr(self, 'llm_learning'):
            self.llm_learning.update_configuration(new_config.get('llm_learning', {}))
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


# 使用示例
if __name__ == "__main__":
    # 创建集成自学习识别系统
    recognition_system = IntegratedSelfLearningRecognition()
    
    # 模拟测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 测试不同识别模式
    modes = [
        RecognitionMode.OFFLINE_ONLY,
        RecognitionMode.HYBRID_AUTO,
        RecognitionMode.SELF_LEARNING
    ]
    
    for mode in modes:
        print(f"\n测试识别模式: {mode.value}")
        
        result = recognition_system.recognize(
            test_image,
            context={"location": "医院", "time": "下午"},
            mode=mode
        )
        
        print(f"识别对象: {result.object_type}")
        print(f"置信度: {result.confidence:.3f} ({result.confidence_level.value})")
        print(f"识别方法: {result.recognition_method}")
        print(f"处理时间: {result.processing_time:.3f}秒")
        print(f"紧急程度: {result.emergency_level}")
        print(f"需要学习: {result.requires_learning}")
        print(f"学习触发: {result.learning_triggered}")
        print(f"建议行动: {result.suggested_actions}")
    
    # 获取统计信息
    stats = recognition_system.get_recognition_statistics()
    print(f"\n系统统计: {stats}")
    
    # 清理资源
    recognition_system.cleanup()
    
    print("集成自学习识别系统测试完成")