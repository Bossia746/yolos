#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能多目标识别系统
集成优先级处理、资源管理和自适应策略的综合识别系统
确保在人、宠物、静物、动物等多种目标同时出现时的最佳处理效果
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict, deque
import json
import yaml
from pathlib import Path

from .priority_recognition_system import (
    PriorityRecognitionSystem, 
    ObjectCategory, 
    PriorityLevel, 
    SceneContext,
    DetectedObject,
    ProcessingStrategy
)
from .integrated_self_learning_recognition import IntegratedSelfLearningRecognition
from .hybrid_recognition_system import HybridRecognitionSystem

logger = logging.getLogger(__name__)

class ResourceStatus(Enum):
    """资源状态"""
    ABUNDANT = "abundant"      # 资源充足
    MODERATE = "moderate"      # 资源适中
    LIMITED = "limited"        # 资源有限
    CRITICAL = "critical"      # 资源紧张

class AdaptiveStrategy(Enum):
    """自适应策略"""
    QUALITY_FIRST = "quality_first"        # 质量优先
    SPEED_FIRST = "speed_first"           # 速度优先
    BALANCED = "balanced"                 # 平衡模式
    RESOURCE_AWARE = "resource_aware"     # 资源感知

@dataclass
class SystemMetrics:
    """系统指标"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    processing_queue_length: int = 0
    average_processing_time: float = 0.0
    success_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)

@dataclass
class RecognitionContext:
    """识别上下文"""
    scene_type: SceneContext
    lighting_condition: str = "normal"
    motion_level: str = "low"
    object_density: str = "low"
    priority_objects: List[ObjectCategory] = field(default_factory=list)
    time_constraints: Optional[float] = None
    quality_requirements: str = "medium"

class IntelligentMultiTargetSystem:
    """智能多目标识别系统"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # 初始化子系统
        self.priority_system = PriorityRecognitionSystem(self.config.get('priority_system', {}))
        self.self_learning_system = IntegratedSelfLearningRecognition(self.config.get('self_learning', {}))
        self.hybrid_system = HybridRecognitionSystem(self.config.get('hybrid_system', {}))
        
        # 系统状态
        self.current_context = RecognitionContext(SceneContext.GENERAL_RECOGNITION)
        self.current_strategy = AdaptiveStrategy.BALANCED
        self.resource_status = ResourceStatus.ABUNDANT
        
        # 性能监控
        self.metrics_history = deque(maxlen=100)
        self.performance_monitor = threading.Thread(target=self._monitor_performance, daemon=True)
        self.performance_monitor.start()
        
        # 自适应控制
        self.adaptation_enabled = self.config.get('enable_adaptation', True)
        self.adaptation_interval = self.config.get('adaptation_interval', 5.0)
        self.last_adaptation = time.time()
        
        # 结果缓存
        self.result_cache = {}
        self.cache_ttl = self.config.get('cache_ttl', 30.0)
        
        # 统计信息
        self.stats = {
            'total_frames': 0,
            'total_objects': 0,
            'successful_recognitions': 0,
            'failed_recognitions': 0,
            'cache_hits': 0,
            'adaptation_count': 0,
            'strategy_changes': defaultdict(int),
            'category_distribution': defaultdict(int),
            'processing_times': deque(maxlen=1000)
        }
        
        self.logger.info("智能多目标识别系统初始化完成")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = {
            'max_objects_per_frame': 15,
            'processing_timeout': 8.0,
            'quality_threshold': 0.5,
            'enable_caching': True,
            'enable_adaptation': True,
            'adaptation_interval': 5.0,
            'cache_ttl': 30.0,
            'resource_monitoring': True,
            'parallel_processing': True,
            'max_workers': 6,
            
            # 优先级配置
            'priority_weights': {
                'human': 1.0,
                'emergency': 2.0,
                'medical': 1.5,
                'security': 1.3,
                'pet': 0.8,
                'static': 0.5
            },
            
            # 自适应阈值
            'adaptation_thresholds': {
                'cpu_high': 80.0,
                'memory_high': 85.0,
                'queue_length_high': 10,
                'processing_time_high': 3.0,
                'success_rate_low': 0.7
            },
            
            # 策略配置
            'strategies': {
                'quality_first': {
                    'max_processing_time': 10.0,
                    'quality_threshold': 0.8,
                    'parallel_limit': 3
                },
                'speed_first': {
                    'max_processing_time': 2.0,
                    'quality_threshold': 0.4,
                    'parallel_limit': 8
                },
                'balanced': {
                    'max_processing_time': 5.0,
                    'quality_threshold': 0.6,
                    'parallel_limit': 6
                }
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
            except Exception as e:
                self.logger.warning(f"加载配置文件失败: {e}，使用默认配置")
        
        return default_config
    
    def set_recognition_context(self, context: RecognitionContext):
        """设置识别上下文"""
        self.current_context = context
        self.priority_system.set_scene_context(context.scene_type)
        
        # 根据上下文调整策略
        self._adapt_to_context()
        
        self.logger.info(f"识别上下文更新: {context.scene_type.value}")
    
    def _adapt_to_context(self):
        """根据上下文自适应调整"""
        context = self.current_context
        
        # 医疗监控场景 - 质量优先
        if context.scene_type == SceneContext.MEDICAL_MONITORING:
            self.current_strategy = AdaptiveStrategy.QUALITY_FIRST
            
        # 安防监控场景 - 速度优先
        elif context.scene_type == SceneContext.SECURITY_SURVEILLANCE:
            self.current_strategy = AdaptiveStrategy.SPEED_FIRST
            
        # 交通监控场景 - 资源感知
        elif context.scene_type == SceneContext.TRAFFIC_MONITORING:
            self.current_strategy = AdaptiveStrategy.RESOURCE_AWARE
            
        # 其他场景 - 平衡模式
        else:
            self.current_strategy = AdaptiveStrategy.BALANCED
        
        self.stats['strategy_changes'][self.current_strategy.value] += 1
    
    def recognize_multi_targets(self, image: np.ndarray, 
                              detection_results: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """多目标识别主入口"""
        start_time = time.time()
        frame_id = f"frame_{int(time.time() * 1000)}"
        
        try:
            # 检查缓存
            if self.config.get('enable_caching', True):
                cached_result = self._check_cache(image)
                if cached_result:
                    self.stats['cache_hits'] += 1
                    return cached_result
            
            # 预处理和目标检测
            if detection_results is None:
                detection_results = self._detect_objects(image)
            
            # 过滤和优先级分配
            prioritized_objects = self._filter_and_prioritize(image, detection_results)
            
            # 资源评估和策略调整
            if self.adaptation_enabled:
                self._evaluate_and_adapt()
            
            # 智能处理分配
            processing_plan = self._create_processing_plan(prioritized_objects)
            
            # 执行识别
            recognition_results = self._execute_recognition_plan(image, processing_plan)
            
            # 结果融合和后处理
            final_result = self._fuse_and_postprocess(recognition_results, prioritized_objects)
            
            # 更新缓存
            if self.config.get('enable_caching', True):
                self._update_cache(image, final_result)
            
            # 更新统计
            processing_time = time.time() - start_time
            self._update_statistics(final_result, processing_time)
            
            # 添加元信息
            final_result.update({
                'frame_id': frame_id,
                'processing_time': processing_time,
                'strategy_used': self.current_strategy.value,
                'context': self.current_context.scene_type.value,
                'resource_status': self.resource_status.value,
                'timestamp': time.time()
            })
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"多目标识别失败: {e}")
            self.stats['failed_recognitions'] += 1
            return {
                'frame_id': frame_id,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'timestamp': time.time()
            }
    
    def _detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """目标检测"""
        # 使用混合系统进行基础检测
        detection_result = self.hybrid_system.detect_objects(image)
        
        # 转换为标准格式
        objects = []
        if 'detections' in detection_result:
            for detection in detection_result['detections']:
                objects.append({
                    'class': detection.get('class', 'unknown'),
                    'confidence': detection.get('confidence', 0.0),
                    'bbox': detection.get('bbox', (0, 0, 0, 0)),
                    'features': detection.get('features', {})
                })
        
        return objects
    
    def _filter_and_prioritize(self, image: np.ndarray, 
                             detection_results: List[Dict[str, Any]]) -> List[DetectedObject]:
        """过滤和优先级分配"""
        # 质量过滤
        quality_threshold = self.config.get('quality_threshold', 0.5)
        filtered_objects = [
            obj for obj in detection_results 
            if obj.get('confidence', 0) >= quality_threshold
        ]
        
        # 数量限制
        max_objects = self.config.get('max_objects_per_frame', 15)
        if len(filtered_objects) > max_objects:
            # 按置信度排序，保留前N个
            filtered_objects.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            filtered_objects = filtered_objects[:max_objects]
            self.logger.info(f"目标数量限制: {len(detection_results)} -> {max_objects}")
        
        # 优先级分配
        prioritized_objects = self.priority_system.detect_and_prioritize(image, filtered_objects)
        
        return prioritized_objects
    
    def _evaluate_and_adapt(self):
        """评估系统状态并自适应调整"""
        current_time = time.time()
        if current_time - self.last_adaptation < self.adaptation_interval:
            return
        
        # 获取当前系统指标
        metrics = self._get_current_metrics()
        self.metrics_history.append(metrics)
        
        # 评估资源状态
        self._evaluate_resource_status(metrics)
        
        # 策略调整
        if self._should_adapt_strategy(metrics):
            self._adapt_strategy(metrics)
            self.stats['adaptation_count'] += 1
        
        self.last_adaptation = current_time
    
    def _get_current_metrics(self) -> SystemMetrics:
        """获取当前系统指标"""
        try:
            import psutil
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            
            # GPU使用率（如果可用）
            gpu_usage = 0.0
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage = gpus[0].load * 100
            except ImportError:
                pass
            
        except ImportError:
            cpu_usage = 0.0
            memory_usage = 0.0
            gpu_usage = 0.0
        
        # 处理队列长度
        queue_length = (
            len(self.priority_system.high_priority_queue) +
            len(self.priority_system.normal_priority_queue) +
            len(self.priority_system.low_priority_queue)
        )
        
        # 平均处理时间
        avg_time = 0.0
        if self.stats['processing_times']:
            avg_time = sum(self.stats['processing_times']) / len(self.stats['processing_times'])
        
        # 成功率
        total_attempts = self.stats['successful_recognitions'] + self.stats['failed_recognitions']
        success_rate = (
            self.stats['successful_recognitions'] / total_attempts 
            if total_attempts > 0 else 1.0
        )
        
        return SystemMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            processing_queue_length=queue_length,
            average_processing_time=avg_time,
            success_rate=success_rate
        )
    
    def _evaluate_resource_status(self, metrics: SystemMetrics):
        """评估资源状态"""
        thresholds = self.config.get('adaptation_thresholds', {})
        
        # 计算资源压力分数
        pressure_score = 0
        
        if metrics.cpu_usage > thresholds.get('cpu_high', 80):
            pressure_score += 2
        elif metrics.cpu_usage > 60:
            pressure_score += 1
        
        if metrics.memory_usage > thresholds.get('memory_high', 85):
            pressure_score += 2
        elif metrics.memory_usage > 70:
            pressure_score += 1
        
        if metrics.processing_queue_length > thresholds.get('queue_length_high', 10):
            pressure_score += 2
        elif metrics.processing_queue_length > 5:
            pressure_score += 1
        
        if metrics.average_processing_time > thresholds.get('processing_time_high', 3.0):
            pressure_score += 1
        
        # 更新资源状态
        if pressure_score >= 5:
            self.resource_status = ResourceStatus.CRITICAL
        elif pressure_score >= 3:
            self.resource_status = ResourceStatus.LIMITED
        elif pressure_score >= 1:
            self.resource_status = ResourceStatus.MODERATE
        else:
            self.resource_status = ResourceStatus.ABUNDANT
    
    def _should_adapt_strategy(self, metrics: SystemMetrics) -> bool:
        """判断是否需要调整策略"""
        thresholds = self.config.get('adaptation_thresholds', {})
        
        # 性能问题
        if (metrics.cpu_usage > thresholds.get('cpu_high', 80) or
            metrics.memory_usage > thresholds.get('memory_high', 85) or
            metrics.processing_queue_length > thresholds.get('queue_length_high', 10)):
            return True
        
        # 质量问题
        if metrics.success_rate < thresholds.get('success_rate_low', 0.7):
            return True
        
        # 时间问题
        if metrics.average_processing_time > thresholds.get('processing_time_high', 3.0):
            return True
        
        return False
    
    def _adapt_strategy(self, metrics: SystemMetrics):
        """自适应策略调整"""
        old_strategy = self.current_strategy
        
        # 资源紧张 -> 速度优先
        if self.resource_status in [ResourceStatus.CRITICAL, ResourceStatus.LIMITED]:
            self.current_strategy = AdaptiveStrategy.SPEED_FIRST
            
        # 成功率低 -> 质量优先
        elif metrics.success_rate < 0.7:
            self.current_strategy = AdaptiveStrategy.QUALITY_FIRST
            
        # 资源充足 -> 平衡模式
        elif self.resource_status == ResourceStatus.ABUNDANT:
            self.current_strategy = AdaptiveStrategy.BALANCED
            
        # 其他情况 -> 资源感知
        else:
            self.current_strategy = AdaptiveStrategy.RESOURCE_AWARE
        
        if old_strategy != self.current_strategy:
            self.logger.info(f"策略调整: {old_strategy.value} -> {self.current_strategy.value}")
            self._apply_strategy_config()
    
    def _apply_strategy_config(self):
        """应用策略配置"""
        strategy_config = self.config.get('strategies', {}).get(self.current_strategy.value, {})
        
        # 更新处理超时
        if 'max_processing_time' in strategy_config:
            self.config['processing_timeout'] = strategy_config['max_processing_time']
        
        # 更新质量阈值
        if 'quality_threshold' in strategy_config:
            self.config['quality_threshold'] = strategy_config['quality_threshold']
        
        # 更新并行限制
        if 'parallel_limit' in strategy_config:
            self.config['max_workers'] = strategy_config['parallel_limit']
    
    def _create_processing_plan(self, prioritized_objects: List[DetectedObject]) -> Dict[str, Any]:
        """创建处理计划"""
        plan = {
            'critical_objects': [],
            'high_priority_objects': [],
            'normal_objects': [],
            'low_priority_objects': [],
            'processing_strategy': self.current_strategy.value,
            'resource_allocation': {}
        }
        
        # 按优先级分组
        for obj in prioritized_objects:
            if obj.priority == PriorityLevel.CRITICAL:
                plan['critical_objects'].append(obj)
            elif obj.priority == PriorityLevel.HIGH:
                plan['high_priority_objects'].append(obj)
            elif obj.priority.value >= PriorityLevel.MEDIUM.value:
                plan['normal_objects'].append(obj)
            else:
                plan['low_priority_objects'].append(obj)
        
        # 资源分配策略
        total_objects = len(prioritized_objects)
        if total_objects > 0:
            # 关键目标分配50%资源
            critical_ratio = min(0.5, len(plan['critical_objects']) / total_objects)
            # 高优先级分配30%资源
            high_ratio = min(0.3, len(plan['high_priority_objects']) / total_objects)
            # 其余分配剩余资源
            remaining_ratio = 1.0 - critical_ratio - high_ratio
            
            plan['resource_allocation'] = {
                'critical': critical_ratio,
                'high': high_ratio,
                'normal': remaining_ratio * 0.7,
                'low': remaining_ratio * 0.3
            }
        
        return plan
    
    def _execute_recognition_plan(self, image: np.ndarray, 
                                plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """执行识别计划"""
        results = []
        
        # 按优先级顺序处理
        processing_groups = [
            ('critical', plan['critical_objects']),
            ('high', plan['high_priority_objects']),
            ('normal', plan['normal_objects']),
            ('low', plan['low_priority_objects'])
        ]
        
        for group_name, objects in processing_groups:
            if not objects:
                continue
            
            group_results = self._process_object_group(image, objects, group_name, plan)
            results.extend(group_results)
            
            # 检查时间限制
            if self._should_stop_processing():
                self.logger.info(f"达到时间限制，停止处理剩余{group_name}组目标")
                break
        
        return results
    
    def _process_object_group(self, image: np.ndarray, objects: List[DetectedObject],
                            group_name: str, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """处理目标组"""
        if not objects:
            return []
        
        # 根据策略选择处理方式
        if self.current_strategy == AdaptiveStrategy.SPEED_FIRST:
            return self._process_group_parallel(image, objects, group_name)
        elif self.current_strategy == AdaptiveStrategy.QUALITY_FIRST:
            return self._process_group_sequential(image, objects, group_name)
        else:
            # 平衡模式或资源感知模式
            if len(objects) <= 3 or self.resource_status == ResourceStatus.LIMITED:
                return self._process_group_sequential(image, objects, group_name)
            else:
                return self._process_group_parallel(image, objects, group_name)
    
    def _process_group_sequential(self, image: np.ndarray, objects: List[DetectedObject],
                                group_name: str) -> List[Dict[str, Any]]:
        """顺序处理目标组"""
        results = []
        
        for obj in objects:
            try:
                result = self._process_single_object(image, obj)
                results.append(result)
                
                # 检查是否需要停止
                if self._should_stop_processing():
                    break
                    
            except Exception as e:
                self.logger.error(f"处理{group_name}组目标失败: {e}")
                results.append({
                    'object': obj,
                    'error': str(e),
                    'group': group_name
                })
        
        return results
    
    def _process_group_parallel(self, image: np.ndarray, objects: List[DetectedObject],
                              group_name: str) -> List[Dict[str, Any]]:
        """并行处理目标组"""
        results = []
        
        # 限制并行数量
        max_workers = min(len(objects), self.config.get('max_workers', 6))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            future_to_obj = {
                executor.submit(self._process_single_object, image, obj): obj
                for obj in objects
            }
            
            # 收集结果
            timeout = self.config.get('processing_timeout', 8.0)
            for future in as_completed(future_to_obj, timeout=timeout):
                obj = future_to_obj[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"并行处理{group_name}组目标失败: {e}")
                    results.append({
                        'object': obj,
                        'error': str(e),
                        'group': group_name
                    })
        
        return results
    
    def _process_single_object(self, image: np.ndarray, obj: DetectedObject) -> Dict[str, Any]:
        """处理单个目标"""
        start_time = time.time()
        
        # 使用优先级系统处理
        priority_result = self.priority_system._process_single_object(image, obj)
        
        # 如果需要自学习
        if (obj.category == ObjectCategory.UNKNOWN or 
            priority_result.get('confidence', 0) < 0.6):
            
            # 提取ROI
            x, y, w, h = obj.bbox
            roi = image[y:y+h, x:x+w] if w > 0 and h > 0 else image
            
            # 使用自学习系统
            try:
                learning_result = self.self_learning_system.recognize(roi)
                if learning_result and hasattr(learning_result, 'confidence'):
                    if learning_result.confidence > priority_result.get('confidence', 0):
                        priority_result.update({
                            'enhanced_by_learning': True,
                            'learning_confidence': learning_result.confidence,
                            'learning_description': getattr(learning_result, 'description', '')
                        })
            except Exception as e:
                self.logger.warning(f"自学习处理失败: {e}")
        
        # 添加处理信息
        priority_result.update({
            'processing_time': time.time() - start_time,
            'processed_by': 'intelligent_multi_target_system',
            'strategy': self.current_strategy.value
        })
        
        return priority_result
    
    def _should_stop_processing(self) -> bool:
        """判断是否应该停止处理"""
        # 检查时间约束
        if self.current_context.time_constraints:
            # 这里需要实现时间检查逻辑
            pass
        
        # 检查资源状态
        if self.resource_status == ResourceStatus.CRITICAL:
            return True
        
        return False
    
    def _fuse_and_postprocess(self, recognition_results: List[Dict[str, Any]], 
                            prioritized_objects: List[DetectedObject]) -> Dict[str, Any]:
        """结果融合和后处理"""
        # 统计信息
        total_objects = len(prioritized_objects)
        processed_objects = len([r for r in recognition_results if 'error' not in r])
        
        # 按类别分组
        category_results = defaultdict(list)
        for result in recognition_results:
            if 'error' not in result and 'object' in result:
                category = result['object'].category.value
                category_results[category].append(result)
        
        # 检测紧急情况
        emergency_alerts = []
        for result in recognition_results:
            if result.get('emergency', False):
                emergency_alerts.append({
                    'type': result.get('category', 'unknown'),
                    'alert_level': result.get('alert_level', 'medium'),
                    'description': result.get('analysis', {}).get('description', ''),
                    'bbox': result.get('object', {}).bbox if hasattr(result.get('object', {}), 'bbox') else None
                })
        
        # 构建最终结果
        final_result = {
            'success': True,
            'total_objects_detected': total_objects,
            'objects_processed': processed_objects,
            'processing_success_rate': processed_objects / total_objects if total_objects > 0 else 1.0,
            
            # 分类结果
            'results_by_category': dict(category_results),
            'category_counts': {cat: len(results) for cat, results in category_results.items()},
            
            # 紧急情况
            'emergency_alerts': emergency_alerts,
            'has_emergency': len(emergency_alerts) > 0,
            
            # 详细结果
            'detailed_results': recognition_results,
            
            # 系统状态
            'system_status': {
                'resource_status': self.resource_status.value,
                'current_strategy': self.current_strategy.value,
                'context': self.current_context.scene_type.value
            }
        }
        
        return final_result
    
    def _check_cache(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """检查结果缓存"""
        if not self.config.get('enable_caching', True):
            return None
        
        # 简单的图像哈希缓存（实际应用中可以使用更复杂的方法）
        image_hash = hash(image.tobytes())
        
        if image_hash in self.result_cache:
            cached_data = self.result_cache[image_hash]
            if time.time() - cached_data['timestamp'] < self.cache_ttl:
                return cached_data['result']
            else:
                # 缓存过期，删除
                del self.result_cache[image_hash]
        
        return None
    
    def _update_cache(self, image: np.ndarray, result: Dict[str, Any]):
        """更新结果缓存"""
        if not self.config.get('enable_caching', True):
            return
        
        image_hash = hash(image.tobytes())
        self.result_cache[image_hash] = {
            'result': result,
            'timestamp': time.time()
        }
        
        # 限制缓存大小
        if len(self.result_cache) > 100:
            # 删除最旧的缓存项
            oldest_key = min(self.result_cache.keys(), 
                           key=lambda k: self.result_cache[k]['timestamp'])
            del self.result_cache[oldest_key]
    
    def _update_statistics(self, result: Dict[str, Any], processing_time: float):
        """更新统计信息"""
        self.stats['total_frames'] += 1
        self.stats['total_objects'] += result.get('total_objects_detected', 0)
        self.stats['processing_times'].append(processing_time)
        
        if result.get('success', False):
            self.stats['successful_recognitions'] += 1
        else:
            self.stats['failed_recognitions'] += 1
        
        # 更新类别分布
        for category, count in result.get('category_counts', {}).items():
            self.stats['category_distribution'][category] += count
    
    def _monitor_performance(self):
        """性能监控线程"""
        while True:
            try:
                time.sleep(10)  # 每10秒监控一次
                
                if self.config.get('resource_monitoring', True):
                    metrics = self._get_current_metrics()
                    
                    # 记录性能日志
                    if metrics.cpu_usage > 90 or metrics.memory_usage > 95:
                        self.logger.warning(
                            f"系统资源紧张 - CPU: {metrics.cpu_usage:.1f}%, "
                            f"内存: {metrics.memory_usage:.1f}%"
                        )
                    
                    # 清理过期缓存
                    self._cleanup_cache()
                    
            except Exception as e:
                self.logger.error(f"性能监控异常: {e}")
    
    def _cleanup_cache(self):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = [
            key for key, data in self.result_cache.items()
            if current_time - data['timestamp'] > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.result_cache[key]
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        current_metrics = self._get_current_metrics()
        
        return {
            'current_context': self.current_context.scene_type.value,
            'current_strategy': self.current_strategy.value,
            'resource_status': self.resource_status.value,
            'current_metrics': {
                'cpu_usage': current_metrics.cpu_usage,
                'memory_usage': current_metrics.memory_usage,
                'gpu_usage': current_metrics.gpu_usage,
                'queue_length': current_metrics.processing_queue_length,
                'avg_processing_time': current_metrics.average_processing_time,
                'success_rate': current_metrics.success_rate
            },
            'statistics': dict(self.stats),
            'cache_size': len(self.result_cache),
            'adaptation_enabled': self.adaptation_enabled
        }
    
    def shutdown(self):
        """关闭系统"""
        self.logger.info("正在关闭智能多目标识别系统...")
        
        # 关闭子系统
        if hasattr(self.priority_system, 'shutdown'):
            self.priority_system.shutdown()
        
        # 清理资源
        self.result_cache.clear()
        
        self.logger.info("智能多目标识别系统已关闭")


# 使用示例
if __name__ == "__main__":
    # 创建智能多目标识别系统
    system = IntelligentMultiTargetSystem()
    
    # 设置医疗监控上下文
    context = RecognitionContext(
        scene_type=SceneContext.MEDICAL_MONITORING,
        lighting_condition="normal",
        priority_objects=[ObjectCategory.HUMAN, ObjectCategory.HUMAN_FACE],
        quality_requirements="high"
    )
    system.set_recognition_context(context)
    
    # 模拟图像和检测结果
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    detection_results = [
        {'class': 'person', 'confidence': 0.9, 'bbox': (100, 100, 200, 300)},
        {'class': 'cat', 'confidence': 0.8, 'bbox': (300, 200, 150, 100)},
        {'class': 'face', 'confidence': 0.95, 'bbox': (120, 110, 80, 100)},
        {'class': 'chair', 'confidence': 0.7, 'bbox': (50, 50, 100, 150)},
        {'class': 'plant', 'confidence': 0.6, 'bbox': (400, 300, 100, 120)}
    ]
    
    # 执行多目标识别
    result = system.recognize_multi_targets(image, detection_results)
    
    # 打印结果
    print("=== 智能多目标识别结果 ===")
    print(f"处理策略: {result.get('strategy_used')}")
    print(f"场景上下文: {result.get('context')}")
    print(f"资源状态: {result.get('resource_status')}")
    print(f"处理时间: {result.get('processing_time', 0):.3f}秒")
    print(f"检测目标数: {result.get('total_objects_detected', 0)}")
    print(f"处理成功数: {result.get('objects_processed', 0)}")
    print(f"成功率: {result.get('processing_success_rate', 0):.2%}")
    
    if result.get('has_emergency'):
        print("\n⚠️ 紧急情况警报:")
        for alert in result.get('emergency_alerts', []):
            print(f"  - {alert['type']}: {alert['alert_level']} - {alert['description']}")
    
    print(f"\n类别分布:")
    for category, count in result.get('category_counts', {}).items():
        print(f"  {category}: {count}")
    
    # 获取系统状态
    status = system.get_system_status()
    print(f"\n=== 系统状态 ===")
    print(f"CPU使用率: {status['current_metrics']['cpu_usage']:.1f}%")
    print(f"内存使用率: {status['current_metrics']['memory_usage']:.1f}%")
    print(f"平均处理时间: {status['current_metrics']['avg_processing_time']:.3f}秒")
    print(f"缓存大小: {status['cache_size']}")
    print(f"自适应调整次数: {status['statistics']['adaptation_count']}")
    
    # 关闭系统
    system.shutdown()