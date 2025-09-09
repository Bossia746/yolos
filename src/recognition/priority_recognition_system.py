#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优先级识别系统
处理多目标同时出现时的智能优先级分配和处理策略
确保系统在复杂场景下的健壮性和可用性
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class ObjectCategory(Enum):
    """目标类别"""
    HUMAN = "human"                    # 人类
    HUMAN_FACE = "human_face"         # 人脸
    HUMAN_GESTURE = "human_gesture"   # 手势
    HUMAN_POSE = "human_pose"         # 姿态
    PET = "pet"                       # 宠物
    WILD_ANIMAL = "wild_animal"       # 野生动物
    PLANT = "plant"                   # 植物
    STATIC_OBJECT = "static_object"   # 静物
    VEHICLE = "vehicle"               # 车辆
    MEDICAL_ITEM = "medical_item"     # 医疗用品
    DANGEROUS_ITEM = "dangerous_item" # 危险物品
    UNKNOWN = "unknown"               # 未知物体

class PriorityLevel(IntEnum):
    """优先级等级 (数值越高优先级越高)"""
    CRITICAL = 100      # 紧急情况 (跌倒、危险物品)
    HIGH = 80          # 高优先级 (人脸、医疗相关)
    MEDIUM_HIGH = 60   # 中高优先级 (人体、手势)
    MEDIUM = 40        # 中等优先级 (宠物、车辆)
    MEDIUM_LOW = 30    # 中低优先级 (植物、静物)
    LOW = 20           # 低优先级 (背景物体)
    IGNORE = 0         # 忽略

class ProcessingStrategy(Enum):
    """处理策略"""
    SEQUENTIAL = "sequential"         # 顺序处理
    PARALLEL = "parallel"            # 并行处理
    PRIORITY_FIRST = "priority_first" # 优先级优先
    CONTEXT_AWARE = "context_aware"   # 上下文感知

class SceneContext(Enum):
    """场景上下文"""
    MEDICAL_MONITORING = "medical_monitoring"     # 医疗监控
    SECURITY_SURVEILLANCE = "security_surveillance" # 安防监控
    HOME_AUTOMATION = "home_automation"           # 家庭自动化
    TRAFFIC_MONITORING = "traffic_monitoring"     # 交通监控
    GENERAL_RECOGNITION = "general_recognition"   # 通用识别

@dataclass
class DetectedObject:
    """检测到的目标"""
    category: ObjectCategory
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    priority: PriorityLevel
    processing_time: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """计算优先级分数"""
        self.priority_score = self._calculate_priority_score()
    
    def _calculate_priority_score(self) -> float:
        """计算综合优先级分数"""
        base_score = float(self.priority.value)
        confidence_bonus = self.confidence * 20  # 置信度加成
        size_bonus = self._calculate_size_bonus()
        urgency_bonus = self._calculate_urgency_bonus()
        
        return base_score + confidence_bonus + size_bonus + urgency_bonus
    
    def _calculate_size_bonus(self) -> float:
        """计算尺寸加成"""
        area = self.bbox[2] * self.bbox[3]
        # 面积越大，重要性可能越高
        return min(area / 10000, 10)  # 最大10分加成
    
    def _calculate_urgency_bonus(self) -> float:
        """计算紧急程度加成"""
        urgency_keywords = ['fall', 'emergency', 'danger', 'medical', 'alert']
        bonus = 0
        for keyword in urgency_keywords:
            if keyword in str(self.details).lower():
                bonus += 15
        return min(bonus, 30)  # 最大30分加成

@dataclass
class RecognitionTask:
    """识别任务"""
    image: np.ndarray
    roi: Optional[Tuple[int, int, int, int]]  # 感兴趣区域
    category: ObjectCategory
    priority: PriorityLevel
    timeout: float = 5.0  # 超时时间
    callback: Optional[callable] = None

class PriorityRecognitionSystem:
    """优先级识别系统"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # 优先级配置
        self.priority_rules = self._initialize_priority_rules()
        self.context_rules = self._initialize_context_rules()
        
        # 处理队列
        self.high_priority_queue = deque()
        self.normal_priority_queue = deque()
        self.low_priority_queue = deque()
        
        # 线程池
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.get('max_workers', 4)
        )
        
        # 统计信息
        self.stats = {
            'total_objects': 0,
            'processed_objects': 0,
            'priority_distribution': defaultdict(int),
            'average_processing_time': 0.0,
            'queue_lengths': {'high': 0, 'normal': 0, 'low': 0}
        }
        
        # 当前场景上下文
        self.current_context = SceneContext.GENERAL_RECOGNITION
        
        # 处理策略
        self.processing_strategy = ProcessingStrategy.CONTEXT_AWARE
        
        # 启动处理线程
        self._start_processing_threads()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'max_workers': 4,
            'queue_timeout': 30.0,
            'priority_threshold': 50,
            'confidence_threshold': 0.3,
            'max_objects_per_frame': 20,
            'processing_timeout': 10.0,
            'enable_parallel_processing': True,
            'enable_context_adaptation': True,
            'enable_dynamic_priority': True
        }
    
    def _initialize_priority_rules(self) -> Dict[ObjectCategory, Dict[str, Any]]:
        """初始化优先级规则"""
        return {
            # 人类相关 - 最高优先级
            ObjectCategory.HUMAN: {
                'base_priority': PriorityLevel.HIGH,
                'medical_context_bonus': 20,
                'security_context_bonus': 15,
                'fall_detection_bonus': 50,
                'max_processing_time': 2.0
            },
            ObjectCategory.HUMAN_FACE: {
                'base_priority': PriorityLevel.HIGH,
                'medical_context_bonus': 25,
                'security_context_bonus': 20,
                'emotion_analysis_bonus': 10,
                'max_processing_time': 1.5
            },
            ObjectCategory.HUMAN_GESTURE: {
                'base_priority': PriorityLevel.MEDIUM_HIGH,
                'interaction_context_bonus': 15,
                'emergency_gesture_bonus': 40,
                'max_processing_time': 1.0
            },
            ObjectCategory.HUMAN_POSE: {
                'base_priority': PriorityLevel.MEDIUM_HIGH,
                'fall_detection_bonus': 50,
                'activity_analysis_bonus': 10,
                'max_processing_time': 2.0
            },
            
            # 动物相关
            ObjectCategory.PET: {
                'base_priority': PriorityLevel.MEDIUM,
                'home_context_bonus': 15,
                'behavior_analysis_bonus': 10,
                'max_processing_time': 3.0
            },
            ObjectCategory.WILD_ANIMAL: {
                'base_priority': PriorityLevel.MEDIUM,
                'security_context_bonus': 25,
                'danger_assessment_bonus': 30,
                'max_processing_time': 3.0
            },
            
            # 物品相关
            ObjectCategory.MEDICAL_ITEM: {
                'base_priority': PriorityLevel.MEDIUM_HIGH,
                'medical_context_bonus': 30,
                'emergency_context_bonus': 40,
                'max_processing_time': 2.0
            },
            ObjectCategory.DANGEROUS_ITEM: {
                'base_priority': PriorityLevel.CRITICAL,
                'security_context_bonus': 50,
                'immediate_alert_bonus': 50,
                'max_processing_time': 1.0
            },
            ObjectCategory.VEHICLE: {
                'base_priority': PriorityLevel.MEDIUM,
                'traffic_context_bonus': 30,
                'security_context_bonus': 15,
                'max_processing_time': 2.5
            },
            
            # 环境相关
            ObjectCategory.PLANT: {
                'base_priority': PriorityLevel.MEDIUM_LOW,
                'botanical_context_bonus': 20,
                'health_assessment_bonus': 15,
                'max_processing_time': 4.0
            },
            ObjectCategory.STATIC_OBJECT: {
                'base_priority': PriorityLevel.LOW,
                'inventory_context_bonus': 10,
                'change_detection_bonus': 15,
                'max_processing_time': 5.0
            },
            
            # 未知物体
            ObjectCategory.UNKNOWN: {
                'base_priority': PriorityLevel.MEDIUM_LOW,
                'learning_opportunity_bonus': 20,
                'novelty_bonus': 15,
                'max_processing_time': 6.0
            }
        }
    
    def _initialize_context_rules(self) -> Dict[SceneContext, Dict[str, Any]]:
        """初始化上下文规则"""
        return {
            SceneContext.MEDICAL_MONITORING: {
                'priority_categories': [
                    ObjectCategory.HUMAN,
                    ObjectCategory.HUMAN_FACE,
                    ObjectCategory.HUMAN_POSE,
                    ObjectCategory.MEDICAL_ITEM
                ],
                'priority_multiplier': {
                    ObjectCategory.HUMAN: 1.5,
                    ObjectCategory.HUMAN_FACE: 1.4,
                    ObjectCategory.HUMAN_POSE: 1.6,
                    ObjectCategory.MEDICAL_ITEM: 1.3
                },
                'max_concurrent_tasks': 6,
                'emergency_keywords': ['fall', 'emergency', 'pain', 'help']
            },
            
            SceneContext.SECURITY_SURVEILLANCE: {
                'priority_categories': [
                    ObjectCategory.HUMAN,
                    ObjectCategory.HUMAN_FACE,
                    ObjectCategory.DANGEROUS_ITEM,
                    ObjectCategory.VEHICLE
                ],
                'priority_multiplier': {
                    ObjectCategory.HUMAN: 1.3,
                    ObjectCategory.HUMAN_FACE: 1.5,
                    ObjectCategory.DANGEROUS_ITEM: 2.0,
                    ObjectCategory.VEHICLE: 1.2
                },
                'max_concurrent_tasks': 8,
                'security_keywords': ['intrusion', 'weapon', 'suspicious', 'alert']
            },
            
            SceneContext.HOME_AUTOMATION: {
                'priority_categories': [
                    ObjectCategory.HUMAN,
                    ObjectCategory.HUMAN_GESTURE,
                    ObjectCategory.PET,
                    ObjectCategory.STATIC_OBJECT
                ],
                'priority_multiplier': {
                    ObjectCategory.HUMAN: 1.2,
                    ObjectCategory.HUMAN_GESTURE: 1.4,
                    ObjectCategory.PET: 1.1,
                    ObjectCategory.STATIC_OBJECT: 0.8
                },
                'max_concurrent_tasks': 4,
                'automation_keywords': ['gesture', 'command', 'control', 'automation']
            },
            
            SceneContext.TRAFFIC_MONITORING: {
                'priority_categories': [
                    ObjectCategory.VEHICLE,
                    ObjectCategory.HUMAN,
                    ObjectCategory.DANGEROUS_ITEM
                ],
                'priority_multiplier': {
                    ObjectCategory.VEHICLE: 1.5,
                    ObjectCategory.HUMAN: 1.3,
                    ObjectCategory.DANGEROUS_ITEM: 1.8
                },
                'max_concurrent_tasks': 10,
                'traffic_keywords': ['accident', 'violation', 'congestion', 'emergency']
            },
            
            SceneContext.GENERAL_RECOGNITION: {
                'priority_categories': list(ObjectCategory),
                'priority_multiplier': {cat: 1.0 for cat in ObjectCategory},
                'max_concurrent_tasks': 6,
                'general_keywords': ['unknown', 'analyze', 'identify', 'classify']
            }
        }
    
    def set_scene_context(self, context: SceneContext):
        """设置场景上下文"""
        self.current_context = context
        self.logger.info(f"场景上下文切换到: {context.value}")
        
        # 根据上下文调整处理策略
        self._adapt_processing_strategy()
    
    def _adapt_processing_strategy(self):
        """根据上下文调整处理策略"""
        context_config = self.context_rules[self.current_context]
        
        # 调整线程池大小
        max_workers = min(
            context_config.get('max_concurrent_tasks', 6),
            self.config.get('max_workers', 4)
        )
        
        # 重新创建线程池（如果需要）
        if max_workers != self.executor._max_workers:
            self.executor.shutdown(wait=False)
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def detect_and_prioritize(self, image: np.ndarray, 
                            detected_objects: List[Dict[str, Any]]) -> List[DetectedObject]:
        """检测并分配优先级"""
        prioritized_objects = []
        
        for obj_data in detected_objects:
            # 确定目标类别
            category = self._determine_category(obj_data)
            
            # 计算基础优先级
            base_priority = self._calculate_base_priority(category, obj_data)
            
            # 应用上下文调整
            adjusted_priority = self._apply_context_adjustment(category, base_priority, obj_data)
            
            # 创建检测对象
            detected_obj = DetectedObject(
                category=category,
                confidence=obj_data.get('confidence', 0.0),
                bbox=obj_data.get('bbox', (0, 0, 0, 0)),
                priority=adjusted_priority,
                details=obj_data
            )
            
            prioritized_objects.append(detected_obj)
        
        # 按优先级分数排序
        prioritized_objects.sort(key=lambda x: x.priority_score, reverse=True)
        
        # 限制最大处理数量
        max_objects = self.config.get('max_objects_per_frame', 20)
        if len(prioritized_objects) > max_objects:
            self.logger.warning(f"检测到{len(prioritized_objects)}个目标，限制为{max_objects}个")
            prioritized_objects = prioritized_objects[:max_objects]
        
        return prioritized_objects
    
    def _determine_category(self, obj_data: Dict[str, Any]) -> ObjectCategory:
        """确定目标类别"""
        class_name = obj_data.get('class', '').lower()
        
        # 人类相关
        if any(keyword in class_name for keyword in ['person', 'human', 'people']):
            return ObjectCategory.HUMAN
        elif any(keyword in class_name for keyword in ['face', 'head']):
            return ObjectCategory.HUMAN_FACE
        elif any(keyword in class_name for keyword in ['hand', 'gesture']):
            return ObjectCategory.HUMAN_GESTURE
        elif any(keyword in class_name for keyword in ['pose', 'body', 'skeleton']):
            return ObjectCategory.HUMAN_POSE
        
        # 动物相关
        elif any(keyword in class_name for keyword in ['dog', 'cat', 'pet', 'animal']):
            return ObjectCategory.PET
        elif any(keyword in class_name for keyword in ['wild', 'bear', 'wolf', 'tiger']):
            return ObjectCategory.WILD_ANIMAL
        
        # 车辆相关
        elif any(keyword in class_name for keyword in ['car', 'truck', 'vehicle', 'bus']):
            return ObjectCategory.VEHICLE
        
        # 医疗相关
        elif any(keyword in class_name for keyword in ['medicine', 'pill', 'syringe', 'medical']):
            return ObjectCategory.MEDICAL_ITEM
        
        # 危险物品
        elif any(keyword in class_name for keyword in ['knife', 'gun', 'weapon', 'fire']):
            return ObjectCategory.DANGEROUS_ITEM
        
        # 植物相关
        elif any(keyword in class_name for keyword in ['plant', 'flower', 'tree', 'leaf']):
            return ObjectCategory.PLANT
        
        # 静物
        elif any(keyword in class_name for keyword in ['chair', 'table', 'book', 'bottle']):
            return ObjectCategory.STATIC_OBJECT
        
        # 默认未知
        else:
            return ObjectCategory.UNKNOWN
    
    def _calculate_base_priority(self, category: ObjectCategory, 
                               obj_data: Dict[str, Any]) -> PriorityLevel:
        """计算基础优先级"""
        rules = self.priority_rules.get(category, {})
        base_priority = rules.get('base_priority', PriorityLevel.LOW)
        
        # 检查紧急情况
        if self._is_emergency_situation(obj_data):
            return PriorityLevel.CRITICAL
        
        return base_priority
    
    def _is_emergency_situation(self, obj_data: Dict[str, Any]) -> bool:
        """检查是否为紧急情况"""
        emergency_keywords = ['fall', 'emergency', 'danger', 'alert', 'help', 'accident']
        
        # 检查描述中的紧急关键词
        description = str(obj_data.get('description', '')).lower()
        for keyword in emergency_keywords:
            if keyword in description:
                return True
        
        # 检查特定的紧急情况
        if obj_data.get('fall_detected', False):
            return True
        
        if obj_data.get('weapon_detected', False):
            return True
        
        return False
    
    def _apply_context_adjustment(self, category: ObjectCategory, 
                                base_priority: PriorityLevel,
                                obj_data: Dict[str, Any]) -> PriorityLevel:
        """应用上下文调整"""
        context_config = self.context_rules[self.current_context]
        
        # 获取优先级倍数
        multiplier = context_config.get('priority_multiplier', {}).get(category, 1.0)
        
        # 计算调整后的优先级值
        adjusted_value = int(base_priority.value * multiplier)
        
        # 限制在有效范围内
        adjusted_value = max(0, min(100, adjusted_value))
        
        # 转换回优先级等级
        if adjusted_value >= 90:
            return PriorityLevel.CRITICAL
        elif adjusted_value >= 70:
            return PriorityLevel.HIGH
        elif adjusted_value >= 50:
            return PriorityLevel.MEDIUM_HIGH
        elif adjusted_value >= 35:
            return PriorityLevel.MEDIUM
        elif adjusted_value >= 20:
            return PriorityLevel.MEDIUM_LOW
        elif adjusted_value >= 10:
            return PriorityLevel.LOW
        else:
            return PriorityLevel.IGNORE
    
    def process_objects(self, image: np.ndarray, 
                       detected_objects: List[DetectedObject]) -> List[Dict[str, Any]]:
        """处理检测到的目标"""
        results = []
        
        # 根据处理策略分配任务
        if self.processing_strategy == ProcessingStrategy.SEQUENTIAL:
            results = self._process_sequential(image, detected_objects)
        elif self.processing_strategy == ProcessingStrategy.PARALLEL:
            results = self._process_parallel(image, detected_objects)
        elif self.processing_strategy == ProcessingStrategy.PRIORITY_FIRST:
            results = self._process_priority_first(image, detected_objects)
        else:  # CONTEXT_AWARE
            results = self._process_context_aware(image, detected_objects)
        
        # 更新统计信息
        self._update_statistics(detected_objects, results)
        
        return results
    
    def _process_sequential(self, image: np.ndarray, 
                          detected_objects: List[DetectedObject]) -> List[Dict[str, Any]]:
        """顺序处理"""
        results = []
        
        for obj in detected_objects:
            start_time = time.time()
            
            try:
                result = self._process_single_object(image, obj)
                result['processing_time'] = time.time() - start_time
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"处理目标失败: {e}")
                results.append({
                    'object': obj,
                    'error': str(e),
                    'processing_time': time.time() - start_time
                })
        
        return results
    
    def _process_parallel(self, image: np.ndarray, 
                         detected_objects: List[DetectedObject]) -> List[Dict[str, Any]]:
        """并行处理"""
        results = []
        
        # 提交所有任务
        future_to_obj = {}
        for obj in detected_objects:
            future = self.executor.submit(self._process_single_object, image, obj)
            future_to_obj[future] = obj
        
        # 收集结果
        for future in as_completed(future_to_obj, timeout=self.config.get('processing_timeout', 10.0)):
            obj = future_to_obj[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                self.logger.error(f"并行处理目标失败: {e}")
                results.append({
                    'object': obj,
                    'error': str(e)
                })
        
        return results
    
    def _process_priority_first(self, image: np.ndarray, 
                              detected_objects: List[DetectedObject]) -> List[Dict[str, Any]]:
        """优先级优先处理"""
        results = []
        
        # 按优先级分组
        critical_objects = [obj for obj in detected_objects if obj.priority == PriorityLevel.CRITICAL]
        high_objects = [obj for obj in detected_objects if obj.priority == PriorityLevel.HIGH]
        other_objects = [obj for obj in detected_objects if obj.priority.value < PriorityLevel.HIGH.value]
        
        # 优先处理紧急和高优先级目标
        for obj_group in [critical_objects, high_objects]:
            if obj_group:
                group_results = self._process_parallel(image, obj_group)
                results.extend(group_results)
        
        # 处理其他目标（如果时间允许）
        if other_objects:
            remaining_results = self._process_parallel(image, other_objects[:5])  # 限制数量
            results.extend(remaining_results)
        
        return results
    
    def _process_context_aware(self, image: np.ndarray, 
                             detected_objects: List[DetectedObject]) -> List[Dict[str, Any]]:
        """上下文感知处理"""
        context_config = self.context_rules[self.current_context]
        priority_categories = context_config.get('priority_categories', [])
        
        # 分离优先类别和其他类别
        priority_objects = [obj for obj in detected_objects if obj.category in priority_categories]
        other_objects = [obj for obj in detected_objects if obj.category not in priority_categories]
        
        results = []
        
        # 优先处理重要类别
        if priority_objects:
            priority_results = self._process_priority_first(image, priority_objects)
            results.extend(priority_results)
        
        # 处理其他类别（资源允许的情况下）
        if other_objects and len(results) < context_config.get('max_concurrent_tasks', 6):
            remaining_slots = context_config.get('max_concurrent_tasks', 6) - len(results)
            other_results = self._process_parallel(image, other_objects[:remaining_slots])
            results.extend(other_results)
        
        return results
    
    def _process_single_object(self, image: np.ndarray, obj: DetectedObject) -> Dict[str, Any]:
        """处理单个目标"""
        start_time = time.time()
        
        # 提取ROI
        x, y, w, h = obj.bbox
        roi = image[y:y+h, x:x+w] if w > 0 and h > 0 else image
        
        # 根据类别选择处理方法
        if obj.category == ObjectCategory.HUMAN:
            result = self._process_human(roi, obj)
        elif obj.category == ObjectCategory.HUMAN_FACE:
            result = self._process_face(roi, obj)
        elif obj.category == ObjectCategory.HUMAN_GESTURE:
            result = self._process_gesture(roi, obj)
        elif obj.category == ObjectCategory.HUMAN_POSE:
            result = self._process_pose(roi, obj)
        elif obj.category == ObjectCategory.PET:
            result = self._process_pet(roi, obj)
        elif obj.category == ObjectCategory.MEDICAL_ITEM:
            result = self._process_medical_item(roi, obj)
        elif obj.category == ObjectCategory.DANGEROUS_ITEM:
            result = self._process_dangerous_item(roi, obj)
        else:
            result = self._process_generic_object(roi, obj)
        
        # 添加通用信息
        result.update({
            'object': obj,
            'processing_time': time.time() - start_time,
            'timestamp': time.time()
        })
        
        return result
    
    def _process_human(self, roi: np.ndarray, obj: DetectedObject) -> Dict[str, Any]:
        """处理人体目标"""
        result = {
            'category': 'human',
            'confidence': obj.confidence,
            'analysis': {}
        }
        
        # 跌倒检测
        if self.current_context == SceneContext.MEDICAL_MONITORING:
            fall_detected = self._detect_fall(roi, obj)
            result['analysis']['fall_detected'] = fall_detected
            if fall_detected:
                result['emergency'] = True
                result['alert_level'] = 'critical'
        
        # 行为分析
        behavior = self._analyze_behavior(roi, obj)
        result['analysis']['behavior'] = behavior
        
        return result
    
    def _process_face(self, roi: np.ndarray, obj: DetectedObject) -> Dict[str, Any]:
        """处理人脸目标"""
        result = {
            'category': 'face',
            'confidence': obj.confidence,
            'analysis': {}
        }
        
        # 表情分析
        emotion = self._analyze_emotion(roi)
        result['analysis']['emotion'] = emotion
        
        # 医疗分析（如果在医疗上下文中）
        if self.current_context == SceneContext.MEDICAL_MONITORING:
            health_indicators = self._analyze_health_indicators(roi)
            result['analysis']['health'] = health_indicators
        
        return result
    
    def _process_gesture(self, roi: np.ndarray, obj: DetectedObject) -> Dict[str, Any]:
        """处理手势目标"""
        result = {
            'category': 'gesture',
            'confidence': obj.confidence,
            'analysis': {}
        }
        
        # 手势识别
        gesture_type = self._recognize_gesture(roi)
        result['analysis']['gesture_type'] = gesture_type
        
        # 紧急手势检测
        if gesture_type in ['help', 'emergency', 'stop']:
            result['emergency'] = True
            result['alert_level'] = 'high'
        
        return result
    
    def _process_pose(self, roi: np.ndarray, obj: DetectedObject) -> Dict[str, Any]:
        """处理姿态目标"""
        result = {
            'category': 'pose',
            'confidence': obj.confidence,
            'analysis': {}
        }
        
        # 姿态分析
        pose_data = self._analyze_pose(roi)
        result['analysis']['pose'] = pose_data
        
        # 异常姿态检测
        if pose_data.get('abnormal', False):
            result['alert_level'] = 'medium'
        
        return result
    
    def _process_pet(self, roi: np.ndarray, obj: DetectedObject) -> Dict[str, Any]:
        """处理宠物目标"""
        result = {
            'category': 'pet',
            'confidence': obj.confidence,
            'analysis': {}
        }
        
        # 宠物种类识别
        pet_type = self._identify_pet_type(roi)
        result['analysis']['pet_type'] = pet_type
        
        # 行为分析
        behavior = self._analyze_pet_behavior(roi)
        result['analysis']['behavior'] = behavior
        
        return result
    
    def _process_medical_item(self, roi: np.ndarray, obj: DetectedObject) -> Dict[str, Any]:
        """处理医疗用品目标"""
        result = {
            'category': 'medical_item',
            'confidence': obj.confidence,
            'analysis': {}
        }
        
        # 医疗用品识别
        item_type = self._identify_medical_item(roi)
        result['analysis']['item_type'] = item_type
        
        # 使用状态检测
        usage_status = self._check_medical_item_usage(roi)
        result['analysis']['usage_status'] = usage_status
        
        return result
    
    def _process_dangerous_item(self, roi: np.ndarray, obj: DetectedObject) -> Dict[str, Any]:
        """处理危险物品目标"""
        result = {
            'category': 'dangerous_item',
            'confidence': obj.confidence,
            'analysis': {},
            'emergency': True,
            'alert_level': 'critical'
        }
        
        # 危险物品类型识别
        danger_type = self._identify_danger_type(roi)
        result['analysis']['danger_type'] = danger_type
        
        # 威胁等级评估
        threat_level = self._assess_threat_level(roi, danger_type)
        result['analysis']['threat_level'] = threat_level
        
        return result
    
    def _process_generic_object(self, roi: np.ndarray, obj: DetectedObject) -> Dict[str, Any]:
        """处理通用目标"""
        result = {
            'category': obj.category.value,
            'confidence': obj.confidence,
            'analysis': {}
        }
        
        # 基础特征提取
        features = self._extract_basic_features(roi)
        result['analysis']['features'] = features
        
        return result
    
    # 辅助方法（简化实现）
    def _detect_fall(self, roi: np.ndarray, obj: DetectedObject) -> bool:
        """跌倒检测"""
        # 简化实现：基于边界框的宽高比
        _, _, w, h = obj.bbox
        aspect_ratio = w / h if h > 0 else 0
        return aspect_ratio > 1.5  # 宽度大于高度可能表示跌倒
    
    def _analyze_behavior(self, roi: np.ndarray, obj: DetectedObject) -> str:
        """行为分析"""
        return "normal"  # 简化实现
    
    def _analyze_emotion(self, roi: np.ndarray) -> str:
        """表情分析"""
        return "neutral"  # 简化实现
    
    def _analyze_health_indicators(self, roi: np.ndarray) -> Dict[str, Any]:
        """健康指标分析"""
        return {"status": "normal"}  # 简化实现
    
    def _recognize_gesture(self, roi: np.ndarray) -> str:
        """手势识别"""
        return "unknown"  # 简化实现
    
    def _analyze_pose(self, roi: np.ndarray) -> Dict[str, Any]:
        """姿态分析"""
        return {"abnormal": False}  # 简化实现
    
    def _identify_pet_type(self, roi: np.ndarray) -> str:
        """宠物类型识别"""
        return "unknown"  # 简化实现
    
    def _analyze_pet_behavior(self, roi: np.ndarray) -> str:
        """宠物行为分析"""
        return "normal"  # 简化实现
    
    def _identify_medical_item(self, roi: np.ndarray) -> str:
        """医疗用品识别"""
        return "unknown"  # 简化实现
    
    def _check_medical_item_usage(self, roi: np.ndarray) -> str:
        """医疗用品使用状态检测"""
        return "unused"  # 简化实现
    
    def _identify_danger_type(self, roi: np.ndarray) -> str:
        """危险物品类型识别"""
        return "unknown"  # 简化实现
    
    def _assess_threat_level(self, roi: np.ndarray, danger_type: str) -> str:
        """威胁等级评估"""
        return "medium"  # 简化实现
    
    def _extract_basic_features(self, roi: np.ndarray) -> Dict[str, Any]:
        """基础特征提取"""
        return {"color": "unknown", "shape": "unknown"}  # 简化实现
    
    def _start_processing_threads(self):
        """启动处理线程"""
        # 这里可以启动后台处理线程
        pass
    
    def _update_statistics(self, detected_objects: List[DetectedObject], 
                          results: List[Dict[str, Any]]):
        """更新统计信息"""
        self.stats['total_objects'] += len(detected_objects)
        self.stats['processed_objects'] += len(results)
        
        # 更新优先级分布
        for obj in detected_objects:
            self.stats['priority_distribution'][obj.priority.name] += 1
        
        # 更新平均处理时间
        processing_times = [r.get('processing_time', 0) for r in results if 'processing_time' in r]
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            self.stats['average_processing_time'] = (
                self.stats['average_processing_time'] * 0.9 + avg_time * 0.1
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'current_context': self.current_context.value,
            'processing_strategy': self.processing_strategy.value,
            'queue_lengths': {
                'high': len(self.high_priority_queue),
                'normal': len(self.normal_priority_queue),
                'low': len(self.low_priority_queue)
            }
        }
    
    def shutdown(self):
        """关闭系统"""
        self.executor.shutdown(wait=True)
        self.logger.info("优先级识别系统已关闭")


# 使用示例
if __name__ == "__main__":
    # 创建优先级识别系统
    priority_system = PriorityRecognitionSystem()
    
    # 设置医疗监控场景
    priority_system.set_scene_context(SceneContext.MEDICAL_MONITORING)
    
    # 模拟检测结果
    detected_objects = [
        {'class': 'person', 'confidence': 0.9, 'bbox': (100, 100, 200, 300)},
        {'class': 'cat', 'confidence': 0.8, 'bbox': (300, 200, 150, 100)},
        {'class': 'chair', 'confidence': 0.7, 'bbox': (50, 50, 100, 150)},
        {'class': 'face', 'confidence': 0.95, 'bbox': (120, 110, 80, 100)}
    ]
    
    # 创建模拟图像
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 检测并分配优先级
    prioritized_objects = priority_system.detect_and_prioritize(image, detected_objects)
    
    # 处理目标
    results = priority_system.process_objects(image, prioritized_objects)
    
    # 打印结果
    print("处理结果:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result}")
    
    # 打印统计信息
    print("\n统计信息:")
    stats = priority_system.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 关闭系统
    priority_system.shutdown()