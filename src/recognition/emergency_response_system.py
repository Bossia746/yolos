#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
紧急响应系统
专门用于AIoT无人机械的紧急医疗响应
集成摔倒检测、面部生理分析和自动报警功能
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
import time
import json
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import queue

from .medical_facial_analyzer import MedicalFacialAnalyzer, FacialAnalysisResult, HealthStatus
from .anti_spoofing_detector import AntiSpoofingDetector

class EmergencyType(Enum):
    """紧急情况类型"""
    FALL_DETECTED = "fall_detected"
    MEDICAL_EMERGENCY = "medical_emergency"
    UNCONSCIOUS = "unconscious"
    CARDIAC_EVENT = "cardiac_event"
    RESPIRATORY_DISTRESS = "respiratory_distress"
    STROKE_SUSPECTED = "stroke_suspected"
    TRAUMA = "trauma"
    UNKNOWN_EMERGENCY = "unknown_emergency"

class ResponseLevel(Enum):
    """响应级别"""
    MONITORING = "monitoring"
    ALERT = "alert"
    URGENT = "urgent"
    CRITICAL = "critical"
    IMMEDIATE = "immediate"

@dataclass
class EmergencyEvent:
    """紧急事件数据"""
    event_id: str
    timestamp: float
    emergency_type: EmergencyType
    response_level: ResponseLevel
    location: Optional[Tuple[float, float]] = None  # GPS坐标
    patient_info: Optional[Dict[str, Any]] = None
    medical_analysis: Optional[FacialAnalysisResult] = None
    fall_detection_data: Optional[Dict[str, Any]] = None
    environmental_data: Optional[Dict[str, Any]] = None
    response_actions: List[str] = None
    estimated_response_time: Optional[float] = None

@dataclass
class DroneStatus:
    """无人机状态"""
    drone_id: str
    position: Tuple[float, float, float]  # x, y, z
    battery_level: float
    camera_status: bool
    communication_status: bool
    mission_status: str
    last_update: float

class EmergencyResponseSystem:
    """紧急响应系统"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # 初始化分析器
        self.medical_analyzer = MedicalFacialAnalyzer()
        self.anti_spoofing = AntiSpoofingDetector()
        
        # 事件队列和历史记录
        self.event_queue = queue.Queue()
        self.active_events = {}
        self.event_history = []
        
        # 无人机状态
        self.drone_status = {}
        
        # 回调函数
        self.emergency_callbacks = []
        self.alert_callbacks = []
        
        # 监控线程
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # 摔倒检测相关
        self.fall_detector = self._initialize_fall_detector()
        self.previous_frame = None
        self.person_tracker = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            # 摔倒检测参数
            'fall_detection': {
                'enable': True,
                'sensitivity': 0.7,
                'min_fall_duration': 1.0,  # 秒
                'position_change_threshold': 50,  # 像素
                'aspect_ratio_threshold': 1.5
            },
            
            # 医疗分析参数
            'medical_analysis': {
                'enable': True,
                'analysis_interval': 2.0,  # 秒
                'critical_threshold': 80,
                'urgent_threshold': 60
            },
            
            # 响应参数
            'response': {
                'auto_alert_threshold': 4,  # 紧急等级
                'max_response_time': 300,  # 秒
                'retry_attempts': 3,
                'escalation_time': 60  # 秒
            },
            
            # 无人机参数
            'drone': {
                'approach_distance': 2.0,  # 米
                'hover_height': 1.5,  # 米
                'battery_warning_level': 20,  # 百分比
                'max_mission_time': 1800  # 秒
            },
            
            # 通信参数
            'communication': {
                'emergency_contacts': [],
                'medical_center_api': None,
                'backup_communication': True
            }
        }
    
    def _initialize_fall_detector(self):
        """初始化摔倒检测器"""
        # 使用YOLO或其他目标检测模型
        # 这里使用简化的基于轮廓的检测
        return {
            'background_subtractor': cv2.createBackgroundSubtractorMOG2(detectShadows=True),
            'min_contour_area': 1000,
            'max_contour_area': 50000
        }
    
    def start_monitoring(self):
        """开始监控"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            self.logger.info("紧急响应监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("紧急响应监控已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 处理事件队列
                self._process_event_queue()
                
                # 检查活跃事件状态
                self._check_active_events()
                
                # 更新无人机状态
                self._update_drone_status()
                
                time.sleep(0.1)  # 100ms循环
                
            except Exception as e:
                self.logger.error(f"监控循环错误: {e}")
    
    def process_frame(self, frame: np.ndarray, 
                     drone_id: str = "default",
                     gps_location: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """处理视频帧"""
        results = {
            'timestamp': time.time(),
            'drone_id': drone_id,
            'fall_detected': False,
            'medical_analysis': None,
            'emergency_events': [],
            'recommendations': []
        }
        
        try:
            # 1. 摔倒检测
            if self.config['fall_detection']['enable']:
                fall_result = self._detect_fall(frame)
                results['fall_detected'] = fall_result['detected']
                results['fall_data'] = fall_result
                
                if fall_result['detected']:
                    # 创建摔倒紧急事件
                    event = self._create_fall_emergency_event(
                        fall_result, gps_location, drone_id
                    )
                    results['emergency_events'].append(event)
                    self.event_queue.put(event)
            
            # 2. 医疗面部分析
            if self.config['medical_analysis']['enable']:
                medical_result = self.medical_analyzer.analyze_facial_health(
                    frame, self.previous_frame
                )
                results['medical_analysis'] = medical_result
                
                # 检查是否需要紧急响应
                if medical_result.emergency_level >= self.config['response']['auto_alert_threshold']:
                    event = self._create_medical_emergency_event(
                        medical_result, gps_location, drone_id
                    )
                    results['emergency_events'].append(event)
                    self.event_queue.put(event)
            
            # 3. 反欺骗检测（确保不是误报）
            spoofing_result = self.anti_spoofing.detect_spoofing(frame, self.previous_frame)
            if not spoofing_result.is_real:
                results['recommendations'].append("检测到可能的误报，建议人工确认")
            
            # 4. 综合评估和建议
            recommendations = self._generate_response_recommendations(results)
            results['recommendations'].extend(recommendations)
            
            # 更新前一帧
            self.previous_frame = frame.copy()
            
        except Exception as e:
            self.logger.error(f"帧处理失败: {e}")
            results['error'] = str(e)
        
        return results
    
    def _detect_fall(self, frame: np.ndarray) -> Dict[str, Any]:
        """检测摔倒"""
        fall_result = {
            'detected': False,
            'confidence': 0.0,
            'person_positions': [],
            'fall_type': None,
            'analysis_details': {}
        }
        
        try:
            # 背景减除
            fg_mask = self.fall_detector['background_subtractor'].apply(frame)
            
            # 形态学操作去噪
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 过滤轮廓
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if (self.fall_detector['min_contour_area'] < area < 
                    self.fall_detector['max_contour_area']):
                    valid_contours.append(contour)
            
            # 分析每个有效轮廓
            for contour in valid_contours:
                # 计算边界框
                x, y, w, h = cv2.boundingRect(contour)
                
                # 计算长宽比
                aspect_ratio = w / max(h, 1)
                
                # 摔倒检测逻辑
                if aspect_ratio > self.config['fall_detection']['aspect_ratio_threshold']:
                    # 人体呈水平状态，可能摔倒
                    fall_confidence = min(aspect_ratio / 2.0, 1.0)
                    
                    if fall_confidence > self.config['fall_detection']['sensitivity']:
                        fall_result['detected'] = True
                        fall_result['confidence'] = fall_confidence
                        fall_result['person_positions'].append((x, y, w, h))
                        fall_result['fall_type'] = 'horizontal_position'
                
                # 记录分析详情
                fall_result['analysis_details'][f'contour_{len(fall_result["analysis_details"])}'] = {
                    'position': (x, y, w, h),
                    'area': cv2.contourArea(contour),
                    'aspect_ratio': aspect_ratio
                }
            
        except Exception as e:
            self.logger.error(f"摔倒检测失败: {e}")
            fall_result['error'] = str(e)
        
        return fall_result
    
    def _create_fall_emergency_event(self, fall_data: Dict[str, Any], 
                                   location: Optional[Tuple[float, float]], 
                                   drone_id: str) -> EmergencyEvent:
        """创建摔倒紧急事件"""
        event_id = f"fall_{int(time.time() * 1000)}"
        
        # 确定响应级别
        confidence = fall_data.get('confidence', 0.0)
        if confidence > 0.9:
            response_level = ResponseLevel.CRITICAL
        elif confidence > 0.7:
            response_level = ResponseLevel.URGENT
        else:
            response_level = ResponseLevel.ALERT
        
        # 生成响应行动
        response_actions = [
            "无人机接近患者位置",
            "启动医疗面部分析",
            "记录现场情况",
            "准备联系急救服务"
        ]
        
        if response_level in [ResponseLevel.CRITICAL, ResponseLevel.URGENT]:
            response_actions.insert(-1, "立即联系急救服务")
        
        return EmergencyEvent(
            event_id=event_id,
            timestamp=time.time(),
            emergency_type=EmergencyType.FALL_DETECTED,
            response_level=response_level,
            location=location,
            fall_detection_data=fall_data,
            response_actions=response_actions,
            estimated_response_time=self._estimate_response_time(location, drone_id)
        )
    
    def _create_medical_emergency_event(self, medical_result: FacialAnalysisResult,
                                      location: Optional[Tuple[float, float]],
                                      drone_id: str) -> EmergencyEvent:
        """创建医疗紧急事件"""
        event_id = f"medical_{int(time.time() * 1000)}"
        
        # 确定紧急类型
        emergency_type = EmergencyType.MEDICAL_EMERGENCY
        
        # 基于症状细化紧急类型
        from .medical_facial_analyzer import FacialSymptom
        if FacialSymptom.LOSS_OF_CONSCIOUSNESS in medical_result.symptoms:
            emergency_type = EmergencyType.UNCONSCIOUS
        elif FacialSymptom.ASYMMETRIC_FACE in medical_result.symptoms:
            emergency_type = EmergencyType.STROKE_SUSPECTED
        elif FacialSymptom.LABORED_BREATHING in medical_result.symptoms:
            emergency_type = EmergencyType.RESPIRATORY_DISTRESS
        elif FacialSymptom.CYANOSIS in medical_result.symptoms:
            emergency_type = EmergencyType.CARDIAC_EVENT
        
        # 确定响应级别
        if medical_result.emergency_level >= 5:
            response_level = ResponseLevel.IMMEDIATE
        elif medical_result.emergency_level >= 4:
            response_level = ResponseLevel.CRITICAL
        elif medical_result.emergency_level >= 3:
            response_level = ResponseLevel.URGENT
        else:
            response_level = ResponseLevel.ALERT
        
        # 生成响应行动
        response_actions = medical_result.recommendations.copy()
        response_actions.extend([
            "持续监测患者状态",
            "记录症状变化",
            "准备提供医疗指导"
        ])
        
        return EmergencyEvent(
            event_id=event_id,
            timestamp=time.time(),
            emergency_type=emergency_type,
            response_level=response_level,
            location=location,
            medical_analysis=medical_result,
            response_actions=response_actions,
            estimated_response_time=self._estimate_response_time(location, drone_id)
        )
    
    def _estimate_response_time(self, location: Optional[Tuple[float, float]], 
                              drone_id: str) -> float:
        """估算响应时间"""
        base_time = 60.0  # 基础响应时间（秒）
        
        # 基于无人机状态调整
        if drone_id in self.drone_status:
            drone = self.drone_status[drone_id]
            
            # 基于电池电量调整
            if drone.battery_level < 30:
                base_time += 30
            
            # 基于通信状态调整
            if not drone.communication_status:
                base_time += 60
        
        # 基于位置调整（如果有GPS数据）
        if location:
            # 这里可以集成地图API计算实际距离
            # 简化处理：假设距离影响
            base_time += 30  # 假设需要额外30秒到达
        
        return base_time
    
    def _generate_response_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """生成响应建议"""
        recommendations = []
        
        # 基于摔倒检测结果
        if results.get('fall_detected', False):
            recommendations.extend([
                "立即接近患者确认状态",
                "检查患者意识和呼吸",
                "避免移动患者直到确认无脊椎损伤"
            ])
        
        # 基于医疗分析结果
        medical_result = results.get('medical_analysis')
        if medical_result and medical_result.emergency_level >= 3:
            recommendations.extend([
                "启动紧急医疗协议",
                "联系最近的医疗机构",
                "准备提供远程医疗指导"
            ])
        
        # 基于无人机状态
        drone_id = results.get('drone_id', 'default')
        if drone_id in self.drone_status:
            drone = self.drone_status[drone_id]
            if drone.battery_level < 30:
                recommendations.append("无人机电量不足，考虑更换或充电")
        
        return recommendations
    
    def _process_event_queue(self):
        """处理事件队列"""
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                self._handle_emergency_event(event)
            except queue.Empty:
                break
            except Exception as e:
                self.logger.error(f"事件处理失败: {e}")
    
    def _handle_emergency_event(self, event: EmergencyEvent):
        """处理紧急事件"""
        try:
            # 添加到活跃事件
            self.active_events[event.event_id] = event
            
            # 记录到历史
            self.event_history.append(event)
            
            # 执行响应行动
            self._execute_response_actions(event)
            
            # 触发回调
            self._trigger_callbacks(event)
            
            self.logger.info(f"处理紧急事件: {event.event_id} - {event.emergency_type.value}")
            
        except Exception as e:
            self.logger.error(f"紧急事件处理失败: {e}")
    
    def _execute_response_actions(self, event: EmergencyEvent):
        """执行响应行动"""
        for action in event.response_actions or []:
            try:
                self._execute_single_action(action, event)
            except Exception as e:
                self.logger.error(f"响应行动执行失败 '{action}': {e}")
    
    def _execute_single_action(self, action: str, event: EmergencyEvent):
        """执行单个响应行动"""
        # 这里可以集成实际的无人机控制API
        self.logger.info(f"执行响应行动: {action}")
        
        # 示例行动处理
        if "联系急救服务" in action:
            self._contact_emergency_services(event)
        elif "接近患者" in action:
            self._approach_patient(event)
        elif "记录现场" in action:
            self._record_scene(event)
    
    def _contact_emergency_services(self, event: EmergencyEvent):
        """联系急救服务"""
        # 实际实现中会调用急救服务API
        emergency_data = {
            'event_id': event.event_id,
            'emergency_type': event.emergency_type.value,
            'location': event.location,
            'timestamp': event.timestamp,
            'severity': event.response_level.value
        }
        
        self.logger.critical(f"紧急呼叫: {json.dumps(emergency_data, indent=2)}")
    
    def _approach_patient(self, event: EmergencyEvent):
        """接近患者"""
        # 实际实现中会控制无人机移动
        self.logger.info(f"无人机接近患者位置: {event.location}")
    
    def _record_scene(self, event: EmergencyEvent):
        """记录现场"""
        # 实际实现中会保存视频和图像
        self.logger.info(f"记录紧急现场: {event.event_id}")
    
    def _trigger_callbacks(self, event: EmergencyEvent):
        """触发回调函数"""
        callbacks = self.emergency_callbacks if event.response_level in [
            ResponseLevel.CRITICAL, ResponseLevel.IMMEDIATE
        ] else self.alert_callbacks
        
        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"回调函数执行失败: {e}")
    
    def _check_active_events(self):
        """检查活跃事件状态"""
        current_time = time.time()
        completed_events = []
        
        for event_id, event in self.active_events.items():
            # 检查事件是否超时
            if (current_time - event.timestamp > 
                self.config['response']['max_response_time']):
                self.logger.warning(f"事件超时: {event_id}")
                completed_events.append(event_id)
        
        # 移除完成的事件
        for event_id in completed_events:
            del self.active_events[event_id]
    
    def _update_drone_status(self):
        """更新无人机状态"""
        # 实际实现中会从无人机获取状态
        current_time = time.time()
        
        for drone_id in self.drone_status:
            # 检查通信超时
            if (current_time - self.drone_status[drone_id].last_update > 30):
                self.drone_status[drone_id].communication_status = False
    
    def register_emergency_callback(self, callback: Callable[[EmergencyEvent], None]):
        """注册紧急事件回调"""
        self.emergency_callbacks.append(callback)
    
    def register_alert_callback(self, callback: Callable[[EmergencyEvent], None]):
        """注册警报事件回调"""
        self.alert_callbacks.append(callback)
    
    def update_drone_status(self, drone_id: str, status_data: Dict[str, Any]):
        """更新无人机状态"""
        self.drone_status[drone_id] = DroneStatus(
            drone_id=drone_id,
            position=status_data.get('position', (0, 0, 0)),
            battery_level=status_data.get('battery_level', 100),
            camera_status=status_data.get('camera_status', True),
            communication_status=status_data.get('communication_status', True),
            mission_status=status_data.get('mission_status', 'idle'),
            last_update=time.time()
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'monitoring_active': self.monitoring_active,
            'active_events_count': len(self.active_events),
            'total_events_processed': len(self.event_history),
            'drone_count': len(self.drone_status),
            'system_uptime': time.time() - (self.event_history[0].timestamp if self.event_history else time.time()),
            'active_events': [
                {
                    'event_id': event.event_id,
                    'type': event.emergency_type.value,
                    'level': event.response_level.value,
                    'duration': time.time() - event.timestamp
                }
                for event in self.active_events.values()
            ]
        }
    
    def get_emergency_statistics(self, time_window: int = 3600) -> Dict[str, Any]:
        """获取紧急情况统计"""
        current_time = time.time()
        recent_events = [
            event for event in self.event_history
            if current_time - event.timestamp <= time_window
        ]
        
        if not recent_events:
            return {'message': '指定时间窗口内无紧急事件'}
        
        # 统计分析
        emergency_types = {}
        response_levels = {}
        
        for event in recent_events:
            emergency_types[event.emergency_type.value] = emergency_types.get(event.emergency_type.value, 0) + 1
            response_levels[event.response_level.value] = response_levels.get(event.response_level.value, 0) + 1
        
        return {
            'time_window': time_window,
            'total_events': len(recent_events),
            'emergency_types': emergency_types,
            'response_levels': response_levels,
            'average_response_time': np.mean([
                event.estimated_response_time for event in recent_events
                if event.estimated_response_time
            ]) if recent_events else 0
        }


# 使用示例
if __name__ == "__main__":
    # 创建紧急响应系统
    emergency_system = EmergencyResponseSystem()
    
    # 注册回调函数
    def emergency_callback(event: EmergencyEvent):
        print(f"紧急事件: {event.emergency_type.value} - 级别: {event.response_level.value}")
    
    emergency_system.register_emergency_callback(emergency_callback)
    
    # 启动监控
    emergency_system.start_monitoring()
    
    # 模拟处理视频帧
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 更新无人机状态
    emergency_system.update_drone_status("drone_001", {
        'position': (10.0, 20.0, 5.0),
        'battery_level': 85,
        'camera_status': True,
        'communication_status': True,
        'mission_status': 'patrolling'
    })
    
    # 处理帧
    result = emergency_system.process_frame(
        test_frame, 
        drone_id="drone_001",
        gps_location=(39.9042, 116.4074)  # 北京坐标示例
    )
    
    print("处理结果:")
    print(f"摔倒检测: {result['fall_detected']}")
    print(f"紧急事件数: {len(result['emergency_events'])}")
    print(f"建议数: {len(result['recommendations'])}")
    
    # 获取系统状态
    status = emergency_system.get_system_status()
    print(f"\n系统状态: {status}")
    
    # 停止监控
    time.sleep(2)
    emergency_system.stop_monitoring()