#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USB摄像头医疗检测系统
专门用于通过USB摄像头进行实时医疗监控和紧急响应
集成面部生理分析、摔倒检测和紧急响应功能
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
import threading
import queue
import json
from dataclasses import dataclass, asdict
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation

from .medical_facial_analyzer import MedicalFacialAnalyzer, FacialAnalysisResult, HealthStatus, FacialSymptom
from .emergency_response_system import EmergencyResponseSystem, EmergencyEvent, ResponseLevel
from .image_quality_enhancer import ImageQualityEnhancer

@dataclass
class CameraConfig:
    """摄像头配置"""
    camera_id: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    auto_focus: bool = True
    exposure: int = -1  # 自动曝光
    brightness: int = 128
    contrast: int = 128

@dataclass
class MonitoringSession:
    """监控会话数据"""
    session_id: str
    start_time: float
    patient_id: Optional[str] = None
    location: Optional[str] = None
    total_frames: int = 0
    emergency_events: int = 0
    average_health_score: float = 0.0
    session_notes: str = ""

class USBMedicalCameraSystem:
    """USB摄像头医疗检测系统"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.medical_analyzer = MedicalFacialAnalyzer()
        self.emergency_system = EmergencyResponseSystem()
        self.quality_enhancer = ImageQualityEnhancer()
        
        # 摄像头相关
        self.camera = None
        self.camera_config = CameraConfig()
        self.is_recording = False
        self.recording_thread = None
        
        # 数据存储
        self.frame_queue = queue.Queue(maxsize=30)
        self.analysis_results = []
        self.current_session = None
        
        # GUI相关
        self.root = None
        self.gui_components = {}
        self.plots = {}
        
        # 实时数据
        self.real_time_data = {
            'timestamps': [],
            'health_scores': [],
            'heart_rates': [],
            'risk_scores': [],
            'emergency_levels': []
        }
        
        # 注册紧急事件回调
        self.emergency_system.register_emergency_callback(self._handle_emergency_alert)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            # 摄像头设置
            'camera': {
                'default_id': 0,
                'resolution': (640, 480),
                'fps': 30,
                'buffer_size': 1
            },
            
            # 分析设置
            'analysis': {
                'interval': 1.0,  # 秒
                'enable_quality_enhancement': True,
                'enable_emergency_detection': True,
                'save_analysis_results': True
            },
            
            # GUI设置
            'gui': {
                'window_title': "YOLOS医疗监控系统",
                'window_size': (1200, 800),
                'update_interval': 100,  # ms
                'max_plot_points': 100
            },
            
            # 存储设置
            'storage': {
                'save_frames': False,
                'save_directory': './medical_monitoring_data',
                'max_session_duration': 3600  # 秒
            },
            
            # 警报设置
            'alerts': {
                'enable_sound': True,
                'enable_popup': True,
                'auto_save_emergency': True
            }
        }
    
    def initialize_camera(self, camera_id: Optional[int] = None) -> bool:
        """初始化摄像头"""
        try:
            if camera_id is None:
                camera_id = self.config['camera']['default_id']
            
            self.camera = cv2.VideoCapture(camera_id)
            
            if not self.camera.isOpened():
                self.logger.error(f"无法打开摄像头 {camera_id}")
                return False
            
            # 设置摄像头参数
            width, height = self.config['camera']['resolution']
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.camera.set(cv2.CAP_PROP_FPS, self.config['camera']['fps'])
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, self.config['camera']['buffer_size'])
            
            # 验证设置
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"摄像头初始化成功: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            self.camera_config.camera_id = camera_id
            self.camera_config.width = actual_width
            self.camera_config.height = actual_height
            self.camera_config.fps = actual_fps
            
            return True
            
        except Exception as e:
            self.logger.error(f"摄像头初始化失败: {e}")
            return False
    
    def start_monitoring(self, patient_id: Optional[str] = None, 
                        location: Optional[str] = None) -> bool:
        """开始监控"""
        try:
            if self.camera is None:
                if not self.initialize_camera():
                    return False
            
            # 创建新的监控会话
            session_id = f"session_{int(time.time())}"
            self.current_session = MonitoringSession(
                session_id=session_id,
                start_time=time.time(),
                patient_id=patient_id,
                location=location
            )
            
            # 启动紧急响应系统
            self.emergency_system.start_monitoring()
            
            # 启动录制线程
            self.is_recording = True
            self.recording_thread = threading.Thread(target=self._recording_loop)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
            self.logger.info(f"开始监控会话: {session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"启动监控失败: {e}")
            return False
    
    def stop_monitoring(self):
        """停止监控"""
        try:
            self.is_recording = False
            
            if self.recording_thread:
                self.recording_thread.join(timeout=5)
            
            self.emergency_system.stop_monitoring()
            
            if self.camera:
                self.camera.release()
                self.camera = None
            
            # 保存会话数据
            if self.current_session:
                self._save_session_data()
            
            self.logger.info("监控已停止")
            
        except Exception as e:
            self.logger.error(f"停止监控失败: {e}")
    
    def _recording_loop(self):
        """录制循环"""
        last_analysis_time = 0
        analysis_interval = self.config['analysis']['interval']
        
        while self.is_recording:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    self.logger.warning("无法读取摄像头帧")
                    continue
                
                current_time = time.time()
                
                # 图像质量增强
                if self.config['analysis']['enable_quality_enhancement']:
                    is_acceptable, quality_metrics = self.quality_enhancer.is_image_acceptable(frame)
                    if not is_acceptable:
                        frame = self.quality_enhancer.enhance_image(frame, quality_metrics)
                
                # 添加到帧队列
                if not self.frame_queue.full():
                    self.frame_queue.put((frame.copy(), current_time))
                
                # 定期进行医疗分析
                if current_time - last_analysis_time >= analysis_interval:
                    self._perform_medical_analysis(frame, current_time)
                    last_analysis_time = current_time
                
                # 更新会话统计
                if self.current_session:
                    self.current_session.total_frames += 1
                
                time.sleep(0.01)  # 10ms延迟
                
            except Exception as e:
                self.logger.error(f"录制循环错误: {e}")
    
    def _perform_medical_analysis(self, frame: np.ndarray, timestamp: float):
        """执行医疗分析"""
        try:
            # 获取前一帧用于比较
            previous_frame = None
            if len(self.analysis_results) > 0:
                previous_frame = self.analysis_results[-1].get('frame')
            
            # 医疗面部分析
            medical_result = self.medical_analyzer.analyze_facial_health(frame, previous_frame)
            
            # 紧急响应处理
            emergency_result = None
            if self.config['analysis']['enable_emergency_detection']:
                emergency_result = self.emergency_system.process_frame(
                    frame, 
                    drone_id="usb_camera",
                    gps_location=None
                )
            
            # 整合分析结果
            analysis_data = {
                'timestamp': timestamp,
                'frame': frame.copy() if self.config['storage']['save_frames'] else None,
                'medical_analysis': medical_result,
                'emergency_analysis': emergency_result,
                'session_id': self.current_session.session_id if self.current_session else None
            }
            
            # 保存结果
            self.analysis_results.append(analysis_data)
            
            # 更新实时数据
            self._update_real_time_data(analysis_data)
            
            # 更新会话统计
            if self.current_session:
                self._update_session_statistics(analysis_data)
            
            # 保存分析结果
            if self.config['analysis']['save_analysis_results']:
                self._save_analysis_result(analysis_data)
            
        except Exception as e:
            self.logger.error(f"医疗分析失败: {e}")
    
    def _update_real_time_data(self, analysis_data: Dict[str, Any]):
        """更新实时数据"""
        try:
            medical_result = analysis_data['medical_analysis']
            
            # 添加新数据点
            self.real_time_data['timestamps'].append(analysis_data['timestamp'])
            self.real_time_data['health_scores'].append(
                1.0 - (medical_result.risk_score / 100.0)  # 转换为健康分数
            )
            self.real_time_data['risk_scores'].append(medical_result.risk_score)
            self.real_time_data['emergency_levels'].append(medical_result.emergency_level)
            
            # 添加心率数据
            heart_rate = 0
            if medical_result.vital_signs.heart_rate:
                heart_rate = medical_result.vital_signs.heart_rate
            self.real_time_data['heart_rates'].append(heart_rate)
            
            # 限制数据点数量
            max_points = self.config['gui']['max_plot_points']
            for key in self.real_time_data:
                if len(self.real_time_data[key]) > max_points:
                    self.real_time_data[key] = self.real_time_data[key][-max_points:]
            
        except Exception as e:
            self.logger.error(f"实时数据更新失败: {e}")
    
    def _update_session_statistics(self, analysis_data: Dict[str, Any]):
        """更新会话统计"""
        try:
            if not self.current_session:
                return
            
            medical_result = analysis_data['medical_analysis']
            emergency_result = analysis_data['emergency_analysis']
            
            # 更新紧急事件计数
            if emergency_result and len(emergency_result.get('emergency_events', [])) > 0:
                self.current_session.emergency_events += len(emergency_result['emergency_events'])
            
            # 更新平均健康分数
            health_score = 1.0 - (medical_result.risk_score / 100.0)
            if self.current_session.average_health_score == 0:
                self.current_session.average_health_score = health_score
            else:
                # 移动平均
                alpha = 0.1
                self.current_session.average_health_score = (
                    alpha * health_score + 
                    (1 - alpha) * self.current_session.average_health_score
                )
            
        except Exception as e:
            self.logger.error(f"会话统计更新失败: {e}")
    
    def _handle_emergency_alert(self, event: EmergencyEvent):
        """处理紧急警报"""
        try:
            self.logger.critical(f"紧急警报: {event.emergency_type.value}")
            
            # 弹窗警报
            if self.config['alerts']['enable_popup'] and self.root:
                self._show_emergency_popup(event)
            
            # 声音警报
            if self.config['alerts']['enable_sound']:
                self._play_alert_sound(event.response_level)
            
            # 自动保存紧急数据
            if self.config['alerts']['auto_save_emergency']:
                self._save_emergency_data(event)
            
        except Exception as e:
            self.logger.error(f"紧急警报处理失败: {e}")
    
    def _show_emergency_popup(self, event: EmergencyEvent):
        """显示紧急弹窗"""
        def show_popup():
            title = f"紧急警报 - {event.response_level.value.upper()}"
            message = f"""
紧急类型: {event.emergency_type.value}
响应级别: {event.response_level.value}
时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event.timestamp))}

建议行动:
{chr(10).join(event.response_actions or [])}
"""
            messagebox.showerror(title, message)
        
        # 在主线程中显示弹窗
        if self.root:
            self.root.after(0, show_popup)
    
    def _play_alert_sound(self, response_level: ResponseLevel):
        """播放警报声音"""
        try:
            # 这里可以集成音频播放库
            # 简化实现：打印声音提示
            if response_level in [ResponseLevel.CRITICAL, ResponseLevel.IMMEDIATE]:
                print("\a" * 3)  # 系统蜂鸣声
            else:
                print("\a")
        except Exception as e:
            self.logger.error(f"警报声音播放失败: {e}")
    
    def _save_emergency_data(self, event: EmergencyEvent):
        """保存紧急数据"""
        try:
            import os
            
            # 创建保存目录
            save_dir = self.config['storage']['save_directory']
            os.makedirs(save_dir, exist_ok=True)
            
            # 保存事件数据
            event_file = os.path.join(save_dir, f"emergency_{event.event_id}.json")
            with open(event_file, 'w', encoding='utf-8') as f:
                # 转换为可序列化的格式
                event_data = {
                    'event_id': event.event_id,
                    'timestamp': event.timestamp,
                    'emergency_type': event.emergency_type.value,
                    'response_level': event.response_level.value,
                    'location': event.location,
                    'response_actions': event.response_actions,
                    'estimated_response_time': event.estimated_response_time
                }
                
                # 添加医疗分析数据
                if event.medical_analysis:
                    event_data['medical_analysis'] = {
                        'health_status': event.medical_analysis.health_status.value,
                        'symptoms': [s.value for s in event.medical_analysis.symptoms],
                        'risk_score': event.medical_analysis.risk_score,
                        'emergency_level': event.medical_analysis.emergency_level,
                        'confidence': event.medical_analysis.confidence
                    }
                
                json.dump(event_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"紧急数据已保存: {event_file}")
            
        except Exception as e:
            self.logger.error(f"紧急数据保存失败: {e}")
    
    def _save_analysis_result(self, analysis_data: Dict[str, Any]):
        """保存分析结果"""
        try:
            # 简化保存：只保存关键数据
            result_summary = {
                'timestamp': analysis_data['timestamp'],
                'session_id': analysis_data['session_id'],
                'health_status': analysis_data['medical_analysis'].health_status.value,
                'risk_score': analysis_data['medical_analysis'].risk_score,
                'emergency_level': analysis_data['medical_analysis'].emergency_level,
                'symptoms': [s.value for s in analysis_data['medical_analysis'].symptoms]
            }
            
            # 这里可以保存到数据库或文件
            # 简化实现：添加到内存列表
            
        except Exception as e:
            self.logger.error(f"分析结果保存失败: {e}")
    
    def _save_session_data(self):
        """保存会话数据"""
        try:
            if not self.current_session:
                return
            
            import os
            
            save_dir = self.config['storage']['save_directory']
            os.makedirs(save_dir, exist_ok=True)
            
            session_file = os.path.join(save_dir, f"session_{self.current_session.session_id}.json")
            
            session_data = {
                'session_id': self.current_session.session_id,
                'start_time': self.current_session.start_time,
                'end_time': time.time(),
                'duration': time.time() - self.current_session.start_time,
                'patient_id': self.current_session.patient_id,
                'location': self.current_session.location,
                'total_frames': self.current_session.total_frames,
                'emergency_events': self.current_session.emergency_events,
                'average_health_score': self.current_session.average_health_score,
                'session_notes': self.current_session.session_notes
            }
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"会话数据已保存: {session_file}")
            
        except Exception as e:
            self.logger.error(f"会话数据保存失败: {e}")
    
    def create_gui(self):
        """创建GUI界面"""
        try:
            self.root = tk.Tk()
            self.root.title(self.config['gui']['window_title'])
            self.root.geometry(f"{self.config['gui']['window_size'][0]}x{self.config['gui']['window_size'][1]}")
            
            # 创建主框架
            main_frame = ttk.Frame(self.root)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # 创建控制面板
            self._create_control_panel(main_frame)
            
            # 创建视频显示区域
            self._create_video_display(main_frame)
            
            # 创建数据显示区域
            self._create_data_display(main_frame)
            
            # 创建图表区域
            self._create_charts(main_frame)
            
            # 启动GUI更新循环
            self._start_gui_updates()
            
            return self.root
            
        except Exception as e:
            self.logger.error(f"GUI创建失败: {e}")
            return None
    
    def _create_control_panel(self, parent):
        """创建控制面板"""
        control_frame = ttk.LabelFrame(parent, text="控制面板")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 按钮框架
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 开始/停止按钮
        self.gui_components['start_button'] = ttk.Button(
            button_frame, text="开始监控", command=self._on_start_monitoring
        )
        self.gui_components['start_button'].pack(side=tk.LEFT, padx=(0, 10))
        
        self.gui_components['stop_button'] = ttk.Button(
            button_frame, text="停止监控", command=self._on_stop_monitoring, state=tk.DISABLED
        )
        self.gui_components['stop_button'].pack(side=tk.LEFT, padx=(0, 10))
        
        # 患者ID输入
        ttk.Label(button_frame, text="患者ID:").pack(side=tk.LEFT, padx=(20, 5))
        self.gui_components['patient_id_entry'] = ttk.Entry(button_frame, width=15)
        self.gui_components['patient_id_entry'].pack(side=tk.LEFT, padx=(0, 10))
        
        # 位置输入
        ttk.Label(button_frame, text="位置:").pack(side=tk.LEFT, padx=(10, 5))
        self.gui_components['location_entry'] = ttk.Entry(button_frame, width=20)
        self.gui_components['location_entry'].pack(side=tk.LEFT)
    
    def _create_video_display(self, parent):
        """创建视频显示区域"""
        video_frame = ttk.LabelFrame(parent, text="实时视频")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # 视频标签
        self.gui_components['video_label'] = ttk.Label(video_frame, text="摄像头未启动")
        self.gui_components['video_label'].pack(expand=True)
    
    def _create_data_display(self, parent):
        """创建数据显示区域"""
        data_frame = ttk.LabelFrame(parent, text="实时数据")
        data_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # 创建数据标签
        self.gui_components['health_status_label'] = ttk.Label(data_frame, text="健康状态: 未知")
        self.gui_components['health_status_label'].pack(anchor=tk.W, padx=10, pady=5)
        
        self.gui_components['risk_score_label'] = ttk.Label(data_frame, text="风险分数: 0")
        self.gui_components['risk_score_label'].pack(anchor=tk.W, padx=10, pady=5)
        
        self.gui_components['emergency_level_label'] = ttk.Label(data_frame, text="紧急等级: 1")
        self.gui_components['emergency_level_label'].pack(anchor=tk.W, padx=10, pady=5)
        
        self.gui_components['heart_rate_label'] = ttk.Label(data_frame, text="心率: --")
        self.gui_components['heart_rate_label'].pack(anchor=tk.W, padx=10, pady=5)
        
        self.gui_components['symptoms_label'] = ttk.Label(data_frame, text="症状: 无")
        self.gui_components['symptoms_label'].pack(anchor=tk.W, padx=10, pady=5)
        
        # 会话信息
        ttk.Separator(data_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=10)
        
        self.gui_components['session_label'] = ttk.Label(data_frame, text="会话: 未开始")
        self.gui_components['session_label'].pack(anchor=tk.W, padx=10, pady=5)
        
        self.gui_components['frames_label'] = ttk.Label(data_frame, text="帧数: 0")
        self.gui_components['frames_label'].pack(anchor=tk.W, padx=10, pady=5)
        
        self.gui_components['events_label'] = ttk.Label(data_frame, text="紧急事件: 0")
        self.gui_components['events_label'].pack(anchor=tk.W, padx=10, pady=5)
    
    def _create_charts(self, parent):
        """创建图表区域"""
        # 这里可以添加matplotlib图表
        # 简化实现：只创建占位符
        chart_frame = ttk.LabelFrame(parent, text="健康趋势图表")
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        chart_label = ttk.Label(chart_frame, text="图表区域 (可扩展)")
        chart_label.pack(expand=True)
    
    def _start_gui_updates(self):
        """启动GUI更新循环"""
        def update_gui():
            try:
                self._update_video_display()
                self._update_data_display()
                
                # 继续更新
                if self.root:
                    self.root.after(self.config['gui']['update_interval'], update_gui)
            except Exception as e:
                self.logger.error(f"GUI更新失败: {e}")
        
        update_gui()
    
    def _update_video_display(self):
        """更新视频显示"""
        try:
            if not self.frame_queue.empty():
                frame, timestamp = self.frame_queue.get()
                
                # 调整帧大小用于显示
                display_frame = cv2.resize(frame, (400, 300))
                
                # 转换为RGB格式
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
                # 转换为PIL图像
                from PIL import Image, ImageTk
                pil_image = Image.fromarray(display_frame)
                photo = ImageTk.PhotoImage(pil_image)
                
                # 更新标签
                self.gui_components['video_label'].configure(image=photo)
                self.gui_components['video_label'].image = photo  # 保持引用
        except Exception as e:
            self.logger.error(f"视频显示更新失败: {e}")
    
    def _update_data_display(self):
        """更新数据显示"""
        try:
            if self.analysis_results:
                latest_result = self.analysis_results[-1]
                medical_result = latest_result['medical_analysis']
                
                # 更新健康数据
                self.gui_components['health_status_label'].configure(
                    text=f"健康状态: {medical_result.health_status.value}"
                )
                self.gui_components['risk_score_label'].configure(
                    text=f"风险分数: {medical_result.risk_score:.1f}"
                )
                self.gui_components['emergency_level_label'].configure(
                    text=f"紧急等级: {medical_result.emergency_level}"
                )
                
                # 更新生命体征
                heart_rate = medical_result.vital_signs.heart_rate or "--"
                self.gui_components['heart_rate_label'].configure(
                    text=f"心率: {heart_rate}"
                )
                
                # 更新症状
                symptoms = [s.value for s in medical_result.symptoms] or ["无"]
                symptoms_text = ", ".join(symptoms[:3])  # 只显示前3个症状
                if len(symptoms) > 3:
                    symptoms_text += "..."
                self.gui_components['symptoms_label'].configure(
                    text=f"症状: {symptoms_text}"
                )
            
            # 更新会话信息
            if self.current_session:
                duration = time.time() - self.current_session.start_time
                self.gui_components['session_label'].configure(
                    text=f"会话: {self.current_session.session_id} ({duration:.0f}s)"
                )
                self.gui_components['frames_label'].configure(
                    text=f"帧数: {self.current_session.total_frames}"
                )
                self.gui_components['events_label'].configure(
                    text=f"紧急事件: {self.current_session.emergency_events}"
                )
        except Exception as e:
            self.logger.error(f"数据显示更新失败: {e}")
    
    def _on_start_monitoring(self):
        """开始监控按钮回调"""
        try:
            patient_id = self.gui_components['patient_id_entry'].get().strip()
            location = self.gui_components['location_entry'].get().strip()
            
            if self.start_monitoring(
                patient_id=patient_id if patient_id else None,
                location=location if location else None
            ):
                self.gui_components['start_button'].configure(state=tk.DISABLED)
                self.gui_components['stop_button'].configure(state=tk.NORMAL)
                messagebox.showinfo("成功", "监控已启动")
            else:
                messagebox.showerror("错误", "监控启动失败")
        except Exception as e:
            messagebox.showerror("错误", f"启动失败: {e}")
    
    def _on_stop_monitoring(self):
        """停止监控按钮回调"""
        try:
            self.stop_monitoring()
            self.gui_components['start_button'].configure(state=tk.NORMAL)
            self.gui_components['stop_button'].configure(state=tk.DISABLED)
            messagebox.showinfo("成功", "监控已停止")
        except Exception as e:
            messagebox.showerror("错误", f"停止失败: {e}")
    
    def run_gui(self):
        """运行GUI"""
        try:
            if self.root is None:
                self.create_gui()
            
            if self.root:
                self.root.mainloop()
        except Exception as e:
            self.logger.error(f"GUI运行失败: {e}")
    
    def get_monitoring_report(self) -> Dict[str, Any]:
        """获取监控报告"""
        try:
            if not self.current_session:
                return {'error': '没有活跃的监控会话'}
            
            # 计算统计数据
            total_analyses = len(self.analysis_results)
            emergency_events = self.current_session.emergency_events
            
            # 健康趋势分析
            health_trend = "稳定"
            if len(self.real_time_data['health_scores']) >= 10:
                recent_scores = self.real_time_data['health_scores'][-10:]
                early_scores = self.real_time_data['health_scores'][-20:-10] if len(self.real_time_data['health_scores']) >= 20 else recent_scores
                
                recent_avg = np.mean(recent_scores)
                early_avg = np.mean(early_scores)
                
                if recent_avg > early_avg + 0.1:
                    health_trend = "改善"
                elif recent_avg < early_avg - 0.1:
                    health_trend = "恶化"
            
            report = {
                'session_info': {
                    'session_id': self.current_session.session_id,
                    'patient_id': self.current_session.patient_id,
                    'location': self.current_session.location,
                    'start_time': self.current_session.start_time,
                    'duration': time.time() - self.current_session.start_time,
                    'total_frames': self.current_session.total_frames
                },
                'health_statistics': {
                    'total_analyses': total_analyses,
                    'emergency_events': emergency_events,
                    'average_health_score': self.current_session.average_health_score,
                    'health_trend': health_trend,
                    'current_health_status': self.analysis_results[-1]['medical_analysis'].health_status.value if self.analysis_results else 'unknown'
                },
                'system_performance': {
                    'analysis_rate': total_analyses / (time.time() - self.current_session.start_time) if total_analyses > 0 else 0,
                    'emergency_rate': emergency_events / total_analyses if total_analyses > 0 else 0
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"监控报告生成失败: {e}")
            return {'error': str(e)}


# 使用示例和主程序入口
def main():
    """主程序入口"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 创建医疗摄像头系统
        medical_system = USBMedicalCameraSystem()
        
        # 创建并运行GUI
        medical_system.create_gui()
        medical_system.run_gui()
        
    except Exception as e:
        logging.error(f"程序运行失败: {e}")
    finally:
        # 清理资源
        if 'medical_system' in locals():
            medical_system.stop_monitoring()

if __name__ == "__main__":
    main()