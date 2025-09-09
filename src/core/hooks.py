#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOS Hook系统
借鉴MMDetection的Hook设计，提供灵活的训练和推理扩展机制
"""

import time
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from .registry import register_hook
from ..utils.logging_manager import LoggingManager


class BaseHook(ABC):
    """Hook基类"""
    
    def __init__(self, priority: int = 50):
        self.priority = priority
        self.logger = LoggingManager().get_logger(self.__class__.__name__)
    
    def before_detection(self, frame_info: Dict[str, Any]):
        """检测前回调"""
        pass
    
    def after_detection(self, results: List[Any], frame_info: Dict[str, Any]):
        """检测后回调"""
        pass
    
    def before_training_epoch(self, epoch: int, trainer_state: Dict[str, Any]):
        """训练轮次开始前回调"""
        pass
    
    def after_training_epoch(self, epoch: int, trainer_state: Dict[str, Any]):
        """训练轮次结束后回调"""
        pass
    
    def before_training_step(self, step: int, batch_data: Any):
        """训练步骤开始前回调"""
        pass
    
    def after_training_step(self, step: int, loss: float, metrics: Dict[str, float]):
        """训练步骤结束后回调"""
        pass
    
    def on_exception(self, exception: Exception, context: Dict[str, Any]):
        """异常处理回调"""
        pass


@register_hook('medical_monitoring')
class MedicalMonitoringHook(BaseHook):
    """医疗监控Hook"""
    
    def __init__(self, alert_config: Dict[str, Any], priority: int = 80):
        super().__init__(priority)
        self.alert_config = alert_config
        
        # 初始化医疗分析模块
        self.fall_detector = self._init_fall_detector()
        self.medication_tracker = self._init_medication_tracker()
        self.vital_monitor = self._init_vital_monitor()
        self.alert_system = self._init_alert_system()
        
        # 状态跟踪
        self.last_alert_time = {}
        self.alert_cooldown = 30  # 30秒冷却时间
    
    def _init_fall_detector(self):
        """初始化跌倒检测器"""
        from ..medical.fall_detector import FallDetector
        return FallDetector(
            sensitivity=self.alert_config.get('fall_sensitivity', 0.8),
            min_confidence=self.alert_config.get('fall_confidence', 0.7)
        )
    
    def _init_medication_tracker(self):
        """初始化药物跟踪器"""
        from ..medical.medication_tracker import MedicationTracker
        return MedicationTracker(
            schedule_path=self.alert_config.get('medication_schedule'),
            reminder_enabled=self.alert_config.get('medication_reminder', True)
        )
    
    def _init_vital_monitor(self):
        """初始化生命体征监控器"""
        from ..medical.vital_monitor import VitalSignsMonitor
        return VitalSignsMonitor(
            face_analysis=self.alert_config.get('face_analysis', True),
            heart_rate_detection=self.alert_config.get('heart_rate_detection', False)
        )
    
    def _init_alert_system(self):
        """初始化报警系统"""
        from ..medical.alert_system import AlertSystem
        return AlertSystem(
            emergency_contacts=self.alert_config.get('emergency_contacts', []),
            sms_enabled=self.alert_config.get('sms_enabled', False),
            email_enabled=self.alert_config.get('email_enabled', True),
            local_alarm=self.alert_config.get('local_alarm', True)
        )
    
    def after_detection(self, results: List[Any], frame_info: Dict[str, Any]):
        """检测后进行医疗分析"""
        current_time = time.time()
        
        try:
            # 跌倒检测
            if self.alert_config.get('fall_detection', True):
                fall_result = self.fall_detector.analyze(results, frame_info)
                if fall_result.is_fall and self._should_send_alert('fall', current_time):
                    self.alert_system.send_emergency_alert(
                        alert_type='fall_detected',
                        message=f"检测到跌倒事件，置信度: {fall_result.confidence:.2f}",
                        location=frame_info.get('location', 'Unknown'),
                        timestamp=current_time,
                        image=frame_info.get('frame')
                    )
                    self.last_alert_time['fall'] = current_time
            
            # 药物服用监控
            if self.alert_config.get('medication_tracking', True):
                medication_result = self.medication_tracker.check_medication(results, current_time)
                if medication_result.missed_dose and self._should_send_alert('medication', current_time):
                    self.alert_system.send_reminder(
                        alert_type='medication_reminder',
                        message=f"服药提醒: {medication_result.medication_name}",
                        scheduled_time=medication_result.scheduled_time,
                        current_time=current_time
                    )
                    self.last_alert_time['medication'] = current_time
            
            # 生命体征监控
            if self.alert_config.get('vital_monitoring', True):
                vital_result = self.vital_monitor.analyze(results, frame_info)
                if vital_result.abnormal and self._should_send_alert('vital', current_time):
                    self.alert_system.send_health_alert(
                        alert_type='vital_signs_abnormal',
                        message=f"生命体征异常: {vital_result.description}",
                        vital_data=vital_result.data,
                        timestamp=current_time
                    )
                    self.last_alert_time['vital'] = current_time
                    
        except Exception as e:
            self.logger.error(f"医疗监控处理失败: {e}")
    
    def _should_send_alert(self, alert_type: str, current_time: float) -> bool:
        """检查是否应该发送报警"""
        last_time = self.last_alert_time.get(alert_type, 0)
        return (current_time - last_time) > self.alert_cooldown


@register_hook('performance_optimization')
class PerformanceOptimizationHook(BaseHook):
    """性能优化Hook"""
    
    def __init__(self, target_fps: float = 30.0, priority: int = 60):
        super().__init__(priority)
        self.target_fps = target_fps
        
        # 性能监控
        self.fps_history = []
        self.inference_times = []
        self.memory_usage = []
        
        # 自适应控制器
        self.fps_controller = AdaptiveFPSController(target_fps)
        self.memory_optimizer = MemoryOptimizer()
        self.model_switcher = DynamicModelSwitcher()
        
        # 统计信息
        self.frame_count = 0
        self.start_time = time.time()
    
    def before_detection(self, frame_info: Dict[str, Any]):
        """检测前性能优化"""
        # 记录开始时间
        frame_info['inference_start_time'] = time.time()
        
        # 检查系统负载
        system_load = self._get_system_load()
        
        # 动态调整检测参数
        if system_load > 0.8:
            self.fps_controller.reduce_fps()
            self.model_switcher.switch_to_lighter_model()
            self.logger.info(f"系统负载过高 ({system_load:.2f})，降低检测频率")
        elif system_load < 0.3:
            self.fps_controller.increase_fps()
            self.model_switcher.switch_to_better_model()
    
    def after_detection(self, results: List[Any], frame_info: Dict[str, Any]):
        """检测后性能统计"""
        # 计算推理时间
        start_time = frame_info.get('inference_start_time', time.time())
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # 更新FPS统计
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        if elapsed_time > 0:
            current_fps = self.frame_count / elapsed_time
            self.fps_history.append(current_fps)
            
            # 保持历史记录长度
            if len(self.fps_history) > 100:
                self.fps_history.pop(0)
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)
        
        # 性能优化决策
        self.fps_controller.update_performance(inference_time, current_fps)
        
        # 内存优化
        if self.frame_count % 50 == 0:  # 每50帧检查一次内存
            self.memory_optimizer.cleanup_if_needed()
        
        # 性能日志
        if self.frame_count % 100 == 0:
            avg_fps = sum(self.fps_history[-10:]) / min(10, len(self.fps_history))
            avg_inference = sum(self.inference_times[-10:]) / min(10, len(self.inference_times))
            self.logger.info(f"性能统计 - FPS: {avg_fps:.1f}, 推理时间: {avg_inference*1000:.1f}ms")
    
    def _get_system_load(self) -> float:
        """获取系统负载"""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1) / 100.0
        except ImportError:
            return 0.5  # 默认中等负载


@register_hook('logging')
class LoggingHook(BaseHook):
    """日志记录Hook"""
    
    def __init__(self, log_interval: int = 50, save_results: bool = False, priority: int = 30):
        super().__init__(priority)
        self.log_interval = log_interval
        self.save_results = save_results
        
        # 统计信息
        self.detection_count = 0
        self.total_objects = 0
        self.class_counts = {}
        
        # 结果保存
        if save_results:
            self.results_dir = Path('logs/detection_results')
            self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def after_detection(self, results: List[Any], frame_info: Dict[str, Any]):
        """记录检测结果"""
        self.detection_count += 1
        self.total_objects += len(results)
        
        # 统计类别
        for result in results:
            class_name = getattr(result, 'class_name', 'unknown')
            self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1
        
        # 定期日志输出
        if self.detection_count % self.log_interval == 0:
            avg_objects = self.total_objects / self.detection_count
            self.logger.info(f"检测统计 - 总帧数: {self.detection_count}, "
                           f"平均目标数: {avg_objects:.1f}, "
                           f"类别分布: {dict(list(self.class_counts.items())[:5])}")
        
        # 保存结果
        if self.save_results:
            self._save_detection_results(results, frame_info)
    
    def _save_detection_results(self, results: List[Any], frame_info: Dict[str, Any]):
        """保存检测结果"""
        timestamp = frame_info.get('timestamp', time.time())
        result_data = {
            'timestamp': timestamp,
            'frame_id': frame_info.get('frame_id', self.detection_count),
            'detections': []
        }
        
        for result in results:
            detection = {
                'class_name': getattr(result, 'class_name', 'unknown'),
                'confidence': getattr(result, 'confidence', 0.0),
                'bbox': getattr(result, 'bbox', None)
            }
            result_data['detections'].append(detection)
        
        # 保存到JSON文件
        result_file = self.results_dir / f"detection_{timestamp:.0f}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)


@register_hook('checkpoint')
class CheckpointHook(BaseHook):
    """检查点保存Hook"""
    
    def __init__(self, save_interval: int = 1000, max_keep: int = 5, priority: int = 40):
        super().__init__(priority)
        self.save_interval = save_interval
        self.max_keep = max_keep
        
        self.checkpoint_dir = Path('checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.step_count = 0
    
    def after_training_step(self, step: int, loss: float, metrics: Dict[str, float]):
        """训练步骤后保存检查点"""
        self.step_count += 1
        
        if self.step_count % self.save_interval == 0:
            self._save_checkpoint(step, loss, metrics)
    
    def _save_checkpoint(self, step: int, loss: float, metrics: Dict[str, float]):
        """保存检查点"""
        checkpoint_data = {
            'step': step,
            'loss': loss,
            'metrics': metrics,
            'timestamp': time.time()
        }
        
        checkpoint_file = self.checkpoint_dir / f"checkpoint_step_{step}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        self.logger.info(f"保存检查点: {checkpoint_file}")
        
        # 清理旧检查点
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """清理旧检查点"""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*.json"))
        if len(checkpoints) > self.max_keep:
            # 按修改时间排序，删除最旧的
            checkpoints.sort(key=lambda x: x.stat().st_mtime)
            for old_checkpoint in checkpoints[:-self.max_keep]:
                old_checkpoint.unlink()
                self.logger.info(f"删除旧检查点: {old_checkpoint}")


# 辅助类
class AdaptiveFPSController:
    """自适应FPS控制器"""
    
    def __init__(self, target_fps: float):
        self.target_fps = target_fps
        self.current_fps = target_fps
        self.adjustment_factor = 0.1
    
    def update_performance(self, inference_time: float, current_fps: float):
        """更新性能并调整FPS"""
        if current_fps < self.target_fps * 0.8:
            self.reduce_fps()
        elif current_fps > self.target_fps * 1.2:
            self.increase_fps()
    
    def reduce_fps(self):
        """降低FPS"""
        self.current_fps = max(5.0, self.current_fps * (1 - self.adjustment_factor))
    
    def increase_fps(self):
        """提高FPS"""
        self.current_fps = min(self.target_fps, self.current_fps * (1 + self.adjustment_factor))


class MemoryOptimizer:
    """内存优化器"""
    
    def __init__(self):
        self.memory_threshold = 0.8  # 80%内存使用率阈值
    
    def cleanup_if_needed(self):
        """如需要则清理内存"""
        try:
            import psutil
            import gc
            
            memory_percent = psutil.virtual_memory().percent / 100.0
            if memory_percent > self.memory_threshold:
                gc.collect()
                # 如果使用CUDA，清理GPU内存
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
        except ImportError:
            pass


class DynamicModelSwitcher:
    """动态模型切换器"""
    
    def __init__(self):
        self.current_model = 's'  # 当前模型大小
        self.model_hierarchy = ['n', 's', 'm', 'l', 'x']
    
    def switch_to_lighter_model(self):
        """切换到更轻量的模型"""
        current_idx = self.model_hierarchy.index(self.current_model)
        if current_idx > 0:
            self.current_model = self.model_hierarchy[current_idx - 1]
    
    def switch_to_better_model(self):
        """切换到更好的模型"""
        current_idx = self.model_hierarchy.index(self.current_model)
        if current_idx < len(self.model_hierarchy) - 1:
            self.current_model = self.model_hierarchy[current_idx + 1]


# Hook管理器
class HookManager:
    """Hook管理器"""
    
    def __init__(self):
        self.hooks: List[BaseHook] = []
    
    def add_hook(self, hook: BaseHook):
        """添加Hook"""
        self.hooks.append(hook)
        # 按优先级排序
        self.hooks.sort(key=lambda x: x.priority, reverse=True)
    
    def remove_hook(self, hook: BaseHook):
        """移除Hook"""
        if hook in self.hooks:
            self.hooks.remove(hook)
    
    def call_before_detection(self, frame_info: Dict[str, Any]):
        """调用检测前Hook"""
        for hook in self.hooks:
            try:
                hook.before_detection(frame_info)
            except Exception as e:
                hook.logger.error(f"Hook {hook.__class__.__name__} before_detection failed: {e}")
    
    def call_after_detection(self, results: List[Any], frame_info: Dict[str, Any]):
        """调用检测后Hook"""
        for hook in self.hooks:
            try:
                hook.after_detection(results, frame_info)
            except Exception as e:
                hook.logger.error(f"Hook {hook.__class__.__name__} after_detection failed: {e}")
    
    def call_before_training_epoch(self, epoch: int, trainer_state: Dict[str, Any]):
        """调用训练轮次开始前Hook"""
        for hook in self.hooks:
            try:
                hook.before_training_epoch(epoch, trainer_state)
            except Exception as e:
                hook.logger.error(f"Hook {hook.__class__.__name__} before_training_epoch failed: {e}")
    
    def call_after_training_epoch(self, epoch: int, trainer_state: Dict[str, Any]):
        """调用训练轮次结束后Hook"""
        for hook in self.hooks:
            try:
                hook.after_training_epoch(epoch, trainer_state)
            except Exception as e:
                hook.logger.error(f"Hook {hook.__class__.__name__} after_training_epoch failed: {e}")
    
    def call_before_training_step(self, step: int, batch_data: Any):
        """调用训练步骤开始前Hook"""
        for hook in self.hooks:
            try:
                hook.before_training_step(step, batch_data)
            except Exception as e:
                hook.logger.error(f"Hook {hook.__class__.__name__} before_training_step failed: {e}")
    
    def call_after_training_step(self, step: int, loss: float, metrics: Dict[str, float]):
        """调用训练步骤结束后Hook"""
        for hook in self.hooks:
            try:
                hook.after_training_step(step, loss, metrics)
            except Exception as e:
                hook.logger.error(f"Hook {hook.__class__.__name__} after_training_step failed: {e}")
    
    def call_on_exception(self, exception: Exception, context: Dict[str, Any]):
        """调用异常处理Hook"""
        for hook in self.hooks:
            try:
                hook.on_exception(exception, context)
            except Exception as e:
                hook.logger.error(f"Hook {hook.__class__.__name__} on_exception failed: {e}")
    
    def clear_hooks(self):
        """清空所有Hook"""
        self.hooks.clear()
    
    def list_hooks(self) -> List[str]:
        """列出所有Hook"""
        return [hook.__class__.__name__ for hook in self.hooks]