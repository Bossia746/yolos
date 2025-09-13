#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
响应速度优化模块
实现异步处理、帧跳跃、预测缓存等机制来提升系统响应速度
专为实时跟踪场景设计，确保低延迟和高响应性
"""

import time
import asyncio
import threading
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque, OrderedDict
from concurrent.futures import ThreadPoolExecutor, Future
import queue
import weakref

# 可选依赖
try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)

class ProcessingMode(Enum):
    """处理模式"""
    SYNCHRONOUS = "synchronous"  # 同步处理
    ASYNCHRONOUS = "asynchronous"  # 异步处理
    PIPELINE = "pipeline"  # 流水线处理
    ADAPTIVE = "adaptive"  # 自适应处理

class FrameSkipStrategy(Enum):
    """帧跳跃策略"""
    NONE = "none"  # 不跳帧
    FIXED = "fixed"  # 固定跳帧
    ADAPTIVE = "adaptive"  # 自适应跳帧
    SMART = "smart"  # 智能跳帧
    MOTION_BASED = "motion_based"  # 基于运动的跳帧

class CacheStrategy(Enum):
    """缓存策略"""
    LRU = "lru"  # 最近最少使用
    LFU = "lfu"  # 最少使用频率
    TTL = "ttl"  # 生存时间
    ADAPTIVE = "adaptive"  # 自适应缓存
    PREDICTIVE = "predictive"  # 预测性缓存

@dataclass
class FrameInfo:
    """帧信息"""
    frame_id: int
    timestamp: float
    data: np.ndarray
    priority: int = 1  # 优先级
    processing_time: float = 0.0
    skip_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()

@dataclass
class ProcessingResult:
    """处理结果"""
    frame_id: int
    result: Any
    processing_time: float
    timestamp: float
    success: bool = True
    error: Optional[str] = None
    cached: bool = False
    skipped: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ResponseConfig:
    """响应优化配置"""
    processing_mode: ProcessingMode = ProcessingMode.ASYNCHRONOUS
    frame_skip_strategy: FrameSkipStrategy = FrameSkipStrategy.ADAPTIVE
    cache_strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    
    # 异步处理配置
    max_concurrent_tasks: int = 4
    task_timeout: float = 1.0
    
    # 帧跳跃配置
    target_fps: float = 30.0
    max_skip_frames: int = 3
    skip_threshold_ms: float = 33.0  # 33ms = 30fps
    
    # 缓存配置
    cache_size: int = 100
    cache_ttl: float = 5.0
    enable_prediction_cache: bool = True
    
    # 性能配置
    enable_profiling: bool = True
    enable_adaptive_optimization: bool = True
    response_time_target: float = 16.0  # 16ms = 60fps

class AsyncTaskManager:
    """异步任务管理器
    
    管理异步处理任务，提供任务调度和结果收集
    """
    
    def __init__(self, max_workers: int = 4, timeout: float = 1.0):
        self.max_workers = max_workers
        self.timeout = timeout
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 任务管理
        self.active_tasks: Dict[int, Future] = {}
        self.task_results: Dict[int, ProcessingResult] = {}
        self.task_queue = queue.PriorityQueue()
        
        # 统计信息
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.timeout_tasks = 0
        
        # 控制标志
        self.running = False
        self.worker_thread = None
    
    def start(self):
        """启动任务管理器"""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        self.logger.info(f"异步任务管理器启动，最大工作线程数: {self.max_workers}")
    
    def stop(self):
        """停止任务管理器"""
        if not self.running:
            return
        
        self.running = False
        
        # 等待工作线程结束
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
        
        # 关闭线程池
        self.executor.shutdown(wait=True)
        
        # 清理任务
        self.active_tasks.clear()
        self.task_results.clear()
        
        self.logger.info("异步任务管理器已停止")
    
    def submit_task(self, frame_info: FrameInfo, processing_func: Callable, *args, **kwargs) -> int:
        """提交异步任务
        
        Args:
            frame_info: 帧信息
            processing_func: 处理函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            int: 任务ID
        """
        if not self.running:
            self.start()
        
        # 添加到任务队列
        priority = -frame_info.priority  # 负数表示高优先级
        self.task_queue.put((priority, frame_info.frame_id, frame_info, processing_func, args, kwargs))
        
        return frame_info.frame_id
    
    def _worker_loop(self):
        """工作线程循环"""
        while self.running:
            try:
                # 获取任务
                try:
                    priority, task_id, frame_info, func, args, kwargs = self.task_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # 检查是否已有太多活跃任务
                if len(self.active_tasks) >= self.max_workers:
                    # 等待一些任务完成
                    self._cleanup_completed_tasks()
                    if len(self.active_tasks) >= self.max_workers:
                        # 如果仍然太多，跳过这个任务
                        self.logger.warning(f"任务队列满，跳过任务 {task_id}")
                        continue
                
                # 提交任务到线程池
                future = self.executor.submit(self._execute_task, task_id, frame_info, func, args, kwargs)
                self.active_tasks[task_id] = future
                
                self.task_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"工作线程错误: {e}")
    
    def _execute_task(self, task_id: int, frame_info: FrameInfo, func: Callable, args: tuple, kwargs: dict) -> ProcessingResult:
        """执行任务
        
        Args:
            task_id: 任务ID
            frame_info: 帧信息
            func: 处理函数
            args: 位置参数
            kwargs: 关键字参数
            
        Returns:
            ProcessingResult: 处理结果
        """
        start_time = time.time()
        
        try:
            # 执行处理函数
            result = func(frame_info.data, *args, **kwargs)
            
            processing_time = (time.time() - start_time) * 1000
            
            # 创建结果
            processing_result = ProcessingResult(
                frame_id=task_id,
                result=result,
                processing_time=processing_time,
                timestamp=time.time(),
                success=True
            )
            
            # 存储结果
            self.task_results[task_id] = processing_result
            self.completed_tasks += 1
            
            return processing_result
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            
            # 创建错误结果
            error_result = ProcessingResult(
                frame_id=task_id,
                result=None,
                processing_time=processing_time,
                timestamp=time.time(),
                success=False,
                error=str(e)
            )
            
            self.task_results[task_id] = error_result
            self.failed_tasks += 1
            
            self.logger.error(f"任务 {task_id} 执行失败: {e}")
            return error_result
    
    def get_result(self, task_id: int, timeout: Optional[float] = None) -> Optional[ProcessingResult]:
        """获取任务结果
        
        Args:
            task_id: 任务ID
            timeout: 超时时间
            
        Returns:
            Optional[ProcessingResult]: 处理结果
        """
        # 检查是否已有结果
        if task_id in self.task_results:
            return self.task_results.pop(task_id)
        
        # 检查是否有活跃任务
        if task_id in self.active_tasks:
            future = self.active_tasks[task_id]
            
            try:
                # 等待任务完成
                result = future.result(timeout=timeout or self.timeout)
                self.active_tasks.pop(task_id, None)
                return result
                
            except TimeoutError:
                self.timeout_tasks += 1
                self.logger.warning(f"任务 {task_id} 超时")
                return None
            except Exception as e:
                self.active_tasks.pop(task_id, None)
                self.logger.error(f"获取任务 {task_id} 结果失败: {e}")
                return None
        
        return None
    
    def _cleanup_completed_tasks(self):
        """清理已完成的任务"""
        completed_ids = []
        
        for task_id, future in self.active_tasks.items():
            if future.done():
                completed_ids.append(task_id)
        
        for task_id in completed_ids:
            self.active_tasks.pop(task_id, None)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            "active_tasks": len(self.active_tasks),
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "timeout_tasks": self.timeout_tasks,
            "success_rate": self.completed_tasks / max(self.completed_tasks + self.failed_tasks, 1),
            "queue_size": self.task_queue.qsize()
        }

class FrameSkipper:
    """帧跳跃器
    
    根据不同策略决定是否跳过帧处理
    """
    
    def __init__(self, strategy: FrameSkipStrategy = FrameSkipStrategy.ADAPTIVE, config: ResponseConfig = None):
        self.strategy = strategy
        self.config = config or ResponseConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 跳帧统计
        self.total_frames = 0
        self.skipped_frames = 0
        self.processing_times = deque(maxlen=30)  # 最近30帧的处理时间
        
        # 自适应参数
        self.current_skip_rate = 0
        self.target_processing_time = self.config.skip_threshold_ms
        
        # 运动检测（用于基于运动的跳帧）
        self.previous_frame = None
        self.motion_threshold = 0.1
    
    def should_skip_frame(self, frame_info: FrameInfo, current_load: float = 0.0) -> Tuple[bool, str]:
        """判断是否应该跳过帧
        
        Args:
            frame_info: 帧信息
            current_load: 当前系统负载 (0-1)
            
        Returns:
            Tuple[bool, str]: (是否跳过, 跳过原因)
        """
        self.total_frames += 1
        
        if self.strategy == FrameSkipStrategy.NONE:
            return False, ""
        
        elif self.strategy == FrameSkipStrategy.FIXED:
            # 固定跳帧：每N帧处理一次
            skip_interval = max(1, int(self.config.max_skip_frames))
            should_skip = (self.total_frames % (skip_interval + 1)) != 0
            reason = f"固定跳帧，间隔={skip_interval}" if should_skip else ""
            
        elif self.strategy == FrameSkipStrategy.ADAPTIVE:
            # 自适应跳帧：根据处理时间动态调整
            should_skip, reason = self._adaptive_skip_decision(frame_info, current_load)
            
        elif self.strategy == FrameSkipStrategy.SMART:
            # 智能跳帧：综合考虑多个因素
            should_skip, reason = self._smart_skip_decision(frame_info, current_load)
            
        elif self.strategy == FrameSkipStrategy.MOTION_BASED:
            # 基于运动的跳帧：运动小时跳帧
            should_skip, reason = self._motion_based_skip_decision(frame_info)
            
        else:
            should_skip, reason = False, ""
        
        if should_skip:
            self.skipped_frames += 1
            frame_info.skip_reason = reason
        
        return should_skip, reason
    
    def _adaptive_skip_decision(self, frame_info: FrameInfo, current_load: float) -> Tuple[bool, str]:
        """自适应跳帧决策
        
        Args:
            frame_info: 帧信息
            current_load: 当前负载
            
        Returns:
            Tuple[bool, str]: (是否跳过, 原因)
        """
        # 计算平均处理时间
        if self.processing_times:
            avg_processing_time = np.mean(list(self.processing_times))
        else:
            avg_processing_time = 0
        
        # 根据处理时间调整跳帧率
        if avg_processing_time > self.target_processing_time * 1.5:
            # 处理时间过长，增加跳帧
            self.current_skip_rate = min(self.current_skip_rate + 0.1, 0.8)
        elif avg_processing_time < self.target_processing_time * 0.8:
            # 处理时间较短，减少跳帧
            self.current_skip_rate = max(self.current_skip_rate - 0.05, 0.0)
        
        # 考虑系统负载
        load_factor = current_load * 0.5
        effective_skip_rate = min(self.current_skip_rate + load_factor, 0.9)
        
        # 决策
        should_skip = np.random.random() < effective_skip_rate
        reason = f"自适应跳帧，跳帧率={effective_skip_rate:.2f}" if should_skip else ""
        
        return should_skip, reason
    
    def _smart_skip_decision(self, frame_info: FrameInfo, current_load: float) -> Tuple[bool, str]:
        """智能跳帧决策
        
        Args:
            frame_info: 帧信息
            current_load: 当前负载
            
        Returns:
            Tuple[bool, str]: (是否跳过, 原因)
        """
        reasons = []
        skip_score = 0.0
        
        # 因素1：处理时间
        if self.processing_times:
            avg_time = np.mean(list(self.processing_times))
            if avg_time > self.target_processing_time:
                skip_score += 0.3
                reasons.append("处理时间过长")
        
        # 因素2：系统负载
        if current_load > 0.7:
            skip_score += 0.4
            reasons.append("系统负载高")
        
        # 因素3：帧优先级
        if frame_info.priority < 5:  # 低优先级帧
            skip_score += 0.2
            reasons.append("低优先级")
        
        # 因素4：时间间隔
        current_time = time.time()
        if hasattr(self, 'last_process_time'):
            time_since_last = current_time - self.last_process_time
            if time_since_last < 0.01:  # 10ms内的帧
                skip_score += 0.3
                reasons.append("时间间隔过短")
        
        # 决策
        should_skip = skip_score > 0.5
        reason = ", ".join(reasons) if should_skip else ""
        
        if not should_skip:
            self.last_process_time = current_time
        
        return should_skip, reason
    
    def _motion_based_skip_decision(self, frame_info: FrameInfo) -> Tuple[bool, str]:
        """基于运动的跳帧决策
        
        Args:
            frame_info: 帧信息
            
        Returns:
            Tuple[bool, str]: (是否跳过, 原因)
        """
        if cv2 is None or self.previous_frame is None:
            self.previous_frame = frame_info.data
            return False, ""
        
        try:
            # 计算帧差
            current_gray = cv2.cvtColor(frame_info.data, cv2.COLOR_BGR2GRAY) if len(frame_info.data.shape) == 3 else frame_info.data
            prev_gray = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY) if len(self.previous_frame.shape) == 3 else self.previous_frame
            
            # 调整尺寸以匹配
            if current_gray.shape != prev_gray.shape:
                prev_gray = cv2.resize(prev_gray, (current_gray.shape[1], current_gray.shape[0]))
            
            # 计算运动量
            diff = cv2.absdiff(current_gray, prev_gray)
            motion_score = np.mean(diff) / 255.0
            
            # 更新前一帧
            self.previous_frame = frame_info.data
            
            # 决策：运动量小时跳帧
            should_skip = motion_score < self.motion_threshold
            reason = f"运动量低 ({motion_score:.3f})" if should_skip else ""
            
            return should_skip, reason
            
        except Exception as e:
            self.logger.error(f"运动检测失败: {e}")
            return False, ""
    
    def update_processing_time(self, processing_time: float):
        """更新处理时间
        
        Args:
            processing_time: 处理时间（毫秒）
        """
        self.processing_times.append(processing_time)
    
    def get_skip_stats(self) -> Dict[str, Any]:
        """获取跳帧统计
        
        Returns:
            Dict[str, Any]: 跳帧统计
        """
        skip_rate = self.skipped_frames / max(self.total_frames, 1)
        avg_processing_time = np.mean(list(self.processing_times)) if self.processing_times else 0
        
        return {
            "total_frames": self.total_frames,
            "skipped_frames": self.skipped_frames,
            "skip_rate": skip_rate,
            "avg_processing_time_ms": avg_processing_time,
            "strategy": self.strategy.value,
            "current_skip_rate": getattr(self, 'current_skip_rate', 0)
        }

class PredictiveCache:
    """预测性缓存
    
    智能缓存处理结果和预测未来需求
    """
    
    def __init__(self, strategy: CacheStrategy = CacheStrategy.ADAPTIVE, max_size: int = 100, ttl: float = 5.0):
        self.strategy = strategy
        self.max_size = max_size
        self.ttl = ttl
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 缓存存储
        if strategy == CacheStrategy.LRU:
            self.cache = OrderedDict()
        else:
            self.cache = {}
        
        # 缓存元数据
        self.access_times = {}  # 访问时间
        self.access_counts = {}  # 访问次数
        self.creation_times = {}  # 创建时间
        self.hit_counts = {}  # 命中次数
        
        # 预测模型
        self.access_patterns = deque(maxlen=1000)  # 访问模式
        self.prediction_accuracy = 0.0
        
        # 统计信息
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 清理线程
        self.cleanup_thread = None
        self.running = False
    
    def start(self):
        """启动缓存管理"""
        if self.running:
            return
        
        self.running = True
        
        # 启动清理线程
        if self.strategy in [CacheStrategy.TTL, CacheStrategy.ADAPTIVE]:
            self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self.cleanup_thread.start()
        
        self.logger.info(f"预测性缓存启动，策略: {self.strategy.value}")
    
    def stop(self):
        """停止缓存管理"""
        if not self.running:
            return
        
        self.running = False
        
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=1.0)
        
        self.logger.info("预测性缓存已停止")
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项
        
        Args:
            key: 缓存键
            
        Returns:
            Optional[Any]: 缓存值
        """
        self.total_requests += 1
        current_time = time.time()
        
        # 记录访问模式
        self.access_patterns.append((key, current_time))
        
        # 检查是否存在
        if key not in self.cache:
            self.cache_misses += 1
            return None
        
        # 检查TTL
        if self.strategy in [CacheStrategy.TTL, CacheStrategy.ADAPTIVE]:
            if current_time - self.creation_times.get(key, 0) > self.ttl:
                self._remove_key(key)
                self.cache_misses += 1
                return None
        
        # 更新访问信息
        self.access_times[key] = current_time
        self.access_counts[key] = self.access_counts.get(key, 0) + 1
        self.hit_counts[key] = self.hit_counts.get(key, 0) + 1
        
        # LRU更新
        if self.strategy == CacheStrategy.LRU:
            self.cache.move_to_end(key)
        
        self.cache_hits += 1
        return self.cache[key]
    
    def put(self, key: str, value: Any, priority: int = 1):
        """存储缓存项
        
        Args:
            key: 缓存键
            value: 缓存值
            priority: 优先级
        """
        current_time = time.time()
        
        # 检查容量
        if len(self.cache) >= self.max_size:
            self._evict_items()
        
        # 存储
        self.cache[key] = value
        self.creation_times[key] = current_time
        self.access_times[key] = current_time
        self.access_counts[key] = 1
        self.hit_counts[key] = 0
        
        # 预测性预加载
        if self.strategy == CacheStrategy.PREDICTIVE:
            self._predictive_preload(key)
    
    def _evict_items(self):
        """驱逐缓存项"""
        if not self.cache:
            return
        
        current_time = time.time()
        
        if self.strategy == CacheStrategy.LRU:
            # 移除最近最少使用的项
            key = next(iter(self.cache))
            self._remove_key(key)
            
        elif self.strategy == CacheStrategy.LFU:
            # 移除使用频率最低的项
            key = min(self.access_counts.keys(), key=lambda k: self.access_counts.get(k, 0))
            self._remove_key(key)
            
        elif self.strategy == CacheStrategy.TTL:
            # 移除过期的项
            expired_keys = [
                key for key, creation_time in self.creation_times.items()
                if current_time - creation_time > self.ttl
            ]
            for key in expired_keys:
                self._remove_key(key)
            
            # 如果没有过期项，移除最旧的
            if not expired_keys and self.cache:
                oldest_key = min(self.creation_times.keys(), key=lambda k: self.creation_times[k])
                self._remove_key(oldest_key)
                
        elif self.strategy == CacheStrategy.ADAPTIVE:
            # 自适应驱逐：综合考虑多个因素
            scores = {}
            for key in self.cache.keys():
                age = current_time - self.creation_times.get(key, current_time)
                access_freq = self.access_counts.get(key, 1)
                last_access = current_time - self.access_times.get(key, current_time)
                hit_rate = self.hit_counts.get(key, 0) / max(self.access_counts.get(key, 1), 1)
                
                # 计算驱逐分数（分数越高越容易被驱逐）
                score = age * 0.3 + last_access * 0.4 - access_freq * 0.2 - hit_rate * 0.1
                scores[key] = score
            
            # 移除分数最高的项
            if scores:
                key_to_remove = max(scores.keys(), key=lambda k: scores[k])
                self._remove_key(key_to_remove)
    
    def _remove_key(self, key: str):
        """移除缓存键
        
        Args:
            key: 缓存键
        """
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.access_counts.pop(key, None)
        self.creation_times.pop(key, None)
        self.hit_counts.pop(key, None)
    
    def _predictive_preload(self, key: str):
        """预测性预加载
        
        Args:
            key: 当前访问的键
        """
        # 分析访问模式，预测可能需要的键
        # 这里实现简单的序列预测
        if len(self.access_patterns) < 10:
            return
        
        # 查找访问序列模式
        recent_keys = [pattern[0] for pattern in list(self.access_patterns)[-10:]]
        
        # 简单的下一个键预测（基于历史序列）
        for i in range(len(recent_keys) - 2):
            if recent_keys[i] == key:
                next_key = recent_keys[i + 1]
                # 这里可以触发预加载逻辑
                self.logger.debug(f"预测下一个可能访问的键: {next_key}")
    
    def _cleanup_loop(self):
        """清理循环"""
        while self.running:
            try:
                current_time = time.time()
                
                # 清理过期项
                expired_keys = [
                    key for key, creation_time in self.creation_times.items()
                    if current_time - creation_time > self.ttl
                ]
                
                for key in expired_keys:
                    self._remove_key(key)
                
                if expired_keys:
                    self.logger.debug(f"清理过期缓存项: {len(expired_keys)}")
                
                # 等待
                time.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"缓存清理错误: {e}")
    
    def invalidate(self, pattern: str = None):
        """失效缓存
        
        Args:
            pattern: 匹配模式（如果为None则清空所有）
        """
        if pattern is None:
            # 清空所有
            cleared_count = len(self.cache)
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
            self.creation_times.clear()
            self.hit_counts.clear()
            
            self.logger.info(f"清空所有缓存，清理项目数: {cleared_count}")
        else:
            # 按模式清理
            keys_to_remove = [key for key in self.cache.keys() if pattern in key]
            for key in keys_to_remove:
                self._remove_key(key)
            
            self.logger.info(f"按模式清理缓存 '{pattern}'，清理项目数: {len(keys_to_remove)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计
        
        Returns:
            Dict[str, Any]: 缓存统计
        """
        hit_rate = self.cache_hits / max(self.total_requests, 1)
        
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "strategy": self.strategy.value,
            "prediction_accuracy": self.prediction_accuracy
        }

class ResponseOptimizer:
    """响应速度优化器主类
    
    整合异步处理、帧跳跃、预测缓存等优化技术
    """
    
    def __init__(self, config: ResponseConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化组件
        self.task_manager = AsyncTaskManager(
            max_workers=config.max_concurrent_tasks,
            timeout=config.task_timeout
        )
        
        self.frame_skipper = FrameSkipper(
            strategy=config.frame_skip_strategy,
            config=config
        )
        
        self.cache = PredictiveCache(
            strategy=config.cache_strategy,
            max_size=config.cache_size,
            ttl=config.cache_ttl
        )
        
        # 性能监控
        self.response_times = deque(maxlen=100)
        self.processing_stats = {
            "total_frames": 0,
            "processed_frames": 0,
            "cached_results": 0,
            "async_tasks": 0
        }
        
        # 自适应优化
        self.adaptive_enabled = config.enable_adaptive_optimization
        self.performance_target = config.response_time_target
        
        self.logger.info(f"响应速度优化器初始化完成，模式: {config.processing_mode.value}")
    
    def start(self):
        """启动优化器"""
        self.task_manager.start()
        self.cache.start()
        
        self.logger.info("响应速度优化器已启动")
    
    def stop(self):
        """停止优化器"""
        self.task_manager.stop()
        self.cache.stop()
        
        self.logger.info("响应速度优化器已停止")
    
    def process_frame(self, frame_data: np.ndarray, processing_func: Callable, 
                     frame_id: Optional[int] = None, priority: int = 1, 
                     enable_cache: bool = True, **kwargs) -> Optional[ProcessingResult]:
        """处理帧
        
        Args:
            frame_data: 帧数据
            processing_func: 处理函数
            frame_id: 帧ID
            priority: 优先级
            enable_cache: 是否启用缓存
            **kwargs: 额外参数
            
        Returns:
            Optional[ProcessingResult]: 处理结果
        """
        start_time = time.time()
        
        # 生成帧ID
        if frame_id is None:
            frame_id = int(time.time() * 1000000)  # 微秒时间戳
        
        # 创建帧信息
        frame_info = FrameInfo(
            frame_id=frame_id,
            timestamp=start_time,
            data=frame_data,
            priority=priority
        )
        
        self.processing_stats["total_frames"] += 1
        
        # 检查缓存
        cache_key = self._generate_cache_key(frame_data, kwargs) if enable_cache else None
        if cache_key and self.config.enable_prediction_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self.processing_stats["cached_results"] += 1
                
                # 创建缓存结果
                result = ProcessingResult(
                    frame_id=frame_id,
                    result=cached_result,
                    processing_time=0.0,
                    timestamp=time.time(),
                    cached=True
                )
                
                self._update_response_time(time.time() - start_time)
                return result
        
        # 检查是否跳帧
        current_load = len(self.task_manager.active_tasks) / self.config.max_concurrent_tasks
        should_skip, skip_reason = self.frame_skipper.should_skip_frame(frame_info, current_load)
        
        if should_skip:
            result = ProcessingResult(
                frame_id=frame_id,
                result=None,
                processing_time=0.0,
                timestamp=time.time(),
                skipped=True,
                metadata={"skip_reason": skip_reason}
            )
            
            self._update_response_time(time.time() - start_time)
            return result
        
        # 选择处理模式
        if self.config.processing_mode == ProcessingMode.SYNCHRONOUS:
            result = self._process_synchronous(frame_info, processing_func, **kwargs)
            
        elif self.config.processing_mode == ProcessingMode.ASYNCHRONOUS:
            result = self._process_asynchronous(frame_info, processing_func, **kwargs)
            
        elif self.config.processing_mode == ProcessingMode.PIPELINE:
            result = self._process_pipeline(frame_info, processing_func, **kwargs)
            
        elif self.config.processing_mode == ProcessingMode.ADAPTIVE:
            result = self._process_adaptive(frame_info, processing_func, **kwargs)
            
        else:
            result = self._process_synchronous(frame_info, processing_func, **kwargs)
        
        # 缓存结果
        if result and result.success and cache_key and enable_cache:
            self.cache.put(cache_key, result.result, priority)
        
        # 更新统计
        if result and not result.skipped:
            self.processing_stats["processed_frames"] += 1
            self.frame_skipper.update_processing_time(result.processing_time)
        
        self._update_response_time(time.time() - start_time)
        
        # 自适应优化
        if self.adaptive_enabled:
            self._adaptive_optimization(result)
        
        return result
    
    def _process_synchronous(self, frame_info: FrameInfo, processing_func: Callable, **kwargs) -> ProcessingResult:
        """同步处理
        
        Args:
            frame_info: 帧信息
            processing_func: 处理函数
            **kwargs: 额外参数
            
        Returns:
            ProcessingResult: 处理结果
        """
        start_time = time.time()
        
        try:
            result = processing_func(frame_info.data, **kwargs)
            processing_time = (time.time() - start_time) * 1000
            
            return ProcessingResult(
                frame_id=frame_info.frame_id,
                result=result,
                processing_time=processing_time,
                timestamp=time.time(),
                success=True
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            
            return ProcessingResult(
                frame_id=frame_info.frame_id,
                result=None,
                processing_time=processing_time,
                timestamp=time.time(),
                success=False,
                error=str(e)
            )
    
    def _process_asynchronous(self, frame_info: FrameInfo, processing_func: Callable, **kwargs) -> Optional[ProcessingResult]:
        """异步处理
        
        Args:
            frame_info: 帧信息
            processing_func: 处理函数
            **kwargs: 额外参数
            
        Returns:
            Optional[ProcessingResult]: 处理结果
        """
        # 提交异步任务
        task_id = self.task_manager.submit_task(frame_info, processing_func, **kwargs)
        self.processing_stats["async_tasks"] += 1
        
        # 立即尝试获取结果（非阻塞）
        result = self.task_manager.get_result(task_id, timeout=0.001)
        
        return result
    
    def _process_pipeline(self, frame_info: FrameInfo, processing_func: Callable, **kwargs) -> Optional[ProcessingResult]:
        """流水线处理
        
        Args:
            frame_info: 帧信息
            processing_func: 处理函数
            **kwargs: 额外参数
            
        Returns:
            Optional[ProcessingResult]: 处理结果
        """
        # 流水线处理：提交当前任务，获取之前的结果
        task_id = self.task_manager.submit_task(frame_info, processing_func, **kwargs)
        
        # 尝试获取之前任务的结果
        if hasattr(self, '_previous_task_id'):
            result = self.task_manager.get_result(self._previous_task_id, timeout=0.01)
            self._previous_task_id = task_id
            return result
        else:
            self._previous_task_id = task_id
            return None
    
    def _process_adaptive(self, frame_info: FrameInfo, processing_func: Callable, **kwargs) -> Optional[ProcessingResult]:
        """自适应处理
        
        Args:
            frame_info: 帧信息
            processing_func: 处理函数
            **kwargs: 额外参数
            
        Returns:
            Optional[ProcessingResult]: 处理结果
        """
        # 根据当前负载选择处理模式
        current_load = len(self.task_manager.active_tasks) / self.config.max_concurrent_tasks
        avg_response_time = np.mean(list(self.response_times)) if self.response_times else 0
        
        if current_load < 0.5 and avg_response_time < self.performance_target:
            # 负载低，使用同步处理
            return self._process_synchronous(frame_info, processing_func, **kwargs)
        elif current_load < 0.8:
            # 中等负载，使用异步处理
            return self._process_asynchronous(frame_info, processing_func, **kwargs)
        else:
            # 高负载，使用流水线处理
            return self._process_pipeline(frame_info, processing_func, **kwargs)
    
    def _generate_cache_key(self, frame_data: np.ndarray, kwargs: dict) -> str:
        """生成缓存键
        
        Args:
            frame_data: 帧数据
            kwargs: 参数
            
        Returns:
            str: 缓存键
        """
        # 简单的哈希方法（实际应用中可能需要更复杂的键生成策略）
        data_hash = hash(frame_data.tobytes())
        kwargs_hash = hash(str(sorted(kwargs.items())))
        
        return f"frame_{data_hash}_{kwargs_hash}"
    
    def _update_response_time(self, response_time: float):
        """更新响应时间
        
        Args:
            response_time: 响应时间（秒）
        """
        response_time_ms = response_time * 1000
        self.response_times.append(response_time_ms)
    
    def _adaptive_optimization(self, result: Optional[ProcessingResult]):
        """自适应优化
        
        Args:
            result: 处理结果
        """
        if not result:
            return
        
        # 计算性能指标
        avg_response_time = np.mean(list(self.response_times)) if self.response_times else 0
        
        # 如果响应时间超过目标，调整策略
        if avg_response_time > self.performance_target * 1.5:
            # 响应时间过长，启用更激进的优化
            if self.config.frame_skip_strategy == FrameSkipStrategy.NONE:
                self.config.frame_skip_strategy = FrameSkipStrategy.ADAPTIVE
                self.frame_skipper.strategy = FrameSkipStrategy.ADAPTIVE
                self.logger.info("启用自适应跳帧以改善响应时间")
            
            if self.config.processing_mode == ProcessingMode.SYNCHRONOUS:
                self.config.processing_mode = ProcessingMode.ASYNCHRONOUS
                self.logger.info("切换到异步处理模式")
        
        elif avg_response_time < self.performance_target * 0.5:
            # 响应时间很好，可以减少优化
            if self.config.frame_skip_strategy == FrameSkipStrategy.ADAPTIVE:
                self.config.frame_skip_strategy = FrameSkipStrategy.NONE
                self.frame_skipper.strategy = FrameSkipStrategy.NONE
                self.logger.info("禁用跳帧以提高处理质量")
    
    def get_pending_results(self) -> List[ProcessingResult]:
        """获取待处理的异步结果
        
        Returns:
            List[ProcessingResult]: 结果列表
        """
        results = []
        
        # 检查所有活跃任务
        for task_id in list(self.task_manager.active_tasks.keys()):
            result = self.task_manager.get_result(task_id, timeout=0.001)
            if result:
                results.append(result)
        
        return results
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """获取优化报告
        
        Returns:
            Dict[str, Any]: 优化报告
        """
        avg_response_time = np.mean(list(self.response_times)) if self.response_times else 0
        
        return {
            "config": {
                "processing_mode": self.config.processing_mode.value,
                "frame_skip_strategy": self.config.frame_skip_strategy.value,
                "cache_strategy": self.config.cache_strategy.value,
                "target_fps": self.config.target_fps,
                "response_time_target_ms": self.performance_target
            },
            "performance_metrics": {
                "avg_response_time_ms": avg_response_time,
                "total_frames": self.processing_stats["total_frames"],
                "processed_frames": self.processing_stats["processed_frames"],
                "processing_rate": self.processing_stats["processed_frames"] / max(self.processing_stats["total_frames"], 1),
                "cached_results": self.processing_stats["cached_results"],
                "cache_hit_rate": self.processing_stats["cached_results"] / max(self.processing_stats["total_frames"], 1)
            },
            "task_manager_stats": self.task_manager.get_stats(),
            "frame_skip_stats": self.frame_skipper.get_skip_stats(),
            "cache_stats": self.cache.get_stats(),
            "recommendations": self._generate_optimization_recommendations()
        }
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """生成优化建议
        
        Returns:
            List[str]: 优化建议
        """
        recommendations = []
        
        avg_response_time = np.mean(list(self.response_times)) if self.response_times else 0
        cache_stats = self.cache.get_stats()
        skip_stats = self.frame_skipper.get_skip_stats()
        
        # 响应时间建议
        if avg_response_time > self.performance_target * 2:
            recommendations.append("响应时间过长，建议启用更激进的跳帧策略")
        
        # 缓存建议
        if cache_stats["hit_rate"] < 0.3:
            recommendations.append("缓存命中率低，考虑调整缓存策略或增加缓存大小")
        
        # 跳帧建议
        if skip_stats["skip_rate"] > 0.7:
            recommendations.append("跳帧率过高，可能影响跟踪质量，考虑优化处理算法")
        
        # 异步处理建议
        task_stats = self.task_manager.get_stats()
        if task_stats["success_rate"] < 0.9:
            recommendations.append("异步任务成功率低，检查处理函数的稳定性")
        
        return recommendations if recommendations else ["当前优化配置表现良好"]
    
    def cleanup(self):
        """清理资源"""
        try:
            self.stop()
            self.cache.invalidate()  # 清空缓存
            
            self.logger.info("响应速度优化器资源清理完成")
            
        except Exception as e:
            self.logger.error(f"资源清理失败: {e}")

# 测试代码
if __name__ == "__main__":
    # 创建配置
    config = ResponseConfig(
        processing_mode=ProcessingMode.ADAPTIVE,
        frame_skip_strategy=FrameSkipStrategy.SMART,
        cache_strategy=CacheStrategy.ADAPTIVE,
        target_fps=30.0,
        max_concurrent_tasks=4,
        enable_adaptive_optimization=True
    )
    
    # 创建优化器
    optimizer = ResponseOptimizer(config)
    optimizer.start()
    
    # 模拟处理函数
    def mock_processing_func(frame_data, delay=0.01):
        time.sleep(delay)  # 模拟处理时间
        return {"detection": "person", "confidence": 0.95}
    
    print("开始响应速度测试...")
    
    # 测试处理
    for i in range(20):
        # 模拟帧数据
        frame_data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 处理帧
        result = optimizer.process_frame(
            frame_data=frame_data,
            processing_func=mock_processing_func,
            frame_id=i,
            priority=1,
            delay=0.02 if i % 3 == 0 else 0.01  # 模拟不同的处理时间
        )
        
        if result:
            status = "缓存" if result.cached else ("跳过" if result.skipped else "处理")
            print(f"帧 {i}: {status}, 耗时: {result.processing_time:.2f}ms")
        
        time.sleep(0.03)  # 模拟帧间隔
    
    # 获取待处理结果
    pending_results = optimizer.get_pending_results()
    print(f"\n待处理结果数: {len(pending_results)}")
    
    # 获取优化报告
    report = optimizer.get_optimization_report()
    print("\n优化报告:")
    for key, value in report.items():
        print(f"  {key}: {value}")
    
    # 清理资源
    optimizer.cleanup()