#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
事件系统模块

提供统一的事件发布/订阅机制，支持模块间的松耦合通信。
包含事件总线、事件处理器、异步事件处理等功能。

作者: YOLOS团队
日期: 2024
"""

import asyncio
import threading
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
from concurrent.futures import ThreadPoolExecutor
import logging
from copy import deepcopy


class EventPriority(Enum):
    """事件优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class EventType(Enum):
    """系统事件类型"""
    # 系统事件
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_ERROR = "system.error"
    
    # 模块事件
    MODULE_LOADED = "module.loaded"
    MODULE_UNLOADED = "module.unloaded"
    MODULE_ERROR = "module.error"
    
    # 配置事件
    CONFIG_CHANGED = "config.changed"
    CONFIG_RELOADED = "config.reloaded"
    
    # 性能事件
    PERFORMANCE_ALERT = "performance.alert"
    PERFORMANCE_REPORT = "performance.report"
    
    # 推理事件
    INFERENCE_START = "inference.start"
    INFERENCE_COMPLETE = "inference.complete"
    INFERENCE_ERROR = "inference.error"
    
    # 自定义事件
    CUSTOM = "custom"


@dataclass
class Event:
    """事件数据类"""
    type: Union[EventType, str]
    data: Any = None
    source: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    priority: EventPriority = EventPriority.NORMAL
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.type, EventType):
            self.type = self.type.value


class IEventHandler(ABC):
    """事件处理器接口"""
    
    @abstractmethod
    def handle(self, event: Event) -> bool:
        """处理事件
        
        Args:
            event: 事件对象
            
        Returns:
            bool: 是否成功处理
        """
        pass
    
    @property
    def priority(self) -> EventPriority:
        """处理器优先级"""
        return EventPriority.NORMAL
    
    @property
    def async_handler(self) -> bool:
        """是否为异步处理器"""
        return False


class AsyncEventHandler(IEventHandler):
    """异步事件处理器基类"""
    
    @abstractmethod
    async def handle_async(self, event: Event) -> bool:
        """异步处理事件"""
        pass
    
    def handle(self, event: Event) -> bool:
        """同步接口包装"""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.handle_async(event))
        except RuntimeError:
            # 如果没有事件循环，创建新的
            return asyncio.run(self.handle_async(event))
    
    @property
    def async_handler(self) -> bool:
        return True


@dataclass
class EventSubscription:
    """事件订阅信息"""
    event_type: str
    handler: IEventHandler
    priority: EventPriority
    filter_func: Optional[Callable[[Event], bool]] = None
    once: bool = False  # 是否只执行一次
    weak_ref: bool = True  # 是否使用弱引用
    
    def matches(self, event: Event) -> bool:
        """检查是否匹配事件"""
        # 检查事件类型
        if self.event_type != "*" and not self._match_event_type(event.type):
            return False
        
        # 应用过滤器
        if self.filter_func and not self.filter_func(event):
            return False
        
        return True
    
    def _match_event_type(self, event_type: str) -> bool:
        """匹配事件类型（支持通配符）"""
        if self.event_type == event_type:
            return True
        
        # 支持通配符匹配
        if self.event_type.endswith("*"):
            prefix = self.event_type[:-1]
            return event_type.startswith(prefix)
        
        return False


class EventBus:
    """事件总线"""
    
    def __init__(self, max_workers: int = 4):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._subscriptions: Dict[str, List[EventSubscription]] = defaultdict(list)
        self._global_handlers: List[EventSubscription] = []
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._event_history: List[Event] = []
        self._max_history = 1000
        self._middleware: List[Callable[[Event], Event]] = []
        
        # 异步支持
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None
        self._async_tasks: Set[asyncio.Task] = set()
    
    def subscribe(self, 
                 event_type: Union[EventType, str], 
                 handler: Union[IEventHandler, Callable[[Event], bool]],
                 priority: EventPriority = EventPriority.NORMAL,
                 filter_func: Optional[Callable[[Event], bool]] = None,
                 once: bool = False) -> str:
        """订阅事件
        
        Args:
            event_type: 事件类型
            handler: 事件处理器或函数
            priority: 优先级
            filter_func: 过滤函数
            once: 是否只执行一次
            
        Returns:
            str: 订阅ID
        """
        if isinstance(event_type, EventType):
            event_type = event_type.value
        
        # 包装函数为处理器
        if callable(handler) and not isinstance(handler, IEventHandler):
            handler = FunctionEventHandler(handler)
        
        subscription = EventSubscription(
            event_type=event_type,
            handler=handler,
            priority=priority,
            filter_func=filter_func,
            once=once
        )
        
        with self._lock:
            if event_type == "*":
                self._global_handlers.append(subscription)
                # 按优先级排序
                self._global_handlers.sort(key=lambda s: s.priority.value, reverse=True)
            else:
                self._subscriptions[event_type].append(subscription)
                # 按优先级排序
                self._subscriptions[event_type].sort(key=lambda s: s.priority.value, reverse=True)
        
        subscription_id = f"{event_type}_{id(handler)}"
        self.logger.debug(f"Subscribed to {event_type} with handler {handler.__class__.__name__}")
        return subscription_id
    
    def unsubscribe(self, event_type: Union[EventType, str], handler: IEventHandler):
        """取消订阅"""
        if isinstance(event_type, EventType):
            event_type = event_type.value
        
        with self._lock:
            if event_type == "*":
                self._global_handlers = [s for s in self._global_handlers if s.handler != handler]
            else:
                self._subscriptions[event_type] = [
                    s for s in self._subscriptions[event_type] if s.handler != handler
                ]
        
        self.logger.debug(f"Unsubscribed from {event_type}")
    
    def publish(self, event: Event, async_mode: bool = False) -> bool:
        """发布事件
        
        Args:
            event: 事件对象
            async_mode: 是否异步处理
            
        Returns:
            bool: 是否成功发布
        """
        try:
            # 应用中间件
            for middleware in self._middleware:
                event = middleware(event)
            
            # 记录事件历史
            self._add_to_history(event)
            
            # 获取匹配的订阅
            subscriptions = self._get_matching_subscriptions(event)
            
            if not subscriptions:
                self.logger.debug(f"No handlers for event {event.type}")
                return True
            
            # 处理事件
            if async_mode:
                return self._publish_async(event, subscriptions)
            else:
                return self._publish_sync(event, subscriptions)
        
        except Exception as e:
            self.logger.error(f"Error publishing event {event.type}: {e}")
            return False
    
    def _get_matching_subscriptions(self, event: Event) -> List[EventSubscription]:
        """获取匹配的订阅"""
        subscriptions = []
        
        with self._lock:
            # 精确匹配
            if event.type in self._subscriptions:
                subscriptions.extend([
                    s for s in self._subscriptions[event.type] 
                    if s.matches(event)
                ])
            
            # 通配符匹配
            for event_pattern, subs in self._subscriptions.items():
                if event_pattern.endswith("*"):
                    prefix = event_pattern[:-1]
                    if event.type.startswith(prefix):
                        subscriptions.extend([
                            s for s in subs if s.matches(event)
                        ])
            
            # 全局处理器
            subscriptions.extend([
                s for s in self._global_handlers if s.matches(event)
            ])
        
        # 按优先级排序
        subscriptions.sort(key=lambda s: s.priority.value, reverse=True)
        return subscriptions
    
    def _publish_sync(self, event: Event, subscriptions: List[EventSubscription]) -> bool:
        """同步发布事件"""
        success = True
        to_remove = []
        
        for subscription in subscriptions:
            try:
                if subscription.handler.async_handler:
                    # 异步处理器在线程池中执行
                    future = self._executor.submit(subscription.handler.handle, event)
                    result = future.result(timeout=30)  # 30秒超时
                else:
                    result = subscription.handler.handle(event)
                
                if not result:
                    success = False
                
                # 标记一次性订阅待删除
                if subscription.once:
                    to_remove.append(subscription)
            
            except Exception as e:
                self.logger.error(f"Error in event handler {subscription.handler.__class__.__name__}: {e}")
                success = False
        
        # 移除一次性订阅
        self._remove_subscriptions(to_remove)
        
        return success
    
    def _publish_async(self, event: Event, subscriptions: List[EventSubscription]) -> bool:
        """异步发布事件"""
        if not self._async_loop:
            self._async_loop = asyncio.new_event_loop()
            threading.Thread(target=self._run_async_loop, daemon=True).start()
        
        # 在异步循环中处理
        future = asyncio.run_coroutine_threadsafe(
            self._handle_async_event(event, subscriptions),
            self._async_loop
        )
        
        return future.result(timeout=30)
    
    async def _handle_async_event(self, event: Event, subscriptions: List[EventSubscription]) -> bool:
        """异步处理事件"""
        success = True
        to_remove = []
        tasks = []
        
        for subscription in subscriptions:
            if subscription.handler.async_handler:
                task = asyncio.create_task(subscription.handler.handle_async(event))
            else:
                # 同步处理器在线程池中执行
                task = asyncio.create_task(
                    asyncio.get_event_loop().run_in_executor(
                        self._executor, subscription.handler.handle, event
                    )
                )
            
            tasks.append((task, subscription))
            self._async_tasks.add(task)
        
        # 等待所有任务完成
        for task, subscription in tasks:
            try:
                result = await task
                if not result:
                    success = False
                
                if subscription.once:
                    to_remove.append(subscription)
            
            except Exception as e:
                self.logger.error(f"Error in async event handler: {e}")
                success = False
            
            finally:
                self._async_tasks.discard(task)
        
        # 移除一次性订阅
        self._remove_subscriptions(to_remove)
        
        return success
    
    def _run_async_loop(self):
        """运行异步事件循环"""
        asyncio.set_event_loop(self._async_loop)
        self._async_loop.run_forever()
    
    def _remove_subscriptions(self, subscriptions: List[EventSubscription]):
        """移除订阅"""
        with self._lock:
            for subscription in subscriptions:
                if subscription in self._global_handlers:
                    self._global_handlers.remove(subscription)
                else:
                    for event_type, subs in self._subscriptions.items():
                        if subscription in subs:
                            subs.remove(subscription)
                            break
    
    def _add_to_history(self, event: Event):
        """添加到事件历史"""
        with self._lock:
            self._event_history.append(deepcopy(event))
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)
    
    def add_middleware(self, middleware: Callable[[Event], Event]):
        """添加中间件"""
        self._middleware.append(middleware)
    
    def get_event_history(self, event_type: Optional[str] = None, limit: int = 100) -> List[Event]:
        """获取事件历史"""
        with self._lock:
            history = self._event_history[-limit:]
            if event_type:
                history = [e for e in history if e.type == event_type]
            return history
    
    def clear_history(self):
        """清空事件历史"""
        with self._lock:
            self._event_history.clear()
    
    def shutdown(self):
        """关闭事件总线"""
        self.logger.info("Shutting down event bus")
        
        # 取消所有异步任务
        for task in self._async_tasks:
            task.cancel()
        
        # 关闭异步循环
        if self._async_loop:
            self._async_loop.call_soon_threadsafe(self._async_loop.stop)
        
        # 关闭线程池
        self._executor.shutdown(wait=True)


class FunctionEventHandler(IEventHandler):
    """函数事件处理器包装器"""
    
    def __init__(self, func: Callable[[Event], bool]):
        self.func = func
    
    def handle(self, event: Event) -> bool:
        return self.func(event)


# 全局事件总线实例
global_event_bus = EventBus()


def get_event_bus() -> EventBus:
    """获取全局事件总线"""
    return global_event_bus


def publish_event(event_type: Union[EventType, str], 
                 data: Any = None, 
                 source: Optional[str] = None,
                 priority: EventPriority = EventPriority.NORMAL,
                 **metadata) -> bool:
    """发布事件的便捷函数"""
    event = Event(
        type=event_type,
        data=data,
        source=source,
        priority=priority,
        metadata=metadata
    )
    return global_event_bus.publish(event)


def subscribe_event(event_type: Union[EventType, str], 
                   handler: Union[IEventHandler, Callable[[Event], bool]],
                   priority: EventPriority = EventPriority.NORMAL) -> str:
    """订阅事件的便捷函数"""
    return global_event_bus.subscribe(event_type, handler, priority)


def unsubscribe_event(event_type: Union[EventType, str], handler: IEventHandler):
    """取消订阅的便捷函数"""
    global_event_bus.unsubscribe(event_type, handler)