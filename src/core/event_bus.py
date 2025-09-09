"""事件总线系统"""

from typing import Dict, List, Callable, Any, Optional
import threading
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import weakref
import asyncio
from concurrent.futures import ThreadPoolExecutor


@dataclass
class Event:
    """事件数据结构"""
    name: str
    data: Any
    timestamp: datetime
    source: Optional[str] = None
    priority: int = 0  # 0=normal, 1=high, 2=critical


class EventHandler:
    """事件处理器包装类"""
    
    def __init__(self, callback: Callable, priority: int = 0, once: bool = False):
        self.callback = callback
        self.priority = priority
        self.once = once
        self.call_count = 0
    
    def __call__(self, event: Event) -> Any:
        self.call_count += 1
        return self.callback(event)
    
    def __lt__(self, other):
        return self.priority > other.priority  # 高优先级排在前面


class EventBus:
    """事件总线
    
    提供发布-订阅模式的事件通信机制
    """
    
    def __init__(self, max_workers: int = 4):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 事件处理器存储
        self._handlers: Dict[str, List[EventHandler]] = defaultdict(list)
        self._wildcard_handlers: List[EventHandler] = []
        
        # 事件历史
        self._event_history: List[Event] = []
        self._max_history = 1000
        
        # 线程安全
        self._lock = threading.RLock()
        
        # 异步处理
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._async_handlers: Dict[str, List[EventHandler]] = defaultdict(list)
        
        # 事件统计
        self._stats = {
            'events_emitted': 0,
            'events_handled': 0,
            'handlers_registered': 0,
            'errors': 0
        }
        
        # 弱引用支持
        self._weak_handlers: Dict[str, List] = defaultdict(list)
    
    def on(self, event_name: str, callback: Callable, priority: int = 0, once: bool = False) -> str:
        """注册事件处理器
        
        Args:
            event_name: 事件名称，支持通配符 '*'
            callback: 回调函数
            priority: 优先级 (0=normal, 1=high, 2=critical)
            once: 是否只执行一次
            
        Returns:
            str: 处理器ID，用于取消注册
        """
        with self._lock:
            handler = EventHandler(callback, priority, once)
            handler_id = f"{event_name}_{id(handler)}"
            
            if event_name == '*':
                self._wildcard_handlers.append(handler)
                self._wildcard_handlers.sort()
            else:
                self._handlers[event_name].append(handler)
                self._handlers[event_name].sort()
            
            self._stats['handlers_registered'] += 1
            self.logger.debug(f"Registered handler for event '{event_name}' with priority {priority}")
            
            return handler_id
    
    def on_weak(self, event_name: str, obj: Any, method_name: str, priority: int = 0) -> str:
        """注册弱引用事件处理器
        
        Args:
            event_name: 事件名称
            obj: 对象实例
            method_name: 方法名称
            priority: 优先级
            
        Returns:
            str: 处理器ID
        """
        def callback_wrapper(event: Event):
            obj_ref = weak_obj()
            if obj_ref is not None:
                method = getattr(obj_ref, method_name, None)
                if method:
                    return method(event)
        
        weak_obj = weakref.ref(obj)
        handler = EventHandler(callback_wrapper, priority)
        handler_id = f"{event_name}_weak_{id(handler)}"
        
        with self._lock:
            self._weak_handlers[event_name].append((weak_obj, handler))
            self._handlers[event_name].append(handler)
            self._handlers[event_name].sort()
        
        return handler_id
    
    def on_async(self, event_name: str, callback: Callable, priority: int = 0) -> str:
        """注册异步事件处理器
        
        Args:
            event_name: 事件名称
            callback: 异步回调函数
            priority: 优先级
            
        Returns:
            str: 处理器ID
        """
        with self._lock:
            handler = EventHandler(callback, priority)
            handler_id = f"{event_name}_async_{id(handler)}"
            
            self._async_handlers[event_name].append(handler)
            self._async_handlers[event_name].sort()
            
            self.logger.debug(f"Registered async handler for event '{event_name}'")
            return handler_id
    
    def off(self, event_name: str, handler_id: str = None) -> bool:
        """取消注册事件处理器
        
        Args:
            event_name: 事件名称
            handler_id: 处理器ID，如果为None则移除所有处理器
            
        Returns:
            bool: 是否成功移除
        """
        with self._lock:
            if handler_id is None:
                # 移除所有处理器
                removed = len(self._handlers.get(event_name, []))
                if event_name in self._handlers:
                    del self._handlers[event_name]
                if event_name in self._async_handlers:
                    removed += len(self._async_handlers[event_name])
                    del self._async_handlers[event_name]
                
                self.logger.debug(f"Removed all {removed} handlers for event '{event_name}'")
                return removed > 0
            else:
                # 移除特定处理器
                # 这里简化实现，实际应该根据handler_id精确移除
                return False
    
    def emit(self, event_name: str, data: Any = None, source: str = None, priority: int = 0) -> List[Any]:
        """发射事件
        
        Args:
            event_name: 事件名称
            data: 事件数据
            source: 事件源
            priority: 事件优先级
            
        Returns:
            List[Any]: 处理器返回值列表
        """
        event = Event(
            name=event_name,
            data=data,
            timestamp=datetime.now(),
            source=source,
            priority=priority
        )
        
        with self._lock:
            # 记录事件历史
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)
            
            self._stats['events_emitted'] += 1
        
        self.logger.debug(f"Emitting event '{event_name}' from {source}")
        
        results = []
        
        # 处理同步处理器
        handlers = self._get_handlers(event_name)
        for handler in handlers:
            try:
                result = handler(event)
                results.append(result)
                self._stats['events_handled'] += 1
                
                # 如果是一次性处理器，移除它
                if handler.once:
                    self._remove_handler(event_name, handler)
            
            except Exception as e:
                self.logger.error(f"Error in event handler for '{event_name}': {e}")
                self._stats['errors'] += 1
        
        # 处理异步处理器
        async_handlers = self._async_handlers.get(event_name, [])
        for handler in async_handlers:
            try:
                future = self._executor.submit(handler, event)
                results.append(future)
            except Exception as e:
                self.logger.error(f"Error submitting async handler for '{event_name}': {e}")
                self._stats['errors'] += 1
        
        return results
    
    def emit_async(self, event_name: str, data: Any = None, source: str = None) -> None:
        """异步发射事件
        
        Args:
            event_name: 事件名称
            data: 事件数据
            source: 事件源
        """
        self._executor.submit(self.emit, event_name, data, source)
    
    def wait_for_event(self, event_name: str, timeout: float = None) -> Optional[Event]:
        """等待特定事件
        
        Args:
            event_name: 事件名称
            timeout: 超时时间（秒）
            
        Returns:
            Optional[Event]: 事件对象，超时返回None
        """
        import time
        
        start_time = time.time()
        
        while True:
            # 检查最近的事件
            with self._lock:
                for event in reversed(self._event_history):
                    if event.name == event_name:
                        return event
            
            if timeout and (time.time() - start_time) > timeout:
                return None
            
            time.sleep(0.01)  # 短暂休眠
    
    def _get_handlers(self, event_name: str) -> List[EventHandler]:
        """获取事件处理器列表
        
        Args:
            event_name: 事件名称
            
        Returns:
            List[EventHandler]: 处理器列表
        """
        handlers = []
        
        # 特定事件处理器
        handlers.extend(self._handlers.get(event_name, []))
        
        # 通配符处理器
        handlers.extend(self._wildcard_handlers)
        
        # 清理失效的弱引用处理器
        self._cleanup_weak_handlers(event_name)
        
        return sorted(handlers)
    
    def _cleanup_weak_handlers(self, event_name: str) -> None:
        """清理失效的弱引用处理器
        
        Args:
            event_name: 事件名称
        """
        if event_name in self._weak_handlers:
            valid_handlers = []
            handlers_to_remove = []
            
            for weak_obj, handler in self._weak_handlers[event_name]:
                if weak_obj() is not None:
                    valid_handlers.append((weak_obj, handler))
                else:
                    handlers_to_remove.append(handler)
            
            self._weak_handlers[event_name] = valid_handlers
            
            # 从主处理器列表中移除失效的处理器
            for handler in handlers_to_remove:
                if handler in self._handlers[event_name]:
                    self._handlers[event_name].remove(handler)
    
    def _remove_handler(self, event_name: str, handler: EventHandler) -> None:
        """移除处理器
        
        Args:
            event_name: 事件名称
            handler: 处理器对象
        """
        if event_name in self._handlers and handler in self._handlers[event_name]:
            self._handlers[event_name].remove(handler)
        
        if handler in self._wildcard_handlers:
            self._wildcard_handlers.remove(handler)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取事件统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        with self._lock:
            return {
                **self._stats,
                'active_handlers': sum(len(handlers) for handlers in self._handlers.values()),
                'wildcard_handlers': len(self._wildcard_handlers),
                'async_handlers': sum(len(handlers) for handlers in self._async_handlers.values()),
                'event_history_size': len(self._event_history)
            }
    
    def get_recent_events(self, count: int = 10) -> List[Event]:
        """获取最近的事件
        
        Args:
            count: 事件数量
            
        Returns:
            List[Event]: 最近的事件列表
        """
        with self._lock:
            return self._event_history[-count:]
    
    def clear_history(self) -> None:
        """清空事件历史"""
        with self._lock:
            self._event_history.clear()
            self.logger.info("Event history cleared")
    
    def shutdown(self) -> None:
        """关闭事件总线"""
        self.logger.info("Shutting down event bus")
        
        # 关闭线程池
        self._executor.shutdown(wait=True)
        
        # 清理处理器
        with self._lock:
            self._handlers.clear()
            self._wildcard_handlers.clear()
            self._async_handlers.clear()
            self._weak_handlers.clear()
        
        self.logger.info("Event bus shutdown complete")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


# 全局事件总线实例
_global_event_bus = None


def get_global_event_bus() -> EventBus:
    """获取全局事件总线实例
    
    Returns:
        EventBus: 全局事件总线
    """
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus


def set_global_event_bus(event_bus: EventBus) -> None:
    """设置全局事件总线实例
    
    Args:
        event_bus: 事件总线实例
    """
    global _global_event_bus
    _global_event_bus = event_bus