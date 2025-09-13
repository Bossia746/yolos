#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOS 依赖注入容器
提供统一的依赖管理和服务定位功能

作者: YOLOS团队
版本: 1.0.0
创建时间: 2025-09-11
"""

import threading
import inspect
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Type, TypeVar, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import logging
from contextlib import contextmanager

from .exceptions import YOLOSException
from .plugin_manager import PluginDependencyError

T = TypeVar('T')


class ServiceLifetime(Enum):
    """服务生命周期"""
    SINGLETON = "singleton"    # 单例
    TRANSIENT = "transient"    # 瞬态
    SCOPED = "scoped"          # 作用域


@dataclass
class ServiceDescriptor:
    """服务描述符"""
    service_type: Type
    implementation_type: Optional[Type] = None
    factory: Optional[Callable] = None
    instance: Optional[Any] = None
    lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT
    dependencies: List[Type] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


class IServiceProvider(ABC):
    """服务提供者接口"""
    
    @abstractmethod
    def get_service(self, service_type: Type[T]) -> Optional[T]:
        """获取服务"""
        pass
    
    @abstractmethod
    def get_required_service(self, service_type: Type[T]) -> T:
        """获取必需的服务"""
        pass
    
    @abstractmethod
    def get_services(self, service_type: Type[T]) -> List[T]:
        """获取所有指定类型的服务"""
        pass


class ServiceContainer(IServiceProvider):
    """依赖注入容器"""
    
    def __init__(self):
        self._services: Dict[Type, List[ServiceDescriptor]] = {}
        self._instances: Dict[Type, Any] = {}
        self._scoped_instances: Dict[str, Dict[Type, Any]] = {}
        self._lock = threading.RLock()
        self._current_scope: Optional[str] = None
        self._resolution_stack: Set[Type] = set()  # 用于循环依赖检测
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 注册容器自身
        self.register_instance(IServiceProvider, self)
        self.register_instance(ServiceContainer, self)
    
    def register_singleton(self, service_type: Type[T], 
                          implementation_type: Optional[Type[T]] = None,
                          factory: Optional[Callable[[], T]] = None,
                          instance: Optional[T] = None) -> 'ServiceContainer':
        """注册单例服务"""
        return self._register_service(
            service_type, implementation_type, factory, instance, ServiceLifetime.SINGLETON
        )
    
    def register_transient(self, service_type: Type[T],
                          implementation_type: Optional[Type[T]] = None,
                          factory: Optional[Callable[[], T]] = None) -> 'ServiceContainer':
        """注册瞬态服务"""
        return self._register_service(
            service_type, implementation_type, factory, None, ServiceLifetime.TRANSIENT
        )
    
    def register_scoped(self, service_type: Type[T],
                       implementation_type: Optional[Type[T]] = None,
                       factory: Optional[Callable[[], T]] = None) -> 'ServiceContainer':
        """注册作用域服务"""
        return self._register_service(
            service_type, implementation_type, factory, None, ServiceLifetime.SCOPED
        )
    
    def register_instance(self, service_type: Type[T], instance: T) -> 'ServiceContainer':
        """注册实例"""
        return self._register_service(
            service_type, None, None, instance, ServiceLifetime.SINGLETON
        )
    
    def _register_service(self, service_type: Type, implementation_type: Optional[Type],
                         factory: Optional[Callable], instance: Optional[Any],
                         lifetime: ServiceLifetime) -> 'ServiceContainer':
        """注册服务的内部方法"""
        with self._lock:
            if service_type not in self._services:
                self._services[service_type] = []
            
            # 分析依赖
            dependencies = []
            if implementation_type:
                dependencies = self._analyze_dependencies(implementation_type)
            elif factory:
                dependencies = self._analyze_factory_dependencies(factory)
            
            descriptor = ServiceDescriptor(
                service_type=service_type,
                implementation_type=implementation_type,
                factory=factory,
                instance=instance,
                lifetime=lifetime,
                dependencies=dependencies
            )
            
            self._services[service_type].append(descriptor)
            
            # 如果是单例且有实例，直接存储
            if lifetime == ServiceLifetime.SINGLETON and instance is not None:
                self._instances[service_type] = instance
        
        return self
    
    def _analyze_dependencies(self, cls: Type) -> List[Type]:
        """分析类的依赖"""
        dependencies = []
        
        try:
            # 获取构造函数签名
            sig = inspect.signature(cls.__init__)
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                
                # 获取参数类型
                if param.annotation != inspect.Parameter.empty:
                    dependencies.append(param.annotation)
        except Exception:
            pass
        
        return dependencies
    
    def _analyze_factory_dependencies(self, factory: Callable) -> List[Type]:
        """分析工厂函数的依赖"""
        dependencies = []
        
        try:
            sig = inspect.signature(factory)
            for param_name, param in sig.parameters.items():
                if param.annotation != inspect.Parameter.empty:
                    dependencies.append(param.annotation)
        except Exception:
            pass
        
        return dependencies
    
    def get_service(self, service_type: Type[T]) -> Optional[T]:
        """获取服务"""
        try:
            return self._resolve_service(service_type)
        except Exception:
            return None
    
    def get_required_service(self, service_type: Type[T]) -> T:
        """获取必需的服务"""
        service = self._resolve_service(service_type)
        if service is None:
            raise PluginDependencyError(f"Required service {service_type.__name__} not found")
        return service
    
    def get_services(self, service_type: Type[T]) -> List[T]:
        """获取所有指定类型的服务"""
        services = []
        
        with self._lock:
            if service_type in self._services:
                for descriptor in self._services[service_type]:
                    try:
                        service = self._create_instance(descriptor)
                        if service is not None:
                            services.append(service)
                    except Exception:
                        continue
        
        return services
    
    @contextmanager
    def _dependency_resolution_context(self, service_type: Type):
        """依赖解析上下文管理器，用于循环依赖检测"""
        if service_type in self._resolution_stack:
            cycle_path = ' -> '.join([t.__name__ for t in self._resolution_stack]) + f' -> {service_type.__name__}'
            raise PluginDependencyError(f"Circular dependency detected: {cycle_path}")
        
        self._resolution_stack.add(service_type)
        try:
            yield
        finally:
            self._resolution_stack.discard(service_type)
    
    def _resolve_service(self, service_type: Type[T]) -> Optional[T]:
        """解析服务"""
        with self._lock:
            # 检查是否已注册
            if service_type not in self._services:
                self.logger.debug(f"Service {service_type.__name__} not registered")
                return None
            
            # 使用循环依赖检测上下文
            with self._dependency_resolution_context(service_type):
                # 获取最后注册的服务描述符
                descriptor = self._services[service_type][-1]
                
                try:
                    # 根据生命周期返回实例
                    if descriptor.lifetime == ServiceLifetime.SINGLETON:
                        return self._get_singleton_instance(descriptor)
                    elif descriptor.lifetime == ServiceLifetime.SCOPED:
                        return self._get_scoped_instance(descriptor)
                    else:  # TRANSIENT
                        return self._create_instance(descriptor)
                except Exception as e:
                    self.logger.error(f"Failed to resolve service {service_type.__name__}: {e}")
                    raise PluginDependencyError(f"Service resolution failed for {service_type.__name__}: {e}") from e
    
    def _get_singleton_instance(self, descriptor: ServiceDescriptor) -> Any:
        """获取单例实例"""
        service_type = descriptor.service_type
        
        if service_type in self._instances:
            return self._instances[service_type]
        
        instance = self._create_instance(descriptor)
        if instance is not None:
            self._instances[service_type] = instance
        
        return instance
    
    def _get_scoped_instance(self, descriptor: ServiceDescriptor) -> Any:
        """获取作用域实例"""
        if self._current_scope is None:
            raise PluginDependencyError("No active scope for scoped service")
        
        service_type = descriptor.service_type
        
        if self._current_scope not in self._scoped_instances:
            self._scoped_instances[self._current_scope] = {}
        
        scope_instances = self._scoped_instances[self._current_scope]
        
        if service_type in scope_instances:
            return scope_instances[service_type]
        
        instance = self._create_instance(descriptor)
        if instance is not None:
            scope_instances[service_type] = instance
        
        return instance
    
    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """创建服务实例"""
        service_name = descriptor.service_type.__name__
        self.logger.debug(f"Creating instance of {service_name}")
        
        try:
            # 如果有预设实例，直接返回
            if descriptor.instance is not None:
                self.logger.debug(f"Returning pre-configured instance of {service_name}")
                return descriptor.instance
            
            # 如果有工厂函数，使用工厂创建
            if descriptor.factory is not None:
                self.logger.debug(f"Creating {service_name} using factory function")
                return self._invoke_factory(descriptor.factory, descriptor.dependencies)
            
            # 使用实现类型创建
            if descriptor.implementation_type is not None:
                impl_name = descriptor.implementation_type.__name__
                self.logger.debug(f"Creating {service_name} using implementation {impl_name}")
                return self._create_from_type(descriptor.implementation_type, descriptor.dependencies)
            
            # 使用服务类型创建（如果是具体类）
            if not inspect.isabstract(descriptor.service_type):
                self.logger.debug(f"Creating {service_name} using service type directly")
                return self._create_from_type(descriptor.service_type, descriptor.dependencies)
            
            raise PluginDependencyError(f"Cannot create instance of abstract type {service_name} without implementation or factory")
            
        except PluginDependencyError:
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error creating instance of {service_name}: {e}")
            raise PluginDependencyError(f"Failed to create instance of {service_name}: {e}") from e
    
    def _invoke_factory(self, factory: Callable, dependencies: List[Type]) -> Any:
        """调用工厂函数"""
        factory_name = getattr(factory, '__name__', str(factory))
        self.logger.debug(f"Invoking factory {factory_name} with {len(dependencies)} dependencies")
        
        # 解析依赖参数
        args = []
        for i, dep_type in enumerate(dependencies):
            try:
                dep_instance = self._resolve_service(dep_type)
                if dep_instance is None:
                    raise PluginDependencyError(f"Cannot resolve dependency {dep_type.__name__} for factory {factory_name}")
                args.append(dep_instance)
                self.logger.debug(f"Resolved dependency {i+1}/{len(dependencies)}: {dep_type.__name__}")
            except Exception as e:
                raise PluginDependencyError(f"Failed to resolve dependency {dep_type.__name__} for factory {factory_name}: {e}") from e
        
        try:
            result = factory(*args)
            self.logger.debug(f"Factory {factory_name} successfully created instance")
            return result
        except Exception as e:
            self.logger.error(f"Factory {factory_name} failed to create instance: {e}")
            raise PluginDependencyError(f"Factory {factory_name} execution failed: {e}") from e
    
    def _create_from_type(self, cls: Type, dependencies: List[Type]) -> Any:
        """从类型创建实例"""
        class_name = cls.__name__
        self.logger.debug(f"Creating instance of {class_name} with {len(dependencies)} dependencies")
        
        # 解析构造函数依赖
        args = []
        for i, dep_type in enumerate(dependencies):
            try:
                dep_instance = self._resolve_service(dep_type)
                if dep_instance is None:
                    raise PluginDependencyError(f"Cannot resolve dependency {dep_type.__name__} for {class_name}")
                args.append(dep_instance)
                self.logger.debug(f"Resolved dependency {i+1}/{len(dependencies)}: {dep_type.__name__}")
            except Exception as e:
                raise PluginDependencyError(f"Failed to resolve dependency {dep_type.__name__} for {class_name}: {e}") from e
        
        try:
            instance = cls(*args)
            self.logger.debug(f"Successfully created instance of {class_name}")
            return instance
        except Exception as e:
            self.logger.error(f"Failed to instantiate {class_name}: {e}")
            raise PluginDependencyError(f"Constructor of {class_name} failed: {e}") from e
    
    def create_scope(self, scope_id: Optional[str] = None) -> 'ServiceScope':
        """创建服务作用域"""
        if scope_id is None:
            scope_id = f"scope_{id(threading.current_thread())}"
        
        return ServiceScope(self, scope_id)
    
    def _enter_scope(self, scope_id: str):
        """进入作用域"""
        with self._lock:
            self._current_scope = scope_id
            if scope_id not in self._scoped_instances:
                self._scoped_instances[scope_id] = {}
    
    def _exit_scope(self, scope_id: str):
        """退出作用域"""
        with self._lock:
            if scope_id in self._scoped_instances:
                # 清理作用域实例
                scope_instances = self._scoped_instances[scope_id]
                for instance in scope_instances.values():
                    if hasattr(instance, 'dispose'):
                        try:
                            instance.dispose()
                        except Exception:
                            pass
                
                del self._scoped_instances[scope_id]
            
            if self._current_scope == scope_id:
                self._current_scope = None
    
    def is_registered(self, service_type: Type) -> bool:
        """检查服务是否已注册"""
        with self._lock:
            return service_type in self._services
    
    def get_registered_services(self) -> List[Type]:
        """获取所有已注册的服务类型"""
        with self._lock:
            return list(self._services.keys())
    
    def get_service_info(self, service_type: Type) -> Optional[Dict[str, Any]]:
        """获取服务的详细信息"""
        with self._lock:
            if service_type not in self._services:
                return None
            
            descriptors = self._services[service_type]
            return {
                'service_type': service_type.__name__,
                'registrations': [
                    {
                        'implementation': desc.implementation_type.__name__ if desc.implementation_type else None,
                        'factory': getattr(desc.factory, '__name__', str(desc.factory)) if desc.factory else None,
                        'lifetime': desc.lifetime.value,
                        'dependencies': [dep.__name__ for dep in desc.dependencies],
                        'tags': desc.tags,
                        'has_instance': desc.instance is not None
                    }
                    for desc in descriptors
                ]
            }
    
    def validate_dependencies(self) -> List[str]:
        """验证所有注册服务的依赖关系"""
        issues = []
        
        with self._lock:
            for service_type, descriptors in self._services.items():
                for desc in descriptors:
                    # 检查依赖是否都已注册
                    for dep_type in desc.dependencies:
                        if dep_type not in self._services:
                            issues.append(f"Service {service_type.__name__} depends on unregistered service {dep_type.__name__}")
                    
                    # 检查是否存在循环依赖（静态检查）
                    try:
                        self._check_circular_dependencies(service_type, set())
                    except PluginDependencyError as e:
                        issues.append(str(e))
        
        return issues
    
    def _check_circular_dependencies(self, service_type: Type, visited: Set[Type]):
        """静态检查循环依赖"""
        if service_type in visited:
            cycle_path = ' -> '.join([t.__name__ for t in visited]) + f' -> {service_type.__name__}'
            raise PluginDependencyError(f"Static circular dependency detected: {cycle_path}")
        
        if service_type not in self._services:
            return
        
        visited.add(service_type)
        descriptor = self._services[service_type][-1]
        
        for dep_type in descriptor.dependencies:
            self._check_circular_dependencies(dep_type, visited.copy())
    
    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """获取依赖关系图"""
        graph = {}
        
        with self._lock:
            for service_type, descriptors in self._services.items():
                service_name = service_type.__name__
                # 使用最后注册的描述符
                descriptor = descriptors[-1]
                graph[service_name] = [dep.__name__ for dep in descriptor.dependencies]
        
        return graph
    
    def clear(self):
        """清空容器"""
        with self._lock:
            # 清理单例实例
            for instance in self._instances.values():
                if hasattr(instance, 'dispose'):
                    try:
                        instance.dispose()
                    except Exception:
                        pass
            
            # 清理作用域实例
            for scope_instances in self._scoped_instances.values():
                for instance in scope_instances.values():
                    if hasattr(instance, 'dispose'):
                        try:
                            instance.dispose()
                        except Exception:
                            pass
            
            self._services.clear()
            self._instances.clear()
            self._scoped_instances.clear()
            self._current_scope = None


class ServiceScope:
    """服务作用域"""
    
    def __init__(self, container: ServiceContainer, scope_id: str):
        self.container = container
        self.scope_id = scope_id
    
    def __enter__(self):
        self.container._enter_scope(self.scope_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.container._exit_scope(self.scope_id)


# 装饰器支持
def injectable(cls: Type[T]) -> Type[T]:
    """标记类为可注入的"""
    cls._injectable = True
    return cls


def inject(service_type: Type[T]) -> Callable:
    """依赖注入装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 从全局容器获取服务
            service = global_container.get_required_service(service_type)
            return func(service, *args, **kwargs)
        return wrapper
    return decorator


# 全局容器实例
global_container = ServiceContainer()


def get_container() -> ServiceContainer:
    """获取全局容器"""
    return global_container


def configure_services(configurator: Callable[[ServiceContainer], None]):
    """配置服务"""
    configurator(global_container)


if __name__ == "__main__":
    # 测试代码
    
    # 定义测试接口和实现
    class ILogger(ABC):
        @abstractmethod
        def log(self, message: str):
            pass
    
    class ConsoleLogger(ILogger):
        def log(self, message: str):
            print(f"[LOG] {message}")
    
    class IRepository(ABC):
        @abstractmethod
        def get_data(self) -> str:
            pass
    
    class DatabaseRepository(IRepository):
        def __init__(self, logger: ILogger):
            self.logger = logger
        
        def get_data(self) -> str:
            self.logger.log("Getting data from database")
            return "database_data"
    
    # 配置容器
    container = ServiceContainer()
    container.register_singleton(ILogger, ConsoleLogger)
    container.register_transient(IRepository, DatabaseRepository)
    
    # 测试依赖注入
    repo = container.get_required_service(IRepository)
    data = repo.get_data()
    print(f"Data: {data}")
    
    # 测试作用域
    with container.create_scope() as scope:
        scoped_repo = container.get_required_service(IRepository)
        print(f"Scoped data: {scoped_repo.get_data()}")
    
    print("Dependency injection test completed")