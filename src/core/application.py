#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
应用程序框架模块

提供统一的应用程序生命周期管理，整合所有核心模块。
包含应用启动、关闭、模块管理、配置管理等功能。

作者: YOLOS团队
日期: 2024
"""

import asyncio
import atexit
import signal
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union
import logging
from concurrent.futures import ThreadPoolExecutor

# 导入核心模块
from .config_manager import ConfigManager, get_config_manager
from .dependency_injection import ServiceContainer, get_container
from .event_system import EventBus, EventType, Event, get_event_bus, publish_event
from .exceptions import ExceptionHandler, get_exception_handler
from .module_manager import ModuleManager, get_module_manager
from .performance_monitor import PerformanceMonitor, get_performance_monitor
from .platform_compatibility import PlatformCompatibilityManager
from .commercial_standards import CommercialStandardsValidator


class ApplicationState(Enum):
    """应用程序状态"""
    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ApplicationConfig:
    """应用程序配置"""
    name: str = "YOLOS"
    version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    config_dir: str = "config"
    modules_dir: str = "modules"
    data_dir: str = "data"
    temp_dir: str = "temp"
    max_workers: int = 4
    enable_performance_monitoring: bool = True
    enable_commercial_validation: bool = True
    auto_load_modules: bool = True
    graceful_shutdown_timeout: int = 30
    
    # 平台兼容性配置
    supported_platforms: List[str] = field(default_factory=lambda: ["windows", "linux", "macos"])
    supported_applications: List[str] = field(default_factory=lambda: [
        "face_recognition", "pose_estimation", "object_detection", 
        "pet_detection", "plant_recognition", "static_analysis"
    ])


class Application:
    """YOLOS应用程序主类"""
    
    def __init__(self, config: Optional[ApplicationConfig] = None):
        self.config = config or ApplicationConfig()
        self.state = ApplicationState.INITIALIZING
        self.logger = self._setup_logging()
        
        # 核心组件
        self.config_manager: Optional[ConfigManager] = None
        self.container: Optional[ServiceContainer] = None
        self.event_bus: Optional[EventBus] = None
        self.exception_handler: Optional[ExceptionHandler] = None
        self.module_manager: Optional[ModuleManager] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.platform_manager: Optional[PlatformCompatibilityManager] = None
        self.standards_validator: Optional[CommercialStandardsValidator] = None
        
        # 运行时状态
        self._startup_time: Optional[float] = None
        self._shutdown_event = threading.Event()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None
        self._signal_handlers_registered = False
        
        # 注册退出处理
        atexit.register(self.shutdown)
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志系统"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('yolos.log')
            ]
        )
        return logging.getLogger(self.__class__.__name__)
    
    def initialize(self) -> bool:
        """初始化应用程序"""
        try:
            self.logger.info(f"Initializing {self.config.name} v{self.config.version}")
            self.state = ApplicationState.INITIALIZING
            
            # 创建必要的目录
            self._create_directories()
            
            # 初始化核心组件
            self._initialize_core_components()
            
            # 注册服务
            self._register_services()
            
            # 加载配置
            self._load_configuration()
            
            # 验证平台兼容性
            if not self._validate_platform_compatibility():
                self.logger.warning("Platform compatibility validation failed")
            
            # 验证商用标准
            if self.config.enable_commercial_validation:
                if not self._validate_commercial_standards():
                    self.logger.warning("Commercial standards validation failed")
            
            self.logger.info("Application initialized successfully")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to initialize application: {e}")
            self.state = ApplicationState.ERROR
            return False
    
    def _create_directories(self):
        """创建必要的目录"""
        directories = [
            self.config.config_dir,
            self.config.modules_dir,
            self.config.data_dir,
            self.config.temp_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _initialize_core_components(self):
        """初始化核心组件"""
        # 配置管理器
        self.config_manager = get_config_manager()
        
        # 依赖注入容器
        self.container = get_container()
        
        # 事件总线
        self.event_bus = get_event_bus()
        
        # 异常处理器
        self.exception_handler = get_exception_handler()
        
        # 模块管理器
        self.module_manager = get_module_manager()
        
        # 性能监控器
        if self.config.enable_performance_monitoring:
            self.performance_monitor = get_performance_monitor()
        
        # 平台兼容性管理器
        self.platform_manager = PlatformCompatibilityManager()
        
        # 商用标准验证器
        if self.config.enable_commercial_validation:
            self.standards_validator = CommercialStandardsValidator()
        
        # 线程池
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
    
    def _register_services(self):
        """注册服务到依赖注入容器"""
        # 注册核心服务
        self.container.register_singleton(ConfigManager, lambda: self.config_manager)
        self.container.register_singleton(EventBus, lambda: self.event_bus)
        self.container.register_singleton(ExceptionHandler, lambda: self.exception_handler)
        self.container.register_singleton(ModuleManager, lambda: self.module_manager)
        
        if self.performance_monitor:
            self.container.register_singleton(PerformanceMonitor, lambda: self.performance_monitor)
        
        if self.platform_manager:
            self.container.register_singleton(PlatformCompatibilityManager, lambda: self.platform_manager)
        
        if self.standards_validator:
            self.container.register_singleton(CommercialStandardsValidator, lambda: self.standards_validator)
        
        # 注册应用程序本身
        self.container.register_singleton(Application, lambda: self)
    
    def _load_configuration(self):
        """加载配置"""
        try:
            # 加载所有配置
            config = self.config_manager.load_all_configs()
            
            # 更新应用配置
            if 'application' in config:
                app_config = config['application']
                for key, value in app_config.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
            
            self.logger.info("Configuration loaded successfully")
        
        except Exception as e:
            self.logger.warning(f"Failed to load configuration: {e}")
    
    def _validate_platform_compatibility(self) -> bool:
        """验证平台兼容性"""
        try:
            if not self.platform_manager:
                return True
            
            compatibility_report = self.platform_manager.check_compatibility()
            
            if compatibility_report.overall_status == "success":
                self.logger.info("Platform compatibility validation passed")
                return True
            else:
                self.logger.warning(f"Platform compatibility issues: {compatibility_report.recommendations}")
                return False
        
        except Exception as e:
            self.logger.error(f"Platform compatibility validation error: {e}")
            return False
    
    def _validate_commercial_standards(self) -> bool:
        """验证商用标准"""
        try:
            if not self.standards_validator:
                return True
            
            validation_result = self.standards_validator.validate_system()
            
            if validation_result.overall_level == "production":
                self.logger.info("Commercial standards validation passed")
                return True
            else:
                self.logger.warning(f"Commercial standards issues: {validation_result.recommendations}")
                return False
        
        except Exception as e:
            self.logger.error(f"Commercial standards validation error: {e}")
            return False
    
    def start(self) -> bool:
        """启动应用程序"""
        try:
            if self.state != ApplicationState.INITIALIZING:
                if not self.initialize():
                    return False
            
            self.logger.info("Starting application")
            self.state = ApplicationState.STARTING
            self._startup_time = time.time()
            
            # 注册信号处理器
            self._register_signal_handlers()
            
            # 启动性能监控
            if self.performance_monitor:
                self.performance_monitor.start_monitoring()
            
            # 发布启动事件
            publish_event(EventType.SYSTEM_STARTUP, {
                'application': self.config.name,
                'version': self.config.version,
                'startup_time': self._startup_time
            })
            
            # 自动加载模块
            if self.config.auto_load_modules:
                self._load_modules()
            
            self.state = ApplicationState.RUNNING
            self.logger.info(f"Application started successfully in {time.time() - self._startup_time:.2f}s")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to start application: {e}")
            self.state = ApplicationState.ERROR
            return False
    
    def _register_signal_handlers(self):
        """注册信号处理器"""
        if self._signal_handlers_registered:
            return
        
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down gracefully")
            self.shutdown()
        
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            self._signal_handlers_registered = True
        except ValueError:
            # 在某些环境中可能无法注册信号处理器
            self.logger.warning("Could not register signal handlers")
    
    def _load_modules(self):
        """加载模块"""
        try:
            if self.module_manager:
                modules_loaded = self.module_manager.discover_and_load_modules(self.config.modules_dir)
                self.logger.info(f"Loaded {len(modules_loaded)} modules")
        
        except Exception as e:
            self.logger.error(f"Failed to load modules: {e}")
    
    def run(self) -> int:
        """运行应用程序"""
        try:
            if not self.start():
                return 1
            
            # 主循环
            self.logger.info("Application is running. Press Ctrl+C to stop.")
            
            try:
                while self.state == ApplicationState.RUNNING:
                    if self._shutdown_event.wait(timeout=1.0):
                        break
            except KeyboardInterrupt:
                self.logger.info("Received keyboard interrupt")
            
            return 0
        
        except Exception as e:
            self.logger.error(f"Application runtime error: {e}")
            return 1
        
        finally:
            self.shutdown()
    
    def shutdown(self, timeout: Optional[int] = None):
        """关闭应用程序"""
        if self.state in [ApplicationState.STOPPING, ApplicationState.STOPPED]:
            return
        
        self.logger.info("Shutting down application")
        self.state = ApplicationState.STOPPING
        
        timeout = timeout or self.config.graceful_shutdown_timeout
        
        try:
            # 发布关闭事件
            publish_event(EventType.SYSTEM_SHUTDOWN, {
                'application': self.config.name,
                'shutdown_time': time.time()
            })
            
            # 停止模块
            if self.module_manager:
                self.module_manager.unload_all_modules()
            
            # 停止性能监控
            if self.performance_monitor:
                self.performance_monitor.stop_monitoring()
            
            # 关闭事件总线
            if self.event_bus:
                self.event_bus.shutdown()
            
            # 关闭线程池
            if self._executor:
                self._executor.shutdown(wait=True)
            
            # 设置关闭事件
            self._shutdown_event.set()
            
            self.state = ApplicationState.STOPPED
            self.logger.info("Application shutdown completed")
        
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            self.state = ApplicationState.ERROR
    
    @contextmanager
    def application_context(self):
        """应用程序上下文管理器"""
        try:
            if not self.start():
                raise RuntimeError("Failed to start application")
            yield self
        finally:
            self.shutdown()
    
    def get_service(self, service_type: Type) -> Any:
        """获取服务实例"""
        if not self.container:
            raise RuntimeError("Application not initialized")
        return self.container.resolve(service_type)
    
    def register_service(self, service_type: Type, factory, singleton: bool = True):
        """注册服务"""
        if not self.container:
            raise RuntimeError("Application not initialized")
        
        if singleton:
            self.container.register_singleton(service_type, factory)
        else:
            self.container.register_transient(service_type, factory)
    
    def get_status(self) -> Dict[str, Any]:
        """获取应用程序状态"""
        status = {
            'name': self.config.name,
            'version': self.config.version,
            'state': self.state.value,
            'startup_time': self._startup_time,
            'uptime': time.time() - self._startup_time if self._startup_time else 0
        }
        
        # 添加性能信息
        if self.performance_monitor:
            status['performance'] = self.performance_monitor.get_current_metrics().__dict__
        
        # 添加模块信息
        if self.module_manager:
            status['modules'] = {
                'loaded': len(self.module_manager.get_loaded_modules()),
                'available': len(self.module_manager.get_available_modules())
            }
        
        return status


# 全局应用程序实例
_global_app: Optional[Application] = None


def create_application(config: Optional[ApplicationConfig] = None) -> Application:
    """创建应用程序实例"""
    global _global_app
    if _global_app is None:
        _global_app = Application(config)
    return _global_app


def get_application() -> Optional[Application]:
    """获取全局应用程序实例"""
    return _global_app


def run_application(config: Optional[ApplicationConfig] = None) -> int:
    """运行应用程序的便捷函数"""
    app = create_application(config)
    return app.run()


if __name__ == "__main__":
    # 直接运行时的入口点
    import sys
    
    config = ApplicationConfig(
        debug=True,
        log_level="DEBUG"
    )
    
    sys.exit(run_application(config))