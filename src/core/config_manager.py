#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOS 配置管理系统
提供统一的配置加载、验证和管理功能

作者: YOLOS团队
版本: 2.0.0
更新时间: 2025-09-11
"""

import os
import yaml
import json
import toml
import threading
import hashlib
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Type, Callable, Pattern
from pathlib import Path
import logging
from copy import deepcopy
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
import re
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .exceptions import YOLOSException, ConfigurationError
try:
    from .event_bus import EventBus
except ImportError:
    EventBus = None


class ConfigValidationError(ConfigurationError):
    """配置验证错误"""
    pass


class ConfigSchemaError(ConfigurationError):
    """配置模式错误"""
    pass


class ConfigFormat(Enum):
    """配置文件格式"""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    INI = "ini"
    ENV = "env"


class ConfigSourceType(Enum):
    """配置源类型"""
    FILE = "file"
    ENVIRONMENT = "environment"
    COMMAND_LINE = "command_line"
    REMOTE = "remote"
    DATABASE = "database"


@dataclass
class ConfigSource:
    """配置源信息"""
    name: str
    source_type: ConfigSourceType
    location: str
    format: ConfigFormat
    priority: int = 0  # 数字越大优先级越高
    watch: bool = False
    encrypted: bool = False
    last_modified: Optional[datetime] = None
    checksum: Optional[str] = None


@dataclass
class ConfigProperty:
    """配置属性定义"""
    type: Type
    description: str = ""
    required: bool = False
    default: Any = None
    validator: Optional[Callable[[Any], bool]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[Pattern] = None
    choices: Optional[List[Any]] = None
    nested_schema: Optional['ConfigSchema'] = None

@dataclass
class ConfigSchema:
    """增强的配置模式定义"""
    name: str
    description: str
    version: str = "1.0.0"
    properties: Dict[str, ConfigProperty] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)
    defaults: Dict[str, Any] = field(default_factory=dict)
    validators: Dict[str, Callable] = field(default_factory=dict)
    
    def add_property(self, name: str, prop: ConfigProperty) -> 'ConfigSchema':
        """添加属性定义"""
        self.properties[name] = prop
        if prop.required:
            self.required.append(name)
        if prop.default is not None:
            self.defaults[name] = prop.default
        return self
    
    def validate_value(self, key: str, value: Any) -> bool:
        """验证单个值"""
        if key not in self.properties:
            return True  # 未定义的属性默认通过
        
        prop = self.properties[key]
        
        # 类型检查
        if not isinstance(value, prop.type):
            try:
                # 尝试类型转换
                value = prop.type(value)
            except (ValueError, TypeError):
                return False
        
        # 范围检查
        if prop.min_value is not None and value < prop.min_value:
            return False
        if prop.max_value is not None and value > prop.max_value:
            return False
        
        # 模式匹配
        if prop.pattern is not None and isinstance(value, str):
            if not prop.pattern.match(value):
                return False
        
        # 选择检查
        if prop.choices is not None and value not in prop.choices:
            return False
        
        # 自定义验证器
        if prop.validator is not None:
            return prop.validator(value)
        
        return True


class IConfigProvider(ABC):
    """配置提供者接口"""
    
    @abstractmethod
    def load_config(self, source: ConfigSource) -> Dict[str, Any]:
        """加载配置"""
        pass
    
    @abstractmethod
    def save_config(self, source: ConfigSource, config: Dict[str, Any]) -> bool:
        """保存配置"""
        pass
    
    @abstractmethod
    def watch_config(self, source: ConfigSource, callback: Callable) -> bool:
        """监视配置变化"""
        pass


class ConfigFileWatcher(FileSystemEventHandler):
    """配置文件监控处理器"""
    
    def __init__(self, config_manager: 'ConfigManager', source: ConfigSource):
        super().__init__()
        self.config_manager = config_manager
        self.source = source
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def on_modified(self, event):
        """文件修改事件处理"""
        if not event.is_directory and event.src_path == self.source.location:
            self.logger.info(f"Config file {self.source.name} modified, reloading...")
            try:
                # 延迟一点时间确保文件写入完成
                time.sleep(0.1)
                self.config_manager._reload_config_source(self.source)
            except Exception as e:
                self.logger.error(f"Failed to reload config {self.source.name}: {e}")


class FileConfigProvider(IConfigProvider):
    """文件配置提供者"""
    
    def __init__(self):
        self.observers: Dict[str, Observer] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_config(self, source: ConfigSource) -> Dict[str, Any]:
        """从文件加载配置"""
        if not os.path.exists(source.location):
            return {}
        
        try:
            with open(source.location, 'r', encoding='utf-8') as f:
                if source.format == ConfigFormat.JSON:
                    return json.load(f)
                elif source.format == ConfigFormat.YAML:
                    return yaml.safe_load(f) or {}
                elif source.format == ConfigFormat.TOML:
                    return toml.load(f)
                else:
                    raise ConfigurationError(f"Unsupported format: {source.format}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load config from {source.location}: {e}")
    
    def save_config(self, source: ConfigSource, config: Dict[str, Any]) -> bool:
        """保存配置到文件"""
        try:
            os.makedirs(os.path.dirname(source.location), exist_ok=True)
            
            with open(source.location, 'w', encoding='utf-8') as f:
                if source.format == ConfigFormat.JSON:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                elif source.format == ConfigFormat.YAML:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                elif source.format == ConfigFormat.TOML:
                    toml.dump(config, f)
                else:
                    raise ConfigurationError(f"Unsupported format: {source.format}")
            
            return True
        except Exception as e:
            raise ConfigurationError(f"Failed to save config to {source.location}: {e}")
    
    def watch_config(self, source: ConfigSource, callback: Callable) -> bool:
        """监视文件配置变化"""
        if not os.path.exists(source.location):
            self.logger.warning(f"Config file {source.location} does not exist, cannot watch")
            return False
        
        try:
            # 创建观察者
            observer = Observer()
            
            # 创建事件处理器
            event_handler = ConfigFileWatcher(callback, source)
            
            # 监控文件所在目录
            watch_dir = os.path.dirname(os.path.abspath(source.location))
            observer.schedule(event_handler, watch_dir, recursive=False)
            
            # 启动观察者
            observer.start()
            
            # 保存观察者引用
            self.observers[source.name] = observer
            
            self.logger.info(f"Started watching config file: {source.location}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start watching config file {source.location}: {e}")
            return False
    
    def stop_watching(self, source_name: str) -> bool:
        """停止监控配置文件"""
        if source_name in self.observers:
            try:
                observer = self.observers[source_name]
                observer.stop()
                observer.join()
                del self.observers[source_name]
                self.logger.info(f"Stopped watching config: {source_name}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to stop watching config {source_name}: {e}")
                return False
        return False
    
    def stop_all_watching(self):
        """停止所有文件监控"""
        for source_name in list(self.observers.keys()):
            self.stop_watching(source_name)


class EnvironmentConfigProvider(IConfigProvider):
    """环境变量配置提供者"""
    
    def load_config(self, source: ConfigSource) -> Dict[str, Any]:
        """从环境变量加载配置"""
        config = {}
        prefix = source.location.upper() + "_" if source.location else ""
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                
                try:
                    config[config_key] = json.loads(value)
                except json.JSONDecodeError:
                    if value.lower() in ('true', 'false'):
                        config[config_key] = value.lower() == 'true'
                    elif value.isdigit():
                        config[config_key] = int(value)
                    elif '.' in value and value.replace('.', '').isdigit():
                        config[config_key] = float(value)
                    else:
                        config[config_key] = value
        
        return config
    
    def save_config(self, source: ConfigSource, config: Dict[str, Any]) -> bool:
        """保存配置到环境变量"""
        prefix = source.location.upper() + "_" if source.location else ""
        
        for key, value in config.items():
            env_key = prefix + key.upper()
            if isinstance(value, (dict, list)):
                os.environ[env_key] = json.dumps(value)
            else:
                os.environ[env_key] = str(value)
        
        return True
    
    def watch_config(self, source: ConfigSource, callback: Callable) -> bool:
        """环境变量不支持监视"""
        return False


class ConfigManager:
    """配置管理器
    
    支持多层级配置覆盖、动态更新、配置验证、热重载等功能
    """
    
    def __init__(self, event_bus: EventBus = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.event_bus = event_bus
        
        # 配置存储
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._sources: List[ConfigSource] = []
        self._schemas: Dict[str, ConfigSchema] = {}
        self._providers: Dict[ConfigSourceType, IConfigProvider] = {
            ConfigSourceType.FILE: FileConfigProvider(),
            ConfigSourceType.ENVIRONMENT: EnvironmentConfigProvider()
        }
        
        # 配置监听和热重载
        self._watchers: Dict[str, List[Callable]] = {}
        self._hot_reload_enabled = True
        self._config_checksums: Dict[str, str] = {}
        self._lock = threading.RLock()
        
        # 添加默认配置源
        self._add_default_sources()
    
    def _add_default_sources(self):
        """添加默认配置源"""
        # 主配置文件
        self.add_source(ConfigSource(
            name="main_config",
            source_type=ConfigSourceType.FILE,
            location="config/config.yaml",
            format=ConfigFormat.YAML,
            priority=100
        ))
        
        # 环境特定配置
        env = os.getenv('YOLOS_ENV', 'development')
        self.add_source(ConfigSource(
            name=f"{env}_config",
            source_type=ConfigSourceType.FILE,
            location=f"config/{env}.yaml",
            format=ConfigFormat.YAML,
            priority=200
        ))
        
        # 环境变量
        self.add_source(ConfigSource(
            name="env_config",
            source_type=ConfigSourceType.ENVIRONMENT,
            location="YOLOS",
            format=ConfigFormat.ENV,
            priority=300
        ))
        
        # 本地覆盖配置
        self.add_source(ConfigSource(
            name="local_config",
            source_type=ConfigSourceType.FILE,
            location="config/local.yaml",
            format=ConfigFormat.YAML,
            priority=400
        ))
    
    def add_source(self, source: ConfigSource):
        """添加配置源"""
        with self._lock:
            self._sources.append(source)
            # 按优先级排序
            self._sources.sort(key=lambda s: s.priority)
    
    def register_schema(self, schema: ConfigSchema):
        """注册配置模式"""
        with self._lock:
            self._schemas[schema.name] = schema
    
    def load_all_configs(self) -> Dict[str, Any]:
        """加载所有配置"""
        with self._lock:
            merged_config = {}
            
            # 按优先级顺序加载配置
            for source in self._sources:
                try:
                    provider = self._providers.get(source.source_type)
                    if provider:
                        config = provider.load_config(source)
                        if config:
                            # 计算并存储配置校验和
                            checksum = self._calculate_config_checksum(config)
                            self._config_checksums[source.name] = checksum
                            
                            # 深度合并配置
                            merged_config = self._deep_merge(merged_config, config)
                except Exception as e:
                    self.logger.warning(f"Failed to load config from {source.name}: {e}")
            
            # 应用默认值
            for schema in self._schemas.values():
                if schema.defaults:
                    default_config = deepcopy(schema.defaults)
                    merged_config = self._deep_merge(default_config, merged_config)
            
            # 验证配置
            self._validate_config(merged_config)
            
            self._configs['merged'] = merged_config
            
            # 如果启用了热重载，开始监控文件
            if self._hot_reload_enabled:
                for source in self._sources:
                    if source.watch and source.source_type == ConfigSourceType.FILE:
                        provider = self._providers[source.source_type]
                        if hasattr(provider, 'watch_config'):
                            provider.watch_config(source, self)
            
            return merged_config
    
    def get_config(self, key: str = None, default: Any = None) -> Any:
        """获取配置值"""
        with self._lock:
            if 'merged' not in self._configs:
                self.load_all_configs()
            
            config = self._configs['merged']
            
            if key is None:
                return config
            
            # 支持点号分隔的键路径
            keys = key.split('.')
            value = config
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
    
    def set_config(self, key: str, value: Any, persist: bool = False):
        """设置配置值"""
        with self._lock:
            if 'merged' not in self._configs:
                self.load_all_configs()
            
            # 更新内存中的配置
            keys = key.split('.')
            config = self._configs['merged']
            
            # 导航到目标位置
            current = config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            current[keys[-1]] = value
            
            # 如果需要持久化
            if persist:
                self._persist_config_change(key, value)
            
            # 通知监听器
            self._notify_watchers(key, value)
    
    def _persist_config_change(self, key: str, value: Any):
        """持久化配置变更"""
        # 查找可写的配置源（通常是本地配置文件）
        writable_source = None
        for source in reversed(self._sources):
            if source.source_type == ConfigSourceType.FILE and source.name == "local_config":
                writable_source = source
                break
        
        if writable_source:
            try:
                provider = self._providers[writable_source.source_type]
                
                # 加载现有配置
                existing_config = provider.load_config(writable_source)
                
                # 更新配置
                keys = key.split('.')
                current = existing_config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                
                current[keys[-1]] = value
                
                # 保存配置
                provider.save_config(writable_source, existing_config)
                
            except Exception as e:
                self.logger.warning(f"Failed to persist config change: {e}")
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并字典"""
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    def _validate_config(self, config: Dict[str, Any]):
        """增强的配置验证"""
        validation_errors = []
        
        for schema_name, schema in self._schemas.items():
            try:
                # 检查必需字段
                for required_field in schema.required:
                    if required_field not in config:
                        validation_errors.append(f"Required field '{required_field}' missing in {schema_name}")
                
                # 验证属性
                for field_name, field_value in config.items():
                    if field_name in schema.properties:
                        if not schema.validate_value(field_name, field_value):
                            prop = schema.properties[field_name]
                            validation_errors.append(
                                f"Validation failed for field '{field_name}' in {schema_name}: "
                                f"value '{field_value}' does not meet requirements for type {prop.type.__name__}"
                            )
                
                # 执行自定义验证器
                for field, validator in schema.validators.items():
                    if field in config:
                        try:
                            if not validator(config[field]):
                                validation_errors.append(f"Custom validation failed for field '{field}' in {schema_name}")
                        except Exception as e:
                            validation_errors.append(f"Validator error for field '{field}' in {schema_name}: {e}")
                            
            except Exception as e:
                validation_errors.append(f"Schema validation error for {schema_name}: {e}")
        
        if validation_errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in validation_errors)
            raise ConfigValidationError(error_msg)
    
    def _calculate_config_checksum(self, config: Dict[str, Any]) -> str:
        """计算配置的校验和"""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _reload_config_source(self, source: ConfigSource):
        """重新加载指定配置源"""
        try:
            provider = self._providers.get(source.source_type)
            if not provider:
                return
            
            # 加载新配置
            new_config = provider.load_config(source)
            if not new_config:
                return
            
            # 计算校验和
            new_checksum = self._calculate_config_checksum(new_config)
            old_checksum = self._config_checksums.get(source.name)
            
            # 如果配置没有变化，跳过
            if new_checksum == old_checksum:
                return
            
            self.logger.info(f"Config source {source.name} changed, reloading all configs")
            
            # 更新校验和
            self._config_checksums[source.name] = new_checksum
            
            # 重新加载所有配置
            old_config = self._configs.get('merged', {})
            new_merged_config = self.load_all_configs()
            
            # 检测变化并通知监听器
            self._notify_config_changes(old_config, new_merged_config)
            
            # 发送事件
            if self.event_bus:
                self.event_bus.emit('config.reloaded', {
                    'source': source.name,
                    'old_config': old_config,
                    'new_config': new_merged_config
                })
                
        except Exception as e:
            self.logger.error(f"Failed to reload config source {source.name}: {e}")
    
    def _notify_config_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any]):
        """通知配置变化"""
        def find_changes(old_dict, new_dict, prefix=""):
            changes = []
            
            # 检查新增和修改的键
            for key, new_value in new_dict.items():
                full_key = f"{prefix}.{key}" if prefix else key
                
                if key not in old_dict:
                    changes.append((full_key, None, new_value))
                elif old_dict[key] != new_value:
                    if isinstance(old_dict[key], dict) and isinstance(new_value, dict):
                        changes.extend(find_changes(old_dict[key], new_value, full_key))
                    else:
                        changes.append((full_key, old_dict[key], new_value))
            
            # 检查删除的键
            for key, old_value in old_dict.items():
                if key not in new_dict:
                    full_key = f"{prefix}.{key}" if prefix else key
                    changes.append((full_key, old_value, None))
            
            return changes
        
        changes = find_changes(old_config, new_config)
        
        for key, old_value, new_value in changes:
            self._notify_watchers(key, new_value, old_value)
    
    def enable_hot_reload(self, enabled: bool = True):
        """启用或禁用热重载"""
        self._hot_reload_enabled = enabled
        
        if enabled:
            # 为所有支持监控的配置源启动监控
            for source in self._sources:
                if source.watch and source.source_type == ConfigSourceType.FILE:
                    provider = self._providers[source.source_type]
                    provider.watch_config(source, self)
        else:
            # 停止所有监控
            file_provider = self._providers.get(ConfigSourceType.FILE)
            if hasattr(file_provider, 'stop_all_watching'):
                file_provider.stop_all_watching()
    
    def watch_config(self, key: str, callback: Callable[[str, Any], None]):
        """监视配置变化"""
        with self._lock:
            if key not in self._watchers:
                self._watchers[key] = []
            self._watchers[key].append(callback)
    
    def _notify_watchers(self, key: str, value: Any, old_value: Any = None):
        """通知配置监听器"""
        # 通知精确匹配的监听器
        if key in self._watchers:
            for callback in self._watchers[key]:
                try:
                    callback(key, value)
                except Exception as e:
                    self.logger.error(f"Config watcher error: {e}")
        
        # 通知通配符监听器
        for watch_key, callbacks in self._watchers.items():
            if watch_key.endswith('*') and key.startswith(watch_key[:-1]):
                for callback in callbacks:
                    try:
                        callback(key, value)
                    except Exception as e:
                        self.logger.error(f"Config watcher error: {e}")
    
    def reload_config(self):
        """重新加载配置"""
        with self._lock:
            old_config = self._configs.get('merged', {})
            new_config = self.load_all_configs()
            
            # 检查变化并通知
            self._detect_changes(old_config, new_config)
    
    def _detect_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any], prefix: str = ""):
        """检测配置变化"""
        # 检查新增和修改的键
        for key, value in new_config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if key not in old_config:
                # 新增的键
                self._notify_watchers(full_key, value)
            elif old_config[key] != value:
                if isinstance(value, dict) and isinstance(old_config[key], dict):
                    # 递归检查嵌套字典
                    self._detect_changes(old_config[key], value, full_key)
                else:
                    # 修改的键
                    self._notify_watchers(full_key, value)
        
        # 检查删除的键
        for key in old_config:
            if key not in new_config:
                full_key = f"{prefix}.{key}" if prefix else key
                self._notify_watchers(full_key, None)
    
    def export_config(self, format: ConfigFormat = ConfigFormat.YAML) -> str:
        """导出配置"""
        config = self.get_config()
        
        if format == ConfigFormat.JSON:
            return json.dumps(config, indent=2, ensure_ascii=False)
        elif format == ConfigFormat.YAML:
            return yaml.dump(config, default_flow_style=False, allow_unicode=True)
        elif format == ConfigFormat.TOML:
            return toml.dumps(config)
        else:
            raise ConfigurationError(f"Unsupported export format: {format}")


# 全局配置管理器实例
global_config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    """获取全局配置管理器"""
    return global_config_manager


def get_config(key: str = None, default: Any = None) -> Any:
    """获取配置的便捷函数"""
    return global_config_manager.get_config(key, default)


def set_config(key: str, value: Any, persist: bool = False):
    """设置配置的便捷函数"""
    global_config_manager.set_config(key, value, persist)


def watch_config(key: str, callback: Callable[[str, Any], None]):
    """监视配置的便捷函数"""
    global_config_manager.watch_config(key, callback)


class ExtendedConfigManager(ConfigManager):
    """扩展配置管理器"""
    
    def __init__(self):
        super().__init__()
        # 支持的配置文件格式
        self._supported_formats = {'.yaml', '.yml', '.json'}
        # 初始化默认配置
        self._init_default_config()
    
    def _init_default_config(self) -> None:
        """初始化默认配置"""
        self._default_config = {
            'system': {
                'platform': 'auto',
                'log_level': 'INFO',
                'max_memory_usage': '80%',
                'debug': False
            },
            'plugins': {
                'enabled': [],
                'auto_discover': True,
                'plugin_directories': [
                    'src/plugins',
                    'plugins',
                    'src/recognition'
                ]
            },
            'hardware': {
                'camera': {
                    'type': 'auto',
                    'resolution': [640, 480],
                    'fps': 30
                },
                'compute': {
                    'device': 'auto',
                    'batch_size': 1,
                    'num_threads': 4
                }
            },
            'performance': {
                'detection_interval': 1,
                'max_concurrent_tasks': 2,
                'memory_optimization': True,
                'model_caching': True
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/yolos.log',
                'max_size': '10MB',
                'backup_count': 5
            }
        }
        
        # 设置验证规则
        self._setup_validation_rules()
    
    def _setup_validation_rules(self) -> None:
        """设置配置验证规则"""
        self._validation_rules = {
            'system.log_level': {
                'type': str,
                'choices': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            },
            'system.max_memory_usage': {
                'type': [str, int],
                'pattern': r'^\d+%?$'  # 支持百分比或绝对值
            },
            'hardware.camera.resolution': {
                'type': list,
                'length': 2,
                'item_type': int,
                'min_value': 1
            },
            'hardware.camera.fps': {
                'type': int,
                'min_value': 1,
                'max_value': 120
            },
            'performance.detection_interval': {
                'type': int,
                'min_value': 1
            },
            'performance.max_concurrent_tasks': {
                'type': int,
                'min_value': 1,
                'max_value': 16
            }
        }
    
    def load_config(self, config_path: str, priority: int = 0) -> bool:
        """加载配置文件
        
        Args:
            config_path: 配置文件路径
            priority: 优先级（数字越大优先级越高）
            
        Returns:
            bool: 加载是否成功
        """
        try:
            path = Path(config_path)
            if not path.exists():
                self.logger.warning(f"Config file not found: {config_path}")
                return False
            
            # 检查文件格式
            if path.suffix.lower() not in self._supported_formats:
                self.logger.error(f"Unsupported config format: {path.suffix}")
                return False
            
            # 读取配置文件
            config_data = self._read_config_file(path)
            if config_data is None:
                return False
            
            with self._lock:
                # 创建配置源
                source = ConfigSource(
                    name=path.name,
                    path=str(path),
                    priority=priority,
                    last_modified=datetime.fromtimestamp(path.stat().st_mtime),
                    checksum=self._calculate_checksum(path)
                )
                
                # 移除同路径的旧配置源
                self._config_sources = [s for s in self._config_sources if s.path != str(path)]
                
                # 添加新配置源
                self._config_sources.append(source)
                self._config_sources.sort(key=lambda x: x.priority)
                
                # 重新构建配置
                self._rebuild_config()
                
                self.logger.info(f"Loaded config from {config_path} with priority {priority}")
                
                # 发送配置更新事件
                if self.event_bus:
                    self.event_bus.emit('config_loaded', {
                        'source': source.name,
                        'path': config_path,
                        'priority': priority
                    })
                
                return True
        
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
            return False
    
    def _read_config_file(self, path: Path) -> Optional[Dict]:
        """读取配置文件
        
        Args:
            path: 配置文件路径
            
        Returns:
            Optional[Dict]: 配置数据
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                if path.suffix.lower() in {'.yaml', '.yml'}:
                    return yaml.safe_load(f)
                elif path.suffix.lower() == '.json':
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to read config file {path}: {e}")
        
        return None
    
    def _calculate_checksum(self, path: Path) -> str:
        """计算文件校验和
        
        Args:
            path: 文件路径
            
        Returns:
            str: 校验和
        """
        import hashlib
        
        try:
            with open(path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    def _rebuild_config(self) -> None:
        """重新构建配置"""
        # 从默认配置开始
        new_config = deepcopy(self._default_config)
        
        # 按优先级应用配置源
        for source in self._config_sources:
            try:
                config_data = self._read_config_file(Path(source.path))
                if config_data:
                    self._deep_merge(new_config, config_data)
            except Exception as e:
                self.logger.error(f"Failed to apply config from {source.path}: {e}")
        
        # 应用环境变量
        self._apply_env_vars(new_config)
        
        # 验证配置
        if self._validate_config(new_config):
            old_config = self._config
            self._config = new_config
            
            # 检测配置变化
            changes = self._detect_changes(old_config, new_config)
            if changes:
                self._notify_config_changes(changes)
        else:
            self.logger.error("Config validation failed, keeping old config")
    
    def _deep_merge(self, target: Dict, source: Dict) -> None:
        """深度合并字典
        
        Args:
            target: 目标字典
            source: 源字典
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = deepcopy(value)
    
    def _apply_env_vars(self, config: Dict) -> None:
        """应用环境变量
        
        Args:
            config: 配置字典
        """
        for key, value in os.environ.items():
            if key.startswith(self.env_prefix):
                config_key = key[len(self.env_prefix):].lower().replace('_', '.')
                self._set_nested_value(config, config_key, self._parse_env_value(value))
    
    def _parse_env_value(self, value: str) -> Any:
        """解析环境变量值
        
        Args:
            value: 环境变量值
            
        Returns:
            Any: 解析后的值
        """
        # 尝试解析为JSON
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # 尝试解析为布尔值
        if value.lower() in {'true', 'false'}:
            return value.lower() == 'true'
        
        # 尝试解析为数字
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # 返回字符串
        return value
    
    def _set_nested_value(self, config: Dict, key_path: str, value: Any) -> None:
        """设置嵌套配置值
        
        Args:
            config: 配置字典
            key_path: 键路径（用.分隔）
            value: 值
        """
        keys = key_path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _validate_config(self, config: Dict) -> bool:
        """验证配置
        
        Args:
            config: 配置字典
            
        Returns:
            bool: 验证是否通过
        """
        try:
            for key_path, rules in self._validation_rules.items():
                value = self._get_nested_value(config, key_path)
                if value is not None:
                    self._validate_value(key_path, value, rules)
            return True
        except ConfigValidationError as e:
            self.logger.error(f"Config validation error: {e}")
            return False
    
    def _validate_value(self, key_path: str, value: Any, rules: Dict) -> None:
        """验证单个配置值
        
        Args:
            key_path: 键路径
            value: 值
            rules: 验证规则
        """
        # 类型检查
        if 'type' in rules:
            expected_types = rules['type'] if isinstance(rules['type'], list) else [rules['type']]
            if not any(isinstance(value, t) for t in expected_types):
                raise ConfigValidationError(f"{key_path}: expected {expected_types}, got {type(value)}")
        
        # 选择检查
        if 'choices' in rules and value not in rules['choices']:
            raise ConfigValidationError(f"{key_path}: value '{value}' not in {rules['choices']}")
        
        # 数值范围检查
        if isinstance(value, (int, float)):
            if 'min_value' in rules and value < rules['min_value']:
                raise ConfigValidationError(f"{key_path}: value {value} < {rules['min_value']}")
            if 'max_value' in rules and value > rules['max_value']:
                raise ConfigValidationError(f"{key_path}: value {value} > {rules['max_value']}")
        
        # 列表长度检查
        if isinstance(value, list) and 'length' in rules and len(value) != rules['length']:
            raise ConfigValidationError(f"{key_path}: expected length {rules['length']}, got {len(value)}")
        
        # 正则表达式检查
        if 'pattern' in rules and isinstance(value, str):
            import re
            if not re.match(rules['pattern'], value):
                raise ConfigValidationError(f"{key_path}: value '{value}' doesn't match pattern {rules['pattern']}")
    
    def _get_nested_value(self, config: Dict, key_path: str) -> Any:
        """获取嵌套配置值
        
        Args:
            config: 配置字典
            key_path: 键路径（用.分隔）
            
        Returns:
            Any: 配置值
        """
        keys = key_path.split('.')
        current = config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def _detect_changes(self, old_config: Dict, new_config: Dict, prefix: str = '') -> List[str]:
        """检测配置变化
        
        Args:
            old_config: 旧配置
            new_config: 新配置
            prefix: 键前缀
            
        Returns:
            List[str]: 变化的键列表
        """
        changes = []
        
        # 检查新增和修改的键
        for key, value in new_config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if key not in old_config:
                changes.append(full_key)
            elif isinstance(value, dict) and isinstance(old_config[key], dict):
                changes.extend(self._detect_changes(old_config[key], value, full_key))
            elif old_config[key] != value:
                changes.append(full_key)
        
        # 检查删除的键
        for key in old_config:
            if key not in new_config:
                full_key = f"{prefix}.{key}" if prefix else key
                changes.append(full_key)
        
        return changes
    
    def _notify_config_changes(self, changes: List[str]) -> None:
        """通知配置变化
        
        Args:
            changes: 变化的键列表
        """
        self.logger.info(f"Config changed: {changes}")
        
        # 通知监听器
        for key_pattern, callbacks in self._watchers.items():
            for change in changes:
                if self._match_pattern(change, key_pattern):
                    for callback in callbacks:
                        try:
                            callback(change, self.get(change))
                        except Exception as e:
                            self.logger.error(f"Error in config watcher callback: {e}")
        
        # 发送事件
        if self.event_bus:
            self.event_bus.emit('config_changed', {
                'changes': changes,
                'config': self._config
            })
    
    def _match_pattern(self, key: str, pattern: str) -> bool:
        """匹配键模式
        
        Args:
            key: 键
            pattern: 模式（支持通配符*）
            
        Returns:
            bool: 是否匹配
        """
        import fnmatch
        return fnmatch.fnmatch(key, pattern)
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值
        
        Args:
            key: 配置键（支持点分隔的嵌套键）
            default: 默认值
            
        Returns:
            Any: 配置值
        """
        with self._lock:
            value = self._get_nested_value(self._config, key)
            return value if value is not None else default
    
    def set(self, key: str, value: Any, persist: bool = False) -> None:
        """设置配置值
        
        Args:
            key: 配置键
            value: 配置值
            persist: 是否持久化到文件
        """
        with self._lock:
            old_value = self.get(key)
            self._set_nested_value(self._config, key, value)
            
            if old_value != value:
                self._notify_config_changes([key])
            
            if persist:
                self._persist_config()
    
    def watch(self, key_pattern: str, callback: callable) -> None:
        """监听配置变化
        
        Args:
            key_pattern: 键模式（支持通配符*）
            callback: 回调函数 callback(key, value)
        """
        if key_pattern not in self._watchers:
            self._watchers[key_pattern] = []
        self._watchers[key_pattern].append(callback)
    
    def unwatch(self, key_pattern: str, callback: callable = None) -> None:
        """取消监听配置变化
        
        Args:
            key_pattern: 键模式
            callback: 回调函数，如果为None则移除所有回调
        """
        if key_pattern in self._watchers:
            if callback is None:
                del self._watchers[key_pattern]
            else:
                self._watchers[key_pattern] = [cb for cb in self._watchers[key_pattern] if cb != callback]
    
    def _persist_config(self) -> None:
        """持久化配置到文件"""
        # 简化实现，实际应该选择合适的配置文件进行持久化
        config_file = Path('config/runtime_config.yaml')
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
            self.logger.info(f"Config persisted to {config_file}")
        except Exception as e:
            self.logger.error(f"Failed to persist config: {e}")
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有配置
        
        Returns:
            Dict[str, Any]: 完整配置字典
        """
        with self._lock:
            return deepcopy(self._config)
    
    def get_sources(self) -> List[Dict]:
        """获取配置源信息
        
        Returns:
            List[Dict]: 配置源列表
        """
        with self._lock:
            return [asdict(source) for source in self._config_sources]
    
    def reload(self) -> None:
        """重新加载所有配置"""
        with self._lock:
            self.logger.info("Reloading all configurations")
            self._rebuild_config()
    
    def reset_to_defaults(self) -> None:
        """重置为默认配置"""
        with self._lock:
            old_config = self._config
            self._config = deepcopy(self._default_config)
            
            changes = self._detect_changes(old_config, self._config)
            if changes:
                self._notify_config_changes(changes)
            
            self.logger.info("Config reset to defaults")