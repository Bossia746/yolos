"""配置管理器"""

import os
import yaml
import json
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging
from copy import deepcopy
import threading
from dataclasses import dataclass, asdict
from datetime import datetime

from .event_bus import EventBus


@dataclass
class ConfigSource:
    """配置源信息"""
    name: str
    path: str
    priority: int  # 数字越大优先级越高
    last_modified: datetime
    checksum: str


class ConfigValidationError(Exception):
    """配置验证错误"""
    pass


class ConfigManager:
    """配置管理器
    
    支持多层级配置覆盖、动态更新、配置验证等功能
    """
    
    def __init__(self, event_bus: EventBus = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.event_bus = event_bus
        
        # 配置存储
        self._config: Dict[str, Any] = {}
        self._config_sources: List[ConfigSource] = []
        self._default_config: Dict[str, Any] = {}
        
        # 配置监听
        self._watchers: Dict[str, List[callable]] = {}
        self._file_watchers: Dict[str, Any] = {}  # 文件监听器
        
        # 线程安全
        self._lock = threading.RLock()
        
        # 配置验证规则
        self._validation_rules: Dict[str, Dict] = {}
        
        # 环境变量前缀
        self.env_prefix = "YOLOS_"
        
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