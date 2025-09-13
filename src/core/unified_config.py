"""统一配置管理系统

提供分层配置管理，支持环境特定配置和动态配置更新
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging
from dataclasses import dataclass, field
from enum import Enum
import threading
from copy import deepcopy
import time
import re


class ConfigFormat(Enum):
    """配置文件格式"""
    YAML = "yaml"
    JSON = "json"
    TOML = "toml"


class ConfigLevel(Enum):
    """配置层级"""
    DEFAULT = 1      # 默认配置
    SYSTEM = 2       # 系统配置
    USER = 3         # 用户配置
    ENVIRONMENT = 4  # 环境配置
    RUNTIME = 5      # 运行时配置


@dataclass
class ConfigSource:
    """配置源信息"""
    path: str
    format: ConfigFormat
    level: ConfigLevel
    required: bool = False
    watch: bool = False
    last_modified: Optional[float] = None


class ConfigChangeEvent:
    """配置变更事件"""
    
    def __init__(self, key: str, old_value: Any, new_value: Any, source: str):
        self.key = key
        self.old_value = old_value
        self.new_value = new_value
        self.source = source
        self.timestamp = time.time()


class UnifiedConfigManager:
    """统一配置管理器
    
    支持多层级配置、环境特定配置、动态更新和配置监听
    """
    
    def __init__(self, base_dir: str = None, environment: str = None):
        self.logger = logging.getLogger(__name__)
        self.base_dir = Path(base_dir or "config")
        self.environment = environment or os.getenv('YOLOS_ENV', 'development')
        
        # 配置存储
        self._configs: Dict[ConfigLevel, Dict[str, Any]] = {
            level: {} for level in ConfigLevel
        }
        
        # 配置源管理
        self._sources: List[ConfigSource] = []
        self._merged_config: Dict[str, Any] = {}
        
        # 变更监听
        self._change_listeners: List[callable] = []
        self._lock = threading.RLock()
        
        # 初始化配置
        self._initialize_default_sources()
        self._load_all_configs()
    
    def _initialize_default_sources(self):
        """初始化默认配置源"""
        # 默认配置
        self.add_source(
            path=str(self.base_dir / "default_config.yaml"),
            level=ConfigLevel.DEFAULT,
            required=True
        )
        
        # 系统配置
        self.add_source(
            path=str(self.base_dir / "system_config.yaml"),
            level=ConfigLevel.SYSTEM,
            required=False
        )
        
        # 环境特定配置
        env_config_path = self.base_dir / f"{self.environment}_config.yaml"
        if env_config_path.exists():
            self.add_source(
                path=str(env_config_path),
                level=ConfigLevel.ENVIRONMENT,
                required=False
            )
        
        # 用户配置
        user_config_path = self.base_dir / "user_config.yaml"
        if user_config_path.exists():
            self.add_source(
                path=str(user_config_path),
                level=ConfigLevel.USER,
                required=False
            )
    
    def add_source(self, path: str, level: ConfigLevel, required: bool = False, 
                   watch: bool = False, format: ConfigFormat = None) -> bool:
        """添加配置源
        
        Args:
            path: 配置文件路径
            level: 配置层级
            required: 是否必需
            watch: 是否监听文件变化
            format: 配置文件格式
            
        Returns:
            bool: 添加是否成功
        """
        try:
            # 自动检测格式
            if format is None:
                ext = Path(path).suffix.lower()
                if ext in ['.yaml', '.yml']:
                    format = ConfigFormat.YAML
                elif ext == '.json':
                    format = ConfigFormat.JSON
                elif ext == '.toml':
                    format = ConfigFormat.TOML
                else:
                    self.logger.warning(f"Unknown config format for {path}, assuming YAML")
                    format = ConfigFormat.YAML
            
            source = ConfigSource(
                path=path,
                format=format,
                level=level,
                required=required,
                watch=watch
            )
            
            self._sources.append(source)
            self.logger.info(f"Added config source: {path} (level: {level.name})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add config source {path}: {e}")
            return False
    
    def _load_all_configs(self):
        """加载所有配置"""
        with self._lock:
            # 按层级顺序加载
            for source in sorted(self._sources, key=lambda s: s.level.value):
                self._load_config_source(source)
            
            # 合并配置
            self._merge_configs()
    
    def _load_config_source(self, source: ConfigSource) -> bool:
        """加载单个配置源
        
        Args:
            source: 配置源
            
        Returns:
            bool: 加载是否成功
        """
        try:
            if not os.path.exists(source.path):
                if source.required:
                    raise FileNotFoundError(f"Required config file not found: {source.path}")
                else:
                    self.logger.debug(f"Optional config file not found: {source.path}")
                    return False
            
            # 检查文件修改时间
            mtime = os.path.getmtime(source.path)
            if source.last_modified and mtime <= source.last_modified:
                return True  # 文件未修改
            
            # 加载配置
            with open(source.path, 'r', encoding='utf-8') as f:
                if source.format == ConfigFormat.YAML:
                    config_data = yaml.safe_load(f) or {}
                elif source.format == ConfigFormat.JSON:
                    config_data = json.load(f)
                elif source.format == ConfigFormat.TOML:
                    import toml
                    config_data = toml.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {source.format}")
            
            # 应用环境变量替换
            config_data = self._apply_env_vars(config_data)
            
            # 存储配置
            self._configs[source.level] = config_data
            source.last_modified = mtime
            
            self.logger.debug(f"Loaded config from {source.path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load config from {source.path}: {e}")
            if source.required:
                raise
            return False
    
    def _merge_configs(self):
        """合并所有层级的配置"""
        merged = {}
        
        # 按层级顺序合并（低层级优先，高层级覆盖）
        for level in ConfigLevel:
            if level in self._configs:
                merged = self._deep_merge(merged, self._configs[level])
        
        old_config = deepcopy(self._merged_config)
        self._merged_config = merged
        
        # 触发变更事件
        self._notify_config_changes(old_config, merged)
    
    def _apply_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """应用环境变量替换"""
        def replace_env_vars(obj):
            if isinstance(obj, dict):
                return {k: replace_env_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_env_vars(item) for item in obj]
            elif isinstance(obj, str):
                # 支持 ${VAR} 和 ${VAR:default} 格式
                pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
                
                def replacer(match):
                    var_name = match.group(1)
                    default_value = match.group(2) if match.group(2) is not None else ''
                    return os.getenv(var_name, default_value)
                
                return re.sub(pattern, replacer, obj)
            else:
                return obj
        
        return replace_env_vars(config)
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并字典
        
        Args:
            base: 基础字典
            override: 覆盖字典
            
        Returns:
            Dict[str, Any]: 合并后的字典
        """
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值
        
        Args:
            key: 配置键（支持点分隔的嵌套键）
            default: 默认值
            
        Returns:
            Any: 配置值
        """
        try:
            keys = key.split('.')
            value = self._merged_config
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
            
        except Exception as e:
            self.logger.error(f"Error getting config key {key}: {e}")
            return default
    
    def set(self, key: str, value: Any, level: ConfigLevel = ConfigLevel.RUNTIME) -> bool:
        """设置配置值
        
        Args:
            key: 配置键
            value: 配置值
            level: 配置层级
            
        Returns:
            bool: 设置是否成功
        """
        try:
            with self._lock:
                keys = key.split('.')
                config = self._configs.setdefault(level, {})
                
                # 导航到目标位置
                for k in keys[:-1]:
                    config = config.setdefault(k, {})
                
                old_value = config.get(keys[-1])
                config[keys[-1]] = value
                
                # 重新合并配置
                self._merge_configs()
                
                self.logger.debug(f"Set config {key} = {value} at level {level.name}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error setting config key {key}: {e}")
            return False
    
    def has(self, key: str) -> bool:
        """检查配置键是否存在
        
        Args:
            key: 配置键
            
        Returns:
            bool: 是否存在
        """
        return self.get(key, object()) is not object()
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """获取配置段
        
        Args:
            section: 段名称
            
        Returns:
            Dict[str, Any]: 配置段
        """
        return self.get(section, {})
    
    def add_change_listener(self, listener: callable):
        """添加配置变更监听器
        
        Args:
            listener: 监听器函数
        """
        self._change_listeners.append(listener)
    
    def remove_change_listener(self, listener: callable):
        """移除配置变更监听器
        
        Args:
            listener: 监听器函数
        """
        if listener in self._change_listeners:
            self._change_listeners.remove(listener)
    
    def _notify_config_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any]):
        """通知配置变更
        
        Args:
            old_config: 旧配置
            new_config: 新配置
        """
        changes = self._find_changes(old_config, new_config)
        
        for change in changes:
            for listener in self._change_listeners:
                try:
                    listener(change)
                except Exception as e:
                    self.logger.error(f"Error in config change listener: {e}")
    
    def _find_changes(self, old: Dict[str, Any], new: Dict[str, Any], prefix: str = "") -> List[ConfigChangeEvent]:
        """查找配置变更
        
        Args:
            old: 旧配置
            new: 新配置
            prefix: 键前缀
            
        Returns:
            List[ConfigChangeEvent]: 变更事件列表
        """
        changes = []
        
        # 检查新增和修改
        for key, new_value in new.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if key not in old:
                # 新增
                changes.append(ConfigChangeEvent(full_key, None, new_value, "config"))
            elif old[key] != new_value:
                if isinstance(old[key], dict) and isinstance(new_value, dict):
                    # 递归检查嵌套字典
                    changes.extend(self._find_changes(old[key], new_value, full_key))
                else:
                    # 修改
                    changes.append(ConfigChangeEvent(full_key, old[key], new_value, "config"))
        
        # 检查删除
        for key, old_value in old.items():
            if key not in new:
                full_key = f"{prefix}.{key}" if prefix else key
                changes.append(ConfigChangeEvent(full_key, old_value, None, "config"))
        
        return changes
    
    def reload(self) -> bool:
        """重新加载所有配置
        
        Returns:
            bool: 重新加载是否成功
        """
        try:
            self.logger.info("Reloading all configurations")
            self._load_all_configs()
            return True
        except Exception as e:
            self.logger.error(f"Failed to reload configurations: {e}")
            return False
    
    def save_runtime_config(self, path: str = None) -> bool:
        """保存运行时配置
        
        Args:
            path: 保存路径
            
        Returns:
            bool: 保存是否成功
        """
        try:
            if not path:
                path = str(self.base_dir / "runtime_config.yaml")
            
            runtime_config = self._configs.get(ConfigLevel.RUNTIME, {})
            
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(runtime_config, f, default_flow_style=False, allow_unicode=True)
            
            self.logger.info(f"Saved runtime config to {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save runtime config: {e}")
            return False
    
    def get_all_configs(self) -> Dict[str, Any]:
        """获取所有合并后的配置
        
        Returns:
            Dict[str, Any]: 所有配置
        """
        return deepcopy(self._merged_config)
    
    def get_config_info(self) -> Dict[str, Any]:
        """获取配置信息
        
        Returns:
            Dict[str, Any]: 配置信息
        """
        return {
            'environment': self.environment,
            'base_dir': str(self.base_dir),
            'sources': [
                {
                    'path': source.path,
                    'level': source.level.name,
                    'format': source.format.value,
                    'required': source.required,
                    'exists': os.path.exists(source.path)
                }
                for source in self._sources
            ],
            'total_keys': len(self._merged_config)
        }


# 全局配置管理器实例
_global_config_manager: Optional[UnifiedConfigManager] = None


def get_config_manager() -> UnifiedConfigManager:
    """获取全局配置管理器
    
    Returns:
        UnifiedConfigManager: 配置管理器实例
    """
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = UnifiedConfigManager()
    return _global_config_manager


def init_config_manager(base_dir: str = None, environment: str = None) -> UnifiedConfigManager:
    """初始化全局配置管理器
    
    Args:
        base_dir: 配置基础目录
        environment: 环境名称
        
    Returns:
        UnifiedConfigManager: 配置管理器实例
    """
    global _global_config_manager
    _global_config_manager = UnifiedConfigManager(base_dir, environment)
    return _global_config_manager


# 便捷函数
def get_config(key: str, default: Any = None) -> Any:
    """获取配置值"""
    return get_config_manager().get(key, default)


def set_config(key: str, value: Any, level: ConfigLevel = ConfigLevel.RUNTIME) -> bool:
    """设置配置值"""
    return get_config_manager().set(key, value, level)


def has_config(key: str) -> bool:
    """检查配置是否存在"""
    return get_config_manager().has(key)


def get_config_section(section: str) -> Dict[str, Any]:
    """获取配置段"""
    return get_config_manager().get_section(section)