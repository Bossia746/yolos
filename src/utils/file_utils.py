#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件工具 - 简化版本
用于AIoT兼容性测试
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class FileManager:
    """文件管理器"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
    
    def ensure_dir(self, dir_path: Union[str, Path]) -> Path:
        """确保目录存在"""
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def read_json(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """读取JSON文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"读取JSON文件失败 {file_path}: {e}")
            return {}
    
    def write_json(self, data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
        """写入JSON文件"""
        try:
            self.ensure_dir(Path(file_path).parent)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"写入JSON文件失败 {file_path}: {e}")
            return False
    
    def read_yaml(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """读取YAML文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"读取YAML文件失败 {file_path}: {e}")
            return {}
    
    def write_yaml(self, data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
        """写入YAML文件"""
        try:
            self.ensure_dir(Path(file_path).parent)
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            return True
        except Exception as e:
            logger.error(f"写入YAML文件失败 {file_path}: {e}")
            return False
    
    def list_files(self, 
                   directory: Union[str, Path], 
                   pattern: str = "*",
                   recursive: bool = False) -> List[Path]:
        """列出文件"""
        try:
            dir_path = Path(directory)
            if not dir_path.exists():
                return []
            
            if recursive:
                return list(dir_path.rglob(pattern))
            else:
                return list(dir_path.glob(pattern))
        except Exception as e:
            logger.error(f"列出文件失败 {directory}: {e}")
            return []
    
    def copy_file(self, src: Union[str, Path], dst: Union[str, Path]) -> bool:
        """复制文件"""
        try:
            import shutil
            self.ensure_dir(Path(dst).parent)
            shutil.copy2(src, dst)
            return True
        except Exception as e:
            logger.error(f"复制文件失败 {src} -> {dst}: {e}")
            return False
    
    def move_file(self, src: Union[str, Path], dst: Union[str, Path]) -> bool:
        """移动文件"""
        try:
            import shutil
            self.ensure_dir(Path(dst).parent)
            shutil.move(str(src), str(dst))
            return True
        except Exception as e:
            logger.error(f"移动文件失败 {src} -> {dst}: {e}")
            return False
    
    def delete_file(self, file_path: Union[str, Path]) -> bool:
        """删除文件"""
        try:
            Path(file_path).unlink(missing_ok=True)
            return True
        except Exception as e:
            logger.error(f"删除文件失败 {file_path}: {e}")
            return False
    
    def get_file_size(self, file_path: Union[str, Path]) -> int:
        """获取文件大小"""
        try:
            return Path(file_path).stat().st_size
        except Exception as e:
            logger.error(f"获取文件大小失败 {file_path}: {e}")
            return 0
    
    def file_exists(self, file_path: Union[str, Path]) -> bool:
        """检查文件是否存在"""
        return Path(file_path).exists()

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.file_manager = FileManager()
        self.configs = {}
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """加载配置"""
        if config_name in self.configs:
            return self.configs[config_name]
        
        # 尝试加载YAML配置
        yaml_path = self.config_dir / f"{config_name}.yaml"
        if yaml_path.exists():
            config = self.file_manager.read_yaml(yaml_path)
            self.configs[config_name] = config
            return config
        
        # 尝试加载JSON配置
        json_path = self.config_dir / f"{config_name}.json"
        if json_path.exists():
            config = self.file_manager.read_json(json_path)
            self.configs[config_name] = config
            return config
        
        logger.warning(f"配置文件不存在: {config_name}")
        return {}
    
    def save_config(self, config_name: str, config_data: Dict[str, Any]) -> bool:
        """保存配置"""
        self.file_manager.ensure_dir(self.config_dir)
        
        # 默认保存为YAML格式
        yaml_path = self.config_dir / f"{config_name}.yaml"
        success = self.file_manager.write_yaml(config_data, yaml_path)
        
        if success:
            self.configs[config_name] = config_data
        
        return success
    
    def get_config(self, config_name: str, key: str, default: Any = None) -> Any:
        """获取配置项"""
        config = self.load_config(config_name)
        return config.get(key, default)
    
    def set_config(self, config_name: str, key: str, value: Any) -> bool:
        """设置配置项"""
        config = self.load_config(config_name)
        config[key] = value
        return self.save_config(config_name, config)
    
    def list_configs(self) -> List[str]:
        """列出所有配置"""
        configs = []
        
        # 查找YAML配置文件
        yaml_files = self.file_manager.list_files(self.config_dir, "*.yaml")
        configs.extend([f.stem for f in yaml_files])
        
        # 查找JSON配置文件
        json_files = self.file_manager.list_files(self.config_dir, "*.json")
        configs.extend([f.stem for f in json_files])
        
        return list(set(configs))

def ensure_directory(path: Union[str, Path]) -> Path:
    """确保目录存在"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def read_text_file(file_path: Union[str, Path]) -> str:
    """读取文本文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"读取文本文件失败 {file_path}: {e}")
        return ""

def write_text_file(content: str, file_path: Union[str, Path]) -> bool:
    """写入文本文件"""
    try:
        ensure_directory(Path(file_path).parent)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        logger.error(f"写入文本文件失败 {file_path}: {e}")
        return False

def get_project_root() -> Path:
    """获取项目根目录"""
    current = Path(__file__).parent
    while current.parent != current:
        if (current / "setup.py").exists() or (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path.cwd()

# 为了兼容性，创建FileUtils类作为FileManager的别名
class FileUtils(FileManager):
    """文件工具类（FileManager的别名）"""
    pass