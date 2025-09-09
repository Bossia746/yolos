#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一配置管理器
管理所有识别场景的配置，确保离线和在线学习的协调
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class SceneConfig:
    """场景配置"""
    name: str
    classes: List[str]
    input_size: tuple
    model_type: str
    offline_ready: bool = False
    online_enabled: bool = True
    priority: int = 1
    additional_attributes: Dict[str, Any] = None

@dataclass
class ModelConfig:
    """模型配置"""
    scene: str
    model_path: str
    weights_path: str
    config_path: str
    accuracy: float
    created_time: str
    version: str
    size_mb: float

class UnifiedConfigManager:
    """统一配置管理器"""
    
    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置文件路径
        self.main_config_file = self.config_dir / "main_config.yaml"
        self.scenes_config_file = self.config_dir / "scenes_config.yaml"
        self.models_config_file = self.config_dir / "models_config.yaml"
        
        # 加载配置
        self.main_config = self._load_main_config()
        self.scenes_config = self._load_scenes_config()
        self.models_config = self._load_models_config()
        
        logger.info("统一配置管理器初始化完成")
    
    def _load_main_config(self) -> Dict[str, Any]:
        """加载主配置"""
        default_config = {
            'system': {
                'name': 'YOLOS Hybrid Recognition System',
                'version': '2.0.0',
                'offline_first': True,
                'online_fallback': True,
                'network_check_interval': 30,
                'cache_enabled': True,
                'cache_max_size': 1000
            },
            'paths': {
                'models_dir': './models',
                'offline_models_dir': './models/offline_models',
                'datasets_dir': './datasets',
                'logs_dir': './log',
                'temp_dir': './temp'
            },
            'training': {
                'default_epochs': 50,
                'batch_size': 16,
                'learning_rate': 0.001,
                'validation_split': 0.2,
                'early_stopping_patience': 10
            },
            'performance': {
                'max_processing_time': 5.0,
                'target_fps': 30,
                'memory_limit_mb': 2048
            }
        }
        
        if self.main_config_file.exists():
            try:
                with open(self.main_config_file, 'r', encoding='utf-8') as f:
                    loaded_config = yaml.safe_load(f)
                    # 合并配置
                    self._deep_update(default_config, loaded_config)
            except Exception as e:
                logger.error(f"加载主配置失败: {e}")
        
        # 保存配置
        self._save_yaml_config(self.main_config_file, default_config)
        return default_config
    
    def _load_scenes_config(self) -> Dict[str, SceneConfig]:
        """加载场景配置"""
        default_scenes = {
            'pets': SceneConfig(
                name='pets',
                classes=['dog', 'cat', 'bird', 'rabbit', 'hamster', 'fish', 'parrot', 'canary', 'goldfish', 'turtle'],
                input_size=(224, 224),
                model_type='classification',
                additional_attributes={'colors': ['brown', 'black', 'white', 'gray', 'orange', 'yellow']}
            ),
            'plants': SceneConfig(
                name='plants',
                classes=['rose', 'sunflower', 'tulip', 'daisy', 'lily', 'orchid', 'cactus', 'fern', 'bamboo', 'tree'],
                input_size=(224, 224),
                model_type='classification',
                additional_attributes={'health_states': ['healthy', 'diseased', 'wilted', 'flowering']}
            ),
            'traffic': SceneConfig(
                name='traffic',
                classes=['stop_sign', 'yield_sign', 'speed_limit', 'traffic_light_red', 'traffic_light_green', 'pedestrian_crossing'],
                input_size=(224, 224),
                model_type='detection'
            ),
            'public_signs': SceneConfig(
                name='public_signs',
                classes=['restroom', 'exit', 'elevator', 'stairs', 'parking', 'hospital', 'pharmacy', 'restaurant'],
                input_size=(224, 224),
                model_type='detection'
            ),
            'medicines': SceneConfig(
                name='medicines',
                classes=['pill_round', 'pill_oval', 'capsule', 'tablet', 'liquid_bottle', 'injection'],
                input_size=(224, 224),
                model_type='classification',
                additional_attributes={'colors': ['white', 'red', 'blue', 'yellow', 'green', 'pink']}
            ),
            'qr_codes': SceneConfig(
                name='qr_codes',
                classes=['qr_code', 'data_matrix', 'aztec_code'],
                input_size=(224, 224),
                model_type='detection'
            ),
            'barcodes': SceneConfig(
                name='barcodes',
                classes=['ean13', 'ean8', 'upc_a', 'code128', 'code39'],
                input_size=(224, 224),
                model_type='detection'
            ),
            'dynamic_objects': SceneConfig(
                name='dynamic_objects',
                classes=['person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck', 'airplane', 'boat'],
                input_size=(224, 224),
                model_type='detection',
                additional_attributes={'motion_types': ['static', 'slow', 'medium', 'fast']}
            ),
            'human_actions': SceneConfig(
                name='human_actions',
                classes=['standing', 'walking', 'running', 'sitting', 'jumping', 'waving', 'pointing', 'clapping'],
                input_size=(224, 224),
                model_type='classification'
            )
        }
        
        if self.scenes_config_file.exists():
            try:
                with open(self.scenes_config_file, 'r', encoding='utf-8') as f:
                    loaded_scenes = yaml.safe_load(f)
                    # 转换为SceneConfig对象
                    for scene_name, scene_data in loaded_scenes.items():
                        if scene_name in default_scenes:
                            # 更新现有场景
                            for key, value in scene_data.items():
                                if hasattr(default_scenes[scene_name], key):
                                    setattr(default_scenes[scene_name], key, value)
            except Exception as e:
                logger.error(f"加载场景配置失败: {e}")
        
        # 保存配置
        scenes_dict = {name: asdict(config) for name, config in default_scenes.items()}
        self._save_yaml_config(self.scenes_config_file, scenes_dict)
        
        return default_scenes
    
    def _load_models_config(self) -> Dict[str, ModelConfig]:
        """加载模型配置"""
        models_config = {}
        
        if self.models_config_file.exists():
            try:
                with open(self.models_config_file, 'r', encoding='utf-8') as f:
                    loaded_models = yaml.safe_load(f)
                    for model_name, model_data in loaded_models.items():
                        models_config[model_name] = ModelConfig(**model_data)
            except Exception as e:
                logger.error(f"加载模型配置失败: {e}")
        
        return models_config
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """深度更新字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _save_yaml_config(self, file_path: Path, config: Dict):
        """保存YAML配置"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
        except Exception as e:
            logger.error(f"保存配置失败 {file_path}: {e}")
    
    def get_scene_config(self, scene: str) -> Optional[SceneConfig]:
        """获取场景配置"""
        return self.scenes_config.get(scene)
    
    def get_all_scenes(self) -> List[str]:
        """获取所有场景名称"""
        return list(self.scenes_config.keys())
    
    def update_scene_config(self, scene: str, config: SceneConfig):
        """更新场景配置"""
        self.scenes_config[scene] = config
        scenes_dict = {name: asdict(config) for name, config in self.scenes_config.items()}
        self._save_yaml_config(self.scenes_config_file, scenes_dict)
    
    def register_model(self, model_config: ModelConfig):
        """注册模型"""
        self.models_config[model_config.scene] = model_config
        models_dict = {name: asdict(config) for name, config in self.models_config.items()}
        self._save_yaml_config(self.models_config_file, models_dict)
    
    def get_model_config(self, scene: str) -> Optional[ModelConfig]:
        """获取模型配置"""
        return self.models_config.get(scene)
    
    def get_system_config(self) -> Dict[str, Any]:
        """获取系统配置"""
        return self.main_config
    
    def update_system_config(self, config_path: str, value: Any):
        """更新系统配置"""
        keys = config_path.split('.')
        current = self.main_config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
        self._save_yaml_config(self.main_config_file, self.main_config)
    
    def get_offline_readiness_report(self) -> Dict[str, Any]:
        """获取离线就绪报告"""
        report = {
            'total_scenes': len(self.scenes_config),
            'offline_ready_scenes': 0,
            'missing_models': [],
            'scene_details': {}
        }
        
        for scene_name, scene_config in self.scenes_config.items():
            model_config = self.models_config.get(scene_name)
            
            if model_config and scene_config.offline_ready:
                report['offline_ready_scenes'] += 1
                report['scene_details'][scene_name] = {
                    'status': 'ready',
                    'accuracy': model_config.accuracy,
                    'model_size_mb': model_config.size_mb
                }
            else:
                report['missing_models'].append(scene_name)
                report['scene_details'][scene_name] = {
                    'status': 'missing',
                    'reason': 'No trained model' if not model_config else 'Not marked as offline ready'
                }
        
        report['offline_readiness_percentage'] = (report['offline_ready_scenes'] / report['total_scenes']) * 100
        
        return report
    
    def export_config_package(self, output_path: str):
        """导出配置包"""
        import zipfile
        
        output_file = Path(output_path)
        
        with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 添加配置文件
            if self.main_config_file.exists():
                zipf.write(self.main_config_file, "config/main_config.yaml")
            
            if self.scenes_config_file.exists():
                zipf.write(self.scenes_config_file, "config/scenes_config.yaml")
            
            if self.models_config_file.exists():
                zipf.write(self.models_config_file, "config/models_config.yaml")
            
            # 添加README
            readme_content = self._generate_config_readme()
            zipf.writestr("README.md", readme_content)
        
        logger.info(f"配置包已导出: {output_file}")
    
    def _generate_config_readme(self) -> str:
        """生成配置说明文档"""
        report = self.get_offline_readiness_report()
        
        readme = f"""# YOLOS 混合识别系统配置包

## 系统信息
- 系统名称: {self.main_config['system']['name']}
- 版本: {self.main_config['system']['version']}
- 离线优先: {self.main_config['system']['offline_first']}
- 在线回退: {self.main_config['system']['online_fallback']}

## 离线就绪状态
- 总场景数: {report['total_scenes']}
- 离线就绪场景: {report['offline_ready_scenes']}
- 就绪率: {report['offline_readiness_percentage']:.1f}%

## 支持的识别场景

"""
        
        for scene_name, scene_config in self.scenes_config.items():
            readme += f"### {scene_name.upper()}\n"
            readme += f"- 类别数: {len(scene_config.classes)}\n"
            readme += f"- 模型类型: {scene_config.model_type}\n"
            readme += f"- 输入尺寸: {scene_config.input_size}\n"
            readme += f"- 离线就绪: {'✓' if scene_config.offline_ready else '✗'}\n"
            readme += f"- 支持类别: {', '.join(scene_config.classes[:5])}{'...' if len(scene_config.classes) > 5 else ''}\n\n"
        
        readme += """## 使用说明

1. 解压配置包到项目根目录
2. 运行 `python scripts/setup_hybrid_system.py` 初始化系统
3. 使用 `python scripts/train_offline_models.py` 训练离线模型
4. 启动混合识别系统

## 配置文件说明

- `main_config.yaml`: 主系统配置
- `scenes_config.yaml`: 场景识别配置
- `models_config.yaml`: 模型注册信息
"""
        
        return readme

# 全局配置管理器实例
_config_manager = None

def get_config_manager() -> UnifiedConfigManager:
    """获取全局配置管理器"""
    global _config_manager
    if _config_manager is None:
        _config_manager = UnifiedConfigManager()
    return _config_manager

if __name__ == "__main__":
    # 示例使用
    manager = UnifiedConfigManager()
    
    # 获取离线就绪报告
    report = manager.get_offline_readiness_report()
    print(f"离线就绪报告: {json.dumps(report, indent=2, ensure_ascii=False)}")
    
    # 导出配置包
    manager.export_config_package("config_package.zip")