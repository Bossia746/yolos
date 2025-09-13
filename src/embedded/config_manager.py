#!/usr/bin/env python3
"""
嵌入式配置管理系统
简化配置复杂度，适配资源受限环境
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

class ConfigFormat(Enum):
    """配置文件格式"""
    JSON = "json"
    YAML = "yaml"
    MINIMAL = "minimal"  # 最小化格式，仅包含关键参数

class ConfigLevel(Enum):
    """配置复杂度等级"""
    MINIMAL = 1      # 最小配置，仅核心参数
    BASIC = 2        # 基础配置，常用参数
    STANDARD = 3     # 标准配置，完整参数
    ADVANCED = 4     # 高级配置，所有参数

@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str = "yolov11n"
    model_path: str = ""
    input_size: int = 320
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    max_detections: int = 100
    
    # 优化参数
    quantization: bool = True
    pruning: bool = False
    distillation: bool = False
    
    # 精度设置
    precision: str = "int8"  # float32, float16, int8
    
@dataclass
class HardwareConfig:
    """硬件配置"""
    platform: str = "auto"
    max_memory_mb: int = 100
    max_storage_mb: int = 500
    cpu_threads: int = 1
    enable_gpu: bool = False
    
    # 性能设置
    performance_mode: str = "balanced"  # power_save, balanced, performance
    thermal_limit: float = 80.0
    
@dataclass
class DeploymentConfig:
    """部署配置"""
    batch_size: int = 1
    max_inference_time_ms: int = 1000
    enable_caching: bool = True
    cache_size_mb: int = 10
    
    # 输出设置
    output_format: str = "json"  # json, binary, minimal
    save_results: bool = False
    
@dataclass
class EmbeddedConfig:
    """嵌入式完整配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    
    # 元数据
    version: str = "1.0"
    created_at: str = ""
    platform_detected: str = ""
    
class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_model_config(config: ModelConfig) -> List[str]:
        """验证模型配置"""
        errors = []
        
        if config.input_size < 32 or config.input_size > 1024:
            errors.append("输入尺寸必须在32-1024之间")
            
        if not 0.0 <= config.confidence_threshold <= 1.0:
            errors.append("置信度阈值必须在0.0-1.0之间")
            
        if not 0.0 <= config.nms_threshold <= 1.0:
            errors.append("NMS阈值必须在0.0-1.0之间")
            
        if config.max_detections < 1 or config.max_detections > 1000:
            errors.append("最大检测数必须在1-1000之间")
            
        if config.precision not in ["float32", "float16", "int8"]:
            errors.append("精度设置必须是float32、float16或int8")
            
        return errors
        
    @staticmethod
    def validate_hardware_config(config: HardwareConfig) -> List[str]:
        """验证硬件配置"""
        errors = []
        
        if config.max_memory_mb < 1:
            errors.append("最大内存必须大于1MB")
            
        if config.max_storage_mb < 1:
            errors.append("最大存储必须大于1MB")
            
        if config.cpu_threads < 1 or config.cpu_threads > 16:
            errors.append("CPU线程数必须在1-16之间")
            
        if config.performance_mode not in ["power_save", "balanced", "performance"]:
            errors.append("性能模式必须是power_save、balanced或performance")
            
        if config.thermal_limit < 40.0 or config.thermal_limit > 100.0:
            errors.append("温度限制必须在40-100°C之间")
            
        return errors
        
    @staticmethod
    def validate_deployment_config(config: DeploymentConfig) -> List[str]:
        """验证部署配置"""
        errors = []
        
        if config.batch_size < 1 or config.batch_size > 32:
            errors.append("批处理大小必须在1-32之间")
            
        if config.max_inference_time_ms < 10 or config.max_inference_time_ms > 10000:
            errors.append("最大推理时间必须在10-10000ms之间")
            
        if config.cache_size_mb < 1 or config.cache_size_mb > 1000:
            errors.append("缓存大小必须在1-1000MB之间")
            
        if config.output_format not in ["json", "binary", "minimal"]:
            errors.append("输出格式必须是json、binary或minimal")
            
        return errors
        
    @classmethod
    def validate_config(cls, config: EmbeddedConfig) -> List[str]:
        """验证完整配置"""
        errors = []
        
        errors.extend(cls.validate_model_config(config.model))
        errors.extend(cls.validate_hardware_config(config.hardware))
        errors.extend(cls.validate_deployment_config(config.deployment))
        
        return errors

class ConfigOptimizer:
    """配置优化器"""
    
    @staticmethod
    def optimize_for_platform(config: EmbeddedConfig, platform: str) -> EmbeddedConfig:
        """根据平台优化配置"""
        optimized = EmbeddedConfig(
            model=ModelConfig(**asdict(config.model)),
            hardware=HardwareConfig(**asdict(config.hardware)),
            deployment=DeploymentConfig(**asdict(config.deployment))
        )
        
        platform_lower = platform.lower()
        
        if "esp32" in platform_lower:
            # ESP32优化
            optimized.model.input_size = min(optimized.model.input_size, 320)
            optimized.model.quantization = True
            optimized.model.precision = "int8"
            optimized.model.max_detections = min(optimized.model.max_detections, 20)
            
            optimized.hardware.max_memory_mb = min(optimized.hardware.max_memory_mb, 3)
            optimized.hardware.cpu_threads = 1
            optimized.hardware.enable_gpu = False
            optimized.hardware.performance_mode = "power_save"
            
            optimized.deployment.batch_size = 1
            optimized.deployment.cache_size_mb = min(optimized.deployment.cache_size_mb, 1)
            optimized.deployment.output_format = "minimal"
            
        elif "raspberry_pi_zero" in platform_lower:
            # 树莓派Zero优化
            optimized.model.input_size = min(optimized.model.input_size, 416)
            optimized.model.quantization = True
            optimized.model.precision = "int8"
            
            optimized.hardware.max_memory_mb = min(optimized.hardware.max_memory_mb, 400)
            optimized.hardware.cpu_threads = 1
            optimized.hardware.performance_mode = "balanced"
            
            optimized.deployment.batch_size = 1
            optimized.deployment.cache_size_mb = min(optimized.deployment.cache_size_mb, 20)
            
        elif "raspberry_pi_4" in platform_lower:
            # 树莓派4优化
            optimized.model.input_size = min(optimized.model.input_size, 640)
            optimized.model.precision = "float16" if optimized.model.precision == "float32" else optimized.model.precision
            
            optimized.hardware.max_memory_mb = min(optimized.hardware.max_memory_mb, 3500)
            optimized.hardware.cpu_threads = min(optimized.hardware.cpu_threads, 4)
            optimized.hardware.performance_mode = "performance"
            
            optimized.deployment.batch_size = min(optimized.deployment.batch_size, 2)
            optimized.deployment.cache_size_mb = min(optimized.deployment.cache_size_mb, 100)
            
        elif "jetson" in platform_lower:
            # Jetson优化
            optimized.hardware.enable_gpu = True
            optimized.hardware.cpu_threads = min(optimized.hardware.cpu_threads, 4)
            optimized.hardware.performance_mode = "performance"
            
            optimized.deployment.batch_size = min(optimized.deployment.batch_size, 4)
            optimized.deployment.cache_size_mb = min(optimized.deployment.cache_size_mb, 200)
            
        return optimized
        
    @staticmethod
    def optimize_for_memory(config: EmbeddedConfig, memory_limit_mb: int) -> EmbeddedConfig:
        """根据内存限制优化配置"""
        optimized = EmbeddedConfig(
            model=ModelConfig(**asdict(config.model)),
            hardware=HardwareConfig(**asdict(config.hardware)),
            deployment=DeploymentConfig(**asdict(config.deployment))
        )
        
        # 调整内存相关参数
        optimized.hardware.max_memory_mb = min(optimized.hardware.max_memory_mb, memory_limit_mb)
        
        if memory_limit_mb < 50:
            # 极低内存
            optimized.model.input_size = min(optimized.model.input_size, 224)
            optimized.model.quantization = True
            optimized.model.precision = "int8"
            optimized.deployment.cache_size_mb = 1
            optimized.deployment.batch_size = 1
            
        elif memory_limit_mb < 200:
            # 低内存
            optimized.model.input_size = min(optimized.model.input_size, 320)
            optimized.model.quantization = True
            optimized.deployment.cache_size_mb = min(optimized.deployment.cache_size_mb, 10)
            
        elif memory_limit_mb < 1000:
            # 中等内存
            optimized.model.input_size = min(optimized.model.input_size, 416)
            optimized.deployment.cache_size_mb = min(optimized.deployment.cache_size_mb, 50)
            
        return optimized

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # 默认配置
        self.default_config = EmbeddedConfig()
        
        # 配置缓存
        self._config_cache: Dict[str, EmbeddedConfig] = {}
        
    def create_minimal_config(self, platform: str = "auto") -> EmbeddedConfig:
        """创建最小配置"""
        config = EmbeddedConfig(
            model=ModelConfig(
                model_name="yolov11n",
                input_size=320,
                confidence_threshold=0.5,
                quantization=True,
                precision="int8"
            ),
            hardware=HardwareConfig(
                platform=platform,
                max_memory_mb=100,
                cpu_threads=1,
                performance_mode="balanced"
            ),
            deployment=DeploymentConfig(
                batch_size=1,
                cache_size_mb=10,
                output_format="json"
            )
        )
        
        return config
        
    def create_platform_config(self, platform: str) -> EmbeddedConfig:
        """创建平台特定配置"""
        base_config = self.create_minimal_config(platform)
        optimized_config = ConfigOptimizer.optimize_for_platform(base_config, platform)
        optimized_config.platform_detected = platform
        
        return optimized_config
        
    def load_config(self, config_path: str, format_type: ConfigFormat = ConfigFormat.JSON) -> Optional[EmbeddedConfig]:
        """加载配置文件"""
        try:
            config_file = Path(config_path)
            
            if not config_file.exists():
                self.logger.error(f"配置文件不存在: {config_path}")
                return None
                
            # 检查缓存
            cache_key = str(config_file.absolute())
            if cache_key in self._config_cache:
                return self._config_cache[cache_key]
                
            with open(config_file, 'r', encoding='utf-8') as f:
                if format_type == ConfigFormat.JSON:
                    data = json.load(f)
                elif format_type == ConfigFormat.YAML:
                    data = yaml.safe_load(f)
                elif format_type == ConfigFormat.MINIMAL:
                    # 最小格式：每行一个key=value
                    data = {}
                    for line in f:
                        line = line.strip()
                        if line and '=' in line and not line.startswith('#'):
                            key, value = line.split('=', 1)
                            # 简单类型转换
                            try:
                                if value.lower() in ['true', 'false']:
                                    value = value.lower() == 'true'
                                elif value.isdigit():
                                    value = int(value)
                                elif '.' in value and value.replace('.', '').isdigit():
                                    value = float(value)
                            except:
                                pass
                            data[key.strip()] = value.strip()
                else:
                    self.logger.error(f"不支持的配置格式: {format_type}")
                    return None
                    
            # 转换为配置对象
            config = self._dict_to_config(data)
            
            # 验证配置
            errors = ConfigValidator.validate_config(config)
            if errors:
                self.logger.warning(f"配置验证警告: {errors}")
                
            # 缓存配置
            self._config_cache[cache_key] = config
            
            return config
            
        except Exception as e:
            self.logger.error(f"加载配置失败: {e}")
            return None
            
    def save_config(self, config: EmbeddedConfig, config_path: str, 
                   format_type: ConfigFormat = ConfigFormat.JSON,
                   level: ConfigLevel = ConfigLevel.STANDARD) -> bool:
        """保存配置文件"""
        try:
            config_file = Path(config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 根据级别过滤配置
            data = self._config_to_dict(config, level)
            
            with open(config_file, 'w', encoding='utf-8') as f:
                if format_type == ConfigFormat.JSON:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                elif format_type == ConfigFormat.YAML:
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
                elif format_type == ConfigFormat.MINIMAL:
                    # 最小格式：扁平化key=value
                    flat_data = self._flatten_dict(data)
                    for key, value in flat_data.items():
                        f.write(f"{key}={value}\n")
                        
            # 更新缓存
            cache_key = str(config_file.absolute())
            self._config_cache[cache_key] = config
            
            return True
            
        except Exception as e:
            self.logger.error(f"保存配置失败: {e}")
            return False
            
    def get_config_templates(self) -> Dict[str, EmbeddedConfig]:
        """获取配置模板"""
        templates = {
            "minimal": self.create_minimal_config(),
            "esp32": self.create_platform_config("esp32"),
            "esp32_s3": self.create_platform_config("esp32_s3"),
            "raspberry_pi_zero": self.create_platform_config("raspberry_pi_zero"),
            "raspberry_pi_4": self.create_platform_config("raspberry_pi_4"),
            "jetson_nano": self.create_platform_config("jetson_nano")
        }
        
        return templates
        
    def optimize_config(self, config: EmbeddedConfig, 
                       platform: Optional[str] = None,
                       memory_limit_mb: Optional[int] = None) -> EmbeddedConfig:
        """优化配置"""
        optimized = config
        
        if platform:
            optimized = ConfigOptimizer.optimize_for_platform(optimized, platform)
            
        if memory_limit_mb:
            optimized = ConfigOptimizer.optimize_for_memory(optimized, memory_limit_mb)
            
        return optimized
        
    def validate_config(self, config: EmbeddedConfig) -> List[str]:
        """验证配置"""
        return ConfigValidator.validate_config(config)
        
    def _dict_to_config(self, data: Dict[str, Any]) -> EmbeddedConfig:
        """字典转配置对象"""
        try:
            # 提取各部分配置
            model_data = data.get('model', {})
            hardware_data = data.get('hardware', {})
            deployment_data = data.get('deployment', {})
            
            # 创建配置对象
            model_config = ModelConfig(**{k: v for k, v in model_data.items() 
                                        if k in ModelConfig.__dataclass_fields__})
            hardware_config = HardwareConfig(**{k: v for k, v in hardware_data.items() 
                                              if k in HardwareConfig.__dataclass_fields__})
            deployment_config = DeploymentConfig(**{k: v for k, v in deployment_data.items() 
                                                  if k in DeploymentConfig.__dataclass_fields__})
            
            config = EmbeddedConfig(
                model=model_config,
                hardware=hardware_config,
                deployment=deployment_config
            )
            
            # 设置元数据
            if 'version' in data:
                config.version = data['version']
            if 'created_at' in data:
                config.created_at = data['created_at']
            if 'platform_detected' in data:
                config.platform_detected = data['platform_detected']
                
            return config
            
        except Exception as e:
            self.logger.error(f"配置转换失败: {e}")
            return self.default_config
            
    def _config_to_dict(self, config: EmbeddedConfig, level: ConfigLevel) -> Dict[str, Any]:
        """配置对象转字典"""
        data = asdict(config)
        
        # 根据级别过滤
        if level == ConfigLevel.MINIMAL:
            # 仅保留核心参数
            minimal_data = {
                'model': {
                    'model_name': data['model']['model_name'],
                    'input_size': data['model']['input_size'],
                    'confidence_threshold': data['model']['confidence_threshold'],
                    'quantization': data['model']['quantization'],
                    'precision': data['model']['precision']
                },
                'hardware': {
                    'platform': data['hardware']['platform'],
                    'max_memory_mb': data['hardware']['max_memory_mb'],
                    'cpu_threads': data['hardware']['cpu_threads']
                },
                'deployment': {
                    'batch_size': data['deployment']['batch_size'],
                    'output_format': data['deployment']['output_format']
                }
            }
            return minimal_data
            
        elif level == ConfigLevel.BASIC:
            # 移除高级参数
            for section in ['model', 'hardware', 'deployment']:
                if section in data:
                    # 保留常用参数，移除高级参数
                    if section == 'model':
                        data[section].pop('distillation', None)
                    elif section == 'hardware':
                        data[section].pop('thermal_limit', None)
                        
        return data
        
    def _flatten_dict(self, data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """扁平化字典"""
        flat = {}
        
        for key, value in data.items():
            new_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                flat.update(self._flatten_dict(value, new_key))
            else:
                flat[new_key] = value
                
        return flat
        
    def clear_cache(self):
        """清空配置缓存"""
        self._config_cache.clear()
        
    def get_config_size(self, config: EmbeddedConfig, format_type: ConfigFormat) -> int:
        """获取配置文件大小（字节）"""
        try:
            data = self._config_to_dict(config, ConfigLevel.STANDARD)
            
            if format_type == ConfigFormat.JSON:
                content = json.dumps(data, ensure_ascii=False)
            elif format_type == ConfigFormat.YAML:
                content = yaml.dump(data, default_flow_style=False, allow_unicode=True)
            elif format_type == ConfigFormat.MINIMAL:
                flat_data = self._flatten_dict(data)
                content = "\n".join([f"{k}={v}" for k, v in flat_data.items()])
            else:
                return 0
                
            return len(content.encode('utf-8'))
            
        except Exception:
            return 0

# 全局配置管理器实例
_global_config_manager: Optional[ConfigManager] = None

def get_config_manager() -> ConfigManager:
    """获取配置管理器"""
    global _global_config_manager
    
    if _global_config_manager is None:
        _global_config_manager = ConfigManager()
        
    return _global_config_manager

def create_default_config(platform: str = "auto") -> EmbeddedConfig:
    """创建默认配置"""
    manager = get_config_manager()
    return manager.create_platform_config(platform)

def load_config_file(config_path: str) -> Optional[EmbeddedConfig]:
    """加载配置文件"""
    manager = get_config_manager()
    
    # 自动检测格式
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        format_type = ConfigFormat.YAML
    elif config_path.endswith('.json'):
        format_type = ConfigFormat.JSON
    else:
        format_type = ConfigFormat.MINIMAL
        
    return manager.load_config(config_path, format_type)

if __name__ == "__main__":
    # 测试代码
    print("配置管理系统测试")
    print("=" * 50)
    
    # 创建配置管理器
    manager = ConfigManager()
    
    # 测试配置模板
    templates = manager.get_config_templates()
    print(f"\n可用配置模板: {list(templates.keys())}")
    
    # 测试ESP32配置
    esp32_config = templates['esp32']
    print(f"\nESP32配置:")
    print(f"  模型: {esp32_config.model.model_name}")
    print(f"  输入尺寸: {esp32_config.model.input_size}")
    print(f"  精度: {esp32_config.model.precision}")
    print(f"  最大内存: {esp32_config.hardware.max_memory_mb}MB")
    print(f"  CPU线程: {esp32_config.hardware.cpu_threads}")
    
    # 验证配置
    errors = manager.validate_config(esp32_config)
    print(f"\n配置验证: {'通过' if not errors else f'错误: {errors}'}")
    
    # 测试配置保存和加载
    test_config_path = "test_config.json"
    
    # 保存配置
    if manager.save_config(esp32_config, test_config_path, ConfigFormat.JSON, ConfigLevel.MINIMAL):
        print(f"\n配置已保存到: {test_config_path}")
        
        # 获取配置文件大小
        size = manager.get_config_size(esp32_config, ConfigFormat.JSON)
        print(f"配置文件大小: {size} 字节")
        
        # 加载配置
        loaded_config = manager.load_config(test_config_path, ConfigFormat.JSON)
        if loaded_config:
            print(f"配置加载成功")
            print(f"  模型名称: {loaded_config.model.model_name}")
            print(f"  平台: {loaded_config.hardware.platform}")
        else:
            print("配置加载失败")
            
    # 测试内存优化
    print(f"\n内存优化测试:")
    original_memory = esp32_config.hardware.max_memory_mb
    optimized_config = manager.optimize_config(esp32_config, memory_limit_mb=50)
    print(f"  原始内存限制: {original_memory}MB")
    print(f"  优化后内存限制: {optimized_config.hardware.max_memory_mb}MB")
    print(f"  输入尺寸调整: {esp32_config.model.input_size} -> {optimized_config.model.input_size}")
    
    # 清理测试文件
    try:
        os.remove(test_config_path)
        print(f"\n测试文件已清理")
    except:
        pass
        
    print("\n测试完成")