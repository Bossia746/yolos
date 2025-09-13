#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOS多平台兼容性管理器

提供跨平台兼容性检测和适配功能，支持：
- Windows、Linux、macOS平台检测
- AIoT设备兼容性验证
- 硬件资源检测和优化
- 平台特定配置管理

Author: YOLOS Team
Version: 1.0.0
"""

import os
import sys
import platform
import subprocess
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from .exceptions import (
    ErrorCode, SystemException, create_exception,
    exception_handler
)


class PlatformType(Enum):
    """平台类型枚举"""
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    ANDROID = "android"
    IOS = "ios"
    RASPBERRY_PI = "raspberry_pi"
    JETSON = "jetson"
    UNKNOWN = "unknown"


class DeviceType(Enum):
    """设备类型枚举"""
    DESKTOP = "desktop"
    LAPTOP = "laptop"
    SERVER = "server"
    EMBEDDED = "embedded"
    MOBILE = "mobile"
    IOT = "iot"
    EDGE = "edge"
    UNKNOWN = "unknown"


@dataclass
class HardwareInfo:
    """硬件信息"""
    cpu_count: int
    cpu_architecture: str
    memory_total_gb: float
    gpu_count: int
    gpu_info: List[Dict[str, Any]] = field(default_factory=list)
    storage_info: Dict[str, Any] = field(default_factory=dict)
    network_interfaces: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'cpu_count': self.cpu_count,
            'cpu_architecture': self.cpu_architecture,
            'memory_total_gb': self.memory_total_gb,
            'gpu_count': self.gpu_count,
            'gpu_info': self.gpu_info,
            'storage_info': self.storage_info,
            'network_interfaces': self.network_interfaces
        }


@dataclass
class PlatformInfo:
    """平台信息"""
    platform_type: PlatformType
    device_type: DeviceType
    os_name: str
    os_version: str
    python_version: str
    architecture: str
    hardware: HardwareInfo
    capabilities: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'platform_type': self.platform_type.value,
            'device_type': self.device_type.value,
            'os_name': self.os_name,
            'os_version': self.os_version,
            'python_version': self.python_version,
            'architecture': self.architecture,
            'hardware': self.hardware.to_dict(),
            'capabilities': self.capabilities,
            'limitations': self.limitations
        }


@dataclass
class CompatibilityResult:
    """兼容性检测结果"""
    is_compatible: bool
    compatibility_score: float  # 0.0 - 1.0
    supported_features: List[str]
    unsupported_features: List[str]
    warnings: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_compatible': self.is_compatible,
            'compatibility_score': self.compatibility_score,
            'supported_features': self.supported_features,
            'unsupported_features': self.unsupported_features,
            'warnings': self.warnings,
            'recommendations': self.recommendations
        }


class PlatformCompatibilityManager:
    """平台兼容性管理器"""
    
    def __init__(self, logger_name: str = "yolos.platform"):
        self.logger = logging.getLogger(logger_name)
        self.platform_info: Optional[PlatformInfo] = None
        
        # 支持的应用类型和对应的硬件要求
        self.application_requirements = {
            'human_detection': {
                'min_memory_gb': 2.0,
                'min_cpu_cores': 2,
                'gpu_required': False,
                'supported_platforms': [PlatformType.WINDOWS, PlatformType.LINUX, PlatformType.MACOS]
            },
            'face_recognition': {
                'min_memory_gb': 4.0,
                'min_cpu_cores': 4,
                'gpu_required': False,
                'gpu_recommended': True,
                'supported_platforms': [PlatformType.WINDOWS, PlatformType.LINUX, PlatformType.MACOS]
            },
            'pose_estimation': {
                'min_memory_gb': 6.0,
                'min_cpu_cores': 4,
                'gpu_required': True,
                'supported_platforms': [PlatformType.WINDOWS, PlatformType.LINUX, PlatformType.MACOS]
            },
            'gesture_recognition': {
                'min_memory_gb': 4.0,
                'min_cpu_cores': 4,
                'gpu_required': False,
                'gpu_recommended': True,
                'supported_platforms': [PlatformType.WINDOWS, PlatformType.LINUX, PlatformType.MACOS, PlatformType.RASPBERRY_PI]
            },
            'fall_detection': {
                'min_memory_gb': 3.0,
                'min_cpu_cores': 2,
                'gpu_required': False,
                'real_time_required': True,
                'supported_platforms': [PlatformType.WINDOWS, PlatformType.LINUX, PlatformType.MACOS, PlatformType.RASPBERRY_PI, PlatformType.JETSON]
            },
            'object_detection': {
                'min_memory_gb': 4.0,
                'min_cpu_cores': 4,
                'gpu_required': False,
                'gpu_recommended': True,
                'supported_platforms': [PlatformType.WINDOWS, PlatformType.LINUX, PlatformType.MACOS]
            },
            'pet_detection': {
                'min_memory_gb': 3.0,
                'min_cpu_cores': 2,
                'gpu_required': False,
                'supported_platforms': [PlatformType.WINDOWS, PlatformType.LINUX, PlatformType.MACOS, PlatformType.RASPBERRY_PI]
            },
            'plant_recognition': {
                'min_memory_gb': 2.0,
                'min_cpu_cores': 2,
                'gpu_required': False,
                'supported_platforms': [PlatformType.WINDOWS, PlatformType.LINUX, PlatformType.MACOS, PlatformType.RASPBERRY_PI]
            }
        }
        
        # 初始化平台信息
        self._detect_platform()
    
    @exception_handler(ErrorCode.SYSTEM_ERROR)
    def _detect_platform(self):
        """检测当前平台信息"""
        try:
            # 检测平台类型
            platform_type = self._detect_platform_type()
            device_type = self._detect_device_type()
            
            # 获取系统信息
            os_name = platform.system()
            os_version = platform.release()
            python_version = platform.python_version()
            architecture = platform.machine()
            
            # 获取硬件信息
            hardware = self._detect_hardware()
            
            # 检测平台能力和限制
            capabilities, limitations = self._detect_capabilities_and_limitations(platform_type, hardware)
            
            self.platform_info = PlatformInfo(
                platform_type=platform_type,
                device_type=device_type,
                os_name=os_name,
                os_version=os_version,
                python_version=python_version,
                architecture=architecture,
                hardware=hardware,
                capabilities=capabilities,
                limitations=limitations
            )
            
            self.logger.info(f"平台检测完成: {platform_type.value} ({device_type.value})")
            
        except Exception as e:
            raise create_exception(
                ErrorCode.SYSTEM_ERROR,
                f"平台检测失败: {e}",
                {'component': 'PlatformCompatibilityManager'}
            )
    
    def _detect_platform_type(self) -> PlatformType:
        """检测平台类型"""
        system = platform.system().lower()
        
        if system == "windows":
            return PlatformType.WINDOWS
        elif system == "linux":
            # 检测是否为特殊Linux设备
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read().lower()
                    if 'raspberry pi' in cpuinfo or 'bcm' in cpuinfo:
                        return PlatformType.RASPBERRY_PI
                    elif 'tegra' in cpuinfo or 'jetson' in cpuinfo:
                        return PlatformType.JETSON
            except:
                pass
            return PlatformType.LINUX
        elif system == "darwin":
            return PlatformType.MACOS
        else:
            return PlatformType.UNKNOWN
    
    def _detect_device_type(self) -> DeviceType:
        """检测设备类型"""
        if not PSUTIL_AVAILABLE:
            return DeviceType.UNKNOWN
        
        try:
            # 基于硬件特征判断设备类型
            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            # 检测是否有电池（笔记本电脑）
            has_battery = False
            try:
                battery = psutil.sensors_battery()
                has_battery = battery is not None
            except:
                pass
            
            # 判断逻辑
            if cpu_count >= 16 and memory_gb >= 32:
                return DeviceType.SERVER
            elif has_battery:
                return DeviceType.LAPTOP
            elif cpu_count <= 4 and memory_gb <= 8:
                return DeviceType.EMBEDDED
            else:
                return DeviceType.DESKTOP
                
        except Exception as e:
            self.logger.warning(f"设备类型检测失败: {e}")
            return DeviceType.UNKNOWN
    
    def _detect_hardware(self) -> HardwareInfo:
        """检测硬件信息"""
        cpu_count = 1
        cpu_architecture = platform.machine()
        memory_total_gb = 1.0
        gpu_count = 0
        gpu_info = []
        storage_info = {}
        network_interfaces = []
        
        if PSUTIL_AVAILABLE:
            try:
                cpu_count = psutil.cpu_count()
                memory_total_gb = psutil.virtual_memory().total / (1024**3)
                
                # 存储信息
                disk_usage = psutil.disk_usage('/')
                storage_info = {
                    'total_gb': disk_usage.total / (1024**3),
                    'free_gb': disk_usage.free / (1024**3),
                    'used_gb': disk_usage.used / (1024**3)
                }
                
                # 网络接口
                network_interfaces = list(psutil.net_if_addrs().keys())
                
            except Exception as e:
                self.logger.warning(f"系统信息检测失败: {e}")
        
        # GPU信息
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                gpu_count = len(gpus)
                gpu_info = [
                    {
                        'name': gpu.name,
                        'memory_total': gpu.memoryTotal,
                        'driver_version': gpu.driver
                    }
                    for gpu in gpus
                ]
            except Exception as e:
                self.logger.warning(f"GPU信息检测失败: {e}")
        
        return HardwareInfo(
            cpu_count=cpu_count,
            cpu_architecture=cpu_architecture,
            memory_total_gb=memory_total_gb,
            gpu_count=gpu_count,
            gpu_info=gpu_info,
            storage_info=storage_info,
            network_interfaces=network_interfaces
        )
    
    def _detect_capabilities_and_limitations(
        self,
        platform_type: PlatformType,
        hardware: HardwareInfo
    ) -> Tuple[List[str], List[str]]:
        """检测平台能力和限制"""
        capabilities = []
        limitations = []
        
        # 基础能力
        capabilities.append("basic_cv_operations")
        
        # GPU能力
        if hardware.gpu_count > 0:
            capabilities.extend(["gpu_acceleration", "deep_learning", "real_time_processing"])
        else:
            limitations.append("no_gpu_acceleration")
        
        # 内存能力
        if hardware.memory_total_gb >= 8:
            capabilities.append("large_model_support")
        elif hardware.memory_total_gb < 4:
            limitations.append("limited_memory")
        
        # CPU能力
        if hardware.cpu_count >= 4:
            capabilities.append("multi_threading")
        else:
            limitations.append("limited_cpu_cores")
        
        # 平台特定能力和限制
        if platform_type == PlatformType.WINDOWS:
            capabilities.extend(["directx_support", "cuda_support"])
        elif platform_type == PlatformType.LINUX:
            capabilities.extend(["docker_support", "cuda_support", "opencl_support"])
        elif platform_type == PlatformType.MACOS:
            capabilities.append("metal_support")
            limitations.append("no_cuda_support")
        elif platform_type in [PlatformType.RASPBERRY_PI, PlatformType.JETSON]:
            capabilities.extend(["edge_computing", "low_power"])
            limitations.extend(["limited_performance", "arm_architecture"])
        
        return capabilities, limitations
    
    def check_application_compatibility(
        self,
        application_type: str
    ) -> CompatibilityResult:
        """检查应用兼容性"""
        if not self.platform_info:
            raise create_exception(
                ErrorCode.SYSTEM_ERROR,
                "平台信息未初始化",
                {'component': 'PlatformCompatibilityManager'}
            )
        
        if application_type not in self.application_requirements:
            raise create_exception(
                ErrorCode.INVALID_PARAMETER,
                f"不支持的应用类型: {application_type}",
                {'application_type': application_type}
            )
        
        requirements = self.application_requirements[application_type]
        supported_features = []
        unsupported_features = []
        warnings = []
        recommendations = []
        
        # 检查平台支持
        platform_supported = self.platform_info.platform_type in requirements['supported_platforms']
        if not platform_supported:
            unsupported_features.append("platform_not_supported")
        else:
            supported_features.append("platform_supported")
        
        # 检查内存要求
        min_memory = requirements.get('min_memory_gb', 0)
        if self.platform_info.hardware.memory_total_gb >= min_memory:
            supported_features.append("sufficient_memory")
        else:
            unsupported_features.append("insufficient_memory")
            recommendations.append(f"建议增加内存至{min_memory}GB以上")
        
        # 检查CPU要求
        min_cpu_cores = requirements.get('min_cpu_cores', 1)
        if self.platform_info.hardware.cpu_count >= min_cpu_cores:
            supported_features.append("sufficient_cpu")
        else:
            unsupported_features.append("insufficient_cpu")
            recommendations.append(f"建议使用{min_cpu_cores}核以上CPU")
        
        # 检查GPU要求
        gpu_required = requirements.get('gpu_required', False)
        gpu_recommended = requirements.get('gpu_recommended', False)
        
        if gpu_required:
            if self.platform_info.hardware.gpu_count > 0:
                supported_features.append("gpu_available")
            else:
                unsupported_features.append("gpu_required_but_not_available")
                recommendations.append("此应用需要GPU支持，请安装兼容的显卡")
        elif gpu_recommended:
            if self.platform_info.hardware.gpu_count > 0:
                supported_features.append("gpu_available")
            else:
                warnings.append("建议使用GPU以获得更好的性能")
                recommendations.append("考虑添加GPU以提升处理速度")
        
        # 检查实时处理要求
        real_time_required = requirements.get('real_time_required', False)
        if real_time_required:
            if "real_time_processing" in self.platform_info.capabilities:
                supported_features.append("real_time_capable")
            else:
                warnings.append("实时处理性能可能不足")
                recommendations.append("考虑优化算法或升级硬件以满足实时处理需求")
        
        # 计算兼容性分数
        total_checks = len(supported_features) + len(unsupported_features)
        compatibility_score = len(supported_features) / total_checks if total_checks > 0 else 0.0
        
        # 判断是否兼容
        is_compatible = (
            platform_supported and
            len(unsupported_features) == 0 and
            compatibility_score >= 0.7
        )
        
        return CompatibilityResult(
            is_compatible=is_compatible,
            compatibility_score=compatibility_score,
            supported_features=supported_features,
            unsupported_features=unsupported_features,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def get_platform_info(self) -> Optional[PlatformInfo]:
        """获取平台信息"""
        return self.platform_info
    
    def get_supported_applications(self) -> List[str]:
        """获取支持的应用类型列表"""
        if not self.platform_info:
            return []
        
        supported_apps = []
        for app_type in self.application_requirements:
            result = self.check_application_compatibility(app_type)
            if result.is_compatible:
                supported_apps.append(app_type)
        
        return supported_apps
    
    def generate_compatibility_report(self) -> Dict[str, Any]:
        """生成兼容性报告"""
        if not self.platform_info:
            return {'error': '平台信息未初始化'}
        
        report = {
            'platform_info': self.platform_info.to_dict(),
            'application_compatibility': {},
            'overall_score': 0.0,
            'recommendations': []
        }
        
        total_score = 0.0
        app_count = 0
        all_recommendations = set()
        
        # 检查所有应用类型的兼容性
        for app_type in self.application_requirements:
            try:
                result = self.check_application_compatibility(app_type)
                report['application_compatibility'][app_type] = result.to_dict()
                total_score += result.compatibility_score
                app_count += 1
                all_recommendations.update(result.recommendations)
            except Exception as e:
                self.logger.error(f"应用{app_type}兼容性检查失败: {e}")
        
        # 计算总体分数
        report['overall_score'] = total_score / app_count if app_count > 0 else 0.0
        report['recommendations'] = list(all_recommendations)
        
        return report
    
    def get_optimization_suggestions(self) -> List[str]:
        """获取平台优化建议"""
        if not self.platform_info:
            return ["平台信息未初始化"]
        
        suggestions = []
        
        # 内存优化建议
        if self.platform_info.hardware.memory_total_gb < 8:
            suggestions.append("考虑增加系统内存以支持更多应用类型")
        
        # GPU优化建议
        if self.platform_info.hardware.gpu_count == 0:
            suggestions.append("添加GPU可显著提升深度学习模型的处理速度")
        
        # CPU优化建议
        if self.platform_info.hardware.cpu_count < 4:
            suggestions.append("升级到多核CPU可提升并行处理能力")
        
        # 平台特定建议
        if self.platform_info.platform_type == PlatformType.RASPBERRY_PI:
            suggestions.extend([
                "使用模型量化技术减少内存占用",
                "启用硬件加速（如果可用）",
                "考虑使用轻量级模型版本"
            ])
        elif self.platform_info.platform_type == PlatformType.MACOS:
            suggestions.append("考虑使用Metal Performance Shaders进行GPU加速")
        
        return suggestions


# 全局平台兼容性管理器
global_platform_manager = PlatformCompatibilityManager()


def get_platform_manager() -> PlatformCompatibilityManager:
    """获取全局平台兼容性管理器"""
    return global_platform_manager


def check_platform_compatibility(application_type: str) -> CompatibilityResult:
    """检查平台兼容性（便捷函数）"""
    return global_platform_manager.check_application_compatibility(application_type)


if __name__ == "__main__":
    # 测试平台兼容性检测
    manager = PlatformCompatibilityManager()
    
    print("平台信息:")
    platform_info = manager.get_platform_info()
    if platform_info:
        print(json.dumps(platform_info.to_dict(), ensure_ascii=False, indent=2))
    
    print("\n兼容性报告:")
    report = manager.generate_compatibility_report()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    
    print("\n优化建议:")
    suggestions = manager.get_optimization_suggestions()
    for suggestion in suggestions:
        print(f"- {suggestion}")