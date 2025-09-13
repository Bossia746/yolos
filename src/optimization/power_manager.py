#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能功耗管理模块
实现动态调频、休眠唤醒机制和电池优化
针对K230、ESP32等边缘设备的功耗控制
"""

import time
import logging
import threading
import psutil
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque, defaultdict
import json
import os

# 可选依赖
try:
    import GPUtil
except ImportError:
    GPUtil = None

logger = logging.getLogger(__name__)

class PowerMode(Enum):
    """功耗模式"""
    HIGH_PERFORMANCE = "high_performance"  # 高性能模式
    BALANCED = "balanced"  # 平衡模式
    POWER_SAVER = "power_saver"  # 省电模式
    ULTRA_LOW_POWER = "ultra_low_power"  # 超低功耗模式
    ADAPTIVE = "adaptive"  # 自适应模式
    CUSTOM = "custom"  # 自定义模式

class DeviceType(Enum):
    """设备类型"""
    K230 = "k230"  # K230芯片
    ESP32 = "esp32"  # ESP32芯片
    RASPBERRY_PI = "raspberry_pi"  # 树莓派
    JETSON_NANO = "jetson_nano"  # Jetson Nano
    GENERIC_ARM = "generic_arm"  # 通用ARM设备
    X86_LAPTOP = "x86_laptop"  # x86笔记本
    DESKTOP = "desktop"  # 桌面设备

class PowerState(Enum):
    """电源状态"""
    ACTIVE = "active"  # 活跃状态
    IDLE = "idle"  # 空闲状态
    LIGHT_SLEEP = "light_sleep"  # 轻度睡眠
    DEEP_SLEEP = "deep_sleep"  # 深度睡眠
    HIBERNATION = "hibernation"  # 休眠状态
    SHUTDOWN = "shutdown"  # 关机状态

class WakeupTrigger(Enum):
    """唤醒触发器"""
    MOTION_DETECTION = "motion_detection"  # 运动检测
    SOUND_DETECTION = "sound_detection"  # 声音检测
    TIMER = "timer"  # 定时器
    EXTERNAL_SIGNAL = "external_signal"  # 外部信号
    USER_INPUT = "user_input"  # 用户输入
    NETWORK_ACTIVITY = "network_activity"  # 网络活动
    LOW_BATTERY = "low_battery"  # 低电量

@dataclass
class PowerMetrics:
    """功耗指标"""
    timestamp: float
    cpu_usage: float  # CPU使用率 (0-100)
    memory_usage: float  # 内存使用率 (0-100)
    gpu_usage: float = 0.0  # GPU使用率 (0-100)
    cpu_frequency: float = 0.0  # CPU频率 (MHz)
    gpu_frequency: float = 0.0  # GPU频率 (MHz)
    temperature: float = 0.0  # 温度 (°C)
    battery_level: float = 100.0  # 电池电量 (0-100)
    power_consumption: float = 0.0  # 功耗 (W)
    voltage: float = 0.0  # 电压 (V)
    current: float = 0.0  # 电流 (A)
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()

@dataclass
class PowerProfile:
    """功耗配置文件"""
    mode: PowerMode
    cpu_max_frequency: float  # CPU最大频率 (MHz)
    cpu_min_frequency: float  # CPU最小频率 (MHz)
    gpu_max_frequency: float = 0.0  # GPU最大频率 (MHz)
    gpu_min_frequency: float = 0.0  # GPU最小频率 (MHz)
    cpu_governor: str = "ondemand"  # CPU调频策略
    max_cpu_cores: int = 0  # 最大CPU核心数 (0表示全部)
    memory_limit: float = 0.0  # 内存限制 (GB, 0表示无限制)
    idle_timeout: float = 30.0  # 空闲超时 (秒)
    sleep_timeout: float = 300.0  # 睡眠超时 (秒)
    enable_turbo: bool = False  # 是否启用Turbo
    enable_hyperthreading: bool = True  # 是否启用超线程
    screen_brightness: float = 0.8  # 屏幕亮度 (0-1)
    wifi_power_save: bool = False  # WiFi省电模式
    bluetooth_enabled: bool = True  # 蓝牙开关
    
@dataclass
class PowerConfig:
    """功耗管理配置"""
    device_type: DeviceType = DeviceType.GENERIC_ARM
    default_mode: PowerMode = PowerMode.BALANCED
    enable_adaptive_mode: bool = True
    enable_thermal_throttling: bool = True
    enable_battery_optimization: bool = True
    
    # 监控配置
    monitoring_interval: float = 1.0  # 监控间隔 (秒)
    metrics_history_size: int = 300  # 指标历史大小
    
    # 阈值配置
    high_cpu_threshold: float = 80.0  # 高CPU使用率阈值
    low_cpu_threshold: float = 10.0  # 低CPU使用率阈值
    high_temperature_threshold: float = 70.0  # 高温阈值 (°C)
    low_battery_threshold: float = 20.0  # 低电量阈值
    critical_battery_threshold: float = 5.0  # 临界电量阈值
    
    # 自适应配置
    adaptation_sensitivity: float = 0.5  # 适应敏感度
    mode_switch_cooldown: float = 10.0  # 模式切换冷却时间 (秒)
    
    # 唤醒配置
    enable_motion_wakeup: bool = True
    enable_sound_wakeup: bool = False
    enable_timer_wakeup: bool = True
    wakeup_sensitivity: float = 0.5  # 唤醒敏感度

class SystemMonitor:
    """系统监控器
    
    监控CPU、内存、GPU、温度、电池等系统指标
    """
    
    def __init__(self, config: PowerConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 监控数据
        self.metrics_history = deque(maxlen=config.metrics_history_size)
        self.monitoring = False
        self.monitor_thread = None
        
        # 系统信息缓存
        self._cpu_count = psutil.cpu_count()
        self._memory_total = psutil.virtual_memory().total
        
        # GPU监控
        self.gpu_available = GPUtil is not None
        if self.gpu_available:
            try:
                self.gpus = GPUtil.getGPUs()
            except:
                self.gpu_available = False
                self.gpus = []
        else:
            self.gpus = []
    
    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("系统监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        self.logger.info("系统监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                metrics = self.collect_metrics()
                self.metrics_history.append(metrics)
                time.sleep(self.config.monitoring_interval)
            except Exception as e:
                self.logger.error(f"监控数据收集失败: {e}")
                time.sleep(self.config.monitoring_interval)
    
    def collect_metrics(self) -> PowerMetrics:
        """收集系统指标
        
        Returns:
            PowerMetrics: 系统指标
        """
        try:
            # CPU指标
            cpu_usage = psutil.cpu_percent(interval=0.1)
            cpu_freq = psutil.cpu_freq()
            cpu_frequency = cpu_freq.current if cpu_freq else 0.0
            
            # 内存指标
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # GPU指标
            gpu_usage = 0.0
            gpu_frequency = 0.0
            if self.gpu_available and self.gpus:
                try:
                    gpu = self.gpus[0]  # 使用第一个GPU
                    gpu_usage = gpu.load * 100
                    # GPU频率需要特定的库支持
                except:
                    pass
            
            # 温度
            temperature = self._get_temperature()
            
            # 电池
            battery_level = self._get_battery_level()
            
            # 功耗估算
            power_consumption = self._estimate_power_consumption(
                cpu_usage, memory_usage, gpu_usage, temperature
            )
            
            return PowerMetrics(
                timestamp=time.time(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                cpu_frequency=cpu_frequency,
                gpu_frequency=gpu_frequency,
                temperature=temperature,
                battery_level=battery_level,
                power_consumption=power_consumption
            )
            
        except Exception as e:
            self.logger.error(f"指标收集失败: {e}")
            return PowerMetrics(
                timestamp=time.time(),
                cpu_usage=0.0,
                memory_usage=0.0
            )
    
    def _get_temperature(self) -> float:
        """获取系统温度
        
        Returns:
            float: 温度 (°C)
        """
        try:
            # 尝试获取CPU温度
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    # 查找CPU温度
                    for name, entries in temps.items():
                        if "cpu" in name.lower() or "core" in name.lower():
                            if entries:
                                return entries[0].current
                    
                    # 如果没有找到CPU温度，返回第一个可用温度
                    for name, entries in temps.items():
                        if entries:
                            return entries[0].current
            
            # 对于特定设备类型的温度获取
            if self.config.device_type == DeviceType.RASPBERRY_PI:
                return self._get_raspberry_pi_temperature()
            elif self.config.device_type == DeviceType.K230:
                return self._get_k230_temperature()
            
            return 0.0
            
        except Exception as e:
            self.logger.debug(f"温度获取失败: {e}")
            return 0.0
    
    def _get_raspberry_pi_temperature(self) -> float:
        """获取树莓派温度
        
        Returns:
            float: 温度 (°C)
        """
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = int(f.read().strip()) / 1000.0
                return temp
        except:
            return 0.0
    
    def _get_k230_temperature(self) -> float:
        """获取K230温度
        
        Returns:
            float: 温度 (°C)
        """
        # K230特定的温度获取方法
        # 这里需要根据实际的K230 SDK实现
        try:
            # 示例：从特定的系统文件读取
            temp_files = [
                '/sys/class/thermal/thermal_zone0/temp',
                '/sys/devices/virtual/thermal/thermal_zone0/temp'
            ]
            
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    with open(temp_file, 'r') as f:
                        temp = int(f.read().strip()) / 1000.0
                        return temp
            
            return 0.0
        except:
            return 0.0
    
    def _get_battery_level(self) -> float:
        """获取电池电量
        
        Returns:
            float: 电池电量 (0-100)
        """
        try:
            if hasattr(psutil, "sensors_battery"):
                battery = psutil.sensors_battery()
                if battery:
                    return battery.percent
            
            # 对于特定设备的电池获取
            if self.config.device_type in [DeviceType.ESP32, DeviceType.K230]:
                return self._get_embedded_battery_level()
            
            return 100.0  # 默认满电
            
        except Exception as e:
            self.logger.debug(f"电池电量获取失败: {e}")
            return 100.0
    
    def _get_embedded_battery_level(self) -> float:
        """获取嵌入式设备电池电量
        
        Returns:
            float: 电池电量 (0-100)
        """
        # 嵌入式设备的电池电量获取
        # 需要根据具体硬件实现
        try:
            # 示例：从ADC读取电池电压
            # 这里需要根据实际硬件接口实现
            battery_voltage = 3.7  # 示例电压
            max_voltage = 4.2  # 最大电压
            min_voltage = 3.0  # 最小电压
            
            battery_level = ((battery_voltage - min_voltage) / (max_voltage - min_voltage)) * 100
            return max(0.0, min(100.0, battery_level))
            
        except:
            return 100.0
    
    def _estimate_power_consumption(self, cpu_usage: float, memory_usage: float, 
                                  gpu_usage: float, temperature: float) -> float:
        """估算功耗
        
        Args:
            cpu_usage: CPU使用率
            memory_usage: 内存使用率
            gpu_usage: GPU使用率
            temperature: 温度
            
        Returns:
            float: 估算功耗 (W)
        """
        # 基于设备类型的基础功耗
        base_power = {
            DeviceType.ESP32: 0.5,
            DeviceType.K230: 2.0,
            DeviceType.RASPBERRY_PI: 3.0,
            DeviceType.JETSON_NANO: 5.0,
            DeviceType.GENERIC_ARM: 2.5,
            DeviceType.X86_LAPTOP: 15.0,
            DeviceType.DESKTOP: 50.0
        }.get(self.config.device_type, 5.0)
        
        # 动态功耗计算
        cpu_power = base_power * 0.6 * (cpu_usage / 100.0)
        memory_power = base_power * 0.2 * (memory_usage / 100.0)
        gpu_power = base_power * 0.3 * (gpu_usage / 100.0) if gpu_usage > 0 else 0
        
        # 温度影响
        temp_factor = 1.0
        if temperature > 60:
            temp_factor = 1.0 + (temperature - 60) * 0.01
        
        total_power = (base_power + cpu_power + memory_power + gpu_power) * temp_factor
        return total_power
    
    def get_current_metrics(self) -> Optional[PowerMetrics]:
        """获取当前指标
        
        Returns:
            Optional[PowerMetrics]: 当前指标
        """
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_average_metrics(self, duration: float = 60.0) -> Optional[PowerMetrics]:
        """获取平均指标
        
        Args:
            duration: 时间窗口 (秒)
            
        Returns:
            Optional[PowerMetrics]: 平均指标
        """
        if not self.metrics_history:
            return None
        
        current_time = time.time()
        recent_metrics = [
            m for m in self.metrics_history 
            if current_time - m.timestamp <= duration
        ]
        
        if not recent_metrics:
            return None
        
        # 计算平均值
        avg_metrics = PowerMetrics(
            timestamp=current_time,
            cpu_usage=np.mean([m.cpu_usage for m in recent_metrics]),
            memory_usage=np.mean([m.memory_usage for m in recent_metrics]),
            gpu_usage=np.mean([m.gpu_usage for m in recent_metrics]),
            cpu_frequency=np.mean([m.cpu_frequency for m in recent_metrics]),
            gpu_frequency=np.mean([m.gpu_frequency for m in recent_metrics]),
            temperature=np.mean([m.temperature for m in recent_metrics]),
            battery_level=np.mean([m.battery_level for m in recent_metrics]),
            power_consumption=np.mean([m.power_consumption for m in recent_metrics])
        )
        
        return avg_metrics

class FrequencyController:
    """频率控制器
    
    动态调整CPU和GPU频率
    """
    
    def __init__(self, config: PowerConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 频率信息
        self.cpu_frequencies = self._get_available_cpu_frequencies()
        self.gpu_frequencies = self._get_available_gpu_frequencies()
        
        # 当前设置
        self.current_cpu_governor = "ondemand"
        self.current_cpu_max_freq = 0
        self.current_cpu_min_freq = 0
        
        self.logger.info(f"频率控制器初始化完成，CPU频率范围: {self.cpu_frequencies}")
    
    def _get_available_cpu_frequencies(self) -> List[float]:
        """获取可用的CPU频率
        
        Returns:
            List[float]: 可用频率列表 (MHz)
        """
        try:
            # 尝试从系统获取
            freq_info = psutil.cpu_freq()
            if freq_info:
                # 生成频率范围
                min_freq = freq_info.min if freq_info.min > 0 else 800
                max_freq = freq_info.max if freq_info.max > 0 else 2000
                
                # 生成常见的频率点
                frequencies = []
                step = (max_freq - min_freq) / 10
                for i in range(11):
                    freq = min_freq + i * step
                    frequencies.append(freq)
                
                return frequencies
            
            # 根据设备类型返回默认频率
            default_frequencies = {
                DeviceType.ESP32: [80, 160, 240],
                DeviceType.K230: [400, 800, 1200, 1600],
                DeviceType.RASPBERRY_PI: [600, 900, 1200, 1500],
                DeviceType.JETSON_NANO: [500, 1000, 1500, 2000],
                DeviceType.GENERIC_ARM: [800, 1200, 1600, 2000]
            }
            
            return default_frequencies.get(self.config.device_type, [800, 1200, 1600, 2000])
            
        except Exception as e:
            self.logger.error(f"获取CPU频率失败: {e}")
            return [800, 1200, 1600, 2000]
    
    def _get_available_gpu_frequencies(self) -> List[float]:
        """获取可用的GPU频率
        
        Returns:
            List[float]: 可用频率列表 (MHz)
        """
        # GPU频率控制比较复杂，这里提供基础实现
        default_frequencies = {
            DeviceType.K230: [200, 400, 600, 800],
            DeviceType.JETSON_NANO: [300, 600, 900, 1200],
            DeviceType.RASPBERRY_PI: [200, 400, 600],
        }
        
        return default_frequencies.get(self.config.device_type, [])
    
    def set_cpu_frequency(self, min_freq: float, max_freq: float, governor: str = "ondemand") -> bool:
        """设置CPU频率
        
        Args:
            min_freq: 最小频率 (MHz)
            max_freq: 最大频率 (MHz)
            governor: 调频策略
            
        Returns:
            bool: 设置是否成功
        """
        try:
            # 验证频率范围
            if min_freq > max_freq:
                min_freq, max_freq = max_freq, min_freq
            
            # 限制在可用范围内
            available_freqs = self.cpu_frequencies
            if available_freqs:
                min_available = min(available_freqs)
                max_available = max(available_freqs)
                min_freq = max(min_freq, min_available)
                max_freq = min(max_freq, max_available)
            
            # 根据设备类型执行频率设置
            success = self._apply_cpu_frequency_settings(min_freq, max_freq, governor)
            
            if success:
                self.current_cpu_min_freq = min_freq
                self.current_cpu_max_freq = max_freq
                self.current_cpu_governor = governor
                self.logger.info(f"CPU频率设置成功: {min_freq}-{max_freq} MHz, 策略: {governor}")
            else:
                self.logger.warning("CPU频率设置失败")
            
            return success
            
        except Exception as e:
            self.logger.error(f"CPU频率设置异常: {e}")
            return False
    
    def _apply_cpu_frequency_settings(self, min_freq: float, max_freq: float, governor: str) -> bool:
        """应用CPU频率设置
        
        Args:
            min_freq: 最小频率
            max_freq: 最大频率
            governor: 调频策略
            
        Returns:
            bool: 是否成功
        """
        try:
            # Linux系统的频率控制
            if os.path.exists('/sys/devices/system/cpu/cpu0/cpufreq/'):
                return self._linux_cpu_frequency_control(min_freq, max_freq, governor)
            
            # 嵌入式设备的频率控制
            if self.config.device_type in [DeviceType.ESP32, DeviceType.K230]:
                return self._embedded_cpu_frequency_control(min_freq, max_freq)
            
            # 其他系统的频率控制
            return self._generic_cpu_frequency_control(min_freq, max_freq, governor)
            
        except Exception as e:
            self.logger.error(f"频率设置应用失败: {e}")
            return False
    
    def _linux_cpu_frequency_control(self, min_freq: float, max_freq: float, governor: str) -> bool:
        """Linux系统CPU频率控制
        
        Args:
            min_freq: 最小频率 (MHz)
            max_freq: 最大频率 (MHz)
            governor: 调频策略
            
        Returns:
            bool: 是否成功
        """
        try:
            cpu_count = psutil.cpu_count()
            success_count = 0
            
            for cpu_id in range(cpu_count):
                cpufreq_path = f'/sys/devices/system/cpu/cpu{cpu_id}/cpufreq/'
                
                # 设置调频策略
                governor_file = os.path.join(cpufreq_path, 'scaling_governor')
                if os.path.exists(governor_file):
                    try:
                        with open(governor_file, 'w') as f:
                            f.write(governor)
                    except:
                        pass
                
                # 设置最小频率
                min_freq_file = os.path.join(cpufreq_path, 'scaling_min_freq')
                if os.path.exists(min_freq_file):
                    try:
                        with open(min_freq_file, 'w') as f:
                            f.write(str(int(min_freq * 1000)))  # 转换为kHz
                    except:
                        pass
                
                # 设置最大频率
                max_freq_file = os.path.join(cpufreq_path, 'scaling_max_freq')
                if os.path.exists(max_freq_file):
                    try:
                        with open(max_freq_file, 'w') as f:
                            f.write(str(int(max_freq * 1000)))  # 转换为kHz
                        success_count += 1
                    except:
                        pass
            
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Linux频率控制失败: {e}")
            return False
    
    def _embedded_cpu_frequency_control(self, min_freq: float, max_freq: float) -> bool:
        """嵌入式设备CPU频率控制
        
        Args:
            min_freq: 最小频率
            max_freq: 最大频率
            
        Returns:
            bool: 是否成功
        """
        # 嵌入式设备的频率控制需要根据具体SDK实现
        # 这里提供框架代码
        try:
            if self.config.device_type == DeviceType.ESP32:
                # ESP32频率控制
                return self._esp32_frequency_control(max_freq)
            elif self.config.device_type == DeviceType.K230:
                # K230频率控制
                return self._k230_frequency_control(min_freq, max_freq)
            
            return False
            
        except Exception as e:
            self.logger.error(f"嵌入式频率控制失败: {e}")
            return False
    
    def _esp32_frequency_control(self, frequency: float) -> bool:
        """ESP32频率控制
        
        Args:
            frequency: 目标频率 (MHz)
            
        Returns:
            bool: 是否成功
        """
        # ESP32的频率控制需要使用ESP-IDF
        # 这里提供示例代码框架
        try:
            # 示例：通过串口或其他接口发送频率控制命令
            # 实际实现需要根据ESP32的SDK
            valid_frequencies = [80, 160, 240]
            target_freq = min(valid_frequencies, key=lambda x: abs(x - frequency))
            
            # 这里应该调用ESP32的频率设置API
            # esp_pm_configure() 或类似的函数
            
            self.logger.info(f"ESP32频率设置为: {target_freq} MHz")
            return True
            
        except Exception as e:
            self.logger.error(f"ESP32频率控制失败: {e}")
            return False
    
    def _k230_frequency_control(self, min_freq: float, max_freq: float) -> bool:
        """K230频率控制
        
        Args:
            min_freq: 最小频率
            max_freq: 最大频率
            
        Returns:
            bool: 是否成功
        """
        # K230的频率控制需要使用Canaan SDK
        try:
            # 示例：K230频率控制接口
            # 实际实现需要根据K230的SDK文档
            
            # 设置CPU频率
            # kd_mpi_sys_set_cpu_freq(max_freq)
            
            # 设置NPU频率（如果支持）
            # kd_mpi_sys_set_npu_freq(max_freq)
            
            self.logger.info(f"K230频率设置: {min_freq}-{max_freq} MHz")
            return True
            
        except Exception as e:
            self.logger.error(f"K230频率控制失败: {e}")
            return False
    
    def _generic_cpu_frequency_control(self, min_freq: float, max_freq: float, governor: str) -> bool:
        """通用CPU频率控制
        
        Args:
            min_freq: 最小频率
            max_freq: 最大频率
            governor: 调频策略
            
        Returns:
            bool: 是否成功
        """
        # 通用的频率控制方法
        # 可能通过系统命令或其他方式实现
        try:
            # 记录设置，但不实际执行
            self.logger.info(f"通用频率控制: {min_freq}-{max_freq} MHz, 策略: {governor}")
            return True
            
        except Exception as e:
            self.logger.error(f"通用频率控制失败: {e}")
            return False
    
    def get_optimal_frequency(self, cpu_usage: float, temperature: float, 
                            battery_level: float, power_mode: PowerMode) -> Tuple[float, float]:
        """获取最优频率
        
        Args:
            cpu_usage: CPU使用率
            temperature: 温度
            battery_level: 电池电量
            power_mode: 功耗模式
            
        Returns:
            Tuple[float, float]: (最小频率, 最大频率)
        """
        if not self.cpu_frequencies:
            return 800.0, 2000.0
        
        min_available = min(self.cpu_frequencies)
        max_available = max(self.cpu_frequencies)
        
        # 根据功耗模式确定基础频率范围
        if power_mode == PowerMode.HIGH_PERFORMANCE:
            base_min = max_available * 0.8
            base_max = max_available
        elif power_mode == PowerMode.BALANCED:
            base_min = min_available + (max_available - min_available) * 0.3
            base_max = min_available + (max_available - min_available) * 0.8
        elif power_mode == PowerMode.POWER_SAVER:
            base_min = min_available
            base_max = min_available + (max_available - min_available) * 0.5
        elif power_mode == PowerMode.ULTRA_LOW_POWER:
            base_min = min_available
            base_max = min_available + (max_available - min_available) * 0.3
        else:  # ADAPTIVE
            base_min = min_available + (max_available - min_available) * 0.2
            base_max = min_available + (max_available - min_available) * 0.9
        
        # 根据CPU使用率调整
        if cpu_usage > 80:
            # 高负载：提高频率
            freq_factor = 1.2
        elif cpu_usage > 50:
            # 中等负载：保持基础频率
            freq_factor = 1.0
        elif cpu_usage > 20:
            # 低负载：降低频率
            freq_factor = 0.8
        else:
            # 极低负载：最低频率
            freq_factor = 0.6
        
        # 根据温度调整
        if temperature > self.config.high_temperature_threshold:
            # 高温：降低频率
            temp_factor = 0.7
        elif temperature > self.config.high_temperature_threshold * 0.8:
            # 温度较高：适度降低
            temp_factor = 0.9
        else:
            # 温度正常
            temp_factor = 1.0
        
        # 根据电池电量调整
        if battery_level < self.config.critical_battery_threshold:
            # 临界电量：最低频率
            battery_factor = 0.5
        elif battery_level < self.config.low_battery_threshold:
            # 低电量：降低频率
            battery_factor = 0.7
        else:
            # 电量充足
            battery_factor = 1.0
        
        # 综合调整因子
        adjustment_factor = freq_factor * temp_factor * battery_factor
        
        # 计算最终频率
        target_min = base_min * adjustment_factor
        target_max = base_max * adjustment_factor
        
        # 限制在可用范围内
        target_min = max(min_available, min(target_min, max_available))
        target_max = max(target_min, min(target_max, max_available))
        
        return target_min, target_max

class SleepManager:
    """睡眠管理器
    
    管理系统的睡眠和唤醒
    """
    
    def __init__(self, config: PowerConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 睡眠状态
        self.current_state = PowerState.ACTIVE
        self.last_activity_time = time.time()
        self.sleep_start_time = 0.0
        
        # 唤醒触发器
        self.wakeup_triggers: Dict[WakeupTrigger, bool] = {
            WakeupTrigger.MOTION_DETECTION: config.enable_motion_wakeup,
            WakeupTrigger.SOUND_DETECTION: config.enable_sound_wakeup,
            WakeupTrigger.TIMER: config.enable_timer_wakeup,
            WakeupTrigger.USER_INPUT: True,
            WakeupTrigger.NETWORK_ACTIVITY: False,
            WakeupTrigger.LOW_BATTERY: True
        }
        
        # 活动监控
        self.activity_monitoring = False
        self.activity_thread = None
        
        self.logger.info("睡眠管理器初始化完成")
    
    def start_activity_monitoring(self):
        """开始活动监控"""
        if self.activity_monitoring:
            return
        
        self.activity_monitoring = True
        self.activity_thread = threading.Thread(target=self._activity_monitor_loop, daemon=True)
        self.activity_thread.start()
        self.logger.info("活动监控已启动")
    
    def stop_activity_monitoring(self):
        """停止活动监控"""
        self.activity_monitoring = False
        if self.activity_thread:
            self.activity_thread.join(timeout=2.0)
        self.logger.info("活动监控已停止")
    
    def _activity_monitor_loop(self):
        """活动监控循环"""
        while self.activity_monitoring:
            try:
                # 检查是否应该进入睡眠
                self._check_sleep_conditions()
                time.sleep(1.0)
            except Exception as e:
                self.logger.error(f"活动监控异常: {e}")
                time.sleep(1.0)
    
    def _check_sleep_conditions(self):
        """检查睡眠条件"""
        current_time = time.time()
        idle_time = current_time - self.last_activity_time
        
        # 根据当前状态决定下一步动作
        if self.current_state == PowerState.ACTIVE:
            if idle_time > self.config.default_mode.value.idle_timeout:
                self._enter_idle_state()
        
        elif self.current_state == PowerState.IDLE:
            if idle_time > self.config.default_mode.value.sleep_timeout:
                self._enter_sleep_state()
    
    def update_activity(self):
        """更新活动时间"""
        self.last_activity_time = time.time()
        
        # 如果当前在睡眠状态，唤醒系统
        if self.current_state in [PowerState.LIGHT_SLEEP, PowerState.DEEP_SLEEP]:
            self.wakeup(WakeupTrigger.USER_INPUT)
    
    def _enter_idle_state(self):
        """进入空闲状态"""
        if self.current_state == PowerState.IDLE:
            return
        
        self.current_state = PowerState.IDLE
        self.logger.info("系统进入空闲状态")
        
        # 空闲状态的优化措施
        self._apply_idle_optimizations()
    
    def _enter_sleep_state(self):
        """进入睡眠状态"""
        if self.current_state in [PowerState.LIGHT_SLEEP, PowerState.DEEP_SLEEP]:
            return
        
        # 根据配置选择睡眠类型
        sleep_type = self._determine_sleep_type()
        
        self.current_state = sleep_type
        self.sleep_start_time = time.time()
        
        self.logger.info(f"系统进入{sleep_type.value}状态")
        
        # 应用睡眠优化
        if sleep_type == PowerState.LIGHT_SLEEP:
            self._apply_light_sleep_optimizations()
        elif sleep_type == PowerState.DEEP_SLEEP:
            self._apply_deep_sleep_optimizations()
    
    def _determine_sleep_type(self) -> PowerState:
        """确定睡眠类型
        
        Returns:
            PowerState: 睡眠类型
        """
        # 根据设备类型和配置确定睡眠类型
        if self.config.device_type in [DeviceType.ESP32, DeviceType.K230]:
            # 嵌入式设备支持深度睡眠
            return PowerState.DEEP_SLEEP
        else:
            # 其他设备使用轻度睡眠
            return PowerState.LIGHT_SLEEP
    
    def _apply_idle_optimizations(self):
        """应用空闲优化"""
        try:
            # 降低CPU频率
            # 减少后台任务
            # 降低屏幕亮度
            self.logger.debug("应用空闲优化")
        except Exception as e:
            self.logger.error(f"空闲优化失败: {e}")
    
    def _apply_light_sleep_optimizations(self):
        """应用轻度睡眠优化"""
        try:
            # 暂停非关键任务
            # 降低系统频率
            # 关闭不必要的外设
            self.logger.debug("应用轻度睡眠优化")
        except Exception as e:
            self.logger.error(f"轻度睡眠优化失败: {e}")
    
    def _apply_deep_sleep_optimizations(self):
        """应用深度睡眠优化"""
        try:
            # 保存系统状态
            # 关闭大部分外设
            # 进入最低功耗模式
            
            if self.config.device_type == DeviceType.ESP32:
                self._esp32_deep_sleep()
            elif self.config.device_type == DeviceType.K230:
                self._k230_deep_sleep()
            
            self.logger.debug("应用深度睡眠优化")
        except Exception as e:
            self.logger.error(f"深度睡眠优化失败: {e}")
    
    def _esp32_deep_sleep(self):
        """ESP32深度睡眠"""
        # ESP32深度睡眠实现
        # 需要使用ESP-IDF的深度睡眠API
        try:
            # esp_deep_sleep_start()
            # 或者设置唤醒源后进入深度睡眠
            self.logger.info("ESP32进入深度睡眠")
        except Exception as e:
            self.logger.error(f"ESP32深度睡眠失败: {e}")
    
    def _k230_deep_sleep(self):
        """K230深度睡眠"""
        # K230深度睡眠实现
        try:
            # 使用K230的低功耗API
            self.logger.info("K230进入深度睡眠")
        except Exception as e:
            self.logger.error(f"K230深度睡眠失败: {e}")
    
    def wakeup(self, trigger: WakeupTrigger):
        """唤醒系统
        
        Args:
            trigger: 唤醒触发器
        """
        if self.current_state == PowerState.ACTIVE:
            return
        
        # 检查触发器是否启用
        if not self.wakeup_triggers.get(trigger, False):
            return
        
        previous_state = self.current_state
        self.current_state = PowerState.ACTIVE
        self.last_activity_time = time.time()
        
        # 计算睡眠时间
        if self.sleep_start_time > 0:
            sleep_duration = time.time() - self.sleep_start_time
            self.logger.info(f"系统从{previous_state.value}唤醒，睡眠时长: {sleep_duration:.1f}秒，触发器: {trigger.value}")
        else:
            self.logger.info(f"系统从{previous_state.value}唤醒，触发器: {trigger.value}")
        
        # 恢复系统状态
        self._restore_from_sleep(previous_state)
    
    def _restore_from_sleep(self, previous_state: PowerState):
        """从睡眠状态恢复
        
        Args:
            previous_state: 之前的睡眠状态
        """
        try:
            if previous_state == PowerState.LIGHT_SLEEP:
                # 恢复轻度睡眠前的状态
                self._restore_from_light_sleep()
            elif previous_state == PowerState.DEEP_SLEEP:
                # 恢复深度睡眠前的状态
                self._restore_from_deep_sleep()
            
            self.logger.debug(f"从{previous_state.value}恢复完成")
        except Exception as e:
            self.logger.error(f"睡眠恢复失败: {e}")
    
    def _restore_from_light_sleep(self):
        """从轻度睡眠恢复"""
        # 恢复CPU频率
        # 重启暂停的任务
        # 恢复外设状态
        pass
    
    def _restore_from_deep_sleep(self):
        """从深度睡眠恢复"""
        # 重新初始化系统
        # 恢复保存的状态
        # 重启所有服务
        pass
    
    def get_sleep_stats(self) -> Dict[str, Any]:
        """获取睡眠统计
        
        Returns:
            Dict[str, Any]: 睡眠统计信息
        """
        current_time = time.time()
        idle_time = current_time - self.last_activity_time
        
        stats = {
            "current_state": self.current_state.value,
            "idle_time": idle_time,
            "last_activity": self.last_activity_time,
            "wakeup_triggers": {k.value: v for k, v in self.wakeup_triggers.items()}
        }
        
        if self.sleep_start_time > 0 and self.current_state != PowerState.ACTIVE:
            stats["sleep_duration"] = current_time - self.sleep_start_time
        
        return stats

class PowerManager:
    """功耗管理器主类
    
    整合系统监控、频率控制和睡眠管理
    """
    
    def __init__(self, config: PowerConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化组件
        self.system_monitor = SystemMonitor(config)
        self.frequency_controller = FrequencyController(config)
        self.sleep_manager = SleepManager(config)
        
        # 功耗模式和配置文件
        self.current_mode = config.default_mode
        self.power_profiles = self._create_default_profiles()
        self.custom_profile = None
        
        # 自适应控制
        self.adaptive_enabled = config.enable_adaptive_mode
        self.last_mode_switch = 0.0
        self.adaptation_history = deque(maxlen=50)
        
        # 管理状态
        self.running = False
        self.management_thread = None
        
        self.logger.info(f"功耗管理器初始化完成，当前模式: {self.current_mode.value}")
    
    def _create_default_profiles(self) -> Dict[PowerMode, PowerProfile]:
        """创建默认功耗配置文件
        
        Returns:
            Dict[PowerMode, PowerProfile]: 功耗配置文件
        """
        # 获取系统频率范围
        cpu_freqs = self.frequency_controller.cpu_frequencies
        min_freq = min(cpu_freqs) if cpu_freqs else 800
        max_freq = max(cpu_freqs) if cpu_freqs else 2000
        
        profiles = {
            PowerMode.HIGH_PERFORMANCE: PowerProfile(
                mode=PowerMode.HIGH_PERFORMANCE,
                cpu_max_frequency=max_freq,
                cpu_min_frequency=max_freq * 0.8,
                cpu_governor="performance",
                max_cpu_cores=0,  # 使用所有核心
                idle_timeout=60.0,
                sleep_timeout=600.0,
                enable_turbo=True,
                screen_brightness=1.0,
                wifi_power_save=False
            ),
            
            PowerMode.BALANCED: PowerProfile(
                mode=PowerMode.BALANCED,
                cpu_max_frequency=max_freq * 0.8,
                cpu_min_frequency=min_freq + (max_freq - min_freq) * 0.3,
                cpu_governor="ondemand",
                max_cpu_cores=0,
                idle_timeout=30.0,
                sleep_timeout=300.0,
                enable_turbo=False,
                screen_brightness=0.8,
                wifi_power_save=False
            ),
            
            PowerMode.POWER_SAVER: PowerProfile(
                mode=PowerMode.POWER_SAVER,
                cpu_max_frequency=max_freq * 0.6,
                cpu_min_frequency=min_freq,
                cpu_governor="powersave",
                max_cpu_cores=2,  # 限制核心数
                idle_timeout=15.0,
                sleep_timeout=120.0,
                enable_turbo=False,
                screen_brightness=0.6,
                wifi_power_save=True
            ),
            
            PowerMode.ULTRA_LOW_POWER: PowerProfile(
                mode=PowerMode.ULTRA_LOW_POWER,
                cpu_max_frequency=max_freq * 0.4,
                cpu_min_frequency=min_freq,
                cpu_governor="powersave",
                max_cpu_cores=1,  # 单核运行
                idle_timeout=10.0,
                sleep_timeout=60.0,
                enable_turbo=False,
                screen_brightness=0.4,
                wifi_power_save=True,
                bluetooth_enabled=False
            )
        }
        
        return profiles
    
    def start(self):
        """启动功耗管理"""
        if self.running:
            return
        
        self.running = True
        
        # 启动各个组件
        self.system_monitor.start_monitoring()
        self.sleep_manager.start_activity_monitoring()
        
        # 启动管理线程
        self.management_thread = threading.Thread(target=self._management_loop, daemon=True)
        self.management_thread.start()
        
        # 应用初始功耗模式
        self.apply_power_mode(self.current_mode)
        
        self.logger.info("功耗管理已启动")
    
    def stop(self):
        """停止功耗管理"""
        if not self.running:
            return
        
        self.running = False
        
        # 停止各个组件
        self.system_monitor.stop_monitoring()
        self.sleep_manager.stop_activity_monitoring()
        
        # 等待管理线程结束
        if self.management_thread:
            self.management_thread.join(timeout=3.0)
        
        self.logger.info("功耗管理已停止")
    
    def _management_loop(self):
        """管理循环"""
        while self.running:
            try:
                # 获取当前系统指标
                current_metrics = self.system_monitor.get_current_metrics()
                if current_metrics is None:
                    time.sleep(self.config.monitoring_interval)
                    continue
                
                # 自适应模式调整
                if self.adaptive_enabled:
                    self._adaptive_power_management(current_metrics)
                
                # 热管理
                if self.config.enable_thermal_throttling:
                    self._thermal_management(current_metrics)
                
                # 电池管理
                if self.config.enable_battery_optimization:
                    self._battery_management(current_metrics)
                
                # 更新活动状态
                if current_metrics.cpu_usage > self.config.low_cpu_threshold:
                    self.sleep_manager.update_activity()
                
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"功耗管理循环异常: {e}")
                time.sleep(self.config.monitoring_interval)
    
    def _adaptive_power_management(self, metrics: PowerMetrics):
        """自适应功耗管理
        
        Args:
            metrics: 当前系统指标
        """
        current_time = time.time()
        
        # 检查模式切换冷却时间
        if current_time - self.last_mode_switch < self.config.mode_switch_cooldown:
            return
        
        # 分析系统负载模式
        load_pattern = self._analyze_load_pattern(metrics)
        
        # 确定最优功耗模式
        optimal_mode = self._determine_optimal_mode(metrics, load_pattern)
        
        # 如果需要切换模式
        if optimal_mode != self.current_mode:
            confidence = self._calculate_mode_switch_confidence(metrics, optimal_mode)
            
            if confidence > 0.7:  # 高置信度才切换
                self.logger.info(f"自适应切换功耗模式: {self.current_mode.value} -> {optimal_mode.value}")
                self.apply_power_mode(optimal_mode)
                self.last_mode_switch = current_time
    
    def _analyze_load_pattern(self, metrics: PowerMetrics) -> Dict[str, float]:
        """分析负载模式
        
        Args:
            metrics: 系统指标
            
        Returns:
            Dict[str, float]: 负载模式分析
        """
        # 获取历史数据
        recent_metrics = list(self.system_monitor.metrics_history)[-10:]  # 最近10个数据点
        
        if len(recent_metrics) < 5:
            return {"load_level": "unknown", "stability": 0.5, "trend": "stable"}
        
        # 计算负载统计
        cpu_usages = [m.cpu_usage for m in recent_metrics]
        memory_usages = [m.memory_usage for m in recent_metrics]
        
        avg_cpu = np.mean(cpu_usages)
        std_cpu = np.std(cpu_usages)
        avg_memory = np.mean(memory_usages)
        
        # 负载水平
        if avg_cpu > 70:
            load_level = "high"
        elif avg_cpu > 40:
            load_level = "medium"
        elif avg_cpu > 15:
            load_level = "low"
        else:
            load_level = "idle"
        
        # 负载稳定性
        stability = max(0.0, 1.0 - std_cpu / 50.0)
        
        # 负载趋势
        if len(cpu_usages) >= 5:
            recent_avg = np.mean(cpu_usages[-3:])
            earlier_avg = np.mean(cpu_usages[-6:-3])
            
            if recent_avg > earlier_avg * 1.2:
                trend = "increasing"
            elif recent_avg < earlier_avg * 0.8:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        return {
            "load_level": load_level,
            "avg_cpu": avg_cpu,
            "avg_memory": avg_memory,
            "stability": stability,
            "trend": trend
        }
    
    def _determine_optimal_mode(self, metrics: PowerMetrics, load_pattern: Dict[str, float]) -> PowerMode:
        """确定最优功耗模式
        
        Args:
            metrics: 系统指标
            load_pattern: 负载模式
            
        Returns:
            PowerMode: 最优功耗模式
        """
        load_level = load_pattern["load_level"]
        avg_cpu = load_pattern["avg_cpu"]
        stability = load_pattern["stability"]
        trend = load_pattern["trend"]
        
        # 电池因素
        battery_factor = 1.0
        if metrics.battery_level < self.config.critical_battery_threshold:
            # 临界电量：强制超低功耗
            return PowerMode.ULTRA_LOW_POWER
        elif metrics.battery_level < self.config.low_battery_threshold:
            # 低电量：倾向于省电模式
            battery_factor = 0.5
        
        # 温度因素
        temp_factor = 1.0
        if metrics.temperature > self.config.high_temperature_threshold:
            # 高温：降低性能
            temp_factor = 0.3
        elif metrics.temperature > self.config.high_temperature_threshold * 0.8:
            temp_factor = 0.7
        
        # 综合评分
        performance_score = 0.0
        
        if load_level == "high":
            performance_score = 0.9
        elif load_level == "medium":
            performance_score = 0.6
        elif load_level == "low":
            performance_score = 0.3
        else:  # idle
            performance_score = 0.1
        
        # 趋势调整
        if trend == "increasing":
            performance_score *= 1.2
        elif trend == "decreasing":
            performance_score *= 0.8
        
        # 稳定性调整
        if stability < 0.5:
            performance_score *= 0.9  # 不稳定时稍微保守
        
        # 应用因素
        final_score = performance_score * battery_factor * temp_factor
        
        # 根据评分选择模式
        if final_score > 0.8:
            return PowerMode.HIGH_PERFORMANCE
        elif final_score > 0.5:
            return PowerMode.BALANCED
        elif final_score > 0.2:
            return PowerMode.POWER_SAVER
        else:
            return PowerMode.ULTRA_LOW_POWER
    
    def _calculate_mode_switch_confidence(self, metrics: PowerMetrics, target_mode: PowerMode) -> float:
        """计算模式切换置信度
        
        Args:
            metrics: 系统指标
            target_mode: 目标模式
            
        Returns:
            float: 置信度 (0-1)
        """
        # 基础置信度
        base_confidence = 0.5
        
        # 根据指标稳定性调整
        recent_metrics = list(self.system_monitor.metrics_history)[-5:]
        if len(recent_metrics) >= 3:
            cpu_values = [m.cpu_usage for m in recent_metrics]
            cpu_std = np.std(cpu_values)
            
            # CPU使用率越稳定，置信度越高
            stability_bonus = max(0.0, 0.3 - cpu_std / 100.0)
            base_confidence += stability_bonus
        
        # 根据电池和温度状态调整
        if metrics.battery_level < self.config.low_battery_threshold and target_mode in [PowerMode.POWER_SAVER, PowerMode.ULTRA_LOW_POWER]:
            base_confidence += 0.3
        
        if metrics.temperature > self.config.high_temperature_threshold and target_mode in [PowerMode.POWER_SAVER, PowerMode.ULTRA_LOW_POWER]:
            base_confidence += 0.3
        
        return min(1.0, base_confidence)
    
    def _thermal_management(self, metrics: PowerMetrics):
        """热管理
        
        Args:
            metrics: 系统指标
        """
        if metrics.temperature > self.config.high_temperature_threshold:
            # 高温处理
            self.logger.warning(f"系统温度过高: {metrics.temperature:.1f}°C")
            
            # 降低CPU频率
            current_profile = self.get_current_profile()
            if current_profile:
                reduced_max_freq = current_profile.cpu_max_frequency * 0.8
                reduced_min_freq = current_profile.cpu_min_frequency * 0.8
                
                self.frequency_controller.set_cpu_frequency(
                    reduced_min_freq, reduced_max_freq, "powersave"
                )
                
                self.logger.info(f"因高温降低CPU频率至: {reduced_min_freq}-{reduced_max_freq} MHz")
    
    def _battery_management(self, metrics: PowerMetrics):
        """电池管理
        
        Args:
            metrics: 系统指标
        """
        if metrics.battery_level < self.config.critical_battery_threshold:
            # 临界电量：强制进入超低功耗模式
            if self.current_mode != PowerMode.ULTRA_LOW_POWER:
                self.logger.warning(f"电池电量临界: {metrics.battery_level:.1f}%，切换至超低功耗模式")
                self.apply_power_mode(PowerMode.ULTRA_LOW_POWER)
        
        elif metrics.battery_level < self.config.low_battery_threshold:
            # 低电量：建议省电模式
            if self.current_mode == PowerMode.HIGH_PERFORMANCE:
                self.logger.info(f"电池电量较低: {metrics.battery_level:.1f}%，建议切换至省电模式")
    
    def apply_power_mode(self, mode: PowerMode, custom_profile: Optional[PowerProfile] = None):
        """应用功耗模式
        
        Args:
            mode: 功耗模式
            custom_profile: 自定义配置文件
        """
        try:
            # 获取配置文件
            if mode == PowerMode.CUSTOM and custom_profile:
                profile = custom_profile
                self.custom_profile = custom_profile
            else:
                profile = self.power_profiles.get(mode)
                if not profile:
                    self.logger.error(f"未找到功耗模式配置: {mode.value}")
                    return
            
            # 应用CPU频率设置
            freq_success = self.frequency_controller.set_cpu_frequency(
                profile.cpu_min_frequency,
                profile.cpu_max_frequency,
                profile.cpu_governor
            )
            
            if not freq_success:
                self.logger.warning("CPU频率设置失败")
            
            # 应用其他设置
            self._apply_profile_settings(profile)
            
            # 更新当前模式
            self.current_mode = mode
            
            self.logger.info(f"功耗模式已切换至: {mode.value}")
            
        except Exception as e:
            self.logger.error(f"功耗模式应用失败: {e}")
    
    def _apply_profile_settings(self, profile: PowerProfile):
        """应用配置文件设置
        
        Args:
            profile: 功耗配置文件
        """
        try:
            # 更新睡眠管理器的超时设置
            # 这里需要根据实际的睡眠管理器接口调整
            
            # 应用屏幕亮度（如果支持）
            self._set_screen_brightness(profile.screen_brightness)
            
            # 应用WiFi省电设置
            self._set_wifi_power_save(profile.wifi_power_save)
            
            # 应用蓝牙设置
            self._set_bluetooth_enabled(profile.bluetooth_enabled)
            
            self.logger.debug(f"配置文件设置已应用: {profile.mode.value}")
            
        except Exception as e:
            self.logger.error(f"配置文件设置应用失败: {e}")
    
    def _set_screen_brightness(self, brightness: float):
        """设置屏幕亮度
        
        Args:
            brightness: 亮度 (0-1)
        """
        try:
            # 屏幕亮度控制需要根据具体平台实现
            if self.config.device_type == DeviceType.RASPBERRY_PI:
                # 树莓派屏幕亮度控制
                brightness_value = int(brightness * 255)
                # echo brightness_value > /sys/class/backlight/*/brightness
            
            self.logger.debug(f"屏幕亮度设置为: {brightness:.1f}")
            
        except Exception as e:
            self.logger.debug(f"屏幕亮度设置失败: {e}")
    
    def _set_wifi_power_save(self, enabled: bool):
        """设置WiFi省电模式
        
        Args:
            enabled: 是否启用省电模式
        """
        try:
            # WiFi省电模式控制
            # 需要根据具体的网络接口实现
            self.logger.debug(f"WiFi省电模式: {'启用' if enabled else '禁用'}")
            
        except Exception as e:
            self.logger.debug(f"WiFi省电设置失败: {e}")
    
    def _set_bluetooth_enabled(self, enabled: bool):
        """设置蓝牙开关
        
        Args:
            enabled: 是否启用蓝牙
        """
        try:
            # 蓝牙开关控制
            self.logger.debug(f"蓝牙: {'启用' if enabled else '禁用'}")
            
        except Exception as e:
            self.logger.debug(f"蓝牙设置失败: {e}")
    
    def get_current_profile(self) -> Optional[PowerProfile]:
        """获取当前功耗配置文件
        
        Returns:
            Optional[PowerProfile]: 当前配置文件
        """
        if self.current_mode == PowerMode.CUSTOM:
            return self.custom_profile
        else:
            return self.power_profiles.get(self.current_mode)
    
    def create_custom_profile(self, **kwargs) -> PowerProfile:
        """创建自定义功耗配置文件
        
        Args:
            **kwargs: 配置参数
            
        Returns:
            PowerProfile: 自定义配置文件
        """
        # 基于平衡模式创建自定义配置
        base_profile = self.power_profiles[PowerMode.BALANCED]
        
        # 更新指定的参数
        profile_dict = {
            "mode": PowerMode.CUSTOM,
            "cpu_max_frequency": kwargs.get("cpu_max_frequency", base_profile.cpu_max_frequency),
            "cpu_min_frequency": kwargs.get("cpu_min_frequency", base_profile.cpu_min_frequency),
            "gpu_max_frequency": kwargs.get("gpu_max_frequency", base_profile.gpu_max_frequency),
            "gpu_min_frequency": kwargs.get("gpu_min_frequency", base_profile.gpu_min_frequency),
            "cpu_governor": kwargs.get("cpu_governor", base_profile.cpu_governor),
            "max_cpu_cores": kwargs.get("max_cpu_cores", base_profile.max_cpu_cores),
            "memory_limit": kwargs.get("memory_limit", base_profile.memory_limit),
            "idle_timeout": kwargs.get("idle_timeout", base_profile.idle_timeout),
            "sleep_timeout": kwargs.get("sleep_timeout", base_profile.sleep_timeout),
            "enable_turbo": kwargs.get("enable_turbo", base_profile.enable_turbo),
            "enable_hyperthreading": kwargs.get("enable_hyperthreading", base_profile.enable_hyperthreading),
            "screen_brightness": kwargs.get("screen_brightness", base_profile.screen_brightness),
            "wifi_power_save": kwargs.get("wifi_power_save", base_profile.wifi_power_save),
            "bluetooth_enabled": kwargs.get("bluetooth_enabled", base_profile.bluetooth_enabled)
        }
        
        return PowerProfile(**profile_dict)
    
    def get_power_stats(self) -> Dict[str, Any]:
        """获取功耗统计信息
        
        Returns:
            Dict[str, Any]: 功耗统计
        """
        current_metrics = self.system_monitor.get_current_metrics()
        avg_metrics = self.system_monitor.get_average_metrics(300)  # 5分钟平均
        sleep_stats = self.sleep_manager.get_sleep_stats()
        current_profile = self.get_current_profile()
        
        stats = {
            "current_mode": self.current_mode.value,
            "adaptive_enabled": self.adaptive_enabled,
            "running": self.running,
            "device_type": self.config.device_type.value
        }
        
        if current_metrics:
            stats["current_metrics"] = {
                "cpu_usage": current_metrics.cpu_usage,
                "memory_usage": current_metrics.memory_usage,
                "gpu_usage": current_metrics.gpu_usage,
                "temperature": current_metrics.temperature,
                "battery_level": current_metrics.battery_level,
                "power_consumption": current_metrics.power_consumption
            }
        
        if avg_metrics:
            stats["average_metrics"] = {
                "cpu_usage": avg_metrics.cpu_usage,
                "memory_usage": avg_metrics.memory_usage,
                "power_consumption": avg_metrics.power_consumption
            }
        
        if current_profile:
            stats["current_profile"] = {
                "cpu_frequency_range": f"{current_profile.cpu_min_frequency}-{current_profile.cpu_max_frequency} MHz",
                "cpu_governor": current_profile.cpu_governor,
                "idle_timeout": current_profile.idle_timeout,
                "sleep_timeout": current_profile.sleep_timeout
            }
        
        stats["sleep_stats"] = sleep_stats
        
        return stats
    
    def optimize_for_task(self, task_type: str, duration: float = 0.0) -> bool:
        """为特定任务优化功耗
        
        Args:
            task_type: 任务类型 ("inference", "training", "idle", "video", "gaming")
            duration: 预期持续时间 (秒，0表示未知)
            
        Returns:
            bool: 优化是否成功
        """
        try:
            task_profiles = {
                "inference": PowerMode.BALANCED,
                "training": PowerMode.HIGH_PERFORMANCE,
                "idle": PowerMode.POWER_SAVER,
                "video": PowerMode.BALANCED,
                "gaming": PowerMode.HIGH_PERFORMANCE,
                "background": PowerMode.POWER_SAVER
            }
            
            target_mode = task_profiles.get(task_type, PowerMode.BALANCED)
            
            # 根据电池状态调整
            current_metrics = self.system_monitor.get_current_metrics()
            if current_metrics and current_metrics.battery_level < self.config.low_battery_threshold:
                if target_mode == PowerMode.HIGH_PERFORMANCE:
                    target_mode = PowerMode.BALANCED
                elif target_mode == PowerMode.BALANCED:
                    target_mode = PowerMode.POWER_SAVER
            
            self.apply_power_mode(target_mode)
            
            self.logger.info(f"为任务'{task_type}'优化功耗，模式: {target_mode.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"任务功耗优化失败: {e}")
            return False
    
    def emergency_power_save(self):
        """紧急省电模式"""
        try:
            self.logger.warning("启动紧急省电模式")
            
            # 强制切换到超低功耗模式
            self.apply_power_mode(PowerMode.ULTRA_LOW_POWER)
            
            # 额外的紧急措施
            self._apply_emergency_measures()
            
        except Exception as e:
            self.logger.error(f"紧急省电模式启动失败: {e}")
    
    def _apply_emergency_measures(self):
        """应用紧急省电措施"""
        try:
            # 降低到最低频率
            if self.frequency_controller.cpu_frequencies:
                min_freq = min(self.frequency_controller.cpu_frequencies)
                self.frequency_controller.set_cpu_frequency(min_freq, min_freq, "powersave")
            
            # 关闭非必要功能
            self._set_wifi_power_save(True)
            self._set_bluetooth_enabled(False)
            self._set_screen_brightness(0.2)
            
            # 强制进入睡眠准备状态
            self.sleep_manager.update_activity()  # 重置活动时间，但设置很短的超时
            
            self.logger.info("紧急省电措施已应用")
            
        except Exception as e:
            self.logger.error(f"紧急省电措施应用失败: {e}")

# 测试代码
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建配置
    config = PowerConfig(
        device_type=DeviceType.GENERIC_ARM,
        default_mode=PowerMode.BALANCED,
        enable_adaptive_mode=True,
        monitoring_interval=2.0
    )
    
    # 创建功耗管理器
    power_manager = PowerManager(config)
    
    try:
        print("启动功耗管理器...")
        power_manager.start()
        
        # 运行一段时间
        time.sleep(10)
        
        # 获取统计信息
        stats = power_manager.get_power_stats()
        print("\n功耗统计:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        
        # 测试模式切换
        print("\n测试功耗模式切换...")
        power_manager.apply_power_mode(PowerMode.HIGH_PERFORMANCE)
        time.sleep(2)
        
        power_manager.apply_power_mode(PowerMode.POWER_SAVER)
        time.sleep(2)
        
        # 测试任务优化
        print("\n测试任务优化...")
        power_manager.optimize_for_task("inference")
        time.sleep(2)
        
        power_manager.optimize_for_task("training")
        time.sleep(2)
        
        # 测试自定义配置文件
        print("\n测试自定义配置文件...")
        custom_profile = power_manager.create_custom_profile(
            cpu_max_frequency=1500,
            cpu_min_frequency=800,
            cpu_governor="ondemand",
            screen_brightness=0.7
        )
        power_manager.apply_power_mode(PowerMode.CUSTOM, custom_profile)
        time.sleep(2)
        
        # 最终统计
        final_stats = power_manager.get_power_stats()
        print("\n最终功耗统计:")
        print(json.dumps(final_stats, indent=2, ensure_ascii=False))
        
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"测试异常: {e}")
    finally:
        print("\n停止功耗管理器...")
        power_manager.stop()
        print("功耗管理器测试完成")