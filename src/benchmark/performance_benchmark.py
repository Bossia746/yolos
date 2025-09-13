#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多平台性能基准测试系统
验证YOLOV特征聚合、时序聚合和跟踪功能的优化效果
支持不同平台和设备的性能评估
"""

import cv2
import numpy as np
import time
import psutil
import platform
import logging
import json
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns

from ..detection.factory import DetectorFactory
from ..detection.feature_aggregation import AggregationConfig, PlatformType
from ..detection.temporal_aggregator import TemporalConfig
from ..tracking.tracking_integration import IntegratedTrackingConfig, TrackingMode
from ..models.yolo_factory import YOLOFactory

logger = logging.getLogger(__name__)

class BenchmarkType(Enum):
    """基准测试类型"""
    DETECTION_SPEED = "detection_speed"  # 检测速度
    DETECTION_ACCURACY = "detection_accuracy"  # 检测精度
    MEMORY_USAGE = "memory_usage"  # 内存使用
    CPU_USAGE = "cpu_usage"  # CPU使用率
    TRACKING_PERFORMANCE = "tracking_performance"  # 跟踪性能
    AGGREGATION_EFFECTIVENESS = "aggregation_effectiveness"  # 聚合效果
    PLATFORM_COMPATIBILITY = "platform_compatibility"  # 平台兼容性
    REAL_TIME_PERFORMANCE = "real_time_performance"  # 实时性能

class TestScenario(Enum):
    """测试场景"""
    SINGLE_OBJECT = "single_object"  # 单目标
    MULTI_OBJECT = "multi_object"  # 多目标
    CROWDED_SCENE = "crowded_scene"  # 拥挤场景
    FAST_MOTION = "fast_motion"  # 快速运动
    OCCLUSION = "occlusion"  # 遮挡
    LIGHTING_CHANGE = "lighting_change"  # 光照变化
    SCALE_VARIATION = "scale_variation"  # 尺度变化

@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    test_types: List[BenchmarkType] = field(default_factory=lambda: list(BenchmarkType))
    test_scenarios: List[TestScenario] = field(default_factory=lambda: list(TestScenario))
    test_duration: int = 60  # 测试持续时间（秒）
    warmup_duration: int = 10  # 预热时间（秒）
    sample_interval: float = 0.1  # 采样间隔（秒）
    
    # 测试数据
    test_video_path: Optional[str] = None
    test_image_dir: Optional[str] = None
    synthetic_data: bool = True  # 使用合成数据
    
    # 平台配置
    target_platforms: List[PlatformType] = field(default_factory=lambda: [PlatformType.DESKTOP])
    
    # 输出配置
    output_dir: str = "benchmark_results"
    save_plots: bool = True
    save_detailed_logs: bool = True
    generate_report: bool = True

@dataclass
class BenchmarkResult:
    """基准测试结果"""
    test_type: BenchmarkType
    scenario: TestScenario
    platform: PlatformType
    
    # 性能指标
    avg_fps: float = 0.0
    min_fps: float = 0.0
    max_fps: float = 0.0
    std_fps: float = 0.0
    
    avg_latency: float = 0.0  # 毫秒
    avg_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    peak_cpu_percent: float = 0.0
    
    # 检测指标
    detection_count: int = 0
    tracking_accuracy: float = 0.0
    aggregation_improvement: float = 0.0
    
    # 详细数据
    fps_history: List[float] = field(default_factory=list)
    latency_history: List[float] = field(default_factory=list)
    memory_history: List[float] = field(default_factory=list)
    cpu_history: List[float] = field(default_factory=list)
    
    # 元数据
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0
    error_count: int = 0
    notes: str = ""

class SystemMonitor:
    """系统监控器"""
    
    def __init__(self, sample_interval: float = 0.1):
        self.sample_interval = sample_interval
        self.monitoring = False
        self.monitor_thread = None
        
        # 监控数据
        self.cpu_history = deque(maxlen=1000)
        self.memory_history = deque(maxlen=1000)
        self.timestamp_history = deque(maxlen=1000)
        
        # 进程监控
        self.process = psutil.Process()
    
    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("系统监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("系统监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                # CPU使用率
                cpu_percent = self.process.cpu_percent()
                
                # 内存使用
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                # 记录数据
                current_time = time.time()
                self.cpu_history.append(cpu_percent)
                self.memory_history.append(memory_mb)
                self.timestamp_history.append(current_time)
                
                time.sleep(self.sample_interval)
                
            except Exception as e:
                logger.error(f"监控错误: {e}")
                time.sleep(self.sample_interval)
    
    def get_stats(self) -> Dict[str, float]:
        """获取统计信息"""
        if not self.cpu_history or not self.memory_history:
            return {
                'avg_cpu': 0.0, 'peak_cpu': 0.0,
                'avg_memory': 0.0, 'peak_memory': 0.0
            }
        
        return {
            'avg_cpu': np.mean(self.cpu_history),
            'peak_cpu': np.max(self.cpu_history),
            'avg_memory': np.mean(self.memory_history),
            'peak_memory': np.max(self.memory_history)
        }
    
    def reset(self):
        """重置监控数据"""
        self.cpu_history.clear()
        self.memory_history.clear()
        self.timestamp_history.clear()

class SyntheticDataGenerator:
    """合成数据生成器"""
    
    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.frame_count = 0
    
    def generate_frame(self, scenario: TestScenario) -> np.ndarray:
        """生成测试帧"""
        frame = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
        
        if scenario == TestScenario.SINGLE_OBJECT:
            self._add_single_object(frame)
        elif scenario == TestScenario.MULTI_OBJECT:
            self._add_multiple_objects(frame)
        elif scenario == TestScenario.CROWDED_SCENE:
            self._add_crowded_objects(frame)
        elif scenario == TestScenario.FAST_MOTION:
            self._add_fast_moving_object(frame)
        elif scenario == TestScenario.OCCLUSION:
            self._add_occluded_objects(frame)
        elif scenario == TestScenario.LIGHTING_CHANGE:
            self._add_lighting_variation(frame)
        elif scenario == TestScenario.SCALE_VARIATION:
            self._add_scale_variation(frame)
        
        self.frame_count += 1
        return frame
    
    def _add_single_object(self, frame: np.ndarray):
        """添加单个目标"""
        x = int(self.width * 0.4 + 50 * np.sin(self.frame_count * 0.1))
        y = int(self.height * 0.5)
        cv2.rectangle(frame, (x, y), (x + 80, y + 120), (0, 255, 0), -1)
    
    def _add_multiple_objects(self, frame: np.ndarray):
        """添加多个目标"""
        for i in range(3):
            x = int(self.width * (0.2 + i * 0.3) + 30 * np.sin(self.frame_count * 0.1 + i))
            y = int(self.height * 0.5 + 20 * np.cos(self.frame_count * 0.1 + i))
            cv2.rectangle(frame, (x, y), (x + 60, y + 100), (0, 255, 0), -1)
    
    def _add_crowded_objects(self, frame: np.ndarray):
        """添加拥挤场景"""
        for i in range(8):
            x = int((i % 4) * self.width / 4 + 20 * np.sin(self.frame_count * 0.05 + i))
            y = int((i // 4) * self.height / 2 + self.height * 0.25)
            cv2.rectangle(frame, (x, y), (x + 40, y + 80), (0, 255, 0), -1)
    
    def _add_fast_moving_object(self, frame: np.ndarray):
        """添加快速移动目标"""
        x = int((self.frame_count * 10) % self.width)
        y = int(self.height * 0.5)
        cv2.rectangle(frame, (x, y), (x + 60, y + 100), (0, 255, 0), -1)
    
    def _add_occluded_objects(self, frame: np.ndarray):
        """添加遮挡目标"""
        # 背景目标
        cv2.rectangle(frame, (200, 150), (280, 300), (0, 255, 0), -1)
        # 前景遮挡物
        x = int(150 + 100 * np.sin(self.frame_count * 0.05))
        cv2.rectangle(frame, (x, 100), (x + 80, 350), (255, 0, 0), -1)
    
    def _add_lighting_variation(self, frame: np.ndarray):
        """添加光照变化"""
        brightness = int(128 + 100 * np.sin(self.frame_count * 0.02))
        frame[:] = np.clip(frame.astype(np.int16) + brightness - 128, 0, 255).astype(np.uint8)
        cv2.rectangle(frame, (250, 150), (330, 300), (0, 255, 0), -1)
    
    def _add_scale_variation(self, frame: np.ndarray):
        """添加尺度变化"""
        scale = 0.5 + 0.4 * np.sin(self.frame_count * 0.03)
        size = int(80 * scale)
        x = int(self.width * 0.5 - size / 2)
        y = int(self.height * 0.5 - size / 2)
        cv2.rectangle(frame, (x, y), (x + size, y + int(size * 1.5)), (0, 255, 0), -1)

class PerformanceBenchmark:
    """性能基准测试器"""
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.results: List[BenchmarkResult] = []
        self.system_monitor = SystemMonitor(self.config.sample_interval)
        self.data_generator = SyntheticDataGenerator()
        
        # 创建输出目录
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"性能基准测试器初始化完成，输出目录: {self.output_dir}")
    
    def run_benchmark(self) -> List[BenchmarkResult]:
        """运行基准测试"""
        logger.info("开始性能基准测试")
        
        # 获取系统信息
        self._log_system_info()
        
        # 运行所有测试
        for platform in self.config.target_platforms:
            for test_type in self.config.test_types:
                for scenario in self.config.test_scenarios:
                    try:
                        result = self._run_single_test(test_type, scenario, platform)
                        self.results.append(result)
                        logger.info(f"完成测试: {test_type.value} - {scenario.value} - {platform.value}")
                    except Exception as e:
                        logger.error(f"测试失败: {test_type.value} - {scenario.value} - {e}")
        
        # 生成报告
        if self.config.generate_report:
            self._generate_report()
        
        logger.info(f"基准测试完成，共完成 {len(self.results)} 项测试")
        return self.results
    
    def _run_single_test(self, test_type: BenchmarkType, scenario: TestScenario, 
                        platform: PlatformType) -> BenchmarkResult:
        """运行单个测试"""
        result = BenchmarkResult(
            test_type=test_type,
            scenario=scenario,
            platform=platform
        )
        
        # 创建检测器配置
        detector_config = self._create_detector_config(platform)
        
        # 创建检测器
        if test_type == BenchmarkType.TRACKING_PERFORMANCE:
            detector = DetectorFactory.create_detector_with_tracking(
                'realtime', 'enhanced', **detector_config
            )
        else:
            detector = DetectorFactory.create_detector('realtime', **detector_config)
        
        # 启动系统监控
        self.system_monitor.reset()
        self.system_monitor.start_monitoring()
        
        try:
            # 预热
            self._warmup_detector(detector, scenario)
            
            # 执行测试
            start_time = time.time()
            self._execute_test(detector, scenario, result)
            result.duration = time.time() - start_time
            
            # 获取系统统计
            system_stats = self.system_monitor.get_stats()
            result.avg_cpu_percent = system_stats['avg_cpu']
            result.peak_cpu_percent = system_stats['peak_cpu']
            result.avg_memory_mb = system_stats['avg_memory']
            result.peak_memory_mb = system_stats['peak_memory']
            
            # 计算性能指标
            self._calculate_metrics(result)
            
        finally:
            self.system_monitor.stop_monitoring()
        
        return result
    
    def _create_detector_config(self, platform: PlatformType) -> Dict[str, Any]:
        """创建检测器配置"""
        # 根据平台优化配置
        if platform == PlatformType.RASPBERRY_PI:
            return {
                'device': 'cpu',
                'enable_aggregation': True,
                'platform_type': platform,
                'aggregation_config': AggregationConfig(
                    platform_type=platform,
                    max_buffer_size=5,
                    confidence_threshold=0.3
                )
            }
        elif platform == PlatformType.ESP32:
            return {
                'device': 'cpu',
                'enable_aggregation': True,
                'platform_type': platform,
                'aggregation_config': AggregationConfig(
                    platform_type=platform,
                    max_buffer_size=3,
                    confidence_threshold=0.4
                )
            }
        else:  # DESKTOP
            return {
                'device': 'auto',
                'enable_aggregation': True,
                'platform_type': platform,
                'aggregation_config': AggregationConfig(
                    platform_type=platform,
                    max_buffer_size=10,
                    confidence_threshold=0.25
                )
            }
    
    def _warmup_detector(self, detector, scenario: TestScenario):
        """预热检测器"""
        logger.info(f"预热检测器 ({self.config.warmup_duration}秒)")
        
        start_time = time.time()
        while time.time() - start_time < self.config.warmup_duration:
            frame = self.data_generator.generate_frame(scenario)
            try:
                detector.detect(frame)
            except Exception as e:
                logger.warning(f"预热检测失败: {e}")
            time.sleep(0.033)  # ~30 FPS
    
    def _execute_test(self, detector, scenario: TestScenario, result: BenchmarkResult):
        """执行测试"""
        logger.info(f"执行测试 ({self.config.test_duration}秒)")
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < self.config.test_duration:
            frame_start = time.time()
            
            # 生成测试帧
            frame = self.data_generator.generate_frame(scenario)
            
            try:
                # 执行检测
                detections = detector.detect(frame)
                
                # 记录检测结果
                if detections:
                    result.detection_count += len(detections)
                
                # 计算FPS和延迟
                frame_time = time.time() - frame_start
                fps = 1.0 / frame_time if frame_time > 0 else 0
                latency = frame_time * 1000  # 毫秒
                
                result.fps_history.append(fps)
                result.latency_history.append(latency)
                
                frame_count += 1
                
            except Exception as e:
                result.error_count += 1
                logger.warning(f"检测错误: {e}")
            
            # 控制帧率
            time.sleep(max(0, 0.033 - (time.time() - frame_start)))
        
        logger.info(f"测试完成，处理了 {frame_count} 帧，检测到 {result.detection_count} 个目标")
    
    def _calculate_metrics(self, result: BenchmarkResult):
        """计算性能指标"""
        if result.fps_history:
            result.avg_fps = np.mean(result.fps_history)
            result.min_fps = np.min(result.fps_history)
            result.max_fps = np.max(result.fps_history)
            result.std_fps = np.std(result.fps_history)
        
        if result.latency_history:
            result.avg_latency = np.mean(result.latency_history)
    
    def _log_system_info(self):
        """记录系统信息"""
        system_info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'architecture': platform.architecture(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / 1024**3,
            'opencv_version': cv2.__version__
        }
        
        logger.info(f"系统信息: {json.dumps(system_info, indent=2)}")
        
        # 保存系统信息
        with open(self.output_dir / 'system_info.json', 'w') as f:
            json.dump(system_info, f, indent=2)
    
    def _generate_report(self):
        """生成测试报告"""
        logger.info("生成测试报告")
        
        # 保存详细结果
        self._save_detailed_results()
        
        # 生成图表
        if self.config.save_plots:
            self._generate_plots()
        
        # 生成汇总报告
        self._generate_summary_report()
    
    def _save_detailed_results(self):
        """保存详细结果"""
        results_data = []
        
        for result in self.results:
            result_dict = {
                'test_type': result.test_type.value,
                'scenario': result.scenario.value,
                'platform': result.platform.value,
                'avg_fps': result.avg_fps,
                'min_fps': result.min_fps,
                'max_fps': result.max_fps,
                'std_fps': result.std_fps,
                'avg_latency': result.avg_latency,
                'avg_memory_mb': result.avg_memory_mb,
                'peak_memory_mb': result.peak_memory_mb,
                'avg_cpu_percent': result.avg_cpu_percent,
                'peak_cpu_percent': result.peak_cpu_percent,
                'detection_count': result.detection_count,
                'duration': result.duration,
                'error_count': result.error_count,
                'timestamp': result.timestamp
            }
            results_data.append(result_dict)
        
        with open(self.output_dir / 'detailed_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
    
    def _generate_plots(self):
        """生成性能图表"""
        try:
            # FPS对比图
            self._plot_fps_comparison()
            
            # 内存使用图
            self._plot_memory_usage()
            
            # CPU使用图
            self._plot_cpu_usage()
            
            # 延迟分布图
            self._plot_latency_distribution()
            
        except Exception as e:
            logger.error(f"生成图表失败: {e}")
    
    def _plot_fps_comparison(self):
        """绘制FPS对比图"""
        plt.figure(figsize=(12, 8))
        
        # 按平台分组
        platform_data = defaultdict(list)
        for result in self.results:
            platform_data[result.platform.value].append(result.avg_fps)
        
        platforms = list(platform_data.keys())
        fps_values = [platform_data[p] for p in platforms]
        
        plt.boxplot(fps_values, labels=platforms)
        plt.title('FPS Performance Comparison Across Platforms')
        plt.ylabel('FPS')
        plt.xlabel('Platform')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fps_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_memory_usage(self):
        """绘制内存使用图"""
        plt.figure(figsize=(12, 6))
        
        scenarios = [r.scenario.value for r in self.results]
        memory_avg = [r.avg_memory_mb for r in self.results]
        memory_peak = [r.peak_memory_mb for r in self.results]
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        plt.bar(x - width/2, memory_avg, width, label='Average Memory', alpha=0.8)
        plt.bar(x + width/2, memory_peak, width, label='Peak Memory', alpha=0.8)
        
        plt.title('Memory Usage by Test Scenario')
        plt.ylabel('Memory (MB)')
        plt.xlabel('Test Scenario')
        plt.xticks(x, scenarios, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'memory_usage.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cpu_usage(self):
        """绘制CPU使用图"""
        plt.figure(figsize=(10, 6))
        
        test_types = [r.test_type.value for r in self.results]
        cpu_usage = [r.avg_cpu_percent for r in self.results]
        
        plt.bar(test_types, cpu_usage, alpha=0.7)
        plt.title('CPU Usage by Test Type')
        plt.ylabel('CPU Usage (%)')
        plt.xlabel('Test Type')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cpu_usage.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_latency_distribution(self):
        """绘制延迟分布图"""
        plt.figure(figsize=(10, 6))
        
        latencies = [r.avg_latency for r in self.results if r.avg_latency > 0]
        
        if latencies:
            plt.hist(latencies, bins=20, alpha=0.7, edgecolor='black')
            plt.title('Detection Latency Distribution')
            plt.xlabel('Latency (ms)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_latency = np.mean(latencies)
            plt.axvline(mean_latency, color='red', linestyle='--', 
                       label=f'Mean: {mean_latency:.1f}ms')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'latency_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_summary_report(self):
        """生成汇总报告"""
        report_lines = [
            "# YOLOS 性能基准测试报告",
            f"\n生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"测试总数: {len(self.results)}",
            "\n## 测试概览",
        ]
        
        # 按平台汇总
        platform_summary = defaultdict(list)
        for result in self.results:
            platform_summary[result.platform.value].append(result)
        
        for platform, results in platform_summary.items():
            avg_fps = np.mean([r.avg_fps for r in results if r.avg_fps > 0])
            avg_memory = np.mean([r.avg_memory_mb for r in results if r.avg_memory_mb > 0])
            avg_cpu = np.mean([r.avg_cpu_percent for r in results if r.avg_cpu_percent > 0])
            
            report_lines.extend([
                f"\n### {platform}",
                f"- 平均FPS: {avg_fps:.1f}",
                f"- 平均内存使用: {avg_memory:.1f} MB",
                f"- 平均CPU使用: {avg_cpu:.1f}%",
                f"- 测试数量: {len(results)}"
            ])
        
        # 性能建议
        report_lines.extend([
            "\n## 性能建议",
            self._generate_performance_recommendations()
        ])
        
        # 保存报告
        with open(self.output_dir / 'benchmark_report.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"测试报告已保存到: {self.output_dir / 'benchmark_report.md'}")
    
    def _generate_performance_recommendations(self) -> str:
        """生成性能建议"""
        recommendations = []
        
        # 分析结果
        avg_fps_by_platform = {}
        for result in self.results:
            platform = result.platform.value
            if platform not in avg_fps_by_platform:
                avg_fps_by_platform[platform] = []
            if result.avg_fps > 0:
                avg_fps_by_platform[platform].append(result.avg_fps)
        
        for platform, fps_list in avg_fps_by_platform.items():
            avg_fps = np.mean(fps_list)
            
            if avg_fps < 10:
                recommendations.append(
                    f"- {platform}: 性能较低 (平均{avg_fps:.1f} FPS)，建议降低输入分辨率或使用更轻量的模型"
                )
            elif avg_fps < 20:
                recommendations.append(
                    f"- {platform}: 性能中等 (平均{avg_fps:.1f} FPS)，可考虑优化聚合策略"
                )
            else:
                recommendations.append(
                    f"- {platform}: 性能良好 (平均{avg_fps:.1f} FPS)，可启用更多高级功能"
                )
        
        return '\n'.join(recommendations) if recommendations else "所有平台性能表现良好"
    
    def get_results(self) -> List[BenchmarkResult]:
        """获取测试结果"""
        return self.results
    
    def clear_results(self):
        """清空测试结果"""
        self.results.clear()
        logger.info("测试结果已清空")

# 使用示例
if __name__ == "__main__":
    # 创建基准测试配置
    config = BenchmarkConfig(
        test_types=[
            BenchmarkType.DETECTION_SPEED,
            BenchmarkType.MEMORY_USAGE,
            BenchmarkType.TRACKING_PERFORMANCE
        ],
        test_scenarios=[
            TestScenario.SINGLE_OBJECT,
            TestScenario.MULTI_OBJECT,
            TestScenario.FAST_MOTION
        ],
        target_platforms=[
            PlatformType.DESKTOP,
            PlatformType.RASPBERRY_PI
        ],
        test_duration=30,
        warmup_duration=5
    )
    
    # 创建基准测试器
    benchmark = PerformanceBenchmark(config)
    
    # 运行测试
    results = benchmark.run_benchmark()
    
    # 打印结果摘要
    print(f"\n基准测试完成，共完成 {len(results)} 项测试")
    for result in results:
        print(f"{result.test_type.value} - {result.scenario.value} - {result.platform.value}: "
              f"{result.avg_fps:.1f} FPS, {result.avg_memory_mb:.1f} MB")