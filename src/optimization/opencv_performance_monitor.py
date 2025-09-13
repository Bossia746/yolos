#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV性能监控器
实时监控OpenCV在不同场景下的性能表现，提供优化建议
"""

import cv2
import numpy as np
import time
import psutil
import threading
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import deque
from enum import Enum
import matplotlib.pyplot as plt
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SceneType(Enum):
    """场景类型"""
    STATIC = "static"  # 静态场景
    DYNAMIC = "dynamic"  # 动态场景
    MIXED = "mixed"  # 混合场景
    LOW_LIGHT = "low_light"  # 低光照
    HIGH_MOTION = "high_motion"  # 高运动
    CROWDED = "crowded"  # 拥挤场景

class ProcessingStage(Enum):
    """处理阶段"""
    CAPTURE = "capture"  # 图像捕获
    PREPROCESSING = "preprocessing"  # 预处理
    DETECTION = "detection"  # 检测
    TRACKING = "tracking"  # 跟踪
    POSTPROCESSING = "postprocessing"  # 后处理
    DISPLAY = "display"  # 显示

@dataclass
class PerformanceMetrics:
    """性能指标"""
    fps: float
    processing_time: float
    memory_usage: float
    cpu_usage: float
    gpu_usage: float
    frame_drops: int
    latency: float
    throughput: float

@dataclass
class SceneMetrics:
    """场景指标"""
    scene_type: SceneType
    complexity_score: float
    motion_level: float
    lighting_quality: float
    object_count: int
    occlusion_level: float

@dataclass
class OptimizationSuggestion:
    """优化建议"""
    category: str
    priority: str  # high, medium, low
    description: str
    implementation: str
    expected_improvement: float

class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.stage_times = {}
        self.start_times = {}
        
    def start_stage(self, stage: ProcessingStage):
        """开始计时"""
        self.start_times[stage] = time.perf_counter()
    
    def end_stage(self, stage: ProcessingStage) -> float:
        """结束计时并返回耗时"""
        if stage in self.start_times:
            elapsed = time.perf_counter() - self.start_times[stage]
            if stage not in self.stage_times:
                self.stage_times[stage] = deque(maxlen=100)
            self.stage_times[stage].append(elapsed)
            return elapsed
        return 0.0
    
    def get_average_time(self, stage: ProcessingStage) -> float:
        """获取平均耗时"""
        if stage in self.stage_times and self.stage_times[stage]:
            return sum(self.stage_times[stage]) / len(self.stage_times[stage])
        return 0.0
    
    def get_stage_distribution(self) -> Dict[str, float]:
        """获取各阶段时间分布"""
        total_time = sum(self.get_average_time(stage) for stage in ProcessingStage)
        if total_time == 0:
            return {}
        
        return {
            stage.value: (self.get_average_time(stage) / total_time) * 100
            for stage in ProcessingStage
        }

class SystemMonitor:
    """系统监控器"""
    
    def __init__(self):
        self.cpu_history = deque(maxlen=100)
        self.memory_history = deque(maxlen=100)
        self.gpu_history = deque(maxlen=100)
        
    def update_metrics(self) -> Dict[str, float]:
        """更新系统指标"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.cpu_history.append(cpu_percent)
        
        # 内存使用率
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        self.memory_history.append(memory_percent)
        
        # GPU使用率（如果可用）
        gpu_percent = self._get_gpu_usage()
        self.gpu_history.append(gpu_percent)
        
        return {
            'cpu': cpu_percent,
            'memory': memory_percent,
            'gpu': gpu_percent
        }
    
    def _get_gpu_usage(self) -> float:
        """获取GPU使用率"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100
        except ImportError:
            pass
        return 0.0
    
    def get_average_usage(self) -> Dict[str, float]:
        """获取平均使用率"""
        return {
            'cpu': sum(self.cpu_history) / len(self.cpu_history) if self.cpu_history else 0,
            'memory': sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0,
            'gpu': sum(self.gpu_history) / len(self.gpu_history) if self.gpu_history else 0
        }

class SceneAnalyzer:
    """场景分析器"""
    
    def __init__(self):
        self.motion_detector = cv2.createBackgroundSubtractorMOG2()
        self.previous_frame = None
        
    def analyze_scene(self, frame: np.ndarray) -> SceneMetrics:
        """分析场景特征"""
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 计算复杂度分数
        complexity_score = self._calculate_complexity(gray)
        
        # 计算运动水平
        motion_level = self._calculate_motion(gray)
        
        # 计算光照质量
        lighting_quality = self._calculate_lighting_quality(gray)
        
        # 估算对象数量
        object_count = self._estimate_object_count(gray)
        
        # 计算遮挡水平
        occlusion_level = self._calculate_occlusion(gray)
        
        # 确定场景类型
        scene_type = self._determine_scene_type(
            motion_level, lighting_quality, object_count
        )
        
        return SceneMetrics(
            scene_type=scene_type,
            complexity_score=complexity_score,
            motion_level=motion_level,
            lighting_quality=lighting_quality,
            object_count=object_count,
            occlusion_level=occlusion_level
        )
    
    def _calculate_complexity(self, gray: np.ndarray) -> float:
        """计算图像复杂度"""
        # 使用Laplacian方差作为复杂度指标
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return min(laplacian_var / 1000.0, 1.0)  # 归一化到0-1
    
    def _calculate_motion(self, gray: np.ndarray) -> float:
        """计算运动水平"""
        if self.previous_frame is None:
            self.previous_frame = gray.copy()
            return 0.0
        
        # 计算帧差
        diff = cv2.absdiff(self.previous_frame, gray)
        motion_pixels = np.sum(diff > 30)
        total_pixels = gray.shape[0] * gray.shape[1]
        
        self.previous_frame = gray.copy()
        
        return motion_pixels / total_pixels
    
    def _calculate_lighting_quality(self, gray: np.ndarray) -> float:
        """计算光照质量"""
        # 计算直方图
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # 计算对比度（标准差）
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        
        # 归一化光照质量分数
        # 好的光照应该有适中的亮度和良好的对比度
        brightness_score = 1.0 - abs(mean_val - 128) / 128
        contrast_score = min(std_val / 64.0, 1.0)
        
        return (brightness_score + contrast_score) / 2.0
    
    def _estimate_object_count(self, gray: np.ndarray) -> int:
        """估算对象数量"""
        # 使用边缘检测和轮廓查找
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤小轮廓
        significant_contours = [c for c in contours if cv2.contourArea(c) > 100]
        
        return len(significant_contours)
    
    def _calculate_occlusion(self, gray: np.ndarray) -> float:
        """计算遮挡水平"""
        # 使用形态学操作检测遮挡
        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # 计算闭运算前后的差异
        diff = cv2.absdiff(gray, closing)
        occlusion_pixels = np.sum(diff > 10)
        total_pixels = gray.shape[0] * gray.shape[1]
        
        return occlusion_pixels / total_pixels
    
    def _determine_scene_type(self, motion_level: float, 
                            lighting_quality: float, 
                            object_count: int) -> SceneType:
        """确定场景类型"""
        if lighting_quality < 0.3:
            return SceneType.LOW_LIGHT
        elif motion_level > 0.3:
            return SceneType.HIGH_MOTION
        elif object_count > 10:
            return SceneType.CROWDED
        elif motion_level > 0.1:
            return SceneType.DYNAMIC
        elif motion_level < 0.05:
            return SceneType.STATIC
        else:
            return SceneType.MIXED

class OptimizationEngine:
    """优化引擎"""
    
    def __init__(self):
        self.optimization_rules = self._load_optimization_rules()
    
    def _load_optimization_rules(self) -> Dict[str, Any]:
        """加载优化规则"""
        return {
            'fps_thresholds': {
                'excellent': 60,
                'good': 30,
                'acceptable': 15,
                'poor': 10
            },
            'memory_thresholds': {
                'low': 50,
                'medium': 70,
                'high': 85
            },
            'cpu_thresholds': {
                'low': 30,
                'medium': 60,
                'high': 80
            }
        }
    
    def analyze_performance(self, 
                          performance: PerformanceMetrics,
                          scene: SceneMetrics,
                          stage_distribution: Dict[str, float]) -> List[OptimizationSuggestion]:
        """分析性能并生成优化建议"""
        suggestions = []
        
        # FPS优化建议
        suggestions.extend(self._analyze_fps(performance, scene))
        
        # 内存优化建议
        suggestions.extend(self._analyze_memory(performance))
        
        # CPU优化建议
        suggestions.extend(self._analyze_cpu(performance, stage_distribution))
        
        # 场景特定优化建议
        suggestions.extend(self._analyze_scene_specific(scene))
        
        # 按优先级排序
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        suggestions.sort(key=lambda x: priority_order.get(x.priority, 3))
        
        return suggestions
    
    def _analyze_fps(self, performance: PerformanceMetrics, 
                    scene: SceneMetrics) -> List[OptimizationSuggestion]:
        """分析FPS性能"""
        suggestions = []
        thresholds = self.optimization_rules['fps_thresholds']
        
        if performance.fps < thresholds['poor']:
            suggestions.append(OptimizationSuggestion(
                category="FPS优化",
                priority="high",
                description=f"当前FPS过低({performance.fps:.1f})，严重影响实时性",
                implementation="降低输入分辨率、减少处理复杂度、启用GPU加速",
                expected_improvement=50.0
            ))
        elif performance.fps < thresholds['acceptable']:
            suggestions.append(OptimizationSuggestion(
                category="FPS优化",
                priority="medium",
                description=f"FPS需要改善({performance.fps:.1f})，可能影响用户体验",
                implementation="优化算法参数、使用多线程处理",
                expected_improvement=30.0
            ))
        
        return suggestions
    
    def _analyze_memory(self, performance: PerformanceMetrics) -> List[OptimizationSuggestion]:
        """分析内存使用"""
        suggestions = []
        thresholds = self.optimization_rules['memory_thresholds']
        
        if performance.memory_usage > thresholds['high']:
            suggestions.append(OptimizationSuggestion(
                category="内存优化",
                priority="high",
                description=f"内存使用过高({performance.memory_usage:.1f}%)，可能导致系统不稳定",
                implementation="释放未使用的缓冲区、减少图像缓存、优化数据结构",
                expected_improvement=25.0
            ))
        elif performance.memory_usage > thresholds['medium']:
            suggestions.append(OptimizationSuggestion(
                category="内存优化",
                priority="medium",
                description=f"内存使用较高({performance.memory_usage:.1f}%)，建议优化",
                implementation="启用内存池、减少临时对象创建",
                expected_improvement=15.0
            ))
        
        return suggestions
    
    def _analyze_cpu(self, performance: PerformanceMetrics,
                    stage_distribution: Dict[str, float]) -> List[OptimizationSuggestion]:
        """分析CPU使用"""
        suggestions = []
        thresholds = self.optimization_rules['cpu_thresholds']
        
        if performance.cpu_usage > thresholds['high']:
            # 找出最耗时的处理阶段
            max_stage = max(stage_distribution.items(), key=lambda x: x[1])
            
            suggestions.append(OptimizationSuggestion(
                category="CPU优化",
                priority="high",
                description=f"CPU使用过高({performance.cpu_usage:.1f}%)，{max_stage[0]}阶段占{max_stage[1]:.1f}%",
                implementation=f"优化{max_stage[0]}算法、启用SIMD指令、使用GPU加速",
                expected_improvement=35.0
            ))
        
        return suggestions
    
    def _analyze_scene_specific(self, scene: SceneMetrics) -> List[OptimizationSuggestion]:
        """场景特定优化建议"""
        suggestions = []
        
        if scene.scene_type == SceneType.LOW_LIGHT:
            suggestions.append(OptimizationSuggestion(
                category="场景优化",
                priority="medium",
                description="低光照场景检测，建议启用图像增强",
                implementation="启用自适应直方图均衡化、降噪处理",
                expected_improvement=20.0
            ))
        
        elif scene.scene_type == SceneType.HIGH_MOTION:
            suggestions.append(OptimizationSuggestion(
                category="场景优化",
                priority="medium",
                description="高运动场景检测，建议优化跟踪算法",
                implementation="使用卡尔曼滤波、增加预测机制",
                expected_improvement=25.0
            ))
        
        elif scene.scene_type == SceneType.CROWDED:
            suggestions.append(OptimizationSuggestion(
                category="场景优化",
                priority="high",
                description="拥挤场景检测，建议优化多目标处理",
                implementation="使用分层检测、ROI优化、并行处理",
                expected_improvement=40.0
            ))
        
        if scene.complexity_score > 0.8:
            suggestions.append(OptimizationSuggestion(
                category="复杂度优化",
                priority="medium",
                description=f"场景复杂度较高({scene.complexity_score:.2f})，建议简化处理",
                implementation="降低检测精度、使用快速算法、区域采样",
                expected_improvement=30.0
            ))
        
        return suggestions

class OpenCVPerformanceMonitor:
    """OpenCV性能监控器主类"""
    
    def __init__(self, config_path: str = None):
        self.profiler = PerformanceProfiler()
        self.system_monitor = SystemMonitor()
        self.scene_analyzer = SceneAnalyzer()
        self.optimization_engine = OptimizationEngine()
        
        # 性能历史记录
        self.fps_history = deque(maxlen=1000)
        self.latency_history = deque(maxlen=1000)
        self.frame_count = 0
        self.start_time = time.time()
        
        # 监控状态
        self.monitoring = False
        self.monitor_thread = None
        
        # 报告数据
        self.performance_reports = []
        
    def start_monitoring(self):
        """开始监控"""
        self.monitoring = True
        self.start_time = time.time()
        self.frame_count = 0
        
        # 启动系统监控线程
        self.monitor_thread = threading.Thread(target=self._system_monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("性能监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        logger.info("性能监控已停止")
    
    def _system_monitor_loop(self):
        """系统监控循环"""
        while self.monitoring:
            self.system_monitor.update_metrics()
            time.sleep(0.5)  # 每0.5秒更新一次
    
    def process_frame(self, frame: np.ndarray, 
                     processing_func: Callable[[np.ndarray], Any]) -> Tuple[Any, PerformanceMetrics]:
        """处理帧并监控性能"""
        frame_start_time = time.perf_counter()
        
        # 捕获阶段
        self.profiler.start_stage(ProcessingStage.CAPTURE)
        # 这里假设frame已经是捕获的结果
        self.profiler.end_stage(ProcessingStage.CAPTURE)
        
        # 预处理阶段
        self.profiler.start_stage(ProcessingStage.PREPROCESSING)
        # 场景分析
        scene_metrics = self.scene_analyzer.analyze_scene(frame)
        self.profiler.end_stage(ProcessingStage.PREPROCESSING)
        
        # 主处理阶段（检测/跟踪等）
        self.profiler.start_stage(ProcessingStage.DETECTION)
        result = processing_func(frame)
        self.profiler.end_stage(ProcessingStage.DETECTION)
        
        # 后处理阶段
        self.profiler.start_stage(ProcessingStage.POSTPROCESSING)
        # 这里可以添加后处理逻辑
        self.profiler.end_stage(ProcessingStage.POSTPROCESSING)
        
        # 计算性能指标
        frame_end_time = time.perf_counter()
        processing_time = frame_end_time - frame_start_time
        
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # 计算FPS
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        self.fps_history.append(fps)
        
        # 计算延迟
        latency = processing_time * 1000  # 转换为毫秒
        self.latency_history.append(latency)
        
        # 获取系统使用率
        system_usage = self.system_monitor.get_average_usage()
        
        # 创建性能指标
        performance_metrics = PerformanceMetrics(
            fps=fps,
            processing_time=processing_time,
            memory_usage=system_usage['memory'],
            cpu_usage=system_usage['cpu'],
            gpu_usage=system_usage['gpu'],
            frame_drops=0,  # 这里需要根据实际情况计算
            latency=latency,
            throughput=1.0 / processing_time if processing_time > 0 else 0
        )
        
        # 生成优化建议（每100帧分析一次）
        if self.frame_count % 100 == 0:
            stage_distribution = self.profiler.get_stage_distribution()
            suggestions = self.optimization_engine.analyze_performance(
                performance_metrics, scene_metrics, stage_distribution
            )
            
            if suggestions:
                self._log_optimization_suggestions(suggestions)
        
        return result, performance_metrics
    
    def _log_optimization_suggestions(self, suggestions: List[OptimizationSuggestion]):
        """记录优化建议"""
        logger.info("=== 性能优化建议 ===")
        for suggestion in suggestions[:3]:  # 只显示前3个最重要的建议
            logger.info(f"[{suggestion.priority.upper()}] {suggestion.category}: {suggestion.description}")
            logger.info(f"  实施方案: {suggestion.implementation}")
            logger.info(f"  预期改善: {suggestion.expected_improvement:.1f}%")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.fps_history or not self.latency_history:
            return {}
        
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        avg_latency = sum(self.latency_history) / len(self.latency_history)
        system_usage = self.system_monitor.get_average_usage()
        stage_distribution = self.profiler.get_stage_distribution()
        
        return {
            'performance': {
                'avg_fps': avg_fps,
                'min_fps': min(self.fps_history),
                'max_fps': max(self.fps_history),
                'avg_latency_ms': avg_latency,
                'min_latency_ms': min(self.latency_history),
                'max_latency_ms': max(self.latency_history),
                'total_frames': self.frame_count
            },
            'system_usage': system_usage,
            'stage_distribution': stage_distribution,
            'monitoring_duration': time.time() - self.start_time
        }
    
    def generate_performance_report(self, save_path: str = None) -> str:
        """生成性能报告"""
        summary = self.get_performance_summary()
        
        if not summary:
            return "没有性能数据可用于生成报告"
        
        report = f"""
# OpenCV性能监控报告

## 监控概览
- 监控时长: {summary['monitoring_duration']:.1f}秒
- 总处理帧数: {summary['performance']['total_frames']}
- 平均FPS: {summary['performance']['avg_fps']:.1f}
- 平均延迟: {summary['performance']['avg_latency_ms']:.1f}ms

## 性能指标
### 帧率统计
- 平均FPS: {summary['performance']['avg_fps']:.1f}
- 最低FPS: {summary['performance']['min_fps']:.1f}
- 最高FPS: {summary['performance']['max_fps']:.1f}

### 延迟统计
- 平均延迟: {summary['performance']['avg_latency_ms']:.1f}ms
- 最低延迟: {summary['performance']['min_latency_ms']:.1f}ms
- 最高延迟: {summary['performance']['max_latency_ms']:.1f}ms

## 系统资源使用
- 平均CPU使用率: {summary['system_usage']['cpu']:.1f}%
- 平均内存使用率: {summary['system_usage']['memory']:.1f}%
- 平均GPU使用率: {summary['system_usage']['gpu']:.1f}%

## 处理阶段分布
"""
        
        for stage, percentage in summary['stage_distribution'].items():
            report += f"- {stage}: {percentage:.1f}%\n"
        
        report += """

## 优化建议
基于当前性能数据，建议关注以下优化方向：
1. 如果FPS低于30，考虑降低输入分辨率或简化算法
2. 如果CPU使用率超过80%，考虑启用GPU加速
3. 如果内存使用率超过70%，考虑优化内存管理
4. 定期监控性能变化，及时调整优化策略
"""
        
        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                logger.info(f"性能报告已保存到: {save_path}")
            except Exception as e:
                logger.error(f"保存报告失败: {e}")
        
        return report
    
    def plot_performance_charts(self, save_dir: str = None):
        """绘制性能图表"""
        if not self.fps_history or not self.latency_history:
            logger.warning("没有足够的数据绘制图表")
            return
        
        try:
            # 创建子图
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('OpenCV性能监控图表', fontsize=16)
            
            # FPS历史图
            ax1.plot(list(self.fps_history))
            ax1.set_title('FPS历史')
            ax1.set_xlabel('帧数')
            ax1.set_ylabel('FPS')
            ax1.grid(True)
            
            # 延迟历史图
            ax2.plot(list(self.latency_history))
            ax2.set_title('延迟历史')
            ax2.set_xlabel('帧数')
            ax2.set_ylabel('延迟(ms)')
            ax2.grid(True)
            
            # 系统使用率
            usage = self.system_monitor.get_average_usage()
            ax3.bar(['CPU', 'Memory', 'GPU'], 
                   [usage['cpu'], usage['memory'], usage['gpu']])
            ax3.set_title('平均系统使用率')
            ax3.set_ylabel('使用率(%)')
            
            # 处理阶段分布
            stage_dist = self.profiler.get_stage_distribution()
            if stage_dist:
                stages = list(stage_dist.keys())
                percentages = list(stage_dist.values())
                ax4.pie(percentages, labels=stages, autopct='%1.1f%%')
                ax4.set_title('处理阶段时间分布')
            
            plt.tight_layout()
            
            if save_dir:
                save_path = Path(save_dir) / f"performance_charts_{int(time.time())}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"性能图表已保存到: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"绘制图表失败: {e}")

def test_performance_monitor():
    """测试性能监控器"""
    print("=== OpenCV性能监控器测试 ===")
    
    # 创建监控器
    monitor = OpenCVPerformanceMonitor()
    
    # 开始监控
    monitor.start_monitoring()
    
    try:
        # 模拟处理函数
        def dummy_processing(frame):
            # 模拟一些图像处理操作
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            return edges
        
        # 创建测试视频捕获
        cap = cv2.VideoCapture(0)  # 使用摄像头，如果没有摄像头可以使用视频文件
        
        if not cap.isOpened():
            # 如果没有摄像头，创建模拟帧
            print("未检测到摄像头，使用模拟数据")
            for i in range(200):
                # 创建模拟帧
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                
                # 处理帧
                result, metrics = monitor.process_frame(frame, dummy_processing)
                
                # 显示进度
                if i % 50 == 0:
                    print(f"已处理 {i} 帧, 当前FPS: {metrics.fps:.1f}")
                
                # 模拟处理延迟
                time.sleep(0.01)
        else:
            print("使用摄像头进行实时监控")
            frame_count = 0
            
            while frame_count < 200:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 处理帧
                result, metrics = monitor.process_frame(frame, dummy_processing)
                
                # 显示结果
                cv2.imshow('Original', frame)
                cv2.imshow('Processed', result)
                
                # 显示性能信息
                if frame_count % 30 == 0:
                    print(f"帧 {frame_count}: FPS={metrics.fps:.1f}, 延迟={metrics.latency:.1f}ms")
                
                frame_count += 1
                
                # 按'q'退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
    
    finally:
        # 停止监控
        monitor.stop_monitoring()
    
    # 生成报告
    print("\n=== 性能报告 ===")
    report = monitor.generate_performance_report()
    print(report)
    
    # 获取性能摘要
    summary = monitor.get_performance_summary()
    print(f"\n=== 性能摘要 ===")
    print(f"平均FPS: {summary['performance']['avg_fps']:.1f}")
    print(f"平均延迟: {summary['performance']['avg_latency_ms']:.1f}ms")
    print(f"CPU使用率: {summary['system_usage']['cpu']:.1f}%")
    print(f"内存使用率: {summary['system_usage']['memory']:.1f}%")
    
    print("\n性能监控测试完成！")

if __name__ == "__main__":
    test_performance_monitor()