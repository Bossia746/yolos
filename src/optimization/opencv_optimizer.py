#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV升级优化模块
针对YOLO项目中OpenCV的使用进行性能优化和版本升级
支持动静场景配合使用和不同场景的应用效果提升
"""

import cv2
import numpy as np
import logging
import time
import threading
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import psutil
import os

logger = logging.getLogger(__name__)

class OptimizationLevel(Enum):
    """优化级别"""
    BASIC = "basic"          # 基础优化
    ADVANCED = "advanced"    # 高级优化
    EXTREME = "extreme"      # 极限优化

class SceneType(Enum):
    """场景类型枚举"""
    STATIC = "static"      # 静态场景
    DYNAMIC = "dynamic"    # 动态场景
    MIXED = "mixed"        # 混合场景
    REALTIME = "realtime"  # 实时场景
    LOW_LIGHT = "low_light"  # 低光照场景
    CROWDED = "crowded"    # 拥挤场景

class ProcessingMode(Enum):
    """处理模式"""
    CPU_ONLY = "cpu_only"    # 仅CPU
    GPU_ACCELERATED = "gpu"  # GPU加速
    MULTI_THREADED = "mt"    # 多线程
    OPTIMIZED = "optimized"  # 优化模式

@dataclass
class OpenCVConfig:
    """OpenCV配置信息"""
    version: str
    build_info: str
    has_gpu_support: bool
    has_threading_support: bool
    available_backends: List[str]
    performance_flags: Dict[str, bool]

@dataclass
class OptimizationResult:
    """优化结果"""
    original_fps: float
    optimized_fps: float
    improvement_ratio: float
    memory_usage_before: float
    memory_usage_after: float
    optimization_applied: List[str]
    scene_type: SceneType
    processing_time: float

class OpenCVVersionManager:
    """OpenCV版本管理器"""
    
    def __init__(self):
        self.current_config = self._analyze_current_opencv()
        self.recommended_version = "4.10.0"  # 推荐版本
        self.minimum_version = "4.8.0"       # 最低版本
        
    def _analyze_current_opencv(self) -> OpenCVConfig:
        """分析当前OpenCV配置"""
        try:
            version = cv2.__version__
            build_info = cv2.getBuildInformation()
            
            # 检查GPU支持
            has_gpu = self._check_gpu_support(build_info)
            
            # 检查线程支持
            has_threading = self._check_threading_support(build_info)
            
            # 获取可用后端
            backends = self._get_available_backends()
            
            # 性能标志
            perf_flags = self._get_performance_flags(build_info)
            
            return OpenCVConfig(
                version=version,
                build_info=build_info,
                has_gpu_support=has_gpu,
                has_threading_support=has_threading,
                available_backends=backends,
                performance_flags=perf_flags
            )
            
        except Exception as e:
            logger.error(f"分析OpenCV配置失败: {e}")
            return OpenCVConfig(
                version="unknown",
                build_info="",
                has_gpu_support=False,
                has_threading_support=False,
                available_backends=[],
                performance_flags={}
            )
    
    def _check_gpu_support(self, build_info: str) -> bool:
        """检查GPU支持"""
        gpu_indicators = ['CUDA', 'OpenCL', 'NVIDIA', 'GPU']
        return any(indicator in build_info for indicator in gpu_indicators)
    
    def _check_threading_support(self, build_info: str) -> bool:
        """检查线程支持"""
        threading_indicators = ['TBB', 'OpenMP', 'PTHREADS']
        return any(indicator in build_info for indicator in threading_indicators)
    
    def _get_available_backends(self) -> List[str]:
        """获取可用后端"""
        backends = []
        try:
            # 检查DNN后端
            if hasattr(cv2.dnn, 'DNN_BACKEND_OPENCV'):
                backends.append('OpenCV')
            if hasattr(cv2.dnn, 'DNN_BACKEND_CUDA'):
                backends.append('CUDA')
            if hasattr(cv2.dnn, 'DNN_BACKEND_OPENVINO'):
                backends.append('OpenVINO')
        except:
            pass
        return backends
    
    def _get_performance_flags(self, build_info: str) -> Dict[str, bool]:
        """获取性能标志"""
        flags = {
            'optimized': 'OPTIMIZATION' in build_info,
            'simd': any(x in build_info for x in ['SSE', 'AVX', 'NEON']),
            'parallel': any(x in build_info for x in ['TBB', 'OpenMP']),
            'gpu_ready': any(x in build_info for x in ['CUDA', 'OpenCL'])
        }
        return flags
    
    def check_upgrade_needed(self) -> Tuple[bool, str]:
        """检查是否需要升级"""
        current_version = self.current_config.version
        
        try:
            current_parts = [int(x) for x in current_version.split('.')[:3]]
            recommended_parts = [int(x) for x in self.recommended_version.split('.')[:3]]
            
            if current_parts < recommended_parts:
                return True, f"建议从 {current_version} 升级到 {self.recommended_version}"
            else:
                return False, f"当前版本 {current_version} 已是最新"
                
        except Exception as e:
            logger.error(f"版本比较失败: {e}")
            return True, "无法确定版本，建议升级"
    
    def get_upgrade_benefits(self) -> List[str]:
        """获取升级收益"""
        benefits = [
            "🚀 性能提升: 新版本包含更多SIMD优化",
            "🎯 精度改进: DNN模块精度和稳定性提升",
            "🔧 Bug修复: 修复已知的内存泄漏和崩溃问题",
            "📱 兼容性: 更好的硬件和操作系统兼容性",
            "⚡ GPU加速: 改进的CUDA和OpenCL支持",
            "🧵 多线程: 优化的并行处理能力"
        ]
        return benefits

class SceneAnalyzer:
    """场景分析器"""
    
    def __init__(self):
        self.motion_threshold = 0.1
        self.static_frame_count = 0
        self.dynamic_frame_count = 0
        self.previous_frame = None
        
    def analyze_scene_type(self, frame: np.ndarray) -> SceneType:
        """分析场景类型"""
        if self.previous_frame is None:
            self.previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return SceneType.STATIC
        
        # 计算帧差
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(self.previous_frame, current_gray)
        motion_score = np.mean(frame_diff) / 255.0
        
        # 更新统计
        if motion_score < self.motion_threshold:
            self.static_frame_count += 1
        else:
            self.dynamic_frame_count += 1
        
        # 更新前一帧
        self.previous_frame = current_gray
        
        # 判断场景类型
        total_frames = self.static_frame_count + self.dynamic_frame_count
        if total_frames < 10:  # 初始阶段
            return SceneType.MIXED
        
        dynamic_ratio = self.dynamic_frame_count / total_frames
        
        if dynamic_ratio > 0.7:
            return SceneType.DYNAMIC
        elif dynamic_ratio < 0.3:
            return SceneType.STATIC
        else:
            return SceneType.MIXED
    
    def get_scene_statistics(self) -> Dict[str, Any]:
        """获取场景统计信息"""
        total = self.static_frame_count + self.dynamic_frame_count
        if total == 0:
            return {'static_ratio': 0, 'dynamic_ratio': 0, 'total_frames': 0}
        
        return {
            'static_ratio': self.static_frame_count / total,
            'dynamic_ratio': self.dynamic_frame_count / total,
            'total_frames': total,
            'motion_threshold': self.motion_threshold
        }

class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.ADVANCED):
        self.optimization_level = optimization_level
        self.thread_pool = ThreadPoolExecutor(max_workers=psutil.cpu_count())
        self.optimizations_applied = []
        
    def optimize_for_scene(self, scene_type: SceneType) -> Dict[str, Any]:
        """根据场景类型优化"""
        optimizations = {}
        
        if scene_type == SceneType.STATIC:
            optimizations.update(self._optimize_static_scene())
        elif scene_type == SceneType.DYNAMIC:
            optimizations.update(self._optimize_dynamic_scene())
        elif scene_type == SceneType.MIXED:
            optimizations.update(self._optimize_mixed_scene())
        elif scene_type == SceneType.REALTIME:
            optimizations.update(self._optimize_realtime_scene())
        
        return optimizations
    
    def _optimize_static_scene(self) -> Dict[str, Any]:
        """静态场景优化"""
        optimizations = {
            'frame_skip': 2,  # 跳帧处理
            'roi_processing': True,  # ROI处理
            'background_subtraction': False,  # 关闭背景减除
            'motion_detection': False,  # 关闭运动检测
            'quality_priority': True,  # 质量优先
            'cache_enabled': True  # 启用缓存
        }
        self.optimizations_applied.append('static_scene_optimization')
        return optimizations
    
    def _optimize_dynamic_scene(self) -> Dict[str, Any]:
        """动态场景优化"""
        optimizations = {
            'frame_skip': 0,  # 不跳帧
            'roi_processing': False,  # 全帧处理
            'background_subtraction': True,  # 启用背景减除
            'motion_detection': True,  # 启用运动检测
            'quality_priority': False,  # 速度优先
            'multi_threading': True,  # 多线程处理
            'gpu_acceleration': True  # GPU加速
        }
        self.optimizations_applied.append('dynamic_scene_optimization')
        return optimizations
    
    def _optimize_mixed_scene(self) -> Dict[str, Any]:
        """混合场景优化"""
        optimizations = {
            'adaptive_processing': True,  # 自适应处理
            'frame_skip': 1,  # 轻度跳帧
            'roi_processing': True,  # ROI处理
            'background_subtraction': True,  # 背景减除
            'motion_detection': True,  # 运动检测
            'quality_balance': True,  # 质量平衡
            'cache_enabled': True  # 缓存
        }
        self.optimizations_applied.append('mixed_scene_optimization')
        return optimizations
    
    def _optimize_realtime_scene(self) -> Dict[str, Any]:
        """实时场景优化"""
        optimizations = {
            'low_latency': True,  # 低延迟
            'frame_skip': 0,  # 不跳帧
            'fast_algorithms': True,  # 快速算法
            'reduced_precision': True,  # 降低精度
            'parallel_processing': True,  # 并行处理
            'memory_optimization': True  # 内存优化
        }
        self.optimizations_applied.append('realtime_scene_optimization')
        return optimizations
    
    def apply_opencv_optimizations(self) -> None:
        """应用OpenCV优化设置"""
        try:
            # 设置线程数
            cv2.setNumThreads(psutil.cpu_count())
            self.optimizations_applied.append('thread_optimization')
            
            # 启用优化
            if hasattr(cv2, 'setUseOptimized'):
                cv2.setUseOptimized(True)
                self.optimizations_applied.append('use_optimized')
            
            # 设置内存管理
            if hasattr(cv2, 'setBufferPoolUsage'):
                cv2.setBufferPoolUsage(True)
                self.optimizations_applied.append('buffer_pool')
            
            logger.info(f"已应用OpenCV优化: {self.optimizations_applied}")
            
        except Exception as e:
            logger.error(f"应用OpenCV优化失败: {e}")

class AdaptiveProcessor:
    """自适应处理器"""
    
    def __init__(self):
        self.scene_analyzer = SceneAnalyzer()
        self.performance_optimizer = PerformanceOptimizer()
        self.current_optimizations = {}
        self.performance_history = []
        
    def process_frame_adaptive(self, frame: np.ndarray, 
                             detection_func: callable) -> Tuple[Any, Dict[str, Any]]:
        """自适应帧处理"""
        start_time = time.time()
        
        # 分析场景类型
        scene_type = self.scene_analyzer.analyze_scene_type(frame)
        
        # 获取优化配置
        if scene_type.value not in [opt.get('scene_type') for opt in [self.current_optimizations]]:
            self.current_optimizations = self.performance_optimizer.optimize_for_scene(scene_type)
            self.current_optimizations['scene_type'] = scene_type.value
        
        # 应用优化处理
        processed_frame = self._apply_frame_optimizations(frame, self.current_optimizations)
        
        # 执行检测
        detection_result = detection_func(processed_frame)
        
        # 记录性能
        processing_time = time.time() - start_time
        self.performance_history.append({
            'scene_type': scene_type.value,
            'processing_time': processing_time,
            'frame_size': frame.shape,
            'optimizations': list(self.current_optimizations.keys())
        })
        
        # 保持历史记录在合理范围内
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-50:]
        
        return detection_result, {
            'scene_type': scene_type,
            'processing_time': processing_time,
            'optimizations_applied': self.performance_optimizer.optimizations_applied,
            'scene_stats': self.scene_analyzer.get_scene_statistics()
        }
    
    def _apply_frame_optimizations(self, frame: np.ndarray, 
                                 optimizations: Dict[str, Any]) -> np.ndarray:
        """应用帧优化"""
        processed_frame = frame.copy()
        
        # 跳帧处理
        if optimizations.get('frame_skip', 0) > 0:
            # 这里可以实现跳帧逻辑
            pass
        
        # ROI处理
        if optimizations.get('roi_processing', False):
            # 可以实现ROI裁剪
            h, w = processed_frame.shape[:2]
            roi_margin = 0.1
            y1, y2 = int(h * roi_margin), int(h * (1 - roi_margin))
            x1, x2 = int(w * roi_margin), int(w * (1 - roi_margin))
            processed_frame = processed_frame[y1:y2, x1:x2]
        
        # 质量调整
        if optimizations.get('reduced_precision', False):
            # 降低分辨率以提高速度
            h, w = processed_frame.shape[:2]
            processed_frame = cv2.resize(processed_frame, (w//2, h//2))
        
        return processed_frame
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        if not self.performance_history:
            return {'message': '暂无性能数据'}
        
        # 计算统计信息
        processing_times = [h['processing_time'] for h in self.performance_history]
        scene_types = [h['scene_type'] for h in self.performance_history]
        
        report = {
            'total_frames': len(self.performance_history),
            'avg_processing_time': np.mean(processing_times),
            'min_processing_time': np.min(processing_times),
            'max_processing_time': np.max(processing_times),
            'avg_fps': 1.0 / np.mean(processing_times) if np.mean(processing_times) > 0 else 0,
            'scene_distribution': {scene: scene_types.count(scene) for scene in set(scene_types)},
            'optimizations_used': list(set(
                opt for h in self.performance_history 
                for opt in h.get('optimizations', [])
            ))
        }
        
        return report

class OpenCVOptimizer:
    """OpenCV优化器主类"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.version_manager = OpenCVVersionManager()
        self.adaptive_processor = AdaptiveProcessor()
        self.optimization_results = []
        
        # 应用基础优化
        self.adaptive_processor.performance_optimizer.apply_opencv_optimizations()
        
    def analyze_current_setup(self) -> Dict[str, Any]:
        """分析当前设置"""
        config = self.version_manager.current_config
        upgrade_needed, upgrade_msg = self.version_manager.check_upgrade_needed()
        
        analysis = {
            'opencv_version': config.version,
            'gpu_support': config.has_gpu_support,
            'threading_support': config.has_threading_support,
            'available_backends': config.available_backends,
            'performance_flags': config.performance_flags,
            'upgrade_needed': upgrade_needed,
            'upgrade_message': upgrade_msg,
            'upgrade_benefits': self.version_manager.get_upgrade_benefits() if upgrade_needed else []
        }
        
        return analysis
    
    def optimize_detection_pipeline(self, frames: List[np.ndarray], 
                                  detection_func: callable) -> OptimizationResult:
        """优化检测管道"""
        if not frames:
            raise ValueError("帧列表不能为空")
        
        # 记录初始性能
        start_time = time.time()
        original_results = []
        for frame in frames[:10]:  # 测试前10帧
            frame_start = time.time()
            detection_func(frame)
            original_results.append(time.time() - frame_start)
        
        original_fps = 1.0 / np.mean(original_results) if original_results else 0
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # 应用优化处理
        optimized_results = []
        scene_types = []
        
        for frame in frames[:10]:  # 测试前10帧
            frame_start = time.time()
            result, metadata = self.adaptive_processor.process_frame_adaptive(frame, detection_func)
            optimized_results.append(time.time() - frame_start)
            scene_types.append(metadata['scene_type'])
        
        optimized_fps = 1.0 / np.mean(optimized_results) if optimized_results else 0
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # 计算改进比例
        improvement_ratio = optimized_fps / original_fps if original_fps > 0 else 1.0
        
        # 确定主要场景类型
        main_scene_type = max(set(scene_types), key=scene_types.count)
        
        result = OptimizationResult(
            original_fps=original_fps,
            optimized_fps=optimized_fps,
            improvement_ratio=improvement_ratio,
            memory_usage_before=memory_before,
            memory_usage_after=memory_after,
            optimization_applied=self.adaptive_processor.performance_optimizer.optimizations_applied,
            scene_type=main_scene_type,
            processing_time=time.time() - start_time
        )
        
        self.optimization_results.append(result)
        return result
    
    def get_optimization_recommendations(self) -> List[str]:
        """获取优化建议"""
        recommendations = []
        
        config = self.version_manager.current_config
        
        # 版本升级建议
        upgrade_needed, upgrade_msg = self.version_manager.check_upgrade_needed()
        if upgrade_needed:
            recommendations.append(f"📦 {upgrade_msg}")
        
        # GPU支持建议
        if not config.has_gpu_support:
            recommendations.append("🎮 安装支持GPU的OpenCV版本以获得更好性能")
        
        # 线程支持建议
        if not config.has_threading_support:
            recommendations.append("🧵 启用多线程支持以提升并行处理能力")
        
        # 性能标志建议
        if not config.performance_flags.get('optimized', False):
            recommendations.append("⚡ 启用编译优化标志")
        
        if not config.performance_flags.get('simd', False):
            recommendations.append("🚀 启用SIMD指令集优化")
        
        # 场景特定建议
        if self.optimization_results:
            latest_result = self.optimization_results[-1]
            if latest_result.improvement_ratio < 1.2:
                recommendations.append("🔧 当前优化效果有限，建议检查硬件配置")
            
            if latest_result.scene_type == SceneType.DYNAMIC:
                recommendations.append("🎬 动态场景检测到，建议启用GPU加速")
            elif latest_result.scene_type == SceneType.STATIC:
                recommendations.append("📷 静态场景检测到，建议启用缓存优化")
        
        # 内存优化建议
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
        if memory_usage > 1000:  # 超过1GB
            recommendations.append("💾 内存使用较高，建议启用内存优化")
        
        return recommendations
    
    def export_optimization_report(self, filepath: str) -> None:
        """导出优化报告"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'opencv_analysis': self.analyze_current_setup(),
            'performance_report': self.adaptive_processor.get_performance_report(),
            'optimization_results': [
                {
                    'original_fps': result.original_fps,
                    'optimized_fps': result.optimized_fps,
                    'improvement_ratio': result.improvement_ratio,
                    'memory_before': result.memory_usage_before,
                    'memory_after': result.memory_usage_after,
                    'scene_type': result.scene_type.value,
                    'optimizations': result.optimization_applied
                }
                for result in self.optimization_results
            ],
            'recommendations': self.get_optimization_recommendations()
        }
        
        try:
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"优化报告已导出到: {filepath}")
        except Exception as e:
            logger.error(f"导出报告失败: {e}")

# 测试代码
def test_opencv_optimizer():
    """测试OpenCV优化器"""
    print("🔧 OpenCV优化器测试")
    
    # 创建优化器
    optimizer = OpenCVOptimizer()
    
    # 分析当前设置
    print("\n📊 当前OpenCV设置分析:")
    analysis = optimizer.analyze_current_setup()
    for key, value in analysis.items():
        print(f"  {key}: {value}")
    
    # 创建测试帧
    test_frames = []
    for i in range(20):
        # 创建不同类型的测试帧
        if i < 5:  # 静态帧
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        elif i < 15:  # 动态帧
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        else:  # 混合帧
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 150
            frame[100:200, 100:200] = np.random.randint(0, 255, (100, 100, 3))
        
        test_frames.append(frame)
    
    # 模拟检测函数
    def mock_detection(frame):
        # 模拟YOLO检测延迟
        time.sleep(0.01)  # 10ms延迟
        return [{'class': 'person', 'confidence': 0.9, 'bbox': [100, 100, 200, 200]}]
    
    # 运行优化测试
    print("\n🚀 运行优化测试...")
    result = optimizer.optimize_detection_pipeline(test_frames, mock_detection)
    
    print(f"\n📈 优化结果:")
    print(f"  原始FPS: {result.original_fps:.2f}")
    print(f"  优化后FPS: {result.optimized_fps:.2f}")
    print(f"  性能提升: {result.improvement_ratio:.2f}x")
    print(f"  内存使用 (前): {result.memory_usage_before:.1f} MB")
    print(f"  内存使用 (后): {result.memory_usage_after:.1f} MB")
    print(f"  主要场景类型: {result.scene_type.value}")
    print(f"  应用的优化: {result.optimization_applied}")
    
    # 获取优化建议
    print("\n💡 优化建议:")
    recommendations = optimizer.get_optimization_recommendations()
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    # 获取性能报告
    print("\n📊 性能报告:")
    perf_report = optimizer.adaptive_processor.get_performance_report()
    for key, value in perf_report.items():
        print(f"  {key}: {value}")
    
    # 导出报告
    report_path = "opencv_optimization_report.json"
    optimizer.export_optimization_report(report_path)
    print(f"\n📄 详细报告已导出到: {report_path}")
    
    print("\n✅ OpenCV优化器测试完成")

if __name__ == "__main__":
    test_opencv_optimizer()