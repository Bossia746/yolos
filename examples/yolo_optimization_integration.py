#!/usr/bin/env python3
"""
YOLO优化集成示例
展示如何在YOLOS系统中集成最新的YOLO算法优化
"""

import sys
import os
import time
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from models.yolov11_detector import YOLOv11Detector
    from models.advanced_yolo_optimizations import ModelOptimizer, PerformanceProfiler
    from utils.camera_utils import initialize_camera
    from utils.visualization_utils import draw_detections
    from core.logger import get_logger
    from core.config import YOLOSConfig
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("使用简化版本继续运行...")
    
    # 简化的日志记录
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 模拟的检测结果类
    class DetectionResult:
        def __init__(self, bbox, class_id, class_name, confidence):
            self.bbox = bbox
            self.class_id = class_id
            self.class_name = class_name
            self.confidence = confidence
else:
    logger = get_logger(__name__)

class YOLOOptimizationDemo:
    """YOLO优化演示类"""
    
    def __init__(self):
        self.config = self._load_config()
        self.detectors = {}
        self.performance_data = {}
        self.current_detector = None
        
    def _load_config(self):
        """加载配置"""
        return {
            'model_variants': ['n', 's', 'm'],  # nano, small, medium
            'optimization_techniques': [
                'baseline',
                'half_precision', 
                'quantization',
                'tensorrt'
            ],
            'test_duration': 30,  # 测试持续时间(秒)
            'confidence_threshold': 0.5,
            'nms_threshold': 0.4
        }
        
    def initialize_detectors(self):
        """初始化不同的检测器变体"""
        print("🔧 初始化YOLO检测器变体...")
        
        for variant in self.config['model_variants']:
            try:
                # 基础检测器
                detector = YOLOv11Detector(
                    model_size=variant,
                    confidence_threshold=self.config['confidence_threshold'],
                    half_precision=False
                )
                self.detectors[f'yolov11{variant}_baseline'] = detector
                print(f"   ✅ YOLOv11{variant} 基础版本")
                
                # 半精度优化版本
                detector_fp16 = YOLOv11Detector(
                    model_size=variant,
                    confidence_threshold=self.config['confidence_threshold'],
                    half_precision=True
                )
                self.detectors[f'yolov11{variant}_fp16'] = detector_fp16
                print(f"   ✅ YOLOv11{variant} 半精度版本")
                
            except Exception as e:
                print(f"   ❌ 初始化YOLOv11{variant}失败: {e}")
                # 创建模拟检测器
                self.detectors[f'yolov11{variant}_baseline'] = self._create_mock_detector(variant)
                
        print(f"📊 共初始化 {len(self.detectors)} 个检测器变体")
        
    def _create_mock_detector(self, variant):
        """创建模拟检测器用于演示"""
        class MockDetector:
            def __init__(self, variant):
                self.variant = variant
                self.model_size = variant
                
            def detect(self, frame):
                # 模拟检测延迟
                time.sleep(0.01 if variant == 'n' else 0.02 if variant == 's' else 0.03)
                
                # 返回模拟检测结果
                h, w = frame.shape[:2]
                return [
                    DetectionResult(
                        bbox=(w//4, h//4, w//2, h//2),
                        class_id=0,
                        class_name='person',
                        confidence=0.85
                    )
                ]
                
        return MockDetector(variant)
        
    def benchmark_detector(self, detector_name: str, test_frames: List[np.ndarray]) -> Dict[str, float]:
        """基准测试单个检测器"""
        detector = self.detectors[detector_name]
        
        print(f"🔍 测试 {detector_name}...")
        
        start_time = time.time()
        total_detections = 0
        processed_frames = 0
        
        for frame in test_frames:
            try:
                results = detector.detect(frame)
                total_detections += len(results) if results else 0
                processed_frames += 1
            except Exception as e:
                print(f"   检测失败: {e}")
                continue
                
        end_time = time.time()
        
        # 计算性能指标
        total_time = end_time - start_time
        fps = processed_frames / total_time if total_time > 0 else 0
        avg_detections = total_detections / processed_frames if processed_frames > 0 else 0
        
        metrics = {
            'fps': fps,
            'total_time': total_time,
            'avg_detections': avg_detections,
            'processed_frames': processed_frames
        }
        
        print(f"   📈 FPS: {fps:.2f}, 平均检测数: {avg_detections:.1f}")
        
        return metrics
        
    def run_comprehensive_benchmark(self):
        """运行综合基准测试"""
        print("\n🚀 开始综合性能基准测试")
        print("=" * 50)
        
        # 准备测试数据
        test_frames = self._prepare_test_frames()
        
        # 测试所有检测器
        for detector_name in self.detectors.keys():
            metrics = self.benchmark_detector(detector_name, test_frames)
            self.performance_data[detector_name] = metrics
            
        # 生成性能报告
        self._generate_performance_report()
        
    def _prepare_test_frames(self, num_frames: int = 50) -> List[np.ndarray]:
        """准备测试帧"""
        print("📸 准备测试数据...")
        
        frames = []
        
        # 尝试从摄像头获取
        cap = initialize_camera(0) if 'initialize_camera' in globals() else None
        
        if cap is not None:
            print("   从摄像头采集测试帧...")
            collected = 0
            while collected < num_frames:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (640, 480))
                    frames.append(frame)
                    collected += 1
                    
                    if collected % 10 == 0:
                        print(f"   已采集 {collected}/{num_frames} 帧")
                else:
                    break
                    
            cap.release()
        else:
            print("   生成合成测试帧...")
            # 生成合成测试数据
            for i in range(num_frames):
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                
                # 添加一些几何形状模拟目标
                cv2.rectangle(frame, (100, 100), (200, 200), (255, 0, 0), -1)
                cv2.circle(frame, (400, 300), 50, (0, 255, 0), -1)
                cv2.rectangle(frame, (300, 50), (500, 150), (0, 0, 255), 2)
                
                frames.append(frame)
                
        print(f"   ✅ 准备完成，共 {len(frames)} 帧测试数据")
        return frames
        
    def _generate_performance_report(self):
        """生成性能报告"""
        print("\n📊 性能测试结果")
        print("=" * 50)
        
        # 按FPS排序
        sorted_results = sorted(
            self.performance_data.items(),
            key=lambda x: x[1]['fps'],
            reverse=True
        )
        
        print(f"{'检测器':<20} {'FPS':<8} {'平均检测数':<10} {'总时间(s)':<10}")
        print("-" * 50)
        
        for detector_name, metrics in sorted_results:
            print(f"{detector_name:<20} {metrics['fps']:<8.2f} "
                  f"{metrics['avg_detections']:<10.1f} {metrics['total_time']:<10.2f}")
                  
        # 找出最佳性能
        best_fps = max(self.performance_data.items(), key=lambda x: x[1]['fps'])
        best_accuracy = max(self.performance_data.items(), key=lambda x: x[1]['avg_detections'])
        
        print(f"\n🏆 性能冠军:")
        print(f"   最快速度: {best_fps[0]} ({best_fps[1]['fps']:.2f} FPS)")
        print(f"   最多检测: {best_accuracy[0]} ({best_accuracy[1]['avg_detections']:.1f} 个目标)")
        
        # 保存详细报告
        self._save_detailed_report()
        
    def _save_detailed_report(self):
        """保存详细报告"""
        report_dir = Path('test_results/yolo_optimization')
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        report_path = report_dir / f'yolo_optimization_report_{timestamp}.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("YOLO优化集成测试报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"测试配置: {self.config}\n\n")
            
            f.write("详细性能数据:\n")
            f.write("-" * 30 + "\n")
            
            for detector_name, metrics in self.performance_data.items():
                f.write(f"\n{detector_name}:\n")
                for key, value in metrics.items():
                    f.write(f"  {key}: {value}\n")
                    
            # 优化建议
            f.write(f"\n优化建议:\n")
            f.write("-" * 20 + "\n")
            f.write("1. 实时应用推荐使用YOLOv11n + 半精度优化\n")
            f.write("2. 高精度需求推荐使用YOLOv11m基础版本\n")
            f.write("3. 移动端部署考虑量化优化\n")
            f.write("4. GPU部署可启用TensorRT加速\n")
            
        print(f"📄 详细报告已保存: {report_path}")
        
    def demonstrate_real_time_optimization(self):
        """演示实时优化效果"""
        print("\n🎥 实时优化效果演示")
        print("=" * 50)
        
        # 选择最快的检测器
        if not self.performance_data:
            print("⚠️ 请先运行基准测试")
            return
            
        fastest_detector = max(
            self.performance_data.items(),
            key=lambda x: x[1]['fps']
        )[0]
        
        print(f"🚀 使用最快检测器: {fastest_detector}")
        
        # 初始化摄像头
        cap = initialize_camera(0) if 'initialize_camera' in globals() else None
        
        if cap is None:
            print("❌ 无法访问摄像头，跳过实时演示")
            return
            
        detector = self.detectors[fastest_detector]
        
        print("📹 开始实时检测 (按 'q' 退出)...")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 执行检测
                results = detector.detect(frame)
                
                # 绘制结果
                if 'draw_detections' in globals() and results:
                    frame = draw_detections(frame, results)
                else:
                    # 简单绘制
                    for result in results or []:
                        if hasattr(result, 'bbox'):
                            x, y, w, h = result.bbox
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(frame, f"{result.class_name}: {result.confidence:.2f}",
                                      (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 显示FPS
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                cv2.putText(frame, f"FPS: {fps:.1f} | Model: {fastest_detector}",
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 显示帧
                cv2.imshow('YOLO优化演示', frame)
                
                # 检查退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n⏹️ 演示被用户中断")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
        print(f"✅ 实时演示完成，平均FPS: {fps:.2f}")
        
    def show_optimization_comparison(self):
        """显示优化对比"""
        print("\n📈 YOLO优化技术对比")
        print("=" * 50)
        
        optimization_info = {
            'YOLOv11架构优化': {
                'C3k2模块': '改进的特征提取，减少参数量',
                '增强SPPF': '更好的多尺度特征融合',
                '优化检测头': '提高小目标检测精度'
            },
            '推理优化技术': {
                '半精度(FP16)': '2倍内存节省，1.5-2倍速度提升',
                '动态量化': '4倍模型压缩，轻微精度损失',
                'TensorRT': '3-5倍GPU推理加速'
            },
            '训练优化策略': {
                'Mosaic增强': '提高模型泛化能力',
                'Auto-anchor': '自动优化anchor尺寸',
                'Label smoothing': '减少过拟合风险'
            }
        }
        
        for category, techniques in optimization_info.items():
            print(f"\n🔧 {category}:")
            for technique, description in techniques.items():
                print(f"   • {technique}: {description}")
                
        # 性能提升预期
        print(f"\n📊 性能提升预期:")
        print(f"   • YOLOv11 vs YOLOv8: +15% mAP, +20% FPS")
        print(f"   • 半精度优化: +50-100% FPS")
        print(f"   • TensorRT优化: +200-400% FPS")
        print(f"   • 模型量化: -75% 模型大小")
        
    def run_complete_demo(self):
        """运行完整演示"""
        print("🎯 YOLO优化集成完整演示")
        print("=" * 60)
        
        try:
            # 1. 初始化检测器
            self.initialize_detectors()
            
            # 2. 显示优化对比信息
            self.show_optimization_comparison()
            
            # 3. 运行性能基准测试
            self.run_comprehensive_benchmark()
            
            # 4. 实时演示(可选)
            user_input = input("\n是否运行实时摄像头演示? (y/n): ").lower().strip()
            if user_input == 'y':
                self.demonstrate_real_time_optimization()
            
            print("\n✅ 完整演示结束！")
            print("📁 查看 test_results/yolo_optimization/ 目录获取详细报告")
            
        except KeyboardInterrupt:
            print("\n⏹️ 演示被用户中断")
        except Exception as e:
            print(f"\n❌ 演示过程中发生错误: {e}")
            logger.error(f"Demo error: {e}")

def main():
    """主函数"""
    print("YOLO优化集成演示")
    print("展示最新YOLO算法优化在实际系统中的应用")
    print()
    
    # 创建演示实例
    demo = YOLOOptimizationDemo()
    
    # 运行完整演示
    demo.run_complete_demo()

if __name__ == "__main__":
    main()