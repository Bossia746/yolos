#!/usr/bin/env python3
"""
YOLO性能对比测试
测试不同YOLO版本在相同数据集上的性能表现
"""

import sys
import os
import time
import numpy as np
import cv2
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from models.yolov11_detector import YOLOv11Detector
    from models.advanced_yolo_optimizations import ModelOptimizer, PerformanceProfiler
    from models.yolo_benchmark_system import YOLOBenchmarkSuite
    from utils.camera_utils import initialize_camera
    from utils.visualization_utils import draw_detections
    from core.logger import get_logger
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保项目路径正确且所有依赖已安装")
    sys.exit(1)

logger = get_logger(__name__)

class YOLOPerformanceComparison:
    """YOLO性能对比测试类"""
    
    def __init__(self):
        self.results = {}
        self.test_images = []
        self.benchmark_suite = YOLOBenchmarkSuite(output_dir='test_results/yolo_comparison')
        
    def prepare_test_data(self, num_frames=50):
        """准备测试数据"""
        print("📸 准备测试数据...")
        
        # 尝试从摄像头获取测试帧
        cap = initialize_camera(0)
        if cap is None:
            print("⚠️ 无法访问摄像头，使用合成测试数据")
            self._generate_synthetic_data(num_frames)
            return
            
        frames_collected = 0
        while frames_collected < num_frames:
            ret, frame = cap.read()
            if ret:
                # 调整图像大小以提高测试效率
                frame = cv2.resize(frame, (640, 480))
                self.test_images.append(frame.copy())
                frames_collected += 1
                
                if frames_collected % 10 == 0:
                    print(f"   已收集 {frames_collected}/{num_frames} 帧")
            else:
                break
                
        cap.release()
        print(f"✅ 测试数据准备完成，共 {len(self.test_images)} 帧")
        
    def _generate_synthetic_data(self, num_frames):
        """生成合成测试数据"""
        for i in range(num_frames):
            # 创建随机彩色图像
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # 添加一些简单的几何形状模拟目标
            cv2.rectangle(frame, (100, 100), (200, 200), (255, 0, 0), -1)
            cv2.circle(frame, (400, 300), 50, (0, 255, 0), -1)
            
            self.test_images.append(frame)
            
    def test_yolov11_variants(self):
        """测试YOLOv11不同变体"""
        print("\n🔬 测试YOLOv11不同变体...")
        
        variants = ['n', 's', 'm']  # nano, small, medium
        
        for variant in variants:
            print(f"\n📊 测试YOLOv11-{variant}...")
            
            try:
                # 创建检测器
                detector = YOLOv11Detector(
                    model_size=variant,
                    half_precision=True,
                    confidence_threshold=0.5
                )
                
                # 性能测试
                start_time = time.time()
                detections_count = 0
                
                for i, frame in enumerate(self.test_images[:20]):  # 测试前20帧
                    try:
                        results = detector.detect(frame)
                        detections_count += len(results) if results else 0
                        
                        if i % 5 == 0:
                            print(f"   处理进度: {i+1}/20")
                            
                    except Exception as e:
                        print(f"   检测失败: {e}")
                        continue
                        
                end_time = time.time()
                
                # 计算性能指标
                total_time = end_time - start_time
                fps = 20 / total_time if total_time > 0 else 0
                avg_detections = detections_count / 20
                
                self.results[f'YOLOv11-{variant}'] = {
                    'total_time': total_time,
                    'fps': fps,
                    'avg_detections': avg_detections,
                    'model_size': variant
                }
                
                print(f"   ✅ 完成 - FPS: {fps:.2f}, 平均检测数: {avg_detections:.1f}")
                
            except Exception as e:
                print(f"   ❌ YOLOv11-{variant} 测试失败: {e}")
                
    def test_optimization_techniques(self):
        """测试优化技术"""
        print("\n⚡ 测试优化技术...")
        
        try:
            # 创建基础检测器
            base_detector = YOLOv11Detector(model_size='n')
            
            # 创建优化器
            optimizer = ModelOptimizer()
            profiler = PerformanceProfiler()
            
            # 测试基础性能
            print("   测试基础性能...")
            base_time = self._benchmark_detector(base_detector, "基础YOLOv11-n")
            
            # 测试量化优化
            print("   测试量化优化...")
            try:
                quantized_detector = optimizer.quantize_model(base_detector)
                quantized_time = self._benchmark_detector(quantized_detector, "量化YOLOv11-n")
                
                speedup = base_time / quantized_time if quantized_time > 0 else 0
                print(f"   量化加速比: {speedup:.2f}x")
                
            except Exception as e:
                print(f"   量化测试失败: {e}")
                
        except Exception as e:
            print(f"   优化技术测试失败: {e}")
            
    def _benchmark_detector(self, detector, name):
        """基准测试检测器"""
        start_time = time.time()
        
        for frame in self.test_images[:10]:  # 测试前10帧
            try:
                detector.detect(frame)
            except:
                continue
                
        end_time = time.time()
        total_time = end_time - start_time
        
        fps = 10 / total_time if total_time > 0 else 0
        print(f"   {name}: {fps:.2f} FPS")
        
        return total_time
        
    def run_comprehensive_benchmark(self):
        """运行综合基准测试"""
        print("\n📈 运行综合基准测试...")
        
        try:
            # 使用基准测试套件
            test_config = {
                'models': ['yolov11n', 'yolov11s'],
                'batch_sizes': [1, 4],
                'input_sizes': [(640, 640)],
                'precision': ['fp32', 'fp16']
            }
            
            results = self.benchmark_suite.run_comprehensive_benchmark(
                test_images=self.test_images[:10],
                config=test_config
            )
            
            print("   ✅ 综合基准测试完成")
            return results
            
        except Exception as e:
            print(f"   ❌ 综合基准测试失败: {e}")
            return None
            
    def generate_report(self):
        """生成性能报告"""
        print("\n📋 生成性能报告...")
        
        if not self.results:
            print("   ⚠️ 没有测试结果可报告")
            return
            
        # 创建报告目录
        report_dir = Path('test_results/yolo_performance')
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文本报告
        report_path = report_dir / f'performance_report_{int(time.time())}.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("YOLO性能对比测试报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"测试图像数量: {len(self.test_images)}\n\n")
            
            f.write("性能结果:\n")
            f.write("-" * 30 + "\n")
            
            # 按FPS排序
            sorted_results = sorted(
                self.results.items(), 
                key=lambda x: x[1]['fps'], 
                reverse=True
            )
            
            for model_name, metrics in sorted_results:
                f.write(f"\n{model_name}:\n")
                f.write(f"  FPS: {metrics['fps']:.2f}\n")
                f.write(f"  总时间: {metrics['total_time']:.2f}s\n")
                f.write(f"  平均检测数: {metrics['avg_detections']:.1f}\n")
                
        print(f"   📄 报告已保存: {report_path}")
        
        # 打印控制台摘要
        print("\n🏆 性能排行榜:")
        for i, (model_name, metrics) in enumerate(sorted_results, 1):
            print(f"   {i}. {model_name}: {metrics['fps']:.2f} FPS")
            
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始YOLO性能对比测试")
        print("=" * 50)
        
        try:
            # 准备测试数据
            self.prepare_test_data(30)
            
            # 运行各项测试
            self.test_yolov11_variants()
            self.test_optimization_techniques()
            self.run_comprehensive_benchmark()
            
            # 生成报告
            self.generate_report()
            
            print("\n✅ 所有测试完成！")
            
        except KeyboardInterrupt:
            print("\n⏹️ 测试被用户中断")
        except Exception as e:
            print(f"\n❌ 测试过程中发生错误: {e}")
            logger.error(f"Performance test error: {e}")

def main():
    """主函数"""
    print("YOLO性能对比测试工具")
    print("支持YOLOv11各变体性能测试和优化技术验证")
    print()
    
    # 创建测试实例
    tester = YOLOPerformanceComparison()
    
    # 运行测试
    tester.run_all_tests()

if __name__ == "__main__":
    main()