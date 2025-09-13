#!/usr/bin/env python3
"""距离测量功能测试脚本

测试摄像头测距功能的各个组件，包括：
1. 距离估算器测试
2. 物体检测器测试
3. 相机标定工具测试
4. GUI界面测试
"""

import cv2
import numpy as np
import os
import sys
import json
from pathlib import Path
import time

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from recognition.distance_estimator import DistanceEstimator, RealTimeDistanceEstimator
    from recognition.enhanced_object_detector import EnhancedObjectDetector
    from recognition.camera_calibration_tool import CameraCalibrationTool
    from gui.distance_measurement_gui import DistanceMeasurementGUI
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保所有必要的模块都已正确安装")
    sys.exit(1)


class DistanceMeasurementTester:
    """距离测量功能测试器"""
    
    def __init__(self):
        self.test_results = []
        self.test_images_dir = Path("test_images")
        self.test_images_dir.mkdir(exist_ok=True)
        
        print("🚀 距离测量功能测试器初始化完成")
    
    def test_object_detector(self):
        """测试物体检测器"""
        print("\n📋 测试物体检测器...")
        
        detector = EnhancedObjectDetector()
        
        # 创建测试图像
        test_image = self.create_test_image()
        
        try:
            # 测试边缘检测
            edge_detections = detector.detect_by_edge(test_image, 'rectangle')
            print(f"  ✅ 边缘检测: 检测到 {len(edge_detections)} 个矩形物体")
            
            # 测试颜色检测
            color_detections = detector.detect_by_color(test_image, 'white')
            print(f"  ✅ 颜色检测: 检测到 {len(color_detections)} 个白色物体")
            
            # 测试最大物体检测
            largest = detector.detect_largest_object(test_image)
            if largest:
                print(f"  ✅ 最大物体检测: 面积 {largest['area']:.0f} 像素")
            else:
                print("  ⚠️ 最大物体检测: 未检测到物体")
            
            self.test_results.append({
                'test': 'object_detector',
                'status': 'passed',
                'details': f'检测到 {len(edge_detections)} 个边缘物体'
            })
            
        except Exception as e:
            print(f"  ❌ 物体检测器测试失败: {e}")
            self.test_results.append({
                'test': 'object_detector',
                'status': 'failed',
                'error': str(e)
            })
    
    def test_distance_estimator(self):
        """测试距离估算器"""
        print("\n📏 测试距离估算器...")
        
        estimator = DistanceEstimator()
        
        try:
            # 设置测试焦距
            estimator.focal_length = 500.0
            
            # 创建测试图像
            test_image = self.create_test_image_with_known_object()
            
            # 测试距离估算
            known_width = 21.0  # A4纸宽度 (cm)
            result = estimator.estimate_distance(test_image, known_width)
            
            if result:
                distance = result['distance']
                pixel_width = result['pixel_width']
                print(f"  ✅ 距离估算成功: {distance:.1f} cm")
                print(f"  📐 像素宽度: {pixel_width:.1f} pixels")
                print(f"  🎯 焦距: {estimator.focal_length:.1f}")
                
                self.test_results.append({
                    'test': 'distance_estimator',
                    'status': 'passed',
                    'distance': distance,
                    'pixel_width': pixel_width
                })
            else:
                print("  ⚠️ 距离估算: 未检测到目标物体")
                self.test_results.append({
                    'test': 'distance_estimator',
                    'status': 'no_detection'
                })
                
        except Exception as e:
            print(f"  ❌ 距离估算器测试失败: {e}")
            self.test_results.append({
                'test': 'distance_estimator',
                'status': 'failed',
                'error': str(e)
            })
    
    def test_calibration_tool(self):
        """测试标定工具"""
        print("\n🎯 测试相机标定工具...")
        
        calibration_tool = CameraCalibrationTool()
        
        try:
            # 测试已知物体信息
            known_objects = calibration_tool.known_objects
            print(f"  ✅ 支持的物体类型: {list(known_objects.keys())}")
            
            # 测试标定摘要
            summary = calibration_tool.get_calibration_summary()
            print(f"  📊 标定摘要: {summary.get('message', '有标定记录')}")
            
            # 测试添加自定义物体
            calibration_tool.add_custom_object("test_object", 10.0, 5.0, "cm")
            print("  ✅ 自定义物体添加成功")
            
            self.test_results.append({
                'test': 'calibration_tool',
                'status': 'passed',
                'supported_objects': len(known_objects)
            })
            
        except Exception as e:
            print(f"  ❌ 标定工具测试失败: {e}")
            self.test_results.append({
                'test': 'calibration_tool',
                'status': 'failed',
                'error': str(e)
            })
    
    def test_camera_access(self):
        """测试摄像头访问"""
        print("\n📹 测试摄像头访问...")
        
        try:
            # 尝试打开摄像头
            cap = cv2.VideoCapture(0)
            
            if cap.isOpened():
                # 读取一帧
                ret, frame = cap.read()
                if ret:
                    h, w = frame.shape[:2]
                    print(f"  ✅ 摄像头访问成功: {w}x{h}")
                    
                    # 保存测试图像
                    test_image_path = self.test_images_dir / "camera_test.jpg"
                    cv2.imwrite(str(test_image_path), frame)
                    print(f"  💾 测试图像已保存: {test_image_path}")
                    
                    self.test_results.append({
                        'test': 'camera_access',
                        'status': 'passed',
                        'resolution': f'{w}x{h}'
                    })
                else:
                    print("  ⚠️ 摄像头无法读取图像")
                    self.test_results.append({
                        'test': 'camera_access',
                        'status': 'no_frame'
                    })
            else:
                print("  ❌ 无法打开摄像头")
                self.test_results.append({
                    'test': 'camera_access',
                    'status': 'failed',
                    'error': 'Cannot open camera'
                })
            
            cap.release()
            
        except Exception as e:
            print(f"  ❌ 摄像头测试失败: {e}")
            self.test_results.append({
                'test': 'camera_access',
                'status': 'failed',
                'error': str(e)
            })
    
    def test_gui_components(self):
        """测试GUI组件"""
        print("\n🖥️ 测试GUI组件...")
        
        try:
            # 测试GUI类导入和初始化
            print("  📦 测试GUI类导入...")
            
            # 这里不实际创建GUI窗口，只测试类的可用性
            gui_class = DistanceMeasurementGUI
            print("  ✅ GUI类导入成功")
            
            self.test_results.append({
                'test': 'gui_components',
                'status': 'passed',
                'note': 'GUI类可用，需要手动测试界面'
            })
            
        except Exception as e:
            print(f"  ❌ GUI组件测试失败: {e}")
            self.test_results.append({
                'test': 'gui_components',
                'status': 'failed',
                'error': str(e)
            })
    
    def create_test_image(self) -> np.ndarray:
        """创建测试图像"""
        # 创建640x480的黑色图像
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 添加一个白色矩形 (模拟A4纸)
        cv2.rectangle(image, (200, 150), (440, 330), (255, 255, 255), -1)
        
        # 添加一些噪声
        noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)
        image = cv2.add(image, noise)
        
        return image
    
    def create_test_image_with_known_object(self) -> np.ndarray:
        """创建包含已知物体的测试图像"""
        # 创建更大的图像
        image = np.zeros((600, 800, 3), dtype=np.uint8)
        
        # 添加背景纹理
        cv2.rectangle(image, (0, 0), (800, 600), (50, 50, 50), -1)
        
        # 添加A4纸模拟 (21cm x 29.7cm, 假设在30cm距离处)
        # 假设焦距500，30cm距离，21cm宽度对应约350像素
        paper_width = 350
        paper_height = int(paper_width * 29.7 / 21)  # 保持A4比例
        
        x = (800 - paper_width) // 2
        y = (600 - paper_height) // 2
        
        # 绘制白色纸张
        cv2.rectangle(image, (x, y), (x + paper_width, y + paper_height), (240, 240, 240), -1)
        
        # 添加边框
        cv2.rectangle(image, (x, y), (x + paper_width, y + paper_height), (200, 200, 200), 2)
        
        # 添加一些文字模拟
        cv2.putText(image, "A4 Paper", (x + 50, y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        
        return image
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🧪 开始距离测量功能全面测试\n")
        print("=" * 50)
        
        # 运行各项测试
        self.test_object_detector()
        self.test_distance_estimator()
        self.test_calibration_tool()
        self.test_camera_access()
        self.test_gui_components()
        
        # 生成测试报告
        self.generate_test_report()
    
    def generate_test_report(self):
        """生成测试报告"""
        print("\n" + "=" * 50)
        print("📊 测试报告")
        print("=" * 50)
        
        passed_tests = 0
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            test_name = result['test']
            status = result['status']
            
            if status == 'passed':
                print(f"✅ {test_name}: 通过")
                passed_tests += 1
            elif status == 'failed':
                print(f"❌ {test_name}: 失败 - {result.get('error', '未知错误')}")
            else:
                print(f"⚠️ {test_name}: {status}")
                passed_tests += 0.5  # 部分通过
        
        success_rate = (passed_tests / total_tests) * 100
        print(f"\n📈 测试通过率: {success_rate:.1f}% ({passed_tests}/{total_tests})")
        
        # 保存测试结果
        report_file = "distance_measurement_test_report.json"
        test_report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': success_rate,
            'results': self.test_results
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(test_report, f, indent=2, ensure_ascii=False)
        
        print(f"💾 详细测试报告已保存: {report_file}")
        
        # 给出建议
        print("\n💡 使用建议:")
        if success_rate >= 80:
            print("  🎉 系统功能基本正常，可以开始使用GUI进行测试")
            print("  🚀 运行: python -m src.gui.distance_measurement_gui")
        else:
            print("  ⚠️ 部分功能存在问题，建议先解决失败的测试项")
        
        print("  📖 查看详细文档了解使用方法")
        print("  🎯 建议先进行相机标定以获得准确的测量结果")


def main():
    """主函数"""
    print("🎯 YOLOS 摄像头距离测量功能测试")
    print("基于相似三角形原理的距离测量系统测试")
    print()
    
    tester = DistanceMeasurementTester()
    
    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n⏹️ 测试被用户中断")
    except Exception as e:
        print(f"\n💥 测试过程中发生意外错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🏁 测试完成")


if __name__ == "__main__":
    main()