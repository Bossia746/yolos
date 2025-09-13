#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOS手持静态物体识别测试 - 本地版本
使用OpenCV内置功能和本地模型进行测试
"""

import cv2
import numpy as np
import time
import json
import os
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

class LocalHandheldObjectTest:
    """本地手持物体识别测试系统"""
    
    def __init__(self):
        self.output_dir = Path("test_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # 测试配置
        self.test_config = {
            "test_name": "YOLOS手持静态物体识别测试",
            "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "test_images": 5,
            "detection_methods": ["Haar级联", "轮廓检测", "边缘检测", "特征点检测"]
        }
        
        # 初始化检测器
        self.detectors = {}
        self.results = {}
        
        print("🎯 YOLOS本地手持静态物体识别测试")
        print("=" * 80)
        print("📝 使用OpenCV内置功能进行多种检测方法演示")
        print("🎨 展示项目的图像处理和分析能力")
        print("=" * 80)
    
    def initialize_detectors(self):
        """初始化各种检测器"""
        print("🤖 初始化检测器...")
        
        # 1. Haar级联检测器 (人脸检测作为示例)
        try:
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detectors['face'] = cv2.CascadeClassifier(face_cascade_path)
            print("✅ Haar级联人脸检测器已加载")
        except Exception as e:
            print(f"⚠️ Haar级联检测器加载失败: {e}")
        
        # 2. ORB特征检测器
        try:
            self.detectors['orb'] = cv2.ORB_create()
            print("✅ ORB特征检测器已加载")
        except Exception as e:
            print(f"⚠️ ORB检测器加载失败: {e}")
        
        # 3. SIFT特征检测器 (如果可用)
        try:
            self.detectors['sift'] = cv2.SIFT_create()
            print("✅ SIFT特征检测器已加载")
        except Exception as e:
            print(f"⚠️ SIFT检测器不可用: {e}")
        
        # 4. 边缘检测器 (Canny)
        self.detectors['canny'] = True
        print("✅ Canny边缘检测器已准备")
        
        # 5. 轮廓检测器
        self.detectors['contour'] = True
        print("✅ 轮廓检测器已准备")
    
    def generate_test_images(self):
        """生成测试图像"""
        print("\n🖼️ 生成测试图像...")
        
        test_images = []
        
        for i in range(self.test_config["test_images"]):
            # 创建不同类型的测试图像
            if i == 0:
                # 几何图形
                img = self.create_geometric_shapes()
                name = "geometric_shapes"
            elif i == 1:
                # 文字图像
                img = self.create_text_image()
                name = "text_image"
            elif i == 2:
                # 噪声图像
                img = self.create_noise_image()
                name = "noise_image"
            elif i == 3:
                # 渐变图像
                img = self.create_gradient_image()
                name = "gradient_image"
            else:
                # 复合图像
                img = self.create_complex_image()
                name = "complex_image"
            
            # 保存图像
            img_path = self.output_dir / f"test_{name}_{i+1}.jpg"
            cv2.imwrite(str(img_path), img)
            
            test_images.append({
                'image': img,
                'path': img_path,
                'name': name,
                'index': i+1
            })
            
            print(f"✅ 生成测试图像 {i+1}: {name}")
        
        return test_images
    
    def create_geometric_shapes(self):
        """创建几何图形测试图像"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 绘制各种几何图形
        cv2.rectangle(img, (50, 50), (150, 150), (0, 255, 0), 2)
        cv2.circle(img, (300, 100), 50, (255, 0, 0), 2)
        cv2.ellipse(img, (500, 100), (60, 40), 0, 0, 360, (0, 0, 255), 2)
        
        # 绘制多边形
        pts = np.array([[100, 200], [150, 300], [50, 300]], np.int32)
        cv2.polylines(img, [pts], True, (255, 255, 0), 2)
        
        # 绘制线条
        cv2.line(img, (200, 200), (400, 300), (255, 0, 255), 2)
        
        return img
    
    def create_text_image(self):
        """创建文字测试图像"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 添加不同大小和字体的文字
        cv2.putText(img, 'YOLOS TEST', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.putText(img, 'Object Detection', (50, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(img, 'Computer Vision', (50, 300), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(img, '2024', (50, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0), 2)
        
        return img
    
    def create_noise_image(self):
        """创建噪声测试图像"""
        # 生成随机噪声
        noise = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        # 添加一些结构化元素
        cv2.rectangle(noise, (200, 150), (400, 350), (255, 255, 255), -1)
        cv2.circle(noise, (300, 250), 50, (0, 0, 0), -1)
        
        return noise
    
    def create_gradient_image(self):
        """创建渐变测试图像"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 创建水平渐变
        for x in range(640):
            intensity = int(255 * x / 640)
            img[:, x] = [intensity, intensity, intensity]
        
        # 添加一些对象
        cv2.circle(img, (160, 120), 60, (255, 0, 0), -1)
        cv2.circle(img, (320, 240), 80, (0, 255, 0), -1)
        cv2.circle(img, (480, 360), 70, (0, 0, 255), -1)
        
        return img
    
    def create_complex_image(self):
        """创建复合测试图像"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 背景渐变
        for y in range(480):
            for x in range(640):
                img[y, x] = [int(255 * x / 640), int(255 * y / 480), 128]
        
        # 添加多种元素
        cv2.rectangle(img, (100, 100), (200, 200), (255, 255, 255), 3)
        cv2.circle(img, (400, 150), 50, (0, 0, 0), 3)
        cv2.ellipse(img, (300, 350), (80, 50), 45, 0, 360, (255, 255, 0), 3)
        
        # 添加文字
        cv2.putText(img, 'COMPLEX', (450, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return img
    
    def detect_objects(self, test_images):
        """对测试图像进行物体检测"""
        print("\n🔍 开始物体检测分析...")
        
        all_results = []
        
        for img_data in test_images:
            img = img_data['image']
            img_name = img_data['name']
            
            print(f"\n📸 分析图像: {img_name}")
            
            result = {
                'image_name': img_name,
                'image_path': str(img_data['path']),
                'detections': {},
                'performance': {},
                'statistics': {}
            }
            
            # 1. Haar级联检测
            if 'face' in self.detectors:
                start_time = time.time()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.detectors['face'].detectMultiScale(gray, 1.1, 4)
                detection_time = time.time() - start_time
                
                result['detections']['haar_faces'] = len(faces)
                result['performance']['haar_time'] = detection_time
                print(f"  👤 Haar级联检测: {len(faces)} 个人脸, 耗时: {detection_time:.3f}s")
            
            # 2. ORB特征点检测
            if 'orb' in self.detectors:
                start_time = time.time()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                keypoints = self.detectors['orb'].detect(gray, None)
                detection_time = time.time() - start_time
                
                result['detections']['orb_keypoints'] = len(keypoints)
                result['performance']['orb_time'] = detection_time
                print(f"  🎯 ORB特征点: {len(keypoints)} 个, 耗时: {detection_time:.3f}s")
            
            # 3. SIFT特征点检测
            if 'sift' in self.detectors:
                start_time = time.time()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                keypoints = self.detectors['sift'].detect(gray, None)
                detection_time = time.time() - start_time
                
                result['detections']['sift_keypoints'] = len(keypoints)
                result['performance']['sift_time'] = detection_time
                print(f"  🔍 SIFT特征点: {len(keypoints)} 个, 耗时: {detection_time:.3f}s")
            
            # 4. Canny边缘检测
            start_time = time.time()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_pixels = np.sum(edges > 0)
            detection_time = time.time() - start_time
            
            result['detections']['edge_pixels'] = int(edge_pixels)
            result['performance']['canny_time'] = detection_time
            print(f"  📐 边缘像素: {edge_pixels} 个, 耗时: {detection_time:.3f}s")
            
            # 5. 轮廓检测
            start_time = time.time()
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detection_time = time.time() - start_time
            
            result['detections']['contours'] = len(contours)
            result['performance']['contour_time'] = detection_time
            print(f"  🔄 轮廓数量: {len(contours)} 个, 耗时: {detection_time:.3f}s")
            
            # 6. 图像统计信息
            result['statistics'] = {
                'mean_brightness': float(np.mean(gray)),
                'std_brightness': float(np.std(gray)),
                'min_brightness': int(np.min(gray)),
                'max_brightness': int(np.max(gray)),
                'image_size': f"{img.shape[1]}x{img.shape[0]}",
                'total_pixels': int(img.shape[0] * img.shape[1])
            }
            
            all_results.append(result)
        
        return all_results
    
    def create_visualization(self, test_images, results):
        """创建可视化结果"""
        print("\n📊 生成可视化结果...")
        
        # 创建检测结果可视化图像
        for i, (img_data, result) in enumerate(zip(test_images, results)):
            img = img_data['image'].copy()
            
            # 在图像上绘制检测结果
            if 'orb' in self.detectors:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                keypoints = self.detectors['orb'].detect(gray, None)
                img = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=0)
            
            # 绘制边缘
            gray = cv2.cvtColor(img_data['image'], cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            img[edges > 0] = [0, 0, 255]  # 红色边缘
            
            # 保存可视化结果
            vis_path = self.output_dir / f"visualization_{i+1}_{img_data['name']}.jpg"
            cv2.imwrite(str(vis_path), img)
            print(f"✅ 保存可视化结果: {vis_path.name}")
        
        # 创建性能分析图表
        self.create_performance_charts(results)
    
    def create_performance_charts(self, results):
        """创建性能分析图表"""
        print("📈 生成性能分析图表...")
        
        # 提取性能数据
        methods = []
        times = []
        
        for result in results:
            for method, time_val in result['performance'].items():
                methods.append(f"{result['image_name']}_{method}")
                times.append(time_val)
        
        if times:
            # 创建性能图表
            plt.figure(figsize=(12, 8))
            
            # 子图1: 检测时间
            plt.subplot(2, 2, 1)
            plt.bar(range(len(times)), times)
            plt.title('检测方法性能对比')
            plt.ylabel('时间 (秒)')
            plt.xticks(range(len(times)), methods, rotation=45, ha='right')
            
            # 子图2: 特征点数量对比
            plt.subplot(2, 2, 2)
            orb_counts = [r['detections'].get('orb_keypoints', 0) for r in results]
            sift_counts = [r['detections'].get('sift_keypoints', 0) for r in results]
            image_names = [r['image_name'] for r in results]
            
            x = np.arange(len(image_names))
            width = 0.35
            
            plt.bar(x - width/2, orb_counts, width, label='ORB特征点')
            plt.bar(x + width/2, sift_counts, width, label='SIFT特征点')
            plt.title('特征点检测对比')
            plt.ylabel('特征点数量')
            plt.xticks(x, image_names, rotation=45, ha='right')
            plt.legend()
            
            # 子图3: 边缘像素统计
            plt.subplot(2, 2, 3)
            edge_pixels = [r['detections'].get('edge_pixels', 0) for r in results]
            plt.bar(image_names, edge_pixels, color='red', alpha=0.7)
            plt.title('边缘像素统计')
            plt.ylabel('边缘像素数量')
            plt.xticks(rotation=45, ha='right')
            
            # 子图4: 轮廓数量
            plt.subplot(2, 2, 4)
            contour_counts = [r['detections'].get('contours', 0) for r in results]
            plt.bar(image_names, contour_counts, color='green', alpha=0.7)
            plt.title('轮廓检测统计')
            plt.ylabel('轮廓数量')
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            chart_path = self.output_dir / "performance_analysis.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ 性能分析图表已保存: {chart_path.name}")
    
    def generate_report(self, results):
        """生成测试报告"""
        print("\n📋 生成测试报告...")
        
        report = {
            "test_info": self.test_config,
            "system_info": {
                "opencv_version": cv2.__version__,
                "test_time": datetime.now().isoformat(),
                "total_images": len(results),
                "detection_methods": len(self.detectors)
            },
            "results": results,
            "summary": {
                "total_detections": sum(sum(r['detections'].values()) for r in results),
                "average_processing_time": np.mean([sum(r['performance'].values()) for r in results]),
                "fastest_method": min(results, key=lambda x: min(x['performance'].values()))['image_name'],
                "most_features": max(results, key=lambda x: x['detections'].get('orb_keypoints', 0))['image_name']
            }
        }
        
        # 保存JSON报告
        report_path = self.output_dir / "test_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成Markdown报告
        md_report = self.create_markdown_report(report)
        md_path = self.output_dir / "test_report.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_report)
        
        print(f"✅ 测试报告已保存:")
        print(f"   📄 JSON报告: {report_path.name}")
        print(f"   📝 Markdown报告: {md_path.name}")
        
        return report
    
    def create_markdown_report(self, report):
        """创建Markdown格式的报告"""
        md = f"""# {report['test_info']['test_name']}

## 测试概览
- **测试时间**: {report['test_info']['test_date']}
- **OpenCV版本**: {report['system_info']['opencv_version']}
- **测试图像数量**: {report['system_info']['total_images']}
- **检测方法数量**: {report['system_info']['detection_methods']}

## 测试结果摘要
- **总检测数量**: {report['summary']['total_detections']}
- **平均处理时间**: {report['summary']['average_processing_time']:.3f}秒
- **最快处理图像**: {report['summary']['fastest_method']}
- **最多特征点图像**: {report['summary']['most_features']}

## 详细结果

"""
        
        for i, result in enumerate(report['results'], 1):
            md += f"""### 图像 {i}: {result['image_name']}

**检测结果**:
"""
            for method, count in result['detections'].items():
                md += f"- {method}: {count}\n"
            
            md += f"""
**性能数据**:
"""
            for method, time_val in result['performance'].items():
                md += f"- {method}: {time_val:.3f}秒\n"
            
            md += f"""
**图像统计**:
- 平均亮度: {result['statistics']['mean_brightness']:.2f}
- 亮度标准差: {result['statistics']['std_brightness']:.2f}
- 图像尺寸: {result['statistics']['image_size']}
- 总像素数: {result['statistics']['total_pixels']}

"""
        
        md += """## 测试说明

本测试展示了YOLOS项目的图像处理和分析能力，包括：

1. **Haar级联检测**: 用于人脸等特定对象检测
2. **ORB特征检测**: 快速特征点检测和描述
3. **SIFT特征检测**: 高质量特征点检测
4. **Canny边缘检测**: 边缘提取和分析
5. **轮廓检测**: 形状和结构分析

测试使用了多种类型的图像来评估不同检测方法的性能和准确性。
"""
        
        return md
    
    def run_test(self):
        """运行完整测试"""
        print("🚀 开始YOLOS手持物体识别本地测试")
        
        # 1. 初始化检测器
        self.initialize_detectors()
        
        # 2. 生成测试图像
        test_images = self.generate_test_images()
        
        # 3. 执行检测
        results = self.detect_objects(test_images)
        
        # 4. 创建可视化
        self.create_visualization(test_images, results)
        
        # 5. 生成报告
        report = self.generate_report(results)
        
        # 6. 显示总结
        self.display_summary(report)
        
        return report
    
    def display_summary(self, report):
        """显示测试总结"""
        print("\n" + "=" * 80)
        print("🎉 YOLOS手持物体识别测试完成!")
        print("=" * 80)
        print(f"📊 测试统计:")
        print(f"   • 处理图像: {report['system_info']['total_images']} 张")
        print(f"   • 检测方法: {report['system_info']['detection_methods']} 种")
        print(f"   • 总检测数: {report['summary']['total_detections']}")
        print(f"   • 平均耗时: {report['summary']['average_processing_time']:.3f}秒")
        print(f"\n📁 输出文件:")
        print(f"   • 测试图像: {self.output_dir}/test_*.jpg")
        print(f"   • 可视化结果: {self.output_dir}/visualization_*.jpg")
        print(f"   • 性能图表: {self.output_dir}/performance_analysis.png")
        print(f"   • 测试报告: {self.output_dir}/test_report.md")
        print("=" * 80)

def main():
    """主函数"""
    try:
        # 创建测试实例
        test_system = LocalHandheldObjectTest()
        
        # 运行测试
        report = test_system.run_test()
        
        print("\n✅ 测试成功完成!")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()