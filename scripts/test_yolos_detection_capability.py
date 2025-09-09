#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOS检测能力测试脚本
验证项目是否具备YOLO12级别的多人同框检测和复杂场景识别能力
"""

import os
import sys
import cv2
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Tuple
import requests
import argparse

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    print("错误: ultralytics未安装，请运行: pip install ultralytics")
    sys.exit(1)

from src.models.yolov11_detector import YOLOv11Detector
from src.core.config import YOLOSConfig
from src.utils.logging_manager import LoggingManager


class YOLOSDetectionTester:
    """
    YOLOS检测能力测试器
    
    测试场景:
    1. 多人同框检测
    2. 复杂背景识别
    3. 小目标检测
    4. 遮挡场景处理
    5. 运动模糊处理
    """
    
    def __init__(self):
        """初始化测试器"""
        self.logger = LoggingManager().get_logger("DetectionTester")
        self.device = self._get_device()
        
        # 初始化YOLOS检测器
        self.yolos_detector = YOLOv11Detector(
            model_size='s',
            device=self.device,
            confidence_threshold=0.25,
            iou_threshold=0.45
        )
        
        # 初始化标准YOLO11用于对比
        self.standard_yolo = YOLO('yolo11s.pt')
        
        self.logger.info("🚀 YOLOS检测能力测试器初始化完成")
        self.logger.info(f"📱 使用设备: {self.device}")
    
    def _get_device(self) -> str:
        """获取最优设备"""
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def download_test_images(self, output_dir: str = "./test_images") -> List[str]:
        """下载测试图像"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 测试图像URL列表（多人、复杂场景）
        test_images = {
            "zidane.jpg": "https://ultralytics.com/images/zidane.jpg",
            "bus.jpg": "https://ultralytics.com/images/bus.jpg", 
            "street.jpg": "https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg",
            "crowd.jpg": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=640&h=480&fit=crop",
            "sports.jpg": "https://images.unsplash.com/photo-1551698618-1dfe5d97d256?w=640&h=480&fit=crop"
        }
        
        downloaded_images = []
        
        for filename, url in test_images.items():
            filepath = os.path.join(output_dir, filename)
            
            if os.path.exists(filepath):
                self.logger.info(f"✅ 图像已存在: {filename}")
                downloaded_images.append(filepath)
                continue
            
            try:
                self.logger.info(f"📥 下载图像: {filename}")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                downloaded_images.append(filepath)
                self.logger.info(f"✅ 下载完成: {filename}")
                
            except Exception as e:
                self.logger.warning(f"⚠️ 下载失败 {filename}: {e}")
        
        return downloaded_images
    
    def create_test_scenarios(self, output_dir: str = "./test_scenarios") -> List[str]:
        """创建测试场景图像"""
        os.makedirs(output_dir, exist_ok=True)
        
        scenarios = []
        
        # 场景1: 多人聚集
        multi_person_img = self._create_multi_person_scenario()
        multi_person_path = os.path.join(output_dir, "multi_person_scenario.jpg")
        cv2.imwrite(multi_person_path, multi_person_img)
        scenarios.append(multi_person_path)
        
        # 场景2: 复杂背景
        complex_bg_img = self._create_complex_background_scenario()
        complex_bg_path = os.path.join(output_dir, "complex_background_scenario.jpg")
        cv2.imwrite(complex_bg_path, complex_bg_img)
        scenarios.append(complex_bg_path)
        
        # 场景3: 小目标检测
        small_objects_img = self._create_small_objects_scenario()
        small_objects_path = os.path.join(output_dir, "small_objects_scenario.jpg")
        cv2.imwrite(small_objects_path, small_objects_img)
        scenarios.append(small_objects_path)
        
        self.logger.info(f"✅ 创建了 {len(scenarios)} 个测试场景")
        return scenarios
    
    def _create_multi_person_scenario(self) -> np.ndarray:
        """创建多人场景"""
        # 创建640x640的图像
        img = np.random.randint(50, 200, (640, 640, 3), dtype=np.uint8)
        
        # 添加多个人形轮廓（简化版）
        for i in range(5):
            x = np.random.randint(50, 550)
            y = np.random.randint(50, 550)
            w, h = 80, 160
            
            # 绘制人形轮廓
            cv2.rectangle(img, (x, y), (x+w, y+h), (100, 150, 200), -1)
            cv2.circle(img, (x+w//2, y+20), 15, (200, 180, 160), -1)  # 头部
        
        return img
    
    def _create_complex_background_scenario(self) -> np.ndarray:
        """创建复杂背景场景"""
        # 创建复杂背景
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # 添加噪声和纹理
        noise = np.random.normal(0, 50, (640, 640, 3))
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        # 添加几何形状作为干扰
        for _ in range(20):
            x1, y1 = np.random.randint(0, 640, 2)
            x2, y2 = np.random.randint(0, 640, 2)
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.line(img, (x1, y1), (x2, y2), color, 2)
        
        return img
    
    def _create_small_objects_scenario(self) -> np.ndarray:
        """创建小目标场景"""
        img = np.ones((640, 640, 3), dtype=np.uint8) * 128
        
        # 添加小目标
        for _ in range(10):
            x = np.random.randint(10, 630)
            y = np.random.randint(10, 630)
            size = np.random.randint(5, 20)
            color = tuple(np.random.randint(100, 255, 3).tolist())
            cv2.circle(img, (x, y), size, color, -1)
        
        return img
    
    def test_yolos_detection(self, image_path: str) -> Dict:
        """测试YOLOS检测能力"""
        self.logger.info(f"🔍 测试YOLOS检测: {os.path.basename(image_path)}")
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            self.logger.error(f"❌ 无法读取图像: {image_path}")
            return {}
        
        # YOLOS检测
        start_time = time.time()
        yolos_results = self.yolos_detector.detect(image)
        yolos_time = time.time() - start_time
        
        # 标准YOLO11检测（对比）
        start_time = time.time()
        standard_results = self.standard_yolo(image, verbose=False)
        standard_time = time.time() - start_time
        
        # 分析结果
        yolos_detections = len(yolos_results) if yolos_results else 0
        standard_detections = len(standard_results[0].boxes) if standard_results[0].boxes is not None else 0
        
        result = {
            'image_path': image_path,
            'yolos_detections': yolos_detections,
            'standard_detections': standard_detections,
            'yolos_time': yolos_time,
            'standard_time': standard_time,
            'yolos_results': yolos_results,
            'standard_results': standard_results
        }
        
        self.logger.info(f"📊 YOLOS检测: {yolos_detections}个目标, 耗时: {yolos_time:.3f}s")
        self.logger.info(f"📊 标准YOLO: {standard_detections}个目标, 耗时: {standard_time:.3f}s")
        
        return result
    
    def visualize_detection_results(self, result: Dict, output_dir: str = "./detection_results"):
        """可视化检测结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        image_path = result['image_path']
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 读取原图
        original_image = cv2.imread(image_path)
        if original_image is None:
            return
        
        # YOLOS结果可视化
        yolos_image = original_image.copy()
        if result['yolos_results']:
            yolos_image = self._draw_yolos_detections(yolos_image, result['yolos_results'])
        
        # 标准YOLO结果可视化
        standard_image = original_image.copy()
        if result['standard_results']:
            standard_image = result['standard_results'][0].plot()
        
        # 创建对比图
        comparison_image = self._create_comparison_image(
            original_image, yolos_image, standard_image,
            result['yolos_detections'], result['standard_detections'],
            result['yolos_time'], result['standard_time']
        )
        
        # 保存结果
        output_path = os.path.join(output_dir, f"{image_name}_comparison.jpg")
        cv2.imwrite(output_path, comparison_image)
        
        self.logger.info(f"💾 检测结果已保存: {output_path}")
        
        return output_path
    
    def _draw_yolos_detections(self, image: np.ndarray, detections: List) -> np.ndarray:
        """绘制YOLOS检测结果"""
        for detection in detections:
            if hasattr(detection, 'bbox') and hasattr(detection, 'class_name'):
                x1, y1, x2, y2 = map(int, detection.bbox)
                confidence = getattr(detection, 'confidence', 0.0)
                class_name = detection.class_name
                
                # 绘制边界框
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 绘制标签
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(image, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return image
    
    def _create_comparison_image(self, original: np.ndarray, yolos: np.ndarray, 
                               standard: np.ndarray, yolos_count: int, 
                               standard_count: int, yolos_time: float, 
                               standard_time: float) -> np.ndarray:
        """创建对比图像"""
        h, w = original.shape[:2]
        
        # 创建3列对比图
        comparison = np.zeros((h + 100, w * 3, 3), dtype=np.uint8)
        
        # 放置图像
        comparison[50:h+50, 0:w] = original
        comparison[50:h+50, w:2*w] = yolos
        comparison[50:h+50, 2*w:3*w] = standard
        
        # 添加标题
        cv2.putText(comparison, "Original", (w//2-50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, f"YOLOS ({yolos_count} objs, {yolos_time:.3f}s)", 
                   (w + w//2-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, f"Standard YOLO ({standard_count} objs, {standard_time:.3f}s)", 
                   (2*w + w//2-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 添加性能对比
        performance_text = f"YOLOS vs Standard: Detection={yolos_count}vs{standard_count}, Speed={yolos_time:.3f}vs{standard_time:.3f}s"
        cv2.putText(comparison, performance_text, (10, h+80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return comparison
    
    def run_comprehensive_test(self, test_images: List[str] = None) -> Dict:
        """运行综合检测测试"""
        self.logger.info("🚀 开始YOLOS检测能力综合测试...")
        
        # 如果没有提供测试图像，下载默认图像
        if not test_images:
            test_images = self.download_test_images()
            test_images.extend(self.create_test_scenarios())
        
        results = []
        total_yolos_detections = 0
        total_standard_detections = 0
        total_yolos_time = 0
        total_standard_time = 0
        
        for image_path in test_images:
            if not os.path.exists(image_path):
                self.logger.warning(f"⚠️ 图像不存在: {image_path}")
                continue
            
            # 测试检测
            result = self.test_yolos_detection(image_path)
            if result:
                results.append(result)
                
                # 可视化结果
                self.visualize_detection_results(result)
                
                # 累计统计
                total_yolos_detections += result['yolos_detections']
                total_standard_detections += result['standard_detections']
                total_yolos_time += result['yolos_time']
                total_standard_time += result['standard_time']
        
        # 生成测试报告
        test_summary = {
            'total_images': len(results),
            'total_yolos_detections': total_yolos_detections,
            'total_standard_detections': total_standard_detections,
            'avg_yolos_time': total_yolos_time / len(results) if results else 0,
            'avg_standard_time': total_standard_time / len(results) if results else 0,
            'yolos_fps': len(results) / total_yolos_time if total_yolos_time > 0 else 0,
            'standard_fps': len(results) / total_standard_time if total_standard_time > 0 else 0,
            'results': results
        }
        
        self._generate_test_report(test_summary)
        
        return test_summary
    
    def _generate_test_report(self, summary: Dict):
        """生成测试报告"""
        report_path = "./YOLOS_Detection_Capability_Report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# YOLOS检测能力测试报告\n\n")
            f.write(f"## 测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 测试概览\n\n")
            f.write(f"- **测试图像数量**: {summary['total_images']}\n")
            f.write(f"- **YOLOS总检测数**: {summary['total_yolos_detections']}\n")
            f.write(f"- **标准YOLO总检测数**: {summary['total_standard_detections']}\n")
            f.write(f"- **YOLOS平均耗时**: {summary['avg_yolos_time']:.3f}s\n")
            f.write(f"- **标准YOLO平均耗时**: {summary['avg_standard_time']:.3f}s\n")
            f.write(f"- **YOLOS平均FPS**: {summary['yolos_fps']:.1f}\n")
            f.write(f"- **标准YOLO平均FPS**: {summary['standard_fps']:.1f}\n\n")
            
            f.write("## 性能对比\n\n")
            f.write("| 指标 | YOLOS | 标准YOLO11 | 对比 |\n")
            f.write("|------|-------|------------|------|\n")
            
            detection_ratio = summary['total_yolos_detections'] / summary['total_standard_detections'] if summary['total_standard_detections'] > 0 else 0
            speed_ratio = summary['standard_fps'] / summary['yolos_fps'] if summary['yolos_fps'] > 0 else 0
            
            f.write(f"| 检测数量 | {summary['total_yolos_detections']} | {summary['total_standard_detections']} | {detection_ratio:.2f}x |\n")
            f.write(f"| 平均FPS | {summary['yolos_fps']:.1f} | {summary['standard_fps']:.1f} | {speed_ratio:.2f}x |\n")
            f.write(f"| 平均耗时 | {summary['avg_yolos_time']:.3f}s | {summary['avg_standard_time']:.3f}s | - |\n\n")
            
            f.write("## 详细结果\n\n")
            f.write("| 图像 | YOLOS检测数 | 标准YOLO检测数 | YOLOS耗时 | 标准YOLO耗时 |\n")
            f.write("|------|-------------|----------------|-----------|---------------|\n")
            
            for result in summary['results']:
                image_name = os.path.basename(result['image_path'])
                f.write(f"| {image_name} | {result['yolos_detections']} | {result['standard_detections']} | {result['yolos_time']:.3f}s | {result['standard_time']:.3f}s |\n")
            
            f.write("\n## 结论\n\n")
            
            if detection_ratio >= 0.9:
                f.write("✅ **YOLOS检测能力优秀**: 检测数量与标准YOLO11相当或更好\n")
            elif detection_ratio >= 0.7:
                f.write("⚠️ **YOLOS检测能力良好**: 检测数量略低于标准YOLO11，但在可接受范围内\n")
            else:
                f.write("❌ **YOLOS检测能力需要改进**: 检测数量明显低于标准YOLO11\n")
            
            if speed_ratio <= 1.2:
                f.write("✅ **YOLOS速度性能优秀**: 速度与标准YOLO11相当或更快\n")
            elif speed_ratio <= 2.0:
                f.write("⚠️ **YOLOS速度性能良好**: 速度略慢于标准YOLO11，但在可接受范围内\n")
            else:
                f.write("❌ **YOLOS速度性能需要优化**: 速度明显慢于标准YOLO11\n")
        
        self.logger.info(f"📊 测试报告已生成: {report_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="YOLOS检测能力测试工具")
    parser.add_argument('--images', nargs='+', help='测试图像路径列表')
    parser.add_argument('--download', action='store_true', help='下载测试图像')
    parser.add_argument('--scenarios', action='store_true', help='创建测试场景')
    parser.add_argument('--output', type=str, default='./detection_results', help='输出目录')
    
    args = parser.parse_args()
    
    # 初始化测试器
    tester = YOLOSDetectionTester()
    
    print("🔍 YOLOS检测能力测试工具")
    print("=" * 50)
    print("📋 测试项目:")
    print("  - 多人同框检测")
    print("  - 复杂背景识别") 
    print("  - 小目标检测")
    print("  - 性能对比分析")
    print("=" * 50)
    
    # 准备测试图像
    test_images = []
    
    if args.images:
        test_images.extend(args.images)
    
    if args.download:
        print("📥 下载测试图像...")
        downloaded = tester.download_test_images()
        test_images.extend(downloaded)
    
    if args.scenarios:
        print("🎭 创建测试场景...")
        scenarios = tester.create_test_scenarios()
        test_images.extend(scenarios)
    
    if not test_images:
        print("📥 使用默认测试图像...")
        test_images = tester.download_test_images()
        test_images.extend(tester.create_test_scenarios())
    
    # 运行测试
    print(f"\n🚀 开始测试 {len(test_images)} 张图像...")
    summary = tester.run_comprehensive_test(test_images)
    
    # 显示结果
    print("\n📊 测试结果:")
    print(f"  - 测试图像: {summary['total_images']}")
    print(f"  - YOLOS检测: {summary['total_yolos_detections']} 个目标")
    print(f"  - 标准YOLO: {summary['total_standard_detections']} 个目标")
    print(f"  - YOLOS FPS: {summary['yolos_fps']:.1f}")
    print(f"  - 标准YOLO FPS: {summary['standard_fps']:.1f}")
    
    detection_ratio = summary['total_yolos_detections'] / summary['total_standard_detections'] if summary['total_standard_detections'] > 0 else 0
    
    print(f"\n🎯 检测能力评估:")
    if detection_ratio >= 0.9:
        print("  ✅ YOLOS具备优秀的多人同框和复杂场景检测能力!")
    elif detection_ratio >= 0.7:
        print("  ⚠️ YOLOS具备良好的检测能力，略低于标准YOLO11")
    else:
        print("  ❌ YOLOS检测能力需要进一步优化")
    
    print(f"\n📁 详细报告: ./YOLOS_Detection_Capability_Report.md")
    print(f"📁 可视化结果: {args.output}/")


if __name__ == "__main__":
    main()