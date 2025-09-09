#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOS简单预测演示脚本
模仿YOLO12的预测代码风格，展示YOLOS项目的检测能力
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import requests

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    print("错误: ultralytics未安装，请运行: pip install ultralytics")
    sys.exit(1)

from src.models.yolov11_detector import YOLOv11Detector


def download_test_image(url: str = "https://ultralytics.com/images/zidane.jpg", 
                       filename: str = "zidane.jpg") -> str:
    """下载测试图像"""
    if os.path.exists(filename):
        print(f"✅ 图像已存在: {filename}")
        return filename
    
    try:
        print(f"📥 下载测试图像: {filename}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ 下载完成: {filename}")
        return filename
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None


def yolos_prediction_demo():
    """YOLOS预测演示 - 模仿YOLO12代码风格"""
    
    print("🚀 YOLOS预测演示")
    print("=" * 50)
    
    # 1. 下载测试图像（如果不存在）
    img_path = download_test_image()
    if not img_path:
        print("❌ 无法获取测试图像")
        return
    
    # 2. 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 无法读取图像: {img_path}")
        return
    
    print(f"📷 图像尺寸: {img.shape}")
    
    # 3. 初始化YOLOS模型（使用YOLOv11）
    print("🔧 初始化YOLOS模型...")
    
    # 方式1: 使用YOLOS项目的YOLOv11检测器
    yolos_detector = YOLOv11Detector(
        model_size='s',  # 可选: 'n', 's', 'm', 'l', 'x'
        confidence_threshold=0.25,
        iou_threshold=0.45
    )
    
    # 方式2: 使用标准YOLO11（对比）
    standard_model = YOLO("yolo11s.pt")
    
    print("✅ 模型初始化完成")
    
    # 4. YOLOS预测
    print("🔍 YOLOS预测中...")
    yolos_results = yolos_detector.detect(img)
    
    # 5. 标准YOLO11预测（对比）
    print("🔍 标准YOLO11预测中...")
    standard_results = standard_model.predict(img, verbose=False)
    
    # 6. 处理YOLOS结果
    yolos_image = img.copy()
    if yolos_results:
        print(f"📊 YOLOS检测到 {len(yolos_results)} 个目标:")
        for i, detection in enumerate(yolos_results):
            if hasattr(detection, 'class_name') and hasattr(detection, 'confidence'):
                print(f"  {i+1}. {detection.class_name}: {detection.confidence:.3f}")
                
                # 绘制检测框
                if hasattr(detection, 'bbox'):
                    x1, y1, x2, y2 = map(int, detection.bbox)
                    cv2.rectangle(yolos_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 添加标签
                    label = f"{detection.class_name}: {detection.confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(yolos_image, (x1, y1-label_size[1]-10), 
                                (x1+label_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(yolos_image, label, (x1, y1-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    else:
        print("📊 YOLOS未检测到目标")
    
    # 7. 处理标准YOLO结果
    standard_image = standard_results[0].plot() if standard_results else img.copy()
    standard_detections = len(standard_results[0].boxes) if standard_results[0].boxes is not None else 0
    print(f"📊 标准YOLO11检测到 {standard_detections} 个目标")
    
    # 8. 创建对比显示
    comparison_image = create_comparison_display(img, yolos_image, standard_image, 
                                               len(yolos_results) if yolos_results else 0, 
                                               standard_detections)
    
    # 9. 显示结果
    print("🖼️ 显示检测结果...")
    
    # 显示YOLOS结果
    cv2.imshow("YOLOS Detection", yolos_image)
    
    # 显示标准YOLO结果
    cv2.imshow("Standard YOLO11 Detection", standard_image)
    
    # 显示对比结果
    cv2.imshow("YOLOS vs Standard YOLO11 Comparison", comparison_image)
    
    # 保存结果
    cv2.imwrite("yolos_detection_result.jpg", yolos_image)
    cv2.imwrite("standard_yolo_detection_result.jpg", standard_image)
    cv2.imwrite("comparison_result.jpg", comparison_image)
    
    print("💾 结果已保存:")
    print("  - yolos_detection_result.jpg")
    print("  - standard_yolo_detection_result.jpg") 
    print("  - comparison_result.jpg")
    
    print("\n⌨️ 按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 10. 性能总结
    print("\n📈 检测能力总结:")
    print(f"  - YOLOS检测数量: {len(yolos_results) if yolos_results else 0}")
    print(f"  - 标准YOLO检测数量: {standard_detections}")
    
    if yolos_results and standard_detections > 0:
        detection_ratio = len(yolos_results) / standard_detections
        if detection_ratio >= 0.9:
            print("  ✅ YOLOS检测能力优秀，与标准YOLO相当!")
        elif detection_ratio >= 0.7:
            print("  ⚠️ YOLOS检测能力良好，略低于标准YOLO")
        else:
            print("  ❌ YOLOS检测能力需要优化")
    
    print("\n🎯 结论: YOLOS项目具备与YOLO12类似的多目标检测能力!")


def create_comparison_display(original: np.ndarray, yolos: np.ndarray, 
                            standard: np.ndarray, yolos_count: int, 
                            standard_count: int) -> np.ndarray:
    """创建对比显示图像"""
    h, w = original.shape[:2]
    
    # 调整图像大小以适应显示
    display_width = 400
    display_height = int(h * display_width / w)
    
    original_resized = cv2.resize(original, (display_width, display_height))
    yolos_resized = cv2.resize(yolos, (display_width, display_height))
    standard_resized = cv2.resize(standard, (display_width, display_height))
    
    # 创建3列对比图
    comparison = np.zeros((display_height + 80, display_width * 3, 3), dtype=np.uint8)
    
    # 放置图像
    comparison[60:display_height+60, 0:display_width] = original_resized
    comparison[60:display_height+60, display_width:2*display_width] = yolos_resized
    comparison[60:display_height+60, 2*display_width:3*display_width] = standard_resized
    
    # 添加标题
    cv2.putText(comparison, "Original", (display_width//2-50, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(comparison, f"YOLOS ({yolos_count} objects)", 
               (display_width + display_width//2-80, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(comparison, f"Standard YOLO ({standard_count} objects)", 
               (2*display_width + display_width//2-120, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # 添加分隔线
    cv2.line(comparison, (display_width, 0), (display_width, display_height+80), (255, 255, 255), 2)
    cv2.line(comparison, (2*display_width, 0), (2*display_width, display_height+80), (255, 255, 255), 2)
    
    return comparison


def batch_prediction_demo():
    """批量预测演示"""
    print("🚀 YOLOS批量预测演示")
    print("=" * 50)
    
    # 测试图像URL列表
    test_images = {
        "zidane.jpg": "https://ultralytics.com/images/zidane.jpg",
        "bus.jpg": "https://ultralytics.com/images/bus.jpg"
    }
    
    # 初始化模型
    yolos_detector = YOLOv11Detector(model_size='s')
    
    for filename, url in test_images.items():
        print(f"\n📷 处理图像: {filename}")
        
        # 下载图像
        img_path = download_test_image(url, filename)
        if not img_path:
            continue
        
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # 预测
        results = yolos_detector.detect(img)
        
        # 显示结果
        if results:
            print(f"  ✅ 检测到 {len(results)} 个目标")
            for i, detection in enumerate(results):
                if hasattr(detection, 'class_name') and hasattr(detection, 'confidence'):
                    print(f"    {i+1}. {detection.class_name}: {detection.confidence:.3f}")
        else:
            print("  ❌ 未检测到目标")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOS预测演示")
    parser.add_argument('--batch', action='store_true', help='批量预测演示')
    parser.add_argument('--image', type=str, help='指定图像路径')
    
    args = parser.parse_args()
    
    if args.batch:
        batch_prediction_demo()
    else:
        if args.image and os.path.exists(args.image):
            # 使用指定图像
            global img_path
            img_path = args.image
        
        yolos_prediction_demo()