#!/usr/bin/env python3
"""
YOLOS 基础检测示例
演示如何使用YOLOS进行图像检测
"""

import sys
import os
import cv2
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.yolo_factory import YOLOFactory
from detection.image_detector import ImageDetector


def main():
    print("YOLOS 基础检测示例")
    print("==================")
    
    # 1. 创建检测器
    print("1. 初始化检测器...")
    detector = ImageDetector(
        model_type='yolov8',
        model_path=os.path.join(os.getcwd(), 'module', 'yolov8n.pt'),
        device='auto'
    )
    
    # 2. 检测单张图像
    print("2. 检测图像...")
    
    # 使用摄像头拍照或加载测试图像
    try:
        # 尝试使用摄像头
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("使用摄像头拍照...")
            ret, frame = cap.read()
            if ret:
                cv2.imwrite("test_image.jpg", frame)
                image_path = "test_image.jpg"
            else:
                raise Exception("摄像头拍照失败")
            cap.release()
        else:
            raise Exception("无法打开摄像头")
            
    except Exception as e:
        print(f"摄像头错误: {e}")
        print("请将测试图像放在当前目录并命名为 test_image.jpg")
        image_path = "test_image.jpg"
        
        if not Path(image_path).exists():
            print("未找到测试图像，创建示例图像...")
            # 创建一个简单的测试图像
            import numpy as np
            test_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.rectangle(test_img, (100, 100), (300, 300), (0, 255, 0), 2)
            cv2.putText(test_img, "Test Image", (150, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(image_path, test_img)
    
    # 执行检测
    try:
        results = detector.detect_image(
            image_path=image_path,
            output_path="detected_image.jpg",
            save_results=True
        )
        
        # 3. 显示结果
        print("3. 检测结果:")
        print(f"   检测到 {len(results)} 个目标")
        
        for i, result in enumerate(results):
            print(f"   目标 {i+1}:")
            print(f"     类别: {result['class_name']}")
            print(f"     置信度: {result['confidence']:.2f}")
            print(f"     边界框: {result['bbox']}")
        
        # 4. 显示图像
        if Path("detected_image.jpg").exists():
            print("4. 显示检测结果...")
            img = cv2.imread("detected_image.jpg")
            cv2.imshow("检测结果", img)
            print("按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        print("检测完成!")
        
    except Exception as e:
        print(f"检测失败: {e}")


if __name__ == "__main__":
    main()