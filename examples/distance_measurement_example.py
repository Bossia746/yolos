#!/usr/bin/env python3
"""距离测量功能示例代码

本示例展示了如何使用YOLOS距离测量功能的各种用法：
1. 基础距离测量
2. 实时距离测量
3. 相机标定
4. 批量图像处理
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path
import time

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from recognition.distance_estimator import DistanceEstimator, RealTimeDistanceEstimator
from recognition.enhanced_object_detector import EnhancedObjectDetector
from recognition.camera_calibration_tool import CameraCalibrationTool


def example_1_basic_distance_measurement():
    """示例1: 基础距离测量"""
    print("\n=== 示例1: 基础距离测量 ===")
    
    # 创建距离估算器
    estimator = DistanceEstimator()
    
    # 设置焦距（这里使用示例值，实际使用时需要标定）
    estimator.focal_length = 500.0
    print(f"使用焦距: {estimator.focal_length}")
    
    # 创建测试图像（模拟A4纸）
    test_image = create_test_image_with_a4_paper()
    
    # 测量A4纸的距离
    known_width = 21.0  # A4纸宽度 (cm)
    result = estimator.estimate_distance(test_image, known_width)
    
    if result:
        print(f"✅ 检测成功!")
        print(f"   距离: {result['distance']:.1f} cm")
        print(f"   像素宽度: {result['pixel_width']:.1f} pixels")
        print(f"   置信度: {result['confidence']:.2f}")
        print(f"   边界框: {result['bbox']}")
        
        # 保存结果图像
        result_image = estimator.draw_results(test_image, [result])
        cv2.imwrite('example_1_result.jpg', result_image)
        print(f"   结果图像已保存: example_1_result.jpg")
    else:
        print("❌ 未检测到目标物体")


def example_2_real_time_measurement():
    """示例2: 实时距离测量"""
    print("\n=== 示例2: 实时距离测量 ===")
    print("按 'q' 键退出，按 's' 键截图")
    
    # 创建实时估算器
    real_time_estimator = RealTimeDistanceEstimator()
    real_time_estimator.focal_length = 500.0
    
    # 尝试打开摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ 无法打开摄像头，使用测试图像代替")
        # 使用测试图像模拟实时测量
        test_image = create_test_image_with_a4_paper()
        result_frame = real_time_estimator.process_frame(test_image, known_width=21.0)
        cv2.imshow('Real-time Distance Measurement (Test Image)', result_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    
    print("✅ 摄像头已打开，开始实时测量...")
    
    frame_count = 0
    screenshot_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法读取摄像头帧")
            break
        
        # 实时测距（假设测量A4纸）
        result_frame = real_time_estimator.process_frame(frame, known_width=21.0)
        
        # 添加帮助信息
        cv2.putText(result_frame, "Press 'q' to quit, 's' to screenshot", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示结果
        cv2.imshow('Real-time Distance Measurement', result_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_count += 1
            filename = f'realtime_screenshot_{screenshot_count}.jpg'
            cv2.imwrite(filename, result_frame)
            print(f"📸 截图已保存: {filename}")
        
        frame_count += 1
        if frame_count % 30 == 0:  # 每30帧显示一次统计
            stats = real_time_estimator.get_statistics()
            if stats['total_detections'] > 0:
                print(f"📊 统计: 检测 {stats['total_detections']} 次, "
                      f"平均距离 {stats['average_distance']:.1f} cm")
    
    cap.release()
    cv2.destroyAllWindows()
    print("🏁 实时测量结束")


def example_3_camera_calibration():
    """示例3: 相机标定"""
    print("\n=== 示例3: 相机标定 ===")
    
    # 创建标定工具
    calibration_tool = CameraCalibrationTool()
    
    # 显示支持的物体类型
    known_objects = calibration_tool.known_objects
    print(f"支持的物体类型: {list(known_objects.keys())}")
    
    # 创建标定图像（A4纸在30cm距离处）
    calibration_image = create_calibration_image()
    known_distance = 30.0  # cm
    object_type = 'A4_paper'
    
    print(f"使用 {object_type} 在 {known_distance} cm 距离处进行标定...")
    
    # 执行标定
    focal_length = calibration_tool.calibrate_with_known_object(
        calibration_image, object_type, known_distance
    )
    
    if focal_length:
        print(f"✅ 标定成功!")
        print(f"   计算得到的焦距: {focal_length:.2f}")
        
        # 保存标定结果
        camera_name = 'example_camera'
        calibration_tool.save_calibration(camera_name, focal_length)
        print(f"   标定结果已保存为: {camera_name}")
        
        # 验证标定
        verification_result = calibration_tool.verify_calibration(
            calibration_image, object_type, known_distance, focal_length
        )
        
        if verification_result:
            error_percentage = verification_result['error_percentage']
            print(f"   标定验证: 误差 {error_percentage:.1f}%")
            
            if error_percentage < 10:
                print(f"   ✅ 标定质量: 优秀")
            elif error_percentage < 20:
                print(f"   ⚠️ 标定质量: 良好")
            else:
                print(f"   ❌ 标定质量: 需要重新标定")
    else:
        print("❌ 标定失败，请检查图像质量")
    
    # 显示标定摘要
    summary = calibration_tool.get_calibration_summary()
    print(f"\n📊 标定摘要:")
    print(f"   {summary.get('message', '无标定记录')}")
    if 'calibrations' in summary:
        for name, data in summary['calibrations'].items():
            print(f"   - {name}: 焦距 {data['focal_length']:.2f}")


def example_4_batch_processing():
    """示例4: 批量图像处理"""
    print("\n=== 示例4: 批量图像处理 ===")
    
    # 创建距离估算器
    estimator = DistanceEstimator()
    estimator.focal_length = 500.0
    
    # 创建测试图像集
    test_images = {
        'close_a4.jpg': (create_test_image_with_a4_paper(distance_simulation='close'), 21.0),
        'medium_a4.jpg': (create_test_image_with_a4_paper(distance_simulation='medium'), 21.0),
        'far_a4.jpg': (create_test_image_with_a4_paper(distance_simulation='far'), 21.0),
    }
    
    print(f"处理 {len(test_images)} 张测试图像...")
    
    results = []
    
    for filename, (image, known_width) in test_images.items():
        print(f"\n处理: {filename}")
        
        # 保存测试图像
        cv2.imwrite(filename, image)
        
        # 测量距离
        result = estimator.estimate_distance(image, known_width)
        
        if result:
            distance = result['distance']
            confidence = result['confidence']
            print(f"  ✅ 距离: {distance:.1f} cm, 置信度: {confidence:.2f}")
            
            # 保存结果图像
            result_image = estimator.draw_results(image, [result])
            result_filename = f"result_{filename}"
            cv2.imwrite(result_filename, result_image)
            
            results.append({
                'filename': filename,
                'distance': distance,
                'confidence': confidence,
                'success': True
            })
        else:
            print(f"  ❌ 检测失败")
            results.append({
                'filename': filename,
                'success': False
            })
    
    # 统计结果
    successful = sum(1 for r in results if r['success'])
    print(f"\n📊 批量处理结果:")
    print(f"   成功: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
    
    if successful > 0:
        distances = [r['distance'] for r in results if r['success']]
        avg_distance = sum(distances) / len(distances)
        print(f"   平均距离: {avg_distance:.1f} cm")
        print(f"   距离范围: {min(distances):.1f} - {max(distances):.1f} cm")


def example_5_object_detection_showcase():
    """示例5: 物体检测功能展示"""
    print("\n=== 示例5: 物体检测功能展示 ===")
    
    # 创建检测器
    detector = EnhancedObjectDetector()
    
    # 创建包含多种物体的测试图像
    test_image = create_complex_test_image()
    cv2.imwrite('complex_test_image.jpg', test_image)
    
    print("测试不同的检测方法...")
    
    # 1. 边缘检测
    print("\n1. 边缘检测:")
    rectangles = detector.detect_by_edge(test_image, 'rectangle')
    circles = detector.detect_by_edge(test_image, 'circle')
    print(f"   检测到 {len(rectangles)} 个矩形")
    print(f"   检测到 {len(circles)} 个圆形")
    
    # 2. 颜色检测
    print("\n2. 颜色检测:")
    white_objects = detector.detect_by_color(test_image, 'white')
    red_objects = detector.detect_by_color(test_image, 'red')
    blue_objects = detector.detect_by_color(test_image, 'blue')
    print(f"   检测到 {len(white_objects)} 个白色物体")
    print(f"   检测到 {len(red_objects)} 个红色物体")
    print(f"   检测到 {len(blue_objects)} 个蓝色物体")
    
    # 3. 最大物体检测
    print("\n3. 最大物体检测:")
    largest = detector.detect_largest_object(test_image)
    if largest:
        print(f"   最大物体面积: {largest['area']:.0f} 像素")
        print(f"   边界框: {largest['bbox']}")
    
    # 4. 形状分析
    print("\n4. 形状分析:")
    all_detections = rectangles + circles + white_objects
    for i, detection in enumerate(all_detections[:5]):  # 只显示前5个
        shape_info = detector.analyze_shape(detection['contour'])
        print(f"   物体 {i+1}: {shape_info}")
    
    # 5. 可视化所有检测结果
    print("\n5. 生成可视化结果...")
    
    # 分别可视化不同类型的检测
    edge_result = detector.visualize_detection(test_image.copy(), rectangles + circles, 
                                             color=(0, 255, 0), label_prefix="Edge")
    cv2.imwrite('detection_edge_result.jpg', edge_result)
    
    color_result = detector.visualize_detection(test_image.copy(), white_objects + red_objects, 
                                              color=(255, 0, 0), label_prefix="Color")
    cv2.imwrite('detection_color_result.jpg', color_result)
    
    if largest:
        largest_result = detector.visualize_detection(test_image.copy(), [largest], 
                                                    color=(0, 0, 255), label_prefix="Largest")
        cv2.imwrite('detection_largest_result.jpg', largest_result)
    
    print("   检测结果图像已保存")


def create_test_image_with_a4_paper(distance_simulation='medium'):
    """创建包含A4纸的测试图像"""
    # 创建800x600的图像
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # 添加背景纹理
    cv2.rectangle(image, (0, 0), (800, 600), (40, 40, 40), -1)
    
    # 根据距离模拟调整A4纸大小
    if distance_simulation == 'close':
        paper_width = 420  # 模拟15cm距离
    elif distance_simulation == 'far':
        paper_width = 210  # 模拟60cm距离
    else:  # medium
        paper_width = 350  # 模拟30cm距离
    
    paper_height = int(paper_width * 29.7 / 21)  # 保持A4比例
    
    x = (800 - paper_width) // 2
    y = (600 - paper_height) // 2
    
    # 绘制白色A4纸
    cv2.rectangle(image, (x, y), (x + paper_width, y + paper_height), (240, 240, 240), -1)
    
    # 添加边框
    cv2.rectangle(image, (x, y), (x + paper_width, y + paper_height), (200, 200, 200), 2)
    
    # 添加文字
    cv2.putText(image, "A4 Paper", (x + 20, y + 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
    
    # 添加一些噪声
    noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
    image = cv2.add(image, noise)
    
    return image


def create_calibration_image():
    """创建标定用的图像"""
    return create_test_image_with_a4_paper('medium')


def create_complex_test_image():
    """创建包含多种物体的复杂测试图像"""
    # 创建1000x800的图像
    image = np.zeros((800, 1000, 3), dtype=np.uint8)
    
    # 添加渐变背景
    for y in range(800):
        color_value = int(30 + (y / 800) * 50)
        cv2.line(image, (0, y), (1000, y), (color_value, color_value, color_value), 1)
    
    # 添加白色矩形（A4纸）
    cv2.rectangle(image, (100, 100), (350, 400), (240, 240, 240), -1)
    cv2.rectangle(image, (100, 100), (350, 400), (200, 200, 200), 2)
    
    # 添加红色圆形
    cv2.circle(image, (600, 200), 80, (0, 0, 255), -1)
    cv2.circle(image, (600, 200), 80, (0, 0, 200), 2)
    
    # 添加蓝色矩形
    cv2.rectangle(image, (700, 400), (900, 600), (255, 0, 0), -1)
    cv2.rectangle(image, (700, 400), (900, 600), (200, 0, 0), 2)
    
    # 添加绿色椭圆
    cv2.ellipse(image, (300, 600), (100, 60), 0, 0, 360, (0, 255, 0), -1)
    cv2.ellipse(image, (300, 600), (100, 60), 0, 0, 360, (0, 200, 0), 2)
    
    # 添加一些小的白色圆点
    for i in range(10):
        x = np.random.randint(50, 950)
        y = np.random.randint(50, 750)
        radius = np.random.randint(10, 30)
        cv2.circle(image, (x, y), radius, (255, 255, 255), -1)
    
    return image


def main():
    """主函数 - 运行所有示例"""
    print("🎯 YOLOS 距离测量功能示例")
    print("本示例展示了距离测量功能的各种用法")
    print("=" * 50)
    
    try:
        # 运行所有示例
        example_1_basic_distance_measurement()
        example_3_camera_calibration()
        example_4_batch_processing()
        example_5_object_detection_showcase()
        
        # 询问是否运行实时测量示例
        print("\n" + "=" * 50)
        response = input("是否运行实时测量示例？(需要摄像头) [y/N]: ")
        if response.lower() in ['y', 'yes']:
            example_2_real_time_measurement()
        
        print("\n🎉 所有示例运行完成!")
        print("\n📁 生成的文件:")
        generated_files = [
            'example_1_result.jpg',
            'close_a4.jpg', 'medium_a4.jpg', 'far_a4.jpg',
            'result_close_a4.jpg', 'result_medium_a4.jpg', 'result_far_a4.jpg',
            'complex_test_image.jpg',
            'detection_edge_result.jpg', 'detection_color_result.jpg', 'detection_largest_result.jpg'
        ]
        
        for filename in generated_files:
            if os.path.exists(filename):
                print(f"  ✅ {filename}")
        
        print("\n💡 提示:")
        print("  - 查看生成的图像文件了解检测效果")
        print("  - 运行 GUI 界面进行交互式测试")
        print("  - 阅读使用文档了解更多功能")
        
    except KeyboardInterrupt:
        print("\n⏹️ 示例被用户中断")
    except Exception as e:
        print(f"\n💥 运行示例时发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()