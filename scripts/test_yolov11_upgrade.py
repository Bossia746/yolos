#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOS YOLOv11升级测试脚本
验证升级后的系统功能
"""

import sys
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_optimized_system():
    """测试优化系统"""
    print("🧪 开始测试YOLOv11优化系统...")
    
    try:
        # 导入优化系统
        from src.models.optimized_yolov11_system import OptimizedYOLOv11System, OptimizationConfig
        
        # 创建配置
        config = OptimizationConfig(
            model_size='s',
            platform='pc',
            target_fps=30.0,
            adaptive_inference=True,
            edge_optimization=False
        )
        
        print("✅ 配置创建成功")
        
        # 创建检测系统
        detector = OptimizedYOLOv11System(config)
        print("✅ 检测系统创建成功")
        
        # 测试图像检测
        import numpy as np
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        start_time = time.time()
        results = detector.detect_adaptive(test_image)
        inference_time = time.time() - start_time
        
        print(f"✅ 图像检测成功")
        print(f"   推理时间: {inference_time*1000:.1f}ms")
        print(f"   检测数量: {len(results)}")
        
        # 获取性能统计
        stats = detector.get_performance_stats()
        print(f"✅ 性能统计获取成功")
        print(f"   当前FPS: {stats.get('current_fps', 0):.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_factory_integration():
    """测试工厂集成"""
    print("\n🏭 测试检测器工厂集成...")
    
    try:
        from src.detection.factory import DetectorFactory
        
        # 测试YOLOv11检测器创建
        config = {
            'model_size': 's',
            'device': 'auto',
            'confidence_threshold': 0.25,
            'platform': 'pc',
            'adaptive_inference': True
        }
        
        detector = DetectorFactory.create_detector('yolov11', config)
        print("✅ YOLOv11检测器创建成功")
        
        # 列出可用检测器
        available = DetectorFactory.list_available_detectors()
        print(f"✅ 可用检测器: {available}")
        
        return True
        
    except Exception as e:
        print(f"❌ 工厂测试失败: {e}")
        return False

def test_configuration():
    """测试配置系统"""
    print("\n⚙️ 测试配置系统...")
    
    try:
        from src.core.config import load_config
        
        # 加载优化配置
        config_path = project_root / "config" / "yolov11_optimized.yaml"
        if config_path.exists():
            config = load_config(str(config_path))
            print("✅ 配置文件加载成功")
            print(f"   检测类型: {config.get('detection', {}).get('type')}")
            print(f"   模型大小: {config.get('detection', {}).get('model_size')}")
        else:
            print("⚠️ 配置文件不存在，使用默认配置")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return False

def performance_comparison():
    """性能对比测试"""
    print("\n📊 执行性能对比测试...")
    
    try:
        import numpy as np
        
        # 生成测试图像
        test_images = [
            np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            for _ in range(10)
        ]
        
        # 测试不同模型大小
        model_sizes = ['n', 's']
        results = {}
        
        for size in model_sizes:
            print(f"   测试YOLOv11{size.upper()}...")
            
            try:
                from src.models.optimized_yolov11_system import OptimizedYOLOv11System, OptimizationConfig
                
                config = OptimizationConfig(
                    model_size=size,
                    platform='pc',
                    adaptive_inference=False
                )
                
                detector = OptimizedYOLOv11System(config)
                
                # 执行测试
                start_time = time.time()
                total_detections = 0
                
                for image in test_images:
                    detections = detector.detect_adaptive(image)
                    total_detections += len(detections)
                
                total_time = time.time() - start_time
                avg_fps = len(test_images) / total_time
                
                results[size] = {
                    'fps': avg_fps,
                    'total_time': total_time,
                    'detections': total_detections
                }
                
                print(f"     FPS: {avg_fps:.1f}")
                print(f"     总检测数: {total_detections}")
                
            except Exception as e:
                print(f"     ❌ 测试失败: {e}")
        
        # 显示对比结果
        if len(results) > 1:
            print("\n📈 性能对比结果:")
            for size, stats in results.items():
                print(f"   YOLOv11{size.upper()}: {stats['fps']:.1f} FPS")
        
        return True
        
    except Exception as e:
        print(f"❌ 性能对比失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 YOLOS YOLOv11升级验证测试")
    print("=" * 50)
    
    tests = [
        ("优化系统", test_optimized_system),
        ("工厂集成", test_factory_integration),
        ("配置系统", test_configuration),
        ("性能对比", performance_comparison)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 测试: {test_name}")
        print("-" * 30)
        
        if test_func():
            passed += 1
            print(f"✅ {test_name} 测试通过")
        else:
            print(f"❌ {test_name} 测试失败")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！YOLOv11升级成功！")
        print("\n🚀 快速开始:")
        print("   python scripts/start_yolov11_optimized.py camera")
        print("   python scripts/start_yolov11_optimized.py benchmark")
    else:
        print("⚠️ 部分测试失败，请检查错误信息")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)