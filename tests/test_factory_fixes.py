#!/usr/bin/env python3
"""
测试工厂类修复的脚本
验证所有工厂类是否正常工作
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_models_factory():
    """测试 models 工厂类"""
    print("测试 models 工厂类...")
    try:
        from src.models import BaseYOLOModel, YOLOFactory
        
        # 测试 BaseYOLOModel 是否存在
        print(f"✅ BaseYOLOModel 类存在: {BaseYOLOModel}")
        
        # 测试工厂方法
        available_models = YOLOFactory.list_available()
        print(f"✅ 可用模型: {available_models}")
        
        # 测试兼容性方法
        methods = ['list_available', 'get_available', 'list_types', 'get_types']
        for method in methods:
            if hasattr(YOLOFactory, method):
                result = getattr(YOLOFactory, method)()
                print(f"✅ {method}: {len(result)} 个模型")
            else:
                print(f"❌ 缺少方法: {method}")
        
        return True
    except Exception as e:
        print(f"❌ models 工厂测试失败: {e}")
        return False

def test_detection_factory():
    """测试 detection 工厂类"""
    print("\n测试 detection 工厂类...")
    try:
        from src.detection import DetectorFactory
        
        # 测试获取可用检测器
        available_detectors = DetectorFactory.get_available_detectors()
        print(f"✅ 可用检测器: {available_detectors}")
        
        # 测试兼容性方法
        methods = ['list_available', 'get_available', 'list_types', 'get_types']
        for method in methods:
            if hasattr(DetectorFactory, method):
                result = getattr(DetectorFactory, method)()
                print(f"✅ {method}: {len(result)} 个检测器")
            else:
                print(f"❌ 缺少方法: {method}")
        
        # 测试创建跟踪配置
        try:
            tracking_config = DetectorFactory.create_tracking_config('enhanced')
            print(f"✅ 跟踪配置创建成功: {type(tracking_config)}")
        except Exception as e:
            print(f"⚠️  跟踪配置创建警告: {e}")
        
        return True
    except Exception as e:
        print(f"❌ detection 工厂测试失败: {e}")
        return False

def test_recognition_factory():
    """测试 recognition 工厂类"""
    print("\n测试 recognition 工厂类...")
    try:
        from src.recognition.factory import RecognizerFactory
        
        # 测试获取可用识别器
        available_recognizers = RecognizerFactory.get_available_recognizers()
        print(f"✅ 可用识别器: {available_recognizers}")
        
        # 测试兼容性方法
        methods = ['list_available', 'get_available', 'list_types', 'get_types']
        for method in methods:
            if hasattr(RecognizerFactory, method):
                result = getattr(RecognizerFactory, method)()
                print(f"✅ {method}: {len(result)} 个识别器")
            else:
                print(f"❌ 缺少方法: {method}")
        
        return True
    except Exception as e:
        print(f"❌ recognition 工厂测试失败: {e}")
        return False

def test_imports():
    """测试导入问题"""
    print("\n测试导入问题...")
    try:
        # 测试 detection 模块导入
        from src.detection import DetectorFactory
        print("✅ detection 模块导入成功")
        
        # 测试 models 模块导入
        from src.models import BaseYOLOModel
        print("✅ models 模块导入成功")
        
        # 测试 recognition 模块导入
        from src.recognition import RecognizerFactory
        print("✅ recognition 模块导入成功")
        
        return True
    except Exception as e:
        print(f"❌ 导入测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("YOLOS 工厂类修复验证测试")
    print("=" * 60)
    
    results = []
    
    # 运行各项测试
    results.append(test_imports())
    results.append(test_models_factory())
    results.append(test_detection_factory())
    results.append(test_recognition_factory())
    
    # 统计结果
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！工厂类修复成功！")
        return 0
    else:
        print("⚠️  部分测试失败，需要进一步修复")
        return 1

if __name__ == "__main__":
    sys.exit(main())