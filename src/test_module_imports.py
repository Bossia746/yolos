#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块导入验证脚本 - 在src目录下运行
"""

import sys
import traceback

def test_import(module_name, items=None):
    """测试模块导入"""
    try:
        if items:
            module = __import__(module_name, fromlist=items)
            missing_items = []
            for item in items:
                if not hasattr(module, item):
                    missing_items.append(item)
            
            if missing_items:
                print(f"❌ {module_name} - 缺少属性: {missing_items}")
                return False
            else:
                print(f"✅ {module_name} - 导入成功 ({len(items)} 项)")
        else:
            __import__(module_name)
            print(f"✅ {module_name} - 导入成功")
        return True
    except Exception as e:
        print(f"❌ {module_name} - 导入失败: {e}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=== YOLOS 模块导入验证 (src目录) ===")
    print()
    
    success_count = 0
    total_count = 0
    
    # 测试核心模块
    print("📦 核心模块 (core):")
    core_items = [
        'TaskType', 'ObjectType', 'Status', 'Priority',
        'BoundingBox', 'DetectionResult', 'ProcessingResult',
        'ErrorCode', 'YOLOSException', 'exception_handler',
        'get_logger', 'YOLOSLogger'
    ]
    if test_import('core', core_items):
        success_count += 1
    total_count += 1
    
    # 测试识别模块
    print("\n🤖 识别模块 (recognition):")
    recognition_items = [
        'PoseRecognizer', 'BaseRecognizer',
        'ExerciseType', 'PoseState', 'RecognizerType',
        'PoseRecognizerConfig', 'KeypointConfig'
    ]
    if test_import('recognition', recognition_items):
        success_count += 1
    total_count += 1
    
    # 测试检测模块
    print("\n🔍 检测模块 (detection):")
    detection_items = [
        'DetectorFactory', 'RealtimeDetector',
        'ImageDetector', 'VideoDetector', 'CameraDetector'
    ]
    if test_import('detection', detection_items):
        success_count += 1
    total_count += 1
    
    # 测试模型模块
    print("\n🧠 模型模块 (models):")
    models_items = [
        'YOLOFactory', 'YOLOv5Model', 'YOLOv8Model',
        'YOLOv11Detector', 'UnifiedModelManager'
    ]
    if test_import('models', models_items):
        success_count += 1
    total_count += 1
    
    # 测试工具模块
    print("\n🔧 工具模块 (utils):")
    if test_import('utils', None):
        success_count += 1
    total_count += 1
    
    # 输出结果
    print("\n" + "="*50)
    print(f"导入测试完成: {success_count}/{total_count} 成功")
    
    if success_count == total_count:
        print("🎉 所有模块导入正常!")
        return 0
    else:
        print(f"⚠️  {total_count - success_count} 个模块存在导入问题")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)