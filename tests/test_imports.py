#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块导入验证脚本
用于测试各模块的导入是否正常工作
"""

import sys
import traceback
from pathlib import Path

# 添加src目录到Python路径
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_import(module_name, items=None):
    """测试模块导入"""
    try:
        if items:
            module = __import__(module_name, fromlist=items)
            for item in items:
                if not hasattr(module, item):
                    print(f"❌ {module_name}.{item} - 属性不存在")
                    return False
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
    print("=== YOLOS 模块导入验证 ===")
    print()
    
    success_count = 0
    total_count = 0
    
    # 测试核心模块
    print("📦 核心模块 (core):")
    core_items = [
        'TaskType', 'ObjectType', 'Status', 'Priority',
        'BoundingBox', 'DetectionResult', 'ProcessingResult',
        'Point2D', 'Keypoint', 'ImageInfo',
        'ErrorCode', 'YOLOSException', 'SystemException', 'ModelException',
        'DataException', 'ConfigurationError', 'exception_handler',
        'get_logger', 'configure_logging', 'YOLOSLogger'
    ]
    if test_import('core', core_items):
        success_count += 1
    total_count += 1
    
    # 测试识别模块
    print("\n🤖 识别模块 (recognition):")
    recognition_items = [
        'PoseRecognizer', 'BaseRecognizer',
        'ExerciseType', 'PoseState', 'RecognizerType',
        'PoseRecognizerConfig', 'KeypointConfig',
        'PoseAnalysisResult'
    ]
    if test_import('recognition', recognition_items):
        success_count += 1
    total_count += 1
    
    # 测试其他模块
    modules_to_test = [
        ('detection', None),
        ('models', None),
        ('utils', None)
    ]
    
    print("\n🔧 其他模块:")
    for module_name, items in modules_to_test:
        if test_import(module_name, items):
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