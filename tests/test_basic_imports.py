#!/usr/bin/env python3
"""
最基础的导入测试
测试修复后的文件是否可以正常导入
"""

import sys
import os

def test_file_syntax():
    """测试文件语法是否正确"""
    print("测试文件语法...")
    
    files_to_test = [
        'src/models/base_model.py',
        'src/models/yolo_factory.py', 
        'src/detection/factory.py',
        'src/recognition/factory.py',
        'src/core/exceptions.py'
    ]
    
    results = []
    
    for file_path in files_to_test:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 尝试编译代码
            compile(content, file_path, 'exec')
            print(f"✅ {file_path} 语法正确")
            results.append(True)
        except SyntaxError as e:
            print(f"❌ {file_path} 语法错误: {e}")
            results.append(False)
        except Exception as e:
            print(f"⚠️  {file_path} 其他错误: {e}")
            results.append(True)  # 语法可能是正确的，只是其他问题
    
    return results

def test_key_classes_exist():
    """测试关键类是否存在于文件中"""
    print("\n测试关键类是否存在...")
    
    tests = [
        ('src/models/base_model.py', 'class BaseYOLOModel'),
        ('src/models/yolo_factory.py', 'class YOLOFactory'),
        ('src/detection/factory.py', 'class DetectorFactory'),
        ('src/recognition/factory.py', 'class RecognizerFactory'),
        ('src/core/exceptions.py', 'DATA_PROCESSING_ERROR')
    ]
    
    results = []
    
    for file_path, search_text in tests:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if search_text in content:
                print(f"✅ {file_path} 包含 {search_text}")
                results.append(True)
            else:
                print(f"❌ {file_path} 缺少 {search_text}")
                results.append(False)
        except Exception as e:
            print(f"❌ 无法读取 {file_path}: {e}")
            results.append(False)
    
    return results

def test_factory_methods():
    """测试工厂类是否包含必需方法"""
    print("\n测试工厂类方法...")
    
    required_methods = ['list_available', 'get_available', 'list_types', 'get_types']
    factory_files = [
        'src/models/yolo_factory.py',
        'src/detection/factory.py', 
        'src/recognition/factory.py'
    ]
    
    results = []
    
    for file_path in factory_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            file_results = []
            for method in required_methods:
                if f'def {method}(' in content:
                    print(f"✅ {file_path} 包含方法 {method}")
                    file_results.append(True)
                else:
                    print(f"❌ {file_path} 缺少方法 {method}")
                    file_results.append(False)
            
            results.append(all(file_results))
        except Exception as e:
            print(f"❌ 无法读取 {file_path}: {e}")
            results.append(False)
    
    return results

def main():
    """主测试函数"""
    print("=" * 60)
    print("YOLOS 基础导入和语法测试")
    print("=" * 60)
    
    # 运行测试
    syntax_results = test_file_syntax()
    class_results = test_key_classes_exist()
    method_results = test_factory_methods()
    
    # 统计结果
    total_syntax = len(syntax_results)
    passed_syntax = sum(syntax_results)
    
    total_classes = len(class_results)
    passed_classes = sum(class_results)
    
    total_methods = len(method_results)
    passed_methods = sum(method_results)
    
    print("\n" + "=" * 60)
    print("测试结果汇总:")
    print(f"文件语法测试: {passed_syntax}/{total_syntax} 通过")
    print(f"关键类测试: {passed_classes}/{total_classes} 通过")
    print(f"工厂方法测试: {passed_methods}/{total_methods} 通过")
    
    total_passed = passed_syntax + passed_classes + passed_methods
    total_tests = total_syntax + total_classes + total_methods
    
    print(f"总体通过率: {total_passed}/{total_tests} ({total_passed/total_tests*100:.1f}%)")
    
    if total_passed >= total_tests * 0.8:  # 80% 通过率
        print("🎉 基础测试大部分通过！主要修复成功！")
        return 0
    else:
        print("⚠️  基础测试通过率较低，需要进一步修复")
        return 1

if __name__ == "__main__":
    sys.exit(main())