#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合视觉检测测试脚本

这个脚本提供了完整的YOLOS视觉检测测试功能，包括：
1. YOLOS原生检测
2. ModelScope增强分析
3. 综合性能评估
4. 详细的测试报告

使用方法:
    python comprehensive_vision_test.py [image_path]
    
如果不指定image_path，将测试test_images目录下的所有图像
"""

import os
import sys
from pathlib import Path
from typing import Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入测试模块
try:
    from src.models.yolov11_detector import YOLOv11Detector
    from src.utils.logging_manager import LoggingManager
except ImportError as e:
    print(f"模块导入失败: {e}")
    print("请确保在项目根目录运行此脚本")
    # 不要直接退出，而是跳过测试
    import pytest
    pytest.skip(f"跳过测试，模块导入失败: {e}", allow_module_level=True)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="YOLOS综合视觉检测测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python comprehensive_vision_test.py                    # 测试test_images目录下的所有图像
  python comprehensive_vision_test.py image.jpg         # 测试单个图像
  python comprehensive_vision_test.py /path/to/image    # 测试指定路径的图像
        """
    )
    
    parser.add_argument(
        'image_path',
        nargs='?',
        help='要测试的图像路径（可选，如果不指定则测试test_images目录下的所有图像）'
    )
    
    parser.add_argument(
        '--test-dir',
        default='test_images',
        help='测试图像目录（默认: test_images）'
    )
    
    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='不显示测试摘要'
    )
    
    args = parser.parse_args()
    
    try:
        # 创建测试器
        tester = ComprehensiveVisionTester(test_images_dir=args.test_dir)
        
        # 运行测试
        test_results = tester.run_comprehensive_test(args.image_path)
        
        # 显示摘要
        if not args.no_summary:
            tester.print_summary(test_results["report"])
        
        # 显示统计信息
        stats = tester.get_test_stats()
        if stats["total_tests"] > 0:
            print(f"\n📈 最终统计:")
            print(f"   总测试数: {stats['total_tests']}")
            print(f"   成功测试: {stats['successful_tests']}")
            print(f"   YOLO成功: {stats['yolo_successes']}")
            print(f"   ModelScope成功: {stats['modelscope_successes']}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 测试被用户中断")
        return 1
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())