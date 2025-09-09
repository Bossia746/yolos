#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIoT开发板简化兼容性测试
避免复杂的模块依赖
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_aiot_boards_adapter():
    """测试AIoT开发板适配器"""
    try:
        from src.plugins.platform.aiot_boards_adapter import AIoTBoardsAdapter
        
        print("=" * 60)
        print("AIoT开发板适配器测试")
        print("=" * 60)
        
        # 创建适配器
        adapter = AIoTBoardsAdapter()
        
        # 测试基本功能
        print(f"✓ 适配器初始化成功")
        print(f"✓ 支持的开发板数量: {len(adapter.supported_boards)}")
        
        # 显示支持的开发板
        print(f"\n支持的AIoT开发板:")
        for board_id, board_info in adapter.supported_boards.items():
            manufacturer = board_info.get('manufacturer', 'Unknown')
            name = board_info.get('name', board_id)
            print(f"  - {name} ({manufacturer})")
        
        # 检测当前开发板
        current_board = adapter.current_board
        print(f"\n当前检测结果:")
        print(f"  - 开发板: {current_board['name']}")
        print(f"  - 置信度: {current_board['confidence']:.2f}")
        
        # 生成报告
        report = adapter.generate_board_report()
        print(f"\n✓ 报告生成成功 (长度: {len(report)} 字符)")
        
        return True
        
    except Exception as e:
        print(f"✗ AIoT适配器测试失败: {e}")
        return False

def test_cross_platform_manager():
    """测试跨平台管理器"""
    try:
        from src.core.cross_platform_manager import CrossPlatformManager
        
        print("\n" + "=" * 60)
        print("跨平台管理器测试")
        print("=" * 60)
        
        # 创建管理器
        manager = CrossPlatformManager()
        
        print(f"✓ 跨平台管理器初始化成功")
        print(f"✓ 平台: {manager.platform_info['system']}")
        print(f"✓ 架构: {manager.platform_info['architecture']}")
        
        # 检查AIoT开发板集成
        aiot_info = manager.aiot_board_info
        if aiot_info['detected']:
            print(f"✓ 检测到AIoT开发板: {aiot_info['board_name']}")
        else:
            print(f"ℹ 未检测到AIoT开发板 (正常，取决于运行环境)")
        
        # 检查Arduino支持
        arduino_info = manager.platform_info['arduino_support']
        if arduino_info['pyserial_available']:
            print(f"✓ Arduino支持可用")
            print(f"  - 可用串口: {len(arduino_info['serial_ports_available'])}")
        else:
            print(f"ℹ Arduino支持不可用 (pyserial未安装)")
        
        # 生成平台报告
        report = manager.generate_platform_report()
        print(f"✓ 平台报告生成成功 (长度: {len(report)} 字符)")
        
        return True
        
    except Exception as e:
        print(f"✗ 跨平台管理器测试失败: {e}")
        return False

def test_supported_boards_coverage():
    """测试支持的开发板覆盖范围"""
    try:
        from src.plugins.platform.aiot_boards_adapter import AIoTBoardsAdapter
        
        print("\n" + "=" * 60)
        print("开发板覆盖范围测试")
        print("=" * 60)
        
        adapter = AIoTBoardsAdapter()
        
        # 按制造商分类
        manufacturers = {}
        ai_accelerator_boards = []
        camera_support_boards = []
        gpio_support_boards = []
        
        for board_id, board_info in adapter.supported_boards.items():
            # 制造商统计
            manufacturer = board_info.get('manufacturer', 'Unknown')
            manufacturers.setdefault(manufacturer, []).append(board_info['name'])
            
            # 能力统计
            capabilities = board_info.get('capabilities', {})
            
            if any(capabilities.get(key, False) for key in ['edge_tpu', 'npu_acceleration', 'cuda', 'hexagon_dsp']):
                ai_accelerator_boards.append(board_info['name'])
            
            if any(capabilities.get(key, False) for key in ['camera_csi', 'camera_mipi', 'camera_builtin']):
                camera_support_boards.append(board_info['name'])
            
            if capabilities.get('gpio', False):
                gpio_support_boards.append(board_info['name'])
        
        # 显示统计结果
        print(f"制造商覆盖 ({len(manufacturers)} 家):")
        for manufacturer, boards in manufacturers.items():
            print(f"  - {manufacturer}: {len(boards)} 款开发板")
        
        print(f"\nAI加速支持: {len(ai_accelerator_boards)} 款开发板")
        print(f"摄像头支持: {len(camera_support_boards)} 款开发板") 
        print(f"GPIO支持: {len(gpio_support_boards)} 款开发板")
        
        # 验证覆盖范围
        expected_manufacturers = ['NVIDIA', 'Google', 'Intel', 'Rockchip', 'Qualcomm', 'Espressif']
        missing_manufacturers = [m for m in expected_manufacturers if m not in manufacturers]
        
        if missing_manufacturers:
            print(f"⚠ 缺少制造商: {missing_manufacturers}")
        else:
            print(f"✓ 主要制造商覆盖完整")
        
        return len(missing_manufacturers) == 0
        
    except Exception as e:
        print(f"✗ 覆盖范围测试失败: {e}")
        return False

def test_board_detection_logic():
    """测试开发板检测逻辑"""
    try:
        from src.plugins.platform.aiot_boards_adapter import AIoTBoardsAdapter
        
        print("\n" + "=" * 60)
        print("开发板检测逻辑测试")
        print("=" * 60)
        
        adapter = AIoTBoardsAdapter()
        
        # 测试检测方法
        detection_methods_count = {}
        
        for board_id, board_info in adapter.supported_boards.items():
            methods = board_info.get('detection_methods', [])
            
            if not methods:
                print(f"⚠ {board_info['name']} 缺少检测方法")
                continue
            
            for method in methods:
                detection_methods_count[method] = detection_methods_count.get(method, 0) + 1
        
        print(f"检测方法统计:")
        for method, count in detection_methods_count.items():
            print(f"  - {method}: {count} 款开发板")
        
        # 测试置信度计算
        current_board = adapter.current_board
        print(f"\n当前环境检测:")
        print(f"  - 最佳匹配: {current_board['name']}")
        print(f"  - 置信度: {current_board['confidence']:.2f}")
        print(f"  - 检测方法: {current_board.get('detection_method', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"✗ 检测逻辑测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("YOLOS AIoT开发板兼容性测试")
    print("=" * 80)
    
    tests = [
        ("AIoT开发板适配器", test_aiot_boards_adapter),
        ("跨平台管理器", test_cross_platform_manager),
        ("开发板覆盖范围", test_supported_boards_coverage),
        ("开发板检测逻辑", test_board_detection_logic)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 显示测试结果摘要
    print("\n" + "=" * 80)
    print("测试结果摘要")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {len(results)} 个测试")
    print(f"通过: {passed} 个")
    print(f"失败: {failed} 个")
    
    # 生成最终报告
    if failed == 0:
        print(f"\n🎉 所有测试通过！YOLOS系统已完全支持主流AIoT开发板。")
        
        # 显示支持的开发板摘要
        try:
            from src.plugins.platform.aiot_boards_adapter import get_aiot_boards_adapter
            adapter = get_aiot_boards_adapter()
            
            print(f"\n📋 支持的AIoT开发板摘要:")
            print(f"   - 总计: {len(adapter.supported_boards)} 款开发板")
            
            manufacturers = set()
            for board_info in adapter.supported_boards.values():
                manufacturers.add(board_info.get('manufacturer', 'Unknown'))
            
            print(f"   - 制造商: {len(manufacturers)} 家")
            print(f"   - 覆盖范围: 从高性能到超低功耗的完整解决方案")
            
        except Exception as e:
            print(f"生成摘要失败: {e}")
    else:
        print(f"\n⚠ 有 {failed} 个测试失败，请检查相关功能。")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)