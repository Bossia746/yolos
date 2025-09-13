#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K210平台支持测试脚本
测试AIoT开发板适配器中的K210平台检测和配置功能
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.plugins.platform.aiot_boards_adapter import AIoTBoardsAdapter
except ImportError:
    # 尝试直接导入
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'plugins', 'platform'))
    from aiot_boards_adapter import AIoTBoardsAdapter

import json

def test_k210_detection():
    """测试K210检测功能"""
    print("=== K210平台检测测试 ===")
    
    adapter = AIoTBoardsAdapter()
    
    # 测试获取开发板信息
    board_info = adapter.get_board_info()
    supported_boards = board_info['supported_boards']
    print(f"支持的开发板数量: {len(supported_boards)}")
    
    # 查找K210配置
    k210_found = False
    for board_name in supported_boards:
        if 'k210' in board_name.lower():
            k210_found = True
            print(f"找到K210配置: {board_name}")
            # 获取K210的详细配置
            k210_config = adapter.supported_boards.get(board_name, {})
            if k210_config and 'info' in k210_config:
                print(f"  - CPU: {k210_config['info']['cpu']}")
                print(f"  - 内存: {k210_config['info']['memory']}")
                print(f"  - AI加速器: {k210_config['info']['ai_accelerator']}")
                print(f"  - 检测方法: {k210_config['detection_method']}")
                print(f"  - 功能: {list(k210_config['capabilities'].keys())}")
            else:
                print(f"  - 配置详情: {k210_config}")
            break
    
    if not k210_found:
        print("❌ 未找到K210配置")
        return False
    else:
        print("✅ K210配置已正确添加")
        return True

def test_k210_configuration():
    """测试K210配置生成"""
    print("\n=== K210配置生成测试 ===")
    
    adapter = AIoTBoardsAdapter()
    
    # 模拟K210检测结果
    adapter.current_board = {
        'name': 'Kendryte K210',
        'confidence': 0.9,
        'info': {
            'cpu': 'RISC-V Dual Core 400MHz',
            'memory': '8MB SRAM',
            'ai_accelerator': 'KPU 0.25 TOPS',
            'os_support': ['FreeRTOS', 'RT-Thread', 'Bare Metal']
        },
        'capabilities': {
            'kpu': True,
            'camera_dvp': True,
            'uart': True,
            'spi': True,
            'i2c': True,
            'gpio': True,
            'pwm': True
        },
        'detection_method': 'serial_usb'
    }
    
    # 获取配置
    config = adapter._get_board_config()
    
    print("生成的K210配置:")
    print(json.dumps(config, indent=2, ensure_ascii=False))
    
    # 验证关键配置
    checks = [
        ('AI加速器支持', 'kpu' in config['ai_acceleration']['available_accelerators']),
        ('KPU为首选加速器', config['ai_acceleration']['preferred_accelerator'] == 'kpu'),
        ('内存限制正确', config['optimization']['memory_limit_gb'] == 0.5),
        ('单核处理', config['optimization']['max_workers'] == 1),
        ('图像尺寸优化', config['optimization']['image_max_size'] == (224, 224)),
        ('批处理禁用', config['optimization']['batch_size'] == 1),
        ('AI加速器启用', config['optimization']['use_ai_accelerator'] == True)
    ]
    
    all_passed = True
    for check_name, result in checks:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}: {result}")
        if not result:
            all_passed = False
    
    return all_passed

def test_k210_detection_methods():
    """测试K210检测方法"""
    print("\n=== K210检测方法测试 ===")
    
    adapter = AIoTBoardsAdapter()
    
    # 测试串口检测方法
    try:
        # 这些方法在实际硬件不存在时会返回False，但不应该抛出异常
        serial_result = adapter._check_k210_serial()
        print(f"串口检测方法: {serial_result} (预期: False，因为没有实际硬件)")
        
        # 模拟开发板信息用于USB检测
        mock_board_info = {
            'usb_identifiers': [{'vid': '0403', 'pid': '6001'}]
        }
        usb_result = adapter._check_k210_usb(mock_board_info)
        print(f"USB检测方法: {usb_result} (预期: False，因为没有实际硬件)")
        
        print("✅ 检测方法函数正常工作")
        return True
    except Exception as e:
        print(f"❌ 检测方法出错: {e}")
        return False

def main():
    """主测试函数"""
    print("K210平台支持测试开始...\n")
    
    tests = [
        ("K210检测功能", test_k210_detection),
        ("K210配置生成", test_k210_configuration),
        ("K210检测方法", test_k210_detection_methods)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}测试出错: {e}")
            results.append((test_name, False))
    
    print("\n=== 测试结果汇总 ===")
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 项测试通过")
    
    if passed == len(results):
        print("🎉 所有测试通过！K210平台支持已成功实现。")
        return True
    else:
        print("⚠️  部分测试失败，请检查实现。")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)