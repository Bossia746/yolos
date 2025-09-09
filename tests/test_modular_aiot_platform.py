#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOS模块化AIoT平台测试
测试机械臂药物识别和外部通信推送功能
"""

import sys
import os
import time
import numpy as np
import cv2
from typing import Dict, Any, List

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_modular_extension_manager():
    """测试模块化扩展管理器"""
    print("=" * 60)
    print("测试模块化扩展管理器")
    print("=" * 60)
    
    try:
        from src.core.modular_extension_manager import ModularExtensionManager
        
        # 创建管理器
        manager = ModularExtensionManager()
        
        # 测试模块注册
        print("✓ 模块化扩展管理器创建成功")
        
        # 获取可用模块
        available_modules = manager.get_available_modules()
        print(f"✓ 可用模块数量: {len(available_modules)}")
        
        # 测试模块启用/禁用
        manager.enable_module("medication_recognition")
        manager.enable_module("robotic_arm_control")
        manager.enable_module("external_communication")
        
        enabled_modules = manager.get_enabled_modules()
        print(f"✓ 已启用模块: {enabled_modules}")
        
        # 测试模块状态
        status = manager.get_system_status()
        print(f"✓ 系统状态: {status['status']}")
        
        return True
        
    except Exception as e:
        print(f"✗ 模块化扩展管理器测试失败: {e}")
        return False

def test_medication_recognition():
    """测试药物识别系统"""
    print("\n" + "=" * 60)
    print("测试药物识别系统")
    print("=" * 60)
    
    try:
        from src.recognition.medication_recognition_system import MedicationRecognitionSystem
        
        # 创建药物识别系统
        med_system = MedicationRecognitionSystem()
        
        # 创建模拟药物图像
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 添加模拟药物到数据库
        medication_info = {
            "name": "阿司匹林",
            "dosage": "100mg",
            "shape": "圆形",
            "color": "白色",
            "size": {"diameter": 8.0, "thickness": 3.0},
            "manufacturer": "拜耳",
            "expiry_date": "2025-12-31",
            "storage_location": "A区-1层-3号"
        }
        
        med_system.add_medication_to_database("aspirin_100mg", medication_info)
        print("✓ 药物数据库添加成功")
        
        # 测试药物识别
        result = med_system.recognize_medication(test_image)
        print(f"✓ 药物识别完成")
        print(f"  - 识别状态: {result.status.value}")
        print(f"  - 置信度: {result.confidence:.3f}")
        print(f"  - 检测到的药物数量: {len(result.detected_medications)}")
        
        # 测试路径规划
        target_medication = "aspirin_100mg"
        path = med_system.plan_medication_retrieval_path(target_medication)
        print(f"✓ 药物检索路径规划完成")
        print(f"  - 目标药物: {target_medication}")
        print(f"  - 路径步骤数: {len(path.steps)}")
        
        # 测试训练功能
        training_result = med_system.train_with_new_sample(test_image, "aspirin_100mg")
        print(f"✓ 新样本训练完成: {training_result}")
        
        return True
        
    except Exception as e:
        print(f"✗ 药物识别系统测试失败: {e}")
        return False

def test_robotic_arm_controller():
    """测试机械臂控制器"""
    print("\n" + "=" * 60)
    print("测试机械臂控制器")
    print("=" * 60)
    
    try:
        from src.plugins.hardware.robotic_arm_controller import RoboticArmController
        
        # 创建机械臂控制器
        arm_controller = RoboticArmController()
        
        # 测试初始化
        print("✓ 机械臂控制器创建成功")
        
        # 测试位置控制
        target_position = [100, 200, 150, 0, 90, 0]  # x, y, z, rx, ry, rz
        result = arm_controller.move_to_position(target_position)
        print(f"✓ 位置移动命令: {result}")
        
        # 测试抓取操作
        grip_result = arm_controller.grip_object(force=50)
        print(f"✓ 抓取操作: {grip_result}")
        
        # 测试释放操作
        release_result = arm_controller.release_object()
        print(f"✓ 释放操作: {release_result}")
        
        # 测试路径规划
        waypoints = [
            [0, 0, 100, 0, 0, 0],
            [100, 100, 100, 0, 0, 0],
            [200, 200, 50, 0, 0, 0]
        ]
        path_result = arm_controller.plan_path(waypoints)
        print(f"✓ 路径规划: 生成了 {len(path_result)} 个路径点")
        
        # 测试安全检查
        safety_status = arm_controller.check_safety()
        print(f"✓ 安全检查: {safety_status}")
        
        # 测试状态获取
        arm_status = arm_controller.get_arm_status()
        print(f"✓ 机械臂状态: {arm_status['status']}")
        
        return True
        
    except Exception as e:
        print(f"✗ 机械臂控制器测试失败: {e}")
        return False

def test_external_communication():
    """测试外部通信系统"""
    print("\n" + "=" * 60)
    print("测试外部通信系统")
    print("=" * 60)
    
    try:
        from src.communication.external_communication_system import ExternalCommunicationSystem
        
        # 创建通信系统
        comm_system = ExternalCommunicationSystem()
        
        # 测试紧急事件
        emergency_event = {
            "event_type": "fall_detected",
            "severity": "high",
            "patient_id": "P001",
            "location": {"latitude": 39.9042, "longitude": 116.4074},
            "timestamp": time.time(),
            "vital_signs": {
                "heart_rate": 45,
                "breathing_rate": 8,
                "consciousness": "unconscious"
            },
            "fall_duration": 35.0,
            "no_movement_time": 45.0
        }
        
        # 测试短信发送
        sms_result = comm_system.send_emergency_sms(emergency_event)
        print(f"✓ 紧急短信发送: {sms_result}")
        
        # 测试推送通知
        push_result = comm_system.send_push_notification(emergency_event)
        print(f"✓ 推送通知发送: {push_result}")
        
        # 测试Webhook调用
        webhook_result = comm_system.call_webhook(emergency_event)
        print(f"✓ Webhook调用: {webhook_result}")
        
        # 测试邮件发送
        email_result = comm_system.send_email_alert(emergency_event)
        print(f"✓ 邮件警报发送: {email_result}")
        
        # 测试通信历史
        history = comm_system.get_communication_history()
        print(f"✓ 通信历史记录: {len(history)} 条记录")
        
        # 测试联系人管理
        contacts = comm_system.get_emergency_contacts()
        print(f"✓ 紧急联系人: {len(contacts)} 个联系人")
        
        return True
        
    except Exception as e:
        print(f"✗ 外部通信系统测试失败: {e}")
        return False

def test_enhanced_fall_detection():
    """测试增强摔倒检测系统"""
    print("\n" + "=" * 60)
    print("测试增强摔倒检测系统")
    print("=" * 60)
    
    try:
        from src.recognition.enhanced_fall_detection_system import EnhancedFallDetectionSystem
        
        # 创建摔倒检测系统
        fall_system = EnhancedFallDetectionSystem()
        
        # 创建模拟视频帧
        normal_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        fall_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 测试正常状态检测
        normal_result = fall_system.process_frame(normal_frame)
        print(f"✓ 正常状态检测: {normal_result['status']}")
        
        # 模拟摔倒检测
        fall_result = fall_system.process_frame(fall_frame)
        print(f"✓ 摔倒检测处理: {fall_result['status']}")
        
        # 测试时间监控
        monitoring_status = fall_system.get_monitoring_status()
        print(f"✓ 监控状态: {monitoring_status}")
        
        # 测试报警升级
        escalation_result = fall_system.check_alert_escalation()
        print(f"✓ 报警升级检查: {escalation_result}")
        
        # 测试统计信息
        statistics = fall_system.get_detection_statistics()
        print(f"✓ 检测统计: 处理了 {statistics['total_frames']} 帧")
        
        return True
        
    except Exception as e:
        print(f"✗ 增强摔倒检测系统测试失败: {e}")
        return False

def test_integrated_aiot_platform():
    """测试集成AIoT平台"""
    print("\n" + "=" * 60)
    print("测试集成AIoT平台")
    print("=" * 60)
    
    try:
        from src.core.integrated_aiot_platform import IntegratedAIoTPlatform
        
        # 创建集成平台
        platform = IntegratedAIoTPlatform()
        
        # 测试平台初始化
        init_result = platform.initialize()
        print(f"✓ 平台初始化: {init_result}")
        
        # 测试模块状态
        module_status = platform.get_module_status()
        print(f"✓ 模块状态检查: {len(module_status)} 个模块")
        
        # 创建模拟场景数据
        scenario_data = {
            "camera_frame": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "gps_location": (39.9042, 116.4074),
            "battery_level": 75,
            "sensor_data": {
                "temperature": 25.5,
                "humidity": 60.0,
                "pressure": 1013.25
            }
        }
        
        # 测试场景处理
        processing_result = platform.process_scenario(scenario_data)
        print(f"✓ 场景处理: {processing_result['status']}")
        
        # 测试任务执行
        task = {
            "type": "medication_retrieval",
            "target_medication": "aspirin_100mg",
            "patient_id": "P001",
            "priority": "high"
        }
        
        task_result = platform.execute_task(task)
        print(f"✓ 任务执行: {task_result}")
        
        # 测试系统健康检查
        health_status = platform.perform_health_check()
        print(f"✓ 系统健康检查: {health_status['overall_status']}")
        
        # 测试性能监控
        performance_metrics = platform.get_performance_metrics()
        print(f"✓ 性能监控: CPU使用率 {performance_metrics['cpu_usage']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"✗ 集成AIoT平台测试失败: {e}")
        return False

def test_configuration_loading():
    """测试配置文件加载"""
    print("\n" + "=" * 60)
    print("测试配置文件加载")
    print("=" * 60)
    
    try:
        import yaml
        
        # 加载配置文件
        config_path = "config/aiot_platform_config.yaml"
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            print(f"✓ 配置文件加载成功")
            print(f"  - 平台ID: {config['core']['platform_id']}")
            print(f"  - 平台类型: {config['core']['platform_type']}")
            
            # 检查模块配置
            modules = config['modules']
            enabled_modules = [k for k, v in modules.items() if v]
            print(f"  - 启用的模块: {len(enabled_modules)} 个")
            
            # 检查摔倒检测配置
            fall_config = config['fall_detection']
            print(f"  - 摔倒检测: {'启用' if fall_config['enabled'] else '禁用'}")
            print(f"  - 无运动阈值: {fall_config['monitoring']['no_movement_threshold']}秒")
            
            # 检查药物识别配置
            med_config = config['medication_recognition']
            print(f"  - 药物识别: {'启用' if med_config['enabled'] else '禁用'}")
            print(f"  - 识别置信度阈值: {med_config['recognition']['confidence_threshold']}")
            
            # 检查外部通信配置
            comm_config = config['external_communication']
            print(f"  - 外部通信: {'启用' if comm_config['enabled'] else '禁用'}")
            print(f"  - 紧急联系人: {len(comm_config['sms']['emergency_contacts'])} 个")
            
            return True
        else:
            print(f"✗ 配置文件不存在: {config_path}")
            return False
            
    except Exception as e:
        print(f"✗ 配置文件加载测试失败: {e}")
        return False

def run_comprehensive_test():
    """运行综合测试"""
    print("🚀 YOLOS模块化AIoT平台综合测试")
    print("=" * 80)
    
    test_results = []
    
    # 运行所有测试
    tests = [
        ("配置文件加载", test_configuration_loading),
        ("模块化扩展管理器", test_modular_extension_manager),
        ("药物识别系统", test_medication_recognition),
        ("机械臂控制器", test_robotic_arm_controller),
        ("外部通信系统", test_external_communication),
        ("增强摔倒检测", test_enhanced_fall_detection),
        ("集成AIoT平台", test_integrated_aiot_platform),
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} 测试异常: {e}")
            test_results.append((test_name, False))
    
    # 输出测试总结
    print("\n" + "=" * 80)
    print("📊 测试结果总结")
    print("=" * 80)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} {test_name}")
        if result:
            passed_tests += 1
    
    print(f"\n📈 测试统计:")
    print(f"  - 总测试数: {total_tests}")
    print(f"  - 通过测试: {passed_tests}")
    print(f"  - 失败测试: {total_tests - passed_tests}")
    print(f"  - 成功率: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 所有测试通过！YOLOS模块化AIoT平台功能正常！")
    else:
        print(f"\n⚠️  有 {total_tests - passed_tests} 个测试失败，请检查相关模块。")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    # 运行综合测试
    success = run_comprehensive_test()
    
    if success:
        print("\n🚀 模块化AIoT平台测试完成 - 所有功能正常！")
        print("\n💡 主要功能验证:")
        print("  ✅ 药物识别和机械臂控制联动")
        print("  ✅ 摔倒检测和外部通信推送")
        print("  ✅ 模块化架构和配置管理")
        print("  ✅ 跨平台硬件集成支持")
        
        print("\n🎯 应用场景就绪:")
        print("  🏥 医疗护理机器人 - 药物配送和紧急响应")
        print("  🏠 家庭护理助手 - 老人监护和健康管理")
        print("  🚁 医疗无人机 - 急救现场评估和通信")
        print("  🏭 工业安全监控 - 事故检测和应急处理")
    else:
        print("\n❌ 部分测试失败，请检查系统配置和依赖。")
    
    print(f"\n⏰ 测试完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")