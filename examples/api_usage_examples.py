#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOS 外部API使用示例
展示如何通过语音控制AIoT设备进行专项识别任务
"""

import sys
import os
import time
import asyncio
from typing import Dict, Any

# 添加SDK路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sdk.yolos_client_sdk import create_client, YOLOSClient, RecognitionResult

def example_1_basic_device_control():
    """示例1: 基础设备控制"""
    print("=" * 60)
    print("示例1: 基础设备控制")
    print("=" * 60)
    
    # 创建客户端
    client = create_client("http://localhost:8080")
    
    try:
        # 检查服务健康状态
        health = client.health_check()
        print(f"服务状态: {health}")
        
        # 获取设备当前状态
        status = client.get_device_status()
        print(f"设备当前位置: {status.position}")
        print(f"摄像头角度: {status.camera_angle}")
        print(f"电池电量: {status.battery_level}%")
        
        # 移动设备到客厅
        print("\n移动设备到客厅...")
        success = client.move_device_to_location("客厅")
        print(f"移动结果: {'成功' if success else '失败'}")
        
        # 旋转摄像头
        print("\n旋转摄像头...")
        success = client.rotate_camera(pan=45, tilt=15)
        print(f"旋转结果: {'成功' if success else '失败'}")
        
        # 调整缩放
        print("\n调整摄像头缩放...")
        success = client.zoom_camera(2.0)
        print(f"缩放结果: {'成功' if success else '失败'}")
        
        # 拍照
        print("\n拍照...")
        photo_result = client.take_photo()
        print(f"拍照结果: {photo_result}")
        
    except Exception as e:
        print(f"错误: {e}")
    finally:
        client.close()

def example_2_voice_control():
    """示例2: 语音控制"""
    print("=" * 60)
    print("示例2: 语音控制")
    print("=" * 60)
    
    client = create_client("http://localhost:8080")
    
    try:
        # 语音命令示例
        voice_commands = [
            "移动到客厅",
            "拍照",
            "识别药物",
            "监控宠物",
            "返回原点"
        ]
        
        for command in voice_commands:
            print(f"\n执行语音命令: {command}")
            result = client.execute_voice_command(command)
            print(f"执行结果: {result}")
            time.sleep(2)  # 等待2秒
        
        # 实际语音监听（需要麦克风）
        print("\n开始语音监听（10秒）...")
        voice_result = client.listen_voice_command(timeout=10.0)
        
        if voice_result.success:
            print(f"识别到语音: {voice_result.command_text}")
            print(f"置信度: {voice_result.confidence}")
            print(f"执行的操作: {voice_result.actions_taken}")
        else:
            print(f"语音识别失败: {voice_result.error_message}")
    
    except Exception as e:
        print(f"错误: {e}")
    finally:
        client.close()

def example_3_medication_detection():
    """示例3: 药物识别专项任务"""
    print("=" * 60)
    print("示例3: 药物识别专项任务")
    print("=" * 60)
    
    client = create_client("http://localhost:8080")
    
    try:
        # 场景：通过语音指令让设备移动到药箱位置并识别药物
        
        # 1. 语音指令移动到药箱位置
        print("步骤1: 移动到药箱位置")
        move_result = client.execute_voice_command("移动到厨房")  # 假设药箱在厨房
        print(f"移动结果: {move_result}")
        
        # 2. 调整摄像头角度以便更好地拍摄药箱
        print("\n步骤2: 调整摄像头角度")
        client.rotate_camera(pan=0, tilt=-30)  # 向下倾斜30度
        client.zoom_camera(2.5)  # 放大2.5倍
        
        # 3. 拍照
        print("\n步骤3: 拍摄药箱")
        photo_result = client.take_photo()
        print(f"拍照结果: {photo_result}")
        
        # 4. 启动药物识别任务
        print("\n步骤4: 启动药物识别任务")
        task_id = client.start_recognition_task(
            task_type="medication_detection",
            parameters={
                "confidence_threshold": 0.8,
                "enable_ocr": True,
                "check_expiry_date": True
            },
            priority=8  # 高优先级
        )
        print(f"任务ID: {task_id}")
        
        # 5. 等待任务完成
        print("\n步骤5: 等待识别完成...")
        result = client.wait_for_task_completion(task_id, timeout=30.0)
        print(f"识别结果: {result}")
        
        # 6. 如果检测到药物，进行详细分析
        if result.get('detected_medications'):
            medications = result['detected_medications']
            print(f"\n检测到 {len(medications)} 种药物:")
            
            for i, med in enumerate(medications, 1):
                print(f"  {i}. {med.get('name', '未知药物')}")
                print(f"     剂量: {med.get('dosage', '未知')}")
                print(f"     有效期: {med.get('expiry_date', '未知')}")
                print(f"     置信度: {med.get('confidence', 0):.2f}")
        
        # 7. 返回原点
        print("\n步骤6: 返回原点")
        client.return_home()
        
    except Exception as e:
        print(f"错误: {e}")
    finally:
        client.close()

def example_4_pet_monitoring():
    """示例4: 宠物监控专项任务"""
    print("=" * 60)
    print("示例4: 宠物监控专项任务")
    print("=" * 60)
    
    client = create_client("http://localhost:8080")
    
    try:
        # 场景：通过语音指令巡视各个房间寻找宠物
        
        locations = ["客厅", "卧室", "阳台"]
        
        for location in locations:
            print(f"\n检查位置: {location}")
            
            # 1. 移动到指定位置
            print(f"移动到{location}...")
            success = client.move_device_to_location(location)
            if not success:
                print(f"移动到{location}失败，跳过")
                continue
            
            # 2. 360度旋转搜索宠物
            print("360度搜索宠物...")
            for angle in [0, 90, 180, 270]:
                client.rotate_camera(pan=angle, tilt=0)
                time.sleep(1)  # 等待摄像头稳定
                
                # 3. 拍照并识别
                photo_result = client.take_photo()
                
                # 4. 宠物检测
                result = client.monitor_pet("current_frame")  # 使用当前帧
                
                if result.success and result.detected_objects:
                    pets = [obj for obj in result.detected_objects if 'pet' in obj.get('category', '')]
                    
                    if pets:
                        print(f"在{location}发现宠物!")
                        for pet in pets:
                            print(f"  物种: {pet.get('species', '未知')}")
                            print(f"  品种: {pet.get('breed', '未知')}")
                            print(f"  活动状态: {pet.get('activity', '未知')}")
                            print(f"  健康状态: {pet.get('health_status', '未知')}")
                        
                        # 发现宠物后可以进行持续监控
                        print("开始持续监控...")
                        monitoring_task_id = client.start_recognition_task(
                            task_type="pet_monitoring",
                            parameters={
                                "continuous_monitoring": True,
                                "duration": 300,  # 监控5分钟
                                "alert_on_abnormal_behavior": True
                            },
                            priority=6
                        )
                        print(f"监控任务ID: {monitoring_task_id}")
                        break
            else:
                print(f"在{location}未发现宠物")
        
    except Exception as e:
        print(f"错误: {e}")
    finally:
        client.close()

def example_5_fall_detection():
    """示例5: 跌倒检测专项任务"""
    print("=" * 60)
    print("示例5: 跌倒检测专项任务")
    print("=" * 60)
    
    client = create_client("http://localhost:8080")
    
    try:
        # 场景：老人监护，巡视各个房间进行跌倒检测
        
        print("启动老人监护模式...")
        
        # 1. 设置监护区域
        monitoring_areas = [
            {"name": "客厅", "priority": "high"},
            {"name": "卧室", "priority": "high"},
            {"name": "厨房", "priority": "medium"},
            {"name": "阳台", "priority": "low"}
        ]
        
        # 2. 开始巡视监控
        for area in monitoring_areas:
            area_name = area["name"]
            priority = area["priority"]
            
            print(f"\n监控区域: {area_name} (优先级: {priority})")
            
            # 移动到监控区域
            success = client.move_device_to_location(area_name)
            if not success:
                print(f"无法到达{area_name}，跳过")
                continue
            
            # 调整摄像头以获得最佳监控角度
            client.rotate_camera(pan=0, tilt=-15)  # 稍微向下倾斜
            client.zoom_camera(1.5)  # 适度放大
            
            # 启动跌倒检测任务
            task_id = client.start_recognition_task(
                task_type="fall_detection",
                parameters={
                    "real_time_monitoring": True,
                    "sensitivity": "high" if priority == "high" else "medium",
                    "emergency_alert": True,
                    "monitoring_duration": 60  # 监控1分钟
                },
                priority=9  # 跌倒检测优先级很高
            )
            
            print(f"跌倒检测任务启动，任务ID: {task_id}")
            
            # 等待监控结果
            try:
                result = client.wait_for_task_completion(task_id, timeout=65.0)
                
                if result.get('fall_detected'):
                    print("⚠️  检测到跌倒事件!")
                    print(f"置信度: {result.get('pose_analysis', {}).get('confidence', 0):.2f}")
                    
                    # 紧急处理流程
                    print("执行紧急响应流程...")
                    
                    # 1. 拍摄现场照片
                    emergency_photo = client.take_photo()
                    print(f"紧急现场照片: {emergency_photo}")
                    
                    # 2. 开始录像
                    client.start_recording()
                    print("开始紧急录像...")
                    
                    # 3. 进行医疗状况分析
                    medical_task_id = client.start_recognition_task(
                        task_type="medical_analysis",
                        parameters={
                            "emergency_mode": True,
                            "vital_signs_check": True
                        },
                        priority=10  # 最高优先级
                    )
                    
                    medical_result = client.wait_for_task_completion(medical_task_id, timeout=30.0)
                    print(f"医疗分析结果: {medical_result}")
                    
                    # 4. 这里应该触发外部报警系统
                    print("应该触发外部报警系统...")
                    
                    break  # 发现跌倒后停止巡视
                else:
                    print(f"{area_name}区域正常，未检测到跌倒")
                    
            except Exception as e:
                print(f"监控任务异常: {e}")
        
        print("\n监护巡视完成")
        
    except Exception as e:
        print(f"错误: {e}")
    finally:
        client.close()

def example_6_integrated_voice_workflow():
    """示例6: 集成语音工作流"""
    print("=" * 60)
    print("示例6: 集成语音工作流")
    print("=" * 60)
    
    client = create_client("http://localhost:8080", enable_websocket=True)
    
    try:
        # 连接WebSocket以支持实时语音交互
        if not client.connect_websocket():
            print("WebSocket连接失败，使用HTTP模式")
        
        print("YOLOS智能助手已启动")
        print("支持的语音命令:")
        print("- '移动到[位置]' - 移动设备")
        print("- '识别药物' - 药物检测")
        print("- '监控宠物' - 宠物监控")
        print("- '检测跌倒' - 跌倒检测")
        print("- '拍照' - 拍摄照片")
        print("- '返回' - 返回原点")
        print("- '退出' - 结束程序")
        
        # 语音命令处理循环
        while True:
            print("\n请说出您的指令...")
            
            # 监听语音命令
            voice_result = client.listen_voice_command(timeout=15.0)
            
            if voice_result.success:
                command = voice_result.command_text
                print(f"收到指令: {command}")
                
                # 退出命令
                if "退出" in command or "结束" in command:
                    print("正在退出...")
                    break
                
                # 执行语音命令
                try:
                    execution_result = client.execute_voice_command(command)
                    print(f"执行结果: {execution_result}")
                    
                    # 如果是识别任务，等待结果
                    if execution_result.get('task_id'):
                        task_id = execution_result['task_id']
                        print(f"等待任务完成: {task_id}")
                        
                        task_result = client.wait_for_task_completion(task_id, timeout=30.0)
                        print(f"任务结果: {task_result}")
                
                except Exception as e:
                    print(f"命令执行失败: {e}")
            
            else:
                print(f"语音识别失败: {voice_result.error_message}")
                print("请重新说出指令")
    
    except KeyboardInterrupt:
        print("\n用户中断，正在退出...")
    except Exception as e:
        print(f"错误: {e}")
    finally:
        client.close()

def example_7_batch_processing():
    """示例7: 批量处理任务"""
    print("=" * 60)
    print("示例7: 批量处理任务")
    print("=" * 60)
    
    client = create_client("http://localhost:8080")
    
    try:
        # 批量巡视和识别任务
        locations = ["客厅", "卧室", "厨房", "阳台"]
        recognition_tasks = ["general_recognition", "pet_monitoring", "object_inventory"]
        
        all_results = {}
        
        for location in locations:
            print(f"\n处理位置: {location}")
            location_results = {}
            
            # 移动到位置
            success = client.move_device_to_location(location)
            if not success:
                print(f"无法到达{location}")
                continue
            
            # 拍照
            photo_result = client.take_photo()
            location_results['photo'] = photo_result
            
            # 执行多种识别任务
            task_ids = []
            for task_type in recognition_tasks:
                task_id = client.start_recognition_task(
                    task_type=task_type,
                    parameters={"location": location},
                    priority=5
                )
                task_ids.append((task_type, task_id))
                print(f"启动{task_type}任务: {task_id}")
            
            # 等待所有任务完成
            task_results = {}
            for task_type, task_id in task_ids:
                try:
                    result = client.wait_for_task_completion(task_id, timeout=30.0)
                    task_results[task_type] = result
                    print(f"{task_type}完成: {len(result.get('detected_objects', []))}个对象")
                except Exception as e:
                    print(f"{task_type}任务失败: {e}")
                    task_results[task_type] = {"error": str(e)}
            
            location_results['recognition'] = task_results
            all_results[location] = location_results
        
        # 生成总结报告
        print("\n" + "="*60)
        print("批量处理总结报告")
        print("="*60)
        
        for location, results in all_results.items():
            print(f"\n位置: {location}")
            
            if 'photo' in results:
                print(f"  拍照: 成功")
            
            if 'recognition' in results:
                for task_type, result in results['recognition'].items():
                    if 'error' in result:
                        print(f"  {task_type}: 失败 - {result['error']}")
                    else:
                        object_count = len(result.get('detected_objects', []))
                        print(f"  {task_type}: 成功 - {object_count}个对象")
        
        # 返回原点
        client.return_home()
        print("\n已返回原点")
        
    except Exception as e:
        print(f"错误: {e}")
    finally:
        client.close()

def main():
    """主函数 - 运行所有示例"""
    print("YOLOS 外部API使用示例")
    print("支持语音控制的AIoT设备专项识别任务")
    print("="*80)
    
    examples = [
        ("基础设备控制", example_1_basic_device_control),
        ("语音控制", example_2_voice_control),
        ("药物识别专项任务", example_3_medication_detection),
        ("宠物监控专项任务", example_4_pet_monitoring),
        ("跌倒检测专项任务", example_5_fall_detection),
        ("集成语音工作流", example_6_integrated_voice_workflow),
        ("批量处理任务", example_7_batch_processing),
    ]
    
    print("可用示例:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")
    
    print("0. 运行所有示例")
    print("q. 退出")
    
    while True:
        choice = input("\n请选择要运行的示例 (0-7, q): ").strip()
        
        if choice.lower() == 'q':
            break
        
        try:
            if choice == '0':
                # 运行所有示例
                for name, func in examples:
                    print(f"\n正在运行: {name}")
                    func()
                    input("按Enter继续下一个示例...")
            else:
                index = int(choice) - 1
                if 0 <= index < len(examples):
                    name, func = examples[index]
                    print(f"\n正在运行: {name}")
                    func()
                else:
                    print("无效选择，请重新输入")
        
        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            print("\n示例被中断")
        except Exception as e:
            print(f"示例运行出错: {e}")

if __name__ == "__main__":
    main()