#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的ModelScope服务测试
"""

import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from openai import OpenAI

def test_api_connection():
    """测试API连接"""
    try:
        print("=== 测试ModelScope API连接 ===")
        
        client = OpenAI(
            base_url='https://api-inference.modelscope.cn/v1',
            api_key='*****'
        )
        
        # 测试文本请求
        response = client.chat.completions.create(
            model='Qwen/Qwen2.5-VL-72B-Instruct',
            messages=[{
                'role': 'user',
                'content': [{'type': 'text', 'text': '你好，请简单介绍一下你的视觉分析能力。'}]
            }],
            max_tokens=200,
            timeout=30
        )
        
        if response.choices:
            print("✓ API连接成功")
            print(f"响应: {response.choices[0].message.content}")
            return True
        else:
            print("✗ API响应为空")
            return False
            
    except Exception as e:
        print(f"✗ API连接失败: {e}")
        return False

def create_test_image():
    """创建测试图像"""
    try:
        # 创建目录
        test_dir = Path("test_images")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建测试图像
        image = np.zeros((400, 600, 3), dtype=np.uint8)
        
        # 背景色
        image[:] = (240, 248, 255)  # 淡蓝色
        
        # 添加一些几何图形
        cv2.rectangle(image, (50, 50), (200, 150), (0, 255, 0), -1)  # 绿色矩形
        cv2.circle(image, (400, 100), 50, (255, 0, 0), -1)  # 蓝色圆形
        cv2.ellipse(image, (300, 250), (80, 40), 0, 0, 360, (0, 0, 255), -1)  # 红色椭圆
        
        # 添加文字
        cv2.putText(image, "YOLOS Test Image", (150, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(image, "Medical AI System", (150, 350), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        
        # 保存图像
        test_image_path = test_dir / "test_image.jpg"
        cv2.imwrite(str(test_image_path), image)
        
        print(f"✓ 创建测试图像: {test_image_path}")
        return str(test_image_path)
        
    except Exception as e:
        print(f"✗ 创建测试图像失败: {e}")
        return None

def test_image_analysis(image_path):
    """测试图像分析"""
    try:
        print(f"\n=== 测试图像分析 ===")
        print(f"分析图像: {image_path}")
        
        client = OpenAI(
            base_url='https://api-inference.modelscope.cn/v1',
            api_key='*****'
        )
        
        # 读取并编码图像
        import base64
        with open(image_path, 'rb') as f:
            image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # 构建分析提示
        prompt = """
        请详细分析这张图像，并按照以下JSON格式返回结果：
        
        {
          "scene_description": "详细的场景描述",
          "detected_objects": [
            {
              "name": "对象名称",
              "confidence": 0.95,
              "category": "对象类别",
              "location": "位置描述"
            }
          ],
          "scene_category": "场景类别",
          "overall_confidence": 0.85,
          "technical_details": {
            "image_quality": "excellent/good/fair/poor",
            "lighting_conditions": "描述",
            "clarity": "清晰度评估"
          }
        }
        """
        
        start_time = time.time()
        
        # 发送请求
        response = client.chat.completions.create(
            model='Qwen/Qwen2.5-VL-72B-Instruct',
            messages=[{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/jpeg;base64,{image_base64}'
                        }
                    }
                ]
            }],
            max_tokens=2048,
            temperature=0.1,
            timeout=60
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if response.choices:
            result = response.choices[0].message.content
            
            print("✓ 图像分析成功")
            print(f"处理时间: {processing_time:.2f}秒")
            print(f"分析结果:")
            print("-" * 50)
            print(result)
            print("-" * 50)
            
            # 尝试解析JSON
            try:
                import json
                if result.strip().startswith('{'):
                    parsed_result = json.loads(result)
                    print("\n✓ JSON格式解析成功")
                    print(f"场景描述: {parsed_result.get('scene_description', 'N/A')}")
                    print(f"场景类别: {parsed_result.get('scene_category', 'N/A')}")
                    print(f"整体置信度: {parsed_result.get('overall_confidence', 'N/A')}")
                    
                    detected_objects = parsed_result.get('detected_objects', [])
                    if detected_objects:
                        print(f"检测到 {len(detected_objects)} 个对象:")
                        for i, obj in enumerate(detected_objects):
                            print(f"  {i+1}. {obj.get('name', 'Unknown')} - {obj.get('category', 'Unknown')}")
                else:
                    print("⚠ 结果不是JSON格式，但分析成功")
            except json.JSONDecodeError:
                print("⚠ JSON解析失败，但分析成功")
            
            return True
        else:
            print("✗ 图像分析失败：API返回空结果")
            return False
            
    except Exception as e:
        print(f"✗ 图像分析失败: {e}")
        return False

def test_medical_analysis():
    """测试医疗场景分析"""
    try:
        print(f"\n=== 测试医疗场景分析 ===")
        
        client = OpenAI(
            base_url='https://api-inference.modelscope.cn/v1',
            api_key='*****'
        )
        
        # 医疗分析提示
        medical_prompt = """
        作为YOLOS医疗AI系统的视觉分析专家，请分析以下医疗场景需求：
        
        1. 如果图像中包含药物，请识别药品类型、包装信息
        2. 如果图像中包含人员，请评估健康状态、是否有跌倒风险
        3. 如果图像中包含医疗器械，请识别类型和用途
        4. 评估整体安全状况和紧急程度
        
        请提供专业的医疗分析建议。
        """
        
        response = client.chat.completions.create(
            model='Qwen/Qwen2.5-VL-72B-Instruct',
            messages=[{
                'role': 'user',
                'content': [{'type': 'text', 'text': medical_prompt}]
            }],
            max_tokens=1000,
            temperature=0.05,  # 医疗场景需要更保守
            timeout=30
        )
        
        if response.choices:
            result = response.choices[0].message.content
            print("✓ 医疗分析能力测试成功")
            print("医疗AI分析能力:")
            print("-" * 50)
            print(result)
            print("-" * 50)
            return True
        else:
            print("✗ 医疗分析测试失败")
            return False
            
    except Exception as e:
        print(f"✗ 医疗分析测试失败: {e}")
        return False

def main():
    """主函数"""
    print("YOLOS ModelScope服务简化测试")
    print("=" * 60)
    
    # 测试结果统计
    test_results = []
    
    # 1. 测试API连接
    api_success = test_api_connection()
    test_results.append(("API连接", api_success))
    
    if not api_success:
        print("\n❌ API连接失败，无法继续测试")
        return
    
    # 2. 创建测试图像
    test_image_path = create_test_image()
    if test_image_path:
        test_results.append(("测试图像创建", True))
        
        # 3. 测试图像分析
        analysis_success = test_image_analysis(test_image_path)
        test_results.append(("图像分析", analysis_success))
    else:
        test_results.append(("测试图像创建", False))
        test_results.append(("图像分析", False))
    
    # 4. 测试医疗分析
    medical_success = test_medical_analysis()
    test_results.append(("医疗分析", medical_success))
    
    # 输出测试总结
    print("\n" + "=" * 60)
    print("测试结果总结:")
    print("=" * 60)
    
    success_count = 0
    for test_name, success in test_results:
        status = "✓ 成功" if success else "✗ 失败"
        print(f"{test_name:<15}: {status}")
        if success:
            success_count += 1
    
    print(f"\n总体结果: {success_count}/{len(test_results)} 项测试通过")
    
    if success_count == len(test_results):
        print("🎉 所有测试通过！ModelScope集成成功！")
    elif success_count > 0:
        print("⚠️  部分测试通过，系统基本可用")
    else:
        print("❌ 测试失败，请检查配置和网络连接")
    
    print("\n下一步建议:")
    if api_success:
        print("1. ✅ ModelScope API集成成功")
        print("2. 📝 可以开始集成到YOLOS主系统")
        print("3. 🔧 根据实际需求调整配置参数")
        print("4. 📊 进行性能基准测试")
        print("5. 🚀 部署到生产环境")
    else:
        print("1. 🔍 检查API密钥是否正确")
        print("2. 🌐 检查网络连接")
        print("3. 📋 查看详细错误日志")

if __name__ == "__main__":
    main()