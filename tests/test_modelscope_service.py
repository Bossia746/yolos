#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试ModelScope大模型服务
"""

import os
import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_modelscope_service():
    """测试ModelScope服务"""
    try:
        # 导入服务
        from src.recognition.modelscope_llm_service import ModelScopeLLMService, analyze_training_images
        
        print("=== ModelScope大模型服务测试 ===")
        
        # 创建服务实例
        service = ModelScopeLLMService()
        
        # 检查服务状态
        status = service.get_service_status()
        print(f"服务状态: {status}")
        
        # 启动服务
        print("\n启动服务...")
        if service.start_service():
            print("✓ 服务启动成功")
        else:
            print("✗ 服务启动失败")
            return
        
        # 测试训练图像分析
        print("\n=== 分析训练图像 ===")
        
        # 检查训练图像目录
        training_dir = Path("test_images")
        if not training_dir.exists():
            print(f"训练图像目录不存在: {training_dir}")
            print("创建测试目录和图像...")
            training_dir.mkdir(parents=True, exist_ok=True)
            
            # 如果没有图像，提示用户
            print("请将测试图像放入 'test_images' 目录")
            return
        
        # 获取图像文件
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(list(training_dir.glob(f"*{ext}")))
            image_files.extend(list(training_dir.glob(f"*{ext.upper()}")))
        
        if not image_files:
            print("未找到图像文件，创建测试用例...")
            # 创建一个简单的测试图像
            import cv2
            import numpy as np
            
            # 创建一个简单的测试图像
            test_image = np.zeros((300, 400, 3), dtype=np.uint8)
            test_image[:] = (100, 150, 200)  # 浅蓝色背景
            
            # 添加一些文字
            cv2.putText(test_image, "YOLOS Test Image", (50, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            test_image_path = training_dir / "test_image.jpg"
            cv2.imwrite(str(test_image_path), test_image)
            print(f"创建测试图像: {test_image_path}")
            
            image_files = [test_image_path]
        
        print(f"找到 {len(image_files)} 张图像")
        
        # 分析第一张图像
        if image_files:
            first_image = str(image_files[0])
            print(f"\n分析图像: {first_image}")
            
            start_time = time.time()
            result = service.analyze_image(first_image, "general")
            end_time = time.time()
            
            if result:
                print("✓ 分析成功")
                print(f"处理时间: {end_time - start_time:.2f}秒")
                print(f"场景描述: {result.scene_description}")
                print(f"场景类别: {result.scene_category}")
                print(f"置信度: {result.overall_confidence}")
                print(f"使用模型: {result.model_used}")
                
                if result.detected_objects:
                    print("检测到的对象:")
                    for obj in result.detected_objects:
                        print(f"  - {obj}")
                
                print(f"原始响应: {result.raw_response[:200]}...")
            else:
                print("✗ 分析失败")
        
        # 获取服务统计
        print("\n=== 服务统计 ===")
        final_status = service.get_service_status()
        print(f"总请求数: {final_status['stats']['total_requests']}")
        print(f"成功请求数: {final_status['stats']['successful_requests']}")
        print(f"失败请求数: {final_status['stats']['failed_requests']}")
        print(f"缓存命中数: {final_status['stats']['cache_hits']}")
        
        # 停止服务
        print("\n停止服务...")
        service.stop_service()
        print("✓ 服务已停止")
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保已安装所需的依赖包:")
        print("pip install openai opencv-python numpy pyyaml psutil")
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_simple_api_call():
    """测试简单的API调用"""
    try:
        from openai import OpenAI
        
        print("=== 测试ModelScope API连接 ===")
        
        client = OpenAI(
            base_url='https://api-inference.modelscope.cn/v1',
            api_key='*****'
        )
        
        # 测试简单的文本请求
        response = client.chat.completions.create(
            model='Qwen/Qwen2.5-VL-72B-Instruct',
            messages=[{
                'role': 'user',
                'content': [{'type': 'text', 'text': '你好，请简单介绍一下你自己。'}]
            }],
            max_tokens=100,
            timeout=30
        )
        
        if response.choices:
            print("✓ API连接成功")
            print(f"响应: {response.choices[0].message.content}")
        else:
            print("✗ API响应为空")
            
    except Exception as e:
        print(f"API测试失败: {e}")

def main():
    """主函数"""
    print("YOLOS ModelScope服务测试")
    print("=" * 50)
    
    # 首先测试API连接
    test_simple_api_call()
    
    print("\n" + "=" * 50)
    
    # 然后测试完整服务
    test_modelscope_service()

if __name__ == "__main__":
    main()