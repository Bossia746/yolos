#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOS视觉识别测试脚本
使用ModelScope进行图像分析并生成HTML报告
"""

import os
import cv2
import json
import base64
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI

class VisualRecognitionTester:
    """视觉识别测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.client = OpenAI(
            base_url='https://api-inference.modelscope.cn/v1',
            api_key='*****'
        )
        
        self.model_name = 'Qwen/Qwen2.5-VL-72B-Instruct'
        self.test_results = []
        self.start_time = datetime.now()
        
        print("YOLOS视觉识别测试器初始化完成")
        print(f"使用模型: {self.model_name}")
        print(f"测试开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """将图像编码为base64"""
        try:
            with open(image_path, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return f"data:image/jpeg;base64,{encoded_string}"
        except Exception as e:
            print(f"图像编码失败 {image_path}: {e}")
            return None
    
    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        """获取图像基本信息"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "无法读取图像"}
            
            height, width, channels = image.shape
            file_size = os.path.getsize(image_path)
            
            return {
                "width": width,
                "height": height,
                "channels": channels,
                "file_size": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_image_with_modelscope(self, image_path: str, prompt: str = None) -> Dict[str, Any]:
        """使用ModelScope分析图像"""
        try:
            print(f"正在分析图像: {os.path.basename(image_path)}")
            start_time = time.time()
            
            # 编码图像
            image_base64 = self.encode_image_to_base64(image_path)
            if not image_base64:
                return {"error": "图像编码失败"}
            
            # 默认提示词
            if not prompt:
                prompt = """请详细描述这幅图像，包括：
1. 主要物体和人物
2. 场景环境
3. 颜色和光线
4. 可能的用途或含义
5. 任何值得注意的细节
请用中文回答。"""
            
            # 调用ModelScope API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': [{
                        'type': 'text',
                        'text': prompt
                    }, {
                        'type': 'image_url',
                        'image_url': {
                            'url': image_base64
                        }
                    }]
                }],
                max_tokens=2048,
                temperature=0.1
            )
            
            processing_time = time.time() - start_time
            
            # 解析响应
            if response.choices and len(response.choices) > 0:
                analysis_result = response.choices[0].message.content
                
                return {
                    "success": True,
                    "analysis": analysis_result,
                    "processing_time": round(processing_time, 2),
                    "model_used": self.model_name,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {"error": "API响应为空"}
                
        except Exception as e:
            processing_time = time.time() - start_time if 'start_time' in locals() else 0
            return {
                "error": str(e),
                "processing_time": round(processing_time, 2)
            }
    
    def test_multiple_prompts(self, image_path: str) -> List[Dict[str, Any]]:
        """使用多个提示词测试同一图像"""
        prompts = [
            {
                "name": "通用描述",
                "prompt": "请详细描述这幅图像的内容，包括主要物体、人物、场景和环境。"
            },
            {
                "name": "医疗健康分析",
                "prompt": "从医疗健康的角度分析这幅图像，识别是否有人员、医疗设备、健康相关的物品或场景。"
            },
            {
                "name": "安全监控分析",
                "prompt": "从安全监控的角度分析这幅图像，识别人员活动、潜在风险、异常情况等。"
            },
            {
                "name": "物体检测",
                "prompt": "请识别图像中的所有物体，并尽可能准确地描述它们的位置、大小和特征。"
            }
        ]
        
        results = []
        for prompt_info in prompts:
            print(f"  - 测试提示词: {prompt_info['name']}")
            result = self.analyze_image_with_modelscope(image_path, prompt_info['prompt'])
            result['prompt_name'] = prompt_info['name']
            result['prompt'] = prompt_info['prompt']
            results.append(result)
            
            # 避免API限流
            time.sleep(1)
        
        return results
    
    def test_all_images(self, image_dir: str) -> List[Dict[str, Any]]:
        """测试目录中的所有图像"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        # 查找所有图像文件
        for file in os.listdir(image_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(image_dir, file))
        
        print(f"发现 {len(image_files)} 张图像文件")
        print("图像文件列表:")
        for img_file in image_files:
            print(f"  - {os.path.basename(img_file)}")
        print()
        
        all_results = []
        
        for i, image_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] 处理图像: {os.path.basename(image_path)}")
            
            # 获取图像基本信息
            image_info = self.get_image_info(image_path)
            
            # 进行多提示词测试
            analysis_results = self.test_multiple_prompts(image_path)
            
            # 整合结果
            result = {
                "image_path": image_path,
                "image_name": os.path.basename(image_path),
                "image_info": image_info,
                "analysis_results": analysis_results,
                "test_timestamp": datetime.now().isoformat()
            }
            
            all_results.append(result)
            print(f"  ✓ 完成图像分析，共 {len(analysis_results)} 个测试结果")
            print()
        
        return all_results
    
    def generate_html_report(self, results: List[Dict[str, Any]], output_path: str = "visual_recognition_report.html"):
        """生成HTML报告"""
        print("正在生成HTML报告...")
        
        # 计算统计信息
        total_images = len(results)
        total_tests = sum(len(result['analysis_results']) for result in results)
        successful_tests = sum(
            sum(1 for analysis in result['analysis_results'] if analysis.get('success', False))
            for result in results
        )
        
        average_processing_time = 0
        if successful_tests > 0:
            total_time = sum(
                sum(analysis.get('processing_time', 0) for analysis in result['analysis_results'] if analysis.get('success', False))
                for result in results
            )
            average_processing_time = round(total_time / successful_tests, 2)
        
        # HTML模板
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLOS视觉识别测试报告</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #007bff;
        }}
        .header h1 {{
            color: #007bff;
            margin-bottom: 10px;
        }}
        .header .subtitle {{
            color: #666;
            font-size: 18px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #007bff, #0056b3);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            font-size: 24px;
        }}
        .stat-card p {{
            margin: 0;
            font-size: 14px;
            opacity: 0.9;
        }}
        .image-section {{
            margin-bottom: 50px;
            border: 1px solid #ddd;
            border-radius: 10px;
            overflow: hidden;
        }}
        .image-header {{
            background-color: #f8f9fa;
            padding: 20px;
            border-bottom: 1px solid #ddd;
        }}
        .image-header h2 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .image-info {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }}
        .info-item {{
            background-color: white;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #e9ecef;
        }}
        .info-item strong {{
            color: #007bff;
        }}
        .image-display {{
            text-align: center;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .image-display img {{
            max-width: 100%;
            max-height: 400px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .analysis-results {{
            padding: 20px;
        }}
        .analysis-item {{
            margin-bottom: 30px;
            border-left: 4px solid #007bff;
            padding-left: 20px;
        }}
        .analysis-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .analysis-title {{
            font-size: 18px;
            font-weight: bold;
            color: #007bff;
        }}
        .analysis-meta {{
            font-size: 12px;
            color: #666;
        }}
        .analysis-content {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            white-space: pre-wrap;
            line-height: 1.8;
        }}
        .success {{
            border-left-color: #28a745;
        }}
        .success .analysis-title {{
            color: #28a745;
        }}
        .error {{
            border-left-color: #dc3545;
        }}
        .error .analysis-title {{
            color: #dc3545;
        }}
        .error .analysis-content {{
            background-color: #f8d7da;
            color: #721c24;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
        }}
        .prompt-info {{
            background-color: #e3f2fd;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
            font-size: 14px;
            color: #1565c0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 YOLOS视觉识别测试报告</h1>
            <div class="subtitle">基于ModelScope Qwen2.5-VL-72B-Instruct模型</div>
            <div class="subtitle">测试时间: {self.start_time.strftime('%Y年%m月%d日 %H:%M:%S')}</div>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <h3>{total_images}</h3>
                <p>测试图像数量</p>
            </div>
            <div class="stat-card">
                <h3>{total_tests}</h3>
                <p>总测试次数</p>
            </div>
            <div class="stat-card">
                <h3>{successful_tests}</h3>
                <p>成功测试次数</p>
            </div>
            <div class="stat-card">
                <h3>{round(successful_tests/total_tests*100, 1) if total_tests > 0 else 0}%</h3>
                <p>成功率</p>
            </div>
            <div class="stat-card">
                <h3>{average_processing_time}s</h3>
                <p>平均处理时间</p>
            </div>
        </div>
"""
        
        # 为每个图像生成详细报告
        for i, result in enumerate(results, 1):
            image_name = result['image_name']
            image_info = result['image_info']
            analysis_results = result['analysis_results']
            
            # 图像信息
            info_html = ""
            if 'error' not in image_info:
                info_html = f"""
                <div class="image-info">
                    <div class="info-item"><strong>尺寸:</strong> {image_info['width']} × {image_info['height']}</div>
                    <div class="info-item"><strong>通道:</strong> {image_info['channels']}</div>
                    <div class="info-item"><strong>文件大小:</strong> {image_info['file_size_mb']} MB</div>
                </div>
                """
            
            # 图像显示
            image_path_for_html = result['image_path'].replace('\\', '/')
            
            html_content += f"""
        <div class="image-section">
            <div class="image-header">
                <h2>📷 图像 {i}: {image_name}</h2>
                {info_html}
            </div>
            
            <div class="image-display">
                <img src="{image_path_for_html}" alt="{image_name}" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                <div style="display:none; color:#666; padding:20px;">图像加载失败: {image_name}</div>
            </div>
            
            <div class="analysis-results">
                <h3>🤖 AI分析结果</h3>
"""
            
            # 分析结果
            for j, analysis in enumerate(analysis_results, 1):
                success_class = "success" if analysis.get('success', False) else "error"
                
                if analysis.get('success', False):
                    content = analysis['analysis']
                    meta_info = f"处理时间: {analysis['processing_time']}秒 | 模型: {analysis['model_used']}"
                else:
                    content = f"错误: {analysis.get('error', '未知错误')}"
                    meta_info = f"处理时间: {analysis.get('processing_time', 0)}秒"
                
                html_content += f"""
                <div class="analysis-item {success_class}">
                    <div class="analysis-header">
                        <div class="analysis-title">{analysis.get('prompt_name', f'测试 {j}')}</div>
                        <div class="analysis-meta">{meta_info}</div>
                    </div>
                    <div class="prompt-info">
                        <strong>提示词:</strong> {analysis.get('prompt', '无')}
                    </div>
                    <div class="analysis-content">{content}</div>
                </div>
"""
            
            html_content += """
            </div>
        </div>
"""
        
        # 结束HTML
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        html_content += f"""
        <div class="footer">
            <p>📊 报告生成完成</p>
            <p>总耗时: {round(total_duration, 2)}秒 | 生成时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>YOLOS项目 - 智能视觉识别系统</p>
        </div>
    </div>
</body>
</html>
"""
        
        # 保存HTML文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ HTML报告已生成: {output_path}")
        return output_path

def main():
    """主函数"""
    print("🚀 启动YOLOS视觉识别测试")
    print("=" * 60)
    
    # 创建测试器
    tester = VisualRecognitionTester()
    
    # 测试图像目录
    image_dir = "test_images"
    
    if not os.path.exists(image_dir):
        print(f"❌ 图像目录不存在: {image_dir}")
        return
    
    try:
        # 执行测试
        results = tester.test_all_images(image_dir)
        
        if not results:
            print("❌ 没有找到可测试的图像文件")
            return
        
        # 生成HTML报告
        report_path = tester.generate_html_report(results)
        
        # 显示总结
        print("\n" + "=" * 60)
        print("📋 测试总结")
        print("=" * 60)
        print(f"✓ 测试图像数量: {len(results)}")
        
        total_tests = sum(len(result['analysis_results']) for result in results)
        successful_tests = sum(
            sum(1 for analysis in result['analysis_results'] if analysis.get('success', False))
            for result in results
        )
        
        print(f"✓ 总测试次数: {total_tests}")
        print(f"✓ 成功测试次数: {successful_tests}")
        print(f"✓ 成功率: {round(successful_tests/total_tests*100, 1)}%" if total_tests > 0 else "✓ 成功率: 0%")
        print(f"✓ HTML报告: {report_path}")
        
        # 显示每个图像的结果摘要
        print("\n📸 图像分析摘要:")
        for i, result in enumerate(results, 1):
            success_count = sum(1 for analysis in result['analysis_results'] if analysis.get('success', False))
            total_count = len(result['analysis_results'])
            print(f"  {i}. {result['image_name']}: {success_count}/{total_count} 成功")
        
        print(f"\n🎉 测试完成！请打开 {report_path} 查看详细报告")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()