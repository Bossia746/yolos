#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化YOLO检测报告生成器
包含原图和检测结果的对比展示
"""

import os
import cv2
import base64
from datetime import datetime
from pathlib import Path
import json

def image_to_base64(image_path):
    """将图像转换为base64编码"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"❌ 转换图像失败 {image_path}: {e}")
        return None

def create_visual_comparison_report():
    """创建包含原图和检测结果对比的可视化报告"""
    
    # 图像路径
    image_dir = Path("test_images")
    annotated_dir = Path(".")  # 检测结果图像在当前目录
    
    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    original_images = []
    
    if image_dir.exists():
        for ext in image_extensions:
            original_images.extend(list(image_dir.glob(f"*{ext}")))
            original_images.extend(list(image_dir.glob(f"*{ext.upper()}")))
    
    print(f"📁 发现 {len(original_images)} 张原始图像")
    
    # HTML模板开始
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLO检测可视化对比报告</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .comparison-item {{
            margin: 30px;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .item-header {{
            background: #f8f9fa;
            padding: 20px;
            border-bottom: 1px solid #e0e0e0;
        }}
        .item-title {{
            font-size: 1.4em;
            font-weight: bold;
            color: #2c3e50;
            margin: 0 0 10px 0;
        }}
        .item-info {{
            color: #666;
            font-size: 0.9em;
        }}
        .image-comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0;
        }}
        .image-section {{
            padding: 20px;
        }}
        .image-section h3 {{
            margin: 0 0 15px 0;
            color: #34495e;
            font-size: 1.1em;
            text-align: center;
            padding: 10px;
            border-radius: 5px;
        }}
        .original-section h3 {{
            background: #e8f5e8;
            color: #27ae60;
        }}
        .detected-section h3 {{
            background: #e8f4fd;
            color: #3498db;
        }}
        .image-container {{
            text-align: center;
            border: 2px dashed #ddd;
            border-radius: 8px;
            padding: 10px;
            min-height: 200px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .image-container img {{
            max-width: 100%;
            max-height: 400px;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .no-image {{
            color: #999;
            font-style: italic;
        }}
        .detection-stats {{
            background: #f8f9fa;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }}
        .stat-item {{
            text-align: center;
            padding: 8px;
            background: white;
            border-radius: 4px;
            border: 1px solid #e0e0e0;
        }}
        .stat-value {{
            font-size: 1.2em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .stat-label {{
            font-size: 0.8em;
            color: #666;
            margin-top: 2px;
        }}
        .detection-details {{
            margin-top: 15px;
            padding: 10px;
            background: #fff;
            border-radius: 5px;
            border: 1px solid #e0e0e0;
        }}
        .object-list {{
            list-style: none;
            padding: 0;
            margin: 10px 0;
        }}
        .object-item {{
            padding: 5px 10px;
            margin: 3px 0;
            background: #f0f8ff;
            border-radius: 3px;
            border-left: 3px solid #3498db;
            font-size: 0.9em;
        }}
        .confidence-high {{ border-left-color: #27ae60; background: #f0fff0; }}
        .confidence-medium {{ border-left-color: #f39c12; background: #fffbf0; }}
        .confidence-low {{ border-left-color: #e74c3c; background: #fff0f0; }}
        
        @media (max-width: 768px) {{
            .image-comparison {{
                grid-template-columns: 1fr;
            }}
            .container {{
                margin: 10px;
            }}
            .header {{
                padding: 20px;
            }}
            .header h1 {{
                font-size: 2em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 YOLO检测可视化对比报告</h1>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>原图 vs 检测结果对比分析</p>
        </div>
"""
    
    # 处理每张图像
    processed_count = 0
    for i, original_path in enumerate(original_images[:10], 1):  # 限制处理前10张图像
        print(f"[{i}] 处理图像: {original_path.name}")
        
        # 查找对应的检测结果图像
        annotated_name = f"annotated_{original_path.stem}.jpg"
        annotated_path = annotated_dir / annotated_name
        
        # 转换图像为base64
        original_b64 = image_to_base64(original_path)
        annotated_b64 = None
        
        if annotated_path.exists():
            annotated_b64 = image_to_base64(annotated_path)
        
        # 读取检测结果（如果存在结果文件）
        detection_info = {
            'objects_count': 0,
            'confidence_avg': 0.0,
            'processing_time': 0.0,
            'objects': [],
            'status': '未找到检测结果'
        }
        
        # 尝试从之前的测试结果中获取信息
        result_file = Path(f"detection_result_{original_path.stem}.json")
        if result_file.exists():
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    detection_info = json.load(f)
            except:
                pass
        
        # 生成HTML内容
        html_content += f"""
        <div class="comparison-item">
            <div class="item-header">
                <div class="item-title">📷 图像 {i}: {original_path.name}</div>
                <div class="item-info">
                    文件大小: {original_path.stat().st_size / 1024:.1f} KB | 
                    修改时间: {datetime.fromtimestamp(original_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M')}
                </div>
            </div>
            
            <div class="image-comparison">
                <div class="image-section original-section">
                    <h3>🖼️ 原始图像</h3>
                    <div class="image-container">
"""
        
        if original_b64:
            html_content += f'<img src="data:image/jpeg;base64,{original_b64}" alt="原始图像">'
        else:
            html_content += '<div class="no-image">❌ 无法加载原始图像</div>'
        
        html_content += """
                    </div>
                </div>
                
                <div class="image-section detected-section">
                    <h3>🎯 YOLO检测结果</h3>
                    <div class="image-container">
"""
        
        if annotated_b64:
            html_content += f'<img src="data:image/jpeg;base64,{annotated_b64}" alt="检测结果">'
        else:
            html_content += '<div class="no-image">❌ 未找到检测结果图像</div>'
        
        html_content += f"""
                    </div>
                    
                    <div class="detection-stats">
                        <strong>📊 检测统计</strong>
                        <div class="stats-grid">
                            <div class="stat-item">
                                <div class="stat-value">{detection_info.get('objects_count', 0)}</div>
                                <div class="stat-label">检测物体</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value">{detection_info.get('confidence_avg', 0):.1f}%</div>
                                <div class="stat-label">平均置信度</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value">{detection_info.get('processing_time', 0):.3f}s</div>
                                <div class="stat-label">处理时间</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value">{len(set([obj.get('class', '') for obj in detection_info.get('objects', [])]))}</div>
                                <div class="stat-label">物体类别</div>
                            </div>
                        </div>
                    </div>
"""
        
        # 添加检测详情
        if detection_info.get('objects'):
            html_content += """
                    <div class="detection-details">
                        <strong>🔍 检测详情</strong>
                        <ul class="object-list">
"""
            for obj in detection_info['objects']:
                confidence = obj.get('confidence', 0) * 100
                class_name = obj.get('class', '未知')
                
                # 根据置信度设置样式
                if confidence >= 80:
                    css_class = 'confidence-high'
                elif confidence >= 60:
                    css_class = 'confidence-medium'
                else:
                    css_class = 'confidence-low'
                
                html_content += f'<li class="object-item {css_class}">{class_name} - {confidence:.1f}%</li>'
            
            html_content += """
                        </ul>
                    </div>
"""
        else:
            html_content += f"""
                    <div class="detection-details">
                        <strong>ℹ️ 状态</strong>
                        <p style="margin: 10px 0; color: #666;">{detection_info.get('status', '未检测到任何物体')}</p>
                    </div>
"""
        
        html_content += """
                </div>
            </div>
        </div>
"""
        
        processed_count += 1
    
    # HTML结尾
    html_content += f"""
        <div style="text-align: center; padding: 30px; background: #f8f9fa; color: #666;">
            <p>📈 共处理 {processed_count} 张图像 | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>💡 提示: 绿色边框表示高置信度(≥80%)，橙色表示中等置信度(60-80%)，红色表示低置信度(<60%)</p>
        </div>
    </div>
</body>
</html>
"""
    
    # 保存报告
    report_path = "visual_yolo_comparison_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ 可视化对比报告已生成: {report_path}")
    return report_path

if __name__ == "__main__":
    print("🎨 生成YOLO检测可视化对比报告...")
    report_path = create_visual_comparison_report()
    print(f"🎉 报告生成完成: {report_path}")