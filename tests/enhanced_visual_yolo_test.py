#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版YOLO可视化测试
生成包含原图和检测结果对比的完整报告
"""

import os
import cv2
import base64
import json
from datetime import datetime
from pathlib import Path
import numpy as np

try:
    from ultralytics import YOLO
    yolo_available = True
    print("✓ Ultralytics YOLO库可用")
except ImportError:
    yolo_available = False
    print("❌ Ultralytics YOLO库不可用")

def image_to_base64(image_path):
    """将图像转换为base64编码"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"❌ 转换图像失败 {image_path}: {e}")
        return None

def detect_with_yolo(image_path, model):
    """使用YOLO进行检测并返回详细结果"""
    try:
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            return None, None, {"error": "无法读取图像"}
        
        # YOLO检测
        results = model(image)
        result = results[0]
        
        # 提取检测信息
        detection_info = {
            'objects': [],
            'objects_count': 0,
            'confidence_avg': 0.0,
            'confidence_max': 0.0,
            'processing_time': 0.0,
            'image_size': f"{image.shape[1]}x{image.shape[0]}",
            'status': 'success'
        }
        
        if result.boxes is not None and len(result.boxes) > 0:
            confidences = []
            for box in result.boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                # 边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                obj_info = {
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2],
                    'class_id': class_id
                }
                detection_info['objects'].append(obj_info)
                confidences.append(confidence)
            
            detection_info['objects_count'] = len(detection_info['objects'])
            if confidences:
                detection_info['confidence_avg'] = float(np.mean(confidences)) * 100
                detection_info['confidence_max'] = float(np.max(confidences)) * 100
        
        # 生成带标注的图像
        annotated_image = result.plot()
        
        # 保存标注图像
        annotated_path = f"annotated_{Path(image_path).stem}.jpg"
        cv2.imwrite(annotated_path, annotated_image)
        
        return annotated_path, annotated_image, detection_info
        
    except Exception as e:
        return None, None, {"error": str(e)}

def create_enhanced_visual_report():
    """创建增强版可视化报告"""
    
    # 初始化YOLO模型
    model = None
    global yolo_available
    if yolo_available:
        try:
            model = YOLO('yolov8n.pt')
            print("✓ YOLO模型加载成功")
        except Exception as e:
            print(f"❌ YOLO模型加载失败: {e}")
            yolo_available = False
    
    # 图像路径
    image_dir = Path("test_images")
    
    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    original_images = []
    
    if image_dir.exists():
        for ext in image_extensions:
            original_images.extend(list(image_dir.glob(f"*{ext}")))
            original_images.extend(list(image_dir.glob(f"*{ext.upper()}")))
    
    # 去重
    original_images = list(set(original_images))
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
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
            position: relative;
        }}
        .header::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="75" cy="75" r="1" fill="rgba(255,255,255,0.1)"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
            opacity: 0.3;
        }}
        .header h1 {{
            margin: 0;
            font-size: 3em;
            font-weight: 300;
            position: relative;
            z-index: 1;
        }}
        .header p {{
            margin: 15px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
            position: relative;
            z-index: 1;
        }}
        .summary {{
            background: #f8f9fa;
            padding: 30px;
            border-bottom: 1px solid #e0e0e0;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-left: 4px solid #3498db;
        }}
        .summary-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        .summary-label {{
            color: #666;
            font-size: 0.9em;
        }}
        .comparison-item {{
            margin: 0;
            border-bottom: 1px solid #f0f0f0;
        }}
        .comparison-item:last-child {{
            border-bottom: none;
        }}
        .item-header {{
            background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 25px 30px;
            border-bottom: 1px solid #e0e0e0;
        }}
        .item-title {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
            margin: 0 0 10px 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .item-info {{
            color: #666;
            font-size: 0.95em;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        .image-comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            min-height: 500px;
        }}
        .image-section {{
            padding: 30px;
            display: flex;
            flex-direction: column;
        }}
        .image-section h3 {{
            margin: 0 0 20px 0;
            color: #34495e;
            font-size: 1.3em;
            text-align: center;
            padding: 15px;
            border-radius: 8px;
            font-weight: 600;
        }}
        .original-section {{
            border-right: 1px solid #e0e0e0;
        }}
        .original-section h3 {{
            background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%);
            color: #155724;
            border: 1px solid #c3e6cb;
        }}
        .detected-section h3 {{
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            color: #0d47a1;
            border: 1px solid #90caf9;
        }}
        .image-container {{
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 2px dashed #ddd;
            border-radius: 10px;
            padding: 20px;
            background: #fafafa;
            transition: all 0.3s ease;
        }}
        .image-container:hover {{
            border-color: #3498db;
            background: #f0f8ff;
        }}
        .image-container img {{
            max-width: 100%;
            max-height: 400px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transition: transform 0.3s ease;
        }}
        .image-container img:hover {{
            transform: scale(1.02);
        }}
        .no-image {{
            color: #999;
            font-style: italic;
            text-align: center;
            padding: 40px;
        }}
        .detection-stats {{
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 20px;
            margin: 20px 0;
            border-radius: 10px;
            border-left: 4px solid #3498db;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .stat-item {{
            text-align: center;
            padding: 15px;
            background: white;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .stat-value {{
            font-size: 1.4em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .stat-label {{
            font-size: 0.85em;
            color: #666;
            margin-top: 5px;
        }}
        .detection-details {{
            margin-top: 20px;
            padding: 15px;
            background: white;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }}
        .object-list {{
            list-style: none;
            padding: 0;
            margin: 15px 0;
            display: grid;
            gap: 8px;
        }}
        .object-item {{
            padding: 12px 15px;
            border-radius: 6px;
            font-size: 0.95em;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.3s ease;
        }}
        .object-item:hover {{
            transform: translateX(5px);
        }}
        .confidence-high {{ 
            border-left: 4px solid #27ae60; 
            background: linear-gradient(90deg, #d4edda 0%, #f0fff0 100%);
            color: #155724;
        }}
        .confidence-medium {{ 
            border-left: 4px solid #f39c12; 
            background: linear-gradient(90deg, #fff3cd 0%, #fffbf0 100%);
            color: #856404;
        }}
        .confidence-low {{ 
            border-left: 4px solid #e74c3c; 
            background: linear-gradient(90deg, #f8d7da 0%, #fff0f0 100%);
            color: #721c24;
        }}
        .confidence-badge {{
            background: rgba(255,255,255,0.8);
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .footer {{
            text-align: center;
            padding: 40px;
            background: #f8f9fa;
            color: #666;
            border-top: 1px solid #e0e0e0;
        }}
        
        @media (max-width: 768px) {{
            .image-comparison {{
                grid-template-columns: 1fr;
            }}
            .original-section {{
                border-right: none;
                border-bottom: 1px solid #e0e0e0;
            }}
            .container {{
                margin: 10px;
                border-radius: 10px;
            }}
            .header {{
                padding: 30px 20px;
            }}
            .header h1 {{
                font-size: 2.2em;
            }}
            .item-info {{
                flex-direction: column;
                gap: 10px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 YOLO检测可视化对比报告</h1>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>原图 vs 检测结果完整对比分析</p>
        </div>
        
        <div class="summary">
            <h2 style="margin: 0 0 20px 0; color: #2c3e50;">📊 检测概览</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="summary-value" id="total-images">0</div>
                    <div class="summary-label">处理图像</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value" id="total-objects">0</div>
                    <div class="summary-label">检测物体</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value" id="avg-confidence">0%</div>
                    <div class="summary-label">平均置信度</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value" id="unique-classes">0</div>
                    <div class="summary-label">物体类别</div>
                </div>
            </div>
        </div>
"""
    
    # 处理每张图像
    processed_count = 0
    total_objects = 0
    all_confidences = []
    all_classes = set()
    
    for i, original_path in enumerate(original_images[:8], 1):  # 限制处理前8张图像
        print(f"[{i}] 处理图像: {original_path.name}")
        
        # 进行YOLO检测
        annotated_path = None
        detection_info = {
            'objects_count': 0,
            'confidence_avg': 0.0,
            'confidence_max': 0.0,
            'processing_time': 0.0,
            'objects': [],
            'status': '检测失败',
            'image_size': '未知'
        }
        
        if yolo_available and model:
            start_time = datetime.now()
            annotated_path, annotated_image, detection_info = detect_with_yolo(original_path, model)
            end_time = datetime.now()
            detection_info['processing_time'] = (end_time - start_time).total_seconds()
        
        # 转换图像为base64
        original_b64 = image_to_base64(original_path)
        annotated_b64 = None
        
        if annotated_path and Path(annotated_path).exists():
            annotated_b64 = image_to_base64(annotated_path)
        
        # 更新统计信息
        if 'error' not in detection_info:
            total_objects += detection_info.get('objects_count', 0)
            if detection_info.get('objects'):
                for obj in detection_info['objects']:
                    all_confidences.append(obj.get('confidence', 0) * 100)
                    all_classes.add(obj.get('class', ''))
        
        # 生成HTML内容
        file_size = original_path.stat().st_size / 1024
        mod_time = datetime.fromtimestamp(original_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
        
        html_content += f"""
        <div class="comparison-item">
            <div class="item-header">
                <div class="item-title">
                    <span>📷</span>
                    <span>图像 {i}: {original_path.name}</span>
                </div>
                <div class="item-info">
                    <span>📁 文件大小: {file_size:.1f} KB</span>
                    <span>🕒 修改时间: {mod_time}</span>
                    <span>📐 图像尺寸: {detection_info.get('image_size', '未知')}</span>
                    <span>⚡ 处理时间: {detection_info.get('processing_time', 0):.3f}s</span>
                </div>
            </div>
            
            <div class="image-comparison">
                <div class="image-section original-section">
                    <h3>🖼️ 原始图像</h3>
                    <div class="image-container">
"""
        
        if original_b64:
            html_content += f'<img src="data:image/jpeg;base64,{original_b64}" alt="原始图像 {original_path.name}">'
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
            html_content += f'<img src="data:image/jpeg;base64,{annotated_b64}" alt="检测结果 {original_path.name}">'
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
                                <div class="stat-value">{detection_info.get('confidence_max', 0):.1f}%</div>
                                <div class="stat-label">最高置信度</div>
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
                bbox = obj.get('bbox', [0, 0, 0, 0])
                
                # 根据置信度设置样式
                if confidence >= 80:
                    css_class = 'confidence-high'
                elif confidence >= 60:
                    css_class = 'confidence-medium'
                else:
                    css_class = 'confidence-low'
                
                html_content += f"""
                        <li class="object-item {css_class}">
                            <span>{class_name}</span>
                            <span class="confidence-badge">{confidence:.1f}%</span>
                        </li>"""
            
            html_content += """
                        </ul>
                    </div>
"""
        else:
            status_msg = detection_info.get('status', '未检测到任何物体')
            if 'error' in detection_info:
                status_msg = f"❌ 检测错误: {detection_info['error']}"
            
            html_content += f"""
                    <div class="detection-details">
                        <strong>ℹ️ 状态</strong>
                        <p style="margin: 15px 0; color: #666; text-align: center; padding: 20px; background: #f8f9fa; border-radius: 5px;">{status_msg}</p>
                    </div>
"""
        
        html_content += """
                </div>
            </div>
        </div>
"""
        
        processed_count += 1
    
    # 计算总体统计
    avg_confidence = np.mean(all_confidences) if all_confidences else 0
    
    # HTML结尾
    html_content += f"""
        <div class="footer">
            <h3 style="color: #2c3e50; margin-bottom: 20px;">📈 检测总结</h3>
            <p><strong>处理图像:</strong> {processed_count} 张 | <strong>检测物体:</strong> {total_objects} 个 | <strong>平均置信度:</strong> {avg_confidence:.1f}% | <strong>物体类别:</strong> {len(all_classes)} 种</p>
            <p style="margin-top: 15px; font-size: 0.9em;">💡 <strong>置信度说明:</strong> 绿色表示高置信度(≥80%)，橙色表示中等置信度(60-80%)，红色表示低置信度(<60%)</p>
            <p style="margin-top: 10px; font-size: 0.9em;">🔧 <strong>技术信息:</strong> 使用YOLOv8n模型 | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
    
    <script>
        // 更新概览统计
        document.getElementById('total-images').textContent = '{processed_count}';
        document.getElementById('total-objects').textContent = '{total_objects}';
        document.getElementById('avg-confidence').textContent = '{avg_confidence:.1f}%';
        document.getElementById('unique-classes').textContent = '{len(all_classes)}';
    </script>
</body>
</html>
"""
    
    # 保存报告
    report_path = "enhanced_visual_yolo_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ 增强版可视化报告已生成: {report_path}")
    print(f"📊 统计信息: {processed_count}张图像, {total_objects}个物体, {len(all_classes)}种类别")
    
    return report_path

if __name__ == "__main__":
    print("🎨 生成增强版YOLO检测可视化报告...")
    report_path = create_enhanced_visual_report()
    print(f"🎉 报告生成完成: {report_path}")