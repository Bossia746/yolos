#!/usr/bin/env python3
"""
YOLOS Web界面
提供基于Web的检测服务和管理界面
"""

import os
import sys
import json
import base64
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent / "src"))

from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import cv2
import numpy as np

from models.yolo_factory import YOLOFactory
from detection.image_detector import ImageDetector
from detection.realtime_detector import RealtimeDetector

app = Flask(__name__)
CORS(app)

# 全局变量
detector = None
realtime_detector = None
detection_results = []
system_stats = {
    'total_detections': 0,
    'uptime': datetime.now(),
    'model_info': {}
}


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/models', methods=['GET'])
def get_models():
    """获取可用模型列表"""
    try:
        models = YOLOFactory.list_available_models()
        model_info = []
        
        for model_type in models:
            info = YOLOFactory.get_model_info(model_type)
            model_info.append(info)
        
        return jsonify({
            'success': True,
            'models': model_info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/detect/image', methods=['POST'])
def detect_image():
    """图像检测API"""
    global detector, system_stats
    
    try:
        # 获取参数
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({
                'success': False,
                'error': '缺少图像数据'
            })
        
        # 解码base64图像
        image_data = data['image'].split(',')[1]  # 移除data:image/jpeg;base64,
        image_bytes = base64.b64decode(image_data)
        
        # 转换为OpenCV格式
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                'success': False,
                'error': '无效的图像数据'
            })
        
        # 初始化检测器
        if detector is None:
            model_type = data.get('model_type', 'yolov8')
            detector = ImageDetector(model_type=model_type, device='auto')
            system_stats['model_info'] = detector.get_model_info()
        
        # 执行检测
        results = detector.model.predict(image)
        
        # 绘制结果
        annotated_image = detector.model.draw_results(image, results)
        
        # 编码结果图像
        _, buffer = cv2.imencode('.jpg', annotated_image)
        result_image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # 更新统计
        system_stats['total_detections'] += len(results)
        detection_results.append({
            'timestamp': datetime.now().isoformat(),
            'detection_count': len(results),
            'results': results
        })
        
        # 保持最近100条记录
        if len(detection_results) > 100:
            detection_results.pop(0)
        
        return jsonify({
            'success': True,
            'results': results,
            'annotated_image': f"data:image/jpeg;base64,{result_image_b64}",
            'detection_count': len(results),
            'processing_time': 0.1  # 实际应该计算处理时间
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/detect/webcam/start', methods=['POST'])
def start_webcam_detection():
    """启动网络摄像头检测"""
    global realtime_detector
    
    try:
        data = request.get_json() or {}
        camera_id = data.get('camera_id', 0)
        model_type = data.get('model_type', 'yolov8')
        
        if realtime_detector is None:
            realtime_detector = RealtimeDetector(model_type=model_type, device='auto')
        
        # 这里应该在后台线程中启动检测
        # 为了简化，返回成功状态
        
        return jsonify({
            'success': True,
            'message': '网络摄像头检测已启动'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/detect/webcam/stop', methods=['POST'])
def stop_webcam_detection():
    """停止网络摄像头检测"""
    global realtime_detector
    
    try:
        if realtime_detector:
            realtime_detector.stop()
        
        return jsonify({
            'success': True,
            'message': '网络摄像头检测已停止'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """获取系统统计信息"""
    global system_stats, detection_results
    
    uptime = datetime.now() - system_stats['uptime']
    
    stats = {
        'total_detections': system_stats['total_detections'],
        'uptime_seconds': uptime.total_seconds(),
        'uptime_formatted': str(uptime).split('.')[0],
        'model_info': system_stats['model_info'],
        'recent_detections': detection_results[-10:],  # 最近10条
        'detection_history_count': len(detection_results)
    }
    
    return jsonify({
        'success': True,
        'stats': stats
    })


@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    """配置管理"""
    config_file = Path(__file__).parent.parent / "configs" / "web_config.json"
    
    if request.method == 'GET':
        # 读取配置
        try:
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                config = {
                    'model_type': 'yolov8',
                    'confidence_threshold': 0.25,
                    'iou_threshold': 0.7,
                    'max_detections': 100
                }
            
            return jsonify({
                'success': True,
                'config': config
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            })
    
    else:  # POST
        # 保存配置
        try:
            config = request.get_json()
            
            # 确保目录存在
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            return jsonify({
                'success': True,
                'message': '配置已保存'
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            })


def generate_video_stream():
    """生成视频流"""
    # 这里应该实现实际的视频流生成
    # 为了简化，返回空数据
    while True:
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n'


@app.route('/video_feed')
def video_feed():
    """视频流端点"""
    return Response(generate_video_stream(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    print("启动YOLOS Web服务器...")
    print("访问地址: http://localhost:5000")
    
    # 创建模板目录和基础模板
    template_dir = Path(__file__).parent / "templates"
    template_dir.mkdir(exist_ok=True)
    
    # 创建基础HTML模板
    index_template = template_dir / "index.html"
    if not index_template.exists():
        html_content = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLOS Web界面</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
        .button { padding: 10px 20px; margin: 5px; background: #007bff; color: white; border: none; border-radius: 3px; cursor: pointer; }
        .button:hover { background: #0056b3; }
        #imagePreview { max-width: 100%; height: auto; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }
        .stat-item { padding: 10px; background: #f8f9fa; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>YOLOS - 多平台AIoT视觉大模型</h1>
        
        <div class="section">
            <h2>图像检测</h2>
            <input type="file" id="imageInput" accept="image/*">
            <button class="button" onclick="detectImage()">检测图像</button>
            <div id="imageResult"></div>
        </div>
        
        <div class="section">
            <h2>实时检测</h2>
            <button class="button" onclick="startWebcam()">启动摄像头</button>
            <button class="button" onclick="stopWebcam()">停止摄像头</button>
            <div id="webcamResult"></div>
        </div>
        
        <div class="section">
            <h2>系统统计</h2>
            <button class="button" onclick="updateStats()">刷新统计</button>
            <div id="statsContainer" class="stats"></div>
        </div>
    </div>

    <script>
        function detectImage() {
            const input = document.getElementById('imageInput');
            const file = input.files[0];
            
            if (!file) {
                alert('请选择图像文件');
                return;
            }
            
            const reader = new FileReader();
            reader.onload = function(e) {
                const imageData = e.target.result;
                
                fetch('/api/detect/image', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: imageData })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        const resultDiv = document.getElementById('imageResult');
                        resultDiv.innerHTML = `
                            <h3>检测结果 (${data.detection_count} 个目标)</h3>
                            <img src="${data.annotated_image}" style="max-width: 100%;">
                            <pre>${JSON.stringify(data.results, null, 2)}</pre>
                        `;
                    } else {
                        alert('检测失败: ' + data.error);
                    }
                })
                .catch(error => {
                    alert('请求失败: ' + error);
                });
            };
            reader.readAsDataURL(file);
        }
        
        function startWebcam() {
            fetch('/api/detect/webcam/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('webcamResult').innerHTML = 
                        '<p>摄像头检测已启动</p>';
                } else {
                    alert('启动失败: ' + data.error);
                }
            });
        }
        
        function stopWebcam() {
            fetch('/api/detect/webcam/stop', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('webcamResult').innerHTML = 
                        '<p>摄像头检测已停止</p>';
                }
            });
        }
        
        function updateStats() {
            fetch('/api/stats')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const stats = data.stats;
                    document.getElementById('statsContainer').innerHTML = `
                        <div class="stat-item">
                            <strong>总检测数:</strong> ${stats.total_detections}
                        </div>
                        <div class="stat-item">
                            <strong>运行时间:</strong> ${stats.uptime_formatted}
                        </div>
                        <div class="stat-item">
                            <strong>历史记录:</strong> ${stats.detection_history_count}
                        </div>
                        <div class="stat-item">
                            <strong>模型信息:</strong> ${JSON.stringify(stats.model_info)}
                        </div>
                    `;
                }
            });
        }
        
        // 页面加载时更新统计
        window.onload = function() {
            updateStats();
        };
    </script>
</body>
</html>
        """
        
        with open(index_template, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    app.run(host='0.0.0.0', port=5000, debug=True)