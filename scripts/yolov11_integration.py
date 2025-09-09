#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv11集成模块
实现YOLOv11模型的集成和优化
"""

import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Any, Optional, Tuple
import time

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("❌ Ultralytics库不可用，请安装: pip install ultralytics")

class YOLOv11Detector:
    """YOLOv11检测器 - 支持最新的YOLOv11模型"""
    
    def __init__(self, model_size: str = 'n', device: str = 'auto'):
        """
        初始化YOLOv11检测器
        
        Args:
            model_size: 模型大小 ('n', 's', 'm', 'l', 'x')
            device: 设备类型 ('auto', 'cpu', 'cuda')
        """
        self.model_size = model_size
        self.device = device
        self.model = None
        self.model_path = f"yolo11{model_size}.pt"
        self.performance_stats = {
            'total_detections': 0,
            'total_processing_time': 0.0,
            'average_fps': 0.0,
            'confidence_scores': []
        }
        
        if ULTRALYTICS_AVAILABLE:
            self._load_model()
        else:
            print("⚠️ YOLOv11检测器初始化失败：缺少依赖")
    
    def _load_model(self):
        """加载YOLOv11模型"""
        try:
            print(f"🔄 加载YOLOv11{self.model_size}模型...")
            self.model = YOLO(self.model_path)
            
            # 设置设备
            if self.device == 'auto':
                import torch
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            print(f"✅ YOLOv11{self.model_size}模型加载成功 (设备: {self.device})")
            
            # 预热模型
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy_image, device=self.device, verbose=False)
            print("🔥 模型预热完成")
            
        except Exception as e:
            print(f"❌ YOLOv11模型加载失败: {e}")
            self.model = None
    
    def detect(self, image: np.ndarray, conf_threshold: float = 0.25, 
               iou_threshold: float = 0.45) -> Dict[str, Any]:
        """
        使用YOLOv11进行目标检测
        
        Args:
            image: 输入图像
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
            
        Returns:
            检测结果字典
        """
        if self.model is None:
            return self._create_error_result("模型未加载")
        
        start_time = time.time()
        
        try:
            # YOLOv11检测
            results = self.model(
                image,
                conf=conf_threshold,
                iou=iou_threshold,
                device=self.device,
                verbose=False
            )
            
            processing_time = time.time() - start_time
            
            # 解析结果
            detection_result = self._parse_results(results[0], processing_time)
            
            # 更新性能统计
            self._update_performance_stats(detection_result, processing_time)
            
            return detection_result
            
        except Exception as e:
            return self._create_error_result(f"检测失败: {str(e)}")
    
    def _parse_results(self, result, processing_time: float) -> Dict[str, Any]:
        """解析YOLOv11检测结果"""
        detection_info = {
            'model_version': 'YOLOv11',
            'model_size': self.model_size,
            'device': self.device,
            'processing_time': processing_time,
            'objects': [],
            'objects_count': 0,
            'confidence_avg': 0.0,
            'confidence_max': 0.0,
            'image_size': f"{result.orig_shape[1]}x{result.orig_shape[0]}",
            'status': 'success'
        }
        
        if result.boxes is not None and len(result.boxes) > 0:
            confidences = []
            
            for box in result.boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                
                # 边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                obj_info = {
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'class_id': class_id,
                    'area': (x2 - x1) * (y2 - y1)
                }
                
                detection_info['objects'].append(obj_info)
                confidences.append(confidence)
            
            detection_info['objects_count'] = len(detection_info['objects'])
            if confidences:
                detection_info['confidence_avg'] = float(np.mean(confidences)) * 100
                detection_info['confidence_max'] = float(np.max(confidences)) * 100
        
        return detection_info
    
    def _create_error_result(self, error_msg: str) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            'model_version': 'YOLOv11',
            'model_size': self.model_size,
            'error': error_msg,
            'objects': [],
            'objects_count': 0,
            'status': 'error'
        }
    
    def _update_performance_stats(self, result: Dict[str, Any], processing_time: float):
        """更新性能统计"""
        self.performance_stats['total_detections'] += 1
        self.performance_stats['total_processing_time'] += processing_time
        
        if processing_time > 0:
            fps = 1.0 / processing_time
            self.performance_stats['average_fps'] = (
                self.performance_stats['average_fps'] * (self.performance_stats['total_detections'] - 1) + fps
            ) / self.performance_stats['total_detections']
        
        if result.get('objects'):
            for obj in result['objects']:
                self.performance_stats['confidence_scores'].append(obj['confidence'])
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        stats = self.performance_stats.copy()
        
        if stats['confidence_scores']:
            stats['confidence_mean'] = float(np.mean(stats['confidence_scores']))
            stats['confidence_std'] = float(np.std(stats['confidence_scores']))
            stats['confidence_min'] = float(np.min(stats['confidence_scores']))
            stats['confidence_max'] = float(np.max(stats['confidence_scores']))
        
        return stats
    
    def detect_batch(self, images: List[np.ndarray], **kwargs) -> List[Dict[str, Any]]:
        """批量检测"""
        results = []
        for image in images:
            result = self.detect(image, **kwargs)
            results.append(result)
        return results
    
    def create_annotated_image(self, image: np.ndarray, detection_result: Dict[str, Any]) -> np.ndarray:
        """创建带标注的图像"""
        if detection_result.get('status') != 'success' or not detection_result.get('objects'):
            return image.copy()
        
        annotated = image.copy()
        
        for obj in detection_result['objects']:
            bbox = obj['bbox']
            class_name = obj['class']
            confidence = obj['confidence']
            
            # 绘制边界框
            x1, y1, x2, y2 = bbox
            color = self._get_class_color(obj['class_id'])
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"{class_name} {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated
    
    def _get_class_color(self, class_id: int) -> Tuple[int, int, int]:
        """获取类别颜色"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
        ]
        return colors[class_id % len(colors)]

def compare_yolo_versions():
    """比较不同YOLO版本的性能"""
    print("🔄 开始YOLO版本性能比较...")
    
    # 测试图像路径
    test_images_dir = Path("test_images")
    if not test_images_dir.exists():
        print("❌ 测试图像目录不存在")
        return
    
    # 获取测试图像
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(list(test_images_dir.glob(f"*{ext}")))
    
    if not image_files:
        print("❌ 未找到测试图像")
        return
    
    # 限制测试图像数量
    test_images = image_files[:5]
    
    # 初始化检测器
    detectors = {}
    
    # YOLOv8
    try:
        yolov8 = YOLO('yolov8n.pt')
        detectors['YOLOv8n'] = yolov8
        print("✅ YOLOv8n加载成功")
    except Exception as e:
        print(f"❌ YOLOv8n加载失败: {e}")
    
    # YOLOv11
    yolov11_detector = YOLOv11Detector('n')
    if yolov11_detector.model is not None:
        detectors['YOLOv11n'] = yolov11_detector
        print("✅ YOLOv11n加载成功")
    
    if not detectors:
        print("❌ 没有可用的检测器")
        return
    
    # 性能比较结果
    comparison_results = {}
    
    for model_name, detector in detectors.items():
        print(f"\n🔍 测试 {model_name}...")
        
        total_time = 0
        total_objects = 0
        confidences = []
        
        for i, image_path in enumerate(test_images, 1):
            print(f"  [{i}/{len(test_images)}] {image_path.name}")
            
            # 读取图像
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            
            # 检测
            start_time = time.time()
            
            if isinstance(detector, YOLOv11Detector):
                result = detector.detect(image)
                processing_time = result.get('processing_time', 0)
                objects_count = result.get('objects_count', 0)
                if result.get('objects'):
                    confidences.extend([obj['confidence'] for obj in result['objects']])
            else:
                # YOLOv8
                results = detector(image, verbose=False)
                processing_time = time.time() - start_time
                
                result = results[0]
                objects_count = len(result.boxes) if result.boxes is not None else 0
                
                if result.boxes is not None:
                    for box in result.boxes:
                        confidences.append(float(box.conf[0]))
            
            total_time += processing_time
            total_objects += objects_count
        
        # 计算统计信息
        avg_time = total_time / len(test_images)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        avg_confidence = np.mean(confidences) if confidences else 0
        
        comparison_results[model_name] = {
            'average_processing_time': avg_time,
            'average_fps': avg_fps,
            'total_objects_detected': total_objects,
            'average_confidence': avg_confidence,
            'confidence_std': np.std(confidences) if confidences else 0,
            'images_processed': len(test_images)
        }
    
    # 生成比较报告
    generate_comparison_report(comparison_results)
    
    return comparison_results

def generate_comparison_report(results: Dict[str, Dict[str, Any]]):
    """生成YOLO版本比较报告"""
    
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLO版本性能比较报告</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .comparison-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
            padding: 40px;
        }}
        .model-card {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 25px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-left: 4px solid #3498db;
        }}
        .model-name {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 20px;
            text-align: center;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #e0e0e0;
        }}
        .metric:last-child {{
            border-bottom: none;
        }}
        .metric-label {{
            color: #666;
            font-weight: 500;
        }}
        .metric-value {{
            font-weight: bold;
            color: #2c3e50;
        }}
        .performance-summary {{
            background: #e8f5e8;
            padding: 30px;
            text-align: center;
        }}
        .summary-title {{
            font-size: 1.8em;
            color: #27ae60;
            margin-bottom: 20px;
        }}
        .winner {{
            border-left-color: #27ae60;
            background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%);
        }}
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 YOLO版本性能比较报告</h1>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="comparison-grid">
"""
    
    # 找出最佳性能
    best_fps = max(results.values(), key=lambda x: x['average_fps'])['average_fps']
    best_confidence = max(results.values(), key=lambda x: x['average_confidence'])['average_confidence']
    
    for model_name, stats in results.items():
        is_winner = (stats['average_fps'] == best_fps or 
                    stats['average_confidence'] == best_confidence)
        
        card_class = "model-card winner" if is_winner else "model-card"
        
        html_content += f"""
            <div class="{card_class}">
                <div class="model-name">{model_name}</div>
                <div class="metric">
                    <span class="metric-label">平均处理时间</span>
                    <span class="metric-value">{stats['average_processing_time']:.3f}s</span>
                </div>
                <div class="metric">
                    <span class="metric-label">平均FPS</span>
                    <span class="metric-value">{stats['average_fps']:.1f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">检测物体总数</span>
                    <span class="metric-value">{stats['total_objects_detected']}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">平均置信度</span>
                    <span class="metric-value">{stats['average_confidence']:.1%}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">置信度标准差</span>
                    <span class="metric-value">{stats['confidence_std']:.3f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">处理图像数</span>
                    <span class="metric-value">{stats['images_processed']}</span>
                </div>
            </div>
"""
    
    html_content += f"""
        </div>
        
        <div class="performance-summary">
            <div class="summary-title">📊 性能总结</div>
            <p><strong>测试结论:</strong> 基于 {list(results.values())[0]['images_processed']} 张测试图像的性能比较</p>
            <p><strong>最佳FPS:</strong> {best_fps:.1f} | <strong>最佳置信度:</strong> {best_confidence:.1%}</p>
        </div>
        
        <div class="footer">
            <p>🔧 技术说明: 测试在相同硬件环境下进行，结果仅供参考</p>
            <p>📈 建议: 根据具体应用场景选择合适的模型版本</p>
        </div>
    </div>
</body>
</html>
"""
    
    # 保存报告
    report_path = "yolo_version_comparison_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ YOLO版本比较报告已生成: {report_path}")

if __name__ == "__main__":
    print("🚀 YOLOv11集成测试开始...")
    
    # 测试YOLOv11检测器
    detector = YOLOv11Detector('n')
    
    if detector.model is not None:
        print("✅ YOLOv11检测器初始化成功")
        
        # 运行版本比较
        comparison_results = compare_yolo_versions()
        
        if comparison_results:
            print("\n📊 性能比较完成!")
            for model, stats in comparison_results.items():
                print(f"{model}: {stats['average_fps']:.1f} FPS, {stats['average_confidence']:.1%} 置信度")
    else:
        print("❌ YOLOv11检测器初始化失败")