#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测精度优化模块
实现多种检测精度提升策略
"""

import cv2
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

@dataclass
class OptimizationConfig:
    """优化配置"""
    # 图像预处理
    enable_preprocessing: bool = True
    resize_strategy: str = "adaptive"  # "fixed", "adaptive", "multi_scale"
    target_size: int = 640
    
    # 检测参数
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_det: int = 300
    
    # 后处理优化
    enable_nms_optimization: bool = True
    enable_confidence_calibration: bool = True
    
    # 多尺度检测
    enable_multi_scale: bool = False
    scales: List[int] = None
    
    # 测试时增强 (TTA)
    enable_tta: bool = False
    tta_scales: List[float] = None
    tta_flips: List[bool] = None

class DetectionAccuracyOptimizer:
    """检测精度优化器"""
    
    def __init__(self, model_path: str = "yolov8n.pt", config: OptimizationConfig = None):
        self.model_path = model_path
        self.config = config or OptimizationConfig()
        self.model = None
        self.optimization_stats = {
            'total_optimizations': 0,
            'accuracy_improvements': [],
            'processing_time_overhead': []
        }
        
        # 设置默认值
        if self.config.scales is None:
            self.config.scales = [480, 640, 800]
        if self.config.tta_scales is None:
            self.config.tta_scales = [0.8, 1.0, 1.2]
        if self.config.tta_flips is None:
            self.config.tta_flips = [False, True]
        
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        if not ULTRALYTICS_AVAILABLE:
            print("❌ Ultralytics库不可用")
            return
        
        try:
            self.model = YOLO(self.model_path)
            print(f"✅ 模型加载成功: {self.model_path}")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
    
    def optimize_detection(self, image: np.ndarray) -> Dict[str, Any]:
        """优化检测流程"""
        if self.model is None:
            return self._create_error_result("模型未加载")
        
        start_time = time.time()
        
        # 1. 图像预处理优化
        if self.config.enable_preprocessing:
            processed_image = self._preprocess_image(image)
        else:
            processed_image = image
        
        # 2. 多尺度检测
        if self.config.enable_multi_scale:
            detection_result = self._multi_scale_detection(processed_image)
        elif self.config.enable_tta:
            detection_result = self._tta_detection(processed_image)
        else:
            detection_result = self._single_scale_detection(processed_image)
        
        # 3. 后处理优化
        if self.config.enable_nms_optimization:
            detection_result = self._optimize_nms(detection_result)
        
        if self.config.enable_confidence_calibration:
            detection_result = self._calibrate_confidence(detection_result)
        
        processing_time = time.time() - start_time
        detection_result['optimization_time'] = processing_time
        detection_result['optimization_config'] = self.config.__dict__
        
        # 更新统计
        self._update_stats(detection_result, processing_time)
        
        return detection_result
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """图像预处理优化"""
        processed = image.copy()
        
        # 1. 自适应直方图均衡化
        if len(processed.shape) == 3:
            # 转换到LAB色彩空间
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # 对L通道进行CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # 合并通道
            processed = cv2.merge([l, a, b])
            processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)
        
        # 2. 去噪
        processed = cv2.bilateralFilter(processed, 9, 75, 75)
        
        # 3. 锐化
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        processed = cv2.filter2D(processed, -1, kernel)
        
        # 4. 尺寸调整策略
        if self.config.resize_strategy == "adaptive":
            processed = self._adaptive_resize(processed)
        elif self.config.resize_strategy == "fixed":
            processed = cv2.resize(processed, (self.config.target_size, self.config.target_size))
        
        return processed
    
    def _adaptive_resize(self, image: np.ndarray) -> np.ndarray:
        """自适应尺寸调整"""
        h, w = image.shape[:2]
        
        # 计算最佳尺寸
        max_size = max(h, w)
        if max_size > self.config.target_size:
            scale = self.config.target_size / max_size
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # 确保尺寸是32的倍数（YOLO要求）
            new_w = (new_w // 32) * 32
            new_h = (new_h // 32) * 32
            
            return cv2.resize(image, (new_w, new_h))
        
        return image
    
    def _single_scale_detection(self, image: np.ndarray) -> Dict[str, Any]:
        """单尺度检测"""
        results = self.model(
            image,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            max_det=self.config.max_det,
            verbose=False
        )
        
        return self._parse_yolo_results(results[0])
    
    def _multi_scale_detection(self, image: np.ndarray) -> Dict[str, Any]:
        """多尺度检测"""
        all_detections = []
        
        for scale in self.config.scales:
            # 调整图像尺寸
            h, w = image.shape[:2]
            scale_factor = scale / max(h, w)
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            
            # 确保尺寸是32的倍数
            new_w = (new_w // 32) * 32
            new_h = (new_h // 32) * 32
            
            scaled_image = cv2.resize(image, (new_w, new_h))
            
            # 检测
            results = self.model(
                scaled_image,
                conf=self.config.conf_threshold * 0.8,  # 降低阈值
                iou=self.config.iou_threshold,
                max_det=self.config.max_det,
                verbose=False
            )
            
            # 将检测结果缩放回原始尺寸
            scale_back_x = w / new_w
            scale_back_y = h / new_h
            
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # 缩放坐标
                    x1 *= scale_back_x
                    y1 *= scale_back_y
                    x2 *= scale_back_x
                    y2 *= scale_back_y
                    
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(box.conf[0]),
                        'class_id': int(box.cls[0]),
                        'class': self.model.names[int(box.cls[0])],
                        'scale': scale
                    }
                    all_detections.append(detection)
        
        # 合并多尺度检测结果
        merged_detections = self._merge_multi_scale_detections(all_detections)
        
        return {
            'objects': merged_detections,
            'objects_count': len(merged_detections),
            'method': 'multi_scale',
            'scales_used': self.config.scales,
            'status': 'success'
        }
    
    def _tta_detection(self, image: np.ndarray) -> Dict[str, Any]:
        """测试时增强检测"""
        all_detections = []
        
        for scale in self.config.tta_scales:
            for flip in self.config.tta_flips:
                # 应用变换
                transformed_image = image.copy()
                
                # 缩放
                if scale != 1.0:
                    h, w = image.shape[:2]
                    new_h, new_w = int(h * scale), int(w * scale)
                    transformed_image = cv2.resize(transformed_image, (new_w, new_h))
                
                # 翻转
                if flip:
                    transformed_image = cv2.flip(transformed_image, 1)
                
                # 检测
                results = self.model(
                    transformed_image,
                    conf=self.config.conf_threshold * 0.9,
                    iou=self.config.iou_threshold,
                    max_det=self.config.max_det,
                    verbose=False
                )
                
                # 逆变换检测结果
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        # 逆缩放
                        if scale != 1.0:
                            x1 /= scale
                            y1 /= scale
                            x2 /= scale
                            y2 /= scale
                        
                        # 逆翻转
                        if flip:
                            img_w = image.shape[1]
                            x1, x2 = img_w - x2, img_w - x1
                        
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(box.conf[0]),
                            'class_id': int(box.cls[0]),
                            'class': self.model.names[int(box.cls[0])],
                            'tta_params': {'scale': scale, 'flip': flip}
                        }
                        all_detections.append(detection)
        
        # 合并TTA结果
        merged_detections = self._merge_tta_detections(all_detections)
        
        return {
            'objects': merged_detections,
            'objects_count': len(merged_detections),
            'method': 'tta',
            'tta_config': {
                'scales': self.config.tta_scales,
                'flips': self.config.tta_flips
            },
            'status': 'success'
        }
    
    def _merge_multi_scale_detections(self, detections: List[Dict]) -> List[Dict]:
        """合并多尺度检测结果"""
        if not detections:
            return []
        
        # 按类别分组
        class_groups = {}
        for det in detections:
            class_id = det['class_id']
            if class_id not in class_groups:
                class_groups[class_id] = []
            class_groups[class_id].append(det)
        
        merged = []
        for class_id, group in class_groups.items():
            # 对每个类别应用NMS
            boxes = np.array([det['bbox'] for det in group])
            scores = np.array([det['confidence'] for det in group])
            
            # OpenCV NMS
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(), scores.tolist(), 
                self.config.conf_threshold, self.config.iou_threshold
            )
            
            if len(indices) > 0:
                for i in indices.flatten():
                    merged.append(group[i])
        
        return merged
    
    def _merge_tta_detections(self, detections: List[Dict]) -> List[Dict]:
        """合并TTA检测结果"""
        if not detections:
            return []
        
        # 使用加权平均合并相似检测
        merged = []
        used_indices = set()
        
        for i, det1 in enumerate(detections):
            if i in used_indices:
                continue
            
            similar_detections = [det1]
            used_indices.add(i)
            
            for j, det2 in enumerate(detections[i+1:], i+1):
                if j in used_indices:
                    continue
                
                # 检查是否为相似检测
                if (det1['class_id'] == det2['class_id'] and 
                    self._calculate_iou(det1['bbox'], det2['bbox']) > 0.5):
                    similar_detections.append(det2)
                    used_indices.add(j)
            
            # 合并相似检测
            if len(similar_detections) > 1:
                merged_det = self._average_detections(similar_detections)
                merged.append(merged_det)
            else:
                merged.append(det1)
        
        return merged
    
    def _average_detections(self, detections: List[Dict]) -> Dict:
        """平均多个检测结果"""
        weights = [det['confidence'] for det in detections]
        total_weight = sum(weights)
        
        # 加权平均边界框
        avg_bbox = [0, 0, 0, 0]
        for det, weight in zip(detections, weights):
            for i in range(4):
                avg_bbox[i] += det['bbox'][i] * weight / total_weight
        
        # 平均置信度
        avg_confidence = sum(weights) / len(weights)
        
        return {
            'bbox': avg_bbox,
            'confidence': avg_confidence,
            'class_id': detections[0]['class_id'],
            'class': detections[0]['class'],
            'merged_count': len(detections)
        }
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """计算IoU"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 计算交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 计算并集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _optimize_nms(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """优化NMS"""
        if not detection_result.get('objects'):
            return detection_result
        
        # 实现更精细的NMS策略
        objects = detection_result['objects']
        
        # 按类别分组
        class_groups = {}
        for obj in objects:
            class_id = obj['class_id']
            if class_id not in class_groups:
                class_groups[class_id] = []
            class_groups[class_id].append(obj)
        
        optimized_objects = []
        for class_id, group in class_groups.items():
            if len(group) <= 1:
                optimized_objects.extend(group)
                continue
            
            # 软NMS实现
            group_sorted = sorted(group, key=lambda x: x['confidence'], reverse=True)
            keep = []
            
            while group_sorted:
                current = group_sorted.pop(0)
                keep.append(current)
                
                # 降低重叠检测的置信度
                remaining = []
                for det in group_sorted:
                    iou = self._calculate_iou(current['bbox'], det['bbox'])
                    if iou > 0.3:  # 软阈值
                        # 降低置信度而不是完全删除
                        det['confidence'] *= (1 - iou)
                    
                    if det['confidence'] > self.config.conf_threshold * 0.5:
                        remaining.append(det)
                
                group_sorted = remaining
            
            optimized_objects.extend(keep)
        
        detection_result['objects'] = optimized_objects
        detection_result['objects_count'] = len(optimized_objects)
        detection_result['nms_optimized'] = True
        
        return detection_result
    
    def _calibrate_confidence(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """置信度校准"""
        if not detection_result.get('objects'):
            return detection_result
        
        for obj in detection_result['objects']:
            original_conf = obj['confidence']
            
            # 基于边界框大小的校准
            bbox = obj['bbox']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            # 面积校准因子
            if area < 1000:  # 小物体
                area_factor = 0.9
            elif area > 50000:  # 大物体
                area_factor = 1.1
            else:
                area_factor = 1.0
            
            # 应用校准
            calibrated_conf = original_conf * area_factor
            calibrated_conf = max(0.0, min(1.0, calibrated_conf))  # 限制在[0,1]
            
            obj['confidence'] = calibrated_conf
            obj['original_confidence'] = original_conf
        
        detection_result['confidence_calibrated'] = True
        return detection_result
    
    def _parse_yolo_results(self, result) -> Dict[str, Any]:
        """解析YOLO结果"""
        detection_info = {
            'objects': [],
            'objects_count': 0,
            'status': 'success'
        }
        
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                obj_info = {
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2],
                    'class_id': class_id
                }
                
                detection_info['objects'].append(obj_info)
            
            detection_info['objects_count'] = len(detection_info['objects'])
        
        return detection_info
    
    def _create_error_result(self, error_msg: str) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            'error': error_msg,
            'objects': [],
            'objects_count': 0,
            'status': 'error'
        }
    
    def _update_stats(self, result: Dict[str, Any], processing_time: float):
        """更新统计信息"""
        self.optimization_stats['total_optimizations'] += 1
        self.optimization_stats['processing_time_overhead'].append(processing_time)
        
        if result.get('status') == 'success':
            confidence_scores = [obj['confidence'] for obj in result.get('objects', [])]
            if confidence_scores:
                avg_confidence = np.mean(confidence_scores)
                self.optimization_stats['accuracy_improvements'].append(avg_confidence)
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        stats = self.optimization_stats.copy()
        
        if stats['accuracy_improvements']:
            stats['average_confidence'] = float(np.mean(stats['accuracy_improvements']))
            stats['confidence_std'] = float(np.std(stats['accuracy_improvements']))
        
        if stats['processing_time_overhead']:
            stats['average_overhead'] = float(np.mean(stats['processing_time_overhead']))
        
        return stats

def test_accuracy_optimization():
    """测试精度优化"""
    print("🔄 开始检测精度优化测试...")
    
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
    
    test_images = image_files[:3]  # 限制测试图像数量
    
    # 不同优化配置
    configs = {
        'baseline': OptimizationConfig(
            enable_preprocessing=False,
            enable_multi_scale=False,
            enable_tta=False
        ),
        'preprocessing_only': OptimizationConfig(
            enable_preprocessing=True,
            enable_multi_scale=False,
            enable_tta=False
        ),
        'multi_scale': OptimizationConfig(
            enable_preprocessing=True,
            enable_multi_scale=True,
            enable_tta=False,
            scales=[480, 640, 800]
        ),
        'tta': OptimizationConfig(
            enable_preprocessing=True,
            enable_multi_scale=False,
            enable_tta=True,
            tta_scales=[0.9, 1.0, 1.1],
            tta_flips=[False, True]
        ),
        'full_optimization': OptimizationConfig(
            enable_preprocessing=True,
            enable_multi_scale=False,  # 避免过度计算
            enable_tta=True,
            enable_nms_optimization=True,
            enable_confidence_calibration=True,
            tta_scales=[0.95, 1.0, 1.05],
            tta_flips=[False, True]
        )
    }
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"\n🔍 测试配置: {config_name}")
        
        optimizer = DetectionAccuracyOptimizer(config=config)
        if optimizer.model is None:
            continue
        
        config_results = []
        
        for i, image_path in enumerate(test_images, 1):
            print(f"  [{i}/{len(test_images)}] {image_path.name}")
            
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            
            result = optimizer.optimize_detection(image)
            config_results.append(result)
        
        # 计算配置统计
        total_objects = sum(r.get('objects_count', 0) for r in config_results)
        confidences = []
        processing_times = []
        
        for r in config_results:
            if r.get('objects'):
                confidences.extend([obj['confidence'] for obj in r['objects']])
            processing_times.append(r.get('optimization_time', 0))
        
        results[config_name] = {
            'total_objects': total_objects,
            'average_confidence': float(np.mean(confidences)) if confidences else 0,
            'confidence_std': float(np.std(confidences)) if confidences else 0,
            'average_processing_time': float(np.mean(processing_times)),
            'images_processed': len(config_results)
        }
    
    # 生成优化报告
    generate_optimization_report(results)
    
    return results

def generate_optimization_report(results: Dict[str, Dict[str, Any]]):
    """生成优化报告"""
    
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>检测精度优化报告</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #2d3436 0%, #636e72 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .optimization-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 25px;
            padding: 40px;
        }}
        .config-card {{
            background: #f8f9fa;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border-left: 5px solid #74b9ff;
            transition: transform 0.3s ease;
        }}
        .config-card:hover {{
            transform: translateY(-5px);
        }}
        .config-name {{
            font-size: 1.4em;
            font-weight: bold;
            color: #2d3436;
            margin-bottom: 20px;
            text-align: center;
            text-transform: uppercase;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid #e0e0e0;
        }}
        .metric:last-child {{
            border-bottom: none;
        }}
        .metric-label {{
            color: #636e72;
            font-weight: 500;
        }}
        .metric-value {{
            font-weight: bold;
            color: #2d3436;
            font-size: 1.1em;
        }}
        .best-config {{
            border-left-color: #00b894;
            background: linear-gradient(135deg, #d1f2eb 0%, #a3e4d7 100%);
        }}
        .performance-summary {{
            background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .summary-title {{
            font-size: 1.8em;
            margin-bottom: 20px;
        }}
        .improvement-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .improvement-item {{
            background: rgba(255,255,255,0.2);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .improvement-value {{
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .footer {{
            background: #f8f9fa;
            padding: 25px;
            text-align: center;
            color: #636e72;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 检测精度优化报告</h1>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>多种优化策略性能对比分析</p>
        </div>
        
        <div class="optimization-grid">
"""
    
    # 找出最佳配置
    best_confidence = max(results.values(), key=lambda x: x['average_confidence'])['average_confidence']
    best_objects = max(results.values(), key=lambda x: x['total_objects'])['total_objects']
    
    for config_name, stats in results.items():
        is_best = (stats['average_confidence'] == best_confidence or 
                  stats['total_objects'] == best_objects)
        
        card_class = "config-card best-config" if is_best else "config-card"
        
        html_content += f"""
            <div class="{card_class}">
                <div class="config-name">{config_name.replace('_', ' ')}</div>
                <div class="metric">
                    <span class="metric-label">检测物体总数</span>
                    <span class="metric-value">{stats['total_objects']}</span>
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
                    <span class="metric-label">平均处理时间</span>
                    <span class="metric-value">{stats['average_processing_time']:.3f}s</span>
                </div>
                <div class="metric">
                    <span class="metric-label">处理图像数</span>
                    <span class="metric-value">{stats['images_processed']}</span>
                </div>
            </div>
"""
    
    # 计算改进统计
    baseline_confidence = results.get('baseline', {}).get('average_confidence', 0)
    best_improvement = ((best_confidence - baseline_confidence) / baseline_confidence * 100) if baseline_confidence > 0 else 0
    
    html_content += f"""
        </div>
        
        <div class="performance-summary">
            <div class="summary-title">📊 优化效果总结</div>
            <div class="improvement-stats">
                <div class="improvement-item">
                    <div class="improvement-value">{best_improvement:.1f}%</div>
                    <div>最大置信度提升</div>
                </div>
                <div class="improvement-item">
                    <div class="improvement-value">{best_confidence:.1%}</div>
                    <div>最佳平均置信度</div>
                </div>
                <div class="improvement-item">
                    <div class="improvement-value">{best_objects}</div>
                    <div>最多检测物体</div>
                </div>
                <div class="improvement-item">
                    <div class="improvement-value">{len(results)}</div>
                    <div>测试配置数量</div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p><strong>🔧 优化建议:</strong> 根据应用场景选择合适的优化策略</p>
            <p><strong>⚡ 性能权衡:</strong> 精度提升通常伴随处理时间增加</p>
            <p><strong>📈 最佳实践:</strong> 在精度和速度之间找到平衡点</p>
        </div>
    </div>
</body>
</html>
"""
    
    # 保存报告
    report_path = "detection_accuracy_optimization_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ 检测精度优化报告已生成: {report_path}")

if __name__ == "__main__":
    print("🚀 检测精度优化测试开始...")
    
    results = test_accuracy_optimization()
    
    if results:
        print("\n📊 优化测试完成!")
        for config, stats in results.items():
            print(f"{config}: {stats['average_confidence']:.1%} 置信度, {stats['total_objects']} 物体")
    else:
        print("❌ 优化测试失败")