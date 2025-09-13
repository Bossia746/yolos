#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOS手持静态物体识别专业测试 - 修正版
展示项目的完整检测能力和性能优化特性
"""

import cv2
import numpy as np
import time
import json
import threading
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt

# 添加项目路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / 'src'))

# 尝试导入ultralytics YOLO
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
    print("✅ Ultralytics YOLO 可用")
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("❌ Ultralytics YOLO 不可用")

# 导入测试配置
try:
    from .test_config import YOLOSTestConfig
except ImportError:
    print("❌ 无法导入YOLOSTestConfig，使用本地配置")

# TestConfig is now imported from .test_config as YOLOSTestConfig

@dataclass
class DetectionStats:
    """检测统计信息"""
    model_name: str
    total_detections: int = 0
    unique_objects: int = 0
    avg_confidence: float = 0.0
    avg_inference_time: float = 0.0
    fps: float = 0.0
    object_categories: Dict[str, int] = None
    confidence_distribution: List[float] = None
    inference_times: List[float] = None
    
    def __post_init__(self):
        if self.object_categories is None:
            self.object_categories = {}
        if self.confidence_distribution is None:
            self.confidence_distribution = []
        if self.inference_times is None:
            self.inference_times = []

class YOLODetector:
    """YOLO检测器包装类"""
    
    def __init__(self, model_name: str, confidence_threshold: float = 0.25, iou_threshold: float = 0.45):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        print(f"🤖 正在加载 {model_name} 模型...")
        
        try:
            self.model = YOLO(f'{model_name}.pt')
            print(f"✅ {model_name} 模型加载成功")
        except Exception as e:
            print(f"❌ {model_name} 模型加载失败: {e}")
            raise
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """执行检测"""
        try:
            results = self.model(image, conf=self.confidence_threshold, iou=self.iou_threshold, verbose=False)
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                        x1, y1, x2, y2 = box
                        class_name = self.model.names.get(cls_id, f"class_{cls_id}")
                        
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(conf),
                            'class_id': int(cls_id),
                            'class_name': class_name
                        })
            
            return detections
            
        except Exception as e:
            print(f"检测失败: {e}")
            return []

class HandheldObjectRecognitionTest:
    """手持静态物体识别测试系统"""
    
    def __init__(self, config: YOLOSTestConfig):
        self.config = config
        
        # 创建输出目录
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 初始化检测器
        self.detectors = {}
        self.detection_stats = {}
        
        # 测试数据
        self.test_images = []
        self.test_results = {}
        
        # 摄像头
        self.camera = None
        self.is_testing = False
        
        print("🎯 手持物体识别测试系统初始化完成")
    
    def initialize_detectors(self):
        """初始化所有检测器"""
        print("🤖 正在初始化检测器...")
        
        if not ULTRALYTICS_AVAILABLE:
            print("❌ Ultralytics不可用，无法加载YOLO模型")
            return False
        
        for model_name in self.config.models_to_test:
            try:
                detector = YOLODetector(
                    model_name=model_name,
                    confidence_threshold=self.config.confidence_threshold,
                    iou_threshold=self.config.iou_threshold
                )
                
                self.detectors[model_name] = detector
                self.detection_stats[model_name] = DetectionStats(model_name=model_name)
                
            except Exception as e:
                print(f"❌ {model_name} 加载失败: {e}")
        
        if not self.detectors:
            print("❌ 没有成功加载任何检测器")
            return False
        
        print(f"✅ 成功加载 {len(self.detectors)} 个检测器")
        return True
    
    def setup_camera(self):
        """设置摄像头"""
        print("📷 正在初始化摄像头...")
        
        self.camera = cv2.VideoCapture(self.config.camera_id)
        
        if not self.camera.isOpened():
            raise RuntimeError(f"无法打开摄像头 {self.config.camera_id}")
        
        # 设置摄像头参数
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # 获取实际参数
        width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.camera.get(cv2.CAP_PROP_FPS))
        
        print(f"✅ 摄像头配置: {width}x{height} @ {fps}FPS")
        
        # 预热摄像头
        for _ in range(10):
            ret, _ = self.camera.read()
            if not ret:
                raise RuntimeError("摄像头预热失败")
        
        print("✅ 摄像头初始化完成")
        return True
    
    def run_comprehensive_test(self):
        """运行综合测试"""
        print("🚀 开始手持静态物体识别综合测试")
        print(f"⏱️ 测试时长: {self.config.test_duration}秒")
        print(f"🤖 测试模型: {', '.join(self.config.models_to_test)}")
        
        try:
            # 初始化系统
            if not self.initialize_detectors():
                return False
            
            if not self.setup_camera():
                return False
            
            # 创建视频录制器
            video_writer = None
            if self.config.save_video:
                video_writer = self._create_video_writer()
            
            # 开始测试
            self.is_testing = True
            start_time = time.time()
            last_capture_time = 0
            frame_count = 0
            
            print("\n" + "="*80)
            print("🎯 YOLOS手持静态物体识别专业测试")
            print("="*80)
            print("📋 操作说明:")
            print("  🤏 手持不同物体在摄像头前展示")
            print("  ⌨️  按 'c' 键手动捕获当前帧进行分析")
            print("  💾 按 's' 键保存当前检测结果")
            print("  📊 按 'p' 键显示实时性能统计")
            print("  🚪 按 'q' 键退出测试")
            print("="*80)
            
            while self.is_testing:
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                # 检查测试时间
                if elapsed_time >= self.config.test_duration:
                    print("⏰ 测试时间到达，自动结束")
                    break
                
                # 读取摄像头帧
                ret, frame = self.camera.read()
                if not ret:
                    print("❌ 无法读取摄像头帧")
                    break
                
                frame_count += 1
                
                # 创建显示帧
                display_frame = frame.copy()
                
                # 自动捕获逻辑
                if (current_time - last_capture_time) >= self.config.capture_interval:
                    self._capture_and_analyze_frame(frame, f"auto_{len(self.test_images)}")
                    last_capture_time = current_time
                
                # 绘制实时信息
                self._draw_test_info(display_frame, elapsed_time, len(self.test_images))
                
                # 显示最新检测结果
                if self.test_results:
                    latest_result = list(self.test_results.values())[-1]
                    self._draw_latest_detection_results(display_frame, latest_result)
                
                # 录制视频
                if video_writer:
                    video_writer.write(display_frame)
                
                # 显示图像
                cv2.imshow('YOLOS手持物体识别专业测试', display_frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("👋 用户退出测试")
                    break
                elif key == ord('c'):
                    self._capture_and_analyze_frame(frame, f"manual_{len(self.test_images)}")
                elif key == ord('s'):
                    self._save_current_results(frame)
                elif key == ord('p'):
                    self._print_realtime_stats()
            
            # 清理资源
            if video_writer:
                video_writer.release()
            
            self.camera.release()
            cv2.destroyAllWindows()
            
            # 生成测试报告
            if self.config.generate_report:
                self._generate_comprehensive_report()
            
            print("✅ 测试完成")
            return True
            
        except Exception as e:
            print(f"❌ 测试过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.is_testing = False
            if self.camera:
                self.camera.release()
            cv2.destroyAllWindows()
    
    def _capture_and_analyze_frame(self, frame: np.ndarray, frame_id: str):
        """捕获并分析帧"""
        print(f"📸 捕获帧进行分析: {frame_id}")
        
        # 保存原始图像
        if self.config.save_images:
            image_path = self.output_dir / f"captured_{frame_id}.jpg"
            cv2.imwrite(str(image_path), frame)
        
        # 存储测试图像
        self.test_images.append((frame_id, frame.copy()))
        
        # 对所有模型进行检测
        frame_results = {}
        
        for model_name, detector in self.detectors.items():
            try:
                start_time = time.time()
                
                # 执行检测
                detections = detector.detect(frame)
                
                inference_time = time.time() - start_time
                
                # 更新统计信息
                self._update_detection_stats(model_name, detections, inference_time)
                
                # 存储结果
                frame_results[model_name] = {
                    'detections': detections,
                    'inference_time': inference_time,
                    'timestamp': time.time()
                }
                
                print(f"🔍 {model_name}: {len(detections)}个检测, {inference_time:.3f}s")
                
            except Exception as e:
                print(f"❌ {model_name} 检测失败: {e}")
                frame_results[model_name] = {
                    'detections': [],
                    'inference_time': 0,
                    'error': str(e)
                }
        
        # 存储帧结果
        self.test_results[frame_id] = frame_results
        
        # 创建对比可视化
        if len(frame_results) > 1:
            self._create_detection_comparison(frame, frame_results, frame_id)
    
    def _update_detection_stats(self, model_name: str, detections: List[Dict], inference_time: float):
        """更新检测统计信息"""
        stats = self.detection_stats[model_name]
        
        # 基础统计
        stats.total_detections += len(detections)
        
        # 推理时间统计
        stats.inference_times.append(inference_time)
        stats.avg_inference_time = np.mean(stats.inference_times)
        stats.fps = 1.0 / stats.avg_inference_time if stats.avg_inference_time > 0 else 0
        
        # 置信度统计
        if detections:
            confidences = [d['confidence'] for d in detections]
            stats.confidence_distribution.extend(confidences)
            stats.avg_confidence = np.mean(stats.confidence_distribution)
            
            # 类别统计
            for detection in detections:
                class_name = detection['class_name']
                stats.object_categories[class_name] = stats.object_categories.get(class_name, 0) + 1
            
            stats.unique_objects = len(stats.object_categories)
    
    def _create_detection_comparison(self, frame: np.ndarray, results: Dict, frame_id: str):
        """创建检测结果对比图"""
        num_models = len(results)
        fig, axes = plt.subplots(1, num_models, figsize=(5*num_models, 5))
        
        if num_models == 1:
            axes = [axes]
        
        for idx, (model_name, result) in enumerate(results.items()):
            if idx >= len(axes):
                break
                
            # 绘制检测结果
            annotated_frame = self._draw_detections_on_frame(
                frame.copy(), result['detections'], model_name
            )
            
            # 转换BGR到RGB
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            axes[idx].imshow(annotated_frame_rgb)
            axes[idx].set_title(f"{model_name}\n检测数: {len(result['detections'])}, "
                              f"时间: {result['inference_time']:.3f}s")
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"comparison_{frame_id}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _draw_detections_on_frame(self, frame: np.ndarray, detections: List[Dict], model_name: str) -> np.ndarray:
        """在帧上绘制检测结果"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # 选择颜色
            color = colors[detection['class_id'] % len(colors)]
            
            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"{detection['class_name']} {detection['confidence']:.2f}"
            
            # 计算文本大小
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            
            # 绘制文本背景
            cv2.rectangle(frame, (x1, y1 - text_height - 5), 
                         (x1 + text_width, y1), color, -1)
            
            # 绘制文本
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def _draw_test_info(self, frame: np.ndarray, elapsed_time: float, captured_frames: int):
        """绘制测试信息"""
        info_lines = [
            f"测试时间: {elapsed_time:.1f}s / {self.config.test_duration}s",
            f"已捕获帧数: {captured_frames}",
            f"活跃模型: {len(self.detectors)}",
            f"按 'c' 手动捕获, 'q' 退出"
        ]
        
        # 绘制半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 绘制文本
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (20, 35 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    def _draw_latest_detection_results(self, frame: np.ndarray, latest_result: Dict):
        """绘制最新检测结果摘要"""
        y_offset = 150
        
        # 绘制背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, y_offset), (500, y_offset + 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 标题
        cv2.putText(frame, "最新检测结果:", (20, y_offset + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 各模型结果
        line_y = y_offset + 50
        for model_name, result in latest_result.items():
            if 'error' in result:
                text = f"{model_name}: 错误"
                color = (0, 0, 255)
            else:
                detections = result['detections']
                inference_time = result['inference_time']
                text = f"{model_name}: {len(detections)}个目标, {inference_time:.3f}s"
                color = (0, 255, 0)
            
            cv2.putText(frame, text, (30, line_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            line_y += 25
    
    def _create_video_writer(self):
        """创建视频录制器"""
        if not self.config.save_video:
            return None
        
        # 获取摄像头参数
        width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 20  # 录制FPS
        
        # 创建视频文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = self.output_dir / f"test_video_{timestamp}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        print(f"🎥 开始录制视频: {video_path}")
        return video_writer
    
    def _save_current_results(self, frame: np.ndarray):
        """保存当前结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存当前帧
        frame_path = self.output_dir / f"manual_save_{timestamp}.jpg"
        cv2.imwrite(str(frame_path), frame)
        
        # 保存检测结果
        if self.test_results:
            latest_result = list(self.test_results.values())[-1]
            result_path = self.output_dir / f"detection_result_{timestamp}.json"
            
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(latest_result, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"💾 结果已保存: {frame_path}")
    
    def _print_realtime_stats(self):
        """打印实时统计信息"""
        print("\n" + "="*80)
        print("📊 实时性能统计")
        print("="*80)
        
        for model_name, stats in self.detection_stats.items():
            print(f"\n🔍 {model_name}:")
            print(f"  📈 总检测数: {stats.total_detections}")
            print(f"  ⏱️  平均推理时间: {stats.avg_inference_time:.3f}s")
            print(f"  🚀 FPS: {stats.fps:.1f}")
            print(f"  🎯 平均置信度: {stats.avg_confidence:.3f}")
            print(f"  🏷️  检测类别数: {stats.unique_objects}")
            
            if stats.object_categories:
                print("  🔍 检测到的物体:")
                for obj_name, count in sorted(stats.object_categories.items(), 
                                            key=lambda x: x[1], reverse=True)[:5]:
                    print(f"    • {obj_name}: {count}次")
        
        print("="*80)
    
    def _generate_comprehensive_report(self):
        """生成综合测试报告"""
        print("📋 正在生成综合测试报告...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.output_dir / f"report_{timestamp}"
        report_dir.mkdir(exist_ok=True)
        
        # 1. 生成性能对比图表
        self._create_performance_charts(report_dir)
        
        # 2. 生成HTML报告
        self._create_html_report(report_dir, timestamp)
        
        # 3. 保存原始数据
        self._save_raw_data(report_dir)
        
        print(f"📋 测试报告已生成: {report_dir}")
        print(f"🌐 请打开 {report_dir}/test_report.html 查看详细报告")
    
    def _create_performance_charts(self, report_dir: Path):
        """创建性能对比图表"""
        try:
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 1. 推理时间对比
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            model_names = list(self.detection_stats.keys())
            inference_times = [stats.avg_inference_time for stats in self.detection_stats.values()]
            fps_values = [stats.fps for stats in self.detection_stats.values()]
            total_detections = [stats.total_detections for stats in self.detection_stats.values()]
            avg_confidences = [stats.avg_confidence for stats in self.detection_stats.values()]
            
            # 推理时间对比
            bars1 = ax1.bar(model_names, inference_times, color='skyblue')
            ax1.set_title('Average Inference Time Comparison')
            ax1.set_ylabel('Time (seconds)')
            ax1.tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, time_val in zip(bars1, inference_times):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{time_val:.3f}s', ha='center', va='bottom')
            
            # FPS对比
            bars2 = ax2.bar(model_names, fps_values, color='lightgreen')
            ax2.set_title('FPS Performance Comparison')
            ax2.set_ylabel('FPS')
            ax2.tick_params(axis='x', rotation=45)
            
            for bar, fps_val in zip(bars2, fps_values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{fps_val:.1f}', ha='center', va='bottom')
            
            # 检测数量对比
            bars3 = ax3.bar(model_names, total_detections, color='orange')
            ax3.set_title('Total Detections Comparison')
            ax3.set_ylabel('Detection Count')
            ax3.tick_params(axis='x', rotation=45)
            
            for bar, det_count in zip(bars3, total_detections):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{det_count}', ha='center', va='bottom')
            
            # 平均置信度对比
            bars4 = ax4.bar(model_names, avg_confidences, color='pink')
            ax4.set_title('Average Confidence Comparison')
            ax4.set_ylabel('Confidence')
            ax4.tick_params(axis='x', rotation=45)
            
            for bar, conf_val in zip(bars4, avg_confidences):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{conf_val:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(report_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️ 图表生成失败: {e}")
    
    def _create_html_report(self, report_dir: Path, timestamp: str):
        """创建HTML测试报告"""
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLOS手持物体识别测试报告</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 20px; }}
        .section {{ margin: 30px 0; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .stat-card {{ background: #f9f9f9; padding: 15px; border-radius: 8px; border-left: 4px solid #4CAF50; }}
        .chart-container {{ text-align: center; margin: 20px 0; }}
        .chart-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #4CAF50; color: white; }}
        .highlight {{ color: #4CAF50; font-weight: bold; }}
        .config-section {{ background: #e8f5e8; padding: 15px; border-radius: 8px; }}
        .emoji {{ font-size: 1.2em; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><span class="emoji">🎯</span> YOLOS手持物体识别专业测试报告</h1>
            <p>测试时间: {timestamp}</p>
            <p>测试时长: {self.config.test_duration}秒 | 捕获帧数: {len(self.test_images)}</p>
        </div>
        
        <div class="section">
            <h2><span class="emoji">📋</span> 测试配置</h2>
            <div class="config-section">
                <p><strong>测试模型:</strong> {', '.join(self.config.models_to_test)}</p>
                <p><strong>置信度阈值:</strong> {self.config.confidence_threshold}</p>
                <p><strong>IoU阈值:</strong> {self.config.iou_threshold}</p>
                <p><strong>半精度推理:</strong> {'启用' if self.config.enable_half_precision else '禁用'}</p>
                <p><strong>捕获间隔:</strong> {self.config.capture_interval}秒</p>
            </div>
        </div>
        
        <div class="section">
            <h2><span class="emoji">📊</span> 性能统计</h2>
            <div class="stats-grid">
        """
        
        # 添加各模型统计卡片
        for model_name, stats in self.detection_stats.items():
            html_content += f"""
                <div class="stat-card">
                    <h3><span class="emoji">🤖</span> {model_name}</h3>
                    <p><strong>总检测数:</strong> <span class="highlight">{stats.total_detections}</span></p>
                    <p><strong>平均推理时间:</strong> <span class="highlight">{stats.avg_inference_time:.3f}s</span></p>
                    <p><strong>FPS:</strong> <span class="highlight">{stats.fps:.1f}</span></p>
                    <p><strong>平均置信度:</strong> <span class="highlight">{stats.avg_confidence:.3f}</span></p>
                    <p><strong>检测类别数:</strong> <span class="highlight">{stats.unique_objects}</span></p>
                </div>
            """
        
        html_content += """
            </div>
        </div>
        """
        
        # 如果有性能图表，添加它
        if (report_dir / 'performance_comparison.png').exists():
            html_content += """
        <div class="section">
            <h2><span class="emoji">📈</span> 性能对比图表</h2>
            <div class="chart-container">
                <img src="performance_comparison.png" alt="性能对比图表">
            </div>
        </div>
            """
        
        html_content += """
        <div class="section">
            <h2><span class="emoji">🔍</span> 详细检测结果</h2>
            <table>
                <thead>
                    <tr>
                        <th>模型</th>
                        <th>总检测数</th>
                        <th>平均推理时间</th>
                        <th>FPS</th>
                        <th>平均置信度</th>
                        <th>主要检测类别</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # 添加详细统计表格
        for model_name, stats in self.detection_stats.items():
            top_categories = sorted(stats.object_categories.items(), 
                                  key=lambda x: x[1], reverse=True)[:3]
            top_categories_str = ', '.join([f"{cat}({count})" for cat, count in top_categories])
            
            html_content += f"""
                    <tr>
                        <td>{model_name}</td>
                        <td>{stats.total_detections}</td>
                        <td>{stats.avg_inference_time:.3f}s</td>
                        <td>{stats.fps:.1f}</td>
                        <td>{stats.avg_confidence:.3f}</td>
                        <td>{top_categories_str}</td>
                    </tr>
            """
        
        html_content += """
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2><span class="emoji">💡</span> 测试总结</h2>
            <div class="config-section">
        """
        
        # 生成测试总结
        if self.detection_stats:
            best_fps_model = max(self.detection_stats.items(), key=lambda x: x[1].fps)
            best_accuracy_model = max(self.detection_stats.items(), key=lambda x: x[1].avg_confidence)
            most_detections_model = max(self.detection_stats.items(), key=lambda x: x[1].total_detections)
            
            html_content += f"""
                <p><strong><span class="emoji">🚀</span> 最快模型:</strong> {best_fps_model[0]} (FPS: {best_fps_model[1].fps:.1f})</p>
                <p><strong><span class="emoji">🎯</span> 最高置信度:</strong> {best_accuracy_model[0]} (置信度: {best_accuracy_model[1].avg_confidence:.3f})</p>
                <p><strong><span class="emoji">🔍</span> 检测数最多:</strong> {most_detections_model[0]} (检测数: {most_detections_model[1].total_detections})</p>
            """
        
        html_content += f"""
                <p><strong><span class="emoji">📊</span> 测试完成度:</strong> 成功测试了 {len(self.detectors)} 个模型，捕获了 {len(self.test_images)} 帧图像</p>
            </div>
        </div>
        
        <div class="section">
            <p style="text-align: center; color: #666; margin-top: 40px;">
                <span class="emoji">⏰</span> 报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
                <span class="emoji">🎯</span> YOLOS智能视频识别系统
            </p>
        </div>
    </div>
</body>
</html>
        """
        
        # 保存HTML报告
        with open(report_dir / 'test_report.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _save_raw_data(self, report_dir: Path):
        """保存原始测试数据"""
        # 保存统计数据
        stats_data = {}
        for model_name, stats in self.detection_stats.items():
            stats_data[model_name] = asdict(stats)
        
        with open(report_dir / 'detection_stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存测试结果
        with open(report_dir / 'test_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存测试配置
        with open(report_dir / 'test_config.json', 'w', encoding='utf-8') as f:
            json.dump(asdict(self.config), f, ensure_ascii=False, indent=2)

def main():
    """主函数"""
    print("🎯 YOLOS手持静态物体识别专业测试")
    print("="*80)
    
    # 检查基本依赖
    if not ULTRALYTICS_AVAILABLE:
        print("❌ Ultralytics YOLO 不可用，请安装: pip install ultralytics")
        return
    
    # 检查摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头，请检查摄像头连接")
        return
    cap.release()
    
    print("✅ 系统检查通过")
    
    # 创建测试配置
    config = TestConfig(
        test_duration=90,  # 90秒测试
        capture_interval=4.0,  # 每4秒自动捕获
        models_to_test=['yolov8n', 'yolov8s'],  # 使用可靠的模型
        confidence_threshold=0.25,
        enable_half_precision=True,
        save_images=True,
        save_video=True,
        generate_report=True
    )
    
    # 创建测试系统
    test_system = HandheldObjectRecognitionTest(config)
    
    try:
        # 运行测试
        success = test_system.run_comprehensive_test()
        
        if success:
            print("\n✅ 测试完成！")
            print(f"📁 结果保存在: {test_system.output_dir}")
            print("📋 请查看生成的HTML报告获取详细分析结果")
        else:
            print("\n❌ 测试未能完成")
        
    except KeyboardInterrupt:
        print("\n⚠️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()