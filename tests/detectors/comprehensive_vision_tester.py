#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合视觉测试器模块
提供完整的视觉检测测试功能
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from .yolos_native_detector import YOLOSNativeDetector
from .modelscope_analyzer import ModelScopeEnhancedAnalyzer


class ComprehensiveVisionTester:
    """综合视觉测试器"""
    
    def __init__(self, test_images_dir: str = "test_images"):
        """初始化测试器"""
        self.test_images_dir = Path(test_images_dir)
        self.results_dir = Path("test_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 初始化检测器
        print("🔧 初始化检测器...")
        self.yolo_detector = YOLOSNativeDetector()
        self.modelscope_analyzer = ModelScopeEnhancedAnalyzer()
        
        # 测试统计
        self.test_stats = {
            "total_tests": 0,
            "successful_tests": 0,
            "failed_tests": 0,
            "yolo_successes": 0,
            "modelscope_successes": 0,
            "start_time": None,
            "end_time": None
        }
        
        print("✅ 综合视觉测试器初始化完成")
    
    def run_comprehensive_test(self, image_path: Optional[str] = None) -> Dict[str, Any]:
        """运行综合测试"""
        print("\n" + "="*60)
        print("🚀 开始综合视觉检测测试")
        print("="*60)
        
        self.test_stats["start_time"] = datetime.now()
        
        if image_path:
            # 测试单个图像
            results = self._test_single_image(image_path)
        else:
            # 测试目录中的所有图像
            results = self._test_all_images()
        
        self.test_stats["end_time"] = datetime.now()
        
        # 生成测试报告
        report = self._generate_test_report(results)
        
        # 保存结果
        self._save_results(results, report)
        
        print("\n" + "="*60)
        print("✅ 综合测试完成")
        print("="*60)
        
        return {
            "results": results,
            "report": report,
            "stats": self.test_stats
        }
    
    def _test_single_image(self, image_path: str) -> List[Dict[str, Any]]:
        """测试单个图像"""
        if not os.path.exists(image_path):
            print(f"❌ 图像文件不存在: {image_path}")
            return []
        
        print(f"\n📸 测试图像: {image_path}")
        return [self._run_detection_pipeline(image_path)]
    
    def _test_all_images(self) -> List[Dict[str, Any]]:
        """测试所有图像"""
        if not self.test_images_dir.exists():
            print(f"❌ 测试图像目录不存在: {self.test_images_dir}")
            print("💡 请创建test_images目录并放入测试图像")
            return []
        
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # 查找所有图像文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.test_images_dir.glob(f"*{ext}"))
            image_files.extend(self.test_images_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"❌ 在{self.test_images_dir}中未找到图像文件")
            print(f"💡 支持的格式: {', '.join(image_extensions)}")
            return []
        
        print(f"\n📁 找到 {len(image_files)} 个测试图像")
        
        results = []
        for i, image_file in enumerate(image_files, 1):
            print(f"\n📸 [{i}/{len(image_files)}] 测试: {image_file.name}")
            result = self._run_detection_pipeline(str(image_file))
            results.append(result)
            
            # 简短的进度反馈
            if result["yolo_result"]["success"]:
                print(f"   ✓ YOLO检测成功")
            else:
                print(f"   ✗ YOLO检测失败")
            
            if result["modelscope_result"]["success"]:
                print(f"   ✓ ModelScope分析成功")
            else:
                print(f"   ✗ ModelScope分析失败")
        
        return results
    
    def _run_detection_pipeline(self, image_path: str) -> Dict[str, Any]:
        """运行完整的检测流水线"""
        self.test_stats["total_tests"] += 1
        
        pipeline_start = time.time()
        
        # 1. YOLO检测
        print("   🔍 运行YOLO检测...")
        yolo_result = self.yolo_detector.detect_objects(image_path)
        
        if yolo_result["success"]:
            self.test_stats["yolo_successes"] += 1
        
        # 2. ModelScope增强分析
        print("   🧠 运行ModelScope增强分析...")
        modelscope_result = self.modelscope_analyzer.analyze_with_context(
            image_path, yolo_result
        )
        
        if modelscope_result["success"]:
            self.test_stats["modelscope_successes"] += 1
        
        # 3. 综合评估
        overall_success = yolo_result["success"] and modelscope_result["success"]
        if overall_success:
            self.test_stats["successful_tests"] += 1
        else:
            self.test_stats["failed_tests"] += 1
        
        pipeline_time = time.time() - pipeline_start
        
        return {
            "image_path": image_path,
            "image_name": os.path.basename(image_path),
            "timestamp": datetime.now().isoformat(),
            "yolo_result": yolo_result,
            "modelscope_result": modelscope_result,
            "overall_success": overall_success,
            "total_pipeline_time": round(pipeline_time, 3)
        }
    
    def _generate_test_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成测试报告"""
        if not results:
            return {"error": "没有测试结果"}
        
        # 计算统计信息
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r["overall_success"])
        yolo_successes = sum(1 for r in results if r["yolo_result"]["success"])
        modelscope_successes = sum(1 for r in results if r["modelscope_result"]["success"])
        
        # 计算平均处理时间
        yolo_times = [r["yolo_result"].get("processing_time", 0) for r in results if r["yolo_result"]["success"]]
        modelscope_times = [r["modelscope_result"].get("processing_time", 0) for r in results if r["modelscope_result"]["success"]]
        pipeline_times = [r["total_pipeline_time"] for r in results]
        
        avg_yolo_time = sum(yolo_times) / len(yolo_times) if yolo_times else 0
        avg_modelscope_time = sum(modelscope_times) / len(modelscope_times) if modelscope_times else 0
        avg_pipeline_time = sum(pipeline_times) / len(pipeline_times) if pipeline_times else 0
        
        # 检测统计
        detection_stats = self._calculate_detection_stats(results)
        
        # 测试持续时间
        duration = None
        if self.test_stats["start_time"] and self.test_stats["end_time"]:
            duration = (self.test_stats["end_time"] - self.test_stats["start_time"]).total_seconds()
        
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": total_tests - successful_tests,
                "success_rate": round(successful_tests / total_tests * 100, 2) if total_tests > 0 else 0,
                "test_duration_seconds": round(duration, 2) if duration else None
            },
            "component_performance": {
                "yolo_detector": {
                    "success_count": yolo_successes,
                    "success_rate": round(yolo_successes / total_tests * 100, 2) if total_tests > 0 else 0,
                    "avg_processing_time": round(avg_yolo_time, 3)
                },
                "modelscope_analyzer": {
                    "success_count": modelscope_successes,
                    "success_rate": round(modelscope_successes / total_tests * 100, 2) if total_tests > 0 else 0,
                    "avg_processing_time": round(avg_modelscope_time, 3),
                    "available": self.modelscope_analyzer.is_available()
                }
            },
            "performance_metrics": {
                "avg_yolo_time": round(avg_yolo_time, 3),
                "avg_modelscope_time": round(avg_modelscope_time, 3),
                "avg_pipeline_time": round(avg_pipeline_time, 3)
            },
            "detection_statistics": detection_stats,
            "timestamp": datetime.now().isoformat()
        }
        
        return report
    
    def _calculate_detection_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算检测统计信息"""
        total_detections = 0
        class_counts = {}
        confidence_scores = []
        
        for result in results:
            if result["yolo_result"]["success"]:
                detections = result["yolo_result"].get("detections", [])
                total_detections += len(detections)
                
                for detection in detections:
                    class_name = detection.get("class", "unknown")
                    confidence = detection.get("confidence", 0)
                    
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    confidence_scores.append(confidence)
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        return {
            "total_detections": total_detections,
            "avg_detections_per_image": round(total_detections / len(results), 2) if results else 0,
            "class_distribution": dict(sorted(class_counts.items(), key=lambda x: x[1], reverse=True)),
            "avg_confidence": round(avg_confidence, 3),
            "confidence_range": {
                "min": round(min(confidence_scores), 3) if confidence_scores else 0,
                "max": round(max(confidence_scores), 3) if confidence_scores else 0
            }
        }
    
    def _save_results(self, results: List[Dict[str, Any]], report: Dict[str, Any]):
        """保存测试结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果
        results_file = self.results_dir / f"test_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存测试报告
        report_file = self.results_dir / f"test_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 结果已保存:")
        print(f"   详细结果: {results_file}")
        print(f"   测试报告: {report_file}")
    
    def print_summary(self, report: Dict[str, Any]):
        """打印测试摘要"""
        if "error" in report:
            print(f"❌ {report['error']}")
            return
        
        summary = report["test_summary"]
        performance = report["component_performance"]
        metrics = report["performance_metrics"]
        
        print("\n" + "="*50)
        print("📊 测试摘要")
        print("="*50)
        print(f"总测试数: {summary['total_tests']}")
        print(f"成功测试: {summary['successful_tests']}")
        print(f"失败测试: {summary['failed_tests']}")
        print(f"成功率: {summary['success_rate']}%")
        
        if summary.get('test_duration_seconds'):
            print(f"测试耗时: {summary['test_duration_seconds']}秒")
        
        print("\n📈 组件性能:")
        print(f"YOLO检测器: {performance['yolo_detector']['success_rate']}% 成功率, 平均 {performance['yolo_detector']['avg_processing_time']}秒")
        print(f"ModelScope分析器: {performance['modelscope_analyzer']['success_rate']}% 成功率, 平均 {performance['modelscope_analyzer']['avg_processing_time']}秒")
        
        print(f"\n⏱️ 平均处理时间: {metrics['avg_pipeline_time']}秒/图像")
        
        if "detection_statistics" in report:
            stats = report["detection_statistics"]
            print(f"\n🎯 检测统计:")
            print(f"总检测数: {stats['total_detections']}")
            print(f"平均每图检测数: {stats['avg_detections_per_image']}")
            print(f"平均置信度: {stats['avg_confidence']}")
            
            if stats['class_distribution']:
                print("\n🏷️ 检测类别分布:")
                for class_name, count in list(stats['class_distribution'].items())[:5]:
                    print(f"   {class_name}: {count}")
    
    def get_test_stats(self) -> Dict[str, Any]:
        """获取测试统计信息"""
        return self.test_stats.copy()