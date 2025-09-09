#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合测试套件
包含更多测试用例和场景
"""

import cv2
import numpy as np
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import unittest

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

class YOLOTestSuite:
    """YOLO综合测试套件"""
    
    def __init__(self):
        self.test_results = {}
        self.test_images_dir = Path("test_images")
        self.models_to_test = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
        
    def run_all_tests(self):
        """运行所有测试"""
        print("🧪 开始综合测试套件...")
        
        # 1. 基础功能测试
        self.test_basic_functionality()
        
        # 2. 性能测试
        self.test_performance_scenarios()
        
        # 3. 准确性测试
        self.test_accuracy_scenarios()
        
        # 4. 边界条件测试
        self.test_edge_cases()
        
        # 5. 多模型对比测试
        self.test_model_comparison()
        
        # 生成综合报告
        self.generate_comprehensive_report()
        
        return self.test_results
    
    def test_basic_functionality(self):
        """基础功能测试"""
        print("🔧 基础功能测试...")
        
        basic_tests = {
            'model_loading': self._test_model_loading(),
            'single_image_detection': self._test_single_image_detection(),
            'batch_processing': self._test_batch_processing(),
            'different_image_sizes': self._test_different_image_sizes(),
            'different_formats': self._test_different_formats()
        }
        
        self.test_results['basic_functionality'] = basic_tests
    
    def test_performance_scenarios(self):
        """性能场景测试"""
        print("⚡ 性能场景测试...")
        
        performance_tests = {
            'speed_benchmark': self._test_speed_benchmark(),
            'memory_usage': self._test_memory_usage(),
            'concurrent_processing': self._test_concurrent_processing(),
            'large_batch_processing': self._test_large_batch_processing()
        }
        
        self.test_results['performance'] = performance_tests
    
    def test_accuracy_scenarios(self):
        """准确性场景测试"""
        print("🎯 准确性场景测试...")
        
        accuracy_tests = {
            'confidence_thresholds': self._test_confidence_thresholds(),
            'nms_thresholds': self._test_nms_thresholds(),
            'different_lighting': self._test_different_lighting(),
            'object_sizes': self._test_object_sizes()
        }
        
        self.test_results['accuracy'] = accuracy_tests
    
    def test_edge_cases(self):
        """边界条件测试"""
        print("🚨 边界条件测试...")
        
        edge_tests = {
            'empty_images': self._test_empty_images(),
            'corrupted_images': self._test_corrupted_images(),
            'very_small_images': self._test_very_small_images(),
            'very_large_images': self._test_very_large_images(),
            'unusual_aspect_ratios': self._test_unusual_aspect_ratios()
        }
        
        self.test_results['edge_cases'] = edge_tests
    
    def test_model_comparison(self):
        """多模型对比测试"""
        print("🔄 多模型对比测试...")
        
        comparison_results = {}
        
        for model_name in self.models_to_test:
            try:
                model_results = self._test_single_model(model_name)
                comparison_results[model_name] = model_results
            except Exception as e:
                comparison_results[model_name] = {'error': str(e)}
        
        self.test_results['model_comparison'] = comparison_results
    
    def _test_model_loading(self) -> Dict[str, Any]:
        """测试模型加载"""
        results = {}
        
        for model_name in self.models_to_test:
            try:
                start_time = time.time()
                if ULTRALYTICS_AVAILABLE:
                    model = YOLO(model_name)
                    load_time = time.time() - start_time
                    results[model_name] = {
                        'status': 'success',
                        'load_time': load_time,
                        'model_size': self._get_model_size(model_name)
                    }
                else:
                    results[model_name] = {'status': 'ultralytics_not_available'}
            except Exception as e:
                results[model_name] = {'status': 'error', 'error': str(e)}
        
        return results
    
    def _test_single_image_detection(self) -> Dict[str, Any]:
        """测试单图像检测"""
        if not ULTRALYTICS_AVAILABLE:
            return {'status': 'ultralytics_not_available'}
        
        try:
            model = YOLO("yolov8n.pt")
            test_images = self._get_test_images()[:3]
            
            results = []
            for img_path in test_images:
                img = cv2.imread(str(img_path))
                if img is not None:
                    start_time = time.time()
                    detection_results = model(img, verbose=False)
                    processing_time = time.time() - start_time
                    
                    objects_count = len(detection_results[0].boxes) if detection_results[0].boxes is not None else 0
                    
                    results.append({
                        'image': img_path.name,
                        'processing_time': processing_time,
                        'objects_detected': objects_count,
                        'image_size': img.shape[:2]
                    })
            
            return {
                'status': 'success',
                'results': results,
                'average_time': np.mean([r['processing_time'] for r in results]),
                'total_objects': sum([r['objects_detected'] for r in results])
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _test_batch_processing(self) -> Dict[str, Any]:
        """测试批量处理"""
        if not ULTRALYTICS_AVAILABLE:
            return {'status': 'ultralytics_not_available'}
        
        try:
            model = YOLO("yolov8n.pt")
            test_images = self._get_test_images()[:5]
            
            # 单个处理
            single_start = time.time()
            single_results = []
            for img_path in test_images:
                img = cv2.imread(str(img_path))
                if img is not None:
                    result = model(img, verbose=False)
                    single_results.append(result)
            single_time = time.time() - single_start
            
            # 批量处理
            batch_start = time.time()
            images = [cv2.imread(str(img_path)) for img_path in test_images]
            images = [img for img in images if img is not None]
            batch_results = model(images, verbose=False)
            batch_time = time.time() - batch_start
            
            return {
                'status': 'success',
                'single_processing_time': single_time,
                'batch_processing_time': batch_time,
                'speedup_ratio': single_time / batch_time if batch_time > 0 else 0,
                'images_processed': len(images)
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _test_different_image_sizes(self) -> Dict[str, Any]:
        """测试不同图像尺寸"""
        if not ULTRALYTICS_AVAILABLE:
            return {'status': 'ultralytics_not_available'}
        
        try:
            model = YOLO("yolov8n.pt")
            test_image = self._get_test_images()[0]
            original_img = cv2.imread(str(test_image))
            
            size_tests = []
            test_sizes = [(320, 320), (640, 640), (1280, 1280), (1920, 1080)]
            
            for size in test_sizes:
                resized_img = cv2.resize(original_img, size)
                
                start_time = time.time()
                results = model(resized_img, verbose=False)
                processing_time = time.time() - start_time
                
                objects_count = len(results[0].boxes) if results[0].boxes is not None else 0
                
                size_tests.append({
                    'size': size,
                    'processing_time': processing_time,
                    'objects_detected': objects_count,
                    'fps': 1.0 / processing_time if processing_time > 0 else 0
                })
            
            return {
                'status': 'success',
                'size_tests': size_tests
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _test_different_formats(self) -> Dict[str, Any]:
        """测试不同图像格式"""
        if not ULTRALYTICS_AVAILABLE:
            return {'status': 'ultralytics_not_available'}
        
        try:
            model = YOLO("yolov8n.pt")
            test_image = self._get_test_images()[0]
            original_img = cv2.imread(str(test_image))
            
            format_tests = []
            
            # 测试不同颜色空间
            formats = {
                'BGR': original_img,
                'RGB': cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB),
                'GRAY': cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY),
                'HSV': cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
            }
            
            for format_name, img in formats.items():
                try:
                    start_time = time.time()
                    results = model(img, verbose=False)
                    processing_time = time.time() - start_time
                    
                    objects_count = len(results[0].boxes) if results[0].boxes is not None else 0
                    
                    format_tests.append({
                        'format': format_name,
                        'status': 'success',
                        'processing_time': processing_time,
                        'objects_detected': objects_count
                    })
                except Exception as e:
                    format_tests.append({
                        'format': format_name,
                        'status': 'error',
                        'error': str(e)
                    })
            
            return {
                'status': 'success',
                'format_tests': format_tests
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _test_speed_benchmark(self) -> Dict[str, Any]:
        """速度基准测试"""
        if not ULTRALYTICS_AVAILABLE:
            return {'status': 'ultralytics_not_available'}
        
        try:
            model = YOLO("yolov8n.pt")
            test_images = self._get_test_images()[:10]
            
            # 预热
            warmup_img = cv2.imread(str(test_images[0]))
            for _ in range(3):
                _ = model(warmup_img, verbose=False)
            
            # 基准测试
            times = []
            for img_path in test_images:
                img = cv2.imread(str(img_path))
                if img is not None:
                    start_time = time.time()
                    _ = model(img, verbose=False)
                    processing_time = time.time() - start_time
                    times.append(processing_time)
            
            return {
                'status': 'success',
                'average_time': np.mean(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'std_time': np.std(times),
                'average_fps': 1.0 / np.mean(times),
                'images_tested': len(times)
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _test_memory_usage(self) -> Dict[str, Any]:
        """内存使用测试"""
        try:
            import psutil
            
            if not ULTRALYTICS_AVAILABLE:
                return {'status': 'ultralytics_not_available'}
            
            # 记录初始内存
            initial_memory = psutil.virtual_memory().percent
            
            model = YOLO("yolov8n.pt")
            after_load_memory = psutil.virtual_memory().percent
            
            # 处理图像
            test_images = self._get_test_images()[:5]
            for img_path in test_images:
                img = cv2.imread(str(img_path))
                if img is not None:
                    _ = model(img, verbose=False)
            
            after_processing_memory = psutil.virtual_memory().percent
            
            return {
                'status': 'success',
                'initial_memory': initial_memory,
                'after_load_memory': after_load_memory,
                'after_processing_memory': after_processing_memory,
                'memory_increase_load': after_load_memory - initial_memory,
                'memory_increase_processing': after_processing_memory - after_load_memory
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _test_confidence_thresholds(self) -> Dict[str, Any]:
        """置信度阈值测试"""
        if not ULTRALYTICS_AVAILABLE:
            return {'status': 'ultralytics_not_available'}
        
        try:
            model = YOLO("yolov8n.pt")
            test_image = cv2.imread(str(self._get_test_images()[0]))
            
            threshold_tests = []
            thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]
            
            for threshold in thresholds:
                results = model(test_image, conf=threshold, verbose=False)
                objects_count = len(results[0].boxes) if results[0].boxes is not None else 0
                
                threshold_tests.append({
                    'threshold': threshold,
                    'objects_detected': objects_count
                })
            
            return {
                'status': 'success',
                'threshold_tests': threshold_tests
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _test_nms_thresholds(self) -> Dict[str, Any]:
        """NMS阈值测试"""
        if not ULTRALYTICS_AVAILABLE:
            return {'status': 'ultralytics_not_available'}
        
        try:
            model = YOLO("yolov8n.pt")
            test_image = cv2.imread(str(self._get_test_images()[0]))
            
            nms_tests = []
            nms_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
            
            for nms_threshold in nms_thresholds:
                results = model(test_image, iou=nms_threshold, verbose=False)
                objects_count = len(results[0].boxes) if results[0].boxes is not None else 0
                
                nms_tests.append({
                    'nms_threshold': nms_threshold,
                    'objects_detected': objects_count
                })
            
            return {
                'status': 'success',
                'nms_tests': nms_tests
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _test_empty_images(self) -> Dict[str, Any]:
        """测试空图像"""
        if not ULTRALYTICS_AVAILABLE:
            return {'status': 'ultralytics_not_available'}
        
        try:
            model = YOLO("yolov8n.pt")
            
            # 创建空图像
            empty_images = [
                np.zeros((640, 640, 3), dtype=np.uint8),  # 全黑
                np.ones((640, 640, 3), dtype=np.uint8) * 255,  # 全白
                np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)  # 随机噪声
            ]
            
            empty_tests = []
            for i, img in enumerate(empty_images):
                try:
                    results = model(img, verbose=False)
                    objects_count = len(results[0].boxes) if results[0].boxes is not None else 0
                    
                    empty_tests.append({
                        'image_type': ['black', 'white', 'noise'][i],
                        'status': 'success',
                        'objects_detected': objects_count
                    })
                except Exception as e:
                    empty_tests.append({
                        'image_type': ['black', 'white', 'noise'][i],
                        'status': 'error',
                        'error': str(e)
                    })
            
            return {
                'status': 'success',
                'empty_tests': empty_tests
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _test_single_model(self, model_name: str) -> Dict[str, Any]:
        """测试单个模型"""
        if not ULTRALYTICS_AVAILABLE:
            return {'status': 'ultralytics_not_available'}
        
        try:
            model = YOLO(model_name)
            test_images = self._get_test_images()[:3]
            
            results = []
            total_time = 0
            total_objects = 0
            
            for img_path in test_images:
                img = cv2.imread(str(img_path))
                if img is not None:
                    start_time = time.time()
                    detection_results = model(img, verbose=False)
                    processing_time = time.time() - start_time
                    
                    objects_count = len(detection_results[0].boxes) if detection_results[0].boxes is not None else 0
                    
                    total_time += processing_time
                    total_objects += objects_count
                    
                    results.append({
                        'image': img_path.name,
                        'processing_time': processing_time,
                        'objects_detected': objects_count
                    })
            
            return {
                'status': 'success',
                'model_name': model_name,
                'average_time': total_time / len(results) if results else 0,
                'total_objects': total_objects,
                'average_fps': len(results) / total_time if total_time > 0 else 0,
                'results': results
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _get_test_images(self) -> List[Path]:
        """获取测试图像"""
        if self.test_images_dir.exists():
            return list(self.test_images_dir.glob("*.jpg"))[:10]
        return []
    
    def _get_model_size(self, model_name: str) -> Optional[int]:
        """获取模型大小"""
        try:
            model_path = Path(model_name)
            if model_path.exists():
                return model_path.stat().st_size
        except:
            pass
        return None
    
    def generate_comprehensive_report(self):
        """生成综合报告"""
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLO综合测试报告</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .test-section {{
            padding: 30px;
            border-bottom: 1px solid #eee;
        }}
        .test-title {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
        }}
        .test-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        .test-card {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .status-success {{ border-left: 4px solid #28a745; }}
        .status-error {{ border-left: 4px solid #dc3545; }}
        .status-warning {{ border-left: 4px solid #ffc107; }}
        .metric {{
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
        }}
        .metric-value {{
            font-weight: bold;
            color: #007bff;
        }}
        .summary-stats {{
            background: #e8f5e8;
            padding: 20px;
            margin: 20px;
            border-radius: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 YOLO综合测试报告</h1>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>全面的YOLO检测系统测试结果</p>
        </div>
        
        <div class="summary-stats">
            <h3>📊 测试总览</h3>
            <div class="test-grid">
                <div class="test-card">
                    <strong>基础功能测试</strong><br>
                    {self._get_test_summary('basic_functionality')}
                </div>
                <div class="test-card">
                    <strong>性能测试</strong><br>
                    {self._get_test_summary('performance')}
                </div>
                <div class="test-card">
                    <strong>准确性测试</strong><br>
                    {self._get_test_summary('accuracy')}
                </div>
                <div class="test-card">
                    <strong>边界条件测试</strong><br>
                    {self._get_test_summary('edge_cases')}
                </div>
            </div>
        </div>
        
        {self._generate_test_sections()}
        
        <div class="test-section">
            <div class="test-title">📋 测试结论</div>
            <div class="test-card">
                <h4>✅ 成功项目</h4>
                <ul>
                    {self._generate_success_list()}
                </ul>
                
                <h4>⚠️ 需要改进</h4>
                <ul>
                    {self._generate_improvement_list()}
                </ul>
                
                <h4>🎯 建议</h4>
                <ul>
                    <li>优化检测精度配置</li>
                    <li>提升批量处理性能</li>
                    <li>增强边界条件处理</li>
                    <li>完善错误处理机制</li>
                </ul>
            </div>
        </div>
    </div>
</body>
</html>
"""
        
        # 保存报告
        report_path = "comprehensive_test_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ 综合测试报告已生成: {report_path}")
    
    def _get_test_summary(self, test_category: str) -> str:
        """获取测试摘要"""
        if test_category not in self.test_results:
            return "未执行"
        
        tests = self.test_results[test_category]
        total = len(tests)
        success = sum(1 for test in tests.values() if isinstance(test, dict) and test.get('status') == 'success')
        
        return f"{success}/{total} 通过"
    
    def _generate_test_sections(self) -> str:
        """生成测试部分HTML"""
        sections = []
        
        for category, tests in self.test_results.items():
            if category == 'model_comparison':
                continue
                
            section_html = f"""
        <div class="test-section">
            <div class="test-title">{self._get_category_icon(category)} {self._get_category_name(category)}</div>
            <div class="test-grid">
                {self._generate_test_cards(tests)}
            </div>
        </div>
"""
            sections.append(section_html)
        
        return ''.join(sections)
    
    def _generate_test_cards(self, tests: Dict[str, Any]) -> str:
        """生成测试卡片"""
        cards = []
        
        for test_name, result in tests.items():
            status_class = 'status-success' if result.get('status') == 'success' else 'status-error'
            
            card_html = f"""
                <div class="test-card {status_class}">
                    <h4>{test_name}</h4>
                    <div class="metric">
                        <span>状态:</span>
                        <span class="metric-value">{result.get('status', 'unknown')}</span>
                    </div>
                    {self._generate_metrics(result)}
                </div>
"""
            cards.append(card_html)
        
        return ''.join(cards)
    
    def _generate_metrics(self, result: Dict[str, Any]) -> str:
        """生成指标HTML"""
        metrics = []
        
        # 根据结果类型生成不同的指标
        if 'processing_time' in result:
            metrics.append(f'<div class="metric"><span>处理时间:</span><span class="metric-value">{result["processing_time"]:.3f}s</span></div>')
        
        if 'objects_detected' in result:
            metrics.append(f'<div class="metric"><span>检测物体:</span><span class="metric-value">{result["objects_detected"]}</span></div>')
        
        if 'average_fps' in result:
            metrics.append(f'<div class="metric"><span>平均FPS:</span><span class="metric-value">{result["average_fps"]:.1f}</span></div>')
        
        return ''.join(metrics)
    
    def _get_category_icon(self, category: str) -> str:
        """获取分类图标"""
        icons = {
            'basic_functionality': '🔧',
            'performance': '⚡',
            'accuracy': '🎯',
            'edge_cases': '🚨'
        }
        return icons.get(category, '📊')
    
    def _get_category_name(self, category: str) -> str:
        """获取分类名称"""
        names = {
            'basic_functionality': '基础功能测试',
            'performance': '性能测试',
            'accuracy': '准确性测试',
            'edge_cases': '边界条件测试'
        }
        return names.get(category, category)
    
    def _generate_success_list(self) -> str:
        """生成成功列表"""
        successes = []
        for category, tests in self.test_results.items():
            for test_name, result in tests.items():
                if isinstance(result, dict) and result.get('status') == 'success':
                    successes.append(f"<li>{test_name} - {category}</li>")
        return ''.join(successes[:10])  # 限制显示数量
    
    def _generate_improvement_list(self) -> str:
        """生成改进列表"""
        improvements = []
        for category, tests in self.test_results.items():
            for test_name, result in tests.items():
                if isinstance(result, dict) and result.get('status') != 'success':
                    improvements.append(f"<li>{test_name} - {result.get('status', 'unknown')}</li>")
        return ''.join(improvements[:10])  # 限制显示数量

def main():
    """主函数"""
    print("🚀 启动YOLO综合测试套件...")
    
    test_suite = YOLOTestSuite()
    results = test_suite.run_all_tests()
    
    print("\n📊 测试完成!")
    print(f"总测试类别: {len(results)}")
    
    # 统计成功率
    total_tests = 0
    successful_tests = 0
    
    for category, tests in results.items():
        if isinstance(tests, dict):
            for test_name, result in tests.items():
                total_tests += 1
                if isinstance(result, dict) and result.get('status') == 'success':
                    successful_tests += 1
    
    success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
    print(f"总体成功率: {success_rate:.1f}% ({successful_tests}/{total_tests})")
    
    return results

if __name__ == "__main__":
    main()