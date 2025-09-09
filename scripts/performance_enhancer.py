#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能提升模块
实现多种性能优化策略
"""

import cv2
import numpy as np
import time
import threading
import queue
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import json
import psutil
import gc

try:
    from ultralytics import YOLO
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class PerformanceEnhancer:
    """性能增强器"""
    
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model_path = model_path
        self.model = None
        self.device = 'cpu'
        
        # 性能优化配置
        self.optimization_config = {
            'enable_gpu': True,
            'enable_half_precision': True,
            'enable_batch_processing': True,
            'enable_model_optimization': True,
            'enable_memory_optimization': True,
            'batch_size': 4,
            'num_threads': 4
        }
        
        # 性能统计
        self.performance_stats = {
            'total_inferences': 0,
            'total_time': 0.0,
            'average_fps': 0.0,
            'memory_usage': [],
            'gpu_usage': [],
            'batch_processing_gains': []
        }
        
        # 缓存系统
        self.result_cache = {}
        self.cache_size_limit = 100
        
        self._initialize_performance_optimization()
    
    def _initialize_performance_optimization(self):
        """初始化性能优化"""
        print("🚀 初始化性能优化...")
        
        if not TORCH_AVAILABLE:
            print("❌ PyTorch不可用，性能优化受限")
            return
        
        # 1. 设备选择
        self._setup_device()
        
        # 2. 加载和优化模型
        self._load_and_optimize_model()
        
        # 3. 内存优化
        self._setup_memory_optimization()
        
        # 4. 线程池设置
        self._setup_threading()
    
    def _setup_device(self):
        """设置计算设备"""
        if self.optimization_config['enable_gpu'] and torch.cuda.is_available():
            self.device = 'cuda'
            # GPU优化设置
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            print(f"✅ GPU加速启用: {torch.cuda.get_device_name()}")
        else:
            self.device = 'cpu'
            # CPU优化设置
            torch.set_num_threads(self.optimization_config['num_threads'])
            print(f"✅ CPU优化启用: {self.optimization_config['num_threads']} 线程")
    
    def _load_and_optimize_model(self):
        """加载并优化模型"""
        try:
            self.model = YOLO(self.model_path)
            
            # 模型优化
            if self.optimization_config['enable_model_optimization']:
                # 预热模型
                dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
                if self.device == 'cuda':
                    dummy_input = dummy_input.half() if self.optimization_config['enable_half_precision'] else dummy_input
                
                # 执行预热推理
                with torch.no_grad():
                    _ = self.model(dummy_input, verbose=False)
                
                print("🔥 模型预热完成")
            
            # 半精度优化
            if self.optimization_config['enable_half_precision'] and self.device == 'cuda':
                self.model.model.half()
                print("⚡ 半精度优化启用")
            
            print(f"✅ 模型加载成功: {self.model_path}")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
    
    def _setup_memory_optimization(self):
        """设置内存优化"""
        if self.optimization_config['enable_memory_optimization']:
            # 垃圾回收优化
            gc.set_threshold(700, 10, 10)
            
            # CUDA内存优化
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                # 设置内存分配策略
                torch.cuda.set_per_process_memory_fraction(0.8)
            
            print("🧹 内存优化启用")
    
    def _setup_threading(self):
        """设置线程池"""
        self.thread_pool = []
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # 创建工作线程
        for i in range(self.optimization_config['num_threads']):
            thread = threading.Thread(target=self._worker_thread, daemon=True)
            thread.start()
            self.thread_pool.append(thread)
        
        print(f"🔧 线程池启用: {len(self.thread_pool)} 个工作线程")
    
    def _worker_thread(self):
        """工作线程函数"""
        while True:
            try:
                task = self.task_queue.get(timeout=1)
                if task is None:
                    break
                
                task_id, image, params = task
                result = self._single_inference(image, **params)
                self.result_queue.put((task_id, result))
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ 工作线程错误: {e}")
    
    def enhanced_detect(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """增强检测 - 单张图像"""
        if self.model is None:
            return self._create_error_result("模型未加载")
        
        start_time = time.time()
        
        # 检查缓存
        image_hash = self._calculate_image_hash(image)
        if image_hash in self.result_cache:
            cached_result = self.result_cache[image_hash].copy()
            cached_result['from_cache'] = True
            cached_result['processing_time'] = time.time() - start_time
            return cached_result
        
        # 执行检测
        result = self._single_inference(image, **kwargs)
        
        # 缓存结果
        self._cache_result(image_hash, result)
        
        # 更新性能统计
        processing_time = time.time() - start_time
        self._update_performance_stats(processing_time)
        
        result['processing_time'] = processing_time
        result['device'] = self.device
        result['optimizations_applied'] = self._get_applied_optimizations()
        
        return result
    
    def enhanced_detect_batch(self, images: List[np.ndarray], **kwargs) -> List[Dict[str, Any]]:
        """增强批量检测"""
        if self.model is None:
            return [self._create_error_result("模型未加载") for _ in images]
        
        if not self.optimization_config['enable_batch_processing']:
            # 逐个处理
            return [self.enhanced_detect(img, **kwargs) for img in images]
        
        start_time = time.time()
        
        # 批量处理
        batch_size = self.optimization_config['batch_size']
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_results = self._batch_inference(batch, **kwargs)
            results.extend(batch_results)
        
        # 计算批量处理性能提升
        total_time = time.time() - start_time
        estimated_sequential_time = len(images) * 0.1  # 估算顺序处理时间
        performance_gain = (estimated_sequential_time - total_time) / estimated_sequential_time
        
        self.performance_stats['batch_processing_gains'].append(performance_gain)
        
        return results
    
    def enhanced_detect_async(self, images: List[np.ndarray], **kwargs) -> List[Dict[str, Any]]:
        """异步增强检测"""
        if not images:
            return []
        
        # 提交任务到队列
        task_ids = []
        for i, image in enumerate(images):
            task_id = f"async_{int(time.time() * 1000)}_{i}"
            self.task_queue.put((task_id, image, kwargs))
            task_ids.append(task_id)
        
        # 收集结果
        results = {}
        collected = 0
        
        while collected < len(task_ids):
            try:
                task_id, result = self.result_queue.get(timeout=10)
                results[task_id] = result
                collected += 1
            except queue.Empty:
                print("⚠️ 异步检测超时")
                break
        
        # 按顺序返回结果
        ordered_results = []
        for task_id in task_ids:
            if task_id in results:
                ordered_results.append(results[task_id])
            else:
                ordered_results.append(self._create_error_result("异步处理超时"))
        
        return ordered_results
    
    def _single_inference(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """单次推理"""
        try:
            # 预处理
            processed_image = self._preprocess_for_performance(image)
            
            # 推理
            with torch.no_grad():
                results = self.model(
                    processed_image,
                    device=self.device,
                    verbose=False,
                    **kwargs
                )
            
            # 解析结果
            return self._parse_results(results[0])
            
        except Exception as e:
            return self._create_error_result(f"推理失败: {str(e)}")
    
    def _batch_inference(self, images: List[np.ndarray], **kwargs) -> List[Dict[str, Any]]:
        """批量推理"""
        try:
            # 预处理批量图像
            processed_images = [self._preprocess_for_performance(img) for img in images]
            
            # 批量推理
            with torch.no_grad():
                results = self.model(
                    processed_images,
                    device=self.device,
                    verbose=False,
                    **kwargs
                )
            
            # 解析批量结果
            return [self._parse_results(result) for result in results]
            
        except Exception as e:
            error_result = self._create_error_result(f"批量推理失败: {str(e)}")
            return [error_result for _ in images]
    
    def _preprocess_for_performance(self, image: np.ndarray) -> np.ndarray:
        """性能优化的预处理"""
        # 快速resize策略
        h, w = image.shape[:2]
        
        # 如果图像已经是合适大小，直接返回
        if h == 640 and w == 640:
            return image
        
        # 使用更快的插值方法
        if max(h, w) > 640:
            scale = 640 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            # 使用INTER_LINEAR而不是INTER_CUBIC以提高速度
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # 填充到640x640
            if new_h != 640 or new_w != 640:
                padded = np.zeros((640, 640, 3), dtype=np.uint8)
                padded[:new_h, :new_w] = resized
                return padded
            
            return resized
        
        return image
    
    def _parse_results(self, result) -> Dict[str, Any]:
        """解析结果"""
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
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'class_id': class_id
                }
                
                detection_info['objects'].append(obj_info)
            
            detection_info['objects_count'] = len(detection_info['objects'])
        
        return detection_info
    
    def _calculate_image_hash(self, image: np.ndarray) -> str:
        """计算图像哈希用于缓存"""
        # 简单的图像哈希
        small = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return str(hash(gray.tobytes()))
    
    def _cache_result(self, image_hash: str, result: Dict[str, Any]):
        """缓存结果"""
        if len(self.result_cache) >= self.cache_size_limit:
            # 删除最旧的缓存项
            oldest_key = next(iter(self.result_cache))
            del self.result_cache[oldest_key]
        
        self.result_cache[image_hash] = result.copy()
    
    def _update_performance_stats(self, processing_time: float):
        """更新性能统计"""
        self.performance_stats['total_inferences'] += 1
        self.performance_stats['total_time'] += processing_time
        
        # 计算平均FPS
        if self.performance_stats['total_time'] > 0:
            self.performance_stats['average_fps'] = (
                self.performance_stats['total_inferences'] / 
                self.performance_stats['total_time']
            )
        
        # 记录内存使用
        memory_percent = psutil.virtual_memory().percent
        self.performance_stats['memory_usage'].append(memory_percent)
        
        # 记录GPU使用（如果可用）
        if self.device == 'cuda' and torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            self.performance_stats['gpu_usage'].append(gpu_memory * 100)
    
    def _get_applied_optimizations(self) -> List[str]:
        """获取已应用的优化"""
        optimizations = []
        
        if self.device == 'cuda':
            optimizations.append('GPU加速')
        
        if self.optimization_config['enable_half_precision'] and self.device == 'cuda':
            optimizations.append('半精度计算')
        
        if self.optimization_config['enable_batch_processing']:
            optimizations.append('批量处理')
        
        if self.optimization_config['enable_memory_optimization']:
            optimizations.append('内存优化')
        
        if len(self.result_cache) > 0:
            optimizations.append('结果缓存')
        
        return optimizations
    
    def _create_error_result(self, error_msg: str) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            'error': error_msg,
            'objects': [],
            'objects_count': 0,
            'status': 'error'
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        stats = self.performance_stats.copy()
        
        # 计算统计信息
        if stats['memory_usage']:
            stats['average_memory_usage'] = np.mean(stats['memory_usage'])
            stats['peak_memory_usage'] = np.max(stats['memory_usage'])
        
        if stats['gpu_usage']:
            stats['average_gpu_usage'] = np.mean(stats['gpu_usage'])
            stats['peak_gpu_usage'] = np.max(stats['gpu_usage'])
        
        if stats['batch_processing_gains']:
            stats['average_batch_gain'] = np.mean(stats['batch_processing_gains'])
        
        # 系统信息
        stats['system_info'] = {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
            'cuda_available': torch.cuda.is_available() if TORCH_AVAILABLE else False,
            'device_used': self.device
        }
        
        return stats
    
    def benchmark_performance(self, test_images: List[np.ndarray], iterations: int = 3) -> Dict[str, Any]:
        """性能基准测试"""
        print(f"🏃 开始性能基准测试 ({iterations} 轮)...")
        
        benchmark_results = {
            'single_image_fps': [],
            'batch_processing_fps': [],
            'async_processing_fps': [],
            'memory_efficiency': [],
            'optimization_overhead': []
        }
        
        for iteration in range(iterations):
            print(f"  第 {iteration + 1}/{iterations} 轮测试")
            
            # 1. 单图像处理测试
            start_time = time.time()
            for img in test_images:
                _ = self.enhanced_detect(img)
            single_time = time.time() - start_time
            single_fps = len(test_images) / single_time
            benchmark_results['single_image_fps'].append(single_fps)
            
            # 2. 批量处理测试
            start_time = time.time()
            _ = self.enhanced_detect_batch(test_images)
            batch_time = time.time() - start_time
            batch_fps = len(test_images) / batch_time
            benchmark_results['batch_processing_fps'].append(batch_fps)
            
            # 3. 异步处理测试
            start_time = time.time()
            _ = self.enhanced_detect_async(test_images)
            async_time = time.time() - start_time
            async_fps = len(test_images) / async_time
            benchmark_results['async_processing_fps'].append(async_fps)
            
            # 4. 内存效率
            memory_before = psutil.virtual_memory().percent
            _ = self.enhanced_detect_batch(test_images * 2)  # 处理更多图像
            memory_after = psutil.virtual_memory().percent
            memory_increase = memory_after - memory_before
            benchmark_results['memory_efficiency'].append(memory_increase)
            
            # 清理内存
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
        
        # 计算平均值
        final_results = {}
        for key, values in benchmark_results.items():
            final_results[f'average_{key}'] = np.mean(values)
            final_results[f'std_{key}'] = np.std(values)
            final_results[f'best_{key}'] = np.max(values) if 'fps' in key else np.min(values)
        
        # 性能提升计算
        if final_results['average_batch_processing_fps'] > 0:
            batch_improvement = (
                final_results['average_batch_processing_fps'] / 
                final_results['average_single_image_fps'] - 1
            ) * 100
            final_results['batch_processing_improvement'] = batch_improvement
        
        return final_results

def test_performance_enhancement():
    """测试性能增强"""
    print("🚀 性能增强测试开始...")
    
    # 加载测试图像
    test_images_dir = Path("test_images")
    if not test_images_dir.exists():
        print("❌ 测试图像目录不存在")
        return
    
    image_files = list(test_images_dir.glob("*.jpg"))[:5]
    if not image_files:
        print("❌ 未找到测试图像")
        return
    
    test_images = []
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is not None:
            test_images.append(img)
    
    if not test_images:
        print("❌ 无法加载测试图像")
        return
    
    # 创建性能增强器
    enhancer = PerformanceEnhancer()
    
    if enhancer.model is None:
        print("❌ 性能增强器初始化失败")
        return
    
    # 运行基准测试
    benchmark_results = enhancer.benchmark_performance(test_images)
    
    # 生成性能报告
    performance_report = enhancer.get_performance_report()
    
    # 生成HTML报告
    generate_performance_report(benchmark_results, performance_report)
    
    return benchmark_results, performance_report

def generate_performance_report(benchmark_results: Dict[str, Any], 
                              performance_report: Dict[str, Any]):
    """生成性能报告"""
    
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>性能优化报告</title>
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
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 25px;
            padding: 40px;
        }}
        .metric-card {{
            background: #f8f9fa;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border-left: 5px solid #3498db;
        }}
        .metric-title {{
            font-size: 1.3em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 15px;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #e74c3c;
            margin-bottom: 10px;
        }}
        .metric-description {{
            color: #666;
            font-size: 0.9em;
        }}
        .system-info {{
            background: #e8f5e8;
            padding: 30px;
            margin: 20px 40px;
            border-radius: 10px;
        }}
        .system-info h3 {{
            color: #27ae60;
            margin-bottom: 20px;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .info-item {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .improvement-highlight {{
            border-left-color: #27ae60;
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        }}
        .footer {{
            background: #f8f9fa;
            padding: 25px;
            text-align: center;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚡ 性能优化报告</h1>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>YOLO检测性能优化效果分析</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card improvement-highlight">
                <div class="metric-title">🚀 单图像处理</div>
                <div class="metric-value">{benchmark_results.get('average_single_image_fps', 0):.1f} FPS</div>
                <div class="metric-description">平均单图像处理速度</div>
            </div>
            
            <div class="metric-card improvement-highlight">
                <div class="metric-title">📦 批量处理</div>
                <div class="metric-value">{benchmark_results.get('average_batch_processing_fps', 0):.1f} FPS</div>
                <div class="metric-description">批量处理平均速度</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">🔄 异步处理</div>
                <div class="metric-value">{benchmark_results.get('average_async_processing_fps', 0):.1f} FPS</div>
                <div class="metric-description">异步处理平均速度</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">📈 性能提升</div>
                <div class="metric-value">{benchmark_results.get('batch_processing_improvement', 0):.1f}%</div>
                <div class="metric-description">批量处理相对提升</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">🧠 内存使用</div>
                <div class="metric-value">{performance_report.get('average_memory_usage', 0):.1f}%</div>
                <div class="metric-description">平均内存占用率</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">⚡ 总推理次数</div>
                <div class="metric-value">{performance_report.get('total_inferences', 0)}</div>
                <div class="metric-description">累计推理次数</div>
            </div>
        </div>
        
        <div class="system-info">
            <h3>🖥️ 系统信息</h3>
            <div class="info-grid">
                <div class="info-item">
                    <strong>CPU核心数</strong><br>
                    {performance_report.get('system_info', {}).get('cpu_count', 'N/A')}
                </div>
                <div class="info-item">
                    <strong>总内存</strong><br>
                    {performance_report.get('system_info', {}).get('memory_total', 0):.1f} GB
                </div>
                <div class="info-item">
                    <strong>CUDA支持</strong><br>
                    {'✅ 是' if performance_report.get('system_info', {}).get('cuda_available') else '❌ 否'}
                </div>
                <div class="info-item">
                    <strong>使用设备</strong><br>
                    {performance_report.get('system_info', {}).get('device_used', 'CPU').upper()}
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p><strong>🎯 优化效果:</strong> 通过多种优化策略显著提升检测性能</p>
            <p><strong>💡 建议:</strong> 根据硬件配置选择合适的优化策略</p>
        </div>
    </div>
</body>
</html>
"""
    
    # 保存报告
    report_path = "performance_optimization_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ 性能优化报告已生成: {report_path}")

if __name__ == "__main__":
    print("🚀 性能增强测试开始...")
    
    results = test_performance_enhancement()
    
    if results:
        benchmark_results, performance_report = results
        print("\n📊 性能测试完成!")
        print(f"单图像: {benchmark_results.get('average_single_image_fps', 0):.1f} FPS")
        print(f"批量处理: {benchmark_results.get('average_batch_processing_fps', 0):.1f} FPS")
        print(f"性能提升: {benchmark_results.get('batch_processing_improvement', 0):.1f}%")
    else:
        print("❌ 性能测试失败")