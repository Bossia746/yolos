#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ModelScope大模型服务
集成ModelScope的Qwen2.5-VL视觉大模型，提供图像识别和分析能力
"""

import os
import cv2
import json
import base64
import hashlib
import requests
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import time
import yaml
from datetime import datetime
import sqlite3
import threading
from queue import Queue, Empty
import psutil
from openai import OpenAI

# 导入YOLOS核心模块
try:
    from ..core.types import DetectionResult, ObjectType
    from ..utils.logging_manager import LoggingManager
except ImportError:
    # 如果导入失败，使用基础日志
    import logging
    LoggingManager = None

@dataclass
class ModelScopeLLMResult:
    """ModelScope大模型分析结果"""
    scene_description: str
    detected_objects: List[Dict[str, Any]]
    scene_category: str
    overall_confidence: float
    safety_assessment: Dict[str, Any]
    medical_relevance: Dict[str, Any]
    technical_details: Dict[str, Any]
    learning_insights: Dict[str, Any]
    processing_time: float
    model_used: str
    timestamp: str
    raw_response: str

@dataclass
class ProcessingTask:
    """处理任务"""
    task_id: str
    image_path: str
    image_data: np.ndarray
    task_type: str
    priority: int
    created_at: float
    callback: Optional[callable] = None

class ModelScopeLLMService:
    """ModelScope大模型服务"""
    
    def __init__(self, config_path: str = "config/modelscope_llm_config.yaml"):
        """初始化ModelScope大模型服务"""
        self.config_path = config_path
        self.config = self._load_config()
        
        # 初始化日志
        if LoggingManager:
            self.logger = LoggingManager.get_logger("ModelScopeLLMService")
        else:
            self.logger = logging.getLogger("ModelScopeLLMService")
            self.logger.setLevel(logging.INFO)
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            base_url=self.config['modelscope_api']['api']['base_url'],
            api_key=self.config['modelscope_api']['api']['api_key']
        )
        
        # 服务状态
        self.is_running = False
        self.api_available = False
        
        # 任务队列
        self.task_queue = Queue(maxsize=self.config['performance']['request_queue_size'])
        self.result_cache = {}
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'average_response_time': 0.0,
            'api_calls_today': 0,
            'api_calls_hour': 0
        }
        
        # 配额管理
        self.quota_stats = {
            'daily_requests': 0,
            'hourly_requests': 0,
            'last_reset_day': datetime.now().day,
            'last_reset_hour': datetime.now().hour
        }
        
        # 数据库连接
        self.db_path = Path(self.config['storage']['base_path']) / "modelscope_llm.db"
        self._init_database()
        
        # 工作线程
        self.worker_threads = []
        
        self.logger.info("ModelScope大模型服务初始化完成")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            # 使用默认配置
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'modelscope_api': {
                'enabled': True,
                'api': {
                    'base_url': 'https://api-inference.modelscope.cn/v1',
                    'api_key': '*****',
                    'timeout': 60
                },
                'models': {
                    'primary_vision_model': {
                        'name': 'Qwen/Qwen2.5-VL-72B-Instruct',
                        'enabled': True,
                        'max_tokens': 2048,
                        'temperature': 0.1
                    }
                }
            },
            'performance': {
                'max_concurrent_requests': 4,
                'request_queue_size': 100,
                'worker_threads': 4
            },
            'storage': {
                'base_path': 'data/modelscope_llm'
            },
            'cache': {
                'enabled': True
            }
        }
    
    def _init_database(self):
        """初始化数据库"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 创建结果表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT UNIQUE,
                    image_path TEXT,
                    image_hash TEXT,
                    result_json TEXT,
                    raw_response TEXT,
                    processing_time REAL,
                    model_used TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建学习记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_hash TEXT,
                    original_result TEXT,
                    improved_result TEXT,
                    feedback TEXT,
                    learning_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建性能统计表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT,
                    metric_value REAL,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("数据库初始化完成")
            
        except Exception as e:
            self.logger.error(f"数据库初始化失败: {e}")
    
    def start_service(self) -> bool:
        """启动服务"""
        try:
            self.logger.info("正在启动ModelScope大模型服务...")
            
            # 检查API连接
            if not self._check_api_connectivity():
                self.logger.error("ModelScope API连接失败，请检查网络和API密钥")
                return False
            
            # 启动工作线程
            self._start_worker_threads()
            
            # 启动监控线程
            self._start_monitoring()
            
            self.is_running = True
            self.logger.info("ModelScope大模型服务启动成功")
            return True
            
        except Exception as e:
            self.logger.error(f"服务启动失败: {e}")
            return False
    
    def stop_service(self):
        """停止服务"""
        try:
            self.logger.info("正在停止ModelScope大模型服务...")
            
            self.is_running = False
            
            # 等待工作线程结束
            for thread in self.worker_threads:
                thread.join(timeout=5)
            
            self.logger.info("ModelScope大模型服务已停止")
            
        except Exception as e:
            self.logger.error(f"服务停止失败: {e}")
    
    def _check_api_connectivity(self) -> bool:
        """检查API连接"""
        try:
            # 发送简单的测试请求
            response = self.client.chat.completions.create(
                model=self.config['modelscope_api']['models']['primary_vision_model']['name'],
                messages=[{
                    'role': 'user',
                    'content': [{'type': 'text', 'text': 'Hello'}]
                }],
                max_tokens=10,
                timeout=10
            )
            
            self.api_available = True
            self.logger.info("ModelScope API连接正常")
            return True
            
        except Exception as e:
            self.logger.error(f"API连接测试失败: {e}")
            self.api_available = False
            return False
    
    def _start_worker_threads(self):
        """启动工作线程"""
        num_workers = self.config['performance']['worker_threads']
        
        for i in range(num_workers):
            thread = threading.Thread(
                target=self._worker_thread,
                name=f"ModelScope-Worker-{i}",
                daemon=True
            )
            thread.start()
            self.worker_threads.append(thread)
        
        self.logger.info(f"启动了 {num_workers} 个工作线程")
    
    def _worker_thread(self):
        """工作线程"""
        while self.is_running:
            try:
                # 从队列获取任务
                task = self.task_queue.get(timeout=1)
                
                # 处理任务
                result = self._process_task(task)
                
                # 调用回调函数
                if task.callback:
                    task.callback(result)
                
                self.task_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"工作线程错误: {e}")
    
    def _start_monitoring(self):
        """启动监控线程"""
        def monitoring_thread():
            while self.is_running:
                try:
                    self._collect_metrics()
                    time.sleep(60)  # 每分钟收集一次指标
                except Exception as e:
                    self.logger.error(f"监控线程错误: {e}")
        
        thread = threading.Thread(target=monitoring_thread, name="ModelScope-Monitor", daemon=True)
        thread.start()
    
    def _collect_metrics(self):
        """收集性能指标"""
        try:
            # 系统资源使用情况
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            
            # 记录到数据库
            self._record_metric("cpu_usage", cpu_usage)
            self._record_metric("memory_usage", memory_usage)
            self._record_metric("queue_size", self.task_queue.qsize())
            
            # 更新统计信息
            if self.stats['total_requests'] > 0:
                success_rate = self.stats['successful_requests'] / self.stats['total_requests']
                self._record_metric("success_rate", success_rate)
            
        except Exception as e:
            self.logger.error(f"指标收集失败: {e}")
    
    def _record_metric(self, metric_name: str, value: float):
        """记录指标到数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO performance_stats (metric_name, metric_value) VALUES (?, ?)",
                (metric_name, value)
            )
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"记录指标失败: {e}")
    
    def analyze_image(self, image_path: str, task_type: str = "general") -> Optional[ModelScopeLLMResult]:
        """分析图像"""
        try:
            start_time = time.time()
            
            # 检查服务状态
            if not self.is_running:
                self.logger.error("服务未运行")
                return None
            
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"无法读取图像: {image_path}")
                return None
            
            # 生成图像哈希
            image_hash = self._generate_image_hash(image)
            
            # 检查缓存
            if self.config.get('cache', {}).get('enabled', True):
                cached_result = self._get_cached_result(image_hash)
                if cached_result:
                    self.stats['cache_hits'] += 1
                    self.logger.info(f"使用缓存结果: {image_path}")
                    return cached_result
            
            # 预处理图像
            processed_image = self._preprocess_image(image)
            
            # 编码图像
            image_base64 = self._encode_image(processed_image)
            
            # 选择模型
            model_name = self._select_model(task_type)
            
            # 生成提示词
            prompt = self._generate_prompt(task_type)
            
            # 调用ModelScope API
            raw_result = self._call_modelscope_api(model_name, image_base64, prompt)
            
            if raw_result:
                # 解析结果
                result = self._parse_model_result(raw_result, model_name, time.time() - start_time)
                
                # 缓存结果
                if self.config.get('cache', {}).get('enabled', True):
                    self._cache_result(image_hash, result)
                
                # 保存到数据库
                self._save_result_to_db(image_path, image_hash, result)
                
                # 更新统计
                self.stats['successful_requests'] += 1
                self.stats['total_requests'] += 1
                
                self.logger.info(f"图像分析完成: {image_path}, 耗时: {result.processing_time:.2f}s")
                return result
            else:
                self.stats['failed_requests'] += 1
                self.stats['total_requests'] += 1
                return None
                
        except Exception as e:
            self.logger.error(f"图像分析失败: {e}")
            self.stats['failed_requests'] += 1
            self.stats['total_requests'] += 1
            return None
    
    def analyze_image_batch(self, image_paths: List[str], task_type: str = "general") -> List[Optional[ModelScopeLLMResult]]:
        """批量分析图像"""
        results = []
        
        self.logger.info(f"开始批量分析 {len(image_paths)} 张图像")
        
        for i, image_path in enumerate(image_paths):
            self.logger.info(f"处理图像 {i+1}/{len(image_paths)}: {image_path}")
            
            result = self.analyze_image(image_path, task_type)
            results.append(result)
            
            # 显示进度
            if (i + 1) % 10 == 0:
                self.logger.info(f"已完成 {i+1}/{len(image_paths)} 张图像的分析")
            
            # 批处理间隔，避免API限流
            time.sleep(0.1)
        
        self.logger.info(f"批量分析完成，成功: {sum(1 for r in results if r is not None)}/{len(results)}")
        return results
    
    def _generate_image_hash(self, image: np.ndarray) -> str:
        """生成图像哈希"""
        # 使用图像内容生成MD5哈希
        image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
        return hashlib.md5(image_bytes).hexdigest()
    
    def _get_cached_result(self, image_hash: str) -> Optional[ModelScopeLLMResult]:
        """获取缓存结果"""
        return self.result_cache.get(image_hash)
    
    def _cache_result(self, image_hash: str, result: ModelScopeLLMResult):
        """缓存结果"""
        cache_size = self.config.get('performance', {}).get('result_cache_size', 1000)
        
        if len(self.result_cache) >= cache_size:
            # 删除最旧的缓存
            oldest_key = next(iter(self.result_cache))
            del self.result_cache[oldest_key]
        
        self.result_cache[image_hash] = result
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """预处理图像"""
        try:
            config = self.config.get('image_processing', {}).get('preprocessing', {})
            
            if not config.get('enabled', True):
                return image
            
            processed = image.copy()
            
            # 调整大小
            if config.get('resize_enabled', True):
                max_width = config.get('max_width', 1024)
                max_height = config.get('max_height', 1024)
                
                h, w = processed.shape[:2]
                if w > max_width or h > max_height:
                    scale = min(max_width / w, max_height / h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # 质量增强
            if config.get('quality_enhancement', True):
                # CLAHE对比度增强
                lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # 降噪
            if config.get('noise_reduction', True):
                processed = cv2.bilateralFilter(processed, 9, 75, 75)
            
            return processed
            
        except Exception as e:
            self.logger.error(f"图像预处理失败: {e}")
            return image
    
    def _encode_image(self, image: np.ndarray) -> str:
        """编码图像为base64"""
        try:
            quality = self.config.get('image_processing', {}).get('encoding', {}).get('compression_quality', 85)
            
            # 编码为JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, buffer = cv2.imencode('.jpg', image, encode_param)
            
            # 转换为base64
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return image_base64
            
        except Exception as e:
            self.logger.error(f"图像编码失败: {e}")
            return ""
    
    def _select_model(self, task_type: str) -> str:
        """选择合适的模型"""
        models_config = self.config['modelscope_api']['models']
        
        # 根据任务类型选择模型
        if models_config.get('primary_vision_model', {}).get('enabled', True):
            return models_config['primary_vision_model']['name']
        elif models_config.get('fallback_vision_model', {}).get('enabled', True):
            return models_config['fallback_vision_model']['name']
        else:
            return "Qwen/Qwen2.5-VL-72B-Instruct"  # 默认模型
    
    def _generate_prompt(self, task_type: str) -> str:
        """生成提示词"""
        prompts_config = self.config.get('prompts', {})
        
        system_prompt = prompts_config.get('system_prompt', '')
        
        if task_type == "medical":
            task_prompt = prompts_config.get('medical_prompt', '')
        elif task_type == "learning":
            task_prompt = prompts_config.get('learning_prompt', '')
        elif task_type == "simple":
            task_prompt = prompts_config.get('simple_description_prompt', '')
        else:
            task_prompt = prompts_config.get('detailed_analysis_prompt', '')
        
        return f"{system_prompt}\n\n{task_prompt}"
    
    def _call_modelscope_api(self, model_name: str, image_base64: str, prompt: str) -> Optional[str]:
        """调用ModelScope API"""
        try:
            timeout = self.config['modelscope_api']['api']['timeout']
            
            # 构建消息
            messages = [{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/jpeg;base64,{image_base64}'
                        }
                    }
                ]
            }]
            
            # 发送请求
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=2048,
                temperature=0.1,
                stream=False,
                timeout=timeout
            )
            
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                self.logger.error("API返回空结果")
                return None
                
        except Exception as e:
            self.logger.error(f"调用ModelScope API失败: {e}")
            return None
    
    def _parse_model_result(self, raw_result: str, model_name: str, processing_time: float) -> ModelScopeLLMResult:
        """解析模型结果"""
        try:
            # 尝试解析JSON格式的结果
            if raw_result.strip().startswith('{'):
                try:
                    parsed = json.loads(raw_result)
                    return ModelScopeLLMResult(
                        scene_description=parsed.get('scene_description', ''),
                        detected_objects=parsed.get('detected_objects', []),
                        scene_category=parsed.get('scene_category', 'unknown'),
                        overall_confidence=parsed.get('overall_confidence', 0.5),
                        safety_assessment=parsed.get('safety_assessment', {}),
                        medical_relevance=parsed.get('medical_relevance', {}),
                        technical_details=parsed.get('technical_details', {}),
                        learning_insights=parsed.get('learning_insights', {}),
                        processing_time=processing_time,
                        model_used=model_name,
                        timestamp=datetime.now().isoformat(),
                        raw_response=raw_result
                    )
                except json.JSONDecodeError:
                    pass
            
            # 如果不是JSON格式，创建基础结果
            return ModelScopeLLMResult(
                scene_description=raw_result[:500] if raw_result else "无描述",
                detected_objects=[],
                scene_category='unknown',
                overall_confidence=0.6,
                safety_assessment={'safety_level': 'unknown'},
                medical_relevance={'medical_content': False},
                technical_details={'image_quality': 'unknown'},
                learning_insights={'key_features': []},
                processing_time=processing_time,
                model_used=model_name,
                timestamp=datetime.now().isoformat(),
                raw_response=raw_result
            )
            
        except Exception as e:
            self.logger.error(f"解析模型结果失败: {e}")
            return ModelScopeLLMResult(
                scene_description="解析失败",
                detected_objects=[],
                scene_category='error',
                overall_confidence=0.0,
                safety_assessment={},
                medical_relevance={},
                technical_details={},
                learning_insights={},
                processing_time=processing_time,
                model_used=model_name,
                timestamp=datetime.now().isoformat(),
                raw_response=raw_result or ""
            )
    
    def _save_result_to_db(self, image_path: str, image_hash: str, result: ModelScopeLLMResult):
        """保存结果到数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO analysis_results 
                (task_id, image_path, image_hash, result_json, raw_response, processing_time, model_used)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                f"{image_hash}_{int(time.time())}",
                image_path,
                image_hash,
                json.dumps(asdict(result), ensure_ascii=False),
                result.raw_response,
                result.processing_time,
                result.model_used
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"保存结果到数据库失败: {e}")
    
    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        return {
            'is_running': self.is_running,
            'api_available': self.api_available,
            'queue_size': self.task_queue.qsize(),
            'worker_threads': len(self.worker_threads),
            'stats': self.stats.copy(),
            'quota_stats': self.quota_stats.copy()
        }
    
    def get_analysis_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取分析历史"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT image_path, result_json, processing_time, model_used, created_at
                FROM analysis_results
                ORDER BY created_at DESC
                LIMIT ?
            ''', (limit,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'image_path': row[0],
                    'result': json.loads(row[1]),
                    'processing_time': row[2],
                    'model_used': row[3],
                    'created_at': row[4]
                })
            
            conn.close()
            return results
            
        except Exception as e:
            self.logger.error(f"获取分析历史失败: {e}")
            return []

# 全局服务实例
_modelscope_llm_service = None

def get_modelscope_llm_service() -> ModelScopeLLMService:
    """获取ModelScope大模型服务实例"""
    global _modelscope_llm_service
    if _modelscope_llm_service is None:
        _modelscope_llm_service = ModelScopeLLMService()
    return _modelscope_llm_service

# 便捷函数
def analyze_image_with_modelscope(image_path: str, task_type: str = "general") -> Optional[ModelScopeLLMResult]:
    """使用ModelScope分析图像的便捷函数"""
    service = get_modelscope_llm_service()
    if not service.is_running:
        service.start_service()
    return service.analyze_image(image_path, task_type)

def analyze_training_images() -> List[Optional[ModelScopeLLMResult]]:
    """分析训练图像的便捷函数"""
    service = get_modelscope_llm_service()
    if not service.is_running:
        service.start_service()
    
    # 获取训练图像路径
    training_image_dir = Path("test_images")
    if not training_image_dir.exists():
        print(f"测试图像目录不存在: {training_image_dir}")
        return []
    
    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(training_image_dir.glob(f"*{ext}"))
        image_paths.extend(training_image_dir.glob(f"*{ext.upper()}"))
    
    if not image_paths:
        print("未找到训练图像")
        return []
    
    print(f"找到 {len(image_paths)} 张训练图像")
    
    # 批量分析
    results = service.analyze_image_batch([str(p) for p in image_paths], "learning")
    
    return results