#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地大模型服务
集成Ollama等本地大模型，提供图像识别和分析能力
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

# 导入YOLOS核心模块
from ..core.types import DetectionResult, ObjectType
from ..utils.logging_manager import LoggingManager

@dataclass
class LocalLLMResult:
    """本地大模型分析结果"""
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

class LocalLLMService:
    """本地大模型服务"""
    
    def __init__(self, config_path: str = "config/local_llm_config.yaml"):
        """初始化本地大模型服务"""
        self.config_path = config_path
        self.config = self._load_config()
        
        # 初始化日志
        self.logger = LoggingManager.get_logger("LocalLLMService")
        
        # 服务状态
        self.is_running = False
        self.models_loaded = False
        
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
            'models_loaded': 0
        }
        
        # 数据库连接
        self.db_path = Path(self.config['storage']['base_path']) / "local_llm.db"
        self._init_database()
        
        # 工作线程
        self.worker_threads = []
        
        self.logger.info("本地大模型服务初始化完成")
    
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
            'local_llm': {
                'enabled': True,
                'server': {
                    'host': 'localhost',
                    'port': 11434,
                    'base_url': 'http://localhost:11434',
                    'timeout': 60
                },
                'models': {
                    'primary_vision_model': {
                        'name': 'llava:7b',
                        'enabled': True,
                        'max_tokens': 1024,
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
                'base_path': 'data/local_llm'
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
            self.logger.info("正在启动本地大模型服务...")
            
            # 检查Ollama服务
            if not self._check_ollama_service():
                self.logger.error("Ollama服务未运行，请先启动Ollama")
                return False
            
            # 检查和下载模型
            if self.config['local_llm']['auto_download']['enabled']:
                self._ensure_models_available()
            
            # 启动工作线程
            self._start_worker_threads()
            
            # 启动监控线程
            self._start_monitoring()
            
            self.is_running = True
            self.logger.info("本地大模型服务启动成功")
            return True
            
        except Exception as e:
            self.logger.error(f"服务启动失败: {e}")
            return False
    
    def stop_service(self):
        """停止服务"""
        try:
            self.logger.info("正在停止本地大模型服务...")
            
            self.is_running = False
            
            # 等待工作线程结束
            for thread in self.worker_threads:
                thread.join(timeout=5)
            
            self.logger.info("本地大模型服务已停止")
            
        except Exception as e:
            self.logger.error(f"服务停止失败: {e}")
    
    def _check_ollama_service(self) -> bool:
        """检查Ollama服务状态"""
        try:
            base_url = self.config['local_llm']['server']['base_url']
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _ensure_models_available(self):
        """确保模型可用"""
        try:
            base_url = self.config['local_llm']['server']['base_url']
            
            # 获取已安装的模型
            response = requests.get(f"{base_url}/api/tags")
            if response.status_code != 200:
                self.logger.error("无法获取模型列表")
                return
            
            installed_models = {model['name'] for model in response.json().get('models', [])}
            
            # 检查需要的模型
            models_config = self.config['local_llm']['models']
            required_models = []
            
            for model_config in models_config.values():
                if model_config.get('enabled', False):
                    required_models.append(model_config['name'])
            
            # 下载缺失的模型
            for model_name in required_models:
                if model_name not in installed_models:
                    self.logger.info(f"正在下载模型: {model_name}")
                    self._download_model(model_name)
            
            self.models_loaded = True
            self.logger.info("模型检查完成")
            
        except Exception as e:
            self.logger.error(f"模型检查失败: {e}")
    
    def _download_model(self, model_name: str):
        """下载模型"""
        try:
            base_url = self.config['local_llm']['server']['base_url']
            
            # 发送下载请求
            response = requests.post(
                f"{base_url}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=self.config['local_llm']['auto_download']['download_timeout']
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if 'status' in data:
                            self.logger.info(f"下载进度: {data['status']}")
                        if data.get('status') == 'success':
                            self.logger.info(f"模型 {model_name} 下载完成")
                            break
            else:
                self.logger.error(f"模型下载失败: {response.text}")
                
        except Exception as e:
            self.logger.error(f"下载模型 {model_name} 失败: {e}")
    
    def _start_worker_threads(self):
        """启动工作线程"""
        num_workers = self.config['performance']['worker_threads']
        
        for i in range(num_workers):
            thread = threading.Thread(
                target=self._worker_thread,
                name=f"LocalLLM-Worker-{i}",
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
                    time.sleep(self.config.get('monitoring', {}).get('performance_monitoring', {}).get('metrics_interval', 60))
                except Exception as e:
                    self.logger.error(f"监控线程错误: {e}")
        
        thread = threading.Thread(target=monitoring_thread, name="LocalLLM-Monitor", daemon=True)
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
    
    def analyze_image(self, image_path: str, task_type: str = "general") -> Optional[LocalLLMResult]:
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
            
            # 调用本地大模型
            raw_result = self._call_local_model(model_name, image_base64, prompt)
            
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
    
    def analyze_image_batch(self, image_paths: List[str], task_type: str = "general") -> List[Optional[LocalLLMResult]]:
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
        
        self.logger.info(f"批量分析完成，成功: {sum(1 for r in results if r is not None)}/{len(results)}")
        return results
    
    def _generate_image_hash(self, image: np.ndarray) -> str:
        """生成图像哈希"""
        # 使用图像内容生成MD5哈希
        image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
        return hashlib.md5(image_bytes).hexdigest()
    
    def _get_cached_result(self, image_hash: str) -> Optional[LocalLLMResult]:
        """获取缓存结果"""
        return self.result_cache.get(image_hash)
    
    def _cache_result(self, image_hash: str, result: LocalLLMResult):
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
        models_config = self.config['local_llm']['models']
        
        # 根据任务类型选择模型
        if task_type == "medical" and models_config.get('medical_model', {}).get('enabled', False):
            return models_config['medical_model']['name']
        elif models_config.get('primary_vision_model', {}).get('enabled', True):
            return models_config['primary_vision_model']['name']
        elif models_config.get('fallback_vision_model', {}).get('enabled', True):
            return models_config['fallback_vision_model']['name']
        else:
            return "llava:7b"  # 默认模型
    
    def _generate_prompt(self, task_type: str) -> str:
        """生成提示词"""
        prompts_config = self.config.get('prompts', {})
        
        system_prompt = prompts_config.get('system_prompt', '')
        
        if task_type == "medical":
            task_prompt = prompts_config.get('medical_prompt', '')
        elif task_type == "learning":
            task_prompt = prompts_config.get('learning_prompt', '')
        else:
            task_prompt = prompts_config.get('detailed_analysis_prompt', '')
        
        return f"{system_prompt}\n\n{task_prompt}"
    
    def _call_local_model(self, model_name: str, image_base64: str, prompt: str) -> Optional[str]:
        """调用本地大模型"""
        try:
            base_url = self.config['local_llm']['server']['base_url']
            timeout = self.config['local_llm']['server']['timeout']
            
            # 构建请求数据
            request_data = {
                "model": model_name,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 2048
                }
            }
            
            # 发送请求
            response = requests.post(
                f"{base_url}/api/generate",
                json=request_data,
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                self.logger.error(f"模型调用失败: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"调用本地模型失败: {e}")
            return None
    
    def _parse_model_result(self, raw_result: str, model_name: str, processing_time: float) -> LocalLLMResult:
        """解析模型结果"""
        try:
            # 尝试解析JSON格式的结果
            if raw_result.strip().startswith('{'):
                try:
                    parsed = json.loads(raw_result)
                    return LocalLLMResult(
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
                        timestamp=datetime.now().isoformat()
                    )
                except json.JSONDecodeError:
                    pass
            
            # 如果不是JSON格式，创建基础结果
            return LocalLLMResult(
                scene_description=raw_result[:500],  # 截取前500字符作为描述
                detected_objects=[],
                scene_category='unknown',
                overall_confidence=0.6,
                safety_assessment={'safety_level': 'unknown'},
                medical_relevance={'medical_content': False},
                technical_details={'image_quality': 'unknown'},
                learning_insights={'key_features': []},
                processing_time=processing_time,
                model_used=model_name,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"解析模型结果失败: {e}")
            return LocalLLMResult(
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
                timestamp=datetime.now().isoformat()
            )
    
    def _save_result_to_db(self, image_path: str, image_hash: str, result: LocalLLMResult):
        """保存结果到数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO analysis_results 
                (task_id, image_path, image_hash, result_json, processing_time, model_used)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                f"{image_hash}_{int(time.time())}",
                image_path,
                image_hash,
                json.dumps(asdict(result), ensure_ascii=False),
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
            'models_loaded': self.models_loaded,
            'queue_size': self.task_queue.qsize(),
            'worker_threads': len(self.worker_threads),
            'stats': self.stats.copy(),
            'ollama_available': self._check_ollama_service()
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
_local_llm_service = None

def get_local_llm_service() -> LocalLLMService:
    """获取本地大模型服务实例"""
    global _local_llm_service
    if _local_llm_service is None:
        _local_llm_service = LocalLLMService()
    return _local_llm_service