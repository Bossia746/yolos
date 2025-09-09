#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合识别系统
结合在线学习和离线学习，确保弱网环境下的可用性
"""

import os
import json
import logging
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import cv2
import torch
import requests
from dataclasses import dataclass
from enum import Enum

# 导入现有识别器
from ..plugins.domain.plant_recognition import PlantRecognitionPlugin
from ..plugins.domain.dynamic_object_recognition import DynamicObjectRecognitionPlugin
from ..recognition.enhanced_object_recognizer import EnhancedObjectRecognizer
from ..training.offline_training_manager import OfflineTrainingManager

logger = logging.getLogger(__name__)

class NetworkStatus(Enum):
    """网络状态"""
    ONLINE = "online"
    OFFLINE = "offline"
    WEAK = "weak"

@dataclass
class RecognitionRequest:
    """识别请求"""
    scene: str
    image: np.ndarray
    timestamp: float
    priority: int = 1  # 1=高, 2=中, 3=低
    use_online: bool = True

@dataclass
class RecognitionResponse:
    """识别响应"""
    scene: str
    results: List[Dict[str, Any]]
    confidence: float
    processing_time: float
    source: str  # 'offline', 'online', 'hybrid'
    timestamp: float

class HybridRecognitionSystem:
    """混合识别系统"""
    
    def __init__(self, 
                 offline_models_dir: str = "./models/offline_models",
                 online_fallback: bool = True,
                 network_check_interval: int = 30):
        """
        初始化混合识别系统
        
        Args:
            offline_models_dir: 离线模型目录
            online_fallback: 是否启用在线回退
            network_check_interval: 网络检查间隔(秒)
        """
        self.offline_models_dir = Path(offline_models_dir)
        self.online_fallback = online_fallback
        self.network_check_interval = network_check_interval
        
        # 网络状态
        self.network_status = NetworkStatus.OFFLINE
        self.last_network_check = 0
        
        # 离线训练管理器
        self.offline_manager = OfflineTrainingManager(str(self.offline_models_dir.parent))
        
        # 离线模型缓存
        self.offline_models = {}
        
        # 现有识别器
        self.recognizers = {}
        
        # 请求队列和缓存
        self.request_queue = []
        self.response_cache = {}
        self.cache_max_size = 1000
        
        # 性能统计
        self.stats = {
            'total_requests': 0,
            'offline_requests': 0,
            'online_requests': 0,
            'hybrid_requests': 0,
            'cache_hits': 0,
            'average_response_time': 0.0
        }
        
        # 初始化系统
        self._init_system()
        
        # 启动网络监控线程
        self._start_network_monitor()
        
        logger.info("混合识别系统初始化完成")
    
    def _init_system(self):
        """初始化系统"""
        # 检查网络状态
        self._check_network_status()
        
        # 加载离线模型
        self._load_offline_models()
        
        # 初始化现有识别器
        self._init_existing_recognizers()
        
        # 确保离线模型可用性
        self._ensure_offline_readiness()
    
    def _check_network_status(self) -> NetworkStatus:
        """检查网络状态"""
        current_time = time.time()
        
        # 避免频繁检查
        if current_time - self.last_network_check < self.network_check_interval:
            return self.network_status
        
        self.last_network_check = current_time
        
        try:
            # 测试网络连接
            response = requests.get('https://www.google.com', timeout=5)
            if response.status_code == 200:
                # 测试网络速度
                start_time = time.time()
                test_response = requests.get('https://httpbin.org/bytes/1024', timeout=10)
                end_time = time.time()
                
                speed = 1024 / (end_time - start_time)  # bytes/second
                
                if speed > 10240:  # > 10KB/s
                    self.network_status = NetworkStatus.ONLINE
                else:
                    self.network_status = NetworkStatus.WEAK
            else:
                self.network_status = NetworkStatus.OFFLINE
                
        except:
            self.network_status = NetworkStatus.OFFLINE
        
        logger.info(f"网络状态: {self.network_status.value}")
        return self.network_status
    
    def _load_offline_models(self):
        """加载离线模型"""
        logger.info("加载离线模型...")
        
        # 获取离线模型状态
        status = self.offline_manager.get_offline_status()
        
        for scene in status['scenes_status']:
            if status['scenes_status'][scene].get('trained', False):
                try:
                    model = self.offline_manager.load_offline_model(scene)
                    if model is not None:
                        self.offline_models[scene] = model
                        logger.info(f"✓ 离线模型加载: {scene}")
                    else:
                        logger.warning(f"✗ 离线模型加载失败: {scene}")
                except Exception as e:
                    logger.error(f"加载离线模型异常 {scene}: {e}")
        
        logger.info(f"离线模型加载完成: {len(self.offline_models)}/{len(status['scenes_status'])}")
    
    def _init_existing_recognizers(self):
        """初始化现有识别器"""
        try:
            # 植物识别
            self.recognizers['plants'] = PlantRecognitionPlugin()
            logger.info("✓ 植物识别器初始化")
        except Exception as e:
            logger.warning(f"植物识别器初始化失败: {e}")
        
        try:
            # 动态物体识别
            self.recognizers['dynamic_objects'] = DynamicObjectRecognitionPlugin()
            logger.info("✓ 动态物体识别器初始化")
        except Exception as e:
            logger.warning(f"动态物体识别器初始化失败: {e}")
        
        try:
            # 静物识别器
            self.recognizers['objects'] = EnhancedObjectRecognizer()
            logger.info("✓ 静物识别器初始化")
        except Exception as e:
            logger.warning(f"静物识别器初始化失败: {e}")
    
    def _ensure_offline_readiness(self):
        """确保离线模型就绪"""
        status = self.offline_manager.get_offline_status()
        
        if not status['offline_ready']:
            logger.warning("离线模型未完全就绪，开始训练缺失的模型...")
            
            # 训练缺失的模型
            for scene, scene_status in status['scenes_status'].items():
                if not scene_status.get('trained', False):
                    logger.info(f"训练离线模型: {scene}")
                    success = self.offline_manager.train_offline_model(scene, epochs=20)
                    if success:
                        # 重新加载模型
                        model = self.offline_manager.load_offline_model(scene)
                        if model is not None:
                            self.offline_models[scene] = model
                            logger.info(f"✓ 离线模型训练并加载: {scene}")
    
    def _start_network_monitor(self):
        """启动网络监控线程"""
        def monitor_network():
            while True:
                try:
                    self._check_network_status()
                    time.sleep(self.network_check_interval)
                except Exception as e:
                    logger.error(f"网络监控异常: {e}")
                    time.sleep(5)
        
        monitor_thread = threading.Thread(target=monitor_network, daemon=True)
        monitor_thread.start()
        logger.info("网络监控线程已启动")
    
    def recognize(self, request: RecognitionRequest) -> RecognitionResponse:
        """执行识别"""
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        # 检查缓存
        cache_key = self._generate_cache_key(request)
        if cache_key in self.response_cache:
            self.stats['cache_hits'] += 1
            cached_response = self.response_cache[cache_key]
            cached_response.timestamp = time.time()
            return cached_response
        
        # 选择识别策略
        response = self._execute_recognition_strategy(request)
        
        # 更新统计
        processing_time = time.time() - start_time
        response.processing_time = processing_time
        self.stats['average_response_time'] = (
            (self.stats['average_response_time'] * (self.stats['total_requests'] - 1) + processing_time) /
            self.stats['total_requests']
        )
        
        # 缓存结果
        self._cache_response(cache_key, response)
        
        return response
    
    def _execute_recognition_strategy(self, request: RecognitionRequest) -> RecognitionResponse:
        """执行识别策略"""
        scene = request.scene
        image = request.image
        
        # 策略1: 优先使用离线模型
        if scene in self.offline_models:
            try:
                results = self._recognize_offline(scene, image)
                if results and len(results) > 0:
                    self.stats['offline_requests'] += 1
                    return RecognitionResponse(
                        scene=scene,
                        results=results,
                        confidence=np.mean([r.get('confidence', 0.0) for r in results]),
                        processing_time=0.0,  # 将在外部设置
                        source='offline',
                        timestamp=time.time()
                    )
            except Exception as e:
                logger.warning(f"离线识别失败 {scene}: {e}")
        
        # 策略2: 使用现有识别器
        if scene in self.recognizers:
            try:
                results = self._recognize_with_existing(scene, image)
                if results and len(results) > 0:
                    self.stats['hybrid_requests'] += 1
                    return RecognitionResponse(
                        scene=scene,
                        results=results,
                        confidence=np.mean([r.get('confidence', 0.0) for r in results]),
                        processing_time=0.0,
                        source='hybrid',
                        timestamp=time.time()
                    )
            except Exception as e:
                logger.warning(f"现有识别器失败 {scene}: {e}")
        
        # 策略3: 在线识别（如果网络可用且允许）
        if (self.online_fallback and 
            request.use_online and 
            self.network_status != NetworkStatus.OFFLINE):
            try:
                results = self._recognize_online(scene, image)
                if results and len(results) > 0:
                    self.stats['online_requests'] += 1
                    return RecognitionResponse(
                        scene=scene,
                        results=results,
                        confidence=np.mean([r.get('confidence', 0.0) for r in results]),
                        processing_time=0.0,
                        source='online',
                        timestamp=time.time()
                    )
            except Exception as e:
                logger.warning(f"在线识别失败 {scene}: {e}")
        
        # 策略4: 基础识别（保底方案）
        results = self._recognize_basic(scene, image)
        return RecognitionResponse(
            scene=scene,
            results=results,
            confidence=0.3,  # 基础识别置信度较低
            processing_time=0.0,
            source='basic',
            timestamp=time.time()
        )
    
    def _recognize_offline(self, scene: str, image: np.ndarray) -> List[Dict[str, Any]]:
        """离线识别"""
        model = self.offline_models[scene]
        
        # 预处理图像
        processed_image = self._preprocess_image(image, (224, 224))
        
        # 模型推理
        with torch.no_grad():
            outputs = model(processed_image.unsqueeze(0))
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # 获取类别名称
        config_path = self.offline_manager.scene_dirs[scene] / 'configs' / f'{scene}_config.json'
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        class_names = config_data['scene_config']['classes']
        
        result = {
            'class_id': predicted.item(),
            'class_name': class_names[predicted.item()],
            'confidence': confidence.item(),
            'bbox': (0, 0, image.shape[1], image.shape[0]),
            'source': 'offline_model'
        }
        
        return [result]
    
    def _recognize_with_existing(self, scene: str, image: np.ndarray) -> List[Dict[str, Any]]:
        """使用现有识别器"""
        recognizer = self.recognizers.get(scene)
        if recognizer is None:
            return []
        
        try:
            if scene == 'plants':
                # 植物识别
                detections = recognizer.detect(image)
                results = []
                for detection in detections:
                    result = {
                        'class_name': detection.plant_type.value,
                        'confidence': detection.confidence,
                        'bbox': (detection.bbox.x, detection.bbox.y, 
                                detection.bbox.width, detection.bbox.height),
                        'source': 'plant_recognizer'
                    }
                    results.append(result)
                return results
            
            elif scene == 'dynamic_objects':
                # 动态物体识别
                motion_mask = np.zeros(image.shape[:2], dtype=np.uint8)  # 简化的运动掩码
                detections = recognizer.detect(image, motion_mask)
                results = []
                for detection in detections:
                    result = {
                        'class_name': detection.object_type.value,
                        'confidence': detection.confidence,
                        'bbox': (detection.bbox.x, detection.bbox.y,
                                detection.bbox.width, detection.bbox.height),
                        'source': 'dynamic_object_recognizer'
                    }
                    results.append(result)
                return results
            
            elif scene == 'objects':
                # 静物识别
                annotated_frame, object_results = recognizer.detect_objects(image)
                return object_results
            
        except Exception as e:
            logger.error(f"现有识别器执行失败 {scene}: {e}")
        
        return []
    
    def _recognize_online(self, scene: str, image: np.ndarray) -> List[Dict[str, Any]]:
        """在线识别（模拟实现）"""
        # 这里应该调用在线API服务
        # 为了演示，返回模拟结果
        
        if self.network_status == NetworkStatus.WEAK:
            # 弱网环境下使用简化的在线服务
            time.sleep(2)  # 模拟网络延迟
        else:
            time.sleep(0.5)  # 模拟正常网络延迟
        
        # 模拟在线识别结果
        mock_results = [
            {
                'class_name': f'{scene}_online_result',
                'confidence': 0.85,
                'bbox': (50, 50, 100, 100),
                'source': 'online_api'
            }
        ]
        
        return mock_results
    
    def _recognize_basic(self, scene: str, image: np.ndarray) -> List[Dict[str, Any]]:
        """基础识别（保底方案）"""
        # 使用传统计算机视觉方法
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 简单的边缘检测
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        results = []
        for i, contour in enumerate(contours[:3]):  # 最多返回3个结果
            area = cv2.contourArea(contour)
            if area > 1000:  # 过滤小区域
                x, y, w, h = cv2.boundingRect(contour)
                result = {
                    'class_name': f'{scene}_basic_{i}',
                    'confidence': 0.3,
                    'bbox': (x, y, w, h),
                    'source': 'basic_cv'
                }
                results.append(result)
        
        return results
    
    def _preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> torch.Tensor:
        """预处理图像"""
        # 调整尺寸
        resized = cv2.resize(image, target_size)
        
        # 转换为RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 归一化
        normalized = rgb.astype(np.float32) / 255.0
        
        # 标准化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        standardized = (normalized - mean) / std
        
        # 转换为tensor
        tensor = torch.from_numpy(standardized).permute(2, 0, 1)
        
        return tensor
    
    def _generate_cache_key(self, request: RecognitionRequest) -> str:
        """生成缓存键"""
        # 使用图像哈希和场景生成键
        image_hash = hash(request.image.tobytes())
        return f"{request.scene}_{image_hash}"
    
    def _cache_response(self, key: str, response: RecognitionResponse):
        """缓存响应"""
        if len(self.response_cache) >= self.cache_max_size:
            # 删除最旧的缓存项
            oldest_key = min(self.response_cache.keys(), 
                           key=lambda k: self.response_cache[k].timestamp)
            del self.response_cache[oldest_key]
        
        self.response_cache[key] = response
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        offline_status = self.offline_manager.get_offline_status()
        
        return {
            'network_status': self.network_status.value,
            'offline_models_loaded': len(self.offline_models),
            'existing_recognizers': len(self.recognizers),
            'offline_readiness': offline_status['offline_ready'],
            'cache_size': len(self.response_cache),
            'performance_stats': self.stats.copy(),
            'scenes_status': offline_status['scenes_status']
        }
    
    def recognize_scene(self, scene: str, image: np.ndarray, 
                       use_online: bool = True, priority: int = 1) -> RecognitionResponse:
        """便捷的场景识别接口"""
        request = RecognitionRequest(
            scene=scene,
            image=image,
            timestamp=time.time(),
            priority=priority,
            use_online=use_online
        )
        
        return self.recognize(request)
    
    def batch_recognize(self, requests: List[RecognitionRequest]) -> List[RecognitionResponse]:
        """批量识别"""
        responses = []
        
        # 按优先级排序
        sorted_requests = sorted(requests, key=lambda r: r.priority)
        
        for request in sorted_requests:
            response = self.recognize(request)
            responses.append(response)
        
        return responses

# 便捷函数
def create_hybrid_system(offline_models_dir: str = "./models/offline_models") -> HybridRecognitionSystem:
    """创建混合识别系统"""
    return HybridRecognitionSystem(offline_models_dir)

if __name__ == "__main__":
    # 示例使用
    system = create_hybrid_system()
    
    # 检查系统状态
    status = system.get_system_status()
    print(f"系统状态: {json.dumps(status, indent=2)}")
    
    # 测试识别
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 识别宠物
    response = system.recognize_scene('pets', test_image)
    print(f"宠物识别结果: {response}")
    
    # 识别植物
    response = system.recognize_scene('plants', test_image)
    print(f"植物识别结果: {response}")