#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOS 客户端SDK
为第三方系统提供便捷的API调用接口
"""

import os
import sys
import json
import time
import base64
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
import queue

# HTTP客户端
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# WebSocket客户端
try:
    import socketio
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False
    print("Warning: python-socketio not available. Install: pip install python-socketio")

# 图像处理
import cv2
import numpy as np
from PIL import Image
import io

@dataclass
class YOLOSConfig:
    """YOLOS客户端配置"""
    api_base_url: str = "http://localhost:8080"
    api_key: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    retry_backoff_factor: float = 0.3
    enable_websocket: bool = True
    websocket_timeout: int = 60

@dataclass
class RecognitionResult:
    """识别结果数据结构"""
    success: bool
    task_id: Optional[str] = None
    detected_objects: Optional[List[Dict]] = None
    processing_time: Optional[float] = None
    confidence_scores: Optional[Dict] = None
    emergency_alerts: Optional[List[Dict]] = None
    recommendations: Optional[List[str]] = None
    error_message: Optional[str] = None
    timestamp: Optional[datetime] = None

@dataclass
class DeviceStatus:
    """设备状态数据结构"""
    position: Dict[str, float]
    camera_angle: Dict[str, float]
    zoom_level: float
    recording: bool
    online: bool
    battery_level: int
    last_update: datetime

@dataclass
class VoiceCommandResult:
    """语音命令结果数据结构"""
    success: bool
    command_text: Optional[str] = None
    confidence: Optional[float] = None
    actions_taken: Optional[List[Dict]] = None
    error_message: Optional[str] = None

class YOLOSClientError(Exception):
    """YOLOS客户端异常"""
    pass

class YOLOSClient:
    """YOLOS客户端SDK主类"""
    
    def __init__(self, config: YOLOSConfig = None):
        """
        初始化YOLOS客户端
        
        Args:
            config: 客户端配置，如果为None则使用默认配置
        """
        self.config = config or YOLOSConfig()
        self.logger = self._setup_logger()
        
        # 初始化HTTP会话
        self.session = requests.Session()
        self._setup_http_session()
        
        # WebSocket客户端
        self.sio = None
        self.websocket_connected = False
        self.event_handlers = {}
        
        # 任务管理
        self.active_tasks = {}
        self.task_callbacks = {}
        
        if self.config.enable_websocket and SOCKETIO_AVAILABLE:
            self._setup_websocket()
        
        self.logger.info(f"YOLOS客户端初始化完成，API地址: {self.config.api_base_url}")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("YOLOSClient")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _setup_http_session(self):
        """设置HTTP会话"""
        # 重试策略
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.retry_backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # 设置默认头部
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'YOLOS-Client-SDK/2.0.0'
        })
        
        # 设置API密钥
        if self.config.api_key:
            self.session.headers.update({
                'X-API-Key': self.config.api_key
            })
    
    def _setup_websocket(self):
        """设置WebSocket连接"""
        if not SOCKETIO_AVAILABLE:
            self.logger.warning("WebSocket功能不可用，请安装python-socketio")
            return
        
        self.sio = socketio.Client()
        
        @self.sio.event
        def connect():
            self.websocket_connected = True
            self.logger.info("WebSocket连接成功")
        
        @self.sio.event
        def disconnect():
            self.websocket_connected = False
            self.logger.info("WebSocket连接断开")
        
        @self.sio.event
        def task_completed(data):
            """任务完成事件处理"""
            task_id = data.get('task_id')
            if task_id in self.task_callbacks:
                callback = self.task_callbacks[task_id]
                if callback:
                    try:
                        callback(data)
                    except Exception as e:
                        self.logger.error(f"任务回调执行失败: {e}")
                del self.task_callbacks[task_id]
        
        @self.sio.event
        def device_position_updated(data):
            """设备位置更新事件处理"""
            if 'device_status_update' in self.event_handlers:
                self.event_handlers['device_status_update'](data)
        
        @self.sio.event
        def voice_command_received(data):
            """语音命令接收事件处理"""
            if 'voice_command' in self.event_handlers:
                self.event_handlers['voice_command'](data)
    
    def connect_websocket(self) -> bool:
        """连接WebSocket"""
        if not self.sio:
            self.logger.warning("WebSocket未初始化")
            return False
        
        try:
            websocket_url = self.config.api_base_url.replace('http', 'ws')
            self.sio.connect(websocket_url, wait_timeout=self.config.websocket_timeout)
            return True
        except Exception as e:
            self.logger.error(f"WebSocket连接失败: {e}")
            return False
    
    def disconnect_websocket(self):
        """断开WebSocket连接"""
        if self.sio and self.websocket_connected:
            self.sio.disconnect()
    
    def set_event_handler(self, event_type: str, handler: Callable):
        """设置事件处理器"""
        self.event_handlers[event_type] = handler
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """发送HTTP请求"""
        url = f"{self.config.api_base_url}{endpoint}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                timeout=self.config.timeout,
                **kwargs
            )
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API请求失败: {e}")
            raise YOLOSClientError(f"API请求失败: {str(e)}")
        except json.JSONDecodeError as e:
            self.logger.error(f"响应JSON解析失败: {e}")
            raise YOLOSClientError(f"响应格式错误: {str(e)}")
    
    # ==================== 健康检查和状态 ====================
    
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            Dict: 健康状态信息
        """
        return self._make_request('GET', '/api/health')
    
    def get_device_status(self) -> DeviceStatus:
        """
        获取设备状态
        
        Returns:
            DeviceStatus: 设备状态对象
        """
        response = self._make_request('GET', '/api/device/status')
        
        if response.get('success'):
            data = response.get('data', {})
            return DeviceStatus(
                position=data.get('position', {}),
                camera_angle=data.get('camera_angle', {}),
                zoom_level=data.get('zoom_level', 1.0),
                recording=data.get('recording', False),
                online=data.get('online', True),
                battery_level=data.get('battery_level', 100),
                last_update=datetime.now()
            )
        else:
            raise YOLOSClientError(response.get('message', '获取设备状态失败'))
    
    # ==================== 设备控制 ====================
    
    def move_device(self, x: float, y: float, z: float = 0.0) -> bool:
        """
        移动设备到指定位置
        
        Args:
            x: X坐标
            y: Y坐标
            z: Z坐标（可选）
            
        Returns:
            bool: 移动是否成功
        """
        data = {
            'position': {'x': x, 'y': y, 'z': z}
        }
        
        response = self._make_request('POST', '/api/device/move', json=data)
        return response.get('success', False)
    
    def move_device_to_location(self, location_name: str) -> bool:
        """
        移动设备到预定义位置
        
        Args:
            location_name: 位置名称（如"客厅"、"卧室"等）
            
        Returns:
            bool: 移动是否成功
        """
        # 预定义位置映射
        locations = {
            "客厅": {"x": 0, "y": 0, "z": 0},
            "卧室": {"x": 5, "y": 0, "z": 0},
            "厨房": {"x": 0, "y": 5, "z": 0},
            "阳台": {"x": 5, "y": 5, "z": 0},
            "门口": {"x": -2, "y": 0, "z": 0},
            "窗边": {"x": 0, "y": -2, "z": 0},
        }
        
        if location_name not in locations:
            raise YOLOSClientError(f"未知位置: {location_name}")
        
        pos = locations[location_name]
        return self.move_device(pos['x'], pos['y'], pos['z'])
    
    def rotate_camera(self, pan: float, tilt: float) -> bool:
        """
        旋转摄像头
        
        Args:
            pan: 水平角度（-180到180度）
            tilt: 垂直角度（-90到90度）
            
        Returns:
            bool: 旋转是否成功
        """
        data = {
            'command': 'rotate_camera',
            'parameters': {'pan': pan, 'tilt': tilt}
        }
        
        response = self._make_request('POST', '/api/device/control', json=data)
        return response.get('success', False)
    
    def zoom_camera(self, zoom_level: float) -> bool:
        """
        调整摄像头缩放
        
        Args:
            zoom_level: 缩放级别（0.5到5.0）
            
        Returns:
            bool: 缩放是否成功
        """
        data = {
            'command': 'zoom',
            'parameters': {'level': zoom_level}
        }
        
        response = self._make_request('POST', '/api/device/control', json=data)
        return response.get('success', False)
    
    def take_photo(self) -> Dict[str, Any]:
        """
        拍照
        
        Returns:
            Dict: 拍照结果信息
        """
        data = {
            'command': 'take_photo',
            'parameters': {}
        }
        
        response = self._make_request('POST', '/api/device/control', json=data)
        
        if response.get('success'):
            return response.get('data', {})
        else:
            raise YOLOSClientError(response.get('message', '拍照失败'))
    
    def start_recording(self) -> bool:
        """
        开始录像
        
        Returns:
            bool: 录像是否开始成功
        """
        data = {
            'command': 'start_recording',
            'parameters': {}
        }
        
        response = self._make_request('POST', '/api/device/control', json=data)
        return response.get('success', False)
    
    def stop_recording(self) -> bool:
        """
        停止录像
        
        Returns:
            bool: 录像是否停止成功
        """
        data = {
            'command': 'stop_recording',
            'parameters': {}
        }
        
        response = self._make_request('POST', '/api/device/control', json=data)
        return response.get('success', False)
    
    def return_home(self) -> bool:
        """
        返回原点位置
        
        Returns:
            bool: 返回是否成功
        """
        return self.move_device(0, 0, 0)
    
    # ==================== 语音控制 ====================
    
    def listen_voice_command(self, timeout: float = 5.0) -> VoiceCommandResult:
        """
        监听语音命令
        
        Args:
            timeout: 监听超时时间（秒）
            
        Returns:
            VoiceCommandResult: 语音命令结果
        """
        data = {'timeout': timeout}
        
        try:
            response = self._make_request('POST', '/api/voice/listen', json=data)
            
            if response.get('success'):
                command_data = response.get('data', {})
                command_info = command_data.get('command', {})
                
                return VoiceCommandResult(
                    success=True,
                    command_text=command_info.get('command_text'),
                    confidence=command_info.get('confidence'),
                    actions_taken=command_data.get('result', {}).get('actions', [])
                )
            else:
                return VoiceCommandResult(
                    success=False,
                    error_message=response.get('message', '语音命令处理失败')
                )
                
        except YOLOSClientError as e:
            return VoiceCommandResult(
                success=False,
                error_message=str(e)
            )
    
    def start_voice_listening_async(self, callback: Callable[[VoiceCommandResult], None]):
        """
        异步开始语音监听
        
        Args:
            callback: 语音命令结果回调函数
        """
        if not self.sio or not self.websocket_connected:
            if not self.connect_websocket():
                raise YOLOSClientError("WebSocket连接失败，无法启动异步语音监听")
        
        def voice_handler(data):
            result = VoiceCommandResult(
                success=True,
                command_text=data.get('command', {}).get('command_text'),
                confidence=data.get('command', {}).get('confidence'),
                actions_taken=data.get('result', {}).get('actions', [])
            )
            callback(result)
        
        self.set_event_handler('voice_command', voice_handler)
        self.sio.emit('start_voice_listening', {'timeout': 10.0})
    
    # ==================== 识别任务 ====================
    
    def recognize_image(self, image: Union[str, np.ndarray, Image.Image], 
                       task_type: str = "general_recognition",
                       **kwargs) -> RecognitionResult:
        """
        识别图像
        
        Args:
            image: 图像数据（文件路径、numpy数组或PIL图像）
            task_type: 识别任务类型
            **kwargs: 其他参数
            
        Returns:
            RecognitionResult: 识别结果
        """
        # 处理图像数据
        image_base64 = self._prepare_image_data(image)
        
        data = {
            'image_base64': image_base64,
            'task_type': task_type,
            **kwargs
        }
        
        try:
            response = self._make_request('POST', '/api/recognition/image', json=data)
            
            if response.get('success'):
                result_data = response.get('data', {})
                
                return RecognitionResult(
                    success=True,
                    detected_objects=result_data.get('detected_objects', []),
                    processing_time=result_data.get('processing_time'),
                    timestamp=datetime.now()
                )
            else:
                return RecognitionResult(
                    success=False,
                    error_message=response.get('message', '图像识别失败')
                )
                
        except YOLOSClientError as e:
            return RecognitionResult(
                success=False,
                error_message=str(e)
            )
    
    def start_recognition_task(self, task_type: str, parameters: Dict[str, Any] = None,
                              priority: int = 5, callback: Callable = None) -> str:
        """
        启动识别任务
        
        Args:
            task_type: 任务类型
            parameters: 任务参数
            priority: 任务优先级（1-10）
            callback: 任务完成回调函数
            
        Returns:
            str: 任务ID
        """
        data = {
            'task_type': task_type,
            'parameters': parameters or {},
            'priority': priority
        }
        
        response = self._make_request('POST', '/api/recognition/start', json=data)
        
        if response.get('success'):
            task_id = response.get('data', {}).get('task_id')
            
            if callback:
                self.task_callbacks[task_id] = callback
            
            return task_id
        else:
            raise YOLOSClientError(response.get('message', '任务创建失败'))
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            Dict: 任务状态信息
        """
        response = self._make_request('GET', f'/api/recognition/status/{task_id}')
        
        if response.get('success'):
            return response.get('data', {})
        else:
            raise YOLOSClientError(response.get('message', '任务状态查询失败'))
    
    def wait_for_task_completion(self, task_id: str, timeout: float = 60.0) -> Dict[str, Any]:
        """
        等待任务完成
        
        Args:
            task_id: 任务ID
            timeout: 超时时间（秒）
            
        Returns:
            Dict: 任务结果
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                status = self.get_task_status(task_id)
                
                if status.get('status') == 'completed':
                    return status.get('result', {})
                elif status.get('status') == 'failed':
                    raise YOLOSClientError(f"任务执行失败: {status.get('result', {}).get('error', '未知错误')}")
                
                time.sleep(1.0)  # 等待1秒后重试
                
            except YOLOSClientError:
                # 任务可能已完成并从活动任务中移除
                break
        
        raise YOLOSClientError(f"任务等待超时: {task_id}")
    
    # ==================== 专项识别功能 ====================
    
    def detect_medication(self, image: Union[str, np.ndarray, Image.Image]) -> RecognitionResult:
        """
        药物检测
        
        Args:
            image: 图像数据
            
        Returns:
            RecognitionResult: 药物检测结果
        """
        return self.recognize_image(image, task_type="medication_detection")
    
    def monitor_pet(self, image: Union[str, np.ndarray, Image.Image]) -> RecognitionResult:
        """
        宠物监控
        
        Args:
            image: 图像数据
            
        Returns:
            RecognitionResult: 宠物监控结果
        """
        return self.recognize_image(image, task_type="pet_monitoring")
    
    def detect_fall(self, image: Union[str, np.ndarray, Image.Image]) -> RecognitionResult:
        """
        跌倒检测
        
        Args:
            image: 图像数据
            
        Returns:
            RecognitionResult: 跌倒检测结果
        """
        return self.recognize_image(image, task_type="fall_detection")
    
    def security_surveillance(self, image: Union[str, np.ndarray, Image.Image]) -> RecognitionResult:
        """
        安全监控
        
        Args:
            image: 图像数据
            
        Returns:
            RecognitionResult: 安全监控结果
        """
        return self.recognize_image(image, task_type="security_surveillance")
    
    def analyze_medical_condition(self, image: Union[str, np.ndarray, Image.Image]) -> RecognitionResult:
        """
        医疗状况分析
        
        Args:
            image: 图像数据
            
        Returns:
            RecognitionResult: 医疗分析结果
        """
        return self.recognize_image(image, task_type="medical_analysis")
    
    def recognize_gesture(self, image: Union[str, np.ndarray, Image.Image]) -> RecognitionResult:
        """
        手势识别
        
        Args:
            image: 图像数据
            
        Returns:
            RecognitionResult: 手势识别结果
        """
        return self.recognize_image(image, task_type="gesture_recognition")
    
    # ==================== 便捷方法 ====================
    
    def execute_voice_command(self, command_text: str) -> Dict[str, Any]:
        """
        执行语音命令（文本形式）
        
        Args:
            command_text: 命令文本
            
        Returns:
            Dict: 执行结果
        """
        # 解析命令并执行相应操作
        command_text = command_text.strip().lower()
        
        # 移动命令
        if "移动到客厅" in command_text or "去客厅" in command_text:
            success = self.move_device_to_location("客厅")
            return {"success": success, "action": "move_to_living_room"}
        
        elif "移动到卧室" in command_text or "去卧室" in command_text:
            success = self.move_device_to_location("卧室")
            return {"success": success, "action": "move_to_bedroom"}
        
        elif "移动到厨房" in command_text or "去厨房" in command_text:
            success = self.move_device_to_location("厨房")
            return {"success": success, "action": "move_to_kitchen"}
        
        # 拍照命令
        elif "拍照" in command_text or "拍张照片" in command_text:
            result = self.take_photo()
            return {"success": True, "action": "take_photo", "result": result}
        
        # 识别命令
        elif "识别药物" in command_text or "检查药品" in command_text:
            # 先拍照，然后识别
            photo_result = self.take_photo()
            if photo_result:
                # 这里应该获取拍摄的图像进行识别
                # 现在只是返回任务ID
                task_id = self.start_recognition_task("medication_detection")
                return {"success": True, "action": "medication_detection", "task_id": task_id}
        
        elif "监控宠物" in command_text or "看看宠物" in command_text:
            task_id = self.start_recognition_task("pet_monitoring")
            return {"success": True, "action": "pet_monitoring", "task_id": task_id}
        
        # 返回原点
        elif "回到原点" in command_text or "返回" in command_text:
            success = self.return_home()
            return {"success": success, "action": "return_home"}
        
        else:
            return {"success": False, "error": f"未识别的命令: {command_text}"}
    
    def batch_recognition(self, images: List[Union[str, np.ndarray, Image.Image]], 
                         task_type: str = "general_recognition") -> List[RecognitionResult]:
        """
        批量图像识别
        
        Args:
            images: 图像列表
            task_type: 识别任务类型
            
        Returns:
            List[RecognitionResult]: 识别结果列表
        """
        results = []
        
        for image in images:
            try:
                result = self.recognize_image(image, task_type)
                results.append(result)
            except Exception as e:
                results.append(RecognitionResult(
                    success=False,
                    error_message=str(e)
                ))
        
        return results
    
    def _prepare_image_data(self, image: Union[str, np.ndarray, Image.Image]) -> str:
        """
        准备图像数据为base64格式
        
        Args:
            image: 图像数据
            
        Returns:
            str: base64编码的图像数据
        """
        if isinstance(image, str):
            # 文件路径
            if not os.path.exists(image):
                raise YOLOSClientError(f"图像文件不存在: {image}")
            
            with open(image, 'rb') as f:
                image_data = f.read()
        
        elif isinstance(image, np.ndarray):
            # OpenCV图像
            _, buffer = cv2.imencode('.jpg', image)
            image_data = buffer.tobytes()
        
        elif isinstance(image, Image.Image):
            # PIL图像
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            image_data = buffer.getvalue()
        
        else:
            raise YOLOSClientError(f"不支持的图像类型: {type(image)}")
        
        return base64.b64encode(image_data).decode('utf-8')
    
    def close(self):
        """关闭客户端连接"""
        if self.sio and self.websocket_connected:
            self.disconnect_websocket()
        
        if self.session:
            self.session.close()
        
        self.logger.info("YOLOS客户端已关闭")

# ==================== 便捷函数 ====================

def create_client(api_url: str = "http://localhost:8080", 
                 api_key: str = None, 
                 enable_websocket: bool = True) -> YOLOSClient:
    """
    创建YOLOS客户端实例
    
    Args:
        api_url: API服务地址
        api_key: API密钥
        enable_websocket: 是否启用WebSocket
        
    Returns:
        YOLOSClient: 客户端实例
    """
    config = YOLOSConfig(
        api_base_url=api_url,
        api_key=api_key,
        enable_websocket=enable_websocket
    )
    
    return YOLOSClient(config)

def quick_recognition(image_path: str, 
                     task_type: str = "general_recognition",
                     api_url: str = "http://localhost:8080") -> RecognitionResult:
    """
    快速图像识别
    
    Args:
        image_path: 图像文件路径
        task_type: 识别任务类型
        api_url: API服务地址
        
    Returns:
        RecognitionResult: 识别结果
    """
    client = create_client(api_url, enable_websocket=False)
    
    try:
        result = client.recognize_image(image_path, task_type)
        return result
    finally:
        client.close()

def voice_control_demo():
    """语音控制演示"""
    client = create_client()
    
    try:
        print("YOLOS语音控制演示")
        print("说出命令，如：'移动到客厅'、'拍照'、'识别药物'等")
        
        while True:
            print("\n正在监听语音命令...")
            result = client.listen_voice_command(timeout=10.0)
            
            if result.success:
                print(f"识别到命令: {result.command_text}")
                print(f"置信度: {result.confidence:.2f}")
                print(f"执行的操作: {result.actions_taken}")
            else:
                print(f"语音识别失败: {result.error_message}")
            
            # 询问是否继续
            continue_input = input("继续监听？(y/n): ").strip().lower()
            if continue_input != 'y':
                break
    
    except KeyboardInterrupt:
        print("\n演示结束")
    finally:
        client.close()

if __name__ == "__main__":
    # 运行语音控制演示
    voice_control_demo()