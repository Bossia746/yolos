#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOS 外部接口系统
支持语音控制、设备移动、专项识别等功能的RESTful API
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import base64
import io

# Web框架
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading
import queue
import time

# 语音处理
try:
    import speech_recognition as sr
    import pyttsx3
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
    print("Warning: Speech recognition not available. Install: pip install SpeechRecognition pyttsx3")

# 图像处理
import cv2
import numpy as np
from PIL import Image

# 项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from recognition.priority_recognition_system import PriorityRecognitionSystem
    from recognition.intelligent_multi_target_system import IntelligentMultiTargetSystem
    from core.integrated_aiot_platform import IntegratedAIoTPlatform
    from communication.external_communication_system import ExternalCommunicationSystem
    from utils.logger import setup_logger
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")

class TaskType(Enum):
    """任务类型枚举"""
    GENERAL_RECOGNITION = "general_recognition"
    MEDICATION_DETECTION = "medication_detection"
    PET_MONITORING = "pet_monitoring"
    FALL_DETECTION = "fall_detection"
    SECURITY_SURVEILLANCE = "security_surveillance"
    PLANT_MONITORING = "plant_monitoring"
    OBJECT_INVENTORY = "object_inventory"
    MEDICAL_ANALYSIS = "medical_analysis"
    GESTURE_RECOGNITION = "gesture_recognition"
    TRAFFIC_MONITORING = "traffic_monitoring"

class DeviceCommand(Enum):
    """设备控制命令枚举"""
    MOVE_TO_POSITION = "move_to_position"
    ROTATE_CAMERA = "rotate_camera"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    FOCUS_ADJUST = "focus_adjust"
    LIGHTING_CONTROL = "lighting_control"
    START_RECORDING = "start_recording"
    STOP_RECORDING = "stop_recording"
    TAKE_PHOTO = "take_photo"
    RETURN_HOME = "return_home"

@dataclass
class VoiceCommand:
    """语音命令数据结构"""
    command_text: str
    confidence: float
    timestamp: datetime
    task_type: Optional[TaskType] = None
    device_command: Optional[DeviceCommand] = None
    parameters: Optional[Dict[str, Any]] = None
    target_location: Optional[Dict[str, float]] = None

@dataclass
class RecognitionTask:
    """识别任务数据结构"""
    task_id: str
    task_type: TaskType
    priority: int
    parameters: Dict[str, Any]
    created_at: datetime
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    device_commands: Optional[List[DeviceCommand]] = None

@dataclass
class APIResponse:
    """API响应数据结构"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error_code: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class VoiceCommandProcessor:
    """语音命令处理器"""
    
    def __init__(self):
        self.logger = setup_logger("VoiceCommandProcessor")
        
        if SPEECH_AVAILABLE:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            self.tts_engine = pyttsx3.init()
            self._setup_tts()
        else:
            self.recognizer = None
            self.microphone = None
            self.tts_engine = None
        
        # 语音命令模式映射
        self.command_patterns = {
            # 移动控制命令
            "移动到": DeviceCommand.MOVE_TO_POSITION,
            "去到": DeviceCommand.MOVE_TO_POSITION,
            "前往": DeviceCommand.MOVE_TO_POSITION,
            "转向": DeviceCommand.ROTATE_CAMERA,
            "旋转": DeviceCommand.ROTATE_CAMERA,
            "放大": DeviceCommand.ZOOM_IN,
            "缩小": DeviceCommand.ZOOM_OUT,
            "拍照": DeviceCommand.TAKE_PHOTO,
            "录像": DeviceCommand.START_RECORDING,
            "停止录像": DeviceCommand.STOP_RECORDING,
            "回到原点": DeviceCommand.RETURN_HOME,
            "返回": DeviceCommand.RETURN_HOME,
            
            # 识别任务命令
            "识别药物": TaskType.MEDICATION_DETECTION,
            "检查药品": TaskType.MEDICATION_DETECTION,
            "监控宠物": TaskType.PET_MONITORING,
            "看看宠物": TaskType.PET_MONITORING,
            "检测跌倒": TaskType.FALL_DETECTION,
            "安全监控": TaskType.SECURITY_SURVEILLANCE,
            "监控植物": TaskType.PLANT_MONITORING,
            "检查植物": TaskType.PLANT_MONITORING,
            "清点物品": TaskType.OBJECT_INVENTORY,
            "医疗分析": TaskType.MEDICAL_ANALYSIS,
            "健康检查": TaskType.MEDICAL_ANALYSIS,
            "手势识别": TaskType.GESTURE_RECOGNITION,
            "交通监控": TaskType.TRAFFIC_MONITORING,
        }
        
        # 位置关键词映射
        self.location_keywords = {
            "客厅": {"x": 0, "y": 0, "z": 0},
            "卧室": {"x": 5, "y": 0, "z": 0},
            "厨房": {"x": 0, "y": 5, "z": 0},
            "阳台": {"x": 5, "y": 5, "z": 0},
            "门口": {"x": -2, "y": 0, "z": 0},
            "窗边": {"x": 0, "y": -2, "z": 0},
            "左边": {"x": -1, "y": 0, "z": 0},
            "右边": {"x": 1, "y": 0, "z": 0},
            "前面": {"x": 0, "y": 1, "z": 0},
            "后面": {"x": 0, "y": -1, "z": 0},
        }
    
    def _setup_tts(self):
        """设置TTS引擎"""
        if self.tts_engine:
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # 尝试设置中文语音
                for voice in voices:
                    if 'chinese' in voice.name.lower() or 'zh' in voice.id.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            
            self.tts_engine.setProperty('rate', 150)  # 语速
            self.tts_engine.setProperty('volume', 0.8)  # 音量
    
    def listen_for_command(self, timeout: float = 5.0) -> Optional[VoiceCommand]:
        """监听语音命令"""
        if not SPEECH_AVAILABLE:
            self.logger.warning("语音识别不可用")
            return None
        
        try:
            with self.microphone as source:
                self.logger.info("正在监听语音命令...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
            
            # 识别语音
            command_text = self.recognizer.recognize_google(audio, language='zh-CN')
            confidence = 0.8  # Google API不提供置信度，使用默认值
            
            self.logger.info(f"识别到语音命令: {command_text}")
            
            # 解析命令
            voice_command = self._parse_command(command_text, confidence)
            return voice_command
            
        except sr.WaitTimeoutError:
            self.logger.info("语音监听超时")
            return None
        except sr.UnknownValueError:
            self.logger.warning("无法识别语音内容")
            return None
        except sr.RequestError as e:
            self.logger.error(f"语音识别服务错误: {e}")
            return None
        except Exception as e:
            self.logger.error(f"语音识别异常: {e}")
            return None
    
    def _parse_command(self, command_text: str, confidence: float) -> VoiceCommand:
        """解析语音命令"""
        command_text = command_text.strip()
        task_type = None
        device_command = None
        parameters = {}
        target_location = None
        
        # 检查设备控制命令
        for keyword, cmd in self.command_patterns.items():
            if keyword in command_text:
                if isinstance(cmd, DeviceCommand):
                    device_command = cmd
                    # 解析位置信息
                    if cmd == DeviceCommand.MOVE_TO_POSITION:
                        target_location = self._extract_location(command_text)
                elif isinstance(cmd, TaskType):
                    task_type = cmd
                break
        
        # 提取其他参数
        if "持续" in command_text:
            # 提取持续时间
            import re
            duration_match = re.search(r'持续(\d+)(分钟|秒)', command_text)
            if duration_match:
                duration = int(duration_match.group(1))
                unit = duration_match.group(2)
                if unit == "分钟":
                    duration *= 60
                parameters['duration'] = duration
        
        return VoiceCommand(
            command_text=command_text,
            confidence=confidence,
            timestamp=datetime.now(),
            task_type=task_type,
            device_command=device_command,
            parameters=parameters,
            target_location=target_location
        )
    
    def _extract_location(self, command_text: str) -> Optional[Dict[str, float]]:
        """从命令中提取位置信息"""
        for location_name, coordinates in self.location_keywords.items():
            if location_name in command_text:
                return coordinates
        
        # 尝试提取数字坐标
        import re
        coord_pattern = r'坐标\s*\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?'
        match = re.search(coord_pattern, command_text)
        if match:
            return {
                "x": float(match.group(1)),
                "y": float(match.group(2)),
                "z": 0.0
            }
        
        return None
    
    def speak(self, text: str):
        """语音播报"""
        if self.tts_engine:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                self.logger.error(f"语音播报失败: {e}")
        else:
            self.logger.info(f"TTS不可用，文本内容: {text}")

class ExternalAPISystem:
    """外部API系统"""
    
    def __init__(self, config_path: str = "config/external_api_config.yaml"):
        self.logger = setup_logger("ExternalAPISystem")
        self.config_path = config_path
        self.config = self._load_config()
        
        # 初始化Flask应用
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'yolos_api_secret_key'
        CORS(self.app)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # 初始化组件
        self.voice_processor = VoiceCommandProcessor()
        self.recognition_system = None
        self.multi_target_system = None
        self.aiot_platform = None
        self.communication_system = None
        
        # 任务管理
        self.task_queue = queue.Queue()
        self.active_tasks = {}
        self.task_results = {}
        
        # 设备状态
        self.device_status = {
            "position": {"x": 0, "y": 0, "z": 0},
            "camera_angle": {"pan": 0, "tilt": 0},
            "zoom_level": 1.0,
            "recording": False,
            "online": True,
            "battery_level": 100
        }
        
        # 注册API路由
        self._register_routes()
        self._register_socketio_events()
        
        # 启动后台任务处理线程
        self.task_processor_thread = threading.Thread(target=self._process_tasks, daemon=True)
        self.task_processor_thread.start()
        
        self.logger.info("外部API系统初始化完成")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = {
            "api": {
                "host": "0.0.0.0",
                "port": 8080,
                "debug": False
            },
            "voice": {
                "enabled": True,
                "language": "zh-CN",
                "timeout": 5.0
            },
            "device": {
                "max_move_distance": 10.0,
                "move_speed": 1.0,
                "camera_range": {"pan": 180, "tilt": 90},
                "zoom_range": {"min": 0.5, "max": 5.0}
            },
            "recognition": {
                "default_confidence_threshold": 0.6,
                "max_concurrent_tasks": 5
            }
        }
        
        try:
            import yaml
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    # 合并默认配置
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
        except Exception as e:
            self.logger.warning(f"配置文件加载失败，使用默认配置: {e}")
        
        return default_config
    
    def _register_routes(self):
        """注册API路由"""
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """健康检查"""
            return jsonify(asdict(APIResponse(
                success=True,
                message="API服务正常运行",
                data={
                    "version": "2.0.0",
                    "status": "healthy",
                    "uptime": time.time() - self.start_time if hasattr(self, 'start_time') else 0
                }
            )))
        
        @self.app.route('/api/voice/listen', methods=['POST'])
        def listen_voice_command():
            """监听语音命令"""
            try:
                data = request.get_json() or {}
                timeout = data.get('timeout', self.config['voice']['timeout'])
                
                voice_command = self.voice_processor.listen_for_command(timeout)
                
                if voice_command:
                    # 处理语音命令
                    result = self._handle_voice_command(voice_command)
                    
                    return jsonify(asdict(APIResponse(
                        success=True,
                        message="语音命令处理成功",
                        data={
                            "command": asdict(voice_command),
                            "result": result
                        }
                    )))
                else:
                    return jsonify(asdict(APIResponse(
                        success=False,
                        message="未检测到语音命令",
                        error_code="NO_VOICE_DETECTED"
                    )))
                    
            except Exception as e:
                self.logger.error(f"语音命令处理异常: {e}")
                return jsonify(asdict(APIResponse(
                    success=False,
                    message=f"语音命令处理失败: {str(e)}",
                    error_code="VOICE_PROCESSING_ERROR"
                )))
        
        @self.app.route('/api/device/move', methods=['POST'])
        def move_device():
            """移动设备到指定位置"""
            try:
                data = request.get_json()
                target_position = data.get('position', {})
                
                result = self._move_device(target_position)
                
                return jsonify(asdict(APIResponse(
                    success=result['success'],
                    message=result['message'],
                    data=result.get('data')
                )))
                
            except Exception as e:
                self.logger.error(f"设备移动异常: {e}")
                return jsonify(asdict(APIResponse(
                    success=False,
                    message=f"设备移动失败: {str(e)}",
                    error_code="DEVICE_MOVE_ERROR"
                )))
        
        @self.app.route('/api/device/control', methods=['POST'])
        def control_device():
            """设备控制"""
            try:
                data = request.get_json()
                command = data.get('command')
                parameters = data.get('parameters', {})
                
                result = self._control_device(command, parameters)
                
                return jsonify(asdict(APIResponse(
                    success=result['success'],
                    message=result['message'],
                    data=result.get('data')
                )))
                
            except Exception as e:
                self.logger.error(f"设备控制异常: {e}")
                return jsonify(asdict(APIResponse(
                    success=False,
                    message=f"设备控制失败: {str(e)}",
                    error_code="DEVICE_CONTROL_ERROR"
                )))
        
        @self.app.route('/api/recognition/start', methods=['POST'])
        def start_recognition_task():
            """启动识别任务"""
            try:
                data = request.get_json()
                task_type = data.get('task_type')
                parameters = data.get('parameters', {})
                priority = data.get('priority', 5)
                
                task = self._create_recognition_task(task_type, parameters, priority)
                
                return jsonify(asdict(APIResponse(
                    success=True,
                    message="识别任务已创建",
                    data={
                        "task_id": task.task_id,
                        "task": asdict(task)
                    }
                )))
                
            except Exception as e:
                self.logger.error(f"识别任务创建异常: {e}")
                return jsonify(asdict(APIResponse(
                    success=False,
                    message=f"识别任务创建失败: {str(e)}",
                    error_code="TASK_CREATION_ERROR"
                )))
        
        @self.app.route('/api/recognition/status/<task_id>', methods=['GET'])
        def get_task_status(task_id):
            """获取任务状态"""
            try:
                if task_id in self.active_tasks:
                    task = self.active_tasks[task_id]
                    return jsonify(asdict(APIResponse(
                        success=True,
                        message="任务状态获取成功",
                        data=asdict(task)
                    )))
                elif task_id in self.task_results:
                    result = self.task_results[task_id]
                    return jsonify(asdict(APIResponse(
                        success=True,
                        message="任务已完成",
                        data=result
                    )))
                else:
                    return jsonify(asdict(APIResponse(
                        success=False,
                        message="任务不存在",
                        error_code="TASK_NOT_FOUND"
                    )))
                    
            except Exception as e:
                self.logger.error(f"任务状态查询异常: {e}")
                return jsonify(asdict(APIResponse(
                    success=False,
                    message=f"任务状态查询失败: {str(e)}",
                    error_code="STATUS_QUERY_ERROR"
                )))
        
        @self.app.route('/api/device/status', methods=['GET'])
        def get_device_status():
            """获取设备状态"""
            return jsonify(asdict(APIResponse(
                success=True,
                message="设备状态获取成功",
                data=self.device_status
            )))
        
        @self.app.route('/api/recognition/image', methods=['POST'])
        def recognize_image():
            """图像识别接口"""
            try:
                # 处理上传的图像
                if 'image' in request.files:
                    image_file = request.files['image']
                    image_data = image_file.read()
                elif 'image_base64' in request.get_json():
                    image_base64 = request.get_json()['image_base64']
                    image_data = base64.b64decode(image_base64)
                else:
                    return jsonify(asdict(APIResponse(
                        success=False,
                        message="未提供图像数据",
                        error_code="NO_IMAGE_DATA"
                    )))
                
                # 转换为OpenCV格式
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    return jsonify(asdict(APIResponse(
                        success=False,
                        message="图像格式无效",
                        error_code="INVALID_IMAGE_FORMAT"
                    )))
                
                # 获取识别参数
                data = request.get_json() or {}
                task_type = data.get('task_type', 'general_recognition')
                
                # 执行识别
                result = self._process_image_recognition(image, task_type, data)
                
                return jsonify(asdict(APIResponse(
                    success=True,
                    message="图像识别完成",
                    data=result
                )))
                
            except Exception as e:
                self.logger.error(f"图像识别异常: {e}")
                return jsonify(asdict(APIResponse(
                    success=False,
                    message=f"图像识别失败: {str(e)}",
                    error_code="IMAGE_RECOGNITION_ERROR"
                )))
    
    def _register_socketio_events(self):
        """注册WebSocket事件"""
        
        @self.socketio.on('connect')
        def handle_connect():
            self.logger.info(f"客户端连接: {request.sid}")
            emit('connected', {'message': 'WebSocket连接成功'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.logger.info(f"客户端断开: {request.sid}")
        
        @self.socketio.on('start_voice_listening')
        def handle_start_voice_listening(data):
            """开始语音监听"""
            try:
                timeout = data.get('timeout', 5.0)
                voice_command = self.voice_processor.listen_for_command(timeout)
                
                if voice_command:
                    result = self._handle_voice_command(voice_command)
                    emit('voice_command_received', {
                        'command': asdict(voice_command),
                        'result': result
                    })
                else:
                    emit('voice_command_timeout', {'message': '未检测到语音命令'})
                    
            except Exception as e:
                emit('voice_command_error', {'error': str(e)})
        
        @self.socketio.on('device_control')
        def handle_device_control(data):
            """设备控制"""
            try:
                command = data.get('command')
                parameters = data.get('parameters', {})
                
                result = self._control_device(command, parameters)
                emit('device_control_result', result)
                
            except Exception as e:
                emit('device_control_error', {'error': str(e)})
    
    def _handle_voice_command(self, voice_command: VoiceCommand) -> Dict[str, Any]:
        """处理语音命令"""
        result = {"command_processed": False, "actions": []}
        
        try:
            # 语音反馈
            self.voice_processor.speak(f"收到命令: {voice_command.command_text}")
            
            # 处理设备控制命令
            if voice_command.device_command:
                if voice_command.device_command == DeviceCommand.MOVE_TO_POSITION:
                    if voice_command.target_location:
                        move_result = self._move_device(voice_command.target_location)
                        result["actions"].append({
                            "type": "device_move",
                            "result": move_result
                        })
                        
                        if move_result["success"]:
                            self.voice_processor.speak("正在移动到指定位置")
                        else:
                            self.voice_processor.speak("移动失败")
                
                elif voice_command.device_command == DeviceCommand.TAKE_PHOTO:
                    photo_result = self._take_photo()
                    result["actions"].append({
                        "type": "take_photo",
                        "result": photo_result
                    })
                    self.voice_processor.speak("已拍照")
                
                # 其他设备控制命令...
                
            # 处理识别任务命令
            if voice_command.task_type:
                task = self._create_recognition_task(
                    voice_command.task_type.value,
                    voice_command.parameters or {},
                    priority=8  # 语音命令优先级较高
                )
                
                result["actions"].append({
                    "type": "recognition_task",
                    "task_id": task.task_id,
                    "task_type": task.task_type.value
                })
                
                self.voice_processor.speak(f"开始执行{self._get_task_name(voice_command.task_type)}任务")
            
            result["command_processed"] = True
            
        except Exception as e:
            self.logger.error(f"语音命令处理异常: {e}")
            result["error"] = str(e)
            self.voice_processor.speak("命令处理失败")
        
        return result
    
    def _get_task_name(self, task_type: TaskType) -> str:
        """获取任务中文名称"""
        task_names = {
            TaskType.MEDICATION_DETECTION: "药物识别",
            TaskType.PET_MONITORING: "宠物监控",
            TaskType.FALL_DETECTION: "跌倒检测",
            TaskType.SECURITY_SURVEILLANCE: "安全监控",
            TaskType.PLANT_MONITORING: "植物监控",
            TaskType.OBJECT_INVENTORY: "物品清点",
            TaskType.MEDICAL_ANALYSIS: "医疗分析",
            TaskType.GESTURE_RECOGNITION: "手势识别",
            TaskType.TRAFFIC_MONITORING: "交通监控",
            TaskType.GENERAL_RECOGNITION: "通用识别"
        }
        return task_names.get(task_type, "未知任务")
    
    def _move_device(self, target_position: Dict[str, float]) -> Dict[str, Any]:
        """移动设备"""
        try:
            current_pos = self.device_status["position"]
            
            # 计算移动距离
            distance = ((target_position.get("x", 0) - current_pos["x"]) ** 2 + 
                       (target_position.get("y", 0) - current_pos["y"]) ** 2) ** 0.5
            
            max_distance = self.config["device"]["max_move_distance"]
            if distance > max_distance:
                return {
                    "success": False,
                    "message": f"移动距离超出限制 ({distance:.2f} > {max_distance})"
                }
            
            # 模拟移动过程
            move_speed = self.config["device"]["move_speed"]
            move_time = distance / move_speed
            
            # 更新设备位置
            self.device_status["position"].update(target_position)
            
            # 通过WebSocket广播位置更新
            self.socketio.emit('device_position_updated', {
                'position': self.device_status["position"],
                'move_time': move_time
            })
            
            return {
                "success": True,
                "message": "设备移动成功",
                "data": {
                    "new_position": self.device_status["position"],
                    "move_distance": distance,
                    "move_time": move_time
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"设备移动失败: {str(e)}"
            }
    
    def _control_device(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """控制设备"""
        try:
            if command == "rotate_camera":
                pan = parameters.get("pan", 0)
                tilt = parameters.get("tilt", 0)
                
                # 检查角度限制
                camera_range = self.config["device"]["camera_range"]
                if abs(pan) > camera_range["pan"] or abs(tilt) > camera_range["tilt"]:
                    return {
                        "success": False,
                        "message": "摄像头角度超出范围"
                    }
                
                self.device_status["camera_angle"] = {"pan": pan, "tilt": tilt}
                
                return {
                    "success": True,
                    "message": "摄像头旋转成功",
                    "data": {"camera_angle": self.device_status["camera_angle"]}
                }
            
            elif command == "zoom":
                zoom_level = parameters.get("level", 1.0)
                zoom_range = self.config["device"]["zoom_range"]
                
                if zoom_level < zoom_range["min"] or zoom_level > zoom_range["max"]:
                    return {
                        "success": False,
                        "message": "缩放级别超出范围"
                    }
                
                self.device_status["zoom_level"] = zoom_level
                
                return {
                    "success": True,
                    "message": "缩放调整成功",
                    "data": {"zoom_level": zoom_level}
                }
            
            elif command == "start_recording":
                self.device_status["recording"] = True
                return {
                    "success": True,
                    "message": "开始录像"
                }
            
            elif command == "stop_recording":
                self.device_status["recording"] = False
                return {
                    "success": True,
                    "message": "停止录像"
                }
            
            else:
                return {
                    "success": False,
                    "message": f"未知设备命令: {command}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"设备控制失败: {str(e)}"
            }
    
    def _take_photo(self) -> Dict[str, Any]:
        """拍照"""
        try:
            # 模拟拍照过程
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"photo_{timestamp}.jpg"
            
            # 这里应该调用实际的摄像头拍照功能
            # 现在只是模拟
            
            return {
                "success": True,
                "message": "拍照成功",
                "data": {
                    "filename": filename,
                    "timestamp": timestamp,
                    "position": self.device_status["position"].copy(),
                    "camera_angle": self.device_status["camera_angle"].copy()
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"拍照失败: {str(e)}"
            }
    
    def _create_recognition_task(self, task_type: str, parameters: Dict[str, Any], priority: int = 5) -> RecognitionTask:
        """创建识别任务"""
        task_id = f"task_{int(time.time() * 1000)}"
        
        task = RecognitionTask(
            task_id=task_id,
            task_type=TaskType(task_type),
            priority=priority,
            parameters=parameters,
            created_at=datetime.now()
        )
        
        # 添加到任务队列
        self.task_queue.put(task)
        self.active_tasks[task_id] = task
        
        self.logger.info(f"创建识别任务: {task_id}, 类型: {task_type}")
        
        return task
    
    def _process_tasks(self):
        """后台任务处理线程"""
        while True:
            try:
                # 获取任务（阻塞等待）
                task = self.task_queue.get(timeout=1.0)
                
                self.logger.info(f"开始处理任务: {task.task_id}")
                
                # 更新任务状态
                task.status = "processing"
                
                # 执行任务
                result = self._execute_recognition_task(task)
                
                # 更新任务结果
                task.status = "completed"
                task.result = result
                
                # 移动到结果存储
                self.task_results[task.task_id] = asdict(task)
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
                
                # 通过WebSocket广播任务完成
                self.socketio.emit('task_completed', {
                    'task_id': task.task_id,
                    'result': result
                })
                
                self.logger.info(f"任务完成: {task.task_id}")
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"任务处理异常: {e}")
                if 'task' in locals():
                    task.status = "failed"
                    task.result = {"error": str(e)}
    
    def _execute_recognition_task(self, task: RecognitionTask) -> Dict[str, Any]:
        """执行识别任务"""
        try:
            # 这里应该调用实际的识别系统
            # 现在只是模拟结果
            
            task_type = task.task_type
            parameters = task.parameters
            
            # 模拟处理时间
            processing_time = np.random.uniform(1.0, 3.0)
            time.sleep(processing_time)
            
            # 生成模拟结果
            if task_type == TaskType.MEDICATION_DETECTION:
                result = {
                    "detected_medications": [
                        {
                            "name": "阿司匹林",
                            "confidence": 0.92,
                            "dosage": "100mg",
                            "expiry_date": "2025-12-31"
                        }
                    ],
                    "total_count": 1,
                    "processing_time": processing_time
                }
            
            elif task_type == TaskType.PET_MONITORING:
                result = {
                    "detected_pets": [
                        {
                            "species": "cat",
                            "breed": "橘猫",
                            "confidence": 0.88,
                            "activity": "sleeping",
                            "health_status": "normal"
                        }
                    ],
                    "total_count": 1,
                    "processing_time": processing_time
                }
            
            elif task_type == TaskType.FALL_DETECTION:
                result = {
                    "fall_detected": False,
                    "human_count": 1,
                    "pose_analysis": {
                        "standing": True,
                        "confidence": 0.95
                    },
                    "processing_time": processing_time
                }
            
            else:
                result = {
                    "task_type": task_type.value,
                    "status": "completed",
                    "processing_time": processing_time,
                    "message": f"{task_type.value}任务执行完成"
                }
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "status": "failed"
            }
    
    def _process_image_recognition(self, image: np.ndarray, task_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """处理图像识别"""
        try:
            # 这里应该调用实际的识别系统
            # 现在只是返回模拟结果
            
            height, width = image.shape[:2]
            
            result = {
                "image_info": {
                    "width": width,
                    "height": height,
                    "channels": image.shape[2] if len(image.shape) > 2 else 1
                },
                "task_type": task_type,
                "detected_objects": [
                    {
                        "category": "human",
                        "confidence": 0.89,
                        "bbox": [100, 100, 200, 300]
                    }
                ],
                "processing_time": 0.5,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            raise Exception(f"图像识别处理失败: {str(e)}")
    
    def run(self, host: str = None, port: int = None, debug: bool = None):
        """启动API服务"""
        self.start_time = time.time()
        
        host = host or self.config["api"]["host"]
        port = port or self.config["api"]["port"]
        debug = debug if debug is not None else self.config["api"]["debug"]
        
        self.logger.info(f"启动外部API服务: http://{host}:{port}")
        
        # 初始化识别系统
        try:
            self.recognition_system = PriorityRecognitionSystem()
            self.multi_target_system = IntelligentMultiTargetSystem()
            self.logger.info("识别系统初始化成功")
        except Exception as e:
            self.logger.warning(f"识别系统初始化失败: {e}")
        
        # 启动服务
        self.socketio.run(self.app, host=host, port=port, debug=debug)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOS 外部API系统')
    parser.add_argument('--host', default='0.0.0.0', help='服务器地址')
    parser.add_argument('--port', type=int, default=8080, help='服务器端口')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    parser.add_argument('--config', default='config/external_api_config.yaml', help='配置文件路径')
    
    args = parser.parse_args()
    
    # 创建并启动API系统
    api_system = ExternalAPISystem(args.config)
    api_system.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()