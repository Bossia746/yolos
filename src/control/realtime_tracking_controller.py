#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时跟踪控制接口
与ESP32小车控制系统集成，实现人体跟随功能
解决人脸跟随在侧脸、背身等场景下的局限性
"""

import time
import json
import math
import logging
import threading
import queue
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import cv2

# 通信相关
try:
    import serial
    import socket
    import websocket
    import paho.mqtt.client as mqtt
except ImportError:
    # 可选依赖，根据实际需求安装
    pass

# 导入相关模块
try:
    from ..tracking.multimodal_fusion_tracker import MultimodalFusionTracker, TrackingTarget
    from ..models.human_body_detector import HumanBodyDetector, HumanBodyDetection
    from ..deployment.edge_device_optimizer import EdgeDeviceOptimizer
except ImportError:
    # 兼容性导入
    pass

logger = logging.getLogger(__name__)

class ControlMode(Enum):
    """控制模式"""
    FOLLOW = "follow"  # 跟随模式
    PATROL = "patrol"  # 巡逻模式
    MANUAL = "manual"  # 手动模式
    STANDBY = "standby"  # 待机模式
    EMERGENCY_STOP = "emergency_stop"  # 紧急停止

class MovementCommand(Enum):
    """运动指令"""
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"
    ROTATE_LEFT = "rotate_left"
    ROTATE_RIGHT = "rotate_right"
    STOP = "stop"
    SPEED_UP = "speed_up"
    SLOW_DOWN = "slow_down"

class CommunicationProtocol(Enum):
    """通信协议"""
    SERIAL = "serial"
    TCP = "tcp"
    UDP = "udp"
    WEBSOCKET = "websocket"
    MQTT = "mqtt"
    HTTP = "http"

@dataclass
class TargetInfo:
    """目标信息"""
    track_id: str
    position: Tuple[float, float]  # 图像坐标 (x, y)
    bbox: Tuple[int, int, int, int]  # 边界框 (x1, y1, x2, y2)
    confidence: float
    distance: float  # 估计距离（米）
    angle: float  # 相对角度（度）
    velocity: Tuple[float, float]  # 速度向量
    timestamp: float
    is_lost: bool = False
    missing_frames: int = 0

@dataclass
class ControlCommand:
    """控制指令"""
    command: MovementCommand
    speed: float  # 速度 [0, 1]
    duration: float  # 持续时间（秒）
    angle: float = 0.0  # 转向角度（度）
    timestamp: float = field(default_factory=time.time)
    priority: int = 1  # 优先级 [1-10]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FollowingConfig:
    """跟随配置"""
    # 跟随参数
    target_distance: float = 1.5  # 目标距离（米）
    distance_tolerance: float = 0.3  # 距离容忍度
    angle_tolerance: float = 10.0  # 角度容忍度（度）
    
    # 运动参数
    max_speed: float = 0.8  # 最大速度
    min_speed: float = 0.1  # 最小速度
    acceleration: float = 0.5  # 加速度
    turn_speed: float = 0.6  # 转向速度
    
    # 安全参数
    min_distance: float = 0.5  # 最小安全距离
    max_distance: float = 5.0  # 最大跟随距离
    obstacle_distance: float = 0.3  # 障碍物检测距离
    
    # 跟踪参数
    lost_timeout: float = 3.0  # 目标丢失超时（秒）
    search_timeout: float = 10.0  # 搜索超时（秒）
    confidence_threshold: float = 0.5  # 置信度阈值
    
    # 控制参数
    control_frequency: float = 10.0  # 控制频率（Hz）
    smooth_factor: float = 0.7  # 平滑因子
    pid_kp: float = 1.0  # PID比例系数
    pid_ki: float = 0.1  # PID积分系数
    pid_kd: float = 0.05  # PID微分系数

@dataclass
class CommunicationConfig:
    """通信配置"""
    protocol: CommunicationProtocol
    
    # 串口配置
    serial_port: str = "COM3"
    baud_rate: int = 115200
    
    # 网络配置
    host: str = "192.168.1.100"
    port: int = 8080
    
    # MQTT配置
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    mqtt_topic: str = "robot/control"
    
    # 其他配置
    timeout: float = 1.0
    retry_count: int = 3
    heartbeat_interval: float = 5.0

class PIDController:
    """PID控制器"""
    
    def __init__(self, kp: float, ki: float, kd: float, output_limits: Tuple[float, float] = (-1.0, 1.0)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        
        # 内部状态
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = time.time()
    
    def update(self, setpoint: float, measured_value: float) -> float:
        """更新PID控制器
        
        Args:
            setpoint: 设定值
            measured_value: 测量值
            
        Returns:
            float: 控制输出
        """
        current_time = time.time()
        dt = current_time - self.prev_time
        
        if dt <= 0.0:
            return 0.0
        
        # 计算误差
        error = setpoint - measured_value
        
        # 比例项
        proportional = self.kp * error
        
        # 积分项
        self.integral += error * dt
        integral = self.ki * self.integral
        
        # 微分项
        derivative = self.kd * (error - self.prev_error) / dt
        
        # 总输出
        output = proportional + integral + derivative
        
        # 限制输出范围
        output = max(self.output_limits[0], min(output, self.output_limits[1]))
        
        # 更新状态
        self.prev_error = error
        self.prev_time = current_time
        
        return output
    
    def reset(self):
        """重置PID控制器"""
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = time.time()

class CommunicationManager:
    """通信管理器"""
    
    def __init__(self, config: CommunicationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 连接状态
        self.is_connected = False
        self.connection = None
        
        # 消息队列
        self.send_queue = queue.Queue()
        self.receive_queue = queue.Queue()
        
        # 线程
        self.send_thread = None
        self.receive_thread = None
        self.heartbeat_thread = None
        
        # 统计信息
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "connection_errors": 0,
            "last_heartbeat": 0.0
        }
    
    def connect(self) -> bool:
        """建立连接
        
        Returns:
            bool: 是否连接成功
        """
        try:
            if self.config.protocol == CommunicationProtocol.SERIAL:
                return self._connect_serial()
            elif self.config.protocol == CommunicationProtocol.TCP:
                return self._connect_tcp()
            elif self.config.protocol == CommunicationProtocol.UDP:
                return self._connect_udp()
            elif self.config.protocol == CommunicationProtocol.WEBSOCKET:
                return self._connect_websocket()
            elif self.config.protocol == CommunicationProtocol.MQTT:
                return self._connect_mqtt()
            else:
                self.logger.error(f"不支持的通信协议: {self.config.protocol}")
                return False
                
        except Exception as e:
            self.logger.error(f"连接失败: {e}")
            self.stats["connection_errors"] += 1
            return False
    
    def _connect_serial(self) -> bool:
        """串口连接"""
        try:
            import serial
            self.connection = serial.Serial(
                port=self.config.serial_port,
                baudrate=self.config.baud_rate,
                timeout=self.config.timeout
            )
            self.is_connected = True
            self.logger.info(f"串口连接成功: {self.config.serial_port}")
            return True
        except Exception as e:
            self.logger.error(f"串口连接失败: {e}")
            return False
    
    def _connect_tcp(self) -> bool:
        """TCP连接"""
        try:
            import socket
            self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.connection.settimeout(self.config.timeout)
            self.connection.connect((self.config.host, self.config.port))
            self.is_connected = True
            self.logger.info(f"TCP连接成功: {self.config.host}:{self.config.port}")
            return True
        except Exception as e:
            self.logger.error(f"TCP连接失败: {e}")
            return False
    
    def _connect_udp(self) -> bool:
        """UDP连接"""
        try:
            import socket
            self.connection = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.connection.settimeout(self.config.timeout)
            self.is_connected = True
            self.logger.info(f"UDP连接成功: {self.config.host}:{self.config.port}")
            return True
        except Exception as e:
            self.logger.error(f"UDP连接失败: {e}")
            return False
    
    def _connect_websocket(self) -> bool:
        """WebSocket连接"""
        try:
            import websocket
            url = f"ws://{self.config.host}:{self.config.port}"
            self.connection = websocket.create_connection(url, timeout=self.config.timeout)
            self.is_connected = True
            self.logger.info(f"WebSocket连接成功: {url}")
            return True
        except Exception as e:
            self.logger.error(f"WebSocket连接失败: {e}")
            return False
    
    def _connect_mqtt(self) -> bool:
        """MQTT连接"""
        try:
            import paho.mqtt.client as mqtt
            self.connection = mqtt.Client()
            self.connection.connect(self.config.mqtt_broker, self.config.mqtt_port, 60)
            self.connection.loop_start()
            self.is_connected = True
            self.logger.info(f"MQTT连接成功: {self.config.mqtt_broker}:{self.config.mqtt_port}")
            return True
        except Exception as e:
            self.logger.error(f"MQTT连接失败: {e}")
            return False
    
    def send_command(self, command: ControlCommand) -> bool:
        """发送控制指令
        
        Args:
            command: 控制指令
            
        Returns:
            bool: 是否发送成功
        """
        if not self.is_connected:
            self.logger.warning("未连接，无法发送指令")
            return False
        
        try:
            # 将指令转换为JSON格式
            message = {
                "command": command.command.value,
                "speed": command.speed,
                "duration": command.duration,
                "angle": command.angle,
                "timestamp": command.timestamp,
                "priority": command.priority,
                "metadata": command.metadata
            }
            
            message_str = json.dumps(message)
            
            # 根据协议发送消息
            if self.config.protocol == CommunicationProtocol.SERIAL:
                self.connection.write((message_str + '\n').encode())
            elif self.config.protocol == CommunicationProtocol.TCP:
                self.connection.send((message_str + '\n').encode())
            elif self.config.protocol == CommunicationProtocol.UDP:
                self.connection.sendto(message_str.encode(), (self.config.host, self.config.port))
            elif self.config.protocol == CommunicationProtocol.WEBSOCKET:
                self.connection.send(message_str)
            elif self.config.protocol == CommunicationProtocol.MQTT:
                self.connection.publish(self.config.mqtt_topic, message_str)
            
            self.stats["messages_sent"] += 1
            self.logger.debug(f"发送指令: {message}")
            return True
            
        except Exception as e:
            self.logger.error(f"发送指令失败: {e}")
            return False
    
    def receive_message(self) -> Optional[Dict[str, Any]]:
        """接收消息
        
        Returns:
            Optional[Dict[str, Any]]: 接收到的消息
        """
        if not self.receive_queue.empty():
            return self.receive_queue.get()
        return None
    
    def start_heartbeat(self):
        """启动心跳"""
        def heartbeat_loop():
            while self.is_connected:
                try:
                    heartbeat_command = ControlCommand(
                        command=MovementCommand.STOP,
                        speed=0.0,
                        duration=0.0,
                        metadata={"type": "heartbeat"}
                    )
                    self.send_command(heartbeat_command)
                    self.stats["last_heartbeat"] = time.time()
                    time.sleep(self.config.heartbeat_interval)
                except Exception as e:
                    self.logger.error(f"心跳发送失败: {e}")
                    break
        
        self.heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
    
    def disconnect(self):
        """断开连接"""
        self.is_connected = False
        
        if self.connection:
            try:
                if self.config.protocol == CommunicationProtocol.SERIAL:
                    self.connection.close()
                elif self.config.protocol in [CommunicationProtocol.TCP, CommunicationProtocol.UDP]:
                    self.connection.close()
                elif self.config.protocol == CommunicationProtocol.WEBSOCKET:
                    self.connection.close()
                elif self.config.protocol == CommunicationProtocol.MQTT:
                    self.connection.loop_stop()
                    self.connection.disconnect()
            except Exception as e:
                self.logger.error(f"断开连接时出错: {e}")
        
        self.logger.info("连接已断开")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()

class MotionPlanner:
    """运动规划器"""
    
    def __init__(self, config: FollowingConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # PID控制器
        self.distance_pid = PIDController(
            kp=config.pid_kp,
            ki=config.pid_ki,
            kd=config.pid_kd,
            output_limits=(-config.max_speed, config.max_speed)
        )
        
        self.angle_pid = PIDController(
            kp=config.pid_kp * 0.5,
            ki=config.pid_ki * 0.5,
            kd=config.pid_kd * 0.5,
            output_limits=(-config.turn_speed, config.turn_speed)
        )
        
        # 运动历史
        self.motion_history = []
        self.last_command_time = 0.0
    
    def plan_motion(self, target: Optional[TargetInfo], obstacles: List[Any] = None) -> ControlCommand:
        """规划运动
        
        Args:
            target: 目标信息
            obstacles: 障碍物信息
            
        Returns:
            ControlCommand: 控制指令
        """
        current_time = time.time()
        
        # 如果没有目标或目标丢失
        if not target or target.is_lost:
            return self._handle_lost_target()
        
        # 检查目标置信度
        if target.confidence < self.config.confidence_threshold:
            return self._handle_low_confidence_target(target)
        
        # 安全检查
        if self._check_safety_constraints(target, obstacles):
            return ControlCommand(
                command=MovementCommand.STOP,
                speed=0.0,
                duration=0.1,
                metadata={"reason": "safety_constraint"}
            )
        
        # 计算运动指令
        return self._calculate_motion_command(target)
    
    def _handle_lost_target(self) -> ControlCommand:
        """处理目标丢失"""
        self.logger.debug("目标丢失，执行搜索行为")
        
        # 简单的搜索策略：原地旋转
        return ControlCommand(
            command=MovementCommand.ROTATE_LEFT,
            speed=self.config.turn_speed * 0.5,
            duration=0.5,
            metadata={"reason": "target_lost_search"}
        )
    
    def _handle_low_confidence_target(self, target: TargetInfo) -> ControlCommand:
        """处理低置信度目标"""
        self.logger.debug(f"目标置信度低: {target.confidence}")
        
        # 降低速度，谨慎跟随
        return ControlCommand(
            command=MovementCommand.STOP,
            speed=0.0,
            duration=0.2,
            metadata={"reason": "low_confidence"}
        )
    
    def _check_safety_constraints(self, target: TargetInfo, obstacles: List[Any]) -> bool:
        """检查安全约束
        
        Args:
            target: 目标信息
            obstacles: 障碍物列表
            
        Returns:
            bool: 是否违反安全约束
        """
        # 检查最小距离
        if target.distance < self.config.min_distance:
            self.logger.warning(f"目标距离过近: {target.distance}m")
            return True
        
        # 检查最大距离
        if target.distance > self.config.max_distance:
            self.logger.warning(f"目标距离过远: {target.distance}m")
            return True
        
        # 检查障碍物
        if obstacles:
            for obstacle in obstacles:
                obstacle_distance = getattr(obstacle, 'distance', float('inf'))
                if obstacle_distance < self.config.obstacle_distance:
                    self.logger.warning(f"检测到障碍物: {obstacle_distance}m")
                    return True
        
        return False
    
    def _calculate_motion_command(self, target: TargetInfo) -> ControlCommand:
        """计算运动指令
        
        Args:
            target: 目标信息
            
        Returns:
            ControlCommand: 控制指令
        """
        # 距离控制
        distance_error = target.distance - self.config.target_distance
        distance_output = self.distance_pid.update(0, distance_error)
        
        # 角度控制
        angle_output = self.angle_pid.update(0, target.angle)
        
        # 确定主要运动方向
        if abs(distance_error) > self.config.distance_tolerance:
            # 需要前进或后退
            if distance_error > 0:
                # 目标太远，前进
                command = MovementCommand.FORWARD
                speed = min(abs(distance_output), self.config.max_speed)
            else:
                # 目标太近，后退
                command = MovementCommand.BACKWARD
                speed = min(abs(distance_output), self.config.max_speed)
        elif abs(target.angle) > self.config.angle_tolerance:
            # 需要转向
            if target.angle > 0:
                command = MovementCommand.ROTATE_RIGHT
            else:
                command = MovementCommand.ROTATE_LEFT
            speed = min(abs(angle_output), self.config.turn_speed)
        else:
            # 位置合适，停止
            command = MovementCommand.STOP
            speed = 0.0
        
        # 应用平滑因子
        if self.motion_history:
            last_speed = self.motion_history[-1].get('speed', 0.0)
            speed = last_speed * self.config.smooth_factor + speed * (1 - self.config.smooth_factor)
        
        # 限制速度范围
        if speed > 0:
            speed = max(self.config.min_speed, min(speed, self.config.max_speed))
        
        # 创建控制指令
        control_command = ControlCommand(
            command=command,
            speed=speed,
            duration=1.0 / self.config.control_frequency,
            angle=target.angle,
            metadata={
                "target_distance": target.distance,
                "target_angle": target.angle,
                "distance_error": distance_error,
                "pid_distance_output": distance_output,
                "pid_angle_output": angle_output
            }
        )
        
        # 记录运动历史
        self.motion_history.append({
            'command': command.value,
            'speed': speed,
            'timestamp': time.time()
        })
        
        # 限制历史记录长度
        if len(self.motion_history) > 10:
            self.motion_history.pop(0)
        
        return control_command
    
    def reset(self):
        """重置运动规划器"""
        self.distance_pid.reset()
        self.angle_pid.reset()
        self.motion_history.clear()
        self.logger.info("运动规划器已重置")

class RealtimeTrackingController:
    """实时跟踪控制器主类"""
    
    def __init__(self, 
                 following_config: FollowingConfig,
                 communication_config: CommunicationConfig,
                 tracker: Optional[MultimodalFusionTracker] = None):
        self.following_config = following_config
        self.communication_config = communication_config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 核心组件
        self.tracker = tracker
        self.communication_manager = CommunicationManager(communication_config)
        self.motion_planner = MotionPlanner(following_config)
        
        # 控制状态
        self.control_mode = ControlMode.STANDBY
        self.is_running = False
        self.current_target = None
        
        # 线程管理
        self.control_thread = None
        self.control_lock = threading.Lock()
        
        # 性能统计
        self.stats = {
            "control_cycles": 0,
            "avg_control_frequency": 0.0,
            "target_lost_count": 0,
            "emergency_stops": 0,
            "total_distance_traveled": 0.0
        }
        
        # 回调函数
        self.callbacks = {
            "on_target_acquired": [],
            "on_target_lost": [],
            "on_emergency_stop": [],
            "on_mode_changed": []
        }
        
        self.logger.info("实时跟踪控制器初始化完成")
    
    def start(self) -> bool:
        """启动控制器
        
        Returns:
            bool: 是否启动成功
        """
        try:
            # 建立通信连接
            if not self.communication_manager.connect():
                self.logger.error("通信连接失败")
                return False
            
            # 启动心跳
            self.communication_manager.start_heartbeat()
            
            # 启动控制线程
            self.is_running = True
            self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
            self.control_thread.start()
            
            # 设置为跟随模式
            self.set_control_mode(ControlMode.FOLLOW)
            
            self.logger.info("实时跟踪控制器启动成功")
            return True
            
        except Exception as e:
            self.logger.error(f"控制器启动失败: {e}")
            return False
    
    def stop(self):
        """停止控制器"""
        self.logger.info("正在停止实时跟踪控制器")
        
        # 发送停止指令
        stop_command = ControlCommand(
            command=MovementCommand.STOP,
            speed=0.0,
            duration=0.0
        )
        self.communication_manager.send_command(stop_command)
        
        # 停止控制循环
        self.is_running = False
        
        # 等待线程结束
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=2.0)
        
        # 断开通信
        self.communication_manager.disconnect()
        
        # 重置状态
        self.control_mode = ControlMode.STANDBY
        self.current_target = None
        
        self.logger.info("实时跟踪控制器已停止")
    
    def set_control_mode(self, mode: ControlMode):
        """设置控制模式
        
        Args:
            mode: 控制模式
        """
        with self.control_lock:
            old_mode = self.control_mode
            self.control_mode = mode
            
            self.logger.info(f"控制模式切换: {old_mode.value} -> {mode.value}")
            
            # 触发回调
            self._trigger_callbacks("on_mode_changed", old_mode, mode)
            
            # 模式特定处理
            if mode == ControlMode.EMERGENCY_STOP:
                self._handle_emergency_stop()
            elif mode == ControlMode.STANDBY:
                self._handle_standby_mode()
    
    def update_tracking(self, detections: List[Any], image: Optional[np.ndarray] = None):
        """更新跟踪信息
        
        Args:
            detections: 检测结果
            image: 当前帧图像
        """
        if not self.tracker:
            return
        
        try:
            # 更新跟踪器
            tracking_results = self.tracker.update(
                face_detections=None,
                body_detections=detections,
                pose_detections=None,
                image=image
            )
            
            # 获取主要目标
            primary_target = self.tracker.get_primary_target()
            
            if primary_target:
                # 转换为目标信息
                target_info = self._convert_to_target_info(primary_target, image)
                
                with self.control_lock:
                    old_target = self.current_target
                    self.current_target = target_info
                    
                    # 如果是新目标
                    if not old_target or old_target.track_id != target_info.track_id:
                        self._trigger_callbacks("on_target_acquired", target_info)
            else:
                # 目标丢失
                with self.control_lock:
                    if self.current_target and not self.current_target.is_lost:
                        self.current_target.is_lost = True
                        self.current_target.missing_frames += 1
                        self.stats["target_lost_count"] += 1
                        self._trigger_callbacks("on_target_lost", self.current_target)
                        
        except Exception as e:
            self.logger.error(f"跟踪更新失败: {e}")
    
    def _convert_to_target_info(self, tracking_target: TrackingTarget, image: Optional[np.ndarray]) -> TargetInfo:
        """转换跟踪目标为目标信息
        
        Args:
            tracking_target: 跟踪目标
            image: 图像
            
        Returns:
            TargetInfo: 目标信息
        """
        # 获取位置信息
        if tracking_target.position_history:
            position = tracking_target.position_history[-1]
        else:
            position = (0, 0)
        
        # 获取边界框（从主要特征）
        bbox = (0, 0, 100, 100)
        confidence = 0.5
        
        for feature in tracking_target.features.values():
            bbox = feature.bbox
            confidence = feature.confidence
            break
        
        # 估计距离和角度
        distance, angle = self._estimate_distance_and_angle(bbox, image)
        
        # 计算平均置信度
        if tracking_target.confidence_history:
            avg_confidence = np.mean(list(tracking_target.confidence_history))
        else:
            avg_confidence = confidence
        
        return TargetInfo(
            track_id=tracking_target.track_id,
            position=position,
            bbox=bbox,
            confidence=avg_confidence,
            distance=distance,
            angle=angle,
            velocity=tracking_target.velocity,
            timestamp=time.time(),
            is_lost=False,
            missing_frames=tracking_target.missing_frames
        )
    
    def _estimate_distance_and_angle(self, bbox: Tuple[int, int, int, int], image: Optional[np.ndarray]) -> Tuple[float, float]:
        """估计距离和角度
        
        Args:
            bbox: 边界框
            image: 图像
            
        Returns:
            Tuple[float, float]: (距离, 角度)
        """
        x1, y1, x2, y2 = bbox
        
        # 计算边界框中心
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # 计算边界框大小
        width = x2 - x1
        height = y2 - y1
        
        # 简单的距离估计（基于边界框大小）
        # 假设人体高度约1.7米，根据边界框高度估计距离
        if image is not None:
            image_height = image.shape[0]
            # 距离 = (实际高度 * 焦距) / 像素高度
            # 这里使用简化的估计公式
            distance = max(0.5, 3.0 * (image_height / max(height, 1)) / 100.0)
        else:
            distance = 2.0  # 默认距离
        
        # 角度估计（基于水平偏移）
        if image is not None:
            image_width = image.shape[1]
            image_center_x = image_width / 2
            
            # 计算角度（度）
            # 假设相机水平视角为60度
            horizontal_fov = 60.0
            pixel_offset = center_x - image_center_x
            angle = (pixel_offset / image_center_x) * (horizontal_fov / 2)
        else:
            angle = 0.0  # 默认角度
        
        return distance, angle
    
    def _control_loop(self):
        """控制循环"""
        self.logger.info("控制循环启动")
        
        control_period = 1.0 / self.following_config.control_frequency
        last_control_time = time.time()
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # 控制频率限制
                if current_time - last_control_time < control_period:
                    time.sleep(0.001)  # 短暂休眠
                    continue
                
                # 执行控制逻辑
                with self.control_lock:
                    if self.control_mode == ControlMode.FOLLOW:
                        self._execute_follow_control()
                    elif self.control_mode == ControlMode.PATROL:
                        self._execute_patrol_control()
                    elif self.control_mode == ControlMode.MANUAL:
                        self._execute_manual_control()
                    # STANDBY和EMERGENCY_STOP模式不需要主动控制
                
                # 更新统计信息
                self.stats["control_cycles"] += 1
                actual_frequency = 1.0 / (current_time - last_control_time)
                self.stats["avg_control_frequency"] = (
                    self.stats["avg_control_frequency"] * 0.9 + actual_frequency * 0.1
                )
                
                last_control_time = current_time
                
            except Exception as e:
                self.logger.error(f"控制循环异常: {e}")
                time.sleep(0.1)
        
        self.logger.info("控制循环结束")
    
    def _execute_follow_control(self):
        """执行跟随控制"""
        # 规划运动
        control_command = self.motion_planner.plan_motion(self.current_target)
        
        # 发送控制指令
        if control_command:
            success = self.communication_manager.send_command(control_command)
            if not success:
                self.logger.warning("控制指令发送失败")
    
    def _execute_patrol_control(self):
        """执行巡逻控制"""
        # 简单的巡逻逻辑：缓慢旋转
        patrol_command = ControlCommand(
            command=MovementCommand.ROTATE_LEFT,
            speed=0.2,
            duration=0.5,
            metadata={"mode": "patrol"}
        )
        
        self.communication_manager.send_command(patrol_command)
    
    def _execute_manual_control(self):
        """执行手动控制"""
        # 手动控制模式下，等待外部指令
        # 这里可以实现遥控器或其他输入设备的处理
        pass
    
    def _handle_emergency_stop(self):
        """处理紧急停止"""
        self.logger.warning("执行紧急停止")
        
        stop_command = ControlCommand(
            command=MovementCommand.STOP,
            speed=0.0,
            duration=0.0,
            priority=10,  # 最高优先级
            metadata={"emergency": True}
        )
        
        self.communication_manager.send_command(stop_command)
        self.stats["emergency_stops"] += 1
        
        # 触发回调
        self._trigger_callbacks("on_emergency_stop")
    
    def _handle_standby_mode(self):
        """处理待机模式"""
        self.logger.info("进入待机模式")
        
        # 发送停止指令
        stop_command = ControlCommand(
            command=MovementCommand.STOP,
            speed=0.0,
            duration=0.0
        )
        
        self.communication_manager.send_command(stop_command)
        
        # 重置运动规划器
        self.motion_planner.reset()
    
    def emergency_stop(self):
        """紧急停止"""
        self.set_control_mode(ControlMode.EMERGENCY_STOP)
    
    def manual_control(self, command: MovementCommand, speed: float = 0.5, duration: float = 1.0):
        """手动控制
        
        Args:
            command: 运动指令
            speed: 速度
            duration: 持续时间
        """
        if self.control_mode != ControlMode.MANUAL:
            self.logger.warning("非手动模式，无法执行手动控制")
            return
        
        manual_command = ControlCommand(
            command=command,
            speed=speed,
            duration=duration,
            metadata={"manual": True}
        )
        
        self.communication_manager.send_command(manual_command)
    
    def add_callback(self, event: str, callback: Callable):
        """添加回调函数
        
        Args:
            event: 事件名称
            callback: 回调函数
        """
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def _trigger_callbacks(self, event: str, *args):
        """触发回调函数
        
        Args:
            event: 事件名称
            *args: 回调参数
        """
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(*args)
                except Exception as e:
                    self.logger.error(f"回调函数执行失败: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取控制器状态
        
        Returns:
            Dict[str, Any]: 状态信息
        """
        with self.control_lock:
            status = {
                "control_mode": self.control_mode.value,
                "is_running": self.is_running,
                "is_connected": self.communication_manager.is_connected,
                "current_target": {
                    "track_id": self.current_target.track_id if self.current_target else None,
                    "distance": self.current_target.distance if self.current_target else None,
                    "angle": self.current_target.angle if self.current_target else None,
                    "confidence": self.current_target.confidence if self.current_target else None,
                    "is_lost": self.current_target.is_lost if self.current_target else True
                },
                "stats": self.stats.copy(),
                "communication_stats": self.communication_manager.get_stats()
            }
        
        return status
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置信息
        
        Returns:
            Dict[str, Any]: 配置信息
        """
        return {
            "following_config": {
                "target_distance": self.following_config.target_distance,
                "max_speed": self.following_config.max_speed,
                "control_frequency": self.following_config.control_frequency,
                "lost_timeout": self.following_config.lost_timeout
            },
            "communication_config": {
                "protocol": self.communication_config.protocol.value,
                "host": self.communication_config.host,
                "port": self.communication_config.port
            }
        }

# 测试代码
if __name__ == "__main__":
    # 创建配置
    following_config = FollowingConfig(
        target_distance=1.5,
        max_speed=0.6,
        control_frequency=10.0
    )
    
    communication_config = CommunicationConfig(
        protocol=CommunicationProtocol.SERIAL,
        serial_port="COM3",
        baud_rate=115200
    )
    
    # 创建控制器
    controller = RealtimeTrackingController(
        following_config=following_config,
        communication_config=communication_config
    )
    
    # 添加回调函数
    def on_target_acquired(target):
        print(f"目标获取: {target.track_id}")
    
    def on_target_lost(target):
        print(f"目标丢失: {target.track_id}")
    
    controller.add_callback("on_target_acquired", on_target_acquired)
    controller.add_callback("on_target_lost", on_target_lost)
    
    # 启动控制器
    if controller.start():
        print("控制器启动成功")
        
        try:
            # 模拟运行
            time.sleep(5)
            
            # 获取状态
            status = controller.get_status()
            print(f"控制器状态: {status}")
            
        except KeyboardInterrupt:
            print("用户中断")
        finally:
            controller.stop()
    else:
        print("控制器启动失败")