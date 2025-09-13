#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运动预测模块
整合多种预测算法，提升跟踪稳定性和连续性
支持卡尔曼滤波、粒子滤波、LSTM等多种预测方法
"""

import time
import math
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque

# 可选依赖
try:
    import cv2
except ImportError:
    cv2 = None

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None

logger = logging.getLogger(__name__)

class PredictionMethod(Enum):
    """预测方法"""
    KALMAN = "kalman"  # 卡尔曼滤波
    PARTICLE = "particle"  # 粒子滤波
    LSTM = "lstm"  # LSTM神经网络
    POLYNOMIAL = "polynomial"  # 多项式拟合
    PHYSICS = "physics"  # 物理模型
    ENSEMBLE = "ensemble"  # 集成方法

class MotionModel(Enum):
    """运动模型"""
    CONSTANT_VELOCITY = "constant_velocity"  # 匀速运动
    CONSTANT_ACCELERATION = "constant_acceleration"  # 匀加速运动
    RANDOM_WALK = "random_walk"  # 随机游走
    COORDINATED_TURN = "coordinated_turn"  # 协调转弯
    BICYCLE = "bicycle"  # 自行车模型

@dataclass
class MotionState:
    """运动状态"""
    position: Tuple[float, float]  # 位置 (x, y)
    velocity: Tuple[float, float]  # 速度 (vx, vy)
    acceleration: Tuple[float, float] = (0.0, 0.0)  # 加速度 (ax, ay)
    orientation: float = 0.0  # 方向角（弧度）
    angular_velocity: float = 0.0  # 角速度
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0  # 状态置信度
    
    def to_vector(self) -> np.ndarray:
        """转换为状态向量"""
        return np.array([
            self.position[0], self.position[1],
            self.velocity[0], self.velocity[1],
            self.acceleration[0], self.acceleration[1],
            self.orientation, self.angular_velocity
        ], dtype=np.float32)
    
    @classmethod
    def from_vector(cls, vector: np.ndarray, timestamp: float = None, confidence: float = 1.0) -> 'MotionState':
        """从状态向量创建"""
        if timestamp is None:
            timestamp = time.time()
        
        return cls(
            position=(vector[0], vector[1]),
            velocity=(vector[2], vector[3]),
            acceleration=(vector[4], vector[5]) if len(vector) > 5 else (0.0, 0.0),
            orientation=vector[6] if len(vector) > 6 else 0.0,
            angular_velocity=vector[7] if len(vector) > 7 else 0.0,
            timestamp=timestamp,
            confidence=confidence
        )
    
    def distance_to(self, other: 'MotionState') -> float:
        """计算到另一个状态的距离"""
        dx = self.position[0] - other.position[0]
        dy = self.position[1] - other.position[1]
        return math.sqrt(dx * dx + dy * dy)
    
    def speed(self) -> float:
        """计算速度大小"""
        vx, vy = self.velocity
        return math.sqrt(vx * vx + vy * vy)

@dataclass
class PredictionResult:
    """预测结果"""
    predicted_state: MotionState
    uncertainty: np.ndarray  # 不确定性矩阵
    confidence: float  # 预测置信度
    method: PredictionMethod  # 使用的预测方法
    computation_time: float  # 计算时间（毫秒）
    metadata: Dict[str, Any] = field(default_factory=dict)

class ExtendedKalmanFilter:
    """扩展卡尔曼滤波器
    
    支持非线性运动模型
    """
    
    def __init__(self, motion_model: MotionModel = MotionModel.CONSTANT_VELOCITY):
        self.motion_model = motion_model
        
        # 状态维度
        if motion_model == MotionModel.CONSTANT_VELOCITY:
            self.state_dim = 4  # [x, y, vx, vy]
        elif motion_model == MotionModel.CONSTANT_ACCELERATION:
            self.state_dim = 6  # [x, y, vx, vy, ax, ay]
        else:
            self.state_dim = 8  # 完整状态
        
        # 初始化状态
        self.state = np.zeros(self.state_dim, dtype=np.float32)
        self.covariance = np.eye(self.state_dim, dtype=np.float32) * 1000
        
        # 过程噪声
        self.process_noise = self._create_process_noise()
        
        # 观测噪声
        self.observation_noise = np.eye(2, dtype=np.float32) * 10
        
        # 时间步长
        self.dt = 1.0 / 30.0  # 默认30FPS
        self.last_update_time = time.time()
    
    def _create_process_noise(self) -> np.ndarray:
        """创建过程噪声矩阵"""
        Q = np.eye(self.state_dim, dtype=np.float32)
        
        if self.motion_model == MotionModel.CONSTANT_VELOCITY:
            # 位置和速度的噪声
            Q[0, 0] = Q[1, 1] = 0.1  # 位置噪声
            Q[2, 2] = Q[3, 3] = 1.0  # 速度噪声
        elif self.motion_model == MotionModel.CONSTANT_ACCELERATION:
            # 位置、速度和加速度的噪声
            Q[0, 0] = Q[1, 1] = 0.1  # 位置噪声
            Q[2, 2] = Q[3, 3] = 1.0  # 速度噪声
            Q[4, 4] = Q[5, 5] = 5.0  # 加速度噪声
        else:
            # 完整状态噪声
            Q *= 0.1
            Q[2:4, 2:4] *= 10  # 速度噪声更大
            Q[4:6, 4:6] *= 50  # 加速度噪声更大
        
        return Q
    
    def _state_transition_function(self, state: np.ndarray, dt: float) -> np.ndarray:
        """状态转移函数
        
        Args:
            state: 当前状态
            dt: 时间步长
            
        Returns:
            np.ndarray: 预测状态
        """
        if self.motion_model == MotionModel.CONSTANT_VELOCITY:
            # 匀速运动模型
            F = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)
            return F @ state
        
        elif self.motion_model == MotionModel.CONSTANT_ACCELERATION:
            # 匀加速运动模型
            F = np.array([
                [1, 0, dt, 0, 0.5*dt*dt, 0],
                [0, 1, 0, dt, 0, 0.5*dt*dt],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ], dtype=np.float32)
            return F @ state
        
        elif self.motion_model == MotionModel.COORDINATED_TURN:
            # 协调转弯模型
            if len(state) >= 8:
                x, y, vx, vy, ax, ay, theta, omega = state[:8]
                
                # 非线性状态转移
                if abs(omega) > 1e-6:
                    sin_omega_dt = math.sin(omega * dt)
                    cos_omega_dt = math.cos(omega * dt)
                    
                    new_x = x + (vx * sin_omega_dt + vy * (cos_omega_dt - 1)) / omega
                    new_y = y + (vy * sin_omega_dt - vx * (cos_omega_dt - 1)) / omega
                    new_vx = vx * cos_omega_dt - vy * sin_omega_dt
                    new_vy = vx * sin_omega_dt + vy * cos_omega_dt
                else:
                    # 直线运动
                    new_x = x + vx * dt
                    new_y = y + vy * dt
                    new_vx = vx
                    new_vy = vy
                
                new_state = state.copy()
                new_state[0] = new_x
                new_state[1] = new_y
                new_state[2] = new_vx
                new_state[3] = new_vy
                new_state[6] = theta + omega * dt
                
                return new_state
        
        # 默认：简单的线性模型
        return state
    
    def _jacobian_F(self, state: np.ndarray, dt: float) -> np.ndarray:
        """计算状态转移函数的雅可比矩阵
        
        Args:
            state: 当前状态
            dt: 时间步长
            
        Returns:
            np.ndarray: 雅可比矩阵
        """
        if self.motion_model == MotionModel.CONSTANT_VELOCITY:
            return np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)
        
        elif self.motion_model == MotionModel.CONSTANT_ACCELERATION:
            return np.array([
                [1, 0, dt, 0, 0.5*dt*dt, 0],
                [0, 1, 0, dt, 0, 0.5*dt*dt],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ], dtype=np.float32)
        
        elif self.motion_model == MotionModel.COORDINATED_TURN and len(state) >= 8:
            # 协调转弯模型的雅可比矩阵（简化版本）
            omega = state[7] if len(state) > 7 else 0.0
            
            if abs(omega) > 1e-6:
                sin_omega_dt = math.sin(omega * dt)
                cos_omega_dt = math.cos(omega * dt)
                
                F = np.eye(self.state_dim, dtype=np.float32)
                F[0, 2] = sin_omega_dt / omega
                F[0, 3] = (cos_omega_dt - 1) / omega
                F[1, 2] = -(cos_omega_dt - 1) / omega
                F[1, 3] = sin_omega_dt / omega
                F[2, 2] = cos_omega_dt
                F[2, 3] = -sin_omega_dt
                F[3, 2] = sin_omega_dt
                F[3, 3] = cos_omega_dt
                F[6, 7] = dt
                
                return F
        
        # 默认：单位矩阵
        return np.eye(self.state_dim, dtype=np.float32)
    
    def predict(self, dt: Optional[float] = None) -> MotionState:
        """预测下一个状态
        
        Args:
            dt: 时间步长
            
        Returns:
            MotionState: 预测状态
        """
        if dt is None:
            current_time = time.time()
            dt = current_time - self.last_update_time
            self.last_update_time = current_time
        
        # 状态预测
        predicted_state = self._state_transition_function(self.state, dt)
        
        # 协方差预测
        F = self._jacobian_F(self.state, dt)
        predicted_covariance = F @ self.covariance @ F.T + self.process_noise
        
        # 更新内部状态
        self.state = predicted_state
        self.covariance = predicted_covariance
        
        # 转换为MotionState
        return self._state_to_motion_state(predicted_state)
    
    def update(self, observation: Tuple[float, float], observation_covariance: Optional[np.ndarray] = None):
        """更新状态
        
        Args:
            observation: 观测值 (x, y)
            observation_covariance: 观测协方差矩阵
        """
        if observation_covariance is None:
            observation_covariance = self.observation_noise
        
        # 观测矩阵（只观测位置）
        H = np.zeros((2, self.state_dim), dtype=np.float32)
        H[0, 0] = 1  # x
        H[1, 1] = 1  # y
        
        # 观测值
        z = np.array([observation[0], observation[1]], dtype=np.float32)
        
        # 预测观测值
        predicted_observation = H @ self.state
        
        # 创新（残差）
        innovation = z - predicted_observation
        
        # 创新协方差
        S = H @ self.covariance @ H.T + observation_covariance
        
        # 卡尔曼增益
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        # 状态更新
        self.state = self.state + K @ innovation
        
        # 协方差更新
        I = np.eye(self.state_dim, dtype=np.float32)
        self.covariance = (I - K @ H) @ self.covariance
    
    def _state_to_motion_state(self, state: np.ndarray) -> MotionState:
        """将状态向量转换为MotionState
        
        Args:
            state: 状态向量
            
        Returns:
            MotionState: 运动状态
        """
        position = (state[0], state[1])
        velocity = (state[2], state[3]) if len(state) > 3 else (0.0, 0.0)
        acceleration = (state[4], state[5]) if len(state) > 5 else (0.0, 0.0)
        orientation = state[6] if len(state) > 6 else 0.0
        angular_velocity = state[7] if len(state) > 7 else 0.0
        
        return MotionState(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            orientation=orientation,
            angular_velocity=angular_velocity,
            timestamp=time.time()
        )
    
    def initialize(self, initial_state: MotionState):
        """初始化滤波器
        
        Args:
            initial_state: 初始状态
        """
        state_vector = initial_state.to_vector()
        self.state = state_vector[:self.state_dim]
        
        # 重置协方差
        self.covariance = np.eye(self.state_dim, dtype=np.float32) * 100
        
        self.last_update_time = initial_state.timestamp

class ParticleFilter:
    """粒子滤波器
    
    用于处理非线性、非高斯的运动模型
    """
    
    def __init__(self, num_particles: int = 100, motion_model: MotionModel = MotionModel.CONSTANT_VELOCITY):
        self.num_particles = num_particles
        self.motion_model = motion_model
        
        # 粒子集合
        self.particles = None
        self.weights = None
        
        # 噪声参数
        self.process_noise_std = 5.0
        self.observation_noise_std = 10.0
        
        self.last_update_time = time.time()
    
    def initialize(self, initial_state: MotionState, initial_covariance: Optional[np.ndarray] = None):
        """初始化粒子滤波器
        
        Args:
            initial_state: 初始状态
            initial_covariance: 初始协方差
        """
        state_vector = initial_state.to_vector()
        state_dim = len(state_vector)
        
        if initial_covariance is None:
            initial_covariance = np.eye(state_dim) * 100
        
        # 生成初始粒子
        self.particles = np.random.multivariate_normal(
            state_vector, initial_covariance, self.num_particles
        ).astype(np.float32)
        
        # 初始化权重
        self.weights = np.ones(self.num_particles, dtype=np.float32) / self.num_particles
        
        self.last_update_time = initial_state.timestamp
    
    def predict(self, dt: Optional[float] = None) -> MotionState:
        """预测下一个状态
        
        Args:
            dt: 时间步长
            
        Returns:
            MotionState: 预测状态
        """
        if self.particles is None:
            raise ValueError("粒子滤波器未初始化")
        
        if dt is None:
            current_time = time.time()
            dt = current_time - self.last_update_time
            self.last_update_time = current_time
        
        # 对每个粒子进行状态转移
        for i in range(self.num_particles):
            self.particles[i] = self._state_transition(self.particles[i], dt)
            
            # 添加过程噪声
            noise = np.random.normal(0, self.process_noise_std, len(self.particles[i]))
            self.particles[i] += noise
        
        # 计算加权平均状态
        weighted_state = np.average(self.particles, weights=self.weights, axis=0)
        
        return MotionState.from_vector(weighted_state)
    
    def update(self, observation: Tuple[float, float]):
        """更新粒子权重
        
        Args:
            observation: 观测值 (x, y)
        """
        if self.particles is None:
            raise ValueError("粒子滤波器未初始化")
        
        # 计算每个粒子的似然度
        for i in range(self.num_particles):
            # 计算观测似然度
            predicted_obs = self.particles[i][:2]  # 位置
            distance = np.linalg.norm(predicted_obs - np.array(observation))
            
            # 高斯似然度
            likelihood = np.exp(-0.5 * (distance / self.observation_noise_std) ** 2)
            self.weights[i] *= likelihood
        
        # 归一化权重
        self.weights += 1e-300  # 避免除零
        self.weights /= np.sum(self.weights)
        
        # 重采样（如果有效粒子数太少）
        effective_particles = 1.0 / np.sum(self.weights ** 2)
        if effective_particles < self.num_particles / 2:
            self._resample()
    
    def _state_transition(self, state: np.ndarray, dt: float) -> np.ndarray:
        """粒子状态转移
        
        Args:
            state: 当前状态
            dt: 时间步长
            
        Returns:
            np.ndarray: 新状态
        """
        if self.motion_model == MotionModel.CONSTANT_VELOCITY:
            # 匀速运动
            new_state = state.copy()
            if len(state) >= 4:
                new_state[0] += state[2] * dt  # x += vx * dt
                new_state[1] += state[3] * dt  # y += vy * dt
        
        elif self.motion_model == MotionModel.RANDOM_WALK:
            # 随机游走
            new_state = state.copy()
            if len(state) >= 4:
                # 速度随机变化
                new_state[2] += np.random.normal(0, 1) * dt
                new_state[3] += np.random.normal(0, 1) * dt
                # 位置更新
                new_state[0] += new_state[2] * dt
                new_state[1] += new_state[3] * dt
        
        else:
            # 默认：保持状态不变
            new_state = state.copy()
        
        return new_state
    
    def _resample(self):
        """重采样粒子"""
        # 系统重采样
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0  # 确保最后一个值为1
        
        # 生成均匀分布的随机数
        u = np.random.uniform(0, 1/self.num_particles)
        indices = []
        
        for i in range(self.num_particles):
            u_i = u + i / self.num_particles
            j = np.searchsorted(cumulative_sum, u_i)
            indices.append(j)
        
        # 重采样粒子
        self.particles = self.particles[indices]
        
        # 重置权重
        self.weights = np.ones(self.num_particles, dtype=np.float32) / self.num_particles

class PolynomialPredictor:
    """多项式预测器
    
    使用多项式拟合进行轨迹预测
    """
    
    def __init__(self, degree: int = 2, history_length: int = 10):
        self.degree = degree
        self.history_length = history_length
        
        # 历史轨迹
        self.position_history = deque(maxlen=history_length)
        self.time_history = deque(maxlen=history_length)
    
    def add_observation(self, position: Tuple[float, float], timestamp: float):
        """添加观测
        
        Args:
            position: 位置
            timestamp: 时间戳
        """
        self.position_history.append(position)
        self.time_history.append(timestamp)
    
    def predict(self, future_time: float) -> Optional[MotionState]:
        """预测未来状态
        
        Args:
            future_time: 未来时间戳
            
        Returns:
            Optional[MotionState]: 预测状态
        """
        if len(self.position_history) < self.degree + 1:
            return None
        
        try:
            # 转换为numpy数组
            times = np.array(list(self.time_history))
            positions = np.array(list(self.position_history))
            
            # 归一化时间
            t0 = times[0]
            times_norm = times - t0
            future_time_norm = future_time - t0
            
            # 分别拟合x和y坐标
            x_coeffs = np.polyfit(times_norm, positions[:, 0], self.degree)
            y_coeffs = np.polyfit(times_norm, positions[:, 1], self.degree)
            
            # 预测位置
            pred_x = np.polyval(x_coeffs, future_time_norm)
            pred_y = np.polyval(y_coeffs, future_time_norm)
            
            # 计算速度（一阶导数）
            if self.degree >= 1:
                x_vel_coeffs = np.polyder(x_coeffs)
                y_vel_coeffs = np.polyder(y_coeffs)
                pred_vx = np.polyval(x_vel_coeffs, future_time_norm)
                pred_vy = np.polyval(y_vel_coeffs, future_time_norm)
            else:
                pred_vx = pred_vy = 0.0
            
            # 计算加速度（二阶导数）
            if self.degree >= 2:
                x_acc_coeffs = np.polyder(x_vel_coeffs)
                y_acc_coeffs = np.polyder(y_vel_coeffs)
                pred_ax = np.polyval(x_acc_coeffs, future_time_norm)
                pred_ay = np.polyval(y_acc_coeffs, future_time_norm)
            else:
                pred_ax = pred_ay = 0.0
            
            return MotionState(
                position=(pred_x, pred_y),
                velocity=(pred_vx, pred_vy),
                acceleration=(pred_ax, pred_ay),
                timestamp=future_time
            )
            
        except Exception as e:
            logger.warning(f"多项式预测失败: {e}")
            return None

class PhysicsBasedPredictor:
    """基于物理的预测器
    
    考虑物理约束的运动预测
    """
    
    def __init__(self, max_acceleration: float = 5.0, max_velocity: float = 10.0):
        self.max_acceleration = max_acceleration
        self.max_velocity = max_velocity
        
        # 当前状态
        self.current_state = None
    
    def update_state(self, state: MotionState):
        """更新当前状态
        
        Args:
            state: 当前状态
        """
        self.current_state = state
    
    def predict(self, dt: float) -> Optional[MotionState]:
        """预测未来状态
        
        Args:
            dt: 时间步长
            
        Returns:
            Optional[MotionState]: 预测状态
        """
        if self.current_state is None:
            return None
        
        # 当前状态
        x, y = self.current_state.position
        vx, vy = self.current_state.velocity
        ax, ay = self.current_state.acceleration
        
        # 限制加速度
        acc_magnitude = math.sqrt(ax * ax + ay * ay)
        if acc_magnitude > self.max_acceleration:
            scale = self.max_acceleration / acc_magnitude
            ax *= scale
            ay *= scale
        
        # 预测速度
        new_vx = vx + ax * dt
        new_vy = vy + ay * dt
        
        # 限制速度
        vel_magnitude = math.sqrt(new_vx * new_vx + new_vy * new_vy)
        if vel_magnitude > self.max_velocity:
            scale = self.max_velocity / vel_magnitude
            new_vx *= scale
            new_vy *= scale
        
        # 预测位置
        new_x = x + vx * dt + 0.5 * ax * dt * dt
        new_y = y + vy * dt + 0.5 * ay * dt * dt
        
        return MotionState(
            position=(new_x, new_y),
            velocity=(new_vx, new_vy),
            acceleration=(ax, ay),
            timestamp=self.current_state.timestamp + dt
        )

class EnsemblePredictor:
    """集成预测器
    
    结合多种预测方法的结果
    """
    
    def __init__(self):
        # 预测器列表
        self.predictors = {
            PredictionMethod.KALMAN: ExtendedKalmanFilter(),
            PredictionMethod.PARTICLE: ParticleFilter(num_particles=50),
            PredictionMethod.POLYNOMIAL: PolynomialPredictor(),
            PredictionMethod.PHYSICS: PhysicsBasedPredictor()
        }
        
        # 权重
        self.weights = {
            PredictionMethod.KALMAN: 0.4,
            PredictionMethod.PARTICLE: 0.3,
            PredictionMethod.POLYNOMIAL: 0.2,
            PredictionMethod.PHYSICS: 0.1
        }
        
        # 性能历史
        self.performance_history = {method: deque(maxlen=10) for method in self.predictors.keys()}
    
    def initialize(self, initial_state: MotionState):
        """初始化所有预测器
        
        Args:
            initial_state: 初始状态
        """
        # 初始化卡尔曼滤波器
        self.predictors[PredictionMethod.KALMAN].initialize(initial_state)
        
        # 初始化粒子滤波器
        self.predictors[PredictionMethod.PARTICLE].initialize(initial_state)
        
        # 初始化多项式预测器
        self.predictors[PredictionMethod.POLYNOMIAL].add_observation(
            initial_state.position, initial_state.timestamp
        )
        
        # 初始化物理预测器
        self.predictors[PredictionMethod.PHYSICS].update_state(initial_state)
    
    def update(self, observation: Tuple[float, float], timestamp: float):
        """更新所有预测器
        
        Args:
            observation: 观测值
            timestamp: 时间戳
        """
        # 更新卡尔曼滤波器
        self.predictors[PredictionMethod.KALMAN].update(observation)
        
        # 更新粒子滤波器
        self.predictors[PredictionMethod.PARTICLE].update(observation)
        
        # 更新多项式预测器
        self.predictors[PredictionMethod.POLYNOMIAL].add_observation(observation, timestamp)
        
        # 更新物理预测器
        current_state = MotionState(
            position=observation,
            velocity=(0, 0),  # 需要从历史计算
            timestamp=timestamp
        )
        self.predictors[PredictionMethod.PHYSICS].update_state(current_state)
    
    def predict(self, dt: float) -> PredictionResult:
        """集成预测
        
        Args:
            dt: 时间步长
            
        Returns:
            PredictionResult: 预测结果
        """
        start_time = time.time()
        
        # 收集所有预测结果
        predictions = {}
        
        # 卡尔曼滤波预测
        try:
            predictions[PredictionMethod.KALMAN] = self.predictors[PredictionMethod.KALMAN].predict(dt)
        except Exception as e:
            logger.warning(f"卡尔曼滤波预测失败: {e}")
        
        # 粒子滤波预测
        try:
            predictions[PredictionMethod.PARTICLE] = self.predictors[PredictionMethod.PARTICLE].predict(dt)
        except Exception as e:
            logger.warning(f"粒子滤波预测失败: {e}")
        
        # 多项式预测
        try:
            future_time = time.time() + dt
            poly_pred = self.predictors[PredictionMethod.POLYNOMIAL].predict(future_time)
            if poly_pred:
                predictions[PredictionMethod.POLYNOMIAL] = poly_pred
        except Exception as e:
            logger.warning(f"多项式预测失败: {e}")
        
        # 物理预测
        try:
            predictions[PredictionMethod.PHYSICS] = self.predictors[PredictionMethod.PHYSICS].predict(dt)
        except Exception as e:
            logger.warning(f"物理预测失败: {e}")
        
        # 加权融合
        if predictions:
            fused_state = self._fuse_predictions(predictions)
            
            # 计算不确定性
            uncertainty = self._compute_uncertainty(predictions, fused_state)
            
            # 计算置信度
            confidence = self._compute_confidence(predictions)
            
            computation_time = (time.time() - start_time) * 1000
            
            return PredictionResult(
                predicted_state=fused_state,
                uncertainty=uncertainty,
                confidence=confidence,
                method=PredictionMethod.ENSEMBLE,
                computation_time=computation_time,
                metadata={
                    "num_predictors": len(predictions),
                    "active_methods": list(predictions.keys())
                }
            )
        else:
            # 如果所有预测都失败，返回默认结果
            return PredictionResult(
                predicted_state=MotionState(position=(0, 0), velocity=(0, 0)),
                uncertainty=np.eye(2) * 1000,
                confidence=0.0,
                method=PredictionMethod.ENSEMBLE,
                computation_time=(time.time() - start_time) * 1000
            )
    
    def _fuse_predictions(self, predictions: Dict[PredictionMethod, MotionState]) -> MotionState:
        """融合多个预测结果
        
        Args:
            predictions: 预测结果字典
            
        Returns:
            MotionState: 融合后的状态
        """
        # 计算加权平均
        total_weight = 0.0
        weighted_position = np.array([0.0, 0.0])
        weighted_velocity = np.array([0.0, 0.0])
        weighted_acceleration = np.array([0.0, 0.0])
        
        for method, state in predictions.items():
            weight = self.weights.get(method, 0.1)
            total_weight += weight
            
            weighted_position += np.array(state.position) * weight
            weighted_velocity += np.array(state.velocity) * weight
            weighted_acceleration += np.array(state.acceleration) * weight
        
        if total_weight > 0:
            weighted_position /= total_weight
            weighted_velocity /= total_weight
            weighted_acceleration /= total_weight
        
        return MotionState(
            position=(weighted_position[0], weighted_position[1]),
            velocity=(weighted_velocity[0], weighted_velocity[1]),
            acceleration=(weighted_acceleration[0], weighted_acceleration[1]),
            timestamp=time.time()
        )
    
    def _compute_uncertainty(self, predictions: Dict[PredictionMethod, MotionState], fused_state: MotionState) -> np.ndarray:
        """计算预测不确定性
        
        Args:
            predictions: 预测结果
            fused_state: 融合状态
            
        Returns:
            np.ndarray: 不确定性矩阵
        """
        if len(predictions) <= 1:
            return np.eye(2) * 100
        
        # 计算位置方差
        positions = np.array([list(state.position) for state in predictions.values()])
        fused_position = np.array(fused_state.position)
        
        # 计算协方差矩阵
        deviations = positions - fused_position
        covariance = np.cov(deviations.T)
        
        # 确保是正定矩阵
        if covariance.ndim == 0:
            covariance = np.array([[covariance, 0], [0, covariance]])
        elif covariance.ndim == 1:
            covariance = np.diag(covariance)
        
        return covariance
    
    def _compute_confidence(self, predictions: Dict[PredictionMethod, MotionState]) -> float:
        """计算预测置信度
        
        Args:
            predictions: 预测结果
            
        Returns:
            float: 置信度
        """
        if len(predictions) <= 1:
            return 0.5
        
        # 基于预测一致性计算置信度
        positions = np.array([list(state.position) for state in predictions.values()])
        
        # 计算位置标准差
        std_x = np.std(positions[:, 0])
        std_y = np.std(positions[:, 1])
        avg_std = (std_x + std_y) / 2
        
        # 将标准差转换为置信度
        confidence = max(0.0, min(1.0, 1.0 - avg_std / 100.0))
        
        return confidence
    
    def adapt_weights(self, ground_truth: MotionState, predictions: Dict[PredictionMethod, MotionState]):
        """自适应调整权重
        
        Args:
            ground_truth: 真实状态
            predictions: 预测结果
        """
        # 计算每个预测器的误差
        errors = {}
        for method, prediction in predictions.items():
            error = prediction.distance_to(ground_truth)
            errors[method] = error
            
            # 记录性能历史
            self.performance_history[method].append(error)
        
        # 基于历史性能调整权重
        for method in self.weights.keys():
            if method in self.performance_history and len(self.performance_history[method]) > 0:
                avg_error = np.mean(list(self.performance_history[method]))
                # 误差越小，权重越大
                self.weights[method] = 1.0 / (1.0 + avg_error)
        
        # 归一化权重
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            for method in self.weights.keys():
                self.weights[method] /= total_weight

class MotionPredictor:
    """运动预测器主类
    
    统一的运动预测接口
    """
    
    def __init__(self, 
                 method: PredictionMethod = PredictionMethod.ENSEMBLE,
                 motion_model: MotionModel = MotionModel.CONSTANT_VELOCITY,
                 **kwargs):
        self.method = method
        self.motion_model = motion_model
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 创建预测器
        if method == PredictionMethod.KALMAN:
            self.predictor = ExtendedKalmanFilter(motion_model)
        elif method == PredictionMethod.PARTICLE:
            num_particles = kwargs.get('num_particles', 100)
            self.predictor = ParticleFilter(num_particles, motion_model)
        elif method == PredictionMethod.POLYNOMIAL:
            degree = kwargs.get('degree', 2)
            history_length = kwargs.get('history_length', 10)
            self.predictor = PolynomialPredictor(degree, history_length)
        elif method == PredictionMethod.PHYSICS:
            max_acceleration = kwargs.get('max_acceleration', 5.0)
            max_velocity = kwargs.get('max_velocity', 10.0)
            self.predictor = PhysicsBasedPredictor(max_acceleration, max_velocity)
        elif method == PredictionMethod.ENSEMBLE:
            self.predictor = EnsemblePredictor()
        else:
            raise ValueError(f"不支持的预测方法: {method}")
        
        # 状态
        self.is_initialized = False
        self.last_update_time = 0.0
        
        # 统计信息
        self.stats = {
            "predictions_made": 0,
            "avg_computation_time": 0.0,
            "avg_confidence": 0.0,
            "last_prediction_error": 0.0
        }
        
        self.logger.info(f"运动预测器初始化完成，方法: {method.value}")
    
    def initialize(self, initial_state: MotionState):
        """初始化预测器
        
        Args:
            initial_state: 初始状态
        """
        try:
            if hasattr(self.predictor, 'initialize'):
                self.predictor.initialize(initial_state)
            
            self.is_initialized = True
            self.last_update_time = initial_state.timestamp
            
            self.logger.info(f"预测器初始化完成，初始位置: {initial_state.position}")
            
        except Exception as e:
            self.logger.error(f"预测器初始化失败: {e}")
            raise
    
    def update(self, observation: Tuple[float, float], timestamp: Optional[float] = None):
        """更新预测器
        
        Args:
            observation: 观测值 (x, y)
            timestamp: 时间戳
        """
        if not self.is_initialized:
            # 自动初始化
            if timestamp is None:
                timestamp = time.time()
            
            initial_state = MotionState(
                position=observation,
                velocity=(0, 0),
                timestamp=timestamp
            )
            self.initialize(initial_state)
            return
        
        try:
            if timestamp is None:
                timestamp = time.time()
            
            # 更新预测器
            if hasattr(self.predictor, 'update'):
                if self.method == PredictionMethod.ENSEMBLE:
                    self.predictor.update(observation, timestamp)
                else:
                    self.predictor.update(observation)
            
            self.last_update_time = timestamp
            
        except Exception as e:
            self.logger.error(f"预测器更新失败: {e}")
    
    def predict(self, dt: float) -> PredictionResult:
        """进行预测
        
        Args:
            dt: 预测时间步长
            
        Returns:
            PredictionResult: 预测结果
        """
        if not self.is_initialized:
            return PredictionResult(
                predicted_state=MotionState(position=(0, 0), velocity=(0, 0)),
                uncertainty=np.eye(2) * 1000,
                confidence=0.0,
                method=self.method,
                computation_time=0.0
            )
        
        start_time = time.time()
        
        try:
            if self.method == PredictionMethod.ENSEMBLE:
                result = self.predictor.predict(dt)
            else:
                predicted_state = self.predictor.predict(dt)
                
                # 创建预测结果
                result = PredictionResult(
                    predicted_state=predicted_state,
                    uncertainty=np.eye(2) * 50,  # 默认不确定性
                    confidence=0.8,  # 默认置信度
                    method=self.method,
                    computation_time=(time.time() - start_time) * 1000
                )
            
            # 更新统计信息
            self.stats["predictions_made"] += 1
            self.stats["avg_computation_time"] = (
                self.stats["avg_computation_time"] * 0.9 + result.computation_time * 0.1
            )
            self.stats["avg_confidence"] = (
                self.stats["avg_confidence"] * 0.9 + result.confidence * 0.1
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"预测失败: {e}")
            
            # 返回默认结果
            return PredictionResult(
                predicted_state=MotionState(position=(0, 0), velocity=(0, 0)),
                uncertainty=np.eye(2) * 1000,
                confidence=0.0,
                method=self.method,
                computation_time=(time.time() - start_time) * 1000
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return self.stats.copy()
    
    def reset(self):
        """重置预测器"""
        self.is_initialized = False
        self.last_update_time = 0.0
        
        # 重置统计信息
        self.stats = {
            "predictions_made": 0,
            "avg_computation_time": 0.0,
            "avg_confidence": 0.0,
            "last_prediction_error": 0.0
        }
        
        self.logger.info("预测器已重置")

# 测试代码
if __name__ == "__main__":
    # 创建运动预测器
    predictor = MotionPredictor(
        method=PredictionMethod.ENSEMBLE,
        motion_model=MotionModel.CONSTANT_VELOCITY
    )
    
    # 模拟轨迹
    trajectory = [
        (100, 100), (105, 102), (110, 104), (115, 106), (120, 108)
    ]
    
    # 初始化
    initial_state = MotionState(
        position=trajectory[0],
        velocity=(5, 2),
        timestamp=time.time()
    )
    predictor.initialize(initial_state)
    
    # 更新和预测
    for i, pos in enumerate(trajectory[1:], 1):
        # 更新
        predictor.update(pos, time.time() + i * 0.1)
        
        # 预测
        result = predictor.predict(0.1)
        
        print(f"步骤 {i}:")
        print(f"  观测位置: {pos}")
        print(f"  预测位置: {result.predicted_state.position}")
        print(f"  预测置信度: {result.confidence:.3f}")
        print(f"  计算时间: {result.computation_time:.2f}ms")
        print()
    
    # 获取统计信息
    stats = predictor.get_stats()
    print(f"统计信息: {stats}")