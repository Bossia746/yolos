#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人体姿态识别和运动计数系统
基于YOLO的人体关键点检测，实现各种运动动作的自动计数
支持俯卧撑、深蹲、压腿、高抬腿等多种运动监测
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

try:
    from ultralytics import YOLO, solutions
except ImportError:
    YOLO = None
    solutions = None
    logging.warning("Ultralytics not available, pose recognition will be limited")

logger = logging.getLogger(__name__)

class ExerciseType(Enum):
    """运动类型枚举"""
    PUSHUP = "pushup"  # 俯卧撑
    SQUAT = "squat"  # 深蹲
    LEG_PRESS = "leg_press"  # 压腿
    HIGH_KNEE = "high_knee"  # 高抬腿
    PLANK = "plank"  # 平板支撑
    JUMPING_JACK = "jumping_jack"  # 开合跳
    CUSTOM = "custom"  # 自定义

class PoseState(Enum):
    """姿态状态"""
    UP = "up"  # 伸展状态
    DOWN = "down"  # 收缩状态
    TRANSITION = "transition"  # 过渡状态
    UNKNOWN = "unknown"  # 未知状态

@dataclass
class KeypointConfig:
    """关键点配置"""
    keypoints: List[int]  # 关键点索引
    up_angle: float = 145.0  # 伸展角度阈值
    down_angle: float = 100.0  # 收缩角度阈值
    angle_tolerance: float = 10.0  # 角度容差
    min_confidence: float = 0.5  # 最小置信度
    
    # YOLO人体17个关键点定义
    KEYPOINT_NAMES = {
        0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
        5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
        9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
        13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"
    }

# 导入统一的ProcessingResult和异常处理
from ..core.types import ProcessingResult, TaskType, Status
from ..core.exceptions import YOLOSException, ErrorCode, exception_handler

@dataclass
class PoseAnalysisResult:
    """姿态分析结果数据类"""
    success: bool = False
    angle: float = 0.0
    confidence: float = 0.0
    state: PoseState = PoseState.UNKNOWN
    keypoints: Optional[np.ndarray] = None
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_processing_result(self) -> ProcessingResult:
        """转换为标准ProcessingResult"""
        return ProcessingResult(
            task_type=TaskType.RECOGNITION,
            status=Status.SUCCESS if self.success else Status.FAILED,
            processing_time=0.0,
            error_message=self.error_message or "",
            metadata={
                'angle': self.angle,
                'confidence': self.confidence,
                'state': self.state.value,
                'keypoints_count': len(self.keypoints) if self.keypoints is not None else 0,
                'timestamp': self.timestamp
            }
        )


@dataclass
class ExerciseStats:
    """运动统计信息"""
    exercise_type: ExerciseType
    count: int = 0
    total_time: float = 0.0
    avg_duration_per_rep: float = 0.0
    current_state: PoseState = PoseState.UNKNOWN
    last_state_change: float = 0.0
    angles_history: List[float] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)
    
    def update_stats(self, angle: float, confidence: float, timestamp: float):
        """更新统计信息"""
        self.angles_history.append(angle)
        self.confidence_scores.append(confidence)
        self.total_time = timestamp
        
        # 保持历史记录在合理范围内
        if len(self.angles_history) > 1000:
            self.angles_history = self.angles_history[-500:]
            self.confidence_scores = self.confidence_scores[-500:]
        
        # 计算平均每次动作时长
        if self.count > 0:
            self.avg_duration_per_rep = self.total_time / self.count

from .base_recognizer import BaseRecognizer, RecognizerType, RecognizerConfig

@dataclass
class PoseRecognizerConfig(RecognizerConfig):
    """姿态识别器配置"""
    exercise_type: ExerciseType = ExerciseType.PUSHUP
    keypoint_config: Optional[KeypointConfig] = None
    
    def __post_init__(self):
        """后初始化处理"""
        super().__post_init__() if hasattr(super(), '__post_init__') else None
        if self.keypoint_config is None:
            self.keypoint_config = PoseRecognizer.EXERCISE_CONFIGS.get(
                self.exercise_type, 
                PoseRecognizer.EXERCISE_CONFIGS[ExerciseType.PUSHUP]
            )

class PoseRecognizer(BaseRecognizer):
    """人体姿态识别器
    
    基于YOLO的人体关键点检测，支持多种运动类型的识别和计数。
    可以使用Ultralytics的AIGym解决方案或自定义实现。
    
    支持的运动类型:
    - 俯卧撑 (PUSHUP)
    - 深蹲 (SQUAT) 
    - 压腿 (LEG_PRESS)
    - 高抬腿 (HIGH_KNEE)
    
    Attributes:
        config: 姿态识别器配置
        exercise_type: 运动类型
        keypoint_config: 关键点配置
        model: YOLO模型实例
        gym_solution: AIGym解决方案实例
        stats: 运动统计信息
    """
    
    # 预定义运动配置
    EXERCISE_CONFIGS = {
        ExerciseType.PUSHUP: KeypointConfig(
            keypoints=[5, 7, 9],  # 左肩、左肘、左腕
            up_angle=145.0,
            down_angle=100.0
        ),
        ExerciseType.SQUAT: KeypointConfig(
            keypoints=[6, 12, 14],  # 右肩、右髋、右膝
            up_angle=160.0,
            down_angle=90.0
        ),
        ExerciseType.LEG_PRESS: KeypointConfig(
            keypoints=[11, 13, 15],  # 左髋、左膝、左踝
            up_angle=150.0,
            down_angle=100.0
        ),
        ExerciseType.HIGH_KNEE: KeypointConfig(
            keypoints=[12, 14, 16],  # 右髋、右膝、右踝
            up_angle=90.0,  # 高抬腿时角度较小
            down_angle=160.0  # 放下时角度较大
        )
    }
    
    def __init__(self, 
                 model_path: str = "yolo11n-pose.pt",
                 exercise_type: ExerciseType = ExerciseType.PUSHUP,
                 custom_config: Optional[KeypointConfig] = None,
                 enable_visualization: bool = True,
                 confidence_threshold: float = 0.5,
                 device: str = "auto"):
        """
        初始化姿态识别器
        
        Args:
            model_path: YOLO模型路径
            exercise_type: 运动类型
            custom_config: 自定义关键点配置
            enable_visualization: 是否启用可视化
            confidence_threshold: 置信度阈值
            device: 设备类型
        """
        # 创建配置对象
        pose_config = PoseRecognizerConfig(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=device,
            enable_visualization=enable_visualization,
            exercise_type=exercise_type,
            keypoint_config=custom_config
        )
        
        # 调用父类初始化
        super().__init__(pose_config, RecognizerType.POSE)
        
        # 姿态识别器特有属性
        self.exercise_type = exercise_type
        self.keypoint_config = pose_config.keypoint_config
        self.gym_solution = None
        
        # 初始化统计信息
        self.stats = ExerciseStats(exercise_type=exercise_type)
        
        # 状态跟踪
        self.last_state = PoseState.UNKNOWN
        self.last_angle = 0.0
        self.frame_count = 0
        
        logger.info(f"姿态识别器初始化完成: {exercise_type.value}, 关键点: {self.keypoint_config.keypoints}")
    
    def initialize(self) -> bool:
        """初始化识别器
        
        Returns:
            bool: 初始化是否成功
        """
        if self.initialized:
            return True
        
        try:
            self._load_model()
            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"姿态识别器初始化失败: {e}")
            return False
    
    def recognize(self, image: np.ndarray, **kwargs) -> ProcessingResult:
        """执行姿态识别
        
        Args:
            image: 输入图像，形状为 (H, W, C)，BGR格式
            **kwargs: 额外参数
            
        Returns:
            ProcessingResult: 标准化识别结果
        """
        import time
        start_time = time.time()
        
        try:
            # 验证输入
            self.validate_input(image)
            
            # 确保已初始化
            if not self.initialized:
                self.initialize()
            
            # 执行姿态分析
            analysis_result = self.analyze_pose(image)
            processing_time = time.time() - start_time
            
            # 更新统计信息
            success = analysis_result.get('analysis_result', {}).get('success', False)
            self._update_stats(success, processing_time)
            
            # 转换为标准ProcessingResult
            if 'analysis_result' in analysis_result and analysis_result['analysis_result']:
                pose_result = analysis_result['analysis_result']
                if hasattr(pose_result, 'to_processing_result'):
                    result = pose_result.to_processing_result()
                    result.processing_time = processing_time
                    return result
            
            # 创建默认结果
            return ProcessingResult(
                task_type=TaskType.RECOGNITION,
                status=Status.SUCCESS if success else Status.FAILED,
                processing_time=processing_time,
                metadata={
                    'count': analysis_result.get('count', 0),
                    'current_state': analysis_result.get('current_state', PoseState.UNKNOWN).value,
                    'keypoints_detected': analysis_result.get('keypoints') is not None
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(False, processing_time)
            
            return ProcessingResult(
                task_type=TaskType.RECOGNITION,
                status=Status.FAILED,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def cleanup(self) -> None:
        """清理资源"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.gym_solution is not None:
                del self.gym_solution
                self.gym_solution = None
            
            self.initialized = False
            logger.info("姿态识别器资源清理完成")
            
        except Exception as e:
            logger.warning(f"资源清理时出现警告: {e}")
    
    @exception_handler(ErrorCode.MODEL_LOAD_ERROR)
    def _load_model(self):
        """加载YOLO模型"""
        if YOLO is None:
            raise YOLOSException(
                ErrorCode.DEPENDENCY_ERROR,
                "Ultralytics YOLO not available. Please install ultralytics package."
            )
        
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            logger.info(f"Model file not found at {model_path}, downloading...")
            # YOLO会自动下载模型
        
        self.model = YOLO(self.config.model_path)
        logger.info(f"Model loaded successfully: {self.config.model_path}")
        
        # 如果支持solutions，使用AIGym
        if solutions and hasattr(solutions, 'AIGym'):
            try:
                self.gym_solution = solutions.AIGym(
                    show=self.config.enable_visualization,
                    kpts=self.keypoint_config.keypoints,
                    model=self.config.model_path,
                    up_angle=self.keypoint_config.up_angle,
                    down_angle=self.keypoint_config.down_angle
                )
                logger.info("使用Ultralytics AIGym解决方案")
            except Exception as e:
                logger.warning(f"AIGym初始化失败: {e}")
                self.gym_solution = None
        else:
            logger.info("使用自定义姿态识别实现")
    
    def calculate_angle(self, p1: Tuple[float, float], 
                      p2: Tuple[float, float], 
                      p3: Tuple[float, float]) -> float:
        """计算三点间的角度"""
        try:
            # 向量计算
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            # 计算角度
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 防止数值误差
            angle = np.arccos(cos_angle) * 180 / np.pi
            
            return angle
        except Exception as e:
            logger.warning(f"角度计算失败: {e}")
            return 0.0
    
    def _calculate_angle(self, p1: Tuple[float, float], 
                       p2: Tuple[float, float], 
                       p3: Tuple[float, float]) -> float:
        """私有角度计算方法（向后兼容）"""
        return self.calculate_angle(p1, p2, p3)
    
    @exception_handler(ErrorCode.DATA_PROCESSING_ERROR)
    def extract_keypoints(self, results) -> Optional[List[Tuple[float, float, float]]]:
        """从YOLO结果中提取关键点
        
        Args:
            results: YOLO模型推理结果
            
        Returns:
            Optional[List[Tuple[float, float, float]]]: 关键点列表，格式为[(x, y, confidence), ...]
            
        Raises:
            YOLOSException: 当数据处理失败时
        """
        if not results or len(results) == 0:
            return None
        
        # 获取第一个检测到的人体
        result = results[0]
        if not hasattr(result, 'keypoints') or result.keypoints is None:
            return None
        
        keypoints = result.keypoints.data[0]  # 第一个人的关键点
        
        # 提取所需的关键点 (x, y, confidence)
        extracted_points = []
        for kpt_idx in self.keypoint_config.keypoints:
            if kpt_idx < len(keypoints):
                x, y, conf = keypoints[kpt_idx]
                extracted_points.append((float(x), float(y), float(conf)))
            else:
                extracted_points.append((0.0, 0.0, 0.0))
        
        return extracted_points
    
    def analyze_pose(self, frame: np.ndarray) -> Dict[str, Any]:
        """分析姿态
        
        Args:
            frame: 输入图像帧
            
        Returns:
            Dict[str, Any]: 包含分析结果的字典，包括PoseAnalysisResult
        """
        self.frame_count += 1
        current_time = time.time()
        start_time = time.time()
        
        result = {
            'angle': 0.0,
            'state': PoseState.UNKNOWN,
            'count': self.stats.count,
            'confidence': 0.0,
            'keypoints': None,
            'frame_with_annotations': frame.copy(),
            'processing_time': 0.0,
            'analysis_result': None
        }
        
        try:
            # 如果有AIGym解决方案，优先使用
            if self.gym_solution:
                processed_frame = self.gym_solution(frame)
                # AIGym会自动处理计数，我们需要从中提取信息
                # 这里简化处理，实际使用时可能需要更复杂的状态同步
                result['frame_with_annotations'] = processed_frame
                result['processing_time'] = time.time() - start_time
                result['analysis_result'] = PoseAnalysisResult(
                    success=True,
                    error_message="Using AIGym solution"
                )
                return result
            
            # 使用自定义实现
            if self.model is None:
                result['processing_time'] = time.time() - start_time
                result['analysis_result'] = PoseAnalysisResult(
                    success=False,
                    error_message="Model not loaded"
                )
                return result
            
            # YOLO推理
            results = self.model(frame, verbose=False)
            
            # 提取关键点
            keypoints = self.extract_keypoints(results)
            if not keypoints or len(keypoints) < 3:
                result['processing_time'] = time.time() - start_time
                result['analysis_result'] = PoseAnalysisResult(
                    success=False,
                    error_message="Insufficient keypoints detected"
                )
                return result
            
            # 检查置信度
            min_confidence = min(kpt[2] for kpt in keypoints)
            if min_confidence < self.keypoint_config.min_confidence:
                result['processing_time'] = time.time() - start_time
                result['analysis_result'] = PoseAnalysisResult(
                    success=False,
                    confidence=min_confidence,
                    error_message=f"Low confidence: {min_confidence:.2f}"
                )
                return result
            
            # 计算角度
            p1 = (keypoints[0][0], keypoints[0][1])
            p2 = (keypoints[1][0], keypoints[1][1])  # 中心点
            p3 = (keypoints[2][0], keypoints[2][1])
            
            angle = self.calculate_angle(p1, p2, p3)
            
            # 判断状态
            new_state = self._determine_state(angle)
            
            # 更新计数
            if self._should_increment_count(new_state):
                self.stats.count += 1
                logger.info(f"动作计数: {self.stats.count}")
            
            # 更新统计
            self.stats.update_stats(angle, min_confidence, current_time)
            self.stats.current_state = new_state
            self.last_angle = angle
            
            # 创建分析结果
            analysis_result = PoseAnalysisResult(
                success=True,
                angle=angle,
                confidence=min_confidence,
                state=new_state,
                keypoints=np.array(keypoints),
                timestamp=current_time
            )
            
            # 绘制可视化
            if self.enable_visualization:
                annotated_frame = self._draw_annotations(frame, keypoints, angle, new_state)
                result['frame_with_annotations'] = annotated_frame
            
            result.update({
                'angle': angle,
                'state': new_state,
                'count': self.stats.count,
                'confidence': min_confidence,
                'keypoints': keypoints,
                'processing_time': time.time() - start_time,
                'analysis_result': analysis_result
            })
            
        except Exception as e:
            logger.error(f"姿态分析失败: {e}")
            result['processing_time'] = time.time() - start_time
            result['analysis_result'] = PoseAnalysisResult(
                success=False,
                error_message=str(e)
            )
        
        return result
    
    def _determine_state(self, angle: float) -> PoseState:
        """根据角度判断状态"""
        if angle >= self.keypoint_config.up_angle - self.keypoint_config.angle_tolerance:
            return PoseState.UP
        elif angle <= self.keypoint_config.down_angle + self.keypoint_config.angle_tolerance:
            return PoseState.DOWN
        else:
            return PoseState.TRANSITION
    
    def _should_increment_count(self, new_state: PoseState) -> bool:
        """判断是否应该增加计数"""
        # 简单的状态机：从DOWN到UP时计数
        if (self.stats.current_state == PoseState.DOWN and 
            new_state == PoseState.UP):
            return True
        return False
    
    def _draw_annotations(self, frame: np.ndarray, 
                         keypoints: List[Tuple[float, float, float]], 
                         angle: float, state: PoseState) -> np.ndarray:
        """绘制标注"""
        annotated_frame = frame.copy()
        
        try:
            # 绘制关键点
            for i, (x, y, conf) in enumerate(keypoints):
                if conf > self.keypoint_config.min_confidence:
                    cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                    cv2.putText(annotated_frame, f"{i}", (int(x), int(y-10)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 绘制连线
            if len(keypoints) >= 3:
                points = [(int(kpt[0]), int(kpt[1])) for kpt in keypoints[:3]]
                cv2.line(annotated_frame, points[0], points[1], (255, 0, 0), 2)
                cv2.line(annotated_frame, points[1], points[2], (255, 0, 0), 2)
            
            # 显示信息
            info_text = [
                f"Exercise: {self.exercise_type.value}",
                f"Count: {self.stats.count}",
                f"Angle: {angle:.1f}°",
                f"State: {state.value}",
                f"Frame: {self.frame_count}"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(annotated_frame, text, (10, 30 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 状态指示器
            color = (0, 255, 0) if state == PoseState.UP else (0, 0, 255) if state == PoseState.DOWN else (0, 255, 255)
            cv2.rectangle(annotated_frame, (frame.shape[1] - 50, 10), 
                         (frame.shape[1] - 10, 50), color, -1)
            
        except Exception as e:
            logger.warning(f"标注绘制失败: {e}")
        
        return annotated_frame
    
    def process_video(self, video_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """处理视频文件"""
        logger.info(f"开始处理视频: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 输出视频
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 分析姿态
                result = self.analyze_pose(frame)
                
                # 保存帧
                if writer and self.enable_visualization:
                    writer.write(result['frame_with_annotations'])
                
                # 显示进度
                if frame_idx % 30 == 0:
                    progress = (frame_idx / total_frames) * 100
                    logger.info(f"处理进度: {progress:.1f}% ({frame_idx}/{total_frames})")
                
                frame_idx += 1
            
        finally:
            cap.release()
            if writer:
                writer.release()
        
        # 返回统计结果
        return {
            'total_count': self.stats.count,
            'total_frames': frame_idx,
            'avg_angle': np.mean(self.stats.angles_history) if self.stats.angles_history else 0,
            'avg_confidence': np.mean(self.stats.confidence_scores) if self.stats.confidence_scores else 0,
            'exercise_type': self.exercise_type.value,
            'processing_time': self.stats.total_time
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = ExerciseStats(exercise_type=self.exercise_type)
        self.frame_count = 0
        self.state_start_time = time.time()
        logger.info("统计信息已重置")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'count': self.stats.count,
            'total_time': self.stats.total_time,
            'avg_duration_per_rep': self.stats.avg_duration_per_rep,
            'current_state': self.stats.current_state.value,
            'frame_count': self.frame_count,
            'avg_angle': np.mean(self.stats.angles_history) if self.stats.angles_history else 0,
            'avg_confidence': np.mean(self.stats.confidence_scores) if self.stats.confidence_scores else 0
        }
    
    def set_exercise_type(self, exercise_type: ExerciseType, 
                         custom_config: Optional[KeypointConfig] = None):
        """设置运动类型"""
        self.exercise_type = exercise_type
        
        if custom_config:
            self.keypoint_config = custom_config
        else:
            self.keypoint_config = self.EXERCISE_CONFIGS.get(
                exercise_type, 
                self.EXERCISE_CONFIGS[ExerciseType.PUSHUP]
            )
        
        # 重新初始化AIGym（如果使用）
        if self.gym_solution and solutions:
            try:
                self.gym_solution = solutions.AIGym(
                    show=self.enable_visualization,
                    kpts=self.config.keypoints,
                    model=self.model_path,
                    up_angle=self.config.up_angle,
                    down_angle=self.config.down_angle
                )
            except Exception as e:
                logger.warning(f"AIGym重新初始化失败: {e}")
        
        self.reset_stats()
        logger.info(f"运动类型已设置为: {exercise_type.value}")

# 使用示例
if __name__ == "__main__":
    # 创建姿态识别器
    recognizer = PoseRecognizer(
        model_path="yolo11n-pose.pt",
        exercise_type=ExerciseType.PUSHUP,
        enable_visualization=True
    )
    
    # 处理视频
    try:
        video_path = "test_videos/pushup_demo.mp4"
        output_path = "output/pushup_analysis.mp4"
        
        results = recognizer.process_video(video_path, output_path)
        
        print(f"\n分析完成!")
        print(f"总计数: {results['total_count']}")
        print(f"平均角度: {results['avg_angle']:.1f}°")
        print(f"平均置信度: {results['avg_confidence']:.2f}")
        
    except Exception as e:
        print(f"处理失败: {e}")
        
        # 使用摄像头进行实时检测
        print("\n切换到摄像头实时检测...")
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            result = recognizer.analyze_pose(frame)
            
            cv2.imshow('Pose Recognition', result['frame_with_annotations'])
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()