#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化工具函数
统一的绘制和显示功能
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from ..core.types import DetectionResult, RecognitionResult, TrackingResult, BoundingBox, Keypoint

logger = logging.getLogger(__name__)

# ============================================================================
# 颜色定义
# ============================================================================

# 预定义颜色 (BGR格式)
COLORS = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255),
    'cyan': (255, 255, 0),
    'magenta': (255, 0, 255),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'orange': (0, 165, 255),
    'purple': (128, 0, 128),
    'pink': (203, 192, 255),
    'brown': (42, 42, 165),
    'gray': (128, 128, 128),
    'lime': (0, 255, 0),
    'navy': (128, 0, 0),
}

# 类别颜色映射
CLASS_COLORS = [
    (0, 0, 255),    # 红色
    (0, 255, 0),    # 绿色
    (255, 0, 0),    # 蓝色
    (0, 255, 255),  # 黄色
    (255, 255, 0),  # 青色
    (255, 0, 255),  # 品红
    (0, 165, 255),  # 橙色
    (128, 0, 128),  # 紫色
    (203, 192, 255), # 粉色
    (42, 42, 165),  # 棕色
]

def get_class_color(class_id: int) -> Tuple[int, int, int]:
    """根据类别ID获取颜色"""
    return CLASS_COLORS[class_id % len(CLASS_COLORS)]

def generate_random_color() -> Tuple[int, int, int]:
    """生成随机颜色"""
    return tuple(np.random.randint(0, 256, 3).tolist())

# ============================================================================
# 基础绘制函数
# ============================================================================

def draw_rectangle(
    image: np.ndarray,
    bbox: BoundingBox,
    color: Tuple[int, int, int] = COLORS['green'],
    thickness: int = 2
) -> np.ndarray:
    """绘制矩形框"""
    try:
        cv2.rectangle(
            image,
            (bbox.x, bbox.y),
            (bbox.x2, bbox.y2),
            color,
            thickness
        )
        return image
    except Exception as e:
        logger.error(f"绘制矩形失败: {e}")
        return image

def draw_text(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int] = COLORS['white'],
    font_scale: float = 0.6,
    thickness: int = 2,
    background_color: Optional[Tuple[int, int, int]] = None
) -> np.ndarray:
    """绘制文本"""
    try:
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 计算文本大小
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # 绘制背景
        if background_color:
            cv2.rectangle(
                image,
                (position[0], position[1] - text_height - baseline),
                (position[0] + text_width, position[1] + baseline),
                background_color,
                -1
            )
        
        # 绘制文本
        cv2.putText(
            image,
            text,
            position,
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )
        
        return image
    except Exception as e:
        logger.error(f"绘制文本失败: {e}")
        return image

def draw_circle(
    image: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    color: Tuple[int, int, int] = COLORS['red'],
    thickness: int = -1
) -> np.ndarray:
    """绘制圆形"""
    try:
        cv2.circle(image, center, radius, color, thickness)
        return image
    except Exception as e:
        logger.error(f"绘制圆形失败: {e}")
        return image

def draw_line(
    image: np.ndarray,
    start_point: Tuple[int, int],
    end_point: Tuple[int, int],
    color: Tuple[int, int, int] = COLORS['blue'],
    thickness: int = 2
) -> np.ndarray:
    """绘制直线"""
    try:
        cv2.line(image, start_point, end_point, color, thickness)
        return image
    except Exception as e:
        logger.error(f"绘制直线失败: {e}")
        return image

# ============================================================================
# 检测结果绘制
# ============================================================================

def draw_detection(
    image: np.ndarray,
    detection: DetectionResult,
    color: Optional[Tuple[int, int, int]] = None,
    show_confidence: bool = True,
    show_class_name: bool = True,
    thickness: int = 2
) -> np.ndarray:
    """绘制单个检测结果"""
    try:
        # 确定颜色
        if color is None:
            color = get_class_color(detection.class_id)
        
        # 绘制边界框
        image = draw_rectangle(image, detection.bbox, color, thickness)
        
        # 准备标签文本
        label_parts = []
        if show_class_name:
            label_parts.append(detection.class_name)
        if show_confidence:
            label_parts.append(f"{detection.confidence:.2f}")
        
        if label_parts:
            label = " ".join(label_parts)
            
            # 绘制标签
            label_position = (detection.bbox.x, detection.bbox.y - 10)
            image = draw_text(
                image,
                label,
                label_position,
                COLORS['white'],
                font_scale=0.6,
                thickness=2,
                background_color=color
            )
        
        return image
    except Exception as e:
        logger.error(f"绘制检测结果失败: {e}")
        return image

def draw_detections(
    image: np.ndarray,
    detections: List[DetectionResult],
    colors: Optional[List[Tuple[int, int, int]]] = None,
    show_confidence: bool = True,
    show_class_name: bool = True,
    thickness: int = 2
) -> np.ndarray:
    """绘制多个检测结果"""
    try:
        for i, detection in enumerate(detections):
            color = colors[i] if colors and i < len(colors) else None
            image = draw_detection(
                image, detection, color, show_confidence, show_class_name, thickness
            )
        return image
    except Exception as e:
        logger.error(f"绘制检测结果失败: {e}")
        return image

# ============================================================================
# 关键点绘制
# ============================================================================

def draw_keypoint(
    image: np.ndarray,
    keypoint: Keypoint,
    color: Tuple[int, int, int] = COLORS['red'],
    radius: int = 3,
    show_label: bool = False
) -> np.ndarray:
    """绘制单个关键点"""
    try:
        if not keypoint.visible:
            return image
        
        center = (int(keypoint.point.x), int(keypoint.point.y))
        
        # 绘制关键点
        image = draw_circle(image, center, radius, color, -1)
        
        # 绘制标签
        if show_label and keypoint.label:
            label_position = (center[0] + 5, center[1] - 5)
            image = draw_text(
                image,
                keypoint.label,
                label_position,
                color,
                font_scale=0.4
            )
        
        return image
    except Exception as e:
        logger.error(f"绘制关键点失败: {e}")
        return image

def draw_keypoints(
    image: np.ndarray,
    keypoints: List[Keypoint],
    colors: Optional[List[Tuple[int, int, int]]] = None,
    radius: int = 3,
    show_labels: bool = False,
    connect_points: bool = False,
    connections: Optional[List[Tuple[int, int]]] = None
) -> np.ndarray:
    """绘制多个关键点"""
    try:
        # 绘制关键点
        for i, keypoint in enumerate(keypoints):
            color = colors[i] if colors and i < len(colors) else COLORS['red']
            image = draw_keypoint(image, keypoint, color, radius, show_labels)
        
        # 绘制连接线
        if connect_points and connections:
            for start_idx, end_idx in connections:
                if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                    keypoints[start_idx].visible and keypoints[end_idx].visible):
                    
                    start_point = (int(keypoints[start_idx].point.x), int(keypoints[start_idx].point.y))
                    end_point = (int(keypoints[end_idx].point.x), int(keypoints[end_idx].point.y))
                    
                    image = draw_line(image, start_point, end_point, COLORS['green'], 2)
        
        return image
    except Exception as e:
        logger.error(f"绘制关键点失败: {e}")
        return image

# ============================================================================
# 识别结果绘制
# ============================================================================

def draw_recognition(
    image: np.ndarray,
    recognition: RecognitionResult,
    color: Optional[Tuple[int, int, int]] = None,
    show_attributes: bool = True,
    show_keypoints: bool = True
) -> np.ndarray:
    """绘制识别结果"""
    try:
        # 绘制检测框
        image = draw_detection(image, recognition.detection, color)
        
        # 绘制关键点
        if show_keypoints and recognition.keypoints:
            image = draw_keypoints(image, recognition.keypoints, radius=2)
        
        # 绘制属性信息
        if show_attributes and recognition.attributes:
            y_offset = recognition.detection.bbox.y2 + 20
            for key, value in recognition.attributes.items():
                text = f"{key}: {value}"
                image = draw_text(
                    image,
                    text,
                    (recognition.detection.bbox.x, y_offset),
                    COLORS['yellow'],
                    font_scale=0.5
                )
                y_offset += 20
        
        return image
    except Exception as e:
        logger.error(f"绘制识别结果失败: {e}")
        return image

def draw_recognitions(
    image: np.ndarray,
    recognitions: List[RecognitionResult],
    show_attributes: bool = True,
    show_keypoints: bool = True
) -> np.ndarray:
    """绘制多个识别结果"""
    try:
        for recognition in recognitions:
            image = draw_recognition(image, recognition, None, show_attributes, show_keypoints)
        return image
    except Exception as e:
        logger.error(f"绘制识别结果失败: {e}")
        return image

# ============================================================================
# 跟踪结果绘制
# ============================================================================

def draw_tracking(
    image: np.ndarray,
    tracking: TrackingResult,
    show_trajectory: bool = True,
    show_velocity: bool = True,
    trajectory_length: int = 30
) -> np.ndarray:
    """绘制跟踪结果"""
    try:
        # 绘制识别结果
        color = get_class_color(tracking.track_id)
        image = draw_recognition(image, tracking.recognition, color)
        
        # 绘制轨迹ID
        track_text = f"ID: {tracking.track_id}"
        id_position = (tracking.recognition.detection.bbox.x, tracking.recognition.detection.bbox.y - 30)
        image = draw_text(
            image,
            track_text,
            id_position,
            color,
            font_scale=0.6,
            background_color=COLORS['black']
        )
        
        # 绘制轨迹
        if show_trajectory and len(tracking.trajectory) > 1:
            trajectory_points = tracking.trajectory[-trajectory_length:]
            for i in range(1, len(trajectory_points)):
                start_point = (int(trajectory_points[i-1].x), int(trajectory_points[i-1].y))
                end_point = (int(trajectory_points[i].x), int(trajectory_points[i].y))
                
                # 轨迹颜色渐变
                alpha = i / len(trajectory_points)
                line_color = tuple(int(c * alpha) for c in color)
                
                image = draw_line(image, start_point, end_point, line_color, 2)
        
        # 绘制速度向量
        if show_velocity and tracking.velocity != (0.0, 0.0):
            center = tracking.recognition.detection.bbox.center
            velocity_end = (
                int(center[0] + tracking.velocity[0] * 10),
                int(center[1] + tracking.velocity[1] * 10)
            )
            image = draw_line(image, center, velocity_end, COLORS['cyan'], 3)
        
        return image
    except Exception as e:
        logger.error(f"绘制跟踪结果失败: {e}")
        return image

def draw_trackings(
    image: np.ndarray,
    trackings: List[TrackingResult],
    show_trajectory: bool = True,
    show_velocity: bool = True
) -> np.ndarray:
    """绘制多个跟踪结果"""
    try:
        for tracking in trackings:
            image = draw_tracking(image, tracking, show_trajectory, show_velocity)
        return image
    except Exception as e:
        logger.error(f"绘制跟踪结果失败: {e}")
        return image

# ============================================================================
# 信息显示
# ============================================================================

def draw_fps(
    image: np.ndarray,
    fps: float,
    position: Tuple[int, int] = (10, 30),
    color: Tuple[int, int, int] = COLORS['green']
) -> np.ndarray:
    """绘制FPS信息"""
    fps_text = f"FPS: {fps:.1f}"
    return draw_text(image, fps_text, position, color, font_scale=0.7, thickness=2)

def draw_status(
    image: np.ndarray,
    status: str,
    position: Tuple[int, int] = (10, 60),
    color: Tuple[int, int, int] = COLORS['yellow']
) -> np.ndarray:
    """绘制状态信息"""
    status_text = f"Status: {status}"
    return draw_text(image, status_text, position, color, font_scale=0.6, thickness=2)

def draw_info_panel(
    image: np.ndarray,
    info: Dict[str, Any],
    position: Tuple[int, int] = (10, 10),
    color: Tuple[int, int, int] = COLORS['white'],
    background_color: Tuple[int, int, int] = COLORS['black']
) -> np.ndarray:
    """绘制信息面板"""
    try:
        y_offset = position[1]
        
        for key, value in info.items():
            text = f"{key}: {value}"
            image = draw_text(
                image,
                text,
                (position[0], y_offset),
                color,
                font_scale=0.5,
                thickness=1,
                background_color=background_color
            )
            y_offset += 25
        
        return image
    except Exception as e:
        logger.error(f"绘制信息面板失败: {e}")
        return image

# ============================================================================
# 便捷函数
# ============================================================================

def create_visualization(
    image: np.ndarray,
    detections: Optional[List[DetectionResult]] = None,
    recognitions: Optional[List[RecognitionResult]] = None,
    trackings: Optional[List[TrackingResult]] = None,
    fps: Optional[float] = None,
    status: Optional[str] = None,
    info: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """创建完整的可视化图像"""
    try:
        result_image = image.copy()
        
        # 绘制检测结果
        if detections:
            result_image = draw_detections(result_image, detections)
        
        # 绘制识别结果
        if recognitions:
            result_image = draw_recognitions(result_image, recognitions)
        
        # 绘制跟踪结果
        if trackings:
            result_image = draw_trackings(result_image, trackings)
        
        # 绘制FPS
        if fps is not None:
            result_image = draw_fps(result_image, fps)
        
        # 绘制状态
        if status:
            result_image = draw_status(result_image, status)
        
        # 绘制信息面板
        if info:
            result_image = draw_info_panel(result_image, info)
        
        return result_image
    except Exception as e:
        logger.error(f"创建可视化失败: {e}")
        return image

if __name__ == "__main__":
    # 测试代码
    print("🧪 测试可视化工具...")
    
    # 创建测试图像
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 测试基础绘制
    from ..core.types import create_detection_result, ObjectType
    
    # 创建测试检测结果
    detection = create_detection_result(
        bbox=(100, 100, 200, 150),
        class_id=0,
        class_name="person",
        confidence=0.95,
        object_type=ObjectType.PERSON
    )
    
    # 绘制检测结果
    result_image = draw_detection(test_image, detection)
    
    # 绘制FPS和状态
    result_image = draw_fps(result_image, 30.5)
    result_image = draw_status(result_image, "Running")
    
    # 绘制信息面板
    info = {
        "Objects": 1,
        "Model": "YOLOv8",
        "Device": "CPU"
    }
    result_image = draw_info_panel(result_image, info, (400, 10))
    
    print("✅ 可视化工具测试完成")
    print(f"结果图像尺寸: {result_image.shape}")