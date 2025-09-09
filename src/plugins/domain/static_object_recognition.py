"""静态物体识别插件

支持多种静态物体的检测和识别，包括家具、工具、设备、建筑物等。
提供物体分类、位置检测、状态监测等功能。
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from ...core.base_plugin import DomainPlugin, PluginMetadata, PluginCapability
from ...core.event_bus import EventBus


class ObjectCategory(Enum):
    """物体类别枚举"""
    FURNITURE = "furniture"
    APPLIANCE = "appliance"
    TOOL = "tool"
    ELECTRONIC = "electronic"
    VEHICLE = "vehicle"
    BUILDING = "building"
    CONTAINER = "container"
    DECORATION = "decoration"
    EQUIPMENT = "equipment"
    UNKNOWN = "unknown"


class ObjectType(Enum):
    """具体物体类型枚举"""
    # 家具
    CHAIR = "chair"
    TABLE = "table"
    SOFA = "sofa"
    BED = "bed"
    DESK = "desk"
    CABINET = "cabinet"
    
    # 家电
    TV = "tv"
    REFRIGERATOR = "refrigerator"
    MICROWAVE = "microwave"
    WASHING_MACHINE = "washing_machine"
    AIR_CONDITIONER = "air_conditioner"
    
    # 工具
    HAMMER = "hammer"
    SCREWDRIVER = "screwdriver"
    WRENCH = "wrench"
    DRILL = "drill"
    
    # 电子设备
    COMPUTER = "computer"
    LAPTOP = "laptop"
    PHONE = "phone"
    TABLET = "tablet"
    CAMERA = "camera"
    
    # 交通工具
    CAR = "car"
    BICYCLE = "bicycle"
    MOTORCYCLE = "motorcycle"
    
    # 容器
    BOX = "box"
    BAG = "bag"
    BOTTLE = "bottle"
    CUP = "cup"
    
    UNKNOWN = "unknown"


class ObjectState(Enum):
    """物体状态枚举"""
    NORMAL = "normal"
    DAMAGED = "damaged"
    MISSING = "missing"
    MOVED = "moved"
    OCCUPIED = "occupied"
    EMPTY = "empty"
    ON = "on"
    OFF = "off"
    UNKNOWN = "unknown"


@dataclass
class ObjectDetection:
    """物体检测结果"""
    object_type: ObjectType
    category: ObjectCategory
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    brand: Optional[str] = None
    model: Optional[str] = None
    color: Optional[str] = None
    size_estimate: Optional[Tuple[float, float, float]] = None  # 长宽高估计
    

@dataclass
class ObjectStateResult:
    """物体状态检测结果"""
    state: ObjectState
    confidence: float
    change_detected: bool
    last_seen: float  # 上次检测到的时间戳
    position_stable: bool  # 位置是否稳定
    

@dataclass
class ObjectRecognitionResult:
    """静态物体识别综合结果"""
    detection: ObjectDetection
    state: Optional[ObjectStateResult] = None
    tracking_id: Optional[int] = None
    timestamp: float = 0.0


class BaseObjectDetector(ABC):
    """物体检测器基类"""
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[ObjectDetection]:
        """检测物体"""
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[ObjectType]:
        """获取支持的物体类型"""
        pass


class BaseObjectStateMonitor(ABC):
    """物体状态监测器基类"""
    
    @abstractmethod
    def monitor_state(self, frame: np.ndarray, detection: ObjectDetection,
                     history: List[ObjectDetection]) -> ObjectStateResult:
        """监测物体状态"""
        pass


class TemplateMatchingDetector(BaseObjectDetector):
    """基于模板匹配的物体检测器"""
    
    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold
        self.templates = {}
        self._load_templates()
    
    def _load_templates(self):
        """加载物体模板"""
        # 这里应该加载预定义的物体模板
        # 为了演示，我们创建一些简单的几何形状模板
        
        # 矩形模板（代表桌子、电视等）
        rect_template = np.zeros((60, 100), dtype=np.uint8)
        cv2.rectangle(rect_template, (5, 5), (95, 55), 255, 2)
        self.templates[ObjectType.TABLE] = rect_template
        self.templates[ObjectType.TV] = rect_template
        
        # 正方形模板（代表椅子、盒子等）
        square_template = np.zeros((60, 60), dtype=np.uint8)
        cv2.rectangle(square_template, (5, 5), (55, 55), 255, 2)
        self.templates[ObjectType.CHAIR] = square_template
        self.templates[ObjectType.BOX] = square_template
        
        # 圆形模板（代表瓶子、杯子等）
        circle_template = np.zeros((60, 60), dtype=np.uint8)
        cv2.circle(circle_template, (30, 30), 25, 255, 2)
        self.templates[ObjectType.BOTTLE] = circle_template
        self.templates[ObjectType.CUP] = circle_template
    
    def detect(self, frame: np.ndarray) -> List[ObjectDetection]:
        """检测物体"""
        detections = []
        
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 对每个模板进行匹配
        for object_type, template in self.templates.items():
            # 多尺度模板匹配
            for scale in [0.5, 0.8, 1.0, 1.2, 1.5]:
                # 缩放模板
                h, w = template.shape
                new_h, new_w = int(h * scale), int(w * scale)
                if new_h > edges.shape[0] or new_w > edges.shape[1]:
                    continue
                
                scaled_template = cv2.resize(template, (new_w, new_h))
                
                # 模板匹配
                result = cv2.matchTemplate(edges, scaled_template, cv2.TM_CCOEFF_NORMED)
                
                # 查找匹配位置
                locations = np.where(result >= self.confidence_threshold)
                
                for pt in zip(*locations[::-1]):
                    x, y = pt
                    confidence = result[y, x]
                    
                    # 创建检测结果
                    category = self._get_category(object_type)
                    color = self._estimate_color(frame, x, y, new_w, new_h)
                    
                    detection = ObjectDetection(
                        object_type=object_type,
                        category=category,
                        confidence=confidence,
                        bbox=(x, y, new_w, new_h),
                        color=color,
                        size_estimate=self._estimate_size(new_w, new_h)
                    )
                    detections.append(detection)
        
        # 非最大抑制去除重复检测
        detections = self._non_max_suppression(detections)
        
        return detections
    
    def _get_category(self, object_type: ObjectType) -> ObjectCategory:
        """根据物体类型获取类别"""
        furniture_types = [ObjectType.CHAIR, ObjectType.TABLE, ObjectType.SOFA, 
                          ObjectType.BED, ObjectType.DESK, ObjectType.CABINET]
        appliance_types = [ObjectType.TV, ObjectType.REFRIGERATOR, ObjectType.MICROWAVE,
                          ObjectType.WASHING_MACHINE, ObjectType.AIR_CONDITIONER]
        tool_types = [ObjectType.HAMMER, ObjectType.SCREWDRIVER, ObjectType.WRENCH, ObjectType.DRILL]
        electronic_types = [ObjectType.COMPUTER, ObjectType.LAPTOP, ObjectType.PHONE, 
                           ObjectType.TABLET, ObjectType.CAMERA]
        vehicle_types = [ObjectType.CAR, ObjectType.BICYCLE, ObjectType.MOTORCYCLE]
        container_types = [ObjectType.BOX, ObjectType.BAG, ObjectType.BOTTLE, ObjectType.CUP]
        
        if object_type in furniture_types:
            return ObjectCategory.FURNITURE
        elif object_type in appliance_types:
            return ObjectCategory.APPLIANCE
        elif object_type in tool_types:
            return ObjectCategory.TOOL
        elif object_type in electronic_types:
            return ObjectCategory.ELECTRONIC
        elif object_type in vehicle_types:
            return ObjectCategory.VEHICLE
        elif object_type in container_types:
            return ObjectCategory.CONTAINER
        else:
            return ObjectCategory.UNKNOWN
    
    def _estimate_color(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> str:
        """估计物体颜色"""
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return "unknown"
        
        # 计算平均颜色
        mean_color = np.mean(roi.reshape(-1, 3), axis=0)
        b, g, r = mean_color
        
        # 简单的颜色分类
        if r > g and r > b:
            return "red"
        elif g > r and g > b:
            return "green"
        elif b > r and b > g:
            return "blue"
        elif r > 150 and g > 150 and b > 150:
            return "white"
        elif r < 50 and g < 50 and b < 50:
            return "black"
        else:
            return "mixed"
    
    def _estimate_size(self, width: int, height: int) -> Tuple[float, float, float]:
        """估计物体尺寸（简单的像素到厘米转换）"""
        # 假设1像素 = 0.5厘米（这个需要根据实际情况校准）
        pixel_to_cm = 0.5
        length = width * pixel_to_cm
        width_cm = height * pixel_to_cm
        height_cm = max(length, width_cm) * 0.8  # 估计高度
        
        return (length, width_cm, height_cm)
    
    def _non_max_suppression(self, detections: List[ObjectDetection], 
                           overlap_threshold: float = 0.5) -> List[ObjectDetection]:
        """非最大抑制去除重复检测"""
        if not detections:
            return []
        
        # 按置信度排序
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        filtered_detections = []
        
        for detection in detections:
            # 检查是否与已选择的检测重叠
            overlap = False
            for selected in filtered_detections:
                if self._calculate_iou(detection.bbox, selected.bbox) > overlap_threshold:
                    overlap = True
                    break
            
            if not overlap:
                filtered_detections.append(detection)
        
        return filtered_detections
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                      bbox2: Tuple[int, int, int, int]) -> float:
        """计算两个边界框的IoU"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # 计算交集
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # 计算并集
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_supported_types(self) -> List[ObjectType]:
        """获取支持的物体类型"""
        return list(self.templates.keys())


class ObjectStateMonitor(BaseObjectStateMonitor):
    """物体状态监测器"""
    
    def __init__(self):
        self.position_threshold = 20  # 位置变化阈值（像素）
        self.stability_frames = 5     # 稳定性判断所需帧数
    
    def monitor_state(self, frame: np.ndarray, detection: ObjectDetection,
                     history: List[ObjectDetection]) -> ObjectStateResult:
        """监测物体状态"""
        if not history:
            return ObjectStateResult(
                state=ObjectState.NORMAL,
                confidence=0.5,
                change_detected=False,
                last_seen=0.0,
                position_stable=True
            )
        
        # 检查位置变化
        current_center = self._get_center(detection.bbox)
        position_changes = []
        
        for past_detection in history[-self.stability_frames:]:
            past_center = self._get_center(past_detection.bbox)
            distance = np.linalg.norm(np.array(current_center) - np.array(past_center))
            position_changes.append(distance)
        
        # 判断位置稳定性
        avg_change = np.mean(position_changes) if position_changes else 0
        position_stable = avg_change < self.position_threshold
        
        # 判断物体状态
        state = ObjectState.NORMAL
        change_detected = False
        
        if not position_stable:
            state = ObjectState.MOVED
            change_detected = True
        
        # 检查物体是否被占用（基于颜色变化）
        if self._detect_occlusion(frame, detection, history):
            state = ObjectState.OCCUPIED
            change_detected = True
        
        # 检查物体是否损坏（基于边缘完整性）
        if self._detect_damage(frame, detection):
            state = ObjectState.DAMAGED
            change_detected = True
        
        return ObjectStateResult(
            state=state,
            confidence=0.7,
            change_detected=change_detected,
            last_seen=0.0,  # 应该从时间戳获取
            position_stable=position_stable
        )
    
    def _get_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """获取边界框中心点"""
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)
    
    def _detect_occlusion(self, frame: np.ndarray, detection: ObjectDetection,
                         history: List[ObjectDetection]) -> bool:
        """检测物体是否被遮挡"""
        if not history:
            return False
        
        x, y, w, h = detection.bbox
        current_roi = frame[y:y+h, x:x+w]
        
        # 与历史图像比较（这里简化处理）
        # 实际应该保存历史图像进行比较
        gray = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        
        # 如果边缘比例显著降低，可能被遮挡
        return edge_ratio < 0.05
    
    def _detect_damage(self, frame: np.ndarray, detection: ObjectDetection) -> bool:
        """检测物体是否损坏"""
        x, y, w, h = detection.bbox
        roi = frame[y:y+h, x:x+w]
        
        # 检测不规则边缘或缺失部分
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False
        
        # 分析轮廓的完整性
        largest_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest_contour, True)
        area = cv2.contourArea(largest_contour)
        
        if area == 0:
            return False
        
        # 计算紧凑度（周长²/面积）
        compactness = (perimeter * perimeter) / area
        
        # 如果紧凑度过高，可能表示形状不规则（损坏）
        return compactness > 50


class StaticObjectRecognitionPlugin(DomainPlugin):
    """静态物体识别插件"""
    
    def __init__(self):
        metadata = PluginMetadata(
            name="static_object_recognition",
            version="1.0.0",
            description="静态物体识别和状态监测插件",
            author="YOLOS Team",
            capabilities=[
                PluginCapability.DETECTION,
                PluginCapability.TRACKING,
                PluginCapability.ANALYSIS
            ],
            dependencies=["opencv-python", "numpy"]
        )
        super().__init__(metadata)
        
        self.detector = None
        self.state_monitor = None
        self.tracking_history = {}
        self.next_tracking_id = 1
        
        # 统计信息
        self.stats = {
            'total_detections': 0,
            'category_counts': {category.value: 0 for category in ObjectCategory},
            'object_type_counts': {obj_type.value: 0 for obj_type in ObjectType},
            'state_counts': {state.value: 0 for state in ObjectState},
            'state_changes': 0
        }
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化插件"""
        try:
            # 初始化检测器
            confidence_threshold = config.get('object_confidence_threshold', 0.6)
            self.detector = TemplateMatchingDetector(confidence_threshold)
            
            # 初始化状态监测器
            self.state_monitor = ObjectStateMonitor()
            
            self.logger.info("Static object recognition plugin initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize static object recognition plugin: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray, **kwargs) -> List[ObjectRecognitionResult]:
        """处理视频帧"""
        if not self.detector:
            return []
        
        results = []
        
        try:
            # 检测物体
            detections = self.detector.detect(frame)
            
            for detection in detections:
                # 更新统计
                self.stats['total_detections'] += 1
                self.stats['category_counts'][detection.category.value] += 1
                self.stats['object_type_counts'][detection.object_type.value] += 1
                
                # 跟踪处理
                tracking_id = self._assign_tracking_id(detection)
                
                # 状态监测
                state_result = None
                if self.state_monitor and tracking_id in self.tracking_history:
                    history = self.tracking_history[tracking_id]
                    state_result = self.state_monitor.monitor_state(
                        frame, detection, history
                    )
                    self.stats['state_counts'][state_result.state.value] += 1
                    
                    # 状态变化事件
                    if state_result.change_detected:
                        self.stats['state_changes'] += 1
                        EventBus.emit('object_state_change', {
                            'tracking_id': tracking_id,
                            'object_type': detection.object_type.value,
                            'old_state': 'normal',  # 简化处理
                            'new_state': state_result.state.value
                        })
                
                # 更新跟踪历史
                if tracking_id not in self.tracking_history:
                    self.tracking_history[tracking_id] = []
                self.tracking_history[tracking_id].append(detection)
                
                # 限制历史长度
                if len(self.tracking_history[tracking_id]) > 20:
                    self.tracking_history[tracking_id] = self.tracking_history[tracking_id][-20:]
                
                # 创建结果
                result = ObjectRecognitionResult(
                    detection=detection,
                    state=state_result,
                    tracking_id=tracking_id,
                    timestamp=kwargs.get('timestamp', 0.0)
                )
                results.append(result)
        
        except Exception as e:
            self.logger.error(f"Error processing frame in static object recognition: {e}")
        
        return results
    
    def _assign_tracking_id(self, detection: ObjectDetection) -> int:
        """分配跟踪ID"""
        # 简单的跟踪逻辑：基于位置距离和物体类型
        min_distance = float('inf')
        best_id = None
        
        current_center = self._get_center(detection.bbox)
        
        for tracking_id, history in self.tracking_history.items():
            if history and history[-1].object_type == detection.object_type:
                last_center = self._get_center(history[-1].bbox)
                distance = np.linalg.norm(np.array(current_center) - np.array(last_center))
                
                if distance < min_distance and distance < 50:  # 静态物体移动较少
                    min_distance = distance
                    best_id = tracking_id
        
        if best_id is None:
            best_id = self.next_tracking_id
            self.next_tracking_id += 1
        
        return best_id
    
    def _get_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """获取边界框中心点"""
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)
    
    def get_supported_domains(self) -> List[str]:
        """获取支持的识别领域"""
        return ['objects', 'furniture', 'appliances', 'tools', 'equipment']
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
    
    def cleanup(self):
        """清理资源"""
        self.tracking_history.clear()
        self.stats = {
            'total_detections': 0,
            'category_counts': {category.value: 0 for category in ObjectCategory},
            'object_type_counts': {obj_type.value: 0 for obj_type in ObjectType},
            'state_counts': {state.value: 0 for state in ObjectState},
            'state_changes': 0
        }
        self.logger.info("Static object recognition plugin cleaned up")


# 导出插件类
__all__ = [
    'StaticObjectRecognitionPlugin',
    'ObjectCategory',
    'ObjectType',
    'ObjectState',
    'ObjectDetection',
    'ObjectStateResult',
    'ObjectRecognitionResult'
]