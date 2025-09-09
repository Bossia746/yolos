"""植物识别插件

支持多种植物的检测、识别和健康监测，包括室内植物、农作物、花卉等。
提供物种分类、健康状态评估、生长监测等功能。
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import math

from ...core.base_plugin import DomainPlugin, PluginMetadata, PluginCapability
from ...core.event_bus import EventBus


class PlantType(Enum):
    """植物类型枚举"""
    HOUSEPLANT = "houseplant"
    FLOWER = "flower"
    TREE = "tree"
    SHRUB = "shrub"
    HERB = "herb"
    VEGETABLE = "vegetable"
    FRUIT = "fruit"
    GRASS = "grass"
    SUCCULENT = "succulent"
    FERN = "fern"
    UNKNOWN = "unknown"


class PlantHealthStatus(Enum):
    """植物健康状态枚举"""
    HEALTHY = "healthy"
    STRESSED = "stressed"
    DISEASED = "diseased"
    PEST_INFECTED = "pest_infected"
    NUTRIENT_DEFICIENT = "nutrient_deficient"
    OVERWATERED = "overwatered"
    UNDERWATERED = "underwatered"
    DYING = "dying"
    UNKNOWN = "unknown"


class GrowthStage(Enum):
    """生长阶段枚举"""
    SEEDLING = "seedling"
    JUVENILE = "juvenile"
    MATURE = "mature"
    FLOWERING = "flowering"
    FRUITING = "fruiting"
    DORMANT = "dormant"
    UNKNOWN = "unknown"


@dataclass
class PlantDetection:
    """植物检测结果"""
    plant_type: PlantType
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    species: Optional[str] = None
    variety: Optional[str] = None
    size_estimate: Optional[float] = None  # 估计尺寸（cm）
    

@dataclass
class PlantHealthResult:
    """植物健康评估结果"""
    status: PlantHealthStatus
    confidence: float
    leaf_health_score: float  # 叶片健康评分 0-1
    color_analysis: Dict[str, float]  # 颜色分析结果
    disease_indicators: List[str]  # 疾病指标
    recommendations: List[str]  # 护理建议
    

@dataclass
class PlantGrowthResult:
    """植物生长分析结果"""
    stage: GrowthStage
    confidence: float
    growth_rate: float  # 生长速度评估
    size_change: float  # 尺寸变化百分比
    leaf_count_estimate: int  # 叶片数量估计
    flowering_detected: bool  # 是否检测到开花
    

@dataclass
class PlantRecognitionResult:
    """植物识别综合结果"""
    detection: PlantDetection
    health: Optional[PlantHealthResult] = None
    growth: Optional[PlantGrowthResult] = None
    tracking_id: Optional[int] = None
    timestamp: float = 0.0


class BasePlantDetector(ABC):
    """植物检测器基类"""
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[PlantDetection]:
        """检测植物"""
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[PlantType]:
        """获取支持的植物类型"""
        pass


class BasePlantHealthAnalyzer(ABC):
    """植物健康分析器基类"""
    
    @abstractmethod
    def analyze_health(self, frame: np.ndarray, detection: PlantDetection) -> PlantHealthResult:
        """分析植物健康状态"""
        pass


class BasePlantGrowthMonitor(ABC):
    """植物生长监测器基类"""
    
    @abstractmethod
    def monitor_growth(self, frame: np.ndarray, detection: PlantDetection,
                      history: List[PlantDetection]) -> PlantGrowthResult:
        """监测植物生长"""
        pass


class ColorBasedPlantDetector(BasePlantDetector):
    """基于颜色的植物检测器"""
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
    
    def detect(self, frame: np.ndarray) -> List[PlantDetection]:
        """检测植物"""
        detections = []
        
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 定义绿色范围（植物叶片）
        green_lower1 = np.array([35, 40, 40])
        green_upper1 = np.array([85, 255, 255])
        
        # 创建绿色掩码
        green_mask = cv2.inRange(hsv, green_lower1, green_upper1)
        
        # 形态学操作去噪
        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # 最小面积阈值
                x, y, w, h = cv2.boundingRect(contour)
                
                # 计算置信度（基于面积和形状）
                aspect_ratio = w / h if h > 0 else 0
                confidence = min(0.9, (area / 10000) * (1 - abs(aspect_ratio - 1)))
                
                if confidence >= self.confidence_threshold:
                    # 估计植物类型（基于形状和大小）
                    plant_type = self._classify_plant_type(w, h, aspect_ratio)
                    
                    detection = PlantDetection(
                        plant_type=plant_type,
                        confidence=confidence,
                        bbox=(x, y, w, h),
                        size_estimate=max(w, h) * 0.1  # 简单的尺寸估计
                    )
                    detections.append(detection)
        
        return detections
    
    def _classify_plant_type(self, width: int, height: int, aspect_ratio: float) -> PlantType:
        """基于形状特征分类植物类型"""
        size = max(width, height)
        
        if size < 100:
            return PlantType.HERB
        elif size < 200:
            if aspect_ratio > 1.5:
                return PlantType.GRASS
            else:
                return PlantType.HOUSEPLANT
        elif size < 400:
            return PlantType.SHRUB
        else:
            return PlantType.TREE
    
    def get_supported_types(self) -> List[PlantType]:
        """获取支持的植物类型"""
        return [PlantType.HOUSEPLANT, PlantType.FLOWER, PlantType.TREE, 
                PlantType.SHRUB, PlantType.HERB, PlantType.GRASS]


class PlantHealthAnalyzer(BasePlantHealthAnalyzer):
    """植物健康分析器"""
    
    def analyze_health(self, frame: np.ndarray, detection: PlantDetection) -> PlantHealthResult:
        """分析植物健康状态"""
        x, y, w, h = detection.bbox
        plant_roi = frame[y:y+h, x:x+w]
        
        # 颜色分析
        color_analysis = self._analyze_colors(plant_roi)
        
        # 叶片健康评分
        leaf_health_score = self._calculate_leaf_health_score(color_analysis)
        
        # 疾病指标检测
        disease_indicators = self._detect_disease_indicators(plant_roi, color_analysis)
        
        # 健康状态判断
        status = self._determine_health_status(leaf_health_score, disease_indicators)
        
        # 生成建议
        recommendations = self._generate_recommendations(status, color_analysis, disease_indicators)
        
        return PlantHealthResult(
            status=status,
            confidence=0.7,
            leaf_health_score=leaf_health_score,
            color_analysis=color_analysis,
            disease_indicators=disease_indicators,
            recommendations=recommendations
        )
    
    def _analyze_colors(self, roi: np.ndarray) -> Dict[str, float]:
        """分析植物区域的颜色分布"""
        if roi.size == 0:
            return {'green_ratio': 0.0, 'brown_ratio': 0.0, 'yellow_ratio': 0.0}
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        total_pixels = roi.shape[0] * roi.shape[1]
        
        # 绿色范围
        green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        green_ratio = np.sum(green_mask > 0) / total_pixels
        
        # 黄色范围（可能表示营养不良）
        yellow_mask = cv2.inRange(hsv, np.array([20, 40, 40]), np.array([35, 255, 255]))
        yellow_ratio = np.sum(yellow_mask > 0) / total_pixels
        
        # 棕色范围（可能表示枯萎）
        brown_mask = cv2.inRange(hsv, np.array([10, 40, 20]), np.array([20, 255, 100]))
        brown_ratio = np.sum(brown_mask > 0) / total_pixels
        
        return {
            'green_ratio': green_ratio,
            'yellow_ratio': yellow_ratio,
            'brown_ratio': brown_ratio
        }
    
    def _calculate_leaf_health_score(self, color_analysis: Dict[str, float]) -> float:
        """计算叶片健康评分"""
        green_ratio = color_analysis.get('green_ratio', 0)
        yellow_ratio = color_analysis.get('yellow_ratio', 0)
        brown_ratio = color_analysis.get('brown_ratio', 0)
        
        # 健康评分：绿色比例高，黄色和棕色比例低
        health_score = green_ratio - (yellow_ratio * 0.5) - (brown_ratio * 0.8)
        return max(0.0, min(1.0, health_score))
    
    def _detect_disease_indicators(self, roi: np.ndarray, color_analysis: Dict[str, float]) -> List[str]:
        """检测疾病指标"""
        indicators = []
        
        yellow_ratio = color_analysis.get('yellow_ratio', 0)
        brown_ratio = color_analysis.get('brown_ratio', 0)
        green_ratio = color_analysis.get('green_ratio', 0)
        
        if yellow_ratio > 0.3:
            indicators.append('叶片发黄')
        
        if brown_ratio > 0.2:
            indicators.append('叶片枯萎')
        
        if green_ratio < 0.3:
            indicators.append('绿色素不足')
        
        # 检测斑点（简单的边缘检测）
        if roi.size > 0:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = np.sum(edges > 0) / edges.size
            
            if edge_ratio > 0.1:
                indicators.append('可能有病斑')
        
        return indicators
    
    def _determine_health_status(self, health_score: float, indicators: List[str]) -> PlantHealthStatus:
        """确定健康状态"""
        if health_score > 0.8 and len(indicators) == 0:
            return PlantHealthStatus.HEALTHY
        elif health_score > 0.6 and len(indicators) <= 1:
            return PlantHealthStatus.STRESSED
        elif '叶片发黄' in indicators and '绿色素不足' in indicators:
            return PlantHealthStatus.NUTRIENT_DEFICIENT
        elif '叶片枯萎' in indicators:
            if health_score < 0.2:
                return PlantHealthStatus.DYING
            else:
                return PlantHealthStatus.UNDERWATERED
        elif '可能有病斑' in indicators:
            return PlantHealthStatus.DISEASED
        else:
            return PlantHealthStatus.STRESSED
    
    def _generate_recommendations(self, status: PlantHealthStatus, 
                                color_analysis: Dict[str, float],
                                indicators: List[str]) -> List[str]:
        """生成护理建议"""
        recommendations = []
        
        if status == PlantHealthStatus.HEALTHY:
            recommendations.append('植物状态良好，继续当前护理方式')
        elif status == PlantHealthStatus.NUTRIENT_DEFICIENT:
            recommendations.append('建议施肥，补充氮磷钾营养元素')
        elif status == PlantHealthStatus.UNDERWATERED:
            recommendations.append('增加浇水频率，保持土壤湿润')
        elif status == PlantHealthStatus.OVERWATERED:
            recommendations.append('减少浇水，改善排水条件')
        elif status == PlantHealthStatus.DISEASED:
            recommendations.append('检查病虫害，必要时使用杀菌剂')
        elif status == PlantHealthStatus.DYING:
            recommendations.append('植物状况严重，建议咨询专业园艺师')
        else:
            recommendations.append('密切观察植物状态变化')
        
        # 基于颜色分析的额外建议
        if color_analysis.get('yellow_ratio', 0) > 0.2:
            recommendations.append('叶片发黄可能缺乏氮元素')
        
        return recommendations


class PlantGrowthMonitor(BasePlantGrowthMonitor):
    """植物生长监测器"""
    
    def monitor_growth(self, frame: np.ndarray, detection: PlantDetection,
                      history: List[PlantDetection]) -> PlantGrowthResult:
        """监测植物生长"""
        # 估计生长阶段
        stage = self._estimate_growth_stage(detection, frame)
        
        # 计算生长速度
        growth_rate = self._calculate_growth_rate(detection, history)
        
        # 计算尺寸变化
        size_change = self._calculate_size_change(detection, history)
        
        # 估计叶片数量
        leaf_count = self._estimate_leaf_count(frame, detection)
        
        # 检测开花
        flowering_detected = self._detect_flowering(frame, detection)
        
        return PlantGrowthResult(
            stage=stage,
            confidence=0.6,
            growth_rate=growth_rate,
            size_change=size_change,
            leaf_count_estimate=leaf_count,
            flowering_detected=flowering_detected
        )
    
    def _estimate_growth_stage(self, detection: PlantDetection, frame: np.ndarray) -> GrowthStage:
        """估计生长阶段"""
        size = max(detection.bbox[2], detection.bbox[3])
        
        if size < 50:
            return GrowthStage.SEEDLING
        elif size < 150:
            return GrowthStage.JUVENILE
        elif size < 300:
            return GrowthStage.MATURE
        else:
            # 检查是否有花朵颜色
            if self._detect_flowering(frame, detection):
                return GrowthStage.FLOWERING
            else:
                return GrowthStage.MATURE
    
    def _calculate_growth_rate(self, detection: PlantDetection, 
                             history: List[PlantDetection]) -> float:
        """计算生长速度"""
        if len(history) < 2:
            return 0.0
        
        current_size = max(detection.bbox[2], detection.bbox[3])
        past_size = max(history[0].bbox[2], history[0].bbox[3])
        
        if past_size == 0:
            return 0.0
        
        # 假设历史记录跨度为一周
        growth_rate = (current_size - past_size) / past_size
        return max(0.0, growth_rate)
    
    def _calculate_size_change(self, detection: PlantDetection,
                             history: List[PlantDetection]) -> float:
        """计算尺寸变化百分比"""
        if not history:
            return 0.0
        
        current_area = detection.bbox[2] * detection.bbox[3]
        last_area = history[-1].bbox[2] * history[-1].bbox[3]
        
        if last_area == 0:
            return 0.0
        
        return (current_area - last_area) / last_area * 100
    
    def _estimate_leaf_count(self, frame: np.ndarray, detection: PlantDetection) -> int:
        """估计叶片数量"""
        x, y, w, h = detection.bbox
        roi = frame[y:y+h, x:x+w]
        
        if roi.size == 0:
            return 0
        
        # 简单的叶片计数：基于绿色区域的连通组件
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        
        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找连通组件
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤小的噪声区域
        leaf_contours = [c for c in contours if cv2.contourArea(c) > 50]
        
        return len(leaf_contours)
    
    def _detect_flowering(self, frame: np.ndarray, detection: PlantDetection) -> bool:
        """检测是否开花"""
        x, y, w, h = detection.bbox
        roi = frame[y:y+h, x:x+w]
        
        if roi.size == 0:
            return False
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 检测花朵常见颜色（红、粉、白、黄）
        flower_colors = [
            (np.array([0, 50, 50]), np.array([10, 255, 255])),    # 红色
            (np.array([170, 50, 50]), np.array([180, 255, 255])), # 红色（另一端）
            (np.array([140, 50, 50]), np.array([170, 255, 255])), # 粉色
            (np.array([20, 50, 50]), np.array([30, 255, 255])),   # 黄色
        ]
        
        total_flower_pixels = 0
        total_pixels = roi.shape[0] * roi.shape[1]
        
        for lower, upper in flower_colors:
            mask = cv2.inRange(hsv, lower, upper)
            total_flower_pixels += np.sum(mask > 0)
        
        flower_ratio = total_flower_pixels / total_pixels
        return flower_ratio > 0.05  # 5%以上的花朵颜色像素


class PlantRecognitionPlugin(DomainPlugin):
    """植物识别插件"""
    
    def __init__(self):
        metadata = PluginMetadata(
            name="plant_recognition",
            version="1.0.0",
            description="植物识别和健康监测插件",
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
        self.health_analyzer = None
        self.growth_monitor = None
        self.tracking_history = {}
        self.next_tracking_id = 1
        
        # 统计信息
        self.stats = {
            'total_detections': 0,
            'plant_type_counts': {ptype.value: 0 for ptype in PlantType},
            'health_status_counts': {status.value: 0 for status in PlantHealthStatus},
            'growth_stage_counts': {stage.value: 0 for stage in GrowthStage},
            'health_alerts': 0
        }
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化插件"""
        try:
            # 初始化检测器
            confidence_threshold = config.get('plant_confidence_threshold', 0.5)
            self.detector = ColorBasedPlantDetector(confidence_threshold)
            
            # 初始化健康分析器
            self.health_analyzer = PlantHealthAnalyzer()
            
            # 初始化生长监测器
            self.growth_monitor = PlantGrowthMonitor()
            
            self.logger.info("Plant recognition plugin initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize plant recognition plugin: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray, **kwargs) -> List[PlantRecognitionResult]:
        """处理视频帧"""
        if not self.detector:
            return []
        
        results = []
        
        try:
            # 检测植物
            detections = self.detector.detect(frame)
            
            for detection in detections:
                # 更新统计
                self.stats['total_detections'] += 1
                self.stats['plant_type_counts'][detection.plant_type.value] += 1
                
                # 跟踪处理
                tracking_id = self._assign_tracking_id(detection)
                
                # 健康分析
                health_result = None
                if self.health_analyzer:
                    health_result = self.health_analyzer.analyze_health(frame, detection)
                    self.stats['health_status_counts'][health_result.status.value] += 1
                    
                    # 健康警报
                    if health_result.status in [PlantHealthStatus.DISEASED, 
                                               PlantHealthStatus.DYING,
                                               PlantHealthStatus.PEST_INFECTED]:
                        self.stats['health_alerts'] += 1
                        EventBus.emit('plant_health_alert', {
                            'tracking_id': tracking_id,
                            'plant_type': detection.plant_type.value,
                            'status': health_result.status.value,
                            'recommendations': health_result.recommendations
                        })
                
                # 生长监测
                growth_result = None
                if self.growth_monitor and tracking_id in self.tracking_history:
                    history = self.tracking_history[tracking_id]
                    growth_result = self.growth_monitor.monitor_growth(
                        frame, detection, history
                    )
                    self.stats['growth_stage_counts'][growth_result.stage.value] += 1
                
                # 更新跟踪历史
                if tracking_id not in self.tracking_history:
                    self.tracking_history[tracking_id] = []
                self.tracking_history[tracking_id].append(detection)
                
                # 限制历史长度
                if len(self.tracking_history[tracking_id]) > 30:
                    self.tracking_history[tracking_id] = self.tracking_history[tracking_id][-30:]
                
                # 创建结果
                result = PlantRecognitionResult(
                    detection=detection,
                    health=health_result,
                    growth=growth_result,
                    tracking_id=tracking_id,
                    timestamp=kwargs.get('timestamp', 0.0)
                )
                results.append(result)
        
        except Exception as e:
            self.logger.error(f"Error processing frame in plant recognition: {e}")
        
        return results
    
    def _assign_tracking_id(self, detection: PlantDetection) -> int:
        """分配跟踪ID"""
        # 简单的跟踪逻辑：基于位置距离
        min_distance = float('inf')
        best_id = None
        
        current_center = self._get_center(detection.bbox)
        
        for tracking_id, history in self.tracking_history.items():
            if history and history[-1].plant_type == detection.plant_type:
                last_center = self._get_center(history[-1].bbox)
                distance = np.linalg.norm(np.array(current_center) - np.array(last_center))
                
                if distance < min_distance and distance < 50:  # 植物移动较少
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
        return ['plants', 'vegetation', 'agriculture', 'gardening']
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
    
    def cleanup(self):
        """清理资源"""
        self.tracking_history.clear()
        self.stats = {
            'total_detections': 0,
            'plant_type_counts': {ptype.value: 0 for ptype in PlantType},
            'health_status_counts': {status.value: 0 for status in PlantHealthStatus},
            'growth_stage_counts': {stage.value: 0 for stage in GrowthStage},
            'health_alerts': 0
        }
        self.logger.info("Plant recognition plugin cleaned up")


# 导出插件类
__all__ = [
    'PlantRecognitionPlugin',
    'PlantType',
    'PlantHealthStatus',
    'GrowthStage',
    'PlantDetection',
    'PlantHealthResult',
    'PlantGrowthResult',
    'PlantRecognitionResult'
]