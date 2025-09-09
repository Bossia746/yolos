#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
反欺骗检测模块
检测海报、照片、视频等虚假目标，防止误判
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
import logging
from dataclasses import dataclass
from enum import Enum

class SpoofingType(Enum):
    """欺骗类型"""
    REAL = "real"
    PHOTO = "photo"
    VIDEO = "video"
    POSTER = "poster"
    SCREEN = "screen"
    MASK = "mask"
    UNKNOWN = "unknown"

@dataclass
class SpoofingDetectionResult:
    """反欺骗检测结果"""
    is_real: bool
    spoofing_type: SpoofingType
    confidence: float
    evidence: Dict[str, float]
    risk_level: str  # low, medium, high

class AntiSpoofingDetector:
    """反欺骗检测器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # 初始化检测器
        self._init_detectors()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            # 纹理分析参数
            'texture_window_size': 15,
            'texture_threshold': 0.3,
            
            # 频域分析参数
            'frequency_threshold': 0.4,
            'high_freq_ratio_threshold': 0.15,
            
            # 光流分析参数
            'optical_flow_threshold': 2.0,
            'motion_consistency_threshold': 0.7,
            
            # 深度分析参数
            'depth_variance_threshold': 100,
            'edge_density_threshold': 0.2,
            
            # 反射分析参数
            'reflection_pattern_threshold': 0.6,
            'specular_threshold': 200,
            
            # 颜色分析参数
            'color_temperature_range': (2500, 7500),
            'saturation_threshold': 0.3,
            
            # 综合判断阈值
            'real_confidence_threshold': 0.7,
            'spoofing_confidence_threshold': 0.6,
            
            # 时序分析参数
            'temporal_window': 10,
            'consistency_threshold': 0.8
        }
    
    def _init_detectors(self):
        """初始化检测器"""
        # 初始化光流检测器
        self.optical_flow = cv2.calcOpticalFlowPyrLK
        
        # 初始化特征检测器
        self.orb = cv2.ORB_create()
        
        # 历史帧缓存
        self.frame_history = []
        self.max_history_size = self.config['temporal_window']
        
        # 检测结果历史
        self.detection_history = []
    
    def detect_spoofing(self, image: np.ndarray, 
                       previous_frame: Optional[np.ndarray] = None) -> SpoofingDetectionResult:
        """检测图像是否为欺骗攻击"""
        try:
            evidence = {}
            
            # 1. 纹理分析
            texture_score = self._analyze_texture(image)
            evidence['texture_analysis'] = texture_score
            
            # 2. 频域分析
            frequency_score = self._analyze_frequency_domain(image)
            evidence['frequency_analysis'] = frequency_score
            
            # 3. 边缘分析
            edge_score = self._analyze_edge_characteristics(image)
            evidence['edge_analysis'] = edge_score
            
            # 4. 颜色分析
            color_score = self._analyze_color_characteristics(image)
            evidence['color_analysis'] = color_score
            
            # 5. 反射分析
            reflection_score = self._analyze_reflection_patterns(image)
            evidence['reflection_analysis'] = reflection_score
            
            # 6. 运动分析（如果有前一帧）
            if previous_frame is not None:
                motion_score = self._analyze_motion_patterns(image, previous_frame)
                evidence['motion_analysis'] = motion_score
            else:
                evidence['motion_analysis'] = 0.5  # 中性分数
            
            # 7. 深度线索分析
            depth_score = self._analyze_depth_cues(image)
            evidence['depth_analysis'] = depth_score
            
            # 8. 屏幕检测
            screen_score = self._detect_screen_patterns(image)
            evidence['screen_detection'] = screen_score
            
            # 综合判断
            is_real, spoofing_type, confidence = self._make_final_decision(evidence)
            
            # 计算风险等级
            risk_level = self._calculate_risk_level(confidence, spoofing_type)
            
            # 更新历史记录
            self._update_history(image, is_real, confidence)
            
            # 时序一致性检查
            temporal_confidence = self._check_temporal_consistency()
            confidence = (confidence + temporal_confidence) / 2
            
            return SpoofingDetectionResult(
                is_real=is_real,
                spoofing_type=spoofing_type,
                confidence=confidence,
                evidence=evidence,
                risk_level=risk_level
            )
            
        except Exception as e:
            self.logger.error(f"反欺骗检测失败: {e}")
            return SpoofingDetectionResult(
                is_real=True,  # 默认认为是真实的，避免误报
                spoofing_type=SpoofingType.UNKNOWN,
                confidence=0.5,
                evidence={},
                risk_level="medium"
            )
    
    def _analyze_texture(self, image: np.ndarray) -> float:
        """分析纹理特征"""
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 计算局部二值模式 (LBP)
        lbp = self._calculate_lbp(gray)
        
        # 计算纹理方差
        texture_variance = np.var(lbp)
        
        # 计算纹理均匀性
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(float)
        hist /= hist.sum()
        uniformity = np.sum(hist ** 2)
        
        # 真实物体通常有更丰富的纹理
        texture_richness = 1.0 - uniformity
        
        # 综合纹理分数 (值越高越可能是真实的)
        texture_score = (texture_variance / 1000 + texture_richness) / 2
        return min(texture_score, 1.0)
    
    def _calculate_lbp(self, gray: np.ndarray) -> np.ndarray:
        """计算局部二值模式"""
        rows, cols = gray.shape
        lbp = np.zeros_like(gray)
        
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                center = gray[i, j]
                code = 0
                
                # 8邻域比较
                neighbors = [
                    gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                    gray[i, j+1], gray[i+1, j+1], gray[i+1, j],
                    gray[i+1, j-1], gray[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                
                lbp[i, j] = code
        
        return lbp
    
    def _analyze_frequency_domain(self, image: np.ndarray) -> float:
        """分析频域特征"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 傅里叶变换
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # 计算高频成分比例
        rows, cols = magnitude_spectrum.shape
        center_row, center_col = rows // 2, cols // 2
        
        # 定义高频区域 (外围区域)
        y, x = np.ogrid[:rows, :cols]
        mask = (x - center_col) ** 2 + (y - center_row) ** 2 > (min(rows, cols) // 4) ** 2
        
        high_freq_energy = np.sum(magnitude_spectrum[mask])
        total_energy = np.sum(magnitude_spectrum)
        
        high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
        
        # 真实物体通常有更多高频细节
        return min(high_freq_ratio / self.config['high_freq_ratio_threshold'], 1.0)
    
    def _analyze_edge_characteristics(self, image: np.ndarray) -> float:
        """分析边缘特征"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Canny边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 计算边缘密度
        edge_density = np.sum(edges > 0) / edges.size
        
        # 计算边缘连续性
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # 计算轮廓长度分布
            contour_lengths = [cv2.arcLength(contour, True) for contour in contours]
            avg_contour_length = np.mean(contour_lengths)
            
            # 真实物体通常有更连续的边缘
            continuity_score = min(avg_contour_length / 100, 1.0)
        else:
            continuity_score = 0.0
        
        # 综合边缘分数
        edge_score = (edge_density / self.config['edge_density_threshold'] + continuity_score) / 2
        return min(edge_score, 1.0)
    
    def _analyze_color_characteristics(self, image: np.ndarray) -> float:
        """分析颜色特征"""
        if len(image.shape) != 3:
            return 0.5  # 灰度图无法分析颜色
        
        # 转换到HSV色彩空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 计算饱和度分布
        saturation = hsv[:, :, 1] / 255.0
        avg_saturation = np.mean(saturation)
        
        # 计算色调分布
        hue = hsv[:, :, 0]
        hue_hist, _ = np.histogram(hue, bins=180, range=(0, 180))
        hue_entropy = -np.sum((hue_hist / np.sum(hue_hist)) * np.log2(hue_hist / np.sum(hue_hist) + 1e-10))
        
        # 真实物体通常有更自然的颜色分布
        saturation_score = min(avg_saturation / self.config['saturation_threshold'], 1.0)
        diversity_score = min(hue_entropy / 6.0, 1.0)  # 最大熵约为log2(180)≈7.5
        
        return (saturation_score + diversity_score) / 2
    
    def _analyze_reflection_patterns(self, image: np.ndarray) -> float:
        """分析反射模式"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 检测镜面反射
        specular_mask = gray > self.config['specular_threshold']
        specular_ratio = np.sum(specular_mask) / specular_mask.size
        
        # 分析反射的空间分布
        if specular_ratio > 0.01:  # 如果有足够的高亮区域
            # 计算高亮区域的连通性
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            specular_cleaned = cv2.morphologyEx(specular_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(specular_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                # 计算反射区域的形状特征
                areas = [cv2.contourArea(contour) for contour in contours]
                avg_area = np.mean(areas)
                
                # 照片/海报通常有更规则的反射模式
                regularity_score = 1.0 - min(np.std(areas) / (avg_area + 1e-10), 1.0)
            else:
                regularity_score = 0.5
        else:
            regularity_score = 0.8  # 没有明显反射，可能是真实物体
        
        # 真实物体的反射通常更不规则
        return 1.0 - regularity_score
    
    def _analyze_motion_patterns(self, current_frame: np.ndarray, 
                               previous_frame: np.ndarray) -> float:
        """分析运动模式"""
        if len(current_frame.shape) == 3:
            gray1 = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = previous_frame
            gray2 = current_frame
        
        # 计算光流
        flow = cv2.calcOpticalFlowPyrLK(gray1, gray2, None, None)
        
        # 检测特征点
        corners = cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.01, minDistance=10)
        
        if corners is not None and len(corners) > 10:
            # 跟踪特征点
            next_corners, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, corners, None)
            
            # 计算运动向量
            good_corners = corners[status == 1]
            good_next = next_corners[status == 1]
            
            if len(good_corners) > 5:
                motion_vectors = good_next - good_corners
                motion_magnitudes = np.linalg.norm(motion_vectors, axis=2).flatten()
                
                # 分析运动一致性
                if len(motion_magnitudes) > 0:
                    motion_std = np.std(motion_magnitudes)
                    motion_mean = np.mean(motion_magnitudes)
                    
                    # 真实3D物体的运动通常更一致
                    consistency = 1.0 - min(motion_std / (motion_mean + 1e-10), 1.0)
                    return consistency
        
        return 0.5  # 无法确定
    
    def _analyze_depth_cues(self, image: np.ndarray) -> float:
        """分析深度线索"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 计算梯度
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算梯度幅值
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 分析梯度分布
        gradient_variance = np.var(gradient_magnitude)
        
        # 计算局部对比度变化
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
        
        contrast_variation = np.var(local_variance)
        
        # 真实3D物体通常有更丰富的深度变化
        depth_score = min((gradient_variance + contrast_variation) / 10000, 1.0)
        return depth_score
    
    def _detect_screen_patterns(self, image: np.ndarray) -> float:
        """检测屏幕模式"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 检测摩尔纹 (屏幕拍摄常见)
        # 使用FFT检测周期性模式
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # 寻找周期性峰值
        rows, cols = magnitude_spectrum.shape
        center_row, center_col = rows // 2, cols // 2
        
        # 在特定频率范围内寻找峰值
        peak_threshold = np.mean(magnitude_spectrum) + 2 * np.std(magnitude_spectrum)
        peaks = magnitude_spectrum > peak_threshold
        
        # 排除DC分量
        peaks[center_row-2:center_row+3, center_col-2:center_col+3] = False
        
        peak_count = np.sum(peaks)
        
        # 屏幕通常有更多周期性模式
        screen_likelihood = min(peak_count / 100, 1.0)
        
        return screen_likelihood
    
    def _make_final_decision(self, evidence: Dict[str, float]) -> Tuple[bool, SpoofingType, float]:
        """做出最终判断"""
        # 权重配置
        weights = {
            'texture_analysis': 0.2,
            'frequency_analysis': 0.15,
            'edge_analysis': 0.15,
            'color_analysis': 0.1,
            'reflection_analysis': 0.15,
            'motion_analysis': 0.1,
            'depth_analysis': 0.1,
            'screen_detection': 0.05
        }
        
        # 计算加权分数
        real_score = sum(evidence.get(key, 0.5) * weight for key, weight in weights.items())
        
        # 特殊检测逻辑
        spoofing_type = SpoofingType.REAL
        
        # 屏幕检测
        if evidence.get('screen_detection', 0) > 0.7:
            spoofing_type = SpoofingType.SCREEN
            real_score *= 0.3
        
        # 反射模式异常
        elif evidence.get('reflection_analysis', 0.5) < 0.3:
            spoofing_type = SpoofingType.PHOTO
            real_score *= 0.5
        
        # 纹理过于简单
        elif evidence.get('texture_analysis', 0.5) < 0.2:
            spoofing_type = SpoofingType.POSTER
            real_score *= 0.4
        
        # 运动不一致
        elif evidence.get('motion_analysis', 0.5) < 0.3:
            spoofing_type = SpoofingType.VIDEO
            real_score *= 0.6
        
        # 判断是否为真实
        is_real = real_score >= self.config['real_confidence_threshold']
        confidence = real_score if is_real else (1.0 - real_score)
        
        return is_real, spoofing_type, confidence
    
    def _calculate_risk_level(self, confidence: float, spoofing_type: SpoofingType) -> str:
        """计算风险等级"""
        if confidence >= 0.8:
            return "low"
        elif confidence >= 0.6:
            return "medium"
        else:
            return "high"
    
    def _update_history(self, image: np.ndarray, is_real: bool, confidence: float):
        """更新历史记录"""
        # 更新帧历史
        if len(self.frame_history) >= self.max_history_size:
            self.frame_history.pop(0)
        self.frame_history.append(image.copy())
        
        # 更新检测历史
        if len(self.detection_history) >= self.max_history_size:
            self.detection_history.pop(0)
        self.detection_history.append({'is_real': is_real, 'confidence': confidence})
    
    def _check_temporal_consistency(self) -> float:
        """检查时序一致性"""
        if len(self.detection_history) < 3:
            return 0.5
        
        # 计算最近几次检测的一致性
        recent_results = self.detection_history[-5:]
        real_count = sum(1 for result in recent_results if result['is_real'])
        consistency = abs(real_count / len(recent_results) - 0.5) * 2  # 转换到0-1范围
        
        return consistency
    
    def get_spoofing_explanation(self, result: SpoofingDetectionResult) -> str:
        """获取检测结果解释"""
        if result.is_real:
            return f"检测为真实目标 (置信度: {result.confidence:.2f})"
        
        explanations = {
            SpoofingType.PHOTO: "检测为照片攻击 - 缺乏3D深度信息和自然纹理",
            SpoofingType.VIDEO: "检测为视频攻击 - 运动模式不自然",
            SpoofingType.POSTER: "检测为海报攻击 - 纹理过于简单和规则",
            SpoofingType.SCREEN: "检测为屏幕攻击 - 存在周期性模式和摩尔纹",
            SpoofingType.MASK: "检测为面具攻击 - 面部特征不自然",
            SpoofingType.UNKNOWN: "检测为可疑目标 - 无法确定具体攻击类型"
        }
        
        base_explanation = explanations.get(result.spoofing_type, "检测为欺骗攻击")
        return f"{base_explanation} (置信度: {result.confidence:.2f}, 风险等级: {result.risk_level})"


# 使用示例
if __name__ == "__main__":
    # 创建反欺骗检测器
    detector = AntiSpoofingDetector()
    
    # 模拟测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 执行检测
    result = detector.detect_spoofing(test_image)
    
    print(f"检测结果: {result.is_real}")
    print(f"欺骗类型: {result.spoofing_type.value}")
    print(f"置信度: {result.confidence:.2f}")
    print(f"风险等级: {result.risk_level}")
    
    # 获取解释
    explanation = detector.get_spoofing_explanation(result)
    print(f"解释: {explanation}")
    
    # 显示证据
    print("\n检测证据:")
    for key, value in result.evidence.items():
        print(f"  {key}: {value:.3f}")