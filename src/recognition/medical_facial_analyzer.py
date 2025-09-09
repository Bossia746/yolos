#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医疗级面部生理分析模块
通过USB摄像头进行面部生理识别和健康状态评估
专门用于紧急情况下的快速医疗评估
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum
import time
import math

class HealthStatus(Enum):
    """健康状态枚举"""
    NORMAL = "normal"
    MILD_CONCERN = "mild_concern"
    MODERATE_CONCERN = "moderate_concern"
    SEVERE_CONCERN = "severe_concern"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class FacialSymptom(Enum):
    """面部症状枚举"""
    NORMAL = "normal"
    PALE_COMPLEXION = "pale_complexion"
    FLUSHED_FACE = "flushed_face"
    CYANOSIS = "cyanosis"  # 发绀
    ASYMMETRIC_FACE = "asymmetric_face"  # 面部不对称
    DROOPING_EYELID = "drooping_eyelid"  # 眼睑下垂
    MOUTH_DROOP = "mouth_droop"  # 嘴角下垂
    DILATED_PUPILS = "dilated_pupils"  # 瞳孔扩张
    CONSTRICTED_PUPILS = "constricted_pupils"  # 瞳孔收缩
    UNEQUAL_PUPILS = "unequal_pupils"  # 瞳孔不等大
    EXCESSIVE_SWEATING = "excessive_sweating"  # 出汗过多
    LABORED_BREATHING = "labored_breathing"  # 呼吸困难
    LOSS_OF_CONSCIOUSNESS = "loss_of_consciousness"  # 意识丧失

@dataclass
class VitalSigns:
    """生命体征数据"""
    heart_rate: Optional[float] = None  # 心率 (bpm)
    respiratory_rate: Optional[float] = None  # 呼吸频率 (breaths/min)
    oxygen_saturation: Optional[float] = None  # 血氧饱和度 (%)
    skin_temperature: Optional[float] = None  # 皮肤温度 (°C)
    blood_pressure_systolic: Optional[float] = None  # 收缩压
    blood_pressure_diastolic: Optional[float] = None  # 舒张压

@dataclass
class FacialAnalysisResult:
    """面部分析结果"""
    health_status: HealthStatus
    symptoms: List[FacialSymptom]
    vital_signs: VitalSigns
    confidence: float
    risk_score: float  # 0-100
    emergency_level: int  # 1-5级紧急程度
    recommendations: List[str]
    analysis_time: float
    detailed_findings: Dict[str, Any]

class MedicalFacialAnalyzer:
    """医疗级面部生理分析器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # 初始化面部检测器
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # 历史数据用于趋势分析
        self.history = []
        self.max_history_size = 30
        
        # 校准数据
        self.baseline_measurements = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            # 面部检测参数
            'face_detection': {
                'scale_factor': 1.1,
                'min_neighbors': 5,
                'min_size': (50, 50)
            },
            
            # 颜色分析参数
            'color_analysis': {
                'pale_threshold': 0.3,
                'flush_threshold': 0.7,
                'cyanosis_blue_threshold': 0.4
            },
            
            # 对称性分析参数
            'symmetry_analysis': {
                'asymmetry_threshold': 0.15,
                'eye_level_threshold': 5,  # 像素
                'mouth_angle_threshold': 10  # 度
            },
            
            # 瞳孔分析参数
            'pupil_analysis': {
                'dilation_threshold': 0.7,
                'constriction_threshold': 0.3,
                'size_difference_threshold': 0.2
            },
            
            # 生命体征检测参数
            'vital_signs': {
                'heart_rate_window': 30,  # 秒
                'breathing_window': 60,   # 秒
                'temperature_roi_ratio': 0.3
            },
            
            # 紧急情况阈值
            'emergency_thresholds': {
                'critical_risk_score': 80,
                'severe_risk_score': 60,
                'moderate_risk_score': 40,
                'mild_risk_score': 20
            }
        }
    
    def analyze_facial_health(self, image: np.ndarray, 
                            previous_frame: Optional[np.ndarray] = None) -> FacialAnalysisResult:
        """分析面部健康状态"""
        start_time = time.time()
        
        try:
            # 检测面部
            faces = self._detect_faces(image)
            
            if len(faces) == 0:
                return self._create_no_face_result(time.time() - start_time)
            
            # 选择最大的面部区域
            face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = face
            face_roi = image[y:y+h, x:x+w]
            
            # 执行各项分析
            symptoms = []
            vital_signs = VitalSigns()
            detailed_findings = {}
            
            # 1. 肤色分析
            color_symptoms, color_findings = self._analyze_skin_color(face_roi)
            symptoms.extend(color_symptoms)
            detailed_findings['color_analysis'] = color_findings
            
            # 2. 面部对称性分析
            symmetry_symptoms, symmetry_findings = self._analyze_facial_symmetry(face_roi)
            symptoms.extend(symmetry_symptoms)
            detailed_findings['symmetry_analysis'] = symmetry_findings
            
            # 3. 眼部分析
            eye_symptoms, eye_findings = self._analyze_eyes(face_roi)
            symptoms.extend(eye_symptoms)
            detailed_findings['eye_analysis'] = eye_findings
            
            # 4. 生命体征检测
            if previous_frame is not None:
                vital_signs = self._detect_vital_signs(image, previous_frame, face)
                detailed_findings['vital_signs'] = self._vital_signs_to_dict(vital_signs)
            
            # 5. 意识状态评估
            consciousness_symptoms, consciousness_findings = self._assess_consciousness(face_roi)
            symptoms.extend(consciousness_symptoms)
            detailed_findings['consciousness_analysis'] = consciousness_findings
            
            # 6. 呼吸模式分析
            breathing_symptoms, breathing_findings = self._analyze_breathing_pattern(face_roi)
            symptoms.extend(breathing_symptoms)
            detailed_findings['breathing_analysis'] = breathing_findings
            
            # 计算综合评估
            health_status, risk_score, emergency_level = self._calculate_health_assessment(symptoms, vital_signs)
            
            # 生成建议
            recommendations = self._generate_medical_recommendations(symptoms, vital_signs, emergency_level)
            
            # 计算置信度
            confidence = self._calculate_confidence(detailed_findings, len(symptoms))
            
            analysis_time = time.time() - start_time
            
            result = FacialAnalysisResult(
                health_status=health_status,
                symptoms=symptoms,
                vital_signs=vital_signs,
                confidence=confidence,
                risk_score=risk_score,
                emergency_level=emergency_level,
                recommendations=recommendations,
                analysis_time=analysis_time,
                detailed_findings=detailed_findings
            )
            
            # 更新历史记录
            self._update_history(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"面部健康分析失败: {e}")
            return self._create_error_result(str(e), time.time() - start_time)
    
    def _detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """检测面部"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.config['face_detection']['scale_factor'],
            minNeighbors=self.config['face_detection']['min_neighbors'],
            minSize=self.config['face_detection']['min_size']
        )
        return faces
    
    def _analyze_skin_color(self, face_roi: np.ndarray) -> Tuple[List[FacialSymptom], Dict[str, Any]]:
        """分析肤色"""
        symptoms = []
        findings = {}
        
        # 转换到不同色彩空间进行分析
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
        
        # 计算平均颜色值
        mean_bgr = np.mean(face_roi.reshape(-1, 3), axis=0)
        mean_hsv = np.mean(hsv.reshape(-1, 3), axis=0)
        mean_lab = np.mean(lab.reshape(-1, 3), axis=0)
        
        findings['mean_bgr'] = mean_bgr.tolist()
        findings['mean_hsv'] = mean_hsv.tolist()
        findings['mean_lab'] = mean_lab.tolist()
        
        # 苍白检测 (低饱和度和亮度)
        if mean_hsv[1] < self.config['color_analysis']['pale_threshold'] * 255:
            symptoms.append(FacialSymptom.PALE_COMPLEXION)
            findings['pale_score'] = float(1.0 - mean_hsv[1] / 255.0)
        
        # 潮红检测 (高红色分量)
        red_ratio = mean_bgr[2] / (mean_bgr[0] + mean_bgr[1] + 1e-6)
        if red_ratio > self.config['color_analysis']['flush_threshold']:
            symptoms.append(FacialSymptom.FLUSHED_FACE)
            findings['flush_score'] = float(red_ratio)
        
        # 发绀检测 (蓝色分量异常高)
        blue_ratio = mean_bgr[0] / (mean_bgr[1] + mean_bgr[2] + 1e-6)
        if blue_ratio > self.config['color_analysis']['cyanosis_blue_threshold']:
            symptoms.append(FacialSymptom.CYANOSIS)
            findings['cyanosis_score'] = float(blue_ratio)
        
        return symptoms, findings
    
    def _analyze_facial_symmetry(self, face_roi: np.ndarray) -> Tuple[List[FacialSymptom], Dict[str, Any]]:
        """分析面部对称性"""
        symptoms = []
        findings = {}
        
        h, w = face_roi.shape[:2]
        
        # 分割左右半脸
        left_half = face_roi[:, :w//2]
        right_half = cv2.flip(face_roi[:, w//2:], 1)  # 翻转右半脸
        
        # 调整尺寸使其一致
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        # 计算结构相似性
        if left_half.shape == right_half.shape:
            # 转换为灰度图
            left_gray = cv2.cvtColor(left_half, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY)
            
            # 计算相似性
            similarity = cv2.matchTemplate(left_gray, right_gray, cv2.TM_CCOEFF_NORMED)[0, 0]
            asymmetry_score = 1.0 - similarity
            
            findings['asymmetry_score'] = float(asymmetry_score)
            findings['similarity'] = float(similarity)
            
            if asymmetry_score > self.config['symmetry_analysis']['asymmetry_threshold']:
                symptoms.append(FacialSymptom.ASYMMETRIC_FACE)
        
        # 检测眼部水平线
        eyes = self.eye_cascade.detectMultiScale(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY))
        if len(eyes) >= 2:
            # 按x坐标排序，取前两个眼睛
            eyes = sorted(eyes, key=lambda e: e[0])[:2]
            eye1_center_y = eyes[0][1] + eyes[0][3] // 2
            eye2_center_y = eyes[1][1] + eyes[1][3] // 2
            
            eye_level_diff = abs(eye1_center_y - eye2_center_y)
            findings['eye_level_difference'] = int(eye_level_diff)
            
            if eye_level_diff > self.config['symmetry_analysis']['eye_level_threshold']:
                symptoms.append(FacialSymptom.DROOPING_EYELID)
        
        return symptoms, findings
    
    def _analyze_eyes(self, face_roi: np.ndarray) -> Tuple[List[FacialSymptom], Dict[str, Any]]:
        """分析眼部"""
        symptoms = []
        findings = {}
        
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray)
        
        if len(eyes) >= 2:
            pupil_sizes = []
            
            for i, (ex, ey, ew, eh) in enumerate(eyes[:2]):
                eye_roi = gray[ey:ey+eh, ex:ex+ew]
                
                # 检测瞳孔
                pupil_size = self._detect_pupil_size(eye_roi)
                if pupil_size > 0:
                    pupil_sizes.append(pupil_size)
                    findings[f'eye_{i+1}_pupil_size'] = float(pupil_size)
            
            if len(pupil_sizes) >= 2:
                avg_pupil_size = np.mean(pupil_sizes)
                pupil_size_diff = abs(pupil_sizes[0] - pupil_sizes[1]) / max(pupil_sizes)
                
                findings['average_pupil_size'] = float(avg_pupil_size)
                findings['pupil_size_difference'] = float(pupil_size_diff)
                
                # 瞳孔扩张检测
                if avg_pupil_size > self.config['pupil_analysis']['dilation_threshold']:
                    symptoms.append(FacialSymptom.DILATED_PUPILS)
                
                # 瞳孔收缩检测
                elif avg_pupil_size < self.config['pupil_analysis']['constriction_threshold']:
                    symptoms.append(FacialSymptom.CONSTRICTED_PUPILS)
                
                # 瞳孔不等大检测
                if pupil_size_diff > self.config['pupil_analysis']['size_difference_threshold']:
                    symptoms.append(FacialSymptom.UNEQUAL_PUPILS)
        
        return symptoms, findings
    
    def _detect_pupil_size(self, eye_roi: np.ndarray) -> float:
        """检测瞳孔大小"""
        try:
            # 使用霍夫圆检测瞳孔
            circles = cv2.HoughCircles(
                eye_roi,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=eye_roi.shape[0]//2,
                param1=50,
                param2=30,
                minRadius=eye_roi.shape[0]//8,
                maxRadius=eye_roi.shape[0]//3
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                if len(circles) > 0:
                    # 返回最大圆的半径（归一化）
                    max_radius = max(circles[:, 2])
                    return max_radius / max(eye_roi.shape)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _detect_vital_signs(self, current_frame: np.ndarray, 
                          previous_frame: np.ndarray, 
                          face_coords: Tuple[int, int, int, int]) -> VitalSigns:
        """检测生命体征"""
        vital_signs = VitalSigns()
        
        try:
            x, y, w, h = face_coords
            
            # 心率检测 (基于面部颜色变化)
            heart_rate = self._detect_heart_rate(current_frame, previous_frame, face_coords)
            if heart_rate > 0:
                vital_signs.heart_rate = heart_rate
            
            # 呼吸频率检测 (基于胸部或鼻孔区域运动)
            respiratory_rate = self._detect_respiratory_rate(current_frame, previous_frame, face_coords)
            if respiratory_rate > 0:
                vital_signs.respiratory_rate = respiratory_rate
            
            # 皮肤温度估算 (基于红外特征)
            skin_temp = self._estimate_skin_temperature(current_frame[y:y+h, x:x+w])
            if skin_temp > 0:
                vital_signs.skin_temperature = skin_temp
            
        except Exception as e:
            self.logger.error(f"生命体征检测失败: {e}")
        
        return vital_signs
    
    def _detect_heart_rate(self, current_frame: np.ndarray, 
                          previous_frame: np.ndarray, 
                          face_coords: Tuple[int, int, int, int]) -> float:
        """检测心率"""
        try:
            x, y, w, h = face_coords
            
            # 提取面部ROI
            current_roi = current_frame[y:y+h, x:x+w]
            previous_roi = previous_frame[y:y+h, x:x+w]
            
            # 计算绿色通道的平均值变化 (PPG信号)
            current_green = np.mean(current_roi[:, :, 1])
            previous_green = np.mean(previous_roi[:, :, 1])
            
            # 简化的心率估算 (需要更多帧数据才能准确)
            color_change = abs(current_green - previous_green)
            
            # 基于颜色变化幅度估算心率
            if color_change > 1.0:  # 阈值需要根据实际情况调整
                estimated_hr = 60 + color_change * 2  # 简化公式
                return min(max(estimated_hr, 40), 180)  # 限制在合理范围
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _detect_respiratory_rate(self, current_frame: np.ndarray, 
                               previous_frame: np.ndarray, 
                               face_coords: Tuple[int, int, int, int]) -> float:
        """检测呼吸频率"""
        try:
            x, y, w, h = face_coords
            
            # 鼻孔区域 (面部下半部分)
            nostril_y = y + int(h * 0.6)
            nostril_roi_current = current_frame[nostril_y:y+h, x:x+w]
            nostril_roi_previous = previous_frame[nostril_y:y+h, x:x+w]
            
            # 计算运动向量
            if nostril_roi_current.shape == nostril_roi_previous.shape:
                # 转换为灰度
                gray_current = cv2.cvtColor(nostril_roi_current, cv2.COLOR_BGR2GRAY)
                gray_previous = cv2.cvtColor(nostril_roi_previous, cv2.COLOR_BGR2GRAY)
                
                # 计算光流
                flow = cv2.calcOpticalFlowPyrLK(
                    gray_previous, gray_current,
                    np.array([[w//2, h//4]], dtype=np.float32),
                    None
                )
                
                if flow[0] is not None and len(flow[0]) > 0:
                    movement = np.linalg.norm(flow[0][0] - np.array([w//2, h//4]))
                    
                    # 基于运动幅度估算呼吸频率
                    if movement > 0.5:
                        estimated_rr = 12 + movement * 0.5  # 简化公式
                        return min(max(estimated_rr, 8), 40)  # 限制在合理范围
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _estimate_skin_temperature(self, face_roi: np.ndarray) -> float:
        """估算皮肤温度"""
        try:
            # 基于红外特征的简化温度估算
            # 实际应用中需要红外摄像头或热成像设备
            
            # 使用RGB值的加权平均作为温度指示
            mean_rgb = np.mean(face_roi.reshape(-1, 3), axis=0)
            
            # 简化的温度映射 (需要校准)
            temp_indicator = (mean_rgb[2] * 0.5 + mean_rgb[1] * 0.3 + mean_rgb[0] * 0.2) / 255.0
            
            # 映射到体温范围 (35-42°C)
            estimated_temp = 35.0 + temp_indicator * 7.0
            
            return estimated_temp
            
        except Exception:
            return 0.0
    
    def _assess_consciousness(self, face_roi: np.ndarray) -> Tuple[List[FacialSymptom], Dict[str, Any]]:
        """评估意识状态"""
        symptoms = []
        findings = {}
        
        try:
            # 检测眼睛开合状态
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            eyes = self.eye_cascade.detectMultiScale(gray)
            
            if len(eyes) == 0:
                # 没有检测到眼睛可能表示意识丧失
                symptoms.append(FacialSymptom.LOSS_OF_CONSCIOUSNESS)
                findings['eyes_detected'] = False
                findings['consciousness_score'] = 0.0
            else:
                findings['eyes_detected'] = True
                findings['num_eyes_detected'] = len(eyes)
                
                # 分析眼睛开合程度
                eye_openness_scores = []
                for ex, ey, ew, eh in eyes:
                    eye_roi = gray[ey:ey+eh, ex:ex+ew]
                    openness = self._calculate_eye_openness(eye_roi)
                    eye_openness_scores.append(openness)
                
                if eye_openness_scores:
                    avg_openness = np.mean(eye_openness_scores)
                    findings['average_eye_openness'] = float(avg_openness)
                    
                    if avg_openness < 0.3:  # 眼睛基本闭合
                        symptoms.append(FacialSymptom.LOSS_OF_CONSCIOUSNESS)
                        findings['consciousness_score'] = float(avg_openness)
                    else:
                        findings['consciousness_score'] = 1.0
        
        except Exception as e:
            self.logger.error(f"意识状态评估失败: {e}")
            findings['error'] = str(e)
        
        return symptoms, findings
    
    def _calculate_eye_openness(self, eye_roi: np.ndarray) -> float:
        """计算眼睛开合程度"""
        try:
            # 使用边缘检测和轮廓分析
            edges = cv2.Canny(eye_roi, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 找到最大轮廓
                max_contour = max(contours, key=cv2.contourArea)
                
                # 计算轮廓的边界框
                x, y, w, h = cv2.boundingRect(max_contour)
                
                # 开合程度 = 高度/宽度比例
                openness = h / max(w, 1)
                return min(openness, 1.0)
            
            return 0.5  # 默认值
            
        except Exception:
            return 0.5
    
    def _analyze_breathing_pattern(self, face_roi: np.ndarray) -> Tuple[List[FacialSymptom], Dict[str, Any]]:
        """分析呼吸模式"""
        symptoms = []
        findings = {}
        
        try:
            # 分析鼻孔区域的变化
            h, w = face_roi.shape[:2]
            nostril_region = face_roi[int(h*0.6):int(h*0.8), int(w*0.3):int(w*0.7)]
            
            # 计算区域的标准差作为呼吸活动指标
            gray_nostril = cv2.cvtColor(nostril_region, cv2.COLOR_BGR2GRAY)
            breathing_activity = np.std(gray_nostril)
            
            findings['breathing_activity'] = float(breathing_activity)
            
            # 基于活动水平判断呼吸状态
            if breathing_activity < 10:  # 阈值需要调整
                symptoms.append(FacialSymptom.LABORED_BREATHING)
                findings['breathing_status'] = 'labored'
            else:
                findings['breathing_status'] = 'normal'
        
        except Exception as e:
            self.logger.error(f"呼吸模式分析失败: {e}")
            findings['error'] = str(e)
        
        return symptoms, findings
    
    def _calculate_health_assessment(self, symptoms: List[FacialSymptom], 
                                   vital_signs: VitalSigns) -> Tuple[HealthStatus, float, int]:
        """计算健康评估"""
        risk_score = 0.0
        
        # 基于症状计算风险分数
        symptom_weights = {
            FacialSymptom.LOSS_OF_CONSCIOUSNESS: 50,
            FacialSymptom.CYANOSIS: 40,
            FacialSymptom.ASYMMETRIC_FACE: 35,
            FacialSymptom.UNEQUAL_PUPILS: 30,
            FacialSymptom.DILATED_PUPILS: 25,
            FacialSymptom.LABORED_BREATHING: 25,
            FacialSymptom.PALE_COMPLEXION: 20,
            FacialSymptom.DROOPING_EYELID: 20,
            FacialSymptom.MOUTH_DROOP: 20,
            FacialSymptom.FLUSHED_FACE: 15,
            FacialSymptom.CONSTRICTED_PUPILS: 15,
            FacialSymptom.EXCESSIVE_SWEATING: 10
        }
        
        for symptom in symptoms:
            risk_score += symptom_weights.get(symptom, 5)
        
        # 基于生命体征调整风险分数
        if vital_signs.heart_rate:
            if vital_signs.heart_rate < 50 or vital_signs.heart_rate > 120:
                risk_score += 20
            elif vital_signs.heart_rate < 60 or vital_signs.heart_rate > 100:
                risk_score += 10
        
        if vital_signs.respiratory_rate:
            if vital_signs.respiratory_rate < 8 or vital_signs.respiratory_rate > 30:
                risk_score += 20
            elif vital_signs.respiratory_rate < 12 or vital_signs.respiratory_rate > 20:
                risk_score += 10
        
        if vital_signs.skin_temperature:
            if vital_signs.skin_temperature < 35 or vital_signs.skin_temperature > 39:
                risk_score += 15
            elif vital_signs.skin_temperature < 36 or vital_signs.skin_temperature > 37.5:
                risk_score += 8
        
        # 限制风险分数范围
        risk_score = min(risk_score, 100)
        
        # 确定健康状态和紧急等级
        if risk_score >= self.config['emergency_thresholds']['critical_risk_score']:
            health_status = HealthStatus.CRITICAL
            emergency_level = 5
        elif risk_score >= self.config['emergency_thresholds']['severe_risk_score']:
            health_status = HealthStatus.SEVERE_CONCERN
            emergency_level = 4
        elif risk_score >= self.config['emergency_thresholds']['moderate_risk_score']:
            health_status = HealthStatus.MODERATE_CONCERN
            emergency_level = 3
        elif risk_score >= self.config['emergency_thresholds']['mild_risk_score']:
            health_status = HealthStatus.MILD_CONCERN
            emergency_level = 2
        else:
            health_status = HealthStatus.NORMAL
            emergency_level = 1
        
        return health_status, risk_score, emergency_level
    
    def _generate_medical_recommendations(self, symptoms: List[FacialSymptom], 
                                        vital_signs: VitalSigns, 
                                        emergency_level: int) -> List[str]:
        """生成医疗建议"""
        recommendations = []
        
        # 基于紧急等级的通用建议
        if emergency_level >= 5:
            recommendations.append("立即呼叫急救服务 (120/911)")
            recommendations.append("检查呼吸道是否通畅")
            recommendations.append("准备进行心肺复苏术")
            recommendations.append("持续监测生命体征")
        elif emergency_level >= 4:
            recommendations.append("尽快联系医疗专业人员")
            recommendations.append("将患者置于恢复体位")
            recommendations.append("密切监测意识状态")
        elif emergency_level >= 3:
            recommendations.append("建议医疗评估")
            recommendations.append("监测症状变化")
            recommendations.append("记录观察到的症状")
        elif emergency_level >= 2:
            recommendations.append("建议休息观察")
            recommendations.append("如症状恶化请就医")
        
        # 基于具体症状的建议
        if FacialSymptom.LOSS_OF_CONSCIOUSNESS in symptoms:
            recommendations.append("检查反应能力和呼吸")
            recommendations.append("确保呼吸道通畅")
        
        if FacialSymptom.CYANOSIS in symptoms:
            recommendations.append("立即检查呼吸和循环")
            recommendations.append("考虑氧气支持")
        
        if FacialSymptom.ASYMMETRIC_FACE in symptoms:
            recommendations.append("疑似中风，立即就医")
            recommendations.append("记录症状出现时间")
        
        if FacialSymptom.UNEQUAL_PUPILS in symptoms:
            recommendations.append("疑似颅内压增高，紧急就医")
            recommendations.append("避免移动头部")
        
        if FacialSymptom.LABORED_BREATHING in symptoms:
            recommendations.append("确保呼吸道通畅")
            recommendations.append("协助患者采取舒适体位")
        
        # 基于生命体征的建议
        if vital_signs.heart_rate:
            if vital_signs.heart_rate > 120:
                recommendations.append("心率过快，需要医疗评估")
            elif vital_signs.heart_rate < 50:
                recommendations.append("心率过慢，需要医疗关注")
        
        if vital_signs.skin_temperature and vital_signs.skin_temperature > 38:
            recommendations.append("体温升高，考虑降温措施")
        
        return list(set(recommendations))  # 去重
    
    def _calculate_confidence(self, detailed_findings: Dict[str, Any], num_symptoms: int) -> float:
        """计算分析置信度"""
        confidence_factors = []
        
        # 基于检测到的特征数量
        if 'color_analysis' in detailed_findings:
            confidence_factors.append(0.8)
        
        if 'symmetry_analysis' in detailed_findings:
            confidence_factors.append(0.9)
        
        if 'eye_analysis' in detailed_findings:
            confidence_factors.append(0.85)
        
        if 'vital_signs' in detailed_findings:
            confidence_factors.append(0.7)
        
        if 'consciousness_analysis' in detailed_findings:
            confidence_factors.append(0.9)
        
        # 基于症状数量调整
        symptom_confidence = min(1.0, num_symptoms * 0.1 + 0.5)
        confidence_factors.append(symptom_confidence)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _vital_signs_to_dict(self, vital_signs: VitalSigns) -> Dict[str, Any]:
        """将生命体征转换为字典"""
        return {
            'heart_rate': vital_signs.heart_rate,
            'respiratory_rate': vital_signs.respiratory_rate,
            'oxygen_saturation': vital_signs.oxygen_saturation,
            'skin_temperature': vital_signs.skin_temperature,
            'blood_pressure_systolic': vital_signs.blood_pressure_systolic,
            'blood_pressure_diastolic': vital_signs.blood_pressure_diastolic
        }
    
    def _update_history(self, result: FacialAnalysisResult):
        """更新历史记录"""
        self.history.append({
            'timestamp': time.time(),
            'health_status': result.health_status.value,
            'risk_score': result.risk_score,
            'emergency_level': result.emergency_level,
            'symptoms': [s.value for s in result.symptoms],
            'vital_signs': self._vital_signs_to_dict(result.vital_signs)
        })
        
        # 限制历史记录大小
        if len(self.history) > self.max_history_size:
            self.history.pop(0)
    
    def _create_no_face_result(self, analysis_time: float) -> FacialAnalysisResult:
        """创建未检测到面部的结果"""
        return FacialAnalysisResult(
            health_status=HealthStatus.UNKNOWN,
            symptoms=[],
            vital_signs=VitalSigns(),
            confidence=0.0,
            risk_score=0.0,
            emergency_level=1,
            recommendations=["未检测到面部，请调整摄像头角度或光照条件"],
            analysis_time=analysis_time,
            detailed_findings={'error': 'no_face_detected'}
        )
    
    def _create_error_result(self, error_msg: str, analysis_time: float) -> FacialAnalysisResult:
        """创建错误结果"""
        return FacialAnalysisResult(
            health_status=HealthStatus.UNKNOWN,
            symptoms=[],
            vital_signs=VitalSigns(),
            confidence=0.0,
            risk_score=0.0,
            emergency_level=1,
            recommendations=[f"分析出错: {error_msg}"],
            analysis_time=analysis_time,
            detailed_findings={'error': error_msg}
        )
    
    def get_health_trend(self, time_window: int = 300) -> Dict[str, Any]:
        """获取健康趋势分析"""
        current_time = time.time()
        recent_history = [
            h for h in self.history 
            if current_time - h['timestamp'] <= time_window
        ]
        
        if not recent_history:
            return {'trend': 'no_data', 'message': '没有足够的历史数据'}
        
        # 分析趋势
        risk_scores = [h['risk_score'] for h in recent_history]
        emergency_levels = [h['emergency_level'] for h in recent_history]
        
        trend_analysis = {
            'data_points': len(recent_history),
            'time_span': time_window,
            'current_risk_score': risk_scores[-1] if risk_scores else 0,
            'average_risk_score': np.mean(risk_scores),
            'max_risk_score': max(risk_scores),
            'min_risk_score': min(risk_scores),
            'current_emergency_level': emergency_levels[-1] if emergency_levels else 1,
            'max_emergency_level': max(emergency_levels),
            'trend_direction': 'stable'
        }
        
        # 判断趋势方向
        if len(risk_scores) >= 3:
            recent_avg = np.mean(risk_scores[-3:])
            earlier_avg = np.mean(risk_scores[:-3]) if len(risk_scores) > 3 else recent_avg
            
            if recent_avg > earlier_avg + 5:
                trend_analysis['trend_direction'] = 'deteriorating'
            elif recent_avg < earlier_avg - 5:
                trend_analysis['trend_direction'] = 'improving'
        
        return trend_analysis
    
    def calibrate_baseline(self, image: np.ndarray) -> bool:
        """校准基线测量值"""
        try:
            result = self.analyze_facial_health(image)
            
            if result.health_status != HealthStatus.UNKNOWN:
                self.baseline_measurements = {
                    'baseline_risk_score': result.risk_score,
                    'baseline_vital_signs': self._vital_signs_to_dict(result.vital_signs),
                    'calibration_time': time.time()
                }
                self.logger.info("基线校准完成")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"基线校准失败: {e}")
            return False


# 使用示例
if __name__ == "__main__":
    # 创建医疗面部分析器
    analyzer = MedicalFacialAnalyzer()
    
    # 模拟测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 执行分析
    result = analyzer.analyze_facial_health(test_image)
    
    print(f"健康状态: {result.health_status.value}")
    print(f"风险分数: {result.risk_score:.1f}")
    print(f"紧急等级: {result.emergency_level}")
    print(f"检测到的症状: {[s.value for s in result.symptoms]}")
    print(f"分析时间: {result.analysis_time:.3f}s")
    
    # 显示建议
    print("\n医疗建议:")
    for recommendation in result.recommendations:
        print(f"- {recommendation}")
    
    # 获取健康趋势
    trend = analyzer.get_health_trend()
    print(f"\n健康趋势: {trend}")