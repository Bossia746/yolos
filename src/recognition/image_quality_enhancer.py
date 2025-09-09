#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像质量增强模块
处理反光、曝光、光线偏暗等图像质量问题
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging
from dataclasses import dataclass

@dataclass
class ImageQualityMetrics:
    """图像质量指标"""
    brightness: float
    contrast: float
    sharpness: float
    noise_level: float
    overexposure_ratio: float
    underexposure_ratio: float
    reflection_score: float
    quality_score: float

class ImageQualityEnhancer:
    """图像质量增强器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            # 亮度调整参数
            'brightness_target': 128,
            'brightness_tolerance': 30,
            
            # 对比度增强参数
            'contrast_alpha': 1.2,
            'contrast_beta': 10,
            
            # 反光检测参数
            'reflection_threshold': 240,
            'reflection_area_threshold': 0.05,
            
            # 曝光检测参数
            'overexposure_threshold': 250,
            'underexposure_threshold': 20,
            'exposure_area_threshold': 0.1,
            
            # 降噪参数
            'denoise_strength': 10,
            'denoise_template_window': 7,
            'denoise_search_window': 21,
            
            # 锐化参数
            'sharpen_strength': 0.5,
            
            # 质量阈值
            'min_quality_score': 0.6
        }
    
    def analyze_image_quality(self, image: np.ndarray) -> ImageQualityMetrics:
        """分析图像质量"""
        try:
            # 转换为灰度图像进行分析
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 计算亮度
            brightness = np.mean(gray)
            
            # 计算对比度
            contrast = np.std(gray)
            
            # 计算锐度 (拉普拉斯方差)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = np.var(laplacian)
            
            # 计算噪声水平
            noise_level = self._estimate_noise_level(gray)
            
            # 检测过曝和欠曝
            overexposure_ratio = np.sum(gray > self.config['overexposure_threshold']) / gray.size
            underexposure_ratio = np.sum(gray < self.config['underexposure_threshold']) / gray.size
            
            # 检测反光
            reflection_score = self._detect_reflection(image)
            
            # 计算综合质量分数
            quality_score = self._calculate_quality_score(
                brightness, contrast, sharpness, noise_level,
                overexposure_ratio, underexposure_ratio, reflection_score
            )
            
            return ImageQualityMetrics(
                brightness=brightness,
                contrast=contrast,
                sharpness=sharpness,
                noise_level=noise_level,
                overexposure_ratio=overexposure_ratio,
                underexposure_ratio=underexposure_ratio,
                reflection_score=reflection_score,
                quality_score=quality_score
            )
            
        except Exception as e:
            self.logger.error(f"图像质量分析失败: {e}")
            return ImageQualityMetrics(0, 0, 0, 1, 1, 1, 1, 0)
    
    def _estimate_noise_level(self, gray: np.ndarray) -> float:
        """估计噪声水平"""
        # 使用高斯拉普拉斯算子估计噪声
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        convolved = cv2.filter2D(gray, cv2.CV_64F, kernel)
        noise_variance = np.var(convolved)
        
        # 归一化到0-1范围
        return min(noise_variance / 10000, 1.0)
    
    def _detect_reflection(self, image: np.ndarray) -> float:
        """检测反光区域"""
        if len(image.shape) == 3:
            # 转换到HSV色彩空间
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]
        else:
            v_channel = image
        
        # 检测高亮区域
        bright_mask = v_channel > self.config['reflection_threshold']
        
        # 形态学操作去除噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bright_mask = cv2.morphologyEx(bright_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        # 计算反光区域比例
        reflection_ratio = np.sum(bright_mask) / bright_mask.size
        
        return min(reflection_ratio / self.config['reflection_area_threshold'], 1.0)
    
    def _calculate_quality_score(self, brightness: float, contrast: float, 
                               sharpness: float, noise_level: float,
                               overexposure_ratio: float, underexposure_ratio: float,
                               reflection_score: float) -> float:
        """计算综合质量分数"""
        
        # 亮度分数 (理想亮度128附近)
        brightness_score = 1.0 - abs(brightness - self.config['brightness_target']) / 128.0
        brightness_score = max(0, brightness_score)
        
        # 对比度分数 (对比度越高越好，但有上限)
        contrast_score = min(contrast / 50.0, 1.0)
        
        # 锐度分数 (锐度越高越好，但有上限)
        sharpness_score = min(sharpness / 1000.0, 1.0)
        
        # 噪声分数 (噪声越低越好)
        noise_score = 1.0 - noise_level
        
        # 曝光分数 (过曝和欠曝都会降低分数)
        exposure_score = 1.0 - (overexposure_ratio + underexposure_ratio)
        exposure_score = max(0, exposure_score)
        
        # 反光分数 (反光越少越好)
        reflection_quality_score = 1.0 - reflection_score
        
        # 加权平均
        weights = {
            'brightness': 0.2,
            'contrast': 0.15,
            'sharpness': 0.2,
            'noise': 0.15,
            'exposure': 0.2,
            'reflection': 0.1
        }
        
        quality_score = (
            weights['brightness'] * brightness_score +
            weights['contrast'] * contrast_score +
            weights['sharpness'] * sharpness_score +
            weights['noise'] * noise_score +
            weights['exposure'] * exposure_score +
            weights['reflection'] * reflection_quality_score
        )
        
        return max(0, min(1, quality_score))
    
    def enhance_image(self, image: np.ndarray, 
                     quality_metrics: Optional[ImageQualityMetrics] = None) -> np.ndarray:
        """增强图像质量"""
        try:
            if quality_metrics is None:
                quality_metrics = self.analyze_image_quality(image)
            
            enhanced = image.copy()
            
            # 1. 亮度调整
            enhanced = self._adjust_brightness(enhanced, quality_metrics.brightness)
            
            # 2. 对比度增强
            enhanced = self._enhance_contrast(enhanced, quality_metrics.contrast)
            
            # 3. 反光处理
            if quality_metrics.reflection_score > 0.3:
                enhanced = self._reduce_reflection(enhanced)
            
            # 4. 曝光修正
            if quality_metrics.overexposure_ratio > 0.1 or quality_metrics.underexposure_ratio > 0.1:
                enhanced = self._correct_exposure(enhanced)
            
            # 5. 降噪
            if quality_metrics.noise_level > 0.3:
                enhanced = self._denoise_image(enhanced)
            
            # 6. 锐化
            if quality_metrics.sharpness < 500:
                enhanced = self._sharpen_image(enhanced)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"图像增强失败: {e}")
            return image
    
    def _adjust_brightness(self, image: np.ndarray, current_brightness: float) -> np.ndarray:
        """调整亮度"""
        target_brightness = self.config['brightness_target']
        brightness_diff = target_brightness - current_brightness
        
        # 限制调整幅度
        brightness_diff = np.clip(brightness_diff, -50, 50)
        
        # 应用亮度调整
        adjusted = cv2.convertScaleAbs(image, alpha=1.0, beta=brightness_diff)
        return adjusted
    
    def _enhance_contrast(self, image: np.ndarray, current_contrast: float) -> np.ndarray:
        """增强对比度"""
        # 使用CLAHE (对比度限制自适应直方图均衡)
        if len(image.shape) == 3:
            # 彩色图像：在LAB色彩空间的L通道应用CLAHE
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # 灰度图像：直接应用CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        return enhanced
    
    def _reduce_reflection(self, image: np.ndarray) -> np.ndarray:
        """减少反光"""
        # 使用inpainting技术修复反光区域
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 检测反光区域
        reflection_mask = gray > self.config['reflection_threshold']
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        reflection_mask = cv2.morphologyEx(reflection_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        # 使用inpainting修复
        if len(image.shape) == 3:
            enhanced = cv2.inpaint(image, reflection_mask, 3, cv2.INPAINT_TELEA)
        else:
            enhanced = cv2.inpaint(image, reflection_mask, 3, cv2.INPAINT_TELEA)
        
        return enhanced
    
    def _correct_exposure(self, image: np.ndarray) -> np.ndarray:
        """修正曝光"""
        # 使用Gamma校正
        if len(image.shape) == 3:
            # 转换到HSV，只调整V通道
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2].astype(np.float32) / 255.0
            
            # 计算适当的gamma值
            mean_brightness = np.mean(v_channel)
            if mean_brightness < 0.3:  # 欠曝
                gamma = 0.7
            elif mean_brightness > 0.7:  # 过曝
                gamma = 1.3
            else:
                gamma = 1.0
            
            # 应用gamma校正
            v_corrected = np.power(v_channel, gamma)
            hsv[:, :, 2] = (v_corrected * 255).astype(np.uint8)
            
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            # 灰度图像直接应用gamma校正
            normalized = image.astype(np.float32) / 255.0
            mean_brightness = np.mean(normalized)
            
            if mean_brightness < 0.3:
                gamma = 0.7
            elif mean_brightness > 0.7:
                gamma = 1.3
            else:
                gamma = 1.0
            
            corrected = np.power(normalized, gamma)
            enhanced = (corrected * 255).astype(np.uint8)
        
        return enhanced
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """图像降噪"""
        if len(image.shape) == 3:
            # 彩色图像使用Non-local Means降噪
            denoised = cv2.fastNlMeansDenoisingColored(
                image, None,
                self.config['denoise_strength'],
                self.config['denoise_strength'],
                self.config['denoise_template_window'],
                self.config['denoise_search_window']
            )
        else:
            # 灰度图像降噪
            denoised = cv2.fastNlMeansDenoising(
                image, None,
                self.config['denoise_strength'],
                self.config['denoise_template_window'],
                self.config['denoise_search_window']
            )
        
        return denoised
    
    def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """图像锐化"""
        # 使用Unsharp Mask锐化
        if len(image.shape) == 3:
            # 彩色图像
            blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
            sharpened = cv2.addWeighted(image, 1.0 + self.config['sharpen_strength'], 
                                      blurred, -self.config['sharpen_strength'], 0)
        else:
            # 灰度图像
            blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
            sharpened = cv2.addWeighted(image, 1.0 + self.config['sharpen_strength'], 
                                      blurred, -self.config['sharpen_strength'], 0)
        
        return sharpened
    
    def is_image_acceptable(self, image: np.ndarray) -> Tuple[bool, ImageQualityMetrics]:
        """判断图像质量是否可接受"""
        quality_metrics = self.analyze_image_quality(image)
        is_acceptable = quality_metrics.quality_score >= self.config['min_quality_score']
        
        return is_acceptable, quality_metrics
    
    def get_enhancement_recommendations(self, quality_metrics: ImageQualityMetrics) -> Dict[str, str]:
        """获取图像增强建议"""
        recommendations = {}
        
        if quality_metrics.brightness < 80:
            recommendations['brightness'] = "图像过暗，建议增加光源或调整曝光"
        elif quality_metrics.brightness > 180:
            recommendations['brightness'] = "图像过亮，建议减少光源或调整曝光"
        
        if quality_metrics.contrast < 30:
            recommendations['contrast'] = "对比度过低，建议调整光照条件"
        
        if quality_metrics.reflection_score > 0.3:
            recommendations['reflection'] = "检测到反光，建议调整光源角度或使用偏振滤镜"
        
        if quality_metrics.overexposure_ratio > 0.1:
            recommendations['overexposure'] = "存在过曝区域，建议降低曝光或使用渐变滤镜"
        
        if quality_metrics.underexposure_ratio > 0.1:
            recommendations['underexposure'] = "存在欠曝区域，建议增加补光"
        
        if quality_metrics.noise_level > 0.3:
            recommendations['noise'] = "噪声水平较高，建议降低ISO或改善光照"
        
        if quality_metrics.sharpness < 300:
            recommendations['sharpness'] = "图像模糊，建议检查对焦或减少相机抖动"
        
        return recommendations


class AdaptiveImageProcessor:
    """自适应图像处理器"""
    
    def __init__(self):
        self.enhancer = ImageQualityEnhancer()
        self.logger = logging.getLogger(__name__)
        
    def process_image_stream(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """处理图像流"""
        try:
            # 分析图像质量
            is_acceptable, quality_metrics = self.enhancer.is_image_acceptable(image)
            
            processing_info = {
                'original_quality': quality_metrics.quality_score,
                'is_acceptable': is_acceptable,
                'enhancements_applied': []
            }
            
            if not is_acceptable:
                # 应用图像增强
                enhanced_image = self.enhancer.enhance_image(image, quality_metrics)
                
                # 记录应用的增强
                if quality_metrics.brightness < 80 or quality_metrics.brightness > 180:
                    processing_info['enhancements_applied'].append('brightness_adjustment')
                
                if quality_metrics.contrast < 30:
                    processing_info['enhancements_applied'].append('contrast_enhancement')
                
                if quality_metrics.reflection_score > 0.3:
                    processing_info['enhancements_applied'].append('reflection_reduction')
                
                if quality_metrics.overexposure_ratio > 0.1 or quality_metrics.underexposure_ratio > 0.1:
                    processing_info['enhancements_applied'].append('exposure_correction')
                
                if quality_metrics.noise_level > 0.3:
                    processing_info['enhancements_applied'].append('denoising')
                
                if quality_metrics.sharpness < 300:
                    processing_info['enhancements_applied'].append('sharpening')
                
                # 重新评估增强后的质量
                enhanced_quality = self.enhancer.analyze_image_quality(enhanced_image)
                processing_info['enhanced_quality'] = enhanced_quality.quality_score
                
                return enhanced_image, processing_info
            else:
                return image, processing_info
                
        except Exception as e:
            self.logger.error(f"图像处理失败: {e}")
            return image, {'error': str(e)}


# 使用示例
if __name__ == "__main__":
    # 创建图像质量增强器
    enhancer = ImageQualityEnhancer()
    
    # 模拟测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 分析图像质量
    quality_metrics = enhancer.analyze_image_quality(test_image)
    print(f"图像质量分数: {quality_metrics.quality_score:.2f}")
    
    # 检查是否可接受
    is_acceptable, _ = enhancer.is_image_acceptable(test_image)
    print(f"图像质量可接受: {is_acceptable}")
    
    # 获取增强建议
    recommendations = enhancer.get_enhancement_recommendations(quality_metrics)
    for issue, suggestion in recommendations.items():
        print(f"{issue}: {suggestion}")
    
    # 增强图像
    enhanced_image = enhancer.enhance_image(test_image)
    print("图像增强完成")# -*- coding: utf-8 -*-
"""
图像质量增强模块
处理反光、曝光、光线偏暗等图像质量问题
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging
from dataclasses import dataclass

@dataclass
class ImageQualityMetrics:
    """图像质量指标"""
    brightness: float
    contrast: float
    sharpness: float
    noise_level: float
    overexposure_ratio: float
    underexposure_ratio: float
    reflection_score: float
    quality_score: float

class ImageQualityEnhancer:
    """图像质量增强器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            # 亮度调整参数
            'brightness_target': 128,
            'brightness_tolerance': 30,
            
            # 对比度增强参数
            'contrast_alpha': 1.2,
            'contrast_beta': 10,
            
            # 反光检测参数
            'reflection_threshold': 240,
            'reflection_area_threshold': 0.05,
            
            # 曝光检测参数
            'overexposure_threshold': 250,
            'underexposure_threshold': 20,
            'exposure_area_threshold': 0.1,
            
            # 降噪参数
            'denoise_strength': 10,
            'denoise_template_window': 7,
            'denoise_search_window': 21,
            
            # 锐化参数
            'sharpen_strength': 0.5,
            
            # 质量阈值
            'min_quality_score': 0.6
        }
    
    def analyze_image_quality(self, image: np.ndarray) -> ImageQualityMetrics:
        """分析图像质量"""
        try:
            # 转换为灰度图像进行分析
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 计算亮度
            brightness = np.mean(gray)
            
            # 计算对比度
            contrast = np.std(gray)
            
            # 计算锐度 (拉普拉斯方差)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = np.var(laplacian)
            
            # 计算噪声水平
            noise_level = self._estimate_noise_level(gray)
            
            # 检测过曝和欠曝
            overexposure_ratio = np.sum(gray > self.config['overexposure_threshold']) / gray.size
            underexposure_ratio = np.sum(gray < self.config['underexposure_threshold']) / gray.size
            
            # 检测反光
            reflection_score = self._detect_reflection(image)
            
            # 计算综合质量分数
            quality_score = self._calculate_quality_score(
                brightness, contrast, sharpness, noise_level,
                overexposure_ratio, underexposure_ratio, reflection_score
            )
            
            return ImageQualityMetrics(
                brightness=brightness,
                contrast=contrast,
                sharpness=sharpness,
                noise_level=noise_level,
                overexposure_ratio=overexposure_ratio,
                underexposure_ratio=underexposure_ratio,
                reflection_score=reflection_score,
                quality_score=quality_score
            )
            
        except Exception as e:
            self.logger.error(f"图像质量分析失败: {e}")
            return ImageQualityMetrics(0, 0, 0, 1, 1, 1, 1, 0)
    
    def _estimate_noise_level(self, gray: np.ndarray) -> float:
        """估计噪声水平"""
        # 使用高斯拉普拉斯算子估计噪声
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        convolved = cv2.filter2D(gray, cv2.CV_64F, kernel)
        noise_variance = np.var(convolved)
        
        # 归一化到0-1范围
        return min(noise_variance / 10000, 1.0)
    
    def _detect_reflection(self, image: np.ndarray) -> float:
        """检测反光区域"""
        if len(image.shape) == 3:
            # 转换到HSV色彩空间
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]
        else:
            v_channel = image
        
        # 检测高亮区域
        bright_mask = v_channel > self.config['reflection_threshold']
        
        # 形态学操作去除噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bright_mask = cv2.morphologyEx(bright_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        # 计算反光区域比例
        reflection_ratio = np.sum(bright_mask) / bright_mask.size
        
        return min(reflection_ratio / self.config['reflection_area_threshold'], 1.0)
    
    def _calculate_quality_score(self, brightness: float, contrast: float, 
                               sharpness: float, noise_level: float,
                               overexposure_ratio: float, underexposure_ratio: float,
                               reflection_score: float) -> float:
        """计算综合质量分数"""
        
        # 亮度分数 (理想亮度128附近)
        brightness_score = 1.0 - abs(brightness - self.config['brightness_target']) / 128.0
        brightness_score = max(0, brightness_score)
        
        # 对比度分数 (对比度越高越好，但有上限)
        contrast_score = min(contrast / 50.0, 1.0)
        
        # 锐度分数 (锐度越高越好，但有上限)
        sharpness_score = min(sharpness / 1000.0, 1.0)
        
        # 噪声分数 (噪声越低越好)
        noise_score = 1.0 - noise_level
        
        # 曝光分数 (过曝和欠曝都会降低分数)
        exposure_score = 1.0 - (overexposure_ratio + underexposure_ratio)
        exposure_score = max(0, exposure_score)
        
        # 反光分数 (反光越少越好)
        reflection_quality_score = 1.0 - reflection_score
        
        # 加权平均
        weights = {
            'brightness': 0.2,
            'contrast': 0.15,
            'sharpness': 0.2,
            'noise': 0.15,
            'exposure': 0.2,
            'reflection': 0.1
        }
        
        quality_score = (
            weights['brightness'] * brightness_score +
            weights['contrast'] * contrast_score +
            weights['sharpness'] * sharpness_score +
            weights['noise'] * noise_score +
            weights['exposure'] * exposure_score +
            weights['reflection'] * reflection_quality_score
        )
        
        return max(0, min(1, quality_score))
    
    def enhance_image(self, image: np.ndarray, 
                     quality_metrics: Optional[ImageQualityMetrics] = None) -> np.ndarray:
        """增强图像质量"""
        try:
            if quality_metrics is None:
                quality_metrics = self.analyze_image_quality(image)
            
            enhanced = image.copy()
            
            # 1. 亮度调整
            enhanced = self._adjust_brightness(enhanced, quality_metrics.brightness)
            
            # 2. 对比度增强
            enhanced = self._enhance_contrast(enhanced, quality_metrics.contrast)
            
            # 3. 反光处理
            if quality_metrics.reflection_score > 0.3:
                enhanced = self._reduce_reflection(enhanced)
            
            # 4. 曝光修正
            if quality_metrics.overexposure_ratio > 0.1 or quality_metrics.underexposure_ratio > 0.1:
                enhanced = self._correct_exposure(enhanced)
            
            # 5. 降噪
            if quality_metrics.noise_level > 0.3:
                enhanced = self._denoise_image(enhanced)
            
            # 6. 锐化
            if quality_metrics.sharpness < 500:
                enhanced = self._sharpen_image(enhanced)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"图像增强失败: {e}")
            return image
    
    def _adjust_brightness(self, image: np.ndarray, current_brightness: float) -> np.ndarray:
        """调整亮度"""
        target_brightness = self.config['brightness_target']
        brightness_diff = target_brightness - current_brightness
        
        # 限制调整幅度
        brightness_diff = np.clip(brightness_diff, -50, 50)
        
        # 应用亮度调整
        adjusted = cv2.convertScaleAbs(image, alpha=1.0, beta=brightness_diff)
        return adjusted
    
    def _enhance_contrast(self, image: np.ndarray, current_contrast: float) -> np.ndarray:
        """增强对比度"""
        # 使用CLAHE (对比度限制自适应直方图均衡)
        if len(image.shape) == 3:
            # 彩色图像：在LAB色彩空间的L通道应用CLAHE
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # 灰度图像：直接应用CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        return enhanced
    
    def _reduce_reflection(self, image: np.ndarray) -> np.ndarray:
        """减少反光"""
        # 使用inpainting技术修复反光区域
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 检测反光区域
        reflection_mask = gray > self.config['reflection_threshold']
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        reflection_mask = cv2.morphologyEx(reflection_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        # 使用inpainting修复
        if len(image.shape) == 3:
            enhanced = cv2.inpaint(image, reflection_mask, 3, cv2.INPAINT_TELEA)
        else:
            enhanced = cv2.inpaint(image, reflection_mask, 3, cv2.INPAINT_TELEA)
        
        return enhanced
    
    def _correct_exposure(self, image: np.ndarray) -> np.ndarray:
        """修正曝光"""
        # 使用Gamma校正
        if len(image.shape) == 3:
            # 转换到HSV，只调整V通道
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2].astype(np.float32) / 255.0
            
            # 计算适当的gamma值
            mean_brightness = np.mean(v_channel)
            if mean_brightness < 0.3:  # 欠曝
                gamma = 0.7
            elif mean_brightness > 0.7:  # 过曝
                gamma = 1.3
            else:
                gamma = 1.0
            
            # 应用gamma校正
            v_corrected = np.power(v_channel, gamma)
            hsv[:, :, 2] = (v_corrected * 255).astype(np.uint8)
            
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            # 灰度图像直接应用gamma校正
            normalized = image.astype(np.float32) / 255.0
            mean_brightness = np.mean(normalized)
            
            if mean_brightness < 0.3:
                gamma = 0.7
            elif mean_brightness > 0.7:
                gamma = 1.3
            else:
                gamma = 1.0
            
            corrected = np.power(normalized, gamma)
            enhanced = (corrected * 255).astype(np.uint8)
        
        return enhanced
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """图像降噪"""
        if len(image.shape) == 3:
            # 彩色图像使用Non-local Means降噪
            denoised = cv2.fastNlMeansDenoisingColored(
                image, None,
                self.config['denoise_strength'],
                self.config['denoise_strength'],
                self.config['denoise_template_window'],
                self.config['denoise_search_window']
            )
        else:
            # 灰度图像降噪
            denoised = cv2.fastNlMeansDenoising(
                image, None,
                self.config['denoise_strength'],
                self.config['denoise_template_window'],
                self.config['denoise_search_window']
            )
        
        return denoised
    
    def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """图像锐化"""
        # 使用Unsharp Mask锐化
        if len(image.shape) == 3:
            # 彩色图像
            blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
            sharpened = cv2.addWeighted(image, 1.0 + self.config['sharpen_strength'], 
                                      blurred, -self.config['sharpen_strength'], 0)
        else:
            # 灰度图像
            blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
            sharpened = cv2.addWeighted(image, 1.0 + self.config['sharpen_strength'], 
                                      blurred, -self.config['sharpen_strength'], 0)
        
        return sharpened
    
    def is_image_acceptable(self, image: np.ndarray) -> Tuple[bool, ImageQualityMetrics]:
        """判断图像质量是否可接受"""
        quality_metrics = self.analyze_image_quality(image)
        is_acceptable = quality_metrics.quality_score >= self.config['min_quality_score']
        
        return is_acceptable, quality_metrics
    
    def get_enhancement_recommendations(self, quality_metrics: ImageQualityMetrics) -> Dict[str, str]:
        """获取图像增强建议"""
        recommendations = {}
        
        if quality_metrics.brightness < 80:
            recommendations['brightness'] = "图像过暗，建议增加光源或调整曝光"
        elif quality_metrics.brightness > 180:
            recommendations['brightness'] = "图像过亮，建议减少光源或调整曝光"
        
        if quality_metrics.contrast < 30:
            recommendations['contrast'] = "对比度过低，建议调整光照条件"
        
        if quality_metrics.reflection_score > 0.3:
            recommendations['reflection'] = "检测到反光，建议调整光源角度或使用偏振滤镜"
        
        if quality_metrics.overexposure_ratio > 0.1:
            recommendations['overexposure'] = "存在过曝区域，建议降低曝光或使用渐变滤镜"
        
        if quality_metrics.underexposure_ratio > 0.1:
            recommendations['underexposure'] = "存在欠曝区域，建议增加补光"
        
        if quality_metrics.noise_level > 0.3:
            recommendations['noise'] = "噪声水平较高，建议降低ISO或改善光照"
        
        if quality_metrics.sharpness < 300:
            recommendations['sharpness'] = "图像模糊，建议检查对焦或减少相机抖动"
        
        return recommendations


class AdaptiveImageProcessor:
    """自适应图像处理器"""
    
    def __init__(self):
        self.enhancer = ImageQualityEnhancer()
        self.logger = logging.getLogger(__name__)
        
    def process_image_stream(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """处理图像流"""
        try:
            # 分析图像质量
            is_acceptable, quality_metrics = self.enhancer.is_image_acceptable(image)
            
            processing_info = {
                'original_quality': quality_metrics.quality_score,
                'is_acceptable': is_acceptable,
                'enhancements_applied': []
            }
            
            if not is_acceptable:
                # 应用图像增强
                enhanced_image = self.enhancer.enhance_image(image, quality_metrics)
                
                # 记录应用的增强
                if quality_metrics.brightness < 80 or quality_metrics.brightness > 180:
                    processing_info['enhancements_applied'].append('brightness_adjustment')
                
                if quality_metrics.contrast < 30:
                    processing_info['enhancements_applied'].append('contrast_enhancement')
                
                if quality_metrics.reflection_score > 0.3:
                    processing_info['enhancements_applied'].append('reflection_reduction')
                
                if quality_metrics.overexposure_ratio > 0.1 or quality_metrics.underexposure_ratio > 0.1:
                    processing_info['enhancements_applied'].append('exposure_correction')
                
                if quality_metrics.noise_level > 0.3:
                    processing_info['enhancements_applied'].append('denoising')
                
                if quality_metrics.sharpness < 300:
                    processing_info['enhancements_applied'].append('sharpening')
                
                # 重新评估增强后的质量
                enhanced_quality = self.enhancer.analyze_image_quality(enhanced_image)
                processing_info['enhanced_quality'] = enhanced_quality.quality_score
                
                return enhanced_image, processing_info
            else:
                return image, processing_info
                
        except Exception as e:
            self.logger.error(f"图像处理失败: {e}")
            return image, {'error': str(e)}


# 使用示例
if __name__ == "__main__":
    # 创建图像质量增强器
    enhancer = ImageQualityEnhancer()
    
    # 模拟测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 分析图像质量
    quality_metrics = enhancer.analyze_image_quality(test_image)
    print(f"图像质量分数: {quality_metrics.quality_score:.2f}")
    
    # 检查是否可接受
    is_acceptable, _ = enhancer.is_image_acceptable(test_image)
    print(f"图像质量可接受: {is_acceptable}")
    
    # 获取增强建议
    recommendations = enhancer.get_enhancement_recommendations(quality_metrics)
    for issue, suggestion in recommendations.items():
        print(f"{issue}: {suggestion}")
    
    # 增强图像
    enhanced_image = enhancer.enhance_image(test_image)
    print("图像增强完成")