"""优化版面部识别模块 - 解决数据乱码问题，提升性能和准确率"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Tuple, Optional, Any, Union
import os
import pickle
import json
from pathlib import Path
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import logging
from dataclasses import dataclass

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 尝试导入高级面部识别库
try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logger.warning("InsightFace not available")

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logger.warning("face_recognition not available")

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    logger.warning("dlib not available")


@dataclass
class FaceDetectionResult:
    """面部检测结果数据类"""
    face_id: int
    identity: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    landmarks: List[Tuple[int, int]]
    age: Optional[int] = None
    gender: Optional[str] = None
    emotion: Optional[str] = None
    embedding: Optional[List[float]] = None
    quality_score: Optional[float] = None


class OptimizedFaceRecognizer:
    """优化版面部识别器 - 高性能、高准确率、防乱码"""
    
    def __init__(self, 
                 face_database_path: Optional[str] = None,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5,
                 use_insightface: bool = True,
                 max_faces: int = 10,
                 enable_threading: bool = True,
                 encoding: str = 'utf-8'):
        """
        初始化优化版面部识别器
        
        Args:
            face_database_path: 人脸数据库路径
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
            use_insightface: 是否使用InsightFace
            max_faces: 最大检测人脸数
            enable_threading: 是否启用多线程
            encoding: 字符编码格式
        """
        self.face_database_path = face_database_path
        self.use_insightface = use_insightface and INSIGHTFACE_AVAILABLE
        self.max_faces = max_faces
        self.enable_threading = enable_threading
        self.encoding = encoding
        
        # 性能统计
        self.performance_stats = {
            'total_detections': 0,
            'avg_processing_time': 0.0,
            'successful_recognitions': 0,
            'failed_recognitions': 0
        }
        
        # 线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=4) if enable_threading else None
        
        # MediaPipe初始化
        self._init_mediapipe(min_detection_confidence, min_tracking_confidence)
        
        # InsightFace初始化
        self._init_insightface()
        
        # 人脸数据库
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_features = []  # InsightFace特征
        self.face_database_lock = threading.Lock()
        
        # 加载现有数据库
        if face_database_path and os.path.exists(face_database_path):
            self.load_face_database(face_database_path)
        
        # 表情识别映射
        self.emotion_names = {
            'angry': '愤怒',
            'disgust': '厌恶', 
            'fear': '恐惧',
            'happy': '高兴',
            'sad': '悲伤',
            'surprise': '惊讶',
            'neutral': '中性'
        }
        
        # 质量评估阈值
        self.quality_thresholds = {
            'min_face_size': 50,
            'max_blur_variance': 100,
            'min_brightness': 50,
            'max_brightness': 200
        }
        
        logger.info(f"优化版面部识别器初始化完成 - InsightFace: {self.use_insightface}")
    
    def _init_mediapipe(self, min_detection_confidence: float, min_tracking_confidence: float):
        """初始化MediaPipe组件"""
        try:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            # 使用最新的BlazeFace模型
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,  # 1为长距离高精度模型
                min_detection_confidence=min_detection_confidence
            )
            
            # 面部网格检测 - 468个关键点
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=self.max_faces,
                refine_landmarks=True,  # 启用精细关键点
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            
            logger.info("MediaPipe初始化成功")
        except Exception as e:
            logger.error(f"MediaPipe初始化失败: {e}")
            raise
    
    def _init_insightface(self):
        """初始化InsightFace"""
        self.insight_app = None
        if self.use_insightface:
            try:
                # 检查module文件夹中的buffalo_l.zip文件
                local_model_path = os.path.join(os.getcwd(), 'module', 'buffalo_l.zip')
                if os.path.exists(local_model_path):
                    logger.info(f"发现本地模型文件: {local_model_path}")
                    # 设置InsightFace使用本地模型目录
                    model_dir = os.path.join(os.getcwd(), 'module', 'buffalo_l')
                    os.makedirs(os.path.dirname(model_dir), exist_ok=True)
                    
                    # 如果模型目录不存在，解压本地zip文件
                    if not os.path.exists(model_dir):
                        import zipfile
                        logger.info("正在解压本地模型文件...")
                        with zipfile.ZipFile(local_model_path, 'r') as zip_ref:
                            zip_ref.extractall(os.path.join(os.getcwd(), 'module'))
                        logger.info("模型文件解压完成")
                else:
                    # 设置国内镜像环境变量作为备选
                    os.environ['INSIGHTFACE_MODEL_URL'] = 'https://mirror.ghproxy.com/https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip'
                
                logger.info("正在初始化InsightFace模型...")
                # 如果本地模型存在，直接指定模型路径
                if os.path.exists(model_dir):
                    self.insight_app = insightface.app.FaceAnalysis(
                        name='buffalo_l',
                        root=os.path.join(os.getcwd(), 'module'),
                        providers=['CPUExecutionProvider'],
                        allowed_modules=['detection', 'recognition']
                    )
                else:
                    # 使用默认配置，会从网络下载
                    self.insight_app = insightface.app.FaceAnalysis(
                        name='buffalo_l',
                        providers=['CPUExecutionProvider'],
                        allowed_modules=['detection', 'recognition']
                    )
                
                logger.info("正在准备InsightFace模型...")
                self.insight_app.prepare(ctx_id=0, det_size=(640, 640))
                logger.info("InsightFace模型加载成功")
                
            except Exception as e:
                logger.error(f"InsightFace加载失败: {e}")
                logger.info("将使用MediaPipe作为备选方案")
                self.insight_app = None
                self.use_insightface = False
    
    def detect_faces(self, image: np.ndarray) -> Tuple[np.ndarray, List[FaceDetectionResult]]:
        """
        检测图像中的人脸并进行识别
        
        Args:
            image: 输入图像
            
        Returns:
            Tuple[np.ndarray, List[FaceDetectionResult]]: (标注图像, 检测结果列表)
        """
        start_time = time.time()
        
        # 输入验证和预处理
        if image is None or image.size == 0:
            logger.warning("输入图像为空")
            return image, []
        
        # 确保图像编码正确
        image = self._ensure_proper_encoding(image)
        
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            annotated_image = image.copy()
            detections = []
            
            # 使用InsightFace进行高精度检测
            if self.use_insightface and self.insight_app is not None:
                detections = self._detect_with_insightface(rgb_image, annotated_image)
            else:
                # 使用MediaPipe进行检测
                detections = self._detect_with_mediapipe(rgb_image, annotated_image)
            
            # 更新性能统计
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time, len(detections))
            
            return annotated_image, detections
            
        except Exception as e:
            logger.error(f"人脸检测错误: {e}")
            return image, []
    
    def _ensure_proper_encoding(self, image: np.ndarray) -> np.ndarray:
        """确保图像编码正确，防止乱码"""
        try:
            # 检查图像数据类型
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            
            # 检查图像范围
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            
            # 确保图像连续性
            if not image.flags['C_CONTIGUOUS']:
                image = np.ascontiguousarray(image)
            
            return image
        except Exception as e:
            logger.error(f"图像编码处理错误: {e}")
            return image
    
    def _detect_with_insightface(self, rgb_image: np.ndarray, annotated_image: np.ndarray) -> List[FaceDetectionResult]:
        """使用InsightFace进行检测"""
        detections = []
        
        try:
            faces = self.insight_app.get(rgb_image)
            
            for i, face in enumerate(faces):
                if i >= self.max_faces:
                    break
                
                # 提取边界框
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                
                # 质量评估
                quality_score = self._assess_face_quality(rgb_image[y1:y2, x1:x2])
                if quality_score is None:
                    quality_score = 0.0
                if quality_score < 0.5:
                    continue
                
                # 人脸识别
                identity = self._recognize_face_insightface(face.embedding)
                
                # 年龄和性别估计
                age = getattr(face, 'age', None)
                sex_value = getattr(face, 'sex', 0)
                if sex_value is None:
                    sex_value = 0
                gender = 'Male' if sex_value > 0.5 else 'Female'
                
                # 表情识别
                emotion = self._detect_emotion(face)
                
                # 绘制检测结果
                self._draw_enhanced_face_detection(
                    annotated_image, (x1, y1, x2, y2), 
                    identity, face.det_score, age, gender, emotion
                )
                
                detection = FaceDetectionResult(
                    face_id=i,
                    identity=self._safe_encode_string(identity),
                    bbox=(x1, y1, x2, y2),
                    confidence=float(face.det_score.item() if hasattr(face.det_score, 'item') else face.det_score),
                    age=int(age.item() if hasattr(age, 'item') else age) if age is not None else None,
                    gender=gender,
                    emotion=emotion,
                    landmarks=face.kps.tolist() if hasattr(face, 'kps') else [],
                    embedding=face.embedding.tolist(),
                    quality_score=quality_score
                )
                detections.append(detection)
                
        except Exception as e:
            logger.error(f"InsightFace检测错误: {e}")
        
        return detections
    
    def _detect_with_mediapipe(self, rgb_image: np.ndarray, annotated_image: np.ndarray) -> List[FaceDetectionResult]:
        """使用MediaPipe进行检测"""
        detections = []
        
        try:
            detection_results = self.face_detection.process(rgb_image)
            mesh_results = self.face_mesh.process(rgb_image)
            
            if detection_results.detections:
                for i, detection in enumerate(detection_results.detections):
                    if i >= self.max_faces:
                        break
                    
                    # 获取边界框
                    bbox = self._get_bbox_from_detection(detection, rgb_image.shape)
                    x1, y1, x2, y2 = bbox
                    
                    # 提取人脸区域
                    face_image = rgb_image[y1:y2, x1:x2]
                    if face_image.size == 0:
                        continue
                    
                    # 质量评估
                    quality_score = self._assess_face_quality(face_image)
                    if quality_score < 0.3:
                        continue
                    
                    # 人脸识别
                    identity = self._recognize_face_traditional(face_image)
                    
                    # 获取关键点
                    landmarks = []
                    if mesh_results.multi_face_landmarks and i < len(mesh_results.multi_face_landmarks):
                        landmarks = self._extract_face_landmarks(
                            mesh_results.multi_face_landmarks[i], rgb_image.shape
                        )
                    
                    # 绘制检测结果
                    self._draw_face_detection(annotated_image, bbox, identity, detection.score[0])
                    
                    # 绘制关键点
                    if len(landmarks) > 0:
                        self._draw_face_landmarks(annotated_image, landmarks)
                    
                    detection_info = FaceDetectionResult(
                        face_id=i,
                        identity=self._safe_encode_string(identity),
                        bbox=bbox,
                        confidence=float(detection.score[0].item() if hasattr(detection.score[0], 'item') else detection.score[0]),
                        landmarks=landmarks,
                        quality_score=quality_score
                    )
                    detections.append(detection_info)
                    
        except Exception as e:
            logger.error(f"MediaPipe检测错误: {e}")
        
        return detections
    
    def _safe_encode_string(self, text: str) -> str:
        """安全编码字符串，防止乱码"""
        try:
            if isinstance(text, bytes):
                return text.decode(self.encoding, errors='replace')
            elif isinstance(text, str):
                return text.encode(self.encoding, errors='replace').decode(self.encoding)
            else:
                return str(text)
        except Exception as e:
            logger.warning(f"字符串编码错误: {e}")
            return "Unknown"
    
    def _assess_face_quality(self, face_image: np.ndarray) -> float:
        """评估人脸图像质量"""
        try:
            if face_image.size == 0:
                return 0.0
            
            h, w = face_image.shape[:2]
            
            # 尺寸检查
            size_score = min(w, h) / self.quality_thresholds['min_face_size']
            size_score = min(1.0, size_score)
            
            # 模糊度检查
            gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY) if len(face_image.shape) == 3 else face_image
            blur_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_score = min(1.0, blur_variance / self.quality_thresholds['max_blur_variance'])
            
            # 亮度检查
            brightness = np.mean(gray)
            if brightness < self.quality_thresholds['min_brightness']:
                brightness_score = brightness / self.quality_thresholds['min_brightness']
            elif brightness > self.quality_thresholds['max_brightness']:
                brightness_score = self.quality_thresholds['max_brightness'] / brightness
            else:
                brightness_score = 1.0
            
            # 综合质量分数
            quality_score = (size_score * 0.3 + blur_score * 0.5 + brightness_score * 0.2)
            return min(1.0, quality_score)
            
        except Exception as e:
            logger.error(f"质量评估错误: {e}")
            return 0.5
    
    def _detect_emotion(self, face) -> Optional[str]:
        """检测表情（如果可用）"""
        try:
            if hasattr(face, 'emotion'):
                emotion_scores = face.emotion
                if emotion_scores is not None and len(emotion_scores) >= 7:
                    emotion_idx = np.argmax(emotion_scores)
                    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
                    return self.emotion_names.get(emotions[emotion_idx], 'neutral')
            return None
        except Exception as e:
            logger.error(f"表情检测错误: {e}")
            return None
    
    def _update_performance_stats(self, processing_time: float, num_faces: int):
        """更新性能统计"""
        self.performance_stats['total_detections'] += 1
        
        # 计算平均处理时间
        total_time = (self.performance_stats['avg_processing_time'] * 
                     (self.performance_stats['total_detections'] - 1) + processing_time)
        self.performance_stats['avg_processing_time'] = total_time / self.performance_stats['total_detections']
        
        if num_faces > 0:
            self.performance_stats['successful_recognitions'] += 1
        else:
            self.performance_stats['failed_recognitions'] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        stats = self.performance_stats.copy()
        stats['success_rate'] = (
            stats['successful_recognitions'] / max(1, stats['total_detections'])
        )
        return stats
    
    def optimize_for_realtime(self):
        """优化实时性能"""
        try:
            # 降低检测置信度以提高速度
            if hasattr(self, 'face_detection'):
                self.face_detection = self.mp_face_detection.FaceDetection(
                    model_selection=0,  # 使用短距离快速模型
                    min_detection_confidence=0.5
                )
            
            # 减少最大检测人脸数
            self.max_faces = min(5, self.max_faces)
            
            logger.info("实时性能优化完成")
        except Exception as e:
            logger.error(f"性能优化失败: {e}")
    
    # 继承原有的其他方法...
    def _get_bbox_from_detection(self, detection, image_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        """从MediaPipe检测结果获取边界框"""
        h, w, _ = image_shape
        bbox = detection.location_data.relative_bounding_box
        
        # 确保标量转换，避免数组转换错误
        x1_val = bbox.xmin * w
        y1_val = bbox.ymin * h
        x2_val = (bbox.xmin + bbox.width) * w
        y2_val = (bbox.ymin + bbox.height) * h
        
        x1 = int(x1_val.item() if hasattr(x1_val, 'item') else x1_val)
        y1 = int(y1_val.item() if hasattr(y1_val, 'item') else y1_val)
        x2 = int(x2_val.item() if hasattr(x2_val, 'item') else x2_val)
        y2 = int(y2_val.item() if hasattr(y2_val, 'item') else y2_val)
        
        # 确保边界框在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        return (x1, y1, x2, y2)
    
    def _extract_face_landmarks(self, face_landmarks, image_shape: Tuple[int, int, int]) -> List[Tuple[int, int]]:
        """提取面部关键点坐标"""
        h, w, _ = image_shape
        landmarks = []
        
        for landmark in face_landmarks.landmark:
            # 确保标量转换，避免数组转换错误
            x_val = landmark.x * w
            y_val = landmark.y * h
            x = int(x_val.item() if hasattr(x_val, 'item') else x_val)
            y = int(y_val.item() if hasattr(y_val, 'item') else y_val)
            landmarks.append((x, y))
        
        return landmarks
    
    def _recognize_face_insightface(self, embedding: np.ndarray) -> str:
        """使用InsightFace特征进行人脸识别"""
        if not self.known_face_features:
            return "Unknown"
        
        try:
            with self.face_database_lock:
                # 计算余弦相似度
                similarities = []
                for known_embedding in self.known_face_features:
                    similarity = np.dot(embedding, known_embedding) / (
                        np.linalg.norm(embedding) * np.linalg.norm(known_embedding)
                    )
                    similarities.append(similarity)
                
                # 找到最高相似度
                max_similarity = max(similarities)
                if max_similarity > 0.6:  # 相似度阈值
                    best_match_index = similarities.index(max_similarity)
                    return self.known_face_names[best_match_index]
                
                return "Unknown"
                
        except Exception as e:
            logger.error(f"InsightFace识别错误: {e}")
            return "Unknown"
    
    def _recognize_face_traditional(self, face_image: np.ndarray) -> str:
        """使用传统方法进行人脸识别"""
        if not self.known_face_encodings or not FACE_RECOGNITION_AVAILABLE:
            return "Unknown"
        
        try:
            # 转换为RGB格式
            if len(face_image.shape) == 3:
                face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                face_image_rgb = face_image
            
            # 获取人脸编码
            face_encodings = face_recognition.face_encodings(face_image_rgb)
            
            if len(face_encodings) > 0:
                face_encoding = face_encodings[0]
                
                # 比较已知人脸
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, face_encoding, tolerance=0.6
                )
                
                if True in matches:
                    match_index = matches.index(True)
                    return self.known_face_names[match_index]
            
            return "Unknown"
            
        except Exception as e:
            logger.error(f"传统人脸识别错误: {e}")
            return "Unknown"
    
    def _draw_enhanced_face_detection(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                                    identity: str, confidence: float, age: Optional[int], 
                                    gender: str, emotion: Optional[str]):
        """绘制增强的人脸检测结果"""
        x1, y1, x2, y2 = bbox
        
        # 绘制边界框
        color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 准备标签文本
        label_parts = [identity]
        if age is not None:
            label_parts.append(f"{age}岁")
        if gender:
            label_parts.append(gender)
        if emotion:
            label_parts.append(emotion)
        
        label = " | ".join(label_parts)
        confidence_text = f"({confidence:.2f})"
        
        # 绘制标签背景
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(image, (x1, y1 - 30), (x1 + label_size[0] + 10, y1), color, -1)
        
        # 绘制文本
        cv2.putText(image, label, (x1 + 5, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, confidence_text, (x1 + 5, y1 - 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _draw_face_detection(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                           identity: str, confidence: float):
        """绘制基础人脸检测结果"""
        x1, y1, x2, y2 = bbox
        
        # 绘制边界框
        color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签
        label = f"{identity} ({confidence:.2f})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(image, (x1, y1 - 25), (x1 + label_size[0] + 10, y1), color, -1)
        cv2.putText(image, label, (x1 + 5, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _draw_face_landmarks(self, image: np.ndarray, landmarks: List[Tuple[int, int]], 
                           draw_all: bool = False):
        """绘制面部关键点"""
        if len(landmarks) == 0:
            return
        
        # 只绘制重要的关键点（眼睛、鼻子、嘴巴）
        important_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        for i, (x, y) in enumerate(landmarks):
            if draw_all or i in important_indices:
                cv2.circle(image, (x, y), 1, (0, 255, 255), -1)
    
    def close(self):
        """清理资源"""
        try:
            if self.thread_pool:
                self.thread_pool.shutdown(wait=True)
            
            if hasattr(self, 'face_detection'):
                self.face_detection.close()
            
            if hasattr(self, 'face_mesh'):
                self.face_mesh.close()
            
            if self.insight_app:
                self.insight_app = None
            
            logger.info("优化版面部识别器资源清理完成")
        except Exception as e:
            logger.error(f"资源清理错误: {e}")


# 工厂函数
def create_optimized_face_recognizer(**kwargs) -> OptimizedFaceRecognizer:
    """创建优化版面部识别器"""
    return OptimizedFaceRecognizer(**kwargs)