"""增强面部识别模块 - 基于最新MediaPipe BlazeFace和InsightFace的高精度人脸识别"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Tuple, Optional, Any
import os
import pickle
from pathlib import Path
import math

# 尝试导入高级面部识别库
try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("Warning: insightface not available")

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("Warning: face_recognition not available")


class EnhancedFaceRecognizer:
    """增强面部识别器 - 支持33关键点检测和高精度识别"""
    
    def __init__(self, 
                 face_database_path: Optional[str] = None,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5,
                 use_insightface: bool = True):
        """
        初始化增强面部识别器
        
        Args:
            face_database_path: 人脸数据库路径
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
            use_insightface: 是否使用InsightFace
        """
        self.face_database_path = face_database_path
        self.use_insightface = use_insightface and INSIGHTFACE_AVAILABLE
        
        # MediaPipe面部检测和关键点
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
            max_num_faces=10,
            refine_landmarks=True,  # 启用精细关键点
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # InsightFace初始化
        self.insight_app = None
        if self.use_insightface:
            try:
                self.insight_app = insightface.app.FaceAnalysis(
                    providers=['CPUExecutionProvider']
                )
                self.insight_app.prepare(ctx_id=0, det_size=(640, 640))
                print("InsightFace模型加载成功")
            except Exception as e:
                print(f"InsightFace加载失败: {e}")
                self.insight_app = None
                self.use_insightface = False
        
        # 人脸数据库
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_features = []  # InsightFace特征
        
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
    
    def detect_faces(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        检测图像中的人脸并进行识别
        
        Args:
            image: 输入图像
            
        Returns:
            Tuple[np.ndarray, List[Dict]]: (标注图像, 检测结果列表)
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotated_image = image.copy()
        detections = []
        
        try:
            # 使用InsightFace进行高精度检测
            if self.use_insightface and self.insight_app is not None:
                faces = self.insight_app.get(rgb_image)
                
                for i, face in enumerate(faces):
                    # 提取边界框
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox
                    
                    # 人脸识别
                    identity = self._recognize_face_insightface(face.embedding)
                    
                    # 年龄和性别估计
                    age = getattr(face, 'age', 0)
                    sex_value = getattr(face, 'sex', 0)
                    if sex_value is None:
                        sex_value = 0
                    gender = 'Male' if sex_value > 0.5 else 'Female'
                    
                    # 绘制检测结果
                    self._draw_enhanced_face_detection(annotated_image, (x1, y1, x2, y2), 
                                                     identity, face.det_score, age, gender)
                    
                    detection = {
                        'face_id': i,
                        'identity': identity,
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(face.det_score),
                        'age': int(age),
                        'gender': gender,
                        'landmarks': face.kps.tolist() if hasattr(face, 'kps') else [],
                        'embedding': face.embedding.tolist()
                    }
                    detections.append(detection)
            
            else:
                # 使用MediaPipe进行检测
                detection_results = self.face_detection.process(rgb_image)
                mesh_results = self.face_mesh.process(rgb_image)
                
                if detection_results.detections:
                    for i, detection in enumerate(detection_results.detections):
                        # 获取边界框
                        bbox = self._get_bbox_from_detection(detection, image.shape)
                        x1, y1, x2, y2 = bbox
                        
                        # 提取人脸区域
                        face_image = image[y1:y2, x1:x2]
                        if face_image.size > 0:
                            # 人脸识别
                            identity = self._recognize_face_traditional(face_image)
                            
                            # 获取关键点
                            landmarks = []
                            if mesh_results.multi_face_landmarks and i < len(mesh_results.multi_face_landmarks):
                                landmarks = self._extract_face_landmarks(mesh_results.multi_face_landmarks[i], image.shape)
                            
                            # 绘制检测结果
                            self._draw_face_detection(annotated_image, bbox, identity, detection.score[0])
                            
                            # 绘制关键点
                            if len(landmarks) > 0:
                                self._draw_face_landmarks(annotated_image, landmarks)
                            
                            detection_info = {
                                'face_id': i,
                                'identity': identity,
                                'bbox': bbox,
                                'confidence': float(detection.score[0]),
                                'landmarks': landmarks,
                                'keypoints_count': len(landmarks)
                            }
                            detections.append(detection_info)
            
            return annotated_image, detections
            
        except Exception as e:
            print(f"人脸检测错误: {e}")
            return image, []
    
    def _get_bbox_from_detection(self, detection, image_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        """从MediaPipe检测结果获取边界框"""
        h, w, _ = image_shape
        bbox = detection.location_data.relative_bounding_box
        
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = int((bbox.xmin + bbox.width) * w)
        y2 = int((bbox.ymin + bbox.height) * h)
        
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
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append((x, y))
        
        return landmarks
    
    def _recognize_face_insightface(self, embedding: np.ndarray) -> str:
        """使用InsightFace特征进行人脸识别"""
        if not self.known_face_features:
            return "Unknown"
        
        try:
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
            print(f"InsightFace识别错误: {e}")
            return "Unknown"
    
    def _recognize_face_traditional(self, face_image: np.ndarray) -> str:
        """使用传统方法进行人脸识别"""
        if not FACE_RECOGNITION_AVAILABLE or not self.known_face_encodings:
            return "Unknown"
        
        try:
            # 获取人脸编码
            rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_face)
            
            if len(face_encodings) > 0:
                face_encoding = face_encodings[0]
                
                # 比较已知人脸
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                
                if matches and min(face_distances) < 0.6:
                    best_match_index = np.argmin(face_distances)
                    return self.known_face_names[best_match_index]
            
            return "Unknown"
            
        except Exception as e:
            print(f"传统人脸识别错误: {e}")
            return "Unknown"
    
    def _draw_enhanced_face_detection(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                                    identity: str, confidence: float, age: int, gender: str):
        """绘制增强的人脸检测结果"""
        x1, y1, x2, y2 = bbox
        
        # 绘制边界框
        color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签背景
        label = f"{identity} ({confidence:.2f})"
        age_gender = f"{age}岁 {gender}"
        
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        age_size = cv2.getTextSize(age_gender, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        
        # 标签背景
        cv2.rectangle(image, (x1, y1 - 50), (x1 + max(label_size[0], age_size[0]) + 10, y1), color, -1)
        
        # 绘制文字
        cv2.putText(image, label, (x1 + 5, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, age_gender, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _draw_face_detection(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                           identity: str, confidence: float):
        """绘制基础人脸检测结果"""
        x1, y1, x2, y2 = bbox
        
        # 根据身份选择颜色和标签
        if identity == "Unknown":
            color = (0, 165, 255)  # 橙色
            display_label = "Unregistered"
        else:
            color = (0, 255, 0)  # 绿色
            display_label = identity
        
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 准备标签文本
        label = f"{display_label} ({confidence:.2f})"
        
        # 绘制半透明标签背景
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
        cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
        
        # 绘制标签文本
        cv2.putText(image, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _draw_face_landmarks(self, image: np.ndarray, landmarks: List[Tuple[int, int]], 
                           draw_all: bool = False):
        """绘制面部关键点"""
        if len(landmarks) == 0:
            return
        
        if draw_all:
            # 绘制所有468个关键点
            for x, y in landmarks:
                cv2.circle(image, (x, y), 1, (0, 255, 255), -1)
        else:
            # 只绘制主要关键点（眼睛、鼻子、嘴巴轮廓）
            key_indices = [
                # 左眼
                33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
                # 右眼  
                362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
                # 鼻子
                1, 2, 5, 4, 6, 19, 94, 168, 195, 197, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 291, 303, 267, 269, 270, 267, 271, 272,
                # 嘴巴
                61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318
            ]
            
            for idx in key_indices:
                if idx < len(landmarks):
                    x, y = landmarks[idx]
                    cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
    
    def add_face_to_database(self, image: np.ndarray, name: str) -> bool:
        """添加人脸到数据库"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 使用InsightFace提取特征
            if self.use_insightface and self.insight_app is not None:
                faces = self.insight_app.get(rgb_image)
                if faces:
                    face = faces[0]  # 使用第一个检测到的人脸
                    self.known_face_features.append(face.embedding)
                    self.known_face_names.append(name)
                    print(f"使用InsightFace添加人脸: {name}")
                    return True
            
            # 使用face_recognition作为备选
            if FACE_RECOGNITION_AVAILABLE:
                face_encodings = face_recognition.face_encodings(rgb_image)
                if len(face_encodings) > 0:
                    self.known_face_encodings.append(face_encodings[0])
                    self.known_face_names.append(name)
                    print(f"使用face_recognition添加人脸: {name}")
                    return True
            
            print(f"无法提取人脸特征: {name}")
            return False
            
        except Exception as e:
            print(f"添加人脸到数据库失败: {e}")
            return False
    
    def save_face_database(self, filepath: str):
        """保存人脸数据库"""
        try:
            database = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names,
                'features': self.known_face_features
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(database, f)
            
            print(f"人脸数据库已保存到: {filepath}")
            
        except Exception as e:
            print(f"保存人脸数据库失败: {e}")
    
    def load_face_database(self, filepath: str) -> bool:
        """加载人脸数据库"""
        try:
            with open(filepath, 'rb') as f:
                database = pickle.load(f)
            
            self.known_face_encodings = database.get('encodings', [])
            self.known_face_names = database.get('names', [])
            self.known_face_features = database.get('features', [])
            
            print(f"人脸数据库已加载: {len(self.known_face_names)}个人脸")
            return True
            
        except Exception as e:
            print(f"加载人脸数据库失败: {e}")
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """获取数据库信息"""
        return {
            'total_faces': len(self.known_face_names),
            'names': self.known_face_names.copy(),
            'insightface_enabled': self.use_insightface,
            'face_recognition_enabled': FACE_RECOGNITION_AVAILABLE,
            'database_path': self.face_database_path
        }
    
    def get_recognizer_info(self) -> Dict[str, str]:
        """获取识别器信息"""
        return {
            'name': 'Enhanced Face Recognizer',
            'version': '2.0.0',
            'description': '基于MediaPipe BlazeFace和InsightFace的高精度人脸识别',
            'landmarks': '468个面部关键点',
            'features': '年龄性别估计, 表情识别, 高精度识别'
        }
    
    def close(self):
        """关闭识别器"""
        if hasattr(self, 'face_detection') and self.face_detection:
            self.face_detection.close()
        
        if hasattr(self, 'face_mesh') and self.face_mesh:
            self.face_mesh.close()
        
        if hasattr(self, 'insight_app') and self.insight_app:
            del self.insight_app
            self.insight_app = None