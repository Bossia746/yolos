"""面部识别模块 - 基于MediaPipe和face_recognition的人脸检测和身份识别"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Tuple, Optional, Any
import os
import pickle
from pathlib import Path

# 尝试导入face_recognition，如果失败则使用MediaPipe替代
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("Warning: face_recognition not available, using MediaPipe only for face detection")


class FaceRecognizer:
    """面部识别器"""
    
    def __init__(self, 
                 face_database_path: Optional[str] = None,
                 min_detection_confidence: float = 0.7,
                 model: str = 'hog'):
        """
        初始化面部识别器
        
        Args:
            face_database_path: 人脸数据库路径
            min_detection_confidence: 最小检测置信度
            model: 人脸检测模型 ('hog' 或 'cnn')
        """
        self.face_database_path = face_database_path
        self.model = model
        self.face_recognition_available = FACE_RECOGNITION_AVAILABLE
        
        # MediaPipe人脸检测
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0为短距离模型，1为长距离模型
            min_detection_confidence=min_detection_confidence
        )
        
        # 人脸数据库
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_database = {}
        
        # 加载人脸数据库
        if face_database_path and os.path.exists(face_database_path):
            self.load_face_database(face_database_path)
            
        if not self.face_recognition_available:
            print("Warning: Face recognition features disabled due to missing dependencies")
    
    def detect_faces(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        检测人脸并进行身份识别
        
        Args:
            image: 输入图像
            
        Returns:
            annotated_image: 标注后的图像
            results: 检测结果列表
        """
        # 使用MediaPipe检测人脸
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_results = self.face_detection.process(rgb_image)
        
        annotated_image = image.copy()
        detection_results = []
        
        if mp_results.detections:
            for idx, detection in enumerate(mp_results.detections):
                # 获取边界框
                bbox = self._get_bbox_from_detection(detection, image.shape)
                
                # 提取人脸区域进行识别
                face_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                
                # 进行人脸识别
                identity = self._recognize_face(face_image)
                
                # 获取置信度
                confidence = detection.score[0]
                
                # 绘制检测结果
                self._draw_face_detection(annotated_image, bbox, identity, confidence)
                
                # 保存检测结果
                detection_results.append({
                    'face_id': idx,
                    'bbox': bbox,
                    'identity': identity,
                    'confidence': confidence,
                    'landmarks': self._get_face_landmarks(face_image) if face_image.size > 0 else None
                })
        
        return annotated_image, detection_results
    
    def _get_bbox_from_detection(self, detection, image_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        """从MediaPipe检测结果获取边界框"""
        h, w = image_shape[:2]
        bbox = detection.location_data.relative_bounding_box
        
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        # 确保边界框在图像范围内
        x = max(0, x)
        y = max(0, y)
        width = min(width, w - x)
        height = min(height, h - y)
        
        return (x, y, x + width, y + height)
    
    def _recognize_face(self, face_image: np.ndarray) -> str:
        """识别人脸身份"""
        if not self.face_recognition_available:
            return "Unknown"
            
        if face_image.size == 0 or len(self.known_face_encodings) == 0:
            return "Unknown"
        
        try:
            # 使用face_recognition库进行人脸编码
            rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_face, model='small')
            
            if len(face_encodings) == 0:
                return "Unknown"
            
            face_encoding = face_encodings[0]
            
            # 与已知人脸进行比较
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    return self.known_face_names[best_match_index]
            
            return "Unknown"
        
        except Exception as e:
            print(f"人脸识别错误: {e}")
            return "Unknown"
    
    def _get_face_landmarks(self, face_image: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """获取人脸关键点"""
        if not self.face_recognition_available:
            return None
            
        try:
            rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_landmarks_list = face_recognition.face_landmarks(rgb_face)
            
            if len(face_landmarks_list) > 0:
                landmarks = []
                for landmark_dict in face_landmarks_list[0].values():
                    landmarks.extend(landmark_dict)
                return landmarks
            
            return None
        
        except Exception:
            return None
    
    def _draw_face_detection(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                           identity: str, confidence: float):
        """在图像上绘制人脸检测结果"""
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
    
    def add_face_to_database(self, image: np.ndarray, name: str) -> bool:
        """添加人脸到数据库"""
        if not self.face_recognition_available:
            print("人脸识别功能不可用，无法添加人脸到数据库")
            return False
            
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_image)
            
            if len(face_encodings) > 0:
                face_encoding = face_encodings[0]
                
                # 检查是否已存在该人脸
                if name in self.known_face_names:
                    # 更新现有编码
                    index = self.known_face_names.index(name)
                    self.known_face_encodings[index] = face_encoding
                else:
                    # 添加新人脸
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_names.append(name)
                
                # 更新数据库字典
                self.face_database[name] = face_encoding
                
                print(f"成功添加人脸: {name}")
                return True
            else:
                print(f"未在图像中检测到人脸: {name}")
                return False
        
        except Exception as e:
            print(f"添加人脸到数据库时出错: {e}")
            return False
    
    def remove_face_from_database(self, name: str) -> bool:
        """从数据库中移除人脸"""
        try:
            if name in self.known_face_names:
                index = self.known_face_names.index(name)
                self.known_face_encodings.pop(index)
                self.known_face_names.pop(index)
                
                if name in self.face_database:
                    del self.face_database[name]
                
                print(f"成功移除人脸: {name}")
                return True
            else:
                print(f"数据库中未找到人脸: {name}")
                return False
        
        except Exception as e:
            print(f"移除人脸时出错: {e}")
            return False
    
    def save_face_database(self, filepath: str):
        """保存人脸数据库"""
        try:
            database_data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names,
                'database': self.face_database
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(database_data, f)
            
            print(f"人脸数据库已保存到: {filepath}")
        
        except Exception as e:
            print(f"保存人脸数据库时出错: {e}")
    
    def load_face_database(self, filepath: str) -> bool:
        """加载人脸数据库"""
        try:
            with open(filepath, 'rb') as f:
                database_data = pickle.load(f)
            
            self.known_face_encodings = database_data.get('encodings', [])
            self.known_face_names = database_data.get('names', [])
            self.face_database = database_data.get('database', {})
            
            print(f"成功加载人脸数据库: {len(self.known_face_names)} 个人脸")
            return True
        
        except Exception as e:
            print(f"加载人脸数据库时出错: {e}")
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """获取数据库信息"""
        return {
            'total_faces': len(self.known_face_names),
            'face_names': self.known_face_names.copy(),
            'database_path': self.face_database_path
        }
    
    def close(self):
        """释放资源"""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()