"""多模态识别检测器 - 整合手势识别、面部识别和身体姿势识别功能"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import time
import threading
from queue import Queue

from .gesture_recognizer import GestureRecognizer
from .face_recognizer import FaceRecognizer
from .pose_recognizer import PoseRecognizer
from .enhanced_gesture_recognizer import EnhancedGestureRecognizer
from .enhanced_face_recognizer import EnhancedFaceRecognizer
from .yolov7_pose_recognizer import YOLOv7PoseRecognizer
from .fall_detector import FallDetector


class MultimodalDetector:
    """多模态识别检测器 - 集成最新的视觉算法"""
    
    def __init__(self, 
                 enable_gesture: bool = True,
                 enable_face: bool = True,
                 enable_pose: bool = True,
                 enable_fall_detection: bool = True,
                 use_enhanced_algorithms: bool = True,
                 face_database_path: Optional[str] = None,
                 detection_interval: int = 1):
        """
        初始化多模态识别检测器
        
        Args:
            enable_gesture: 启用手势识别
            enable_face: 启用面部识别
            enable_pose: 启用身体姿势识别
            enable_fall_detection: 启用摔倒检测
            use_enhanced_algorithms: 使用增强算法（YOLOv7、MediaPipe BlazeFace等）
            face_database_path: 人脸数据库路径
            detection_interval: 检测间隔（帧数）
        """
        self.enable_gesture = enable_gesture
        self.enable_face = enable_face
        self.enable_pose = enable_pose
        self.enable_fall_detection = enable_fall_detection
        self.use_enhanced_algorithms = use_enhanced_algorithms
        self.detection_interval = detection_interval
        self.frame_count = 0
        
        # 初始化各个识别器
        self.gesture_recognizer = None
        self.face_recognizer = None
        self.pose_recognizer = None
        self.fall_detector = None
        
        # 选择算法版本
        if enable_gesture:
            if use_enhanced_algorithms:
                try:
                    self.gesture_recognizer = EnhancedGestureRecognizer()
                    print("使用增强手势识别算法")
                except Exception as e:
                    print(f"增强手势识别初始化失败，使用基础版本: {e}")
                    self.gesture_recognizer = GestureRecognizer()
            else:
                self.gesture_recognizer = GestureRecognizer()
        
        if enable_face:
            if use_enhanced_algorithms:
                try:
                    self.face_recognizer = EnhancedFaceRecognizer(face_database_path=face_database_path)
                    print("使用增强面部识别算法")
                except Exception as e:
                    print(f"增强面部识别初始化失败，使用基础版本: {e}")
                    self.face_recognizer = FaceRecognizer(face_database_path=face_database_path)
            else:
                self.face_recognizer = FaceRecognizer(face_database_path=face_database_path)
        
        if enable_pose:
            if use_enhanced_algorithms:
                try:
                    self.pose_recognizer = YOLOv7PoseRecognizer()
                    print("使用YOLOv7姿态识别算法")
                except Exception as e:
                    print(f"YOLOv7姿态识别初始化失败，使用基础版本: {e}")
                    self.pose_recognizer = PoseRecognizer()
            else:
                self.pose_recognizer = PoseRecognizer()
        
        if enable_fall_detection:
            try:
                self.fall_detector = FallDetector()
                print("摔倒检测器初始化成功")
            except Exception as e:
                print(f"摔倒检测器初始化失败: {e}")
                self.fall_detector = None
        
        # 回调函数
        self.detection_callback = None
        self.frame_callback = None
        
        # 统计信息
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'gesture_detections': 0,
            'face_detections': 0,
            'pose_detections': 0,
            'fall_detections': 0,
            'processing_time': 0.0,
            'fps': 0.0
        }
        
        self.start_time = time.time()
    
    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        执行多模态识别检测
        
        Args:
            image: 输入图像
            
        Returns:
            annotated_image: 标注后的图像
            results: 检测结果字典
        """
        start_time = time.time()
        self.frame_count += 1
        self.stats['total_frames'] += 1
        
        # 初始化结果
        results = {
            'timestamp': time.time(),
            'frame_id': self.frame_count,
            'gesture_results': [],
            'face_results': [],
            'pose_results': {},
            'fall_results': {},
            'processing_time': 0.0
        }
        
        annotated_image = image.copy()
        
        # 根据检测间隔决定是否进行检测
        if self.frame_count % self.detection_interval == 0:
            self.stats['processed_frames'] += 1
            
            # 手势识别
            if self.enable_gesture and self.gesture_recognizer:
                try:
                    if isinstance(self.gesture_recognizer, EnhancedGestureRecognizer):
                        # 使用增强手势识别
                        gesture_results = self.gesture_recognizer.recognize_gestures(annotated_image)
                        annotated_image = self.gesture_recognizer.draw_annotations(annotated_image, gesture_results)
                        results['gesture_results'] = gesture_results
                        if gesture_results.get('hands_detected', 0) > 0:
                            self.stats['gesture_detections'] += gesture_results['hands_detected']
                    else:
                        # 使用基础手势识别
                        gesture_image, gesture_results = self.gesture_recognizer.detect_hands(annotated_image)
                        annotated_image = gesture_image
                        results['gesture_results'] = gesture_results
                        if gesture_results:
                            # 检查gesture_results的类型
                            if hasattr(gesture_results, 'hands_detected'):
                                # GestureRecognitionResult对象
                                self.stats['gesture_detections'] += gesture_results.hands_detected
                            elif isinstance(gesture_results, (list, tuple)):
                                # 列表或元组
                                self.stats['gesture_detections'] += len(gesture_results)
                            else:
                                # 其他类型，尝试转换为整数
                                try:
                                    self.stats['gesture_detections'] += int(gesture_results)
                                except (ValueError, TypeError):
                                    self.stats['gesture_detections'] += 1
                except Exception as e:
                    print(f"手势识别错误: {e}")
                    results['gesture_results'] = []
            
            # 面部识别
            if self.enable_face and self.face_recognizer:
                try:
                    if isinstance(self.face_recognizer, EnhancedFaceRecognizer):
                        # 使用增强面部识别
                        face_results = self.face_recognizer.detect_and_recognize_faces(annotated_image)
                        annotated_image = self.face_recognizer.draw_annotations(annotated_image, face_results)
                        results['face_results'] = face_results
                        if face_results.get('faces_detected', 0) > 0:
                            self.stats['face_detections'] += face_results['faces_detected']
                    else:
                        # 使用基础面部识别
                        face_image, face_results = self.face_recognizer.detect_faces(annotated_image)
                        annotated_image = face_image
                        results['face_results'] = face_results
                        if face_results:
                            self.stats['face_detections'] += len(face_results)
                except Exception as e:
                    print(f"面部识别错误: {e}")
                    results['face_results'] = []
            
            # 身体姿势识别
            if self.enable_pose and self.pose_recognizer:
                try:
                    if isinstance(self.pose_recognizer, YOLOv7PoseRecognizer):
                        # 使用YOLOv7姿态识别
                        pose_results = self.pose_recognizer.detect_pose(annotated_image)
                        annotated_image = self.pose_recognizer.draw_annotations(annotated_image, pose_results)
                        results['pose_results'] = pose_results
                        if pose_results.get('persons_detected', 0) > 0:
                            self.stats['pose_detections'] += pose_results['persons_detected']
                    else:
                        # 使用基础姿态识别
                        pose_image, pose_results = self.pose_recognizer.detect_pose(annotated_image)
                        annotated_image = pose_image
                        results['pose_results'] = pose_results
                        if pose_results.get('pose_detected', False):
                            self.stats['pose_detections'] += 1
                except Exception as e:
                    print(f"身体姿势识别错误: {e}")
                    results['pose_results'] = {}
            
            # 摔倒检测
            if self.enable_fall_detection and self.fall_detector and results['pose_results']:
                try:
                    # 从姿态结果中提取关键点
                    pose_landmarks = None
                    bbox = None
                    
                    if isinstance(self.pose_recognizer, YOLOv7PoseRecognizer):
                        # YOLOv7格式
                        if results['pose_results'].get('persons_detected', 0) > 0:
                            person_data = results['pose_results']['persons'][0]  # 检测第一个人
                            pose_landmarks = person_data.get('keypoints', [])
                            bbox = person_data.get('bbox')
                    else:
                        # 基础格式
                        if results['pose_results'].get('pose_detected', False):
                            pose_landmarks = results['pose_results'].get('landmarks', [])
                    
                    if pose_landmarks:
                        fall_results = self.fall_detector.detect_fall(pose_landmarks, bbox)
                        results['fall_results'] = fall_results
                        
                        if fall_results.get('fall_detected', False):
                            self.stats['fall_detections'] += 1
                            # 在图像上绘制摔倒警告
                            cv2.putText(annotated_image, "FALL DETECTED!", (50, 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                            cv2.putText(annotated_image, f"Confidence: {fall_results['fall_confidence']:.2f}", 
                                       (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        
                except Exception as e:
                    print(f"摔倒检测错误: {e}")
                    results['fall_results'] = {}
        
        # 计算处理时间
        processing_time = time.time() - start_time
        results['processing_time'] = processing_time
        self.stats['processing_time'] += processing_time
        
        # 计算FPS
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            self.stats['fps'] = self.stats['total_frames'] / elapsed_time
        
        # 绘制统计信息
        self._draw_stats(annotated_image)
        
        # 调用回调函数
        if self.detection_callback:
            try:
                self.detection_callback(annotated_image, results)
            except Exception as e:
                print(f"检测回调函数错误: {e}")
        
        if self.frame_callback:
            try:
                self.frame_callback(annotated_image)
            except Exception as e:
                print(f"帧回调函数错误: {e}")
        
        return annotated_image, results
    
    def detect_from_camera(self, 
                          camera_id: int = 0,
                          display: bool = True,
                          save_video: Optional[str] = None) -> None:
        """
        从摄像头进行实时多模态识别
        
        Args:
            camera_id: 摄像头ID
            display: 是否显示结果
            save_video: 保存视频路径
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {camera_id}")
        
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # 视频写入器
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(save_video, fourcc, 20.0, (640, 480))
        
        print("多模态识别检测器已启动")
        print("按 'q' 键退出")
        print("按 's' 键保存当前帧")
        print("按 'r' 键重置统计信息")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("无法读取摄像头帧")
                    break
                
                # 执行检测
                annotated_frame, results = self.detect(frame)
                
                # 保存视频
                if video_writer:
                    video_writer.write(annotated_frame)
                
                # 显示结果
                if display:
                    cv2.imshow('多模态识别检测器', annotated_frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # 保存当前帧
                    timestamp = int(time.time())
                    filename = f"multimodal_detection_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"已保存帧: {filename}")
                elif key == ord('r'):
                    # 重置统计信息
                    self.reset_stats()
                    print("统计信息已重置")
        
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            if display:
                cv2.destroyAllWindows()
    
    def set_callbacks(self, 
                     detection_callback: Optional[Callable] = None,
                     frame_callback: Optional[Callable] = None):
        """
        设置回调函数
        
        Args:
            detection_callback: 检测结果回调函数
            frame_callback: 帧处理回调函数
        """
        self.detection_callback = detection_callback
        self.frame_callback = frame_callback
    
    def set_detection_params(self, 
                           enable_gesture: Optional[bool] = None,
                           enable_face: Optional[bool] = None,
                           enable_pose: Optional[bool] = None,
                           detection_interval: Optional[int] = None):
        """
        设置检测参数
        
        Args:
            enable_gesture: 启用手势识别
            enable_face: 启用面部识别
            enable_pose: 启用身体姿势识别
            detection_interval: 检测间隔
        """
        if enable_gesture is not None:
            self.enable_gesture = enable_gesture
        if enable_face is not None:
            self.enable_face = enable_face
        if enable_pose is not None:
            self.enable_pose = enable_pose
        if detection_interval is not None:
            self.detection_interval = detection_interval
    
    def add_face_to_database(self, image: np.ndarray, name: str) -> bool:
        """
        添加人脸到数据库
        
        Args:
            image: 包含人脸的图像
            name: 人脸标识名称
            
        Returns:
            是否添加成功
        """
        if self.face_recognizer:
            return self.face_recognizer.add_face_to_database(image, name)
        return False
    
    def remove_face_from_database(self, name: str) -> bool:
        """
        从数据库中移除人脸
        
        Args:
            name: 人脸标识名称
            
        Returns:
            是否移除成功
        """
        if self.face_recognizer:
            return self.face_recognizer.remove_face_from_database(name)
        return False
    
    def save_face_database(self, filepath: str):
        """保存人脸数据库"""
        if self.face_recognizer:
            self.face_recognizer.save_face_database(filepath)
    
    def load_face_database(self, filepath: str) -> bool:
        """加载人脸数据库"""
        if self.face_recognizer:
            return self.face_recognizer.load_face_database(filepath)
        return False
    
    def get_supported_gestures(self) -> Dict[str, str]:
        """获取支持的手势列表"""
        if self.gesture_recognizer:
            return self.gesture_recognizer.get_gesture_info()
        return {}
    
    def get_supported_poses(self) -> Dict[str, str]:
        """获取支持的姿势列表"""
        if self.pose_recognizer:
            return self.pose_recognizer.get_pose_info()
        return {}
    
    def get_face_database_info(self) -> Dict[str, Any]:
        """获取人脸数据库信息"""
        if self.face_recognizer:
            return self.face_recognizer.get_database_info()
        return {}
    
    def _draw_stats(self, image: np.ndarray):
        """在图像上绘制统计信息"""
        # 准备统计文本 - 使用英文避免乱码
        stats_text = [
            f"FPS: {self.stats['fps']:.1f}",
            f"Frames: {self.stats['total_frames']}",
            f"Processed: {self.stats['processed_frames']}",
            f"Gestures: {self.stats['gesture_detections']}",
            f"Faces: {self.stats['face_detections']}",
            f"Poses: {self.stats['pose_detections']}"
        ]
        
        # 添加摔倒检测统计
        if self.enable_fall_detection:
            stats_text.append(f"Falls: {self.stats.get('fall_detections', 0)}")
        
        # 绘制半透明背景
        overlay = image.copy()
        cv2.rectangle(overlay, (5, 5), (200, 25 + len(stats_text) * 25), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # 绘制文本
        y_offset = 30
        for i, text in enumerate(stats_text):
            cv2.putText(image, text, (15, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'gesture_detections': 0,
            'face_detections': 0,
            'pose_detections': 0,
            'fall_detections': 0,
            'processing_time': 0.0,
            'fps': 0.0
        }
        self.frame_count = 0
        self.start_time = time.time()
    
    def close(self):
        """释放资源"""
        if self.gesture_recognizer:
            self.gesture_recognizer.close()
        if self.face_recognizer:
            self.face_recognizer.close()
        if self.pose_recognizer:
            self.pose_recognizer.close()
        
        print("多模态识别检测器已关闭")