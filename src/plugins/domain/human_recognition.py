"""人类识别插件"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import time

from ...core.base_plugin import DomainPlugin, PluginCapability, PluginStatus
from ...core.event_bus import EventBus


class HumanFeature(Enum):
    """人类特征类型"""
    FACE = "face"
    GESTURE = "gesture"
    POSE = "pose"
    FALL_DETECTION = "fall_detection"
    AGE_GENDER = "age_gender"
    EMOTION = "emotion"
    ACTIVITY = "activity"


@dataclass
class FaceDetection:
    """面部检测结果"""
    bbox: List[int]  # [x, y, w, h]
    confidence: float
    landmarks: Optional[List[List[int]]] = None  # 面部关键点
    age: Optional[int] = None
    gender: Optional[str] = None
    emotion: Optional[str] = None
    identity: Optional[str] = None  # 人脸识别ID


@dataclass
class GestureDetection:
    """手势检测结果"""
    gesture_type: str
    confidence: float
    hand_landmarks: List[List[int]]  # 手部关键点
    bbox: List[int]  # 手部边界框
    hand_side: str  # left/right


@dataclass
class PoseDetection:
    """姿势检测结果"""
    keypoints: List[List[int]]  # 身体关键点 [x, y, confidence]
    bbox: List[int]  # 人体边界框
    pose_type: Optional[str] = None  # 姿势类型
    activity: Optional[str] = None  # 活动类型


@dataclass
class FallDetection:
    """摔倒检测结果"""
    is_fall: bool
    confidence: float
    fall_type: Optional[str] = None  # 摔倒类型
    severity: Optional[str] = None  # 严重程度
    timestamp: Optional[float] = None


@dataclass
class HumanDetectionResult:
    """人类检测综合结果"""
    person_id: str
    bbox: List[int]  # 整体人体边界框
    confidence: float
    
    # 各种特征检测结果
    face: Optional[FaceDetection] = None
    gestures: List[GestureDetection] = None
    pose: Optional[PoseDetection] = None
    fall: Optional[FallDetection] = None
    
    # 元数据
    timestamp: float = 0.0
    frame_id: int = 0
    
    def __post_init__(self):
        if self.gestures is None:
            self.gestures = []
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class HumanRecognitionPlugin(DomainPlugin):
    """人类识别插件
    
    整合面部识别、手势识别、姿势检测、摔倒检测等功能
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 插件元数据
        self.metadata.name = "Human Recognition Plugin"
        self.metadata.version = "1.0.0"
        self.metadata.description = "Comprehensive human recognition including face, gesture, pose, and fall detection"
        self.metadata.author = "YOLOS Team"
        self.metadata.capabilities = [
            PluginCapability.DETECTION,
            PluginCapability.RECOGNITION,
            PluginCapability.TRACKING,
            PluginCapability.ANALYSIS
        ]
        self.metadata.dependencies = ["opencv-python", "mediapipe", "numpy"]
        
        # 功能模块
        self._face_detector = None
        self._gesture_detector = None
        self._pose_detector = None
        self._fall_detector = None
        
        # 启用的功能
        self.enabled_features = set()
        
        # 跟踪器
        self._person_tracker = {}
        self._next_person_id = 1
        
        # 性能统计
        self._stats = {
            'total_detections': 0,
            'face_detections': 0,
            'gesture_detections': 0,
            'pose_detections': 0,
            'fall_detections': 0,
            'avg_processing_time': 0.0
        }
    
    def initialize(self) -> bool:
        """初始化插件"""
        try:
            self.logger.info("Initializing Human Recognition Plugin")
            
            # 解析配置
            self._parse_config()
            
            # 初始化各个检测器
            if HumanFeature.FACE in self.enabled_features:
                self._init_face_detector()
            
            if HumanFeature.GESTURE in self.enabled_features:
                self._init_gesture_detector()
            
            if HumanFeature.POSE in self.enabled_features:
                self._init_pose_detector()
            
            if HumanFeature.FALL_DETECTION in self.enabled_features:
                self._init_fall_detector()
            
            self.status = PluginStatus.ACTIVE
            self.logger.info(f"Human Recognition Plugin initialized with features: {self.enabled_features}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Human Recognition Plugin: {e}")
            self.status = PluginStatus.ERROR
            return False
    
    def _parse_config(self) -> None:
        """解析配置"""
        # 默认启用所有功能
        default_features = [HumanFeature.FACE, HumanFeature.GESTURE, HumanFeature.POSE, HumanFeature.FALL_DETECTION]
        
        enabled_features_config = self.config.get('enabled_features', [f.value for f in default_features])
        
        for feature_name in enabled_features_config:
            try:
                feature = HumanFeature(feature_name)
                self.enabled_features.add(feature)
            except ValueError:
                self.logger.warning(f"Unknown human feature: {feature_name}")
    
    def _init_face_detector(self) -> None:
        """初始化面部检测器"""
        try:
            import mediapipe as mp
            
            self._mp_face_detection = mp.solutions.face_detection
            self._mp_face_mesh = mp.solutions.face_mesh
            self._mp_drawing = mp.solutions.drawing_utils
            
            # 面部检测
            face_config = self.config.get('face_detection', {})
            self._face_detector = self._mp_face_detection.FaceDetection(
                model_selection=face_config.get('model_selection', 0),
                min_detection_confidence=face_config.get('min_detection_confidence', 0.5)
            )
            
            # 面部网格（用于关键点）
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=face_config.get('max_num_faces', 5),
                refine_landmarks=True,
                min_detection_confidence=face_config.get('min_detection_confidence', 0.5),
                min_tracking_confidence=face_config.get('min_tracking_confidence', 0.5)
            )
            
            self.logger.info("Face detector initialized")
            
        except ImportError:
            self.logger.error("MediaPipe not available for face detection")
            self.enabled_features.discard(HumanFeature.FACE)
    
    def _init_gesture_detector(self) -> None:
        """初始化手势检测器"""
        try:
            import mediapipe as mp
            
            self._mp_hands = mp.solutions.hands
            
            gesture_config = self.config.get('gesture_detection', {})
            self._gesture_detector = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=gesture_config.get('max_num_hands', 4),
                min_detection_confidence=gesture_config.get('min_detection_confidence', 0.5),
                min_tracking_confidence=gesture_config.get('min_tracking_confidence', 0.5)
            )
            
            # 手势分类器（简化实现）
            self._gesture_classifier = self._create_gesture_classifier()
            
            self.logger.info("Gesture detector initialized")
            
        except ImportError:
            self.logger.error("MediaPipe not available for gesture detection")
            self.enabled_features.discard(HumanFeature.GESTURE)
    
    def _init_pose_detector(self) -> None:
        """初始化姿势检测器"""
        try:
            import mediapipe as mp
            
            self._mp_pose = mp.solutions.pose
            
            pose_config = self.config.get('pose_detection', {})
            self._pose_detector = self._mp_pose.Pose(
                static_image_mode=False,
                model_complexity=pose_config.get('model_complexity', 1),
                smooth_landmarks=pose_config.get('smooth_landmarks', True),
                enable_segmentation=pose_config.get('enable_segmentation', False),
                smooth_segmentation=pose_config.get('smooth_segmentation', True),
                min_detection_confidence=pose_config.get('min_detection_confidence', 0.5),
                min_tracking_confidence=pose_config.get('min_tracking_confidence', 0.5)
            )
            
            self.logger.info("Pose detector initialized")
            
        except ImportError:
            self.logger.error("MediaPipe not available for pose detection")
            self.enabled_features.discard(HumanFeature.POSE)
    
    def _init_fall_detector(self) -> None:
        """初始化摔倒检测器"""
        try:
            # 摔倒检测基于姿势分析
            if HumanFeature.POSE not in self.enabled_features:
                self.logger.warning("Fall detection requires pose detection")
                self.enabled_features.discard(HumanFeature.FALL_DETECTION)
                return
            
            fall_config = self.config.get('fall_detection', {})
            self._fall_threshold = fall_config.get('fall_threshold', 0.7)
            self._fall_history_size = fall_config.get('history_size', 10)
            self._fall_history = {}
            
            self.logger.info("Fall detector initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize fall detector: {e}")
            self.enabled_features.discard(HumanFeature.FALL_DETECTION)
    
    def _create_gesture_classifier(self) -> Dict[str, Any]:
        """创建手势分类器（简化实现）"""
        return {
            'gestures': {
                'thumbs_up': self._detect_thumbs_up,
                'peace': self._detect_peace_sign,
                'fist': self._detect_fist,
                'open_palm': self._detect_open_palm,
                'pointing': self._detect_pointing
            }
        }
    
    def process_frame(self, frame: np.ndarray, frame_id: int = 0) -> List[HumanDetectionResult]:
        """处理单帧图像
        
        Args:
            frame: 输入图像
            frame_id: 帧ID
            
        Returns:
            List[HumanDetectionResult]: 检测结果列表
        """
        start_time = time.time()
        results = []
        
        try:
            # 转换颜色空间
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 检测人体（使用简化的人体检测）
            person_bboxes = self._detect_persons(frame)
            
            for i, bbox in enumerate(person_bboxes):
                person_id = self._get_or_create_person_id(bbox, frame_id)
                
                result = HumanDetectionResult(
                    person_id=person_id,
                    bbox=bbox,
                    confidence=0.8,  # 简化实现
                    frame_id=frame_id
                )
                
                # 提取人体区域
                x, y, w, h = bbox
                person_roi = rgb_frame[y:y+h, x:x+w]
                
                # 面部检测
                if HumanFeature.FACE in self.enabled_features:
                    result.face = self._detect_face(person_roi, bbox)
                
                # 手势检测
                if HumanFeature.GESTURE in self.enabled_features:
                    result.gestures = self._detect_gestures(person_roi, bbox)
                
                # 姿势检测
                if HumanFeature.POSE in self.enabled_features:
                    result.pose = self._detect_pose(person_roi, bbox)
                
                # 摔倒检测
                if HumanFeature.FALL_DETECTION in self.enabled_features and result.pose:
                    result.fall = self._detect_fall(result.pose, person_id)
                
                results.append(result)
            
            # 更新统计信息
            processing_time = time.time() - start_time
            self._update_stats(results, processing_time)
            
            # 发送事件
            if self.event_bus and results:
                self.event_bus.emit('human_detection_completed', {
                    'plugin': self.metadata.name,
                    'frame_id': frame_id,
                    'detections': len(results),
                    'processing_time': processing_time
                })
            
        except Exception as e:
            self.logger.error(f"Error processing frame {frame_id}: {e}")
        
        return results
    
    def _detect_persons(self, frame: np.ndarray) -> List[List[int]]:
        """检测人体（简化实现）
        
        Args:
            frame: 输入图像
            
        Returns:
            List[List[int]]: 人体边界框列表 [[x, y, w, h], ...]
        """
        # 这里应该使用YOLO或其他人体检测模型
        # 简化实现：使用OpenCV的HOG人体检测器
        try:
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            boxes, weights = hog.detectMultiScale(
                frame,
                winStride=(8, 8),
                padding=(32, 32),
                scale=1.05
            )
            
            # 转换格式并过滤
            person_bboxes = []
            for (x, y, w, h), weight in zip(boxes, weights):
                if weight > 0.5:  # 置信度阈值
                    person_bboxes.append([int(x), int(y), int(w), int(h)])
            
            return person_bboxes
            
        except Exception as e:
            self.logger.error(f"Error in person detection: {e}")
            return []
    
    def _get_or_create_person_id(self, bbox: List[int], frame_id: int) -> str:
        """获取或创建人员ID（简化的跟踪）
        
        Args:
            bbox: 边界框
            frame_id: 帧ID
            
        Returns:
            str: 人员ID
        """
        # 简化的跟踪算法：基于边界框重叠度
        x, y, w, h = bbox
        center = (x + w // 2, y + h // 2)
        
        # 查找最近的已知人员
        min_distance = float('inf')
        closest_id = None
        
        for person_id, info in self._person_tracker.items():
            if frame_id - info['last_frame'] > 30:  # 超时清理
                continue
            
            prev_center = info['center']
            distance = np.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
            
            if distance < min_distance and distance < 100:  # 距离阈值
                min_distance = distance
                closest_id = person_id
        
        if closest_id:
            # 更新已有人员
            self._person_tracker[closest_id].update({
                'center': center,
                'bbox': bbox,
                'last_frame': frame_id
            })
            return closest_id
        else:
            # 创建新人员
            new_id = f"person_{self._next_person_id}"
            self._next_person_id += 1
            
            self._person_tracker[new_id] = {
                'center': center,
                'bbox': bbox,
                'last_frame': frame_id,
                'created_frame': frame_id
            }
            
            return new_id
    
    def _detect_face(self, person_roi: np.ndarray, person_bbox: List[int]) -> Optional[FaceDetection]:
        """检测面部
        
        Args:
            person_roi: 人体区域图像
            person_bbox: 人体边界框
            
        Returns:
            Optional[FaceDetection]: 面部检测结果
        """
        try:
            if self._face_detector is None:
                return None
            
            results = self._face_detector.process(person_roi)
            
            if results.detections:
                detection = results.detections[0]  # 取第一个检测结果
                
                # 转换坐标到原图坐标系
                h, w = person_roi.shape[:2]
                bbox_rel = detection.location_data.relative_bounding_box
                
                face_x = int(bbox_rel.xmin * w) + person_bbox[0]
                face_y = int(bbox_rel.ymin * h) + person_bbox[1]
                face_w = int(bbox_rel.width * w)
                face_h = int(bbox_rel.height * h)
                
                # 获取关键点
                landmarks = None
                if hasattr(detection.location_data, 'relative_keypoints'):
                    landmarks = []
                    for keypoint in detection.location_data.relative_keypoints:
                        kp_x = int(keypoint.x * w) + person_bbox[0]
                        kp_y = int(keypoint.y * h) + person_bbox[1]
                        landmarks.append([kp_x, kp_y])
                
                return FaceDetection(
                    bbox=[face_x, face_y, face_w, face_h],
                    confidence=detection.score[0],
                    landmarks=landmarks
                )
            
        except Exception as e:
            self.logger.error(f"Error in face detection: {e}")
        
        return None
    
    def _detect_gestures(self, person_roi: np.ndarray, person_bbox: List[int]) -> List[GestureDetection]:
        """检测手势
        
        Args:
            person_roi: 人体区域图像
            person_bbox: 人体边界框
            
        Returns:
            List[GestureDetection]: 手势检测结果列表
        """
        gestures = []
        
        try:
            if self._gesture_detector is None:
                return gestures
            
            results = self._gesture_detector.process(person_roi)
            
            if results.multi_hand_landmarks:
                h, w = person_roi.shape[:2]
                
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # 转换关键点坐标
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        x = int(landmark.x * w) + person_bbox[0]
                        y = int(landmark.y * h) + person_bbox[1]
                        landmarks.append([x, y])
                    
                    # 计算手部边界框
                    xs = [lm[0] for lm in landmarks]
                    ys = [lm[1] for lm in landmarks]
                    hand_bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
                    
                    # 识别手势类型
                    gesture_type, confidence = self._classify_gesture(landmarks)
                    
                    # 确定左右手
                    hand_side = "right" if i < len(results.multi_handedness) else "unknown"
                    if i < len(results.multi_handedness):
                        hand_side = results.multi_handedness[i].classification[0].label.lower()
                    
                    gestures.append(GestureDetection(
                        gesture_type=gesture_type,
                        confidence=confidence,
                        hand_landmarks=landmarks,
                        bbox=hand_bbox,
                        hand_side=hand_side
                    ))
            
        except Exception as e:
            self.logger.error(f"Error in gesture detection: {e}")
        
        return gestures
    
    def _classify_gesture(self, landmarks: List[List[int]]) -> Tuple[str, float]:
        """分类手势（简化实现）
        
        Args:
            landmarks: 手部关键点
            
        Returns:
            Tuple[str, float]: 手势类型和置信度
        """
        # 简化的手势识别逻辑
        try:
            # 检查各种手势
            for gesture_name, detector_func in self._gesture_classifier['gestures'].items():
                confidence = detector_func(landmarks)
                if confidence > 0.7:
                    return gesture_name, confidence
            
            return "unknown", 0.0
            
        except Exception as e:
            self.logger.error(f"Error in gesture classification: {e}")
            return "unknown", 0.0
    
    def _detect_thumbs_up(self, landmarks: List[List[int]]) -> float:
        """检测竖拇指手势"""
        # 简化实现：检查拇指是否伸直且其他手指弯曲
        try:
            # 拇指关键点索引：4
            # 其他手指尖索引：8, 12, 16, 20
            thumb_tip = landmarks[4]
            thumb_mcp = landmarks[2]
            
            finger_tips = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
            finger_mcps = [landmarks[6], landmarks[10], landmarks[14], landmarks[18]]
            
            # 检查拇指是否向上
            thumb_up = thumb_tip[1] < thumb_mcp[1]
            
            # 检查其他手指是否弯曲
            fingers_down = 0
            for tip, mcp in zip(finger_tips, finger_mcps):
                if tip[1] > mcp[1]:  # 手指尖在掌关节下方
                    fingers_down += 1
            
            if thumb_up and fingers_down >= 3:
                return 0.8
            
        except (IndexError, TypeError):
            pass
        
        return 0.0
    
    def _detect_peace_sign(self, landmarks: List[List[int]]) -> float:
        """检测V字手势"""
        try:
            # 检查食指和中指是否伸直，其他手指弯曲
            index_tip = landmarks[8]
            index_mcp = landmarks[6]
            middle_tip = landmarks[12]
            middle_mcp = landmarks[10]
            
            # 检查食指和中指是否伸直
            index_straight = index_tip[1] < index_mcp[1]
            middle_straight = middle_tip[1] < middle_mcp[1]
            
            # 检查其他手指是否弯曲
            ring_bent = landmarks[16][1] > landmarks[14][1]
            pinky_bent = landmarks[20][1] > landmarks[18][1]
            
            if index_straight and middle_straight and ring_bent and pinky_bent:
                return 0.8
            
        except (IndexError, TypeError):
            pass
        
        return 0.0
    
    def _detect_fist(self, landmarks: List[List[int]]) -> float:
        """检测拳头手势"""
        try:
            # 检查所有手指是否弯曲
            finger_tips = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
            finger_mcps = [landmarks[6], landmarks[10], landmarks[14], landmarks[18]]
            
            bent_fingers = 0
            for tip, mcp in zip(finger_tips, finger_mcps):
                if tip[1] > mcp[1]:  # 手指尖在掌关节下方
                    bent_fingers += 1
            
            if bent_fingers >= 3:
                return 0.8
            
        except (IndexError, TypeError):
            pass
        
        return 0.0
    
    def _detect_open_palm(self, landmarks: List[List[int]]) -> float:
        """检测张开手掌"""
        try:
            # 检查所有手指是否伸直
            finger_tips = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
            finger_mcps = [landmarks[6], landmarks[10], landmarks[14], landmarks[18]]
            
            straight_fingers = 0
            for tip, mcp in zip(finger_tips, finger_mcps):
                if tip[1] < mcp[1]:  # 手指尖在掌关节上方
                    straight_fingers += 1
            
            if straight_fingers >= 3:
                return 0.8
            
        except (IndexError, TypeError):
            pass
        
        return 0.0
    
    def _detect_pointing(self, landmarks: List[List[int]]) -> float:
        """检测指向手势"""
        try:
            # 检查食指是否伸直，其他手指弯曲
            index_tip = landmarks[8]
            index_mcp = landmarks[6]
            
            other_tips = [landmarks[12], landmarks[16], landmarks[20]]
            other_mcps = [landmarks[10], landmarks[14], landmarks[18]]
            
            # 食指伸直
            index_straight = index_tip[1] < index_mcp[1]
            
            # 其他手指弯曲
            bent_fingers = 0
            for tip, mcp in zip(other_tips, other_mcps):
                if tip[1] > mcp[1]:
                    bent_fingers += 1
            
            if index_straight and bent_fingers >= 2:
                return 0.8
            
        except (IndexError, TypeError):
            pass
        
        return 0.0
    
    def _detect_pose(self, person_roi: np.ndarray, person_bbox: List[int]) -> Optional[PoseDetection]:
        """检测人体姿势
        
        Args:
            person_roi: 人体区域图像
            person_bbox: 人体边界框
            
        Returns:
            Optional[PoseDetection]: 姿势检测结果
        """
        try:
            if self._pose_detector is None:
                return None
            
            results = self._pose_detector.process(person_roi)
            
            if results.pose_landmarks:
                h, w = person_roi.shape[:2]
                
                # 转换关键点坐标
                keypoints = []
                for landmark in results.pose_landmarks.landmark:
                    x = int(landmark.x * w) + person_bbox[0]
                    y = int(landmark.y * h) + person_bbox[1]
                    confidence = landmark.visibility
                    keypoints.append([x, y, confidence])
                
                # 分析姿势类型
                pose_type = self._analyze_pose_type(keypoints)
                activity = self._analyze_activity(keypoints)
                
                return PoseDetection(
                    keypoints=keypoints,
                    bbox=person_bbox,
                    pose_type=pose_type,
                    activity=activity
                )
            
        except Exception as e:
            self.logger.error(f"Error in pose detection: {e}")
        
        return None
    
    def _analyze_pose_type(self, keypoints: List[List[int]]) -> str:
        """分析姿势类型
        
        Args:
            keypoints: 身体关键点
            
        Returns:
            str: 姿势类型
        """
        try:
            # 简化的姿势分析
            # 关键点索引参考MediaPipe Pose
            nose = keypoints[0]
            left_shoulder = keypoints[11]
            right_shoulder = keypoints[12]
            left_hip = keypoints[23]
            right_hip = keypoints[24]
            
            # 计算身体角度
            shoulder_center = [(left_shoulder[0] + right_shoulder[0]) // 2,
                             (left_shoulder[1] + right_shoulder[1]) // 2]
            hip_center = [(left_hip[0] + right_hip[0]) // 2,
                         (left_hip[1] + right_hip[1]) // 2]
            
            # 身体倾斜角度
            if abs(shoulder_center[0] - hip_center[0]) > 50:
                return "leaning"
            
            # 检查是否坐着（简化判断）
            if nose[1] - hip_center[1] < 100:  # 头部和臀部距离较近
                return "sitting"
            
            return "standing"
            
        except (IndexError, TypeError):
            return "unknown"
    
    def _analyze_activity(self, keypoints: List[List[int]]) -> str:
        """分析活动类型
        
        Args:
            keypoints: 身体关键点
            
        Returns:
            str: 活动类型
        """
        try:
            # 简化的活动分析
            left_wrist = keypoints[15]
            right_wrist = keypoints[16]
            left_shoulder = keypoints[11]
            right_shoulder = keypoints[12]
            
            # 检查手臂位置
            left_arm_raised = left_wrist[1] < left_shoulder[1]
            right_arm_raised = right_wrist[1] < right_shoulder[1]
            
            if left_arm_raised and right_arm_raised:
                return "arms_raised"
            elif left_arm_raised or right_arm_raised:
                return "waving"
            
            return "normal"
            
        except (IndexError, TypeError):
            return "unknown"
    
    def _detect_fall(self, pose: PoseDetection, person_id: str) -> Optional[FallDetection]:
        """检测摔倒
        
        Args:
            pose: 姿势检测结果
            person_id: 人员ID
            
        Returns:
            Optional[FallDetection]: 摔倒检测结果
        """
        try:
            if not pose or not pose.keypoints:
                return None
            
            # 计算身体方向和角度
            fall_score = self._calculate_fall_score(pose.keypoints)
            
            # 维护历史记录
            if person_id not in self._fall_history:
                self._fall_history[person_id] = []
            
            self._fall_history[person_id].append(fall_score)
            
            # 保持历史记录大小
            if len(self._fall_history[person_id]) > self._fall_history_size:
                self._fall_history[person_id].pop(0)
            
            # 分析摔倒趋势
            recent_scores = self._fall_history[person_id][-5:]  # 最近5帧
            avg_score = sum(recent_scores) / len(recent_scores)
            
            is_fall = avg_score > self._fall_threshold
            
            if is_fall:
                # 确定摔倒类型和严重程度
                fall_type = self._classify_fall_type(pose.keypoints)
                severity = "high" if avg_score > 0.9 else "medium"
                
                return FallDetection(
                    is_fall=True,
                    confidence=avg_score,
                    fall_type=fall_type,
                    severity=severity,
                    timestamp=time.time()
                )
            
        except Exception as e:
            self.logger.error(f"Error in fall detection: {e}")
        
        return FallDetection(is_fall=False, confidence=0.0)
    
    def _calculate_fall_score(self, keypoints: List[List[int]]) -> float:
        """计算摔倒分数
        
        Args:
            keypoints: 身体关键点
            
        Returns:
            float: 摔倒分数 (0-1)
        """
        try:
            # 关键点索引
            nose = keypoints[0]
            left_shoulder = keypoints[11]
            right_shoulder = keypoints[12]
            left_hip = keypoints[23]
            right_hip = keypoints[24]
            left_ankle = keypoints[27]
            right_ankle = keypoints[28]
            
            # 计算身体中心点
            shoulder_center = [(left_shoulder[0] + right_shoulder[0]) // 2,
                             (left_shoulder[1] + right_shoulder[1]) // 2]
            hip_center = [(left_hip[0] + right_hip[0]) // 2,
                         (left_hip[1] + right_hip[1]) // 2]
            ankle_center = [(left_ankle[0] + right_ankle[0]) // 2,
                           (left_ankle[1] + right_ankle[1]) // 2]
            
            # 计算身体角度（相对于垂直方向）
            body_vector = [hip_center[0] - shoulder_center[0],
                          hip_center[1] - shoulder_center[1]]
            
            # 计算与垂直方向的角度
            import math
            angle = math.atan2(abs(body_vector[0]), abs(body_vector[1]))
            angle_degrees = math.degrees(angle)
            
            # 角度越大，摔倒可能性越高
            angle_score = min(angle_degrees / 45.0, 1.0)  # 45度为最大角度
            
            # 检查头部位置（头部过低表示可能摔倒）
            head_height_score = 0.0
            if nose[1] > hip_center[1]:  # 头部低于臀部
                head_height_score = 0.5
            if nose[1] > ankle_center[1]:  # 头部低于脚踝
                head_height_score = 1.0
            
            # 综合评分
            fall_score = (angle_score * 0.6 + head_height_score * 0.4)
            
            return min(fall_score, 1.0)
            
        except (IndexError, TypeError, ZeroDivisionError):
            return 0.0
    
    def _classify_fall_type(self, keypoints: List[List[int]]) -> str:
        """分类摔倒类型
        
        Args:
            keypoints: 身体关键点
            
        Returns:
            str: 摔倒类型
        """
        try:
            # 简化的摔倒类型分类
            nose = keypoints[0]
            left_shoulder = keypoints[11]
            right_shoulder = keypoints[12]
            left_hip = keypoints[23]
            right_hip = keypoints[24]
            
            shoulder_center = [(left_shoulder[0] + right_shoulder[0]) // 2,
                             (left_shoulder[1] + right_shoulder[1]) // 2]
            hip_center = [(left_hip[0] + right_hip[0]) // 2,
                         (left_hip[1] + right_hip[1]) // 2]
            
            # 判断摔倒方向
            if nose[0] < shoulder_center[0] - 50:
                return "fall_left"
            elif nose[0] > shoulder_center[0] + 50:
                return "fall_right"
            elif nose[1] > hip_center[1]:
                return "fall_forward"
            else:
                return "fall_backward"
            
        except (IndexError, TypeError):
            return "unknown_fall"
    
    def _update_stats(self, results: List[HumanDetectionResult], processing_time: float) -> None:
        """更新统计信息
        
        Args:
            results: 检测结果
            processing_time: 处理时间
        """
        self._stats['total_detections'] += len(results)
        
        for result in results:
            if result.face:
                self._stats['face_detections'] += 1
            if result.gestures:
                self._stats['gesture_detections'] += len(result.gestures)
            if result.pose:
                self._stats['pose_detections'] += 1
            if result.fall and result.fall.is_fall:
                self._stats['fall_detections'] += 1
        
        # 更新平均处理时间
        total_frames = self._stats['total_detections']
        if total_frames > 0:
            self._stats['avg_processing_time'] = (
                (self._stats['avg_processing_time'] * (total_frames - len(results)) + 
                 processing_time * len(results)) / total_frames
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return self._stats.copy()
    
    def cleanup(self) -> None:
        """清理资源"""
        try:
            if self._face_detector:
                self._face_detector.close()
            if self._gesture_detector:
                self._gesture_detector.close()
            if self._pose_detector:
                self._pose_detector.close()
            
            self._person_tracker.clear()
            self._fall_history.clear()
            
            self.status = PluginStatus.INACTIVE
            self.logger.info("Human Recognition Plugin cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def get_supported_domains(self) -> List[str]:
        """获取支持的领域
        
        Returns:
            List[str]: 支持的领域列表
        """
        return ["human", "person", "face", "gesture", "pose", "fall_detection"]
    
    def get_config_schema(self) -> Dict[str, Any]:
        """获取配置模式
        
        Returns:
            Dict[str, Any]: 配置模式
        """
        return {
            "type": "object",
            "properties": {
                "enabled_features": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [f.value for f in HumanFeature]
                    },
                    "default": [f.value for f in HumanFeature]
                },
                "face_detection": {
                    "type": "object",
                    "properties": {
                        "model_selection": {"type": "integer", "default": 0},
                        "min_detection_confidence": {"type": "number", "default": 0.5},
                        "max_num_faces": {"type": "integer", "default": 5}
                    }
                },
                "gesture_detection": {
                    "type": "object",
                    "properties": {
                        "max_num_hands": {"type": "integer", "default": 4},
                        "min_detection_confidence": {"type": "number", "default": 0.5}
                    }
                },
                "pose_detection": {
                    "type": "object",
                    "properties": {
                        "model_complexity": {"type": "integer", "default": 1},
                        "min_detection_confidence": {"type": "number", "default": 0.5}
                    }
                },
                "fall_detection": {
                    "type": "object",
                    "properties": {
                        "fall_threshold": {"type": "number", "default": 0.7},
                        "history_size": {"type": "integer", "default": 10}
                    }
                }
            }
        }