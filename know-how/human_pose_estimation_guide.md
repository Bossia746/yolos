# 人体姿势识别技术指南

## 概述
本文档整理了人体姿势识别技术的实现方法、常见问题和最佳实践，涵盖MediaPipe Pose、YOLO Pose、OpenPose等主流技术栈。

## 技术栈对比

### 1. MediaPipe Pose <mcreference link="https://pmc.ncbi.nlm.nih.gov/articles/PMC11566680/" index="3">3</mcreference>

#### 核心特性
- **33个身体关键点**: 全身姿态检测，包括面部轮廓
- **实时性能**: 专为移动和嵌入式设备优化
- **BlazePose架构**: 2020年发布的快速准确姿态估计解决方案
- **跨平台支持**: Web、移动应用、桌面应用

#### 关键点结构
```python
# MediaPipe Pose 33个关键点索引
POSE_LANDMARKS = {
    # 面部 (0-10)
    'NOSE': 0, 'LEFT_EYE_INNER': 1, 'LEFT_EYE': 2, 'LEFT_EYE_OUTER': 3,
    'RIGHT_EYE_INNER': 4, 'RIGHT_EYE': 5, 'RIGHT_EYE_OUTER': 6,
    'LEFT_EAR': 7, 'RIGHT_EAR': 8, 'MOUTH_LEFT': 9, 'MOUTH_RIGHT': 10,
    
    # 上身 (11-16)
    'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
    'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15, 'RIGHT_WRIST': 16,
    
    # 下身 (17-22)
    'LEFT_HIP': 23, 'RIGHT_HIP': 24,
    'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28,
    
    # 手部细节 (17-22)
    'LEFT_PINKY': 17, 'RIGHT_PINKY': 18,
    'LEFT_INDEX': 19, 'RIGHT_INDEX': 20,
    'LEFT_THUMB': 21, 'RIGHT_THUMB': 22,
    
    # 脚部细节 (29-32)
    'LEFT_HEEL': 29, 'RIGHT_HEEL': 30,
    'LEFT_FOOT_INDEX': 31, 'RIGHT_FOOT_INDEX': 32
}
```

### 2. YOLO Pose <mcreference link="https://docs.ultralytics.com/tasks/pose/" index="1">1</mcreference>

#### 核心特性
- **COCO格式**: 17个关键点，专注主要身体关节
- **多人检测**: 同时检测多个人的姿态
- **高精度**: 在COCO数据集上预训练
- **实时推理**: 优化的推理速度

#### COCO 17关键点结构
```python
# COCO格式17个关键点
COCO_KEYPOINTS = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}
```

### 3. 其他主流模型 <mcreference link="https://pmc.ncbi.nlm.nih.gov/articles/PMC11566680/" index="3">3</mcreference>

#### OpenPose
- **多人实时检测**: 业界标准
- **多模态**: 身体、手部、面部同时检测
- **高精度**: 学术界广泛使用

#### HRNet (High-Resolution Network)
- **高分辨率特征**: 保持细节信息
- **SOTA性能**: 在多个基准测试中表现优异
- **计算密集**: 需要较强的计算资源

#### MoveNet
- **轻量级**: 专为移动设备设计
- **17个关键点**: COCO格式兼容
- **TensorFlow Lite**: 移动端优化

## 实现方案

### 1. MediaPipe Pose实现

```python
import cv2
import mediapipe as mp
import numpy as np

class PoseEstimator:
    def __init__(self, 
                 static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 enable_segmentation=False,
                 smooth_segmentation=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles
    
    def detect_pose(self, image, draw=True):
        """检测人体姿态"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        
        landmarks = []
        if results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(
                    image, 
                    results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_draw_styles.get_default_pose_landmarks_style()
                )
            
            # 提取关键点坐标
            for lm in results.pose_landmarks.landmark:
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append([cx, cy, lm.visibility])
        
        return image, landmarks, results.segmentation_mask
    
    def calculate_angle(self, point1, point2, point3):
        """计算三点之间的角度"""
        a = np.array(point1[:2])  # 第一个点
        b = np.array(point2[:2])  # 顶点
        c = np.array(point3[:2])  # 第三个点
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        
        return angle
```

### 2. YOLO Pose实现 <mcreference link="https://docs.ultralytics.com/tasks/pose/" index="1">1</mcreference>

```python
from ultralytics import YOLO
import cv2
import numpy as np

class YOLOPoseEstimator:
    def __init__(self, model_path='yolo11n-pose.pt'):
        self.model = YOLO(model_path)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    def detect_pose(self, image, conf_threshold=0.5):
        """使用YOLO检测多人姿态"""
        results = self.model(image, conf=conf_threshold)
        
        detections = []
        for result in results:
            if result.keypoints is not None:
                keypoints = result.keypoints.data.cpu().numpy()
                boxes = result.boxes.data.cpu().numpy() if result.boxes else None
                
                for i, kpts in enumerate(keypoints):
                    detection = {
                        'keypoints': kpts,  # shape: (17, 3) - x, y, confidence
                        'bbox': boxes[i] if boxes is not None else None,
                        'person_id': i
                    }
                    detections.append(detection)
        
        return detections
    
    def draw_pose(self, image, detections):
        """绘制姿态检测结果"""
        # COCO骨架连接
        skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]
        ]
        
        for detection in detections:
            keypoints = detection['keypoints']
            
            # 绘制关键点
            for i, (x, y, conf) in enumerate(keypoints):
                if conf > 0.5:  # 置信度阈值
                    cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
                    cv2.putText(image, str(i), (int(x), int(y)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # 绘制骨架
            for connection in skeleton:
                kpt_a, kpt_b = connection
                if (kpt_a < len(keypoints) and kpt_b < len(keypoints) and 
                    keypoints[kpt_a][2] > 0.5 and keypoints[kpt_b][2] > 0.5):
                    
                    x1, y1 = int(keypoints[kpt_a][0]), int(keypoints[kpt_a][1])
                    x2, y2 = int(keypoints[kpt_b][0]), int(keypoints[kpt_b][1])
                    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        return image
```

### 3. 多人姿态检测集成 <mcreference link="https://www.reddit.com/r/computervision/comments/1kfyol4/need_help_in_our_human_pose_detection_project/" index="1">1</mcreference>

```python
class MultiPersonPoseEstimator:
    def __init__(self, detection_model='yolo11n.pt', pose_model='mediapipe'):
        # 人体检测模型
        self.detector = YOLO(detection_model)
        
        # 姿态估计模型
        if pose_model == 'mediapipe':
            self.pose_estimator = PoseEstimator()
        elif pose_model == 'yolo':
            self.pose_estimator = YOLOPoseEstimator()
    
    def detect_multi_person_pose(self, image):
        """检测多人姿态"""
        # 1. 检测人体边界框
        detection_results = self.detector(image, classes=[0])  # 只检测人
        
        poses = []
        for result in detection_results:
            if result.boxes is not None:
                boxes = result.boxes.data.cpu().numpy()
                
                for box in boxes:
                    x1, y1, x2, y2, conf, cls = box
                    if conf > 0.5:  # 置信度阈值
                        # 2. 裁剪人体区域
                        person_roi = image[int(y1):int(y2), int(x1):int(x2)]
                        
                        # 3. 对每个人进行姿态估计
                        if isinstance(self.pose_estimator, PoseEstimator):
                            _, landmarks, _ = self.pose_estimator.detect_pose(person_roi, draw=False)
                            
                            # 将相对坐标转换为绝对坐标
                            if landmarks:
                                absolute_landmarks = []
                                for lm in landmarks:
                                    abs_x = lm[0] + int(x1)
                                    abs_y = lm[1] + int(y1)
                                    absolute_landmarks.append([abs_x, abs_y, lm[2]])
                                
                                poses.append({
                                    'bbox': [x1, y1, x2, y2],
                                    'landmarks': absolute_landmarks,
                                    'confidence': conf
                                })
        
        return poses
```

## 姿态分析与应用

### 1. 姿态分类

```python
class PoseClassifier:
    def __init__(self):
        self.pose_thresholds = {
            'standing': {'hip_knee_angle': (160, 180), 'knee_ankle_angle': (160, 180)},
            'sitting': {'hip_knee_angle': (80, 120), 'knee_ankle_angle': (80, 120)},
            'lying': {'shoulder_hip_angle': (160, 180)},
            'squatting': {'hip_knee_angle': (30, 90), 'knee_ankle_angle': (30, 90)}
        }
    
    def classify_pose(self, landmarks):
        """基于关键点分类姿态"""
        if not landmarks or len(landmarks) < 17:
            return "unknown"
        
        # 计算关键角度
        angles = self.calculate_body_angles(landmarks)
        
        # 姿态分类逻辑
        for pose_name, thresholds in self.pose_thresholds.items():
            if self.check_pose_criteria(angles, thresholds):
                return pose_name
        
        return "unknown"
    
    def calculate_body_angles(self, landmarks):
        """计算身体关键角度"""
        angles = {}
        
        try:
            # 髋-膝-踝角度（左腿）
            if all(landmarks[i][2] > 0.5 for i in [11, 13, 15]):  # 左髋、左膝、左踝
                angles['left_hip_knee_ankle'] = self.calculate_angle(
                    landmarks[11], landmarks[13], landmarks[15]
                )
            
            # 肩-髋角度（躯干倾斜）
            if all(landmarks[i][2] > 0.5 for i in [5, 6, 11, 12]):  # 双肩、双髋
                shoulder_center = [(landmarks[5][0] + landmarks[6][0]) / 2,
                                 (landmarks[5][1] + landmarks[6][1]) / 2]
                hip_center = [(landmarks[11][0] + landmarks[12][0]) / 2,
                            (landmarks[11][1] + landmarks[12][1]) / 2]
                
                # 计算躯干与垂直方向的角度
                trunk_angle = np.arctan2(
                    abs(shoulder_center[0] - hip_center[0]),
                    abs(shoulder_center[1] - hip_center[1])
                ) * 180 / np.pi
                
                angles['trunk_angle'] = trunk_angle
        
        except Exception as e:
            print(f"Angle calculation error: {e}")
        
        return angles
    
    def check_pose_criteria(self, angles, thresholds):
        """检查姿态是否符合标准"""
        for angle_name, (min_val, max_val) in thresholds.items():
            if angle_name in angles:
                if not (min_val <= angles[angle_name] <= max_val):
                    return False
            else:
                return False  # 缺少必要角度信息
        return True
```

### 2. 运动分析 <mcreference link="https://learnopencv.com/building-a-body-posture-analysis-system-using-mediapipe/" index="4">4</mcreference>

```python
class PostureAnalyzer:
    def __init__(self):
        self.pose_history = []
        self.max_history = 30  # 保存30帧历史
    
    def analyze_posture(self, landmarks):
        """分析身体姿态"""
        if not landmarks:
            return {"status": "no_pose_detected"}
        
        self.pose_history.append(landmarks)
        if len(self.pose_history) > self.max_history:
            self.pose_history.pop(0)
        
        analysis = {
            "head_position": self.analyze_head_position(landmarks),
            "shoulder_alignment": self.analyze_shoulder_alignment(landmarks),
            "spine_curvature": self.analyze_spine_curvature(landmarks),
            "overall_score": 0
        }
        
        # 计算综合评分
        analysis["overall_score"] = self.calculate_posture_score(analysis)
        
        return analysis
    
    def analyze_head_position(self, landmarks):
        """分析头部位置"""
        try:
            nose = landmarks[0]
            left_shoulder = landmarks[5]
            right_shoulder = landmarks[6]
            
            # 计算肩膀中心点
            shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            
            # 头部前倾程度
            head_forward = abs(nose[0] - shoulder_center_x)
            
            if head_forward < 20:
                return {"status": "good", "forward_distance": head_forward}
            elif head_forward < 40:
                return {"status": "moderate", "forward_distance": head_forward}
            else:
                return {"status": "poor", "forward_distance": head_forward}
        
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def analyze_shoulder_alignment(self, landmarks):
        """分析肩膀对齐"""
        try:
            left_shoulder = landmarks[5]
            right_shoulder = landmarks[6]
            
            # 计算肩膀高度差
            height_diff = abs(left_shoulder[1] - right_shoulder[1])
            
            if height_diff < 10:
                return {"status": "aligned", "height_difference": height_diff}
            elif height_diff < 25:
                return {"status": "slightly_uneven", "height_difference": height_diff}
            else:
                return {"status": "uneven", "height_difference": height_diff}
        
        except Exception as e:
            return {"status": "error", "message": str(e)}
```

## 常见问题与解决方案

### 1. 关键点检测不稳定

#### 问题症状
- 关键点抖动
- 检测中断
- 精度下降

#### 解决方案
```python
class LandmarkStabilizer:
    def __init__(self, smoothing_factor=0.7):
        self.smoothing_factor = smoothing_factor
        self.previous_landmarks = None
        self.confidence_threshold = 0.5
    
    def stabilize_landmarks(self, current_landmarks):
        """稳定关键点检测"""
        if not current_landmarks:
            return self.previous_landmarks
        
        # 过滤低置信度关键点
        filtered_landmarks = []
        for lm in current_landmarks:
            if len(lm) >= 3 and lm[2] > self.confidence_threshold:
                filtered_landmarks.append(lm)
            else:
                # 使用前一帧的对应点
                if (self.previous_landmarks and 
                    len(self.previous_landmarks) > len(filtered_landmarks)):
                    filtered_landmarks.append(self.previous_landmarks[len(filtered_landmarks)])
                else:
                    filtered_landmarks.append([0, 0, 0])  # 默认值
        
        # 时间平滑
        if self.previous_landmarks:
            smoothed_landmarks = []
            for curr, prev in zip(filtered_landmarks, self.previous_landmarks):
                smoothed_x = (self.smoothing_factor * curr[0] + 
                            (1 - self.smoothing_factor) * prev[0])
                smoothed_y = (self.smoothing_factor * curr[1] + 
                            (1 - self.smoothing_factor) * prev[1])
                smoothed_conf = max(curr[2], prev[2])  # 保持较高置信度
                
                smoothed_landmarks.append([smoothed_x, smoothed_y, smoothed_conf])
            
            self.previous_landmarks = smoothed_landmarks
        else:
            self.previous_landmarks = filtered_landmarks
        
        return self.previous_landmarks
```

### 2. 多人场景处理 <mcreference link="https://www.reddit.com/r/deeplearning/comments/1kfyrfy/need_help_in_our_human_pose_detection_project/" index="2">2</mcreference>

#### 问题症状
- 人员ID混乱
- 遮挡处理困难
- 性能下降

#### 解决方案
```python
class MultiPersonTracker:
    def __init__(self, max_disappeared=10):
        self.next_id = 0
        self.persons = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
    
    def update(self, detections):
        """更新多人跟踪"""
        if len(detections) == 0:
            # 标记所有人员为消失
            for person_id in list(self.disappeared.keys()):
                self.disappeared[person_id] += 1
                if self.disappeared[person_id] > self.max_disappeared:
                    self.deregister(person_id)
            return self.persons
        
        if len(self.persons) == 0:
            # 注册新检测到的人员
            for detection in detections:
                self.register(detection)
        else:
            # 计算现有人员与新检测的距离
            person_centroids = [self.get_centroid(person['landmarks']) 
                              for person in self.persons.values()]
            detection_centroids = [self.get_centroid(det['landmarks']) 
                                 for det in detections]
            
            # 使用匈牙利算法进行最优匹配
            matches = self.hungarian_matching(person_centroids, detection_centroids)
            
            # 更新匹配的人员
            for person_idx, detection_idx in matches:
                person_id = list(self.persons.keys())[person_idx]
                self.persons[person_id] = detections[detection_idx]
                del self.disappeared[person_id]
            
            # 处理未匹配的检测（新人员）
            unmatched_detections = set(range(len(detections))) - set([m[1] for m in matches])
            for detection_idx in unmatched_detections:
                self.register(detections[detection_idx])
            
            # 处理未匹配的现有人员（消失的人员）
            unmatched_persons = set(range(len(self.persons))) - set([m[0] for m in matches])
            for person_idx in unmatched_persons:
                person_id = list(self.persons.keys())[person_idx]
                self.disappeared[person_id] = self.disappeared.get(person_id, 0) + 1
                
                if self.disappeared[person_id] > self.max_disappeared:
                    self.deregister(person_id)
        
        return self.persons
    
    def register(self, detection):
        """注册新人员"""
        self.persons[self.next_id] = detection
        self.disappeared[self.next_id] = 0
        self.next_id += 1
    
    def deregister(self, person_id):
        """注销人员"""
        del self.persons[person_id]
        del self.disappeared[person_id]
    
    def get_centroid(self, landmarks):
        """计算关键点质心"""
        if not landmarks:
            return (0, 0)
        
        valid_points = [lm for lm in landmarks if len(lm) >= 3 and lm[2] > 0.5]
        if not valid_points:
            return (0, 0)
        
        x = sum(point[0] for point in valid_points) / len(valid_points)
        y = sum(point[1] for point in valid_points) / len(valid_points)
        return (x, y)
```

### 3. 性能优化

#### 帧率优化
```python
class PoseEstimationOptimizer:
    def __init__(self, target_fps=30):
        self.target_fps = target_fps
        self.frame_skip = 1
        self.processing_times = []
        self.max_time_samples = 10
    
    def adaptive_frame_skip(self, processing_time):
        """自适应帧跳跃"""
        self.processing_times.append(processing_time)
        if len(self.processing_times) > self.max_time_samples:
            self.processing_times.pop(0)
        
        avg_time = sum(self.processing_times) / len(self.processing_times)
        target_time = 1.0 / self.target_fps
        
        if avg_time > target_time * 1.5:
            self.frame_skip = min(self.frame_skip + 1, 5)
        elif avg_time < target_time * 0.8:
            self.frame_skip = max(self.frame_skip - 1, 1)
        
        return self.frame_skip
    
    def should_process_frame(self, frame_count):
        """判断是否应该处理当前帧"""
        return frame_count % self.frame_skip == 0
```

## 应用场景

### 1. 健身动作分析
```python
class FitnessAnalyzer:
    def __init__(self):
        self.exercise_templates = {
            'squat': {
                'key_angles': ['hip_knee_ankle'],
                'angle_ranges': [(30, 90)],
                'rep_threshold': 10
            },
            'pushup': {
                'key_angles': ['shoulder_elbow_wrist'],
                'angle_ranges': [(90, 180)],
                'rep_threshold': 15
            }
        }
        self.rep_counter = 0
        self.exercise_state = 'up'
    
    def analyze_exercise(self, landmarks, exercise_type):
        """分析运动动作"""
        if exercise_type not in self.exercise_templates:
            return {"error": "Unknown exercise type"}
        
        template = self.exercise_templates[exercise_type]
        angles = self.calculate_exercise_angles(landmarks, template['key_angles'])
        
        # 动作计数逻辑
        if exercise_type == 'squat':
            return self.count_squats(angles)
        elif exercise_type == 'pushup':
            return self.count_pushups(angles)
```

### 2. 姿态矫正系统 <mcreference link="https://learnopencv.com/building-a-body-posture-analysis-system-using-mediapipe/" index="4">4</mcreference>
```python
class PostureCorrectionSystem:
    def __init__(self):
        self.alert_threshold = 5  # 连续5帧不良姿态触发警告
        self.poor_posture_count = 0
    
    def check_posture(self, landmarks):
        """检查姿态并提供矫正建议"""
        analysis = self.analyze_posture(landmarks)
        
        suggestions = []
        if analysis['head_position']['status'] == 'poor':
            suggestions.append("请将头部向后收，保持颈部自然曲线")
        
        if analysis['shoulder_alignment']['status'] == 'uneven':
            suggestions.append("请调整肩膀高度，保持水平对齐")
        
        # 警告逻辑
        if suggestions:
            self.poor_posture_count += 1
            if self.poor_posture_count >= self.alert_threshold:
                return {
                    "alert": True,
                    "suggestions": suggestions,
                    "severity": "high" if len(suggestions) > 2 else "medium"
                }
        else:
            self.poor_posture_count = 0
        
        return {"alert": False, "suggestions": suggestions}
```

## 调试与测试

### 1. 可视化调试 <mcreference link="https://docs.ultralytics.com/tasks/pose/" index="1">1</mcreference>
```python
def visualize_pose_debug(image, landmarks, angles=None):
    """可视化姿态检测调试信息"""
    debug_image = image.copy()
    
    # 绘制关键点
    for i, (x, y, conf) in enumerate(landmarks):
        color = (0, 255, 0) if conf > 0.5 else (0, 0, 255)
        cv2.circle(debug_image, (int(x), int(y)), 5, color, -1)
        cv2.putText(debug_image, f"{i}:{conf:.2f}", (int(x), int(y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # 显示角度信息
    if angles:
        y_offset = 30
        for angle_name, angle_value in angles.items():
            cv2.putText(debug_image, f"{angle_name}: {angle_value:.1f}°", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 25
    
    return debug_image
```

### 2. 性能基准测试
```python
def benchmark_pose_estimation(estimator, test_images, iterations=100):
    """姿态估计性能测试"""
    import time
    
    total_time = 0
    successful_detections = 0
    
    for _ in range(iterations):
        for image in test_images:
            start_time = time.time()
            
            if isinstance(estimator, PoseEstimator):
                _, landmarks, _ = estimator.detect_pose(image, draw=False)
                success = len(landmarks) > 0
            elif isinstance(estimator, YOLOPoseEstimator):
                detections = estimator.detect_pose(image)
                success = len(detections) > 0
            
            end_time = time.time()
            total_time += (end_time - start_time)
            
            if success:
                successful_detections += 1
    
    total_frames = iterations * len(test_images)
    avg_time = total_time / total_frames
    detection_rate = successful_detections / total_frames
    
    print(f"Average processing time: {avg_time:.3f}s")
    print(f"Detection rate: {detection_rate:.2%}")
    print(f"FPS: {1/avg_time:.1f}")
    
    return {
        'avg_time': avg_time,
        'detection_rate': detection_rate,
        'fps': 1/avg_time
    }
```

## 最佳实践总结

### 1. 模型选择策略
- **实时应用**: MediaPipe Pose (移动端) / YOLO Pose (服务器端)
- **高精度需求**: HRNet / OpenPose
- **多人场景**: YOLO Pose + 跟踪算法
- **边缘设备**: MoveNet / BlazePose

### 2. 性能优化建议
- **预处理**: 图像缩放、ROI裁剪
- **后处理**: 关键点平滑、异常值过滤
- **并行处理**: 多线程、GPU加速
- **自适应**: 动态调整处理频率

### 3. 质量保证
- **置信度阈值**: 根据应用场景调整
- **时间一致性**: 跨帧平滑处理
- **异常检测**: 识别和处理异常姿态

---

**更新日期**: 2024-01-XX  
**维护者**: YOLOS项目团队  
**版本**: 1.0