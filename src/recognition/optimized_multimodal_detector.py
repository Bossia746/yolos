"""优化版多模态识别检测器 - 集成所有优化算法，解决数据乱码问题"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import time
import threading
from queue import Queue, Empty
import logging
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
import locale

# 配置日志和编码
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置系统编码为UTF-8
if sys.platform.startswith('win'):
    import codecs
    import io
    # 检查是否已经是文本流
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# 导入优化版识别器
try:
    from .optimized_face_recognizer import OptimizedFaceRecognizer, create_optimized_face_recognizer
    OPTIMIZED_FACE_AVAILABLE = True
except ImportError:
    OPTIMIZED_FACE_AVAILABLE = False
    logger.warning("优化版面部识别器不可用")

try:
    from .optimized_gesture_recognizer import OptimizedGestureRecognizer, create_optimized_gesture_recognizer
    OPTIMIZED_GESTURE_AVAILABLE = True
except ImportError:
    OPTIMIZED_GESTURE_AVAILABLE = False
    logger.warning("优化版手势识别器不可用")

try:
    from .optimized_pose_recognizer import OptimizedPoseRecognizer, create_optimized_pose_recognizer
    OPTIMIZED_POSE_AVAILABLE = True
except ImportError:
    OPTIMIZED_POSE_AVAILABLE = False
    logger.warning("优化版姿势识别器不可用")

try:
    from .optimized_fall_detector import OptimizedFallDetector, create_optimized_fall_detector
    OPTIMIZED_FALL_AVAILABLE = True
except ImportError:
    OPTIMIZED_FALL_AVAILABLE = False
    logger.warning("优化版摔倒检测器不可用")

# 备用导入
try:
    from .enhanced_face_recognizer import EnhancedFaceRecognizer
    from .enhanced_gesture_recognizer import EnhancedGestureRecognizer
    from .yolov7_pose_recognizer import YOLOv7PoseRecognizer
    from .fall_detector import FallDetector
    FALLBACK_AVAILABLE = True
except ImportError:
    FALLBACK_AVAILABLE = False
    logger.warning("备用识别器不可用")


@dataclass
class DetectionResult:
    """检测结果数据类"""
    timestamp: float
    frame_id: int
    face_results: Dict[str, Any]
    gesture_results: Any
    pose_results: Dict[str, Any]
    fall_results: Dict[str, Any]
    performance_metrics: Dict[str, float]
    encoding_info: Dict[str, str]
    error_info: Dict[str, str]


@dataclass
class SystemStatus:
    """系统状态数据类"""
    total_frames: int
    processed_frames: int
    fps: float
    avg_processing_time: float
    memory_usage: float
    cpu_usage: float
    active_algorithms: List[str]
    error_count: int
    last_error: str


class OptimizedMultimodalDetector:
    """优化版多模态识别检测器 - 高性能、低延迟、数据无乱码"""
    
    def __init__(self,
                 enable_face: bool = True,
                 enable_gesture: bool = True,
                 enable_pose: bool = True,
                 enable_fall_detection: bool = True,
                 face_database_path: Optional[str] = None,
                 detection_interval: int = 1,
                 max_workers: int = 4,
                 enable_async_processing: bool = True,
                 enable_result_caching: bool = True,
                 cache_size: int = 100,
                 encoding: str = 'utf-8',
                 performance_monitoring: bool = True,
                 use_insightface: bool = True):
        """
        初始化优化版多模态识别检测器
        
        Args:
            enable_face: 启用面部识别
            enable_gesture: 启用手势识别
            enable_pose: 启用身体姿势识别
            enable_fall_detection: 启用摔倒检测
            face_database_path: 人脸数据库路径
            detection_interval: 检测间隔（帧数）
            max_workers: 最大工作线程数
            enable_async_processing: 启用异步处理
            enable_result_caching: 启用结果缓存
            cache_size: 缓存大小
            encoding: 字符编码
            performance_monitoring: 启用性能监控
        """
        self.enable_face = enable_face
        self.enable_gesture = enable_gesture
        self.enable_pose = enable_pose
        self.enable_fall_detection = enable_fall_detection
        self.use_insightface = use_insightface
        self.detection_interval = detection_interval
        self.max_workers = max_workers
        self.enable_async_processing = enable_async_processing
        self.enable_result_caching = enable_result_caching
        self.cache_size = cache_size
        self.encoding = encoding
        self.performance_monitoring = performance_monitoring
        
        # 设置编码
        self._setup_encoding()
        
        # 初始化识别器
        self.face_recognizer = None
        self.gesture_recognizer = None
        self.pose_recognizer = None
        self.fall_detector = None
        
        # 初始化各个识别器
        self._init_recognizers(face_database_path)
        
        # 线程池
        if enable_async_processing:
            self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        else:
            self.thread_pool = None
        
        # 结果缓存
        self.result_cache = {} if enable_result_caching else None
        self.cache_keys = [] if enable_result_caching else None
        
        # 性能统计
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'face_detections': 0,
            'gesture_detections': 0,
            'pose_detections': 0,
            'fall_detections': 0,
            'processing_times': [],
            'fps_history': [],
            'error_count': 0,
            'last_error': '',
            'memory_usage': 0.0,
            'cpu_usage': 0.0
        }
        
        # 回调函数
        self.detection_callback = None
        self.frame_callback = None
        self.error_callback = None
        
        # 状态管理
        self.frame_count = 0
        self.start_time = time.time()
        self.is_running = False
        self.lock = threading.Lock()
        
        # 错误处理
        self.error_queue = Queue(maxsize=100)
        
        logger.info("优化版多模态识别检测器初始化完成")
    
    def _setup_encoding(self):
        """设置字符编码"""
        try:
            # 设置默认编码
            if hasattr(sys, 'setdefaultencoding'):
                sys.setdefaultencoding(self.encoding)
            
            # 设置locale
            try:
                locale.setlocale(locale.LC_ALL, '')
            except locale.Error:
                logger.warning("无法设置locale")
            
            # 确保OpenCV支持中文
            cv2.setUseOptimized(True)
            
            logger.info(f"字符编码设置为: {self.encoding}")
            
        except Exception as e:
            logger.error(f"编码设置失败: {e}")
    
    def _init_recognizers(self, face_database_path: Optional[str]):
        """初始化识别器"""
        try:
            # 初始化面部识别器
            if self.enable_face:
                if OPTIMIZED_FACE_AVAILABLE:
                    self.face_recognizer = create_optimized_face_recognizer(
                        face_database_path=face_database_path,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.5,
                        use_insightface=self.use_insightface,
                        max_faces=10,
                        enable_threading=True,
                        encoding='utf-8'
                    )
                    logger.info("使用优化版面部识别器")
                elif FALLBACK_AVAILABLE:
                    self.face_recognizer = EnhancedFaceRecognizer(face_database_path=face_database_path)
                    logger.info("使用增强版面部识别器")
                else:
                    logger.error("面部识别器不可用")
            
            # 初始化手势识别器
            if self.enable_gesture:
                if OPTIMIZED_GESTURE_AVAILABLE:
                    self.gesture_recognizer = create_optimized_gesture_recognizer(
                        max_num_hands=4,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.5,
                        sequence_length=30,
                        enable_threading=True,
                        enable_dynamic_gestures=True,
                        enable_interaction_detection=True
                    )
                    logger.info("使用优化版手势识别器")
                elif FALLBACK_AVAILABLE:
                    self.gesture_recognizer = EnhancedGestureRecognizer()
                    logger.info("使用增强版手势识别器")
                else:
                    logger.error("手势识别器不可用")
            
            # 初始化姿势识别器
            if self.enable_pose:
                if OPTIMIZED_POSE_AVAILABLE:
                    self.pose_recognizer = create_optimized_pose_recognizer(
                        max_num_persons=10,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.5,
                        model_complexity=1,
                        enable_yolo=True,
                        enable_mediapipe=True,
                        enable_threading=True,
                        enable_tracking=True,
                        enable_activity_recognition=True,
                        sequence_length=30
                    )
                    logger.info("使用优化版姿势识别器")
                elif FALLBACK_AVAILABLE:
                    self.pose_recognizer = YOLOv7PoseRecognizer()
                    logger.info("使用YOLOv7姿势识别器")
                else:
                    logger.error("姿势识别器不可用")
            
            # 初始化摔倒检测器
            if self.enable_fall_detection:
                if OPTIMIZED_FALL_AVAILABLE:
                    self.fall_detector = create_optimized_fall_detector(
                        sensitivity=0.7,
                        min_fall_duration=0.5,
                        max_false_positive_rate=0.05,
                        enable_multi_stage=True,
                        enable_velocity_analysis=True,
                        enable_pose_analysis=True,
                        enable_environmental_analysis=True,
                        sequence_length=30,
                        alert_delay=1.0
                    )
                    logger.info("使用优化版摔倒检测器")
                elif FALLBACK_AVAILABLE:
                    self.fall_detector = FallDetector()
                    logger.info("使用基础摔倒检测器")
                else:
                    logger.error("摔倒检测器不可用")
            
        except Exception as e:
            logger.error(f"识别器初始化失败: {e}")
            self._handle_error("识别器初始化", str(e))
    
    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, DetectionResult]:
        """
        执行多模态识别检测
        
        Args:
            image: 输入图像
            
        Returns:
            Tuple[np.ndarray, DetectionResult]: 标注后的图像和检测结果
        """
        start_time = time.time()
        
        with self.lock:
            self.frame_count += 1
            self.stats['total_frames'] += 1
        
        try:
            # 检查缓存
            cache_key = None
            if self.enable_result_caching:
                cache_key = self._generate_cache_key(image)
                if cache_key in self.result_cache:
                    cached_result = self.result_cache[cache_key]
                    logger.debug("使用缓存结果")
                    return cached_result
            
            # 初始化结果
            face_results = {}
            gesture_results = {}
            pose_results = {}
            fall_results = {}
            error_info = {}
            
            annotated_image = image.copy()
            
            # 根据检测间隔决定是否进行检测
            if self.frame_count % self.detection_interval == 0:
                with self.lock:
                    self.stats['processed_frames'] += 1
                
                if self.enable_async_processing and self.thread_pool:
                    # 异步处理
                    futures = []
                    
                    if self.enable_face and self.face_recognizer:
                        future = self.thread_pool.submit(self._detect_faces, image)
                        futures.append(('face', future))
                    
                    if self.enable_gesture and self.gesture_recognizer:
                        future = self.thread_pool.submit(self._detect_gestures, image)
                        futures.append(('gesture', future))
                    
                    if self.enable_pose and self.pose_recognizer:
                        future = self.thread_pool.submit(self._detect_poses, image)
                        futures.append(('pose', future))
                    
                    # 收集结果
                    for detection_type, future in futures:
                        try:
                            result = future.result(timeout=1.0)  # 1秒超时
                            if detection_type == 'face':
                                face_results = result
                            elif detection_type == 'gesture':
                                gesture_results = result
                            elif detection_type == 'pose':
                                pose_results = result
                        except Exception as e:
                            error_info[detection_type] = str(e)
                            logger.error(f"{detection_type}检测异步处理错误: {e}")
                
                else:
                    # 同步处理
                    if self.enable_face and self.face_recognizer:
                        face_results = self._detect_faces(image)
                    
                    if self.enable_gesture and self.gesture_recognizer:
                        gesture_results = self._detect_gestures(image)
                    
                    if self.enable_pose and self.pose_recognizer:
                        pose_results = self._detect_poses(image)
                
                # 摔倒检测（基于姿势结果）
                if self.enable_fall_detection and self.fall_detector and pose_results:
                    fall_results = self._detect_falls(pose_results)
                
                # 绘制标注
                annotated_image = self._draw_annotations(annotated_image, 
                                                       face_results, 
                                                       gesture_results, 
                                                       pose_results, 
                                                       fall_results)
                
                # 更新统计
                self._update_statistics(face_results, gesture_results, pose_results, fall_results)
            
            # 计算性能指标
            processing_time = time.time() - start_time
            performance_metrics = self._calculate_performance_metrics(processing_time)
            
            # 编码信息
            encoding_info = {
                'system_encoding': sys.getdefaultencoding(),
                'locale_encoding': locale.getpreferredencoding(),
                'opencv_version': cv2.__version__
            }
            
            # 创建检测结果
            detection_result = DetectionResult(
                timestamp=time.time(),
                frame_id=self.frame_count,
                face_results=face_results,
                gesture_results=gesture_results,
                pose_results=pose_results,
                fall_results=fall_results,
                performance_metrics=performance_metrics,
                encoding_info=encoding_info,
                error_info=error_info
            )
            
            # 缓存结果
            if self.enable_result_caching and cache_key:
                self._cache_result(cache_key, (annotated_image, detection_result))
            
            # 调用回调函数
            self._call_callbacks(annotated_image, detection_result)
            
            return annotated_image, detection_result
            
        except Exception as e:
            logger.error(f"检测过程错误: {e}")
            self._handle_error("检测过程", str(e))
            
            # 返回默认结果
            return image, self._get_default_result()
    
    def _detect_faces(self, image: np.ndarray) -> Dict[str, Any]:
        """检测面部"""
        try:
            if hasattr(self.face_recognizer, 'detect_faces'):
                if OPTIMIZED_FACE_AVAILABLE and isinstance(self.face_recognizer, OptimizedFaceRecognizer):
                    # OptimizedFaceRecognizer返回(annotated_image, detections)
                    annotated_image, detections = self.face_recognizer.detect_faces(image)
                    return {
                        'faces_detected': len(detections),
                        'detections': detections,
                        'annotated_image': annotated_image
                    }
                else:
                    # 适配其他面部识别器
                    result = self.face_recognizer.detect_faces(image)
                    if isinstance(result, tuple):
                        return {'faces_detected': len(result[1]) if len(result) > 1 else 0, 'detections': result[1] if len(result) > 1 else []}
                    return result if isinstance(result, dict) else {}
            return {'faces_detected': 0, 'detections': []}
        except Exception as e:
            logger.error(f"面部检测错误: {e}")
            return {'faces_detected': 0, 'detections': []}
    
    def _detect_gestures(self, image: np.ndarray) -> Any:
        """检测手势"""
        try:
            if hasattr(self.gesture_recognizer, 'detect_gestures'):
                if OPTIMIZED_GESTURE_AVAILABLE and isinstance(self.gesture_recognizer, OptimizedGestureRecognizer):
                    # 直接返回GestureRecognitionResult对象
                    result = self.gesture_recognizer.detect_gestures(image)
                    return result
                else:
                    # 适配其他手势识别器
                    result = self.gesture_recognizer.recognize_gestures(image)
                    if isinstance(result, tuple):
                        return {'gestures_detected': len(result[1]) if len(result) > 1 else 0, 'detections': result[1] if len(result) > 1 else []}
                    return result if isinstance(result, dict) else {'gestures_detected': 0, 'detections': []}
            return {'gestures_detected': 0, 'detections': []}
        except Exception as e:
            logger.error(f"手势检测错误: {e}")
            return {'gestures_detected': 0, 'detections': []}
    
    def _detect_poses(self, image: np.ndarray) -> Dict[str, Any]:
        """检测姿势"""
        try:
            if hasattr(self.pose_recognizer, 'detect_poses'):
                if OPTIMIZED_POSE_AVAILABLE and isinstance(self.pose_recognizer, OptimizedPoseRecognizer):
                    result = self.pose_recognizer.detect_poses(image)
                    # 确保返回Dict格式
                    if isinstance(result, tuple):
                        return {'poses_detected': len(result[1]) if len(result) > 1 else 0, 'detections': result[1] if len(result) > 1 else []}
                    return result if isinstance(result, dict) else {'poses_detected': 0, 'detections': []}
                else:
                    # 适配其他姿势识别器
                    result = self.pose_recognizer.detect_pose(image)
                    if isinstance(result, tuple):
                        return {'poses_detected': len(result[1]) if len(result) > 1 else 0, 'detections': result[1] if len(result) > 1 else []}
                    return result if isinstance(result, dict) else {'poses_detected': 0, 'detections': []}
            return {'poses_detected': 0, 'detections': []}
        except Exception as e:
            logger.error(f"姿势检测错误: {e}")
            return {'poses_detected': 0, 'detections': []}
    
    def _detect_falls(self, pose_results: Dict[str, Any]) -> Dict[str, Any]:
        """检测摔倒"""
        try:
            if hasattr(self.fall_detector, 'detect_falls'):
                if OPTIMIZED_FALL_AVAILABLE and isinstance(self.fall_detector, OptimizedFallDetector):
                    # 从姿势结果中提取人员数据
                    persons_data = self._extract_persons_data(pose_results)
                    if len(persons_data) > 0:
                        result = self.fall_detector.detect_falls(persons_data)
                        return asdict(result) if hasattr(result, '__dict__') else result
                else:
                    # 适配其他摔倒检测器
                    if 'persons' in pose_results and pose_results['persons']:
                        person_data = pose_results['persons'][0]
                        keypoints = person_data.get('keypoints', [])
                        bbox = person_data.get('bbox')
                        return self.fall_detector.detect_fall(keypoints, bbox)
            return {}
        except Exception as e:
            logger.error(f"摔倒检测错误: {e}")
            return {}
    
    def _extract_persons_data(self, pose_results: Dict[str, Any]) -> List[Any]:
        """从姿势结果中提取人员数据"""
        try:
            persons_data = []
            if 'persons' in pose_results:
                for i, person in enumerate(pose_results['persons']):
                    # 创建简单的人员数据对象
                    class PersonData:
                        def __init__(self, person_id, bbox, keypoints, pose_type='unknown', confidence=0.5):
                            self.person_id = person_id
                            self.bbox = bbox
                            self.keypoints = keypoints
                            self.pose_type = pose_type
                            self.confidence = confidence
                            self.stability_score = confidence
                    
                    person_data = PersonData(
                        person_id=i,
                        bbox=person.get('bbox', (0, 0, 0, 0)),
                        keypoints=person.get('keypoints', []),
                        pose_type=person.get('pose_type', 'unknown'),
                        confidence=person.get('confidence', 0.5)
                    )
                    persons_data.append(person_data)
            
            return persons_data
        except Exception as e:
            logger.error(f"人员数据提取错误: {e}")
            return []
    
    def _draw_annotations(self, image: np.ndarray, 
                        face_results: Dict[str, Any],
                        gesture_results: Any,
                        pose_results: Dict[str, Any],
                        fall_results: Dict[str, Any]) -> np.ndarray:
        """绘制所有标注"""
        try:
            annotated_image = image.copy()
            
            # 绘制面部标注
            if face_results and hasattr(self.face_recognizer, 'draw_annotations'):
                try:
                    annotated_image = self.face_recognizer.draw_annotations(annotated_image, face_results)
                except Exception as e:
                    logger.error(f"面部标注绘制错误: {e}")
            
            # 绘制手势标注
            if gesture_results and hasattr(self.gesture_recognizer, 'draw_annotations'):
                try:
                    annotated_image = self.gesture_recognizer.draw_annotations(annotated_image, gesture_results)
                except Exception as e:
                    logger.error(f"手势标注绘制错误: {e}")
            
            # 绘制姿势标注
            if pose_results and hasattr(self.pose_recognizer, 'draw_annotations'):
                try:
                    annotated_image = self.pose_recognizer.draw_annotations(annotated_image, pose_results)
                except Exception as e:
                    logger.error(f"姿势标注绘制错误: {e}")
            
            # 绘制摔倒警告
            if fall_results:
                try:
                    self._draw_fall_alerts(annotated_image, fall_results)
                except Exception as e:
                    logger.error(f"摔倒警告绘制错误: {e}")
            
            # 绘制系统信息
            self._draw_system_info(annotated_image)
            
            return annotated_image
            
        except Exception as e:
            logger.error(f"标注绘制错误: {e}")
            return image
    
    def _draw_fall_alerts(self, image: np.ndarray, fall_results: Dict[str, Any]):
        """绘制摔倒警告"""
        try:
            if isinstance(fall_results, dict):
                # 处理优化版摔倒检测结果
                if 'fall_detections' in fall_results:
                    for detection in fall_results['fall_detections']:
                        if detection.get('is_falling', False):
                            # 绘制警告文本
                            alert_text = f"摔倒警告! 置信度: {detection.get('fall_confidence', 0):.2f}"
                            cv2.putText(image, alert_text.encode('utf-8').decode('utf-8'), 
                                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                            
                            # 绘制边界框
                            bbox = detection.get('bbox', (0, 0, 0, 0))
                            if bbox != (0, 0, 0, 0):
                                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
                
                elif fall_results.get('fall_detected', False):
                    # 处理基础摔倒检测结果
                    alert_text = f"摔倒检测! 置信度: {fall_results.get('fall_confidence', 0):.2f}"
                    cv2.putText(image, alert_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        except Exception as e:
            logger.error(f"摔倒警告绘制错误: {e}")
    
    def _draw_system_info(self, image: np.ndarray):
        """绘制系统信息"""
        try:
            if not self.performance_monitoring:
                return
            
            # 计算FPS
            elapsed_time = time.time() - self.start_time
            fps = self.stats['total_frames'] / elapsed_time if elapsed_time > 0 else 0
            
            # 绘制信息
            info_lines = [
                f"FPS: {fps:.1f}",
                f"帧数: {self.stats['total_frames']}",
                f"处理帧: {self.stats['processed_frames']}",
                f"错误: {self.stats['error_count']}"
            ]
            
            y_offset = image.shape[0] - 100
            for i, line in enumerate(info_lines):
                cv2.putText(image, line, (10, y_offset + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        except Exception as e:
            logger.error(f"系统信息绘制错误: {e}")
    
    def _update_statistics(self, face_results: Dict[str, Any],
                          gesture_results: Any,
                          pose_results: Dict[str, Any],
                          fall_results: Dict[str, Any]):
        """更新统计信息"""
        try:
            with self.lock:
                # 面部检测统计
                if face_results:
                    face_count = face_results.get('faces_detected', 0)
                    if face_count > 0:
                        self.stats['face_detections'] += face_count
                
                # 手势检测统计
                if gesture_results:
                    if hasattr(gesture_results, 'hands_detected'):
                        # GestureRecognitionResult对象
                        gesture_count = gesture_results.hands_detected
                    else:
                        # Dict格式
                        gesture_count = gesture_results.get('hands_detected', 0)
                    if gesture_count > 0:
                        self.stats['gesture_detections'] += gesture_count
                
                # 姿势检测统计
                if pose_results:
                    pose_count = pose_results.get('persons_detected', 0)
                    if pose_count > 0:
                        self.stats['pose_detections'] += pose_count
                
                # 摔倒检测统计
                if fall_results:
                    if isinstance(fall_results, dict):
                        if fall_results.get('active_falls', 0) > 0:
                            self.stats['fall_detections'] += fall_results['active_falls']
                        elif fall_results.get('fall_detected', False):
                            self.stats['fall_detections'] += 1
        
        except Exception as e:
            logger.error(f"统计更新错误: {e}")
    
    def _calculate_performance_metrics(self, processing_time: float) -> Dict[str, float]:
        """计算性能指标"""
        try:
            with self.lock:
                self.stats['processing_times'].append(processing_time)
                
                # 保持最近100个处理时间
                if len(self.stats['processing_times']) > 100:
                    self.stats['processing_times'] = self.stats['processing_times'][-100:]
                
                # 计算平均处理时间
                avg_processing_time = sum(self.stats['processing_times']) / len(self.stats['processing_times'])
                
                # 计算FPS
                elapsed_time = time.time() - self.start_time
                fps = self.stats['total_frames'] / elapsed_time if elapsed_time > 0 else 0
                
                self.stats['fps_history'].append(fps)
                if len(self.stats['fps_history']) > 100:
                    self.stats['fps_history'] = self.stats['fps_history'][-100:]
                
                return {
                    'processing_time': processing_time,
                    'avg_processing_time': avg_processing_time,
                    'fps': fps,
                    'avg_fps': sum(self.stats['fps_history']) / len(self.stats['fps_history']),
                    'memory_usage': self.stats['memory_usage'],
                    'cpu_usage': self.stats['cpu_usage']
                }
        
        except Exception as e:
            logger.error(f"性能指标计算错误: {e}")
            return {'processing_time': processing_time, 'fps': 0.0}
    
    def _generate_cache_key(self, image: np.ndarray) -> str:
        """生成缓存键"""
        try:
            # 使用图像哈希作为缓存键
            image_hash = hash(image.tobytes())
            return f"{image_hash}_{self.frame_count % 10}"
        except Exception:
            return f"frame_{self.frame_count}"
    
    def _cache_result(self, cache_key: str, result: Tuple[np.ndarray, DetectionResult]):
        """缓存结果"""
        try:
            if len(self.result_cache) >= self.cache_size:
                # 移除最旧的缓存
                oldest_key = self.cache_keys.pop(0)
                del self.result_cache[oldest_key]
            
            self.result_cache[cache_key] = result
            self.cache_keys.append(cache_key)
        
        except Exception as e:
            logger.error(f"结果缓存错误: {e}")
    
    def _call_callbacks(self, annotated_image: np.ndarray, detection_result: DetectionResult):
        """调用回调函数"""
        try:
            if self.detection_callback:
                self.detection_callback(annotated_image, detection_result)
            
            if self.frame_callback:
                self.frame_callback(annotated_image)
        
        except Exception as e:
            logger.error(f"回调函数调用错误: {e}")
            self._handle_error("回调函数", str(e))
    
    def _handle_error(self, context: str, error_message: str):
        """处理错误"""
        try:
            with self.lock:
                self.stats['error_count'] += 1
                self.stats['last_error'] = f"{context}: {error_message}"
            
            # 添加到错误队列
            try:
                self.error_queue.put_nowait({
                    'timestamp': time.time(),
                    'context': context,
                    'message': error_message
                })
            except:
                pass  # 队列满时忽略
            
            # 调用错误回调
            if self.error_callback:
                try:
                    self.error_callback(context, error_message)
                except:
                    pass  # 避免回调函数错误导致的递归
        
        except Exception:
            pass  # 避免错误处理本身出错
    
    def _get_default_result(self) -> DetectionResult:
        """获取默认结果"""
        return DetectionResult(
            timestamp=time.time(),
            frame_id=self.frame_count,
            face_results={},
            gesture_results={},
            pose_results={},
            fall_results={},
            performance_metrics={'processing_time': 0.0, 'fps': 0.0},
            encoding_info={'system_encoding': sys.getdefaultencoding()},
            error_info={}
        )
    
    def detect_from_camera(self, 
                          camera_id: int = 0,
                          display: bool = True,
                          save_video: Optional[str] = None,
                          window_name: str = "优化版多模态识别") -> None:
        """
        从摄像头进行实时多模态识别
        
        Args:
            camera_id: 摄像头ID
            display: 是否显示结果
            save_video: 保存视频路径
            window_name: 窗口名称
        """
        cap = None
        video_writer = None
        
        try:
            # 打开摄像头
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开摄像头 {camera_id}")
            
            # 设置摄像头参数
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # 获取实际参数
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"摄像头参数: {width}x{height} @ {fps}fps")
            
            # 视频写入器
            if save_video:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(save_video, fourcc, 20.0, (width, height))
                logger.info(f"开始录制视频: {save_video}")
            
            # 创建窗口
            if display:
                cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            
            self.is_running = True
            logger.info("开始实时检测...")
            
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("无法读取摄像头帧")
                    break
                
                # 执行检测
                annotated_frame, detection_result = self.detect(frame)
                
                # 显示结果
                if display:
                    cv2.imshow(window_name, annotated_frame)
                    
                    # 检查按键
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' 或 ESC
                        break
                    elif key == ord('s'):  # 's' 保存截图
                        screenshot_path = f"screenshot_{int(time.time())}.jpg"
                        cv2.imwrite(screenshot_path, annotated_frame)
                        logger.info(f"截图已保存: {screenshot_path}")
                    elif key == ord('r'):  # 'r' 重置统计
                        self.reset_statistics()
                        logger.info("统计信息已重置")
                
                # 保存视频
                if video_writer:
                    video_writer.write(annotated_frame)
                
                # 性能监控
                if self.performance_monitoring and self.frame_count % 100 == 0:
                    self._log_performance_stats()
        
        except KeyboardInterrupt:
            logger.info("用户中断检测")
        except Exception as e:
            logger.error(f"摄像头检测错误: {e}")
            self._handle_error("摄像头检测", str(e))
        
        finally:
            # 清理资源
            self.is_running = False
            
            if cap:
                cap.release()
            
            if video_writer:
                video_writer.release()
            
            if display:
                cv2.destroyAllWindows()
            
            logger.info("摄像头检测已停止")
    
    def _log_performance_stats(self):
        """记录性能统计"""
        try:
            elapsed_time = time.time() - self.start_time
            fps = self.stats['total_frames'] / elapsed_time if elapsed_time > 0 else 0
            
            logger.info(f"性能统计 - FPS: {fps:.1f}, 总帧数: {self.stats['total_frames']}, "
                       f"处理帧: {self.stats['processed_frames']}, 错误: {self.stats['error_count']}")
        
        except Exception as e:
            logger.error(f"性能统计记录错误: {e}")
    
    def set_callbacks(self, 
                     detection_callback: Optional[Callable] = None,
                     frame_callback: Optional[Callable] = None,
                     error_callback: Optional[Callable] = None):
        """
        设置回调函数
        
        Args:
            detection_callback: 检测结果回调函数
            frame_callback: 帧处理回调函数
            error_callback: 错误处理回调函数
        """
        self.detection_callback = detection_callback
        self.frame_callback = frame_callback
        self.error_callback = error_callback
        logger.info("回调函数已设置")
    
    def get_system_status(self) -> SystemStatus:
        """获取系统状态"""
        try:
            elapsed_time = time.time() - self.start_time
            fps = self.stats['total_frames'] / elapsed_time if elapsed_time > 0 else 0
            
            avg_processing_time = 0.0
            if self.stats['processing_times']:
                avg_processing_time = sum(self.stats['processing_times']) / len(self.stats['processing_times'])
            
            active_algorithms = []
            if self.enable_face and self.face_recognizer:
                active_algorithms.append('面部识别')
            if self.enable_gesture and self.gesture_recognizer:
                active_algorithms.append('手势识别')
            if self.enable_pose and self.pose_recognizer:
                active_algorithms.append('姿势识别')
            if self.enable_fall_detection and self.fall_detector:
                active_algorithms.append('摔倒检测')
            
            return SystemStatus(
                total_frames=self.stats['total_frames'],
                processed_frames=self.stats['processed_frames'],
                fps=fps,
                avg_processing_time=avg_processing_time,
                memory_usage=self.stats['memory_usage'],
                cpu_usage=self.stats['cpu_usage'],
                active_algorithms=active_algorithms,
                error_count=self.stats['error_count'],
                last_error=self.stats['last_error']
            )
        
        except Exception as e:
            logger.error(f"系统状态获取错误: {e}")
            return SystemStatus(
                total_frames=0, processed_frames=0, fps=0.0, avg_processing_time=0.0,
                memory_usage=0.0, cpu_usage=0.0, active_algorithms=[], 
                error_count=1, last_error=str(e)
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取详细统计信息"""
        with self.lock:
            return {
                'performance': self.stats.copy(),
                'system_status': asdict(self.get_system_status()),
                'configuration': {
                    'enable_face': self.enable_face,
                    'enable_gesture': self.enable_gesture,
                    'enable_pose': self.enable_pose,
                    'enable_fall_detection': self.enable_fall_detection,
                    'detection_interval': self.detection_interval,
                    'max_workers': self.max_workers,
                    'enable_async_processing': self.enable_async_processing,
                    'encoding': self.encoding
                }
            }
    
    def reset_statistics(self):
        """重置统计信息"""
        with self.lock:
            self.stats = {
                'total_frames': 0,
                'processed_frames': 0,
                'face_detections': 0,
                'gesture_detections': 0,
                'pose_detections': 0,
                'fall_detections': 0,
                'processing_times': [],
                'fps_history': [],
                'error_count': 0,
                'last_error': '',
                'memory_usage': 0.0,
                'cpu_usage': 0.0
            }
            
            self.frame_count = 0
            self.start_time = time.time()
            
            # 清空缓存
            if self.result_cache:
                self.result_cache.clear()
                self.cache_keys.clear()
            
            # 清空错误队列
            while not self.error_queue.empty():
                try:
                    self.error_queue.get_nowait()
                except Empty:
                    break
        
        logger.info("统计信息已重置")
    
    def stop(self):
        """停止检测"""
        self.is_running = False
        logger.info("检测已停止")
    
    def cleanup(self):
        """清理资源"""
        try:
            self.is_running = False
            
            # 关闭线程池
            if self.thread_pool:
                self.thread_pool.shutdown(wait=True)
            
            # 清理识别器
            if hasattr(self.face_recognizer, 'cleanup'):
                self.face_recognizer.cleanup()
            
            if hasattr(self.gesture_recognizer, 'cleanup'):
                self.gesture_recognizer.cleanup()
            
            if hasattr(self.pose_recognizer, 'cleanup'):
                self.pose_recognizer.cleanup()
            
            if hasattr(self.fall_detector, 'cleanup'):
                self.fall_detector.cleanup()
            
            logger.info("优化版多模态识别检测器资源清理完成")
        
        except Exception as e:
            logger.error(f"资源清理错误: {e}")


def create_optimized_multimodal_detector(**kwargs) -> OptimizedMultimodalDetector:
    """创建优化版多模态检测器的工厂函数"""
    return OptimizedMultimodalDetector(**kwargs)


# 预设配置
MULTIMODAL_DETECTOR_CONFIGS = {
    'high_performance': {
        'enable_face': True,
        'enable_gesture': True,
        'enable_pose': True,
        'enable_fall_detection': True,
        'detection_interval': 1,
        'max_workers': 6,
        'enable_async_processing': True,
        'enable_result_caching': True,
        'cache_size': 200,
        'performance_monitoring': True
    },
    'balanced': {
        'enable_face': True,
        'enable_gesture': True,
        'enable_pose': True,
        'enable_fall_detection': True,
        'detection_interval': 2,
        'max_workers': 4,
        'enable_async_processing': True,
        'enable_result_caching': True,
        'cache_size': 100,
        'performance_monitoring': True
    },
    'lightweight': {
        'enable_face': True,
        'enable_gesture': True,
        'enable_pose': True,
        'enable_fall_detection': False,
        'detection_interval': 1,
        'max_workers': 2,
        'enable_async_processing': False,
        'enable_result_caching': False,
        'cache_size': 0,
        'performance_monitoring': False
    },
    'low_resource': {
        'enable_face': True,
        'enable_gesture': False,
        'enable_pose': True,
        'enable_fall_detection': True,
        'detection_interval': 3,
        'max_workers': 2,
        'enable_async_processing': False,
        'enable_result_caching': False,
        'cache_size': 0,
        'performance_monitoring': False
    }
}


def create_multimodal_detector_from_config(config_name: str = 'balanced', **kwargs) -> OptimizedMultimodalDetector:
    """根据预设配置创建多模态检测器"""
    if config_name not in MULTIMODAL_DETECTOR_CONFIGS:
        logger.warning(f"未知配置: {config_name}，使用默认配置")
        config_name = 'balanced'
    
    config = MULTIMODAL_DETECTOR_CONFIGS[config_name].copy()
    config.update(kwargs)  # 允许覆盖配置
    
    return OptimizedMultimodalDetector(**config)