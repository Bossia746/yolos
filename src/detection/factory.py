#!/usr/bin/env python3
"""
检测器工厂类
提供统一的检测器创建接口
"""

from typing import Dict, Any, Optional
try:
    from .realtime_detector import RealtimeDetector
except ImportError:
    RealtimeDetector = None

try:
    from .image_detector import ImageDetector
except ImportError:
    ImageDetector = None

try:
    from .video_detector import VideoDetector
except ImportError:
    VideoDetector = None

try:
    from .camera_detector import CameraDetector
except ImportError:
    CameraDetector = None

try:
    from .enhanced_realtime_detector import EnhancedRealtimeDetector
except ImportError:
    EnhancedRealtimeDetector = None

try:
    from ..tracking.tracking_integration import IntegratedTrackingConfig, TrackingMode
except ImportError:
    IntegratedTrackingConfig = None
    TrackingMode = None

try:
    from ..tracking.multi_object_tracker import TrackingConfig, TrackingStrategy
except ImportError:
    TrackingConfig = None
    TrackingStrategy = None

class DetectorFactory:
    """检测器工厂类"""
    
    @classmethod
    def _get_detector_registry(cls):
        """动态获取检测器注册表"""
        registry = {}
        
        if RealtimeDetector is not None:
            registry['realtime'] = RealtimeDetector
            registry['yolo'] = RealtimeDetector  # 默认使用实时检测器
        
        if ImageDetector is not None:
            registry['image'] = ImageDetector
        
        if VideoDetector is not None:
            registry['video'] = VideoDetector
        
        if CameraDetector is not None:
            registry['camera'] = CameraDetector
        
        if EnhancedRealtimeDetector is not None:
            registry['yolov11'] = EnhancedRealtimeDetector  # YOLOv11增强检测器
            registry['enhanced'] = EnhancedRealtimeDetector  # 增强版检测器
        
        return registry
    
    # 保持向后兼容的静态注册表
    _detector_registry = {}
    
    @classmethod
    def create_detector(cls, detector_type: str, config: Dict[str, Any]):
        """
        创建检测器实例
        
        Args:
            detector_type: 检测器类型
            config: 配置参数
            
        Returns:
            检测器实例
        """
        registry = cls._get_detector_registry()
        registry.update(cls._detector_registry)  # 合并自定义注册表
        
        if detector_type not in registry:
            available_types = list(registry.keys())
            raise ValueError(f"不支持的检测器类型: {detector_type}，可用类型: {available_types}")
        
        detector_class = registry[detector_type]
        
        # 检查检测器类是否可用
        if detector_class is None:
            raise RuntimeError(f"检测器类 {detector_type} 不可用，可能缺少依赖")
        
        try:
            # 根据不同检测器类型传递不同参数
            if detector_type in ['realtime', 'yolo']:
                return detector_class(
                    model_type=config.get('model_type', 'yolov8'),
                    model_path=config.get('model_path'),
                    device=config.get('device', 'auto'),
                    enable_tracking=config.get('enable_tracking', True),
                    tracking_config=config.get('tracking_config')
                )
            elif detector_type in ['yolov11', 'enhanced']:
                # 增强检测器使用配置对象
                try:
                    from ..models.optimized_yolov11_system import OptimizationConfig
                    opt_config = OptimizationConfig(
                        model_size=config.get('model_size', 's'),
                        device=config.get('device', 'auto'),
                        confidence_threshold=config.get('confidence_threshold', 0.25),
                        iou_threshold=config.get('iou_threshold', 0.45),
                        target_fps=config.get('target_fps', 30.0),
                        platform=config.get('platform', 'pc'),
                        adaptive_inference=config.get('adaptive_inference', True),
                        edge_optimization=config.get('edge_optimization', False)
                    )
                    return detector_class(opt_config)
                except ImportError:
                    # 如果优化配置不可用，使用基本参数
                    return detector_class(
                        model_size=config.get('model_size', 's'),
                        device=config.get('device', 'auto'),
                        **{k: v for k, v in config.items() if k not in ['model_size', 'device']}
                    )
            elif detector_type == 'camera':
                return detector_class(
                    model_type=config.get('model_type', 'yolov8'),
                    model_path=config.get('model_path'),
                    device=config.get('device', 'auto')
                )
            elif detector_type in ['image', 'video']:
                return detector_class(
                    model_type=config.get('model_type', 'yolov8'),
                    model_path=config.get('model_path'),
                    device=config.get('device', 'auto'),
                    enable_tracking=config.get('enable_tracking', True),
                    tracking_config=config.get('tracking_config')
                )
            else:
                return detector_class(**config)
                
        except Exception as e:
            raise RuntimeError(f"创建检测器失败: {e}")
    
    @classmethod
    def register_detector(cls, name: str, detector_class):
        """
        注册新的检测器类型
        
        Args:
            name: 检测器名称
            detector_class: 检测器类
        """
        cls._detector_registry[name] = detector_class
    
    @classmethod
    def get_available_detectors(cls):
        """
        获取可用的检测器类型
        
        Returns:
            可用检测器类型列表
        """
        registry = cls._get_detector_registry()
        registry.update(cls._detector_registry)  # 合并自定义注册表
        return list(registry.keys())
    
    @classmethod
    def create_tracking_config(cls, mode: str = 'enhanced', **kwargs):
        """创建跟踪配置"""
        # 如果跟踪模块不可用，返回简单配置
        if IntegratedTrackingConfig is None or TrackingMode is None or TrackingConfig is None:
            return {
                'mode': mode,
                'enabled': kwargs.get('enabled', True),
                'max_missing_frames': kwargs.get('max_missing_frames', 10),
                'iou_threshold': kwargs.get('iou_threshold', 0.3),
                'distance_threshold': kwargs.get('distance_threshold', 100.0),
                'max_tracks': kwargs.get('max_tracks', 100)
            }
        
        mode_mapping = {
            'disabled': TrackingMode.DISABLED,
            'basic': TrackingMode.BASIC,
            'enhanced': TrackingMode.ENHANCED,
            'temporal': TrackingMode.TEMPORAL,
            'full': TrackingMode.FULL
        }
        
        tracking_mode = mode_mapping.get(mode, TrackingMode.ENHANCED)
        
        # 创建跟踪器配置
        tracking_config = TrackingConfig(
            strategy=TrackingStrategy.HYBRID,
            max_missing_frames=kwargs.get('max_missing_frames', 10),
            iou_threshold=kwargs.get('iou_threshold', 0.3),
            distance_threshold=kwargs.get('distance_threshold', 100.0),
            max_tracks=kwargs.get('max_tracks', 100)
        )
        
        # 创建集成配置
        integrated_config = IntegratedTrackingConfig(
            mode=tracking_mode,
            tracking_config=tracking_config,
            enable_feature_extraction=kwargs.get('enable_feature_extraction', True),
            platform_optimization=kwargs.get('platform_optimization', True),
            max_concurrent_tracks=kwargs.get('max_concurrent_tracks', 50)
        )
        
        return integrated_config
    
    @classmethod
    def create_detector_with_tracking(cls, detector_type: str, tracking_mode: str = 'enhanced', **kwargs):
        """创建带跟踪功能的检测器"""
        # 创建跟踪配置
        tracking_config = cls.create_tracking_config(tracking_mode, **kwargs)
        
        # 创建检测器配置
        config = kwargs.copy()
        config['enable_tracking'] = True
        config['tracking_config'] = tracking_config
        
        # 创建检测器
        return cls.create_detector(detector_type, config)
    
    @classmethod
    def list_available(cls):
        """列出可用检测器类型（兼容性方法）"""
        return cls.get_available_detectors()
    
    @classmethod
    def get_available(cls):
        """获取可用检测器类型（兼容性方法）"""
        return cls.get_available_detectors()
    
    @classmethod
    def list_types(cls):
        """列出检测器类型（兼容性方法）"""
        return cls.get_available_detectors()
    
    @classmethod
    def get_types(cls):
        """获取检测器类型（兼容性方法）"""
        return cls.get_available_detectors()