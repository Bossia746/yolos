#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相机工具函数
统一的相机初始化和管理功能
"""

import cv2
import logging
import time
from typing import Optional, Tuple, Union, Dict, Any
from ..core.types import CameraConfig, ImageInfo

logger = logging.getLogger(__name__)

class CameraManager:
    """相机管理器"""
    
    def __init__(self):
        self.cameras: Dict[str, cv2.VideoCapture] = {}
        self.configs: Dict[str, CameraConfig] = {}
    
    def initialize_camera(
        self, 
        camera_id: str = "default",
        config: Optional[CameraConfig] = None
    ) -> Optional[cv2.VideoCapture]:
        """
        初始化相机
        
        Args:
            camera_id: 相机标识符
            config: 相机配置，如果为None则使用默认配置
            
        Returns:
            cv2.VideoCapture对象，失败返回None
        """
        try:
            # 使用默认配置
            if config is None:
                config = CameraConfig()
            
            logger.info(f"初始化相机 {camera_id}...")
            
            # 创建VideoCapture对象
            cap = cv2.VideoCapture(config.device_id)
            
            if not cap.isOpened():
                logger.error(f"无法打开相机设备: {config.device_id}")
                return None
            
            # 设置相机参数
            self._configure_camera(cap, config)
            
            # 验证相机设置
            if not self._verify_camera(cap, config):
                logger.error("相机验证失败")
                cap.release()
                return None
            
            # 保存相机和配置
            self.cameras[camera_id] = cap
            self.configs[camera_id] = config
            
            logger.info(f"相机 {camera_id} 初始化成功")
            return cap
            
        except Exception as e:
            logger.error(f"相机初始化失败: {e}")
            return None
    
    def _configure_camera(self, cap: cv2.VideoCapture, config: CameraConfig):
        """配置相机参数"""
        try:
            # 设置分辨率
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
            
            # 设置帧率
            cap.set(cv2.CAP_PROP_FPS, config.fps)
            
            # 设置曝光
            if not config.auto_exposure and config.exposure >= 0:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 手动曝光
                cap.set(cv2.CAP_PROP_EXPOSURE, config.exposure)
            else:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # 自动曝光
            
            # 设置增益
            if config.gain >= 0:
                cap.set(cv2.CAP_PROP_GAIN, config.gain)
            
            # 设置缓冲区大小
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
        except Exception as e:
            logger.warning(f"设置相机参数时出现警告: {e}")
    
    def _verify_camera(self, cap: cv2.VideoCapture, config: CameraConfig) -> bool:
        """验证相机设置"""
        try:
            # 读取一帧测试
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.error("无法从相机读取帧")
                return False
            
            # 检查分辨率
            actual_height, actual_width = frame.shape[:2]
            if abs(actual_width - config.width) > 50 or abs(actual_height - config.height) > 50:
                logger.warning(f"相机分辨率不匹配: 期望{config.width}x{config.height}, 实际{actual_width}x{actual_height}")
            
            # 检查帧率
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            if abs(actual_fps - config.fps) > 5:
                logger.warning(f"相机帧率不匹配: 期望{config.fps}, 实际{actual_fps}")
            
            logger.info(f"相机验证成功: {actual_width}x{actual_height}@{actual_fps}fps")
            return True
            
        except Exception as e:
            logger.error(f"相机验证失败: {e}")
            return False
    
    def get_camera(self, camera_id: str = "default") -> Optional[cv2.VideoCapture]:
        """获取相机对象"""
        return self.cameras.get(camera_id)
    
    def get_config(self, camera_id: str = "default") -> Optional[CameraConfig]:
        """获取相机配置"""
        return self.configs.get(camera_id)
    
    def release_camera(self, camera_id: str = "default") -> bool:
        """释放相机"""
        try:
            if camera_id in self.cameras:
                self.cameras[camera_id].release()
                del self.cameras[camera_id]
                del self.configs[camera_id]
                logger.info(f"相机 {camera_id} 已释放")
                return True
            return False
        except Exception as e:
            logger.error(f"释放相机失败: {e}")
            return False
    
    def release_all(self):
        """释放所有相机"""
        for camera_id in list(self.cameras.keys()):
            self.release_camera(camera_id)
    
    def capture_frame(self, camera_id: str = "default") -> Tuple[bool, Optional[Any], Optional[ImageInfo]]:
        """
        捕获帧
        
        Returns:
            (success, frame, image_info)
        """
        try:
            cap = self.cameras.get(camera_id)
            if cap is None:
                return False, None, None
            
            ret, frame = cap.read()
            if not ret or frame is None:
                return False, None, None
            
            # 创建图像信息
            height, width, channels = frame.shape
            config = self.configs.get(camera_id)
            
            image_info = ImageInfo(
                width=width,
                height=height,
                channels=channels,
                format="BGR",
                source=f"camera_{camera_id}",
                timestamp=time.time()
            )
            
            return True, frame, image_info
            
        except Exception as e:
            logger.error(f"捕获帧失败: {e}")
            return False, None, None
    
    def is_camera_available(self, camera_id: str = "default") -> bool:
        """检查相机是否可用"""
        cap = self.cameras.get(camera_id)
        return cap is not None and cap.isOpened()
    
    def get_camera_info(self, camera_id: str = "default") -> Optional[Dict[str, Any]]:
        """获取相机信息"""
        try:
            cap = self.cameras.get(camera_id)
            config = self.configs.get(camera_id)
            
            if cap is None or config is None:
                return None
            
            return {
                'camera_id': camera_id,
                'is_opened': cap.isOpened(),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'exposure': cap.get(cv2.CAP_PROP_EXPOSURE),
                'gain': cap.get(cv2.CAP_PROP_GAIN),
                'config': config.to_dict()
            }
            
        except Exception as e:
            logger.error(f"获取相机信息失败: {e}")
            return None

# 全局相机管理器实例
_camera_manager = CameraManager()

# ============================================================================
# 便捷函数
# ============================================================================

def initialize_camera(
    device_id: Union[int, str] = 0,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    camera_id: str = "default"
) -> Optional[cv2.VideoCapture]:
    """
    便捷的相机初始化函数
    
    Args:
        device_id: 设备ID
        width: 宽度
        height: 高度
        fps: 帧率
        camera_id: 相机标识符
        
    Returns:
        cv2.VideoCapture对象或None
    """
    config = CameraConfig(
        device_id=device_id,
        width=width,
        height=height,
        fps=fps
    )
    
    return _camera_manager.initialize_camera(camera_id, config)

def get_camera(camera_id: str = "default") -> Optional[cv2.VideoCapture]:
    """获取相机对象"""
    return _camera_manager.get_camera(camera_id)

def release_camera(camera_id: str = "default") -> bool:
    """释放相机"""
    return _camera_manager.release_camera(camera_id)

def capture_frame(camera_id: str = "default") -> Tuple[bool, Optional[Any], Optional[ImageInfo]]:
    """捕获帧"""
    return _camera_manager.capture_frame(camera_id)

def is_camera_available(camera_id: str = "default") -> bool:
    """检查相机是否可用"""
    return _camera_manager.is_camera_available(camera_id)

def get_camera_info(camera_id: str = "default") -> Optional[Dict[str, Any]]:
    """获取相机信息"""
    return _camera_manager.get_camera_info(camera_id)

def cleanup_cameras():
    """清理所有相机资源"""
    _camera_manager.release_all()

def list_available_cameras() -> List[int]:
    """列出可用的相机设备"""
    available_cameras = []
    
    # 测试前10个设备ID
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    
    return available_cameras

def test_camera_resolution(device_id: Union[int, str] = 0) -> List[Tuple[int, int]]:
    """测试相机支持的分辨率"""
    supported_resolutions = []
    
    # 常见分辨率列表
    test_resolutions = [
        (320, 240),   # QVGA
        (640, 480),   # VGA
        (800, 600),   # SVGA
        (1024, 768),  # XGA
        (1280, 720),  # HD
        (1280, 960),  # SXGA
        (1920, 1080), # Full HD
        (2560, 1440), # QHD
        (3840, 2160), # 4K
    ]
    
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        return supported_resolutions
    
    try:
        for width, height in test_resolutions:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_width == width and actual_height == height:
                ret, frame = cap.read()
                if ret and frame is not None:
                    supported_resolutions.append((width, height))
    
    finally:
        cap.release()
    
    return supported_resolutions

if __name__ == "__main__":
    # 测试代码
    print("🧪 测试相机工具...")
    
    # 列出可用相机
    cameras = list_available_cameras()
    print(f"可用相机: {cameras}")
    
    if cameras:
        # 测试第一个相机
        device_id = cameras[0]
        print(f"测试相机 {device_id}...")
        
        # 测试分辨率
        resolutions = test_camera_resolution(device_id)
        print(f"支持的分辨率: {resolutions}")
        
        # 初始化相机
        cap = initialize_camera(device_id, 640, 480, 30)
        if cap:
            print("✅ 相机初始化成功")
            
            # 获取相机信息
            info = get_camera_info()
            print(f"相机信息: {info}")
            
            # 捕获几帧
            for i in range(3):
                ret, frame, img_info = capture_frame()
                if ret:
                    print(f"帧 {i+1}: {img_info.width}x{img_info.height}")
                else:
                    print(f"帧 {i+1}: 捕获失败")
            
            # 清理
            cleanup_cameras()
            print("✅ 相机已清理")
        else:
            print("❌ 相机初始化失败")
    else:
        print("❌ 未找到可用相机")
    
    print("✅ 相机工具测试完成")