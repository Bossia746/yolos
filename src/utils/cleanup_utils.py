#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理工具函数
统一的资源清理和内存管理功能
"""

import gc
import cv2
import logging
import threading
import time
from typing import List, Dict, Any, Optional, Callable
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class ResourceManager:
    """资源管理器"""
    
    def __init__(self):
        self.resources: Dict[str, Any] = {}
        self.cleanup_callbacks: Dict[str, Callable] = {}
        self.lock = threading.Lock()
    
    def register_resource(
        self, 
        name: str, 
        resource: Any, 
        cleanup_callback: Optional[Callable] = None
    ):
        """注册资源"""
        with self.lock:
            self.resources[name] = resource
            if cleanup_callback:
                self.cleanup_callbacks[name] = cleanup_callback
            logger.debug(f"注册资源: {name}")
    
    def unregister_resource(self, name: str) -> bool:
        """注销资源"""
        with self.lock:
            if name in self.resources:
                # 执行清理回调
                if name in self.cleanup_callbacks:
                    try:
                        self.cleanup_callbacks[name](self.resources[name])
                    except Exception as e:
                        logger.error(f"清理资源 {name} 失败: {e}")
                
                del self.resources[name]
                if name in self.cleanup_callbacks:
                    del self.cleanup_callbacks[name]
                
                logger.debug(f"注销资源: {name}")
                return True
            return False
    
    def cleanup_all(self):
        """清理所有资源"""
        with self.lock:
            for name in list(self.resources.keys()):
                self.unregister_resource(name)
    
    def get_resource(self, name: str) -> Any:
        """获取资源"""
        with self.lock:
            return self.resources.get(name)
    
    def list_resources(self) -> List[str]:
        """列出所有资源"""
        with self.lock:
            return list(self.resources.keys())

# 全局资源管理器
_resource_manager = ResourceManager()

# ============================================================================
# 相机清理函数
# ============================================================================

def cleanup_camera(camera: cv2.VideoCapture) -> bool:
    """清理相机资源"""
    try:
        if camera and camera.isOpened():
            camera.release()
            logger.debug("相机资源已释放")
            return True
        return False
    except Exception as e:
        logger.error(f"清理相机失败: {e}")
        return False

def cleanup_cameras(cameras: List[cv2.VideoCapture]) -> int:
    """清理多个相机"""
    cleaned_count = 0
    for camera in cameras:
        if cleanup_camera(camera):
            cleaned_count += 1
    return cleaned_count

# ============================================================================
# 窗口清理函数
# ============================================================================

def cleanup_windows(window_names: Optional[List[str]] = None) -> bool:
    """清理OpenCV窗口"""
    try:
        if window_names:
            for window_name in window_names:
                cv2.destroyWindow(window_name)
        else:
            cv2.destroyAllWindows()
        
        # 等待窗口关闭
        cv2.waitKey(1)
        logger.debug("OpenCV窗口已清理")
        return True
    except Exception as e:
        logger.error(f"清理窗口失败: {e}")
        return False

# ============================================================================
# 内存清理函数
# ============================================================================

def cleanup_memory(force_gc: bool = True) -> Dict[str, Any]:
    """清理内存"""
    try:
        import psutil
        import os
        
        # 获取清理前的内存信息
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # 强制垃圾回收
        if force_gc:
            collected = gc.collect()
        else:
            collected = 0
        
        # 获取清理后的内存信息
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_freed = memory_before - memory_after
        
        result = {
            'memory_before_mb': round(memory_before, 2),
            'memory_after_mb': round(memory_after, 2),
            'memory_freed_mb': round(memory_freed, 2),
            'objects_collected': collected
        }
        
        logger.debug(f"内存清理完成: {result}")
        return result
        
    except ImportError:
        # 如果没有psutil，只执行垃圾回收
        if force_gc:
            collected = gc.collect()
            return {'objects_collected': collected}
        return {}
    except Exception as e:
        logger.error(f"内存清理失败: {e}")
        return {}

# ============================================================================
# 线程清理函数
# ============================================================================

def cleanup_threads(threads: List[threading.Thread], timeout: float = 5.0) -> Dict[str, int]:
    """清理线程"""
    try:
        alive_count = 0
        stopped_count = 0
        timeout_count = 0
        
        for thread in threads:
            if thread.is_alive():
                alive_count += 1
                
                # 等待线程结束
                thread.join(timeout=timeout)
                
                if thread.is_alive():
                    timeout_count += 1
                    logger.warning(f"线程 {thread.name} 清理超时")
                else:
                    stopped_count += 1
                    logger.debug(f"线程 {thread.name} 已停止")
        
        result = {
            'alive_count': alive_count,
            'stopped_count': stopped_count,
            'timeout_count': timeout_count
        }
        
        logger.debug(f"线程清理完成: {result}")
        return result
        
    except Exception as e:
        logger.error(f"线程清理失败: {e}")
        return {}

# ============================================================================
# 文件清理函数
# ============================================================================

def cleanup_temp_files(temp_dirs: List[str], max_age_hours: float = 24.0) -> Dict[str, int]:
    """清理临时文件"""
    import os
    import glob
    from pathlib import Path
    
    try:
        deleted_count = 0
        error_count = 0
        total_size = 0
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for temp_dir in temp_dirs:
            if not os.path.exists(temp_dir):
                continue
            
            # 查找临时文件
            patterns = ['*.tmp', '*.temp', '*.cache', '*.log']
            
            for pattern in patterns:
                for file_path in glob.glob(os.path.join(temp_dir, pattern)):
                    try:
                        file_stat = os.stat(file_path)
                        file_age = current_time - file_stat.st_mtime
                        
                        if file_age > max_age_seconds:
                            file_size = file_stat.st_size
                            os.remove(file_path)
                            deleted_count += 1
                            total_size += file_size
                            logger.debug(f"删除临时文件: {file_path}")
                    
                    except Exception as e:
                        error_count += 1
                        logger.error(f"删除文件 {file_path} 失败: {e}")
        
        result = {
            'deleted_count': deleted_count,
            'error_count': error_count,
            'total_size_mb': round(total_size / 1024 / 1024, 2)
        }
        
        logger.info(f"临时文件清理完成: {result}")
        return result
        
    except Exception as e:
        logger.error(f"临时文件清理失败: {e}")
        return {}

# ============================================================================
# GPU清理函数
# ============================================================================

def cleanup_gpu_memory() -> Dict[str, Any]:
    """清理GPU内存"""
    result = {}
    
    try:
        # PyTorch GPU清理
        import torch
        if torch.cuda.is_available():
            memory_before = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            memory_after = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            
            result['pytorch'] = {
                'memory_before_mb': round(memory_before, 2),
                'memory_after_mb': round(memory_after, 2),
                'memory_freed_mb': round(memory_before - memory_after, 2)
            }
            logger.debug("PyTorch GPU内存已清理")
    
    except ImportError:
        pass
    except Exception as e:
        logger.error(f"PyTorch GPU清理失败: {e}")
    
    try:
        # TensorFlow GPU清理
        import tensorflow as tf
        if tf.config.list_physical_devices('GPU'):
            tf.keras.backend.clear_session()
            result['tensorflow'] = {'status': 'cleared'}
            logger.debug("TensorFlow GPU内存已清理")
    
    except ImportError:
        pass
    except Exception as e:
        logger.error(f"TensorFlow GPU清理失败: {e}")
    
    return result

# ============================================================================
# 综合清理函数
# ============================================================================

def cleanup(
    cameras: Optional[List[cv2.VideoCapture]] = None,
    windows: Optional[List[str]] = None,
    threads: Optional[List[threading.Thread]] = None,
    temp_dirs: Optional[List[str]] = None,
    cleanup_memory: bool = True,
    cleanup_gpu: bool = True,
    cleanup_resources: bool = True
) -> Dict[str, Any]:
    """
    综合清理函数
    
    Args:
        cameras: 要清理的相机列表
        windows: 要清理的窗口名称列表
        threads: 要清理的线程列表
        temp_dirs: 要清理的临时目录列表
        cleanup_memory: 是否清理内存
        cleanup_gpu: 是否清理GPU内存
        cleanup_resources: 是否清理注册的资源
    
    Returns:
        清理结果统计
    """
    logger.info("开始系统清理...")
    
    results = {
        'timestamp': time.time(),
        'success': True,
        'details': {}
    }
    
    try:
        # 清理相机
        if cameras:
            camera_count = cleanup_cameras(cameras)
            results['details']['cameras'] = {'cleaned_count': camera_count}
        
        # 清理窗口
        if windows or windows is None:  # None表示清理所有窗口
            window_success = cleanup_windows(windows)
            results['details']['windows'] = {'success': window_success}
        
        # 清理线程
        if threads:
            thread_result = cleanup_threads(threads)
            results['details']['threads'] = thread_result
        
        # 清理临时文件
        if temp_dirs:
            temp_result = cleanup_temp_files(temp_dirs)
            results['details']['temp_files'] = temp_result
        
        # 清理注册的资源
        if cleanup_resources:
            resource_count = len(_resource_manager.list_resources())
            _resource_manager.cleanup_all()
            results['details']['resources'] = {'cleaned_count': resource_count}
        
        # 清理GPU内存
        if cleanup_gpu:
            gpu_result = cleanup_gpu_memory()
            if gpu_result:
                results['details']['gpu'] = gpu_result
        
        # 清理系统内存
        if cleanup_memory:
            memory_result = cleanup_memory(force_gc=True)
            results['details']['memory'] = memory_result
        
        logger.info(f"系统清理完成: {results}")
        
    except Exception as e:
        logger.error(f"系统清理失败: {e}")
        results['success'] = False
        results['error'] = str(e)
    
    return results

# ============================================================================
# 上下文管理器
# ============================================================================

@contextmanager
def managed_camera(device_id: int = 0):
    """相机资源管理上下文"""
    camera = None
    try:
        camera = cv2.VideoCapture(device_id)
        if not camera.isOpened():
            raise RuntimeError(f"无法打开相机 {device_id}")
        yield camera
    finally:
        if camera:
            cleanup_camera(camera)

@contextmanager
def managed_window(window_name: str):
    """窗口资源管理上下文"""
    try:
        yield window_name
    finally:
        cleanup_windows([window_name])

@contextmanager
def managed_resources():
    """资源管理上下文"""
    try:
        yield _resource_manager
    finally:
        _resource_manager.cleanup_all()

# ============================================================================
# 便捷函数
# ============================================================================

def register_resource(name: str, resource: Any, cleanup_callback: Optional[Callable] = None):
    """注册资源到全局管理器"""
    _resource_manager.register_resource(name, resource, cleanup_callback)

def unregister_resource(name: str) -> bool:
    """从全局管理器注销资源"""
    return _resource_manager.unregister_resource(name)

def cleanup_all_resources():
    """清理所有注册的资源"""
    _resource_manager.cleanup_all()

def get_resource(name: str) -> Any:
    """从全局管理器获取资源"""
    return _resource_manager.get_resource(name)

def quick_cleanup():
    """快速清理常用资源"""
    return cleanup(
        windows=None,  # 清理所有窗口
        cleanup_memory=True,
        cleanup_gpu=True,
        cleanup_resources=True
    )

if __name__ == "__main__":
    # 测试代码
    print("🧪 测试清理工具...")
    
    # 测试资源管理器
    with managed_resources() as rm:
        # 注册测试资源
        test_data = {"test": "data"}
        rm.register_resource("test_data", test_data)
        
        print(f"注册的资源: {rm.list_resources()}")
        
        # 资源会在上下文结束时自动清理
    
    # 测试内存清理
    memory_result = cleanup_memory()
    print(f"内存清理结果: {memory_result}")
    
    # 测试GPU清理
    gpu_result = cleanup_gpu_memory()
    if gpu_result:
        print(f"GPU清理结果: {gpu_result}")
    
    # 测试快速清理
    quick_result = quick_cleanup()
    print(f"快速清理结果: {quick_result['success']}")
    
    print("✅ 清理工具测试完成")