#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
树莓派适配器
为树莓派提供优化的识别功能
"""

import os
import sys
import time
import logging
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import cv2

# 树莓派特定导入
try:
    import RPi.GPIO as GPIO
    RPI_GPIO_AVAILABLE = True
except ImportError:
    RPI_GPIO_AVAILABLE = False
    logging.warning("RPi.GPIO不可用，GPIO功能将被禁用")

try:
    from picamera import PiCamera
    from picamera.array import PiRGBArray
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    logging.warning("PiCamera不可用，将使用USB摄像头")

# 导入混合识别系统
try:
    from ...recognition.hybrid_recognition_system import HybridRecognitionSystem, RecognitionRequest
    from ...core.cross_platform_manager import get_platform_manager
except ImportError:
    logging.error("无法导入混合识别系统")
    HybridRecognitionSystem = None

logger = logging.getLogger(__name__)

class RaspberryPiAdapter:
    """树莓派适配器"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        
        # 系统信息
        self.pi_model = self._detect_pi_model()
        self.cpu_temp_file = Path('/sys/class/thermal/thermal_zone0/temp')
        
        # GPIO设置
        self.gpio_initialized = False
        self.led_pin = self.config.get('led_pin', 18)
        self.button_pin = self.config.get('button_pin', 2)
        
        # 摄像头设置
        self.camera = None
        self.camera_type = None
        
        # 识别系统
        self.recognition_system = None
        
        # 性能监控
        self.performance_stats = {
            'total_recognitions': 0,
            'average_processing_time': 0.0,
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'temperature': 0.0
        }
        
        # 初始化组件
        self._init_gpio()
        self._init_camera()
        self._init_recognition_system()
        
        logger.info(f"树莓派适配器初始化完成 - 型号: {self.pi_model}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'camera': {
                'resolution': (640, 480),
                'framerate': 15,
                'rotation': 0,
                'use_picamera': PICAMERA_AVAILABLE,
                'usb_camera_id': 0
            },
            'gpio': {
                'led_pin': 18,
                'button_pin': 2,
                'use_gpio': RPI_GPIO_AVAILABLE
            },
            'recognition': {
                'offline_first': True,
                'max_processing_time': 10.0,
                'enable_performance_monitoring': True
            },
            'optimization': {
                'cpu_limit': 80,  # CPU使用率限制
                'memory_limit': 512,  # 内存限制(MB)
                'temperature_limit': 70,  # 温度限制(°C)
                'enable_gpu': False  # 树莓派GPU支持有限
            }
        }
    
    def _detect_pi_model(self) -> str:
        """检测树莓派型号"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
            
            if 'Pi 4' in cpuinfo:
                return 'Raspberry Pi 4'
            elif 'Pi 3' in cpuinfo:
                return 'Raspberry Pi 3'
            elif 'Pi 2' in cpuinfo:
                return 'Raspberry Pi 2'
            elif 'Pi Zero' in cpuinfo:
                return 'Raspberry Pi Zero'
            else:
                return 'Raspberry Pi (Unknown)'
                
        except Exception:
            return 'Unknown'
    
    def _init_gpio(self):
        """初始化GPIO"""
        if not self.config['gpio']['use_gpio'] or not RPI_GPIO_AVAILABLE:
            logger.info("GPIO功能已禁用")
            return
        
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            # 设置LED引脚
            GPIO.setup(self.led_pin, GPIO.OUT)
            GPIO.output(self.led_pin, GPIO.LOW)
            
            # 设置按钮引脚
            GPIO.setup(self.button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            
            # 添加按钮中断
            GPIO.add_event_detect(
                self.button_pin, 
                GPIO.FALLING, 
                callback=self._button_callback, 
                bouncetime=300
            )
            
            self.gpio_initialized = True
            logger.info("✓ GPIO初始化成功")
            
        except Exception as e:
            logger.error(f"GPIO初始化失败: {e}")
            self.gpio_initialized = False
    
    def _init_camera(self):
        """初始化摄像头"""
        camera_config = self.config['camera']
        
        # 尝试使用PiCamera
        if camera_config['use_picamera'] and PICAMERA_AVAILABLE:
            try:
                self.camera = PiCamera()
                self.camera.resolution = camera_config['resolution']
                self.camera.framerate = camera_config['framerate']
                self.camera.rotation = camera_config['rotation']
                
                # 预热摄像头
                time.sleep(2)
                
                self.camera_type = 'picamera'
                logger.info("✓ PiCamera初始化成功")
                return
                
            except Exception as e:
                logger.warning(f"PiCamera初始化失败: {e}")
        
        # 回退到USB摄像头
        try:
            self.camera = cv2.VideoCapture(camera_config['usb_camera_id'])
            
            if self.camera.isOpened():
                # 设置分辨率
                width, height = camera_config['resolution']
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.camera.set(cv2.CAP_PROP_FPS, camera_config['framerate'])
                
                self.camera_type = 'usb'
                logger.info("✓ USB摄像头初始化成功")
            else:
                raise Exception("无法打开USB摄像头")
                
        except Exception as e:
            logger.error(f"摄像头初始化失败: {e}")
            self.camera = None
            self.camera_type = None
    
    def _init_recognition_system(self):
        """初始化识别系统"""
        if HybridRecognitionSystem is None:
            logger.error("混合识别系统不可用")
            return
        
        try:
            # 获取平台管理器并优化配置
            platform_manager = get_platform_manager()
            
            # 为树莓派优化系统配置
            self.recognition_system = HybridRecognitionSystem(
                offline_models_dir="./models/offline_models",
                online_fallback=True,
                network_check_interval=60  # 树莓派网络检查间隔更长
            )
            
            logger.info("✓ 识别系统初始化成功")
            
        except Exception as e:
            logger.error(f"识别系统初始化失败: {e}")
            self.recognition_system = None
    
    def _button_callback(self, channel):
        """按钮回调函数"""
        logger.info("按钮被按下，执行识别")
        
        # 闪烁LED表示开始识别
        self._blink_led(3, 0.1)
        
        # 执行识别
        threading.Thread(target=self._button_recognition, daemon=True).start()
    
    def _button_recognition(self):
        """按钮触发的识别"""
        try:
            result = self.recognize("pets")
            if result and result.results:
                # 识别成功，长亮LED
                self._set_led(True)
                time.sleep(2)
                self._set_led(False)
            else:
                # 识别失败，快速闪烁
                self._blink_led(5, 0.2)
                
        except Exception as e:
            logger.error(f"按钮识别失败: {e}")
    
    def capture_image(self) -> Optional[np.ndarray]:
        """拍摄图像"""
        if self.camera is None:
            logger.error("摄像头未初始化")
            return None
        
        try:
            if self.camera_type == 'picamera':
                # 使用PiCamera
                raw_capture = PiRGBArray(self.camera)
                self.camera.capture(raw_capture, format="bgr")
                image = raw_capture.array
                raw_capture.truncate(0)
                
            elif self.camera_type == 'usb':
                # 使用USB摄像头
                ret, image = self.camera.read()
                if not ret:
                    logger.error("无法从USB摄像头读取图像")
                    return None
            else:
                logger.error("未知摄像头类型")
                return None
            
            logger.info(f"图像拍摄成功: {image.shape}")
            return image
            
        except Exception as e:
            logger.error(f"图像拍摄失败: {e}")
            return None
    
    def recognize(self, scene: str = "pets", use_online: bool = True) -> Optional[Any]:
        """执行识别"""
        if self.recognition_system is None:
            logger.error("识别系统未初始化")
            return None
        
        start_time = time.time()
        
        try:
            # 检查系统资源
            if not self._check_system_resources():
                logger.warning("系统资源不足，跳过识别")
                return None
            
            # 拍摄图像
            image = self.capture_image()
            if image is None:
                return None
            
            # 执行识别
            response = self.recognition_system.recognize_scene(
                scene, image, use_online=use_online
            )
            
            # 更新性能统计
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time)
            
            logger.info(f"识别完成: {scene}, 来源: {response.source}, "
                       f"耗时: {processing_time:.2f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"识别失败: {e}")
            return None
    
    def continuous_recognition(self, scene: str = "pets", 
                             interval: float = 5.0, 
                             max_iterations: int = 100):
        """连续识别模式"""
        logger.info(f"开始连续识别: {scene}, 间隔: {interval}s")
        
        iteration = 0
        while iteration < max_iterations:
            try:
                # 检查系统状态
                if not self._check_system_health():
                    logger.warning("系统健康检查失败，暂停识别")
                    time.sleep(interval * 2)
                    continue
                
                # 执行识别
                result = self.recognize(scene)
                
                if result and result.results:
                    logger.info(f"识别结果: {result.results}")
                    
                    # 识别成功时闪烁LED
                    if self.gpio_initialized:
                        self._blink_led(2, 0.3)
                
                # 等待下次识别
                time.sleep(interval)
                iteration += 1
                
            except KeyboardInterrupt:
                logger.info("用户中断连续识别")
                break
            except Exception as e:
                logger.error(f"连续识别异常: {e}")
                time.sleep(1)
        
        logger.info("连续识别结束")
    
    def _check_system_resources(self) -> bool:
        """检查系统资源"""
        try:
            # 检查CPU使用率
            cpu_usage = self._get_cpu_usage()
            if cpu_usage > self.config['optimization']['cpu_limit']:
                logger.warning(f"CPU使用率过高: {cpu_usage}%")
                return False
            
            # 检查内存使用
            memory_usage = self._get_memory_usage()
            if memory_usage > self.config['optimization']['memory_limit']:
                logger.warning(f"内存使用过高: {memory_usage}MB")
                return False
            
            # 检查温度
            temperature = self._get_cpu_temperature()
            if temperature > self.config['optimization']['temperature_limit']:
                logger.warning(f"CPU温度过高: {temperature}°C")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"资源检查失败: {e}")
            return True  # 检查失败时允许继续
    
    def _check_system_health(self) -> bool:
        """检查系统健康状态"""
        try:
            # 基础资源检查
            if not self._check_system_resources():
                return False
            
            # 检查摄像头状态
            if self.camera is None:
                logger.error("摄像头不可用")
                return False
            
            # 检查识别系统状态
            if self.recognition_system is None:
                logger.error("识别系统不可用")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return False
    
    def _get_cpu_usage(self) -> float:
        """获取CPU使用率"""
        try:
            result = subprocess.run(['top', '-bn1'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            
            for line in lines:
                if 'Cpu(s):' in line:
                    # 解析CPU使用率
                    parts = line.split(',')
                    for part in parts:
                        if 'id' in part:  # idle
                            idle = float(part.split('%')[0].strip().split()[-1])
                            return 100 - idle
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _get_memory_usage(self) -> float:
        """获取内存使用量(MB)"""
        try:
            result = subprocess.run(['free', '-m'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            
            for line in lines:
                if 'Mem:' in line:
                    parts = line.split()
                    used = float(parts[2])
                    return used
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _get_cpu_temperature(self) -> float:
        """获取CPU温度"""
        try:
            if self.cpu_temp_file.exists():
                with open(self.cpu_temp_file, 'r') as f:
                    temp_str = f.read().strip()
                    return float(temp_str) / 1000.0
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _update_performance_stats(self, processing_time: float):
        """更新性能统计"""
        self.performance_stats['total_recognitions'] += 1
        
        # 更新平均处理时间
        total = self.performance_stats['total_recognitions']
        current_avg = self.performance_stats['average_processing_time']
        new_avg = (current_avg * (total - 1) + processing_time) / total
        self.performance_stats['average_processing_time'] = new_avg
        
        # 更新系统状态
        self.performance_stats['cpu_usage'] = self._get_cpu_usage()
        self.performance_stats['memory_usage'] = self._get_memory_usage()
        self.performance_stats['temperature'] = self._get_cpu_temperature()
    
    def _set_led(self, state: bool):
        """设置LED状态"""
        if self.gpio_initialized:
            GPIO.output(self.led_pin, GPIO.HIGH if state else GPIO.LOW)
    
    def _blink_led(self, times: int, interval: float):
        """闪烁LED"""
        if not self.gpio_initialized:
            return
        
        for _ in range(times):
            self._set_led(True)
            time.sleep(interval)
            self._set_led(False)
            time.sleep(interval)
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            'pi_model': self.pi_model,
            'camera_type': self.camera_type,
            'camera_available': self.camera is not None,
            'gpio_initialized': self.gpio_initialized,
            'recognition_system_available': self.recognition_system is not None,
            'performance_stats': self.performance_stats.copy(),
            'system_resources': {
                'cpu_usage': self._get_cpu_usage(),
                'memory_usage': self._get_memory_usage(),
                'temperature': self._get_cpu_temperature()
            },
            'timestamp': time.time()
        }
        
        return status
    
    def cleanup(self):
        """清理资源"""
        try:
            # 清理摄像头
            if self.camera is not None:
                if self.camera_type == 'picamera':
                    self.camera.close()
                elif self.camera_type == 'usb':
                    self.camera.release()
            
            # 清理GPIO
            if self.gpio_initialized:
                GPIO.cleanup()
            
            logger.info("树莓派适配器清理完成")
            
        except Exception as e:
            logger.error(f"清理异常: {e}")

# 便捷函数
def create_raspberry_pi_adapter(config: Optional[Dict] = None) -> RaspberryPiAdapter:
    """创建树莓派适配器"""
    return RaspberryPiAdapter(config)

# 主程序示例
def main():
    """主程序"""
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 创建适配器
        adapter = create_raspberry_pi_adapter()
        
        # 检查系统状态
        status = adapter.get_system_status()
        logger.info(f"系统状态: {status}")
        
        # 单次识别测试
        logger.info("=== 单次识别测试 ===")
        result = adapter.recognize("pets")
        if result:
            logger.info(f"识别结果: {result}")
        
        # 连续识别（可选）
        logger.info("=== 连续识别模式 ===")
        adapter.continuous_recognition("pets", interval=10, max_iterations=5)
        
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"程序异常: {e}")
    finally:
        try:
            adapter.cleanup()
        except:
            pass

if __name__ == "__main__":
    main()