#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESP32摄像头识别适配器
为ESP32-CAM提供轻量级识别功能
"""

import gc
import json
import time
import network
import urequests
from machine import Pin, I2C, SPI
import camera

# ESP32特定配置
ESP32_CONFIG = {
    'camera': {
        'pin_pwdn': 32,
        'pin_reset': -1,
        'pin_xclk': 0,
        'pin_sscb_sda': 26,
        'pin_sscb_scl': 27,
        'pin_d7': 35,
        'pin_d6': 34,
        'pin_d5': 39,
        'pin_d4': 36,
        'pin_d3': 21,
        'pin_d2': 19,
        'pin_d1': 18,
        'pin_d0': 5,
        'pin_vsync': 25,
        'pin_href': 23,
        'pin_pclk': 22,
        'xclk_freq_hz': 20000000,
        'pixel_format': camera.PIXFORMAT_JPEG,
        'frame_size': camera.FRAMESIZE_QVGA,  # 320x240
        'jpeg_quality': 12,
        'fb_count': 1
    },
    'network': {
        'ssid': 'YourWiFi',
        'password': 'YourPassword',
        'timeout': 10
    },
    'recognition': {
        'server_url': 'http://192.168.1.100:5000',
        'offline_mode': True,
        'cache_size': 10,
        'max_image_size': 10240  # 10KB
    }
}

class ESP32RecognitionAdapter:
    """ESP32识别适配器"""
    
    def __init__(self, config=None):
        self.config = config or ESP32_CONFIG
        self.camera_initialized = False
        self.wifi_connected = False
        self.recognition_cache = {}
        self.led_pin = Pin(4, Pin.OUT)  # ESP32-CAM LED
        
        print("ESP32识别适配器初始化...")
        
        # 初始化摄像头
        self.init_camera()
        
        # 初始化WiFi
        self.init_wifi()
        
        print("ESP32识别适配器就绪")
    
    def init_camera(self):
        """初始化摄像头"""
        try:
            camera.init(**self.config['camera'])
            self.camera_initialized = True
            print("✓ 摄像头初始化成功")
            
            # 测试拍照
            self.led_pin.on()  # 开启LED
            time.sleep(0.1)
            buf = camera.capture()
            self.led_pin.off()  # 关闭LED
            
            if buf:
                print(f"✓ 摄像头测试成功，图像大小: {len(buf)} bytes")
            else:
                print("✗ 摄像头测试失败")
                
        except Exception as e:
            print(f"✗ 摄像头初始化失败: {e}")
            self.camera_initialized = False
    
    def init_wifi(self):
        """初始化WiFi连接"""
        try:
            wlan = network.WLAN(network.STA_IF)
            wlan.active(True)
            
            if not wlan.isconnected():
                print("连接WiFi...")
                wlan.connect(
                    self.config['network']['ssid'],
                    self.config['network']['password']
                )
                
                # 等待连接
                timeout = self.config['network']['timeout']
                while not wlan.isconnected() and timeout > 0:
                    time.sleep(1)
                    timeout -= 1
                    print(".", end="")
                
                print()
            
            if wlan.isconnected():
                self.wifi_connected = True
                print(f"✓ WiFi连接成功: {wlan.ifconfig()[0]}")
            else:
                self.wifi_connected = False
                print("✗ WiFi连接失败，将使用离线模式")
                
        except Exception as e:
            print(f"✗ WiFi初始化失败: {e}")
            self.wifi_connected = False
    
    def capture_image(self):
        """拍摄图像"""
        if not self.camera_initialized:
            print("摄像头未初始化")
            return None
        
        try:
            # 开启LED指示拍照
            self.led_pin.on()
            time.sleep(0.1)
            
            # 拍摄图像
            buf = camera.capture()
            
            # 关闭LED
            self.led_pin.off()
            
            if buf and len(buf) > 0:
                print(f"图像拍摄成功: {len(buf)} bytes")
                return buf
            else:
                print("图像拍摄失败")
                return None
                
        except Exception as e:
            print(f"拍摄异常: {e}")
            self.led_pin.off()
            return None
    
    def recognize_offline(self, image_data, scene="pets"):
        """离线识别（基础模式）"""
        try:
            # ESP32离线识别非常简化
            # 基于图像大小和基本特征进行简单判断
            
            image_size = len(image_data)
            
            # 生成缓存键
            cache_key = f"{scene}_{image_size}_{hash(image_data[:100]) % 1000}"
            
            # 检查缓存
            if cache_key in self.recognition_cache:
                print("使用缓存结果")
                return self.recognition_cache[cache_key]
            
            # 简单的基于大小的分类
            result = {
                'scene': scene,
                'results': [],
                'confidence': 0.3,
                'source': 'esp32_offline',
                'timestamp': time.time()
            }
            
            if scene == "pets":
                if image_size > 8000:
                    result['results'] = [{'class_name': 'large_pet', 'confidence': 0.4}]
                elif image_size > 5000:
                    result['results'] = [{'class_name': 'medium_pet', 'confidence': 0.3}]
                else:
                    result['results'] = [{'class_name': 'small_pet', 'confidence': 0.2}]
            
            elif scene == "qr_codes":
                # 简单的QR码检测（基于图像特征）
                if self._detect_qr_pattern(image_data):
                    result['results'] = [{'class_name': 'qr_code', 'confidence': 0.6}]
                    result['confidence'] = 0.6
            
            # 缓存结果
            if len(self.recognition_cache) >= self.config['recognition']['cache_size']:
                # 清理最旧的缓存
                oldest_key = min(self.recognition_cache.keys())
                del self.recognition_cache[oldest_key]
            
            self.recognition_cache[cache_key] = result
            
            print(f"离线识别完成: {result}")
            return result
            
        except Exception as e:
            print(f"离线识别失败: {e}")
            return {
                'scene': scene,
                'results': [],
                'confidence': 0.0,
                'source': 'esp32_error',
                'timestamp': time.time()
            }
    
    def recognize_online(self, image_data, scene="pets"):
        """在线识别"""
        if not self.wifi_connected:
            print("WiFi未连接，使用离线模式")
            return self.recognize_offline(image_data, scene)
        
        try:
            # 检查图像大小
            if len(image_data) > self.config['recognition']['max_image_size']:
                print(f"图像过大 ({len(image_data)} bytes)，压缩后重试")
                # 这里可以添加图像压缩逻辑
                return self.recognize_offline(image_data, scene)
            
            # 准备请求数据
            url = f"{self.config['recognition']['server_url']}/recognize"
            
            # 将图像编码为base64（简化版）
            import ubinascii
            image_b64 = ubinascii.b2a_base64(image_data).decode().strip()
            
            data = {
                'scene': scene,
                'image': image_b64,
                'source': 'esp32',
                'timestamp': time.time()
            }
            
            # 发送请求
            print("发送在线识别请求...")
            response = urequests.post(
                url,
                json=data,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                result['source'] = 'esp32_online'
                print(f"在线识别成功: {result}")
                response.close()
                return result
            else:
                print(f"在线识别失败: HTTP {response.status_code}")
                response.close()
                return self.recognize_offline(image_data, scene)
                
        except Exception as e:
            print(f"在线识别异常: {e}")
            return self.recognize_offline(image_data, scene)
    
    def _detect_qr_pattern(self, image_data):
        """简单的QR码模式检测"""
        try:
            # 非常简化的QR码检测
            # 检查JPEG头部和一些基本模式
            if len(image_data) < 100:
                return False
            
            # 检查JPEG标识
            if image_data[:2] != b'\xff\xd8':
                return False
            
            # 简单的模式检测（实际应用中需要更复杂的算法）
            # 这里只是示例
            pattern_count = 0
            for i in range(50, min(len(image_data), 200)):
                if image_data[i] == 0xFF or image_data[i] == 0x00:
                    pattern_count += 1
            
            # 如果有足够的黑白模式，可能是QR码
            return pattern_count > 20
            
        except:
            return False
    
    def recognize(self, scene="pets", use_online=True):
        """执行识别"""
        print(f"开始识别场景: {scene}")
        
        # 拍摄图像
        image_data = self.capture_image()
        if not image_data:
            return None
        
        # 执行识别
        if use_online and self.wifi_connected:
            result = self.recognize_online(image_data, scene)
        else:
            result = self.recognize_offline(image_data, scene)
        
        # 清理内存
        gc.collect()
        
        return result
    
    def continuous_recognition(self, scene="pets", interval=5, max_iterations=100):
        """连续识别模式"""
        print(f"开始连续识别: {scene}, 间隔: {interval}s")
        
        iteration = 0
        while iteration < max_iterations:
            try:
                print(f"\n--- 第 {iteration + 1} 次识别 ---")
                
                result = self.recognize(scene)
                if result:
                    print(f"识别结果: {result}")
                    
                    # 如果识别到目标，闪烁LED
                    if result['results'] and result['confidence'] > 0.5:
                        for _ in range(3):
                            self.led_pin.on()
                            time.sleep(0.2)
                            self.led_pin.off()
                            time.sleep(0.2)
                
                # 等待下次识别
                time.sleep(interval)
                iteration += 1
                
            except KeyboardInterrupt:
                print("用户中断识别")
                break
            except Exception as e:
                print(f"识别异常: {e}")
                time.sleep(1)
        
        print("连续识别结束")
    
    def get_status(self):
        """获取系统状态"""
        import esp32
        
        status = {
            'camera_initialized': self.camera_initialized,
            'wifi_connected': self.wifi_connected,
            'cache_size': len(self.recognition_cache),
            'free_memory': gc.mem_free(),
            'hall_sensor': esp32.hall_sensor(),
            'temperature': (esp32.raw_temperature() - 32) * 5 / 9,  # 转换为摄氏度
            'timestamp': time.time()
        }
        
        return status
    
    def cleanup(self):
        """清理资源"""
        try:
            if self.camera_initialized:
                camera.deinit()
            
            self.led_pin.off()
            self.recognition_cache.clear()
            gc.collect()
            
            print("ESP32适配器清理完成")
            
        except Exception as e:
            print(f"清理异常: {e}")

# 便捷函数
def create_esp32_adapter(config=None):
    """创建ESP32适配器"""
    return ESP32RecognitionAdapter(config)

# 主程序示例
def main():
    """主程序"""
    try:
        # 创建适配器
        adapter = create_esp32_adapter()
        
        # 检查状态
        status = adapter.get_status()
        print(f"系统状态: {status}")
        
        # 单次识别测试
        print("\n=== 单次识别测试 ===")
        result = adapter.recognize("pets")
        if result:
            print(f"识别结果: {result}")
        
        # 连续识别（可选）
        # print("\n=== 连续识别模式 ===")
        # adapter.continuous_recognition("pets", interval=10, max_iterations=5)
        
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序异常: {e}")
    finally:
        try:
            adapter.cleanup()
        except:
            pass

if __name__ == "__main__":
    main()