#!/usr/bin/env python3
"""
YOLOS MQTT通信示例
演示如何通过MQTT发送检测结果
"""

import sys
import time
import cv2
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from detection.camera_detector import CameraDetector
    from communication.mqtt_client import MQTTClient
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保相关模块已正确安装")
    sys.exit(1)


class MQTTDetectionSystem:
    """MQTT检测系统"""
    
    def __init__(self):
        # 初始化MQTT客户端
        self.mqtt_client = MQTTClient(
            broker_host="localhost",
            broker_port=1883,
            client_id="yolos_detector_001"
        )
        
        # 初始化摄像头检测器
        self.detector = CameraDetector(
            model_type='yolov8',
            device='auto'
        )
        
        # 设置回调函数
        self.detector.set_callbacks(
            detection_callback=self.on_detection,
            frame_callback=self.on_frame
        )
        
        self.mqtt_client.set_callbacks(
            connection_callback=self.on_mqtt_connected,
            disconnection_callback=self.on_mqtt_disconnected
        )
    
    def on_detection(self, frame, results):
        """检测回调函数"""
        if results:
            print(f"检测到 {len(results)} 个目标")
            
            # 发送检测结果到MQTT
            image_info = {
                'width': frame.shape[1],
                'height': frame.shape[0],
                'channels': frame.shape[2],
                'source': 'camera'
            }
            
            self.mqtt_client.publish_detection_result(
                results=results,
                image_info=image_info
            )
    
    def on_frame(self, frame):
        """帧处理回调函数"""
        # 在帧上添加系统信息
        stats = self.detector.get_stats()
        
        # 显示FPS
        cv2.putText(frame, f"FPS: {stats['fps']:.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示MQTT状态
        mqtt_status = "Connected" if self.mqtt_client.is_connected else "Disconnected"
        color = (0, 255, 0) if self.mqtt_client.is_connected else (0, 0, 255)
        cv2.putText(frame, f"MQTT: {mqtt_status}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame
    
    def on_mqtt_connected(self):
        """MQTT连接回调"""
        print("MQTT连接成功")
        
        # 订阅命令主题
        self.mqtt_client.subscribe("yolos/commands", self.handle_command)
        
        # 发送系统状态
        status = {
            'system': 'yolos_detector',
            'status': 'online',
            'camera_type': self.detector.camera_type,
            'model_info': self.detector.get_stats()['model_info']
        }
        self.mqtt_client.publish_system_status(status)
    
    def on_mqtt_disconnected(self):
        """MQTT断开回调"""
        print("MQTT连接断开")
    
    def handle_command(self, topic, message):
        """处理MQTT命令"""
        print(f"收到命令: {message}")
        
        if isinstance(message, dict):
            command = message.get('command')
            
            if command == 'start_detection':
                print("开始检测")
            elif command == 'stop_detection':
                print("停止检测")
                self.detector.stop_detection()
            elif command == 'get_status':
                status = self.detector.get_stats()
                self.mqtt_client.publish_system_status(status)
    
    def run(self):
        """运行系统"""
        print("启动MQTT检测系统...")
        
        # 连接MQTT
        if not self.mqtt_client.connect():
            print("MQTT连接失败")
            return
        
        try:
            # 开始检测
            print("开始摄像头检测...")
            self.detector.start_detection(display=True)
            
        except KeyboardInterrupt:
            print("\n系统被用户中断")
        except Exception as e:
            print(f"系统运行出错: {e}")
        finally:
            self.detector.stop_detection()
            self.mqtt_client.disconnect()
            print("系统已停止")


def main():
    """主函数"""
    print("YOLOS MQTT检测系统")
    print("==================")
    print("确保MQTT服务器正在运行 (如 mosquitto)")
    print("按 Ctrl+C 退出")
    
    try:
        system = MQTTDetectionSystem()
        system.run()
    except Exception as e:
        print(f"系统启动失败: {e}")


if __name__ == "__main__":
    main()