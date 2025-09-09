"""
MQTT通信客户端
"""

import json
import time
import threading
from typing import Dict, Any, Optional, Callable
import paho.mqtt.client as mqtt
from datetime import datetime


class MQTTClient:
    """MQTT通信客户端"""
    
    def __init__(self, 
                 broker_host: str = "localhost",
                 broker_port: int = 1883,
                 client_id: Optional[str] = None,
                 username: Optional[str] = None,
                 password: Optional[str] = None):
        """
        初始化MQTT客户端
        
        Args:
            broker_host: MQTT代理服务器地址
            broker_port: MQTT代理服务器端口
            client_id: 客户端ID
            username: 用户名
            password: 密码
        """
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client_id = client_id or f"yolos_client_{int(time.time())}"
        
        # 创建MQTT客户端
        self.client = mqtt.Client(self.client_id)
        
        # 设置认证
        if username and password:
            self.client.username_pw_set(username, password)
        
        # 设置回调函数
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.on_publish = self._on_publish
        
        # 状态变量
        self.is_connected = False
        self.subscribed_topics = set()
        
        # 回调函数
        self.message_callbacks: Dict[str, Callable] = {}
        self.connection_callback: Optional[Callable] = None
        self.disconnection_callback: Optional[Callable] = None
        
        # 消息统计
        self.messages_sent = 0
        self.messages_received = 0
        
    def connect(self, keepalive: int = 60) -> bool:
        """连接到MQTT代理"""
        try:
            print(f"连接到MQTT代理: {self.broker_host}:{self.broker_port}")
            self.client.connect(self.broker_host, self.broker_port, keepalive)
            self.client.loop_start()
            
            # 等待连接建立
            timeout = 10
            start_time = time.time()
            while not self.is_connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if self.is_connected:
                print("MQTT连接成功")
                return True
            else:
                print("MQTT连接超时")
                return False
                
        except Exception as e:
            print(f"MQTT连接失败: {e}")
            return False
    
    def disconnect(self):
        """断开MQTT连接"""
        if self.is_connected:
            self.client.loop_stop()
            self.client.disconnect()
            print("MQTT连接已断开")
    
    def subscribe(self, topic: str, callback: Optional[Callable] = None, qos: int = 0):
        """订阅主题"""
        if not self.is_connected:
            print("MQTT未连接，无法订阅主题")
            return False
        
        try:
            result, _ = self.client.subscribe(topic, qos)
            if result == mqtt.MQTT_ERR_SUCCESS:
                self.subscribed_topics.add(topic)
                if callback:
                    self.message_callbacks[topic] = callback
                print(f"订阅主题成功: {topic}")
                return True
            else:
                print(f"订阅主题失败: {topic}, 错误码: {result}")
                return False
        except Exception as e:
            print(f"订阅主题异常: {e}")
            return False
    
    def unsubscribe(self, topic: str):
        """取消订阅主题"""
        if not self.is_connected:
            return False
        
        try:
            result, _ = self.client.unsubscribe(topic)
            if result == mqtt.MQTT_ERR_SUCCESS:
                self.subscribed_topics.discard(topic)
                self.message_callbacks.pop(topic, None)
                print(f"取消订阅成功: {topic}")
                return True
            else:
                print(f"取消订阅失败: {topic}")
                return False
        except Exception as e:
            print(f"取消订阅异常: {e}")
            return False
    
    def publish(self, topic: str, payload: Any, qos: int = 0, retain: bool = False) -> bool:
        """发布消息"""
        if not self.is_connected:
            print("MQTT未连接，无法发布消息")
            return False
        
        try:
            # 序列化payload
            if isinstance(payload, (dict, list)):
                payload = json.dumps(payload, ensure_ascii=False)
            elif not isinstance(payload, (str, bytes)):
                payload = str(payload)
            
            # 发布消息
            result = self.client.publish(topic, payload, qos, retain)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                self.messages_sent += 1
                return True
            else:
                print(f"发布消息失败: {topic}, 错误码: {result.rc}")
                return False
                
        except Exception as e:
            print(f"发布消息异常: {e}")
            return False
    
    def publish_detection_result(self, 
                               results: list,
                               image_info: Dict[str, Any],
                               topic_prefix: str = "yolos/detection"):
        """发布检测结果"""
        timestamp = datetime.now().isoformat()
        
        # 构造消息
        message = {
            'timestamp': timestamp,
            'client_id': self.client_id,
            'image_info': image_info,
            'detection_count': len(results),
            'detections': results
        }
        
        # 发布到主结果主题
        self.publish(f"{topic_prefix}/results", message)
        
        # 发布检测统计
        stats = {
            'timestamp': timestamp,
            'client_id': self.client_id,
            'detection_count': len(results),
            'objects': [r['class_name'] for r in results]
        }
        self.publish(f"{topic_prefix}/stats", stats)
        
        # 按类别发布
        for result in results:
            class_topic = f"{topic_prefix}/objects/{result['class_name']}"
            self.publish(class_topic, {
                'timestamp': timestamp,
                'confidence': result['confidence'],
                'bbox': result['bbox']
            })
    
    def publish_system_status(self, 
                            status: Dict[str, Any],
                            topic: str = "yolos/system/status"):
        """发布系统状态"""
        status['timestamp'] = datetime.now().isoformat()
        status['client_id'] = self.client_id
        self.publish(topic, status)
    
    def set_callbacks(self, 
                     connection_callback: Optional[Callable] = None,
                     disconnection_callback: Optional[Callable] = None):
        """设置回调函数"""
        self.connection_callback = connection_callback
        self.disconnection_callback = disconnection_callback
    
    def _on_connect(self, client, userdata, flags, rc):
        """连接回调"""
        if rc == 0:
            self.is_connected = True
            print(f"MQTT连接成功，客户端ID: {self.client_id}")
            if self.connection_callback:
                self.connection_callback()
        else:
            print(f"MQTT连接失败，返回码: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """断开连接回调"""
        self.is_connected = False
        print(f"MQTT连接断开，返回码: {rc}")
        if self.disconnection_callback:
            self.disconnection_callback()
    
    def _on_message(self, client, userdata, msg):
        """消息接收回调"""
        try:
            topic = msg.topic
            payload = msg.payload.decode('utf-8')
            
            self.messages_received += 1
            
            # 尝试解析JSON
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                data = payload
            
            print(f"收到消息 - 主题: {topic}, 内容: {data}")
            
            # 调用特定主题的回调函数
            if topic in self.message_callbacks:
                self.message_callbacks[topic](topic, data)
            
        except Exception as e:
            print(f"处理消息异常: {e}")
    
    def _on_publish(self, client, userdata, mid):
        """发布回调"""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'client_id': self.client_id,
            'broker': f"{self.broker_host}:{self.broker_port}",
            'is_connected': self.is_connected,
            'subscribed_topics': list(self.subscribed_topics),
            'messages_sent': self.messages_sent,
            'messages_received': self.messages_received
        }
    
    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect()