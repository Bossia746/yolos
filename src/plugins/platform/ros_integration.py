#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS集成插件
支持ROS1和ROS2环境下的混合识别系统
"""

import os
import sys
import logging
import threading
import time
from typing import Dict, List, Optional, Any, Callable
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# 检测ROS版本
ROS_VERSION = os.environ.get('ROS_VERSION', '0')

if ROS_VERSION == '1':
    # ROS1 imports
    try:
        import rospy
        from sensor_msgs.msg import Image, CompressedImage
        from std_msgs.msg import String, Header
        from geometry_msgs.msg import Point, Pose2D
        from cv_bridge import CvBridge
        ROS1_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"ROS1导入失败: {e}")
        ROS1_AVAILABLE = False
        
elif ROS_VERSION == '2':
    # ROS2 imports
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image, CompressedImage
        from std_msgs.msg import String, Header
        from geometry_msgs.msg import Point, Pose2D
        from cv_bridge import CvBridge
        ROS2_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"ROS2导入失败: {e}")
        ROS2_AVAILABLE = False
else:
    ROS1_AVAILABLE = False
    ROS2_AVAILABLE = False

# 导入混合识别系统
try:
    from ...recognition.hybrid_recognition_system import HybridRecognitionSystem, RecognitionRequest, RecognitionResponse
except ImportError:
    logger.error("无法导入混合识别系统")
    HybridRecognitionSystem = None

class ROSHybridRecognitionNode:
    """ROS混合识别节点基类"""
    
    def __init__(self, node_name: str = "hybrid_recognition_node"):
        self.node_name = node_name
        self.bridge = CvBridge()
        self.recognition_system = None
        
        # 初始化混合识别系统
        if HybridRecognitionSystem:
            try:
                self.recognition_system = HybridRecognitionSystem()
                logger.info("混合识别系统初始化成功")
            except Exception as e:
                logger.error(f"混合识别系统初始化失败: {e}")
        
        # 回调函数
        self.result_callbacks: List[Callable] = []
        
        logger.info(f"ROS混合识别节点初始化: {node_name}")

class ROS1HybridRecognitionNode(ROSHybridRecognitionNode):
    """ROS1混合识别节点"""
    
    def __init__(self, node_name: str = "hybrid_recognition_node"):
        if not ROS1_AVAILABLE:
            raise ImportError("ROS1不可用")
        
        super().__init__(node_name)
        
        # 初始化ROS1节点
        rospy.init_node(self.node_name, anonymous=True)
        
        # 订阅者
        self.image_sub = rospy.Subscriber(
            '/camera/image_raw', Image, self.image_callback, queue_size=1
        )
        self.compressed_image_sub = rospy.Subscriber(
            '/camera/image_raw/compressed', CompressedImage, 
            self.compressed_image_callback, queue_size=1
        )
        self.scene_sub = rospy.Subscriber(
            '/recognition/scene_request', String, self.scene_callback, queue_size=10
        )
        
        # 发布者
        self.result_pub = rospy.Publisher(
            '/recognition/results', String, queue_size=10
        )
        self.annotated_image_pub = rospy.Publisher(
            '/recognition/annotated_image', Image, queue_size=1
        )
        
        # 服务（如果需要）
        # self.recognition_service = rospy.Service(
        #     '/recognition/recognize', RecognitionSrv, self.recognition_service_callback
        # )
        
        # 当前场景
        self.current_scene = "pets"
        
        logger.info("ROS1混合识别节点启动完成")
    
    def image_callback(self, msg: Image):
        """图像回调函数"""
        try:
            # 转换ROS图像到OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 执行识别
            self.process_image(cv_image, msg.header)
            
        except Exception as e:
            logger.error(f"图像处理失败: {e}")
    
    def compressed_image_callback(self, msg: CompressedImage):
        """压缩图像回调函数"""
        try:
            # 解压缩图像
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # 执行识别
            self.process_image(cv_image, msg.header)
            
        except Exception as e:
            logger.error(f"压缩图像处理失败: {e}")
    
    def scene_callback(self, msg: String):
        """场景切换回调"""
        self.current_scene = msg.data
        logger.info(f"切换识别场景: {self.current_scene}")
    
    def process_image(self, cv_image: np.ndarray, header: Header):
        """处理图像并执行识别"""
        if self.recognition_system is None:
            return
        
        try:
            # 创建识别请求
            request = RecognitionRequest(
                scene=self.current_scene,
                image=cv_image,
                timestamp=time.time(),
                priority=1,
                use_online=True
            )
            
            # 执行识别
            response = self.recognition_system.recognize(request)
            
            # 发布结果
            self.publish_results(response, header)
            
            # 发布标注图像
            self.publish_annotated_image(cv_image, response, header)
            
        except Exception as e:
            logger.error(f"识别处理失败: {e}")
    
    def publish_results(self, response: RecognitionResponse, header: Header):
        """发布识别结果"""
        try:
            import json
            
            result_data = {
                'scene': response.scene,
                'results': response.results,
                'confidence': response.confidence,
                'processing_time': response.processing_time,
                'source': response.source,
                'timestamp': response.timestamp,
                'header': {
                    'seq': header.seq,
                    'stamp': {
                        'secs': header.stamp.secs,
                        'nsecs': header.stamp.nsecs
                    },
                    'frame_id': header.frame_id
                }
            }
            
            result_msg = String()
            result_msg.data = json.dumps(result_data)
            
            self.result_pub.publish(result_msg)
            
        except Exception as e:
            logger.error(f"结果发布失败: {e}")
    
    def publish_annotated_image(self, cv_image: np.ndarray, 
                              response: RecognitionResponse, header: Header):
        """发布标注图像"""
        try:
            # 在图像上绘制识别结果
            annotated_image = cv_image.copy()
            
            for result in response.results:
                if 'bbox' in result:
                    x, y, w, h = result['bbox']
                    
                    # 绘制边界框
                    cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # 绘制标签
                    label = f"{result.get('class_name', 'unknown')}: {result.get('confidence', 0):.2f}"
                    cv2.putText(annotated_image, label, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 添加系统信息
            info_text = f"Scene: {response.scene} | Source: {response.source} | Time: {response.processing_time:.3f}s"
            cv2.putText(annotated_image, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 转换并发布
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
            annotated_msg.header = header
            
            self.annotated_image_pub.publish(annotated_msg)
            
        except Exception as e:
            logger.error(f"标注图像发布失败: {e}")
    
    def run(self):
        """运行节点"""
        logger.info("ROS1混合识别节点开始运行...")
        rospy.spin()

class ROS2HybridRecognitionNode(Node, ROSHybridRecognitionNode):
    """ROS2混合识别节点"""
    
    def __init__(self, node_name: str = "hybrid_recognition_node"):
        if not ROS2_AVAILABLE:
            raise ImportError("ROS2不可用")
        
        # 初始化ROS2节点
        Node.__init__(self, node_name)
        ROSHybridRecognitionNode.__init__(self, node_name)
        
        # 订阅者
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 1
        )
        self.compressed_image_sub = self.create_subscription(
            CompressedImage, '/camera/image_raw/compressed', 
            self.compressed_image_callback, 1
        )
        self.scene_sub = self.create_subscription(
            String, '/recognition/scene_request', self.scene_callback, 10
        )
        
        # 发布者
        self.result_pub = self.create_publisher(String, '/recognition/results', 10)
        self.annotated_image_pub = self.create_publisher(
            Image, '/recognition/annotated_image', 1
        )
        
        # 服务（如果需要）
        # self.recognition_service = self.create_service(
        #     RecognitionSrv, '/recognition/recognize', self.recognition_service_callback
        # )
        
        # 当前场景
        self.current_scene = "pets"
        
        logger.info("ROS2混合识别节点启动完成")
    
    def image_callback(self, msg: Image):
        """图像回调函数"""
        try:
            # 转换ROS图像到OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 执行识别
            self.process_image(cv_image, msg.header)
            
        except Exception as e:
            self.get_logger().error(f"图像处理失败: {e}")
    
    def compressed_image_callback(self, msg: CompressedImage):
        """压缩图像回调函数"""
        try:
            # 解压缩图像
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # 执行识别
            self.process_image(cv_image, msg.header)
            
        except Exception as e:
            self.get_logger().error(f"压缩图像处理失败: {e}")
    
    def scene_callback(self, msg: String):
        """场景切换回调"""
        self.current_scene = msg.data
        self.get_logger().info(f"切换识别场景: {self.current_scene}")
    
    def process_image(self, cv_image: np.ndarray, header: Header):
        """处理图像并执行识别"""
        if self.recognition_system is None:
            return
        
        try:
            # 创建识别请求
            request = RecognitionRequest(
                scene=self.current_scene,
                image=cv_image,
                timestamp=time.time(),
                priority=1,
                use_online=True
            )
            
            # 执行识别
            response = self.recognition_system.recognize(request)
            
            # 发布结果
            self.publish_results(response, header)
            
            # 发布标注图像
            self.publish_annotated_image(cv_image, response, header)
            
        except Exception as e:
            self.get_logger().error(f"识别处理失败: {e}")
    
    def publish_results(self, response: RecognitionResponse, header: Header):
        """发布识别结果"""
        try:
            import json
            
            result_data = {
                'scene': response.scene,
                'results': response.results,
                'confidence': response.confidence,
                'processing_time': response.processing_time,
                'source': response.source,
                'timestamp': response.timestamp,
                'header': {
                    'stamp': {
                        'sec': header.stamp.sec,
                        'nanosec': header.stamp.nanosec
                    },
                    'frame_id': header.frame_id
                }
            }
            
            result_msg = String()
            result_msg.data = json.dumps(result_data)
            
            self.result_pub.publish(result_msg)
            
        except Exception as e:
            self.get_logger().error(f"结果发布失败: {e}")
    
    def publish_annotated_image(self, cv_image: np.ndarray, 
                              response: RecognitionResponse, header: Header):
        """发布标注图像"""
        try:
            # 在图像上绘制识别结果
            annotated_image = cv_image.copy()
            
            for result in response.results:
                if 'bbox' in result:
                    x, y, w, h = result['bbox']
                    
                    # 绘制边界框
                    cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # 绘制标签
                    label = f"{result.get('class_name', 'unknown')}: {result.get('confidence', 0):.2f}"
                    cv2.putText(annotated_image, label, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 添加系统信息
            info_text = f"Scene: {response.scene} | Source: {response.source} | Time: {response.processing_time:.3f}s"
            cv2.putText(annotated_image, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 转换并发布
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
            annotated_msg.header = header
            
            self.annotated_image_pub.publish(annotated_msg)
            
        except Exception as e:
            self.get_logger().error(f"标注图像发布失败: {e}")

def create_ros_node(node_name: str = "hybrid_recognition_node"):
    """创建ROS节点"""
    if ROS_VERSION == '1' and ROS1_AVAILABLE:
        return ROS1HybridRecognitionNode(node_name)
    elif ROS_VERSION == '2' and ROS2_AVAILABLE:
        return ROS2HybridRecognitionNode(node_name)
    else:
        raise RuntimeError(f"不支持的ROS版本或ROS不可用: ROS{ROS_VERSION}")

def main():
    """主函数"""
    try:
        # 创建ROS节点
        node = create_ros_node()
        
        if ROS_VERSION == '1':
            # ROS1运行方式
            node.run()
        elif ROS_VERSION == '2':
            # ROS2运行方式
            rclpy.spin(node)
            node.destroy_node()
            rclpy.shutdown()
            
    except KeyboardInterrupt:
        logger.info("节点被用户中断")
    except Exception as e:
        logger.error(f"节点运行失败: {e}")

if __name__ == "__main__":
    main()