#!/usr/bin/env python3
"""
YOLOS ROS检测节点
支持ROS1和ROS2
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# 添加YOLOS路径
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent / "src"))

try:
    # ROS1
    import rospy
    from sensor_msgs.msg import Image
    from std_msgs.msg import Header
    from cv_bridge import CvBridge
    from yolos_ros.msg import Detection, DetectionArray
    ROS_VERSION = 1
except ImportError:
    try:
        # ROS2
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image
        from std_msgs.msg import Header
        from cv_bridge import CvBridge
        from yolos_ros.msg import Detection, DetectionArray
        ROS_VERSION = 2
    except ImportError:
        print("未找到ROS环境")
        sys.exit(1)

from models.yolo_factory import YOLOFactory


class YOLOSDetectionNode:
    """YOLOS检测节点"""
    
    def __init__(self):
        self.bridge = CvBridge()
        
        if ROS_VERSION == 1:
            self._init_ros1()
        else:
            self._init_ros2()
        
        # 初始化YOLO模型
        self.model_type = self.get_param('model_type', 'yolov8')
        self.model_path = self.get_param('model_path', None)
        self.device = self.get_param('device', 'auto')
        
        self.model = YOLOFactory.create_model(
            self.model_type, 
            self.model_path, 
            self.device
        )
        
        self.log_info(f"YOLOS检测节点已启动 - 模型: {self.model_type}")
    
    def _init_ros1(self):
        """初始化ROS1"""
        rospy.init_node('yolos_detection_node', anonymous=True)
        
        # 订阅图像话题
        self.image_sub = rospy.Subscriber(
            '/camera/image_raw', 
            Image, 
            self.image_callback
        )
        
        # 发布检测结果
        self.detection_pub = rospy.Publisher(
            '/yolos/detections', 
            DetectionArray, 
            queue_size=10
        )
        
        # 发布标注图像
        self.image_pub = rospy.Publisher(
            '/yolos/image_annotated', 
            Image, 
            queue_size=10
        )
    
    def _init_ros2(self):
        """初始化ROS2"""
        rclpy.init()
        self.node = Node('yolos_detection_node')
        
        # 订阅图像话题
        self.image_sub = self.node.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        # 发布检测结果
        self.detection_pub = self.node.create_publisher(
            DetectionArray,
            '/yolos/detections',
            10
        )
        
        # 发布标注图像
        self.image_pub = self.node.create_publisher(
            Image,
            '/yolos/image_annotated',
            10
        )
    
    def get_param(self, name, default):
        """获取参数"""
        if ROS_VERSION == 1:
            return rospy.get_param(f'~{name}', default)
        else:
            return self.node.get_parameter_or(name, default).value
    
    def log_info(self, msg):
        """记录信息"""
        if ROS_VERSION == 1:
            rospy.loginfo(msg)
        else:
            self.node.get_logger().info(msg)
    
    def image_callback(self, msg):
        """图像回调函数"""
        try:
            # 转换图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 执行检测
            results = self.model.predict(cv_image)
            
            # 发布检测结果
            self.publish_detections(results, msg.header)
            
            # 发布标注图像
            annotated_image = self.model.draw_results(cv_image, results)
            self.publish_annotated_image(annotated_image, msg.header)
            
        except Exception as e:
            self.log_info(f"检测过程中出错: {e}")
    
    def publish_detections(self, results, header):
        """发布检测结果"""
        detection_array = DetectionArray()
        detection_array.header = header
        detection_array.detection_count = len(results)
        detection_array.model_name = self.model_type
        
        for result in results:
            detection = Detection()
            detection.class_name = result['class_name']
            detection.class_id = result['class_id']
            detection.confidence = result['confidence']
            
            # 计算中心点
            x1, y1, x2, y2 = result['bbox']
            detection.center.x = (x1 + x2) / 2
            detection.center.y = (y1 + y2) / 2
            detection.center.z = 0.0
            
            detection.width = x2 - x1
            detection.height = y2 - y1
            
            detection_array.detections.append(detection)
        
        self.detection_pub.publish(detection_array)
    
    def publish_annotated_image(self, image, header):
        """发布标注图像"""
        try:
            img_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
            img_msg.header = header
            self.image_pub.publish(img_msg)
        except Exception as e:
            self.log_info(f"发布标注图像失败: {e}")
    
    def run(self):
        """运行节点"""
        if ROS_VERSION == 1:
            rospy.spin()
        else:
            rclpy.spin(self.node)


def main():
    try:
        node = YOLOSDetectionNode()
        node.run()
    except KeyboardInterrupt:
        print("节点被用户中断")
    except Exception as e:
        print(f"节点运行出错: {e}")
    finally:
        if ROS_VERSION == 2:
            rclpy.shutdown()


if __name__ == '__main__':
    main()