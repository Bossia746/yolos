#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
稳定版YOLOS GUI - 基于BaseYOLOSGUI的稳定实现
使用OpenCV窗口显示，解决摄像头和参数问题
"""

import cv2
import numpy as np
import time
import json
import logging
import random
import tkinter as tk
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import sys
import os

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.gui.base_gui import BaseYOLOSGUI

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stable_yolos.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class StableYOLOSDetector:
    """稳定版YOLOS检测器 - 模拟YOLO检测"""
    
    def __init__(self):
        # 检测参数
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        
        # 模拟检测历史
        self.detection_history = []
        self.max_history = 10
        
        # 模拟目标类别
        self.classes = ['person', 'car', 'dog', 'cat', 'bicycle', 'bottle', 'chair', 'book']
        
    def update_parameters(self, confidence: float, nms: float):
        """更新检测参数"""
        self.confidence_threshold = confidence
        self.nms_threshold = nms
        
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """模拟目标检测"""
        detections = []
        h, w = frame.shape[:2]
        
        # 模拟检测结果 - 添加随机性来模拟真实检测
        num_objects = random.randint(1, 3)
        
        for i in range(num_objects):
            # 随机选择类别
            class_name = random.choice(self.classes)
            
            # 生成随机但合理的检测框
            x = random.randint(50, w - 200)
            y = random.randint(50, h - 150)
            width = random.randint(80, 200)
            height = random.randint(60, 150)
            
            # 确保检测框在图像范围内
            x = max(0, min(x, w - width))
            y = max(0, min(y, h - height))
            
            # 生成置信度（受阈值影响）
            confidence = random.uniform(self.confidence_threshold, 1.0)
            
            detection = {
                'class': class_name,
                'confidence': confidence,
                'bbox': (x, y, width, height),
                'center': (x + width//2, y + height//2)
            }
            
            detections.append(detection)
        
        # 记录检测历史
        self.detection_history.append(len(detections))
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)
            
        return detections


class StableYOLOSGUI(BaseYOLOSGUI):
    """稳定版YOLOS GUI - 使用OpenCV窗口显示"""
    
    def __init__(self):
        # 初始化检测器
        self.detector = StableYOLOSDetector()
        
        # 使用OpenCV显示模式
        self.use_opencv_display = True
        
        super().__init__(title="YOLOS - 稳定版检测系统", 
                        config_file="stable_gui_config.json")
        
        # OpenCV窗口相关
        self.frame_width = 640
        self.frame_height = 480
        
        # 检测控制
        self.detection_interval = 3  # 每3帧检测一次
        self.display_interval = 1    # 每帧更新显示
        self.result_hold_frames = 15 # 结果保持15帧
        
        # 结果缓存
        self.cached_results = []
        self.result_cache_time = 0
        
        # 初始化时间
        self.start_time = time.time()
        
        # 参数调整步长
        self.param_adjustment_step = 0.05
        
        # 统计信息
        self.stats = {
            'total_detections': 0,
            'detection_types': {},
            'session_start': time.time()
        }
        
        # 初始化摄像头
        if not self.initialize_camera():
            logger.error("摄像头初始化失败")
            # 不直接返回，允许程序继续运行
        
    def setup_ui(self):
        """重写基类UI设置，使用OpenCV显示"""
        # 稳定版使用OpenCV窗口，不需要Tkinter界面
        pass
    
    def load_model(self, model_path: str) -> bool:
        """加载模型（稳定版使用内置检测器）"""
        try:
            logger.info(f"稳定版使用内置检测器，模型路径: {model_path}")
            return True
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False
    
    def perform_detection(self, frame):
        """执行目标检测"""
        try:
            # 执行检测
            results = self.detector.detect_objects(frame)
            
            # 更新统计信息
            self.stats['total_detections'] += len(results)
            for result in results:
                class_name = result['class']
                self.stats['detection_types'][class_name] = self.stats['detection_types'].get(class_name, 0) + 1
            
            # 缓存结果
            self.cached_results = results
            self.result_cache_time = time.time()
            
            return results
            
        except Exception as e:
            logger.error(f"检测失败: {e}")
            return []
    
    def process_frame(self, frame):
        """处理单帧图像"""
        # 计算FPS
        self.frame_count += 1
        current_time = time.time()
        if self.frame_count % 30 == 0:
            elapsed = current_time - self.start_time
            self.fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        # 执行检测
        results = []
        if self.is_detecting:
            results = self.perform_detection(frame)
        
        return frame, results
    
    def get_detection_results(self) -> List[Dict]:
        """获取检测结果"""
        return getattr(self, 'cached_results', [])
    
    def on_model_changed(self, model_path: str):
        """模型变更回调"""
        logger.info(f"模型变更: {model_path}")
        self.load_model(model_path)
    
    def initialize_camera(self) -> bool:
        """初始化摄像头 - 优先使用内置摄像头"""
        logger.info("正在初始化摄像头...")
        
        # 首先尝试内置摄像头 (index 0)
        logger.info("尝试内置摄像头 (索引 0)")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                self.cap = cap
                logger.info(f"内置摄像头启动成功: {frame.shape}")
            else:
                cap.release()
        
        # 如果内置摄像头失败，尝试外部摄像头
        if self.cap is None:
            for index in [1, 2, 3, 4]:
                logger.info(f"尝试摄像头索引 {index}")
                cap = cv2.VideoCapture(index)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.cap = cap
                        logger.info(f"摄像头 {index} 启动成功: {frame.shape}")
                        break
                    else:
                        cap.release()
                else:
                    cap.release()
        
        if self.cap is None:
            logger.error("没有可用的摄像头")
            return False
        
        # 设置摄像头参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 获取实际尺寸
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"摄像头设置完成: {self.frame_width}x{self.frame_height}")
        return True
    
    def draw_info_panel(self, frame: np.ndarray, results: List[Dict]) -> np.ndarray:
        """绘制信息面板"""
        # 左上角信息面板
        panel_width = 280
        panel_height = 120
        
        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 统计信息
        avg_detections = np.mean(self.detector.detection_history) if self.detector.detection_history else 0
        
        # 显示信息
        font_scale = 0.5
        thickness = 1
        y_offset = 25
        
        # 基本信息
        cv2.putText(frame, f"检测数量: {len(results)}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        cv2.putText(frame, f"检测状态: {'开启' if self.is_detecting else '关闭'}", 
                   (10, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0) if self.is_detecting else (0, 0, 255), thickness)
        
        cv2.putText(frame, f"平均检测: {avg_detections:.1f}", 
                   (10, y_offset + 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        cv2.putText(frame, f"FPS: {self.fps:.1f} | 帧数: {self.frame_count}", 
                   (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        # 参数信息
        cv2.putText(frame, f"置信度: {self.confidence_threshold:.2f} | NMS: {self.nms_threshold:.2f}", 
                   (10, y_offset + 80), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), thickness)
        
        return frame
    
    def draw_controls_help(self, frame: np.ndarray) -> np.ndarray:
        """绘制控制帮助"""
        # 底部控制说明
        help_height = 60
        help_y = self.frame_height - help_height
        
        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, help_y), (self.frame_width, self.frame_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # 控制说明
        controls = [
            "控制: [空格]开始/停止检测 [↑↓]调整置信度 [←→]调整NMS [S]截图 [R]重置 [Q/ESC]退出",
            f"当前参数: 置信度={self.confidence_threshold:.2f} NMS={self.nms_threshold:.2f}"
        ]
        
        font_scale = 0.4
        thickness = 1
        
        for i, text in enumerate(controls):
            cv2.putText(frame, text, (5, help_y + 20 + i*15), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        return frame
    
    def draw_detections(self, frame: np.ndarray, results: List[Dict]) -> np.ndarray:
        """绘制检测结果"""
        for detection in results:
            class_name = detection['class']
            confidence = detection['confidence']
            x, y, w, h = detection['bbox']
            
            # 根据置信度选择颜色
            if confidence > 0.8:
                color = (0, 255, 0)  # 绿色 - 高置信度
            elif confidence > 0.6:
                color = (0, 255, 255)  # 黄色 - 中等置信度
            else:
                color = (0, 165, 255)  # 橙色 - 低置信度
            
            # 绘制边界框
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # 绘制标签
            label = f"{class_name}: {confidence:.2f}"
            
            # 标签背景
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x, y - label_h - 10), (x + label_w + 4, y), color, -1)
            cv2.putText(frame, label, (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 绘制中心点
            center_x, center_y = detection['center']
            cv2.circle(frame, (center_x, center_y), 3, color, -1)
        
        return frame
    
    def update_parameters(self, confidence_delta: float = 0, nms_delta: float = 0):
        """更新检测参数"""
        # 更新置信度
        if confidence_delta != 0:
            self.confidence_threshold = max(0.1, min(1.0, self.confidence_threshold + confidence_delta))
            
        # 更新NMS阈值
        if nms_delta != 0:
            self.nms_threshold = max(0.1, min(1.0, self.nms_threshold + nms_delta))
            
        # 更新检测器参数
        self.detector.update_parameters(self.confidence_threshold, self.nms_threshold)
        
        logger.info(f"参数更新: 置信度={self.confidence_threshold:.2f}, NMS={self.nms_threshold:.2f}")
    
    def log_detections(self, results: List[Dict]):
        """记录检测结果"""
        if results:
            for detection in results:
                class_name = detection['class']
                confidence = detection['confidence']
                
                log_msg = f"[检测] 类别: {class_name}, 置信度: {confidence:.3f}"
                logger.info(log_msg)
            
            # 更新统计
            self.stats['total_detections'] += len(results)
    
    def run(self):
        """运行主循环"""
        if not self.cap:
            logger.error("摄像头未初始化")
            return
        
        logger.info("启动稳定版YOLOS GUI...")
        logger.info("检测方法: 模拟YOLO目标检测")
        logger.info("控制说明: 空格键切换检测，方向键调整参数，Q键退出")
        
        # 稳定化等待
        logger.info("摄像头稳定中... (30帧)")
        for i in range(30):
            ret, frame = self.cap.read()
            if not ret:
                logger.error("稳定化期间读取帧失败")
                return
            cv2.imshow('YOLOS - 稳定化中...', frame)
            cv2.waitKey(1)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("从摄像头读取帧失败")
                    break
                
                self.frame_count += 1
                current_time = time.time()
                
                # 计算FPS
                if self.frame_count % 30 == 0:
                    elapsed = current_time - self.start_time
                    self.fps = self.frame_count / elapsed if elapsed > 0 else 0
                
                # 检测控制
                should_detect = (self.frame_count % self.detection_interval == 0) and self.is_detecting
                should_update_display = (self.frame_count % self.display_interval == 0)
                
                # 执行检测
                if should_detect:
                    frame, results = self.process_frame(frame)
                    
                    if results:
                        self.cached_results = results
                        self.result_cache_time = self.frame_count
                        self.log_detections(results)
                else:
                    frame, _ = self.process_frame(frame)
                
                # 使用缓存结果
                display_results = []
                if self.cached_results and (self.frame_count - self.result_cache_time) < self.result_hold_frames:
                    display_results = self.cached_results
                
                # 绘制结果
                if display_results and should_update_display:
                    frame = self.draw_detections(frame, display_results)
                
                # 绘制界面元素
                frame = self.draw_info_panel(frame, display_results)
                frame = self.draw_controls_help(frame)
                
                # 显示画面
                cv2.imshow('YOLOS - 稳定版目标检测系统', frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord(' '):  # 空格键切换检测
                    self.is_detecting = not self.is_detecting
                    status = "开启" if self.is_detecting else "关闭"
                    logger.info(f"检测状态: {status}")
                    print(f"🎯 检测{status}")
                elif key == 82:  # 上箭头 - 增加置信度
                    self.update_parameters(confidence_delta=0.05)
                elif key == 84:  # 下箭头 - 减少置信度
                    self.update_parameters(confidence_delta=-0.05)
                elif key == 81:  # 左箭头 - 减少NMS
                    self.update_parameters(nms_delta=-0.05)
                elif key == 83:  # 右箭头 - 增加NMS
                    self.update_parameters(nms_delta=0.05)
                elif key == ord('s'):  # 保存截图
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"yolos_screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    logger.info(f"截图已保存: {filename}")
                    print(f"📸 截图已保存: {filename}")
                elif key == ord('r'):  # 重置统计
                    self.stats = {
                        'total_detections': 0,
                        'detection_types': {},
                        'session_start': time.time()
                    }
                    self.detector.detection_history = []
                    logger.info("统计已重置")
                    print("📊 统计已重置")
        
        except KeyboardInterrupt:
            logger.info("用户中断")
        except Exception as e:
            logger.error(f"主循环错误: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        logger.info("清理资源中...")
        
        # 显示最终统计
        session_duration = time.time() - self.stats['session_start']
        
        stats_text = f"""
🎯 YOLOS 稳定版会话总结
========================================
持续时间: {session_duration:.1f} 秒
总帧数: {self.frame_count}
平均FPS: {self.fps:.1f}
总检测数: {self.stats['total_detections']}
最终参数: 置信度={self.confidence_threshold:.2f}, NMS={self.nms_threshold:.2f}
"""
        
        print(stats_text)
        logger.info("会话完成")
        
        # 释放摄像头
        if self.cap:
            self.cap.release()
        
        # 关闭窗口
        cv2.destroyAllWindows()


def main():
    """主函数"""
    print("🎯 YOLOS 稳定版目标检测系统")
    print("基于稳定的摄像头架构")
    print("=" * 45)
    print("功能特性:")
    print("  🎥 稳定的摄像头处理")
    print("  🎯 实时目标检测模拟")
    print("  ⚙️ 动态参数调整")
    print("  📊 性能监控")
    print("  🔧 交互式控制")
    print()
    
    try:
        gui = StableYOLOSGUI()
        gui.run()
        
    except Exception as e:
        logger.error(f"启动YOLOS系统失败: {e}")
        print(f"❌ 错误: {e}")


if __name__ == "__main__":
    main()