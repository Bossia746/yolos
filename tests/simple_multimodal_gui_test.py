#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""简化版多模态识别GUI测试 - 专注于基本功能验证"""

import sys
import os
import cv2
import numpy as np
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

# 设置编码
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('simple_multimodal_test.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class SimpleMultimodalTester:
    """简化版多模态识别测试器"""
    
    def __init__(self):
        self.face_recognizer = None
        self.gesture_recognizer = None
        self.pose_recognizer = None
        self.cap = None
        self.running = False
        self.frame_count = 0
        self.detection_stats = {
            'faces': 0,
            'gestures': 0,
            'poses': 0
        }
        
    def initialize_recognizers(self):
        """初始化识别器"""
        logger.info("🚀 初始化识别器...")
        
        # 初始化面部识别器
        try:
            from recognition.face_recognizer import FaceRecognizer
            self.face_recognizer = FaceRecognizer()
            logger.info("✅ 面部识别器初始化成功")
        except Exception as e:
            logger.warning(f"⚠️ 面部识别器初始化失败: {e}")
        
        # 初始化手势识别器
        try:
            from recognition.gesture_recognizer import GestureRecognizer
            self.gesture_recognizer = GestureRecognizer()
            logger.info("✅ 手势识别器初始化成功")
        except Exception as e:
            logger.warning(f"⚠️ 手势识别器初始化失败: {e}")
        
        # 初始化姿势识别器
        try:
            from recognition.pose_recognizer import PoseRecognizer
            self.pose_recognizer = PoseRecognizer()
            logger.info("✅ 姿势识别器初始化成功")
        except Exception as e:
            logger.warning(f"⚠️ 姿势识别器初始化失败: {e}")
        
        return True
    
    def initialize_camera(self):
        """初始化摄像头"""
        logger.info("📷 初始化摄像头...")
        
        try:
            self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                logger.error("❌ 无法打开摄像头")
                return False
            
            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # 测试读取
            ret, frame = self.cap.read()
            if ret and frame is not None:
                logger.info(f"✅ 摄像头初始化成功: {frame.shape}")
                return True
            else:
                logger.error("❌ 摄像头测试读取失败")
                return False
                
        except Exception as e:
            logger.error(f"❌ 摄像头初始化失败: {e}")
            return False
    
    def detect_multimodal(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """执行多模态检测"""
        annotated_frame = frame.copy()
        results = {
            'faces': [],
            'gestures': [],
            'poses': []
        }
        
        # 面部检测
        if self.face_recognizer:
            try:
                face_frame, face_results = self.face_recognizer.detect_faces(frame)
                if face_results:
                    results['faces'] = face_results
                    self.detection_stats['faces'] += len(face_results)
                    annotated_frame = face_frame
            except Exception as e:
                logger.debug(f"面部检测错误: {e}")
        
        # 手势检测
        if self.gesture_recognizer:
            try:
                gesture_frame, gesture_results = self.gesture_recognizer.detect_hands(annotated_frame)
                if gesture_results:
                    results['gestures'] = gesture_results
                    self.detection_stats['gestures'] += len(gesture_results)
                    annotated_frame = gesture_frame
            except Exception as e:
                logger.debug(f"手势检测错误: {e}")
        
        # 姿势检测
        if self.pose_recognizer:
            try:
                pose_frame, pose_result = self.pose_recognizer.detect_pose(annotated_frame)
                if pose_result and pose_result.get('pose_detected', False):
                    results['poses'] = [pose_result]
                    self.detection_stats['poses'] += 1
                    annotated_frame = pose_frame
            except Exception as e:
                logger.debug(f"姿势检测错误: {e}")
        
        return annotated_frame, results
    
    def draw_info_overlay(self, frame: np.ndarray, fps: float):
        """绘制信息覆盖层"""
        try:
            # 创建半透明背景
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # 绘制文本信息
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            color = (255, 255, 255)
            thickness = 1
            
            # 当前时间
            current_time = datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, f"Time: {current_time}", (10, 25), font, font_scale, color, thickness)
            
            # FPS信息
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 50), font, font_scale, color, thickness)
            
            # 帧计数
            cv2.putText(frame, f"Frame: {self.frame_count}", (10, 75), font, font_scale, color, thickness)
            
            # 检测统计
            stats_text = f"Faces: {self.detection_stats['faces']} | "
            stats_text += f"Gestures: {self.detection_stats['gestures']} | "
            stats_text += f"Poses: {self.detection_stats['poses']}"
            cv2.putText(frame, stats_text, (10, 100), font, 0.5, color, thickness)
            
            # 控制说明
            control_text = "Controls: [Q]uit | [S]ave | [R]eset | [SPACE]Pause"
            cv2.putText(frame, control_text, (10, frame.shape[0] - 10), font, 0.4, (0, 255, 0), 1)
            
        except Exception as e:
            logger.warning(f"绘制信息覆盖层失败: {e}")
    
    def save_screenshot(self, frame: np.ndarray):
        """保存截图"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"multimodal_screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            logger.info(f"📸 截图已保存: {filename}")
        except Exception as e:
            logger.error(f"保存截图失败: {e}")
    
    def reset_stats(self):
        """重置统计"""
        self.detection_stats = {'faces': 0, 'gestures': 0, 'poses': 0}
        self.frame_count = 0
        logger.info("📊 统计已重置")
    
    def run_test(self):
        """运行测试"""
        logger.info("🖥️ 开始简化版多模态识别GUI测试")
        
        # 初始化识别器
        if not self.initialize_recognizers():
            logger.error("❌ 识别器初始化失败")
            return False
        
        # 初始化摄像头
        if not self.initialize_camera():
            logger.error("❌ 摄像头初始化失败")
            return False
        
        # 创建窗口
        window_name = "Simple Multimodal Recognition Test"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        logger.info("🎬 开始实时检测，按 'q' 退出...")
        
        self.running = True
        paused = False
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        try:
            while self.running:
                if not paused:
                    # 读取摄像头帧
                    ret, frame = self.cap.read()
                    
                    if not ret or frame is None:
                        logger.warning("⚠️ 无法读取摄像头帧")
                        continue
                    
                    self.frame_count += 1
                    fps_counter += 1
                    
                    # 执行多模态检测
                    start_time = time.time()
                    try:
                        annotated_frame, results = self.detect_multimodal(frame)
                        processing_time = time.time() - start_time
                    except Exception as e:
                        logger.warning(f"检测处理失败: {e}")
                        annotated_frame = frame
                        processing_time = 0
                    
                    # 计算FPS
                    current_time = time.time()
                    if current_time - fps_start_time >= 1.0:
                        current_fps = fps_counter / (current_time - fps_start_time)
                        fps_counter = 0
                        fps_start_time = current_time
                    
                    # 绘制信息覆盖层
                    self.draw_info_overlay(annotated_frame, current_fps)
                    
                    # 显示帧
                    cv2.imshow(window_name, annotated_frame)
                
                # 处理键盘输入
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' 或 ESC 退出
                    logger.info("用户请求退出")
                    break
                elif key == ord('s'):  # 's' 保存截图
                    if 'annotated_frame' in locals():
                        self.save_screenshot(annotated_frame)
                elif key == ord('r'):  # 'r' 重置统计
                    self.reset_stats()
                elif key == ord(' '):  # 空格键暂停/继续
                    paused = not paused
                    status = "暂停" if paused else "继续"
                    logger.info(f"📹 视频 {status}")
                
                # 检查窗口是否被关闭
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("窗口被关闭")
                    break
            
            # 生成测试报告
            self.generate_report(current_fps)
            
            logger.info("✅ 测试完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 测试过程中发生错误: {e}")
            return False
        
        finally:
            self.cleanup()
    
    def generate_report(self, fps: float):
        """生成测试报告"""
        logger.info("\n" + "="*60)
        logger.info("📊 多模态识别测试报告")
        logger.info("="*60)
        logger.info(f"总帧数: {self.frame_count}")
        logger.info(f"平均FPS: {fps:.1f}")
        logger.info(f"面部检测次数: {self.detection_stats['faces']}")
        logger.info(f"手势检测次数: {self.detection_stats['gestures']}")
        logger.info(f"姿势检测次数: {self.detection_stats['poses']}")
        
        # 计算检测率
        if self.frame_count > 0:
            face_rate = (self.detection_stats['faces'] / self.frame_count) * 100
            gesture_rate = (self.detection_stats['gestures'] / self.frame_count) * 100
            pose_rate = (self.detection_stats['poses'] / self.frame_count) * 100
            
            logger.info(f"面部检测率: {face_rate:.1f}%")
            logger.info(f"手势检测率: {gesture_rate:.1f}%")
            logger.info(f"姿势检测率: {pose_rate:.1f}%")
        
        logger.info("="*60)
    
    def cleanup(self):
        """清理资源"""
        logger.info("🧹 清理资源...")
        
        self.running = False
        
        # 释放摄像头
        if self.cap:
            self.cap.release()
        
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()
        
        # 清理识别器
        if self.face_recognizer and hasattr(self.face_recognizer, 'close'):
            self.face_recognizer.close()
        if self.gesture_recognizer and hasattr(self.gesture_recognizer, 'close'):
            self.gesture_recognizer.close()
        if self.pose_recognizer and hasattr(self.pose_recognizer, 'close'):
            self.pose_recognizer.close()
        
        logger.info("✅ 资源清理完成")


def main():
    """主函数"""
    print("🖥️ 简化版多模态识别GUI测试")
    print("=" * 50)
    print("功能说明:")
    print("- 实时显示摄像头画面")
    print("- 同时进行面部、手势、身体姿势识别")
    print("- 显示FPS和检测统计")
    print("- 按 'q' 或 ESC 退出")
    print("- 按 's' 保存截图")
    print("- 按 'r' 重置统计")
    print("- 按空格键暂停/继续")
    print("=" * 50)
    
    try:
        tester = SimpleMultimodalTester()
        success = tester.run_test()
        
        if success:
            print("\n🎉 测试完成！")
            sys.exit(0)
        else:
            print("\n❌ 测试失败！")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ 测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 测试过程发生异常: {e}")
        logger.error(f"主程序异常: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()