#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI版多模态识别系统测试脚本 - 带窗口显示的真实测试"""

import sys
import os
import cv2
import numpy as np
import time
import logging
import traceback
from pathlib import Path
import json
import locale
import codecs
import threading
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

# 导入配置管理器
try:
    from core.config_manager import ConfigManager
    config_manager = ConfigManager(str(project_root / 'configs' / 'default_config.yaml'))
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    config_manager = None

# 设置编码
if sys.platform.startswith('win'):
    # Windows系统编码设置
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    
    # 设置控制台编码
    try:
        os.system('chcp 65001 > nul')
    except:
        pass
    
    # 设置环境变量
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['LANG'] = 'zh_CN.UTF-8'

# 设置locale
try:
    locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_ALL, 'Chinese_China.65001')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, '')
        except locale.Error:
            pass

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('gui_test.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 导入优化版多模态检测器
try:
    from recognition.optimized_multimodal_detector import (
        OptimizedMultimodalDetector,
        create_optimized_multimodal_detector,
        create_multimodal_detector_from_config,
        MULTIMODAL_DETECTOR_CONFIGS
    )
    OPTIMIZED_DETECTOR_AVAILABLE = True
    logger.info("✅ 优化版多模态检测器导入成功")
except ImportError as e:
    OPTIMIZED_DETECTOR_AVAILABLE = False
    logger.error(f"❌ 优化版多模态检测器导入失败: {e}")

# 备用导入
try:
    from recognition.multimodal_detector import MultimodalDetector
    FALLBACK_DETECTOR_AVAILABLE = True
    logger.info("✅ 备用多模态检测器导入成功")
except ImportError as e:
    FALLBACK_DETECTOR_AVAILABLE = False
    logger.error(f"❌ 备用多模态检测器导入失败: {e}")


class GUIMultimodalTester:
    """GUI版多模态识别系统测试器"""
    
    def __init__(self):
        self.detector = None
        self.cap = None
        self.running = False
        self.frame_count = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.detection_stats = {
            'total_detections': 0,
            'face_detections': 0,
            'gesture_detections': 0,
            'pose_detections': 0,
            'fall_detections': 0
        }
        
        # 创建测试结果目录
        self.results_dir = project_root / 'gui_test_results'
        self.results_dir.mkdir(exist_ok=True)
        
    def initialize_detector(self):
        """初始化检测器"""
        logger.info("🚀 初始化多模态检测器...")
        
        try:
            if OPTIMIZED_DETECTOR_AVAILABLE:
                # 使用优化版检测器，采用轻量级配置
                logger.info("使用优化版多模态检测器 - 平衡配置")
                
                # 读取配置文件中的face设置
                use_insightface = True  # 默认值
                if CONFIG_AVAILABLE and config_manager:
                    try:
                        face_config = config_manager.get_config().get('multimodal', {}).get('face', {})
                        use_insightface = face_config.get('use_insightface', True)
                        logger.info(f"从配置文件读取 use_insightface: {use_insightface}")
                    except Exception as e:
                        logger.warning(f"读取配置失败，使用默认值: {e}")
                
                # 使用平衡配置，确保所有功能正常工作
                self.detector = create_multimodal_detector_from_config(
                    config_name='balanced',
                    encoding='utf-8',
                    use_insightface=use_insightface
                )
                
                if self.detector:
                    logger.info("✅ 优化版检测器初始化成功")
                    try:
                        status = self.detector.get_system_status()
                        logger.info(f"活跃算法: {status.active_algorithms}")
                    except:
                        logger.info("检测器状态获取失败，但检测器已初始化")
                    return True
                else:
                    logger.error("❌ 优化版检测器初始化失败")
                    
            elif FALLBACK_DETECTOR_AVAILABLE:
                # 使用备用检测器
                logger.info("使用备用多模态检测器")
                self.detector = MultimodalDetector()
                if self.detector:
                    logger.info("✅ 备用检测器初始化成功")
                    return True
            
            logger.error("❌ 没有可用的检测器")
            return False
                
        except Exception as e:
            logger.error(f"❌ 检测器初始化失败: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            return False
    
    def initialize_camera(self):
        """初始化摄像头"""
        logger.info("📷 初始化摄像头...")
        
        try:
            # 尝试打开摄像头
            self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                logger.warning("⚠️ 无法打开默认摄像头，尝试其他摄像头...")
                
                # 尝试其他摄像头ID
                for camera_id in range(1, 5):
                    self.cap = cv2.VideoCapture(camera_id)
                    if self.cap.isOpened():
                        logger.info(f"✅ 摄像头 {camera_id} 可用")
                        break
                    self.cap.release()
                
                if not self.cap.isOpened():
                    logger.error("❌ 没有可用的摄像头")
                    return False
            
            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # 获取实际参数
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"摄像头参数: {width}x{height} @ {fps}fps")
            
            # 测试读取几帧
            for i in range(3):
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    logger.info(f"✅ 摄像头测试帧 {i+1} 读取成功: {frame.shape}")
                else:
                    logger.warning(f"⚠️ 摄像头测试帧 {i+1} 读取失败")
            
            logger.info("✅ 摄像头初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ 摄像头初始化失败: {e}")
            return False
    
    def draw_info_overlay(self, frame):
        """绘制信息覆盖层"""
        try:
            # 创建半透明背景
            overlay = frame.copy()
            
            # 绘制顶部信息栏
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # 绘制文本信息
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            color = (255, 255, 255)
            thickness = 1
            
            # 当前时间
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, f"Time: {current_time}", (10, 25), font, font_scale, color, thickness)
            
            # FPS信息
            cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 50), font, font_scale, color, thickness)
            
            # 帧计数
            cv2.putText(frame, f"Frame: {self.frame_count}", (10, 75), font, font_scale, color, thickness)
            
            # 检测统计 - 使用英文避免乱码
            stats_text = f"Face: {self.detection_stats['face_detections']} | "
            stats_text += f"Gesture: {self.detection_stats['gesture_detections']} | "
            stats_text += f"Pose: {self.detection_stats['pose_detections']} | "
            stats_text += f"Fall: {self.detection_stats['fall_detections']}"
            cv2.putText(frame, stats_text, (10, 100), font, 0.5, color, thickness)
            
            # 绘制底部控制信息
            control_text = "Controls: [Q]uit | [S]ave Screenshot | [R]eset Stats | [SPACE]Pause"
            text_size = cv2.getTextSize(control_text, font, 0.5, thickness)[0]
            cv2.rectangle(frame, (0, frame.shape[0] - 30), (text_size[0] + 20, frame.shape[0]), (0, 0, 0), -1)
            cv2.putText(frame, control_text, (10, frame.shape[0] - 10), font, 0.5, (255, 255, 255), thickness)
            
        except Exception as e:
            logger.warning(f"绘制信息覆盖层失败: {e}")
    
    def update_detection_stats(self, detection_result):
        """更新检测统计"""
        try:
            if detection_result:
                self.detection_stats['total_detections'] += 1
                
                # 统计各类检测结果
                if hasattr(detection_result, 'face_results') and detection_result.face_results:
                    self.detection_stats['face_detections'] += len(detection_result.face_results)
                
                if hasattr(detection_result, 'gesture_results') and detection_result.gesture_results:
                    # 检查gesture_results的类型
                    gesture_results = detection_result.gesture_results
                    if hasattr(gesture_results, 'hands_detected'):
                        # GestureRecognitionResult对象
                        self.detection_stats['gesture_detections'] += gesture_results.hands_detected
                    elif isinstance(gesture_results, (list, tuple)):
                        # 列表或元组
                        self.detection_stats['gesture_detections'] += len(gesture_results)
                    else:
                        # 其他类型，尝试转换为整数
                        try:
                            self.detection_stats['gesture_detections'] += int(gesture_results)
                        except (ValueError, TypeError):
                            self.detection_stats['gesture_detections'] += 1
                
                if hasattr(detection_result, 'pose_results') and detection_result.pose_results:
                    self.detection_stats['pose_detections'] += len(detection_result.pose_results)
                
                if hasattr(detection_result, 'fall_results') and detection_result.fall_results:
                    self.detection_stats['fall_detections'] += len(detection_result.fall_results)
                    
        except Exception as e:
            logger.warning(f"更新检测统计失败: {e}")
    
    def calculate_fps(self):
        """计算FPS"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:  # 每秒更新一次FPS
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def save_screenshot(self, frame):
        """保存截图"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = self.results_dir / f"screenshot_{timestamp}.jpg"
            cv2.imwrite(str(screenshot_path), frame)
            logger.info(f"📸 截图已保存: {screenshot_path}")
        except Exception as e:
            logger.error(f"保存截图失败: {e}")
    
    def reset_stats(self):
        """重置统计信息"""
        self.detection_stats = {
            'total_detections': 0,
            'face_detections': 0,
            'gesture_detections': 0,
            'pose_detections': 0,
            'fall_detections': 0
        }
        self.frame_count = 0
        logger.info("📊 统计信息已重置")
    
    def run_gui_test(self):
        """运行GUI测试"""
        logger.info("🖥️ 开始GUI版多模态识别测试...")
        
        # 初始化检测器
        if not self.initialize_detector():
            logger.error("❌ 检测器初始化失败，无法继续测试")
            return False
        
        # 初始化摄像头
        if not self.initialize_camera():
            logger.error("❌ 摄像头初始化失败，无法继续测试")
            return False
        
        # 创建窗口
        window_name = "YOLOS 多模态识别系统 - GUI测试"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        logger.info("🎬 开始实时检测，按 'q' 退出...")
        
        self.running = True
        paused = False
        
        try:
            while self.running:
                if not paused:
                    # 读取摄像头帧
                    ret, frame = self.cap.read()
                    
                    if not ret or frame is None:
                        logger.warning("⚠️ 无法读取摄像头帧")
                        continue
                    
                    self.frame_count += 1
                    
                    # 执行多模态检测
                    start_time = time.time()
                    try:
                        annotated_frame, detection_result = self.detector.detect(frame)
                        processing_time = time.time() - start_time
                        
                        # 更新统计信息
                        self.update_detection_stats(detection_result)
                        
                        # 使用检测结果帧或原始帧
                        display_frame = annotated_frame if annotated_frame is not None else frame
                        
                    except Exception as detection_error:
                        logger.warning(f"检测处理失败: {detection_error}")
                        display_frame = frame
                        processing_time = 0
                    
                    # 绘制信息覆盖层
                    self.draw_info_overlay(display_frame)
                    
                    # 计算FPS
                    self.calculate_fps()
                    
                    # 显示帧
                    cv2.imshow(window_name, display_frame)
                
                # 处理键盘输入
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' 或 ESC 退出
                    logger.info("用户请求退出")
                    break
                elif key == ord('s'):  # 's' 保存截图
                    if 'display_frame' in locals():
                        self.save_screenshot(display_frame)
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
            self.generate_test_report()
            
            logger.info("✅ GUI测试完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ GUI测试过程中发生错误: {e}")
            return False
        
        finally:
            # 清理资源
            self.cleanup()
    
    def generate_test_report(self):
        """生成测试报告"""
        try:
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'test_type': 'gui_multimodal_test',
                'total_frames': self.frame_count,
                'final_fps': self.current_fps,
                'detection_statistics': self.detection_stats.copy(),
                'system_info': {
                    'python_version': sys.version,
                    'opencv_version': cv2.__version__,
                    'platform': sys.platform,
                    'optimized_detector_available': OPTIMIZED_DETECTOR_AVAILABLE,
                    'fallback_detector_available': FALLBACK_DETECTOR_AVAILABLE
                }
            }
            
            # 保存JSON报告
            report_path = self.results_dir / 'gui_test_report.json'
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"📄 GUI测试报告已保存: {report_path}")
            
            # 打印摘要
            logger.info("\n" + "="*60)
            logger.info("📊 GUI测试摘要")
            logger.info("="*60)
            logger.info(f"总帧数: {self.frame_count}")
            logger.info(f"平均FPS: {self.current_fps:.1f}")
            logger.info(f"总检测次数: {self.detection_stats['total_detections']}")
            logger.info(f"面部检测: {self.detection_stats['face_detections']}")
            logger.info(f"手势检测: {self.detection_stats['gesture_detections']}")
            logger.info(f"姿势检测: {self.detection_stats['pose_detections']}")
            logger.info(f"摔倒检测: {self.detection_stats['fall_detections']}")
            
        except Exception as e:
            logger.error(f"生成测试报告失败: {e}")
    
    def cleanup(self):
        """清理资源"""
        logger.info("🧹 清理资源...")
        
        self.running = False
        
        # 释放摄像头
        if self.cap:
            self.cap.release()
        
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()
        
        # 清理检测器
        if self.detector and hasattr(self.detector, 'cleanup'):
            try:
                self.detector.cleanup()
            except Exception as cleanup_error:
                logger.warning(f"清理检测器时出现警告: {cleanup_error}")
        
        logger.info("✅ 资源清理完成")


def main():
    """主函数"""
    print("🖥️ GUI版多模态识别系统测试")
    print("=" * 50)
    print("功能说明:")
    print("- 实时显示摄像头画面和检测结果")
    print("- 显示FPS、帧数和检测统计")
    print("- 支持截图保存和统计重置")
    print("- 按 'q' 或 ESC 退出")
    print("- 按 's' 保存截图")
    print("- 按 'r' 重置统计")
    print("- 按空格键暂停/继续")
    print("=" * 50)
    
    try:
        # 创建测试器
        tester = GUIMultimodalTester()
        
        # 运行GUI测试
        success = tester.run_gui_test()
        
        if success:
            print("\n🎉 GUI测试完成！")
            sys.exit(0)
        else:
            print("\n❌ GUI测试失败！")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ 测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 测试过程发生异常: {e}")
        logger.error(f"主程序异常: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()