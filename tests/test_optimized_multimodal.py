#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""优化版多模态识别系统测试脚本 - 验证所有算法并修复编码问题"""

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

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

# 设置编码
if sys.platform.startswith('win'):
    # Windows系统编码设置
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    
    # 设置控制台编码
    os.system('chcp 65001 > nul')
    
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
        logging.FileHandler('multimodal_test.log', encoding='utf-8')
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
    traceback.print_exc()

# 备用导入
try:
    from recognition.multimodal_detector import MultimodalDetector
    FALLBACK_DETECTOR_AVAILABLE = True
    logger.info("✅ 备用多模态检测器导入成功")
except ImportError as e:
    FALLBACK_DETECTOR_AVAILABLE = False
    logger.error(f"❌ 备用多模态检测器导入失败: {e}")


class MultimodalTester:
    """多模态识别系统测试器"""
    
    def __init__(self):
        self.detector = None
        self.test_results = {
            'encoding_test': False,
            'camera_test': False,
            'face_recognition_test': False,
            'gesture_recognition_test': False,
            'pose_recognition_test': False,
            'fall_detection_test': False,
            'performance_test': False,
            'integration_test': False
        }
        self.test_images = []
        self.test_videos = []
        
    def setup_test_environment(self):
        """设置测试环境"""
        logger.info("🔧 设置测试环境...")
        
        try:
            # 检查OpenCV版本和编码支持
            logger.info(f"OpenCV版本: {cv2.__version__}")
            logger.info(f"系统编码: {sys.getdefaultencoding()}")
            logger.info(f"文件系统编码: {sys.getfilesystemencoding()}")
            logger.info(f"Locale编码: {locale.getpreferredencoding()}")
            
            # 测试中文字符处理
            test_chinese = "测试中文字符: 面部识别、手势识别、姿势识别、摔倒检测"
            logger.info(test_chinese)
            
            # 创建测试目录
            test_dir = project_root / 'test_results'
            test_dir.mkdir(exist_ok=True)
            
            # 生成测试图像
            self._generate_test_images(test_dir)
            
            logger.info("✅ 测试环境设置完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 测试环境设置失败: {e}")
            traceback.print_exc()
            return False
    
    def _generate_test_images(self, test_dir: Path):
        """生成测试图像"""
        try:
            # 创建简单的测试图像
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # 绘制一些基本形状作为测试
            cv2.rectangle(test_image, (100, 100), (200, 200), (255, 0, 0), 2)
            cv2.circle(test_image, (400, 300), 50, (0, 255, 0), 2)
            
            # 添加中文文本测试
            try:
                # 使用默认字体
                cv2.putText(test_image, "Test Image", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            except Exception as e:
                logger.warning(f"文本绘制警告: {e}")
            
            # 保存测试图像
            test_image_path = test_dir / 'test_image.jpg'
            cv2.imwrite(str(test_image_path), test_image)
            self.test_images.append(str(test_image_path))
            
            logger.info(f"测试图像已生成: {test_image_path}")
            
        except Exception as e:
            logger.error(f"测试图像生成失败: {e}")
    
    def test_encoding(self):
        """测试编码处理"""
        logger.info("🔤 开始编码测试...")
        
        try:
            # 测试中文字符串处理
            chinese_texts = [
                "面部识别",
                "手势识别", 
                "身体姿势识别",
                "摔倒检测",
                "多模态识别系统",
                "实时视频处理"
            ]
            
            for text in chinese_texts:
                # 测试编码转换
                utf8_bytes = text.encode('utf-8')
                decoded_text = utf8_bytes.decode('utf-8')
                
                if text == decoded_text:
                    logger.info(f"✅ 编码测试通过: {text}")
                else:
                    logger.error(f"❌ 编码测试失败: {text} != {decoded_text}")
                    return False
            
            # 测试JSON序列化
            test_data = {
                "算法类型": ["面部识别", "手势识别", "姿势识别", "摔倒检测"],
                "状态": "正常运行",
                "时间戳": time.time()
            }
            
            json_str = json.dumps(test_data, ensure_ascii=False, indent=2)
            parsed_data = json.loads(json_str)
            
            if test_data == parsed_data:
                logger.info("✅ JSON编码测试通过")
            else:
                logger.error("❌ JSON编码测试失败")
                return False
            
            self.test_results['encoding_test'] = True
            logger.info("✅ 编码测试完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 编码测试失败: {e}")
            traceback.print_exc()
            return False
    
    def test_camera_access(self):
        """测试摄像头访问"""
        logger.info("📷 开始摄像头测试...")
        
        try:
            # 尝试打开摄像头
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                logger.warning("⚠️ 无法打开默认摄像头，尝试其他摄像头...")
                
                # 尝试其他摄像头ID
                for camera_id in range(1, 5):
                    cap = cv2.VideoCapture(camera_id)
                    if cap.isOpened():
                        logger.info(f"✅ 摄像头 {camera_id} 可用")
                        break
                else:
                    logger.error("❌ 没有可用的摄像头")
                    return False
            
            # 设置摄像头参数
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # 读取几帧测试
            for i in range(5):
                ret, frame = cap.read()
                if not ret:
                    logger.error(f"❌ 无法读取摄像头帧 {i+1}")
                    cap.release()
                    return False
                
                logger.info(f"✅ 成功读取摄像头帧 {i+1}: {frame.shape}")
                time.sleep(0.1)
            
            cap.release()
            
            self.test_results['camera_test'] = True
            logger.info("✅ 摄像头测试完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 摄像头测试失败: {e}")
            traceback.print_exc()
            return False
    
    def test_detector_initialization(self):
        """测试检测器初始化"""
        logger.info("🚀 开始检测器初始化测试...")
        
        try:
            if OPTIMIZED_DETECTOR_AVAILABLE:
                # 测试优化版检测器
                logger.info("测试优化版多模态检测器...")
                
                # 测试不同配置
                for config_name in MULTIMODAL_DETECTOR_CONFIGS.keys():
                    logger.info(f"测试配置: {config_name}")
                    
                    detector = create_multimodal_detector_from_config(
                        config_name=config_name,
                        encoding='utf-8'
                    )
                    
                    if detector:
                        logger.info(f"✅ {config_name} 配置初始化成功")
                        
                        # 获取系统状态
                        status = detector.get_system_status()
                        logger.info(f"活跃算法: {status.active_algorithms}")
                        
                        detector.cleanup()
                    else:
                        logger.error(f"❌ {config_name} 配置初始化失败")
                        return False
                
                # 使用balanced配置作为主检测器
                self.detector = create_multimodal_detector_from_config(
                    config_name='balanced',
                    encoding='utf-8'
                )
                
            elif FALLBACK_DETECTOR_AVAILABLE:
                # 使用备用检测器
                logger.info("使用备用多模态检测器...")
                self.detector = MultimodalDetector()
            
            else:
                logger.error("❌ 没有可用的检测器")
                return False
            
            if self.detector:
                logger.info("✅ 检测器初始化成功")
                return True
            else:
                logger.error("❌ 检测器初始化失败")
                return False
                
        except Exception as e:
            logger.error(f"❌ 检测器初始化测试失败: {e}")
            traceback.print_exc()
            return False
    
    def test_individual_algorithms(self):
        """测试各个算法"""
        logger.info("🧠 开始各个算法测试...")
        
        if not self.detector:
            logger.error("❌ 检测器未初始化")
            return False
        
        try:
            # 使用测试图像
            if self.test_images:
                test_image = cv2.imread(self.test_images[0])
            else:
                # 创建简单测试图像
                test_image = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.rectangle(test_image, (200, 150), (400, 350), (255, 255, 255), -1)
            
            logger.info(f"测试图像尺寸: {test_image.shape}")
            
            # 执行检测
            start_time = time.time()
            annotated_image, detection_result = self.detector.detect(test_image)
            processing_time = time.time() - start_time
            
            logger.info(f"检测处理时间: {processing_time:.3f}秒")
            
            # 检查结果
            if hasattr(detection_result, 'face_results'):
                logger.info(f"面部识别结果: {len(detection_result.face_results)} 项")
                self.test_results['face_recognition_test'] = True
            
            if hasattr(detection_result, 'gesture_results'):
                logger.info(f"手势识别结果: {len(detection_result.gesture_results)} 项")
                self.test_results['gesture_recognition_test'] = True
            
            if hasattr(detection_result, 'pose_results'):
                logger.info(f"姿势识别结果: {len(detection_result.pose_results)} 项")
                self.test_results['pose_recognition_test'] = True
            
            if hasattr(detection_result, 'fall_results'):
                logger.info(f"摔倒检测结果: {len(detection_result.fall_results)} 项")
                self.test_results['fall_detection_test'] = True
            
            # 保存结果图像
            if annotated_image is not None:
                result_path = project_root / 'test_results' / 'detection_result.jpg'
                cv2.imwrite(str(result_path), annotated_image)
                logger.info(f"检测结果已保存: {result_path}")
            
            logger.info("✅ 各个算法测试完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 各个算法测试失败: {e}")
            traceback.print_exc()
            return False
    
    def test_performance(self):
        """测试性能"""
        logger.info("⚡ 开始性能测试...")
        
        if not self.detector:
            logger.error("❌ 检测器未初始化")
            return False
        
        try:
            # 创建测试图像
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # 性能测试参数
            num_iterations = 10
            processing_times = []
            
            logger.info(f"执行 {num_iterations} 次性能测试...")
            
            for i in range(num_iterations):
                start_time = time.time()
                
                # 执行检测
                annotated_image, detection_result = self.detector.detect(test_image)
                
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                logger.info(f"第 {i+1} 次测试: {processing_time:.3f}秒")
            
            # 计算统计信息
            avg_time = sum(processing_times) / len(processing_times)
            min_time = min(processing_times)
            max_time = max(processing_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            logger.info(f"性能统计:")
            logger.info(f"  平均处理时间: {avg_time:.3f}秒")
            logger.info(f"  最短处理时间: {min_time:.3f}秒")
            logger.info(f"  最长处理时间: {max_time:.3f}秒")
            logger.info(f"  平均FPS: {fps:.1f}")
            
            # 获取系统统计
            if hasattr(self.detector, 'get_statistics'):
                stats = self.detector.get_statistics()
                logger.info(f"系统统计: {json.dumps(stats, ensure_ascii=False, indent=2)}")
            
            self.test_results['performance_test'] = True
            logger.info("✅ 性能测试完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 性能测试失败: {e}")
            traceback.print_exc()
            return False
    
    def test_real_time_detection(self, duration: int = 10):
        """测试实时检测"""
        logger.info(f"🎥 开始 {duration} 秒实时检测测试...")
        
        if not self.detector:
            logger.error("❌ 检测器未初始化")
            return False
        
        try:
            # 尝试打开摄像头
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logger.warning("⚠️ 摄像头不可用，使用模拟数据测试")
                return self._test_simulated_real_time(duration)
            
            # 设置摄像头参数
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            start_time = time.time()
            frame_count = 0
            
            logger.info("开始实时检测，按 'q' 键提前退出...")
            
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    logger.error("❌ 无法读取摄像头帧")
                    break
                
                # 执行检测
                annotated_frame, detection_result = self.detector.detect(frame)
                
                # 显示结果
                cv2.imshow('实时多模态识别测试', annotated_frame)
                
                frame_count += 1
                
                # 检查退出条件
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("用户请求退出")
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
            # 计算统计
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            logger.info(f"实时检测统计:")
            logger.info(f"  总时间: {elapsed_time:.1f}秒")
            logger.info(f"  处理帧数: {frame_count}")
            logger.info(f"  平均FPS: {fps:.1f}")
            
            self.test_results['integration_test'] = True
            logger.info("✅ 实时检测测试完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 实时检测测试失败: {e}")
            traceback.print_exc()
            return False
    
    def _test_simulated_real_time(self, duration: int):
        """模拟实时检测测试"""
        logger.info("使用模拟数据进行实时检测测试...")
        
        try:
            start_time = time.time()
            frame_count = 0
            
            while time.time() - start_time < duration:
                # 生成随机测试图像
                test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                
                # 添加一些形状
                cv2.rectangle(test_frame, (100, 100), (200, 200), (255, 0, 0), 2)
                cv2.circle(test_frame, (400, 300), 50, (0, 255, 0), 2)
                
                # 执行检测
                annotated_frame, detection_result = self.detector.detect(test_frame)
                
                frame_count += 1
                
                # 模拟帧率
                time.sleep(0.033)  # ~30 FPS
            
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            logger.info(f"模拟实时检测统计:")
            logger.info(f"  总时间: {elapsed_time:.1f}秒")
            logger.info(f"  处理帧数: {frame_count}")
            logger.info(f"  平均FPS: {fps:.1f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 模拟实时检测测试失败: {e}")
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("🧪 开始完整测试套件...")
        
        test_sequence = [
            ("环境设置", self.setup_test_environment),
            ("编码测试", self.test_encoding),
            ("摄像头测试", self.test_camera_access),
            ("检测器初始化", self.test_detector_initialization),
            ("算法测试", self.test_individual_algorithms),
            ("性能测试", self.test_performance),
            ("实时检测测试", lambda: self.test_real_time_detection(5))
        ]
        
        passed_tests = 0
        total_tests = len(test_sequence)
        
        for test_name, test_func in test_sequence:
            logger.info(f"\n{'='*50}")
            logger.info(f"执行测试: {test_name}")
            logger.info(f"{'='*50}")
            
            try:
                if test_func():
                    logger.info(f"✅ {test_name} - 通过")
                    passed_tests += 1
                else:
                    logger.error(f"❌ {test_name} - 失败")
            except Exception as e:
                logger.error(f"❌ {test_name} - 异常: {e}")
                traceback.print_exc()
        
        # 生成测试报告
        self._generate_test_report(passed_tests, total_tests)
        
        # 清理资源
        if self.detector:
            self.detector.cleanup()
        
        return passed_tests == total_tests
    
    def _generate_test_report(self, passed_tests: int, total_tests: int):
        """生成测试报告"""
        logger.info(f"\n{'='*60}")
        logger.info("📊 测试报告")
        logger.info(f"{'='*60}")
        
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        logger.info(f"总测试数: {total_tests}")
        logger.info(f"通过测试: {passed_tests}")
        logger.info(f"失败测试: {total_tests - passed_tests}")
        logger.info(f"成功率: {success_rate:.1f}%")
        
        logger.info("\n详细结果:")
        for test_name, result in self.test_results.items():
            status = "✅ 通过" if result else "❌ 失败"
            logger.info(f"  {test_name}: {status}")
        
        # 保存报告到文件
        try:
            report_data = {
                'timestamp': time.time(),
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'success_rate': success_rate,
                'detailed_results': self.test_results,
                'system_info': {
                    'python_version': sys.version,
                    'opencv_version': cv2.__version__,
                    'platform': sys.platform,
                    'encoding': sys.getdefaultencoding()
                }
            }
            
            report_path = project_root / 'test_results' / 'test_report.json'
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"\n📄 测试报告已保存: {report_path}")
            
        except Exception as e:
            logger.error(f"保存测试报告失败: {e}")
        
        if success_rate >= 80:
            logger.info("\n🎉 测试整体通过！系统可以正常使用。")
        else:
            logger.warning("\n⚠️ 部分测试失败，请检查相关问题。")


def main():
    """主函数"""
    print("🚀 优化版多模态识别系统测试")
    print("=" * 50)
    
    try:
        # 创建测试器
        tester = MultimodalTester()
        
        # 运行所有测试
        success = tester.run_all_tests()
        
        if success:
            print("\n🎉 所有测试通过！")
            sys.exit(0)
        else:
            print("\n❌ 部分测试失败！")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ 测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 测试过程发生异常: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()