#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大模型自学习系统测试脚本
验证YOLOS自学习功能的完整性和正确性
"""

import os
import sys
import cv2
import numpy as np
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SelfLearningSystemTester:
    """大模型自学习系统测试器"""
    
    def __init__(self):
        self.test_results = []
        self.test_images = []
        
        # 创建测试图像
        self.create_test_images()
        
    def create_test_images(self):
        """创建测试图像"""
        logger.info("创建测试图像...")
        
        # 创建不同类型的测试图像
        test_scenarios = [
            ("unknown_object", self.create_unknown_object_image),
            ("medical_scene", self.create_medical_scene_image),
            ("fall_detection", self.create_fall_detection_image),
            ("medication", self.create_medication_image),
            ("low_quality", self.create_low_quality_image)
        ]
        
        for name, creator_func in test_scenarios:
            try:
                image = creator_func()
                self.test_images.append((name, image))
                logger.info(f"创建测试图像: {name}")
            except Exception as e:
                logger.error(f"创建测试图像失败 {name}: {e}")
    
    def create_unknown_object_image(self) -> np.ndarray:
        """创建未知对象图像"""
        # 创建一个复杂的几何图形，模拟未知对象
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 添加背景噪声
        noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)
        image = cv2.add(image, noise)
        
        # 绘制复杂图形
        center = (320, 240)
        
        # 绘制多个重叠的圆形和多边形
        cv2.circle(image, center, 100, (100, 150, 200), -1)
        cv2.circle(image, (center[0]-30, center[1]-30), 60, (200, 100, 150), -1)
        
        # 绘制不规则多边形
        points = np.array([
            [center[0]-80, center[1]+50],
            [center[0]-20, center[1]+80],
            [center[0]+40, center[1]+60],
            [center[0]+60, center[1]+20],
            [center[0]+20, center[1]-40]
        ], np.int32)
        cv2.fillPoly(image, [points], (150, 200, 100))
        
        # 添加纹理
        for i in range(0, 640, 20):
            cv2.line(image, (i, 0), (i, 480), (50, 50, 50), 1)
        for i in range(0, 480, 20):
            cv2.line(image, (0, i), (640, i), (50, 50, 50), 1)
        
        return image
    
    def create_medical_scene_image(self) -> np.ndarray:
        """创建医疗场景图像"""
        # 创建医疗设备场景
        image = np.ones((480, 640, 3), dtype=np.uint8) * 240  # 浅灰色背景
        
        # 绘制医疗设备轮廓
        # 心电监护仪
        cv2.rectangle(image, (50, 50), (250, 200), (80, 80, 80), -1)
        cv2.rectangle(image, (60, 60), (240, 120), (0, 0, 0), -1)  # 屏幕
        
        # 绘制心电图波形
        points = []
        for x in range(60, 240, 2):
            y = 90 + int(20 * np.sin((x-60) * 0.1)) + int(10 * np.sin((x-60) * 0.3))
            points.append((x, y))
        
        for i in range(len(points)-1):
            cv2.line(image, points[i], points[i+1], (0, 255, 0), 2)
        
        # 输液袋
        cv2.ellipse(image, (400, 150), (60, 80), 0, 0, 360, (200, 200, 255), -1)
        cv2.line(image, (400, 230), (400, 350), (100, 100, 100), 3)
        
        # 药瓶
        cv2.rectangle(image, (500, 300), (580, 400), (255, 255, 255), -1)
        cv2.rectangle(image, (500, 300), (580, 320), (255, 100, 100), -1)  # 标签
        
        # 添加文字标识
        cv2.putText(image, "MEDICAL", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return image
    
    def create_fall_detection_image(self) -> np.ndarray:
        """创建跌倒检测图像"""
        # 创建人物跌倒场景
        image = np.ones((480, 640, 3), dtype=np.uint8) * 200  # 地面色
        
        # 绘制地面
        cv2.rectangle(image, (0, 400), (640, 480), (150, 150, 150), -1)
        
        # 绘制跌倒的人物轮廓（简化的人形）
        # 头部
        cv2.circle(image, (320, 350), 30, (200, 180, 160), -1)
        
        # 身体（水平躺着）
        cv2.ellipse(image, (320, 380), (80, 25), 0, 0, 360, (100, 100, 200), -1)
        
        # 手臂
        cv2.ellipse(image, (280, 370), (30, 10), 45, 0, 360, (200, 180, 160), -1)
        cv2.ellipse(image, (360, 390), (30, 10), -30, 0, 360, (200, 180, 160), -1)
        
        # 腿部
        cv2.ellipse(image, (290, 400), (40, 15), 20, 0, 360, (100, 100, 200), -1)
        cv2.ellipse(image, (350, 405), (40, 15), -10, 0, 360, (100, 100, 200), -1)
        
        # 添加阴影效果
        cv2.ellipse(image, (320, 420), (100, 20), 0, 0, 360, (120, 120, 120), -1)
        
        # 添加警告标识
        cv2.putText(image, "FALL DETECTED", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return image
    
    def create_medication_image(self) -> np.ndarray:
        """创建药物图像"""
        # 创建药物识别场景
        image = np.ones((480, 640, 3), dtype=np.uint8) * 250  # 白色背景
        
        # 绘制药盒
        cv2.rectangle(image, (200, 150), (440, 300), (100, 150, 200), -1)
        cv2.rectangle(image, (210, 160), (430, 200), (255, 255, 255), -1)  # 标签区域
        
        # 绘制药片
        for i in range(3):
            for j in range(4):
                x = 220 + j * 50
                y = 220 + i * 25
                cv2.circle(image, (x, y), 8, (255, 255, 255), -1)
                cv2.circle(image, (x, y), 8, (100, 100, 100), 1)
        
        # 添加药品信息文字
        cv2.putText(image, "MEDICATION", (220, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(image, "500mg", (220, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # 绘制条形码
        for i in range(20):
            x = 350 + i * 4
            if i % 3 == 0:
                cv2.line(image, (x, 250), (x, 280), (0, 0, 0), 2)
            else:
                cv2.line(image, (x, 250), (x, 280), (0, 0, 0), 1)
        
        return image
    
    def create_low_quality_image(self) -> np.ndarray:
        """创建低质量图像"""
        # 创建模糊、噪声较多的图像
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 添加一些基本形状
        cv2.rectangle(image, (100, 100), (300, 300), (128, 128, 128), -1)
        cv2.circle(image, (200, 200), 50, (200, 200, 200), -1)
        
        # 添加模糊效果
        image = cv2.GaussianBlur(image, (15, 15), 0)
        
        # 添加噪声
        noise = np.random.randint(-50, 50, image.shape, dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def test_llm_self_learning_system(self):
        """测试大模型自学习系统"""
        logger.info("开始测试大模型自学习系统...")
        
        try:
            from recognition.llm_self_learning_system import LLMSelfLearningSystem
            
            # 创建系统实例
            config = {
                'llm': {'enabled': True},
                'self_learning': {'enabled': True, 'auto_trigger': True},
                'api_keys': {
                    'openai': 'test_key',  # 测试用密钥
                    'claude': 'test_key',
                    'qwen': 'test_key'
                }
            }
            
            llm_system = LLMSelfLearningSystem(config)
            
            # 测试基本功能
            test_image = self.test_images[0][1]  # 使用第一个测试图像
            
            # 测试触发条件判断
            should_learn = llm_system.should_trigger_self_learning(
                prediction_confidence=0.2,
                prediction_result="unknown_object"
            )
            
            assert should_learn, "应该触发自学习但没有触发"
            
            # 测试知识库查询
            results = llm_system.query_knowledge_base("medical")
            assert isinstance(results, list), "知识库查询结果应该是列表"
            
            # 测试统计信息
            stats = llm_system.get_learning_statistics()
            assert isinstance(stats, dict), "统计信息应该是字典"
            
            self.test_results.append({
                'test_name': 'LLM Self Learning System',
                'status': 'PASS',
                'details': '基本功能测试通过'
            })
            
            logger.info("大模型自学习系统测试通过")
            
        except ImportError as e:
            self.test_results.append({
                'test_name': 'LLM Self Learning System',
                'status': 'SKIP',
                'details': f'模块导入失败: {e}'
            })
            logger.warning(f"跳过大模型自学习系统测试: {e}")
            
        except Exception as e:
            self.test_results.append({
                'test_name': 'LLM Self Learning System',
                'status': 'FAIL',
                'details': f'测试失败: {e}'
            })
            logger.error(f"大模型自学习系统测试失败: {e}")
    
    def test_integrated_recognition_system(self):
        """测试集成识别系统"""
        logger.info("开始测试集成识别系统...")
        
        try:
            from recognition.integrated_self_learning_recognition import (
                IntegratedSelfLearningRecognition, RecognitionMode
            )
            
            # 创建系统实例（使用最小配置）
            config = {
                'recognition': {'default_mode': 'offline_only'},
                'self_learning': {'enabled': False},  # 禁用以避免API调用
                'quality_control': {'enable_quality_enhancement': False, 'enable_anti_spoofing': False},
                'medical': {'enable_medical_analysis': False, 'enable_fall_detection': False}
            }
            
            recognition_system = IntegratedSelfLearningRecognition(config)
            
            # 测试不同识别模式
            test_image = self.test_images[0][1]
            
            # 测试离线模式
            result = recognition_system.recognize(
                test_image,
                context={'test': True},
                mode=RecognitionMode.OFFLINE_ONLY
            )
            
            assert hasattr(result, 'object_type'), "结果应该包含object_type属性"
            assert hasattr(result, 'confidence'), "结果应该包含confidence属性"
            assert hasattr(result, 'processing_time'), "结果应该包含processing_time属性"
            
            # 测试批量识别
            test_images = [img[1] for img in self.test_images[:2]]
            batch_results = recognition_system.batch_recognize(
                test_images,
                mode=RecognitionMode.OFFLINE_ONLY
            )
            
            assert len(batch_results) == len(test_images), "批量识别结果数量不匹配"
            
            # 测试统计信息
            stats = recognition_system.get_recognition_statistics()
            assert isinstance(stats, dict), "统计信息应该是字典"
            
            self.test_results.append({
                'test_name': 'Integrated Recognition System',
                'status': 'PASS',
                'details': '集成识别系统测试通过'
            })
            
            logger.info("集成识别系统测试通过")
            
        except ImportError as e:
            self.test_results.append({
                'test_name': 'Integrated Recognition System',
                'status': 'SKIP',
                'details': f'模块导入失败: {e}'
            })
            logger.warning(f"跳过集成识别系统测试: {e}")
            
        except Exception as e:
            self.test_results.append({
                'test_name': 'Integrated Recognition System',
                'status': 'FAIL',
                'details': f'测试失败: {e}'
            })
            logger.error(f"集成识别系统测试失败: {e}")
    
    def test_configuration_loading(self):
        """测试配置文件加载"""
        logger.info("开始测试配置文件加载...")
        
        try:
            import yaml
            
            config_path = Path("config/self_learning_config.yaml")
            
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                # 验证配置结构
                required_sections = ['system', 'llm_providers', 'self_learning', 'recognition_modes']
                for section in required_sections:
                    assert section in config, f"配置文件缺少必需的节: {section}"
                
                # 验证大模型配置
                assert 'openai_gpt4v' in config['llm_providers'], "缺少OpenAI配置"
                assert 'claude_vision' in config['llm_providers'], "缺少Claude配置"
                
                # 验证自学习配置
                assert 'enabled' in config['self_learning'], "缺少自学习启用配置"
                assert 'triggers' in config['self_learning'], "缺少自学习触发配置"
                
                self.test_results.append({
                    'test_name': 'Configuration Loading',
                    'status': 'PASS',
                    'details': '配置文件加载和验证通过'
                })
                
                logger.info("配置文件测试通过")
                
            else:
                self.test_results.append({
                    'test_name': 'Configuration Loading',
                    'status': 'SKIP',
                    'details': '配置文件不存在'
                })
                logger.warning("配置文件不存在，跳过测试")
                
        except Exception as e:
            self.test_results.append({
                'test_name': 'Configuration Loading',
                'status': 'FAIL',
                'details': f'配置文件测试失败: {e}'
            })
            logger.error(f"配置文件测试失败: {e}")
    
    def test_image_processing(self):
        """测试图像处理功能"""
        logger.info("开始测试图像处理功能...")
        
        try:
            # 测试基本图像操作
            test_image = self.test_images[0][1]
            
            # 测试图像尺寸调整
            resized = cv2.resize(test_image, (224, 224))
            assert resized.shape == (224, 224, 3), "图像尺寸调整失败"
            
            # 测试颜色空间转换
            gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            assert len(gray.shape) == 2, "灰度转换失败"
            
            # 测试图像增强
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            assert enhanced.shape == gray.shape, "图像增强失败"
            
            # 测试噪声添加
            noise = np.random.randint(-10, 10, test_image.shape, dtype=np.int16)
            noisy = np.clip(test_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            assert noisy.shape == test_image.shape, "噪声添加失败"
            
            self.test_results.append({
                'test_name': 'Image Processing',
                'status': 'PASS',
                'details': '图像处理功能测试通过'
            })
            
            logger.info("图像处理功能测试通过")
            
        except Exception as e:
            self.test_results.append({
                'test_name': 'Image Processing',
                'status': 'FAIL',
                'details': f'图像处理测试失败: {e}'
            })
            logger.error(f"图像处理测试失败: {e}")
    
    def test_data_storage(self):
        """测试数据存储功能"""
        logger.info("开始测试数据存储功能...")
        
        try:
            # 创建测试数据目录
            test_data_dir = Path("data/test_self_learning")
            test_data_dir.mkdir(parents=True, exist_ok=True)
            
            # 测试JSON文件读写
            test_data = {
                'test_key': 'test_value',
                'timestamp': time.time(),
                'test_list': [1, 2, 3],
                'test_dict': {'nested': 'value'}
            }
            
            json_file = test_data_dir / "test.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)
            
            # 读取并验证
            with open(json_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            
            assert loaded_data['test_key'] == test_data['test_key'], "JSON数据不匹配"
            
            # 测试图像文件保存
            test_image = self.test_images[0][1]
            image_file = test_data_dir / "test_image.jpg"
            cv2.imwrite(str(image_file), test_image)
            
            # 读取并验证
            loaded_image = cv2.imread(str(image_file))
            assert loaded_image is not None, "图像文件保存/读取失败"
            assert loaded_image.shape == test_image.shape, "图像尺寸不匹配"
            
            # 清理测试文件
            json_file.unlink()
            image_file.unlink()
            test_data_dir.rmdir()
            
            self.test_results.append({
                'test_name': 'Data Storage',
                'status': 'PASS',
                'details': '数据存储功能测试通过'
            })
            
            logger.info("数据存储功能测试通过")
            
        except Exception as e:
            self.test_results.append({
                'test_name': 'Data Storage',
                'status': 'FAIL',
                'details': f'数据存储测试失败: {e}'
            })
            logger.error(f"数据存储测试失败: {e}")
    
    def test_gui_components(self):
        """测试GUI组件"""
        logger.info("开始测试GUI组件...")
        
        try:
            # 测试GUI依赖
            import tkinter as tk
            from PIL import Image, ImageTk
            
            # 创建测试窗口
            root = tk.Tk()
            root.withdraw()  # 隐藏窗口
            
            # 测试图像转换
            test_image = self.test_images[0][1]
            image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            tk_image = ImageTk.PhotoImage(pil_image)
            
            assert tk_image.width() > 0, "图像转换失败"
            assert tk_image.height() > 0, "图像转换失败"
            
            # 测试GUI组件创建
            frame = tk.Frame(root)
            label = tk.Label(frame, text="测试标签")
            button = tk.Button(frame, text="测试按钮")
            
            # 清理
            root.destroy()
            
            self.test_results.append({
                'test_name': 'GUI Components',
                'status': 'PASS',
                'details': 'GUI组件测试通过'
            })
            
            logger.info("GUI组件测试通过")
            
        except ImportError as e:
            self.test_results.append({
                'test_name': 'GUI Components',
                'status': 'SKIP',
                'details': f'GUI依赖缺失: {e}'
            })
            logger.warning(f"跳过GUI组件测试: {e}")
            
        except Exception as e:
            self.test_results.append({
                'test_name': 'GUI Components',
                'status': 'FAIL',
                'details': f'GUI组件测试失败: {e}'
            })
            logger.error(f"GUI组件测试失败: {e}")
    
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("开始运行大模型自学习系统完整测试...")
        
        # 运行各项测试
        test_methods = [
            self.test_configuration_loading,
            self.test_image_processing,
            self.test_data_storage,
            self.test_gui_components,
            self.test_llm_self_learning_system,
            self.test_integrated_recognition_system
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                logger.error(f"测试方法 {test_method.__name__} 执行失败: {e}")
        
        # 生成测试报告
        self.generate_test_report()
    
    def generate_test_report(self):
        """生成测试报告"""
        logger.info("生成测试报告...")
        
        # 统计测试结果
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r['status'] == 'PASS'])
        failed_tests = len([r for r in self.test_results if r['status'] == 'FAIL'])
        skipped_tests = len([r for r in self.test_results if r['status'] == 'SKIP'])
        
        # 生成报告内容
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("YOLOS 大模型自学习系统测试报告")
        report_lines.append("=" * 60)
        report_lines.append(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"总测试数: {total_tests}")
        report_lines.append(f"通过: {passed_tests}")
        report_lines.append(f"失败: {failed_tests}")
        report_lines.append(f"跳过: {skipped_tests}")
        report_lines.append(f"成功率: {passed_tests/total_tests*100:.1f}%")
        report_lines.append("")
        
        # 详细测试结果
        report_lines.append("详细测试结果:")
        report_lines.append("-" * 60)
        
        for result in self.test_results:
            status_symbol = {
                'PASS': '✓',
                'FAIL': '✗',
                'SKIP': '○'
            }.get(result['status'], '?')
            
            report_lines.append(f"{status_symbol} {result['test_name']}: {result['status']}")
            report_lines.append(f"   详情: {result['details']}")
            report_lines.append("")
        
        # 系统信息
        report_lines.append("系统信息:")
        report_lines.append("-" * 60)
        report_lines.append(f"Python版本: {sys.version}")
        report_lines.append(f"OpenCV版本: {cv2.__version__}")
        report_lines.append(f"NumPy版本: {np.__version__}")
        
        try:
            import torch
            report_lines.append(f"PyTorch版本: {torch.__version__}")
        except ImportError:
            report_lines.append("PyTorch: 未安装")
        
        try:
            import yaml
            report_lines.append(f"PyYAML版本: {yaml.__version__}")
        except ImportError:
            report_lines.append("PyYAML: 未安装")
        
        report_lines.append("")
        
        # 建议和总结
        report_lines.append("测试总结:")
        report_lines.append("-" * 60)
        
        if failed_tests == 0:
            report_lines.append("🎉 所有测试都通过了！系统功能正常。")
        else:
            report_lines.append(f"⚠️  有 {failed_tests} 个测试失败，请检查相关功能。")
        
        if skipped_tests > 0:
            report_lines.append(f"ℹ️  有 {skipped_tests} 个测试被跳过，可能是由于依赖缺失。")
        
        report_lines.append("")
        report_lines.append("建议:")
        if failed_tests > 0:
            report_lines.append("- 检查失败的测试项目并修复相关问题")
        if skipped_tests > 0:
            report_lines.append("- 安装缺失的依赖以启用完整功能")
        report_lines.append("- 配置大模型API密钥以启用自学习功能")
        report_lines.append("- 定期运行测试以确保系统稳定性")
        
        report_lines.append("=" * 60)
        
        # 输出报告
        report_content = "\n".join(report_lines)
        print(report_content)
        
        # 保存报告到文件
        try:
            report_dir = Path("test_results")
            report_dir.mkdir(exist_ok=True)
            
            report_file = report_dir / f"self_learning_test_report_{int(time.time())}.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"测试报告已保存到: {report_file}")
            
        except Exception as e:
            logger.error(f"保存测试报告失败: {e}")
        
        return passed_tests == total_tests


def main():
    """主函数"""
    print("YOLOS 大模型自学习系统测试")
    print("=" * 50)
    
    # 创建测试器
    tester = SelfLearningSystemTester()
    
    # 运行测试
    success = tester.run_all_tests()
    
    # 返回结果
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()