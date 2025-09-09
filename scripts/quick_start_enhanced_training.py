#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强训练系统快速启动脚本
一键完成数据准备、模型训练和测试
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.download_training_datasets import DatasetDownloader
from src.models.pretrained_model_loader import PretrainedModelLoader
from src.training.enhanced_human_trainer import EnhancedHumanTrainer, TrainingConfig
from src.recognition.improved_multimodal_detector import create_improved_multimodal_system

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuickStartManager:
    """快速启动管理器"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.datasets_dir = self.base_dir / "datasets"
        self.models_dir = self.base_dir / "models"
        
        # 创建必要目录
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        (self.models_dir / "pretrained").mkdir(parents=True, exist_ok=True)
        (self.models_dir / "human_recognition").mkdir(parents=True, exist_ok=True)
    
    def step1_prepare_environment(self):
        """步骤1: 准备环境"""
        logger.info("=== 步骤1: 准备环境 ===")
        
        # 检查依赖
        required_packages = [
            'torch', 'torchvision', 'opencv-python', 'numpy', 
            'scikit-learn', 'albumentations', 'tqdm', 'requests'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                logger.info(f"✓ {package} 已安装")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"✗ {package} 未安装")
        
        if missing_packages:
            logger.error(f"请安装缺失的包: pip install {' '.join(missing_packages)}")
            return False
        
        logger.info("环境检查完成")
        return True
    
    def step2_download_datasets(self, use_synthetic: bool = True):
        """步骤2: 下载数据集"""
        logger.info("=== 步骤2: 下载训练数据集 ===")
        
        downloader = DatasetDownloader(str(self.datasets_dir))
        
        if use_synthetic:
            # 创建合成数据集用于快速测试
            logger.info("创建合成数据集...")
            success = downloader.create_synthetic_dataset(1000)
            if success:
                logger.info("✓ 合成数据集创建成功")
            else:
                logger.error("✗ 合成数据集创建失败")
                return False
        else:
            # 下载真实数据集
            logger.info("下载真实数据集...")
            datasets_to_download = ['stanford40', 'coco_pose']
            
            for dataset_name in datasets_to_download:
                logger.info(f"下载 {dataset_name}...")
                success = downloader.download_dataset(dataset_name)
                if success:
                    logger.info(f"✓ {dataset_name} 下载成功")
                else:
                    logger.warning(f"✗ {dataset_name} 下载失败")
        
        return True
    
    def step3_download_pretrained_models(self):
        """步骤3: 下载预训练模型"""
        logger.info("=== 步骤3: 下载预训练模型 ===")
        
        model_loader = PretrainedModelLoader(str(self.models_dir / "pretrained"))
        
        # 下载所有预训练模型
        model_loader.download_all_models()
        
        # 验证下载
        model_loader.list_available_models()
        
        logger.info("预训练模型下载完成")
        return True
    
    def step4_train_model(self, 
                         dataset_name: str = "synthetic_human_actions",
                         epochs: int = 20,
                         batch_size: int = 8):
        """步骤4: 训练模型"""
        logger.info("=== 步骤4: 训练自定义模型 ===")
        
        # 训练配置
        config = TrainingConfig(
            batch_size=batch_size,
            learning_rate=0.001,
            epochs=epochs,
            validation_split=0.2,
            early_stopping_patience=10
        )
        
        # 创建训练器
        trainer = EnhancedHumanTrainer(config)
        
        # 数据集配置
        dataset_configs = [
            {
                'name': dataset_name,
                'path': str(self.datasets_dir / dataset_name),
                'format': 'custom',
                'description': f'{dataset_name} 数据集'
            }
        ]
        
        try:
            # 开始训练
            logger.info(f"开始训练模型，数据集: {dataset_name}")
            model_path = trainer.train(
                dataset_configs, 
                output_dir=str(self.models_dir / "human_recognition")
            )
            
            logger.info(f"✓ 模型训练完成: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"✗ 模型训练失败: {e}")
            return False
    
    def step5_test_system(self):
        """步骤5: 测试系统"""
        logger.info("=== 步骤5: 测试增强识别系统 ===")
        
        try:
            # 创建改进的识别系统
            detector = create_improved_multimodal_system({
                'face_database_path': str(self.base_dir / 'data' / 'face_database.pkl'),
                'use_pretrained_models': True,
                'detection_interval': 1
            })
            
            # 创建测试图像
            import numpy as np
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # 执行检测
            logger.info("执行多模态检测测试...")
            result = detector.detect_multimodal(test_frame)
            
            # 显示结果
            logger.info("检测结果:")
            logger.info(f"  面部检测: {len(result.faces)} 个")
            logger.info(f"  手势检测: {len(result.gestures)} 个")
            logger.info(f"  姿势检测: {len(result.poses)} 个")
            logger.info(f"  动作识别: {len(result.actions)} 个")
            logger.info(f"  摔倒检测: {len(result.falls)} 个")
            
            # 显示置信度
            if result.confidence_scores:
                logger.info("置信度分数:")
                for detection_type, confidence in result.confidence_scores.items():
                    logger.info(f"  {detection_type}: {confidence:.3f}")
            
            # 性能报告
            report = detector.get_performance_report()
            logger.info(f"处理性能: {report['processing_performance']['fps']:.1f} FPS")
            
            logger.info("✓ 系统测试完成")
            return True
            
        except Exception as e:
            logger.error(f"✗ 系统测试失败: {e}")
            return False
    
    def run_full_pipeline(self, 
                         use_synthetic: bool = True,
                         epochs: int = 20,
                         batch_size: int = 8):
        """运行完整流程"""
        logger.info("开始增强训练系统完整流程")
        
        steps = [
            ("准备环境", lambda: self.step1_prepare_environment()),
            ("下载数据集", lambda: self.step2_download_datasets(use_synthetic)),
            ("下载预训练模型", lambda: self.step3_download_pretrained_models()),
            ("训练模型", lambda: self.step4_train_model(
                "synthetic_human_actions" if use_synthetic else "stanford40",
                epochs, batch_size
            )),
            ("测试系统", lambda: self.step5_test_system())
        ]
        
        for step_name, step_func in steps:
            logger.info(f"\n{'='*50}")
            logger.info(f"执行: {step_name}")
            logger.info(f"{'='*50}")
            
            try:
                success = step_func()
                if not success:
                    logger.error(f"步骤失败: {step_name}")
                    return False
            except Exception as e:
                logger.error(f"步骤异常: {step_name} - {e}")
                return False
        
        logger.info("\n" + "="*50)
        logger.info("🎉 增强训练系统部署完成!")
        logger.info("="*50)
        
        # 提供使用示例
        self.show_usage_examples()
        
        return True
    
    def show_usage_examples(self):
        """显示使用示例"""
        logger.info("\n使用示例:")
        logger.info("1. 实时摄像头识别:")
        logger.info("   python -c \"")
        logger.info("   from src.recognition.improved_multimodal_detector import create_improved_multimodal_system")
        logger.info("   import cv2")
        logger.info("   detector = create_improved_multimodal_system()")
        logger.info("   cap = cv2.VideoCapture(0)")
        logger.info("   while True:")
        logger.info("       ret, frame = cap.read()")
        logger.info("       if ret:")
        logger.info("           result = detector.detect_multimodal(frame)")
        logger.info("           print(f'动作: {len(result.actions)}')")
        logger.info("       if cv2.waitKey(1) & 0xFF == ord('q'): break")
        logger.info("   \"")
        
        logger.info("\n2. 批量图像处理:")
        logger.info("   python scripts/batch_recognition.py --input-dir ./test_images")
        
        logger.info("\n3. 继续训练模型:")
        logger.info("   python scripts/continue_training.py --model-path ./models/human_recognition/best_model.pth")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='增强训练系统快速启动')
    parser.add_argument('--step', type=int, choices=[1,2,3,4,5], 
                       help='执行特定步骤 (1-5)')
    parser.add_argument('--full', action='store_true', 
                       help='执行完整流程')
    parser.add_argument('--synthetic', action='store_true', default=True,
                       help='使用合成数据集 (默认)')
    parser.add_argument('--real-data', action='store_true',
                       help='使用真实数据集')
    parser.add_argument('--epochs', type=int, default=20,
                       help='训练轮数 (默认: 20)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='批次大小 (默认: 8)')
    parser.add_argument('--base-dir', type=str, default='.',
                       help='基础目录 (默认: 当前目录)')
    
    args = parser.parse_args()
    
    # 创建管理器
    manager = QuickStartManager(args.base_dir)
    
    use_synthetic = args.synthetic and not args.real_data
    
    if args.full:
        # 执行完整流程
        success = manager.run_full_pipeline(
            use_synthetic=use_synthetic,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        sys.exit(0 if success else 1)
    
    elif args.step:
        # 执行特定步骤
        step_functions = {
            1: manager.step1_prepare_environment,
            2: lambda: manager.step2_download_datasets(use_synthetic),
            3: manager.step3_download_pretrained_models,
            4: lambda: manager.step4_train_model(
                "synthetic_human_actions" if use_synthetic else "stanford40",
                args.epochs, args.batch_size
            ),
            5: manager.step5_test_system
        }
        
        success = step_functions[args.step]()
        sys.exit(0 if success else 1)
    
    else:
        # 显示帮助
        parser.print_help()
        print("\n示例用法:")
        print("  完整流程:     python scripts/quick_start_enhanced_training.py --full")
        print("  使用真实数据: python scripts/quick_start_enhanced_training.py --full --real-data")
        print("  执行特定步骤: python scripts/quick_start_enhanced_training.py --step 4")
        print("  自定义训练:   python scripts/quick_start_enhanced_training.py --step 4 --epochs 50 --batch-size 16")

if __name__ == "__main__":
    main()