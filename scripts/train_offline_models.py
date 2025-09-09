#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线模型训练脚本
为所有识别场景训练离线模型
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.offline_training_manager import OfflineTrainingManager
from src.core.unified_config_manager import get_config_manager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('train_offline_models.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='训练离线识别模型')
    parser.add_argument('--scene', help='指定训练场景（不指定则训练所有场景）')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮次')
    parser.add_argument('--samples', type=int, default=1000, help='合成数据集样本数')
    parser.add_argument('--models-dir', default='./models', help='模型保存目录')
    
    args = parser.parse_args()
    
    try:
        # 初始化管理器
        config_manager = get_config_manager()
        offline_manager = OfflineTrainingManager(args.models_dir)
        
        # 获取要训练的场景
        if args.scene:
            scenes = [args.scene] if args.scene in config_manager.get_all_scenes() else []
            if not scenes:
                logger.error(f"未知场景: {args.scene}")
                sys.exit(1)
        else:
            scenes = config_manager.get_all_scenes()
        
        logger.info(f"开始训练 {len(scenes)} 个场景的离线模型")
        
        success_count = 0
        
        for scene in scenes:
            logger.info(f"{'='*50}")
            logger.info(f"训练场景: {scene}")
            logger.info(f"{'='*50}")
            
            try:
                # 创建数据集
                logger.info(f"创建合成数据集: {args.samples} 样本")
                dataset_success = offline_manager.create_offline_dataset(scene, args.samples)
                
                if not dataset_success:
                    logger.error(f"数据集创建失败: {scene}")
                    continue
                
                # 训练模型
                logger.info(f"开始训练模型: {args.epochs} 轮次")
                training_success = offline_manager.train_offline_model(scene, args.epochs)
                
                if training_success:
                    logger.info(f"✓ 场景 {scene} 训练成功")
                    success_count += 1
                    
                    # 更新配置
                    scene_config = config_manager.get_scene_config(scene)
                    if scene_config:
                        scene_config.offline_ready = True
                        config_manager.update_scene_config(scene, scene_config)
                        logger.info(f"✓ 场景 {scene} 配置已更新")
                else:
                    logger.error(f"✗ 场景 {scene} 训练失败")
                    
            except Exception as e:
                logger.error(f"训练场景 {scene} 时发生异常: {e}")
        
        # 生成训练报告
        logger.info(f"{'='*50}")
        logger.info("训练完成总结")
        logger.info(f"{'='*50}")
        logger.info(f"总场景数: {len(scenes)}")
        logger.info(f"成功训练: {success_count}")
        logger.info(f"失败场景: {len(scenes) - success_count}")
        logger.info(f"成功率: {success_count / len(scenes) * 100:.1f}%")
        
        # 获取系统状态
        status = offline_manager.get_offline_status()
        logger.info(f"系统离线就绪: {status['offline_ready']}")
        
        if status['offline_ready']:
            logger.info("🎉 所有场景训练完成，系统已离线就绪！")
        else:
            logger.warning("⚠️ 部分场景训练失败，系统未完全离线就绪")
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()