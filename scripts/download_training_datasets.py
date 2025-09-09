#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练数据集下载器
自动下载和准备人体识别相关的公开数据集
"""

import os
import requests
import zipfile
import tarfile
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging
from tqdm import tqdm
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetDownloader:
    """数据集下载器"""
    
    def __init__(self, base_dir: str = "./datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 公开数据集配置
        self.datasets_config = {
            'stanford40': {
                'name': 'Stanford 40 Actions',
                'url': 'http://vision.stanford.edu/Datasets/Stanford40_JPEGImages.tar',
                'description': '40种人体动作数据集',
                'format': 'tar',
                'size': '2.8GB',
                'classes': [
                    'applauding', 'blowing_bubbles', 'brushing_teeth', 'cleaning_the_floor',
                    'climbing', 'cooking', 'cutting_trees', 'cutting_vegetables', 'drinking',
                    'feeding_a_horse', 'fishing', 'fixing_a_bike', 'fixing_a_car', 'gardening',
                    'holding_an_umbrella', 'jumping', 'looking_through_a_microscope',
                    'looking_through_a_telescope', 'playing_guitar', 'playing_violin',
                    'pouring_liquid', 'pushing_a_cart', 'reading', 'phoning', 'riding_a_bike',
                    'riding_a_horse', 'rowing_a_boat', 'running', 'shooting_an_arrow',
                    'smoking', 'taking_photos', 'texting_message', 'throwing_frisby',
                    'using_a_computer', 'walking_the_dog', 'washing_dishes', 'watching_TV',
                    'waving_hands', 'writing_on_a_board', 'writing_on_a_book'
                ]
            },
            'ucf101': {
                'name': 'UCF-101 Action Recognition',
                'url': 'https://www.crcv.ucf.edu/data/UCF101/UCF101.rar',
                'description': '101种人体动作视频数据集',
                'format': 'rar',
                'size': '6.5GB',
                'note': '需要手动下载'
            },
            'mpii_pose': {
                'name': 'MPII Human Pose Dataset',
                'url': 'http://human-pose.mpi-inf.mpg.de/contents/mpii_human_pose_v1.tar.gz',
                'description': '人体姿势关键点数据集',
                'format': 'tar.gz',
                'size': '12.9GB',
                'keypoints': 16
            },
            'coco_pose': {
                'name': 'COCO Person Keypoints',
                'train_url': 'http://images.cocodataset.org/zips/train2017.zip',
                'val_url': 'http://images.cocodataset.org/zips/val2017.zip',
                'annotations_url': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
                'description': 'COCO人体关键点数据集',
                'format': 'zip',
                'size': '25GB',
                'keypoints': 17
            },
            'jhmdb': {
                'name': 'J-HMDB Dataset',
                'url': 'http://jhmdb.is.tue.mpg.de/challenge/JHMDB/datasets/JHMDB_video.tar.gz',
                'description': '人体动作视频数据集',
                'format': 'tar.gz',
                'size': '2GB',
                'classes': 21
            }
        }
    
    def download_file(self, url: str, filepath: Path, chunk_size: int = 8192) -> bool:
        """下载文件"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                desc=filepath.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"下载完成: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"下载失败 {url}: {e}")
            return False
    
    def extract_archive(self, archive_path: Path, extract_to: Path) -> bool:
        """解压文件"""
        try:
            extract_to.mkdir(parents=True, exist_ok=True)
            
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.suffix == '.tar' or archive_path.name.endswith('.tar.gz'):
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)
            else:
                logger.error(f"不支持的压缩格式: {archive_path}")
                return False
            
            logger.info(f"解压完成: {archive_path} -> {extract_to}")
            return True
            
        except Exception as e:
            logger.error(f"解压失败 {archive_path}: {e}")
            return False
    
    def download_stanford40(self) -> bool:
        """下载Stanford 40 Actions数据集"""
        dataset_name = 'stanford40'
        config = self.datasets_config[dataset_name]
        
        dataset_dir = self.base_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # 下载
        archive_path = dataset_dir / 'Stanford40_JPEGImages.tar'
        if not archive_path.exists():
            logger.info(f"下载 {config['name']}...")
            if not self.download_file(config['url'], archive_path):
                return False
        
        # 解压
        extract_dir = dataset_dir / 'images'
        if not extract_dir.exists():
            logger.info("解压数据集...")
            if not self.extract_archive(archive_path, dataset_dir):
                return False
        
        # 创建标注文件
        self.create_stanford40_annotations(dataset_dir, config['classes'])
        
        return True
    
    def create_stanford40_annotations(self, dataset_dir: Path, class_names: List[str]):
        """为Stanford40创建标注文件"""
        annotations = []
        images_dir = dataset_dir / 'JPEGImages'
        
        if not images_dir.exists():
            logger.warning(f"图像目录不存在: {images_dir}")
            return
        
        class_to_id = {name: idx for idx, name in enumerate(class_names)}
        
        for image_file in images_dir.glob('*.jpg'):
            # 从文件名推断类别
            filename = image_file.stem
            action_class = None
            
            for class_name in class_names:
                if class_name in filename.lower():
                    action_class = class_name
                    break
            
            if action_class:
                annotation = {
                    'image_path': f'JPEGImages/{image_file.name}',
                    'action_label': class_to_id[action_class],
                    'action_name': action_class,
                    'pose_keypoints': [],  # 需要后续处理添加
                    'gesture_label': -1
                }
                annotations.append(annotation)
        
        # 保存标注
        annotations_file = dataset_dir / 'annotations.json'
        with open(annotations_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        logger.info(f"创建标注文件: {annotations_file}, 共 {len(annotations)} 个样本")
    
    def download_coco_pose(self) -> bool:
        """下载COCO姿势数据集"""
        dataset_name = 'coco_pose'
        config = self.datasets_config[dataset_name]
        
        dataset_dir = self.base_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # 下载训练图像
        train_zip = dataset_dir / 'train2017.zip'
        if not train_zip.exists():
            logger.info("下载COCO训练图像...")
            if not self.download_file(config['train_url'], train_zip):
                return False
        
        # 下载验证图像
        val_zip = dataset_dir / 'val2017.zip'
        if not val_zip.exists():
            logger.info("下载COCO验证图像...")
            if not self.download_file(config['val_url'], val_zip):
                return False
        
        # 下载标注
        ann_zip = dataset_dir / 'annotations_trainval2017.zip'
        if not ann_zip.exists():
            logger.info("下载COCO标注...")
            if not self.download_file(config['annotations_url'], ann_zip):
                return False
        
        # 解压
        for zip_file in [train_zip, val_zip, ann_zip]:
            extract_dir = dataset_dir
            if not (dataset_dir / zip_file.stem).exists():
                logger.info(f"解压 {zip_file.name}...")
                if not self.extract_archive(zip_file, extract_dir):
                    return False
        
        # 转换COCO标注格式
        self.convert_coco_annotations(dataset_dir)
        
        return True
    
    def convert_coco_annotations(self, dataset_dir: Path):
        """转换COCO标注格式"""
        try:
            # 加载COCO标注
            train_ann_file = dataset_dir / 'annotations' / 'person_keypoints_train2017.json'
            val_ann_file = dataset_dir / 'annotations' / 'person_keypoints_val2017.json'
            
            for ann_file, split in [(train_ann_file, 'train'), (val_ann_file, 'val')]:
                if not ann_file.exists():
                    continue
                
                with open(ann_file, 'r') as f:
                    coco_data = json.load(f)
                
                # 转换格式
                annotations = []
                image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
                
                for ann in coco_data['annotations']:
                    if ann['num_keypoints'] > 0:  # 只保留有关键点的标注
                        image_filename = image_id_to_filename.get(ann['image_id'])
                        if image_filename:
                            annotation = {
                                'image_path': f'{split}2017/{image_filename}',
                                'action_label': 0,  # 通用人体动作
                                'action_name': 'person',
                                'pose_keypoints': ann['keypoints'],
                                'gesture_label': -1,
                                'bbox': ann['bbox']
                            }
                            annotations.append(annotation)
                
                # 保存转换后的标注
                output_file = dataset_dir / f'{split}_annotations.json'
                with open(output_file, 'w') as f:
                    json.dump(annotations, f, indent=2)
                
                logger.info(f"转换COCO标注完成: {output_file}, 共 {len(annotations)} 个样本")
        
        except Exception as e:
            logger.error(f"转换COCO标注失败: {e}")
    
    def create_synthetic_dataset(self, num_samples: int = 1000) -> bool:
        """创建合成训练数据集（用于快速测试）"""
        dataset_name = 'synthetic_human_actions'
        dataset_dir = self.base_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        images_dir = dataset_dir / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # 动作类别
        action_classes = [
            'standing', 'walking', 'running', 'sitting', 'jumping',
            'waving', 'pointing', 'clapping', 'stretching', 'bending'
        ]
        
        annotations = []
        
        logger.info(f"生成 {num_samples} 个合成样本...")
        
        for i in tqdm(range(num_samples)):
            # 生成随机图像
            image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # 添加一些简单的形状模拟人体
            center_x, center_y = np.random.randint(50, 174, 2)
            
            # 绘制简单的人体轮廓
            cv2.circle(image, (center_x, center_y - 30), 15, (255, 255, 255), -1)  # 头部
            cv2.rectangle(image, (center_x - 10, center_y - 15), (center_x + 10, center_y + 30), (255, 255, 255), -1)  # 身体
            
            # 随机选择动作类别
            action_idx = np.random.randint(0, len(action_classes))
            action_name = action_classes[action_idx]
            
            # 生成随机关键点
            keypoints = []
            for j in range(17):  # COCO格式17个关键点
                x = center_x + np.random.randint(-20, 20)
                y = center_y + np.random.randint(-40, 40)
                visibility = np.random.choice([0, 1, 2])  # 0:不可见, 1:遮挡, 2:可见
                keypoints.extend([x, y, visibility])
            
            # 保存图像
            image_filename = f'synthetic_{i:06d}.jpg'
            image_path = images_dir / image_filename
            cv2.imwrite(str(image_path), image)
            
            # 创建标注
            annotation = {
                'image_path': f'images/{image_filename}',
                'action_label': action_idx,
                'action_name': action_name,
                'pose_keypoints': keypoints,
                'gesture_label': np.random.randint(-1, 5)  # -1表示无手势，0-4表示不同手势
            }
            annotations.append(annotation)
        
        # 保存标注文件
        annotations_file = dataset_dir / 'annotations.json'
        with open(annotations_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        # 保存数据集信息
        dataset_info = {
            'name': dataset_name,
            'description': '合成人体动作数据集，用于快速测试',
            'num_samples': num_samples,
            'num_classes': len(action_classes),
            'class_names': action_classes,
            'image_size': [224, 224],
            'keypoints_format': 'COCO',
            'created_by': 'DatasetDownloader'
        }
        
        info_file = dataset_dir / 'dataset_info.json'
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        logger.info(f"合成数据集创建完成: {dataset_dir}")
        return True
    
    def list_available_datasets(self):
        """列出可用的数据集"""
        print("\n=== 可用的训练数据集 ===")
        for name, config in self.datasets_config.items():
            print(f"\n{name}:")
            print(f"  名称: {config['name']}")
            print(f"  描述: {config['description']}")
            print(f"  大小: {config.get('size', '未知')}")
            if 'classes' in config:
                print(f"  类别数: {len(config['classes'])}")
            if 'keypoints' in config:
                print(f"  关键点数: {config['keypoints']}")
            if 'note' in config:
                print(f"  注意: {config['note']}")
    
    def download_dataset(self, dataset_name: str) -> bool:
        """下载指定数据集"""
        if dataset_name not in self.datasets_config:
            logger.error(f"未知数据集: {dataset_name}")
            return False
        
        if dataset_name == 'stanford40':
            return self.download_stanford40()
        elif dataset_name == 'coco_pose':
            return self.download_coco_pose()
        elif dataset_name == 'synthetic_human_actions':
            return self.create_synthetic_dataset()
        else:
            logger.warning(f"数据集 {dataset_name} 需要手动下载")
            config = self.datasets_config[dataset_name]
            print(f"请手动下载: {config.get('url', '无URL')}")
            return False
    
    def download_all_available(self):
        """下载所有可用的数据集"""
        available_datasets = ['stanford40', 'coco_pose', 'synthetic_human_actions']
        
        for dataset_name in available_datasets:
            logger.info(f"开始下载数据集: {dataset_name}")
            success = self.download_dataset(dataset_name)
            if success:
                logger.info(f"数据集 {dataset_name} 下载成功")
            else:
                logger.error(f"数据集 {dataset_name} 下载失败")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='训练数据集下载器')
    parser.add_argument('--dataset', type=str, help='指定要下载的数据集名称')
    parser.add_argument('--list', action='store_true', help='列出所有可用数据集')
    parser.add_argument('--all', action='store_true', help='下载所有可用数据集')
    parser.add_argument('--synthetic', action='store_true', help='创建合成数据集')
    parser.add_argument('--base-dir', type=str, default='./datasets', help='数据集保存目录')
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.base_dir)
    
    if args.list:
        downloader.list_available_datasets()
    elif args.all:
        downloader.download_all_available()
    elif args.synthetic:
        downloader.create_synthetic_dataset(1000)
    elif args.dataset:
        downloader.download_dataset(args.dataset)
    else:
        print("请指定操作，使用 --help 查看帮助")

if __name__ == "__main__":
    main()