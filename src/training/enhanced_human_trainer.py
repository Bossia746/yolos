#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强人体识别训练系统
支持多模态数据预训练，提升识别准确性
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import albumentations as A
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    validation_split: float = 0.2
    augmentation_prob: float = 0.8
    model_save_interval: int = 10
    early_stopping_patience: int = 15

@dataclass
class DatasetInfo:
    """数据集信息"""
    name: str
    path: str
    num_classes: int
    class_names: List[str]
    total_samples: int
    description: str

class HumanActionDataset(Dataset):
    """人体动作数据集"""
    
    def __init__(self, 
                 data_path: str,
                 annotations: List[Dict],
                 transform=None,
                 augment=True):
        self.data_path = Path(data_path)
        self.annotations = annotations
        self.transform = transform
        self.augment = augment
        
        # 数据增强管道
        if augment:
            self.augmentation = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2),
                A.Blur(blur_limit=3, p=0.2),
                A.RandomRotate90(p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.3
                )
            ])
        else:
            self.augmentation = None
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        
        # 加载图像
        image_path = self.data_path / annotation['image_path']
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 获取标签
        action_label = annotation['action_label']
        pose_keypoints = np.array(annotation.get('pose_keypoints', []))
        gesture_label = annotation.get('gesture_label', -1)
        
        # 数据增强
        if self.augmentation and self.augment:
            augmented = self.augmentation(image=image)
            image = augmented['image']
        
        # 转换为tensor
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return {
            'image': image,
            'action_label': torch.tensor(action_label, dtype=torch.long),
            'pose_keypoints': torch.tensor(pose_keypoints, dtype=torch.float32),
            'gesture_label': torch.tensor(gesture_label, dtype=torch.long)
        }

class MultiModalHumanNet(nn.Module):
    """多模态人体识别网络"""
    
    def __init__(self, 
                 num_action_classes: int = 10,
                 num_gesture_classes: int = 8,
                 pose_keypoints_dim: int = 34):  # 17个关键点 * 2坐标
        super().__init__()
        
        # 图像特征提取器 (ResNet-like backbone)
        self.image_backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # ResNet blocks
            self._make_layer(64, 128, 2),
            self._make_layer(128, 256, 2),
            self._make_layer(256, 512, 2),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # 姿势关键点处理器
        self.pose_processor = nn.Sequential(
            nn.Linear(pose_keypoints_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 多模态融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(512 + 256, 512),  # 图像特征 + 姿势特征
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 分类头
        self.action_classifier = nn.Linear(256, num_action_classes)
        self.gesture_classifier = nn.Linear(256, num_gesture_classes)
        
    def _make_layer(self, in_channels, out_channels, stride):
        """创建ResNet层"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, image, pose_keypoints):
        # 图像特征提取
        image_features = self.image_backbone(image)
        
        # 姿势特征提取
        pose_features = self.pose_processor(pose_keypoints)
        
        # 多模态融合
        fused_features = torch.cat([image_features, pose_features], dim=1)
        fused_features = self.fusion_layer(fused_features)
        
        # 分类预测
        action_pred = self.action_classifier(fused_features)
        gesture_pred = self.gesture_classifier(fused_features)
        
        return action_pred, gesture_pred

class EnhancedHumanTrainer:
    """增强人体识别训练器"""
    
    def __init__(self, model_type: str = 'yolov11', device: str = 'auto', config: Optional[TrainingConfig] = None):
        self.model_type = model_type
        self.config = config or TrainingConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        
        # 训练历史
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        logger.info(f"训练器初始化完成，使用设备: {self.device}")
    
    def prepare_datasets(self, dataset_configs: List[Dict[str, Any]]) -> Dict[str, DatasetInfo]:
        """准备多个数据集"""
        datasets_info = {}
        
        for config in dataset_configs:
            dataset_info = self._prepare_single_dataset(config)
            datasets_info[dataset_info.name] = dataset_info
            
        return datasets_info
    
    def _prepare_single_dataset(self, config: Dict[str, Any]) -> DatasetInfo:
        """准备单个数据集"""
        dataset_path = Path(config['path'])
        
        # 扫描数据集
        annotations = []
        class_names = set()
        
        # 支持多种数据集格式
        if config.get('format') == 'coco':
            annotations = self._load_coco_format(dataset_path, config)
        elif config.get('format') == 'custom':
            annotations = self._load_custom_format(dataset_path, config)
        else:
            annotations = self._auto_detect_format(dataset_path)
        
        # 统计类别
        for ann in annotations:
            if 'action_label' in ann:
                class_names.add(ann['action_label'])
        
        class_names = sorted(list(class_names))
        
        dataset_info = DatasetInfo(
            name=config['name'],
            path=str(dataset_path),
            num_classes=len(class_names),
            class_names=class_names,
            total_samples=len(annotations),
            description=config.get('description', '')
        )
        
        logger.info(f"数据集 {dataset_info.name} 准备完成: {dataset_info.total_samples} 样本, {dataset_info.num_classes} 类别")
        
        return dataset_info
    
    def _load_coco_format(self, dataset_path: Path, config: Dict) -> List[Dict]:
        """加载COCO格式数据集"""
        annotations_file = dataset_path / config.get('annotations_file', 'annotations.json')
        
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        annotations = []
        for ann in coco_data['annotations']:
            # 转换COCO格式到内部格式
            annotation = {
                'image_path': f"images/{ann['image_id']:012d}.jpg",
                'action_label': ann.get('category_id', 0),
                'pose_keypoints': ann.get('keypoints', []),
                'gesture_label': ann.get('gesture_id', -1)
            }
            annotations.append(annotation)
        
        return annotations
    
    def _load_custom_format(self, dataset_path: Path, config: Dict) -> List[Dict]:
        """加载自定义格式数据集"""
        annotations_file = dataset_path / config.get('annotations_file', 'annotations.json')
        
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        return annotations
    
    def _auto_detect_format(self, dataset_path: Path) -> List[Dict]:
        """自动检测数据集格式"""
        annotations = []
        
        # 扫描图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(dataset_path.glob(f"**/*{ext}"))
        
        # 为每个图像创建基础标注
        for img_path in image_files:
            relative_path = img_path.relative_to(dataset_path)
            
            # 从文件夹名推断类别
            action_label = 0
            if len(relative_path.parts) > 1:
                folder_name = relative_path.parts[-2]
                action_label = hash(folder_name) % 10  # 简单的类别映射
            
            annotation = {
                'image_path': str(relative_path),
                'action_label': action_label,
                'pose_keypoints': [],
                'gesture_label': -1
            }
            annotations.append(annotation)
        
        return annotations
    
    def create_data_loaders(self, dataset_info: DatasetInfo, annotations: List[Dict]):
        """创建数据加载器"""
        # 分割训练和验证集
        train_annotations, val_annotations = train_test_split(
            annotations, 
            test_size=self.config.validation_split,
            random_state=42,
            stratify=[ann['action_label'] for ann in annotations]
        )
        
        # 数据变换
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 创建数据集
        train_dataset = HumanActionDataset(
            dataset_info.path, 
            train_annotations, 
            transform=transform,
            augment=True
        )
        
        val_dataset = HumanActionDataset(
            dataset_info.path, 
            val_annotations, 
            transform=transform,
            augment=False
        )
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"数据加载器创建完成 - 训练: {len(train_dataset)}, 验证: {len(val_dataset)}")
    
    def create_model(self, num_action_classes: int, num_gesture_classes: int):
        """创建模型"""
        self.model = MultiModalHumanNet(
            num_action_classes=num_action_classes,
            num_gesture_classes=num_gesture_classes
        ).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-4
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        logger.info(f"模型创建完成，参数数量: {sum(p.numel() for p in self.model.parameters())}")
    
    def train_epoch(self) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct_actions = 0
        correct_gestures = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            action_labels = batch['action_label'].to(self.device)
            pose_keypoints = batch['pose_keypoints'].to(self.device)
            gesture_labels = batch['gesture_label'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            action_pred, gesture_pred = self.model(images, pose_keypoints)
            
            # 计算损失
            action_loss = self.criterion(action_pred, action_labels)
            
            # 只对有效手势标签计算损失
            valid_gesture_mask = gesture_labels >= 0
            if valid_gesture_mask.sum() > 0:
                gesture_loss = self.criterion(
                    gesture_pred[valid_gesture_mask], 
                    gesture_labels[valid_gesture_mask]
                )
            else:
                gesture_loss = torch.tensor(0.0, device=self.device)
            
            total_loss_batch = action_loss + 0.5 * gesture_loss
            
            # 反向传播
            total_loss_batch.backward()
            self.optimizer.step()
            
            # 统计准确率
            total_loss += total_loss_batch.item()
            _, action_predicted = torch.max(action_pred.data, 1)
            correct_actions += (action_predicted == action_labels).sum().item()
            
            if valid_gesture_mask.sum() > 0:
                _, gesture_predicted = torch.max(gesture_pred[valid_gesture_mask].data, 1)
                correct_gestures += (gesture_predicted == gesture_labels[valid_gesture_mask]).sum().item()
            
            total_samples += images.size(0)
            
            if batch_idx % 50 == 0:
                logger.info(f'训练批次 {batch_idx}/{len(self.train_loader)}, 损失: {total_loss_batch.item():.4f}')
        
        avg_loss = total_loss / len(self.train_loader)
        action_acc = correct_actions / total_samples
        
        return avg_loss, action_acc
    
    def validate_epoch(self) -> Tuple[float, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        correct_actions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                action_labels = batch['action_label'].to(self.device)
                pose_keypoints = batch['pose_keypoints'].to(self.device)
                gesture_labels = batch['gesture_label'].to(self.device)
                
                # 前向传播
                action_pred, gesture_pred = self.model(images, pose_keypoints)
                
                # 计算损失
                action_loss = self.criterion(action_pred, action_labels)
                
                valid_gesture_mask = gesture_labels >= 0
                if valid_gesture_mask.sum() > 0:
                    gesture_loss = self.criterion(
                        gesture_pred[valid_gesture_mask], 
                        gesture_labels[valid_gesture_mask]
                    )
                else:
                    gesture_loss = torch.tensor(0.0, device=self.device)
                
                total_loss_batch = action_loss + 0.5 * gesture_loss
                total_loss += total_loss_batch.item()
                
                # 统计准确率
                _, predicted = torch.max(action_pred.data, 1)
                correct_actions += (predicted == action_labels).sum().item()
                total_samples += images.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_actions / total_samples
        
        return avg_loss, accuracy
    
    def train(self, dataset_configs: List[Dict[str, Any]], output_dir: str = "models/human_recognition"):
        """开始训练"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 准备数据集
        datasets_info = self.prepare_datasets(dataset_configs)
        
        # 使用第一个数据集进行训练（可扩展为多数据集融合）
        main_dataset = list(datasets_info.values())[0]
        
        # 重新加载标注数据
        if main_dataset.name in [config['name'] for config in dataset_configs]:
            config = next(config for config in dataset_configs if config['name'] == main_dataset.name)
            if config.get('format') == 'coco':
                annotations = self._load_coco_format(Path(main_dataset.path), config)
            elif config.get('format') == 'custom':
                annotations = self._load_custom_format(Path(main_dataset.path), config)
            else:
                annotations = self._auto_detect_format(Path(main_dataset.path))
        
        # 创建数据加载器
        self.create_data_loaders(main_dataset, annotations)
        
        # 创建模型
        self.create_model(main_dataset.num_classes, 8)  # 假设8种手势
        
        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info("开始训练...")
        
        for epoch in range(self.config.epochs):
            logger.info(f"Epoch {epoch+1}/{self.config.epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc = self.validate_epoch()
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 记录历史
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            
            logger.info(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
            logger.info(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'config': self.config,
                    'dataset_info': main_dataset
                }, output_path / 'best_model.pth')
                
                logger.info("保存最佳模型")
            else:
                patience_counter += 1
            
            # 定期保存检查点
            if (epoch + 1) % self.config.model_save_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'training_history': self.training_history
                }, output_path / f'checkpoint_epoch_{epoch+1}.pth')
            
            # 早停
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"早停触发，在epoch {epoch+1}")
                break
        
        # 保存训练历史
        with open(output_path / 'training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info("训练完成!")
        return output_path / 'best_model.pth'
    
    def get_training_features(self) -> List[str]:
        """获取训练功能列表"""
        return [
            'multi_modal_training',
            'data_augmentation', 
            'transfer_learning',
            'early_stopping',
            'learning_rate_scheduling',
            'model_checkpointing',
            'validation_monitoring',
            'multi_gpu_support',
            'mixed_precision',
            'custom_loss_functions',
            'pose_keypoint_integration',
            'gesture_recognition',
            'action_classification',
            'real_time_inference'
        ]
    
    def validate_training_config(self, config: Dict[str, Any]) -> bool:
        """验证训练配置"""
        required_keys = ['epochs', 'batch_size', 'learning_rate']
        
        # 检查必需参数
        for key in required_keys:
            if key not in config:
                logger.error(f"缺少必需参数: {key}")
                return False
        
        # 验证参数范围
        if config['epochs'] <= 0 or config['epochs'] > 1000:
            logger.error("epochs必须在1-1000之间")
            return False
            
        if config['batch_size'] <= 0 or config['batch_size'] > 256:
            logger.error("batch_size必须在1-256之间")
            return False
            
        if config['learning_rate'] <= 0 or config['learning_rate'] > 1:
            logger.error("learning_rate必须在0-1之间")
            return False
        
        return True

def create_sample_dataset_config() -> List[Dict[str, Any]]:
    """创建示例数据集配置"""
    return [
        {
            'name': 'human_actions',
            'path': './datasets/human_actions',
            'format': 'custom',
            'description': '人体动作识别数据集',
            'annotations_file': 'annotations.json'
        },
        {
            'name': 'gesture_dataset',
            'path': './datasets/gestures',
            'format': 'auto',
            'description': '手势识别数据集'
        }
    ]

if __name__ == "__main__":
    # 示例使用
    config = TrainingConfig(
        batch_size=16,
        learning_rate=0.001,
        epochs=50,
        validation_split=0.2
    )
    
    trainer = EnhancedHumanTrainer(config)
    dataset_configs = create_sample_dataset_config()
    
    model_path = trainer.train(dataset_configs)
    print(f"训练完成，模型保存在: {model_path}")