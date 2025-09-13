#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线训练管理器
支持所有识别场景的预训练，确保弱网环境下的可用性
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import cv2
import torch
import torch.nn as nn
from dataclasses import dataclass
from datetime import datetime
from ..models.enhanced_mish_activation import EnhancedMish, MishVariants

logger = logging.getLogger(__name__)

@dataclass
class OfflineModelConfig:
    """离线模型配置"""
    model_name: str
    model_type: str  # 'classification', 'detection', 'recognition'
    input_size: Tuple[int, int]
    num_classes: int
    class_names: List[str]
    model_path: str
    weights_path: str
    config_path: str
    created_time: str
    version: str

class OfflineTrainingManager:
    """离线训练管理器"""
    
    def __init__(self, base_dir: str = "./models"):
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "offline_models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # 各场景模型目录
        self.scene_dirs = {
            'pets': self.models_dir / 'pets',
            'plants': self.models_dir / 'plants', 
            'traffic': self.models_dir / 'traffic',
            'public_signs': self.models_dir / 'public_signs',
            'medicines': self.models_dir / 'medicines',
            'qr_codes': self.models_dir / 'qr_codes',
            'barcodes': self.models_dir / 'barcodes',
            'dynamic_objects': self.models_dir / 'dynamic_objects',
            'human_actions': self.models_dir / 'human_actions'
        }
        
        # 创建所有场景目录
        for scene_dir in self.scene_dirs.values():
            scene_dir.mkdir(parents=True, exist_ok=True)
            (scene_dir / 'weights').mkdir(exist_ok=True)
            (scene_dir / 'configs').mkdir(exist_ok=True)
            (scene_dir / 'datasets').mkdir(exist_ok=True)
        
        # 模型注册表
        self.model_registry = {}
        self.load_model_registry()
        
        # 场景配置
        self.scene_configs = self._init_scene_configs()
        
        logger.info("离线训练管理器初始化完成")
    
    def _init_scene_configs(self) -> Dict[str, Dict]:
        """初始化各场景配置"""
        return {
            'pets': {
                'classes': [
                    'dog', 'cat', 'bird', 'rabbit', 'hamster', 'fish',
                    'parrot', 'canary', 'goldfish', 'turtle', 'snake'
                ],
                'colors': ['brown', 'black', 'white', 'gray', 'orange', 'yellow', 'green'],
                'input_size': (224, 224),
                'model_type': 'classification'
            },
            'plants': {
                'classes': [
                    'rose', 'sunflower', 'tulip', 'daisy', 'lily', 'orchid',
                    'cactus', 'fern', 'bamboo', 'tree', 'grass', 'moss'
                ],
                'health_states': ['healthy', 'diseased', 'wilted', 'flowering'],
                'input_size': (224, 224),
                'model_type': 'classification'
            },
            'traffic': {
                'classes': [
                    'stop_sign', 'yield_sign', 'speed_limit', 'no_entry',
                    'traffic_light_red', 'traffic_light_yellow', 'traffic_light_green',
                    'pedestrian_crossing', 'school_zone', 'construction'
                ],
                'input_size': (224, 224),
                'model_type': 'detection'
            },
            'public_signs': {
                'classes': [
                    'restroom', 'exit', 'elevator', 'stairs', 'parking',
                    'hospital', 'pharmacy', 'restaurant', 'hotel', 'bank'
                ],
                'input_size': (224, 224),
                'model_type': 'detection'
            },
            'medicines': {
                'classes': [
                    'pill_round', 'pill_oval', 'capsule', 'tablet',
                    'liquid_bottle', 'injection', 'inhaler', 'patch'
                ],
                'colors': ['white', 'red', 'blue', 'yellow', 'green', 'pink'],
                'input_size': (224, 224),
                'model_type': 'classification'
            },
            'qr_codes': {
                'classes': ['qr_code', 'data_matrix', 'aztec_code'],
                'input_size': (224, 224),
                'model_type': 'detection'
            },
            'barcodes': {
                'classes': [
                    'ean13', 'ean8', 'upc_a', 'upc_e', 'code128',
                    'code39', 'code93', 'codabar'
                ],
                'input_size': (224, 224),
                'model_type': 'detection'
            },
            'dynamic_objects': {
                'classes': [
                    'person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck',
                    'airplane', 'boat', 'train', 'skateboard', 'surfboard'
                ],
                'motion_types': ['static', 'slow', 'medium', 'fast'],
                'input_size': (224, 224),
                'model_type': 'detection'
            },
            'human_actions': {
                'classes': [
                    'standing', 'walking', 'running', 'sitting', 'jumping',
                    'waving', 'pointing', 'clapping', 'stretching', 'bending'
                ],
                'input_size': (224, 224),
                'model_type': 'classification'
            }
        }
    
    def create_offline_dataset(self, scene: str, num_samples: int = 1000) -> bool:
        """为指定场景创建离线数据集"""
        if scene not in self.scene_configs:
            logger.error(f"未知场景: {scene}")
            return False
        
        config = self.scene_configs[scene]
        dataset_dir = self.scene_dirs[scene] / 'datasets' / 'synthetic'
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        images_dir = dataset_dir / 'images'
        images_dir.mkdir(exist_ok=True)
        
        logger.info(f"为场景 {scene} 创建 {num_samples} 个合成样本")
        
        annotations = []
        
        for i in range(num_samples):
            # 生成合成图像
            image = self._generate_synthetic_image(scene, config)
            
            # 随机选择类别
            class_idx = np.random.randint(0, len(config['classes']))
            class_name = config['classes'][class_idx]
            
            # 保存图像
            image_filename = f"{scene}_{i:06d}.jpg"
            image_path = images_dir / image_filename
            cv2.imwrite(str(image_path), image)
            
            # 创建标注
            annotation = {
                'image_path': f'images/{image_filename}',
                'class_id': class_idx,
                'class_name': class_name,
                'scene': scene,
                'synthetic': True
            }
            
            # 添加场景特定信息
            if 'colors' in config:
                color_idx = np.random.randint(0, len(config['colors']))
                annotation['color'] = config['colors'][color_idx]
            
            if 'motion_types' in config:
                motion_idx = np.random.randint(0, len(config['motion_types']))
                annotation['motion_type'] = config['motion_types'][motion_idx]
            
            annotations.append(annotation)
        
        # 保存标注文件
        annotations_file = dataset_dir / 'annotations.json'
        with open(annotations_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        # 保存数据集信息
        dataset_info = {
            'scene': scene,
            'num_samples': num_samples,
            'num_classes': len(config['classes']),
            'class_names': config['classes'],
            'input_size': config['input_size'],
            'created_time': datetime.now().isoformat(),
            'synthetic': True
        }
        
        info_file = dataset_dir / 'dataset_info.json'
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        logger.info(f"场景 {scene} 数据集创建完成: {dataset_dir}")
        return True
    
    def _generate_synthetic_image(self, scene: str, config: Dict) -> np.ndarray:
        """生成场景特定的合成图像"""
        height, width = config['input_size']
        image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        if scene == 'pets':
            # 绘制宠物轮廓
            center_x, center_y = width // 2, height // 2
            # 头部
            cv2.circle(image, (center_x, center_y - 30), 20, (255, 255, 255), -1)
            # 身体
            cv2.ellipse(image, (center_x, center_y + 10), (25, 40), 0, 0, 360, (255, 255, 255), -1)
            
        elif scene == 'plants':
            # 绘制植物形状
            # 茎
            cv2.line(image, (width//2, height-10), (width//2, height//2), (0, 255, 0), 3)
            # 叶子
            for i in range(3):
                y = height//2 + i * 20
                cv2.ellipse(image, (width//2 - 15, y), (10, 15), 45, 0, 360, (0, 255, 0), -1)
                cv2.ellipse(image, (width//2 + 15, y), (10, 15), -45, 0, 360, (0, 255, 0), -1)
        
        elif scene == 'traffic':
            # 绘制交通标志
            # 八角形停止标志
            if np.random.random() > 0.5:
                points = np.array([
                    [width//2 - 30, height//2 - 10],
                    [width//2 - 10, height//2 - 30],
                    [width//2 + 10, height//2 - 30],
                    [width//2 + 30, height//2 - 10],
                    [width//2 + 30, height//2 + 10],
                    [width//2 + 10, height//2 + 30],
                    [width//2 - 10, height//2 + 30],
                    [width//2 - 30, height//2 + 10]
                ], np.int32)
                cv2.fillPoly(image, [points], (0, 0, 255))
        
        elif scene == 'medicines':
            # 绘制药物形状
            if np.random.random() > 0.5:
                # 圆形药片
                cv2.circle(image, (width//2, height//2), 25, (255, 255, 255), -1)
                cv2.circle(image, (width//2, height//2), 25, (0, 0, 0), 2)
            else:
                # 胶囊
                cv2.ellipse(image, (width//2, height//2), (15, 30), 0, 0, 360, (255, 0, 0), -1)
        
        elif scene == 'qr_codes' or scene == 'barcodes':
            # 绘制码图案
            if scene == 'qr_codes':
                # QR码模式
                for i in range(0, width, 10):
                    for j in range(0, height, 10):
                        if np.random.random() > 0.5:
                            cv2.rectangle(image, (i, j), (i+8, j+8), (0, 0, 0), -1)
            else:
                # 条形码模式
                for i in range(0, width, 3):
                    if np.random.random() > 0.5:
                        cv2.line(image, (i, 50), (i, height-50), (0, 0, 0), 2)
        
        return image
    
    def train_offline_model(self, scene: str, epochs: int = 50) -> bool:
        """训练离线模型"""
        if scene not in self.scene_configs:
            logger.error(f"未知场景: {scene}")
            return False
        
        config = self.scene_configs[scene]
        
        # 检查数据集
        dataset_dir = self.scene_dirs[scene] / 'datasets' / 'synthetic'
        if not dataset_dir.exists():
            logger.info(f"数据集不存在，创建合成数据集: {scene}")
            self.create_offline_dataset(scene)
        
        # 创建模型
        model = self._create_scene_model(scene, config)
        if model is None:
            return False
        
        # 加载数据
        train_loader, val_loader = self._create_data_loaders(scene, dataset_dir)
        if train_loader is None:
            return False
        
        # 训练配置
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        best_val_acc = 0.0
        
        logger.info(f"开始训练场景 {scene} 的离线模型")
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_acc = train_correct / train_total
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_acc = val_correct / val_total
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save_offline_model(scene, model, config, val_acc)
        
        logger.info(f"场景 {scene} 训练完成，最佳验证准确率: {best_val_acc:.3f}")
        return True
    
    def _create_scene_model(self, scene: str, config: Dict) -> Optional[nn.Module]:
        """为场景创建模型"""
        try:
            num_classes = len(config['classes'])
            
            # 简单的CNN模型
            model = nn.Sequential(
                # 特征提取层
                nn.Conv2d(3, 32, 3, padding=1),
                MishVariants.adaptive_mish(learnable=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(32, 64, 3, padding=1),
                MishVariants.adaptive_mish(learnable=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 128, 3, padding=1),
                MishVariants.standard_mish(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(128, 256, 3, padding=1),
                MishVariants.standard_mish(),
                nn.AdaptiveAvgPool2d((1, 1)),
                
                # 分类层
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                MishVariants.adaptive_mish(learnable=True),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
            
            return model
            
        except Exception as e:
            logger.error(f"创建模型失败 {scene}: {e}")
            return None
    
    def _create_data_loaders(self, scene: str, dataset_dir: Path):
        """创建数据加载器"""
        try:
            from torch.utils.data import Dataset, DataLoader
            import torchvision.transforms as transforms
            
            # 加载标注
            annotations_file = dataset_dir / 'annotations.json'
            with open(annotations_file, 'r') as f:
                annotations = json.load(f)
            
            # 数据变换
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            # 自定义数据集类
            class SceneDataset(Dataset):
                def __init__(self, annotations, dataset_dir, transform=None):
                    self.annotations = annotations
                    self.dataset_dir = dataset_dir
                    self.transform = transform
                
                def __len__(self):
                    return len(self.annotations)
                
                def __getitem__(self, idx):
                    ann = self.annotations[idx]
                    image_path = self.dataset_dir / ann['image_path']
                    
                    image = cv2.imread(str(image_path))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    if self.transform:
                        image = self.transform(image)
                    
                    label = ann['class_id']
                    return image, label
            
            # 分割数据集
            split_idx = int(0.8 * len(annotations))
            train_annotations = annotations[:split_idx]
            val_annotations = annotations[split_idx:]
            
            # 创建数据集
            train_dataset = SceneDataset(train_annotations, dataset_dir, transform)
            val_dataset = SceneDataset(val_annotations, dataset_dir, transform)
            
            # 创建数据加载器
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
            
            return train_loader, val_loader
            
        except Exception as e:
            logger.error(f"创建数据加载器失败 {scene}: {e}")
            return None, None
    
    def _save_offline_model(self, scene: str, model: nn.Module, config: Dict, accuracy: float):
        """保存离线模型"""
        try:
            # 模型配置
            model_config = OfflineModelConfig(
                model_name=f"{scene}_offline_model",
                model_type=config['model_type'],
                input_size=config['input_size'],
                num_classes=len(config['classes']),
                class_names=config['classes'],
                model_path=str(self.scene_dirs[scene] / 'weights' / f"{scene}_model.pth"),
                weights_path=str(self.scene_dirs[scene] / 'weights' / f"{scene}_weights.pth"),
                config_path=str(self.scene_dirs[scene] / 'configs' / f"{scene}_config.json"),
                created_time=datetime.now().isoformat(),
                version="1.0.0"
            )
            
            # 保存模型权重
            torch.save(model.state_dict(), model_config.weights_path)
            
            # 保存完整模型
            torch.save(model, model_config.model_path)
            
            # 保存配置
            config_data = {
                'model_config': model_config.__dict__,
                'scene_config': config,
                'training_accuracy': accuracy,
                'offline_ready': True
            }
            
            with open(model_config.config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            # 注册模型
            self.model_registry[scene] = model_config
            self.save_model_registry()
            
            logger.info(f"离线模型已保存: {scene} (准确率: {accuracy:.3f})")
            
        except Exception as e:
            logger.error(f"保存模型失败 {scene}: {e}")
    
    def load_offline_model(self, scene: str) -> Optional[nn.Module]:
        """加载离线模型"""
        if scene not in self.model_registry:
            logger.warning(f"场景 {scene} 的离线模型未找到")
            return None
        
        try:
            model_config = self.model_registry[scene]
            model = torch.load(model_config.model_path, map_location='cpu')
            model.eval()
            
            logger.info(f"离线模型加载成功: {scene}")
            return model
            
        except Exception as e:
            logger.error(f"加载离线模型失败 {scene}: {e}")
            return None
    
    def train_all_scenes(self, epochs: int = 30):
        """训练所有场景的离线模型"""
        logger.info("开始训练所有场景的离线模型")
        
        for scene in self.scene_configs.keys():
            logger.info(f"训练场景: {scene}")
            success = self.train_offline_model(scene, epochs)
            if success:
                logger.info(f"✓ 场景 {scene} 训练完成")
            else:
                logger.error(f"✗ 场景 {scene} 训练失败")
        
        logger.info("所有场景训练完成")
    
    def get_offline_status(self) -> Dict[str, Any]:
        """获取离线模型状态"""
        status = {
            'total_scenes': len(self.scene_configs),
            'trained_scenes': len(self.model_registry),
            'scenes_status': {},
            'offline_ready': len(self.model_registry) == len(self.scene_configs)
        }
        
        for scene in self.scene_configs.keys():
            if scene in self.model_registry:
                config_path = self.model_registry[scene].config_path
                try:
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    
                    status['scenes_status'][scene] = {
                        'trained': True,
                        'accuracy': config_data.get('training_accuracy', 0.0),
                        'created_time': self.model_registry[scene].created_time,
                        'model_size': os.path.getsize(self.model_registry[scene].model_path) / (1024*1024)  # MB
                    }
                except:
                    status['scenes_status'][scene] = {'trained': False, 'error': 'Config file missing'}
            else:
                status['scenes_status'][scene] = {'trained': False}
        
        return status
    
    def save_model_registry(self):
        """保存模型注册表"""
        registry_file = self.models_dir / 'model_registry.json'
        registry_data = {
            scene: config.__dict__ for scene, config in self.model_registry.items()
        }
        
        with open(registry_file, 'w') as f:
            json.dump(registry_data, f, indent=2)
    
    def load_model_registry(self):
        """加载模型注册表"""
        registry_file = self.models_dir / 'model_registry.json'
        
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    registry_data = json.load(f)
                
                for scene, config_dict in registry_data.items():
                    self.model_registry[scene] = OfflineModelConfig(**config_dict)
                
                logger.info(f"加载模型注册表: {len(self.model_registry)} 个模型")
                
            except Exception as e:
                logger.error(f"加载模型注册表失败: {e}")
    
    def export_offline_package(self, output_path: str):
        """导出离线模型包"""
        import zipfile
        
        output_file = Path(output_path)
        
        with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 添加所有模型文件
            for scene, config in self.model_registry.items():
                # 模型权重
                if os.path.exists(config.model_path):
                    zipf.write(config.model_path, f"models/{scene}/model.pth")
                
                # 配置文件
                if os.path.exists(config.config_path):
                    zipf.write(config.config_path, f"models/{scene}/config.json")
            
            # 添加注册表
            registry_file = self.models_dir / 'model_registry.json'
            if registry_file.exists():
                zipf.write(registry_file, "model_registry.json")
        
        logger.info(f"离线模型包已导出: {output_file}")

if __name__ == "__main__":
    # 示例使用
    manager = OfflineTrainingManager()
    
    # 训练所有场景
    manager.train_all_scenes(epochs=20)
    
    # 检查状态
    status = manager.get_offline_status()
    print(f"离线模型状态: {status}")
    
    # 导出离线包
    manager.export_offline_package("offline_models.zip")