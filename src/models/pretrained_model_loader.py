#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预训练模型加载器
支持加载各种预训练的人体识别模型
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

class PretrainedModelLoader:
    """预训练模型加载器"""
    
    def __init__(self, models_dir: str = "./models/pretrained"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # 预训练模型配置
        self.model_configs = {
            'resnet50_action': {
                'name': 'ResNet50 Action Recognition',
                'url': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
                'description': 'ResNet50预训练模型，适用于动作识别',
                'input_size': (224, 224),
                'num_classes': 1000,
                'architecture': 'resnet50'
            },
            'mobilenet_v3_pose': {
                'name': 'MobileNetV3 Pose Estimation',
                'url': 'https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth',
                'description': 'MobileNetV3轻量级姿势估计模型',
                'input_size': (224, 224),
                'num_classes': 1000,
                'architecture': 'mobilenet_v3'
            },
            'efficientnet_b0': {
                'name': 'EfficientNet-B0',
                'url': 'https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth',
                'description': 'EfficientNet-B0高效特征提取器',
                'input_size': (224, 224),
                'num_classes': 1000,
                'architecture': 'efficientnet_b0'
            }
        }
        
        # 已加载的模型缓存
        self.loaded_models = {}
        
        # 数据预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def download_model(self, model_name: str) -> bool:
        """下载预训练模型"""
        if model_name not in self.model_configs:
            logger.error(f"未知模型: {model_name}")
            return False
        
        config = self.model_configs[model_name]
        model_path = self.models_dir / f"{model_name}.pth"
        
        if model_path.exists():
            logger.info(f"模型已存在: {model_path}")
            return True
        
        try:
            logger.info(f"下载模型: {config['name']}")
            response = requests.get(config['url'], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(model_path, 'wb') as f, tqdm(
                desc=model_name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"模型下载完成: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"下载模型失败 {model_name}: {e}")
            return False
    
    def load_model(self, model_name: str, num_classes: Optional[int] = None) -> Optional[nn.Module]:
        """加载预训练模型"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        if model_name not in self.model_configs:
            logger.error(f"未知模型: {model_name}")
            return None
        
        config = self.model_configs[model_name]
        
        # 下载模型（如果需要）
        if not self.download_model(model_name):
            return None
        
        try:
            # 创建模型架构
            if config['architecture'] == 'resnet50':
                model = models.resnet50(pretrained=False)
                if num_classes and num_classes != 1000:
                    model.fc = nn.Linear(model.fc.in_features, num_classes)
            
            elif config['architecture'] == 'mobilenet_v3':
                model = models.mobilenet_v3_large(pretrained=False)
                if num_classes and num_classes != 1000:
                    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
            
            elif config['architecture'] == 'efficientnet_b0':
                model = models.efficientnet_b0(pretrained=False)
                if num_classes and num_classes != 1000:
                    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
            
            else:
                logger.error(f"不支持的架构: {config['architecture']}")
                return None
            
            # 加载预训练权重
            model_path = self.models_dir / f"{model_name}.pth"
            if model_path.exists():
                try:
                    state_dict = torch.load(model_path, map_location='cpu')
                    
                    # 处理不匹配的键
                    model_dict = model.state_dict()
                    pretrained_dict = {k: v for k, v in state_dict.items() 
                                     if k in model_dict and v.size() == model_dict[k].size()}
                    
                    model_dict.update(pretrained_dict)
                    model.load_state_dict(model_dict)
                    
                    logger.info(f"加载预训练权重: {len(pretrained_dict)}/{len(model_dict)} 层")
                    
                except Exception as e:
                    logger.warning(f"加载预训练权重失败，使用随机初始化: {e}")
            
            model.eval()
            self.loaded_models[model_name] = model
            
            logger.info(f"模型加载成功: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"加载模型失败 {model_name}: {e}")
            return None
    
    def extract_features(self, model: nn.Module, image: np.ndarray) -> Optional[np.ndarray]:
        """提取图像特征"""
        try:
            # 预处理图像
            if len(image.shape) == 3:
                image_tensor = self.transform(image).unsqueeze(0)
            else:
                logger.error("输入图像格式错误")
                return None
            
            # 提取特征
            with torch.no_grad():
                # 移除最后的分类层，获取特征
                if hasattr(model, 'fc'):  # ResNet
                    features = model.avgpool(model.layer4(
                        model.layer3(model.layer2(model.layer1(
                            model.maxpool(model.relu(model.bn1(model.conv1(image_tensor))))
                        )))
                    ))
                    features = torch.flatten(features, 1)
                
                elif hasattr(model, 'classifier'):  # MobileNet, EfficientNet
                    features = model.features(image_tensor)
                    features = model.avgpool(features)
                    features = torch.flatten(features, 1)
                
                else:
                    # 通用方法：移除最后一层
                    layers = list(model.children())[:-1]
                    feature_extractor = nn.Sequential(*layers)
                    features = feature_extractor(image_tensor)
                    features = torch.flatten(features, 1)
            
            return features.numpy()
            
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            return None
    
    def create_action_classifier(self, 
                               backbone_name: str,
                               num_action_classes: int,
                               freeze_backbone: bool = True) -> Optional[nn.Module]:
        """创建动作分类器"""
        backbone = self.load_model(backbone_name)
        if backbone is None:
            return None
        
        try:
            # 冻结backbone参数
            if freeze_backbone:
                for param in backbone.parameters():
                    param.requires_grad = False
            
            # 获取特征维度
            if hasattr(backbone, 'fc'):
                feature_dim = backbone.fc.in_features
                backbone.fc = nn.Identity()  # 移除分类层
            elif hasattr(backbone, 'classifier'):
                if isinstance(backbone.classifier, nn.Sequential):
                    feature_dim = backbone.classifier[-1].in_features
                    backbone.classifier[-1] = nn.Identity()
                else:
                    feature_dim = backbone.classifier.in_features
                    backbone.classifier = nn.Identity()
            else:
                logger.error("无法确定特征维度")
                return None
            
            # 创建新的分类器
            classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_action_classes)
            )
            
            # 组合模型
            model = nn.Sequential(backbone, classifier)
            
            logger.info(f"动作分类器创建成功: {backbone_name} -> {num_action_classes} 类")
            return model
            
        except Exception as e:
            logger.error(f"创建动作分类器失败: {e}")
            return None
    
    def list_available_models(self):
        """列出可用的预训练模型"""
        print("\n=== 可用的预训练模型 ===")
        for name, config in self.model_configs.items():
            model_path = self.models_dir / f"{name}.pth"
            status = "已下载" if model_path.exists() else "未下载"
            
            print(f"\n{name} ({status}):")
            print(f"  名称: {config['name']}")
            print(f"  描述: {config['description']}")
            print(f"  输入尺寸: {config['input_size']}")
            print(f"  架构: {config['architecture']}")
    
    def download_all_models(self):
        """下载所有预训练模型"""
        for model_name in self.model_configs.keys():
            self.download_model(model_name)

class EnhancedFeatureExtractor:
    """增强特征提取器"""
    
    def __init__(self, model_loader: PretrainedModelLoader):
        self.model_loader = model_loader
        self.models = {}
        
        # 加载多个预训练模型
        self.load_ensemble_models()
    
    def load_ensemble_models(self):
        """加载集成模型"""
        model_names = ['resnet50_action', 'mobilenet_v3_pose', 'efficientnet_b0']
        
        for name in model_names:
            model = self.model_loader.load_model(name)
            if model is not None:
                self.models[name] = model
                logger.info(f"集成模型加载: {name}")
    
    def extract_multi_scale_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """提取多尺度特征"""
        features = {}
        
        # 多个尺度
        scales = [(224, 224), (256, 256), (192, 192)]
        
        for scale in scales:
            # 调整图像尺寸
            resized_image = cv2.resize(image, scale)
            
            # 从每个模型提取特征
            for model_name, model in self.models.items():
                try:
                    # 临时修改transform
                    transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize(scale),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                           std=[0.229, 0.224, 0.225])
                    ])
                    
                    image_tensor = transform(resized_image).unsqueeze(0)
                    
                    with torch.no_grad():
                        if hasattr(model, 'fc'):
                            # ResNet特征提取
                            x = model.conv1(image_tensor)
                            x = model.bn1(x)
                            x = model.relu(x)
                            x = model.maxpool(x)
                            x = model.layer1(x)
                            x = model.layer2(x)
                            x = model.layer3(x)
                            x = model.layer4(x)
                            x = model.avgpool(x)
                            feature = torch.flatten(x, 1)
                        else:
                            # 其他模型
                            feature = model.features(image_tensor)
                            feature = torch.flatten(feature, 1)
                    
                    key = f"{model_name}_{scale[0]}x{scale[1]}"
                    features[key] = feature.numpy()
                    
                except Exception as e:
                    logger.warning(f"特征提取失败 {model_name} at {scale}: {e}")
        
        return features
    
    def fuse_features(self, multi_features: Dict[str, np.ndarray]) -> np.ndarray:
        """融合多模型特征"""
        if not multi_features:
            return np.array([])
        
        # 特征标准化
        normalized_features = []
        for key, feature in multi_features.items():
            # L2标准化
            norm = np.linalg.norm(feature)
            if norm > 0:
                normalized_feature = feature / norm
            else:
                normalized_feature = feature
            normalized_features.append(normalized_feature.flatten())
        
        # 连接所有特征
        fused_feature = np.concatenate(normalized_features)
        
        return fused_feature

def create_enhanced_recognition_system():
    """创建增强识别系统示例"""
    # 初始化模型加载器
    model_loader = PretrainedModelLoader()
    
    # 下载预训练模型
    model_loader.download_all_models()
    
    # 创建特征提取器
    feature_extractor = EnhancedFeatureExtractor(model_loader)
    
    # 创建动作分类器
    action_classifier = model_loader.create_action_classifier(
        'resnet50_action', 
        num_action_classes=10,
        freeze_backbone=True
    )
    
    return {
        'model_loader': model_loader,
        'feature_extractor': feature_extractor,
        'action_classifier': action_classifier
    }

if __name__ == "__main__":
    # 示例使用
    model_loader = PretrainedModelLoader()
    
    # 列出可用模型
    model_loader.list_available_models()
    
    # 下载所有模型
    model_loader.download_all_models()
    
    # 加载模型
    resnet_model = model_loader.load_model('resnet50_action', num_classes=10)
    
    if resnet_model:
        print("模型加载成功！")
        
        # 测试特征提取
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        features = model_loader.extract_features(resnet_model, test_image)
        
        if features is not None:
            print(f"特征提取成功，特征维度: {features.shape}")