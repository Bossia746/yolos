"""
图像检测器
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import json

from ..models.yolo_factory import YOLOFactory


class ImageDetector:
    """图像检测器"""
    
    def __init__(self, 
                 model_type: str = 'yolov8',
                 model_path: Optional[str] = None,
                 device: str = 'auto'):
        """
        初始化图像检测器
        
        Args:
            model_type: 模型类型
            model_path: 模型路径
            device: 设备类型
        """
        self.model = YOLOFactory.create_model(model_type, model_path, device)
    
    def detect_image(self, 
                    image_path: str, 
                    output_path: Optional[str] = None,
                    save_results: bool = True,
                    **kwargs) -> List[Dict[str, Any]]:
        """
        检测单张图像
        
        Args:
            image_path: 图像路径
            output_path: 输出路径
            save_results: 是否保存结果
            **kwargs: 其他参数
            
        Returns:
            检测结果列表
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 执行检测
        results = self.model.predict(image, **kwargs)
        
        # 保存结果
        if save_results:
            # 绘制检测结果
            annotated_image = self.model.draw_results(image, results)
            
            # 确定输出路径
            if output_path is None:
                input_path = Path(image_path)
                output_path = input_path.parent / f"{input_path.stem}_detected{input_path.suffix}"
            
            # 保存图像
            cv2.imwrite(str(output_path), annotated_image)
            print(f"检测结果已保存到: {output_path}")
            
            # 保存JSON结果
            json_path = Path(output_path).with_suffix('.json')
            self._save_json_results(results, json_path, image_path)
        
        return results
    
    def detect_batch(self, 
                    image_paths: List[str],
                    output_dir: Optional[str] = None,
                    **kwargs) -> Dict[str, List[Dict[str, Any]]]:
        """
        批量检测图像
        
        Args:
            image_paths: 图像路径列表
            output_dir: 输出目录
            **kwargs: 其他参数
            
        Returns:
            检测结果字典
        """
        results = {}
        
        # 创建输出目录
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        for i, image_path in enumerate(image_paths):
            try:
                print(f"处理图像 {i+1}/{len(image_paths)}: {image_path}")
                
                # 确定输出路径
                if output_dir:
                    input_name = Path(image_path).name
                    output_file = output_path / f"detected_{input_name}"
                else:
                    output_file = None
                
                # 检测图像
                image_results = self.detect_image(
                    image_path, 
                    str(output_file) if output_file else None,
                    save_results=bool(output_dir),
                    **kwargs
                )
                
                results[image_path] = image_results
                
            except Exception as e:
                print(f"处理图像 {image_path} 时出错: {e}")
                results[image_path] = []
        
        # 保存批量结果摘要
        if output_dir:
            summary_path = output_path / "batch_results.json"
            self._save_batch_summary(results, summary_path)
        
        return results
    
    def detect_directory(self, 
                        input_dir: str,
                        output_dir: Optional[str] = None,
                        extensions: List[str] = None,
                        **kwargs) -> Dict[str, List[Dict[str, Any]]]:
        """
        检测目录中的所有图像
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            extensions: 支持的文件扩展名
            **kwargs: 其他参数
            
        Returns:
            检测结果字典
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        # 查找图像文件
        input_path = Path(input_dir)
        image_paths = []
        
        for ext in extensions:
            image_paths.extend(input_path.glob(f"*{ext}"))
            image_paths.extend(input_path.glob(f"*{ext.upper()}"))
        
        image_paths = [str(p) for p in image_paths]
        
        if not image_paths:
            print(f"在目录 {input_dir} 中未找到图像文件")
            return {}
        
        print(f"找到 {len(image_paths)} 个图像文件")
        
        # 批量检测
        return self.detect_batch(image_paths, output_dir, **kwargs)
    
    def _save_json_results(self, results: List[Dict[str, Any]], json_path: Path, image_path: str):
        """保存JSON格式的检测结果"""
        json_data = {
            'image_path': image_path,
            'detections': results,
            'model_info': self.model.get_model_info()
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    def _save_batch_summary(self, results: Dict[str, List[Dict[str, Any]]], summary_path: Path):
        """保存批量检测结果摘要"""
        summary = {
            'total_images': len(results),
            'total_detections': sum(len(detections) for detections in results.values()),
            'model_info': self.model.get_model_info(),
            'results': results
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"批量检测摘要已保存到: {summary_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return self.model.get_model_info()