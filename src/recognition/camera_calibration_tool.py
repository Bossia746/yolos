"""相机标定工具 - 用于距离测量的相机焦距标定

提供交互式相机标定功能，支持多种标定方法和标定数据管理。
专门为距离测量应用优化。
"""

import cv2
import numpy as np
import json
import os
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import time
from datetime import datetime

from .distance_estimator import DistanceEstimator
from .enhanced_object_detector import EnhancedObjectDetector


class CameraCalibrationTool:
    """相机标定工具"""
    
    def __init__(self, config_dir: str = "calibration_data"):
        """
        初始化相机标定工具
        
        Args:
            config_dir: 标定数据存储目录
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self.distance_estimator = DistanceEstimator()
        self.object_detector = EnhancedObjectDetector()
        
        # 标定状态
        self.is_calibrating = False
        self.calibration_samples = []
        self.current_sample = {}
        
        # 标定参数
        self.known_objects = {
            'A4_paper': {'width': 21.0, 'height': 29.7, 'unit': 'cm'},
            'letter_paper': {'width': 8.5, 'height': 11.0, 'unit': 'inch'},
            'credit_card': {'width': 8.56, 'height': 5.398, 'unit': 'cm'},
            'coin_1yuan': {'width': 2.5, 'height': 2.5, 'unit': 'cm'},
            'smartphone': {'width': 7.0, 'height': 14.0, 'unit': 'cm'}
        }
        
        # 加载已有标定数据
        self.load_calibration_history()
    
    def load_calibration_history(self):
        """加载标定历史"""
        history_file = self.config_dir / "calibration_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    self.calibration_history = json.load(f)
                print(f"已加载 {len(self.calibration_history)} 条标定记录")
            except Exception as e:
                print(f"加载标定历史失败: {e}")
                self.calibration_history = []
        else:
            self.calibration_history = []
    
    def save_calibration_history(self):
        """保存标定历史"""
        history_file = self.config_dir / "calibration_history.json"
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.calibration_history, f, indent=2, ensure_ascii=False)
            print(f"标定历史已保存: {history_file}")
        except Exception as e:
            print(f"保存标定历史失败: {e}")
    
    def interactive_calibration(self, camera_id: int = 0, object_type: str = 'A4_paper'):
        """
        交互式相机标定
        
        Args:
            camera_id: 摄像头ID
            object_type: 已知物体类型
        """
        if object_type not in self.known_objects:
            print(f"不支持的物体类型: {object_type}")
            print(f"支持的类型: {list(self.known_objects.keys())}")
            return
        
        known_obj = self.known_objects[object_type]
        print(f"\n开始标定相机焦距")
        print(f"使用物体: {object_type}")
        print(f"物体尺寸: {known_obj['width']} x {known_obj['height']} {known_obj['unit']}")
        print("\n操作说明:")
        print("- 将物体放在不同距离处")
        print("- 按 's' 保存当前样本")
        print("- 按 'c' 完成标定")
        print("- 按 'r' 重置标定")
        print("- 按 'q' 退出")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"无法打开摄像头 {camera_id}")
            return
        
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_calibrating = True
        self.calibration_samples = []
        
        try:
            while self.is_calibrating:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 检测物体
                detection = self._detect_calibration_object(frame, object_type)
                
                # 绘制检测结果
                display_frame = self._draw_calibration_interface(frame, detection, object_type)
                
                cv2.imshow('Camera Calibration', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and detection:
                    self._save_calibration_sample(detection, object_type)
                elif key == ord('c'):
                    self._complete_calibration(object_type)
                elif key == ord('r'):
                    self._reset_calibration()
                
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def _detect_calibration_object(self, frame: np.ndarray, object_type: str) -> Optional[Dict[str, Any]]:
        """检测标定物体"""
        if object_type in ['A4_paper', 'letter_paper']:
            # 检测纸张类物体
            return self.object_detector.detect_paper_like_object(frame)
        elif object_type == 'credit_card':
            # 检测矩形物体
            detections = self.object_detector.detect_by_edge(frame, 'rectangle')
            if detections:
                # 选择长宽比接近信用卡的物体
                for detection in detections:
                    aspect_ratio = detection['aspect_ratio']
                    if 1.5 <= aspect_ratio <= 1.7:  # 信用卡长宽比约1.586
                        return detection
            return detections[0] if detections else None
        elif object_type in ['coin_1yuan']:
            # 检测圆形物体
            detections = self.object_detector.detect_by_edge(frame, 'circle')
            return detections[0] if detections else None
        else:
            # 默认检测最大物体
            return self.object_detector.detect_largest_object(frame)
    
    def _draw_calibration_interface(self, frame: np.ndarray, detection: Optional[Dict[str, Any]], 
                                   object_type: str) -> np.ndarray:
        """绘制标定界面"""
        display_frame = frame.copy()
        
        # 绘制检测结果
        if detection:
            # 绘制边界框
            bbox = detection['bbox']
            x, y, w, h = bbox
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 绘制中心点
            center = detection['center']
            cv2.circle(display_frame, center, 5, (255, 0, 0), -1)
            
            # 显示物体信息
            info_text = f"Object: {object_type}"
            cv2.putText(display_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            size_text = f"Size: {w}x{h} pixels"
            cv2.putText(display_frame, size_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            area_text = f"Area: {detection['area']:.0f}"
            cv2.putText(display_frame, area_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # 未检测到物体
            cv2.putText(display_frame, "No object detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 显示标定状态
        status_text = f"Samples: {len(self.calibration_samples)}"
        cv2.putText(display_frame, status_text, (10, display_frame.shape[0] - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # 显示操作提示
        if detection:
            hint_text = "Press 's' to save sample"
            cv2.putText(display_frame, hint_text, (10, display_frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return display_frame
    
    def _save_calibration_sample(self, detection: Dict[str, Any], object_type: str):
        """保存标定样本"""
        # 获取距离输入
        print(f"\n检测到物体，像素宽度: {detection['width']:.2f}")
        try:
            distance = float(input("请输入物体到相机的实际距离 (cm): "))
            
            sample = {
                'object_type': object_type,
                'pixel_width': detection['width'],
                'pixel_height': detection['height'],
                'actual_distance': distance,
                'detection_info': detection,
                'timestamp': datetime.now().isoformat()
            }
            
            self.calibration_samples.append(sample)
            print(f"样本已保存 (总计: {len(self.calibration_samples)} 个)")
            
        except ValueError:
            print("输入无效，样本未保存")
    
    def _complete_calibration(self, object_type: str):
        """完成标定"""
        if len(self.calibration_samples) < 2:
            print("至少需要2个样本才能完成标定")
            return
        
        known_obj = self.known_objects[object_type]
        known_width = known_obj['width']
        
        # 计算每个样本的焦距
        focal_lengths = []
        for sample in self.calibration_samples:
            focal_length = (sample['pixel_width'] * sample['actual_distance']) / known_width
            focal_lengths.append(focal_length)
            print(f"样本 {len(focal_lengths)}: 距离={sample['actual_distance']:.1f}cm, "
                  f"像素宽度={sample['pixel_width']:.1f}, 焦距={focal_length:.2f}")
        
        # 计算平均焦距
        avg_focal_length = np.mean(focal_lengths)
        std_focal_length = np.std(focal_lengths)
        
        print(f"\n标定结果:")
        print(f"平均焦距: {avg_focal_length:.2f}")
        print(f"标准差: {std_focal_length:.2f}")
        print(f"变异系数: {(std_focal_length/avg_focal_length)*100:.1f}%")
        
        # 保存标定结果
        calibration_result = {
            'focal_length': avg_focal_length,
            'std_deviation': std_focal_length,
            'object_type': object_type,
            'known_width': known_width,
            'unit': known_obj['unit'],
            'samples': self.calibration_samples,
            'timestamp': datetime.now().isoformat(),
            'sample_count': len(self.calibration_samples)
        }
        
        # 更新距离估算器
        self.distance_estimator.focal_length = avg_focal_length
        self.distance_estimator.calibration_data = calibration_result
        self.distance_estimator._save_config()
        
        # 保存到历史记录
        self.calibration_history.append(calibration_result)
        self.save_calibration_history()
        
        print(f"标定完成！焦距已设置为: {avg_focal_length:.2f}")
        self.is_calibrating = False
    
    def _reset_calibration(self):
        """重置标定"""
        self.calibration_samples = []
        print("标定已重置")
    
    def batch_calibration_from_images(self, image_dir: str, object_type: str, 
                                     distances: List[float]) -> bool:
        """
        从图像批量标定
        
        Args:
            image_dir: 图像目录
            object_type: 物体类型
            distances: 对应每张图像的距离列表
            
        Returns:
            标定是否成功
        """
        image_dir = Path(image_dir)
        if not image_dir.exists():
            print(f"图像目录不存在: {image_dir}")
            return False
        
        # 获取图像文件
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        image_files.sort()
        
        if len(image_files) != len(distances):
            print(f"图像数量({len(image_files)})与距离数量({len(distances)})不匹配")
            return False
        
        samples = []
        known_obj = self.known_objects[object_type]
        known_width = known_obj['width']
        
        for img_file, distance in zip(image_files, distances):
            print(f"处理图像: {img_file.name}")
            
            # 读取图像
            image = cv2.imread(str(img_file))
            if image is None:
                print(f"无法读取图像: {img_file}")
                continue
            
            # 检测物体
            detection = self._detect_calibration_object(image, object_type)
            if detection is None:
                print(f"未在图像中检测到物体: {img_file.name}")
                continue
            
            # 创建样本
            sample = {
                'object_type': object_type,
                'pixel_width': detection['width'],
                'pixel_height': detection['height'],
                'actual_distance': distance,
                'image_file': str(img_file),
                'detection_info': detection,
                'timestamp': datetime.now().isoformat()
            }
            samples.append(sample)
            
            print(f"  像素宽度: {detection['width']:.1f}, 距离: {distance}cm")
        
        if len(samples) < 2:
            print("有效样本不足，无法完成标定")
            return False
        
        # 计算焦距
        focal_lengths = []
        for sample in samples:
            focal_length = (sample['pixel_width'] * sample['actual_distance']) / known_width
            focal_lengths.append(focal_length)
        
        avg_focal_length = np.mean(focal_lengths)
        std_focal_length = np.std(focal_lengths)
        
        # 保存标定结果
        calibration_result = {
            'focal_length': avg_focal_length,
            'std_deviation': std_focal_length,
            'object_type': object_type,
            'known_width': known_width,
            'unit': known_obj['unit'],
            'samples': samples,
            'timestamp': datetime.now().isoformat(),
            'sample_count': len(samples),
            'calibration_method': 'batch_from_images'
        }
        
        # 更新距离估算器
        self.distance_estimator.focal_length = avg_focal_length
        self.distance_estimator.calibration_data = calibration_result
        self.distance_estimator._save_config()
        
        # 保存到历史记录
        self.calibration_history.append(calibration_result)
        self.save_calibration_history()
        
        print(f"\n批量标定完成:")
        print(f"平均焦距: {avg_focal_length:.2f}")
        print(f"标准差: {std_focal_length:.2f}")
        print(f"有效样本: {len(samples)}")
        
        return True
    
    def validate_calibration(self, test_image_path: str, actual_distance: float, 
                           object_type: str) -> Dict[str, Any]:
        """
        验证标定结果
        
        Args:
            test_image_path: 测试图像路径
            actual_distance: 实际距离
            object_type: 物体类型
            
        Returns:
            验证结果
        """
        if self.distance_estimator.focal_length is None:
            return {'error': '相机未标定'}
        
        # 读取测试图像
        test_image = cv2.imread(test_image_path)
        if test_image is None:
            return {'error': f'无法读取图像: {test_image_path}'}
        
        # 估算距离
        known_width = self.known_objects[object_type]['width']
        result = self.distance_estimator.estimate_distance(test_image, known_width)
        
        if result is None:
            return {'error': '无法检测到目标物体'}
        
        estimated_distance = result['distance']
        error = abs(estimated_distance - actual_distance)
        error_percentage = (error / actual_distance) * 100
        
        validation_result = {
            'test_image': test_image_path,
            'actual_distance': actual_distance,
            'estimated_distance': estimated_distance,
            'absolute_error': error,
            'error_percentage': error_percentage,
            'focal_length': self.distance_estimator.focal_length,
            'pixel_width': result['pixel_width'],
            'known_width': known_width,
            'object_type': object_type
        }
        
        return validation_result
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """获取标定摘要"""
        if not self.calibration_history:
            return {'message': '无标定记录'}
        
        latest_calibration = self.calibration_history[-1]
        
        summary = {
            'total_calibrations': len(self.calibration_history),
            'latest_calibration': {
                'focal_length': latest_calibration['focal_length'],
                'object_type': latest_calibration['object_type'],
                'sample_count': latest_calibration['sample_count'],
                'timestamp': latest_calibration['timestamp'],
                'std_deviation': latest_calibration.get('std_deviation', 0)
            },
            'available_objects': list(self.known_objects.keys()),
            'is_calibrated': self.distance_estimator.focal_length is not None
        }
        
        return summary
    
    def add_custom_object(self, name: str, width: float, height: float, unit: str = 'cm'):
        """添加自定义物体"""
        self.known_objects[name] = {
            'width': width,
            'height': height,
            'unit': unit
        }
        print(f"已添加自定义物体: {name} ({width}x{height} {unit})")


if __name__ == "__main__":
    # 使用示例
    calibration_tool = CameraCalibrationTool()
    
    # 交互式标定
    calibration_tool.interactive_calibration(camera_id=0, object_type='A4_paper')
    
    # 获取标定摘要
    summary = calibration_tool.get_calibration_summary()
    print(json.dumps(summary, indent=2, ensure_ascii=False))