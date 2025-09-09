"""
视频检测器
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import json
import time

from ..models.yolo_factory import YOLOFactory


class VideoDetector:
    """视频检测器"""
    
    def __init__(self, 
                 model_type: str = 'yolov8',
                 model_path: Optional[str] = None,
                 device: str = 'auto'):
        """
        初始化视频检测器
        
        Args:
            model_type: 模型类型
            model_path: 模型路径
            device: 设备类型
        """
        self.model = YOLOFactory.create_model(model_type, model_path, device)
        self.progress_callback: Optional[Callable] = None
    
    def set_progress_callback(self, callback: Callable):
        """设置进度回调函数"""
        self.progress_callback = callback
    
    def detect_video(self, 
                    video_path: str,
                    output_path: Optional[str] = None,
                    save_frames: bool = False,
                    frame_interval: int = 1,
                    **kwargs) -> Dict[str, Any]:
        """
        检测视频
        
        Args:
            video_path: 视频路径
            output_path: 输出视频路径
            save_frames: 是否保存检测帧
            frame_interval: 帧间隔（每N帧检测一次）
            **kwargs: 其他参数
            
        Returns:
            检测结果统计
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"视频信息: {width}x{height}, {fps}FPS, {total_frames}帧, {duration:.1f}秒")
        
        # 设置输出视频
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 创建帧保存目录
        frames_dir = None
        if save_frames:
            video_name = Path(video_path).stem
            frames_dir = Path(f"frames_{video_name}")
            frames_dir.mkdir(exist_ok=True)
        
        # 检测统计
        stats = {
            'video_path': video_path,
            'total_frames': total_frames,
            'processed_frames': 0,
            'total_detections': 0,
            'detection_history': [],
            'processing_time': 0,
            'fps_avg': 0
        }
        
        frame_idx = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_results = []
                
                # 按间隔检测
                if frame_idx % frame_interval == 0:
                    # 执行检测
                    results = self.model.predict(frame, **kwargs)
                    frame_results = results
                    
                    # 绘制结果
                    annotated_frame = self.model.draw_results(frame, results)
                    
                    # 更新统计
                    stats['total_detections'] += len(results)
                    stats['detection_history'].append({
                        'frame': frame_idx,
                        'timestamp': frame_idx / fps,
                        'detections': len(results),
                        'objects': [r['class_name'] for r in results]
                    })
                    
                    # 保存帧
                    if save_frames and results:
                        frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
                        cv2.imwrite(str(frame_path), annotated_frame)
                else:
                    annotated_frame = frame
                
                # 写入输出视频
                if out:
                    out.write(annotated_frame)
                
                frame_idx += 1
                stats['processed_frames'] = frame_idx
                
                # 更新进度
                if self.progress_callback and frame_idx % 30 == 0:
                    progress = (frame_idx / total_frames) * 100
                    self.progress_callback(progress, frame_idx, total_frames)
                
                # 控制台进度显示
                if frame_idx % 100 == 0:
                    progress = (frame_idx / total_frames) * 100
                    elapsed = time.time() - start_time
                    fps_current = frame_idx / elapsed if elapsed > 0 else 0
                    print(f"进度: {progress:.1f}% ({frame_idx}/{total_frames}), "
                          f"当前FPS: {fps_current:.1f}")
                    
        finally:
            cap.release()
            if out:
                out.release()
        
        # 计算最终统计
        end_time = time.time()
        stats['processing_time'] = end_time - start_time
        stats['fps_avg'] = stats['processed_frames'] / stats['processing_time']
        
        # 保存检测结果
        if output_path:
            json_path = Path(output_path).with_suffix('.json')
            self._save_video_results(stats, json_path)
        
        print(f"视频处理完成:")
        print(f"  处理帧数: {stats['processed_frames']}")
        print(f"  总检测数: {stats['total_detections']}")
        print(f"  处理时间: {stats['processing_time']:.1f}秒")
        print(f"  平均FPS: {stats['fps_avg']:.1f}")
        
        return stats
    
    def detect_webcam(self, 
                     camera_id: int = 0,
                     output_path: Optional[str] = None,
                     max_duration: Optional[int] = None,
                     **kwargs):
        """
        检测网络摄像头
        
        Args:
            camera_id: 摄像头ID
            output_path: 输出视频路径
            max_duration: 最大录制时长（秒）
            **kwargs: 其他参数
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开摄像头: {camera_id}")
        
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # 获取实际参数
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # 设置输出视频
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print("开始网络摄像头检测，按 'q' 退出")
        if max_duration:
            print(f"最大录制时长: {max_duration}秒")
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 检查时长限制
                if max_duration and (time.time() - start_time) > max_duration:
                    break
                
                # 执行检测
                results = self.model.predict(frame, **kwargs)
                
                # 绘制结果
                annotated_frame = self.model.draw_results(frame, results)
                
                # 显示FPS
                frame_count += 1
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 显示检测数量
                cv2.putText(annotated_frame, f"Objects: {len(results)}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 写入输出视频
                if out:
                    out.write(annotated_frame)
                
                # 显示图像
                cv2.imshow('网络摄像头检测', annotated_frame)
                
                # 检查退出条件
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            elapsed = time.time() - start_time
            print(f"录制完成: {frame_count}帧, {elapsed:.1f}秒, 平均FPS: {frame_count/elapsed:.1f}")
    
    def _save_video_results(self, stats: Dict[str, Any], json_path: Path):
        """保存视频检测结果"""
        stats['model_info'] = self.model.get_model_info()
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"检测结果已保存到: {json_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return self.model.get_model_info()