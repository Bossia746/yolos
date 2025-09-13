#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ModelScope增强分析器模块
提供基于ModelScope的增强分析功能
"""

import base64
import time
from typing import Dict, Any, Optional
from pathlib import Path

try:
    from openai import OpenAI
    MODELSCOPE_AVAILABLE = True
except ImportError:
    MODELSCOPE_AVAILABLE = False
    print("⚠️ OpenAI库未安装，将跳过ModelScope测试")


class ModelScopeEnhancedAnalyzer:
    """ModelScope增强分析器"""
    
    def __init__(self):
        """初始化增强分析器"""
        self.available = MODELSCOPE_AVAILABLE
        self.client = None
        
        if self.available:
            try:
                # 初始化OpenAI客户端（用于ModelScope API）
                self.client = OpenAI(
                    api_key="your-api-key",  # 实际使用时需要配置
                    base_url="https://api.modelscope.cn/v1"
                )
                print("✓ ModelScope增强分析器初始化成功")
            except Exception as e:
                print(f"⚠️ ModelScope客户端初始化失败: {e}")
                self.available = False
        
        if not self.available:
            print("⚠️ 使用模拟ModelScope分析结果")
    
    def analyze_with_context(self, image_path: str, yolo_results: Dict[str, Any]) -> Dict[str, Any]:
        """基于YOLO结果进行增强分析"""
        try:
            start_time = time.time()
            
            if self.available and self.client:
                # 使用真实的ModelScope API
                return self._real_modelscope_analysis(image_path, yolo_results, start_time)
            else:
                # 模拟ModelScope分析结果
                return self._mock_modelscope_analysis(image_path, yolo_results, start_time)
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "method": "ModelScope Enhanced",
                "processing_time": time.time() - start_time
            }
    
    def _real_modelscope_analysis(self, image_path: str, yolo_results: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """真实的ModelScope分析"""
        try:
            # 编码图像
            base64_image = self._encode_image(image_path)
            if not base64_image:
                raise Exception("图像编码失败")
            
            # 构建上下文提示
            context_prompt = self._build_context_prompt(yolo_results)
            
            # 调用ModelScope API
            response = self.client.chat.completions.create(
                model="qwen-vl-plus",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": context_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            analysis_text = response.choices[0].message.content
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "method": "ModelScope Enhanced Analysis",
                "analysis": analysis_text,
                "context_used": context_prompt,
                "processing_time": round(processing_time, 3),
                "model": "qwen-vl-plus"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "method": "ModelScope Enhanced",
                "processing_time": time.time() - start_time
            }
    
    def _mock_modelscope_analysis(self, image_path: str, yolo_results: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """模拟ModelScope分析结果"""
        try:
            # 模拟处理时间
            import time
            time.sleep(0.5)  # 模拟API调用延迟
            
            # 基于YOLO结果生成模拟分析
            detections = yolo_results.get('detections', [])
            detection_count = len(detections)
            
            if detection_count == 0:
                analysis_text = "图像分析：这是一个相对简洁的场景，没有检测到明显的物体。可能是空旷的环境或者光线条件不佳。建议检查图像质量和光照条件。"
            elif detection_count <= 2:
                analysis_text = f"图像分析：检测到{detection_count}个物体。这是一个相对简单的场景，物体分布较为稀疏。场景整体较为清晰，适合进行精确的物体识别和分析。"
            elif detection_count <= 5:
                analysis_text = f"图像分析：检测到{detection_count}个物体。这是一个中等复杂度的场景，物体分布适中。场景包含多种元素，体现了日常生活或工作环境的典型特征。"
            else:
                analysis_text = f"图像分析：检测到{detection_count}个物体。这是一个复杂的场景，物体密度较高。场景信息丰富，可能包含多种交互关系和上下文信息。"
            
            # 添加基于检测物体的具体分析
            if detections:
                object_types = [det.get('class', 'unknown') for det in detections]
                unique_objects = list(set(object_types))
                
                if 'person' in object_types:
                    analysis_text += " 场景中包含人物，这表明这是一个有人活动的环境。"
                
                if any(obj in object_types for obj in ['chair', 'table', 'sofa']):
                    analysis_text += " 检测到家具物品，这可能是室内环境。"
                
                if any(obj in object_types for obj in ['car', 'truck', 'bus']):
                    analysis_text += " 检测到交通工具，这可能是户外或交通场景。"
                
                analysis_text += f" 物体类型包括：{', '.join(unique_objects[:5])}等。"
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "method": "Mock ModelScope Analysis",
                "analysis": analysis_text,
                "context_used": self._build_context_prompt(yolo_results),
                "processing_time": round(processing_time, 3),
                "model": "mock-qwen-vl-plus",
                "note": "模拟结果 - 实际部署时将使用真实ModelScope API"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "method": "Mock ModelScope",
                "processing_time": time.time() - start_time
            }
    
    def _encode_image(self, image_path: str) -> Optional[str]:
        """将图像编码为base64"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"图像编码失败: {e}")
            return None
    
    def _build_context_prompt(self, yolo_results: Dict[str, Any]) -> str:
        """构建上下文提示"""
        detections = yolo_results.get('detections', [])
        detection_count = len(detections)
        
        prompt = f"""请基于以下YOLO检测结果对图像进行深度分析：

检测到的物体数量：{detection_count}

检测详情：
"""
        
        for i, detection in enumerate(detections[:10]):  # 限制显示前10个检测结果
            class_name = detection.get('class', 'unknown')
            confidence = detection.get('confidence', 0)
            prompt += f"{i+1}. {class_name} (置信度: {confidence:.2f})\n"
        
        prompt += """
请提供以下分析：
1. 场景整体描述和环境类型
2. 物体之间的空间关系和交互
3. 可能的活动或事件推断
4. 安全性和注意事项
5. 改进建议（如果适用）

请用中文回答，保持客观和专业。"""
        
        return prompt
    
    def is_available(self) -> bool:
        """检查分析器是否可用"""
        return self.available
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if self.available and self.client:
            return {
                "model_type": "qwen-vl-plus",
                "provider": "ModelScope",
                "available": True
            }
        else:
            return {
                "model_type": "mock-qwen-vl-plus",
                "provider": "Mock",
                "available": False,
                "note": "使用模拟分析结果"
            }