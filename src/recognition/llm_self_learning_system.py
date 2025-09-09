#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大模型自学习系统
通过大模型API实现未知场景的自动识别和学习
"""

import cv2
import numpy as np
import base64
import json
import time
import requests
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

class LLMProvider(Enum):
    """大模型提供商"""
    OPENAI_GPT4V = "openai_gpt4v"
    CLAUDE_VISION = "claude_vision"
    GEMINI_PRO_VISION = "gemini_pro_vision"
    QWEN_VL = "qwen_vl"
    BAIDU_ERNIE_VL = "baidu_ernie_vl"
    ZHIPU_GLM4V = "zhipu_glm4v"

class SceneCategory(Enum):
    """场景类别"""
    UNKNOWN = "unknown"
    MEDICAL = "medical"
    MEDICATION = "medication"
    EMERGENCY = "emergency"
    DAILY_OBJECT = "daily_object"
    PERSON = "person"
    ANIMAL = "animal"
    PLANT = "plant"
    VEHICLE = "vehicle"
    FOOD = "food"
    TOOL = "tool"
    FURNITURE = "furniture"
    ELECTRONIC = "electronic"
    CLOTHING = "clothing"
    DOCUMENT = "document"

@dataclass
class LLMAnalysisResult:
    """大模型分析结果"""
    scene_description: str
    detected_objects: List[Dict[str, Any]]
    scene_category: SceneCategory
    confidence: float
    suggested_actions: List[str]
    learning_keywords: List[str]
    safety_assessment: Dict[str, Any]
    medical_relevance: Dict[str, Any]
    timestamp: float
    
@dataclass
class SelfLearningRecord:
    """自学习记录"""
    image_id: str
    original_prediction: Optional[str]
    llm_analysis: LLMAnalysisResult
    user_feedback: Optional[Dict[str, Any]]
    learning_outcome: str
    created_at: float
    updated_at: float

class LLMSelfLearningSystem:
    """大模型自学习系统"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # 初始化大模型客户端
        self.llm_clients = self._initialize_llm_clients()
        
        # 学习记录存储
        self.learning_records: List[SelfLearningRecord] = []
        self.knowledge_base: Dict[str, Any] = {}
        
        # 性能统计
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'learning_records': 0,
            'knowledge_entries': 0
        }
        
        # 加载已有知识库
        self._load_knowledge_base()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            # 大模型配置
            'llm': {
                'enabled': True,
                'primary_provider': LLMProvider.OPENAI_GPT4V,
                'fallback_providers': [LLMProvider.CLAUDE_VISION, LLMProvider.QWEN_VL],
                'max_retries': 3,
                'timeout': 30.0,
                'rate_limit_delay': 1.0
            },
            
            # API配置
            'api_keys': {
                'openai': 'your_openai_api_key',
                'claude': 'your_claude_api_key',
                'gemini': 'your_gemini_api_key',
                'qwen': 'your_qwen_api_key',
                'baidu': 'your_baidu_api_key',
                'zhipu': 'your_zhipu_api_key'
            },
            
            # 自学习配置
            'self_learning': {
                'confidence_threshold': 0.3,  # 低于此阈值触发自学习
                'auto_learning': True,        # 自动学习模式
                'require_confirmation': False, # 是否需要用户确认
                'max_learning_records': 10000, # 最大学习记录数
                'knowledge_update_interval': 3600  # 知识库更新间隔(秒)
            },
            
            # 图像处理配置
            'image_processing': {
                'max_image_size': (1024, 1024),  # 最大图像尺寸
                'image_quality': 85,              # JPEG质量
                'auto_enhance': True,             # 自动图像增强
                'include_metadata': True          # 包含元数据
            },
            
            # 提示词配置
            'prompts': {
                'system_prompt': """你是一个专业的视觉分析AI助手，专门用于医疗护理和安全监控场景。
请仔细分析图像并提供详细的场景描述和对象识别结果。""",
                
                'analysis_prompt': """请分析这张图像并提供以下信息：
1. 详细的场景描述
2. 检测到的所有对象及其属性
3. 场景类别分类
4. 安全性评估
5. 医疗相关性分析
6. 建议的处理行动
7. 学习关键词

请以JSON格式返回结果。""",
                
                'medical_prompt': """特别关注以下医疗相关内容：
- 药物和医疗器械
- 人员健康状态
- 紧急情况标识
- 安全隐患
- 需要医疗关注的情况"""
            },
            
            # 存储配置
            'storage': {
                'knowledge_base_path': 'data/knowledge_base.json',
                'learning_records_path': 'data/learning_records.json',
                'images_path': 'data/self_learning_images/',
                'backup_interval': 86400  # 备份间隔(秒)
            }
        }
    
    def _initialize_llm_clients(self) -> Dict[LLMProvider, Any]:
        """初始化大模型客户端"""
        clients = {}
        
        try:
            # OpenAI GPT-4V
            if self.config['api_keys'].get('openai'):
                clients[LLMProvider.OPENAI_GPT4V] = self._create_openai_client()
            
            # Claude Vision
            if self.config['api_keys'].get('claude'):
                clients[LLMProvider.CLAUDE_VISION] = self._create_claude_client()
            
            # Gemini Pro Vision
            if self.config['api_keys'].get('gemini'):
                clients[LLMProvider.GEMINI_PRO_VISION] = self._create_gemini_client()
            
            # 通义千问VL
            if self.config['api_keys'].get('qwen'):
                clients[LLMProvider.QWEN_VL] = self._create_qwen_client()
            
            # 百度文心一言VL
            if self.config['api_keys'].get('baidu'):
                clients[LLMProvider.BAIDU_ERNIE_VL] = self._create_baidu_client()
            
            # 智谱GLM-4V
            if self.config['api_keys'].get('zhipu'):
                clients[LLMProvider.ZHIPU_GLM4V] = self._create_zhipu_client()
                
        except Exception as e:
            self.logger.error(f"初始化大模型客户端失败: {e}")
        
        return clients
    
    def _create_openai_client(self) -> Dict[str, Any]:
        """创建OpenAI客户端"""
        return {
            'api_key': self.config['api_keys']['openai'],
            'base_url': 'https://api.openai.com/v1',
            'model': 'gpt-4-vision-preview',
            'max_tokens': 2000
        }
    
    def _create_claude_client(self) -> Dict[str, Any]:
        """创建Claude客户端"""
        return {
            'api_key': self.config['api_keys']['claude'],
            'base_url': 'https://api.anthropic.com/v1',
            'model': 'claude-3-opus-20240229',
            'max_tokens': 2000
        }
    
    def _create_gemini_client(self) -> Dict[str, Any]:
        """创建Gemini客户端"""
        return {
            'api_key': self.config['api_keys']['gemini'],
            'base_url': 'https://generativelanguage.googleapis.com/v1beta',
            'model': 'gemini-pro-vision',
            'max_tokens': 2000
        }
    
    def _create_qwen_client(self) -> Dict[str, Any]:
        """创建通义千问客户端"""
        return {
            'api_key': self.config['api_keys']['qwen'],
            'base_url': 'https://dashscope.aliyuncs.com/api/v1',
            'model': 'qwen-vl-plus',
            'max_tokens': 2000
        }
    
    def _create_baidu_client(self) -> Dict[str, Any]:
        """创建百度文心客户端"""
        return {
            'api_key': self.config['api_keys']['baidu'],
            'base_url': 'https://aip.baidubce.com/rpc/2.0/ai_custom/v1',
            'model': 'ernie-bot-4.0',
            'max_tokens': 2000
        }
    
    def _create_zhipu_client(self) -> Dict[str, Any]:
        """创建智谱GLM客户端"""
        return {
            'api_key': self.config['api_keys']['zhipu'],
            'base_url': 'https://open.bigmodel.cn/api/paas/v4',
            'model': 'glm-4v',
            'max_tokens': 2000
        }
    
    def analyze_unknown_scene(self, image: np.ndarray, 
                            context: Optional[Dict[str, Any]] = None,
                            original_prediction: Optional[str] = None) -> LLMAnalysisResult:
        """分析未知场景"""
        try:
            # 预处理图像
            processed_image = self._preprocess_image(image)
            
            # 编码图像
            image_base64 = self._encode_image_to_base64(processed_image)
            
            # 构建分析请求
            analysis_request = self._build_analysis_request(
                image_base64, context, original_prediction
            )
            
            # 调用大模型API
            llm_response = self._call_llm_api(analysis_request)
            
            # 解析响应
            analysis_result = self._parse_llm_response(llm_response)
            
            # 更新统计
            self.stats['total_queries'] += 1
            self.stats['successful_queries'] += 1
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"场景分析失败: {e}")
            self.stats['total_queries'] += 1
            self.stats['failed_queries'] += 1
            
            # 返回默认结果
            return self._create_default_analysis_result()
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """预处理图像"""
        # 调整图像尺寸
        max_size = self.config['image_processing']['max_image_size']
        h, w = image.shape[:2]
        
        if h > max_size[1] or w > max_size[0]:
            scale = min(max_size[0]/w, max_size[1]/h)
            new_w, new_h = int(w*scale), int(h*scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 图像增强
        if self.config['image_processing']['auto_enhance']:
            # 自适应直方图均衡化
            if len(image.shape) == 3:
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                lab[:,:,0] = clahe.apply(lab[:,:,0])
                image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                image = clahe.apply(image)
        
        return image
    
    def _encode_image_to_base64(self, image: np.ndarray) -> str:
        """将图像编码为base64"""
        # 编码为JPEG
        quality = self.config['image_processing']['image_quality']
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode('.jpg', image, encode_param)
        
        # 转换为base64
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    
    def _build_analysis_request(self, image_base64: str, 
                              context: Optional[Dict[str, Any]] = None,
                              original_prediction: Optional[str] = None) -> Dict[str, Any]:
        """构建分析请求"""
        # 基础提示词
        system_prompt = self.config['prompts']['system_prompt']
        analysis_prompt = self.config['prompts']['analysis_prompt']
        medical_prompt = self.config['prompts']['medical_prompt']
        
        # 添加上下文信息
        context_info = ""
        if context:
            context_info = f"\n\n上下文信息：\n{json.dumps(context, ensure_ascii=False, indent=2)}"
        
        # 添加原始预测信息
        prediction_info = ""
        if original_prediction:
            prediction_info = f"\n\n原始系统预测：{original_prediction}\n请分析此预测是否准确，并提供更准确的识别结果。"
        
        # 构建完整提示词
        full_prompt = f"{analysis_prompt}\n{medical_prompt}{context_info}{prediction_info}"
        
        return {
            'system_prompt': system_prompt,
            'user_prompt': full_prompt,
            'image_base64': image_base64,
            'max_tokens': 2000,
            'temperature': 0.1
        }
    
    def _call_llm_api(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """调用大模型API"""
        primary_provider = self.config['llm']['primary_provider']
        fallback_providers = self.config['llm']['fallback_providers']
        
        # 尝试主要提供商
        providers_to_try = [primary_provider] + fallback_providers
        
        for provider in providers_to_try:
            if provider not in self.llm_clients:
                continue
                
            try:
                response = self._call_specific_llm(provider, request)
                if response:
                    return response
            except Exception as e:
                self.logger.warning(f"调用{provider.value}失败: {e}")
                continue
        
        raise Exception("所有大模型API调用失败")
    
    def _call_specific_llm(self, provider: LLMProvider, 
                          request: Dict[str, Any]) -> Dict[str, Any]:
        """调用特定的大模型"""
        client = self.llm_clients[provider]
        
        if provider == LLMProvider.OPENAI_GPT4V:
            return self._call_openai_api(client, request)
        elif provider == LLMProvider.CLAUDE_VISION:
            return self._call_claude_api(client, request)
        elif provider == LLMProvider.GEMINI_PRO_VISION:
            return self._call_gemini_api(client, request)
        elif provider == LLMProvider.QWEN_VL:
            return self._call_qwen_api(client, request)
        elif provider == LLMProvider.BAIDU_ERNIE_VL:
            return self._call_baidu_api(client, request)
        elif provider == LLMProvider.ZHIPU_GLM4V:
            return self._call_zhipu_api(client, request)
        else:
            raise Exception(f"不支持的大模型提供商: {provider}")
    
    def _call_openai_api(self, client: Dict[str, Any], 
                        request: Dict[str, Any]) -> Dict[str, Any]:
        """调用OpenAI API"""
        headers = {
            'Authorization': f"Bearer {client['api_key']}",
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': client['model'],
            'messages': [
                {
                    'role': 'system',
                    'content': request['system_prompt']
                },
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': request['user_prompt']
                        },
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': f"data:image/jpeg;base64,{request['image_base64']}"
                            }
                        }
                    ]
                }
            ],
            'max_tokens': request['max_tokens'],
            'temperature': request['temperature']
        }
        
        response = requests.post(
            f"{client['base_url']}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.config['llm']['timeout']
        )
        
        response.raise_for_status()
        return response.json()
    
    def _call_claude_api(self, client: Dict[str, Any], 
                        request: Dict[str, Any]) -> Dict[str, Any]:
        """调用Claude API"""
        headers = {
            'x-api-key': client['api_key'],
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01'
        }
        
        payload = {
            'model': client['model'],
            'max_tokens': request['max_tokens'],
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image',
                            'source': {
                                'type': 'base64',
                                'media_type': 'image/jpeg',
                                'data': request['image_base64']
                            }
                        },
                        {
                            'type': 'text',
                            'text': request['user_prompt']
                        }
                    ]
                }
            ]
        }
        
        response = requests.post(
            f"{client['base_url']}/messages",
            headers=headers,
            json=payload,
            timeout=self.config['llm']['timeout']
        )
        
        response.raise_for_status()
        return response.json()
    
    def _call_qwen_api(self, client: Dict[str, Any], 
                      request: Dict[str, Any]) -> Dict[str, Any]:
        """调用通义千问API"""
        headers = {
            'Authorization': f"Bearer {client['api_key']}",
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': client['model'],
            'input': {
                'messages': [
                    {
                        'role': 'user',
                        'content': [
                            {
                                'image': f"data:image/jpeg;base64,{request['image_base64']}"
                            },
                            {
                                'text': request['user_prompt']
                            }
                        ]
                    }
                ]
            },
            'parameters': {
                'max_tokens': request['max_tokens'],
                'temperature': request['temperature']
            }
        }
        
        response = requests.post(
            f"{client['base_url']}/services/aigc/multimodal-generation/generation",
            headers=headers,
            json=payload,
            timeout=self.config['llm']['timeout']
        )
        
        response.raise_for_status()
        return response.json()
    
    def _call_baidu_api(self, client: Dict[str, Any], 
                       request: Dict[str, Any]) -> Dict[str, Any]:
        """调用百度文心API"""
        # 百度API需要先获取access_token
        access_token = self._get_baidu_access_token(client['api_key'])
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        payload = {
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': request['user_prompt']
                        },
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': f"data:image/jpeg;base64,{request['image_base64']}"
                            }
                        }
                    ]
                }
            ],
            'max_output_tokens': request['max_tokens'],
            'temperature': request['temperature']
        }
        
        response = requests.post(
            f"{client['base_url']}/wenxinworkshop/chat/{client['model']}?access_token={access_token}",
            headers=headers,
            json=payload,
            timeout=self.config['llm']['timeout']
        )
        
        response.raise_for_status()
        return response.json()
    
    def _call_zhipu_api(self, client: Dict[str, Any], 
                       request: Dict[str, Any]) -> Dict[str, Any]:
        """调用智谱GLM API"""
        headers = {
            'Authorization': f"Bearer {client['api_key']}",
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': client['model'],
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': request['user_prompt']
                        },
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': f"data:image/jpeg;base64,{request['image_base64']}"
                            }
                        }
                    ]
                }
            ],
            'max_tokens': request['max_tokens'],
            'temperature': request['temperature']
        }
        
        response = requests.post(
            f"{client['base_url']}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.config['llm']['timeout']
        )
        
        response.raise_for_status()
        return response.json()
    
    def _get_baidu_access_token(self, api_key: str) -> str:
        """获取百度API访问令牌"""
        # 这里需要实现百度OAuth2.0流程
        # 简化实现，实际使用时需要完整的OAuth流程
        return "mock_access_token"
    
    def _parse_llm_response(self, response: Dict[str, Any]) -> LLMAnalysisResult:
        """解析大模型响应"""
        try:
            # 提取响应内容
            content = self._extract_response_content(response)
            
            # 尝试解析JSON
            if content.startswith('{') and content.endswith('}'):
                parsed_data = json.loads(content)
            else:
                # 如果不是JSON格式，使用文本解析
                parsed_data = self._parse_text_response(content)
            
            # 构建分析结果
            return LLMAnalysisResult(
                scene_description=parsed_data.get('scene_description', content[:200]),
                detected_objects=parsed_data.get('detected_objects', []),
                scene_category=SceneCategory(parsed_data.get('scene_category', 'unknown')),
                confidence=parsed_data.get('confidence', 0.8),
                suggested_actions=parsed_data.get('suggested_actions', []),
                learning_keywords=parsed_data.get('learning_keywords', []),
                safety_assessment=parsed_data.get('safety_assessment', {}),
                medical_relevance=parsed_data.get('medical_relevance', {}),
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"解析大模型响应失败: {e}")
            return self._create_default_analysis_result()
    
    def _extract_response_content(self, response: Dict[str, Any]) -> str:
        """提取响应内容"""
        # OpenAI格式
        if 'choices' in response:
            return response['choices'][0]['message']['content']
        
        # Claude格式
        if 'content' in response:
            return response['content'][0]['text']
        
        # 通义千问格式
        if 'output' in response:
            return response['output']['text']
        
        # 百度格式
        if 'result' in response:
            return response['result']
        
        # 默认格式
        return str(response)
    
    def _parse_text_response(self, content: str) -> Dict[str, Any]:
        """解析文本响应"""
        # 简单的文本解析逻辑
        parsed_data = {
            'scene_description': content,
            'detected_objects': [],
            'scene_category': 'unknown',
            'confidence': 0.7,
            'suggested_actions': [],
            'learning_keywords': [],
            'safety_assessment': {},
            'medical_relevance': {}
        }
        
        # 提取关键信息
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if '场景描述' in line or 'scene description' in line.lower():
                parsed_data['scene_description'] = line.split(':', 1)[-1].strip()
            elif '对象' in line or 'object' in line.lower():
                # 简单的对象提取
                objects = line.split(':', 1)[-1].strip().split(',')
                parsed_data['detected_objects'] = [{'name': obj.strip()} for obj in objects]
        
        return parsed_data
    
    def _create_default_analysis_result(self) -> LLMAnalysisResult:
        """创建默认分析结果"""
        return LLMAnalysisResult(
            scene_description="无法识别的场景",
            detected_objects=[],
            scene_category=SceneCategory.UNKNOWN,
            confidence=0.0,
            suggested_actions=["需要人工确认"],
            learning_keywords=[],
            safety_assessment={'status': 'unknown'},
            medical_relevance={'relevance': 'unknown'},
            timestamp=time.time()
        )
    
    def should_trigger_self_learning(self, prediction_confidence: float,
                                   prediction_result: Optional[str] = None) -> bool:
        """判断是否应该触发自学习"""
        if not self.config['llm']['enabled']:
            return False
        
        if not self.config['self_learning']['auto_learning']:
            return False
        
        # 置信度低于阈值
        threshold = self.config['self_learning']['confidence_threshold']
        if prediction_confidence < threshold:
            return True
        
        # 预测结果为未知
        if prediction_result and 'unknown' in prediction_result.lower():
            return True
        
        return False
    
    def learn_from_analysis(self, image: np.ndarray, 
                          analysis_result: LLMAnalysisResult,
                          original_prediction: Optional[str] = None,
                          user_feedback: Optional[Dict[str, Any]] = None) -> bool:
        """从分析结果中学习"""
        try:
            # 生成图像ID
            image_id = self._generate_image_id(image)
            
            # 保存学习图像
            self._save_learning_image(image, image_id)
            
            # 创建学习记录
            learning_record = SelfLearningRecord(
                image_id=image_id,
                original_prediction=original_prediction,
                llm_analysis=analysis_result,
                user_feedback=user_feedback,
                learning_outcome="pending",
                created_at=time.time(),
                updated_at=time.time()
            )
            
            # 添加到学习记录
            self.learning_records.append(learning_record)
            
            # 更新知识库
            self._update_knowledge_base(analysis_result)
            
            # 更新统计
            self.stats['learning_records'] += 1
            
            # 保存学习记录
            self._save_learning_records()
            
            return True
            
        except Exception as e:
            self.logger.error(f"学习过程失败: {e}")
            return False
    
    def _generate_image_id(self, image: np.ndarray) -> str:
        """生成图像ID"""
        # 使用图像内容和时间戳生成唯一ID
        import hashlib
        image_hash = hashlib.md5(image.tobytes()).hexdigest()
        timestamp = str(int(time.time()))
        return f"img_{timestamp}_{image_hash[:8]}"
    
    def _save_learning_image(self, image: np.ndarray, image_id: str):
        """保存学习图像"""
        images_path = Path(self.config['storage']['images_path'])
        images_path.mkdir(parents=True, exist_ok=True)
        
        image_file = images_path / f"{image_id}.jpg"
        cv2.imwrite(str(image_file), image)
    
    def _update_knowledge_base(self, analysis_result: LLMAnalysisResult):
        """更新知识库"""
        # 添加场景描述
        scene_key = f"scene_{analysis_result.scene_category.value}"
        if scene_key not in self.knowledge_base:
            self.knowledge_base[scene_key] = {
                'descriptions': [],
                'objects': [],
                'keywords': [],
                'count': 0
            }
        
        self.knowledge_base[scene_key]['descriptions'].append(analysis_result.scene_description)
        self.knowledge_base[scene_key]['objects'].extend(analysis_result.detected_objects)
        self.knowledge_base[scene_key]['keywords'].extend(analysis_result.learning_keywords)
        self.knowledge_base[scene_key]['count'] += 1
        
        # 添加对象信息
        for obj in analysis_result.detected_objects:
            obj_name = obj.get('name', 'unknown')
            obj_key = f"object_{obj_name.lower().replace(' ', '_')}"
            
            if obj_key not in self.knowledge_base:
                self.knowledge_base[obj_key] = {
                    'attributes': [],
                    'contexts': [],
                    'count': 0
                }
            
            self.knowledge_base[obj_key]['attributes'].append(obj)
            self.knowledge_base[obj_key]['contexts'].append(analysis_result.scene_description)
            self.knowledge_base[obj_key]['count'] += 1
        
        # 更新统计
        self.stats['knowledge_entries'] = len(self.knowledge_base)
        
        # 保存知识库
        self._save_knowledge_base()
    
    def _load_knowledge_base(self):
        """加载知识库"""
        try:
            kb_path = Path(self.config['storage']['knowledge_base_path'])
            if kb_path.exists():
                with open(kb_path, 'r', encoding='utf-8') as f:
                    self.knowledge_base = json.load(f)
                self.stats['knowledge_entries'] = len(self.knowledge_base)
        except Exception as e:
            self.logger.error(f"加载知识库失败: {e}")
            self.knowledge_base = {}
    
    def _save_knowledge_base(self):
        """保存知识库"""
        try:
            kb_path = Path(self.config['storage']['knowledge_base_path'])
            kb_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(kb_path, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"保存知识库失败: {e}")
    
    def _save_learning_records(self):
        """保存学习记录"""
        try:
            records_path = Path(self.config['storage']['learning_records_path'])
            records_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 转换为可序列化格式
            serializable_records = []
            for record in self.learning_records:
                serializable_records.append({
                    'image_id': record.image_id,
                    'original_prediction': record.original_prediction,
                    'llm_analysis': {
                        'scene_description': record.llm_analysis.scene_description,
                        'detected_objects': record.llm_analysis.detected_objects,
                        'scene_category': record.llm_analysis.scene_category.value,
                        'confidence': record.llm_analysis.confidence,
                        'suggested_actions': record.llm_analysis.suggested_actions,
                        'learning_keywords': record.llm_analysis.learning_keywords,
                        'safety_assessment': record.llm_analysis.safety_assessment,
                        'medical_relevance': record.llm_analysis.medical_relevance,
                        'timestamp': record.llm_analysis.timestamp
                    },
                    'user_feedback': record.user_feedback,
                    'learning_outcome': record.learning_outcome,
                    'created_at': record.created_at,
                    'updated_at': record.updated_at
                })
            
            with open(records_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_records, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"保存学习记录失败: {e}")
    
    def query_knowledge_base(self, query: str) -> List[Dict[str, Any]]:
        """查询知识库"""
        results = []
        query_lower = query.lower()
        
        for key, value in self.knowledge_base.items():
            # 检查键名匹配
            if query_lower in key.lower():
                results.append({
                    'key': key,
                    'data': value,
                    'relevance': 1.0
                })
                continue
            
            # 检查描述匹配
            if 'descriptions' in value:
                for desc in value['descriptions']:
                    if query_lower in desc.lower():
                        results.append({
                            'key': key,
                            'data': value,
                            'relevance': 0.8
                        })
                        break
            
            # 检查关键词匹配
            if 'keywords' in value:
                for keyword in value['keywords']:
                    if query_lower in keyword.lower():
                        results.append({
                            'key': key,
                            'data': value,
                            'relevance': 0.6
                        })
                        break
        
        # 按相关性排序
        results.sort(key=lambda x: x['relevance'], reverse=True)
        return results
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """获取学习统计信息"""
        return {
            'total_queries': self.stats['total_queries'],
            'successful_queries': self.stats['successful_queries'],
            'failed_queries': self.stats['failed_queries'],
            'success_rate': self.stats['successful_queries'] / max(self.stats['total_queries'], 1),
            'learning_records': self.stats['learning_records'],
            'knowledge_entries': self.stats['knowledge_entries'],
            'enabled_providers': list(self.llm_clients.keys()),
            'knowledge_base_size': len(self.knowledge_base)
        }
    
    def update_configuration(self, new_config: Dict[str, Any]):
        """更新配置"""
        # 深度合并配置
        def deep_merge(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_merge(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_merge(self.config, new_config)
        
        # 重新初始化客户端
        self.llm_clients = self._initialize_llm_clients()


# 使用示例
if __name__ == "__main__":
    # 创建大模型自学习系统
    llm_system = LLMSelfLearningSystem()
    
    # 模拟未知场景图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 检查是否需要自学习
    should_learn = llm_system.should_trigger_self_learning(
        prediction_confidence=0.2,
        prediction_result="unknown_object"
    )
    
    if should_learn:
        print("触发自学习模式...")
        
        # 分析未知场景
        analysis_result = llm_system.analyze_unknown_scene(
            test_image,
            context={"location": "医院病房", "time": "下午2点"},
            original_prediction="unknown_object"
        )
        
        print(f"场景描述: {analysis_result.scene_description}")
        print(f"场景类别: {analysis_result.scene_category.value}")
        print(f"检测对象: {[obj.get('name') for obj in analysis_result.detected_objects]}")
        print(f"建议行动: {analysis_result.suggested_actions}")
        
        # 学习新知识
        learning_success = llm_system.learn_from_analysis(
            test_image,
            analysis_result,
            original_prediction="unknown_object"
        )
        
        print(f"学习结果: {'成功' if learning_success else '失败'}")
        
        # 获取学习统计
        stats = llm_system.get_learning_statistics()
        print(f"学习统计: {stats}")
    
    print("大模型自学习系统测试完成")