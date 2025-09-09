# YOLOS ModelScope大模型集成指南

## 概述

本文档介绍如何在YOLOS项目中集成ModelScope的Qwen2.5-VL视觉大模型，实现智能图像识别和分析功能。

## 功能特性

### 核心功能
- **多模态视觉理解**: 支持图像+文本的多模态分析
- **智能场景识别**: 自动识别图像中的场景、对象和活动
- **医疗健康分析**: 专门针对医疗场景的智能分析
- **安全监控**: 跌倒检测、异常行为识别
- **自学习能力**: 从分析结果中持续学习和改进

### 技术特性
- **高性能处理**: 支持批量处理和并发请求
- **智能缓存**: 避免重复分析相同图像
- **配额管理**: 自动管理API调用配额
- **实时监控**: 性能指标和健康状态监控
- **数据持久化**: 分析结果和学习数据的持久化存储

## 快速开始

### 1. 安装依赖

```bash
# 安装基础依赖
pip install -r requirements_modelscope.txt

# 或者单独安装核心依赖
pip install openai opencv-python numpy pyyaml psutil
```

### 2. 配置服务

编辑 `config/modelscope_llm_config.yaml` 文件：

```yaml
modelscope_api:
  api:
    base_url: "https://api-inference.modelscope.cn/v1"
    api_key: "your-api-key-here"  # 替换为你的API密钥
  models:
    primary_vision_model:
      name: "Qwen/Qwen2.5-VL-72B-Instruct"
      enabled: true
```

### 3. 基础使用

```python
from src.recognition.modelscope_llm_service import ModelScopeLLMService

# 创建服务实例
service = ModelScopeLLMService()

# 启动服务
service.start_service()

# 分析单张图像
result = service.analyze_image("path/to/image.jpg", "general")

if result:
    print(f"场景描述: {result.scene_description}")
    print(f"检测对象: {result.detected_objects}")
    print(f"安全评估: {result.safety_assessment}")

# 停止服务
service.stop_service()
```

### 4. 便捷函数

```python
from src.recognition.modelscope_llm_service import analyze_image_with_modelscope

# 直接分析图像
result = analyze_image_with_modelscope("image.jpg", "medical")
```

## 详细配置

### API配置

```yaml
modelscope_api:
  api:
    base_url: "https://api-inference.modelscope.cn/v1"
    api_key: "ms-your-token-here"
    timeout: 60
    max_retries: 3
    retry_delay: 2.0
```

### 模型配置

```yaml
models:
  primary_vision_model:
    name: "Qwen/Qwen2.5-VL-72B-Instruct"
    enabled: true
    max_tokens: 2048
    temperature: 0.1
    
  fallback_vision_model:
    name: "Qwen/Qwen2-VL-7B-Instruct"
    enabled: true
    max_tokens: 1024
```

### 图像处理配置

```yaml
image_processing:
  preprocessing:
    enabled: true
    resize_enabled: true
    max_width: 1024
    max_height: 1024
    quality_enhancement: true
    noise_reduction: true
```

### 性能配置

```yaml
performance:
  max_concurrent_requests: 4
  request_queue_size: 100
  worker_threads: 8
  
  batch_processing:
    enabled: true
    batch_size: 8
    batch_timeout: 10
```

## 使用场景

### 1. 医疗健康分析

```python
# 分析医疗相关图像
result = service.analyze_image("medical_image.jpg", "medical")

# 检查医疗相关性
if result.medical_relevance['medical_content']:
    urgency = result.medical_relevance['urgency_level']
    advice = result.medical_relevance['medical_advice']
    print(f"紧急程度: {urgency}")
    print(f"医疗建议: {advice}")
```

### 2. 安全监控

```python
# 安全监控分析
result = service.analyze_image("security_camera.jpg", "security")

# 检查安全状态
safety_level = result.safety_assessment['safety_level']
if safety_level == 'danger':
    risk_factors = result.safety_assessment['risk_factors']
    print(f"发现危险: {risk_factors}")
```

### 3. 批量处理

```python
# 批量分析训练图像
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = service.analyze_image_batch(image_paths, "learning")

for i, result in enumerate(results):
    if result:
        print(f"图像 {i+1}: {result.scene_description}")
```

### 4. 自学习模式

```python
# 启用学习模式
result = service.analyze_image("learning_sample.jpg", "learning")

# 获取学习洞察
insights = result.learning_insights
key_features = insights['key_features']
improvements = insights['improvement_suggestions']
```

## 提示词定制

### 系统提示词

```yaml
prompts:
  system_prompt: |
    你是YOLOS智能识别系统的视觉分析专家。
    请准确分析图像内容，特别关注：
    1. 医疗健康相关内容
    2. 安全监控场景
    3. 日常生活物品识别
    4. 人员和动物识别
    5. 环境安全评估
```

### 专用提示词

```yaml
  medical_prompt: |
    这是医疗健康相关的图像分析任务。请特别关注：
    1. 药物识别：药品名称、剂量、有效期
    2. 医疗器械：类型、用途、使用状态
    3. 健康状态：面部表情、身体姿态、异常症状
    4. 紧急情况：跌倒、意外伤害、急救需求
```

## 监控和维护

### 服务状态监控

```python
# 获取服务状态
status = service.get_service_status()
print(f"运行状态: {status['is_running']}")
print(f"API可用: {status['api_available']}")
print(f"队列大小: {status['queue_size']}")
print(f"统计信息: {status['stats']}")
```

### 性能指标

```python
# 获取分析历史
history = service.get_analysis_history(limit=50)
for record in history:
    print(f"图像: {record['image_path']}")
    print(f"处理时间: {record['processing_time']:.2f}s")
    print(f"模型: {record['model_used']}")
```

### 配额管理

服务自动管理API配额，包括：
- 每日请求限制
- 每小时请求限制
- 并发请求限制
- 自动告警和降级

## 错误处理

### 常见错误

1. **API连接失败**
   ```python
   # 检查网络连接和API密钥
   if not service._check_api_connectivity():
       print("API连接失败，请检查配置")
   ```

2. **图像读取失败**
   ```python
   # 确保图像文件存在且格式正确
   import cv2
   image = cv2.imread("image.jpg")
   if image is None:
       print("无法读取图像文件")
   ```

3. **配额超限**
   ```python
   # 检查配额状态
   status = service.get_service_status()
   quota = status['quota_stats']
   print(f"今日已用: {quota['daily_requests']}")
   ```

### 日志配置

```yaml
logging:
  level: "INFO"
  file_logging:
    enabled: true
    log_file: "logs/modelscope_llm_system.log"
  categories:
    api_requests: true
    image_processing: true
    recognition_results: true
    error_tracking: true
```

## 最佳实践

### 1. 性能优化

- 启用图像缓存避免重复分析
- 使用批处理提高吞吐量
- 合理设置并发数量
- 定期清理缓存和日志

### 2. 安全考虑

- 保护API密钥安全
- 限制访问权限
- 定期备份分析数据
- 监控异常访问

### 3. 成本控制

- 设置合理的配额限制
- 使用缓存减少API调用
- 选择合适的模型规格
- 监控使用量和成本

## 扩展开发

### 自定义分析任务

```python
class CustomAnalysisTask:
    def __init__(self, service):
        self.service = service
    
    def analyze_custom_scene(self, image_path):
        # 自定义提示词
        custom_prompt = "请分析这个特定场景..."
        
        # 调用服务
        result = self.service.analyze_image(image_path, "custom")
        
        # 自定义后处理
        return self.post_process_result(result)
```

### 集成其他服务

```python
# 与YOLOS核心系统集成
from src.core.detection_engine import DetectionEngine

class IntegratedAnalysis:
    def __init__(self):
        self.llm_service = ModelScopeLLMService()
        self.detection_engine = DetectionEngine()
    
    def comprehensive_analysis(self, image_path):
        # 传统检测
        detection_result = self.detection_engine.detect(image_path)
        
        # LLM分析
        llm_result = self.llm_service.analyze_image(image_path)
        
        # 融合结果
        return self.merge_results(detection_result, llm_result)
```

## 故障排除

### 常见问题

1. **服务启动失败**
   - 检查配置文件格式
   - 验证API密钥有效性
   - 确认网络连接正常

2. **分析结果异常**
   - 检查图像质量和格式
   - 验证提示词设置
   - 查看详细日志信息

3. **性能问题**
   - 调整并发设置
   - 优化图像预处理
   - 检查系统资源使用

### 调试模式

```yaml
development:
  debug_mode: true
  verbose_logging: true
  test_mode: false
```

## 更新和维护

### 版本更新

1. 备份当前配置和数据
2. 更新代码和依赖
3. 测试新功能
4. 逐步部署到生产环境

### 数据维护

- 定期清理过期缓存
- 备份重要分析结果
- 监控数据库大小
- 优化查询性能

## 支持和反馈

如有问题或建议，请：

1. 查看日志文件获取详细错误信息
2. 检查配置文件设置
3. 参考本文档的故障排除部分
4. 联系技术支持团队

---

*本文档持续更新中，请关注最新版本。*