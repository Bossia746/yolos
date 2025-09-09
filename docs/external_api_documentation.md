# YOLOS 外部API系统完整文档

## 概述

YOLOS外部API系统是一个功能强大的RESTful API服务，专为AIoT设备的语音控制和智能识别任务而设计。系统支持通过语音指令控制设备移动、摄像头操作，并执行各种专项识别任务，如药物检测、宠物监控、跌倒检测等。

## 🎯 核心特性

### 1. 语音控制能力
- **实时语音识别**: 支持中文语音命令识别
- **智能命令解析**: 自动解析设备控制和识别任务指令
- **语音反馈**: TTS语音播报执行结果
- **异步语音监听**: WebSocket实时语音交互

### 2. 设备控制功能
- **精确移动控制**: 支持坐标和预定义位置移动
- **摄像头控制**: 360度旋转、缩放、对焦调节
- **安全区域管理**: 预定义安全区域和禁止区域
- **电源管理**: 电池监控和自动充电

### 3. 专项识别任务
- **药物检测**: OCR识别、有效期检查、剂量分析
- **宠物监控**: 物种识别、行为分析、健康监控
- **跌倒检测**: 实时姿态分析、紧急报警
- **安全监控**: 入侵检测、危险物品识别
- **医疗分析**: 面部健康评估、症状检测
- **手势识别**: 手势命令解释、智能交互

### 4. 集成能力
- **RESTful API**: 标准HTTP接口，易于集成
- **WebSocket支持**: 实时双向通信
- **客户端SDK**: Python SDK简化开发
- **多平台支持**: 支持各种AIoT开发板

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    第三方应用系统                              │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP/WebSocket
┌─────────────────────▼───────────────────────────────────────┐
│                 YOLOS 外部API系统                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │  语音处理   │ │  设备控制   │ │  任务管理   │           │
│  │    模块     │ │    模块     │ │    模块     │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────┬───────────────────────────────────────┘
                      │ 内部调用
┌─────────────────────▼───────────────────────────────────────┐
│                YOLOS 核心识别系统                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ 多目标识别  │ │ 优先级处理  │ │ 自学习系统  │           │
│  │    系统     │ │    系统     │ │             │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────┬───────────────────────────────────────┘
                      │ 硬件控制
┌─────────────────────▼───────────────────────────────────────┐
│                   AIoT设备层                                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │   摄像头    │ │  移动平台   │ │  传感器     │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

## 📋 API接口文档

### 基础接口

#### 健康检查
```http
GET /api/health
```

**响应示例:**
```json
{
  "success": true,
  "message": "API服务正常运行",
  "data": {
    "version": "2.0.0",
    "status": "healthy",
    "uptime": 3600
  },
  "timestamp": "2025-01-09T08:00:00Z"
}
```

#### 获取设备状态
```http
GET /api/device/status
```

**响应示例:**
```json
{
  "success": true,
  "message": "设备状态获取成功",
  "data": {
    "position": {"x": 0, "y": 0, "z": 0},
    "camera_angle": {"pan": 0, "tilt": 0},
    "zoom_level": 1.0,
    "recording": false,
    "online": true,
    "battery_level": 85
  }
}
```

### 设备控制接口

#### 移动设备
```http
POST /api/device/move
```

**请求参数:**
```json
{
  "position": {
    "x": 5.0,
    "y": 3.0,
    "z": 0.0
  }
}
```

**响应示例:**
```json
{
  "success": true,
  "message": "设备移动成功",
  "data": {
    "new_position": {"x": 5.0, "y": 3.0, "z": 0.0},
    "move_distance": 5.83,
    "move_time": 5.83
  }
}
```

#### 设备控制
```http
POST /api/device/control
```

**请求参数 (摄像头旋转):**
```json
{
  "command": "rotate_camera",
  "parameters": {
    "pan": 45,
    "tilt": -15
  }
}
```

**请求参数 (摄像头缩放):**
```json
{
  "command": "zoom",
  "parameters": {
    "level": 2.5
  }
}
```

**请求参数 (拍照):**
```json
{
  "command": "take_photo",
  "parameters": {}
}
```

### 语音控制接口

#### 监听语音命令
```http
POST /api/voice/listen
```

**请求参数:**
```json
{
  "timeout": 10.0
}
```

**响应示例:**
```json
{
  "success": true,
  "message": "语音命令处理成功",
  "data": {
    "command": {
      "command_text": "移动到客厅",
      "confidence": 0.92,
      "timestamp": "2025-01-09T08:00:00Z",
      "task_type": null,
      "device_command": "move_to_position",
      "target_location": {"x": 0, "y": 0, "z": 0}
    },
    "result": {
      "command_processed": true,
      "actions": [
        {
          "type": "device_move",
          "result": {
            "success": true,
            "message": "设备移动成功"
          }
        }
      ]
    }
  }
}
```

### 识别任务接口

#### 启动识别任务
```http
POST /api/recognition/start
```

**请求参数:**
```json
{
  "task_type": "medication_detection",
  "parameters": {
    "confidence_threshold": 0.8,
    "enable_ocr": true,
    "check_expiry_date": true
  },
  "priority": 8
}
```

**响应示例:**
```json
{
  "success": true,
  "message": "识别任务已创建",
  "data": {
    "task_id": "task_1704787200000",
    "task": {
      "task_id": "task_1704787200000",
      "task_type": "medication_detection",
      "priority": 8,
      "status": "pending",
      "created_at": "2025-01-09T08:00:00Z"
    }
  }
}
```

#### 获取任务状态
```http
GET /api/recognition/status/{task_id}
```

**响应示例:**
```json
{
  "success": true,
  "message": "任务已完成",
  "data": {
    "task_id": "task_1704787200000",
    "status": "completed",
    "result": {
      "detected_medications": [
        {
          "name": "阿司匹林",
          "confidence": 0.92,
          "dosage": "100mg",
          "expiry_date": "2025-12-31"
        }
      ],
      "total_count": 1,
      "processing_time": 2.5
    }
  }
}
```

#### 图像识别
```http
POST /api/recognition/image
```

**请求参数:**
```json
{
  "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
  "task_type": "pet_monitoring",
  "confidence_threshold": 0.7
}
```

**响应示例:**
```json
{
  "success": true,
  "message": "图像识别完成",
  "data": {
    "image_info": {
      "width": 1920,
      "height": 1080,
      "channels": 3
    },
    "task_type": "pet_monitoring",
    "detected_objects": [
      {
        "category": "pet",
        "species": "cat",
        "breed": "橘猫",
        "confidence": 0.89,
        "bbox": [100, 100, 200, 300],
        "activity": "sleeping",
        "health_status": "normal"
      }
    ],
    "processing_time": 1.2,
    "timestamp": "2025-01-09T08:00:00Z"
  }
}
```

## 🔌 WebSocket事件

### 连接事件
```javascript
// 连接成功
socket.on('connected', (data) => {
  console.log('WebSocket连接成功:', data.message);
});

// 连接断开
socket.on('disconnect', () => {
  console.log('WebSocket连接断开');
});
```

### 语音控制事件
```javascript
// 开始语音监听
socket.emit('start_voice_listening', {timeout: 10.0});

// 语音命令接收
socket.on('voice_command_received', (data) => {
  console.log('语音命令:', data.command.command_text);
  console.log('执行结果:', data.result);
});

// 语音监听超时
socket.on('voice_command_timeout', (data) => {
  console.log('语音监听超时:', data.message);
});
```

### 设备控制事件
```javascript
// 设备控制
socket.emit('device_control', {
  command: 'rotate_camera',
  parameters: {pan: 45, tilt: -15}
});

// 设备控制结果
socket.on('device_control_result', (data) => {
  console.log('设备控制结果:', data);
});

// 设备位置更新
socket.on('device_position_updated', (data) => {
  console.log('设备新位置:', data.position);
});
```

### 任务管理事件
```javascript
// 任务完成通知
socket.on('task_completed', (data) => {
  console.log('任务完成:', data.task_id);
  console.log('任务结果:', data.result);
});
```

## 🛠️ 客户端SDK使用

### 安装依赖
```bash
pip install requests python-socketio opencv-python pillow numpy
```

### 基础使用
```python
from src.sdk.yolos_client_sdk import create_client

# 创建客户端
client = create_client("http://localhost:8080")

# 健康检查
health = client.health_check()
print(f"服务状态: {health}")

# 设备控制
client.move_device_to_location("客厅")
client.rotate_camera(pan=45, tilt=-15)
client.zoom_camera(2.0)

# 拍照
photo_result = client.take_photo()
print(f"拍照结果: {photo_result}")

# 语音控制
voice_result = client.listen_voice_command(timeout=10.0)
if voice_result.success:
    print(f"语音命令: {voice_result.command_text}")

# 识别任务
result = client.detect_medication("path/to/image.jpg")
if result.success:
    print(f"检测到药物: {result.detected_objects}")

# 关闭客户端
client.close()
```

### 语音工作流示例
```python
from src.sdk.yolos_client_sdk import create_client

client = create_client("http://localhost:8080", enable_websocket=True)

try:
    # 连接WebSocket
    client.connect_websocket()
    
    while True:
        print("请说出指令...")
        
        # 监听语音命令
        voice_result = client.listen_voice_command(timeout=15.0)
        
        if voice_result.success:
            command = voice_result.command_text
            print(f"收到指令: {command}")
            
            # 执行语音命令
            result = client.execute_voice_command(command)
            print(f"执行结果: {result}")
            
            # 如果是识别任务，等待结果
            if result.get('task_id'):
                task_result = client.wait_for_task_completion(
                    result['task_id'], timeout=30.0
                )
                print(f"识别结果: {task_result}")
        
        else:
            print("语音识别失败，请重试")

except KeyboardInterrupt:
    print("程序结束")
finally:
    client.close()
```

## 🎯 应用场景示例

### 1. 智慧医疗场景

#### 药物管理助手
```python
# 语音指令: "去药箱检查药物"
client = create_client()

# 1. 移动到药箱位置
client.execute_voice_command("移动到厨房")

# 2. 调整摄像头角度
client.rotate_camera(pan=0, tilt=-30)
client.zoom_camera(2.5)

# 3. 拍照并识别药物
photo_result = client.take_photo()
medication_result = client.detect_medication("current_frame")

# 4. 分析结果
if medication_result.success:
    for med in medication_result.detected_objects:
        print(f"药物: {med['name']}")
        print(f"剂量: {med['dosage']}")
        print(f"有效期: {med['expiry_date']}")
```

#### 老人跌倒监护
```python
# 语音指令: "开始老人监护"
client = create_client()

# 启动跌倒检测任务
task_id = client.start_recognition_task(
    task_type="fall_detection",
    parameters={
        "real_time_monitoring": True,
        "emergency_alert": True,
        "sensitivity": "high"
    },
    priority=9
)

# 监控结果处理
def handle_fall_detection(result):
    if result.get('fall_detected'):
        print("⚠️ 检测到跌倒!")
        
        # 紧急响应
        client.take_photo()  # 拍摄现场
        client.start_recording()  # 开始录像
        
        # 医疗分析
        medical_task = client.start_recognition_task("medical_analysis")
        # 触发外部报警...

# 等待任务完成
result = client.wait_for_task_completion(task_id)
handle_fall_detection(result)
```

### 2. 智能家居场景

#### 宠物监护助手
```python
# 语音指令: "找找我的猫咪"
client = create_client()

locations = ["客厅", "卧室", "阳台"]

for location in locations:
    print(f"搜索位置: {location}")
    
    # 移动到位置
    client.move_device_to_location(location)
    
    # 360度搜索
    for angle in [0, 90, 180, 270]:
        client.rotate_camera(pan=angle, tilt=0)
        
        # 宠物检测
        result = client.monitor_pet("current_frame")
        
        if result.success and result.detected_objects:
            pets = [obj for obj in result.detected_objects 
                   if 'pet' in obj.get('category', '')]
            
            if pets:
                print(f"在{location}发现宠物!")
                for pet in pets:
                    print(f"物种: {pet['species']}")
                    print(f"活动: {pet['activity']}")
                
                # 开始持续监控
                monitoring_task = client.start_recognition_task(
                    "pet_monitoring",
                    parameters={"continuous_monitoring": True}
                )
                break
```

#### 手势控制系统
```python
# 语音指令: "启动手势控制"
client = create_client()

# 启动手势识别
task_id = client.start_recognition_task(
    task_type="gesture_recognition",
    parameters={
        "real_time_recognition": True,
        "gesture_commands": {
            "wave": "greeting",
            "point": "selection",
            "thumbs_up": "confirmation",
            "stop_hand": "halt"
        }
    }
)

# 手势命令处理
def handle_gesture_command(result):
    gestures = result.get('detected_gestures', [])
    
    for gesture in gestures:
        command = gesture.get('command')
        confidence = gesture.get('confidence', 0)
        
        if confidence > 0.8:
            if command == "greeting":
                print("检测到问候手势")
                # 执行问候响应
            elif command == "selection":
                print("检测到指向手势")
                # 执行选择操作
            elif command == "confirmation":
                print("检测到确认手势")
                # 执行确认操作
            elif command == "halt":
                print("检测到停止手势")
                # 执行停止操作

# 处理手势识别结果
result = client.wait_for_task_completion(task_id)
handle_gesture_command(result)
```

### 3. 安防监控场景

#### 智能安防巡逻
```python
# 语音指令: "开始安防巡逻"
client = create_client()

patrol_points = [
    {"name": "大门", "position": {"x": -2, "y": 0, "z": 0}},
    {"name": "窗户", "position": {"x": 0, "y": -2, "z": 0}},
    {"name": "后门", "position": {"x": 5, "y": 5, "z": 0}}
]

for point in patrol_points:
    print(f"巡逻点: {point['name']}")
    
    # 移动到巡逻点
    pos = point['position']
    client.move_device(pos['x'], pos['y'], pos['z'])
    
    # 360度监控
    for angle in range(0, 360, 45):
        client.rotate_camera(pan=angle, tilt=0)
        
        # 安全监控
        result = client.security_surveillance("current_frame")
        
        if result.success:
            alerts = [obj for obj in result.detected_objects 
                     if obj.get('threat_level', 'low') in ['high', 'critical']]
            
            if alerts:
                print(f"⚠️ 在{point['name']}发现安全威胁!")
                
                for alert in alerts:
                    print(f"威胁类型: {alert['category']}")
                    print(f"威胁等级: {alert['threat_level']}")
                
                # 紧急处理
                client.take_photo()  # 拍摄证据
                client.start_recording()  # 开始录像
                # 触发报警系统...
                
                break
```

## ⚙️ 配置说明

### API服务配置
```yaml
api:
  host: "0.0.0.0"
  port: 8080
  debug: false
  max_request_size: "50MB"
  request_timeout: 30
  
  authentication:
    enabled: false
    api_key_header: "X-API-Key"
    valid_api_keys:
      - "yolos_api_key_demo"
```

### 语音处理配置
```yaml
voice:
  enabled: true
  language: "zh-CN"
  timeout: 5.0
  confidence_threshold: 0.7
  
  tts:
    enabled: true
    voice_rate: 150
    voice_volume: 0.8
```

### 设备控制配置
```yaml
device:
  movement:
    max_move_distance: 10.0
    move_speed: 1.0
    position_tolerance: 0.1
    
  camera:
    rotation_range:
      pan_min: -180
      pan_max: 180
      tilt_min: -90
      tilt_max: 90
    
    zoom_range:
      min: 0.5
      max: 5.0
```

### 识别任务配置
```yaml
recognition:
  default_confidence_threshold: 0.6
  max_concurrent_tasks: 5
  task_timeout: 30
  
  task_priorities:
    emergency_detection: 10
    fall_detection: 9
    security_surveillance: 8
    medical_analysis: 7
    medication_detection: 6
```

## 🚀 部署指南

### 1. 环境准备
```bash
# 安装Python依赖
pip install -r requirements.txt

# 安装语音处理依赖
pip install SpeechRecognition pyttsx3

# 安装Web框架依赖
pip install flask flask-cors flask-socketio

# 安装客户端依赖
pip install requests python-socketio
```

### 2. 启动API服务
```bash
# 启动API服务器
python src/api/external_api_system.py --host 0.0.0.0 --port 8080

# 或使用配置文件
python src/api/external_api_system.py --config config/external_api_config.yaml
```

### 3. 测试API服务
```bash
# 健康检查
curl http://localhost:8080/api/health

# 获取设备状态
curl http://localhost:8080/api/device/status

# 移动设备
curl -X POST http://localhost:8080/api/device/move \
  -H "Content-Type: application/json" \
  -d '{"position": {"x": 1, "y": 1, "z": 0}}'
```

### 4. 运行示例程序
```bash
# 运行API使用示例
python examples/api_usage_examples.py

# 运行语音控制演示
python src/sdk/yolos_client_sdk.py
```

## 🔧 故障排除

### 常见问题

#### 1. 语音识别不工作
**问题**: 语音命令无法识别
**解决方案**:
- 检查麦克风权限和设备
- 安装语音识别依赖: `pip install SpeechRecognition pyttsx3`
- 检查网络连接（Google语音识别需要网络）
- 调整语音识别置信度阈值

#### 2. WebSocket连接失败
**问题**: 实时功能不可用
**解决方案**:
- 检查防火墙设置
- 确认WebSocket端口未被占用
- 安装WebSocket依赖: `pip install python-socketio`
- 检查客户端WebSocket配置

#### 3. 设备控制无响应
**问题**: 设备移动或摄像头控制失败
**解决方案**:
- 检查设备连接状态
- 验证移动范围是否在安全区域内
- 检查设备电池电量
- 确认设备权限和驱动程序

#### 4. 识别任务超时
**问题**: 识别任务长时间无响应
**解决方案**:
- 增加任务超时时间
- 检查系统资源使用情况
- 降低图像分辨率
- 启用GPU加速

## 📊 性能优化

### 1. API性能优化
- 启用请求缓存
- 使用连接池
- 配置适当的超时时间
- 启用压缩传输

### 2. 识别性能优化
- 使用GPU加速
- 启用模型量化
- 批量处理图像
- 优化图像预处理

### 3. 网络优化
- 使用WebSocket减少连接开销
- 启用数据压缩
- 配置CDN加速
- 优化图像传输格式

## 📈 监控和日志

### 性能监控
```python
# 获取API性能指标
response = requests.get("http://localhost:8080/api/metrics")
metrics = response.json()

print(f"请求数: {metrics['request_count']}")
print(f"平均响应时间: {metrics['avg_response_time']}ms")
print(f"错误率: {metrics['error_rate']}%")
```

### 日志配置
```yaml
logging:
  level: "INFO"
  file_logging:
    enabled: true
    log_file: "logs/external_api.log"
    max_file_size: "100MB"
    backup_count: 5
  
  categories:
    api_requests: true
    voice_commands: true
    device_control: true
    recognition_tasks: true
```

## 🔐 安全考虑

### 1. API安全
- 启用API密钥认证
- 配置HTTPS加密
- 设置访问控制列表
- 启用请求速率限制

### 2. 数据安全
- 加密敏感数据传输
- 定期清理临时文件
- 限制文件上传大小
- 验证输入数据格式

### 3. 设备安全
- 设置安全移动区域
- 限制设备控制权限
- 监控异常操作
- 启用紧急停止功能

## 📚 扩展开发

### 自定义识别任务
```python
# 添加新的识别任务类型
class CustomRecognitionTask:
    def __init__(self):
        self.task_type = "custom_detection"
    
    def process(self, image, parameters):
        # 自定义识别逻辑
        results = self.custom_detection_algorithm(image)
        return results
    
    def custom_detection_algorithm(self, image):
        # 实现自定义检测算法
        pass

# 注册自定义任务
api_system.register_custom_task("custom_detection", CustomRecognitionTask)
```

### 自定义语音命令
```python
# 添加新的语音命令映射
custom_commands = {
    "自定义命令": {
        "command_type": "custom_action",
        "parameters": {"action": "custom_function"}
    }
}

api_system.add_voice_commands(custom_commands)
```

## 📞 技术支持

- **API文档**: [完整API文档](docs/api_reference.md)
- **SDK文档**: [Python SDK文档](docs/python_sdk.md)
- **示例代码**: [GitHub示例仓库](examples/)
- **问题反馈**: [GitHub Issues](https://github.com/your-repo/yolos/issues)

---

*YOLOS外部API系统 - 让AIoT设备更智能、更易用、更强大*