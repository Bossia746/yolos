#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arduino平台适配器
为Arduino环境提供YOLOS识别系统支持
"""

import os
import json
import logging
import time
import serial
import struct
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import cv2

logger = logging.getLogger(__name__)

class ArduinoAdapter:
    """Arduino平台适配器"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        
        # Arduino连接配置
        self.serial_port = self.config.get('serial_port', 'COM3')  # Windows默认
        self.baud_rate = self.config.get('baud_rate', 115200)
        self.timeout = self.config.get('timeout', 1.0)
        
        # 串口连接
        self.serial_connection = None
        
        # Arduino兼容的识别模型
        self.arduino_models = self._init_arduino_models()
        
        # 数据缓冲区
        self.image_buffer = bytearray()
        self.result_buffer = []
        
        logger.info("Arduino适配器初始化完成")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'serial_port': self._detect_arduino_port(),
            'baud_rate': 115200,
            'timeout': 1.0,
            'image_format': 'jpeg',
            'max_image_size': 320 * 240,  # Arduino内存限制
            'recognition_modes': ['simple_object', 'color_detection', 'motion_detection'],
            'communication_protocol': 'binary',
            'enable_compression': True,
            'max_results': 5
        }
    
    def _detect_arduino_port(self) -> str:
        """自动检测Arduino端口"""
        import platform
        
        system = platform.system().lower()
        
        if system == 'windows':
            # Windows系统常见Arduino端口
            common_ports = ['COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8']
        elif system == 'darwin':  # macOS
            common_ports = ['/dev/tty.usbmodem*', '/dev/tty.usbserial*']
        else:  # Linux
            common_ports = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0', '/dev/ttyACM1']
        
        # 尝试检测可用端口
        try:
            import serial.tools.list_ports
            
            available_ports = [port.device for port in serial.tools.list_ports.comports()]
            
            for port in common_ports:
                if '*' in port:
                    # 通配符匹配
                    import glob
                    matching_ports = glob.glob(port)
                    if matching_ports:
                        return matching_ports[0]
                elif port in available_ports:
                    return port
            
            # 返回第一个可用端口
            if available_ports:
                return available_ports[0]
                
        except ImportError:
            logger.warning("pyserial未安装，无法自动检测Arduino端口")
        
        # 默认端口
        return 'COM3' if system == 'windows' else '/dev/ttyUSB0'
    
    def _init_arduino_models(self) -> Dict[str, Any]:
        """初始化Arduino兼容的识别模型"""
        models = {
            'color_detector': ArduinoColorDetector(),
            'motion_detector': ArduinoMotionDetector(),
            'simple_object_detector': ArduinoSimpleObjectDetector(),
            'edge_detector': ArduinoEdgeDetector(),
            'blob_detector': ArduinoBlobDetector()
        }
        
        return models
    
    def connect(self) -> bool:
        """连接Arduino"""
        try:
            if self.serial_connection and self.serial_connection.is_open:
                logger.info("Arduino已连接")
                return True
            
            self.serial_connection = serial.Serial(
                port=self.serial_port,
                baudrate=self.baud_rate,
                timeout=self.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            
            # 等待Arduino初始化
            time.sleep(2)
            
            # 发送握手信号
            if self._send_handshake():
                logger.info(f"✓ Arduino连接成功: {self.serial_port}")
                return True
            else:
                logger.error("Arduino握手失败")
                return False
                
        except Exception as e:
            logger.error(f"Arduino连接失败: {e}")
            return False
    
    def disconnect(self):
        """断开Arduino连接"""
        try:
            if self.serial_connection and self.serial_connection.is_open:
                self.serial_connection.close()
                logger.info("Arduino连接已断开")
        except Exception as e:
            logger.error(f"断开Arduino连接失败: {e}")
    
    def _send_handshake(self) -> bool:
        """发送握手信号"""
        try:
            # 发送握手命令
            handshake_cmd = b'YOLOS_HANDSHAKE\n'
            self.serial_connection.write(handshake_cmd)
            
            # 等待响应
            response = self.serial_connection.readline().decode('utf-8').strip()
            
            return response == 'YOLOS_ACK'
            
        except Exception as e:
            logger.error(f"握手失败: {e}")
            return False
    
    def send_image_for_recognition(self, image: np.ndarray, recognition_type: str = 'simple_object') -> Dict[str, Any]:
        """发送图像进行识别"""
        if not self.serial_connection or not self.serial_connection.is_open:
            logger.error("Arduino未连接")
            return {'error': 'Arduino not connected'}
        
        try:
            # 预处理图像（适应Arduino内存限制）
            processed_image = self._preprocess_image_for_arduino(image)
            
            # 选择识别模型
            if recognition_type not in self.arduino_models:
                logger.error(f"不支持的识别类型: {recognition_type}")
                return {'error': f'Unsupported recognition type: {recognition_type}'}
            
            model = self.arduino_models[recognition_type]
            
            # 本地处理（减少Arduino负担）
            local_result = model.process(processed_image)
            
            # 发送简化的结果到Arduino
            arduino_result = self._send_result_to_arduino(local_result, recognition_type)
            
            return {
                'recognition_type': recognition_type,
                'local_result': local_result,
                'arduino_result': arduino_result,
                'image_size': processed_image.shape,
                'processing_time': time.time()
            }
            
        except Exception as e:
            logger.error(f"Arduino图像识别失败: {e}")
            return {'error': str(e)}
    
    def _preprocess_image_for_arduino(self, image: np.ndarray) -> np.ndarray:
        """为Arduino预处理图像"""
        # 调整图像大小（Arduino内存限制）
        max_width = 160
        max_height = 120
        
        h, w = image.shape[:2]
        
        # 计算缩放比例
        scale = min(max_width / w, max_height / h)
        
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 转换为灰度图（节省内存）
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        return image
    
    def _send_result_to_arduino(self, result: Dict[str, Any], recognition_type: str) -> Dict[str, Any]:
        """发送识别结果到Arduino"""
        try:
            # 构建Arduino命令
            cmd_data = {
                'type': recognition_type,
                'result': self._simplify_result_for_arduino(result),
                'timestamp': int(time.time())
            }
            
            # 序列化为JSON（Arduino可解析）
            cmd_json = json.dumps(cmd_data, separators=(',', ':'))
            cmd_bytes = cmd_json.encode('utf-8') + b'\n'
            
            # 发送命令
            self.serial_connection.write(cmd_bytes)
            
            # 等待Arduino响应
            response = self.serial_connection.readline().decode('utf-8').strip()
            
            if response.startswith('ACK'):
                return {'status': 'success', 'arduino_response': response}
            else:
                return {'status': 'error', 'arduino_response': response}
                
        except Exception as e:
            logger.error(f"发送结果到Arduino失败: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _simplify_result_for_arduino(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """简化识别结果（适应Arduino处理能力）"""
        simplified = {}
        
        # 只保留关键信息
        if 'objects_detected' in result:
            simplified['objects'] = min(result['objects_detected'], 3)  # 最多3个对象
        
        if 'colors_detected' in result:
            simplified['colors'] = result['colors_detected'][:2]  # 最多2种颜色
        
        if 'motion_detected' in result:
            simplified['motion'] = result['motion_detected']
        
        if 'confidence' in result:
            simplified['conf'] = round(result['confidence'], 2)
        
        # 位置信息（简化）
        if 'bbox' in result:
            bbox = result['bbox']
            simplified['pos'] = {
                'x': int(bbox[0]),
                'y': int(bbox[1]),
                'w': int(bbox[2] - bbox[0]),
                'h': int(bbox[3] - bbox[1])
            }
        
        return simplified
    
    def receive_sensor_data(self) -> Optional[Dict[str, Any]]:
        """接收Arduino传感器数据"""
        if not self.serial_connection or not self.serial_connection.is_open:
            return None
        
        try:
            if self.serial_connection.in_waiting > 0:
                data_line = self.serial_connection.readline().decode('utf-8').strip()
                
                if data_line.startswith('SENSOR:'):
                    sensor_json = data_line[7:]  # 移除'SENSOR:'前缀
                    sensor_data = json.loads(sensor_json)
                    
                    return {
                        'timestamp': time.time(),
                        'sensor_data': sensor_data,
                        'source': 'arduino'
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"接收Arduino传感器数据失败: {e}")
            return None
    
    def send_control_command(self, command: str, parameters: Optional[Dict] = None) -> bool:
        """发送控制命令到Arduino"""
        if not self.serial_connection or not self.serial_connection.is_open:
            logger.error("Arduino未连接")
            return False
        
        try:
            cmd_data = {
                'cmd': command,
                'params': parameters or {}
            }
            
            cmd_json = json.dumps(cmd_data, separators=(',', ':'))
            cmd_bytes = cmd_json.encode('utf-8') + b'\n'
            
            self.serial_connection.write(cmd_bytes)
            
            # 等待确认
            response = self.serial_connection.readline().decode('utf-8').strip()
            
            return response == 'CMD_ACK'
            
        except Exception as e:
            logger.error(f"发送Arduino控制命令失败: {e}")
            return False
    
    def get_arduino_status(self) -> Dict[str, Any]:
        """获取Arduino状态"""
        if not self.serial_connection or not self.serial_connection.is_open:
            return {'connected': False, 'status': 'disconnected'}
        
        try:
            # 发送状态查询命令
            self.serial_connection.write(b'STATUS\n')
            
            response = self.serial_connection.readline().decode('utf-8').strip()
            
            if response.startswith('STATUS:'):
                status_json = response[7:]
                status_data = json.loads(status_json)
                
                return {
                    'connected': True,
                    'status': 'connected',
                    'arduino_info': status_data
                }
            
            return {'connected': True, 'status': 'unknown'}
            
        except Exception as e:
            logger.error(f"获取Arduino状态失败: {e}")
            return {'connected': False, 'status': 'error', 'message': str(e)}

class ArduinoColorDetector:
    """Arduino颜色检测器"""
    
    def process(self, image: np.ndarray) -> Dict[str, Any]:
        """处理颜色检测"""
        # 定义颜色范围（HSV）
        color_ranges = {
            'red': [(0, 50, 50), (10, 255, 255)],
            'green': [(40, 50, 50), (80, 255, 255)],
            'blue': [(100, 50, 50), (130, 255, 255)],
            'yellow': [(20, 50, 50), (40, 255, 255)],
            'orange': [(10, 50, 50), (20, 255, 255)]
        }
        
        # 转换为HSV
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            # 灰度图转HSV（模拟）
            hsv = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
        
        detected_colors = []
        
        for color_name, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            pixel_count = cv2.countNonZero(mask)
            
            if pixel_count > 100:  # 阈值
                percentage = (pixel_count / (image.shape[0] * image.shape[1])) * 100
                detected_colors.append({
                    'color': color_name,
                    'percentage': round(percentage, 2),
                    'pixel_count': pixel_count
                })
        
        # 按百分比排序
        detected_colors.sort(key=lambda x: x['percentage'], reverse=True)
        
        return {
            'colors_detected': detected_colors[:3],  # 最多3种颜色
            'primary_color': detected_colors[0]['color'] if detected_colors else 'unknown',
            'confidence': detected_colors[0]['percentage'] / 100 if detected_colors else 0.0
        }

class ArduinoMotionDetector:
    """Arduino运动检测器"""
    
    def __init__(self):
        self.previous_frame = None
    
    def process(self, image: np.ndarray) -> Dict[str, Any]:
        """处理运动检测"""
        if self.previous_frame is None:
            self.previous_frame = image.copy()
            return {
                'motion_detected': False,
                'motion_area': 0,
                'confidence': 0.0
            }
        
        # 计算帧差
        diff = cv2.absdiff(self.previous_frame, image)
        
        # 二值化
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # 计算运动区域
        motion_pixels = cv2.countNonZero(thresh)
        total_pixels = image.shape[0] * image.shape[1]
        motion_percentage = (motion_pixels / total_pixels) * 100
        
        # 更新前一帧
        self.previous_frame = image.copy()
        
        motion_detected = motion_percentage > 5.0  # 5%阈值
        
        return {
            'motion_detected': motion_detected,
            'motion_area': round(motion_percentage, 2),
            'confidence': min(motion_percentage / 20.0, 1.0)  # 归一化到0-1
        }

class ArduinoSimpleObjectDetector:
    """Arduino简单物体检测器"""
    
    def process(self, image: np.ndarray) -> Dict[str, Any]:
        """处理简单物体检测"""
        # 使用轮廓检测
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > 500:  # 最小面积阈值
                x, y, w, h = cv2.boundingRect(contour)
                
                # 简单形状分类
                shape = self._classify_shape(contour)
                
                objects.append({
                    'bbox': (x, y, x + w, y + h),
                    'area': int(area),
                    'shape': shape,
                    'confidence': min(area / 5000.0, 1.0)
                })
        
        # 按面积排序
        objects.sort(key=lambda x: x['area'], reverse=True)
        
        return {
            'objects_detected': len(objects),
            'objects': objects[:3],  # 最多3个对象
            'largest_object': objects[0] if objects else None
        }
    
    def _classify_shape(self, contour) -> str:
        """简单形状分类"""
        # 计算轮廓近似
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        vertices = len(approx)
        
        if vertices == 3:
            return 'triangle'
        elif vertices == 4:
            return 'rectangle'
        elif vertices > 8:
            return 'circle'
        else:
            return 'polygon'

class ArduinoEdgeDetector:
    """Arduino边缘检测器"""
    
    def process(self, image: np.ndarray) -> Dict[str, Any]:
        """处理边缘检测"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Canny边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 统计边缘像素
        edge_pixels = cv2.countNonZero(edges)
        total_pixels = gray.shape[0] * gray.shape[1]
        edge_percentage = (edge_pixels / total_pixels) * 100
        
        return {
            'edges_detected': edge_pixels > 0,
            'edge_density': round(edge_percentage, 2),
            'edge_pixels': edge_pixels,
            'confidence': min(edge_percentage / 10.0, 1.0)
        }

class ArduinoBlobDetector:
    """Arduino斑点检测器"""
    
    def process(self, image: np.ndarray) -> Dict[str, Any]:
        """处理斑点检测"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 简单斑点检测器参数
        params = cv2.SimpleBlobDetector_Params()
        
        # 设置阈值
        params.minThreshold = 10
        params.maxThreshold = 200
        
        # 按面积过滤
        params.filterByArea = True
        params.minArea = 100
        params.maxArea = 5000
        
        # 按圆度过滤
        params.filterByCircularity = True
        params.minCircularity = 0.1
        
        # 创建检测器
        detector = cv2.SimpleBlobDetector_create(params)
        
        # 检测斑点
        keypoints = detector.detect(gray)
        
        blobs = []
        for kp in keypoints:
            blobs.append({
                'center': (int(kp.pt[0]), int(kp.pt[1])),
                'size': round(kp.size, 2),
                'confidence': round(kp.response, 2)
            })
        
        return {
            'blobs_detected': len(blobs),
            'blobs': blobs[:5],  # 最多5个斑点
            'largest_blob': max(blobs, key=lambda x: x['size']) if blobs else None
        }

# Arduino代码生成器
class ArduinoCodeGenerator:
    """Arduino代码生成器"""
    
    @staticmethod
    def generate_yolos_arduino_sketch() -> str:
        """生成YOLOS Arduino代码"""
        return '''
/*
 * YOLOS Arduino Integration Sketch
 * 支持与YOLOS识别系统的通信
 */

#include <ArduinoJson.h>

// 串口配置
const int BAUD_RATE = 115200;
const int BUFFER_SIZE = 512;

// LED引脚（用于状态指示）
const int STATUS_LED = 13;
const int RECOGNITION_LED = 12;

// 传感器引脚
const int MOTION_SENSOR = 2;
const int LIGHT_SENSOR = A0;
const int TEMP_SENSOR = A1;

// 全局变量
char inputBuffer[BUFFER_SIZE];
int bufferIndex = 0;
bool systemReady = false;

void setup() {
  // 初始化串口
  Serial.begin(BAUD_RATE);
  
  // 初始化引脚
  pinMode(STATUS_LED, OUTPUT);
  pinMode(RECOGNITION_LED, OUTPUT);
  pinMode(MOTION_SENSOR, INPUT);
  
  // 启动指示
  digitalWrite(STATUS_LED, HIGH);
  
  // 等待系统稳定
  delay(1000);
  
  systemReady = true;
  Serial.println("YOLOS_ARDUINO_READY");
}

void loop() {
  // 处理串口数据
  if (Serial.available()) {
    processSerialData();
  }
  
  // 发送传感器数据
  sendSensorData();
  
  // 状态指示
  updateStatusLED();
  
  delay(100);
}

void processSerialData() {
  while (Serial.available() && bufferIndex < BUFFER_SIZE - 1) {
    char c = Serial.read();
    
    if (c == '\\n') {
      inputBuffer[bufferIndex] = '\\0';
      processCommand(inputBuffer);
      bufferIndex = 0;
    } else {
      inputBuffer[bufferIndex++] = c;
    }
  }
}

void processCommand(const char* command) {
  // 握手命令
  if (strcmp(command, "YOLOS_HANDSHAKE") == 0) {
    Serial.println("YOLOS_ACK");
    return;
  }
  
  // 状态查询
  if (strcmp(command, "STATUS") == 0) {
    sendStatus();
    return;
  }
  
  // JSON命令解析
  DynamicJsonDocument doc(256);
  DeserializationError error = deserializeJson(doc, command);
  
  if (error) {
    Serial.println("JSON_ERROR");
    return;
  }
  
  // 处理识别结果
  if (doc.containsKey("type")) {
    processRecognitionResult(doc);
  }
  
  // 处理控制命令
  if (doc.containsKey("cmd")) {
    processControlCommand(doc);
  }
}

void processRecognitionResult(JsonDocument& doc) {
  String type = doc["type"];
  JsonObject result = doc["result"];
  
  // 识别结果LED指示
  digitalWrite(RECOGNITION_LED, HIGH);
  
  // 根据识别类型执行相应动作
  if (type == "color_detection") {
    handleColorDetection(result);
  } else if (type == "motion_detection") {
    handleMotionDetection(result);
  } else if (type == "simple_object") {
    handleObjectDetection(result);
  }
  
  Serial.println("ACK_RECOGNITION");
  
  // 延时后关闭LED
  delay(200);
  digitalWrite(RECOGNITION_LED, LOW);
}

void handleColorDetection(JsonObject& result) {
  if (result.containsKey("colors")) {
    // 根据检测到的颜色执行动作
    // 例如：控制RGB LED显示相应颜色
  }
}

void handleMotionDetection(JsonObject& result) {
  if (result["motion"]) {
    // 检测到运动时的动作
    // 例如：触发警报或记录事件
  }
}

void handleObjectDetection(JsonObject& result) {
  if (result.containsKey("objects")) {
    int objectCount = result["objects"];
    // 根据检测到的对象数量执行动作
  }
}

void processControlCommand(JsonDocument& doc) {
  String cmd = doc["cmd"];
  JsonObject params = doc["params"];
  
  if (cmd == "led_control") {
    controlLED(params);
  } else if (cmd == "sensor_config") {
    configureSensors(params);
  }
  
  Serial.println("CMD_ACK");
}

void controlLED(JsonObject& params) {
  if (params.containsKey("pin") && params.containsKey("state")) {
    int pin = params["pin"];
    bool state = params["state"];
    digitalWrite(pin, state ? HIGH : LOW);
  }
}

void configureSensors(JsonObject& params) {
  // 配置传感器参数
  // 例如：设置采样频率、阈值等
}

void sendSensorData() {
  static unsigned long lastSensorRead = 0;
  unsigned long currentTime = millis();
  
  // 每秒发送一次传感器数据
  if (currentTime - lastSensorRead >= 1000) {
    DynamicJsonDocument doc(128);
    
    doc["motion"] = digitalRead(MOTION_SENSOR);
    doc["light"] = analogRead(LIGHT_SENSOR);
    doc["temp"] = analogRead(TEMP_SENSOR);
    doc["uptime"] = currentTime;
    
    Serial.print("SENSOR:");
    serializeJson(doc, Serial);
    Serial.println();
    
    lastSensorRead = currentTime;
  }
}

void sendStatus() {
  DynamicJsonDocument doc(128);
  
  doc["ready"] = systemReady;
  doc["uptime"] = millis();
  doc["free_memory"] = getFreeMemory();
  doc["version"] = "1.0.0";
  
  Serial.print("STATUS:");
  serializeJson(doc, Serial);
  Serial.println();
}

int getFreeMemory() {
  extern int __heap_start, *__brkval;
  int v;
  return (int) &v - (__brkval == 0 ? (int) &__heap_start : (int) __brkval);
}

void updateStatusLED() {
  static unsigned long lastBlink = 0;
  static bool ledState = false;
  
  unsigned long currentTime = millis();
  
  if (systemReady) {
    // 系统就绪时慢闪
    if (currentTime - lastBlink >= 1000) {
      ledState = !ledState;
      digitalWrite(STATUS_LED, ledState ? HIGH : LOW);
      lastBlink = currentTime;
    }
  } else {
    // 系统未就绪时快闪
    if (currentTime - lastBlink >= 200) {
      ledState = !ledState;
      digitalWrite(STATUS_LED, ledState ? HIGH : LOW);
      lastBlink = currentTime;
    }
  }
}
'''

# 便捷函数
def create_arduino_adapter(config: Optional[Dict] = None) -> ArduinoAdapter:
    """创建Arduino适配器"""
    return ArduinoAdapter(config)

def generate_arduino_sketch_file(output_path: str = "./arduino_yolos_sketch.ino"):
    """生成Arduino代码文件"""
    sketch_code = ArduinoCodeGenerator.generate_yolos_arduino_sketch()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(sketch_code)
    
    logger.info(f"Arduino代码已生成: {output_path}")

if __name__ == "__main__":
    # 测试Arduino适配器
    adapter = create_arduino_adapter()
    
    # 尝试连接Arduino
    if adapter.connect():
        print("✓ Arduino连接成功")
        
        # 测试图像识别
        test_image = np.random.randint(0, 255, (120, 160), dtype=np.uint8)
        result = adapter.send_image_for_recognition(test_image, 'color_detection')
        print(f"识别结果: {result}")
        
        # 获取Arduino状态
        status = adapter.get_arduino_status()
        print(f"Arduino状态: {status}")
        
        adapter.disconnect()
    else:
        print("✗ Arduino连接失败")
    
    # 生成Arduino代码
    generate_arduino_sketch_file()
    print("✓ Arduino代码已生成")