/*
 * YOLOS ESP32-CAM 视觉检测模块
 * 支持图像采集、WiFi传输、MQTT通信
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <WebServer.h>
#include <ESPmDNS.h>

// 摄像头引脚定义 (AI-Thinker ESP32-CAM)
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// LED和按钮引脚
#define LED_BUILTIN        4
#define FLASH_LED         4
#define BUTTON_PIN        0

// WiFi配置
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// MQTT配置
const char* mqtt_server = "192.168.1.100";
const int mqtt_port = 1883;
const char* mqtt_user = "";
const char* mqtt_password = "";
const char* client_id = "yolos_esp32_cam";

// 话题定义
const char* topic_image = "yolos/esp32/image";
const char* topic_status = "yolos/esp32/status";
const char* topic_command = "yolos/esp32/command";
const char* topic_config = "yolos/esp32/config";

// 全局对象
WiFiClient espClient;
PubSubClient mqtt_client(espClient);
WebServer server(80);

// 配置参数
struct Config {
  int image_quality = 10;        // 0-63, 越小质量越高
  framesize_t frame_size = FRAMESIZE_QVGA;  // 图像尺寸
  int capture_interval = 5000;   // 拍照间隔(ms)
  bool auto_capture = false;     // 自动拍照
  bool flash_enabled = false;    // 闪光灯
  int detection_threshold = 50;  // 检测阈值
} config;

// 状态变量
unsigned long last_capture = 0;
unsigned long last_status = 0;
bool camera_initialized = false;
int capture_count = 0;
int upload_count = 0;

void setup() {
  Serial.begin(115200);
  Serial.println("YOLOS ESP32-CAM 启动中...");
  
  // 初始化GPIO
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(FLASH_LED, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  
  // 初始化摄像头
  if (initCamera()) {
    Serial.println("摄像头初始化成功");
    camera_initialized = true;
  } else {
    Serial.println("摄像头初始化失败");
  }
  
  // 连接WiFi
  connectWiFi();
  
  // 初始化MQTT
  mqtt_client.setServer(mqtt_server, mqtt_port);
  mqtt_client.setCallback(mqttCallback);
  
  // 初始化Web服务器
  setupWebServer();
  
  // 启动mDNS
  if (MDNS.begin("yolos-esp32")) {
    Serial.println("mDNS启动成功: yolos-esp32.local");
  }
  
  Serial.println("系统初始化完成");
  blinkLED(3, 200);  // 启动指示
}

void loop() {
  // 处理MQTT连接
  if (!mqtt_client.connected()) {
    reconnectMQTT();
  }
  mqtt_client.loop();
  
  // 处理Web服务器
  server.handleClient();
  
  // 检查按钮
  if (digitalRead(BUTTON_PIN) == LOW) {
    delay(50);  // 消抖
    if (digitalRead(BUTTON_PIN) == LOW) {
      captureAndSend();
      while (digitalRead(BUTTON_PIN) == LOW) delay(10);
    }
  }
  
  // 自动拍照
  if (config.auto_capture && camera_initialized) {
    if (millis() - last_capture > config.capture_interval) {
      captureAndSend();
    }
  }
  
  // 发送状态信息
  if (millis() - last_status > 30000) {  // 每30秒
    sendStatus();
    last_status = millis();
  }
  
  delay(100);
}

bool initCamera() {
  camera_config_t camera_config;
  camera_config.ledc_channel = LEDC_CHANNEL_0;
  camera_config.ledc_timer = LEDC_TIMER_0;
  camera_config.pin_d0 = Y2_GPIO_NUM;
  camera_config.pin_d1 = Y3_GPIO_NUM;
  camera_config.pin_d2 = Y4_GPIO_NUM;
  camera_config.pin_d3 = Y5_GPIO_NUM;
  camera_config.pin_d4 = Y6_GPIO_NUM;
  camera_config.pin_d5 = Y7_GPIO_NUM;
  camera_config.pin_d6 = Y8_GPIO_NUM;
  camera_config.pin_d7 = Y9_GPIO_NUM;
  camera_config.pin_xclk = XCLK_GPIO_NUM;
  camera_config.pin_pclk = PCLK_GPIO_NUM;
  camera_config.pin_vsync = VSYNC_GPIO_NUM;
  camera_config.pin_href = HREF_GPIO_NUM;
  camera_config.pin_sscb_sda = SIOD_GPIO_NUM;
  camera_config.pin_sscb_scl = SIOC_GPIO_NUM;
  camera_config.pin_pwdn = PWDN_GPIO_NUM;
  camera_config.pin_reset = RESET_GPIO_NUM;
  camera_config.xclk_freq_hz = 20000000;
  camera_config.pixel_format = PIXFORMAT_JPEG;
  
  // 根据PSRAM选择帧缓冲区
  if (psramFound()) {
    camera_config.frame_size = FRAMESIZE_UXGA;
    camera_config.jpeg_quality = 10;
    camera_config.fb_count = 2;
  } else {
    camera_config.frame_size = FRAMESIZE_SVGA;
    camera_config.jpeg_quality = 12;
    camera_config.fb_count = 1;
  }
  
  // 初始化摄像头
  esp_err_t err = esp_camera_init(&camera_config);
  if (err != ESP_OK) {
    Serial.printf("摄像头初始化失败: 0x%x", err);
    return false;
  }
  
  // 设置摄像头参数
  sensor_t* s = esp_camera_sensor_get();
  s->set_framesize(s, config.frame_size);
  s->set_quality(s, config.image_quality);
  
  return true;
}

void connectWiFi() {
  WiFi.begin(ssid, password);
  Serial.print("连接WiFi");
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.print("WiFi连接成功! IP地址: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println();
    Serial.println("WiFi连接失败");
  }
}

void reconnectMQTT() {
  while (!mqtt_client.connected()) {
    Serial.print("连接MQTT服务器...");
    
    if (mqtt_client.connect(client_id, mqtt_user, mqtt_password)) {
      Serial.println("MQTT连接成功");
      
      // 订阅命令话题
      mqtt_client.subscribe(topic_command);
      mqtt_client.subscribe(topic_config);
      
      // 发送上线消息
      sendStatus();
      
    } else {
      Serial.print("MQTT连接失败, rc=");
      Serial.print(mqtt_client.state());
      Serial.println(" 5秒后重试");
      delay(5000);
    }
  }
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  String message;
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }
  
  Serial.printf("收到MQTT消息 [%s]: %s\n", topic, message.c_str());
  
  if (strcmp(topic, topic_command) == 0) {
    handleCommand(message);
  } else if (strcmp(topic, topic_config) == 0) {
    handleConfig(message);
  }
}

void handleCommand(String command) {
  if (command == "capture") {
    captureAndSend();
  } else if (command == "status") {
    sendStatus();
  } else if (command == "restart") {
    ESP.restart();
  } else if (command == "flash_on") {
    digitalWrite(FLASH_LED, HIGH);
  } else if (command == "flash_off") {
    digitalWrite(FLASH_LED, LOW);
  }
}

void handleConfig(String configJson) {
  DynamicJsonDocument doc(1024);
  deserializeJson(doc, configJson);
  
  if (doc.containsKey("image_quality")) {
    config.image_quality = doc["image_quality"];
  }
  if (doc.containsKey("capture_interval")) {
    config.capture_interval = doc["capture_interval"];
  }
  if (doc.containsKey("auto_capture")) {
    config.auto_capture = doc["auto_capture"];
  }
  if (doc.containsKey("flash_enabled")) {
    config.flash_enabled = doc["flash_enabled"];
  }
  
  // 应用摄像头设置
  sensor_t* s = esp_camera_sensor_get();
  if (s) {
    s->set_quality(s, config.image_quality);
  }
  
  Serial.println("配置已更新");
  sendStatus();
}

void captureAndSend() {
  if (!camera_initialized) {
    Serial.println("摄像头未初始化");
    return;
  }
  
  digitalWrite(LED_BUILTIN, HIGH);
  
  // 开启闪光灯
  if (config.flash_enabled) {
    digitalWrite(FLASH_LED, HIGH);
    delay(100);
  }
  
  // 拍照
  camera_fb_t* fb = esp_camera_fb_get();
  
  // 关闭闪光灯
  if (config.flash_enabled) {
    digitalWrite(FLASH_LED, LOW);
  }
  
  if (!fb) {
    Serial.println("拍照失败");
    digitalWrite(LED_BUILTIN, LOW);
    return;
  }
  
  capture_count++;
  last_capture = millis();
  
  Serial.printf("拍照成功: %d bytes\n", fb->len);
  
  // 发送图像数据
  if (mqtt_client.connected()) {
    // 创建图像信息JSON
    DynamicJsonDocument doc(512);
    doc["timestamp"] = millis();
    doc["client_id"] = client_id;
    doc["image_size"] = fb->len;
    doc["width"] = fb->width;
    doc["height"] = fb->height;
    doc["format"] = "jpeg";
    doc["quality"] = config.image_quality;
    doc["capture_count"] = capture_count;
    
    String imageInfo;
    serializeJson(doc, imageInfo);
    
    // 发送图像信息
    mqtt_client.publish((String(topic_image) + "/info").c_str(), imageInfo.c_str());
    
    // 分块发送图像数据
    const int chunk_size = 1024;
    int total_chunks = (fb->len + chunk_size - 1) / chunk_size;
    
    for (int i = 0; i < total_chunks; i++) {
      int start = i * chunk_size;
      int end = min(start + chunk_size, (int)fb->len);
      int current_size = end - start;
      
      String chunk_topic = String(topic_image) + "/data/" + String(i);
      mqtt_client.publish(chunk_topic.c_str(), fb->buf + start, current_size);
      
      delay(10);  // 避免发送过快
    }
    
    // 发送完成标志
    DynamicJsonDocument complete_doc(256);
    complete_doc["total_chunks"] = total_chunks;
    complete_doc["total_size"] = fb->len;
    complete_doc["timestamp"] = millis();
    
    String completeInfo;
    serializeJson(complete_doc, completeInfo);
    mqtt_client.publish((String(topic_image) + "/complete").c_str(), completeInfo.c_str());
    
    upload_count++;
    Serial.printf("图像上传完成: %d chunks\n", total_chunks);
  }
  
  esp_camera_fb_return(fb);
  digitalWrite(LED_BUILTIN, LOW);
}

void sendStatus() {
  DynamicJsonDocument doc(1024);
  doc["timestamp"] = millis();
  doc["client_id"] = client_id;
  doc["wifi_connected"] = WiFi.status() == WL_CONNECTED;
  doc["wifi_rssi"] = WiFi.RSSI();
  doc["ip_address"] = WiFi.localIP().toString();
  doc["mqtt_connected"] = mqtt_client.connected();
  doc["camera_initialized"] = camera_initialized;
  doc["free_heap"] = ESP.getFreeHeap();
  doc["uptime"] = millis();
  doc["capture_count"] = capture_count;
  doc["upload_count"] = upload_count;
  doc["auto_capture"] = config.auto_capture;
  doc["capture_interval"] = config.capture_interval;
  doc["image_quality"] = config.image_quality;
  
  String status;
  serializeJson(doc, status);
  
  if (mqtt_client.connected()) {
    mqtt_client.publish(topic_status, status.c_str());
  }
  
  Serial.println("状态信息已发送");
}

void setupWebServer() {
  // 主页
  server.on("/", HTTP_GET, []() {
    String html = "<html><head><title>YOLOS ESP32-CAM</title></head>";
    html += "<body><h1>YOLOS ESP32-CAM</h1>";
    html += "<p>状态: " + String(camera_initialized ? "正常" : "摄像头错误") + "</p>";
    html += "<p>WiFi: " + WiFi.localIP().toString() + "</p>";
    html += "<p>拍照次数: " + String(capture_count) + "</p>";
    html += "<p><a href='/capture'>拍照</a></p>";
    html += "<p><a href='/stream'>视频流</a></p>";
    html += "<p><a href='/status'>状态</a></p>";
    html += "</body></html>";
    server.send(200, "text/html", html);
  });
  
  // 拍照
  server.on("/capture", HTTP_GET, []() {
    captureAndSend();
    server.send(200, "text/plain", "拍照完成");
  });
  
  // 状态
  server.on("/status", HTTP_GET, []() {
    DynamicJsonDocument doc(1024);
    doc["camera_initialized"] = camera_initialized;
    doc["wifi_connected"] = WiFi.status() == WL_CONNECTED;
    doc["mqtt_connected"] = mqtt_client.connected();
    doc["capture_count"] = capture_count;
    doc["free_heap"] = ESP.getFreeHeap();
    
    String status;
    serializeJson(doc, status);
    server.send(200, "application/json", status);
  });
  
  server.begin();
  Serial.println("Web服务器已启动");
}

void blinkLED(int times, int delay_ms) {
  for (int i = 0; i < times; i++) {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(delay_ms);
    digitalWrite(LED_BUILTIN, LOW);
    delay(delay_ms);
  }
}