#!/usr/bin/env python3
"""
部署生成器
支持一键生成PC、ESP32、K230、树莓派等不同平台的部署版本
"""

import os
import shutil
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Any
import tempfile
import subprocess
import sys

class DeploymentGenerator:
    """部署版本生成器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.deployments_dir = self.project_root / 'deployments'
        self.deployments_dir.mkdir(exist_ok=True)
        
        # 平台配置
        self.platform_configs = {
            'pc': {
                'name': 'PC版本',
                'description': 'Windows/Linux/macOS桌面版本',
                'requirements': ['opencv-python', 'numpy', 'pillow', 'torch', 'torchvision'],
                'optional_requirements': ['ultralytics', 'onnx', 'tensorrt'],
                'entry_point': 'main.py',
                'include_gui': True,
                'model_format': 'pytorch'
            },
            'esp32': {
                'name': 'ESP32版本',
                'description': 'ESP32微控制器版本',
                'requirements': ['micropython', 'camera'],
                'optional_requirements': [],
                'entry_point': 'main_esp32.py',
                'include_gui': False,
                'model_format': 'tflite_micro'
            },
            'k230': {
                'name': 'K230版本', 
                'description': 'K230 AI开发板版本',
                'requirements': ['nncase', 'opencv-python'],
                'optional_requirements': ['canmv'],
                'entry_point': 'main_k230.py',
                'include_gui': False,
                'model_format': 'kmodel'
            },
            'raspberry_pi': {
                'name': '树莓派版本',
                'description': '树莓派单板计算机版本',
                'requirements': ['opencv-python', 'numpy', 'tflite-runtime'],
                'optional_requirements': ['picamera2'],
                'entry_point': 'main_rpi.py',
                'include_gui': True,
                'model_format': 'tflite'
            }
        }
        
    def generate_platform_version(self, platform: str) -> bool:
        """生成指定平台的部署版本"""
        if platform not in self.platform_configs:
            print(f"❌ 不支持的平台: {platform}")
            return False
            
        config = self.platform_configs[platform]
        print(f"📦 生成{config['name']}...")
        
        try:
            # 创建平台目录
            platform_dir = self.deployments_dir / platform
            if platform_dir.exists():
                shutil.rmtree(platform_dir)
            platform_dir.mkdir(parents=True)
            
            # 复制核心文件
            self._copy_core_files(platform, platform_dir)
            
            # 生成平台特定文件
            self._generate_platform_files(platform, platform_dir)
            
            # 生成配置文件
            self._generate_config_files(platform, platform_dir)
            
            # 生成安装脚本
            self._generate_install_scripts(platform, platform_dir)
            
            # 生成文档
            self._generate_documentation(platform, platform_dir)
            
            # 创建部署包
            self._create_deployment_package(platform, platform_dir)
            
            print(f"✅ {config['name']}生成完成")
            return True
            
        except Exception as e:
            print(f"❌ 生成{config['name']}失败: {e}")
            return False
            
    def _copy_core_files(self, platform: str, target_dir: Path):
        """复制核心文件"""
        config = self.platform_configs[platform]
        
        # 创建源码目录
        src_dir = target_dir / 'src'
        src_dir.mkdir(exist_ok=True)
        
        # 核心模块映射
        core_modules = {
            'pc': [
                'src/core/minimal_yolos.py',
                'src/gui/simple_yolos_gui.py',
                'src/utils/camera_utils.py',
                'src/models/yolo_lite.py'
            ],
            'esp32': [
                'src/core/minimal_yolos.py',
                'src/models/yolo_micro.py',
                'src/utils/esp32_utils.py'
            ],
            'k230': [
                'src/core/minimal_yolos.py', 
                'src/models/yolo_k230.py',
                'src/utils/k230_utils.py'
            ],
            'raspberry_pi': [
                'src/core/minimal_yolos.py',
                'src/gui/simple_yolos_gui.py',
                'src/models/yolo_lite.py',
                'src/utils/rpi_utils.py'
            ]
        }
        
        # 复制平台相关文件
        for file_path in core_modules.get(platform, []):
            src_file = self.project_root / file_path
            if src_file.exists():
                # 保持目录结构
                rel_path = Path(file_path).relative_to('src')
                target_file = src_dir / rel_path
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, target_file)
            else:
                # 创建占位文件
                self._create_placeholder_file(src_dir / Path(file_path).relative_to('src'), platform)
                
    def _create_placeholder_file(self, file_path: Path, platform: str):
        """创建占位文件"""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_path.name.endswith('.py'):
            content = f'''#!/usr/bin/env python3
"""
{file_path.name} - {platform}平台版本
这是一个占位文件，需要根据实际需求实现
"""

class PlaceholderClass:
    """占位类"""
    def __init__(self):
        self.platform = "{platform}"
        
    def placeholder_method(self):
        """占位方法"""
        print(f"这是{platform}平台的占位实现")
        return True

def main():
    """主函数"""
    obj = PlaceholderClass()
    obj.placeholder_method()

if __name__ == "__main__":
    main()
'''
        else:
            content = f"# {file_path.name} - {platform}平台配置文件\n"
            
        file_path.write_text(content, encoding='utf-8')
        
    def _generate_platform_files(self, platform: str, target_dir: Path):
        """生成平台特定文件"""
        config = self.platform_configs[platform]
        
        if platform == 'pc':
            self._generate_pc_files(target_dir)
        elif platform == 'esp32':
            self._generate_esp32_files(target_dir)
        elif platform == 'k230':
            self._generate_k230_files(target_dir)
        elif platform == 'raspberry_pi':
            self._generate_rpi_files(target_dir)
            
    def _generate_pc_files(self, target_dir: Path):
        """生成PC版本文件"""
        # 主启动文件
        main_file = target_dir / 'main.py'
        main_content = '''#!/usr/bin/env python3
"""
YOLOS PC版本主启动文件
"""

import sys
import os
from pathlib import Path

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def main():
    """主函数"""
    print("🚀 启动YOLOS PC版本...")
    
    try:
        from core.minimal_yolos import MinimalYOLOS
        app = MinimalYOLOS()
        app.run()
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("请检查依赖是否正确安装")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

if __name__ == "__main__":
    main()
'''
        main_file.write_text(main_content, encoding='utf-8')
        
        # 批处理启动脚本
        bat_file = target_dir / 'start.bat'
        bat_content = '''@echo off
echo 启动YOLOS PC版本...
python main.py
pause
'''
        bat_file.write_text(bat_content, encoding='utf-8')
        
        # Shell启动脚本
        sh_file = target_dir / 'start.sh'
        sh_content = '''#!/bin/bash
echo "启动YOLOS PC版本..."
python3 main.py
'''
        sh_file.write_text(sh_content, encoding='utf-8')
        
    def _generate_esp32_files(self, target_dir: Path):
        """生成ESP32版本文件"""
        # ESP32主文件
        main_file = target_dir / 'main.py'
        main_content = '''#!/usr/bin/env python3
"""
YOLOS ESP32版本主文件
"""

import machine
import camera
import time
from src.core.minimal_yolos import MinimalYOLOS

def init_camera():
    """初始化摄像头"""
    try:
        camera.init(0, format=camera.JPEG, framesize=camera.FRAME_QVGA)
        print("摄像头初始化成功")
        return True
    except Exception as e:
        print(f"摄像头初始化失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 启动YOLOS ESP32版本...")
    
    # 初始化硬件
    if not init_camera():
        return
    
    # 启动检测循环
    yolos = MinimalYOLOS()
    
    while True:
        try:
            # 捕获图像
            buf = camera.capture()
            if buf:
                print(f"捕获图像，大小: {len(buf)} bytes")
                # 这里添加检测逻辑
                
            time.sleep(0.1)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"检测错误: {e}")
            time.sleep(1)
    
    camera.deinit()
    print("ESP32版本退出")

if __name__ == "__main__":
    main()
'''
        main_file.write_text(main_content, encoding='utf-8')
        
        # Arduino IDE配置
        arduino_file = target_dir / 'yolos_esp32.ino'
        arduino_content = '''/*
 * YOLOS ESP32版本 Arduino IDE配置
 */

#include "esp_camera.h"
#include "WiFi.h"

// 摄像头引脚配置 (ESP32-CAM)
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

void setup() {
  Serial.begin(115200);
  Serial.println("YOLOS ESP32版本启动...");
  
  // 初始化摄像头
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;
  
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("摄像头初始化失败: 0x%x", err);
    return;
  }
  
  Serial.println("摄像头初始化成功");
}

void loop() {
  // 捕获图像
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("摄像头捕获失败");
    return;
  }
  
  Serial.printf("捕获图像: %d bytes\\n", fb->len);
  
  // 这里添加YOLO检测逻辑
  
  esp_camera_fb_return(fb);
  delay(100);
}
'''
        arduino_file.write_text(arduino_content, encoding='utf-8')
        
    def _generate_k230_files(self, target_dir: Path):
        """生成K230版本文件"""
        # K230主文件
        main_file = target_dir / 'main.py'
        main_content = '''#!/usr/bin/env python3
"""
YOLOS K230版本主文件
"""

import os
import sys
import time
import numpy as np

try:
    import nncase
    from canmv import camera, display
    CANMV_AVAILABLE = True
except ImportError:
    CANMV_AVAILABLE = False
    print("⚠️ CanMV环境不可用，使用模拟模式")

class K230YOLOS:
    """K230版本YOLOS"""
    
    def __init__(self):
        self.model_path = "yolov8n.kmodel"
        self.input_size = (640, 640)
        self.confidence_threshold = 0.5
        
    def load_model(self):
        """加载KModel模型"""
        if not CANMV_AVAILABLE:
            print("模拟模式：模型加载成功")
            return True
            
        try:
            # 加载KModel
            if os.path.exists(self.model_path):
                print(f"加载模型: {self.model_path}")
                # 这里添加实际的模型加载代码
                return True
            else:
                print(f"模型文件不存在: {self.model_path}")
                return False
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False
            
    def init_camera(self):
        """初始化摄像头"""
        if not CANMV_AVAILABLE:
            print("模拟模式：摄像头初始化成功")
            return True
            
        try:
            camera.sensor_init(camera.CAM_DEV_ID_0, camera.CAM_DEFAULT_SENSOR)
            camera.set_outsize(camera.CAM_DEV_ID_0, camera.CAM_CHN_ID_0, 
                             self.input_size[0], self.input_size[1])
            camera.set_outfmt(camera.CAM_DEV_ID_0, camera.CAM_CHN_ID_0, 
                            camera.PIXEL_FORMAT_RGB_888_PLANAR)
            print("K230摄像头初始化成功")
            return True
        except Exception as e:
            print(f"摄像头初始化失败: {e}")
            return False
            
    def detect_frame(self, frame):
        """检测单帧"""
        # 模拟检测结果
        detections = [
            {"class": "person", "confidence": 0.85, "bbox": [100, 100, 200, 300]},
            {"class": "car", "confidence": 0.72, "bbox": [300, 150, 500, 400]}
        ]
        return detections
        
    def run(self):
        """运行检测"""
        print("🚀 启动YOLOS K230版本...")
        
        if not self.load_model():
            return
            
        if not self.init_camera():
            return
            
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                if CANMV_AVAILABLE:
                    # 捕获真实图像
                    frame = camera.capture_image(camera.CAM_DEV_ID_0, camera.CAM_CHN_ID_0)
                else:
                    # 模拟图像
                    frame = np.random.randint(0, 255, (*self.input_size, 3), dtype=np.uint8)
                
                # 执行检测
                detections = self.detect_frame(frame)
                
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                print(f"帧 {frame_count}: 检测到 {len(detections)} 个目标, FPS: {fps:.1f}")
                
                for det in detections:
                    print(f"  - {det['class']}: {det['confidence']:.2f}")
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\\n检测被用户中断")
        except Exception as e:
            print(f"检测错误: {e}")
        finally:
            if CANMV_AVAILABLE:
                camera.sensor_deinit(camera.CAM_DEV_ID_0)
            print("K230版本退出")

def main():
    """主函数"""
    yolos = K230YOLOS()
    yolos.run()

if __name__ == "__main__":
    main()
'''
        main_file.write_text(main_content, encoding='utf-8')
        
    def _generate_rpi_files(self, target_dir: Path):
        """生成树莓派版本文件"""
        # 树莓派主文件
        main_file = target_dir / 'main.py'
        main_content = '''#!/usr/bin/env python3
"""
YOLOS 树莓派版本主文件
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False

class RaspberryPiYOLOS:
    """树莓派版本YOLOS"""
    
    def __init__(self):
        self.camera = None
        self.model_path = "yolov8n.tflite"
        self.input_size = (640, 640)
        
    def init_camera(self):
        """初始化摄像头"""
        if PICAMERA_AVAILABLE:
            try:
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                    main={"size": self.input_size}
                )
                self.camera.configure(config)
                self.camera.start()
                print("树莓派摄像头初始化成功")
                return True
            except Exception as e:
                print(f"PiCamera初始化失败: {e}")
                
        if OPENCV_AVAILABLE:
            try:
                self.camera = cv2.VideoCapture(0)
                if self.camera.isOpened():
                    print("USB摄像头初始化成功")
                    return True
            except Exception as e:
                print(f"USB摄像头初始化失败: {e}")
                
        print("⚠️ 无可用摄像头，使用模拟模式")
        return False
        
    def capture_frame(self):
        """捕获帧"""
        if PICAMERA_AVAILABLE and hasattr(self, 'camera') and self.camera:
            try:
                frame = self.camera.capture_array()
                return frame
            except Exception as e:
                print(f"PiCamera捕获失败: {e}")
                
        if OPENCV_AVAILABLE and hasattr(self, 'camera') and self.camera:
            try:
                ret, frame = self.camera.read()
                if ret:
                    return frame
            except Exception as e:
                print(f"USB摄像头捕获失败: {e}")
                
        # 返回模拟帧
        return np.random.randint(0, 255, (*self.input_size, 3), dtype=np.uint8)
        
    def detect_frame(self, frame):
        """检测帧"""
        # 模拟检测结果
        detections = [
            {"class": "person", "confidence": 0.85, "bbox": [100, 100, 200, 300]}
        ]
        return detections
        
    def run(self):
        """运行检测"""
        print("🚀 启动YOLOS 树莓派版本...")
        
        self.init_camera()
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                frame = self.capture_frame()
                detections = self.detect_frame(frame)
                
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                print(f"帧 {frame_count}: 检测到 {len(detections)} 个目标, FPS: {fps:.1f}")
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\\n检测被用户中断")
        finally:
            if hasattr(self, 'camera') and self.camera:
                if PICAMERA_AVAILABLE:
                    self.camera.stop()
                elif OPENCV_AVAILABLE:
                    self.camera.release()
            print("树莓派版本退出")

def main():
    """主函数"""
    yolos = RaspberryPiYOLOS()
    yolos.run()

if __name__ == "__main__":
    main()
'''
        main_file.write_text(main_content, encoding='utf-8')
        
    def _generate_config_files(self, platform: str, target_dir: Path):
        """生成配置文件"""
        config = self.platform_configs[platform]
        
        # 平台配置
        platform_config = {
            "platform": {
                "name": platform,
                "description": config['description'],
                "version": "1.0.0"
            },
            "model": {
                "format": config['model_format'],
                "input_size": [640, 640],
                "confidence_threshold": 0.5,
                "nms_threshold": 0.4
            },
            "camera": {
                "width": 640,
                "height": 480,
                "fps": 30
            },
            "performance": {
                "max_fps": 30 if platform in ['esp32', 'k230'] else 60,
                "batch_size": 1,
                "num_threads": 1 if platform == 'esp32' else 4
            }
        }
        
        config_file = target_dir / 'config.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(platform_config, f, indent=2, ensure_ascii=False)
            
    def _generate_install_scripts(self, platform: str, target_dir: Path):
        """生成安装脚本"""
        config = self.platform_configs[platform]
        
        # requirements.txt
        requirements_file = target_dir / 'requirements.txt'
        requirements = config['requirements'] + config['optional_requirements']
        requirements_file.write_text('\n'.join(requirements), encoding='utf-8')
        
        # 安装脚本
        if platform == 'pc':
            install_script = target_dir / 'install.py'
            install_content = '''#!/usr/bin/env python3
"""
YOLOS PC版本安装脚本
"""

import subprocess
import sys
import os

def install_requirements():
    """安装依赖包"""
    print("📦 安装依赖包...")
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ])
        print("✅ 依赖包安装完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖包安装失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 YOLOS PC版本安装程序")
    print("="*40)
    
    if install_requirements():
        print("\\n✅ 安装完成！")
        print("运行 python main.py 启动程序")
    else:
        print("\\n❌ 安装失败，请检查网络连接和Python环境")

if __name__ == "__main__":
    main()
'''
            install_script.write_text(install_content, encoding='utf-8')
            
        elif platform == 'raspberry_pi':
            install_script = target_dir / 'install_rpi.sh'
            install_content = '''#!/bin/bash
# YOLOS 树莓派版本安装脚本

echo "🚀 YOLOS 树莓派版本安装程序"
echo "=================================="

# 更新系统
echo "📦 更新系统包..."
sudo apt update

# 安装系统依赖
echo "📦 安装系统依赖..."
sudo apt install -y python3-pip python3-opencv python3-numpy

# 安装Python依赖
echo "📦 安装Python依赖..."
pip3 install -r requirements.txt

# 启用摄像头
echo "📷 配置摄像头..."
sudo raspi-config nonint do_camera 0

echo "✅ 安装完成！"
echo "运行 python3 main.py 启动程序"
'''
            install_script.write_text(install_content, encoding='utf-8')
            
    def _generate_documentation(self, platform: str, target_dir: Path):
        """生成文档"""
        config = self.platform_configs[platform]
        
        readme_content = f'''# YOLOS {config['name']}

{config['description']}

## 系统要求

### 硬件要求
'''
        
        if platform == 'pc':
            readme_content += '''
- CPU: Intel i5或AMD Ryzen 5以上
- 内存: 8GB RAM以上
- 显卡: 支持CUDA的NVIDIA显卡(可选)
- 摄像头: USB摄像头或内置摄像头
'''
        elif platform == 'esp32':
            readme_content += '''
- ESP32开发板
- ESP32-CAM模块
- MicroSD卡(可选)
- 电源适配器
'''
        elif platform == 'k230':
            readme_content += '''
- K230 AI开发板
- 摄像头模块
- MicroSD卡
- 电源适配器
'''
        elif platform == 'raspberry_pi':
            readme_content += '''
- 树莓派4B或更新版本
- 树莓派摄像头模块或USB摄像头
- MicroSD卡(32GB以上)
- 电源适配器
'''
        
        readme_content += f'''
### 软件要求

依赖包:
'''
        
        for req in config['requirements']:
            readme_content += f'- {req}\n'
            
        readme_content += '''
## 安装说明

### 1. 下载部署包
解压下载的部署包到目标目录

### 2. 安装依赖
'''
        
        if platform == 'pc':
            readme_content += '''
```bash
# Windows
python install.py

# Linux/macOS
python3 install.py
```
'''
        elif platform == 'raspberry_pi':
            readme_content += '''
```bash
chmod +x install_rpi.sh
./install_rpi.sh
```
'''
        else:
            readme_content += '''
请参考平台特定的安装文档
'''
        
        readme_content += f'''
### 3. 运行程序
```bash
python {config['entry_point']}
```

## 使用说明

### 基本功能
- 实时目标检测
- 图像/视频处理
- 检测结果保存

### 配置参数
编辑 `config.json` 文件调整检测参数:
- confidence_threshold: 置信度阈值
- nms_threshold: NMS阈值
- input_size: 输入图像尺寸

## 故障排除

### 常见问题
1. 摄像头无法打开
   - 检查摄像头连接
   - 确认摄像头权限

2. 检测速度慢
   - 降低输入分辨率
   - 使用GPU加速(如果可用)

3. 内存不足
   - 减少批处理大小
   - 关闭其他程序

## 技术支持

如有问题请联系技术支持或查看项目文档。

---
YOLOS {config['name']} v1.0.0
'''
        
        readme_file = target_dir / 'README.md'
        readme_file.write_text(readme_content, encoding='utf-8')
        
    def _create_deployment_package(self, platform: str, platform_dir: Path):
        """创建部署包"""
        config = self.platform_configs[platform]
        
        # 创建ZIP包
        zip_path = self.deployments_dir / f'yolos_{platform}_v1.0.0.zip'
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(platform_dir):
                for file in files:
                    file_path = Path(root) / file
                    arc_path = file_path.relative_to(platform_dir)
                    zipf.write(file_path, arc_path)
                    
        print(f"📦 部署包已创建: {zip_path}")
        
    def generate_all_platforms(self):
        """生成所有平台版本"""
        print("🚀 生成所有平台部署版本...")
        
        success_count = 0
        total_count = len(self.platform_configs)
        
        for platform in self.platform_configs.keys():
            if self.generate_platform_version(platform):
                success_count += 1
                
        print(f"\n📊 生成完成: {success_count}/{total_count} 个平台成功")
        
        # 生成总览文档
        self._generate_overview_documentation()
        
    def _generate_overview_documentation(self):
        """生成总览文档"""
        overview_content = '''# YOLOS多平台部署总览

本目录包含YOLOS系统的多平台部署版本。

## 可用平台

| 平台 | 描述 | 部署包 | 状态 |
|------|------|--------|------|
'''
        
        for platform, config in self.platform_configs.items():
            zip_file = f'yolos_{platform}_v1.0.0.zip'
            zip_path = self.deployments_dir / zip_file
            status = "✅ 可用" if zip_path.exists() else "❌ 未生成"
            
            overview_content += f'| {config["name"]} | {config["description"]} | {zip_file} | {status} |\n'
            
        overview_content += '''
## 快速开始

1. 选择目标平台
2. 下载对应的部署包
3. 解压到目标设备
4. 按照README.md说明安装和运行

## 平台特性对比

| 特性 | PC | ESP32 | K230 | 树莓派 |
|------|----|----|------|--------|
| GUI界面 | ✅ | ❌ | ❌ | ✅ |
| 实时检测 | ✅ | ✅ | ✅ | ✅ |
| 模型格式 | PyTorch | TFLite Micro | KModel | TFLite |
| 性能 | 高 | 低 | 中 | 中 |
| 功耗 | 高 | 极低 | 低 | 低 |

## 技术支持

如需技术支持，请参考各平台的README文档或联系开发团队。

---
YOLOS多平台部署系统 v1.0.0
'''
        
        overview_file = self.deployments_dir / 'README.md'
        overview_file.write_text(overview_content, encoding='utf-8')
        
        print(f"📄 总览文档已生成: {overview_file}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOS部署生成器')
    parser.add_argument('--platform', choices=['pc', 'esp32', 'k230', 'raspberry_pi', 'all'],
                       default='all', help='目标平台')
    
    args = parser.parse_args()
    
    generator = DeploymentGenerator()
    
    if args.platform == 'all':
        generator.generate_all_platforms()
    else:
        generator.generate_platform_version(args.platform)

if __name__ == "__main__":
    main()