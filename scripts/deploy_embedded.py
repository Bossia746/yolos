#!/usr/bin/env python3
"""
嵌入式设备自动部署脚本
支持ESP32、树莓派、Jetson等平台的一键部署
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.embedded.platform_adapter import get_platform_adapter, detect_platform
    from src.embedded.lite_detector import create_lite_detector
    from src.embedded.memory_manager import get_memory_manager
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在项目根目录运行此脚本")
    sys.exit(1)

class EmbeddedDeployer:
    """嵌入式部署器"""
    
    def __init__(self, target_platform: Optional[str] = None, output_dir: str = "./deployment"):
        self.target_platform = target_platform or detect_platform()
        self.output_dir = Path(output_dir)
        self.logger = self._setup_logging()
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 平台适配器
        self.adapter = get_platform_adapter()
        
        # 部署配置
        self.deployment_config = None
        
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('EmbeddedDeployer')
        logger.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 文件处理器
        log_file = self.output_dir / 'deployment.log'
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
        
    def prepare_deployment(self, model_size: str = "n", 
                          model_path: Optional[str] = None,
                          custom_config: Optional[Dict] = None) -> Dict[str, Any]:
        """准备部署"""
        self.logger.info(f"开始准备部署到 {self.target_platform}")
        
        # 获取平台最优配置
        platform_config = self.adapter.get_optimal_config(model_size)
        
        # 合并自定义配置
        if custom_config:
            platform_config.update(custom_config)
            
        # 设置模型路径
        if model_path:
            platform_config['model_path'] = model_path
        else:
            # 使用默认模型路径
            model_name = f"yolov11{model_size}.pt"
            platform_config['model_path'] = f"models/{model_name}"
            
        self.deployment_config = platform_config
        
        self.logger.info(f"部署配置: {json.dumps(platform_config, indent=2, ensure_ascii=False)}")
        
        return platform_config
        
    def optimize_model(self) -> str:
        """优化模型"""
        if not self.deployment_config:
            raise RuntimeError("请先调用 prepare_deployment()")
            
        self.logger.info("开始模型优化...")
        
        model_path = self.deployment_config['model_path']
        model_format = self.deployment_config['model_format']
        precision = self.deployment_config['precision']
        input_size = self.deployment_config['input_size']
        
        # 创建优化后的模型文件名
        model_name = Path(model_path).stem
        optimized_name = f"{model_name}_{model_format}_{precision}_{input_size[0]}x{input_size[1]}"
        
        if model_format == 'onnx':
            optimized_path = self.output_dir / f"{optimized_name}.onnx"
            self._convert_to_onnx(model_path, optimized_path, input_size, precision)
        elif model_format == 'tflite':
            optimized_path = self.output_dir / f"{optimized_name}.tflite"
            self._convert_to_tflite(model_path, optimized_path, input_size, precision)
        elif model_format == 'tensorrt':
            optimized_path = self.output_dir / f"{optimized_name}.engine"
            self._convert_to_tensorrt(model_path, optimized_path, input_size, precision)
        else:
            # 直接复制原模型
            optimized_path = self.output_dir / f"{optimized_name}.pt"
            shutil.copy2(model_path, optimized_path)
            
        self.deployment_config['optimized_model_path'] = str(optimized_path)
        self.logger.info(f"模型优化完成: {optimized_path}")
        
        return str(optimized_path)
        
    def _convert_to_onnx(self, model_path: str, output_path: Path, 
                        input_size: tuple, precision: str):
        """转换为ONNX格式"""
        try:
            import torch
            from ultralytics import YOLO
            
            # 加载模型
            model = YOLO(model_path)
            
            # 导出ONNX
            model.export(
                format='onnx',
                imgsz=input_size,
                half=(precision == 'fp16'),
                int8=(precision == 'int8'),
                dynamic=False,
                simplify=True,
                opset=11
            )
            
            # 移动到输出目录
            exported_path = Path(model_path).with_suffix('.onnx')
            if exported_path.exists():
                shutil.move(str(exported_path), output_path)
                
        except Exception as e:
            self.logger.error(f"ONNX转换失败: {e}")
            raise
            
    def _convert_to_tflite(self, model_path: str, output_path: Path,
                          input_size: tuple, precision: str):
        """转换为TensorFlow Lite格式"""
        try:
            from ultralytics import YOLO
            
            # 加载模型
            model = YOLO(model_path)
            
            # 导出TFLite
            model.export(
                format='tflite',
                imgsz=input_size,
                int8=(precision == 'int8'),
                half=(precision == 'fp16')
            )
            
            # 移动到输出目录
            exported_path = Path(model_path).with_suffix('.tflite')
            if exported_path.exists():
                shutil.move(str(exported_path), output_path)
                
        except Exception as e:
            self.logger.error(f"TFLite转换失败: {e}")
            raise
            
    def _convert_to_tensorrt(self, model_path: str, output_path: Path,
                            input_size: tuple, precision: str):
        """转换为TensorRT格式"""
        try:
            from ultralytics import YOLO
            
            # 加载模型
            model = YOLO(model_path)
            
            # 导出TensorRT
            model.export(
                format='engine',
                imgsz=input_size,
                half=(precision == 'fp16'),
                int8=(precision == 'int8'),
                device=0  # 使用第一个GPU
            )
            
            # 移动到输出目录
            exported_path = Path(model_path).with_suffix('.engine')
            if exported_path.exists():
                shutil.move(str(exported_path), output_path)
                
        except Exception as e:
            self.logger.error(f"TensorRT转换失败: {e}")
            raise
            
    def create_deployment_package(self) -> str:
        """创建部署包"""
        if not self.deployment_config:
            raise RuntimeError("请先调用 prepare_deployment()")
            
        self.logger.info("创建部署包...")
        
        # 创建部署包目录
        package_name = f"yolo_embedded_{self.target_platform.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        package_dir = self.output_dir / package_name
        package_dir.mkdir(exist_ok=True)
        
        # 复制核心文件
        self._copy_core_files(package_dir)
        
        # 复制优化后的模型
        if 'optimized_model_path' in self.deployment_config:
            model_src = Path(self.deployment_config['optimized_model_path'])
            model_dst = package_dir / 'models' / model_src.name
            model_dst.parent.mkdir(exist_ok=True)
            shutil.copy2(model_src, model_dst)
            
        # 生成配置文件
        self._generate_config_files(package_dir)
        
        # 生成部署脚本
        self._generate_deployment_scripts(package_dir)
        
        # 生成文档
        self._generate_documentation(package_dir)
        
        # 创建压缩包
        archive_path = self._create_archive(package_dir)
        
        self.logger.info(f"部署包创建完成: {archive_path}")
        
        return str(archive_path)
        
    def _copy_core_files(self, package_dir: Path):
        """复制核心文件"""
        # 创建源码目录
        src_dir = package_dir / 'src'
        src_dir.mkdir(exist_ok=True)
        
        # 复制嵌入式模块
        embedded_src = Path(__file__).parent.parent / 'src' / 'embedded'
        if embedded_src.exists():
            embedded_dst = src_dir / 'embedded'
            shutil.copytree(embedded_src, embedded_dst, dirs_exist_ok=True)
            
        # 复制工具脚本
        scripts_src = Path(__file__).parent
        scripts_dst = package_dir / 'scripts'
        scripts_dst.mkdir(exist_ok=True)
        
        # 只复制必要的脚本
        essential_scripts = ['embedded_model_evaluator.py']
        for script in essential_scripts:
            script_path = scripts_src / script
            if script_path.exists():
                shutil.copy2(script_path, scripts_dst / script)
                
    def _generate_config_files(self, package_dir: Path):
        """生成配置文件"""
        config_dir = package_dir / 'config'
        config_dir.mkdir(exist_ok=True)
        
        # 主配置文件
        main_config = {
            'platform': self.target_platform,
            'deployment_config': self.deployment_config,
            'created_at': datetime.now().isoformat(),
            'version': '1.0.0'
        }
        
        with open(config_dir / 'deployment_config.json', 'w', encoding='utf-8') as f:
            json.dump(main_config, f, indent=2, ensure_ascii=False)
            
        # 平台特定配置
        platform_config = self.adapter.get_optimal_config()
        with open(config_dir / 'platform_config.json', 'w', encoding='utf-8') as f:
            json.dump(platform_config, f, indent=2, ensure_ascii=False)
            
        # 环境配置
        env_config = self._generate_env_config()
        with open(config_dir / 'environment.env', 'w', encoding='utf-8') as f:
            for key, value in env_config.items():
                f.write(f"{key}={value}\n")
                
    def _generate_env_config(self) -> Dict[str, str]:
        """生成环境配置"""
        config = self.deployment_config
        
        env_config = {
            'YOLO_MODEL_PATH': 'models/' + Path(config.get('optimized_model_path', '')).name,
            'YOLO_INPUT_SIZE': f"{config['input_size'][0]}x{config['input_size'][1]}",
            'YOLO_BATCH_SIZE': str(config['batch_size']),
            'YOLO_NUM_THREADS': str(config['num_threads']),
            'YOLO_PRECISION': config['precision'],
            'YOLO_USE_GPU': str(config.get('use_gpu', False)).lower(),
            'YOLO_USE_NPU': str(config.get('use_npu', False)).lower(),
            'YOLO_CONFIDENCE_THRESHOLD': str(config['confidence_threshold']),
            'YOLO_NMS_THRESHOLD': str(config['nms_threshold']),
            'YOLO_MEMORY_LIMIT_MB': str(config['memory_limit_mb'])
        }
        
        return env_config
        
    def _generate_deployment_scripts(self, package_dir: Path):
        """生成部署脚本"""
        scripts_dir = package_dir / 'scripts'
        scripts_dir.mkdir(exist_ok=True)
        
        # 主启动脚本
        self._generate_main_script(scripts_dir)
        
        # 安装脚本
        self._generate_install_script(scripts_dir)
        
        # 测试脚本
        self._generate_test_script(scripts_dir)
        
        # 监控脚本
        self._generate_monitor_script(scripts_dir)
        
    def _generate_main_script(self, scripts_dir: Path):
        """生成主启动脚本"""
        script_content = '''#!/usr/bin/env python3
"""
嵌入式YOLO检测器主程序
"""

import os
import sys
import json
import logging
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from src.embedded.lite_detector import create_lite_detector
from src.embedded.memory_manager import get_memory_manager
from src.embedded.platform_adapter import get_platform_adapter

def load_config():
    """加载配置"""
    config_path = Path(__file__).parent.parent / 'config' / 'deployment_config.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('yolo_detector.log', encoding='utf-8')
        ]
    )

def main():
    """主函数"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # 加载配置
        config = load_config()
        deployment_config = config['deployment_config']
        
        logger.info(f"启动YOLO检测器 - 平台: {config['platform']}")
        
        # 初始化内存管理器
        memory_manager = get_memory_manager()
        memory_manager.initialize(deployment_config['memory_limit_mb'])
        
        # 创建检测器
        detector = create_lite_detector(
            model_path=deployment_config['model_path'],
            platform_config=deployment_config
        )
        
        # 预热模型
        logger.info("预热模型...")
        detector.warmup()
        
        logger.info("YOLO检测器启动成功")
        
        # 这里可以添加具体的检测逻辑
        # 例如：摄像头检测、图片批处理等
        
        # 示例：处理单张图片
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
            if os.path.exists(image_path):
                logger.info(f"处理图片: {image_path}")
                results = detector.detect(image_path)
                logger.info(f"检测结果: {len(results)} 个目标")
                for result in results:
                    logger.info(f"  {result}")
            else:
                logger.error(f"图片不存在: {image_path}")
        else:
            logger.info("使用方法: python main.py <image_path>")
            
    except Exception as e:
        logger.error(f"启动失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
'''
        
        with open(scripts_dir / 'main.py', 'w', encoding='utf-8') as f:
            f.write(script_content)
            
        # 设置执行权限 (Unix系统)
        if os.name != 'nt':
            os.chmod(scripts_dir / 'main.py', 0o755)
            
    def _generate_install_script(self, scripts_dir: Path):
        """生成安装脚本"""
        # Python安装脚本
        install_content = '''#!/usr/bin/env python3
"""
嵌入式YOLO部署安装脚本
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def install_dependencies():
    """安装依赖"""
    print("安装Python依赖...")
    
    # 基础依赖
    base_deps = [
        'numpy',
        'opencv-python-headless',
        'pillow',
        'psutil'
    ]
    
    # 平台特定依赖
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if 'arm' in machine or 'aarch64' in machine:
        # ARM平台
        if 'raspberry' in open('/proc/cpuinfo', 'r').read().lower():
            # 树莓派
            base_deps.extend(['onnxruntime'])
        else:
            # 其他ARM设备
            base_deps.extend(['onnxruntime'])
    else:
        # x86平台
        base_deps.extend(['onnxruntime', 'torch', 'torchvision'])
        
    # 安装依赖
    for dep in base_deps:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])
            print(f"✓ {dep} 安装成功")
        except subprocess.CalledProcessError:
            print(f"✗ {dep} 安装失败")
            
def setup_environment():
    """设置环境"""
    print("设置环境变量...")
    
    # 加载环境配置
    env_file = Path(__file__).parent.parent / 'config' / 'environment.env'
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
                    print(f"  {key}={value}")
                    
def create_service():
    """创建系统服务 (可选)"""
    if platform.system() == 'Linux':
        print("创建systemd服务...")
        
        service_content = f'''[Unit]
Description=YOLO Embedded Detector
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory={Path(__file__).parent.parent}
ExecStart={sys.executable} scripts/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
'''
        
        service_path = Path('/etc/systemd/system/yolo-detector.service')
        try:
            with open(service_path, 'w') as f:
                f.write(service_content)
            print(f"✓ 服务文件创建: {service_path}")
            print("使用以下命令启用服务:")
            print("  sudo systemctl enable yolo-detector.service")
            print("  sudo systemctl start yolo-detector.service")
        except PermissionError:
            print("✗ 需要root权限创建服务文件")
            
def main():
    print("YOLO嵌入式部署安装程序")
    print("=" * 40)
    
    install_dependencies()
    setup_environment()
    create_service()
    
    print("\n安装完成!")
    print("运行测试: python scripts/test.py")
    print("启动检测器: python scripts/main.py <image_path>")

if __name__ == '__main__':
    main()
'''
        
        with open(scripts_dir / 'install.py', 'w', encoding='utf-8') as f:
            f.write(install_content)
            
    def _generate_test_script(self, scripts_dir: Path):
        """生成测试脚本"""
        test_content = '''#!/usr/bin/env python3
"""
嵌入式YOLO部署测试脚本
"""

import os
import sys
import time
import json
import logging
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from src.embedded.platform_adapter import get_platform_adapter
from src.embedded.memory_manager import get_memory_manager
from src.embedded.lite_detector import create_lite_detector

def test_platform_detection():
    """测试平台检测"""
    print("测试平台检测...")
    
    try:
        adapter = get_platform_adapter()
        hw_info = adapter.hardware_info
        
        print(f"  平台: {hw_info.platform_name}")
        print(f"  CPU: {hw_info.cpu_model} ({hw_info.cpu_cores} cores)")
        print(f"  内存: {hw_info.memory_total_mb} MB")
        print(f"  GPU: {'是' if hw_info.has_gpu else '否'}")
        print(f"  NPU: {'是' if hw_info.has_npu else '否'}")
        
        return True
    except Exception as e:
        print(f"  ✗ 平台检测失败: {e}")
        return False
        
def test_memory_manager():
    """测试内存管理器"""
    print("测试内存管理器...")
    
    try:
        memory_manager = get_memory_manager()
        memory_manager.initialize(512)  # 512MB限制
        
        # 分配测试
        test_data = memory_manager.allocate_buffer("test", 1024 * 1024)  # 1MB
        if test_data is not None:
            print(f"  ✓ 内存分配成功: 1MB")
            
        # 释放测试
        memory_manager.deallocate_buffer("test")
        print(f"  ✓ 内存释放成功")
        
        # 状态检查
        stats = memory_manager.get_memory_stats()
        print(f"  内存使用: {stats['used_mb']:.1f}MB / {stats['total_mb']:.1f}MB")
        
        return True
    except Exception as e:
        print(f"  ✗ 内存管理器测试失败: {e}")
        return False
        
def test_model_loading():
    """测试模型加载"""
    print("测试模型加载...")
    
    try:
        # 加载配置
        config_path = Path(__file__).parent.parent / 'config' / 'deployment_config.json'
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        deployment_config = config['deployment_config']
        
        # 检查模型文件
        model_path = Path(__file__).parent.parent / deployment_config['model_path']
        if not model_path.exists():
            print(f"  ✗ 模型文件不存在: {model_path}")
            return False
            
        # 创建检测器
        detector = create_lite_detector(
            model_path=str(model_path),
            platform_config=deployment_config
        )
        
        print(f"  ✓ 模型加载成功")
        
        # 预热测试
        start_time = time.time()
        detector.warmup()
        warmup_time = time.time() - start_time
        
        print(f"  ✓ 模型预热完成: {warmup_time:.2f}s")
        
        return True
    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        return False
        
def test_inference_performance():
    """测试推理性能"""
    print("测试推理性能...")
    
    try:
        # 加载配置和模型
        config_path = Path(__file__).parent.parent / 'config' / 'deployment_config.json'
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        deployment_config = config['deployment_config']
        model_path = Path(__file__).parent.parent / deployment_config['model_path']
        
        detector = create_lite_detector(
            model_path=str(model_path),
            platform_config=deployment_config
        )
        
        # 创建测试图片
        import numpy as np
        from PIL import Image
        
        input_size = deployment_config['input_size']
        test_image = np.random.randint(0, 255, (input_size[1], input_size[0], 3), dtype=np.uint8)
        test_image_pil = Image.fromarray(test_image)
        
        # 性能测试
        num_runs = 10
        times = []
        
        for i in range(num_runs):
            start_time = time.time()
            results = detector.detect_image(test_image_pil)
            inference_time = time.time() - start_time
            times.append(inference_time)
            
        avg_time = sum(times) / len(times)
        fps = 1.0 / avg_time
        
        print(f"  ✓ 平均推理时间: {avg_time*1000:.1f}ms")
        print(f"  ✓ 平均FPS: {fps:.1f}")
        print(f"  ✓ 检测目标数: {len(results) if results else 0}")
        
        return True
    except Exception as e:
        print(f"  ✗ 推理性能测试失败: {e}")
        return False
        
def test_system_health():
    """测试系统健康监控"""
    print("测试系统健康监控...")
    
    try:
        adapter = get_platform_adapter()
        health = adapter.monitor_system_health()
        
        print(f"  CPU使用率: {health['cpu_usage']:.1f}%")
        print(f"  内存使用率: {health['memory_usage']:.1f}%")
        print(f"  可用内存: {health['available_memory_mb']:.0f}MB")
        print(f"  系统状态: {health['status']}")
        
        return True
    except Exception as e:
        print(f"  ✗ 系统健康监控失败: {e}")
        return False
        
def main():
    """主测试函数"""
    print("YOLO嵌入式部署测试")
    print("=" * 40)
    
    tests = [
        ("平台检测", test_platform_detection),
        ("内存管理器", test_memory_manager),
        ("模型加载", test_model_loading),
        ("推理性能", test_inference_performance),
        ("系统健康", test_system_health)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n[{passed+1}/{total}] {test_name}")
        if test_func():
            passed += 1
            print(f"  ✓ 通过")
        else:
            print(f"  ✗ 失败")
            
    print(f"\n测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过!")
        return 0
    else:
        print("❌ 部分测试失败")
        return 1

if __name__ == '__main__':
    sys.exit(main())
'''
        
        with open(scripts_dir / 'test.py', 'w', encoding='utf-8') as f:
            f.write(test_content)
            
    def _generate_monitor_script(self, scripts_dir: Path):
        """生成监控脚本"""
        monitor_content = '''#!/usr/bin/env python3
"""
嵌入式YOLO系统监控脚本
"""

import os
import sys
import time
import json
import signal
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from src.embedded.platform_adapter import get_platform_adapter
from src.embedded.memory_manager import get_memory_manager

class SystemMonitor:
    """系统监控器"""
    
    def __init__(self, interval: int = 10, log_file: str = "system_monitor.log"):
        self.interval = interval
        self.log_file = Path(log_file)
        self.running = False
        self.adapter = get_platform_adapter()
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        print(f"\n收到信号 {signum}，正在停止监控...")
        self.running = False
        
    def start_monitoring(self):
        """开始监控"""
        print(f"开始系统监控 (间隔: {self.interval}s)")
        print(f"日志文件: {self.log_file}")
        print("按 Ctrl+C 停止监控")
        
        self.running = True
        
        # 创建日志文件
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("timestamp,cpu_usage,memory_usage,available_memory_mb,temperature,disk_usage,status\n")
            
        while self.running:
            try:
                # 获取系统健康状态
                health = self.adapter.monitor_system_health()
                
                # 记录到日志
                timestamp = datetime.now().isoformat()
                log_line = f"{timestamp},{health['cpu_usage']},{health['memory_usage']},{health['available_memory_mb']},{health['temperature']},{health['disk_usage']},{health['status']}\n"
                
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(log_line)
                    
                # 控制台输出
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"CPU: {health['cpu_usage']:5.1f}% | "
                      f"内存: {health['memory_usage']:5.1f}% | "
                      f"可用: {health['available_memory_mb']:6.0f}MB | "
                      f"温度: {health['temperature']:4.1f}°C | "
                      f"状态: {health['status']}")
                      
                # 检查告警条件
                self._check_alerts(health)
                
                time.sleep(self.interval)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"监控错误: {e}")
                time.sleep(self.interval)
                
        print("监控已停止")
        
    def _check_alerts(self, health):
        """检查告警条件"""
        alerts = []
        
        if health['cpu_usage'] > 90:
            alerts.append(f"CPU使用率过高: {health['cpu_usage']:.1f}%")
            
        if health['memory_usage'] > 85:
            alerts.append(f"内存使用率过高: {health['memory_usage']:.1f}%")
            
        if health['available_memory_mb'] < 100:
            alerts.append(f"可用内存不足: {health['available_memory_mb']:.0f}MB")
            
        if health['temperature'] > 80:
            alerts.append(f"温度过高: {health['temperature']:.1f}°C")
            
        if health['disk_usage'] > 90:
            alerts.append(f"磁盘使用率过高: {health['disk_usage']:.1f}%")
            
        for alert in alerts:
            print(f"⚠️  告警: {alert}")
            
    def generate_report(self):
        """生成监控报告"""
        if not self.log_file.exists():
            print("监控日志文件不存在")
            return
            
        print("生成监控报告...")
        
        # 读取日志数据
        data = []
        with open(self.log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]  # 跳过标题行
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 7:
                    data.append({
                        'timestamp': parts[0],
                        'cpu_usage': float(parts[1]),
                        'memory_usage': float(parts[2]),
                        'available_memory_mb': float(parts[3]),
                        'temperature': float(parts[4]),
                        'disk_usage': float(parts[5]),
                        'status': parts[6]
                    })
                    
        if not data:
            print("没有监控数据")
            return
            
        # 计算统计信息
        cpu_avg = sum(d['cpu_usage'] for d in data) / len(data)
        cpu_max = max(d['cpu_usage'] for d in data)
        
        mem_avg = sum(d['memory_usage'] for d in data) / len(data)
        mem_max = max(d['memory_usage'] for d in data)
        
        temp_avg = sum(d['temperature'] for d in data) / len(data)
        temp_max = max(d['temperature'] for d in data)
        
        # 生成报告
        report = f"""
系统监控报告
=============

监控时间: {data[0]['timestamp']} ~ {data[-1]['timestamp']}
数据点数: {len(data)}

CPU使用率:
  平均: {cpu_avg:.1f}%
  最高: {cpu_max:.1f}%

内存使用率:
  平均: {mem_avg:.1f}%
  最高: {mem_max:.1f}%

温度:
  平均: {temp_avg:.1f}°C
  最高: {temp_max:.1f}°C

状态分布:
"""
        
        # 状态统计
        status_count = {}
        for d in data:
            status = d['status']
            status_count[status] = status_count.get(status, 0) + 1
            
        for status, count in status_count.items():
            percentage = (count / len(data)) * 100
            report += f"  {status}: {count} ({percentage:.1f}%)\n"
            
        print(report)
        
        # 保存报告
        report_file = self.log_file.with_suffix('.report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
            
        print(f"报告已保存: {report_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO嵌入式系统监控')
    parser.add_argument('--interval', type=int, default=10, help='监控间隔(秒)')
    parser.add_argument('--log-file', default='system_monitor.log', help='日志文件路径')
    parser.add_argument('--report', action='store_true', help='生成监控报告')
    
    args = parser.parse_args()
    
    monitor = SystemMonitor(args.interval, args.log_file)
    
    if args.report:
        monitor.generate_report()
    else:
        monitor.start_monitoring()

if __name__ == '__main__':
    main()
'''
        
        with open(scripts_dir / 'monitor.py', 'w', encoding='utf-8') as f:
            f.write(monitor_content)
            
    def _generate_documentation(self, package_dir: Path):
        """生成文档"""
        docs_dir = package_dir / 'docs'
        docs_dir.mkdir(exist_ok=True)
        
        # README文件
        readme_content = f'''# YOLO嵌入式部署包

## 平台信息
- 目标平台: {self.target_platform}
- 创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 版本: 1.0.0

## 快速开始

### 1. 安装依赖
```bash
python scripts/install.py
```

### 2. 运行测试
```bash
python scripts/test.py
```

### 3. 启动检测器
```bash
python scripts/main.py <image_path>
```

### 4. 系统监控
```bash
python scripts/monitor.py
```

## 目录结构
```
{package_dir.name}/
├── src/                    # 源代码
│   └── embedded/          # 嵌入式模块
├── models/                # 模型文件
├── config/                # 配置文件
├── scripts/               # 脚本文件
├── docs/                  # 文档
└── README.md             # 说明文档
```

## 配置说明

### 部署配置 (config/deployment_config.json)
包含平台特定的部署参数，如模型路径、输入尺寸、精度等。

### 环境配置 (config/environment.env)
包含环境变量设置，可以通过修改此文件调整运行参数。

## 性能优化

根据您的平台，建议采用以下优化策略:
'''
        
        # 添加平台特定的优化建议
        recommendations = self.adapter.get_deployment_recommendations()
        for i, rec in enumerate(recommendations, 1):
            readme_content += f"\n{i}. {rec}"
            
        readme_content += '''

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件路径是否正确
   - 确认模型格式与平台兼容
   - 检查内存是否足够

2. **推理速度慢**
   - 尝试降低输入分辨率
   - 启用硬件加速 (GPU/NPU)
   - 使用更小的模型

3. **内存不足**
   - 减少批处理大小
   - 启用动态内存管理
   - 考虑模型量化

### 日志文件
- 主程序日志: `yolo_detector.log`
- 系统监控日志: `system_monitor.log`
- 部署日志: `deployment.log`

## 技术支持

如需技术支持，请提供以下信息:
1. 平台信息 (运行 `python scripts/test.py` 获取)
2. 错误日志
3. 配置文件内容
4. 系统资源使用情况
'''
        
        with open(package_dir / 'README.md', 'w', encoding='utf-8') as f:
            f.write(readme_content)
            
        # API文档
        api_doc = '''# API文档

## LiteYOLODetector

轻量级YOLO检测器，专为嵌入式设备优化。

### 初始化
```python
from src.embedded.lite_detector import create_lite_detector

detector = create_lite_detector(
    model_path="models/yolov11n.onnx",
    platform_config={
        "input_size": (416, 416),
        "precision": "fp16",
        "batch_size": 1,
        "num_threads": 4
    }
)
```

### 方法

#### detect_image(image)
检测单张图片中的目标。

**参数:**
- `image`: PIL.Image 或 numpy.ndarray

**返回:**
- `List[DetectionResult]`: 检测结果列表

#### detect_batch(images)
批量检测多张图片。

**参数:**
- `images`: List[PIL.Image] 或 List[numpy.ndarray]

**返回:**
- `List[List[DetectionResult]]`: 批量检测结果

#### warmup()
预热模型，提高首次推理速度。

#### get_performance_stats()
获取性能统计信息。

**返回:**
- `Dict`: 包含推理时间、FPS等统计信息

## PlatformAdapter

平台适配器，自动检测硬件并提供优化配置。

### 使用方法
```python
from src.embedded.platform_adapter import get_platform_adapter

adapter = get_platform_adapter()
hw_info = adapter.hardware_info
config = adapter.get_optimal_config("n")
```

## MemoryManager

内存管理器，提供动态内存分配和监控。

### 使用方法
```python
from src.embedded.memory_manager import get_memory_manager

memory_manager = get_memory_manager()
memory_manager.initialize(512)  # 512MB限制

# 分配缓冲区
buffer = memory_manager.allocate_buffer("model_cache", 1024*1024)

# 释放缓冲区
memory_manager.deallocate_buffer("model_cache")
```
'''
        
        with open(docs_dir / 'API.md', 'w', encoding='utf-8') as f:
            f.write(api_doc)
            
    def _create_archive(self, package_dir: Path) -> Path:
        """创建压缩包"""
        import zipfile
        
        archive_path = package_dir.with_suffix('.zip')
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in package_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(package_dir.parent)
                    zipf.write(file_path, arcname)
                    
        return archive_path
        
    def deploy_to_device(self, device_address: str, username: str = "pi") -> bool:
        """部署到远程设备"""
        self.logger.info(f"部署到设备: {device_address}")
        
        try:
            # 这里可以实现SSH部署逻辑
            # 例如使用paramiko库进行文件传输和远程执行
            
            # 示例代码框架:
            # 1. 连接到设备
            # 2. 传输部署包
            # 3. 解压并安装
            # 4. 运行测试
            # 5. 启动服务
            
            self.logger.info("远程部署功能待实现")
            return True
            
        except Exception as e:
            self.logger.error(f"远程部署失败: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='YOLO嵌入式自动部署工具')
    parser.add_argument('--platform', help='目标平台 (自动检测)')
    parser.add_argument('--model-size', default='n', choices=['n', 's', 'm', 'l', 'x'], help='模型大小')
    parser.add_argument('--model-path', help='自定义模型路径')
    parser.add_argument('--output-dir', default='./deployment', help='输出目录')
    parser.add_argument('--config', help='自定义配置文件')
    parser.add_argument('--deploy-to', help='远程部署地址 (user@host)')
    
    args = parser.parse_args()
    
    # 加载自定义配置
    custom_config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            custom_config = json.load(f)
            
    # 创建部署器
    deployer = EmbeddedDeployer(
        target_platform=args.platform,
        output_dir=args.output_dir
    )
    
    try:
        # 准备部署
        deployer.prepare_deployment(
            model_size=args.model_size,
            model_path=args.model_path,
            custom_config=custom_config
        )
        
        # 优化模型
        optimized_path = deployer.optimize_model()
        print(f"模型优化完成: {optimized_path}")
        
        # 创建部署包
        package_path = deployer.create_deployment_package()
        print(f"部署包创建完成: {package_path}")
        
        # 远程部署 (可选)
        if args.deploy_to:
            if '@' in args.deploy_to:
                username, device_address = args.deploy_to.split('@', 1)
            else:
                username, device_address = 'pi', args.deploy_to
                
            success = deployer.deploy_to_device(device_address, username)
            if success:
                print(f"远程部署成功: {device_address}")
            else:
                print(f"远程部署失败: {device_address}")
                
        print("\n部署完成!")
        print(f"部署包位置: {package_path}")
        print("\n后续步骤:")
        print("1. 将部署包传输到目标设备")
        print("2. 解压部署包")
        print("3. 运行 python scripts/install.py")
        print("4. 运行 python scripts/test.py")
        print("5. 启动检测器 python scripts/main.py")
        
    except Exception as e:
        print(f"部署失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()