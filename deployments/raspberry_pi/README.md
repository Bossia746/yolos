# YOLOS 树莓派版本

树莓派单板计算机版本

## 系统要求

### 硬件要求

- 树莓派4B或更新版本
- 树莓派摄像头模块或USB摄像头
- MicroSD卡(32GB以上)
- 电源适配器

### 软件要求

依赖包:
- opencv-python
- numpy
- tflite-runtime

## 安装说明

### 1. 下载部署包
解压下载的部署包到目标目录

### 2. 安装依赖

```bash
chmod +x install_rpi.sh
./install_rpi.sh
```

### 3. 运行程序
```bash
python main_rpi.py
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
YOLOS 树莓派版本 v1.0.0
