#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIoT开发板适配器
支持主流AIoT开发板环境的YOLOS识别系统
"""

import os
import sys
import json
import logging
import time
import subprocess
import platform
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import cv2

logger = logging.getLogger(__name__)

class AIoTBoardsAdapter:
    """AIoT开发板适配器"""
    
    def __init__(self):
        self.supported_boards = self._init_supported_boards()
        self.current_board = self._detect_current_board()
        self.board_config = self._get_board_config()
        
        logger.info(f"AIoT开发板适配器初始化完成 - 当前板子: {self.current_board['name']}")
    
    def _init_supported_boards(self) -> Dict[str, Dict[str, Any]]:
        """初始化支持的AIoT开发板"""
        boards = {
            # ===== NVIDIA 系列 =====
            'jetson_nano': {
                'name': 'NVIDIA Jetson Nano',
                'manufacturer': 'NVIDIA',
                'cpu': 'ARM Cortex-A57 Quad-core',
                'gpu': 'NVIDIA Maxwell 128 CUDA cores',
                'memory': '4GB LPDDR4',
                'ai_accelerator': 'GPU',
                'os_support': ['Ubuntu 18.04', 'JetPack'],
                'detection_methods': ['jetson_stats', '/proc/device-tree/model'],
                'keywords': ['jetson', 'nano', 'tegra'],
                'capabilities': {
                    'deep_learning': True,
                    'opencv_gpu': True,
                    'tensorrt': True,
                    'cuda': True,
                    'camera_csi': True,
                    'gpio': True
                }
            },
            
            'jetson_xavier_nx': {
                'name': 'NVIDIA Jetson Xavier NX',
                'manufacturer': 'NVIDIA',
                'cpu': 'ARM Cortex-A78AE 6-core',
                'gpu': 'NVIDIA Volta 384 CUDA cores',
                'memory': '8GB LPDDR4x',
                'ai_accelerator': 'GPU + DLA',
                'os_support': ['Ubuntu 18.04', 'JetPack'],
                'detection_methods': ['jetson_stats', '/proc/device-tree/model'],
                'keywords': ['jetson', 'xavier', 'nx'],
                'capabilities': {
                    'deep_learning': True,
                    'opencv_gpu': True,
                    'tensorrt': True,
                    'cuda': True,
                    'camera_csi': True,
                    'gpio': True,
                    'dla_acceleration': True
                }
            },
            
            'jetson_agx_xavier': {
                'name': 'NVIDIA Jetson AGX Xavier',
                'manufacturer': 'NVIDIA',
                'cpu': 'ARM Cortex-A78AE 8-core',
                'gpu': 'NVIDIA Volta 512 CUDA cores',
                'memory': '32GB LPDDR4x',
                'ai_accelerator': 'GPU + DLA',
                'os_support': ['Ubuntu 18.04', 'JetPack'],
                'detection_methods': ['jetson_stats', '/proc/device-tree/model'],
                'keywords': ['jetson', 'agx', 'xavier'],
                'capabilities': {
                    'deep_learning': True,
                    'opencv_gpu': True,
                    'tensorrt': True,
                    'cuda': True,
                    'camera_csi': True,
                    'gpio': True,
                    'dla_acceleration': True,
                    'high_performance': True
                }
            },
            
            'jetson_orin_nano': {
                'name': 'NVIDIA Jetson Orin Nano',
                'manufacturer': 'NVIDIA',
                'cpu': 'ARM Cortex-A78AE 6-core',
                'gpu': 'NVIDIA Ampere 1024 CUDA cores',
                'memory': '8GB LPDDR5',
                'ai_accelerator': 'GPU + DLA + PVA',
                'os_support': ['Ubuntu 20.04', 'JetPack 5.x'],
                'detection_methods': ['jetson_stats', '/proc/device-tree/model'],
                'keywords': ['jetson', 'orin', 'nano'],
                'capabilities': {
                    'deep_learning': True,
                    'opencv_gpu': True,
                    'tensorrt': True,
                    'cuda': True,
                    'camera_csi': True,
                    'gpio': True,
                    'dla_acceleration': True,
                    'pva_acceleration': True,
                    'latest_generation': True
                }
            },
            
            # ===== Google 系列 =====
            'coral_dev_board': {
                'name': 'Google Coral Dev Board',
                'manufacturer': 'Google',
                'cpu': 'NXP i.MX 8M SOC (Quad Cortex-A53)',
                'gpu': 'Vivante GC7000Lite',
                'memory': '1GB LPDDR4',
                'ai_accelerator': 'Edge TPU',
                'os_support': ['Mendel Linux'],
                'detection_methods': ['/sys/firmware/devicetree/base/model'],
                'keywords': ['coral', 'mendel', 'edge-tpu'],
                'capabilities': {
                    'deep_learning': True,
                    'edge_tpu': True,
                    'tensorflow_lite': True,
                    'camera_mipi': True,
                    'gpio': True,
                    'low_power': True
                }
            },
            
            'coral_dev_board_micro': {
                'name': 'Google Coral Dev Board Micro',
                'manufacturer': 'Google',
                'cpu': 'NXP i.MX RT1176 (Cortex-M7)',
                'gpu': None,
                'memory': '64MB SDRAM',
                'ai_accelerator': 'Edge TPU',
                'os_support': ['FreeRTOS'],
                'detection_methods': ['usb_device_info'],
                'keywords': ['coral', 'micro', 'freertos'],
                'capabilities': {
                    'deep_learning': True,
                    'edge_tpu': True,
                    'tensorflow_lite_micro': True,
                    'ultra_low_power': True,
                    'real_time': True
                }
            },
            
            # ===== Intel 系列 =====
            'intel_nuc': {
                'name': 'Intel NUC',
                'manufacturer': 'Intel',
                'cpu': 'Intel Core i3/i5/i7',
                'gpu': 'Intel Iris Xe / UHD Graphics',
                'memory': '8GB-64GB DDR4',
                'ai_accelerator': 'Intel Graphics / Movidius',
                'os_support': ['Ubuntu', 'Windows', 'OpenVINO'],
                'detection_methods': ['dmidecode', 'lscpu'],
                'keywords': ['nuc', 'intel'],
                'capabilities': {
                    'deep_learning': True,
                    'openvino': True,
                    'intel_gpu': True,
                    'high_performance': True,
                    'x86_compatibility': True
                }
            },
            
            'intel_neural_compute_stick': {
                'name': 'Intel Neural Compute Stick 2',
                'manufacturer': 'Intel',
                'cpu': 'Host dependent',
                'gpu': None,
                'memory': 'Host dependent',
                'ai_accelerator': 'Intel Movidius Myriad X VPU',
                'os_support': ['Ubuntu', 'Windows', 'Raspberry Pi OS'],
                'detection_methods': ['lsusb', 'openvino_detection'],
                'keywords': ['movidius', 'ncs2', 'myriad'],
                'capabilities': {
                    'deep_learning': True,
                    'openvino': True,
                    'usb_acceleration': True,
                    'portable': True,
                    'low_power': True
                }
            },
            
            # ===== Rockchip 系列 =====
            'rk3588': {
                'name': 'Rockchip RK3588 Boards',
                'manufacturer': 'Rockchip',
                'cpu': 'ARM Cortex-A76 + A55 8-core',
                'gpu': 'Mali-G610 MP4',
                'memory': '4GB-32GB LPDDR4/5',
                'ai_accelerator': 'NPU 6 TOPS',
                'os_support': ['Ubuntu', 'Debian', 'Android'],
                'detection_methods': ['/proc/cpuinfo', '/sys/firmware/devicetree/base/model'],
                'keywords': ['rk3588', 'rockchip'],
                'capabilities': {
                    'deep_learning': True,
                    'npu_acceleration': True,
                    'rknn_toolkit': True,
                    'mali_gpu': True,
                    'camera_mipi': True,
                    'gpio': True,
                    'high_performance': True
                }
            },
            
            'rk3566': {
                'name': 'Rockchip RK3566 Boards',
                'manufacturer': 'Rockchip',
                'cpu': 'ARM Cortex-A55 Quad-core',
                'gpu': 'Mali-G52 2EE',
                'memory': '2GB-8GB LPDDR4',
                'ai_accelerator': 'NPU 0.8 TOPS',
                'os_support': ['Ubuntu', 'Debian', 'Android'],
                'detection_methods': ['/proc/cpuinfo', '/sys/firmware/devicetree/base/model'],
                'keywords': ['rk3566', 'rockchip'],
                'capabilities': {
                    'deep_learning': True,
                    'npu_acceleration': True,
                    'rknn_toolkit': True,
                    'mali_gpu': True,
                    'camera_mipi': True,
                    'gpio': True,
                    'mid_range': True
                }
            },
            
            # ===== Qualcomm 系列 =====
            'qualcomm_rb5': {
                'name': 'Qualcomm RB5 Platform',
                'manufacturer': 'Qualcomm',
                'cpu': 'Snapdragon 865 (Kryo 585)',
                'gpu': 'Adreno 650',
                'memory': '8GB LPDDR5',
                'ai_accelerator': 'Hexagon 698 DSP (15 TOPS)',
                'os_support': ['Ubuntu', 'Android', 'QNX'],
                'detection_methods': ['/proc/cpuinfo', 'getprop'],
                'keywords': ['snapdragon', 'rb5', 'qualcomm'],
                'capabilities': {
                    'deep_learning': True,
                    'hexagon_dsp': True,
                    'adreno_gpu': True,
                    'snpe_runtime': True,
                    'camera_mipi': True,
                    'gpio': True,
                    'high_performance': True,
                    '5g_support': True
                }
            },
            
            # ===== Amlogic 系列 =====
            'amlogic_a311d': {
                'name': 'Amlogic A311D Boards',
                'manufacturer': 'Amlogic',
                'cpu': 'ARM Cortex-A73 + A53 6-core',
                'gpu': 'Mali-G52 MP4',
                'memory': '4GB LPDDR4',
                'ai_accelerator': 'NPU 5.0 TOPS',
                'os_support': ['Ubuntu', 'Android'],
                'detection_methods': ['/proc/cpuinfo', '/sys/firmware/devicetree/base/model'],
                'keywords': ['a311d', 'amlogic', 'vim3'],
                'capabilities': {
                    'deep_learning': True,
                    'npu_acceleration': True,
                    'mali_gpu': True,
                    'camera_mipi': True,
                    'gpio': True,
                    'cost_effective': True
                }
            },
            
            # ===== Kendryte 系列 =====
            'k210': {
                'name': 'Kendryte K210',
                'manufacturer': 'Kendryte',
                'cpu': 'RISC-V Dual-core 64-bit @ 400MHz',
                'gpu': 'None',
                'memory': '8MB SRAM',
                'ai_accelerator': 'KPU (0.25 TOPS)',
                'os_support': ['FreeRTOS', 'RT-Thread', 'MicroPython'],
                'detection_methods': ['serial_detection', 'usb_vid_pid'],
                'keywords': ['k210', 'kendryte', 'sipeed', 'maixpy'],
                'usb_identifiers': [
                    {'vid': '0403', 'pid': '6001'},  # FTDI
                    {'vid': '1A86', 'pid': '7523'},  # CH340
                    {'vid': '10C4', 'pid': 'EA60'}   # CP2102
                ],
                'capabilities': {
                    'deep_learning': True,
                    'kpu_acceleration': True,
                    'micropython': True,
                    'camera_dvp': True,
                    'gpio': True,
                    'spi': True,
                    'i2c': True,
                    'uart': True,
                    'low_power': True,
                    'edge_ai': True,
                    'real_time': True
                },
                'optimization': {
                    'max_workers': 1,
                    'memory_limit_mb': 6,
                    'batch_size': 1,
                    'image_max_size': (224, 224),
                    'enable_multiprocessing': False,
                    'use_kpu': True,
                    'quantization': 'int8',
                    'model_format': 'kmodel'
                }
            },
            
            # ===== MediaTek 系列 =====
            'mediatek_genio': {
                'name': 'MediaTek Genio Platform',
                'manufacturer': 'MediaTek',
                'cpu': 'ARM Cortex-A78 + A55',
                'gpu': 'Mali-G57 MC3',
                'memory': '4GB-8GB LPDDR4x',
                'ai_accelerator': 'APU 3.0 (4 TOPS)',
                'os_support': ['Ubuntu', 'Android', 'Yocto'],
                'detection_methods': ['/proc/cpuinfo'],
                'keywords': ['genio', 'mediatek', 'mt8395'],
                'capabilities': {
                    'deep_learning': True,
                    'apu_acceleration': True,
                    'mali_gpu': True,
                    'camera_mipi': True,
                    'gpio': True,
                    'wifi6_support': True
                }
            },
            
            # ===== Allwinner 系列 =====
            'allwinner_h6': {
                'name': 'Allwinner H6 Boards',
                'manufacturer': 'Allwinner',
                'cpu': 'ARM Cortex-A53 Quad-core',
                'gpu': 'Mali-T720 MP2',
                'memory': '1GB-4GB DDR3',
                'ai_accelerator': None,
                'os_support': ['Ubuntu', 'Debian', 'Android'],
                'detection_methods': ['/proc/cpuinfo', '/sys/firmware/devicetree/base/model'],
                'keywords': ['h6', 'allwinner', 'orange'],
                'capabilities': {
                    'deep_learning': False,
                    'mali_gpu': True,
                    'camera_csi': True,
                    'gpio': True,
                    'low_cost': True,
                    'basic_cv': True
                }
            },
            
            # ===== 树莓派系列 (扩展) =====
            'raspberry_pi_4': {
                'name': 'Raspberry Pi 4',
                'manufacturer': 'Raspberry Pi Foundation',
                'cpu': 'ARM Cortex-A72 Quad-core',
                'gpu': 'VideoCore VI',
                'memory': '2GB-8GB LPDDR4',
                'ai_accelerator': None,
                'os_support': ['Raspberry Pi OS', 'Ubuntu'],
                'detection_methods': ['/proc/cpuinfo', '/proc/device-tree/model'],
                'keywords': ['raspberry', 'pi', 'bcm2711'],
                'capabilities': {
                    'deep_learning': True,
                    'opencv': True,
                    'camera_csi': True,
                    'gpio': True,
                    'popular': True,
                    'community_support': True
                }
            },
            
            'raspberry_pi_5': {
                'name': 'Raspberry Pi 5',
                'manufacturer': 'Raspberry Pi Foundation',
                'cpu': 'ARM Cortex-A76 Quad-core',
                'gpu': 'VideoCore VII',
                'memory': '4GB-8GB LPDDR4x',
                'ai_accelerator': None,
                'os_support': ['Raspberry Pi OS', 'Ubuntu'],
                'detection_methods': ['/proc/cpuinfo', '/proc/device-tree/model'],
                'keywords': ['raspberry', 'pi', 'bcm2712'],
                'capabilities': {
                    'deep_learning': True,
                    'opencv': True,
                    'camera_csi': True,
                    'gpio': True,
                    'latest_generation': True,
                    'improved_performance': True
                }
            },
            
            # ===== ESP32 系列 (扩展) =====
            'esp32_s3': {
                'name': 'ESP32-S3',
                'manufacturer': 'Espressif',
                'cpu': 'Xtensa LX7 Dual-core',
                'gpu': None,
                'memory': '512KB SRAM + 8MB PSRAM',
                'ai_accelerator': 'Vector instructions',
                'os_support': ['ESP-IDF', 'Arduino', 'MicroPython'],
                'detection_methods': ['esp_chip_info'],
                'keywords': ['esp32', 's3', 'espressif'],
                'capabilities': {
                    'deep_learning': True,
                    'tflite_micro': True,
                    'camera_support': True,
                    'wifi': True,
                    'bluetooth': True,
                    'ultra_low_power': True,
                    'edge_ai': True
                }
            },
            
            'esp32_cam': {
                'name': 'ESP32-CAM',
                'manufacturer': 'Espressif',
                'cpu': 'Xtensa LX6 Dual-core',
                'gpu': None,
                'memory': '520KB SRAM + 4MB PSRAM',
                'ai_accelerator': None,
                'os_support': ['ESP-IDF', 'Arduino'],
                'detection_methods': ['esp_chip_info'],
                'keywords': ['esp32', 'cam', 'ov2640'],
                'capabilities': {
                    'deep_learning': False,
                    'basic_cv': True,
                    'camera_builtin': True,
                    'wifi': True,
                    'ultra_low_cost': True,
                    'iot_focused': True
                }
            },
            
            # ===== STM32 系列 =====
            'stm32h7': {
                'name': 'STM32H7 Series',
                'manufacturer': 'STMicroelectronics',
                'cpu': 'ARM Cortex-M7 (up to 550MHz)',
                'gpu': 'Chrom-ART Accelerator',
                'memory': '1MB SRAM + 2MB Flash',
                'ai_accelerator': 'ARM CMSIS-NN',
                'os_support': ['STM32CubeIDE', 'Arduino', 'Mbed', 'FreeRTOS'],
                'detection_methods': ['stm32_device_id', 'usb_device_info'],
                'keywords': ['stm32', 'h7', 'cortex-m7', 'stmicroelectronics'],
                'capabilities': {
                    'deep_learning': True,
                    'tflite_micro': True,
                    'cmsis_nn': True,
                    'dsp_acceleration': True,
                    'camera_dcmi': True,
                    'gpio': True,
                    'real_time': True,
                    'low_power': True,
                    'industrial_grade': True
                }
            },
            
            'stm32f7': {
                'name': 'STM32F7 Series',
                'manufacturer': 'STMicroelectronics',
                'cpu': 'ARM Cortex-M7 (up to 216MHz)',
                'gpu': 'Chrom-ART Accelerator',
                'memory': '512KB SRAM + 2MB Flash',
                'ai_accelerator': 'ARM CMSIS-NN',
                'os_support': ['STM32CubeIDE', 'Arduino', 'Mbed'],
                'detection_methods': ['stm32_device_id', 'usb_device_info'],
                'keywords': ['stm32', 'f7', 'cortex-m7', 'stmicroelectronics'],
                'capabilities': {
                    'deep_learning': True,
                    'tflite_micro': True,
                    'cmsis_nn': True,
                    'dsp_acceleration': True,
                    'camera_dcmi': True,
                    'gpio': True,
                    'real_time': True,
                    'cost_effective': True
                }
            },
            
            'stm32f4': {
                'name': 'STM32F4 Series',
                'manufacturer': 'STMicroelectronics',
                'cpu': 'ARM Cortex-M4 (up to 180MHz)',
                'gpu': 'Chrom-ART Accelerator (F429/439)',
                'memory': '256KB SRAM + 1MB Flash',
                'ai_accelerator': 'ARM CMSIS-NN',
                'os_support': ['STM32CubeIDE', 'Arduino', 'Mbed'],
                'detection_methods': ['stm32_device_id', 'usb_device_info'],
                'keywords': ['stm32', 'f4', 'cortex-m4', 'stmicroelectronics'],
                'capabilities': {
                    'deep_learning': False,  # 有限的AI能力
                    'basic_cv': True,
                    'cmsis_nn': True,
                    'dsp_acceleration': True,
                    'camera_dcmi': True,
                    'gpio': True,
                    'real_time': True,
                    'widely_used': True
                }
            },
            
            'stm32l4': {
                'name': 'STM32L4 Series',
                'manufacturer': 'STMicroelectronics',
                'cpu': 'ARM Cortex-M4 (up to 120MHz)',
                'gpu': None,
                'memory': '320KB SRAM + 1MB Flash',
                'ai_accelerator': 'ARM CMSIS-NN',
                'os_support': ['STM32CubeIDE', 'Arduino', 'Mbed'],
                'detection_methods': ['stm32_device_id', 'usb_device_info'],
                'keywords': ['stm32', 'l4', 'cortex-m4', 'ultra-low-power'],
                'capabilities': {
                    'deep_learning': False,
                    'basic_cv': True,
                    'cmsis_nn': True,
                    'ultra_low_power': True,
                    'battery_powered': True,
                    'gpio': True,
                    'real_time': True,
                    'sensor_fusion': True
                }
            },
            
            'stm32mp1': {
                'name': 'STM32MP1 Series',
                'manufacturer': 'STMicroelectronics',
                'cpu': 'ARM Cortex-A7 + Cortex-M4',
                'gpu': 'Vivante GC400T',
                'memory': '512MB DDR3 + 708KB SRAM',
                'ai_accelerator': 'GPU + CMSIS-NN',
                'os_support': ['Linux', 'STM32CubeIDE', 'OpenSTLinux'],
                'detection_methods': ['stm32_device_id', '/proc/cpuinfo'],
                'keywords': ['stm32', 'mp1', 'cortex-a7', 'linux'],
                'capabilities': {
                    'deep_learning': True,
                    'linux_support': True,
                    'dual_core': True,
                    'gpu_acceleration': True,
                    'camera_csi': True,
                    'ethernet': True,
                    'gpio': True,
                    'industrial_iot': True,
                    'heterogeneous_computing': True
                }
            },
            
            'stm32ai_series': {
                'name': 'STM32 AI Ecosystem',
                'manufacturer': 'STMicroelectronics',
                'cpu': 'Various ARM Cortex-M/A',
                'gpu': 'Various',
                'memory': 'Various',
                'ai_accelerator': 'STM32Cube.AI + X-CUBE-AI',
                'os_support': ['STM32CubeIDE', 'STM32CubeMX'],
                'detection_methods': ['stm32_ai_detection'],
                'keywords': ['stm32', 'cube.ai', 'x-cube-ai', 'neural'],
                'capabilities': {
                    'deep_learning': True,
                    'neural_network_optimization': True,
                    'model_compression': True,
                    'quantization': True,
                    'stm32cube_ai': True,
                    'cross_platform': True,
                    'ecosystem_support': True,
                    'camera_dcmi': True
                }
            },
            
            # ===== 新增支持摄像头的开发板 =====
            'hikey_970': {
                'name': 'HiKey 970',
                'manufacturer': 'HiSilicon',
                'cpu': 'Kirin 970 (4x Cortex-A73 + 4x Cortex-A53)',
                'gpu': 'Mali-G72 MP12',
                'memory': '6GB LPDDR4X',
                'ai_accelerator': 'NPU (Neural Processing Unit)',
                'os_support': ['Android', 'Ubuntu'],
                'detection_methods': ['/proc/cpuinfo'],
                'keywords': ['hikey', '970', 'kirin'],
                'capabilities': {
                    'deep_learning': True,
                    'npu_acceleration': True,
                    'mali_gpu': True,
                    'camera_mipi': True,
                    'gpio': True,
                    'high_performance': True
                }
            },
            
            'odroid_n2': {
                'name': 'ODROID-N2',
                'manufacturer': 'Hardkernel',
                'cpu': 'Amlogic S922X (4x Cortex-A73 + 2x Cortex-A53)',
                'gpu': 'Mali-G52 MP6',
                'memory': '2GB/4GB DDR4',
                'ai_accelerator': None,
                'os_support': ['Ubuntu', 'Android'],
                'detection_methods': ['/proc/cpuinfo'],
                'keywords': ['odroid', 'n2', 's922x'],
                'capabilities': {
                    'deep_learning': True,
                    'mali_gpu': True,
                    'camera_mipi': True,
                    'gpio': True,
                    'cost_effective': True
                }
            }
        }
        
        return boards
    
    def _detect_current_board(self) -> Dict[str, Any]:
        """检测当前AIoT开发板"""
        detected_board = {
            'name': 'Unknown',
            'confidence': 0.0,
            'detection_method': 'none',
            'capabilities': {}
        }
        
        for board_id, board_info in self.supported_boards.items():
            confidence = self._calculate_board_confidence(board_info)
            
            if confidence > detected_board['confidence']:
                detected_board = {
                    'id': board_id,
                    'name': board_info['name'],
                    'confidence': confidence,
                    'detection_method': self._get_detection_method(board_info),
                    'capabilities': board_info['capabilities'],
                    'info': board_info
                }
        
        return detected_board
    
    def _calculate_board_confidence(self, board_info: Dict[str, Any]) -> float:
        """计算开发板检测置信度"""
        confidence = 0.0
        
        try:
            # 检查关键词匹配
            keywords = board_info.get('keywords', [])
            
            # 检查CPU信息
            if os.path.exists('/proc/cpuinfo'):
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read().lower()
                    
                for keyword in keywords:
                    if keyword.lower() in cpuinfo:
                        confidence += 0.3
            
            # 检查设备树模型
            if os.path.exists('/proc/device-tree/model'):
                with open('/proc/device-tree/model', 'r') as f:
                    model = f.read().lower()
                    
                for keyword in keywords:
                    if keyword.lower() in model:
                        confidence += 0.4
            
            # 检查系统信息
            if os.path.exists('/sys/firmware/devicetree/base/model'):
                with open('/sys/firmware/devicetree/base/model', 'r') as f:
                    model = f.read().lower()
                    
                for keyword in keywords:
                    if keyword.lower() in model:
                        confidence += 0.4
            
            # 特殊检测方法
            if 'jetson_stats' in board_info.get('detection_methods', []):
                confidence += self._check_jetson_stats()
            
            if 'openvino_detection' in board_info.get('detection_methods', []):
                confidence += self._check_openvino_support()
            
            if 'esp_chip_info' in board_info.get('detection_methods', []):
                confidence += self._check_esp32_info()
            
            if 'stm32_device_id' in board_info.get('detection_methods', []):
                confidence += self._check_stm32_info()
            
            if 'stm32_ai_detection' in board_info.get('detection_methods', []):
                confidence += self._check_stm32_info() * 0.8  # STM32 AI生态系统检测
            
            # K210检测方法
            if 'serial_detection' in board_info.get('detection_methods', []):
                confidence += self._check_k210_serial()
            
            if 'usb_vid_pid' in board_info.get('detection_methods', []):
                confidence += self._check_k210_usb(board_info)
            
        except Exception as e:
            logger.warning(f"开发板检测异常: {e}")
        
        return min(confidence, 1.0)
    
    def _check_jetson_stats(self) -> float:
        """检查Jetson统计信息"""
        try:
            import jetson.stats
            return 0.8
        except ImportError:
            try:
                result = subprocess.run(['jetson_stats'], capture_output=True, text=True, timeout=5)
                return 0.6 if result.returncode == 0 else 0.0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return 0.0
    
    def _check_openvino_support(self) -> float:
        """检查OpenVINO支持"""
        try:
            import openvino
            return 0.7
        except ImportError:
            try:
                result = subprocess.run(['lsusb'], capture_output=True, text=True, timeout=5)
                if 'movidius' in result.stdout.lower() or '03e7:' in result.stdout:
                    return 0.8
                return 0.0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return 0.0
    
    def _check_esp32_info(self) -> float:
        """检查ESP32信息"""
        # 这个方法主要用于ESP32环境下的MicroPython或Arduino IDE
        try:
            # 检查是否在ESP32环境中运行
            if hasattr(sys, 'platform') and 'esp32' in sys.platform:
                return 0.9
            
            # 检查串口设备
            if os.path.exists('/dev/ttyUSB0') or os.path.exists('/dev/ttyACM0'):
                return 0.3
            
            return 0.0
        except Exception:
            return 0.0
    
    def _check_stm32_info(self) -> float:
        """检查STM32信息"""
        try:
            # 检查STM32 USB设备
            try:
                import serial.tools.list_ports
                ports = serial.tools.list_ports.comports()
                
                stm32_identifiers = [
                    ('0483', '374B'),  # STM32 STLink
                    ('0483', '3748'),  # STM32 Virtual COM Port
                    ('0483', '5740'),  # STM32 DFU
                    ('0483', 'DF11'),  # STM32 DFU Bootloader
                ]
                
                for port in ports:
                    hwid = port.hwid.upper()
                    for vid, pid in stm32_identifiers:
                        if f'VID_{vid}' in hwid and f'PID_{pid}' in hwid:
                            return 0.8
                
                # 检查STM32相关描述
                for port in ports:
                    description = port.description.lower()
                    if any(keyword in description for keyword in ['stm32', 'stlink', 'nucleo', 'discovery']):
                        return 0.6
                
            except ImportError:
                pass
            
            # 检查STM32CubeIDE或相关工具
            stm32_tools = [
                'STM32CubeIDE',
                'STM32CubeMX', 
                'STM32CubeProgrammer'
            ]
            
            for tool in stm32_tools:
                try:
                    result = subprocess.run([tool, '--version'], 
                                         capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        return 0.7
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _check_k210_serial(self) -> float:
        """检查K210串口连接"""
        try:
            import serial.tools.list_ports
            
            ports = serial.tools.list_ports.comports()
            k210_indicators = [
                'k210', 'kendryte', 'sipeed', 'maixpy',
                'ch340', 'ch341', 'ftdi', 'cp210'
            ]
            
            for port in ports:
                port_desc = port.description.lower()
                port_hwid = port.hwid.lower()
                
                # 检查描述中的K210相关关键词
                for indicator in k210_indicators:
                    if indicator in port_desc or indicator in port_hwid:
                        return 0.7
                
                # 检查常见的K210开发板串口芯片
                if any(chip in port_desc for chip in ['ch340', 'ch341', 'ftdi', 'cp210']):
                    # 尝试串口通信验证
                    if self._verify_k210_serial(port.device):
                        return 0.8
            
            return 0.0
            
        except ImportError:
            logger.warning("pyserial未安装，无法检测K210串口")
            return 0.0
        except Exception as e:
            logger.debug(f"K210串口检测失败: {e}")
            return 0.0
    
    def _check_k210_usb(self, board_info: Dict[str, Any]) -> float:
        """检查K210 USB设备"""
        try:
            usb_identifiers = board_info.get('usb_identifiers', [])
            
            # Windows系统使用wmic命令
            if platform.system() == 'Windows':
                try:
                    result = subprocess.run(
                        ['wmic', 'path', 'win32_pnpentity', 'get', 'deviceid'],
                        capture_output=True, text=True, timeout=10
                    )
                    
                    for identifier in usb_identifiers:
                        vid = identifier['vid'].upper()
                        pid = identifier['pid'].upper()
                        usb_id = f"VID_{vid}&PID_{pid}"
                        
                        if usb_id in result.stdout.upper():
                            return 0.6
                    
                except Exception:
                    pass
            
            # Linux/macOS系统使用lsusb
            else:
                try:
                    result = subprocess.run(['lsusb'], capture_output=True, text=True, timeout=5)
                    
                    for identifier in usb_identifiers:
                        vid = identifier['vid']
                        pid = identifier['pid']
                        usb_pattern = f"{vid}:{pid}"
                        
                        if usb_pattern in result.stdout:
                            return 0.6
                    
                except Exception:
                    pass
            
            return 0.0
            
        except Exception as e:
            logger.debug(f"K210 USB检测失败: {e}")
            return 0.0
    
    def _verify_k210_serial(self, port: str) -> bool:
        """验证K210串口通信"""
        try:
            import serial
            
            # 尝试打开串口
            with serial.Serial(port, 115200, timeout=1) as ser:
                # 发送简单的测试命令
                test_commands = [
                    b'\r\n',  # 回车换行
                    b'help\r\n',  # help命令
                    b'print("test")\r\n'  # MicroPython测试
                ]
                
                for cmd in test_commands:
                    ser.write(cmd)
                    time.sleep(0.1)
                    
                    # 读取响应
                    response = ser.read(ser.in_waiting or 64)
                    if response:
                        response_str = response.decode('utf-8', errors='ignore').lower()
                        
                        # 检查K210相关的响应
                        k210_responses = [
                            'micropython', 'maixpy', 'k210', 'kendryte',
                            'repl', '>>>', 'help', 'test'
                        ]
                        
                        for resp in k210_responses:
                            if resp in response_str:
                                return True
                
                return False
                
        except Exception as e:
            logger.debug(f"K210串口验证失败 {port}: {e}")
            return False
    
    def _get_detection_method(self, board_info: Dict[str, Any]) -> str:
        """获取检测方法"""
        methods = board_info.get('detection_methods', [])
        
        for method in methods:
            if method == '/proc/cpuinfo' and os.path.exists('/proc/cpuinfo'):
                return 'cpuinfo'
            elif method == '/proc/device-tree/model' and os.path.exists('/proc/device-tree/model'):
                return 'device_tree'
            elif method == 'jetson_stats':
                return 'jetson_stats'
            elif method == 'openvino_detection':
                return 'openvino'
        
        return 'heuristic'
    
    def _get_board_config(self) -> Dict[str, Any]:
        """获取开发板配置"""
        if self.current_board['name'] == 'Unknown':
            return self._get_default_config()
        
        board_info = self.current_board.get('info', {})
        capabilities = self.current_board.get('capabilities', {})
        
        config = {
            'optimization': self._get_optimization_config(board_info, capabilities),
            'ai_acceleration': self._get_ai_acceleration_config(capabilities),
            'camera_config': self._get_camera_config(capabilities),
            'gpio_config': self._get_gpio_config(capabilities),
            'performance_limits': self._get_performance_limits(board_info)
        }
        
        return config
    
    def _get_optimization_config(self, board_info: Dict, capabilities: Dict) -> Dict[str, Any]:
        """获取优化配置"""
        config = {
            'max_workers': 2,
            'memory_limit_gb': 1.0,
            'batch_size': 1,
            'image_max_size': (640, 480),
            'use_gpu': False,
            'use_ai_accelerator': False
        }
        
        # 根据内存调整
        memory_str = board_info.get('memory', '1GB')
        if 'GB' in memory_str:
            memory_gb = int(memory_str.split('GB')[0].split()[-1])
            config['memory_limit_gb'] = max(0.5, memory_gb * 0.6)
            
            if memory_gb >= 8:
                config['max_workers'] = 4
                config['batch_size'] = 2
            elif memory_gb >= 4:
                config['max_workers'] = 3
                config['batch_size'] = 1
        
        # GPU加速
        if capabilities.get('cuda', False) or capabilities.get('mali_gpu', False):
            config['use_gpu'] = True
        
        # AI加速器
        if any(capabilities.get(key, False) for key in ['edge_tpu', 'npu_acceleration', 'hexagon_dsp', 'apu_acceleration', 'kpu']):
            config['use_ai_accelerator'] = True
        
        # K210特定优化
        if capabilities.get('kpu', False):
            config['memory_limit_gb'] = 0.5  # K210内存限制
            config['max_workers'] = 1  # 单核处理
            config['image_max_size'] = (224, 224)  # KPU最佳输入尺寸
            config['batch_size'] = 1  # 不支持批处理
        
        return config
    
    def _get_ai_acceleration_config(self, capabilities: Dict) -> Dict[str, Any]:
        """获取AI加速配置"""
        config = {
            'available_accelerators': [],
            'preferred_accelerator': None,
            'frameworks': []
        }
        
        # Edge TPU (Google Coral)
        if capabilities.get('edge_tpu', False):
            config['available_accelerators'].append('edge_tpu')
            config['frameworks'].extend(['tensorflow_lite', 'pycoral'])
            config['preferred_accelerator'] = 'edge_tpu'
        
        # NVIDIA GPU/TensorRT
        if capabilities.get('cuda', False):
            config['available_accelerators'].append('cuda')
            config['frameworks'].extend(['pytorch', 'tensorflow', 'tensorrt'])
            if not config['preferred_accelerator']:
                config['preferred_accelerator'] = 'cuda'
        
        # NPU (Rockchip/Amlogic)
        if capabilities.get('npu_acceleration', False):
            config['available_accelerators'].append('npu')
            config['frameworks'].extend(['rknn_toolkit', 'tengine'])
            if not config['preferred_accelerator']:
                config['preferred_accelerator'] = 'npu'
        
        # Hexagon DSP (Qualcomm)
        if capabilities.get('hexagon_dsp', False):
            config['available_accelerators'].append('hexagon_dsp')
            config['frameworks'].extend(['snpe', 'qnn'])
            if not config['preferred_accelerator']:
                config['preferred_accelerator'] = 'hexagon_dsp'
        
        # Intel Movidius
        if capabilities.get('openvino', False):
            config['available_accelerators'].append('movidius')
            config['frameworks'].extend(['openvino'])
            if not config['preferred_accelerator']:
                config['preferred_accelerator'] = 'movidius'
        
        # K210 KPU (Kendryte)
        if capabilities.get('kpu', False):
            config['available_accelerators'].append('kpu')
            config['frameworks'].extend(['nncase', 'kendryte_standalone_sdk'])
            if not config['preferred_accelerator']:
                config['preferred_accelerator'] = 'kpu'
        
        return config
    
    def _get_camera_config(self, capabilities: Dict) -> Dict[str, Any]:
        """获取摄像头配置"""
        config = {
            'supported_interfaces': [],
            'default_resolution': (640, 480),
            'max_fps': 30
        }
        
        if capabilities.get('camera_csi', False):
            config['supported_interfaces'].append('csi')
            config['default_resolution'] = (1920, 1080)
            config['max_fps'] = 60
        
        if capabilities.get('camera_mipi', False):
            config['supported_interfaces'].append('mipi')
            config['default_resolution'] = (1920, 1080)
            config['max_fps'] = 30
        
        if capabilities.get('camera_builtin', False):
            config['supported_interfaces'].append('builtin')
            config['default_resolution'] = (800, 600)
            config['max_fps'] = 25
        
        # USB摄像头支持（通用）
        config['supported_interfaces'].append('usb')
        
        return config
    
    def _get_gpio_config(self, capabilities: Dict) -> Dict[str, Any]:
        """获取GPIO配置"""
        config = {
            'gpio_available': capabilities.get('gpio', False),
            'gpio_library': None,
            'pin_count': 0
        }
        
        if capabilities.get('gpio', False):
            # 根据开发板类型设置GPIO库
            if 'raspberry' in self.current_board['name'].lower():
                config['gpio_library'] = 'RPi.GPIO'
                config['pin_count'] = 40
            elif 'jetson' in self.current_board['name'].lower():
                config['gpio_library'] = 'Jetson.GPIO'
                config['pin_count'] = 40
            else:
                config['gpio_library'] = 'generic'
                config['pin_count'] = 20
        
        return config
    
    def _get_performance_limits(self, board_info: Dict) -> Dict[str, Any]:
        """获取性能限制"""
        limits = {
            'max_concurrent_streams': 1,
            'max_model_size_mb': 100,
            'thermal_throttling': True,
            'power_management': True
        }
        
        # 根据AI加速器调整
        ai_accelerator = board_info.get('ai_accelerator')
        if ai_accelerator:
            if 'TOPS' in str(ai_accelerator):
                # 提取TOPS数值
                tops_str = str(ai_accelerator)
                import re
                tops_match = re.search(r'(\d+(?:\.\d+)?)\s*TOPS', tops_str)
                if tops_match:
                    tops = float(tops_match.group(1))
                    limits['max_concurrent_streams'] = min(4, int(tops / 2))
                    limits['max_model_size_mb'] = min(500, int(tops * 50))
        
        return limits
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'optimization': {
                'max_workers': 1,
                'memory_limit_gb': 0.5,
                'batch_size': 1,
                'image_max_size': (320, 240),
                'use_gpu': False,
                'use_ai_accelerator': False
            },
            'ai_acceleration': {
                'available_accelerators': [],
                'preferred_accelerator': None,
                'frameworks': ['opencv', 'numpy']
            },
            'camera_config': {
                'supported_interfaces': ['usb'],
                'default_resolution': (640, 480),
                'max_fps': 30
            },
            'gpio_config': {
                'gpio_available': False,
                'gpio_library': None,
                'pin_count': 0
            },
            'performance_limits': {
                'max_concurrent_streams': 1,
                'max_model_size_mb': 50,
                'thermal_throttling': True,
                'power_management': True
            }
        }
    
    def get_board_info(self) -> Dict[str, Any]:
        """获取开发板信息"""
        return {
            'detected_board': self.current_board,
            'supported_boards': list(self.supported_boards.keys()),
            'board_config': self.board_config,
            'system_info': {
                'platform': platform.platform(),
                'architecture': platform.machine(),
                'python_version': platform.python_version()
            }
        }
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """获取优化建议"""
        recommendations = {
            'model_optimization': [],
            'runtime_optimization': [],
            'hardware_optimization': [],
            'framework_recommendations': []
        }
        
        capabilities = self.current_board.get('capabilities', {})
        
        # 模型优化建议
        if capabilities.get('edge_tpu', False):
            recommendations['model_optimization'].extend([
                '使用TensorFlow Lite模型',
                '量化模型到INT8',
                '使用Edge TPU编译器优化模型'
            ])
            recommendations['framework_recommendations'].append('PyCoral + TensorFlow Lite')
        
        if capabilities.get('cuda', False):
            recommendations['model_optimization'].extend([
                '使用TensorRT优化模型',
                '启用FP16精度',
                '使用CUDA内存池'
            ])
            recommendations['framework_recommendations'].append('TensorRT + PyTorch/TensorFlow')
        
        if capabilities.get('npu_acceleration', False):
            recommendations['model_optimization'].extend([
                '使用RKNN工具链转换模型',
                '优化模型结构适配NPU',
                '使用NPU专用算子'
            ])
            recommendations['framework_recommendations'].append('RKNN-Toolkit')
        
        # 运行时优化建议
        if self.board_config['optimization']['memory_limit_gb'] < 2:
            recommendations['runtime_optimization'].extend([
                '减少批处理大小',
                '使用内存映射文件',
                '启用模型缓存'
            ])
        
        if capabilities.get('ultra_low_power', False):
            recommendations['runtime_optimization'].extend([
                '使用动态频率调节',
                '启用睡眠模式',
                '优化推理间隔'
            ])
        
        # 硬件优化建议
        if capabilities.get('thermal_throttling', False):
            recommendations['hardware_optimization'].extend([
                '添加散热片',
                '监控温度',
                '降低工作频率'
            ])
        
        return recommendations
    
    def install_board_dependencies(self) -> Dict[str, bool]:
        """安装开发板依赖"""
        results = {}
        capabilities = self.current_board.get('capabilities', {})
        
        try:
            # Edge TPU依赖
            if capabilities.get('edge_tpu', False):
                results['pycoral'] = self._install_pycoral()
                results['tflite_runtime'] = self._install_tflite_runtime()
            
            # CUDA依赖
            if capabilities.get('cuda', False):
                results['torch_cuda'] = self._check_torch_cuda()
                results['tensorrt'] = self._check_tensorrt()
            
            # OpenVINO依赖
            if capabilities.get('openvino', False):
                results['openvino'] = self._install_openvino()
            
            # GPIO依赖
            if capabilities.get('gpio', False):
                results['gpio_library'] = self._install_gpio_library()
            
        except Exception as e:
            logger.error(f"依赖安装失败: {e}")
            results['error'] = str(e)
        
        return results
    
    def _install_pycoral(self) -> bool:
        """安装PyCoral"""
        try:
            import pycoral
            return True
        except ImportError:
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'pycoral'], check=True)
                return True
            except subprocess.CalledProcessError:
                return False
    
    def _install_tflite_runtime(self) -> bool:
        """安装TensorFlow Lite Runtime"""
        try:
            import tflite_runtime
            return True
        except ImportError:
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'tflite-runtime'], check=True)
                return True
            except subprocess.CalledProcessError:
                return False
    
    def _check_torch_cuda(self) -> bool:
        """检查PyTorch CUDA支持"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _check_tensorrt(self) -> bool:
        """检查TensorRT支持"""
        try:
            import tensorrt
            return True
        except ImportError:
            return False
    
    def _install_openvino(self) -> bool:
        """安装OpenVINO"""
        try:
            import openvino
            return True
        except ImportError:
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'openvino'], check=True)
                return True
            except subprocess.CalledProcessError:
                return False
    
    def _install_gpio_library(self) -> bool:
        """安装GPIO库"""
        gpio_lib = self.board_config['gpio_config']['gpio_library']
        
        if gpio_lib == 'RPi.GPIO':
            try:
                import RPi.GPIO
                return True
            except ImportError:
                try:
                    subprocess.run([sys.executable, '-m', 'pip', 'install', 'RPi.GPIO'], check=True)
                    return True
                except subprocess.CalledProcessError:
                    return False
        
        elif gpio_lib == 'Jetson.GPIO':
            try:
                import Jetson.GPIO
                return True
            except ImportError:
                try:
                    subprocess.run([sys.executable, '-m', 'pip', 'install', 'Jetson.GPIO'], check=True)
                    return True
                except subprocess.CalledProcessError:
                    return False
        
        return True  # 通用GPIO不需要特殊库
    
    def generate_board_report(self) -> str:
        """生成开发板报告"""
        board_info = self.get_board_info()
        recommendations = self.get_optimization_recommendations()
        
        report = f"""
# AIoT开发板兼容性报告

## 检测到的开发板
- **名称**: {self.current_board['name']}
- **置信度**: {self.current_board['confidence']:.2f}
- **检测方法**: {self.current_board['detection_method']}

## 硬件信息
"""
        
        if self.current_board['name'] != 'Unknown':
            info = self.current_board['info']
            report += f"""- **制造商**: {info.get('manufacturer', 'Unknown')}
- **CPU**: {info.get('cpu', 'Unknown')}
- **GPU**: {info.get('gpu', 'None')}
- **内存**: {info.get('memory', 'Unknown')}
- **AI加速器**: {info.get('ai_accelerator', 'None')}
"""
        
        report += f"""
## 支持的能力
"""
        
        capabilities = self.current_board.get('capabilities', {})
        for capability, supported in capabilities.items():
            status = '✓' if supported else '✗'
            report += f"- **{capability.replace('_', ' ').title()}**: {status}\n"
        
        report += f"""
## 优化配置
- **最大工作线程**: {self.board_config['optimization']['max_workers']}
- **内存限制**: {self.board_config['optimization']['memory_limit_gb']:.1f} GB
- **批处理大小**: {self.board_config['optimization']['batch_size']}
- **最大图像尺寸**: {self.board_config['optimization']['image_max_size']}
- **GPU加速**: {'启用' if self.board_config['optimization']['use_gpu'] else '禁用'}
- **AI加速器**: {'启用' if self.board_config['optimization']['use_ai_accelerator'] else '禁用'}

## 推荐的AI框架
"""
        
        frameworks = self.board_config['ai_acceleration']['frameworks']
        for framework in frameworks:
            report += f"- {framework}\n"
        
        report += f"""
## 优化建议

### 模型优化
"""
        for suggestion in recommendations['model_optimization']:
            report += f"- {suggestion}\n"
        
        report += f"""
### 运行时优化
"""
        for suggestion in recommendations['runtime_optimization']:
            report += f"- {suggestion}\n"
        
        report += f"""
### 硬件优化
"""
        for suggestion in recommendations['hardware_optimization']:
            report += f"- {suggestion}\n"
        
        return report

# 便捷函数
def get_aiot_boards_adapter() -> AIoTBoardsAdapter:
    """获取AIoT开发板适配器"""
    return AIoTBoardsAdapter()

def detect_current_aiot_board() -> Dict[str, Any]:
    """检测当前AIoT开发板"""
    adapter = get_aiot_boards_adapter()
    return adapter.get_board_info()

if __name__ == "__main__":
    # 测试AIoT开发板适配器
    adapter = AIoTBoardsAdapter()
    
    # 生成开发板报告
    report = adapter.generate_board_report()
    print(report)
    
    # 获取优化建议
    recommendations = adapter.get_optimization_recommendations()
    print(f"\n优化建议:")
    print(f"模型优化: {recommendations['model_optimization']}")
    print(f"框架推荐: {recommendations['framework_recommendations']}")
    
    # 安装依赖
    print(f"\n安装依赖...")
    install_results = adapter.install_board_dependencies()
    for dep, success in install_results.items():
        status = '✓' if success else '✗'
        print(f"{dep}: {status}")