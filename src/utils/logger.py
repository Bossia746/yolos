#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志工具 - 简化版本
用于AIoT兼容性测试
"""

import logging
import sys
from typing import Optional

def setup_logger(name: str = "yolos", level: str = "INFO") -> logging.Logger:
    """设置日志器"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, level.upper()))
    
    return logger

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """获取日志器"""
    return logging.getLogger(name or "yolos")