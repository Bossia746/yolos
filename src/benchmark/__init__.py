#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOS 基准测试模块
提供多平台性能基准测试功能
"""

from .performance_benchmark import (
    PerformanceBenchmark,
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkType,
    TestScenario,
    SystemMonitor,
    SyntheticDataGenerator
)

from .benchmark_runner import BenchmarkRunner

__all__ = [
    'PerformanceBenchmark',
    'BenchmarkConfig', 
    'BenchmarkResult',
    'BenchmarkType',
    'TestScenario',
    'SystemMonitor',
    'SyntheticDataGenerator',
    'BenchmarkRunner'
]

__version__ = '1.0.0'
__author__ = 'YOLOS Team'
__description__ = 'Multi-platform performance benchmark system for YOLOS'