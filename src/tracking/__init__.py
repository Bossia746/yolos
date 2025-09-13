#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跟踪模块
提供多目标跟踪功能，支持视频检测中的目标连续跟踪
"""

from .multi_object_tracker import (
    MultiObjectTracker,
    TrackingConfig,
    TrackingStrategy,
    TrackState,
    Detection,
    Track
)

from .tracking_integration import (
    TrackingIntegration,
    IntegratedTrackingConfig,
    TrackingMode
)

__all__ = [
    'MultiObjectTracker',
    'TrackingConfig',
    'TrackingStrategy',
    'TrackState',
    'Detection',
    'Track',
    'TrackingIntegration',
    'IntegratedTrackingConfig',
    'TrackingMode'
]

__version__ = '1.0.0'
__author__ = 'YOLOS Team'
__description__ = '多目标跟踪系统，基于YOLOV论文思想实现'