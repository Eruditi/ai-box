#!/usr/bin/env python3
"""
AI算法模块 - 支持50种智能分析算法
"""

from .algorithm_base import AlgorithmBase, AlgorithmResult
from .algorithm_manager import AlgorithmManager
from .person_violation import *
from .environment_abnormal import *
from .perimeter_alert import *
from .behavior_alert import *
from .structured_analysis import *
from .face_recognition import *

__all__ = [
    'AlgorithmBase',
    'AlgorithmResult',
    'AlgorithmManager',
]
