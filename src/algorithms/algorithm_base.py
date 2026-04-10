#!/usr/bin/env python3
"""
算法基类 - 所有算法的基础类
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import cv2
import numpy as np


class AlgorithmCategory(Enum):
    """算法类型枚举"""
    FACE_RECOGNITION = 1
    STRUCTURED_ANALYSIS = 2
    PERSON_VIOLATION = 3
    ENVIRONMENT_ABNORMAL = 4
    PERIMETER_ALERT = 5
    BEHAVIOR_ALERT = 6


@dataclass
class AlgorithmResult:
    """算法结果"""
    algorithm_id: int
    algorithm_name: str
    category: AlgorithmCategory
    detected: bool = False
    confidence: float = 0.0
    bounding_box: Optional[Tuple[int, int, int, int]] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'algorithm_id': self.algorithm_id,
            'algorithm_name': self.algorithm_name,
            'category': self.category.name,
            'detected': self.detected,
            'confidence': float(self.confidence),
            'bounding_box': list(self.bounding_box) if self.bounding_box else None,
            'extra_data': self.extra_data,
            'timestamp': self.timestamp
        }


class AlgorithmBase(ABC):
    """算法基类"""
    
    ALGORITHM_ID: int = 0
    ALGORITHM_NAME: str = ""
    CATEGORY: AlgorithmCategory = AlgorithmCategory.PERSON_VIOLATION

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)

    @abstractmethod
    def initialize(self) -> bool:
        """初始化算法"""
        pass

    @abstractmethod
    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        """处理帧并返回结果"""
        pass

    def visualize(self, frame: np.ndarray, result: AlgorithmResult) -> np.ndarray:
        """可视化结果"""
        if result.detected and result.bounding_box:
            x, y, w, h = result.bounding_box
            color = self._get_alert_color(result.category)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{result.algorithm_name}: {result.confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def _get_alert_color(self, category: AlgorithmCategory) -> Tuple[int, int, int]:
        """获取告警颜色"""
        color_map = {
            AlgorithmCategory.PERSON_VIOLATION: (0, 0, 255),
            AlgorithmCategory.ENVIRONMENT_ABNORMAL: (0, 165, 255),
            AlgorithmCategory.PERIMETER_ALERT: (255, 0, 0),
            AlgorithmCategory.BEHAVIOR_ALERT: (255, 0, 255),
            AlgorithmCategory.STRUCTURED_ANALYSIS: (0, 255, 0),
            AlgorithmCategory.FACE_RECOGNITION: (255, 255, 0),
        }
        return color_map.get(category, (0, 255, 0))

    def release(self):
        """释放资源"""
        pass
