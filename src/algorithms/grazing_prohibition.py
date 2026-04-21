#!/usr/bin/env python3
"""
禁牧识别算法
使用 YOLO 检测牲畜（牛、羊、马等）
"""

import cv2
import numpy as np
from typing import Dict, Any, List

from .algorithm_base import AlgorithmBase, AlgorithmResult, AlgorithmCategory
from .yolo_engine import YOLOEngine

LIVESTOCK_CLASSES = [17, 18, 19]


class LivestockDetectionAlgorithm(AlgorithmBase):
    """牲畜检测 - ID: 65"""
    ALGORITHM_ID = 65
    ALGORITHM_NAME = "牲畜检测"
    CATEGORY = AlgorithmCategory.STRUCTURED_ANALYSIS

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.yolo = None

    def initialize(self) -> bool:
        try:
            self.yolo = YOLOEngine()
            return self.yolo.is_available()
        except Exception:
            return False

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        if self.yolo is None or not self.yolo.is_available():
            return result

        if not self._is_valid_frame(frame):
            return result

        detections = self.yolo.detect(frame, classes=LIVESTOCK_CLASSES, conf=0.3)

        livestock_boxes = []
        for det in detections:
            bbox = det['bbox']
            x, y, x2, y2 = bbox
            w, h = x2 - x, y2 - y
            livestock_boxes.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'type': det['class_name'],
                'confidence': det['confidence']
            })

        if len(livestock_boxes) > 0:
            result.detected = True
            result.confidence = max(b['confidence'] for b in livestock_boxes)
            result.extra_data = {
                'livestock_count': len(livestock_boxes),
                'livestock_boxes': livestock_boxes
            }

        return result


class GrazingProhibitionAlgorithm(AlgorithmBase):
    """禁牧区域识别 - ID: 66"""
    ALGORITHM_ID = 66
    ALGORITHM_NAME = "禁牧区域识别"
    CATEGORY = AlgorithmCategory.PERIMETER_ALERT

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.yolo = None
        self.prohibited_areas = config.get('prohibited_areas', [])

    def initialize(self) -> bool:
        try:
            self.yolo = YOLOEngine()
            return self.yolo.is_available()
        except Exception:
            return False

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        if self.yolo is None or not self.yolo.is_available():
            return result

        if not self._is_valid_frame(frame):
            return result

        detections = self.yolo.detect(frame, classes=LIVESTOCK_CLASSES, conf=0.3)

        livestock_in_prohibited = []
        for det in detections:
            bbox = det['bbox']
            x, y, x2, y2 = bbox
            cx, cy = (x + x2) // 2, (y + y2) // 2

            for area in self.prohibited_areas:
                if isinstance(area, dict) and all(k in area for k in ['x', 'y', 'w', 'h']):
                    ax, ay, aw, ah = area['x'], area['y'], area['w'], area['h']
                    if ax <= cx <= ax + aw and ay <= cy <= ay + ah:
                        livestock_in_prohibited.append({
                            'x': x, 'y': y, 'w': x2 - x, 'h': y2 - y,
                            'type': det['class_name'],
                            'confidence': det['confidence']
                        })
                        break

        if len(livestock_in_prohibited) > 0:
            result.detected = True
            result.confidence = max(b['confidence'] for b in livestock_in_prohibited)
            result.extra_data = {
                'violation_count': len(livestock_in_prohibited),
                'violation_boxes': livestock_in_prohibited,
                'prohibited_areas': self.prohibited_areas
            }

        return result


class GrazingMonitoringAlgorithm(AlgorithmBase):
    """放牧监控 - ID: 67"""
    ALGORITHM_ID = 67
    ALGORITHM_NAME = "放牧监控"
    CATEGORY = AlgorithmCategory.BEHAVIOR_ALERT

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.yolo = None
        self.livestock_history = []

    def initialize(self) -> bool:
        try:
            self.yolo = YOLOEngine()
            return self.yolo.is_available()
        except Exception:
            return False

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        if self.yolo is None or not self.yolo.is_available():
            return result

        if not self._is_valid_frame(frame):
            return result

        detections = self.yolo.detect(frame, classes=LIVESTOCK_CLASSES, conf=0.3)
        livestock_count = len(detections)

        self.livestock_history.append(livestock_count)
        if len(self.livestock_history) > 10:
            self.livestock_history.pop(0)

        if len(self.livestock_history) >= 5:
            avg_count = sum(self.livestock_history) / len(self.livestock_history)
            max_count = max(self.livestock_history)

            result.detected = True
            result.confidence = min(1.0, livestock_count / 5)
            result.extra_data = {
                'current_livestock': livestock_count,
                'average_livestock': round(avg_count, 1),
                'max_livestock': max_count,
                'grazing_intensity': 'high' if avg_count > 5 else 'medium' if avg_count > 2 else 'low'
            }

        return result
