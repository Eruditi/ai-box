#!/usr/bin/env python3
"""
禁牧识别算法
"""

import cv2
import numpy as np
from typing import Dict, Any, List

from .algorithm_base import AlgorithmBase, AlgorithmResult, AlgorithmCategory


class LivestockDetectionAlgorithm(AlgorithmBase):
    """牲畜检测 - ID: 65"""
    ALGORITHM_ID = 65
    ALGORITHM_NAME = "牲畜检测"
    CATEGORY = AlgorithmCategory.STRUCTURED_ANALYSIS

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.hog = None

    def initialize(self) -> bool:
        try:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            return True
        except Exception:
            return False

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        if self.hog is None:
            return result

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes, weights = self.hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)

        # 过滤掉可能是人的检测结果，保留可能是牲畜的结果
        livestock_boxes = []
        for (x, y, w, h) in boxes:
            # 牲畜通常比人更宽或更矮
            aspect_ratio = w / h
            if 0.8 < aspect_ratio < 1.5:
                livestock_boxes.append((x, y, w, h))

        if len(livestock_boxes) > 0:
            result.detected = True
            result.confidence = 0.8
            result.extra_data = {
                'livestock_count': len(livestock_boxes),
                'livestock_boxes': [{'x': x, 'y': y, 'w': w, 'h': h} for (x, y, w, h) in livestock_boxes]
            }

        return result


class GrazingProhibitionAlgorithm(AlgorithmBase):
    """禁牧区域识别 - ID: 66"""
    ALGORITHM_ID = 66
    ALGORITHM_NAME = "禁牧区域识别"
    CATEGORY = AlgorithmCategory.PERIMETER_ALERT

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.hog = None
        self.prohibited_areas = config.get('prohibited_areas', [])

    def initialize(self) -> bool:
        try:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            return True
        except Exception:
            return False

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        if self.hog is None:
            return result

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes, weights = self.hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)

        # 检测禁牧区域内的牲畜
        livestock_in_prohibited = []
        for (x, y, w, h) in boxes:
            # 过滤可能的牲畜
            aspect_ratio = w / h
            if 0.8 < aspect_ratio < 1.5:
                # 检查是否在禁牧区域内
                for area in self.prohibited_areas:
                    if isinstance(area, dict) and 'x' in area and 'y' in area and 'w' in area and 'h' in area:
                        ax, ay, aw, ah = area['x'], area['y'], area['w'], area['h']
                        if ax < x < ax + aw and ay < y < ay + ah:
                            livestock_in_prohibited.append((x, y, w, h))

        if len(livestock_in_prohibited) > 0:
            result.detected = True
            result.confidence = 0.9
            result.extra_data = {
                'violation_count': len(livestock_in_prohibited),
                'violation_boxes': [{'x': x, 'y': y, 'w': w, 'h': h} for (x, y, w, h) in livestock_in_prohibited],
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
        self.hog = None
        self.livestock_history = []

    def initialize(self) -> bool:
        try:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            return True
        except Exception:
            return False

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        if self.hog is None:
            return result

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes, weights = self.hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)

        # 检测牲畜
        livestock_count = 0
        for (x, y, w, h) in boxes:
            aspect_ratio = w / h
            if 0.8 < aspect_ratio < 1.5:
                livestock_count += 1

        self.livestock_history.append(livestock_count)
        if len(self.livestock_history) > 10:
            self.livestock_history.pop(0)

        if len(self.livestock_history) > 5:
            # 分析放牧活动
            avg_count = sum(self.livestock_history) / len(self.livestock_history)
            max_count = max(self.livestock_history)
            
            result.detected = True
            result.confidence = 0.8
            result.extra_data = {
                'current_livestock': livestock_count,
                'average_livestock': avg_count,
                'max_livestock': max_count,
                'grazing_intensity': 'high' if avg_count > 5 else 'medium' if avg_count > 2 else 'low'
            }

        return result