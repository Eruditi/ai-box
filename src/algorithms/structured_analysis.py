#!/usr/bin/env python3
"""
结构化解析算法 - ID 25-29
基于 YOLOv8 目标检测
"""

import cv2
import numpy as np
from typing import Dict, Any

from .algorithm_base import AlgorithmBase, AlgorithmResult, AlgorithmCategory
from .yolo_engine import get_yolo_engine


class FaceDetectionAlgorithm(AlgorithmBase):
    """人脸 - ID 25"""
    ALGORITHM_ID = 25
    ALGORITHM_NAME = "人脸"
    CATEGORY = AlgorithmCategory.STRUCTURED_ANALYSIS

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.yolo = None

    def initialize(self) -> bool:
        self.yolo = get_yolo_engine()
        return True

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        detections = self.yolo.detect(frame, classes=[0])
        faces = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            face_h = y2 - y1
            face_w = x2 - x1
            if 0.5 < face_h / face_w < 2.0 and face_h * face_w > 1000:
                faces.append(det)

        if faces:
            best = max(faces, key=lambda d: d['confidence'])
            x1, y1, x2, y2 = best['bbox']
            result.detected = True
            result.confidence = best['confidence']
            result.bounding_box = (x1, y1, x2 - x1, y2 - y1)
            result.extra_data = {'face_count': len(faces)}

        return result


class HumanShapeAlgorithm(AlgorithmBase):
    """人形 - ID 26"""
    ALGORITHM_ID = 26
    ALGORITHM_NAME = "人形"
    CATEGORY = AlgorithmCategory.STRUCTURED_ANALYSIS

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.yolo = None

    def initialize(self) -> bool:
        self.yolo = get_yolo_engine()
        return True

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        detections = self.yolo.detect(frame, classes=[0])
        if detections:
            best = max(detections, key=lambda d: d['confidence'])
            x1, y1, x2, y2 = best['bbox']
            result.detected = True
            result.confidence = best['confidence']
            result.bounding_box = (x1, y1, x2 - x1, y2 - y1)
            result.extra_data = {'person_count': len(detections)}

        return result


class MotorVehicleAlgorithm(AlgorithmBase):
    """机动车 - ID 27"""
    ALGORITHM_ID = 27
    ALGORITHM_NAME = "机动车"
    CATEGORY = AlgorithmCategory.STRUCTURED_ANALYSIS

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.yolo = None

    def initialize(self) -> bool:
        self.yolo = get_yolo_engine()
        return True

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        detections = self.yolo.detect(frame, classes=[2, 5, 7])
        if detections:
            best = max(detections, key=lambda d: d['confidence'])
            x1, y1, x2, y2 = best['bbox']
            result.detected = True
            result.confidence = best['confidence']
            result.bounding_box = (x1, y1, x2 - x1, y2 - y1)
            result.extra_data = {
                'vehicle_count': len(detections),
                'vehicle_types': [d['class_name'] for d in detections]
            }

        return result


class NonMotorVehicleAlgorithm(AlgorithmBase):
    """非机动车 - ID 28"""
    ALGORITHM_ID = 28
    ALGORITHM_NAME = "非机动车"
    CATEGORY = AlgorithmCategory.STRUCTURED_ANALYSIS

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.yolo = None

    def initialize(self) -> bool:
        self.yolo = get_yolo_engine()
        return True

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        detections = self.yolo.detect(frame, classes=[1, 3])
        if detections:
            best = max(detections, key=lambda d: d['confidence'])
            x1, y1, x2, y2 = best['bbox']
            result.detected = True
            result.confidence = best['confidence']
            result.bounding_box = (x1, y1, x2 - x1, y2 - y1)
            result.extra_data = {
                'vehicle_count': len(detections),
                'vehicle_types': [d['class_name'] for d in detections]
            }

        return result


class LicensePlateAlgorithm(AlgorithmBase):
    """车牌 - ID 29"""
    ALGORITHM_ID = 29
    ALGORITHM_NAME = "车牌"
    CATEGORY = AlgorithmCategory.STRUCTURED_ANALYSIS

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.yolo = None

    def initialize(self) -> bool:
        self.yolo = get_yolo_engine()
        return True

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        vehicle_dets = self.yolo.detect(frame, classes=[2, 5, 7])
        for vdet in vehicle_dets:
            x1, y1, x2, y2 = vdet['bbox']
            vh = y2 - y1
            plate_region = frame[y2 - vh // 4:y2, x1:x2]
            if plate_region.size == 0:
                continue

            hsv = cv2.cvtColor(plate_region, cv2.COLOR_BGR2HSV)
            blue_plate = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
            yellow_plate = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))
            green_plate = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
            plate_mask = cv2.bitwise_or(cv2.bitwise_or(blue_plate, yellow_plate), green_plate)

            plate_pixels = cv2.countNonZero(plate_mask)
            region_area = plate_region.shape[0] * plate_region.shape[1]

            if region_area > 0 and plate_pixels / region_area > 0.15:
                result.detected = True
                result.confidence = min(0.85, vdet['confidence'] + 0.1)
                result.bounding_box = (x1, y2 - vh // 4, x2 - x1, vh // 4)
                result.extra_data = {'plate_color_ratio': float(plate_pixels / region_area)}
                break

        return result
