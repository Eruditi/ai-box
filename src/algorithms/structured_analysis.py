#!/usr/bin/env python3
"""
全结构化解析算法 - ID 25-29
"""

import cv2
import numpy as np
from typing import Dict, Any

from .algorithm_base import AlgorithmBase, AlgorithmResult, AlgorithmCategory


class FaceDetectionAlgorithm(AlgorithmBase):
    """人脸 - ID 25"""
    ALGORITHM_ID = 25
    ALGORITHM_NAME = "人脸"
    CATEGORY = AlgorithmCategory.STRUCTURED_ANALYSIS

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.face_cascade = None

    def initialize(self) -> bool:
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            return True
        except Exception:
            return False

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        if self.face_cascade is None:
            return result

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            result.detected = True
            result.confidence = 0.85
            result.bounding_box = (x, y, w, h)

        return result


class HumanShapeAlgorithm(AlgorithmBase):
    """人形 - ID 26"""
    ALGORITHM_ID = 26
    ALGORITHM_NAME = "人形"
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

        if len(boxes) > 0:
            x, y, w, h = boxes[0]
            result.detected = True
            result.confidence = float(weights[0]) if len(weights) > 0 else 0.7
            result.bounding_box = (x, y, w, h)

        return result


class MotorVehicleAlgorithm(AlgorithmBase):
    """机动车 - ID 27"""
    ALGORITHM_ID = 27
    ALGORITHM_NAME = "机动车"
    CATEGORY = AlgorithmCategory.STRUCTURED_ANALYSIS

    def initialize(self) -> bool:
        return True

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if 5000 < area < 50000:
                x, y, w_box, h_box = cv2.boundingRect(contour)
                aspect_ratio = w_box / h_box if h_box > 0 else 0

                if 1.2 < aspect_ratio < 3.0:
                    result.detected = True
                    result.confidence = 0.7
                    result.bounding_box = (x, y, w_box, h_box)
                    break

        return result


class NonMotorVehicleAlgorithm(AlgorithmBase):
    """非机动车 - ID 28"""
    ALGORITHM_ID = 28
    ALGORITHM_NAME = "非机动车"
    CATEGORY = AlgorithmCategory.STRUCTURED_ANALYSIS

    def initialize(self) -> bool:
        return True

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if 1000 < area < 10000:
                x, y, w_box, h_box = cv2.boundingRect(contour)
                aspect_ratio = w_box / h_box if h_box > 0 else 0

                if 0.8 < aspect_ratio < 2.0:
                    result.detected = True
                    result.confidence = 0.65
                    result.bounding_box = (x, y, w_box, h_box)
                    break

        return result


class LicensePlateAlgorithm(AlgorithmBase):
    """车牌 - ID 29"""
    ALGORITHM_ID = 29
    ALGORITHM_NAME = "车牌"
    CATEGORY = AlgorithmCategory.STRUCTURED_ANALYSIS

    def initialize(self) -> bool:
        return True

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
        yellow_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([40, 255, 255]))
        green_mask = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))

        plate_mask = cv2.bitwise_or(blue_mask, yellow_mask)
        plate_mask = cv2.bitwise_or(plate_mask, green_mask)

        contours, _ = cv2.findContours(plate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 5000:
                x, y, w_box, h_box = cv2.boundingRect(contour)
                aspect_ratio = w_box / h_box if h_box > 0 else 0

                if 2.0 < aspect_ratio < 5.0:
                    result.detected = True
                    result.confidence = 0.7
                    result.bounding_box = (x, y, w_box, h_box)
                    break

        return result
