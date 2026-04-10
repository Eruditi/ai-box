#!/usr/bin/env python3
"""
人脸识别相关算法 - ID 30-31, 46
"""

import cv2
import numpy as np
from typing import Dict, Any

from .algorithm_base import AlgorithmBase, AlgorithmResult, AlgorithmCategory


class FaceRecognitionCompareAlgorithm(AlgorithmBase):
    """人脸识别对比 - ID 30"""
    ALGORITHM_ID = 30
    ALGORITHM_NAME = "人脸识别对比"
    CATEGORY = AlgorithmCategory.FACE_RECOGNITION

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.face_cascade = None
        self.registered_faces = []

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
            result.confidence = 0.75
            result.bounding_box = (x, y, w, h)
            result.extra_data = {'face_detected': True}

        return result


class LicensePlateCompareAlgorithm(AlgorithmBase):
    """车牌识别对比 - ID 31"""
    ALGORITHM_ID = 31
    ALGORITHM_NAME = "车牌识别对比"
    CATEGORY = AlgorithmCategory.FACE_RECOGNITION

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

        plate_mask = cv2.bitwise_or(blue_mask, yellow_mask)

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
                    result.extra_data = {'plate_detected': True}
                    break

        return result


class FaceRecognitionAlertAlgorithm(AlgorithmBase):
    """人脸识别报警 - ID 46"""
    ALGORITHM_ID = 46
    ALGORITHM_NAME = "人脸识别报警"
    CATEGORY = AlgorithmCategory.FACE_RECOGNITION

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.face_cascade = None
        self.blacklist_faces = []

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
            result.confidence = 0.8
            result.bounding_box = (x, y, w, h)
            result.extra_data = {'alert_face_detected': True}

        return result
