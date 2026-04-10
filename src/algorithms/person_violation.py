#!/usr/bin/env python3
"""
人员违规检测算法 - ID 1-5, 37, 41
"""

import cv2
import numpy as np
from typing import Dict, Any

from .algorithm_base import AlgorithmBase, AlgorithmResult, AlgorithmCategory


class NoHelmetAlgorithm(AlgorithmBase):
    """未佩戴安全帽报警 - ID 1"""
    ALGORITHM_ID = 1
    ALGORITHM_NAME = "未佩戴安全帽报警"
    CATEGORY = AlgorithmCategory.PERSON_VIOLATION

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.face_cascade = None

    def initialize(self) -> bool:
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            return True
        except Exception as e:
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
            for (x, y, w, h) in faces:
                helmet_region = frame[max(0, y - h//2):y, x:x + w]
                if helmet_region.size > 0:
                    hsv = cv2.cvtColor(helmet_region, cv2.COLOR_BGR2HSV)
                    yellow_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255]))
                    orange_mask = cv2.inRange(hsv, np.array([5, 100, 100]), np.array([15, 255, 255]))
                    white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
                    
                    yellow_pixels = cv2.countNonZero(yellow_mask)
                    orange_pixels = cv2.countNonZero(orange_mask)
                    white_pixels = cv2.countNonZero(white_mask)
                    
                    total_helmet_pixels = yellow_pixels + orange_pixels + white_pixels
                    region_area = helmet_region.shape[0] * helmet_region.shape[1]
                    
                    if region_area > 0 and total_helmet_pixels / region_area < 0.1:
                        result.detected = True
                        result.confidence = 0.75
                        result.bounding_box = (x, y, w, h)
                        break

        return result


class NoMaskAlgorithm(AlgorithmBase):
    """未戴口罩报警 - ID 2"""
    ALGORITHM_ID = 2
    ALGORITHM_NAME = "未戴口罩报警"
    CATEGORY = AlgorithmCategory.PERSON_VIOLATION

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.face_cascade = None

    def initialize(self) -> bool:
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            return True
        except Exception as e:
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
            for (x, y, w, h) in faces:
                mouth_region = frame[y + h//2:y + h, x:x + w]
                if mouth_region.size > 0:
                    hsv = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2HSV)
                    skin_mask1 = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255]))
                    skin_mask2 = cv2.inRange(hsv, np.array([170, 20, 70]), np.array([180, 255, 255]))
                    skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
                    
                    skin_pixels = cv2.countNonZero(skin_mask)
                    region_area = mouth_region.shape[0] * mouth_region.shape[1]
                    
                    if region_area > 0 and skin_pixels / region_area > 0.3:
                        result.detected = True
                        result.confidence = 0.7
                        result.bounding_box = (x, y, w, h)
                        break

        return result


class NoWorkwearAlgorithm(AlgorithmBase):
    """未穿戴工作服报警 - ID 3"""
    ALGORITHM_ID = 3
    ALGORITHM_NAME = "未穿戴工作服报警"
    CATEGORY = AlgorithmCategory.PERSON_VIOLATION

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

    def initialize(self) -> bool:
        return True

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        h, w = frame.shape[:2]
        roi = frame[h//3:, :]
        
        if roi.size > 0:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
            
            blue_pixels = cv2.countNonZero(blue_mask)
            roi_area = roi.shape[0] * roi.shape[1]
            
            if roi_area > 0 and blue_pixels / roi_area < 0.05:
                result.detected = True
                result.confidence = 0.65
                result.bounding_box = (0, h//3, w, h*2//3)

        return result


class NoSafetyBeltAlgorithm(AlgorithmBase):
    """未佩戴安全带报警 - ID 4"""
    ALGORITHM_ID = 4
    ALGORITHM_NAME = "未佩戴安全带报警"
    CATEGORY = AlgorithmCategory.PERSON_VIOLATION

    def initialize(self) -> bool:
        return True

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        h, w = frame.shape[:2]
        roi = frame[:, w//4:w*3//4]
        
        if roi.size > 0:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
            
            if lines is None or len(lines) < 2:
                result.detected = True
                result.confidence = 0.6
                result.bounding_box = (w//4, 0, w//2, h)

        return result


class NoReflectiveVestAlgorithm(AlgorithmBase):
    """未佩戴反光衣报警 - ID 5"""
    ALGORITHM_ID = 5
    ALGORITHM_NAME = "未佩戴反光衣报警"
    CATEGORY = AlgorithmCategory.PERSON_VIOLATION

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
        
        yellow_mask = cv2.inRange(hsv, np.array([20, 100, 150]), np.array([40, 255, 255]))
        orange_mask = cv2.inRange(hsv, np.array([5, 100, 150]), np.array([15, 255, 255]))
        reflective_mask = cv2.bitwise_or(yellow_mask, orange_mask)
        
        reflective_pixels = cv2.countNonZero(reflective_mask)
        total_area = h * w
        
        if total_area > 0 and reflective_pixels / total_area < 0.01:
            result.detected = True
            result.confidence = 0.6
            result.bounding_box = (0, 0, w, h)

        return result


class NoHelmetRidingAlgorithm(AlgorithmBase):
    """骑车未带安全帽 - ID 37"""
    ALGORITHM_ID = 37
    ALGORITHM_NAME = "骑车未带安全帽"
    CATEGORY = AlgorithmCategory.PERSON_VIOLATION

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
            for (x, y, w, h) in faces:
                result.detected = True
                result.confidence = 0.7
                result.bounding_box = (x, y, w, h)
                break

        return result


class MotorcycleInGasStationAlgorithm(AlgorithmBase):
    """骑摩托车进加油站 - ID 41"""
    ALGORITHM_ID = 41
    ALGORITHM_NAME = "骑摩托车进加油站"
    CATEGORY = AlgorithmCategory.PERSON_VIOLATION

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
        
        red_mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        red_pixels = cv2.countNonZero(red_mask)
        
        if red_pixels > 1000:
            result.detected = True
            result.confidence = 0.65
            result.bounding_box = (0, 0, w, h)

        return result
