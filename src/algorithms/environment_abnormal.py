#!/usr/bin/env python3
"""
环境异常检测算法 - ID 6-9, 47-49
"""

import cv2
import numpy as np
from typing import Dict, Any

from .algorithm_base import AlgorithmBase, AlgorithmResult, AlgorithmCategory


class FireDetectionAlgorithm(AlgorithmBase):
    """火焰报警 - ID 6"""
    ALGORITHM_ID = 6
    ALGORITHM_NAME = "火焰报警"
    CATEGORY = AlgorithmCategory.ENVIRONMENT_ABNORMAL

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
        
        lower_fire = np.array([0, 50, 50])
        upper_fire = np.array([35, 255, 255])
        fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)
        
        fire_pixels = cv2.countNonZero(fire_mask)
        
        if fire_pixels > 500:
            result.detected = True
            result.confidence = min(0.9, fire_pixels / 5000)
            contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w_box, h_box = cv2.boundingRect(contours[0])
                result.bounding_box = (x, y, w_box, h_box)

        return result


class SmokeDetectionAlgorithm(AlgorithmBase):
    """烟雾报警 - ID 7"""
    ALGORITHM_ID = 7
    ALGORITHM_NAME = "烟雾报警"
    CATEGORY = AlgorithmCategory.ENVIRONMENT_ABNORMAL

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
        
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        brightness = np.mean(gray)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        avg_saturation = np.mean(sat)
        
        if avg_saturation < 50 and brightness > 100:
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges) / (h * w * 255)
            
            if edge_density < 0.05:
                result.detected = True
                result.confidence = 0.65
                result.bounding_box = (0, 0, w, h)

        return result


class FireEquipmentAlgorithm(AlgorithmBase):
    """消防设施检测 - ID 8"""
    ALGORITHM_ID = 8
    ALGORITHM_NAME = "消防设施检测"
    CATEGORY = AlgorithmCategory.ENVIRONMENT_ABNORMAL

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
        
        if red_pixels > 500:
            result.detected = True
            result.confidence = 0.6
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w_box, h_box = cv2.boundingRect(contours[0])
                result.bounding_box = (x, y, w_box, h_box)

        return result


class DebrisDetectionAlgorithm(AlgorithmBase):
    """杂物堆放 - ID 9"""
    ALGORITHM_ID = 9
    ALGORITHM_NAME = "杂物堆放"
    CATEGORY = AlgorithmCategory.ENVIRONMENT_ABNORMAL

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
        
        large_contours = [c for c in contours if cv2.contourArea(c) > 1000]
        
        if len(large_contours) > 3:
            result.detected = True
            result.confidence = 0.6
            result.bounding_box = (0, 0, w, h)

        return result


class CameraOcclusionAlgorithm(AlgorithmBase):
    """摄像头遮挡 - ID 47"""
    ALGORITHM_ID = 47
    ALGORITHM_NAME = "摄像头遮挡"
    CATEGORY = AlgorithmCategory.ENVIRONMENT_ABNORMAL

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
        
        brightness = np.mean(gray)
        std_dev = np.std(gray)
        
        if brightness < 30 or std_dev < 20:
            result.detected = True
            result.confidence = 0.8
            result.bounding_box = (0, 0, w, h)

        return result


class CameraShiftAlgorithm(AlgorithmBase):
    """摄像头偏移 - ID 48"""
    ALGORITHM_ID = 48
    ALGORITHM_NAME = "摄像头偏移"
    CATEGORY = AlgorithmCategory.ENVIRONMENT_ABNORMAL

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.reference_frame = None
        self.frame_count = 0

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
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.reference_frame is None or self.frame_count % 100 == 0:
            self.reference_frame = gray
            self.frame_count += 1
            return result

        self.frame_count += 1

        frame_delta = cv2.absdiff(self.reference_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        changed_pixels = cv2.countNonZero(thresh)
        total_pixels = h * w

        if changed_pixels / total_pixels > 0.5:
            result.detected = True
            result.confidence = 0.7
            result.bounding_box = (0, 0, w, h)
            self.reference_frame = gray

        return result


class LeakDetectionAlgorithm(AlgorithmBase):
    """跑冒滴漏 - ID 49"""
    ALGORITHM_ID = 49
    ALGORITHM_NAME = "跑冒滴漏"
    CATEGORY = AlgorithmCategory.ENVIRONMENT_ABNORMAL

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.previous_frames = []

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

        self.previous_frames.append(gray)
        if len(self.previous_frames) > 10:
            self.previous_frames.pop(0)

        if len(self.previous_frames) >= 5:
            diffs = []
            for i in range(1, len(self.previous_frames)):
                diff = cv2.absdiff(self.previous_frames[i-1], self.previous_frames[i])
                diffs.append(diff)

            if diffs:
                avg_diff = np.mean(diffs, axis=0).astype(np.uint8)
                motion_pixels = cv2.countNonZero(avg_diff > 10)
                
                if motion_pixels > 1000:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    water_mask = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([130, 255, 255]))
                    water_pixels = cv2.countNonZero(water_mask)
                    
                    if water_pixels > 500:
                        result.detected = True
                        result.confidence = 0.65
                        result.bounding_box = (0, 0, w, h)

        return result
