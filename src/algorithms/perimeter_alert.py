#!/usr/bin/env python3
"""
周界警戒算法 - ID 10-15, 32-36, 38-43
"""

import cv2
import numpy as np
from typing import Dict, Any, List

from .algorithm_base import AlgorithmBase, AlgorithmResult, AlgorithmCategory


class VehicleParkingAlgorithm(AlgorithmBase):
    """车辆禁停 - ID 10"""
    ALGORITHM_ID = 10
    ALGORITHM_NAME = "车辆禁停"
    CATEGORY = AlgorithmCategory.PERIMETER_ALERT

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
                    result.confidence = 0.65
                    result.bounding_box = (x, y, w_box, h_box)
                    break

        return result


class VehicleLeaveAlgorithm(AlgorithmBase):
    """车辆离开 - ID 11"""
    ALGORITHM_ID = 11
    ALGORITHM_NAME = "车辆离开"
    CATEGORY = AlgorithmCategory.PERIMETER_ALERT

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.previous_vehicle_count = 0

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
        
        vehicle_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5000 < area < 50000:
                vehicle_count += 1
        
        if self.previous_vehicle_count > 0 and vehicle_count < self.previous_vehicle_count:
            result.detected = True
            result.confidence = 0.6
            result.bounding_box = (0, 0, w, h)
        
        self.previous_vehicle_count = vehicle_count
        return result


class PersonLoiteringAlgorithm(AlgorithmBase):
    """人员徘徊 - ID 12"""
    ALGORITHM_ID = 12
    ALGORITHM_NAME = "人员徘徊"
    CATEGORY = AlgorithmCategory.PERIMETER_ALERT

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.tracked_positions = []
        self.loiter_threshold = 100

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
        
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                for (x, y, w_face, h_face) in faces:
                    center_x = x + w_face // 2
                    center_y = y + h_face // 2
                    self.tracked_positions.append((center_x, center_y))
                    
                    if len(self.tracked_positions) > 50:
                        self.tracked_positions.pop(0)
                    
                    if len(self.tracked_positions) >= 20:
                        positions = np.array(self.tracked_positions)
                        movement = np.std(positions, axis=0)
                        
                        if np.mean(movement) < 50:
                            result.detected = True
                            result.confidence = 0.7
                            result.bounding_box = (x, y, w_face, h_face)
                            break
        except Exception:
            pass

        return result


class ClimbOverAlgorithm(AlgorithmBase):
    """翻墙检测 - ID 13"""
    ALGORITHM_ID = 13
    ALGORITHM_NAME = "翻墙检测"
    CATEGORY = AlgorithmCategory.PERIMETER_ALERT

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
            if 1000 < area < 20000:
                x, y, w_box, h_box = cv2.boundingRect(contour)
                aspect_ratio = h_box / w_box if w_box > 0 else 0
                
                if aspect_ratio > 2.0 and y < h // 3:
                    result.detected = True
                    result.confidence = 0.65
                    result.bounding_box = (x, y, w_box, h_box)
                    break

        return result


class IntrusionAlgorithm(AlgorithmBase):
    """入侵 - ID 14"""
    ALGORITHM_ID = 14
    ALGORITHM_NAME = "入侵"
    CATEGORY = AlgorithmCategory.PERIMETER_ALERT

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

        if self.reference_frame is None or self.frame_count % 50 == 0:
            self.reference_frame = gray
            self.frame_count += 1
            return result

        if self.reference_frame.shape != gray.shape:
            self.reference_frame = gray
            self.frame_count += 1
            return result

        self.frame_count += 1

        frame_delta = cv2.absdiff(self.reference_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w_box, h_box = cv2.boundingRect(contour)
                result.detected = True
                result.confidence = 0.75
                result.bounding_box = (x, y, w_box, h_box)
                break

        return result


class CrossBorderAlgorithm(AlgorithmBase):
    """越界 - ID 15"""
    ALGORITHM_ID = 15
    ALGORITHM_NAME = "越界"
    CATEGORY = AlgorithmCategory.PERIMETER_ALERT

    def initialize(self) -> bool:
        return True

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        h, w = frame.shape[:2]
        
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w_face, h_face) in faces:
                if x < w // 4 or x + w_face > w * 3 // 4:
                    result.detected = True
                    result.confidence = 0.7
                    result.bounding_box = (x, y, w_face, h_face)
                    break
        except Exception:
            pass

        return result


class OverCapacityAlgorithm(AlgorithmBase):
    """超员 - ID 32"""
    ALGORITHM_ID = 32
    ALGORITHM_NAME = "超员"
    CATEGORY = AlgorithmCategory.PERIMETER_ALERT

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.capacity_threshold = config.get('capacity_threshold', 5)

    def initialize(self) -> bool:
        return True

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        h, w = frame.shape[:2]
        
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > self.capacity_threshold:
                result.detected = True
                result.confidence = 0.8
                result.bounding_box = (0, 0, w, h)
                result.extra_data = {'person_count': len(faces)}
        except Exception:
            pass

        return result


class UnderCapacityAlgorithm(AlgorithmBase):
    """少员 - ID 33"""
    ALGORITHM_ID = 33
    ALGORITHM_NAME = "少员"
    CATEGORY = AlgorithmCategory.PERIMETER_ALERT

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.min_capacity = config.get('min_capacity', 2)

    def initialize(self) -> bool:
        return True

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        h, w = frame.shape[:2]
        
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if 0 < len(faces) < self.min_capacity:
                result.detected = True
                result.confidence = 0.7
                result.bounding_box = (0, 0, w, h)
                result.extra_data = {'person_count': len(faces)}
        except Exception:
            pass

        return result


class PersonLeaveAlgorithm(AlgorithmBase):
    """人员离开 - ID 34"""
    ALGORITHM_ID = 34
    ALGORITHM_NAME = "人员离开"
    CATEGORY = AlgorithmCategory.PERIMETER_ALERT

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.previous_count = 0

    def initialize(self) -> bool:
        return True

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        h, w = frame.shape[:2]
        
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            current_count = len(faces)
            
            if self.previous_count > 0 and current_count < self.previous_count:
                result.detected = True
                result.confidence = 0.65
                result.bounding_box = (0, 0, w, h)
            
            self.previous_count = current_count
        except Exception:
            pass

        return result


class NonMotorParkingAlgorithm(AlgorithmBase):
    """非机动车禁停 - ID 35"""
    ALGORITHM_ID = 35
    ALGORITHM_NAME = "非机动车禁停"
    CATEGORY = AlgorithmCategory.PERIMETER_ALERT

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
                    result.confidence = 0.6
                    result.bounding_box = (x, y, w_box, h_box)
                    break

        return result


class NonMotorLeaveAlgorithm(AlgorithmBase):
    """非机动车离开 - ID 36"""
    ALGORITHM_ID = 36
    ALGORITHM_NAME = "非机动车离开"
    CATEGORY = AlgorithmCategory.PERIMETER_ALERT

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.previous_count = 0

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
        
        current_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 1000 < area < 10000:
                current_count += 1
        
        if self.previous_count > 0 and current_count < self.previous_count:
            result.detected = True
            result.confidence = 0.6
            result.bounding_box = (0, 0, w, h)
        
        self.previous_count = current_count
        return result


class VehicleOverCountAlgorithm(AlgorithmBase):
    """机动车超出数量 - ID 38"""
    ALGORITHM_ID = 38
    ALGORITHM_NAME = "机动车超出数量"
    CATEGORY = AlgorithmCategory.PERIMETER_ALERT

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.max_count = config.get('max_count', 3)

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
        
        vehicle_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5000 < area < 50000:
                vehicle_count += 1
        
        if vehicle_count > self.max_count:
            result.detected = True
            result.confidence = 0.7
            result.bounding_box = (0, 0, w, h)
            result.extra_data = {'vehicle_count': vehicle_count}

        return result


class VehicleUnderCountAlgorithm(AlgorithmBase):
    """机动车少于数量 - ID 39"""
    ALGORITHM_ID = 39
    ALGORITHM_NAME = "机动车少于数量"
    CATEGORY = AlgorithmCategory.PERIMETER_ALERT

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.min_count = config.get('min_count', 1)

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
        
        vehicle_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5000 < area < 50000:
                vehicle_count += 1
        
        if 0 < vehicle_count < self.min_count:
            result.detected = True
            result.confidence = 0.65
            result.bounding_box = (0, 0, w, h)
            result.extra_data = {'vehicle_count': vehicle_count}

        return result


class HazardVehicleAlgorithm(AlgorithmBase):
    """危化品车辆禁入 - ID 40"""
    ALGORITHM_ID = 40
    ALGORITHM_NAME = "危化品车辆禁入"
    CATEGORY = AlgorithmCategory.PERIMETER_ALERT

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
        
        yellow_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([40, 255, 255]))
        orange_mask = cv2.inRange(hsv, np.array([5, 100, 100]), np.array([15, 255, 255]))
        red_mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
        
        hazard_mask = cv2.bitwise_or(yellow_mask, orange_mask)
        hazard_mask = cv2.bitwise_or(hazard_mask, red_mask1)
        hazard_mask = cv2.bitwise_or(hazard_mask, red_mask2)
        
        hazard_pixels = cv2.countNonZero(hazard_mask)
        
        if hazard_pixels > 3000:
            contours, _ = cv2.findContours(hazard_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w_box, h_box = cv2.boundingRect(contours[0])
                result.detected = True
                result.confidence = 0.7
                result.bounding_box = (x, y, w_box, h_box)

        return result


class UnloadingProcedureAlgorithm(AlgorithmBase):
    """卸油流程不规范 - ID 42"""
    ALGORITHM_ID = 42
    ALGORITHM_NAME = "卸油流程不规范"
    CATEGORY = AlgorithmCategory.PERIMETER_ALERT

    def initialize(self) -> bool:
        return True

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        if not self._is_valid_frame(frame):
            return result

        h, w = frame.shape[:2]
        
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                result.detected = True
                result.confidence = 0.6
                result.bounding_box = (0, 0, w, h)
        except Exception:
            pass

        return result


class SignDetectionAlgorithm(AlgorithmBase):
    """标识牌识别 - ID 43"""
    ALGORITHM_ID = 43
    ALGORITHM_NAME = "标识牌识别"
    CATEGORY = AlgorithmCategory.PERIMETER_ALERT

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
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 5000:
                x, y, w_box, h_box = cv2.boundingRect(contour)
                aspect_ratio = w_box / h_box if h_box > 0 else 0
                
                if 0.7 < aspect_ratio < 1.5:
                    result.detected = True
                    result.confidence = 0.65
                    result.bounding_box = (x, y, w_box, h_box)
                    break

        return result
