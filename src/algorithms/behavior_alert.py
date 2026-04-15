#!/usr/bin/env python3
"""
行为警戒算法 - ID 16-24, 44-45, 50
"""

import cv2
import numpy as np
from typing import Dict, Any

from .algorithm_base import AlgorithmBase, AlgorithmResult, AlgorithmCategory


class FallDetectionAlgorithm(AlgorithmBase):
    """摔倒检测 - ID 16"""
    ALGORITHM_ID = 16
    ALGORITHM_NAME = "摔倒检测"
    CATEGORY = AlgorithmCategory.BEHAVIOR_ALERT

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.previous_positions = []

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
            if 2000 < area < 20000:
                x, y, w_box, h_box = cv2.boundingRect(contour)
                aspect_ratio = w_box / h_box if h_box > 0 else 0
                
                if aspect_ratio > 1.5:
                    result.detected = True
                    result.confidence = 0.65
                    result.bounding_box = (x, y, w_box, h_box)
                    break

        return result


class SmokingDetectionAlgorithm(AlgorithmBase):
    """抽烟检测 - ID 17"""
    ALGORITHM_ID = 17
    ALGORITHM_NAME = "抽烟检测"
    CATEGORY = AlgorithmCategory.BEHAVIOR_ALERT

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
            
            if len(faces) > 0:
                for (x, y, w_face, h_face) in faces:
                    mouth_region = frame[y + h//2:y + h, x:x + w]
                    if mouth_region.size > 0:
                        hsv = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2HSV)
                        white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
                        
                        white_pixels = cv2.countNonZero(white_mask)
                        if white_pixels > 100:
                            result.detected = True
                            result.confidence = 0.6
                            result.bounding_box = (x, y, w_face, h_face)
                            break
        except Exception:
            pass

        return result


class PhoneCallAlgorithm(AlgorithmBase):
    """打电话 - ID 18"""
    ALGORITHM_ID = 18
    ALGORITHM_NAME = "打电话"
    CATEGORY = AlgorithmCategory.BEHAVIOR_ALERT

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
            
            if len(faces) > 0:
                for (x, y, w_face, h_face) in faces:
                    hand_region = frame[y:y + h_face, max(0, x - w_face//2):x + w_face + w_face//2]
                    if hand_region.size > 0:
                        hsv = cv2.cvtColor(hand_region, cv2.COLOR_BGR2HSV)
                        skin_mask1 = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255]))
                        skin_mask2 = cv2.inRange(hsv, np.array([170, 20, 70]), np.array([180, 255, 255]))
                        skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
                        
                        skin_pixels = cv2.countNonZero(skin_mask)
                        if skin_pixels > 500:
                            result.detected = True
                            result.confidence = 0.6
                            result.bounding_box = (x, y, w_face, h_face)
                            break
        except Exception:
            pass

        return result


class PhoneUsingAlgorithm(AlgorithmBase):
    """看手机 - ID 19"""
    ALGORITHM_ID = 19
    ALGORITHM_NAME = "看手机"
    CATEGORY = AlgorithmCategory.BEHAVIOR_ALERT

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
        
        bright_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 50, 255]))
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 2000:
                x, y, w_box, h_box = cv2.boundingRect(contour)
                aspect_ratio = w_box / h_box if h_box > 0 else 0
                
                if 0.5 < aspect_ratio < 2.0:
                    result.detected = True
                    result.confidence = 0.55
                    result.bounding_box = (x, y, w_box, h_box)
                    break

        return result


class RunningAlgorithm(AlgorithmBase):
    """人员奔跑 - ID 20"""
    ALGORITHM_ID = 20
    ALGORITHM_NAME = "人员奔跑"
    CATEGORY = AlgorithmCategory.BEHAVIOR_ALERT

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.previous_frame = None
        self.previous_positions = []

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

        if self.previous_frame is None:
            self.previous_frame = gray
            return result

        if self.previous_frame.shape != gray.shape:
            self.previous_frame = gray
            return result

        frame_delta = cv2.absdiff(self.previous_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        changed_pixels = cv2.countNonZero(thresh)
        if changed_pixels > 50000:
            result.detected = True
            result.confidence = 0.65
            result.bounding_box = (0, 0, w, h)

        self.previous_frame = gray
        return result


class SleepingOnJobAlgorithm(AlgorithmBase):
    """睡岗检测 - ID 21"""
    ALGORITHM_ID = 21
    ALGORITHM_NAME = "睡岗检测"
    CATEGORY = AlgorithmCategory.BEHAVIOR_ALERT

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.motionless_frames = 0

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
        
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                self.motionless_frames += 1
                if self.motionless_frames > 100:
                    result.detected = True
                    result.confidence = 0.7
                    x, y, w_face, h_face = faces[0]
                    result.bounding_box = (x, y, w_face, h_face)
            else:
                self.motionless_frames = 0
        except Exception:
            pass

        return result


class PersonAbsentAlgorithm(AlgorithmBase):
    """人员离岗 - ID 22"""
    ALGORITHM_ID = 22
    ALGORITHM_NAME = "人员离岗"
    CATEGORY = AlgorithmCategory.BEHAVIOR_ALERT

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.absent_frames = 0

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
            
            if len(faces) == 0:
                self.absent_frames += 1
                if self.absent_frames > 50:
                    result.detected = True
                    result.confidence = 0.7
                    result.bounding_box = (0, 0, w, h)
            else:
                self.absent_frames = 0
        except Exception:
            pass

        return result


class CrowdGatheringAlgorithm(AlgorithmBase):
    """人员聚众 - ID 23"""
    ALGORITHM_ID = 23
    ALGORITHM_NAME = "人员聚众"
    CATEGORY = AlgorithmCategory.BEHAVIOR_ALERT

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.crowd_threshold = config.get('crowd_threshold', 4)

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
            
            if len(faces) >= self.crowd_threshold:
                result.detected = True
                result.confidence = 0.75
                result.bounding_box = (0, 0, w, h)
                result.extra_data = {'person_count': len(faces)}
        except Exception:
            pass

        return result


class FightingAlgorithm(AlgorithmBase):
    """人员扭打 - ID 24"""
    ALGORITHM_ID = 24
    ALGORITHM_NAME = "人员扭打"
    CATEGORY = AlgorithmCategory.BEHAVIOR_ALERT

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.previous_frame = None
        self.rapid_motion_frames = 0

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

        if self.previous_frame is None:
            self.previous_frame = gray
            return result

        if self.previous_frame.shape != gray.shape:
            self.previous_frame = gray
            return result

        frame_delta = cv2.absdiff(self.previous_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        changed_pixels = cv2.countNonZero(thresh)
        
        if changed_pixels > 30000:
            self.rapid_motion_frames += 1
            if self.rapid_motion_frames > 20:
                result.detected = True
                result.confidence = 0.6
                result.bounding_box = (0, 0, w, h)
        else:
            self.rapid_motion_frames = max(0, self.rapid_motion_frames - 1)

        self.previous_frame = gray
        return result


class PersonLoiteringBehaviorAlgorithm(AlgorithmBase):
    """人员滞留 - ID 44"""
    ALGORITHM_ID = 44
    ALGORITHM_NAME = "人员滞留"
    CATEGORY = AlgorithmCategory.BEHAVIOR_ALERT

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.tracked_positions = []
        self.loiter_threshold_frames = 150

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
            
            if len(faces) > 0:
                for (x, y, w_face, h_face) in faces:
                    center_x = x + w_face // 2
                    center_y = y + h_face // 2
                    self.tracked_positions.append((center_x, center_y))
                    
                    if len(self.tracked_positions) > 200:
                        self.tracked_positions.pop(0)
                    
                    if len(self.tracked_positions) >= self.loiter_threshold_frames:
                        positions = np.array(self.tracked_positions)
                        movement = np.std(positions, axis=0)
                        
                        if np.mean(movement) < 80:
                            result.detected = True
                            result.confidence = 0.7
                            result.bounding_box = (x, y, w_face, h_face)
                            break
            else:
                if len(self.tracked_positions) > 0:
                    self.tracked_positions.pop(0)
        except Exception:
            pass

        return result


class HelpGestureAlgorithm(AlgorithmBase):
    """举手求救 - ID 45"""
    ALGORITHM_ID = 45
    ALGORITHM_NAME = "举手求救"
    CATEGORY = AlgorithmCategory.BEHAVIOR_ALERT

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
        
        skin_mask1 = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255]))
        skin_mask2 = cv2.inRange(hsv, np.array([170, 20, 70]), np.array([180, 255, 255]))
        skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
        
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 5000:
                x, y, w_box, h_box = cv2.boundingRect(contour)
                aspect_ratio = h_box / w_box if w_box > 0 else 0
                
                if aspect_ratio > 1.5 and y < h // 3:
                    result.detected = True
                    result.confidence = 0.6
                    result.bounding_box = (x, y, w_box, h_box)
                    break

        return result


class FatigueDrivingAlgorithm(AlgorithmBase):
    """疲劳驾驶 - ID 50"""
    ALGORITHM_ID = 50
    ALGORITHM_NAME = "疲劳驾驶"
    CATEGORY = AlgorithmCategory.BEHAVIOR_ALERT

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.eye_closed_frames = 0

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
            
            if len(faces) > 0:
                for (x, y, w_face, h_face) in faces:
                    eye_region = gray[y + h_face//4:y + h_face//2, x:x + w_face]
                    if eye_region.size > 0:
                        brightness = np.mean(eye_region)
                        
                        if brightness < 80:
                            self.eye_closed_frames += 1
                            if self.eye_closed_frames > 30:
                                result.detected = True
                                result.confidence = 0.65
                                result.bounding_box = (x, y, w_face, h_face)
                                break
                        else:
                            self.eye_closed_frames = 0
        except Exception:
            pass

        return result
