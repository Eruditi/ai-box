#!/usr/bin/env python3
"""
人员违规检测算法 - ID 1-5, 37, 41
基于 YOLOv8 目标检测 + 颜色/区域分析
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional

from .algorithm_base import AlgorithmBase, AlgorithmResult, AlgorithmCategory, safe_parse_detection
from .yolo_engine import get_yolo_engine, HELMET_COLORS


class NoHelmetAlgorithm(AlgorithmBase):
    """未佩戴安全帽报警 - ID 1"""
    ALGORITHM_ID = 1
    ALGORITHM_NAME = "未佩戴安全帽报警"
    CATEGORY = AlgorithmCategory.PERSON_VIOLATION

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

        try:
            detections = self.yolo.detect(frame, classes=[0])
            
            if not detections:
                return result
            
            for det in detections:
                try:
                    parsed = safe_parse_detection(det)
                    if not parsed:
                        continue

                    x1, y1, x2, y2 = parsed['bbox']
                    confidence = parsed['confidence']

                    head_h = max(1, (y2 - y1) // 3)
                    head_region = frame[max(0, y1 - head_h):min(frame.shape[0], y1 + head_h), max(0, x1):min(frame.shape[1], x2)]
                    
                    if head_region.size == 0 or head_region.shape[0] == 0 or head_region.shape[1] == 0:
                        continue

                    hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
                    helmet_pixel_count = 0
                    for color_name, color_range in HELMET_COLORS.items():
                        mask = cv2.inRange(hsv, np.array(color_range['lower']), np.array(color_range['upper']))
                        helmet_pixel_count += cv2.countNonZero(mask)

                    region_area = head_region.shape[0] * head_region.shape[1]
                    helmet_ratio = helmet_pixel_count / region_area if region_area > 0 else 0

                    if helmet_ratio < 0.08:
                        result.detected = True
                        result.confidence = min(0.95, confidence + 0.2)
                        result.bounding_box = (x1, y1, x2 - x1, y2 - y1)
                        result.extra_data = {'person_confidence': confidence, 'helmet_ratio': round(helmet_ratio, 3)}
                        break
                        
                except Exception as det_err:
                    logging.debug(f"[安全帽检测] 跳过无效检测: {det_err}")
                    continue
                    
        except Exception as e:
            logging.error(f"[安全帽检测] 处理失败: {e}")
        
        return result


class NoMaskAlgorithm(AlgorithmBase):
    """未戴口罩报警 - ID 2"""
    ALGORITHM_ID = 2
    ALGORITHM_NAME = "未戴口罩报警"
    CATEGORY = AlgorithmCategory.PERSON_VIOLATION

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

        try:
            detections = self.yolo.detect(frame, classes=[0])
            
            for det in detections:
                parsed = safe_parse_detection(det)
                if not parsed:
                    continue
                    
                x1, y1, x2, y2 = parsed['bbox']
                confidence = parsed['confidence']
                face_h = y2 - y1
                mouth_region = frame[y1 + face_h * 2 // 3:min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                if mouth_region.size == 0 or mouth_region.shape[0] == 0:
                    continue

                hsv = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2HSV)
                skin_mask1 = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255]))
                skin_mask2 = cv2.inRange(hsv, np.array([170, 20, 70]), np.array([180, 255, 255]))
                skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
                skin_pixels = cv2.countNonZero(skin_mask)
                region_area = mouth_region.shape[0] * mouth_region.shape[1]

                if region_area > 0 and skin_pixels / region_area > 0.25:
                    result.detected = True
                    result.confidence = min(0.9, confidence + 0.15)
                    result.bounding_box = (x1, y1, x2 - x1, y2 - y1)
                    break
        except Exception as e:
            logging.error(f"[口罩检测] 处理失败: {e}")

        return result


class NoWorkwearAlgorithm(AlgorithmBase):
    """未穿戴工作服报警 - ID 3"""
    ALGORITHM_ID = 3
    ALGORITHM_NAME = "未穿戴工作服报警"
    CATEGORY = AlgorithmCategory.PERSON_VIOLATION

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

        try:
            detections = self.yolo.detect(frame, classes=[0])
            
            for det in detections:
                parsed = safe_parse_detection(det)
                if not parsed:
                    continue
                    
                x1, y1, x2, y2 = parsed['bbox']
                confidence = parsed['confidence']
                body_h = y2 - y1
                body_region = frame[y1 + body_h // 3:min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                if body_region.size == 0 or body_region.shape[0] == 0:
                    continue

                hsv = cv2.cvtColor(body_region, cv2.COLOR_BGR2HSV)
                blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
                orange_mask = cv2.inRange(hsv, np.array([5, 100, 100]), np.array([20, 255, 255]))
                workwear_mask = cv2.bitwise_or(blue_mask, orange_mask)
                workwear_pixels = cv2.countNonZero(workwear_mask)
                region_area = body_region.shape[0] * body_region.shape[1]

                if region_area > 0 and workwear_pixels / region_area < 0.05:
                    result.detected = True
                    result.confidence = min(0.85, confidence + 0.1)
                    result.bounding_box = (x1, y1, x2 - x1, y2 - y1)
                    break
        except Exception as e:
            logging.error(f"[工作服检测] 处理失败: {e}")

        return result


class NoSafetyBeltAlgorithm(AlgorithmBase):
    """未佩戴安全带报警 - ID 4"""
    ALGORITHM_ID = 4
    ALGORITHM_NAME = "未佩戴安全带报警"
    CATEGORY = AlgorithmCategory.PERSON_VIOLATION

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

        try:
            detections = self.yolo.detect(frame, classes=[0])
            
            for det in detections:
                parsed = safe_parse_detection(det)
                if not parsed:
                    continue
                    
                x1, y1, x2, y2 = parsed['bbox']
                confidence = parsed['confidence']
                body_region = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                if body_region.size == 0 or body_region.shape[0] == 0:
                    continue

                gray = cv2.cvtColor(body_region, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=30, maxLineGap=10)

                has_diagonal_line = False
                if lines is not None:
                    for line in lines:
                        x_l, y_l, x_r, y_r = line[0]
                        angle = abs(np.arctan2(y_r - y_l, x_r - x_l) * 180 / np.pi)
                        if 20 < angle < 70:
                            has_diagonal_line = True
                            break

                if not has_diagonal_line:
                    result.detected = True
                    result.confidence = min(0.8, confidence + 0.1)
                    result.bounding_box = (x1, y1, x2 - x1, y2 - y1)
                    break
        except Exception as e:
            logging.error(f"[安全带检测] 处理失败: {e}")

        return result


class NoReflectiveVestAlgorithm(AlgorithmBase):
    """未佩戴反光衣报警 - ID 5"""
    ALGORITHM_ID = 5
    ALGORITHM_NAME = "未佩戴反光衣报警"
    CATEGORY = AlgorithmCategory.PERSON_VIOLATION

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

        try:
            detections = self.yolo.detect(frame, classes=[0])
            
            for det in detections:
                parsed = safe_parse_detection(det)
                if not parsed:
                    continue
                    
                x1, y1, x2, y2 = parsed['bbox']
                confidence = parsed['confidence']
                body_h = y2 - y1
                body_region = frame[y1 + body_h // 4:min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                if body_region.size == 0 or body_region.shape[0] == 0:
                    continue

                hsv = cv2.cvtColor(body_region, cv2.COLOR_BGR2HSV)
                yellow_mask = cv2.inRange(hsv, np.array([20, 100, 150]), np.array([40, 255, 255]))
                orange_mask = cv2.inRange(hsv, np.array([5, 100, 150]), np.array([15, 255, 255]))
                reflective_mask = cv2.bitwise_or(yellow_mask, orange_mask)
                reflective_pixels = cv2.countNonZero(reflective_mask)
                region_area = body_region.shape[0] * body_region.shape[1]

                if region_area > 0 and reflective_pixels / region_area < 0.02:
                    result.detected = True
                    result.confidence = min(0.8, confidence + 0.1)
                    result.bounding_box = (x1, y1, x2 - x1, y2 - y1)
                    break
        except Exception as e:
            logging.error(f"[反光衣检测] 处理失败: {e}")


class NoHelmetRidingAlgorithm(AlgorithmBase):
    """骑车未带安全帽 - ID 37"""
    ALGORITHM_ID = 37
    ALGORITHM_NAME = "骑车未带安全帽"
    CATEGORY = AlgorithmCategory.PERSON_VIOLATION

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

        try:
            detections = self.yolo.detect(frame, classes=[0, 1, 2, 3])
            
            persons = []
            vehicles = []
            
            for det in detections:
                parsed = safe_parse_detection(det)
                if not parsed:
                    continue

                bbox = parsed['bbox']
                confidence = parsed['confidence']
                class_id = parsed['class_id']

                if class_id == 0:
                    persons.append({'bbox': bbox, 'confidence': confidence})
                elif class_id in (1, 2, 3):
                    vehicles.append({'bbox': bbox})

            for person in persons:
                px1, py1, px2, py2 = person['bbox']
                on_vehicle = False
                for v in vehicles:
                    vx1, vy1, vx2, vy2 = v['bbox']
                    if (px1 < vx2 and px2 > vx1 and py2 > vy1 and
                            abs(py2 - vy1) < (vy2 - vy1) // 2):
                        on_vehicle = True
                        break

                if on_vehicle:
                    head_h = max(1, (py2 - py1) // 3)
                    head_region = frame[max(0, py1 - head_h):min(frame.shape[0], py1 + head_h), max(0, px1):min(frame.shape[1], px2)]
                    if head_region.size == 0 or head_region.shape[0] == 0:
                        continue

                    hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
                    helmet_pixels = 0
                    for color_range in HELMET_COLORS.values():
                        mask = cv2.inRange(hsv, np.array(color_range['lower']),
                                           np.array(color_range['upper']))
                        helmet_pixels += cv2.countNonZero(mask)

                    region_area = head_region.shape[0] * head_region.shape[1]
                    if region_area > 0 and helmet_pixels / region_area < 0.08:
                        result.detected = True
                        result.confidence = min(0.9, person['confidence'] + 0.15)
                        result.bounding_box = (px1, py1, px2 - px1, py2 - py1)
                        break
        except Exception as e:
            logging.error(f"[骑车安全帽检测] 处理失败: {e}")


class MotorcycleInGasStationAlgorithm(AlgorithmBase):
    """骑摩托车进加油站 - ID 41"""
    ALGORITHM_ID = 41
    ALGORITHM_NAME = "骑摩托车进加油站"
    CATEGORY = AlgorithmCategory.PERSON_VIOLATION

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

        try:
            detections = self.yolo.detect(frame, classes=[0, 3])

            motorcycles = []
            persons = []
            for det in detections:
                parsed = safe_parse_detection(det)
                if not parsed:
                    continue
                if parsed['class_id'] == 3:
                    motorcycles.append(parsed)
                elif parsed['class_id'] == 0:
                    persons.append(parsed)

            for moto in motorcycles:
                mx1, my1, mx2, my2 = moto['bbox']
                for person in persons:
                    px1, py1, px2, py2 = person['bbox']
                    if (px1 < mx2 and px2 > mx1 and
                            abs(py2 - my1) < (my2 - my1) // 2):
                        result.detected = True
                        result.confidence = min(0.85, moto['confidence'] + person['confidence'])
                        result.bounding_box = (min(mx1, px1), min(my1, py1),
                                               max(mx2, px2) - min(mx1, px1),
                                               max(my2, py2) - min(my1, py1))
                        break
                if result.detected:
                    break
        except Exception as e:
            logging.error(f"[加油站摩托车检测] 处理失败: {e}")

        return result
