#!/usr/bin/env python3
"""
环境异常检测算法 - ID 6-9, 47-49
基于 YOLOv8 目标检测 + HSV 颜色分析 + 时域验证
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from collections import deque
import time

from .algorithm_base import AlgorithmBase, AlgorithmResult, AlgorithmCategory
from .yolo_engine import get_yolo_engine, FIRE_COLORS, SMOKE_COLOR


class TemporalValidator:
    """时域验证器 - 连续帧确认机制"""
    
    def __init__(self, min_frames: int = 3, max_age: float = 1.0, iou_threshold: float = 0.3):
        self.min_frames = min_frames
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.detection_history: deque = deque(maxlen=20)
        self.consecutive_count = 0
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """计算两个边界框的IoU"""
        x1_1, y1_1, w1, h1 = box1
        x1_2, y1_2, w2, h2 = box2
        
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x1_1 + w1, x1_2 + w2)
        y_bottom = min(y1_1 + h1, y1_2 + h2)
        
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update(self, detected: bool, bbox: Tuple[int, int, int, int] = None) -> Tuple[bool, float]:
        """
        更新时域状态
        返回: (是否确认检测, 置信度)
        """
        current_time = time.time()
        
        # 清除过期记录
        while self.detection_history and current_time - self.detection_history[0][2] > self.max_age:
            self.detection_history.popleft()
        
        if not detected:
            # 检测中断，重置计数器
            if len(self.detection_history) > 0 and current_time - self.detection_history[-1][2] < 0.5:
                pass  # 允许短暂丢失
            else:
                self.consecutive_count = 0
                self.detection_history.clear()
            return False, 0.0
        
        # 检查与最近检测的空间一致性
        is_consistent = True
        if bbox and len(self.detection_history) > 0:
            last_bbox = self.detection_history[-1][0]
            if last_bbox:
                iou = self._calculate_iou(bbox, last_bbox)
                if iou < self.iou_threshold:
                    is_consistent = False
                    self.consecutive_count = 0
        
        # 添加新检测记录
        avg_conf = 0.6
        if self.detection_history:
            avg_conf = (self.detection_history[-1][1] + 0.7) / 2
        
        self.detection_history.append((bbox, avg_conf, current_time))
        
        if is_consistent:
            self.consecutive_count += 1
        else:
            self.consecutive_count = 1  # 新位置开始重新计数
        
        # 检查是否达到确认阈值
        if self.consecutive_count >= self.min_frames:
            final_conf = min(0.95, avg_conf + 0.15 * (self.consecutive_count / self.min_frames))
            return True, final_conf
        
        return False, 0.0


class FireDetectionAlgorithm(AlgorithmBase):
    """火焰报警 - ID 6 (带时域验证)"""
    ALGORITHM_ID = 6
    ALGORITHM_NAME = "火焰报警"
    CATEGORY = AlgorithmCategory.ENVIRONMENT_ABNORMAL

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.yolo = None
        self.temporal_validator = TemporalValidator(min_frames=3, max_age=1.0)

    def initialize(self) -> bool:
        self.yolo = get_yolo_engine()
        return True

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        fire_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for color_range in FIRE_COLORS.values():
            mask = cv2.inRange(hsv, np.array(color_range['lower']), np.array(color_range['upper']))
            fire_mask = cv2.bitwise_or(fire_mask, mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_detected = False
        current_bbox = None
        max_density = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                x, y, w, h = cv2.boundingRect(contour)
                fire_pixels = cv2.countNonZero(fire_mask[y:y + h, x:x + w])
                region_area = w * h
                density = fire_pixels / region_area if region_area > 0 else 0
                if density > 0.3 and density > max_density:
                    max_density = density
                    current_detected = True
                    current_bbox = (x, y, w, h)
        
        # 时域验证
        confirmed, confidence = self.temporal_validator.update(current_detected, current_bbox)
        
        if confirmed:
            result.detected = True
            result.confidence = confidence
            result.bounding_box = current_bbox
            result.extra_data = {
                'fire_area': int(current_bbox[2] * current_bbox[3]) if current_bbox else 0,
                'density': float(max_density),
                'temporal_frames': self.temporal_validator.min_frames,
                'method': 'hsv_temporal'
            }

        return result


class SmokeDetectionAlgorithm(AlgorithmBase):
    """烟雾报警 - ID 7 (带时域验证)"""
    ALGORITHM_ID = 7
    ALGORITHM_NAME = "烟雾报警"
    CATEGORY = AlgorithmCategory.ENVIRONMENT_ABNORMAL

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.temporal_validator = TemporalValidator(min_frames=4, max_age=1.5)

    def initialize(self) -> bool:
        return True

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        smoke_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for color_range in SMOKE_COLOR.values():
            mask = cv2.inRange(hsv, np.array(color_range['lower']), np.array(color_range['upper']))
            smoke_mask = cv2.bitwise_or(smoke_mask, mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel)
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_detected = False
        current_bbox = None
        max_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 2000 and area > max_area:
                x, y, w, h = cv2.boundingRect(contour)
                max_area = area
                current_detected = True
                current_bbox = (x, y, w, h)
        
        # 时域验证（烟雾需要更多帧确认）
        confirmed, confidence = self.temporal_validator.update(current_detected, current_bbox)
        
        if confirmed:
            result.detected = True
            result.confidence = confidence
            result.bounding_box = current_bbox
            result.extra_data = {
                'smoke_area': int(max_area),
                'temporal_frames': self.temporal_validator.min_frames,
                'method': 'hsv_temporal'
            }

        return result


class FireEquipmentAlgorithm(AlgorithmBase):
    """消防设施检测 - ID 8"""
    ALGORITHM_ID = 8
    ALGORITHM_NAME = "消防设施检测"
    CATEGORY = AlgorithmCategory.ENVIRONMENT_ABNORMAL

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

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / w if w > 0 else 0
            if area > 300 and 1.5 < aspect_ratio < 5.0:
                result.detected = True
                result.confidence = 0.7
                result.bounding_box = (x, y, w, h)
                result.extra_data = {'area': int(area), 'aspect_ratio': float(aspect_ratio)}
                break

        return result


class DebrisDetectionAlgorithm(AlgorithmBase):
    """杂物堆放 - ID 9"""
    ALGORITHM_ID = 9
    ALGORITHM_NAME = "杂物堆放"
    CATEGORY = AlgorithmCategory.ENVIRONMENT_ABNORMAL

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.yolo = None
        self.prev_frame = None

    def initialize(self) -> bool:
        self.yolo = get_yolo_engine()
        return True

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dilated = cv2.dilate(edges, kernel)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_contours = [c for c in contours if cv2.contourArea(c) > 5000]

        if len(large_contours) >= 2:
            all_points = np.vstack(large_contours)
            x, y, w, h = cv2.boundingRect(all_points)
            result.detected = True
            result.confidence = min(0.8, len(large_contours) * 0.15 + 0.3)
            result.bounding_box = (x, y, w, h)

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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray)
        std_val = np.std(gray)

        if mean_val < 30 and std_val < 15:
            result.detected = True
            result.confidence = min(0.95, (30 - mean_val) / 30 + 0.5)
            h, w = frame.shape[:2]
            result.bounding_box = (0, 0, w, h)
            result.extra_data = {'mean_brightness': float(mean_val), 'std_deviation': float(std_val)}

        return result


class CameraShiftAlgorithm(AlgorithmBase):
    """摄像头偏移 - ID 48"""
    ALGORITHM_ID = 48
    ALGORITHM_NAME = "摄像头偏移"
    CATEGORY = AlgorithmCategory.ENVIRONMENT_ABNORMAL

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.prev_frame = None

    def initialize(self) -> bool:
        return True

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_frame is not None:
            prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            mean_magnitude = np.mean(magnitude)

            if mean_magnitude > 5.0:
                result.detected = True
                result.confidence = min(0.9, mean_magnitude / 20 + 0.3)
                h, w = frame.shape[:2]
                result.bounding_box = (0, 0, w, h)
                result.extra_data = {'mean_flow': float(mean_magnitude)}

        self.prev_frame = frame.copy()
        return result


class LeakDetectionAlgorithm(AlgorithmBase):
    """跑冒滴漏 - ID 49"""
    ALGORITHM_ID = 49
    ALGORITHM_NAME = "跑冒滴漏"
    CATEGORY = AlgorithmCategory.ENVIRONMENT_ABNORMAL

    def initialize(self) -> bool:
        return True

    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dark_liquid = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
        bright_liquid = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 50, 255]))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        for mask in [dark_liquid, bright_liquid]:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:
                    x, y, w, h = cv2.boundingRect(contour)
                    result.detected = True
                    result.confidence = 0.7
                    result.bounding_box = (x, y, w, h)
                    return result

        return result
