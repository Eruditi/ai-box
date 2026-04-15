#!/usr/bin/env python3
"""
无人机挂载吊舱烟火识别算法
针对高空、森林、山区场景优化
"""

import cv2
import numpy as np
from typing import Dict, Any

from .algorithm_base import AlgorithmBase, AlgorithmResult, AlgorithmCategory


class DroneFireDetectionAlgorithm(AlgorithmBase):
    """无人机火焰检测 - ID 68"""
    ALGORITHM_ID = 68
    ALGORITHM_NAME = "无人机火焰检测"
    CATEGORY = AlgorithmCategory.ENVIRONMENT_ABNORMAL
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.min_fire_size = config.get('min_fire_size', 30)  # 最小火焰像素数
        self.min_smoke_size = config.get('min_smoke_size', 100)  # 最小烟雾像素数
        self.detection_threshold = config.get('detection_threshold', 0.6)  # 检测阈值
    
    def initialize(self) -> bool:
        return True
    
    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )
        
        h, w = frame.shape[:2]
        
        # 1. 火焰检测（针对高空场景优化）
        fire_detected, fire_confidence, fire_box = self._detect_fire(frame)
        
        # 2. 烟雾检测（针对高空场景优化）
        smoke_detected, smoke_confidence, smoke_box = self._detect_smoke(frame)
        
        if fire_detected or smoke_detected:
            result.detected = True
            if fire_confidence > smoke_confidence:
                result.confidence = fire_confidence
                result.bounding_box = fire_box
                result.extra_data = {"type": "fire"}
            else:
                result.confidence = smoke_confidence
                result.bounding_box = smoke_box
                result.extra_data = {"type": "smoke"}
        
        return result
    
    def _detect_fire(self, frame: np.ndarray) -> tuple:
        """检测火焰"""
        h, w = frame.shape[:2]
        
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 火焰颜色范围（针对高空场景优化）
        lower_fire1 = np.array([0, 120, 120])
        upper_fire1 = np.array([30, 255, 255])
        lower_fire2 = np.array([170, 120, 120])
        upper_fire2 = np.array([180, 255, 255])
        
        fire_mask1 = cv2.inRange(hsv, lower_fire1, upper_fire1)
        fire_mask2 = cv2.inRange(hsv, lower_fire2, upper_fire2)
        fire_mask = cv2.bitwise_or(fire_mask1, fire_mask2)
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)
        
        # 计算火焰像素数
        fire_pixels = cv2.countNonZero(fire_mask)
        
        if fire_pixels > self.min_fire_size:
            # 查找轮廓
            contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # 找到最大的轮廓
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w_box, h_box = cv2.boundingRect(largest_contour)
                
                # 计算置信度
                confidence = min(0.95, (fire_pixels / 1000) * 0.5 + (cv2.contourArea(largest_contour) / (w * h)) * 0.5)
                
                return True, confidence, (x, y, w_box, h_box)
        
        return False, 0.0, None
    
    def _detect_smoke(self, frame: np.ndarray) -> tuple:
        """检测烟雾"""
        h, w = frame.shape[:2]
        
        # 转换到灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        
        # 计算亮度和标准差
        brightness = np.mean(gray)
        std_dev = np.std(gray)
        
        # 转换到HSV计算饱和度
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv[:, :, 1])
        
        # 烟雾特征：低饱和度、中等亮度、低对比度
        if saturation < 60 and brightness > 80 and brightness < 200 and std_dev < 40:
            # 边缘检测
            edges = cv2.Canny(blurred, 30, 90)
            edge_density = np.sum(edges) / (h * w * 255)
            
            if edge_density < 0.03:
                # 计算烟雾区域
                # 使用自适应阈值
                _, smoke_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # 形态学操作
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
                smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel)
                smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, kernel)
                
                smoke_pixels = cv2.countNonZero(smoke_mask)
                
                if smoke_pixels > self.min_smoke_size:
                    # 查找轮廓
                    contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        x, y, w_box, h_box = cv2.boundingRect(largest_contour)
                        
                        # 计算置信度
                        confidence = min(0.9, (1 - edge_density) * 0.6 + (smoke_pixels / (w * h)) * 0.4)
                        
                        return True, confidence, (x, y, w_box, h_box)
        
        return False, 0.0, None


class DroneFireSmokeAnalyzer(AlgorithmBase):
    """无人机烟火综合分析 - ID 69"""
    ALGORITHM_ID = 69
    ALGORITHM_NAME = "无人机烟火综合分析"
    CATEGORY = AlgorithmCategory.ENVIRONMENT_ABNORMAL
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.fire_detector = DroneFireDetectionAlgorithm(config)
        self.frame_buffer = []
        self.max_buffer_size = 5
        self.detection_history = []
    
    def initialize(self) -> bool:
        return self.fire_detector.initialize()
    
    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )
        
        # 添加当前帧到缓冲区
        self.frame_buffer.append(frame)
        if len(self.frame_buffer) > self.max_buffer_size:
            self.frame_buffer.pop(0)
        
        # 处理当前帧
        current_result = self.fire_detector.process(frame, context)
        
        # 添加到历史
        self.detection_history.append(current_result.detected)
        if len(self.detection_history) > 3:
            self.detection_history.pop(0)
        
        # 综合判断：连续3帧中有2帧检测到
        if sum(self.detection_history) >= 2:
            result.detected = True
            result.confidence = current_result.confidence
            result.bounding_box = current_result.bounding_box
            result.extra_data = current_result.extra_data
        
        return result
