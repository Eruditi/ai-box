#!/usr/bin/env python3
"""
行为警戒算法 - 升级版
使用 YOLOv8 姿态估计替代传统轮廓检测
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging

from .algorithm_base import AlgorithmBase, AlgorithmResult, AlgorithmCategory


class PoseEstimator:
    """姿态估计器 - 使用YOLOv8关键点检测"""
    
    def __init__(self, model_path: str = 'yolov8n-pose.pt'):
        self.model = None
        self.model_path = model_path
        self.initialized = False
        
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    def initialize(self) -> bool:
        """初始化模型"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self.initialized = True
            logging.info(f"[姿态估计] YOLOv8-pose 模型加载成功: {self.model_path}")
            return True
        except Exception as e:
            logging.warning(f"[姿态估计] YOLOv8-pose 加载失败，回退到传统方法: {e}")
            self.initialized = False
            return False
    
    def detect_poses(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """检测人体姿态"""
        if not self.initialized or self.model is None:
            return []
        
        try:
            results = self.model(frame, verbose=False)
            
            poses = []
            for result in results:
                if result.keypoints is not None:
                    boxes = result.boxes
                    keypoints = result.keypoints
                    
                    for i in range(len(keypoints)):
                        box = boxes[i] if i < len(boxes) else None
                        
                        pose_data = {
                            'keypoints': keypoints[i].data.cpu().numpy()[0] if len(keypoints[i].data) > 0 else None,
                            'confidence': float(boxes[i].conf[0]) if box is not None else 0.0,
                            'bbox': tuple(map(int, boxes[i].xyxy[0].tolist())) if box is not None else None
                        }
                        
                        poses.append(pose_data)
            
            return poses
        except Exception as e:
            logging.error(f"[姿态估计] 检测失败: {e}")
            return []
    
    def get_keypoint(self, keypoints: np.ndarray, name: str) -> Optional[Tuple[float, float, float]]:
        """获取指定关键点"""
        if keypoints is None or len(keypoints) < 17:
            return None
        
        idx = self.keypoint_names.index(name) if name in self.keypoint_names else -1
        if idx < 0 or idx >= len(keypoints):
            return None
        
        x, y, conf = keypoints[idx]
        if conf > 0.5:
            return (float(x), float(y), float(conf))
        return None


class FallDetectionAlgorithmV2(AlgorithmBase):
    """摔倒检测 V2 - 使用姿态估计"""
    ALGORITHM_ID = 16
    ALGORITHM_NAME = "摔倒检测"
    CATEGORY = AlgorithmCategory.BEHAVIOR_ALERT
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.pose_estimator = PoseEstimator()
        self.fall_angle_threshold = config.get('fall_angle_threshold', 60) if config else 60
        self.fall_duration_threshold = config.get('fall_duration_threshold', 2.0) if config else 2.0
        self.fall_history: Dict[str, List[float]] = {}
    
    def initialize(self) -> bool:
        return self.pose_estimator.initialize()
    
    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )
        
        poses = self.pose_estimator.detect_poses(frame)
        
        for pose in poses:
            keypoints = pose['keypoints']
            confidence = pose['confidence']
            bbox = pose['bbox']
            
            if keypoints is None:
                continue
            
            left_shoulder = self.pose_estimator.get_keypoint(keypoints, 'left_shoulder')
            right_shoulder = self.pose_estimator.get_keypoint(keypoints, 'right_shoulder')
            left_hip = self.pose_estimator.get_keypoint(keypoints, 'left_hip')
            right_hip = self.pose_estimator.get_keypoint(keypoints, 'right_hip')
            
            if left_shoulder and right_shoulder and left_hip and right_hip:
                shoulder_center = np.array([
                    (left_shoulder[0] + right_shoulder[0]) / 2,
                    (left_shoulder[1] + right_shoulder[1]) / 2
                ])
                
                hip_center = np.array([
                    (left_hip[0] + right_hip[0]) / 2,
                    (left_hip[1] + right_hip[1]) / 2
                ])
                
                torso_vector = shoulder_center - hip_center
                angle = np.abs(np.arctan2(torso_vector[0], torso_vector[1]) * 180 / np.pi)
                
                if angle > self.fall_angle_threshold:
                    result.detected = True
                    result.confidence = min(0.95, confidence + 0.2)
                    result.bounding_box = bbox
                    result.extra_data = {
                        'fall_angle': float(angle),
                        'method': 'yolov8_pose'
                    }
                    break
        
        if not result.detected:
            result = self._fallback_detection(frame, result)
        
        return result
    
    def _fallback_detection(self, frame: np.ndarray, result: AlgorithmResult) -> AlgorithmResult:
        """回退到传统检测方法"""
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
                    result.extra_data = {'method': 'contour_fallback'}
                    break
        
        return result


class SmokingDetectionAlgorithmV2(AlgorithmBase):
    """抽烟检测 V2 - 使用YOLO目标检测"""
    ALGORITHM_ID = 17
    ALGORITHM_NAME = "抽烟检测"
    CATEGORY = AlgorithmCategory.BEHAVIOR_ALERT
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.yolo = None
        self.initialized = False
    
    def initialize(self) -> bool:
        try:
            from ultralytics import YOLO
            self.yolo = YOLO('yolov8n.pt')
            self.initialized = True
            return True
        except Exception as e:
            logging.warning(f"[抽烟检测] YOLO加载失败: {e}")
            return False
    
    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )
        
        if self.yolo:
            detections = self.yolo(frame, classes=[0], verbose=False)
            
            for det in detections:
                if len(det.boxes) > 0:
                    for box in det.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        
                        person_region = frame[y1:y2, x1:x2]
                        if person_region.size > 0:
                            hsv = cv2.cvtColor(person_region, cv2.COLOR_BGR2HSV)
                            white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
                            
                            white_pixels = cv2.countNonZero(white_mask)
                            if white_pixels > 100:
                                result.detected = True
                                result.confidence = min(0.85, conf + 0.1)
                                result.bounding_box = (x1, y1, x2-x1, y2-y1)
                                result.extra_data = {'method': 'yolov8'}
                                return result
        
        result = self._fallback_detection(frame, result)
        return result
    
    def _fallback_detection(self, frame: np.ndarray, result: AlgorithmResult) -> AlgorithmResult:
        """回退检测"""
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                for (x, y, w_face, h_face) in faces:
                    mouth_region = frame[y + h_face//2:y + h_face, x:x + w_face]
                    if mouth_region.size > 0:
                        hsv = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2HSV)
                        white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
                        
                        white_pixels = cv2.countNonZero(white_mask)
                        if white_pixels > 100:
                            result.detected = True
                            result.confidence = 0.6
                            result.bounding_box = (x, y, w_face, h_face)
                            result.extra_data = {'method': 'haar_fallback'}
                            break
        except Exception:
            pass
        
        return result


class FightingDetectionAlgorithm(AlgorithmBase):
    """打架检测 - ID 20"""
    ALGORITHM_ID = 20
    ALGORITHM_NAME = "打架检测"
    CATEGORY = AlgorithmCategory.BEHAVIOR_ALERT
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.pose_estimator = PoseEstimator()
        self.prev_keypoints = []
        self.motion_threshold = config.get('motion_threshold', 30) if config else 30
    
    def initialize(self) -> bool:
        return self.pose_estimator.initialize()
    
    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )
        
        poses = self.pose_estimator.detect_poses(frame)
        
        if len(poses) >= 2:
            motions = []
            
            for pose in poses:
                keypoints = pose['keypoints']
                if keypoints is not None and len(self.prev_keypoints) > 0:
                    for prev_kp in self.prev_keypoints:
                        if prev_kp is not None and keypoints.shape == prev_kp.shape:
                            diff = np.abs(keypoints[:, :2] - prev_kp[:, :2])
                            motion = np.mean(diff)
                            motions.append(motion)
            
            if motions and np.mean(motions) > self.motion_threshold:
                result.detected = True
                result.confidence = min(0.85, np.mean(motions) / 100)
                result.extra_data = {
                    'avg_motion': float(np.mean(motions)),
                    'person_count': len(poses)
                }
        
        self.prev_keypoints = [p['keypoints'] for p in poses if p['keypoints'] is not None]
        
        return result


class ClimbingDetectionAlgorithm(AlgorithmBase):
    """攀爬检测 - ID 21"""
    ALGORITHM_ID = 21
    ALGORITHM_NAME = "攀爬检测"
    CATEGORY = AlgorithmCategory.BEHAVIOR_ALERT
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.pose_estimator = PoseEstimator()
    
    def initialize(self) -> bool:
        return self.pose_estimator.initialize()
    
    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> AlgorithmResult:
        result = AlgorithmResult(
            algorithm_id=self.ALGORITHM_ID,
            algorithm_name=self.ALGORITHM_NAME,
            category=self.CATEGORY
        )
        
        poses = self.pose_estimator.detect_poses(frame)
        
        for pose in poses:
            keypoints = pose['keypoints']
            confidence = pose['confidence']
            bbox = pose['bbox']
            
            if keypoints is None:
                continue
            
            left_ankle = self.pose_estimator.get_keypoint(keypoints, 'left_ankle')
            right_ankle = self.pose_estimator.get_keypoint(keypoints, 'right_ankle')
            left_hip = self.pose_estimator.get_keypoint(keypoints, 'left_hip')
            right_hip = self.pose_estimator.get_keypoint(keypoints, 'right_hip')
            
            if left_ankle and right_ankle and left_hip and right_hip:
                ankle_y = (left_ankle[1] + right_ankle[1]) / 2
                hip_y = (left_hip[1] + right_hip[1]) / 2
                
                leg_height = ankle_y - hip_y
                
                left_knee = self.pose_estimator.get_keypoint(keypoints, 'left_knee')
                right_knee = self.pose_estimator.get_keypoint(keypoints, 'right_knee')
                
                if left_knee and right_knee:
                    knee_y = (left_knee[1] + right_knee[1]) / 2
                    
                    if abs(knee_y - hip_y) < leg_height * 0.3:
                        result.detected = True
                        result.confidence = min(0.85, confidence + 0.15)
                        result.bounding_box = bbox
                        result.extra_data = {'method': 'yolov8_pose'}
                        break
        
        return result
