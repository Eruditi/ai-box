#!/usr/bin/env python3
"""
跨摄像头关联告警系统
实现跨摄像头的目标跟踪和关联告警
"""

import os
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum

import numpy as np


class CorrelationType(Enum):
    """关联类型"""
    LOITERING = "loitering"
    ENTRY_EXIT = "entry_exit"
    CROSS_CAMERA = "cross_camera"
    ABNORMAL_PATTERN = "abnormal_pattern"
    CROWD_GATHERING = "crowd_gathering"


@dataclass
class TrackedObject:
    """跟踪对象"""
    object_id: str
    object_type: str
    first_seen: float
    last_seen: float
    camera_sources: Set[str]
    positions: List[Tuple[str, Tuple[int, int, int, int], float]]
    confidence: float
    features: Optional[np.ndarray] = None
    
    def get_duration(self) -> float:
        return self.last_seen - self.first_seen
    
    def get_camera_count(self) -> int:
        return len(self.camera_sources)


@dataclass
class CorrelationAlert:
    """关联告警"""
    alert_id: str
    correlation_type: CorrelationType
    severity: str
    cameras: List[str]
    object_ids: List[str]
    start_time: float
    end_time: float
    description: str
    extra_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CameraZone:
    """摄像头区域"""
    camera_source: str
    zone_id: str
    zone_name: str
    polygon: List[Tuple[int, int]]
    zone_type: str
    entry_zones: List[str] = field(default_factory=list)
    exit_zones: List[str] = field(default_factory=list)


class ObjectTracker:
    """对象跟踪器"""
    
    def __init__(self, max_history: int = 300, similarity_threshold: float = 0.7):
        self.max_history = max_history
        self.similarity_threshold = similarity_threshold
        
        self.tracked_objects: Dict[str, TrackedObject] = {}
        self.object_counter = 0
        self._lock = threading.Lock()
        
        self.feature_extractor = FeatureExtractor()
    
    def update(self,
              camera_source: str,
              detections: List[Dict[str, Any]],
              frame: np.ndarray = None) -> List[TrackedObject]:
        """更新跟踪"""
        with self._lock:
            now = time.time()
            updated_objects = []
            
            for det in detections:
                object_type = det.get('type', 'person')
                bbox = det.get('bbox')
                confidence = det.get('confidence', 0.0)
                
                if bbox is None:
                    continue
                
                features = None
                if frame is not None:
                    features = self.feature_extractor.extract(frame, bbox)
                
                matched_id = self._find_matching_object(
                    camera_source, bbox, object_type, features
                )
                
                if matched_id:
                    obj = self.tracked_objects[matched_id]
                    obj.last_seen = now
                    obj.camera_sources.add(camera_source)
                    obj.positions.append((camera_source, bbox, now))
                    if len(obj.positions) > self.max_history:
                        obj.positions = obj.positions[-self.max_history:]
                    if features is not None:
                        obj.features = features
                    updated_objects.append(obj)
                else:
                    self.object_counter += 1
                    object_id = f"obj_{int(now*1000)}_{self.object_counter}"
                    
                    new_obj = TrackedObject(
                        object_id=object_id,
                        object_type=object_type,
                        first_seen=now,
                        last_seen=now,
                        camera_sources={camera_source},
                        positions=[(camera_source, bbox, now)],
                        confidence=confidence,
                        features=features
                    )
                    
                    self.tracked_objects[object_id] = new_obj
                    updated_objects.append(new_obj)
            
            self._cleanup_stale_objects(now)
            
            return updated_objects
    
    def _find_matching_object(self,
                             camera_source: str,
                             bbox: Tuple[int, int, int, int],
                             object_type: str,
                             features: np.ndarray = None) -> Optional[str]:
        """查找匹配对象"""
        now = time.time()
        
        for obj_id, obj in self.tracked_objects.items():
            if obj.object_type != object_type:
                continue
            
            if now - obj.last_seen > 30:
                continue
            
            last_pos = None
            for pos in reversed(obj.positions):
                if pos[0] == camera_source:
                    last_pos = pos[1]
                    break
            
            if last_pos is None:
                continue
            
            iou = self._compute_iou(bbox, last_pos)
            if iou > 0.3:
                return obj_id
            
            if features is not None and obj.features is not None:
                similarity = self._compute_feature_similarity(features, obj.features)
                if similarity > self.similarity_threshold:
                    return obj_id
        
        return None
    
    def _compute_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """计算IoU"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _compute_feature_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """计算特征相似度"""
        if features1 is None or features2 is None:
            return 0.0
        
        if features1.shape != features2.shape:
            return 0.0
        
        dot = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
    
    def _cleanup_stale_objects(self, now: float, max_age: float = 60.0):
        """清理过期对象"""
        stale_ids = [
            obj_id for obj_id, obj in self.tracked_objects.items()
            if now - obj.last_seen > max_age
        ]
        
        for obj_id in stale_ids:
            del self.tracked_objects[obj_id]
    
    def get_object(self, object_id: str) -> Optional[TrackedObject]:
        """获取对象"""
        return self.tracked_objects.get(object_id)
    
    def get_active_objects(self, camera_source: str = None) -> List[TrackedObject]:
        """获取活跃对象"""
        now = time.time()
        active = []
        
        for obj in self.tracked_objects.values():
            if now - obj.last_seen > 10:
                continue
            
            if camera_source and camera_source not in obj.camera_sources:
                continue
            
            active.append(obj)
        
        return active


class FeatureExtractor:
    """特征提取器"""
    
    def extract(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """提取特征"""
        x, y, w, h = bbox
        roi = frame[y:y+h, x:x+w]
        
        if roi.size == 0:
            return np.zeros(128)
        
        roi_resized = cv2.resize(roi, (64, 128))
        
        if len(roi_resized.shape) == 3:
            hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)
            h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256])
            color_features = np.concatenate([h_hist.flatten(), s_hist.flatten(), v_hist.flatten()])
        else:
            color_features = cv2.calcHist([roi_resized], [0], None, [32], [0, 256]).flatten()
        
        gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY) if len(roi_resized.shape) == 3 else roi_resized
        
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, angle = cv2.cartToPolar(grad_x, grad_y)
        
        hist_mag = np.histogram(mag, bins=16, range=(0, 255))[0]
        hist_angle = np.histogram(angle, bins=16, range=(0, 2*np.pi))[0]
        texture_features = np.concatenate([hist_mag, hist_angle])
        
        features = np.concatenate([color_features, texture_features])
        features = features / (np.linalg.norm(features) + 1e-7)
        
        return features


class CorrelationDetector:
    """关联检测器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        self.loitering_threshold = self.config.get('loitering_threshold', 300)
        self.cross_camera_threshold = self.config.get('cross_camera_threshold', 60)
        self.crowd_threshold = self.config.get('crowd_threshold', 5)
        
        self.camera_zones: Dict[str, CameraZone] = {}
        self.alert_history: List[CorrelationAlert] = []
        self._lock = threading.Lock()
    
    def register_zone(self, zone: CameraZone):
        """注册区域"""
        with self._lock:
            self.camera_zones[zone.zone_id] = zone
            logging.info(f"[跨摄像头关联] 注册区域: {zone.zone_name} ({zone.camera_source})")
    
    def detect_correlations(self, 
                           tracked_objects: List[TrackedObject],
                           camera_source: str) -> List[CorrelationAlert]:
        """检测关联"""
        alerts = []
        now = time.time()
        
        with self._lock:
            alerts.extend(self._detect_loitering(tracked_objects, camera_source, now))
            alerts.extend(self._detect_cross_camera(tracked_objects, camera_source, now))
            alerts.extend(self._detect_crowd_gathering(tracked_objects, camera_source, now))
            alerts.extend(self._detect_entry_exit(tracked_objects, camera_source, now))
            
            self.alert_history.extend(alerts)
            
            self._cleanup_old_alerts(now)
        
        return alerts
    
    def _detect_loitering(self, 
                         tracked_objects: List[TrackedObject],
                         camera_source: str,
                         now: float) -> List[CorrelationAlert]:
        """检测滞留"""
        alerts = []
        
        for obj in tracked_objects:
            if camera_source not in obj.camera_sources:
                continue
            
            duration = obj.get_duration()
            if duration > self.loitering_threshold:
                alert_id = f"loitering_{obj.object_id}_{int(now)}"
                
                alert = CorrelationAlert(
                    alert_id=alert_id,
                    correlation_type=CorrelationType.LOITERING,
                    severity='medium',
                    cameras=[camera_source],
                    object_ids=[obj.object_id],
                    start_time=obj.first_seen,
                    end_time=now,
                    description=f"检测到滞留: {obj.object_type} 在 {camera_source} 停留 {int(duration)} 秒",
                    extra_data={
                        'duration': duration,
                        'object_type': obj.object_type
                    }
                )
                
                alerts.append(alert)
        
        return alerts
    
    def _detect_cross_camera(self,
                            tracked_objects: List[TrackedObject],
                            camera_source: str,
                            now: float) -> List[CorrelationAlert]:
        """检测跨摄像头"""
        alerts = []
        
        for obj in tracked_objects:
            if obj.get_camera_count() >= 2:
                cameras = list(obj.camera_sources)
                
                alert_id = f"cross_{obj.object_id}_{int(now)}"
                
                alert = CorrelationAlert(
                    alert_id=alert_id,
                    correlation_type=CorrelationType.CROSS_CAMERA,
                    severity='low',
                    cameras=cameras,
                    object_ids=[obj.object_id],
                    start_time=obj.first_seen,
                    end_time=now,
                    description=f"检测到跨摄像头移动: {obj.object_type} 出现在 {len(cameras)} 个摄像头",
                    extra_data={
                        'camera_count': len(cameras),
                        'cameras': cameras,
                        'object_type': obj.object_type
                    }
                )
                
                alerts.append(alert)
        
        return alerts
    
    def _detect_crowd_gathering(self,
                               tracked_objects: List[TrackedObject],
                               camera_source: str,
                               now: float) -> List[CorrelationAlert]:
        """检测人群聚集"""
        alerts = []
        
        recent_objects = [
            obj for obj in tracked_objects
            if camera_source in obj.camera_sources and now - obj.last_seen < 30
        ]
        
        if len(recent_objects) >= self.crowd_threshold:
            alert_id = f"crowd_{camera_source}_{int(now)}"
            
            alert = CorrelationAlert(
                alert_id=alert_id,
                correlation_type=CorrelationType.CROWD_GATHERING,
                severity='high',
                cameras=[camera_source],
                object_ids=[obj.object_id for obj in recent_objects],
                start_time=now - 30,
                end_time=now,
                description=f"检测到人群聚集: {camera_source} 有 {len(recent_objects)} 人",
                extra_data={
                    'count': len(recent_objects),
                    'threshold': self.crowd_threshold
                }
            )
            
            alerts.append(alert)
        
        return alerts
    
    def _detect_entry_exit(self,
                          tracked_objects: List[TrackedObject],
                          camera_source: str,
                          now: float) -> List[CorrelationAlert]:
        """检测进出"""
        alerts = []
        
        for zone_id, zone in self.camera_zones.items():
            if zone.camera_source != camera_source:
                continue
            
            for obj in tracked_objects:
                if camera_source not in obj.camera_sources:
                    continue
                
                entered = False
                exited = False
                
                for pos in obj.positions:
                    if pos[0] != camera_source:
                        continue
                    
                    bbox = pos[1]
                    center = (bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2)
                    
                    if self._point_in_polygon(center, zone.polygon):
                        if not entered:
                            entered = True
                    else:
                        if entered:
                            exited = True
                
                if entered and exited:
                    alert_id = f"entry_exit_{obj.object_id}_{zone.zone_id}_{int(now)}"
                    
                    alert = CorrelationAlert(
                        alert_id=alert_id,
                        correlation_type=CorrelationType.ENTRY_EXIT,
                        severity='low',
                        cameras=[camera_source],
                        object_ids=[obj.object_id],
                        start_time=obj.first_seen,
                        end_time=now,
                        description=f"检测到进出: {obj.object_type} 进入了 {zone.zone_name}",
                        extra_data={
                            'zone_id': zone.zone_id,
                            'zone_name': zone.zone_name,
                            'object_type': obj.object_type
                        }
                    )
                    
                    alerts.append(alert)
        
        return alerts
    
    def _point_in_polygon(self, point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
        """判断点是否在多边形内"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _cleanup_old_alerts(self, now: float, max_age: float = 3600):
        """清理旧告警"""
        self.alert_history = [
            alert for alert in self.alert_history
            if now - alert.end_time < max_age
        ]
    
    def get_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取告警"""
        with self._lock:
            return [
                {
                    'alert_id': alert.alert_id,
                    'type': alert.correlation_type.value,
                    'severity': alert.severity,
                    'cameras': alert.cameras,
                    'description': alert.description,
                    'start_time': datetime.fromtimestamp(alert.start_time).strftime('%Y-%m-%d %H:%M:%S'),
                    'end_time': datetime.fromtimestamp(alert.end_time).strftime('%Y-%m-%d %H:%M:%S'),
                    'duration': alert.end_time - alert.start_time
                }
                for alert in sorted(self.alert_history, key=lambda x: x.end_time, reverse=True)[:limit]
            ]


class CrossCameraCorrelator:
    """跨摄像头关联器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        self.object_tracker = ObjectTracker()
        self.correlation_detector = CorrelationDetector(config)
        
        self._lock = threading.Lock()
        
        logging.info("[跨摄像头关联] 关联器初始化完成")
    
    def process_detections(self,
                          camera_source: str,
                          detections: List[Dict[str, Any]],
                          frame: np.ndarray = None) -> List[CorrelationAlert]:
        """处理检测结果"""
        with self._lock:
            tracked_objects = self.object_tracker.update(camera_source, detections, frame)
            
            alerts = self.correlation_detector.detect_correlations(tracked_objects, camera_source)
            
            return alerts
    
    def register_zone(self, 
                     camera_source: str,
                     zone_id: str,
                     zone_name: str,
                     polygon: List[Tuple[int, int]],
                     zone_type: str = "general"):
        """注册区域"""
        zone = CameraZone(
            camera_source=camera_source,
            zone_id=zone_id,
            zone_name=zone_name,
            polygon=polygon,
            zone_type=zone_type
        )
        
        self.correlation_detector.register_zone(zone)
    
    def get_tracked_objects(self, camera_source: str = None) -> List[Dict[str, Any]]:
        """获取跟踪对象"""
        objects = self.object_tracker.get_active_objects(camera_source)
        
        return [
            {
                'object_id': obj.object_id,
                'object_type': obj.object_type,
                'cameras': list(obj.camera_sources),
                'duration': obj.get_duration(),
                'first_seen': datetime.fromtimestamp(obj.first_seen).strftime('%Y-%m-%d %H:%M:%S'),
                'last_seen': datetime.fromtimestamp(obj.last_seen).strftime('%Y-%m-%d %H:%M:%S')
            }
            for obj in objects
        ]
    
    def get_correlation_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取关联告警"""
        return self.correlation_detector.get_alerts(limit)


import cv2

_cross_camera_correlator: Optional[CrossCameraCorrelator] = None


def get_cross_camera_correlator(config: Dict[str, Any] = None) -> CrossCameraCorrelator:
    """获取跨摄像头关联器单例"""
    global _cross_camera_correlator
    if _cross_camera_correlator is None:
        _cross_camera_correlator = CrossCameraCorrelator(config)
    return _cross_camera_correlator
