#!/usr/bin/env python3
"""
质量控制模块 - 增强版
集成误报学习、场景自适应、跨摄像头关联
"""

import time
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque

import numpy as np

from algorithms.algorithm_base import AlgorithmResult, AlgorithmCategory


@dataclass
class QualityConfig:
    """质量控制配置"""
    min_confidence: float = 0.65
    cooldown_seconds: float = 60.0
    consecutive_frames: int = 2
    dynamic_cooldown: bool = True
    alert_aggregation: bool = True
    aggregation_window: float = 5.0
    max_history: int = 30
    enable_false_positive_learning: bool = True
    enable_scene_adaptive: bool = True
    enable_cross_camera: bool = True


class QualityController:
    """
    质量控制器 - 增强版
    
    功能：
    1. 置信度过滤
    2. 冷却期控制（动态调整）
    3. 连续帧确认
    4. 告警聚合
    5. 帧有效性检测
    6. 误报学习（可选）
    7. 场景自适应（可选）
    8. 跨摄像头关联（可选）
    """
    
    DEPARTMENT_FPR = {
        'person': 0.20,
        'vehicle': 0.15,
        'environment': 0.35,
        'behavior': 0.28,
    }
    
    def __init__(self, config: QualityConfig = None):
        self.config = config or QualityConfig()
        
        self.cooldown_map: Dict[str, float] = {}
        self.pending_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_frequency: Dict[str, List[float]] = defaultdict(list)
        self.recent_alerts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.history: Dict[str, List[AlgorithmResult]] = defaultdict(list)
        
        self._lock = threading.Lock()
        
        self.false_positive_learner = None
        self.scene_adaptive_controller = None
        self.cross_camera_correlator = None
        
        if self.config.enable_false_positive_learning:
            try:
                from false_positive_learner import get_false_positive_learner
                self.false_positive_learner = get_false_positive_learner()
                logging.info("[质量控制] 误报学习已启用")
            except Exception as e:
                logging.warning(f"[质量控制] 误报学习加载失败: {e}")
        
        if self.config.enable_scene_adaptive:
            try:
                from scene_adaptive import get_scene_adaptive_controller
                self.scene_adaptive_controller = get_scene_adaptive_controller()
                logging.info("[质量控制] 场景自适应已启用")
            except Exception as e:
                logging.warning(f"[质量控制] 场景自适应加载失败: {e}")
        
        if self.config.enable_cross_camera:
            try:
                from cross_camera_correlator import get_cross_camera_correlator
                self.cross_camera_correlator = get_cross_camera_correlator()
                logging.info("[质量控制] 跨摄像头关联已启用")
            except Exception as e:
                logging.warning(f"[质量控制] 跨摄像头关联加载失败: {e}")
        
        self.stats = {
            'total_input': 0,
            'passed': 0,
            'filtered_low_confidence': 0,
            'filtered_cooldown': 0,
            'filtered_not_confirmed': 0,
            'aggregated': 0,
            'false_positive_filtered': 0,
            'scene_adjusted': 0,
        }
    
    def process(self, results: List[AlgorithmResult], 
                frame: np.ndarray = None,
                camera_id: str = "default") -> List[AlgorithmResult]:
        """
        处理算法结果，返回通过质量控制的告警
        """
        with self._lock:
            self.stats['total_input'] += len(results)
            
            if frame is not None and not self._is_valid_frame(frame):
                return []
            
            if self.scene_adaptive_controller and frame is not None:
                self.scene_adaptive_controller.update_scene(camera_id, frame)
            
            passed = []
            now = time.time()
            
            for result in results:
                if not result.detected:
                    continue
                
                min_conf = self.config.min_confidence
                cooldown_mult = 1.0
                
                if self.scene_adaptive_controller:
                    adjusted = self.scene_adaptive_controller.get_adjusted_thresholds(
                        camera_id, result.algorithm_id
                    )
                    min_conf = adjusted.min_confidence
                    cooldown_mult = adjusted.cooldown_seconds / self.config.cooldown_seconds
                    self.stats['scene_adjusted'] += 1
                
                if result.confidence < min_conf:
                    self.stats['filtered_low_confidence'] += 1
                    continue
                
                if self.false_positive_learner:
                    is_fp, adjusted_conf = self.false_positive_learner.check_false_positive(
                        algorithm_id=result.algorithm_id,
                        confidence=result.confidence,
                        bbox=result.bounding_box,
                        frame=frame,
                        camera_source=camera_id
                    )
                    
                    if is_fp:
                        self.stats['false_positive_filtered'] += 1
                        continue
                    
                    result.confidence = adjusted_conf
                
                cooldown_key = self._make_cooldown_key(result, camera_id)
                base_cooldown = self.cooldown_map.get(cooldown_key, 0)
                if now < base_cooldown:
                    self.stats['filtered_cooldown'] += 1
                    continue
                
                if self.config.consecutive_frames > 1:
                    confirm_key = f"{result.algorithm_id}:{camera_id}"
                    if not self._check_consecutive(confirm_key, result):
                        self.stats['filtered_not_confirmed'] += 1
                        continue
                
                cooldown_dur = self._calc_cooldown(result, camera_id) * cooldown_mult
                self.cooldown_map[cooldown_key] = now + cooldown_dur
                
                passed.append(result)
            
            if self.config.alert_aggregation:
                passed = self._aggregate_alerts(passed, camera_id)
            
            if self.cross_camera_correlator and passed:
                detections = [
                    {
                        'type': r.algorithm_name,
                        'bbox': r.bounding_box,
                        'confidence': r.confidence
                    }
                    for r in passed if r.bounding_box
                ]
                if detections:
                    self.cross_camera_correlator.process_detections(
                        camera_id, detections, frame
                    )
            
            self.stats['passed'] += len(passed)
            
            self._cleanup(now)
            
            return passed
    
    def _is_valid_frame(self, frame: np.ndarray) -> bool:
        """检查帧有效性"""
        if frame is None or frame.size == 0:
            return False
        
        if cv2 is None:
            return True
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            mean_val = np.mean(gray)
            std_val = np.std(gray)
            
            if mean_val < 30 and std_val < 15:
                return False
            if mean_val < 20:
                return False
            if std_val < 5:
                return False
            if mean_val > 245 and std_val < 20:
                return False
            
            h, w = gray.shape[:2]
            edge_map = cv2.Canny(gray, 50, 150)
            edge_ratio = cv2.countNonZero(edge_map) / (h * w)
            if edge_ratio < 0.001:
                return False
        except Exception:
            return True
        
        return True
    
    def _make_cooldown_key(self, result: AlgorithmResult, camera_id: str) -> str:
        """生成冷却键"""
        if result.bounding_box:
            x, y, w, h = result.bounding_box
            grid_x = int(x // 100) * 100
            grid_y = int(y // 100) * 100
            location = f"{grid_x}_{grid_y}"
        else:
            location = "full"
        return f"{result.algorithm_id}:{camera_id}:{location}"
    
    def _calc_cooldown(self, result: AlgorithmResult, camera_id: str) -> float:
        """计算动态冷却时间"""
        base = self.config.cooldown_seconds
        
        conf_factor = max(0.5, 1.5 - result.confidence)
        
        if self.config.dynamic_cooldown:
            freq_key = f"{result.algorithm_id}:{camera_id}"
            now = time.time()
            recent = [t for t in self.alert_frequency[freq_key] if now - t < 300]
            self.alert_frequency[freq_key] = recent
            
            if len(recent) > 10:
                freq_factor = 2.0
            elif len(recent) > 5:
                freq_factor = 1.5
            elif len(recent) > 2:
                freq_factor = 1.2
            else:
                freq_factor = 1.0
            
            self.alert_frequency[freq_key].append(now)
        else:
            freq_factor = 1.0
        
        return base * conf_factor * freq_factor
    
    def _check_consecutive(self, key: str, result: AlgorithmResult) -> bool:
        """连续帧确认"""
        now = time.time()
        
        if key not in self.pending_alerts:
            self.pending_alerts[key] = {
                'count': 1,
                'first_seen': now,
                'last_seen': now,
                'confidence_sum': result.confidence
            }
            return False
        
        pending = self.pending_alerts[key]
        
        if now - pending['last_seen'] > 2.0:
            pending['count'] = 1
            pending['first_seen'] = now
            pending['confidence_sum'] = result.confidence
        else:
            pending['count'] += 1
            pending['confidence_sum'] += result.confidence
        
        pending['last_seen'] = now
        
        if pending['count'] >= self.config.consecutive_frames:
            avg_conf = pending['confidence_sum'] / pending['count']
            if avg_conf >= self.config.min_confidence:
                del self.pending_alerts[key]
                return True
        
        return False
    
    def _aggregate_alerts(self, alerts: List[AlgorithmResult], 
                          camera_id: str) -> List[AlgorithmResult]:
        """告警聚合"""
        now = time.time()
        aggregated = []
        
        for alert in alerts:
            if alert.bounding_box:
                x, y, w, h = alert.bounding_box
                grid_key = f"{alert.algorithm_id}:{int(x//50)}_{int(y//50)}"
            else:
                grid_key = f"{alert.algorithm_id}:full"
            
            alert_key = f"{grid_key}:{camera_id}"
            
            recent = self.recent_alerts.get(alert_key, [])
            recent = [a for a in recent if now - a['time'] < self.config.aggregation_window]
            
            if recent:
                recent[0]['count'] += 1
                recent[0]['time'] = now
                if alert.confidence > recent[0]['confidence']:
                    recent[0]['confidence'] = alert.confidence
                    recent[0]['result'] = alert
                self.recent_alerts[alert_key] = recent
                self.stats['aggregated'] += 1
            else:
                self.recent_alerts[alert_key] = [{
                    'time': now,
                    'confidence': alert.confidence,
                    'result': alert,
                    'count': 1
                }]
                aggregated.append(alert)
        
        return aggregated
    
    def _cleanup(self, now: float):
        """清理过期数据"""
        for key in list(self.cooldown_map.keys()):
            if now > self.cooldown_map[key] + 300:
                del self.cooldown_map[key]
        
        for key in list(self.recent_alerts.keys()):
            self.recent_alerts[key] = [a for a in self.recent_alerts[key] 
                                        if now - a['time'] < self.config.aggregation_window * 2]
            if not self.recent_alerts[key]:
                del self.recent_alerts[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'pass_rate': self.stats['passed'] / max(1, self.stats['total_input']),
            'pending_alerts': len(self.pending_alerts),
            'active_cooldowns': len(self.cooldown_map),
        }


try:
    import cv2
except ImportError:
    cv2 = None

_quality_controller = None

def get_quality_controller(config: Dict[str, Any] = None) -> QualityController:
    """获取质量控制器单例"""
    global _quality_controller
    if _quality_controller is None:
        qc_config = QualityConfig(
            min_confidence=config.get('min_confidence', 0.65) if config else 0.65,
            cooldown_seconds=config.get('cooldown_seconds', 60.0) if config else 60.0,
            consecutive_frames=config.get('consecutive_frames', 2) if config else 2,
            dynamic_cooldown=config.get('dynamic_cooldown', True) if config else True,
            alert_aggregation=config.get('alert_aggregation', True) if config else True,
            aggregation_window=config.get('aggregation_window', 5.0) if config else 5.0,
        )
        _quality_controller = QualityController(qc_config)
    return _quality_controller
