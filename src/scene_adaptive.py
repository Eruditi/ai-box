#!/usr/bin/env python3
"""
场景自适应阈值调整系统
根据环境条件自动调整检测参数
"""

import os
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, time as dt_time
from enum import Enum

import numpy as np


class SceneType(Enum):
    """场景类型"""
    DAY = "day"
    NIGHT = "night"
    DAWN_DUSK = "dawn_dusk"
    RAINY = "rainy"
    FOGGY = "foggy"
    SNOWY = "snowy"
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    UNKNOWN = "unknown"


class WeatherCondition(Enum):
    """天气条件"""
    CLEAR = "clear"
    CLOUDY = "cloudy"
    RAIN = "rain"
    HEAVY_RAIN = "heavy_rain"
    FOG = "fog"
    SNOW = "snow"
    UNKNOWN = "unknown"


@dataclass
class SceneState:
    """场景状态"""
    scene_type: SceneType
    weather: WeatherCondition
    brightness: float
    contrast: float
    noise_level: float
    motion_level: float
    timestamp: float
    confidence: float = 1.0


@dataclass
class AdaptiveThresholds:
    """自适应阈值"""
    min_confidence: float = 0.65
    cooldown_seconds: float = 60.0
    consecutive_frames: int = 2
    detection_sensitivity: float = 1.0
    noise_filter: float = 0.5
    motion_threshold: float = 25.0


@dataclass
class AlgorithmThresholds:
    """算法特定阈值"""
    algorithm_id: int
    algorithm_name: str
    base_thresholds: AdaptiveThresholds
    scene_adjustments: Dict[SceneType, Dict[str, float]] = field(default_factory=dict)


class SceneAnalyzer:
    """场景分析器"""
    
    def __init__(self):
        self.brightness_history: deque = deque(maxlen=30)
        self.contrast_history: deque = deque(maxlen=30)
        self.noise_history: deque = deque(maxlen=30)
        self.motion_history: deque = deque(maxlen=30)
        self.prev_frame: Optional[np.ndarray] = None
    
    def analyze_frame(self, frame: np.ndarray) -> SceneState:
        """分析帧场景"""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        brightness = self._compute_brightness(gray)
        contrast = self._compute_contrast(gray)
        noise_level = self._compute_noise_level(gray)
        motion_level = self._compute_motion_level(gray)
        
        self.brightness_history.append(brightness)
        self.contrast_history.append(contrast)
        self.noise_history.append(noise_level)
        self.motion_history.append(motion_level)
        
        scene_type = self._determine_scene_type(brightness, contrast)
        weather = self._estimate_weather(brightness, contrast, noise_level)
        
        self.prev_frame = gray.copy()
        
        return SceneState(
            scene_type=scene_type,
            weather=weather,
            brightness=brightness,
            contrast=contrast,
            noise_level=noise_level,
            motion_level=motion_level,
            timestamp=time.time()
        )
    
    def _compute_brightness(self, gray: np.ndarray) -> float:
        """计算亮度"""
        return float(np.mean(gray))
    
    def _compute_contrast(self, gray: np.ndarray) -> float:
        """计算对比度"""
        return float(np.std(gray))
    
    def _compute_noise_level(self, gray: np.ndarray) -> float:
        """计算噪声水平"""
        if gray.shape[0] < 3 or gray.shape[1] < 3:
            return 0.0
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise = np.var(laplacian)
        
        return float(min(noise / 1000.0, 1.0))
    
    def _compute_motion_level(self, gray: np.ndarray) -> float:
        """计算运动水平"""
        if self.prev_frame is None:
            return 0.0
        
        if self.prev_frame.shape != gray.shape:
            self.prev_frame = gray.copy()
            return 0.0
        
        diff = cv2.absdiff(self.prev_frame, gray)
        motion = np.mean(diff)
        
        return float(motion)
    
    def _determine_scene_type(self, brightness: float, contrast: float) -> SceneType:
        """确定场景类型"""
        current_hour = datetime.now().hour
        
        if brightness < 30:
            return SceneType.NIGHT
        elif brightness < 60:
            if 5 <= current_hour <= 8 or 17 <= current_hour <= 20:
                return SceneType.DAWN_DUSK
            else:
                return SceneType.NIGHT
        elif brightness > 200:
            return SceneType.DAY
        else:
            if 5 <= current_hour <= 8 or 17 <= current_hour <= 20:
                return SceneType.DAWN_DUSK
            return SceneType.DAY
    
    def _estimate_weather(self, brightness: float, contrast: float, noise_level: float) -> WeatherCondition:
        """估计天气条件"""
        if len(self.brightness_history) < 10:
            return WeatherCondition.UNKNOWN
        
        avg_brightness = np.mean(list(self.brightness_history))
        avg_contrast = np.mean(list(self.contrast_history))
        avg_noise = np.mean(list(self.noise_history))
        
        if avg_noise > 0.7:
            return WeatherCondition.HEAVY_RAIN
        elif avg_noise > 0.5:
            if avg_contrast < 30:
                return WeatherCondition.FOG
            return WeatherCondition.RAIN
        elif avg_contrast < 25 and avg_brightness < 100:
            return WeatherCondition.FOG
        elif avg_brightness < 80 and avg_contrast < 40:
            return WeatherCondition.CLOUDY
        else:
            return WeatherCondition.CLEAR


class ThresholdAdjuster:
    """阈值调整器"""
    
    DEFAULT_ADJUSTMENTS = {
        SceneType.DAY: {
            'confidence_factor': 1.0,
            'cooldown_factor': 1.0,
            'sensitivity_factor': 1.0,
        },
        SceneType.NIGHT: {
            'confidence_factor': 0.85,
            'cooldown_factor': 1.5,
            'sensitivity_factor': 1.3,
        },
        SceneType.DAWN_DUSK: {
            'confidence_factor': 0.9,
            'cooldown_factor': 1.2,
            'sensitivity_factor': 1.1,
        },
        SceneType.RAINY: {
            'confidence_factor': 0.8,
            'cooldown_factor': 1.8,
            'sensitivity_factor': 0.7,
        },
        SceneType.FOGGY: {
            'confidence_factor': 0.75,
            'cooldown_factor': 2.0,
            'sensitivity_factor': 0.6,
        },
        SceneType.SNOWY: {
            'confidence_factor': 0.7,
            'cooldown_factor': 2.0,
            'sensitivity_factor': 0.5,
        },
    }
    
    WEATHER_ADJUSTMENTS = {
        WeatherCondition.CLEAR: {'confidence_factor': 1.0},
        WeatherCondition.CLOUDY: {'confidence_factor': 0.95},
        WeatherCondition.RAIN: {'confidence_factor': 0.85, 'cooldown_factor': 1.3},
        WeatherCondition.HEAVY_RAIN: {'confidence_factor': 0.7, 'cooldown_factor': 1.8},
        WeatherCondition.FOG: {'confidence_factor': 0.75, 'cooldown_factor': 1.5},
        WeatherCondition.SNOW: {'confidence_factor': 0.7, 'cooldown_factor': 1.6},
    }
    
    def __init__(self):
        self.algorithm_thresholds: Dict[int, AlgorithmThresholds] = {}
        self._lock = threading.Lock()
    
    def register_algorithm(self, 
                          algorithm_id: int, 
                          algorithm_name: str,
                          base_thresholds: AdaptiveThresholds = None):
        """注册算法阈值"""
        with self._lock:
            if base_thresholds is None:
                base_thresholds = AdaptiveThresholds()
            
            self.algorithm_thresholds[algorithm_id] = AlgorithmThresholds(
                algorithm_id=algorithm_id,
                algorithm_name=algorithm_name,
                base_thresholds=base_thresholds,
                scene_adjustments=self.DEFAULT_ADJUSTMENTS.copy()
            )
            
            logging.info(f"[场景自适应] 注册算法: {algorithm_name} (ID: {algorithm_id})")
    
    def adjust_thresholds(self,
                         algorithm_id: int,
                         scene_state: SceneState) -> AdaptiveThresholds:
        """调整阈值"""
        with self._lock:
            if algorithm_id not in self.algorithm_thresholds:
                return AdaptiveThresholds()
            
            algo_thresholds = self.algorithm_thresholds[algorithm_id]
            base = algo_thresholds.base_thresholds
            
            scene_adj = self.DEFAULT_ADJUSTMENTS.get(scene_state.scene_type, {})
            weather_adj = self.WEATHER_ADJUSTMENTS.get(scene_state.weather, {})
            
            confidence_factor = scene_adj.get('confidence_factor', 1.0) * weather_adj.get('confidence_factor', 1.0)
            cooldown_factor = scene_adj.get('cooldown_factor', 1.0) * weather_adj.get('cooldown_factor', 1.0)
            sensitivity_factor = scene_adj.get('sensitivity_factor', 1.0)
            
            if scene_state.noise_level > 0.5:
                confidence_factor *= 0.9
                cooldown_factor *= 1.2
            
            if scene_state.motion_level > 50:
                sensitivity_factor *= 1.1
            
            adjusted = AdaptiveThresholds(
                min_confidence=base.min_confidence * confidence_factor,
                cooldown_seconds=base.cooldown_seconds * cooldown_factor,
                consecutive_frames=max(1, int(base.consecutive_frames * (1 + (1 - sensitivity_factor)))),
                detection_sensitivity=base.detection_sensitivity * sensitivity_factor,
                noise_filter=base.noise_filter * (1 + scene_state.noise_level),
                motion_threshold=base.motion_threshold * sensitivity_factor
            )
            
            return adjusted


class SceneAdaptiveController:
    """场景自适应控制器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        self.scene_analyzer = SceneAnalyzer()
        self.threshold_adjuster = ThresholdAdjuster()
        
        self.scene_states: Dict[str, SceneState] = {}
        self.adjusted_thresholds: Dict[str, Dict[int, AdaptiveThresholds]] = {}
        
        self.update_interval = self.config.get('update_interval', 5.0)
        self._lock = threading.Lock()
        
        self._register_default_algorithms()
        
        logging.info("[场景自适应] 控制器初始化完成")
    
    def _register_default_algorithms(self):
        """注册默认算法"""
        default_algorithms = [
            (1, "安全帽检测", AdaptiveThresholds(min_confidence=0.6, cooldown_seconds=45)),
            (2, "人脸识别", AdaptiveThresholds(min_confidence=0.7, cooldown_seconds=30)),
            (6, "火焰检测", AdaptiveThresholds(min_confidence=0.65, cooldown_seconds=15)),
            (7, "烟雾检测", AdaptiveThresholds(min_confidence=0.6, cooldown_seconds=20)),
            (16, "人员检测", AdaptiveThresholds(min_confidence=0.65, cooldown_seconds=30)),
            (17, "入侵检测", AdaptiveThresholds(min_confidence=0.7, cooldown_seconds=60)),
        ]
        
        for algo_id, algo_name, thresholds in default_algorithms:
            self.threshold_adjuster.register_algorithm(algo_id, algo_name, thresholds)
    
    def update_scene(self, camera_source: str, frame: np.ndarray):
        """更新场景状态"""
        with self._lock:
            scene_state = self.scene_analyzer.analyze_frame(frame)
            self.scene_states[camera_source] = scene_state
            
            if camera_source not in self.adjusted_thresholds:
                self.adjusted_thresholds[camera_source] = {}
            
            for algo_id in self.threshold_adjuster.algorithm_thresholds.keys():
                adjusted = self.threshold_adjuster.adjust_thresholds(algo_id, scene_state)
                self.adjusted_thresholds[camera_source][algo_id] = adjusted
    
    def get_adjusted_thresholds(self, 
                               camera_source: str, 
                               algorithm_id: int) -> AdaptiveThresholds:
        """获取调整后的阈值"""
        with self._lock:
            if camera_source in self.adjusted_thresholds:
                if algorithm_id in self.adjusted_thresholds[camera_source]:
                    return self.adjusted_thresholds[camera_source][algorithm_id]
            
            if algorithm_id in self.threshold_adjuster.algorithm_thresholds:
                return self.threshold_adjuster.algorithm_thresholds[algorithm_id].base_thresholds
            
            return AdaptiveThresholds()
    
    def get_scene_info(self, camera_source: str) -> Dict[str, Any]:
        """获取场景信息"""
        with self._lock:
            if camera_source not in self.scene_states:
                return {
                    'scene_type': SceneType.UNKNOWN.value,
                    'weather': WeatherCondition.UNKNOWN.value,
                    'brightness': 0,
                    'contrast': 0,
                    'noise_level': 0,
                    'motion_level': 0
                }
            
            state = self.scene_states[camera_source]
            return {
                'scene_type': state.scene_type.value,
                'weather': state.weather.value,
                'brightness': round(state.brightness, 2),
                'contrast': round(state.contrast, 2),
                'noise_level': round(state.noise_level, 3),
                'motion_level': round(state.motion_level, 2),
                'timestamp': state.timestamp
            }
    
    def get_all_scenes(self) -> Dict[str, Dict[str, Any]]:
        """获取所有场景信息"""
        with self._lock:
            return {
                source: self.get_scene_info(source)
                for source in self.scene_states.keys()
            }
    
    def register_algorithm(self,
                          algorithm_id: int,
                          algorithm_name: str,
                          base_thresholds: AdaptiveThresholds = None):
        """注册算法"""
        self.threshold_adjuster.register_algorithm(algorithm_id, algorithm_name, base_thresholds)


import cv2

_scene_adaptive_controller: Optional[SceneAdaptiveController] = None


def get_scene_adaptive_controller(config: Dict[str, Any] = None) -> SceneAdaptiveController:
    """获取场景自适应控制器单例"""
    global _scene_adaptive_controller
    if _scene_adaptive_controller is None:
        _scene_adaptive_controller = SceneAdaptiveController(config)
    return _scene_adaptive_controller
