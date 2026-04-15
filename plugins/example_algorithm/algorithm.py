#!/usr/bin/env python3
"""
示例算法插件
演示如何开发自定义算法插件
"""

import cv2
import numpy as np
from typing import Dict, Any


class ExampleAlgorithm:
    """亮度检测算法示例
    
    检测画面亮度，当亮度超过阈值时触发告警
    可用于监控画面异常（过曝、过暗等）
    """
    
    ALGORITHM_ID = 100
    ALGORITHM_NAME = "亮度检测"
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.threshold = self.config.get('threshold', 0.5)
        self.brightness_threshold = self.config.get('brightness_threshold', 100)
        self._initialized = False
    
    def initialize(self) -> bool:
        """初始化算法"""
        self._initialized = True
        return True
    
    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """处理帧
        
        Args:
            frame: 输入帧 (BGR格式)
            context: 上下文信息
            
        Returns:
            检测结果字典
        """
        result = {
            'algorithm_id': self.ALGORITHM_ID,
            'algorithm_name': self.ALGORITHM_NAME,
            'detected': False,
            'confidence': 0.0,
            'bounding_box': None,
            'extra_data': {}
        }
        
        if frame is None or frame.size == 0:
            return result
        
        h, w = frame.shape[:2]
        
        brightness = np.mean(frame)
        
        if brightness > self.brightness_threshold:
            result['detected'] = True
            result['confidence'] = min(1.0, brightness / 255)
            result['extra_data'] = {
                'brightness': float(brightness),
                'frame_size': f"{w}x{h}",
                'threshold': self.brightness_threshold
            }
        
        return result
    
    def get_info(self) -> Dict[str, Any]:
        """获取算法信息"""
        return {
            'id': self.ALGORITHM_ID,
            'name': self.ALGORITHM_NAME,
            'description': '检测画面亮度异常',
            'config': self.config,
            'initialized': self._initialized
        }
