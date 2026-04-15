#!/usr/bin/env python3
"""
test_algorithm 算法插件
使用 AI Box Plugin SDK v1.0.0 开发
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional

from plugin_sdk import BaseAlgorithm, AlgorithmMetadata, AlgorithmCategory, DetectionResult


class MainAlgorithm(BaseAlgorithm):
    """主算法类"""
    
    @classmethod
    def get_metadata(cls) -> AlgorithmMetadata:
        """获取算法元数据"""
        return AlgorithmMetadata(
            algorithm_id=999,
            name="test_algorithm检测",
            description="检测test_algorithm的算法",
            category=AlgorithmCategory.STRUCTURED_ANALYSIS,
            version="1.0.0",
            author="Your Name",
            tags=["test_algorithm", "detection"],
            min_confidence=0.5,
            config_schema={
                "threshold": {
                    "type": "number",
                    "default": 0.5,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "检测阈值"
                }
            }
        )
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.threshold = self.config.get('threshold', 0.5)
        self.initialized = False
    
    def initialize(self, config: Dict[str, Any] = None) -> bool:
        """初始化算法"""
        if config:
            self.config.update(config)
            self.threshold = self.config.get('threshold', 0.5)
        
        if not self.validate_config(self.config):
            return False
        
        self.initialized = True
        return True
    
    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> DetectionResult:
        """处理帧"""
        if not self.initialized:
            return DetectionResult(detected=False, confidence=0.0)
        
        h, w = frame.shape[:2]
        
        brightness = np.mean(frame)
        
        detected = brightness > 100
        confidence = min(1.0, brightness / 255) if detected else 0.0
        
        return DetectionResult(
            detected=detected and confidence >= self.threshold,
            confidence=confidence,
            bounding_box=(0, 0, w, h) if detected else None,
            label="test_algorithm",
            attributes={
                "brightness": float(brightness)
            }
        )
    
    def cleanup(self):
        """清理资源"""
        self.initialized = False
