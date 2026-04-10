#!/usr/bin/env python3
"""
AI分析模块 - 集成50种智能算法
"""

import os
import sys
import time
import logging
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from algorithms.algorithm_base import AlgorithmResult
from algorithms.algorithm_manager import AlgorithmManager


class EnhancedAIAnalyzer:
    def __init__(self, config):
        self.config = config
        self.running = False
        self.thread = None
        self.results: Dict[str, List[AlgorithmResult]] = {}
        self.results_lock = threading.Lock()
        self.camera_manager = None
        
        self.algorithm_manager = AlgorithmManager(config)
        self.enabled_algorithms = self._load_enabled_algorithms()
        
    def _load_enabled_algorithms(self) -> List[int]:
        enabled = self.config.get('ai.enabled_algorithms', [])
        if not enabled:
            return list(range(1, 51))
        return enabled
        
    def initialize(self) -> bool:
        return self.algorithm_manager.initialize_all()
        
    def _analysis_loop(self):
        while self.running:
            try:
                if self.camera_manager:
                    # 兼容两种摄像头管理器
                    cameras = self.camera_manager.get_all_cameras()
                    for camera in cameras:
                        frame = camera.get_frame()
                        if frame is not None:
                            # 确定摄像头源标识符
                            if hasattr(camera, 'source'):
                                source = camera.source
                            elif hasattr(camera, 'config') and hasattr(camera.config, 'source'):
                                source = camera.config.source
                            else:
                                source = str(id(camera))
                            
                            # 检测是否为无人机摄像头（基于源标识符或配置）
                            is_drone_camera = self._is_drone_camera(source)
                            
                            # 根据摄像头类型选择合适的算法
                            if is_drone_camera:
                                # 无人机摄像头使用专用算法
                                drone_algorithms = [68, 69]  # 无人机烟火检测算法
                                results = self.algorithm_manager.process_frame(
                                    frame, drone_algorithms
                                )
                            else:
                                # 普通摄像头使用配置的算法
                                results = self.algorithm_manager.process_frame(
                                    frame, self.enabled_algorithms
                                )
                            
                            with self.results_lock:
                                self.results[source] = results
            except Exception as e:
                logging.error(f"Analysis error: {e}")
            
            time.sleep(0.033)
    
    def _is_drone_camera(self, source: str) -> bool:
        """检测是否为无人机摄像头"""
        # 基于源标识符判断
        drone_indicators = ['drone', 'uav', 'air', 'sky', 'rtsp://drone', 'drone://']
        source_lower = source.lower()
        
        for indicator in drone_indicators:
            if indicator in source_lower:
                return True
        
        return False

    def start(self):
        logging.info("Starting Enhanced AI analyzer...")
        self.initialize()
        self.running = True
        self.thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.thread.start()
        logging.info("Enhanced AI analyzer started")

    def stop(self):
        logging.info("Stopping Enhanced AI analyzer...")
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        self.algorithm_manager.release()
        logging.info("Enhanced AI analyzer stopped")

    def set_camera_manager(self, camera_manager):
        self.camera_manager = camera_manager

    def get_results(self, camera_source: str = None) -> List[AlgorithmResult]:
        with self.results_lock:
            if camera_source:
                return self.results.get(camera_source, [])
            else:
                all_results = []
                for results in self.results.values():
                    all_results.extend(results)
                return all_results

    def visualize(self, frame: np.ndarray, results: List[AlgorithmResult]) -> np.ndarray:
        return self.algorithm_manager.visualize_results(frame, results)
        
    def get_all_algorithms_info(self) -> List[Dict]:
        return self.algorithm_manager.get_all_algorithm_info()
        
    def enable_algorithm(self, algo_id: int, enabled: bool = True):
        self.algorithm_manager.enable_algorithm(algo_id, enabled)


AIAnalyzer = EnhancedAIAnalyzer
