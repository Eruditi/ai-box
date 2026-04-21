#!/usr/bin/env python3
"""
AI分析模块 - 简化版
直接处理流程：算法检测 → 质量控制 → 告警输出
"""

import os
import sys
import time
import logging
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from algorithms.algorithm_base import AlgorithmResult, AlgorithmCategory
from algorithms.algorithm_manager import AlgorithmManager
from quality_controller import get_quality_controller
from alert_database import get_alert_db
from llm_engine import LLMEngine
from alert_pusher import AlertPusher


class SimpleAIAnalyzer:
    """
    简化版AI分析器
    
    流程：
    1. 获取摄像头帧
    2. 调用算法检测
    3. 质量控制过滤
    4. 输出告警
    """
    
    def __init__(self, config):
        self.config = config
        self.running = False
        self.thread = None
        self.results: Dict[str, List[AlgorithmResult]] = {}
        self.results_lock = threading.Lock()
        self.camera_manager = None
        
        self.algorithm_manager = AlgorithmManager(config)
        self.enabled_algorithms = self._load_enabled_algorithms()
        
        qc_config = config.get('audit', {})
        qc_config.update(config.get('supervisor', {}))
        self.quality_controller = get_quality_controller(qc_config)
        
        self.frame_counter = 0
        
        self.alert_db = get_alert_db()
        self.llm_engine = LLMEngine(config)
        self.alert_pusher = AlertPusher(config)
    
    def _load_enabled_algorithms(self) -> List[int]:
        """加载启用的算法列表"""
        enabled = self.config.get('ai.enabled_algorithms', [])
        if not enabled:
            enabled = list(self.algorithm_manager.algorithm_configs.keys())
            logging.info(f"[算法] 默认启用所有算法: {len(enabled)} 个")
        return enabled
    
    def initialize(self) -> bool:
        """初始化算法"""
        return self.algorithm_manager.initialize_all()
    
    def set_camera_manager(self, camera_manager):
        """设置摄像头管理器"""
        self.camera_manager = camera_manager
    
    def start(self):
        """启动分析线程"""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.thread.start()
        logging.info("AI分析器 已启动")
    
    def stop(self):
        """停止分析线程"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        logging.info("AI分析器 已停止")
    
    def _analysis_loop(self):
        """主分析循环"""
        while self.running:
            try:
                if not self.camera_manager:
                    time.sleep(0.1)
                    continue
                
                cameras = self.camera_manager.get_all_cameras()
                
                if not cameras:
                    time.sleep(0.5)
                    continue
                
                for camera in cameras:
                    self._process_camera_frame(camera)
                
                self.frame_counter += 1
                time.sleep(0.033)
                
            except Exception as e:
                logging.error(f"[分析循环] 异常: {e}", exc_info=True)
                time.sleep(0.1)
    
    def _process_camera_frame(self, camera):
        """处理单个摄像头的帧"""
        if not camera.connected:
            return
        
        frame = camera.get_frame()
        if frame is None:
            return
        
        enabled_algos = self._get_camera_enabled_algorithms(camera)
        self._log_frame_info(camera, enabled_algos)
        
        if not enabled_algos:
            return
        
        all_results = self.algorithm_manager.process_frame(frame, enabled_algos)
        self._log_detection_results(camera, all_results)
        
        passed_alerts = self.quality_controller.process(
            all_results, frame, camera.source
        )
        
        if passed_alerts:
            logging.info(f"[AI分析] {camera.name} 产生 {len(passed_alerts)} 条告警")
            self._handle_alerts(passed_alerts, camera.source, frame)
        
        with self.results_lock:
            self.results[camera.source] = passed_alerts
    
    def _log_frame_info(self, camera, enabled_algos):
        """记录帧处理信息"""
        if self.frame_counter % 100 == 0:
            logging.info(f"[AI分析] 帧#{self.frame_counter} | 摄像头: {camera.name} | 启用算法: {enabled_algos}")
        
        if not enabled_algos and self.frame_counter % 200 == 0:
            logging.warning(f"[AI分析] 摄像头 {camera.name} 未启用任何算法，跳过分析")
    
    def _log_detection_results(self, camera, all_results):
        """记录检测结果"""
        if all_results and self.frame_counter % 50 == 0:
            logging.debug(f"[AI分析] {camera.name} 检测到 {len(all_results)} 个结果")
    
    def _get_camera_enabled_algorithms(self, camera) -> List[int]:
        """获取摄像头启用的算法"""
        if hasattr(camera, 'settings') and camera.settings:
            if not camera.settings.get('ai_enabled', True):
                return []
            if 'enabled_algorithms' in camera.settings:
                enabled = camera.settings['enabled_algorithms']
                return enabled if enabled else self.enabled_algorithms
        return self.enabled_algorithms
    
    def _handle_alerts(self, alerts: List[AlgorithmResult], source: str, frame: np.ndarray):
        """处理告警"""
        for alert in alerts:
            alert_data = {
                'algorithm_id': alert.algorithm_id,
                'algorithm_name': alert.algorithm_name,
                'category': alert.category.name if hasattr(alert.category, 'name') else str(alert.category),
                'camera_source': source,
                'confidence': float(alert.confidence) if alert.confidence else 0,
                'bbox': str(alert.bounding_box) if alert.bounding_box else None,
                'extra_data': alert.extra_data if alert.extra_data else None,
                'timestamp': time.time()
            }
            
            try:
                self.alert_db.add_alert(
                    algorithm_id=alert_data['algorithm_id'],
                    algorithm_name=alert_data['algorithm_name'],
                    category=alert_data['category'],
                    camera_source=source,
                    confidence=alert_data['confidence'],
                    bbox=alert_data['bbox'],
                    extra_data=str(alert_data['extra_data']) if alert_data['extra_data'] else None
                )
            except Exception as e:
                logging.error(f"[告警数据库] 写入失败: {e}")
            
            # WebSocket 实时广播告警
            if hasattr(self, '_web_server') and self._web_server:
                try:
                    self._web_server.broadcast_alert(alert_data)
                except Exception as e:
                    logging.debug(f"[WebSocket] 广播失败: {e}")
            
            if self.alert_pusher and self.alert_pusher.is_available():
                try:
                    self.alert_pusher.push_alert(alert_data)
                except Exception as e:
                    logging.error(f"[告警推送] 推送失败: {e}")
            
            if self.llm_engine and self.llm_engine.is_available():
                try:
                    summary = self.llm_engine.summarize_alert({
                        'algorithm_name': alert.algorithm_name,
                        'camera_source': source,
                        'confidence': alert.confidence,
                    })
                    if summary:
                        self.alert_db.add_alert_summary(summary)
                except Exception as e:
                    logging.error(f"[LLM摘要] 生成失败: {e}")
    
    def get_results(self, camera_source: str = None) -> List[AlgorithmResult]:
        """获取分析结果"""
        with self.results_lock:
            if camera_source:
                return self.results.get(camera_source, [])
            return self.results
    
    def get_all_algorithms_info(self) -> List[Dict[str, Any]]:
        """获取所有算法信息"""
        return self.algorithm_manager.get_all_algorithm_info()
    
    def enable_algorithm(self, algo_id: int, enabled: bool = True):
        """启用/禁用算法"""
        self.algorithm_manager.enable_algorithm(algo_id, enabled)
    
    def get_alerts(self, limit: int = 100, offset: int = 0,
                   camera_source: str = None, algorithm_id: int = None,
                   since: float = None, date: str = None, alert_type: str = None) -> List[Dict[str, Any]]:
        """获取告警记录"""
        return self.alert_db.get_alerts(limit, offset, camera_source, algorithm_id, since, date, alert_type)
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """获取告警统计"""
        today = self.alert_db.get_today_stats()
        month = self.alert_db.get_month_stats()
        return {
            'todayAlerts': today.get('todayAlerts', 0),
            'monthAlerts': month,
            'byCategory': today.get('byCategory', {}),
        }
    
    def get_quality_stats(self) -> Dict[str, Any]:
        """获取质量控制统计"""
        return self.quality_controller.get_stats()


AIAnalyzer = SimpleAIAnalyzer
EnhancedAIAnalyzer = SimpleAIAnalyzer
