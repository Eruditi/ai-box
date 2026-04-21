#!/usr/bin/env python3
"""
AI分析模块 - 三省六部完整流程
太子(消息分拣) → 中书省(起草) → 门下省(审核) → 尚书省(派发) → 六部(执行) → 御史台(验收)
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
from gate_reviewer import GateReviewer, AuditResult
from supervisor import Supervisor, VerificationResult
from secretariat import Secretariat, DispatchTask
from alert_database import get_alert_db
from llm_engine import LLMEngine
from alert_pusher import AlertPusher


class EnhancedAIAnalyzer:
    """
    三省六部AI分析器

    流程：
    1. 太子（分拣）     - 判断帧来源，分配处理策略
    2. 中书省（起草）    - 调用算法，生成原始结果
    3. 门下省（审核）    - 过滤误报，逻辑校验，置信度修正
    4. 尚书省（派发）    - 将打回任务重新派发到对应部门
    5. 六部（执行）      - 各部门算法实际执行（algorithm_manager）
    6. 御史台（验收）    - 终审质量，决策输出/打回
    """

    def __init__(self, config):
        self.config = config
        self.running = False
        self.thread = None
        self.results: Dict[str, List[AlgorithmResult]] = {}
        self.results_lock = threading.Lock()
        self.camera_manager = None

        # ===== 三省六部初始化 =====
        # 中书省 - 算法调度
        self.algorithm_manager = AlgorithmManager(config)
        self.enabled_algorithms = self._load_enabled_algorithms()

        # 门下省 - 结果审核
        self.gate_reviewer = GateReviewer(config.get('audit', {}))

        # 御史台 - 质量验收
        self.supervisor = Supervisor(config.get('supervisor', {}))

        # 尚书省 - 任务派发
        self.secretariat = Secretariat(config.get('secretariat', {}))

        # 注册六部处理器到尚书省
        self._register_department_handlers()

        # 帧计数（用于场景基线更新）
        self.frame_counter = 0

        # 告警数据库
        self.alert_db = get_alert_db()
        
        # LLM引擎
        self.llm_engine = LLMEngine(config)
        
        # 告警推送
        self.alert_pusher = AlertPusher(config)

    def _register_department_handlers(self):
        """注册六部处理器到尚书省"""
        def minbu_handler(frame, algo_ids, context):
            return self.algorithm_manager.process_frame(frame, algo_ids, context)

        def libu_handler(frame, algo_ids, context):
            return self.algorithm_manager.process_frame(frame, algo_ids, context)

        def libu_edu_handler(frame, algo_ids, context):
            return self.algorithm_manager.process_frame(frame, algo_ids, context)

        def bingbu_handler(frame, algo_ids, context):
            return self.algorithm_manager.process_frame(frame, algo_ids, context)

        def gongbu_handler(frame, algo_ids, context):
            return self.algorithm_manager.process_frame(frame, algo_ids, context)

        def xingbu_handler(frame, algo_ids, context):
            return self.algorithm_manager.process_frame(frame, algo_ids, context)

        self.secretariat.register_department_handler('minbu', minbu_handler)
        self.secretariat.register_department_handler('libu', libu_handler)
        self.secretariat.register_department_handler('libu_edu', libu_edu_handler)
        self.secretariat.register_department_handler('bingbu', bingbu_handler)
        self.secretariat.register_department_handler('gongbu', gongbu_handler)
        self.secretariat.register_department_handler('xingbu', xingbu_handler)

        logging.info("[AI分析器] 三省六部初始化完成")

    def _load_enabled_algorithms(self) -> List[int]:
        enabled = self.config.get('ai.enabled_algorithms', [])
        if enabled is None or len(enabled) == 0:
            return []
        return enabled
    
    def _get_camera_enabled_algorithms(self, camera) -> List[int]:
        """获取摄像头启用的算法列表"""
        if hasattr(camera, 'settings') and camera.settings:
            if not camera.settings.get('ai_enabled', True):
                logging.info(f"[摄像头算法] {camera.source} AI已禁用")
                return []
            
            if 'enabled_algorithms' in camera.settings:
                enabled = camera.settings['enabled_algorithms']
                logging.info(f"[摄像头算法] {camera.source} 启用算法: {enabled}")
                return enabled if enabled else []
        
        logging.info(f"[摄像头算法] {camera.source} 使用全局默认: {self.enabled_algorithms}")
        return self.enabled_algorithms

    def initialize(self) -> bool:
        return self.algorithm_manager.initialize_all()

    def _analysis_loop(self):
        """
        三省六部核心循环
        每帧按顺序执行：太子分拣 → 中书省起草 → 门下省审核 → 尚书省派发 → 六部执行 → 御史台验收
        """
        while self.running:
            try:
                if self.camera_manager:
                    cameras = self.camera_manager.get_all_cameras()
                    for camera in cameras:
                        frame = camera.get_frame()
                        if frame is None:
                            continue

                        source = self._get_camera_source(camera)
                        is_drone = self._is_drone_camera(source)
                        self.frame_counter += 1

                        camera_algos = self._get_camera_enabled_algorithms(camera)
                        
                        if not camera_algos and not is_drone:
                            continue

                        # ===== 中书省：起草并派发任务 =====
                        context = {
                            'source': source,
                            'is_drone': is_drone,
                            'enabled_algorithms': camera_algos
                        }
                        task_id = self.secretariat.receive_new_task(
                            source, frame, context
                        )

                        # ===== 尚书省：调度并执行任务 =====
                        raw_results = []
                        dispatch_tasks = self.secretariat.dispatch_batch(max_count=6)
                        
                        for task in dispatch_tasks:
                            results = self.secretariat.execute_task(task, frame, context)
                            raw_results.extend(results)

                        # ===== 门下省：审核 =====
                        audit_results, audit_stats = self.gate_reviewer.audit(
                            raw_results, frame, source
                        )

                        # ===== 御史台：验收 =====
                        verification = self.supervisor.verify(
                            audit_results, frame, source
                        )

                        # ===== 存储最终结果 =====
                        final_results = verification.final_output

                        with self.results_lock:
                            self.results[source] = final_results

                        for r in final_results:
                            if r.detected:
                                try:
                                    self.alert_db.add_alert(
                                        algorithm_id=r.algorithm_id,
                                        algorithm_name=r.algorithm_name,
                                        category=r.category.name,
                                        camera_source=source,
                                        confidence=r.confidence,
                                        bbox=str(r.bounding_box) if r.bounding_box else None,
                                        extra_data=str(r.extra_data) if r.extra_data else None
                                    )
                                except Exception as db_err:
                                    logging.error(f"[告警数据库] 写入失败: {db_err}")

                                if self.alert_pusher and self.alert_pusher.is_available():
                                    alert_data = {
                                        'algorithm_id': r.algorithm_id,
                                        'algorithm_name': r.algorithm_name,
                                        'category': r.category.name,
                                        'camera_source': source,
                                        'confidence': r.confidence,
                                        'bbox': str(r.bounding_box) if r.bounding_box else None,
                                        'extra_data': str(r.extra_data) if r.extra_data else None,
                                        'datetime': time.strftime('%Y-%m-%d %H:%M:%S')
                                    }
                                    try:
                                        self.alert_pusher.push_alert(alert_data)
                                    except Exception as push_err:
                                        logging.error(f"[告警推送] 推送失败: {push_err}")

                                if self.llm_engine and self.llm_engine.is_available():
                                    summary = self.llm_engine.summarize_alert(alert_data)
                                    try:
                                        self.alert_db.add_alert_summary(summary)
                                    except Exception as llm_err:
                                        logging.error(f"[LLM摘要] 生成失败: {llm_err}")

                        if self.frame_counter % 30 == 0 or verification.verdict == 'fail':
                            self._log_workflow_stats(source, audit_stats, verification)

            except Exception as e:
                logging.error(f"[分析循环] 异常: {e}")

            time.sleep(0.033)

    def _process_retry_queue(self, frame: np.ndarray) -> List[DispatchTask]:
        """尚书省处理御史台打回的重做任务"""
        retry_tasks = []
        while True:
            task = self.supervisor.get_retry_task()
            if task is None:
                break
            # 尚书省重新派发
            self.secretariat.receive_retry_task(task)
            # 立即取出执行
            dispatch_task = self.secretariat.get_next_task_for_department(task.department)
            if dispatch_task:
                retry_tasks.append(dispatch_task)
        return retry_tasks

    def _get_camera_source(self, camera) -> str:
        """获取摄像头标识"""
        if hasattr(camera, 'source'):
            return camera.source
        elif hasattr(camera, 'config') and hasattr(camera.config, 'source'):
            return camera.config.source
        return str(id(camera))

    def _is_drone_camera(self, source: str) -> bool:
        """太子分拣逻辑：判断是否无人机摄像头"""
        drone_indicators = ['drone', 'uav', 'air', 'sky', 'rtsp://drone', 'drone://']
        source_lower = source.lower()
        return any(ind in source_lower for ind in drone_indicators)

    def _log_workflow_stats(self, source: str, audit_stats: Dict,
                             verification: VerificationResult):
        """记录三省六部流程统计"""
        quality = self.supervisor.get_overall_quality()
        dept_names = {
            'minbu': '户部', 'libu': '吏部', 'bingbu': '兵部',
            'gongbu': '工部', 'xingbu': '刑部', 'libu_edu': '礼部'
        }
        dept_info = ', '.join(
            f"{dept_names.get(d, d)}:{count}"
            for d, count in audit_stats.get('by_department', {}).items()
        )
        reject_info = ', '.join(
            f"{r}:{c}" for r, c in audit_stats.get('reject_reasons', {}).items()
        )

        logging.info(
            f"[{source}] 三省六部统计 | "
            f"审核通过:{audit_stats.get('passed', 0)}/{audit_stats.get('total', 0)} | "
            f"部门:{dept_info or '无数据'} | "
            f"驳回原因:{reject_info or '无'} | "
            f"验收:{verification.verdict}({verification.quality_score:.2f}) | "
            f"重试队列:{len(verification.retry_queue)}"
        )

    def start(self):
        logging.info("Starting 三省六部 AI分析器...")
        self.initialize()
        self.running = True
        self.thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.thread.start()
        logging.info("三省六部 AI分析器 已启动")

    def stop(self):
        logging.info("Stopping 三省六部 AI分析器...")
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        self.algorithm_manager.release()
        logging.info("三省六部 AI分析器 已停止")

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

    # ===== 三省六部状态查询 =====

    def get_workflow_status(self) -> Dict[str, Any]:
        """获取完整的三省六部工作状态"""
        return {
            'gate_reviewer': {
                'cooldown_map_size': len(self.gate_reviewer.cooldown_map),
                'history_cameras': list(self.gate_reviewer.history.keys()),
            },
            'supervisor': {
                **self.supervisor.stats,
                'quality': self.supervisor.get_overall_quality(),
                'retry_queue_preview': self.supervisor.peek_retry_queue()[:5],
            },
            'secretariat': {
                **self.secretariat.stats,
                'queue_status': self.secretariat.get_queue_status(),
            }
        }


AIAnalyzer = EnhancedAIAnalyzer
