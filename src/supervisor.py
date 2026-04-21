#!/usr/bin/env python3
"""
御史台 - 结果质量验收模块
职能：终审输出结果，判定成功/失败，失败则打回对应部门重做
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque

from algorithms.algorithm_base import AlgorithmResult, AlgorithmCategory


@dataclass
class VerificationResult:
    """验收结果"""
    audit_results: List[Any]          # 门下省审核结果
    final_output: List[AlgorithmResult]  # 最终输出（通过验收的）
    retry_queue: List[Dict[str, Any]]    # 打回重做的任务
    quality_score: float              # 本轮质量评分 (0-1)
    verdict: str                      # "pass" | "partial" | "fail"
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'verdict': self.verdict,
            'quality_score': self.quality_score,
            'total_audited': len(self.audit_results),
            'passed_final': len(self.final_output),
            'retry_count': len(self.retry_queue),
            'stats': self.stats
        }


@dataclass
class RetryTask:
    """打回重做的任务"""
    algorithm_id: int
    department: str          # 打回哪个部门
    reason: str              # 打回原因
    frame_context: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)


class Supervisor:
    """
    御史台 - 质量验收与最终裁决

    工作流程：
    1. 接收门下省审核后的结果
    2. 执行最终质量评估
    3. 决策：通过 / 部分通过 / 打回重做
    4. 打回重做的任务进入重试队列
    5. 跟踪每个部门的处理质量
    """

    DEPARTMENT_NAMES = {
        'minbu': '户部',
        'libu': '吏部',
        'libu_edu': '礼部',
        'bingbu': '兵部',
        'gongbu': '工部',
        'xingbu': '刑部',
        'unknown': '未知'
    }

    # 各部门质量基准（最低可接受分）
    DEPARTMENT_THRESHOLDS = {
        'minbu': 0.7,
        'libu': 0.65,
        'libu_edu': 0.7,
        'bingbu': 0.6,
        'gongbu': 0.55,
        'xingbu': 0.65,
        'unknown': 0.5,
    }

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.running = False

        # 重试队列
        self.retry_queue: deque = deque(maxlen=200)

        # 每个算法的重试计数（用于判断是否应该放弃）
        self.retry_counts: Dict[int, int] = defaultdict(int)
        self.max_retries_per_algo = self.config.get('max_retries_per_algo', 3)

        # 质量历史（滑动窗口）
        self.quality_history: deque = deque(maxlen=100)

        # 部门级质量追踪
        self.dept_quality: Dict[str, List[float]] = defaultdict(list)

        # 告警聚合配置
        self.alert_aggregation = self.config.get('alert_aggregation', True)
        self.aggregation_window = self.config.get('aggregation_window', 5)
        self.recent_alerts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # 统计
        self.stats = {
            'total_frames': 0,
            'passed_frames': 0,
            'partial_frames': 0,
            'failed_frames': 0,
            'total_retries': 0,
            'abandoned_tasks': 0,
            'aggregated_alerts': 0,
        }

    def verify(self, audit_results: List[Any], frame: Any = None,
              camera_id: str = "default") -> VerificationResult:
        """
        御史台终审
        输入：门下省审核后的结果列表
        输出：验收结果（通过/打回）
        """
        self.stats['total_frames'] += 1

        if not audit_results:
            return VerificationResult(
                audit_results=[],
                final_output=[],
                retry_queue=[],
                quality_score=0.0,
                verdict="pass",
                stats=self._build_stats()
            )

        # 分离通过和驳回的结果
        passed = [a for a in audit_results if a.passed]
        rejected = [a for a in audit_results if not a.passed]

        # ===== 验收决策 =====
        total_input = len(audit_results)
        passed_count = len(passed)

        # 计算本轮质量分
        quality_score = passed_count / total_input if total_input > 0 else 1.0

        # 统计各部门情况
        dept_stats = self._count_by_department(audit_results)
        self._update_quality_history(quality_score, dept_stats)

        # ===== 决策逻辑 =====
        verdict, final_output, retry_tasks = self._make_verdict(
            passed, rejected, audit_results, quality_score, dept_stats
        )

        # ===== 告警聚合 =====
        if self.alert_aggregation and final_output:
            final_output = self._aggregate_alerts(final_output, camera_id)

        # 更新统计
        if verdict == "pass":
            self.stats['passed_frames'] += 1
        elif verdict == "partial":
            self.stats['partial_frames'] += 1
        else:
            self.stats['failed_frames'] += 1

        # 处理打回任务
        for task in retry_tasks:
            self._enqueue_retry(task)

        self.stats['total_retries'] += len(retry_tasks)

        return VerificationResult(
            audit_results=audit_results,
            final_output=final_output,
            retry_queue=list(self.retry_queue),
            quality_score=quality_score,
            verdict=verdict,
            stats=self._build_stats()
        )

    def _make_verdict(self, passed: List[Any], rejected: List[Any],
                      all_audit: List[Any], quality_score: float,
                      dept_stats: Dict[str, Dict]) -> Tuple[str, List, List]:
        """
        御史台做最终裁决

        裁决规则：
        1. 质量分 >= 0.8 → pass（直接输出）
        2. 质量分 0.5-0.8 → partial（输出通过结果，驳回结果打回重做）
        3. 质量分 < 0.5 → fail（所有结果打回重做）
        4. 如果某部门质量分低于阈值，该部门所有结果打回
        5. 超过最大重试次数的结果，放弃并记录
        """

        retry_tasks: List[Dict[str, Any]] = []
        final_output: List[AlgorithmResult] = []

        # 检查各部门质量
        low_quality_depts = set()
        for dept, stats in dept_stats.items():
            if stats['count'] > 0:
                dept_quality = stats['passed'] / stats['count']
                threshold = self.DEPARTMENT_THRESHOLDS.get(dept, 0.5)
                if dept_quality < threshold:
                    low_quality_depts.add(dept)

        # 裁决
        if quality_score >= 0.8:
            verdict = "pass"
            for a in passed:
                final_output.append(self._build_final_result(a))
            # 高质量时，驳回的结果也放行（宽容处理）
            for a in rejected:
                a.original.confidence = a.corrected_confidence if a.corrected_confidence else a.original.confidence
                final_output.append(a.original)

        elif quality_score >= 0.5:
            verdict = "partial"
            # 通过的输出
            for a in passed:
                final_output.append(self._build_final_result(a))
            # 驳回的打回重做
            for a in rejected:
                if a.needs_retry:
                    task = self._build_retry_task(a, dept_stats)
                    retry_tasks.append(task)
                else:
                    # 不需要重试的，也记录但不输出
                    pass

        else:
            verdict = "fail"
            for a in passed:
                if a.original.confidence >= 0.7:
                    final_output.append(self._build_final_result(a))
            for a in all_audit:
                if a.needs_retry and a not in passed:
                    task = self._build_retry_task(a, dept_stats)
                    retry_tasks.append(task)

        return verdict, final_output, retry_tasks

    def _build_final_result(self, audit_result: Any) -> AlgorithmResult:
        """构建最终输出结果（带上修正后的置信度）"""
        result = audit_result.original
        if audit_result.corrected_confidence:
            result.confidence = audit_result.corrected_confidence
        return result

    def _build_retry_task(self, audit_result: Any,
                          dept_stats: Dict[str, Dict]) -> Dict[str, Any]:
        """构建打回任务"""
        algo_id = audit_result.original.algorithm_id
        dept = self._get_department(algo_id)

        self.retry_counts[algo_id] += 1

        return {
            'algorithm_id': algo_id,
            'department': dept,
            'department_name': self.DEPARTMENT_NAMES.get(dept, dept),
            'reason': audit_result.rejected_reason,
            'retry_count': self.retry_counts[algo_id],
            'max_retries': self.max_retries_per_algo,
            'original_confidence': audit_result.original.confidence,
            'corrected_confidence': audit_result.corrected_confidence,
            'abandoned': self.retry_counts[algo_id] >= self.max_retries_per_algo
        }

    def _enqueue_retry(self, task: Dict[str, Any]):
        """将打回任务加入重试队列"""
        if task.get('abandoned'):
            self.stats['abandoned_tasks'] += 1
            logging.warning(f"[御史台] 任务已放弃: 算法{task['algorithm_id']} "
                           f"({task['department_name']}) 超过最大重试次数")
            return

        retry_task = RetryTask(
            algorithm_id=task['algorithm_id'],
            department=task['department'],
            reason=task['reason'],
            retry_count=task['retry_count'],
            max_retries=task['max_retries']
        )
        self.retry_queue.append(retry_task)

    def _count_by_department(self, audit_results: List[Any]) -> Dict[str, Dict]:
        """按部门统计审核结果"""
        stats = defaultdict(lambda: {'count': 0, 'passed': 0, 'rejected': 0})

        for a in audit_results:
            dept = self._get_department(a.original.algorithm_id)
            stats[dept]['count'] += 1
            if a.passed:
                stats[dept]['passed'] += 1
            else:
                stats[dept]['rejected'] += 1

        return dict(stats)

    def _get_department(self, algo_id: int) -> str:
        """根据算法ID找到对应部门"""
        dept_map = {
            'minbu': list(range(25, 32)) + [46],
            'libu': [1, 2, 3, 4, 5, 37, 41],
            'libu_edu': list(range(51, 65)),
            'bingbu': list(range(10, 16)) + list(range(32, 40)) + [43, 56, 66, 67],
            'gongbu': [6, 7, 8, 9, 47, 48, 49, 55, 57, 61, 68, 69],
            'xingbu': list(range(16, 25)) + [44, 45, 50, 52, 53, 54, 58, 59, 60, 62],
        }

        for dept, ids in dept_map.items():
            if algo_id in ids:
                return dept
        return 'unknown'

    def _update_quality_history(self, score: float,
                                dept_stats: Dict[str, Dict]):
        """更新质量历史"""
        self.quality_history.append(score)

        for dept, stats in dept_stats.items():
            if stats['count'] > 0:
                dq = stats['passed'] / stats['count']
                self.dept_quality[dept].append(dq)
                if len(self.dept_quality[dept]) > 50:
                    self.dept_quality[dept] = self.dept_quality[dept][-50:]

    def _aggregate_alerts(self, alerts: List[AlgorithmResult], 
                          camera_id: str) -> List[AlgorithmResult]:
        """
        告警聚合 - 合并相似告警，减少重复推送
        
        聚合规则：
        1. 同一算法、同一位置、5秒内的告警合并
        2. 合并时取最高置信度
        3. 记录聚合次数
        """
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
            recent = [a for a in recent if now - a['time'] < self.aggregation_window]
            
            if recent:
                recent[0]['count'] += 1
                recent[0]['time'] = now
                if alert.confidence > recent[0]['confidence']:
                    recent[0]['confidence'] = alert.confidence
                    recent[0]['result'] = alert
                self.recent_alerts[alert_key] = recent
                self.stats['aggregated_alerts'] += 1
            else:
                self.recent_alerts[alert_key] = [{
                    'time': now,
                    'confidence': alert.confidence,
                    'result': alert,
                    'count': 1
                }]
                aggregated.append(alert)
        
        for key in list(self.recent_alerts.keys()):
            self.recent_alerts[key] = [a for a in self.recent_alerts[key] 
                                        if now - a['time'] < self.aggregation_window * 2]
            if not self.recent_alerts[key]:
                del self.recent_alerts[key]
        
        return aggregated

    def get_retry_task(self) -> Optional[RetryTask]:
        """获取下一个需要重做的任务（尚书省会调用这个）"""
        if self.retry_queue:
            task = self.retry_queue.popleft()
            # 刷新重试计数
            self.retry_counts[task.algorithm_id] = task.retry_count
            return task
        return None

    def peek_retry_queue(self) -> List[Dict[str, Any]]:
        """窥视重试队列（不取出）"""
        return [
            {
                'algorithm_id': t.algorithm_id,
                'department': t.department,
                'department_name': self.DEPARTMENT_NAMES.get(t.department, t.department),
                'reason': t.reason,
                'retry_count': t.retry_count,
                'max_retries': t.max_retries,
                'waiting_time': f"{(time.time() - t.created_at):.1f}s"
            }
            for t in self.retry_queue
        ]

    def get_overall_quality(self) -> Dict[str, Any]:
        """获取整体质量报告"""
        if not self.quality_history:
            return {'score': 0, 'trend': 'unknown', 'dept_quality': {}}

        scores = list(self.quality_history)
        avg_score = sum(scores) / len(scores)

        # 计算趋势（最近10帧 vs 前10帧）
        if len(scores) >= 20:
            recent = sum(scores[-10:]) / 10
            previous = sum(scores[-20:-10]) / 10
            trend = 'improving' if recent > previous else 'declining' if recent < previous else 'stable'
        else:
            trend = 'insufficient_data'

        # 各部门质量
        dept_quality_report = {}
        for dept, scores in self.dept_quality.items():
            if scores:
                dept_quality_report[self.DEPARTMENT_NAMES.get(dept, dept)] = {
                    'avg': round(sum(scores) / len(scores), 3),
                    'recent': round(sum(scores[-5:]) / min(5, len(scores)), 3),
                    'samples': len(scores)
                }

        return {
            'score': round(avg_score, 3),
            'trend': trend,
            'frames_processed': self.stats['total_frames'],
            'pass_rate': round(self.stats['passed_frames'] / max(1, self.stats['total_frames']), 3),
            'dept_quality': dept_quality_report
        }

    def _build_stats(self) -> Dict[str, Any]:
        """构建统计信息"""
        return {
            **self.stats,
            'retry_queue_size': len(self.retry_queue),
            'quality_trend': self.get_overall_quality()
        }

    def reset_stats(self):
        """重置统计"""
        self.stats = {
            'total_frames': 0,
            'passed_frames': 0,
            'partial_frames': 0,
            'failed_frames': 0,
            'total_retries': 0,
            'abandoned_tasks': 0,
        }
        self.quality_history.clear()
        self.dept_quality.clear()
        self.retry_counts.clear()
