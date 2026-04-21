#!/usr/bin/env python3
"""
门下省 - 结果审核模块
职能：校验中书省起草的结果，过滤误报，逻辑矛盾检测，智能合并
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

from algorithms.algorithm_base import AlgorithmResult, AlgorithmCategory


@dataclass
class AuditResult:
    """审核结果"""
    original: AlgorithmResult          # 原始结果
    passed: bool                       # 是否通过审核
    rejected_reason: str = ""          # 驳回原因
    corrected_confidence: float = 0.0   # 修正后的置信度
    needs_retry: bool = False          # 是否需要重做
    retry_target: str = ""             # 重试目标部门

    def to_dict(self) -> Dict[str, Any]:
        return {
            'original': self.original.to_dict(),
            'passed': self.passed,
            'rejected_reason': self.rejected_reason,
            'corrected_confidence': self.corrected_confidence,
            'needs_retry': self.needs_retry,
            'retry_target': self.retry_target
        }


class GateReviewer:
    """门下省 - 审核所有算法结果"""

    # 部门映射：算法ID范围 → 对应六部
    DEPARTMENT_MAP = {
        # 户部 - 人脸识别、结构化分析
        'minbu': list(range(25, 32)) + [46, 51],  # 人脸/人形/车牌/学生统计
        # 吏部 - 人员违规
        'libu': [1, 2, 3, 4, 5, 37, 41],           # 安全帽/口罩/工作服/安全带/反光衣
        # 礼部 - 课堂分析
        'libu_edu': list(range(51, 65)),           # 课堂分析系列
        # 兵部 - 周界告警
        'bingbu': list(range(10, 16)) + list(range(32, 40)) + [43, 56, 66, 67],
        # 工部 - 环境异常
        'gongbu': [6, 7, 8, 9, 47, 48, 49, 55, 57, 61, 68, 69],
        # 刑部 - 行为告警
        'xingbu': list(range(16, 25)) + [44, 45, 50, 52, 53, 54, 58, 59, 60, 62],
    }

    # 各部门基础误报率（经验值，用于置信度修正）
    DEPARTMENT_FPR = {
        'minbu': 0.15,
        'libu': 0.25,
        'libu_edu': 0.20,
        'bingbu': 0.30,
        'gongbu': 0.35,
        'xingbu': 0.28,
    }

    # 置信度修正系数（基于历史误报率）
    CONFIDENCE_BOOST = {
        'minbu': 1.0,
        'libu': 0.9,
        'libu_edu': 0.95,
        'bingbu': 0.85,
        'gongbu': 0.8,
        'xingbu': 0.88,
    }

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.cooldown_map: Dict[str, float] = {}  # "algo_id:camera_id:location_key" → 冷却截止时间
        self.cooldown_seconds = self.config.get('cooldown_seconds', 60)
        self.min_confidence = self.config.get('min_confidence', 0.65)
        self.consecutive_frames = self.config.get('consecutive_frames', 2)
        self.dynamic_cooldown = self.config.get('dynamic_cooldown', True)
        self.history: Dict[str, List[AlgorithmResult]] = defaultdict(list)
        self.max_history = 30
        self._scene_baseline = None
        
        self.pending_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_frequency: Dict[str, List[float]] = defaultdict(list)
        self.alert_count_history: Dict[str, int] = defaultdict(int)

    def audit(self, results: List[AlgorithmResult], frame: np.ndarray = None,
              camera_id: str = "default") -> Tuple[List[AuditResult], Dict[str, Any]]:
        """
        审核一帧的所有算法结果
        返回：(通过的审核结果列表, 审核统计)
        """
        audit_results: List[AuditResult] = []
        stats = {
            'total': len(results),
            'passed': 0,
            'rejected': 0,
            'retry': 0,
            'by_department': defaultdict(int),
            'reject_reasons': defaultdict(int)
        }

        # 1. 场景基线更新（每10帧更新一次）
        if frame is not None:
            self._update_baseline(frame, camera_id)

        # 2. 逻辑矛盾检测（先跑，得到矛盾关系）
        contradictions = self._detect_contradictions(results, camera_id)

        # 3. 逐条审核
        for result in results:
            audit = self._audit_single(result, results, contradictions, frame, camera_id)
            audit_results.append(audit)

            dept = self._get_department(result.algorithm_id)
            stats['by_department'][dept] += 1

            if audit.passed:
                stats['passed'] += 1
            else:
                stats['rejected'] += 1
                stats['reject_reasons'][audit.rejected_reason] += 1
                if audit.needs_retry:
                    stats['retry'] += 1

        # 4. 去重：合并同帧同区域的重复检测
        audit_results = self._deduplicate(audit_results)

        # 5. 更新历史
        for r in audit_results:
            if r.passed:
                self.history[camera_id].append(r.original)
        if len(self.history[camera_id]) > self.max_history:
            self.history[camera_id] = self.history[camera_id][-self.max_history:]

        return audit_results, stats

    def _audit_single(self, result: AlgorithmResult, all_results: List[AlgorithmResult],
                      contradictions: Dict[int, List[int]], frame: np.ndarray = None,
                      camera_id: str = "default") -> AuditResult:
        """审核单条结果"""
        dept = self._get_department(result.algorithm_id)
        cooldown_key = self._make_cooldown_key(result, camera_id)

        # ===== 规则1：冷却期检查 =====
        cooldown_until = self.cooldown_map.get(cooldown_key, 0)
        if time.time() < cooldown_until:
            return AuditResult(
                original=result,
                passed=False,
                rejected_reason="cooldown",
                needs_retry=False,
                retry_target=""
            )

        # ===== 规则2：基础置信度过滤 =====
        base_threshold = self.min_confidence
        if result.confidence < base_threshold:
            return AuditResult(
                original=result,
                passed=False,
                rejected_reason="low_confidence",
                needs_retry=False,
                retry_target=""
            )

        # ===== 规则3：逻辑矛盾过滤 =====
        if result.algorithm_id in contradictions:
            conflicting_ids = contradictions[result.algorithm_id]
            for other in all_results:
                if other.algorithm_id in conflicting_ids and other.detected:
                    # 矛盾存在，降低置信度后重新评估
                    adjusted_conf = result.confidence * 0.5
                    if adjusted_conf < base_threshold:
                        return AuditResult(
                            original=result,
                            passed=False,
                            rejected_reason="contradiction",
                            needs_retry=True,
                            retry_target=dept
                        )

        # ===== 规则4：场景异常检测（火焰/烟雾误报） =====
        if dept == 'gongbu' and result.detected and frame is not None:
            scene_check = self._scene_sanity_check(result, frame)
            if not scene_check['passed']:
                # 场景原因误报，降低置信度
                adjusted = result.confidence * scene_check['factor']
                if adjusted < base_threshold:
                    return AuditResult(
                        original=result,
                        passed=False,
                        rejected_reason=f"scene_{scene_check['reason']}",
                        needs_retry=True,
                        retry_target=dept
                    )

        # ===== 规则5：历史一致性检查 =====
        if result.detected:
            consistency = self._check_historical_consistency(result, camera_id)
            if not consistency['consistent']:
                corrected = result.confidence * consistency['factor']
                return AuditResult(
                    original=result,
                    passed=corrected >= base_threshold,
                    rejected_reason="low_historical_consistency" if corrected < base_threshold else "",
                    corrected_confidence=corrected,
                    needs_retry=False
                )

        # ===== 规则6：连续帧确认机制 =====
        if self.consecutive_frames > 1 and result.detected:
            confirm_key = f"{result.algorithm_id}:{camera_id}"
            if not self._check_consecutive_frames(confirm_key, result):
                return AuditResult(
                    original=result,
                    passed=False,
                    rejected_reason="not_confirmed",
                    needs_retry=False,
                    retry_target=""
                )

        # ===== 通过审核 =====
        corrected = result.confidence * self.CONFIDENCE_BOOST.get(dept, 0.85)

        cooldown_dur = self._calc_cooldown_dynamic(result.confidence, dept, camera_id)
        self.cooldown_map[cooldown_key] = time.time() + cooldown_dur

        return AuditResult(
            original=result,
            passed=True,
            corrected_confidence=corrected
        )

    def _detect_contradictions(self, results: List[AlgorithmResult],
                                camera_id: str) -> Dict[int, List[int]]:
        """
        检测逻辑矛盾
        例如：人脸检测(25)+活体检测(46)同时无结果 → 矛盾
              车牌识别(29)+车牌对比(31)同时有结果 → 人脸和人形矛盾
        """
        detected_ids = {r.algorithm_id for r in results if r.detected}

        contradictions: Dict[int, List[int]] = {}

        # 矛盾规则1：人脸检测 vs 人形检测（互斥）
        # 如果同时检测到人脸和人形，且人脸框和人形框高度差>50%，可能是误报
        face_results = [r for r in results if r.algorithm_id == 25 and r.detected]
        human_results = [r for r in results if r.algorithm_id == 26 and r.detected]
        if face_results and human_results:
            for f in face_results:
                contradictions.setdefault(f.algorithm_id, []).append(26)
            for h in human_results:
                contradictions.setdefault(h.algorithm_id, []).append(25)

        # 矛盾规则2：火焰(6)+烟雾(7)只有一个时，降权
        # 矛盾规则3：入侵(14)+越界(15)同时触发 → 认为是同一次事件，只保留一个
        if 14 in detected_ids and 15 in detected_ids:
            # 保留14（入侵），去掉15的优先级
            if 15 in contradictions:
                pass  # 不重复

        # 矛盾规则4：睡岗(21)+离岗(22)互斥
        if 21 in detected_ids and 22 in detected_ids:
            contradictions.setdefault(21, []).append(22)
            contradictions.setdefault(22, []).append(21)

        # 矛盾规则5：人脸识别(30)和车牌识别(29)不应该在同类区域冲突
        # （这里简化处理，实际需要框的IOU判断）

        return contradictions

    def _scene_sanity_check(self, result: AlgorithmResult, frame: np.ndarray) -> Dict[str, Any]:
        """
        场景 sanity check
        检测环境类算法是否因为场景原因误报
        """
        h, w = frame.shape[:2]
        brightness = np.mean(frame)

        if result.algorithm_id == 6:  # 火焰检测
            # 明亮场景（可能是阳光直射）容易误报
            if brightness > 180:
                # 检查画面中是否有橙色/红色区域面积过大（可能是天空/夕阳）
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                red_mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
                red_mask2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
                red_mask = cv2.bitwise_or(red_mask1, red_mask2)
                red_ratio = cv2.countNonZero(red_mask) / (h * w)
                if red_ratio > 0.4:  # 红色区域超过40%，很可能是夕阳/天空
                    return {'passed': False, 'reason': 'sunset_sky', 'factor': 0.3}

        elif result.algorithm_id == 7:  # 烟雾检测
            # 雾气重的场景容易误报
            if brightness > 200:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                contrast = np.std(gray)
                if contrast < 20:  # 低对比度 = 雾/烟
                    return {'passed': False, 'reason': 'foggy_scene', 'factor': 0.4}

        elif result.algorithm_id == 47:  # 摄像头遮挡
            # 如果画面整体偏暗，可能只是晚上，不是遮挡
            if brightness < 30:
                return {'passed': False, 'reason': 'night_scene', 'factor': 0.2}

        return {'passed': True}

    def _check_historical_consistency(self, result: AlgorithmResult,
                                       camera_id: str) -> Dict[str, Any]:
        """
        检查历史一致性
        如果某个算法在历史上频繁告警又频繁消失，说明误报率高
        """
        history = self.history.get(camera_id, [])
        if len(history) < 3:
            return {'consistent': True, 'factor': 1.0}

        # 找同类算法的历史记录
        same_algo = [r for r in history if r.algorithm_id == result.algorithm_id]
        if len(same_algo) < 2:
            return {'consistent': True, 'factor': 1.0}

        # 检查最近N帧中同类结果的状态变化次数
        recent = same_algo[-5:]
        if len(recent) >= 3:
            transitions = sum(1 for i in range(1, len(recent))
                              if recent[i].detected != recent[i-1].detected)
            # 如果在3帧内状态变了2次以上，说明不稳定
            if transitions >= 2:
                return {'consistent': False, 'factor': 0.7}

        return {'consistent': True, 'factor': 1.0}

    def _update_baseline(self, frame: np.ndarray, camera_id: str):
        """更新场景基线"""
        self._scene_baseline = {
            'brightness': np.mean(frame),
            'std': np.std(frame)
        }

    def _get_department(self, algo_id: int) -> str:
        """根据算法ID找到对应部门"""
        for dept, ids in self.DEPARTMENT_MAP.items():
            if algo_id in ids:
                return dept
        return 'unknown'

    def _calc_cooldown(self, confidence: float, dept: str) -> float:
        """计算冷却时间"""
        base = self.cooldown_seconds
        conf_factor = max(0.5, 1.5 - confidence)
        dept_factor = 1.0 / self.CONFIDENCE_BOOST.get(dept, 0.85)
        return base * conf_factor * (dept_factor / 2)
    
    def _calc_cooldown_dynamic(self, confidence: float, dept: str, camera_id: str) -> float:
        """动态计算冷却时间 - 基于告警频率调整"""
        base = self.cooldown_seconds
        
        conf_factor = max(0.5, 1.5 - confidence)
        dept_factor = 1.0 / self.CONFIDENCE_BOOST.get(dept, 0.85)
        
        if self.dynamic_cooldown:
            freq_key = f"{dept}:{camera_id}"
            now = time.time()
            
            recent_alerts = [t for t in self.alert_frequency[freq_key] if now - t < 300]
            self.alert_frequency[freq_key] = recent_alerts
            
            if len(recent_alerts) > 10:
                freq_factor = 2.0
            elif len(recent_alerts) > 5:
                freq_factor = 1.5
            elif len(recent_alerts) > 2:
                freq_factor = 1.2
            else:
                freq_factor = 1.0
            
            self.alert_frequency[freq_key].append(now)
        else:
            freq_factor = 1.0
        
        return base * conf_factor * (dept_factor / 2) * freq_factor
    
    def _check_consecutive_frames(self, key: str, result: AlgorithmResult) -> bool:
        """检查连续帧确认机制"""
        now = time.time()
        
        if key not in self.pending_alerts:
            self.pending_alerts[key] = {
                'count': 1,
                'first_seen': now,
                'last_seen': now,
                'bbox': result.bounding_box,
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
        pending['bbox'] = result.bounding_box
        
        if pending['count'] >= self.consecutive_frames:
            avg_confidence = pending['confidence_sum'] / pending['count']
            if avg_confidence >= self.min_confidence:
                del self.pending_alerts[key]
                return True
        
        return False
    
    def _make_cooldown_key(self, result: AlgorithmResult, camera_id: str) -> str:
        """生成冷却键：算法ID + 摄像头ID + 位置区域"""
        location_key = ""
        if result.bounding_box:
            x, y, w, h = result.bounding_box
            grid_x = int(x // 100) * 100
            grid_y = int(y // 100) * 100
            location_key = f"{grid_x}_{grid_y}"
        return f"{result.algorithm_id}:{camera_id}:{location_key}"

    def _deduplicate(self, audit_results: List[AuditResult]) -> List[AuditResult]:
        """合并同帧同区域的重复检测（IOU去重）"""
        if not audit_results:
            return []

        passed = [a for a in audit_results if a.passed]
        rejected = [a for a in audit_results if not a.passed]

        # 只对有 bounding_box 的结果做 IOU 去重
        deduplicated = []
        used_indices = set()

        for i, a in enumerate(passed):
            if a.original.bounding_box is None:
                deduplicated.append(a)
                continue

            is_duplicate = False
            for j in range(i + 1, len(passed)):
                if j in used_indices:
                    continue
                b = passed[j]
                if b.original.bounding_box is None:
                    continue
                if a.original.algorithm_id == b.original.algorithm_id:
                    continue  # 不同算法不合并
                iou = self._calc_iou(a.original.bounding_box, b.original.bounding_box)
                if iou > 0.7:  # IOU>0.7认为是重复
                    # 保留置信度高的
                    if a.corrected_confidence < b.corrected_confidence:
                        is_duplicate = True
                        break
                    else:
                        used_indices.add(j)

            if not is_duplicate:
                deduplicated.append(a)
            else:
                rejected.append(a)

        return deduplicated + rejected

    def _calc_iou(self, box1: Tuple[int, int, int, int],
                  box2: Tuple[int, int, int, int]) -> float:
        """计算两个框的IOU"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def get_department_stats(self, camera_id: str = "default") -> Dict[str, Any]:
        """获取各部门统计"""
        history = self.history.get(camera_id, [])
        stats = defaultdict(lambda: {'total': 0, 'detected': 0})

        for r in history:
            dept = self._get_department(r.algorithm_id)
            stats[dept]['total'] += 1
            if r.detected:
                stats[dept]['detected'] += 1

        return dict(stats)


import cv2
