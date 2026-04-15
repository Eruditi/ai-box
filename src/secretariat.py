#!/usr/bin/env python3
"""
尚书省 - 任务派发与调度模块
职能：根据御史台的打回任务，将任务重新派发到对应部门执行

调度策略：
1. 优先级调度 - 高优先级任务优先执行
2. 负载均衡 - 根据部门队列长度动态分配
3. 优先级抢占 - 紧急任务可以插队
4. 任务超时 - 防止任务长时间占用
"""

import time
import logging
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

from algorithms.algorithm_base import AlgorithmResult


class ScheduleStrategy(Enum):
    """调度策略"""
    PRIORITY = "priority"           # 纯优先级
    LOAD_BALANCE = "load_balance"   # 负载均衡
    ROUND_ROBIN = "round_robin"     # 轮询
    ADAPTIVE = "adaptive"           # 自适应（根据负载动态切换）


@dataclass
class DispatchTask:
    """派发任务"""
    task_id: str
    algorithm_ids: List[int]
    department: str
    department_name: str
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    source: str = "unknown"
    timeout: float = 5.0
    started_at: float = 0.0
    completed: bool = False
    preempted: bool = False


@dataclass
class DepartmentLoad:
    """部门负载信息"""
    department: str
    queue_length: int = 0
    avg_process_time: float = 0.1
    total_processed: int = 0
    last_process_time: float = 0.0
    load_score: float = 0.0


class Secretariat:
    """
    尚书省 - 任务派发中枢

    调度策略：
    - PRIORITY: 按优先级从高到低执行
    - LOAD_BALANCE: 优先分配给负载低的部门
    - ROUND_ROBIN: 轮询各部门
    - ADAPTIVE: 根据系统负载动态切换策略
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

    PRIORITY_LABELS = {
        0: '低',
        1: '中',
        2: '高',
        3: '紧急'
    }

    PRIORITY_THRESHOLDS = {
        3: 0.95,
        2: 0.85,
        1: 0.65,
        0: 0.0,
    }

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.running = False

        self.queues: Dict[str, deque] = {
            dept: deque(maxlen=500) for dept in list(self.DEPARTMENT_NAMES.keys())
        }
        self.queues['unknown'] = deque(maxlen=200)

        self.task_counter = 0
        self.handlers: Dict[str, Callable] = {}
        
        self.strategy = ScheduleStrategy(
            self.config.get('strategy', 'adaptive')
        )
        
        self.department_loads: Dict[str, DepartmentLoad] = {
            dept: DepartmentLoad(department=dept) 
            for dept in self.DEPARTMENT_NAMES.keys()
        }
        
        self.round_robin_index = 0
        self.preemption_enabled = self.config.get('preemption_enabled', True)
        self.max_concurrent_per_dept = self.config.get('max_concurrent_per_dept', 3)
        self.active_tasks: Dict[str, List[DispatchTask]] = {
            dept: [] for dept in self.DEPARTMENT_NAMES.keys()
        }
        
        self.lock = threading.Lock()

        self.stats = {
            'total_dispatched': 0,
            'total_preempted': 0,
            'total_timeout': 0,
            'by_department': {name: 0 for name in self.DEPARTMENT_NAMES.values()},
            'by_priority': {label: 0 for label in self.PRIORITY_LABELS.values()},
            'by_source': {'normal': 0, 'retry': 0, 'scheduled': 0},
            'strategy_usage': {s.value: 0 for s in ScheduleStrategy}
        }

    def receive_new_task(self, camera_id: str, frame: Any,
                         context: Dict[str, Any] = None) -> str:
        context = context or {}
        enabled_algos = context.get('enabled_algorithms', [])
        is_drone = context.get('is_drone', False)

        task_id = self._new_task_id()

        if is_drone:
            self._enqueue(task_id, [68, 69], 'gongbu', priority=2, source='normal')
        elif enabled_algos:
            by_dept = self._group_by_department(enabled_algos)
            for dept, algo_ids in by_dept.items():
                priority = self._calc_initial_priority(algo_ids, dept)
                self._enqueue(task_id, algo_ids, dept, priority=priority, source='normal')
        else:
            self._dispatch_all(task_id, priority=1, source='normal')

        return task_id

    def receive_retry_task(self, task: Any) -> Optional[str]:
        if not task or not task.algorithm_id:
            return None

        dept = self._get_department(task.algorithm_id)
        task_id = self._new_task_id()
        priority = self._calc_priority_from_retry(task)

        self._enqueue(
            task_id,
            [task.algorithm_id],
            dept,
            priority=priority,
            source='retry'
        )

        logging.info(f"[尚书省] 打回任务重新派发: 算法{task.algorithm_id} "
                    f"({self.DEPARTMENT_NAMES.get(dept, dept)}), "
                    f"优先级={self.PRIORITY_LABELS.get(priority, '低')}")

        return task_id

    def dispatch_next(self) -> Optional[DispatchTask]:
        with self.lock:
            self._check_timeouts()
            
            if self.strategy == ScheduleStrategy.ADAPTIVE:
                strategy = self._select_adaptive_strategy()
            else:
                strategy = self.strategy
            
            self.stats['strategy_usage'][strategy.value] += 1
            
            if strategy == ScheduleStrategy.PRIORITY:
                return self._dispatch_by_priority()
            elif strategy == ScheduleStrategy.LOAD_BALANCE:
                return self._dispatch_by_load_balance()
            elif strategy == ScheduleStrategy.ROUND_ROBIN:
                return self._dispatch_by_round_robin()
            else:
                return self._dispatch_by_priority()

    def dispatch_batch(self, max_count: int = 10) -> List[DispatchTask]:
        tasks = []
        while len(tasks) < max_count:
            task = self.dispatch_next()
            if task is None:
                break
            tasks.append(task)
        return tasks

    def get_next_task_for_department(self, department: str) -> Optional[DispatchTask]:
        queue = self.queues.get(department, deque())
        if not queue:
            return None

        best = None
        for task in queue:
            if best is None or task.priority > best.priority:
                best = task

        if best:
            queue.remove(best)
            self._record_dispatch(best)
        return best

    def register_department_handler(self, department: str,
                                    handler: Callable[[Any, List[int], Dict], List[AlgorithmResult]]):
        self.handlers[department] = handler
        logging.info(f"[尚书省] {self.DEPARTMENT_NAMES.get(department, department)} 已注册处理器")

    def unregister_department_handler(self, department: str):
        if department in self.handlers:
            del self.handlers[department]

    def execute_task(self, task: DispatchTask, frame: Any,
                     context: Dict[str, Any] = None) -> List[AlgorithmResult]:
        handler = self.handlers.get(task.department)
        if handler is None:
            logging.warning(f"[尚书省] 部门 {task.department} 未注册处理器")
            return []

        task.started_at = time.time()
        
        with self.lock:
            self.active_tasks[task.department].append(task)
        
        if self.preemption_enabled:
            self._check_preemption(task)

        try:
            results = handler(frame, task.algorithm_ids, context or {})
            task.completed = True
            
            process_time = time.time() - task.started_at
            self._update_department_load(task.department, process_time)
            
            return results
        except Exception as e:
            logging.error(f"[尚书省] 执行任务失败: {e}")
            return []
        finally:
            with self.lock:
                if task in self.active_tasks.get(task.department, []):
                    self.active_tasks[task.department].remove(task)

    def get_queue_status(self) -> Dict[str, Any]:
        status = {}
        for dept, queue in self.queues.items():
            if len(queue) > 0:
                priorities = [t.priority for t in queue]
                oldest = min(queue, key=lambda t: t.created_at)
                load = self.department_loads.get(dept)
                status[self.DEPARTMENT_NAMES.get(dept, dept)] = {
                    'count': len(queue),
                    'max_priority': max(priorities) if priorities else 0,
                    'oldest_wait': f"{(time.time() - oldest.created_at):.1f}s",
                    'load_score': load.load_score if load else 0,
                    'avg_process_time': load.avg_process_time if load else 0
                }
        return status

    def get_load_balance_status(self) -> Dict[str, Any]:
        return {
            'strategy': self.strategy.value,
            'department_loads': {
                self.DEPARTMENT_NAMES.get(dept, dept): {
                    'queue_length': load.queue_length,
                    'load_score': round(load.load_score, 2),
                    'avg_process_time': round(load.avg_process_time, 3),
                    'total_processed': load.total_processed
                }
                for dept, load in self.department_loads.items()
            },
            'active_tasks': {
                self.DEPARTMENT_NAMES.get(dept, dept): len(tasks)
                for dept, tasks in self.active_tasks.items()
            }
        }

    def clear_queue(self, department: str = None):
        if department:
            self.queues.get(department, deque()).clear()
        else:
            for q in self.queues.values():
                q.clear()

    def cancel_task(self, task_id: str) -> bool:
        for queue in self.queues.values():
            for task in queue:
                if task.task_id == task_id:
                    queue.remove(task)
                    return True
        return False

    def set_strategy(self, strategy: str):
        try:
            self.strategy = ScheduleStrategy(strategy)
            logging.info(f"[尚书省] 调度策略切换为: {strategy}")
        except ValueError:
            logging.warning(f"[尚书省] 未知调度策略: {strategy}")

    def _enqueue(self, task_id: str, algorithm_ids: List[int], department: str,
                 priority: int = 0, source: str = "normal"):
        task = DispatchTask(
            task_id=task_id,
            algorithm_ids=algorithm_ids,
            department=department,
            department_name=self.DEPARTMENT_NAMES.get(department, department),
            priority=priority,
            source=source
        )
        queue = self.queues.get(department, deque())
        queue.append(task)
        self._update_queue_length(department)

    def _dispatch_by_priority(self) -> Optional[DispatchTask]:
        for priority in sorted(self.PRIORITY_LABELS.keys(), reverse=True):
            for dept in self.DEPARTMENT_NAMES.keys():
                queue = self.queues.get(dept)
                if queue:
                    for task in queue:
                        if task.priority == priority:
                            queue.remove(task)
                            self._record_dispatch(task)
                            return task
        return None

    def _dispatch_by_load_balance(self) -> Optional[DispatchTask]:
        best_dept = None
        best_score = float('inf')
        
        for dept in self.DEPARTMENT_NAMES.keys():
            queue = self.queues.get(dept)
            if queue and len(queue) > 0:
                load = self.department_loads.get(dept)
                score = load.load_score if load else 0
                score += len(queue) * 0.1
                score += len(self.active_tasks.get(dept, [])) * 0.5
                
                if score < best_score:
                    best_score = score
                    best_dept = dept
        
        if best_dept:
            queue = self.queues.get(best_dept)
            if queue:
                best_task = max(queue, key=lambda t: t.priority)
                queue.remove(best_task)
                self._record_dispatch(best_task)
                return best_task
        return None

    def _dispatch_by_round_robin(self) -> Optional[DispatchTask]:
        depts = list(self.DEPARTMENT_NAMES.keys())
        for _ in range(len(depts)):
            dept = depts[self.round_robin_index % len(depts)]
            self.round_robin_index += 1
            
            queue = self.queues.get(dept)
            if queue and len(queue) > 0:
                best_task = max(queue, key=lambda t: t.priority)
                queue.remove(best_task)
                self._record_dispatch(best_task)
                return best_task
        return None

    def _select_adaptive_strategy(self) -> ScheduleStrategy:
        total_queued = sum(len(q) for q in self.queues.values())
        
        if total_queued == 0:
            return ScheduleStrategy.PRIORITY
        
        load_variance = self._calculate_load_variance()
        
        if load_variance > 0.5:
            return ScheduleStrategy.LOAD_BALANCE
        elif total_queued > 50:
            return ScheduleStrategy.ROUND_ROBIN
        else:
            return ScheduleStrategy.PRIORITY

    def _calculate_load_variance(self) -> float:
        loads = [load.load_score for load in self.department_loads.values()]
        if not loads:
            return 0
        mean = sum(loads) / len(loads)
        if mean == 0:
            return 0
        variance = sum((l - mean) ** 2 for l in loads) / len(loads)
        return variance / (mean ** 2) if mean > 0 else 0

    def _check_preemption(self, new_task: DispatchTask):
        if new_task.priority < 3:
            return
        
        dept = new_task.department
        active = self.active_tasks.get(dept, [])
        
        for task in active:
            if task.priority < new_task.priority and not task.completed:
                task.preempted = True
                self.stats['total_preempted'] += 1
                logging.info(f"[尚书省] 任务抢占: {new_task.task_id}(优先级{new_task.priority}) "
                           f"抢占 {task.task_id}(优先级{task.priority})")

    def _check_timeouts(self):
        now = time.time()
        for dept, tasks in self.active_tasks.items():
            for task in tasks[:]:
                if task.started_at > 0 and (now - task.started_at) > task.timeout:
                    task.preempted = True
                    tasks.remove(task)
                    self.stats['total_timeout'] += 1
                    logging.warning(f"[尚书省] 任务超时: {task.task_id} ({dept})")

    def _update_department_load(self, department: str, process_time: float):
        load = self.department_loads.get(department)
        if load:
            load.total_processed += 1
            load.last_process_time = process_time
            alpha = 0.3
            load.avg_process_time = alpha * process_time + (1 - alpha) * load.avg_process_time
            queue_len = len(self.queues.get(department, []))
            active_count = len(self.active_tasks.get(department, []))
            load.load_score = load.avg_process_time * (queue_len + active_count)

    def _update_queue_length(self, department: str):
        load = self.department_loads.get(department)
        if load:
            load.queue_length = len(self.queues.get(department, []))

    def _calc_initial_priority(self, algo_ids: List[int], dept: str) -> int:
        if dept == 'gongbu':
            return 2
        elif dept in ['bingbu', 'xingbu']:
            return 1
        else:
            return 0

    def _dispatch_all(self, task_id: str, priority: int, source: str):
        all_depts = ['minbu', 'libu', 'bingbu', 'gongbu', 'xingbu']
        for dept in all_depts:
            algo_ids = self._get_department_algorithms(dept)
            if algo_ids:
                self._enqueue(task_id, algo_ids, dept, priority, source)

    def _group_by_department(self, algorithm_ids: List[int]) -> Dict[str, List[int]]:
        groups: Dict[str, List[int]] = {}
        for algo_id in algorithm_ids:
            dept = self._get_department(algo_id)
            groups.setdefault(dept, []).append(algo_id)
        return groups

    def _get_department(self, algo_id: int) -> str:
        dept_map = {
            'minbu': list(range(25, 32)) + [46, 51],
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

    def _get_department_algorithms(self, department: str) -> List[int]:
        dept_map = {
            'minbu': list(range(25, 32)) + [46, 51],
            'libu': [1, 2, 3, 4, 5, 37, 41],
            'libu_edu': list(range(51, 65)),
            'bingbu': list(range(10, 16)) + list(range(32, 40)) + [43, 56, 66, 67],
            'gongbu': [6, 7, 8, 9, 47, 48, 49, 55, 57, 61, 68, 69],
            'xingbu': list(range(16, 25)) + [44, 45, 50, 52, 53, 54, 58, 59, 60, 62],
        }
        return dept_map.get(department, [])

    def _calc_priority_from_retry(self, task: Any) -> int:
        retry_count = getattr(task, 'retry_count', 1)
        if retry_count >= 3:
            return 3
        elif retry_count == 2:
            return 2
        else:
            return 1

    def _record_dispatch(self, task: DispatchTask):
        self.stats['total_dispatched'] += 1
        self.stats['by_department'][task.department_name] = \
            self.stats['by_department'].get(task.department_name, 0) + 1
        self.stats['by_priority'][self.PRIORITY_LABELS.get(task.priority, '低')] = \
            self.stats['by_priority'].get(self.PRIORITY_LABELS.get(task.priority, '低'), 0) + 1
        self.stats['by_source'][task.source] = \
            self.stats['by_source'].get(task.source, 0) + 1

    def _new_task_id(self) -> str:
        self.task_counter += 1
        return f"task_{int(time.time()*1000)}_{self.task_counter}"
