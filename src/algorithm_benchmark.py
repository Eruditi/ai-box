#!/usr/bin/env python3
"""
算法精度评测标准
提供统一的算法性能评估框架
"""

import os
import json
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np


@dataclass
class BoundingBox:
    """边界框"""
    x: int
    y: int
    width: int
    height: int
    label: str = ""
    confidence: float = 1.0


@dataclass
class GroundTruth:
    """标注真值"""
    image_id: str
    boxes: List[BoundingBox]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Detection:
    """检测结果"""
    image_id: str
    boxes: List[BoundingBox]
    inference_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationMetrics:
    """评估指标"""
    algorithm_id: int
    algorithm_name: str
    
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    iou_threshold: float = 0.5
    
    avg_inference_time: float = 0.0
    fps: float = 0.0
    
    total_images: int = 0
    total_gt_boxes: int = 0
    total_det_boxes: int = 0
    
    timestamp: float = field(default_factory=time.time)
    
    def calculate_metrics(self):
        """计算评估指标"""
        if self.true_positives + self.false_positives > 0:
            self.precision = self.true_positives / (self.true_positives + self.false_positives)
        
        if self.true_positives + self.false_negatives > 0:
            self.recall = self.true_positives / (self.true_positives + self.false_negatives)
        
        if self.precision + self.recall > 0:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)


@dataclass
class BenchmarkDataset:
    """基准测试数据集"""
    name: str
    description: str
    images: List[str]
    ground_truths: Dict[str, GroundTruth]
    categories: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCalculator:
    """指标计算器"""
    
    @staticmethod
    def compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
        """计算IoU"""
        x1_min, y1_min = box1.x, box1.y
        x1_max, y1_max = box1.x + box1.width, box1.y + box1.height
        
        x2_min, y2_min = box2.x, box2.y
        x2_max, y2_max = box2.x + box2.width, box2.y + box2.height
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        box1_area = box1.width * box1.height
        box2_area = box2.width * box2.height
        
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    @staticmethod
    def compute_ap(recalls: List[float], precisions: List[float]) -> float:
        """计算AP (Average Precision)"""
        recalls = [0.0] + recalls + [1.0]
        precisions = [0.0] + precisions + [0.0]
        
        for i in range(len(precisions) - 1, 0, -1):
            precisions[i-1] = max(precisions[i-1], precisions[i])
        
        ap = 0.0
        for i in range(len(recalls) - 1):
            if recalls[i+1] != recalls[i]:
                ap += (recalls[i+1] - recalls[i]) * precisions[i+1]
        
        return ap
    
    @staticmethod
    def compute_map(aps: List[float]) -> float:
        """计算mAP"""
        if not aps:
            return 0.0
        return sum(aps) / len(aps)


class AlgorithmEvaluator:
    """算法评估器"""
    
    def __init__(self, iou_threshold: float = 0.5, confidence_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.metrics_calculator = MetricsCalculator()
    
    def evaluate(self,
                algorithm_id: int,
                algorithm_name: str,
                ground_truths: Dict[str, GroundTruth],
                detections: Dict[str, Detection]) -> EvaluationMetrics:
        """评估算法"""
        metrics = EvaluationMetrics(
            algorithm_id=algorithm_id,
            algorithm_name=algorithm_name,
            iou_threshold=self.iou_threshold
        )
        
        total_inference_time = 0.0
        
        for image_id, gt in ground_truths.items():
            metrics.total_images += 1
            metrics.total_gt_boxes += len(gt.boxes)
            
            if image_id not in detections:
                metrics.false_negatives += len(gt.boxes)
                continue
            
            det = detections[image_id]
            metrics.total_det_boxes += len(det.boxes)
            total_inference_time += det.inference_time
            
            filtered_boxes = [
                box for box in det.boxes
                if box.confidence >= self.confidence_threshold
            ]
            
            gt_matched = [False] * len(gt.boxes)
            det_matched = [False] * len(filtered_boxes)
            
            for i, gt_box in enumerate(gt.boxes):
                best_iou = 0.0
                best_j = -1
                
                for j, det_box in enumerate(filtered_boxes):
                    if det_matched[j]:
                        continue
                    
                    if gt_box.label and det_box.label and gt_box.label != det_box.label:
                        continue
                    
                    iou = self.metrics_calculator.compute_iou(gt_box, det_box)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                
                if best_iou >= self.iou_threshold and best_j >= 0:
                    gt_matched[i] = True
                    det_matched[best_j] = True
                    metrics.true_positives += 1
                else:
                    metrics.false_negatives += 1
            
            for j, matched in enumerate(det_matched):
                if not matched:
                    metrics.false_positives += 1
        
        metrics.calculate_metrics()
        
        if metrics.total_images > 0:
            metrics.avg_inference_time = total_inference_time / metrics.total_images
            if metrics.avg_inference_time > 0:
                metrics.fps = 1.0 / metrics.avg_inference_time
        
        return metrics


class BenchmarkManager:
    """基准测试管理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        self.data_dir = Path(self.config.get('data_dir', 'data/benchmarks'))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.datasets: Dict[str, BenchmarkDataset] = {}
        self.evaluations: Dict[int, List[EvaluationMetrics]] = defaultdict(list)
        
        self._lock = threading.Lock()
        
        self._load_datasets()
        
        logging.info(f"[算法评测] 初始化完成，已加载 {len(self.datasets)} 个数据集")
    
    def _load_datasets(self):
        """加载数据集"""
        for dataset_dir in self.data_dir.iterdir():
            if dataset_dir.is_dir():
                config_file = dataset_dir / "dataset.json"
                if config_file.exists():
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        gt_file = dataset_dir / "ground_truth.json"
                        ground_truths = {}
                        if gt_file.exists():
                            with open(gt_file, 'r', encoding='utf-8') as f:
                                gt_data = json.load(f)
                                for img_id, gt_dict in gt_data.items():
                                    boxes = [
                                        BoundingBox(**box)
                                        for box in gt_dict.get('boxes', [])
                                    ]
                                    ground_truths[img_id] = GroundTruth(
                                        image_id=img_id,
                                        boxes=boxes,
                                        metadata=gt_dict.get('metadata', {})
                                    )
                        
                        dataset = BenchmarkDataset(
                            name=data.get('name', dataset_dir.name),
                            description=data.get('description', ''),
                            images=data.get('images', []),
                            ground_truths=ground_truths,
                            categories=data.get('categories', []),
                            metadata=data.get('metadata', {})
                        )
                        
                        self.datasets[dataset.name] = dataset
                        logging.info(f"[算法评测] 加载数据集: {dataset.name}")
                        
                    except Exception as e:
                        logging.error(f"[算法评测] 加载数据集失败 {dataset_dir}: {e}")
    
    def create_dataset(self,
                      name: str,
                      description: str,
                      image_paths: List[str],
                      annotations: Dict[str, List[Dict[str, Any]]],
                      categories: List[str] = None) -> BenchmarkDataset:
        """创建数据集"""
        dataset_dir = self.data_dir / name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        ground_truths = {}
        for img_path in image_paths:
            img_id = Path(img_path).stem
            
            boxes = []
            if img_id in annotations:
                for ann in annotations[img_id]:
                    boxes.append(BoundingBox(
                        x=ann.get('x', 0),
                        y=ann.get('y', 0),
                        width=ann.get('width', 0),
                        height=ann.get('height', 0),
                        label=ann.get('label', ''),
                        confidence=1.0
                    ))
            
            ground_truths[img_id] = GroundTruth(
                image_id=img_id,
                boxes=boxes
            )
        
        dataset = BenchmarkDataset(
            name=name,
            description=description,
            images=image_paths,
            ground_truths=ground_truths,
            categories=categories or []
        )
        
        config_data = {
            'name': name,
            'description': description,
            'images': image_paths,
            'categories': categories or [],
            'created_at': datetime.now().isoformat()
        }
        
        with open(dataset_dir / "dataset.json", 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        gt_data = {
            img_id: {
                'boxes': [asdict(box) for box in gt.boxes],
                'metadata': gt.metadata
            }
            for img_id, gt in ground_truths.items()
        }
        
        with open(dataset_dir / "ground_truth.json", 'w', encoding='utf-8') as f:
            json.dump(gt_data, f, indent=2, ensure_ascii=False)
        
        self.datasets[name] = dataset
        logging.info(f"[算法评测] 创建数据集: {name}")
        
        return dataset
    
    def run_evaluation(self,
                      algorithm_id: int,
                      algorithm_name: str,
                      dataset_name: str,
                      detection_func,
                      iou_threshold: float = 0.5,
                      confidence_threshold: float = 0.5) -> EvaluationMetrics:
        """运行评估"""
        if dataset_name not in self.datasets:
            raise ValueError(f"数据集不存在: {dataset_name}")
        
        dataset = self.datasets[dataset_name]
        
        detections = {}
        
        for image_path in dataset.images:
            img_id = Path(image_path).stem
            
            try:
                import cv2
                frame = cv2.imread(image_path)
                
                if frame is None:
                    logging.warning(f"[算法评测] 无法读取图像: {image_path}")
                    continue
                
                start_time = time.time()
                result = detection_func(frame)
                inference_time = time.time() - start_time
                
                boxes = []
                if isinstance(result, list):
                    for det in result:
                        if isinstance(det, dict):
                            boxes.append(BoundingBox(
                                x=det.get('x', 0),
                                y=det.get('y', 0),
                                width=det.get('width', 0),
                                height=det.get('height', 0),
                                label=det.get('label', ''),
                                confidence=det.get('confidence', 0.0)
                            ))
                
                detections[img_id] = Detection(
                    image_id=img_id,
                    boxes=boxes,
                    inference_time=inference_time
                )
                
            except Exception as e:
                logging.error(f"[算法评测] 检测失败 {image_path}: {e}")
        
        evaluator = AlgorithmEvaluator(iou_threshold, confidence_threshold)
        metrics = evaluator.evaluate(
            algorithm_id,
            algorithm_name,
            dataset.ground_truths,
            detections
        )
        
        with self._lock:
            self.evaluations[algorithm_id].append(metrics)
        
        self._save_evaluation(metrics, dataset_name)
        
        logging.info(f"[算法评测] 评估完成: {algorithm_name}, "
                    f"Precision={metrics.precision:.3f}, "
                    f"Recall={metrics.recall:.3f}, "
                    f"F1={metrics.f1_score:.3f}")
        
        return metrics
    
    def _save_evaluation(self, metrics: EvaluationMetrics, dataset_name: str):
        """保存评估结果"""
        eval_dir = self.data_dir / "evaluations"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{metrics.algorithm_id}_{dataset_name}_{int(metrics.timestamp)}.json"
        filepath = eval_dir / filename
        
        data = asdict(metrics)
        data['dataset'] = dataset_name
        data['evaluated_at'] = datetime.fromtimestamp(metrics.timestamp).isoformat()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def get_algorithm_history(self, algorithm_id: int) -> List[Dict[str, Any]]:
        """获取算法评估历史"""
        with self._lock:
            return [
                {
                    'algorithm_id': m.algorithm_id,
                    'algorithm_name': m.algorithm_name,
                    'precision': round(m.precision, 4),
                    'recall': round(m.recall, 4),
                    'f1_score': round(m.f1_score, 4),
                    'avg_inference_time': round(m.avg_inference_time, 4),
                    'fps': round(m.fps, 2),
                    'total_images': m.total_images,
                    'timestamp': datetime.fromtimestamp(m.timestamp).strftime('%Y-%m-%d %H:%M:%S')
                }
                for m in self.evaluations.get(algorithm_id, [])
            ]
    
    def get_leaderboard(self, metric: str = 'f1_score') -> List[Dict[str, Any]]:
        """获取排行榜"""
        leaderboard = []
        
        with self._lock:
            for algo_id, metrics_list in self.evaluations.items():
                if metrics_list:
                    latest = max(metrics_list, key=lambda m: m.timestamp)
                    leaderboard.append({
                        'algorithm_id': algo_id,
                        'algorithm_name': latest.algorithm_name,
                        'precision': round(latest.precision, 4),
                        'recall': round(latest.recall, 4),
                        'f1_score': round(latest.f1_score, 4),
                        'fps': round(latest.fps, 2),
                        'evaluated_at': datetime.fromtimestamp(latest.timestamp).strftime('%Y-%m-%d %H:%M:%S')
                    })
        
        leaderboard.sort(key=lambda x: x.get(metric, 0), reverse=True)
        
        return leaderboard
    
    def compare_algorithms(self, algorithm_ids: List[int]) -> Dict[str, Any]:
        """对比算法"""
        comparison = {
            'algorithms': [],
            'metrics_comparison': {}
        }
        
        with self._lock:
            for algo_id in algorithm_ids:
                if algo_id in self.evaluations and self.evaluations[algo_id]:
                    latest = max(self.evaluations[algo_id], key=lambda m: m.timestamp)
                    comparison['algorithms'].append({
                        'id': algo_id,
                        'name': latest.algorithm_name,
                        'metrics': asdict(latest)
                    })
        
        if comparison['algorithms']:
            metrics_names = ['precision', 'recall', 'f1_score', 'fps']
            for metric in metrics_names:
                comparison['metrics_comparison'][metric] = [
                    {
                        'algorithm_id': algo['id'],
                        'algorithm_name': algo['name'],
                        'value': algo['metrics'].get(metric, 0)
                    }
                    for algo in comparison['algorithms']
                ]
        
        return comparison


_benchmark_manager: Optional[BenchmarkManager] = None


def get_benchmark_manager(config: Dict[str, Any] = None) -> BenchmarkManager:
    """获取基准测试管理器单例"""
    global _benchmark_manager
    if _benchmark_manager is None:
        _benchmark_manager = BenchmarkManager(config)
    return _benchmark_manager
