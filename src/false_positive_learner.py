#!/usr/bin/env python3
"""
AI误报学习系统
用户标记误报后系统自动学习，提升检测精度
"""

import os
import json
import time
import logging
import threading
import hashlib
import pickle
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np


@dataclass
class FalsePositiveSample:
    """误报样本"""
    sample_id: str
    algorithm_id: int
    algorithm_name: str
    camera_source: str
    timestamp: float
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]]
    feature_hash: str
    image_path: Optional[str] = None
    reason: str = ""
    user_id: str = "system"
    verified: bool = False


@dataclass
class LearningRule:
    """学习规则"""
    rule_id: str
    algorithm_id: int
    feature_pattern: str
    confidence_adjustment: float
    filter_condition: Dict[str, Any]
    created_at: float
    hit_count: int = 0
    last_hit: float = 0.0
    effectiveness: float = 1.0


class FeatureExtractor:
    """特征提取器"""
    
    @staticmethod
    def extract_histogram_features(frame: np.ndarray, bbox: Tuple[int, int, int, int] = None) -> np.ndarray:
        """提取颜色直方图特征"""
        if bbox:
            x, y, w, h = bbox
            roi = frame[y:y+h, x:x+w]
        else:
            roi = frame
        
        if len(roi.shape) == 3:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256])
            hist = np.concatenate([h_hist.flatten(), s_hist.flatten(), v_hist.flatten()])
        else:
            hist = cv2.calcHist([roi], [0], None, [32], [0, 256]).flatten()
        
        cv2.normalize(hist, hist)
        return hist
    
    @staticmethod
    def extract_texture_features(frame: np.ndarray, bbox: Tuple[int, int, int, int] = None) -> np.ndarray:
        """提取纹理特征"""
        if bbox:
            x, y, w, h = bbox
            roi = frame[y:y+h, x:x+w]
        else:
            roi = frame
        
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        
        gray = cv2.resize(gray, (64, 64))
        
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        mag = np.sqrt(grad_x**2 + grad_y**2)
        angle = np.arctan2(grad_y, grad_x) * 180 / np.pi
        
        hist_mag = np.histogram(mag, bins=16, range=(0, 255))[0]
        hist_angle = np.histogram(angle, bins=16, range=(-180, 180))[0]
        
        features = np.concatenate([hist_mag, hist_angle])
        features = features / (np.sum(features) + 1e-7)
        
        return features
    
    @staticmethod
    def extract_spatial_features(bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]) -> np.ndarray:
        """提取空间特征"""
        if bbox is None:
            return np.zeros(6)
        
        x, y, w, h = bbox
        frame_h, frame_w = frame_shape[:2]
        
        center_x = (x + w/2) / frame_w
        center_y = (y + h/2) / frame_h
        width_ratio = w / frame_w
        height_ratio = h / frame_h
        aspect_ratio = w / (h + 1e-7)
        area_ratio = (w * h) / (frame_w * frame_h)
        
        return np.array([center_x, center_y, width_ratio, height_ratio, aspect_ratio, area_ratio])
    
    @staticmethod
    def compute_feature_hash(features: np.ndarray) -> str:
        """计算特征哈希"""
        feature_bytes = features.tobytes()
        return hashlib.md5(feature_bytes).hexdigest()


class FalsePositiveLearner:
    """误报学习器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        self.data_dir = Path(self.config.get('data_dir', 'data/false_positive_learning'))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.samples_file = self.data_dir / 'samples.json'
        self.rules_file = self.data_dir / 'rules.json'
        self.model_file = self.data_dir / 'model.pkl'
        
        self.samples: Dict[str, FalsePositiveSample] = {}
        self.rules: Dict[str, LearningRule] = {}
        self.feature_cache: Dict[str, np.ndarray] = {}
        
        self.min_samples_for_rule = self.config.get('min_samples_for_rule', 5)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.85)
        self.confidence_penalty = self.config.get('confidence_penalty', 0.3)
        
        self._lock = threading.Lock()
        
        self._load_data()
        
        logging.info(f"[误报学习] 初始化完成，已加载 {len(self.samples)} 个样本，{len(self.rules)} 条规则")
    
    def _load_data(self):
        """加载数据"""
        if self.samples_file.exists():
            try:
                with open(self.samples_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.samples = {
                        k: FalsePositiveSample(**v) 
                        for k, v in data.items()
                    }
            except Exception as e:
                logging.error(f"[误报学习] 加载样本失败: {e}")
        
        if self.rules_file.exists():
            try:
                with open(self.rules_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.rules = {
                        k: LearningRule(**v)
                        for k, v in data.items()
                    }
            except Exception as e:
                logging.error(f"[误报学习] 加载规则失败: {e}")
    
    def _save_data(self):
        """保存数据"""
        try:
            with open(self.samples_file, 'w', encoding='utf-8') as f:
                json.dump({k: asdict(v) for k, v in self.samples.items()}, f, ensure_ascii=False, indent=2)
            
            with open(self.rules_file, 'w', encoding='utf-8') as f:
                json.dump({k: asdict(v) for k, v in self.rules.items()}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"[误报学习] 保存数据失败: {e}")
    
    def mark_false_positive(self, 
                           algorithm_id: int,
                           algorithm_name: str,
                           camera_source: str,
                           confidence: float,
                           bbox: Tuple[int, int, int, int] = None,
                           frame: np.ndarray = None,
                           reason: str = "",
                           user_id: str = "system") -> str:
        """标记误报"""
        with self._lock:
            sample_id = f"fp_{int(time.time()*1000)}_{algorithm_id}"
            
            feature_hash = ""
            if frame is not None:
                features = self._extract_features(frame, bbox)
                feature_hash = FeatureExtractor.compute_feature_hash(features)
                self.feature_cache[sample_id] = features
            
            sample = FalsePositiveSample(
                sample_id=sample_id,
                algorithm_id=algorithm_id,
                algorithm_name=algorithm_name,
                camera_source=camera_source,
                timestamp=time.time(),
                confidence=confidence,
                bbox=bbox,
                feature_hash=feature_hash,
                reason=reason,
                user_id=user_id
            )
            
            self.samples[sample_id] = sample
            self._save_data()
            
            self._check_and_create_rule(algorithm_id)
            
            logging.info(f"[误报学习] 标记误报: {algorithm_name} @ {camera_source}, 原因: {reason}")
            
            return sample_id
    
    def _extract_features(self, frame: np.ndarray, bbox: Tuple[int, int, int, int] = None) -> np.ndarray:
        """提取综合特征"""
        hist_features = FeatureExtractor.extract_histogram_features(frame, bbox)
        texture_features = FeatureExtractor.extract_texture_features(frame, bbox)
        spatial_features = FeatureExtractor.extract_spatial_features(bbox, frame.shape)
        
        return np.concatenate([hist_features, texture_features, spatial_features])
    
    def _check_and_create_rule(self, algorithm_id: int):
        """检查并创建规则"""
        algo_samples = [s for s in self.samples.values() if s.algorithm_id == algorithm_id]
        
        if len(algo_samples) < self.min_samples_for_rule:
            return
        
        feature_groups: Dict[str, List[FalsePositiveSample]] = defaultdict(list)
        for sample in algo_samples:
            if sample.feature_hash:
                feature_groups[sample.feature_hash].append(sample)
        
        for feature_hash, samples in feature_groups.items():
            if len(samples) >= self.min_samples_for_rule:
                existing_rule = self._find_existing_rule(algorithm_id, feature_hash)
                if existing_rule:
                    continue
                
                rule_id = f"rule_{int(time.time()*1000)}_{algorithm_id}"
                
                cameras = list(set(s.camera_source for s in samples))
                avg_confidence = np.mean([s.confidence for s in samples])
                
                rule = LearningRule(
                    rule_id=rule_id,
                    algorithm_id=algorithm_id,
                    feature_pattern=feature_hash,
                    confidence_adjustment=self.confidence_penalty,
                    filter_condition={
                        'cameras': cameras,
                        'confidence_range': [avg_confidence - 0.1, avg_confidence + 0.1],
                        'min_samples': len(samples)
                    },
                    created_at=time.time()
                )
                
                self.rules[rule_id] = rule
                logging.info(f"[误报学习] 创建规则: {rule_id}, 算法: {algorithm_id}, 样本数: {len(samples)}")
        
        self._save_data()
    
    def _find_existing_rule(self, algorithm_id: int, feature_hash: str) -> Optional[LearningRule]:
        """查找现有规则"""
        for rule in self.rules.values():
            if rule.algorithm_id == algorithm_id and rule.feature_pattern == feature_hash:
                return rule
        return None
    
    def check_false_positive(self,
                            algorithm_id: int,
                            confidence: float,
                            bbox: Tuple[int, int, int, int] = None,
                            frame: np.ndarray = None,
                            camera_source: str = "") -> Tuple[bool, float]:
        """检查是否为误报"""
        with self._lock:
            if not self.rules:
                return False, confidence
            
            current_features = None
            if frame is not None:
                current_features = self._extract_features(frame, bbox)
            
            for rule in self.rules.values():
                if rule.algorithm_id != algorithm_id:
                    continue
                
                if camera_source not in rule.filter_condition.get('cameras', []):
                    continue
                
                conf_range = rule.filter_condition.get('confidence_range', [0, 1])
                if not (conf_range[0] <= confidence <= conf_range[1]):
                    continue
                
                if current_features is not None:
                    rule_features = self.feature_cache.get(rule.feature_pattern)
                    if rule_features is not None:
                        similarity = self._compute_similarity(current_features, rule_features)
                        if similarity > self.similarity_threshold:
                            rule.hit_count += 1
                            rule.last_hit = time.time()
                            adjusted_confidence = confidence * (1 - rule.confidence_penalty)
                            
                            logging.debug(f"[误报学习] 命中规则: {rule.rule_id}, 相似度: {similarity:.2f}, "
                                        f"置信度调整: {confidence:.2f} -> {adjusted_confidence:.2f}")
                            
                            return True, adjusted_confidence
            
            return False, confidence
    
    def _compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """计算特征相似度"""
        if features1.shape != features2.shape:
            return 0.0
        
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            algo_stats = defaultdict(lambda: {'samples': 0, 'rules': 0})
            
            for sample in self.samples.values():
                algo_stats[sample.algorithm_id]['samples'] += 1
            
            for rule in self.rules.values():
                algo_stats[rule.algorithm_id]['rules'] += 1
            
            return {
                'total_samples': len(self.samples),
                'total_rules': len(self.rules),
                'algorithm_stats': dict(algo_stats),
                'recent_samples': [
                    {
                        'id': s.sample_id,
                        'algorithm': s.algorithm_name,
                        'camera': s.camera_source,
                        'time': datetime.fromtimestamp(s.timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                        'reason': s.reason
                    }
                    for s in sorted(self.samples.values(), key=lambda x: x.timestamp, reverse=True)[:10]
                ],
                'active_rules': [
                    {
                        'id': r.rule_id,
                        'algorithm_id': r.algorithm_id,
                        'hit_count': r.hit_count,
                        'effectiveness': r.effectiveness,
                        'created': datetime.fromtimestamp(r.created_at).strftime('%Y-%m-%d %H:%M:%S')
                    }
                    for r in sorted(self.rules.values(), key=lambda x: x.hit_count, reverse=True)[:10]
                ]
            }
    
    def delete_sample(self, sample_id: str) -> bool:
        """删除样本"""
        with self._lock:
            if sample_id in self.samples:
                del self.samples[sample_id]
                if sample_id in self.feature_cache:
                    del self.feature_cache[sample_id]
                self._save_data()
                return True
            return False
    
    def delete_rule(self, rule_id: str) -> bool:
        """删除规则"""
        with self._lock:
            if rule_id in self.rules:
                del self.rules[rule_id]
                self._save_data()
                return True
            return False


import cv2

_false_positive_learner: Optional[FalsePositiveLearner] = None


def get_false_positive_learner(config: Dict[str, Any] = None) -> FalsePositiveLearner:
    """获取误报学习器单例"""
    global _false_positive_learner
    if _false_positive_learner is None:
        _false_positive_learner = FalsePositiveLearner(config)
    return _false_positive_learner
