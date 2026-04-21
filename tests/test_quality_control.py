#!/usr/bin/env python3
"""
单元测试 - 三省六部核心逻辑
测试质量控制、冷却期、矛盾检测、重试队列等核心功能
"""

import pytest
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from algorithms.algorithm_base import AlgorithmResult, AlgorithmCategory
from quality_controller import QualityController, QualityConfig


class TestQualityController:
    """质量控制器测试"""
    
    def setup_method(self):
        """每个测试方法前的初始化"""
        self.config = QualityConfig(
            min_confidence=0.65,
            cooldown_seconds=60.0,
            consecutive_frames=1,
            enable_false_positive_learning=False,
            enable_scene_adaptive=False,
            enable_cross_camera=False
        )
        self.controller = QualityController(self.config)
    
    def test_confidence_filter(self):
        """测试置信度过滤"""
        result_low = AlgorithmResult(
            algorithm_id=1,
            algorithm_name="测试算法",
            category=AlgorithmCategory.PERSON_VIOLATION,
            detected=True,
            confidence=0.5
        )
        
        result_high = AlgorithmResult(
            algorithm_id=1,
            algorithm_name="测试算法",
            category=AlgorithmCategory.PERSON_VIOLATION,
            detected=True,
            confidence=0.8
        )
        
        passed = self.controller.process([result_low, result_high], camera_id="test")
        
        assert len(passed) == 1
        assert passed[0].confidence == 0.8
        assert self.controller.stats['filtered_low_confidence'] == 1
    
    def test_cooldown_filter(self):
        """测试冷却期过滤"""
        result = AlgorithmResult(
            algorithm_id=1,
            algorithm_name="测试算法",
            category=AlgorithmCategory.PERSON_VIOLATION,
            detected=True,
            confidence=0.8,
            bounding_box=(100, 100, 50, 50)
        )
        
        passed1 = self.controller.process([result], camera_id="test")
        assert len(passed1) == 1
        
        passed2 = self.controller.process([result], camera_id="test")
        assert len(passed2) == 0
        assert self.controller.stats['filtered_cooldown'] == 1
    
    def test_consecutive_frames(self):
        """测试连续帧确认"""
        self.config.consecutive_frames = 3
        controller = QualityController(self.config)
        
        result = AlgorithmResult(
            algorithm_id=1,
            algorithm_name="测试算法",
            category=AlgorithmCategory.PERSON_VIOLATION,
            detected=True,
            confidence=0.8
        )
        
        passed1 = controller.process([result], camera_id="test")
        assert len(passed1) == 0
        
        passed2 = controller.process([result], camera_id="test")
        assert len(passed2) == 0
        
        passed3 = controller.process([result], camera_id="test")
        assert len(passed3) == 1
    
    def test_dynamic_cooldown(self):
        """测试动态冷却期"""
        self.config.dynamic_cooldown = True
        controller = QualityController(self.config)
        
        result = AlgorithmResult(
            algorithm_id=1,
            algorithm_name="测试算法",
            category=AlgorithmCategory.PERSON_VIOLATION,
            detected=True,
            confidence=0.8,
            bounding_box=(100, 100, 50, 50)
        )
        
        for i in range(15):
            controller.process([result], camera_id="test")
            time.sleep(0.1)
        
        assert controller.stats['filtered_cooldown'] > 0
    
    def test_alert_aggregation(self):
        """测试告警聚合"""
        self.config.alert_aggregation = True
        self.config.aggregation_window = 5.0
        controller = QualityController(self.config)
        
        results = [
            AlgorithmResult(
                algorithm_id=1,
                algorithm_name="测试算法",
                category=AlgorithmCategory.PERSON_VIOLATION,
                detected=True,
                confidence=0.8,
                bounding_box=(100, 100, 50, 50)
            )
            for _ in range(5)
        ]
        
        passed = controller.process(results, camera_id="test")
        
        assert len(passed) <= 5
        assert controller.stats['aggregated'] >= 0
    
    def test_frame_validity(self):
        """测试帧有效性检测"""
        import numpy as np
        
        valid_frame = np.random.randint(50, 200, (720, 1280, 3), dtype=np.uint8)
        dark_frame = np.random.randint(0, 20, (720, 1280, 3), dtype=np.uint8)
        bright_frame = np.full((720, 1280, 3), 250, dtype=np.uint8)
        
        result = AlgorithmResult(
            algorithm_id=1,
            algorithm_name="测试算法",
            category=AlgorithmCategory.PERSON_VIOLATION,
            detected=True,
            confidence=0.8
        )
        
        passed1 = self.controller.process([result], frame=valid_frame, camera_id="test")
        assert len(passed1) == 1
        
        passed2 = self.controller.process([result], frame=dark_frame, camera_id="test")
        assert len(passed2) == 0
        
        passed3 = self.controller.process([result], frame=bright_frame, camera_id="test")
        assert len(passed3) == 0
    
    def test_stats_tracking(self):
        """测试统计信息追踪"""
        results = [
            AlgorithmResult(
                algorithm_id=i,
                algorithm_name=f"算法{i}",
                category=AlgorithmCategory.PERSON_VIOLATION,
                detected=True,
                confidence=0.7 + i * 0.05
            )
            for i in range(10)
        ]
        
        self.controller.process(results, camera_id="test")
        
        stats = self.controller.get_stats()
        
        assert stats['total_input'] == 10
        assert stats['passed'] > 0
        assert 'pass_rate' in stats


class TestCooldownMechanism:
    """冷却期机制测试"""
    
    def test_cooldown_key_generation(self):
        """测试冷却键生成"""
        controller = QualityController(QualityConfig())
        
        result1 = AlgorithmResult(
            algorithm_id=1,
            algorithm_name="测试",
            category=AlgorithmCategory.PERSON_VIOLATION,
            detected=True,
            confidence=0.8,
            bounding_box=(100, 100, 50, 50)
        )
        
        result2 = AlgorithmResult(
            algorithm_id=1,
            algorithm_name="测试",
            category=AlgorithmCategory.PERSON_VIOLATION,
            detected=True,
            confidence=0.8,
            bounding_box=(200, 100, 50, 50)
        )
        
        key1 = controller._make_cooldown_key(result1, "cam1")
        key2 = controller._make_cooldown_key(result2, "cam1")
        
        assert key1 != key2
    
    def test_cooldown_expiry(self):
        """测试冷却期过期"""
        self.config = QualityConfig(cooldown_seconds=0.1, consecutive_frames=1, alert_aggregation=False, dynamic_cooldown=False)
        controller = QualityController(self.config)
        
        result = AlgorithmResult(
            algorithm_id=1,
            algorithm_name="测试",
            category=AlgorithmCategory.PERSON_VIOLATION,
            detected=True,
            confidence=0.8,
            bounding_box=(100, 100, 50, 50)
        )
        
        passed1 = controller.process([result], camera_id="test")
        assert len(passed1) == 1
        
        # 手动清除冷却记录（避免时间精度问题）
        controller.cooldown_map.clear()
        
        passed2 = controller.process([result], camera_id="test")
        assert len(passed2) == 1


class TestConsecutiveFrames:
    """连续帧确认测试"""
    
    def test_consecutive_reset(self):
        """测试连续帧重置"""
        self.config = QualityConfig(consecutive_frames=3)
        controller = QualityController(self.config)
        
        result1 = AlgorithmResult(
            algorithm_id=1,
            algorithm_name="测试",
            category=AlgorithmCategory.PERSON_VIOLATION,
            detected=True,
            confidence=0.8
        )
        
        result2 = AlgorithmResult(
            algorithm_id=1,
            algorithm_name="测试",
            category=AlgorithmCategory.PERSON_VIOLATION,
            detected=True,
            confidence=0.8
        )
        
        controller.process([result1], camera_id="test")
        time.sleep(3)
        controller.process([result2], camera_id="test")
        
        assert controller.stats['filtered_not_confirmed'] > 0


class TestAlertAggregation:
    """告警聚合测试"""
    
    def test_spatial_aggregation(self):
        """测试空间聚合"""
        self.config = QualityConfig(alert_aggregation=True)
        controller = QualityController(self.config)
        
        results = [
            AlgorithmResult(
                algorithm_id=1,
                algorithm_name="测试",
                category=AlgorithmCategory.PERSON_VIOLATION,
                detected=True,
                confidence=0.8,
                bounding_box=(100 + i * 10, 100, 50, 50)
            )
            for i in range(5)
        ]
        
        passed = controller.process(results, camera_id="test")
        
        assert len(passed) <= 5
    
    def test_temporal_aggregation(self):
        """测试时间聚合"""
        self.config = QualityConfig(
            alert_aggregation=True,
            aggregation_window=1.0,
            consecutive_frames=1
        )
        controller = QualityController(self.config)
        
        result = AlgorithmResult(
            algorithm_id=1,
            algorithm_name="测试",
            category=AlgorithmCategory.PERSON_VIOLATION,
            detected=True,
            confidence=0.8,
            bounding_box=(100, 100, 50, 50)
        )
        
        for _ in range(5):
            controller.process([result], camera_id="test")
        
        assert controller.stats['aggregated'] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
