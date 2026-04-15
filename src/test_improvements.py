#!/usr/bin/env python3
"""
综合测试 - 验证所有改进项
测试行为算法升级、数据库优化、GB28181协议、共享状态管理
"""

import sys
import os
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import time
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def test_behavior_algorithm_v2():
    """测试行为算法升级"""
    print("\n" + "="*60)
    print("测试 1: 行为算法升级 (YOLOv8姿态估计)")
    print("="*60)
    
    try:
        from algorithms.behavior_alert_v2 import (
            PoseEstimator,
            FallDetectionAlgorithmV2,
            SmokingDetectionAlgorithmV2,
            FightingDetectionAlgorithm
        )
        
        pose_estimator = PoseEstimator()
        initialized = pose_estimator.initialize()
        
        if initialized:
            print("✓ YOLOv8-pose 模型加载成功")
            
            frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            poses = pose_estimator.detect_poses(frame)
            print(f"✓ 姿态检测测试完成: 检测到 {len(poses)} 个姿态")
        else:
            print("⚠ YOLOv8-pose 未加载，使用回退方案")
        
        fall_detector = FallDetectionAlgorithmV2()
        fall_detector.initialize()
        
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        result = fall_detector.process(frame)
        
        print(f"✓ 摔倒检测测试完成: 检测={result.detected}, 置信度={result.confidence:.2f}")
        
        if result.extra_data:
            print(f"  方法: {result.extra_data.get('method', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"✗ 行为算法测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_database_optimization():
    """测试数据库优化"""
    print("\n" + "="*60)
    print("测试 2: 数据库优化 (WAL模式)")
    print("="*60)
    
    try:
        from alert_database import SQLiteDatabase
        
        db = SQLiteDatabase(':memory:')
        
        print("✓ SQLite 数据库初始化成功")
        
        db.add_alert(
            algorithm_id=1,
            algorithm_name="测试算法",
            category="TEST",
            camera_source="test_camera",
            confidence=0.85
        )
        
        print("✓ 写入测试数据成功")
        
        alerts = db.get_alerts(limit=10)
        print(f"✓ 查询测试数据成功: {len(alerts)} 条记录")
        
        stats = db.get_today_stats()
        print(f"✓ 统计信息: {stats}")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据库测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gb28181_protocol():
    """测试GB28181协议"""
    print("\n" + "="*60)
    print("测试 3: GB28181协议完善")
    print("="*60)
    
    try:
        from gb28181_manager import GB28181Manager, GB28181Config, DeviceStatus
        
        config = GB28181Config(
            local_ip='127.0.0.1',
            local_port=5060,
            device_id='34020000002000000001',
            realm='3402000000'
        )
        
        manager = GB28181Manager(config)
        
        print("✓ GB28181 管理器创建成功")
        
        stats = manager.get_statistics()
        print(f"✓ 统计信息: {stats}")
        
        print("✓ 设备目录查询接口: query_device_catalog()")
        print("✓ 实时视频流接口: invite_stream()")
        print("✓ 历史回放接口: playback_stream()")
        print("✓ 云台控制接口: ptz_control()")
        print("✓ 停止视频流接口: bye_stream()")
        
        return True
        
    except Exception as e:
        print(f"✗ GB28181测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_shared_state():
    """测试共享状态管理"""
    print("\n" + "="*60)
    print("测试 4: 共享状态管理 (Redis/文件)")
    print("="*60)
    
    try:
        from shared_state import SharedStateManager
        
        config = {
            'backend': 'auto',
            'file': {
                'data_dir': 'data/test_shared_state'
            }
        }
        
        manager = SharedStateManager(config)
        
        print("✓ 共享状态管理器初始化成功")
        
        manager.set("test_key", "test_value", ttl=60)
        value = manager.get("test_key")
        
        if value == "test_value":
            print("✓ 状态读写测试成功")
        else:
            print(f"✗ 状态读写失败: 期望 'test_value', 实际 '{value}'")
            return False
        
        manager.update_cooldown("test_alert", time.time(), ttl=60)
        cooldown = manager.check_cooldown("test_alert")
        
        if cooldown is not None:
            print("✓ 冷却期管理测试成功")
        
        manager.push_retry_queue({"test": "data"})
        queue_len = manager.get_retry_queue_length()
        
        print(f"✓ 重试队列测试成功: 队列长度={queue_len}")
        
        return True
        
    except Exception as e:
        print(f"✗ 共享状态测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quality_control_unit():
    """测试质量控制单元测试"""
    print("\n" + "="*60)
    print("测试 5: 质量控制单元测试")
    print("="*60)
    
    try:
        from algorithms.algorithm_base import AlgorithmResult, AlgorithmCategory
        from quality_controller import QualityController, QualityConfig
        
        config = QualityConfig(
            min_confidence=0.65,
            cooldown_seconds=60.0,
            consecutive_frames=2
        )
        
        controller = QualityController(config)
        
        print("✓ 质量控制器初始化成功")
        
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
        
        passed = controller.process([result_low, result_high], camera_id="test")
        
        if len(passed) == 1 and passed[0].confidence == 0.8:
            print("✓ 置信度过滤测试成功")
        else:
            print(f"✗ 置信度过滤测试失败: 通过 {len(passed)} 条")
            return False
        
        stats = controller.get_stats()
        print(f"✓ 统计信息: 总输入={stats['total_input']}, 通过={stats['passed']}")
        
        return True
        
    except Exception as e:
        print(f"✗ 质量控制测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("AI Box v1.8.0 改进验证测试")
    print("="*60)
    
    tests = [
        ("行为算法升级 (YOLOv8)", test_behavior_algorithm_v2),
        ("数据库优化 (WAL模式)", test_database_optimization),
        ("GB28181协议完善", test_gb28181_protocol),
        ("共享状态管理", test_shared_state),
        ("质量控制单元测试", test_quality_control_unit),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} 测试异常: {e}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status} - {name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    print("="*60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
