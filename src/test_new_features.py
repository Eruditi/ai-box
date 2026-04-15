#!/usr/bin/env python3
"""
测试新功能模块
验证误报学习、场景自适应、跨摄像头关联、插件SDK、算法评测
"""

import sys
import os
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_false_positive_learner():
    """测试误报学习系统"""
    print("\n" + "="*60)
    print("测试 1: AI误报学习系统")
    print("="*60)
    
    try:
        from false_positive_learner import get_false_positive_learner
        
        learner = get_false_positive_learner({'data_dir': 'data/test_fp_learning'})
        
        sample_id = learner.mark_false_positive(
            algorithm_id=1,
            algorithm_name="安全帽检测",
            camera_source="Camera_1",
            confidence=0.75,
            bbox=(100, 100, 50, 50),
            reason="误报：实际是帽子不是安全帽",
            user_id="test_user"
        )
        
        print(f"✅ 标记误报成功: {sample_id}")
        
        stats = learner.get_statistics()
        print(f"✅ 统计信息: {stats['total_samples']} 个样本, {stats['total_rules']} 条规则")
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_scene_adaptive():
    """测试场景自适应系统"""
    print("\n" + "="*60)
    print("测试 2: 场景自适应阈值调整")
    print("="*60)
    
    try:
        from scene_adaptive import get_scene_adaptive_controller
        
        controller = get_scene_adaptive_controller()
        
        test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        controller.update_scene("Camera_1", test_frame)
        
        scene_info = controller.get_scene_info("Camera_1")
        print(f"✅ 场景信息: 类型={scene_info['scene_type']}, 天气={scene_info['weather']}")
        
        thresholds = controller.get_adjusted_thresholds("Camera_1", 1)
        print(f"✅ 调整后阈值: 置信度={thresholds.min_confidence:.3f}, 冷却={thresholds.cooldown_seconds:.1f}s")
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_cross_camera_correlator():
    """测试跨摄像头关联系统"""
    print("\n" + "="*60)
    print("测试 3: 跨摄像头关联告警")
    print("="*60)
    
    try:
        from cross_camera_correlator import get_cross_camera_correlator
        
        correlator = get_cross_camera_correlator()
        
        detections = [
            {'type': 'person', 'bbox': (100, 100, 50, 100), 'confidence': 0.85},
            {'type': 'person', 'bbox': (200, 150, 50, 100), 'confidence': 0.90}
        ]
        
        test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        alerts = correlator.process_detections("Camera_1", detections, test_frame)
        print(f"✅ 处理检测结果: {len(alerts)} 个关联告警")
        
        objects = correlator.get_tracked_objects("Camera_1")
        print(f"✅ 跟踪对象: {len(objects)} 个活跃对象")
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_plugin_sdk():
    """测试插件SDK"""
    print("\n" + "="*60)
    print("测试 4: 插件开发SDK")
    print("="*60)
    
    try:
        from plugin_sdk import PluginSDK
        
        plugin_path = PluginSDK.create_plugin_template(
            "plugins/test_plugins",
            "test_algorithm",
            999
        )
        print(f"✅ 创建插件模板: {plugin_path}")
        
        validation = PluginSDK.validate_plugin(plugin_path)
        print(f"✅ 验证插件: 有效={validation['valid']}, 错误={validation['errors']}")
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_algorithm_benchmark():
    """测试算法评测系统"""
    print("\n" + "="*60)
    print("测试 5: 算法精度评测")
    print("="*60)
    
    try:
        from algorithm_benchmark import get_benchmark_manager, BoundingBox, GroundTruth
        
        manager = get_benchmark_manager({'data_dir': 'data/test_benchmarks'})
        
        dataset = manager.create_dataset(
            name="test_dataset",
            description="测试数据集",
            image_paths=["test_image_1.jpg", "test_image_2.jpg"],
            annotations={
                "test_image_1": [
                    {"x": 100, "y": 100, "width": 50, "height": 50, "label": "person"}
                ]
            },
            categories=["person"]
        )
        print(f"✅ 创建数据集: {dataset.name}, {len(dataset.images)} 张图像")
        
        leaderboard = manager.get_leaderboard()
        print(f"✅ 排行榜: {len(leaderboard)} 个算法")
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def test_quality_controller_integration():
    """测试质量控制集成"""
    print("\n" + "="*60)
    print("测试 6: 质量控制器集成")
    print("="*60)
    
    try:
        from quality_controller import QualityController, QualityConfig
        
        config = QualityConfig(
            enable_false_positive_learning=True,
            enable_scene_adaptive=True,
            enable_cross_camera=True
        )
        
        controller = QualityController(config)
        
        print(f"✅ 误报学习: {'已启用' if controller.false_positive_learner else '未启用'}")
        print(f"✅ 场景自适应: {'已启用' if controller.scene_adaptive_controller else '未启用'}")
        print(f"✅ 跨摄像头关联: {'已启用' if controller.cross_camera_correlator else '未启用'}")
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("AI Box v1.7.0 新功能测试")
    print("="*60)
    
    tests = [
        ("AI误报学习系统", test_false_positive_learner),
        ("场景自适应阈值", test_scene_adaptive),
        ("跨摄像头关联告警", test_cross_camera_correlator),
        ("插件开发SDK", test_plugin_sdk),
        ("算法精度评测", test_algorithm_benchmark),
        ("质量控制集成", test_quality_controller_integration),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} 测试异常: {e}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    print("="*60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
