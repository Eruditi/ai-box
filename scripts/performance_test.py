#!/usr/bin/env python3
"""
性能测试脚本
测试系统在不同摄像头数量下的性能表现
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.multiprocess_manager import MultiprocessManager
from src.config_manager import ConfigManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_test_config(camera_count):
    """生成测试配置"""
    return {
        'multiprocess': {
            'enabled': True,
            'max_processes': camera_count
        },
        'camera': {
            'auto_detect': False,
            'input_sources': [f'rtsp://test_camera_{i}' for i in range(camera_count)]
        },
        'ai': {
            'enabled': True,
            'enabled_algorithms': [1, 2, 3]  # 只启用几个基础算法
        },
        'health': {
            'check_interval': 60,
            'auto_restart': False
        }
    }


def run_performance_test(camera_count, test_duration=30):
    """运行性能测试"""
    print(f"[DEBUG] Starting performance test with {camera_count} cameras")
    logger.info(f"Starting performance test with {camera_count} cameras")
    
    # 创建临时配置
    test_config = generate_test_config(camera_count)
    print(f"[DEBUG] Generated test config: {test_config}")
    
    # 创建配置管理器
    config_manager = ConfigManager('config/test_config.yaml')
    config_manager.config = test_config
    config_manager._save_config()
    print(f"[DEBUG] Saved test config")
    
    try:
        # 创建多进程管理器
        print(f"[DEBUG] Creating MultiprocessManager")
        manager = MultiprocessManager(test_config)
        
        # 启动管理器
        start_time = time.time()
        print(f"[DEBUG] Starting manager")
        manager.start()
        print(f"[DEBUG] Manager started")
        
        # 等待系统稳定
        time.sleep(5)
        print(f"[DEBUG] System stabilized")
        
        # 开始测试
        print(f"[DEBUG] Testing with {camera_count} cameras for {test_duration} seconds")
        logger.info(f"Testing with {camera_count} cameras for {test_duration} seconds")
        
        # 记录测试期间的性能指标
        import psutil
        process = psutil.Process()
        
        cpu_usages = []
        memory_usages = []
        
        test_start = time.time()
        while time.time() - test_start < test_duration:
            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory().percent
            cpu_usages.append(cpu)
            memory_usages.append(memory)
            print(f"[DEBUG] CPU: {cpu}%, Memory: {memory}%")
            time.sleep(1)
        
        # 停止管理器
        print(f"[DEBUG] Stopping manager")
        manager.stop()
        test_end = time.time()
        print(f"[DEBUG] Manager stopped")
        
        # 计算性能指标
        avg_cpu = sum(cpu_usages) / len(cpu_usages)
        max_cpu = max(cpu_usages)
        avg_memory = sum(memory_usages) / len(memory_usages)
        max_memory = max(memory_usages)
        test_duration = test_end - test_start
        
        # 打印测试结果
        print("=" * 60)
        print(f"Performance Test Results for {camera_count} cameras")
        print("=" * 60)
        print(f"Test Duration: {test_duration:.2f} seconds")
        print(f"Average CPU Usage: {avg_cpu:.2f}%")
        print(f"Max CPU Usage: {max_cpu:.2f}%")
        print(f"Average Memory Usage: {avg_memory:.2f}%")
        print(f"Max Memory Usage: {max_memory:.2f}%")
        print(f"Active Camera Processes: {len(manager.get_active_cameras())}")
        print("=" * 60)
        
        logger.info("=" * 60)
        logger.info(f"Performance Test Results for {camera_count} cameras")
        logger.info("=" * 60)
        logger.info(f"Test Duration: {test_duration:.2f} seconds")
        logger.info(f"Average CPU Usage: {avg_cpu:.2f}%")
        logger.info(f"Max CPU Usage: {max_cpu:.2f}%")
        logger.info(f"Average Memory Usage: {avg_memory:.2f}%")
        logger.info(f"Max Memory Usage: {max_memory:.2f}%")
        logger.info(f"Active Camera Processes: {len(manager.get_active_cameras())}")
        logger.info("=" * 60)
        
        return {
            'camera_count': camera_count,
            'duration': test_duration,
            'avg_cpu': avg_cpu,
            'max_cpu': max_cpu,
            'avg_memory': avg_memory,
            'max_memory': max_memory,
            'active_processes': len(manager.get_active_cameras())
        }
        
    except Exception as e:
        print(f"[ERROR] Performance test error: {e}")
        logger.error(f"Performance test error: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # 清理临时配置
        if os.path.exists('config/test_config.yaml'):
            os.remove('config/test_config.yaml')
            print(f"[DEBUG] Removed test config file")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Performance test for AI Box')
    parser.add_argument('--cameras', type=int, nargs='+', default=[4, 8, 16, 32],
                      help='Camera counts to test')
    parser.add_argument('--duration', type=int, default=30,
                      help='Test duration in seconds')
    
    args = parser.parse_args()
    
    results = []
    
    for camera_count in args.cameras:
        result = run_performance_test(camera_count, args.duration)
        if result:
            results.append(result)
        # 休息一下，让系统恢复
        time.sleep(5)
    
    # 打印汇总结果
    if results:
        logger.info("\n" + "=" * 80)
        logger.info("PERFORMANCE TEST SUMMARY")
        logger.info("=" * 80)
        logger.info(f"{'Cameras':<10} {'Duration':<10} {'Avg CPU':<10} {'Max CPU':<10} {'Avg Mem':<10} {'Max Mem':<10} {'Processes':<10}")
        logger.info("-" * 80)
        
        for result in results:
            logger.info(f"{result['camera_count']:<10} {result['duration']:<10.2f} {result['avg_cpu']:<10.2f} {result['max_cpu']:<10.2f} {result['avg_memory']:<10.2f} {result['max_memory']:<10.2f} {result['active_processes']:<10}")
        
        logger.info("=" * 80)


if __name__ == "__main__":
    main()
