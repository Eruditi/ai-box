#!/usr/bin/env python3
"""
简单性能测试脚本
测试多进程架构的性能
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import multiprocessing
from multiprocessing import Process, Queue, Pipe

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def worker_process(process_id, queue, pipe):
    """工作进程"""
    try:
        logging.info(f"Worker process {process_id} started")
        running = True
        while running:
            # 检查控制命令
            if pipe.poll(timeout=0.01):
                command = pipe.recv()
                if command == 'stop':
                    running = False
                    break
            
            # 模拟处理任务
            time.sleep(0.033)  # 约30fps
            
            # 发送结果
            result = {
                'process_id': process_id,
                'timestamp': time.time(),
                'data': f'Test data from process {process_id}'
            }
            queue.put(result)
    except Exception as e:
        logging.error(f"Worker process {process_id} error: {e}")
    finally:
        logging.info(f"Worker process {process_id} stopped")


def run_performance_test(process_count, test_duration=30):
    """运行性能测试"""
    logger.info(f"Starting performance test with {process_count} processes")
    
    try:
        # 创建结果队列
        result_queue = Queue(maxsize=1000)
        
        # 创建进程和管道
        processes = []
        pipes = []
        
        # 启动进程
        start_time = time.time()
        for i in range(process_count):
            parent_pipe, child_pipe = Pipe()
            process = Process(
                target=worker_process,
                args=(i, result_queue, child_pipe)
            )
            process.start()
            processes.append(process)
            pipes.append(parent_pipe)
        
        # 等待系统稳定
        time.sleep(5)
        
        # 开始测试
        logger.info(f"Testing with {process_count} processes for {test_duration} seconds")
        
        # 记录测试期间的性能指标
        import psutil
        
        cpu_usages = []
        memory_usages = []
        queue_sizes = []
        
        test_start = time.time()
        while time.time() - test_start < test_duration:
            cpu_usages.append(psutil.cpu_percent())
            memory_usages.append(psutil.virtual_memory().percent)
            queue_sizes.append(result_queue.qsize())
            time.sleep(1)
        
        # 停止进程
        for pipe in pipes:
            pipe.send('stop')
        
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        
        test_end = time.time()
        
        # 计算性能指标
        avg_cpu = sum(cpu_usages) / len(cpu_usages)
        max_cpu = max(cpu_usages)
        avg_memory = sum(memory_usages) / len(memory_usages)
        max_memory = max(memory_usages)
        avg_queue_size = sum(queue_sizes) / len(queue_sizes)
        max_queue_size = max(queue_sizes)
        test_duration = test_end - test_start
        
        # 打印测试结果
        print("=" * 60)
        print(f"Performance Test Results for {process_count} processes")
        print("=" * 60)
        print(f"Test Duration: {test_duration:.2f} seconds")
        print(f"Average CPU Usage: {avg_cpu:.2f}%")
        print(f"Max CPU Usage: {max_cpu:.2f}%")
        print(f"Average Memory Usage: {avg_memory:.2f}%")
        print(f"Max Memory Usage: {max_memory:.2f}%")
        print(f"Average Queue Size: {avg_queue_size:.2f}")
        print(f"Max Queue Size: {max_queue_size:.2f}")
        print("=" * 60)
        
        logger.info("=" * 60)
        logger.info(f"Performance Test Results for {process_count} processes")
        logger.info("=" * 60)
        logger.info(f"Test Duration: {test_duration:.2f} seconds")
        logger.info(f"Average CPU Usage: {avg_cpu:.2f}%")
        logger.info(f"Max CPU Usage: {max_cpu:.2f}%")
        logger.info(f"Average Memory Usage: {avg_memory:.2f}%")
        logger.info(f"Max Memory Usage: {max_memory:.2f}%")
        logger.info(f"Average Queue Size: {avg_queue_size:.2f}")
        logger.info(f"Max Queue Size: {max_queue_size:.2f}")
        logger.info("=" * 60)
        
        return {
            'process_count': process_count,
            'duration': test_duration,
            'avg_cpu': avg_cpu,
            'max_cpu': max_cpu,
            'avg_memory': avg_memory,
            'max_memory': max_memory,
            'avg_queue_size': avg_queue_size,
            'max_queue_size': max_queue_size
        }
        
    except Exception as e:
        logger.error(f"Performance test error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Simple performance test for multiprocess architecture')
    parser.add_argument('--processes', type=int, nargs='+', default=[4, 8, 16, 32, 64],
                      help='Process counts to test')
    parser.add_argument('--duration', type=int, default=30,
                      help='Test duration in seconds')
    
    args = parser.parse_args()
    
    results = []
    
    for process_count in args.processes:
        result = run_performance_test(process_count, args.duration)
        if result:
            results.append(result)
        # 休息一下，让系统恢复
        time.sleep(5)
    
    # 打印汇总结果
    if results:
        print("\n" + "=" * 100)
        print("PERFORMANCE TEST SUMMARY")
        print("=" * 100)
        print(f"{'Processes':<10} {'Duration':<10} {'Avg CPU':<10} {'Max CPU':<10} {'Avg Mem':<10} {'Max Mem':<10} {'Avg Queue':<10} {'Max Queue':<10}")
        print("-" * 100)
        
        for result in results:
            print(f"{result['process_count']:<10} {result['duration']:<10.2f} {result['avg_cpu']:<10.2f} {result['max_cpu']:<10.2f} {result['avg_memory']:<10.2f} {result['max_memory']:<10.2f} {result['avg_queue_size']:<10.2f} {result['max_queue_size']:<10.2f}")
        
        print("=" * 100)


if __name__ == "__main__":
    main()
