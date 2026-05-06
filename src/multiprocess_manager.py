#!/usr/bin/env python3
"""
多进程管理器
负责管理系统的多进程架构，实现进程间通信
"""

import os
import sys
import time
import logging
import threading
import multiprocessing
from pathlib import Path
from typing import List, Dict, Optional, Any
from multiprocessing import Process, Queue, Pipe

import cv2
import numpy as np

from camera_manager import Camera, CameraManager
from ai_analyzer_simple import SimpleAIAnalyzer
from hardware_decoder import decoder_manager


class CameraProcess(Process):
    """摄像头处理进程"""
    def __init__(self, camera_source: str, camera_name: str, config: Dict, 
                 result_queue: Queue, control_pipe: Pipe, process_id: int):
        super().__init__()
        self.camera_source = camera_source
        self.camera_name = camera_name
        self.config = config
        self.result_queue = result_queue
        self.control_pipe = control_pipe
        self.process_id = process_id
        self.running = False
        self.camera = None
        self.ai_analyzer = None
    
    def run(self):
        """进程运行函数"""
        try:
            # 配置日志
            logger = multiprocessing.get_logger()
            logger.setLevel(logging.INFO)
            
            logger.info(f"Starting camera process {self.process_id} for {self.camera_name}")
            
            # 初始化摄像头
            self.camera = Camera(self.camera_source, self.camera_name)
            if not self.camera.connect():
                logger.error(f"Failed to connect to camera {self.camera_name}")
                return
            
            # 初始化AI分析器
            self.ai_analyzer = SimpleAIAnalyzer(self.config)
            if not self.ai_analyzer.initialize():
                logger.error(f"Failed to initialize AI analyzer for {self.camera_name}")
                return
            
            # 开始捕获
            self.camera.start_capture()
            self.running = True
            
            # 主循环
            while self.running:
                # 检查控制命令
                if self.control_pipe.poll(timeout=0.01):
                    command = self.control_pipe.recv()
                    if command == 'stop':
                        self.running = False
                        break
                    elif command == 'restart':
                        self.camera.stop_capture()
                        time.sleep(1)
                        self.camera.start_capture()
                
                # 读取帧
                frame = self.camera.get_frame()
                if frame is not None:
                    # 分析帧
                    results = self.ai_analyzer.algorithm_manager.process_frame(
                        frame, self.ai_analyzer.enabled_algorithms
                    )
                    
                    # 发送结果到主进程
                    if results:
                        result_data = {
                            'camera_source': self.camera_source,
                            'camera_name': self.camera_name,
                            'results': results,
                            'timestamp': time.time()
                        }
                        self.result_queue.put(result_data)
                
                time.sleep(0.033)  # 约30fps
                
        except Exception as e:
            logger.error(f"Error in camera process {self.process_id}: {e}")
        finally:
            # 清理资源
            if self.camera:
                self.camera.stop_capture()
            if self.ai_analyzer:
                self.ai_analyzer.stop()
            logger.info(f"Camera process {self.process_id} stopped")


class MultiprocessManager:
    """多进程管理器"""
    def __init__(self, config):
        self.config = config
        self.processes: Dict[str, CameraProcess] = {}
        self.result_queue = Queue(maxsize=1000)
        self.control_pipes: Dict[str, Pipe] = {}
        self.camera_manager = CameraManager(config)
        self.results: Dict[str, Any] = {}
        self.running = False
        self.monitor_thread = None
        self.memory_monitor_thread = None
    
    def start(self):
        """启动多进程系统"""
        logging.info("Starting multiprocess manager...")
        self.running = True
        
        # 启动摄像头管理器
        self.camera_manager.start()
        
        # 为每个摄像头创建进程
        cameras = self.camera_manager.get_all_cameras()
        for i, camera in enumerate(cameras):
            self._start_camera_process(camera, i)
        
        # 启动结果监控线程
        self.monitor_thread = threading.Thread(
            target=self._monitor_results, daemon=True
        )
        self.monitor_thread.start()
        
        # 启动内存监控线程
        self.memory_monitor_thread = threading.Thread(
            target=self._monitor_memory, daemon=True
        )
        self.memory_monitor_thread.start()
        
        logging.info(f"Multiprocess manager started with {len(self.processes)} camera processes")
    
    def _start_camera_process(self, camera: Camera, process_id: int):
        """启动单个摄像头进程"""
        source = camera.source
        if source not in self.processes:
            # 创建控制管道
            parent_pipe, child_pipe = Pipe()
            
            # 创建进程
            process = CameraProcess(
                camera_source=source,
                camera_name=camera.name,
                config=self.config,
                result_queue=self.result_queue,
                control_pipe=child_pipe,
                process_id=process_id
            )
            
            # 启动进程
            process.start()
            
            # 保存进程和管道
            self.processes[source] = process
            self.control_pipes[source] = parent_pipe
            
            logging.info(f"Started camera process {process_id} for {camera.name}")
    
    def _monitor_results(self):
        """监控结果队列"""
        while self.running:
            try:
                # 处理队列中的所有结果，避免队列积压
                while not self.result_queue.empty():
                    try:
                        result_data = self.result_queue.get(timeout=0.1)
                        source = result_data['camera_source']
                        # 只保留最新的结果，避免内存积累
                        self.results[source] = result_data
                    except Exception:
                        break
                time.sleep(0.01)
            except Exception as e:
                logging.error(f"Error monitoring results: {e}")
                time.sleep(1)
    
    def _monitor_memory(self):
        """监控内存使用情况"""
        try:
            import psutil
            while self.running:
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                memory_percent = process.memory_percent()
                
                # 打印内存使用情况
                if memory_percent > 80:
                    logging.warning(f"High memory usage: {memory_percent:.2f}% ({memory_info.rss / 1024 / 1024:.2f} MB)")
                    # 可以在这里添加内存释放逻辑
                
                # 清理过期的结果数据
                current_time = time.time()
                expired_sources = []
                for source, result_data in self.results.items():
                    if current_time - result_data.get('timestamp', 0) > 300:  # 5分钟过期
                        expired_sources.append(source)
                
                for source in expired_sources:
                    del self.results[source]
                
                time.sleep(10)
        except ImportError:
            logging.warning("psutil not available, memory monitoring disabled")
        except Exception as e:
            logging.error(f"Error monitoring memory: {e}")
    
    def stop(self):
        """停止多进程系统"""
        logging.info("Stopping multiprocess manager...")
        self.running = False
        
        # 停止所有摄像头进程
        for source, process in self.processes.items():
            try:
                self.control_pipes[source].send('stop')
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
            except Exception as e:
                logging.error(f"Error stopping process for {source}: {e}")
        
        # 停止摄像头管理器
        self.camera_manager.stop()
        
        # 停止监控线程
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        if self.memory_monitor_thread:
            self.memory_monitor_thread.join(timeout=2)
        
        # 释放队列
        self.result_queue.close()
        self.result_queue.join_thread()
        
        # 清理
        self.processes.clear()
        self.control_pipes.clear()
        self.results.clear()
        
        logging.info("Multiprocess manager stopped")
    
    def add_camera(self, source: str, name: str = None):
        """添加摄像头进程"""
        # 先通过摄像头管理器添加摄像头
        self.camera_manager._add_camera(source)
        
        # 找到新添加的摄像头
        camera = self.camera_manager.get_camera(source)
        if camera:
            process_id = len(self.processes)
            self._start_camera_process(camera, process_id)
            return True
        return False
    
    def remove_camera(self, source: str):
        """移除摄像头进程"""
        if source in self.processes:
            # 停止进程
            try:
                self.control_pipes[source].send('stop')
                self.processes[source].join(timeout=3)
                if self.processes[source].is_alive():
                    self.processes[source].terminate()
            except Exception as e:
                logging.error(f"Error removing process for {source}: {e}")
            
            # 从摄像头管理器移除
            self.camera_manager._remove_camera(source)
            
            # 清理
            del self.processes[source]
            del self.control_pipes[source]
            if source in self.results:
                del self.results[source]
            
            return True
        return False
    
    def get_results(self, camera_source: str = None) -> List[Any]:
        """获取分析结果"""
        if camera_source:
            return [self.results.get(camera_source, {})]
        else:
            return list(self.results.values())
    
    def get_active_cameras(self) -> List[str]:
        """获取活跃的摄像头列表"""
        return list(self.processes.keys())
    
    def restart_camera(self, source: str):
        """重启摄像头进程"""
        if source in self.control_pipes:
            try:
                self.control_pipes[source].send('restart')
                return True
            except Exception as e:
                logging.error(f"Error restarting camera {source}: {e}")
                return False
        return False
