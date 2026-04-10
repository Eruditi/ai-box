#!/usr/bin/env python3
"""
高性能摄像头管理模块
支持336路视频接入、批处理、硬件加速
"""

import os
import time
import logging
import threading
import queue
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class ProtocolType(Enum):
    """视频接入协议类型"""
    RTSP = "rtsp"
    ONVIF = "onvif"
    GB28181 = "gb28181"
    USB = "usb"


@dataclass
class CameraConfig:
    """摄像头配置"""
    source: str
    protocol: ProtocolType
    name: str = None
    resolution: Tuple[int, int] = (1920, 1080)
    fps: int = 30
    device_id: str = None
    server_id: str = None


class HighPerformanceCamera:
    """高性能摄像头类"""
    
    def __init__(self, config: CameraConfig):
        self.config = config
        self.name = config.name or f"Camera_{config.source}"
        self.capture = None
        self.connected = False
        self.frame_queue = queue.Queue(maxsize=30)
        self.running = False
        self.capture_thread = None
        self.last_frame_time = 0
        self.frame_count = 0
        self.fps_actual = 0
        
    def connect(self) -> bool:
        try:
            if self.config.protocol == ProtocolType.USB:
                cap = cv2.VideoCapture(int(self.config.source.split('/dev/video')[-1]))
            else:
                cap = cv2.VideoCapture(self.config.source)
            
            if self.config.resolution:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])
            
            if self.config.fps:
                cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            if cap.isOpened():
                self.capture = cap
                self.connected = True
                logging.info(f"Camera connected: {self.name} ({self.config.source})")
                return True
            else:
                logging.error(f"Failed to open camera: {self.config.source}")
                return False
        except Exception as e:
            logging.error(f"Error connecting to camera {self.config.source}: {e}")
            return False
    
    def disconnect(self):
        if self.capture and self.capture.isOpened():
            self.capture.release()
        self.connected = False
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        logging.info(f"Camera disconnected: {self.name}")
    
    def start_capture(self):
        if not self.connected:
            if not self.connect():
                return
        
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
    
    def stop_capture(self):
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        self.disconnect()
    
    def _capture_loop(self):
        while self.running and self.connected:
            try:
                start_time = time.time()
                ret, frame = self.capture.read()
                if ret:
                    try:
                        self.frame_queue.put_nowait(frame.copy())
                    except queue.Full:
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(frame.copy())
                        except queue.Empty:
                            pass
                    
                    self.frame_count += 1
                    current_time = time.time()
                    if current_time - self.last_frame_time >= 1.0:
                        self.fps_actual = self.frame_count
                        self.frame_count = 0
                        self.last_frame_time = current_time
                else:
                    logging.warning(f"Failed to read frame from {self.name}")
                    time.sleep(0.01)
            except Exception as e:
                logging.error(f"Capture error for {self.name}: {e}")
                time.sleep(0.1)
    
    def get_frame(self) -> Optional[np.ndarray]:
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_frame_blocking(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class HighPerformanceCameraManager:
    """高性能摄像头管理器 - 支持336路视频接入"""
    
    def __init__(self, config):
        self.config = config
        self.cameras: Dict[str, HighPerformanceCamera] = {}
        self.running = False
        self.batch_size = config.get('camera.batch_size', 16)
        self.max_cameras = config.get('camera.max_cameras', 336)
        self.camera_groups: List[List[HighPerformanceCamera]] = []
        self.processing_threads: List[threading.Thread] = []
        self.result_callback = None
        self.lock = threading.Lock()
    
    def parse_camera_config(self, source_config: Dict) -> CameraConfig:
        """解析摄像头配置"""
        protocol = ProtocolType.RTSP
        source = source_config.get('url', source_config.get('source', ''))
        
        if source.startswith('/dev/video'):
            protocol = ProtocolType.USB
        elif source_config.get('type') == 'onvif':
            protocol = ProtocolType.ONVIF
        elif source_config.get('type') == 'gb28181':
            protocol = ProtocolType.GB28181
        
        return CameraConfig(
            source=source,
            protocol=protocol,
            name=source_config.get('name'),
            device_id=source_config.get('device_id'),
            server_id=source_config.get('server_id')
        )
    
    def add_camera(self, config: CameraConfig) -> bool:
        """添加摄像头"""
        with self.lock:
            if len(self.cameras) >= self.max_cameras:
                logging.error(f"Max cameras reached: {self.max_cameras}")
                return False
            
            if config.source in self.cameras:
                logging.warning(f"Camera already exists: {config.source}")
                return False
            
            camera = HighPerformanceCamera(config)
            if camera.connect():
                self.cameras[config.source] = camera
                camera.start_capture()
                self._reorganize_camera_groups()
                logging.info(f"Camera added: {config.name} ({config.source})")
                return True
            return False
    
    def remove_camera(self, source: str):
        """移除摄像头"""
        with self.lock:
            if source in self.cameras:
                self.cameras[source].stop_capture()
                del self.cameras[source]
                self._reorganize_camera_groups()
                logging.info(f"Camera removed: {source}")
    
    def _reorganize_camera_groups(self):
        """重组摄像头分组用于批处理"""
        camera_list = list(self.cameras.values())
        self.camera_groups = []
        
        for i in range(0, len(camera_list), self.batch_size):
            group = camera_list[i:i + self.batch_size]
            self.camera_groups.append(group)
    
    def _processing_worker(self, group_index: int):
        """处理工作线程"""
        while self.running:
            try:
                if group_index < len(self.camera_groups):
                    group = self.camera_groups[group_index]
                    frames = []
                    camera_sources = []
                    
                    for camera in group:
                        frame = camera.get_frame()
                        if frame is not None:
                            frames.append(frame)
                            camera_sources.append(camera.config.source)
                    
                    if frames and self.result_callback:
                        self.result_callback(camera_sources, frames)
                
                time.sleep(0.001)
            except Exception as e:
                logging.error(f"Processing worker error (group {group_index}): {e}")
                time.sleep(0.1)
    
    def set_result_callback(self, callback):
        """设置结果回调函数"""
        self.result_callback = callback
    
    def start(self):
        """启动摄像头管理器"""
        logging.info("Starting high performance camera manager...")
        self.running = True
        
        configured_sources = self.config.get('camera.input_sources', [])
        for source_config in configured_sources:
            if isinstance(source_config, dict):
                config = self.parse_camera_config(source_config)
            else:
                config = CameraConfig(
                    source=str(source_config),
                    protocol=ProtocolType.USB if str(source_config).startswith('/dev/video') else ProtocolType.RTSP
                )
            self.add_camera(config)
        
        num_threads = min(len(self.camera_groups), os.cpu_count() or 4)
        for i in range(num_threads):
            thread = threading.Thread(target=self._processing_worker, args=(i,), daemon=True)
            thread.start()
            self.processing_threads.append(thread)
        
        logging.info(f"High performance camera manager started. Active cameras: {len(self.cameras)}, Groups: {len(self.camera_groups)}")
    
    def stop(self):
        """停止摄像头管理器"""
        logging.info("Stopping high performance camera manager...")
        self.running = False
        
        for thread in self.processing_threads:
            thread.join(timeout=2)
        
        for camera in self.cameras.values():
            camera.stop_capture()
        
        self.cameras.clear()
        self.camera_groups.clear()
        logging.info("High performance camera manager stopped")
    
    def get_camera(self, source: str) -> Optional[HighPerformanceCamera]:
        """获取指定摄像头"""
        with self.lock:
            return self.cameras.get(source)
    
    def get_all_cameras(self) -> List[HighPerformanceCamera]:
        """获取所有摄像头"""
        with self.lock:
            return list(self.cameras.values())
    
    def get_batch_frames(self) -> Tuple[List[str], List[np.ndarray]]:
        """批量获取帧"""
        sources = []
        frames = []
        with self.lock:
            for camera in self.cameras.values():
                frame = camera.get_frame()
                if frame is not None:
                    sources.append(camera.config.source)
                    frames.append(frame)
        return sources, frames
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        with self.lock:
            stats = {
                'total_cameras': len(self.cameras),
                'connected_cameras': sum(1 for c in self.cameras.values() if c.connected),
                'camera_groups': len(self.camera_groups),
                'cameras_per_group': self.batch_size,
                'camera_details': []
            }
            
            for source, camera in self.cameras.items():
                stats['camera_details'].append({
                    'source': source,
                    'name': camera.name,
                    'connected': camera.connected,
                    'fps': camera.fps_actual,
                    'protocol': camera.config.protocol.value
                })
            
            return stats
