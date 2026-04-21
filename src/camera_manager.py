#!/usr/bin/env python3
"""
摄像头管理模块
支持USB摄像头、IP摄像头自动检测和管理
"""

import os
import time
import logging
import threading
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np

from hardware_decoder import decoder_manager


class Camera:
    def __init__(self, source: str, name: str = None):
        self.source = source
        self.name = name or f"Camera_{source}"
        self.decoder = None
        self.connected = False
        self.frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.thread = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.settings = {
            'ai_enabled': True,
            'enabled_algorithms': None,
            'resolution': '1280x720',
            'fps': 30,
            'detection_interval': 1.0
        }

    def connect(self) -> bool:
        try:
            # 使用硬件解码器
            self.decoder = decoder_manager.get_decoder(self.source)
            if self.decoder and self.decoder.connected:
                self.connected = True
                self.reconnect_attempts = 0
                logging.info(f"Camera connected: {self.name} ({self.source})")
                return True
            else:
                self.reconnect_attempts += 1
                logging.error(f"Failed to open camera: {self.name} ({self.source}), attempts: {self.reconnect_attempts}")
                return False
        except Exception as e:
            self.reconnect_attempts += 1
            logging.error(f"Error connecting to camera {self.name} ({self.source}): {e}, attempts: {self.reconnect_attempts}")
            return False

    def disconnect(self):
        if self.decoder:
            self.decoder.release()
        self.connected = False
        logging.info(f"Camera disconnected: {self.name}")

    def start_capture(self):
        if not self.connected:
            if not self.connect():
                return
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop_capture(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        self.disconnect()

    def _capture_loop(self):
        last_frame_time = 0
        while self.running:
            try:
                if not self.connected:
                    if self.reconnect_attempts < self.max_reconnect_attempts:
                        logging.info(f"Attempting to reconnect camera {self.name} ({self.reconnect_attempts+1}/{self.max_reconnect_attempts})")
                        self.connect()
                        self.reconnect_attempts += 1
                        time.sleep(2)
                        continue
                    else:
                        self.reconnect_attempts = 0
                        time.sleep(5)
                        continue
                
                frame_delay = 0
                if hasattr(self.decoder, 'frame_delay'):
                    frame_delay = self.decoder.frame_delay
                
                ret, frame = self.decoder.read()
                if ret:
                    with self.frame_lock:
                        self.frame = frame.copy()
                    self.reconnect_attempts = 0
                    
                    if frame_delay > 0:
                        elapsed = time.time() - last_frame_time
                        sleep_time = frame_delay - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        last_frame_time = time.time()
                else:
                    logging.warning(f"Failed to read frame from {self.name}")
                    self.connected = False
                    time.sleep(0.1)
            except Exception as e:
                logging.error(f"Capture error for {self.name}: {e}")
                self.connected = False
                time.sleep(1)

    def get_frame(self) -> Optional[np.ndarray]:
        with self.frame_lock:
            return self.frame.copy() if self.frame is not None else None


class CameraManager:
    def __init__(self, config):
        self.config = config
        self.cameras: Dict[str, Camera] = {}
        self.running = False
        self.monitor_thread = None
        self.monitor_running = False

    def _scan_usb_cameras(self) -> List[str]:
        cameras = []
        # Windows 摄像头检测 - 不实际打开摄像头，只检查索引
        if os.name == 'nt':
            # 在 Windows 上，我们假设前几个索引可能有摄像头
            # 用户可以通过 API 手动添加摄像头
            # 或者我们可以尝试打开，但添加延迟
            for i in range(3):
                try:
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        cameras.append(str(i))
                        cap.release()
                        time.sleep(0.5)
                except Exception:
                    pass
        else:
            # Linux 摄像头检测
            for i in range(10):
                path = f"/dev/video{i}"
                if os.path.exists(path):
                    cameras.append(path)
        return cameras

    def _detect_cameras(self) -> List[str]:
        detected = []
        
        if self.config.get('camera.auto_detect', True):
            usb_cameras = self._scan_usb_cameras()
            detected.extend(usb_cameras)
            if usb_cameras:
                logging.info(f"Detected USB cameras: {usb_cameras}")
        
        configured_sources = self.config.get('camera.input_sources', [])
        for source in configured_sources:
            if source not in detected:
                # 支持摄像头、RTSP、HTTP 和视频文件
                if source.startswith('rtsp://') or source.startswith('http://') or os.path.exists(source):
                    detected.append(source)
        
        return detected

    def _add_camera(self, source: str):
        if source not in self.cameras:
            name = f"Camera_{len(self.cameras) + 1}"
            camera = Camera(source, name)
            connected = camera.connect()
            if not connected:
                logging.warning(f"Camera {source} first connect failed, retrying...")
                time.sleep(1)
                connected = camera.connect()
            self.cameras[source] = camera
            if connected:
                camera.start_capture()
                logging.info(f"Camera {source} connected and capturing")
            else:
                logging.warning(f"Camera {source} added but not yet connected, will retry in capture loop")

    def _remove_camera(self, source: str):
        if source in self.cameras:
            self.cameras[source].stop_capture()
            del self.cameras[source]

    def _monitor_cameras(self):
        while self.monitor_running:
            try:
                detected = self._detect_cameras()
                
                for source in detected:
                    if source not in self.cameras:
                        logging.info(f"New camera detected: {source}")
                        self._add_camera(source)
                
                for source in list(self.cameras.keys()):
                    if source.startswith('/dev/video') and source not in detected:
                        logging.info(f"Camera disconnected: {source}")
                        self._remove_camera(source)
                
            except Exception as e:
                logging.error(f"Camera monitoring error: {e}")
            
            time.sleep(2)

    def start(self):
        logging.info("Starting camera manager...")
        self.running = True
        
        if self.config.get('camera.auto_detect', True):
            initial_cameras = self._detect_cameras()
            for source in initial_cameras:
                self._add_camera(source)
        
            self.monitor_running = True
            self.monitor_thread = threading.Thread(target=self._monitor_cameras, daemon=True)
            self.monitor_thread.start()
        else:
            logging.info("Auto-detect disabled. Waiting for manual camera addition...")
        
        logging.info(f"Camera manager started. Active cameras: {len(self.cameras)}")

    def stop(self):
        logging.info("Stopping camera manager...")
        self.running = False
        self.monitor_running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=3)
        
        for camera in self.cameras.values():
            camera.stop_capture()
        
        self.cameras.clear()
        logging.info("Camera manager stopped")

    def get_camera(self, source: str) -> Optional[Camera]:
        return self.cameras.get(source)

    def get_all_cameras(self) -> List[Camera]:
        return list(self.cameras.values())

    def get_primary_camera(self) -> Optional[Camera]:
        cameras = self.get_all_cameras()
        return cameras[0] if cameras else None
    
    def add_camera(self, source: str, name: str = None, default_algorithms: List[int] = None) -> bool:
        """公开方法：添加摄像头"""
        if source in self.cameras:
            logging.warning(f"Camera already exists: {source}")
            return False
        
        camera_name = name or f"Camera_{len(self.cameras) + 1}"
        camera = Camera(source, camera_name)
        
        # 设置默认启用的算法（如果提供）
        if default_algorithms:
            camera.settings['enabled_algorithms'] = default_algorithms
            logging.info(f"[Camera] {camera_name} 默认启用算法: {default_algorithms}")
        
        if camera.connect():
            self.cameras[source] = camera
            camera.start_capture()
            logging.info(f"Camera added: {camera_name} ({source})")
            return True
        else:
            logging.error(f"Failed to add camera: {source}")
            return False
    
    def remove_camera(self, source: str) -> bool:
        """公开方法：移除摄像头"""
        if source not in self.cameras:
            logging.warning(f"Camera not found: {source}")
            return False
        
        self.cameras[source].stop_capture()
        del self.cameras[source]
        logging.info(f"Camera removed: {source}")
        return True
