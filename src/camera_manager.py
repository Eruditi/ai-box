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


class Camera:
    def __init__(self, source: str, name: str = None):
        self.source = source
        self.name = name or f"Camera_{source}"
        self.capture = None
        self.connected = False
        self.frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.thread = None

    def connect(self) -> bool:
        try:
            if self.source.startswith('/dev/video'):
                cap = cv2.VideoCapture(int(self.source.split('/dev/video')[-1]))
            else:
                cap = cv2.VideoCapture(self.source)
            
            if cap.isOpened():
                self.capture = cap
                self.connected = True
                logging.info(f"Camera connected: {self.name} ({self.source})")
                return True
            else:
                logging.error(f"Failed to open camera: {self.source}")
                return False
        except Exception as e:
            logging.error(f"Error connecting to camera {self.source}: {e}")
            return False

    def disconnect(self):
        if self.capture and self.capture.isOpened():
            self.capture.release()
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
        while self.running and self.connected:
            try:
                ret, frame = self.capture.read()
                if ret:
                    with self.frame_lock:
                        self.frame = frame.copy()
                else:
                    logging.warning(f"Failed to read frame from {self.name}")
                    time.sleep(0.1)
            except Exception as e:
                logging.error(f"Capture error for {self.name}: {e}")
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
                if source.startswith('rtsp://') or source.startswith('http://'):
                    detected.append(source)
        
        return detected

    def _add_camera(self, source: str):
        if source not in self.cameras:
            name = f"Camera_{len(self.cameras) + 1}"
            camera = Camera(source, name)
            if camera.connect():
                self.cameras[source] = camera
                camera.start_capture()

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
        
        initial_cameras = self._detect_cameras()
        for source in initial_cameras:
            self._add_camera(source)
        
        if self.config.get('camera.auto_detect', True):
            self.monitor_running = True
            self.monitor_thread = threading.Thread(target=self._monitor_cameras, daemon=True)
            self.monitor_thread.start()
        
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
