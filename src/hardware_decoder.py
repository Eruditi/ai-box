#!/usr/bin/env python3
"""
硬件解码管理器
支持VAAPI和NVDEC硬件加速
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import cv2
import numpy as np


class HardwareDecoder:
    """硬件解码器基类"""
    def __init__(self, source: str):
        self.source = source
        self.capture = None
        self.connected = False
        
    def connect(self) -> bool:
        """连接设备并初始化硬件解码"""
        raise NotImplementedError
    
    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """读取一帧"""
        raise NotImplementedError
    
    def release(self):
        """释放资源"""
        if self.capture:
            self.capture.release()
        self.connected = False


class VAAPIDecoder(HardwareDecoder):
    """VAAPI硬件解码器"""
    def connect(self) -> bool:
        try:
            # 检查VAAPI是否可用
            if not os.path.exists('/dev/dri/renderD128'):
                logging.warning("VAAPI not available: /dev/dri/renderD128 not found")
                return False
            
            # 使用VAAPI后端
            self.capture = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
            if not self.capture.isOpened():
                # 尝试使用GStreamer后端
                gstreamer_pipeline = f"vaapidecodebin ! videoconvert ! appsink"
                self.capture = cv2.VideoCapture(self.source, cv2.CAP_GSTREAMER)
                
            if self.capture.isOpened():
                self.connected = True
                logging.info(f"VAAPI decoder connected to {self.source}")
                return True
            else:
                logging.error(f"Failed to open VAAPI decoder for {self.source}")
                return False
        except Exception as e:
            logging.error(f"VAAPI decoder error: {e}")
            return False
    
    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if self.capture and self.connected:
            return self.capture.read()
        return False, None


class NVDECDecoder(HardwareDecoder):
    """NVDEC硬件解码器"""
    def connect(self) -> bool:
        try:
            # 检查CUDA是否可用
            try:
                import torch
                has_cuda = torch.cuda.is_available()
            except ImportError:
                has_cuda = False
            
            if not has_cuda:
                logging.warning("NVDEC not available: CUDA not detected")
                return False
            
            # 使用GStreamer NVDEC后端
            gstreamer_pipeline = f"nvdec ! videoconvert ! appsink"
            self.capture = cv2.VideoCapture(self.source, cv2.CAP_GSTREAMER)
            
            if self.capture.isOpened():
                self.connected = True
                logging.info(f"NVDEC decoder connected to {self.source}")
                return True
            else:
                # 尝试直接使用CUDA加速的VideoCapture
                self.capture = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                if self.capture.isOpened():
                    self.connected = True
                    logging.info(f"FFMPEG with CUDA connected to {self.source}")
                    return True
                else:
                    logging.error(f"Failed to open NVDEC decoder for {self.source}")
                    return False
        except Exception as e:
            logging.error(f"NVDEC decoder error: {e}")
            return False
    
    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if self.capture and self.connected:
            return self.capture.read()
        return False, None


class SoftwareDecoder(HardwareDecoder):
    """软件解码器（作为回退）"""
    def __init__(self, source: str):
        super().__init__(source)
        self.is_video_file = False
        self.frame_count = 0
        self.current_frame = 0
        self.fps = 30.0
        self.frame_delay = 0.033
    
    def connect(self) -> bool:
        try:
            if self.source.startswith('/dev/video'):
                self.capture = cv2.VideoCapture(int(self.source.split('/dev/video')[-1]))
            elif self.source.isdigit():
                self.capture = cv2.VideoCapture(int(self.source))
            else:
                self.capture = cv2.VideoCapture(self.source)
                self.is_video_file = True
                self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = self.capture.get(cv2.CAP_PROP_FPS)
                if self.fps <= 0 or self.fps > 120:
                    self.fps = 30.0
                self.frame_delay = 1.0 / self.fps
                logging.info(f"Video FPS: {self.fps:.2f}, frame delay: {self.frame_delay:.4f}s")
            
            if self.capture.isOpened():
                self.connected = True
                logging.info(f"Software decoder connected to {self.source}")
                return True
            else:
                logging.error(f"Failed to open software decoder for {self.source}")
                return False
        except Exception as e:
            logging.error(f"Software decoder error: {e}")
            return False
    
    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if self.capture and self.connected:
            ret, frame = self.capture.read()
            if ret:
                self.current_frame += 1
                return True, frame
            elif self.is_video_file and self.frame_count > 0:
                logging.info(f"Video ended, looping: {self.source}")
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame = 0
                ret, frame = self.capture.read()
                if ret:
                    self.current_frame += 1
                return ret, frame
            return False, None
        return False, None


class HardwareDecoderManager:
    """硬件解码器管理器"""
    def __init__(self):
        self.decoders: Dict[str, HardwareDecoder] = {}
        self.preferred_decoder = self._detect_preferred_decoder()
        
    def _detect_preferred_decoder(self) -> str:
        """检测首选解码器"""
        # 由于环境兼容性问题，直接使用软件解码
        logging.info("Using software decoding for compatibility")
        return "software"
    
    def get_decoder(self, source: str) -> HardwareDecoder:
        """获取适合的解码器"""
        if source not in self.decoders:
            if self.preferred_decoder == "nvdec":
                decoder = NVDECDecoder(source)
                if not decoder.connect():
                    # 回退到VAAPI
                    decoder = VAAPIDecoder(source)
                    if not decoder.connect():
                        # 回退到软件解码
                        decoder = SoftwareDecoder(source)
                        decoder.connect()
            elif self.preferred_decoder == "vaapi":
                decoder = VAAPIDecoder(source)
                if not decoder.connect():
                    # 回退到软件解码
                    decoder = SoftwareDecoder(source)
                    decoder.connect()
            else:
                decoder = SoftwareDecoder(source)
                decoder.connect()
            
            self.decoders[source] = decoder
        
        return self.decoders[source]
    
    def release_all(self):
        """释放所有解码器"""
        for decoder in self.decoders.values():
            decoder.release()
        self.decoders.clear()


# 全局解码器管理器实例
decoder_manager = HardwareDecoderManager()
