#!/usr/bin/env python3
"""
AI盒子主程序
即插即用的AI摄像头分析系统
"""

import os
import sys
import time
import logging
import signal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from camera_manager import CameraManager
from high_performance_camera_manager import HighPerformanceCameraManager
from ai_analyzer import AIAnalyzer
from web_server import WebServer
from config_manager import ConfigManager
from health_monitor import HealthMonitor

class AIBox:
    def __init__(self, config_path):
        self.config = ConfigManager(config_path)
        self.setup_logging()
        
        self.camera_manager = None
        self.ai_analyzer = None
        self.web_server = None
        self.health_monitor = None
        self.running = False
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logging.info("=" * 50)
        logging.info(f"{self.config.get('system.name')} v{self.config.get('system.version')}")
        logging.info("=" * 50)

    def setup_logging(self):
        log_dir = Path(self.config.get('storage.log_dir'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_level = getattr(logging, self.config.get('system.log_level', 'INFO'))
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'ai-box.log'),
                logging.StreamHandler()
            ]
        )

    def _signal_handler(self, sig, frame):
        logging.info(f"Received signal {sig}, shutting down...")
        self.stop()

    def start(self):
        logging.info("Starting AI Box system...")
        
        try:
            # 根据配置选择摄像头管理器
            if self.config.get('camera.high_performance', False):
                logging.info("Using high performance camera manager")
                self.camera_manager = HighPerformanceCameraManager(self.config)
            else:
                logging.info("Using standard camera manager")
                self.camera_manager = CameraManager(self.config)
            
            self.ai_analyzer = AIAnalyzer(self.config)
            self.web_server = WebServer(self.config, self.camera_manager, self.ai_analyzer)
            self.health_monitor = HealthMonitor(self.config, self)
            
            self.camera_manager.start()
            time.sleep(1)
            
            if self.config.get('ai.enabled'):
                self.ai_analyzer.start()
            
            if self.config.get('web.enabled'):
                self.web_server.start()
            
            self.health_monitor.start()
            
            self.running = True
            logging.info("AI Box system started successfully!")
            
            self._main_loop()
            
        except Exception as e:
            logging.error(f"Failed to start AI Box: {e}")
            self.stop()
            raise

    def _main_loop(self):
        while self.running:
            time.sleep(1)

    def stop(self):
        logging.info("Stopping AI Box system...")
        self.running = False
        
        if self.health_monitor:
            self.health_monitor.stop()
        
        if self.web_server:
            self.web_server.stop()
        
        if self.ai_analyzer:
            self.ai_analyzer.stop()
        
        if self.camera_manager:
            self.camera_manager.stop()
        
        logging.info("AI Box system stopped.")

    def restart(self):
        logging.info("Restarting AI Box system...")
        self.stop()
        time.sleep(2)
        self.start()

def main():
    config_path = os.getenv('AI_BOX_CONFIG', 'config/settings.yaml')
    
    ai_box = AIBox(config_path)
    ai_box.start()

if __name__ == '__main__':
    main()
