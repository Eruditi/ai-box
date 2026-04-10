#!/usr/bin/env python3
"""
健康监控模块
监控系统健康状态，自动恢复
"""

import time
import logging
import threading
import psutil


class HealthMonitor:
    def __init__(self, config, ai_box):
        self.config = config
        self.ai_box = ai_box
        self.running = False
        self.thread = None
        self.check_interval = config.get('health.check_interval', 30)
        self.auto_restart = config.get('health.auto_restart', True)

    def get_system_stats(self) -> dict:
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_mb': memory.used / 1024 / 1024,
            'memory_total_mb': memory.total / 1024 / 1024,
            'disk_percent': disk.percent,
            'disk_used_gb': disk.used / 1024 / 1024 / 1024,
            'disk_total_gb': disk.total / 1024 / 1024 / 1024
        }

    def _check_health(self):
        try:
            stats = self.get_system_stats()
            
            logging.debug(f"Health check - CPU: {stats['cpu_percent']}%, "
                         f"Memory: {stats['memory_percent']}%, "
                         f"Disk: {stats['disk_percent']}%")
            
            if stats['memory_percent'] > 90:
                logging.warning(f"High memory usage: {stats['memory_percent']}%")
                if self.auto_restart:
                    logging.warning("Memory critical, attempting restart...")
                    threading.Thread(target=self.ai_box.restart, daemon=True).start()
            
            if stats['disk_percent'] > 90:
                logging.error(f"Critical disk usage: {stats['disk_percent']}%")
            
        except Exception as e:
            logging.error(f"Health check error: {e}")

    def _monitor_loop(self):
        while self.running:
            self._check_health()
            time.sleep(self.check_interval)

    def start(self):
        logging.info("Starting health monitor...")
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logging.info("Health monitor started")

    def stop(self):
        logging.info("Stopping health monitor...")
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        logging.info("Health monitor stopped")
