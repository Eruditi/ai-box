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
        
        # 磁盘监控 - Windows 兼容
        try:
            import os
            if os.name == 'nt':  # Windows
                disk = psutil.disk_usage('C:\\')
            else:  # Linux/Unix
                disk = psutil.disk_usage('/')
        except Exception:
            disk = psutil.disk_usage('/')  # fallback
        
        # 网络监控
        try:
            net_io = psutil.net_io_counters()
            net_stats = {
                'bytes_sent_mb': net_io.bytes_sent / 1024 / 1024,
                'bytes_recv_mb': net_io.bytes_recv / 1024 / 1024,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'errin': net_io.errin,
                'errout': net_io.errout
            }
        except Exception:
            net_stats = {'error': 'Network stats unavailable'}
        
        # 进程监控
        process_stats = {
            'process_count': len(psutil.pids()),
            'current_process_memory_mb': psutil.Process().memory_info().rss / 1024 / 1024
        }
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_mb': memory.used / 1024 / 1024,
            'memory_total_mb': memory.total / 1024 / 1024,
            'disk_percent': disk.percent,
            'disk_used_gb': disk.used / 1024 / 1024 / 1024,
            'disk_total_gb': disk.total / 1024 / 1024 / 1024,
            'network': net_stats,
            'process': process_stats
        }

    def _check_health(self):
        try:
            stats = self.get_system_stats()
            
            # 详细的健康状态日志
            logging.debug(f"Health check - CPU: {stats['cpu_percent']}%, "
                         f"Memory: {stats['memory_percent']}%, "
                         f"Disk: {stats['disk_percent']}%, "
                         f"Processes: {stats['process']['process_count']}")
            
            # 健康状态检查
            issues = []
            
            # CPU检查
            if stats['cpu_percent'] > 90:
                issues.append(f"High CPU usage: {stats['cpu_percent']}%")
                logging.warning(issues[-1])
            
            # 内存检查
            if stats['memory_percent'] > 90:
                issues.append(f"High memory usage: {stats['memory_percent']}%")
                logging.warning(issues[-1])
                if self.auto_restart:
                    logging.warning("Memory critical, attempting restart...")
                    threading.Thread(target=self.ai_box.restart, daemon=True).start()
            
            # 磁盘检查
            if stats['disk_percent'] > 90:
                issues.append(f"Critical disk usage: {stats['disk_percent']}%")
                logging.error(issues[-1])
            
            # 网络检查
            if 'network' in stats and 'errin' in stats['network']:
                if stats['network']['errin'] > 100 or stats['network']['errout'] > 100:
                    issues.append(f"Network errors detected: in={stats['network']['errin']}, out={stats['network']['errout']}")
                    logging.warning(issues[-1])
            
            # 摄像头状态检查
            if hasattr(self.ai_box, 'camera_manager') and self.ai_box.camera_manager:
                cameras = self.ai_box.camera_manager.get_all_cameras()
                connected_cameras = [cam for cam in cameras if cam.connected]
                logging.debug(f"Camera status: {len(connected_cameras)}/{len(cameras)} connected")
                if len(connected_cameras) < len(cameras):
                    issues.append(f"Some cameras disconnected: {len(connected_cameras)}/{len(cameras)} connected")
                    logging.warning(issues[-1])
            
            # 多进程状态检查
            if hasattr(self.ai_box, 'process_manager') and self.ai_box.process_manager:
                active_cameras = self.ai_box.process_manager.get_active_cameras()
                logging.debug(f"Multiprocess status: {len(active_cameras)} active camera processes")
            
            # 记录健康状态
            if issues:
                logging.info(f"Health check found {len(issues)} issues: {'; '.join(issues)}")
            else:
                logging.debug("Health check: All systems normal")
            
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
