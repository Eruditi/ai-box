#!/usr/bin/env python3
"""
配置管理模块
支持热配置更新
"""

import os
import yaml
import time
import threading
from pathlib import Path
from typing import Any, List, Callable


class ConfigManager:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.last_modified_time = self._get_modified_time()
        self.callbacks: List[Callable] = []
        self.monitor_thread = None
        self.running = False
        
    def _load_config(self) -> dict:
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value

    def set(self, key: str, value: Any):
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self._save_config()

    def _save_config(self):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True)
        self.last_modified_time = self._get_modified_time()
        self._notify_callbacks()
    
    def _get_modified_time(self) -> float:
        if self.config_path.exists():
            return self.config_path.stat().st_mtime
        return 0
    
    def add_callback(self, callback: Callable):
        """添加配置变化回调"""
        if callback not in self.callbacks:
            self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable):
        """移除配置变化回调"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def _notify_callbacks(self):
        """通知所有回调配置已变化"""
        for callback in self.callbacks:
            try:
                callback(self.config)
            except Exception as e:
                logging.error(f"Error in config callback: {e}")
    
    def start_monitoring(self):
        """开始监控配置文件变化"""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_config, daemon=True)
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控配置文件变化"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
    
    def _monitor_config(self):
        """监控配置文件变化"""
        while self.running:
            try:
                current_modified = self._get_modified_time()
                if current_modified > self.last_modified_time:
                    # 配置文件已修改
                    new_config = self._load_config()
                    if new_config != self.config:
                        self.config = new_config
                        self.last_modified_time = current_modified
                        logging.info("Config file updated, reloading...")
                        self._notify_callbacks()
                time.sleep(2)  # 每2秒检查一次
            except Exception as e:
                logging.error(f"Error monitoring config: {e}")
                time.sleep(5)
    
    def reload(self):
        """手动重新加载配置"""
        self.config = self._load_config()
        self.last_modified_time = self._get_modified_time()
        self._notify_callbacks()
        return self.config
