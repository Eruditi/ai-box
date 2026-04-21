#!/usr/bin/env python3
"""
共享状态管理器
用于多进程架构下的状态同步
支持 Redis 和文件两种模式
"""

import os
import json
import time
import logging
import threading
import pickle
from typing import Dict, Any, Optional, List
from pathlib import Path
from abc import ABC, abstractmethod


class BaseStateManager(ABC):
    """状态管理器基类"""
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        pass
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        pass
    
    @abstractmethod
    def setex(self, key: str, ttl: int, value: Any) -> bool:
        pass
    
    @abstractmethod
    def rpush(self, queue: str, value: Any) -> bool:
        pass
    
    @abstractmethod
    def lpop(self, queue: str) -> Optional[Any]:
        pass
    
    @abstractmethod
    def llen(self, queue: str) -> int:
        pass


class RedisStateManager(BaseStateManager):
    """Redis 状态管理器"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0, password: str = None):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.client = None
        self._connect()
    
    def _connect(self):
        """连接Redis"""
        try:
            import redis
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=False
            )
            self.client.ping()
            logging.info(f"[Redis状态] 连接成功: {self.host}:{self.port}")
        except Exception as e:
            logging.warning(f"[Redis状态] 连接失败: {e}")
            self.client = None
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        if not self.client:
            return False
        try:
            serialized = pickle.dumps(value)
            if ttl:
                return self.client.setex(key, ttl, serialized)
            return self.client.set(key, serialized)
        except Exception as e:
            logging.error(f"[Redis状态] 设置失败: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        if not self.client:
            return None
        try:
            value = self.client.get(key)
            if value:
                return pickle.loads(value)
            return None
        except Exception as e:
            logging.error(f"[Redis状态] 获取失败: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        if not self.client:
            return False
        try:
            return self.client.delete(key) > 0
        except Exception as e:
            logging.error(f"[Redis状态] 删除失败: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        if not self.client:
            return False
        try:
            return self.client.exists(key) > 0
        except Exception as e:
            logging.error(f"[Redis状态] 检查存在失败: {e}")
            return False
    
    def setex(self, key: str, ttl: int, value: Any) -> bool:
        return self.set(key, value, ttl)
    
    def rpush(self, queue: str, value: Any) -> bool:
        if not self.client:
            return False
        try:
            serialized = pickle.dumps(value)
            self.client.rpush(queue, serialized)
            return True
        except Exception as e:
            logging.error(f"[Redis状态] 推送队列失败: {e}")
            return False
    
    def lpop(self, queue: str) -> Optional[Any]:
        if not self.client:
            return None
        try:
            value = self.client.lpop(queue)
            if value:
                return pickle.loads(value)
            return None
        except Exception as e:
            logging.error(f"[Redis状态] 弹出队列失败: {e}")
            return None
    
    def llen(self, queue: str) -> int:
        if not self.client:
            return 0
        try:
            return self.client.llen(queue)
        except Exception as e:
            logging.error(f"[Redis状态] 获取队列长度失败: {e}")
            return 0


class FileStateManager(BaseStateManager):
    """文件状态管理器 - Redis不可用时的回退方案"""
    
    def __init__(self, data_dir: str = 'data/shared_state'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.cache: Dict[str, Any] = {}
        self.ttls: Dict[str, float] = {}
        logging.info(f"[文件状态] 初始化完成: {self.data_dir}")
    
    def _get_file_path(self, key: str) -> Path:
        safe_key = key.replace('/', '_').replace('\\', '_')
        return self.data_dir / f"{safe_key}.pkl"
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        with self.lock:
            try:
                file_path = self._get_file_path(key)
                with open(file_path, 'wb') as f:
                    pickle.dump(value, f)
                
                self.cache[key] = value
                if ttl:
                    self.ttls[key] = time.time() + ttl
                
                return True
            except Exception as e:
                logging.error(f"[文件状态] 设置失败: {e}")
                return False
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            try:
                if key in self.ttls:
                    if time.time() > self.ttls[key]:
                        self.delete(key)
                        return None
                
                if key in self.cache:
                    return self.cache[key]
                
                file_path = self._get_file_path(key)
                if file_path.exists():
                    with open(file_path, 'rb') as f:
                        value = pickle.load(f)
                        self.cache[key] = value
                        return value
                
                return None
            except Exception as e:
                logging.error(f"[文件状态] 获取失败: {e}")
                return None
    
    def delete(self, key: str) -> bool:
        with self.lock:
            try:
                file_path = self._get_file_path(key)
                if file_path.exists():
                    file_path.unlink()
                
                if key in self.cache:
                    del self.cache[key]
                if key in self.ttls:
                    del self.ttls[key]
                
                return True
            except Exception as e:
                logging.error(f"[文件状态] 删除失败: {e}")
                return False
    
    def exists(self, key: str) -> bool:
        return self.get(key) is not None
    
    def setex(self, key: str, ttl: int, value: Any) -> bool:
        return self.set(key, value, ttl)
    
    def rpush(self, queue: str, value: Any) -> bool:
        with self.lock:
            try:
                queue_file = self.data_dir / f"queue_{queue}.json"
                
                items = []
                if queue_file.exists():
                    with open(queue_file, 'r') as f:
                        items = json.load(f)
                
                items.append({
                    'value': pickle.dumps(value).hex(),
                    'timestamp': time.time()
                })
                
                with open(queue_file, 'w') as f:
                    json.dump(items, f)
                
                return True
            except Exception as e:
                logging.error(f"[文件状态] 推送队列失败: {e}")
                return False
    
    def lpop(self, queue: str) -> Optional[Any]:
        with self.lock:
            try:
                queue_file = self.data_dir / f"queue_{queue}.json"
                
                if not queue_file.exists():
                    return None
                
                with open(queue_file, 'r') as f:
                    items = json.load(f)
                
                if not items:
                    return None
                
                item = items.pop(0)
                
                with open(queue_file, 'w') as f:
                    json.dump(items, f)
                
                return pickle.loads(bytes.fromhex(item['value']))
            except Exception as e:
                logging.error(f"[文件状态] 弹出队列失败: {e}")
                return None
    
    def llen(self, queue: str) -> int:
        with self.lock:
            try:
                queue_file = self.data_dir / f"queue_{queue}.json"
                
                if not queue_file.exists():
                    return 0
                
                with open(queue_file, 'r') as f:
                    items = json.load(f)
                
                return len(items)
            except Exception as e:
                logging.error(f"[文件状态] 获取队列长度失败: {e}")
                return 0


class SharedStateManager:
    """共享状态管理器 - 自动选择最佳实现"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.backend: Optional[BaseStateManager] = None
        self._init_backend()
    
    def _init_backend(self):
        """初始化后端"""
        backend_type = self.config.get('backend', 'auto')
        
        if backend_type == 'redis' or backend_type == 'auto':
            redis_config = self.config.get('redis', {})
            try:
                self.backend = RedisStateManager(
                    host=redis_config.get('host', 'localhost'),
                    port=redis_config.get('port', 6379),
                    db=redis_config.get('db', 0),
                    password=redis_config.get('password')
                )
                
                if self.backend.client:
                    logging.info("[共享状态] 使用 Redis 后端")
                    return
            except Exception as e:
                logging.warning(f"[共享状态] Redis 初始化失败: {e}")
        
        if backend_type in ['file', 'auto']:
            file_config = self.config.get('file', {})
            self.backend = FileStateManager(
                data_dir=file_config.get('data_dir', 'data/shared_state')
            )
            logging.info("[共享状态] 使用文件后端")
    
    def update_cooldown(self, key: str, timestamp: float, ttl: int = 300) -> bool:
        """更新冷却期状态"""
        return self.backend.setex(f"cooldown:{key}", ttl, timestamp)
    
    def check_cooldown(self, key: str) -> Optional[float]:
        """检查冷却期"""
        return self.backend.get(f"cooldown:{key}")
    
    def push_retry_queue(self, alert: Dict[str, Any]) -> bool:
        """推送重试队列"""
        return self.backend.rpush("retry_queue", alert)
    
    def pop_retry_queue(self) -> Optional[Dict[str, Any]]:
        """弹出重试队列"""
        return self.backend.lpop("retry_queue")
    
    def get_retry_queue_length(self) -> int:
        """获取重试队列长度"""
        return self.backend.llen("retry_queue")
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        return self.backend.set(key, value, ttl)
    
    def get(self, key: str) -> Optional[Any]:
        return self.backend.get(key)
    
    def delete(self, key: str) -> bool:
        return self.backend.delete(key)
    
    def exists(self, key: str) -> bool:
        return self.backend.exists(key)


_shared_state_manager: Optional[SharedStateManager] = None


def get_shared_state_manager(config: Dict[str, Any] = None) -> SharedStateManager:
    """获取共享状态管理器单例"""
    global _shared_state_manager
    if _shared_state_manager is None:
        _shared_state_manager = SharedStateManager(config)
    return _shared_state_manager
