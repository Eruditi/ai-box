#!/usr/bin/env python3
"""
告警数据库 - 支持 SQLite 和 PostgreSQL
提供统一的数据库抽象层
"""

import os
import time
import logging
import threading
import sqlite3
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
from abc import ABC, abstractmethod


class BaseDatabase(ABC):
    """数据库基类"""
    
    @abstractmethod
    def add_alert(self, algorithm_id: int, algorithm_name: str, category: str,
                  camera_source: str, confidence: float, bbox: str = None,
                  extra_data: str = None):
        pass
    
    @abstractmethod
    def get_alerts(self, limit: int = 100, offset: int = 0,
                   camera_source: str = None, algorithm_id: int = None,
                   since: float = None, date: str = None, alert_type: str = None,
                   start_time: float = None, end_time: float = None) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def get_alert_count(self, since: float = None) -> int:
        pass
    
    @abstractmethod
    def get_today_stats(self) -> Dict[str, int]:
        pass
    
    @abstractmethod
    def get_month_stats(self) -> int:
        pass
    
    @abstractmethod
    def cleanup_old_records(self, days: int = 90):
        pass


class SQLiteDatabase(BaseDatabase):
    """SQLite 数据库实现"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, db_path: str = None):
        if self._initialized:
            return
        self._initialized = True

        if db_path is None:
            db_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, 'alerts.db')

        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, timeout=10)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=268435456")
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                datetime TEXT NOT NULL,
                algorithm_id INTEGER NOT NULL,
                algorithm_name TEXT NOT NULL,
                category TEXT NOT NULL,
                camera_source TEXT NOT NULL,
                confidence REAL NOT NULL,
                detected INTEGER NOT NULL DEFAULT 1,
                bbox TEXT,
                extra_data TEXT
            )
        ''')
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)
        ''')
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_alerts_algorithm_id ON alerts(algorithm_id)
        ''')
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_alerts_camera ON alerts(camera_source)
        ''')
        conn.commit()

    def add_alert(self, algorithm_id: int, algorithm_name: str, category: str,
                  camera_source: str, confidence: float, bbox: str = None,
                  extra_data: str = None):
        """添加告警记录"""
        try:
            conn = self._get_conn()
            now = time.time()
            dt_str = datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S')
            conn.execute('''
                INSERT INTO alerts (timestamp, datetime, algorithm_id, algorithm_name,
                                    category, camera_source, confidence, detected, bbox, extra_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            ''', (now, dt_str, algorithm_id, algorithm_name, category,
                  camera_source, confidence, bbox, extra_data))
            conn.commit()
        except Exception as e:
            logging.error(f"[SQLite] 写入失败: {e}")

    def get_alerts(self, limit: int = 100, offset: int = 0,
                   camera_source: str = None, algorithm_id: int = None,
                   since: float = None, date: str = None, alert_type: str = None,
                   start_time: float = None, end_time: float = None) -> List[Dict[str, Any]]:
        """查询告警记录"""
        try:
            conn = self._get_conn()
            conditions = []
            params = []

            if camera_source:
                conditions.append("camera_source = ?")
                params.append(camera_source)
            if algorithm_id:
                conditions.append("algorithm_id = ?")
                params.append(algorithm_id)
            if since:
                conditions.append("timestamp >= ?")
                params.append(since)
            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time)
            if end_time:
                conditions.append("timestamp < ?")
                params.append(end_time)
            if date:
                conditions.append("date(datetime) = ?")
                params.append(date)
            if alert_type:
                type_category_map = {
                    'face_recognition': ['FACE_RECOGNITION'],
                    'structured_analysis': ['STRUCTURED_ANALYSIS'],
                    'person_violation': ['PERSON_VIOLATION'],
                    'environment_abnormal': ['ENVIRONMENT_ABNORMAL'],
                    'perimeter_alert': ['PERIMETER_ALERT'],
                    'behavior_alert': ['BEHAVIOR_ALERT'],
                }
                categories = type_category_map.get(alert_type, [alert_type.upper()])
                placeholders = ','.join(['?' for _ in categories])
                conditions.append(f"category IN ({placeholders})")
                params.extend(categories)

            where = " WHERE " + " AND ".join(conditions) if conditions else ""
            query = f"SELECT * FROM alerts{where} ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            rows = conn.execute(query, params).fetchall()
            results = []
            for row in rows:
                r = dict(row)
                results.append({
                    'id': r['id'],
                    'time': r['datetime'],
                    'timestamp': r['timestamp'],
                    'location': r['camera_source'],
                    'source': r['camera_source'],
                    'camera_name': r['camera_source'],
                    'type': r['algorithm_name'],
                    'algorithm_name': r['algorithm_name'],
                    'algorithm_id': r['algorithm_id'],
                    'category': r['category'],
                    'confidence': r['confidence'],
                    'bbox': r['bbox'],
                    'extra_data': r['extra_data']
                })
            return results
        except Exception as e:
            logging.error(f"[SQLite] 查询失败: {e}")
            return []

    def get_alert_count(self, since: float = None) -> int:
        """获取告警数量"""
        try:
            conn = self._get_conn()
            if since:
                row = conn.execute("SELECT COUNT(*) FROM alerts WHERE timestamp >= ?",
                                   (since,)).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()
            return row[0] if row else 0
        except Exception as e:
            logging.error(f"[SQLite] 统计失败: {e}")
            return 0

    def get_today_stats(self) -> Dict[str, int]:
        """获取今日统计"""
        try:
            today_start = datetime.now().replace(hour=0, minute=0, second=0,
                                                  microsecond=0).timestamp()
            conn = self._get_conn()
            total = conn.execute("SELECT COUNT(*) FROM alerts WHERE timestamp >= ?",
                                 (today_start,)).fetchone()[0]

            by_category = {}
            rows = conn.execute('''
                SELECT category, COUNT(*) as cnt FROM alerts
                WHERE timestamp >= ? GROUP BY category
            ''', (today_start,)).fetchall()
            for row in rows:
                by_category[row[0]] = row[1]

            return {'todayAlerts': total, 'byCategory': by_category}
        except Exception as e:
            logging.error(f"[SQLite] 今日统计失败: {e}")
            return {'todayAlerts': 0, 'byCategory': {}}

    def get_month_stats(self) -> int:
        """获取本月告警数"""
        try:
            month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0,
                                                  microsecond=0).timestamp()
            conn = self._get_conn()
            return conn.execute("SELECT COUNT(*) FROM alerts WHERE timestamp >= ?",
                                (month_start,)).fetchone()[0]
        except Exception as e:
            return 0

    def add_alert_summary(self, summary: Dict[str, Any]) -> bool:
        """添加告警摘要"""
        try:
            conn = self._get_conn()
            conn.execute('''
                CREATE TABLE IF NOT EXISTS alert_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    summary TEXT,
                    timestamp REAL,
                    alert_count INTEGER,
                    camera_sources TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.execute('''
                INSERT INTO alert_summaries (summary, timestamp, alert_count, camera_sources)
                VALUES (?, ?, ?, ?)
            ''', (
                summary.get('summary', ''),
                summary.get('timestamp', time.time()),
                summary.get('alert_count', 0),
                summary.get('camera_sources', '')
            ))
            conn.commit()
            return True
        except Exception as e:
            logging.error(f"[SQLite] 写入摘要失败: {e}")
            return False

    def test_connection(self) -> bool:
        """测试数据库连接"""
        try:
            conn = self._get_conn()
            conn.execute("SELECT 1")
            return True
        except Exception as e:
            logging.error(f"[SQLite] 连接测试失败: {e}")
            return False

    def cleanup_old_records(self, days: int = 90):
        """清理旧记录"""
        try:
            cutoff = time.time() - days * 86400
            conn = self._get_conn()
            conn.execute("DELETE FROM alerts WHERE timestamp < ?", (cutoff,))
            conn.commit()
        except Exception as e:
            logging.error(f"[SQLite] 清理失败: {e}")


class PostgreSQLDatabase(BaseDatabase):
    """PostgreSQL 数据库实现"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', 5432)
        self.database = config.get('database', 'ai_box')
        self.user = config.get('user', 'postgres')
        self.password = config.get('password', '')
        self._pool = None
        self._init_db()
    
    def _get_connection(self):
        """获取数据库连接"""
        try:
            import psycopg2
            from psycopg2 import pool
            
            if self._pool is None:
                self._pool = pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=10,
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    user=self.user,
                    password=self.password
                )
            
            return self._pool.getconn()
        except ImportError:
            logging.error("[PostgreSQL] psycopg2 未安装，请运行: pip install psycopg2-binary")
            return None
        except Exception as e:
            logging.error(f"[PostgreSQL] 连接失败: {e}")
            return None
    
    def _return_connection(self, conn):
        """归还连接"""
        if self._pool and conn:
            self._pool.putconn(conn)
    
    def _init_db(self):
        """初始化数据库表"""
        conn = self._get_connection()
        if not conn:
            return
        
        try:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id SERIAL PRIMARY KEY,
                    timestamp DOUBLE PRECISION NOT NULL,
                    datetime TIMESTAMP NOT NULL,
                    algorithm_id INTEGER NOT NULL,
                    algorithm_name VARCHAR(255) NOT NULL,
                    category VARCHAR(100) NOT NULL,
                    camera_source VARCHAR(500) NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL,
                    detected INTEGER NOT NULL DEFAULT 1,
                    bbox TEXT,
                    extra_data TEXT
                )
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_alerts_algorithm_id ON alerts(algorithm_id)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_alerts_camera ON alerts(camera_source)
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS inspection_reports (
                    id SERIAL PRIMARY KEY,
                    report_id VARCHAR(100) UNIQUE NOT NULL,
                    report_type VARCHAR(20) NOT NULL,
                    period VARCHAR(100) NOT NULL,
                    generated_at TIMESTAMP NOT NULL,
                    summary TEXT,
                    total_alerts INTEGER DEFAULT 0,
                    alerts_by_type JSONB,
                    alerts_by_camera JSONB,
                    patterns JSONB,
                    recommendations JSONB,
                    risk_level VARCHAR(20)
                )
            ''')
            
            conn.commit()
            logging.info("[PostgreSQL] 数据库初始化完成")
        except Exception as e:
            logging.error(f"[PostgreSQL] 初始化失败: {e}")
        finally:
            self._return_connection(conn)
    
    def add_alert(self, algorithm_id: int, algorithm_name: str, category: str,
                  camera_source: str, confidence: float, bbox: str = None,
                  extra_data: str = None):
        """添加告警记录"""
        conn = self._get_connection()
        if not conn:
            return
        
        try:
            cursor = conn.cursor()
            now = time.time()
            dt = datetime.fromtimestamp(now)
            
            cursor.execute('''
                INSERT INTO alerts (timestamp, datetime, algorithm_id, algorithm_name,
                                    category, camera_source, confidence, detected, bbox, extra_data)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 1, %s, %s)
            ''', (now, dt, algorithm_id, algorithm_name, category,
                  camera_source, confidence, bbox, extra_data))
            conn.commit()
        except Exception as e:
            logging.error(f"[PostgreSQL] 写入失败: {e}")
        finally:
            self._return_connection(conn)
    
    def get_alerts(self, limit: int = 100, offset: int = 0,
                   camera_source: str = None, algorithm_id: int = None,
                   since: float = None, date: str = None, alert_type: str = None,
                   start_time: float = None, end_time: float = None) -> List[Dict[str, Any]]:
        """查询告警记录"""
        conn = self._get_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            conditions = []
            params = []

            if camera_source:
                conditions.append("camera_source = %s")
                params.append(camera_source)
            if algorithm_id:
                conditions.append("algorithm_id = %s")
                params.append(algorithm_id)
            if since:
                conditions.append("timestamp >= %s")
                params.append(since)
            if start_time:
                conditions.append("timestamp >= %s")
                params.append(start_time)
            if end_time:
                conditions.append("timestamp < %s")
                params.append(end_time)
            if date:
                conditions.append("DATE(datetime) = %s")
                params.append(date)
            if alert_type:
                type_category_map = {
                    'face_recognition': ['FACE_RECOGNITION'],
                    'structured_analysis': ['STRUCTURED_ANALYSIS'],
                    'person_violation': ['PERSON_VIOLATION'],
                    'environment_abnormal': ['ENVIRONMENT_ABNORMAL'],
                    'perimeter_alert': ['PERIMETER_ALERT'],
                    'behavior_alert': ['BEHAVIOR_ALERT'],
                }
                categories = type_category_map.get(alert_type, [alert_type.upper()])
                placeholders = ','.join(['%s' for _ in categories])
                conditions.append(f"category IN ({placeholders})")
                params.extend(categories)

            where = " WHERE " + " AND ".join(conditions) if conditions else ""
            query = f"SELECT * FROM alerts{where} ORDER BY timestamp DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])

            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            columns = [desc[0] for desc in cursor.description]
            results = []
            for row in rows:
                r = dict(zip(columns, row))
                results.append({
                    'id': r['id'],
                    'time': r['datetime'].strftime('%Y-%m-%d %H:%M:%S') if r.get('datetime') else '',
                    'timestamp': r['timestamp'],
                    'location': r['camera_source'],
                    'source': r['camera_source'],
                    'camera_name': r['camera_source'],
                    'type': r['algorithm_name'],
                    'algorithm_name': r['algorithm_name'],
                    'algorithm_id': r['algorithm_id'],
                    'category': r['category'],
                    'confidence': r['confidence'],
                    'bbox': r['bbox'],
                    'extra_data': r['extra_data']
                })
            return results
        except Exception as e:
            logging.error(f"[PostgreSQL] 查询失败: {e}")
            return []
        finally:
            self._return_connection(conn)
    
    def get_alert_count(self, since: float = None) -> int:
        """获取告警数量"""
        conn = self._get_connection()
        if not conn:
            return 0
        
        try:
            cursor = conn.cursor()
            if since:
                cursor.execute("SELECT COUNT(*) FROM alerts WHERE timestamp >= %s", (since,))
            else:
                cursor.execute("SELECT COUNT(*) FROM alerts")
            return cursor.fetchone()[0]
        except Exception as e:
            logging.error(f"[PostgreSQL] 统计失败: {e}")
            return 0
        finally:
            self._return_connection(conn)
    
    def get_today_stats(self) -> Dict[str, int]:
        """获取今日统计"""
        conn = self._get_connection()
        if not conn:
            return {'todayAlerts': 0, 'byCategory': {}}
        
        try:
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM alerts WHERE datetime >= %s", (today_start,))
            total = cursor.fetchone()[0]

            cursor.execute('''
                SELECT category, COUNT(*) as cnt FROM alerts
                WHERE datetime >= %s GROUP BY category
            ''', (today_start,))
            rows = cursor.fetchall()
            
            by_category = {row[0]: row[1] for row in rows}
            return {'todayAlerts': total, 'byCategory': by_category}
        except Exception as e:
            logging.error(f"[PostgreSQL] 今日统计失败: {e}")
            return {'todayAlerts': 0, 'byCategory': {}}
        finally:
            self._return_connection(conn)
    
    def get_month_stats(self) -> int:
        """获取本月告警数"""
        conn = self._get_connection()
        if not conn:
            return 0
        
        try:
            month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM alerts WHERE datetime >= %s", (month_start,))
            return cursor.fetchone()[0]
        except Exception as e:
            return 0
        finally:
            self._return_connection(conn)
    
    def cleanup_old_records(self, days: int = 90):
        """清理旧记录"""
        conn = self._get_connection()
        if not conn:
            return
        
        try:
            cutoff = time.time() - days * 86400
            cursor = conn.cursor()
            cursor.execute("DELETE FROM alerts WHERE timestamp < %s", (cutoff,))
            conn.commit()
        except Exception as e:
            logging.error(f"[PostgreSQL] 清理失败: {e}")
        finally:
            self._return_connection(conn)
    
    def save_report(self, report_data: Dict[str, Any]) -> bool:
        """保存巡检报告"""
        conn = self._get_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            import json
            
            cursor.execute('''
                INSERT INTO inspection_reports 
                (report_id, report_type, period, generated_at, summary, total_alerts,
                 alerts_by_type, alerts_by_camera, patterns, recommendations, risk_level)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (report_id) DO UPDATE SET
                    summary = EXCLUDED.summary,
                    total_alerts = EXCLUDED.total_alerts,
                    alerts_by_type = EXCLUDED.alerts_by_type,
                    alerts_by_camera = EXCLUDED.alerts_by_camera,
                    patterns = EXCLUDED.patterns,
                    recommendations = EXCLUDED.recommendations,
                    risk_level = EXCLUDED.risk_level
            ''', (
                report_data['report_id'],
                report_data.get('report_type', 'daily'),
                report_data['period'],
                report_data['generated_at'],
                report_data['summary'],
                report_data['total_alerts'],
                json.dumps(report_data['alerts_by_type']),
                json.dumps(report_data['alerts_by_camera']),
                json.dumps(report_data['patterns']),
                json.dumps(report_data['recommendations']),
                report_data['risk_level']
            ))
            conn.commit()
            return True
        except Exception as e:
            logging.error(f"[PostgreSQL] 保存报告失败: {e}")
            return False
        finally:
            self._return_connection(conn)


def get_alert_db(config: Dict[str, Any] = None, db_path: str = None) -> BaseDatabase:
    """获取告警数据库实例
    
    Args:
        config: 配置字典，包含数据库类型和连接信息
        db_path: SQLite 数据库路径（可选）
    
    Returns:
        数据库实例
    """
    if config is None:
        config = {}
    
    db_type = config.get('database.type', 'sqlite')
    
    if db_type == 'postgresql':
        pg_config = {
            'host': config.get('database.host', 'localhost'),
            'port': config.get('database.port', 5432),
            'database': config.get('database.name', 'ai_box'),
            'user': config.get('database.user', 'postgres'),
            'password': config.get('database.password', ''),
        }
        try:
            return PostgreSQLDatabase(pg_config)
        except Exception as e:
            logging.warning(f"[数据库] PostgreSQL 初始化失败，回退到 SQLite: {e}")
            return SQLiteDatabase(db_path)
    else:
        return SQLiteDatabase(db_path)


AlertDatabase = SQLiteDatabase
