#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高性能存储管理器
支持RAID配置、存储监控、数据分层管理
"""
import os
import sys
import time
import shutil
import psutil
import subprocess
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAIDLevel(Enum):
    """RAID级别"""
    RAID0 = 0
    RAID1 = 1
    RAID5 = 5
    RAID6 = 6
    RAID10 = 10


class StorageTier(Enum):
    """存储层级"""
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


@dataclass
class DiskInfo:
    """磁盘信息"""
    device: str
    size_gb: float
    used_gb: float
    free_gb: float
    mount_point: str
    filesystem: str
    is_ssd: bool = False


@dataclass
class RAIDInfo:
    """RAID信息"""
    name: str
    level: RAIDLevel
    devices: List[str]
    size_gb: float
    status: str
    mount_point: str


@dataclass
class StoragePolicy:
    """存储策略"""
    tier: StorageTier
    retention_days: int
    compression: bool = False
    encryption: bool = False


class StorageManager:
    """高性能存储管理器"""
    
    def __init__(self, config: dict):
        """初始化存储管理器"""
        self.config = config
        self.storage_config = config.get('storage', {})
        self.raid_config = self.storage_config.get('raid', {})
        self.tiers_config = self.storage_config.get('tiers', {})
        
        self.raid_arrays: Dict[str, RAIDInfo] = {}
        self.disks: Dict[str, DiskInfo] = {}
        self.storage_policies: Dict[str, StoragePolicy] = {}
        
        self._init_default_policies()
        self._scan_disks()
        self._scan_raid()
        
        logger.info("存储管理器初始化完成")
    
    def _init_default_policies(self):
        """初始化默认存储策略"""
        self.storage_policies['video_hot'] = StoragePolicy(
            tier=StorageTier.HOT,
            retention_days=7,
            compression=False
        )
        self.storage_policies['video_warm'] = StoragePolicy(
            tier=StorageTier.WARM,
            retention_days=30,
            compression=True
        )
        self.storage_policies['video_cold'] = StoragePolicy(
            tier=StorageTier.COLD,
            retention_days=90,
            compression=True
        )
    
    def _scan_disks(self):
        """扫描系统磁盘"""
        try:
            partitions = psutil.disk_partitions()
            for part in partitions:
                try:
                    usage = psutil.disk_usage(part.mountpoint)
                    is_ssd = self._is_ssd(part.device)
                    
                    disk_info = DiskInfo(
                        device=part.device,
                        size_gb=usage.total / (1024 ** 3),
                        used_gb=usage.used / (1024 ** 3),
                        free_gb=usage.free / (1024 ** 3),
                        mount_point=part.mountpoint,
                        filesystem=part.fstype,
                        is_ssd=is_ssd
                    )
                    self.disks[part.device] = disk_info
                except Exception as e:
                    logger.warning(f"无法获取磁盘 {part.device} 信息: {e}")
        except Exception as e:
            logger.error(f"扫描磁盘失败: {e}")
    
    def _is_ssd(self, device: str) -> bool:
        """检查是否为SSD"""
        try:
            device_name = os.path.basename(device)
            rotational_path = f"/sys/block/{device_name}/queue/rotational"
            if os.path.exists(rotational_path):
                with open(rotational_path, 'r') as f:
                    return int(f.read().strip()) == 0
        except Exception:
            pass
        return False
    
    def _scan_raid(self):
        """扫描RAID阵列"""
        try:
            if os.path.exists('/proc/mdstat'):
                with open('/proc/mdstat', 'r') as f:
                    mdstat_content = f.read()
                
                self._parse_mdstat(mdstat_content)
        except Exception as e:
            logger.error(f"扫描RAID失败: {e}")
    
    def _parse_mdstat(self, content: str):
        """解析mdstat内容"""
        lines = content.split('\n')
        current_array = None
        
        for line in lines:
            if line.startswith('md') and ':' in line:
                parts = line.split(':')
                current_array = parts[0].strip()
                
                if 'raid0' in line.lower():
                    level = RAIDLevel.RAID0
                elif 'raid1' in line.lower():
                    level = RAIDLevel.RAID1
                elif 'raid5' in line.lower():
                    level = RAIDLevel.RAID5
                elif 'raid6' in line.lower():
                    level = RAIDLevel.RAID6
                elif 'raid10' in line.lower():
                    level = RAIDLevel.RAID10
                else:
                    level = RAIDLevel.RAID0
                
                devices = [dev for dev in line.split() if dev.startswith(('sd', 'nvme'))]
                
                self.raid_arrays[current_array] = RAIDInfo(
                    name=current_array,
                    level=level,
                    devices=devices,
                    size_gb=0,
                    status="active",
                    mount_point=""
                )
            
            elif current_array and 'active' in line:
                if 'UUU' in line or '_' not in line:
                    self.raid_arrays[current_array].status = "healthy"
                else:
                    self.raid_arrays[current_array].status = "degraded"
    
    def create_raid(self, name: str, level: RAIDLevel, devices: List[str], 
                   mount_point: str, filesystem: str = "ext4") -> bool:
        """创建RAID阵列"""
        try:
            logger.info(f"创建RAID {name}, 级别: {level}, 设备: {devices}")
            
            device_str = ' '.join(devices)
            
            cmd = f"mdadm --create --verbose /dev/{name} --level={level.value} " \
                  f"--raid-devices={len(devices)} {device_str}"
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"创建RAID失败: {result.stderr}")
                return False
            
            time.sleep(5)
            
            mkfs_cmd = f"mkfs.{filesystem} /dev/{name}"
            result = subprocess.run(mkfs_cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"格式化RAID失败: {result.stderr}")
                return False
            
            os.makedirs(mount_point, exist_ok=True)
            
            mount_cmd = f"mount /dev/{name} {mount_point}"
            result = subprocess.run(mount_cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"挂载RAID失败: {result.stderr}")
                return False
            
            self._scan_raid()
            logger.info(f"RAID {name} 创建成功")
            return True
            
        except Exception as e:
            logger.error(f"创建RAID异常: {e}")
            return False
    
    def get_storage_status(self) -> Dict:
        """获取存储状态"""
        total_gb = sum(d.size_gb for d in self.disks.values())
        used_gb = sum(d.used_gb for d in self.disks.values())
        free_gb = sum(d.free_gb for d in self.disks.values())
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_gb': round(total_gb, 2),
            'used_gb': round(used_gb, 2),
            'free_gb': round(free_gb, 2),
            'usage_percent': round((used_gb / total_gb * 100) if total_gb > 0 else 0, 2),
            'disks': [
                {
                    'device': d.device,
                    'size_gb': round(d.size_gb, 2),
                    'used_gb': round(d.used_gb, 2),
                    'free_gb': round(d.free_gb, 2),
                    'mount_point': d.mount_point,
                    'filesystem': d.filesystem,
                    'is_ssd': d.is_ssd
                }
                for d in self.disks.values()
            ],
            'raid_arrays': [
                {
                    'name': r.name,
                    'level': r.level.name,
                    'devices': r.devices,
                    'size_gb': round(r.size_gb, 2),
                    'status': r.status,
                    'mount_point': r.mount_point
                }
                for r in self.raid_arrays.values()
            ]
        }
    
    def manage_video_storage(self, video_path: str, policy_name: str = 'video_warm') -> bool:
        """管理视频存储"""
        try:
            if policy_name not in self.storage_policies:
                logger.error(f"未知的存储策略: {policy_name}")
                return False
            
            policy = self.storage_policies[policy_name]
            
            if not os.path.exists(video_path):
                logger.warning(f"视频路径不存在: {video_path}")
                return False
            
            self._cleanup_old_videos(video_path, policy.retention_days)
            
            return True
            
        except Exception as e:
            logger.error(f"管理视频存储失败: {e}")
            return False
    
    def _cleanup_old_videos(self, video_path: str, retention_days: int):
        """清理过期视频"""
        try:
            now = time.time()
            cutoff = now - (retention_days * 86400)
            
            cleaned_count = 0
            cleaned_size = 0
            
            for root, dirs, files in os.walk(video_path):
                for file in files:
                    if file.endswith(('.mp4', '.avi', '.mkv', '.h264', '.h265')):
                        file_path = os.path.join(root, file)
                        try:
                            file_mtime = os.path.getmtime(file_path)
                            if file_mtime < cutoff:
                                file_size = os.path.getsize(file_path)
                                os.remove(file_path)
                                cleaned_count += 1
                                cleaned_size += file_size
                                logger.debug(f"删除过期视频: {file_path}")
                        except Exception as e:
                            logger.warning(f"删除文件失败 {file_path}: {e}")
            
            if cleaned_count > 0:
                logger.info(f"清理了 {cleaned_count} 个过期视频, 释放 {cleaned_size / (1024**3):.2f} GB")
            
        except Exception as e:
            logger.error(f"清理过期视频失败: {e}")
    
    def get_available_storage(self, path: str = '/') -> float:
        """获取可用存储空间(GB)"""
        try:
            usage = psutil.disk_usage(path)
            return usage.free / (1024 ** 3)
        except Exception as e:
            logger.error(f"获取可用存储空间失败: {e}")
            return 0.0
    
    def check_storage_health(self) -> Dict:
        """检查存储健康状态"""
        status = {
            'overall_status': 'healthy',
            'issues': [],
            'warnings': []
        }
        
        for disk in self.disks.values():
            usage_percent = (disk.used_gb / disk.size_gb * 100) if disk.size_gb > 0 else 0
            
            if usage_percent > 90:
                status['issues'].append(f"磁盘 {disk.device} 使用率过高: {usage_percent:.1f}%")
                status['overall_status'] = 'critical'
            elif usage_percent > 80:
                status['warnings'].append(f"磁盘 {disk.device} 使用率警告: {usage_percent:.1f}%")
                if status['overall_status'] == 'healthy':
                    status['overall_status'] = 'warning'
        
        for raid in self.raid_arrays.values():
            if raid.status == 'degraded':
                status['issues'].append(f"RAID阵列 {raid.name} 状态降级")
                status['overall_status'] = 'critical'
        
        return status


if __name__ == '__main__':
    import yaml
    import os
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'config', 'settings_high_performance.yaml')
    
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        print("使用默认配置进行测试...")
        config = {}
    else:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    
    storage_manager = StorageManager(config)
    
    print("=== 存储状态 ===")
    status = storage_manager.get_storage_status()
    print(f"总容量: {status['total_gb']} GB")
    print(f"已使用: {status['used_gb']} GB")
    print(f"可用: {status['free_gb']} GB")
    print(f"使用率: {status['usage_percent']}%")
    
    print("\n=== 存储健康检查 ===")
    health = storage_manager.check_storage_health()
    print(f"整体状态: {health['overall_status']}")
    if health['issues']:
        print("问题:")
        for issue in health['issues']:
            print(f"  - {issue}")
    if health['warnings']:
        print("警告:")
        for warning in health['warnings']:
            print(f"  - {warning}")
    
    print("\n=== 磁盘列表 ===")
    for disk_info in status['disks']:
        print(f"  {disk_info['device']}: {disk_info['size_gb']} GB, "
              f"挂载点: {disk_info['mount_point']}, "
              f"{'SSD' if disk_info['is_ssd'] else 'HDD'}")
