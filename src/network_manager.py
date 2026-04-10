#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网络接口管理器
支持多址设定、负载均衡、主备模式
"""
import os
import subprocess
import re
import time
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


class NetworkMode(Enum):
    """网络模式"""
    MULTI_ADDRESS = "multi_address"
    LOAD_BALANCING = "load_balancing"
    ACTIVE_BACKUP = "active_backup"


class InterfaceStatus(Enum):
    """接口状态"""
    UP = "up"
    DOWN = "down"
    UNKNOWN = "unknown"


@dataclass
class NetworkInterface:
    """网络接口信息"""
    name: str
    status: InterfaceStatus
    mac_address: str
    ip_addresses: List[str]
    netmask: str
    gateway: str
    mtu: int
    speed: str
    is_primary: bool = False


@dataclass
class BondConfig:
    """Bonding配置"""
    name: str
    mode: NetworkMode
    slaves: List[str]
    primary: Optional[str] = None
    miimon: int = 100
    updelay: int = 200
    downdelay: int = 200


class NetworkManager:
    """网络接口管理器"""
    
    def __init__(self, config: dict):
        """初始化网络管理器"""
        self.config = config
        self.network_config = config.get('network', {})
        self.interfaces_config = self.network_config.get('interfaces', [])
        
        self.interfaces: Dict[str, NetworkInterface] = {}
        self.bonds: Dict[str, BondConfig] = {}
        
        self._scan_interfaces()
        self._load_bonds()
        
        logger.info("网络管理器初始化完成")
    
    def _scan_interfaces(self):
        """扫描网络接口"""
        try:
            result = subprocess.run(['ip', 'link', 'show'], capture_output=True, text=True)
            if result.returncode == 0:
                self._parse_interfaces(result.stdout)
        except Exception as e:
            logger.error(f"扫描网络接口失败: {e}")
    
    def _parse_interfaces(self, output: str):
        """解析接口信息"""
        current_interface = None
        
        lines = output.split('\n')
        for line in lines:
            match = re.match(r'^(\d+): ([a-zA-Z0-9]+):', line)
            if match:
                if current_interface:
                    self.interfaces[current_interface.name] = current_interface
                
                name = match.group(2)
                status = InterfaceStatus.UNKNOWN
                if 'UP' in line:
                    status = InterfaceStatus.UP
                elif 'DOWN' in line:
                    status = InterfaceStatus.DOWN
                
                mac_match = re.search(r'link/ether ([0-9a-fA-F:]+)', line)
                mac_address = mac_match.group(1) if mac_match else ""
                
                current_interface = NetworkInterface(
                    name=name,
                    status=status,
                    mac_address=mac_address,
                    ip_addresses=[],
                    netmask="",
                    gateway="",
                    mtu=1500,
                    speed="",
                    is_primary=False
                )
                
                mtu_match = re.search(r'mtu (\d+)', line)
                if mtu_match:
                    current_interface.mtu = int(mtu_match.group(1))
            
            elif current_interface:
                inet_match = re.search(r'inet (\d+\.\d+\.\d+\.\d+)/(\d+)', line)
                if inet_match:
                    ip = inet_match.group(1)
                    prefix = int(inet_match.group(2))
                    current_interface.ip_addresses.append(ip)
                    current_interface.netmask = self._prefix_to_netmask(prefix)
        
        if current_interface:
            self.interfaces[current_interface.name] = current_interface
        
        self._get_gateways()
        self._get_interface_speeds()
    
    def _prefix_to_netmask(self, prefix: int) -> str:
        """前缀转子网掩码"""
        mask = (0xffffffff << (32 - prefix)) & 0xffffffff
        return f"{(mask >> 24) & 0xff}.{(mask >> 16) & 0xff}.{(mask >> 8) & 0xff}.{mask & 0xff}"
    
    def _get_gateways(self):
        """获取网关信息"""
        try:
            result = subprocess.run(['ip', 'route', 'show'], capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.startswith('default'):
                        parts = line.split()
                        if len(parts) >= 5:
                            gateway = parts[2]
                            interface = parts[4]
                            if interface in self.interfaces:
                                self.interfaces[interface].gateway = gateway
        except Exception as e:
            logger.error(f"获取网关信息失败: {e}")
    
    def _get_interface_speeds(self):
        """获取接口速度"""
        for name, interface in self.interfaces.items():
            try:
                speed_path = f"/sys/class/net/{name}/speed"
                if os.path.exists(speed_path):
                    with open(speed_path, 'r') as f:
                        speed = f.read().strip()
                        if speed.isdigit():
                            interface.speed = f"{speed} Mbps"
            except Exception:
                pass
    
    def _load_bonds(self):
        """加载Bonding配置"""
        try:
            if os.path.exists('/proc/net/bonding'):
                for bond_name in os.listdir('/proc/net/bonding'):
                    bond_path = f"/proc/net/bonding/{bond_name}"
                    if os.path.isfile(bond_path):
                        self._parse_bond(bond_name, bond_path)
        except Exception as e:
            logger.error(f"加载Bonding配置失败: {e}")
    
    def _parse_bond(self, name: str, path: str):
        """解析Bonding配置"""
        try:
            with open(path, 'r') as f:
                content = f.read()
            
            mode = NetworkMode.ACTIVE_BACKUP
            if 'balance-rr' in content:
                mode = NetworkMode.LOAD_BALANCING
            elif 'active-backup' in content:
                mode = NetworkMode.ACTIVE_BACKUP
            
            slaves = []
            slave_matches = re.findall(r'Slave Interface: ([a-zA-Z0-9]+)', content)
            slaves = slave_matches
            
            primary = None
            primary_match = re.search(r'Primary Slave: ([a-zA-Z0-9]+)', content)
            if primary_match:
                primary = primary_match.group(1)
            
            self.bonds[name] = BondConfig(
                name=name,
                mode=mode,
                slaves=slaves,
                primary=primary
            )
        except Exception as e:
            logger.error(f"解析Bonding配置失败: {e}")
    
    def create_bond(self, name: str, mode: NetworkMode, slaves: List[str],
                   primary: Optional[str] = None) -> bool:
        """创建Bonding接口"""
        try:
            logger.info(f"创建Bonding {name}, 模式: {mode}, 从接口: {slaves}")
            
            bond_mode_map = {
                NetworkMode.LOAD_BALANCING: 0,
                NetworkMode.ACTIVE_BACKUP: 1
            }
            
            if mode not in bond_mode_map:
                logger.error(f"不支持的Bonding模式: {mode}")
                return False
            
            if primary and primary not in slaves:
                logger.error(f"主接口 {primary} 不在从接口列表中")
                return False
            
            for slave in slaves:
                if slave not in self.interfaces:
                    logger.error(f"接口 {slave} 不存在")
                    return False
            
            modprobe_cmd = "modprobe bonding"
            subprocess.run(modprobe_cmd, shell=True, capture_output=True, text=True)
            
            echo_cmd = f"echo +{name} > /sys/class/net/bonding_masters"
            result = subprocess.run(echo_cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0 and result.stderr and "File exists" not in result.stderr:
                logger.error(f"创建Bonding主接口失败: {result.stderr}")
                return False
            
            mode_cmd = f"echo {bond_mode_map[mode]} > /sys/class/net/{name}/bonding/mode"
            subprocess.run(mode_cmd, shell=True, capture_output=True, text=True)
            
            miimon_cmd = f"echo 100 > /sys/class/net/{name}/bonding/miimon"
            subprocess.run(miimon_cmd, shell=True, capture_output=True, text=True)
            
            if primary:
                primary_cmd = f"echo {primary} > /sys/class/net/{name}/bonding/primary"
                subprocess.run(primary_cmd, shell=True, capture_output=True, text=True)
            
            for slave in slaves:
                subprocess.run(f"ip link set {slave} down", shell=True, capture_output=True, text=True)
                slave_cmd = f"echo +{slave} > /sys/class/net/{name}/bonding/slaves"
                result = subprocess.run(slave_cmd, shell=True, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.warning(f"添加从接口 {slave} 失败: {result.stderr}")
            
            subprocess.run(f"ip link set {name} up", shell=True, capture_output=True, text=True)
            
            self._load_bonds()
            self._scan_interfaces()
            logger.info(f"Bonding {name} 创建成功")
            return True
            
        except Exception as e:
            logger.error(f"创建Bonding异常: {e}")
            return False
    
    def configure_interface(self, name: str, ip_address: str, netmask: str,
                           gateway: Optional[str] = None, mtu: Optional[int] = None) -> bool:
        """配置网络接口"""
        try:
            logger.info(f"配置接口 {name}: {ip_address}/{netmask}")
            
            if name not in self.interfaces:
                logger.error(f"接口 {name} 不存在")
                return False
            
            flush_cmd = f"ip addr flush dev {name}"
            subprocess.run(flush_cmd, shell=True, capture_output=True, text=True)
            
            addr_cmd = f"ip addr add {ip_address}/{self._netmask_to_prefix(netmask)} dev {name}"
            result = subprocess.run(addr_cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"配置IP地址失败: {result.stderr}")
                return False
            
            if gateway:
                route_cmd = f"ip route add default via {gateway} dev {name}"
                subprocess.run(route_cmd, shell=True, capture_output=True, text=True)
            
            if mtu:
                mtu_cmd = f"ip link set dev {name} mtu {mtu}"
                subprocess.run(mtu_cmd, shell=True, capture_output=True, text=True)
            
            up_cmd = f"ip link set {name} up"
            subprocess.run(up_cmd, shell=True, capture_output=True, text=True)
            
            self._scan_interfaces()
            logger.info(f"接口 {name} 配置成功")
            return True
            
        except Exception as e:
            logger.error(f"配置接口异常: {e}")
            return False
    
    def _netmask_to_prefix(self, netmask: str) -> int:
        """子网掩码转前缀"""
        parts = list(map(int, netmask.split('.')))
        binary = ''.join(f'{part:08b}' for part in parts)
        return binary.count('1')
    
    def add_ip_alias(self, interface_name: str, ip_address: str, netmask: str) -> bool:
        """添加IP别名（多址设定）"""
        try:
            logger.info(f"为接口 {interface_name} 添加IP别名: {ip_address}")
            
            if interface_name not in self.interfaces:
                logger.error(f"接口 {interface_name} 不存在")
                return False
            
            prefix = self._netmask_to_prefix(netmask)
            cmd = f"ip addr add {ip_address}/{prefix} dev {interface_name}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"添加IP别名失败: {result.stderr}")
                return False
            
            self._scan_interfaces()
            logger.info(f"IP别名添加成功")
            return True
            
        except Exception as e:
            logger.error(f"添加IP别名异常: {e}")
            return False
    
    def get_network_status(self) -> Dict:
        """获取网络状态"""
        return {
            'timestamp': datetime.now().isoformat(),
            'interfaces': [
                {
                    'name': iface.name,
                    'status': iface.status.value,
                    'mac_address': iface.mac_address,
                    'ip_addresses': iface.ip_addresses,
                    'netmask': iface.netmask,
                    'gateway': iface.gateway,
                    'mtu': iface.mtu,
                    'speed': iface.speed,
                    'is_primary': iface.is_primary
                }
                for iface in self.interfaces.values()
            ],
            'bonds': [
                {
                    'name': bond.name,
                    'mode': bond.mode.value,
                    'slaves': bond.slaves,
                    'primary': bond.primary
                }
                for bond in self.bonds.values()
            ]
        }
    
    def check_network_health(self) -> Dict:
        """检查网络健康状态"""
        status = {
            'overall_status': 'healthy',
            'issues': [],
            'warnings': []
        }
        
        for name, interface in self.interfaces.items():
            if interface.status == InterfaceStatus.DOWN:
                if name.startswith(('eth', 'en')):
                    status['issues'].append(f"物理接口 {name} 已断开")
                    status['overall_status'] = 'critical'
                elif not name.startswith(('lo', 'docker')):
                    status['warnings'].append(f"接口 {name} 已断开")
                    if status['overall_status'] == 'healthy':
                        status['overall_status'] = 'warning'
        
        for name, bond in self.bonds.items():
            active_slaves = [s for s in bond.slaves 
                            if s in self.interfaces 
                            and self.interfaces[s].status == InterfaceStatus.UP]
            
            if len(active_slaves) == 0:
                status['issues'].append(f"Bonding {name} 无可用从接口")
                status['overall_status'] = 'critical'
            elif len(active_slaves) < len(bond.slaves):
                status['warnings'].append(f"Bonding {name} 部分从接口不可用")
                if status['overall_status'] == 'healthy':
                    status['overall_status'] = 'warning'
        
        return status


if __name__ == '__main__':
    import yaml
    
    config_path = '/workspace/ai-box/config/settings_high_performance.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    network_manager = NetworkManager(config)
    
    print("=== 网络状态 ===")
    status = network_manager.get_network_status()
    
    print("\n接口列表:")
    for iface in status['interfaces']:
        print(f"  {iface['name']}:")
        print(f"    状态: {iface['status']}")
        print(f"    MAC: {iface['mac_address']}")
        print(f"    IP: {', '.join(iface['ip_addresses'])}")
        print(f"    网关: {iface['gateway']}")
        print(f"    速度: {iface['speed']}")
    
    if status['bonds']:
        print("\nBonding配置:")
        for bond in status['bonds']:
            print(f"  {bond['name']}:")
            print(f"    模式: {bond['mode']}")
            print(f"    从接口: {', '.join(bond['slaves'])}")
            if bond['primary']:
                print(f"    主接口: {bond['primary']}")
    
    print("\n=== 网络健康检查 ===")
    health = network_manager.check_network_health()
    print(f"整体状态: {health['overall_status']}")
    if health['issues']:
        print("问题:")
        for issue in health['issues']:
            print(f"  - {issue}")
    if health['warnings']:
        print("警告:")
        for warning in health['warnings']:
            print(f"  - {warning}")
