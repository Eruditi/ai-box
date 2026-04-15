#!/usr/bin/env python3
"""
GB28181协议管理器
支持GB28181标准协议和级联功能
"""

import logging
import threading
import time
import hashlib
import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class SipMethod(Enum):
    """SIP方法类型"""
    REGISTER = "REGISTER"
    INVITE = "INVITE"
    BYE = "BYE"
    MESSAGE = "MESSAGE"
    NOTIFY = "NOTIFY"
    SUBSCRIBE = "SUBSCRIBE"


class DeviceStatus(Enum):
    """设备状态"""
    OFFLINE = "offline"
    ONLINE = "online"
    REGISTERING = "registering"
    ERROR = "error"


@dataclass
class GB28181Device:
    """GB28181设备信息"""
    device_id: str
    name: str
    ip: str
    port: int
    manufacturer: str = ""
    model: str = ""
    firmware: str = ""
    status: DeviceStatus = DeviceStatus.OFFLINE
    last_seen: float = 0
    channels: List[Dict] = None
    register_expires: int = 3600


@dataclass
class GB28181Config:
    """GB28181配置"""
    local_ip: str
    local_port: int
    device_id: str
    realm: str
    keepalive_interval: int = 60
    register_expires: int = 3600


class GB28181Manager:
    """GB28181协议管理器"""
    
    def __init__(self, config: GB28181Config):
        self.config = config
        self.devices: Dict[str, GB28181Device] = {}
        self.running = False
        self.server_socket = None
        self.server_thread = None
        self.keepalive_thread = None
        self.cascade_servers: List[Dict] = []
        self.call_sessions: Dict[str, Any] = {}
        self.lock = threading.Lock()
        
    def _generate_sip_message(
        self,
        method: SipMethod,
        from_uri: str,
        to_uri: str,
        call_id: str = None,
        cseq: int = 1,
        body: str = "",
        via: str = None
    ) -> str:
        """生成SIP消息"""
        if not call_id:
            call_id = hashlib.md5(f"{time.time()}{random.random()}".encode()).hexdigest()
        
        via_addr = via or f"SIP/2.0/UDP {self.config.local_ip}:{self.config.local_port};rport"
        from_tag = hashlib.md5(f"{time.time()}{self.config.device_id}".encode()).hexdigest()[:8]
        to_tag = hashlib.md5(f"{time.time()}{to_uri}".encode()).hexdigest()[:8]
        
        message = (
            f"{method.value} sip:{to_uri} SIP/2.0\r\n"
            f"Via: {via_addr}\r\n"
            f"From: <sip:{from_uri}>;tag={from_tag}\r\n"
            f"To: <sip:{to_uri}>\r\n"
            f"Call-ID: {call_id}\r\n"
            f"CSeq: {cseq} {method.value}\r\n"
            f"Contact: <sip:{self.config.device_id}@{self.config.local_ip}:{self.config.local_port}>\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Max-Forwards: 70\r\n"
            f"User-Agent: AI Edge Server v2.0\r\n"
            "\r\n"
            f"{body}"
        )
        return message
    
    def _generate_register_body(self) -> str:
        """生成注册消息体（设备目录）"""
        body = (
            '<?xml version="1.0" encoding="GB2312"?>\r\n'
            '<Query>\r\n'
            f'  <CmdType>Catalog</CmdType>\r\n'
            f'  <SN>{int(time.time())}</SN>\r\n'
            f'  <DeviceID>{self.config.device_id}</DeviceID>\r\n'
            '</Query>\r\n'
        )
        return body
    
    def _process_incoming_message(self, data: bytes, addr: tuple) -> Optional[str]:
        """处理接收到的SIP消息"""
        try:
            message = data.decode('utf-8', errors='ignore')
            logging.info(f"Received message from {addr}: {message[:200]}...")
            
            if "REGISTER" in message:
                return self._handle_register(message, addr)
            elif "INVITE" in message:
                return self._handle_invite(message, addr)
            elif "BYE" in message:
                return self._handle_bye(message, addr)
            elif "MESSAGE" in message:
                return self._handle_message(message, addr)
            
            return None
        except Exception as e:
            logging.error(f"Error processing incoming message: {e}")
            return None
    
    def _handle_register(self, message: str, addr: tuple) -> Optional[str]:
        """处理注册请求"""
        try:
            device_id = self._extract_device_id(message)
            if not device_id:
                return None
            
            with self.lock:
                if device_id not in self.devices:
                    device = GB28181Device(
                        device_id=device_id,
                        name=f"Device_{device_id[-8:]}",
                        ip=addr[0],
                        port=addr[1],
                        status=DeviceStatus.REGISTERING
                    )
                    self.devices[device_id] = device
                else:
                    device = self.devices[device_id]
                    device.ip = addr[0]
                    device.port = addr[1]
                
                device.last_seen = time.time()
                device.status = DeviceStatus.ONLINE
            
            response = (
                "SIP/2.0 200 OK\r\n"
                "Via: SIP/2.0/UDP " + addr[0] + ":" + str(addr[1]) + "\r\n"
                f"From: <sip:{device_id}>\r\n"
                f"To: <sip:{device_id}>;tag={hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}\r\n"
                f"Call-ID: {hashlib.md5(str(time.time()).encode()).hexdigest()}\r\n"
                f"CSeq: 1 REGISTER\r\n"
                "Contact: <sip:" + self.config.device_id + "@" + self.config.local_ip + ":" + str(self.config.local_port) + ">\r\n"
                f"Expires: {self.config.register_expires}\r\n"
                "Content-Length: 0\r\n"
                "\r\n"
            )
            return response
        except Exception as e:
            logging.error(f"Error handling register: {e}")
            return None
    
    def _handle_invite(self, message: str, addr: tuple) -> Optional[str]:
        """处理INVITE请求（视频流请求）"""
        try:
            call_id = self._extract_call_id(message)
            device_id = self._extract_device_id(message)
            
            if call_id and device_id:
                with self.lock:
                    self.call_sessions[call_id] = {
                        'device_id': device_id,
                        'addr': addr,
                        'start_time': time.time()
                    }
            
            response = (
                "SIP/2.0 200 OK\r\n"
                "Via: SIP/2.0/UDP " + addr[0] + ":" + str(addr[1]) + "\r\n"
                f"From: <sip:{device_id}>\r\n"
                f"To: <sip:{device_id}>;tag={hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}\r\n"
                f"Call-ID: {call_id}\r\n"
                "CSeq: 1 INVITE\r\n"
                "Content-Type: application/sdp\r\n"
                "Content-Length: 0\r\n"
                "\r\n"
            )
            return response
        except Exception as e:
            logging.error(f"Error handling invite: {e}")
            return None
    
    def _handle_bye(self, message: str, addr: tuple) -> Optional[str]:
        """处理BYE请求"""
        try:
            call_id = self._extract_call_id(message)
            
            if call_id in self.call_sessions:
                with self.lock:
                    del self.call_sessions[call_id]
            
            response = (
                "SIP/2.0 200 OK\r\n"
                "Via: SIP/2.0/UDP " + addr[0] + ":" + str(addr[1]) + "\r\n"
                f"Call-ID: {call_id}\r\n"
                "CSeq: 1 BYE\r\n"
                "Content-Length: 0\r\n"
                "\r\n"
            )
            return response
        except Exception as e:
            logging.error(f"Error handling bye: {e}")
            return None
    
    def _handle_message(self, message: str, addr: tuple) -> Optional[str]:
        """处理MESSAGE请求"""
        try:
            response = (
                "SIP/2.0 200 OK\r\n"
                "Via: SIP/2.0/UDP " + addr[0] + ":" + str(addr[1]) + "\r\n"
                "CSeq: 1 MESSAGE\r\n"
                "Content-Length: 0\r\n"
                "\r\n"
            )
            return response
        except Exception as e:
            logging.error(f"Error handling message: {e}")
            return None
    
    def _extract_device_id(self, message: str) -> Optional[str]:
        """从消息中提取设备ID"""
        import re
        match = re.search(r'sip:(\d+)', message)
        return match.group(1) if match else None
    
    def _extract_call_id(self, message: str) -> Optional[str]:
        """从消息中提取Call-ID"""
        import re
        match = re.search(r'Call-ID:\s*([^\r\n]+)', message)
        return match.group(1) if match else None
    
    def _server_loop(self):
        """服务器主循环"""
        import socket
        
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.server_socket.bind((self.config.local_ip, self.config.local_port))
            self.server_socket.settimeout(1.0)
            
            logging.info(f"GB28181 server started on {self.config.local_ip}:{self.config.local_port}")
            
            while self.running:
                try:
                    data, addr = self.server_socket.recvfrom(65535)
                    response = self._process_incoming_message(data, addr)
                    
                    if response:
                        self.server_socket.sendto(response.encode('utf-8'), addr)
                except socket.timeout:
                    continue
                except Exception as e:
                    logging.error(f"Server loop error: {e}")
                    continue
        except Exception as e:
            logging.error(f"Failed to start GB28181 server: {e}")
        finally:
            if self.server_socket:
                self.server_socket.close()
    
    def _keepalive_loop(self):
        """保活循环"""
        while self.running:
            try:
                current_time = time.time()
                
                with self.lock:
                    for device_id, device in list(self.devices.items()):
                        if device.status == DeviceStatus.ONLINE:
                            if current_time - device.last_seen > device.register_expires:
                                device.status = DeviceStatus.OFFLINE
                                logging.warning(f"Device {device_id} offline (timeout)")
                
                time.sleep(self.config.keepalive_interval)
            except Exception as e:
                logging.error(f"Keepalive loop error: {e}")
                time.sleep(1)
    
    def add_cascade_server(self, ip: str, port: int, server_id: str):
        """添加级联上级服务器"""
        self.cascade_servers.append({
            'ip': ip,
            'port': port,
            'server_id': server_id,
            'registered': False
        })
        logging.info(f"Added cascade server: {ip}:{port} ({server_id})")
    
    def start(self):
        """启动GB28181服务"""
        logging.info("Starting GB28181 manager...")
        self.running = True
        
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self.server_thread.start()
        
        self.keepalive_thread = threading.Thread(target=self._keepalive_loop, daemon=True)
        self.keepalive_thread.start()
        
        logging.info("GB28181 manager started")
    
    def stop(self):
        """停止GB28181服务"""
        logging.info("Stopping GB28181 manager...")
        self.running = False
        
        if self.server_thread:
            self.server_thread.join(timeout=3)
        
        if self.keepalive_thread:
            self.keepalive_thread.join(timeout=3)
        
        self.call_sessions.clear()
        logging.info("GB28181 manager stopped")
    
    def get_device(self, device_id: str) -> Optional[GB28181Device]:
        """获取设备信息"""
        with self.lock:
            return self.devices.get(device_id)
    
    def get_all_devices(self) -> List[GB28181Device]:
        """获取所有设备"""
        with self.lock:
            return list(self.devices.values())
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        with self.lock:
            return {
                'total_devices': len(self.devices),
                'online_devices': sum(1 for d in self.devices.values() if d.status == DeviceStatus.ONLINE),
                'offline_devices': sum(1 for d in self.devices.values() if d.status == DeviceStatus.OFFLINE),
                'active_calls': len(self.call_sessions),
                'cascade_servers': len(self.cascade_servers)
            }
    
    def query_device_catalog(self, device_id: str) -> bool:
        """查询设备目录"""
        try:
            device = self.get_device(device_id)
            if not device or device.status != DeviceStatus.ONLINE:
                logging.warning(f"Device {device_id} not available")
                return False
            
            message = self._generate_sip_message(
                method=SipMethod.MESSAGE,
                from_uri=self.config.device_id,
                to_uri=device_id,
                body=self._generate_catalog_query(device_id)
            )
            
            if self.server_socket:
                self.server_socket.sendto(
                    message.encode('utf-8'),
                    (device.ip, device.port)
                )
                logging.info(f"Sent catalog query to {device_id}")
                return True
            
            return False
        except Exception as e:
            logging.error(f"Failed to query device catalog: {e}")
            return False
    
    def _generate_catalog_query(self, device_id: str) -> str:
        """生成目录查询消息体"""
        return (
            '<?xml version="1.0" encoding="GB2312"?>\r\n'
            '<Query>\r\n'
            '  <CmdType>Catalog</CmdType>\r\n'
            f'  <SN>{int(time.time())}</SN>\r\n'
            f'  <DeviceID>{device_id}</DeviceID>\r\n'
            '</Query>\r\n'
        )
    
    def invite_stream(self, device_id: str, channel_id: str, ssrc: str = None) -> bool:
        """请求实时视频流"""
        try:
            device = self.get_device(device_id)
            if not device or device.status != DeviceStatus.ONLINE:
                logging.warning(f"Device {device_id} not available")
                return False
            
            if not ssrc:
                ssrc = f"{random.randint(0, 0xFFFFFFFF):010d}"
            
            sdp_body = self._generate_sdp_invite(device_id, channel_id, ssrc)
            
            message = self._generate_sip_message(
                method=SipMethod.INVITE,
                from_uri=self.config.device_id,
                to_uri=f"{device_id}_{channel_id}",
                body=sdp_body
            )
            
            if self.server_socket:
                self.server_socket.sendto(
                    message.encode('utf-8'),
                    (device.ip, device.port)
                )
                logging.info(f"Sent INVITE to {device_id} channel {channel_id}")
                return True
            
            return False
        except Exception as e:
            logging.error(f"Failed to invite stream: {e}")
            return False
    
    def _generate_sdp_invite(self, device_id: str, channel_id: str, ssrc: str) -> str:
        """生成INVITE的SDP消息体"""
        return (
            'v=0\r\n'
            f'o=- 0 0 IN IP4 {self.config.local_ip}\r\n'
            's=Play\r\n'
            f'c=IN IP4 {self.config.local_ip}\r\n'
            't=0 0\r\n'
            'm=video 0 RTP/AVP 96\r\n'
            'a=rtpmap:96 PS/90000\r\n'
            f'a=recvonly\r\n'
            f'y={ssrc}\r\n'
            f'f=\r\n'
        )
    
    def playback_stream(self, device_id: str, channel_id: str, 
                       start_time: str, end_time: str, ssrc: str = None) -> bool:
        """请求历史回放流"""
        try:
            device = self.get_device(device_id)
            if not device or device.status != DeviceStatus.ONLINE:
                logging.warning(f"Device {device_id} not available")
                return False
            
            if not ssrc:
                ssrc = f"{random.randint(0, 0xFFFFFFFF):010d}"
            
            sdp_body = self._generate_sdp_playback(device_id, channel_id, start_time, end_time, ssrc)
            
            message = self._generate_sip_message(
                method=SipMethod.INVITE,
                from_uri=self.config.device_id,
                to_uri=f"{device_id}_{channel_id}",
                body=sdp_body
            )
            
            if self.server_socket:
                self.server_socket.sendto(
                    message.encode('utf-8'),
                    (device.ip, device.port)
                )
                logging.info(f"Sent playback INVITE to {device_id} channel {channel_id}")
                return True
            
            return False
        except Exception as e:
            logging.error(f"Failed to playback stream: {e}")
            return False
    
    def _generate_sdp_playback(self, device_id: str, channel_id: str,
                               start_time: str, end_time: str, ssrc: str) -> str:
        """生成回放的SDP消息体"""
        return (
            'v=0\r\n'
            f'o=- 0 0 IN IP4 {self.config.local_ip}\r\n'
            's=Playback\r\n'
            f'c=IN IP4 {self.config.local_ip}\r\n'
            f't={start_time} {end_time}\r\n'
            'm=video 0 RTP/AVP 96\r\n'
            'a=rtpmap:96 PS/90000\r\n'
            f'a=recvonly\r\n'
            f'y={ssrc}\r\n'
            f'f=\r\n'
        )
    
    def ptz_control(self, device_id: str, channel_id: str, 
                   action: str, params: Dict[str, Any]) -> bool:
        """云台控制"""
        try:
            device = self.get_device(device_id)
            if not device or device.status != DeviceStatus.ONLINE:
                logging.warning(f"Device {device_id} not available")
                return False
            
            ptz_body = self._generate_ptz_control(channel_id, action, params)
            
            message = self._generate_sip_message(
                method=SipMethod.MESSAGE,
                from_uri=self.config.device_id,
                to_uri=device_id,
                body=ptz_body
            )
            
            if self.server_socket:
                self.server_socket.sendto(
                    message.encode('utf-8'),
                    (device.ip, device.port)
                )
                logging.info(f"Sent PTZ control to {device_id}: {action}")
                return True
            
            return False
        except Exception as e:
            logging.error(f"Failed to PTZ control: {e}")
            return False
    
    def _generate_ptz_control(self, channel_id: str, action: str, params: Dict[str, Any]) -> str:
        """生成云台控制消息体"""
        action_map = {
            'stop': 'stop',
            'left': 'left',
            'right': 'right',
            'up': 'up',
            'down': 'down',
            'zoom_in': 'zoom_in',
            'zoom_out': 'zoom_out',
            'focus_near': 'focus_near',
            'focus_far': 'focus_far',
            'iris_small': 'iris_small',
            'iris_large': 'iris_large'
        }
        
        cmd_type = action_map.get(action, 'stop')
        
        return (
            '<?xml version="1.0" encoding="GB2312"?>\r\n'
            '<Control>\r\n'
            '  <CmdType>DeviceControl</CmdType>\r\n'
            f'  <SN>{int(time.time())}</SN>\r\n'
            f'  <DeviceID>{channel_id}</DeviceID>\r\n'
            '  <PTZCmd>\r\n'
            f'    <Action>{cmd_type}</Action>\r\n'
            f'    <HorSpeed>{params.get("hor_speed", 0)}</HorSpeed>\r\n'
            f'    <VerSpeed>{params.get("ver_speed", 0)}</VerSpeed>\r\n'
            f'    <ZoomSpeed>{params.get("zoom_speed", 0)}</ZoomSpeed>\r\n'
            '  </PTZCmd>\r\n'
            '</Control>\r\n'
        )
    
    def bye_stream(self, device_id: str, channel_id: str, call_id: str) -> bool:
        """停止视频流"""
        try:
            device = self.get_device(device_id)
            if not device:
                logging.warning(f"Device {device_id} not found")
                return False
            
            message = self._generate_sip_message(
                method=SipMethod.BYE,
                from_uri=self.config.device_id,
                to_uri=f"{device_id}_{channel_id}",
                call_id=call_id
            )
            
            if self.server_socket:
                self.server_socket.sendto(
                    message.encode('utf-8'),
                    (device.ip, device.port)
                )
                logging.info(f"Sent BYE to {device_id} channel {channel_id}")
                return True
            
            return False
        except Exception as e:
            logging.error(f"Failed to bye stream: {e}")
            return False


def create_gb28181_manager_from_config(config) -> GB28181Manager:
    """从配置创建GB28181管理器"""
    gb28181_config = GB28181Config(
        local_ip=config.get('gb28181.server_ip', '0.0.0.0'),
        local_port=config.get('gb28181.server_port', 5060),
        device_id=config.get('gb28181.device_id', '34020000002000000001'),
        realm=config.get('gb28181.realm', '3402000000'),
        keepalive_interval=config.get('gb28181.keepalive_interval', 60),
        register_expires=config.get('gb28181.register_expires', 3600)
    )
    
    manager = GB28181Manager(gb28181_config)
    
    cascade_config = config.get('gb28181.cascade.upper_servers', [])
    for server in cascade_config:
        manager.add_cascade_server(
            ip=server.get('ip'),
            port=server.get('port', 5060),
            server_id=server.get('id')
        )
    
    return manager
