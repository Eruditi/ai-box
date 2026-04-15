#!/usr/bin/env python3
"""
云端协同模块
支持与云端大模型、云存储、云平台对接
"""

import os
import json
import logging
import time
import hashlib
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import urllib.request
import urllib.error


class CloudProvider(Enum):
    """云服务提供商"""
    ALIYUN = "aliyun"
    TENCENT = "tencent"
    HUAWEI = "huawei"
    BAIDU = "baidu"
    CUSTOM = "custom"


@dataclass
class CloudConfig:
    """云端配置"""
    provider: CloudProvider
    endpoint: str
    api_key: str = ""
    secret_key: str = ""
    region: str = ""
    bucket: str = ""
    enabled: bool = True


class CloudClient:
    """云端客户端"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.cloud_config: Optional[CloudConfig] = None
        self._load_config()
    
    def _load_config(self):
        """加载配置"""
        cloud_config = self.config.get('cloud', {})
        
        if not cloud_config.get('enabled', False):
            return
        
        try:
            provider = CloudProvider(cloud_config.get('provider', 'custom').lower())
        except ValueError:
            provider = CloudProvider.CUSTOM
        
        self.cloud_config = CloudConfig(
            provider=provider,
            endpoint=cloud_config.get('endpoint', ''),
            api_key=cloud_config.get('api_key', ''),
            secret_key=cloud_config.get('secret_key', ''),
            region=cloud_config.get('region', ''),
            bucket=cloud_config.get('bucket', ''),
            enabled=True
        )
        
        logging.info(f"[云端协同] 配置加载完成: provider={provider.value}")
    
    def is_available(self) -> bool:
        """检查云端服务是否可用"""
        return self.cloud_config is not None and self.cloud_config.enabled
    
    def upload_alert(self, alert: Dict[str, Any], frame_data: bytes = None) -> Optional[str]:
        """上传告警到云端"""
        if not self.is_available():
            return None
        
        try:
            url = f"{self.cloud_config.endpoint}/api/alerts"
            
            data = {
                'alert': alert,
                'timestamp': time.time(),
                'device_id': self._get_device_id()
            }
            
            if frame_data:
                data['has_frame'] = True
            
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f"Bearer {self.cloud_config.api_key}",
                'X-Device-ID': self._get_device_id()
            }
            
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode('utf-8'),
                headers=headers,
                method='POST'
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result.get('alert_id')
                
        except Exception as e:
            logging.error(f"[云端协同] 上传告警失败: {e}")
            return None
    
    def upload_frame(self, frame_data: bytes, alert_id: str = None) -> Optional[str]:
        """上传帧图像到云端"""
        if not self.is_available():
            return None
        
        try:
            url = f"{self.cloud_config.endpoint}/api/frames/upload"
            
            boundary = f"----WebKitFormBoundary{int(time.time()*1000)}"
            
            body = []
            body.append(f'--{boundary}'.encode())
            body.append(f'Content-Disposition: form-data; name="frame"; filename="frame.jpg"'.encode())
            body.append(b'Content-Type: image/jpeg')
            body.append(b'')
            body.append(frame_data)
            
            if alert_id:
                body.append(f'--{boundary}'.encode())
                body.append(f'Content-Disposition: form-data; name="alert_id"'.encode())
                body.append(b'')
                body.append(alert_id.encode())
            
            body.append(f'--{boundary}--'.encode())
            
            headers = {
                'Content-Type': f'multipart/form-data; boundary={boundary}',
                'Authorization': f"Bearer {self.cloud_config.api_key}"
            }
            
            req = urllib.request.Request(
                url,
                data=b'\r\n'.join(body),
                headers=headers,
                method='POST'
            )
            
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result.get('frame_url')
                
        except Exception as e:
            logging.error(f"[云端协同] 上传帧失败: {e}")
            return None
    
    def request_analysis(self, frame_url: str, analysis_type: str = "deep") -> Optional[Dict[str, Any]]:
        """请求云端深度分析"""
        if not self.is_available():
            return None
        
        try:
            url = f"{self.cloud_config.endpoint}/api/analysis/{analysis_type}"
            
            data = {
                'frame_url': frame_url,
                'device_id': self._get_device_id(),
                'timestamp': time.time()
            }
            
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f"Bearer {self.cloud_config.api_key}"
            }
            
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode('utf-8'),
                headers=headers,
                method='POST'
            )
            
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result
                
        except Exception as e:
            logging.error(f"[云端协同] 请求分析失败: {e}")
            return None
    
    def sync_stats(self, stats: Dict[str, Any]) -> bool:
        """同步统计数据到云端"""
        if not self.is_available():
            return False
        
        try:
            url = f"{self.cloud_config.endpoint}/api/stats/sync"
            
            data = {
                'device_id': self._get_device_id(),
                'stats': stats,
                'timestamp': time.time()
            }
            
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f"Bearer {self.cloud_config.api_key}"
            }
            
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode('utf-8'),
                headers=headers,
                method='POST'
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                return True
                
        except Exception as e:
            logging.error(f"[云端协同] 同步统计失败: {e}")
            return False
    
    def get_device_config(self) -> Optional[Dict[str, Any]]:
        """从云端获取设备配置"""
        if not self.is_available():
            return None
        
        try:
            url = f"{self.cloud_config.endpoint}/api/devices/{self._get_device_id()}/config"
            
            headers = {
                'Authorization': f"Bearer {self.cloud_config.api_key}"
            }
            
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result.get('config')
                
        except Exception as e:
            logging.error(f"[云端协同] 获取配置失败: {e}")
            return None
    
    def report_heartbeat(self, status: Dict[str, Any]) -> bool:
        """上报心跳"""
        if not self.is_available():
            return False
        
        try:
            url = f"{self.cloud_config.endpoint}/api/devices/{self._get_device_id()}/heartbeat"
            
            data = {
                'status': status,
                'timestamp': time.time()
            }
            
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f"Bearer {self.cloud_config.api_key}"
            }
            
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode('utf-8'),
                headers=headers,
                method='POST'
            )
            
            with urllib.request.urlopen(req, timeout=5) as response:
                return True
                
        except Exception as e:
            logging.debug(f"[云端协同] 心跳上报失败: {e}")
            return False
    
    def _get_device_id(self) -> str:
        """获取设备ID"""
        device_id_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'device_id')
        
        if os.path.exists(device_id_file):
            with open(device_id_file, 'r') as f:
                return f.read().strip()
        
        import uuid
        device_id = str(uuid.uuid4())[:16]
        
        os.makedirs(os.path.dirname(device_id_file), exist_ok=True)
        with open(device_id_file, 'w') as f:
            f.write(device_id)
        
        return device_id
    
    def get_status(self) -> Dict[str, Any]:
        """获取云端协同状态"""
        if self.cloud_config is None:
            return {'enabled': False, 'available': False}
        
        return {
            'enabled': self.cloud_config.enabled,
            'available': self.is_available(),
            'provider': self.cloud_config.provider.value,
            'endpoint': self.cloud_config.endpoint,
            'device_id': self._get_device_id()
        }


class CloudLLMClient:
    """云端大模型客户端"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.endpoint = self.config.get('endpoint', '')
        self.api_key = self.config.get('api_key', '')
    
    def analyze_frame(self, frame_data: bytes, prompt: str = "描述这张图片中的异常情况") -> Optional[str]:
        """使用云端大模型分析帧"""
        try:
            import base64
            
            url = f"{self.endpoint}/v1/chat/completions"
            
            frame_base64 = base64.b64encode(frame_data).decode('utf-8')
            
            data = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{frame_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 500
            }
            
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f"Bearer {self.api_key}"
            }
            
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode('utf-8'),
                headers=headers,
                method='POST'
            )
            
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result['choices'][0]['message']['content']
                
        except Exception as e:
            logging.error(f"[云端LLM] 分析失败: {e}")
            return None


_cloud_client = None

def get_cloud_client(config: Dict[str, Any] = None) -> CloudClient:
    """获取云端客户端单例"""
    global _cloud_client
    if _cloud_client is None:
        _cloud_client = CloudClient(config)
    return _cloud_client
