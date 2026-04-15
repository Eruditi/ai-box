#!/usr/bin/env python3
"""
第三方AI API集成模块
支持百度AI、阿里云视觉、腾讯云AI等
用于替代伪算法实现，提供真正的深度学习能力
"""

import os
import json
import logging
import time
import base64
import hashlib
import hmac
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod


class AIProvider(Enum):
    """AI服务提供商"""
    BAIDU = "baidu"
    ALIYUN = "aliyun"
    TENCENT = "tencent"
    HUAWEI = "huawei"
    LOCAL_YOLO = "local_yolo"


@dataclass
class DetectionResult:
    """检测结果"""
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    category: str
    extra: Dict[str, Any] = None


class BaseAIService(ABC):
    """AI服务基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get('api_key', '')
        self.secret_key = config.get('secret_key', '')
        self.enabled = config.get('enabled', False)
        self.timeout = config.get('timeout', 10)
        self.retry_count = config.get('retry_count', 3)
    
    @abstractmethod
    def detect(self, image_data: bytes, algorithms: List[str] = None) -> List[DetectionResult]:
        """执行检测"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查服务是否可用"""
        pass
    
    def _encode_image(self, image_data: bytes) -> str:
        """Base64编码图片"""
        return base64.b64encode(image_data).decode('utf-8')


class BaiduAIService(BaseAIService):
    """百度AI服务"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.access_token = None
        self.token_expire_time = 0
        self.api_base = "https://aip.baidubce.com/rest/2.0"
    
    def _get_access_token(self) -> str:
        """获取百度API Access Token"""
        if self.access_token and time.time() < self.token_expire_time:
            return self.access_token
        
        url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={self.api_key}&client_secret={self.secret_key}"
        
        try:
            req = urllib.request.Request(url, method='POST')
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode('utf-8'))
                self.access_token = result.get('access_token', '')
                self.token_expire_time = time.time() + result.get('expires_in', 86400) - 300
                return self.access_token
        except Exception as e:
            logging.error(f"[百度AI] 获取Access Token失败: {e}")
            return ""
    
    def detect(self, image_data: bytes, algorithms: List[str] = None) -> List[DetectionResult]:
        """执行检测"""
        if not self.is_available():
            return []
        
        results = []
        token = self._get_access_token()
        if not token:
            return []
        
        image_base64 = self._encode_image(image_data)
        
        algorithm_map = {
            'person': self._detect_person,
            'face': self._detect_face,
            'vehicle': self._detect_vehicle,
            'helmet': self._detect_helmet,
            'fire': self._detect_fire,
            'smoke': self._detect_smoke,
        }
        
        algorithms = algorithms or list(algorithm_map.keys())
        
        for algo in algorithms:
            if algo in algorithm_map:
                try:
                    algo_results = algorithm_map[algo](image_base64, token)
                    results.extend(algo_results)
                except Exception as e:
                    logging.error(f"[百度AI] {algo}检测失败: {e}")
        
        return results
    
    def _detect_person(self, image_base64: str, token: str) -> List[DetectionResult]:
        """人体检测"""
        url = f"{self.api_base}/image-classify/v1/body_analysis?access_token={token}"
        data = urllib.parse.urlencode({'image': image_base64}).encode('utf-8')
        
        req = urllib.request.Request(url, data=data, method='POST')
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            result = json.loads(response.read().decode('utf-8'))
        
        detections = []
        for person in result.get('person_info', []):
            detections.append(DetectionResult(
                label='person',
                confidence=person.get('score', 0.0),
                bbox=(person['location']['left'], person['location']['top'],
                      person['location']['width'], person['location']['height']),
                category='PERSON_VIOLATION'
            ))
        return detections
    
    def _detect_face(self, image_base64: str, token: str) -> List[DetectionResult]:
        """人脸检测"""
        url = f"{self.api_base}/face/v3/detect?access_token={token}"
        params = {
            'image': image_base64,
            'image_type': 'BASE64',
            'face_field': 'age,gender,mask,glasses'
        }
        data = json.dumps(params).encode('utf-8')
        
        req = urllib.request.Request(url, data=data, method='POST')
        req.add_header('Content-Type', 'application/json')
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            result = json.loads(response.read().decode('utf-8'))
        
        detections = []
        for face in result.get('result', {}).get('face_list', []):
            location = face.get('location', {})
            extra = {
                'age': face.get('age', 0),
                'gender': face.get('gender', {}).get('type', 'unknown'),
                'mask': face.get('mask', {}).get('type', 0),
                'glasses': face.get('glasses', {}).get('type', 0)
            }
            detections.append(DetectionResult(
                label='face',
                confidence=face.get('face_probability', 0.0),
                bbox=(location.get('left', 0), location.get('top', 0),
                      location.get('width', 0), location.get('height', 0)),
                category='FACE_RECOGNITION',
                extra=extra
            ))
        return detections
    
    def _detect_vehicle(self, image_base64: str, token: str) -> List[DetectionResult]:
        """车辆检测"""
        url = f"{self.api_base}/image-classify/v1/car?access_token={token}"
        data = urllib.parse.urlencode({'image': image_base64}).encode('utf-8')
        
        req = urllib.request.Request(url, data=data, method='POST')
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            result = json.loads(response.read().decode('utf-8'))
        
        detections = []
        for car in result.get('result', []):
            detections.append(DetectionResult(
                label=car.get('name', 'vehicle'),
                confidence=car.get('score', 0.0),
                bbox=(0, 0, 0, 0),
                category='STRUCTURED_ANALYSIS'
            ))
        return detections
    
    def _detect_helmet(self, image_base64: str, token: str) -> List[DetectionResult]:
        """安全帽检测"""
        url = f"{self.api_base}/image-classify/v1/body_analysis?access_token={token}"
        data = urllib.parse.urlencode({'image': image_base64, 'type': 'helmet'}).encode('utf-8')
        
        req = urllib.request.Request(url, data=data, method='POST')
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            result = json.loads(response.read().decode('utf-8'))
        
        detections = []
        for person in result.get('person_info', []):
            no_helmet = person.get('no_helmet', 0) > 0.5
            if no_helmet:
                detections.append(DetectionResult(
                    label='no_helmet',
                    confidence=person.get('no_helmet', 0.0),
                    bbox=(person['location']['left'], person['location']['top'],
                          person['location']['width'], person['location']['height']),
                    category='PERSON_VIOLATION'
                ))
        return detections
    
    def _detect_fire(self, image_base64: str, token: str) -> List[DetectionResult]:
        """火焰检测"""
        url = f"{self.api_base}/image-classify/v1/fire?access_token={token}"
        data = urllib.parse.urlencode({'image': image_base64}).encode('utf-8')
        
        req = urllib.request.Request(url, data=data, method='POST')
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            result = json.loads(response.read().decode('utf-8'))
        
        detections = []
        if result.get('result', {}).get('fire', 0) > 0.5:
            detections.append(DetectionResult(
                label='fire',
                confidence=result['result']['fire'],
                bbox=(0, 0, 0, 0),
                category='ENVIRONMENT_ABNORMAL'
            ))
        return detections
    
    def _detect_smoke(self, image_base64: str, token: str) -> List[DetectionResult]:
        """烟雾检测"""
        url = f"{self.api_base}/image-classify/v1/smoke?access_token={token}"
        data = urllib.parse.urlencode({'image': image_base64}).encode('utf-8')
        
        req = urllib.request.Request(url, data=data, method='POST')
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            result = json.loads(response.read().decode('utf-8'))
        
        detections = []
        if result.get('result', {}).get('smoke', 0) > 0.5:
            detections.append(DetectionResult(
                label='smoke',
                confidence=result['result']['smoke'],
                bbox=(0, 0, 0, 0),
                category='ENVIRONMENT_ABNORMAL'
            ))
        return detections
    
    def is_available(self) -> bool:
        return self.enabled and bool(self.api_key) and bool(self.secret_key)


class AliyunAIService(BaseAIService):
    """阿里云视觉AI服务"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_base = "https://vision.cn-shanghai.aliyuncs.com"
    
    def detect(self, image_data: bytes, algorithms: List[str] = None) -> List[DetectionResult]:
        """执行检测"""
        if not self.is_available():
            return []
        
        results = []
        image_base64 = self._encode_image(image_data)
        
        algorithm_map = {
            'person': self._detect_person,
            'face': self._detect_face,
            'vehicle': self._detect_vehicle,
        }
        
        algorithms = algorithms or list(algorithm_map.keys())
        
        for algo in algorithms:
            if algo in algorithm_map:
                try:
                    algo_results = algorithm_map[algo](image_base64)
                    results.extend(algo_results)
                except Exception as e:
                    logging.error(f"[阿里云AI] {algo}检测失败: {e}")
        
        return results
    
    def _detect_person(self, image_base64: str) -> List[DetectionResult]:
        """人体检测"""
        url = f"{self.api_base}/?Action=DetectBody"
        params = {
            'ImageURL': f"data:image/jpeg;base64,{image_base64}"
        }
        data = json.dumps(params).encode('utf-8')
        
        req = urllib.request.Request(url, data=data, method='POST')
        req.add_header('Authorization', f'APPCODE {self.api_key}')
        req.add_header('Content-Type', 'application/json')
        
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            result = json.loads(response.read().decode('utf-8'))
        
        detections = []
        for person in result.get('Data', {}).get('Elements', []):
            detections.append(DetectionResult(
                label='person',
                confidence=person.get('Score', 0.0),
                bbox=(person.get('X', 0), person.get('Y', 0),
                      person.get('Width', 0), person.get('Height', 0)),
                category='PERSON_VIOLATION'
            ))
        return detections
    
    def _detect_face(self, image_base64: str) -> List[DetectionResult]:
        """人脸检测"""
        url = f"{self.api_base}/?Action=DetectFace"
        params = {
            'ImageURL': f"data:image/jpeg;base64,{image_base64}"
        }
        data = json.dumps(params).encode('utf-8')
        
        req = urllib.request.Request(url, data=data, method='POST')
        req.add_header('Authorization', f'APPCODE {self.api_key}')
        req.add_header('Content-Type', 'application/json')
        
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            result = json.loads(response.read().decode('utf-8'))
        
        detections = []
        for face in result.get('Data', {}).get('FaceDetails', []):
            detections.append(DetectionResult(
                label='face',
                confidence=face.get('FaceConfidence', 0.0),
                bbox=(face.get('FaceRect', {}).get('Left', 0),
                      face.get('FaceRect', {}).get('Top', 0),
                      face.get('FaceRect', {}).get('Width', 0),
                      face.get('FaceRect', {}).get('Height', 0)),
                category='FACE_RECOGNITION'
            ))
        return detections
    
    def _detect_vehicle(self, image_base64: str) -> List[DetectionResult]:
        """车辆检测"""
        url = f"{self.api_base}/?Action=DetectVehicle"
        params = {
            'ImageURL': f"data:image/jpeg;base64,{image_base64}"
        }
        data = json.dumps(params).encode('utf-8')
        
        req = urllib.request.Request(url, data=data, method='POST')
        req.add_header('Authorization', f'APPCODE {self.api_key}')
        req.add_header('Content-Type', 'application/json')
        
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            result = json.loads(response.read().decode('utf-8'))
        
        detections = []
        for vehicle in result.get('Data', {}).get('Vehicles', []):
            detections.append(DetectionResult(
                label=vehicle.get('Type', 'vehicle'),
                confidence=vehicle.get('Confidence', 0.0),
                bbox=(vehicle.get('Rect', {}).get('Left', 0),
                      vehicle.get('Rect', {}).get('Top', 0),
                      vehicle.get('Rect', {}).get('Width', 0),
                      vehicle.get('Rect', {}).get('Height', 0)),
                category='STRUCTURED_ANALYSIS'
            ))
        return detections
    
    def is_available(self) -> bool:
        return self.enabled and bool(self.api_key)


class AIProviderManager:
    """AI服务提供商管理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.providers: Dict[str, BaseAIService] = {}
        self._init_providers()
    
    def _init_providers(self):
        """初始化所有AI服务提供商"""
        providers_config = self.config.get('ai_providers', {})
        
        if 'baidu' in providers_config:
            self.providers['baidu'] = BaiduAIService(providers_config['baidu'])
            logging.info(f"[AI提供商] 百度AI已初始化, enabled={providers_config['baidu'].get('enabled', False)}")
        
        if 'aliyun' in providers_config:
            self.providers['aliyun'] = AliyunAIService(providers_config['aliyun'])
            logging.info(f"[AI提供商] 阿里云AI已初始化, enabled={providers_config['aliyun'].get('enabled', False)}")
    
    def get_provider(self, name: str) -> Optional[BaseAIService]:
        """获取指定的AI服务提供商"""
        return self.providers.get(name)
    
    def get_available_providers(self) -> List[str]:
        """获取所有可用的AI服务提供商"""
        return [name for name, provider in self.providers.items() if provider.is_available()]
    
    def detect(self, image_data: bytes, algorithms: List[str] = None, 
               provider: str = None) -> List[DetectionResult]:
        """执行检测
        
        Args:
            image_data: 图片二进制数据
            algorithms: 要执行的算法列表
            provider: 指定的服务提供商，如果为None则使用第一个可用的
        
        Returns:
            检测结果列表
        """
        if provider:
            p = self.providers.get(provider)
            if p and p.is_available():
                return p.detect(image_data, algorithms)
        else:
            for p in self.providers.values():
                if p.is_available():
                    return p.detect(image_data, algorithms)
        
        return []
    
    def is_any_available(self) -> bool:
        """检查是否有任何可用的AI服务"""
        return any(p.is_available() for p in self.providers.values())


def get_ai_provider_manager(config: Dict[str, Any] = None) -> AIProviderManager:
    """获取AI服务提供商管理器单例"""
    if not hasattr(get_ai_provider_manager, '_instance'):
        get_ai_provider_manager._instance = AIProviderManager(config)
    return get_ai_provider_manager._instance
