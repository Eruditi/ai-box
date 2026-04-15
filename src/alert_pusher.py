#!/usr/bin/env python3
"""
告警推送模块
支持企业微信、钉钉、飞书机器人推送
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import urllib.request
import urllib.error


class PushChannel(Enum):
    """推送渠道"""
    WEWORK = "wework"       # 企业微信
    DINGTALK = "dingtalk"   # 钉钉
    FEISHU = "feishu"       # 飞书
    WEBHOOK = "webhook"     # 通用Webhook


@dataclass
class PushConfig:
    """推送配置"""
    channel: PushChannel
    webhook_url: str
    secret: str = ""
    enabled: bool = True
    cooldown_seconds: int = 60
    max_per_hour: int = 100


class AlertPusher:
    """告警推送器"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.channels: Dict[str, PushConfig] = {}
        self.cooldown_map: Dict[str, float] = {}
        self.hourly_count: Dict[str, List[float]] = {}
        self._load_config()

    def _load_config(self):
        """加载配置"""
        push_config = self.config.get('push', {})
        
        if push_config.get('wework', {}).get('enabled'):
            self.channels['wework'] = PushConfig(
                channel=PushChannel.WEWORK,
                webhook_url=push_config['wework'].get('webhook_url', ''),
                secret=push_config['wework'].get('secret', ''),
                enabled=True,
                cooldown_seconds=push_config['wework'].get('cooldown_seconds', 60),
                max_per_hour=push_config['wework'].get('max_per_hour', 100)
            )
        
        if push_config.get('dingtalk', {}).get('enabled'):
            self.channels['dingtalk'] = PushConfig(
                channel=PushChannel.DINGTALK,
                webhook_url=push_config['dingtalk'].get('webhook_url', ''),
                secret=push_config['dingtalk'].get('secret', ''),
                enabled=True,
                cooldown_seconds=push_config['dingtalk'].get('cooldown_seconds', 60),
                max_per_hour=push_config['dingtalk'].get('max_per_hour', 100)
            )
        
        if push_config.get('feishu', {}).get('enabled'):
            self.channels['feishu'] = PushConfig(
                channel=PushChannel.FEISHU,
                webhook_url=push_config['feishu'].get('webhook_url', ''),
                secret=push_config['feishu'].get('secret', ''),
                enabled=True,
                cooldown_seconds=push_config['feishu'].get('cooldown_seconds', 60),
                max_per_hour=push_config['feishu'].get('max_per_hour', 100)
            )

    def push_alert(self, alert: Dict[str, Any]) -> bool:
        """推送告警"""
        alert_type = alert.get('algorithm_name', 'unknown')
        cooldown_key = f"{alert_type}:{alert.get('camera_source', '')}"
        
        results = []
        for name, channel_config in self.channels.items():
            if not channel_config.enabled:
                continue
            
            if not self._check_rate_limit(name, cooldown_key, channel_config):
                continue
            
            try:
                if channel_config.channel == PushChannel.WEWORK:
                    result = self._push_wework(alert, channel_config)
                elif channel_config.channel == PushChannel.DINGTALK:
                    result = self._push_dingtalk(alert, channel_config)
                elif channel_config.channel == PushChannel.FEISHU:
                    result = self._push_feishu(alert, channel_config)
                else:
                    result = False
                
                if result:
                    self._update_rate_limit(name, cooldown_key)
                    results.append(True)
                    
            except Exception as e:
                logging.error(f"[告警推送] {name} 推送失败: {e}")
        
        return len(results) > 0

    def _check_rate_limit(self, channel_name: str, cooldown_key: str, config: PushConfig) -> bool:
        """检查频率限制"""
        now = time.time()
        
        cooldown_until = self.cooldown_map.get(f"{channel_name}:{cooldown_key}", 0)
        if now < cooldown_until:
            return False
        
        hourly_key = f"{channel_name}:{int(now // 3600)}"
        hourly_times = self.hourly_count.get(hourly_key, [])
        hourly_times = [t for t in hourly_times if now - t < 3600]
        
        if len(hourly_times) >= config.max_per_hour:
            logging.warning(f"[告警推送] {channel_name} 达到小时限制: {config.max_per_hour}")
            return False
        
        return True

    def _update_rate_limit(self, channel_name: str, cooldown_key: str):
        """更新频率限制"""
        now = time.time()
        self.cooldown_map[f"{channel_name}:{cooldown_key}"] = now
        
        hourly_key = f"{channel_name}:{int(now // 3600)}"
        if hourly_key not in self.hourly_count:
            self.hourly_count[hourly_key] = []
        self.hourly_count[hourly_key].append(now)

    def _push_wework(self, alert: Dict[str, Any], config: PushConfig) -> bool:
        """推送企业微信"""
        message = self._format_message(alert)
        
        data = {
            "msgtype": "markdown",
            "markdown": {
                "content": message
            }
        }
        
        return self._send_webhook(config.webhook_url, data)

    def _push_dingtalk(self, alert: Dict[str, Any], config: PushConfig) -> bool:
        """推送钉钉"""
        message = self._format_message(alert)
        
        data = {
            "msgtype": "markdown",
            "markdown": {
                "title": "AI Box 告警通知",
                "text": message
            }
        }
        
        url = config.webhook_url
        if config.secret:
            import hmac
            import hashlib
            import base64
            import urllib.parse
            
            timestamp = str(round(time.time() * 1000))
            string_to_sign = f"{timestamp}\n{config.secret}"
            hmac_code = hmac.new(
                config.secret.encode('utf-8'),
                string_to_sign.encode('utf-8'),
                digestmod=hashlib.sha256
            ).digest()
            sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
            url = f"{config.webhook_url}&timestamp={timestamp}&sign={sign}"
        
        return self._send_webhook(url, data)

    def _push_feishu(self, alert: Dict[str, Any], config: PushConfig) -> bool:
        """推送飞书"""
        message = self._format_message(alert)
        
        data = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": "AI Box 告警通知"
                    },
                    "template": "red"
                },
                "elements": [
                    {
                        "tag": "markdown",
                        "content": message
                    }
                ]
            }
        }
        
        return self._send_webhook(config.webhook_url, data)

    def _format_message(self, alert: Dict[str, Any]) -> str:
        """格式化消息"""
        algo_name = alert.get('algorithm_name', '未知告警')
        camera = alert.get('camera_source', '未知摄像头')
        confidence = alert.get('confidence', 0)
        timestamp = alert.get('datetime', time.strftime('%Y-%m-%d %H:%M:%S'))
        category = alert.get('category', '未知类型')
        
        confidence_emoji = "🔴" if confidence >= 0.8 else "🟡" if confidence >= 0.6 else "🟢"
        
        message = f"""## {confidence_emoji} AI Box 告警通知

**告警类型**: {algo_name}
**告警级别**: {category}
**置信度**: {confidence:.0%}
**摄像头**: {camera}
**时间**: {timestamp}

---
*由 AI Box 三省六部智能分析系统自动推送*"""
        
        return message

    def _send_webhook(self, url: str, data: Dict[str, Any]) -> bool:
        """发送Webhook请求"""
        try:
            json_data = json.dumps(data).encode('utf-8')
            req = urllib.request.Request(
                url,
                data=json_data,
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'AI-Box/1.0'
                }
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))
                
                if result.get('errcode', 0) == 0 or result.get('StatusCode', 0) == 0:
                    logging.info(f"[告警推送] 推送成功")
                    return True
                else:
                    logging.error(f"[告警推送] 推送失败: {result}")
                    return False
                    
        except urllib.error.URLError as e:
            logging.error(f"[告警推送] 网络错误: {e}")
            return False
        except Exception as e:
            logging.error(f"[告警推送] 发送失败: {e}")
            return False

    def push_summary(self, summary: str) -> bool:
        """推送摘要报告"""
        results = []
        for name, channel_config in self.channels.items():
            if not channel_config.enabled:
                continue
            
            try:
                if channel_config.channel == PushChannel.WEWORK:
                    data = {
                        "msgtype": "markdown",
                        "markdown": {"content": summary}
                    }
                    result = self._send_webhook(channel_config.webhook_url, data)
                elif channel_config.channel == PushChannel.DINGTALK:
                    data = {
                        "msgtype": "markdown",
                        "markdown": {
                            "title": "AI Box 值班报告",
                            "text": summary
                        }
                    }
                    result = self._send_webhook(channel_config.webhook_url, data)
                elif channel_config.channel == PushChannel.FEISHU:
                    data = {
                        "msg_type": "text",
                        "content": {"text": summary}
                    }
                    result = self._send_webhook(channel_config.webhook_url, data)
                else:
                    result = False
                
                results.append(result)
                
            except Exception as e:
                logging.error(f"[告警推送] 摘要推送失败 {name}: {e}")
        
        return any(results)

    def add_channel(self, name: str, channel: PushChannel, webhook_url: str, 
                   secret: str = "", enabled: bool = True):
        """添加推送渠道"""
        self.channels[name] = PushConfig(
            channel=channel,
            webhook_url=webhook_url,
            secret=secret,
            enabled=enabled
        )
        logging.info(f"[告警推送] 添加渠道: {name}")

    def remove_channel(self, name: str):
        """移除推送渠道"""
        if name in self.channels:
            del self.channels[name]
            logging.info(f"[告警推送] 移除渠道: {name}")

    def get_status(self) -> Dict[str, Any]:
        """获取推送状态"""
        return {
            'channels': {
                name: {
                    'type': config.channel.value,
                    'enabled': config.enabled,
                    'webhook_configured': bool(config.webhook_url)
                }
                for name, config in self.channels.items()
            },
            'hourly_stats': {
                key: len(times) for key, times in self.hourly_count.items()
            }
        }
    
    def is_available(self) -> bool:
        """检查是否有可用的推送渠道"""
        return len(self.channels) > 0 and any(c.enabled for c in self.channels.values())
