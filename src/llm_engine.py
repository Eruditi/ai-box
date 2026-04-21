#!/usr/bin/env python3
"""
LLM 接入模块
支持 OpenAI、智谱AI、通义千问、DeepSeek 等大模型
实现自然语言告警摘要和智能问答
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import urllib.request
import urllib.error


class LLMProvider(Enum):
    """LLM 提供商"""
    OPENAI = "openai"
    ZHIPU = "zhipu"           # 智谱AI (ChatGLM)
    QWEN = "qwen"             # 通义千问
    DEEPSEEK = "deepseek"     # DeepSeek
    OLLAMA = "ollama"         # 本地 Ollama
    CUSTOM = "custom"         # 自定义 API


@dataclass
class LLMConfig:
    """LLM 配置"""
    provider: LLMProvider
    api_key: str
    api_base: str = ""
    model: str = ""
    max_tokens: int = 500
    temperature: float = 0.7
    enabled: bool = True


DEFAULT_CONFIGS = {
    LLMProvider.OPENAI: {
        "api_base": "https://api.openai.com/v1",
        "model": "gpt-3.5-turbo"
    },
    LLMProvider.ZHIPU: {
        "api_base": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash"
    },
    LLMProvider.QWEN: {
        "api_base": "https://dashscope.aliyuncs.com/api/v1",
        "model": "qwen-turbo"
    },
    LLMProvider.DEEPSEEK: {
        "api_base": "https://api.deepseek.com/v1",
        "model": "deepseek-chat"
    },
    LLMProvider.OLLAMA: {
        "api_base": "http://localhost:11434/api",
        "model": "llama3"
    }
}

ALERT_SUMMARY_PROMPT = """你是一个智能监控助手。请根据以下告警信息生成简洁的自然语言摘要。

告警信息：
- 类型：{algorithm_name}
- 位置：{camera_source}
- 时间：{datetime}
- 置信度：{confidence}
- 类别：{category}

请用1-2句话描述这个告警，包含关键信息。不要输出多余内容。"""

DAILY_REPORT_PROMPT = """你是一个智能监控助手。请根据今日告警数据生成值班报告。

今日告警统计：
{stats_text}

请生成一份简洁的值班报告，包含：
1. 总体情况概述
2. 主要告警类型分析
3. 需要关注的问题

报告要求简洁专业，不超过200字。"""

QA_SYSTEM_PROMPT = """你是AI Box智能监控助手，负责回答关于监控系统的各种问题。

你可以回答以下类型的问题：
1. 今日/本周/本月的告警统计
2. 特定摄像头的告警情况
3. 特定类型告警的分析
4. 系统运行状态

请用简洁专业的语言回答用户问题。"""


class LLMEngine:
    """LLM 引擎"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.llm_config: Optional[LLMConfig] = None
        self.conversation_history: List[Dict[str, str]] = []
        self._load_config()

    def _load_config(self):
        """加载配置"""
        llm_config = self.config.get('llm', {})
        
        if not llm_config.get('enabled', False):
            return
        
        provider_str = llm_config.get('provider', 'openai').lower()
        try:
            provider = LLMProvider(provider_str)
        except ValueError:
            provider = LLMProvider.OPENAI
        
        default = DEFAULT_CONFIGS.get(provider, {})
        
        self.llm_config = LLMConfig(
            provider=provider,
            api_key=llm_config.get('api_key', ''),
            api_base=llm_config.get('api_base', default.get('api_base', '')),
            model=llm_config.get('model', default.get('model', '')),
            max_tokens=llm_config.get('max_tokens', 500),
            temperature=llm_config.get('temperature', 0.7),
            enabled=True
        )
        
        logging.info(f"[LLM引擎] 初始化完成: provider={provider.value}, model={self.llm_config.model}")

    def is_available(self) -> bool:
        """检查 LLM 是否可用"""
        return self.llm_config is not None and self.llm_config.enabled and bool(self.llm_config.api_key)

    def summarize_alert(self, alert: Dict[str, Any]) -> str:
        """生成告警摘要"""
        if not self.is_available():
            return self._fallback_summary(alert)
        
        prompt = ALERT_SUMMARY_PROMPT.format(
            algorithm_name=alert.get('algorithm_name', '未知'),
            camera_source=alert.get('camera_source', '未知'),
            datetime=alert.get('datetime', time.strftime('%Y-%m-%d %H:%M:%S')),
            confidence=f"{alert.get('confidence', 0):.0%}",
            category=alert.get('category', '未知')
        )
        
        try:
            response = self._call_llm(prompt)
            return response.strip()
        except Exception as e:
            logging.error(f"[LLM引擎] 告警摘要生成失败: {e}")
            return self._fallback_summary(alert)

    def generate_daily_report(self, stats: Dict[str, Any]) -> str:
        """生成每日报告"""
        if not self.is_available():
            return self._fallback_report(stats)
        
        stats_text = self._format_stats(stats)
        prompt = DAILY_REPORT_PROMPT.format(stats_text=stats_text)
        
        try:
            response = self._call_llm(prompt)
            return response.strip()
        except Exception as e:
            logging.error(f"[LLM引擎] 每日报告生成失败: {e}")
            return self._fallback_report(stats)

    def answer_question(self, question: str, context: Dict[str, Any] = None) -> str:
        """回答用户问题"""
        if not self.is_available():
            return "LLM 服务未配置，请先在设置中配置 API Key。"
        
        context = context or {}
        context_text = ""
        
        if 'stats' in context:
            context_text += f"\n当前统计数据：\n{self._format_stats(context['stats'])}"
        
        if 'recent_alerts' in context:
            alerts_text = "\n".join([
                f"- {a.get('datetime', '')} {a.get('algorithm_name', '')} @ {a.get('camera_source', '')}"
                for a in context['recent_alerts'][:10]
            ])
            context_text += f"\n最近告警：\n{alerts_text}"
        
        messages = [
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            {"role": "user", "content": f"{context_text}\n\n用户问题：{question}"}
        ]
        
        try:
            response = self._call_llm_chat(messages)
            return response.strip()
        except Exception as e:
            logging.error(f"[LLM引擎] 问答失败: {e}")
            return f"抱歉，处理问题时出错：{e}"

    def _call_llm(self, prompt: str) -> str:
        """调用 LLM API"""
        if self.llm_config.provider == LLMProvider.OLLAMA:
            return self._call_ollama(prompt)
        else:
            return self._call_openai_compatible(prompt)

    def _call_llm_chat(self, messages: List[Dict[str, str]]) -> str:
        """调用 LLM Chat API"""
        if self.llm_config.provider == LLMProvider.OLLAMA:
            return self._call_ollama_chat(messages)
        else:
            return self._call_openai_chat(messages)

    def _call_openai_compatible(self, prompt: str) -> str:
        """调用 OpenAI 兼容 API"""
        url = f"{self.llm_config.api_base}/chat/completions"
        
        data = {
            "model": self.llm_config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.llm_config.max_tokens,
            "temperature": self.llm_config.temperature
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.llm_config.api_key}"
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode('utf-8'),
            headers=headers,
            method='POST'
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result['choices'][0]['message']['content']

    def _call_openai_chat(self, messages: List[Dict[str, str]]) -> str:
        """调用 OpenAI Chat API"""
        url = f"{self.llm_config.api_base}/chat/completions"
        
        data = {
            "model": self.llm_config.model,
            "messages": messages,
            "max_tokens": self.llm_config.max_tokens,
            "temperature": self.llm_config.temperature
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.llm_config.api_key}"
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode('utf-8'),
            headers=headers,
            method='POST'
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result['choices'][0]['message']['content']

    def _call_ollama(self, prompt: str) -> str:
        """调用 Ollama API"""
        url = f"{self.llm_config.api_base}/generate"
        
        data = {
            "model": self.llm_config.model,
            "prompt": prompt,
            "stream": False
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode('utf-8'),
            headers={"Content-Type": "application/json"},
            method='POST'
        )
        
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get('response', '')

    def _call_ollama_chat(self, messages: List[Dict[str, str]]) -> str:
        """调用 Ollama Chat API"""
        url = f"{self.llm_config.api_base}/chat"
        
        data = {
            "model": self.llm_config.model,
            "messages": messages,
            "stream": False
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode('utf-8'),
            headers={"Content-Type": "application/json"},
            method='POST'
        )
        
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get('message', {}).get('content', '')

    def _fallback_summary(self, alert: Dict[str, Any]) -> str:
        """回退摘要（无 LLM 时）"""
        algo = alert.get('algorithm_name', '未知告警')
        camera = alert.get('camera_source', '未知位置')
        conf = alert.get('confidence', 0)
        return f"检测到【{algo}】，位置：{camera}，置信度：{conf:.0%}"

    def _fallback_report(self, stats: Dict[str, Any]) -> str:
        """回退报告（无 LLM 时）"""
        total = stats.get('total_alerts', 0)
        by_type = stats.get('by_type', {})
        top_type = max(by_type.items(), key=lambda x: x[1], default=('无', 0))
        
        return f"今日共产生 {total} 条告警，主要类型为【{top_type[0]}】({top_type[1]}次)。"

    def _format_stats(self, stats: Dict[str, Any]) -> str:
        """格式化统计数据"""
        lines = []
        lines.append(f"总告警数：{stats.get('total_alerts', 0)}")
        
        by_type = stats.get('by_type', {})
        if by_type:
            lines.append("\n按类型分布：")
            for t, c in sorted(by_type.items(), key=lambda x: -x[1])[:10]:
                lines.append(f"  - {t}: {c}次")
        
        by_camera = stats.get('by_camera', {})
        if by_camera:
            lines.append("\n按摄像头分布：")
            for c, n in sorted(by_camera.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"  - {c}: {n}次")
        
        return "\n".join(lines)

    def get_status(self) -> Dict[str, Any]:
        """获取 LLM 状态"""
        if self.llm_config is None:
            return {'enabled': False, 'available': False}
        
        return {
            'enabled': self.llm_config.enabled,
            'available': self.is_available(),
            'provider': self.llm_config.provider.value,
            'model': self.llm_config.model,
            'api_configured': bool(self.llm_config.api_key)
        }

    def configure(self, provider: str, api_key: str, model: str = "", 
                  api_base: str = "", temperature: float = 0.7):
        """动态配置 LLM"""
        try:
            p = LLMProvider(provider.lower())
        except ValueError:
            p = LLMProvider.OPENAI
        
        default = DEFAULT_CONFIGS.get(p, {})
        
        self.llm_config = LLMConfig(
            provider=p,
            api_key=api_key,
            api_base=api_base or default.get('api_base', ''),
            model=model or default.get('model', ''),
            temperature=temperature,
            enabled=True
        )
        
        logging.info(f"[LLM引擎] 配置更新: provider={p.value}, model={self.llm_config.model}")
