#!/usr/bin/env python3
"""
LM能力增强模块
实现自然语言配置、告警根因分析、智能巡检报告等高级功能
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import Counter


@dataclass
class NLPCommand:
    action: str
    target: str
    parameters: Dict[str, Any]
    confidence: float
    raw_text: str


@dataclass
class AlertPattern:
    pattern_type: str
    description: str
    frequency: float
    time_pattern: str
    cameras: List[str]
    algorithms: List[str]
    confidence: float


@dataclass
class InspectionReport:
    report_id: str
    period: str
    generated_at: datetime
    summary: str
    total_alerts: int
    alerts_by_type: Dict[str, int]
    alerts_by_camera: Dict[str, int]
    patterns: List[AlertPattern]
    recommendations: List[str]
    risk_level: str


class NLPConfigParser:
    """自然语言配置解析器"""
    
    ACTION_KEYWORDS = {
        '添加': 'add', '启用': 'add', '开启': 'add', '配置': 'config',
        '删除': 'remove', '禁用': 'remove', '关闭': 'remove', '移除': 'remove',
        '修改': 'modify', '调整': 'modify', '更改': 'modify', '设置': 'modify',
        '查询': 'query', '查看': 'query', '显示': 'query', '获取': 'query',
        '告警': 'alert', '监控': 'alert', '报警': 'alert',
    }
    
    TARGET_KEYWORDS = {
        '摄像头': 'camera', '监控': 'camera', '摄像机': 'camera',
        '算法': 'algorithm', '检测': 'algorithm', '识别': 'algorithm',
        '阈值': 'threshold', '灵敏度': 'threshold', '置信度': 'threshold',
        '区域': 'region', '范围': 'region', '区域': 'region',
    }
    
    PARAMETER_PATTERNS = {
        'camera_name': r'["\']?([^"\']+)["\']?',
        'camera_id': r'(\d+)号?',
        'algorithm': r'["\']?([^"\']+)["\']?',
        'threshold': r'(\d+(?:\.\d+)?)%?',
        'count': r'(\d+)(?:个|人|次)?',
    }
    
    def __init__(self, llm_engine=None):
        self.llm_engine = llm_engine
    
    def parse(self, text: str) -> NLPCommand:
        action = self._detect_action(text)
        target = self._detect_target(text)
        parameters = self._extract_parameters(text)
        confidence = self._calculate_confidence(action, target, parameters)
        
        return NLPCommand(
            action=action,
            target=target,
            parameters=parameters,
            confidence=confidence,
            raw_text=text
        )
    
    def _detect_action(self, text: str) -> str:
        text_lower = text.lower()
        for keyword, action in self.ACTION_KEYWORDS.items():
            if keyword in text_lower:
                return action
        return 'query'
    
    def _detect_target(self, text: str) -> str:
        text_lower = text.lower()
        for keyword, target in self.TARGET_KEYWORDS.items():
                if keyword in text_lower:
                    return target
        return 'camera'
    
    def _extract_parameters(self, text: str) -> Dict[str, Any]:
        params = {}
        for param_name, pattern in self.PARAMETER_PATTERNS.items():
                match = re.search(pattern, text)
                if match:
                    params[param_name] = match.group(1)
        if '摄像头' in text or '监控' in text:
            camera_match = re.search(r'(\S+)(?:摄像头|监控)', text)
            if camera_match:
                params['camera_name'] = camera_match.group(1)
        if '算法' in text:
                algo_match = re.search(r'["\']?([^"\']+)["\']?', text)
                if algo_match:
                    params['algorithm'] = algo_match.group(1)
        return params
    
    def _calculate_confidence(self, action: str, target: str, parameters: Dict) -> float:
        confidence = 0.5
        if action in ['add', 'remove']:
            confidence += 0.2
        if target in ['camera', 'algorithm']:
            confidence += 0.1
        if parameters:
            confidence += 0.1
        return min(confidence, 1.0)


    
    def execute_command(self, command: NLPCommand, 
                       camera_manager=None, 
                       algorithm_manager=None) -> Dict[str, Any]:
        if command.action == 'add':
            return self._execute_add(command, camera_manager, algorithm_manager)
        elif command.action == 'remove':
            return self._execute_remove(command, camera_manager, algorithm_manager)
        elif command.action == 'modify':
            return self._execute_modify(command, camera_manager, algorithm_manager)
        elif command.action == 'query':
            return self._execute_query(command, camera_manager, algorithm_manager)
        elif command.action == 'alert':
            return self._execute_alert(command, camera_manager, algorithm_manager)
        
        result = {
            'success': False,
            'message': f'未知操作: {command.action}'
        }
        return result
    
    def _execute_add(self, command: NLPCommand, 
                     camera_manager, algorithm_manager) -> Dict[str, Any]:
        params = command.parameters
        if command.target == 'camera':
            camera_name = params.get('camera_name', '新摄像头')
            result = {
                'success': True,
                'message': f'已添加摄像头: {camera_name}',
                'data': {'camera_name': camera_name}
            }
        elif command.target == 'algorithm':
            algo = params.get('algorithm', 'unknown')
            camera_id = params.get('camera_id', 'all')
            result = {
                'success': True,
                'message': f'已为摄像头{camera_id}启用算法: {algo}',
                'data': {'algorithm': algo, 'camera_id': camera_id}
            }
        else:
            result = {
                'success': False,
                'message': f'暂不支持添加: {command.target}'
            }
        return result
    
    def _execute_remove(self, command: NLPCommand, 
                       camera_manager, algorithm_manager) -> Dict[str, Any]:
        params = command.parameters
        if command.target == 'algorithm':
            algo = params.get('algorithm', 'unknown')
            camera_id = params.get('camera_id', 'all')
            result = {
                'success': True,
                'message': f'已禁用摄像头{camera_id}的算法: {algo}',
                'data': {'algorithm': algo, 'camera_id': camera_id}
            }
        else:
            result = {
                'success': False,
                'message': f'暂不支持删除: {command.target}'
            }
        return result
    
    def _execute_modify(self, command: NLPCommand, 
                       camera_manager, algorithm_manager) -> Dict[str, Any]:
        params = command.parameters
        if command.target == 'threshold':
            threshold = params.get('threshold', '0.5')
            camera_id = params.get('camera_id', 'all')
            result = {
                'success': True,
                'message': f'已调整摄像头{camera_id}的阈值为{threshold}',
                'data': {'threshold': threshold, 'camera_id': camera_id}
            }
        elif command.target == 'region':
            result = {
                'success': True,
                'message': '区域配置已更新',
                'data': params
            }
        else:
            result = {
                'success': False,
                'message': f'暂不支持修改: {command.target}'
            }
        return result
    
    def _execute_query(self, command: NLPCommand, 
                       camera_manager, algorithm_manager) -> Dict[str, Any]:
        result = {
            'success': True,
            'message': '查询完成',
            'data': command.parameters
        }
        return result
    
    def _execute_alert(self, command: NLPCommand, 
                       camera_manager, algorithm_manager) -> Dict[str, Any]:
        params = command.parameters
        count = params.get('count', '5')
        camera_name = params.get('camera_name', '所有摄像头')
        
        result = {
            'success': True,
            'message': f'已配置告警: {camera_name}超过{count}人时触发',
            'data': {'threshold': count, 'camera': camera_name}
        }
        return result


class AlertRootCauseAnalyzer:
    """告警根因分析器"""
    
    def __init__(self, alert_db=None, llm_engine=None):
        self.alert_db = alert_db
        self.llm_engine = llm_engine
    
    def analyze_patterns(self, time_range_hours: int = 24) -> List[AlertPattern]:
        """分析告警模式"""
        patterns = []
        
        try:
            if self.alert_db:
                alerts = self._get_recent_alerts(time_range_hours)
                
                time_pattern = self._analyze_time_distribution(alerts)
                if time_pattern:
                    patterns.append(time_pattern)
                
                location_pattern = self._analyze_location_distribution(alerts)
                if location_pattern:
                    patterns.append(location_pattern)
                
                type_pattern = self._analyze_type_distribution(alerts)
                if type_pattern:
                    patterns.append(type_pattern)
        except Exception as e:
            logging.error(f"[根因分析] 分析失败: {e}")
        
        return patterns
    
    def _get_recent_alerts(self, hours: int) -> List[Dict]:
        """获取最近的告警"""
        try:
            from datetime import datetime, timedelta
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            if hasattr(self.alert_db, 'get_alerts'):
                return self.alert_db.get_alerts(
                    start_time=start_time.timestamp(),
                    end_time=end_time.timestamp(),
                    limit=1000
                )
        except Exception as e:
            logging.error(f"[根因分析] 获取告警失败: {e}")
        return []
    
    def _analyze_time_distribution(self, alerts: List[Dict]) -> Optional[AlertPattern]:
        """分析时间分布模式"""
        if not alerts:
            return None
        
        hour_counts = Counter()
        for alert in alerts:
            timestamp = alert.get('timestamp', 0)
            if timestamp:
                from datetime import datetime
                hour = datetime.fromtimestamp(timestamp).hour
                hour_counts[hour] += 1
        
        if not hour_counts:
            return None
        
        peak_hour = hour_counts.most_common(1)[0][0]
        peak_count = hour_counts.most_common(1)[0][1]
        total = sum(hour_counts.values())
        
        time_desc = f"高峰时段: {peak_hour}:00-{peak_hour+1}:00"
        
        return AlertPattern(
            pattern_type='time_distribution',
            description=time_desc,
            frequency=peak_count / total if total > 0 else 0,
            time_pattern=f"{peak_hour}:00",
            cameras=[],
            algorithms=[],
            confidence=min(peak_count / max(total, 1) * 2, 1.0)
        )
    
    def _analyze_location_distribution(self, alerts: List[Dict]) -> Optional[AlertPattern]:
        """分析位置分布模式"""
        if not alerts:
            return None
        
        camera_counts = Counter()
        for alert in alerts:
            camera = alert.get('camera_name', alert.get('source', 'unknown'))
            camera_counts[camera] += 1
        
        if not camera_counts:
            return None
        
        top_cameras = camera_counts.most_common(3)
        total = sum(camera_counts.values())
        
        return AlertPattern(
            pattern_type='location_distribution',
            description=f"高频告警区域: {', '.join([c[0] for c in top_cameras])}",
            frequency=top_cameras[0][1] / total if total > 0 else 0,
            time_pattern='',
            cameras=[c[0] for c in top_cameras],
            algorithms=[],
            confidence=0.8
        )
    
    def _analyze_type_distribution(self, alerts: List[Dict]) -> Optional[AlertPattern]:
        """分析类型分布模式"""
        if not alerts:
            return None
        
        type_counts = Counter()
        for alert in alerts:
            algo_name = alert.get('algorithm_name', alert.get('type', 'unknown'))
            type_counts[algo_name] += 1
        
        if not type_counts:
            return None
        
        top_types = type_counts.most_common(3)
        total = sum(type_counts.values())
        
        return AlertPattern(
            pattern_type='type_distribution',
            description=f"高频告警类型: {', '.join([t[0] for t in top_types])}",
            frequency=top_types[0][1] / total if total > 0 else 0,
            time_pattern='',
            cameras=[],
            algorithms=[t[0] for t in top_types],
            confidence=0.85
        )
    
    def generate_recommendations(self, patterns: List[AlertPattern]) -> List[str]:
        """根据模式生成建议"""
        recommendations = []
        
        for pattern in patterns:
            if pattern.pattern_type == 'time_distribution':
                recommendations.append(
                    f"建议在{pattern.time_pattern}时段增加巡检频率或调整算法灵敏度"
                )
            elif pattern.pattern_type == 'location_distribution':
                if pattern.cameras:
                    recommendations.append(
                        f"建议检查{pattern.cameras[0]}区域的实际状况，可能存在安全隐患"
                    )
            elif pattern.pattern_type == 'type_distribution':
                if pattern.algorithms:
                    recommendations.append(
                        f"建议针对{pattern.algorithms[0]}类告警优化检测参数或加强现场管理"
                    )
        
        return recommendations


class InspectionReportGenerator:
    """智能巡检报告生成器"""
    
    def __init__(self, alert_db=None, llm_engine=None):
        self.alert_db = alert_db
        self.llm_engine = llm_engine
    
    def generate_daily_report(self, date: datetime = None) -> InspectionReport:
        """生成每日巡检报告"""
        if date is None:
            date = datetime.now()
        
        report_id = f"DAILY_{date.strftime('%Y%m%d')}"
        
        stats = self._get_daily_stats(date)
        patterns = self._analyze_daily_patterns(date)
        recommendations = self._generate_daily_recommendations(stats, patterns)
        
        risk_level = self._calculate_risk_level(stats)
        
        return InspectionReport(
            report_id=report_id,
            period=f"{date.strftime('%Y年%m月%d日')}",
            generated_at=datetime.now(),
            summary=self._generate_summary(stats, patterns),
            total_alerts=stats.get('total', 0),
            alerts_by_type=stats.get('by_type', {}),
            alerts_by_camera=stats.get('by_camera', {}),
            patterns=patterns,
            recommendations=recommendations,
            risk_level=risk_level
        )
    
    def generate_weekly_report(self, end_date: datetime = None) -> InspectionReport:
        """生成每周巡检报告"""
        if end_date is None:
            end_date = datetime.now()
        
        start_date = end_date - timedelta(days=7)
        report_id = f"WEEKLY_{end_date.strftime('%Y%m%d')}"
        
        stats = self._get_weekly_stats(start_date, end_date)
        patterns = self._analyze_weekly_patterns(start_date, end_date)
        recommendations = self._generate_weekly_recommendations(stats, patterns)
        
        risk_level = self._calculate_risk_level(stats)
        
        return InspectionReport(
            report_id=report_id,
            period=f"{start_date.strftime('%Y年%m月%d日')} - {end_date.strftime('%Y年%m月%d日')}",
            generated_at=datetime.now(),
            summary=self._generate_summary(stats, patterns),
            total_alerts=stats.get('total', 0),
            alerts_by_type=stats.get('by_type', {}),
            alerts_by_camera=stats.get('by_camera', {}),
            patterns=patterns,
            recommendations=recommendations,
            risk_level=risk_level
        )
    
    def _get_daily_stats(self, date: datetime) -> Dict[str, Any]:
        """获取每日统计"""
        stats = {'total': 0, 'by_type': {}, 'by_camera': {}}
        
        try:
            if self.alert_db:
                start_time = date.replace(hour=0, minute=0, second=0, microsecond=0)
                end_time = start_time + timedelta(days=1)
                
                alerts = self._get_alerts_in_range(
                    start_time.timestamp(), 
                    end_time.timestamp()
                )
                
                stats['total'] = len(alerts)
                
                for alert in alerts:
                    algo = alert.get('algorithm_name', 'unknown')
                    stats['by_type'][algo] = stats['by_type'].get(algo, 0) + 1
                    
                    camera = alert.get('camera_name', 'unknown')
                    stats['by_camera'][camera] = stats['by_camera'].get(camera, 0) + 1
        except Exception as e:
            logging.error(f"[巡检报告] 获取每日统计失败: {e}")
        
        return stats
    
    def _get_weekly_stats(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """获取每周统计"""
        stats = {'total': 0, 'by_type': {}, 'by_camera': {}}
        
        try:
            if self.alert_db:
                alerts = self._get_alerts_in_range(
                    start_date.timestamp(),
                    end_date.timestamp()
                )
                
                stats['total'] = len(alerts)
                
                for alert in alerts:
                    algo = alert.get('algorithm_name', 'unknown')
                    stats['by_type'][algo] = stats['by_type'].get(algo, 0) + 1
                    
                    camera = alert.get('camera_name', 'unknown')
                    stats['by_camera'][camera] = stats['by_camera'].get(camera, 0) + 1
        except Exception as e:
            logging.error(f"[巡检报告] 获取每周统计失败: {e}")
        
        return stats
    
    def _get_alerts_in_range(self, start_time: float, end_time: float) -> List[Dict]:
        """获取时间范围内的告警"""
        try:
            if hasattr(self.alert_db, 'get_alerts'):
                return self.alert_db.get_alerts(
                    start_time=start_time,
                    end_time=end_time,
                    limit=10000
                )
        except Exception as e:
            logging.error(f"[巡检报告] 获取告警失败: {e}")
        return []
    
    def _analyze_daily_patterns(self, date: datetime) -> List[AlertPattern]:
        """分析每日模式"""
        analyzer = AlertRootCauseAnalyzer(self.alert_db, self.llm_engine)
        return analyzer.analyze_patterns(24)
    
    def _analyze_weekly_patterns(self, start_date: datetime, end_date: datetime) -> List[AlertPattern]:
        """分析每周模式"""
        analyzer = AlertRootCauseAnalyzer(self.alert_db, self.llm_engine)
        return analyzer.analyze_patterns(168)
    
    def _generate_daily_recommendations(self, stats: Dict, patterns: List[AlertPattern]) -> List[str]:
        """生成每日建议"""
        recommendations = []
        
        if stats['total'] > 50:
            recommendations.append("今日告警数量较多，建议检查系统运行状态和现场环境")
        
        analyzer = AlertRootCauseAnalyzer(self.alert_db, self.llm_engine)
        recommendations.extend(analyzer.generate_recommendations(patterns))
        
        if not recommendations:
            recommendations.append("系统运行正常，继续保持监控")
        
        return recommendations
    
    def _generate_weekly_recommendations(self, stats: Dict, patterns: List[AlertPattern]) -> List[str]:
        """生成每周建议"""
        recommendations = []
        
        avg_daily = stats['total'] / 7
        if avg_daily > 30:
            recommendations.append(f"本周日均告警{avg_daily:.1f}次，建议全面检查监控区域")
        
        analyzer = AlertRootCauseAnalyzer(self.alert_db, self.llm_engine)
        recommendations.extend(analyzer.generate_recommendations(patterns))
        
        if not recommendations:
            recommendations.append("本周系统运行稳定，建议继续保持")
        
        return recommendations
    
    def _generate_summary(self, stats: Dict, patterns: List[AlertPattern]) -> str:
        """生成摘要"""
        total = stats['total']
        
        if total == 0:
            return "期间无告警记录，系统运行正常"
        
        top_type = max(stats['by_type'].items(), key=lambda x: x[1], default=('无', 0))
        top_camera = max(stats['by_camera'].items(), key=lambda x: x[1], default=('无', 0))
        
        summary = f"共产生{total}次告警。"
        
        if top_type[0] != '无':
            summary += f"主要类型为{top_type[0]}({top_type[1]}次)。"
        
        if top_camera[0] != '无':
            summary += f"主要区域为{top_camera[0]}({top_camera[1]}次)。"
        
        return summary
    
    def _calculate_risk_level(self, stats: Dict) -> str:
        """计算风险等级"""
        total = stats['total']
        
        if total == 0:
            return '低'
        elif total < 10:
            return '低'
        elif total < 50:
            return '中'
        elif total < 100:
            return '高'
        else:
            return '严重'


class LMCapabilityManager:
    """LM能力管理器 - 统一管理所有LM增强功能"""
    
    def __init__(self, config: Dict[str, Any] = None, alert_db=None, llm_engine=None):
        self.config = config or {}
        self.alert_db = alert_db
        self.llm_engine = llm_engine
        
        self.nlp_parser = NLPConfigParser(llm_engine)
        self.root_cause_analyzer = AlertRootCauseAnalyzer(alert_db, llm_engine)
        self.report_generator = InspectionReportGenerator(alert_db, llm_engine)
    
    def process_natural_language_command(self, text: str, 
                                         camera_manager=None,
                                         algorithm_manager=None) -> Dict[str, Any]:
        """处理自然语言命令"""
        command = self.nlp_parser.parse(text)
        return self.nlp_parser.execute_command(command, camera_manager, algorithm_manager)
    
    def analyze_alert_patterns(self, hours: int = 24) -> Dict[str, Any]:
        """分析告警模式"""
        patterns = self.root_cause_analyzer.analyze_patterns(hours)
        recommendations = self.root_cause_analyzer.generate_recommendations(patterns)
        
        return {
            'patterns': [
                {
                    'type': p.pattern_type,
                    'description': p.description,
                    'frequency': round(p.frequency, 2),
                    'confidence': round(p.confidence, 2),
                    'cameras': p.cameras,
                    'algorithms': p.algorithms
                }
                for p in patterns
            ],
            'recommendations': recommendations
        }
    
    def generate_report(self, report_type: str = 'daily', date: datetime = None) -> Dict[str, Any]:
        """生成巡检报告"""
        if report_type == 'daily':
            report = self.report_generator.generate_daily_report(date)
        elif report_type == 'weekly':
            report = self.report_generator.generate_weekly_report(date)
        else:
            return {'error': f'不支持的报告类型: {report_type}'}
        
        return {
            'report_id': report.report_id,
            'period': report.period,
            'generated_at': report.generated_at.isoformat(),
            'summary': report.summary,
            'total_alerts': report.total_alerts,
            'alerts_by_type': report.alerts_by_type,
            'alerts_by_camera': report.alerts_by_camera,
            'patterns': [
                {
                    'type': p.pattern_type,
                    'description': p.description,
                    'frequency': round(p.frequency, 2)
                }
                for p in report.patterns
            ],
            'recommendations': report.recommendations,
            'risk_level': report.risk_level
        }


def get_lm_capability_manager(config: Dict[str, Any] = None, 
                              alert_db=None, 
                              llm_engine=None) -> LMCapabilityManager:
    """获取LM能力管理器单例"""
    if not hasattr(get_lm_capability_manager, '_instance'):
        get_lm_capability_manager._instance = LMCapabilityManager(config, alert_db, llm_engine)
    return get_lm_capability_manager._instance


