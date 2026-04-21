#!/usr/bin/env python3
"""
插件开发SDK
提供完整的插件开发工具和接口
"""

import os
import json
import logging
import importlib
import inspect
from typing import Dict, Any, List, Optional, Type, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from abc import ABC, abstractmethod
from enum import Enum


class AlgorithmCategory(Enum):
    """算法类别"""
    FACE_RECOGNITION = "face_recognition"
    STRUCTURED_ANALYSIS = "structured_analysis"
    PERSON_VIOLATION = "person_violation"
    ENVIRONMENT_ABNORMAL = "environment_abnormal"
    PERIMETER_ALERT = "perimeter_alert"
    BEHAVIOR_ALERT = "behavior_alert"


@dataclass
class AlgorithmMetadata:
    """算法元数据"""
    algorithm_id: int
    name: str
    description: str
    category: AlgorithmCategory
    version: str = "1.0.0"
    author: str = "Unknown"
    tags: List[str] = field(default_factory=list)
    min_confidence: float = 0.5
    supported_inputs: List[str] = field(default_factory=lambda: ["image", "video"])
    output_format: Dict[str, Any] = field(default_factory=dict)
    config_schema: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionResult:
    """检测结果"""
    detected: bool
    confidence: float
    bounding_box: Optional[tuple] = None
    label: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    extra_data: Dict[str, Any] = field(default_factory=dict)


class BaseAlgorithm(ABC):
    """算法基类 - 所有插件算法必须继承此类"""
    
    @classmethod
    @abstractmethod
    def get_metadata(cls) -> AlgorithmMetadata:
        """获取算法元数据"""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any] = None) -> bool:
        """初始化算法"""
        pass
    
    @abstractmethod
    def process(self, frame, context: Dict[str, Any] = None) -> DetectionResult:
        """处理帧"""
        pass
    
    def cleanup(self):
        """清理资源"""
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """验证配置"""
        metadata = self.get_metadata()
        schema = metadata.config_schema
        
        if not schema:
            return True
        
        for key, spec in schema.items():
            if spec.get('required', False) and key not in config:
                logging.error(f"[SDK] 缺少必需配置: {key}")
                return False
            
            if key in config:
                value = config[key]
                expected_type = spec.get('type')
                
                if expected_type == 'number':
                    if not isinstance(value, (int, float)):
                        logging.error(f"[SDK] 配置类型错误: {key} 应为数字")
                        return False
                    
                    min_val = spec.get('minimum')
                    max_val = spec.get('maximum')
                    if min_val is not None and value < min_val:
                        logging.error(f"[SDK] 配置值过小: {key} >= {min_val}")
                        return False
                    if max_val is not None and value > max_val:
                        logging.error(f"[SDK] 配置值过大: {key} <= {max_val}")
                        return False
                
                elif expected_type == 'string':
                    if not isinstance(value, str):
                        logging.error(f"[SDK] 配置类型错误: {key} 应为字符串")
                        return False
                    
                    enum_values = spec.get('enum')
                    if enum_values and value not in enum_values:
                        logging.error(f"[SDK] 配置值无效: {key} 应为 {enum_values} 之一")
                        return False
                
                elif expected_type == 'boolean':
                    if not isinstance(value, bool):
                        logging.error(f"[SDK] 配置类型错误: {key} 应为布尔值")
                        return False
        
        return True


class PluginSDK:
    """插件开发SDK"""
    
    SDK_VERSION = "1.0.0"
    
    @staticmethod
    def create_plugin_template(plugin_dir: str, plugin_name: str, algorithm_id: int):
        """创建插件模板"""
        plugin_path = Path(plugin_dir) / plugin_name
        plugin_path.mkdir(parents=True, exist_ok=True)
        
        plugin_config = {
            "name": plugin_name,
            "version": "1.0.0",
            "description": f"{plugin_name} 算法插件",
            "author": "Your Name",
            "sdk_version": PluginSDK.SDK_VERSION,
            "algorithms": [
                {
                    "id": algorithm_id,
                    "name": f"{plugin_name}检测",
                    "class": "MainAlgorithm",
                    "category": "structured_analysis",
                    "config": {}
                }
            ]
        }
        
        with open(plugin_path / "plugin.json", 'w', encoding='utf-8') as f:
            json.dump(plugin_config, f, indent=2, ensure_ascii=False)
        
        algorithm_code = f'''#!/usr/bin/env python3
"""
{plugin_name} 算法插件
使用 AI Box Plugin SDK v{PluginSDK.SDK_VERSION} 开发
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional

from plugin_sdk import BaseAlgorithm, AlgorithmMetadata, AlgorithmCategory, DetectionResult


class MainAlgorithm(BaseAlgorithm):
    """主算法类"""
    
    @classmethod
    def get_metadata(cls) -> AlgorithmMetadata:
        """获取算法元数据"""
        return AlgorithmMetadata(
            algorithm_id={algorithm_id},
            name="{plugin_name}检测",
            description="检测{plugin_name}的算法",
            category=AlgorithmCategory.STRUCTURED_ANALYSIS,
            version="1.0.0",
            author="Your Name",
            tags=["{plugin_name}", "detection"],
            min_confidence=0.5,
            config_schema={{
                "threshold": {{
                    "type": "number",
                    "default": 0.5,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "检测阈值"
                }}
            }}
        )
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {{}}
        self.threshold = self.config.get('threshold', 0.5)
        self.initialized = False
    
    def initialize(self, config: Dict[str, Any] = None) -> bool:
        """初始化算法"""
        if config:
            self.config.update(config)
            self.threshold = self.config.get('threshold', 0.5)
        
        if not self.validate_config(self.config):
            return False
        
        self.initialized = True
        return True
    
    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> DetectionResult:
        """处理帧"""
        if not self.initialized:
            return DetectionResult(detected=False, confidence=0.0)
        
        h, w = frame.shape[:2]
        
        brightness = np.mean(frame)
        
        detected = brightness > 100
        confidence = min(1.0, brightness / 255) if detected else 0.0
        
        return DetectionResult(
            detected=detected and confidence >= self.threshold,
            confidence=confidence,
            bounding_box=(0, 0, w, h) if detected else None,
            label="{plugin_name}",
            attributes={{
                "brightness": float(brightness)
            }}
        )
    
    def cleanup(self):
        """清理资源"""
        self.initialized = False
'''
        
        with open(plugin_path / "algorithm.py", 'w', encoding='utf-8') as f:
            f.write(algorithm_code)
        
        readme_content = f'''# {plugin_name} 插件

## 描述
{plugin_name} 检测算法插件

## 版本
- 插件版本: 1.0.0
- SDK版本: {PluginSDK.SDK_VERSION}

## 配置参数
| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| threshold | number | 0.5 | 检测阈值 (0.0-1.0) |

## 使用方法
1. 将插件目录放置在 `plugins/` 文件夹下
2. 重启 AI Box 服务
3. 插件将自动加载

## 开发指南
参考 `algorithm.py` 中的实现

## API 参考
- `get_metadata()`: 返回算法元数据
- `initialize(config)`: 初始化算法
- `process(frame, context)`: 处理帧并返回结果
- `cleanup()`: 清理资源
'''
        
        with open(plugin_path / "README.md", 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        logging.info(f"[SDK] 创建插件模板: {plugin_path}")
        return str(plugin_path)
    
    @staticmethod
    def validate_plugin(plugin_dir: str) -> Dict[str, Any]:
        """验证插件"""
        plugin_path = Path(plugin_dir)
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        config_file = plugin_path / "plugin.json"
        if not config_file.exists():
            results['valid'] = False
            results['errors'].append("缺少 plugin.json 配置文件")
            return results
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            required_fields = ['name', 'version', 'algorithms']
            for field in required_fields:
                if field not in config:
                    results['valid'] = False
                    results['errors'].append(f"配置文件缺少必需字段: {field}")
            
            results['info']['name'] = config.get('name')
            results['info']['version'] = config.get('version')
            results['info']['algorithm_count'] = len(config.get('algorithms', []))
            
        except json.JSONDecodeError as e:
            results['valid'] = False
            results['errors'].append(f"配置文件JSON格式错误: {e}")
            return results
        
        algorithm_file = plugin_path / "algorithm.py"
        if not algorithm_file.exists():
            results['valid'] = False
            results['errors'].append("缺少 algorithm.py 算法文件")
        else:
            try:
                import sys
                sys.path.insert(0, str(plugin_path))
                
                import importlib.util
                spec = importlib.util.spec_from_file_location("algorithm", algorithm_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                algorithm_class = None
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and issubclass(obj, BaseAlgorithm) and obj != BaseAlgorithm:
                        algorithm_class = obj
                        break
                
                if algorithm_class is None:
                    results['valid'] = False
                    results['errors'].append("未找到继承自 BaseAlgorithm 的算法类")
                else:
                    try:
                        metadata = algorithm_class.get_metadata()
                        results['info']['algorithm_name'] = metadata.name
                        results['info']['algorithm_id'] = metadata.algorithm_id
                        results['info']['algorithm_category'] = metadata.category.value
                    except Exception as e:
                        results['warnings'].append(f"获取算法元数据失败: {e}")
                
            except Exception as e:
                results['valid'] = False
                results['errors'].append(f"算法文件加载失败: {e}")
        
        return results
    
    @staticmethod
    def get_algorithm_template(category: AlgorithmCategory) -> str:
        """获取算法模板代码"""
        templates = {
            AlgorithmCategory.FACE_RECOGNITION: '''
class FaceRecognitionAlgorithm(BaseAlgorithm):
    """人脸识别算法模板"""
    
    @classmethod
    def get_metadata(cls) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            algorithm_id=0,
            name="人脸识别",
            description="人脸检测与识别",
            category=AlgorithmCategory.FACE_RECOGNITION,
            config_schema={
                "model_path": {"type": "string", "required": True},
                "confidence_threshold": {"type": "number", "default": 0.8}
            }
        )
    
    def process(self, frame, context=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            return DetectionResult(
                detected=True,
                confidence=0.85,
                bounding_box=(int(x), int(y), int(w), int(h)),
                label="face"
            )
        
        return DetectionResult(detected=False, confidence=0.0)
''',
            AlgorithmCategory.PERSON_VIOLATION: '''
class PersonViolationAlgorithm(BaseAlgorithm):
    """人员违规检测算法模板"""
    
    @classmethod
    def get_metadata(cls) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            algorithm_id=0,
            name="人员违规检测",
            description="检测人员违规行为",
            category=AlgorithmCategory.PERSON_VIOLATION,
            config_schema={
                "roi": {"type": "array", "description": "检测区域"},
                "violation_type": {"type": "string", "enum": ["intrusion", "loitering"]}
            }
        )
    
    def process(self, frame, context=None):
        detections = self.detector.detect(frame)
        
        for det in detections:
            if self._in_roi(det.bbox):
                return DetectionResult(
                    detected=True,
                    confidence=det.confidence,
                    bounding_box=det.bbox,
                    label="violation"
                )
        
        return DetectionResult(detected=False, confidence=0.0)
''',
        }
        
        return templates.get(category, "")


class PluginTester:
    """插件测试器"""
    
    def __init__(self, plugin_dir: str):
        self.plugin_dir = Path(plugin_dir)
        self.results = []
    
    def run_tests(self) -> Dict[str, Any]:
        """运行测试"""
        results = {
            'passed': 0,
            'failed': 0,
            'tests': []
        }
        
        validation = PluginSDK.validate_plugin(str(self.plugin_dir))
        if not validation['valid']:
            results['failed'] += 1
            results['tests'].append({
                'name': 'validation',
                'passed': False,
                'errors': validation['errors']
            })
            return results
        
        results['passed'] += 1
        results['tests'].append({
            'name': 'validation',
            'passed': True
        })
        
        return results


def generate_plugin_documentation(plugin_dir: str) -> str:
    """生成插件文档"""
    plugin_path = Path(plugin_dir)
    config_file = plugin_path / "plugin.json"
    
    if not config_file.exists():
        return "# 错误: 找不到插件配置文件"
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    doc = f'''# {config.get('name', 'Unknown')} 插件文档

## 基本信息
- **名称**: {config.get('name')}
- **版本**: {config.get('version')}
- **作者**: {config.get('author', 'Unknown')}
- **描述**: {config.get('description', '')}

## 算法列表

'''
    
    for algo in config.get('algorithms', []):
        doc += f'''### {algo.get('name')}

- **ID**: {algo.get('id')}
- **类别**: {algo.get('category')}
- **类名**: {algo.get('class')}

#### 配置参数
```json
{json.dumps(algo.get('config', {}), indent=2, ensure_ascii=False)}
```

'''
    
    return doc
