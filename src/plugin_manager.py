#!/usr/bin/env python3
"""
插件系统
支持动态加载算法、第三方扩展
"""

import os
import sys
import json
import logging
import importlib
import importlib.util
from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum


class PluginStatus(Enum):
    """插件状态"""
    LOADED = "loaded"
    FAILED = "failed"
    DISABLED = "disabled"
    PENDING = "pending"


@dataclass
class PluginInfo:
    """插件信息"""
    name: str
    version: str
    description: str
    author: str
    algorithms: List[Dict[str, Any]]
    path: str
    status: PluginStatus = PluginStatus.PENDING
    error_message: str = ""
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlgorithmPlugin:
    """算法插件"""
    algorithm_id: int
    algorithm_name: str
    algorithm_class: Type
    category: str
    config: Dict[str, Any] = field(default_factory=dict)


class PluginManager:
    """插件管理器"""
    
    PLUGIN_DIR = "plugins"
    PLUGIN_CONFIG_FILE = "plugin.json"
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.plugins: Dict[str, PluginInfo] = {}
        self.algorithm_plugins: Dict[int, AlgorithmPlugin] = {}
        self.plugin_dir = Path(self.config.get('plugin_dir', self.PLUGIN_DIR))
        self._ensure_plugin_dir()
    
    def _ensure_plugin_dir(self):
        """确保插件目录存在"""
        if not self.plugin_dir.is_absolute():
            self.plugin_dir = Path(__file__).parent.parent / self.plugin_dir
        
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
        
        example_dir = self.plugin_dir / "example_algorithm"
        if not example_dir.exists():
            self._create_example_plugin(example_dir)
    
    def _create_example_plugin(self, plugin_dir: Path):
        """创建示例插件"""
        plugin_dir.mkdir(parents=True, exist_ok=True)
        
        plugin_config = {
            "name": "example_algorithm",
            "version": "1.0.0",
            "description": "示例算法插件",
            "author": "AI Box Team",
            "algorithms": [
                {
                    "id": 100,
                    "name": "示例检测",
                    "class": "ExampleAlgorithm",
                    "category": "structured_analysis",
                    "config": {
                        "threshold": 0.5
                    }
                }
            ]
        }
        
        with open(plugin_dir / self.PLUGIN_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(plugin_config, f, indent=2, ensure_ascii=False)
        
        algorithm_code = '''#!/usr/bin/env python3
"""示例算法插件"""

import cv2
import numpy as np
from typing import Dict, Any

class ExampleAlgorithm:
    """示例检测算法"""
    
    ALGORITHM_ID = 100
    ALGORITHM_NAME = "示例检测"
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.threshold = self.config.get('threshold', 0.5)
    
    def initialize(self) -> bool:
        return True
    
    def process(self, frame: np.ndarray, context: Dict[str, Any] = None) -> Dict[str, Any]:
        result = {
            'algorithm_id': self.ALGORITHM_ID,
            'algorithm_name': self.ALGORITHM_NAME,
            'detected': False,
            'confidence': 0.0,
            'bounding_box': None,
            'extra_data': {}
        }
        
        h, w = frame.shape[:2]
        brightness = np.mean(frame)
        
        if brightness > 100:
            result['detected'] = True
            result['confidence'] = min(1.0, brightness / 255)
            result['extra_data'] = {
                'brightness': brightness,
                'frame_size': f"{w}x{h}"
            }
        
        return result
'''
        
        with open(plugin_dir / "algorithm.py", 'w', encoding='utf-8') as f:
            f.write(algorithm_code)
        
        logging.info(f"[插件系统] 创建示例插件: {plugin_dir}")
    
    def discover_plugins(self) -> List[str]:
        """发现所有插件"""
        discovered = []
        
        if not self.plugin_dir.exists():
            return discovered
        
        for item in self.plugin_dir.iterdir():
            if item.is_dir():
                config_file = item / self.PLUGIN_CONFIG_FILE
                if config_file.exists():
                    discovered.append(str(item))
        
        logging.info(f"[插件系统] 发现 {len(discovered)} 个插件")
        return discovered
    
    def load_plugin(self, plugin_path: str) -> Optional[PluginInfo]:
        """加载插件"""
        plugin_dir = Path(plugin_path)
        config_file = plugin_dir / self.PLUGIN_CONFIG_FILE
        
        if not config_file.exists():
            logging.error(f"[插件系统] 插件配置文件不存在: {config_file}")
            return None
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                plugin_config = json.load(f)
            
            plugin_info = PluginInfo(
                name=plugin_config.get('name', plugin_dir.name),
                version=plugin_config.get('version', '1.0.0'),
                description=plugin_config.get('description', ''),
                author=plugin_config.get('author', 'Unknown'),
                algorithms=plugin_config.get('algorithms', []),
                path=str(plugin_dir),
                config=plugin_config.get('config', {})
            )
            
            if plugin_info.name in self.plugins:
                logging.warning(f"[插件系统] 插件已加载: {plugin_info.name}")
                return self.plugins[plugin_info.name]
            
            sys.path.insert(0, str(plugin_dir))
            
            for algo_config in plugin_info.algorithms:
                self._load_algorithm(plugin_dir, algo_config, plugin_info)
            
            plugin_info.status = PluginStatus.LOADED
            self.plugins[plugin_info.name] = plugin_info
            
            logging.info(f"[插件系统] 插件加载成功: {plugin_info.name} v{plugin_info.version}, "
                        f"{len(plugin_info.algorithms)} 个算法")
            
            return plugin_info
            
        except Exception as e:
            logging.error(f"[插件系统] 插件加载失败: {plugin_path}, 错误: {e}")
            return None
    
    def _load_algorithm(self, plugin_dir: Path, algo_config: Dict[str, Any], 
                       plugin_info: PluginInfo):
        """加载插件中的算法"""
        algo_id = algo_config.get('id')
        algo_name = algo_config.get('name', f'Algorithm_{algo_id}')
        algo_class_name = algo_config.get('class', 'Algorithm')
        algo_category = algo_config.get('category', 'structured_analysis')
        algo_config_dict = algo_config.get('config', {})
        
        try:
            module_name = "algorithm"
            spec = importlib.util.spec_from_file_location(
                module_name,
                plugin_dir / "algorithm.py"
            )
            
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                if hasattr(module, algo_class_name):
                    algo_class = getattr(module, algo_class_name)
                    
                    algo_plugin = AlgorithmPlugin(
                        algorithm_id=algo_id,
                        algorithm_name=algo_name,
                        algorithm_class=algo_class,
                        category=algo_category,
                        config=algo_config_dict
                    )
                    
                    self.algorithm_plugins[algo_id] = algo_plugin
                    logging.info(f"[插件系统] 算法加载成功: {algo_id} - {algo_name}")
                else:
                    logging.error(f"[插件系统] 算法类不存在: {algo_class_name}")
            else:
                logging.error(f"[插件系统] 无法加载算法模块: {plugin_dir / 'algorithm.py'}")
                
        except Exception as e:
            logging.error(f"[插件系统] 算法加载失败: {algo_name}, 错误: {e}")
    
    def load_all_plugins(self) -> Dict[str, PluginInfo]:
        """加载所有插件"""
        discovered = self.discover_plugins()
        
        for plugin_path in discovered:
            self.load_plugin(plugin_path)
        
        return self.plugins
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """卸载插件"""
        if plugin_name not in self.plugins:
            return False
        
        plugin_info = self.plugins[plugin_name]
        
        for algo_config in plugin_info.algorithms:
            algo_id = algo_config.get('id')
            if algo_id in self.algorithm_plugins:
                del self.algorithm_plugins[algo_id]
        
        del self.plugins[plugin_name]
        
        logging.info(f"[插件系统] 插件卸载成功: {plugin_name}")
        return True
    
    def get_algorithm(self, algorithm_id: int) -> Optional[AlgorithmPlugin]:
        """获取算法插件"""
        return self.algorithm_plugins.get(algorithm_id)
    
    def create_algorithm_instance(self, algorithm_id: int, 
                                  config: Dict[str, Any] = None) -> Optional[Any]:
        """创建算法实例"""
        algo_plugin = self.get_algorithm(algorithm_id)
        
        if algo_plugin is None:
            return None
        
        merged_config = {**algo_plugin.config, **(config or {})}
        
        try:
            instance = algo_plugin.algorithm_class(merged_config)
            return instance
        except Exception as e:
            logging.error(f"[插件系统] 创建算法实例失败: {algorithm_id}, 错误: {e}")
            return None
    
    def get_all_algorithms(self) -> List[Dict[str, Any]]:
        """获取所有插件算法"""
        algorithms = []
        
        for algo_id, algo_plugin in self.algorithm_plugins.items():
            algorithms.append({
                'id': algo_id,
                'name': algo_plugin.algorithm_name,
                'category': algo_plugin.category,
                'config': algo_plugin.config
            })
        
        return algorithms
    
    def get_status(self) -> Dict[str, Any]:
        """获取插件系统状态"""
        return {
            'plugin_count': len(self.plugins),
            'algorithm_count': len(self.algorithm_plugins),
            'plugins': {
                name: {
                    'version': info.version,
                    'status': info.status.value,
                    'algorithm_count': len(info.algorithms)
                }
                for name, info in self.plugins.items()
            }
        }


_plugin_manager = None

def get_plugin_manager(config: Dict[str, Any] = None) -> PluginManager:
    """获取插件管理器单例"""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager(config)
    return _plugin_manager
