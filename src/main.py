#!/usr/bin/env python3
"""
AI盒子主程序
即插即用的AI摄像头分析系统
"""

import os
import sys
import time
import logging
import signal
from pathlib import Path

if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.info("Starting module imports...")

try:
    from config_manager import ConfigManager
    logging.info("Imported config_manager successfully")
except Exception as e:
    logging.error(f"Error importing config_manager: {e}", exc_info=True)
    sys.exit(1)

try:
    from health_monitor import HealthMonitor
    logging.info("Imported health_monitor successfully")
except Exception as e:
    logging.error(f"Error importing health_monitor: {e}", exc_info=True)
    sys.exit(1)

try:
    from web_server import WebServer
    logging.info("Imported web_server successfully")
except Exception as e:
    logging.error(f"Error importing web_server: {e}", exc_info=True)
    sys.exit(1)

try:
    from ai_analyzer_simple import AIAnalyzer
    logging.info("Imported ai_analyzer (simplified) successfully")
except Exception as e:
    logging.error(f"Error importing ai_analyzer: {e}", exc_info=True)
    sys.exit(1)

try:
    from camera_manager import CameraManager
    logging.info("Imported camera_manager successfully")
except Exception as e:
    logging.error(f"Error importing camera_manager: {e}", exc_info=True)
    sys.exit(1)

try:
    from high_performance_camera_manager import HighPerformanceCameraManager
    logging.info("Imported high_performance_camera_manager successfully")
except Exception as e:
    logging.error(f"Error importing high_performance_camera_manager: {e}", exc_info=True)
    sys.exit(1)

try:
    from multiprocess_manager import MultiprocessManager
    logging.info("Imported multiprocess_manager successfully")
except Exception as e:
    logging.error(f"Error importing multiprocess_manager: {e}", exc_info=True)
    sys.exit(1)

try:
    from gpu_accelerator import get_gpu_accelerator
    logging.info("Imported gpu_accelerator successfully")
except Exception as e:
    logging.warning(f"GPU accelerator not available: {e}")

try:
    from plugin_manager import get_plugin_manager
    logging.info("Imported plugin_manager successfully")
except Exception as e:
    logging.warning(f"Plugin manager not available: {e}")

try:
    from cloud_client import get_cloud_client
    logging.info("Imported cloud_client successfully")
except Exception as e:
    logging.warning(f"Cloud client not available: {e}")

logging.info("All modules imported successfully")

class AIBox:
    def __init__(self, config_path):
        self.config = ConfigManager(config_path)
        self.setup_logging()
        
        self.camera_manager = None
        self.ai_analyzer = None
        self.web_server = None
        self.health_monitor = None
        self.process_manager = None
        self.gpu_accelerator = None
        self.plugin_manager = None
        self.cloud_client = None
        self.running = False
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.config.start_monitoring()
        self.config.add_callback(self._config_changed_callback)
        
        logging.info("=" * 50)
        logging.info(f"{self.config.get('system.name')} v{self.config.get('system.version')}")
        logging.info("=" * 50)

    def setup_logging(self):
        log_level_str = self.config.get('system.log_level', 'INFO')
        log_level = getattr(logging, log_level_str, logging.INFO)
        
        # 获取日志目录，确保在项目根目录下创建
        log_dir_str = self.config.get('storage.log_dir', '')
        
        handlers = [logging.StreamHandler()]
        
        # 只有配置了日志目录才创建文件日志
        if log_dir_str:
            # 确保日志目录在项目根目录下
            project_root = Path(__file__).parent.parent
            log_dir = project_root / log_dir_str
            log_dir.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_dir / 'ai-box.log'))
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers
        )

    def _signal_handler(self, sig, frame):
        logging.info(f"Received signal {sig}, shutting down...")
        self.stop()

    def start(self):
        logging.info("Starting AI Box system...")
        
        try:
            # 自动检测并启用GPU加速（如果有GPU）
            gpu_enabled_config = self.config.get('gpu.enabled', False)
            gpu_auto_detected = False
            
            if not gpu_enabled_config:
                logging.info("[GPU] 检测GPU可用性...")
                try:
                    import torch
                    if torch.cuda.is_available():
                        device_name = torch.cuda.get_device_name(0)
                        device_count = torch.cuda.device_count()
                        props = torch.cuda.get_device_properties(0)
                        total_memory_gb = props.total_memory / (1024**3)
                        compute_capability = f"{props.major}.{props.minor}"
                        
                        logging.info(f"[GPU] ✅ 检测到 {device_count} 个GPU设备")
                        logging.info(f"[GPU]    型号: {device_name}")
                        logging.info(f"[GPU]    显存: {total_memory_gb:.1f} GB")
                        logging.info(f"[GPU]    计算能力: {compute_capability}")
                        
                        # 自动启用GPU
                        gpu_auto_detected = True
                        self.config.set('gpu.enabled', True)
                        logging.info("[GPU] 已自动启用GPU加速")
                    else:
                        logging.info("[GPU] 未检测到CUDA GPU，使用CPU模式")
                except ImportError:
                    logging.info("[GPU] PyTorch未安装，跳过GPU检测")
                except Exception as e:
                    logging.warning(f"[GPU] GPU检测失败: {e}")
            
            # 初始化GPU加速器
            if gpu_enabled_config or gpu_auto_detected:
                logging.info("Initializing GPU accelerator...")
                try:
                    from gpu_accelerator import get_gpu_accelerator
                    self.gpu_accelerator = get_gpu_accelerator(self.config.config)
                    info = self.gpu_accelerator.get_info()
                    
                    status = "可用" if info.available else "不可用"
                    memory_gb = info.memory_total / (1024**3) if info.memory_total > 0 else 0
                    
                    logging.info(f"[GPU] 加速器初始化成功")
                    logging.info(f"[GPU]    类型: {info.type.value}")
                    logging.info(f"[GPU]    名称: {info.name}")
                    logging.info(f"[GPU]    显存: {memory_gb:.1f} GB")
                    logging.info(f"[GPU]    状态: {status}")
                    logging.info(f"[GPU]    计算能力: {info.compute_capability}")
                except Exception as e:
                    logging.warning(f"[GPU] Accelerator init failed: {e}")
            
            if self.config.get('plugin.enabled', False):
                logging.info("Loading plugins...")
                try:
                    from plugin_manager import get_plugin_manager
                    self.plugin_manager = get_plugin_manager(self.config.config)
                    if self.config.get('plugin.auto_load', True):
                        self.plugin_manager.load_all_plugins()
                    logging.info(f"[Plugin] {len(self.plugin_manager.plugins)} plugins loaded")
                except Exception as e:
                    logging.warning(f"[Plugin] Manager init failed: {e}")
            
            if self.config.get('cloud.enabled', False):
                logging.info("Connecting to cloud...")
                try:
                    from cloud_client import get_cloud_client
                    self.cloud_client = get_cloud_client(self.config.config)
                    if self.cloud_client.is_available():
                        self.cloud_client.report_heartbeat({'status': 'starting'})
                        logging.info("[Cloud] Connected successfully")
                except Exception as e:
                    logging.warning(f"[Cloud] Connection failed: {e}")
            
            if self.config.get('multiprocess.enabled', True):
                logging.info("Using multiprocess architecture")
                self.process_manager = MultiprocessManager(self.config)
                self.process_manager.start()
                
                self.web_server = WebServer(self.config, None, None, self)
                self.web_server.set_process_manager(self.process_manager)
                self.health_monitor = HealthMonitor(self.config, self)
                
                if self.config.get('web.enabled'):
                    self.web_server.start()
                
                self.health_monitor.start()
            else:
                logging.info("Using traditional architecture")
                if self.config.get('camera.high_performance', False):
                    logging.info("Using high performance camera manager")
                    self.camera_manager = HighPerformanceCameraManager(self.config)
                else:
                    logging.info("Using standard camera manager")
                    self.camera_manager = CameraManager(self.config)
                
                self.ai_analyzer = AIAnalyzer(self.config)
                # 关键：将摄像头管理器连接到AI分析器，使其能获取摄像头设置的算法
                self.ai_analyzer.set_camera_manager(self.camera_manager)
                logging.info("[系统] AI分析器已连接摄像头管理器")
                
                self.web_server = WebServer(self.config, self.camera_manager, self.ai_analyzer, self)
                # 连接WebSocket广播到AI分析器（实时告警推送）
                self.ai_analyzer._web_server = self.web_server
                logging.info("[系统] WebSocket实时告警已启用")
                
                self.health_monitor = HealthMonitor(self.config, self)
                
                self.camera_manager.start()
                time.sleep(1)
                
                if self.config.get('ai.enabled', True):
                    self.ai_analyzer.start()
                    logging.info("[系统] AI分析器已启动，开始处理视频流")
                
                if self.config.get('web.enabled'):
                    self.web_server.start()
                
                self.health_monitor.start()
            
            self.running = True
            logging.info("AI Box system started successfully!")
            
            self._main_loop()
            
        except Exception as e:
            logging.error(f"Failed to start AI Box: {e}")
            self.stop()
            raise

    def _main_loop(self):
        while self.running:
            time.sleep(1)

    def stop(self):
        logging.info("Stopping AI Box system...")
        self.running = False
        
        if self.cloud_client and self.cloud_client.is_available():
            self.cloud_client.report_heartbeat({'status': 'stopping'})
        
        if self.health_monitor:
            self.health_monitor.stop()
        
        if self.web_server:
            self.web_server.stop()
        
        if self.ai_analyzer:
            self.ai_analyzer.stop()
        
        if self.camera_manager:
            self.camera_manager.stop()
        
        if self.process_manager:
            self.process_manager.stop()
        
        logging.info("AI Box system stopped.")

    def restart(self):
        logging.info("Restarting AI Box system...")
        self.stop()
        time.sleep(2)
        self.start()
    
    def _config_changed_callback(self, new_config):
        """配置变化回调处理"""
        logging.info("Configuration changed, applying updates...")
        
        # 处理不同类型的配置变化
        try:
            # 检查是否需要重启系统
            if self._needs_restart(new_config):
                logging.info("Configuration change requires system restart")
                self.restart()
            else:
                # 处理不需要重启的配置变化
                self._apply_config_updates(new_config)
        except Exception as e:
            logging.error(f"Error handling config change: {e}")
    
    def _needs_restart(self, new_config) -> bool:
        """检查是否需要重启系统"""
        # 这些配置变化需要重启系统
        restart_keys = [
            'multiprocess.enabled',
            'camera.high_performance',
            'web.host',
            'web.port'
        ]
        
        for key in restart_keys:
            old_value = self.config.get(key)
            new_value = self._get_nested_value(new_config, key)
            if old_value != new_value:
                return True
        
        return False
    
    def _get_nested_value(self, config, key):
        """获取嵌套配置值"""
        keys = key.split('.')
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        
        return value
    
    def _apply_config_updates(self, new_config):
        """应用不需要重启的配置更新"""
        # 处理AI算法配置
        if hasattr(self, 'ai_analyzer') and self.ai_analyzer:
            enabled_algorithms = new_config.get('ai', {}).get('enabled_algorithms', [])
            if enabled_algorithms:
                logging.info(f"Updating enabled algorithms: {enabled_algorithms}")
                # 这里可以添加更新算法的逻辑
        
        # 处理健康监控配置
        if hasattr(self, 'health_monitor') and self.health_monitor:
            check_interval = new_config.get('health', {}).get('check_interval')
            if check_interval:
                self.health_monitor.check_interval = check_interval
                logging.info(f"Updated health check interval: {check_interval}s")
        
        # 处理摄像头配置
        if hasattr(self, 'camera_manager') and self.camera_manager:
            # 这里可以添加摄像头配置更新的逻辑
            pass
        
        # 处理多进程配置
        if hasattr(self, 'process_manager') and self.process_manager:
            # 这里可以添加多进程配置更新的逻辑
            pass
        
        logging.info("Configuration updates applied successfully")

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.info("Starting AI Box main function...")
    
    # 获取配置文件的绝对路径
    config_path = os.getenv('AI_BOX_CONFIG', 'config/settings.yaml')
    if not os.path.isabs(config_path):
        # 如果是相对路径，转换为相对于项目根目录的绝对路径
        project_root = Path(__file__).parent.parent
        config_path = str(project_root / config_path)
    
    logging.info(f"Using config path: {config_path}")
    logging.info(f"Config file exists: {Path(config_path).exists()}")
    
    try:
        ai_box = AIBox(config_path)
        logging.info("AIBox instance created successfully")
        ai_box.start()
    except Exception as e:
        logging.error(f"Error in main function: {e}", exc_info=True)

if __name__ == '__main__':
    main()
