#!/usr/bin/env python3
"""
YOLOv8 推理引擎
提供统一的目标检测能力，供所有算法共享

支持：
- 多种模型大小：n(nano)/s(small)/m(medium)/l(large)/x(xlarge)
- GPU加速：CUDA/TensorRT/ONNX Runtime
- 自动回退：OpenCV DNN
"""

import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from enum import Enum

import cv2
import numpy as np


class ModelSize(Enum):
    """YOLO模型大小"""
    NANO = 'n'
    SMALL = 's'
    MEDIUM = 'm'
    LARGE = 'l'
    XLARGE = 'x'


class AcceleratorType(Enum):
    """加速器类型"""
    CPU = 'cpu'
    CUDA = 'cuda'
    TENSORRT = 'tensorrt'
    ONNX = 'onnx'
    OPENVINO = 'openvino'


MODEL_INFO = {
    ModelSize.NANO: {'name': 'yolov8n.pt', 'params': '3.2M', 'speed': 'fastest', 'accuracy': 'basic'},
    ModelSize.SMALL: {'name': 'yolov8s.pt', 'params': '11.2M', 'speed': 'fast', 'accuracy': 'good'},
    ModelSize.MEDIUM: {'name': 'yolov8m.pt', 'params': '25.9M', 'speed': 'medium', 'accuracy': 'better'},
    ModelSize.LARGE: {'name': 'yolov8l.pt', 'params': '43.7M', 'speed': 'slow', 'accuracy': 'best'},
    ModelSize.XLARGE: {'name': 'yolov8x.pt', 'params': '68.2M', 'speed': 'slowest', 'accuracy': 'best'},
}


class YOLOEngine:
    """YOLOv8 推理引擎（单例模式）"""

    _instance = None
    _lock = __import__('threading').Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, model_size: str = 'n', accelerator: str = 'auto'):
        if self._initialized:
            return
        self._initialized = True
        
        try:
            self.model_size = ModelSize(model_size.lower())
        except ValueError:
            self.model_size = ModelSize.NANO
            logging.warning(f"[YOLO引擎] 未知模型大小 '{model_size}'，使用默认 nano")
        
        self.accelerator = self._detect_accelerator(accelerator)
        self.model = None
        self.model_name = MODEL_INFO[self.model_size]['name']
        self.model_path = self._find_model(self.model_name)
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.device = self._get_device()
        
        self._load_model()

    @classmethod
    def reset(cls):
        """重置单例，允许用新参数重新创建实例"""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._initialized = False
                cls._instance = None

    def _detect_accelerator(self, accelerator: str) -> AcceleratorType:
        """检测可用的加速器"""
        if accelerator == 'auto':
            if self._check_tensorrt():
                return AcceleratorType.TENSORRT
            elif self._check_cuda():
                return AcceleratorType.CUDA
            elif self._check_onnx():
                return AcceleratorType.ONNX
            else:
                return AcceleratorType.CPU
        else:
            try:
                return AcceleratorType(accelerator.lower())
            except ValueError:
                return AcceleratorType.CPU

    def _check_cuda(self) -> bool:
        """检查CUDA是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _check_tensorrt(self) -> bool:
        """检查TensorRT是否可用"""
        try:
            import tensorrt
            return True
        except ImportError:
            return False

    def _check_onnx(self) -> bool:
        """检查ONNX Runtime是否可用"""
        try:
            import onnxruntime
            return 'CUDAExecutionProvider' in onnxruntime.get_available_providers()
        except ImportError:
            return False

    def _get_device(self) -> str:
        """获取推理设备"""
        if self.accelerator in [AcceleratorType.CUDA, AcceleratorType.TENSORRT]:
            return 'cuda:0'
        elif self.accelerator == AcceleratorType.ONNX:
            return 'cuda' if self._check_cuda() else 'cpu'
        else:
            return 'cpu'

    def _find_model(self, model_name: str) -> Optional[str]:
        """查找模型文件"""
        search_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'yolo', model_name),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', model_name),
            model_name,
        ]
        for path in search_paths:
            if os.path.exists(path):
                return path
        return None

    def _load_model(self):
        """加载 YOLOv8 模型"""
        try:
            from ultralytics import YOLO
            
            if self.accelerator == AcceleratorType.TENSORRT:
                self._load_tensorrt_model()
            elif self.accelerator == AcceleratorType.ONNX:
                self._load_onnx_model()
            else:
                self._load_pytorch_model()
                
            logging.info(f"[YOLO引擎] 加载模型: {self.model_name}, "
                        f"大小={self.model_size.value}, "
                        f"加速器={self.accelerator.value}, "
                        f"设备={self.device}")
                        
        except ImportError:
            logging.warning("[YOLO引擎] ultralytics 未安装，使用 OpenCV DNN 回退")
            self.model = None
        except Exception as e:
            logging.error(f"[YOLO引擎] 加载模型失败: {e}")
            self.model = None

    def _load_pytorch_model(self):
        """加载PyTorch模型"""
        from ultralytics import YOLO
        
        if self.model_path and os.path.exists(self.model_path):
            self.model = YOLO(self.model_path)
        else:
            self.model = YOLO(self.model_name)
            self._save_downloaded_model()
        
        if self.device.startswith('cuda'):
            self.model.to(self.device)

    def _load_tensorrt_model(self):
        """加载TensorRT模型"""
        from ultralytics import YOLO
        
        trt_model_name = self.model_name.replace('.pt', '.engine')
        trt_path = self._find_model(trt_model_name)
        
        if trt_path and os.path.exists(trt_path):
            self.model = YOLO(trt_path)
            logging.info(f"[YOLO引擎] 加载TensorRT引擎: {trt_path}")
        else:
            self.model = YOLO(self.model_name)
            self.model.to(self.device)
            logging.info(f"[YOLO引擎] TensorRT引擎不存在，使用PyTorch模型")

    def _load_onnx_model(self):
        """加载ONNX模型"""
        from ultralytics import YOLO
        
        onnx_model_name = self.model_name.replace('.pt', '.onnx')
        onnx_path = self._find_model(onnx_model_name)
        
        if onnx_path and os.path.exists(onnx_path):
            self.model = YOLO(onnx_path)
            logging.info(f"[YOLO引擎] 加载ONNX模型: {onnx_path}")
        else:
            self.model = YOLO(self.model_name)
            if self.model_path:
                onnx_export_path = self.model_path.replace('.pt', '.onnx')
                self.model.export(format='onnx', opset=12, simplify=True)
                logging.info(f"[YOLO引擎] 导出ONNX模型: {onnx_export_path}")

    def _save_downloaded_model(self):
        """保存自动下载的模型"""
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'yolo')
        os.makedirs(model_dir, exist_ok=True)
        save_path = os.path.join(model_dir, self.model_name)
        if not os.path.exists(save_path):
            import shutil
            src = self.model.ckpt_path if hasattr(self.model, 'ckpt_path') else None
            if src and os.path.exists(src):
                shutil.copy2(src, save_path)

    def detect(self, frame: np.ndarray, classes: List[int] = None,
               conf: float = None) -> List[Dict[str, Any]]:
        if self.model is None:
            return self._fallback_detect(frame, classes)

        try:
            conf = conf or self.conf_threshold
            results = self.model(frame, conf=conf, iou=self.iou_threshold,
                                 classes=classes, verbose=False, device=self.device)

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                for i in range(len(boxes)):
                    try:
                        box_data = boxes.xyxy[i].cpu().numpy()
                        if isinstance(box_data, (tuple, list)):
                            box = [int(box_data[0]), int(box_data[1]), int(box_data[2]), int(box_data[3])]
                        else:
                            box = [int(box_data[0]), int(box_data[1]), int(box_data[2]), int(box_data[3])]
                        
                        cls_id = int(boxes.cls[i].cpu().numpy())
                        confidence = float(boxes.conf[i].cpu().numpy())
                        cls_name = self.model.names.get(cls_id, str(cls_id))

                        detections.append({
                            'bbox': box,
                            'confidence': confidence,
                            'class_id': cls_id,
                            'class_name': cls_name,
                        })
                    except Exception as box_err:
                        logging.debug(f"[YOLO引擎] 跳过无效检测框: {box_err}")
                        continue
            return detections
        except Exception as e:
            logging.error(f"[YOLO引擎] 检测失败: {e}")
            return self._fallback_detect(frame, classes)

    def _fallback_detect(self, frame: np.ndarray, classes: List[int] = None) -> List[Dict[str, Any]]:
        """OpenCV DNN 回退检测"""
        detections = []

        if classes is None or any(c in (classes or []) for c in [0]):
            try:
                if not self._is_safe_for_hog(frame):
                    return detections
                hog = cv2.HOGDescriptor()
                hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                found, _ = hog.detectMultiScale(frame, winStride=(8, 8), padding=(4, 4), scale=1.05)
                for (x, y, bw, bh) in found:
                    detections.append({
                        'bbox': [int(x), int(y), int(x + bw), int(y + bh)],
                        'confidence': 0.3,
                        'class_id': 0,
                        'class_name': 'person',
                    })
            except Exception:
                pass

        return detections

    def _is_safe_for_hog(self, frame: np.ndarray) -> bool:
        """检查帧是否足够大以安全运行HOG检测"""
        if frame is None or frame.size == 0:
            return False
        h, w = frame.shape[:2]
        return h >= 128 and w >= 128

    def is_available(self) -> bool:
        """检查模型是否可用"""
        return self.model is not None

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = MODEL_INFO.get(self.model_size, {})
        return {
            'name': self.model_name,
            'size': self.model_size.value,
            'params': info.get('params', 'unknown'),
            'speed': info.get('speed', 'unknown'),
            'accuracy': info.get('accuracy', 'unknown'),
            'accelerator': self.accelerator.value,
            'device': self.device,
            'available': self.is_available()
        }

    def get_coco_classes(self) -> Dict[int, str]:
        """获取 COCO 类别映射"""
        if self.model is not None:
            return self.model.names
        return {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus',
            7: 'truck', 9: 'traffic light', 11: 'stop sign', 13: 'bench',
            15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
        }

    def set_model_size(self, size: str):
        """动态切换模型大小"""
        try:
            new_size = ModelSize(size.lower())
            if new_size != self.model_size:
                self.model_size = new_size
                self.model_name = MODEL_INFO[self.model_size]['name']
                self.model_path = self._find_model(self.model_name)
                self._load_model()
                logging.info(f"[YOLO引擎] 切换模型: {self.model_name}")
        except ValueError:
            logging.warning(f"[YOLO引擎] 未知模型大小: {size}")

    def set_accelerator(self, accelerator: str):
        """动态切换加速器"""
        self.accelerator = self._detect_accelerator(accelerator)
        self.device = self._get_device()
        if self.model and self.device.startswith('cuda'):
            try:
                self.model.to(self.device)
                logging.info(f"[YOLO引擎] 切换设备: {self.device}")
            except Exception as e:
                logging.error(f"[YOLO引擎] 切换设备失败: {e}")


COCO_TO_ALGORITHM = {
    0: [1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    2: [10, 11, 27, 38, 39],
    3: [10, 11, 27, 37],
    5: [10, 11, 27],
    7: [10, 11, 27, 38, 39, 40],
    17: [65, 66, 67],
    18: [65, 66, 67],
    19: [65, 66, 67],
}

HELMET_COLORS = {
    'red': ([0, 0, 150], [50, 50, 255]),
    'yellow': ([20, 150, 150], [60, 255, 255]),
    'blue': ([100, 50, 0], [255, 150, 50]),
    'white': ([200, 200, 200], [255, 255, 255]),
    'orange': ([0, 100, 200], [50, 180, 255]),
}

FIRE_COLORS = {
    'flame_lower': {'lower': [0, 100, 200], 'upper': [50, 255, 255]},
    'flame_upper': {'lower': [0, 50, 150], 'upper': [50, 200, 255]},
}

SMOKE_COLOR = {
    'gray_smoke': {'lower': [0, 0, 100], 'upper': [180, 50, 200]},
    'white_smoke': {'lower': [0, 0, 180], 'upper': [180, 30, 255]},
}

_yolo_instance = None
_yolo_params = {}

def get_yolo_engine(model_size: str = 'n', accelerator: str = 'auto') -> YOLOEngine:
    """获取 YOLO 引擎单例（兼容旧接口）"""
    global _yolo_instance, _yolo_params
    new_params = {'model_size': model_size, 'accelerator': accelerator}
    if _yolo_instance is not None and _yolo_params != new_params:
        logging.info(f"[YOLO引擎] 参数变化 {_yolo_params} -> {new_params}，重新创建实例")
        YOLOEngine.reset()
        _yolo_instance = None
    if _yolo_instance is None:
        _yolo_params = new_params
        _yolo_instance = YOLOEngine(model_size=model_size, accelerator=accelerator)
    return _yolo_instance
