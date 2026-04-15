#!/usr/bin/env python3
"""
GPU 加速模块
支持 TensorRT、ONNX Runtime、CUDA 加速
"""

import os
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np


class AcceleratorType(Enum):
    """加速器类型"""
    CPU = "cpu"
    CUDA = "cuda"
    TENSORRT = "tensorrt"
    ONNX = "onnx"
    OPENVINO = "openvino"


@dataclass
class AcceleratorInfo:
    """加速器信息"""
    type: AcceleratorType
    name: str
    memory_total: int = 0
    memory_free: int = 0
    compute_capability: str = ""
    available: bool = False


class GPUAccelerator:
    """GPU 加速器管理"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.accelerator_type = AcceleratorType.CPU
        self.device_id = self.config.get('device_id', 0)
        self.tensorrt_enabled = False
        self.cuda_enabled = False
        self.onnx_enabled = False
        self._detect_accelerator()
    
    def _detect_accelerator(self):
        """检测可用的加速器"""
        if self._check_tensorrt():
            self.accelerator_type = AcceleratorType.TENSORRT
            self.tensorrt_enabled = True
            logging.info("[GPU加速] TensorRT 可用")
        elif self._check_cuda():
            self.accelerator_type = AcceleratorType.CUDA
            self.cuda_enabled = True
            logging.info("[GPU加速] CUDA 可用")
        elif self._check_onnx_gpu():
            self.accelerator_type = AcceleratorType.ONNX
            self.onnx_enabled = True
            logging.info("[GPU加速] ONNX Runtime GPU 可用")
        else:
            self.accelerator_type = AcceleratorType.CPU
            logging.info("[GPU加速] 使用 CPU")
    
    def _check_cuda(self) -> bool:
        """检查 CUDA 是否可用"""
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    device_name = torch.cuda.get_device_name(self.device_id)
                    logging.info(f"[GPU加速] 检测到 CUDA 设备: {device_name}")
                    return True
        except ImportError:
            pass
        except Exception as e:
            logging.debug(f"[GPU加速] CUDA 检测失败: {e}")
        return False
    
    def _check_tensorrt(self) -> bool:
        """检查 TensorRT 是否可用"""
        try:
            import tensorrt as trt
            logging.info(f"[GPU加速] TensorRT 版本: {trt.__version__}")
            return True
        except ImportError:
            pass
        except Exception as e:
            logging.debug(f"[GPU加速] TensorRT 检测失败: {e}")
        return False
    
    def _check_onnx_gpu(self) -> bool:
        """检查 ONNX Runtime GPU 是否可用"""
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in providers:
                logging.info("[GPU加速] ONNX Runtime CUDA Provider 可用")
                return True
        except ImportError:
            pass
        except Exception as e:
            logging.debug(f"[GPU加速] ONNX Runtime GPU 检测失败: {e}")
        return False
    
    def get_info(self) -> AcceleratorInfo:
        """获取加速器信息"""
        info = AcceleratorInfo(
            type=self.accelerator_type,
            name="Unknown",
            available=self.accelerator_type != AcceleratorType.CPU
        )
        
        if self.cuda_enabled or self.tensorrt_enabled:
            try:
                import torch
                if torch.cuda.is_available():
                    info.name = torch.cuda.get_device_name(self.device_id)
                    info.memory_total = torch.cuda.get_device_properties(self.device_id).total_memory
                    info.memory_free = torch.cuda.memory_reserved(self.device_id)
                    info.compute_capability = str(torch.cuda.get_device_properties(self.device_id).major) + "." + str(torch.cuda.get_device_properties(self.device_id).minor)
            except Exception:
                pass
        
        return info
    
    def optimize_model(self, model_path: str, output_path: str = None) -> Optional[str]:
        """优化模型（转换为 TensorRT 或 ONNX）"""
        if not output_path:
            output_path = model_path.replace('.pt', '.engine' if self.tensorrt_enabled else '.onnx')
        
        if self.tensorrt_enabled:
            return self._export_tensorrt(model_path, output_path)
        elif self.cuda_enabled:
            return self._export_onnx(model_path, output_path)
        
        return None
    
    def _export_tensorrt(self, model_path: str, output_path: str) -> Optional[str]:
        """导出 TensorRT 引擎"""
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            model.export(format='engine', device=self.device_id)
            logging.info(f"[GPU加速] TensorRT 引擎导出成功: {output_path}")
            return output_path
        except Exception as e:
            logging.error(f"[GPU加速] TensorRT 导出失败: {e}")
            return None
    
    def _export_onnx(self, model_path: str, output_path: str) -> Optional[str]:
        """导出 ONNX 模型"""
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            model.export(format='onnx', opset=12, simplify=True)
            logging.info(f"[GPU加速] ONNX 模型导出成功: {output_path}")
            return output_path
        except Exception as e:
            logging.error(f"[GPU加速] ONNX 导出失败: {e}")
            return None
    
    def benchmark(self, model, input_shape: Tuple[int, int, int] = (3, 640, 640), 
                  iterations: int = 100) -> Dict[str, float]:
        """性能基准测试"""
        import cv2
        
        dummy_input = np.random.randint(0, 255, (input_shape[1], input_shape[2], input_shape[0]), dtype=np.uint8)
        
        times = []
        for _ in range(10):
            _ = model(dummy_input)
        
        for _ in range(iterations):
            start = time.perf_counter()
            _ = model(dummy_input)
            times.append(time.perf_counter() - start)
        
        avg_time = sum(times) / len(times)
        fps = 1.0 / avg_time
        
        return {
            'avg_time_ms': avg_time * 1000,
            'fps': fps,
            'min_time_ms': min(times) * 1000,
            'max_time_ms': max(times) * 1000,
            'accelerator': self.accelerator_type.value
        }


class TensorRTInference:
    """TensorRT 推理引擎"""
    
    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.stream = None
        self.buffers = []
        self._load_engine()
    
    def _load_engine(self):
        """加载 TensorRT 引擎"""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            logger = trt.Logger(trt.Logger.WARNING)
            
            with open(self.engine_path, 'rb') as f:
                runtime = trt.Runtime(logger)
                self.engine = runtime.deserialize_cuda_engine(f.read())
            
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()
            
            for binding in self.engine:
                size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                self.buffers.append(int(device_mem))
            
            logging.info(f"[TensorRT] 引擎加载成功: {self.engine_path}")
            
        except ImportError as e:
            logging.error(f"[TensorRT] 缺少依赖: {e}")
        except Exception as e:
            logging.error(f"[TensorRT] 加载失败: {e}")
    
    def infer(self, input_data: np.ndarray) -> Optional[np.ndarray]:
        """执行推理"""
        if self.context is None:
            return None
        
        try:
            import pycuda.driver as cuda
            
            input_data = input_data.astype(np.float32)
            cuda.memcpy_htod_async(self.buffers[0], input_data.ravel(), self.stream)
            
            self.context.execute_async_v2(self.buffers, self.stream.handle)
            
            output = np.empty(self.buffers[1].size, dtype=np.float32)
            cuda.memcpy_dtoh_async(output, self.buffers[1], self.stream)
            
            self.stream.synchronize()
            
            return output
        except Exception as e:
            logging.error(f"[TensorRT] 推理失败: {e}")
            return None


class ONNXInference:
    """ONNX Runtime 推理引擎"""
    
    def __init__(self, onnx_path: str, use_gpu: bool = True):
        self.onnx_path = onnx_path
        self.session = None
        self._load_session(use_gpu)
    
    def _load_session(self, use_gpu: bool):
        """加载 ONNX 会话"""
        try:
            import onnxruntime as ort
            
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
            
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.session = ort.InferenceSession(self.onnx_path, sess_options, providers=providers)
            
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            
            logging.info(f"[ONNX] 会话加载成功: {self.onnx_path}, Provider: {self.session.get_providers()}")
            
        except ImportError:
            logging.error("[ONNX] onnxruntime 未安装")
        except Exception as e:
            logging.error(f"[ONNX] 加载失败: {e}")
    
    def infer(self, input_data: np.ndarray) -> Optional[np.ndarray]:
        """执行推理"""
        if self.session is None:
            return None
        
        try:
            input_data = input_data.astype(np.float32)
            outputs = self.session.run(self.output_names, {self.input_name: input_data})
            return outputs[0]
        except Exception as e:
            logging.error(f"[ONNX] 推理失败: {e}")
            return None


_gpu_accelerator = None

def get_gpu_accelerator(config: Dict[str, Any] = None) -> GPUAccelerator:
    """获取 GPU 加速器单例"""
    global _gpu_accelerator
    if _gpu_accelerator is None:
        _gpu_accelerator = GPUAccelerator(config)
    return _gpu_accelerator
