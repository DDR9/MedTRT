"""
Zero-Copy TensorRT 10.x Python Runner (`src/medtrt/trt_runner.py`).

Engineering Features:
- Binds PyTorch CUDA tensor pointers directly (`tensor.data_ptr()`) into `context.set_tensor_address()`.
- Executes on a dedicated non-default CUDA stream (`self.exec_stream`) synchronized with the caller's
  PyTorch stream via lightweight `torch.cuda.Event` dependencies, eliminating TensorRT 10.x's
  default-stream warning and implicit host-synchronization penalty.
- Supports optional pre-allocated output buffer reuse (`reuse_output_buffer=True`) for static-shape
  patch loops, avoiding per-call `cudaMallocAsync` / caching-allocator overhead.
- Strictly zero host copies (`host_copies_count == 0`) in the inference loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import tensorrt as trt
import torch

from medtrt.export import (
    ENGINE_FP32_PATH,
    ENGINE_PATH,
    build_trt_fp16_engine,
    build_trt_fp32_engine,
)


class TRTRunner:
    """
    Zero-copy TensorRT 10.x execution wrapper directly compatible with MONAI `SlidingWindowInferer`.
    """

    def __init__(
        self,
        engine_path: str | Path = ENGINE_PATH,
        device: str | torch.device = "cuda:0",
        reuse_output_buffer: bool = False,
        auto_build: bool = False,
    ) -> None:
        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            if not auto_build and self.engine_path not in (ENGINE_PATH, ENGINE_FP32_PATH):
                raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")
            if "fp32" in self.engine_path.name.lower():
                build_trt_fp32_engine(engine_path=self.engine_path)
            else:
                build_trt_fp16_engine(engine_path=self.engine_path)

        self.device = torch.device(device)
        self.reuse_output_buffer = reuse_output_buffer
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)

        with open(self.engine_path, "rb") as f:
            engine_bytes = f.read()

        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")

        self.input_name = "input"
        self.output_name = "output"
        for idx in range(self.engine.num_io_tensors):
            t_name = self.engine.get_tensor_name(idx)
            t_mode = self.engine.get_tensor_mode(t_name)
            if t_mode == trt.TensorIOMode.INPUT:
                self.input_name = t_name
            elif t_mode == trt.TensorIOMode.OUTPUT:
                self.output_name = t_name

        # Dedicated non-default CUDA stream to avoid TensorRT enqueueV3 default-stream sync overhead
        self.exec_stream = torch.cuda.Stream(device=self.device)
        self._input_ready_event = torch.cuda.Event(enable_timing=False)
        self._output_ready_event = torch.cuda.Event(enable_timing=False)
        self._output_pool: Dict[Tuple[int, ...], torch.Tensor] = {}
        self._last_input_shape: Tuple[int, ...] | None = None

        self.engine_file_mb = self.engine_path.stat().st_size / (1024 ** 2)
        if hasattr(self.engine, "get_device_memory_size_v2"):
            ctx_bytes = self.engine.get_device_memory_size_v2()
        else:
            ctx_bytes = 0
        self.device_ctx_mb = float(ctx_bytes) / (1024 ** 2)
        self.trt_vram_mb = self.engine_file_mb + self.device_ctx_mb

        self.forward_calls_count: int = 0
        self.host_copies_count: int = 0

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Execute zero-copy TensorRT forward pass on dedicated non-default CUDA stream.
        """
        if not inputs.is_cuda:
            self.host_copies_count += 1
            raise RuntimeError(
                "TRTRunner received a non-CUDA tensor. Host tensors are forbidden in the inference loop."
            )

        x = (
            inputs
            if (inputs.is_contiguous() and inputs.dtype == torch.float32)
            else inputs.to(dtype=torch.float32).contiguous()
        )
        shape: Tuple[int, ...] = tuple(int(d) for d in x.shape)
        batch_size = shape[0]

        if shape != self._last_input_shape:
            if not self.context.set_input_shape(self.input_name, shape):
                raise RuntimeError(f"TensorRT set_input_shape failed for shape {shape}")
            self._last_input_shape = shape

        out_shape = (batch_size, 2, shape[2], shape[3], shape[4])
        if self.reuse_output_buffer:
            out = self._output_pool.get(out_shape)
            if out is None or out.device != x.device:
                out = torch.empty(out_shape, dtype=torch.float32, device=x.device)
                self._output_pool[out_shape] = out
        else:
            out = torch.empty(out_shape, dtype=torch.float32, device=x.device)

        caller_stream = torch.cuda.current_stream(x.device)
        self._input_ready_event.record(caller_stream)
        self.exec_stream.wait_event(self._input_ready_event)

        self.context.set_tensor_address(self.input_name, int(x.data_ptr()))
        self.context.set_tensor_address(self.output_name, int(out.data_ptr()))

        ok = self.context.execute_async_v3(stream_handle=self.exec_stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 returned False.")

        self._output_ready_event.record(self.exec_stream)
        caller_stream.wait_event(self._output_ready_event)

        self.forward_calls_count += 1
        return out
