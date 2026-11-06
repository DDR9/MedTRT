"""
Shared Benchmark Utilities & Metadata Controls (`benchmarks/common.py`).

Provides unified CUDA event timing, synchronized wall-clock measurement, hardware telemetry,
and statistical distribution summaries (`mean`, `median`, `std`, `iqr`, `min`, `max`, `p90`, `p95`,
`ci95_lower`, `ci95_upper`) across all modules in `benchmarks/`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch

MEDTRT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = MEDTRT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medtrt.bench import query_gpu_telemetry
from medtrt.config import RESULTS_DIR, run_metadata
from medtrt.timing import compute_distribution_stats, nvtx_range


def measure_cuda_callable(
    fn: Callable[[], Any],
    warmup_iters: int = 10,
    timed_iters: int = 30,
    nvtx_name: str = "cuda_kernel",
    device: str | torch.device = "cuda:0",
) -> Dict[str, Any]:
    """
    Time a GPU callable using `torch.cuda.Event(enable_timing=True)` and synchronized wall-clock
    timers over `warmup_iters` warm-up passes and `timed_iters` measured iterations.
    """
    dev = torch.device(device)
    torch.cuda.synchronize(dev)

    for _ in range(warmup_iters):
        _ = fn()
    torch.cuda.synchronize(dev)

    torch.cuda.reset_peak_memory_stats(dev)
    base_alloc = torch.cuda.memory_allocated(dev)

    cuda_ms_list: List[float] = []
    wall_ms_list: List[float] = []

    for _ in range(timed_iters):
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        with nvtx_range(nvtx_name):
            start_ev.record()
            _ = fn()
            end_ev.record()
            end_ev.synchronize()
        t1 = time.perf_counter()
        cuda_ms_list.append(float(start_ev.elapsed_time(end_ev)))
        wall_ms_list.append((t1 - t0) * 1000.0)

    peak_alloc_mb = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
    inc_alloc_mb = max(0.0, (torch.cuda.max_memory_allocated(dev) - base_alloc) / (1024 ** 2))
    peak_res_mb = torch.cuda.max_memory_reserved(dev) / (1024 ** 2)

    return {
        "warmup_iters": warmup_iters,
        "timed_iters": timed_iters,
        "cuda_event_ms": compute_distribution_stats(cuda_ms_list),
        "wall_clock_ms": compute_distribution_stats(wall_ms_list),
        "peak_allocated_vram_mb": round(float(peak_alloc_mb), 2),
        "incremental_allocated_vram_mb": round(float(inc_alloc_mb), 2),
        "peak_reserved_vram_mb": round(float(peak_res_mb), 2),
    }


def save_benchmark_artifact(filename: str, payload: Dict[str, Any]) -> Path:
    """Save benchmark result dictionary with provenance metadata to `results/<filename>`."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / filename
    full_payload = {
        "run": run_metadata(),
        "gpu_telemetry": query_gpu_telemetry(),
        **payload,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)
    return out_path
