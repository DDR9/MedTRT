"""
Preprocessing Stage & Custom CUDA Kernel Benchmark (`benchmarks/benchmark_preprocessing.py`).

Evaluates:
1. NIfTI Disk Ingestion: On-the-fly Gzip (`.nii.gz`) vs. Uncompressed Memory-Mapped (`.nii` + H2D)
   vs. Pinned Host Memory (`pin_memory()` + asynchronous `non_blocking=True` H2D).
2. Spatial & Intensity Preprocessing (`Orientationd(RAS) + Spacingd(1.5, 1.5, 2.0) + ScaleIntensityRanged`):
   - MONAI CPU (`transforms="cpu"`)
   - MONAI GPU (`transforms="gpu"`, `grid_sample` with 3D meshgrid allocation)
   - Custom CUDA C++ Fused Kernel (`fused_preprocess_affine_trilinear_hu_f32`)
3. Standalone Intensity Normalization (`ScaleIntensityRanged`):
   - CPU PyTorch
   - GPU PyTorch (`torch.clamp` + affine scale)
   - Custom CUDA C++ Scalar (`hu_window_normalize_f32`)
   - Custom CUDA C++ Vectorized `float4` (`hu_window_normalize_vec4_f32`)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import torch
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged

from benchmarks.common import measure_cuda_callable, save_benchmark_artifact
from medtrt.config import MEDTRT_ROOT
from medtrt.cuda_kernels import cuda_fused_preprocess, cuda_hu_normalize
from medtrt.data import HELDOUT_JSON_PATH, ensure_uncompressed_nifti, load_nifti_mmap
from medtrt.infer import _spatial_and_intensity
from medtrt.timing import compute_distribution_stats


def run_preprocessing_benchmarks(warmup_iters: int = 5, timed_iters: int = 20) -> Dict[str, Any]:
    import json

    device = torch.device("cuda:0")
    with open(HELDOUT_JSON_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)["cases"]

    # Use representative large CT volume spleen_6 (512 x 512 x 135)
    case = next(c for c in cases if c["case_id"] == "spleen_6")
    img_path = MEDTRT_ROOT / case["image"]
    ensure_uncompressed_nifti(img_path)

    # 1. Ingestion comparison (load + H2D to cuda:0)
    gzip_loader = Compose([LoadImaged(keys=["image"]), EnsureChannelFirstd(keys=["image"])])

    def _ingest_gzip() -> torch.Tensor:
        d = gzip_loader({"image": str(img_path)})
        return d["image"].to(device, non_blocking=False)

    def _ingest_mmap() -> torch.Tensor:
        d = load_nifti_mmap({"image": str(img_path)})
        return d["image"].to(device, non_blocking=False)

    # Pre-pinned host buffer transfer benchmark
    d_sample = load_nifti_mmap({"image": str(img_path)})
    cpu_pageable = d_sample["image"].contiguous()
    cpu_pinned = cpu_pageable.pin_memory()

    h2d_pageable = measure_cuda_callable(
        lambda: cpu_pageable.to(device, non_blocking=False),
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        nvtx_name="h2d_pageable",
    )
    h2d_pinned_async = measure_cuda_callable(
        lambda: cpu_pinned.to(device, non_blocking=True),
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        nvtx_name="h2d_pinned_async",
    )
    ingest_gzip_stats = measure_cuda_callable(
        _ingest_gzip, warmup_iters=2, timed_iters=5, nvtx_name="ingest_gzip"
    )
    ingest_mmap_stats = measure_cuda_callable(
        _ingest_mmap, warmup_iters=3, timed_iters=10, nvtx_name="ingest_mmap"
    )

    # 2. Full Preprocessing (Orientation + Spacing + HU Normalization) on spleen_6
    pre_tf = Compose(_spatial_and_intensity())
    orig_affine_np = d_sample["image"].affine.cpu().numpy()
    raw_gpu = d_sample["image"].to(device)

    # CPU MONAI preprocessing
    cpu_times: List[float] = []
    for i in range(warmup_iters + 5):
        d_in = load_nifti_mmap({"image": str(img_path)})
        t0 = time.perf_counter()
        _ = pre_tf(d_in)
        t1 = time.perf_counter()
        if i >= warmup_iters:
            cpu_times.append((t1 - t0) * 1000.0)

    def _monai_gpu_pre() -> torch.Tensor:
        d_gpu = {"image": raw_gpu.clone()}
        return pre_tf(d_gpu)["image"]

    def _cuda_fused_pre() -> torch.Tensor:
        out, _, _, _ = cuda_fused_preprocess(raw_gpu, orig_affine_np)
        return out

    monai_gpu_stats = measure_cuda_callable(
        _monai_gpu_pre, warmup_iters=warmup_iters, timed_iters=timed_iters, nvtx_name="monai_gpu_preprocess"
    )
    cuda_fused_stats = measure_cuda_callable(
        _cuda_fused_pre, warmup_iters=warmup_iters, timed_iters=timed_iters, nvtx_name="cuda_fused_preprocess"
    )

    # Verify numerical accuracy between MONAI GPU and Custom CUDA Fused
    ref_pre = _monai_gpu_pre()
    cus_pre = _cuda_fused_pre()
    pre_diff = (cus_pre - ref_pre).abs()

    # 3. Standalone HU Windowing & Normalization on (1, 512, 512, 135) = 35.39M voxels (141.56 MB R + 141.56 MB W = 283.12 MB)
    vol_gpu = raw_gpu.to(torch.float32).contiguous()
    out_buf = torch.empty_like(vol_gpu)
    bytes_transferred = 2 * vol_gpu.numel() * 4  # 1 float32 read + 1 float32 write per element

    pt_hu_stats = measure_cuda_callable(
        lambda: torch.clamp((vol_gpu - (-57.0)) / (164.0 - (-57.0)), 0.0, 1.0),
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        nvtx_name="pytorch_hu_norm",
    )
    cuda_scalar_stats = measure_cuda_callable(
        lambda: cuda_hu_normalize(vol_gpu, vectorized=False, out=out_buf),
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        nvtx_name="cuda_hu_scalar",
    )
    cuda_vec4_stats = measure_cuda_callable(
        lambda: cuda_hu_normalize(vol_gpu, vectorized=True, out=out_buf),
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        nvtx_name="cuda_hu_vec4",
    )

    for st in (pt_hu_stats, cuda_scalar_stats, cuda_vec4_stats):
        med_s = st["cuda_event_ms"]["median"] / 1000.0
        st["achieved_bandwidth_gbps"] = round((bytes_transferred / 1e9) / max(med_s, 1e-9), 2)

    payload = {
        "case_id": "spleen_6",
        "shape": list(vol_gpu.shape),
        "ingestion": {
            "gzip_load_and_h2d": ingest_gzip_stats,
            "mmap_load_and_h2d": ingest_mmap_stats,
            "h2d_only_pageable": h2d_pageable,
            "h2d_only_pinned_async": h2d_pinned_async,
            "pinned_vs_pageable_speedup": round(
                h2d_pageable["cuda_event_ms"]["median"] / max(h2d_pinned_async["cuda_event_ms"]["median"], 1e-6), 2
            ),
        },
        "full_preprocess": {
            "monai_cpu_ms": compute_distribution_stats(cpu_times),
            "monai_gpu": monai_gpu_stats,
            "custom_cuda_fused": cuda_fused_stats,
            "speedup_cuda_vs_monai_gpu": round(
                monai_gpu_stats["cuda_event_ms"]["median"] / max(cuda_fused_stats["cuda_event_ms"]["median"], 1e-6), 2
            ),
            "vram_reduction_factor": round(
                monai_gpu_stats["incremental_allocated_vram_mb"]
                / max(cuda_fused_stats["incremental_allocated_vram_mb"], 0.1),
                2,
            ),
            "max_abs_diff_vs_monai": float(pre_diff.max().item()),
            "mean_abs_diff_vs_monai": float(pre_diff.mean().item()),
        },
        "hu_normalization_kernel": {
            "bytes_read_plus_write": bytes_transferred,
            "pytorch_cuda": pt_hu_stats,
            "custom_cuda_scalar": cuda_scalar_stats,
            "custom_cuda_vec4": cuda_vec4_stats,
        },
    }
    out_path = save_benchmark_artifact("benchmark_preprocessing.json", payload)
    print(f"[PASS] Saved preprocessing benchmark to {out_path}")
    return payload


if __name__ == "__main__":
    run_preprocessing_benchmarks()
