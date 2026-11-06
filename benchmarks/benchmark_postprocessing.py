"""
Postprocessing Stage & Custom CUDA Kernel Benchmark (`benchmarks/benchmark_postprocessing.py`).

Evaluates:
1. Full Postprocessing (`Softmax + Inverse Resample to Native Space + Argmax Discretization`):
   - MONAI CPU Trilinear `Invertd` (Baseline host path)
   - MONAI GPU Trilinear `Invertd` (Bundle-faithful GPU `grid_sample`)
   - MONAI GPU Nearest-Neighbor `uint8` `Invertd` (`trt_fp16_nn`)
   - Custom CUDA C++ Fused Postprocess (`fused_softmax_2ch_p1_f32` + `fused_invert_trilinear_threshold_u8`)
2. Standalone Voxel Counting (`count_nonzero_u8` vs `torch.count_nonzero` vs `np.count_nonzero`).
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

import nibabel as nib
import numpy as np
import torch
from monai.transforms import Compose

from benchmarks.common import measure_cuda_callable, save_benchmark_artifact
from medtrt.config import MEDTRT_ROOT
from medtrt.cuda_kernels import (
    compute_voxel_to_voxel_matrix,
    cuda_count_nonzero_u8,
    cuda_fused_invert_postprocess,
)
from medtrt.data import HELDOUT_JSON_PATH, load_nifti_mmap
from medtrt.evaluate import compute_foreground_dice
from medtrt.infer import (
    _spatial_and_intensity,
    build_gpu_postprocessing,
    build_inference_postprocessing,
    build_nearest_label_postprocessing,
    build_sliding_window_inferer,
)
from medtrt.timing import compute_distribution_stats
from medtrt.trt_runner import TRTRunner


def run_postprocessing_benchmarks(warmup_iters: int = 5, timed_iters: int = 20) -> Dict[str, Any]:
    device = torch.device("cuda:0")
    with open(HELDOUT_JSON_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)["cases"]

    case = next(c for c in cases if c["case_id"] == "spleen_6")
    img_path = MEDTRT_ROOT / case["image"]
    lbl_path = MEDTRT_ROOT / case["label"]
    gt_mask = np.asarray(nib.load(str(lbl_path)).dataobj).astype(np.uint8)

    pre_tf = Compose(_spatial_and_intensity())
    post_cpu_tf = build_inference_postprocessing(pre_tf)
    post_gpu_trilinear_tf = build_gpu_postprocessing(pre_tf)
    post_gpu_nearest_tf = build_nearest_label_postprocessing(pre_tf)

    d = load_nifti_mmap({"image": str(img_path)})
    orig_affine = d["image"].affine.cpu().numpy()
    orig_shape = tuple(int(s) for s in d["image"].shape[1:])
    d["image"] = d["image"].to(device)
    d = pre_tf(d)
    res_affine = d["image"].affine.cpu().numpy()
    m_inv = compute_voxel_to_voxel_matrix(res_affine, orig_affine)

    runner = TRTRunner(device=device)
    inferer = build_sliding_window_inferer(sw_batch_size=8, overlap=0.25)
    with torch.no_grad():
        logits_gpu = inferer(inputs=d["image"].unsqueeze(0), network=runner)[0].detach().float()
    logits_cpu = logits_gpu.cpu()

    # 1. CPU MONAI Trilinear Invertd (3 timed passes due to ~5s duration per pass)
    cpu_ms: List[float] = []
    mask_cpu = None
    for i in range(4):
        d_cpu = {"image": d["image"].cpu(), "pred": logits_cpu.clone()}
        t0 = time.perf_counter()
        out_c = post_cpu_tf(d_cpu)["pred"]
        t1 = time.perf_counter()
        if i >= 1:
            cpu_ms.append((t1 - t0) * 1000.0)
        mask_cpu = out_c.squeeze(0).numpy().astype(np.uint8)

    # 2. GPU MONAI Trilinear Invertd
    def _gpu_trilinear() -> torch.Tensor:
        d_g = {"image": d["image"], "pred": logits_gpu}
        return post_gpu_trilinear_tf(d_g)["pred"]

    gpu_trilinear_stats = measure_cuda_callable(
        _gpu_trilinear, warmup_iters=warmup_iters, timed_iters=timed_iters, nvtx_name="post_monai_gpu_trilinear"
    )
    mask_gpu_tri = _gpu_trilinear().squeeze(0).cpu().numpy().astype(np.uint8)

    # 3. GPU MONAI Nearest-Neighbor uint8 Invertd
    def _gpu_nearest() -> torch.Tensor:
        d_g = {"image": d["image"], "pred": logits_gpu}
        return post_gpu_nearest_tf(d_g)["pred"]

    gpu_nearest_stats = measure_cuda_callable(
        _gpu_nearest, warmup_iters=warmup_iters, timed_iters=timed_iters, nvtx_name="post_monai_gpu_nearest"
    )
    mask_gpu_nn = _gpu_nearest().squeeze(0).cpu().numpy().astype(np.uint8)

    # 4. Custom CUDA C++ Fused Postprocess (Softmax + Register-Affine Trilinear + Argmax + Voxel Count)
    def _cuda_fused() -> torch.Tensor:
        m_out, _ = cuda_fused_invert_postprocess(logits_gpu, orig_shape, m_inv, return_voxel_count=True)
        return m_out

    cuda_fused_stats = measure_cuda_callable(
        _cuda_fused, warmup_iters=warmup_iters, timed_iters=timed_iters, nvtx_name="post_cuda_fused"
    )
    mask_cuda_t, fg_count_cuda = cuda_fused_invert_postprocess(logits_gpu, orig_shape, m_inv, return_voxel_count=True)
    mask_cuda = mask_cuda_t.squeeze(0).cpu().numpy().astype(np.uint8)

    # 5. Standalone Voxel Counting on (1, 512, 512, 135) uint8 mask
    pt_count_stats = measure_cuda_callable(
        lambda: int(torch.count_nonzero(mask_cuda_t).item()),
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        nvtx_name="pytorch_count_nonzero",
    )
    cuda_count_stats = measure_cuda_callable(
        lambda: cuda_count_nonzero_u8(mask_cuda_t),
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        nvtx_name="cuda_count_nonzero_u8",
    )

    payload = {
        "case_id": "spleen_6",
        "native_shape": list(orig_shape),
        "resampled_shape": list(logits_gpu.shape[1:]),
        "methods": {
            "monai_cpu_trilinear": {
                "latency_ms": compute_distribution_stats(cpu_ms),
                "dice": round(compute_foreground_dice(mask_cpu, gt_mask), 6),
                "voxel_agreement_vs_monai_gpu_pct": round(float((mask_cpu == mask_gpu_tri).mean() * 100.0), 5),
            },
            "monai_gpu_trilinear": {
                **gpu_trilinear_stats,
                "dice": round(compute_foreground_dice(mask_gpu_tri, gt_mask), 6),
                "voxel_agreement_vs_monai_gpu_pct": 100.0,
            },
            "monai_gpu_nearest_u8": {
                **gpu_nearest_stats,
                "dice": round(compute_foreground_dice(mask_gpu_nn, gt_mask), 6),
                "voxel_agreement_vs_monai_gpu_pct": round(float((mask_gpu_nn == mask_gpu_tri).mean() * 100.0), 5),
            },
            "custom_cuda_fused_trilinear": {
                **cuda_fused_stats,
                "dice": round(compute_foreground_dice(mask_cuda, gt_mask), 6),
                "fg_voxel_count": fg_count_cuda,
                "voxel_agreement_vs_monai_gpu_pct": round(float((mask_cuda == mask_gpu_tri).mean() * 100.0), 5),
            },
        },
        "speedups": {
            "custom_cuda_vs_monai_cpu": round(
                float(np.median(cpu_ms)) / max(cuda_fused_stats["cuda_event_ms"]["median"], 1e-6), 2
            ),
            "custom_cuda_vs_monai_gpu_trilinear": round(
                gpu_trilinear_stats["cuda_event_ms"]["median"]
                / max(cuda_fused_stats["cuda_event_ms"]["median"], 1e-6),
                2,
            ),
            "custom_cuda_vs_monai_gpu_nearest": round(
                gpu_nearest_stats["cuda_event_ms"]["median"]
                / max(cuda_fused_stats["cuda_event_ms"]["median"], 1e-6),
                2,
            ),
            "vram_reduction_vs_monai_gpu_trilinear": round(
                gpu_trilinear_stats["incremental_allocated_vram_mb"]
                / max(cuda_fused_stats["incremental_allocated_vram_mb"], 0.1),
                2,
            ),
        },
        "voxel_counting": {
            "pytorch_count_nonzero": pt_count_stats,
            "custom_cuda_count_nonzero_u8": cuda_count_stats,
            "exact_count_match": bool(fg_count_cuda == int(np.count_nonzero(mask_cuda == 1))),
        },
    }
    out_path = save_benchmark_artifact("benchmark_postprocessing.json", payload)
    print(f"[PASS] Saved postprocessing benchmark to {out_path}")
    return payload


if __name__ == "__main__":
    run_postprocessing_benchmarks()
