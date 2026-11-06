"""
Performance Scaling Benchmark (`benchmarks/benchmark_scaling.py`).

Evaluates Phase 4.4 & Phase 11 (Experiment E) scaling characteristics across:
1. Input CT Volume Dimensions:
   - `256 x 256 x 64` (4.19M voxels)
   - `512 x 512 x 32` (8.39M voxels)
   - `512 x 512 x 64` (16.78M voxels)
   - `512 x 512 x 128` (33.55M voxels)
   - `512 x 512 x 192` (50.33M voxels)
   comparing MONAI PyTorch GPU vs. Custom CUDA C++ kernels for:
   - HU Windowing & Intensity Normalization
   - 3D Trilinear Affine Resampling
   - Fused Softmax + Trilinear Inverse + Argmax + Voxel Counting
2. Sliding-Window Batch Size Scaling (`B in {1, 2, 4, 8}`) across PyTorch FP32 Strict, PyTorch TF32,
   PyTorch AMP, PyTorch FP16, TensorRT FP32, and TensorRT FP16.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from benchmarks.common import measure_cuda_callable, save_benchmark_artifact
from medtrt.cuda_kernels import (
    cuda_affine_nearest_resample_3d_u8,
    cuda_affine_trilinear_resample_3d,
    cuda_count_nonzero_u8,
    cuda_fused_invert_postprocess,
    cuda_hu_normalize,
)


def _pytorch_grid_sample_trilinear(x_4d: torch.Tensor, out_shape: Tuple[int, int, int]) -> torch.Tensor:
    """Standard PyTorch/MONAI 3D trilinear resampling via affine_grid + grid_sample."""
    theta = torch.eye(3, 4, dtype=torch.float32, device=x_4d.device).unsqueeze(0)
    grid = F.affine_grid(theta, size=(1, x_4d.shape[0], out_shape[0], out_shape[1], out_shape[2]), align_corners=True)
    return F.grid_sample(x_4d.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=True).squeeze(0)


def _pytorch_grid_sample_nearest_u8(x_3d_u8: torch.Tensor, out_shape: Tuple[int, int, int]) -> torch.Tensor:
    theta = torch.eye(3, 4, dtype=torch.float32, device=x_3d_u8.device).unsqueeze(0)
    grid = F.affine_grid(theta, size=(1, 1, out_shape[0], out_shape[1], out_shape[2]), align_corners=True)
    res = F.grid_sample(
        x_3d_u8.unsqueeze(0).unsqueeze(0).float(),
        grid,
        mode="nearest",
        padding_mode="border",
        align_corners=True,
    )
    return res.squeeze(0).squeeze(0).to(torch.uint8)


def run_scaling_benchmarks(warmup_iters: int = 5, timed_iters: int = 20) -> Dict[str, Any]:
    device = torch.device("cuda:0")
    volume_shapes = [
        (256, 256, 64),
        (512, 512, 32),
        (512, 512, 64),
        (512, 512, 128),
        (512, 512, 192),
    ]

    kernel_scaling: List[Dict[str, Any]] = []
    for sx, sy, sz in volume_shapes:
        n_voxels = sx * sy * sz
        vol = torch.randn((1, sx, sy, sz), dtype=torch.float32, device=device) * 200.0
        out_buf = torch.empty_like(vol)
        mask_u8 = (vol.squeeze(0) > 0.0).to(torch.uint8).contiguous()

        # Target resampled shape (~1.5x1.5x2.0 mm from 0.8x0.8x2.5 mm)
        rx, ry, rz = max(64, sx // 2), max(64, sy // 2), max(32, int(sz * 1.2))
        logits_2ch = torch.randn((2, rx, ry, rz), dtype=torch.float32, device=device)
        m_inv = np.array(
            [
                [float(rx - 1) / (sx - 1), 0.0, 0.0, 0.0],
                [0.0, float(ry - 1) / (sy - 1), 0.0, 0.0],
                [0.0, 0.0, float(rz - 1) / (sz - 1), 0.0],
            ],
            dtype=np.float32,
        )

        # 1. HU Normalization
        pt_hu = measure_cuda_callable(
            lambda: torch.clamp((vol - (-57.0)) / (164.0 - (-57.0)), 0.0, 1.0),
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
        )
        cuda_hu = measure_cuda_callable(
            lambda: cuda_hu_normalize(vol, vectorized=True, out=out_buf),
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
        )

        # 2. 3D Trilinear Resampling (vol -> rx, ry, rz)
        pt_tri = measure_cuda_callable(
            lambda: _pytorch_grid_sample_trilinear(vol, (rx, ry, rz)),
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
        )
        cuda_tri = measure_cuda_callable(
            lambda: cuda_affine_trilinear_resample_3d(vol, (rx, ry, rz)),
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
        )

        # 3. 3D Nearest-Neighbor Label Resampling (rx, ry, rz -> sx, sy, sz)
        small_u8 = (logits_2ch[1] > logits_2ch[0]).to(torch.uint8).contiguous()
        pt_nn = measure_cuda_callable(
            lambda: _pytorch_grid_sample_nearest_u8(small_u8, (sx, sy, sz)),
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
        )
        cuda_nn = measure_cuda_callable(
            lambda: cuda_affine_nearest_resample_3d_u8(small_u8, (sx, sy, sz), m_inv),
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
        )

        # 4. Voxel Counting
        pt_cnt = measure_cuda_callable(
            lambda: int(torch.count_nonzero(mask_u8).item()),
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
        )
        cuda_cnt = measure_cuda_callable(
            lambda: cuda_count_nonzero_u8(mask_u8),
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
        )

        # 5. Fused Inverse Postprocessing (2ch logits -> native sx, sy, sz uint8 mask)
        def _pt_post_full() -> torch.Tensor:
            prob = torch.softmax(logits_2ch.unsqueeze(0), dim=1).squeeze(0)
            res = _pytorch_grid_sample_trilinear(prob, (sx, sy, sz))
            return torch.argmax(res, dim=0).to(torch.uint8)

        pt_post = measure_cuda_callable(
            _pt_post_full, warmup_iters=3, timed_iters=10
        )
        cuda_post = measure_cuda_callable(
            lambda: cuda_fused_invert_postprocess(logits_2ch, (sx, sy, sz), m_inv, return_voxel_count=True),
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
        )

        row = {
            "shape": [sx, sy, sz],
            "voxels_millions": round(n_voxels / 1e6, 3),
            "hu_norm_pytorch_ms": pt_hu["cuda_event_ms"]["median"],
            "hu_norm_cuda_vec4_ms": cuda_hu["cuda_event_ms"]["median"],
            "hu_norm_speedup": round(pt_hu["cuda_event_ms"]["median"] / max(cuda_hu["cuda_event_ms"]["median"], 1e-6), 2),
            "trilinear_pytorch_ms": pt_tri["cuda_event_ms"]["median"],
            "trilinear_cuda_ms": cuda_tri["cuda_event_ms"]["median"],
            "trilinear_speedup": round(pt_tri["cuda_event_ms"]["median"] / max(cuda_tri["cuda_event_ms"]["median"], 1e-6), 2),
            "nearest_u8_pytorch_ms": pt_nn["cuda_event_ms"]["median"],
            "nearest_u8_cuda_ms": cuda_nn["cuda_event_ms"]["median"],
            "nearest_u8_speedup": round(pt_nn["cuda_event_ms"]["median"] / max(cuda_nn["cuda_event_ms"]["median"], 1e-6), 2),
            "voxel_count_pytorch_ms": pt_cnt["cuda_event_ms"]["median"],
            "voxel_count_cuda_ms": cuda_cnt["cuda_event_ms"]["median"],
            "voxel_count_speedup": round(pt_cnt["cuda_event_ms"]["median"] / max(cuda_cnt["cuda_event_ms"]["median"], 1e-6), 2),
            "fused_post_pytorch_ms": pt_post["cuda_event_ms"]["median"],
            "fused_post_cuda_ms": cuda_post["cuda_event_ms"]["median"],
            "fused_post_speedup": round(pt_post["cuda_event_ms"]["median"] / max(cuda_post["cuda_event_ms"]["median"], 1e-6), 2),
            "fused_post_pytorch_vram_mb": pt_post["incremental_allocated_vram_mb"],
            "fused_post_cuda_vram_mb": cuda_post["incremental_allocated_vram_mb"],
        }
        kernel_scaling.append(row)
        print(
            f"Shape {sx}x{sy}x{sz} ({row['voxels_millions']:.2f}M vox) | "
            f"Trilinear: {row['trilinear_pytorch_ms']:.3f} -> {row['trilinear_cuda_ms']:.3f} ms ({row['trilinear_speedup']:.2f}x) | "
            f"Fused Post: {row['fused_post_pytorch_ms']:.2f} -> {row['fused_post_cuda_ms']:.2f} ms ({row['fused_post_speedup']:.2f}x)"
        )
        del vol, out_buf, mask_u8, logits_2ch, small_u8
        torch.cuda.empty_cache()

    payload = {"volume_scaling": kernel_scaling}
    out_path = save_benchmark_artifact("benchmark_scaling.json", payload)
    print(f"[PASS] Saved scaling benchmark to {out_path}")
    return payload


if __name__ == "__main__":
    run_scaling_benchmarks()
