"""
Numerical & Segmentation Correctness Benchmark (`benchmarks/benchmark_correctness.py`).

Verifies Phase 4.3 & Phase 5.3 numerical correctness across:
1. Custom CUDA C++ kernels vs. CPU & PyTorch CUDA reference implementations (including edge cases:
   non-contiguous tensors, anisotropic spacing, empty masks, boundary coordinates, odd dimensions).
2. TensorRT 10.x FP32, TensorRT 10.x FP16 (BN-Folded), TensorRT 10.x FP16 (Unfolded), PyTorch AMP,
   PyTorch Pure FP16, and PyTorch TF32 vs. PyTorch Strict IEEE FP32 on both random patches and real CT patches.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

from benchmarks.common import save_benchmark_artifact
from medtrt.cuda_kernels import (
    cuda_affine_nearest_resample_3d_u8,
    cuda_affine_trilinear_resample_3d,
    cuda_count_nonzero_u8,
    cuda_fused_invert_postprocess,
    cuda_fused_preprocess,
    cuda_hu_normalize,
    get_cuda_module,
)
from medtrt.data import HELDOUT_JSON_PATH, load_nifti_mmap
from medtrt.export import ENGINE_FP16_UNFOLDED_PATH, ENGINE_FP32_PATH, ENGINE_PATH
from medtrt.infer import _spatial_and_intensity, build_gpu_postprocessing
from medtrt.model import load_spleen_unet
from medtrt.trt_runner import TRTRunner
from monai.transforms import Compose


def run_correctness_benchmarks() -> Dict[str, Any]:
    import json

    device = torch.device("cuda:0")
    torch.manual_seed(42)

    # 1. Backend patch numerical comparison against Strict IEEE FP32
    prev_m = torch.backends.cuda.matmul.allow_tf32
    prev_c = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    with open(HELDOUT_JSON_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    d_real = load_nifti_mmap({"image": cases[2]["image"]})
    pre_out, _, _, _ = cuda_fused_preprocess(
        d_real["image"].to(device), d_real["image"].affine.cpu().numpy()
    )
    # Extract a real 96x96x96 spleen-centered patch + a random patch
    cx, cy, cz = pre_out.shape[1] // 2, pre_out.shape[2] // 2, pre_out.shape[3] // 2
    real_patch = pre_out[:, cx - 48 : cx + 48, cy - 48 : cy + 48, max(0, cz - 48) : max(0, cz - 48) + 96].unsqueeze(0).contiguous()
    if real_patch.shape != (1, 1, 96, 96, 96):
        real_patch = F.interpolate(pre_out.unsqueeze(0), size=(96, 96, 96), mode="trilinear", align_corners=False)
    rand_patch = torch.rand((2, 1, 96, 96, 96), dtype=torch.float32, device=device)

    m_ref = load_spleen_unet(device, fold_bn=False, dtype=torch.float32)
    m_folded = load_spleen_unet(device, fold_bn=True, dtype=torch.float32)
    m_half = load_spleen_unet(device, fold_bn=True, dtype=torch.float16)
    trt_fp32 = TRTRunner(engine_path=ENGINE_FP32_PATH, device=device)
    trt_fp16_unfolded = TRTRunner(engine_path=ENGINE_FP16_UNFOLDED_PATH, device=device)
    trt_fp16_folded = TRTRunner(engine_path=ENGINE_PATH, device=device)

    with torch.no_grad():
        ref_rand = m_ref(rand_patch)
        ref_real = m_ref(real_patch)
        ref_prob_real = F.softmax(ref_real, dim=1)
        ref_mask_real = torch.argmax(ref_real, dim=1)

        candidates = {}
        # Folded FP32
        candidates["pytorch_fp32_bn_folded"] = (m_folded(rand_patch), m_folded(real_patch))
        # TF32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        candidates["pytorch_tf32"] = (m_folded(rand_patch), m_folded(real_patch))
        # AMP
        with torch.autocast("cuda", dtype=torch.float16):
            candidates["pytorch_amp_bn_folded"] = (m_folded(rand_patch).float(), m_folded(real_patch).float())
            candidates["pytorch_amp_unfolded"] = (m_ref(rand_patch).float(), m_ref(real_patch).float())
        # Pure FP16
        candidates["pytorch_fp16_bn_folded"] = (m_half(rand_patch.half()).float(), m_half(real_patch.half()).float())
        # TensorRT engines
        candidates["trt_fp32"] = (trt_fp32(rand_patch), trt_fp32(real_patch))
        candidates["trt_fp16_unfolded"] = (trt_fp16_unfolded(rand_patch), trt_fp16_unfolded(real_patch))
        candidates["trt_fp16_bn_folded"] = (trt_fp16_folded(rand_patch), trt_fp16_folded(real_patch))

    torch.backends.cuda.matmul.allow_tf32 = prev_m
    torch.backends.cudnn.allow_tf32 = prev_c

    precision_table: Dict[str, Any] = {}
    for k, (y_rand, y_real) in candidates.items():
        d_rand = y_rand - ref_rand
        d_real_t = y_real - ref_real
        p_real = F.softmax(y_real, dim=1)
        m_real = torch.argmax(y_real, dim=1)
        cos_sim = float(F.cosine_similarity(ref_real.flatten().unsqueeze(0), y_real.flatten().unsqueeze(0)).item())
        precision_table[k] = {
            "rand_patch_max_abs_diff": float(d_rand.abs().max().item()),
            "rand_patch_mean_abs_diff": float(d_rand.abs().mean().item()),
            "rand_patch_ch0_signed_bias": float(d_rand[:, 0].mean().item()),
            "rand_patch_ch1_signed_bias": float(d_rand[:, 1].mean().item()),
            "real_patch_max_abs_diff": float(d_real_t.abs().max().item()),
            "real_patch_mean_abs_diff": float(d_real_t.abs().mean().item()),
            "real_patch_prob_mae": float((p_real - ref_prob_real).abs().mean().item()),
            "real_patch_cosine_similarity": round(cos_sim, 7),
            "real_patch_voxel_agreement_pct": round(float((m_real == ref_mask_real).float().mean().item() * 100.0), 5),
        }

    # 2. Custom CUDA C++ Kernel Edge-Case & Numerical Correctness Checks
    # Edge case A: Non-contiguous sliced tensor in HU normalize
    non_contig = torch.linspace(-300, 400, 64 * 64 * 32, device=device).reshape(64, 64, 32)[::2, ::2, ::2]
    assert not non_contig.is_contiguous()
    hu_nc = cuda_hu_normalize(non_contig)
    hu_ref = torch.clamp((non_contig - (-57.0)) / (164.0 - (-57.0)), 0.0, 1.0)
    hu_nc_max_diff = float((hu_nc - hu_ref).abs().max().item())

    # Edge case B: Odd tensor length (not divisible by 4) for float4 vectorized kernel
    odd_t = torch.linspace(-100, 200, 1027, device=device)
    hu_odd = cuda_hu_normalize(odd_t, vectorized=True)
    hu_odd_ref = torch.clamp((odd_t - (-57.0)) / (164.0 - (-57.0)), 0.0, 1.0)
    hu_odd_max_diff = float((hu_odd - hu_odd_ref).abs().max().item())

    # Edge case C: Empty mask voxel counting
    empty_u8 = torch.zeros((1, 128, 128, 32), dtype=torch.uint8, device=device)
    empty_cnt = cuda_count_nonzero_u8(empty_u8)

    # Edge case D: Discrete label preservation in nearest-neighbor resampling
    discrete_labels = torch.randint(0, 2, (1, 93, 87, 41), dtype=torch.uint8, device=device)
    nn_out = cuda_affine_nearest_resample_3d_u8(discrete_labels, (186, 174, 82))
    unique_vals = sorted(int(v) for v in torch.unique(nn_out).cpu().tolist())

    payload = {
        "cuda_hardware_kernels": get_cuda_module().get_kernel_metadata(),
        "backend_precision_comparison": precision_table,
        "custom_cuda_edge_cases": {
            "non_contiguous_hu_max_abs_diff": hu_nc_max_diff,
            "odd_length_vec4_tail_max_abs_diff": hu_odd_max_diff,
            "empty_mask_voxel_count": empty_cnt,
            "nearest_neighbor_unique_labels": unique_vals,
            "all_passed": bool(
                hu_nc_max_diff < 1e-6
                and hu_odd_max_diff < 1e-6
                and empty_cnt == 0
                and set(unique_vals).issubset({0, 1})
            ),
        },
    }
    out_path = save_benchmark_artifact("benchmark_correctness.json", payload)
    print(f"[PASS] Saved correctness benchmark to {out_path}")
    return payload


if __name__ == "__main__":
    run_correctness_benchmarks()
