"""
Unit and Correctness Tests for Custom CUDA C++ Medical Image Kernels (`tests/test_cuda_kernels.py`).
Covers Phase 9.1 (HU windowing, resampling, custom CUDA kernels, sliding-window stitching)
and Phase 9.3 (cross-backend equivalence).
"""

from __future__ import annotations

import json
import numpy as np
import nibabel as nib
import pytest
import torch
import torch.nn.functional as F

from medtrt.config import MEDTRT_ROOT
from medtrt.cuda_kernels import (
    cuda_hu_window_normalize,
    cuda_affine_trilinear_resample_3d,
    cuda_fused_preprocess,
    cuda_fused_softmax_2ch,
    cuda_fused_invert_postprocess,
    cuda_count_nonzero_u8,
    cuda_sliding_window_accumulate,
)
from medtrt.data import HELDOUT_JSON_PATH
from medtrt.infer import build_pipeline
from medtrt.profiles import get_profile

DEVICE = torch.device("cuda:0")


def test_hu_window_normalize_scalar_and_vec4_exact() -> None:
    torch.manual_seed(42)
    x = torch.randn((1, 65, 63, 31), device=DEVICE, dtype=torch.float32) * 200.0 + 30.0
    a_min, a_max = -57.0, 164.0
    ref = torch.clamp((x - a_min) / (a_max - a_min), 0.0, 1.0)

    out_scalar = cuda_hu_window_normalize(x, a_min=a_min, a_max=a_max, vectorized=False)
    out_vec4 = cuda_hu_window_normalize(x, a_min=a_min, a_max=a_max, vectorized=True)

    assert torch.allclose(out_scalar, ref, atol=1e-6)
    assert torch.allclose(out_vec4, ref, atol=1e-6)
    assert float(out_vec4.min()) >= 0.0
    assert float(out_vec4.max()) <= 1.0


def test_affine_trilinear_identity() -> None:
    torch.manual_seed(7)
    src = torch.rand((1, 32, 32, 16), device=DEVICE, dtype=torch.float32)
    ident = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    out = cuda_affine_trilinear_resample_3d(src, (32, 32, 16), ident)
    assert torch.allclose(out, src, atol=1e-6)


def test_fused_preprocess_matches_monai_on_real_ct() -> None:
    with open(HELDOUT_JSON_PATH, "r", encoding="utf-8") as f:
        case = next(c for c in json.load(f)["cases"] if c["case_id"] == "spleen_6")
    img_path = MEDTRT_ROOT / case["image"]

    load_tf, pre_tf, _ = build_pipeline(get_profile("trt_fp16_gpu"))
    d = load_tf({"image": str(img_path)})
    d["image"] = d["image"].to(DEVICE)
    monai_out = pre_tf(d)["image"].as_subclass(torch.Tensor)

    nii = nib.load(str(img_path))
    raw_gpu = torch.as_tensor(np.asarray(nii.dataobj, dtype=np.float32), device=DEVICE)
    cuda_out, orig_shape, res_aff, m_inv = cuda_fused_preprocess(
        raw_gpu, np.asarray(nii.affine, dtype=np.float64)
    )

    assert cuda_out.shape == monai_out.shape
    max_diff = float((cuda_out - monai_out).abs().max().item())
    mean_diff = float((cuda_out - monai_out).abs().mean().item())
    assert max_diff < 1e-4, f"Expected max diff < 1e-4 vs MONAI, got {max_diff}"
    assert mean_diff < 1e-6, f"Expected mean diff < 1e-6 vs MONAI, got {mean_diff}"
    assert orig_shape == tuple(nii.shape[:3])
    assert m_inv.shape == (3, 4)


def test_fused_softmax_2ch_matches_pytorch() -> None:
    torch.manual_seed(13)
    logits = torch.randn((1, 2, 48, 48, 24), device=DEVICE, dtype=torch.float32) * 8.0
    ref = F.softmax(logits, dim=1)[0, 1].contiguous()
    out = cuda_fused_softmax_2ch(logits)
    assert out.shape == (48, 48, 24)
    assert torch.allclose(out, ref, atol=1e-6)


def test_fused_invert_postprocess_matches_geometry() -> None:
    with open(HELDOUT_JSON_PATH, "r", encoding="utf-8") as f:
        case = next(c for c in json.load(f)["cases"] if c["case_id"] == "spleen_6")
    img_path = MEDTRT_ROOT / case["image"]
    nii = nib.load(str(img_path))
    orig_aff = np.asarray(nii.affine, dtype=np.float64)
    raw_gpu = torch.as_tensor(np.asarray(nii.dataobj, dtype=np.float32), device=DEVICE)
    pre_t, orig_shape, _, m_inv = cuda_fused_preprocess(raw_gpu, orig_aff)

    _, dx, dy, dz = pre_t.shape
    zz, yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, dx, device=DEVICE),
        torch.linspace(-1, 1, dy, device=DEVICE),
        torch.linspace(-1, 1, dz, device=DEVICE),
        indexing="ij",
    )
    dist = torch.sqrt((zz - 0.1) ** 2 + (yy + 0.1) ** 2 + xx ** 2)
    fg_logit = (0.35 - dist) * 15.0
    bg_logit = -fg_logit
    logits = torch.stack([bg_logit, fg_logit], dim=0).unsqueeze(0).contiguous()

    mask_u8, vox_count = cuda_fused_invert_postprocess(
        logits,
        out_spatial_shape=orig_shape,
        voxel_matrix_3x4=m_inv,
        return_voxel_count=True,
    )
    assert mask_u8.shape == (1, *orig_shape)
    assert mask_u8.dtype == torch.uint8
    assert vox_count == int((mask_u8 > 0).sum().item())
    assert vox_count > 1000


def test_count_nonzero_u8_warp_shuffle_reduction() -> None:
    mask = torch.zeros((129, 131, 67), device=DEVICE, dtype=torch.uint8)
    mask[10:80, 20:90, 5:45] = 1
    expected = (80 - 10) * (90 - 20) * (45 - 5)
    got = cuda_count_nonzero_u8(mask)
    assert got == expected


def test_gaussian_accumulate_patch_kernel() -> None:
    out_sum = torch.zeros((2, 64, 64, 64), device=DEVICE, dtype=torch.float32)
    out_w = torch.zeros((64, 64, 64), device=DEVICE, dtype=torch.float32)
    patch = torch.ones((2, 32, 32, 32), device=DEVICE, dtype=torch.float32) * 2.5
    gauss = torch.full((32, 32, 32), 0.5, device=DEVICE, dtype=torch.float32)

    cuda_sliding_window_accumulate(patch, gauss, out_sum, out_w, (8, 8, 8))
    assert float(out_sum[:, 16, 16, 16].mean().item()) == pytest.approx(1.25, abs=1e-6)
    assert float(out_w[16, 16, 16].item()) == pytest.approx(0.5, abs=1e-6)
    assert float(out_sum[:, 0, 0, 0].sum().item()) == 0.0
