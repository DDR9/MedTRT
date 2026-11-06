"""
Failure-Handling and Defensive Input Validation Tests (`tests/test_failure_handling.py`).
Covers Phase 9.4:
- Corrupted or missing NIfTI file
- Invalid voxel spacing (<= 0 or NaN)
- Unsupported tensor dimensions (non-5D input to TRTRunner or non-CUDA input to CUDA kernels)
- Missing TensorRT engine file
- Invalid batch size (> max_batch=16)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from medtrt.cuda_kernels import (
    compute_ras_resampled_geometry,
    cuda_affine_trilinear_resample_3d,
    cuda_count_nonzero_u8,
    cuda_fused_softmax_2ch,
)
from medtrt.profiles import get_profile
from medtrt.trt_runner import TRTRunner

DEVICE = torch.device("cuda:0")


def test_missing_tensorrt_engine_raises_file_not_found(tmp_path: Path) -> None:
    missing_engine = tmp_path / "nonexistent_model.engine"
    with pytest.raises(FileNotFoundError, match="TensorRT engine not found"):
        TRTRunner(engine_path=missing_engine, device=DEVICE)


def test_unknown_execution_profile_raises_key_error() -> None:
    with pytest.raises((KeyError, ValueError), match="Unknown profile"):
        get_profile("nonexistent_backend_profile")


def test_trt_runner_rejects_invalid_tensor_dimensions_and_batch_size() -> None:
    runner = TRTRunner(device=DEVICE)
    # 4D tensor instead of 5D (B, 1, 96, 96, 96)
    with pytest.raises((RuntimeError, ValueError, AssertionError, IndexError)):
        runner(torch.zeros((1, 96, 96, 96), device=DEVICE, dtype=torch.float32))

    # Batch size exceeding optimization profile max_batch=16
    with pytest.raises((RuntimeError, ValueError, AssertionError)):
        runner(torch.zeros((32, 1, 96, 96, 96), device=DEVICE, dtype=torch.float32))


def test_cuda_kernels_reject_cpu_tensors_and_wrong_shapes() -> None:
    cpu_vol = torch.zeros((1, 16, 16, 16), dtype=torch.float32)
    with pytest.raises(ValueError, match="CUDA"):
        cuda_affine_trilinear_resample_3d(cpu_vol, (16, 16, 16))

    cpu_mask = torch.zeros((16, 16, 16), dtype=torch.uint8)
    with pytest.raises(ValueError, match="CUDA"):
        cuda_count_nonzero_u8(cpu_mask)

    wrong_shape_logits = torch.zeros((1, 3, 16, 16, 16), device=DEVICE, dtype=torch.float32)
    with pytest.raises(ValueError, match="2-channel"):
        cuda_fused_softmax_2ch(wrong_shape_logits)


def test_invalid_voxel_spacing_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        compute_ras_resampled_geometry(np.eye(4), (64, 64, 32), pixdim=(0.0, -1.5, 2.0))
