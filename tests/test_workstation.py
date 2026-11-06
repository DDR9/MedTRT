"""
Workstation Tests (`tests/test_workstation.py`). Need no GPU or data.

Precision flags are process-wide and `build_predictor` sets them only on a cache miss, so the page must
set them itself on every run. Otherwise a cached profile runs at the precision of the last one built.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import workstation  # noqa: E402


def _tf32_flags() -> Tuple[bool, bool]:
    return torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32


def _set_tf32_flags(allowed: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = allowed
    torch.backends.cudnn.allow_tf32 = allowed


@pytest.fixture(autouse=True)
def restore_precision() -> Iterator[None]:
    matmul, cudnn = _tf32_flags()
    matmul_precision = torch.get_float32_matmul_precision()
    yield
    torch.backends.cuda.matmul.allow_tf32 = matmul
    torch.backends.cudnn.allow_tf32 = cudnn
    torch.set_float32_matmul_precision(matmul_precision)


@pytest.mark.parametrize(
    ("profile_name", "left_by_last_build", "expected"),
    [("pytorch_fp32_strict", True, False), ("pytorch_tf32", False, True)],
)
def test_cached_predictor_runs_at_its_own_precision(
    monkeypatch: pytest.MonkeyPatch, profile_name: str, left_by_last_build: bool, expected: bool
) -> None:
    seen: List[Tuple[bool, bool]] = []

    def timed_pass(**_: Any) -> Dict[str, float]:
        seen.append(_tf32_flags())
        return dict.fromkeys(("sw_network", "preprocess", "postprocess", "e2e_total", "peak_vram_mb"), 0.0)

    def inference(**_: Any) -> Dict[str, np.ndarray]:
        seen.append(_tf32_flags())
        return {"pred_mask_3d": np.zeros((2, 2, 2), dtype=np.uint8)}

    monkeypatch.setattr(workstation, "get_cached_predictor", lambda *_: (object(), False, 0.0))
    monkeypatch.setattr(workstation, "build_pipeline", lambda _: None)
    monkeypatch.setattr(workstation, "run_single_timed_volume_pass", timed_pass)
    monkeypatch.setattr(workstation, "infer_volume", inference)
    _set_tf32_flags(left_by_last_build)

    workstation.execute_profile_on_volume(Path("volume.nii.gz"), profile_name, (1.0, 1.0, 1.0), np.eye(4), None)

    assert seen == [(expected, expected)] * 2


def test_fp32_vs_fp16_difference_heatmap_rendering() -> None:
    ct_2d = np.linspace(-100.0, 200.0, 64 * 64, dtype=np.float32).reshape(64, 64)
    fp32_mask = np.zeros((64, 64), dtype=np.uint8)
    fp16_mask = np.zeros((64, 64), dtype=np.uint8)
    fp32_mask[16:48, 16:48] = 1
    fp16_mask[16:50, 16:50] = 1  # 2-pixel boundary dilation

    heat = workstation.compute_fp32_vs_fp16_heatmap_2d(fp16_mask, fp32_mask)
    assert heat.shape == (64, 64)
    assert heat.dtype == np.float32
    assert float(heat.min()) >= 0.0 and float(heat.max()) == pytest.approx(1.0)
    assert float(heat[49, 32]) == pytest.approx(1.0)  # disagreeing boundary voxel peaks at 1.0

    rgb_base = workstation.render_plane_rgb(ct_2d, None, None, None, 400.0, 40.0, "Hidden", 0.5)
    rgb_heat = workstation.render_plane_rgb(
        ct_2d, fp16_mask, None, fp32_mask, 400.0, 40.0, "FP32-vs-FP16 Difference Heatmap", 0.65
    )
    assert rgb_heat.shape == (64, 64, 3)
    assert rgb_heat.dtype == np.uint8
    assert np.any(rgb_heat != rgb_base)
