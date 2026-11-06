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
