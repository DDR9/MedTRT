"""
Unit Tests for MONAI Inference Preprocessing Transforms (`tests/test_transforms.py`).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from medtrt.config import MEDTRT_ROOT
from medtrt.data import HELDOUT_JSON_PATH
from medtrt.infer import build_inference_preprocessing


def test_inference_transform_shapes_and_bounds() -> None:
    assert HELDOUT_JSON_PATH.exists(), "heldout_cases.json must exist"
    with open(HELDOUT_JSON_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)["cases"]

    first_img = MEDTRT_ROOT / cases[0]["image"]
    pre_tf = build_inference_preprocessing()
    out = pre_tf({"image": str(first_img)})
    img_t = out["image"]

    assert isinstance(img_t, torch.Tensor)
    assert img_t.ndim == 4, f"Expected 4D channel-first tensor (1, H, W, D), got {img_t.shape}"
    assert img_t.shape[0] == 1, f"Expected 1 input channel, got {img_t.shape[0]}"
    assert float(img_t.min().item()) >= 0.0 - 1e-6, f"Intensity min out of [0, 1]: {img_t.min().item()}"
    assert float(img_t.max().item()) <= 1.0 + 1e-6, f"Intensity max out of [0, 1]: {img_t.max().item()}"
