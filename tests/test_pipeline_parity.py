"""
Pipeline Parity Tests (`tests/test_pipeline_parity.py`). Require a CUDA GPU, data, and the engine.

The GPU path is only a valid speedup if it computes the same thing as the host path.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from medtrt.config import MEDTRT_ROOT
from medtrt.data import HELDOUT_JSON_PATH
from medtrt.infer import build_nearest_label_postprocessing, build_pipeline, infer_volume
from medtrt.profiles import build_predictor, get_profile

DEVICE = torch.device("cuda:0")


def _mask(profile_name: str, image: str) -> np.ndarray:
    profile = get_profile(profile_name)
    predictor, use_amp, _ = build_predictor(profile, DEVICE)
    return infer_volume(MEDTRT_ROOT / image, predictor, profile, DEVICE, use_amp)["pred_mask_3d"]


def _smallest_case() -> str:
    with open(HELDOUT_JSON_PATH, "r", encoding="utf-8") as f:
        return next(c for c in json.load(f)["cases"] if c["case_id"] == "spleen_8")["image"]


def test_gpu_path_reproduces_host_path() -> None:
    image = _smallest_case()
    host, gpu = _mask("trt_fp16", image) == 1, _mask("trt_fp16_gpu", image) == 1
    overlap = 2.0 * np.count_nonzero(host & gpu) / (np.count_nonzero(host) + np.count_nonzero(gpu))
    assert overlap > 0.999, f"GPU path diverged from host path: Dice between them {overlap:.5f}"


def test_fast_profile_uses_the_bundle_inverse() -> None:
    profile = get_profile("trt_fp16_optimized")
    assert profile.inverse == "trilinear"
    _, pre_tf, post_tf = build_pipeline(profile)
    names = [type(t).__name__ for t in post_tf.transforms]
    assert names.index("Activationsd") < names.index("Invertd") < names.index("AsDiscreted")
    nearest = [type(t).__name__ for t in build_nearest_label_postprocessing(pre_tf).transforms]
    assert nearest.index("AsDiscreted") < nearest.index("Invertd")
