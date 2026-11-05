"""
Unit Tests for ONNX & TensorRT 10.x Engine Numerical Accuracy (`tests/test_engine_accuracy.py`).
"""

from __future__ import annotations

import json
import torch
import torch.nn.functional as F

from medtrt.config import RESULTS_DIR
from medtrt.model import load_spleen_unet
from medtrt.trt_runner import TRTRunner


def test_onnx_export_accuracy_record() -> None:
    trt_build_json = RESULTS_DIR / "trt_build.json"
    assert trt_build_json.exists(), "results/trt_build.json must exist"
    with open(trt_build_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    max_diff = float(meta["onnx"]["max_abs_diff_vs_pytorch_fp32"])
    assert max_diff < 1e-4, f"ONNX vs PyTorch max abs diff {max_diff} >= 1e-4"


def test_trt_fp16_zero_copy_and_patch_tolerance() -> None:
    device = torch.device("cuda:0")
    pt_model = load_spleen_unet(device)
    trt_runner = TRTRunner(device=device)

    # Zero-intensity air/background patch in normalized [0, 1] inference domain
    patch = torch.zeros((1, 1, 96, 96, 96), dtype=torch.float32, device=device)

    with torch.no_grad():
        pt_logits = pt_model(patch)
        trt_logits = trt_runner(patch)

        pt_prob = F.softmax(pt_logits, dim=1)
        trt_prob = F.softmax(trt_logits, dim=1)
        cos_sim = float(
            F.cosine_similarity(pt_logits.flatten().unsqueeze(0), trt_logits.flatten().unsqueeze(0)).item()
        )
        prob_mae = float(torch.mean(torch.abs(pt_prob - trt_prob)).item())
        pt_mask = torch.argmax(pt_logits, dim=1).float()
        trt_mask = torch.argmax(trt_logits, dim=1).float()
        mask_linf = float(torch.max(torch.abs(pt_mask - trt_mask)).item())

    assert trt_runner.host_copies_count == 0, "Zero host copies allowed in TRTRunner"
    assert trt_runner.forward_calls_count >= 1
    assert cos_sim > 0.9999, f"TRT FP16 vs PyTorch FP32 cosine similarity {cos_sim} <= 0.9999"
    assert prob_mae < 1e-3, f"TRT FP16 vs PyTorch FP32 probability MAE {prob_mae} >= 1e-3"
    assert mask_linf < 1e-3, f"TRT FP16 vs PyTorch FP32 discrete mask L_inf {mask_linf} >= 1e-3"
