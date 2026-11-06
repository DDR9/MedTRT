"""
End-to-End Integration, Cross-Backend Regression, and Segmentation Metric Tests
(`tests/test_integration_pipeline.py`).
Covers Phase 9.1 (metric calculation), Phase 9.2 (end-to-end integration), and Phase 9.3 (cross-backend regression).
"""

from __future__ import annotations

import json
import nibabel as nib
import numpy as np
import pytest
import torch

from medtrt.config import MEDTRT_ROOT
from medtrt.data import HELDOUT_JSON_PATH
from medtrt.evaluate import (
    compute_all_segmentation_metrics,
    compute_foreground_dice,
    compute_surface_distances_mm,
    compute_volume_cm3,
)
from medtrt.infer import infer_volume
from medtrt.profiles import build_predictor, get_profile

DEVICE = torch.device("cuda:0")


def test_segmentation_metrics_known_synthetic_masks() -> None:
    # Two identical cubes -> Dice=1.0, IoU=1.0, Sens=1.0, Spec=1.0, Prec=1.0, VS=1.0, HD95=0.0, ASSD=0.0
    gt = np.zeros((32, 32, 32), dtype=np.uint8)
    gt[8:24, 8:24, 8:24] = 1
    pred_exact = gt.copy()

    assert compute_foreground_dice(pred_exact, gt) == pytest.approx(1.0, abs=1e-6)
    cm = compute_all_segmentation_metrics(pred_exact, gt, (1.0, 1.0, 2.0))
    assert cm["dice"] == pytest.approx(1.0, abs=1e-6)
    assert cm["iou"] == pytest.approx(1.0, abs=1e-6)
    assert cm["sensitivity"] == pytest.approx(1.0, abs=1e-6)
    assert cm["specificity"] == pytest.approx(1.0, abs=1e-6)
    assert cm["precision"] == pytest.approx(1.0, abs=1e-6)
    assert cm["volumetric_similarity"] == pytest.approx(1.0, abs=1e-6)

    hd95, assd = compute_surface_distances_mm(pred_exact, gt, (1.0, 1.0, 2.0))
    assert hd95 == pytest.approx(0.0, abs=1e-6)
    assert assd == pytest.approx(0.0, abs=1e-6)

    # Half-overlapping cube: pred covers half of gt -> TP = N/2, FN = N/2, FP = 0
    # Dice = 2*(0.5) / (0.5 + 1.0) = 2/3, IoU = 0.5, Sensitivity = 0.5, Precision = 1.0
    pred_half = np.zeros_like(gt)
    pred_half[8:16, 8:24, 8:24] = 1
    assert compute_foreground_dice(pred_half, gt) == pytest.approx(2.0 / 3.0, abs=1e-5)
    cm_half = compute_all_segmentation_metrics(pred_half, gt, (1.0, 1.0, 2.0))
    assert cm_half["iou"] == pytest.approx(0.5, abs=1e-5)
    assert cm_half["sensitivity"] == pytest.approx(0.5, abs=1e-5)
    assert cm_half["precision"] == pytest.approx(1.0, abs=1e-5)


def test_cross_backend_end_to_end_equivalence_on_real_ct() -> None:
    """Verify PyTorch FP32, TensorRT FP16 (CUDA), and TensorRT FP16 (Optimized) agree on real CT."""
    with open(HELDOUT_JSON_PATH, "r", encoding="utf-8") as f:
        case = next(c for c in json.load(f)["cases"] if c["case_id"] == "spleen_6")
    img_path = MEDTRT_ROOT / case["image"]

    prof_pt = get_profile("pytorch_fp32")
    pred_pt, amp_pt, _ = build_predictor(prof_pt, DEVICE)
    res_pt = infer_volume(img_path, pred_pt, prof_pt, DEVICE, use_amp=amp_pt)

    prof_cuda = get_profile("trt_fp16_cuda")
    pred_cuda, amp_cuda, _ = build_predictor(prof_cuda, DEVICE)
    res_cuda = infer_volume(img_path, pred_cuda, prof_cuda, DEVICE, use_amp=amp_cuda)

    prof_opt = get_profile("trt_fp16_optimized")
    res_opt = infer_volume(img_path, pred_cuda, prof_opt, DEVICE, use_amp=amp_cuda)

    mask_pt = res_pt["pred_mask_3d"].astype(np.uint8)
    mask_cuda = res_cuda["pred_mask_3d"].astype(np.uint8)
    mask_opt = res_opt["pred_mask_3d"].astype(np.uint8)

    # Verify identical output mask geometry
    assert mask_pt.shape == mask_cuda.shape == mask_opt.shape

    # Cross-backend Dice between PyTorch FP32 and TensorRT FP16 + Custom CUDA must exceed 0.995
    cross_dice_cuda = compute_foreground_dice(mask_cuda, mask_pt)
    cross_dice_opt = compute_foreground_dice(mask_opt, mask_pt)
    assert cross_dice_cuda > 0.995, f"Expected cross-backend Dice > 0.995, got {cross_dice_cuda:.6f}"
    assert cross_dice_opt > 0.990, f"Expected optimized cross-backend Dice > 0.990, got {cross_dice_opt:.6f}"

    # Verify volumetric agreement within 1.5 cm^3 on spleen_6
    spacing_xyz = tuple(float(s) for s in nib.load(str(img_path)).header.get_zooms()[:3])
    _, vol_pt = compute_volume_cm3(mask_pt, spacing_xyz)
    _, vol_cuda = compute_volume_cm3(mask_cuda, spacing_xyz)
    assert abs(vol_cuda - vol_pt) < 1.5, f"Volume drift too large: {vol_cuda:.2f} vs {vol_pt:.2f} cm^3"
