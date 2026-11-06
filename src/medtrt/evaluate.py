"""
Advanced Quantitative Medical Image Validation & Volumetry (`src/medtrt/evaluate.py`).

Implements all 10 required Phase 6 segmentation and volumetric metrics:
1. Dice similarity coefficient (`dice`)
2. Intersection over Union / Jaccard (`iou`)
3. Sensitivity / Recall (`sensitivity`)
4. Specificity (`specificity`)
5. Precision / Positive Predictive Value (`precision`)
6. Volumetric Similarity (`volumetric_similarity`)
7. Absolute Volume Error in cm^3 (`abs_volume_error_cm3`, plus signed `v_diff_cm3`)
8. Relative Volume Error in % (`rel_volume_error_pct`, plus `abs_rel_volume_error_pct`)
9. Hausdorff Distance 95th percentile in physical mm (`hd95_mm`)
10. Average Symmetric Surface Distance in physical mm (`assd_mm`)

Physical Volume Calculation (Phase 6.3):
- Supports both voxel spacing tuple `(s_x, s_y, s_z)` and full 4x4 voxel-to-world affine matrix `A`:
  `voxel_volume_mm3 = |det(A_3x3)|`
  `segmented_volume_cm3 = (foreground_voxel_count * voxel_volume_mm3) / 1000.0`
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import nibabel as nib
import numpy as np
import torch
from monai.metrics import (
    DiceMetric,
    compute_average_surface_distance,
    compute_hausdorff_distance,
)
from monai.transforms import AsDiscrete

from medtrt.config import MEDTRT_ROOT, RESULTS_DIR, run_metadata
from medtrt.data import HELDOUT_JSON_PATH, resolve_heldout_split
from medtrt.infer import infer_volume
from medtrt.profiles import PROFILES, build_predictor, get_profile
from medtrt.timing import compute_distribution_stats


def compute_voxel_volume_mm3(
    spacing_or_affine: Tuple[float, float, float] | np.ndarray | torch.Tensor,
) -> float:
    """
    Calculate physical single-voxel volume in mm^3.
    - If a 4x4 or 3x3 affine transformation matrix A is provided, uses |det(A_3x3)|.
    - If a 3-element spacing tuple (s_x, s_y, s_z) is provided, uses |s_x * s_y * s_z|.
    """
    arr = np.asarray(spacing_or_affine, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] >= 3:
        a_3x3 = arr[:3, :3]
        det_val = float(abs(np.linalg.det(a_3x3)))
        if det_val <= 0.0 or not np.isfinite(det_val):
            raise ValueError(f"Degenerate or singular spatial affine matrix: det={det_val}")
        return det_val
    if arr.ndim == 1 and arr.size == 3:
        if np.any(arr <= 0.0) or not np.all(np.isfinite(arr)):
            raise ValueError(f"Invalid voxel spacing: {arr}")
        return float(arr[0] * arr[1] * arr[2])
    raise ValueError(f"Expected 3-tuple spacing or 3x3/4x4 affine matrix, got shape {arr.shape}")


def compute_volume_cm3(
    mask_3d: np.ndarray,
    spacing_xyz: Tuple[float, float, float] | np.ndarray,
    affine: np.ndarray | None = None,
) -> Tuple[int, float]:
    """
    Calculate foreground voxel count N and physical organ volume in cm^3:
        Voxel volume = |det(A_3x3)| (when affine is provided) or (s_x * s_y * s_z)
        Segmented volume = (N * Voxel volume) / 1000.0 (cm^3)
    """
    voxels = int(np.count_nonzero(mask_3d == 1))
    voxel_vol_mm3 = compute_voxel_volume_mm3(affine if affine is not None else spacing_xyz)
    volume_cm3 = (voxels * voxel_vol_mm3) / 1000.0
    return voxels, float(volume_cm3)


def compute_foreground_dice(pred_mask_3d: np.ndarray, gt_mask_3d: np.ndarray) -> float:
    """
    Calculate foreground Spleen Dice similarity coefficient via `monai.metrics.DiceMetric(include_background=False)`.
    """
    p_any = bool(np.any(pred_mask_3d == 1))
    g_any = bool(np.any(gt_mask_3d == 1))
    if not p_any and not g_any:
        return 1.0
    if not p_any or not g_any:
        return 0.0

    to_onehot = AsDiscrete(to_onehot=2)
    pred_t = torch.from_numpy(pred_mask_3d.astype(np.int64)).unsqueeze(0)
    gt_t = torch.from_numpy(gt_mask_3d.astype(np.int64)).unsqueeze(0)

    pred_oh = to_onehot(pred_t).unsqueeze(0)  # (1, 2, X, Y, Z)
    gt_oh = to_onehot(gt_t).unsqueeze(0)      # (1, 2, X, Y, Z)

    metric = DiceMetric(include_background=False, reduction="mean")
    metric(y_pred=pred_oh, y=gt_oh)
    score = float(metric.aggregate().item())
    metric.reset()
    return score


def compute_surface_distances_mm(
    pred_mask_3d: np.ndarray,
    gt_mask_3d: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
) -> Tuple[float, float]:
    """
    Compute 95th percentile Hausdorff Distance (`hd95_mm`) and Average Symmetric Surface Distance (`assd_mm`)
    in physical millimeters (`mm`) using MONAI's validated surface distance implementations on the
    padded union bounding box of `pred | gt`.
    Empty-mask convention:
    - Both empty -> (0.0, 0.0)
    - One empty  -> (inf, inf)
    """
    p = pred_mask_3d == 1
    g = gt_mask_3d == 1
    if not np.any(p) and not np.any(g):
        return 0.0, 0.0
    if not np.any(p) or not np.any(g):
        return float("inf"), float("inf")

    union = p | g
    coords = np.argwhere(union)
    mins = np.maximum(coords.min(axis=0) - 2, 0)
    maxs = np.minimum(coords.max(axis=0) + 3, np.array(p.shape))

    p_crop = p[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
    g_crop = g[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]

    # MONAI tensor shape (B=1, C=1, X, Y, Z) with include_background=True on single foreground channel
    p_t = torch.from_numpy(p_crop[None, None].astype(np.float32))
    g_t = torch.from_numpy(g_crop[None, None].astype(np.float32))

    hd95 = float(
        compute_hausdorff_distance(
            y_pred=p_t,
            y=g_t,
            include_background=True,
            percentile=95.0,
            spacing=spacing_xyz,
        ).item()
    )
    assd = float(
        compute_average_surface_distance(
            y_pred=p_t,
            y=g_t,
            include_background=True,
            symmetric=True,
            spacing=spacing_xyz,
        ).item()
    )
    return hd95, assd


def compute_all_segmentation_metrics(
    pred_mask_3d: np.ndarray,
    gt_mask_3d: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    affine: np.ndarray | None = None,
) -> Dict[str, Any]:
    """
    Compute all 10 segmentation and volumetric metrics required by Phase 6.2 & 6.3.
    """
    if pred_mask_3d.shape != gt_mask_3d.shape:
        raise ValueError(f"Shape mismatch: pred {pred_mask_3d.shape} vs gt {gt_mask_3d.shape}")

    p = pred_mask_3d == 1
    g = gt_mask_3d == 1

    tp = int(np.count_nonzero(p & g))
    fp = int(np.count_nonzero(p & ~g))
    fn = int(np.count_nonzero(~p & g))
    tn = int(p.size - tp - fp - fn)

    if tp + fp + fn == 0:
        dice = 1.0
        iou = 1.0
        sensitivity = 1.0
        precision = 1.0
    else:
        dice = (2.0 * tp) / max(2 * tp + fp + fn, 1)
        iou = tp / max(tp + fp + fn, 1)
        sensitivity = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
        precision = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0

    specificity = tn / max(tn + fp, 1) if (tn + fp) > 0 else 1.0

    voxels_pred, v_pred_cm3 = compute_volume_cm3(pred_mask_3d, spacing_xyz, affine=affine)
    voxels_gt, v_gt_cm3 = compute_volume_cm3(gt_mask_3d, spacing_xyz, affine=affine)
    voxel_vol_mm3 = compute_voxel_volume_mm3(affine if affine is not None else spacing_xyz)

    v_diff_cm3 = v_pred_cm3 - v_gt_cm3
    abs_v_err_cm3 = abs(v_diff_cm3)
    rel_v_err_pct = (100.0 * v_diff_cm3 / v_gt_cm3) if v_gt_cm3 > 0 else 0.0
    abs_rel_v_err_pct = abs(rel_v_err_pct)
    vol_sim = (
        1.0 - (abs_v_err_cm3 / (v_pred_cm3 + v_gt_cm3))
        if (v_pred_cm3 + v_gt_cm3) > 0
        else 1.0
    )

    hd95_mm, assd_mm = compute_surface_distances_mm(pred_mask_3d, gt_mask_3d, spacing_xyz)

    return {
        "voxels_pred": voxels_pred,
        "voxels_gt": voxels_gt,
        "voxel_volume_mm3": round(voxel_vol_mm3, 6),
        "v_pred_cm3": round(v_pred_cm3, 4),
        "v_gt_cm3": round(v_gt_cm3, 4),
        "v_diff_cm3": round(v_diff_cm3, 4),
        "abs_volume_error_cm3": round(abs_v_err_cm3, 4),
        "rel_volume_error_pct": round(rel_v_err_pct, 4),
        "abs_rel_volume_error_pct": round(abs_rel_v_err_pct, 4),
        "volumetric_similarity": round(vol_sim, 6),
        "dice": round(dice, 6),
        "iou": round(iou, 6),
        "sensitivity": round(sensitivity, 6),
        "specificity": round(specificity, 6),
        "precision": round(precision, 6),
        "hd95_mm": round(hd95_mm, 4),
        "assd_mm": round(assd_mm, 4),
        "tp_voxels": tp,
        "fp_voxels": fp,
        "fn_voxels": fn,
        "tn_voxels": tn,
    }


def evaluate_backend(backend: str) -> Dict[str, Any]:
    """
    Run evaluation for profile `backend` across all held-out cases and write
    `results/eval_<backend>.csv` and `results/eval_<backend>.meta.json`.
    """
    if not HELDOUT_JSON_PATH.exists():
        heldout_info = resolve_heldout_split()
    else:
        with open(HELDOUT_JSON_PATH, "r", encoding="utf-8") as f:
            heldout_info = json.load(f)

    device = torch.device("cuda:0")
    profile = get_profile(backend)
    predictor, use_amp, _ = build_predictor(profile, device)

    cases: List[Dict[str, str]] = heldout_info["cases"]
    rows: List[Dict[str, Any]] = []

    print("=" * 108)
    print(f"EVALUATING BACKEND: {backend} ON {len(cases)} HELD-OUT CASES")
    print("=" * 108)
    print(
        f"{'Case ID':<10} | {'Shape':<14} | {'Dice':<8} | {'IoU':<8} | {'HD95(mm)':<8} | "
        f"{'ASSD(mm)':<8} | {'V_pred(cm³)':<11} | {'V_gt(cm³)':<9} | {'dV(cm³)':<8}"
    )
    print("-" * 108)

    for item in cases:
        case_id = item["case_id"]
        img_path = MEDTRT_ROOT / item["image"]
        lbl_path = MEDTRT_ROOT / item["label"]

        orig_nii = nib.load(str(img_path))
        gt_nii = nib.load(str(lbl_path))
        gt_mask_3d = np.asarray(gt_nii.dataobj).astype(np.uint8)
        zooms = orig_nii.header.get_zooms()[:3]
        spacing_xyz = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
        affine = np.asarray(orig_nii.affine, dtype=np.float64)

        inf_out = infer_volume(
            image_path=img_path,
            predictor=predictor,
            profile=profile,
            device=device,
            use_amp=use_amp,
        )
        pred_mask_3d = inf_out["pred_mask_3d"].astype(np.uint8)

        metrics = compute_all_segmentation_metrics(
            pred_mask_3d=pred_mask_3d,
            gt_mask_3d=gt_mask_3d,
            spacing_xyz=spacing_xyz,
            affine=affine,
        )

        row = {
            "case_id": case_id,
            "backend": backend,
            "shape_x": int(orig_nii.shape[0]),
            "shape_y": int(orig_nii.shape[1]),
            "shape_z": int(orig_nii.shape[2]),
            "spacing_x_mm": round(spacing_xyz[0], 6),
            "spacing_y_mm": round(spacing_xyz[1], 6),
            "spacing_z_mm": round(spacing_xyz[2], 6),
            **metrics,
        }
        rows.append(row)
        shape_str = f"{row['shape_x']}x{row['shape_y']}x{row['shape_z']}"
        print(
            f"{case_id:<10} | {shape_str:<14} | {row['dice']:<8.4f} | {row['iou']:<8.4f} | "
            f"{row['hd95_mm']:<8.2f} | {row['assd_mm']:<8.2f} | {row['v_pred_cm3']:<11.2f} | "
            f"{row['v_gt_cm3']:<9.2f} | {row['v_diff_cm3']:+8.2f}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"eval_{backend}.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metric_names = (
        "dice",
        "iou",
        "sensitivity",
        "specificity",
        "precision",
        "volumetric_similarity",
        "abs_volume_error_cm3",
        "rel_volume_error_pct",
        "abs_rel_volume_error_pct",
        "hd95_mm",
        "assd_mm",
        "v_pred_cm3",
        "v_gt_cm3",
        "v_diff_cm3",
    )
    summary_stats = {
        m: compute_distribution_stats([float(r[m]) for r in rows]) for m in metric_names
    }

    meta_payload = {
        "backend": backend,
        "run": run_metadata(),
        "profile": profile.as_dict(),
        "n_cases": len(rows),
        "summary_stats": summary_stats,
    }
    with open(RESULTS_DIR / f"eval_{backend}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)

    mean_dice = summary_stats["dice"]["mean"]
    std_dice = summary_stats["dice"]["std"]
    median_dice = summary_stats["dice"]["median"]
    bundle_ref_dice = float(heldout_info.get("bundle_reference_mean_dice", 0.961))
    diff_from_ref = abs(mean_dice - bundle_ref_dice)

    print("-" * 108)
    print(f"Mean Foreground Dice : {mean_dice:.6f} ± {std_dice:.6f} (Median: {median_dice:.6f})")
    print(f"Mean HD95 / ASSD     : {summary_stats['hd95_mm']['mean']:.2f} mm / {summary_stats['assd_mm']['mean']:.2f} mm")
    print(f"Mean Abs Vol Error   : {summary_stats['abs_volume_error_cm3']['mean']:.2f} cm³ ({summary_stats['abs_rel_volume_error_pct']['mean']:.2f}%)")
    print(f"Bundle Ref Mean Dice : {bundle_ref_dice:.6f} (|Diff| = {diff_from_ref:.6f}, Tol <= 0.02)")
    print(f"Saved Evaluation CSV : {csv_path}")
    print("=" * 108)

    return {
        "backend": backend,
        "csv_path": str(csv_path),
        "mean_dice": mean_dice,
        "std_dice": std_dice,
        "median_dice": median_dice,
        "summary_stats": summary_stats,
        "bundle_ref_dice": bundle_ref_dice,
        "within_tolerance": bool(diff_from_ref <= 0.02),
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedTRT Held-Out Quantitative Medical Validation")
    parser.add_argument("--backend", type=str, required=True, choices=sorted(PROFILES))
    args = parser.parse_args()
    evaluate_backend(backend=args.backend)
