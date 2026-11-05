"""
Dice & Voxel Volumetry Calculator (`src/medtrt/evaluate.py`).

Responsibilities:
- Evaluate a specified backend (`pytorch_fp32`, `pytorch_fp32_strict`, `pytorch_amp`, `trt_fp16`)
  strictly on the held-out validation cases in `results/heldout_cases.json`.
- Compute foreground Dice via `monai.metrics.DiceMetric(include_background=False)`.
- Compute spleen volume in original coordinate space:
  V = (N * s_x * s_y * s_z) / 1000.0 (cm^3).
- Save per-case and summary results to `results/eval_<backend>.csv`.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import nibabel as nib
import numpy as np
import torch
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

from medtrt.config import MEDTRT_ROOT, RESULTS_DIR
from medtrt.data import HELDOUT_JSON_PATH, resolve_heldout_split
from medtrt.infer import infer_volume
from medtrt.model import load_spleen_unet


def compute_volume_cm3(mask_3d: np.ndarray, spacing_xyz: Tuple[float, float, float]) -> Tuple[int, float]:
    """
    Calculate foreground voxel count N and physical spleen volume in cm^3:
        V = (N * s_x * s_y * s_z) / 1000.0
    """
    voxels = int(np.count_nonzero(mask_3d == 1))
    sx, sy, sz = float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])
    volume_cm3 = (voxels * sx * sy * sz) / 1000.0
    return voxels, float(volume_cm3)


def compute_foreground_dice(pred_mask_3d: np.ndarray, gt_mask_3d: np.ndarray) -> float:
    """
    Calculate foreground Spleen Dice similarity coefficient via `monai.metrics.DiceMetric(include_background=False)`.
    """
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


def evaluate_backend(
    backend: str = "pytorch_fp32",
    predictor: Callable[[torch.Tensor], torch.Tensor] | None = None,
    sw_batch_size: int = 4,
    overlap: float = 0.5,
) -> Dict[str, Any]:
    """
    Run held-out evaluation for `backend` and write `results/eval_<backend>.csv`.
    """
    if not HELDOUT_JSON_PATH.exists():
        heldout_info = resolve_heldout_split()
    else:
        with open(HELDOUT_JSON_PATH, "r", encoding="utf-8") as f:
            heldout_info = json.load(f)

    device = torch.device("cuda:0")
    use_amp = False
    gpu_transforms = False

    # Configure backend precision settings
    if backend == "pytorch_fp32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        if predictor is None:
            predictor = load_spleen_unet(device)
    elif backend == "pytorch_fp32_strict":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        if predictor is None:
            predictor = load_spleen_unet(device)
    elif backend == "pytorch_amp":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        use_amp = True
        if predictor is None:
            predictor = load_spleen_unet(device)
    elif backend in ("trt_fp16", "trt_fp16_optimized"):
        if backend == "trt_fp16_optimized":
            gpu_transforms = True
        if predictor is None:
            from medtrt.trt_runner import TRTRunner

            predictor = TRTRunner()
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    cases: List[Dict[str, str]] = heldout_info["cases"]
    rows: List[Dict[str, Any]] = []

    print("=" * 92)
    print(f"EVALUATING BACKEND: {backend} ON {len(cases)} HELD-OUT CASES")
    print("=" * 92)
    print(
        f"{'Case ID':<12} | {'Shape':<16} | {'Spacing (mm)':<22} | {'Dice':<8} | {'V_pred (cm³)':<13} | {'V_gt (cm³)':<11}"
    )
    print("-" * 92)

    for item in cases:
        case_id = item["case_id"]
        img_path = MEDTRT_ROOT / item["image"]
        lbl_path = MEDTRT_ROOT / item["label"]

        orig_nii = nib.load(str(img_path))
        gt_nii = nib.load(str(lbl_path))
        gt_mask_3d = np.asarray(gt_nii.dataobj).astype(np.uint8)
        zooms = orig_nii.header.get_zooms()[:3]
        spacing_xyz = (float(zooms[0]), float(zooms[1]), float(zooms[2]))

        inf_out = infer_volume(
            image_path=img_path,
            predictor=predictor,
            device=device,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            use_amp=use_amp,
            gpu_transforms=gpu_transforms,
        )
        pred_mask_3d = inf_out["pred_mask_3d"].astype(np.uint8)

        dice = compute_foreground_dice(pred_mask_3d, gt_mask_3d)
        voxels_pred, v_pred_cm3 = compute_volume_cm3(pred_mask_3d, spacing_xyz)
        voxels_gt, v_gt_cm3 = compute_volume_cm3(gt_mask_3d, spacing_xyz)

        row = {
            "case_id": case_id,
            "backend": backend,
            "shape_x": int(orig_nii.shape[0]),
            "shape_y": int(orig_nii.shape[1]),
            "shape_z": int(orig_nii.shape[2]),
            "spacing_x_mm": round(spacing_xyz[0], 6),
            "spacing_y_mm": round(spacing_xyz[1], 6),
            "spacing_z_mm": round(spacing_xyz[2], 6),
            "voxels_pred": voxels_pred,
            "voxels_gt": voxels_gt,
            "v_pred_cm3": round(v_pred_cm3, 4),
            "v_gt_cm3": round(v_gt_cm3, 4),
            "v_diff_cm3": round(v_pred_cm3 - v_gt_cm3, 4),
            "dice": round(dice, 6),
        }
        rows.append(row)
        shape_str = f"{row['shape_x']}x{row['shape_y']}x{row['shape_z']}"
        sp_str = f"{spacing_xyz[0]:.2f}x{spacing_xyz[1]:.2f}x{spacing_xyz[2]:.2f}"
        print(
            f"{case_id:<12} | {shape_str:<16} | {sp_str:<22} | {dice:<8.4f} | {v_pred_cm3:<13.2f} | {v_gt_cm3:<11.2f}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"eval_{backend}.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    dices = [r["dice"] for r in rows]
    mean_dice = float(np.mean(dices))
    std_dice = float(np.std(dices))
    median_dice = float(np.median(dices))
    bundle_ref_dice = float(heldout_info.get("bundle_reference_mean_dice", 0.961))
    diff_from_ref = abs(mean_dice - bundle_ref_dice)

    print("-" * 92)
    print(f"Mean Foreground Dice : {mean_dice:.6f} ± {std_dice:.6f} (Median: {median_dice:.6f})")
    print(f"Bundle Ref Mean Dice : {bundle_ref_dice:.6f} (|Diff| = {diff_from_ref:.6f}, Tol <= 0.02)")
    print(f"Saved Evaluation CSV : {csv_path}")
    print("=" * 92)

    return {
        "backend": backend,
        "csv_path": str(csv_path),
        "mean_dice": mean_dice,
        "std_dice": std_dice,
        "median_dice": median_dice,
        "bundle_ref_dice": bundle_ref_dice,
        "within_tolerance": bool(diff_from_ref <= 0.02),
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedTRT Held-Out Evaluation")
    parser.add_argument("--backend", type=str, default="pytorch_fp32")
    args = parser.parse_args()
    res = evaluate_backend(backend=args.backend)
    if args.backend == "pytorch_fp32":
        state_path = RESULTS_DIR / "STATE.json"
        state = {}
        if state_path.exists():
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
        state.update(
            {
                "current_phase": "PHASE_1",
                "phase_1_status": "PASS" if res["within_tolerance"] else "FAIL_DICE_TOLERANCE",
                "phase_1_mean_dice": round(res["mean_dice"], 6),
                "phase_1_bundle_ref_dice": res["bundle_ref_dice"],
                "heldout_cases_json": "results/heldout_cases.json",
                "eval_pytorch_fp32_csv": "results/eval_pytorch_fp32.csv",
                "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        )
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
