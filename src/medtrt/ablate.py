"""
One-Change-at-a-Time Speedup Ablation (`src/medtrt/ablate.py`).

The end-to-end speedup combines several independent changes. This module measures them separately
so each one can be credited correctly, including how far the pipeline gets without TensorRT.
Dice uses a plain NumPy formula here, which also cross-checks the MONAI metric in `evaluate.py`.

Writes `results/ablation.json`.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Callable, Dict, List

import nibabel as nib
import numpy as np
import torch

from medtrt.bench import run_single_timed_volume_pass
from medtrt.config import MEDTRT_ROOT, RESULTS_DIR, run_metadata
from medtrt.data import HELDOUT_JSON_PATH
from medtrt.infer import build_pipeline, infer_volume
from medtrt.profiles import Profile, apply_precision
from medtrt.timing import compute_distribution_stats

_GZIP_CPU = dict(io="gzip", transforms="cpu", inverse="trilinear")
_GZIP_GPU = dict(io="gzip", transforms="gpu", inverse="trilinear")
_MMAP_GPU = dict(io="mmap", transforms="gpu", inverse="trilinear")
_BUNDLE = dict(overlap=0.5, sw_batch_size=4)
_FAST = dict(overlap=0.25, sw_batch_size=4)

LADDER: List[Profile] = [
    Profile("a_host_tf32", "Host path, PyTorch TF32", "pytorch", "tf32", **_GZIP_CPU, **_BUNDLE),
    Profile("b_host_trt", "Host path, TensorRT only", "trt", "fp16", **_GZIP_CPU, **_BUNDLE),
    Profile("c_gpu_gzip_tf32", "GPU transforms, gzip read, PyTorch TF32", "pytorch", "tf32", **_GZIP_GPU, **_BUNDLE),
    Profile("d_gpu_gzip_trt", "GPU transforms, gzip read, TensorRT", "trt", "fp16", **_GZIP_GPU, **_BUNDLE),
    Profile("e_gpu_nii_tf32", "GPU transforms, .nii read, PyTorch TF32", "pytorch", "tf32", **_MMAP_GPU, **_BUNDLE),
    Profile("f_gpu_nii_trt", "GPU transforms, .nii read, TensorRT", "trt", "fp16", **_MMAP_GPU, **_BUNDLE),
    Profile("g_fast_tf32", "Fast windows, PyTorch TF32", "pytorch", "tf32", **_MMAP_GPU, **_FAST),
    Profile("h_fast_trt", "Fast windows, TensorRT", "trt", "fp16", **_MMAP_GPU, **_FAST),
    Profile("i_fast_trt_nn", "Fast windows, TensorRT, nearest-label inverse", "trt", "fp16",
            io="mmap", transforms="gpu", inverse="nearest", **_FAST),  # fmt: skip
]


def _numpy_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred == 1, gt == 1
    return float(2.0 * np.count_nonzero(p & g) / (np.count_nonzero(p) + np.count_nonzero(g)))


def run_ablation(repeats: int = 5, warmup_volumes: int = 3) -> Dict[str, Any]:
    with open(HELDOUT_JSON_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)["cases"]

    device = torch.device("cuda:0")
    from medtrt.model import load_spleen_unet
    from medtrt.trt_runner import TRTRunner

    predictors: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "pytorch": load_spleen_unet(device),
        "trt": TRTRunner(device=device),
    }
    gt_masks = {
        c["case_id"]: np.asarray(nib.load(str(MEDTRT_ROOT / c["label"])).dataobj).astype(np.uint8) for c in cases
    }

    rungs: Dict[str, Any] = {}
    for profile in LADDER:
        apply_precision(profile)
        predictor = predictors[profile.engine]
        pipeline = build_pipeline(profile)

        def one_pass(case: Dict[str, str]) -> Dict[str, float]:
            return run_single_timed_volume_pass(MEDTRT_ROOT / case["image"], predictor, device, pipeline, profile)

        for w_idx in range(warmup_volumes):
            one_pass(cases[w_idx % len(cases)])
        passes = [one_pass(c) for c in cases for _ in range(repeats)]

        dice: Dict[str, float] = {}
        for c in cases:
            mask = infer_volume(MEDTRT_ROOT / c["image"], predictor, profile, device)["pred_mask_3d"]
            dice[c["case_id"]] = round(_numpy_dice(mask, gt_masks[c["case_id"]]), 6)

        stage_keys = ("load", "h2d", "ingest", "preprocess", "sw_network", "sw_total", "postprocess", "e2e_total", "wall_ms")
        summary = {k: compute_distribution_stats([p[k] for p in passes]) for k in stage_keys}
        rungs[profile.name] = {
            "profile": profile.as_dict(),
            "stages_ms": summary,
            "windows_total": int(sum(p["patch_windows"] for p in passes) / repeats),
            "mean_dice": round(float(np.mean(list(dice.values()))), 6),
            "dice_by_case": dice,
        }
        print(
            f"{profile.name:18} e2e={summary['e2e_total']['median']:8.1f} ms  max={summary['e2e_total']['max']:8.1f} ms  "
            f"dice={rungs[profile.name]['mean_dice']:.6f}  {profile.label}",
            flush=True,
        )

    payload = {"run": run_metadata(), "repeats_per_case": repeats, "rungs": rungs}
    with open(RESULTS_DIR / "ablation.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print("[PASS] Saved results/ablation.json")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedTRT speedup ablation")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    run_ablation(repeats=args.repeats)
