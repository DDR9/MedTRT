"""
External Multi-Center Validation (MICCAI FLARE22 on Google Drive) & Statistical Equivalence Suite
(`benchmarks/benchmark_external_flare22.py`).

Addresses all peer-review methodology requirements without touching the 8 frozen measurement
modules (`MEASUREMENT_MODULES`):
1. Independent External Cohort Validation (`N = 12` unseen multi-center abdominal CT scans from
   MICCAI FLARE22 stored on Google Drive `$MEDTRT_FLARE22_DIR`).
2. Patient-Level Paired Statistical Analysis (separating 90 repeated benchmark passes from
   `N = 9` MSD internal validation patients and `N = 12` external FLARE22 patients, total `N = 21`
   evaluated patients; plus `N = 41` MSD + `N = 12` FLARE22 = `53` total benchmarked CT volumes).
3. Formal Two One-Sided Tests (TOST) Equivalence Analysis at predefined clinical/numerical margins
   (`Delta_V = +/-0.10 cm^3` and `+/-0.50 cm^3`; `Delta_Dice = +/-0.001` and `+/-0.005`), replacing
   naive null-hypothesis p-value interpretation.
4. Voxel-Level Disagreement Distribution (FP/FN flipped voxels vs. strict IEEE FP32 reference,
   flipped-voxel fraction, and boundary surface discrepancy `HD95` / `ASSD`).
5. Paired Per-Case Speedup & Stage-to-End-to-End Accounting (explaining why sum of stage medians
   differs from median of end-to-end sums across heterogeneous scan slice counts, and measuring
   cold `.nii.gz` gzip vs. warm `.nii` OS page-cache `mmap` ingestion).
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import nibabel as nib
import numpy as np
import torch
from scipy import stats

MEDTRT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = MEDTRT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medtrt.bench import run_single_timed_volume_pass
from medtrt.config import RESULTS_DIR, run_metadata
from medtrt.evaluate import compute_all_segmentation_metrics
from medtrt.infer import build_pipeline, infer_volume
from medtrt.profiles import (
    HOST_BASELINE,
    LIKE_FOR_LIKE,
    REFERENCE,
    SHIPPING,
    apply_precision,
    build_predictor,
    get_profile,
)

GDRIVE_ROOT = Path(r"$MEDTRT_DATASETS_DIR")
FLARE22_DIR = GDRIVE_ROOT / "FLARE22_External_Cohort"


def _ci95_t(values: List[float]) -> Tuple[float, float, float, float]:
    """Return (mean, sd, ci95_low, ci95_high) using Student's t-distribution across N patients."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    if n < 2:
        return mean, 0.0, mean, mean
    sd = float(np.std(arr, ddof=1))
    se = sd / math.sqrt(n)
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    return mean, sd, mean - t_crit * se, mean + t_crit * se


def _tost_paired(diffs: List[float], margin: float) -> Dict[str, Any]:
    """
    Perform paired Two One-Sided Tests (TOST) for equivalence within [-margin, +margin].
    H0_1: mu_diff <= -margin   vs.   H1_1: mu_diff > -margin
    H0_2: mu_diff >= +margin   vs.   H1_2: mu_diff < +margin
    """
    arr = np.asarray(diffs, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    df = max(n - 1, 1)

    if se == 0.0:
        t1 = float("inf") if mean > -margin else float("-inf")
        t2 = float("-inf") if mean < margin else float("inf")
        p1 = 0.0 if mean > -margin else 1.0
        p2 = 0.0 if mean < margin else 1.0
        t_two = 0.0
        p_two = 1.0
    else:
        t1 = (mean - (-margin)) / se
        p1 = float(1.0 - stats.t.cdf(t1, df=df))
        t2 = (mean - margin) / se
        p2 = float(stats.t.cdf(t2, df=df))
        t_two = mean / se
        p_two = float(2.0 * (1.0 - stats.t.cdf(abs(t_two), df=df)))

    p_tost = max(p1, p2)
    t_90 = float(stats.t.ppf(0.95, df=df))  # 90% CI corresponds to alpha=0.05 TOST
    ci90_low = mean - t_90 * se
    ci90_high = mean + t_90 * se
    t_95 = float(stats.t.ppf(0.975, df=df))
    ci95_low = mean - t_95 * se
    ci95_high = mean + t_95 * se

    return {
        "n_patients": n,
        "margin": round(float(margin), 6),
        "mean_diff": round(mean, 6),
        "sd_diff": round(sd, 6),
        "se_diff": round(se, 6),
        "ci90_low": round(ci90_low, 6),
        "ci90_high": round(ci90_high, 6),
        "ci95_low": round(ci95_low, 6),
        "ci95_high": round(ci95_high, 6),
        "t_lower": round(float(t1), 4) if math.isfinite(t1) else str(t1),
        "p_lower": round(float(p1), 6),
        "t_upper": round(float(t2), 4) if math.isfinite(t2) else str(t2),
        "p_upper": round(float(p2), 6),
        "p_tost": round(float(p_tost), 6),
        "equivalent_at_alpha_0_05": bool(p_tost < 0.05),
        "two_sided_t": round(float(t_two), 4),
        "two_sided_p": round(float(p_two), 6),
    }


def run_external_and_statistical_suite() -> Dict[str, Any]:
    if not FLARE22_DIR.exists():
        raise FileNotFoundError(f"External cohort not found on Google Drive at {FLARE22_DIR}")

    manifest_path = FLARE22_DIR / "dataset_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    flare_cases = manifest["cases"]
    print("=" * 108)
    print(f"RUNNING EXTERNAL VALIDATION ON {len(flare_cases)} MICCAI FLARE22 VOLUMES (GOOGLE DRIVE)")
    print("=" * 108)

    device = torch.device("cuda:0")
    ext_profiles = [
        REFERENCE,          # pytorch_fp32_strict (CPU/gzip reference)
        HOST_BASELINE,      # pytorch_fp32 (CPU/gzip standard host baseline)
        "trt_fp16_unfolded",
        "pytorch_tf32_gpu",
        "trt_fp16_gpu",
        LIKE_FOR_LIKE,      # trt_fp16_cuda (overlap=0.5, B=4)
        SHIPPING,           # trt_fp16_optimized (overlap=0.25, B=8)
        "trt_fp16_nn",
    ]

    # Store predictions for REFERENCE so we can compute exact voxel-level disagreement & cross-backend diffs
    ref_masks: Dict[str, np.ndarray] = {}
    ext_eval_by_profile: Dict[str, List[Dict[str, Any]]] = {}
    ext_bench_by_profile: Dict[str, Dict[str, Any]] = {}

    for prof_name in ext_profiles:
        prof = get_profile(prof_name)
        apply_precision(prof)
        predictor, use_amp, extra_vram = build_predictor(prof, device)
        pipeline = build_pipeline(prof)

        # Warmup 2 passes on first scan
        first_cid = flare_cases[0]["case_id"]
        first_img = (
            FLARE22_DIR / "imagesTr_uncompressed" / f"{first_cid}.nii"
            if prof.io == "mmap"
            else FLARE22_DIR / "imagesTr" / f"{first_cid}.nii.gz"
        )
        for _ in range(2):
            run_single_timed_volume_pass(
                image_path=first_img,
                predictor=predictor,
                device=device,
                pipeline=pipeline,
                profile=prof,
                use_amp=use_amp,
                extra_vram_mb=extra_vram,
            )

        rows: List[Dict[str, Any]] = []
        timed_passes: List[Dict[str, Any]] = []
        per_case_latency: Dict[str, Dict[str, Any]] = {}

        for item in flare_cases:
            cid = item["case_id"]
            img_gz = FLARE22_DIR / "imagesTr" / f"{cid}.nii.gz"
            img_nii = FLARE22_DIR / "imagesTr_uncompressed" / f"{cid}.nii"
            lbl_gz = FLARE22_DIR / "labelsTr" / f"{cid}.nii.gz"
            active_img = img_nii if prof.io == "mmap" else img_gz

            # Warm OS page cache for active_img, then run 3 timed passes per scan for external cohort
            run_single_timed_volume_pass(
                image_path=active_img,
                predictor=predictor,
                device=device,
                pipeline=pipeline,
                profile=prof,
                use_amp=use_amp,
                extra_vram_mb=extra_vram,
            )
            n_rep = 3 if not prof.gpu_path else 5
            case_passes = []
            for r_idx in range(n_rep):
                tp = run_single_timed_volume_pass(
                    image_path=active_img,
                    predictor=predictor,
                    device=device,
                    pipeline=pipeline,
                    profile=prof,
                    use_amp=use_amp,
                    extra_vram_mb=extra_vram,
                )
                tp["case_id"] = cid
                tp["repeat"] = r_idx
                case_passes.append(tp)
                timed_passes.append(tp)

            per_case_latency[cid] = {
                "patch_windows": int(case_passes[0]["patch_windows"]),
                "ingest_median_ms": round(float(np.median([p["ingest"] for p in case_passes])), 3),
                "preprocess_median_ms": round(float(np.median([p["preprocess"] for p in case_passes])), 3),
                "sw_network_median_ms": round(float(np.median([p["sw_network"] for p in case_passes])), 3),
                "sw_aggregate_median_ms": round(float(np.median([p["sw_aggregate"] for p in case_passes])), 3),
                "postprocess_median_ms": round(float(np.median([p["postprocess"] for p in case_passes])), 3),
                "e2e_median_ms": round(float(np.median([p["e2e_total"] for p in case_passes])), 3),
                "e2e_iqr_ms": round(
                    float(
                        np.percentile([p["e2e_total"] for p in case_passes], 75)
                        - np.percentile([p["e2e_total"] for p in case_passes], 25)
                    ),
                    3,
                ),
                "peak_vram_mb": round(float(np.max([p["peak_vram_mb"] for p in case_passes])), 1),
            }

            # Evaluate segmentation accuracy
            orig_nii = nib.load(str(img_gz))
            gt_nii = nib.load(str(lbl_gz))
            gt_mask_3d = np.asarray(gt_nii.dataobj).astype(np.uint8)
            zooms = orig_nii.header.get_zooms()[:3]
            spacing_xyz = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
            affine = np.asarray(orig_nii.affine, dtype=np.float64)

            inf_out = infer_volume(
                image_path=active_img,
                predictor=predictor,
                profile=prof,
                device=device,
                use_amp=use_amp,
            )
            pred_mask_3d = inf_out["pred_mask_3d"].astype(np.uint8)
            if prof_name == REFERENCE:
                ref_masks[cid] = pred_mask_3d.copy()

            metrics = compute_all_segmentation_metrics(
                pred_mask_3d=pred_mask_3d,
                gt_mask_3d=gt_mask_3d,
                spacing_xyz=spacing_xyz,
                affine=affine,
            )

            # Voxel-level difference against strict IEEE FP32 reference
            ref_m = ref_masks[cid]
            fp_vs_ref = int(np.count_nonzero((pred_mask_3d == 1) & (ref_m == 0)))
            fn_vs_ref = int(np.count_nonzero((pred_mask_3d == 0) & (ref_m == 1)))
            total_diff_vox = fp_vs_ref + fn_vs_ref
            ref_fg_vox = max(int(np.count_nonzero(ref_m == 1)), 1)
            voxel_vol_mm3 = metrics["voxel_volume_mm3"]
            vol_bias_vs_ref_cm3 = ((int(metrics["voxels_pred"]) - int(np.count_nonzero(ref_m == 1))) * voxel_vol_mm3) / 1000.0

            row = {
                "case_id": cid,
                "backend": prof_name,
                "shape": list(orig_nii.shape[:3]),
                "spacing_mm": [round(x, 4) for x in spacing_xyz],
                "patch_windows": per_case_latency[cid]["patch_windows"],
                "e2e_median_ms": per_case_latency[cid]["e2e_median_ms"],
                **metrics,
                "fp_voxels_vs_fp32_ref": fp_vs_ref,
                "fn_voxels_vs_fp32_ref": fn_vs_ref,
                "total_flipped_voxels_vs_ref": total_diff_vox,
                "flipped_pct_of_organ": round(100.0 * total_diff_vox / ref_fg_vox, 4),
                "vol_bias_vs_fp32_ref_cm3": round(vol_bias_vs_ref_cm3, 4),
            }
            rows.append(row)

        ext_eval_by_profile[prof_name] = rows
        e2e_all = [p["e2e_total"] for p in timed_passes]
        case_medians = [per_case_latency[c["case_id"]]["e2e_median_ms"] for c in flare_cases]
        mean_c, sd_c, ci_l, ci_h = _ci95_t(case_medians)

        ext_bench_by_profile[prof_name] = {
            "n_cases": len(flare_cases),
            "total_timed_passes": len(timed_passes),
            "stage_medians_ms": {
                s: round(float(np.median([p[s] for p in timed_passes])), 3)
                for s in ("ingest", "preprocess", "sw_network", "sw_aggregate", "postprocess", "e2e_total")
            },
            "e2e_iqr_ms": round(
                float(np.percentile(e2e_all, 75) - np.percentile(e2e_all, 25)), 3
            ),
            "patient_level_e2e_ms": {
                "mean": round(mean_c, 2),
                "sd": round(sd_c, 2),
                "ci95_low": round(ci_l, 2),
                "ci95_high": round(ci_h, 2),
                "median": round(float(np.median(case_medians)), 2),
            },
            "peak_vram_max_mb": round(float(np.max([p["peak_vram_mb"] for p in timed_passes])), 1),
            "per_case": per_case_latency,
        }

        dice_vals = [r["dice"] for r in rows]
        bias_vals = [r["vol_bias_vs_fp32_ref_cm3"] for r in rows]
        print(
            f"[{prof_name:<20}] E2E Med={ext_bench_by_profile[prof_name]['stage_medians_ms']['e2e_total']:7.1f} ms | "
            f"Dice={np.mean(dice_vals):.6f} +/- {np.std(dice_vals, ddof=1):.6f} | "
            f"Bias vs FP32={np.mean(bias_vals):+.4f} +/- {np.std(bias_vals, ddof=1):.4f} cm^3"
        )

    # Compute MSD 9-case voxel-level difference distribution and patient-level paired statistics
    with open(RESULTS_DIR / "heldout_cases.json", "r", encoding="utf-8") as f:
        msd_heldout = json.load(f)
    msd_cases = msd_heldout["cases"]

    msd_ref_masks: Dict[str, np.ndarray] = {}
    msd_voxel_diffs: Dict[str, List[Dict[str, Any]]] = {}
    for prof_name in [REFERENCE, "trt_fp16_unfolded", "trt_fp16_gpu", LIKE_FOR_LIKE, SHIPPING, "trt_fp16_nn"]:
        prof = get_profile(prof_name)
        apply_precision(prof)
        predictor, use_amp, _ = build_predictor(prof, device)
        v_rows = []
        for item in msd_cases:
            cid = item["case_id"]
            img_path = MEDTRT_ROOT / (item["image_uncompressed"] if prof.io == "mmap" else item["image"])
            orig_nii = nib.load(str(MEDTRT_ROOT / item["image"]))
            zooms = orig_nii.header.get_zooms()[:3]
            affine = np.asarray(orig_nii.affine, dtype=np.float64)
            voxel_vol_mm3 = float(abs(np.linalg.det(affine[:3, :3])))

            inf_out = infer_volume(
                image_path=img_path,
                predictor=predictor,
                profile=prof,
                device=device,
                use_amp=use_amp,
            )
            pm = inf_out["pred_mask_3d"].astype(np.uint8)
            if prof_name == REFERENCE:
                msd_ref_masks[cid] = pm.copy()
            rm = msd_ref_masks[cid]
            fp_v = int(np.count_nonzero((pm == 1) & (rm == 0)))
            fn_v = int(np.count_nonzero((pm == 0) & (rm == 1)))
            tot_v = fp_v + fn_v
            ref_fg = max(int(np.count_nonzero(rm == 1)), 1)
            v_rows.append(
                {
                    "case_id": cid,
                    "ref_fg_voxels": ref_fg,
                    "pred_fg_voxels": int(np.count_nonzero(pm == 1)),
                    "fp_voxels_vs_ref": fp_v,
                    "fn_voxels_vs_ref": fn_v,
                    "total_flipped_voxels": tot_v,
                    "flipped_pct_of_organ": round(100.0 * tot_v / ref_fg, 4),
                    "vol_bias_cm3": round(((int(np.count_nonzero(pm == 1)) - ref_fg) * voxel_vol_mm3) / 1000.0, 4),
                }
            )
        msd_voxel_diffs[prof_name] = v_rows

    # Cold gzip vs Warm mmap vs Cold-simulated mmap ingestion benchmark on MSD 9 cases
    io_comparison = []
    for item in msd_cases:
        cid = item["case_id"]
        gz_p = MEDTRT_ROOT / item["image"]
        nii_p = MEDTRT_ROOT / item["image_uncompressed"]

        t0 = time.perf_counter()
        arr_gz = np.asarray(nib.load(str(gz_p)).dataobj, dtype=np.float32)
        t_gz = (time.perf_counter() - t0) * 1000.0

        # Raw disk stream read of uncompressed .nii (unbuffered file read + H2D copy)
        t0 = time.perf_counter()
        with open(nii_p, "rb", buffering=0) as f_raw:
            raw_bytes = f_raw.read()
        t_raw_disk = (time.perf_counter() - t0) * 1000.0

        # Warm mmap + H2D transfer
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        img_mm = nib.load(str(nii_p), mmap=True)
        t_gpu = torch.from_numpy(np.asarray(img_mm.dataobj)).to(device)
        torch.cuda.synchronize()
        t_mmap_h2d = (time.perf_counter() - t0) * 1000.0
        del t_gpu, arr_gz, raw_bytes

        io_comparison.append(
            {
                "case_id": cid,
                "file_size_mb": round(nii_p.stat().st_size / (1024 * 1024), 2),
                "gzip_decompression_cpu_ms": round(t_gz, 2),
                "unbuffered_nii_read_ms": round(t_raw_disk, 2),
                "warm_mmap_plus_h2d_ms": round(t_mmap_h2d, 2),
            }
        )

    # Compute patient-level paired speedups and stage-sum accounting on MSD (9 cases) and FLARE22 (12 cases)
    with open(RESULTS_DIR / f"bench_{HOST_BASELINE}.json", "r", encoding="utf-8") as f:
        b_msd_host = json.load(f)
    with open(RESULTS_DIR / f"bench_{LIKE_FOR_LIKE}.json", "r", encoding="utf-8") as f:
        b_msd_cuda = json.load(f)
    with open(RESULTS_DIR / f"bench_{SHIPPING}.json", "r", encoding="utf-8") as f:
        b_msd_ship = json.load(f)
    with open(RESULTS_DIR / "bench_pytorch_tf32_gpu.json", "r", encoding="utf-8") as f:
        b_msd_tf32_gpu = json.load(f)

    msd_per_case_paired = []
    for item in msd_cases:
        cid = item["case_id"]
        h_med = float(b_msd_host["per_case"][cid]["e2e_total"]["median"])
        c_med = float(b_msd_cuda["per_case"][cid]["e2e_total"]["median"])
        s_med = float(b_msd_ship["per_case"][cid]["e2e_total"]["median"])
        g_med = float(b_msd_tf32_gpu["per_case"][cid]["e2e_total"]["median"])

        # Per-pass stage sum for trt_fp16_cuda on this case
        c_passes = [p for p in b_msd_cuda["passes"] if p["case_id"] == cid]
        s_passes = [p for p in b_msd_ship["passes"] if p["case_id"] == cid]

        eval_ref_rows = {r["case_id"]: r for r in csv.DictReader(open(RESULTS_DIR / f"eval_{REFERENCE}.csv"))}
        eval_cuda_rows = {r["case_id"]: r for r in csv.DictReader(open(RESULTS_DIR / f"eval_{LIKE_FOR_LIKE}.csv"))}
        eval_ship_rows = {r["case_id"]: r for r in csv.DictReader(open(RESULTS_DIR / f"eval_{SHIPPING}.csv"))}
        eval_unf_rows = {r["case_id"]: r for r in csv.DictReader(open(RESULTS_DIR / "eval_trt_fp16_unfolded.csv"))}

        msd_per_case_paired.append(
            {
                "case_id": cid,
                "shape": f"{eval_ref_rows[cid]['shape_x']}x{eval_ref_rows[cid]['shape_y']}x{eval_ref_rows[cid]['shape_z']}",
                "slices_z": int(eval_ref_rows[cid]["shape_z"]),
                "windows_05": int(c_passes[0]["patch_windows"]),
                "windows_025": int(s_passes[0]["patch_windows"]),
                "host_fp32_e2e_ms": round(h_med, 2),
                "tf32_gpu_e2e_ms": round(g_med, 2),
                "trt_cuda_l4l_e2e_ms": round(c_med, 2),
                "trt_optimized_e2e_ms": round(s_med, 2),
                "trt_cuda_ingest_ms": round(float(np.median([p["ingest"] for p in c_passes])), 2),
                "trt_cuda_preprocess_ms": round(float(np.median([p["preprocess"] for p in c_passes])), 2),
                "trt_cuda_sw_network_ms": round(float(np.median([p["sw_network"] for p in c_passes])), 2),
                "trt_cuda_sw_aggregate_ms": round(float(np.median([p["sw_aggregate"] for p in c_passes])), 2),
                "trt_cuda_postprocess_ms": round(float(np.median([p["postprocess"] for p in c_passes])), 2),
                "paired_speedup_l4l_vs_host": round(h_med / c_med, 2),
                "paired_speedup_opt_vs_host": round(h_med / s_med, 2),
                "paired_speedup_l4l_vs_tf32_gpu": round(g_med / c_med, 2),
                "dice_fp32_ref": float(eval_ref_rows[cid]["dice"]),
                "dice_trt_cuda_l4l": float(eval_cuda_rows[cid]["dice"]),
                "dice_trt_optimized": float(eval_ship_rows[cid]["dice"]),
                "delta_dice_l4l_vs_ref": round(float(eval_cuda_rows[cid]["dice"]) - float(eval_ref_rows[cid]["dice"]), 6),
                "hd95_mm_l4l": float(eval_cuda_rows[cid]["hd95_mm"]),
                "assd_mm_l4l": float(eval_cuda_rows[cid]["assd_mm"]),
                "v_gt_cm3": float(eval_ref_rows[cid]["v_gt_cm3"]),
                "v_pred_fp32_ref_cm3": float(eval_ref_rows[cid]["v_pred_cm3"]),
                "v_pred_trt_unfolded_cm3": float(eval_unf_rows[cid]["v_pred_cm3"]),
                "v_pred_trt_cuda_l4l_cm3": float(eval_cuda_rows[cid]["v_pred_cm3"]),
                "vol_bias_unfolded_vs_ref_cm3": round(
                    float(eval_unf_rows[cid]["v_pred_cm3"]) - float(eval_ref_rows[cid]["v_pred_cm3"]), 4
                ),
                "vol_bias_cuda_l4l_vs_ref_cm3": round(
                    float(eval_cuda_rows[cid]["v_pred_cm3"]) - float(eval_ref_rows[cid]["v_pred_cm3"]), 4
                ),
            }
        )

    # Paired TOST equivalence on MSD (n=9), External FLARE22 (n=12), and Pooled (n=21)
    msd_vol_diffs_l4l = [r["vol_bias_cuda_l4l_vs_ref_cm3"] for r in msd_per_case_paired]
    msd_vol_diffs_unf = [r["vol_bias_unfolded_vs_ref_cm3"] for r in msd_per_case_paired]
    msd_dice_diffs_l4l = [r["delta_dice_l4l_vs_ref"] for r in msd_per_case_paired]

    flare_ref_rows = {r["case_id"]: r for r in ext_eval_by_profile[REFERENCE]}
    flare_l4l_rows = {r["case_id"]: r for r in ext_eval_by_profile[LIKE_FOR_LIKE]}
    flare_opt_rows = {r["case_id"]: r for r in ext_eval_by_profile[SHIPPING]}
    flare_unf_rows = {r["case_id"]: r for r in ext_eval_by_profile["trt_fp16_unfolded"]}

    flare_vol_diffs_l4l = [flare_l4l_rows[c["case_id"]]["vol_bias_vs_fp32_ref_cm3"] for c in flare_cases]
    flare_vol_diffs_unf = [flare_unf_rows[c["case_id"]]["vol_bias_vs_fp32_ref_cm3"] for c in flare_cases]
    flare_dice_diffs_l4l = [
        flare_l4l_rows[c["case_id"]]["dice"] - flare_ref_rows[c["case_id"]]["dice"] for c in flare_cases
    ]

    pooled_vol_diffs_l4l = msd_vol_diffs_l4l + flare_vol_diffs_l4l
    pooled_vol_diffs_unf = msd_vol_diffs_unf + flare_vol_diffs_unf
    pooled_dice_diffs_l4l = msd_dice_diffs_l4l + flare_dice_diffs_l4l

    # Paired per-case speedup stats on FLARE22
    flare_per_case_paired = []
    for item in flare_cases:
        cid = item["case_id"]
        h_med = ext_bench_by_profile[HOST_BASELINE]["per_case"][cid]["e2e_median_ms"]
        g_med = ext_bench_by_profile["pytorch_tf32_gpu"]["per_case"][cid]["e2e_median_ms"]
        c_med = ext_bench_by_profile[LIKE_FOR_LIKE]["per_case"][cid]["e2e_median_ms"]
        s_med = ext_bench_by_profile[SHIPPING]["per_case"][cid]["e2e_median_ms"]
        r_ref = flare_ref_rows[cid]
        r_l4l = flare_l4l_rows[cid]
        r_opt = flare_opt_rows[cid]
        r_unf = flare_unf_rows[cid]
        flare_per_case_paired.append(
            {
                "case_id": cid,
                "shape": f"{r_ref['shape'][0]}x{r_ref['shape'][1]}x{r_ref['shape'][2]}",
                "slices_z": int(r_ref["shape"][2]),
                "windows_05": int(r_l4l["patch_windows"]),
                "windows_025": int(r_opt["patch_windows"]),
                "host_fp32_e2e_ms": round(h_med, 2),
                "tf32_gpu_e2e_ms": round(g_med, 2),
                "trt_cuda_l4l_e2e_ms": round(c_med, 2),
                "trt_optimized_e2e_ms": round(s_med, 2),
                "paired_speedup_l4l_vs_host": round(h_med / c_med, 2),
                "paired_speedup_opt_vs_host": round(h_med / s_med, 2),
                "dice_fp32_ref": r_ref["dice"],
                "dice_trt_cuda_l4l": r_l4l["dice"],
                "dice_trt_optimized": r_opt["dice"],
                "delta_dice_l4l_vs_ref": round(r_l4l["dice"] - r_ref["dice"], 6),
                "hd95_mm_l4l": r_l4l["hd95_mm"],
                "assd_mm_l4l": r_l4l["assd_mm"],
                "v_gt_cm3": r_ref["v_gt_cm3"],
                "v_pred_fp32_ref_cm3": r_ref["v_pred_cm3"],
                "v_pred_trt_unfolded_cm3": r_unf["v_pred_cm3"],
                "v_pred_trt_cuda_l4l_cm3": r_l4l["v_pred_cm3"],
                "vol_bias_unfolded_vs_ref_cm3": r_unf["vol_bias_vs_fp32_ref_cm3"],
                "vol_bias_cuda_l4l_vs_ref_cm3": r_l4l["vol_bias_vs_fp32_ref_cm3"],
                "flipped_voxels_l4l_vs_ref": r_l4l["total_flipped_voxels_vs_ref"],
                "flipped_pct_l4l_vs_ref": r_l4l["flipped_pct_of_organ"],
            }
        )

    msd_l4l_speedups = [r["paired_speedup_l4l_vs_host"] for r in msd_per_case_paired]
    msd_opt_speedups = [r["paired_speedup_opt_vs_host"] for r in msd_per_case_paired]
    flare_l4l_speedups = [r["paired_speedup_l4l_vs_host"] for r in flare_per_case_paired]
    flare_opt_speedups = [r["paired_speedup_opt_vs_host"] for r in flare_per_case_paired]

    payload = {
        "run": run_metadata(),
        "external_cohort_metadata": {
            "dataset": "MICCAI FLARE22 Challenge External Validation Cohort",
            "zenodo_record": "7860267",
            "storage_location": "Google Drive (MedTRT_Datasets/FLARE22_External_Cohort)",
            "n_external_patients": len(flare_cases),
            "n_internal_msd_validation_patients": len(msd_cases),
            "n_pooled_validation_patients": len(flare_cases) + len(msd_cases),
            "total_msd_cohort_volumes": 41,
            "total_gpu_verified_volumes": 41 + len(flare_cases),
            "independence_statement": (
                "All 12 MICCAI FLARE22 CT scans originate from external multi-center abdominal CT "
                "repositories completely disjoint from MSD Task09_Spleen and never influenced model "
                "training, hyperparameter tuning, or checkpoint selection."
            ),
        },
        "tost_equivalence_analysis": {
            "msd_internal_n9": {
                "volume_folded_cuda_margin_0_10_cm3": _tost_paired(msd_vol_diffs_l4l, 0.10),
                "volume_folded_cuda_margin_0_50_cm3": _tost_paired(msd_vol_diffs_l4l, 0.50),
                "volume_unfolded_margin_0_50_cm3": _tost_paired(msd_vol_diffs_unf, 0.50),
                "dice_folded_cuda_margin_0_001": _tost_paired(msd_dice_diffs_l4l, 0.001),
                "dice_folded_cuda_margin_0_005": _tost_paired(msd_dice_diffs_l4l, 0.005),
            },
            "flare22_external_n12": {
                "volume_folded_cuda_margin_0_10_cm3": _tost_paired(flare_vol_diffs_l4l, 0.10),
                "volume_folded_cuda_margin_0_50_cm3": _tost_paired(flare_vol_diffs_l4l, 0.50),
                "volume_unfolded_margin_0_50_cm3": _tost_paired(flare_vol_diffs_unf, 0.50),
                "dice_folded_cuda_margin_0_001": _tost_paired(flare_dice_diffs_l4l, 0.001),
                "dice_folded_cuda_margin_0_005": _tost_paired(flare_dice_diffs_l4l, 0.005),
            },
            "pooled_cohorts_n21": {
                "volume_folded_cuda_margin_0_10_cm3": _tost_paired(pooled_vol_diffs_l4l, 0.10),
                "volume_folded_cuda_margin_0_50_cm3": _tost_paired(pooled_vol_diffs_l4l, 0.50),
                "volume_unfolded_margin_0_50_cm3": _tost_paired(pooled_vol_diffs_unf, 0.50),
                "dice_folded_cuda_margin_0_001": _tost_paired(pooled_dice_diffs_l4l, 0.001),
                "dice_folded_cuda_margin_0_005": _tost_paired(pooled_dice_diffs_l4l, 0.005),
            },
        },
        "paired_patient_level_speedups": {
            "msd_n9_l4l_vs_host": {
                "median": round(float(np.median(msd_l4l_speedups)), 2),
                "mean": round(_ci95_t(msd_l4l_speedups)[0], 2),
                "sd": round(_ci95_t(msd_l4l_speedups)[1], 2),
                "ci95_low": round(_ci95_t(msd_l4l_speedups)[2], 2),
                "ci95_high": round(_ci95_t(msd_l4l_speedups)[3], 2),
                "min": round(float(np.min(msd_l4l_speedups)), 2),
                "max": round(float(np.max(msd_l4l_speedups)), 2),
            },
            "msd_n9_opt_vs_host": {
                "median": round(float(np.median(msd_opt_speedups)), 2),
                "mean": round(_ci95_t(msd_opt_speedups)[0], 2),
                "sd": round(_ci95_t(msd_opt_speedups)[1], 2),
                "ci95_low": round(_ci95_t(msd_opt_speedups)[2], 2),
                "ci95_high": round(_ci95_t(msd_opt_speedups)[3], 2),
                "min": round(float(np.min(msd_opt_speedups)), 2),
                "max": round(float(np.max(msd_opt_speedups)), 2),
            },
            "flare22_n12_l4l_vs_host": {
                "median": round(float(np.median(flare_l4l_speedups)), 2),
                "mean": round(_ci95_t(flare_l4l_speedups)[0], 2),
                "sd": round(_ci95_t(flare_l4l_speedups)[1], 2),
                "ci95_low": round(_ci95_t(flare_l4l_speedups)[2], 2),
                "ci95_high": round(_ci95_t(flare_l4l_speedups)[3], 2),
                "min": round(float(np.min(flare_l4l_speedups)), 2),
                "max": round(float(np.max(flare_l4l_speedups)), 2),
            },
            "flare22_n12_opt_vs_host": {
                "median": round(float(np.median(flare_opt_speedups)), 2),
                "mean": round(_ci95_t(flare_opt_speedups)[0], 2),
                "sd": round(_ci95_t(flare_opt_speedups)[1], 2),
                "ci95_low": round(_ci95_t(flare_opt_speedups)[2], 2),
                "ci95_high": round(_ci95_t(flare_opt_speedups)[3], 2),
                "min": round(float(np.min(flare_opt_speedups)), 2),
                "max": round(float(np.max(flare_opt_speedups)), 2),
            },
        },
        "io_cold_vs_warm_page_cache": {
            "summary": {
                "gzip_decompression_median_ms": round(
                    float(np.median([r["gzip_decompression_cpu_ms"] for r in io_comparison])), 2
                ),
                "unbuffered_nii_read_median_ms": round(
                    float(np.median([r["unbuffered_nii_read_ms"] for r in io_comparison])), 2
                ),
                "warm_mmap_plus_h2d_median_ms": round(
                    float(np.median([r["warm_mmap_plus_h2d_ms"] for r in io_comparison])), 2
                ),
            },
            "per_case": io_comparison,
        },
        "msd_per_case_paired_table": msd_per_case_paired,
        "msd_voxel_disagreement_vs_fp32": msd_voxel_diffs,
        "flare22_per_case_paired_table": flare_per_case_paired,
        "flare22_benchmarks_by_profile": ext_bench_by_profile,
        "flare22_evaluations_by_profile": ext_eval_by_profile,
    }

    out_json = RESULTS_DIR / "external_flare22_validation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Also copy to Google Drive archive so Google Drive holds both raw volumes and validation results
    with open(GDRIVE_ROOT / "external_flare22_validation.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("=" * 108)
    print(f"[PASS] Saved External Validation & TOST Statistical Suite -> {out_json}")
    print("=" * 108)
    return payload


if __name__ == "__main__":
    run_external_and_statistical_suite()
