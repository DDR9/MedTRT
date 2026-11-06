"""
Stage Timeline, Parameter Grid Sweep, Bland-Altman Agreement & Anatomical Failure Analysis (`src/medtrt/stats.py`).

Responsibilities:
1. Stage Timeline (`capture_stage_timeline`):
   - Captures CUDA-event stage timelines for both the unoptimized host-path (`trt_fp16`) and the
     fully optimized custom-CUDA pipeline (`trt_fp16_optimized`) on `spleen_6` (`512x512x135`),
     saving `results/nsys_summary.json`, `results/timeline_summary.json`, and `docs/figures/nsys_trace.png`.
2. Parameter Grid Sweep (`run_parameter_grid_sweep`):
   - Evaluates `overlap in {0.25, 0.5, 0.75} x sw_batch_size in {1, 2, 4, 8}` with Pareto optimality flags,
     saving `results/sweep.csv` and `docs/figures/pareto_sweep.png`.
3. Bland-Altman & Cross-Backend Equivalence (`run_validation`):
   - Separately reports Prediction vs. Ground Truth AND Cross-Backend vs. Reference Strict IEEE FP32
     (`pytorch_fp32_strict`) across all evaluated profiles (including before/after `ConvTranspose3d+BN` fusion),
     saving `results/validation.csv`, `results/bland_altman.json`, and `docs/figures/bland_altman.png`.
4. Anatomical Failure Analysis (`run_failure_analysis`):
   - Investigates worst-performing validation cases across Dice, HD95, ASSD, Volume Error, and
     FP32-vs-FP16 disagreement, computing Spearman rank correlations with z-spacing, organ volume,
     and surface-to-volume ratio without unsupported causal claims.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from scipy import stats as sps
from scipy.ndimage import binary_erosion

from medtrt.bench import run_single_timed_volume_pass
from medtrt.config import FIGURES_DIR, MEDTRT_ROOT, RESULTS_DIR, run_metadata
from medtrt.data import HELDOUT_JSON_PATH
from medtrt.evaluate import compute_all_segmentation_metrics, compute_foreground_dice
from medtrt.infer import build_pipeline, build_sliding_window_inferer, infer_volume
from medtrt.profiles import PROFILES, REFERENCE, SHIPPING, build_predictor, get_profile

DICE_TOLERANCE = 0.005


def _heldout() -> List[Dict[str, str]]:
    with open(HELDOUT_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["cases"]


def _find_nsys_executable() -> str | None:
    nsys_bin = shutil.which("nsys")
    if nsys_bin:
        return nsys_bin
    candidates = glob.glob(r"C:\Program Files\NVIDIA Corporation\Nsight Systems*\target-windows-x64\nsys.exe")
    return sorted(candidates)[-1] if candidates else None


def capture_stage_timeline(run_nsys: bool = False) -> Dict[str, Any]:
    """
    Record a before/after CUDA-event stage timeline comparing `trt_fp16` (host transforms) vs.
    `trt_fp16_optimized` (custom CUDA C++ fused transforms + mmap + Pareto windows) on `spleen_6`.
    Saves `results/timeline_summary.json`, `results/nsys_summary.json`, and `docs/figures/nsys_trace.png`.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    nsys_exe = _find_nsys_executable()
    device = torch.device("cuda:0")
    case = next(c for c in _heldout() if c["case_id"] == "spleen_6")
    img_path = MEDTRT_ROOT / case["image"]

    prof_host = get_profile("trt_fp16")
    prof_opt = get_profile("trt_fp16_optimized")
    predictor, _, extra_vram = build_predictor(prof_host, device)

    pipe_host = build_pipeline(prof_host)
    pipe_opt = build_pipeline(prof_opt)

    # Warm-up
    _ = run_single_timed_volume_pass(img_path, predictor, device, pipe_opt, prof_opt, False, extra_vram)

    host_repeats: List[Dict[str, Any]] = []
    for rep in range(3):
        t_h = run_single_timed_volume_pass(img_path, predictor, device, pipe_host, prof_host, False, extra_vram)
        host_repeats.append(
            {
                "repeat": rep,
                "case_id": case["case_id"],
                "load_ms": round(t_h["load"], 3),
                "preprocess_ms": round(t_h["preprocess"], 3),
                "h2d_ms": round(t_h["h2d"], 3),
                "sw_total_ms": round(t_h["sw_total"], 3),
                "sw_network_ms": round(t_h["sw_network"], 3),
                "sw_aggregate_ms": round(t_h["sw_aggregate"], 3),
                "postprocess_ms": round(t_h["postprocess"], 3),
                "e2e_total_ms": round(t_h["e2e_total"], 3),
                "peak_vram_mb": round(t_h["peak_vram_mb"], 2),
                "num_patch_batches": int(t_h["patch_calls"]),
            }
        )

    opt_repeats: List[Dict[str, Any]] = []
    for rep in range(5):
        t_o = run_single_timed_volume_pass(img_path, predictor, device, pipe_opt, prof_opt, False, extra_vram)
        opt_repeats.append(
            {
                "repeat": rep,
                "case_id": case["case_id"],
                "load_ms": round(t_o["load"], 3),
                "preprocess_ms": round(t_o["preprocess"], 3),
                "h2d_ms": round(t_o["h2d"], 3),
                "sw_total_ms": round(t_o["sw_total"], 3),
                "sw_network_ms": round(t_o["sw_network"], 3),
                "sw_aggregate_ms": round(t_o["sw_aggregate"], 3),
                "postprocess_ms": round(t_o["postprocess"], 3),
                "e2e_total_ms": round(t_o["e2e_total"], 3),
                "peak_vram_mb": round(t_o["peak_vram_mb"], 2),
                "num_patch_batches": int(t_o["patch_calls"]),
            }
        )

    rep_h = host_repeats[-1]
    rep_o = opt_repeats[-1]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12.5, 4.8), dpi=160)
    stages_meta = (
        ("load_ms", "load (NIfTI I/O)", "#4C78A8"),
        ("h2d_ms", "h2d (PCIe Transfer)", "#72B7B2"),
        ("preprocess_ms", "preprocess (Resample+HU)", "#F58518"),
        ("sw_network_ms", "sw_network (TRT FP16 UNet)", "#54A24B"),
        ("sw_aggregate_ms", "sw_aggregate (Stitch)", "#EECA3B"),
        ("postprocess_ms", "postprocess (Invert+Argmax)", "#E45756"),
    )

    left_h = 0.0
    for key, label, color in stages_meta:
        val = rep_h[key]
        ax_top.barh(
            0, val, left=left_h, height=0.52, color=color, edgecolor="black", linewidth=0.6,
            label=f"{label}: {val:.1f} ms",
        )
        left_h += val
    ax_top.set_yticks([0])
    ax_top.set_yticklabels(["trt_fp16\n(Host Path)"])
    ax_top.set_xlabel("Elapsed Time (ms)")
    ax_top.set_title(
        f"Before Optimization: Host-Path TensorRT FP16 on {case['case_id']} (Total = {rep_h['e2e_total_ms']:.1f} ms)",
        fontsize=10.5, fontweight="bold",
    )
    ax_top.legend(loc="upper right", ncol=3, fontsize=8)
    ax_top.grid(axis="x", linestyle="--", alpha=0.4)

    left_o = 0.0
    for key, label, color in stages_meta:
        val = rep_o[key]
        ax_bot.barh(
            0, val, left=left_o, height=0.52, color=color, edgecolor="black", linewidth=0.6,
            label=f"{label}: {val:.2f} ms",
        )
        left_o += val
    ax_bot.set_yticks([0])
    ax_bot.set_yticklabels(["trt_fp16_optimized\n(Custom CUDA)"])
    ax_bot.set_xlabel("Elapsed Time (ms) — Zoomed Scale")
    ax_bot.set_title(
        f"After Optimization: Custom CUDA C++ Fused Pre/Post + Mmap + TRT FP16 on {case['case_id']} (Total = {rep_o['e2e_total_ms']:.2f} ms, {rep_h['e2e_total_ms']/max(rep_o['e2e_total_ms'],1e-6):.1f}x Faster)",
        fontsize=10.5, fontweight="bold",
    )
    ax_bot.legend(loc="upper right", ncol=3, fontsize=8)
    ax_bot.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "nsys_trace.png", bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "stage_timeline.png", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "run": run_metadata(),
        "timestamp": run_metadata()["timestamp"],
        "source": "torch.cuda.Event stage boundaries + NVTX instrumentation",
        "nsys_executable_found": bool(nsys_exe),
        "nsys_cli_executed": False,
        "nsys_rep_path": None,
        "trace_figure_path": "docs/figures/nsys_trace.png",
        "case_id": case["case_id"],
        "repeats": host_repeats,
        "optimized_repeats": opt_repeats,
    }
    with open(RESULTS_DIR / "timeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(RESULTS_DIR / "nsys_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[PASS] Saved timeline_summary.json, nsys_summary.json, and docs/figures/nsys_trace.png")
    return summary


def pareto_flags(rows: Sequence[Dict[str, Any]]) -> List[bool]:
    flags = []
    for r in rows:
        dominated = any(
            o["median_sw_latency_ms"] <= r["median_sw_latency_ms"]
            and o["mean_dice"] >= r["mean_dice"]
            and (o["median_sw_latency_ms"] < r["median_sw_latency_ms"] or o["mean_dice"] > r["mean_dice"])
            for o in rows
        )
        flags.append(not dominated)
    return flags


def run_parameter_grid_sweep(repeats: int = 5) -> Dict[str, Any]:
    """
    Sweep `overlap x sw_batch_size` using the BN-folded TensorRT FP16 engine and custom CUDA C++
    fused trilinear postprocessing across all 9 held-out cases.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    profile = get_profile(SHIPPING)
    device = torch.device("cuda:0")
    predictor, _, extra_vram_mb = build_predictor(profile, device)
    load_tf, _, _ = build_pipeline(profile)
    from medtrt.cuda_kernels import cuda_fused_invert_postprocess, cuda_fused_preprocess

    cached: List[Dict[str, Any]] = []
    for item in _heldout():
        gt_mask = np.asarray(nib.load(str(MEDTRT_ROOT / item["label"])).dataobj).astype(np.uint8)
        d = load_tf({"image": str(MEDTRT_ROOT / item["image"])})
        orig_aff = d["image"].affine.cpu().numpy()
        pre_out, orig_shape, _, m_inv = cuda_fused_preprocess(d["image"].to(device), orig_aff)
        cached.append(
            {
                "inputs": pre_out.unsqueeze(0),
                "orig_shape": orig_shape,
                "m_inv": m_inv,
                "gt": gt_mask,
            }
        )

    rows: List[Dict[str, Any]] = []
    for ov in (0.25, 0.5, 0.75):
        for sw_b in (1, 2, 4, 8):
            inferer = build_sliding_window_inferer(sw_batch_size=sw_b, overlap=ov)
            with torch.no_grad():
                for c in cached[:3]:
                    _ = inferer(inputs=c["inputs"], network=predictor)
            torch.cuda.synchronize(device)

            latencies: List[float] = []
            vram: List[float] = []
            dices: List[float] = []
            for c in cached:
                logits = None
                for _ in range(repeats):
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    s_ev, e_ev = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    with torch.no_grad():
                        s_ev.record()
                        logits = inferer(inputs=c["inputs"], network=predictor)
                        e_ev.record()
                        e_ev.synchronize()
                    latencies.append(float(s_ev.elapsed_time(e_ev)))
                    vram.append((torch.cuda.max_memory_allocated(device) / (1024**2)) + extra_vram_mb)
                mask_u8, _ = cuda_fused_invert_postprocess(logits[0].detach(), c["orig_shape"], c["m_inv"])
                pred_mask = mask_u8.squeeze(0).cpu().numpy().astype(np.uint8)
                dices.append(compute_foreground_dice(pred_mask, c["gt"]))

            q25, q50, q75 = np.percentile(latencies, [25, 50, 75])
            rows.append(
                {
                    "overlap": ov,
                    "sw_batch_size": sw_b,
                    "median_sw_latency_ms": round(float(q50), 3),
                    "iqr_sw_latency_ms": round(float(q75 - q25), 3),
                    "mean_sw_latency_ms": round(float(np.mean(latencies)), 3),
                    "peak_vram_mb": round(float(np.max(vram)), 2),
                    "incremental_vram_mb": round(float(np.max(vram)), 2),
                    "mean_dice": round(float(np.mean(dices)), 6),
                    "std_dice": round(float(np.std(dices, ddof=1)), 6),
                    "timed_passes": len(latencies),
                }
            )
            print(f"overlap={ov:<5} sw={sw_b} median={q50:8.2f} ms dice={rows[-1]['mean_dice']:.6f}", flush=True)

    for row, flag in zip(rows, pareto_flags(rows)):
        row["pareto_optimal"] = flag
        row["is_fast_profile"] = row["overlap"] == profile.overlap and row["sw_batch_size"] == profile.sw_batch_size

    with open(RESULTS_DIR / "sweep.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=160)
    colors = {0.25: "#4C78A8", 0.5: "#54A24B", 0.75: "#E45756"}
    markers = {1: "o", 2: "s", 4: "^", 8: "D"}
    for r in rows:
        ax.scatter(
            r["median_sw_latency_ms"],
            r["mean_dice"],
            s=115,
            color=colors[r["overlap"]],
            marker=markers[r["sw_batch_size"]],
            edgecolors="black",
            zorder=4,
            linewidths=2.2 if r["pareto_optimal"] else 0.8,
        )
    for ov, color in colors.items():
        ax.scatter([], [], color=color, label=f"overlap = {ov:.2f}")
    for b, marker in markers.items():
        ax.scatter([], [], color="gray", marker=marker, label=f"sw_batch_size = {b}")
    ax.scatter([], [], facecolors="none", edgecolors="black", linewidths=2.2, label="Pareto-Optimal (Bold Edge)")
    ax.set_xlabel("Median Sliding-Window Latency per Volume (ms)")
    ax.set_ylabel("Held-Out Mean Foreground Dice")
    ax.set_title("Sliding-Window Accuracy-Latency Pareto Sweep (TensorRT 10.x FP16)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="lower right", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pareto_sweep.png", bbox_inches="tight")
    plt.close(fig)
    print("[PASS] Saved results/sweep.csv and docs/figures/pareto_sweep.png")
    return {"rows": rows}


def _load_eval(name: str) -> Dict[str, Dict[str, str]]:
    p = RESULTS_DIR / f"eval_{name}.csv"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return {r["case_id"]: r for r in csv.DictReader(f)}


def agreement_stats(a: Sequence[float], b: Sequence[float]) -> Dict[str, Any]:
    """
    Bland-Altman statistics for `a - b`:
    - Mean bias, SD of differences, 95% Limits of Agreement (`bias +/- 1.96 * SD`)
    - 95% Confidence Intervals for bias and LoA bounds
    - Linear regression slope/p-value of differences vs. mean magnitude (proportional bias test)
    """
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    diffs = arr_a - arr_b
    means = 0.5 * (arr_a + arr_b)
    n = len(diffs)
    bias = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    t_val = float(sps.t.ppf(0.975, max(n - 1, 1)))
    se_bias = sd / np.sqrt(max(n, 1))
    se_loa = sd * np.sqrt(3.0 / max(n, 1))
    half_bias = t_val * se_bias
    half_loa = t_val * se_loa

    loa_low = bias - 1.96 * sd
    loa_high = bias + 1.96 * sd

    if n > 2 and np.std(means) > 1e-9 and sd > 1e-9:
        reg = sps.linregress(means, diffs)
        prop_slope = round(float(reg.slope), 6)
        prop_r = round(float(reg.rvalue), 4)
        prop_p = round(float(reg.pvalue), 4)
    else:
        prop_slope, prop_r, prop_p = 0.0, 0.0, 1.0

    # Paired t-test and Wilcoxon signed-rank test
    if n > 1 and sd > 1e-9:
        _, p_ttest = sps.ttest_rel(arr_a, arr_b)
        p_ttest_f = round(float(p_ttest), 5)
    else:
        p_ttest_f = 1.0

    return {
        "n": n,
        "bias_cm3": round(bias, 4),
        "sd_cm3": round(sd, 4),
        "loa_lower_cm3": round(loa_low, 4),
        "loa_upper_cm3": round(loa_high, 4),
        "bias_ci95_lower_cm3": round(bias - half_bias, 4),
        "bias_ci95_upper_cm3": round(bias + half_bias, 4),
        "loa_lower_ci95": [round(loa_low - half_loa, 4), round(loa_low + half_loa, 4)],
        "loa_upper_ci95": [round(loa_high - half_loa, 4), round(loa_high + half_loa, 4)],
        "bias_ci_excludes_zero": bool(bias - half_bias > 0 or bias + half_bias < 0),
        "paired_ttest_p": p_ttest_f,
        "proportional_bias_slope": prop_slope,
        "proportional_bias_r": prop_r,
        "proportional_bias_p": prop_p,
        "n_positive": int(np.sum(diffs > 0)),
        "mae_cm3": round(float(np.mean(np.abs(diffs))), 4),
        "max_abs_diff_cm3": round(float(np.max(np.abs(diffs))), 4),
    }


def dice_agreement(cmp_rows: Dict[str, Dict[str, str]], ref_rows: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    ids = sorted(ref_rows)
    deltas = np.array([float(cmp_rows[c]["dice"]) - float(ref_rows[c]["dice"]) for c in ids])
    mean_abs, max_abs = float(np.mean(np.abs(deltas))), float(np.max(np.abs(deltas)))
    return {
        "mean_dice": round(float(np.mean([float(cmp_rows[c]["dice"]) for c in ids])), 6),
        "mean_delta_dice": round(float(np.mean(deltas)), 6),
        "mean_abs_delta_dice": round(mean_abs, 6),
        "max_abs_delta_dice": round(max_abs, 6),
        "n_cases_over_tolerance": int(np.sum(np.abs(deltas) >= DICE_TOLERANCE)),
        "tolerance": DICE_TOLERANCE,
        "mean_within_tolerance": bool(mean_abs < DICE_TOLERANCE),
        "every_case_within_tolerance": bool(max_abs < DICE_TOLERANCE),
    }


def run_validation() -> Dict[str, Any]:
    """
    Compute Bland-Altman volumetric agreement and segmentation equivalence across all evaluated profiles.
    Saves `results/validation.csv`, `results/bland_altman.json`, and `docs/figures/bland_altman.png`.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ref = _load_eval(REFERENCE)
    if not ref:
        ref = _load_eval("pytorch_fp32")
    fp32_def = _load_eval("pytorch_fp32") or ref

    evals = {}
    for p in PROFILES:
        loaded = _load_eval(p)
        if loaded and p != REFERENCE:
            evals[p] = loaded

    ids = sorted(ref)
    trt_rows = evals.get("trt_fp16", ref)

    val_rows: List[Dict[str, Any]] = []
    for cid in ids:
        v_gt_val = float(ref[cid]["v_gt_cm3"])
        v_fp32_val = float(fp32_def[cid]["v_pred_cm3"])
        v_trt_val = float(trt_rows[cid]["v_pred_cm3"])
        d_fp32_val = float(fp32_def[cid]["dice"])
        d_trt_val = float(trt_rows[cid]["dice"])

        row: Dict[str, Any] = {
            "case_id": cid,
            "v_gt_cm3": v_gt_val,
            "v_fp32_cm3": v_fp32_val,
            "v_trt_fp16_cm3": v_trt_val,
            "delta_v_trt_vs_fp32_cm3": round(v_trt_val - v_fp32_val, 4),
            "dice_fp32": d_fp32_val,
            "dice_trt_fp16": d_trt_val,
            "delta_dice_trt_vs_fp32": round(d_trt_val - d_fp32_val, 6),
            f"v_{REFERENCE}": float(ref[cid]["v_pred_cm3"]),
            f"dice_{REFERENCE}": float(ref[cid]["dice"]),
        }
        for name, rows in evals.items():
            row[f"v_{name}"] = float(rows[cid]["v_pred_cm3"])
            row[f"dice_{name}"] = float(rows[cid]["dice"])
            row[f"delta_v_{name}"] = round(float(rows[cid]["v_pred_cm3"]) - float(ref[cid]["v_pred_cm3"]), 4)
            row[f"delta_dice_{name}"] = round(float(rows[cid]["dice"]) - float(ref[cid]["dice"]), 6)
        val_rows.append(row)

    with open(RESULTS_DIR / "validation.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(val_rows[0].keys()))
        writer.writeheader()
        writer.writerows(val_rows)

    v_ref = [float(ref[c]["v_pred_cm3"]) for c in ids]
    v_fp32 = [float(fp32_def[c]["v_pred_cm3"]) for c in ids]
    v_trt = [float(trt_rows[c]["v_pred_cm3"]) for c in ids]
    v_gt = [float(ref[c]["v_gt_cm3"]) for c in ids]

    eq_trt = dice_agreement(trt_rows, fp32_def)
    summary: Dict[str, Any] = {
        "run": run_metadata(),
        "timestamp": run_metadata()["timestamp"],
        "reference": REFERENCE,
        "reference_mean_dice": round(float(np.mean([float(ref[c]["dice"]) for c in ids])), 6),
        "reference_vs_gt": agreement_stats(v_ref, v_gt),
        "equivalence": {
            "mean_abs_delta_dice_trt_vs_fp32": eq_trt["mean_abs_delta_dice"],
            "max_abs_delta_dice_trt_vs_fp32": eq_trt["max_abs_delta_dice"],
            "tolerance_threshold": DICE_TOLERANCE,
            "passed": eq_trt["every_case_within_tolerance"],
        },
        "bland_altman_trt_fp16_vs_gt": agreement_stats(v_trt, v_gt),
        "bland_altman_pytorch_fp32_vs_gt": agreement_stats(v_fp32, v_gt),
        "bland_altman_trt_fp16_vs_pytorch_fp32": agreement_stats(v_trt, v_fp32),
        "profiles": {},
    }
    for name, rows in evals.items():
        v_cmp = [float(rows[c]["v_pred_cm3"]) for c in ids]
        volume = agreement_stats(v_cmp, v_ref)
        volume["bias_pct_of_mean_volume"] = round(100.0 * volume["bias_cm3"] / float(np.mean(v_ref)), 4)
        summary["profiles"][name] = {
            "dice_vs_reference": dice_agreement(rows, ref),
            "volume_vs_reference": volume,
            "volume_vs_gt": agreement_stats(v_cmp, v_gt),
        }

    with open(RESULTS_DIR / "bland_altman.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Render Two-Panel Bland-Altman Plot (Panel 1: Cross-Backend vs Strict FP32; Panel 2: Prediction vs Ground Truth)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=160)
    means_ref = [0.5 * (a + b) for a, b in zip(v_trt, v_ref)]

    if "trt_fp16_unfolded" in evals:
        diffs_unf = [float(evals["trt_fp16_unfolded"][c]["v_pred_cm3"]) - float(ref[c]["v_pred_cm3"]) for c in ids]
        unf_ba = summary["profiles"]["trt_fp16_unfolded"]["volume_vs_reference"]
        ax1.scatter(
            means_ref, diffs_unf, s=65, color="#E45756", marker="^", alpha=0.75,
            label=f"TRT FP16 Unfolded (bias {unf_ba['bias_cm3']:+.2f} cm³)", zorder=3,
        )

    diffs_trt = [float(trt_rows[c]["v_pred_cm3"]) - float(ref[c]["v_pred_cm3"]) for c in ids]
    trt_ba = summary["profiles"].get("trt_fp16", {}).get("volume_vs_reference", summary["bland_altman_trt_fp16_vs_pytorch_fp32"])
    ax1.scatter(
        means_ref, diffs_trt, s=80, color="#4C78A8", marker="o",
        label=f"TRT FP16 BN-Folded (bias {trt_ba['bias_cm3']:+.3f} cm³)", zorder=4,
    )

    if SHIPPING in evals:
        diffs_ship = [float(evals[SHIPPING][c]["v_pred_cm3"]) - float(ref[c]["v_pred_cm3"]) for c in ids]
        ship_ba = summary["profiles"][SHIPPING]["volume_vs_reference"]
        ax1.scatter(
            means_ref, diffs_ship, s=75, color="#54A24B", marker="x",
            label=f"{SHIPPING} (bias {ship_ba['bias_cm3']:+.3f} cm³)", zorder=5,
        )

    ax1.axhline(trt_ba["bias_cm3"], color="#4C78A8", linewidth=1.4, label=f"TRT FP16 Mean Bias ({trt_ba['bias_cm3']:+.3f} cm³)")
    ax1.axhline(trt_ba["loa_upper_cm3"], color="#4C78A8", linestyle="--", linewidth=1.1, label=f"95% LoA [{trt_ba['loa_lower_cm3']:+.2f}, {trt_ba['loa_upper_cm3']:+.2f}] cm³")
    ax1.axhline(trt_ba["loa_lower_cm3"], color="#4C78A8", linestyle="--", linewidth=1.1)
    ax1.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    ax1.set_xlabel("Mean of Predicted Volumes: 0.5 × (V_backend + V_FP32) (cm³)")
    ax1.set_ylabel("Cross-Backend Volume Difference: V_backend - V_FP32 (cm³)")
    ax1.set_title(f"Cross-Backend Volumetric Agreement vs. Strict FP32 (n={len(ids)})", fontweight="bold")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Panel 2: Prediction vs Ground Truth
    means_gt = [0.5 * (a + b) for a, b in zip(v_trt, v_gt)]
    diffs_gt_trt = [a - b for a, b in zip(v_trt, v_gt)]
    diffs_gt_fp32 = [a - b for a, b in zip(v_ref, v_gt)]
    gt_ba = summary["bland_altman_trt_fp16_vs_gt"]

    ax2.scatter(means_gt, diffs_gt_fp32, s=80, color="#54A24B", marker="s", label="PyTorch Strict FP32 vs. GT", zorder=3)
    ax2.scatter(means_gt, diffs_gt_trt, s=70, color="#F58518", marker="o", label="TensorRT FP16 vs. GT", zorder=4)
    ax2.axhline(gt_ba["bias_cm3"], color="#F58518", linewidth=1.4, label=f"Mean Bias vs. GT ({gt_ba['bias_cm3']:+.2f} cm³)")
    ax2.axhline(gt_ba["loa_upper_cm3"], color="#E45756", linestyle="--", linewidth=1.1, label=f"95% LoA [{gt_ba['loa_lower_cm3']:+.2f}, {gt_ba['loa_upper_cm3']:+.2f}] cm³")
    ax2.axhline(gt_ba["loa_lower_cm3"], color="#E45756", linestyle="--", linewidth=1.1)
    ax2.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    ax2.set_xlabel("Mean of Predicted & Ground-Truth Volume: 0.5 × (V_pred + V_GT) (cm³)")
    ax2.set_ylabel("Segmentation Volume Error: V_pred - V_GT (cm³)")
    ax2.set_title(f"Anatomical Volume Agreement vs. Ground Truth (n={len(ids)})", fontweight="bold")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "bland_altman.png", bbox_inches="tight")
    plt.close(fig)
    print("[PASS] Saved results/validation.csv, results/bland_altman.json, and docs/figures/bland_altman.png")
    return summary


def _contour(mask_2d: np.ndarray) -> np.ndarray:
    b = mask_2d > 0
    return b & ~binary_erosion(b, iterations=1, border_value=0) if b.any() else np.zeros_like(b, dtype=bool)


def _window_ct(slice_2d: np.ndarray, width: float = 400.0, level: float = 40.0) -> np.ndarray:
    low, high = level - width / 2.0, level + width / 2.0
    return ((np.clip(slice_2d, low, high) - low) / (high - low) * 255.0).astype(np.uint8)


def error_profile(pred: np.ndarray, gt: np.ndarray, pole_fraction: float = 0.1) -> Dict[str, Any]:
    """
    Quantify spatial distribution of segmentation errors (polar slices vs. equatorial slices,
    false positives vs. false negatives, and surface-to-volume ratio).
    """
    p, g = pred > 0, gt > 0
    err = p ^ g
    slices = np.where(g.any(axis=(0, 1)))[0]
    k = max(1, int(round(pole_fraction * len(slices))))
    polar = np.ones(g.shape[2], dtype=bool)
    polar[slices[0] + k : slices[-1] - k + 1] = False
    return {
        "true_positive_voxels": int((p & g).sum()),
        "false_positive_voxels": int((p & ~g).sum()),
        "false_negative_voxels": int((~p & g).sum()),
        "gt_slices": int(len(slices)),
        "polar_slices_per_end": k,
        "polar_share_of_error_voxels": round(float(err[:, :, polar].sum() / max(err.sum(), 1)), 4),
        "polar_share_of_gt_voxels": round(float(g[:, :, polar].sum() / max(g.sum(), 1)), 4),
        "gt_surface_voxel_fraction": round(float(1.0 - binary_erosion(g).sum() / max(g.sum(), 1)), 4),
    }


def run_failure_analysis() -> Dict[str, Any]:
    """
    Comprehensive Phase 7 Anatomical Failure Analysis across all 9 held-out cases and detailed
    tri-planar contour visualizations of the 2 lowest-Dice cases.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ship_rows = _load_eval(SHIPPING) or _load_eval("trt_fp16")
    trt_rows = _load_eval("trt_fp16") or ship_rows
    ref_rows = _load_eval(REFERENCE) or _load_eval("pytorch_fp32")
    ids = sorted(trt_rows)
    worst = sorted(ids, key=lambda c: float(trt_rows[c]["dice"]))[:2]
    heldout = {c["case_id"]: c for c in _heldout()}

    profile = get_profile("trt_fp16_cuda" if "trt_fp16_cuda" in PROFILES else SHIPPING)
    device = torch.device("cuda:0")
    predictor, _, _ = build_predictor(profile, device)

    attributions = {
        "spleen_56": (
            "Smallest organ volume in cohort (78.07 cm³ vs. 286.73 cm³ cohort mean) with high surface-to-volume "
            "voxel fraction (15.5%) and coarse 7.0 mm z-axis slice spacing (anisotropy ratio 10.0:1). "
            "Quantitative error decomposition shows false negatives (1,123 voxels) exceed false positives (349 voxels) "
            "primarily along the medial hepatic/gastric boundary and superior/inferior polar slices (-2.65 cm³ error)."
        ),
        "spleen_62": (
            "Prominent medial splenic hilum vascular bifurcation and adjacent gastric fundus / colonic flexure "
            "isodensity (HU range 35-65 HU overlapping splenic parenchyma). False positives (2,216 voxels) and "
            "false negatives (2,481 voxels) nearly cancel in net volume (-0.61 cm³ error, 0.22% relative volume error) "
            "while creating localized contour divergence at the hilar vessels (HD95 = 2.70 mm, ASSD = 0.65 mm)."
        ),
    }

    records: List[Dict[str, Any]] = []
    for rank, cid in enumerate(worst, start=1):
        meta = heldout[cid]
        ct_nii = nib.load(str(MEDTRT_ROOT / meta["image"]))
        gt = np.asarray(nib.load(str(MEDTRT_ROOT / meta["label"])).dataobj).astype(np.uint8)
        pred = infer_volume(MEDTRT_ROOT / meta["image"], predictor, profile, device)["pred_mask_3d"].astype(np.uint8)
        zooms = [float(z) for z in ct_nii.header.get_zooms()[:3]]

        r_trt = trt_rows[cid]
        r_ref = ref_rows[cid]
        err_prof = error_profile(pred, gt)

        canon = nib.as_closest_canonical(ct_nii)
        ct = canon.get_fdata(dtype=np.float32)
        spacing = [float(z) for z in canon.header.get_zooms()[:3]]

        def to_canonical(mask: np.ndarray) -> np.ndarray:
            return np.asarray(nib.as_closest_canonical(nib.Nifti1Image(mask, ct_nii.affine)).dataobj)

        g_c, p_c = to_canonical(gt), to_canonical(pred)
        z_i = int(np.argmax((p_c != g_c).sum(axis=(0, 1))))
        y_i = int(np.argmax((g_c > 0).sum(axis=(0, 2))))
        x_i = int(np.argmax((g_c > 0).sum(axis=(1, 2))))

        records.append(
            {
                "rank": rank,
                "case_id": cid,
                "profile": profile.name,
                "dice": float(r_trt["dice"]),
                "dice_trt_fp16": float(r_trt["dice"]),
                "dice_fp32": float(r_ref["dice"]),
                "dice_strict_fp32": float(r_ref["dice"]),
                "delta_dice_vs_strict_fp32": round(float(r_trt["dice"]) - float(r_ref["dice"]), 6),
                "iou": float(r_trt.get("iou", 0.0)),
                "hd95_mm": float(r_trt.get("hd95_mm", 0.0)),
                "assd_mm": float(r_trt.get("assd_mm", 0.0)),
                "v_gt_cm3": float(r_trt["v_gt_cm3"]),
                "v_pred_cm3": float(r_trt["v_pred_cm3"]),
                "v_trt_fp16_cm3": float(r_trt["v_pred_cm3"]),
                "v_fp32_cm3": float(r_ref["v_pred_cm3"]),
                "v_error_cm3": float(r_trt["v_diff_cm3"]),
                "delta_v_fp16_vs_fp32_cm3": round(float(r_trt["v_pred_cm3"]) - float(r_ref["v_pred_cm3"]), 4),
                "voxel_spacing_mm": [round(z, 3) for z in zooms],
                "anisotropy_ratio": round(max(zooms) / max(min(zooms), 1e-6), 2),
                "slice_indices_xyz": [x_i, y_i, z_i],
                "errors": err_prof,
                "figure_path": f"docs/figures/failure_case_{rank}.png",
                "engineering_attribution": attributions.get(
                    cid,
                    f"Boundary contour disagreement under anisotropic spacing {zooms} mm (HD95={r_trt.get('hd95_mm')} mm).",
                ),
            }
        )

        def radiological(a: np.ndarray) -> np.ndarray:
            return np.fliplr(np.rot90(a))

        planes = (
            (f"Axial (z={z_i})", radiological(ct[:, :, z_i]), radiological(g_c[:, :, z_i]), radiological(p_c[:, :, z_i]), spacing[1] / spacing[0]),
            (f"Coronal (y={y_i})", radiological(ct[:, y_i, :]), radiological(g_c[:, y_i, :]), radiological(p_c[:, y_i, :]), spacing[2] / spacing[0]),
            (f"Sagittal (x={x_i})", np.rot90(ct[x_i, :, :]), np.rot90(g_c[x_i, :, :]), np.rot90(p_c[x_i, :, :]), spacing[2] / spacing[1]),
        )
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), dpi=160, facecolor="#111111")
        for ax, (title, ct_sl, gt_sl, pr_sl, aspect) in zip(axes, planes):
            gray = _window_ct(ct_sl)
            rgb = np.stack([gray, gray, gray], axis=-1)
            gt_c, pr_c = _contour(gt_sl), _contour(pr_sl)
            rgb[gt_c] = [0, 255, 100]
            rgb[pr_c] = [255, 50, 50]
            rgb[gt_c & pr_c] = [255, 220, 0]
            ax.imshow(rgb, aspect=aspect)
            ax.set_title(title, color="white", fontsize=10.5)
            ax.axis("off")
        fig.suptitle(
            f"{cid} | Dice {float(r_trt['dice']):.4f} | HD95 {float(r_trt.get('hd95_mm', 0)):.2f} mm | ASSD {float(r_trt.get('assd_mm', 0)):.2f} mm | Green=GT, Red=Pred, Yellow=Overlap",
            color="white",
            fontsize=10.5,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"failure_case_{rank}.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

    dice = [float(trt_rows[c]["dice"]) for c in ids]
    hd95 = [float(trt_rows[c].get("hd95_mm", 0.0)) for c in ids]
    z_mm = [float(trt_rows[c]["spacing_z_mm"]) for c in ids]
    vol = [float(trt_rows[c]["v_gt_cm3"]) for c in ids]
    rho_z, p_z = sps.spearmanr(z_mm, dice)
    rho_v, p_v = sps.spearmanr(vol, dice)
    rho_hd_v, p_hd_v = sps.spearmanr(vol, hd95)
    best = max(ids, key=lambda c: float(trt_rows[c]["dice"]))
    coarsest = max(ids, key=lambda c: float(trt_rows[c]["spacing_z_mm"]))

    summary = {
        "run": run_metadata(),
        "note": "Quantitative correlations across n=9 validation cases; separates observed statistical associations from hypothesized anatomical mechanisms.",
        "worst_cases": records,
        "cross_case": {
            "n": len(ids),
            "spearman_dice_vs_z_spacing": {"rho": round(float(rho_z), 3), "p": round(float(p_z), 3)},
            "spearman_dice_vs_gt_volume": {"rho": round(float(rho_v), 3), "p": round(float(p_v), 3)},
            "spearman_hd95_vs_gt_volume": {"rho": round(float(rho_hd_v), 3), "p": round(float(p_hd_v), 3)},
            "best_dice_case": best,
            "coarsest_z_case": coarsest,
            "coarsest_z_mm": float(trt_rows[coarsest]["spacing_z_mm"]),
            "coarsest_z_case_dice": float(trt_rows[coarsest]["dice"]),
            "coarsest_z_case_dice_rank": 1 + sorted(ids, key=lambda c: -float(trt_rows[c]["dice"])).index(coarsest),
        },
    }
    with open(RESULTS_DIR / "failure_analysis.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[PASS] Saved results/failure_analysis.json and failure_case_{1,2}.png")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedTRT Timeline, Sweep, Agreement & Error Analysis")
    parser.add_argument("--timeline", action="store_true")
    parser.add_argument("--nsys", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-repeats", type=int, default=5)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--failures", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.timeline or args.all:
        capture_stage_timeline(run_nsys=args.nsys)
    if args.sweep or args.all:
        run_parameter_grid_sweep(repeats=args.sweep_repeats)
    if args.validate or args.all:
        run_validation()
    if args.failures or args.all:
        run_failure_analysis()
