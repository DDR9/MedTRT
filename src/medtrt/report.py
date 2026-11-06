"""
README Generator (`src/medtrt/report.py`).

Every number in `README.md` is read from `results/`, and every pass/fail word is computed from those
numbers. `render_readme()` is deterministic, so `medtrt.verify_claims` can confirm the published
README is an exact rendering of the results.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from medtrt.config import FIGURES_DIR, MEDTRT_ROOT, RESULTS_DIR
from medtrt.profiles import GPU_BASELINE, HOST_BASELINE, LIKE_FOR_LIKE, PROFILES, REFERENCE, SHIPPING

STAGES = (
    ("load", "load", "#4C78A8"),
    ("preprocess", "preprocess", "#F58518"),
    ("h2d", "h2d", "#72B7B2"),
    ("sw_network", "sw_network", "#54A24B"),
    ("sw_aggregate", "sw_aggregate", "#EECA3B"),
    ("postprocess", "postprocess", "#E45756"),
)


def _json(name: str) -> Dict[str, Any]:
    with open(RESULTS_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


def _csv(name: str) -> List[Dict[str, str]]:
    with open(RESULTS_DIR / name, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _table(header: List[str], rows: List[List[str]], align: List[str] | None = None) -> str:
    marks = align or [":---"] + [":---:"] * (len(header) - 1)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(marks) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _status(ok: bool) -> str:
    return "PASS" if ok else "**FAIL**"


def generate_stage_breakdown_chart(bench: Dict[str, Dict[str, Any]]) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    groups = (
        ("Host path", [k for k in bench if not PROFILES[k].gpu_path]),
        ("GPU path", [k for k in bench if PROFILES[k].gpu_path]),
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=160)
    for ax, (title, keys) in zip(axes, groups):
        x = np.arange(len(keys))
        bottoms = np.zeros(len(keys))
        for stage, label, color in STAGES:
            vals = np.array([bench[k]["stage_latency_ms"][stage]["median"] for k in keys])
            ax.bar(x, vals, bottom=bottoms, width=0.6, color=color, edgecolor="black", linewidth=0.5, label=label)
            bottoms += vals
        for i, k in enumerate(keys):
            ax.text(i, bottoms[i] * 1.01, f"{bench[k]['stage_latency_ms']['e2e_total']['median']:.0f} ms",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")  # fmt: skip
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=20, ha="right", fontsize=8.5)
        ax.set_ylabel("Median latency per volume (ms)")
        ax.set_title(f"{title} (separate y-axes)", fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    out = FIGURES_DIR / "stage_breakdown.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def render_readme() -> str:
    env = _json("env.json")
    build = _json("trt_build.json")
    agreement = _json("bland_altman.json")
    failures = _json("failure_analysis.json")
    timeline = _json("timeline_summary.json")
    ablation = _json("ablation.json")["rungs"]
    probes = _json("probes.json")
    heldout = _json("heldout_cases.json")
    sweep = _csv("sweep.csv")
    bench = {k: _json(f"bench_{k}.json") for k in PROFILES}
    evals = {k: _csv(f"eval_{k}.csv") for k in PROFILES}

    def stage(name: str, key: str) -> Dict[str, float]:
        return bench[name]["stage_latency_ms"][key]

    def e2e(name: str) -> float:
        return stage(name, "e2e_total")["median"]

    def mean_dice(name: str) -> float:
        return float(np.mean([float(r["dice"]) for r in evals[name]]))

    def sd_dice(name: str) -> float:
        return float(np.std([float(r["dice"]) for r in evals[name]], ddof=1))

    ship = PROFILES[SHIPPING]
    ship_agree = agreement["profiles"][SHIPPING]
    ship_dice = ship_agree["dice_vs_reference"]
    ship_vol = ship_agree["volume_vs_reference"]
    per_case = bench[SHIPPING]["per_case"]
    case_medians = {c: v["e2e_total"]["median"] for c, v in per_case.items()}
    max_pass = stage(SHIPPING, "e2e_total")["max"]
    n_under_200 = sum(v < 200.0 for v in case_medians.values())

    net = probes["network_sustained"]
    bundle_key = "0.5|4"
    fast_key = f"{ship.overlap}|{ship.sw_batch_size}"
    fp16 = probes["fp16_numerics"]
    ingest = probes["ingestion"]

    results = _table(
        ["Measure", f"Fast profile (`{SHIPPING}`)", "Notes"],
        [
            ["End-to-end latency", f"`{e2e(SHIPPING):.1f} ms` median", f"Per-case medians `{min(case_medians.values()):.0f}`-`{max(case_medians.values()):.0f} ms`; slowest of 90 passes `{max_pass:.0f} ms`"],
            ["Speedup vs host-path PyTorch", f"`{e2e(HOST_BASELINE) / e2e(SHIPPING):.1f}x`", f"`{e2e(HOST_BASELINE):.0f} ms` baseline with CPU inverse transform and gzip reads"],
            ["Speedup vs GPU-path PyTorch", f"`{e2e(GPU_BASELINE) / e2e(SHIPPING):.2f}x`", f"`{e2e(GPU_BASELINE):.0f} ms` with the same GPU pipeline and bundle windows, no TensorRT"],
            ["TensorRT alone, same pipeline and windows", f"`{e2e(GPU_BASELINE) / e2e(LIKE_FOR_LIKE):.2f}x` end to end", f"`{e2e(GPU_BASELINE):.0f}` to `{e2e(LIKE_FOR_LIKE):.0f} ms`"],
            ["Mean Dice", f"`{ship_dice['mean_dice']:.6f}`", f"True FP32 reference `{agreement['reference_mean_dice']:.6f}`"],
            ["Mean / max per-case Dice change vs FP32", f"`{ship_dice['mean_abs_delta_dice']:.6f}` / `{ship_dice['max_abs_delta_dice']:.6f}`", f"`{ship_dice['n_cases_over_tolerance']}` of `{ship_vol['n']}` cases at or over `{ship_dice['tolerance']}`"],
            ["Volume bias vs FP32", f"`{ship_vol['bias_cm3']:+.2f} cm³` (`{ship_vol['bias_pct_of_mean_volume']:+.2f}%`)", f"95% CI `[{ship_vol['bias_ci95_lower_cm3']:+.2f}, {ship_vol['bias_ci95_upper_cm3']:+.2f}]`; `{ship_vol['n_positive']}` of `{ship_vol['n']}` cases positive"],
            ["Peak VRAM", f"`{bench[SHIPPING]['peak_vram_mb']['max'] / 1024:.2f} GB`", "Largest case"],
        ],
        [":---", ":---", ":---"],
    )

    l4l_dice = agreement["profiles"][LIKE_FOR_LIKE]["dice_vs_reference"]
    targets = _table(
        ["Target", "Measured", "Status"],
        [
            ["Every pass under 500 ms", f"Slowest pass `{max_pass:.1f} ms`", _status(max_pass < 500.0)],
            ["Mean Dice at least 0.960", f"`{ship_dice['mean_dice']:.6f}`", _status(ship_dice["mean_dice"] >= 0.960)],
            ["Mean absolute Dice change under 0.005", f"`{ship_dice['mean_abs_delta_dice']:.6f}`", _status(ship_dice["mean_within_tolerance"])],
            [f"Like-for-like (`{LIKE_FOR_LIKE}`) every case Dice change under 0.005", f"Max `{l4l_dice['max_abs_delta_dice']:.6f}` (`{SHIPPING}` max `{ship_dice['max_abs_delta_dice']:.6f}`)", _status(l4l_dice["every_case_within_tolerance"])],
        ],
    )

    ladder_rows = []
    base = ablation["a_host_tf32"]["stages_ms"]["e2e_total"]["median"]
    for rung in ablation.values():
        p = rung["profile"]
        med = rung["stages_ms"]["e2e_total"]["median"]
        ladder_rows.append(
            [
                p["label"],
                f"`{med:.1f}`",
                f"`{rung['stages_ms']['e2e_total']['max']:.1f}`",
                f"`{base / med:.2f}x`",
                f"`{rung['stages_ms']['sw_network']['median']:.1f}`",
                f"`{rung['mean_dice']:.6f}`",
            ]
        )
    ladder = _table(["Configuration", "E2E median (ms)", "Slowest pass (ms)", "vs first row", "`sw_network` (ms)", "Mean Dice"], ladder_rows)

    def rung_e2e(key: str) -> float:
        return ablation[key]["stages_ms"]["e2e_total"]["median"]

    def rung_net(key: str) -> float:
        return ablation[key]["stages_ms"]["sw_network"]["median"]

    matrix_rows = []
    for k, p in PROFILES.items():
        patch = bench[k]["patch_forward_latency_ms"]
        s_e2e = stage(k, "e2e_total")
        matrix_rows.append(
            [
                f"`{k}`",
                f"`{p.overlap}` / `{p.sw_batch_size}`",
                f"`{patch['batch_1']['median']:.2f}`",
                f"`{patch['batch_4']['median']:.2f}`",
                f"`{stage(k, 'sw_network')['median']:.1f}`",
                f"`{s_e2e['median']:.1f}` [`{s_e2e['iqr']:.1f}`]",
                f"`{s_e2e['max']:.0f}`",
                f"`{bench[k]['peak_vram_mb']['max']:.0f}`",
                f"`{mean_dice(k):.6f}` ± `{sd_dice(k):.4f}`",
                f"`{e2e(HOST_BASELINE) / s_e2e['median']:.2f}x`",
            ]
        )
    matrix = _table(
        ["Profile", "Overlap / batch", "Patch B=1 (ms)", "Patch B=4 (ms)", "`sw_network` (ms)", "E2E median [IQR] (ms)", "Slowest (ms)", "Peak VRAM (MB)", "Mean Dice ± SD", f"vs `{HOST_BASELINE}`"],
        matrix_rows,
    )

    stage_rows = []
    for key in ("load", "h2d", "ingest", "preprocess", "sw_network", "sw_aggregate", "postprocess", "e2e_total", "wall_ms"):
        stage_rows.append([f"`{key}`"] + [f"`{stage(k, key)['median']:.1f}`" for k in PROFILES])
    stages = _table(["Stage (median ms)"] + [f"`{k}`" for k in PROFILES], stage_rows)

    case_rows = []
    shapes = {r["case_id"]: r for r in evals[SHIPPING]}
    for c, v in per_case.items():
        r = shapes[c]
        case_rows.append(
            [
                f"`{c}`",
                f"`{r['shape_x']}x{r['shape_y']}x{r['shape_z']}`",
                f"`{v['patch_windows']}`",
                f"`{v['e2e_total']['median']:.1f}`",
                f"`{v['e2e_total']['max']:.1f}`",
                f"`{v['peak_vram_mb']['max']:.0f}`",
            ]
        )
    cases = _table(["Case", "Shape", "Windows", "E2E median (ms)", "Slowest (ms)", "Peak VRAM (MB)"], case_rows)

    net_rows = []
    for name, label in (
        ("strict_fp32", "PyTorch FP32"),
        ("tf32", "PyTorch TF32"),
        ("tf32_tuned", "PyTorch TF32, `cudnn.benchmark`"),
        ("amp", "PyTorch AMP"),
        ("amp_tuned", "PyTorch AMP, `cudnn.benchmark`"),
        ("trt", "TensorRT FP16"),
    ):
        b_ms = net[f"{name}|{bundle_key}"]["sw_network_ms_median"]
        f_ms = net[f"{name}|{fast_key}"]["sw_network_ms_median"]
        trt_b = net[f"trt|{bundle_key}"]["sw_network_ms_median"]
        net_rows.append([label, f"`{b_ms:.1f}`", f"`{f_ms:.1f}`", f"`{b_ms / trt_b:.2f}x`"])
    network = _table(["Backend", "Bundle windows (ms)", "Fast windows (ms)", "TensorRT speedup, bundle windows"], net_rows)
    windows_bundle = net[f"trt|{bundle_key}"]["windows_all_cases"]
    windows_fast = net[f"trt|{fast_key}"]["windows_all_cases"]

    clock_rows = [
        [f"`{k.split('|')[0]}`", f"`{v['back_to_back_ms_median']:.1f}`", f"`{v['after_3s_idle_ms_median']:.1f}`", f"`{v['idle_inflation']:.2f}x`"]
        for k, v in probes["clock_state"].items()
    ]
    clock = _table(["Backend", "Back-to-back (ms)", "After 3 s idle (ms)", "Inflation"], clock_rows)

    ingestion = _table(
        ["Read path", "Read call (ms)", "Through to GPU (ms)"],
        [
            ["`.nii.gz` with `LoadImaged`", f"`{ingest['gzip']['read_ms_median']:.1f}`", f"`{ingest['gzip']['to_gpu_total_ms_median']:.1f}`"],
            ["`.nii`, eager read", f"`{ingest['nii_eager']['read_ms_median']:.1f}`", f"`{ingest['nii_eager']['to_gpu_total_ms_median']:.1f}`"],
            ["`.nii`, `mmap` (lazy)", f"`{ingest['nii_mmap']['read_ms_median']:.1f}`", f"`{ingest['nii_mmap']['to_gpu_total_ms_median']:.1f}`"],
        ],
    )

    amdahl_rows = []
    for k in (HOST_BASELINE, "trt_fp16", GPU_BASELINE, SHIPPING):
        a = bench[k]["amdahl"]
        amdahl_rows.append(
            [
                f"`{k}`",
                f"`{a['median_sw_network_ms']:.1f}`",
                f"`{a['median_e2e_ms']:.1f}`",
                f"`{a['network_fraction_p'] * 100:.1f}%`",
                f"`{a['theoretical_max_e2e_speedup_1_over_s']:.2f}x`",
            ]
        )
    amdahl = _table(["Profile", "`sw_network` (ms)", "E2E (ms)", "Network share", "Max E2E gain from a faster network"], amdahl_rows)

    sweep_rows = [
        [
            f"`{float(r['overlap']):.2f}`",
            f"`{r['sw_batch_size']}`",
            f"`{float(r['median_sw_latency_ms']):.1f}`",
            f"`{float(r['iqr_sw_latency_ms']):.1f}`",
            f"`{float(r['incremental_vram_mb']):.0f}`",
            f"`{float(r['mean_dice']):.6f}`",
            "yes" if r["pareto_optimal"] == "True" else "",
            "yes" if r["is_fast_profile"] == "True" else "",
        ]
        for r in sweep
    ]
    sweep_table = _table(["Overlap", "Batch", "Median (ms)", "IQR (ms)", "Added VRAM (MB)", "Mean Dice", "Pareto-optimal", "Fast profile"], sweep_rows)
    fast_row = next(r for r in sweep if r["is_fast_profile"] == "True")
    if fast_row["pareto_optimal"] == "True":
        sweep_note = "The fast profile is on the measured Pareto frontier."
    else:
        sweep_note = "The fast profile is not on the measured Pareto frontier of this sweep; its window settings are a judgment call."

    agree_rows = []
    for k, block in agreement["profiles"].items():
        d, v = block["dice_vs_reference"], block["volume_vs_reference"]
        agree_rows.append(
            [
                f"`{k}`",
                f"`{d['mean_dice']:.6f}`",
                f"`{d['mean_delta_dice']:+.6f}`",
                f"`{d['mean_abs_delta_dice']:.6f}`",
                f"`{d['max_abs_delta_dice']:.6f}`",
                f"`{d['n_cases_over_tolerance']}`",
                f"`{v['bias_cm3']:+.2f}` [`{v['bias_ci95_lower_cm3']:+.2f}`, `{v['bias_ci95_upper_cm3']:+.2f}`]",
                f"[`{v['loa_lower_cm3']:+.2f}`, `{v['loa_upper_cm3']:+.2f}`]",
                f"`{v['n_positive']}/{v['n']}`",
                f"`{block['volume_vs_gt']['mae_cm3']:.2f}`",
            ]
        )
    agree = _table(
        ["Profile", "Mean Dice", "Mean ΔDice", "Mean abs ΔDice", "Max abs ΔDice", "Cases ≥ 0.005", "Volume bias (cm³) [95% CI]", "95% LoA (cm³)", "Volume larger than FP32", "MAE vs ground truth (cm³)"],
        agree_rows,
    )

    val_rows = []
    for r in _csv("validation.csv"):
        val_rows.append(
            [
                f"`{r['case_id']}`",
                f"`{float(r['v_gt_cm3']):.2f}`",
                f"`{float(r[f'v_{REFERENCE}']):.2f}`",
                f"`{float(r[f'v_{SHIPPING}']):.2f}`",
                f"`{float(r[f'delta_v_{SHIPPING}']):+.3f}`",
                f"`{float(r[f'dice_{REFERENCE}']):.6f}`",
                f"`{float(r[f'dice_{SHIPPING}']):.6f}`",
                f"`{float(r[f'delta_dice_{SHIPPING}']):+.6f}`",
            ]
        )
    per_case_agree = _table(["Case", "V ground truth", "V FP32", "V fast", "ΔV", "Dice FP32", "Dice fast", "ΔDice"], val_rows)

    cross = failures["cross_case"]
    failure_lines = []
    for w in failures["worst_cases"]:
        e = w["errors"]
        failure_lines.append(
            f"- **`{w['case_id']}`:** Dice `{w['dice']:.4f}` (FP32 `{w['dice_strict_fp32']:.4f}`), volume `{w['v_pred_cm3']:.1f}` vs `{w['v_gt_cm3']:.1f} cm³`, "
            f"spacing `{w['voxel_spacing_mm']}` mm. False-positive voxels `{e['false_positive_voxels']}`, false-negative `{e['false_negative_voxels']}`. "
            f"The `{e['polar_slices_per_end']}` end slices at each pole, plus any slices beyond the organ, hold `{e['polar_share_of_error_voxels'] * 100:.0f}%` of error voxels "
            f"and `{e['polar_share_of_gt_voxels'] * 100:.0f}%` of ground-truth voxels. `{e['gt_surface_voxel_fraction'] * 100:.0f}%` of ground-truth voxels are surface voxels."
        )

    fold = probes.get("fold_effect", {})
    fold_section = ""
    if fold.get("available"):
        rows = [
            [
                label,
                f"`{fold[key]['mean_dice']:.6f}`",
                f"`{fold[key]['mean_abs_delta_dice']:.6f}`",
                f"`{fold[key]['volume_bias_cm3']:+.3f}`",
                f"`{fold[key]['n_positive']}/{fold[key]['n']}`",
            ]
            for label, key in (("Unfolded FP16 engine", "unfolded"), ("Folded FP16 engine", "folded"))
        ]
        fold_section = (
            "### The BatchNorm fold\n\n"
            "The exported ONNX graph keeps `BatchNorm` after each `ConvTranspose3d`. Layer "
            f"`{fold['largest_bn_shift_layer']}` applies a shift of `{fold['largest_bn_shift']:.1f}`, a magnitude where FP16 "
            "values are spaced `0.0625` apart. Folding these layers into the preceding `ConvTranspose3d` in float64 before "
            f"export changes FP32 logits by at most `{fold['fp32_fold_max_abs_logit_diff']:.1e}`, so it does not change the model. "
            "Both engines below run the same inputs and windows, compared with FP32 on the unmodified bundle weights.\n\n"
            + _table(["Engine", "Mean Dice", "Mean abs ΔDice", "Volume bias (cm³)", "Volume larger than FP32"], rows)
            + "\n\nThe unfolded engine's higher Dice is a side effect of that shift, not better accuracy."
        )

    trt_vol = agreement["profiles"][LIKE_FOR_LIKE]["volume_vs_reference"]
    if trt_vol["bias_ci_excludes_zero"]:
        drift_line = (
            f"TensorRT FP16 shifts volume by a small, consistent amount: `{trt_vol['bias_cm3']:+.3f} cm³` at bundle "
            f"windows, with `{trt_vol['n_positive']}` of `{trt_vol['n']}` cases larger than FP32."
        )
    else:
        drift_line = (
            f"TensorRT FP16 shows no systematic volume shift at bundle windows: `{trt_vol['bias_cm3']:+.3f} cm³`, "
            f"with `{trt_vol['n_positive']}` of `{trt_vol['n']}` cases larger than FP32."
        )

    if timeline["nsys_cli_executed"]:
        nsight_line = f"Nsight Systems ran, and its trace is `{timeline['nsys_rep_path']}`."
    else:
        nsight_line = (
            "Nsight Systems was not run for these results. NVTX ranges are in the code, and `--nsys` will use them when "
            "Nsight is installed, but every number here comes from CUDA events and wall-clock timers."
        )

    return f"""# MedTRT: GPU-Accelerated 3D CT Segmentation, Profiled and Validated

> **Research benchmark only.** This is a systems study of one public model on one laptop. It is not a medical device, and it makes no diagnostic, triage, or clinical performance claims.
>
> Every number below is generated from `results/`. To check them without a GPU, run `python -m medtrt.verify_claims`.

## 1. Results

A pretrained MONAI 3D UNet segments the spleen in abdominal CT. All timings are on an `{env['gpu_model']}`.

{results}

### Targets

{targets}

## 2. Where the speedup comes from

Most of the end-to-end gain is not TensorRT. Each row below changes one thing, using the same 9 cases.

{ladder}

- **Keeping the inverse transform on the GPU is the largest single change.** The reference `inference.json` already keeps predictions on the GPU, so the host path overstates the cost of a default deployment. With gzip reads, this alone moves PyTorch from `{rung_e2e('a_host_tf32'):.0f}` to `{rung_e2e('c_gpu_gzip_tf32'):.0f} ms`.
- **Removing gzip from the critical path** moves PyTorch to `{rung_e2e('e_gpu_nii_tf32'):.0f} ms`. It assumes volumes were decompressed ahead of time.
- **TensorRT** then takes the same pipeline from `{rung_e2e('e_gpu_nii_tf32'):.0f}` to `{rung_e2e('f_gpu_nii_trt'):.0f} ms`. PyTorch without TensorRT already meets the 500 ms target.
- **Inverting class probabilities rather than hard labels matters for accuracy.** Nearest-label inversion scores `{ablation['i_fast_trt_nn']['mean_dice']:.6f}` Dice at `{rung_e2e('i_fast_trt_nn'):.0f} ms`. Trilinear scores `{ablation['h_fast_trt']['mean_dice']:.6f}` at `{rung_e2e('h_fast_trt'):.0f} ms`.

## 3. Method

{_table(["Component", "Value"], [
    ["GPU", f"`{env['gpu_model']}`, compute capability `{env['compute_capability']}`, `{env['vram_gb']} GB`"],
    ["Driver", f"`{env['driver_version']}`, `{env.get('driver_model', 'unknown')}` mode, CUDA `{env['cuda_runtime_version']}`, cuDNN `{env['cudnn_version']}`"],
    ["Software", f"PyTorch `{env['pytorch_version']}`, MONAI `{env['monai_version']}`, TensorRT `{env['tensorrt_version']}`, Python `{env['python_version']}`"],
    ["OS", f"`{env['os']}`"],
    ["ONNX export", f"Opset `{build['onnx']['opset_version']}`, max abs diff vs FP32 `{build['onnx']['max_abs_diff_vs_pytorch_fp32']:.3e}` on one synthetic batch"],
    ["TensorRT engine", f"FP16, dynamic batch 1 to 8, optimized for 2, `{build['engine_size_mb']:.2f} MB`"],
], [":---", ":---"])}

- **Data:** Medical Segmentation Decathlon `Task09_Spleen` and the MONAI `spleen_ct_segmentation` bundle v`{heldout.get('bundle_version', '0.6.1')}`, with weights verified by SHA-256.
- **Evaluation cases:** the `{heldout['heldout_count']}` of `{heldout['total_training_cases']}` labeled cases that the bundle reserves for validation. They were never trained on, but the bundle selected its checkpoint by Dice on these same cases, and MSD test labels are not public. There is no untouched test set here, so treat Dice as an optimistic estimate, suitable for comparing backends.
- **Timing:** 3 warm-up volumes, then 10 passes on each of 9 cases, for 90 timed passes per profile. Stages use CUDA events. Each pass is also timed by wall clock, and the two agree (see `wall_ms`). Every pass ends with the label map in host memory.
- **Profiles:** the host path reads gzip and transforms on the CPU. The GPU path reads pre-decompressed `.nii` and transforms on the GPU. `{REFERENCE}` is true FP32 on the unmodified bundle weights and is the numerical reference. All other profiles use the BatchNorm-folded weights. `pytorch_tf32` is PyTorch's default and is not FP32. `{SHIPPING}` is the fast profile. `trt_fp16_nn` is kept only as a comparison.
- **Profiling:** {nsight_line}

## 4. Benchmark

{matrix}

![Stage latency breakdown](docs/figures/stage_breakdown.png)

### Per-stage medians

`ingest` is `load + h2d`. Quote it instead of `load` alone, because `mmap` defers the read into `h2d`.

{stages}

### Per-case latency, fast profile

`{e2e(SHIPPING):.0f} ms` is a median across cases of different sizes: `{n_under_200}` of `{len(case_medians)}` cases have a median under 200 ms.

{cases}

## 5. Network-only comparison

These runs use identical windows on every backend, in a sustained loop. Fast windows use `{windows_fast}` windows across the 9 cases against `{windows_bundle}` for bundle windows, so comparing a fast-window time to a bundle-window time measures less work, not a faster network.

{network}

Inside the real pipeline, TensorRT is `{rung_net('e_gpu_nii_tf32') / rung_net('f_gpu_nii_trt'):.2f}x` faster than TF32 at bundle windows and `{rung_net('g_fast_tf32') / rung_net('h_fast_trt'):.2f}x` at fast windows.

### GPU clock state

`sw_network` depends on what the GPU was doing just before, so values from different pipelines are not directly comparable. The same work takes longer after a 3 s idle gap, which is consistent with the laptop GPU lowering its clocks during long CPU stages. Sustained loops can also run slower than the short bursts inside the real pipeline, so the table above and the in-pipeline ratios both matter.

{clock}

## 6. Ingestion

Warm OS page cache. `mmap` returns before reading, so only the right-hand column is a fair comparison. Nearly all of the gain comes from not decompressing gzip.

{ingestion}

Resampling runs on the GPU in `{probes['resample_dtype']['inverse'][0]['dtype'].replace('torch.', '')}`, which is MONAI's default and is slow on GeForce hardware. Only the final mask copy is `{probes['resample_dtype']['final_mask_dtype'].replace('torch.', '')}`.

## 7. Amdahl analysis

{amdahl}

Once the network is a small share of end-to-end time, a faster network barely moves the total.

## 8. Sliding-window sweep

Uses the fast profile's own post-processing, with 5 timed passes per case on a hot GPU. {sweep_note} The cases are the evaluation cases, so this is selection on the evaluation set.

{sweep_table}

![Sliding-window sweep](docs/figures/pareto_sweep.png)

## 9. Agreement with true FP32

All rows compare with `{REFERENCE}` (mean Dice `{agreement['reference_mean_dice']:.6f}`). The sample is `{ship_vol['n']}` cases, so these limits of agreement are imprecise.

{agree}

{drift_line} Across all sliding windows TensorRT flipped `{fp16['window_voxels_background_to_foreground']}` window-voxels from background to foreground and `{fp16['window_voxels_foreground_to_background']}` the other way. The largest logit difference is `{fp16['max_abs_logit_diff']:.3f}` on logits up to `{fp16['reference_logit_abs_max']:.1f}`.

{fold_section}

### Per-case agreement, fast profile

{per_case_agree}

![Bland-Altman agreement](docs/figures/bland_altman.png)

## 10. Error analysis

These are observations about where errors fall, not explanations of why. FP16 is not the main source of error on these cases: the Dice change from FP32 is far smaller than the distance from a perfect score.

{chr(10).join(failure_lines)}

Across all `{cross['n']}` cases, Dice is not clearly related to slice thickness (Spearman `{cross['spearman_dice_vs_z_spacing']['rho']:+.2f}`, p = `{cross['spearman_dice_vs_z_spacing']['p']:.2f}`) or to spleen volume (Spearman `{cross['spearman_dice_vs_gt_volume']['rho']:+.2f}`, p = `{cross['spearman_dice_vs_gt_volume']['p']:.2f}`). The coarsest scan, `{cross['coarsest_z_case']}` at `{cross['coarsest_z_mm']:.1f} mm`, ranks number `{cross['coarsest_z_case_dice_rank']}` of `{cross['n']}` by Dice.

![Lowest-Dice case](docs/figures/failure_case_1.png)
![Second-lowest-Dice case](docs/figures/failure_case_2.png)

## 11. Limitations

- **Small sample:** 9 cases from one public dataset, which also selected the checkpoint.
- **One machine:** one laptop GPU, whose clocks change with load and temperature. Expect different absolute numbers elsewhere.
- **Best-case I/O:** the GPU path assumes volumes are already decompressed and in the OS page cache. Cold-disk reads and the one-time decompression are not included.
- **Warm process:** model loading, engine deserialization, and CUDA start-up are excluded.
- **Not a product path:** no DICOM ingestion, result saving, or concurrency is measured.
- **Engine builds:** TensorRT selects kernels at build time, so a rebuilt engine can differ slightly.

## 12. Reproduce

Check that this README matches `results/`. No GPU or data needed:

```bash
python -m medtrt.verify_claims
```

Regenerate everything on an NVIDIA GPU. This downloads the dataset and bundle, and takes about an hour:

```bash
pip install -r requirements-lock.txt
pip install -e .
python -m medtrt.pipeline
```

Run the tests, or the viewer:

```bash
python -m pytest tests/
streamlit run app/viewer.py
```

## 13. Data and licenses

The dataset, weights, and engines are not in this repository. `Task09_Spleen` is from the [Medical Segmentation Decathlon](http://medicaldecathlon.com/) under CC-BY-SA 4.0, and the figures in `docs/figures/failure_case_*.png` are derived from it. The MONAI bundle is Apache-2.0.
"""


def write_state() -> None:
    bench = _json(f"bench_{SHIPPING}.json")
    agreement = _json("bland_altman.json")["profiles"][SHIPPING]["dice_vs_reference"]
    state = {
        "current_phase": "COMPLETE",
        "run": bench["run"],
        "fast_profile": SHIPPING,
        "median_e2e_ms": bench["stage_latency_ms"]["e2e_total"]["median"],
        "max_e2e_ms": bench["stage_latency_ms"]["e2e_total"]["max"],
        "mean_dice": agreement["mean_dice"],
    }
    with open(RESULTS_DIR / "STATE.json", "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def generate_readme() -> Path:
    generate_stage_breakdown_chart({k: _json(f"bench_{k}.json") for k in PROFILES})
    readme_path = MEDTRT_ROOT / "README.md"
    with open(readme_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(render_readme())
    write_state()
    print(f"[PASS] Generated {readme_path.name}")
    return readme_path


if __name__ == "__main__":
    generate_readme()
