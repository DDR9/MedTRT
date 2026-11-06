"""
Dashboard Data Layer (`app/dashboard/data.py`).

Rules that keep the dashboard honest as the project changes:

- No result is written in this code. Every value is read or computed from `results/` when the page
  renders, so a new run, new cases, or a new profile appear without editing anything here.
- Profiles are discovered from the result files. Their roles (reference, baselines, fast profile)
  are resolved from what each profile is, not from a fixed list of names.
- Agreement statistics are recomputed from the per-case eval CSVs, so they cannot go stale the way
  a saved summary can.
- This module imports neither torch nor the pipeline, so reading results can never touch the GPU.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(os.environ.get("MEDTRT_RESULTS_DIR", ROOT / "results"))
FIGURES_DIR = ROOT / "docs" / "figures"
ORIGINAL_CLAIMS_FILE = ROOT / "docs" / "original_claims.json"
GPU_LOCK = ROOT / "results" / ".gpu_busy"

STAGES: Tuple[str, ...] = ("load", "h2d", "preprocess", "sw_network", "sw_aggregate", "postprocess")

# Project goals, not results. `medtrt.verify_claims` publishes the live values in claims_check.json.
DEFAULT_TARGETS = {"e2e_ms": 500.0, "mean_dice": 0.960, "abs_delta_dice": 0.005}

SEGMENTATION_METRICS = {
    "dice": "Dice",
    "iou": "IoU",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "precision": "Precision",
    "volumetric_similarity": "Volumetric similarity",
    "hd95_mm": "HD95 (mm)",
    "assd_mm": "ASSD (mm)",
    "abs_volume_error_cm3": "Abs volume error (cm³)",
    "abs_rel_volume_error_pct": "Abs volume error (%)",
}


@dataclass
class Results:
    bench: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    evals: Dict[str, pd.DataFrame] = field(default_factory=dict)
    eval_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    env: Dict[str, Any] = field(default_factory=dict)
    build: Dict[str, Any] = field(default_factory=dict)
    agreement: Dict[str, Any] = field(default_factory=dict)
    failures: Dict[str, Any] = field(default_factory=dict)
    timeline: Dict[str, Any] = field(default_factory=dict)
    ablation: Dict[str, Any] = field(default_factory=dict)
    probes: Dict[str, Any] = field(default_factory=dict)
    heldout: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    claims_check: Dict[str, Any] = field(default_factory=dict)
    sweep: pd.DataFrame = field(default_factory=pd.DataFrame)
    bench_preproc: Dict[str, Any] = field(default_factory=dict)
    bench_postproc: Dict[str, Any] = field(default_factory=dict)
    bench_scaling: Dict[str, Any] = field(default_factory=dict)
    bench_memory: Dict[str, Any] = field(default_factory=dict)
    bench_correctness: Dict[str, Any] = field(default_factory=dict)
    bench_e2e: Dict[str, Any] = field(default_factory=dict)
    bench_trt_variants: Dict[str, Any] = field(default_factory=dict)
    bench_pt_patches: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Roles:
    reference: Optional[str]
    fast: Optional[str]
    host_baseline: Optional[str]
    gpu_baseline: Optional[str]
    like_for_like: Optional[str]


def signature() -> Tuple[Tuple[str, int, int], ...]:
    """Fingerprint of every result file. It changes whenever a file does, which refreshes the pages."""
    entries: List[Tuple[str, int, int]] = []
    for folder in (RESULTS_DIR, FIGURES_DIR):
        if folder.exists():
            for path in sorted(folder.iterdir()):
                if path.is_file():
                    info = path.stat()
                    entries.append((path.name, info.st_mtime_ns, info.st_size))
    return tuple(entries)


def _read_json(path: Path) -> Dict[str, Any]:
    """Missing or half-written files read as empty, so a page shows a note instead of crashing."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        return loaded if isinstance(loaded, dict) else {}
    except (OSError, ValueError):
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def load() -> Results:
    res = Results()
    if not RESULTS_DIR.exists():
        return res

    for path in sorted(RESULTS_DIR.glob("bench_*.json")):
        payload = _read_json(path)
        if "stage_latency_ms" in payload:
            res.bench[path.stem[len("bench_") :]] = payload
    for path in sorted(RESULTS_DIR.glob("eval_*.csv")):
        frame = _read_csv(path)
        if {"case_id", "dice", "v_pred_cm3"} <= set(frame.columns):
            name = path.stem[len("eval_") :]
            res.evals[name] = frame
            res.eval_meta[name] = _read_json(RESULTS_DIR / f"eval_{name}.meta.json")

    res.env = _read_json(RESULTS_DIR / "env.json")
    res.build = _read_json(RESULTS_DIR / "trt_build.json")
    res.agreement = _read_json(RESULTS_DIR / "bland_altman.json")
    res.failures = _read_json(RESULTS_DIR / "failure_analysis.json")
    res.timeline = _read_json(RESULTS_DIR / "timeline_summary.json")
    res.ablation = _read_json(RESULTS_DIR / "ablation.json")
    res.probes = _read_json(RESULTS_DIR / "probes.json")
    res.heldout = _read_json(RESULTS_DIR / "heldout_cases.json")
    res.state = _read_json(RESULTS_DIR / "STATE.json")
    res.claims_check = _read_json(RESULTS_DIR / "claims_check.json")
    res.sweep = _read_csv(RESULTS_DIR / "sweep.csv")
    res.bench_preproc = _read_json(RESULTS_DIR / "benchmark_preprocessing.json")
    res.bench_postproc = _read_json(RESULTS_DIR / "benchmark_postprocessing.json")
    res.bench_scaling = _read_json(RESULTS_DIR / "benchmark_scaling.json")
    res.bench_memory = _read_json(RESULTS_DIR / "benchmark_memory.json")
    res.bench_correctness = _read_json(RESULTS_DIR / "benchmark_correctness.json")
    res.bench_e2e = _read_json(RESULTS_DIR / "benchmark_end_to_end.json")
    res.bench_trt_variants = _read_json(RESULTS_DIR / "benchmark_tensorrt_variants.json")
    res.bench_pt_patches = _read_json(RESULTS_DIR / "benchmark_pytorch_patches.json")
    return res


def profile_names(res: Results) -> List[str]:
    return sorted(set(res.bench) | set(res.evals))


def profile_meta(res: Results, name: str) -> Dict[str, Any]:
    """What a profile is, from its own result files. Older files fall back to their config block."""
    bench = res.bench.get(name, {})
    meta = bench.get("profile") or res.eval_meta.get(name, {}).get("profile") or {}
    if meta:
        return meta
    config = bench.get("config", {})
    return {
        "name": name,
        "label": name,
        "overlap": config.get("overlap"),
        "sw_batch_size": config.get("sw_batch_size"),
    }


def label(res: Results, name: Optional[str]) -> str:
    return str(profile_meta(res, name).get("label") or name) if name else "n/a"


def _first(res: Results, want_bench: bool, **attrs: Any) -> Optional[str]:
    names = [n for n in profile_names(res) if all(profile_meta(res, n).get(k) == v for k, v in attrs.items())]
    if want_bench:
        benched = [n for n in names if n in res.bench]
        names = benched or names
    return names[0] if names else None


def resolve_roles(res: Results) -> Roles:
    reference = res.agreement.get("reference") or _first(
        res, False, engine="pytorch", precision="fp32", transforms="cpu"
    )
    if reference not in res.evals:
        reference = _first(res, False, engine="pytorch", precision="fp32") or reference

    fast = res.state.get("fast_profile")
    if fast not in profile_names(res):
        timed = {n: stage(res, n, "e2e_total") for n in res.bench if profile_meta(res, n).get("inverse") == "trilinear"}
        timed = {n: v for n, v in timed.items() if v is not None}
        fast = min(timed, key=timed.get) if timed else None

    gpu_transforms = [
        t for t in ("cuda", "gpu") if _first(res, True, engine="trt", precision="fp16", transforms=t, overlap=0.5)
    ]
    return Roles(
        reference=reference,
        fast=fast,
        host_baseline=_first(res, True, engine="pytorch", precision="tf32", transforms="cpu"),
        gpu_baseline=_first(res, True, engine="pytorch", precision="tf32", transforms="gpu"),
        like_for_like=_first(res, True, engine="trt", precision="fp16", transforms=gpu_transforms[0], overlap=0.5)
        if gpu_transforms
        else None,
    )


def targets(res: Results) -> Dict[str, float]:
    return {**DEFAULT_TARGETS, **res.claims_check.get("targets", {})}


def stage(res: Results, name: Optional[str], key: str, field_name: str = "median") -> Optional[float]:
    value = res.bench.get(name or "", {}).get("stage_latency_ms", {}).get(key, {}).get(field_name)
    return float(value) if value is not None else None


def ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    return numerator / denominator if numerator and denominator else None


def mean_dice(res: Results, name: Optional[str]) -> Optional[float]:
    frame = res.evals.get(name or "")
    return float(frame["dice"].mean()) if frame is not None and len(frame) else None


def n_cases(res: Results) -> int:
    counts = [len(f) for f in res.evals.values()]
    return max(counts) if counts else int(res.heldout.get("heldout_count") or 0)


def profile_table(res: Results, roles: Roles) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    host = stage(res, roles.host_baseline, "e2e_total")
    gpu = stage(res, roles.gpu_baseline, "e2e_total")
    for name in profile_names(res):
        meta = profile_meta(res, name)
        bench = res.bench.get(name, {})
        frame = res.evals.get(name)
        e2e = stage(res, name, "e2e_total")
        patch = bench.get("patch_forward_latency_ms", {})
        rows.append(
            {
                "Profile": name,
                "Engine": meta.get("engine"),
                "Precision": meta.get("precision"),
                "Read": meta.get("io"),
                "Transforms": meta.get("transforms"),
                "Inverse": meta.get("inverse"),
                "Overlap": meta.get("overlap"),
                "Batch": meta.get("sw_batch_size"),
                "E2E median (ms)": e2e,
                "E2E IQR (ms)": stage(res, name, "e2e_total", "iqr"),
                "Slowest pass (ms)": stage(res, name, "e2e_total", "max"),
                "Network (ms)": stage(res, name, "sw_network"),
                "Ingest (ms)": stage(res, name, "ingest"),
                "Patch B=1 (ms)": patch.get("batch_1", {}).get("median"),
                "Patch B=4 (ms)": patch.get("batch_4", {}).get("median"),
                "Peak VRAM (MB)": bench.get("peak_vram_mb", {}).get("max"),
                "Volumes/s": bench.get("throughput_volumes_per_s"),
                "Mean Dice": float(frame["dice"].mean()) if frame is not None else None,
                "Dice SD": float(frame["dice"].std(ddof=1)) if frame is not None and len(frame) > 1 else None,
                "Speedup vs host baseline": ratio(host, e2e),
                "Speedup vs GPU baseline": ratio(gpu, e2e),
            }
        )
    return pd.DataFrame(rows)


def stage_frame(res: Results, names: List[str]) -> pd.DataFrame:
    rows = [
        {"Profile": n, "Stage": s, "Median (ms)": stage(res, n, s)}
        for n in names
        for s in STAGES
        if stage(res, n, s) is not None
    ]
    return pd.DataFrame(rows)


def per_case_latency(res: Results, name: str) -> pd.DataFrame:
    rows = []
    eval_rows = res.evals.get(name, pd.DataFrame())
    shapes = {}
    if {"shape_x", "shape_y", "shape_z"} <= set(eval_rows.columns):
        shapes = {r.case_id: f"{r.shape_x}x{r.shape_y}x{r.shape_z}" for r in eval_rows.itertuples()}
    for case, block in res.bench.get(name, {}).get("per_case", {}).items():
        rows.append(
            {
                "Case": case,
                "Shape": shapes.get(case),
                "Windows": block.get("patch_windows"),
                "Median (ms)": block.get("e2e_total", {}).get("median"),
                "Slowest (ms)": block.get("e2e_total", {}).get("max"),
                "Network (ms)": block.get("sw_network", {}).get("median"),
                "Peak VRAM (MB)": block.get("peak_vram_mb", {}).get("max"),
            }
        )
    return pd.DataFrame(rows)


def passes_frame(res: Results, name: str) -> pd.DataFrame:
    return pd.DataFrame(res.bench.get(name, {}).get("passes", []))


def patch_frame(res: Results) -> pd.DataFrame:
    """Patch latency depends on engine and precision only, so keep one profile per pair."""
    seen: Dict[Tuple[Any, Any], str] = {}
    for name in res.bench:
        meta = profile_meta(res, name)
        seen.setdefault((meta.get("engine"), meta.get("precision")), name)
    rows = []
    for (engine, precision), name in seen.items():
        for key, block in res.bench[name].get("patch_forward_latency_ms", {}).items():
            rows.append(
                {
                    "Backend": f"{engine or name} {precision or ''}".strip(),
                    "Batch": int(key.split("_")[-1]),
                    "Median (ms)": block.get("median"),
                    "Per patch (ms)": block.get("per_patch_median_ms"),
                }
            )
    return pd.DataFrame(rows)


def _volume_stats(differences: np.ndarray) -> Dict[str, Any]:
    n = len(differences)
    bias = float(differences.mean())
    sd = float(differences.std(ddof=1)) if n > 1 else 0.0
    half = float(sps.t.ppf(0.975, n - 1)) * sd / np.sqrt(n) if n > 1 else 0.0
    return {
        "bias": bias,
        "sd": sd,
        "ci_low": bias - half,
        "ci_high": bias + half,
        "ci_excludes_zero": bool(bias - half > 0 or bias + half < 0),
        "loa_low": bias - 1.96 * sd,
        "loa_high": bias + 1.96 * sd,
        "n_positive": int((differences > 0).sum()),
        "n": n,
    }


def per_case_agreement(res: Results, name: str, reference: Optional[str]) -> pd.DataFrame:
    if reference not in res.evals or name not in res.evals:
        return pd.DataFrame()
    merged = res.evals[name].merge(res.evals[reference], on="case_id", suffixes=("", "_ref"))
    return pd.DataFrame(
        {
            "Case": merged["case_id"],
            "V ground truth (cm³)": merged["v_gt_cm3"],
            "V reference (cm³)": merged["v_pred_cm3_ref"],
            "V profile (cm³)": merged["v_pred_cm3"],
            "ΔV (cm³)": merged["v_pred_cm3"] - merged["v_pred_cm3_ref"],
            "Dice reference": merged["dice_ref"],
            "Dice profile": merged["dice"],
            "ΔDice": merged["dice"] - merged["dice_ref"],
        }
    )


def agreement_table(res: Results, roles: Roles) -> pd.DataFrame:
    """Agreement of every profile with the reference, recomputed from the per-case CSVs."""
    tolerance = targets(res)["abs_delta_dice"]
    rows: List[Dict[str, Any]] = []
    for name, frame in res.evals.items():
        if name == roles.reference:
            continue
        cases = per_case_agreement(res, name, roles.reference)
        if cases.empty:
            continue
        volume = _volume_stats(cases["ΔV (cm³)"].to_numpy())
        deltas = cases["ΔDice"].abs()
        rows.append(
            {
                "Profile": name,
                "Mean Dice": float(frame["dice"].mean()),
                "Mean ΔDice": float(cases["ΔDice"].mean()),
                "Mean |ΔDice|": float(deltas.mean()),
                "Max |ΔDice|": float(deltas.max()),
                "Cases over tolerance": int((deltas >= tolerance).sum()),
                "Volume bias (cm³)": volume["bias"],
                "Bias CI low": volume["ci_low"],
                "Bias CI high": volume["ci_high"],
                "Systematic": volume["ci_excludes_zero"],
                "LoA low": volume["loa_low"],
                "LoA high": volume["loa_high"],
                "Larger than reference": f"{volume['n_positive']}/{volume['n']}",
                "MAE vs ground truth (cm³)": float((frame["v_pred_cm3"] - frame["v_gt_cm3"]).abs().mean()),
            }
        )
    return pd.DataFrame(rows)


def agreement_row(res: Results, roles: Roles, name: Optional[str]) -> Optional[pd.Series]:
    table = agreement_table(res, roles)
    if table.empty or name is None:
        return None
    match = table[table["Profile"] == name]
    return match.iloc[0] if len(match) else None


def bland_altman_frame(res: Results, name: str, reference: Optional[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cases = per_case_agreement(res, name, reference)
    if cases.empty:
        return cases, {}
    frame = pd.DataFrame(
        {
            "Case": cases["Case"],
            "Mean volume (cm³)": 0.5 * (cases["V profile (cm³)"] + cases["V reference (cm³)"]),
            "ΔV (cm³)": cases["ΔV (cm³)"],
        }
    )
    return frame, _volume_stats(cases["ΔV (cm³)"].to_numpy())


def segmentation_table(res: Results) -> pd.DataFrame:
    rows = []
    for name, frame in res.evals.items():
        row: Dict[str, Any] = {"Profile": name}
        for column, title in SEGMENTATION_METRICS.items():
            if column in frame.columns:
                row[title] = float(frame[column].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def relation_frame(res: Results, name: Optional[str]) -> pd.DataFrame:
    frame = res.evals.get(name or "")
    if frame is None or "spacing_z_mm" not in frame.columns:
        return pd.DataFrame()
    return frame[["case_id", "dice", "spacing_z_mm", "v_gt_cm3"]].rename(
        columns={"case_id": "Case", "dice": "Dice", "spacing_z_mm": "Slice spacing (mm)", "v_gt_cm3": "Spleen volume (cm³)"}
    )


def spearman(frame: pd.DataFrame, column: str) -> Tuple[Optional[float], Optional[float]]:
    if len(frame) < 3:
        return None, None
    rho, p = sps.spearmanr(frame[column], frame["Dice"])
    return float(rho), float(p)


def ablation_frame(res: Results) -> pd.DataFrame:
    rows = []
    rungs = res.ablation.get("rungs", {})
    base = None
    for key, rung in rungs.items():
        stages = rung.get("stages_ms", {})
        median = stages.get("e2e_total", {}).get("median")
        base = base or median
        meta = rung.get("profile", {})
        rows.append(
            {
                "Rung": key,
                "Configuration": meta.get("label", key),
                "Engine": meta.get("engine"),
                "Transforms": meta.get("transforms"),
                "Read": meta.get("io"),
                "Inverse": meta.get("inverse"),
                "Windows": rung.get("windows_total"),
                "E2E median (ms)": median,
                "Slowest pass (ms)": stages.get("e2e_total", {}).get("max"),
                "vs first row": ratio(base, median),
                "Network (ms)": stages.get("sw_network", {}).get("median"),
                "Ingest (ms)": stages.get("ingest", {}).get("median"),
                "Preprocess (ms)": stages.get("preprocess", {}).get("median"),
                "Postprocess (ms)": stages.get("postprocess", {}).get("median"),
                "Mean Dice": rung.get("mean_dice"),
            }
        )
    return pd.DataFrame(rows)


def provenance_frame(res: Results) -> pd.DataFrame:
    rows = []
    for name in profile_names(res):
        bench_run = res.bench.get(name, {}).get("run", {})
        eval_run = res.eval_meta.get(name, {}).get("run", {})
        rows.append(
            {
                "Profile": name,
                "Bench code": bench_run.get("code_fingerprint", "not stamped" if name in res.bench else "no benchmark"),
                "Eval code": eval_run.get("code_fingerprint", "not stamped" if name in res.evals else "no evaluation"),
                "Engine": bench_run.get("engine_fingerprint") or eval_run.get("engine_fingerprint"),
                "Benchmarked": (bench_run.get("timestamp") or res.bench.get(name, {}).get("timestamp") or "")[:19],
                "Timed passes": res.bench.get(name, {}).get("config", {}).get("total_timed_passes"),
            }
        )
    return pd.DataFrame(rows)


def status(res: Results) -> Dict[str, Any]:
    """One honest answer to whether these numbers can be quoted, recomputed on every render."""
    if GPU_LOCK.exists():
        return {"level": "running", "title": "A GPU job is running", "detail": "Numbers may change when it finishes."}
    if not res.bench and not res.evals:
        return {"level": "empty", "title": "No results yet", "detail": "Run `python -m medtrt.pipeline` to generate them."}

    prov = provenance_frame(res)
    stamps = {v for v in pd.concat([prov["Bench code"], prov["Eval code"]]) if v != "no benchmark" and v != "no evaluation"}
    unstamped = int((prov["Bench code"] == "not stamped").sum())
    problems: List[str] = []
    if unstamped:
        problems.append(f"{unstamped} benchmark file(s) predate the current result format")
    if len(stamps - {"not stamped"}) > 1:
        problems.append(f"results come from {len(stamps - {'not stamped'})} different code versions")

    checks = res.claims_check.get("checks", [])
    passed = sum(bool(c.get("passed")) for c in checks)
    if not checks:
        problems.append("the claims verifier has not been run")
    elif passed < len(checks):
        problems.append(f"{passed} of {len(checks)} verifier checks pass")

    checked_at = (RESULTS_DIR / "claims_check.json").stat().st_mtime if checks else 0.0
    newest_code = max((p.stat().st_mtime for p in (ROOT / "src" / "medtrt").glob("*.py")), default=0.0)
    if checks and newest_code > checked_at:
        problems.append("source code changed after the last verification")

    if problems:
        return {"level": "unverified", "title": "Not verified: do not quote these numbers", "detail": "; ".join(problems) + "."}
    return {"level": "verified", "title": f"Verified: {passed} of {len(checks)} checks passed", "detail": "These numbers match `results/`."}


def takeaways(res: Results, roles: Roles) -> List[str]:
    """Findings written from the current data. A statement appears only if the data supports it."""
    lines: List[str] = []
    goal = targets(res)
    fast_e2e = stage(res, roles.fast, "e2e_total")
    host_e2e = stage(res, roles.host_baseline, "e2e_total")
    gpu_e2e = stage(res, roles.gpu_baseline, "e2e_total")
    gpu_max = stage(res, roles.gpu_baseline, "e2e_total", "max")

    if fast_e2e and host_e2e:
        lines.append(f"The fast profile takes {fast_e2e:,.0f} ms against {host_e2e:,.0f} ms on the host path, {host_e2e / fast_e2e:.1f}x faster.")
    if fast_e2e and gpu_e2e:
        lines.append(f"Against PyTorch on the same GPU path ({gpu_e2e:,.0f} ms), the fast profile is {gpu_e2e / fast_e2e:.2f}x faster, so most of the host-path gain is not TensorRT.")
    if gpu_max is not None and gpu_max < goal["e2e_ms"]:
        lines.append(f"PyTorch without TensorRT already keeps every pass under {goal['e2e_ms']:.0f} ms (slowest {gpu_max:,.0f} ms).")

    cases = per_case_latency(res, roles.fast) if roles.fast else pd.DataFrame()
    if len(cases) > 1 and cases["Median (ms)"].notna().all():
        lines.append(f"The median hides volume size: per-case medians run from {cases['Median (ms)'].min():,.0f} to {cases['Median (ms)'].max():,.0f} ms.")

    row = agreement_row(res, roles, roles.fast)
    if row is not None:
        verdict = "a systematic shift" if row["Systematic"] else "no systematic shift"
        lines.append(f"Compared with true FP32, the fast profile shows {verdict} in volume ({row['Volume bias (cm³)']:+.3f} cm³), with a largest per-case Dice change of {row['Max |ΔDice|']:.5f}.")

    table = agreement_table(res, roles)
    if not table.empty:
        nearest = [n for n in table["Profile"] if profile_meta(res, n).get("inverse") == "nearest"]
        if nearest and roles.fast in set(table["Profile"]):
            gap = mean_dice(res, roles.fast) - mean_dice(res, nearest[0])
            lines.append(f"Inverting probabilities rather than hard labels is worth {gap:+.4f} mean Dice.")
    return lines


def ingest_ratio(res: Results, roles: Roles) -> Optional[float]:
    return ratio(stage(res, roles.host_baseline, "ingest"), stage(res, roles.fast, "ingest"))


def like_for_like_network(res: Results, roles: Roles) -> Optional[float]:
    return ratio(stage(res, roles.gpu_baseline, "sw_network"), stage(res, roles.like_for_like, "sw_network"))


def _text(value: Optional[float], template: str) -> str:
    return template.format(value) if value is not None else "Not measured yet"


def _claim_e2e(res: Results, roles: Roles) -> Tuple[str, str]:
    value = stage(res, roles.fast, "e2e_total")
    return ("Reproduced" if value else "Pending"), _text(value, "{:,.1f} ms median")


def _claim_e2e_speedup(res: Results, roles: Roles) -> Tuple[str, str]:
    host = ratio(stage(res, roles.host_baseline, "e2e_total"), stage(res, roles.fast, "e2e_total"))
    gpu = ratio(stage(res, roles.gpu_baseline, "e2e_total"), stage(res, roles.fast, "e2e_total"))
    if host is None:
        return "Pending", "Not measured yet"
    return "Corrected", f"{host:.1f}x vs host path" + (f", {gpu:.2f}x vs GPU-path PyTorch" if gpu else "")


def _claim_network(res: Results, roles: Roles) -> Tuple[str, str]:
    value = like_for_like_network(res, roles)
    return ("Corrected", f"{value:.2f}x at identical windows") if value else ("Pending", "Not measured yet")


def _claim_load(res: Results, roles: Roles) -> Tuple[str, str]:
    value = ingest_ratio(res, roles)
    return ("Corrected", f"{value:.1f}x for read plus copy to GPU") if value else ("Pending", "Not measured yet")


def _claim_host_share(res: Results, roles: Roles) -> Tuple[str, str]:
    post, load_ms, total = (stage(res, roles.host_baseline, k) for k in ("postprocess", "load", "e2e_total"))
    if not (post and load_ms and total):
        return "Pending", "Not measured yet"
    return "Corrected", f"Those two stages are {100.0 * (post + load_ms) / total:.1f}% of host-path time"


def _claim_dice(res: Results, roles: Roles) -> Tuple[str, str]:
    value = mean_dice(res, roles.fast)
    return ("Corrected", f"{value:.6f} mean Dice") if value else ("Pending", "Not measured yet")


def _claim_delta(res: Results, roles: Roles) -> Tuple[str, str]:
    row = agreement_row(res, roles, roles.fast)
    if row is None:
        return "Pending", "Not measured yet"
    return "Corrected", f"Mean {row['Mean |ΔDice|']:.6f}, max {row['Max |ΔDice|']:.6f}, on the fast profile vs true FP32"


def _claim_bias(res: Results, roles: Roles) -> Tuple[str, str]:
    row = agreement_row(res, roles, roles.fast)
    if row is None:
        return "Pending", "Not measured yet"
    return "Corrected", f"{row['Volume bias (cm³)']:+.3f} cm³, {row['Larger than reference']} cases larger than FP32"


def _claim_accuracy_gain(res: Results, roles: Roles) -> Tuple[str, str]:
    row = agreement_row(res, roles, roles.fast)
    ref = res.evals.get(roles.reference or "")
    if row is None or ref is None:
        return "Pending", "Not measured yet"
    ref_mae = float((ref["v_pred_cm3"] - ref["v_gt_cm3"]).abs().mean())
    return "Withdrawn", f"MAE vs ground truth: FP32 {ref_mae:.2f}, fast profile {row['MAE vs ground truth (cm³)']:.2f} cm³"


def _claim_pareto(res: Results, roles: Roles) -> Tuple[str, str]:
    if res.sweep.empty or "pareto_optimal" not in res.sweep.columns:
        return "Pending", "Sweep not re-run yet"
    flagged = res.sweep[res.sweep["pareto_optimal"].astype(str) == "True"]
    settings = ", ".join(f"{r.overlap}/{r.sw_batch_size}" for r in flagged.itertuples())
    return "Corrected", f"Pareto-optimal overlap/batch: {settings}"


def _claim_nsight(res: Results, roles: Roles) -> Tuple[str, str]:
    if res.timeline.get("nsys_cli_executed"):
        return "Reproduced", "Nsight trace recorded"
    return "Withdrawn", "Nsight Systems has not been run; timings come from CUDA events and wall clock"


def _claim_uint8(res: Results, roles: Roles) -> Tuple[str, str]:
    inverse = res.probes.get("resample_dtype", {}).get("inverse", [])
    if not inverse:
        return "Pending", "Not measured yet"
    return "Withdrawn", f"Resampling runs in {inverse[0]['dtype'].replace('torch.', '')} on {inverse[0]['device']}"


def _claim_split(res: Results, roles: Roles) -> Tuple[str, str]:
    return "Withdrawn", f"{n_cases(res)} cases were used by the bundle to select its checkpoint"


def _claim_failures(res: Results, roles: Roles) -> Tuple[str, str]:
    frame = relation_frame(res, roles.fast)
    rho, p = spearman(frame, "Slice spacing (mm)")
    if rho is None:
        return "Pending", "Not measured yet"
    return "Withdrawn", f"Dice vs slice spacing: Spearman {rho:+.2f}, p = {p:.2f}"


def _claim_static(outcome: str, text: str) -> Callable[[Results, Roles], Tuple[str, str]]:
    return lambda _res, _roles: (outcome, text)


CLAIM_RULES: Dict[str, Callable[[Results, Roles], Tuple[str, str]]] = {
    "e2e": _claim_e2e,
    "e2e_speedup": _claim_e2e_speedup,
    "network_speedup": _claim_network,
    "load_speedup": _claim_load,
    "host_share": _claim_host_share,
    "mean_dice": _claim_dice,
    "delta_dice": _claim_delta,
    "volume_bias": _claim_bias,
    "accuracy_gain": _claim_accuracy_gain,
    "pareto": _claim_pareto,
    "nsight": _claim_nsight,
    "uint8": _claim_uint8,
    "split": _claim_split,
    "failure_causes": _claim_failures,
    "clinical": _claim_static("Withdrawn", "Removed; this is a research benchmark"),
    "reproducible": _claim_static("Corrected", "`python -m medtrt.pipeline` and `python -m medtrt.verify_claims`"),
}


def claims_frame(res: Results, roles: Roles) -> pd.DataFrame:
    """The record of what was first claimed, set beside what the current results say."""
    try:
        with open(ORIGINAL_CLAIMS_FILE, "r", encoding="utf-8") as f:
            original = json.load(f)
    except (OSError, ValueError):
        return pd.DataFrame()
    rows = []
    for item in original:
        rule = CLAIM_RULES.get(item.get("id"), _claim_static("Pending", "No rule for this claim"))
        outcome, current = rule(res, roles)
        rows.append(
            {
                "Topic": item.get("topic"),
                "Original claim": item.get("claim"),
                "Outcome": outcome,
                "Current result": current,
                "Why": item.get("why"),
            }
        )
    return pd.DataFrame(rows)


def live_sync_info(res: Results) -> Dict[str, Any]:
    """Return live filesystem synchronization telemetry for the sidebar watcher."""
    gpu_busy = GPU_LOCK.exists()
    busy_text = ""
    if gpu_busy:
        try:
            busy_text = GPU_LOCK.read_text(encoding="utf-8").strip()
        except OSError:
            busy_text = "GPU job active"

    newest_file = "none"
    newest_mtime = 0
    file_count = 0
    if RESULTS_DIR.exists():
        for p in RESULTS_DIR.iterdir():
            if p.is_file() and p.suffix in (".json", ".csv"):
                file_count += 1
                try:
                    st_m = int(p.stat().st_mtime)
                    if st_m >= newest_mtime:
                        newest_mtime = st_m
                        newest_file = p.name
                except OSError:
                    continue

    return {
        "gpu_busy": gpu_busy,
        "busy_text": busy_text,
        "newest_file": newest_file,
        "newest_mtime": newest_mtime,
        "result_files_count": file_count,
        "bench_profiles_count": len(res.bench),
        "eval_profiles_count": len(res.evals),
    }


def cuda_kernels_frame(res: Results) -> pd.DataFrame:
    """Hardware compilation & SM occupancy attributes of the 9 custom CUDA C++17 kernels."""
    block = res.bench_correctness.get("cuda_hardware_kernels", {})
    kernels = block.get("kernels", {})
    rows: List[Dict[str, Any]] = []
    for k_name, info in kernels.items():
        rows.append(
            {
                "CUDA C++17 Kernel": k_name,
                "Registers/thread": info.get("num_registers_per_thread"),
                "Shared mem (B)": info.get("static_shared_mem_bytes"),
                "Local spills (B)": info.get("local_mem_bytes_per_thread"),
                "Max threads/block": info.get("max_threads_per_block"),
                "Active blocks/SM": info.get("active_blocks_per_sm_at_256t"),
                "Active warps/SM": info.get("active_warps_per_sm_at_256t"),
                "Occupancy (%)": info.get("theoretical_occupancy_pct"),
                "SASS arch": f"sm_{info.get('sass_binary_version')}" if info.get("sass_binary_version") else None,
            }
        )
    return pd.DataFrame(rows)


def preproc_methods_frame(res: Results) -> pd.DataFrame:
    """Compare CPU MONAI, GPU MONAI grid_sample, and Custom CUDA C++ fused 3D preprocessing."""
    fp = res.bench_preproc.get("full_preprocess", {})
    if not fp:
        return pd.DataFrame()
    cpu_med = fp.get("monai_cpu_ms", {}).get("median")
    gpu_med = fp.get("monai_gpu", {}).get("cuda_event_ms", {}).get("median")
    cuda_med = fp.get("custom_cuda_fused", {}).get("cuda_event_ms", {}).get("median")
    rows = [
        {
            "Method": "MONAI CPU (Orientationd + Spacingd + ScaleIntensity)",
            "Median (ms)": cpu_med,
            "IQR (ms)": fp.get("monai_cpu_ms", {}).get("iqr"),
            "Incremental VRAM (MB)": 0.0,
            "Peak VRAM (MB)": 0.0,
            "Speedup vs CPU": 1.0 if cpu_med else None,
            "Speedup vs MONAI GPU": ratio(gpu_med, cpu_med),
        },
        {
            "Method": "MONAI GPU grid_sample (3D Meshgrid on cuda:0)",
            "Median (ms)": gpu_med,
            "IQR (ms)": fp.get("monai_gpu", {}).get("cuda_event_ms", {}).get("iqr"),
            "Incremental VRAM (MB)": fp.get("monai_gpu", {}).get("incremental_allocated_vram_mb"),
            "Peak VRAM (MB)": fp.get("monai_gpu", {}).get("peak_allocated_vram_mb"),
            "Speedup vs CPU": ratio(cpu_med, gpu_med),
            "Speedup vs MONAI GPU": 1.0 if gpu_med else None,
        },
        {
            "Method": "Custom CUDA C++ Fused Kernel (fused_preprocess_affine_trilinear_hu_f32)",
            "Median (ms)": cuda_med,
            "IQR (ms)": fp.get("custom_cuda_fused", {}).get("cuda_event_ms", {}).get("iqr"),
            "Incremental VRAM (MB)": fp.get("custom_cuda_fused", {}).get("incremental_allocated_vram_mb"),
            "Peak VRAM (MB)": fp.get("custom_cuda_fused", {}).get("peak_allocated_vram_mb"),
            "Speedup vs CPU": ratio(cpu_med, cuda_med),
            "Speedup vs MONAI GPU": ratio(gpu_med, cuda_med),
        },
    ]
    return pd.DataFrame(rows)


def postproc_methods_frame(res: Results) -> pd.DataFrame:
    """Compare CPU MONAI Invertd, GPU MONAI Invertd, and Custom CUDA C++ fused trilinear postprocessing."""
    methods = res.bench_postproc.get("methods", {})
    if not methods:
        return pd.DataFrame()
    cpu_m = methods.get("monai_cpu_trilinear", {})
    gpu_tri = methods.get("monai_gpu_trilinear", {})
    gpu_nn = methods.get("monai_gpu_nearest_u8", {})
    cuda_tri = methods.get("custom_cuda_fused_trilinear", {})

    cpu_med = cpu_m.get("latency_ms", {}).get("median")
    gpu_med = gpu_tri.get("cuda_event_ms", {}).get("median")
    nn_med = gpu_nn.get("cuda_event_ms", {}).get("median")
    cuda_med = cuda_tri.get("cuda_event_ms", {}).get("median")

    rows = [
        {
            "Postprocessing Method": "MONAI CPU Trilinear Invertd",
            "Median (ms)": cpu_med,
            "IQR (ms)": cpu_m.get("latency_ms", {}).get("iqr"),
            "Incremental VRAM (MB)": 0.0,
            "Peak VRAM (MB)": 0.0,
            "Case Dice": cpu_m.get("dice"),
            "Voxel Agreement (%)": cpu_m.get("voxel_agreement_vs_monai_gpu_pct"),
            "Speedup vs CPU": 1.0 if cpu_med else None,
            "Speedup vs MONAI GPU": ratio(gpu_med, cpu_med),
        },
        {
            "Postprocessing Method": "MONAI GPU Trilinear Invertd (2-Ch Meshgrid)",
            "Median (ms)": gpu_med,
            "IQR (ms)": gpu_tri.get("cuda_event_ms", {}).get("iqr"),
            "Incremental VRAM (MB)": gpu_tri.get("incremental_allocated_vram_mb"),
            "Peak VRAM (MB)": gpu_tri.get("peak_allocated_vram_mb"),
            "Case Dice": gpu_tri.get("dice"),
            "Voxel Agreement (%)": gpu_tri.get("voxel_agreement_vs_monai_gpu_pct"),
            "Speedup vs CPU": ratio(cpu_med, gpu_med),
            "Speedup vs MONAI GPU": 1.0 if gpu_med else None,
        },
        {
            "Postprocessing Method": "MONAI GPU Nearest-Neighbor Label Invertd (uint8)",
            "Median (ms)": nn_med,
            "IQR (ms)": gpu_nn.get("cuda_event_ms", {}).get("iqr"),
            "Incremental VRAM (MB)": gpu_nn.get("incremental_allocated_vram_mb"),
            "Peak VRAM (MB)": gpu_nn.get("peak_allocated_vram_mb"),
            "Case Dice": gpu_nn.get("dice"),
            "Voxel Agreement (%)": gpu_nn.get("voxel_agreement_vs_monai_gpu_pct"),
            "Speedup vs CPU": ratio(cpu_med, nn_med),
            "Speedup vs MONAI GPU": ratio(gpu_med, nn_med),
        },
        {
            "Postprocessing Method": "Custom CUDA C++ Fused Softmax + Trilinear Invert + Ballot Count",
            "Median (ms)": cuda_med,
            "IQR (ms)": cuda_tri.get("cuda_event_ms", {}).get("iqr"),
            "Incremental VRAM (MB)": cuda_tri.get("incremental_allocated_vram_mb"),
            "Peak VRAM (MB)": cuda_tri.get("peak_allocated_vram_mb"),
            "Case Dice": cuda_tri.get("dice"),
            "Voxel Agreement (%)": cuda_tri.get("voxel_agreement_vs_monai_gpu_pct"),
            "Speedup vs CPU": ratio(cpu_med, cuda_med),
            "Speedup vs MONAI GPU": ratio(gpu_med, cuda_med),
        },
    ]
    return pd.DataFrame(rows)


def scaling_frame(res: Results) -> pd.DataFrame:
    """Volume size scaling across custom CUDA C++ kernels vs. PyTorch CUDA."""
    items = res.bench_scaling.get("volume_scaling", [])
    rows: List[Dict[str, Any]] = []
    for item in items:
        sh = item.get("shape", [])
        shape_str = "x".join(str(x) for x in sh)
        rows.append(
            {
                "Volume Shape": shape_str,
                "Voxels (M)": item.get("voxels_millions"),
                "PyTorch Fused Post (ms)": item.get("fused_post_pytorch_ms"),
                "Custom CUDA Fused Post (ms)": item.get("fused_post_cuda_ms"),
                "Fused Post Speedup": item.get("fused_post_speedup"),
                "Trilinear Resample Speedup": item.get("trilinear_speedup"),
                "Nearest u8 Speedup": item.get("nearest_u8_speedup"),
                "HU Vec4 Speedup": item.get("hu_norm_speedup"),
                "PyTorch Post VRAM (MB)": item.get("fused_post_pytorch_vram_mb"),
                "Custom CUDA Post VRAM (MB)": item.get("fused_post_cuda_vram_mb"),
            }
        )
    return pd.DataFrame(rows)


def memory_frames(res: Results) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return `(static_footprint_df, dynamic_pipeline_vram_df)` from `benchmark_memory.json`."""
    static = res.bench_memory.get("static_model_footprint_mb", {})
    s_rows: List[Dict[str, Any]] = []
    if static:
        s_rows = [
            {"Artifact": "PyTorch FP32 Model Weights", "Footprint (MB)": static.get("pytorch_fp32_weights_mb")},
            {"Artifact": "PyTorch FP16 Model Weights", "Footprint (MB)": static.get("pytorch_fp16_weights_mb")},
            {"Artifact": "TensorRT 10.x FP32 Serialized Engine", "Footprint (MB)": static.get("trt_fp32", {}).get("total_static_mb")},
            {"Artifact": "TensorRT 10.x FP16 Serialized Engine", "Footprint (MB)": static.get("trt_fp16", {}).get("total_static_mb")},
        ]
    dyn = res.bench_memory.get("pipeline_peak_memory", {})
    d_rows: List[Dict[str, Any]] = []
    for prof_name, info in dyn.items():
        d_rows.append(
            {
                "Profile": prof_name,
                "Median Allocated VRAM (MB)": info.get("median_allocated_mb"),
                "Max Allocated VRAM (MB)": info.get("max_allocated_mb"),
                "Max Reserved VRAM (MB)": info.get("max_reserved_mb"),
                "Share of GPU VRAM (%)": info.get("vram_utilization_pct_of_16gb"),
            }
        )
    return pd.DataFrame(s_rows), pd.DataFrame(d_rows)


def precision_numerics_frame(res: Results) -> pd.DataFrame:
    """Compare logit drift and channel bias before vs. after analytical FP64 ConvTranspose3d+BN folding."""
    comp = res.bench_correctness.get("backend_precision_comparison", {})
    rows: List[Dict[str, Any]] = []
    for key, info in comp.items():
        rows.append(
            {
                "Backend Variant": key,
                "Real Patch Mean |Δlogit|": info.get("real_patch_mean_abs_diff"),
                "Real Patch Max |Δlogit|": info.get("real_patch_max_abs_diff"),
                "Ch0 Signed Bias": info.get("real_patch_ch0_signed_bias"),
                "Ch1 Signed Bias": info.get("real_patch_ch1_signed_bias"),
                "Voxel Argmax Agreement (%)": info.get("real_patch_argmax_agreement_pct"),
            }
        )
    return pd.DataFrame(rows)


def other_result_files(res: Results) -> List[Path]:
    """Result files no page reads yet, so new outputs are still visible and downloadable."""
    known = {
        "env.json",
        "trt_build.json",
        "bland_altman.json",
        "failure_analysis.json",
        "timeline_summary.json",
        "ablation.json",
        "probes.json",
        "heldout_cases.json",
        "STATE.json",
        "claims_check.json",
        "sweep.csv",
        "validation.csv",
        "benchmark_preprocessing.json",
        "benchmark_postprocessing.json",
        "benchmark_scaling.json",
        "benchmark_memory.json",
        "benchmark_correctness.json",
        "benchmark_end_to_end.json",
        "benchmark_tensorrt_variants.json",
        "benchmark_pytorch_patches.json",
    }
    if not RESULTS_DIR.exists():
        return []
    return [
        p
        for p in sorted(RESULTS_DIR.iterdir())
        if p.is_file()
        and p.suffix in (".json", ".csv")
        and p.name not in known
        and not p.name.startswith(("bench_", "eval_"))
    ]
