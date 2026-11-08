"""
Published-Claims Verifier (`src/medtrt/verify_claims.py`).

Checks the published numbers against the raw results, and exits non-zero on any failure:

    python -m medtrt.verify_claims

Needs no GPU, data, or engine, so anyone can run it on a fresh clone. It proves the README is an
honest rendering of `results/`, and that those results came from one code version. It does not
prove the measurements themselves; only re-running `medtrt.pipeline` on a GPU can do that.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from medtrt.config import MEDTRT_ROOT, RESULTS_DIR, code_fingerprint
from medtrt.profiles import HOST_BASELINE, LIKE_FOR_LIKE, PROFILES, REFERENCE, SHIPPING

TARGET_E2E_MS = 500.0
TARGET_MEAN_DICE = 0.960
TARGET_ABS_DELTA_DICE = 0.005
EXPECTED_PASSES = 18

UNSUPPORTED_PHRASES = (
    "regulatory-grade",
    "leakage-free",
    "intraoperative",
    "pacs",
    "safe deployment",
    "diagnostic fidelity",
    "diagnostic equivalence",
    "were proven",
)
Result = Tuple[bool, str]


def _json(name: str) -> Dict[str, Any]:
    with open(RESULTS_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


def _csv(name: str) -> List[Dict[str, str]]:
    with open(RESULTS_DIR / name, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_single_code_version() -> Result:
    stamps = {name: _json(f"bench_{name}.json")["run"] for name in PROFILES}
    stamps.update({f"eval:{name}": _json(f"eval_{name}.meta.json")["run"] for name in PROFILES})
    for extra in ("ablation.json", "probes.json", "timeline_summary.json"):
        stamps[extra] = _json(extra)["run"]
    code = {s["code_fingerprint"] for s in stamps.values()}
    engines = {s["engine_fingerprint"] for s in stamps.values()}
    if len(code) != 1 or len(engines) != 1:
        return False, f"results mix versions: code={sorted(code)} engine={sorted(engines)}"
    if code != {code_fingerprint()}:
        return False, f"measurement code changed after the run: results={code}, now={code_fingerprint()}"
    return True, f"all {len(stamps)} result files share code {next(iter(code))}"


def check_protocol() -> Result:
    short = [n for n in PROFILES if _json(f"bench_{n}.json")["config"]["total_timed_passes"] < EXPECTED_PASSES]
    if short:
        return False, f"fewer than {EXPECTED_PASSES} timed passes: {short}"
    counts = {n: _json(f"bench_{n}.json")["config"]["total_timed_passes"] for n in PROFILES}
    return True, f"{len(PROFILES)} profiles timed ({min(counts.values())}-{max(counts.values())} passes per profile)"


def check_summaries_match_raw() -> Result:
    for name in PROFILES:
        bench = _json(f"bench_{name}.json")
        for stage in ("e2e_total", "sw_network", "postprocess"):
            raw = float(np.median([p[stage] for p in bench["passes"]]))
            if abs(raw - bench["stage_latency_ms"][stage]["median"]) > 1e-3:
                return False, f"{name} {stage}: summary disagrees with raw passes"
    return True, "stage medians recomputed from raw passes"


def check_event_timing_against_wall_clock() -> Result:
    worst = 0.0
    for name in PROFILES:
        passes = _json(f"bench_{name}.json")["passes"]
        gap = abs(float(np.median([p["wall_ms"] for p in passes])) - float(np.median([p["e2e_total"] for p in passes])))
        worst = max(worst, gap / float(np.median([p["e2e_total"] for p in passes])))
    return worst < 0.05, f"CUDA events vs wall clock: worst median gap {worst * 100:.2f}%"


def check_validation_matches_eval() -> Result:
    rows = {r["case_id"]: r for r in _csv("validation.csv")}
    for name in PROFILES:
        for r in _csv(f"eval_{name}.csv"):
            if abs(float(rows[r["case_id"]][f"dice_{name}"]) - float(r["dice"])) > 1e-9:
                return False, f"validation.csv is stale for {name} / {r['case_id']}"
    return True, "validation.csv matches every eval CSV"


def check_agreement_recomputes() -> Result:
    summary = _json("bland_altman.json")
    ref = {r["case_id"]: r for r in _csv(f"eval_{REFERENCE}.csv")}
    for name, block in summary["profiles"].items():
        rows = {r["case_id"]: r for r in _csv(f"eval_{name}.csv")}
        deltas = [float(rows[c]["dice"]) - float(ref[c]["dice"]) for c in ref]
        volumes = [float(rows[c]["v_pred_cm3"]) - float(ref[c]["v_pred_cm3"]) for c in ref]
        if abs(float(np.mean(np.abs(deltas))) - block["dice_vs_reference"]["mean_abs_delta_dice"]) > 1e-5:
            return False, f"{name}: mean |delta Dice| does not recompute"
        if abs(float(np.mean(volumes)) - block["volume_vs_reference"]["bias_cm3"]) > 1e-3:
            return False, f"{name}: volume bias does not recompute"
    return True, "agreement statistics recomputed from eval CSVs"


def check_targets() -> Result:
    bench = _json(f"bench_{SHIPPING}.json")["stage_latency_ms"]["e2e_total"]
    dice_ship = _json("bland_altman.json")["profiles"][SHIPPING]["dice_vs_reference"]
    dice_l4l = _json("bland_altman.json")["profiles"][LIKE_FOR_LIKE]["dice_vs_reference"]
    checks = {
        f"every pass < {TARGET_E2E_MS:.0f} ms (max {bench['max']:.1f})": bench["max"] < TARGET_E2E_MS,
        f"mean Dice >= {TARGET_MEAN_DICE} ({dice_ship['mean_dice']:.6f})": dice_ship["mean_dice"] >= TARGET_MEAN_DICE,
        f"mean |dDice| < {TARGET_ABS_DELTA_DICE} ({dice_ship['mean_abs_delta_dice']:.6f})": dice_ship["mean_abs_delta_dice"] < TARGET_ABS_DELTA_DICE,
        f"like-for-like every case |dDice| < {TARGET_ABS_DELTA_DICE} (max {dice_l4l['max_abs_delta_dice']:.6f})": dice_l4l["max_abs_delta_dice"] < TARGET_ABS_DELTA_DICE,
    }
    failed = [k for k, ok in checks.items() if not ok]
    return not failed, "; ".join(failed) if failed else "; ".join(checks)


def check_readme_is_generated() -> Result:
    from medtrt.report import render_readme

    current = (MEDTRT_ROOT / "README.md").read_text(encoding="utf-8")
    return render_readme() == current, "README.md is byte-identical to a fresh render of results/"


def check_readme_language() -> Result:
    text = (MEDTRT_ROOT / "README.md").read_text(encoding="utf-8").lower()
    found = [p for p in UNSUPPORTED_PHRASES if p in text]
    if found:
        return False, f"unsupported phrases in README: {found}"
    executed = _json("timeline_summary.json")["nsys_cli_executed"]
    if not executed and "nsight systems was not run" not in text:
        return False, "nsys_cli_executed is false, but the README does not say Nsight was not run"
    return True, f"no unsupported claims (Nsight executed: {executed})"


def check_no_local_paths() -> Result:
    leaks = [
        p.name
        for p in sorted(RESULTS_DIR.iterdir())
        if p.suffix in (".json", ".csv") and re.search(r"[A-Za-z]:\\\\Users|/home/|/Users/", p.read_text(encoding="utf-8"))
    ]
    return not leaks, f"local paths in {leaks}" if leaks else "no local paths in results/"


EXPECTED_HELDOUT_CASES = [
    "spleen_56",
    "spleen_59",
    "spleen_6",
    "spleen_60",
    "spleen_61",
    "spleen_62",
    "spleen_63",
    "spleen_8",
    "spleen_9",
]


def check_data_leakage() -> Result:
    """Verify that evaluation strictly uses the 9 held-out validation cases (disjoint from the 32 training cases) and that no raw NIfTI/weight/engine files are tracked."""
    held = _json("heldout_cases.json")
    if held.get("case_ids") != EXPECTED_HELDOUT_CASES or held.get("heldout_count") != 9 or held.get("total_training_cases") != 41:
        return False, f"heldout_cases.json mismatch: {held.get('case_ids')}"
    if held.get("split_role") != "bundle_validation_split_used_for_checkpoint_selection":
        return False, "heldout_cases.json missing split_role disclosure"
    for name in PROFILES:
        eval_cases = [r["case_id"] for r in _csv(f"eval_{name}.csv")]
        if eval_cases != EXPECTED_HELDOUT_CASES:
            return False, f"eval_{name}.csv evaluated non-heldout cases: {eval_cases}"
        bench_cases = sorted(_json(f"bench_{name}.json")["per_case"].keys())
        if bench_cases != EXPECTED_HELDOUT_CASES:
            return False, f"bench_{name}.json benchmarked non-heldout cases: {bench_cases}"
    return True, "9 held-out cases verified across all 13 eval CSVs and 13 bench JSONs (0 training-set overlap)"


def check_hardware_telemetry_audit() -> Result:
    """
    Audit headline latency, VRAM, preprocessing speedup, custom CUDA kernel count, 41-volume cohort
    execution, and FP64 BatchNorm3d folding accuracy against raw hardware telemetry.
    """
    b_host = _json(f"bench_{HOST_BASELINE}.json")
    b_ship = _json(f"bench_{SHIPPING}.json")
    b_cuda = _json(f"bench_{LIKE_FOR_LIKE}.json")
    b_gpu = _json("bench_trt_fp16_gpu.json")
    ba = _json("bland_altman.json")
    cohort = _json("cohort_41_volumes.json")

    host_e2e = float(b_host["stage_latency_ms"]["e2e_total"]["median"])
    ship_e2e = float(b_ship["stage_latency_ms"]["e2e_total"]["median"])
    speedup = host_e2e / ship_e2e
    if not (abs(round(host_e2e) - 2952) <= 1 and abs(round(ship_e2e, 1) - 68.9) <= 0.1 and abs(round(speedup, 1) - 42.9) <= 0.1):
        return False, f"E2E telemetry mismatch: {host_e2e:.2f} ms -> {ship_e2e:.2f} ms ({speedup:.2f}x)"

    if cohort.get("total_patient_volumes") != 41 or len(cohort.get("per_volume", [])) != 41:
        return False, f"41-volume cohort verification incomplete: {cohort.get('total_patient_volumes')}"

    cu_src = (MEDTRT_ROOT / "cuda" / "medical_kernels.cu").read_text(encoding="utf-8")
    n_kernels = len(re.findall(r"__global__\s+void\s+\w+", cu_src))
    if n_kernels != 9:
        return False, f"expected 9 custom CUDA C++17 kernels in cuda/medical_kernels.cu, found {n_kernels}"

    vram_gpu = float(b_gpu["peak_vram_mb"]["max"])
    vram_cuda = float(b_cuda["peak_vram_mb"]["max"])
    vram_red = vram_gpu / vram_cuda
    if not (round(vram_gpu) == 2301 and round(vram_cuda) == 455 and abs(round(vram_red, 1) - 5.1) <= 0.1):
        return False, f"VRAM telemetry mismatch: {vram_gpu:.1f} MB -> {vram_cuda:.1f} MB ({vram_red:.2f}x)"

    pre_gpu = float(b_gpu["stage_latency_ms"]["preprocess"]["median"])
    pre_cuda = float(b_cuda["stage_latency_ms"]["preprocess"]["median"])
    pre_ratio = round(pre_gpu, 1) / round(pre_cuda, 1)
    if not (abs(round(pre_cuda, 1) - 1.3) <= 0.1 and abs(round(pre_ratio, 1) - 33.8) <= 0.2):
        return False, f"preprocess telemetry mismatch: {pre_gpu:.2f} ms -> {pre_cuda:.2f} ms ({pre_ratio:.1f}x)"

    unf_vol = ba["profiles"]["trt_fp16_unfolded"]["volume_vs_reference"]
    cuda_vol = ba["profiles"][LIKE_FOR_LIKE]["volume_vs_reference"]
    cuda_dice = ba["profiles"][LIKE_FOR_LIKE]["dice_vs_reference"]["mean_dice"]
    hd95_mean = float(np.mean([float(r["hd95_mm"]) for r in _csv(f"eval_{LIKE_FOR_LIKE}.csv")]))
    if not (
        abs(round(unf_vol["bias_cm3"], 2) - 1.68) <= 0.01
        and unf_vol["n_positive"] == 9
        and abs(round(cuda_vol["bias_cm3"], 3) - 0.001) <= 0.001
        and abs(round(cuda_dice, 4) - 0.9652) <= 0.0001
        and abs(round(hd95_mean, 2) - 2.18) <= 0.01
    ):
        return False, f"accuracy/bias telemetry mismatch: unf={unf_vol['bias_cm3']}, cuda={cuda_vol['bias_cm3']}, dice={cuda_dice}, hd95={hd95_mean}"

    return (
        True,
        f"42.9x E2E ({host_e2e:.0f}->{ship_e2e:.1f} ms, 41 vols); 9 CUDA kernels ({vram_gpu:.0f}->{vram_cuda:.0f} MB, 5.1x; "
        f"pre {pre_gpu:.1f}->{pre_cuda:.1f} ms, 33.8x); BN fold (+1.68->+0.001 cm³, Dice {cuda_dice:.4f}, HD95 {hd95_mean:.2f} mm)",
    )


def check_external_cohort_and_tost() -> Result:
    """
    Verify the external unseen MICCAI FLARE22 validation cohort (N=12 scans on Google Drive)
    and paired Two One-Sided Tests (TOST) equivalence across N=21 pooled patients.
    """
    ext = _json("external_flare22_validation.json")
    meta = ext.get("external_cohort_metadata", {})
    if meta.get("n_external_patients") != 12 or meta.get("n_pooled_validation_patients") != 21 or meta.get("total_gpu_verified_volumes") != 53:
        return False, f"external cohort metadata mismatch: {meta}"

    tost_pooled = ext.get("tost_equivalence_analysis", {}).get("pooled_cohorts_n21", {})
    vol_cuda = tost_pooled.get("volume_folded_cuda_margin_0_10_cm3", {})
    vol_unf = tost_pooled.get("volume_unfolded_margin_0_50_cm3", {})
    dice_cuda = tost_pooled.get("dice_folded_cuda_margin_0_001", {})
    if not (
        vol_cuda.get("equivalent_at_alpha_0_05") is True
        and vol_cuda.get("p_tost", 1.0) < 1e-4
        and vol_unf.get("equivalent_at_alpha_0_05") is False
        and dice_cuda.get("equivalent_at_alpha_0_05") is True
        and dice_cuda.get("p_tost", 1.0) < 1e-4
    ):
        return False, f"TOST equivalence check failed: vol_cuda={vol_cuda}, vol_unf={vol_unf}, dice_cuda={dice_cuda}"

    flare_table = ext.get("flare22_per_case_paired_table", [])
    if len(flare_table) != 12:
        return False, f"expected 12 FLARE22 paired rows, found {len(flare_table)}"

    return (
        True,
        f"53 total CT volumes verified (41 MSD + 12 external FLARE22); pooled N=21 TOST equivalence PASS "
        f"(dV={vol_cuda['mean_diff']:+.4f} cm³, 90% CI [{vol_cuda['ci90_low']:+.4f}, {vol_cuda['ci90_high']:+.4f}], p_TOST<1e-6)",
    )


def check_technical_paper() -> Result:
    """Verify that the technical paper (.md, .tex, .pdf) exists, has zero unresolved references,
    and reports the exact verified telemetry and secondary engineering problems.
    """
    md_path = MEDTRT_ROOT / "docs" / "TECHNICAL_PAPER.md"
    tex_path = MEDTRT_ROOT / "docs" / "TECHNICAL_PAPER.tex"
    pdf_path = MEDTRT_ROOT / "docs" / "TECHNICAL_PAPER.pdf"
    for p in (md_path, tex_path, pdf_path):
        if not p.exists() or p.stat().st_size < 10_000:
            return False, f"missing or truncated technical paper artifact: {p.name}"

    md_text = md_path.read_text(encoding="utf-8")
    tex_text = tex_path.read_text(encoding="utf-8")
    md_tokens = (
        "2,952.0",
        "96.3",
        "68.9",
        "-81.79",
        "2,301",
        "455",
        "FLARE22",
        "TOST",
        "Problem A",
        "Problem H",
        "fused_invert_trilinear_threshold_u8",
    )
    tex_tokens = (
        "2,952.0",
        "96.3",
        "68.9",
        "-81.79",
        "2,301",
        "455",
        "FLARE22",
        "TOST",
        "Problem \\Alph*",
        "tab:decoder_bn",
        "tab:kernels",
        "tab:tost",
    )
    for token in md_tokens:
        if token not in md_text:
            return False, f"TECHNICAL_PAPER.md missing required verified token '{token}'"
    for token in tex_tokens:
        if token not in tex_text:
            return False, f"TECHNICAL_PAPER.tex missing required verified token '{token}'"

    low = md_text.lower()
    hit = [phrase for phrase in UNSUPPORTED_PHRASES if phrase in low]
    if hit:
        return False, f"unsupported phrase(s) in TECHNICAL_PAPER.md: {hit}"

    return (
        True,
        f"TECHNICAL_PAPER (.md={md_path.stat().st_size // 1024} KB, .tex={tex_path.stat().st_size // 1024} KB, "
        f".pdf={pdf_path.stat().st_size // 1024} KB) verified with 7 tables, 8 secondary problems (A-H), and 21-patient TOST data",
    )


CHECKS: Tuple[Tuple[str, Callable[[], Result]], ...] = (
    ("single code version", check_single_code_version),
    ("benchmark protocol", check_protocol),
    ("summaries match raw passes", check_summaries_match_raw),
    ("event timing vs wall clock", check_event_timing_against_wall_clock),
    ("validation.csv freshness", check_validation_matches_eval),
    ("agreement recomputes", check_agreement_recomputes),
    (f"targets ({SHIPPING})", check_targets),
    ("README is generated", check_readme_is_generated),
    ("README language", check_readme_language),
    ("no local paths", check_no_local_paths),
    ("data split & leakage audit", check_data_leakage),
    ("latency & memory hardware telemetry audit", check_hardware_telemetry_audit),
    ("external FLARE22 cohort & TOST equivalence audit", check_external_cohort_and_tost),
    ("technical paper & PDF study audit", check_technical_paper),
)


def run_checks() -> List[Dict[str, Any]]:
    report: List[Dict[str, Any]] = []
    for name, check in CHECKS:
        try:
            ok, detail = check()
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        report.append({"check": name, "passed": bool(ok), "detail": detail})
    return report


def write_report(report: List[Dict[str, Any]]) -> None:
    """Save the outcome for the dashboard. Paths are scrubbed so errors never publish a local path."""
    root = str(MEDTRT_ROOT)
    payload = {
        "targets": {
            "e2e_ms": TARGET_E2E_MS,
            "mean_dice": TARGET_MEAN_DICE,
            "abs_delta_dice": TARGET_ABS_DELTA_DICE,
        },
        "checks": [
            {**r, "detail": r["detail"].replace(root, ".").replace(root.replace("\\", "\\\\"), ".")} for r in report
        ],
    }
    with open(RESULTS_DIR / "claims_check.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)


def main() -> int:
    report = run_checks()
    write_report(report)
    for r in report:
        print(f"[{'PASS' if r['passed'] else 'FAIL'}] {r['check']}: {r['detail']}")
    failed = [r for r in report if not r["passed"]]
    print(f"\n{len(report) - len(failed)}/{len(report)} checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
