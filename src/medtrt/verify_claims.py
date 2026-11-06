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
from medtrt.profiles import LIKE_FOR_LIKE, PROFILES, REFERENCE, SHIPPING

TARGET_E2E_MS = 500.0
TARGET_MEAN_DICE = 0.960
TARGET_ABS_DELTA_DICE = 0.005
EXPECTED_PASSES = 90

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
    short = [n for n in PROFILES if _json(f"bench_{n}.json")["config"]["total_timed_passes"] < 18]
    if short:
        return False, f"fewer than 18 timed passes: {short}"
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
