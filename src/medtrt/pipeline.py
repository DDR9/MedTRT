"""
End-to-End Pipeline Runner (`src/medtrt/pipeline.py`).

Regenerates every result from a clean checkout, on any OS:

    python -m medtrt.pipeline            # full run, roughly 1 hour on a laptop GPU
    python -m medtrt.pipeline --smoke    # 1 case, 1 repeat, written to a temporary directory

Each step runs in its own process, so precision flags and GPU memory cannot leak between profiles.
Holds `results/.gpu_busy` for the duration so a second GPU job cannot corrupt the timings.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

from medtrt.config import ENGINES_DIR, MEDTRT_ROOT, RESULTS_DIR
from medtrt.profiles import PROFILES

Step = Tuple[str, List[str]]
GPU_LOCK = MEDTRT_ROOT / "results" / ".gpu_busy"


def build_steps(smoke: bool, rebuild_engine: bool, nsys: bool) -> List[Step]:
    steps: List[Step] = [("env", ["medtrt.config", "--env-audit"]), ("data", ["medtrt.data"])]
    if rebuild_engine or not (ENGINES_DIR / "spleen_unet_fp16.engine").exists():
        steps.append(("engine", ["medtrt.export"]))

    if smoke:
        for name in ("trt_fp16", "trt_fp16_optimized"):
            steps.append((f"bench:{name}", ["medtrt.bench", "--backend", name, "--cases", "1", "--warmup", "1", "--repeats", "1"]))
        return steps

    steps += [(f"bench:{name}", ["medtrt.bench", "--backend", name]) for name in PROFILES]
    steps += [(f"eval:{name}", ["medtrt.evaluate", "--backend", name]) for name in PROFILES]
    timeline = ["medtrt.stats", "--timeline"] + (["--nsys"] if nsys else [])
    steps += [
        ("timeline", timeline),
        ("sweep", ["medtrt.stats", "--sweep"]),
        ("ablation", ["medtrt.ablate"]),
        ("probes", ["medtrt.probes"]),
        ("validate", ["medtrt.stats", "--validate"]),
        ("failures", ["medtrt.stats", "--failures"]),
        ("report", ["medtrt.report"]),
        ("verify", ["medtrt.verify_claims"]),
    ]
    return steps


def run_steps(steps: List[Step], env: Dict[str, str], only: str | None) -> None:
    started = time.time()
    selected = [s for s in steps if only is None or s[0].startswith(only)]
    if not selected:
        raise SystemExit(f"No step matches {only!r}. Steps: {[s[0] for s in steps]}")
    for index, (name, module_args) in enumerate(selected, start=1):
        print(f"\n[{index}/{len(selected)}] {name}  (elapsed {(time.time() - started) / 60:.1f} min)", flush=True)
        result = subprocess.run([sys.executable, "-m", *module_args], cwd=str(MEDTRT_ROOT), env=env)
        if result.returncode != 0:
            raise SystemExit(f"Step {name!r} failed with exit code {result.returncode}.")
    print(f"\nCompleted {len(selected)} steps in {(time.time() - started) / 60:.1f} min.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate all MedTRT results")
    parser.add_argument("--smoke", action="store_true", help="Quick check, written to a temporary directory")
    parser.add_argument("--rebuild-engine", action="store_true", help="Rebuild the TensorRT engine first")
    parser.add_argument("--nsys", action="store_true", help="Also run Nsight Systems if it is installed")
    parser.add_argument("--only", type=str, default=None, help="Run only steps whose name starts with this")
    parser.add_argument("--dry-run", action="store_true", help="List the steps without running them")
    args = parser.parse_args()

    steps = build_steps(args.smoke, args.rebuild_engine, args.nsys)
    if args.dry_run:
        for name, module_args in steps:
            print(f"{name:32} python -m {' '.join(module_args)}")
        return

    env = {**os.environ, "PYTHONPATH": str(MEDTRT_ROOT / "src"), "PYTHONIOENCODING": "utf-8"}
    if GPU_LOCK.exists():
        raise SystemExit(f"{GPU_LOCK} exists, so another GPU job may be running. Delete it only if none is.")
    GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    GPU_LOCK.write_text(f"medtrt.pipeline pid {os.getpid()}\n", encoding="utf-8")
    try:
        if args.smoke:
            with tempfile.TemporaryDirectory(prefix="medtrt_smoke_") as tmp:
                env["MEDTRT_RESULTS_DIR"] = tmp
                run_steps(steps, env, args.only)
        else:
            run_steps(steps, env, args.only)
    finally:
        GPU_LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
