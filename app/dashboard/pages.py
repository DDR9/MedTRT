"""
Dashboard Pages (`app/dashboard/pages.py`).

Each page reads `results/` through `dashboard.data`, so numbers and the sentences built from them
update when the files change. A section with no data says which command produces it.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from dashboard import charts, data

COMMAND_ALL = "python -m medtrt.pipeline"
CONFIG = {"displayModeBar": False, "responsive": True}


@st.cache_data(show_spinner=False, max_entries=2)
def _load(_signature: Any) -> data.Results:
    return data.load()


def results() -> data.Results:
    return _load(data.signature())


def _ms(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:,.0f} ms" if value >= 100 else f"{value:,.1f} ms"


def _x(value: Optional[float], digits: int = 2) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{value:,.{digits}f}x"


def _number(value: Optional[float], fmt: str) -> str:
    return "n/a" if value is None or pd.isna(value) else format(value, fmt)


def _columns(frame: pd.DataFrame) -> Dict[str, Any]:
    """Number formats follow the column's unit, so new columns format themselves."""
    config: Dict[str, Any] = {}
    for column in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(frame[column]):
            continue
        name = str(column)
        if "Dice" in name or "Sensitivity" in name or "Specificity" in name or "Precision" == name or "IoU" in name or "similarity" in name:
            fmt = "%.6f"
        elif "Bias CI" in name or "LoA" in name:
            fmt = "%.3f"
        elif "(ms)" in name or "cm³" in name or "(mm)" in name:
            fmt = "%.3f" if "Δ" in name or "bias" in name.lower() else "%.2f"
        elif "(MB)" in name:
            fmt = "%.1f"
        elif "(%)" in name or "Volumes/s" in name or "gain" in name:
            fmt = "%.2f"
        elif "Speedup" in name or "vs first" in name:
            fmt = "%.2fx"
        else:
            continue
        config[column] = st.column_config.NumberColumn(format=fmt)
    return config


def table(frame: pd.DataFrame, **kwargs: Any) -> None:
    st.dataframe(frame, width="stretch", hide_index=True, column_config=_columns(frame), **kwargs)


def show(figure: Any, key: str) -> None:
    st.plotly_chart(figure, width="stretch", theme=None, config=CONFIG, key=key)


def missing(what: str, command: str = COMMAND_ALL) -> None:
    st.info(f"No {what} yet. Run `{command}`.", icon=":material/hourglass_empty:")


def status_text(ok: bool) -> str:
    return "✅ Pass" if ok else "❌ Fail"


def wrapped_table(frame: pd.DataFrame, index: str) -> None:
    """For long text, which `st.dataframe` truncates. The first column becomes the row header."""
    st.table(frame.set_index(index))


def figure(name: str, caption: str = "") -> None:
    path = data.FIGURES_DIR / name
    if path.exists():
        st.image(str(path), caption=caption or None, width="stretch")


def banner(res: data.Results) -> None:
    state = data.status(res)
    text = f"**{state['title']}.** {state['detail']}"
    if state["level"] == "verified":
        st.success(text, icon=":material/verified:")
    elif state["level"] == "unverified":
        st.warning(text, icon=":material/warning:")
    else:
        st.info(text, icon=":material/info:")


@st.fragment(run_every="15s")
def _watch_results() -> None:
    """Refresh every page when any result file changes, with no restart."""
    current = data.signature()
    previous = st.session_state.get("results_signature")
    st.session_state["results_signature"] = current
    if previous is not None and previous != current:
        st.rerun()


def _run_verifier() -> None:
    env = {**os.environ, "PYTHONPATH": str(data.ROOT / "src"), "PYTHONIOENCODING": "utf-8"}
    subprocess.run([sys.executable, "-m", "medtrt.verify_claims"], cwd=str(data.ROOT), env=env, capture_output=True, timeout=180)


def shell() -> None:
    """The parts every page shares."""
    res = results()
    with st.sidebar:
        _watch_results()
        cases = data.n_cases(res)
        st.caption(f"{len(data.profile_names(res))} profiles, {cases} cases. Updates automatically when `results/` changes.")
        if st.button("Re-run claims check", width="stretch", help="Runs `python -m medtrt.verify_claims`. Uses no GPU."):
            with st.spinner("Checking results"):
                _run_verifier()
            st.rerun()
    st.caption("Research benchmark only. Not a medical device. No diagnostic, triage, or clinical performance claims.")
    banner(res)


def overview() -> None:
    res = results()
    roles = data.resolve_roles(res)
    goal = data.targets(res)
    st.title("MedTRT results")
    gpu = res.env.get("gpu_model", "this GPU")
    st.caption(f"3D spleen segmentation on {gpu}: where the time goes, and whether faster profiles stay accurate.")

    if not roles.fast:
        missing("benchmark results")
        return

    row = data.agreement_row(res, roles, roles.fast)
    e2e = data.stage(res, roles.fast, "e2e_total")
    slowest = data.stage(res, roles.fast, "e2e_total", "max")
    host = data.stage(res, roles.host_baseline, "e2e_total")
    gpu_base = data.stage(res, roles.gpu_baseline, "e2e_total")
    dice = data.mean_dice(res, roles.fast)

    st.subheader(f"Fast profile: {data.label(res, roles.fast)}")
    first, second = st.columns(3), st.columns(3)
    first[0].metric("End-to-end median", _ms(e2e), border=True)
    first[1].metric("Slowest pass", _ms(slowest), border=True)
    first[2].metric("Mean Dice", _number(dice, ".6f"), border=True)
    second[0].metric("Speedup vs host path", _x(data.ratio(host, e2e), 1), border=True, help=f"Baseline: {roles.host_baseline}")
    second[1].metric("Speedup vs GPU path", _x(data.ratio(gpu_base, e2e)), border=True, help=f"Baseline: {roles.gpu_baseline}")
    second[2].metric("Volume bias vs FP32", f"{row['Volume bias (cm³)']:+.3f} cm³" if row is not None else "n/a", border=True)

    left, right = st.columns([1, 1.25])
    with left:
        st.subheader("Targets")
        checks = [
            (f"Every pass under {goal['e2e_ms']:.0f} ms", _ms(slowest), slowest is not None and slowest < goal["e2e_ms"]),
            (f"Mean Dice at least {goal['mean_dice']:g}", _number(dice, ".6f"), dice is not None and dice >= goal["mean_dice"]),
            (
                f"Mean |ΔDice| under {goal['abs_delta_dice']:g}",
                _number(row["Mean |ΔDice|"] if row is not None else None, ".6f"),
                row is not None and row["Mean |ΔDice|"] < goal["abs_delta_dice"],
            ),
            (
                f"Every case |ΔDice| under {goal['abs_delta_dice']:g}",
                _number(row["Max |ΔDice|"] if row is not None else None, ".6f"),
                row is not None and row["Max |ΔDice|"] < goal["abs_delta_dice"],
            ),
        ]
        table(pd.DataFrame([{"Target": t, "Measured": m, "Status": status_text(ok)} for t, m, ok in checks]))
        st.subheader("What the numbers say")
        lines = data.takeaways(res, roles)
        st.markdown("\n".join(f"- {line}" for line in lines) if lines else "Not enough results yet.")
    with right:
        st.subheader("End-to-end latency by profile")
        frame = data.profile_table(res, roles).dropna(subset=["E2E median (ms)"]).sort_values("E2E median (ms)")
        show(
            charts.ranked_bars(
                frame["Profile"].tolist(),
                frame["E2E median (ms)"].tolist(),
                "ms",
                highlight=roles.fast,
                threshold=goal["e2e_ms"],
                threshold_label=f"Target {goal['e2e_ms']:.0f} ms",
                x_title="Median end-to-end latency (ms)",
            ),
            "overview_e2e",
        )

    st.subheader("All profiles")
    table(data.profile_table(res, roles))


def speedup() -> None:
    res = results()
    st.title("Where the speedup comes from")
    st.caption("Each row changes one thing, on the same cases, so each gain can be credited correctly.")
    frame = data.ablation_frame(res)
    if frame.empty:
        missing("ablation results", "python -m medtrt.ablate")
        return

    parts = [c for c in ("Ingest (ms)", "Preprocess (ms)", "Network (ms)", "Postprocess (ms)") if c in frame.columns]
    host = frame[frame["Transforms"] == "cpu"]
    gpu = frame[frame["Transforms"] != "cpu"]
    left, right = st.columns(2)
    with left:
        st.subheader("Host path")
        st.caption("Scales differ between these two charts.")
        show(charts.stacked_bars(host, "Configuration", parts), "ladder_host")
    with right:
        st.subheader("GPU path")
        st.caption("Scales differ between these two charts.")
        show(charts.stacked_bars(gpu, "Configuration", parts), "ladder_gpu")

    st.subheader("Every step")
    table(frame.drop(columns=["Rung"]))


def latency() -> None:
    res = results()
    roles = data.resolve_roles(res)
    goal = data.targets(res)
    st.title("Latency")
    if not res.bench:
        missing("benchmark results")
        return

    names = sorted(res.bench)
    default = names.index(roles.fast) if roles.fast in names else 0
    chosen = st.selectbox("Profile for the per-case views", names, index=default)

    st.subheader("Stages by profile")
    st.caption("`ingest` is `load + h2d`. Quote it instead of `load`, because `mmap` defers the read into `h2d`.")
    stages = data.stage_frame(res, names)
    host_names = [n for n in names if data.profile_meta(res, n).get("transforms") in ("cpu", None)]
    gpu_names = [n for n in names if n not in host_names]
    left, right = st.columns(2)
    with left:
        st.markdown("**Host path**")
        if host_names:
            show(charts.stage_stack(stages[stages["Profile"].isin(host_names)]), "stages_host")
    with right:
        st.markdown("**GPU path**")
        if gpu_names:
            show(charts.stage_stack(stages[stages["Profile"].isin(gpu_names)]), "stages_gpu")
    keys = list(data.STAGES) + ["ingest", "non_network", "e2e_total", "wall_ms"]
    wide = pd.DataFrame(
        [{"Stage (median ms)": k, **{n: data.stage(res, n, k) for n in names}} for k in keys]
    ).dropna(axis=1, how="all")
    table(wide.round(2))

    st.subheader(f"Per case: {chosen}")
    cases = data.per_case_latency(res, chosen)
    if cases.empty:
        missing("per-case latency")
    else:
        ordered = cases.sort_values("Median (ms)")
        left, right = st.columns(2)
        with left:
            show(
                charts.ranked_bars(
                    ordered["Case"].tolist(),
                    ordered["Median (ms)"].tolist(),
                    "ms",
                    threshold=goal["e2e_ms"],
                    threshold_label=f"Target {goal['e2e_ms']:.0f} ms",
                    x_title="Median end-to-end latency (ms)",
                ),
                "cases_median",
            )
        with right:
            passes = data.passes_frame(res, chosen)
            if passes.empty:
                missing("raw per-pass timings")
            else:
                show(charts.pass_spread(passes, goal["e2e_ms"], f"Target {goal['e2e_ms']:.0f} ms"), "cases_spread")
        table(cases)

    st.subheader("Patch-level latency")
    st.caption("Isolated forward passes on random input, with no stitching or I/O.")
    patches = data.patch_frame(res)
    if patches.empty:
        missing("patch timings")
    else:
        show(charts.patch_lines(patches), "patch_lines")
        table(patches)

    telemetry = res.bench[chosen].get("config", {})
    before, after = telemetry.get("gpu_telemetry_pre"), telemetry.get("gpu_telemetry_post")
    if before or after:
        st.subheader(f"GPU state during the run: {chosen}")
        table(pd.DataFrame([{"When": "Before", **(before or {})}, {"When": "After", **(after or {})}]))


def network() -> None:
    res = results()
    roles = data.resolve_roles(res)
    st.title("Network, ingestion, and sweep")
    tabs = st.tabs(["Like for like", "Clock state", "Ingestion", "Amdahl", "Sweep"])

    with tabs[0]:
        st.caption("The same windows on every backend. Comparing different window counts measures less work, not a faster network.")
        sustained = res.probes.get("network_sustained", {})
        if sustained:
            rows = []
            for key, block in sustained.items():
                backend, overlap, batch = key.split("|")
                rows.append({"Backend": backend, "Overlap": float(overlap), "Batch": int(batch), "Network (ms)": block.get("sw_network_ms_median"), "Windows": block.get("windows_all_cases")})
            frame = pd.DataFrame(rows)
            bundle = frame[(frame["Overlap"] == 0.5)]
            show(charts.ranked_bars(bundle["Backend"].tolist(), bundle["Network (ms)"].tolist(), "ms", highlight="trt", x_title="Network time per volume at overlap 0.5 (ms)"), "net_bundle")
            table(frame)
        else:
            missing("network comparison", "python -m medtrt.probes")
        value = data.like_for_like_network(res, roles)
        if value:
            st.markdown(f"In the real pipeline at identical windows, TensorRT is **{value:.2f}x** faster than `{roles.gpu_baseline}`.")

    with tabs[1]:
        st.caption("`sw_network` depends on what the GPU was doing just before, so values from different pipelines are not directly comparable.")
        clock = res.probes.get("clock_state", {})
        if clock:
            table(pd.DataFrame([{"Run": k, **v} for k, v in clock.items()]))
        else:
            missing("clock-state results", "python -m medtrt.probes")

    with tabs[2]:
        st.caption("Warm OS page cache. `mmap` returns before reading, so only time through to the GPU is a fair comparison.")
        ingestion = {k: v for k, v in res.probes.get("ingestion", {}).items() if isinstance(v, dict)}
        if ingestion:
            table(pd.DataFrame([{"Read path": k, "Read call (ms)": v.get("read_ms_median"), "Through to GPU (ms)": v.get("to_gpu_total_ms_median")} for k, v in ingestion.items()]))
        else:
            missing("ingestion results", "python -m medtrt.probes")
        dtype = res.probes.get("resample_dtype", {})
        if dtype.get("inverse"):
            st.markdown(f"Resampling runs in `{dtype['inverse'][0]['dtype']}` on `{dtype['inverse'][0]['device']}`. The final mask copy is `{dtype.get('final_mask_dtype')}`.")

    with tabs[3]:
        st.caption("Once the network is a small share of the total, a faster network barely moves it.")
        rows = []
        for name, bench in res.bench.items():
            block = bench.get("amdahl", {})
            if block:
                rows.append(
                    {
                        "Profile": name,
                        "Network (ms)": block.get("median_sw_network_ms"),
                        "E2E (ms)": block.get("median_e2e_ms"),
                        "Network share (%)": 100.0 * block.get("network_fraction_p", 0.0),
                        "Max gain from a faster network": block.get("theoretical_max_e2e_speedup_1_over_s"),
                    }
                )
        table(pd.DataFrame(rows))

    with tabs[4]:
        st.caption("Chosen on the evaluation cases, so this is selection on the evaluation set.")
        if res.sweep.empty:
            missing("sweep results", "python -m medtrt.stats --sweep")
        else:
            show(charts.sweep_scatter(res.sweep), "sweep")
            st.caption("Marker shape shows batch size: circle 1, square 2, triangle 4, diamond 8.")
            table(res.sweep)


def accuracy() -> None:
    res = results()
    roles = data.resolve_roles(res)
    goal = data.targets(res)
    st.title("Accuracy and agreement")
    summary = data.agreement_table(res, roles)
    if summary.empty:
        missing("evaluation results")
        return

    st.caption(f"Compared with `{roles.reference}`, across {data.n_cases(res)} cases. Recomputed from the per-case CSVs on every load.")
    table(summary)

    names = summary["Profile"].tolist()
    default = names.index(roles.fast) if roles.fast in names else 0
    chosen = st.selectbox("Profile for the detail views", names, index=default)
    cases = data.per_case_agreement(res, chosen, roles.reference)

    left, right = st.columns(2)
    with left:
        st.subheader("Dice change by case")
        show(charts.dice_deltas(cases, goal["abs_delta_dice"]), "delta_dice")
        st.caption(f"The shaded band is the ±{goal['abs_delta_dice']:g} tolerance.")
    with right:
        st.subheader("Volume agreement")
        frame, stats = data.bland_altman_frame(res, chosen, roles.reference)
        show(charts.bland_altman(frame, stats), "bland_altman")
        verdict = "excludes zero, so the shift is systematic" if stats["ci_excludes_zero"] else "includes zero, so no systematic shift is shown"
        st.caption(f"95% CI of the bias is [{stats['ci_low']:+.3f}, {stats['ci_high']:+.3f}] cm³, which {verdict}. With {stats['n']} cases, these limits are imprecise.")
    table(cases)

    st.subheader("Segmentation metrics")
    st.caption("Mean across cases, against ground truth.")
    table(data.segmentation_table(res))

    unfolded = [n for n in res.evals if data.profile_meta(res, n).get("precision") == "fp16_unfolded"]
    numerics = res.probes.get("fp16_numerics", {})
    if unfolded or numerics:
        st.subheader("FP16 numerics")
    if unfolded:
        st.caption("The exported graph keeps BatchNorm after each ConvTranspose3d. Folding it before export removes the FP16 volume shift, as these rows show.")
        keep = summary[summary["Profile"].isin(unfolded + ["trt_fp16"])]
        table(keep[["Profile", "Mean Dice", "Mean |ΔDice|", "Volume bias (cm³)", "Larger than reference"]])
    if numerics:
        table(pd.DataFrame([numerics]))


def errors() -> None:
    res = results()
    roles = data.resolve_roles(res)
    st.title("Error analysis")
    st.caption("Where errors occur, not why. These are observations, not proven causes.")

    frame = data.relation_frame(res, roles.fast)
    if frame.empty:
        missing("evaluation results")
        return

    left, right = st.columns(2)
    for column, target, key in ((left, "Slice spacing (mm)", "rel_z"), (right, "Spleen volume (cm³)", "rel_v")):
        with column:
            rho, p = data.spearman(frame, target)
            st.subheader(f"Dice vs {target.split(' (')[0].lower()}")
            show(charts.relation(frame, target), key)
            if rho is not None:
                st.caption(f"Spearman {rho:+.2f}, p = {p:.2f}, n = {len(frame)}.")

    st.subheader("Lowest-Dice cases")
    for item in res.failures.get("worst_cases", []):
        err = item.get("errors", {})
        case_dice = item.get("dice", item.get("dice_trt_fp16"))
        st.markdown(f"**{item.get('case_id')}**: Dice {_number(case_dice, '.4f')}, spacing {item.get('voxel_spacing_mm')} mm")
        if err:
            pretty = [{"Measure": k.replace("_", " ").capitalize(), "Value": v} for k, v in err.items()]
            st.table(pd.DataFrame(pretty).set_index("Measure"))
        figure(Path(item.get("figure_path", "")).name)
    if not res.failures:
        missing("failure analysis", "python -m medtrt.stats --failures")


def claims() -> None:
    res = results()
    roles = data.resolve_roles(res)
    st.title("Original claims vs current results")
    st.caption("The first column is the fixed record of what was claimed. The result column is computed from `results/`.")
    frame = data.claims_frame(res, roles)
    if frame.empty:
        missing("claims record")
        return
    counts = frame["Outcome"].value_counts()
    tiles = st.columns(max(len(counts), 1))
    for tile, (outcome, count) in zip(tiles, counts.items()):
        tile.metric(str(outcome), int(count), border=True)
    outcomes = st.multiselect("Show", sorted(frame["Outcome"].unique()), default=sorted(frame["Outcome"].unique()))
    wrapped_table(frame[frame["Outcome"].isin(outcomes)], "Original claim")

    checks = res.claims_check.get("checks", [])
    st.subheader("Verifier checks")
    if checks:
        wrapped_table(pd.DataFrame([{"Check": c.get("check"), "Status": status_text(bool(c.get("passed"))), "Detail": c.get("detail")} for c in checks]), "Check")
    else:
        missing("verifier report", "python -m medtrt.verify_claims")


def _results_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(data.RESULTS_DIR.iterdir()):
            if path.is_file() and path.suffix in (".json", ".csv"):
                archive.write(path, arcname=path.name)
    return buffer.getvalue()


def method() -> None:
    res = results()
    roles = data.resolve_roles(res)
    st.title("Method and data")

    left, right = st.columns(2)
    with left:
        st.subheader("Environment")
        if res.env:
            table(pd.DataFrame([{"Item": k, "Value": str(v)} for k, v in res.env.items() if k not in ("timestamp", "tensorrt_error")]))
        else:
            missing("environment audit", "python -m medtrt.config --env-audit")
    with right:
        st.subheader("TensorRT engine")
        if res.build:
            flat = {k: v for k, v in res.build.items() if not isinstance(v, (dict, list))}
            flat.update({f"onnx {k}": v for k, v in res.build.get("onnx", {}).items()})
            table(pd.DataFrame([{"Item": k, "Value": str(v)} for k, v in flat.items()]))
        else:
            missing("engine record", "python -m medtrt.export")

    st.subheader("Protocol")
    config = res.bench.get(roles.fast or "", {}).get("config", {})
    cases = data.n_cases(res)
    st.markdown(
        f"- **Timing:** {config.get('warmup_volumes', 'n/a')} warm-up volumes, then {config.get('timed_repeats_per_case', 'n/a')} passes on each of {cases} cases. "
        "Stages use CUDA events, and each pass is also timed by wall clock. A pass ends with the label map in host memory.\n"
        f"- **Reference:** `{roles.reference}`, true FP32 on the unmodified bundle weights.\n"
        f"- **Baselines:** `{roles.host_baseline or 'not benchmarked yet'}` on the host path and `{roles.gpu_baseline or 'not benchmarked yet'}` on the GPU path.\n"
        f"- **Evaluation cases:** the {cases} cases the bundle reserves for validation. They were never trained on, but the bundle selected its checkpoint on them, "
        "and MSD test labels are not public. Treat Dice as optimistic, suitable for comparing backends."
    )

    st.subheader("Limitations")
    nsight = res.timeline.get("nsys_cli_executed")
    st.markdown(
        f"- **Small sample:** {cases} cases from one public dataset, which also selected the checkpoint.\n"
        f"- **One machine:** one {res.env.get('gpu_model', 'GPU')}, whose clocks change with load and temperature.\n"
        "- **Best-case I/O:** the GPU path assumes volumes are already decompressed and in the OS page cache.\n"
        "- **Warm process:** model loading, engine deserialization, and CUDA start-up are excluded.\n"
        "- **Not a product path:** DICOM ingestion, result saving, and concurrency are not measured.\n"
        + ("" if nsight else "- **Profiling:** Nsight Systems has not been run. Timings come from CUDA events and wall clock.\n")
    )

    st.subheader("Reproduce")
    st.markdown("Check that the README matches `results/`. No GPU or data needed:")
    st.code("python -m medtrt.verify_claims", language="bash")
    st.markdown("Regenerate everything on an NVIDIA GPU:")
    st.code(COMMAND_ALL, language="bash")
    st.markdown("Run this dashboard:")
    st.code("streamlit run app/viewer.py --server.port 8502", language="bash")

    st.subheader("Provenance")
    st.caption("Results from different code versions should not be compared.")
    table(data.provenance_frame(res))

    if data.RESULTS_DIR.exists():
        st.download_button("Download all results (.zip)", _results_zip(), "medtrt_results.zip", "application/zip")

    others = data.other_result_files(res)
    if others:
        st.subheader("Other result files")
        chosen = st.selectbox("File", [p.name for p in others])
        path = data.RESULTS_DIR / chosen
        if path.suffix == ".csv":
            table(pd.read_csv(path))
        else:
            st.json(path.read_text(encoding="utf-8"), expanded=False)

    st.subheader("Data and licenses")
    st.markdown(
        "The dataset, weights, and engines are not in this repository. `Task09_Spleen` is from the "
        "[Medical Segmentation Decathlon](http://medicaldecathlon.com/) under CC-BY-SA 4.0, and the failure figures are derived from it. "
        "The MONAI bundle is Apache-2.0."
    )
