"""
Multi-Repeat Scientific GPU Benchmarking Driver (`src/medtrt/bench.py`).

Responsibilities:
- Isolate pure patch forward latency `(B, 1, 96, 96, 96)` from full-volume end-to-end latency.
- Time all 6 pipeline stages (`load`, `preprocess`, `h2d`, `sw_network`, `sw_aggregate`, `postprocess`)
  with `torch.cuda.Event` boundaries inside NVTX ranges, and cross-check whole-pass latency with
  synchronized `time.perf_counter()`.
- Support all 3 transform execution paths (`cpu`, `gpu` via MONAI `grid_sample`, and `cuda` via custom
  CUDA C++ fused kernels).
- Record GPU hardware telemetry (temperature, clock frequency, power draw/limit, allocated & reserved VRAM),
  throughput (`volumes/s`), and stage latency fractions (`%`).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import torch
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged

from medtrt.config import MEDTRT_ROOT, RESULTS_DIR, run_metadata
from medtrt.cuda_kernels import cuda_fused_invert_postprocess, cuda_fused_preprocess
from medtrt.data import HELDOUT_JSON_PATH, ensure_uncompressed_nifti, resolve_heldout_split
from medtrt.infer import (
    Pipeline,
    _spatial_and_intensity,
    build_inference_postprocessing,
    build_pipeline,
    build_sliding_window_inferer,
)
from medtrt.profiles import PROFILES, Profile, build_predictor, get_profile
from medtrt.timing import STAGE_NAMES, CudaStageTimer, InstrumentedPredictor, compute_distribution_stats

SUMMARY_METRICS = STAGE_NAMES + ("ingest", "sw_total", "non_network", "e2e_total", "wall_ms")


def build_split_load_and_preprocess() -> Tuple[Compose, Compose, Compose]:
    """Backward-compatible helper returning `(load_tf, pre_tf, post_tf)` for host transforms."""
    load_tf = Compose([LoadImaged(keys=["image"]), EnsureChannelFirstd(keys=["image"])])
    pre_tf = Compose([*_spatial_and_intensity(), EnsureTyped(keys=["image"])])
    post_tf = build_inference_postprocessing(pre_tf)
    return load_tf, pre_tf, post_tf


def query_gpu_telemetry() -> Dict[str, Any]:
    """Query live GPU clocks, temperature, power draw, power limit, and utilization via nvidia-smi."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,clocks.current.graphics,clocks.current.memory,power.draw,power.limit,pstate,utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        parts = [p.strip() for p in out.splitlines()[0].split(",")]
        return {
            "gpu_temp_c": float(parts[0]) if parts[0].replace(".", "").isdigit() else None,
            "graphics_clock_mhz": float(parts[1]) if parts[1].replace(".", "").isdigit() else None,
            "memory_clock_mhz": float(parts[2]) if parts[2].replace(".", "").isdigit() else None,
            "power_draw_w": float(parts[3]) if parts[3].replace(".", "").isdigit() else None,
            "power_limit_w": float(parts[4]) if parts[4].replace(".", "").isdigit() else None,
            "pstate": parts[5],
            "gpu_util_pct": float(parts[6]) if parts[6].replace(".", "").isdigit() else None,
            "mem_util_pct": float(parts[7]) if parts[7].replace(".", "").isdigit() else None,
        }
    except Exception:
        return {}


def benchmark_pure_patch_forward(
    predictor: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    use_amp: bool = False,
    batch_sizes: Tuple[int, ...] = (1, 2, 4, 8),
    warmup_iters: int = 10,
    timed_iters: int = 30,
    extra_vram_mb: float = 0.0,
) -> Dict[str, Any]:
    """Measure isolated patch forward latency on the GPU, without stitching or host I/O."""
    results: Dict[str, Any] = {}
    for b in batch_sizes:
        patch = torch.randn((b, 1, 96, 96, 96), dtype=torch.float32, device=device)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            for _ in range(warmup_iters):
                _ = predictor(patch)
        torch.cuda.synchronize(device)

        torch.cuda.reset_peak_memory_stats(device)
        latencies_ms: List[float] = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            for _ in range(timed_iters):
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                torch.cuda.nvtx.range_push(f"patch_forward_b{b}")
                start_ev.record()
                _ = predictor(patch)
                end_ev.record()
                end_ev.synchronize()
                torch.cuda.nvtx.range_pop()
                latencies_ms.append(float(start_ev.elapsed_time(end_ev)))

        stats = compute_distribution_stats(latencies_ms)
        stats["per_patch_median_ms"] = round(stats["median"] / b, 4)
        stats["patch_throughput_patches_per_s"] = round(1000.0 * b / max(stats["median"], 1e-6), 2)
        stats["peak_vram_mb"] = round((torch.cuda.max_memory_allocated(device) / (1024**2)) + extra_vram_mb, 2)
        stats["peak_reserved_vram_mb"] = round((torch.cuda.max_memory_reserved(device) / (1024**2)) + extra_vram_mb, 2)
        results[f"batch_{b}"] = stats

        del patch
        torch.cuda.empty_cache()
    return results


def run_single_timed_volume_pass(
    image_path: Path,
    predictor: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    pipeline: Pipeline | None = None,
    profile: Profile | None = None,
    use_amp: bool = False,
    extra_vram_mb: float = 0.0,
) -> Dict[str, float]:
    """
    Time one full-volume pass across `load`, `preprocess`, `h2d`, `sw_network`, `sw_aggregate`,
    and `postprocess`. The pass ends only when the label map is in host memory.
    """
    if profile is None:
        profile = get_profile("pytorch_fp32")
    if pipeline is None:
        pipeline = build_pipeline(profile)

    load_tf, pre_tf, post_tf = pipeline
    inferer = build_sliding_window_inferer(sw_batch_size=profile.sw_batch_size, overlap=profile.overlap)
    timer = CudaStageTimer()
    inst_predictor = InstrumentedPredictor(predictor=predictor, timer=timer, use_amp=use_amp)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    wall_start = time.perf_counter()
    timer.start_e2e()

    with timer.stage("load", sync_on_exit=True):
        data: Dict[str, Any] = load_tf({"image": str(image_path)})

    if profile.custom_cuda_path:
        orig_affine_np = data["image"].affine.cpu().numpy()
        with timer.stage("h2d", sync_on_exit=True):
            raw_gpu = data["image"].to(device, non_blocking=False)
        with timer.stage("preprocess", sync_on_exit=True):
            pre_out, orig_shape, _, m_inv = cuda_fused_preprocess(raw_gpu, orig_affine_np)
            inputs = pre_out.unsqueeze(0)
        with timer.stage("_sw_total", sync_on_exit=True):
            torch.cuda.nvtx.range_push("sw_aggregate")
            try:
                with torch.no_grad():
                    logits = inferer(inputs=inputs, network=inst_predictor)
            finally:
                torch.cuda.nvtx.range_pop()
        with timer.stage("postprocess", sync_on_exit=True):
            mask_u8, _ = cuda_fused_invert_postprocess(
                logits[0].detach(), orig_shape, m_inv, return_voxel_count=False
            )
            _ = mask_u8.cpu()
    elif profile.gpu_path:
        with timer.stage("h2d", sync_on_exit=True):
            data["image"] = data["image"].to(device, non_blocking=False)
        with timer.stage("preprocess", sync_on_exit=True):
            data = pre_tf(data)
            inputs = data["image"].unsqueeze(0)
        with timer.stage("_sw_total", sync_on_exit=True):
            torch.cuda.nvtx.range_push("sw_aggregate")
            try:
                with torch.no_grad():
                    logits = inferer(inputs=inputs, network=inst_predictor)
            finally:
                torch.cuda.nvtx.range_pop()
        with timer.stage("postprocess", sync_on_exit=True):
            data["pred"] = logits[0].detach().float()
            _ = post_tf(data)["pred"].cpu()
    else:
        with timer.stage("preprocess", sync_on_exit=True):
            data = pre_tf(data)
        with timer.stage("h2d", sync_on_exit=True):
            inputs = data["image"].unsqueeze(0).to(device, non_blocking=False)
        with timer.stage("_sw_total", sync_on_exit=True):
            torch.cuda.nvtx.range_push("sw_aggregate")
            try:
                with torch.no_grad():
                    logits = inferer(inputs=inputs, network=inst_predictor)
            finally:
                torch.cuda.nvtx.range_pop()
        with timer.stage("postprocess", sync_on_exit=True):
            data["pred"] = logits[0].detach().float().cpu()
            _ = post_tf(data)["pred"].cpu()

    timer.stop_e2e()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0

    timings = timer.resolve_timings_ms()
    timings["wall_ms"] = wall_ms
    timings["peak_vram_mb"] = float((torch.cuda.max_memory_allocated(device) / (1024**2)) + extra_vram_mb)
    timings["peak_reserved_vram_mb"] = float((torch.cuda.max_memory_reserved(device) / (1024**2)) + extra_vram_mb)
    return timings


def run_benchmark(
    backend: str,
    num_cases: int | None = None,
    warmup_volumes: int = 3,
    timed_repeats: int = 10,
    results_dir: Path = RESULTS_DIR,
) -> Dict[str, Any]:
    """Run the full benchmark for profile `backend` and write `results/bench_<backend>.json`."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable. Stopping per Ground Rule 6.")

    if not HELDOUT_JSON_PATH.exists():
        heldout_info = resolve_heldout_split()
    else:
        with open(HELDOUT_JSON_PATH, "r", encoding="utf-8") as f:
            heldout_info = json.load(f)

    profile = get_profile(backend)
    device = torch.device("cuda:0")
    predictor, use_amp, extra_vram_mb = build_predictor(profile, device)

    cases = heldout_info["cases"]
    if num_cases is not None and num_cases > 0:
        cases = cases[:num_cases]
    if profile.io == "mmap":
        for c_item in cases:
            ensure_uncompressed_nifti(MEDTRT_ROOT / c_item["image"])

    print("=" * 96)
    print(
        f"MEDTRT BENCHMARK: {backend} | cases={len(cases)} | warmup={warmup_volumes} | repeats={timed_repeats} | "
        f"transforms={profile.transforms} | sw_batch_size={profile.sw_batch_size} | overlap={profile.overlap}"
    )
    print("=" * 96)

    patch_bench = benchmark_pure_patch_forward(predictor, device, use_amp=use_amp, extra_vram_mb=extra_vram_mb)
    for b_key, b_val in patch_bench.items():
        print(f"  {b_key}: median={b_val['median']:.2f} ms (IQR={b_val['iqr']:.2f} ms, P95={b_val['p95']:.2f} ms)")

    pipeline = build_pipeline(profile)
    telemetry_pre = query_gpu_telemetry()

    def one_pass(path: Path) -> Dict[str, float]:
        return run_single_timed_volume_pass(path, predictor, device, pipeline, profile, use_amp, extra_vram_mb)

    for w_idx in range(warmup_volumes):
        one_pass(MEDTRT_ROOT / cases[w_idx % len(cases)]["image"])

    all_passes: List[Dict[str, Any]] = []
    per_case: Dict[str, Any] = {}
    metric_keys = list(SUMMARY_METRICS) + ["peak_vram_mb", "peak_reserved_vram_mb"]

    for c_item in cases:
        case_id = c_item["case_id"]
        case_passes = [one_pass(MEDTRT_ROOT / c_item["image"]) for _ in range(timed_repeats)]
        all_passes.extend({"case_id": case_id, "repeat": i, **p} for i, p in enumerate(case_passes))
        c_stats: Dict[str, Any] = {k: compute_distribution_stats([p[k] for p in case_passes]) for k in metric_keys}
        c_stats["patch_calls"] = int(case_passes[0]["patch_calls"])
        c_stats["patch_windows"] = int(case_passes[0]["patch_windows"])
        per_case[case_id] = c_stats
        print(
            f"  {case_id:<10} | e2e median={c_stats['e2e_total']['median']:8.2f} ms "
            f"(P95={c_stats['e2e_total']['p95']:8.2f} ms) | sw_net={c_stats['sw_network']['median']:7.2f} ms "
            f"| post={c_stats['postprocess']['median']:7.2f} ms"
        )

    telemetry_post = query_gpu_telemetry()
    aggregate = {k: compute_distribution_stats([p[k] for p in all_passes]) for k in metric_keys}

    median_e2e = aggregate["e2e_total"]["median"]
    median_net = aggregate["sw_network"]["median"]
    non_net_fraction = max(0.0, median_e2e - median_net) / max(median_e2e, 1e-9)

    stage_sum = sum(aggregate[s]["median"] for s in STAGE_NAMES)
    stage_fractions_pct = {
        s: round(100.0 * aggregate[s]["median"] / max(stage_sum, 1e-9), 2) for s in STAGE_NAMES
    }

    payload: Dict[str, Any] = {
        "backend": backend,
        "run": run_metadata(),
        "profile": profile.as_dict(),
        "config": {
            "roi_size": [96, 96, 96],
            "sw_batch_size": profile.sw_batch_size,
            "overlap": profile.overlap,
            "transforms": profile.transforms,
            "io": profile.io,
            "inverse": profile.inverse,
            "warmup_volumes": warmup_volumes,
            "timed_repeats_per_case": timed_repeats,
            "num_cases": len(cases),
            "total_timed_passes": len(all_passes),
            "use_amp": use_amp,
            "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "gpu_telemetry_pre": telemetry_pre,
            "gpu_telemetry_post": telemetry_post,
        },
        "patch_forward_latency_ms": patch_bench,
        "stage_latency_ms": {k: aggregate[k] for k in SUMMARY_METRICS},
        "stage_fraction_pct": stage_fractions_pct,
        "throughput_volumes_per_s": round(1000.0 / max(median_e2e, 1e-6), 4),
        "peak_vram_mb": aggregate["peak_vram_mb"],
        "peak_reserved_vram_mb": aggregate["peak_reserved_vram_mb"],
        "amdahl": {
            "median_e2e_ms": round(median_e2e, 4),
            "median_sw_network_ms": round(median_net, 4),
            "median_non_network_ms": round(median_e2e - median_net, 4),
            "network_fraction_p": round(median_net / max(median_e2e, 1e-9), 6),
            "non_network_fraction_s": round(non_net_fraction, 6),
            "theoretical_max_e2e_speedup_1_over_s": round(1.0 / max(non_net_fraction, 1e-9), 4),
        },
        "per_case": per_case,
        "passes": all_passes,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    out_json_path = results_dir / f"bench_{backend}.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)

    print("-" * 96)
    for stg in SUMMARY_METRICS:
        s_obj = aggregate[stg]
        print(
            f"  {stg:<14}: median={s_obj['median']:9.2f} ms | IQR={s_obj['iqr']:8.2f} ms | "
            f"P95={s_obj['p95']:9.2f} ms | max={s_obj['max']:9.2f} ms"
        )
    print(f"  Peak Allocated VRAM (max): {aggregate['peak_vram_mb']['max']:.2f} MB | Throughput: {payload['throughput_volumes_per_s']:.2f} vol/s")
    print(f"Saved: {out_json_path}")
    print("=" * 96)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedTRT Multi-Repeat CUDA Stage Benchmark")
    parser.add_argument("--backend", type=str, required=True, choices=sorted(PROFILES))
    parser.add_argument("--cases", type=int, default=9, help="Number of held-out cases to benchmark")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warm-up volume passes")
    parser.add_argument("--repeats", type=int, default=10, help="Number of timed repeats per case")
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR), help="Output directory")
    args = parser.parse_args()
    run_benchmark(
        backend=args.backend,
        num_cases=args.cases,
        warmup_volumes=args.warmup,
        timed_repeats=args.repeats,
        results_dir=Path(args.results_dir),
    )
