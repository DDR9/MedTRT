"""
Phase 2 Multi-Repeat Benchmarking Driver (`src/medtrt/bench.py`).

Responsibilities:
- Isolate pure patch forward latency (`(B, 1, 96, 96, 96)` for `B in {1, 2, 4}`) from
  full-volume end-to-end latency across held-out cases.
- Measure all 6 pipeline stages (`load`, `preprocess`, `h2d`, `sw_network`, `sw_aggregate`, `postprocess`)
  using `torch.cuda.Event(enable_timing=True)` and NVTX markers (`torch.cuda.nvtx.range_push/pop`).
- Execute warm-up (3 full volumes) followed by N >= 10 timed repeats per case.
- Track `torch.cuda.max_memory_allocated()` with `torch.cuda.reset_peak_memory_stats()` per pass.
- Write `results/bench_<backend>.json` with median, IQR, mean, std, Amdahl's Law decomposition, and VRAM.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)

from medtrt.config import MEDTRT_ROOT, RESULTS_DIR
from medtrt.data import (
    HELDOUT_JSON_PATH,
    ensure_uncompressed_nifti,
    load_nifti_mmap,
    resolve_heldout_split,
)
from medtrt.infer import (
    build_inference_postprocessing,
    build_inference_preprocessing,
    build_optimized_gpu_postprocessing,
    build_sliding_window_inferer,
)
from medtrt.model import load_spleen_unet
from medtrt.timing import (
    STAGE_NAMES,
    CudaStageTimer,
    InstrumentedPredictor,
    compute_distribution_stats,
)


def build_split_load_and_preprocess(
    gpu_transforms: bool = False,
) -> Tuple[Callable[[Dict[str, Any]], Dict[str, Any]], Compose, Compose]:
    """
    Return `(load_tf, pre_tf, full_pre_tf)` so `load` and `preprocess` can be timed
    as distinct stages while `pre_tf` is passed to `Invertd` for exact inverse transform.
    When `gpu_transforms=True` (Stage 2 Optimization), `load_tf` uses `load_nifti_mmap`
    (`nibabel.load(..., mmap=True)` on uncompressed `.nii`) and `pre_tf` executes on `cuda:0`.
    """
    if gpu_transforms:
        load_tf = load_nifti_mmap
        pre_tf = Compose(
            [
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-57.0,
                    a_max=164.0,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
            ]
        )
    else:
        load_tf = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
            ]
        )
        pre_tf = Compose(
            [
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-57.0,
                    a_max=164.0,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                EnsureTyped(keys=["image"]),
            ]
        )
    full_pre_tf = build_inference_preprocessing()
    return load_tf, pre_tf, full_pre_tf


def benchmark_io_ingestion(
    cases: List[Dict[str, str]],
    repeats: int = 5,
) -> Dict[str, Any]:
    """
    Compare NIfTI disk I/O ingestion latency across held-out cases:
    1. With on-the-fly gzip decompression (`.nii.gz` via `LoadImaged` + `EnsureChannelFirstd`).
    2. Without on-the-fly gzip decompression (uncompressed `.nii` via `nibabel.load(..., mmap=True)`).
    """
    import time

    gzip_tf = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
        ]
    )
    gzip_latencies_ms: List[float] = []
    mmap_latencies_ms: List[float] = []

    for c_item in cases:
        gz_path = MEDTRT_ROOT / c_item["image"]
        nii_path = ensure_uncompressed_nifti(gz_path)
        # Warm up OS page cache once per file
        _ = load_nifti_mmap({"image": str(nii_path)})
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = gzip_tf({"image": str(gz_path)})
            t1 = time.perf_counter()
            gzip_latencies_ms.append((t1 - t0) * 1000.0)

            t2 = time.perf_counter()
            _ = load_nifti_mmap({"image": str(nii_path)})
            t3 = time.perf_counter()
            mmap_latencies_ms.append((t3 - t2) * 1000.0)

    gzip_stats = compute_distribution_stats(gzip_latencies_ms)
    mmap_stats = compute_distribution_stats(mmap_latencies_ms)
    speedup = gzip_stats["median"] / max(mmap_stats["median"], 1e-6)
    return {
        "gzip_on_the_fly_ms": gzip_stats,
        "uncompressed_mmap_ms": mmap_stats,
        "io_speedup_factor": round(float(speedup), 2),
    }


def benchmark_pure_patch_forward(
    predictor: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    use_amp: bool = False,
    batch_sizes: Tuple[int, ...] = (1, 2, 4),
    warmup_iters: int = 10,
    timed_iters: int = 50,
    extra_vram_mb: float = 0.0,
) -> Dict[str, Any]:
    """
    Measure isolated 3D patch forward pass latency `(B, 1, 96, 96, 96)` on the GPU
    without sliding-window stitching or host I/O.
    """
    results: Dict[str, Any] = {}
    for b in batch_sizes:
        patch = torch.randn((b, 1, 96, 96, 96), dtype=torch.float32, device=device)

        with torch.no_grad():
            for _ in range(warmup_iters):
                if use_amp:
                    with torch.autocast("cuda", dtype=torch.float16):
                        _ = predictor(patch)
                else:
                    _ = predictor(patch)
        torch.cuda.synchronize(device)

        torch.cuda.reset_peak_memory_stats(device)
        latencies_ms: List[float] = []
        with torch.no_grad():
            for _ in range(timed_iters):
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                torch.cuda.nvtx.range_push(f"patch_forward_b{b}")
                start_ev.record()
                if use_amp:
                    with torch.autocast("cuda", dtype=torch.float16):
                        _ = predictor(patch)
                else:
                    _ = predictor(patch)
                end_ev.record()
                end_ev.synchronize()
                torch.cuda.nvtx.range_pop()
                latencies_ms.append(float(start_ev.elapsed_time(end_ev)))

        peak_vram_mb = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) + extra_vram_mb
        stats = compute_distribution_stats(latencies_ms)
        stats["per_patch_median_ms"] = round(stats["median"] / b, 4)
        stats["peak_vram_mb"] = round(float(peak_vram_mb), 2)
        results[f"batch_{b}"] = stats

        del patch
        torch.cuda.empty_cache()

    return results


def run_single_timed_volume_pass(
    image_path: Path,
    predictor: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    load_tf: Compose,
    pre_tf: Compose,
    post_tf: Compose,
    sw_batch_size: int = 4,
    overlap: float = 0.5,
    use_amp: bool = False,
    extra_vram_mb: float = 0.0,
    gpu_transforms: bool = False,
) -> Dict[str, float]:
    """
    Execute a single full-volume inference pass instrumented with `CudaStageTimer`
    and NVTX markers across `load`, `preprocess`, `h2d`, `sw_network`, `sw_aggregate`, `postprocess`.
    When `gpu_transforms=True` (Phase 5 Targeted Optimization), `h2d` transfers the raw loaded volume
    to `cuda:0` immediately after `load`, so both `preprocess` (`Spacingd` + `ScaleIntensityRanged`)
    and `postprocess` (`Activationsd` + `Invertd` + `AsDiscreted`) execute on the GPU.
    """
    inferer = build_sliding_window_inferer(
        roi_size=(96, 96, 96),
        sw_batch_size=sw_batch_size,
        overlap=overlap,
    )
    timer = CudaStageTimer()
    inst_predictor = InstrumentedPredictor(
        predictor=predictor,
        timer=timer,
        use_amp=use_amp,
    )

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    timer.start_e2e()

    # Stage 1: load
    with timer.stage("load", sync_on_exit=True):
        data: Dict[str, Any] = load_tf({"image": str(image_path)})

    if gpu_transforms:
        # Stage 2: h2d (move raw tensor to GPU before spatial resample & intensity scaling)
        with timer.stage("h2d", sync_on_exit=True):
            data["image"] = data["image"].to(device, non_blocking=False)

        # Stage 3: preprocess on GPU
        with timer.stage("preprocess", sync_on_exit=True):
            data = pre_tf(data)
            inputs = data["image"].unsqueeze(0)
    else:
        # Stage 2: preprocess on CPU
        with timer.stage("preprocess", sync_on_exit=True):
            data = pre_tf(data)

        # Stage 3: h2d
        with timer.stage("h2d", sync_on_exit=True):
            inputs = data["image"].unsqueeze(0).to(device, non_blocking=False)

    # Stage 4 & 5: sw_aggregate & sw_network
    with timer.stage("_sw_total", sync_on_exit=True):
        torch.cuda.nvtx.range_push("sw_aggregate")
        try:
            with torch.no_grad():
                logits = inferer(inputs=inputs, network=inst_predictor)
        finally:
            torch.cuda.nvtx.range_pop()

    # Stage 6: postprocess (AsDiscreted -> uint8 -> GPU Invertd(nearest_interp=True))
    with timer.stage("postprocess", sync_on_exit=True):
        if gpu_transforms:
            data["pred"] = logits[0].detach()
            out_data = post_tf(data)
            _ = out_data["pred"].cpu()
        else:
            data["pred"] = logits[0].detach().float().cpu()
            _ = post_tf(data)

    timer.stop_e2e()
    peak_vram_mb = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) + extra_vram_mb

    timings = timer.resolve_timings_ms()
    timings["peak_vram_mb"] = float(peak_vram_mb)
    return timings


def run_benchmark(
    backend: str = "pytorch_fp32",
    predictor: Callable[[torch.Tensor], torch.Tensor] | None = None,
    num_cases: int | None = None,
    warmup_volumes: int = 3,
    timed_repeats: int = 10,
    sw_batch_size: int | None = None,
    overlap: float | None = None,
    extra_vram_mb: float = 0.0,
) -> Dict[str, Any]:
    """
    Execute full Phase 2 / Stage 2 benchmark suite for `backend` and write `results/bench_<backend>.json`.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable. Stopping per Ground Rule 6.")

    if not HELDOUT_JSON_PATH.exists():
        heldout_info = resolve_heldout_split()
    else:
        with open(HELDOUT_JSON_PATH, "r", encoding="utf-8") as f:
            heldout_info = json.load(f)

    device = torch.device("cuda:0")
    use_amp = False
    gpu_transforms = False

    if sw_batch_size is None:
        sw_batch_size = 8 if backend == "trt_fp16_optimized" else 4
    if overlap is None:
        overlap = 0.25 if backend == "trt_fp16_optimized" else 0.5

    if backend == "pytorch_fp32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        if predictor is None:
            predictor = load_spleen_unet(device)
    elif backend == "pytorch_fp32_strict":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        if predictor is None:
            predictor = load_spleen_unet(device)
    elif backend == "pytorch_amp":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        use_amp = True
        if predictor is None:
            predictor = load_spleen_unet(device)
    elif backend in ("trt_fp16", "trt_fp16_optimized"):
        if backend == "trt_fp16_optimized":
            gpu_transforms = True
        if predictor is None:
            from medtrt.trt_runner import TRTRunner

            trt_runner = TRTRunner(device=device)
            predictor = trt_runner
            extra_vram_mb = trt_runner.trt_vram_mb
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    cases = heldout_info["cases"]
    if num_cases is not None and num_cases > 0:
        cases = cases[:num_cases]

    if gpu_transforms:
        for c_item in cases:
            ensure_uncompressed_nifti(MEDTRT_ROOT / c_item["image"])

    print("=" * 92)
    print(
        f"MEDTRT BENCHMARK: backend={backend} | cases={len(cases)} | "
        f"warmup={warmup_volumes} | repeats={timed_repeats} | "
        f"sw_batch_size={sw_batch_size} | overlap={overlap}"
    )
    print("=" * 92)

    # 1. Pure patch forward benchmark
    patch_batch_sizes = (1, 2, 4, 8) if sw_batch_size == 8 else (1, 2, 4)
    print(f"[1/3] Benchmarking isolated 3D patch forward pass (96x96x96, batches={patch_batch_sizes})...")
    patch_bench = benchmark_pure_patch_forward(
        predictor=predictor,
        device=device,
        use_amp=use_amp,
        batch_sizes=patch_batch_sizes,
        warmup_iters=10,
        timed_iters=50,
        extra_vram_mb=extra_vram_mb,
    )
    for b_key, b_val in patch_bench.items():
        print(
            f"      {b_key}: median={b_val['median']:.2f} ms (IQR={b_val['iqr']:.2f} ms, "
            f"per-patch={b_val['per_patch_median_ms']:.2f} ms, VRAM={b_val['peak_vram_mb']:.1f} MB)"
        )

    io_comparison: Dict[str, Any] = {}
    if gpu_transforms:
        print("[1b/3] Benchmarking Disk I/O: on-the-fly gzip (.nii.gz) vs. uncompressed mmap (.nii)...")
        io_comparison = benchmark_io_ingestion(cases, repeats=5)
        gz_m = io_comparison["gzip_on_the_fly_ms"]
        mm_m = io_comparison["uncompressed_mmap_ms"]
        print(
            f"      gzip (.nii.gz) load      : median={gz_m['median']:.2f} ms [IQR={gz_m['iqr']:.2f} ms] "
            f"(mean={gz_m['mean']:.2f} ± {gz_m['std']:.2f} ms)"
        )
        print(
            f"      uncompressed mmap (.nii) : median={mm_m['median']:.2f} ms [IQR={mm_m['iqr']:.2f} ms] "
            f"(mean={mm_m['mean']:.2f} ± {mm_m['std']:.2f} ms) -> {io_comparison['io_speedup_factor']:.2f}x faster I/O"
        )

    load_tf, pre_tf, _ = build_split_load_and_preprocess(gpu_transforms=gpu_transforms)
    post_tf = (
        build_optimized_gpu_postprocessing(pre_tf)
        if gpu_transforms
        else build_inference_postprocessing(pre_tf)
    )

    # 2. Warm-up: 3 full volumes
    print(f"[2/3] Executing {warmup_volumes} full-volume warm-up passes...")
    for w_idx in range(warmup_volumes):
        w_case = cases[w_idx % len(cases)]
        w_img_path = MEDTRT_ROOT / w_case["image"]
        _ = run_single_timed_volume_pass(
            image_path=w_img_path,
            predictor=predictor,
            device=device,
            load_tf=load_tf,
            pre_tf=pre_tf,
            post_tf=post_tf,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            use_amp=use_amp,
            extra_vram_mb=extra_vram_mb,
            gpu_transforms=gpu_transforms,
        )

    # 3. Timed repeats across cases
    print(f"[3/3] Executing {timed_repeats} timed repeats across {len(cases)} held-out case(s)...")
    all_passes: List[Dict[str, Any]] = []
    per_case_summary: Dict[str, Any] = {}

    metric_keys = list(STAGE_NAMES) + ["sw_total", "non_network", "e2e_total", "peak_vram_mb"]

    for c_item in cases:
        case_id = c_item["case_id"]
        img_path = MEDTRT_ROOT / c_item["image"]
        case_passes: List[Dict[str, float]] = []

        for r_idx in range(timed_repeats):
            pass_metrics = run_single_timed_volume_pass(
                image_path=img_path,
                predictor=predictor,
                device=device,
                load_tf=load_tf,
                pre_tf=pre_tf,
                post_tf=post_tf,
                sw_batch_size=sw_batch_size,
                overlap=overlap,
                use_amp=use_amp,
                extra_vram_mb=extra_vram_mb,
                gpu_transforms=gpu_transforms,
            )
            case_passes.append(pass_metrics)
            all_passes.append({"case_id": case_id, "repeat": r_idx, **pass_metrics})

        c_stats = {
            k: compute_distribution_stats([p[k] for p in case_passes])
            for k in metric_keys
        }
        c_stats["patch_calls"] = int(case_passes[0]["patch_calls"])
        per_case_summary[case_id] = c_stats
        print(
            f"      {case_id:<10} | E2E median={c_stats['e2e_total']['median']:7.2f} ms "
            f"(IQR={c_stats['e2e_total']['iqr']:5.2f}) | "
            f"sw_net={c_stats['sw_network']['median']:6.2f} ms | "
            f"sw_agg={c_stats['sw_aggregate']['median']:5.2f} ms | "
            f"VRAM={c_stats['peak_vram_mb']['max']:6.1f} MB"
        )

    aggregate_stats = {
        k: compute_distribution_stats([p[k] for p in all_passes])
        for k in metric_keys
    }

    # Compute Amdahl's Law parameters from median E2E and median sw_network
    median_e2e = aggregate_stats["e2e_total"]["median"]
    median_net = aggregate_stats["sw_network"]["median"]
    median_non_net = max(0.0, median_e2e - median_net)
    non_net_fraction_s = median_non_net / max(median_e2e, 1e-9)
    net_fraction_p = median_net / max(median_e2e, 1e-9)
    theoretical_max_speedup = 1.0 / max(non_net_fraction_s, 1e-9)

    amdahl_summary = {
        "median_e2e_ms": round(median_e2e, 4),
        "median_sw_network_ms": round(median_net, 4),
        "median_non_network_ms": round(median_non_net, 4),
        "network_fraction_p": round(net_fraction_p, 6),
        "non_network_fraction_s": round(non_net_fraction_s, 6),
        "theoretical_max_e2e_speedup_1_over_s": round(theoretical_max_speedup, 4),
    }

    payload = {
        "backend": backend,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config": {
            "roi_size": [96, 96, 96],
            "sw_batch_size": sw_batch_size,
            "overlap": overlap,
            "warmup_volumes": warmup_volumes,
            "timed_repeats_per_case": timed_repeats,
            "num_cases": len(cases),
            "total_timed_passes": len(all_passes),
            "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "use_amp": use_amp,
            "gpu_transforms": gpu_transforms,
            "reader_mode": "uncompressed_mmap_nii" if gpu_transforms else "gzip_nii_gz",
            "postprocess_dtype": "uint8_cuda_nearest" if gpu_transforms else "float32_cpu_trilinear",
        },
        "patch_forward_latency_ms": patch_bench,
        "io_comparison_ms": io_comparison,
        "stage_latency_ms": {
            k: aggregate_stats[k]
            for k in STAGE_NAMES + ("sw_total", "non_network", "e2e_total")
        },
        "peak_vram_mb": aggregate_stats["peak_vram_mb"],
        "amdahl": amdahl_summary,
        "per_case": per_case_summary,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json_path = RESULTS_DIR / f"bench_{backend}.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("-" * 92)
    print(f"STAGE BREAKDOWN ACROSS {len(all_passes)} TIMED PASSES ({backend}):")
    for stg in STAGE_NAMES + ("sw_total", "non_network", "e2e_total"):
        s_obj = aggregate_stats[stg]
        print(
            f"  - {stg:<14}: median = {s_obj['median']:8.2f} ms | "
            f"IQR = {s_obj['iqr']:6.2f} ms | mean = {s_obj['mean']:8.2f} ± {s_obj['std']:6.2f} ms"
        )
    print(
        f"  - {'peak_vram_mb':<14}: median = {aggregate_stats['peak_vram_mb']['median']:8.2f} MB | "
        f"max = {aggregate_stats['peak_vram_mb']['max']:6.2f} MB"
    )
    print(
        f"  - Amdahl's Law  : Network Fraction p = {net_fraction_p*100:.2f}% | "
        f"Non-Network s = {non_net_fraction_s*100:.2f}% | Max E2E Speedup (1/s) = {theoretical_max_speedup:.2f}x"
    )
    print(f"Saved Benchmark JSON: {out_json_path}")
    print("=" * 92)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedTRT Multi-Repeat CUDA Stage Benchmark")
    parser.add_argument("--backend", type=str, default="pytorch_fp32")
    parser.add_argument("--cases", type=int, default=9, help="Number of held-out cases to benchmark")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warm-up volume passes")
    parser.add_argument("--repeats", type=int, default=10, help="Number of timed repeats per case")
    parser.add_argument("--sw-batch-size", type=int, default=None, help="Sliding window batch size")
    parser.add_argument("--overlap", type=float, default=None, help="Sliding window overlap")
    args = parser.parse_args()

    res = run_benchmark(
        backend=args.backend,
        num_cases=args.cases,
        warmup_volumes=args.warmup,
        timed_repeats=args.repeats,
        sw_batch_size=args.sw_batch_size,
        overlap=args.overlap,
    )

    if args.backend == "pytorch_fp32" and args.cases == 9 and args.repeats >= 10:
        state_path = RESULTS_DIR / "STATE.json"
        state = {}
        if state_path.exists():
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
        state.update(
            {
                "current_phase": "PHASE_2",
                "phase_2_status": "PASS",
                "bench_pytorch_fp32_json": "results/bench_pytorch_fp32.json",
                "phase_2_median_e2e_ms": res["stage_latency_ms"]["e2e_total"]["median"],
                "phase_2_median_sw_network_ms": res["stage_latency_ms"]["sw_network"]["median"],
                "phase_2_peak_vram_mb": res["peak_vram_mb"]["max"],
                "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        )
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
