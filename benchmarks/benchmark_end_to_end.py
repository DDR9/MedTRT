"""
End-to-End Pipeline & Controlled Ablation Benchmark (`benchmarks/benchmark_end_to_end.py`).

Implements Phase 2.3 & Phase 11 (Experiment F) one-change-at-a-time ablation study:
1. `pytorch_fp32_strict` (Baseline Strict IEEE FP32, Host I/O & Host Transforms, `overlap=0.5, B=4`)
2. `pytorch_fp32` (Baseline Default TF32, Host I/O & Host Transforms, `overlap=0.5, B=4`)
3. `trt_fp16` (+ TensorRT FP16 only, Host I/O & Host Transforms, `overlap=0.5, B=4`)
4. `trt_fp16_gpu_gzip` (+ MONAI GPU Transforms, Gzip I/O, `overlap=0.5, B=4`)
5. `trt_fp16_gpu` (+ Memory-Mapped Uncompressed `.nii` I/O, MONAI GPU Transforms, `overlap=0.5, B=4`)
6. `trt_fp16_cuda` (+ Custom CUDA C++ Fused Pre/Postprocessing Kernels, `overlap=0.5, B=4`)
7. `trt_fp16_optimized` (+ Pareto Sliding-Window Profile `overlap=0.25, B=8` — Fully Optimized Pipeline)
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

import nibabel as nib
import numpy as np
import torch

from benchmarks.common import save_benchmark_artifact
from medtrt.bench import run_single_timed_volume_pass
from medtrt.config import MEDTRT_ROOT, RESULTS_DIR
from medtrt.data import HELDOUT_JSON_PATH, ensure_uncompressed_nifti
from medtrt.evaluate import compute_foreground_dice
from medtrt.infer import build_pipeline, infer_volume
from medtrt.profiles import Profile, apply_precision
from medtrt.timing import STAGE_NAMES, compute_distribution_stats

ABLATION_CONFIGS: List[Profile] = [
    Profile(
        "1_pytorch_fp32_strict_host",
        "Original PyTorch FP32 (Strict IEEE, Host Path)",
        "pytorch",
        "fp32",
        io="gzip",
        transforms="cpu",
        inverse="trilinear",
        overlap=0.5,
        sw_batch_size=4,
    ),
    Profile(
        "2_pytorch_tf32_host",
        "Original PyTorch TF32 (Default, Host Path)",
        "pytorch",
        "tf32",
        io="gzip",
        transforms="cpu",
        inverse="trilinear",
        overlap=0.5,
        sw_batch_size=4,
    ),
    Profile(
        "3_trt_fp16_host_only",
        "+ TensorRT 10.x FP16 Only (Host Path)",
        "trt",
        "fp16",
        io="gzip",
        transforms="cpu",
        inverse="trilinear",
        overlap=0.5,
        sw_batch_size=4,
    ),
    Profile(
        "4_trt_fp16_monai_gpu_gzip",
        "+ MONAI GPU Pre/Postprocessing (Gzip I/O)",
        "trt",
        "fp16",
        io="gzip",
        transforms="gpu",
        inverse="trilinear",
        overlap=0.5,
        sw_batch_size=4,
    ),
    Profile(
        "5_trt_fp16_monai_gpu_mmap",
        "+ Memory-Mapped .nii Ingestion (MONAI GPU Path)",
        "trt",
        "fp16",
        io="mmap",
        transforms="gpu",
        inverse="trilinear",
        overlap=0.5,
        sw_batch_size=4,
    ),
    Profile(
        "6_trt_fp16_custom_cuda_b4",
        "+ Custom CUDA C++ Fused Pre/Post Kernels (overlap=0.5, B=4)",
        "trt",
        "fp16",
        io="mmap",
        transforms="cuda",
        inverse="trilinear",
        overlap=0.5,
        sw_batch_size=4,
    ),
    Profile(
        "7_trt_fp16_fully_optimized",
        "+ Pareto Sliding Windows (overlap=0.25, B=8, Fully Optimized)",
        "trt",
        "fp16",
        io="mmap",
        transforms="cuda",
        inverse="trilinear",
        overlap=0.25,
        sw_batch_size=8,
    ),
]


def run_e2e_ablation_benchmark(repeats: int = 3, warmup_volumes: int = 2) -> Dict[str, Any]:
    with open(HELDOUT_JSON_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)["cases"]

    for c in cases:
        ensure_uncompressed_nifti(MEDTRT_ROOT / c["image"])

    device = torch.device("cuda:0")
    from medtrt.model import load_spleen_unet
    from medtrt.trt_runner import TRTRunner

    pt_model = load_spleen_unet(device, fold_bn=True)
    trt_runner = TRTRunner(device=device)
    predictors: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "pytorch": pt_model,
        "trt": trt_runner,
    }
    extra_vram_map = {"pytorch": 0.0, "trt": trt_runner.trt_vram_mb}

    gt_masks = {
        c["case_id"]: np.asarray(nib.load(str(MEDTRT_ROOT / c["label"])).dataobj).astype(np.uint8)
        for c in cases
    }

    rungs: Dict[str, Any] = {}
    baseline_e2e_ms: float | None = None
    baseline_strict_e2e_ms: float | None = None

    for prof in ABLATION_CONFIGS:
        apply_precision(prof)
        predictor = predictors[prof.engine]
        extra_vram = extra_vram_map[prof.engine]
        pipeline = build_pipeline(prof)

        def one_pass(case_obj: Dict[str, str]) -> Dict[str, float]:
            return run_single_timed_volume_pass(
                MEDTRT_ROOT / case_obj["image"],
                predictor,
                device,
                pipeline,
                prof,
                use_amp=False,
                extra_vram_mb=extra_vram,
            )

        for w_idx in range(warmup_volumes):
            one_pass(cases[w_idx % len(cases)])

        passes = [one_pass(c) for c in cases for _ in range(repeats)]
        dices = []
        for c in cases:
            mask = infer_volume(MEDTRT_ROOT / c["image"], predictor, prof, device)["pred_mask_3d"].astype(np.uint8)
            dices.append(compute_foreground_dice(mask, gt_masks[c["case_id"]]))

        stage_keys = STAGE_NAMES + ("ingest", "sw_total", "non_network", "e2e_total", "wall_ms", "peak_vram_mb")
        summary = {k: compute_distribution_stats([p[k] for p in passes]) for k in stage_keys}

        med_e2e = summary["e2e_total"]["median"]
        if baseline_strict_e2e_ms is None:
            baseline_strict_e2e_ms = med_e2e
        if prof.name == "2_pytorch_tf32_host":
            baseline_e2e_ms = med_e2e

        ref_base = baseline_e2e_ms if baseline_e2e_ms is not None else baseline_strict_e2e_ms
        speedup_vs_tf32 = ref_base / max(med_e2e, 1e-6)
        speedup_vs_strict = baseline_strict_e2e_ms / max(med_e2e, 1e-6)
        latency_reduction_pct = 100.0 * (1.0 - (med_e2e / max(ref_base, 1e-6)))

        rungs[prof.name] = {
            "label": prof.label,
            "profile": prof.as_dict(),
            "stages_ms": summary,
            "mean_dice": round(float(np.mean(dices)), 6),
            "std_dice": round(float(np.std(dices, ddof=1)), 6),
            "speedup_vs_tf32_baseline": round(speedup_vs_tf32, 2),
            "speedup_vs_strict_fp32": round(speedup_vs_strict, 2),
            "latency_reduction_pct_vs_tf32": round(latency_reduction_pct, 2),
        }
        print(
            f"{prof.name:<28} | e2e={med_e2e:8.2f} ms | sw_net={summary['sw_network']['median']:6.2f} ms "
            f"| post={summary['postprocess']['median']:7.2f} ms | VRAM={summary['peak_vram_mb']['max']:6.1f} MB "
            f"| Dice={rungs[prof.name]['mean_dice']:.6f} | Speedup={speedup_vs_tf32:.2f}x",
            flush=True,
        )

    payload = {"repeats_per_case": repeats, "rungs": rungs}
    out_path = save_benchmark_artifact("benchmark_end_to_end.json", payload)
    print(f"[PASS] Saved end-to-end ablation study to {out_path}")
    return payload


if __name__ == "__main__":
    run_e2e_ablation_benchmark()
