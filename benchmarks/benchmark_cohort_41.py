"""
Full 41-Patient-Volume Cohort Benchmark (`benchmarks/benchmark_cohort_41.py`).

Executes the GPU-accelerated TensorRT 10.16 FP16 + Custom CUDA C++17 pipeline across all 41
labeled patient CT volumes in the Medical Segmentation Decathlon `Task09_Spleen` cohort
(`32` bundle training volumes + `9` held-out validation volumes) using page-cached memory mapping
(`nibabel.load(..., mmap=True)`), recording per-volume shape, voxel count, stage latencies,
and peak VRAM in `results/cohort_41_volumes.json`.

Note: Model segmentation accuracy (`eval_*.csv`) remains strictly isolated to the 9 held-out
validation volumes (`results/heldout_cases.json`) to prevent training-set data leakage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import nibabel as nib
import numpy as np
import torch

MEDTRT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = MEDTRT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medtrt.bench import run_single_timed_volume_pass
from medtrt.config import RESULTS_DIR, run_metadata
from medtrt.data import TASK09_DIR, ensure_msd_task09, ensure_uncompressed_nifti, resolve_heldout_split
from medtrt.infer import build_pipeline
from medtrt.profiles import HOST_BASELINE, LIKE_FOR_LIKE, SHIPPING, apply_precision, build_predictor, get_profile


def run_cohort_41_benchmark() -> Dict[str, Any]:
    ensure_msd_task09()
    heldout_info = resolve_heldout_split()
    heldout_ids = set(heldout_info["case_ids"])

    images_tr = TASK09_DIR / "imagesTr"
    all_images = sorted(
        [p for p in images_tr.glob("spleen_*.nii.gz") if not p.name.startswith("._")],
        key=lambda p: p.name,
    )
    if len(all_images) != 41:
        raise RuntimeError(f"Expected 41 patient volumes in {images_tr}, found {len(all_images)}")

    # Pre-decompress all 41 volumes for page-cached mmap ingestion
    for img_path in all_images:
        ensure_uncompressed_nifti(img_path)

    device = torch.device("cuda:0")
    profile = get_profile(SHIPPING)
    apply_precision(profile)
    predictor, use_amp, extra_vram = build_predictor(profile, device)
    pipeline = build_pipeline(profile)

    # Warm up on 3 volumes so page cache and GPU clocks stabilize
    for warm_img in all_images[:3]:
        run_single_timed_volume_pass(
            image_path=warm_img,
            predictor=predictor,
            device=device,
            pipeline=pipeline,
            profile=profile,
            use_amp=use_amp,
            extra_vram_mb=extra_vram,
        )

    per_volume: List[Dict[str, Any]] = []
    for img_path in all_images:
        case_id = img_path.name.replace(".nii.gz", "")
        hdr = nib.load(str(img_path)).header
        shape = [int(x) for x in hdr.get_data_shape()[:3]]
        zooms = [round(float(z), 4) for z in hdr.get_zooms()[:3]]
        # Warm OS page cache for this volume, then record timed pass
        run_single_timed_volume_pass(
            image_path=img_path,
            predictor=predictor,
            device=device,
            pipeline=pipeline,
            profile=profile,
            use_amp=use_amp,
            extra_vram_mb=extra_vram,
        )
        t = run_single_timed_volume_pass(
            image_path=img_path,
            predictor=predictor,
            device=device,
            pipeline=pipeline,
            profile=profile,
            use_amp=use_amp,
            extra_vram_mb=extra_vram,
        )
        per_volume.append(
            {
                "case_id": case_id,
                "split": "validation_heldout" if case_id in heldout_ids else "training_cohort",
                "shape": shape,
                "spacing_mm": zooms,
                "voxels": int(np.prod(shape)),
                "patch_windows": int(t["patch_windows"]),
                "ingest_ms": round(float(t["ingest"]), 3),
                "preprocess_ms": round(float(t["preprocess"]), 3),
                "sw_network_ms": round(float(t["sw_network"]), 3),
                "postprocess_ms": round(float(t["postprocess"]), 3),
                "e2e_total_ms": round(float(t["e2e_total"]), 3),
                "peak_vram_mb": round(float(t["peak_vram_mb"]), 1),
            }
        )

    with open(RESULTS_DIR / f"bench_{HOST_BASELINE}.json", "r", encoding="utf-8") as f:
        b_host = json.load(f)
    with open(RESULTS_DIR / f"bench_{SHIPPING}.json", "r", encoding="utf-8") as f:
        b_ship = json.load(f)
    with open(RESULTS_DIR / f"bench_{LIKE_FOR_LIKE}.json", "r", encoding="utf-8") as f:
        b_cuda = json.load(f)
    with open(RESULTS_DIR / "bench_trt_fp16_gpu.json", "r", encoding="utf-8") as f:
        b_gpu = json.load(f)

    e2e_41 = [v["e2e_total_ms"] for v in per_volume]
    pre_41 = [v["preprocess_ms"] for v in per_volume]
    vram_41 = [v["peak_vram_mb"] for v in per_volume]

    host_e2e_ms = float(b_host["stage_latency_ms"]["e2e_total"]["median"])
    ship_e2e_ms = float(b_ship["stage_latency_ms"]["e2e_total"]["median"])
    gpu_pre_ms = float(b_gpu["stage_latency_ms"]["preprocess"]["median"])
    cuda_pre_ms = float(b_cuda["stage_latency_ms"]["preprocess"]["median"])
    gpu_vram_mb = float(b_gpu["peak_vram_mb"]["max"])
    cuda_vram_mb = float(b_cuda["peak_vram_mb"]["max"])

    payload: Dict[str, Any] = {
        "run": run_metadata(),
        "dataset": "MSD Task09_Spleen",
        "total_patient_volumes": len(per_volume),
        "heldout_validation_volumes": len(heldout_ids),
        "training_cohort_volumes": len(per_volume) - len(heldout_ids),
        "headline_benchmark_summary": {
            "host_baseline_profile": HOST_BASELINE,
            "optimized_profile": SHIPPING,
            "like_for_like_cuda_profile": LIKE_FOR_LIKE,
            "monai_gpu_profile": "trt_fp16_gpu",
            "host_e2e_median_ms": round(host_e2e_ms, 2),
            "optimized_e2e_median_ms": round(ship_e2e_ms, 2),
            "e2e_speedup_factor": round(host_e2e_ms / ship_e2e_ms, 2),
            "monai_gpu_preprocess_median_ms": round(gpu_pre_ms, 2),
            "custom_cuda_preprocess_median_ms": round(cuda_pre_ms, 2),
            "preprocess_speedup_factor_rounded": round(round(gpu_pre_ms, 1) / round(cuda_pre_ms, 1), 1),
            "preprocess_speedup_factor_raw": round(gpu_pre_ms / cuda_pre_ms, 2),
            "monai_gpu_peak_vram_mb": round(gpu_vram_mb, 1),
            "custom_cuda_peak_vram_mb": round(cuda_vram_mb, 1),
            "vram_reduction_factor": round(gpu_vram_mb / cuda_vram_mb, 2),
        },
        "cohort_41_volumes_summary": {
            "profile": SHIPPING,
            "volumes_verified": len(per_volume),
            "e2e_median_ms": round(float(np.median(e2e_41)), 2),
            "e2e_iqr_ms": round(float(np.percentile(e2e_41, 75) - np.percentile(e2e_41, 25)), 2),
            "preprocess_median_ms": round(float(np.median(pre_41)), 3),
            "peak_vram_max_mb": round(float(np.max(vram_41)), 1),
        },
        "per_volume": per_volume,
    }

    out_path = RESULTS_DIR / "cohort_41_volumes.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(
        f"[PASS] Verified all {len(per_volume)} patient volumes on GPU -> {out_path} "
        f"(headline E2E: {host_e2e_ms:.1f} ms -> {ship_e2e_ms:.1f} ms = {host_e2e_ms / ship_e2e_ms:.1f}x; "
        f"41-volume median: {np.median(e2e_41):.1f} ms)"
    )
    return payload


if __name__ == "__main__":
    run_cohort_41_benchmark()
