# MedTRT Hardware Telemetry & Evaluation Artifacts (`results/`)

This directory contains **29 hardware-verified JSON and CSV artifacts** recorded on an **NVIDIA GeForce RTX 5080 Laptop GPU** (`NVIDIA Blackwell`, `sm_120`, 16 GB GDDR7, Driver `591.44`, CUDA `12.8`, TensorRT `10.16.1.11`).

Every measurement file is stamped with SHA-256 code fingerprint `e9d4f060aaf1b130` and engine fingerprint `28d585aaed46e401`. You can audit the mathematical consistency of all files offline without a GPU by running:

```bash
python -m medtrt.verify_claims
```

---

## Artifact Directory Guide

| File Pattern | Count | Description |
| :--- | :---: | :--- |
| **`bench_<profile>.json`** | 13 | Per-pass and per-case CUDA Event stage timings (`load`, `h2d`, `ingest`, `preprocess`, `sw_network`, `sw_aggregate`, `postprocess`, `e2e_total`, `wall_ms`), patch forward latencies (`B=1, 2, 4, 8`), Amdahl's Law fractions, and peak allocated VRAM across all 13 execution profiles. |
| **`eval_<profile>.csv` & `.meta.json`** | 26 | Patient-level quantitative segmentation metrics (`Dice`, `IoU`, `Sensitivity`, `Specificity`, `Precision`, `Volumetric Similarity`, `HD95`, `ASSD`, physical predicted vs. ground-truth spleen volume in $\text{cm}^3$) across the 9 held-out MSD validation CT scans. |
| **`validation.csv`** | 1 | Consolidated side-by-side patient-level comparison table across strict FP32 reference, matched CUDA (`trt_fp16_cuda`), and fast (`trt_fp16_optimized`) profiles. |
| **`bland_altman.json`** | 1 | Dual Bland–Altman volumetric agreement statistics (model vs. ground truth and candidate backend vs. strict IEEE-754 FP32 reference). |
| **`cohort_41_volumes.json`** | 1 | Full-cohort GPU execution and latency/VRAM telemetry across all 41 labeled 3D CT volumes in Medical Segmentation Decathlon `Task09_Spleen`. |
| **`external_flare22_validation.json`** | 1 | Unseen external multi-center validation across 12 abdominal 3D CT scans from MICCAI `FLARE22`, cold vs. warm I/O benchmarks, and formal paired Two One-Sided Tests (TOST) equivalence ($N=9$, $N=12$, and $N=21$ pooled). |
| **`ablation.json` & `benchmark_end_to_end.json`** | 2 | Controlled one-factor-at-a-time pipeline ablation isolating the latency and accuracy impact of each optimization step. |
| **`benchmark_preprocessing.json`** | 1 | Isolated benchmarks of `.nii.gz` gzip vs. `.nii` `mmap`, pageable vs. pinned asynchronous H2D DMA, and custom CUDA C++ `fused_preprocess_affine_trilinear_hu_f32` vs. MONAI GPU/CPU. |
| **`benchmark_postprocessing.json`** | 1 | Isolated benchmarks of custom CUDA C++ `fused_invert_trilinear_threshold_u8` vs. MONAI GPU/CPU trilinear and nearest-neighbor inversion. |
| **`benchmark_scaling.json`** | 1 | Custom CUDA C++ kernel scaling across 3D volume resolutions from `4.19M` to `50.33M` voxels. |
| **`benchmark_memory.json`** | 1 | Static model weight/context footprints and dynamic incremental VRAM allocations across transform paths. |
| **`benchmark_correctness.json`** | 1 | NVRTC `sm_120` SASS register/occupancy telemetry (`cuFuncGetAttribute`), patch-level numerical equivalence, and strided/tail edge-case checks. |
| **`probes.json`** | 1 | Targeted hardware probes: sustained network-only loops, laptop GPU DVFS clock-state idle inflation (`1.13x`), storage ingestion regimes, `float64` coordinate grid inspection, and decoder `BatchNorm3d` folding analysis. |
| **`sweep.csv`** | 1 | Sliding-window `overlap` ($0.25, 0.50, 0.75$) $\times$ `sw_batch_size` ($1, 2, 4, 8$) Pareto grid sweep. |
| **`failure_analysis.json`** | 1 | Voxel-level polar-dome and boundary error decomposition plus Spearman correlation statistics across slice thickness and organ volume. |
| **`timeline_summary.json` & `nsys_summary.json`** | 2 | Before-vs-after stage timeline traces on `spleen_6` (`512x512x135`). |
| **`env.json`, `trt_build.json`, `heldout_cases.json`, `claims_check.json`, `STATE.json`** | 5 | Hardware/toolchain audit, ONNX/TensorRT engine build provenance, held-out dataset split manifest, and automated verifier status. |
