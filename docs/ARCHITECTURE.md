# MedTRT System Architecture & Reproducible Pipeline Guide (`docs/ARCHITECTURE.md`)

## 1. End-to-End System Architecture

```mermaid
flowchart TD
    subgraph Ingestion ["1. Volume Ingestion (src/medtrt/data.py)"]
        A1["Compressed .nii.gz (LoadImaged Gzip ~360 ms)"]
        A2["Pre-Decompressed .nii (OS Page-Cache mmap + H2D ~9 ms)"]
    end

    subgraph Pre ["2. 3D Spatial & HU Preprocessing"]
        B1["Host MONAI CPU: Orientationd + Spacingd + ScaleIntensity (~438 ms)"]
        B2["MONAI GPU grid_sample: Allocates 1.03 GB 3D Meshgrid (~79 ms E2E / 256 ms spleen_6)"]
        B3["Custom CUDA C++ Kernel: fused_preprocess_affine_trilinear_hu_f32 (~1.3 ms median / 1.93 ms spleen_6)"]
    end

    subgraph Model ["3. 3D Residual UNet Sliding-Window Inference (96x96x96)"]
        C1["PyTorch Strict FP32 / TF32 / AMP (src/medtrt/model.py)"]
        C2["BN-Folded TensorRT 10.16 FP16 Engine: Zero-Copy data_ptr() + Dedicated Stream (src/medtrt/trt_runner.py)"]
    end

    subgraph Post ["4. 3D Inverse Resampling & Discretization"]
        D1["Host MONAI CPU Invertd (~3,025 ms median)"]
        D2["MONAI GPU Trilinear Invertd: Allocates 2.13 GB 3D Meshgrid (~68 ms median / 157 ms spleen_6)"]
        D3["Custom CUDA C++ Fused Postprocess: fused_softmax_2ch_p1_f32 + fused_invert_trilinear_threshold_u8 (~2.6 ms E2E / 1.00 ms spleen_6)"]
    end

    subgraph Out ["5. Quantitative Validation & Interactive Workstation"]
        E1["Physical Volumetry |det(A_3x3)|, Dice, IoU, HD95, ASSD, Bland-Altman (src/medtrt/evaluate.py, stats.py)"]
        E2["Tri-Planar Streamlit Workstation: Side-by-Side Diff Overlay & Presets (app/viewer.py)"]
    end

    A1 --> B1 --> C1 --> D1 --> E1
    A2 --> B2 --> C2 --> D2 --> E1
    A2 --> B3 --> C2 --> D3 --> E1
    D3 --> E2
```

---

## 2. Repository Directory Structure

```text
medtrt/
├── cuda/
│   └── medical_kernels.cu             # 9 Custom CUDA C++17 3D medical imaging kernels (compiled to sm_120)
├── src/medtrt/
│   ├── config.py                      # Hardware/toolchain audit, SHA-256 code & engine provenance stamps
│   ├── data.py                        # MSD Task09_Spleen split resolver & memory-mapped NIfTI loader
│   ├── model.py                       # 3D Residual UNet loader & analytical FP64 ConvTranspose3d+BN folder
│   ├── export.py                      # ONNX Opset 17 exporter & TensorRT 10.x FP32/FP16 engine builder
│   ├── trt_runner.py                  # Zero-copy TensorRT 10.x runner on dedicated non-default CUDA stream
│   ├── cuda_kernels.py                # NVRTC compiler + Windows CUDA Driver API launcher & occupancy queries
│   ├── profiles.py                    # Unified execution profile definitions (CPU, MONAI GPU, Custom CUDA)
│   ├── timing.py                      # CUDA event stage timers, NVTX range wrappers, bootstrap CI statistics
│   ├── infer.py                       # Unified 3D volume inference pipeline supporting all transform backends
│   ├── bench.py                       # Multi-repeat patch & full-volume GPU benchmark harness + telemetry
│   ├── evaluate.py                    # 10-metric medical validation (Dice, IoU, HD95, ASSD, |det(A)| volume)
│   ├── stats.py                       # Timeline, Pareto grid sweep, Bland-Altman, and failure analysis
│   ├── ablate.py                      # Controlled one-change-at-a-time pipeline ablation
│   ├── probes.py                      # Targeted hardware probes (clock state, ingestion, FP16 numerics, BN fold)
│   ├── report.py                      # Deterministic README.md generator from results/
│   ├── verify_claims.py               # Automated claim & provenance verifier (10 integrity checks)
│   └── pipeline.py                    # Single-command end-to-end orchestrator
├── benchmarks/
│   ├── common.py                      # Shared CUDA event timing & artifact serialization utilities
│   ├── benchmark_pytorch.py           # PyTorch FP32 Strict, TF32, AMP, and Pure FP16 patch benchmarks
│   ├── benchmark_tensorrt.py          # TensorRT FP32, FP16 Unfolded, FP16 Folded, & Buffer Reuse benchmarks
│   ├── benchmark_preprocessing.py     # Gzip vs. Mmap vs. Pinned H2D & Custom CUDA preprocessing benchmarks
│   ├── benchmark_postprocessing.py    # CPU vs. MONAI GPU vs. Custom CUDA fused postprocessing benchmarks
│   ├── benchmark_end_to_end.py        # 7-rung controlled end-to-end pipeline ablation study
│   ├── benchmark_memory.py            # Static & dynamic VRAM allocation analysis
│   ├── benchmark_scaling.py           # Volume size scaling (4.2M to 50.3M voxels) across custom CUDA kernels
│   ├── benchmark_correctness.py       # Numerical equivalence & edge-case verification
│   └── benchmark_cohort_41.py         # Full 41-patient-volume GPU execution & telemetry verification
├── analysis/
│   └── generate_reports.py            # Master report & 12-plot generator from measured results/
├── app/
│   ├── viewer.py                      # Streamlit multi-page dashboard entry point
│   └── workstation.py                 # Synchronized tri-planar workstation with real-time FP32-vs-FP16 diff heatmaps
├── tests/                             # 65-test automated verification suite auditing latency & memory telemetry
├── configs/default.yaml               # Declarative pipeline & TensorRT builder configuration
├── reports/                           # Detailed Phase 3-7 technical reports
└── docs/                              # Engineering audit, technical paper, resume evidence, and 12 generated figures
```
