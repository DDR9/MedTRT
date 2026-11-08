# Initial Baseline Audit & Root-Cause Bottleneck Analysis (`docs/BASELINE_AUDIT_REPORT.md`)

> **Audit Date:** 2026-11-05  
> **Target Repository:** `medtrt/` (and legacy prototype `medical-ai/`)  
> **Active Branch:** `feature/medtrt-engineering-validation` (branched from `master` at commit `6ef0c80`)  
> **Hardware Platform:** NVIDIA GeForce RTX 5080 Laptop GPU (Blackwell `sm_120`, 16,302 MiB GDDR7, Driver `591.44`, WDDM)

---

## 1. Executive Summary of Repository Audit

An exhaustive code, artifact, and numerical audit of the existing workspace (`BiomedAi/medtrt` and `BiomedAi/medical-ai`) was performed prior to making engineering modifications. While the repository contains a functional PyTorch + MONAI + TensorRT 10.16 pipeline and real benchmark artifacts in `results/`, **several headline metrics previously reported in `README.md` suffer from methodological confounding, silent numerical drift, or incomplete controls**:

1. **Confounded Network Speedup Claim (`348 ms` $\to$ `21.4 ms`, `16.3×`):**
   - The reported `347.93 ms` baseline came from `pytorch_fp32_strict` (TF32 disabled) at **`overlap=0.5, sw_batch_size=4`** (e.g., 72 patches on `spleen_6`).
   - The reported `21.39 ms` optimized latency came from `trt_fp16_optimized` at **`overlap=0.25, sw_batch_size=8`** (e.g., 19 patches on `spleen_6` — nearly $4\times$ fewer patch evaluations).
   - **Like-for-like network speedup** at identical sliding-window geometry (`overlap=0.5, sw_batch_size=4`) is **`97.91 ms` vs. `347.93 ms` (`3.55×` vs. Strict IEEE FP32)** and **`97.91 ms` vs. `240.11 ms` (`2.45×` vs. Default PyTorch TF32)**. The remaining speedup (`97.91 ms` $\to$ `21.39 ms`, `4.58×`) is algorithmic (reducing sliding-window overlap from `0.50` to `0.25` and increasing batch size from `4` to `8`).

2. **Artificially Inflated Baseline Postprocessing (`2,533 ms` CPU `Invertd`) & Confounded End-to-End Speedup (`3,673 ms` $\to$ `198.3 ms`, `18.5×`):**
   - In commit `6ef0c80`, `src/medtrt/infer.py` explicitly executed `data["pred"] = logits[0].detach().float().cpu()` *before* calling `Invertd`, forcing 3D trilinear spatial resampling of a 2-channel floating-point volume onto the host CPU (`2,365–2,796 ms`).
   - Meanwhile, `trt_fp16_optimized` changed **five independent variables simultaneously**: (i) `.nii.gz` gzip read $\to$ pre-decompressed `.nii` memory-mapped read, (ii) CPU $\to$ GPU preprocessing (`Spacingd`), (iii) PyTorch $\to$ TensorRT FP16, (iv) `overlap=0.5, sw_batch_size=4` $\to$ `overlap=0.25, sw_batch_size=8`, and (v) trilinear probability `Invertd` $\to$ `uint8` nearest-neighbor label `Invertd` (which degraded Mean Dice from `0.965722` to `0.960661`, a `-0.0051` Dice penalty).
   - Furthermore, timing `nibabel.load(..., mmap=True)` in isolation (`2.70 ms`) measured virtual memory mapping rather than physical I/O; the actual page-fault disk read was deferred into `h2d` (`data["image"].to(device)`), which jumped from `4.38 ms` to `49.89 ms`.

3. **Root Cause of Systematic Cross-Backend Volume Bias (`+1.67 ± 0.55 cm³` across 9/9 cases) Uncovered:**
   - Previously, `results/bland_altman.json` reported that `trt_fp16` systematically over-segmented every single validation case (9 out of 9 cases positive, mean bias `+1.6707 ± 0.5515 cm³`) compared to `pytorch_fp32`, and `tests/test_engine_accuracy.py` only tested an **all-zeros input tensor** (`torch.zeros((1, 1, 96, 96, 96))`) to pass `prob_mae < 1e-3`.
   - When tested on non-zero inputs (`torch.rand`), `spleen_unet_fp16.engine`, `pytorch_amp`, and `model.half()` all exhibited a **`0.18–0.24` maximum logit error** and a **systematic signed shift (`-0.159` on background channel 0, `+0.116` on foreground channel 1)**.
   - **First-Principles Root-Cause Identification:** Inspection of the ONNX graph (`engines/spleen_unet_dynamic.onnx`) and `BatchNorm3d` running statistics revealed that PyTorch's ONNX exporter folds `BatchNorm3d` into `Conv3d`, **but does NOT fold `BatchNorm3d` into `ConvTranspose3d`**. In the penultimate decoder block (`model.2.0`), the unfused `ConvTranspose3d` outputs activations with `running_mean = 57.62`, followed by `BatchNorm3d` subtracting a shift of **`81.79`**. In IEEE-754 FP16, values in $[64, 128)$ have an ULP step size of $2^{-4} = 0.0625$, causing severe catastrophic cancellation in FP16 right before the final `1×1×1` output head!
   - **Verified Fix:** Algebraically folding `BatchNorm3d` into `ConvTranspose3d` in FP32 prior to FP16 execution/export reduces FP16 mean absolute logit drift by **`28.5×`** (`0.1377` $\to$ `0.0048`) and eliminates `97%` of the systematic channel bias (`-0.159 / +0.116` $\to$ `-0.0049 / +0.0044`).

4. **Uncommitted Working-Tree State (`medtrt/`):**
   - The working tree of `medtrt/` contained partial refactoring in `src/medtrt/` (`profiles.py`, `ablate.py`, updated `bench.py`, `infer.py`, `stats.py`) that broke `tests/test_engine_accuracy.py` (`AttributeError: 'TRTRunner' object has no attribute 'host_copies_count'`), `app/viewer.py` (`ImportError: cannot import name 'build_split_load_and_preprocess' from 'medtrt.bench'`), and `src/medtrt/report.py` (schema mismatch with updated `stats.py`).

---

## 2. Current Project Architecture

The workspace `BiomedAi/` contains two directories:
- `medical-ai/`: An initial single-volume Streamlit MVP (`app.py`, `inference/segmentation.py`, `models/model_loader.py`, `utils/image_processing.py`) with the Python `.venv` environment (`Python 3.11.15`).
- `medtrt/`: The primary research and benchmarking repository:
  - `src/medtrt/config.py`: Hardware/toolchain audit, SHA-256 code/engine fingerprinting, configuration defaults.
  - `src/medtrt/data.py`: MONAI `spleen_ct_segmentation` v0.6.1 bundle verification (pinned SHA-256 `502c312893994ec0...`), MSD `Task09_Spleen` dataset manager (41 training CT scans), deterministic 9-case validation split, and uncompressed `.nii` memory-mapped loader.
  - `src/medtrt/model.py`: Loads the 3D Residual UNet from `bundles/spleen_ct_segmentation/models/model.pt`.
  - `src/medtrt/export.py`: Exports dynamic-batch ONNX (`opset_version=17`, `B in [1..8]`) and builds serialized TensorRT 10.x engines.
  - `src/medtrt/trt_runner.py`: Zero-copy TensorRT 10.x wrapper binding `torch.Tensor.data_ptr()` via `context.set_tensor_address()` and `context.execute_async_v3(stream_handle)`.
  - `src/medtrt/profiles.py`: Explicit execution profile definitions (`Profile` dataclass) separating engine (`pytorch` vs. `trt`), precision (`fp32`, `tf32`, `amp`, `fp16`), I/O (`gzip` vs. `mmap`), transforms (`cpu` vs. `gpu`), inverse mode (`trilinear` vs. `nearest`), and sliding-window geometry (`overlap`, `sw_batch_size`).
  - `src/medtrt/infer.py`: MONAI preprocessing (`Orientationd(RAS)`, `Spacingd(1.5, 1.5, 2.0, bilinear)`, `ScaleIntensityRanged(-57..164 -> 0..1)`), `SlidingWindowInferer`, and postprocessing (`Activationsd(softmax)`, `Invertd`, `AsDiscreted(argmax)`).
  - `src/medtrt/timing.py`: `CudaStageTimer` and `InstrumentedPredictor` with `torch.cuda.Event(enable_timing=True)` and disjoint NVTX ranges (`load`, `preprocess`, `h2d`, `sw_network`, `sw_aggregate`, `postprocess`).
  - `src/medtrt/bench.py`: Patch-level and full-volume multi-repeat benchmark runner.
  - `src/medtrt/ablate.py`: One-change-at-a-time ablation study ladder.
  - `src/medtrt/evaluate.py`: Per-case Dice and physical volume evaluator.
  - `src/medtrt/stats.py`: Stage timeline generator, `overlap × sw_batch_size` grid sweep, Bland–Altman agreement calculator, and failure case profiler.
  - `src/medtrt/report.py`: Programmatic `README.md` and plot generator.
  - `app/viewer.py`: Tri-planar Streamlit workstation.
  - `tests/`: Pytest suite (`test_engine_accuracy.py`, `test_transforms.py`, `test_volume_math.py`).

---

## 3. Model, Dataset, Hardware, and Toolchain Specifications

### 3.1 Actual Model Architecture
- **Source Bundle:** MONAI Model Zoo `spleen_ct_segmentation` `v0.6.1`
- **Checkpoint Path:** `medtrt/bundles/spleen_ct_segmentation/models/model.pt`
- **Checkpoint SHA-256:** `502c312893994ec071a85c4f2e7d83a43f7789969faa73dae62f1291177f50fe`
- **Architecture:** `monai.networks.nets.UNet`
  - `spatial_dims = 3`
  - `in_channels = 1`, `out_channels = 2` (Channel 0: Background, Channel 1: Spleen)
  - `channels = (16, 32, 64, 128, 256)`
  - `strides = (2, 2, 2, 2)`
  - `num_res_units = 2`
  - `norm = Norm.BATCH` (`BatchNorm3d`, `eps=1e-5`), activation `PReLU`
  - **Sliding-Window Patch ROI:** `(96, 96, 96)`

### 3.2 Dataset & Split Provenance
- **Dataset:** Medical Segmentation Decathlon (MSD) `Task09_Spleen` (`medtrt/data/Task09_Spleen/`)
- **Total Labeled Volumes:** 41 portal-venous phase abdominal CT scans (`imagesTr/` and `labelsTr/`).
- **Held-Out Evaluation Split (`n = 9`):** Alphanumerically sorted final 9 volumes (`@images[-9:]` in `bundles/spleen_ct_segmentation/configs/train.json`):
  - `spleen_56` (`512×512×33`, spacing `0.699×0.699×7.000 mm`)
  - `spleen_59` (`512×512×50`, spacing `0.951×0.951×5.000 mm`)
  - `spleen_6` (`512×512×135`, spacing `0.920×0.920×1.600 mm`)
  - `spleen_60` (`512×512×97`, spacing `0.793×0.793×5.000 mm`)
  - `spleen_61` (`512×512×101`, spacing `0.773×0.773×5.000 mm`)
  - `spleen_62` (`512×512×80`, spacing `0.676×0.676×5.000 mm`)
  - `spleen_63` (`512×512×60`, spacing `0.752×0.752×5.000 mm`)
  - `spleen_8` (`512×512×31`, spacing `0.785×0.785×7.500 mm`)
  - `spleen_9` (`512×512×41`, spacing `0.744×0.744×5.000 mm`)
- **Scientific Caveat on Split:** While these 9 volumes were excluded from gradient updates during bundle training, they served as the validation split used by MONAI's `CheckpointSaver` to select `model.pt`. Because official MSD test set (`imagesTs`) ground-truth labels are not publicly released, these 9 cases represent a validation split (suitable for cross-backend and numerical equivalence evaluation, but optimistic as an estimate of external clinical generalization).

### 3.3 Hardware & Software Environment
| Component | Detected Specification |
|---|---|
| **GPU** | NVIDIA GeForce RTX 5080 Laptop GPU (Blackwell `sm_120`, 16,302 MiB VRAM) |
| **Driver / WDDM** | NVIDIA Driver `591.44` (CUDA Driver API `13.1` / `13010`, WDDM mode) |
| **OS** | Windows 11 (Build `10.0.26300`) |
| **Python** | CPython `3.11.15` (`medical-ai/.venv`) |
| **PyTorch / CUDA / cuDNN** | `torch==2.11.0+cu128` · CUDA Runtime `12.8` · cuDNN `9.19.0` |
| **TensorRT** | `tensorrt==10.16.1.11` (`tensorrt_cu12`) |
| **MONAI / NiBabel / ONNX** | `monai==1.6.1` · `nibabel==5.4.2` · `onnx==1.21.0` · `onnxruntime==1.24.4` |
| **CUDA Runtime Compilation** | `nvrtc64_120_0.dll` (NVRTC `12.8`) + `nvcuda.dll` (CUDA Driver API `13.1`) available in-process |
| **Offline CLI Profilers** | Standalone `nsys.exe`, `ncu.exe`, and `nvcc.exe` are **not installed** on host PATH; GPU kernel profiling and hardware attribute inspection are performed via CUDA Events, PyTorch Kineto/CUDA Profiler (`torch.profiler`), NVRTC (`nvrtc64_120_0.dll`), and CUDA Driver API (`cuFuncGetAttribute`, `cuOccupancyMaxActiveBlocksPerMultiprocessor`). |

---

## 4. Verification of Historical Claims

| Historical Claim | Raw Artifact Present? | Audit Classification | Technical Explanation |
|---|---|---|---|
| PyTorch FP32 network latency = `348 ms` | Yes (`bench_pytorch_fp32_strict.json`: `347.93 ms`) | **Verified (Strict FP32, `overlap=0.5, B=4`)** | Valid for strict IEEE FP32 (`allow_tf32=False`) at `overlap=0.5, sw_batch_size=4`. Default PyTorch (`allow_tf32=True`) is `240.11 ms`. |
| TensorRT FP16 network latency = `21.4 ms` | Yes (`bench_trt_fp16_optimized.json`: `21.39 ms`) | **Verified but Confounded** | Measured at `overlap=0.25, sw_batch_size=8`. At baseline geometry (`overlap=0.5, sw_batch_size=4`), TensorRT FP16 network latency is `97.91 ms`. |
| Network acceleration = `16.3×` | Derived (`347.93 / 21.39 = 16.27×`) | **Confounded Claim** | Combines engine change (`3.55×` vs. Strict FP32) with sliding-window overlap reduction (`0.5` $\to$ `0.25`, `4.58×`). Must be reported separately. |
| Baseline E2E latency = `3,673 ms` | Yes (`bench_pytorch_fp32.json`: `3673.44 ms`) | **Verified (Host-Path Baseline)** | Dominated by CPU `Invertd` (`2,428 ms`) and `.nii.gz` gzip decompression (`671 ms`). |
| Optimized E2E latency = `198.3 ms` (`18.5×`) | Yes (`bench_trt_fp16_optimized.json`: `198.28 ms`) | **Verified but Multi-Factor** | Combines 5 changes; used nearest-neighbor label `Invertd` which reduced Dice by `0.0051`. Trilinear GPU `Invertd` preserves full Dice (`0.9653`) at comparable speed. |
| Mean Dice = `0.961` | Yes (`eval_trt_fp16_optimized.csv`: `0.960661`) | **Suboptimal Postprocessing Artifact** | Caused by nearest-neighbor label `Invertd`. Bundle-faithful trilinear `Invertd` achieves `0.9652–0.9657`. |
| Cross-backend volume bias = `+1.67 ± 0.55 cm³` | Yes (`bland_altman.json`) | **True Bug Identified & Solved** | Caused by unfused `ConvTranspose3d + BatchNorm3d` (`model.2.0.adn.N`, shift `81.79`) in FP16. Folding `BatchNorm3d` into `ConvTranspose3d` before FP16 export eliminates ~97% of this drift. |

---

## 5. Technical Limitations & Proposed Engineering Priorities

1. **Restore Broken Working Tree & Fix `ConvTranspose3d + BatchNorm3d` FP16 Drift (Priority 1):**
   - Fold `BatchNorm3d` into `ConvTranspose3d` before ONNX export and build both **TensorRT FP32 (`spleen_unet_fp32.engine`)** and **TensorRT FP16 (`spleen_unet_fp16.engine`)** engines.
   - Fix `TRTRunner` to use a dedicated non-default CUDA stream (or synchronize cleanly without triggering TensorRT's default-stream warning) and track host copies (`host_copies_count`).
2. **Modular Scientific GPU Benchmarking Suite (`benchmarks/`, Priority 2):**
   - Implement `benchmark_pytorch.py`, `benchmark_tensorrt.py`, `benchmark_preprocessing.py`, `benchmark_postprocessing.py`, `benchmark_end_to_end.py`, `benchmark_memory.py`, `benchmark_scaling.py`, and `benchmark_correctness.py` with full statistical distributions (Mean, Median, SD, IQR, Min, Max, P90, P95, 95% bootstrap CI) and a controlled one-change-at-a-time ablation study.
3. **Custom CUDA C++ Kernels via NVRTC + CUDA Driver API (`cuda/`, Priority 3):**
   - Implement real CUDA C++ kernels (`cuda/medical_kernels.cu`) compiled for `sm_120` via `nvrtc64_120_0.dll` and launched via `nvcuda.dll` (`cuLaunchKernel`), querying hardware occupancy and register allocation via `cuFuncGetAttribute`:
     1. Fused HU Window-Clamp-Normalize (`ScaleIntensityRanged` replacement).
     2. 3D Trilinear Resampling (`Spacingd` / `Invertd` replacement).
     3. 3D Nearest-Neighbor Label Resampling.
     4. Fused 2-Channel Logit Argmax + Discretization (`uint8`) + Atomic Voxel Counting.
     5. 3D Sliding-Window Gaussian-Weighted Patch Accumulation.
   - Benchmark each custom CUDA C++ kernel against CPU and PyTorch CUDA across multiple volume sizes and retain only verified improvements in production.
4. **Comprehensive Medical Image Validation & Failure Analysis (`validation/`, Priority 4):**
   - Expand segmentation metrics to all 10 required metrics: Dice, IoU (Jaccard), Sensitivity, Specificity, Precision, Volumetric Similarity (VS), Absolute Volume Error (AVE), Relative Volume Error (RVE), 95th Percentile Hausdorff Distance (HD95 in mm), and Average Symmetric Surface Distance (ASSD in mm) using full `|det(A_3×3)|` affine Jacobian volume math.
   - Perform separate Bland–Altman analyses (Prediction vs. Ground Truth and Cross-Backend vs. Reference FP32) and detailed anatomical failure analysis on worst-performing cases.
5. **Workstation, Automated Testing, Reports, and Documentation (Priority 5):**
   - Upgrade `app/viewer.py` with side-by-side backend comparison, disagreement/difference mask overlays, and live telemetry.
   - Expand `tests/` to comprehensive unit, integration, cross-backend regression, and failure-handling tests with a formal traceability matrix.
   - Generate all plots and engineering reports programmatically via `analysis/generate_reports.py`.
