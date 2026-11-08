# Key Engineering Findings, Architecture Summary & Technical FAQ (`docs/KEY_FINDINGS_AND_FAQ.md`)

> Every metric below is backed by reproducible benchmark artifacts (`results/bench_*.json`, `results/cohort_41_volumes.json`, `results/benchmark_*.json`, `results/eval_*.csv`, `results/bland_altman.json`) measured on an **NVIDIA GeForce RTX 5080 Laptop GPU** (`sm_120`, 16 GB GDDR7, CUDA 12.8, TensorRT 10.16.1.11) and audited by our 65-test verification suite (`pytest tests/` & `python -m medtrt.verify_claims`).

---

## 1. Executive Summary of Verified Engineering Highlights

- **Accelerated end-to-end 3D CT spleen segmentation by `42.9×` (`2,952 ms` to `68.9 ms`) across 41 patient volumes** by migrating spatial transformations to the GPU and implementing page-cached memory mapping with TensorRT 10.16 FP16 execution (`results/bench_pytorch_fp32.json`, `results/bench_trt_fp16_optimized.json`, `results/cohort_41_volumes.json`).
- **Authored 9 custom CUDA C++17 kernels that compute $3\times 4$ affine resampling coordinates directly in registers, reducing peak pipeline VRAM from `2,301 MB` to `455 MB` (`5.1×` reduction) and accelerating 3D preprocessing `33.8×` (`1.3 ms`)** (`cuda/medical_kernels.cu`, `results/bench_trt_fp16_gpu.json`, `results/bench_trt_fp16_cuda.json`).
- **Eliminated a systematic `+1.68 cm³` organ volumetric dilation across `9/9` validation scans caused by FP16 decoder cancellation** by deriving analytical `float64` `BatchNorm3d` weight folding, achieving **`+0.001 cm³` volume bias, `0.9652` Mean Dice, and `2.18 mm` HD95** (`src/medtrt/model.py`, `results/bland_altman.json`, `results/eval_trt_fp16_cuda.csv`).
- **Developed an interactive tri-planar radiological workstation in Streamlit with real-time FP32-vs-FP16 difference heatmaps, backed by a 65-test verification suite** auditing latency and memory metrics against raw hardware telemetry (`app/workstation.py`, `app/viewer.py`, `tests/`, `src/medtrt/verify_claims.py`).

---

## 2. Systems & GPU Performance Engineering Breakdown

1. **End-to-End 3D Medical Imaging Pipeline Acceleration (`42.9×` / `44.3×` Speedup):**
   Architected a zero-copy GPU 3D CT segmentation pipeline in PyTorch, TensorRT 10.16, and CUDA C++17 on an RTX 5080 (`sm_120`), reducing full-volume median end-to-end latency from **`3050 ms`** (strict FP32) / **`2952 ms`** (default TF32) to **`96.3 ms`** at identical sliding-window geometry (**`30.7×` speedup**) and **`68.9 ms`** (`14.5 volumes/s`, **`42.9×` speedup**) at Pareto-optimal window settings, with all 41 patient CT scans in `Task09_Spleen` verified on the GPU (`results/cohort_41_volumes.json`).
2. **Custom CUDA C++17 3D Affine Resampling & Fused Postprocessing Kernels (`33.8×` Pipeline Preprocess Speedup, `5.1×` Full-Pipeline VRAM Reduction):**
   Authored 9 custom CUDA C++17 kernels (`cuda/medical_kernels.cu`, `50–100%` SM occupancy, `0 B` local register spills) that compute $3\times 4$ voxel-to-voxel affine coordinates in registers rather than materializing MONAI's 1–2 GB 3D coordinate meshgrids, reducing full-pipeline peak VRAM from **`2,301 MB` to `455 MB` (`5.1×` reduction)**, accelerating full-pipeline 3D CT preprocessing by **`33.8×` (`43.9 ms -> 1.3 ms`)** (and isolated `spleen_6` preprocessing by **`21.4×`**, `40.9 ms -> 1.92 ms`, **`7.1×` lower VRAM**), and accelerating 3D trilinear inverse postprocessing by **`124.9×`** (`118.8 ms -> 0.95 ms`, **`50.8×` lower VRAM**) with **`100.00000%` exact voxel agreement**.
3. **TensorRT 10.x FP16 Numerical Stabilization via FP64 `ConvTranspose3d + BatchNorm3d` Folding (`35.6×` Lower Logit Error):**
   Diagnosed an asymmetric ONNX constant-folding omission where unfused decoder `ConvTranspose3d + BatchNorm3d` layers (`shift = -81.79` at FP16 $\text{ULP} = 0.0625$) caused catastrophic FP16 cancellation; implemented analytical FP64 weight folding prior to export, reducing TensorRT FP16 mean absolute logit drift by **`35.6×`** (`0.1377 -> 0.00387`) and cross-backend organ volume bias from **`+1.68 cm³` (`9/9` scans)** to **`+0.001 cm³`**.
4. **NVTX Stage Instrumentation, Non-Default Stream Execution & Amdahl Bottleneck Analysis:**
   Instrumented 6 pipeline stages with `torch.cuda.Event` and NVTX ranges (`src/medtrt/timing.py`), eliminated TensorRT 10.x `enqueueV3` default-stream host synchronization via event-coupled non-default CUDA streams (`src/medtrt/trt_runner.py`), and quantified a `1.13×` laptop GPU clock-state penalty after host idle gaps.

---

## 3. Biomedical Imaging & Quantitative Validation Breakdown

1. **Multi-Metric 3D CT Organ Segmentation & Physical Volumetry Validation ($n = 9$ Held-Out Scans):**
   Validated 3D Residual UNet spleen segmentation across 13 execution profiles using 10 quantitative metrics (Dice, IoU, Sensitivity, Specificity, Precision, Volumetric Similarity, Absolute/Relative Volume Error, `HD95`, and `ASSD` in physical `mm` via $|\det(A_{3\times 3})|$), achieving **`Mean Dice = 0.9652 ± 0.0117` (`Median = 0.9708`)**, **`Mean HD95 = 2.18 ± 0.99 mm`**, and **`Mean ASSD = 0.39 ± 0.12 mm`**.
2. **Bland–Altman Cross-Backend Equivalence & Elimination of Systematic Volumetric Bias:**
   Conducted paired Bland–Altman agreement studies separating model-vs-ground-truth error (`MAE = 5.17 cm³`, `Volumetric Similarity = 0.9905`) from cross-backend FP16-vs-FP32 drift, demonstrating that BN-folded TensorRT FP16 + Custom CUDA C++ kernels achieve **`+0.0010 ± 0.0194 cm³`** volume bias (`95% LoA [-0.037, +0.039] cm³`, `p = 0.885`) and **`Mean |ΔDice| = 0.000014`** (`Max = 0.000029`) against strict IEEE FP32.
3. **Anatomical Failure Mode Decomposition & Probability-Space Trilinear Inversion:**
   Quantified spatial error concentration at superior/inferior organ poles (`22–24%` of error voxels in the outermost `10%` of slices) and hilar vessel bifurcations (`spleen_62`), and proved that continuous-probability trilinear inversion eliminates the **`-0.0046` Mean Dice degradation** of discrete nearest-neighbor label inversion under anisotropic slice thickness (`5.0–8.0 mm`).
4. **65-Test Medical Software Verification Suite & Interactive Tri-Planar Radiological Workstation:**
   Built a 65-test automated verification suite (`pytest tests/`) with requirements traceability auditing latency and memory metrics against raw hardware telemetry (`verify_claims.py`), affine volumetry, custom CUDA kernel edge cases, cross-backend equivalence, and defensive failure handling, alongside a synchronized tri-planar Streamlit workstation (`app/viewer.py` & `app/workstation.py`) featuring real-time FP32-vs-FP16 difference heatmaps and voxel-disagreement overlays.

---

## 4. Frequently Asked Technical Questions (FAQ)

### Q1: Why did your pure network speedup (`17.0×` vs. Strict FP32 / `6.4×` vs. TF32 on `B=4` patches) initially translate to only `1.07×` end-to-end on the host pipeline?
- **Answer:** By Amdahl's Law ($S \le 1/s$), `sw_network` (`285 ms`) was only `7.3%` of the `3,925 ms` host pipeline, whereas CPU trilinear `Invertd` (`3,025 ms`), CPU `Spacingd` (`438 ms`), and `.nii.gz` gzip decompression (`360 ms`) accounted for `92.7%` ($1/s = 1.08\times$). Even after moving transforms to MONAI's GPU `grid_sample` (`trt_fp16_gpu`, `176 ms` E2E), MONAI allocated `1.03–2.13 GB` 3D coordinate meshgrids in FP64/FP32 global memory. Writing custom CUDA C++ kernels that compute $3\times 4$ affine coordinates in registers reduced preprocessing and postprocessing to `~1.0–1.4 ms` each, unlocking the full **`53.5×`** (`trt_fp16_cuda`, `73.4 ms`) and **`105.7×`** (`trt_fp16_optimized`, `37.1 ms`) end-to-end speedup.

### Q2: How did you prove that the original `+1.68 cm³` volume shift in TensorRT FP16 was an export artifact rather than an unavoidable limitation of FP16 segmentation?
- **Answer:** We inspected the exported ONNX graph and every intermediate layer's activation distribution. PyTorch's ONNX exporter folds `Conv3d+BatchNorm3d` in the encoder but leaves `ConvTranspose3d+BatchNorm3d` unfused in the decoder. In `model.2.0.adn.N`, `running_mean = 57.62` and the affine shift is `-81.79`. Because IEEE-754 FP16 has a 10-bit mantissa, its quantization step in $[64, 128)$ is $\text{ULP} = 0.0625$. Quantizing the pre-BN activation and shift to FP16 before subtracting caused catastrophic cancellation that shifted background logit Ch0 by `-0.159` and dilated the spleen mask on all 9/9 scans. Folding `BatchNorm3d` into `ConvTranspose3d` in FP64 prior to export eliminated the large intermediate offset, reducing logit error by `35.6×` and volume bias to `+0.0010 cm³` (`p = 0.885`).
