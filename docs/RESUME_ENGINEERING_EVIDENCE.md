# Phase 14 — Quantified Engineering Evidence & Interview Defense Guide (`docs/RESUME_ENGINEERING_EVIDENCE.md`)

> Every metric below is backed by reproducible benchmark artifacts (`results/bench_*.json`, `results/benchmark_*.json`, `results/eval_*.csv`, `results/bland_altman.json`) measured on an **NVIDIA GeForce RTX 5080 Laptop GPU** (`sm_120`, 16 GB GDDR7, CUDA 12.8, TensorRT 10.16.0.72).

---

## 1. Four Quantified Resume Bullets — NVIDIA (GPU Computing, CUDA C++, TensorRT, Performance Engineering)

1. **End-to-End 3D Medical Imaging Pipeline Acceleration (`54.9×` / `51.3×` Speedup):**
   Architected a zero-copy GPU 3D CT segmentation pipeline in PyTorch, TensorRT 10.16, and CUDA C++17 on an RTX 5080 (`sm_120`), reducing full-volume median end-to-end latency from **`3122 ms`** (strict FP32) / **`3341 ms`** (default TF32) to **`95.1 ms`** at identical sliding-window geometry (**`35.1×` speedup**) and **`60.8 ms`** (`16.4 volumes/s`, **`54.9×` speedup**) at Pareto-optimal window settings.
2. **Custom CUDA C++17 3D Affine Resampling & Fused Postprocessing Kernels (`21×`–`125×` Kernel Speedup, `50.8×` VRAM Reduction):**
   Authored 9 custom CUDA C++17 kernels (`cuda/medical_kernels.cu`, `50–100%` SM occupancy, `0 B` local register spills) that compute $3\times 4$ voxel-to-voxel affine coordinates in registers rather than materializing MONAI's 1–2 GB 3D coordinate meshgrids, accelerating 3D CT preprocessing by **`21.4×`** (`40.9 ms -> 1.92 ms`, **`7.1×` lower VRAM**) and 3D trilinear inverse postprocessing by **`124.9×`** (`118.8 ms -> 0.95 ms`, **`50.8×` lower VRAM**) with **`100.00000%` exact voxel agreement**.
3. **TensorRT 10.x FP16 Numerical Stabilization via FP64 `ConvTranspose3d + BatchNorm3d` Folding (`35.6×` Lower Logit Error):**
   Diagnosed an asymmetric ONNX constant-folding omission where unfused decoder `ConvTranspose3d + BatchNorm3d` layers (`shift = -81.79` at FP16 $\text{ULP} = 0.0625$) caused catastrophic FP16 cancellation; implemented analytical FP64 weight folding prior to export, reducing TensorRT FP16 mean absolute logit drift by **`35.6×`** (`0.1377 -> 0.00387`) and cross-backend organ volume bias by **`620×`** (`+1.675 cm³ -> +0.001 cm³`).
4. **NVTX Stage Instrumentation, Non-Default Stream Execution & Amdahl Bottleneck Analysis:**
   Instrumented 6 pipeline stages with `torch.cuda.Event` and NVTX ranges (`src/medtrt/timing.py`), eliminated TensorRT 10.x `enqueueV3` default-stream host synchronization via event-coupled non-default CUDA streams (`src/medtrt/trt_runner.py`), and quantified a `2.34×` laptop GPU clock-state penalty after host idle gaps.

---

## 2. Four Quantified Resume Bullets — Medtronic (Biomedical Imaging, Quantitative Validation, Verification)

1. **Multi-Metric 3D CT Organ Segmentation & Physical Volumetry Validation ($n = 9$ Held-Out Scans):**
   Validated 3D Residual UNet spleen segmentation across 11 execution profiles using 10 quantitative metrics (Dice, IoU, Sensitivity, Specificity, Precision, Volumetric Similarity, Absolute/Relative Volume Error, `HD95`, and `ASSD` in physical `mm` via $|\det(A_{3\times 3})|$), achieving **`Mean Dice = 0.9652 ± 0.0117` (`Median = 0.9708`)**, **`Mean HD95 = 2.18 ± 0.99 mm`**, and **`Mean ASSD = 0.39 ± 0.12 mm`**.
2. **Bland–Altman Cross-Backend Equivalence & Elimination of Systematic Volumetric Bias:**
   Conducted paired Bland–Altman agreement studies separating model-vs-ground-truth error (`MAE = 5.17 cm³`, `Volumetric Similarity = 0.9905`) from cross-backend FP16-vs-FP32 drift, demonstrating that BN-folded TensorRT FP16 + Custom CUDA C++ kernels achieve **`+0.0010 ± 0.0194 cm³`** volume bias (`95% LoA [-0.037, +0.039] cm³`, `p = 0.885`) and **`Mean |ΔDice| = 0.000014`** (`Max = 0.000029`) against strict IEEE FP32.
3. **Anatomical Failure Mode Decomposition & Probability-Space Trilinear Inversion:**
   Quantified spatial error concentration at superior/inferior organ poles (`22–24%` of error voxels in the outermost `10%` of slices) and hilar vessel bifurcations (`spleen_62`), and proved that continuous-probability trilinear inversion eliminates the **`-0.0046` Mean Dice degradation** of discrete nearest-neighbor label inversion under anisotropic slice thickness (`5.0–8.0 mm`).
4. **Medical Software Verification Suite & Interactive Tri-Planar Radiological Workstation:**
   Built a 40-test automated verification suite (`pytest`) with requirements traceability covering affine volumetry, custom CUDA kernel edge cases, cross-backend equivalence, defensive failure handling, and zero-GPU claim verification (`verify_claims.py`), alongside a synchronized tri-planar Streamlit workstation (`app/viewer.py`) featuring side-by-side backend comparison and voxel-difference overlays.

---

## 3. Technical Interview Defense Q&A

### Q1: Why did your pure network speedup (`17.0×` vs. Strict FP32 / `6.4×` vs. TF32 on `B=4` patches) initially translate to only `1.07×` end-to-end on the host pipeline?
- **Answer:** By Amdahl's Law ($S \le 1/s$), `sw_network` (`285 ms`) was only `7.3%` of the `3,925 ms` host pipeline, whereas CPU trilinear `Invertd` (`3,025 ms`), CPU `Spacingd` (`438 ms`), and `.nii.gz` gzip decompression (`360 ms`) accounted for `92.7%` ($1/s = 1.08\times$). Even after moving transforms to MONAI's GPU `grid_sample` (`trt_fp16_gpu`, `176 ms` E2E), MONAI allocated `1.03–2.13 GB` 3D coordinate meshgrids in FP64/FP32 global memory. Writing custom CUDA C++ kernels that compute $3\times 4$ affine coordinates in registers reduced preprocessing and postprocessing to `~1.0–1.4 ms` each, unlocking the full **`53.5×`** (`trt_fp16_cuda`, `73.4 ms`) and **`105.7×`** (`trt_fp16_optimized`, `37.1 ms`) end-to-end speedup.

### Q2: How did you prove that the original `+1.68 cm³` volume shift in TensorRT FP16 was an export artifact rather than an unavoidable limitation of FP16 segmentation?
- **Answer:** We inspected the exported ONNX graph and every intermediate layer's activation distribution. PyTorch's ONNX exporter folds `Conv3d+BatchNorm3d` in the encoder but leaves `ConvTranspose3d+BatchNorm3d` unfused in the decoder. In `model.2.0.adn.N`, `running_mean = 57.62` and the affine shift is `-81.79`. Because IEEE-754 FP16 has a 10-bit mantissa, its quantization step in $[64, 128)$ is $\text{ULP} = 0.0625$. Quantizing the pre-BN activation and shift to FP16 before subtracting caused catastrophic cancellation that shifted background logit Ch0 by `-0.159` and dilated the spleen mask on all 9/9 scans. Folding `BatchNorm3d` into `ConvTranspose3d` in FP64 prior to export eliminated the large intermediate offset, reducing logit error by `35.6×` and volume bias to `+0.0010 cm³` (`p = 0.885`).
