# MedTRT: Eliminating End-to-End I/O, 3D Affine Resampling, and FP16 Quantization Bottlenecks in Volumetric CT Organ Segmentation

**Author:** Diego Díaz Rivera  
**Affiliation:** Department of Bioengineering, Rice University, Houston, TX, USA  
**Domain:** Biomedical AI Systems & GPU Performance Engineering (`NVIDIA GeForce RTX 5080 Blackwell sm_120`)  
**Repository:** https://github.com/DDR9/MedTRT  
**Date:** November 2026 (`code_fingerprint = e9d4f060aaf1b130`)  
**Compiled PDF Report:** [`../MedTRT_Scientific_Report.pdf`](../MedTRT_Scientific_Report.pdf) | [`TECHNICAL_PAPER.pdf`](TECHNICAL_PAPER.pdf) | [`TECHNICAL_PAPER.tex`](TECHNICAL_PAPER.tex)

---

## Abstract

**Background:** Three-dimensional convolutional neural networks such as the 3D Residual U-Net enable automated volumetric organ segmentation in abdominal computed tomography (CT). Processing full diagnostic volumes ($512 \times 512 \times 31\text{--}168$ voxels, $8.13 \times 10^6$ to $4.40 \times 10^7$ voxels per scan) with anisotropic physical spacing ($1.5\text{--}8.0\text{ mm}$ axial slice thickness) requires a sequential pipeline spanning file decompression, canonical coordinate reorientation, trilinear spatial resampling, overlapping sliding-window patch inference, and inverse spatial resampling to native patient geometry.

**Problem:** Profiling the pretrained MONAI `spleen_ct_segmentation` v0.6.1 bundle across **53 abdominal 3D CT volumes** (`41` Medical Segmentation Decathlon `Task09_Spleen` scans plus `12` unseen external multi-center MICCAI `FLARE22` scans) on an NVIDIA GeForce RTX 5080 Laptop GPU (Blackwell `sm_120`, `16 GB` GDDR7) exposes three bottlenecks that defeat standalone network compilation:
1. **Amdahl's Law Non-Network Bottleneck (`91.38%` of Host Runtime):** In the baseline host pipeline (`pytorch_fp32`, `2,952.0 ms` median end-to-end latency), the 3D U-Net forward pass (`sw_network = 254.6 ms`) consumes only `8.62%` of total runtime, while `91.38%` is spent in CPU inverse resampling (`1,990.4 ms`), `.nii.gz` decompression (`424.7 ms`), and CPU preprocessing (`286.4 ms`), bounding theoretical network-only speedup via Amdahl's Law to $S_{\max} = 1.094\times$.
2. **MONAI GPU `float64` Coordinate Meshgrid VRAM Explosion (`2,301 MB`):** Moving spatial transforms onto the GPU via MONAI's `SpatialResample` backend (`trt_fp16_gpu`, `215.5 ms`) promotes 3D homogeneous coordinates to `torch.float64` on `cuda:0` and materializes dense `(1, D, H, W, 3)` coordinate meshgrids, inflating peak allocated VRAM to **`2,301 MB`** (`5.06x` higher than our register-affine pipeline).
3. **Asymmetric ONNX Constant Folding & Decoder FP16 Cancellation (`+1.68 cm³` / `+1.54 cm³` Dilation):** Compiling the standard ONNX graph into TensorRT 10.16 FP16 (`trt_fp16_unfolded`) induces a systematic volumetric dilation of **`+1.6754 ± 0.5531 cm³`** across **`9/9`** internal MSD scans ($p < 0.0001$) and **`+1.5366 ± 0.7680 cm³`** across **`12/12`** external `FLARE22` scans (`+1.5961 ± 0.6721 cm³` across all `21/21` pooled patients) because PyTorch's ONNX exporter leaves decoder `ConvTranspose3d+BatchNorm3d` blocks unfused, exposing a `-81.79` shift in `model.2.0.adn.N` where binary16 $\text{ULP}_{[64, 128)} = 0.0625$.

**Methods:** Without retraining or altering model weights semantically, we introduce: (i) analytical IEEE-754 `float64` weight folding (`fold_conv_transpose_batchnorm`) prior to ONNX export; (ii) **nine custom CUDA C++17 kernels** (`cuda/medical_kernels.cu`) that compute $3 \times 4$ voxel-to-voxel affine coordinates in scalar registers, fuse Hounsfield Unit (HU) windowing with trilinear preprocessing, and exploit binary Softmax convexity ($p_0 + p_1 = 1$) for single-channel inverse trilinear resampling, thresholding, and warp-ballot (`__ballot_sync`) foreground counting; and (iii) zero-copy TensorRT 10.16 execution paired with OS page-cached `.nii` memory mapping (`mmap`), an interactive 3D tri-planar verification workstation, 66 automated `pytest` tests, and 14 deterministic offline claim checks.

**Results:** At matched sliding windows (`overlap=0.5, B=4`), our controlled pipeline (`trt_fp16_cuda`) lowers median latency from `2,952.0 ms` to **`96.3 ms`** (**`30.67x`** cohort median speedup; **`33.14x`** paired patient-level mean speedup, 95% CI `[27.61x, 38.67x]`) and cuts peak VRAM from `2,301 MB` to **`455 MB`** (**`5.06x`** reduction). Our fast-window operating point (`trt_fp16_optimized`, `overlap=0.25, B=8`) reaches **`68.9 ms`** median latency (**`42.86x`** cohort speedup; **`46.64x`** paired median speedup; `87.44 ms` median across all 41 MSD volumes). Analytical FP64 folding reduces mean absolute logit drift by **`35.6x`** (`0.1377 -> 0.00387`) and satisfies paired Two One-Sided Tests (TOST) equivalence against strict IEEE FP32 within $\Delta_V = \pm 0.10\text{ cm}^3$ across internal MSD (`+0.0010 ± 0.0194 cm³`, $p_{\text{TOST}} < 10^{-6}$), external `FLARE22` (`-0.0110 ± 0.0315 cm³`, $p_{\text{TOST}} < 10^{-6}$), and pooled $N = 21$ patients (`-0.0059 ± 0.0271 cm³`, 90% CI `[-0.0161, +0.0043] cm³`, $p_{\text{TOST}} < 10^{-6}$), maintaining Mean Dice `0.9652`, $\max \lvert\Delta\text{Dice}\rvert = 0.000029$, Mean $\text{HD}_{95} = 2.18\text{ mm}$, and Mean $\text{ASSD} = 0.39\text{ mm}$.

**Conclusion:** Real-time, numerically faithful 3D medical segmentation requires co-designing storage ingestion, register-level GPU spatial kernels, and pre-export algebraic conditioning alongside neural network compilation.

*Research Disclosure: MedTRT is an educational and scientific GPU systems benchmark on public retrospective datasets and is not a cleared clinical medical device.*

---

## 1. Introduction & Central Research Questions

**Central Research Question:** *Can 3D volumetric medical image segmentation achieve order-of-magnitude end-to-end GPU acceleration while preserving formal statistical numerical equivalence with a strict IEEE-754 FP32 reference implementation?*

- **Research Problem A (Systems Performance & Amdahl's Law):** Why does standalone neural network compilation fail to accelerate end-to-end 3D CT segmentation, and how can storage ingestion, 3D affine preprocessing, and inverse probability resampling be co-designed in custom CUDA C++17 to eliminate both CPU bottlenecks and framework GPU memory bloat?
- **Research Problem B (FP16 Numerical Stability):** Why does standard ONNX-to-TensorRT FP16 compilation induce a systematic `+1.60 cm³` volumetric organ dilation across `21/21` CT scans, and can an exact pre-export algebraic transformation restore strict IEEE-754 FP32 equivalence without sacrificing Tensor Core throughput?

---

## 2. Background: The Five Stages of Volumetric 3D CT Segmentation

1. **CT Attenuation Physics & NIfTI Ingestion (`load` + `h2d` = `ingest`):** Reconstructed linear X-ray attenuation $\mu(x,y,z)$ ($\text{cm}^{-1}$) is calibrated to Hounsfield Units ($\text{HU} = 1000 \times (\mu - \mu_{\text{water}}) / (\mu_{\text{water}} - \mu_{\text{air}})$) and stored in NIfTI-1 format (`.nii.gz` or `.nii`) with native voxel-to-world affine matrix $A_{\text{orig}} \in \mathbb{R}^{4 \times 4}$.
2. **Affine Coordinate Geometry, RAS Orientation & Trilinear Resampling (`preprocess`):** Reorients scanner axes to Right-Anterior-Superior (`RAS`), resamples anisotropic voxels to $1.5 \times 1.5 \times 2.0\text{ mm}$ via $\mathbf{x}_s = A_{\text{orig}}^{-1} A_{\text{res}} \mathbf{x}_o = M_{\text{pre}} \mathbf{x}_o$, and normalizes $[-57, 164]\text{ HU} \to [0, 1]$.
3. **3D Residual U-Net Sliding-Window Inference (`sw_network` + `sw_aggregate`):** Evaluates overlapping $96 \times 96 \times 96$ patches (`18` to `125` windows per scan) through the pretrained MONAI 3D Residual U-Net (`spleen_ct_segmentation` v0.6.1) with separable 3D Gaussian importance weighting ($\sigma = 0.125$).
4. **Inverse Spatial Resampling & Discretization (`postprocess`):** Applies Softmax, resamples continuous probabilities back to native patient geometry via $M_{\text{inv}} = M_{\text{pre}}^{-1}$, and thresholds into a binary `uint8` organ mask.
5. **Physical Volumetry & Surface Distance Evaluation:** Computes $V = N_{\text{fg}} \cdot \lvert\det(A_{3\times 3})\rvert / 1000\text{ cm}^3$ alongside `Dice`, `IoU`, `Sensitivity`, `Precision`, `Specificity`, `Volumetric Similarity`, `HD95`, and `ASSD`.

![End-to-End Pipeline Architecture](figures/pipeline_architecture.png)

---

## 3. Baseline Bottlenecks & Custom CUDA C++17 Implementation

### 3.1 Amdahl's Law Formulation
Let $p = T_{\text{net}} / T_{\text{E2E}}$ denote the network runtime fraction (dimensionless). Accelerating the network by $S_{\text{net}}$ yields:

$$S(S_{\text{net}}) = \frac{1}{(1 - p) + \frac{p}{S_{\text{net}}}}, \qquad S_{\max} = \lim_{S_{\text{net}} \to \infty} S(S_{\text{net}}) = \frac{1}{1 - p}$$

In `pytorch_fp32` (`2,952.0 ms` E2E, `254.6 ms` network), $p = 0.0862$ and $S_{\max} = 1.0944\times$.

### 3.2 Nine Custom CUDA C++17 Kernels (`cuda/medical_kernels.cu`)
All 9 `__global__` kernels compile at runtime via NVRTC 12.8 (`--gpu-architecture=compute_120 --use_fast_math -default-device`):

| # | Custom CUDA C++17 Kernel (`cuda/medical_kernels.cu`) | SASS Target | Regs / Thread | Shared / Spill | Active Warps / SM (`256t`) | Primary Pipeline Role & Verified Performance |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 1 | `hu_window_normalize_f32` | `sm_120` | `23` | `0 B / 0 B` | `48 / 48` (**100.0%**) | Scalar grid-stride HU clamp & linear scaling (`0.611 ms`, **463.7 GB/s**) |
| 2 | `hu_window_normalize_vec4_f32` | `sm_120` | `48` | `0 B / 0 B` | `40 / 48` (**83.3%**) | 128-bit vectorized `float4` coalesced HU scaling (`0.607 ms`, **466.7 GB/s**) |
| 3 | `affine_trilinear_resample_3d_f32` | `sm_120` | `72` | `0 B / 0 B` | `24 / 48` (**50.0%**) | Multi-channel register-affine 3D trilinear resampling (**6.4–13.3x faster** than PyTorch) |
| 4 | **`fused_preprocess_affine_trilinear_hu_f32`** | `sm_120` | `37` | `0 B / 0 B` | `48 / 48` (**100.0%**) | Fused `RAS` + $1.5\times 1.5\times 2.0\text{ mm}$ trilinear + HU windowing (**1.27 ms median, 33.8x faster**) |
| 5 | `affine_nearest_resample_3d_u8` | `sm_120` | `16` | `0 B / 0 B` | `48 / 48` (**100.0%**) | Register-affine 3D nearest-neighbor `uint8` resampling (**16.7–48.1x faster** than PyTorch) |
| 6 | **`fused_softmax_2ch_p1_f32`** | `sm_120` | `32` | `0 B / 0 B` | `48 / 48` (**100.0%**) | Numerically stable 2-ch Softmax (`__expf`) to 1-ch foreground probability $p_1$ |
| 7 | **`fused_invert_trilinear_threshold_u8`** | `sm_120` | `38` | `0 B / 0 B` | `48 / 48` (**100.0%**) | Fused 1-ch inverse trilinear + $\ge 0.5$ `uint8` + `__ballot_sync` count (**2.62 ms E2E, 50.8x less VRAM**) |
| 8 | `count_nonzero_u8` | `sm_120` | `36` | `128 B / 0 B` | `48 / 48` (**100.0%**) | Two-stage warp-shuffle (`__shfl_down_sync`) + shared-memory reduction (**8.4x faster**) |
| 9 | `sliding_window_gaussian_accumulate_f32` | `sm_120` | `56` | `0 B / 0 B` | `32 / 48` (**66.7%**) | Fused 3D Gaussian importance weighting & sliding-window patch accumulation |

---

## 4. Numerical Fidelity & Analytical FP64 `ConvTranspose3d + BatchNorm3d` Folding

| Decoder `ConvTranspose3d + BatchNorm3d` Block | $C_{\text{out}}$ | Unfused Conv Bias $b_c$ | BN Running Mean $\mu_c$ | Scale $s_c = \frac{\gamma_c}{\sqrt{\sigma_c^2+\epsilon}}$ | Unfused BN Shift $\beta_c - \mu_c s_c$ | Folded Bias $\widetilde{b}_c = (b_c - \mu_c)s_c + \beta_c$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `model.1.submodule.1.submodule.1.submodule.2.0` (Stage 1) | `64` | `[-0.196, +0.448]` | `[-0.196, +0.448]` | `max 1.3193` | `[-0.5858, +0.3007]` | `max 0.2621` |
| `model.1.submodule.1.submodule.2.0` (Stage 2) | `32` | `[-0.453, +0.395]` | `[-0.453, +0.395]` | `max 1.6689` | `[-0.5089, +0.6604]` | `max 0.2885` |
| `model.1.submodule.2.0` (Stage 3) | `16` | `[-0.569, +0.431]` | `[-0.569, +0.431]` | `max 1.6712` | `[-0.6951, +0.9425]` | `max 0.2956` |
| **`model.2.0` (Channel 0: Background Head)** | `1` (Ch0) | **`+57.295700`** | **`+57.618198`** | **`+1.437999`** | **`-81.791538` (`-81.79`)** | **`+0.599613`** |
| **`model.2.0` (Channel 1: Spleen Foreground Head)** | `1` (Ch1) | **`-18.385847`** | **`-18.462671`** | **`+0.921735`** | **`+16.595694` (`+16.60`)** | **`-0.351193`** |

Promoting $W \in \mathbb{R}^{C_{\text{in}} \times C_{\text{out}} \times k_D \times k_H \times k_W}$ and $\mathbf{b}, \boldsymbol{\gamma}, \boldsymbol{\beta}, \boldsymbol{\mu}, \boldsymbol{\sigma}^2$ to `torch.float64` before ONNX export (`fold_conv_transpose_batchnorm`):

$$s_c = \frac{\gamma_c}{\sqrt{\sigma_c^2 + \epsilon}}, \qquad \widetilde{W}_{i, c, d, h, w} = W_{i, c, d, h, w} \cdot s_c, \qquad \widetilde{b}_c = (b_c - \mu_c) s_c + \beta_c$$

collapses the `-81.79` shift in `model.2.0` into `+0.599613`, reducing FP16 mean absolute logit drift by **`35.6x`** (`0.1377 -> 0.00387`).

---

## 5. Eight Secondary Problems Resolved & Verification Infrastructure

- **Problem A (Deferred `mmap` Page-Fault Accounting & Three-Regime I/O Decomposition):** Reports combined `ingest = load + h2d` (`29.2 ms` warm median) alongside standalone CPU `.nii.gz` decompression (`264.95 ms`) and cold unbuffered NVMe `.nii` reads (`23.83 ms`).
- **Problem B (Staircase Aliasing in Nearest-Neighbor Discrete Inversion `trt_fp16_nn`):** Discretizing before nearest-neighbor inversion degrades Mean Dice by `-0.005004` (`0.960230`). Kernel #7 (`fused_invert_trilinear_threshold_u8`) preserves `100%` trilinear accuracy in `2.62 ms`.
- **Problem C (Paradoxical Masking of the `+1.68 cm³` FP16 Bug by Ground-Truth Capsule Under-Segmentation):** Because the pretrained 3D U-Net under-segments the outer 1-voxel splenic capsule relative to human annotations (`-5.17 ± 5.18 cm³`), the unfolded engine's `+1.68 cm³` dilation partially canceled against ground truth (`4.24 cm³` vs `5.17 cm³` MAE). Dual Bland-Altman and TOST testing against strict FP32 isolates quantization drift from anatomical error.
- **Problem D (The "All-Zeros / Blank Patch" Unit-Test Trap):** Evaluates real anatomical patches centered on the spleen center of mass in `spleen_6` (`patch.std() > 0.05`) across all `169` sliding-window batches (`probability_mae = 1.66e-06`).
- **Problem E (TensorRT 10.x Default-Stream Serialization & Laptop GPU DVFS Clock Inflation `P0 -> P4/P8`):** Binds TensorRT to a dedicated non-default CUDA stream (`self.exec_stream`) and documents the `1.13x` idle clock inflation after 3-second CPU gaps.
- **Problem F (Confounded Speedup Claims & Non-Additivity of Marginal Stage Medians):** Separates matched-window speedups (`trt_fp16_cuda`, `0.5/B=4`: `30.67x` cohort / `33.14x` paired mean) from fast-window speedups (`trt_fp16_optimized`, `0.25/B=8`: `42.86x` cohort / `46.64x` paired median) and reports all six stages explicitly.
- **Problem G (False-Positive / False-Negative Volumetric Cancellation at the Splenic Hilum `spleen_62` vs `spleen_56`):** Pairs volumetric bias with 3D surface distances (`HD95`, `ASSD`) and polar-vs-equatorial voxel error decomposition.
- **Problem H (Process-Wide PyTorch TF32 Flag Bleed Across Cached Streamlit Predictors):** Enforces `apply_precision(profile)` per prediction call in `app/workstation.py`, verified by `tests/test_workstation.py`.

**Software & Verification Layers:** (1) SHA-256 fingerprinted core package (`src/medtrt/`, `code_fingerprint = e9d4f060aaf1b130`), (2) 10-page analytics dashboard and 3D tri-planar workstation (`app/viewer.py`, `app/workstation.py`), and (3) 14-check offline claim verifier (`src/medtrt/verify_claims.py`) + 66 `pytest` tests + GitHub Actions CI.

---

## 6. Master Quantitative Results, Multi-Metric Validation & Paired TOST Equivalence

### Table 3: Complete 13-Profile Benchmark Matrix on Cohort 1 (MSD `N=9`)

| Profile | Path / I/O | Win/B | Ingest | Pre | Net | Agg | Post | E2E Med [IQR] (ms) | Cohort / Paired Speedup | VRAM | Mean Dice ± SD | Vol Bias vs FP32 Ref |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `pytorch_fp32_strict` | CPU / gzip | `0.5/4` | `367.4` | `288.0` | `379.9` | `8.4` | `2024.8` | `3049.8` [`3235.4`] | `0.97x` / `0.94x` | `490 MB` | `0.965233 ± 0.0117` | Reference (`0.000 cm³`) |
| `pytorch_fp32` | CPU / gzip | `0.5/4` | `424.7` | `286.4` | `254.6` | `8.0` | `1990.4` | `2,952.0` [`2970.8`] | `1.00x` / `1.00x` | `487 MB` | `0.965228 ± 0.0117` | `+0.005 ± 0.004 cm³` |
| `pytorch_tf32` | CPU / gzip | `0.5/4` | `424.7` | `286.4` | `254.6` | `8.0` | `1990.4` | `2,952.0` [`2970.8`] | `1.00x` / `1.00x` | `487 MB` | `0.965228 ± 0.0117` | `+0.005 ± 0.004 cm³` |
| `pytorch_amp` | CPU / gzip | `0.5/4` | `419.7` | `271.7` | `213.4` | `9.1` | `2013.6` | `2916.1` [`2954.5`] | `1.01x` / `0.99x` | `457 MB` | `0.965217 ± 0.0118` | `+0.003 ± 0.011 cm³` |
| `pytorch_fp16` | CPU / gzip | `0.5/4` | `340.1` | `281.6` | `213.9` | `9.6` | `1993.2` | `2885.5` [`2822.2`] | `1.02x` / `1.01x` | `458 MB` | `0.965231 ± 0.0118` | `+0.025 ± 0.015 cm³` |
| `trt_fp32` | CPU / gzip | `0.5/4` | `373.0` | `276.7` | `239.8` | `8.9` | `1981.5` | `2940.0` [`3003.2`] | `1.00x` / `1.00x` | `368 MB` | `0.965233 ± 0.0117` | `+0.000 ± 0.000 cm³` |
| `trt_fp16_unfolded` (Bug) | CPU / gzip | `0.5/4` | `406.5` | `273.2` | `176.7` | `36.7` | `1994.5` | `2866.4` [`2909.5`] | `1.03x` / `1.03x` | `359 MB` | `0.965722 ± 0.0120` | **`+1.675 ± 0.553 cm³`** |
| `trt_fp16` (Folded) | CPU / gzip | `0.5/4` | `385.1` | `266.8` | `167.1` | `36.0` | `1993.9` | `2880.3` [`2893.9`] | `1.02x` / `1.04x` | `359 MB` | `0.965226 ± 0.0117` | `+0.003 ± 0.022 cm³` |
| `pytorch_tf32_gpu` | GPU / mmap | `0.5/4` | `38.8` | `54.4` | `197.0` | `6.5` | `62.2` | `357.9` [`244.6`] | `8.25x` / `8.76x` | `2,307 MB` | `0.965227 ± 0.0117` | `+0.003 ± 0.005 cm³` |
| `trt_fp16_gpu` | GPU / mmap | `0.5/4` | `38.9` | `43.9` | `47.6` | `6.5` | `71.1` | `215.5` [`173.7`] | `13.70x` / `14.67x` | `2,301 MB` | `0.965228 ± 0.0117` | `+0.001 ± 0.017 cm³` |
| **`trt_fp16_cuda` (L4L)** | **CUDA / mmap** | **`0.5/4`** | **`29.2`** | **`1.3`** | **`49.0`** | **`6.1`** | **`2.6`** | **`96.3`** [**`58.9`**] | **`30.67x` / `33.14x`** | **`455 MB`** | **`0.965229 ± 0.0117`** | **`+0.001 ± 0.019 cm³`** |
| **`trt_fp16_optimized` (Fast)** | **CUDA / mmap** | **`0.25/8`** | **`29.1`** | **`1.3`** | **`24.8`** | **`4.5`** | **`2.8`** | **`68.9`** [**`50.5`**] | **`42.86x` / `44.87x`** | **`522 MB`** | **`0.964778 ± 0.0135`** | `+0.120 ± 1.182 cm³` |
| `trt_fp16_nn` (Control) | GPU / mmap | `0.25/8` | `38.1` | `41.7` | `20.8` | `4.1` | `69.3` | `178.4` [`141.3`] | `16.55x` / `16.89x` | `2,149 MB` | `0.960230 ± 0.0139` | `+0.481 ± 1.073 cm³` |

### Table 6: Ten-Metric Overlap, Surface Distance, and Volumetry Summary (MSD `N=9`)

| Execution Profile | 3D Dice | IoU | Sensitivity | Precision | Specificity | Vol. Similarity | HD95 (mm) | ASSD (mm) | $V_{\text{pred}}$ (cm³) | MAE vs GT / Bias vs FP32 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `pytorch_fp32_strict` (Ref) | `0.965233` | `0.933019` | `0.956191` | `0.974558` | `0.999853` | `0.990460` | `2.18` | `0.389` | `281.56 ± 128.29` | `5.17 cm³` / `0.0000 cm³` |
| `trt_fp16_unfolded` (Bug) | `0.965722` | `0.933942` | `0.959719` | `0.971909` | `0.999838` | `0.992356` | `2.45` | `0.391` | `283.24 ± 128.83` | `4.24 cm³` / **`+1.6754 cm³`** |
| **`trt_fp16_cuda` (Folded)** | **`0.965229`** | **`0.933011`** | **`0.956189`** | **`0.974553`** | **`0.999853`** | **`0.990461`** | **`2.18`** | **`0.389`** | **`281.56 ± 128.29`** | `5.17 cm³` / **`+0.0010 cm³`** |
| **`trt_fp16_optimized` (Fast)** | **`0.964778`** | **`0.932236`** | **`0.955821`** | **`0.973997`** | **`0.999853`** | **`0.990435`** | **`2.35`** | **`0.417`** | **`281.68 ± 128.52`** | `5.10 cm³` / `+0.1196 cm³` |
| `trt_fp16_nn` (Nearest Inv) | `0.960230` | `0.923798` | `0.951877` | `0.968824` | `0.999822` | `0.991007` | `2.44` | `0.467` | `282.04 ± 128.54` | `4.75 cm³` / `+0.4814 cm³` |

### Table 8: Formal Paired TOST Equivalence Analysis (`α = 0.05`) Across Cohorts (`N=9`, `N=12`, `N=21`)

| Cohort (`N` Patients) | Comparison vs Strict FP32 | Margin `±Δ` | Mean Diff ± SD | 90% CI (TOST Bounds) | 95% CI | `p_TOST` | Equivalent? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| MSD Internal (`N=9`) | `trt_fp16_unfolded` Volume | `±0.50 cm³` | `+1.6754 ± 0.5531 cm³` | `[+1.3326, +2.0183] cm³` | `[+1.2503, +2.1006] cm³` | `0.9999` | **No (Fails)** |
| MSD Internal (`N=9`) | **`trt_fp16_cuda` (Folded) Volume** | **`±0.10 cm³`** | **`+0.0010 ± 0.0194 cm³`** | **`[-0.0110, +0.0130] cm³`** | `[-0.0139, +0.0159] cm³` | **`< 1e-6`** | **Yes (Pass)** |
| MSD Internal (`N=9`) | **`trt_fp16_cuda` (Folded) Dice** | **`±0.0010`** | **`-0.000004 ± 0.000016`** | **`[-0.000014, +0.000006]`** | `[-0.000017, +0.000009]` | **`< 1e-6`** | **Yes (Pass)** |
| FLARE22 External (`N=12`) | `trt_fp16_unfolded` Volume | `±0.50 cm³` | `+1.5366 ± 0.7680 cm³` | `[+1.1385, +1.9348] cm³` | `[+1.0487, +2.0246] cm³` | `0.9997` | **No (Fails)** |
| FLARE22 External (`N=12`) | **`trt_fp16_cuda` (Folded) Volume** | **`±0.10 cm³`** | **`-0.0110 ± 0.0315 cm³`** | **`[-0.0274, +0.0053] cm³`** | `[-0.0311, +0.0090] cm³` | **`< 1e-6`** | **Yes (Pass)** |
| FLARE22 External (`N=12`) | **`trt_fp16_cuda` (Folded) Dice** | **`±0.0010`** | **`-0.000041 ± 0.000092`** | **`[-0.000089, +0.000007]`** | `[-0.000099, +0.000018]` | **`< 1e-6`** | **Yes (Pass)** |
| **Pooled Cohorts (`N=21`)** | `trt_fp16_unfolded` Volume | `±0.50 cm³` | `+1.5961 ± 0.6721 cm³` | `[+1.3432, +1.8491] cm³` | `[+1.2902, +1.9021] cm³` | `1.0000` | **No (Fails)** |
| **Pooled Cohorts (`N=21`)** | **`trt_fp16_cuda` (Folded) Volume** | **`±0.10 cm³`** | **`-0.0059 ± 0.0271 cm³`** | **`[-0.0161, +0.0043] cm³`** | `[-0.0182, +0.0064] cm³` | **`< 1e-6`** | **Yes (Pass)** |
| **Pooled Cohorts (`N=21`)** | **`trt_fp16_cuda` (Folded) Dice** | **`±0.0010`** | **`-0.000025 ± 0.000071`** | **`[-0.000052, +0.000002]`** | `[-0.000058, +0.000007]` | **`< 1e-6`** | **Yes (Pass)** |

---

## 7. Discussion, Limitations, and Future Work

See Sections 9–12 of the compiled 12-page PDF manuscript ([`../MedTRT_Scientific_Report.pdf`](../MedTRT_Scientific_Report.pdf) / [`TECHNICAL_PAPER.pdf`](TECHNICAL_PAPER.pdf)) for the complete discussion of interactive radiological workstations, numerical conditioning for reduced-precision medical AI, register-level GPU co-residency, the 9-point scope and limitations analysis, and 8 future research directions.
