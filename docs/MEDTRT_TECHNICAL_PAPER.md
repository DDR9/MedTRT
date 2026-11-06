# MedTRT: Eliminating End-to-End I/O, Resampling, and FP16 Quantization Bottlenecks in 3D CT Organ Segmentation

**Author:** Diego Díaz Rivera  
**Affiliation:** Biomedical AI Systems & GPU Performance Engineering (`MedTRT Research Study`, NVIDIA GeForce RTX 5080 Blackwell `sm_120`)  
**Date:** November 6, 2026  
**LaTeX / PDF Edition:** [`docs/MEDTRT_TECHNICAL_PAPER.pdf`](MEDTRT_TECHNICAL_PAPER.pdf) ([`docs/MEDTRT_TECHNICAL_PAPER.tex`](MEDTRT_TECHNICAL_PAPER.tex))

---

## Abstract

**Problem:** Deploying 3D convolutional neural networks for volumetric computed tomography (CT) segmentation is widely assumed to be compute-bound by the neural network forward pass. Profiling a pretrained MONAI 3D Residual U-Net (`spleen_ct_segmentation` v0.6.1) across **53 abdominal CT volumes** (`41` Medical Segmentation Decathlon `Task09_Spleen` scans plus `12` external multi-center MICCAI `FLARE22` scans archived on Google Drive) reveals two critical bottlenecks:
1. **Amdahl's Law Non-Network Bottleneck (`91.4%` of Runtime):** The 3D U-Net forward pass (`254.6 ms`) accounts for only `8.6%` of the `2,952.0 ms` host-path end-to-end latency; `91.4%` is consumed by CPU trilinear inverse resampling (`1,990.4 ms`), `.nii.gz` decompression (`424.7 ms`), and CPU spatial resampling (`286.4 ms`). Moving spatial transforms to MONAI's GPU `grid_sample` backend (`215.5 ms`) allocates dense `float64` 3D coordinate meshgrids requiring `2,301 MB` of VRAM (`5.1×` larger than the network).
2. **Systematic FP16 Volumetric Drift (`+1.68 cm³`):** Compiling the exported ONNX graph directly into TensorRT FP16 introduces a systematic volumetric over-segmentation bias of **`+1.675 ± 0.553 cm³`** (`9/9` internal scans, $p < 0.0001$) and **`+1.537 ± 0.768 cm³`** (`12/12` external scans) caused by an unfused decoder `ConvTranspose3d+BatchNorm3d` pair (`model.2.0.adn.N`) where an affine shift of `-81.79` at $\text{ULP}_{\text{FP16}} = 0.0625$ induces catastrophic floating-point cancellation.

**Approach:** Around the frozen public weights, we engineered:
1. An analytical `float64` pre-export weight transformation that algebraically folds `BatchNorm3d` into `ConvTranspose3d` prior to ONNX export and TensorRT 10.16 FP16 compilation;
2. Nine custom CUDA C++17 3D kernels (`sm_120`) that compute $3 \times 4$ voxel-to-voxel affine coordinates directly in registers via fused multiply-add (`__fmaf_rn`), eliminating 3D coordinate meshgrids and fusing Hounsfield Unit windowing (`1.27 ms`) and trilinear postprocessing (`2.62 ms`); and
3. A zero-copy TensorRT execution engine paired with OS page-cache memory-mapped (`mmap`) `.nii` ingestion.

**Results:** Under identical sliding-window settings (`overlap=0.5, B=4`), our like-for-like custom CUDA + TensorRT FP16 pipeline (`trt_fp16_cuda`) reduces median end-to-end latency from `2,952.0 ms` to **`96.3 ms`** (**`30.7×`** cohort median speedup; paired patient-level mean speedup **`33.1×`**, 95% CI `[27.6×, 38.7×]`) and cuts peak VRAM from `2,301 MB` to **`455 MB`** (**`5.1×` reduction**). A fast sliding-window profile (`trt_fp16_optimized`, `overlap=0.25, B=8`) reaches **`68.9 ms`** median latency (**`42.9×`** cohort speedup; paired patient-level median **`46.6×`**). Analytical `ConvTranspose3d+BatchNorm3d` folding reduces mean absolute logit error by **`35.6×`** and achieves **near-FP32 segmentation agreement under the evaluated conditions**: across the pooled $N = 21$ internal and external validation patients, paired Two One-Sided Tests (TOST) establish statistical equivalence within $\Delta_V = \pm 0.10\text{ cm}^3$ (mean bias `-0.0059 ± 0.0271 cm³`, 90% CI `[-0.0161, +0.0043] cm³`, $p_{\text{TOST}} < 10^{-6}$) and $\Delta_{\text{Dice}} = \pm 0.001$ ($p_{\text{TOST}} < 10^{-6}$), with only **`0.009%`** (`8.8 / 95,529`) of organ voxels flipping label on internal scans and **`0.019%`** (`23.2 / 167,879`) on unseen external `FLARE22` scans.

---

## 1. Introduction & Problem Identification

Three-dimensional convolutional architectures such as the 3D Residual U-Net [1, 2] are standard backbones for volumetric CT organ segmentation. Because an abdominal CT volume (`512 × 512 × 31–135` voxels) exceeds typical GPU memory if processed monolithically, clinical research pipelines in MONAI [3] reorient the volume to RAS coordinates, resample to canonical spacing (`1.5 × 1.5 × 2.0 mm`), execute sliding-window inference over overlapping `96 × 96 × 96` patches, and invert the resampled probability volume back to native scanner coordinates.

Profiling the public MONAI `spleen_ct_segmentation` v0.6.1 bundle [4, 5] on the Medical Segmentation Decathlon (MSD) `Task09_Spleen` dataset [6, 7] and the external multi-center MICCAI `FLARE22` dataset [8] on an NVIDIA GeForce RTX 5080 Laptop GPU (Blackwell `sm_120`) exposed three systems and numerical obstacles:

1. **Amdahl's Law Non-Network Bottleneck (`91.4%` of Runtime):** In the baseline host pipeline (`pytorch_fp32`), end-to-end median latency is `2,952.0 ms`, whereas the 3D U-Net forward pass (`sw_network`) takes only `254.6 ms` (`8.6%`). By Amdahl's Law [9], $S_{\max} = (1 - 0.0862)^{-1} = 1.094\times$. Replacing PyTorch with TensorRT FP16 on the host path (`trt_fp16`) lowers network time to `167.1 ms` but yields only a `1.02×` end-to-end speedup (`2,880.3 ms`) because CPU trilinear `Invertd` (`1,993.9 ms`), `.nii.gz` decompression (`385.1 ms`), and CPU `Spacingd` (`266.8 ms`) dominate execution and allow the laptop GPU to downclock to `180 MHz` (`P4`) between scans (`2.34×` idle clock penalty).
2. **MONAI GPU Meshgrid VRAM Overhead (`2,301 MB`):** Moving transforms onto the GPU via MONAI's `grid_sample` backend (`trt_fp16_gpu`) reduces median latency to `215.5 ms` but allocates dense `float64` 3D coordinate meshgrids $(1, D, H, W, 3)$, inflating peak VRAM to `2,301.3 MB`. Inverting discretized labels via nearest-neighbor interpolation (`trt_fp16_nn`) degrades mean Dice by `-0.0050` (`0.960230` vs. `0.965233`).
3. **Systematic FP16 Volumetric Drift (`+1.68 cm³`):** Compiling the exported ONNX graph directly into TensorRT FP16 (`trt_fp16_unfolded`) inflates segmented spleen volume by `+1.6754 ± 0.5531 cm³` ($p < 0.0001$) on `9/9` internal MSD scans and `+1.5366 ± 0.7680 cm³` ($p < 0.0001$) on `12/12` external `FLARE22` scans. Layer-by-layer inspection revealed that PyTorch's ONNX exporter folds `BatchNorm3d` into `Conv3d` in the encoder, but leaves `BatchNorm3d` unfused after `ConvTranspose3d` in the decoder. In layer `model.2.0.adn.N`, `ConvTranspose3d` activations average $\mu = 57.62$ before `BatchNorm3d` applies an affine shift of `-81.79`. In IEEE-754 FP16 (10-bit significand), values in $[64, 128)$ have $\text{ULP} = 2^{-4} = 0.0625$, causing catastrophic cancellation that shifts background logits by `-0.159` and foreground logits by `+0.116`.

---

## 2. Experimental Setup & Methodology

### 2.1 Hardware, Power Limits, & Timing Protocol
All experiments executed on a single workstation equipped with an **NVIDIA GeForce RTX 5080 Laptop GPU** (Blackwell architecture, compute capability `12.0` / `sm_120`, `15.92 GB` GDDR7 VRAM, `90 W` dynamic TGP limit under Windows 11 build `10.0.26300` in `WDDM` driver mode, NVIDIA Driver `591.44`, CUDA `12.8`, cuDNN `9.19.0`, TensorRT `10.16.1.11`, PyTorch `2.11.0+cu128`, MONAI `1.6.1`, and Python `3.11.15`). Active GPU execution runs in performance state `P0` (`1,500+ MHz` graphics clock), whereas host-bound stalls over `2 s` allow hardware power management to drop to `P4` (`180 MHz`).

Each benchmark profile executes `3` untimed warm-up volumes (stabilizing `P0` clocks, cuDNN autotuning, and CUDA allocator pools) followed by `10` timed passes across each of the `9` internal validation volumes (`90` timed passes per profile). Stage boundaries (`load`, `h2d`, `preprocess`, `sw_network`, `sw_aggregate`, `postprocess`, and `e2e_total`) are bracketed by hardware `torch.cuda.Event(enable_timing=True)` pairs synchronized via `torch.cuda.synchronize()` and cross-checked against host `time.perf_counter()` wall-clock timers (worst median discrepancy across all 13 profiles: `0.26%`). Every timed pass ends with the discretized `uint8` 3D segmentation mask transferred back to host RAM.

### 2.2 Ingestion Modes & Stage-Sum Accounting
We distinguish three I/O regimes measured across the 9 MSD volumes (`31–135 MB` uncompressed):
1. **CPU `.nii.gz` gzip decompression:** `265.0 ms` standalone median (`385.1–424.7 ms` inside the full host pipeline);
2. **Cold unbuffered `.nii` disk reads (`open(..., buffering=0)`):** `23.8 ms` median on local NVMe; and
3. **Warm OS page-cache memory mapping (`nibabel.load(..., mmap=True)`):** `load` takes `1.5 ms` to construct the virtual mapping and the page-cached bytes are faulted directly during the `27.6 ms` host-to-device copy, yielding `ingest = load + h2d = 29.2 ms` median on local NVMe and `175.4 ms` over the virtualized Google Drive mount.

Crucially, in tables reporting medians across heterogeneous CT volumes (`31` to `135` slices, `32` to `125` sliding windows), the sum of marginal stage medians is mathematically distinct from the median of end-to-end sums ($\sum_k \text{median}(T_k) \neq \text{median}(\sum_k T_k)$), and omitting `ingest` (`29.2 ms`) and `sw_aggregate` (`6.1 ms`) creates an apparent gap between compute stages (`1.3 + 49.0 + 2.6 = 52.9 ms`) and total latency (`96.3 ms`). Table 1 therefore reports **all six pipeline stages explicitly**, where the sum of all six stage medians on `trt_fp16_cuda` is `29.2 + 1.3 + 49.0 + 6.1 + 2.6 = 88.2 ms` vs. `96.3 ms` median E2E (within single-case medians, e.g., `spleen_59`, the six stages sum to `24.8 + 1.3 + 54.3 + 6.7 + 2.1 = 89.2 ms` vs. `89.7 ms` E2E).

### 2.3 Two-Cohort Validation Design & Statistical Rigor
To separate repeated-pass timing precision (`90` passes) from patient-level biological variation ($N$ independent subjects), all accuracy, volumetric bias, and paired speedup confidence intervals are computed at the **patient level**:
- **Cohort 1 (Internal MSD `Task09_Spleen`, $N = 9$ validation / $41$ total CT scans):** Following `configs/train.json` (`@images[-9:]`), the final 9 alphanumerically sorted volumes (`spleen_56` to `spleen_9`) form the held-out split. While disjoint from the 32 gradient-update scans, the bundle's `CheckpointSaver` selected `model.pt` using validation Dice on these 9 scans; thus absolute Dice (`0.9652`) is an optimistic internal estimate, whereas cross-backend comparisons against strict IEEE FP32 (`pytorch_fp32_strict`) are exact. All `41` MSD volumes are also benchmarked on GPU (`68.9 ms` median on $N=9$; `90.4 ms` across all $N=41$).
- **Cohort 2 (External Unseen MICCAI `FLARE22`, $N = 12$ multi-center CT scans on Google Drive):** To provide an unbiased external test of both generalization and numerical equivalence, we curated `12` independent abdominal CT scans (`FLARE22_Tr_0001` to `0012`, `512 × 512 × 84–110` voxels, spacing `0.64–0.86 × 2.5 mm`, ground-truth spleen volumes `130.8–481.7 cm³`) from the official MICCAI FLARE22 dataset (Zenodo Record `7860267` [8]), archived persistently on Google Drive (`$MEDTRT_FLARE22_DIR`).
- **Formal TOST Equivalence Testing:** Because a non-significant two-sided $t$-test ($p = 0.885$) only fails to reject a zero difference and cannot prove equivalence, we perform paired **Two One-Sided Tests (TOST)** [18] at $\alpha = 0.05$ against predefined equivalence margins of $\Delta_V = \pm 0.10\text{ cm}^3$ ($< 0.05\%$ of mean spleen volume) and $\Delta_V = \pm 0.50\text{ cm}^3$, and $\Delta_{\text{Dice}} = \pm 0.001$ and $\pm 0.005$.

---

## 3. Engineering & Numerical Solutions

### 3.1 Analytical FP64 `ConvTranspose3d+BN` Folding
Prior to ONNX (`opset 17`) export [11], we algebraically fold each decoder `BatchNorm3d` ($\gamma, \beta, \mu, \sigma^2, \epsilon$) into its preceding `ConvTranspose3d` weight $W$ and bias $b$ in `float64`:

$$s_c = \frac{\gamma_c}{\sqrt{\sigma_c^2 + \epsilon}}, \quad \widetilde{W}_{:, c, :, :, :} = W_{:, c, :, :, :} \cdot s_c, \quad \widetilde{b}_c = (b_c - \mu_c)s_c + \beta_c$$

and replace `BatchNorm3d` with `Identity()`. In FP32, this alters logits by at most $1.34 \times 10^{-5}$, while eliminating the `-81.79` intermediate shift prior to FP16 quantization in TensorRT 10.16 [12].

### 3.2 Custom Register-Affine CUDA C++17 Kernels
We implemented 9 custom CUDA C++17 kernels (`cuda/medical_kernels.cu`) compiled for `sm_120` via NVRTC [13]:
- **Fused Preprocessing (`fused_preprocess_affine_trilinear_hu_f32`):** Computes target-to-source coordinates $(x_s, y_s, z_s) = M_{\text{pre}}[x_d, y_d, z_d, 1]^T$ in 12 scalar registers via `__fmaf_rn`, performs trilinear interpolation via read-only `__ldg()` cache loads, and clamps HU intensities ($[-57, 164] \to [0, 1]$) in **`1.27 ms`** median (**`33.8×` faster** than `43.9 ms` MONAI GPU and **`225×` faster** than `286.4 ms` CPU).
- **Fused Postprocessing (`fused_softmax_2ch_p1_f32` + `fused_invert_trilinear_threshold_u8`):** Computes foreground probability $p_1 = (1 + \exp(L_0 - L_1))^{-1}$, applies the register-affine inverse $M_{\text{inv}} = M_{\text{pre}}^{-1}$ via trilinear interpolation, and thresholds at $p_1 \ge 0.5$ into a `uint8` mask in **`2.62 ms`** median (**`27.1×` faster** than `71.1 ms` MONAI GPU and **`760×` faster** than `1,990.4 ms` CPU), achieving `100.0%` exact voxel parity with MONAI GPU trilinear inversion.

---

## 4. Results & Multi-Cohort Validation

### Table 1: Master 90-Pass Benchmark & Quantitative Validation Summary (Cohort 1: MSD $N=9$, 90 Passes/Profile)

| Profile | Path / I/O | Win/B | Ingest (ms) | Pre (ms) | Net (ms) | Agg (ms) | Post (ms) | E2E Med [IQR] (ms) | Cohort / Paired Speedup | VRAM | Mean Dice ± SD | Vol Bias vs. FP32 Ref |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `pytorch_fp32_strict` | CPU / gzip | `0.5/4` | `367.4` | `288.0` | `379.9` | `8.4` | `2024.8` | `3049.8` [`3235.4`] | `0.97×` / `0.97×` | `490 MB` | `0.965233 ± 0.0117` | Reference (`0.000 cm³`) |
| `pytorch_fp32` | CPU / gzip | `0.5/4` | `424.7` | `286.4` | `254.6` | `8.0` | `1990.4` | `2952.0` [`2970.8`] | `1.00×` / `1.00×` | `487 MB` | `0.965228 ± 0.0117` | `+0.005 ± 0.004 cm³` |
| `trt_fp16_unfolded` | CPU / gzip | `0.5/4` | `406.5` | `273.2` | `176.7` | `36.7` | `1994.5` | `2866.4` [`2909.5`] | `1.03×` / `1.03×` | `359 MB` | `0.965722 ± 0.0120` | **`+1.675 ± 0.553 cm³`** |
| `trt_fp16` (Folded) | CPU / gzip | `0.5/4` | `385.1` | `266.8` | `167.1` | `36.0` | `1993.9` | `2880.3` [`2893.9`] | `1.02×` / `1.03×` | `359 MB` | `0.965226 ± 0.0117` | `+0.003 ± 0.022 cm³` |
| `pytorch_tf32_gpu` | GPU / mmap | `0.5/4` | `38.8` | `54.4` | `197.0` | `6.5` | `62.2` | `357.9` [`244.6`] | `8.25×` / `8.77×` | `2307 MB` | `0.965227 ± 0.0117` | `+0.001 ± 0.004 cm³` |
| `trt_fp16_gpu` | GPU / mmap | `0.5/4` | `38.9` | `43.9` | `47.6` | `6.5` | `71.1` | `215.5` [`173.7`] | `13.70×` / `14.81×` | `2301 MB` | `0.965228 ± 0.0117` | `+0.001 ± 0.017 cm³` |
| **`trt_fp16_cuda`** | **CUDA / mmap** | **`0.5/4`** | **`29.2`** | **`1.3`** | **`49.0`** | **`6.1`** | **`2.6`** | **`96.3` [`58.9`]** | **`30.67×` / `33.14×`** | **`455 MB`** | **`0.965229 ± 0.0117`** | **`+0.001 ± 0.019 cm³`** |
| **`trt_fp16_optimized`** | **CUDA / mmap** | **`0.25/8`** | **`29.1`** | **`1.3`** | **`24.8`** | **`4.5`** | **`2.8`** | **`68.9` [`50.5`]** | **`42.86×` / `44.87×`** | **`522 MB`** | **`0.964778 ± 0.0135`** | `+0.120 ± 1.182 cm³` |
| `trt_fp16_nn` | GPU / mmap | `0.25/8` | `38.1` | `41.7` | `20.8` | `4.1` | `69.3` | `178.4` [`141.3`] | `16.55×` / `18.12×` | `2149 MB` | `0.960230 ± 0.0139` | `+0.481 ± 1.073 cm³` |

### Table 2: Controlled One-Factor-at-a-Time Ablation Study (Cohort 1: $N=9$)

| Ablation Step (Single Factor Changed) | E2E Med (ms) | Max (ms) | Speedup | Net (ms) | Mean Dice |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 1. Host path, PyTorch TF32 (`gzip + CPU`) | `2992.5` | `6339.3` | `1.00×` | `239.3` | `0.965228` |
| 2. `+` TensorRT FP16 on host path | `2923.0` | `6015.0` | `1.02×` | `170.3` | `0.965226` |
| 3. `+` MONAI GPU transforms (`gzip` read) | `706.9` | `1404.3` | `4.23×` | `201.1` | `0.965228` |
| 4. `+` TensorRT FP16 with GPU transforms | `564.3` | `1252.9` | `5.30×` | `47.2` | `0.965228` |
| 5. `+` Memory-mapped `.nii` read (TF32) | `363.3` | `686.1` | `8.24×` | `195.9` | `0.965228` |
| 6. `+` TensorRT FP16 with `mmap` + MONAI GPU | `274.8` | `472.5` | `10.89×` | `54.0` | `0.965228` |
| 7. **`+` Custom CUDA kernels (`trt_fp16_cuda`)** | **`96.3`** | **`232.6`** | **`31.07×`** | **`49.0`** | **`0.965229`** |
| 8. **`+` Fast windows (`ov=0.25, B=8`, `optimized`)** | **`68.9`** | **`122.6`** | **`43.43×`** | **`24.8`** | **`0.964778`** |
| 9. Nearest-neighbor label inverse (`trt_fp16_nn`) | `178.4` | `398.5` | `16.77×` | `20.8` | `0.960230` |

### Table 3: Per-Patient Paired Performance & Voxel Disagreement Across Cohort 1 ($N=9$) and External Cohort 2 (`FLARE22`, $N=12$)

| Cohort / Case ID | Shape ($X \times Y \times Z$) | Win (`0.5/0.25`) | Host FP32 (ms) | CUDA L4L (ms) | Fast Opt (ms) | Paired Speedup (L4L / Opt) | Dice (FP32 Ref) | Dice (CUDA L4L) | $\Delta\text{Dice}$ (L4L) | HD95 / ASSD (mm) | $V_{\text{GT}}$ ($\text{cm}^3$) | $\Delta V_{\text{unf}}$ / $\Delta V_{\text{cuda}}$ ($\text{cm}^3$) | Flipped Voxels (% Organ) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cohort 1: MSD `spleen_56`** | `512×512×33` | `32 / 18` | `1543.7` | `47.6` | `36.0` | `32.46× / 42.91×` | `0.936143` | `0.936164` | `+0.000021` | `4.25 / 0.55` | `78.07` | `+0.722 / -0.003` | `3 / 22,038` (`0.0136%`) |
| **Cohort 1: MSD `spleen_59`** | `512×512×50` | `72 / 50` | `2570.3` | `89.7` | `71.6` | `28.66× / 35.88×` | `0.966229` | `0.966251` | `+0.000022` | `2.85 / 0.52` | `426.65` | `+2.284 / +0.027` | `10 / 94,040` (`0.0106%`) |
| **Cohort 1: MSD `spleen_6`** | `512×512×135` | `72 / 50` | `6142.8` | `123.4` | `108.2` | `49.79× / 56.75×` | `0.973239` | `0.973226` | `-0.000013` | `1.60 / 0.48` | `209.52` | `+1.426 / +0.005` | `10 / 154,403` (`0.0065%`) |
| **Cohort 1: MSD `spleen_60`** | `512×512×97` | `125 / 64` | `4936.4` | `155.6` | `101.8` | `31.72× / 48.49×` | `0.971020` | `0.971015` | `-0.000005` | `1.59 / 0.25` | `299.84` | `+1.717 / -0.009` | `11 / 91,659` (`0.0120%`) |
| **Cohort 1: MSD `spleen_61`** | `512×512×101` | `125 / 64` | `5034.3` | `162.4` | `104.8` | `31.00× / 48.03×` | `0.971511` | `0.971502` | `-0.000009` | `1.09 / 0.30` | `163.30` | `+1.149 / +0.003` | `1 / 53,761` (`0.0019%`) |
| **Cohort 1: MSD `spleen_62`** | `512×512×80` | `64 / 27` | `3789.9` | `96.7` | `70.1` | `39.19× / 54.07×` | `0.961134` | `0.961119` | `-0.000015` | `2.44 / 0.43` | `275.93` | `+1.918 / +0.023` | `16 / 120,578` (`0.0133%`) |
| **Cohort 1: MSD `spleen_63`** | `512×512×60` | `75 / 32` | `2952.0` | `98.4` | `63.3` | `30.01× / 46.64×` | `0.963625` | `0.963618` | `-0.000007` | `2.71 / 0.45` | `356.60` | `+1.920 / +0.011` | `12 / 121,013` (`0.0099%`) |
| **Cohort 1: MSD `spleen_8`** | `512×512×31` | `50 / 32` | `1625.1` | `62.1` | `47.9` | `26.19× / 33.92×` | `0.973413` | `0.973384` | `-0.000029` | `1.57 / 0.24` | `501.55` | `+2.488 / -0.037` | `10 / 106,587` (`0.0094%`) |
| **Cohort 1: MSD `spleen_9`** | `512×512×41` | `50 / 32` | `1949.7` | `66.7` | `52.5` | `29.24× / 37.13×` | `0.970787` | `0.970786` | `-0.000001` | `1.49 / 0.30` | `269.15` | `+1.456 / -0.011` | `6 / 95,683` (`0.0063%`) |
| **MSD Mean ± SD ($N=9$)** | --- | `73.9 / 41.0` | `3393.8 ± 1699` | `100.3 ± 40.3` | `72.9 ± 26.5` | **`33.14× / 44.87×`** | `0.965233` | **`0.965229`** | `-0.000004` | **`2.18 / 0.39`** | `286.73` | **`+1.675 / +0.001`** | **`8.8` (`0.0093%`)** |
| **Cohort 2: `FLARE22_Tr_0001`** | `512×512×110` | `32 / 18` | `6367.5` | `257.9` | `243.3` | `24.69× / 26.18×` | `0.961213` | `0.961207` | `-0.000006` | `2.32 / 0.56` | `186.17` | `+0.987 / -0.002` | `16 / 175,548` (`0.0091%`) |
| **Cohort 2: `FLARE22_Tr_0002`** | `512×512×107` | `50 / 32` | `5912.2` | `276.2` | `241.3` | `21.40× / 24.50×` | `0.964399` | `0.964395` | `-0.000004` | `2.50 / 0.56` | `281.25` | `+1.436 / -0.010` | `17 / 198,142` (`0.0086%`) |
| **Cohort 2: `FLARE22_Tr_0003`** | `512×512×101` | `50 / 32` | `5327.0` | `253.6` | `236.9` | `21.01× / 22.49×` | `0.955464` | `0.955433` | `-0.000031` | `2.17 / 0.66` | `163.03` | `+0.795 / +0.007` | `13 / 107,077` (`0.0121%`) |
| **Cohort 2: `FLARE22_Tr_0004`** | `512×512×87` | `50 / 32` | `4679.4` | `232.9` | `198.5` | `20.09× / 23.58×` | `0.960944` | `0.960946` | `+0.000002` | `2.32 / 0.54` | `222.10` | `+1.171 / -0.001` | `15 / 164,121` (`0.0091%`) |
| **Cohort 2: `FLARE22_Tr_0005`** | `512×512×85` | `50 / 32` | `4915.9` | `221.5` | `206.5` | `22.19× / 23.81×` | `0.962419` | `0.962391` | `-0.000028` | `2.45 / 0.52` | `189.61` | `+1.114 / +0.002` | `13 / 122,226` (`0.0106%`) |
| **Cohort 2: `FLARE22_Tr_0006`** | `512×512×93` | `72 / 32` | `5596.4` | `263.3` | `222.6` | `21.26× / 25.14×` | `0.960910` | `0.960888` | `-0.000022` | `2.44 / 0.59` | `174.83` | `+0.941 / -0.004` | `12 / 91,579` (`0.0131%`) |
| **Cohort 2: `FLARE22_Tr_0007`** | `512×512×105` | `50 / 32` | `6092.1` | `257.9` | `247.3` | `23.62× / 24.64×` | `0.845178` | `0.845134` | `-0.000044` | `20.00 / 2.87` | `481.72` | `+2.913 / -0.035` | `52 / 229,102` (`0.0227%`) |
| **Cohort 2: `FLARE22_Tr_0008`** | `512×512×99` | `50 / 32` | `5821.1` | `265.6` | `207.2` | `21.91× / 28.10×` | `0.958981` | `0.958978` | `-0.000003` | `2.21 / 0.57` | `244.87` | `+1.395 / -0.002` | `15 / 157,108` (`0.0095%`) |
| **Cohort 2: `FLARE22_Tr_0009`** | `512×512×87` | `50 / 32` | `4878.4` | `231.1` | `220.9` | `21.11× / 22.08×` | `0.828880` | `0.828552` | `-0.000328` | `10.95 / 2.14` | `146.54` | `+3.200 / -0.104` | `80 / 65,505` (`0.1221%`) |
| **Cohort 2: `FLARE22_Tr_0010`** | `512×512×96` | `50 / 32` | `5168.0` | `251.6` | `197.9` | `20.54× / 26.11×` | `0.966501` | `0.966511` | `+0.000010` | `1.83 / 0.58` | `367.51` | `+1.829 / +0.010` | `8 / 209,759` (`0.0038%`) |
| **Cohort 2: `FLARE22_Tr_0011`** | `512×512×84` | `50 / 32` | `4422.4` | `229.0` | `204.2` | `19.31× / 21.66×` | `0.962932` | `0.962930` | `-0.000002` | `2.30 / 0.58` | `292.36` | `+1.561 / +0.007` | `15 / 189,184` (`0.0079%`) |
| **Cohort 2: `FLARE22_Tr_0012`** | `512×512×103` | `50 / 32` | `5596.1` | `265.1` | `241.1` | `21.11× / 23.21×` | `0.904354` | `0.904320` | `-0.000034` | `23.44 / 1.95` | `130.75` | `+1.099 / -0.002` | `15 / 73,200` (`0.0205%`) |
| **FLARE22 Mean ± SD ($N=12$)** | --- | `50.3 / 30.8` | `5398.0 ± 601` | `250.5 ± 17.5` | `222.3 ± 19.0` | **`21.52× / 24.29×`** | `0.936015` | **`0.935974`** | `-0.000041` | **`6.25 / 1.01`** | `240.06` | **`+1.537 / -0.011`** | **`22.6` (`0.0208%`)** |

### Table 4: Formal Paired Two One-Sided Tests (TOST) Equivalence Analysis vs. Strict IEEE FP32 ($\alpha = 0.05$)

| Cohort ($N$ Patients) | Comparison vs. Strict FP32 | Margin $\pm \Delta$ | Mean Diff ± SD | 90% CI (TOST Bounds) | 95% CI | $p_{\text{TOST}}$ | Equivalent? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| MSD Internal ($N = 9$) | `trt_fp16_unfolded` Volume | $\pm 0.50\text{ cm}^3$ | `+1.6754 ± 0.5531 cm³` | `[+1.3326, +2.0183] cm³` | `[+1.2503, +2.1006] cm³` | `0.9999` | **No (Fails)** |
| MSD Internal ($N = 9$) | **`trt_fp16_cuda` (Folded) Volume** | **$\pm 0.10\text{ cm}^3$** | **`+0.0010 ± 0.0194 cm³`** | **`[-0.0110, +0.0130] cm³`** | `[-0.0139, +0.0159] cm³` | **`< 1e-6`** | **Yes (Pass)** |
| MSD Internal ($N = 9$) | **`trt_fp16_cuda` (Folded) Dice** | **$\pm 0.0010$** | **`-0.000004 ± 0.000016`** | **`[-0.000014, +0.000006]`** | `[-0.000017, +0.000009]` | **`< 1e-6`** | **Yes (Pass)** |
| FLARE22 External ($N = 12$) | `trt_fp16_unfolded` Volume | $\pm 0.50\text{ cm}^3$ | `+1.5366 ± 0.7680 cm³` | `[+1.1385, +1.9348] cm³` | `[+1.0487, +2.0246] cm³` | `0.9997` | **No (Fails)** |
| FLARE22 External ($N = 12$) | **`trt_fp16_cuda` (Folded) Volume** | **$\pm 0.10\text{ cm}^3$** | **`-0.0110 ± 0.0315 cm³`** | **`[-0.0274, +0.0053] cm³`** | `[-0.0311, +0.0090] cm³` | **`< 1e-6`** | **Yes (Pass)** |
| FLARE22 External ($N = 12$) | **`trt_fp16_cuda` (Folded) Dice** | **$\pm 0.0010$** | **`-0.000041 ± 0.000092`** | **`[-0.000089, +0.000007]`** | `[-0.000099, +0.000018]` | **`< 1e-6`** | **Yes (Pass)** |
| **Pooled Cohorts ($N = 21$)** | `trt_fp16_unfolded` Volume | $\pm 0.50\text{ cm}^3$ | `+1.5961 ± 0.6721 cm³` | `[+1.3432, +1.8491] cm³` | `[+1.2902, +1.9021] cm³` | `1.0000` | **No (Fails)** |
| **Pooled Cohorts ($N = 21$)** | **`trt_fp16_cuda` (Folded) Volume** | **$\pm 0.10\text{ cm}^3$** | **`-0.0059 ± 0.0271 cm³`** | **`[-0.0161, +0.0043] cm³`** | `[-0.0182, +0.0064] cm³` | **`< 1e-6`** | **Yes (Pass)** |
| **Pooled Cohorts ($N = 21$)** | **`trt_fp16_cuda` (Folded) Dice** | **$\pm 0.0010$** | **`-0.000025 ± 0.000071`** | **`[-0.000052, +0.000002]`** | `[-0.000058, +0.000007]` | **`< 1e-6`** | **Yes (Pass)** |

---

## 5. Implications & Conclusions

Productionizing 3D medical segmentation models requires systems optimization beyond neural network compilation. By Amdahl's Law, host decompression and CPU trilinear resampling mask GPU inference speedups until replaced by memory-mapped ingestion and register-fused 3D affine CUDA kernels (`30.7×` like-for-like and `42.9×` fast-window E2E speedup, `5.1×` lower VRAM). Furthermore, standard ONNX exporters leave decoder `ConvTranspose3d+BatchNorm3d` layers unfused, introducing a `+1.60 cm³` FP16 volume inflation across `21/21` internal and external CT scans that is completely eliminated ($p_{\text{TOST}} < 10^{-6}$ within $\pm 0.10\text{ cm}^3$) via analytical `float64` weight folding.

---

## 6. References & Full Attribution

1. **E. Kerfoot et al.**, "Left-Ventricle Quantification Using Residual U-Net," in *STACOM 2018, LNCS*, vol. 11395, pp. 371–380, Springer, 2019.
2. **Ö. Çiçek et al.**, "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation," in *MICCAI 2016, LNCS*, vol. 9901, pp. 424–432, 2016.
3. **M. J. Cardoso et al.**, "MONAI: An open-source framework for deep learning in healthcare," *arXiv:2211.02701*, 2022.
4. **MONAI Consortium**, "MONAI Model Zoo: `spleen_ct_segmentation` Bundle v0.6.1," Apache-2.0 License, 2024. https://github.com/Project-MONAI/model-zoo
5. **Y. Xia et al.**, "3D Semi-Supervised Learning with Uncertainty-Aware Multi-View Co-Training," *arXiv:1811.12506*, 2018.
6. **M. Antonelli et al.**, "The Medical Segmentation Decathlon," *Nature Communications*, vol. 13, no. 1, p. 4128, 2022 (CC-BY-SA 4.0).
7. **A. L. Simpson et al.**, "A large annotated medical image dataset for the development and evaluation of segmentation algorithms," *arXiv:1902.09063*, 2019.
8. **J. Ma et al.**, "Fast and Low-GPU-memory Abdomen CT Organ Segmentation: The FLARE Challenge," *Medical Image Analysis*, vol. 82, p. 102616, 2022, and Zenodo Record `7860267`, 2023.
9. **G. M. Amdahl**, "Validity of the single processor approach to achieving large scale computing capabilities," in *Proc. AFIPS*, pp. 483–485, 1967.
10. **A. Paszke et al.**, "PyTorch: An Imperative Style, High-Performance Deep Learning Library," in *NeurIPS*, vol. 32, 2019.
11. **J. Bai et al.**, "ONNX: Open Neural Network Exchange," 2019, and ONNX Runtime Developers, https://onnxruntime.ai/, 2021.
12. **NVIDIA Corporation**, *NVIDIA TensorRT 10.16 Developer Guide & CUDA 12.8 Runtime/Driver API Reference*, 2025.
13. **J. Nickolls, I. Buck, M. Garland, and K. Skadron**, "Scalable Parallel Programming with CUDA," *ACM Queue*, vol. 6, no. 2, pp. 40–53, 2008.
14. **M. Brett et al.**, "nipy/nibabel: 5.4.2," *Zenodo*, 2024; **C. R. Harris et al.**, "Array programming with NumPy," *Nature*, vol. 585, pp. 357–362, 2020; **P. Virtanen et al.**, "SciPy 1.0," *Nature Methods*, vol. 17, pp. 261–272, 2020.
15. **L. R. Dice**, "Measures of the amount of ecologic association between species," *Ecology*, vol. 26, no. 3, pp. 297–302, 1945.
16. **A. A. Taha and A. Hanbury**, "Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool," *BMC Medical Imaging*, vol. 15, no. 1, p. 29, 2015.
17. **J. M. Bland and D. G. Altman**, "Statistical methods for assessing agreement between two methods of clinical measurement," *The Lancet*, vol. 327, no. 8476, pp. 307–310, 1986.
18. **D. J. Schuirmann**, "A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability," *Journal of Pharmacokinetics and Biopharmaceutics*, vol. 15, no. 6, pp. 657–680, 1987.
