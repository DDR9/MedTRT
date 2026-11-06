# MedTRT: Eliminating End-to-End I/O, Resampling, and FP16 Quantization Bottlenecks in 3D CT Organ Segmentation

**Author:** Diego Díaz Rivera  
**Affiliation:** Department of Bioengineering / Biomedical AI Systems Engineering  
**Date:** November 6, 2026  
**Repository:** `MedTRT` (`feature/medtrt-engineering-validation`)  
**Hardware Platform:** NVIDIA GeForce RTX 5080 Laptop GPU (Blackwell `sm_120`, 16 GB GDDR7, CUDA 12.8, TensorRT 10.16.1.11)

---

## Abstract

**Problem Identified:** Deploying 3D convolutional neural networks for volumetric computed tomography (CT) segmentation is frequently assumed to be bottlenecked by neural network forward-pass compute. However, profiling a pretrained MONAI 3D Residual U-Net (`spleen_ct_segmentation` v0.6.1) on the Medical Segmentation Decathlon `Task09_Spleen` dataset (`41` patient CT volumes) reveals two critical system failures. First, by Amdahl's Law, the neural network forward pass (`254.6 ms`) accounts for only `8.6%` of the `2,952.0 ms` host-path end-to-end latency; the remaining `91.4%` is consumed by CPU trilinear inverse resampling (`1,990.4 ms`), `.nii.gz` gzip decompression (`424.7 ms`), and CPU spatial preprocessing (`286.4 ms`). Even when spatial transforms are moved to MONAI's GPU `grid_sample` backend (`215.5 ms` end-to-end), `float64` 3D coordinate meshgrid allocations consume `2,301 MB` (`2.30 GB`) of VRAM (`5.1×` larger than the network itself). Second, compiling the exported ONNX graph into an IEEE-754 half-precision (FP16) TensorRT engine introduces a systematic volumetric over-segmentation bias of `+1.675 ± 0.553 cm³` (`+1.68 cm³`) across 9 out of 9 validation scans ($p < 0.0001$). First-principles layer inspection proves that PyTorch's ONNX exporter folds `BatchNorm3d` into `Conv3d` in the encoder, but leaves `BatchNorm3d` unfused after `ConvTranspose3d` in the decoder, where an activation shift of `-81.79` in layer `model.2.0.adn.N` (at an FP16 unit-in-last-place step size of $\text{ULP} = 2^{-4} = 0.0625$) causes catastrophic floating-point cancellation.

**How We Solved It:** We engineered a three-part systems and numerical solution around the frozen public weights: (1) an analytical `float64` pre-export weight transformation that algebraically folds `BatchNorm3d` parameters into `ConvTranspose3d` weights and biases prior to ONNX export and TensorRT 10.16 FP16 compilation; (2) nine custom CUDA C++17 3D kernels (`sm_120`) that compute $3 \times 4$ voxel-to-voxel affine transformations directly in GPU registers via fused multiply-add (`fmaf`) instructions, eliminating intermediate 3D coordinate meshgrids and fusing Hounsfield Unit (HU) windowing (`1.27 ms` / `1.3 ms` median, a `33.8×` speedup over `43.9 ms` MONAI GPU preprocessing) and softmax-trilinear-discretization postprocessing (`2.62 ms` median); and (3) a zero-copy TensorRT execution wrapper on a dedicated CUDA stream paired with OS page-cache memory-mapped (`mmap`) `.nii` ingestion across all 41 patient volumes.

**Why This Is Important:** Across the 41-patient-volume cohort and 9 held-out validation scans, our like-for-like custom CUDA + TensorRT FP16 pipeline (`trt_fp16_cuda`, `overlap=0.5, batch=4`) reduces median end-to-end latency from `2,952.0 ms` to **`96.3 ms` (`30.7×` speedup)** and peak VRAM from `2,301 MB` (MONAI GPU) to **`455 MB` (`5.06×` / `5.1×` reduction)**, while our fast sliding-window configuration (`trt_fp16_optimized`, `overlap=0.25, batch=8`) achieves **`68.9 ms` median latency (`42.9×` speedup, `14.5 volumes/s`, slowest pass `122.6 ms`)**. Simultaneously, analytical `ConvTranspose3d + BatchNorm3d` folding reduces FP16 logit drift by **`35.6×`** and cuts cross-backend volumetric bias from `+1.68 cm³` (`9/9` scans) to **`+0.001 cm³`** (`+0.0010 ± 0.0194 cm³`, $p = 0.885$, **`0.9652` Mean Dice**, **`2.18 mm` HD95**, mean $|\Delta\text{Dice}| = 0.000014$), backed by an interactive tri-planar Streamlit workstation with real-time FP32-vs-FP16 difference heatmaps and a **65-test verification suite** auditing latency and memory metrics against raw hardware telemetry.

---

## 1. Introduction & Problem Identification

Three-dimensional medical image segmentation models, such as the 3D Residual U-Net [1, 2], operate on high-resolution volumetric CT scans containing $1.5 \times 10^7$ to $4.0 \times 10^7$ voxels per study. Because a full abdominal CT scan exceeds typical GPU activation memory when processed as a single volume, standard clinical research frameworks such as MONAI [3] resample the scan to a canonical isotropic/anisotropic voxel spacing (`1.5 × 1.5 × 2.0 mm`), execute sliding-window inference over overlapping `96 × 96 × 96` 3D patches, and resample the predicted probability map back to the native scanner coordinate grid.

When optimizing such pipelines for low-latency execution, standard practice focuses almost exclusively on accelerating the neural network forward pass via mixed precision (AMP) or TensorRT FP16 compilation [4]. However, our baseline audit of the public MONAI `spleen_ct_segmentation` (v0.6.1) bundle [5] on the Medical Segmentation Decathlon `Task09_Spleen` dataset [6] identified three fundamental problems:

1. **Amdahl's Law Non-Network Bottleneck (`92.2%` of Runtime):**
   In the host-path baseline (`pytorch_fp32`), total end-to-end latency is `3,341.1 ms` median (`IQR 3,180.1 ms`), whereas the 3D U-Net sliding-window forward pass (`sw_network`) requires only `260.8 ms` (`7.8%` of total time). By Amdahl's Law:
   $$S_{\text{max}} = \frac{1}{1 - p} = \frac{1}{1 - 0.0781} = 1.085\times$$
   Even an infinitely fast neural network accelerator cannot speed up the host pipeline by more than `8.5%`. Indeed, dropping TensorRT FP16 into the host pipeline (`trt_fp16`) reduces `sw_network` to `172.8 ms` but leaves end-to-end latency at `2,967.0 ms` (`1.13×`).

2. **GPU Resampling Memory Explosion (`2.30 GB` Peak VRAM):**
   Moving spatial transforms (`Orientationd`, `Spacingd`, `Invertd`) onto the GPU via MONAI's PyTorch `grid_sample` implementation (`trt_fp16_gpu`) reduces end-to-end latency to `218.0 ms`. However, `grid_sample` materializes dense 3D coordinate meshgrids `(1, D, H, W, 3)` in `float64`/`float32` global memory, driving peak allocated VRAM to **`2,301.3 MB`** (`3,667.7 MB` reserved) on a single `512 × 512 × 135` scan (`spleen_6`). Furthermore, attempting to accelerate postprocessing by inverting discrete `uint8` nearest-neighbor labels (`trt_fp16_nn`) degrades mean Dice by `-0.0050` (`0.965233 -> 0.960230`) while saving less than `37 ms`.

3. **Catastrophic FP16 Cancellation in Unfused Decoder Transpose Convolutions (`+1.68 cm³` Bias):**
   When the unmodified MONAI 3D U-Net is exported to ONNX (`opset 17`) and compiled into TensorRT 10.16 FP16 (`trt_fp16_unfolded`), it exhibits a statistically significant volumetric over-segmentation bias of **`+1.6754 ± 0.5531 cm³`** (`95% CI [+1.250, +2.101] cm³`, $p < 0.0001$), inflating predicted spleen volume on **9 out of 9** validation cases. Inspection of the ONNX computation graph revealed that PyTorch's constant folder fuses `BatchNorm3d` into `Conv3d` across all encoder blocks, but **fails to fold `BatchNorm3d` into `ConvTranspose3d`** across the decoder blocks. In the penultimate decoder layer (`model.2.0.adn.N`), the unfused `ConvTranspose3d` outputs activations around `running_mean = 57.62`, followed by a `BatchNorm3d` affine shift of:
   $$\beta - \frac{\gamma \cdot \mu}{\sqrt{\sigma^2 + \epsilon}} = -81.79$$
   In IEEE-754 binary16 (FP16, 10-bit mantissa), numbers in the interval $[64, 128)$ have a quantization step size of $\text{ULP} = 2^{6 - 10} = 2^{-4} = 0.0625$. Quantizing the large pre-normalization activation and subtracting `81.79` in FP16 causes severe catastrophic cancellation, shifting background channel-0 logits by `-0.159` and foreground channel-1 logits by `+0.116` right before the final $1 \times 1 \times 1$ output convolution.

---

## 2. Procedure & Methodology: How We Solved It

### 2.1 Dataset Split & Strict Data-Leakage Verification
All experiments use the public Medical Segmentation Decathlon `Task09_Spleen` dataset [6] (`41` portal-venous phase 3D abdominal CT scans in NIfTI format) and the pretrained MONAI `spleen_ct_segmentation` v0.6.1 checkpoint (`models/model.pt`, SHA-256 `502c312893994ec0...`) [5].
- **Held-Out Evaluation Split ($n = 9$):** Following `configs/train.json` (`@images[-9:]`), the 41 scans are sorted alphanumerically, reserving the first 32 scans strictly for bundle training and the final 9 scans (`spleen_56`, `spleen_59`, `spleen_6`, `spleen_60`, `spleen_61`, `spleen_62`, `spleen_63`, `spleen_8`, `spleen_9`) for evaluation.
- **Data Leakage Audit & Disclosure:** None of these 9 evaluation volumes ever contributed to gradient updates during training. However, as documented in `results/heldout_cases.json` and enforced by `verify_claims.py`, MONAI's `CheckpointSaver` selected `model.pt` based on validation Dice across these same 9 volumes (since official MSD `imagesTs` test labels are not publicly released). Consequently, absolute Dice (`~0.965`) is treated as an optimistic validation estimate, whereas cross-backend numerical and volumetric comparisons against strict IEEE FP32 (`pytorch_fp32_strict`) on these 9 volumes are exact and unbiased.

### 2.2 Analytical FP64 `ConvTranspose3d + BatchNorm3d` Folding
To eliminate the decoder FP16 quantization error without retraining or altering the mathematical function of the pretrained network, we implemented `fold_conv_transpose_bn()` in `src/medtrt/model.py`. For every `ConvTranspose3d` layer with weight tensor $W \in \mathbb{R}^{C_{\text{in}} \times C_{\text{out}} \times K_D \times K_H \times K_W}$ and bias $b \in \mathbb{R}^{C_{\text{out}}}$ followed immediately by `BatchNorm3d` ($\gamma, \beta, \mu, \sigma^2, \epsilon$), we compute the scale factor in `float64`:
$$s_c = \frac{\gamma_c}{\sqrt{\sigma_c^2 + \epsilon}}, \quad \widetilde{W}_{:, c, :, :, :} = W_{:, c, :, :, :} \cdot s_c, \quad \widetilde{b}_c = (b_c - \mu_c) \cdot s_c + \beta_c$$
and replace the `BatchNorm3d` module with `torch.nn.Identity()`. In strict FP32, the folded network matches the unmodified network to within a maximum logit difference of $1.34 \times 10^{-5}$ (floating-point rounding noise), while completely removing the `-81.79` intermediate offset before ONNX (`opset 17`) export and TensorRT 10.16 FP16 compilation (`engines/spleen_unet_fp16.engine`, dynamic batch `1..8`, `13.70 MB`).

### 2.3 Custom Register-Affine CUDA C++17 Kernels (`cuda/medical_kernels.cu`)
To eliminate MONAI's `1–2 GB` 3D meshgrid allocations and `float64` resampling overhead on GeForce hardware, we wrote 9 custom CUDA C++17 kernels compiled at runtime for Blackwell `sm_120` via NVRTC (`nvrtc64_120_0.dll`) and launched via the CUDA Driver API (`nvcuda.dll`):
1. **Fused 3D Affine Preprocessing (`fused_preprocess_affine_trilinear_hu_f32`):** Each GPU thread computes the exact source scanner coordinate $(x_s, y_s, z_s) = M_{\text{pre}} [x_d, y_d, z_d, 1]^T$ in 12 scalar registers using `__fmaf_rn`, performs 8-corner trilinear interpolation with `__ldg()` read-only cache loads, and fuses Hounsfield Unit clamping (`[-57, 164] -> [0, 1]`) in a single pass (`1.37 ms` median vs. `46.5–52.8 ms` MONAI GPU).
2. **Fused 3D Postprocessing (`fused_softmax_2ch_p1_f32` + `fused_invert_trilinear_threshold_u8`):** Converts 2-channel sliding-window logits to continuous foreground probabilities $p_1 = (1 + \exp(L_0 - L_1))^{-1}$, resamples $p_1$ back to the native CT scanner grid via the inverse affine matrix $M_{\text{inv}} = M_{\text{pre}}^{-1}$ in registers, and thresholds at $p_1 \ge 0.5$ directly into a compact `uint8` label map (`2.56 ms` median vs. `68.7–81.2 ms` MONAI GPU and `2,195.3 ms` CPU).

### 2.4 Zero-Copy TensorRT Stream Binding & Memory-Mapped Ingestion
In `src/medtrt/trt_runner.py`, PyTorch CUDA tensor pointers (`tensor.data_ptr()`) are bound directly to TensorRT 10.16 via `context.set_tensor_address()` and executed asynchronously via `context.execute_async_v3(stream)` on a dedicated non-default CUDA stream synchronized via lightweight `torch.cuda.Event` markers. For volume ingestion (`src/medtrt/data.py`), pre-decompressed `.nii` volumes are mapped via `nibabel.load(..., mmap=True)`, reducing total ingestion (`load + h2d`) from `430.1 ms` (`.nii.gz` gzip) to `24.8–27.3 ms`.

---

## 3. Experimental Results

All 13 execution profiles were benchmarked on an NVIDIA GeForce RTX 5080 Laptop GPU using **3 warm-up volumes followed by 10 timed passes on each of the 9 validation cases (90 timed passes per profile)**. Every stage boundary is timed with `torch.cuda.Event(enable_timing=True)` and cross-checked against wall-clock `time.perf_counter()` (worst median discrepancy across all 13 profiles is **`0.18%`**).

### 3.1 Controlled One-Factor-at-a-Time Ablation (`results/ablation.json`)
Table 1 isolates each engineering change sequentially across the 9 validation scans:

| Ablation Step | I/O | Transforms | Engine | Overlap / Batch | `sw_network` (ms) | E2E Median (ms) | Slowest Pass (ms) | Speedup vs. Row 1 | Mean Dice |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Host path, PyTorch TF32** | `.nii.gz` | CPU | PyTorch TF32 | `0.50` / `4` | `239.3` | `2,992.5` | `6,339.3` | `1.00×` | `0.965228` |
| **2. Host path, TensorRT only** | `.nii.gz` | CPU | TensorRT FP16 | `0.50` / `4` | `170.3` | `2,923.0` | `6,015.0` | `1.02×` | `0.965226` |
| **3. GPU transforms, gzip, TF32** | `.nii.gz` | MONAI GPU | PyTorch TF32 | `0.50` / `4` | `201.1` | `706.9` | `1,404.3` | `4.23×` | `0.965228` |
| **4. GPU transforms, gzip, TRT** | `.nii.gz` | MONAI GPU | TensorRT FP16 | `0.50` / `4` | `47.2` | `564.3` | `1,252.9` | `5.30×` | `0.965228` |
| **5. GPU transforms, `.nii`, TF32** | `.nii` mmap | MONAI GPU | PyTorch TF32 | `0.50` / `4` | `195.9` | `363.3` | `686.1` | `8.24×` | `0.965228` |
| **6. GPU transforms, `.nii`, TRT** | `.nii` mmap | MONAI GPU | TensorRT FP16 | `0.50` / `4` | `54.0` | `274.8` | `472.5` | `10.89×` | `0.965228` |
| **7. Fast windows, PyTorch TF32** | `.nii` mmap | MONAI GPU | PyTorch TF32 | `0.25` / `8` | `111.9` | `320.6` | `540.4` | `9.33×` | `0.964771` |
| **8. Fast windows, TensorRT** | `.nii` mmap | MONAI GPU | TensorRT FP16 | `0.25` / `8` | `23.5` | `218.5` | `374.2` | `13.70×` | `0.964767` |
| **9. Fast windows, TRT, nearest** | `.nii` mmap | MONAI GPU | TensorRT FP16 | `0.25` / `8` | `24.1` | `219.0` | `361.6` | `13.66×` | `0.960230` |

### 3.2 Full Profile Benchmark & Custom CUDA Kernels (`results/bench_*.json`, `results/cohort_41_volumes.json`)
Adding our custom CUDA C++17 register-affine kernels (`trt_fp16_cuda` and `trt_fp16_optimized`) eliminates the remaining GPU transform and VRAM bottlenecks:

| Execution Profile | Transforms / I/O | Overlap / Batch | `ingest` (ms) | `preprocess` (ms) | `sw_network` (ms) | `postprocess` (ms) | E2E Median [IQR] (ms) | Slowest (ms) | Peak VRAM (MB) | Mean Dice ± SD | Speedup vs. `pytorch_fp32` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `pytorch_fp32_strict` | `cpu` / `gzip` | `0.5` / `4` | `367.4` | `288.0` | `379.9` | `2,024.8` | `3,049.8` [`3,235.4`] | `6,562` | `490` | `0.965233 ± 0.0117` | `0.97×` |
| `pytorch_fp32` | `cpu` / `gzip` | `0.5` / `4` | `424.7` | `286.4` | `254.6` | `1,990.4` | `2,952.0` [`2,970.8`] | `6,208` | `487` | `0.965228 ± 0.0117` | `1.00×` |
| `pytorch_amp` | `cpu` / `gzip` | `0.5` / `4` | `419.7` | `271.7` | `213.4` | `2,013.6` | `2,916.1` [`2,954.5`] | `6,364` | `457` | `0.965217 ± 0.0118` | `1.01×` |
| `trt_fp32` | `cpu` / `gzip` | `0.5` / `4` | `373.0` | `276.7` | `239.8` | `1,981.5` | `2,940.0` [`3,003.2`] | `6,387` | `368` | `0.965233 ± 0.0117` | `1.00×` |
| `trt_fp16` | `cpu` / `gzip` | `0.5` / `4` | `385.1` | `266.8` | `167.1` | `1,993.9` | `2,880.3` [`2,893.9`] | `6,094` | `359` | `0.965226 ± 0.0117` | `1.02×` |
| `pytorch_tf32_gpu` | `gpu` / `mmap` | `0.5` / `4` | `38.8` | `54.4` | `197.0` | `62.2` | `357.9` [`244.6`] | `631` | `2,307` | `0.965227 ± 0.0117` | `8.25×` |
| `trt_fp16_gpu` | `gpu` / `mmap` | `0.5` / `4` | `38.9` | `43.9` | `47.6` | `71.1` | `215.5` [`173.7`] | `440` | `2,301` | `0.965228 ± 0.0117` | `13.70×` |
| **`trt_fp16_cuda`** | **`cuda` / `mmap`** | **`0.5` / `4`** | **`29.2`** | **`1.3`** | **`49.0`** | **`2.6`** | **`96.3` [`58.9`]** | **`233`** | **`455`** | **`0.965229 ± 0.0117`** | **`30.67×`** |
| **`trt_fp16_optimized`** | **`cuda` / `mmap`** | **`0.25` / `8`** | **`29.1`** | **`1.3`** | **`24.8`** | **`2.8`** | **`68.9` [`50.5`]** | **`123`** | **`522`** | **`0.964778 ± 0.0135`** | **`42.86×`** |
| `trt_fp16_nn` | `gpu` / `mmap` | `0.25` / `8` | `38.1` | `41.7` | `20.8` | `69.3` | `178.4` [`141.3`] | `399` | `2,149` | `0.960230 ± 0.0139` | `16.55×` |

### 3.3 Volumetric & Numerical Equivalence Validation (`results/bland_altman.json`)
Comparing the unfolded vs. folded TensorRT FP16 engines against strict IEEE FP32 (`pytorch_fp32_strict`, Mean Dice `0.965233`, Mean HD95 `2.18 ± 0.99 mm`, Mean ASSD `0.39 ± 0.12 mm`) demonstrates complete elimination of the systematic FP16 drift:

| Profile | Mean Dice | Mean $|\Delta\text{Dice}|$ | Max $|\Delta\text{Dice}|$ | Volume Bias ± SD ($\text{cm}^3$) | 95% CI ($\text{cm}^3$) | 95% LoA ($\text{cm}^3$) | Paired $t$-test $p$ | Cases $>$ FP32 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `trt_fp32` | `0.965233` | `0.000000` | `0.000000` | `+0.0000 ± 0.0000` | `[+0.000, +0.000]` | `[+0.000, +0.000]` | `1.0000` | `0 / 9` |
| **`trt_fp16_unfolded` (Bug)** | `0.965722` | `0.000555` | `0.001386` | **`+1.6754 ± 0.5531`** | **`[+1.250, +2.101]`** | `[+0.591, +2.760]` | **`< 0.0001`** | **`9 / 9`** |
| **`trt_fp16` (Folded)** | `0.965226` | `0.000015` | `0.000033` | **`+0.0027 ± 0.0220`** | `[-0.014, +0.020]` | `[-0.040, +0.046]` | `0.7223` | `5 / 9` |
| **`trt_fp16_cuda` (Folded)** | `0.965229` | **`0.000014`** | **`0.000029`** | **`+0.0010 ± 0.0194`** | `[-0.014, +0.016]` | `[-0.037, +0.039]` | **`0.8847`** | `5 / 9` |
| **`trt_fp16_optimized`** | `0.964778` | `0.001669` | `0.005563` | `+0.1196 ± 1.1815` | `[-0.789, +1.028]` | `[-2.196, +2.435]` | `0.7692` | `5 / 9` |

---

## 4. Why This Work Is Important & Further Implications

1. **System-Level Bottleneck Attribution in 3D Medical AI:**
   Our ablation proves that accelerating 3D patch convolutions alone (`trt_fp16` on the host path) yields almost zero end-to-end benefit (`0.96–1.13×`) and even suffers from laptop GPU power-state downclocking (`2.34×` network latency inflation after a 3-second CPU idle gap). Real-time volumetric inference (`< 100 ms` per CT scan) requires co-designing zero-decompression I/O, register-fused GPU spatial resampling, and zero-copy TensorRT execution.
2. **Preventing Silent Quantization Bias in Clinical Biomarkers:**
   Unfused `ConvTranspose3d + BatchNorm3d` layers are ubiquitous across 3D U-Net, V-Net, DynUNet, and SegResNet architectures exported via PyTorch's ONNX exporter. Because the resulting `+1.68 cm³` volume shift coincidentally moved predictions closer to ground truth on this dataset (lowering MAE from `5.17 cm³` to `4.24 cm³` and slightly raising Dice from `0.9652` to `0.9657`), standard accuracy checks failed to flag it as a quantization bug. Pairing cross-backend Bland–Altman volumetry [7] against strict IEEE FP32 with analytical `float64` pre-export folding prevents silent biomarker drift in longitudinal organ volume tracking.
3. **VRAM Scalability for Multi-Organ and High-Resolution CT Workloads:**
   By replacing `float64` 3D coordinate meshgrids (`2.30 GB` VRAM) with register-only $3 \times 4$ affine transformations (`455 MB` VRAM), multiple 3D segmentation models can co-reside on edge and laptop GPUs alongside interactive radiological visualization workstations.

---

## 5. References & Attribution of All Public Data, Models, and Software

1. **3D Residual U-Net Backbone:** Kerfoot, E., Clough, J., Oksuz, I., Lee, J., King, A. P., & Schnabel, J. A. (2019). *Left-Ventricle Quantification Using Residual U-Net*. In Statistical Atlases and Computational Models of the Heart (STACOM 2018), Lecture Notes in Computer Science, vol. 11395, pp. 371–380. Springer, Cham. https://doi.org/10.1007/978-3-030-12029-0_40
2. **3D U-Net Architecture:** Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., & Ronneberger, O. (2016). *3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation*. In Medical Image Computing and Computer-Assisted Intervention (MICCAI 2016), LNCS vol. 9901, pp. 424–432. https://doi.org/10.1007/978-3-319-46723-8_49
3. **MONAI Framework:** Cardoso, M. J., Li, W., Brown, R., Ma, N., Kerfoot, E., Wang, Y., ... & Ourselin, S. (2022). *MONAI: An open-source framework for deep learning in healthcare*. arXiv preprint arXiv:2211.02701. https://arxiv.org/abs/2211.02701 (Apache-2.0 License).
4. **MONAI Model Zoo Spleen CT Segmentation Bundle (`v0.6.1`):** MONAI Consortium (2024). *MONAI Model Zoo: `spleen_ct_segmentation` Bundle v0.6.1*. https://github.com/Project-MONAI/model-zoo (Apache-2.0 License; reference citation: Xia, Y., et al., *3D Semi-Supervised Learning with Uncertainty-Aware Multi-View Co-Training*, arXiv:1811.12506, 2018).
5. **Medical Segmentation Decathlon (`Task09_Spleen`) Dataset:** Antonelli, M., Reinke, A., Bakas, S., Farahani, K., Kopp-Schneider, A., Landman, B. A., ... & Cardoso, M. J. (2022). *The Medical Segmentation Decathlon*. Nature Communications, 13(1), 4128. https://doi.org/10.1038/s41467-022-30695-9; Simpson, A. L., et al. (2019). *A large annotated medical image dataset for the development and evaluation of segmentation algorithms*. arXiv:1902.09063. Data hosted at http://medicaldecathlon.com/ under CC-BY-SA 4.0 License.
6. **PyTorch Deep Learning Framework:** Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. In Advances in Neural Information Processing Systems (NeurIPS 2019), vol. 32. https://pytorch.org/
7. **Bland–Altman MethodComparison Agreement:** Bland, J. M., & Altman, D. G. (1986). *Statistical methods for assessing agreement between two methods of clinical measurement*. The Lancet, 327(8476), 307–310. https://doi.org/10.1016/S0140-6736(86)90837-8
8. **Segmentation Overlap & Boundary Metrics:** Dice, L. R. (1945). *Measures of the amount of ecologic association between species*. Ecology, 26(3), 297–302; Taha, A. A., & Hanbury, A. (2015). *Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool*. BMC Medical Imaging, 15(1), 29. https://doi.org/10.1186/s12880-015-0068-x
9. **Amdahl's Law:** Amdahl, G. M. (1967). *Validity of the single processor approach to achieving large scale computing capabilities*. In Proceedings of the AFIPS Spring Joint Computer Conference, pp. 483–485. https://doi.org/10.1145/1465482.1465560
10. **NVIDIA CUDA, NVRTC, & TensorRT SDKs:** NVIDIA Corporation (2025). *NVIDIA TensorRT 10.16 Developer Guide & CUDA 12.8 Runtime/Driver API Reference*; Nickolls, J., Buck, I., Garland, M., & Skadron, K. (2008). *Scalable Parallel Programming with CUDA*. ACM Queue, 6(2), 40–53.
11. **ONNX & ONNX Runtime:** Bai, J., Lu, F., Zhang, K., et al. (2019). *ONNX: Open Neural Network Exchange*. https://onnx.ai/; ONNX Runtime Developers (2021). *ONNX Runtime: cross-platform, high performance ML inferencing and training accelerator*. https://onnxruntime.ai/
12. **NiBabel Neuroimaging I/O & Scientific Python Stack:** Brett, M., Markiewicz, C. J., Hanke, M., et al. (2024). *nipy/nibabel: 5.4.2*. Zenodo. https://doi.org/10.5281/zenodo.591597; Harris, C. R., et al. (2020). *Array programming with NumPy*. Nature, 585, 357–362; Virtanen, P., et al. (2020). *SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python*. Nature Methods, 17, 261–272; Hunter, J. D. (2007). *Matplotlib: A 2D Graphics Environment*. Computing in Science & Engineering, 9(3), 90–95.
