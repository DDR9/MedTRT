# Phase 5 — TensorRT 10.x Compilation, Dynamic Optimization Profiles & Numerical Precision Study

> **Implementation Files:**
> - Model Loader & `ConvTranspose3d+BatchNorm3d` FP64 Folder: [`src/medtrt/model.py`](../src/medtrt/model.py)
> - ONNX Opset 17 Exporter & TensorRT 10.16 Engine Builder: [`src/medtrt/export.py`](../src/medtrt/export.py)
> - Zero-Copy Dedicated-Stream TensorRT Runner: [`src/medtrt/trt_runner.py`](../src/medtrt/trt_runner.py)
> - Build Metadata: [`results/trt_build.json`](../results/trt_build.json), [`results/benchmark_tensorrt_variants.json`](../results/benchmark_tensorrt_variants.json)

---

## 1. Root-Cause Discovery: Why Unfolded TensorRT FP16 Exhibited `+1.68 cm³` Systematic Volume Inflation

In the initial baseline repository, TensorRT FP16 exhibited a statistically significant **`+1.675 ± 0.553 cm³` (`p = 0.00002`, 9/9 cases positive)** volume inflation relative to PyTorch FP32. Layer-by-layer numerical inspection (`src/medtrt/probes.py`) revealed the exact mathematical mechanism:

1. **Asymmetric ONNX Constant Folding in PyTorch's Exporter:**
   When exporting `monai.networks.nets.UNet` to ONNX (Opset 17), PyTorch's constant folder fuses `Conv3d + BatchNorm3d` in the encoder path, but **leaves `ConvTranspose3d + BatchNorm3d` unfused** as separate operators in the decoder path.
2. **Catastrophic FP16 Cancellation in Decoder Layer `model.2.0.adn.N`:**
   In `model.2.0.adn.N`, the trained `BatchNorm3d` parameters produce an affine shift of:
   $$\text{shift} = \beta - \frac{\gamma \cdot \mu_{\text{running}}}{\sqrt{\sigma^2_{\text{running}} + \epsilon}} = -81.79$$
   For background voxels where `ConvTranspose3d` outputs values near `+57.62`, the true post-BN activation is near `0.0`. However, in IEEE-754 FP16 (10-bit mantissa), numbers in the interval $[64, 128)$ have a unit-in-last-place spacing of $\text{ULP} = 2^6 \times 2^{-10} = 0.0625$. Rounding the intermediate `ConvTranspose3d` output and shift to FP16 before subtraction introduces a systematic quantization error of up to `0.03125` per decoder stage, shifting background logit Channel 0 by `-0.159` (`+0.130` foreground margin shift) and flipping boundary voxels from background to spleen!
3. **Mathematical Fix — Pre-Export FP64 `ConvTranspose3d + BatchNorm3d` Folding (`fold_conv_transpose_batchnorm`):**
   We implemented exact analytical folding in **IEEE-754 FP64 (`torch.float64`)** prior to FP32/ONNX conversion (`src/medtrt/model.py`):
   $$W_{\text{folded}} = W \cdot \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}}, \qquad b_{\text{folded}} = (b - \mu) \cdot \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} + \beta$$
   - In FP32, this fold is algebraically exact (`max_abs_logit_diff = 1.34e-05`).
   - In TensorRT FP16, eliminating the large intermediate `ConvTranspose3d` offset reduces mean absolute logit error by **`35.6×`** (`0.1377 -> 0.00387`), reduces signed channel bias by **`62.3×`** (`-0.159 -> -0.00255`), and collapses full-volume cross-backend volume bias by **`620×`** (`+1.6754 cm³ -> +0.0027 cm³`, `p = 0.722`, 5/9 cases positive — statistically indistinguishable from zero!).

| TensorRT FP16 Engine Variant | Held-Out Mean Dice | Mean `|ΔDice|` vs. Strict FP32 | Cross-Backend Volume Bias vs. Strict FP32 | Cases Larger Than FP32 | Paired t-test p-value |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Unfolded Legacy FP16 (`spleen_unet_fp16_unfolded.engine`) | `0.965706` | `0.000557` | **`+1.6716 ± 0.5558 cm³`** | `9/9` (Systematic bias) | `0.00002` (Biased) |
| **BN-Folded FP16 (`spleen_unet_fp16.engine`)** | **`0.965229`** | **`0.000014`** | **`+0.0010 ± 0.0194 cm³`** | `5/9` (Random sign) | **`0.72233` (Unbiased)** |

---

## 2. TensorRT 10.16 Engine Compilation & Dynamic Batch Optimization Profile

Both `spleen_unet_fp32.engine` (`22.97 MB`) and `spleen_unet_fp16.engine` (`13.70 MB`) are compiled from ONNX Opset 17 (`max_abs_diff_vs_pytorch_fp32 = 1.91e-6`) with a dynamic batch optimization profile:
- **`min_shape`**: `(1, 1, 96, 96, 96)`
- **`opt_shape`**: `(4, 1, 96, 96, 96)`
- **`max_shape`**: `(16, 1, 96, 96, 96)`
- **Workspace Pool Limit**: `4.0 GB` (`trt.MemoryPoolType.WORKSPACE`)

### Isolated Patch Forward Latency & Throughput Across Batch Sizes (`96 × 96 × 96`)

| Backend / Engine Variant | Engine File | Device Context | Batch B=1 | Batch B=2 | Batch B=4 | Batch B=8 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `trt_fp32` (`spleen_unet_fp32.engine`) | `22.97 MB` | `0.00 MB` | `3.12 ms` (`320.5 p/s`) | `5.51 ms` (`363.2 p/s`) | `11.74 ms` (`340.6 p/s`) | `23.19 ms` (`344.9 p/s`) |
| `trt_fp16_unfolded` (`spleen_unet_fp16_unfolded.engine`) | `13.75 MB` | `0.00 MB` | `1.10 ms` (`911.2 p/s`) | `1.62 ms` (`1235.9 p/s`) | `3.53 ms` (`1134.5 p/s`) | `7.01 ms` (`1141.8 p/s`) |
| `trt_fp16_dynamic_alloc` (`spleen_unet_fp16.engine`) | `13.70 MB` | `0.00 MB` | `1.01 ms` (`985.9 p/s`) | `1.59 ms` (`1260.3 p/s`) | `3.27 ms` (`1225.0 p/s`) | `9.58 ms` (`835.1 p/s`) |
| `trt_fp16_buffer_reuse` (`spleen_unet_fp16.engine`) | `13.70 MB` | `0.00 MB` | `0.98 ms` (`1025.4 p/s`) | `1.62 ms` (`1234.6 p/s`) | `3.29 ms` (`1216.7 p/s`) | `7.51 ms` (`1065.4 p/s`) |

### Comparison Against PyTorch Precision Modes (`Batch B=4`, `4 × 1 × 96 × 96 × 96`)

| Backend & Precision Mode | Batch B=1 Median (ms) | Batch B=4 Median (ms) | Batch B=8 Median (ms) | B=4 Throughput (patches/s) | Speedup vs. PyTorch Strict FP32 (B=4) | Speedup vs. PyTorch Default TF32 (B=4) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| PyTorch Strict IEEE FP32 (`allow_tf32=False`) | `6.96` | `22.36` | `41.60` | `178.9` | `1.00×` | `0.71×` |
| PyTorch Default TF32 (`allow_tf32=True`) | `3.90` | `15.95` | `25.25` | `250.8` | `1.40×` | `1.00×` |
| PyTorch AMP FP16 (`torch.autocast`) | `5.40` | `15.69` | `23.10` | `254.9` | `1.42×` | `1.02×` |
| PyTorch Pure FP16 (`model.half()`) | `4.48` | `14.89` | `22.08` | `268.6` | `1.50×` | `1.07×` |
| TensorRT 10.16 FP32 (`spleen_unet_fp32.engine`) | `3.24` | `13.36` | `29.58` | `299.3` | `1.67×` | `1.19×` |
| **TensorRT 10.16 FP16 (`spleen_unet_fp16.engine`)** | **`1.08`** | **`3.79`** | **`8.89`** | **`1054.3`** | **`5.89×`** | **`4.20×`** |

---

## 3. Patch-Level Numerical Equivalence vs. Strict IEEE FP32 (`spleen_6` Real CT Patch)

| Candidate Backend | Real CT Max `|ΔLogit|` | Real CT Mean `|ΔLogit|` | Ch1 Signed Bias | Softmax Prob MAE | Cosine Similarity | Voxel Argmax Match (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| PyTorch Strict FP32 (`fold_bn=True`) | `0.00001` | `0.000002` | `+0.000000` | `0.000000` | `0.9999999` | `100.0000%` |
| PyTorch TF32 (`allow_tf32=True`) | `0.00549` | `0.000660` | `+0.000360` | `0.000000` | `1.0000000` | `100.0000%` |
| TensorRT 10.16 FP32 (`spleen_unet_fp32.engine`) | `0.00002` | `0.000002` | `-0.000000` | `0.000000` | `1.0000000` | `100.0000%` |
| PyTorch AMP FP16 (`fold_bn=False`, Unfolded) | `0.18124` | `0.076429` | `+0.065562` | `0.000006` | `0.9999957` | `99.9999%` |
| PyTorch AMP FP16 (`fold_bn=True`, BN-Folded) | `0.02166` | `0.003429` | `+0.002917` | `0.000000` | `0.9999998` | `100.0000%` |
| PyTorch Pure FP16 (`model.half()`, BN-Folded) | `0.02451` | `0.003730` | `+0.004427` | `0.000000` | `0.9999998` | `100.0000%` |
| TensorRT 10.16 FP16 (`fold_bn=False`, Unfolded Legacy) | `0.22288` | `0.134978` | `+0.116259` | `0.000011` | `0.9999926` | `99.9998%` |
| **TensorRT 10.16 FP16 (`fold_bn=True`, BN-Folded)** | `0.03749` | `0.004261` | `+0.002611` | `0.000000` | `0.9999998` | `100.0000%` |

Across all `169` sliding-window batches of all 9 held-out CT volumes (`results/probes.json`), BN-folded TensorRT FP16 achieves a mean absolute logit difference of **`0.00398`** (on logits up to `23.0`), a mean probability MAE of **`0.000002`**, and a near-zero foreground margin shift of **`+0.00361`** (`478` window voxels flipped BG→FG vs. `418` flipped FG→BG).
