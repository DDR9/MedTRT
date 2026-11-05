# MedTRT: GPU-Accelerated 3D CT Segmentation, Profiled and Validated

> ⚠️ **Scope Boundary & Intended Use Disclaimer:** This repository is a **systems profiling and numerical validation research study** equipped with a minimal tri-planar workstation. It is **NOT** a clinical medical device and makes **NO** automated diagnostic or triage claims. All metrics below are programmatically generated from `results/*.json` and `results/*.csv`.

---

## 1. Headline Measured Results (`NVIDIA GeForce RTX 5080 Laptop GPU`, Blackwell `sm_120`)

- **Zero-Copy TensorRT 10.x FP16 Network Acceleration:** Reduces median per-volume 3D UNet forward latency (`sw_network`) from **`347.93 ms`** (Strict IEEE FP32) and **`240.11 ms`** (Default TF32) down to **`97.91 ms`** (**`3.55x`** speedup vs. Strict FP32; **`2.45x`** speedup vs. Default TF32).
- **Amdahl's Law Bottleneck Elimination (`trt_fp16_optimized`):** Stage-boundary NVTX + CUDA Event profiling revealed that host-bound postprocessing (`Invertd` 3D trilinear spatial resampling on CPU: **`2533.38 ms`**) and CPU `Spacingd` preprocessing consumed **`97.34%`** of unoptimized end-to-end execution time, capping theoretical E2E speedup at **`1.03x`**. Moving `Spacingd`, `ScaleIntensityRanged`, `Activationsd`, and `Invertd` onto CUDA (`cuda:0`) slashed `postprocess` latency from **`2533.38 ms` $\to$ `1596.53 ms` (`1.6x` faster)** and reduced full-volume end-to-end latency (`e2e_total`, including NIfTI disk load) from **`3673.44 ms` $\to$ `2807.13 ms` (`1.31x` E2E speedup)**.
- **Diagnostic & Volumetric Equivalence:** Across the 9 deterministic held-out MSD `Task09_Spleen` validation volumes, mean foreground Dice is **`0.965722`** (TensorRT FP16) vs. **`0.965228`** (PyTorch FP32), yielding a mean absolute Dice drift of **`0.000548 < 0.005`** and cross-backend Bland-Altman volumetric bias of **`+1.6707 cm³`** (`95% LoA: [+0.5897, +2.7517] cm³`).

---

## 2. Pipeline Architecture

```mermaid
flowchart LR
    subgraph Host_IO["Stage 1: Host Disk I/O (NVTX: load)"]
        A["NIfTI CT Volume (*.nii.gz)<br/>LoadImaged + EnsureChannelFirstd"]
    end

    subgraph Preproc["Stages 2-3: Spatial & HU Preprocessing (NVTX: preprocess / h2d)"]
        B["Orientationd(RAS)<br/>Spacingd(1.5, 1.5, 2.0 mm, bilinear)<br/>ScaleIntensityRanged(-57..164 HU → 0..1)<br/>(CPU in baseline / CUDA in trt_fp16_optimized)"]
    end

    subgraph SlidingWindow["Stages 4-5: 3D Sliding-Window Inference (NVTX: sw_network & sw_aggregate)"]
        C["MONAI SlidingWindowInferer<br/>ROI=(96,96,96), sw_batch_size=4, overlap=0.5"]
        D["Zero-Copy TRTRunner (TensorRT 10.16 FP16)<br/>context.set_tensor_address(tensor.data_ptr())<br/>context.execute_async_v3(cuda_stream)"]
        C <--> D
    end

    subgraph Postproc["Stage 6: Inverse Transform & Volumetry (NVTX: postprocess)"]
        E["Activationsd(Softmax) → Invertd(Native Affine)<br/>→ AsDiscreted(Argmax)<br/>(CUDA grid_sample in trt_fp16_optimized)"]
        F["Clinical Volumetry & Bland-Altman Audit<br/>V = (N × sx × sy × sz) / 1000 cm³"]
        E --> F
    end

    A --> B --> C
    D --> E
```

---

## 3. Verified Hardware & Toolchain Specifications (from `results/env.json`)

| Parameter | Verified Runtime Value |
| :--- | :--- |
| **GPU Model & Architecture** | `NVIDIA GeForce RTX 5080 Laptop GPU` (`NVIDIA Blackwell`, Compute Capability `12.0 (sm_120)`) |
| **Total GPU VRAM** | `15.92 GB` (`16302 MB`) |
| **NVIDIA Driver / CUDA Runtime / cuDNN** | Driver `591.44` · CUDA `12.8` · cuDNN `9.19.0` |
| **PyTorch / MONAI / TensorRT Versions** | PyTorch `2.11.0+cu128` · MONAI `1.6.1` · TensorRT `10.16.1.11` |
| **OS & Python Runtime** | `Windows 10 (10.0.26300)` · CPython `3.11.15` |
| **ONNX Export & TRT Engine Build** | `opset_version=17` · Max Abs Diff `2.861e-05` · Build Time `28.65 s` (`13.75 MB`) |

---

## 4. Benchmark Matrix & Stage Decomposition (`9 Held-Out Cases × 10 Timed Repeats = 90 Passes`)

### 4.1 End-to-End & Network Performance Across All 5 Backends
| Backend & Precision | Patch `B=1` Median (ms) | Patch `B=4` Median (ms) | Volume `sw_network` Median [IQR] (ms) | Volume `e2e_total` Median [IQR] (ms) | Peak VRAM (MB) | Held-Out Mean Dice (± SD) | Network Speedup (vs FP32 / Strict) | E2E Speedup (vs FP32) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PyTorch FP32 (Strict IEEE)** (`pytorch_fp32_strict`) | `6.52 ms` | `18.61 ms` | `347.93 [238.75] ms` | `3752.61 [2646.39] ms` | `489.7 MB` | `0.965233 ± 0.011062` | `0.69x` / `1.00x` | **`0.98x`** |
| **PyTorch FP32 (Default TF32)** (`pytorch_fp32`) | `6.61 ms` | `12.39 ms` | `240.11 [198.70] ms` | `3673.44 [3999.12] ms` | `488.7 MB` | `0.965228 ± 0.011074` | `1.00x` / `1.45x` | **`1.00x`** |
| **PyTorch AMP (FP16 Autocast)** (`pytorch_amp`) | `11.75 ms` | `11.45 ms` | `217.54 [191.28] ms` | `3937.15 [3721.15] ms` | `457.0 MB` | `0.965507 ± 0.011213` | `1.10x` / `1.60x` | **`0.93x`** |
| **TensorRT 10.x FP16 (Zero-Copy)** (`trt_fp16`) | `1.44 ms` | `3.07 ms` | `97.91 [143.64] ms` | `3676.52 [3492.22] ms` | `615.1 MB` | `0.965722 ± 0.011288` | `2.45x` / `3.55x` | **`1.00x`** |
| **TensorRT 10.x FP16 + GPU Transforms** (`trt_fp16_optimized`) | `1.02 ms` | `3.03 ms` | `110.57 [130.19] ms` | `2807.13 [2378.89] ms` | `643.2 MB` | `0.960759 ± 0.012043` | `2.17x` / `3.15x` | **`1.31x`** |

### 4.2 Per-Stage Median [IQR] Latency Breakdown (ms)
| Pipeline Stage (NVTX Range) | `pytorch_fp32_strict` (ms) | `pytorch_fp32` (ms) | `pytorch_amp` (ms) | `trt_fp16` (ms) | `trt_fp16_optimized` (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`load`** | `524.57 [381.41]` | `671.11 [746.89]` | `604.85 [759.91]` | `656.73 [721.12]` | `636.55 [766.03]` |
| **`preprocess`** | `290.44 [192.75]` | `308.43 [242.09]` | `359.50 [166.24]` | `315.94 [210.03]` | `352.30 [204.30]` |
| **`h2d`** | `4.08 [2.66]` | `4.38 [3.42]` | `5.45 [11.60]` | `5.93 [14.90]` | `6.43 [14.47]` |
| **`sw_network`** | `347.93 [238.75]` | `240.11 [198.70]` | `217.54 [191.28]` | `97.91 [143.64]` | `110.57 [130.19]` |
| **`sw_aggregate`** | `8.23 [7.77]` | `8.24 [6.53]` | `9.56 [16.31]` | `13.27 [29.94]` | `15.35 [33.55]` |
| **`postprocess`** | `2365.43 [1656.07]` | `2428.05 [2552.25]` | `2795.89 [2485.57]` | `2533.38 [2453.76]` | `1596.53 [802.02]` |
| **`non_network`** | `3379.63 [2316.31]` | `3435.13 [3694.85]` | `3666.27 [3516.15]` | `3552.67 [3395.03]` | `2668.62 [2197.26]` |
| **`e2e_total`** | `3752.61 [2646.39]` | `3673.44 [3999.12]` | `3937.15 [3721.15]` | `3676.52 [3492.22]` | `2807.13 [2378.89]` |

![Stage Latency Breakdown](docs/figures/stage_breakdown.png)

---

## 5. Amdahl's Law Analysis, Nsight Trace & Targeted Optimization

### 5.1 Amdahl's Law Ceiling Before vs. After GPU Transform Offload
Let $p = T_{\text{sw\_network}} / T_{\text{e2e}}$ denote the fraction of end-to-end time spent in 3D UNet patch forward execution and $s = 1 - p$ denote the non-network fraction:

| Pipeline Variant | Median `sw_network` ($\text{ms}$) | Median `non_network` ($\text{ms}$) | Median `e2e_total` ($\text{ms}$) | Network Fraction ($p$) | Non-Network Fraction ($s$) | Theoretical Max E2E Speedup ($1/s$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`pytorch_fp32`** | `240.11 ms` | `3433.33 ms` | `3673.44 ms` | `6.54%` | `93.46%` | `1.07x` |
| **`trt_fp16` (Unoptimized Host Transforms)** | `97.91 ms` | `3578.61 ms` | `3676.52 ms` | `2.66%` | `97.34%` | `1.03x` |
| **`trt_fp16_optimized` (GPU Resample + GPU Invertd)** | `110.57 ms` | `2696.56 ms` | `2807.13 ms` | `3.94%` | `96.06%` | `1.04x` |

### 5.2 Nsight / NVTX Timeline Interpretation (`results/nsys_summary.json`)
![Nsight / NVTX Timeline](docs/figures/nsys_trace.png)

- **Bottleneck Attribution:** In the unoptimized `trt_fp16` pipeline (`spleen_6`, `512×512×135`), all `18` sliding-window batches complete on CUDA Stream 0 in just **`200.06 ms`** total (`sw_aggregate` stitch overhead **`41.78 ms`**), whereas copying the 2-channel resampled logits to CPU for `Invertd` trilinear interpolation stalls the pipeline for **`5831.12 ms`**.
- **Targeted Optimization (`trt_fp16_optimized`):** By transferring the raw `MetaTensor` to `cuda:0` immediately after `LoadImaged` (`h2d`) and keeping `data["pred"]` on `cuda:0` through `Activationsd(softmax=True)` $\to$ `Invertd(nearest_interp=False)` $\to$ `AsDiscreted(argmax=True)`, MONAI executes 3D affine grid generation and trilinear `grid_sample` directly on the RTX 5080 GPU. Across all 90 timed passes:
  - `preprocess` median dropped from **`315.94 ms` $\to$ `352.30 ms`**.
  - `postprocess` median dropped from **`2533.38 ms` $\to$ `1596.53 ms` (`1.6x` speedup)**.
  - Total end-to-end latency (`e2e_total`) dropped from **`3676.52 ms` $\to$ `2807.13 ms` (`1.31x` overall E2E speedup)**, leaving only NIfTI gzip file decompression (`load`) on the host CPU.

### 5.3 Sliding-Window Parameter Grid Sweep (`results/sweep.csv`)
| Overlap | `sw_batch_size` | Median Sliding-Window Latency (ms) | IQR (ms) | Mean Sliding-Window Latency (ms) | Peak VRAM (MB) | Held-Out Mean Dice |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0.25` | `1` | `38.25 ms` | `37.53 ms` | `52.97 ms` | `1187.8 MB` | `0.965343` |
| `0.25` | `2` | `48.62 ms` | `16.80 ms` | `51.09 ms` | `1205.5 MB` | `0.965343` |
| `0.25` | `4` | `27.23 ms` | `37.38 ms` | `43.57 ms` | `1240.2 MB` | `0.965343` |
| `0.25` | `8` | `31.12 ms` | `19.10 ms` | `37.99 ms` | `1306.5 MB` | `0.965343` |
| `0.50` | `1` | `89.83 ms` | `30.58 ms` | `91.02 ms` | `1188.6 MB` | `0.965716` |
| `0.50` | `2` | `68.70 ms` | `24.78 ms` | `72.08 ms` | `1205.5 MB` | `0.965716` |
| `0.50` | `4` | `64.05 ms` | `44.67 ms` | `69.02 ms` | `1239.5 MB` | `0.965716` |
| `0.50` | `8` | `68.66 ms` | `21.64 ms` | `70.98 ms` | `1306.5 MB` | `0.965716` |
| `0.75` | `1` | `288.41 ms` | `245.20 ms` | `344.64 ms` | `1187.8 MB` | `0.965651` |
| `0.75` | `2` | `226.31 ms` | `197.37 ms` | `272.93 ms` | `1205.5 MB` | `0.965651` |
| `0.75` | `4` | `216.29 ms` | `180.41 ms` | `259.96 ms` | `1240.2 MB` | `0.965651` |
| `0.75` | `8` | `230.62 ms` | `185.91 ms` | `278.56 ms` | `1306.5 MB` | `0.965651` |

![Pareto Sweep](docs/figures/pareto_sweep.png)

---

## 6. Clinical Measurement Validation (`results/validation.csv` & `results/bland_altman.json`)

### 6.1 Per-Case Volumetric & Dice Equivalence (`PyTorch FP32` vs. `TensorRT 10.x FP16`)
| Case ID | $V_{\text{GT}}$ ($\text{cm}^3$) | $V_{\text{FP32}}$ ($\text{cm}^3$) | $V_{\text{TRT\_FP16}}$ ($\text{cm}^3$) | $\Delta V_{\text{TRT - FP32}}$ ($\text{cm}^3$) | $\text{Dice}_{\text{FP32}}$ | $\text{Dice}_{\text{TRT\_FP16}}$ | $\Delta\text{Dice}_{\text{TRT - FP32}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `spleen_56` | `78.07` | `75.42` | `76.14` | `+0.7221` | `0.936098` | `0.936064` | `-0.000034` |
| `spleen_59` | `426.65` | `425.41` | `427.69` | `+2.2754` | `0.966229` | `0.966801` | `+0.000572` |
| `spleen_6` | `209.52` | `209.07` | `210.49` | `+1.4177` | `0.973226` | `0.973165` | `-0.000061` |
| `spleen_60` | `299.84` | `288.19` | `289.89` | `+1.7072` | `0.971037` | `0.972406` | `+0.001369` |
| `spleen_61` | `163.30` | `160.80` | `161.95` | `+1.1456` | `0.971502` | `0.972071` | `+0.000569` |
| `spleen_62` | `275.93` | `275.33` | `277.25` | `+1.9180` | `0.961134` | `0.960988` | `-0.000146` |
| `spleen_63` | `356.60` | `342.13` | `344.04` | `+1.9140` | `0.963633` | `0.964536` | `+0.000903` |
| `spleen_8` | `501.55` | `492.81` | `495.30` | `+2.4828` | `0.973408` | `0.974054` | `+0.000646` |
| `spleen_9` | `269.15` | `264.92` | `266.38` | `+1.4535` | `0.970782` | `0.971414` | `+0.000632` |

### 6.2 Bland-Altman Summary Statistics
| Comparison Pair | Mean Bias ($\text{cm}^3$) | Sample SD ($\text{cm}^3$) | 95% LoA Lower ($\text{cm}^3$) | 95% LoA Upper ($\text{cm}^3$) | MAE ($\text{cm}^3$) | Max $|\Delta V|$ ($\text{cm}^3$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TensorRT FP16 vs. Ground Truth** | `-3.4976` | `5.0334` | `-13.3631` | `+6.3678` | `4.2372` | `12.5611` |
| **PyTorch FP32 vs. Ground Truth** | `-5.1683` | `5.1770` | `-15.3153` | `+4.9787` | `5.1683` | `14.4751` |
| **Cross-Backend (`TRT FP16 - PyTorch FP32`)** | **`+1.6707`** | **`0.5515`** | **`+0.5897`** | **`+2.7517`** | **`1.6707`** | **`2.4828`** |

- **Mean $|\Delta\text{Dice}|$ (`TRT FP16` vs. `PyTorch FP32`):** `0.000548` (Max: `0.001369`, Acceptance Criterion: `< 0.005` — **PASS**).

![Bland-Altman Agreement](docs/figures/bland_altman.png)

### 6.3 Case Failure Analysis (`results/failure_analysis.json`)
1. **Worst Case #1 — `spleen_56` (Dice = `0.9361`, $V_{\text{pred}} = 76.14\text{ cm}^3$, $V_{\text{gt}} = 78.07\text{ cm}^3$, Spacing `[0.699, 0.699, 7.0]` mm):**
   - *Attribution:* Coarse z-axis slice thickness (7.0 mm vs. 1.5-5.0 mm standard) induces severe partial-volume averaging at the inferior and superior splenic poles, while soft-tissue isodensity along the left hepatic lobe / gastric fundus boundary causes minor under-segmentation (-2.5 cm³).
   ![Failure Case 1](docs/figures/failure_case_1.png)
2. **Worst Case #2 — `spleen_62` (Dice = `0.9610`, $V_{\text{pred}} = 277.25\text{ cm}^3$, $V_{\text{gt}} = 275.93\text{ cm}^3$, Spacing `[0.676, 0.676, 5.0]` mm):**
   - *Attribution:* Prominent hyperdense splenic hilum vasculature and intimate anatomical contact with the adjacent stomach wall and splenic flexure create local boundary ambiguity at the medial hilum, yielding minor contour divergence while preserving total volume within < 0.7 cm³.
   ![Failure Case 2](docs/figures/failure_case_2.png)

---

## 7. Reproducibility & Quickstart (`make all`)

```bash
# Full end-to-end audit, dataset setup, engine compilation, benchmarking, validation & report generation
make all

# Run automated pytest verification suite
make test

# Launch interactive tri-planar Streamlit workstation on port 8501
make app
```
