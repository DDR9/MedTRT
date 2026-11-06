# Phase 3 — GPU Profiling, NVTX Timeline Analysis & Bottleneck Decomposition

> **Scope & Hardware:** All measurements were recorded on an **NVIDIA GeForce RTX 5080 Laptop GPU** (`12.0 (sm_120)`, `15.92 GB` GDDR7, Driver `591.44`, CUDA `12.8`, TensorRT `10.16.1.11`). Stage boundaries are bracketed by synchronized `torch.cuda.Event(enable_timing=True)` pairs inside `torch.cuda.nvtx.range_push/pop` markers (`src/medtrt/timing.py`) and cross-checked against wall-clock `time.perf_counter()`.

---

## 1. Six-Stage NVTX Instrumentation Architecture

Every full-volume pass is decomposed into 6 mutually exclusive, hardware-timed NVTX ranges:

1. **`load`**: Disk I/O (`LoadImaged` gzip decompression on host path vs. `nib.load(..., mmap=True)` zero-copy header mapping on GPU/CUDA paths).
2. **`h2d`**: Host-to-Device PCIe memory transfer onto `cuda:0` (`pageable` vs. `pin_memory()` asynchronous DMA).
3. **`preprocess`**: Spatial reorientation to `RAS`, anisotropic resampling to `(1.5, 1.5, 2.0) mm`, and Hounsfield Unit windowing `[-57, 164] -> [0, 1]`.
4. **`sw_network`**: Pure 3D Residual UNet forward execution inside `SlidingWindowInferer`, isolated via `InstrumentedPredictor` CUDA event pairs around every patch batch `(B, 1, 96, 96, 96)`.
5. **`sw_aggregate`**: 3D Gaussian importance-map weighting and overlapping patch accumulation (`_sw_total - sw_network`).
6. **`postprocess`**: Softmax activation, inverse spatial resampling from `(1.5, 1.5, 2.0) mm` back to native patient CT geometry, argmax discretization (`uint8`), and D2H mask completion.

![Stage Latency Breakdown](../docs/figures/stage_breakdown.png)

---

## 2. Stage-by-Stage Latency & Fraction Decomposition (9 Held-Out CT Volumes)

| Profile | `load` ms (%) | `h2d` ms (%) | `preprocess` ms (%) | `sw_network` ms (%) | `sw_aggregate` ms (%) | `postprocess` ms (%) | E2E Median (ms) | Peak VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `pytorch_fp32_strict` | `414.92` (`13.4%`) | `4.03` (`0.1%`) | `268.64` (`8.7%`) | `375.62` (`12.2%`) | `8.04` (`0.3%`) | `2017.36` (`65.3%`) | **`3122.22`** | `490` |
| `pytorch_fp32` | `524.57` (`16.0%`) | `4.17` (`0.1%`) | `284.40` (`8.7%`) | `260.81` (`8.0%`) | `8.19` (`0.2%`) | `2195.26` (`67.0%`) | **`3341.11`** | `487` |
| `trt_fp32` | `382.10` (`13.4%`) | `3.77` (`0.1%`) | `263.51` (`9.2%`) | `241.46` (`8.5%`) | `9.08` (`0.3%`) | `1947.47` (`68.4%`) | **`2835.59`** | `368` |
| `trt_fp16` | `419.07` (`14.2%`) | `13.74` (`0.5%`) | `269.42` (`9.2%`) | `172.83` (`5.9%`) | `32.91` (`1.1%`) | `2035.26` (`69.2%`) | **`2967.01`** | `359` |
| `pytorch_tf32_gpu` | `1.74` (`0.5%`) | `35.13` (`9.4%`) | `52.77` (`14.2%`) | `207.00` (`55.6%`) | `6.65` (`1.8%`) | `68.73` (`18.5%`) | **`375.87`** | `2307` |
| `trt_fp16_gpu` | `1.60` (`0.8%`) | `29.03` (`13.6%`) | `46.50` (`21.8%`) | `48.56` (`22.8%`) | `6.41` (`3.0%`) | `81.20` (`38.1%`) | **`218.04`** | `2301` |
| `trt_fp16_cuda` | `1.57` (`1.8%`) | `25.54` (`29.4%`) | `1.43` (`1.6%`) | `49.56` (`57.1%`) | `6.01` (`6.9%`) | `2.64` (`3.0%`) | **`95.10`** | `455` |
| `trt_fp16_optimized` | `1.55` (`2.6%`) | `23.10` (`39.2%`) | `1.37` (`2.3%`) | `26.21` (`44.5%`) | `4.29` (`7.3%`) | `2.39` (`4.1%`) | **`60.83`** | `522` |

---

## 3. Amdahl's Law Bottleneck Shift Analysis

Amdahl's Law governs the maximum possible end-to-end speedup achievable from accelerating only the neural network (`sw_network`) when non-network stages occupy fraction $s = 1 - p$:

$$S_{\text{max}} = \frac{1}{s + \frac{p}{S_{\text{net}}}} \le \frac{1}{s}$$

| Execution Profile | Median `sw_network` (ms) | Median Non-Network (ms) | Median E2E (ms) | Network Share $p$ (%) | Non-Network Share $s$ (%) | Amdahl Ceiling $1/s$ (×) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `pytorch_fp32_strict` | `375.62` | `2746.59` | `3122.22` | `12.03%` | `87.97%` | **`1.14×`** |
| `pytorch_fp32` | `260.81` | `3080.30` | `3341.11` | `7.81%` | `92.19%` | **`1.08×`** |
| `trt_fp16` | `172.83` | `2794.18` | `2967.01` | `5.82%` | `94.18%` | **`1.06×`** |
| `pytorch_tf32_gpu` | `207.00` | `168.88` | `375.87` | `55.07%` | `44.93%` | **`2.23×`** |
| `trt_fp16_gpu` | `48.56` | `169.48` | `218.04` | `22.27%` | `77.73%` | **`1.29×`** |
| `trt_fp16_cuda` | `49.56` | `45.54` | `95.10` | `52.11%` | `47.89%` | **`2.09×`** |
| `trt_fp16_optimized` | `26.21` | `34.63` | `60.83` | `43.08%` | `56.92%` | **`1.76×`** |

### Key Systems Findings
1. **Why TensorRT Alone Yields Only `1.13×` End-to-End on the Host Pipeline:**
   In `pytorch_fp32`, the neural network (`sw_network = 260.8 ms`) accounts for only **`7.8%`** of total runtime (`3341.1 ms`), while CPU `Invertd` postprocessing (`2195.3 ms`) and CPU preprocessing (`284.4 ms`) consume **`92.2%`**. Even an infinitely fast neural network is capped by Amdahl's Law at **`1.08×`**.
2. **Why MONAI GPU Transforms (`trt_fp16_gpu`) Expose a Second Memory-Bandwidth Bottleneck:**
   Moving `Spacingd` and `Invertd` to PyTorch GPU `grid_sample` (`trt_fp16_gpu`) reduces median E2E latency to `218.04 ms`, but `postprocess` (`81.20 ms`) and `preprocess` (`46.50 ms`) still consume **`77.7%`** of the runtime and spike Peak VRAM to **`2301 MB`** (`2.20 GB`) because MONAI's `SpatialResample` constructs dense 64-bit/32-bit `(1, D, H, W, 3)` coordinate meshgrids in global memory.
3. **Eliminating the Non-Network Bottleneck via Custom CUDA C++ Kernels (`trt_fp16_cuda` & `trt_fp16_optimized`):**
   Replacing MONAI's meshgrid transforms with register-computed 3D affine CUDA C++ kernels (`cuda/medical_kernels.cu`) collapses `preprocess` to **`1.37 ms`** and `postprocess` to **`2.39 ms`**, reducing total median E2E latency to **`95.10 ms`** at bundle windows (`35.1×` speedup) and **`60.83 ms`** at Pareto windows (`54.9×` speedup vs. `pytorch_fp32`, `51.3×` vs. `pytorch_fp32_strict`).

---

## 4. Before vs. After Optimization Timeline (`spleen_6`, `512 × 512 × 135`)

![Stage Timeline Comparison](../docs/figures/nsys_trace.png)

| Stage | Before: `trt_fp16` Host Path (ms) | After: `trt_fp16_optimized` Custom CUDA (ms) | Stage Speedup (×) |
| :--- | :---: | :---: | :---: |
| `load` (NIfTI Disk Read) | `920.69` | `1.50` | `612.2×` |
| `h2d` (Host-to-Device Copy) | `28.73` | `54.19` | `0.53×` (deferred mmap read) |
| `preprocess` (RAS + Resample + HU) | `273.53` | `2.39` | **`114.5×`** |
| `sw_network` (TensorRT FP16 UNet) | `216.99` | `57.70` | **`3.76×`** |
| `sw_aggregate` (Gaussian Stitching) | `40.26` | `7.06` | **`5.70×`** |
| `postprocess` (Softmax + Invert + Argmax) | `4307.68` | `4.46` | **`966.1×`** |
| **Total End-to-End (`e2e_total`)** | **`5788.45`** | **`127.81`** | **`45.3×`** |
| **Peak Allocated VRAM** | `247.8 MB` | `451.3 MB` | Zero meshgrid overhead |

---

## 5. Dedicated Non-Default CUDA Stream & Pinned-Memory Findings

1. **Elimination of TensorRT 10.x Default-Stream Synchronization Overhead (`src/medtrt/trt_runner.py`):**
   Calling `context.execute_async_v3(stream_handle=0)` on the legacy default stream triggers TensorRT 10.x warning `[TRT] [W] Using default stream in enqueueV3() may lead to performance issues due to additional calls to cudaStreamSynchronize()`. Binding a dedicated non-default `torch.cuda.Stream` (`self.exec_stream`) synchronized with the caller's stream via non-blocking `torch.cuda.Event` dependencies (`_input_ready_event` and `_output_ready_event`) eliminates implicit host synchronization.
2. **Dynamic Clock-State Inflation on Laptop GPUs (`results/probes.json`):**
   On the host path, the GPU sits idle for `~3.5-5.0 s` during CPU `Spacingd` and `Invertd`, allowing the RTX 5080 Laptop GPU to drop from boost clocks (`~2250 MHz`, `P0`) to idle power states. Measuring identical `overlap=0.5, B=4` sliding-window inference immediately after a `3.0 s` idle gap vs. back-to-back execution reveals a **`2.34×`** latency inflation on TensorRT (`56.5 ms` hot vs. `132.0 ms` cold) and **`1.09×`** on PyTorch TF32 (`200.3 ms` vs. `217.3 ms`).
3. **Host-to-Device Pageable vs. Pinned Asynchronous DMA (`results/benchmark_preprocessing.json`):**
   Transferring the `512 × 512 × 135` (`141.56 MB`) volume from pageable host memory takes `11.02 ms`, whereas pre-pinned host memory (`pin_memory()`) with `non_blocking=True` transfers in **`5.87 ms`** (**`1.88×` faster**, achieving `~24.8 GB/s` over PCIe Gen4/Gen5).
