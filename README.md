# MedTRT: GPU-Accelerated 3D CT Segmentation, Profiled and Validated

> **Research benchmark only.** This is a systems study of one public model on one laptop. It is not a medical device, and it makes no diagnostic, triage, or clinical performance claims.
>
> Every number below is generated from `results/`. To check them without a GPU, run `python -m medtrt.verify_claims`.

## 1. Results

A pretrained MONAI 3D UNet segments the spleen in abdominal CT. All timings are on an `NVIDIA GeForce RTX 5080 Laptop GPU`.

| Measure | Fast profile (`trt_fp16_optimized`) | Notes |
| :--- | :--- | :--- |
| End-to-end latency | `68.9 ms` median | Per-case medians `36`-`108 ms`; slowest of 90 passes `123 ms` |
| Speedup vs host-path PyTorch | `42.9x` | `2952 ms` baseline with CPU inverse transform and gzip reads |
| Speedup vs GPU-path PyTorch | `5.20x` | `358 ms` with the same GPU pipeline and bundle windows, no TensorRT |
| TensorRT alone, same pipeline and windows | `3.72x` end to end | `358` to `96 ms` |
| Mean Dice | `0.964778` | True FP32 reference `0.965233` |
| Mean / max per-case Dice change vs FP32 | `0.001669` / `0.005563` | `1` of `9` cases at or over `0.005` |
| Volume bias vs FP32 | `+0.12 cm³` (`+0.04%`) | 95% CI `[-0.79, +1.03]`; `5` of `9` cases positive |
| Peak VRAM | `0.51 GB` | Largest case |

### Targets

| Target | Measured | Status |
| :--- | :---: | :---: |
| Every pass under 500 ms | Slowest pass `122.6 ms` | PASS |
| Mean Dice at least 0.960 | `0.964778` | PASS |
| Mean absolute Dice change under 0.005 | `0.001669` | PASS |
| Like-for-like (`trt_fp16_cuda`) every case Dice change under 0.005 | Max `0.000029` (`trt_fp16_optimized` max `0.005563`) | PASS |

## 2. Where the speedup comes from

Most of the end-to-end gain is not TensorRT. Each row below changes one thing, using the same 9 cases.

| Configuration | E2E median (ms) | Slowest pass (ms) | vs first row | `sw_network` (ms) | Mean Dice |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Host path, PyTorch TF32 | `2992.5` | `6339.3` | `1.00x` | `239.3` | `0.965228` |
| Host path, TensorRT only | `2923.0` | `6015.0` | `1.02x` | `170.3` | `0.965226` |
| GPU transforms, gzip read, PyTorch TF32 | `706.9` | `1404.3` | `4.23x` | `201.1` | `0.965228` |
| GPU transforms, gzip read, TensorRT | `564.3` | `1252.9` | `5.30x` | `47.2` | `0.965228` |
| GPU transforms, .nii read, PyTorch TF32 | `363.3` | `686.1` | `8.24x` | `195.9` | `0.965228` |
| GPU transforms, .nii read, TensorRT | `274.8` | `472.5` | `10.89x` | `54.0` | `0.965228` |
| Fast windows, PyTorch TF32 | `320.6` | `540.4` | `9.33x` | `111.9` | `0.964771` |
| Fast windows, TensorRT | `218.5` | `374.2` | `13.70x` | `23.5` | `0.964767` |
| Fast windows, TensorRT, nearest-label inverse | `219.0` | `361.6` | `13.66x` | `24.1` | `0.960230` |

- **Keeping the inverse transform on the GPU is the largest single change.** The reference `inference.json` already keeps predictions on the GPU, so the host path overstates the cost of a default deployment. With gzip reads, this alone moves PyTorch from `2992` to `707 ms`.
- **Removing gzip from the critical path** moves PyTorch to `363 ms`. It assumes volumes were decompressed ahead of time.
- **TensorRT** then takes the same pipeline from `363` to `275 ms`. PyTorch without TensorRT already meets the 500 ms target.
- **Inverting class probabilities rather than hard labels matters for accuracy.** Nearest-label inversion scores `0.960230` Dice at `219 ms`. Trilinear scores `0.964767` at `218 ms`.

## 3. Method

| Component | Value |
| :--- | :--- |
| GPU | `NVIDIA GeForce RTX 5080 Laptop GPU`, compute capability `12.0 (sm_120)`, `15.92 GB` |
| Driver | `591.44`, `WDDM` mode, CUDA `12.8`, cuDNN `9.19.0` |
| Software | PyTorch `2.11.0+cu128`, MONAI `1.6.1`, TensorRT `10.16.1.11`, Python `3.11.15` |
| OS | `Windows 11 (10.0.26300)` |
| ONNX export | Opset `17`, max abs diff vs FP32 `2.670e-05` on one synthetic batch |
| TensorRT engine | FP16, dynamic batch 1 to 8, optimized for 2, `13.70 MB` |

- **Data:** Medical Segmentation Decathlon `Task09_Spleen` and the MONAI `spleen_ct_segmentation` bundle v`0.6.1`, with weights verified by SHA-256.
- **Evaluation cases:** the `9` of `41` labeled cases that the bundle reserves for validation. They were never trained on, but the bundle selected its checkpoint by Dice on these same cases, and MSD test labels are not public. There is no untouched test set here, so treat Dice as an optimistic estimate, suitable for comparing backends.
- **Timing:** 3 warm-up volumes, then 10 passes on each of 9 cases, for 90 timed passes per profile. Stages use CUDA events. Each pass is also timed by wall clock, and the two agree (see `wall_ms`). Every pass ends with the label map in host memory.
- **Profiles:** the host path reads gzip and transforms on the CPU. The GPU path reads pre-decompressed `.nii` and transforms on the GPU. `pytorch_fp32_strict` is true FP32 on the unmodified bundle weights and is the numerical reference. All other profiles use the BatchNorm-folded weights. `pytorch_tf32` is PyTorch's default and is not FP32. `trt_fp16_optimized` is the fast profile. `trt_fp16_nn` is kept only as a comparison.
- **Profiling:** Nsight Systems was not run for these results. NVTX ranges are in the code, and `--nsys` will use them when Nsight is installed, but every number here comes from CUDA events and wall-clock timers.

## 4. Benchmark

| Profile | Overlap / batch | Patch B=1 (ms) | Patch B=4 (ms) | `sw_network` (ms) | E2E median [IQR] (ms) | Slowest (ms) | Peak VRAM (MB) | Mean Dice ± SD | vs `pytorch_fp32` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `pytorch_fp32_strict` | `0.5` / `4` | `6.68` | `19.49` | `379.9` | `3049.8` [`3235.4`] | `6562` | `490` | `0.965233` ± `0.0117` | `0.97x` |
| `pytorch_fp32` | `0.5` / `4` | `3.77` | `12.68` | `254.6` | `2952.0` [`2970.8`] | `6208` | `487` | `0.965228` ± `0.0117` | `1.00x` |
| `pytorch_tf32` | `0.5` / `4` | `3.77` | `12.68` | `254.6` | `2952.0` [`2970.8`] | `6208` | `487` | `0.965228` ± `0.0117` | `1.00x` |
| `pytorch_amp` | `0.5` / `4` | `5.72` | `14.50` | `213.4` | `2916.1` [`2954.5`] | `6364` | `457` | `0.965217` ± `0.0118` | `1.01x` |
| `pytorch_fp16` | `0.5` / `4` | `4.86` | `11.13` | `213.9` | `2885.5` [`2822.2`] | `6230` | `458` | `0.965231` ± `0.0118` | `1.02x` |
| `trt_fp32` | `0.5` / `4` | `3.12` | `11.78` | `239.8` | `2940.0` [`3003.2`] | `6387` | `368` | `0.965233` ± `0.0117` | `1.00x` |
| `trt_fp16_unfolded` | `0.5` / `4` | `0.98` | `3.09` | `176.7` | `2866.4` [`2909.5`] | `6115` | `359` | `0.965722` ± `0.0120` | `1.03x` |
| `trt_fp16` | `0.5` / `4` | `0.88` | `3.01` | `167.1` | `2880.3` [`2893.9`] | `6094` | `359` | `0.965226` ± `0.0117` | `1.02x` |
| `pytorch_tf32_gpu` | `0.5` / `4` | `3.77` | `11.57` | `197.0` | `357.9` [`244.6`] | `631` | `2307` | `0.965227` ± `0.0117` | `8.25x` |
| `trt_fp16_gpu` | `0.5` / `4` | `0.94` | `3.18` | `47.6` | `215.5` [`173.7`] | `440` | `2301` | `0.965228` ± `0.0117` | `13.70x` |
| `trt_fp16_cuda` | `0.5` / `4` | `1.01` | `3.07` | `49.0` | `96.3` [`58.9`] | `233` | `455` | `0.965229` ± `0.0117` | `30.67x` |
| `trt_fp16_optimized` | `0.25` / `8` | `1.08` | `2.98` | `24.8` | `68.9` [`50.5`] | `123` | `522` | `0.964778` ± `0.0135` | `42.86x` |
| `trt_fp16_nn` | `0.25` / `8` | `1.07` | `3.00` | `20.8` | `178.4` [`141.3`] | `399` | `2149` | `0.960230` ± `0.0139` | `16.55x` |

![Stage latency breakdown](docs/figures/stage_breakdown.png)

### Per-stage medians

`ingest` is `load + h2d`. Quote it instead of `load` alone, because `mmap` defers the read into `h2d`.

| Stage (median ms) | `pytorch_fp32_strict` | `pytorch_fp32` | `pytorch_tf32` | `pytorch_amp` | `pytorch_fp16` | `trt_fp32` | `trt_fp16_unfolded` | `trt_fp16` | `pytorch_tf32_gpu` | `trt_fp16_gpu` | `trt_fp16_cuda` | `trt_fp16_optimized` | `trt_fp16_nn` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `load` | `363.9` | `421.6` | `421.6` | `416.3` | `331.7` | `369.4` | `398.1` | `365.3` | `1.5` | `1.5` | `1.5` | `1.5` | `1.5` |
| `h2d` | `3.9` | `3.8` | `3.8` | `4.2` | `4.8` | `3.7` | `16.5` | `14.0` | `37.3` | `37.4` | `27.6` | `27.7` | `36.6` |
| `ingest` | `367.4` | `424.7` | `424.7` | `419.7` | `340.1` | `373.0` | `406.5` | `385.1` | `38.8` | `38.9` | `29.2` | `29.1` | `38.1` |
| `preprocess` | `288.0` | `286.4` | `286.4` | `271.7` | `281.6` | `276.7` | `273.2` | `266.8` | `54.4` | `43.9` | `1.3` | `1.3` | `41.7` |
| `sw_network` | `379.9` | `254.6` | `254.6` | `213.4` | `213.9` | `239.8` | `176.7` | `167.1` | `197.0` | `47.6` | `49.0` | `24.8` | `20.8` |
| `sw_aggregate` | `8.4` | `8.0` | `8.0` | `9.1` | `9.6` | `8.9` | `36.7` | `36.0` | `6.5` | `6.5` | `6.1` | `4.5` | `4.1` |
| `postprocess` | `2024.8` | `1990.4` | `1990.4` | `2013.6` | `1993.2` | `1981.5` | `1994.5` | `1993.9` | `62.2` | `71.1` | `2.6` | `2.8` | `69.3` |
| `e2e_total` | `3049.8` | `2952.0` | `2952.0` | `2916.1` | `2885.5` | `2940.0` | `2866.4` | `2880.3` | `357.9` | `215.5` | `96.3` | `68.9` | `178.4` |
| `wall_ms` | `3050.0` | `2952.1` | `2952.1` | `2916.2` | `2885.6` | `2940.1` | `2866.5` | `2880.5` | `357.9` | `215.5` | `96.5` | `69.0` | `178.5` |

### Per-case latency, fast profile

`69 ms` is a median across cases of different sizes: `9` of `9` cases have a median under 200 ms.

| Case | Shape | Windows | E2E median (ms) | Slowest (ms) | Peak VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `spleen_56` | `512x512x33` | `18` | `36.0` | `38.0` | `285` |
| `spleen_59` | `512x512x50` | `50` | `71.6` | `79.7` | `404` |
| `spleen_6` | `512x512x135` | `50` | `108.2` | `122.6` | `452` |
| `spleen_60` | `512x512x97` | `64` | `101.8` | `109.7` | `522` |
| `spleen_61` | `512x512x101` | `64` | `104.8` | `109.6` | `521` |
| `spleen_62` | `512x512x80` | `27` | `70.1` | `73.6` | `394` |
| `spleen_63` | `512x512x60` | `32` | `63.3` | `67.5` | `362` |
| `spleen_8` | `512x512x31` | `32` | `47.9` | `50.1` | `308` |
| `spleen_9` | `512x512x41` | `32` | `52.5` | `54.7` | `296` |

## 5. Network-only comparison

These runs use identical windows on every backend, in a sustained loop. Fast windows use `369` windows across the 9 cases against `665` for bundle windows, so comparing a fast-window time to a bundle-window time measures less work, not a faster network.

| Backend | Bundle windows (ms) | Fast windows (ms) | TensorRT speedup, bundle windows |
| :--- | :---: | :---: | :---: |
| PyTorch FP32 | `315.5` | `219.1` | `1.08x` |
| PyTorch TF32 | `305.6` | `136.4` | `1.05x` |
| PyTorch TF32, `cudnn.benchmark` | `305.4` | `139.7` | `1.05x` |
| PyTorch AMP | `237.4` | `415.5` | `0.81x` |
| PyTorch AMP, `cudnn.benchmark` | `694.1` | `484.3` | `2.38x` |
| TensorRT FP16 | `292.2` | `239.4` | `1.00x` |

Inside the real pipeline, TensorRT is `3.63x` faster than TF32 at bundle windows and `4.75x` at fast windows.

### GPU clock state

`sw_network` depends on what the GPU was doing just before, so values from different pipelines are not directly comparable. The same work takes longer after a 3 s idle gap, which is consistent with the laptop GPU lowering its clocks during long CPU stages. Sustained loops can also run slower than the short bursts inside the real pipeline, so the table above and the in-pipeline ratios both matter.

| Backend | Back-to-back (ms) | After 3 s idle (ms) | Inflation |
| :--- | :---: | :---: | :---: |
| `trt` | `290.7` | `328.0` | `1.13x` |
| `tf32` | `1119.7` | `1267.6` | `1.13x` |

## 6. Ingestion

Warm OS page cache. `mmap` returns before reading, so only the right-hand column is a fair comparison. Nearly all of the gain comes from not decompressing gzip.

| Read path | Read call (ms) | Through to GPU (ms) |
| :--- | :---: | :---: |
| `.nii.gz` with `LoadImaged` | `388.4` | `430.8` |
| `.nii`, eager read | `207.1` | `249.4` |
| `.nii`, `mmap` (lazy) | `3.2` | `47.2` |

Resampling runs on the GPU in `float64`, which is MONAI's default and is slow on GeForce hardware. Only the final mask copy is `uint8`.

## 7. Amdahl analysis

| Profile | `sw_network` (ms) | E2E (ms) | Network share | Max E2E gain from a faster network |
| :--- | :---: | :---: | :---: | :---: |
| `pytorch_fp32` | `254.6` | `2952.0` | `8.6%` | `1.09x` |
| `trt_fp16` | `167.1` | `2880.3` | `5.8%` | `1.06x` |
| `pytorch_tf32_gpu` | `197.0` | `357.9` | `55.1%` | `2.22x` |
| `trt_fp16_optimized` | `24.8` | `68.9` | `36.0%` | `1.56x` |

Once the network is a small share of end-to-end time, a faster network barely moves the total.

## 8. Sliding-window sweep

Uses the fast profile's own post-processing, with 5 timed passes per case on a hot GPU. The fast profile is not on the measured Pareto frontier of this sweep; its window settings are a judgment call. The cases are the evaluation cases, so this is selection on the evaluation set.

| Overlap | Batch | Median (ms) | IQR (ms) | Added VRAM (MB) | Mean Dice | Pareto-optimal | Fast profile |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0.25` | `1` | `42.3` | `23.5` | `791` | `0.964778` |  |  |
| `0.25` | `2` | `33.2` | `20.6` | `809` | `0.964778` |  |  |
| `0.25` | `4` | `30.4` | `18.6` | `843` | `0.964778` | yes |  |
| `0.25` | `8` | `32.6` | `21.0` | `910` | `0.964778` |  | yes |
| `0.50` | `1` | `88.7` | `32.9` | `791` | `0.965229` |  |  |
| `0.50` | `2` | `71.4` | `53.0` | `808` | `0.965229` |  |  |
| `0.50` | `4` | `66.5` | `23.8` | `842` | `0.965229` | yes |  |
| `0.50` | `8` | `69.3` | `30.8` | `909` | `0.965229` |  |  |
| `0.75` | `1` | `410.3` | `315.6` | `791` | `0.965173` |  |  |
| `0.75` | `2` | `332.3` | `312.5` | `808` | `0.965173` |  |  |
| `0.75` | `4` | `334.6` | `336.1` | `842` | `0.965173` |  |  |
| `0.75` | `8` | `424.5` | `411.6` | `909` | `0.965173` |  |  |

![Sliding-window sweep](docs/figures/pareto_sweep.png)

## 9. Agreement with true FP32

All rows compare with `pytorch_fp32_strict` (mean Dice `0.965233`). The sample is `9` cases, so these limits of agreement are imprecise.

| Profile | Mean Dice | Mean ΔDice | Mean abs ΔDice | Max abs ΔDice | Cases ≥ 0.005 | Volume bias (cm³) [95% CI] | 95% LoA (cm³) | Volume larger than FP32 | MAE vs ground truth (cm³) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `pytorch_fp32` | `0.965228` | `-0.000006` | `0.000011` | `0.000045` | `0` | `+0.00` [`+0.00`, `+0.01`] | [`-0.00`, `+0.01`] | `7/9` | `5.17` |
| `pytorch_tf32` | `0.965228` | `-0.000006` | `0.000011` | `0.000045` | `0` | `+0.00` [`+0.00`, `+0.01`] | [`-0.00`, `+0.01`] | `7/9` | `5.17` |
| `pytorch_amp` | `0.965217` | `-0.000016` | `0.000020` | `0.000116` | `0` | `+0.00` [`-0.01`, `+0.01`] | [`-0.02`, `+0.02`] | `6/9` | `5.17` |
| `pytorch_fp16` | `0.965231` | `-0.000002` | `0.000024` | `0.000066` | `0` | `+0.02` [`+0.01`, `+0.04`] | [`-0.00`, `+0.05`] | `9/9` | `5.15` |
| `trt_fp32` | `0.965233` | `+0.000000` | `0.000000` | `0.000000` | `0` | `+0.00` [`+0.00`, `+0.00`] | [`+0.00`, `+0.00`] | `0/9` | `5.17` |
| `trt_fp16_unfolded` | `0.965722` | `+0.000489` | `0.000555` | `0.001386` | `0` | `+1.68` [`+1.25`, `+2.10`] | [`+0.59`, `+2.76`] | `9/9` | `4.24` |
| `trt_fp16` | `0.965226` | `-0.000007` | `0.000015` | `0.000033` | `0` | `+0.00` [`-0.01`, `+0.02`] | [`-0.04`, `+0.05`] | `5/9` | `5.17` |
| `pytorch_tf32_gpu` | `0.965227` | `-0.000007` | `0.000011` | `0.000045` | `0` | `+0.00` [`-0.00`, `+0.01`] | [`-0.01`, `+0.01`] | `5/9` | `5.17` |
| `trt_fp16_gpu` | `0.965228` | `-0.000006` | `0.000013` | `0.000029` | `0` | `+0.00` [`-0.01`, `+0.01`] | [`-0.03`, `+0.03`] | `4/9` | `5.17` |
| `trt_fp16_cuda` | `0.965229` | `-0.000004` | `0.000014` | `0.000029` | `0` | `+0.00` [`-0.01`, `+0.02`] | [`-0.04`, `+0.04`] | `5/9` | `5.17` |
| `trt_fp16_optimized` | `0.964778` | `-0.000456` | `0.001669` | `0.005563` | `1` | `+0.12` [`-0.79`, `+1.03`] | [`-2.20`, `+2.44`] | `5/9` | `5.11` |
| `trt_fp16_nn` | `0.960230` | `-0.005004` | `0.005004` | `0.011293` | `4` | `+0.48` [`-0.34`, `+1.31`] | [`-1.62`, `+2.58`] | `6/9` | `4.75` |

TensorRT FP16 shows no systematic volume shift at bundle windows: `+0.001 cm³`, with `5` of `9` cases larger than FP32. Across all sliding windows TensorRT flipped `478` window-voxels from background to foreground and `418` the other way. The largest logit difference is `0.072` on logits up to `23.0`.

### The BatchNorm fold

The exported ONNX graph keeps `BatchNorm` after each `ConvTranspose3d`. Layer `model.2.0.adn.N` applies a shift of `81.8`, a magnitude where FP16 values are spaced `0.0625` apart. Folding these layers into the preceding `ConvTranspose3d` in float64 before export changes FP32 logits by at most `1.4e-05`, so it does not change the model. Both engines below run the same inputs and windows, compared with FP32 on the unmodified bundle weights.

| Engine | Mean Dice | Mean abs ΔDice | Volume bias (cm³) | Volume larger than FP32 |
| :--- | :---: | :---: | :---: | :---: |
| Unfolded FP16 engine | `0.965706` | `0.000557` | `+1.672` | `9/9` |
| Folded FP16 engine | `0.965229` | `0.000014` | `+0.001` | `5/9` |

The unfolded engine's higher Dice is a side effect of that shift, not better accuracy.

### Per-case agreement, fast profile

| Case | V ground truth | V FP32 | V fast | ΔV | Dice FP32 | Dice fast | ΔDice |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `spleen_56` | `78.07` | `75.42` | `75.54` | `+0.123` | `0.936143` | `0.930580` | `-0.005563` |
| `spleen_59` | `426.65` | `425.40` | `426.60` | `+1.199` | `0.966229` | `0.965582` | `-0.000647` |
| `spleen_6` | `209.52` | `209.06` | `209.75` | `+0.682` | `0.973239` | `0.972804` | `-0.000435` |
| `spleen_60` | `299.84` | `288.18` | `288.55` | `+0.377` | `0.971020` | `0.971488` | `+0.000468` |
| `spleen_61` | `163.30` | `160.80` | `160.15` | `-0.649` | `0.971511` | `0.970834` | `-0.000677` |
| `spleen_62` | `275.93` | `275.33` | `273.25` | `-2.078` | `0.961134` | `0.959673` | `-0.001461` |
| `spleen_63` | `356.60` | `342.12` | `344.19` | `+2.064` | `0.963625` | `0.968380` | `+0.004755` |
| `spleen_8` | `501.55` | `492.81` | `492.36` | `-0.449` | `0.973413` | `0.972633` | `-0.000780` |
| `spleen_9` | `269.15` | `264.92` | `264.73` | `-0.194` | `0.970787` | `0.971025` | `+0.000238` |

![Bland-Altman agreement](docs/figures/bland_altman.png)

## 10. Error analysis

These are observations about where errors fall, not explanations of why. FP16 is not the main source of error on these cases: the Dice change from FP32 is far smaller than the distance from a perfect score.

- **`spleen_56`:** Dice `0.9361` (FP32 `0.9361`), volume `75.4` vs `78.1 cm³`, spacing `[0.699, 0.699, 7.0]` mm. False-positive voxels `1044`, false-negative `1819`. The `1` end slices at each pole, plus any slices beyond the organ, hold `40%` of error voxels and `14%` of ground-truth voxels. `49%` of ground-truth voxels are surface voxels.
- **`spleen_62`:** Dice `0.9611` (FP32 `0.9611`), volume `275.4` vs `275.9 cm³`, spacing `[0.676, 0.676, 5.0]` mm. False-positive voxels `4566`, false-negative `4821`. The `2` end slices at each pole, plus any slices beyond the organ, hold `17%` of error voxels and `4%` of ground-truth voxels. `21%` of ground-truth voxels are surface voxels.

Across all `9` cases, Dice is not clearly related to slice thickness (Spearman `-0.13`, p = `0.74`) or to spleen volume (Spearman `+0.23`, p = `0.55`). The coarsest scan, `spleen_8` at `7.5 mm`, ranks number `1` of `9` by Dice.

![Lowest-Dice case](docs/figures/failure_case_1.png)
![Second-lowest-Dice case](docs/figures/failure_case_2.png)

## 11. Limitations

- **Small sample:** 9 cases from one public dataset, which also selected the checkpoint.
- **One machine:** one laptop GPU, whose clocks change with load and temperature. Expect different absolute numbers elsewhere.
- **Best-case I/O:** the GPU path assumes volumes are already decompressed and in the OS page cache. Cold-disk reads and the one-time decompression are not included.
- **Warm process:** model loading, engine deserialization, and CUDA start-up are excluded.
- **Not a product path:** no DICOM ingestion, result saving, or concurrency is measured.
- **Engine builds:** TensorRT selects kernels at build time, so a rebuilt engine can differ slightly.

## 12. Reproduce

Check that this README matches `results/`. No GPU or data needed:

```bash
python -m medtrt.verify_claims
```

Regenerate everything on an NVIDIA GPU. This downloads the dataset and bundle, and takes about an hour:

```bash
pip install -r requirements-lock.txt
pip install -e .
python -m medtrt.pipeline
```

Run the tests, or the viewer:

```bash
python -m pytest tests/
streamlit run app/viewer.py
```

## 13. Data and licenses

The dataset, weights, and engines are not in this repository. `Task09_Spleen` is from the [Medical Segmentation Decathlon](http://medicaldecathlon.com/) under CC-BY-SA 4.0, and the figures in `docs/figures/failure_case_*.png` are derived from it. The MONAI bundle is Apache-2.0.
