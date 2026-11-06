# Medtronic Biomedical Imaging, Quantitative Validation & Verification Report

> **Clinical Scope & Intended Use Statement:**
> MedTRT is a quantitative 3D medical imaging research prototype evaluated on the Medical Segmentation Decathlon `Task09_Spleen` dataset ($n = 9$ validation CT volumes). It is **not FDA-approved, not CE-marked, and not certified for clinical diagnosis or surgical navigation**. All engineering verification controls follow medical software verification principles (traceability, deterministic reproducibility, physical coordinate validation, and statistical agreement bounds).

---

## 1. Executive Summary of Biomedical Engineering & Validation Findings

| Validation Dimension | Strict IEEE FP32 Reference | TensorRT FP16 + Custom CUDA (`trt_fp16_cuda`) | Pareto Fast Profile (`trt_fp16_optimized`) | Verification Status |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Spleen Dice ± SD (Median)** | `0.9652 ± 0.0117` (`0.9708`) | **`0.9652 ± 0.0117`** (`0.9708`) | **`0.9648 ± 0.0135`** (`0.9708`) | **PASS** (`>= 0.960`) |
| **Mean IoU / Sensitivity / Precision** | `0.9330` / `0.9562` / `0.9746` | `0.9330` / `0.9562` / `0.9746` | `0.9322` / `0.9571` / `0.9731` | **PASS** |
| **Boundary Surface Distance `HD95` / `ASSD` (mm)** | `2.18 ± 0.99 mm` / `0.39 ± 0.12 mm` | **`2.18 ± 0.99 mm` / `0.39 ± 0.12 mm`** | `2.35 ± 1.24 mm` / `0.42 ± 0.14 mm` | **PASS** (Sub-voxel ASSD) |
| **Cross-Backend Dice Drift (`Mean |ΔDice|` / `Max`)** | Reference (`0.0`) | **`0.000014` / `0.000029`** (`0/9 >= 0.005`) | `0.001669` / `0.005563` (`1/9 >= 0.005`) | **PASS** (`< 0.005` like-for-like) |
| **Cross-Backend Volume Bias vs. FP32 (`95% LoA`)** | Reference (`0.0 cm³`) | **`+0.0010 ± 0.0194 cm³`** (`[-0.037, +0.039] cm³`) | `+0.1196 ± 1.1815 cm³` (`[-2.20, +2.44] cm³`) | **PASS** (`p = 0.885`, unbiased) |
| **Anatomical Volume Error vs. GT (`MAE` / `Vol Sim`)** | `5.17 cm³` (`1.88%`) / `0.9905` | `5.17 cm³` (`1.88%`) / `0.9905` | `5.11 cm³` (`1.84%`) / `0.9907` | **Quantified** |
| **End-to-End Inference Latency (Median / Max)** | `3122.2 ms` / `7759.5 ms` | **`95.10 ms`** / **`240.71 ms`** | **`60.83 ms`** / **`115.30 ms`** | **PASS** (`< 500 ms` all passes) |

---

## 2. Preservation of Physical 3D Patient Geometry & Affine Integrity

In image-guided navigation and volumetric organ assessment, resampling a 3D CT scan to isotropic/canonical model spacing (`1.5 × 1.5 × 2.0 mm` in `RAS`) and inverting the segmentation back to native patient coordinates must preserve exact physical geometry:
1. **Analytical `RAS` Affine & Shape Derivation (`compute_ras_resampled_geometry` in `src/medtrt/cuda_kernels.py`):**
   Matches MONAI's `Orientationd(axcodes="RAS") + Spacingd(pixdim=(1.5, 1.5, 2.0))` output affine matrix to **`4.44e-16`** machine epsilon.
2. **Why Probability-Space Trilinear Inversion Beats Discrete Nearest-Neighbor Inversion:**
   Inverting discretized `uint8` labels (`trt_fp16_nn`) causes staircase aliasing across thick axial slices (`5.0–8.0 mm`), degrading mean Dice from **`0.9648` to `0.9602`** (`-0.0046` Dice drop, with 4/9 cases exceeding the `0.005` tolerance). Our custom CUDA C++ kernel (`fused_invert_trilinear_threshold_u8`) performs trilinear interpolation on continuous foreground probabilities before thresholding at `0.5`, preserving the full **`0.9648–0.9652` Dice** while executing in **`1.00 ms`**.

---

## 3. Software Verification & Requirements Traceability Matrix (`tests/`)

| Requirement ID | Verification Requirement Description | Verification Test Module & Function | Acceptance Criterion | Status |
| :--- | :---: | :---: | :---: | :---: |
| `REQ-GEO-01` | Physical volumetry via affine determinant $|\det(A_{3\times 3})|$ and spacing product | `tests/test_volume_math.py::test_synthetic_binary_mask_volume_cm3` | Exact match to `1e-7 cm³` on synthetic & empty masks | **PASS** |
| `REQ-PRE-01` | HU windowing `[-57, 164] -> [0, 1]` and 4D channel-first tensor geometry | `tests/test_transforms.py`, `tests/test_cuda_kernels.py::test_hu_window_normalize_scalar_and_vec4_exact` | `min >= 0.0`, `max <= 1.0`, `atol <= 1e-6` | **PASS** |
| `REQ-CUDA-01` | Custom CUDA C++ fused 3D preprocessing matches MONAI `Orientationd+Spacingd+ScaleIntensityRanged` | `tests/test_cuda_kernels.py::test_fused_preprocess_matches_monai_on_real_ct` | `max_abs_diff < 1e-4`, `mean_abs_diff < 1e-6` on real CT | **PASS** |
| `REQ-CUDA-02` | Custom CUDA C++ fused postprocessing & warp-shuffle voxel reduction exactness | `tests/test_cuda_kernels.py::test_fused_invert_postprocess_matches_geometry`, `test_count_nonzero_u8_warp_shuffle_reduction` | Exact native shape, `uint8` dtype, exact foreground voxel count | **PASS** |
| `REQ-TRT-01` | Zero host-to-device or device-to-host copies inside TensorRT 10.x forward pass | `tests/test_engine_accuracy.py::test_trt_forward_never_transfers_to_host` | Monkeypatched `.cpu()` / `.numpy()` never invoked; `host_copies_count == 0` | **PASS** |
| `REQ-NUM-01` | `ConvTranspose3d + BatchNorm3d` FP64 folding preserves FP32 model equivalence | `tests/test_engine_accuracy.py::test_batchnorm_fold_does_not_change_the_fp32_model` | `max_abs_diff < 1e-3`, `100%` argmax match | **PASS** |
| `REQ-EQV-01` | End-to-end cross-backend segmentation & volumetric equivalence on real CT | `tests/test_integration_pipeline.py::test_cross_backend_end_to_end_equivalence_on_real_ct` | Cross-backend Dice `> 0.995`, volume drift `< 1.5 cm³` | **PASS** |
| `REQ-SAFE-01` | Defensive rejection of missing engines, invalid spacing, CPU tensors, and out-of-range batch sizes | `tests/test_failure_handling.py` (5 failure-handling tests) | Explicit `FileNotFoundError` / `ValueError` / `RuntimeError` raised | **PASS** |
| `REQ-DOC-01` | Automated audit of published claims, provenance hashes, and clinical scope language | `tests/test_published_results.py` (`src/medtrt/verify_claims.py`) | 10/10 automated integrity checks pass | **PASS** |

---

## 4. Interactive Tri-Planar Clinical Research Workstation (`app/viewer.py`)

The Streamlit workstation (`streamlit run app/viewer.py`) provides:
- **Synchronized Tri-Planar Radiological Inspection:** Axial, Coronal, and Sagittal views with physical aspect-ratio scaling (`s_z / s_x`, `s_z / s_y`), configurable HU window/level presets (`Soft Tissue 400/40`, `Abdominal/Spleen 221/53.5`, `Liver 150/30`, `Bone 1800/400`), and auto-jump to the maximum-area or maximum-disagreement slice.
- **Live Backend & Precision Selector:** Switch between `trt_fp16_optimized`, `trt_fp16_cuda`, `trt_fp16`, `trt_fp32`, `pytorch_fp32`, and `pytorch_fp32_strict`.
- **Side-by-Side Cross-Backend Comparison & Difference-Mask Overlay:** Visualizes voxel-by-voxel disagreement (`|Pred_A - Pred_B|`) between any two backends (e.g., `TensorRT FP16` vs. `PyTorch Strict FP32`) and against ground truth (`Green = GT`, `Red = Pred`, `Yellow = Agreement`).
- **Ground-Truth Safeguard:** Automatically disables ground-truth metrics when inspecting unlabeled volumes (`imagesTs/`) so silent zero-arrays are never reported.
