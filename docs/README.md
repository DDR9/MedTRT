# MedTRT Documentation & Technical Studies Index

Welcome to the **MedTRT** documentation directory. Every number, table, and plot across these documents is deterministically generated from the hardware-verified telemetry in [`../results/`](../results/) and audited by `python -m medtrt.verify_claims`.

---

## 1. Primary Technical Manuscript

| Document | Format | Description |
| :--- | :---: | :--- |
| **[`../MedTRT_Scientific_Report.pdf`](../MedTRT_Scientific_Report.pdf)** | PDF (Root) | **12-Page Two-Column Scientific Report (Root Quick-Access Copy)** — Standalone publication-style report with structured Abstract, mathematical derivations, 8 tables, 13 figures, 9 custom CUDA C++17 kernels, analytical `ConvTranspose3d + BatchNorm3d` FP64 weight folding, and 53-volume TOST equivalence validation. |
| **[`TECHNICAL_PAPER.pdf`](TECHNICAL_PAPER.pdf)** | PDF | **12-Page Two-Column Technical Manuscript** — Identical compiled manuscript archived inside `docs/`. |
| **[`TECHNICAL_PAPER.md`](TECHNICAL_PAPER.md)** | Markdown | Full GitHub-readable Markdown edition of the technical paper. |
| **[`TECHNICAL_PAPER.tex`](TECHNICAL_PAPER.tex)** | LaTeX | Complete LaTeX source code for compiling `MedTRT_Scientific_Report.pdf` and `TECHNICAL_PAPER.pdf`. |

---

## 2. System Architecture, Executive Summary & Master Results Tables

| Document | Focus Area | Description |
| :--- | :--- | :--- |
| **[`ARCHITECTURE.md`](ARCHITECTURE.md)** | System Design | End-to-end 3D CT segmentation pipeline architecture, Mermaid dataflow diagrams, and module-by-module code guide. |
| **[`KEY_FINDINGS_AND_FAQ.md`](KEY_FINDINGS_AND_FAQ.md)** | Executive Summary & FAQ | Concise summary of headline speedups, VRAM reductions, numerical stabilization results, and technical Q&A. |
| **[`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md)** | Performance Tables | Complete 13-profile latency tables, stage breakdowns, sliding-window Pareto sweep, and GPU memory footprint telemetry. |
| **[`VALIDATION_RESULTS.md`](VALIDATION_RESULTS.md)** | Accuracy & Volumetry | Complete 10-metric segmentation accuracy tables (`Dice`, `IoU`, `HD95`, `ASSD`, physical volume) and Bland–Altman agreement bounds. |

---

## 3. Deep-Dive Engineering & Validation Studies

| Document | Subsystem | Key Technical Topics Covered |
| :--- | :--- | :--- |
| **[`GPU_SYSTEMS_ENGINEERING.md`](GPU_SYSTEMS_ENGINEERING.md)** | GPU Systems Overview | End-to-end speedup analysis (`42.9x` fast / `30.7x` matched), 7-rung controlled ablation waterfall, and hardware utilization on NVIDIA Blackwell (`sm_120`). |
| **[`GPU_PROFILING_STUDY.md`](GPU_PROFILING_STUDY.md)** | Timeline & Amdahl's Law | Six-stage NVTX / CUDA Event decomposition (`load`, `h2d`, `preprocess`, `sw_network`, `sw_aggregate`, `postprocess`), Amdahl's Law ceiling, and laptop GPU DVFS clock-state inflation. |
| **[`CUDA_KERNEL_STUDY.md`](CUDA_KERNEL_STUDY.md)** | Custom CUDA C++17 Kernels | Design and register/occupancy profiling of all 9 custom CUDA kernels in [`../cuda/medical_kernels.cu`](../cuda/medical_kernels.cu), register-based $3 \times 4$ affine coordinate math (`5.06x` VRAM reduction), and volume-scaling benchmarks. |
| **[`TENSORRT_PRECISION_STUDY.md`](TENSORRT_PRECISION_STUDY.md)** | FP16 Numerical Stability | First-principles diagnosis of the `-81.79` decoder `BatchNorm3d` FP16 cancellation bug (`+1.68 cm³` volume bias) and derivation of analytical `float64` weight folding (`35.6x` lower logit drift). |
| **[`MEDICAL_VALIDATION_STUDY.md`](MEDICAL_VALIDATION_STUDY.md)** | Volumetry & Equivalence | Physical organ volumetry via affine determinant $|\det(A_{3\times 3})|$, dual Bland–Altman analysis, and paired Two One-Sided Tests (TOST) equivalence across MSD ($N=9$) and external MICCAI `FLARE22` ($N=12$). |
| **[`ANATOMICAL_FAILURE_ANALYSIS.md`](ANATOMICAL_FAILURE_ANALYSIS.md)** | Boundary Error Analysis | Voxel-level decomposition of false positives/negatives at superior/inferior polar domes (`spleen_56`) and hilar vessel bifurcations (`spleen_62`), plus slice-thickness correlation analysis. |
| **[`BIOMEDICAL_VERIFICATION_REPORT.md`](BIOMEDICAL_VERIFICATION_REPORT.md)** | Software Traceability | Requirements traceability matrix (`REQ-GEO-01` through `REQ-DOC-01`) mapping verification requirements to the 66 automated `pytest` tests and interactive 3D Streamlit workstation. |

---

## 4. Reproducibility & Audit Reports

| Document | Purpose |
| :--- | :--- |
| **[`CLAIM_VERIFICATION_REPORT.md`](CLAIM_VERIFICATION_REPORT.md)** | Master claim-to-evidence verification matrix reconciling all published metrics against raw JSON/CSV telemetry (`code_fingerprint = e9d4f060aaf1b130`). |
| **[`BASELINE_AUDIT_REPORT.md`](BASELINE_AUDIT_REPORT.md)** | Initial engineering audit documenting the root-cause discovery of host CPU resampling bottlenecks, confounded window comparisons, and unfolded FP16 drift. |
| **[`figures/`](figures/)** | Directory containing all 16 publication-quality PNG figures generated from `results/`. |
