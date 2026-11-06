"""
CT Workstation Page (`app/workstation.py`).

Opened from the dashboard in `app/viewer.py`. Heavy imports live here so that the results pages never
load the pipeline, and inference runs only when asked, so opening a page cannot disturb a benchmark.

Features:
- Header banner: "Research and Verification Prototype Only — Not for Clinical Diagnostic Use."
- Load held-out cases from `./data/Task09_Spleen/` (`results/heldout_cases.json`) or uploaded NIfTI.
- Synchronized Axial, Coronal, and Sagittal slice sliders with crosshair markers and RAS orientation labels.
- Radiologic windowing presets (Abdominal Soft Tissue: W=400, L=40; Bone: W=1800, L=400; Lung: W=1500, L=-600; Custom)
  applied strictly to screen render buffers, never to model inputs.
- Mask view modes: Translucent Alpha Overlay, 1-Pixel Boundary Contour (Green=GT, Red=Pred, Yellow=Agreement),
  Difference / Disagreement Highlight, or Hidden.
- Side-by-Side Backend Comparison Mode:
  Compare any Primary Backend (e.g., `trt_fp16_optimized`) against a Reference Backend (e.g., `pytorch_fp32_strict`)
  with live disagreement voxel count, volume delta (cm³), latency speedup, and boundary difference visualization.
- Ground-truth safeguard: Dice, HD95, and ASSD metrics are displayed ONLY when ground-truth segmentation
  labels exist for the loaded volume.
- Volume details (voxel count, spacing, HU range, orientation), jump to the peak spleen slice, an interactive
  slice renderer, and `.nii.gz` mask download.
"""

from __future__ import annotations

import gzip
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from scipy.ndimage import binary_erosion

MEDTRT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = MEDTRT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medtrt.bench import run_single_timed_volume_pass
from medtrt.data import HELDOUT_JSON_PATH
from medtrt.evaluate import (
    compute_all_segmentation_metrics,
    compute_volume_cm3,
)
from medtrt.infer import build_pipeline, infer_volume
from medtrt.profiles import PROFILES, REFERENCE, SHIPPING, build_predictor, get_profile

GPU_LOCK = MEDTRT_ROOT / "results" / ".gpu_busy"


@st.cache_resource(show_spinner="Initializing GPU Predictor for Selected Profile...")
def get_cached_predictor(profile_name: str) -> Tuple[Any, bool, float]:
    device = torch.device("cuda:0")
    profile = get_profile(profile_name)
    return build_predictor(profile, device)


@st.cache_data(show_spinner="Loading NIfTI CT Volume & Spatial Metadata...")
def load_cached_nifti(img_path: str, lbl_path: Optional[str] = None) -> Dict[str, Any]:
    nii = nib.load(img_path)
    vol = nii.get_fdata(dtype=np.float32)
    if vol.ndim == 4:
        vol = vol[..., 0]
    zooms = nii.header.get_zooms()[:3]
    spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    affine = np.asarray(nii.affine, dtype=np.float64)
    axcodes = "".join(nib.aff2axcodes(affine))

    gt_vol = None
    if lbl_path and Path(lbl_path).exists():
        gt_vol = np.asarray(nib.load(lbl_path).dataobj).astype(np.uint8)

    return {
        "volume": vol,
        "gt_mask": gt_vol,
        "shape": vol.shape,
        "spacing": spacing,
        "affine": affine,
        "axcodes": axcodes,
        "hu_min": float(vol.min()),
        "hu_max": float(vol.max()),
    }


def apply_display_window(slice_2d: np.ndarray, width: float, level: float) -> np.ndarray:
    """Apply display windowing (W, L) strictly for screen rendering."""
    low = level - (width / 2.0)
    high = level + (width / 2.0)
    clipped = np.clip(slice_2d, low, high)
    return ((clipped - low) / max(high - low, 1e-6) * 255.0).astype(np.uint8)


def extract_1px_contour(mask_2d: np.ndarray) -> np.ndarray:
    b = mask_2d > 0
    if not np.any(b):
        return np.zeros_like(b, dtype=bool)
    eroded = binary_erosion(b, iterations=1, border_value=0)
    return b & (~eroded)


def render_plane_rgb(
    ct_2d: np.ndarray,
    pred_2d: Optional[np.ndarray],
    gt_2d: Optional[np.ndarray],
    cmp_2d: Optional[np.ndarray],
    width: float,
    level: float,
    mask_mode: str,
    alpha: float,
) -> np.ndarray:
    gray = apply_display_window(np.rot90(ct_2d), width=width, level=level)
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)

    if pred_2d is not None:
        pr_rot = np.rot90(pred_2d > 0)
        if mask_mode == "Translucent Overlay":
            if np.any(pr_rot):
                rgb[pr_rot] = (1.0 - alpha) * rgb[pr_rot] + alpha * np.array([255.0, 65.0, 54.0])
        elif mask_mode == "1-Pixel Boundary Contour":
            if gt_2d is not None:
                gt_c = extract_1px_contour(np.rot90(gt_2d))
                rgb[gt_c] = [0.0, 230.0, 118.0]  # Green GT contour
            pr_c = extract_1px_contour(pr_rot)
            rgb[pr_c] = [255.0, 65.0, 54.0]      # Red Predicted contour
            if gt_2d is not None:
                gt_c = extract_1px_contour(np.rot90(gt_2d))
                rgb[gt_c & pr_c] = [255.0, 220.0, 0.0]  # Yellow exact overlap
        elif mask_mode == "Difference / Disagreement Mask":
            ref_rot = np.rot90(cmp_2d > 0) if cmp_2d is not None else (np.rot90(gt_2d > 0) if gt_2d is not None else None)
            if ref_rot is not None:
                agree = pr_rot & ref_rot
                only_pred = pr_rot & (~ref_rot)
                only_ref = (~pr_rot) & ref_rot
                rgb[agree] = (1.0 - 0.35) * rgb[agree] + 0.35 * np.array([0.0, 200.0, 83.0])
                rgb[only_pred] = [255.0, 23.0, 68.0]   # Magenta/Red: Primary only
                rgb[only_ref] = [0.0, 229.0, 255.0]    # Cyan: Reference/GT only

    return np.clip(rgb, 0, 255).astype(np.uint8)


def compute_craniocaudal_length_mm(mask_3d: np.ndarray, spacing_z_mm: float) -> float:
    """Compute craniocaudal length (mm) along the z-axis of the segmented spleen."""
    z_any = np.any(mask_3d > 0, axis=(0, 1))
    z_indices = np.where(z_any)[0]
    if len(z_indices) == 0:
        return 0.0
    span_slices = int(z_indices[-1] - z_indices[0] + 1)
    return float(span_slices * spacing_z_mm)


def execute_profile_on_volume(
    img_path: Path,
    profile_name: str,
    spacing: Tuple[float, float, float],
    affine: np.ndarray,
    gt_vol: Optional[np.ndarray],
) -> Dict[str, Any]:
    device = torch.device("cuda:0")
    profile = get_profile(profile_name)
    predictor, use_amp, extra_vram = get_cached_predictor(profile_name)
    pipeline = build_pipeline(profile)

    timings = run_single_timed_volume_pass(
        image_path=img_path,
        predictor=predictor,
        device=device,
        pipeline=pipeline,
        profile=profile,
        use_amp=use_amp,
        extra_vram_mb=extra_vram,
    )
    inf_out = infer_volume(
        image_path=img_path,
        predictor=predictor,
        profile=profile,
        device=device,
        use_amp=use_amp,
    )
    pred_mask = inf_out["pred_mask_3d"].astype(np.uint8)
    voxels, vol_cm3 = compute_volume_cm3(pred_mask, spacing, affine=affine)
    cc_mm = compute_craniocaudal_length_mm(pred_mask, spacing[2])

    seg_metrics = None
    if gt_vol is not None and gt_vol.shape == pred_mask.shape:
        seg_metrics = compute_all_segmentation_metrics(pred_mask, gt_vol, spacing, affine=affine)

    return {
        "backend": profile_name,
        "label": profile.label,
        "sw_network_ms": timings["sw_network"],
        "preprocess_ms": timings["preprocess"],
        "postprocess_ms": timings["postprocess"],
        "e2e_total_ms": timings["e2e_total"],
        "peak_vram_mb": timings["peak_vram_mb"],
        "voxels": voxels,
        "volume_cm3": vol_cm3,
        "cc_length_mm": cc_mm,
        "metrics": seg_metrics,
        "pred_mask": pred_mask,
    }


def mask_nifti_bytes(mask: np.ndarray, affine: np.ndarray) -> bytes:
    return gzip.compress(nib.Nifti1Image(mask.astype(np.uint8), affine).to_bytes(), compresslevel=1)


def peak_slice(mask: np.ndarray) -> int:
    """Axial slice holding the most segmented voxels."""
    return int(np.argmax(np.count_nonzero(mask > 0, axis=(0, 1))))


def show_plane(rgb: np.ndarray, title: str, ch_x: int, ch_y: int, aspect_ratio: float, interactive: bool, key: str) -> None:
    if interactive:
        height, width = rgb.shape[:2]
        fig = go.Figure(go.Image(z=rgb))
        fig.add_vline(ch_x, line=dict(color="#00E5FF", width=1, dash="dash"))
        fig.add_hline(ch_y, line=dict(color="#00E5FF", width=1, dash="dash"))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=aspect_ratio)
        fig.update_layout(
            title=dict(text=title, font=dict(size=13)),
            margin=dict(l=0, r=0, t=32, b=0),
            height=int(420 * min(max(height * aspect_ratio / max(width, 1), 0.6), 1.4)),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch", theme=None, key=key)
        return
    fig, ax = plt.subplots(figsize=(4.5, 4.5), facecolor="#0e1117")
    ax.imshow(rgb, aspect=aspect_ratio)
    ax.axvline(ch_x, color="#00E5FF", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.axhline(ch_y, color="#00E5FF", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.set_title(title, color="white", fontsize=10.5)
    ax.axis("off")
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def main() -> None:
    st.title("CT workstation")
    st.caption("Inspect any case, run any profile, and compare two profiles side by side.")

    available_backends = list(PROFILES)

    with st.sidebar:
        st.header("⚙️ Execution Backend")
        backend_choice = st.selectbox(
            "Primary Inference Backend",
            options=available_backends,
            index=available_backends.index(SHIPPING),
        )
        enable_comparison = st.checkbox("Enable Cross-Backend Comparison Mode", value=False)
        compare_backend = None
        if enable_comparison:
            compare_backend = st.selectbox(
                "Reference Comparison Backend",
                options=available_backends,
                index=available_backends.index(REFERENCE),
            )

        st.divider()
        st.header("📂 Held-Out Case / NIfTI Input")
        input_mode = st.radio(
            "Volume Source",
            options=["Held-Out Validation Split (MSD Task09)", "Upload NIfTI (.nii / .nii.gz)"],
            index=0,
        )

        selected_img_path: Optional[Path] = None
        selected_lbl_path: Optional[Path] = None
        selected_case_id = "uploaded_ct"

        if input_mode == "Held-Out Validation Split (MSD Task09)" and HELDOUT_JSON_PATH.exists():
            with open(HELDOUT_JSON_PATH, "r", encoding="utf-8") as f:
                heldout_cases = json.load(f)["cases"]
            case_map = {c["case_id"]: c for c in heldout_cases}
            selected_case_id = st.selectbox("Select Held-Out Case", options=list(case_map.keys()), index=2)
            c_info = case_map[selected_case_id]
            selected_img_path = MEDTRT_ROOT / c_info["image"]
            selected_lbl_path = MEDTRT_ROOT / c_info["label"]
        else:
            up_file = st.file_uploader("Upload NIfTI Volume", type=["nii", "gz"])
            if up_file is not None:
                tmp_dir = Path(tempfile.gettempdir()) / "medtrt_uploads"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                tmp_path = tmp_dir / up_file.name
                tmp_path.write_bytes(up_file.getvalue())
                selected_img_path = tmp_path

        st.divider()
        st.header("🖥️ Radiologic Windowing (Display Only)")
        preset = st.selectbox(
            "Window Preset",
            options=[
                "Abdominal Soft Tissue (W:400, L:40)",
                "Bone (W:1800, L:400)",
                "Lung Parenchyma (W:1500, L:-600)",
                "Custom Window",
            ],
            index=0,
        )
        if preset.startswith("Abdominal"):
            win_w, win_l = 400.0, 40.0
        elif preset.startswith("Bone"):
            win_w, win_l = 1800.0, 400.0
        elif preset.startswith("Lung"):
            win_w, win_l = 1500.0, -600.0
        else:
            win_w = float(st.slider("Window Width (W)", 50, 2500, 400))
            win_l = float(st.slider("Window Level (L)", -500, 1000, 40))

        mask_mode = st.radio(
            "Mask View Mode",
            options=[
                "Translucent Overlay",
                "1-Pixel Boundary Contour",
                "Difference / Disagreement Mask",
                "Hidden",
            ],
            index=1,
        )
        overlay_alpha = st.slider("Overlay Opacity", 0.1, 0.9, 0.45, 0.05)
        renderer = st.radio("Slice Renderer", options=["Fast", "Interactive (zoom and pan)"], index=0)

        gpu_busy = GPU_LOCK.exists()
        run_btn = st.button("🚀 Execute 3D Segmentation", type="primary", width="stretch", disabled=gpu_busy)
        if gpu_busy:
            st.caption("A GPU job is running. Inference is disabled so it cannot disturb the timings.")

    if selected_img_path is None or not selected_img_path.exists():
        st.info("Select a held-out validation case or upload a NIfTI volume to begin.")
        return

    nifti_data = load_cached_nifti(
        str(selected_img_path),
        str(selected_lbl_path) if selected_lbl_path else None,
    )
    ct_vol = nifti_data["volume"]
    gt_vol = nifti_data["gt_mask"]
    nx, ny, nz = ct_vol.shape
    sx, sy, sz = nifti_data["spacing"]
    affine = nifti_data["affine"]
    axcodes = nifti_data["axcodes"]

    v1, v2, v3, v4 = st.columns([1.1, 1.5, 1.2, 0.7])
    v1.metric("Shape", f"{nx}×{ny}×{nz}", border=True)
    v2.metric("Spacing (mm)", f"{sx:.2f}×{sy:.2f}×{sz:.2f}", border=True)
    v3.metric("HU Range", f"{nifti_data['hu_min']:.0f} to {nifti_data['hu_max']:.0f}", border=True)
    v4.metric("Orientation", axcodes, border=True)

    state_key = f"{selected_img_path}_{backend_choice}_{compare_backend}"
    if run_btn:
        tel = execute_profile_on_volume(
            img_path=selected_img_path,
            profile_name=backend_choice,
            spacing=(sx, sy, sz),
            affine=affine,
            gt_vol=gt_vol,
        )
        cmp_tel = None
        if enable_comparison and compare_backend:
            cmp_tel = execute_profile_on_volume(
                img_path=selected_img_path,
                profile_name=compare_backend,
                spacing=(sx, sy, sz),
                affine=affine,
                gt_vol=gt_vol,
            )

        fg = np.argwhere(tel["pred_mask"] > 0)
        if len(fg) > 0:
            cx, cy, cz = [int(np.median(fg[:, i])) for i in range(3)]
        else:
            cx, cy, cz = nx // 2, ny // 2, nz // 2

        st.session_state["last_run_key"] = state_key
        st.session_state["telemetry"] = tel
        st.session_state["cmp_telemetry"] = cmp_tel
        st.session_state["default_xyz"] = (cx, cy, cz)
        st.session_state["mask_bytes"] = mask_nifti_bytes(tel["pred_mask"], affine)
        st.session_state["slider_generation"] = st.session_state.get("slider_generation", 0) + 1

    if st.session_state.get("last_run_key") != state_key:
        st.info("Press **Execute 3D Segmentation** in the sidebar to run this case and profile.")
        return

    tel = st.session_state["telemetry"]
    cmp_tel = st.session_state.get("cmp_telemetry")
    pred_mask = tel["pred_mask"] if mask_mode != "Hidden" else None
    cmp_mask = cmp_tel["pred_mask"] if cmp_tel is not None else None
    def_x, def_y, def_z = st.session_state["default_xyz"]

    # Telemetry HUD — only display Dice/HD95 when ground truth exists
    if tel["metrics"] is not None:
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("Backend", f"{tel['backend']}")
        c2.metric("Network Latency", f"{tel['sw_network_ms']:.1f} ms")
        c3.metric("End-to-End Latency", f"{tel['e2e_total_ms']:.1f} ms")
        c4.metric("Spleen Volume", f"{tel['volume_cm3']:.2f} cm³", f"GT: {tel['metrics']['v_gt_cm3']:.2f} cm³")
        c5.metric("Foreground Dice", f"{tel['metrics']['dice']:.4f}", f"HD95: {tel['metrics']['hd95_mm']:.2f} mm")
        c6.metric("Orientation & Span", f"{axcodes} ({tel['cc_length_mm']:.1f} mm)")
        c7.metric("Peak VRAM", f"{tel['peak_vram_mb']:.1f} MB")
    else:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Backend", f"{tel['backend']}")
        c2.metric("Network Latency", f"{tel['sw_network_ms']:.1f} ms")
        c3.metric("End-to-End Latency", f"{tel['e2e_total_ms']:.1f} ms")
        c4.metric("Spleen Volume", f"{tel['volume_cm3']:.2f} cm³")
        c5.metric("Orientation & Span", f"{axcodes} ({tel['cc_length_mm']:.1f} mm)")
        c6.metric("Peak VRAM", f"{tel['peak_vram_mb']:.1f} MB")
        st.caption("ℹ️ Ground-truth mask unavailable for uploaded volume — accuracy metrics (Dice, HD95) suppressed.")

    if cmp_tel is not None:
        dv = tel["volume_cm3"] - cmp_tel["volume_cm3"]
        disagree_vox = int(np.count_nonzero(tel["pred_mask"] != cmp_tel["pred_mask"]))
        speedup = cmp_tel["e2e_total_ms"] / max(tel["e2e_total_ms"], 1e-6)
        st.info(
            f"**Cross-Backend Comparison (`{tel['backend']}` vs. `{cmp_tel['backend']}`):** "
            f"E2E Speedup = **`{speedup:.2f}×`** (`{tel['e2e_total_ms']:.1f} ms` vs. `{cmp_tel['e2e_total_ms']:.1f} ms`) · "
            f"Volume Difference $\\Delta V$ = **`{dv:+.3f} cm³`** (`{tel['volume_cm3']:.2f}` vs. `{cmp_tel['volume_cm3']:.2f} cm³`) · "
            f"Disagreeing Voxels = **`{disagree_vox:,}`** (`{100.0 * (1.0 - disagree_vox / tel['pred_mask'].size):.4f}%` agreement)"
        )

    a1, a2, a3 = st.columns([1, 1, 2])
    with a1:
        if st.button("Jump to Peak Spleen Slice", width="stretch"):
            st.session_state["default_xyz"] = (def_x, def_y, peak_slice(tel["pred_mask"]))
            st.session_state["slider_generation"] = st.session_state.get("slider_generation", 0) + 1
            st.rerun()
    with a2:
        st.download_button(
            "Download Mask (.nii.gz)",
            data=st.session_state["mask_bytes"],
            file_name=f"{selected_case_id}_{tel['backend']}_mask.nii.gz",
            mime="application/gzip",
            width="stretch",
        )
    with a3:
        st.caption(f"Segmented voxels: {tel['voxels']:,}")

    st.divider()

    # Tri-planar synchronized slice sliders
    generation = st.session_state.get("slider_generation", 0)
    s1, s2, s3 = st.columns(3)
    with s1:
        z_idx = st.slider("Axial Slice (Z — Inferior to Superior)", 0, nz - 1, def_z, key=f"axial_{generation}")
    with s2:
        y_idx = st.slider("Coronal Slice (Y — Posterior to Anterior)", 0, ny - 1, def_y, key=f"coronal_{generation}")
    with s3:
        x_idx = st.slider("Sagittal Slice (X — Left to Right)", 0, nx - 1, def_x, key=f"sagittal_{generation}")

    p1, p2, p3 = st.columns(3)
    planes = [
        (
            p1,
            f"Axial (Z = {z_idx}) [{axcodes}]",
            ct_vol[:, :, z_idx],
            pred_mask[:, :, z_idx] if pred_mask is not None else None,
            gt_vol[:, :, z_idx] if gt_vol is not None else None,
            cmp_mask[:, :, z_idx] if cmp_mask is not None else None,
            x_idx,
            ny - 1 - y_idx,
            sy / max(sx, 1e-6),
        ),
        (
            p2,
            f"Coronal (Y = {y_idx}) [{axcodes}]",
            ct_vol[:, y_idx, :],
            pred_mask[:, y_idx, :] if pred_mask is not None else None,
            gt_vol[:, y_idx, :] if gt_vol is not None else None,
            cmp_mask[:, y_idx, :] if cmp_mask is not None else None,
            x_idx,
            nz - 1 - z_idx,
            sz / max(sx, 1e-6),
        ),
        (
            p3,
            f"Sagittal (X = {x_idx}) [{axcodes}]",
            ct_vol[x_idx, :, :],
            pred_mask[x_idx, :, :] if pred_mask is not None else None,
            gt_vol[x_idx, :] if (gt_vol is not None and gt_vol.ndim == 2) else (gt_vol[x_idx, :, :] if gt_vol is not None else None),
            cmp_mask[x_idx, :, :] if cmp_mask is not None else None,
            y_idx,
            nz - 1 - z_idx,
            sz / max(sy, 1e-6),
        ),
    ]

    for col, title, ct_sl, pr_sl, gt_sl, cm_sl, ch_x, ch_y, aspect_ratio in planes:
        with col:
            rgb = render_plane_rgb(
                ct_2d=ct_sl,
                pred_2d=pr_sl,
                gt_2d=gt_sl,
                cmp_2d=cm_sl,
                width=win_w,
                level=win_l,
                mask_mode=mask_mode if mask_mode != "Hidden" else "Translucent Overlay",
                alpha=overlay_alpha,
            )
            show_plane(rgb, title, ch_x, ch_y, aspect_ratio, renderer.startswith("Interactive"), f"plane_{title[:3]}")


main()
