"""
Phase 7 Minimal Tri-Planar Streamlit Workstation (`app/viewer.py`).

Features:
- Header banner: "Research and Verification Prototype Only — Not for Clinical Diagnostic Use."
- Load held-out cases from `./data/Task09_Spleen/` (`results/heldout_cases.json`) or uploaded NIfTI.
- Synchronized Axial, Coronal, and Sagittal slice sliders with crosshair markers.
- Radiologic windowing presets (Abdominal Soft Tissue: W=400, L=40; Bone: W=1800, L=400; Custom)
  applied strictly to screen render buffers, never to model inputs.
- Mask view toggle: Translucent alpha overlay vs. 1-pixel boundary contour.
- Telemetry HUD: Backend selected, pure network latency (ms), E2E latency (ms), spleen volume (cm³),
  craniocaudal span (mm), and peak allocated VRAM (MB).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import streamlit as st
import torch
from scipy.ndimage import binary_erosion

MEDTRT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = MEDTRT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medtrt.bench import build_split_load_and_preprocess, run_single_timed_volume_pass
from medtrt.data import HELDOUT_JSON_PATH
from medtrt.evaluate import compute_foreground_dice, compute_volume_cm3
from medtrt.infer import build_inference_postprocessing, infer_volume
from medtrt.model import load_spleen_unet
from medtrt.trt_runner import TRTRunner

st.set_page_config(
    page_title="MedTRT Tri-Planar Workstation | RTX 5080 Blackwell",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Initializing GPU Backends (PyTorch & TensorRT 10.x FP16)...")
def get_cached_backends() -> Dict[str, Any]:
    device = torch.device("cuda:0")
    pt_model = load_spleen_unet(device)
    trt_runner = TRTRunner(device=device)
    return {
        "device": device,
        "pt_model": pt_model,
        "trt_runner": trt_runner,
    }


@st.cache_data(show_spinner="Loading NIfTI CT Volume...")
def load_cached_nifti(img_path: str, lbl_path: Optional[str] = None) -> Dict[str, Any]:
    nii = nib.load(img_path)
    vol = nii.get_fdata(dtype=np.float32)
    if vol.ndim == 4:
        vol = vol[..., 0]
    zooms = nii.header.get_zooms()[:3]
    spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))

    gt_vol = None
    if lbl_path and Path(lbl_path).exists():
        gt_vol = np.asarray(nib.load(lbl_path).dataobj).astype(np.uint8)

    return {
        "volume": vol,
        "gt_mask": gt_vol,
        "shape": vol.shape,
        "spacing": spacing,
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

    return np.clip(rgb, 0, 255).astype(np.uint8)


def compute_craniocaudal_length_mm(mask_3d: np.ndarray, spacing_z_mm: float) -> float:
    """Compute craniocaudal length (mm) along the z-axis of the segmented spleen."""
    z_any = np.any(mask_3d > 0, axis=(0, 1))
    z_indices = np.where(z_any)[0]
    if len(z_indices) == 0:
        return 0.0
    span_slices = int(z_indices[-1] - z_indices[0] + 1)
    return float(span_slices * spacing_z_mm)


def main() -> None:
    st.warning(
        "⚠️ **Research and Verification Prototype Only — Not for Clinical Diagnostic Use.**",
        icon="⚕️",
    )
    st.title("🫁 MedTRT Tri-Planar CT Segmentation & Telemetry Workstation")

    backends = get_cached_backends()

    with st.sidebar:
        st.header("⚙️ Execution Backend")
        backend_choice = st.selectbox(
            "Select GPU Inference Backend",
            options=[
                "trt_fp16_optimized",
                "trt_fp16",
                "pytorch_amp",
                "pytorch_fp32",
                "pytorch_fp32_strict",
            ],
            index=0,
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
                "Custom Window",
            ],
            index=0,
        )
        if preset.startswith("Abdominal"):
            win_w, win_l = 400.0, 40.0
        elif preset.startswith("Bone"):
            win_w, win_l = 1800.0, 400.0
        else:
            win_w = float(st.slider("Window Width (W)", 50, 2500, 400))
            win_l = float(st.slider("Window Level (L)", -500, 1000, 40))

        mask_mode = st.radio(
            "Mask View Mode",
            options=["Translucent Overlay", "1-Pixel Boundary Contour", "Hidden"],
            index=0,
        )
        overlay_alpha = st.slider("Overlay Opacity", 0.1, 0.9, 0.45, 0.05)

        run_btn = st.button("🚀 Execute 3D Segmentation", type="primary", use_container_width=True)

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

    state_key = f"{selected_img_path}_{backend_choice}"
    if run_btn or st.session_state.get("last_run_key") != state_key:
        device = backends["device"]
        use_amp = backend_choice == "pytorch_amp"
        gpu_tf = backend_choice == "trt_fp16_optimized"
        predictor = (
            backends["trt_runner"]
            if backend_choice.startswith("trt")
            else backends["pt_model"]
        )
        extra_vram = backends["trt_runner"].trt_vram_mb if backend_choice.startswith("trt") else 0.0

        load_tf, pre_tf, _ = build_split_load_and_preprocess()
        post_tf = build_inference_postprocessing(pre_tf)

        timings = run_single_timed_volume_pass(
            image_path=selected_img_path,
            predictor=predictor,
            device=device,
            load_tf=load_tf,
            pre_tf=pre_tf,
            post_tf=post_tf,
            sw_batch_size=4,
            overlap=0.5,
            use_amp=use_amp,
            extra_vram_mb=extra_vram,
            gpu_transforms=gpu_tf,
        )
        inf_out = infer_volume(
            image_path=selected_img_path,
            predictor=predictor,
            device=device,
            sw_batch_size=4,
            overlap=0.5,
            use_amp=use_amp,
            gpu_transforms=gpu_tf,
        )
        pred_mask = inf_out["pred_mask_3d"].astype(np.uint8)
        _, vol_cm3 = compute_volume_cm3(pred_mask, (sx, sy, sz))
        cc_mm = compute_craniocaudal_length_mm(pred_mask, sz)
        dice_val = compute_foreground_dice(pred_mask, gt_vol) if gt_vol is not None else None

        # Default crosshairs at spleen centroid
        fg = np.argwhere(pred_mask > 0)
        if len(fg) > 0:
            cx, cy, cz = [int(np.median(fg[:, i])) for i in range(3)]
        else:
            cx, cy, cz = nx // 2, ny // 2, nz // 2

        st.session_state["last_run_key"] = state_key
        st.session_state["telemetry"] = {
            "backend": backend_choice,
            "sw_network_ms": timings["sw_network"],
            "e2e_total_ms": timings["e2e_total"],
            "peak_vram_mb": timings["peak_vram_mb"],
            "volume_cm3": vol_cm3,
            "cc_length_mm": cc_mm,
            "dice": dice_val,
            "pred_mask": pred_mask,
            "default_xyz": (cx, cy, cz),
        }

    tel = st.session_state["telemetry"]
    pred_mask = tel["pred_mask"] if mask_mode != "Hidden" else None
    def_x, def_y, def_z = tel["default_xyz"]

    # Telemetry HUD
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Backend", f"{tel['backend']}")
    c2.metric("Pure Network Latency", f"{tel['sw_network_ms']:.1f} ms")
    c3.metric("End-to-End Latency", f"{tel['e2e_total_ms']:.1f} ms")
    c4.metric("Spleen Volume", f"{tel['volume_cm3']:.2f} cm³")
    c5.metric("Craniocaudal Span", f"{tel['cc_length_mm']:.1f} mm")
    c6.metric("Peak VRAM", f"{tel['peak_vram_mb']:.1f} MB")

    st.divider()

    # Tri-planar synchronized slice sliders
    s1, s2, s3 = st.columns(3)
    with s1:
        z_idx = st.slider("Axial Slice (Z)", 0, nz - 1, def_z)
    with s2:
        y_idx = st.slider("Coronal Slice (Y)", 0, ny - 1, def_y)
    with s3:
        x_idx = st.slider("Sagittal Slice (X)", 0, nx - 1, def_x)

    p1, p2, p3 = st.columns(3)
    planes = [
        (
            p1,
            f"Axial (Z = {z_idx})",
            ct_vol[:, :, z_idx],
            pred_mask[:, :, z_idx] if pred_mask is not None else None,
            gt_vol[:, :, z_idx] if gt_vol is not None else None,
            x_idx,
            ny - 1 - y_idx,
        ),
        (
            p2,
            f"Coronal (Y = {y_idx})",
            ct_vol[:, y_idx, :],
            pred_mask[:, y_idx, :] if pred_mask is not None else None,
            gt_vol[:, y_idx, :] if gt_vol is not None else None,
            x_idx,
            nz - 1 - z_idx,
        ),
        (
            p3,
            f"Sagittal (X = {x_idx})",
            ct_vol[x_idx, :, :],
            pred_mask[x_idx, :, :] if pred_mask is not None else None,
            gt_vol[x_idx, :, :] if gt_vol is not None else None,
            y_idx,
            nz - 1 - z_idx,
        ),
    ]

    for col, title, ct_sl, pr_sl, gt_sl, ch_x, ch_y in planes:
        with col:
            rgb = render_plane_rgb(
                ct_2d=ct_sl,
                pred_2d=pr_sl,
                gt_2d=gt_sl,
                width=win_w,
                level=win_l,
                mask_mode=mask_mode if mask_mode != "Hidden" else "Translucent Overlay",
                alpha=overlay_alpha,
            )
            fig, ax = plt.subplots(figsize=(4.5, 4.5), facecolor="#0e1117")
            ax.imshow(rgb, aspect="auto")
            # Synchronized crosshair lines
            ax.axvline(ch_x, color="#00E5FF", linestyle="--", linewidth=0.8, alpha=0.8)
            ax.axhline(ch_y, color="#00E5FF", linestyle="--", linewidth=0.8, alpha=0.8)
            ax.set_title(title, color="white", fontsize=11)
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


if __name__ == "__main__":
    main()
