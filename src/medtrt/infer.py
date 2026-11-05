"""
Unified 3D Sliding-Window Inference Wrapper (`src/medtrt/infer.py`).

Strictly implements `bundles/spleen_ct_segmentation/configs/inference.json` transforms:
- `LoadImaged(keys="image")`
- `EnsureChannelFirstd(keys="image")`
- `Orientationd(keys="image", axcodes="RAS")`
- `Spacingd(keys="image", pixdim=[1.5, 1.5, 2.0], mode="bilinear")`
- `ScaleIntensityRanged(keys="image", a_min=-57, a_max=164, b_min=0, b_max=1, clip=True)`
- `EnsureTyped(keys="image")`
And postprocessing:
- `Activationsd(keys="pred", softmax=True)`
- `Invertd(keys="pred", transform=preprocessing, orig_keys="image", nearest_interp=False, to_tensor=True)`
- `AsDiscreted(keys="pred", argmax=True)`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import torch
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)


def build_inference_preprocessing() -> Compose:
    """Exact preprocessing pipeline defined in `configs/inference.json`."""
    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57.0,
                a_max=164.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            EnsureTyped(keys=["image"]),
        ]
    )


def build_inference_postprocessing(preprocessing: Compose) -> Compose:
    """Exact postprocessing pipeline defined in `configs/inference.json`."""
    return Compose(
        [
            Activationsd(keys=["pred"], softmax=True),
            Invertd(
                keys=["pred"],
                transform=preprocessing,
                orig_keys="image",
                nearest_interp=False,
                to_tensor=True,
            ),
            AsDiscreted(keys=["pred"], argmax=True),
        ]
    )


def build_optimized_gpu_postprocessing(preprocessing: Compose) -> Compose:
    """
    Phase 5 Targeted Optimization:
    Perform GPU argmax discretization (`AsDiscreted(argmax=True)`) on the 2-channel logits
    BEFORE spatial inversion (`Invertd(nearest_interp=True)`) on CUDA (`cuda:0`), eliminating
    2-channel host CPU trilinear interpolation overhead.
    """
    return Compose(
        [
            AsDiscreted(keys=["pred"], argmax=True),
            Invertd(
                keys=["pred"],
                transform=preprocessing,
                orig_keys="image",
                nearest_interp=True,
                to_tensor=True,
            ),
        ]
    )


def build_sliding_window_inferer(
    roi_size: Tuple[int, int, int] = (96, 96, 96),
    sw_batch_size: int = 4,
    overlap: float = 0.5,
) -> SlidingWindowInferer:
    """Create `SlidingWindowInferer` matching `configs/inference.json`."""
    return SlidingWindowInferer(
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
    )


def infer_volume(
    image_path: str | Path,
    predictor: Callable[[torch.Tensor], torch.Tensor],
    device: str | torch.device = "cuda:0",
    roi_size: Tuple[int, int, int] = (96, 96, 96),
    sw_batch_size: int = 4,
    overlap: float = 0.5,
    use_amp: bool = False,
    gpu_transforms: bool = False,
) -> Dict[str, Any]:
    """
    Execute full-volume preprocessing, sliding-window inference, and inverse spatial transform
    back to original NIfTI coordinate space.
    When `gpu_transforms=True` (Phase 5 Targeted Optimization), spatial resampling (`Spacingd`),
    intensity scaling (`ScaleIntensityRanged`), softmax (`Activationsd`), and 3D inverse
    resampling (`Invertd`) execute on CUDA (`cuda:0`) instead of host CPU.
    """
    target_device = torch.device(device)
    if gpu_transforms:
        load_tf = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                EnsureTyped(keys=["image"], device=target_device),
            ]
        )
        pre_tf = Compose(
            [
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-57.0,
                    a_max=164.0,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
            ]
        )
        post_tf = build_optimized_gpu_postprocessing(pre_tf)
        data: Dict[str, Any] = load_tf({"image": str(image_path)})
        data = pre_tf(data)
        inputs = data["image"].unsqueeze(0)
    else:
        pre_tf = build_inference_preprocessing()
        post_tf = build_inference_postprocessing(pre_tf)
        data = pre_tf({"image": str(image_path)})
        inputs = data["image"].unsqueeze(0).to(target_device)

    inferer = build_sliding_window_inferer(
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
    )

    with torch.no_grad():
        if use_amp:
            with torch.autocast("cuda", dtype=torch.float16):
                logits = inferer(inputs=inputs, network=predictor)
        else:
            logits = inferer(inputs=inputs, network=predictor)

    if gpu_transforms:
        data["pred"] = logits[0].detach().float()
        out_data = post_tf(data)
        pred_mask_3d = out_data["pred"].squeeze(0).cpu().numpy()
    else:
        data["pred"] = logits[0].detach().float().cpu()
        out_data = post_tf(data)
        pred_mask_3d = out_data["pred"].squeeze(0).cpu().numpy()

    return {
        "pred_mask_3d": pred_mask_3d,
        "logits_resampled": logits[0].detach().cpu(),
        "preprocessed_shape": tuple(data["image"].shape),
    }
