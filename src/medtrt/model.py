"""
MONAI Bundle Network Loader (`src/medtrt/model.py`).
"""

from __future__ import annotations

import torch
from monai.bundle import ConfigParser
from monai.networks.layers import Norm
from monai.networks.nets import UNet

from medtrt.data import BUNDLE_PATH, ensure_bundle


def load_spleen_unet(device: str | torch.device = "cuda:0") -> torch.nn.Module:
    """
    Load the pretrained 3D Residual UNet from `bundles/spleen_ct_segmentation`.
    Enforces strict GPU execution (no silent CPU fallback per Ground Rule 6).
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available. Stopping per Ground Rule 6 (no silent CPU fallback).")

    target_device = torch.device(device)
    bundle_dir = ensure_bundle()
    inference_json = bundle_dir / "configs" / "inference.json"
    model_pt = bundle_dir / "models" / "model.pt"

    try:
        parser = ConfigParser()
        parser.read_config(str(inference_json))
        model = parser.get_parsed_content("network_def")
    except Exception:
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )

    state_dict = torch.load(str(model_pt), map_location=target_device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(target_device)
    model.eval()
    return model
