"""
Execution Profiles (`src/medtrt/profiles.py`).

A `Profile` is the single definition of how a volume is processed. Benchmarking, evaluation,
the sweep, and the viewer all build their pipeline from a profile, so a latency number and a
Dice number never describe two different pipelines.

Supported transform modes (`transforms`):
- `"cpu"`: Host MONAI transforms (`Orientationd`, `Spacingd`, `ScaleIntensityRanged` on CPU, `Invertd` on CPU).
- `"gpu"`: PyTorch/MONAI GPU transforms (`Spacingd` and `Invertd` via `torch.nn.functional.grid_sample` on `cuda:0`).
- `"cuda"`: Custom CUDA C++ fused kernels (`cuda/medical_kernels.cu` via NVRTC + CUDA Driver API),
  eliminating MONAI's 1.0-2.1 GB 3D coordinate meshgrid allocations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Tuple

import torch

from medtrt.export import ENGINE_FP16_UNFOLDED_PATH, ENGINE_FP32_PATH, ENGINE_PATH


@dataclass(frozen=True)
class Profile:
    name: str
    label: str
    engine: str  # "pytorch" | "trt"
    precision: str  # "fp32" | "tf32" | "amp" | "fp16" | "fp16_unfolded"
    io: str  # "gzip" | "mmap"
    transforms: str  # "cpu" | "gpu" | "cuda"
    inverse: str  # "trilinear" (bundle-faithful) | "nearest"
    overlap: float
    sw_batch_size: int

    @property
    def gpu_path(self) -> bool:
        return self.transforms in ("gpu", "cuda")

    @property
    def custom_cuda_path(self) -> bool:
        return self.transforms == "cuda"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


_HOST = dict(io="gzip", transforms="cpu", inverse="trilinear", overlap=0.5, sw_batch_size=4)
_GPU = dict(io="mmap", transforms="gpu", inverse="trilinear", overlap=0.5, sw_batch_size=4)
_CUDA_B4 = dict(io="mmap", transforms="cuda", inverse="trilinear", overlap=0.5, sw_batch_size=4)
_CUDA_FAST = dict(io="mmap", transforms="cuda", inverse="trilinear", overlap=0.25, sw_batch_size=8)

PROFILES: Dict[str, Profile] = {
    p.name: p
    for p in (
        Profile("pytorch_fp32_strict", "PyTorch FP32 (Strict IEEE, Host Path)", "pytorch", "fp32", **_HOST),
        Profile("pytorch_fp32", "PyTorch FP32 (Default TF32, Host Path)", "pytorch", "tf32", **_HOST),
        Profile("pytorch_tf32", "PyTorch TF32 (Host Path)", "pytorch", "tf32", **_HOST),
        Profile("pytorch_amp", "PyTorch AMP (FP16 Autocast, Host Path)", "pytorch", "amp", **_HOST),
        Profile("pytorch_fp16", "PyTorch FP16 (Pure Half, Host Path)", "pytorch", "fp16", **_HOST),
        Profile("trt_fp32", "TensorRT 10.x FP32 (Zero-Copy, Host Path)", "trt", "fp32", **_HOST),
        Profile("trt_fp16_unfolded", "TensorRT 10.x FP16 (Unfolded BN, Host Path)", "trt", "fp16_unfolded", **_HOST),
        Profile("trt_fp16", "TensorRT 10.x FP16 (Zero-Copy, Host Path)", "trt", "fp16", **_HOST),
        Profile("pytorch_tf32_gpu", "PyTorch TF32 (MONAI GPU Path)", "pytorch", "tf32", **_GPU),
        Profile("trt_fp16_gpu", "TensorRT 10.x FP16 (MONAI GPU Path, overlap=0.5)", "trt", "fp16", **_GPU),
        Profile("trt_fp16_cuda", "TensorRT 10.x FP16 (Custom CUDA Kernels, overlap=0.5)", "trt", "fp16", **_CUDA_B4),
        Profile(
            "trt_fp16_optimized",
            "TensorRT 10.x FP16 (Custom CUDA Kernels + Pareto Windows)",
            "trt",
            "fp16",
            **_CUDA_FAST,
        ),
        Profile(
            "trt_fp16_nn",
            "TensorRT 10.x FP16 (Fast Windows, Nearest-Label Inverse)",
            "trt",
            "fp16",
            io="mmap",
            transforms="gpu",
            inverse="nearest",
            overlap=0.25,
            sw_batch_size=8,
        ),
    )
}

REFERENCE = "pytorch_fp32_strict"  # numerical reference for every agreement statistic
HOST_BASELINE = "pytorch_fp32"  # PyTorch default on Ampere/Blackwell, host path
GPU_BASELINE = "pytorch_tf32_gpu"  # MONAI GPU pipeline without TensorRT
SHIPPING = "trt_fp16_optimized"  # recommended fast profile (Custom CUDA + TRT FP16 + Pareto windows)
LIKE_FOR_LIKE = "trt_fp16_cuda"  # identical windows and math to the bundle, with custom CUDA & TRT FP16


def get_profile(name: str) -> Profile:
    if name not in PROFILES:
        raise ValueError(f"Unknown profile {name!r}. Valid profiles: {sorted(PROFILES)}")
    return PROFILES[name]


def apply_precision(profile: Profile) -> None:
    """Set process-wide matmul and cuDNN precision for `profile`."""
    allow_tf32 = profile.precision != "fp32"
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def build_predictor(
    profile: Profile, device: torch.device
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], bool, float]:
    """Return `(predictor, use_amp, extra_vram_mb)` for `profile`."""
    apply_precision(profile)
    if profile.engine == "trt":
        from medtrt.trt_runner import TRTRunner

        if profile.precision == "fp32":
            eng_path = ENGINE_FP32_PATH
        elif profile.precision == "fp16_unfolded":
            eng_path = ENGINE_FP16_UNFOLDED_PATH
        else:
            eng_path = ENGINE_PATH
        runner = TRTRunner(engine_path=eng_path, device=device)
        return runner, False, runner.trt_vram_mb

    from medtrt.model import load_spleen_unet

    if profile.precision == "fp16":
        model_fp16 = load_spleen_unet(device, fold_bn=True, dtype=torch.float16)

        def _fp16_pred(x: torch.Tensor) -> torch.Tensor:
            return model_fp16(x.half()).float()

        return _fp16_pred, False, 0.0

    # For strict FP32 reference, keep original unfolded weights (or folded; both match to 1.1e-5)
    fold_bn = profile.precision in ("amp", "fp16")
    return load_spleen_unet(device, fold_bn=fold_bn), profile.precision == "amp", 0.0
