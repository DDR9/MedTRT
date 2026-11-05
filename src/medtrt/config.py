"""
Central Configuration & Phase 0 Hardware/Toolchain Audit (`src/medtrt/config.py`).
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import monai
import torch

MEDTRT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = MEDTRT_ROOT / "results"
DATA_DIR = MEDTRT_ROOT / "data"
BUNDLES_DIR = MEDTRT_ROOT / "bundles"
ENGINES_DIR = MEDTRT_ROOT / "engines"
FIGURES_DIR = MEDTRT_ROOT / "docs" / "figures"


@dataclass
class MedTRTConfig:
    project_root: str = str(MEDTRT_ROOT)
    roi_size: Tuple[int, int, int] = (96, 96, 96)
    pixdim: Tuple[float, float, float] = (1.5, 1.5, 2.0)
    a_min: float = -57.0
    a_max: float = 164.0
    b_min: float = 0.0
    b_max: float = 1.0
    sw_batch_size_default: int = 2
    overlap_default: float = 0.5
    warmup_volumes: int = 3
    timed_repeats: int = 10


def _query_nvidia_driver() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            timeout=5,
        )
        return out.strip().splitlines()[0]
    except Exception:
        return "unknown"


def _query_tensorrt_status() -> Dict[str, Any]:
    try:
        import tensorrt as trt  # type: ignore

        return {
            "import_ok": True,
            "version": getattr(trt, "__version__", "unknown"),
            "error": None,
        }
    except Exception as exc:
        # Check installed wheel metadata if DLL policy blocks .pyd load
        version_meta = "not_installed"
        try:
            import importlib.metadata as md

            for pkg in ("tensorrt", "tensorrt-cu12", "tensorrt-cu13"):
                try:
                    version_meta = f"{pkg}=={md.version(pkg)}"
                    break
                except md.PackageNotFoundError:
                    continue
        except Exception:
            pass
        return {
            "import_ok": False,
            "version": version_meta,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_env_audit() -> Dict[str, Any]:
    """
    Execute Phase 0 Hardware & Toolchain Audit and write `results/env.json` + `results/STATE.json`.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cuda_available = bool(torch.cuda.is_available())
    gpu_name = "None"
    vram_gb = 0.0
    vram_mb = 0
    compute_capability = "N/A"
    arch_name = "Unknown"

    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        vram_gb = round(props.total_memory / (1024 ** 3), 2)
        vram_mb = int(props.total_memory // (1024 ** 2))
        compute_capability = f"{props.major}.{props.minor} (sm_{props.major}{props.minor})"
        if props.major == 12:
            arch_name = "NVIDIA Blackwell"
        elif props.major == 8:
            arch_name = "NVIDIA Ampere / Ada"
        elif props.major == 9:
            arch_name = "NVIDIA Hopper"

    cudnn_ver = torch.backends.cudnn.version() if cuda_available else None
    cudnn_str = (
        f"{cudnn_ver // 10000}.{(cudnn_ver % 10000) // 100}.{cudnn_ver % 100}"
        if cudnn_ver
        else "N/A"
    )
    trt_status = _query_tensorrt_status()

    env_data: Dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python_version": platform.python_version(),
        "cuda_available": cuda_available,
        "gpu_model": gpu_name,
        "architecture": arch_name,
        "vram_gb": vram_gb,
        "vram_mb": vram_mb,
        "compute_capability": compute_capability,
        "driver_version": _query_nvidia_driver(),
        "cuda_runtime_version": str(torch.version.cuda),
        "cudnn_version": cudnn_str,
        "pytorch_version": torch.__version__,
        "monai_version": monai.__version__,
        "tensorrt_import_ok": trt_status["import_ok"],
        "tensorrt_version": trt_status["version"],
        "tensorrt_error": trt_status["error"],
    }

    env_path = RESULTS_DIR / "env.json"
    with open(env_path, "w", encoding="utf-8") as f:
        json.dump(env_data, f, indent=2)

    state_data = {
        "current_phase": "PHASE_0",
        "phase_0_status": "PASS" if (cuda_available and trt_status["import_ok"]) else "BLOCKED_TRT_IMPORT" if cuda_available else "BLOCKED_NO_CUDA",
        "updated_at": env_data["timestamp"],
        "env_json": str(env_path.relative_to(MEDTRT_ROOT)),
        "notes": trt_status["error"] if not trt_status["import_ok"] else "All Phase 0 checks passed.",
    }
    state_path = RESULTS_DIR / "STATE.json"
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state_data, f, indent=2)

    print("=" * 84)
    print("MEDTRT PHASE 0 — HARDWARE & TOOLCHAIN AUDIT")
    print("=" * 84)
    print(f"{'Metric':<28} | {'Detected Value'}")
    print("-" * 84)
    print(f"{'OS / Platform':<28} | {env_data['os']}")
    print(f"{'Python Version':<28} | {env_data['python_version']}")
    print(f"{'GPU Model':<28} | {env_data['gpu_model']}")
    print(f"{'GPU Architecture':<28} | {env_data['architecture']}")
    print(f"{'VRAM Capacity':<28} | {env_data['vram_gb']} GB ({env_data['vram_mb']} MB)")
    print(f"{'Compute Capability':<28} | {env_data['compute_capability']}")
    print(f"{'NVIDIA Driver Version':<28} | {env_data['driver_version']}")
    print(f"{'CUDA Runtime (PyTorch)':<28} | {env_data['cuda_runtime_version']}")
    print(f"{'cuDNN Version':<28} | {env_data['cudnn_version']}")
    print(f"{'PyTorch Version':<28} | {env_data['pytorch_version']}")
    print(f"{'MONAI Version':<28} | {env_data['monai_version']}")
    print(f"{'TensorRT Version':<28} | {env_data['tensorrt_version']}")
    print(f"{'torch.cuda.is_available()':<28} | {env_data['cuda_available']}")
    print(f"{'import tensorrt':<28} | {'SUCCESS' if env_data['tensorrt_import_ok'] else 'FAILED (' + str(env_data['tensorrt_error']) + ')'}")
    print("=" * 84)
    print(f"Saved: {env_path}")
    print(f"Saved: {state_path}")
    return env_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedTRT Configuration & Environment Audit")
    parser.add_argument("--env-audit", action="store_true", help="Run Phase 0 environment audit")
    args = parser.parse_args()
    run_env_audit()
