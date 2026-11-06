"""
Phase 4 & 5 Dynamic ONNX Export & TensorRT 10.x Engine Compiler (`src/medtrt/export.py`).

Supports:
- Pre-export algebraic fusion of `ConvTranspose3d + BatchNorm3d` (`fold_bn=True`) to prevent
  IEEE-754 FP16 catastrophic cancellation in `model.2.0.adn.N`.
- Dynamic batch ONNX export (`B in [1..8]`, `opset_version=17`) with numerical verification
  against strict IEEE FP32 (`max_abs_diff < 1e-4`).
- TensorRT 10.x engine compilation for both:
  - FP16 (`engines/spleen_unet_fp16.engine`)
  - FP32 (`engines/spleen_unet_fp32.engine`)
  - Unfolded FP16 control (`engines/spleen_unet_fp16_unfolded.engine`) for ablation/precision studies.
"""

from __future__ import annotations

import argparse
import datetime
import inspect
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import onnx
import onnxruntime as ort
import tensorrt as trt
import torch

from medtrt.config import ENGINES_DIR, MEDTRT_ROOT, RESULTS_DIR
from medtrt.model import load_spleen_unet

ONNX_PATH = ENGINES_DIR / "spleen_unet_dynamic.onnx"
ONNX_UNFOLDED_PATH = ENGINES_DIR / "spleen_unet_dynamic_unfolded.onnx"
ENGINE_PATH = ENGINES_DIR / "spleen_unet_fp16.engine"
ENGINE_FP32_PATH = ENGINES_DIR / "spleen_unet_fp32.engine"
ENGINE_FP16_UNFOLDED_PATH = ENGINES_DIR / "spleen_unet_fp16_unfolded.engine"

MIN_SHAPE = (1, 1, 96, 96, 96)
OPT_SHAPE = (2, 1, 96, 96, 96)
MAX_SHAPE = (8, 1, 96, 96, 96)


def _rel(path: Path) -> str:
    """Repo-relative POSIX path, so build metadata never publishes a local username."""
    try:
        return Path(path).resolve().relative_to(MEDTRT_ROOT).as_posix()
    except ValueError:
        return Path(path).name


def export_dynamic_onnx(
    onnx_path: Path = ONNX_PATH,
    fold_bn: bool = True,
) -> Dict[str, Any]:
    """
    Export 3D UNet to ONNX with dynamic batch axis B and verify `max_abs_diff < 1e-4`
    against PyTorch strict IEEE FP32.
    """
    ENGINES_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")

    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    try:
        ref_model = load_spleen_unet(device, fold_bn=False)
        ref_model.eval()
        export_model = load_spleen_unet(device, fold_bn=fold_bn)
        export_model.eval()

        torch.manual_seed(1234)
        dummy_input = torch.randn(OPT_SHAPE, dtype=torch.float32, device=device)

        export_kwargs: Dict[str, Any] = {
            "opset_version": 17,
            "do_constant_folding": True,
            "input_names": ["input"],
            "output_names": ["output"],
            "dynamic_axes": {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        }
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            export_kwargs["dynamo"] = False

        t0 = time.perf_counter()
        with torch.no_grad():
            pt_out = ref_model(dummy_input).detach().cpu().numpy()
            torch.onnx.export(
                export_model,
                dummy_input,
                str(onnx_path),
                **export_kwargs,
            )
        export_time_s = time.perf_counter() - t0
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    ort_sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = ort_sess.run(["output"], {"input": dummy_input.detach().cpu().numpy()})[0]

    max_abs_diff = float(np.max(np.abs(pt_out - ort_out)))
    mean_abs_diff = float(np.mean(np.abs(pt_out - ort_out)))

    print("=" * 84)
    print(f"DYNAMIC ONNX EXPORT & VERIFICATION (fold_bn={fold_bn})")
    print("=" * 84)
    print(f"ONNX Path         : {onnx_path} ({onnx_path.stat().st_size / (1024**2):.2f} MB)")
    print(f"Opset Version     : 17 | Nodes: {len(onnx_model.graph.node)}")
    print(f"Dynamic Batch Axis: B in [1..8], patch=(B, 1, 96, 96, 96)")
    print(f"Export Duration   : {export_time_s:.2f} s")
    print(f"Max Abs Diff      : {max_abs_diff:.3e} (Requirement < 1e-4)")
    print(f"Mean Abs Diff     : {mean_abs_diff:.3e}")
    assert max_abs_diff < 1e-4, f"ONNX max_abs_diff {max_abs_diff} >= 1e-4!"
    print("[PASS] Dynamic ONNX export & numerical verification succeeded.")
    print("=" * 84)

    return {
        "onnx_path": _rel(onnx_path),
        "onnx_size_mb": round(onnx_path.stat().st_size / (1024 ** 2), 3),
        "opset_version": 17,
        "num_nodes": len(onnx_model.graph.node),
        "fold_conv_transpose_bn": fold_bn,
        "export_time_s": round(export_time_s, 3),
        "max_abs_diff_vs_pytorch_fp32": max_abs_diff,
        "mean_abs_diff_vs_pytorch_fp32": mean_abs_diff,
    }


def build_trt_engine(
    onnx_path: Path = ONNX_PATH,
    engine_path: Path = ENGINE_PATH,
    precision: str = "fp16",
    fold_bn: bool = True,
    workspace_gb: int = 4,
) -> Dict[str, Any]:
    """
    Compile a TensorRT 10.x engine (`fp16` or `fp32`) with dynamic batch profile:
        min=(1, 1, 96, 96, 96), opt=(2, 1, 96, 96, 96), max=(8, 1, 96, 96, 96)
    """
    onnx_info = export_dynamic_onnx(onnx_path, fold_bn=fold_bn)

    print("=" * 84)
    print(f"TENSORRT 10.x ENGINE COMPILATION ({precision.upper()}, fold_bn={fold_bn})")
    print("=" * 84)
    print(f"TensorRT Version  : {trt.__version__}")
    print(f"Profile Shapes    : min={MIN_SHAPE}, opt={OPT_SHAPE}, max={MAX_SHAPE}")

    t0 = time.perf_counter()
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError("TensorRT ONNX parse failed:\n" + "\n".join(errs))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1024 ** 3))
    if precision.lower() == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision.lower() == "fp32":
        # Clear TF32 flag if strict IEEE FP32 is desired in TRT FP32 engine
        if hasattr(trt.BuilderFlag, "TF32"):
            config.clear_flag(trt.BuilderFlag.TF32)

    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(input_name, MIN_SHAPE, OPT_SHAPE, MAX_SHAPE)
    config.add_optimization_profile(profile)

    print(f"Compiling serialized {precision.upper()} engine for Blackwell (sm_120)...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT build_serialized_network returned None.")

    ENGINES_DIR.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    build_duration_s = time.perf_counter() - t0
    engine_size_mb = engine_path.stat().st_size / (1024 ** 2)

    print(f"[PASS] Compiled {engine_path} ({engine_size_mb:.2f} MB) in {build_duration_s:.2f} s")
    print("=" * 84)

    build_meta = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "tensorrt_version": trt.__version__,
        "engine_path": _rel(engine_path),
        "engine_size_mb": round(engine_size_mb, 3),
        "build_duration_s": round(build_duration_s, 3),
        "precision": precision.upper(),
        "fold_conv_transpose_bn": fold_bn,
        "min_shape": list(MIN_SHAPE),
        "opt_shape": list(OPT_SHAPE),
        "max_shape": list(MAX_SHAPE),
        "onnx": onnx_info,
    }
    if engine_path == ENGINE_PATH:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "trt_build.json", "w", encoding="utf-8") as f:
            json.dump(build_meta, f, indent=2)

    return build_meta


def build_trt_fp16_engine(
    onnx_path: Path = ONNX_PATH,
    engine_path: Path = ENGINE_PATH,
    workspace_gb: int = 4,
) -> Dict[str, Any]:
    """Build the default BN-folded TensorRT FP16 engine."""
    return build_trt_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        precision="fp16",
        fold_bn=True,
        workspace_gb=workspace_gb,
    )


def build_trt_fp32_engine(
    onnx_path: Path = ONNX_PATH,
    engine_path: Path = ENGINE_FP32_PATH,
    workspace_gb: int = 4,
) -> Dict[str, Any]:
    """Build the TensorRT FP32 engine."""
    return build_trt_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        precision="fp32",
        fold_bn=True,
        workspace_gb=workspace_gb,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedTRT ONNX & TensorRT Engine Builder")
    parser.add_argument("--precision", choices=["fp16", "fp32", "all"], default="all")
    args = parser.parse_args()
    if args.precision in ("fp16", "all"):
        build_trt_fp16_engine()
    if args.precision in ("fp32", "all"):
        build_trt_fp32_engine()
    if args.precision == "all":
        # Unfolded control for `medtrt.probes`, so a clean clone can reproduce the before/after result.
        build_trt_engine(
            onnx_path=ONNX_UNFOLDED_PATH,
            engine_path=ENGINE_FP16_UNFOLDED_PATH,
            precision="fp16",
            fold_bn=False,
        )
