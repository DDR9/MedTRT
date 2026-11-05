"""
Phase 4 Dynamic ONNX Export & TensorRT 10.x FP16 Engine Compiler (`src/medtrt/export.py`).

Responsibilities:
1. Extract the core PyTorch 3D UNet from the MONAI bundle (`load_spleen_unet`).
2. Export to `engines/spleen_unet_dynamic.onnx` with dynamic batch axis `B` (`(B, 1, 96, 96, 96)`),
   `opset_version=17`.
3. Validate numerical equivalence between PyTorch (strict IEEE FP32) and ONNX Runtime on a synthetic
   patch, asserting `max_abs_diff < 1e-4`.
4. Compile `engines/spleen_unet_fp16.engine` using TensorRT 10.x Python API with `BuilderFlag.FP16`
   and optimization profile `min=(1, 1, 96, 96, 96)`, `opt=(2, 1, 96, 96, 96)`, `max=(8, 1, 96, 96, 96)`.
"""

from __future__ import annotations

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

from medtrt.config import ENGINES_DIR, RESULTS_DIR
from medtrt.model import load_spleen_unet

ONNX_PATH = ENGINES_DIR / "spleen_unet_dynamic.onnx"
ENGINE_PATH = ENGINES_DIR / "spleen_unet_fp16.engine"

MIN_SHAPE = (1, 1, 96, 96, 96)
OPT_SHAPE = (2, 1, 96, 96, 96)
MAX_SHAPE = (8, 1, 96, 96, 96)


def export_dynamic_onnx(onnx_path: Path = ONNX_PATH) -> Dict[str, Any]:
    """
    Export 3D UNet to ONNX with dynamic batch axis B and verify `max_abs_diff < 1e-4`
    against PyTorch IEEE FP32.
    """
    ENGINES_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")

    # Use strict IEEE FP32 for exact numerical comparison against ONNX Runtime FP32
    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    try:
        model = load_spleen_unet(device)
        model.eval()

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
            pt_out = model(dummy_input).detach().cpu().numpy()
            torch.onnx.export(
                model,
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
    print("PHASE 4.1 — DYNAMIC ONNX EXPORT & VERIFICATION")
    print("=" * 84)
    print(f"ONNX Path         : {onnx_path} ({onnx_path.stat().st_size / (1024**2):.2f} MB)")
    print(f"Opset Version     : 17")
    print(f"Dynamic Batch Axis: B in [1..8], patch=(B, 1, 96, 96, 96)")
    print(f"Export Duration   : {export_time_s:.2f} s")
    print(f"Max Abs Diff      : {max_abs_diff:.3e} (Requirement < 1e-4)")
    print(f"Mean Abs Diff     : {mean_abs_diff:.3e}")
    assert max_abs_diff < 1e-4, f"ONNX max_abs_diff {max_abs_diff} >= 1e-4!"
    print("[PASS] Dynamic ONNX export & numerical verification succeeded.")
    print("=" * 84)

    return {
        "onnx_path": str(onnx_path),
        "onnx_size_mb": round(onnx_path.stat().st_size / (1024 ** 2), 3),
        "opset_version": 17,
        "export_time_s": round(export_time_s, 3),
        "max_abs_diff_vs_pytorch_fp32": max_abs_diff,
        "mean_abs_diff_vs_pytorch_fp32": mean_abs_diff,
    }


def build_trt_fp16_engine(
    onnx_path: Path = ONNX_PATH,
    engine_path: Path = ENGINE_PATH,
    workspace_gb: int = 4,
) -> Dict[str, Any]:
    """
    Compile TensorRT 10.x FP16 engine (`engines/spleen_unet_fp16.engine`) with dynamic batch profile:
        min=(1, 1, 96, 96, 96), opt=(2, 1, 96, 96, 96), max=(8, 1, 96, 96, 96)
    """
    onnx_info = export_dynamic_onnx(onnx_path)

    print("=" * 84)
    print("PHASE 4.2 — TENSORRT 10.x FP16 ENGINE COMPILATION")
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
    config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(input_name, MIN_SHAPE, OPT_SHAPE, MAX_SHAPE)
    config.add_optimization_profile(profile)

    print("Compiling serialized FP16 engine for Blackwell (sm_120)...")
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
        "engine_path": str(engine_path),
        "engine_size_mb": round(engine_size_mb, 3),
        "build_duration_s": round(build_duration_s, 3),
        "precision": "FP16",
        "min_shape": list(MIN_SHAPE),
        "opt_shape": list(OPT_SHAPE),
        "max_shape": list(MAX_SHAPE),
        "onnx": onnx_info,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "trt_build.json", "w", encoding="utf-8") as f:
        json.dump(build_meta, f, indent=2)

    return build_meta


if __name__ == "__main__":
    build_trt_fp16_engine()
