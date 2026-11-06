"""
TensorRT 10.x Backend & Memory Optimization Benchmark (`benchmarks/benchmark_tensorrt.py`).

Evaluates:
1. TensorRT 10.x FP32 (`spleen_unet_fp32.engine`)
2. TensorRT 10.x FP16 Unfolded (`spleen_unet_fp16_unfolded.engine`)
3. TensorRT 10.x FP16 BN-Folded (`spleen_unet_fp16.engine`, dynamic output allocation)
4. TensorRT 10.x FP16 BN-Folded + Pre-Allocated Output Buffer Reuse (`reuse_output_buffer=True`)
across batch sizes B in {1, 2, 4, 8}.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from benchmarks.common import save_benchmark_artifact
from medtrt.bench import benchmark_pure_patch_forward
from medtrt.export import ENGINE_FP16_UNFOLDED_PATH, ENGINE_FP32_PATH, ENGINE_PATH
from medtrt.trt_runner import TRTRunner


def run_tensorrt_benchmarks(warmup_iters: int = 10, timed_iters: int = 30) -> Dict[str, Any]:
    device = torch.device("cuda:0")
    configs = [
        ("trt_fp32", ENGINE_FP32_PATH, False),
        ("trt_fp16_unfolded", ENGINE_FP16_UNFOLDED_PATH, False),
        ("trt_fp16_dynamic_alloc", ENGINE_PATH, False),
        ("trt_fp16_buffer_reuse", ENGINE_PATH, True),
    ]
    results: Dict[str, Any] = {}
    for name, eng_path, reuse_buf in configs:
        runner = TRTRunner(engine_path=eng_path, device=device, reuse_output_buffer=reuse_buf)
        results[name] = {
            "engine_path": eng_path.name,
            "reuse_output_buffer": reuse_buf,
            "engine_file_mb": round(runner.engine_file_mb, 2),
            "device_ctx_mb": round(runner.device_ctx_mb, 2),
            "patches": benchmark_pure_patch_forward(
                predictor=runner,
                device=device,
                use_amp=False,
                batch_sizes=(1, 2, 4, 8),
                warmup_iters=warmup_iters,
                timed_iters=timed_iters,
                extra_vram_mb=runner.trt_vram_mb,
            ),
        }
        del runner
        torch.cuda.empty_cache()

    out_path = save_benchmark_artifact("benchmark_tensorrt_variants.json", {"variants": results})
    print(f"[PASS] Saved TensorRT benchmark to {out_path}")
    return results


if __name__ == "__main__":
    run_tensorrt_benchmarks()
