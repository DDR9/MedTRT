"""
PyTorch Backend Benchmark (`benchmarks/benchmark_pytorch.py`).

Evaluates:
1. PyTorch Strict IEEE FP32 (`allow_tf32=False`)
2. PyTorch Default TF32 (`allow_tf32=True`)
3. PyTorch AMP FP16 (`torch.autocast`)
4. PyTorch Pure FP16 (`model.half()`)
across batch sizes B in {1, 2, 4, 8} and full-volume execution.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from benchmarks.common import save_benchmark_artifact
from medtrt.bench import benchmark_pure_patch_forward, run_benchmark


def run_pytorch_benchmarks(warmup_iters: int = 10, timed_iters: int = 30) -> Dict[str, Any]:
    device = torch.device("cuda:0")
    from medtrt.profiles import build_predictor, get_profile

    patch_summary: Dict[str, Any] = {}
    for pname in ("pytorch_fp32_strict", "pytorch_fp32", "pytorch_amp", "pytorch_fp16"):
        prof = get_profile(pname)
        pred, use_amp, extra_vram = build_predictor(prof, device)
        patch_summary[pname] = benchmark_pure_patch_forward(
            predictor=pred,
            device=device,
            use_amp=use_amp,
            batch_sizes=(1, 2, 4, 8),
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
            extra_vram_mb=extra_vram,
        )
        del pred
        torch.cuda.empty_cache()

    out_path = save_benchmark_artifact("benchmark_pytorch_patches.json", {"backends": patch_summary})
    print(f"[PASS] Saved PyTorch patch benchmark to {out_path}")
    return patch_summary


if __name__ == "__main__":
    run_pytorch_benchmarks()
