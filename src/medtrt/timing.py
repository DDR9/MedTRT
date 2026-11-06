"""
Phase 2 & 3 CUDA Measurement Harness & NVTX Instrumentation (`src/medtrt/timing.py`).

Responsibilities:
- Provide `torch.cuda.Event(enable_timing=True)` stage boundary timers.
- Wrap all pipeline stages in strictly non-overlapping NVTX ranges (`torch.cuda.nvtx.range_push/pop`):
  `load`, `preprocess`, `h2d`, `sw_network`, `sw_aggregate`, `postprocess`.
- Instrument the `predictor` callable inside `SlidingWindowInferer` with unsynchronized CUDA event pairs
  during the patch loop, synchronizing only at stage boundaries so sliding-window execution is never stalled.
- Compute comprehensive distribution statistics across timed repeats:
  `mean`, `median`, `std`, `iqr`, `q25`, `q75`, `min`, `max`, `p90`, `p95`, `ci95_lower`, `ci95_upper`.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Dict, Iterator, List, Tuple

import numpy as np
import torch
from scipy import stats as sps

STAGE_NAMES: Tuple[str, ...] = (
    "load",
    "preprocess",
    "h2d",
    "sw_network",
    "sw_aggregate",
    "postprocess",
)


@contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    """Push and pop an NVTX range marker around a pipeline stage."""
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


class CudaStageTimer:
    """
    Records `torch.cuda.Event(enable_timing=True)` pairs at stage boundaries
    while wrapping each stage in an NVTX range.
    """

    def __init__(self) -> None:
        self._events: Dict[str, List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = {
            stage: [] for stage in STAGE_NAMES
        }
        self._e2e_start = torch.cuda.Event(enable_timing=True)
        self._e2e_end = torch.cuda.Event(enable_timing=True)
        self.patch_windows: int = 0

    def start_e2e(self) -> None:
        torch.cuda.synchronize()
        self._e2e_start.record()

    def stop_e2e(self) -> None:
        self._e2e_end.record()
        torch.cuda.synchronize()

    @contextmanager
    def stage(self, name: str, sync_on_exit: bool = True) -> Iterator[None]:
        """
        Measure a pipeline stage using `torch.cuda.Event(enable_timing=True)` and NVTX markers.
        """
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        torch.cuda.nvtx.range_push(name)
        start_ev.record()
        try:
            yield
        finally:
            end_ev.record()
            if sync_on_exit:
                end_ev.synchronize()
            torch.cuda.nvtx.range_pop()
            self._events.setdefault(name, []).append((start_ev, end_ev))

    def record_patch_events(
        self, start_ev: torch.cuda.Event, end_ev: torch.cuda.Event
    ) -> None:
        self._events["sw_network"].append((start_ev, end_ev))

    def resolve_timings_ms(self) -> Dict[str, float]:
        """
        Compute elapsed milliseconds for all stages after `stop_e2e()` has synchronized the GPU.
        Separates `sw_network` (sum of patch forward passes) from `sw_aggregate`
        (window slicing, Gaussian importance weighting, and stitching accumulation).
        """
        timings: Dict[str, float] = {}
        for name in ("load", "preprocess", "h2d", "sw_network", "postprocess"):
            pairs = self._events.get(name, [])
            timings[name] = float(
                sum(s.elapsed_time(e) for s, e in pairs)
            )

        sw_total_pairs = self._events.get("_sw_total", [])
        sw_total_ms = float(sum(s.elapsed_time(e) for s, e in sw_total_pairs))
        timings["sw_aggregate"] = max(0.0, sw_total_ms - timings["sw_network"])
        timings["sw_total"] = sw_total_ms
        timings["ingest"] = timings["load"] + timings["h2d"]
        timings["e2e_total"] = float(self._e2e_start.elapsed_time(self._e2e_end))
        timings["non_network"] = max(0.0, timings["e2e_total"] - timings["sw_network"])
        timings["patch_calls"] = float(len(self._events.get("sw_network", [])))
        timings["patch_windows"] = float(self.patch_windows)
        return timings


class InstrumentedPredictor:
    """
    Wraps any patch predictor (`torch.nn.Module` or `TRTRunner`) inside `SlidingWindowInferer`
    so that:
    1. Outer `sw_aggregate` NVTX range is paused during `sw_network` execution and resumed after,
       creating strictly disjoint `sw_aggregate` and `sw_network` ranges in Nsight Systems traces.
    2. `torch.cuda.Event(enable_timing=True)` pairs are recorded around every patch batch forward pass
       without synchronizing inside the patch loop.
    """

    def __init__(
        self,
        predictor: Callable[[torch.Tensor], torch.Tensor],
        timer: CudaStageTimer,
        use_amp: bool = False,
        use_fp16_input: bool = False,
    ) -> None:
        self.predictor = predictor
        self.timer = timer
        self.use_amp = use_amp
        self.use_fp16_input = use_fp16_input

    def __call__(self, patch_batch: torch.Tensor) -> torch.Tensor:
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("sw_network")
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        self.timer.patch_windows += int(patch_batch.shape[0])
        start_ev.record()
        try:
            if self.use_amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    out = self.predictor(patch_batch)
            elif self.use_fp16_input:
                out = self.predictor(patch_batch.half()).float()
            else:
                out = self.predictor(patch_batch)
        finally:
            end_ev.record()
            self.timer.record_patch_events(start_ev, end_ev)
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("sw_aggregate")
        return out


def compute_distribution_stats(values: List[float]) -> Dict[str, float]:
    """
    Calculate mean, median, std, IQR (Q75 - Q25), q25, q75, min, max, p90, p95,
    and 95% confidence interval (`ci95_lower`, `ci95_upper`) for a list of measurements.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {
            k: 0.0
            for k in (
                "median",
                "iqr",
                "q25",
                "q75",
                "mean",
                "std",
                "min",
                "max",
                "p90",
                "p95",
                "ci95_lower",
                "ci95_upper",
            )
        }
    q25, median, q75, p90, p95 = np.percentile(arr, [25.0, 50.0, 75.0, 90.0, 95.0])
    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    if n > 1 and std_v > 0.0:
        half_ci = float(sps.t.ppf(0.975, n - 1)) * std_v / np.sqrt(n)
    else:
        half_ci = 0.0
    return {
        "median": round(float(median), 4),
        "iqr": round(float(q75 - q25), 4),
        "q25": round(float(q25), 4),
        "q75": round(float(q75), 4),
        "mean": round(mean_v, 4),
        "std": round(std_v, 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "p90": round(float(p90), 4),
        "p95": round(float(p95), 4),
        "ci95_lower": round(mean_v - half_ci, 4),
        "ci95_upper": round(mean_v + half_ci, 4),
    }
