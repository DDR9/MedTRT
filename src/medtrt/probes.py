"""
Measurement Probes (`src/medtrt/probes.py`).

Targeted experiments behind statements in the README. Writes `results/probes.json`.

1. Ingestion: time to get a volume onto the GPU from gzip, eager `.nii`, and `mmap` `.nii`. Timing a
   lazy read alone measures a file open, so every path is timed through to data on the device.
2. Resampling dtype: what `grid_sample` actually receives during preprocessing and inversion.
3. FP16 numerics: TensorRT against true FP32 on real CT windows, including the direction of change.
4. Network like-for-like: the same windows on every backend, in a sustained loop, with and without
   `cudnn.benchmark`, so the PyTorch baseline is not an untuned straw man.
5. Clock state: the same inference run back-to-back and after 3 s of GPU idle.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from medtrt.config import MEDTRT_ROOT, RESULTS_DIR, run_metadata
from medtrt.data import HELDOUT_JSON_PATH, ensure_uncompressed_nifti, load_nifti_mmap
from medtrt.export import ENGINE_FP16_UNFOLDED_PATH
from medtrt.infer import build_pipeline, build_sliding_window_inferer, infer_volume
from medtrt.model import load_spleen_unet
from medtrt.profiles import LIKE_FOR_LIKE, SHIPPING, get_profile
from medtrt.trt_runner import TRTRunner

pf = time.perf_counter
DEVICE = torch.device("cuda:0")


def _sync() -> None:
    torch.cuda.synchronize(DEVICE)


def _median(values: List[float]) -> float:
    return round(float(np.median(values)), 3)


def probe_ingestion(cases: List[Dict[str, str]], repeats: int = 5) -> Dict[str, Any]:
    gzip_load = build_pipeline(get_profile("trt_fp16"))[0]

    def gzip_path(gz: str, _nii: str) -> Any:
        return gzip_load({"image": gz})["image"]

    def eager_path(_gz: str, nii: str) -> Any:
        return torch.from_numpy(np.ascontiguousarray(np.asarray(nib.load(nii, mmap=False).dataobj))).unsqueeze(0)

    def mmap_path(_gz: str, nii: str) -> Any:
        return load_nifti_mmap({"image": nii})["image"]

    paths = {"gzip": gzip_path, "nii_eager": eager_path, "nii_mmap": mmap_path}
    timings: Dict[str, Dict[str, List[float]]] = {k: {"read_ms": [], "to_gpu_total_ms": []} for k in paths}
    for c in cases:
        gz = str(MEDTRT_ROOT / c["image"])
        nii = str(ensure_uncompressed_nifti(MEDTRT_ROOT / c["image"]))
        for fn in paths.values():
            _ = fn(gz, nii).to(DEVICE)
        for _ in range(repeats):
            for name, fn in paths.items():
                _sync()
                t0 = pf()
                tensor = fn(gz, nii)
                t1 = pf()
                on_gpu = tensor.to(DEVICE)
                _sync()
                t2 = pf()
                del on_gpu
                timings[name]["read_ms"].append((t1 - t0) * 1e3)
                timings[name]["to_gpu_total_ms"].append((t2 - t0) * 1e3)

    out = {
        name: {"read_ms_median": _median(v["read_ms"]), "to_gpu_total_ms_median": _median(v["to_gpu_total_ms"])}
        for name, v in timings.items()
    }
    out["note"] = "Warm OS page cache. nii_mmap read_ms is a lazy open; its bytes are read during the copy."
    return out


def probe_resample_dtype(case: Dict[str, str], predictor: Callable[[torch.Tensor], torch.Tensor]) -> Dict[str, Any]:
    log: List[Dict[str, Any]] = []
    real = torch.nn.functional.grid_sample

    def spy(*a: Any, **k: Any) -> torch.Tensor:
        inp = k["input"] if "input" in k else a[0]
        log.append({"dtype": str(inp.dtype), "device": str(inp.device), "mode": k.get("mode")})
        return real(*a, **k)

    profile = get_profile(SHIPPING)
    load_tf, pre_tf, post_tf = build_pipeline(profile)
    torch.nn.functional.grid_sample = spy
    try:
        d = load_tf({"image": str(MEDTRT_ROOT / case["image"])})
        d["image"] = d["image"].to(DEVICE)
        d = pre_tf(d)
        n_pre = len(log)
        inferer = build_sliding_window_inferer(sw_batch_size=profile.sw_batch_size, overlap=profile.overlap)
        with torch.no_grad():
            logits = inferer(inputs=d["image"].unsqueeze(0), network=predictor)
        d["pred"] = logits[0].detach().float()
        out = post_tf(d)
    finally:
        torch.nn.functional.grid_sample = real
    return {"preprocess": log[:n_pre], "inverse": log[n_pre:], "final_mask_dtype": str(out["pred"].dtype)}


def probe_fp16_numerics(
    cases: List[Dict[str, str]],
    model: torch.nn.Module,
    trt_runner: TRTRunner,
) -> Dict[str, Any]:
    """Compare TensorRT FP16 with true FP32 on every window of every case."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows: List[Dict[str, float]] = []

    def both(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            ref = model(x)
            trt_out = trt_runner(x).clone()
        ref_cls, trt_cls = ref.argmax(1), trt_out.argmax(1)
        margin_shift = (trt_out[:, 1] - trt_out[:, 0]) - (ref[:, 1] - ref[:, 0])
        rows.append(
            {
                "max_abs": float((ref - trt_out).abs().max()),
                "mean_abs": float((ref - trt_out).abs().mean()),
                "ref_abs_max": float(ref.abs().max()),
                "prob_mae": float((F.softmax(ref, 1) - F.softmax(trt_out, 1)).abs().mean()),
                "margin_shift": float(margin_shift.mean()),
                "bg_to_fg": int(((trt_cls == 1) & (ref_cls == 0)).sum()),
                "fg_to_bg": int(((trt_cls == 0) & (ref_cls == 1)).sum()),
            }
        )
        return ref

    profile = get_profile("trt_fp16_gpu")
    load_tf, pre_tf, _ = build_pipeline(profile)
    for c in cases:
        d = load_tf({"image": str(MEDTRT_ROOT / c["image"])})
        d["image"] = d["image"].to(DEVICE)
        d = pre_tf(d)
        with torch.no_grad():
            build_sliding_window_inferer(sw_batch_size=4, overlap=0.5)(inputs=d["image"].unsqueeze(0), network=both)

    return {
        "sliding_window_calls": len(rows),
        "max_abs_logit_diff": round(max(r["max_abs"] for r in rows), 5),
        "mean_abs_logit_diff": round(float(np.mean([r["mean_abs"] for r in rows])), 5),
        "reference_logit_abs_max": round(max(r["ref_abs_max"] for r in rows), 3),
        "probability_mae": float(np.mean([r["prob_mae"] for r in rows])),
        "mean_foreground_margin_shift": round(float(np.mean([r["margin_shift"] for r in rows])), 5),
        "window_voxels_background_to_foreground": sum(r["bg_to_fg"] for r in rows),
        "window_voxels_foreground_to_background": sum(r["fg_to_bg"] for r in rows),
    }


def probe_fold_effect(
    cases: List[Dict[str, str]],
    reference: torch.nn.Module,
    folded_runner: TRTRunner,
) -> Dict[str, Any]:
    """
    Measure what folding `BatchNorm3d` into `ConvTranspose3d` changes, on identical inputs and windows.
    The reference is true FP32 on the unmodified bundle weights.
    """
    if not ENGINE_FP16_UNFOLDED_PATH.exists():
        return {"available": False}

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    unfolded_runner = TRTRunner(engine_path=ENGINE_FP16_UNFOLDED_PATH, device=DEVICE)
    profile = get_profile(LIKE_FOR_LIKE)

    shifts = {
        name: float((m.bias - m.weight * m.running_mean / (m.running_var + m.eps).sqrt()).abs().max())
        for name, m in reference.named_modules()
        if isinstance(m, torch.nn.BatchNorm3d)
    }
    layer = max(shifts, key=shifts.get)

    load_tf, pre_tf, _ = build_pipeline(profile)
    d = load_tf({"image": str(MEDTRT_ROOT / cases[0]["image"])})
    d["image"] = d["image"].to(DEVICE)
    volume = pre_tf(d)["image"].as_subclass(torch.Tensor)
    starts = [(size - 96) // 2 for size in volume.shape[1:]]
    window = volume[:, starts[0] : starts[0] + 96, starts[1] : starts[1] + 96, starts[2] : starts[2] + 96]
    with torch.no_grad():
        fp32_diff = (reference(window[None]) - load_spleen_unet(DEVICE, fold_bn=True)(window[None])).abs().max()

    def dice(pred: np.ndarray, gt: np.ndarray) -> float:
        return float(2.0 * np.count_nonzero(pred & gt) / (np.count_nonzero(pred) + np.count_nonzero(gt)))

    rows: Dict[str, List[Tuple[float, float, float]]] = {"unfolded": [], "folded": []}
    for c in cases:
        image = MEDTRT_ROOT / c["image"]
        voxel_cm3 = float(np.prod(nib.load(str(image)).header.get_zooms()[:3])) / 1000.0
        gt = np.asarray(nib.load(str(MEDTRT_ROOT / c["label"])).dataobj) == 1
        ref_mask = infer_volume(image, reference, profile, DEVICE)["pred_mask_3d"] == 1
        for name, runner in (("unfolded", unfolded_runner), ("folded", folded_runner)):
            mask = infer_volume(image, runner, profile, DEVICE)["pred_mask_3d"] == 1
            rows[name].append(
                (
                    (np.count_nonzero(mask) - np.count_nonzero(ref_mask)) * voxel_cm3,
                    dice(mask, gt) - dice(ref_mask, gt),
                    dice(mask, gt),
                )
            )

    def summarize(values: List[Tuple[float, float, float]]) -> Dict[str, Any]:
        dv, dd, dc = (np.array(v) for v in zip(*values))
        return {
            "volume_bias_cm3": round(float(dv.mean()), 4),
            "volume_sd_cm3": round(float(dv.std(ddof=1)), 4),
            "n_positive": int((dv > 0).sum()),
            "n": len(dv),
            "mean_abs_delta_dice": round(float(np.abs(dd).mean()), 6),
            "mean_dice": round(float(dc.mean()), 6),
        }

    return {
        "available": True,
        "largest_bn_shift_layer": layer,
        "largest_bn_shift": round(shifts[layer], 2),
        "fp32_fold_max_abs_logit_diff": float(fp32_diff),
        "unfolded": summarize(rows["unfolded"]),
        "folded": summarize(rows["folded"]),
    }


def _timed_inference(
    predictor: Callable[[torch.Tensor], torch.Tensor], use_amp: bool, x: torch.Tensor, overlap: float, sw_batch: int
) -> Tuple[float, int]:
    inferer = build_sliding_window_inferer(sw_batch_size=sw_batch, overlap=overlap)
    events: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []
    windows = [0]

    def wrapped(p: torch.Tensor) -> torch.Tensor:
        s_ev, e_ev = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        windows[0] += int(p.shape[0])
        s_ev.record()
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = predictor(p)
        e_ev.record()
        events.append((s_ev, e_ev))
        return out

    _sync()
    with torch.no_grad():
        inferer(inputs=x, network=wrapped)
    _sync()
    return float(sum(s.elapsed_time(e) for s, e in events)), windows[0]


def _select(name: str, model: torch.nn.Module, trt_runner: TRTRunner) -> Tuple[Callable[..., torch.Tensor], bool]:
    torch.backends.cudnn.benchmark = name.endswith("_tuned")
    if name == "trt":
        return trt_runner, False
    torch.backends.cuda.matmul.allow_tf32 = name != "strict_fp32"
    torch.backends.cudnn.allow_tf32 = name != "strict_fp32"
    return model, name.startswith("amp")


def probe_network(
    vols: List[Tuple[str, torch.Tensor]], model: torch.nn.Module, trt_runner: TRTRunner, repeats: int = 5
) -> Dict[str, Any]:
    fast = get_profile(SHIPPING)
    grids = [(0.5, 4), (fast.overlap, fast.sw_batch_size), (0.25, 8)]
    results: Dict[str, Any] = {}
    for name in ("strict_fp32", "tf32", "tf32_tuned", "amp", "amp_tuned", "trt"):
        predictor, use_amp = _select(name, model, trt_runner)
        for overlap, sw_batch in dict.fromkeys(grids):
            for _, x in vols[:3]:
                _timed_inference(predictor, use_amp, x, overlap, sw_batch)
            samples = [_timed_inference(predictor, use_amp, x, overlap, sw_batch) for _ in range(repeats) for _, x in vols]
            key = f"{name}|{overlap}|{sw_batch}"
            results[key] = {
                "sw_network_ms_median": _median([s[0] for s in samples]),
                "windows_all_cases": sum(s[1] for s in samples[: len(vols)]),
            }
            print(key, results[key], flush=True)
    torch.backends.cudnn.benchmark = False
    return results


def probe_clock_state(x: torch.Tensor, model: torch.nn.Module, trt_runner: TRTRunner) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for name, overlap, sw_batch in (("trt", 0.5, 4), ("tf32", 0.5, 4)):
        predictor, use_amp = _select(name, model, trt_runner)
        _timed_inference(predictor, use_amp, x, overlap, sw_batch)
        back_to_back = [_timed_inference(predictor, use_amp, x, overlap, sw_batch)[0] for _ in range(8)]
        after_idle: List[float] = []
        for _ in range(6):
            time.sleep(3.0)
            after_idle.append(_timed_inference(predictor, use_amp, x, overlap, sw_batch)[0])
        results[f"{name}|{overlap}|{sw_batch}"] = {
            "back_to_back_ms_median": _median(back_to_back),
            "after_3s_idle_ms_median": _median(after_idle),
            "idle_inflation": round(float(np.median(after_idle) / np.median(back_to_back)), 2),
        }
        print(name, results[f"{name}|{overlap}|{sw_batch}"], flush=True)
    return results


def run_probes() -> Dict[str, Any]:
    with open(HELDOUT_JSON_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    trt_runner = TRTRunner(device=DEVICE)
    model = load_spleen_unet(DEVICE)
    reference = load_spleen_unet(DEVICE, fold_bn=False)

    payload: Dict[str, Any] = {"run": run_metadata()}
    payload["ingestion"] = probe_ingestion(cases)
    print("ingestion", payload["ingestion"], flush=True)
    payload["resample_dtype"] = probe_resample_dtype(cases[1], trt_runner)
    print("resample", payload["resample_dtype"], flush=True)
    payload["fp16_numerics"] = probe_fp16_numerics(cases, reference, trt_runner)
    print("numerics", payload["fp16_numerics"], flush=True)
    payload["fold_effect"] = probe_fold_effect(cases, reference, trt_runner)
    print("fold", payload["fold_effect"], flush=True)

    load_tf, pre_tf, _ = build_pipeline(get_profile(SHIPPING))
    vols: List[Tuple[str, torch.Tensor]] = []
    for c in cases:
        d = load_tf({"image": str(MEDTRT_ROOT / c["image"])})
        d["image"] = d["image"].to(DEVICE)
        d = pre_tf(d)
        vols.append((c["case_id"], d["image"].unsqueeze(0).clone().as_subclass(torch.Tensor)))

    payload["network_sustained"] = probe_network(vols, model, trt_runner)
    payload["clock_state"] = probe_clock_state(dict(vols)["spleen_6"], model, trt_runner)

    with open(RESULTS_DIR / "probes.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print("[PASS] Saved results/probes.json")
    return payload


if __name__ == "__main__":
    run_probes()
