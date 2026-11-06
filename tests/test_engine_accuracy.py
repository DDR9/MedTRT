"""
TensorRT Engine Tests (`tests/test_engine_accuracy.py`). Require a CUDA GPU, data, and the engine.

The original project spec asked for a max logit difference under 1e-3. That is not reachable in FP16:
the measured difference on real CT is about 0.24 on logits up to about 23 (see `results/probes.json`).
These tests therefore run on real CT and bound the error at a level FP16 can meet, instead of passing
on a blank input.
"""

from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from medtrt.config import MEDTRT_ROOT, RESULTS_DIR
from medtrt.data import HELDOUT_JSON_PATH
from medtrt.infer import build_pipeline
from medtrt.model import load_spleen_unet
from medtrt.profiles import apply_precision, get_profile
from medtrt.trt_runner import TRTRunner

DEVICE = torch.device("cuda:0")
ROI = 96


@pytest.fixture(scope="module")
def runner() -> TRTRunner:
    return TRTRunner(device=DEVICE)


@pytest.fixture(scope="module")
def spleen_patch() -> torch.Tensor:
    """A real preprocessed CT window centered on the spleen, so the input is not trivial."""
    with open(HELDOUT_JSON_PATH, "r", encoding="utf-8") as f:
        case = next(c for c in json.load(f)["cases"] if c["case_id"] == "spleen_6")
    load_tf, pre_tf, _ = build_pipeline(get_profile("trt_fp16_gpu"))
    data = load_tf({"image": str(MEDTRT_ROOT / case["image"])})
    data["image"] = data["image"].to(DEVICE)
    volume = pre_tf(data)["image"].as_subclass(torch.Tensor)

    label = np.asarray(nib.load(str(MEDTRT_ROOT / case["label"])).dataobj)
    fractions = np.argwhere(label > 0).mean(axis=0) / np.array(label.shape)
    starts = [
        int(np.clip(round(frac * size) - ROI // 2, 0, size - ROI))
        for frac, size in zip(fractions, volume.shape[1:])
    ]
    x, y, z = starts
    patch = volume[:, x : x + ROI, y : y + ROI, z : z + ROI].unsqueeze(0).contiguous().float()
    assert patch.std() > 0.05, "Patch must contain real anatomy"
    return patch


def test_onnx_export_accuracy_record() -> None:
    with open(RESULTS_DIR / "trt_build.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert float(meta["onnx"]["max_abs_diff_vs_pytorch_fp32"]) < 1e-4


def test_trt_forward_never_transfers_to_host(
    runner: TRTRunner, spleen_patch: torch.Tensor, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*_a: object, **_k: object) -> None:
        raise AssertionError("Host transfer attempted inside the TensorRT forward pass")

    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    monkeypatch.setattr(torch.Tensor, "numpy", forbidden)
    out = runner(spleen_patch)
    assert out.is_cuda
    assert out.shape == (1, 2, ROI, ROI, ROI)


def test_trt_rejects_host_tensors(runner: TRTRunner) -> None:
    with pytest.raises(RuntimeError, match="non-CUDA"):
        runner(torch.zeros((1, 1, ROI, ROI, ROI)))


def test_batchnorm_fold_does_not_change_the_fp32_model(spleen_patch: torch.Tensor) -> None:
    apply_precision(get_profile("pytorch_fp32_strict"))
    plain = load_spleen_unet(DEVICE, fold_bn=False)
    folded = load_spleen_unet(DEVICE, fold_bn=True)
    with torch.no_grad():
        a, b = plain(spleen_patch), folded(spleen_patch)
    assert float((a - b).abs().max()) < 1e-3
    assert torch.equal(a.argmax(1), b.argmax(1))


def test_trt_fp16_tracks_fp32_on_real_ct(runner: TRTRunner, spleen_patch: torch.Tensor) -> None:
    apply_precision(get_profile("pytorch_fp32_strict"))
    model = load_spleen_unet(DEVICE, fold_bn=False)
    with torch.no_grad():
        ref = model(spleen_patch)
        out = runner(spleen_patch)

    ref_cls, out_cls = ref.argmax(1), out.argmax(1)
    assert (ref_cls == 1).any(), "Patch must contain predicted spleen for this test to mean anything"
    assert float((ref - out).abs().max()) < 0.5
    assert float((F.softmax(ref, 1) - F.softmax(out, 1)).abs().mean()) < 1e-3
    assert float((ref_cls != out_cls).float().mean()) < 0.01
