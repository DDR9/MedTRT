"""
Dashboard Tests (`tests/test_dashboard.py`). Need no GPU or data.

The dashboard must never hold a fixed result. These tests change the data and confirm the numbers
follow, and confirm that every page still renders with no results at all.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Iterator

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app"
sys.path.insert(0, str(APP))

from dashboard import data  # noqa: E402

PAGES = ("overview", "speedup", "latency", "network", "accuracy", "errors", "constructed", "implications", "claims", "method")
CASES = ("case_a", "case_b", "case_c", "case_d")


def _write_eval(folder: Path, name: str, dice: float, volume: float, precision: str = "fp16", engine: str = "trt") -> None:
    rows = [
        {
            "case_id": case,
            "dice": dice + 0.001 * i,
            "v_pred_cm3": volume + i,
            "v_gt_cm3": 100.0 + 10.0 * i,
            "spacing_z_mm": 1.0 + i,
        }
        for i, case in enumerate(CASES)
    ]
    pd.DataFrame(rows).to_csv(folder / f"eval_{name}.csv", index=False)
    profile = {"name": name, "label": name, "engine": engine, "precision": precision, "transforms": "gpu", "inverse": "trilinear", "overlap": 0.5, "sw_batch_size": 4}
    (folder / f"eval_{name}.meta.json").write_text(json.dumps({"profile": profile, "run": {"code_fingerprint": "abc"}}), encoding="utf-8")


def _write_bench(folder: Path, name: str, median: float, precision: str, engine: str, transforms: str = "gpu") -> None:
    def stats(value: float) -> dict:
        return {"median": value, "iqr": 1.0, "max": value * 1.5}

    payload = {
        "profile": {"name": name, "label": name, "engine": engine, "precision": precision, "transforms": transforms, "inverse": "trilinear", "overlap": 0.5, "sw_batch_size": 4},
        "run": {"code_fingerprint": "abc", "engine_fingerprint": "def", "timestamp": "2026-01-01T00:00:00"},
        "config": {"total_timed_passes": 2, "warmup_volumes": 1, "timed_repeats_per_case": 1},
        "stage_latency_ms": {k: stats(median / 8) for k in data.STAGES} | {"e2e_total": stats(median), "ingest": stats(median / 4)},
        "peak_vram_mb": {"max": 100.0},
        "per_case": {c: {"e2e_total": stats(median), "sw_network": stats(median / 8), "peak_vram_mb": {"max": 100.0}} for c in CASES},
        "patch_forward_latency_ms": {"batch_1": {"median": 1.0, "per_patch_median_ms": 1.0}},
        "passes": [{"case_id": c, "e2e_total": median} for c in CASES],
    }
    (folder / f"bench_{name}.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A small invented result set. Its values exist only to test that the dashboard follows them."""
    _write_eval(tmp_path, "reference_fp32", 0.95, 200.0, precision="fp32", engine="pytorch")
    _write_eval(tmp_path, "fast_trt", 0.96, 205.0)
    _write_bench(tmp_path, "baseline_tf32", 1000.0, "tf32", "pytorch", transforms="cpu")
    _write_bench(tmp_path, "fast_trt", 100.0, "fp16", "trt")
    monkeypatch.setattr(data, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(data, "GPU_LOCK", tmp_path / ".gpu_busy")
    yield tmp_path


def test_a_changed_result_changes_the_number(results: Path) -> None:
    res = data.load()
    roles = data.resolve_roles(res)
    assert data.stage(res, "fast_trt", "e2e_total") == 100.0
    before = data.agreement_row(res, roles, "fast_trt")["Mean Dice"]

    _write_eval(results, "fast_trt", 0.90, 205.0)
    _write_bench(results, "fast_trt", 250.0, "fp16", "trt")
    res = data.load()
    assert data.stage(res, "fast_trt", "e2e_total") == 250.0
    assert data.agreement_row(res, roles, "fast_trt")["Mean Dice"] < before


def test_a_new_profile_appears_without_code_changes(results: Path) -> None:
    _write_eval(results, "brand_new_profile", 0.97, 210.0)
    res = data.load()
    roles = data.resolve_roles(res)
    assert "brand_new_profile" in data.profile_names(res)
    assert "brand_new_profile" in set(data.agreement_table(res, roles)["Profile"])


def test_new_cases_are_counted_without_code_changes(results: Path) -> None:
    extra = pd.read_csv(results / "eval_fast_trt.csv")
    extra = pd.concat([extra, extra.assign(case_id=lambda f: f["case_id"] + "_new")])
    extra.to_csv(results / "eval_fast_trt.csv", index=False)
    assert data.n_cases(data.load()) == 2 * len(CASES)


def test_signature_changes_when_a_result_file_changes(results: Path) -> None:
    before = data.signature()
    _write_bench(results, "fast_trt", 123.0, "fp16", "trt")
    assert data.signature() != before


def test_takeaways_are_built_from_the_data(results: Path) -> None:
    res = data.load()
    roles = data.resolve_roles(res)
    assert any("10.0x faster" in line for line in data.takeaways(res, roles))
    _write_bench(results, "fast_trt", 500.0, "fp16", "trt")
    res = data.load()
    assert any("2.0x faster" in line for line in data.takeaways(res, roles))


def test_mixed_code_versions_are_reported(results: Path) -> None:
    payload = json.loads((results / "bench_fast_trt.json").read_text(encoding="utf-8"))
    payload["run"]["code_fingerprint"] = "different"
    (results / "bench_fast_trt.json").write_text(json.dumps(payload), encoding="utf-8")
    state = data.status(data.load())
    assert state["level"] == "unverified"
    assert "code versions" in state["detail"]


def test_gpu_lock_is_reported(results: Path) -> None:
    (results / ".gpu_busy").write_text("job", encoding="utf-8")
    assert data.status(data.load())["level"] == "running"


def test_reading_results_does_not_load_the_gpu_stack() -> None:
    source = (APP / "dashboard" / "data.py").read_text(encoding="utf-8")
    assert not re.search(r"^\s*(import|from)\s+(torch|medtrt|tensorrt)\b", source, re.M)


def test_no_result_values_are_written_in_dashboard_code() -> None:
    """Long decimals in this code would be results. Only named constants below are allowed."""
    allowed = {"0.960", "0.005", "0.975"}
    for path in (APP / "dashboard").glob("*.py"):
        code = path.read_text(encoding="utf-8")
        found = set(re.findall(r"(?<![\w#.])\d+\.\d{3,}\b", code)) - allowed
        assert not found, f"{path.name} contains literal values: {found}"


def _page_script(name: str) -> str:
    return f"import sys; sys.path.insert(0, {str(APP)!r})\nfrom dashboard import pages\npages.shell()\npages.{name}()\n"


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_with_data(page: str, results: Path) -> None:
    app = AppTest.from_string(_page_script(page), default_timeout=30).run()
    assert not app.exception, app.exception


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_with_no_results(page: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "RESULTS_DIR", tmp_path / "does_not_exist")
    monkeypatch.setattr(data, "GPU_LOCK", tmp_path / ".gpu_busy")
    app = AppTest.from_string(_page_script(page), default_timeout=30).run()
    assert not app.exception, app.exception
