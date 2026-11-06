"""
Metric Tests (`tests/test_metrics.py`). Need no GPU or data.
"""

from __future__ import annotations

import numpy as np
import pytest

from medtrt.evaluate import compute_foreground_dice
from medtrt.stats import agreement_stats, dice_agreement, error_profile, pareto_flags


def test_monai_dice_matches_the_formula() -> None:
    rng = np.random.default_rng(0)
    pred = (rng.random((24, 24, 12)) > 0.5).astype(np.uint8)
    gt = (rng.random((24, 24, 12)) > 0.5).astype(np.uint8)
    expected = 2.0 * np.count_nonzero(pred & gt) / (np.count_nonzero(pred) + np.count_nonzero(gt))
    assert compute_foreground_dice(pred, gt) == pytest.approx(expected, abs=1e-6)


def test_agreement_stats_known_values() -> None:
    s = agreement_stats([11.0, 12.0, 13.0, 14.0], [10.0, 10.0, 10.0, 10.0])
    assert s["bias_cm3"] == pytest.approx(2.5)
    assert s["sd_cm3"] == pytest.approx(1.291, abs=1e-3)
    assert s["loa_upper_cm3"] - s["loa_lower_cm3"] == pytest.approx(2 * 1.96 * 1.291, abs=1e-2)
    assert s["n_positive"] == 4
    assert s["bias_ci_excludes_zero"] is True


def test_agreement_flags_a_systematic_shift_with_no_random_error() -> None:
    s = agreement_stats([1.1, 2.1, 3.1, 4.1, 5.2], [1.0, 2.0, 3.0, 4.0, 5.0])
    assert s["bias_ci_excludes_zero"] is True


def test_dice_agreement_counts_cases_over_tolerance() -> None:
    ref = {"a": {"dice": "0.950"}, "b": {"dice": "0.950"}, "c": {"dice": "0.950"}}
    cmp_rows = {"a": {"dice": "0.951"}, "b": {"dice": "0.940"}, "c": {"dice": "0.945"}}
    d = dice_agreement(cmp_rows, ref)
    assert d["n_cases_over_tolerance"] == 2
    assert d["every_case_within_tolerance"] is False


def test_pareto_flags_reject_dominated_rows() -> None:
    rows = [
        {"median_sw_latency_ms": 10.0, "mean_dice": 0.96},
        {"median_sw_latency_ms": 12.0, "mean_dice": 0.96},
        {"median_sw_latency_ms": 20.0, "mean_dice": 0.97},
    ]
    assert pareto_flags(rows) == [True, False, True]


def test_error_profile_finds_errors_at_the_poles() -> None:
    gt = np.zeros((10, 10, 20), dtype=np.uint8)
    gt[2:8, 2:8, 5:15] = 1
    pred = gt.copy()
    pred[2:8, 2:8, 15] = 1
    e = error_profile(pred, gt)
    assert e["false_positive_voxels"] == 36
    assert e["false_negative_voxels"] == 0
    assert e["polar_share_of_error_voxels"] == 1.0
