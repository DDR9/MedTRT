"""
Unit Tests for Voxel Volumetry Formula (`tests/test_volume_math.py`).
"""

from __future__ import annotations

import numpy as np

from medtrt.evaluate import compute_volume_cm3


def test_synthetic_binary_mask_volume_cm3() -> None:
    # 10 x 20 x 5 = 1000 voxels at 1.5 x 1.5 x 2.0 mm = 4.5 mm^3/voxel -> 4500 mm^3 = 4.5 cm^3
    mask = np.zeros((32, 32, 16), dtype=np.uint8)
    mask[2:12, 5:25, 3:8] = 1

    voxels, vol_cm3 = compute_volume_cm3(mask, (1.5, 1.5, 2.0))
    assert voxels == 1000
    assert np.isclose(vol_cm3, 4.5, atol=1e-7)


def test_empty_mask_volume_zero() -> None:
    mask = np.zeros((16, 16, 16), dtype=np.uint8)
    voxels, vol_cm3 = compute_volume_cm3(mask, (0.75, 0.75, 5.0))
    assert voxels == 0
    assert vol_cm3 == 0.0
