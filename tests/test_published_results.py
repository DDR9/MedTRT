"""
Published-Results Test (`tests/test_published_results.py`). Needs no GPU or data.

Fails if the README, the CSVs, and the raw benchmark passes stop agreeing with each other.
"""

from __future__ import annotations

import pytest

from medtrt.verify_claims import run_checks


@pytest.mark.parametrize("result", run_checks(), ids=lambda r: r["check"])
def test_published_claim(result: dict) -> None:
    assert result["passed"], result["detail"]
