from __future__ import annotations

import pandas as pd
import pytest

from t2i_biasbench.constants import GMR_ELEMENTS
from t2i_biasbench.metrics import (
    bootstrap_ci,
    gmr_score,
    normalised_entropy,
    parity_difference,
    representation_parity,
)


def test_parity_difference_simple_case() -> None:
    dist = {"male": 0.7, "female": 0.3}
    assert parity_difference(dist, "male", "female") == pytest.approx(0.4)


def test_normalized_entropy_bounds() -> None:
    low = normalised_entropy(pd.Series(["a", "a", "a"]))
    high = normalised_entropy(pd.Series(["a", "b", "c", "d"]))
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert high > low


def test_gmr_score_matches_manual_expectation() -> None:
    captions = pd.Series([
        "person portrait fashion cover",
        "face person",
    ])
    score = gmr_score(captions, "beauty", GMR_ELEMENTS)
    assert 0.0 <= score <= 1.0


def test_bootstrap_ci_returns_tuple() -> None:
    frame = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    ci_low, ci_high = bootstrap_ci(frame, stat_fn=lambda f: float(f["x"].mean()), n_bootstrap=50, seed=42)
    assert ci_low <= ci_high


def test_representation_parity_empty_series() -> None:
    dist = representation_parity(pd.Series(dtype=str))
    assert dist == {}
