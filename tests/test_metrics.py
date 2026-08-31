from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score, recall_score

from review_reliability.metrics import (
    budget_metrics,
    classification_metrics,
    expected_calibration_error,
    select_validation_threshold,
)


def test_budget_metrics_use_ceil_and_stable_top_scores() -> None:
    target = [1, 0, 1, 0, 0]
    scores = [0.9, 0.8, 0.7, 0.2, 0.1]
    metrics = budget_metrics(target, scores, budget_share=0.21)
    assert metrics["budget_count"] == 2
    assert metrics["precision_at_budget"] == pytest.approx(0.5)
    assert metrics["recall_at_budget"] == pytest.approx(0.5)
    assert metrics["lift_at_budget"] == pytest.approx(1.25)


def test_known_brier_and_calibration_values() -> None:
    target = [0, 1, 1, 0]
    scores = [0.1, 0.8, 0.7, 0.4]
    metrics = classification_metrics(target, scores, threshold=0.5)
    assert metrics["brier"] == pytest.approx(np.mean([0.01, 0.04, 0.09, 0.16]))
    assert metrics["average_precision_attention"] == metrics["pr_auc_attention"]
    assert expected_calibration_error(target, scores, n_bins=2) == pytest.approx(0.25)
    assert metrics["precision_attention"] == 1.0
    assert metrics["recall_attention"] == 1.0


def test_validation_threshold_is_deterministic() -> None:
    target = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.45, 0.9]
    assert select_validation_threshold(target, scores) == pytest.approx(0.45)


def _reference_threshold(target: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(np.r_[0.0, scores, 1.0])
    best: tuple[float, float, float] | None = None
    for threshold in candidates:
        predicted = (scores >= threshold).astype(int)
        candidate = (
            float(f1_score(target, predicted, zero_division=0)),
            float(recall_score(target, predicted, zero_division=0)),
            -float(threshold),
        )
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    return float(-best[2])


@pytest.mark.parametrize("seed", range(12))
def test_vectorized_threshold_matches_reference_with_ties(seed: int) -> None:
    rng = np.random.default_rng(seed)
    target = rng.integers(0, 2, size=180)
    target[:2] = [0, 1]
    scores = rng.choice(np.linspace(0.0, 1.0, 21), size=len(target), replace=True)
    assert select_validation_threshold(target, scores) == _reference_threshold(target, scores)


def test_metrics_reject_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match="probabilities"):
        classification_metrics([0, 1], [0.2, 1.1], threshold=0.5)


def test_metrics_reject_fractional_target_instead_of_truncating_it() -> None:
    with pytest.raises(ValueError, match="binary"):
        classification_metrics([0.0, 0.9], [0.2, 0.8], threshold=0.5)
