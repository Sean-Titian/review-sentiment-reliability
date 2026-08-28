from __future__ import annotations

import numpy as np
import pytest

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
    assert expected_calibration_error(target, scores, n_bins=2) == pytest.approx(0.25)
    assert metrics["precision_attention"] == 1.0
    assert metrics["recall_attention"] == 1.0


def test_validation_threshold_is_deterministic() -> None:
    target = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.45, 0.9]
    assert select_validation_threshold(target, scores) == pytest.approx(0.45)


def test_metrics_reject_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match="probabilities"):
        classification_metrics([0, 1], [0.2, 1.1], threshold=0.5)


def test_metrics_reject_fractional_target_instead_of_truncating_it() -> None:
    with pytest.raises(ValueError, match="binary"):
        classification_metrics([0.0, 0.9], [0.2, 0.8], threshold=0.5)
