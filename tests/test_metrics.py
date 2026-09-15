from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score, recall_score

from review_reliability.metrics import (
    DEFAULT_CAPACITY_SHARES,
    PRIMARY_CAPACITY_SHARE,
    budget_metrics,
    capacity_curve_metrics,
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
    assert set(metrics) == {
        "budget_share",
        "budget_count",
        "budget_cutoff_tied_rows",
        "budget_cutoff_tied_weight",
        "budget_cutoff_fraction_selected",
        "precision_at_budget",
        "recall_at_budget",
        "lift_at_budget",
    }


def test_budget_cutoff_ties_use_fractional_order_invariant_allocation() -> None:
    target = np.asarray([1, 0, 1, 0])
    scores = np.asarray([0.9, 0.8, 0.8, 0.1])
    expected = budget_metrics(target, scores, budget_share=0.50)
    shuffled = budget_metrics(target[[2, 0, 3, 1]], scores[[2, 0, 3, 1]], budget_share=0.50)

    assert expected["budget_count"] == 2
    assert expected["budget_cutoff_tied_rows"] == 2
    assert expected["budget_cutoff_fraction_selected"] == pytest.approx(0.5)
    assert expected["precision_at_budget"] == pytest.approx(0.75)
    assert expected["recall_at_budget"] == pytest.approx(0.75)
    assert expected["lift_at_budget"] == pytest.approx(1.5)
    for metric in ("precision_at_budget", "recall_at_budget", "lift_at_budget"):
        assert shuffled[metric] == pytest.approx(expected[metric])


def test_frequency_weighted_budget_matches_explicit_cluster_replication() -> None:
    target = np.asarray([1, 0, 1, 0])
    scores = np.asarray([0.9, 0.8, 0.8, 0.1])
    frequency = np.asarray([2, 0, 1, 3])
    weighted = budget_metrics(
        target,
        scores,
        budget_share=0.50,
        sample_weight=frequency,
    )
    repeated_target = np.repeat(target, frequency)
    repeated_scores = np.repeat(scores, frequency)
    explicit = budget_metrics(repeated_target, repeated_scores, budget_share=0.50)

    for metric in (
        "budget_count",
        "budget_cutoff_tied_weight",
        "budget_cutoff_fraction_selected",
        "precision_at_budget",
        "recall_at_budget",
        "lift_at_budget",
    ):
        assert weighted[metric] == pytest.approx(explicit[metric])


def test_budget_weights_must_be_integer_frequencies() -> None:
    with pytest.raises(ValueError, match="integer frequency"):
        budget_metrics(
            [0, 1],
            [0.2, 0.8],
            sample_weight=[0.5, 1.5],
        )


def test_budget_weights_round_numerically_near_integer_frequencies() -> None:
    expected = budget_metrics(
        [0, 1, 1],
        [0.2, 0.8, 0.8],
        budget_share=0.50,
        sample_weight=[1.0, 2.0, 1.0],
    )
    near_integer = budget_metrics(
        [0, 1, 1],
        [0.2, 0.8, 0.8],
        budget_share=0.50,
        sample_weight=[1.0 + 5e-13, 2.0 - 5e-13, 1.0],
    )
    assert near_integer == expected


def test_default_capacity_curve_reports_ordered_effective_capacity_points() -> None:
    target = [1, 0, 1, 0, 1] * 4
    scores = np.linspace(0.99, 0.01, len(target))
    curve = capacity_curve_metrics(target, scores)
    points = curve["points"]

    assert PRIMARY_CAPACITY_SHARE == 0.10
    assert DEFAULT_CAPACITY_SHARES == (0.05, 0.10, 0.20)
    assert curve["budget_shares"] == [0.05, 0.10, 0.20]
    assert [point["budget_share"] for point in points] == [0.05, 0.10, 0.20]
    assert [point["budget_count"] for point in points] == [1, 2, 4]
    assert [point["budget_weight"] for point in points] == [1.0, 2.0, 4.0]
    assert [point["total_weight"] for point in points] == [20.0, 20.0, 20.0]
    assert [point["effective_budget_share"] for point in points] == pytest.approx(
        [0.05, 0.10, 0.20]
    )
    assert all(curve["monotonicity_checks"].values())


@pytest.mark.parametrize(
    ("shares", "message"),
    (
        ((0.05, np.nan, 0.20), "finite"),
        ((0.05, np.inf, 0.20), "finite"),
        ((0.05, True, 0.20), "real numbers"),
        ((0.0, 0.10, 0.20), "lie in"),
        ((-0.05, 0.10, 0.20), "lie in"),
        ((0.05, 0.10, 1.01), "lie in"),
        ((0.05, 0.10, 0.10), "duplicates"),
        ((0.10, 0.05, 0.20), "strictly increasing"),
        ((), "non-empty"),
    ),
)
def test_capacity_curve_rejects_invalid_capacity_sets(
    shares: tuple[object, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        capacity_curve_metrics(
            [0, 1, 0, 1],
            [0.1, 0.9, 0.2, 0.8],
            budget_shares=shares,  # type: ignore[arg-type]
        )


def test_capacity_curve_frequency_weights_match_explicit_replication() -> None:
    target = np.asarray([1, 0, 1, 0, 1, 0])
    scores = np.asarray([0.9, 0.8, 0.8, 0.5, 0.3, 0.1])
    frequency = np.asarray([2, 0, 1, 3, 2, 2])
    weighted = capacity_curve_metrics(
        target,
        scores,
        budget_shares=(0.20, 0.50, 0.80),
        sample_weight=frequency,
    )
    explicit = capacity_curve_metrics(
        np.repeat(target, frequency),
        np.repeat(scores, frequency),
        budget_shares=(0.20, 0.50, 0.80),
    )

    comparable_fields = (
        "budget_share",
        "budget_count",
        "budget_weight",
        "total_weight",
        "effective_budget_share",
        "budget_cutoff_tied_weight",
        "budget_cutoff_fraction_selected",
        "precision_at_budget",
        "recall_at_budget",
        "lift_at_budget",
    )
    for weighted_point, explicit_point in zip(
        weighted["points"], explicit["points"], strict=True
    ):
        for field in comparable_fields:
            assert weighted_point[field] == pytest.approx(explicit_point[field])


def test_capacity_curve_ties_are_order_invariant_and_recall_is_nondecreasing() -> None:
    target = np.asarray([1, 0, 1, 0, 1, 0, 1, 0])
    scores = np.asarray([0.9, 0.8, 0.8, 0.8, 0.5, 0.5, 0.1, 0.1])
    order = np.asarray([3, 0, 6, 2, 7, 1, 5, 4])
    expected = capacity_curve_metrics(
        target,
        scores,
        budget_shares=(0.125, 0.375, 0.625, 1.0),
    )
    reordered = capacity_curve_metrics(
        target[order],
        scores[order],
        budget_shares=(0.125, 0.375, 0.625, 1.0),
    )

    assert reordered == expected
    points = expected["points"]
    counts = [int(point["budget_count"]) for point in points]
    recalls = [float(point["recall_at_budget"]) for point in points]
    assert counts == sorted(counts)
    assert recalls == sorted(recalls)


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
