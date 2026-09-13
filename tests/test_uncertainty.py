from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from review_reliability.public_safety import assert_aggregate_report_safe
from review_reliability.uncertainty import (
    _metric_values,
    conditional_cluster_bootstrap,
)


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(101)
    clusters = np.repeat([f"cluster-{index:02d}" for index in range(40)], 4)
    target = rng.binomial(1, 0.30, size=len(clusters))
    target[:2] = [0, 1]
    scores = np.clip(0.18 + 0.55 * target + rng.normal(0, 0.16, len(target)), 0.01, 0.99)
    return target, scores, clusters


def test_cluster_bootstrap_is_deterministic_order_invariant_and_aggregate_safe() -> None:
    target, scores, clusters = _fixture()
    first = conditional_cluster_bootstrap(
        target,
        scores,
        clusters,
        train_prevalence=0.28,
        draws=80,
        seed=303,
        enforce_release_gate=False,
    )
    order = np.random.default_rng(404).permutation(len(target))
    reordered = conditional_cluster_bootstrap(
        target[order],
        scores[order],
        clusters[order],
        train_prevalence=0.28,
        draws=80,
        seed=303,
        enforce_release_gate=False,
    )

    assert first["draws_attempted"] == first["draws_requested"] == 80
    assert first["draws_replenished"] == 0
    assert first["cluster_unit"] == "near_text_fingerprint"
    assert first["interval_type"] == "marginal percentile, not simultaneous"
    assert first["release_gate"]["enforced"] is False
    assert first["release_gate"]["passed"] is None
    assert first["release_gate"]["criteria_met"] is False
    for metric, interval in first["intervals"].items():
        for field in ("estimate", "lower", "upper", "valid_draw_share"):
            assert reordered["intervals"][metric][field] == pytest.approx(interval[field])
        assert interval["lower"] <= interval["upper"]

    serialized = json.dumps(first)
    assert "cluster-" not in serialized
    assert "resample_indices" not in serialized
    assert_aggregate_report_safe({"conditional_uncertainty": first})


def test_weighted_metrics_match_explicit_cluster_replication() -> None:
    target = np.asarray([0, 1, 0, 1, 1])
    scores = np.asarray([0.1, 0.8, 0.4, 0.8, 0.6])
    weights = np.asarray([2, 1, 0, 3, 2], dtype=float)
    weighted = _metric_values(
        target,
        scores,
        weights,
        train_prevalence=0.35,
        budget_share=0.25,
    )
    frequency = weights.astype(int)
    repeated = _metric_values(
        np.repeat(target, frequency),
        np.repeat(scores, frequency),
        np.ones(int(weights.sum())),
        train_prevalence=0.35,
        budget_share=0.25,
    )
    assert set(weighted) == set(repeated)
    for metric in weighted:
        assert weighted[metric] == pytest.approx(repeated[metric])


def test_degenerate_draws_are_counted_without_replacement() -> None:
    clusters = np.repeat([f"cluster-{index:02d}" for index in range(30)], 2)
    target = np.zeros(len(clusters), dtype=int)
    target[:2] = 1
    scores = np.linspace(0.05, 0.95, len(target))
    result = conditional_cluster_bootstrap(
        target,
        scores,
        clusters,
        train_prevalence=0.10,
        draws=120,
        seed=505,
        enforce_release_gate=False,
    )
    assert 0 < result["degenerate_class_draws"] < result["draws_attempted"]
    assert result["draws_attempted"] == 120
    assert result["draws_replenished"] == 0
    assert result["intervals"]["brier"]["valid_draws"] == 120
    assert result["intervals"]["average_precision_attention"]["valid_draws"] < 120
    assert result["release_gate"]["passed"] is None
    assert result["release_gate"]["criteria_met"] is False


def test_all_tied_scores_omit_budget_intervals() -> None:
    target, _, clusters = _fixture()
    result = conditional_cluster_bootstrap(
        target,
        np.full(len(target), 0.30),
        clusters,
        train_prevalence=0.30,
        draws=40,
        seed=606,
        enforce_release_gate=False,
    )
    assert result["ranking_at_budget_defined"] is False
    assert not any("budget" in metric for metric in result["intervals"])


def test_cluster_bootstrap_rejects_missing_or_single_cluster() -> None:
    with pytest.raises(ValueError, match="present"):
        conditional_cluster_bootstrap(
            [0, 1],
            [0.2, 0.8],
            ["a", None],
            train_prevalence=0.5,
            draws=20,
            enforce_release_gate=False,
        )
    with pytest.raises(ValueError, match="present non-empty strings"):
        conditional_cluster_bootstrap(
            [0, 1],
            [0.2, 0.8],
            ["a", pd.NA],
            train_prevalence=0.5,
            draws=20,
            enforce_release_gate=False,
        )
    with pytest.raises(ValueError, match="present non-empty strings"):
        conditional_cluster_bootstrap(
            [0, 1],
            [0.2, 0.8],
            ["a", 7],
            train_prevalence=0.5,
            draws=20,
            enforce_release_gate=False,
        )
    with pytest.raises(ValueError, match="at least two"):
        conditional_cluster_bootstrap(
            [0, 1],
            [0.2, 0.8],
            ["a", "a"],
            train_prevalence=0.5,
            draws=20,
            enforce_release_gate=False,
        )


def test_release_gate_requires_two_thousand_draws_inside_bootstrap() -> None:
    target, scores, clusters = _fixture()
    with pytest.raises(ValueError, match="at least 2000 draws"):
        conditional_cluster_bootstrap(
            target,
            scores,
            clusters,
            train_prevalence=0.28,
            draws=1_999,
            enforce_release_gate=True,
        )


@pytest.mark.parametrize("draws", (True, 20.5))
def test_cluster_bootstrap_rejects_non_integer_draw_count(draws: object) -> None:
    target, scores, clusters = _fixture()
    with pytest.raises(ValueError, match="draws must be an integer"):
        conditional_cluster_bootstrap(
            target,
            scores,
            clusters,
            train_prevalence=0.28,
            draws=draws,  # type: ignore[arg-type]
            enforce_release_gate=False,
        )


def test_cluster_bootstrap_reports_explicit_custom_cluster_unit() -> None:
    target, scores, clusters = _fixture()
    result = conditional_cluster_bootstrap(
        target,
        scores,
        clusters,
        train_prevalence=0.28,
        draws=20,
        cluster_unit="synthetic_session_group",
        enforce_release_gate=False,
    )
    assert result["cluster_unit"] == "synthetic_session_group"
