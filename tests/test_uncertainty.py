from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import review_reliability.uncertainty as uncertainty_module
from review_reliability.metrics import DEFAULT_CAPACITY_SHARES, PRIMARY_CAPACITY_SHARE
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
    assert first["capacity_curve"]["pre_specified_budget_shares"] == [0.05, 0.10, 0.20]
    assert first["capacity_curve"]["primary_budget_share"] == 0.10
    assert first["capacity_curve"]["matches_release_capacity_contract"] is True
    assert first["capacity_curve"]["capacity_selected_post_hoc"] is False
    assert first["capacity_curve"]["capacity_selection_mode"] == (
        "release_contract_pre_specified"
    )
    assert first["capacity_curve"]["interval_scope"] == (
        "marginal percentile, not simultaneous"
    )
    assert first["capacity_curve"] == reordered["capacity_curve"]
    for metric, interval in first["intervals"].items():
        for field in ("estimate", "lower", "upper", "valid_draw_share"):
            assert reordered["intervals"][metric][field] == pytest.approx(interval[field])
        assert interval["lower"] <= interval["upper"]

    serialized = json.dumps(first)
    assert "cluster-" not in serialized
    assert "resample_indices" not in serialized
    assert "bootstrap_draws" not in serialized
    assert_aggregate_report_safe({"conditional_uncertainty": first})


def test_capacity_curve_primary_point_and_intervals_are_exact_10pct_aliases() -> None:
    target, scores, clusters = _fixture()
    result = conditional_cluster_bootstrap(
        target,
        scores,
        clusters,
        train_prevalence=0.28,
        draws=80,
        seed=707,
        enforce_release_gate=False,
    )

    capacity_curve = result["capacity_curve"]
    points = capacity_curve["points"]
    assert [point["budget_share"] for point in points] == [0.05, 0.10, 0.20]
    assert len(points) == 3
    primary = next(
        point for point in points if point["budget_share"] == PRIMARY_CAPACITY_SHARE
    )
    aliases = {
        "precision_at_budget": "precision_at_10pct_budget",
        "recall_at_budget": "recall_at_10pct_budget",
        "lift_at_budget": "lift_at_10pct_budget",
    }
    for capacity_metric, legacy_metric in aliases.items():
        assert primary["point_estimates"][capacity_metric] == (
            result["intervals"][legacy_metric]["estimate"]
        )
        assert primary["conditional_intervals"][capacity_metric] == result["intervals"][
            legacy_metric
        ]

    expected_point_fields = {
        "budget_count",
        "budget_weight",
        "total_weight",
        "effective_budget_share",
        "budget_cutoff_tied_rows",
        "budget_cutoff_tied_weight",
        "budget_cutoff_fraction_selected",
        "precision_at_budget",
        "recall_at_budget",
        "lift_at_budget",
    }
    assert set(primary["point_estimates"]) == expected_point_fields
    assert set(primary["conditional_intervals"]) == set(aliases)


def test_capacity_curve_uses_one_joint_call_for_each_common_resample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clusters = np.repeat([f"cluster-{index:02d}" for index in range(30)], 2)
    target = np.tile([0, 1], 30)
    scores = np.linspace(0.01, 0.99, len(target))
    calls: list[tuple[tuple[float, ...], np.ndarray]] = []
    real_capacity_curve_metrics = uncertainty_module.capacity_curve_metrics

    def recording_capacity_curve_metrics(
        draw_target: np.ndarray,
        draw_scores: np.ndarray,
        budget_shares: tuple[float, ...],
        *,
        sample_weight: np.ndarray,
    ) -> dict[str, object]:
        calls.append((tuple(budget_shares), np.asarray(sample_weight).copy()))
        return real_capacity_curve_metrics(
            draw_target,
            draw_scores,
            budget_shares=budget_shares,
            sample_weight=sample_weight,
        )

    monkeypatch.setattr(
        uncertainty_module,
        "capacity_curve_metrics",
        recording_capacity_curve_metrics,
    )
    result = conditional_cluster_bootstrap(
        target,
        scores,
        clusters,
        train_prevalence=0.5,
        draws=20,
        seed=808,
        enforce_release_gate=False,
    )

    assert len(calls) == 21  # One point estimate plus one call per joint bootstrap draw.
    assert all(shares == DEFAULT_CAPACITY_SHARES for shares, _ in calls)
    for point in result["capacity_curve"]["points"]:
        for interval in point["conditional_intervals"].values():
            assert interval["valid_draws"] == 20
            assert interval["valid_draw_share"] == 1.0


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
    assert result["capacity_curve"]["points"] == []


def test_release_gate_rejects_tied_scores_without_complete_capacity_points() -> None:
    clusters = np.repeat([f"cluster-{index:02d}" for index in range(50)], 4)
    target = np.tile([0, 1, 0, 1], 50)
    scores = np.full(len(target), 0.5)

    with pytest.raises(RuntimeError, match="interval gate failed"):
        conditional_cluster_bootstrap(
            target,
            scores,
            clusters,
            train_prevalence=0.5,
            draws=2_000,
            enforce_release_gate=True,
        )


def test_diagnostic_capacity_curve_allows_custom_pre_specified_shares() -> None:
    target, scores, clusters = _fixture()
    custom_shares = (0.025, 0.10, 0.25, 0.50)
    result = conditional_cluster_bootstrap(
        target,
        scores,
        clusters,
        train_prevalence=0.28,
        draws=30,
        seed=909,
        capacity_shares=custom_shares,
        enforce_release_gate=False,
    )

    assert result["capacity_curve"]["pre_specified_budget_shares"] == list(
        custom_shares
    )
    assert [
        point["budget_share"] for point in result["capacity_curve"]["points"]
    ] == list(custom_shares)
    assert result["release_gate"]["capacity_shares_exact_default"] is False
    assert result["release_gate"]["criteria_met"] is False
    assert result["capacity_curve"]["matches_release_capacity_contract"] is False
    assert result["capacity_curve"]["capacity_selected_post_hoc"] is None
    assert result["capacity_curve"]["capacity_selection_mode"] == (
        "caller_supplied_diagnostic_unverified"
    )


def test_release_gate_requires_exact_default_capacity_shares() -> None:
    target, scores, clusters = _fixture()
    with pytest.raises(ValueError, match="exact default capacity shares"):
        conditional_cluster_bootstrap(
            target,
            scores,
            clusters,
            train_prevalence=0.28,
            draws=2_000,
            capacity_shares=(0.05, 0.10, 0.25),
            enforce_release_gate=True,
        )


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"confidence_level": 0.50}, "requires 95% confidence"),
        ({"cluster_unit": "product_group"}, "near-text-fingerprint clusters"),
    ),
)
def test_release_gate_locks_interval_level_and_cluster_unit(
    override: dict[str, object], message: str
) -> None:
    target, scores, clusters = _fixture()
    with pytest.raises(ValueError, match=message):
        conditional_cluster_bootstrap(
            target,
            scores,
            clusters,
            train_prevalence=0.28,
            draws=2_000,
            enforce_release_gate=True,
            **override,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("shares", "message"),
    (
        ((0.05, True, 0.20), "real numbers"),
        ((0.05, np.nan, 0.20), "finite"),
        ((0.05, np.inf, 0.20), "finite"),
        ((0.0, 0.10, 0.20), "lie in"),
        ((-0.05, 0.10, 0.20), "lie in"),
        ((0.05, 0.10, 1.01), "lie in"),
        ((0.05, 0.10, 0.10), "duplicates"),
        ((0.10, 0.05, 0.20), "strictly increasing"),
        ((0.05, 0.20), "primary 10%"),
        ((), "non-empty"),
        (None, "iterable"),
    ),
)
def test_cluster_bootstrap_rejects_invalid_capacity_sets(
    shares: object,
    message: str,
) -> None:
    target, scores, clusters = _fixture()
    with pytest.raises(ValueError, match=message):
        conditional_cluster_bootstrap(
            target,
            scores,
            clusters,
            train_prevalence=0.28,
            draws=20,
            capacity_shares=shares,  # type: ignore[arg-type]
            enforce_release_gate=False,
        )


@pytest.mark.parametrize(
    ("primary", "message"),
    (
        (True, "real numbers"),
        (np.nan, "finite"),
        (np.inf, "finite"),
        (0.0, "lie in"),
        (1.01, "lie in"),
        (0.20, "primary 10%"),
    ),
)
def test_cluster_bootstrap_rejects_invalid_primary_capacity_alias(
    primary: object,
    message: str,
) -> None:
    target, scores, clusters = _fixture()
    with pytest.raises(ValueError, match=message):
        conditional_cluster_bootstrap(
            target,
            scores,
            clusters,
            train_prevalence=0.28,
            draws=20,
            budget_share=primary,  # type: ignore[arg-type]
            enforce_release_gate=False,
        )


def test_primary_capacity_alias_preserves_legacy_numeric_tolerance() -> None:
    target, scores, clusters = _fixture()
    result = conditional_cluster_bootstrap(
        target,
        scores,
        clusters,
        train_prevalence=0.28,
        draws=20,
        budget_share=0.10 + 5e-13,
        enforce_release_gate=False,
    )

    assert result["capacity_curve"]["primary_budget_share"] == 0.10


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
