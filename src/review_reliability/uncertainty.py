"""Conditional cluster-bootstrap uncertainty for a frozen scoring rule."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from sklearn.metrics import average_precision_score

from review_reliability.metrics import (
    DEFAULT_CAPACITY_SHARES,
    PRIMARY_CAPACITY_SHARE,
    _validated_budget_share,
    _validated_capacity_shares,
    budget_metrics,
    capacity_curve_metrics,
)

DEFAULT_BOOTSTRAP_DRAWS = 2_000
RELEASE_CONFIDENCE_LEVEL = 0.95
RELEASE_CLUSTER_UNIT = "near_text_fingerprint"
BOOTSTRAP_METRICS = (
    "attention_prevalence",
    "average_precision_attention",
    "average_precision_minus_prevalence",
    "average_precision_over_prevalence",
    "brier",
    "brier_improvement_vs_train_prevalence",
    "precision_at_10pct_budget",
    "recall_at_10pct_budget",
    "lift_at_10pct_budget",
)
CAPACITY_INTERVAL_METRICS = (
    "precision_at_budget",
    "recall_at_budget",
    "lift_at_budget",
)
PRIMARY_CAPACITY_INTERVAL_ALIASES = {
    "precision_at_budget": "precision_at_10pct_budget",
    "recall_at_budget": "recall_at_10pct_budget",
    "lift_at_budget": "lift_at_10pct_budget",
}


def _validated_inputs(
    target: Iterable[int],
    scores: Iterable[float],
    clusters: Iterable[object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_target = np.asarray(list(target))
    probability = np.asarray(list(scores), dtype=float)
    raw_clusters = np.asarray(list(clusters), dtype=object)
    if (
        raw_target.ndim != 1
        or probability.ndim != 1
        or raw_clusters.ndim != 1
        or not (len(raw_target) == len(probability) == len(raw_clusters))
        or len(raw_target) == 0
    ):
        raise ValueError("target, scores, and clusters must be non-empty aligned vectors")
    try:
        numeric_target = raw_target.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("target must be binary") from exc
    if (
        not np.isfinite(numeric_target).all()
        or not set(np.unique(numeric_target)).issubset({0.0, 1.0})
        or np.unique(numeric_target).size != 2
    ):
        raise ValueError("point-estimate target must contain both binary classes")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("scores must be finite probabilities in [0, 1]")
    if any(
        not isinstance(value, str | np.str_) or not str(value).strip()
        for value in raw_clusters
    ):
        raise ValueError("cluster labels must be present non-empty strings")
    cluster_labels = np.asarray([str(value) for value in raw_clusters], dtype=str)
    if len(np.unique(cluster_labels)) < 2:
        raise ValueError("cluster bootstrap requires at least two clusters")
    return numeric_target.astype(int), probability, cluster_labels


def _metric_values(
    target: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    *,
    train_prevalence: float,
    budget_share: float,
) -> dict[str, float]:
    total_weight = float(weights.sum())
    prevalence = float(np.dot(weights, target) / total_weight)
    model_brier = float(np.dot(weights, (target - scores) ** 2) / total_weight)
    baseline_brier = float(
        np.dot(weights, (target - train_prevalence) ** 2) / total_weight
    )
    metrics = {
        "attention_prevalence": prevalence,
        "brier": model_brier,
        "brier_improvement_vs_train_prevalence": baseline_brier - model_brier,
    }
    if np.unique(target[weights > 0]).size != 2:
        return metrics

    average_precision = float(
        average_precision_score(target, scores, sample_weight=weights)
    )
    metrics.update(
        {
            "average_precision_attention": average_precision,
            "average_precision_minus_prevalence": average_precision - prevalence,
            "average_precision_over_prevalence": average_precision / prevalence,
        }
    )
    if np.unique(scores[weights > 0]).size > 1:
        budget = budget_metrics(
            target,
            scores,
            budget_share=budget_share,
            sample_weight=weights,
        )
        metrics.update(
            {
                "precision_at_10pct_budget": float(budget["precision_at_budget"]),
                "recall_at_10pct_budget": float(budget["recall_at_budget"]),
                "lift_at_10pct_budget": float(budget["lift_at_budget"]),
            }
        )
    return metrics


def conditional_cluster_bootstrap(
    target: Iterable[int],
    scores: Iterable[float],
    clusters: Iterable[object],
    *,
    train_prevalence: float,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = 73_031,
    confidence_level: float = RELEASE_CONFIDENCE_LEVEL,
    budget_share: float = PRIMARY_CAPACITY_SHARE,
    capacity_shares: Iterable[float] = DEFAULT_CAPACITY_SHARES,
    cluster_unit: str = RELEASE_CLUSTER_UNIT,
    enforce_release_gate: bool = True,
) -> dict[str, object]:
    """Return marginal percentile intervals from one-way pairs cluster resampling.

    The fitted scoring rule, manifest, and synthetic test fixture are held fixed.
    Cluster multiplicities become row weights, so no row, cluster key, resample
    index, prediction, or bootstrap draw enters the report.
    """

    y_true, probability, cluster_labels = _validated_inputs(target, scores, clusters)
    if not 0.0 < train_prevalence < 1.0:
        raise ValueError("train_prevalence must lie strictly between zero and one")
    if isinstance(draws, bool | np.bool_) or not isinstance(draws, int | np.integer):
        raise ValueError("draws must be an integer")
    if draws < 20:
        raise ValueError("cluster bootstrap requires at least 20 draws")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    confidence_level_exact_default = bool(
        math.isclose(
            confidence_level,
            RELEASE_CONFIDENCE_LEVEL,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    if enforce_release_gate and not confidence_level_exact_default:
        raise ValueError("Release-gated cluster bootstrap requires 95% confidence")
    validated_primary_share = _validated_budget_share(
        budget_share, name="budget_share"
    )
    if not math.isclose(
        validated_primary_share,
        PRIMARY_CAPACITY_SHARE,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("budget_share must equal the primary 10% capacity")
    primary_share = PRIMARY_CAPACITY_SHARE
    try:
        validated_capacity_shares = _validated_capacity_shares(capacity_shares)
    except TypeError as exc:
        raise ValueError("capacity_shares must be an iterable of real numbers") from exc
    if PRIMARY_CAPACITY_SHARE not in validated_capacity_shares:
        raise ValueError("capacity_shares must contain the primary 10% capacity")
    capacity_shares_exact_default = bool(
        validated_capacity_shares == DEFAULT_CAPACITY_SHARES
    )
    if enforce_release_gate and not capacity_shares_exact_default:
        raise ValueError(
            "Release-gated cluster bootstrap requires the exact default capacity shares"
        )
    if enforce_release_gate and draws < DEFAULT_BOOTSTRAP_DRAWS:
        raise ValueError(
            f"Release-gated cluster bootstrap requires at least "
            f"{DEFAULT_BOOTSTRAP_DRAWS} draws"
        )
    if not isinstance(cluster_unit, str) or not cluster_unit.strip():
        raise ValueError("cluster_unit must be a non-empty string")
    cluster_unit = cluster_unit.strip()
    cluster_unit_exact_default = cluster_unit == RELEASE_CLUSTER_UNIT
    if enforce_release_gate and not cluster_unit_exact_default:
        raise ValueError(
            "Release-gated cluster bootstrap requires near-text-fingerprint clusters"
        )

    unique_clusters = np.asarray(sorted(set(cluster_labels)), dtype=str)
    cluster_code = {value: index for index, value in enumerate(unique_clusters)}
    row_codes = np.asarray([cluster_code[value] for value in cluster_labels], dtype=int)
    cluster_sizes = np.bincount(row_codes, minlength=len(unique_clusters)).astype(float)
    cluster_count = int(len(unique_clusters))
    effective_clusters = float(cluster_sizes.sum() ** 2 / np.square(cluster_sizes).sum())
    largest_cluster_share = float(cluster_sizes.max() / cluster_sizes.sum())
    positive_rows = int(y_true.sum())
    negative_rows = int(len(y_true) - positive_rows)
    release_thresholds = {
        "minimum_draws": DEFAULT_BOOTSTRAP_DRAWS,
        "minimum_clusters": 30,
        "minimum_effective_clusters": 20.0,
        "maximum_largest_cluster_share": 0.20,
        "minimum_rows_per_class": 50,
        "minimum_valid_draw_share": 0.95,
    }
    structure_passed = bool(
        cluster_count >= release_thresholds["minimum_clusters"]
        and effective_clusters >= release_thresholds["minimum_effective_clusters"]
        and largest_cluster_share <= release_thresholds["maximum_largest_cluster_share"]
        and min(positive_rows, negative_rows) >= release_thresholds["minimum_rows_per_class"]
    )
    draw_count_passed = bool(draws >= release_thresholds["minimum_draws"])
    if enforce_release_gate and not structure_passed:
        raise RuntimeError("Conditional cluster-bootstrap structure gate failed")

    point_weights = np.ones(len(y_true), dtype=float)
    point = _metric_values(
        y_true,
        probability,
        point_weights,
        train_prevalence=train_prevalence,
        budget_share=primary_share,
    )
    ranking_at_budget_defined = bool(np.unique(probability).size > 1)
    point_capacity_points: list[dict[str, object]] = []
    if ranking_at_budget_defined:
        point_capacity_points = list(
            capacity_curve_metrics(
                y_true,
                probability,
                budget_shares=validated_capacity_shares,
                sample_weight=point_weights,
            )["points"]
        )
        primary_point = next(
            capacity_point
            for capacity_point in point_capacity_points
            if float(capacity_point["budget_share"]) == primary_share
        )
        for capacity_metric, legacy_metric in PRIMARY_CAPACITY_INTERVAL_ALIASES.items():
            if float(primary_point[capacity_metric]) != float(point[legacy_metric]):
                raise RuntimeError("Primary capacity point does not match its 10% alias")
    metrics_to_interval = tuple(metric for metric in BOOTSTRAP_METRICS if metric in point)
    sampled_values: dict[str, list[float]] = {
        metric: [] for metric in metrics_to_interval
    }
    sampled_capacity_values: dict[float, dict[str, list[float]]] = {
        share: {metric: [] for metric in CAPACITY_INTERVAL_METRICS}
        for share in validated_capacity_shares
    }
    degenerate_class_draws = 0
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        sampled_codes = rng.integers(0, cluster_count, size=cluster_count)
        multiplicity = np.bincount(sampled_codes, minlength=cluster_count).astype(float)
        weights = multiplicity[row_codes]
        values = _metric_values(
            y_true,
            probability,
            weights,
            train_prevalence=train_prevalence,
            budget_share=primary_share,
        )
        if "average_precision_attention" not in values:
            degenerate_class_draws += 1
        for metric, value in values.items():
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite bootstrap metric: {metric}")
            if metric in sampled_values:
                sampled_values[metric].append(value)
        if "precision_at_10pct_budget" in values:
            draw_capacity_points = capacity_curve_metrics(
                y_true,
                probability,
                budget_shares=validated_capacity_shares,
                sample_weight=weights,
            )["points"]
            for capacity_point in draw_capacity_points:
                share = float(capacity_point["budget_share"])
                for metric in CAPACITY_INTERVAL_METRICS:
                    value = float(capacity_point[metric])
                    if not math.isfinite(value):
                        raise RuntimeError(
                            f"Non-finite bootstrap capacity metric: {metric} at {share}"
                        )
                    sampled_capacity_values[share][metric].append(value)
            primary_draw_point = next(
                capacity_point
                for capacity_point in draw_capacity_points
                if float(capacity_point["budget_share"]) == primary_share
            )
            for capacity_metric, legacy_metric in (
                PRIMARY_CAPACITY_INTERVAL_ALIASES.items()
            ):
                if float(primary_draw_point[capacity_metric]) != float(
                    values[legacy_metric]
                ):
                    raise RuntimeError(
                        "Primary capacity draw does not match its 10% alias"
                    )

    alpha = 1.0 - confidence_level
    intervals: dict[str, object] = {}
    valid_share_passed = True
    all_points_within_intervals = True
    intervals_finite_and_ordered = True
    for metric in metrics_to_interval:
        values = np.asarray(sampled_values[metric], dtype=float)
        valid_draws = int(len(values))
        if valid_draws == 0:
            raise RuntimeError(f"No valid cluster-bootstrap draws for {metric}")
        valid_share = float(valid_draws / draws)
        if valid_share < release_thresholds["minimum_valid_draw_share"]:
            valid_share_passed = False
        lower, upper = np.quantile(
            values,
            [alpha / 2.0, 1.0 - alpha / 2.0],
            method="linear",
        )
        estimate = float(point[metric])
        interval_valid = bool(
            math.isfinite(estimate)
            and math.isfinite(float(lower))
            and math.isfinite(float(upper))
            and float(lower) <= float(upper)
        )
        intervals_finite_and_ordered &= interval_valid
        contains_point = bool(float(lower) <= estimate <= float(upper))
        all_points_within_intervals &= contains_point
        intervals[metric] = {
            "estimate": estimate,
            "lower": float(lower),
            "upper": float(upper),
            "valid_draws": valid_draws,
            "valid_draw_share": valid_share,
            "point_within_percentile_interval": contains_point,
        }

    capacity_curve_points: list[dict[str, object]] = []
    for capacity_point in point_capacity_points:
        share = float(capacity_point["budget_share"])
        conditional_intervals: dict[str, object] = {}
        for metric in CAPACITY_INTERVAL_METRICS:
            if share == primary_share:
                legacy_metric = PRIMARY_CAPACITY_INTERVAL_ALIASES[metric]
                conditional_intervals[metric] = dict(intervals[legacy_metric])
                continue
            sampled = np.asarray(sampled_capacity_values[share][metric], dtype=float)
            valid_draws = int(len(sampled))
            if valid_draws == 0:
                raise RuntimeError(
                    f"No valid cluster-bootstrap draws for {metric} at capacity {share}"
                )
            valid_share = float(valid_draws / draws)
            if valid_share < release_thresholds["minimum_valid_draw_share"]:
                valid_share_passed = False
            lower, upper = np.quantile(
                sampled,
                [alpha / 2.0, 1.0 - alpha / 2.0],
                method="linear",
            )
            estimate = float(capacity_point[metric])
            interval_valid = bool(
                math.isfinite(estimate)
                and math.isfinite(float(lower))
                and math.isfinite(float(upper))
                and float(lower) <= float(upper)
            )
            intervals_finite_and_ordered &= interval_valid
            contains_point = bool(float(lower) <= estimate <= float(upper))
            all_points_within_intervals &= contains_point
            conditional_intervals[metric] = {
                "estimate": estimate,
                "lower": float(lower),
                "upper": float(upper),
                "valid_draws": valid_draws,
                "valid_draw_share": valid_share,
                "point_within_percentile_interval": contains_point,
            }
        capacity_curve_points.append(
            {
                "budget_share": share,
                "point_estimates": {
                    key: value
                    for key, value in capacity_point.items()
                    if key != "budget_share"
                },
                "conditional_intervals": conditional_intervals,
            }
        )

    capacity_curve = {
        "pre_specified_budget_shares": list(validated_capacity_shares),
        "primary_budget_share": primary_share,
        "matches_release_capacity_contract": capacity_shares_exact_default,
        "capacity_selected_post_hoc": (
            False if capacity_shares_exact_default else None
        ),
        "capacity_selection_mode": (
            "release_contract_pre_specified"
            if capacity_shares_exact_default
            else "caller_supplied_diagnostic_unverified"
        ),
        "scope": (
            "fixed fitted scoring rule, split manifest, and authored synthetic test "
            "fixture under the same one-way pairs cluster resamples"
        ),
        "interval_scope": "marginal percentile, not simultaneous",
        "points": capacity_curve_points,
    }
    capacity_curve_points_complete = bool(
        ranking_at_budget_defined
        and len(capacity_curve_points) == len(validated_capacity_shares)
        and all(
            set(point["conditional_intervals"]) == set(CAPACITY_INTERVAL_METRICS)
            for point in capacity_curve_points
        )
    )

    release_criteria_met = bool(
        draw_count_passed
        and structure_passed
        and capacity_shares_exact_default
        and capacity_curve_points_complete
        and confidence_level_exact_default
        and cluster_unit_exact_default
        and valid_share_passed
        and intervals_finite_and_ordered
    )
    if enforce_release_gate and not release_criteria_met:
        raise RuntimeError("Conditional cluster-bootstrap interval gate failed")

    return {
        "method": "one-way pairs cluster bootstrap with fixed predictions",
        "cluster_unit": cluster_unit,
        "confidence_level": float(confidence_level),
        "interval_type": "marginal percentile, not simultaneous",
        "quantile_method": "linear",
        "draws_requested": int(draws),
        "draws_attempted": int(draws),
        "draws_replenished": 0,
        "degenerate_class_draws": degenerate_class_draws,
        "seed": int(seed),
        "test_rows_after_partition_local_neutral_exclusion": int(len(y_true)),
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
        "clusters": cluster_count,
        "effective_clusters": effective_clusters,
        "largest_cluster_share": largest_cluster_share,
        "ranking_at_budget_defined": ranking_at_budget_defined,
        "budget_metric_omission_rule": (
            "fixed-budget metrics are omitted when all positive-weight scores tie"
        ),
        "conditioning": (
            "fixed fitted scoring rule, split manifest, and authored synthetic test "
            "fixture after partition-local 3-star exclusion"
        ),
        "not_covered": (
            "retraining, threshold selection, crossed user/product dependence, generator or "
            "source uncertainty, temporal drift, and real-data generalization"
        ),
        "release_gate": {
            "enforced": bool(enforce_release_gate),
            "passed": release_criteria_met if enforce_release_gate else None,
            "criteria_met": release_criteria_met,
            "thresholds": release_thresholds,
            "draw_count_passed": draw_count_passed,
            "structure_passed": structure_passed,
            "capacity_shares_exact_default": capacity_shares_exact_default,
            "capacity_curve_points_complete": capacity_curve_points_complete,
            "required_capacity_shares": list(DEFAULT_CAPACITY_SHARES),
            "confidence_level_exact_default": confidence_level_exact_default,
            "required_confidence_level": RELEASE_CONFIDENCE_LEVEL,
            "cluster_unit_exact_default": cluster_unit_exact_default,
            "required_cluster_unit": RELEASE_CLUSTER_UNIT,
            "valid_draw_share_passed": valid_share_passed,
            "intervals_finite_and_ordered": intervals_finite_and_ordered,
            "all_points_within_intervals_diagnostic": all_points_within_intervals,
            "point_containment_is_not_a_validity_requirement": True,
        },
        "intervals": intervals,
        "capacity_curve": capacity_curve,
    }
