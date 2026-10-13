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

DEFAULT_BOOTSTRAP_DRAWS = 20_000
RELEASE_CONFIDENCE_LEVEL = 0.95
RELEASE_CLUSTER_UNIT = "near_text_fingerprint"
FAMILYWISE_CAPACITY_METRIC = "lift_at_budget"
FAMILYWISE_CAPACITY_METHOD = (
    "Bonferroni-adjusted percentile pairs cluster bootstrap band"
)
FAMILYWISE_CAPACITY_COVERAGE_INTERPRETATION = (
    "nominal Bonferroni-adjusted percentile-bootstrap family coverage"
)
FAMILYWISE_CAPACITY_SIZE = len(DEFAULT_CAPACITY_SHARES)
MINIMUM_EXPECTED_FAMILYWISE_TAIL_DRAWS = 100.0
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


def _familywise_capacity_lift_band(
    point_capacity_points: list[dict[str, object]],
    sampled_capacity_values: dict[float, dict[str, list[float]]],
    *,
    draws: int,
    confidence_level: float,
) -> dict[str, object]:
    """Build an aggregate-only familywise band for the three lift estimates.

    The family is deliberately narrow: lift at the pre-specified 5%, 10%, and
    20% capacities. Precision, recall, and cross-metric selection are outside
    the simultaneous-coverage claim. Every family member uses the same cluster
    multiplicity vector on each resample.
    """

    alpha = 1.0 - confidence_level
    tail_probability = alpha / (2.0 * FAMILYWISE_CAPACITY_SIZE)
    per_interval_confidence_level = 1.0 - alpha / FAMILYWISE_CAPACITY_SIZE
    point_by_share = {
        float(point["budget_share"]): point for point in point_capacity_points
    }
    family_points: list[dict[str, object]] = []
    valid_draw_counts: list[int] = []

    for share in DEFAULT_CAPACITY_SHARES:
        point = point_by_share.get(share)
        sampled = sampled_capacity_values.get(share, {}).get(
            FAMILYWISE_CAPACITY_METRIC, []
        )
        values = np.asarray(sampled, dtype=float)
        if point is None or values.size == 0:
            continue
        if not np.isfinite(values).all():
            raise RuntimeError(
                f"Non-finite familywise capacity metric at capacity {share}"
            )
        lower, upper = np.quantile(
            values,
            [tail_probability, 1.0 - tail_probability],
            method="linear",
        )
        estimate = float(point[FAMILYWISE_CAPACITY_METRIC])
        valid_draws = int(values.size)
        valid_draw_counts.append(valid_draws)
        family_points.append(
            {
                "budget_share": share,
                "estimate": estimate,
                "lower": float(lower),
                "upper": float(upper),
                "valid_draws": valid_draws,
                "valid_draw_share": float(valid_draws / draws),
                "point_within_familywise_interval": bool(
                    float(lower) <= estimate <= float(upper)
                ),
            }
        )

    common_valid_draws = bool(
        len(valid_draw_counts) == FAMILYWISE_CAPACITY_SIZE
        and len(set(valid_draw_counts)) == 1
    )
    minimum_valid_draws = min(valid_draw_counts, default=0)
    expected_samples_per_tail = float(minimum_valid_draws * tail_probability)
    return {
        "method": FAMILYWISE_CAPACITY_METHOD,
        "metric": FAMILYWISE_CAPACITY_METRIC,
        "family_confidence_level": float(confidence_level),
        "coverage_interpretation": FAMILYWISE_CAPACITY_COVERAGE_INTERPRETATION,
        "finite_sample_coverage_guarantee": False,
        "family_size": FAMILYWISE_CAPACITY_SIZE,
        "family_members": list(DEFAULT_CAPACITY_SHARES),
        "per_interval_confidence_level": float(per_interval_confidence_level),
        "two_sided_tail_probability_per_interval": float(tail_probability),
        "quantile_method": "linear",
        "draws_attempted": int(draws),
        "common_resamples": True,
        "common_valid_draws": common_valid_draws,
        "expected_samples_per_tail": expected_samples_per_tail,
        "minimum_expected_samples_per_tail": (
            MINIMUM_EXPECTED_FAMILYWISE_TAIL_DRAWS
        ),
        "simultaneous_scope": (
            "nominal familywise coverage for lift_at_budget across the pre-specified "
            "5%, 10%, and 20% capacities only"
        ),
        "covers_precision_or_recall": False,
        "cross_metric_simultaneous_coverage": False,
        "points": family_points,
    }


def _familywise_capacity_lift_release_gate(
    band: dict[str, object],
    marginal_capacity_points: list[dict[str, object]],
    *,
    enforced: bool,
) -> dict[str, object]:
    """Validate the frozen family definition and fail closed for releases."""

    expected_shares = list(DEFAULT_CAPACITY_SHARES)

    def finite_number(value: object) -> float | None:
        if isinstance(value, bool | np.bool_):
            return None
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return numeric if math.isfinite(numeric) else None

    band_confidence = finite_number(band.get("family_confidence_level"))
    per_interval_confidence = finite_number(
        band.get("per_interval_confidence_level")
    )
    tail_probability = finite_number(
        band.get("two_sided_tail_probability_per_interval")
    )
    expected_tail_probability = (1.0 - RELEASE_CONFIDENCE_LEVEL) / (
        2.0 * FAMILYWISE_CAPACITY_SIZE
    )
    expected_per_interval_confidence = 1.0 - (
        1.0 - RELEASE_CONFIDENCE_LEVEL
    ) / FAMILYWISE_CAPACITY_SIZE
    draws_attempted = finite_number(band.get("draws_attempted"))
    expected_samples_per_tail = finite_number(
        band.get("expected_samples_per_tail")
    )
    minimum_expected_samples = finite_number(
        band.get("minimum_expected_samples_per_tail")
    )

    raw_points = band.get("points")
    points = raw_points if isinstance(raw_points, list) else []
    point_by_share: dict[float, dict[str, object]] = {}
    point_shares_valid = True
    for point in points:
        if not isinstance(point, dict):
            point_shares_valid = False
            continue
        share = finite_number(point.get("budget_share"))
        if share is None or share in point_by_share:
            point_shares_valid = False
            continue
        point_by_share[share] = point
    family_points_complete = bool(
        point_shares_valid
        and len(points) == FAMILYWISE_CAPACITY_SIZE
        and list(point_by_share) == expected_shares
    )

    marginal_by_share: dict[float, dict[str, object]] = {}
    for point in marginal_capacity_points:
        if not isinstance(point, dict):
            continue
        share = finite_number(point.get("budget_share"))
        if share is not None:
            marginal_by_share[share] = point

    intervals_finite_and_ordered = family_points_complete
    valid_draw_share_passed = family_points_complete
    valid_draw_accounting_consistent = family_points_complete
    familywise_contains_marginal = family_points_complete
    point_estimates_match_marginal = family_points_complete
    point_containment_diagnostic = family_points_complete
    observed_valid_draw_counts: list[int] = []
    for share in DEFAULT_CAPACITY_SHARES:
        point = point_by_share.get(share)
        marginal_point = marginal_by_share.get(share)
        if point is None or marginal_point is None:
            intervals_finite_and_ordered = False
            valid_draw_share_passed = False
            valid_draw_accounting_consistent = False
            familywise_contains_marginal = False
            point_estimates_match_marginal = False
            point_containment_diagnostic = False
            continue
        estimate = finite_number(point.get("estimate"))
        lower = finite_number(point.get("lower"))
        upper = finite_number(point.get("upper"))
        valid_draws = finite_number(point.get("valid_draws"))
        valid_draw_share = finite_number(point.get("valid_draw_share"))
        interval_valid = bool(
            estimate is not None
            and lower is not None
            and upper is not None
            and lower <= upper
        )
        intervals_finite_and_ordered &= interval_valid
        if valid_draws is None or not valid_draws.is_integer():
            valid_draw_share_passed = False
        else:
            observed_valid_draw_counts.append(int(valid_draws))
        valid_draw_share_passed &= bool(
            valid_draw_share is not None and valid_draw_share >= 0.95
        )
        valid_draw_accounting_consistent &= bool(
            valid_draws is not None
            and valid_draws.is_integer()
            and draws_attempted is not None
            and draws_attempted.is_integer()
            and 0 < valid_draws <= draws_attempted
            and valid_draw_share is not None
            and math.isclose(
                valid_draw_share,
                valid_draws / draws_attempted,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        point_containment_diagnostic &= bool(
            interval_valid and lower <= estimate <= upper
        )

        marginal_intervals = marginal_point.get("conditional_intervals")
        marginal = (
            marginal_intervals.get(FAMILYWISE_CAPACITY_METRIC)
            if isinstance(marginal_intervals, dict)
            else None
        )
        marginal_lower = (
            finite_number(marginal.get("lower"))
            if isinstance(marginal, dict)
            else None
        )
        marginal_upper = (
            finite_number(marginal.get("upper"))
            if isinstance(marginal, dict)
            else None
        )
        marginal_estimate = (
            finite_number(marginal.get("estimate"))
            if isinstance(marginal, dict)
            else None
        )
        point_estimates_match_marginal &= bool(
            estimate is not None
            and marginal_estimate is not None
            and math.isclose(
                estimate,
                marginal_estimate,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        familywise_contains_marginal &= bool(
            interval_valid
            and marginal_lower is not None
            and marginal_upper is not None
            and lower <= marginal_lower + 1e-12
            and upper + 1e-12 >= marginal_upper
        )

    common_valid_draws = bool(
        band.get("common_valid_draws") is True
        and len(observed_valid_draw_counts) == FAMILYWISE_CAPACITY_SIZE
        and len(set(observed_valid_draw_counts)) == 1
    )
    common_resamples = band.get("common_resamples") is True
    draw_count_passed = bool(
        draws_attempted is not None
        and draws_attempted.is_integer()
        and draws_attempted >= DEFAULT_BOOTSTRAP_DRAWS
    )
    expected_tail_matches = bool(
        expected_samples_per_tail is not None
        and tail_probability is not None
        and observed_valid_draw_counts
        and math.isclose(
            expected_samples_per_tail,
            min(observed_valid_draw_counts) * tail_probability,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    tail_resolution_passed = bool(
        expected_samples_per_tail is not None
        and minimum_expected_samples is not None
        and math.isclose(
            minimum_expected_samples,
            MINIMUM_EXPECTED_FAMILYWISE_TAIL_DRAWS,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and expected_samples_per_tail >= minimum_expected_samples
    )

    checks = {
        "method_exact_default": band.get("method") == FAMILYWISE_CAPACITY_METHOD,
        "metric_exact_default": band.get("metric") == FAMILYWISE_CAPACITY_METRIC,
        "family_confidence_level_exact_default": bool(
            band_confidence is not None
            and math.isclose(
                band_confidence,
                RELEASE_CONFIDENCE_LEVEL,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ),
        "family_size_exact_default": band.get("family_size")
        == FAMILYWISE_CAPACITY_SIZE,
        "family_members_exact_default": band.get("family_members")
        == expected_shares,
        "per_interval_confidence_exact_default": bool(
            per_interval_confidence is not None
            and math.isclose(
                per_interval_confidence,
                expected_per_interval_confidence,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ),
        "tail_probability_exact_default": bool(
            tail_probability is not None
            and math.isclose(
                tail_probability,
                expected_tail_probability,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ),
        "quantile_method_exact_default": band.get("quantile_method") == "linear",
        "lift_only_scope_explicit": bool(
            band.get("covers_precision_or_recall") is False
            and band.get("cross_metric_simultaneous_coverage") is False
        ),
        "nominal_coverage_scope_explicit": bool(
            band.get("coverage_interpretation")
            == FAMILYWISE_CAPACITY_COVERAGE_INTERPRETATION
            and band.get("finite_sample_coverage_guarantee") is False
        ),
        "common_resamples": common_resamples,
        "common_valid_draws": common_valid_draws,
        "draw_count_passed": draw_count_passed,
        "family_points_complete": family_points_complete,
        "intervals_finite_and_ordered": intervals_finite_and_ordered,
        "valid_draw_share_passed": valid_draw_share_passed,
        "valid_draw_accounting_consistent": valid_draw_accounting_consistent,
        "point_estimates_match_marginal": point_estimates_match_marginal,
        "familywise_contains_marginal_intervals": familywise_contains_marginal,
        "expected_tail_samples_match_valid_draws": expected_tail_matches,
        "tail_resolution_passed": tail_resolution_passed,
    }
    criteria_met = bool(all(checks.values()))
    if enforced and not criteria_met:
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise RuntimeError(
            "Conditional cluster-bootstrap interval gate failed: "
            f"familywise capacity-lift checks failed ({failed})"
        )
    return {
        "enforced": bool(enforced),
        "passed": criteria_met if enforced else None,
        "criteria_met": criteria_met,
        "checks": checks,
        "required_method": FAMILYWISE_CAPACITY_METHOD,
        "required_metric": FAMILYWISE_CAPACITY_METRIC,
        "required_family_members": expected_shares,
        "required_family_confidence_level": RELEASE_CONFIDENCE_LEVEL,
        "minimum_draws": DEFAULT_BOOTSTRAP_DRAWS,
        "minimum_expected_samples_per_tail": (
            MINIMUM_EXPECTED_FAMILYWISE_TAIL_DRAWS
        ),
        "all_points_within_familywise_intervals_diagnostic": (
            point_containment_diagnostic
        ),
        "point_containment_is_not_a_validity_requirement": True,
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
    """Return marginal intervals and a lift-only familywise capacity band.

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

    familywise_lift_band = _familywise_capacity_lift_band(
        point_capacity_points,
        sampled_capacity_values,
        draws=draws,
        confidence_level=confidence_level,
    )
    familywise_lift_gate = _familywise_capacity_lift_release_gate(
        familywise_lift_band,
        capacity_curve_points,
        enforced=enforce_release_gate,
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
        "interval_scope": (
            "precision, recall, and ordinary lift intervals are marginal; "
            "familywise_lift_band is the only nominal simultaneous band and covers "
            "lift_at_budget at the pre-specified 5%, 10%, and 20% capacities only"
        ),
        "points": capacity_curve_points,
        "familywise_lift_band": familywise_lift_band,
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
        and familywise_lift_gate["criteria_met"]
    )
    if enforce_release_gate and not release_criteria_met:
        raise RuntimeError("Conditional cluster-bootstrap interval gate failed")

    return {
        "method": "one-way pairs cluster bootstrap with fixed predictions",
        "cluster_unit": cluster_unit,
        "confidence_level": float(confidence_level),
        "interval_type": (
            "marginal percentile intervals plus one nominal lift-only Bonferroni "
            "familywise capacity band"
        ),
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
            "familywise_capacity_lift_band_gate": familywise_lift_gate,
        },
        "intervals": intervals,
        "capacity_curve": capacity_curve,
    }
