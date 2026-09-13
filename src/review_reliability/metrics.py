"""Minority-class, calibration, and fixed-budget evaluation metrics."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _as_valid_arrays(
    target: Iterable[int], scores: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    raw_target = np.asarray(list(target))
    probability = np.asarray(list(scores), dtype=float)
    if raw_target.ndim != 1 or probability.ndim != 1 or len(raw_target) != len(probability):
        raise ValueError("target and scores must be equal-length one-dimensional arrays")
    try:
        numeric_target = raw_target.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("target must be a non-empty binary array") from exc
    if (
        len(numeric_target) == 0
        or not np.isfinite(numeric_target).all()
        or not set(np.unique(numeric_target)).issubset({0.0, 1.0})
    ):
        raise ValueError("target must be a non-empty binary array")
    y_true = numeric_target.astype(int)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("scores must be finite probabilities in [0, 1]")
    return y_true, probability


def expected_calibration_error(
    target: Iterable[int], scores: Iterable[float], n_bins: int = 10
) -> float:
    """Compute equal-width expected calibration error."""

    y_true, probability = _as_valid_arrays(target, scores)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.minimum(np.digitize(probability, edges[1:-1], right=False), n_bins - 1)
    error = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if mask.any():
            error += float(mask.mean()) * abs(float(y_true[mask].mean() - probability[mask].mean()))
    return float(error)


def budget_metrics(
    target: Iterable[int],
    scores: Iterable[float],
    budget_share: float = 0.10,
    *,
    sample_weight: Iterable[float] | None = None,
) -> dict[str, float | int]:
    """Evaluate a top-score queue with fractional allocation at the cutoff tie.

    Fractional allocation reports the expected result of choosing uniformly
    within a tied cutoff bucket. This makes the metric invariant to input order
    instead of letting row order decide which equal-score reviews fit the budget.
    """

    if not 0.0 < budget_share <= 1.0:
        raise ValueError("budget_share must lie in (0, 1]")
    y_true, probability = _as_valid_arrays(target, scores)
    if sample_weight is None:
        weights = np.ones(len(y_true), dtype=float)
    else:
        weights = np.asarray(list(sample_weight), dtype=float)
        if weights.ndim != 1 or len(weights) != len(y_true):
            raise ValueError("sample_weight must match the one-dimensional target")
        if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("sample_weight must be finite, non-negative, and have positive mass")
        if not np.allclose(weights, np.round(weights), rtol=0.0, atol=1e-12):
            raise ValueError("sample_weight must contain integer frequency weights")
        weights = np.round(weights)

    total_weight = float(weights.sum())
    budget_count = max(1, int(math.ceil(total_weight * budget_share)))
    budget_weight = min(float(budget_count), total_weight)
    positive_weight = weights > 0
    order = np.argsort(-probability[positive_weight], kind="mergesort")
    ordered_scores = probability[positive_weight][order]
    ordered_target = y_true[positive_weight][order]
    ordered_weights = weights[positive_weight][order]
    cumulative_weight = np.cumsum(ordered_weights)
    cutoff_index = int(np.searchsorted(cumulative_weight, budget_weight, side="left"))
    cutoff_score = ordered_scores[cutoff_index]
    above_cutoff = ordered_scores > cutoff_score
    at_cutoff = ordered_scores == cutoff_score
    weight_above = float(ordered_weights[above_cutoff].sum())
    tied_weight = float(ordered_weights[at_cutoff].sum())
    remaining_weight = max(0.0, budget_weight - weight_above)
    tie_fraction = remaining_weight / tied_weight
    selected_positives = float(
        np.dot(ordered_weights[above_cutoff], ordered_target[above_cutoff])
        + tie_fraction * np.dot(ordered_weights[at_cutoff], ordered_target[at_cutoff])
    )
    positives = float(np.dot(weights, y_true))
    prevalence = float(positives / total_weight)
    precision = float(selected_positives / budget_weight)
    recall = float(selected_positives / positives) if positives else 0.0
    lift = float(precision / prevalence) if prevalence else 0.0
    return {
        "budget_share": float(budget_share),
        "budget_count": budget_count,
        "budget_cutoff_tied_rows": int(at_cutoff.sum()),
        "budget_cutoff_tied_weight": tied_weight,
        "budget_cutoff_fraction_selected": float(tie_fraction),
        "precision_at_budget": precision,
        "recall_at_budget": recall,
        "lift_at_budget": lift,
    }


def select_validation_threshold(target: Iterable[int], scores: Iterable[float]) -> float:
    """Choose an F1 threshold in O(n log n) using validation data only.

    Ties follow the original contract: maximize F1, then recall, then choose the
    smallest threshold. The vectorized implementation avoids scanning the full
    validation set once per unique score.
    """

    y_true, probability = _as_valid_arrays(target, scores)
    if np.unique(y_true).size != 2:
        raise ValueError("Validation target must contain both proxy classes")

    order = np.argsort(probability, kind="mergesort")
    sorted_scores = probability[order]
    sorted_target = y_true[order]
    positive_prefix = np.r_[0, np.cumsum(sorted_target)]
    total_positive = int(positive_prefix[-1])
    thresholds = np.unique(np.r_[0.0, probability, 1.0])
    first_selected = np.searchsorted(sorted_scores, thresholds, side="left")
    true_positive = total_positive - positive_prefix[first_selected]
    predicted_positive = len(y_true) - first_selected
    f1 = np.divide(
        2.0 * true_positive,
        predicted_positive + total_positive,
        out=np.zeros_like(thresholds, dtype=float),
        where=(predicted_positive + total_positive) != 0,
    )
    recall = true_positive / total_positive
    best_f1 = f1.max()
    f1_mask = f1 == best_f1
    best_recall = recall[f1_mask].max()
    return float(thresholds[f1_mask & (recall == best_recall)].min())


def classification_metrics(
    target: Iterable[int],
    scores: Iterable[float],
    *,
    threshold: float,
    budget_share: float = 0.10,
) -> dict[str, float | int]:
    """Return metrics oriented to the minority attention event."""

    y_true, probability = _as_valid_arrays(target, scores)
    if np.unique(y_true).size != 2:
        raise ValueError("Evaluation target must contain both proxy classes")
    predicted = (probability >= threshold).astype(int)
    average_precision = float(average_precision_score(y_true, probability))
    output: dict[str, float | int] = {
        "rows": int(len(y_true)),
        "attention_prevalence": float(y_true.mean()),
        "average_precision_attention": average_precision,
        # Compatibility alias for private adapters created before contract v1.1.
        # Public reports intentionally emit only ``average_precision_attention``.
        "pr_auc_attention": average_precision,
        "brier": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "ece_10_bins": expected_calibration_error(y_true, probability, n_bins=10),
        "precision_attention": float(precision_score(y_true, predicted, zero_division=0)),
        "recall_attention": float(recall_score(y_true, predicted, zero_division=0)),
        "f1_attention": float(f1_score(y_true, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "accuracy_supplementary": float(accuracy_score(y_true, predicted)),
        "roc_auc_supplementary": float(roc_auc_score(y_true, probability)),
        "threshold": float(threshold),
    }
    output.update(budget_metrics(y_true, probability, budget_share))
    return output


def aggregate_scalar_runs(values: list[float]) -> dict[str, float | int]:
    """Summarize descriptive multi-seed variation without claiming a confidence interval."""

    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Run values must be a non-empty finite sequence")
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "min": float(array.min()),
        "max": float(array.max()),
        "runs": int(array.size),
    }
