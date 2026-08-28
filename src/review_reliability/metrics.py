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
    target: Iterable[int], scores: Iterable[float], budget_share: float = 0.10
) -> dict[str, float | int]:
    """Evaluate a deterministic top-score review queue."""

    if not 0.0 < budget_share <= 1.0:
        raise ValueError("budget_share must lie in (0, 1]")
    y_true, probability = _as_valid_arrays(target, scores)
    budget_count = max(1, int(math.ceil(len(y_true) * budget_share)))
    order = np.argsort(-probability, kind="mergesort")
    selected = y_true[order[:budget_count]]
    positives = int(y_true.sum())
    selected_positives = int(selected.sum())
    prevalence = float(y_true.mean())
    precision = float(selected.mean())
    recall = float(selected_positives / positives) if positives else 0.0
    lift = float(precision / prevalence) if prevalence else 0.0
    return {
        "budget_share": float(budget_share),
        "budget_count": budget_count,
        "precision_at_budget": precision,
        "recall_at_budget": recall,
        "lift_at_budget": lift,
    }


def select_validation_threshold(target: Iterable[int], scores: Iterable[float]) -> float:
    """Choose an F1 threshold using validation data only."""

    y_true, probability = _as_valid_arrays(target, scores)
    if np.unique(y_true).size != 2:
        raise ValueError("Validation target must contain both proxy classes")
    candidates = np.unique(np.r_[0.0, probability, 1.0])
    best: tuple[float, float, float] | None = None
    for threshold in candidates:
        predicted = (probability >= threshold).astype(int)
        f1 = float(f1_score(y_true, predicted, zero_division=0))
        recall = float(recall_score(y_true, predicted, zero_division=0))
        candidate = (f1, recall, -float(threshold))
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    return float(-best[2])


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
    output: dict[str, float | int] = {
        "rows": int(len(y_true)),
        "attention_prevalence": float(y_true.mean()),
        "pr_auc_attention": float(average_precision_score(y_true, probability)),
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
