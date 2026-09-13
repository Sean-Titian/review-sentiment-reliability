"""Rolling-origin temporal reliability checks for the synthetic fixture."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from review_reliability.data import (
    RAW_COLUMNS,
    TARGET_COLUMN,
    derive_rating_proxy,
    make_split_keys,
)
from review_reliability.metrics import (
    aggregate_scalar_runs,
    classification_metrics,
    select_validation_threshold,
)
from review_reliability.modeling import (
    FEATURE_COLUMNS,
    fit_text_pipeline,
    predict_attention_probability,
)
from review_reliability.splits import (
    RollingOriginSpec,
    attach_manifest,
    enforce_protocol_isolation,
    make_rolling_origin_manifests,
)

TEMPORAL_MODEL_METRICS = (
    "attention_prevalence",
    "average_precision_attention",
    "brier",
    "ece_10_bins",
    "precision_attention",
    "recall_attention",
    "threshold",
    "budget_count",
    "budget_cutoff_tied_rows",
    "budget_cutoff_tied_weight",
    "budget_cutoff_fraction_selected",
    "precision_at_budget",
    "recall_at_budget",
    "lift_at_budget",
)
TEMPORAL_SUMMARY_METRICS = (
    "attention_prevalence",
    "average_precision_attention",
    "average_precision_minus_prevalence",
    "average_precision_over_prevalence",
    "brier",
    "brier_improvement_vs_train_prevalence",
    "recall_at_budget",
    "lift_at_budget",
)
TEMPORAL_PLACEBO_DRAWS_PER_WINDOW = 20


def _labeled_partition(attached: pd.DataFrame, name: str) -> pd.DataFrame:
    raw = attached.loc[attached["partition"] == name, list(RAW_COLUMNS)].copy()
    labeled = derive_rating_proxy(raw)
    if labeled[TARGET_COLUMN].nunique() != 2:
        raise ValueError(f"Rolling-origin {name} partition must contain both proxy classes")
    return labeled


def _time_bounds(keys: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, dict[str, str]]:
    joined = keys.loc[:, ["review_id", "event_time"]].merge(
        manifest,
        on="review_id",
        validate="one_to_one",
    )
    joined["event_time"] = pd.to_datetime(joined["event_time"], utc=True)
    return {
        partition: {
            "min": joined.loc[joined["partition"] == partition, "event_time"].min().isoformat(),
            "max": joined.loc[joined["partition"] == partition, "event_time"].max().isoformat(),
        }
        for partition in ("train", "validation", "test")
    }


def _cross_boundary_overlap(
    keys: pd.DataFrame,
    manifest: pd.DataFrame,
) -> dict[str, dict[str, int | float]]:
    joined = keys.merge(manifest, on="review_id", validate="one_to_one")
    output: dict[str, dict[str, int | float]] = {}
    test = joined.loc[joined["partition"] == "test"]
    for column in (
        "text_fingerprint",
        "near_text_fingerprint",
        "user_group",
        "product_group",
    ):
        train_values = set(joined.loc[joined["partition"] == "train", column])
        history_values = set(joined.loc[joined["partition"] != "test", column])
        train_overlap = test[column].isin(train_values)
        history_overlap = test[column].isin(history_values)
        output[column] = {
            "train_to_test_shared_groups": int(test.loc[train_overlap, column].nunique()),
            "train_to_test_affected_rows": int(train_overlap.sum()),
            "train_to_test_affected_row_share": float(train_overlap.mean()),
            "history_to_test_shared_groups": int(test.loc[history_overlap, column].nunique()),
            "history_to_test_affected_rows": int(history_overlap.sum()),
            "history_to_test_affected_row_share": float(history_overlap.mean()),
        }
    return output


def _model_summary(
    metrics: dict[str, float | int],
    *,
    train_prevalence_brier: float,
) -> dict[str, float | int]:
    prevalence = float(metrics["attention_prevalence"])
    average_precision = float(metrics["average_precision_attention"])
    output = {key: metrics[key] for key in TEMPORAL_MODEL_METRICS}
    output.update(
        {
            "average_precision_minus_prevalence": average_precision - prevalence,
            "average_precision_over_prevalence": average_precision / prevalence,
            "brier_improvement_vs_train_prevalence": (
                train_prevalence_brier - float(metrics["brier"])
            ),
        }
    )
    return output


def _test_alignment_placebos(
    target: pd.Series,
    scores: np.ndarray,
    *,
    seed: int,
    draws: int,
) -> tuple[dict[str, object], list[dict[str, float]]]:
    controls: list[dict[str, float]] = []
    for draw in range(draws):
        rng = np.random.default_rng(seed + draw * 104_729)
        placebo_target = rng.permutation(target.to_numpy())
        metrics = classification_metrics(
            placebo_target,
            scores,
            threshold=0.5,
            budget_share=0.10,
        )
        prevalence = float(metrics["attention_prevalence"])
        controls.append(
            {
                "roc_auc": float(metrics["roc_auc_supplementary"]),
                "average_precision_over_prevalence": (
                    float(metrics["average_precision_attention"]) / prevalence
                ),
                "lift_at_10pct_budget": float(metrics["lift_at_budget"]),
            }
        )
    summary = {
        metric: aggregate_scalar_runs([control[metric] for control in controls])
        for metric in controls[0]
    }
    return summary, controls


def _temporal_placebo_gate(
    controls: list[dict[str, float]],
    *,
    enforced: bool,
) -> dict[str, object]:
    thresholds = {
        "roc_auc": [0.45, 0.55],
        "average_precision_over_prevalence": [0.80, 1.30],
        "lift_at_10pct_budget": [0.75, 1.35],
    }
    summary = {
        metric: aggregate_scalar_runs([control[metric] for control in controls])
        for metric in controls[0]
    }
    passed = all(
        lower <= float(summary[metric]["mean"]) <= upper
        for metric, (lower, upper) in thresholds.items()
    )
    if enforced and not passed:
        raise RuntimeError(
            "Rolling-origin test-label placebos did not return to chance; "
            "publication must stop for investigation"
        )
    return {
        "passed": passed if enforced else None,
        "enforced": enforced,
        "draws": len(controls),
        "thresholds_for_aggregate_mean": thresholds,
        "summary": summary,
    }


def run_rolling_origin_backtest(
    frame: pd.DataFrame,
    split_keys: pd.DataFrame,
    *,
    spec: RollingOriginSpec | None = None,
    model_seed: int = 61_043,
    placebo_draws_per_window: int = TEMPORAL_PLACEBO_DRAWS_PER_WINDOW,
    enforce_placebo_gate: bool = True,
) -> dict[str, object]:
    """Fit expanding-window models and evaluate disjoint future periods."""

    spec = spec or RollingOriginSpec()
    if placebo_draws_per_window <= 0:
        raise ValueError("Rolling-origin evaluation requires test-label placebo draws")
    if enforce_placebo_gate and placebo_draws_per_window < 20:
        raise ValueError("Release-gated rolling origins require 20 placebos per window")
    expected_keys = make_split_keys(frame).sort_values("review_id").reset_index(drop=True)
    provided_keys = split_keys.sort_values("review_id").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(expected_keys, provided_keys)
    except AssertionError as exc:
        raise ValueError(
            "Rolling-origin split keys must be derived from the supplied raw frame"
        ) from exc
    manifests = make_rolling_origin_manifests(split_keys, spec)
    windows: list[dict[str, object]] = []
    all_test_ids: set[object] = set()
    prior_train_ids: set[object] = set()
    total_test_ids = 0
    training_histories_expanding = True
    time_ordered_windows: list[bool] = []
    all_placebo_controls: list[dict[str, float]] = []
    for index, manifest in enumerate(manifests):
        manifest_ids = set(manifest["review_id"])
        window_keys = split_keys.loc[split_keys["review_id"].isin(manifest_ids)].copy()
        if len(window_keys) != len(manifest):
            raise RuntimeError("Rolling-origin manifest does not match label-free split keys")
        enforce_protocol_isolation(window_keys, manifest, "forward_time")
        window_frame = frame.loc[frame["review_id"].isin(manifest_ids)].copy()
        if len(window_frame) != len(manifest):
            raise RuntimeError("Rolling-origin manifest does not match the supplied raw frame")
        attached = attach_manifest(window_frame, manifest)
        train = _labeled_partition(attached, "train")
        validation = _labeled_partition(attached, "validation")
        test = _labeled_partition(attached, "test")

        seed = model_seed
        diagnostic_seed = model_seed + index * 1_009
        model = fit_text_pipeline(
            train.loc[:, list(FEATURE_COLUMNS)],
            train[TARGET_COLUMN],
            seed=seed,
        )
        validation_scores = predict_attention_probability(
            model,
            validation.loc[:, list(FEATURE_COLUMNS)],
        )
        threshold = select_validation_threshold(validation[TARGET_COLUMN], validation_scores)
        test_scores = predict_attention_probability(
            model,
            test.loc[:, list(FEATURE_COLUMNS)],
        )
        model_metrics = classification_metrics(
            test[TARGET_COLUMN],
            test_scores,
            threshold=threshold,
            budget_share=0.10,
        )
        train_prevalence = float(train[TARGET_COLUMN].mean())
        prevalence_scores = np.full(len(test), train_prevalence)
        prevalence_metrics = classification_metrics(
            test[TARGET_COLUMN],
            prevalence_scores,
            threshold=0.5,
            budget_share=0.10,
        )
        random_rng = np.random.default_rng(diagnostic_seed + 10_003)
        random_scores = random_rng.random(len(test))
        random_metrics = classification_metrics(
            test[TARGET_COLUMN],
            random_scores,
            threshold=0.5,
            budget_share=0.10,
        )
        placebo_summary, placebo_controls = _test_alignment_placebos(
            test[TARGET_COLUMN],
            test_scores,
            seed=diagnostic_seed + 40_009,
            draws=placebo_draws_per_window,
        )
        window_placebo_gate = _temporal_placebo_gate(
            placebo_controls,
            enforced=enforce_placebo_gate,
        )
        all_placebo_controls.extend(placebo_controls)

        test_ids = set(manifest.loc[manifest["partition"] == "test", "review_id"])
        if all_test_ids & test_ids:
            raise RuntimeError("Rolling-origin test horizons overlap")
        all_test_ids |= test_ids
        total_test_ids += len(test_ids)
        train_ids = set(manifest.loc[manifest["partition"] == "train", "review_id"])
        expanding = not prior_train_ids or prior_train_ids < train_ids
        training_histories_expanding &= expanding
        if not expanding:
            raise RuntimeError("Rolling-origin training histories must strictly expand")
        prior_train_ids = train_ids
        bounds = _time_bounds(window_keys, manifest)
        time_ordered_windows.append(
            bounds["train"]["max"] < bounds["validation"]["min"]
            and bounds["validation"]["max"] < bounds["test"]["min"]
        )
        raw_counts = manifest["partition"].value_counts()
        labeled_counts = {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
        }
        windows.append(
            {
                "window": index + 1,
                "model_seed": seed,
                "time_bounds": bounds,
                "raw_partition_rows": {
                    name: int(raw_counts[name])
                    for name in ("train", "validation", "test")
                },
                "labeled_partition_rows": labeled_counts,
                "future_raw_rows_excluded": int(len(split_keys) - len(manifest)),
                "raw_cross_boundary_overlap_audit": _cross_boundary_overlap(
                    window_keys,
                    manifest,
                ),
                "model": _model_summary(
                    model_metrics,
                    train_prevalence_brier=float(prevalence_metrics["brier"]),
                ),
                "baselines": {
                    "train_prevalence_probability": {
                        "train_prevalence": train_prevalence,
                        "test_attention_prevalence": float(
                            prevalence_metrics["attention_prevalence"]
                        ),
                        "average_precision_attention": float(
                            prevalence_metrics["average_precision_attention"]
                        ),
                        "brier": float(prevalence_metrics["brier"]),
                    },
                    "seeded_random_score": {
                        "average_precision_attention": float(
                            random_metrics["average_precision_attention"]
                        ),
                        "brier": float(random_metrics["brier"]),
                        "recall_at_budget": float(random_metrics["recall_at_budget"]),
                        "lift_at_budget": float(random_metrics["lift_at_budget"]),
                    },
                },
                "test_label_alignment_placebo": {
                    "draws": placebo_draws_per_window,
                    "summary": placebo_summary,
                    "gate": window_placebo_gate,
                },
            }
        )

    metric_ranges = {
        metric: aggregate_scalar_runs(
            [float(window["model"][metric]) for window in windows]
        )
        for metric in TEMPORAL_SUMMARY_METRICS
    }
    placebo_gate = _temporal_placebo_gate(
        all_placebo_controls,
        enforced=enforce_placebo_gate,
    )
    return {
        "design": {
            **asdict(spec),
            "training_mode": "expanding",
            "boundary_basis": "label-free row shares snapped to whole timestamps",
            "test_horizons": "non-overlapping",
            "future_rows_after_test_horizon": (
                "excluded from fitting and threshold selection"
            ),
            "label_availability_gap": "not modeled; zero-gap assumption",
            "test_label_alignment_placebos_per_window": placebo_draws_per_window,
        },
        "windows": windows,
        "summary": {
            "windows": len(windows),
            "all_test_horizons_non_overlapping": len(all_test_ids) == total_test_ids,
            "all_partitions_strictly_time_ordered": all(time_ordered_windows),
            "all_training_histories_expanding": training_histories_expanding,
            "metric_ranges": metric_ranges,
            "range_interpretation": (
                "descriptive variation across correlated synthetic rolling origins, "
                "not a confidence interval"
            ),
            "test_label_alignment_placebo_gate": placebo_gate,
        },
        "limitations": [
            (
                "The backtest is synthetic and does not establish real-source temporal "
                "generalization."
            ),
            (
                "Time separation does not isolate recurring text, users, or products; "
                "cross-boundary overlap is reported per window."
            ),
            (
                "No outcome-label availability lag or embargo is modeled, so the design "
                "assumes prior-window proxy labels are available at each refit origin."
            ),
            "Rolling-origin ranges are not independent draws and are not confidence intervals.",
        ],
    }
