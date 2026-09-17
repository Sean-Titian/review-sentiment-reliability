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
    DEFAULT_LABEL_DELAY_DAYS,
    RollingOriginSpec,
    attach_manifest,
    make_label_delay_rolling_manifests,
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


def _full_time_bounds(
    keys: pd.DataFrame,
    manifest: pd.DataFrame,
) -> dict[str, dict[str, str] | None]:
    """Return aggregate time bounds without exposing row identifiers."""

    joined = keys.loc[:, ["review_id", "event_time"]].merge(
        manifest,
        on="review_id",
        validate="one_to_one",
    )
    joined["event_time"] = pd.to_datetime(joined["event_time"], utc=True)
    output: dict[str, dict[str, str] | None] = {}
    for partition in ("train", "validation", "embargo", "test", "future"):
        timestamps = joined.loc[joined["partition"] == partition, "event_time"]
        output[partition] = (
            None
            if timestamps.empty
            else {"min": timestamps.min().isoformat(), "max": timestamps.max().isoformat()}
        )
    return output


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
        history_values = set(
            joined.loc[
                joined["partition"].isin(("train", "validation", "embargo")),
                column,
            ]
        )
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


def _manifest_contract(
    manifest: pd.DataFrame,
    split_keys: pd.DataFrame,
    *,
    delay_days: int,
    window: int,
) -> None:
    """Fail closed when a full label-delay manifest is incomplete or malformed."""

    if tuple(manifest.columns) != ("review_id", "partition"):
        raise RuntimeError("Label-delay manifest must contain only review_id and partition")
    if manifest["review_id"].isna().any() or manifest["review_id"].duplicated().any():
        raise RuntimeError("Label-delay manifest review IDs must be present and unique")
    if set(manifest["review_id"]) != set(split_keys["review_id"]):
        raise RuntimeError("Label-delay manifest must assign every split-key review ID")
    allowed = {"train", "validation", "embargo", "test", "future"}
    observed = set(manifest["partition"])
    if not observed <= allowed or not {"train", "validation", "test"} <= observed:
        raise RuntimeError("Label-delay manifest contains invalid or empty core partitions")
    if manifest.attrs.get("label_delay_days") != delay_days:
        raise RuntimeError("Label-delay manifest metadata does not match its scenario")
    if manifest.attrs.get("window") != window:
        raise RuntimeError("Label-delay manifest metadata does not match its window")


def _strict_partition_order(
    full_bounds: dict[str, dict[str, str] | None],
) -> bool:
    present = [
        full_bounds[name]
        for name in ("train", "validation", "embargo", "test", "future")
        if full_bounds[name] is not None
    ]
    return all(
        left["max"] < right["min"]  # type: ignore[index]
        for left, right in zip(present, present[1:], strict=False)
    )


def _run_label_delay_scenario(
    frame: pd.DataFrame,
    split_keys: pd.DataFrame,
    manifests: tuple[pd.DataFrame, ...],
    *,
    delay_days: int,
    model_seed: int,
    placebo_draws_per_window: int,
    enforce_placebo_gate: bool,
) -> tuple[dict[str, object], tuple[frozenset[object], ...]]:
    windows: list[dict[str, object]] = []
    test_id_sets: list[frozenset[object]] = []
    all_test_ids: set[object] = set()
    prior_train_ids: set[object] = set()
    total_test_ids = 0
    training_histories_expanding = True
    time_ordered_windows: list[bool] = []
    all_placebo_controls: list[dict[str, float]] = []

    for index, manifest in enumerate(manifests):
        window_number = index + 1
        _manifest_contract(
            manifest,
            split_keys,
            delay_days=delay_days,
            window=window_number,
        )
        attached = attach_manifest(frame, manifest)
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

        test_ids = frozenset(
            manifest.loc[manifest["partition"] == "test", "review_id"]
        )
        if all_test_ids & set(test_ids):
            raise RuntimeError("Rolling-origin test horizons overlap")
        all_test_ids |= set(test_ids)
        test_id_sets.append(test_ids)
        total_test_ids += len(test_ids)
        train_ids = set(manifest.loc[manifest["partition"] == "train", "review_id"])
        expanding = not prior_train_ids or prior_train_ids < train_ids
        training_histories_expanding &= expanding
        if not expanding:
            raise RuntimeError("Rolling-origin training histories must strictly expand")
        prior_train_ids = train_ids

        bounds = _time_bounds(split_keys, manifest)
        full_bounds = _full_time_bounds(split_keys, manifest)
        time_ordered_windows.append(_strict_partition_order(full_bounds))
        raw_counts = manifest["partition"].value_counts()
        complete_raw_counts = {
            name: int(raw_counts.get(name, 0))
            for name in ("train", "validation", "embargo", "test", "future")
        }
        if sum(complete_raw_counts.values()) != len(frame):
            raise RuntimeError("Label-delay raw partition counts do not conserve rows")

        maturity_cutoff = manifest.attrs.get("maturity_cutoff")
        test_start = manifest.attrs.get("test_start")
        if not isinstance(maturity_cutoff, str) or not isinstance(test_start, str):
            raise RuntimeError("Label-delay manifest is missing aggregate cutoff metadata")
        cutoff_timestamp = pd.to_datetime(maturity_cutoff, errors="coerce", utc=True)
        test_start_timestamp = pd.to_datetime(test_start, errors="coerce", utc=True)
        if pd.isna(cutoff_timestamp) or pd.isna(test_start_timestamp):
            raise RuntimeError("Label-delay cutoff metadata must contain valid timestamps")
        if cutoff_timestamp != test_start_timestamp - pd.Timedelta(days=delay_days):
            raise RuntimeError("Label-delay maturity cutoff does not match its scenario")
        if test_start_timestamp != pd.Timestamp(bounds["test"]["min"]):
            raise RuntimeError("Label-delay test-start metadata does not match the test rows")
        test_end = manifest.attrs.get("test_end")
        expected_test_end = (
            None
            if full_bounds["future"] is None
            else full_bounds["future"]["min"]
        )
        if test_end != expected_test_end:
            raise RuntimeError("Label-delay test-end metadata does not match future rows")
        actual_validation_rows = manifest.attrs.get("actual_validation_raw_rows")
        if actual_validation_rows != complete_raw_counts["validation"]:
            raise RuntimeError("Label-delay validation metadata does not match raw counts")
        if delay_days == 0 and complete_raw_counts["embargo"] != 0:
            raise RuntimeError("Zero-day label-delay scenario must not embargo rows")
        windows.append(
            {
                "window": window_number,
                "model_seed": seed,
                "label_delay_days": delay_days,
                "label_availability_cutoff": maturity_cutoff,
                "test_start": test_start,
                "test_end": test_end,
                "time_bounds": bounds,
                "full_time_bounds": full_bounds,
                "raw_partition_rows": complete_raw_counts,
                "labeled_partition_rows": {
                    "train": int(len(train)),
                    "validation": int(len(validation)),
                    "test": int(len(test)),
                },
                "target_validation_raw_rows": int(
                    manifest.attrs.get("target_validation_raw_rows", 0)
                ),
                "actual_validation_raw_rows": int(
                    actual_validation_rows
                ),
                "embargo_raw_rows_excluded": complete_raw_counts["embargo"],
                "future_raw_rows_excluded": complete_raw_counts["future"],
                "raw_cross_boundary_overlap_audit": _cross_boundary_overlap(
                    split_keys,
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
    summary = {
        "windows": len(windows),
        "all_test_horizons_non_overlapping": len(all_test_ids) == total_test_ids,
        "all_partitions_strictly_time_ordered": all(time_ordered_windows),
        "all_training_histories_expanding": training_histories_expanding,
        "metric_ranges": metric_ranges,
        "range_interpretation": (
            "descriptive variation across correlated synthetic rolling origins, "
            "not a confidence interval"
        ),
        "test_label_alignment_placebo_gate": _temporal_placebo_gate(
            all_placebo_controls,
            enforced=enforce_placebo_gate,
        ),
    }
    return (
        {
            "label_delay_days": delay_days,
            "authored_sensitivity_only": delay_days != 0,
            "windows": windows,
            "summary": summary,
        },
        tuple(test_id_sets),
    )


def _paired_deltas_vs_zero(
    scenarios: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    zero_windows = scenarios["0"]["windows"]
    if not isinstance(zero_windows, list):
        raise RuntimeError("Zero-delay scenario windows are malformed")
    output: dict[str, dict[str, object]] = {}
    for delay in DEFAULT_LABEL_DELAY_DAYS[1:]:
        delayed_windows = scenarios[str(delay)]["windows"]
        if not isinstance(delayed_windows, list) or len(delayed_windows) != len(zero_windows):
            raise RuntimeError("Label-delay scenarios must have paired rolling windows")
        paired_windows: list[dict[str, object]] = []
        by_metric: dict[str, list[float]] = {
            metric: [] for metric in TEMPORAL_SUMMARY_METRICS
        }
        for zero_window, delayed_window in zip(
            zero_windows,
            delayed_windows,
            strict=True,
        ):
            if zero_window["window"] != delayed_window["window"]:
                raise RuntimeError("Label-delay windows are not aligned")
            metric_deltas = {
                metric: float(delayed_window["model"][metric])
                - float(zero_window["model"][metric])
                for metric in TEMPORAL_SUMMARY_METRICS
            }
            for metric, value in metric_deltas.items():
                by_metric[metric].append(value)
            paired_windows.append(
                {"window": zero_window["window"], "metric_deltas": metric_deltas}
            )
        output[str(delay)] = {
            "reference_delay_days": 0,
            "comparison_delay_days": delay,
            "windows": paired_windows,
            "metric_delta_ranges": {
                metric: aggregate_scalar_runs(values)
                for metric, values in by_metric.items()
            },
            "interpretation": "delayed scenario minus the paired zero-day scenario",
        }
    return output


def run_rolling_origin_backtest(
    frame: pd.DataFrame,
    split_keys: pd.DataFrame,
    *,
    spec: RollingOriginSpec | None = None,
    model_seed: int = 61_043,
    placebo_draws_per_window: int = TEMPORAL_PLACEBO_DRAWS_PER_WINDOW,
    enforce_placebo_gate: bool = True,
) -> dict[str, object]:
    """Evaluate fixed test horizons under 0-, 14-, and 30-day label delays."""

    spec = spec or RollingOriginSpec()
    if placebo_draws_per_window <= 0:
        raise ValueError("Rolling-origin evaluation requires test-label placebo draws")
    if enforce_placebo_gate and placebo_draws_per_window != 20:
        raise ValueError("Release-gated rolling origins require 20 placebos per window")
    expected_keys = make_split_keys(frame).sort_values("review_id").reset_index(drop=True)
    provided_keys = split_keys.sort_values("review_id").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(expected_keys, provided_keys)
    except AssertionError as exc:
        raise ValueError(
            "Rolling-origin split keys must be derived from the supplied raw frame"
        ) from exc

    manifests_by_delay = make_label_delay_rolling_manifests(
        split_keys,
        spec,
        delay_days=DEFAULT_LABEL_DELAY_DAYS,
    )
    if tuple(manifests_by_delay) != DEFAULT_LABEL_DELAY_DAYS:
        raise RuntimeError("Label-delay evaluation must contain exactly 0, 14, and 30 days")

    scenarios: dict[str, dict[str, object]] = {}
    test_ids_by_delay: dict[int, tuple[frozenset[object], ...]] = {}
    for delay_days in DEFAULT_LABEL_DELAY_DAYS:
        scenario, test_ids = _run_label_delay_scenario(
            frame,
            split_keys,
            manifests_by_delay[delay_days],
            delay_days=delay_days,
            model_seed=model_seed,
            placebo_draws_per_window=placebo_draws_per_window,
            enforce_placebo_gate=enforce_placebo_gate,
        )
        scenarios[str(delay_days)] = scenario
        test_ids_by_delay[delay_days] = test_ids

    reference_test_ids = test_ids_by_delay[0]
    all_test_horizons_match_zero_day = not any(
        test_ids_by_delay[delay] != reference_test_ids
        for delay in DEFAULT_LABEL_DELAY_DAYS[1:]
    )
    if not all_test_horizons_match_zero_day:
        raise RuntimeError("Label-delay scenarios must evaluate identical test review IDs")
    for delay in DEFAULT_LABEL_DELAY_DAYS:
        scenarios[str(delay)]["all_test_horizons_match_zero_day"] = (
            test_ids_by_delay[delay] == reference_test_ids
        )

    zero_scenario = scenarios["0"]
    return {
        "design": {
            **asdict(spec),
            "training_mode": "expanding",
            "boundary_basis": "label-free row shares snapped to whole timestamps",
            "test_horizons": "non-overlapping and fixed across label-delay scenarios",
            "future_rows_after_test_horizon": (
                "excluded from fitting and threshold selection"
            ),
            "label_availability_gap": (
                "authored 0-, 14-, and 30-day sensitivity; availability timestamps "
                "were not observed"
            ),
            "test_label_alignment_placebos_per_window": placebo_draws_per_window,
        },
        "windows": zero_scenario["windows"],
        "summary": zero_scenario["summary"],
        "label_delay_sensitivity": {
            "delay_days": list(DEFAULT_LABEL_DELAY_DAYS),
            "scenarios": scenarios,
            "paired_deltas_vs_0": _paired_deltas_vs_zero(scenarios),
            "all_test_horizons_match_zero_day": all_test_horizons_match_zero_day,
            "delay_selected_post_hoc": False,
            "availability_time_observed": False,
            "operational_target_validated": False,
            "interpretation": (
                "The 14- and 30-day delays are authored sensitivity scenarios, "
                "not observed service-level agreements or operational recommendations."
            ),
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
                "Label-availability timestamps were not observed. The 14- and 30-day "
                "embargoes are authored stress scenarios, not validated operational SLAs."
            ),
            "Rolling-origin ranges are not independent draws and are not confidence intervals.",
        ],
    }
