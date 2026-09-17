"""End-to-end split-first reliability evaluation and aggregate report builder."""

from __future__ import annotations

import math
import platform
from collections.abc import Iterable
from dataclasses import asdict

import numpy as np
import pandas as pd
import sklearn

from review_reliability.audit import dataset_audit
from review_reliability.data import (
    RAW_COLUMNS,
    TARGET_COLUMN,
    SyntheticConfig,
    derive_rating_proxy,
    generate_synthetic_reviews,
    make_split_keys,
)
from review_reliability.metrics import (
    DEFAULT_CAPACITY_SHARES,
    PRIMARY_CAPACITY_SHARE,
    aggregate_scalar_runs,
    capacity_curve_metrics,
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
    PROTOCOLS,
    Protocol,
    SplitSpec,
    attach_manifest,
    enforce_protocol_isolation,
    make_split_manifest,
    manifest_metadata,
)
from review_reliability.stress import OOV_SENTINELS, prior_shift_slice, text_stress_cases
from review_reliability.temporal import (
    TEMPORAL_PLACEBO_DRAWS_PER_WINDOW,
    run_rolling_origin_backtest,
)
from review_reliability.uncertainty import (
    DEFAULT_BOOTSTRAP_DRAWS,
    conditional_cluster_bootstrap,
)

REPORT_METRICS = (
    "rows",
    "attention_prevalence",
    "average_precision_attention",
    "brier",
    "log_loss",
    "ece_10_bins",
    "precision_attention",
    "recall_attention",
    "f1_attention",
    "balanced_accuracy",
    "roc_auc_supplementary",
    "accuracy_supplementary",
    "threshold",
    "budget_count",
    "budget_cutoff_tied_rows",
    "budget_cutoff_tied_weight",
    "budget_cutoff_fraction_selected",
    "precision_at_budget",
    "recall_at_budget",
    "lift_at_budget",
)
CONSTANT_BASELINE_METRICS = tuple(
    metric
    for metric in REPORT_METRICS
    if metric
    not in {
        "budget_count",
        "budget_cutoff_tied_rows",
        "budget_cutoff_tied_weight",
        "budget_cutoff_fraction_selected",
        "precision_at_budget",
        "recall_at_budget",
        "lift_at_budget",
    }
)
STRESS_METRICS = (
    "rows",
    "attention_prevalence",
    "average_precision_attention",
    "brier",
    "recall_at_budget",
    "lift_at_budget",
)
PERMUTATION_DRAWS_PER_SPLIT = 5
CAPACITY_REPORT_METRICS = (
    "budget_count",
    "budget_weight",
    "total_weight",
    "effective_budget_share",
    "precision_at_budget",
    "recall_at_budget",
    "lift_at_budget",
)
PRIMARY_CAPACITY_ALIAS_METRICS = (
    "budget_count",
    "budget_cutoff_tied_rows",
    "budget_cutoff_tied_weight",
    "budget_cutoff_fraction_selected",
    "precision_at_budget",
    "recall_at_budget",
    "lift_at_budget",
)
CAPACITY_NULL_THRESHOLDS = (
    {
        "budget_share": 0.05,
        "per_draw": (0.00, 3.00),
        "aggregate_mean": (0.55, 1.50),
    },
    {
        "budget_share": 0.10,
        "per_draw": (0.25, 2.00),
        "aggregate_mean": (0.75, 1.35),
    },
    {
        "budget_share": 0.20,
        "per_draw": (0.40, 1.70),
        "aggregate_mean": (0.80, 1.25),
    },
)
LABEL_DELAY_RAW_PARTITIONS = ("train", "validation", "embargo", "test", "future")


def _partition_labeled(attached: pd.DataFrame, partition: str) -> pd.DataFrame:
    raw_partition = attached.loc[attached["partition"] == partition, list(RAW_COLUMNS)].copy()
    labeled = derive_rating_proxy(raw_partition)
    if labeled[TARGET_COLUMN].nunique() != 2:
        raise ValueError(f"{partition} partition must contain both proxy classes")
    return labeled


def _metric_subset(metrics: dict[str, float | int], keys: Iterable[str]) -> dict[str, float | int]:
    return {key: metrics[key] for key in keys}


def _capacity_points_by_share(curve: dict[str, object]) -> dict[float, dict[str, float | int]]:
    shares = curve.get("budget_shares")
    if shares != list(DEFAULT_CAPACITY_SHARES):
        raise RuntimeError("Capacity curve must use the pre-specified release capacities")
    points = curve.get("points")
    if not isinstance(points, list) or len(points) != len(DEFAULT_CAPACITY_SHARES):
        raise RuntimeError("Capacity curve is missing pre-specified points")
    output: dict[float, dict[str, float | int]] = {}
    for expected_share, point in zip(DEFAULT_CAPACITY_SHARES, points, strict=True):
        if not isinstance(point, dict):
            raise RuntimeError("Capacity point must be an aggregate mapping")
        observed_share = float(point.get("budget_share", float("nan")))
        if not np.isclose(observed_share, expected_share, rtol=0.0, atol=1e-12):
            raise RuntimeError("Capacity points must remain in pre-specified order")
        if any(metric not in point for metric in CAPACITY_REPORT_METRICS):
            raise RuntimeError("Capacity point is missing a required aggregate metric")
        output[expected_share] = point
    return output


def _assert_primary_capacity_alias(
    curve: dict[str, object], metrics: dict[str, float | int]
) -> None:
    primary = _capacity_points_by_share(curve)[PRIMARY_CAPACITY_SHARE]
    for metric in PRIMARY_CAPACITY_ALIAS_METRICS:
        if not math.isclose(
            float(primary[metric]),
            float(metrics[metric]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"Primary capacity point does not match legacy {metric}")


def _aggregate_capacity_curves(curves: list[dict[str, object]]) -> dict[str, object]:
    if not curves:
        raise RuntimeError("Capacity sensitivity requires at least one curve")
    point_maps = [_capacity_points_by_share(curve) for curve in curves]
    return {
        "pre_specified_budget_shares": list(DEFAULT_CAPACITY_SHARES),
        "primary_budget_share": PRIMARY_CAPACITY_SHARE,
        "matches_release_capacity_contract": True,
        "capacity_selected_post_hoc": False,
        "runs": len(curves),
        "points": [
            {
                "budget_share": share,
                "descriptive_ranges": {
                    metric: aggregate_scalar_runs(
                        [float(point_map[share][metric]) for point_map in point_maps]
                    )
                    for metric in CAPACITY_REPORT_METRICS
                },
            }
            for share in DEFAULT_CAPACITY_SHARES
        ],
        "interval_scope": "descriptive matched-seed variation, not a confidence interval",
    }


def _stress_metric_summary(
    metrics: dict[str, float | int],
    baseline: dict[str, float | int],
    scores: np.ndarray,
) -> dict[str, float | int | bool]:
    """Keep budget metrics only when the stress scores define a ranking."""

    summary: dict[str, float | int | bool] = _metric_subset(metrics, STRESS_METRICS)
    ranking_defined = bool(np.unique(scores).size > 1)
    if not ranking_defined:
        summary.pop("recall_at_budget")
        summary.pop("lift_at_budget")
    delta_metrics = ["average_precision_attention", "brier"]
    if ranking_defined:
        delta_metrics.extend(["recall_at_budget", "lift_at_budget"])
    for metric in delta_metrics:
        summary[f"delta_{metric}"] = float(metrics[metric]) - float(baseline[metric])
    summary["ranking_at_budget_defined"] = ranking_defined
    return summary


def _stress_evaluation(
    model: object,
    labeled_test: pd.DataFrame,
    threshold: float,
    baseline: dict[str, float | int],
    baseline_scores: np.ndarray,
    seed: int,
) -> dict[str, dict[str, float | int | bool]]:
    output: dict[str, dict[str, float | int | bool]] = {}
    y_test = labeled_test[TARGET_COLUMN]
    vocabulary = model.named_steps["tfidf"].vocabulary_
    if any(token in vocabulary for token in OOV_SENTINELS):
        raise RuntimeError("A declared OOV stress token appears in the fitted vocabulary")
    for name, features in text_stress_cases(labeled_test, seed).items():
        scores = predict_attention_probability(model, features)
        metrics = classification_metrics(y_test, scores, threshold=threshold, budget_share=0.10)
        summary = _stress_metric_summary(metrics, baseline, scores)
        assembled = model.named_steps["assemble_text"].transform(features)
        vectorizer = model.named_steps["tfidf"]
        preprocess = vectorizer.build_preprocessor()
        tokenize = vectorizer.build_tokenizer()
        token_count = 0
        oov_count = 0
        for document in assembled:
            tokens = tokenize(preprocess(document))
            token_count += len(tokens)
            oov_count += sum(token not in vocabulary for token in tokens)
        matrix = vectorizer.transform(assembled)
        summary["observed_oov_token_share"] = (
            float(oov_count / token_count) if token_count else 0.0
        )
        summary["zero_tfidf_row_share"] = float((matrix.getnnz(axis=1) == 0).mean())
        summary["mean_absolute_probability_change"] = float(
            np.mean(np.abs(scores - baseline_scores))
        )
        output[name] = summary

    shifted = prior_shift_slice(labeled_test, seed)
    shifted_scores = predict_attention_probability(
        model,
        shifted.loc[:, list(FEATURE_COLUMNS)],
    )
    shifted_metrics = classification_metrics(
        shifted[TARGET_COLUMN],
        shifted_scores,
        threshold=threshold,
        budget_share=0.10,
    )
    shifted_summary = _stress_metric_summary(
        shifted_metrics,
        baseline,
        shifted_scores,
    )
    output["higher_attention_prior"] = shifted_summary
    return output


def _conflict_retention_gate(
    frame: pd.DataFrame,
    split_keys: pd.DataFrame,
    partitions: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> dict[str, int | bool]:
    """Verify target-conflicting fingerprints survive partition-local labeling."""

    reference = derive_rating_proxy(frame)[["review_id", TARGET_COLUMN]].merge(
        split_keys[["review_id", "near_text_fingerprint"]],
        on="review_id",
        validate="one_to_one",
    )
    conflict_counts = reference.groupby("near_text_fingerprint")[TARGET_COLUMN].nunique()
    conflict_groups = conflict_counts.index[conflict_counts > 1]
    expected_ids = set(
        reference.loc[
            reference["near_text_fingerprint"].isin(conflict_groups),
            "review_id",
        ]
    )
    retained_ids = set(
        pd.concat(
            [partition["review_id"] for partition in partitions],
            ignore_index=True,
        )
    )
    retained_conflict_rows = len(expected_ids & retained_ids)
    passed = retained_conflict_rows == len(expected_ids)
    if not passed:
        raise RuntimeError("Target-conflicting fingerprint rows were removed after splitting")
    return {
        "expected_conflicting_rows": len(expected_ids),
        "retained_conflicting_rows": retained_conflict_rows,
        "passed": passed,
    }


def _run_protocol_once(
    frame: pd.DataFrame,
    split_keys: pd.DataFrame,
    protocol: Protocol,
    seed: int,
    *,
    permutation_draws: int,
    bootstrap_draws: int = 0,
    enforce_bootstrap_gate: bool = False,
) -> dict[str, object]:
    spec = SplitSpec(protocol=protocol, seed=seed)
    manifest = make_split_manifest(split_keys, spec)
    overlap = enforce_protocol_isolation(split_keys, manifest, protocol)
    attached = attach_manifest(frame, manifest)
    train = _partition_labeled(attached, "train")
    validation = _partition_labeled(attached, "validation")
    test = _partition_labeled(attached, "test")
    conflict_retention = _conflict_retention_gate(
        frame,
        split_keys,
        (train, validation, test),
    )

    model = fit_text_pipeline(
        train.loc[:, list(FEATURE_COLUMNS)], train[TARGET_COLUMN], seed=seed
    )
    validation_scores = predict_attention_probability(
        model, validation.loc[:, list(FEATURE_COLUMNS)]
    )
    threshold = select_validation_threshold(validation[TARGET_COLUMN], validation_scores)
    test_scores = predict_attention_probability(model, test.loc[:, list(FEATURE_COLUMNS)])
    model_metrics = classification_metrics(
        test[TARGET_COLUMN], test_scores, threshold=threshold, budget_share=0.10
    )
    model_capacity_curve = capacity_curve_metrics(test[TARGET_COLUMN], test_scores)
    _assert_primary_capacity_alias(model_capacity_curve, model_metrics)

    train_prevalence = float(train[TARGET_COLUMN].mean())
    prior_test = np.full(len(test), train_prevalence)
    prior_threshold = 0.5
    prior_metrics = classification_metrics(
        test[TARGET_COLUMN], prior_test, threshold=prior_threshold, budget_share=0.10
    )

    random_rng = np.random.default_rng(seed + 10_003)
    random_validation = random_rng.random(len(validation))
    random_test = random_rng.random(len(test))
    random_threshold = select_validation_threshold(validation[TARGET_COLUMN], random_validation)
    random_metrics = classification_metrics(
        test[TARGET_COLUMN], random_test, threshold=random_threshold, budget_share=0.10
    )
    random_capacity_curve = capacity_curve_metrics(test[TARGET_COLUMN], random_test)
    _assert_primary_capacity_alias(random_capacity_curve, random_metrics)

    conditional_uncertainty = None
    if bootstrap_draws:
        cluster_frame = test.loc[:, ["review_id"]].merge(
            split_keys.loc[:, ["review_id", "near_text_fingerprint"]],
            on="review_id",
            validate="one_to_one",
            sort=False,
        )
        conditional_uncertainty = conditional_cluster_bootstrap(
            test[TARGET_COLUMN],
            test_scores,
            cluster_frame["near_text_fingerprint"],
            train_prevalence=train_prevalence,
            draws=bootstrap_draws,
            seed=seed + 70_003,
            enforce_release_gate=enforce_bootstrap_gate,
        )

    partitions = {}
    for name, partition in (("train", train), ("validation", validation), ("test", test)):
        partitions[name] = {
            "labeled_rows": int(len(partition)),
            "attention_prevalence": float(partition[TARGET_COLUMN].mean()),
        }

    train_label_controls: list[dict[str, float | int]] = []
    test_alignment_placebos: list[dict[str, float | int]] = []
    train_label_capacity_controls: list[dict[str, object]] = []
    test_alignment_capacity_placebos: list[dict[str, object]] = []
    for draw in range(permutation_draws):
        permutation_rng = np.random.default_rng(seed + 30_011 + draw * 104_729)
        shuffled_target = permutation_rng.permutation(train[TARGET_COLUMN].to_numpy())
        shuffled_model = fit_text_pipeline(
            train.loc[:, list(FEATURE_COLUMNS)],
            pd.Series(shuffled_target, index=train.index),
            seed=seed + draw + 1,
        )
        shuffled_validation = predict_attention_probability(
            shuffled_model, validation.loc[:, list(FEATURE_COLUMNS)]
        )
        shuffled_threshold = select_validation_threshold(
            validation[TARGET_COLUMN], shuffled_validation
        )
        shuffled_test = predict_attention_probability(
            shuffled_model, test.loc[:, list(FEATURE_COLUMNS)]
        )
        train_label_controls.append(
            classification_metrics(
                test[TARGET_COLUMN],
                shuffled_test,
                threshold=shuffled_threshold,
                budget_share=0.10,
            )
        )
        train_label_capacity_curve = capacity_curve_metrics(
            test[TARGET_COLUMN], shuffled_test
        )
        _assert_primary_capacity_alias(
            train_label_capacity_curve, train_label_controls[-1]
        )
        train_label_capacity_controls.append(train_label_capacity_curve)

        alignment_rng = np.random.default_rng(seed + 40_009 + draw * 104_729)
        placebo_validation_target = alignment_rng.permutation(
            validation[TARGET_COLUMN].to_numpy()
        )
        placebo_test_target = alignment_rng.permutation(test[TARGET_COLUMN].to_numpy())
        placebo_threshold = select_validation_threshold(
            placebo_validation_target,
            validation_scores,
        )
        test_alignment_placebos.append(
            classification_metrics(
                placebo_test_target,
                test_scores,
                threshold=placebo_threshold,
                budget_share=0.10,
            )
        )
        test_alignment_capacity_curve = capacity_curve_metrics(
            placebo_test_target, test_scores
        )
        _assert_primary_capacity_alias(
            test_alignment_capacity_curve, test_alignment_placebos[-1]
        )
        test_alignment_capacity_placebos.append(test_alignment_capacity_curve)

    return {
        "manifest": manifest_metadata(split_keys, manifest, spec),
        "partitions": partitions,
        "overlap_audit": overlap,
        "conflict_retention_gate": conflict_retention,
        "model": _metric_subset(model_metrics, REPORT_METRICS),
        "model_capacity_curve": model_capacity_curve,
        "baselines": {
            "train_prevalence_probability_and_majority_decision": _metric_subset(
                prior_metrics, CONSTANT_BASELINE_METRICS
            ),
            "seeded_random_score": _metric_subset(random_metrics, REPORT_METRICS),
        },
        "seeded_random_capacity_curve": random_capacity_curve,
        "train_label_permutation_controls": [
            _metric_subset(control, REPORT_METRICS) for control in train_label_controls
        ],
        "test_label_alignment_placebos": [
            _metric_subset(control, REPORT_METRICS) for control in test_alignment_placebos
        ],
        "train_label_permutation_capacity_controls": train_label_capacity_controls,
        "test_label_alignment_capacity_placebos": test_alignment_capacity_placebos,
        "stress": _stress_evaluation(
            model,
            test,
            threshold,
            model_metrics,
            test_scores,
            seed,
        ),
        "conditional_cluster_bootstrap": conditional_uncertainty,
    }


def _aggregate_metrics(runs: list[dict[str, float | int]]) -> dict[str, dict[str, float | int]]:
    common_keys = set(runs[0])
    for run in runs[1:]:
        common_keys &= set(run)
    output = {}
    for key in sorted(common_keys):
        values = [float(run[key]) for run in runs]
        output[key] = aggregate_scalar_runs(values)
    return output


def _aggregate_overlap(runs: list[dict[str, object]]) -> dict[str, object]:
    first = runs[0]["overlap_audit"]
    assert isinstance(first, dict)
    output = {}
    for group_key in sorted(first):
        group_runs = []
        for run in runs:
            audit = run["overlap_audit"]
            assert isinstance(audit, dict)
            group_runs.append(audit[group_key])
        output[group_key] = _aggregate_metrics(group_runs)
    return output


def _aggregate_stress_runs(
    runs: list[dict[str, float | int | bool]],
) -> dict[str, object]:
    ranking_flags = [bool(run["ranking_at_budget_defined"]) for run in runs]
    numeric_runs = [
        {
            key: value
            for key, value in run.items()
            if key != "ranking_at_budget_defined"
        }
        for run in runs
    ]
    output: dict[str, object] = _aggregate_metrics(numeric_runs)
    output["ranking_at_budget_status"] = {
        "defined_in_all_runs": all(ranking_flags),
        "undefined_tie_runs": sum(not flag for flag in ranking_flags),
        "runs": len(ranking_flags),
        "rule": "budget ranking metrics are omitted when all stress scores tie",
    }
    return output


def _flatten_control_runs(
    runs: list[dict[str, object]],
    key: str,
) -> list[dict[str, float | int]]:
    flattened: list[dict[str, float | int]] = []
    for run in runs:
        controls = run[key]
        if not isinstance(controls, list):
            raise RuntimeError(f"Invalid negative-control payload: {key}")
        flattened.extend(controls)
    return flattened


def _flatten_capacity_control_runs(
    runs: list[dict[str, object]],
    key: str,
) -> list[dict[str, object]]:
    flattened: list[dict[str, object]] = []
    for run in runs:
        controls = run[key]
        if not isinstance(controls, list):
            raise RuntimeError(f"Invalid capacity-control payload: {key}")
        if not all(isinstance(control, dict) for control in controls):
            raise RuntimeError(f"Capacity controls must be aggregate mappings: {key}")
        flattened.extend(controls)
    return flattened


def _aggregate_protocol_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    partition_output = {}
    for partition in ("train", "validation", "test"):
        partition_runs = []
        for run in runs:
            partitions = run["partitions"]
            assert isinstance(partitions, dict)
            partition_runs.append(partitions[partition])
        partition_output[partition] = _aggregate_metrics(partition_runs)

    model_runs = [run["model"] for run in runs]
    baseline_output = {}
    first_baselines = runs[0]["baselines"]
    assert isinstance(first_baselines, dict)
    for baseline_name in first_baselines:
        baseline_runs = []
        for run in runs:
            baselines = run["baselines"]
            assert isinstance(baselines, dict)
            baseline_runs.append(baselines[baseline_name])
        baseline_output[baseline_name] = _aggregate_metrics(baseline_runs)

    stress_output = {}
    first_stress = runs[0]["stress"]
    assert isinstance(first_stress, dict)
    for stress_name in first_stress:
        stress_runs = []
        for run in runs:
            stresses = run["stress"]
            assert isinstance(stresses, dict)
            stress_runs.append(stresses[stress_name])
        stress_output[stress_name] = _aggregate_stress_runs(stress_runs)

    train_label_controls = _flatten_control_runs(runs, "train_label_permutation_controls")
    alignment_placebos = _flatten_control_runs(runs, "test_label_alignment_placebos")
    conflict_gates = [run["conflict_retention_gate"] for run in runs]
    if not all(isinstance(gate, dict) for gate in conflict_gates):
        raise RuntimeError("Conflict-retention gate evidence is missing")

    manifest = runs[0]["manifest"]
    assert isinstance(manifest, dict)
    result = {
        "partition_summary": partition_output,
        "overlap_audit": _aggregate_overlap(runs),
        "model": _aggregate_metrics(model_runs),
        "baselines": baseline_output,
        "baseline_notes": {
            "train_prevalence_probability_and_majority_decision": (
                "Uses the training prevalence as a constant probability and a fixed 0.50 "
                "decision threshold. Ranking-at-budget metrics are undefined because all "
                "scores tie, so they are intentionally omitted."
            ),
            "seeded_random_score": (
                "Uses independent seeded uniform scores; its validation threshold and ranking "
                "metrics are chance diagnostics."
            ),
        },
        "negative_control_train_label_permutation": (
            _aggregate_metrics(train_label_controls) if train_label_controls else None
        ),
        "placebo_test_label_alignment": (
            _aggregate_metrics(alignment_placebos) if alignment_placebos else None
        ),
        "negative_control_design": {
            "split_runs": len(runs),
            "train_label_permutation_draws": len(train_label_controls),
            "test_label_alignment_draws": len(alignment_placebos),
        },
        "conflict_retention_gate": {
            "all_runs_passed": all(bool(gate["passed"]) for gate in conflict_gates),
            "expected_conflicting_rows": _aggregate_metrics(
                [
                    {
                        "value": int(gate["expected_conflicting_rows"]),
                    }
                    for gate in conflict_gates
                ]
            )["value"],
            "retained_conflicting_rows": _aggregate_metrics(
                [
                    {
                        "value": int(gate["retained_conflicting_rows"]),
                    }
                    for gate in conflict_gates
                ]
            )["value"],
        },
        "stress": stress_output,
    }
    if "time_bounds" in manifest:
        result["time_bounds"] = manifest["time_bounds"]
    return result


def _control_summary(controls: list[dict[str, float | int]]) -> dict[str, object]:
    if not controls:
        raise RuntimeError("A release-gated benchmark requires negative-control draws")
    roc_values = [float(control["roc_auc_supplementary"]) for control in controls]
    ap_ratios = [
        float(control["average_precision_attention"])
        / float(control["attention_prevalence"])
        for control in controls
    ]
    budget_lifts = [float(control["lift_at_budget"]) for control in controls]
    return {
        "draws": len(controls),
        "roc_auc": aggregate_scalar_runs(roc_values),
        "average_precision_over_prevalence": aggregate_scalar_runs(ap_ratios),
        "lift_at_10pct_budget": aggregate_scalar_runs(budget_lifts),
    }


def _capacity_control_summary(curves: list[dict[str, object]]) -> dict[str, object]:
    if not curves:
        raise RuntimeError("A capacity null summary requires capacity curves")
    point_maps = [_capacity_points_by_share(curve) for curve in curves]
    return {
        "pre_specified_budget_shares": list(DEFAULT_CAPACITY_SHARES),
        "primary_budget_share": PRIMARY_CAPACITY_SHARE,
        "matches_release_capacity_contract": True,
        "capacity_selected_post_hoc": False,
        "draws": len(curves),
        "points": [
            {
                "budget_share": share,
                "lift_at_budget": aggregate_scalar_runs(
                    [float(point_map[share]["lift_at_budget"]) for point_map in point_maps]
                ),
            }
            for share in DEFAULT_CAPACITY_SHARES
        ],
    }


def _diagnostic_control_summary(
    runs: list[dict[str, object]],
    *,
    scalar_key: str,
    capacity_key: str,
) -> dict[str, object]:
    controls = _flatten_control_runs(runs, scalar_key)
    capacity_curves = _flatten_capacity_control_runs(runs, capacity_key)
    if len(controls) != len(capacity_curves):
        raise RuntimeError("Scalar and capacity diagnostic null draws do not align")
    for control, capacity_curve in zip(controls, capacity_curves, strict=True):
        _assert_primary_capacity_alias(capacity_curve, control)
    summary = _control_summary(controls)
    summary["capacity_lift"] = _capacity_control_summary(capacity_curves)
    return summary


def _enforce_negative_control_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    """Apply fixed heuristic sanity bounds to two null designs."""

    thresholds = {
        "per_draw": {
            "roc_auc": [0.30, 0.70],
            "average_precision_over_prevalence": [0.50, 1.80],
            "lift_at_10pct_budget": [0.25, 2.00],
        },
        "aggregate_mean": {
            "roc_auc": [0.45, 0.55],
            "average_precision_over_prevalence": [0.80, 1.30],
            "lift_at_10pct_budget": [0.75, 1.35],
        },
        "capacity_lift": [
            {
                "budget_share": item["budget_share"],
                "per_draw": list(item["per_draw"]),
                "aggregate_mean": list(item["aggregate_mean"]),
            }
            for item in CAPACITY_NULL_THRESHOLDS
        ],
    }
    designs = {
        "train_label_permutation": {
            "scalar_controls": _flatten_control_runs(
                runs, "train_label_permutation_controls"
            ),
            "capacity_curves": _flatten_capacity_control_runs(
                runs, "train_label_permutation_capacity_controls"
            ),
        },
        "test_label_alignment_placebo": {
            "scalar_controls": _flatten_control_runs(
                runs, "test_label_alignment_placebos"
            ),
            "capacity_curves": _flatten_capacity_control_runs(
                runs, "test_label_alignment_capacity_placebos"
            ),
        },
    }
    summaries: dict[str, object] = {}
    for design, payload in designs.items():
        controls = payload["scalar_controls"]
        capacity_curves = payload["capacity_curves"]
        if len(controls) != len(capacity_curves):
            raise RuntimeError(f"{design} scalar and capacity null draws do not align")
        for index, (control, capacity_curve) in enumerate(
            zip(controls, capacity_curves, strict=True)
        ):
            _assert_primary_capacity_alias(capacity_curve, control)
            prevalence = float(control["attention_prevalence"])
            average_precision = float(control["average_precision_attention"])
            roc_auc = float(control["roc_auc_supplementary"])
            budget_lift = float(control["lift_at_budget"])
            values = np.asarray(
                [prevalence, average_precision, roc_auc, budget_lift], dtype=float
            )
            ap_ratio = average_precision / prevalence if prevalence > 0 else float("inf")
            if (
                not np.isfinite(values).all()
                or not thresholds["per_draw"]["roc_auc"][0]
                <= roc_auc
                <= thresholds["per_draw"]["roc_auc"][1]
                or not thresholds["per_draw"]["average_precision_over_prevalence"][0]
                <= ap_ratio
                <= thresholds["per_draw"]["average_precision_over_prevalence"][1]
                or not thresholds["per_draw"]["lift_at_10pct_budget"][0]
                <= budget_lift
                <= thresholds["per_draw"]["lift_at_10pct_budget"][1]
            ):
                raise RuntimeError(
                    f"{design} draw {index} did not return sufficiently close to chance; "
                    "publication must stop for investigation"
                )
            capacity_points = _capacity_points_by_share(capacity_curve)
            for capacity_threshold in CAPACITY_NULL_THRESHOLDS:
                share = float(capacity_threshold["budget_share"])
                capacity_lift = float(capacity_points[share]["lift_at_budget"])
                lower, upper = capacity_threshold["per_draw"]
                if not math.isfinite(capacity_lift) or not lower <= capacity_lift <= upper:
                    raise RuntimeError(
                        f"{design} draw {index} capacity {share:.0%} did not return "
                        "sufficiently close to chance; publication must stop for investigation"
                    )

        summary = _control_summary(controls)
        for metric in (
            "roc_auc",
            "average_precision_over_prevalence",
            "lift_at_10pct_budget",
        ):
            mean = float(summary[metric]["mean"])
            lower, upper = thresholds["aggregate_mean"][metric]
            if not lower <= mean <= upper:
                raise RuntimeError(
                    f"{design} aggregate {metric} did not return to chance; "
                    "publication must stop for investigation"
                )
        capacity_summary = _capacity_control_summary(capacity_curves)
        for point, capacity_threshold in zip(
            capacity_summary["points"], CAPACITY_NULL_THRESHOLDS, strict=True
        ):
            mean = float(point["lift_at_budget"]["mean"])
            lower, upper = capacity_threshold["aggregate_mean"]
            if not lower <= mean <= upper:
                share = float(point["budget_share"])
                raise RuntimeError(
                    f"{design} aggregate lift at {share:.0%} capacity did not return to "
                    "chance; publication must stop for investigation"
                )
        summary["capacity_lift"] = capacity_summary
        summaries[design] = summary

    return {
        "passed": True,
        "enforced": True,
        "gate_interpretation": (
            "pre-specified fail-closed heuristic sanity bounds; not p-values, a "
            "multiplicity-adjusted hypothesis test, or a calibrated joint envelope"
        ),
        "multiplicity_controlled": False,
        "permutation_draws_per_split_fixed": PERMUTATION_DRAWS_PER_SPLIT,
        "thresholds": thresholds,
        "designs": summaries,
    }


def _protocol_sensitivity_ladder(
    protocol_reports: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    roles = {
        "row_random": ("naive reference", "no duplicate or entity isolation"),
        "fingerprint_group": (
            "duplicate-isolated strict benchmark",
            "normalized exact and token-set fingerprint isolation",
        ),
        "user_group": ("new-user view", "user-identity isolation"),
        "product_group": ("new-product view", "product-identity isolation"),
        "forward_time": ("future-period view", "future-period temporal separation"),
    }
    naive_model = protocol_reports["row_random"]["model"]
    naive_ap = float(naive_model["average_precision_attention"]["mean"])
    naive_prevalence = float(naive_model["attention_prevalence"]["mean"])
    naive_ap_lift = naive_ap / naive_prevalence
    output = []
    for protocol in PROTOCOLS:
        if protocol not in protocol_reports:
            continue
        model = protocol_reports[protocol]["model"]
        average_precision = float(model["average_precision_attention"]["mean"])
        prevalence = float(model["attention_prevalence"]["mean"])
        ap_lift = average_precision / prevalence
        output.append(
            {
                "protocol": protocol,
                "role": roles[protocol][0],
                "risk_view": roles[protocol][1],
                "average_precision": average_precision,
                "attention_prevalence": prevalence,
                "average_precision_over_prevalence": ap_lift,
                "average_precision_difference_vs_row_random": average_precision - naive_ap,
                "ap_lift_difference_vs_row_random": ap_lift - naive_ap_lift,
                "runs": int(model["average_precision_attention"]["runs"]),
            }
        )
    return output


def _paired_strict_gap(raw_runs: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    row_runs = raw_runs["row_random"]
    strict_runs = raw_runs["fingerprint_group"]
    if len(row_runs) != len(strict_runs):
        raise RuntimeError("Paired strict-gap summary requires matching evaluation seeds")
    ap_differences = []
    ap_lift_differences = []
    for row_run, strict_run in zip(row_runs, strict_runs, strict=True):
        row_model = row_run["model"]
        strict_model = strict_run["model"]
        row_ap = float(row_model["average_precision_attention"])
        strict_ap = float(strict_model["average_precision_attention"])
        row_prevalence = float(row_model["attention_prevalence"])
        strict_prevalence = float(strict_model["attention_prevalence"])
        ap_differences.append(strict_ap - row_ap)
        ap_lift_differences.append(
            strict_ap / strict_prevalence - row_ap / row_prevalence
        )
    return {
        "comparison": "fingerprint_group minus row_random on matched seeds",
        "average_precision_difference": aggregate_scalar_runs(ap_differences),
        "ap_lift_difference": aggregate_scalar_runs(ap_lift_differences),
        "sign_convention": (
            "Negative values mean the fingerprint-isolated view scored lower; positive values "
            "mean it scored higher. Neither direction is required on the synthetic fixture."
        ),
    }


def _label_delay_release_gate(
    rolling_origin: dict[str, object],
    *,
    expected_source_rows: int,
    enforced: bool,
) -> dict[str, object]:
    """Validate the public aggregate contract for label-delay sensitivity."""

    sensitivity = rolling_origin.get("label_delay_sensitivity")
    if not isinstance(sensitivity, dict):
        raise RuntimeError("Rolling-origin output is missing label-delay sensitivity")
    scenarios = sensitivity.get("scenarios")
    if not isinstance(scenarios, dict):
        raise RuntimeError("Label-delay sensitivity is missing scenario mappings")

    expected_delays = list(DEFAULT_LABEL_DELAY_DAYS)
    expected_scenario_keys = {str(delay) for delay in DEFAULT_LABEL_DELAY_DAYS}
    zero_scenario = scenarios.get("0")
    zero_windows = zero_scenario.get("windows") if isinstance(zero_scenario, dict) else None
    zero_summary = zero_scenario.get("summary") if isinstance(zero_scenario, dict) else None
    reference_window_count = len(zero_windows) if isinstance(zero_windows, list) else 0

    checks: dict[str, bool] = {
        "delay_days_exact_release_grid": sensitivity.get("delay_days") == expected_delays,
        "scenario_keys_exact_release_grid": set(scenarios) == expected_scenario_keys,
        "same_test_review_ids_as_zero_day": (
            sensitivity.get("all_test_horizons_match_zero_day") is True
        ),
        "delay_selected_post_hoc_false": (
            sensitivity.get("delay_selected_post_hoc") is False
        ),
        "availability_time_observed_false": (
            sensitivity.get("availability_time_observed") is False
        ),
        "operational_target_validated_false": (
            sensitivity.get("operational_target_validated") is False
        ),
        "zero_day_top_level_windows_compatible": (
            isinstance(zero_windows, list)
            and rolling_origin.get("windows") == zero_windows
        ),
        "zero_day_top_level_summary_compatible": (
            isinstance(zero_summary, dict)
            and rolling_origin.get("summary") == zero_summary
        ),
        "paired_delta_keys_exact": (
            isinstance(sensitivity.get("paired_deltas_vs_0"), dict)
            and set(sensitivity["paired_deltas_vs_0"]) == {"14", "30"}
        ),
        "scenario_delay_labels_match": True,
        "delayed_scenarios_authored_only": True,
        "scenario_window_counts_match_zero_day": reference_window_count > 0,
        "scenario_temporal_integrity_release_ready": True,
        "test_horizon_metadata_match_zero_day": reference_window_count > 0,
        "raw_partition_count_keys_complete": True,
        "raw_partition_counts_conserve_source_rows": True,
        "window_placebo_gates_present": True,
        "window_placebo_gates_release_ready": True,
        "window_placebo_draws_exact_release_count": True,
        "pooled_placebo_gates_present": True,
        "pooled_placebo_gates_release_ready": True,
        "pooled_placebo_draws_complete": True,
    }

    reference_horizons: list[dict[str, object]] = []
    if isinstance(zero_windows, list):
        for window in zero_windows:
            if not isinstance(window, dict):
                checks["test_horizon_metadata_match_zero_day"] = False
                continue
            raw_counts = window.get("raw_partition_rows")
            bounds = window.get("time_bounds")
            reference_horizons.append(
                {
                    "window": window.get("window"),
                    "test_start": window.get("test_start"),
                    "test_end": window.get("test_end"),
                    "test_raw_rows": (
                        raw_counts.get("test") if isinstance(raw_counts, dict) else None
                    ),
                    "test_time_bounds": (
                        bounds.get("test") if isinstance(bounds, dict) else None
                    ),
                }
            )

    observed_window_gates = 0
    observed_pooled_gates = 0
    observed_raw_count_sets = 0
    for delay in DEFAULT_LABEL_DELAY_DAYS:
        scenario = scenarios.get(str(delay))
        if not isinstance(scenario, dict):
            for key in (
                "scenario_delay_labels_match",
                "delayed_scenarios_authored_only",
                "scenario_window_counts_match_zero_day",
                "scenario_temporal_integrity_release_ready",
                "test_horizon_metadata_match_zero_day",
                "raw_partition_count_keys_complete",
                "raw_partition_counts_conserve_source_rows",
                "window_placebo_gates_present",
                "window_placebo_gates_release_ready",
                "window_placebo_draws_exact_release_count",
                "pooled_placebo_gates_present",
                "pooled_placebo_gates_release_ready",
                "pooled_placebo_draws_complete",
            ):
                checks[key] = False
            continue

        checks["scenario_delay_labels_match"] &= scenario.get("label_delay_days") == delay
        checks["same_test_review_ids_as_zero_day"] &= (
            scenario.get("all_test_horizons_match_zero_day") is True
        )
        checks["delayed_scenarios_authored_only"] &= (
            scenario.get("authored_sensitivity_only") is (delay != 0)
        )
        windows = scenario.get("windows")
        summary = scenario.get("summary")
        if not isinstance(windows, list) or not isinstance(summary, dict):
            checks["scenario_window_counts_match_zero_day"] = False
            checks["scenario_temporal_integrity_release_ready"] = False
            checks["test_horizon_metadata_match_zero_day"] = False
            checks["window_placebo_gates_present"] = False
            checks["window_placebo_gates_release_ready"] = False
            checks["window_placebo_draws_exact_release_count"] = False
            checks["pooled_placebo_gates_present"] = False
            checks["pooled_placebo_gates_release_ready"] = False
            checks["pooled_placebo_draws_complete"] = False
            continue
        checks["scenario_window_counts_match_zero_day"] &= (
            len(windows) == reference_window_count
            and summary.get("windows") == reference_window_count
        )
        checks["scenario_temporal_integrity_release_ready"] &= all(
            summary.get(key) is True
            for key in (
                "all_test_horizons_non_overlapping",
                "all_partitions_strictly_time_ordered",
                "all_training_histories_expanding",
            )
        )

        horizons: list[dict[str, object]] = []
        for window in windows:
            if not isinstance(window, dict):
                checks["raw_partition_count_keys_complete"] = False
                checks["raw_partition_counts_conserve_source_rows"] = False
                checks["window_placebo_gates_present"] = False
                checks["window_placebo_gates_release_ready"] = False
                checks["window_placebo_draws_exact_release_count"] = False
                continue
            raw_counts = window.get("raw_partition_rows")
            bounds = window.get("time_bounds")
            horizons.append(
                {
                    "window": window.get("window"),
                    "test_start": window.get("test_start"),
                    "test_end": window.get("test_end"),
                    "test_raw_rows": (
                        raw_counts.get("test") if isinstance(raw_counts, dict) else None
                    ),
                    "test_time_bounds": (
                        bounds.get("test") if isinstance(bounds, dict) else None
                    ),
                }
            )
            raw_keys_exact = (
                isinstance(raw_counts, dict)
                and set(raw_counts) == set(LABEL_DELAY_RAW_PARTITIONS)
                and all(
                    isinstance(raw_counts[name], int) and raw_counts[name] >= 0
                    for name in LABEL_DELAY_RAW_PARTITIONS
                )
            )
            checks["raw_partition_count_keys_complete"] &= raw_keys_exact
            if raw_keys_exact:
                observed_raw_count_sets += 1
                checks["raw_partition_counts_conserve_source_rows"] &= (
                    sum(int(raw_counts[name]) for name in LABEL_DELAY_RAW_PARTITIONS)
                    == expected_source_rows
                )
            else:
                checks["raw_partition_counts_conserve_source_rows"] = False

            placebo = window.get("test_label_alignment_placebo")
            gate = placebo.get("gate") if isinstance(placebo, dict) else None
            gate_present = isinstance(gate, dict)
            checks["window_placebo_gates_present"] &= gate_present
            if gate_present:
                observed_window_gates += 1
                checks["window_placebo_gates_release_ready"] &= (
                    gate.get("enforced") is True and gate.get("passed") is True
                )
                checks["window_placebo_draws_exact_release_count"] &= (
                    placebo.get("draws") == TEMPORAL_PLACEBO_DRAWS_PER_WINDOW
                    and gate.get("draws") == TEMPORAL_PLACEBO_DRAWS_PER_WINDOW
                )
            else:
                checks["window_placebo_gates_release_ready"] = False
                checks["window_placebo_draws_exact_release_count"] = False

        checks["test_horizon_metadata_match_zero_day"] &= (
            horizons == reference_horizons
        )
        pooled_gate = summary.get("test_label_alignment_placebo_gate")
        pooled_present = isinstance(pooled_gate, dict)
        checks["pooled_placebo_gates_present"] &= pooled_present
        if pooled_present:
            observed_pooled_gates += 1
            checks["pooled_placebo_gates_release_ready"] &= (
                pooled_gate.get("enforced") is True and pooled_gate.get("passed") is True
            )
            checks["pooled_placebo_draws_complete"] &= pooled_gate.get("draws") == (
                TEMPORAL_PLACEBO_DRAWS_PER_WINDOW * reference_window_count
            )
        else:
            checks["pooled_placebo_gates_release_ready"] = False
            checks["pooled_placebo_draws_complete"] = False

    expected_window_gates = len(DEFAULT_LABEL_DELAY_DAYS) * reference_window_count
    expected_pooled_gates = len(DEFAULT_LABEL_DELAY_DAYS)
    expected_raw_count_sets = expected_window_gates
    checks["window_placebo_gate_count_complete"] = (
        observed_window_gates == expected_window_gates
    )
    checks["pooled_placebo_gate_count_complete"] = (
        observed_pooled_gates == expected_pooled_gates
    )
    checks["raw_partition_count_set_count_complete"] = (
        observed_raw_count_sets == expected_raw_count_sets
    )

    required_checks = dict(checks)
    passed = all(required_checks.values())
    if enforced and not passed:
        failed = ", ".join(name for name, check in required_checks.items() if not check)
        raise RuntimeError(
            "Label-delay release contract failed; publication must stop: " + failed
        )
    return {
        "passed": passed if enforced else None,
        "enforced": enforced,
        "expected_delay_days": expected_delays,
        "expected_raw_partition_keys": list(LABEL_DELAY_RAW_PARTITIONS),
        "expected_source_rows_per_window": expected_source_rows,
        "expected_placebo_draws_per_window": TEMPORAL_PLACEBO_DRAWS_PER_WINDOW,
        "expected_window_placebo_gates": expected_window_gates,
        "observed_window_placebo_gates": observed_window_gates,
        "expected_pooled_placebo_gates": expected_pooled_gates,
        "observed_pooled_placebo_gates": observed_pooled_gates,
        "expected_raw_partition_count_sets": expected_raw_count_sets,
        "observed_raw_partition_count_sets": observed_raw_count_sets,
        "checks": required_checks,
        "interpretation": (
            "Fail-closed structural and placebo-gate checks for authored label-delay "
            "sensitivity; not evidence that a delay is an observed operational SLA."
        ),
    }


def run_synthetic_benchmark(
    *,
    config: SyntheticConfig | None = None,
    seeds: tuple[int, ...] = (1103, 2909, 4703),
    protocols: tuple[Protocol, ...] = PROTOCOLS,
    permutation_draws: int = PERMUTATION_DRAWS_PER_SPLIT,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    enforce_negative_control: bool = True,
) -> dict[str, object]:
    """Run all public reliability gates and return an aggregate-only artifact."""

    config = config or SyntheticConfig()
    if not seeds:
        raise ValueError("At least one evaluation seed is required")
    if permutation_draws <= 0:
        raise ValueError("At least one permutation draw per strict split is required")
    if bootstrap_draws < 0:
        raise ValueError("bootstrap_draws must be non-negative")
    if enforce_negative_control and permutation_draws != PERMUTATION_DRAWS_PER_SPLIT:
        raise ValueError(
            "Release-gated benchmark requires exactly five permutation draws per strict split"
        )
    if enforce_negative_control and bootstrap_draws < DEFAULT_BOOTSTRAP_DRAWS:
        raise ValueError(
            "Release-gated benchmark requires at least 2,000 cluster-bootstrap draws"
        )
    required_protocols = {"row_random", "fingerprint_group", "forward_time"}
    missing_protocols = required_protocols - set(protocols)
    if missing_protocols:
        raise ValueError(
            "Benchmark summary requires protocols: " + ", ".join(sorted(missing_protocols))
        )

    frame = generate_synthetic_reviews(config)
    split_keys = make_split_keys(frame)
    protocol_reports = {}
    raw_runs: dict[str, list[dict[str, object]]] = {}
    for protocol in protocols:
        protocol_seeds = seeds[:1] if protocol == "forward_time" else seeds
        protocol_runs = [
            _run_protocol_once(
                frame,
                split_keys,
                protocol,
                seed,
                permutation_draws=(
                    permutation_draws if protocol == "fingerprint_group" else 0
                ),
                bootstrap_draws=(
                    bootstrap_draws
                    if protocol == "fingerprint_group" and seed == seeds[0]
                    else 0
                ),
                enforce_bootstrap_gate=(
                    enforce_negative_control
                    if protocol == "fingerprint_group" and seed == seeds[0]
                    else False
                ),
            )
            for seed in protocol_seeds
        ]
        raw_runs[protocol] = protocol_runs
        protocol_reports[protocol] = _aggregate_protocol_runs(protocol_runs)

    if enforce_negative_control:
        negative_control_gate = _enforce_negative_control_runs(
            raw_runs["fingerprint_group"]
        )
    else:
        negative_control_gate = {
            "passed": None,
            "enforced": False,
            "gate_interpretation": (
                "diagnostic summaries only; release heuristic sanity bounds were not enforced"
            ),
            "multiplicity_controlled": False,
            "designs": {
                "train_label_permutation": _diagnostic_control_summary(
                    raw_runs["fingerprint_group"],
                    scalar_key="train_label_permutation_controls",
                    capacity_key="train_label_permutation_capacity_controls",
                ),
                "test_label_alignment_placebo": _diagnostic_control_summary(
                    raw_runs["fingerprint_group"],
                    scalar_key="test_label_alignment_placebos",
                    capacity_key="test_label_alignment_capacity_placebos",
                ),
            },
        }

    audit_manifest = make_split_manifest(
        split_keys, SplitSpec(protocol="row_random", seed=seeds[0])
    )
    audit_attached = attach_manifest(frame, audit_manifest)
    audit_labeled = pd.concat(
        [_partition_labeled(audit_attached, name) for name in ("train", "validation", "test")],
        ignore_index=True,
    )

    sensitivity_ladder = _protocol_sensitivity_ladder(protocol_reports)
    strict_gap = _paired_strict_gap(raw_runs)
    strict_runs = raw_runs["fingerprint_group"]
    queue_capacity_sensitivity = {
        "reference_protocol": "fingerprint_group",
        "matched_evaluation_seeds": list(seeds),
        "pre_specified_budget_shares": list(DEFAULT_CAPACITY_SHARES),
        "primary_budget_share": PRIMARY_CAPACITY_SHARE,
        "matches_release_capacity_contract": True,
        "capacity_selected_post_hoc": False,
        "model_across_matched_seeds": _aggregate_capacity_curves(
            [run["model_capacity_curve"] for run in strict_runs]
        ),
        "seeded_random_score_across_matched_seeds": _aggregate_capacity_curves(
            [run["seeded_random_capacity_curve"] for run in strict_runs]
        ),
        "conditional_interval_path": "conditional_uncertainty.result.capacity_curve.points",
        "negative_control_gate_path": "runtime_controls.negative_control_gate.designs",
        "interpretation": (
            "Pre-specified workload scenarios on an authored synthetic fixture; not observed "
            "staffing capacity, a post-hoc optimum, or an estimate of business value."
        ),
    }
    conditional_uncertainty = raw_runs["fingerprint_group"][0][
        "conditional_cluster_bootstrap"
    ]
    rolling_origin = run_rolling_origin_backtest(
        frame,
        split_keys,
        placebo_draws_per_window=(20 if enforce_negative_control else 5),
        enforce_placebo_gate=enforce_negative_control,
    )
    label_delay_release_gate = _label_delay_release_gate(
        rolling_origin,
        expected_source_rows=len(frame),
        enforced=enforce_negative_control,
    )

    return {
        "contract_version": "5.0",
        "artifact_scope": "aggregate_only_synthetic_reliability_fixture",
        "synthetic_data": True,
        "source_rows_included": False,
        "source_trained_artifacts_included": False,
        "metric_definitions": {
            "average_precision_attention": (
                "Non-interpolated average precision from scikit-learn; prevalence is the "
                "no-skill reference for the attention class."
            ),
            "brier": "Mean squared probability error; lower is better.",
            "roc_auc_supplementary": (
                "Ranking metric reported only as a supplementary discrimination view."
            ),
            "budget_ranking_tie_policy": (
                "A cutoff score tie uses fractional expected allocation, independent of row "
                "order. Stress and conditional-uncertainty summaries omit recall and lift "
                "when all scores tie because no model ranking exists."
            ),
            "conditional_cluster_bootstrap": (
                "A 95% percentile interval for one frozen fingerprint-group scoring rule, "
                "using near-text-fingerprint pairs resampling; separate from multi-seed and "
                "rolling-origin sensitivity ranges."
            ),
            "queue_capacity_sensitivity": (
                "Precision, recall, and lift at pre-specified 5%, 10%, and 20% review-queue "
                "capacities. Matched-seed ranges are descriptive; frozen-rule cluster "
                "bootstrap intervals are marginal, not simultaneous."
            ),
            "label_delay_sensitivity": (
                "Rolling-origin comparisons at pre-specified 0-, 14-, and 30-day proxy-label "
                "availability delays with identical test horizons. Delayed rows are embargoed "
                "from fitting and threshold selection; 14 and 30 days are authored scenarios, "
                "not observed service-level agreements."
            ),
            "negative_control_sanity_gates": (
                "Pre-specified fail-closed heuristic bounds for five null draws per strict "
                "split and their aggregate means; not p-values, family-wise-error control, "
                "or a calibrated joint envelope."
            ),
            "sensitivity_ranges": (
                "Mean, sample standard deviation, minimum, and maximum across deterministic "
                "runs; not confidence intervals."
            ),
        },
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "decision_contract": {
            "analysis_unit": "one unique review submission",
            "prediction_time": "when an unscored review summary and body are submitted",
            "available_model_features": list(FEATURE_COLUMNS),
            "target": "rating-derived needs_attention proxy: 1-2 stars=1, 4-5 stars=0",
            "neutral_policy": "3-star rows are excluded inside each assigned partition",
            "intended_decision": (
                "hypothetically rank manual-review queues at pre-specified 5%, 10%, "
                "and 20% workload shares"
            ),
            "pre_specified_queue_capacity_shares": list(DEFAULT_CAPACITY_SHARES),
            "primary_queue_capacity_share": PRIMARY_CAPACITY_SHARE,
            "capacity_selected_post_hoc": False,
            "staffing_cost_or_value_modeled": False,
            "source_rating_is_target_oracle": True,
            "label_delay_days": list(DEFAULT_LABEL_DELAY_DAYS),
            "availability_time_observed": False,
            "delay_selected_post_hoc": False,
            "operational_target_validated": False,
            "operational_need_validated": False,
            "label_delay_scenario_scope": (
                "14- and 30-day delays are authored sensitivity scenarios, not business SLAs"
            ),
            "scope": "offline synthetic reliability audit, not a deployment claim",
            "causal_claim": False,
        },
        "synthetic_fixture": asdict(config),
        "evaluation_seeds": list(seeds),
        "sensitivity_design": {
            "ladder_interpretation": (
                "protocols isolate different risks and are not a total ordering of difficulty"
            ),
            "hash_protocols": "all listed seeds vary split assignment and model seed",
            "forward_time": (
                "one fixed chronological cutoff with whole-timestamp boundaries using the "
                "first seed; complemented by a separate four-window rolling-origin backtest"
            ),
            "summary": (
                "mean/std/min/max are descriptive sensitivity ranges, not confidence intervals"
            ),
        },
        "data_audit": dataset_audit(frame, audit_labeled),
        "protocols": protocol_reports,
        "protocol_sensitivity_ladder": sensitivity_ladder,
        "paired_strict_gap": strict_gap,
        "queue_capacity_sensitivity": queue_capacity_sensitivity,
        "conditional_uncertainty": {
            "reference_protocol": "fingerprint_group",
            "reference_evaluation_seed": seeds[0],
            "result": conditional_uncertainty,
        },
        "rolling_origin_backtest": rolling_origin,
        "runtime_controls": {
            "split_input_columns": list(split_keys.columns),
            "target_bearing_split_columns": sorted(
                {"rating", TARGET_COLUMN, "target", "label", "y"} & set(split_keys.columns)
            ),
            "neutral_exclusion_stage": "inside_train_validation_test_after_manifest",
            "protocol_isolation_fail_closed": True,
            "whole_timestamp_forward_boundaries": True,
            "conflict_retention_fail_closed": True,
            "conflict_reference_target_use": (
                "audit-only after label-free split construction; never used for assignment"
            ),
            "negative_control_mode": (
                "multi_draw_train_and_test_label_placebos_fail_closed"
                if enforce_negative_control
                else "diagnostic_only_for_demo_or_test"
            ),
            "permutation_draws_per_fingerprint_split": permutation_draws,
            "negative_control_gate": negative_control_gate,
            "conditional_uncertainty_release_gate": (
                conditional_uncertainty["release_gate"]
                if conditional_uncertainty is not None
                else {"passed": None, "enforced": False}
            ),
            "capacity_null_gate_scope": (
                "heuristic train-label permutation and test-label alignment placebo lift "
                "sanity bounds at each pre-specified 5%, 10%, and 20% capacity"
            ),
            "rolling_origin_temporal_separation": True,
            "rolling_origin_cross_boundary_entity_isolation": False,
            "label_delay_days": list(DEFAULT_LABEL_DELAY_DAYS),
            "availability_time_observed": False,
            "delay_selected_post_hoc": False,
            "operational_target_validated": False,
            "operational_need_validated": False,
            "label_delay_release_gate": label_delay_release_gate,
            "legacy_metric_alias": (
                "classification_metrics keeps pr_auc_attention for private adapter "
                "compatibility; contract 5.0 reports emit only average_precision_attention"
            ),
            "public_safety_validation": "performed_by_CLI_before_write",
        },
        "limitations": [
            (
                "All scores come from authored synthetic text and demonstrate the evaluation "
                "system only."
            ),
            (
                "The source rating is the target oracle for this rating-derived proxy; when "
                "ratings are available at decision time, a text score may be operationally "
                "redundant. The proxy is not human-annotated sentiment or causal impact."
            ),
            (
                "No real-data generalization, fairness, moderation, causal, operational, or "
                "business value claim is supported."
            ),
            (
                "The 5%, 10%, and 20% queue shares are pre-specified sensitivity scenarios, "
                "not validated staffing levels; no review cost, action value, or net benefit "
                "is modeled."
            ),
            (
                "Negative-control bounds are heuristic fail-closed sanity checks, not "
                "multiplicity-adjusted inferential tests."
            ),
            (
                "Hash-group protocols approximate deployment risks but cannot reproduce every "
                "source shift."
            ),
            (
                "A corrected split-first source audit exists privately, but source rights, "
                "target validity, and public-safe aggregation still block any real-data "
                "metric release."
            ),
            (
                "Multi-seed min/max values describe protocol and optimizer sensitivity; they are "
                "not sampling confidence intervals."
            ),
            (
                "The conditional cluster-bootstrap interval holds one synthetic fitted model "
                "and test manifest fixed; it does not cover retraining, crossed dependence, "
                "source selection, temporal drift, or real-data generalization."
            ),
            (
                "Rolling-origin windows preserve chronology but allow recurring text, users, "
                "and products across time; those overlaps are audited rather than removed."
            ),
            (
                "Label-availability timestamps were not observed. The 14- and 30-day delays "
                "are authored stress scenarios, not validated operational SLAs, and their "
                "paired rolling-origin ranges are not confidence intervals."
            ),
        ],
    }
