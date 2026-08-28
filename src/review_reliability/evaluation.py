"""End-to-end split-first reliability evaluation and aggregate report builder."""

from __future__ import annotations

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
    PROTOCOLS,
    Protocol,
    SplitSpec,
    attach_manifest,
    make_split_manifest,
    manifest_metadata,
    partition_overlap_audit,
)
from review_reliability.stress import prior_shift_slice, text_stress_cases

REPORT_METRICS = (
    "rows",
    "attention_prevalence",
    "pr_auc_attention",
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
    "precision_at_budget",
    "recall_at_budget",
    "lift_at_budget",
)
CONSTANT_BASELINE_METRICS = tuple(
    metric
    for metric in REPORT_METRICS
    if metric not in {"budget_count", "precision_at_budget", "recall_at_budget", "lift_at_budget"}
)
STRESS_METRICS = (
    "rows",
    "attention_prevalence",
    "pr_auc_attention",
    "brier",
    "recall_at_budget",
    "lift_at_budget",
)


def _partition_labeled(attached: pd.DataFrame, partition: str) -> pd.DataFrame:
    raw_partition = attached.loc[attached["partition"] == partition, list(RAW_COLUMNS)].copy()
    labeled = derive_rating_proxy(raw_partition)
    if labeled[TARGET_COLUMN].nunique() != 2:
        raise ValueError(f"{partition} partition must contain both proxy classes")
    return labeled


def _score_metrics(
    model: object,
    labeled: pd.DataFrame,
    threshold: float,
) -> dict[str, float | int]:
    scores = predict_attention_probability(model, labeled.loc[:, list(FEATURE_COLUMNS)])
    return classification_metrics(
        labeled[TARGET_COLUMN], scores, threshold=threshold, budget_share=0.10
    )


def _metric_subset(metrics: dict[str, float | int], keys: Iterable[str]) -> dict[str, float | int]:
    return {key: metrics[key] for key in keys}


def _stress_evaluation(
    model: object,
    labeled_test: pd.DataFrame,
    threshold: float,
    baseline: dict[str, float | int],
    seed: int,
) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    y_test = labeled_test[TARGET_COLUMN]
    for name, features in text_stress_cases(labeled_test, seed).items():
        scores = predict_attention_probability(model, features)
        metrics = classification_metrics(y_test, scores, threshold=threshold, budget_share=0.10)
        summary = _metric_subset(metrics, STRESS_METRICS)
        for metric in ("pr_auc_attention", "brier", "recall_at_budget", "lift_at_budget"):
            summary[f"delta_{metric}"] = float(metrics[metric]) - float(baseline[metric])
        output[name] = summary

    shifted = prior_shift_slice(labeled_test, seed)
    shifted_metrics = _score_metrics(model, shifted, threshold)
    shifted_summary = _metric_subset(shifted_metrics, STRESS_METRICS)
    for metric in ("pr_auc_attention", "brier", "recall_at_budget", "lift_at_budget"):
        shifted_summary[f"delta_{metric}"] = float(shifted_metrics[metric]) - float(
            baseline[metric]
        )
    output["higher_attention_prior"] = shifted_summary
    return output


def _run_protocol_once(
    frame: pd.DataFrame,
    split_keys: pd.DataFrame,
    protocol: Protocol,
    seed: int,
    *,
    run_permutation: bool,
) -> dict[str, object]:
    spec = SplitSpec(protocol=protocol, seed=seed)
    manifest = make_split_manifest(split_keys, spec)
    attached = attach_manifest(frame, manifest)
    train = _partition_labeled(attached, "train")
    validation = _partition_labeled(attached, "validation")
    test = _partition_labeled(attached, "test")

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

    partitions = {}
    for name, partition in (("train", train), ("validation", validation), ("test", test)):
        partitions[name] = {
            "labeled_rows": int(len(partition)),
            "attention_prevalence": float(partition[TARGET_COLUMN].mean()),
        }

    negative_control: dict[str, float | int] | None = None
    if run_permutation:
        permutation_rng = np.random.default_rng(seed + 30_011)
        shuffled_target = permutation_rng.permutation(train[TARGET_COLUMN].to_numpy())
        shuffled_model = fit_text_pipeline(
            train.loc[:, list(FEATURE_COLUMNS)], pd.Series(shuffled_target), seed=seed + 1
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
        negative_control = classification_metrics(
            test[TARGET_COLUMN],
            shuffled_test,
            threshold=shuffled_threshold,
            budget_share=0.10,
        )

    return {
        "manifest": manifest_metadata(split_keys, manifest, spec),
        "partitions": partitions,
        "overlap_audit": partition_overlap_audit(split_keys, manifest),
        "model": _metric_subset(model_metrics, REPORT_METRICS),
        "baselines": {
            "train_prevalence_probability_and_majority_decision": _metric_subset(
                prior_metrics, CONSTANT_BASELINE_METRICS
            ),
            "seeded_random_score": _metric_subset(random_metrics, REPORT_METRICS),
        },
        "negative_control": (
            _metric_subset(negative_control, REPORT_METRICS) if negative_control else None
        ),
        "stress": _stress_evaluation(model, test, threshold, model_metrics, seed),
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
        stress_output[stress_name] = _aggregate_metrics(stress_runs)

    negative_runs = [run["negative_control"] for run in runs]
    negative_control = None
    if all(run is not None for run in negative_runs):
        negative_control = _aggregate_metrics(negative_runs)

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
        "negative_control_train_label_permutation": negative_control,
        "stress": stress_output,
    }
    if "time_bounds" in manifest:
        result["time_bounds"] = manifest["time_bounds"]
    return result


def _enforce_negative_control_runs(runs: list[dict[str, object]]) -> None:
    for index, run in enumerate(runs):
        control = run["negative_control"]
        if not isinstance(control, dict):
            raise RuntimeError("The strict protocol is missing its label-permutation control")
        prevalence = float(control["attention_prevalence"])
        pr_auc = float(control["pr_auc_attention"])
        roc_auc = float(control["roc_auc_supplementary"])
        lift = pr_auc / prevalence if prevalence else float("inf")
        if not 0.30 <= roc_auc <= 0.70 or lift > 1.80:
            raise RuntimeError(
                f"Label-permutation run {index} did not return sufficiently close to chance; "
                "publication must stop for investigation"
            )


def run_synthetic_benchmark(
    *,
    config: SyntheticConfig | None = None,
    seeds: tuple[int, ...] = (1103, 2909, 4703),
    protocols: tuple[Protocol, ...] = PROTOCOLS,
    enforce_negative_control: bool = True,
) -> dict[str, object]:
    """Run all public reliability gates and return an aggregate-only artifact."""

    config = config or SyntheticConfig()
    if not seeds:
        raise ValueError("At least one evaluation seed is required")
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
                run_permutation=protocol == "fingerprint_group",
            )
            for seed in protocol_seeds
        ]
        raw_runs[protocol] = protocol_runs
        protocol_reports[protocol] = _aggregate_protocol_runs(protocol_runs)

    if enforce_negative_control:
        _enforce_negative_control_runs(raw_runs["fingerprint_group"])

    audit_manifest = make_split_manifest(
        split_keys, SplitSpec(protocol="row_random", seed=seeds[0])
    )
    audit_attached = attach_manifest(frame, audit_manifest)
    audit_labeled = pd.concat(
        [_partition_labeled(audit_attached, name) for name in ("train", "validation", "test")],
        ignore_index=True,
    )

    naive_pr = protocol_reports["row_random"]["model"]["pr_auc_attention"]["mean"]
    strict_pr = protocol_reports["fingerprint_group"]["model"]["pr_auc_attention"]["mean"]
    time_pr = protocol_reports["forward_time"]["model"]["pr_auc_attention"]["mean"]
    naive_prevalence = protocol_reports["row_random"]["model"]["attention_prevalence"]["mean"]
    strict_prevalence = protocol_reports["fingerprint_group"]["model"][
        "attention_prevalence"
    ]["mean"]
    time_prevalence = protocol_reports["forward_time"]["model"]["attention_prevalence"][
        "mean"
    ]

    return {
        "contract_version": "1.0",
        "artifact_scope": "aggregate_only_synthetic_reliability_fixture",
        "synthetic_data": True,
        "source_rows_included": False,
        "source_trained_artifacts_included": False,
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
            "intended_decision": "rank a fixed-capacity manual review queue",
            "causal_claim": False,
        },
        "synthetic_fixture": asdict(config),
        "evaluation_seeds": list(seeds),
        "sensitivity_design": {
            "hash_protocols": "all listed seeds vary split assignment and model seed",
            "forward_time": (
                "one fixed chronological cutoff using the first seed; no repeated-cutoff "
                "uncertainty claim"
            ),
            "summary": (
                "mean/std/min/max are descriptive sensitivity ranges, not confidence intervals"
            ),
        },
        "data_audit": dataset_audit(frame, audit_labeled),
        "protocols": protocol_reports,
        "naive_vs_strict_summary": {
            "metric": "attention-class PR-AUC",
            "row_random_mean": float(naive_pr),
            "row_random_prevalence": float(naive_prevalence),
            "fingerprint_group_mean": float(strict_pr),
            "fingerprint_group_prevalence": float(strict_prevalence),
            "forward_time_mean": float(time_pr),
            "forward_time_prevalence": float(time_prevalence),
            "row_random_minus_fingerprint_group": float(naive_pr - strict_pr),
            "interpretation": (
                "A synthetic diagnostic gap created by planted duplicates and drift; "
                "not an empirical finding about Amazon or production performance."
            ),
        },
        "runtime_controls": {
            "split_input_columns": list(split_keys.columns),
            "target_bearing_split_columns": sorted(
                {"rating", TARGET_COLUMN, "target", "label", "y"} & set(split_keys.columns)
            ),
            "neutral_exclusion_stage": "inside_train_validation_test_after_manifest",
            "negative_control_mode": (
                "each_fingerprint_group_run_fail_closed"
                if enforce_negative_control
                else "disabled_for_demo_or_test"
            ),
            "public_safety_validation": "performed_by_CLI_before_write",
        },
        "limitations": [
            (
                "All scores come from authored synthetic text and demonstrate the evaluation "
                "system only."
            ),
            "The target is a rating-derived proxy, not human-annotated sentiment or causal impact.",
            (
                "No real-data generalization, fairness, moderation, or business value claim "
                "is supported."
            ),
            (
                "Hash-group protocols approximate deployment risks but cannot reproduce every "
                "source shift."
            ),
            (
                "A corrected split-first private source audit is required before publishing "
                "real-data metrics."
            ),
            (
                "Multi-seed min/max values describe protocol and optimizer sensitivity; they are "
                "not sampling confidence intervals."
            ),
        ],
    }
