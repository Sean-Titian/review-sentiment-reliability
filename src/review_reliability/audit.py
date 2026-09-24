"""Aggregate-only duplicate and feature-profile audits."""

from __future__ import annotations

import math

import pandas as pd

from review_reliability.data import (
    TARGET_COLUMN,
    near_text_fingerprint,
    normalize_text,
    text_fingerprint,
)
from review_reliability.metrics import aggregate_scalar_runs
from review_reliability.splits import SplitSpec, make_split_manifest

FEATURE_RECURRENCE_FIELDS = ("summary", "body", "combined_input")
FEATURE_RECURRENCE_METHODS = ("normalized_exact", "token_set_equality")
_RECURRENCE_METRICS = (
    "non_empty_raw_rows",
    "empty_raw_rows",
    "unique_non_empty_groups",
    "cross_partition_groups",
    "cross_partition_affected_raw_rows",
    "cross_partition_affected_raw_row_share",
    "cross_partition_affected_non_empty_raw_row_share",
    "test_rows",
    "non_empty_test_rows",
    "empty_test_rows",
    "cross_partition_affected_test_rows",
    "cross_partition_affected_test_row_share",
    "cross_partition_affected_non_empty_test_row_share",
)
_RECURRENCE_COUNT_METRICS = (
    "non_empty_raw_rows",
    "empty_raw_rows",
    "unique_non_empty_groups",
    "cross_partition_groups",
    "cross_partition_affected_raw_rows",
    "test_rows",
    "non_empty_test_rows",
    "empty_test_rows",
    "cross_partition_affected_test_rows",
)


def dataset_audit(raw_frame: pd.DataFrame, labeled_frame: pd.DataFrame) -> dict[str, object]:
    """Describe synthetic fixture realism without emitting rows, text, or identifiers."""

    raw_canonical_input = (
        raw_frame["summary"].fillna("").astype(str)
        + " "
        + raw_frame["text"].fillna("").astype(str)
    )
    labeled_canonical_input = (
        labeled_frame["summary"].fillna("").astype(str)
        + " "
        + labeled_frame["text"].fillna("").astype(str)
    )
    exact = raw_canonical_input.map(text_fingerprint)
    near = raw_canonical_input.map(near_text_fingerprint)
    labeled_exact = labeled_canonical_input.map(text_fingerprint)
    conflict_sizes = (
        pd.DataFrame({"fingerprint": labeled_exact, "target": labeled_frame[TARGET_COLUMN]})
        .groupby("fingerprint", dropna=False)["target"]
        .nunique()
    )
    conflict_groups = conflict_sizes.index[conflict_sizes > 1]
    conflict_row_share = float(labeled_exact.isin(conflict_groups).mean())
    rating_counts = raw_frame["rating"].value_counts().sort_index()
    rating_distribution = {
        str(int(rating)): {
            "rows": int(count),
            "share": float(count / len(raw_frame)),
        }
        for rating, count in rating_counts.items()
    }
    return {
        "raw_rows": int(len(raw_frame)),
        "labeled_rows_after_partition_local_neutral_exclusion": int(len(labeled_frame)),
        "attention_prevalence": float(labeled_frame[TARGET_COLUMN].mean()),
        "missing_summary_share": float(raw_frame["summary"].isna().mean()),
        "missing_body_share": float(raw_frame["text"].isna().mean()),
        "raw_exact_duplicate_row_share": float(raw_frame["text"].duplicated(keep=False).mean()),
        "normalized_exact_duplicate_row_share": float(exact.duplicated(keep=False).mean()),
        "near_duplicate_row_share": float(near.duplicated(keep=False).mean()),
        "conflicting_proxy_fingerprint_groups": int(len(conflict_groups)),
        "conflicting_proxy_row_share": conflict_row_share,
        "rating_distribution": rating_distribution,
        "join_amplification": "not_applicable_single_table_fixture",
    }


def _feature_values(raw_frame: pd.DataFrame, field: str) -> pd.Series:
    if field == "summary":
        return raw_frame["summary"]
    if field == "body":
        return raw_frame["text"]
    if field == "combined_input":
        return (
            raw_frame["summary"].fillna("").astype(str)
            + " "
            + raw_frame["text"].fillna("").astype(str)
        )
    raise ValueError(f"Unknown recurrence field: {field}")


def _feature_recurrence_metrics(
    values: pd.Series,
    partitions: pd.Series,
    *,
    method: str,
) -> dict[str, int | float]:
    normalized = values.map(normalize_text)
    non_empty = normalized.ne("")
    if method == "normalized_exact":
        fingerprints = normalized.map(text_fingerprint)
    elif method == "token_set_equality":
        fingerprints = normalized.map(near_text_fingerprint)
    else:
        raise ValueError(f"Unknown recurrence method: {method}")

    eligible = pd.DataFrame(
        {
            "fingerprint": fingerprints.loc[non_empty],
            "partition": partitions.loc[non_empty],
        }
    )
    partition_counts = eligible.groupby("fingerprint")["partition"].nunique()
    shared_groups = partition_counts.index[partition_counts > 1]
    affected = eligible["fingerprint"].isin(shared_groups)
    test = eligible["partition"].eq("test")
    affected_test_rows = int((affected & test).sum())
    total_rows = int(len(values))
    non_empty_rows = int(non_empty.sum())
    test_rows = int(partitions.eq("test").sum())
    non_empty_test_rows = int(test.sum())
    affected_rows = int(affected.sum())

    return {
        "non_empty_raw_rows": non_empty_rows,
        "empty_raw_rows": total_rows - non_empty_rows,
        "unique_non_empty_groups": int(len(partition_counts)),
        "cross_partition_groups": int(len(shared_groups)),
        "cross_partition_affected_raw_rows": affected_rows,
        "cross_partition_affected_raw_row_share": float(affected_rows / total_rows),
        "cross_partition_affected_non_empty_raw_row_share": float(
            affected_rows / non_empty_rows if non_empty_rows else 0.0
        ),
        "test_rows": test_rows,
        "non_empty_test_rows": non_empty_test_rows,
        "empty_test_rows": test_rows - non_empty_test_rows,
        "cross_partition_affected_test_rows": affected_test_rows,
        "cross_partition_affected_test_row_share": float(
            affected_test_rows / test_rows if test_rows else 0.0
        ),
        "cross_partition_affected_non_empty_test_row_share": float(
            affected_test_rows / non_empty_test_rows if non_empty_test_rows else 0.0
        ),
    }


def _feature_method_partitions_identical(values: pd.Series) -> bool:
    """Return whether exact and token-set equality induce the same row groups."""

    normalized = values.map(normalize_text)
    normalized = normalized.loc[normalized.ne("")]
    comparison = pd.DataFrame(
        {
            "exact": normalized.map(text_fingerprint),
            "token_set": normalized.map(near_text_fingerprint),
        }
    )
    if comparison.empty:
        return True
    exact_maps_once = comparison.groupby("exact")["token_set"].nunique().eq(1).all()
    token_set_maps_once = comparison.groupby("token_set")["exact"].nunique().eq(1).all()
    return bool(exact_maps_once and token_set_maps_once)


def _feature_recurrence_release_gate(
    runs: list[dict[str, object]],
    *,
    expected_rows: int,
    expected_runs: int,
    enforced: bool,
) -> dict[str, object]:
    required_fields = set(FEATURE_RECURRENCE_FIELDS)
    required_methods = set(FEATURE_RECURRENCE_METHODS)
    run_count_matches = len(runs) == expected_runs
    structures_complete = bool(runs) and run_count_matches
    counts_conserve = bool(runs) and run_count_matches
    subset_counts_consistent = bool(runs) and run_count_matches
    values_valid = bool(runs) and run_count_matches
    combined_isolated = bool(runs) and run_count_matches
    evaluation_seeds = [run.get("evaluation_seed") for run in runs]
    seeds_unique_and_complete = (
        run_count_matches
        and all(isinstance(seed, int) for seed in evaluation_seeds)
        and len(set(evaluation_seeds)) == expected_runs
    )

    for run in runs:
        features = run.get("features")
        structures_complete &= isinstance(features, dict) and set(features) == required_fields
        if not isinstance(features, dict):
            counts_conserve = False
            combined_isolated = False
            continue
        for field in FEATURE_RECURRENCE_FIELDS:
            methods = features.get(field)
            structures_complete &= isinstance(methods, dict) and set(methods) == required_methods
            if not isinstance(methods, dict):
                counts_conserve = False
                combined_isolated = False
                continue
            for method in FEATURE_RECURRENCE_METHODS:
                metrics = methods.get(method)
                structures_complete &= isinstance(metrics, dict) and all(
                    name in metrics for name in _RECURRENCE_METRICS
                )
                if not isinstance(metrics, dict):
                    counts_conserve = False
                    values_valid = False
                    combined_isolated = False
                    continue
                numeric = {
                    name: float(metrics.get(name, float("nan")))
                    for name in _RECURRENCE_METRICS
                }
                values_valid &= all(
                    math.isfinite(value) and value >= 0.0 for value in numeric.values()
                )
                values_valid &= all(
                    numeric[name].is_integer() for name in _RECURRENCE_COUNT_METRICS
                )
                counts_conserve &= (
                    int(metrics.get("non_empty_raw_rows", -1))
                    + int(metrics.get("empty_raw_rows", -1))
                    == expected_rows
                )
                counts_conserve &= (
                    int(metrics.get("non_empty_test_rows", -1))
                    + int(metrics.get("empty_test_rows", -1))
                    == int(metrics.get("test_rows", -2))
                )
                non_empty_rows = int(metrics.get("non_empty_raw_rows", -1))
                affected_rows = int(metrics.get("cross_partition_affected_raw_rows", -2))
                unique_groups = int(metrics.get("unique_non_empty_groups", -1))
                shared_groups = int(metrics.get("cross_partition_groups", -2))
                test_rows = int(metrics.get("test_rows", -1))
                non_empty_test_rows = int(metrics.get("non_empty_test_rows", -1))
                affected_test_rows = int(
                    metrics.get("cross_partition_affected_test_rows", -2)
                )
                values_valid &= 0 <= shared_groups <= unique_groups <= non_empty_rows
                values_valid &= 0 <= affected_rows <= non_empty_rows <= expected_rows
                values_valid &= 0 <= affected_test_rows <= non_empty_test_rows <= test_rows
                subset_counts_consistent &= (
                    0 <= test_rows <= expected_rows
                    and non_empty_test_rows <= non_empty_rows
                    and int(metrics.get("empty_test_rows", -1))
                    <= int(metrics.get("empty_raw_rows", -1))
                    and affected_test_rows <= affected_rows
                )
                expected_raw_share = affected_rows / expected_rows
                expected_non_empty_share = (
                    affected_rows / non_empty_rows if non_empty_rows else 0.0
                )
                expected_test_share = affected_test_rows / test_rows if test_rows else 0.0
                expected_non_empty_test_share = (
                    affected_test_rows / non_empty_test_rows if non_empty_test_rows else 0.0
                )
                values_valid &= math.isclose(
                    float(metrics.get("cross_partition_affected_raw_row_share", -1)),
                    expected_raw_share,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                values_valid &= math.isclose(
                    float(
                        metrics.get(
                            "cross_partition_affected_non_empty_raw_row_share", -1
                        )
                    ),
                    expected_non_empty_share,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                values_valid &= math.isclose(
                    float(metrics.get("cross_partition_affected_test_row_share", -1)),
                    expected_test_share,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                values_valid &= math.isclose(
                    float(
                        metrics.get(
                            "cross_partition_affected_non_empty_test_row_share", -1
                        )
                    ),
                    expected_non_empty_test_share,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                if field == "combined_input":
                    combined_isolated &= all(
                        float(metrics.get(name, -1)) == 0.0
                        for name in (
                            "cross_partition_groups",
                            "cross_partition_affected_raw_rows",
                            "cross_partition_affected_test_rows",
                        )
                    )

    checks = {
        "run_count_matches_evaluation_seeds": bool(run_count_matches),
        "evaluation_seeds_are_present_and_unique": bool(seeds_unique_and_complete),
        "audit_structure_complete": bool(structures_complete),
        "row_counts_conserve_source_and_test": bool(counts_conserve),
        "test_counts_are_consistent_source_subsets": bool(subset_counts_consistent),
        "counts_and_shares_are_finite_nonnegative_and_consistent": bool(values_valid),
        "combined_input_exact_and_token_set_cross_partition_recurrence_zero": bool(
            combined_isolated
        ),
    }
    passed = all(checks.values())
    if enforced and not passed:
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError(f"Feature recurrence release gate failed: {failed}")
    return {
        "enforced": enforced,
        "passed": passed if enforced else None,
        "expected_runs": expected_runs,
        "observed_runs": len(runs),
        "checks": checks,
    }


def feature_recurrence_audit(
    raw_frame: pd.DataFrame,
    split_keys: pd.DataFrame,
    seeds: tuple[int, ...],
    *,
    enforce_release_gate: bool,
) -> dict[str, object]:
    """Audit per-feature equality recurrence under the combined-input split.

    Only aggregate counts and shares are returned. Empty normalized values are
    reported separately and excluded from recurrence groups so missingness does
    not become a universal pseudo-duplicate cluster.
    """

    if not seeds:
        raise ValueError("Feature recurrence audit requires at least one seed")
    if len(raw_frame) != len(split_keys):
        raise ValueError("Raw rows and split keys must have equal length")

    runs: list[dict[str, object]] = []
    partition_equivalence_runs: list[dict[str, bool]] = []
    for seed in seeds:
        manifest = make_split_manifest(
            split_keys,
            SplitSpec(protocol="fingerprint_group", seed=seed),
        )
        partitions = raw_frame[["review_id"]].merge(
            manifest,
            on="review_id",
            how="left",
            validate="one_to_one",
        )["partition"].reset_index(drop=True)
        if partitions.isna().any() or set(partitions) != {"train", "validation", "test"}:
            raise RuntimeError("Feature recurrence manifest must assign every raw row")

        features: dict[str, object] = {}
        partition_equivalence: dict[str, bool] = {}
        for field in FEATURE_RECURRENCE_FIELDS:
            values = _feature_values(raw_frame, field).reset_index(drop=True)
            features[field] = {
                method: _feature_recurrence_metrics(values, partitions, method=method)
                for method in FEATURE_RECURRENCE_METHODS
            }
            partition_equivalence[field] = _feature_method_partitions_identical(values)
        runs.append({"evaluation_seed": int(seed), "features": features})
        partition_equivalence_runs.append(partition_equivalence)

    summary: dict[str, object] = {}
    for field in FEATURE_RECURRENCE_FIELDS:
        summary[field] = {}
        for method in FEATURE_RECURRENCE_METHODS:
            summary[field][method] = {
                metric: aggregate_scalar_runs(
                    [
                        float(run["features"][field][method][metric])
                        for run in runs
                    ]
                )
                for metric in _RECURRENCE_METRICS
            }

    release_gate = _feature_recurrence_release_gate(
        runs,
        expected_rows=len(raw_frame),
        expected_runs=len(seeds),
        enforced=enforce_release_gate,
    )
    methods_identical = {
        field: all(run[field] for run in partition_equivalence_runs)
        for field in FEATURE_RECURRENCE_FIELDS
    }
    return {
        "scope": (
            "raw synthetic rows before partition-local neutral-rating exclusion; audit-only"
        ),
        "reference_protocol": "fingerprint_group",
        "evaluation_seeds": [int(seed) for seed in seeds],
        "split_runs": len(runs),
        "target_bearing_fields_used": [],
        "fields": list(FEATURE_RECURRENCE_FIELDS),
        "methods": {
            "normalized_exact": (
                "Equality after lowercase alphanumeric and whitespace normalization."
            ),
            "token_set_equality": (
                "Equality after normalization, token deduplication, and sorting; not a "
                "threshold, fuzzy, or semantic near-duplicate method."
            ),
        },
        "empty_value_policy": (
            "Empty normalized values are counted separately and excluded from recurrence groups."
        ),
        "summary": summary,
        "methods_identical_on_fixture": methods_identical,
        "release_gate": release_gate,
        "interpretation": {
            "summary": (
                "Authored summary templates are deliberately low-cardinality; recurrence is "
                "exposure, not by itself leakage."
            ),
            "body": (
                "Body-only recurrence can cross partitions even when the combined model input "
                "fingerprint is isolated."
            ),
            "combined_input": (
                "The reference split isolates equality of the combined summary-plus-body input."
            ),
        },
    }
