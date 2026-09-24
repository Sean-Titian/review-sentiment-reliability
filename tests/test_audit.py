from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from review_reliability.audit import (
    FEATURE_RECURRENCE_FIELDS,
    FEATURE_RECURRENCE_METHODS,
    _feature_method_partitions_identical,
    _feature_recurrence_metrics,
    _feature_recurrence_release_gate,
    feature_recurrence_audit,
)
from review_reliability.data import SyntheticConfig, generate_synthetic_reviews, make_split_keys


def test_recurrence_excludes_empty_values_and_uses_explicit_denominators() -> None:
    values = pd.Series(["shared value", "shared value", "other value", None])
    partitions = pd.Series(["train", "test", "validation", "test"])

    metrics = _feature_recurrence_metrics(
        values,
        partitions,
        method="normalized_exact",
    )

    assert metrics["non_empty_raw_rows"] == 3
    assert metrics["empty_raw_rows"] == 1
    assert metrics["cross_partition_groups"] == 1
    assert metrics["cross_partition_affected_raw_rows"] == 2
    assert metrics["cross_partition_affected_non_empty_raw_row_share"] == pytest.approx(
        2 / 3
    )
    assert metrics["test_rows"] == 2
    assert metrics["non_empty_test_rows"] == 1
    assert metrics["cross_partition_affected_test_rows"] == 1
    assert metrics["cross_partition_affected_test_row_share"] == 0.5
    assert metrics["cross_partition_affected_non_empty_test_row_share"] == 1.0


def test_token_set_equality_is_distinct_from_normalized_exact_equality() -> None:
    values = pd.Series(["alpha beta", "beta alpha", "gamma", None])
    partitions = pd.Series(["train", "test", "validation", "test"])

    exact = _feature_recurrence_metrics(values, partitions, method="normalized_exact")
    token_set = _feature_recurrence_metrics(
        values,
        partitions,
        method="token_set_equality",
    )

    assert exact["cross_partition_groups"] == 0
    assert token_set["cross_partition_groups"] == 1
    assert token_set["cross_partition_affected_raw_rows"] == 2
    assert _feature_method_partitions_identical(values) is False


def _valid_gate_run() -> dict[str, object]:
    partitions = pd.Series(["train", "train", "validation", "validation", "test", "test"])
    values_by_field = {
        "summary": pd.Series(["shared", "a", "b", "c", "shared", "d"]),
        "body": pd.Series(["a", "body", "b", "c", "d", "body"]),
        "combined_input": pd.Series(["a0", "a1", "a2", "a3", "a4", "a5"]),
    }
    return {
        "evaluation_seed": 11,
        "features": {
            field: {
                method: _feature_recurrence_metrics(
                    values_by_field[field],
                    partitions,
                    method=method,
                )
                for method in FEATURE_RECURRENCE_METHODS
            }
            for field in FEATURE_RECURRENCE_FIELDS
        },
    }


def test_feature_recurrence_release_gate_fails_closed_on_tampering() -> None:
    run = _valid_gate_run()
    gate = _feature_recurrence_release_gate(
        [run],
        expected_rows=6,
        expected_runs=1,
        enforced=True,
    )
    assert gate["passed"] is True
    assert all(gate["checks"].values())

    overlap = deepcopy(run)
    overlap["features"]["combined_input"]["normalized_exact"][
        "cross_partition_groups"
    ] = 1
    with pytest.raises(RuntimeError, match="combined_input"):
        _feature_recurrence_release_gate(
            [overlap],
            expected_rows=6,
            expected_runs=1,
            enforced=True,
        )

    invalid_count = deepcopy(run)
    invalid_count["features"]["body"]["normalized_exact"]["non_empty_raw_rows"] = -1
    with pytest.raises(RuntimeError, match="finite_nonnegative"):
        _feature_recurrence_release_gate(
            [invalid_count],
            expected_rows=6,
            expected_runs=1,
            enforced=True,
        )

    missing_method = deepcopy(run)
    del missing_method["features"]["summary"]["token_set_equality"]
    with pytest.raises(RuntimeError, match="audit_structure_complete"):
        _feature_recurrence_release_gate(
            [missing_method],
            expected_rows=6,
            expected_runs=1,
            enforced=True,
        )

    impossible_test_subset = deepcopy(run)
    metrics = impossible_test_subset["features"]["body"]["normalized_exact"]
    metrics["cross_partition_affected_test_rows"] = 3
    metrics["cross_partition_affected_non_empty_test_row_share"] = 1.5
    with pytest.raises(RuntimeError, match="consistent_source_subsets"):
        _feature_recurrence_release_gate(
            [impossible_test_subset],
            expected_rows=6,
            expected_runs=1,
            enforced=True,
        )

    fractional_count = deepcopy(run)
    fractional_count["features"]["summary"]["normalized_exact"][
        "unique_non_empty_groups"
    ] = 4.5
    with pytest.raises(RuntimeError, match="finite_nonnegative"):
        _feature_recurrence_release_gate(
            [fractional_count],
            expected_rows=6,
            expected_runs=1,
            enforced=True,
        )


def test_feature_recurrence_release_gate_requires_unique_seed_evidence() -> None:
    first = _valid_gate_run()
    duplicate = deepcopy(first)

    with pytest.raises(RuntimeError, match="present_and_unique"):
        _feature_recurrence_release_gate(
            [first, duplicate],
            expected_rows=6,
            expected_runs=2,
            enforced=True,
        )


def test_synthetic_feature_recurrence_audit_exposes_only_aggregate_limits() -> None:
    frame = generate_synthetic_reviews(SyntheticConfig(n_rows=600, seed=101))
    audit = feature_recurrence_audit(
        frame,
        make_split_keys(frame),
        (17, 23),
        enforce_release_gate=True,
    )

    assert audit["target_bearing_fields_used"] == []
    assert "runs" not in audit
    assert audit["split_runs"] == 2
    assert audit["release_gate"]["passed"] is True
    assert all(audit["release_gate"]["checks"].values())
    summary = audit["summary"]
    for method in FEATURE_RECURRENCE_METHODS:
        assert summary["combined_input"][method]["cross_partition_groups"]["max"] == 0
        assert summary["summary"][method]["cross_partition_groups"]["min"] > 0
        assert summary["body"][method]["cross_partition_groups"]["min"] > 0
    assert "semantic" in audit["methods"]["token_set_equality"]

    target_changed = frame.copy()
    target_changed["rating"] = 6 - target_changed["rating"]
    target_changed["needs_attention"] = 1
    changed_audit = feature_recurrence_audit(
        target_changed,
        make_split_keys(frame),
        (17, 23),
        enforce_release_gate=True,
    )
    assert changed_audit == audit
