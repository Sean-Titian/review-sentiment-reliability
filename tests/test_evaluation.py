from __future__ import annotations

import pytest

from review_reliability.data import SyntheticConfig
from review_reliability.evaluation import (
    _enforce_negative_control_runs,
    run_synthetic_benchmark,
)
from review_reliability.public_safety import assert_aggregate_report_safe


def test_end_to_end_report_is_aggregate_safe_and_complete() -> None:
    report = run_synthetic_benchmark(
        config=SyntheticConfig(n_rows=360, seed=101),
        seeds=(17,),
        protocols=("row_random", "fingerprint_group", "forward_time"),
        enforce_negative_control=False,
    )
    assert report["synthetic_data"] is True
    assert report["source_rows_included"] is False
    assert set(report["protocols"]) == {
        "row_random",
        "fingerprint_group",
        "forward_time",
    }
    strict = report["protocols"]["fingerprint_group"]
    assert strict["overlap_audit"]["near_text_fingerprint"]["shared_groups"]["mean"] == 0
    assert strict["negative_control_train_label_permutation"] is not None
    assert report["decision_contract"]["causal_claim"] is False
    assert report["runtime_controls"]["target_bearing_split_columns"] == []
    assert_aggregate_report_safe(report)


def test_stress_cases_keep_finite_aggregate_metrics() -> None:
    report = run_synthetic_benchmark(
        config=SyntheticConfig(n_rows=320, seed=103),
        seeds=(23,),
        protocols=("row_random", "fingerprint_group", "forward_time"),
        enforce_negative_control=False,
    )
    stress = report["protocols"]["forward_time"]["stress"]
    expected = {
        "missing_summary_20pct",
        "missing_body_20pct",
        "case_and_punctuation",
        "unknown_tokens",
        "body_truncated_8_tokens",
        "token_dropout_25pct",
        "higher_attention_prior",
    }
    assert set(stress) == expected
    for case in stress.values():
        for aggregate in case.values():
            assert np_is_finite(aggregate["mean"])


def np_is_finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


def test_negative_control_is_checked_per_run_and_fails_closed() -> None:
    passing = {
        "negative_control": {
            "attention_prevalence": 0.2,
            "pr_auc_attention": 0.22,
            "roc_auc_supplementary": 0.51,
        }
    }
    _enforce_negative_control_runs([passing, passing])
    failing = {
        "negative_control": {
            "attention_prevalence": 0.2,
            "pr_auc_attention": 0.7,
            "roc_auc_supplementary": 0.9,
        }
    }
    with pytest.raises(RuntimeError, match="run 1"):
        _enforce_negative_control_runs([passing, failing])
