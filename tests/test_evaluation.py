from __future__ import annotations

import json

import pytest

from review_reliability.cli import write_aggregate_report
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
        permutation_draws=1,
        bootstrap_draws=40,
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
    assert strict["placebo_test_label_alignment"] is not None
    assert strict["conflict_retention_gate"]["all_runs_passed"] is True
    assert report["contract_version"] == "3.0"
    assert report["runtime_controls"]["protocol_isolation_fail_closed"] is True
    assert report["decision_contract"]["causal_claim"] is False
    assert report["runtime_controls"]["target_bearing_split_columns"] == []
    assert report["conditional_uncertainty"]["reference_protocol"] == "fingerprint_group"
    assert report["conditional_uncertainty"]["result"]["draws_attempted"] == 40
    assert report["rolling_origin_backtest"]["summary"]["windows"] == 4
    assert_aggregate_report_safe(report)


def test_stress_cases_keep_finite_aggregate_metrics() -> None:
    report = run_synthetic_benchmark(
        config=SyntheticConfig(n_rows=320, seed=103),
        seeds=(23,),
        protocols=("row_random", "fingerprint_group", "forward_time"),
        permutation_draws=1,
        bootstrap_draws=40,
        enforce_negative_control=False,
    )
    stress = report["protocols"]["forward_time"]["stress"]
    expected = {
        "missing_summary_20pct",
        "missing_body_20pct",
        "case_and_punctuation",
        "oov_only_user_content",
        "body_truncated_8_tokens",
        "token_dropout_25pct",
        "token_replacement_25pct",
        "higher_attention_prior",
    }
    assert set(stress) == expected
    for case in stress.values():
        status = case["ranking_at_budget_status"]
        assert status["runs"] >= 1
        for metric, aggregate in case.items():
            if metric == "ranking_at_budget_status":
                continue
            assert np_is_finite(aggregate["mean"])

    oov_only = report["protocols"]["fingerprint_group"]["stress"][
        "oov_only_user_content"
    ]
    assert oov_only["ranking_at_budget_status"]["defined_in_all_runs"] is False
    assert "recall_at_budget" not in oov_only
    assert "lift_at_budget" not in oov_only
    assert "delta_recall_at_budget" not in oov_only
    assert "delta_lift_at_budget" not in oov_only


def np_is_finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


def test_negative_control_is_checked_per_run_and_fails_closed() -> None:
    passing_control = {
        "attention_prevalence": 0.2,
        "average_precision_attention": 0.2,
        "roc_auc_supplementary": 0.50,
        "lift_at_budget": 1.0,
    }
    passing_run = {
        "train_label_permutation_controls": [passing_control] * 5,
        "test_label_alignment_placebos": [passing_control] * 5,
    }
    summary = _enforce_negative_control_runs([passing_run])
    assert summary["passed"] is True
    assert summary["designs"]["train_label_permutation"]["draws"] == 5

    failing_control = {
        **passing_control,
        "roc_auc_supplementary": 0.71,
    }
    failing_run = {
        "train_label_permutation_controls": [passing_control, failing_control],
        "test_label_alignment_placebos": [passing_control] * 2,
    }
    with pytest.raises(RuntimeError, match="draw 1"):
        _enforce_negative_control_runs([failing_run])


def test_negative_control_aggregate_mean_is_fail_closed() -> None:
    shifted = {
        "attention_prevalence": 0.2,
        "average_precision_attention": 0.2,
        "roc_auc_supplementary": 0.56,
        "lift_at_budget": 1.0,
    }
    run = {
        "train_label_permutation_controls": [shifted] * 5,
        "test_label_alignment_placebos": [shifted] * 5,
    }
    with pytest.raises(RuntimeError, match="aggregate roc_auc"):
        _enforce_negative_control_runs([run])


def test_negative_control_requires_draws() -> None:
    run = {
        "train_label_permutation_controls": [],
        "test_label_alignment_placebos": [],
    }
    with pytest.raises(RuntimeError, match="requires negative-control draws"):
        _enforce_negative_control_runs([run])


def test_release_gate_requires_five_permutations_per_strict_split() -> None:
    with pytest.raises(ValueError, match="at least five permutation draws"):
        run_synthetic_benchmark(
            permutation_draws=4,
            enforce_negative_control=True,
        )


def test_release_gate_requires_two_thousand_cluster_bootstrap_draws() -> None:
    with pytest.raises(ValueError, match="2,000 cluster-bootstrap draws"):
        run_synthetic_benchmark(
            bootstrap_draws=1_999,
            enforce_negative_control=True,
        )


def test_report_writer_canonicalizes_immaterial_float_tails(tmp_path) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    write_aggregate_report(
        {"metric": 0.123456789012, "negative_zero": -1e-14},
        first_path,
    )
    write_aggregate_report(
        {"metric": 0.123456789013, "negative_zero": 1e-14},
        second_path,
    )

    assert first_path.read_bytes() == second_path.read_bytes()
    assert b"\r\n" not in first_path.read_bytes()
    payload = json.loads(first_path.read_text(encoding="utf-8"))
    assert payload == {"metric": 0.123456789, "negative_zero": 0.0}
