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
    assert report["contract_version"] == "4.0"
    assert report["runtime_controls"]["protocol_isolation_fail_closed"] is True
    assert report["decision_contract"]["causal_claim"] is False
    assert report["runtime_controls"]["target_bearing_split_columns"] == []
    assert report["conditional_uncertainty"]["reference_protocol"] == "fingerprint_group"
    assert report["conditional_uncertainty"]["result"]["draws_attempted"] == 40
    capacity = report["queue_capacity_sensitivity"]
    assert capacity["pre_specified_budget_shares"] == [0.05, 0.10, 0.20]
    assert capacity["capacity_selected_post_hoc"] is False
    assert [
        point["budget_share"]
        for point in capacity["model_across_matched_seeds"]["points"]
    ] == [0.05, 0.10, 0.20]
    diagnostic_gate = report["runtime_controls"]["negative_control_gate"]
    assert diagnostic_gate["enforced"] is False
    for design in diagnostic_gate["designs"].values():
        assert [
            point["budget_share"] for point in design["capacity_lift"]["points"]
        ] == [0.05, 0.10, 0.20]
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


def _capacity_curve_for_lifts(lifts: tuple[float, float, float]) -> dict[str, object]:
    return {
        "budget_shares": [0.05, 0.10, 0.20],
        "points": [
            {
                "budget_share": share,
                "budget_count": count,
                "budget_weight": float(count),
                "total_weight": 100.0,
                "effective_budget_share": share,
                "budget_cutoff_tied_rows": 1,
                "budget_cutoff_tied_weight": 1.0,
                "budget_cutoff_fraction_selected": 1.0,
                "precision_at_budget": 0.2 * lift,
                "recall_at_budget": share * lift,
                "lift_at_budget": lift,
            }
            for share, count, lift in zip(
                (0.05, 0.10, 0.20), (5, 10, 20), lifts, strict=True
            )
        ],
    }


def _null_control(
    *, roc_auc: float = 0.50, lift_at_budget: float = 1.0
) -> dict[str, float | int]:
    return {
        "attention_prevalence": 0.2,
        "average_precision_attention": 0.2,
        "roc_auc_supplementary": roc_auc,
        "budget_count": 10,
        "budget_cutoff_tied_rows": 1,
        "budget_cutoff_tied_weight": 1.0,
        "budget_cutoff_fraction_selected": 1.0,
        "precision_at_budget": 0.2 * lift_at_budget,
        "recall_at_budget": 0.1 * lift_at_budget,
        "lift_at_budget": lift_at_budget,
    }


def test_negative_control_is_checked_per_run_and_fails_closed() -> None:
    passing_control = _null_control()
    passing_run = {
        "train_label_permutation_controls": [passing_control] * 5,
        "test_label_alignment_placebos": [passing_control] * 5,
        "train_label_permutation_capacity_controls": [
            _capacity_curve_for_lifts((1.0, 1.0, 1.0))
        ]
        * 5,
        "test_label_alignment_capacity_placebos": [
            _capacity_curve_for_lifts((1.0, 1.0, 1.0))
        ]
        * 5,
    }
    summary = _enforce_negative_control_runs([passing_run])
    assert summary["passed"] is True
    assert summary["designs"]["train_label_permutation"]["draws"] == 5

    failing_control = _null_control(roc_auc=0.71)
    failing_run = {
        "train_label_permutation_controls": [passing_control, failing_control],
        "test_label_alignment_placebos": [passing_control] * 2,
        "train_label_permutation_capacity_controls": [
            _capacity_curve_for_lifts((1.0, 1.0, 1.0))
        ]
        * 2,
        "test_label_alignment_capacity_placebos": [
            _capacity_curve_for_lifts((1.0, 1.0, 1.0))
        ]
        * 2,
    }
    with pytest.raises(RuntimeError, match="draw 1"):
        _enforce_negative_control_runs([failing_run])


def test_negative_control_aggregate_mean_is_fail_closed() -> None:
    shifted = _null_control(roc_auc=0.56)
    run = {
        "train_label_permutation_controls": [shifted] * 5,
        "test_label_alignment_placebos": [shifted] * 5,
        "train_label_permutation_capacity_controls": [
            _capacity_curve_for_lifts((1.0, 1.0, 1.0))
        ]
        * 5,
        "test_label_alignment_capacity_placebos": [
            _capacity_curve_for_lifts((1.0, 1.0, 1.0))
        ]
        * 5,
    }
    with pytest.raises(RuntimeError, match="aggregate roc_auc"):
        _enforce_negative_control_runs([run])


def test_negative_control_requires_draws() -> None:
    run = {
        "train_label_permutation_controls": [],
        "test_label_alignment_placebos": [],
        "train_label_permutation_capacity_controls": [],
        "test_label_alignment_capacity_placebos": [],
    }
    with pytest.raises(RuntimeError, match="requires negative-control draws"):
        _enforce_negative_control_runs([run])


@pytest.mark.parametrize(
    ("failing_lifts", "capacity_label"),
    (
        ((3.01, 1.0, 1.0), "5%"),
        ((1.0, 1.0, 1.71), "20%"),
        ((1.0, 1.0, 0.39), "20%"),
    ),
)
def test_capacity_negative_control_is_checked_at_each_pre_specified_share(
    failing_lifts: tuple[float, float, float], capacity_label: str
) -> None:
    passing_control = _null_control()
    run = {
        "train_label_permutation_controls": [passing_control],
        "test_label_alignment_placebos": [passing_control],
        "train_label_permutation_capacity_controls": [
            _capacity_curve_for_lifts(failing_lifts)
        ],
        "test_label_alignment_capacity_placebos": [
            _capacity_curve_for_lifts((1.0, 1.0, 1.0))
        ],
    }
    with pytest.raises(RuntimeError, match=f"capacity {capacity_label}"):
        _enforce_negative_control_runs([run])


def test_primary_capacity_null_point_must_match_legacy_10pct_metric() -> None:
    passing_control = _null_control()
    mismatched_curve = _capacity_curve_for_lifts((1.0, 1.01, 1.0))
    run = {
        "train_label_permutation_controls": [passing_control],
        "test_label_alignment_placebos": [passing_control],
        "train_label_permutation_capacity_controls": [mismatched_curve],
        "test_label_alignment_capacity_placebos": [
            _capacity_curve_for_lifts((1.0, 1.0, 1.0))
        ],
    }
    with pytest.raises(RuntimeError, match="does not match legacy"):
        _enforce_negative_control_runs([run])


@pytest.mark.parametrize("lift", (0.24, 2.01))
def test_legacy_10pct_null_lift_bounds_remain_fail_closed(lift: float) -> None:
    failing_control = _null_control(lift_at_budget=lift)
    failing_curve = _capacity_curve_for_lifts((1.0, lift, 1.0))
    run = {
        "train_label_permutation_controls": [failing_control],
        "test_label_alignment_placebos": [_null_control()],
        "train_label_permutation_capacity_controls": [failing_curve],
        "test_label_alignment_capacity_placebos": [
            _capacity_curve_for_lifts((1.0, 1.0, 1.0))
        ],
    }
    with pytest.raises(RuntimeError, match="draw 0"):
        _enforce_negative_control_runs([run])


def test_capacity_negative_control_aggregate_mean_fails_closed() -> None:
    passing_control = _null_control()
    shifted_curve = _capacity_curve_for_lifts((1.51, 1.0, 1.0))
    passing_curve = _capacity_curve_for_lifts((1.0, 1.0, 1.0))
    run = {
        "train_label_permutation_controls": [passing_control] * 5,
        "test_label_alignment_placebos": [passing_control] * 5,
        "train_label_permutation_capacity_controls": [shifted_curve] * 5,
        "test_label_alignment_capacity_placebos": [passing_curve] * 5,
    }
    with pytest.raises(RuntimeError, match="aggregate lift at 5% capacity"):
        _enforce_negative_control_runs([run])


def test_capacity_null_threshold_report_cannot_mutate_module_contract() -> None:
    passing_control = _null_control()
    passing_curve = _capacity_curve_for_lifts((1.0, 1.0, 1.0))
    run = {
        "train_label_permutation_controls": [passing_control] * 5,
        "test_label_alignment_placebos": [passing_control] * 5,
        "train_label_permutation_capacity_controls": [passing_curve] * 5,
        "test_label_alignment_capacity_placebos": [passing_curve] * 5,
    }

    first = _enforce_negative_control_runs([run])
    first["thresholds"]["capacity_lift"][0]["per_draw"][1] = -1.0
    second = _enforce_negative_control_runs([run])

    assert second["thresholds"]["capacity_lift"][0]["per_draw"] == [0.0, 3.0]


@pytest.mark.parametrize("draws", (4, 6))
def test_release_gate_requires_exactly_five_permutations_per_strict_split(
    draws: int,
) -> None:
    with pytest.raises(ValueError, match="exactly five permutation draws"):
        run_synthetic_benchmark(
            permutation_draws=draws,
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
