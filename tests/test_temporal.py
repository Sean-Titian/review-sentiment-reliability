from __future__ import annotations

import json

import pandas as pd
import pytest

from review_reliability.data import SyntheticConfig, generate_synthetic_reviews, make_split_keys
from review_reliability.public_safety import assert_aggregate_report_safe
from review_reliability.splits import make_label_delay_rolling_manifests
from review_reliability.temporal import (
    TEMPORAL_SUMMARY_METRICS,
    _temporal_placebo_gate,
    run_rolling_origin_backtest,
)


def test_rolling_origin_backtest_is_aggregate_safe_and_temporally_ordered() -> None:
    frame = generate_synthetic_reviews(SyntheticConfig(n_rows=600, seed=707))
    result = run_rolling_origin_backtest(
        frame,
        make_split_keys(frame),
        model_seed=809,
        placebo_draws_per_window=5,
        enforce_placebo_gate=False,
    )

    assert result["summary"]["windows"] == 4
    assert result["summary"]["all_test_horizons_non_overlapping"] is True
    assert result["summary"]["all_partitions_strictly_time_ordered"] is True
    assert result["summary"]["all_training_histories_expanding"] is True
    assert result["summary"]["test_label_alignment_placebo_gate"]["draws"] == 20
    sensitivity = result["label_delay_sensitivity"]
    assert sensitivity["delay_days"] == [0, 14, 30]
    assert set(sensitivity["scenarios"]) == {"0", "14", "30"}
    assert sensitivity["all_test_horizons_match_zero_day"] is True
    assert sensitivity["delay_selected_post_hoc"] is False
    assert sensitivity["availability_time_observed"] is False
    assert sensitivity["operational_target_validated"] is False
    assert result["windows"] == sensitivity["scenarios"]["0"]["windows"]
    assert result["summary"] == sensitivity["scenarios"]["0"]["summary"]
    assert [window["future_raw_rows_excluded"] for window in result["windows"]] == [
        180,
        120,
        60,
        0,
    ]
    for window in result["windows"]:
        bounds = window["time_bounds"]
        assert bounds["train"]["max"] < bounds["validation"]["min"]
        assert bounds["validation"]["max"] < bounds["test"]["min"]
        assert set(window["raw_cross_boundary_overlap_audit"]) == {
            "text_fingerprint",
            "near_text_fingerprint",
            "user_group",
            "product_group",
        }
        assert window["model"]["average_precision_attention"] >= 0.0
        assert window["model"]["budget_cutoff_fraction_selected"] > 0.0

    zero_windows = sensitivity["scenarios"]["0"]["windows"]
    for delay in (0, 14, 30):
        scenario = sensitivity["scenarios"][str(delay)]
        assert scenario["all_test_horizons_match_zero_day"] is True
        assert scenario["summary"]["test_label_alignment_placebo_gate"]["draws"] == 20
        assert scenario["summary"]["test_label_alignment_placebo_gate"]["enforced"] is False
        for zero_window, window in zip(zero_windows, scenario["windows"], strict=True):
            counts = window["raw_partition_rows"]
            assert set(counts) == {"train", "validation", "embargo", "test", "future"}
            assert sum(counts.values()) == len(frame)
            assert window["time_bounds"]["test"] == zero_window["time_bounds"]["test"]
            assert counts["test"] == zero_window["raw_partition_rows"]["test"]
            assert pd.Timestamp(window["label_availability_cutoff"]) == (
                pd.Timestamp(window["test_start"]) - pd.Timedelta(days=delay)
            )
            assert window["test_label_alignment_placebo"]["draws"] == 5
            assert window["test_label_alignment_placebo"]["gate"]["enforced"] is False

    assert set(sensitivity["paired_deltas_vs_0"]) == {"14", "30"}
    for delay, comparison in sensitivity["paired_deltas_vs_0"].items():
        assert len(comparison["windows"]) == 4
        delayed_windows = sensitivity["scenarios"][delay]["windows"]
        for index, paired_window in enumerate(comparison["windows"]):
            for metric in TEMPORAL_SUMMARY_METRICS:
                assert paired_window["metric_deltas"][metric] == pytest.approx(
                    delayed_windows[index]["model"][metric]
                    - zero_windows[index]["model"][metric]
                )

    serialized = json.dumps(result)
    assert "synthetic_review_" not in serialized
    assert "review_id" not in serialized
    assert_aggregate_report_safe({"rolling_origin_backtest": result})


def test_rolling_origin_rejects_mismatched_split_keys() -> None:
    frame = generate_synthetic_reviews(SyntheticConfig(n_rows=400, seed=919))
    keys = make_split_keys(frame)
    keys.loc[0, "event_time"] = keys.loc[0, "event_time"] + pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="derived from the supplied raw frame"):
        run_rolling_origin_backtest(
            frame,
            keys,
            placebo_draws_per_window=5,
            enforce_placebo_gate=False,
        )


def test_release_gated_rolling_origins_require_twenty_placebos_per_window() -> None:
    frame = generate_synthetic_reviews(SyntheticConfig(n_rows=400, seed=929))
    with pytest.raises(ValueError, match="20 placebos per window"):
        run_rolling_origin_backtest(
            frame,
            make_split_keys(frame),
            placebo_draws_per_window=19,
            enforce_placebo_gate=True,
        )

    with pytest.raises(ValueError, match="20 placebos per window"):
        run_rolling_origin_backtest(
            frame,
            make_split_keys(frame),
            placebo_draws_per_window=21,
            enforce_placebo_gate=True,
        )


def test_label_delay_grid_fails_closed_for_invalid_or_infeasible_delays() -> None:
    frame = generate_synthetic_reviews(SyntheticConfig(n_rows=600, seed=937))
    keys = make_split_keys(frame)

    with pytest.raises(ValueError, match="non-negative finite integers"):
        make_label_delay_rolling_manifests(keys, delay_days=(0, -1, 14))
    with pytest.raises(ValueError, match="empty train or validation partition"):
        make_label_delay_rolling_manifests(keys, delay_days=(0, 14, 10_000))


def test_temporal_placebo_gate_passes_fails_closed_and_supports_diagnostics() -> None:
    passing_control = {
        "roc_auc": 0.50,
        "average_precision_over_prevalence": 1.0,
        "lift_at_10pct_budget": 1.0,
    }
    passing = _temporal_placebo_gate([passing_control] * 20, enforced=True)
    assert passing["passed"] is True
    assert passing["enforced"] is True
    assert passing["draws"] == 20

    failing_control = {**passing_control, "roc_auc": 0.56}
    with pytest.raises(RuntimeError, match="did not return to chance"):
        _temporal_placebo_gate([failing_control] * 20, enforced=True)

    diagnostic = _temporal_placebo_gate([failing_control] * 20, enforced=False)
    assert diagnostic["passed"] is None
    assert diagnostic["enforced"] is False
    assert diagnostic["summary"]["roc_auc"]["mean"] == pytest.approx(0.56)
