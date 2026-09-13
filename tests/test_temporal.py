from __future__ import annotations

import json

import pandas as pd
import pytest

from review_reliability.data import SyntheticConfig, generate_synthetic_reviews, make_split_keys
from review_reliability.public_safety import assert_aggregate_report_safe
from review_reliability.temporal import (
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
