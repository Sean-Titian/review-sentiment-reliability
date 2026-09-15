from __future__ import annotations

import json
from pathlib import Path

import pytest

from review_reliability.public_safety import assert_aggregate_report_safe

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {
    ".git",
    ".pytest-tmp",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "build",
    "dist",
    "__pycache__",
}


def _public_files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not (set(path.parts) & EXCLUDED_PARTS)
        and not any(part.endswith(".egg-info") for part in path.parts)
    ]


def test_report_safety_rejects_row_ids_and_local_paths() -> None:
    with pytest.raises(ValueError, match="identifier"):
        assert_aggregate_report_safe({"value": "synthetic_review_000123"})
    with pytest.raises(ValueError, match="local-path"):
        local_path = "C:" + "\\Users\\person\\private.csv"
        assert_aggregate_report_safe({"value": local_path})
    with pytest.raises(ValueError, match="row-level"):
        assert_aggregate_report_safe({"review_texts": ["not allowed"]})
    with pytest.raises(ValueError, match="row-level"):
        assert_aggregate_report_safe({"user_id": "stable-user"})
    with pytest.raises(ValueError, match="local-path"):
        assert_aggregate_report_safe({"value": "/" + "home/person/private.csv"})


def test_repository_contains_no_data_model_archive_or_large_file() -> None:
    forbidden_suffixes = {
        ".csv",
        ".tsv",
        ".parquet",
        ".sqlite",
        ".sqlite3",
        ".db",
        ".jsonl",
        ".ndjson",
        ".pkl",
        ".pickle",
        ".joblib",
        ".onnx",
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
        ".zip",
        ".gz",
        ".7z",
        ".pdf",
        ".docx",
        ".pptx",
        ".mp4",
    }
    files = _public_files()
    assert all(path.suffix.lower() not in forbidden_suffixes for path in files)
    assert all(path.name != ".env" and not path.name.startswith(".env.") for path in files)
    assert all(path.stat().st_size < 1_000_000 for path in files)


def test_repository_ignore_rules_cover_common_private_data_and_model_formats() -> None:
    ignore_text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in (
        "*.jsonl",
        "*.ndjson",
        "*.db",
        "*.pt",
        "*.pth",
        "*.ckpt",
        "*.safetensors",
        ".env",
        ".env.*",
    ):
        assert pattern in ignore_text


@pytest.mark.parametrize(
    "forbidden_key",
    (
        "cluster_ids",
        "cluster_keys",
        "resample_indices",
        "sample_indices",
        "row_indices",
        "bootstrap_draws",
    ),
)
def test_report_safety_rejects_bootstrap_rows_or_cluster_keys(forbidden_key: str) -> None:
    with pytest.raises(ValueError, match="row-level"):
        assert_aggregate_report_safe({forbidden_key: [0, 1]})


def test_repository_text_has_no_private_path_course_name_or_drive_link() -> None:
    forbidden = (
        "C:" + "\\Users\\",
        "/" + "Users/",
        "\u76f4\u901a\u7845\u8c37",
        "drive." + "google.com",
    )
    for path in _public_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert not any(token in text for token in forbidden), path


def test_tracked_synthetic_benchmark_is_aggregate_safe() -> None:
    report_path = ROOT / "reports" / "synthetic-benchmark.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["contract_version"] == "4.0"
    assert report["synthetic_data"] is True
    assert report["source_rows_included"] is False
    assert report["source_trained_artifacts_included"] is False
    assert report["runtime_controls"]["negative_control_mode"] == (
        "multi_draw_train_and_test_label_placebos_fail_closed"
    )
    assert report["runtime_controls"]["negative_control_gate"]["passed"] is True
    assert report["runtime_controls"]["negative_control_gate"]["enforced"] is True
    assert report["runtime_controls"]["negative_control_gate"][
        "multiplicity_controlled"
    ] is False
    assert report["runtime_controls"]["negative_control_gate"][
        "permutation_draws_per_split_fixed"
    ] == 5
    assert report["runtime_controls"]["permutation_draws_per_fingerprint_split"] == 5
    capacity = report["queue_capacity_sensitivity"]
    assert capacity["pre_specified_budget_shares"] == [0.05, 0.10, 0.20]
    assert capacity["matches_release_capacity_contract"] is True
    assert capacity["capacity_selected_post_hoc"] is False
    capacity_null_scope = report["runtime_controls"]["capacity_null_gate_scope"]
    assert capacity_null_scope.startswith("heuristic train-label permutation")
    assert all(share in capacity_null_scope for share in ("5%", "10%", "20%"))
    bootstrap_gate = report["conditional_uncertainty"]["result"]["release_gate"]
    assert bootstrap_gate["enforced"] is True
    assert bootstrap_gate["passed"] is True
    assert bootstrap_gate["capacity_shares_exact_default"] is True
    assert bootstrap_gate["capacity_curve_points_complete"] is True
    assert bootstrap_gate["confidence_level_exact_default"] is True
    assert bootstrap_gate["cluster_unit_exact_default"] is True
    capacity_points = report["conditional_uncertainty"]["result"]["capacity_curve"][
        "points"
    ]
    assert [point["budget_share"] for point in capacity_points] == [0.05, 0.10, 0.20]
    assert all(
        point["conditional_intervals"]["lift_at_budget"]["valid_draws"] == 2_000
        for point in capacity_points
    )
    assert report["rolling_origin_backtest"]["summary"][
        "all_test_horizons_non_overlapping"
    ] is True
    assert report["rolling_origin_backtest"]["summary"][
        "all_training_histories_expanding"
    ] is True
    rolling_windows = report["rolling_origin_backtest"]["windows"]
    assert len(rolling_windows) == 4
    assert all(
        window["test_label_alignment_placebo"]["gate"]["enforced"] is True
        and window["test_label_alignment_placebo"]["gate"]["passed"] is True
        for window in rolling_windows
    )
    pooled_temporal_gate = report["rolling_origin_backtest"]["summary"][
        "test_label_alignment_placebo_gate"
    ]
    assert pooled_temporal_gate["enforced"] is True
    assert pooled_temporal_gate["passed"] is True
    strict_design = report["protocols"]["fingerprint_group"]["negative_control_design"]
    assert strict_design["train_label_permutation_draws"] == 15
    assert strict_design["test_label_alignment_draws"] == 15
    assert "average_precision_attention" in report["metric_definitions"]

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {key for nested in value.values() for key in keys(nested)}
        if isinstance(value, list):
            return {key for nested in value for key in keys(nested)}
        return set()

    assert "pr_auc_attention" not in keys(report)
    assert_aggregate_report_safe(report)
