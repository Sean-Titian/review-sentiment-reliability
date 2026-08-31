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
        ".pkl",
        ".pickle",
        ".joblib",
        ".onnx",
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
    assert all(path.stat().st_size < 1_000_000 for path in files)


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
    assert report["contract_version"] == "2.0"
    assert report["synthetic_data"] is True
    assert report["source_rows_included"] is False
    assert report["runtime_controls"]["negative_control_mode"] == (
        "multi_draw_train_and_test_label_placebos_fail_closed"
    )
    assert report["runtime_controls"]["negative_control_gate"]["passed"] is True
    assert report["runtime_controls"]["permutation_draws_per_fingerprint_split"] == 5
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
