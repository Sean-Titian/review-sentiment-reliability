"""Fail-closed checks for the public aggregate artifact."""

from __future__ import annotations

import json
import re
from typing import Any

_FORBIDDEN_TEXT_PATTERNS = (
    re.compile(r"[A-Za-z]:\\Users\\", re.IGNORECASE),
    re.compile(r"/" + r"Users/", re.IGNORECASE),
    re.compile(r"/" + r"home/", re.IGNORECASE),
    re.compile(r"synthetic_review_\d+", re.IGNORECASE),
    re.compile(r"synthetic_user_\d+", re.IGNORECASE),
    re.compile(r"synthetic_product_\d+", re.IGNORECASE),
    re.compile(r"drive\.google\.com", re.IGNORECASE),
)
_FORBIDDEN_KEYS = {
    "review_ids",
    "user_ids",
    "product_ids",
    "raw_records",
    "row_payloads",
    "review_texts",
    "review_text",
    "review_id",
    "user_id",
    "product_id",
    "data_path",
    "local_path",
    "source_path",
    "predictions_by_row",
    "probabilities_by_row",
}


def assert_aggregate_report_safe(report: dict[str, Any]) -> None:
    """Reject row-level payloads, linkable identifiers, and local paths."""

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if str(key).lower() in _FORBIDDEN_KEYS:
                    raise ValueError(f"Forbidden row-level report key: {key}")
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)
        elif isinstance(value, str):
            for pattern in _FORBIDDEN_TEXT_PATTERNS:
                if pattern.search(value):
                    raise ValueError("Public report contains an identifier or local-path pattern")

    walk(report)
    serialized = json.dumps(report, ensure_ascii=False, allow_nan=False)
    if len(serialized.encode("utf-8")) > 2_000_000:
        raise ValueError("Aggregate report unexpectedly exceeds 2 MB")
