"""Aggregate-only duplicate and feature-profile audits."""

from __future__ import annotations

import pandas as pd

from review_reliability.data import TARGET_COLUMN, near_text_fingerprint, text_fingerprint


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
