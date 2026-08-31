"""Deterministic serving-time text and prevalence stress fixtures."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from review_reliability.data import TARGET_COLUMN
from review_reliability.modeling import FEATURE_COLUMNS

OOV_SENTINELS = (
    "novelsummarytokenzx",
    "novelbodytokenqy",
    "driftwordqz",
    "novelreplacementtokenzx",
)


def _copy_features(labeled_frame: pd.DataFrame) -> pd.DataFrame:
    return labeled_frame.loc[:, list(FEATURE_COLUMNS)].copy()


def text_stress_cases(labeled_test: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """Build feature-only perturbations; targets and post-outcome fields are absent."""

    rng = np.random.default_rng(seed)
    cases: dict[str, pd.DataFrame] = {}

    missing_summary = _copy_features(labeled_test)
    summary_mask = rng.random(len(missing_summary)) < 0.20
    missing_summary.loc[summary_mask, "summary"] = None
    cases["missing_summary_20pct"] = missing_summary

    missing_body = _copy_features(labeled_test)
    body_mask = rng.random(len(missing_body)) < 0.20
    missing_body.loc[body_mask, "text"] = None
    cases["missing_body_20pct"] = missing_body

    case_punctuation = _copy_features(labeled_test)
    for column in FEATURE_COLUMNS:
        case_punctuation[column] = (
            case_punctuation[column]
            .fillna("")
            .astype(str)
            .str.upper()
            .map(lambda value: re.sub(r"[^A-Z0-9\s]", " ", value))
        )
    cases["case_and_punctuation"] = case_punctuation

    oov_only = _copy_features(labeled_test)
    oov_only["summary"] = "novelsummarytokenzx"
    oov_only["text"] = "novelbodytokenqy driftwordqz"
    cases["oov_only_user_content"] = oov_only

    truncated = _copy_features(labeled_test)
    truncated["text"] = truncated["text"].fillna("").astype(str).map(
        lambda value: " ".join(value.split()[:8])
    )
    cases["body_truncated_8_tokens"] = truncated

    dropout = _copy_features(labeled_test)
    dropout["text"] = dropout["text"].fillna("").astype(str).map(
        lambda value: " ".join(
            token for index, token in enumerate(value.split()) if (index + 1) % 4 != 0
        )
    )
    cases["token_dropout_25pct"] = dropout

    replacement = _copy_features(labeled_test)
    replacement["text"] = replacement["text"].fillna("").astype(str).map(
        lambda value: " ".join(
            "novelreplacementtokenzx" if (index + 1) % 4 == 0 else token
            for index, token in enumerate(value.split())
        )
    )
    cases["token_replacement_25pct"] = replacement
    return cases


def prior_shift_slice(labeled_test: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Increase attention prevalence by deterministic negative-class subsampling."""

    positive = labeled_test.loc[labeled_test[TARGET_COLUMN] == 1]
    negative = labeled_test.loc[labeled_test[TARGET_COLUMN] == 0]
    desired_negative = max(1, min(len(negative), len(positive) * 2))
    sampled_negative = negative.sample(n=desired_negative, random_state=seed, replace=False)
    shifted = pd.concat([positive, sampled_negative], ignore_index=True)
    return shifted.sample(frac=1.0, random_state=seed).reset_index(drop=True)
