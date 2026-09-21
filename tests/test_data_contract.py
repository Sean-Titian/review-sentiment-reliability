from __future__ import annotations

import pandas as pd
import pytest

from review_reliability.data import (
    RAW_COLUMNS,
    TARGET_COLUMN,
    SyntheticConfig,
    derive_rating_proxy,
    generate_synthetic_reviews,
    make_split_keys,
    near_text_fingerprint,
    normalize_text,
    text_fingerprint,
    validate_raw_frame,
)


def test_synthetic_fixture_is_deterministic_and_well_formed() -> None:
    config = SyntheticConfig(n_rows=240, seed=31)
    first = generate_synthetic_reviews(config)
    second = generate_synthetic_reviews(config)
    pd.testing.assert_frame_equal(first, second)
    assert tuple(first.columns) == RAW_COLUMNS
    assert first["review_id"].is_unique
    assert set(first["rating"]).issubset({1, 2, 3, 4, 5})
    assert first["text"].isna().any()


def test_synthetic_duplicate_propagation_preserves_missing_body_values() -> None:
    frame = generate_synthetic_reviews()
    present_bodies = frame["text"].dropna().astype(str).str.strip().str.casefold()

    assert not present_bodies.isin({"none", "nan", "nat", "<na>"}).any()
    assert int(frame["text"].isna().sum()) == 81


def test_rating_proxy_mapping_and_neutral_exclusion() -> None:
    frame = generate_synthetic_reviews(SyntheticConfig(n_rows=200, seed=7))
    labeled = derive_rating_proxy(frame)
    assert 3 not in set(labeled["rating"])
    assert (labeled.loc[labeled["rating"].isin([1, 2]), TARGET_COLUMN] == 1).all()
    assert (labeled.loc[labeled["rating"].isin([4, 5]), TARGET_COLUMN] == 0).all()


def test_split_keys_are_label_free() -> None:
    frame = generate_synthetic_reviews(SyntheticConfig(n_rows=120, seed=9))
    keys = make_split_keys(frame)
    assert "rating" not in keys
    assert TARGET_COLUMN not in keys
    assert set(keys) == {
        "review_id",
        "user_group",
        "product_group",
        "event_time",
        "text_fingerprint",
        "near_text_fingerprint",
    }


def test_text_fingerprints_ignore_case_spacing_and_punctuation() -> None:
    first = "Reliable item -- worked well!"
    second = " reliable ITEM worked   well "
    assert normalize_text(first) == normalize_text(second)
    assert text_fingerprint(first) == text_fingerprint(second)
    assert near_text_fingerprint("worked well reliable item") == near_text_fingerprint(second)


def test_raw_contract_rejects_duplicate_id_and_bad_rating() -> None:
    frame = generate_synthetic_reviews(SyntheticConfig(n_rows=120, seed=5))
    duplicate = frame.copy()
    duplicate.loc[1, "review_id"] = duplicate.loc[0, "review_id"]
    with pytest.raises(ValueError, match="unique"):
        validate_raw_frame(duplicate)
    bad_rating = frame.copy()
    bad_rating.loc[0, "rating"] = 9
    with pytest.raises(ValueError, match="rating"):
        validate_raw_frame(bad_rating)
