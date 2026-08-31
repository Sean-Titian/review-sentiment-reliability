"""Contract tests for label-free, split-first evaluation manifests."""

from __future__ import annotations

import pandas as pd
import pytest

from review_reliability.data import (
    SPLIT_KEY_COLUMNS,
    TARGET_COLUMN,
    SyntheticConfig,
    derive_rating_proxy,
    generate_synthetic_reviews,
    make_split_keys,
)
from review_reliability.splits import (
    PROTOCOLS,
    SplitSpec,
    attach_manifest,
    enforce_protocol_isolation,
    make_split_manifest,
    manifest_metadata,
    partition_overlap_audit,
)


@pytest.fixture(scope="module")
def synthetic_reviews() -> pd.DataFrame:
    """Return a deterministic fixture with duplicates, neutrals, and repeated groups."""

    return generate_synthetic_reviews(
        SyntheticConfig(
            n_rows=1_200,
            seed=20260828,
            neutral_rating_rate=0.18,
            duplicate_rate=0.30,
        )
    )


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_rating_and_target_changes_do_not_affect_manifest(
    synthetic_reviews: pd.DataFrame,
    protocol: str,
) -> None:
    original = synthetic_reviews.copy()
    altered = synthetic_reviews.copy()
    altered["rating"] = altered["rating"].map({1: 5, 2: 4, 3: 3, 4: 2, 5: 1})
    altered[TARGET_COLUMN] = altered["rating"].isin([1, 2]).astype("int8")

    original_keys = make_split_keys(original)
    altered_keys = make_split_keys(altered)
    pd.testing.assert_frame_equal(original_keys, altered_keys)

    spec = SplitSpec(protocol=protocol)
    original_manifest = make_split_manifest(original_keys, spec)
    altered_manifest = make_split_manifest(altered_keys, spec)
    pd.testing.assert_frame_equal(original_manifest, altered_manifest)


def test_split_input_contains_only_label_free_keys(synthetic_reviews: pd.DataFrame) -> None:
    split_keys = make_split_keys(synthetic_reviews)

    assert tuple(split_keys.columns) == SPLIT_KEY_COLUMNS
    assert {"rating", TARGET_COLUMN, "target", "label", "y"}.isdisjoint(split_keys.columns)

    contaminated = split_keys.assign(rating=5, needs_attention=0)
    with pytest.raises(ValueError, match="Target-bearing fields are forbidden"):
        make_split_manifest(contaminated, SplitSpec(protocol="row_random"))

    unexpected = split_keys.assign(unapproved_field=1)
    with pytest.raises(ValueError, match="Unexpected fields"):
        make_split_manifest(unexpected, SplitSpec(protocol="forward_time"))


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_manifest_is_independent_of_input_row_order(
    synthetic_reviews: pd.DataFrame,
    protocol: str,
) -> None:
    split_keys = make_split_keys(synthetic_reviews)
    shuffled_keys = split_keys.sample(frac=1.0, random_state=73).reset_index(drop=True)
    spec = SplitSpec(protocol=protocol)

    expected = make_split_manifest(split_keys, spec)
    actual = make_split_manifest(shuffled_keys, spec)

    pd.testing.assert_frame_equal(expected, actual)


@pytest.mark.parametrize(
    ("protocol", "isolated_columns"),
    (
        ("fingerprint_group", ("text_fingerprint", "near_text_fingerprint")),
        ("user_group", ("user_group",)),
        ("product_group", ("product_group",)),
    ),
)
def test_group_protocols_have_no_cross_partition_group_overlap(
    synthetic_reviews: pd.DataFrame,
    protocol: str,
    isolated_columns: tuple[str, ...],
) -> None:
    split_keys = make_split_keys(synthetic_reviews)
    manifest = make_split_manifest(split_keys, SplitSpec(protocol=protocol))
    joined = split_keys.merge(manifest, on="review_id", validate="one_to_one")
    audit = partition_overlap_audit(split_keys, manifest)

    for column in isolated_columns:
        assert joined.groupby(column, dropna=False)["partition"].nunique().max() == 1
        assert audit[column]["shared_groups"] == 0
        assert audit[column]["affected_rows"] == 0


def test_forward_time_partitions_are_strictly_ordered(
    synthetic_reviews: pd.DataFrame,
) -> None:
    split_keys = make_split_keys(synthetic_reviews)
    spec = SplitSpec(protocol="forward_time")
    manifest = make_split_manifest(split_keys, spec)
    joined = split_keys.merge(manifest, on="review_id", validate="one_to_one")

    bounds = {
        partition: pd.to_datetime(
            joined.loc[joined["partition"] == partition, "event_time"], utc=True
        )
        for partition in ("train", "validation", "test")
    }
    assert bounds["train"].max() < bounds["validation"].min()
    assert bounds["validation"].max() < bounds["test"].min()

    metadata = manifest_metadata(split_keys, manifest, spec)
    assert metadata["time_bounds"]["train"]["max"] == bounds["train"].max().isoformat()
    assert (
        metadata["time_bounds"]["validation"]["min"]
        == bounds["validation"].min().isoformat()
    )
    assert metadata["time_bounds"]["test"]["min"] == bounds["test"].min().isoformat()


def test_forward_time_keeps_equal_timestamps_in_one_partition(
    synthetic_reviews: pd.DataFrame,
) -> None:
    split_keys = make_split_keys(synthetic_reviews)
    split_keys["event_time"] = split_keys["event_time"].dt.floor("D")
    manifest = make_split_manifest(split_keys, SplitSpec(protocol="forward_time"))
    joined = split_keys.merge(manifest, on="review_id", validate="one_to_one")

    assert joined.groupby("event_time")["partition"].nunique().max() == 1
    bounds = joined.groupby("partition")["event_time"].agg(["min", "max"])
    assert bounds.loc["train", "max"] < bounds.loc["validation", "min"]
    assert bounds.loc["validation", "max"] < bounds.loc["test", "min"]


def test_forward_time_requires_three_unique_timestamps(
    synthetic_reviews: pd.DataFrame,
) -> None:
    split_keys = make_split_keys(synthetic_reviews.head(12))
    split_keys["event_time"] = [
        pd.Timestamp("2026-01-01", tz="UTC") if index % 2 else pd.Timestamp(
            "2026-01-02", tz="UTC"
        )
        for index in range(len(split_keys))
    ]
    with pytest.raises(ValueError, match="at least three unique timestamps"):
        make_split_manifest(split_keys, SplitSpec(protocol="forward_time"))


@pytest.mark.parametrize(
    ("protocol", "group_column"),
    (
        ("fingerprint_group", "near_text_fingerprint"),
        ("user_group", "user_group"),
        ("product_group", "product_group"),
    ),
)
def test_protocol_isolation_fails_closed_on_contaminated_manifest(
    synthetic_reviews: pd.DataFrame,
    protocol: str,
    group_column: str,
) -> None:
    split_keys = make_split_keys(synthetic_reviews)
    manifest = make_split_manifest(split_keys, SplitSpec(protocol=protocol))
    repeated_group = (
        split_keys[group_column].value_counts().loc[lambda value: value >= 2].index[0]
    )
    repeated_ids = split_keys.loc[
        split_keys[group_column] == repeated_group,
        "review_id",
    ]
    contaminated = manifest.copy()
    first_id = repeated_ids.iloc[0]
    current = contaminated.loc[contaminated["review_id"] == first_id, "partition"].iloc[0]
    replacement = "test" if current != "test" else "train"
    contaminated.loc[contaminated["review_id"] == first_id, "partition"] = replacement

    with pytest.raises(RuntimeError, match=f"{protocol} failed isolation"):
        enforce_protocol_isolation(split_keys, contaminated, protocol)

    row_manifest = make_split_manifest(split_keys, SplitSpec(protocol="row_random"))
    enforce_protocol_isolation(split_keys, row_manifest, "row_random")


def test_forward_time_isolation_gate_rejects_timestamp_contamination(
    synthetic_reviews: pd.DataFrame,
) -> None:
    split_keys = make_split_keys(synthetic_reviews)
    split_keys["event_time"] = split_keys["event_time"].dt.floor("D")
    manifest = make_split_manifest(split_keys, SplitSpec(protocol="forward_time"))
    enforce_protocol_isolation(split_keys, manifest, "forward_time")

    joined = split_keys.merge(manifest, on="review_id", validate="one_to_one")
    repeated_test_time = (
        joined.loc[joined["partition"] == "test", "event_time"]
        .value_counts()
        .loc[lambda value: value >= 2]
        .index[0]
    )
    contaminated_id = joined.loc[
        (joined["partition"] == "test")
        & (joined["event_time"] == repeated_test_time),
        "review_id",
    ].iloc[0]
    contaminated = manifest.copy()
    contaminated.loc[
        contaminated["review_id"] == contaminated_id,
        "partition",
    ] = "train"

    with pytest.raises(RuntimeError, match="timestamp crosses partitions"):
        enforce_protocol_isolation(split_keys, contaminated, "forward_time")


def test_protocol_isolation_rejects_incomplete_or_extra_manifest(
    synthetic_reviews: pd.DataFrame,
) -> None:
    split_keys = make_split_keys(synthetic_reviews)
    manifest = make_split_manifest(split_keys, SplitSpec(protocol="row_random"))

    with pytest.raises(RuntimeError, match="assign exactly"):
        enforce_protocol_isolation(split_keys, manifest.iloc[:-1], "row_random")
    with pytest.raises(RuntimeError, match="contain only"):
        enforce_protocol_isolation(
            split_keys,
            manifest.assign(unexpected=1),
            "row_random",
        )


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_neutral_ratings_are_assigned_before_partition_local_exclusion(
    synthetic_reviews: pd.DataFrame,
    protocol: str,
) -> None:
    split_keys = make_split_keys(synthetic_reviews)
    manifest = make_split_manifest(split_keys, SplitSpec(protocol=protocol))
    assigned = attach_manifest(synthetic_reviews, manifest)
    neutral_ids = set(assigned.loc[assigned["rating"] == 3, "review_id"])

    assert neutral_ids
    assert len(manifest) == len(synthetic_reviews)
    assert neutral_ids <= set(manifest["review_id"])

    labeled_partitions = []
    for partition in ("train", "validation", "test"):
        raw_partition = assigned.loc[assigned["partition"] == partition].copy()
        labeled_partition = derive_rating_proxy(raw_partition)
        assert len(labeled_partition) == int((raw_partition["rating"] != 3).sum())
        assert not labeled_partition["rating"].eq(3).any()
        assert labeled_partition["partition"].eq(partition).all()
        labeled_partitions.append(labeled_partition)

    labeled = pd.concat(labeled_partitions, ignore_index=True)
    assert len(labeled) == int(synthetic_reviews["rating"].ne(3).sum())
    assert neutral_ids.isdisjoint(labeled["review_id"])


def test_conflicting_labels_within_a_fingerprint_are_retained() -> None:
    reviews = generate_synthetic_reviews(
        SyntheticConfig(
            n_rows=600,
            seed=20260829,
            neutral_rating_rate=0.10,
            duplicate_rate=0.60,
        )
    )
    split_keys = make_split_keys(reviews)
    fingerprint_counts = split_keys["text_fingerprint"].value_counts()
    repeated_fingerprint = fingerprint_counts.loc[fingerprint_counts >= 2].index[0]
    conflicting_ids = split_keys.loc[
        split_keys["text_fingerprint"] == repeated_fingerprint, "review_id"
    ].iloc[:2]

    conflicting_reviews = reviews.copy()
    conflicting_reviews.loc[
        conflicting_reviews["review_id"] == conflicting_ids.iloc[0], "rating"
    ] = 1
    conflicting_reviews.loc[
        conflicting_reviews["review_id"] == conflicting_ids.iloc[1], "rating"
    ] = 5

    conflicting_keys = make_split_keys(conflicting_reviews)
    pd.testing.assert_frame_equal(split_keys, conflicting_keys)
    manifest = make_split_manifest(
        conflicting_keys,
        SplitSpec(protocol="fingerprint_group"),
    )
    assigned = attach_manifest(conflicting_reviews, manifest)
    selected = assigned.loc[assigned["review_id"].isin(conflicting_ids)]

    assert len(selected) == 2
    assert selected["partition"].nunique() == 1

    partition = selected["partition"].iloc[0]
    labeled_partition = derive_rating_proxy(
        assigned.loc[assigned["partition"] == partition].copy()
    )
    retained = labeled_partition.loc[labeled_partition["review_id"].isin(conflicting_ids)]

    assert len(retained) == 2
    assert set(retained[TARGET_COLUMN]) == {0, 1}
    assert len(labeled_partition) == int(
        assigned.loc[assigned["partition"] == partition, "rating"].ne(3).sum()
    )
