"""Label-free, deterministic evaluation manifests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from review_reliability.data import SPLIT_KEY_COLUMNS

Protocol = Literal[
    "row_random",
    "fingerprint_group",
    "user_group",
    "product_group",
    "forward_time",
]

PROTOCOLS: tuple[Protocol, ...] = (
    "row_random",
    "fingerprint_group",
    "user_group",
    "product_group",
    "forward_time",
)


@dataclass(frozen=True)
class SplitSpec:
    """Three-way split configuration with no target-bearing fields."""

    protocol: Protocol
    seed: int = 20260828
    train_share: float = 0.60
    validation_share: float = 0.20
    test_share: float = 0.20

    def validate(self) -> None:
        if self.protocol not in PROTOCOLS:
            raise ValueError(f"Unknown protocol: {self.protocol}")
        shares = (self.train_share, self.validation_share, self.test_share)
        if any(share <= 0 for share in shares) or abs(sum(shares) - 1.0) > 1e-12:
            raise ValueError("train, validation, and test shares must be positive and sum to 1")


def _stable_unit_interval(value: object, seed: int) -> float:
    payload = f"{seed}|{value}".encode()
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / float(2**64)


def _hash_partition(value: object, spec: SplitSpec) -> str:
    score = _stable_unit_interval(value, spec.seed)
    if score < spec.train_share:
        return "train"
    if score < spec.train_share + spec.validation_share:
        return "validation"
    return "test"


def _validate_split_keys(split_keys: pd.DataFrame) -> None:
    missing = sorted(set(SPLIT_KEY_COLUMNS) - set(split_keys.columns))
    if missing:
        raise ValueError(f"Missing required label-free split keys: {missing}")
    forbidden = {"rating", "needs_attention", "target", "label", "y"} & set(
        split_keys.columns
    )
    if forbidden:
        raise ValueError(f"Target-bearing fields are forbidden in split input: {sorted(forbidden)}")
    if split_keys["review_id"].duplicated().any():
        raise ValueError("review_id must be unique in split keys")


def make_split_manifest(split_keys: pd.DataFrame, spec: SplitSpec) -> pd.DataFrame:
    """Assign rows before rating-proxy derivation.

    Hash protocols are stable to input order. The forward-time protocol uses
    deterministic time rank and strict, recorded boundaries.
    """

    spec.validate()
    _validate_split_keys(split_keys)
    keys = split_keys.loc[:, list(SPLIT_KEY_COLUMNS)].copy()

    if spec.protocol == "forward_time":
        ordered = keys.sort_values(["event_time", "review_id"], kind="mergesort")
        n_rows = len(ordered)
        train_end = max(1, int(n_rows * spec.train_share))
        validation_end = max(
            train_end + 1, int(n_rows * (spec.train_share + spec.validation_share))
        )
        validation_end = min(validation_end, n_rows - 1)
        partition = pd.Series("test", index=ordered.index, dtype="object")
        partition.iloc[:train_end] = "train"
        partition.iloc[train_end:validation_end] = "validation"
        assigned = pd.DataFrame(
            {"review_id": ordered["review_id"], "partition": partition},
            index=ordered.index,
        )
    else:
        group_column = {
            "row_random": "review_id",
            "fingerprint_group": "near_text_fingerprint",
            "user_group": "user_group",
            "product_group": "product_group",
        }[spec.protocol]
        assigned = pd.DataFrame(
            {
                "review_id": keys["review_id"],
                "partition": keys[group_column].map(lambda value: _hash_partition(value, spec)),
            }
        )

    manifest = assigned.sort_values("review_id", kind="mergesort").reset_index(drop=True)
    counts = manifest["partition"].value_counts()
    if set(counts.index) != {"train", "validation", "test"}:
        raise ValueError("Every split must produce train, validation, and test partitions")
    return manifest


def attach_manifest(frame: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Join a complete one-to-one manifest to the raw review rows."""

    if manifest["review_id"].duplicated().any():
        raise ValueError("Manifest review_id values must be unique")
    merged = frame.merge(manifest, on="review_id", how="left", validate="one_to_one")
    if merged["partition"].isna().any() or len(merged) != len(frame):
        raise ValueError("Manifest must assign every raw review exactly once")
    return merged


def manifest_metadata(split_keys: pd.DataFrame, manifest: pd.DataFrame, spec: SplitSpec) -> dict:
    """Return aggregate-only manifest metadata safe for a public report."""

    joined = split_keys.merge(manifest, on="review_id", validate="one_to_one")
    partition_counts = joined["partition"].value_counts().to_dict()
    metadata: dict[str, object] = {
        "protocol": spec.protocol,
        "seed": spec.seed,
        "raw_partition_rows": {key: int(partition_counts.get(key, 0)) for key in (
            "train",
            "validation",
            "test",
        )},
    }
    if spec.protocol == "forward_time":
        bounds = {}
        for partition in ("train", "validation", "test"):
            timestamps = pd.to_datetime(
                joined.loc[joined["partition"] == partition, "event_time"], utc=True
            )
            bounds[partition] = {
                "min": timestamps.min().isoformat(),
                "max": timestamps.max().isoformat(),
            }
        metadata["time_bounds"] = bounds
    return metadata


def partition_overlap_audit(split_keys: pd.DataFrame, manifest: pd.DataFrame) -> dict:
    """Count grouping values shared by two or more partitions."""

    joined = split_keys.merge(manifest, on="review_id", validate="one_to_one")
    output: dict[str, dict[str, int | float]] = {}
    for column in (
        "user_group",
        "product_group",
        "text_fingerprint",
        "near_text_fingerprint",
    ):
        partition_counts = joined.groupby(column, dropna=False)["partition"].nunique()
        shared_groups = int((partition_counts > 1).sum())
        affected_mask = joined[column].isin(partition_counts.index[partition_counts > 1])
        output[column] = {
            "shared_groups": shared_groups,
            "affected_rows": int(affected_mask.sum()),
            "affected_row_share": float(affected_mask.mean()),
        }
    return output
