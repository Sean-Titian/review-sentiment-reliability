"""Label-free, deterministic evaluation manifests."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from numbers import Integral
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
PROTOCOL_ISOLATION_COLUMNS: dict[Protocol, tuple[str, ...]] = {
    "row_random": (),
    "fingerprint_group": ("text_fingerprint", "near_text_fingerprint"),
    "user_group": ("user_group",),
    "product_group": ("product_group",),
    "forward_time": (),
}

DEFAULT_LABEL_DELAY_DAYS: tuple[int, ...] = (0, 14, 30)


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


@dataclass(frozen=True)
class RollingOriginSpec:
    """Expanding-window backtest configuration over label-free time blocks."""

    first_train_share: float = 0.50
    validation_share: float = 0.10
    test_share: float = 0.10
    step_share: float = 0.10
    windows: int = 4

    def validate(self) -> None:
        shares = (
            self.first_train_share,
            self.validation_share,
            self.test_share,
            self.step_share,
        )
        if any(not math.isfinite(share) or share <= 0 for share in shares):
            raise ValueError("Rolling-origin shares must be positive and finite")
        if isinstance(self.windows, bool) or not isinstance(self.windows, int) or self.windows < 2:
            raise ValueError("Rolling-origin evaluation requires at least two windows")
        if self.step_share + 1e-12 < self.test_share:
            raise ValueError("Rolling-origin step share must prevent overlapping test windows")
        final_test_end = (
            self.first_train_share
            + self.validation_share
            + self.test_share
            + self.step_share * (self.windows - 1)
        )
        if final_test_end > 1.0 + 1e-12:
            raise ValueError("Rolling-origin windows extend beyond the available time range")


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
    extras = sorted(set(split_keys.columns) - set(SPLIT_KEY_COLUMNS))
    if extras:
        raise ValueError(f"Unexpected fields are forbidden in split input: {extras}")
    if split_keys["review_id"].isna().any() or split_keys["review_id"].duplicated().any():
        raise ValueError("review_id must be present and unique in split keys")
    timestamps = pd.to_datetime(split_keys["event_time"], errors="coerce", utc=True)
    if timestamps.isna().any():
        raise ValueError("event_time must be parseable in split keys")


def _whole_timestamp_boundaries(
    keys: pd.DataFrame,
    spec: SplitSpec,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    ordered = keys.sort_values(["event_time", "review_id"], kind="mergesort")
    unique_times = pd.Index(ordered["event_time"].drop_duplicates())
    if len(unique_times) < 3:
        raise ValueError("Forward-time split requires at least three unique timestamps")

    validation_candidate = ordered["event_time"].iloc[int(len(ordered) * spec.train_share)]
    test_candidate = ordered["event_time"].iloc[
        int(len(ordered) * (spec.train_share + spec.validation_share))
    ]
    validation_index = int(unique_times.searchsorted(validation_candidate, side="left"))
    test_index = int(unique_times.searchsorted(test_candidate, side="left"))
    validation_index = max(1, min(validation_index, len(unique_times) - 2))
    test_index = max(validation_index + 1, min(test_index, len(unique_times) - 1))
    return unique_times[validation_index], unique_times[test_index]


def make_split_manifest(split_keys: pd.DataFrame, spec: SplitSpec) -> pd.DataFrame:
    """Assign rows before rating-proxy derivation.

    Hash protocols are stable to input order. The forward-time protocol keeps
    equal timestamps together behind strict, recorded whole-timestamp boundaries.
    """

    spec.validate()
    _validate_split_keys(split_keys)
    keys = split_keys.loc[:, list(SPLIT_KEY_COLUMNS)].copy()
    keys["event_time"] = pd.to_datetime(keys["event_time"], utc=True)

    if spec.protocol == "forward_time":
        validation_start, test_start = _whole_timestamp_boundaries(keys, spec)
        timestamps = pd.to_datetime(keys["event_time"], utc=True)
        partition = pd.Series("test", index=keys.index, dtype="object")
        partition.loc[timestamps < test_start] = "validation"
        partition.loc[timestamps < validation_start] = "train"
        assigned = pd.DataFrame(
            {"review_id": keys["review_id"], "partition": partition},
            index=keys.index,
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


def make_rolling_origin_manifests(
    split_keys: pd.DataFrame,
    spec: RollingOriginSpec | None = None,
) -> tuple[pd.DataFrame, ...]:
    """Build expanding train, validation, and disjoint test windows.

    Boundaries are chosen from label-free row positions and then snapped to the
    beginning of a timestamp block. Rows after a window's test horizon are absent
    from that manifest, so future records cannot enter fitting or threshold choice.
    """

    spec = spec or RollingOriginSpec()
    spec.validate()
    _validate_split_keys(split_keys)
    keys = split_keys.loc[:, list(SPLIT_KEY_COLUMNS)].copy()
    keys["event_time"] = pd.to_datetime(keys["event_time"], utc=True)
    ordered = keys.sort_values(["event_time", "review_id"], kind="mergesort")
    unique_times = pd.Index(ordered["event_time"].drop_duplicates())
    if len(unique_times) < 3 * spec.windows:
        raise ValueError("Rolling-origin evaluation has too few unique timestamp blocks")

    def boundary_index(share: float) -> int:
        position = min(
            int(math.floor(len(ordered) * share + 1e-12)),
            len(ordered) - 1,
        )
        candidate = ordered["event_time"].iloc[position]
        return int(unique_times.searchsorted(candidate, side="left"))

    manifests: list[pd.DataFrame] = []
    prior_test_ids: set[object] = set()
    for window_index in range(spec.windows):
        train_end_share = spec.first_train_share + spec.step_share * window_index
        validation_end_share = train_end_share + spec.validation_share
        test_end_share = validation_end_share + spec.test_share
        train_end_index = boundary_index(train_end_share)
        validation_end_index = boundary_index(validation_end_share)
        test_end_index = (
            len(unique_times)
            if test_end_share >= 1.0 - 1e-12
            else boundary_index(test_end_share)
        )
        if not 0 < train_end_index < validation_end_index < test_end_index <= len(
            unique_times
        ):
            raise ValueError("Rolling-origin boundaries collapse timestamp partitions")

        train_end = unique_times[train_end_index]
        validation_end = unique_times[validation_end_index]
        test_end = unique_times[test_end_index] if test_end_index < len(unique_times) else None
        timestamps = keys["event_time"]
        eligible = pd.Series(True, index=keys.index)
        if test_end is not None:
            eligible &= timestamps < test_end
        partition = pd.Series("test", index=keys.index, dtype="object")
        partition.loc[timestamps < validation_end] = "validation"
        partition.loc[timestamps < train_end] = "train"
        manifest = pd.DataFrame(
            {
                "review_id": keys.loc[eligible, "review_id"],
                "partition": partition.loc[eligible],
            }
        ).sort_values("review_id", kind="mergesort", ignore_index=True)
        if set(manifest["partition"]) != {"train", "validation", "test"}:
            raise ValueError("Every rolling window must contain three non-empty partitions")
        test_ids = set(manifest.loc[manifest["partition"] == "test", "review_id"])
        if prior_test_ids & test_ids:
            raise RuntimeError("Rolling-origin test windows must not overlap")
        prior_test_ids |= test_ids
        manifests.append(manifest)

    return tuple(manifests)


def _validated_label_delays(delay_days: tuple[int, ...]) -> tuple[int, ...]:
    """Return a canonical, fail-closed label-delay sensitivity grid."""

    try:
        delays = tuple(delay_days)
    except TypeError as exc:
        raise ValueError("Label delays must be a finite sequence of integers") from exc
    if not delays or any(
        isinstance(delay, bool) or not isinstance(delay, Integral) or int(delay) < 0
        for delay in delays
    ):
        raise ValueError("Label delays must be non-negative finite integers")
    normalized = tuple(int(delay) for delay in delays)
    if 0 not in normalized:
        raise ValueError("Label delays must include the zero-day reference")
    if any(
        current >= following
        for current, following in zip(normalized[:-1], normalized[1:], strict=True)
    ):
        raise ValueError("Label delays must be strictly increasing with no duplicates")
    return normalized


def _recent_validation_start(
    timestamps: pd.Series,
    *,
    maturity_cutoff: pd.Timestamp,
    target_rows: int,
) -> pd.Timestamp:
    """Choose the closest-size recent whole-block validation history."""

    historical = timestamps.loc[timestamps < maturity_cutoff]
    block_counts = historical.value_counts(sort=False).sort_index()
    if len(block_counts) < 2:
        raise ValueError("Label delay leaves an empty train or validation partition")

    reverse_counts = block_counts.iloc[::-1].cumsum().iloc[::-1]
    candidates = [
        (abs(int(reverse_counts.iloc[index]) - target_rows), -index, index)
        for index in range(1, len(block_counts))
    ]
    _, _, start_index = min(candidates)
    return pd.Timestamp(block_counts.index[start_index])


def make_label_delay_rolling_manifests(
    split_keys: pd.DataFrame,
    spec: RollingOriginSpec | None = None,
    delay_days: tuple[int, ...] = DEFAULT_LABEL_DELAY_DAYS,
) -> dict[int, tuple[pd.DataFrame, ...]]:
    """Build full rolling manifests under pre-specified label-maturity delays.

    The test horizon is frozen to the zero-delay rolling-origin design. For a
    delay ``d``, labels at or after ``test_start - d days`` are embargoed. The
    most recent whole timestamp blocks before that cutoff become validation,
    with a raw-row count as close as possible to the zero-delay validation
    count; all earlier rows become train. Every returned manifest assigns every
    input row exactly once to train, validation, embargo, test, or future.

    Manifests deliberately contain only ``review_id`` and ``partition``. Safe
    window metadata is attached through ``DataFrame.attrs`` using the keys
    ``window``, ``label_delay_days``, ``test_start``, ``maturity_cutoff``,
    ``test_end``, ``target_validation_raw_rows``, and
    ``actual_validation_raw_rows``.
    """

    spec = spec or RollingOriginSpec()
    delays = _validated_label_delays(delay_days)
    base_manifests = make_rolling_origin_manifests(split_keys, spec)
    keys = split_keys.loc[:, list(SPLIT_KEY_COLUMNS)].copy()
    keys["event_time"] = pd.to_datetime(keys["event_time"], utc=True)
    timestamps = keys["event_time"]
    all_ids = set(keys["review_id"])
    output: dict[int, list[pd.DataFrame]] = {delay: [] for delay in delays}

    for window_index, base_manifest in enumerate(base_manifests, start=1):
        base_partition = base_manifest.set_index("review_id")["partition"]
        test_ids = set(
            base_manifest.loc[base_manifest["partition"] == "test", "review_id"]
        )
        test_times = timestamps.loc[keys["review_id"].isin(test_ids)]
        test_start = pd.Timestamp(test_times.min())
        target_validation_rows = int(base_manifest["partition"].eq("validation").sum())
        future_ids = all_ids - set(base_manifest["review_id"])
        future_times = timestamps.loc[keys["review_id"].isin(future_ids)]
        test_end = None if future_times.empty else pd.Timestamp(future_times.min())

        for delay in delays:
            try:
                maturity_cutoff = test_start - pd.Timedelta(days=delay)
            except (OverflowError, ValueError) as exc:
                raise ValueError(
                    "Label delay leaves an empty train or validation partition"
                ) from exc
            if delay == 0:
                partition = keys["review_id"].map(base_partition).fillna("future")
            else:
                validation_start = _recent_validation_start(
                    timestamps,
                    maturity_cutoff=maturity_cutoff,
                    target_rows=target_validation_rows,
                )
                partition = pd.Series("future", index=keys.index, dtype="object")
                partition.loc[timestamps < test_start] = "embargo"
                partition.loc[timestamps < maturity_cutoff] = "validation"
                partition.loc[timestamps < validation_start] = "train"
                partition.loc[keys["review_id"].isin(test_ids)] = "test"

            manifest = pd.DataFrame(
                {"review_id": keys["review_id"], "partition": partition},
                index=keys.index,
            ).sort_values("review_id", kind="mergesort", ignore_index=True)
            if len(manifest) != len(keys) or manifest["review_id"].duplicated().any():
                raise RuntimeError("Label-delay manifest must assign every review exactly once")
            if set(manifest["review_id"]) != all_ids:
                raise RuntimeError("Label-delay manifest review IDs do not match split keys")
            if not {"train", "validation"} <= set(manifest["partition"]):
                raise ValueError("Label delay leaves an empty train or validation partition")
            manifest.attrs = {
                "window": window_index,
                "label_delay_days": delay,
                "test_start": test_start.isoformat(),
                "maturity_cutoff": maturity_cutoff.isoformat(),
                "test_end": None if test_end is None else test_end.isoformat(),
                "target_validation_raw_rows": target_validation_rows,
                "actual_validation_raw_rows": int(
                    manifest["partition"].eq("validation").sum()
                ),
            }
            output[delay].append(manifest)

    return {delay: tuple(manifests) for delay, manifests in output.items()}


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


def enforce_protocol_isolation(
    split_keys: pd.DataFrame,
    manifest: pd.DataFrame,
    protocol: Protocol,
) -> dict[str, dict[str, int | float]]:
    """Return the overlap audit or fail when a grouped protocol leaks its unit."""

    if tuple(manifest.columns) != ("review_id", "partition"):
        raise RuntimeError("Split manifest must contain only review_id and partition")
    if manifest["review_id"].isna().any() or manifest["review_id"].duplicated().any():
        raise RuntimeError("Split manifest review_id values must be present and unique")
    if set(manifest["review_id"]) != set(split_keys["review_id"]):
        raise RuntimeError("Split manifest must assign exactly the split-key review IDs")
    if set(manifest["partition"]) != {"train", "validation", "test"}:
        raise RuntimeError("Split manifest must contain only three non-empty partitions")

    overlap_audit = partition_overlap_audit(split_keys, manifest)
    for column in PROTOCOL_ISOLATION_COLUMNS[protocol]:
        if column not in overlap_audit:
            raise RuntimeError(f"{protocol} isolation audit is missing {column}")
        values = overlap_audit[column]
        required = ("shared_groups", "affected_rows", "affected_row_share")
        if any(key not in values for key in required):
            raise RuntimeError(f"{protocol} isolation audit is incomplete for {column}")
        numeric = [float(values[key]) for key in required]
        if not all(math.isfinite(value) and value == 0.0 for value in numeric):
            raise RuntimeError(
                f"{protocol} failed isolation: {column} crosses partitions"
            )
    if protocol == "forward_time":
        joined = split_keys.loc[:, ["review_id", "event_time"]].merge(
            manifest,
            on="review_id",
            validate="one_to_one",
        )
        timestamps = pd.to_datetime(joined["event_time"], errors="coerce", utc=True)
        if timestamps.isna().any():
            raise RuntimeError("forward_time isolation requires parseable timestamps")
        joined = joined.assign(event_time=timestamps)
        if joined.groupby("event_time")["partition"].nunique().max() != 1:
            raise RuntimeError("forward_time failed isolation: one timestamp crosses partitions")
        bounds = joined.groupby("partition")["event_time"].agg(["min", "max"])
        if not (
            bounds.loc["train", "max"] < bounds.loc["validation", "min"]
            and bounds.loc["validation", "max"] < bounds.loc["test", "min"]
        ):
            raise RuntimeError("forward_time failed isolation: partitions are not ordered")
    return overlap_audit
