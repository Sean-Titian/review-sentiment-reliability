"""Data contract and deterministic, public-safe synthetic review fixture."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

RAW_COLUMNS = (
    "review_id",
    "user_group",
    "product_group",
    "event_time",
    "summary",
    "text",
    "rating",
)
SPLIT_KEY_COLUMNS = (
    "review_id",
    "user_group",
    "product_group",
    "event_time",
    "text_fingerprint",
    "near_text_fingerprint",
)
TARGET_COLUMN = "needs_attention"


@dataclass(frozen=True)
class SyntheticConfig:
    """Configuration for the authored synthetic reliability fixture."""

    n_rows: int = 2400
    seed: int = 20260828
    attention_rate_early: float = 0.18
    attention_rate_late: float = 0.24
    neutral_rating_rate: float = 0.08
    rating_noise: float = 0.10
    text_noise: float = 0.22
    duplicate_rate: float = 0.18
    missing_text_rate: float = 0.025
    late_period_fraction: float = 0.30

    def validate(self) -> None:
        if self.n_rows < 100:
            raise ValueError("n_rows must be at least 100")
        probability_fields = (
            "attention_rate_early",
            "attention_rate_late",
            "neutral_rating_rate",
            "rating_noise",
            "text_noise",
            "duplicate_rate",
            "missing_text_rate",
            "late_period_fraction",
        )
        for name in probability_fields:
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_text(value: object) -> str:
    """Normalize review text without consulting a label or any other row."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return " ".join(_NON_ALNUM.sub(" ", str(value).lower()).split())


def text_fingerprint(value: object) -> str:
    """Return a stable normalized-text fingerprint suitable for grouping."""

    normalized = normalize_text(value)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def near_text_fingerprint(value: object) -> str:
    """Return a coarse token-set fingerprint for a conservative near-duplicate audit."""

    tokens = sorted(set(normalize_text(value).split()))
    signature = " ".join(tokens)
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:20]


def validate_raw_frame(frame: pd.DataFrame) -> None:
    """Fail fast when the raw review contract is incomplete or malformed."""

    missing = sorted(set(RAW_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required raw columns: {missing}")
    if frame.empty:
        raise ValueError("Review frame must not be empty")
    if frame["review_id"].isna().any() or frame["review_id"].duplicated().any():
        raise ValueError("review_id must be present and unique")
    ratings = pd.to_numeric(frame["rating"], errors="coerce")
    if ratings.isna().any() or not ratings.isin([1, 2, 3, 4, 5]).all():
        raise ValueError("rating must contain only integers from 1 through 5")
    timestamps = pd.to_datetime(frame["event_time"], errors="coerce", utc=True)
    if timestamps.isna().any():
        raise ValueError("event_time must be parseable")


def _validate_label_free_split_source(frame: pd.DataFrame) -> None:
    required = {
        "review_id",
        "user_group",
        "product_group",
        "event_time",
        "summary",
        "text",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required label-free split source columns: {missing}")
    if frame.empty:
        raise ValueError("Review frame must not be empty")
    if frame["review_id"].isna().any() or frame["review_id"].duplicated().any():
        raise ValueError("review_id must be present and unique")
    timestamps = pd.to_datetime(frame["event_time"], errors="coerce", utc=True)
    if timestamps.isna().any():
        raise ValueError("event_time must be parseable")


def make_split_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the label-free input accepted by split construction.

    Ratings and derived targets are deliberately absent. This prevents a global
    target-aware filter from influencing cohort assignment.
    """

    _validate_label_free_split_source(frame)
    keys = frame[["review_id", "user_group", "product_group", "event_time"]].copy()
    keys["event_time"] = pd.to_datetime(keys["event_time"], utc=True)
    canonical_input = (
        frame["summary"].fillna("").astype(str)
        + " "
        + frame["text"].fillna("").astype(str)
    )
    keys["text_fingerprint"] = canonical_input.map(text_fingerprint)
    keys["near_text_fingerprint"] = canonical_input.map(near_text_fingerprint)
    return keys.loc[:, list(SPLIT_KEY_COLUMNS)]


def derive_rating_proxy(partition: pd.DataFrame) -> pd.DataFrame:
    """Derive the weak rating proxy inside one already-assigned partition.

    Ratings 1--2 map to ``needs_attention=1`` and ratings 4--5 map to 0.
    Rating 3 is excluded. This is a behavioral proxy, not a human sentiment label.
    """

    validate_raw_frame(partition)
    labeled = partition.loc[partition["rating"] != 3].copy()
    labeled[TARGET_COLUMN] = labeled["rating"].isin([1, 2]).astype("int8")
    return labeled


def _choice(rng: np.random.Generator, values: tuple[str, ...]) -> str:
    return values[int(rng.integers(0, len(values)))]


def generate_synthetic_reviews(config: SyntheticConfig | None = None) -> pd.DataFrame:
    """Generate authored synthetic reviews; no source row or phrase is used."""

    config = config or SyntheticConfig()
    config.validate()
    rng = np.random.default_rng(config.seed)

    negative_early = (
        "arrived with a damaged seal",
        "stopped working sooner than expected",
        "did not match the description",
        "quality felt disappointing",
        "the package leaked in transit",
        "would not choose this again",
    )
    negative_late = (
        "the latest batch felt inconsistent",
        "the closure failed after one use",
        "support did not resolve the issue",
        "the texture changed unexpectedly",
        "the new packaging was difficult to use",
        "performance faded unusually quickly",
    )
    positive_early = (
        "worked as expected",
        "matched the description well",
        "quality was reliable",
        "arrived in good condition",
        "would choose this again",
        "was easy to use",
    )
    positive_late = (
        "the latest batch was consistent",
        "the closure worked on every use",
        "support resolved the question",
        "the texture remained consistent",
        "the new packaging was easy to use",
        "performance stayed reliable",
    )
    shared_openers = (
        "After a week of use",
        "For a routine order",
        "Compared with my expectation",
        "On the first try",
        "After following the directions",
        "For everyday use",
        "When the order arrived",
        "After a second attempt",
    )
    shared_context = (
        "delivery timing was ordinary",
        "the size was what I ordered",
        "the instructions were readable",
        "the outer box looked standard",
        "the price was within my range",
        "the item fit the intended use",
    )
    contrast_negative = (
        "but the main experience still needs attention",
        "although one part of the order was fine",
        "and the problem outweighed the acceptable details",
    )
    contrast_positive = (
        "despite one minor inconvenience",
        "and the small concern did not change the outcome",
        "although the delivery was not perfect",
    )
    summary_negative = (
        "Needs attention",
        "Inconsistent experience",
        "Below expectations",
        "A problem remained",
    )
    summary_positive = (
        "Reliable experience",
        "Met expectations",
        "Worked for my use",
        "Consistent result",
    )

    records: list[dict[str, object]] = []
    latent_states: list[bool] = []
    late_start = int(config.n_rows * (1.0 - config.late_period_fraction))
    start = pd.Timestamp("2024-01-01", tz="UTC")

    for index in range(config.n_rows):
        late_period = index >= late_start
        rate = config.attention_rate_late if late_period else config.attention_rate_early
        attention = bool(rng.random() < rate)

        duplicate_source: int | None = None
        if records and rng.random() < config.duplicate_rate:
            duplicate_source = int(rng.integers(0, len(records)))
            if rng.random() < 0.88:
                attention = latent_states[duplicate_source]

        text_state = attention
        if rng.random() < config.text_noise:
            text_state = not text_state

        if duplicate_source is None:
            negative_pool = negative_late if late_period else negative_early
            positive_pool = positive_late if late_period else positive_early
            core = _choice(rng, negative_pool if text_state else positive_pool)
            context = _choice(rng, shared_context)
            contrast_pool = contrast_negative if text_state else contrast_positive
            contrast = _choice(rng, contrast_pool) if rng.random() < 0.36 else ""
            pieces = [f"{_choice(rng, shared_openers)}, {core}", context, contrast]
            text = ". ".join(piece for piece in pieces if piece).strip() + "."
            summary_pool = summary_negative if text_state else summary_positive
            summary = _choice(rng, summary_pool)
        else:
            text = records[duplicate_source]["text"]
            summary = records[duplicate_source]["summary"]

        if rng.random() < config.missing_text_rate:
            text = None

        rating_state = attention
        if rng.random() < config.rating_noise:
            rating_state = not rating_state
        if rng.random() < config.neutral_rating_rate:
            rating = 3
        elif rating_state:
            rating = int(rng.choice([1, 2], p=[0.38, 0.62]))
        else:
            rating = int(rng.choice([4, 5], p=[0.55, 0.45]))

        records.append(
            {
                "review_id": f"synthetic_review_{index:06d}",
                "user_group": f"synthetic_user_{int(rng.integers(0, 420)):04d}",
                "product_group": f"synthetic_product_{int(rng.integers(0, 72)):03d}",
                "event_time": start + pd.Timedelta(hours=12 * index),
                "summary": summary,
                "text": text,
                "rating": rating,
            }
        )
        latent_states.append(attention)

    frame = pd.DataFrame.from_records(records, columns=RAW_COLUMNS)
    validate_raw_frame(frame)
    return frame
