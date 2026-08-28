"""Single-pipeline text model with an explicit serving contract."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

FEATURE_COLUMNS = ("summary", "text")
FORBIDDEN_SERVING_COLUMNS = {
    "rating",
    "needs_attention",
    "review_id",
    "user_group",
    "product_group",
}


class ReviewTextAssembler(BaseEstimator, TransformerMixin):
    """Combine nullable summary and body fields inside the fitted pipeline."""

    def fit(self, X: pd.DataFrame, y: object = None) -> ReviewTextAssembler:  # noqa: N803
        self._validate(X)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        self._validate(X)
        summary = X["summary"].fillna("").astype(str).str.strip()
        text = X["text"].fillna("").astype(str).str.strip()
        combined = ("[summary] " + summary + " [body] " + text).str.strip()
        return combined.to_numpy(dtype=object)

    @staticmethod
    def _validate(X: pd.DataFrame) -> None:  # noqa: N803
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Review features must be provided as a pandas DataFrame")
        missing = sorted(set(FEATURE_COLUMNS) - set(X.columns))
        if missing:
            raise ValueError(f"Missing required review feature columns: {missing}")


def validate_serving_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept only fields available with a newly submitted, unscored review."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Serving input must be a pandas DataFrame")
    missing = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required serving fields: {missing}")
    forbidden = sorted(FORBIDDEN_SERVING_COLUMNS & set(frame.columns))
    if forbidden:
        raise ValueError(f"Post-outcome or identifier fields are forbidden at serving: {forbidden}")
    extras = sorted(set(frame.columns) - set(FEATURE_COLUMNS))
    if extras:
        raise ValueError(f"Unexpected serving fields: {extras}")
    return frame.loc[:, list(FEATURE_COLUMNS)]


def build_text_pipeline(seed: int = 20260828) -> Pipeline:
    """Build an interpretable TF-IDF logistic-regression baseline."""

    return Pipeline(
        steps=[
            ("assemble_text", ReviewTextAssembler()),
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.995,
                    max_features=12_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=seed,
                    solver="liblinear",
                ),
            ),
        ]
    )


def fit_text_pipeline(features: pd.DataFrame, target: pd.Series, seed: int) -> Pipeline:
    """Fit preprocessing and classifier together on the training partition only."""

    if target.nunique() != 2:
        raise ValueError("Training target must contain both proxy classes")
    model = build_text_pipeline(seed)
    model.fit(features.loc[:, list(FEATURE_COLUMNS)], target)
    return model


def predict_attention_probability(model: Any, frame: pd.DataFrame) -> np.ndarray:
    """Score a serving-shaped frame and return finite attention probabilities."""

    features = validate_serving_frame(frame)
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    classes = list(model.named_steps["classifier"].classes_)
    if 1 not in classes:
        raise ValueError("Fitted classifier does not contain the attention class")
    scores = probabilities[:, classes.index(1)]
    if len(scores) != len(frame) or not np.isfinite(scores).all():
        raise ValueError("Model produced invalid probabilities")
    if ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError("Model probabilities must lie in [0, 1]")
    return scores
