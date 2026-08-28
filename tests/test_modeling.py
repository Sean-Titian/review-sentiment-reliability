from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from review_reliability.data import (
    TARGET_COLUMN,
    SyntheticConfig,
    derive_rating_proxy,
    generate_synthetic_reviews,
)
from review_reliability.modeling import (
    FEATURE_COLUMNS,
    ReviewTextAssembler,
    fit_text_pipeline,
    predict_attention_probability,
    validate_serving_frame,
)


def _fitted_model():
    raw = generate_synthetic_reviews(SyntheticConfig(n_rows=320, seed=19))
    labeled = derive_rating_proxy(raw)
    features = labeled.loc[:, list(FEATURE_COLUMNS)]
    return fit_text_pipeline(features, labeled[TARGET_COLUMN], seed=19), features


def test_single_pipeline_handles_blank_none_and_unknown_tokens() -> None:
    model, _ = _fitted_model()
    serving = pd.DataFrame(
        {
            "summary": [None, "Entirely unseen heading"],
            "text": ["", "noveltokenzx driftwordqy"],
        }
    )
    scores = predict_attention_probability(model, serving)
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()
    assert list(model.named_steps) == ["assemble_text", "tfidf", "classifier"]


def test_serving_contract_rejects_outcome_identifiers_and_missing_fields() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        validate_serving_frame(pd.DataFrame({"summary": ["x"], "text": ["y"], "rating": [1]}))
    with pytest.raises(ValueError, match="Missing"):
        validate_serving_frame(pd.DataFrame({"text": ["y"]}))
    with pytest.raises(ValueError, match="Unexpected"):
        validate_serving_frame(pd.DataFrame({"summary": ["x"], "text": ["y"], "locale": ["z"]}))


def test_column_order_batch_row_and_serialization_parity(tmp_path) -> None:
    model, features = _fitted_model()
    batch = features.iloc[:8].copy()
    expected = predict_attention_probability(model, batch)
    reordered = predict_attention_probability(model, batch.loc[:, ["text", "summary"]])
    rowwise = np.concatenate(
        [predict_attention_probability(model, batch.iloc[[index]]) for index in range(len(batch))]
    )
    bundle_path = tmp_path / "pipeline.joblib"
    joblib.dump(model, bundle_path)
    restored = joblib.load(bundle_path)
    restored_scores = predict_attention_probability(restored, batch)
    np.testing.assert_allclose(expected, reordered, rtol=0, atol=1e-12)
    np.testing.assert_allclose(expected, rowwise, rtol=0, atol=1e-12)
    np.testing.assert_allclose(expected, restored_scores, rtol=0, atol=1e-12)


def test_assembler_requires_dataframe() -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        ReviewTextAssembler().fit([["summary", "body"]])
