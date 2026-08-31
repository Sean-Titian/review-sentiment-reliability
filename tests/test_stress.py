from __future__ import annotations

import numpy as np

from review_reliability.data import (
    TARGET_COLUMN,
    SyntheticConfig,
    derive_rating_proxy,
    generate_synthetic_reviews,
)
from review_reliability.modeling import (
    FEATURE_COLUMNS,
    fit_text_pipeline,
    predict_attention_probability,
)
from review_reliability.stress import OOV_SENTINELS, text_stress_cases


def test_oov_stress_is_real_and_deterministic() -> None:
    raw = generate_synthetic_reviews(SyntheticConfig(n_rows=360, seed=20260831))
    labeled = derive_rating_proxy(raw)
    features = labeled.loc[:, list(FEATURE_COLUMNS)]
    model = fit_text_pipeline(features, labeled[TARGET_COLUMN], seed=20260831)
    cases = text_stress_cases(labeled, seed=20260831)

    vocabulary = model.named_steps["tfidf"].vocabulary_
    assert set(OOV_SENTINELS).isdisjoint(vocabulary)

    oov_only = cases["oov_only_user_content"]
    oov_scores = predict_attention_probability(model, oov_only)
    assert np.isfinite(oov_scores).all()
    np.testing.assert_allclose(oov_scores, np.repeat(oov_scores[0], len(oov_scores)))

    replacement = cases["token_replacement_25pct"]
    original_counts = features["text"].fillna("").astype(str).str.split().str.len()
    replacement_counts = replacement["text"].fillna("").astype(str).str.split().str.len()
    np.testing.assert_array_equal(original_counts.to_numpy(), replacement_counts.to_numpy())

    original_scores = predict_attention_probability(model, features)
    replacement_scores = predict_attention_probability(model, replacement)
    assert np.any(np.abs(original_scores - replacement_scores) > 1e-12)
