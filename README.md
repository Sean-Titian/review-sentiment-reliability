# Review Sentiment Reliability

[![CI](https://github.com/Sean-Titian/review-sentiment-reliability/actions/workflows/ci.yml/badge.svg)](https://github.com/Sean-Titian/review-sentiment-reliability/actions/workflows/ci.yml)
[![Python 3.11--3.13](https://img.shields.io/badge/python-3.11--3.13-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/code-MIT-green.svg)](LICENSE)

A clean-room evaluation harness for a rating-proxy review-text classifier. It
shows how a promising row-random score can be challenged with duplicate-aware,
entity-grouped, rolling forward-time, calibration, fixed-budget, negative-control,
conditional uncertainty, and serving-stress checks before anyone makes a deployment
claim.

The public repository is intentionally **synthetic-only**. It contains no real
review, user or product identifier, source CSV, course material, source-trained
model, or real-data performance claim.

## What this milestone establishes

Version `0.3.0` extends the public artifact to evaluation contract `3.0` without
adding source data or a larger model:

- Four pre-specified rolling origins use expanding history, whole-timestamp
  boundaries, validation-only threshold selection, and non-overlapping future test
  horizons. Future rows are excluded at each origin.
- Exact/near-text, user, and product recurrence across every time boundary is
  measured rather than hidden. Temporal separation is not called leakage-free.
- Four window-specific 20-draw aggregate-mean gates and one pooled 80-draw
  aggregate-mean gate require rolling-window test-label placebos to return to
  chance before the canonical report is written.
- A frozen canonical fingerprint-group scoring rule receives 2,000 one-way pairs
  cluster-bootstrap draws by near-text fingerprint. Its marginal 95% percentile
  intervals are conditional diagnostics, not real-data confidence guarantees.
- Top-10%-budget metrics now fractionally allocate a cutoff tie. Equal scores can no
  longer inherit an accidental row, timestamp, or identifier ordering.
- Ignore rules and report validation reject additional raw-data, model, secret,
  cluster-key, and resample-index formats.

The underlying contract still fixes label-free manifests before partition-local
3-star exclusion, allows only `summary` and `text` at inference, uses one fitted
TF-IDF/logistic-regression pipeline, and compares five complementary split views.
It reports attention-class AP with prevalence, Brier/calibration, fixed-budget
precision/recall/lift, simple baselines, two multi-draw strict-split null designs,
and serving stress without promoting ROC-AUC to the headline.

## Decision contract

| Item | Public contract |
|---|---|
| Analysis unit | One unique review submission |
| Prediction time | When an unscored summary and body are submitted |
| Target | `needs_attention=1` for historical ratings 1--2; `0` for 4--5 |
| Neutral policy | Rating 3 is excluded within each assigned partition |
| Intended decision | Hypothetically rank a fixed-capacity manual review queue |
| Model inputs | Summary and body text only |
| Oracle warning | The source rating defines the target; if it is already visible, the model is redundant |
| Evidence status | Operational need and production value have not been validated |
| Claim boundary | Predictive reliability study; no causal or production-value claim |

The target is a **rating-derived weak proxy**, not a human sentiment label. In a
channel where rating is already visible, this model may be redundant. Transfer
to unscored feedback would require separate validation.

## Synthetic benchmark snapshot

The tracked report was generated with 2,400 authored synthetic rows. Its
generator deliberately plants drift, mixed sentiment, missing text, repeated
entities, duplicate content, and conflicting rating/text signals. About 44.6%
of rows belong to a repeated normalized-text group and 16.2% of labeled rows
belong to a fingerprint with conflicting proxy labels. This is an adversarial
test fixture, not a claim about any commercial review dataset.

| Protocol | Attention prevalence | Average precision (AP) | Brier | Recall @ 10% budget | Lift @ 10% budget |
|---|---:|---:|---:|---:|---:|
| Row-random (naive diagnostic) | 0.269 | 0.443 | 0.179 | 0.200 | 2.00x |
| Fingerprint-group | 0.242 | 0.448 | 0.162 | 0.218 | 2.14x |
| User-group | 0.264 | 0.452 | 0.176 | 0.197 | 1.94x |
| Product-group | 0.242 | 0.462 | 0.160 | 0.221 | 2.20x |
| Forward-time | 0.309 | 0.515 | 0.196 | 0.196 | 1.94x |

AP values are not directly comparable without prevalence because prevalence is
the no-skill AP reference. Here the stricter views do not uniformly lower the
score; that non-monotonic result is kept rather than forcing the expected story.
Across three matched seeds, fingerprint grouping changed AP versus row-random by
`+0.006` on average, ranging from `-0.056` to `+0.078`. These are descriptive
sensitivity results on one synthetic fixture, not a superiority claim.

On the fingerprint-group protocol, the training-prevalence probability baseline
had Brier 0.184 versus 0.162 for the model. Across 15 train-label permutations,
mean AP/prevalence was 1.117, lift@10% was 1.127, and ROC-AUC was 0.500. Across
15 test-label-alignment placebo draws, the corresponding means were
1.009, 0.964, and 0.496. Every draw and both aggregate means passed fixed,
explicit fail-closed chance gates before the report was written.

### Rolling-origin time sensitivity

Each row below is a different chronological test horizon. Training history expands;
the validation and test spans remain adjacent 10% slices, and the four test spans do
not overlap.

| Window | Labeled test rows | Prevalence | AP | AP − prevalence | Brier | Recall @ 10% | Lift @ 10% |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 222 | 0.239 | 0.457 | 0.218 | 0.160 | 0.226 | 2.19x |
| 2 | 222 | 0.239 | 0.523 | 0.284 | 0.157 | 0.283 | 2.73x |
| 3 | 222 | 0.329 | 0.501 | 0.173 | 0.207 | 0.151 | 1.45x |
| 4 | 225 | 0.289 | 0.506 | 0.217 | 0.182 | 0.215 | 2.11x |

Chronology does not create cold-start isolation. Depending on the window,
19.6%–37.1% of raw test rows reuse a near-text fingerprint seen in earlier history,
97.1%–98.8% reuse a user group, and 100% reuse a product group. Those exposures are
reported per window. Across 80 test-label-alignment placebos, mean ROC-AUC was
0.503, AP/prevalence 1.064, and lift@10% 0.972; all four window-specific 20-draw
aggregate-mean gates and the pooled 80-draw aggregate-mean gate passed.

### Conditional strict-split uncertainty

The interval below freezes the first fingerprint-group manifest, fitted model, and
authored synthetic test fixture after partition-local 3-star exclusion. It resamples
302 near-text fingerprint clusters 2,000 times; the effective cluster count is 210.0,
the largest cluster is 1.9% of test rows, and no draw is replenished.

| Metric | Fixed-test estimate | Conditional 95% lower | Conditional 95% upper |
|---|---:|---:|---:|
| Average precision | 0.395 | 0.317 | 0.505 |
| AP − prevalence | 0.147 | 0.093 | 0.239 |
| Brier | 0.169 | 0.149 | 0.188 |
| Brier improvement vs train-prevalence probability | 0.017 | 0.004 | 0.032 |
| Recall @ 10% budget | 0.175 | 0.097 | 0.231 |
| Lift @ 10% budget | 1.73x | 0.96x | 2.28x |

These are marginal percentile intervals conditional on one frozen synthetic scoring
rule—not uncertainty over retraining, threshold selection, crossed user/product
dependence, source selection, future drift, or real-data generalization. The AP
estimate is therefore the canonical seed-1103 point, not the three-seed mean in the
protocol table. The lift@10% interval of `[0.96x, 2.28x]` crosses the chance value
of `1.00x`; it therefore does not establish stable queue benefit and cannot support
a deployment-benefit claim.

The OOV-only stress case was also made structural rather than cosmetic: 100% of
rows produced zero TF-IDF vectors, AP fell exactly to prevalence (0.242), and
the mean absolute probability change was 0.134. Because all resulting scores
tied, arbitrary row-order recall/lift values are omitted. Replacing every fourth
body token (25% where length permits) produced 28.6% observed OOV tokens and a
mean probability change of 0.021. These numbers diagnose this authored fixture
only. Exact gates, ranges, and aggregate evidence are in
[`reports/synthetic-benchmark.json`](reports/synthetic-benchmark.json).

The multi-seed and rolling-origin min/max ranges describe sensitivity on one
synthetic fixture. They are **not** confidence intervals and are kept separate from
the conditional cluster-bootstrap result.

## Reproduce

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .

# Fast synthetic smoke run; writes an ignored demo artifact.
python -m review_reliability demo

# Canonical report: five protocols, four rolling origins, 2,000 conditional
# cluster-bootstrap draws, and strict-split plus temporal label placebos.
python -m review_reliability benchmark
```

Run the release gates:

```bash
ruff check --no-cache src tests
pytest
```

CI runs offline synthetic tests on Python 3.11, 3.12, and 3.13. No test or
workflow downloads review data.

## Evaluation flow

```text
validate label-free split fields
  -> build immutable train / validation / test manifest
  -> fail closed on protocol-specific cross-partition overlap
  -> derive rating proxy and exclude rating 3 inside each partition
  -> verify target-conflicting fingerprints were retained
     (the fingerprint protocol also co-locates them)
  -> fit text pipeline on train only
  -> choose decision threshold on validation only
  -> evaluate test against baselines, two strict-split null designs, and stress cases
  -> replay four whole-timestamp rolling origins with disjoint test horizons
  -> run temporal label placebos and a separate frozen strict-split cluster bootstrap
  -> validate and write aggregate-only JSON
```

Conflicting duplicate labels are retained and co-located for the
fingerprint-group protocol. They are audited after the split; they never decide
which rows are deleted or where a row is assigned.

## Repository map

```text
src/review_reliability/   data, splits, model, temporal, uncertainty, safety, CLI
tests/                    leakage, metric, parity, stress, and safety gates
reports/                  tracked aggregate-only synthetic benchmark
DATA_ACCESS.md            optional source access and rights caveats
DATA_POLICY.md            public/private artifact boundary
THIRD_PARTY_NOTICES.md    provenance, terms, and CC0 caveats
```

## Data and license boundary

The code, documentation, and authored synthetic fixture logic are MIT licensed.
That license does not cover third-party review data. See [DATA_ACCESS.md](DATA_ACCESS.md),
[DATA_POLICY.md](DATA_POLICY.md), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
before attempting any separate private real-data study.

## What this does not prove

- It does not report Amazon or other real-world model performance.
- It does not establish fairness, moderation safety, causal impact, or business
  value.
- Four synthetic rolling origins do not establish real-source temporal
  generalization; repeated text, users, and products still cross time boundaries.
- The cluster-bootstrap intervals condition on one frozen synthetic scoring rule
  and do not cover retraining, crossed dependence, or future drift.
- Token-set grouping is a transparent MVP approximation, not a scalable
  semantic near-duplicate system.
- A corrected split-first source audit exists privately, but data rights,
  target validity, and safe aggregation requirements still block any real-data
  metric.

## Next evidence, not extra buzzwords

1. Resolve source-data rights before considering any real-data aggregate; keep
   the private split-first adapter as audit evidence until then.
2. Validate an independently labeled, operationally useful target for feedback
   that does not already expose the source rating.
3. Add label-availability timestamps or an embargo, plus threshold/capacity
   sensitivity and an explicitly designed time-block uncertainty analysis.
4. Scale near-duplicate detection with MinHash/LSH and quantify false merges.
5. Evaluate calibration transfer and recall under realistic prior drift.
6. Add error slices only when they pass the documented small-cell and
   differencing checks.
