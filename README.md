# Review Sentiment Reliability

[![CI](https://github.com/Sean-Titian/review-sentiment-reliability/actions/workflows/ci.yml/badge.svg)](https://github.com/Sean-Titian/review-sentiment-reliability/actions/workflows/ci.yml)
[![Python 3.11--3.13](https://img.shields.io/badge/python-3.11--3.13-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/code-MIT-green.svg)](LICENSE)

A clean-room evaluation harness for a rating-proxy review-text classifier. It
shows how a promising row-random score can be challenged with duplicate-aware,
entity-grouped, forward-time, calibration, fixed-budget, negative-control, and
serving-stress checks before anyone makes a deployment claim.

The public repository is intentionally **synthetic-only**. It contains no real
review, user or product identifier, source CSV, course material, source-trained
model, or real-data performance claim.

## What this milestone establishes

Release `0.2.0` upgrades the public artifact to evaluation contract `2.0`:

- A decision contract at submission time: only `summary` and `text` are model
  features; rating and identifiers are forbidden at inference.
- A split-first contract: label-free manifests are fixed before 3-star rows are
  excluded and the rating proxy is derived inside each partition.
- One fitted scikit-learn pipeline owns text assembly, TF-IDF, and logistic
  regression for both training and inference.
- Five views form a **risk-sensitivity ladder**: row-random, normalized-text
  group, user group, product group, and whole-timestamp forward time. They
  isolate different failure modes; they are not a total ordering of difficulty.
- Protocol-specific overlap checks and a conflicting-fingerprint retention check
  fail closed before an aggregate report can be released.
- Attention-class average precision (AP), Brier score, calibration error,
  precision/recall, and top-10%-budget lift are reported with prevalence;
  ROC-AUC is supplementary.
- Train-prevalence, random-score, 15 train-label permutations, and 15
  test-label-alignment placebo draws keep leakage checks from resting on one lucky
  null draw.
- Missing fields, true out-of-vocabulary (OOV) text, 25% OOV token replacement,
  punctuation/case changes, truncation, token dropout, and prior shift are
  exercised without changing the serving schema.

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
| Fingerprint-group | 0.242 | 0.448 | 0.162 | 0.217 | 2.14x |
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
mean AP/prevalence was 1.117, lift@10% was 1.128, and ROC-AUC was 0.500. Across
15 test-label-alignment placebo draws, the corresponding means were
1.009, 0.964, and 0.496. Every draw and both aggregate means passed fixed,
explicit fail-closed chance gates before the report was written.

The OOV-only stress case was also made structural rather than cosmetic: 100% of
rows produced zero TF-IDF vectors, AP fell exactly to prevalence (0.242), and
the mean absolute probability change was 0.134. Because all resulting scores
tied, arbitrary row-order recall/lift values are omitted. Replacing every fourth
body token (25% where length permits) produced 28.6% observed OOV tokens and a
mean probability change of 0.021. These numbers diagnose this authored fixture
only. Exact gates, ranges, and aggregate evidence are in
[`reports/synthetic-benchmark.json`](reports/synthetic-benchmark.json).

These min/max ranges describe split or optimizer sensitivity on one synthetic
fixture. They are **not** sampling confidence intervals.

## Reproduce

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .

# Fast synthetic smoke run; writes an ignored demo artifact.
python -m review_reliability demo

# Canonical five-protocol report: three hash-split seeds, whole-time view,
# five permutations per fingerprint split, and matched label-alignment placebos.
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
  -> evaluate test once against baselines, two null designs, and stress cases
  -> validate and write aggregate-only JSON
```

Conflicting duplicate labels are retained and co-located for the
fingerprint-group protocol. They are audited after the split; they never decide
which rows are deleted or where a row is assigned.

## Repository map

```text
src/review_reliability/   data contract, splits, model, metrics, audits, CLI
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
- A single forward cutoff is not a full rolling temporal validation.
- Token-set grouping is a transparent MVP approximation, not a scalable
  semantic near-duplicate system.
- A corrected split-first source audit exists privately, but data rights,
  multi-seed/time sensitivity, and safe aggregation requirements still block
  release of any real-data metric.

## Next evidence, not extra buzzwords

1. Resolve source-data rights before considering any real-data aggregate; keep
   the private split-first adapter as audit evidence until then.
2. Add rolling forward windows, cluster/bootstrap uncertainty, and threshold
   sensitivity.
3. Scale near-duplicate detection with MinHash/LSH and quantify false merges.
4. Evaluate calibration transfer and recall under realistic prior drift.
5. Add error slices only when they pass the documented small-cell and
   differencing checks.
