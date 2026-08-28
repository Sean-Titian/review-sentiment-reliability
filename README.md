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

- A decision contract at submission time: only `summary` and `text` are model
  features; rating and identifiers are forbidden at inference.
- A split-first contract: label-free manifests are fixed before 3-star rows are
  excluded and the rating proxy is derived inside each partition.
- One fitted scikit-learn pipeline owns text assembly, TF-IDF, and logistic
  regression for both training and inference.
- Five evaluation views expose different risks: row-random, normalized-text
  group, user group, product group, and forward time.
- Minority PR-AUC, Brier score, calibration error, precision/recall, and
  top-10%-budget lift are reported with prevalence; ROC-AUC is supplementary.
- Train-prevalence, random-score, and train-label-permutation controls prevent a
  large number from becoming an unsupported headline.
- Missing fields, unknown words, punctuation/case changes, truncation, token
  dropout, and prior shift are exercised without changing the serving schema.

## Decision contract

| Item | Public contract |
|---|---|
| Analysis unit | One unique review submission |
| Prediction time | When an unscored summary and body are submitted |
| Target | `needs_attention=1` for historical ratings 1--2; `0` for 4--5 |
| Neutral policy | Rating 3 is excluded within each assigned partition |
| Intended decision | Rank a fixed-capacity manual review queue |
| Model inputs | Summary and body text only |
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

| Protocol | Attention prevalence | PR-AUC | Brier | Recall @ 10% budget | Lift @ 10% budget |
|---|---:|---:|---:|---:|---:|
| Row-random (naive diagnostic) | 0.269 | 0.443 | 0.179 | 0.200 | 2.00x |
| Fingerprint-group | 0.242 | 0.448 | 0.162 | 0.217 | 2.14x |
| User-group | 0.264 | 0.452 | 0.176 | 0.197 | 1.94x |
| Product-group | 0.242 | 0.462 | 0.160 | 0.221 | 2.20x |
| Forward-time | 0.309 | 0.515 | 0.196 | 0.196 | 1.94x |

Raw PR-AUC values are not directly comparable without prevalence. Here the
strict views do not uniformly lower the score; that non-monotonic result is
kept rather than forcing the expected story. The protocol changes which risk is
tested, not the direction a metric must move.

On the fingerprint-group protocol, the training-prevalence probability baseline
had Brier 0.184 versus 0.162 for the model. The train-label-permutation control
returned mean PR-AUC 0.283 at prevalence 0.242 and ROC-AUC 0.511. Every
permutation run passed the fail-closed chance-range check before the report was
written. The exact aggregate values and descriptive multi-seed ranges are in
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

# Canonical five-protocol, multi-seed report with fail-closed permutation checks.
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
  -> derive rating proxy and exclude rating 3 inside each partition
  -> fit text pipeline on train only
  -> choose decision threshold on validation only
  -> evaluate test once against baselines, controls, and stress cases
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
- The current source benchmark remains private until its old target-aware
  global conflict filtering is replaced by a corrected split-first run.

## Next evidence, not extra buzzwords

1. Add a private, split-first adapter and publish only safe aggregate results if
   data rights permit.
2. Add rolling forward windows, cluster/bootstrap uncertainty, and threshold
   sensitivity.
3. Scale near-duplicate detection with MinHash/LSH and quantify false merges.
4. Evaluate calibration transfer and recall under realistic prior drift.
5. Add error slices only when they are large enough to remain anonymous.
