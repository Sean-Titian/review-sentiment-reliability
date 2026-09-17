# Review Sentiment Reliability

[![CI](https://github.com/Sean-Titian/review-sentiment-reliability/actions/workflows/ci.yml/badge.svg)](https://github.com/Sean-Titian/review-sentiment-reliability/actions/workflows/ci.yml)
[![Python 3.11--3.13](https://img.shields.io/badge/python-3.11--3.13-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/code-MIT-green.svg)](LICENSE)

A clean-room evaluation harness for a rating-proxy review-text classifier. It
shows how a promising row-random score can be challenged with duplicate-aware,
entity-grouped, rolling forward-time, label-availability, calibration, fixed-budget,
negative-control, conditional uncertainty, and serving-stress checks before anyone
makes a deployment claim.

The public repository is intentionally **synthetic-only**. It contains no real
review, user or product identifier, source CSV, course material, source-trained
model, or real-data performance claim.

## What this milestone establishes

Version `0.5.0` extends the public artifact to evaluation contract `5.0` without
adding source data or a larger model:

- Four fixed rolling test horizons are now replayed under pre-specified 0-, 14-,
  and 30-day proxy-label delay scenarios. Every scenario evaluates the same test
  submissions; no delay is selected after seeing the result.
- Each window assigns every raw row to `train`, `validation`, `embargo`, `test`,
  or `future`. Whole timestamp blocks remain intact, counts must conserve all
  2,400 synthetic rows, and embargoed rows cannot enter fitting or threshold
  selection.
- On this fixture, 14- and 30-day scenarios exclude 28 and 60 recent raw rows per
  window while keeping validation and test at 240 raw rows each. Test submissions
  stay identical; validation shifts earlier. The zero-day path reproduces the
  prior rolling-origin manifests and metrics exactly.
- Each delay has four 20-draw window-level test-label placebos; those 80 draws are
  also checked in one pooled gate. All 12 window gates and all three pooled gates pass their
  pre-specified heuristic bounds; these are sanity checks, not p-values.
- Paired metric changes are non-monotonic. Relative to zero delay, mean AP changes by
  +0.002 at 14 days and -0.009 at 30 days across four correlated windows, while
  mean lift-at-10% changes by -0.091 and -0.050. These paired ranges are
  descriptive, not confidence intervals.
- Label-availability timestamps were not observed: 14 and 30 days are authored
  sensitivity scenarios, not measured service levels, optimal embargoes, or
  evidence that the rating-derived proxy is operationally useful.

- The manual-review workload is no longer represented by one arbitrary point.
  Queue shares of 5%, 10%, and 20% are pre-specified and always reported together;
  no capacity is selected after seeing the result.
- The frozen canonical fingerprint-group scoring rule receives the same 2,000
  near-text-cluster resamples at all three capacities. The resulting marginal 95%
  percentile intervals are pointwise diagnostics, not a simultaneous band.
- Three matched strict-split seeds show descriptive capacity sensitivity separately
  from bootstrap uncertainty. These ranges are not confidence intervals.
- Both train-label permutations and test-label-alignment placebos must fall inside
  pre-specified fail-closed heuristic sanity bounds at every capacity. These bounds
  are not p-values or multiplicity-adjusted inference; the 5% bounds are wider to
  reflect the smaller synthetic queue.
- Queue counts use `ceil`, cutoff ties receive fractional expected allocation, and
  the 10% curve point must equal every legacy 10%-budget result exactly.
- The existing four rolling origins, duplicate/entity audits, serving stresses,
  calibration checks, public-safety checks, and source-data boundary remain in force.

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
| Intended decision | Hypothetically rank manual-review queues at pre-specified 5%, 10%, and 20% workload shares |
| Capacity caveat | Sensitivity scenarios only; no staffing cost, action value, or net benefit is modeled |
| Model inputs | Summary and body text only |
| Label-delay sensitivity | Pre-specified 0-, 14-, and 30-day authored scenarios with identical test horizons |
| Availability evidence | No label-availability timestamp was observed; delayed rows are an evaluation stress only |
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
mean AP/prevalence was 1.117 and ROC-AUC was 0.500; mean lifts at 5%/10%/20%
capacity were 1.262/1.127/1.076. Across 15 test-label-alignment placebo draws,
the corresponding AP/prevalence and ROC-AUC means were 1.009 and 0.496, while
capacity lifts were 0.893/0.964/0.958. Every draw and every aggregate capacity
mean passed the fixed heuristic sanity bounds before the report was written. These
bounds can stop a suspicious release, but are not a calibrated joint hypothesis test.

### Queue-capacity sensitivity

The table below uses the first strict fingerprint-group split and one frozen scoring
rule. Queue rows are the actual `ceil` workloads for its 416 labeled test rows. All
three capacities were specified before the canonical run and share the same 2,000
cluster-bootstrap resamples.

| Queue share | Queue rows | Precision | Recall | Lift | Conditional 95% lift interval | Lift range across 3 matched seeds |
|---:|---:|---:|---:|---:|---:|---:|
| 5% | 21 | 0.333 | 0.068 | 1.35x | [0.51x, 2.17x] | 1.35x–2.89x |
| 10% | 42 | 0.429 | 0.175 | 1.73x | [0.96x, 2.28x] | 1.73x–2.35x |
| 20% | 84 | 0.464 | 0.379 | 1.88x | [1.37x, 2.30x] | 1.88x–2.21x |

The 5% and 10% pointwise lift intervals cross the chance value of 1.00x. The 20%
interval does not, conditional on this one frozen synthetic scoring rule. That is
not a staffing recommendation or evidence of business value: the three intervals
are marginal rather than simultaneous (so this is not a family-wise 95% conclusion),
the target is a rating-derived oracle proxy, and review costs and downstream actions
are not modeled. The full report also
retains precision and recall intervals rather than selecting the largest lift.

### Rolling-origin and label-delay sensitivity

The zero-day reference below preserves the previous rolling-origin result. Each row
is a different chronological test horizon. Training history expands; the validation
and test spans remain adjacent 10% slices, and the four test spans do not overlap.

| Window | Labeled test rows | Prevalence | AP | AP − prevalence | Brier | Recall @ 10% | Lift @ 10% |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 222 | 0.239 | 0.457 | 0.218 | 0.160 | 0.226 | 2.19x |
| 2 | 222 | 0.239 | 0.523 | 0.284 | 0.157 | 0.283 | 2.73x |
| 3 | 222 | 0.329 | 0.501 | 0.173 | 0.207 | 0.151 | 1.45x |
| 4 | 225 | 0.289 | 0.506 | 0.217 | 0.182 | 0.215 | 2.11x |

The same four test horizons are then frozen while recent history is embargoed under
the 14- and 30-day authored delay scenarios. Deltas are delayed minus zero-day;
lower Brier is better.

| Delay scenario | Embargoed raw rows per window | Mean AP delta | AP delta range | Mean Brier delta | Mean recall@10% delta | Mean lift@10% delta | Lift@10% delta range |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 days | 0 | 0.000 | [0.000, 0.000] | 0.0000 | 0.000 | 0.000 | [0.000, 0.000] |
| 14 days | 28 | +0.002 | [-0.008, +0.020] | -0.0002 | -0.009 | -0.091 | [-0.182, 0.000] |
| 30 days | 60 | -0.009 | [-0.031, +0.002] | +0.0006 | -0.005 | -0.050 | [-0.182, +0.132] |

Each delay refits the model and reselects its threshold using only its eligible
train and validation rows. These four-window ranges are correlated descriptive
sensitivities, not confidence intervals or evidence that one delay is preferable.
The source provides no observed label-availability time; the source rating defines
the proxy and may already be visible at submission. This exercise tests as-of
evaluation plumbing for a hypothetical target whose label matures after prediction,
not an actual SLA.

Chronology does not create cold-start isolation. Depending on the window,
19.6%–37.1% of raw test rows reuse a near-text fingerprint seen in earlier history,
97.1%–98.8% reuse a user group, and 100% reuse a product group. Those exposures are
reported per window. Across 80 test-label-alignment placebos, mean ROC-AUC was
0.503, AP/prevalence 1.064, and lift@10% 0.972 in the zero-day reference. Across
all three delays, all 12 window-specific 20-draw gates passed; each delay's same
80 draws also passed one pooled gate under the fixed heuristic bounds.

### Conditional strict-split uncertainty

The intervals freeze the first fingerprint-group manifest, fitted model, and
authored synthetic test fixture after partition-local 3-star exclusion. They
resample 302 near-text fingerprint clusters 2,000 times; the effective cluster
count is 210.0, the largest cluster is 1.9% of test rows, and no draw is replenished.
The non-capacity diagnostics are:

| Metric | Fixed-test estimate | Conditional 95% lower | Conditional 95% upper |
|---|---:|---:|---:|
| Average precision | 0.395 | 0.317 | 0.505 |
| AP − prevalence | 0.147 | 0.093 | 0.239 |
| Brier | 0.169 | 0.149 | 0.188 |
| Brier improvement vs train-prevalence probability | 0.017 | 0.004 | 0.032 |

These are marginal percentile intervals conditional on one frozen synthetic scoring
rule—not uncertainty over retraining, threshold selection, crossed user/product
dependence, source selection, future drift, or real-data generalization. The AP
estimate is therefore the canonical seed-1103 point, not the three-seed mean in the
protocol table. Capacity intervals use the same resamples and are shown together in
the pre-specified capacity table above; none can support a deployment-benefit claim.

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

# Canonical report: five protocols, four rolling origins at authored 0/14/30-day
# delays, pre-specified 5/10/20% capacities, 2,000 joint bootstrap draws, and placebos.
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
  -> freeze those horizons across authored 0 / 14 / 30 day label-delay scenarios
  -> embargo unavailable history before fitting and threshold selection
  -> run per-delay temporal label placebos and three-capacity strict-split null gates
  -> use one frozen strict-split cluster bootstrap jointly at 5% / 10% / 20%
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
- The 5%, 10%, and 20% workloads are sensitivity scenarios, not validated staffing
  levels; pointwise intervals do not support choosing a capacity after the fact.
- The 14- and 30-day delays are authored sensitivity scenarios, not observed label
  latency, business SLAs, recommended embargoes, or validation of an operational target.
- Token-set grouping is a transparent MVP approximation, not a scalable
  semantic near-duplicate system.
- A corrected split-first source audit exists privately, but data rights,
  target validity, and safe aggregation requirements still block any real-data
  metric.

## Next evidence, not extra buzzwords

1. Resolve source-data rights before considering any real-data aggregate; keep
   the private split-first adapter as audit evidence until then.
2. Obtain an independently labeled, operationally useful target with an observed
   outcome window and `label_available_at`; the current fixed delays remain a
   synthetic plumbing test until then.
3. Add threshold sensitivity and an explicitly designed time-block uncertainty
   analysis once that target and its maturity process exist.
4. Scale near-duplicate detection with MinHash/LSH and quantify false merges.
5. Evaluate calibration transfer and recall under realistic prior drift.
6. Add error slices only when they pass the documented small-cell and
   differencing checks.
