# Model Ensemble Platform — architecture

## Problem

A single model is rarely the best production answer, but ensembles are
usually shipped on vibes: average a few models, see a bump on one split,
promote. Nobody checks whether the members actually disagree, whether the
gain survives a new seed, whether the ensemble still fits the latency
budget — or what happens at serve time when one member goes slow. This
platform makes each of those failure modes explicit, measured, and gated.

## Components

```
                    ┌─────────────────────┐
                    │   FastAPI service   │  /predict /ensembles /drift
                    │  ensemble/service   │  /promote /rollback /audit
                    └────────┬────────────┘
                             │
                    ┌────────▼────────────┐
                    │      Platform       │  orchestration: build,
                    │  ensemble/platform  │  gate, promote, serve,
                    └────┬─────┬────┬────┘  monitor drift
          ┌──────────────┘     │    └──────────────┐
          ▼                    ▼                   ▼
   ┌─────────────┐   ┌───────────────┐   ┌────────────────┐
   │  Learners   │   │   Ensembles   │   │    Registry    │
   │ logistic /  │   │ voting /      │   │ versioned      │
   │ tree / kNN  │   │ blending /    │   │ ensembles,     │
   │ / naive     │   │ stacking      │   │ 9 gates,       │
   │ Bayes       │   │ (OOF meta)    │   │ rollback       │
   └─────────────┘   └───────────────┘   └────────────────┘
          │                    │                   │
   ┌──────▼──────┐   ┌─────────▼──────┐  ┌────────▼────────┐
   │ Metrics +   │   │   Serving +    │  │ Deployment +    │
   │ Diversity   │   │   Drift        │  │ Audit           │
   │ AUC/F1/ECE, │   │ timeouts,      │  │ hash canary,    │
   │ disagreement│   │ degradation,   │  │ shadow agree,   │
   │ Q, error    │   │ PSI/KS,        │  │ hash-chained    │
   │ correlation │   │ agreement drift│  │ log             │
   └─────────────┘   └────────────────┘  └─────────────────┘
```

### Data (`data.py`)
Seeded synthetic binary classification with a planted *mixture* of
regimes: half the rows follow a linear logit (features 0–2), half follow
an interaction (`x3 * x4 > 0`) plus a quadratic rule on `x5`, with linear
main effects on `x3`/`x4` so a greedy tree can find the structure. The
regime is identified by the last feature (the regime indicator), so the
ensemble can learn to trust different members in different regions. This
is what makes stacking beat the best single member: no one learner wins
both regimes. Labels follow the planted rule with 3% random flips.

### Learners (`learners.py`)
Logistic regression (batch gradient descent with standardisation),
CART decision tree (Gini, best split over sorted thresholds), k-NN
(`math.dist`, vote-share probabilities), Gaussian naive Bayes, and a
majority baseline. All share `fit / predict / predict_proba / clone`;
`clone()` is what makes out-of-fold stacking and multi-seed stability
checks possible without cross-contamination.

### Ensembles (`ensembles.py`)
- **Voting** — soft (mean probability) or hard (vote fraction, smoothed
  away from 0/1 so log-loss stays finite).
- **Weighted blending** — weights proportional to each member's validation
  AUC above chance (0.5), so a chance-level member gets ~zero weight.
  Weights come from a validation slice carved out of training data, never
  the test set.
- **Stacking** — a from-scratch logistic-regression meta-learner trained on
  out-of-fold base predictions plus regime interactions: each member's
  probability, interacted with the regime indicator (the last feature)
  and its complement, plus the indicator itself. That lets the
  meta-learner learn regime-dependent trust ("trust logistic in regime 0,
  the tree in regime 1") — a plain linear meta-learner over probabilities
  alone can only learn one fixed weighting. Base learners are then refit
  on the full training set for serving. `oof_auc()` gives an honest
  in-sample estimate from the same OOF predictions.

### Metrics (`metrics.py`) and diversity (`diversity.py`)
Accuracy, macro F1, ROC-AUC (Mann-Whitney U), log-loss, Brier, per-slice
metrics, and calibration (reliability bins + ECE) — all from scratch.
Diversity: pairwise disagreement, Q-statistic over the members'
correctness table, and Pearson correlation of the error indicators. The
headline number is `beat_single_delta`: ensemble AUC minus best-single
AUC. An ensemble that cannot beat its best member is not an ensemble.

### Registry (`registry.py`)
Ensembles are versioned (`name@version`) with lifecycle
DRAFT → CANDIDATE → ACTIVE → RETIRED. Promotion requires clearing all
nine gates (accuracy/AUC floors, beats-best-single delta, diversity
floor, latency/size/member-count budgets, multi-seed stability, minimum
eval cases). Promotion retires the previous ACTIVE version; `rollback`
restores it. A blocked candidate is marked BLOCKED with the failed gates
recorded.

### Serving (`serving.py`)
Members run in parallel threads with a per-member timeout and an overall
request budget. A member that misses its timeout is *dropped* — the
ensemble serves from the members that answered, and the result is marked
`degraded`. If every member times out, the server falls back to the
previous ACTIVE ensemble instead of failing the request. Latency p50/p95
are tracked in-memory for the latency-budget gate. `DelayedMember` is the
test/demo stand-in for a genuinely slow member.

### Deployment (`deployment.py`)
Deterministic hash routing: SHA-256 of the request id decides the bucket,
so the same request id always lands on the same arm. Canary sends a
configured fraction to the candidate; shadow scores the candidate in the
background and records production-vs-candidate agreement — a candidate
whose shadow agreement falls below the floor is not safe to promote.

### Drift (`drift.py`)
Two signals: **input drift** (PSI + two-sample KS per feature, reference
vs current window) and **agreement drift** (how often the ensemble label
matches the members' majority). Agreement drift is the early warning:
when the input distribution moves, members start disagreeing in new ways
before any labelled outcomes arrive. A `critical` report arms the
stale-ensemble fallback (audited); `warn` means watch it.

### Audit (`audit.py`)
SHA-256 hash-chained log of registry decisions and drift fallbacks;
`verify()` detects any tampering or deletion. The chain rule is
persistence-agnostic.

## Data flow

1. `build_all` generates seeded data, splits it, trains the four base
   learners, and builds voting / blending / stacking on the same split.
2. `evaluate` scores every member and every ensemble on the same held-out
   set: metrics, diversity, slices, calibration, latency/size estimates.
3. `promote_best` picks the highest-AUC ensemble, measures multi-seed
   stability, runs the nine gates, and promotes (or blocks with reasons).
4. `server_for` serves the ACTIVE ensemble with per-member timeouts and
   graceful degradation; `check_drift` watches the serving window.
5. Every registry decision and drift fallback is hash-chained into the
   audit log.

## Key design decisions and trade-offs

- **No numpy/sklearn.** Every learner, metric, and statistic is stdlib, so
  the math is auditable in the source and the core has zero heavy
  dependencies. The cost is speed: k-NN and the tree are slower than
  vectorised libraries, which is why the demo dataset is 1,200 rows.
- **Leakage-safe OOF stacking.** The meta-learner is trained only on
  out-of-fold predictions. Training it on in-sample base predictions
  would inflate its apparent skill and is the most common stacking bug.
- **Parallel members with timeouts.** Ensemble latency is the slowest
  member, not the sum — but only if members run concurrently and a slow
  member can be dropped. Serial execution would make the latency budget
  unmeetable.
- **Agreement drift before labels.** Labelled outcomes arrive late; PSI/KS
  and agreement drift are measurable on the serving window immediately.
- **In-memory state.** Registry, servers, and audit log live in memory.
  That keeps the demo and tests hermetic; a production deployment would
  persist versions and audit entries behind the same interfaces.
- **Synthetic data by design.** The planted mixture exists so the demo can
  *prove* stacking beats the best single member and the gates can be
  exercised deterministically — a biased or degenerate dataset would make
  the gates untestable.

## Running it

```bash
make install   # venv + deps
make test      # pytest
make demo      # build, compare, gate, promote, serve, drift
make serve     # uvicorn on :8000
```
