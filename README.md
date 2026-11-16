# Model Ensemble Platform

**One model is a bet. An ensemble is a portfolio.** A platform to build,
evaluate, and serve model ensembles — voting, weighted blending, and
stacking — from scratch (no sklearn, no numpy, no torch), with diversity
diagnostics, gated promotion, latency-budgeted serving, and drift
monitoring. All learners, metrics, and statistics are implemented in
pure stdlib Python so every number in the system is auditable in source.

## The enterprise problem

A single model is rarely the best answer in production:

- **No learner wins everywhere.** A linear model nails the linear regime
  and fails the interaction regime; a tree does the opposite. Shipping
  one of them means accepting its blind spot.
- **Ensembles ship on vibes.** Teams average a few models, see a bump on
  one split, and promote. Nobody checks whether the members actually
  disagree, whether the gain survives a new seed, or whether the ensemble
  still fits the latency budget.
- **Serving is the weak link.** One slow member can blow the whole
  request budget, and a drifted input distribution silently rots the
  ensemble long before labelled outcomes arrive.

This platform makes each of those failure modes explicit and gated: an
ensemble is promoted only if it beats its best single member, its members
are genuinely diverse, it is stable across seeds, and it fits the latency
and size budgets. At serve time, slow members are dropped instead of
failing the request, and drift is monitored on inputs *and* on
ensemble–member agreement.

## What it demonstrates

- **From-scratch base learners** — logistic regression, CART decision
  tree, k-NN, Gaussian naive Bayes, plus a majority baseline — behind one
  shared `fit / predict / predict_proba` interface.
- **Three ensemble builders** — hard/soft voting, weighted blending with
  weights from validation AUC, and stacking with a from-scratch
  logistic-regression meta-learner trained on **leakage-safe out-of-fold**
  predictions.
- **Diversity diagnostics** — pairwise disagreement, Q-statistic, error
  correlation, and the ensemble-vs-best-single comparison the gate is
  built on.
- **Evaluation** — accuracy, macro F1, ROC-AUC, log-loss, Brier score,
  per-slice metrics (the planted linear vs nonlinear regimes), and
  calibration (reliability bins + ECE), all from scratch.
- **Gated registry** — versioned ensembles (`name@version`),
  DRAFT → CANDIDATE → ACTIVE with nine promotion gates and rollback.
- **Latency-budgeted serving** — per-member timeouts, graceful degradation
  (drop slow members, serve from the rest), stale-ensemble fallback when
  every member times out.
- **Canary / shadow deployment** — deterministic hash A/B routing, shadow
  scoring with agreement floors.
- **Drift monitoring** — PSI and two-sample KS per feature, plus
  ensemble–member agreement drift as an early warning before labels
  arrive.
- **Audit** — SHA-256 hash-chained log of every registry and drift
  decision, verifiable.

## Quickstart

```bash
make install   # create .venv and install dependencies
make test      # run the test suite
make demo      # full end-to-end demo (see below)
make serve     # FastAPI service on :8000
```

Or without make:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python scripts/demo.py
PYTHONPATH=src python -m uvicorn ensemble.service:app --port 8000
```

## The demo

`scripts/demo.py` runs the full pipeline on seeded synthetic data (a
planted mixture: half the rows follow a linear rule, half follow an
interaction + quadratic rule, so no single learner wins everywhere):

1. **Base learners trained and compared** — logistic, tree, k-NN, naive
   Bayes on the same held-out set.
2. **Three ensembles compared** — voting, blending, and stacking; stacking
   beats the best single member on ROC-AUC.
3. **Gates promote the winner** — the stacking ensemble clears all nine
   promotion gates (accuracy/AUC floors, beats-best-single delta,
   diversity floor, latency/size/member-count budgets, multi-seed
   stability, minimum eval cases) and becomes ACTIVE.
4. **Serving demo** — a deliberately slow k-NN member (300 ms) is dropped
   at a 100 ms per-member timeout; the request is still served from the
   other three members, marked degraded.
5. **Drift flagged** — a shifted input window (feature `f0` +2.0) trips
   PSI past the critical threshold; the stale-ensemble fallback is armed.

## API

| Endpoint | Description |
|---|---|
| `GET /health` | Status, ensemble count, ACTIVE key |
| `GET /ensembles` | List registered ensembles with metrics |
| `GET /ensembles/{key}` | Detail: metrics, diversity, gates |
| `POST /ensembles/{key}/promote` | Submit + gate + promote (422 if blocked) |
| `POST /ensembles/{key}/rollback` | Restore a RETIRED version to ACTIVE |
| `POST /predict` | Serve one prediction (ACTIVE by default) |
| `POST /drift` | Drift report for a serving window |
| `GET /audit/verify` | Verify the hash-chained audit log |
| `GET /demo` | Run the end-to-end demo, return headline numbers |

## CLI

```bash
ensemble-platform demo                                  # full demo
ensemble-platform evaluate [--seed N]                   # train + compare all three
ensemble-platform list                                  # registered ensembles
ensemble-platform predict --features "0.1,0.2,..."      # serve one row
```

## Promotion gates

A candidate is promoted only if, on the held-out set:

| Gate | Threshold |
|---|---|
| Accuracy floor | ≥ 0.78 |
| ROC-AUC floor | ≥ 0.82 |
| Beats best single (AUC delta) | ≥ 0.005 |
| Diversity floor (mean pairwise disagreement) | ≥ 0.05 |
| Latency budget (estimated p95) | ≤ 200 ms |
| Member count | ≤ 8 |
| Size budget | ≤ 2048 KB |
| Multi-seed stability (accuracy std) | ≤ 0.02 |
| Minimum evaluated cases | ≥ 100 |

Thresholds are configurable via `MEP_*` environment variables (see
`.env.example`).

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the component map,
data flow, and design decisions (leakage-safe OOF stacking, parallel
members with timeouts, agreement drift as an early warning, no
numpy/sklearn, in-memory state).

## Project layout

```
src/ensemble/
  config.py      Settings (env-overridable gates, budgets, drift)
  models.py      Domain dataclasses
  data.py        Seeded synthetic data (linear + nonlinear regimes)
  learners.py    Logistic / tree / k-NN / naive Bayes / majority (from scratch)
  ensembles.py   Voting / weighted blending / stacking (OOF meta-learner)
  metrics.py     Accuracy, macro F1, ROC-AUC, log-loss, Brier, calibration
  diversity.py   Pairwise disagreement, Q-statistic, error correlation
  evaluation.py  Train members, build + score all three ensembles
  registry.py    Versioned ensembles, promotion gates, rollback
  serving.py     Latency-budgeted inference, timeouts, degradation, fallback
  deployment.py  Canary (hash routing) + shadow with agreement scoring
  drift.py       PSI / KS on inputs and on ensemble–member agreement
  audit.py       SHA-256 hash-chained audit log
  platform.py    Orchestration: build, gate, promote, serve, monitor
  demo.py        End-to-end demo runner
  service.py     FastAPI service
  cli.py         Command-line interface
tests/           pytest suite (learners, ensembles, metrics, diversity,
                 evaluation, registry, serving, deployment, drift, audit,
                 CLI, API)
scripts/demo.py
docs/architecture.md  docs/github-actions-ci.yml
```

## Notes

- **Offline by design.** All data is seeded synthetic; no model API key,
  network access, or real dataset is needed.
- **Clean-room.** All code is original and all data is synthetic. No
  employer code, data, or proprietary material is included.
- **CI** ships as `docs/github-actions-ci.yml` — move it to
  `.github/workflows/ci.yml` to activate (the push integration used for this
  repo cannot carry workflow files, so it lives in `docs/`).
