"""Diversity diagnostics for ensemble members.

An ensemble only beats its best member when the members disagree in
useful ways — they must be wrong on *different* rows. This module measures
that, from scratch:

- **Pairwise disagreement** — share of rows where two members' labels differ.
- **Q-statistic** — ``(N11·N00 − N01·N10) / (N11·N00 + N01·N10)`` over the
  members' correctness table; lower (even negative) means more diverse.
- **Error correlation** — Pearson correlation between the members' 0/1
  error indicator vectors.
- **Ensemble-vs-best-single** — the headline comparison the promotion gate
  is built on.
"""
from __future__ import annotations

import math

from .models import DiversityReport


def _labels(probs: list[float]) -> list[int]:
    return [1 if p >= 0.5 else 0 for p in probs]


def pairwise_disagreement(a: list[int], b: list[int]) -> float:
    if not a:
        return 0.0
    return sum(1 for x, y in zip(a, b) if x != y) / len(a)


def q_statistic(a_correct: list[int], b_correct: list[int]) -> float:
    """Q over correctness indicators (1 = correct). Returns 0 on a
    degenerate table (a pair that never disagrees in correctness)."""
    n11 = sum(1 for x, y in zip(a_correct, b_correct) if x == 1 and y == 1)
    n00 = sum(1 for x, y in zip(a_correct, b_correct) if x == 0 and y == 0)
    n01 = sum(1 for x, y in zip(a_correct, b_correct) if x == 0 and y == 1)
    n10 = sum(1 for x, y in zip(a_correct, b_correct) if x == 1 and y == 0)
    denom = n11 * n00 + n01 * n10
    if denom == 0:
        return 0.0
    return (n11 * n00 - n01 * n10) / denom


def error_correlation(a_err: list[int], b_err: list[int]) -> float:
    """Pearson correlation between two 0/1 error indicator vectors."""
    n = len(a_err)
    if n == 0:
        return 0.0
    ma = sum(a_err) / n
    mb = sum(b_err) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a_err, b_err))
    da = math.sqrt(sum((x - ma) ** 2 for x in a_err))
    db = math.sqrt(sum((y - mb) ** 2 for y in b_err))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def diversity_report(
    member_probs: dict[str, list[float]],
    ensemble_prob: list[float],
    y_true: list[int],
) -> DiversityReport:
    """Full diversity picture for one ensemble against its members."""
    from .metrics import roc_auc

    names = sorted(member_probs)
    labels = {name: _labels(member_probs[name]) for name in names}
    correct = {
        name: [1 if pred == t else 0 for pred, t in zip(labels[name], y_true)]
        for name in names
    }
    errors = {
        name: [1 - c for c in correct[name]] for name in names
    }

    pair_dis: dict[str, float] = {}
    pair_q: dict[str, float] = {}
    pair_corr: list[float] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            key = f"{a} vs {b}"
            pair_dis[key] = pairwise_disagreement(labels[a], labels[b])
            pair_q[key] = q_statistic(correct[a], correct[b])
            pair_corr.append(error_correlation(errors[a], errors[b]))

    member_auc = {name: roc_auc(y_true, member_probs[name]) for name in names}
    best_name = max(member_auc, key=lambda k: member_auc[k])
    best_auc = member_auc[best_name]
    ens_auc = roc_auc(y_true, ensemble_prob)
    return DiversityReport(
        mean_pairwise_disagreement=(
            sum(pair_dis.values()) / len(pair_dis) if pair_dis else 0.0
        ),
        mean_q_statistic=sum(pair_q.values()) / len(pair_q) if pair_q else 0.0,
        mean_error_correlation=sum(pair_corr) / len(pair_corr) if pair_corr else 0.0,
        pairwise_disagreement=pair_dis,
        q_statistics=pair_q,
        best_single_name=best_name,
        best_single_auc=best_auc,
        ensemble_auc=ens_auc,
        beat_single_delta=ens_auc - best_auc,
    )
