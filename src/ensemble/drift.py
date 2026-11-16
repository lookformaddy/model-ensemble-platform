"""Drift monitoring: PSI / KS on inputs and on ensemble agreement.

Two signals are tracked:

- **Input drift** — Population Stability Index (PSI) and the two-sample
  Kolmogorov–Smirnov statistic per feature, reference vs current window.
- **Agreement drift** — how often the ensemble's prediction agrees with
  its members' majority. When the input distribution moves, members start
  disagreeing with each other (and with the ensemble) in new ways; a drop
  in agreement is an early warning that the ensemble is stale even before
  labelled outcomes arrive.

A ``critical`` report means the ensemble should fall back to the previous
stable version (see ``platform.check_drift``); ``warn`` means watch it.
"""
from __future__ import annotations

import math

from .config import Settings
from .models import DriftReport


def _quantile_bins(values: list[float], n_bins: int = 10) -> list[float]:
    ordered = sorted(values)
    if not ordered:
        return []
    edges = []
    for i in range(1, n_bins):
        edges.append(ordered[int(i * len(ordered) / n_bins)])
    return sorted(set(edges))


def _bin_counts(values: list[float], edges: list[float]) -> list[int]:
    counts = [0] * (len(edges) + 1)
    for v in values:
        b = 0
        while b < len(edges) and v >= edges[b]:
            b += 1
        counts[b] += 1
    return counts


def psi(reference: list[float], current: list[float], n_bins: int = 10) -> float:
    """Population Stability Index between two samples of one feature."""
    if not reference or not current:
        return 0.0
    edges = _quantile_bins(reference, n_bins)
    ref_counts = _bin_counts(reference, edges)
    cur_counts = _bin_counts(current, edges)
    total = 0.0
    for rc, cc in zip(ref_counts, cur_counts):
        ref_p = max(rc / len(reference), 1e-6)
        cur_p = max(cc / len(current), 1e-6)
        total += (cur_p - ref_p) * math.log(cur_p / ref_p)
    return total


def ks_statistic(a: list[float], b: list[float]) -> float:
    """Two-sample Kolmogorov–Smirnov statistic."""
    if not a or not b:
        return 0.0
    combined = sorted(set(a) | set(b))
    max_diff = 0.0
    for v in combined:
        cdf_a = sum(1 for x in a if x <= v) / len(a)
        cdf_b = sum(1 for x in b if x <= v) / len(b)
        max_diff = max(max_diff, abs(cdf_a - cdf_b))
    return max_diff


def member_agreement(member_probs: list[list[float]], ensemble_probs: list[float]) -> float:
    """Share of rows where the ensemble label matches the members' majority."""
    if not ensemble_probs:
        return 1.0
    agree = 0
    for i, ep in enumerate(ensemble_probs):
        ens_label = 1 if ep >= 0.5 else 0
        votes = [1 if probs[i] >= 0.5 else 0 for probs in member_probs]
        majority = 1 if sum(votes) * 2 >= len(votes) else 0
        if ens_label == majority:
            agree += 1
    return agree / len(ensemble_probs)


def drift_report(
    reference_X: list[list[float]],
    current_X: list[list[float]],
    feature_names: list[str],
    reference_agreement: float,
    current_agreement: float,
    settings: Settings | None = None,
) -> DriftReport:
    s = settings or Settings()
    psi_per: dict[str, float] = {}
    ks_per: dict[str, float] = {}
    for j, name in enumerate(feature_names):
        ref_col = [row[j] for row in reference_X]
        cur_col = [row[j] for row in current_X]
        psi_per[name] = psi(ref_col, cur_col)
        ks_per[name] = ks_statistic(ref_col, cur_col)
    max_psi = max(psi_per.values()) if psi_per else 0.0
    max_ks = max(ks_per.values()) if ks_per else 0.0
    agreement_drift = abs(reference_agreement - current_agreement)
    if (
        max_psi >= s.psi_critical
        or max_ks >= 2 * s.ks_warn
        or agreement_drift >= s.agreement_drift_warn
    ):
        level = "critical"
    elif max_psi >= s.psi_warn or max_ks >= s.ks_warn:
        level = "warn"
    else:
        level = "ok"
    return DriftReport(
        psi_per_feature=psi_per,
        max_psi=max_psi,
        ks_per_feature=ks_per,
        max_ks=max_ks,
        reference_agreement=reference_agreement,
        current_agreement=current_agreement,
        agreement_drift=agreement_drift,
        flagged=level != "ok",
        level=level,
    )
