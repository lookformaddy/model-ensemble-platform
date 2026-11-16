"""Evaluation metrics, implemented from scratch (stdlib only).

Covers the metrics the promotion gates and the demo rely on: accuracy,
macro F1, ROC-AUC (Mann-Whitney U), log-loss, Brier score, per-slice
metrics, and calibration (reliability bins + expected calibration error).
"""
from __future__ import annotations

import math

from .models import CalibrationBin, Metrics, SliceMetrics


def _clip(p: float, eps: float = 1e-15) -> float:
    return min(1.0 - eps, max(eps, p))


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return 0.0
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def _f1_for_class(y_true: list[int], y_pred: list[int], cls: int) -> float:
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == cls and b == cls)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a != cls and b == cls)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == cls and b != cls)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    return (_f1_for_class(y_true, y_pred, 0) + _f1_for_class(y_true, y_pred, 1)) / 2.0


def roc_auc(y_true: list[int], y_score: list[float]) -> float:
    """Mann-Whitney U formulation; 0.5 when a class is absent."""
    pos = [s for t, s in zip(y_true, y_score) if t == 1]
    neg = [s for t, s in zip(y_true, y_score) if t == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for q in neg:
            if p > q:
                wins += 1.0
            elif p == q:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def log_loss(y_true: list[int], y_prob: list[float]) -> float:
    if not y_true:
        return 0.0
    total = 0.0
    for t, p in zip(y_true, y_prob):
        p = _clip(p)
        total += -(math.log(p) if t == 1 else math.log(1 - p))
    return total / len(y_true)


def brier_score(y_true: list[int], y_prob: list[float]) -> float:
    if not y_true:
        return 0.0
    return sum((p - t) ** 2 for t, p in zip(y_true, y_prob)) / len(y_true)


def compute_metrics(y_true: list[int], y_prob: list[float]) -> Metrics:
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
    return Metrics(
        accuracy=accuracy(y_true, y_pred),
        macro_f1=macro_f1(y_true, y_pred),
        roc_auc=roc_auc(y_true, y_prob),
        log_loss=log_loss(y_true, y_prob),
        brier=brier_score(y_true, y_prob),
        n=len(y_true),
    )


def per_slice_metrics(
    y_true: list[int],
    y_prob: list[float],
    segment: list[int],
) -> list[SliceMetrics]:
    """Accuracy / AUC per regime segment (the slice the members disagree on)."""
    out: list[SliceMetrics] = []
    for seg in sorted(set(segment)):
        idx = [i for i, s in enumerate(segment) if s == seg]
        yt = [y_true[i] for i in idx]
        yp = [y_prob[i] for i in idx]
        out.append(
            SliceMetrics(
                slice_name=f"segment-{seg}",
                n=len(idx),
                accuracy=accuracy(yt, [1 if p >= 0.5 else 0 for p in yp]),
                roc_auc=roc_auc(yt, yp),
            )
        )
    return out


def calibration_table(
    y_true: list[int], y_prob: list[float], n_bins: int = 10
) -> tuple[list[CalibrationBin], float]:
    """Reliability bins + expected calibration error (ECE)."""
    bins: list[CalibrationBin] = []
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        idx = [i for i, p in enumerate(y_prob) if lo <= p < hi or (b == n_bins - 1 and p == 1.0)]
        if not idx:
            bins.append(
                CalibrationBin(bin_lo=lo, bin_hi=hi, n=0, mean_predicted=0.0, observed_rate=0.0)
            )
            continue
        mean_pred = sum(y_prob[i] for i in idx) / len(idx)
        observed = sum(y_true[i] for i in idx) / len(idx)
        bins.append(
            CalibrationBin(
                bin_lo=lo, bin_hi=hi, n=len(idx),
                mean_predicted=mean_pred, observed_rate=observed,
            )
        )
        ece += (len(idx) / n) * abs(mean_pred - observed)
    return bins, ece
