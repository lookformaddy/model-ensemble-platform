"""Seeded synthetic tabular data.

The generator plants a *mixture* of regimes so no single learner wins
everywhere — which is exactly the situation where an ensemble earns its
keep:

- Segment 0 (linear regime): the label follows a linear logit in
  features 0–2. Logistic regression is near-optimal here.
- Segment 1 (nonlinear regime): the label follows an interaction
  (``x3 * x4 > 0``) plus a quadratic rule on ``x5``, with linear main
  effects on ``x3``/``x4`` so a greedy tree can find the structure. A
  decision tree captures this; a linear model cannot.

The regime is *identifiable*: the last feature is the regime indicator
(segment 0 when it is positive, segment 1 when it is negative), so a
learner — or a stacking meta-learner — can learn to trust different
members in different regions of the input space. That is what lets the
stacked ensemble beat its best single member.

Labels follow the planted rule deterministically, with 3% random label
flips (label noise, as in real data), so the Bayes rate is ~0.97 and a
good ensemble lands in the mid-0.80s. All data is synthetic; no
real-world data, no employer data.
"""
from __future__ import annotations

import random

from .models import Dataset, Split


def generate_dataset(
    n: int = 1200,
    n_features: int = 8,
    seed: int = 42,
    label_flip: float = 0.03,
) -> Dataset:
    """Generate a seeded synthetic binary-classification dataset.

    ``label_flip`` is the share of labels randomly flipped after the
    planted rule is applied (label noise; default 3%).
    """
    rng = random.Random(seed)
    feature_names = [f"f{i}" for i in range(n_features)]
    X: list[list[float]] = []
    y: list[int] = []
    segment: list[int] = []
    for _ in range(n):
        xs = [rng.gauss(0.0, 1.0) for _ in range(n_features)]
        # The last feature is the regime indicator: it identifies which
        # rule generated the label, so regime-dependent trust is learnable.
        seg = 0 if xs[-1] > 0 else 1
        if seg == 0:
            logit = 2.5 * xs[0] + 1.8 * xs[1] - 1.5 * xs[2] + 0.5 * xs[6]
        else:
            interaction = 1.0 if xs[3] * xs[4] > 0 else -1.0
            quadratic = 1.0 if abs(xs[5]) > 1.0 else -1.0
            # Main effects on x3/x4 give a greedy tree a foothold: the
            # pure interaction alone has zero marginal signal, which no
            # greedy splitter can see.
            logit = (
                1.5 * xs[3] + 1.5 * xs[4]
                + 2.0 * interaction + 1.5 * quadratic + 0.5 * xs[6]
            )
        label = 1 if logit > 0 else 0
        if rng.random() < label_flip:
            label = 1 - label
        X.append(xs)
        y.append(label)
        segment.append(seg)
    return Dataset(X=X, y=y, feature_names=feature_names, segment=segment)


def train_test_split(data: Dataset, test_fraction: float = 0.2, seed: int = 42) -> Split:
    """Deterministic shuffled split that keeps segment ids aligned."""
    rng = random.Random(seed)
    idx = list(range(data.n))
    rng.shuffle(idx)
    n_test = max(1, int(round(data.n * test_fraction)))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return Split(
        X_train=[data.X[i] for i in train_idx],
        y_train=[data.y[i] for i in train_idx],
        X_test=[data.X[i] for i in test_idx],
        y_test=[data.y[i] for i in test_idx],
        segment_test=[data.segment[i] for i in test_idx],
    )


def kfold_indices(n: int, n_folds: int, seed: int) -> list[list[int]]:
    """Shuffled, roughly equal folds (indices only)."""
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for pos, i in enumerate(idx):
        folds[pos % n_folds].append(i)
    return folds
