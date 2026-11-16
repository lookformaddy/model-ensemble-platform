"""From-scratch base learners (stdlib only — no sklearn / numpy / torch).

Every learner shares one interface so ensembles can treat them uniformly:

- ``fit(X, y)`` — train on rows of floats and 0/1 labels.
- ``predict(X)`` — class labels.
- ``predict_proba(X)`` — probability of class 1, in [0, 1].
- ``clone()`` — a fresh, untrained instance with the same hyper-parameters
  (needed for out-of-fold stacking and multi-seed stability checks).

Learners implemented: logistic regression (gradient descent with feature
standardisation), CART decision tree (Gini, best split over sorted
thresholds), k-nearest neighbours, Gaussian naive Bayes, and a majority
baseline used as the floor every ensemble must beat.
"""
from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections import Counter


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _clip_prob(p: float, eps: float = 1e-9) -> float:
    return min(1.0 - eps, max(eps, p))


class BaseLearner(ABC):
    kind: str = "base"

    @abstractmethod
    def fit(self, X: list[list[float]], y: list[int]) -> "BaseLearner":
        ...

    @abstractmethod
    def predict_proba(self, X: list[list[float]]) -> list[float]:
        ...

    def predict(self, X: list[list[float]]) -> list[int]:
        return [1 if p >= 0.5 else 0 for p in self.predict_proba(X)]

    @abstractmethod
    def clone(self) -> "BaseLearner":
        ...

    @property
    def name(self) -> str:
        return self.kind


class LogisticRegressionLearner(BaseLearner):
    kind = "logistic"

    def __init__(self, lr: float = 0.2, epochs: int = 500, l2: float = 0.001):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self._mean: list[float] = []
        self._std: list[float] = []
        self._w: list[float] = []
        self._b: float = 0.0

    def fit(self, X, y):
        n = len(X)
        d = len(X[0])
        self._mean = [sum(row[j] for row in X) / n for j in range(d)]
        self._std = []
        for j in range(d):
            var = sum((row[j] - self._mean[j]) ** 2 for row in X) / n
            self._std.append(math.sqrt(var) if var > 1e-12 else 1.0)
        Xs = [[(row[j] - self._mean[j]) / self._std[j] for j in range(d)] for row in X]
        self._w = [0.0] * d
        self._b = 0.0
        for _ in range(self.epochs):
            grad_w = [0.0] * d
            grad_b = 0.0
            for xi, yi in zip(Xs, y):
                p = _sigmoid(sum(w * v for w, v in zip(self._w, xi)) + self._b)
                err = p - yi
                for j in range(d):
                    grad_w[j] += err * xi[j]
                grad_b += err
            for j in range(d):
                self._w[j] -= self.lr * (grad_w[j] / n + self.l2 * self._w[j])
            self._b -= self.lr * grad_b / n
        return self

    def predict_proba(self, X):
        out = []
        for row in X:
            xs = [(row[j] - self._mean[j]) / self._std[j] for j in range(len(row))]
            z = sum(w * v for w, v in zip(self._w, xs)) + self._b
            out.append(_clip_prob(_sigmoid(z)))
        return out

    def clone(self):
        return LogisticRegressionLearner(lr=self.lr, epochs=self.epochs, l2=self.l2)


class _TreeNode:
    __slots__ = ("feature", "threshold", "left", "right", "prob", "label")

    def __init__(self, prob: float, label: int):
        self.feature: int | None = None
        self.threshold: float | None = None
        self.left: _TreeNode | None = None
        self.right: _TreeNode | None = None
        self.prob = prob
        self.label = label


def _gini(counts: Counter, total: int) -> float:
    if total == 0:
        return 0.0
    p1 = counts.get(1, 0) / total
    return 1.0 - p1 * p1 - (1 - p1) * (1 - p1)


class DecisionTreeLearner(BaseLearner):
    kind = "tree"

    def __init__(self, max_depth: int = 6, min_samples_split: int = 10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self._root: _TreeNode | None = None
        self._n_features = 0

    def fit(self, X, y):
        self._n_features = len(X[0])
        self._root = self._build(X, y, depth=0)
        return self

    def _build(self, X, y, depth):
        n = len(y)
        n_pos = sum(y)
        prob = n_pos / n if n else 0.0
        node = _TreeNode(prob=prob, label=1 if prob >= 0.5 else 0)
        if depth >= self.max_depth or n < self.min_samples_split or n_pos in (0, n):
            return node
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        parent_gini = _gini(Counter(y), n)
        for j in range(self._n_features):
            order = sorted(range(n), key=lambda i: X[i][j])
            left_counts: Counter = Counter()
            right_counts: Counter = Counter(y)
            for split_pos in range(n - 1):
                i = order[split_pos]
                left_counts[y[i]] += 1
                right_counts[y[i]] -= 1
                if X[order[split_pos]][j] == X[order[split_pos + 1]][j]:
                    continue
                n_left = split_pos + 1
                n_right = n - n_left
                gain = parent_gini - (
                    n_left / n * _gini(left_counts, n_left)
                    + n_right / n * _gini(right_counts, n_right)
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feature = j
                    best_threshold = (X[order[split_pos]][j] + X[order[split_pos + 1]][j]) / 2.0
        if best_feature is None:
            return node
        left_idx = [i for i in range(n) if X[i][best_feature] < best_threshold]
        right_idx = [i for i in range(n) if X[i][best_feature] >= best_threshold]
        node.feature = best_feature
        node.threshold = best_threshold
        node.left = self._build([X[i] for i in left_idx], [y[i] for i in left_idx], depth + 1)
        node.right = self._build([X[i] for i in right_idx], [y[i] for i in right_idx], depth + 1)
        return node

    def _walk(self, row):
        node = self._root
        assert node is not None
        while node.feature is not None:
            node = node.left if row[node.feature] < node.threshold else node.right
            assert node is not None
        return node

    def predict_proba(self, X):
        return [_clip_prob(self._walk(row).prob) for row in X]

    def clone(self):
        return DecisionTreeLearner(
            max_depth=self.max_depth, min_samples_split=self.min_samples_split
        )


class KNearestNeighborsLearner(BaseLearner):
    kind = "knn"

    def __init__(self, k: int = 5):
        self.k = k
        self._X: list[list[float]] = []
        self._y: list[int] = []

    def fit(self, X, y):
        self._X = [list(row) for row in X]
        self._y = list(y)
        return self

    def predict_proba(self, X):
        out = []
        for row in X:
            dists = [(math.dist(row, xi), yi) for xi, yi in zip(self._X, self._y)]
            dists.sort(key=lambda t: t[0])
            k = min(self.k, len(dists))
            votes = sum(yi for _, yi in dists[:k])
            out.append(_clip_prob(votes / k if k else 0.5))
        return out

    def clone(self):
        return KNearestNeighborsLearner(k=self.k)


class GaussianNaiveBayesLearner(BaseLearner):
    kind = "naive_bayes"

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        self._params: dict[int, tuple[list[float], list[float]]] = {}
        self._priors: dict[int, float] = {}

    def fit(self, X, y):
        d = len(X[0])
        for c in (0, 1):
            rows = [row for row, yi in zip(X, y) if yi == c]
            self._priors[c] = len(rows) / len(X) if X else 0.5
            if not rows:
                self._params[c] = ([0.0] * d, [1.0] * d)
                continue
            means = [sum(r[j] for r in rows) / len(rows) for j in range(d)]
            variances = []
            for j in range(d):
                var = sum((r[j] - means[j]) ** 2 for r in rows) / len(rows)
                variances.append(var + self.var_smoothing)
            self._params[c] = (means, variances)
        return self

    def predict_proba(self, X):
        out = []
        for row in X:
            log_probs = {}
            for c in (0, 1):
                means, variances = self._params[c]
                lp = math.log(self._priors.get(c, 0.5) or 1e-12)
                for j, v in enumerate(row):
                    var = variances[j]
                    lp += -0.5 * math.log(2 * math.pi * var) - (v - means[j]) ** 2 / (2 * var)
                log_probs[c] = lp
            m = max(log_probs.values())
            p1 = math.exp(log_probs[1] - m)
            p0 = math.exp(log_probs[0] - m)
            out.append(_clip_prob(p1 / (p0 + p1)))
        return out

    def clone(self):
        return GaussianNaiveBayesLearner(var_smoothing=self.var_smoothing)


class MajorityLearner(BaseLearner):
    """Baseline floor: always predicts the majority class."""

    kind = "majority"

    def __init__(self):
        self._p = 0.5

    def fit(self, X, y):
        self._p = sum(y) / len(y) if y else 0.5
        return self

    def predict_proba(self, X):
        return [_clip_prob(self._p) for _ in X]

    def clone(self):
        return MajorityLearner()


def default_learners(seed: int = 42) -> list[BaseLearner]:
    """The standard four-member base set for this platform."""
    _ = seed  # learners are deterministic; seed is kept for API symmetry
    return [
        LogisticRegressionLearner(),
        DecisionTreeLearner(),
        KNearestNeighborsLearner(),
        GaussianNaiveBayesLearner(),
    ]


def make_learner(kind: str, params: dict | None = None) -> BaseLearner:
    params = params or {}
    if kind == "logistic":
        return LogisticRegressionLearner(**params)
    if kind == "tree":
        return DecisionTreeLearner(**params)
    if kind == "knn":
        return KNearestNeighborsLearner(**params)
    if kind == "naive_bayes":
        return GaussianNaiveBayesLearner(**params)
    if kind == "majority":
        return MajorityLearner()
    raise ValueError(f"unknown learner kind: {kind}")


def seeded_learner(kind: str, seed: int, params: dict | None = None) -> BaseLearner:
    """A learner whose hyper-parameters are jittered by seed (used for the
    multi-seed stability check — same family, slightly different fit)."""
    rng = random.Random(seed)
    params = dict(params or {})
    if kind == "tree":
        params.setdefault("max_depth", rng.choice([5, 6, 7]))
    if kind == "knn":
        params.setdefault("k", rng.choice([3, 5, 7]))
    if kind == "logistic":
        params.setdefault("lr", rng.choice([0.05, 0.1, 0.15]))
    return make_learner(kind, params)
