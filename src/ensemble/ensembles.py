"""Ensemble builders: voting, weighted blending, stacking.

All three share the base-learner interface (``fit`` / ``predict`` /
``predict_proba``) so they can be evaluated, gated, registered, and served
through exactly the same code paths.

- **Voting** — hard (majority label) or soft (mean probability).
- **Weighted blending** — probability average with weights fitted from
  validation performance (AUC), so stronger members count more.
- **Stacking** — a from-scratch logistic-regression meta-learner trained on
  out-of-fold base predictions *plus regime interactions* (each member's
  probability interacted with the regime indicator), so it can learn
  regime-dependent trust ("trust logistic in regime 0, the tree in
  regime 1") instead of one fixed weighting. A plain linear meta-learner
  over probabilities alone cannot express that.
"""
from __future__ import annotations

from .data import kfold_indices
from .learners import BaseLearner, LogisticRegressionLearner
from .metrics import roc_auc


class Ensemble:
    """Common base: a fitted collection of members with a combine rule."""

    kind: str = "ensemble"

    def __init__(self, members: list[BaseLearner]):
        self.members = list(members)

    def fit(self, X: list[list[float]], y: list[int]) -> "Ensemble":
        for m in self.members:
            m.fit(X, y)
        return self

    def member_probs(self, X: list[list[float]]) -> list[list[float]]:
        return [m.predict_proba(X) for m in self.members]

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        raise NotImplementedError

    def predict(self, X: list[list[float]]) -> list[int]:
        return [1 if p >= 0.5 else 0 for p in self.predict_proba(X)]

    @property
    def member_names(self) -> list[str]:
        return [m.name for m in self.members]

    def estimated_size_kb(self) -> float:
        """Rough serialized-size estimate (KB) for the size-budget gate.

        k-NN stores the whole training set; trees store nodes; linear and
        naive-Bayes models are tiny. The estimate is deliberately
        conservative so the gate means something.
        """
        total = 0.0
        for m in self.members:
            if m.kind == "knn":
                total += len(getattr(m, "_X", [])) * 8 * 8 / 1024.0
            elif m.kind == "tree":
                total += 64.0
            elif m.kind == "logistic":
                total += 4.0
            else:
                total += 8.0
        return total


class VotingEnsemble(Ensemble):
    kind = "voting"

    def __init__(self, members: list[BaseLearner], mode: str = "soft"):
        if mode not in ("hard", "soft"):
            raise ValueError(f"voting mode must be 'hard' or 'soft', got {mode!r}")
        super().__init__(members)
        self.mode = mode

    def predict_proba(self, X):
        probs = self.member_probs(X)
        n = len(X)
        if self.mode == "soft":
            return [
                sum(probs[m][i] for m in range(len(self.members))) / len(self.members)
                for i in range(n)
            ]
        # Hard voting: fraction of members voting class 1, smoothed so a
        # unanimous vote is not an absolute 0/1 (keeps log-loss finite).
        out = []
        for i in range(n):
            votes = sum(1 for m in range(len(self.members)) if probs[m][i] >= 0.5)
            out.append(min(0.99, max(0.01, votes / len(self.members))))
        return out


class WeightedBlending(Ensemble):
    kind = "blending"

    def __init__(self, members: list[BaseLearner], weights: list[float] | None = None):
        super().__init__(members)
        self.weights: list[float] = list(weights) if weights else [1.0 / len(members)] * len(members)

    def fit_weights(self, X_val: list[list[float]], y_val: list[int]) -> "WeightedBlending":
        """Set weights proportional to each member's validation AUC above
        chance (0.5), so a chance-level member gets ~zero weight."""
        scores = []
        for m in self.members:
            auc = roc_auc(y_val, m.predict_proba(X_val))
            scores.append(max(0.0, auc - 0.5))
        total = sum(scores)
        if total <= 0:
            self.weights = [1.0 / len(self.members)] * len(self.members)
        else:
            self.weights = [s / total for s in scores]
        return self

    def predict_proba(self, X):
        probs = self.member_probs(X)
        n = len(X)
        return [
            sum(self.weights[m] * probs[m][i] for m in range(len(self.members)))
            for i in range(n)
        ]


class StackingEnsemble(Ensemble):
    kind = "stacking"

    def __init__(
        self,
        members: list[BaseLearner],
        meta_learner: BaseLearner | None = None,
        n_folds: int = 5,
        seed: int = 42,
        regime_feature: int = -1,
    ):
        super().__init__(members)
        # Default meta-learner is logistic regression over regime-aware
        # meta-features (see _meta_features): a linear meta-learner over
        # plain probabilities can only learn one fixed weighting, while
        # the regime interactions let it trust different members in
        # different regions — which is the whole point of stacking here.
        self.meta_learner = meta_learner or LogisticRegressionLearner(
            lr=0.2, epochs=500, l2=0.001
        )
        self.n_folds = n_folds
        self.seed = seed
        # Index of the feature that identifies the regime (default: the
        # last feature). In production this would be a known segment
        # feature (device type, customer tier, ...).
        self.regime_feature = regime_feature
        self.oof_probs: list[list[float]] | None = None
        self._oof_X: list[list[float]] | None = None

    def _meta_features(
        self, X: list[list[float]], probs: list[list[float]]
    ) -> list[list[float]]:
        """Member probabilities + regime interactions.

        For each row: the members' probabilities, each probability
        interacted with the regime indicator (1 when the regime feature
        is positive, 0 otherwise) and with its complement, plus the
        indicator itself. The meta-learner can therefore learn, e.g.,
        "trust logistic in regime 0 and the tree in regime 1".
        """
        out: list[list[float]] = []
        for i, row in enumerate(X):
            r = 1.0 if row[self.regime_feature] > 0 else 0.0
            feats = [probs[mi][i] for mi in range(len(self.members))]
            feats += [probs[mi][i] * r for mi in range(len(self.members))]
            feats += [probs[mi][i] * (1.0 - r) for mi in range(len(self.members))]
            feats.append(r)
            out.append(feats)
        return out

    def fit(self, X, y):
        """Fit with leakage-safe out-of-fold predictions for the stacker."""
        n = len(X)
        folds = kfold_indices(n, self.n_folds, self.seed)
        oof: list[list[float]] = [[0.0] * n for _ in self.members]
        for fold in folds:
            fold_set = set(fold)
            train_idx = [i for i in range(n) if i not in fold_set]
            X_tr = [X[i] for i in train_idx]
            y_tr = [y[i] for i in train_idx]
            X_te = [X[i] for i in fold]
            for mi, member in enumerate(self.members):
                clone = member.clone().fit(X_tr, y_tr)
                probs = clone.predict_proba(X_te)
                for pos, i in enumerate(fold):
                    oof[mi][i] = probs[pos]
        self.oof_probs = oof
        self._oof_X = [list(row) for row in X]
        # Meta-features: member OOF probabilities + regime interactions.
        meta_X = self._meta_features(X, oof)
        self.meta_learner.fit(meta_X, y)
        # Refit base learners on the full training set for serving.
        for m in self.members:
            m.fit(X, y)
        return self

    def predict_proba(self, X):
        probs = self.member_probs(X)
        meta_X = self._meta_features(X, probs)
        return self.meta_learner.predict_proba(meta_X)

    def oof_auc(self, y: list[int]) -> float:
        """AUC of the stacked model evaluated on its own OOF predictions —
        an honest in-sample estimate, since the meta-learner never saw
        these base predictions during training."""
        if self.oof_probs is None or self._oof_X is None:
            raise RuntimeError("stacking ensemble has not been fitted")
        meta_X = self._meta_features(self._oof_X, self.oof_probs)
        return roc_auc(y, self.meta_learner.predict_proba(meta_X))


def build_voting(members: list[BaseLearner], mode: str = "soft") -> VotingEnsemble:
    return VotingEnsemble(members, mode=mode)


def build_blending(
    members: list[BaseLearner],
    X_train: list[list[float]],
    y_train: list[int],
    X_val: list[list[float]],
    y_val: list[int],
) -> WeightedBlending:
    blending = WeightedBlending(members)
    blending.fit(X_train, y_train)
    blending.fit_weights(X_val, y_val)
    return blending


def build_stacking(
    members: list[BaseLearner], n_folds: int = 5, seed: int = 42
) -> StackingEnsemble:
    return StackingEnsemble(members, n_folds=n_folds, seed=seed)
