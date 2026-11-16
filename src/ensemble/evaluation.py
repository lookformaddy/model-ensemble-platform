"""Evaluation: train members, build ensembles, score everything.

One function produces the complete ``EvaluationReport`` the registry gates
on: per-member metrics, ensemble metrics, diversity diagnostics, per-slice
metrics, and calibration — plus latency/size estimates for the budget gates.
"""
from __future__ import annotations

from .config import Settings
from .diversity import diversity_report
from .ensembles import Ensemble, StackingEnsemble, VotingEnsemble, WeightedBlending
from .learners import BaseLearner
from .metrics import calibration_table, compute_metrics, per_slice_metrics
from .models import EvaluationReport, MemberSpec


def member_specs(members: list[BaseLearner]) -> list[MemberSpec]:
    """Default latency estimates per learner kind (ms, steady state)."""
    latency = {"logistic": 0.4, "tree": 0.6, "knn": 6.0, "naive_bayes": 0.5, "majority": 0.1}
    return [
        MemberSpec(name=m.name, kind=m.kind, latency_ms=latency.get(m.kind, 1.0))
        for m in members
    ]


def evaluate(
    ensemble: Ensemble,
    key: str,
    X_test: list[list[float]],
    y_test: list[int],
    segment_test: list[int],
) -> EvaluationReport:
    """Score a fitted ensemble (and each of its members) on held-out data."""
    member_probs = {
        m.name: m.predict_proba(X_test) for m in ensemble.members
    }
    member_metrics = {
        name: compute_metrics(y_test, probs) for name, probs in member_probs.items()
    }
    ens_probs = ensemble.predict_proba(X_test)
    ens_metrics = compute_metrics(y_test, ens_probs)
    diversity = diversity_report(member_probs, ens_probs, y_test)
    slices = per_slice_metrics(y_test, ens_probs, segment_test)
    cal_bins, ece = calibration_table(y_test, ens_probs)
    # Latency: members run in parallel, so the ensemble latency is the
    # slowest member plus a small combine overhead; size is additive.
    est_latency = max((s.latency_ms for s in member_specs(ensemble.members)), default=0.0) + 0.5
    est_size = ensemble.estimated_size_kb()
    return EvaluationReport(
        ensemble_key=key,
        member_metrics=member_metrics,
        ensemble_metrics=ens_metrics,
        diversity=diversity,
        per_slice=slices,
        calibration=cal_bins,
        expected_calibration_error=ece,
        estimated_latency_ms=est_latency,
        estimated_size_kb=est_size,
    )


def train_evaluate_and_build(
    members: list[BaseLearner],
    X_train: list[list[float]],
    y_train: list[int],
    X_test: list[list[float]],
    y_test: list[int],
    segment_test: list[int],
    settings: Settings,
) -> tuple[dict[str, EvaluationReport], dict[str, Ensemble]]:
    """Train the four base learners, build voting / blending / stacking,
    and evaluate all three ensembles on the same held-out set.

    Returns ``(reports, ensembles)`` — the fitted ensembles are returned
    alongside the reports so callers can serve them without refitting.

    Blending weights come from a validation slice carved out of the
    training data (the last 20%), so no test data leaks into the weights.
    """
    n_val = max(20, int(0.2 * len(X_train)))
    X_fit, y_fit = X_train[:-n_val], y_train[:-n_val]
    X_val, y_val = X_train[-n_val:], y_train[-n_val:]

    fitted = [m.clone().fit(X_fit, y_fit) for m in members]

    voting = VotingEnsemble([m.clone() for m in fitted], mode="soft")
    voting.fit(X_fit, y_fit)
    blending = WeightedBlending([m.clone() for m in fitted])
    blending.fit(X_fit, y_fit)
    blending.fit_weights(X_val, y_val)
    stacking = StackingEnsemble([m.clone() for m in fitted], n_folds=settings.n_folds, seed=settings.default_seed)
    stacking.fit(X_fit, y_fit)

    ensembles: dict[str, Ensemble] = {
        "voting-ensemble@1": voting,
        "blending-ensemble@1": blending,
        "stacking-ensemble@1": stacking,
    }
    reports: dict[str, EvaluationReport] = {}
    for key, ens in ensembles.items():
        reports[key] = evaluate(ens, key, X_test, y_test, segment_test)
    return reports, ensembles


def train_and_evaluate_all(
    members: list[BaseLearner],
    X_train: list[list[float]],
    y_train: list[int],
    X_test: list[list[float]],
    y_test: list[int],
    segment_test: list[int],
    settings: Settings,
) -> dict[str, EvaluationReport]:
    """Reports only; see :func:`train_evaluate_and_build`."""
    reports, _ = train_evaluate_and_build(
        members, X_train, y_train, X_test, y_test, segment_test, settings
    )
    return reports


def best_ensemble(reports: dict[str, EvaluationReport]) -> str:
    """The ensemble with the highest held-out ROC-AUC."""
    return max(reports, key=lambda k: reports[k].ensemble_metrics.roc_auc)
